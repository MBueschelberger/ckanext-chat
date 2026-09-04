"""
title: CKAN-Chat
author: Matthias Büschelberger, Thomas Hanke
version: 3.0.0
required_open_webui_version: 0.6.15
"""

from pydantic import BaseModel, Field
import aiohttp
import json
import re
from datetime import datetime, timezone
from urllib.parse import urljoin
import os

FORWARD_PARAMS = {
    "messages",
    "model",
    "temperature",
    "top_p",
    "n",
    "stream",
    "stop",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "response_format",
    "tools",
    "tool_choice",
}


class Pipe:
    file_handler = True

    class UserValves(BaseModel):
        CKAN_INSTANCE: str = Field(
            default="",
            description="Base-URL des Completions-Endpoints",
        )
        CKAN_TOKEN: str = Field(
            default="",
            description="CKAN API-Token (leer lassen für automatische SSO-Authentifizierung)",
        )
        TIMEOUT: int = Field(default=3600, description="Request timeout (seconds)")

    class Valves(BaseModel):
        SSL: bool = Field(default=False, description="Use SSL")

    def __init__(self):
        self.name = "IWM Multi-Agent RAG"
        self.user_valves = self.UserValves()
        self.valves = self.Valves()

    async def on_startup(self):
        print(f"on_startup:{__name__}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")

    async def _status(self, emitter, text: str, done: bool = False):
        if emitter:
            await emitter(
                {"type": "status", "data": {"description": text, "done": done}}
            )

    def _build_history(self, messages):
        """Convert OpenAI-format messages to pydantic-ai history format for the CKAN backend."""
        history = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if p.get("type") == "text"
                )
            if not content:
                continue
            if role == "user":
                history.append(
                    {
                        "kind": "request",
                        "parts": [{"part_kind": "user-prompt", "content": content}],
                    }
                )
            elif role == "assistant":
                history.append(
                    {
                        "kind": "response",
                        "parts": [{"part_kind": "text", "content": content}],
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
        return history

    async def _get_file_bytes(self, file_info: dict) -> tuple[bytes, str, str] | None:
        """Read file bytes from Open WebUI storage. Returns (data, filename, content_type) or None."""
        file_id = file_info.get("id")
        if not file_id:
            return None
        try:
            from open_webui.models.files import Files
            from open_webui.storage.provider import Storage

            file_record = await Files.get_file_by_id(file_id)
            if not file_record or not file_record.path:
                return None
            file_path = Storage.get_file(file_record.path)
            with open(file_path, "rb") as f:
                data = f.read()
            filename = file_info.get("name") or file_record.filename or "upload"
            content_type = (
                file_info.get("content_type")
                or (file_record.meta or {}).get("content_type")
                or "application/octet-stream"
            )
            return data, filename, content_type
        except Exception as e:
            print(f"[IWM Pipe] Failed to read file {file_id}: {e}")
            return None

    async def _pipe_with_upload(
        self, body, valves, files, __event_emitter__, research=False, token=""
    ):
        """Stream via /chat/ask/stream with multipart file upload."""
        messages = body.get("messages", [])
        text = ""
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = " ".join(
                        p.get("text", "") for p in content if p.get("type") == "text"
                    )
                last_user_idx = i
                break

        if not text:
            yield "Fehler: Keine Nachricht gefunden."
            return

        file_result = await self._get_file_bytes(files[0])

        url = urljoin(valves.CKAN_INSTANCE, "/chat/ask/stream")
        headers = {"Authorization": f"Bearer {token}"}

        form = aiohttp.FormData()
        form.add_field("text", text)
        if research:
            form.add_field("research", "true")
        if last_user_idx > 0:
            history = self._build_history(messages[:last_user_idx])
            if history:
                form.add_field("history", json.dumps(history))
        if file_result:
            data, filename, content_type = file_result
            form.add_field("upload", data, filename=filename, content_type=content_type)
            # await self._status(
            #    __event_emitter__, f"Datei '{filename}' wird hochgeladen…"
            # )
        else:
            await self._status(
                __event_emitter__,
                "Datei konnte nicht gelesen werden, sende nur den Prompt…",
            )

        timeout = aiohttp.ClientTimeout(total=valves.TIMEOUT)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url, data=form, headers=headers, ssl=self.valves.SSL
                ) as r:
                    r.raise_for_status()

                    event_type = None
                    async for raw_line in r.content:
                        line = raw_line.decode("utf-8").rstrip("\r\n")

                        if not line:
                            event_type = None
                            continue
                        if line.startswith("event: "):
                            event_type = line[7:].strip()
                            continue
                        if not line.startswith("data: "):
                            continue

                        payload = line[6:]
                        if event_type == "status":
                            try:
                                msg = json.loads(payload).get("message", "")
                                if msg:
                                    await self._status(__event_emitter__, msg)
                            except json.JSONDecodeError:
                                pass

                        elif event_type == "done":
                            try:
                                response_msgs = json.loads(payload).get("response", [])
                                for msg in response_msgs:
                                    if msg.get("kind") == "response":
                                        for part in msg.get("parts", []):
                                            if part.get(
                                                "part_kind"
                                            ) == "text" and part.get("content"):
                                                text_buf = part["content"]
                                                while True:
                                                    rm = re.search(
                                                        r"\[ref\]([^|]*)\|([^\[]*)\[/ref\]\n?",
                                                        text_buf,
                                                    )
                                                    if not rm:
                                                        break
                                                    before = text_buf[: rm.start()]
                                                    if before:
                                                        yield before
                                                    ref_title = rm.group(1).strip()
                                                    ref_url = rm.group(2).strip()
                                                    yield {
                                                        "event": {
                                                            "type": "citation",
                                                            "data": {
                                                                "document": [],
                                                                "metadata": [
                                                                    {"source": ref_title}
                                                                ],
                                                                "source": {
                                                                    "name": ref_title,
                                                                    "url": ref_url,
                                                                },
                                                            },
                                                        }
                                                    }
                                                    text_buf = text_buf[rm.end() :]
                                                if text_buf:
                                                    yield text_buf
                            except json.JSONDecodeError:
                                pass

            await self._status(__event_emitter__, "Fertig", done=True)

        except aiohttp.ClientError as e:
            await self._status(__event_emitter__, f"Fehler: {e}", done=True)
            yield f"Fehler: {e}"
        except Exception as e:
            await self._status(__event_emitter__, f"Fehler: {e}", done=True)
            yield f"Fehler: {e}"

    async def _pipe_streaming(
        self, body, valves, __event_emitter__, research=False, token=""
    ):
        """Stream via /chat/v1/chat/completions (OpenAI-compatible, no file upload)."""
        url = urljoin(valves.CKAN_INSTANCE, "/chat/v1/chat/completions")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        if "user" in body and not isinstance(body["user"], str):
            body["user"] = body["user"].get("id", str(body["user"]))

        filtered_body = {k: v for k, v in body.items() if k in FORWARD_PARAMS}
        filtered_body["stream"] = True
        if research:
            filtered_body["model"] = "research"

        timeout = aiohttp.ClientTimeout(total=valves.TIMEOUT)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    url, json=filtered_body, headers=headers, ssl=self.valves.SSL
                ) as r:
                    r.raise_for_status()

                    buf = ""
                    async for raw_line in r.content:
                        decoded = raw_line.decode("utf-8").strip()
                        if not decoded.startswith("data: "):
                            continue
                        payload = decoded[6:]
                        if payload.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(payload)
                        except json.JSONDecodeError:
                            continue

                        if "error" in chunk and "choices" not in chunk:
                            err_msg = chunk["error"].get("message", "Unknown error")
                            yield f"\n\n**Fehler:** {err_msg}"
                            break

                        delta = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content")
                        )
                        if not delta:
                            continue

                        buf += delta
                        while True:
                            m = re.search(
                                r"\[status\](.*?)\[/status\]\n?",
                                buf,
                                re.DOTALL,
                            )
                            if not m:
                                break
                            before = buf[: m.start()]
                            if before:
                                yield before
                            await self._status(
                                __event_emitter__, " ".join(m.group(1).split())
                            )
                            buf = buf[m.end() :]

                        while True:
                            m = re.search(
                                r"\[ref\]([^|]*)\|([^\[]*)\[/ref\]\n?", buf
                            )
                            if not m:
                                break
                            before = buf[: m.start()]
                            if before:
                                yield before
                            ref_title = m.group(1).strip()
                            ref_url = m.group(2).strip()
                            yield {
                                "event": {
                                    "type": "citation",
                                    "data": {
                                        "document": [],
                                        "metadata": [{"source": ref_title}],
                                        "source": {
                                            "name": ref_title,
                                            "url": ref_url,
                                        },
                                    },
                                }
                            }
                            buf = buf[m.end() :]

                        if "[status]" not in buf and "[ref]" not in buf:
                            if buf:
                                yield buf
                            buf = ""

                    if buf:
                        yield buf

            await self._status(__event_emitter__, "Fertig", done=True)

        except aiohttp.ClientError as e:
            await self._status(__event_emitter__, f"Fehler: {e}", done=True)
            yield f"Fehler: {e}"
        except Exception as e:
            await self._status(__event_emitter__, f"Fehler: {e}", done=True)
            yield f"Fehler: {e}"

    def _resolve_token(self, valves, oauth_token):
        """Return CKAN auth token: manual valve first, then SSO access_token."""
        if valves.CKAN_TOKEN:
            return valves.CKAN_TOKEN
        if oauth_token and isinstance(oauth_token, dict):
            return oauth_token.get("access_token", "")
        return ""

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__=None,
        __files__: list = None,
        __metadata__: dict = None,
        __oauth_token__: dict = None,
    ):
        user_valves = __user__.get("valves", self.user_valves)
        research = (__metadata__ or {}).get("research", False)
        await self._status(
            __event_emitter__, "Recherchiert…" if research else "Denkt nach…"
        )

        files = __files__ or []
        has_upload = any(f.get("type") not in ("image",) and f.get("id") for f in files)

        token = self._resolve_token(user_valves, __oauth_token__)

        if has_upload:
            async for chunk in self._pipe_with_upload(
                body,
                user_valves,
                files,
                __event_emitter__,
                research=research,
                token=token,
            ):
                yield chunk
        else:
            async for chunk in self._pipe_streaming(
                body, user_valves, __event_emitter__, research=research, token=token
            ):
                yield chunk
