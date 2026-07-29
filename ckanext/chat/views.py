import asyncio
import json
import os
import sys
from distutils.util import strtobool
from typing import Any

import ckan.lib.base as base
import ckan.lib.helpers as core_helpers
import ckan.plugins.toolkit as toolkit
from ckan.common import _, current_user
from flask import Blueprint, current_app, jsonify, request
from flask.views import MethodView
from loguru import logger
from pydantic_ai.messages import ModelMessagesTypeAdapter, TextPart
from pydantic_ai.usage import UsageLimits

from ckanext.chat.bot.agent import (exception_to_model_response,
                                    user_input_to_model_request)
from ckanext.chat.helpers import service_available

#mp.set_start_method("spawn", force=True)
logger.remove()
if bool(strtobool(os.environ.get("DEBUG", "false"))):
    log_level = "DEBUG"
else:
    log_level = "ERROR"
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | [{name}] {message}",
    level=log_level,
    enqueue=True,
)

blueprint = Blueprint("chat", __name__)

global_ckan_app = None


@blueprint.before_request
def capture_global_app():
    # This hook is executed in an active application context.
    global global_ckan_app
    if global_ckan_app is None:
        # Capture the global CKAN app from the current request's context
        global_ckan_app = current_app._get_current_object()


class ChatView(MethodView):
    def post(self):
        return core_helpers.redirect_to(
            "chat.chat",
        )

    def get(self):
        if current_user.is_anonymous:
            core_helpers.flash_error(_("Not authorized to see this page"))

            # flask types do not mention that it's possible to return a response
            # from the `before_request` callback
            return core_helpers.redirect_to("user.login")
        # logger.debug(get_ckan_url_patterns())
        return base.render(
            "chat/chat_ui.html",
            extra_vars={
                "service_status": service_available(),
                "token": toolkit.config.get("ckanext.chat.api_token"),
                "api_endpoint": toolkit.config.get("ckanext.chat.completion_url"),
            },
        )

MAX_HISTORY_MESSAGES = 100
MAX_MESSAGE_CONTENT_LENGTH = 50000
VALID_MESSAGE_KINDS = {"request", "response"}


def _validate_history(history_str: str):
    if not history_str:
        return None
    try:
        history_list = json.loads(history_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid history JSON, ignoring")
        return None

    if not isinstance(history_list, list):
        return None
    if len(history_list) > MAX_HISTORY_MESSAGES:
        history_list = history_list[-MAX_HISTORY_MESSAGES:]

    for msg in history_list:
        if isinstance(msg, dict):
            kind = msg.get("kind", "")
            if kind not in VALID_MESSAGE_KINDS:
                logger.warning(f"Rejected history message with invalid kind: {kind}")
                return None
            for part in msg.get("parts", []):
                if isinstance(part, dict):
                    content = part.get("content", "")
                    if isinstance(content, str) and len(content) > MAX_MESSAGE_CONTENT_LENGTH:
                        part["content"] = content[:MAX_MESSAGE_CONTENT_LENGTH]

    return ModelMessagesTypeAdapter.validate_python(history_list)


def ask():
    user_input = request.form.get("text")
    history = request.form.get("history", "")
    research = request.form.get("research", False)
    tkuser = toolkit.current_user
    debug = bool(strtobool(os.environ.get("DEBUG", "false")))

    if tkuser.name is None:
        return {"success": False, "msg": "Must be logged in to view site"}

    try:
        response = asyncio.run(
            _agent_worker(user_input, history, user_id=tkuser.id, research=research),
            debug=debug,
        )
        messages = response.new_messages()
        [
            [
                message.parts.remove(part)
                for part in message.parts
                if isinstance(part, TextPart) and part.content == ""
            ]
            for message in messages
        ]
        return jsonify({"response": messages})

    except Exception as e:
        user_promt = user_input_to_model_request(user_input)
        error_response = exception_to_model_response(e)
        logger.error(error_response)
        return jsonify({"response": [user_promt, error_response]})


async def _agent_worker(prompt: str, history: str, user_id: str, research: bool = False) -> Any:
    from loguru import logger as _logger
    from ckanext.chat.bot.agent import (
        Deps, agent, research_agent,
        mcp_available, get_user_token, config,
    )
    from ckanext.chat.bot.utils import init_dynamic_models, dynamic_models_initialized

    log = _logger.bind(process="worker", user_id=user_id)
    log.debug(f"Worker starting for {user_id}")

    if not dynamic_models_initialized:
        init_dynamic_models()

    deps = Deps(user_id=user_id)
    msg_history = _validate_history(history)

    if mcp_available():
        base = toolkit.config.get("ckanext.chat.mcp_url") or toolkit.config.get("ckan.site_url")
        if not base:
            log.warning("MCP available but no mcp_url or site_url configured, falling back to ckan_agent")
        else:
            token = get_user_token(user_id)
            if token:
                mcp_url = base.rstrip("/") + "/mcp" if not toolkit.config.get("ckanext.chat.mcp_url") else base
                deps.mcp_token = token
                deps.mcp_url = mcp_url
                log.info(f"MCP path enabled, url={mcp_url}")
            else:
                log.warning("MCP available but token creation failed, falling back to ckan_agent")

    if deps.mcp_url:
        log.info("Using MCP execution path (JSON-RPC)")
    else:
        log.info("Using ckan_agent fallback path")

    active_agent = research_agent if research else agent
    limits = (
        UsageLimits(request_limit=10, total_tokens_limit=config.MAX_TOKENS_RESEARCH_AGENT)
        if research else
        UsageLimits(request_limit=6, total_tokens_limit=config.MAX_TOKENS_FRONT_AGENT)
    )

    r = await active_agent.run(
        user_prompt=prompt,
        message_history=msg_history,
        deps=deps,
        usage_limits=limits,
    )

    log.debug(f"Worker done, result type: {type(r)}")
    await _logger.complete()
    return r



blueprint.add_url_rule(
    "/chat",
    view_func=ChatView.as_view(str("chat")),
)

blueprint.add_url_rule(
    "/chat/ask",
    view_func=ask,
    methods=["POST"],
)


def get_blueprint():
    return blueprint
