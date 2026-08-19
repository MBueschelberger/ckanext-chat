# CLAUDE.md — ckanext-chat-bue

## Project Overview

CKAN extension providing an AI chat interface with a pydantic-ai multi-agent system. Deployed as a pip-installable CKAN plugin.

## Tech Stack

- Python 3.8+, CKAN extension (Flask Blueprints)
- pydantic-ai for agent orchestration
- Milvus (pymilvus) for vector search / RAG
- Azure OpenAI or OpenAI-compatible LLM backends
- Loguru for logging, tiktoken for token counting

## Project Structure

```
ckanext/chat/
├── plugin.py          # CKAN plugin entry point (IConfigDeclaration, IBlueprint)
├── api.py             # OpenAI-compatible /chat/v1/chat/completions endpoint
├── views.py           # Browser-facing /chat UI and /chat/ask POST
├── action.py          # CKAN action chat_submit
├── auth.py            # Authorization
├── helpers.py         # Template helpers
├── bot/
│   ├── agent.py       # Multi-agent system (front_agent, research_agent, ckan_agent, rag_agent, doc_agent)
│   └── utils.py       # CKAN action helpers, URL routing, fuzzy search, response truncation
├── templates/         # Jinja2 templates
├── assets/            # JS/CSS
└── tests/             # test_plugin.py (scaffold), test_chat_roundtrip.py (integration)
```

## Agent Architecture (2-Level Model)

```
front_agent / research_agent  (Orchestrator)
  ├── ckan_explore(task)       → ckan_agent (autonomous multi-step CKAN exploration)
  │     └── ckan_action()      (generic: any CKAN action via merge_with_smart_defaults)
  ├── ckan_run(action, params) → direct bypass (merge_with_smart_defaults, all actions incl. _create/_patch)
  ├── literature_search(q)     → rag_agent (budget-controlled vector search via Milvus)
  │     └── rag_search()       (Milvus vector search)
  └── literature_analyse(doc)  → doc_agent (document analysis, only from orchestrator)
```

- `front_agent` — quick coordinator, delegates to tools (max ~3-4 tool calls, quick search only)
- `research_agent` — deep research mode (5-phase workflow, max ~25 tool calls, uses think_model)
- `ckan_agent` — autonomous CKAN explorer with generic `ckan_action` tool → `CKANExploreResult`
- `rag_agent` — vector search via Milvus, budget-controlled (max_searches param) → `LitSearchResult`
- `doc_agent` — document analysis with fuzzy text extraction → `AnalyseResult`

When `ckanext-mcp` is loaded, CKAN data access in `ckan_run` goes through MCP JSON-RPC instead of direct `toolkit.get_action()`.

## Configuration

All config keys use `ckanext.chat.*` prefix in ckan.ini. Key settings:

- `provider` — `azure` or `openai`
- `model_name`, `base_url`, `api_key`, `api_version` — LLM connection
- `think_model_name` — model for research_agent
- `embedding_api`, `embedding_model`, `embedding_timeout` — embedding service
- `milvus_url`, `milvus_token`, `collection_name` — Milvus vector DB
- `ssl_verify` — SSL verification for HTTP calls

Agent timeouts, retries, and token limits are in `AgentConfig` dataclass (top of `bot/agent.py`).

## Commands

```bash
# Install (dev)
pip install -e ".[dev]"

# Run tests
pytest --ckan-ini=test.ini

# Integration test (requires running CKAN instance)
python ckanext/chat/tests/test_chat_roundtrip.py --url http://localhost:80 --token TOKEN --verbose

# Lint (matches CI)
flake8 . --count --select=E901,E999,F821,F822,F823 --show-source --statistics --exclude ckan
```

## Streaming Status Protocol

The `/chat/v1/chat/completions` endpoint (SSE streaming mode) emits inline status markers while sub-agents work:

```
[status]Literature search: "PFAS alternatives"[/status]
[status]Generating embeddings for search queries[/status]
[status]Searching vector database[/status]
[status]Analyzing vector search results: 18 hits[/status]
[status]CKAN: package_search (q=PFAS Alternativen, include_private=True)[/status]
[status]Validating query parameters for package_search (q=PFAS Alternativen)[/status]
[status]Fetching data from CKAN: package_search (q=PFAS Alternativen)[/status]
[status]Processing response from package_search[/status]
[status]CKAN complete: 10/42 items (2.3s)[/status]
[status]Loading document: iwm_bericht_v1204_2023.md[/status]
[status]Document loaded: iwm_bericht_v1204_2023.md (124,500 chars)[/status]
[status]Extracting relevant passages: iwm_bericht_v1204_2023.md[/status]
[status]Analysis complete: iwm_bericht_v1204_2023.md (84.8s)[/status]
```

- Markers appear as content deltas in standard OpenAI SSE chunks, before the final answer text
- Format: `[status]message[/status]\n` — clients parse these to render spinners/badges
- Implemented via `asyncio.Queue` in `Deps.status_queue`, pushed from tools with `_push_status()`
- Only active during streaming (`status_queue` is `None` for non-streaming requests)
- `_run_agent_stream` in `api.py` runs the agent in a background task and polls both the status queue and text output queue every 200ms

### Implementation details

- `Deps.status_queue: Optional[asyncio.Queue]` — set by `_run_agent_stream`, `None` otherwise
- `_push_status(deps, message)` helper in `bot/agent.py` — no-op when queue is `None`
- `literature_analyse` was changed from `@agent.tool_plain` to `@agent.tool` to access `ctx.deps`
- `_run_agent_stream` uses `asyncio.create_task` for the agent worker; main loop exits when `output_queue` receives `None` sentinel or `task.done()` (safety fallback)
- Status-emitting tools: `ckan_run`, `rag_search`, `literature_search`, `literature_analyse`
- Intermediate status messages cover: parameter validation, embedding generation, vector DB search, data fetching (direct/MCP), response processing, document loading, passage extraction, and retries

## Authentication

Both `/chat/ask` and `/chat/ask/stream` support dual authentication:

1. **API token**: `Authorization: Bearer <CKAN_API_TOKEN>` header (validated via `ckan.lib.api_token`)
2. **CKAN session**: standard session cookie (existing browser login)

Token auth is checked first; session auth is the fallback. Implemented in `_authenticate_request()` in `views.py`.

The `/chat/v1/chat/completions` endpoint (`api.py`) uses its own `_authenticate()` with token auth only.

## File Upload (resource_create / resource_patch)

The `/chat/ask` and `/chat/ask/stream` endpoints accept `multipart/form-data` with an optional `upload` field:

```bash
curl -X POST http://localhost:5000/chat/ask/stream \
  -H "Authorization: Bearer <TOKEN>" \
  -F "text=Lade diese CSV als Resource in Dataset xyz hoch" \
  -F "upload=@data.csv"
```

Flow:
1. `views.py` — `_extract_upload()` reads `request.files["upload"]` into an `UploadedFile` dataclass (filename, content_type, bytes)
2. `UploadedFile` is stored in `Deps.uploaded_file` and passed through `_agent_worker`
3. `agent.py` `ckan_run` — when the action is `resource_create` or `resource_patch` and `Deps.uploaded_file` is set, the file is injected as a `werkzeug.FileStorage` into the action parameters before calling `toolkit.get_action()`

Key types:
- `UploadedFile` dataclass in `bot/agent.py` (filename, content_type, data as bytes)
- `Deps.uploaded_file: Optional[UploadedFile]` — set by views.py, consumed by `ckan_run`

## Open WebUI Integration

The pipe function `iwm_rag_streaming.py` connects Open WebUI to the CKAN chat endpoints:

- **Without files**: routes to `/chat/v1/chat/completions` (OpenAI-compatible SSE with `[status]...[/status]` markers)
- **With files**: routes to `/chat/ask/stream` (multipart form with `text` + `upload` fields, SSE with `event: status` / `event: done`)

`file_handler = True` on the Pipe class prevents Open WebUI's default RAG processing. Files are read from Open WebUI storage via `Files.get_file_by_id()` + `Storage.get_file()` and forwarded as multipart upload.

## CKAN Action Execution

`ckan_run` executes ALL CKAN actions directly via `merge_with_smart_defaults` + `_ckan_fetch_data()` (no LLM sub-agent). This applies to all actions including `_create` and `_patch`.

`ckan_explore` delegates open-ended dataset discovery to the autonomous `ckan_agent`, which uses a generic `ckan_action` tool to call any CKAN action. The orchestrator gives a task description and a search budget (`max_searches`).

## Timeout & Retry Strategy

- `literature_search`: timeout 90s, **no retry on timeout** (retrying a slow LLM compounds delay without improving results; other error types still retry up to `MAX_RETRIES_LITERATURE_SEARCH`)
- `literature_analyse`: timeout 180s (large documents like 50k+ char markdown need more processing time for the `doc_agent`)
- `ckan_run`: timeout 90s (unchanged)

## Conventions

- Commit messages: lowercase, optional conventional prefixes (`feat:`, `fix:`, `docs:`, `refactor:`)
- Logging: use `log` (loguru) with module binding, prefer INFO level for operational messages
- Config values: read via `toolkit.config.get("ckanext.chat.<key>", default)` where needed
- No enforced formatter — no black/ruff/isort config
