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

## Agent Architecture

- `front_agent` — main coordinator, delegates to tools (max 3 tool calls)
- `research_agent` — deep research mode (max 10 tool calls, uses think_model)
- `ckan_agent` — validates/optimizes CKAN action calls → `CKANResult`
- `rag_agent` — vector search via Milvus → `LitSearchResult`
- `doc_agent` — document analysis with fuzzy text extraction → `AnalyseResult`

When `ckanext-mcp` is loaded, CKAN data access goes through MCP JSON-RPC instead of direct `toolkit.get_action()`.

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

## Conventions

- Commit messages: lowercase, optional conventional prefixes (`feat:`, `fix:`, `docs:`, `refactor:`)
- Logging: use `log` (loguru) with module binding, prefer INFO level for operational messages
- Config values: read via `toolkit.config.get("ckanext.chat.<key>", default)` where needed
- No enforced formatter — no black/ruff/isort config
