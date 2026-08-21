import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union, Literal

import aiofiles
import aiohttp
import ckan.model as CKANmodel
import ckan.plugins as plugins
import ckan.plugins.toolkit as toolkit
import requests

from flask import Flask
from loguru import logger

from openai.resources.embeddings import Embeddings as OAI_Embeddings
from pydantic import (BaseModel, HttpUrl)
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import (AgentRunError, FallbackExceptionGroup,
                                    ModelHTTPError, ModelRetry,
                                    UnexpectedModelBehavior,
                                    UsageLimitExceeded)
from pydantic_ai.messages import (ModelMessagesTypeAdapter, ModelRequest,
                                  ModelResponse, TextPart, UserPromptPart)
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIModelSettings
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.usage import UsageLimits
from pymilvus import MilvusClient
from ckanext.chat.bot.utils import (
    RouteModel,
    get_ckan_url_patterns,
    get_ckan_action,
    get_ckan_actions,
    fuzzy_search_early_cancel,
    FuncSignature,
    merge_with_smart_defaults,
    smart_truncate_response
)


log = logger.bind(module=__name__)

# --------------------- Configuration Constants ---------------------

@dataclass
class AgentConfig:
    """Centralized configuration for agent timeouts, limits, and retries"""
    # Timeout settings (in seconds)
    CKAN_RUN_TIMEOUT: int = 90
    LITERATURE_SEARCH_TIMEOUT: int = 90
    LITERATURE_ANALYSE_TIMEOUT: int = 180
    
    # Token limits
    MAX_TOKENS_RAG_MODEL: int = 16384
    MAX_TOKENS_CKAN_RUN: int = 128000
    MAX_TOKENS_LITERATURE_SEARCH: int = 128000
    MAX_TOKENS_LITERATURE_ANALYSE: int = 128000
    MAX_TOKENS_FRONT_AGENT: int = 200000
    MAX_TOKENS_RESEARCH_AGENT: int = 1000000
    
    # Retry counts
    MAX_RETRIES_CKAN_AGENT: int = 5
    MAX_RETRIES_LITERATURE_SEARCH: int = 3
    MAX_RETRIES_FRONT_AGENT: int = 3
    MAX_RETRIES_RESEARCH_AGENT: int = 3
    
    # Request limits (per run)
    REQUEST_LIMIT_CKAN_RUN: int = 25
    REQUEST_LIMIT_LITERATURE_SEARCH: int = 10
    REQUEST_LIMIT_LITERATURE_ANALYSE: int = 50
    REQUEST_LIMIT_FRONT_AGENT: int = 10
    REQUEST_LIMIT_RESEARCH_AGENT: int = 50
    
    # Response truncation settings
    SMART_TRUNCATE_MAX_TOKENS: int = 8000
    
    @classmethod
    def from_config(cls) -> 'AgentConfig':
        """Load configuration from CKAN config if available"""
        # Could be extended to read from toolkit.config
        return cls()

# Global config instance
config = AgentConfig.from_config()

# --------------------- Helper Functions ---------------------

app = Flask(__name__)

# --------------------- Model & Agent Setup ---------------------


def build_model(model_name: str = None) -> OpenAIChatModel:
    provider_type = toolkit.config.get("ckanext.chat.provider", "azure")
    name = model_name or toolkit.config.get("ckanext.chat.model_name") or toolkit.config.get("ckanext.chat.deployment", "gpt-4o-mini")
    base_url = toolkit.config.get("ckanext.chat.base_url") or toolkit.config.get("ckanext.chat.completion_url", "https://your.chat.api")
    api_key = toolkit.config.get("ckanext.chat.api_key") or toolkit.config.get("ckanext.chat.api_token", "your-api-token")
    api_version = toolkit.config.get("ckanext.chat.api_version", "2024-06-01")

    if provider_type == "azure":
        provider = AzureProvider(
            azure_endpoint=base_url,
            api_version=api_version,
            api_key=api_key,
        )
    elif provider_type == "openai":
        from pydantic_ai.providers.openai import OpenAIProvider
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider_type!r}. Use 'azure' or 'openai'.")

    return OpenAIChatModel(name, provider=provider)


def mcp_available() -> bool:
    return plugins.plugin_loaded('mcp')


deployment = toolkit.config.get("ckanext.chat.model_name") or toolkit.config.get("ckanext.chat.deployment", "gpt-4o-mini")
rag_model_settings = OpenAIModelSettings(
    model_name=deployment,
    max_tokens=16384,
)
model = build_model()

think_model_name = toolkit.config.get("ckanext.chat.think_model_name") or toolkit.config.get("ckanext.chat.model_name") or "gpt-4.1-mini"
think_model = build_model(think_model_name)

# --------------------- Milvus and CKAN Setup ---------------------

milvus_url = toolkit.config.get("ckanext.chat.milvus_url", "")
milvus_token = toolkit.config.get("ckanext.chat.milvus_token", "")
collection_name = toolkit.config.get("ckanext.chat.collection_name", "")
embedding_model = toolkit.config.get(
    "ckanext.chat.embedding_model", "text-embedding-3-small"
)
embedding_api = toolkit.config.get("ckanext.chat.embedding_api", "")

vector_dim = None
if milvus_url:
    try:
        milvus_client = MilvusClient(uri=milvus_url, token=milvus_token) if milvus_token else MilvusClient(uri=milvus_url)
    except Exception as e:
        log.warning(f"Milvus connection failed: {e}")
        milvus_client = None
    if milvus_client:
        try:
            collection_info = milvus_client.describe_collection(
                collection_name=collection_name
            )
            vector_field = None
            for entry in collection_info["fields"]:
                if "params" in entry.keys() and "dim" in entry["params"].keys():
                    vector_field = entry
                    break
            if vector_field:
                vector_dim = vector_field["params"]["dim"]
                field_name = vector_field["name"]
                log.debug(f"Found vector field: {field_name}")
                log.debug(f"Vector dimension is: {vector_dim}")
            else:
                vector_dim = None
                log.debug("No vector field found in the collection schema.")
        except Exception as e:
            log.warning(f"Milvus collection lookup failed: {e}")
            vector_dim = None
    else:
        log.debug("Milvus client not initialized.")
else:
    milvus_client = None

# Global aiohttp session for connection pooling
_global_http_session: Optional[aiohttp.ClientSession] = None

def get_http_session() -> aiohttp.ClientSession:
    """Get or create global aiohttp session for connection pooling"""
    global _global_http_session
    if _global_http_session is None or _global_http_session.closed:
        # Create session with optimized settings
        connector = aiohttp.TCPConnector(
            limit=100,  # Max connections
            limit_per_host=10,  # Max connections per host
            ttl_dns_cache=300,  # DNS cache for 5 minutes
        )
        timeout = aiohttp.ClientTimeout(total=120, connect=10)
        _global_http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )
    return _global_http_session

@dataclass
class UploadedFile:
    filename: str
    content_type: str
    data: bytes


@dataclass
class Deps:
    user_id: str
    milvus_client: MilvusClient = field(default_factory=lambda: milvus_client)
    openai: OpenAIChatModel = field(default_factory=lambda: model)
    embeddings: Union[OAI_Embeddings, str] = field(default=embedding_api)
    mcp_token: Optional[str] = None
    mcp_url: Optional[str] = None
    embedding_model: str = field(default_factory=lambda: embedding_model)
    max_context_length: int = 8192
    collection_name: str = collection_name
    vector_dim: int = vector_dim
    http_session: aiohttp.ClientSession = field(default_factory=get_http_session)
    status_queue: Optional[asyncio.Queue] = None
    uploaded_file: Optional[UploadedFile] = None
    orchestrator: str = "Front agent"


def _push_status(deps, message: str):
    if deps.status_queue is not None:
        try:
            prefix = deps.orchestrator if hasattr(deps, 'orchestrator') else "Agent"
            if message.startswith("──"):
                padding = " " * len(prefix)
                full = f"{padding} {message}"
            else:
                full = f"{prefix} ── {message}"
            deps.status_queue.put_nowait(full)
        except Exception:
            pass


def _format_params_short(parameters: dict, max_len: int = 80) -> str:
    """Format parameters compactly for status messages and logs."""
    if not parameters:
        return ""
    priority_keys = ['q', 'fq', 'id', 'name']
    parts = []
    for key in priority_keys:
        if key in parameters:
            val = str(parameters[key])
            if len(val) > 60:
                val = val[:57] + "..."
            parts.append(f"{key}={val}")
    if not parts:
        key, val = next(iter(parameters.items()))
        val = str(val)
        if len(val) > 60:
            val = val[:57] + "..."
        parts.append(f"{key}={val}")
    result = ", ".join(parts)
    if len(result) > max_len:
        result = result[:max_len - 3] + "..."
    return result

@dataclass
class StringSlice:
    start: int
    end: int

@dataclass
class TextSlice:
    url: HttpUrl
    text: str
    offset: int
    length: int
    doc_position: float
    update_url_on_init: bool = field(default=True, repr=False)

    def __post_init__(self):
        if not self.update_url_on_init:
            return
        res_uuid = extract_resource_uuid(str(self.url))
        pkg_uuid = extract_dataset_uuid(str(self.url))
        ckan_url = toolkit.config.get("ckan.site_url")
        if res_uuid and pkg_uuid:
            endpoint = "markdown_view.highlight"
            route = get_ckan_url_patterns(endpoint)
            # if route_info is a string, it's an error message
            if isinstance(route, RouteModel):
                fill_vars = {"pkg_id": pkg_uuid, "id": res_uuid, "start": self.offset, "end": self.offset+self.length} # Replace with correct variable names for this route
                self.url = route.build_url(base_url=ckan_url,fill=fill_vars)

    
    
@dataclass
class TextResource:
    url: HttpUrl = None
    _text: Optional[str] = field(init=False, default=None)

    @property
    def text(self) -> Optional[str]:
        return self._text

    @text.setter
    def text(self, value: Optional[str]):
        self._text = value
        self.length = len(value) if value is not None else 0

    length: int = field(init=False, default=0)

    def extract_substring(self, offset: int, length: int) -> TextSlice:
        if self.text is None:
            raise ValueError("Text not loaded")
        text_slice = self.text[offset:offset + length]
        position=float(offset + len(text_slice)) / float(self.length)
        log.debug(f'extracted substring with offset {offset} and length {length}, end position relativ to document {position}')
        return TextSlice(url=self.url,text=text_slice, offset=offset, length=len(text_slice),doc_position=position)
    
    def __getstate__(self):
        # Exclude _text from serialization
        state = self.__dict__.copy()
        state["_text"] = None  # Don't serialize large text
        return state
    

# --------------------- Vector & RAG Models ---------------------


class VectorMeta(BaseModel):
    id: int
    # chunk_id: Optional[int] = None
    start: Optional[int] = None
    end: Optional[int] = None
    # chunks: Optional[HttpUrl] = None
    dataset_id: Optional[str] = None
    # dataset_url: Optional[HttpUrl] = None
    # groups: Optional[list[str]] = None
    # private: Optional[str] = None
    resource_id: Optional[str] = None
    source: Optional[HttpUrl] = None
    #view_url: Optional[list[HttpUrl]] = None
    title: Optional[str] = None


class RagHit(BaseModel):
    id: int
    distance: Optional[float] = None
    entity: VectorMeta
    texts: Optional[List[str]] = None


class LitResult(BaseModel):
    title: str = ""
    summary: str = ""
    authors: str = ""
    string_slices: Optional[list[StringSlice]]
    source: Optional[HttpUrl] = None
    #view_url: Optional[list[HttpUrl]] = None


class LitSearchResult(BaseModel):
    answer: str = ""
    search_str: Optional[list[str]] = None
    results: Optional[list[LitResult]] = None
    error: Optional[List[str]] = None

class AnalyseResult(BaseModel):
    answer: str = ""
    source: HttpUrl
    text_slices: Optional[list[TextSlice]] = None
    error: Optional[List[str]] = None

class DatasetSummary(BaseModel):
    id: str
    title: str
    name: str
    notes: Optional[str] = None
    organization: Optional[str] = None
    tags: Optional[List[str]] = None
    extras: Optional[Dict[str, str]] = None
    resources: Optional[List[Dict[str, Any]]] = None
    relevance: Optional[str] = None

class QueryAttempt(BaseModel):
    action: str
    params: Dict[str, Any]
    result_count: int

class CKANExploreResult(BaseModel):
    summary: str
    datasets_found: List[DatasetSummary]
    queries_tried: List[QueryAttempt]
    suggestions: Optional[str] = None

# --------------------- Updated RAG Agent Prompt ---------------------
rag_prompt = (
    "You perform literature retrieval using vector search and return high-quality scientific citations.\n\n"

    "PROCESS:\n"
    "Step 1: Formulate search query\n"
    "- Rephrase the user's question for better semantic matching\n"
    "- Extract key concepts and technical terms\n"
    "- Create 1-2 focused search queries\n\n"

    "Step 2: Execute rag_search ONCE\n"
    "- Use limit parameter to control result count (default: 5 sources)\n"
    "- rag_search returns RagHit objects ALREADY GROUPED BY SOURCE DOCUMENT\n"
    "- Each RagHit contains: distance (best similarity), entity (metadata), texts (list of matched chunk texts)\n"
    "- Distance is cosine similarity: 1.0 = identical, 0.0 = unrelated. Higher = more relevant.\n\n"

    "Step 3: Analyze results using chunk texts\n"
    "- Each RagHit.texts contains the actual text content of matched chunks from that source\n"
    "- READ these texts to write an informed summary for each source\n"
    "- Create one LitResult per RagHit — return ALL sources, do NOT filter by relevance\n"
    "- Write the summary based on the actual chunk content, not just the title\n"
    "- Fill string_slices with start/end from RagHit.entity\n\n"

    "Step 4: Quality check\n"
    "- Count distinct sources found\n"
    "- If < N distinct sources AND search was restrictive:\n"
    "  * Broaden the query (remove filters, add synonyms)\n"
    "  * Retry search ONCE with modified query\n"
    "- Maximum 2 search attempts total\n\n"

    "Step 5: Format citations\n"
    "- Each source: [Author/Title](source_url)\n"
    "- Add relevance summary (2-3 sentences) based on the chunk texts\n"
    "- Include similarity score if available\n\n"

    "IMPORTANT:\n"
    "- Respect the max rag_search call limit given in your task (default: 1)\n"
    "- Return ALL sources from rag_search — the calling agent decides relevance\n"
    "- Always include metrics (similarity scores, source count)\n"
    "- If searches yield few results, return what you have with explanation\n"
)

# --------------------- Updated Document Agent Prompt ---------------------

doc_prompt = (
    "You analyze documents efficiently to answer questions using adaptive navigation strategies.\n\n"
    
    "PROCESS:\n"
    "Step 1: Quick scan for structure (1 tool call)\n"
    "- Use `get_text_slice(offset=0, length=10000)` to scan beginning\n"
    "- Look for Table of Contents (ToC) or section headings\n"
    "- If ToC exists: extract section names and estimate locations\n"
    "- If no ToC: assume standard structure (Abstract, Intro, Methods, Results, Discussion, Conclusion)\n\n"
    
    "Step 2: Identify relevant sections (analysis, no tool calls)\n"
    "- Based on the question, determine which 2-3 sections likely contain answers\n"
    "- Prioritize: Results > Discussion > Methods > Introduction\n"
    "- For definitions: Introduction or Methods\n"
    "- For findings: Results or Discussion\n\n"
    
    "Step 3: Jump to relevant sections (2-3 tool calls)\n"
    "- Use `precise_text_slice(start_str, end_str)` to navigate directly\n"
    "- Use exact 10-20 character substrings from section headings\n"
    "- Start with most promising section first\n"
    "- Extract text_slice for each relevant passage\n"
    "- IMPORTANT: Never change text_slice.url - it's auto-generated for highlighting\n\n"
    
    "Step 4: Refine best passages (1-2 tool calls if needed)\n"
    "- If initial extraction is too broad, use precise_text_slice to narrow down\n"
    "- Extract 2-5 most relevant passages total\n"
    "- Each passage should directly answer part of the question\n"
    "- Skip table of contents text - extract actual content\n\n"
    
    "Step 5: Synthesize answer\n"
    "- Write coherent response synthesizing all findings\n"
    "- Cite every passage using: [Source Name](text_slice.url)\n"
    "- Every claim must have a citation\n"
    "- Include doc.url as source in output\n\n"
    
    "STRICT LIMITS:\n"
    "- Maximum 5 tool calls total (1 scan + 4 extractions)\n"
    "- Do NOT read linearly through the document\n"
    "- Do NOT extract more than 5 passages\n"
    "- If document is short (<5000 chars), one get_text_slice may suffice\n\n"
    
    "EFFICIENCY:\n"
    "- Jump directly to relevant sections using ToC/headings\n"
    "- Avoid redundant extractions\n"
    "- Stop when you have 2-3 high-quality passages that answer the question\n"
    "- Quality over quantity\n\n"
    
    "IMPORTANT:\n"
    "- text_slice.url points to highlighted passages - use them for citations\n"
    "- Use exact substrings (10-20 chars) for start_str and end_str\n"
    "- Simulate how a skilled researcher navigates, not linear reading\n"
)


# --------------------- Updated Front Agent ---------------------
front_agent_prompt = (
    "You coordinate user requests by delegating to specialized tools efficiently.\n"
    "Answer in the same language as the user.\n\n"

    "EXECUTION BEHAVIOR (CRITICAL — read this first):\n"
    "- When the user asks to create, update, upload, or modify data, EXECUTE IMMEDIATELY.\n"
    "- Do NOT ask for confirmation — the user's instruction IS the confirmation.\n"
    "- Do NOT list what you will do and ask 'shall I proceed?' — just do it.\n"
    "- Do NOT ask for metadata (title, description, tags, author) — extract them yourself from the document context.\n"
    "- Only ask ONE question if BOTH organization AND visibility are missing. Never ask more than one question total.\n"
    "- After execution, report what was done (dataset URL, resource URL, key metadata used).\n\n"

    "UPLOADED FILE HANDLING (CRITICAL):\n"
    "- When the user uploads a document, its text content is ALREADY in your context as document chunks.\n"
    "- Extract title, authors, description, and tags directly from these chunks — no tool calls needed for metadata extraction.\n"
    "- Do NOT call literature_analyse or get_resource_file_contents on uploaded documents — their content is already available to you.\n"
    "- literature_analyse is ONLY for analyzing existing CKAN resources that have valid CKAN download URLs (starting with http).\n"
    "- For resource_create, the uploaded file is AUTOMATICALLY attached by the system. Just provide package_id, name, and format.\n\n"

    "DOCUMENT UPLOAD WORKFLOW:\n"
    "When the user asks to upload/integrate a document into CKAN, execute these steps WITHOUT asking questions:\n\n"
    "Step 1 — Auto-extract metadata from the document chunks in your context:\n"
    "  - title: extract from document content (chapter title, paper title, or filename)\n"
    "  - author: extract author names from document content (look for author sections, affiliations, email addresses)\n"
    "  - notes: write a 2-3 sentence summary based on the document content\n"
    "  - tags: generate 3-6 relevant tags from the document's key topics\n"
    "  - name (slug): auto-generate from title (lowercase, hyphens, no special chars, no umlauts)\n"
    "  - Use the SAME title for both the dataset and the resource\n"
    "  - If the user provides any of these explicitly, use their values instead\n\n"
    "Step 2 — Resolve organization (only if user specified one):\n"
    "  - Call ckan_run('organization_list', {}) to get all orgs\n"
    "  - Match user's input against org names/titles (fuzzy match OK)\n"
    "  - If exactly one match: use it. If ambiguous: ask user to pick.\n"
    "  - If user did NOT specify an org: ask which org to use (this is the ONLY allowed question).\n\n"
    "Step 3 — Create dataset:\n"
    "  - Call ckan_run('package_create', {title, name, notes, author, owner_org, private, tags: [{name: tag}, ...]})\n"
    "  - private defaults to True if user said 'privat'/'private', False if 'public'/'öffentlich'\n"
    "  - If user did not specify visibility AND org: ask ONCE for both, then execute.\n\n"
    "Step 4 — Create resource:\n"
    "  - Call ckan_run('resource_create', {package_id: <id from step 3>, name: <filename>, format: <extension>})\n"
    "  - The uploaded file bytes are AUTOMATICALLY injected — do NOT try to download or reference the file.\n\n"
    "Step 5 — Report result:\n"
    "  - Show: dataset title, URL, organization, visibility, tags, author\n"
    "  - Show: resource name and format\n\n"

    "QUICK SEARCH (for information/knowledge queries):\n"
    "When the user asks a question that seeks information or knowledge, execute these steps:\n\n"

    "Step 1 — SEARCH:\n"
    "- Call literature_search with the user's question rephrased for semantic matching\n\n"

    "Step 2 — VERIFY:\n"
    "- literature_search results include summaries based on actual document content (chunk texts), not just titles.\n"
    "- Read each summary carefully and check: does the content actually address the user's question?\n"
    "- If the summary clearly covers the topic → report as 'directly relevant' finding.\n"
    "- If the summary is about a related but different topic → report as 'thematically related'.\n"
    "- Only call literature_analyse if you need deeper analysis beyond what the summary provides.\n"
    "- PARALLEL EXECUTION: When you need to analyse multiple documents, call ALL literature_analyse\n"
    "  invocations in a SINGLE tool-call turn so they run concurrently.\n\n"

    "Step 3 — SYNTHESIZE:\n"
    "- Clearly separate confirmed results from uncertain/thematic matches in your answer.\n"
    "- Only state what is supported by the actual data returned from tools.\n\n"

    "Keep it fast — max ~3-4 tool calls total.\n\n"

    "EXCEPTION — QUICK SEARCH does NOT apply to:\n"
    "- Create/update/upload/patch operations → use DOCUMENT UPLOAD WORKFLOW or direct ckan_run\n"
    "- Requests to show a specific known dataset/resource → use ckan_run('package_show'/'resource_show') directly\n"
    "- Pure administrative queries (list orgs, show user) → use ckan_run directly\n\n"

    "CKAN TOOLS:\n"
    "- ckan_run(action, params): For direct single CKAN actions (show, create, patch, list orgs, etc.)\n"
    "You CAN create organizations, datasets, resources, and update/patch them.\n"
    "Only delete and purge operations are blocked.\n\n"

    "CKAN QUERY RULES (for ckan_run):\n"
    "- Use package_search (not package_list) for listing/searching datasets\n"
    "- ALWAYS include 'include_private': True for package_search\n"
    "- For specific dataset: package_show with id=DATASET_ID_OR_NAME\n"
    "- For specific resource: resource_show with id=RESOURCE_ID\n"
    "- NEVER execute delete or purge operations\n\n"

    "literature_search: rephrase user query for semantic matching.\n"
    "literature_analyse: ONLY for analyzing existing CKAN resources with valid http(s) download URLs. Never for uploaded files.\n\n"

    "RESPONSE FORMAT:\n"
    "- Clear, direct answer synthesizing all tool results\n"
    "- Citations: [Author Year](dataset_url) — no numbered refs\n"
    "- For CKAN results: include dataset URLs when available\n"
    "- URL RULE: Always truncate resource/download URLs to the dataset URL.\n"
    "  Example: .../dataset/9fee1eac-.../resource/73f3aa3f-.../download/file.md → .../dataset/9fee1eac-...\n"
    "  Cut everything after /dataset/<dataset_id>. This lets users access all resources including PDFs.\n\n"

    "ERROR HANDLING:\n"
    "- Tool fails → interpret error, modify params, retry ONCE\n"
    "- Still fails → report what you found, note the gap\n"
    "- Never fabricate data or URLs\n\n"

    "IMPORTANT:\n"
    "- Keep information queries fast — max ~5-6 tool calls\n"
    "- Accuracy over speed — never claim a document covers a topic without evidence from its title, summary, or content\n"
    "- Quality over quantity\n"
    "- Never change data from tools, except truncating resource URLs to dataset URLs as described above\n"
)

research_agent_prompt = (
    "You conduct deep research by systematically exploring ALL available data sources — literature, CKAN packages, groups, and tags — then synthesize findings.\n"
    "Answer in the same language as the user.\n\n"

    "RESEARCH PROCESS (5 Phases):\n\n"

    "Phase 1: ANALYZE (no tools)\n"
    "- Break down the question into 2-3 key aspects\n"
    "- Formulate 1-2 testable hypotheses\n"
    "- Identify core concepts and technical terms\n"
    "- Plan search strategy across all data sources\n\n"

    "Phase 2: LITERATURE SEARCH (Milvus vector DB)\n"
    "- Call literature_search with rephrased query for each key aspect/hypothesis\n"
    "- ALWAYS rephrase the user question for better semantic matching\n"
    "- Use max_searches=4 to allow the rag_agent more search iterations\n"
    "- Target: 5-7 distinct high-quality sources\n"
    "- Maximum 3 literature_search calls\n\n"

    "Phase 3: CKAN EXPLORATION (autonomous dataset discovery)\n"
    "- Call ckan_explore with a task derived from the user's question and max_searches=10\n"
    "- The ckan_agent will autonomously search packages, browse groups, tags, and curate results\n"
    "- Review the returned datasets and their resources\n"
    "- For the top 1-3 most relevant datasets with PDF or Markdown resources,\n"
    "  call literature_analyse to read the content\n\n"

    "Phase 4: DEEP DOCUMENT ANALYSIS (3-5 analyses max)\n"
    "- Select the top 3-5 most relevant sources found across Phases 2-3\n"
    "- Call literature_analyse on each to extract precise passages with highlight URLs\n"
    "- Cross-verify quantitative claims across sources\n"
    "- Skip documents already analyzed in Phase 3\n"
    "- Maximum 5 literature_analyse calls total\n\n"

    "Phase 5: SYNTHESIZE & REPORT (no tools)\n"
    "- Phases 1-4 are your INTERNAL reasoning — do NOT output them to the user.\n"
    "- The user sees ONLY Phase 5 output — write it as a direct, concise answer.\n"
    "- Filter ruthlessly: only mention sources that are DIRECTLY relevant to the question.\n"
    "- Do NOT list or discuss irrelevant sources — simply omit them.\n"
    "- If no relevant sources were found, say so briefly and suggest next steps.\n"
    "Format:\n"
    "- Start with a direct answer (2-3 sentences)\n"
    "- List only relevant sources with brief description and dataset URL\n"
    "- If useful, add 1-2 suggestions for further steps\n"
    "- Keep the total response similar in length to a front_agent answer (short paragraph + source list)\n\n"

    "Deduplicate — the same dataset may appear in multiple phases; mention it once with all relevant context.\n\n"

    "TOOL USAGE BUDGET:\n"
    "- literature_search: max 3 calls (use max_searches=4 for deeper search)\n"
    "- ckan_explore: typically 1 call with max_searches=10\n"
    "- literature_analyse: max 5 calls total across all phases\n"
    "- ckan_run: for direct single-action calls (package_show, resource_show, etc.)\n"
    "- Total tool calls: aim for 8-15, hard ceiling at 25\n\n"

    "CITATION FORMAT:\n"
    "- Inline: [Author Year](dataset_url)\n"
    "- NO numbered references [1] or [^1^]\n"
    "- Every claim must cite source\n"
    "- URL RULE: Always truncate resource/download URLs to the dataset URL.\n"
    "  Example: .../dataset/9fee1eac-.../resource/73f3aa3f-.../download/file.md → .../dataset/9fee1eac-...\n"
    "  Cut everything after /dataset/<dataset_id>. This lets users access all resources including PDFs.\n\n"

    "QUALITY STANDARDS:\n"
    "- 5+ distinct sources minimum\n"
    "- Cross-verify quantitative data\n"
    "- Note contradictions explicitly\n"
    "- Evidence-based only, no assumptions\n"
    "- Never modify returned URLs, except truncating resource URLs to dataset URLs as described above\n\n"

    "ERROR HANDLING:\n"
    "- Tool fails → interpret error, modify params, retry ONCE\n"
    "- Still fails → note in report, continue with available data\n"
    "- If a phase yields no results, proceed to the next phase\n\n"

    "IMPORTANT:\n"
    "- ALL 5 phases are mandatory as internal reasoning steps\n"
    "- Only Phase 5 output is shown to the user — keep it concise and direct\n"
    "- Accuracy over thoroughness — never claim a document covers a topic without evidence from its actual content\n"
    "- Vector search returns semantically similar results that may NOT be directly relevant — verify before citing\n"
    "- Think strategically before each tool call\n"
    "- Quality over quantity\n"
    "- Complete research even if some sources unavailable\n"
)
# --------------------- System Prompt & Agent ---------------------

ckan_agent_prompt = (
    "You are an autonomous CKAN data explorer. You receive a research task and independently "
    "search CKAN using multiple strategies to find relevant datasets.\n\n"

    "AVAILABLE TOOLS:\n"
    "- ckan_action(action_name, parameters): Execute any CKAN action — the primary tool\n"
    "- ckan_action_names(): Discover all available CKAN actions\n"
    "- ckan_action_details(action): Get parameter docs for a specific action\n\n"

    "COMMON ACTIONS:\n"
    "- package_search: {q: 'keywords', fq: 'groups:name', include_private: True, rows: 10}\n"
    "- group_list: {all_fields: True} — lists all groups with descriptions\n"
    "- tag_list: {all_fields: True} — lists all tags\n"
    "- package_show: {id: 'dataset-id-or-name'} — full dataset details\n"
    "- organization_list: {all_fields: True} — lists all organizations\n"
    "For unknown actions, use ckan_action_details() to discover parameters.\n\n"

    "SEARCH STRATEGY:\n"
    "1. Start with package_search using the most specific terms from the task\n"
    "2. If few or no results: broaden keywords, try synonyms, translate DE↔EN\n"
    "3. Use group_list to discover relevant thematic groups, then search by group (fq='groups:name')\n"
    "4. Use tag_list to discover relevant tags, then search by tag (fq='tags:name')\n"
    "5. For the most promising datasets, use package_show to get full metadata and resource list\n"
    "6. Deduplicate results across searches\n\n"

    "QUERY RULES:\n"
    "- package_search uses Solr syntax: q for full-text, fq for field filters\n"
    "- Combine filters: fq='groups:water AND tags:quality'\n"
    "- Use '*:*' for q when filtering only by fq\n"
    "- include_private: True is auto-added by smart defaults\n\n"

    "OUTPUT:\n"
    "- Write a concise summary answering the task\n"
    "- List all found datasets with title, id, relevance explanation\n"
    "- Include tags and extras metadata for each dataset\n"
    "- Include resource list (name, format, url) for each dataset\n"
    "- Report which queries you tried and how many results each returned\n"
    "- If results are sparse, suggest what else could be tried\n\n"

    "EFFICIENCY:\n"
    "- Stay within the given search budget (max tool calls)\n"
    "- Prioritize quality over quantity — 5 relevant datasets beat 20 irrelevant ones\n"
    "- Stop searching once you have enough relevant results\n"
    "- Do NOT read document contents — just find and describe datasets\n"
)

agent = Agent(
    model=model,
    deps_type=Deps,
    instructions="".join(front_agent_prompt),
    retries=3,
)

research_agent= Agent(
    model=think_model,
    deps_type=Deps,
    instructions="".join(research_agent_prompt),
    retries=3,
)
ckan_agent = Agent(
    model=model,
    deps_type=Deps,
    output_type=CKANExploreResult,
    instructions="".join(ckan_agent_prompt),
    retries=3,
)


rag_agent = Agent(
    model=model,
    deps_type=Deps,
    output_type=LitSearchResult,
    instructions="".join(rag_prompt),
    model_settings=rag_model_settings,
)

doc_agent = Agent(
    model=model,
    deps_type=TextResource,
    output_type=AnalyseResult,
    instructions="".join(doc_prompt),
    retries=3,
    model_settings=rag_model_settings,
)


def convert_to_model_messages(history: str) -> List:
    if history:
        history_list = json.loads(history)
        return ModelMessagesTypeAdapter.validate_python(history_list)
    return None



# --------------------- Front Agent Delegation Tools ---------------------

def normalize_parameters(params: dict) -> dict:
    """
    Normalize parameters from JSON/LLM format to Python format.
    CRITICAL: Converts string booleans to Python booleans for CKAN compatibility.
    
    Conversions:
    - "true" / "True" / true → True (Python bool)
    - "false" / "False" / false → False (Python bool)  
    - "null" / "None" / null → None
    
    This is essential because CKAN actions expect Python booleans, not strings.
    """
    if not isinstance(params, dict):
        return params
    
    normalized = {}
    conversions_made = []
    
    for key, value in params.items():
        original_value = value
        
        # Boolean conversions (critical for CKAN)
        if value is True or value == "true" or value == "True":
            normalized[key] = True
            if original_value != True:
                conversions_made.append(f"{key}: '{original_value}' → True")
        elif value is False or value == "false" or value == "False":
            normalized[key] = False
            if original_value != False:
                conversions_made.append(f"{key}: '{original_value}' → False")
        elif value is None or value == "null" or value == "None":
            normalized[key] = None
            if value != None:
                conversions_made.append(f"{key}: '{original_value}' → None")
        elif isinstance(value, dict):
            normalized[key] = normalize_parameters(value)
        elif isinstance(value, list):
            normalized[key] = [normalize_parameters(item) if isinstance(item, dict) else item for item in value]
        else:
            normalized[key] = value
    
    # Log conversions for debugging
    if conversions_made:
        log.debug(f"normalize_parameters converted: {', '.join(conversions_made)}")
    
    return normalized


async def _mcp_jsonrpc(url: str, token: str, method: str, params: dict = None) -> dict:
    ssl_verify = toolkit.config.get("ckanext.chat.ssl_verify", True)
    async with aiohttp.ClientSession() as session:
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params or {},
        }
        headers = {"Authorization": token, "Content-Type": "application/json"}
        async with session.post(url, json=payload, headers=headers, ssl=ssl_verify) as resp:
            data = await resp.json()
            if "error" in data:
                raise RuntimeError(f"MCP error: {data['error'].get('message', data['error'])}")
            return data.get("result", {})


async def _mcp_fetch_data(url: str, token: str, action: str, params: dict) -> Optional[dict]:
    """Fetch data via MCP JSON-RPC, mapping CKAN action to MCP tool+operation."""
    action_to_mcp = {
        "package_search": ("ckan_package", "search"),
        "package_show": ("ckan_package", "show"),
        "package_list": ("ckan_package", "list"),
        "package_create": ("ckan_package", "create"),
        "package_update": ("ckan_package", "update"),
        "package_patch": ("ckan_package", "patch"),
        "resource_show": ("ckan_resource", "show"),
        "resource_search": ("ckan_resource", "search"),
        "resource_create": ("ckan_resource", "create"),
        "resource_update": ("ckan_resource", "update"),
        "resource_patch": ("ckan_resource", "patch"),
        "organization_list": ("ckan_organization", "list"),
        "organization_show": ("ckan_organization", "show"),
        "organization_create": ("ckan_organization", "create"),
        "organization_update": ("ckan_organization", "update"),
        "organization_patch": ("ckan_organization", "patch"),
        "group_list": ("ckan_group", "list"),
        "group_show": ("ckan_group", "show"),
        "group_create": ("ckan_group", "create"),
        "tag_list": ("ckan_tag", "list"),
        "user_show": ("ckan_user", "show"),
    }
    mapping = action_to_mcp.get(action)
    if not mapping:
        return None
    tool_name, operation = mapping
    try:
        call_args = {**params, "operation": operation}
        result = await _mcp_jsonrpc(url, token, "tools/call", {"name": tool_name, "arguments": call_args})
        content = result.get("content", [])
        texts = [c.get("text", "") for c in content if c.get("type") == "text"]
        if texts:
            try:
                return json.loads(texts[0])
            except (json.JSONDecodeError, IndexError):
                return {"raw": texts[0] if len(texts) == 1 else texts}
        return result
    except Exception as e:
        log.warning(f"MCP fetch failed for {action}, falling back to direct: {e}")
        return None



async def _ckan_fetch_data(deps, action: str, params: dict):
    """Execute a CKAN action via MCP (if available) or direct toolkit call."""
    response = None
    fetch_method = "direct"

    if deps.mcp_url and deps.mcp_token:
        response = await _mcp_fetch_data(
            deps.mcp_url, deps.mcp_token, action, params,
        )
        if response is not None:
            fetch_method = "mcp"

    if response is None:
        user = CKANmodel.User.get(user_reference=deps.user_id)
        context = {
            "user": user.name,
            "auth_user_obj": user,
            "model": CKANmodel,
            "session": CKANmodel.Session,
            "ignore_auth": False,
        }
        if action in ("resource_create", "resource_patch") and deps.uploaded_file:
            import io
            from werkzeug.datastructures import FileStorage
            uf = deps.uploaded_file
            params["upload"] = FileStorage(
                stream=io.BytesIO(uf.data),
                filename=uf.filename,
                content_type=uf.content_type,
            )
            if not params.get("url"):
                params["url"] = uf.filename
            log.info(f"Injected uploaded file '{uf.filename}' into {action}")
        response = toolkit.get_action(action)(context, params)

    return response, fetch_method


@agent.tool
@research_agent.tool
async def ckan_run(ctx: RunContext[Deps], command: str, parameters: dict={}) -> str:
    """
    Executes a CKAN action with the provided parameters.

    This function sends a command to the CKAN agent and waits for execution.
    It logs the command and parameters and handles possible timeouts
    and unexpected errors.
    Args:
        ctx (RunContext[Deps]): The context containing the dependencies for execution.
        command (str): The name of the CKAN action to be executed.
        parameters (dict): A dictionary of parameters required for the CKAN action.
    Returns:
        str: The result of the CKAN action as a JSON string, or an error message in case of failure.
    """
    if is_action_blocked(command):
        return json.dumps({"status": "fail", "action_name": command, "result": f"Action '{command}' is blocked. Destructive actions (_delete, _purge) are not allowed."})

    # Normalize parameters to handle JSON boolean/null conversions
    parameters = normalize_parameters(parameters)

    params_short = _format_params_short(parameters)
    _push_status(ctx.deps, f"Direct CKAN query: {command}" + (f" ({params_short})" if params_short else ""))

    import time as _time
    t0 = _time.monotonic()
    start_time = datetime.now(timezone.utc)
    log.info(f"ckan_run starting: action='{command}', params={json.dumps(parameters, ensure_ascii=False)[:200]}")

    try:
        merged_params = merge_with_smart_defaults(command, parameters)
        _push_status(ctx.deps, f"Direct CKAN query: fetching {command}" + (f" ({params_short})" if params_short else ""))

        t_fetch_start = _time.monotonic()
        response, fetch_method = await _ckan_fetch_data(ctx.deps, command, merged_params)
        t_fetch_end = _time.monotonic()

        _push_status(ctx.deps, f"Direct CKAN query: processing response")
        truncated = smart_truncate_response(response)

        result_dict = {
            'status': 'success',
            'action_name': command,
            'parameters': parameters,
            'result': truncated['data'],
            '_truncated': truncated['truncated'],
            '_truncation_method': truncated['truncation_method'],
            '_total_items': truncated['total_items'],
            '_showing_items': truncated['showing_items'],
            '_estimated_tokens': truncated['estimated_tokens'],
        }

        t_total = _time.monotonic()
        log.info(f"ckan_run complete: action='{command}', "
                f"method={fetch_method}, "
                f"fetch={t_fetch_end - t_fetch_start:.1f}s, total={t_total - t0:.1f}s, "
                f"items={truncated['showing_items']}/{truncated['total_items']}")
        _push_status(ctx.deps, f"Direct CKAN query complete: {truncated['showing_items']}/{truncated['total_items']} items ({t_total - t0:.1f}s)")

        return json.dumps(result_dict)

    except KeyError as e:
        log.warning(f"ckan_run KeyError: action='{command}', error={e}")
        return json.dumps({"status": "fail", "action_name": command, "result": f"Action '{command}' not found: {e}", "comment": "Check action name"})
    except Exception as e:
        log.error(f"ckan_run error: action='{command}', error_type={type(e).__name__}, error={str(e)[:200]}")
        return json.dumps({"status": "fail", "action_name": command, "result": f"{type(e).__name__}: {str(e)}", "comment": "Action failed"})


@research_agent.tool
async def ckan_explore(ctx: RunContext[Deps], task: str, max_searches: int = 6) -> str:
    """Delegate a CKAN research task to the autonomous ckan_agent.

    Use this for open-ended dataset discovery and exploration. The ckan_agent will
    autonomously search, browse groups/tags, and curate results.
    For direct single-action calls (show, create, patch), use ckan_run instead.

    Args:
        ctx: Runtime context
        task: Research task description (e.g. "Find datasets about brass corrosion in drinking water")
        max_searches: Max number of CKAN actions the agent may perform (default 6)

    Returns:
        str: JSON with summary, found datasets, queries tried, and suggestions
    """
    import time as _time
    t0 = _time.monotonic()
    _push_status(ctx.deps, f"CKAN exploration: {task}")
    log.info(f"ckan_explore starting: task='{task[:100]}', max_searches={max_searches}")

    try:
        r = await asyncio.wait_for(
            ckan_agent.run(
                f"{task}\n\nBudget: max {max_searches} tool calls.",
                deps=ctx.deps,
                usage_limits=UsageLimits(
                    request_limit=max_searches + 4,
                    total_tokens_limit=config.MAX_TOKENS_CKAN_RUN,
                ),
            ),
            timeout=config.CKAN_RUN_TIMEOUT,
        )

        usage = r.usage()
        t_total = _time.monotonic() - t0
        log.info(f"ckan_explore complete: {t_total:.1f}s, "
                f"tokens=[req:{usage.request_tokens}, resp:{usage.response_tokens}]")

        ds_info = ""
        if r.output.datasets_found:
            site_url = (toolkit.config.get("ckan.site_url") or "").rstrip("/")
            ds_links = [f"{' '.join(str(d.title or d.name).split())} ({site_url}/dataset/{d.name})" for d in r.output.datasets_found[:5]]
            ds_info = " → " + ", ".join(ds_links)
            if len(r.output.datasets_found) > 5:
                ds_info += f" (+{len(r.output.datasets_found) - 5} more)"
        _push_status(ctx.deps, f"CKAN exploration complete ({t_total:.1f}s){ds_info}")
        return r.output.model_dump_json()

    except asyncio.TimeoutError:
        log.error(f"ckan_explore timeout after {config.CKAN_RUN_TIMEOUT}s")
        return json.dumps({"summary": "Search timed out", "datasets_found": [], "queries_tried": []})
    except UsageLimitExceeded as e:
        log.error(f"ckan_explore usage limit: {e}")
        return json.dumps({"summary": f"Token limit exceeded: {e}", "datasets_found": [], "queries_tried": []})
    except Exception as e:
        log.error(f"ckan_explore error: {type(e).__name__}: {str(e)[:200]}")
        return json.dumps({"summary": f"Error: {type(e).__name__}: {str(e)}", "datasets_found": [], "queries_tried": []})


#@ckan_agent.tool_plain
def ckan_url_patterns(endpoint: str = "") -> RouteModel:
    """Get URL Flask Blueprint routes to views in CKAN if the argument endpoint is None or empty it wil return a list of endpoints. If set to an endpoint it will return the RouteModel containing arguements and the pattern to create the url.

    Args:
        endpoint (str, optional): If empty returns a list of all possible endpoints. If set returns the details of the endpoint. Defaults to "".

    Returns:
        RouteModel: All details on the Route
    """
    routes=get_ckan_url_patterns(endpoint=endpoint)
    return routes

#@ckan_agent.tool_plain
def build_ckan_url(route: RouteModel, fill: Optional[Dict[str, Any]] = None) -> str:
    """
    Build a CKAN URL for the given endpoint and fill in URL variables.

    Args:
        endpoint (str): The CKAN endpoint to build a URL for.
        fill (Optional[Dict[str, Any]]): A dictionary mapping URL variable names to their values.
        base_url (Optional[str]): Override the CKAN base site URL.

    Returns:
        str: The fully constructed CKAN URL.

    Raises:
        ValueError: If the endpoint is not found or required variables are missing.
    """
    base_url= toolkit.config.get("ckan.site_url", "")
    return route.build_url(base_url=base_url or toolkit.config.get("ckan.site_url", ""), fill=fill)


@agent.tool_plain
@research_agent.tool_plain
def get_ckan_action_names() -> Dict[str,str]:
    """Lists all available CKAN actions by action name

    Returns:
        Dict[str,str]: Dictionary of CKAN action names
    """
    return get_ckan_actions()


# --------------------- ckan_agent Tools (autonomous explorer) ---------------------

def _ckan_context(user_id: str) -> dict:
    user = CKANmodel.User.get(user_reference=user_id)
    return {
        "user": user.name,
        "auth_user_obj": user,
        "model": CKANmodel,
        "session": CKANmodel.Session,
        "ignore_auth": False,
    }


@ckan_agent.tool_plain
def ckan_action_names() -> Dict[str, str]:
    """List all available CKAN action names for discovery.

    Returns:
        Dict[str, str]: Action names and their descriptions
    """
    return get_ckan_actions()


@ckan_agent.tool_plain
def ckan_action_details(action: str) -> FuncSignature:
    """Get the parameters and docstring of a specific CKAN action.

    Args:
        action: CKAN action name (e.g. 'package_search', 'group_list')

    Returns:
        FuncSignature: Function signature with docs and parameter info
    """
    return get_ckan_action(action=action)


@ckan_agent.tool
def ckan_action(ctx: RunContext[Deps], action_name: str, parameters: Dict[str, Any] = {}) -> dict:
    """Execute any CKAN action and return the actual data (truncated for large responses).

    This is the primary tool — use it for package_search, group_list, tag_list, package_show,
    organization_list, and any other CKAN action. Parameters are auto-completed via smart defaults.

    Args:
        ctx: Runtime context
        action_name: CKAN action name (e.g. 'package_search', 'group_list')
        parameters: Action parameters (e.g. {'q': 'water', 'fq': 'groups:environment'})

    Returns:
        dict: Action result data (truncated if large)
    """
    if is_action_blocked(action_name):
        return {'error': f"Action '{action_name}' is blocked."}

    parameters = normalize_parameters(parameters)
    merged = merge_with_smart_defaults(action_name, parameters)

    params_short = _format_params_short(parameters)
    _push_status(ctx.deps, f"── CKAN agent: {action_name}" + (f" ({params_short})" if params_short else ""))

    context = _ckan_context(ctx.deps.user_id)
    try:
        response = toolkit.get_action(action_name)(context, merged)
        truncated = smart_truncate_response(response)
        log.info(f"ckan_action: {action_name}, items={truncated['total_items']}")
        return truncated['data']
    except Exception as e:
        log.warning(f"ckan_action error: {action_name}, {type(e).__name__}: {str(e)[:200]}")
        return {'error': f"{type(e).__name__}: {str(e)}"}

def extract_resource_uuid(input_string: str) -> str:
    # Regulärer Ausdruck für UUID zwischen 'resource/' und '/download'
    pattern = r'resource/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})/download'
    match = re.search(pattern, input_string)
    
    if match:
        return match.group(1)  # Gibt die gefundene UUID zurück
    else:
        return None

def extract_dataset_uuid(input_string: str) -> str:
    pattern = r'dataset/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})/resource'
    match = re.search(pattern, input_string)
    
    if match:
        return match.group(1)  # Gibt die gefundene UUID zurück
    else:
        return None

@agent.tool_plain
@research_agent.tool_plain
#@rag_agent.tool_plain
async def get_resource_file_contents(
    resource_url: str,
    ssl_verify: bool = None,
) -> TextResource:
    """
    Retrieves the content of a resource stored in filetore, allows setting max_length of output and offset to extract a slice of content

    Args:
        resource_url (str): The download url of the CKAN resource (must start with http:// or https://)
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to config value.

    Returns:
        TextResource: The raw string content of the file retrieved
    """
    if not resource_url or not resource_url.startswith(("http://", "https://")):
        raise ValueError(
            f"Invalid resource URL: '{resource_url}'. Must be a full HTTP(S) URL. "
            "Do not pass internal file IDs or UUIDs — only CKAN resource download URLs."
        )

    # Read SSL verification from config if not explicitly provided
    if ssl_verify is None:
        ssl_verify = toolkit.config.get("ckanext.chat.ssl_verify", True)

    ckan_url = toolkit.config.get("ckan.site_url")
    try:
        resource = TextResource(url=resource_url)
        resource_id = extract_resource_uuid(resource_url)
        if ckan_url in resource_url and resource_id:
            storage_path = toolkit.config.get("ckan.storage_path", "/var/lib/ckan/default")

            first_level_folder = resource_id[:3]
            second_level_folder = resource_id[3:6]
            file_name = resource_id[6:]

            file_path = os.path.join(
                storage_path,
                "resources",
                first_level_folder,
                second_level_folder,
                file_name,
            )

            log.debug(f"Loading CKAN resource file from: {file_path}")

            try:
                async with aiofiles.open(file_path, "r") as file:
                    resource.text = await file.read()
            except Exception as e:
                raise RuntimeError(f"Failed to read CKAN resource file: {e}")
        else:
            # Use pooled session from Deps - no need to create a new one!
            # This significantly improves performance for multiple downloads
            try:
                # Note: get_resource_file_contents is called from ctx where Deps has http_session
                # For now, create session here but this should be refactored to accept ctx
                async with aiohttp.ClientSession() as session:
                    async with session.get(resource_url, ssl=ssl_verify) as response:
                        response.raise_for_status()
                        content_type = response.headers.get("Content-Type", "")
                        if not content_type.startswith("text/"):
                            raise RuntimeError(f"Unsupported MIME type: {content_type}")
                        resource.text = await response.text()
            except Exception as e:
                resource.text = ""
                raise RuntimeError(f"Failed to download from {resource_url}: {e}")

        log.info(f"TextResource downloaded from {resource_url} with length: {resource.length}")
        return resource

    except Exception as e:
        raise RuntimeError(f"Failed to download and add TextResource: {e}")

@doc_agent.tool
async def get_text_slice(ctx: RunContext[TextResource], offset: int, length: int)->TextSlice:
    return ctx.deps.extract_substring(offset=offset, length=length)


@doc_agent.tool
async def precise_text_slice(
    ctx: RunContext[TextResource],
    start_str: str,
    end_str: str,
    threshold: float = 0.9
) -> TextSlice:
    """
    Finds the start and end offsets of a text slice based on fuzzy matching of start and end strings.

    Args:
        ctx (RunContext[Deps]): The context containing the dependencies.
        start_str (str): The starting string to search for.
        end_str (str): The ending string to search for.
        threshold (float): The threshold for fuzzy matching (default is 0.9).

    Returns:
        Union[Tuple[int, int], str]: A tuple containing:
            - The start and end offsets if both strings are found.
            - An error message if the start string is not found.
    """
    if not ctx.deps.text:
        log.debug("No file loaded in Deps")
        return "No file loaded in Deps"

    text = ctx.deps.text
    offset = 0  # TextResource doesn't have an offset; use 0
    slice_length = ctx.deps.length
    
    # Try exact match first
    lower_text = text.lower()
    lower_start_str = start_str.lower()
    start_idx = lower_text.find(lower_start_str)
    if start_idx != -1:
        start_end_idx = start_idx + len(start_str)
        tail = text[start_end_idx:]
        lower_tail = tail.lower()
        lower_end_str = end_str.lower()
        rel_end_idx = lower_tail.find(lower_end_str)
        if rel_end_idx != -1:
            abs_end_idx = start_end_idx + rel_end_idx + len(end_str)
            # log.debug(
            #     f"Exact match found for '{start_str}...{end_str}' at {start_idx}, end at {abs_end_idx}"
            # )
            return (offset + start_idx, offset + abs_end_idx)

    # Fall back to fuzzy search
    start_match, start_idx, start_end_idx = await fuzzy_search_early_cancel(
        start_str, text, threshold
    )
    if start_idx < 0:
        #log.debug(f"Tried to start pattern: '{start_str}' - but didn't find a match")
        return f"Start string not found: '{start_str}'"

    tail = text[start_end_idx:]
    end_match, rel_end_idx, rel_end_idx_end = await fuzzy_search_early_cancel(
        end_str, tail, threshold
    )
    if rel_end_idx < 0:
        #log.debug(f"Tried to end pattern: '{end_str}' - returning default span")
        return (offset + start_idx, offset + slice_length)

    abs_end_idx = start_end_idx + rel_end_idx_end
    # log.debug(
    #     f"Fuzzy match found for '{start_str}...{end_str}' at {start_idx}, end at {abs_end_idx}"
    # )
    start,end=offset + start_idx, offset + abs_end_idx
    position=float(end) / float(len(text))
    text_slice=TextSlice(url=ctx.deps.url,text=text[start:end],doc_position=position)
    log.debug(f"found: {text_slice}")
    return text_slice



def user_input_to_model_request(input_str: str) -> ModelRequest:
    user_prompt = UserPromptPart(content=input_str)
    return ModelRequest(parts=[user_prompt], kind="request")


def exception_to_model_response(exc: Exception) -> ModelResponse:
    if isinstance(
        exc,
        (
            UsageLimitExceeded,
            ModelRetry,
            UnexpectedModelBehavior,
            AgentRunError,
            ModelHTTPError,
            FallbackExceptionGroup,
        ),
    ):
        error_text = str(exc)
    else:
        error_text = f"An unexpected error occurred: {type(exc).__name__}: {exc}"
    error_part = TextPart(content=error_text)
    return ModelResponse(
        parts=[error_part],
        model_name="pydanticai",
        timestamp=datetime.now(timezone.utc),
        kind="response",
    )


async def get_embedding(chunks: List[str], model: str, api_url, vector_dim: int):
    if not isinstance(api_url, str):
        # must be OAI embeddings
        emb_r = await api_url.create(input=chunks, model=model, dimensions=vector_dim)
        return [vec.embedding for vec in emb_r.data]
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    data = {"chunks": chunks, "model": model}
    embedding_timeout = int(toolkit.config.get("ckanext.chat.embedding_timeout", 15))
    log.info(f"get_embedding requesting {api_url} model={model} chunks={len(chunks)} timeout={embedding_timeout}s")
    response = requests.post(
        api_url, headers=headers, data=json.dumps(data), verify=False, timeout=embedding_timeout
    )
    log.info(f"get_embedding response status={response.status_code}")

    if response.status_code == 200:
        return response.json()["embeddings"]
    else:
        raise RuntimeError(f"Embedding service error {response.status_code}: {response.text}")


async def _fetch_chunks_json(chunks_url: str, deps) -> Optional[List[str]]:
    """Fetch a .chunks JSON file from CKAN and return the list of chunk strings."""
    if not chunks_url:
        return None
    ssl_verify = toolkit.config.get("ckanext.chat.ssl_verify", True)
    ckan_url = toolkit.config.get("ckan.site_url", "")
    try:
        raw = None
        resource_id = extract_resource_uuid(chunks_url)
        if ckan_url and ckan_url in chunks_url and resource_id:
            storage_path = toolkit.config.get("ckan.storage_path", "/var/lib/ckan/default")
            file_path = os.path.join(
                storage_path, "resources",
                resource_id[:3], resource_id[3:6], resource_id[6:],
            )
            async with aiofiles.open(file_path, "r") as f:
                raw = json.loads(await f.read())
        else:
            headers = {}
            if deps.mcp_token:
                headers["Authorization"] = deps.mcp_token
            async with aiohttp.ClientSession() as session:
                async with session.get(chunks_url, headers=headers, ssl=ssl_verify) as resp:
                    resp.raise_for_status()
                    raw = await resp.json()

        if isinstance(raw, dict):
            return raw.get("chunks", [])
        if isinstance(raw, list):
            return raw
        return None
    except Exception as e:
        log.warning(f"_fetch_chunks_json failed for {chunks_url}: {e}")
        return None


def _group_hits_by_source(hits: list, chunk_texts: dict) -> List[RagHit]:
    """Group chunk-level hits by source document. Returns one RagHit per source."""
    source_groups = {}
    for hit_data in hits:
        source = str(hit_data["source"]) if hit_data.get("source") else "unknown"
        if source not in source_groups:
            source_groups[source] = {
                "best_hit": hit_data,
                "best_distance": hit_data.get("distance", 0),
                "chunk_ids": [],
            }
        group = source_groups[source]
        if (hit_data.get("distance") or 0) > group["best_distance"]:
            group["best_hit"] = hit_data
            group["best_distance"] = hit_data.get("distance", 0)
        chunk_id = hit_data.get("chunk_id")
        if chunk_id is not None:
            group["chunk_ids"].append(chunk_id)

    result = []
    for source, group in source_groups.items():
        best = group["best_hit"]
        entity_fields = {k: best.get(k) for k in VectorMeta.__fields__.keys() if k in best}
        if "entity" in best and isinstance(best["entity"], dict):
            entity_fields = best["entity"]
        elif "entity" in best and isinstance(best["entity"], VectorMeta):
            entity_fields = best["entity"].model_dump()

        texts = []
        chunks_url = best.get("chunks_url", "")
        if chunks_url and chunks_url in chunk_texts:
            chunks_list = chunk_texts[chunks_url]
            for cid in sorted(set(group["chunk_ids"])):
                if 0 <= cid < len(chunks_list):
                    texts.append(chunks_list[cid])

        rag_hit = RagHit(
            id=best.get("id", 0),
            distance=group["best_distance"],
            entity=VectorMeta(**entity_fields) if isinstance(entity_fields, dict) else entity_fields,
            texts=texts if texts else None,
        )
        result.append(rag_hit)

    result.sort(key=lambda h: h.distance or 0, reverse=True)
    return result


@rag_agent.tool
async def rag_search(
    ctx: RunContext[Deps], search_query: List[str], limit: int = 3, max_per_source: int = 3
) -> List[RagHit]:
    """Vector rag search using Milvus vector store. Returns one RagHit per source document with chunk texts.

    Args:
        ctx (RunContext[Deps]): Instance of Agent dependencies at runtime, passed in by agent framework by default
        search_query (List[str]): A list of strings for which to do the vector search with.
        limit (int, optional): Limit for amount of source documents to be returned. Defaults to 3.
        max_per_source (int, optional): Max chunks per source document. Defaults to 3.

    Returns:
        List[RagHit]: List of RagHit instances grouped by source document, each with texts from matched chunks.
    """
    if not ctx.deps.milvus_client or not ctx.deps.embeddings:
        return "The Milvus Client was not setup properly, no rag_search supported in the moment."

    queries_preview = " | ".join(search_query)
    _push_status(ctx.deps, f"── RAG agent: vector search ({len(search_query)} queries, limit={limit}): {queries_preview}")
    log.info(f"rag_search starting: queries={len(search_query)} limit={limit} max_per_source={max_per_source}")
    log.info(f"rag_search queries: {search_query}")
    _push_status(ctx.deps, "── RAG agent: generating embeddings")
    query_vectors = await get_embedding(
        search_query,
        model=ctx.deps.embedding_model,
        api_url=ctx.deps.embeddings,
        vector_dim=ctx.deps.vector_dim,
    )
    log.info(f"rag_search embedding done, starting milvus search")
    _push_status(ctx.deps, "── RAG agent: searching vector database")

    output_fields = list(VectorMeta.__fields__.keys()) + ["chunk_id", "chunks"]
    raw_hits = []
    seen_ids = set()
    source_counts = {}
    target_chunks = limit * max_per_source

    while len(raw_hits) < target_chunks:
        filter_ids = list(seen_ids)
        search_res = ctx.deps.milvus_client.search(
            collection_name=ctx.deps.collection_name,
            data=query_vectors,
            search_params={"metric_type": "COSINE", "params": {"level": 1}},
            limit=target_chunks,
            filter_params={"ids": filter_ids} if filter_ids else None,
            filter="id not in {ids}" if filter_ids else None,
            output_fields=output_fields,
            consistency_level="Bounded",
        )
        new_hits = 0
        for result_per_vector in search_res:
            for item in result_per_vector:
                item_id = item.get("id") if hasattr(item, "get") else item["id"]
                if item_id in seen_ids:
                    continue
                seen_ids.add(item_id)

                entity = item.get("entity", {}) if hasattr(item, "get") else item.get("entity", {})
                if not isinstance(entity, dict):
                    entity = {}

                source = str(entity.get("source", "unknown") or "unknown")
                count = source_counts.get(source, 0)
                if count >= max_per_source:
                    continue

                source_counts[source] = count + 1
                hit = {
                    "id": item_id,
                    "distance": item.get("distance", 0) if hasattr(item, "get") else item.get("distance", 0),
                    "entity": entity,
                    "source": source,
                    "chunks_url": str(entity.get("chunks", "") or ""),
                    "chunk_id": entity.get("chunk_id"),
                }
                raw_hits.append(hit)
                new_hits += 1

        if new_hits == 0:
            break
        log.info(f"rag_search milvus: {len(raw_hits)} chunks from {len(source_counts)} sources")

    log.info(f"rag_search: {len(raw_hits)} chunks from {len(source_counts)} sources, fetching chunk texts")
    _push_status(ctx.deps, f"── RAG agent: loading chunk texts ({len(source_counts)} sources)")

    unique_chunks_urls = set(h.get("chunks_url", "") for h in raw_hits if h.get("chunks_url"))
    log.debug(f"rag_search: {len(unique_chunks_urls)} unique chunks URLs to fetch")
    chunk_texts = {}
    for url in unique_chunks_urls:
        try:
            chunks_list = await _fetch_chunks_json(url, ctx.deps)
            if chunks_list:
                chunk_texts[url] = chunks_list
                log.debug(f"rag_search: fetched {len(chunks_list)} chunks from {url}")
            else:
                log.debug(f"rag_search: empty/null chunks from {url}")
        except Exception as e:
            log.warning(f"rag_search: skipping chunks from {url}: {e}")

    try:
        grouped_hits = _group_hits_by_source(raw_hits, chunk_texts)
    except Exception as e:
        log.error(f"rag_search: grouping failed, returning ungrouped hits: {e}")
        grouped_hits = [
            RagHit(
                id=h.get("id", 0),
                distance=h.get("distance", 0),
                entity=VectorMeta(**(h.get("entity", {}) if isinstance(h.get("entity"), dict) else {})),
            )
            for h in raw_hits[:limit]
        ]
    grouped_hits = grouped_hits[:limit]

    for h in grouped_hits:
        n_texts = len(h.texts) if h.texts else 0
        preview = h.texts[0][:150] if h.texts else "NO TEXT"
        log.debug(f"rag_search result: source={h.entity.source}, title={h.entity.title}, "
                   f"distance={h.distance:.3f}, chunks={n_texts}, preview={preview!r}")

    texts_loaded = sum(1 for h in grouped_hits if h.texts)
    src_titles = [" ".join(str(h.entity.title or h.entity.source or "?").split()) for h in grouped_hits[:5]]
    src_info = " → " + ", ".join(src_titles) if src_titles else ""
    if len(grouped_hits) > 5:
        src_info += f" (+{len(grouped_hits) - 5} more)"
    _push_status(ctx.deps, f"── RAG agent: {len(grouped_hits)} sources found ({texts_loaded} with text){src_info}")
    log.info(f"rag_search completed: {len(grouped_hits)} source hits, {texts_loaded} with chunk texts")
    return grouped_hits
        





@agent.tool
@research_agent.tool
async def literature_search(
    ctx: RunContext[Deps], search_question: str, num_results: int = 5, max_searches: int = 1
) -> list[str]:
    """Search literature via vector database (Milvus).

    Args:
        ctx: Runtime context
        search_question: Question rephrased for semantic matching
        num_results: Number of source documents to return (default 5)
        max_searches: Max number of rag_search calls the agent may perform (default 1, research_agent uses up to 4)

    Returns:
        str: JSON with answer, citations, and search metadata
    """
    start_time = datetime.now(timezone.utc)
    _push_status(ctx.deps, f"Literature search: \"{search_question}\"")
    log.info(f"literature_search starting: query='{search_question}...', num_results={num_results}, max_searches={max_searches}")

    for attempt in range(config.MAX_RETRIES_LITERATURE_SEARCH):
        try:
            if attempt > 0:
                _push_status(ctx.deps, f"Literature search: retry (attempt {attempt+1})")
            log.debug(f"literature_search attempt {attempt+1}/{config.MAX_RETRIES_LITERATURE_SEARCH}")
            r = await asyncio.wait_for(
                rag_agent.run(
                    f"Search for documents using this question: {search_question}. "
                    f"Return {num_results} results. You may call rag_search up to {max_searches} times.",
                    deps=ctx.deps,
                    usage_limits=UsageLimits(
                        request_limit=max_searches + 4,
                        total_tokens_limit=config.MAX_TOKENS_LITERATURE_SEARCH
                    ),
                ),
                timeout=config.LITERATURE_SEARCH_TIMEOUT
            )
            
            # Track usage metrics
            usage = r.usage()
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            results = r.output.results or []
            src_titles = [" ".join(str(lr.title or lr.source or "?").split()) for lr in results[:5]]
            src_info = " → " + ", ".join(src_titles) if src_titles else ""
            if len(results) > 5:
                src_info += f" (+{len(results) - 5} more)"
            _push_status(ctx.deps, f"Literature search complete ({duration_ms/1000:.1f}s, {len(results)} sources){src_info}")
            log.info(f"literature_search completed: attempt={attempt+1}, "
                    f"tokens=[request:{usage.request_tokens}, response:{usage.response_tokens}, total:{usage.total_tokens}], "
                    f"duration_ms={duration_ms:.0f}")
            output_json = r.output.model_dump_json()
            log.debug(f"literature_search rag_agent output: {output_json[:2000]}")

            return output_json
            
        except asyncio.TimeoutError:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            log.warning(f"literature_search timeout on attempt {attempt+1}/{config.MAX_RETRIES_LITERATURE_SEARCH}, duration_ms={duration_ms:.0f}")
            return json.dumps({"answer": "", "error": [f"Literature search timed out after {duration_ms/1000:.0f}s"]})
            
        except UsageLimitExceeded as e:
            log.error(f"literature_search usage limit exceeded on attempt {attempt+1}: {e}")
            return json.dumps({"answer": "", "error": [f"Token limit exceeded: {str(e)}"]})
            
        except ModelHTTPError as e:
            log.error(f"literature_search API error on attempt {attempt+1}: status={e.status_code if hasattr(e, 'status_code') else 'unknown'}")
            if attempt == config.MAX_RETRIES_LITERATURE_SEARCH - 1:
                return json.dumps({"answer": "", "error": [f"API error: {str(e)}"]})
            continue
            
        except UnexpectedModelBehavior as e:
            log.error(f"literature_search model behavior error on attempt {attempt+1}: {str(e)[:200]}")
            if attempt == config.MAX_RETRIES_LITERATURE_SEARCH - 1:
                return json.dumps({"answer": "", "error": [f"Model output validation failed: {str(e)}"]})
            continue
            
        except Exception as e:
            log.error(f"literature_search unexpected error on attempt {attempt+1}: error_type={type(e).__name__}, error={str(e)[:200]}")
            if attempt == config.MAX_RETRIES_LITERATURE_SEARCH - 1:
                return json.dumps({"answer": "", "error": [f"Literature search failed: {type(e).__name__}: {str(e)[:200]}"]})
            continue

    return json.dumps({"answer": "", "error": ["All literature_search retries exhausted"]})

@agent.tool
@research_agent.tool
async def literature_analyse(ctx: RunContext[Deps], doc: TextResource, question: str, ssl_verify: bool = None) -> list[str]:
    """
    Analyze a document to answer a question.

    Args:
        ctx: The runtime context with dependencies
        doc: TextResource with document URL
        question: Question to answer
        ssl_verify: Whether to verify SSL certificates. Defaults to config value.

    Returns:
        JSON string with analysis results
    """
    # Read SSL verification from config if not explicitly provided
    if ssl_verify is None:
        ssl_verify = toolkit.config.get("ckanext.chat.ssl_verify", True)

    doc_filename = str(doc.url).rsplit('/', 1)[-1] if doc.url else "unknown"
    _push_status(ctx.deps, f"Document analysis: {doc_filename}")
    start_time = datetime.now(timezone.utc)
    log.info(f"literature_analyse starting: doc_url='{doc.url}', question='{question[:100]}...'")

    _push_status(ctx.deps, f"── Doc agent: loading {doc_filename}")
    try:
        doc=await get_resource_file_contents(resource_url=str(doc.url),ssl_verify=ssl_verify)
        log.debug(f"literature_analyse loaded document: length={doc.length} chars")
        _push_status(ctx.deps, f"── Doc agent: loaded {doc_filename} ({doc.length:,} chars)")
    except Exception as e:
        log.error(f"literature_analyse failed to load document: error={str(e)[:200]}")
        return json.dumps({"answer": "", "source": str(doc.url), "error": [f"Failed to load document: {str(e)}"]})
    
    prompt = (
        f"Analyze the provided TextResource to determine whether it contains an answer to the question below.\n\n"
        f"**Question:** {question}\n\n"
    )

    _push_status(ctx.deps, f"── Doc agent: extracting relevant passages from {doc_filename}")
    try:
        r = await asyncio.wait_for(
            doc_agent.run(
                prompt,
                deps=doc,
                usage_limits=UsageLimits(
                    request_limit=config.REQUEST_LIMIT_LITERATURE_ANALYSE,
                    total_tokens_limit=config.MAX_TOKENS_LITERATURE_ANALYSE
                ),
            ),
            timeout=config.LITERATURE_ANALYSE_TIMEOUT
        )
        
        # Track usage metrics
        usage = r.usage()
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        _push_status(ctx.deps, f"Document analysis complete: {doc_filename} ({duration_ms/1000:.1f}s)")
        log.info(f"literature_analyse completed: "
                f"tokens=[request:{usage.request_tokens}, response:{usage.response_tokens}, total:{usage.total_tokens}], "
                f"duration_ms={duration_ms:.0f}")

        return r.output.model_dump_json()

    except asyncio.TimeoutError:
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        _push_status(ctx.deps, f"Document analysis timeout: {doc_filename}")
        log.error(f"literature_analyse timeout after {duration_ms:.0f}ms, limit={config.LITERATURE_ANALYSE_TIMEOUT}s")
        return json.dumps({"answer": "", "source": str(doc.url), "error": [f"Analysis timeout after {config.LITERATURE_ANALYSE_TIMEOUT} seconds"]})
        
    except UsageLimitExceeded as e:
        log.error(f"literature_analyse usage limit exceeded: {e}")
        return json.dumps({"answer": "", "source": str(doc.url), "error": [f"Token limit exceeded: {str(e)}"]})
        
    except ModelHTTPError as e:
        log.error(f"literature_analyse API error: status={e.status_code if hasattr(e, 'status_code') else 'unknown'}")
        return json.dumps({"answer": "", "source": str(doc.url), "error": [f"API error: {str(e)}"]})
        
    except UnexpectedModelBehavior as e:
        log.error(f"literature_analyse model behavior error: {str(e)[:200]}")
        return json.dumps({"answer": "", "source": str(doc.url), "error": [f"Model output validation failed: {str(e)}"]})
        
    except Exception as e:
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        log.error(f"literature_analyse unexpected error: error_type={type(e).__name__}, error={str(e)[:200]}, duration_ms={duration_ms:.0f}")
        return json.dumps({"answer": "", "source": str(doc.url), "error": [f"Unexpected error: {type(e).__name__}: {str(e)}"]})


BLOCKED_ACTION_SUFFIXES = ("_delete", "_purge")


def is_action_blocked(action_name: str) -> bool:
    return any(action_name.endswith(suffix) for suffix in BLOCKED_ACTION_SUFFIXES)


def get_user_token(user_id: str) -> Optional[str]:
    user = CKANmodel.User.get(user_reference=user_id)
    if not user:
        log.error(f"get_user_token: user not found for id={user_id}")
        return None
    context = {
        "user": user.name,
        "auth_user_obj": user,
        "model": CKANmodel,
        "session": CKANmodel.Session,
        "ignore_auth": False,
    }
    try:
        existing_tokens = toolkit.get_action("api_token_list")(context, {"user": user.name})
        for tok in existing_tokens:
            if tok.get("name") == "chat_agent":
                toolkit.get_action("api_token_revoke")(context, {"jti": tok["id"]})
                break
    except Exception as e:
        log.warning(f"get_user_token: could not check/revoke existing tokens: {e}")

    try:
        response = toolkit.get_action("api_token_create")(context, {"user": user.name, "name": "chat_agent"})
    except Exception as e:
        log.error(f"get_user_token: token creation failed: {e}")
        return None
    if "token" in response:
        token = response["token"]
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return token
    return None
