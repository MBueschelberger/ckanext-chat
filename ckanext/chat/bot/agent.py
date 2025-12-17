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
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
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
    LITERATURE_SEARCH_TIMEOUT: int = 30
    LITERATURE_ANALYSE_TIMEOUT: int = 120
    
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
    REQUEST_LIMIT_FRONT_AGENT: int = 6
    REQUEST_LIMIT_RESEARCH_AGENT: int = 10
    
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

# Azure Setup
deployment = toolkit.config.get("ckanext.chat.deployment", "gpt-4o-mini")
rag_model_settings = OpenAIModelSettings(
    model_name=deployment,
    max_tokens=16384,
    # openai_reasoning_effort= "low"
)
model = OpenAIModel(
    "gpt-4o-mini",
    provider=AzureProvider(
        azure_endpoint=toolkit.config.get(
            "ckanext.chat.completion_url", "https://your.chat.api"
        ),
        api_version="2024-06-01",
        api_key=toolkit.config.get("ckanext.chat.api_token", "your-api-token"),
    ),
)

think_model = OpenAIModel(
    "gpt-4.1-mini",
    provider=AzureProvider(
        azure_endpoint=toolkit.config.get(
            "ckanext.chat.completion_url", "https://your.chat.api"
        ),
        api_version="2024-06-01",
        api_key=toolkit.config.get("ckanext.chat.api_token", "your-api-token"),
    ),
)

# --------------------- Milvus and CKAN Setup ---------------------

milvus_url = toolkit.config.get("ckanext.chat.milvus_url", "")
collection_name = toolkit.config.get("ckanext.chat.collection_name", "")
embedding_model = toolkit.config.get(
    "ckanext.chat.embedding_model", "text-embedding-3-small"
)
embedding_api = toolkit.config.get("ckanext.chat.embedding_api", "")

vector_dim = None
if milvus_url:
    milvus_client = MilvusClient(uri=milvus_url)
    if milvus_client:
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
class Deps:
    user_id: str
    milvus_client: MilvusClient = field(default_factory=lambda: milvus_client)
    openai: OpenAIModel = field(default_factory=lambda: model)
    embeddings: Union[OAI_Embeddings, str] = field(default=embedding_api)
    embedding_model: str = field(default_factory=lambda: embedding_model)
    max_context_length: int = 8192
    collection_name: str = collection_name
    vector_dim: int = vector_dim
    http_session: aiohttp.ClientSession = field(default_factory=get_http_session)
    #file: Optional[TextResource] = None

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


class RagHit(BaseModel):
    id: int
    distance: Optional[float] = None
    entity: VectorMeta


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

class CKANResult(BaseModel):
    """Result from CKAN agent execution"""
    status: Literal['success', 'fail']
    action_name: str = ""
    parameters: Optional[Dict[str,Any]] = {}
    result: str  # The actual result from the CKAN action, or error message
    comment: Optional[str] = None  # Additional suggestions or explanations
    parameters_auto_added: Optional[Dict[str,Any]] = None  # Parameters automatically filled by smart defaults
    metrics: Optional[Dict[str, int]] = None  # Response metrics (size, tokens, item count)
    action_suggestion: Optional[str] = None  # If wrong command was used, suggest the correct one

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
    "- rag_search returns RagHit objects with distance metrics\n"
    "- Closer distance = higher similarity (0.0 = perfect match)\n\n"
    
    "Step 3: Aggregate and rank results\n"
    "- Group RagHit objects by 'source' field\n"
    "- Calculate average similarity per source\n"
    "- Rank sources by: similarity + diversity\n"
    "- Create one LitResult per unique source\n"
    "- Fill string_slices with start/end from RagHit.entity\n\n"
    
    "Step 4: Quality check\n"
    "- Count distinct sources found\n"
    "- If < N distinct sources AND search was restrictive:\n"
    "  * Broaden the query (remove filters, add synonyms)\n"
    "  * Retry search ONCE with modified query\n"
    "- Maximum 2 search attempts total\n\n"
    
    "Step 5: Format citations\n"
    "- Each source: [Author/Title](source_url)\n"
    "- Add relevance summary (2-3 sentences)\n"
    "- Include similarity score if available\n\n"
    
    "IMPORTANT:\n"
    "- NEVER call rag_search more than 2 times\n"
    "- Quality over quantity - 3-5 highly relevant sources better than 10 mediocre ones\n"
    "- Always include metrics (similarity scores, source count)\n"
    "- If second search still yields few results, return what you have with explanation\n"
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
    "You coordinate user requests by delegating to specialized tools efficiently.\n\n"
    
    "DECISION TREE:\n"
    "1. Analyze user question type:\n"
    "   - CKAN data query (datasets, resources, orgs) → use ckan_run\n"
    "   - General knowledge/literature → use literature_search\n"
    "   - Document analysis (specific file) → use literature_analyse\n"
    "   - Mixed query → literature_search first, then ckan_run if needed\n\n"
    
    "2. Execute efficiently:\n"
    "   - Maximum 3 tool calls per response (e.g., search + analyze + ckan)\n"
    "   - Prefer single comprehensive call over multiple small ones\n"
    "   - Only call tools when necessary\n\n"
    
    "TOOL USAGE GUIDELINES:\n\n"
    
    "**ckan_run - CRITICAL RULES:**\n"
    "1. For ANY dataset query (\"show me datasets\", \"what datasets\", \"list datasets\"), ALWAYS use:\n"
    "   → ckan_run('package_search', {'q': '*:*', 'include_private': True})\n"
    "   \n"
    "2. NEVER use 'package_list' - it doesn't show metadata or private datasets\n"
    "   \n"
    "3. For searching specific datasets:\n"
    "   - By tag: ckan_run('package_search', {'q': 'tags:climate', 'include_private': True})\n"
    "   - By keyword: ckan_run('package_search', {'q': 'water', 'include_private': True})\n"
    "   - All datasets: ckan_run('package_search', {'q': '*:*', 'include_private': True})\n"
    "   \n"
    "4. ALWAYS include 'include_private': True (boolean True, NOT string 'true')\n"
    "   - This shows BOTH public AND private datasets user has access to\n"
    "   - Without it, users only see public datasets\n"
    "   \n"
    "5. Example conversation:\n"
    "   User: \"What datasets do I have?\"\n"
    "   ✅ CORRECT: ckan_run('package_search', {'q': '*:*', 'include_private': True})\n"
    "   ❌ WRONG: ckan_run('package_list', {}) or ckan_run('package_list', {'limit': 10})\n"
    "   \n"
    "6. Only use other actions for specific needs:\n"
    "   - 'package_show': Get details of ONE specific dataset by ID\n"
    "   - 'organization_list': List organizations\n"
    "   - 'group_list': List groups\n"
    "   \n"
    "7. Warn before write/delete operations\n\n"
    
    "**literature_search:**\n"
    "- ALWAYS rephrase user query for better semantic matching\n"
    "- Returns LitSearchResult with sources and citations\n"
    "- If results insufficient: try ONE more time with broader query\n"
    "- Use returned similarity scores to rank relevance\n\n"
    
    "**literature_analyse:**\n"
    "- Only when detailed document analysis needed\n"
    "- Returns text_slices with highlight URLs\n"
    "- Each URL format: /highlight/<start>/<end>\n"
    "- NEVER modify returned URLs\n\n"
    
    "RESPONSE FORMAT:\n"
    "- Write clear, direct answer synthesizing tool results\n"
    "- Citations: [Author Year](url) - NO numbered refs like [1]\n"
    "- Math: use $$ delimiters\n"
    "- Include 2-3 follow-up suggestions\n"
    "- For CKAN results: include view_urls when available\n\n"
    
    "ERROR HANDLING:\n"
    "- Tool fails → interpret error, modify params, retry ONCE\n"
    "- Still fails → explain to user and ask for guidance\n"
    "- Never fabricate data or URLs\n\n"
    
    "CKAN STRUCTURE:\n"
    "- Packages (datasets) contain Resources (files/links)\n"
    "- Packages belong to Organizations\n"
    "- Packages can be in multiple Groups\n"
    "- Resources have Views based on format\n\n"
    
    "EFFICIENCY RULES:\n"
    "- Don't call get_ckan_action_names every time - only when uncertain\n"
    "- Combine operations when possible\n"
    "- Stop when sufficient information gathered\n"
    "- Aim for 1-3 tool calls total\n\n"
    
    "IMPORTANT:\n"
    "- Maximum 3 tool calls per user query\n"
    "- Quality over quantity\n"
    "- Never change data from tools, especially URLs\n"
    "- Always verify, never assume\n"
)

research_agent_prompt = (
    "You conduct deep research by systematically exploring literature and synthesizing findings.\n\n"
    
    "RESEARCH PROCESS (5 Phases):\n\n"
    
    "Phase 1: ANALYZE (no tools, 30 seconds thinking)\n"
    "- Break down the question into 2-3 key aspects\n"
    "- Formulate 1-2 testable hypotheses\n"
    "- Identify core concepts and technical terms\n"
    "- Plan search strategy\n\n"
    
    "Phase 2: SEARCH (2-3 searches max)\n"
    "- literature_search with rephrased query for each hypothesis\n"
    "- ALWAYS rephrase user question for better semantic matching\n"
    "- If first search insufficient, broaden query and retry ONCE\n"
    "- Target: 5-7 distinct high-quality sources\n"
    "- Maximum 3 search operations total\n\n"
    
    "Phase 3: ANALYZE DOCUMENTS (3-5 analyses max)\n"
    "- literature_analyse top 3-5 most relevant sources\n"
    "- Extract precise passages with highlight URLs\n"
    "- Note key findings from each source\n"
    "- Cross-verify quantitative claims across sources\n"
    "- Maximum 5 document analyses\n\n"
    
    "Phase 4: SYNTHESIZE (no tools)\n"
    "- Validate/refute initial hypotheses\n"
    "- Identify consensus vs contradictions\n"
    "- Note confidence level for each finding\n"
    "- Prepare structured report\n\n"
    
    "Phase 5: REPORT (structured output)\n"
    "Format:\n"
    "1. Executive Summary (2-3 sentences)\n"
    "2. Key Findings (2-4 subsections)\n"
    "   2.1 [Topic]: Finding + [Evidence](url)\n"
    "   2.2 [Topic]: Finding + [Evidence](url)\n"
    "3. Evidence Summary (list all sources)\n"
    "4. Next Steps (2-3 suggestions)\n\n"
    
    "STRICT LIMITS:\n"
    "- Maximum 10 tool calls total (3 searches + 5 analyses + 2 CKAN)\n"
    "- Stop when 5+ quality sources analyzed\n"
    "- If insufficient results after limits, report what was found\n\n"
    
    "TOOL USAGE:\n"
    "**literature_search:** Rephrase query, max 3 calls\n"
    "**literature_analyse:** Extract precise evidence, max 5 calls\n"
    "**ckan_run:** Only if CKAN-specific question, max 2 calls\n\n"
    
    "CITATION FORMAT:\n"
    "- Inline: [Author Year](highlight_url)\n"
    "- NO numbered references [1] or [^1^]\n"
    "- Every claim must cite source\n"
    "- Use /highlight/<start>/<end> URLs from literature_analyse\n\n"
    
    "QUALITY STANDARDS:\n"
    "- 5+ distinct sources minimum\n"
    "- Cross-verify quantitative data\n"
    "- Note contradictions explicitly\n"
    "- Evidence-based only, no assumptions\n"
    "- Never modify returned URLs\n\n"
    
    "ERROR HANDLING:\n"
    "- Tool fails → interpret error, modify params, retry ONCE\n"
    "- Still fails → note in report, continue with available data\n\n"
    
    "IMPORTANT:\n"
    "- Think strategically before each tool call\n"
    "- Quality over quantity\n"
    "- Stay within 10 tool call budget\n"
    "- Complete research even if some sources unavailable\n"
)
# --------------------- System Prompt & Agent ---------------------

ckan_agent_prompt = (
    "You are an intelligent CKAN action optimizer that corrects, completes, and executes queries efficiently.\n\n"
    
    "YOUR ROLE:\n"
    "1. Receive action + parameters from front_agent\n"
    "2. Optimize the action (correct if suboptimal)\n"
    "3. Complete missing parameters (add defaults)\n"
    "4. Execute via run_action ONCE\n"
    "5. Suggest pagination if response is large\n\n"
    
    "ACTION OPTIMIZATION RULES:\n"
    "Suboptimal Actions → Better Alternatives:\n"
    "- 'package_list' → 'package_search' (richer metadata, supports private datasets)\n"
    "- 'current_package_list_with_resources' → 'package_search' (more flexible, better filtering)\n"
    
    "Why redirect:\n"
    "- package_search: Shows full metadata, supports private datasets, allows filtering\n"
    "- package_list: Only shows names, no metadata, limited usefulness\n\n"
    
    "PARAMETER AUTO-COMPLETION:\n"
    "For 'package_search' (most common), ALWAYS ensure these parameters:\n"
    "- 'q': '*:*' (if missing or empty - means \"all datasets\")\n"
    "- 'include_private': True (CRITICAL - shows both public and private datasets)\n"
    "- 'rows': 10 (pagination - reasonable default)\n"
    "- 'start': 0 (pagination - first page)\n"
    
    "For other actions:\n"
    "- Trust merge_with_smart_defaults() to add required parameters\n"
    "- You focus on dataset search actions\n\n"
    
    "EXECUTION PROCESS:\n"
    "1. Analyze received action and CHANGE IT if needed:\n"
    "   \n"
    "   IF action == 'package_list' OR action == 'current_package_list_with_resources':\n"
    "       action_to_use = 'package_search'  # CHANGE the action name\n"
    "       parameters_to_use = {'q': '*:*', 'include_private': True, 'rows': 10, 'start': 0}\n"
    "   \n"
    "   ELIF action == 'package_search':\n"
    "       action_to_use = 'package_search'  # Keep the action\n"
    "       parameters_to_use = complete parameters (ensure q, include_private, rows, start)\n"
    "   \n"
    "   ELSE:\n"
    "       action_to_use = original action  # Keep as-is\n"
    "       parameters_to_use = original parameters\n"
    
    "2. Complete parameters for package_search:\n"
    "   - If 'q' missing or empty: set to '*:*'\n"
    "   - If 'include_private' missing: set to True\n"
    "   - If 'rows' missing: set to 10\n"
    "   - If 'start' missing: set to 0\n"
    
    "3. Call run_action with the CORRECTED action name + COMPLETED parameters:\n"
    "   run_action(action_to_use, parameters_to_use)\n"
    "   \n"
    "   CRITICAL: Use action_to_use (NOT the original action!)\n"
    
    "4. Check response:\n"
    "   - If success=True and items_returned > 50:\n"
    "     Add comment: 'Large result ({count} items). Consider pagination: rows=10, start=0/10/20...'\n"
  "   - If success=False:\n"
    "     Copy error to result\n"
    
    "5. Return CKANResult with status, action_name, parameters, result, comment\n\n"
    
    "EXAMPLES:\n"
    
    "Example 1 - Action Correction:\n"
    "Input: action='package_list', parameters={}\n"
    "Think: 'package_list is suboptimal, redirecting to package_search'\n"
    "Execute: run_action('package_search', {'q': '*:*', 'include_private': True, 'rows': 10, 'start': 0})\n"
    "Return: status='success', action_name='package_search', comment='Redirected from package_list for richer data'\n"
    
    "Example 2 - Parameter Completion:\n"
    "Input: action='package_search', parameters={'q': 'climate'}\n"
    "Think: 'Missing include_private and pagination'\n"
    "Execute: run_action('package_search', {'q': 'climate', 'include_private': True, 'rows': 10, 'start': 0})\n"
    "Return: status='success', action_name='package_search', parameters_auto_added={'include_private': True, 'rows': 10, 'start': 0}\n"
    
    "Example 3 - Large Result:\n"
    "Execute: run_action returns items_returned=127\n"
    "Return: status='success', comment='Large result (127 items). Consider pagination: rows=10, start=0/10/20...'\n\n"
    
    "CRITICAL RULES:\n"
    "- Call run_action EXACTLY ONCE (after optimization)\n"
    "- ALWAYS add 'include_private': True for package_search\n"
    "- ALWAYS redirect package_list → package_search\n"
    "- Log what you changed for debugging\n"
    "- Be helpful - explain corrections in comment field\n"
)

agent = Agent(
    model=model,
    deps_type=Deps,
    system_prompt="".join(front_agent_prompt),
    retries=3,
    # model_settings=OpenAIModelSettings(openai_reasoning_effort= "low")
)

research_agent= Agent(
    model=think_model,
    deps_type=Deps,
    system_prompt="".join(research_agent_prompt),
    retries=3,
    # model_settings=OpenAIModelSettings(openai_reasoning_effort= "low")
)
ckan_agent = Agent(
    model=model,
    deps_type=Deps,
    output_type=CKANResult,
    system_prompt="".join(ckan_agent_prompt),
    retries=5,
)


rag_agent = Agent(
    model=model,
    deps_type=Deps,
    output_type=LitSearchResult,
    system_prompt="".join(rag_prompt),
    #retries=3,
    model_settings=rag_model_settings,
    # model_settings=OpenAIModelSettings(openai_reasoning_effort= "low")
)

doc_agent = Agent(
    model=model,
    deps_type=TextResource,
    output_type=AnalyseResult,
    system_prompt="".join(doc_prompt),
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
    # Normalize parameters to handle JSON boolean/null conversions
    parameters = normalize_parameters(parameters)
    
    start_time = datetime.now(timezone.utc)
    log.info(f"ckan_run starting: action='{command}', params={json.dumps(parameters)[:100]}")
    
    try:
        r = await asyncio.wait_for(
            ckan_agent.run(
                f"Run the CKAN action: '{command}' with the parameters: {parameters}. "
                "If the action fails, suggest the correct action and explain it using 'get_ckan_action_details'.",
                deps=ctx.deps,
                usage_limits=UsageLimits(
                    request_limit=config.REQUEST_LIMIT_CKAN_RUN,
                    total_tokens_limit=config.MAX_TOKENS_CKAN_RUN
                ),
            ),
            timeout=config.CKAN_RUN_TIMEOUT
        )
        
        # Track usage metrics
        usage = r.usage()
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        log.info(f"ckan_run agent validation: action='{command}', "
                f"tokens=[request:{usage.request_tokens}, response:{usage.response_tokens}, total:{usage.total_tokens}], "
                f"duration_ms={duration_ms:.0f}")
        
        # Log what ckan_agent returned
        ckan_result = r.output
        log.info(f"ckan_run agent result: status={ckan_result.status}, "
                f"action={ckan_result.action_name}, "
                f"result_preview={str(ckan_result.result)[:100]}")
        
        # If successful, fetch and truncate data separately for front_agent
        if ckan_result.status == 'success':
            try:
                # CRITICAL: Use the CORRECTED action name from ckan_agent, not the original command
                corrected_action = ckan_result.action_name or command
                corrected_params = ckan_result.parameters or parameters
                
                # Fetch data separately (not sent to ckan_agent LLM)
                merged_params = merge_with_smart_defaults(corrected_action, corrected_params)
                user = CKANmodel.User.get(user_reference=ctx.deps.user_id)
                context = {
                    "user": user.name,
                    "auth_user_obj": user,
                    "model": CKANmodel,
                    "session": CKANmodel.Session,
                    "ignore_auth": False,
                }
                
                response = toolkit.get_action(corrected_action)(context, merged_params)
                
                # Apply smart truncation
                truncated = smart_truncate_response(response)
                
                # Combine ckan_agent result with truncated data
                result_dict = ckan_result.model_dump()
                result_dict['data'] = truncated['data']
                result_dict['_truncated'] = truncated['truncated']
                result_dict['_truncation_method'] = truncated['truncation_method']
                result_dict['_total_items'] = truncated['total_items']
                result_dict['_showing_items'] = truncated['showing_items']
                result_dict['_estimated_tokens'] = truncated['estimated_tokens']
                
                total_duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                log.info(f"ckan_run data fetch complete: action='{command}', "
                        f"truncation={truncated['truncation_method']}, "
                        f"items={truncated['showing_items']}/{truncated['total_items']}, "
                        f"total_duration_ms={total_duration_ms:.0f}")
                
                final_result = json.dumps(result_dict)
                log.debug(f"ckan_run final result size: {len(final_result)} chars")
                return final_result
            except Exception as e:
                log.error(f"ckan_run data fetch error: {str(e)[:200]}")
                # Return original result if data fetching fails
                return r.output.model_dump_json()
        
        # If failed, log and return as-is
        log.warning(f"ckan_run failed: status={ckan_result.status}, result={ckan_result.result[:200]}")
        return r.output.model_dump_json()
        
    except asyncio.TimeoutError:
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        log.error(f"ckan_run timeout: action='{command}', duration_ms={duration_ms:.0f}, limit=90000ms")
        return json.dumps({"status": "fail", "action_name": command, "result": "Timeout after 90 seconds", "comment": "Operation took too long"})
        
    except UsageLimitExceeded as e:
        log.error(f"ckan_run usage limit exceeded: action='{command}', limit={e}")
        return json.dumps({"status": "fail", "action_name": command, "result": f"Token limit exceeded: {e}", "comment": "Query too complex"})
        
    except ModelHTTPError as e:
        log.error(f"ckan_run API error: action='{command}', status={e.status_code if hasattr(e, 'status_code') else 'unknown'}")
        return json.dumps({"status": "fail", "action_name": command, "result": f"API error: {str(e)}", "comment": "Service unavailable"})
        
    except UnexpectedModelBehavior as e:
        log.error(f"ckan_run model behavior error: action='{command}', error={str(e)[:200]}")
        return json.dumps({"status": "fail", "action_name": command, "result": f"Model output validation failed: {str(e)}", "comment": "Invalid response format"})
        
    except Exception as e:
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        log.error(f"ckan_run unexpected error: action='{command}', error_type={type(e).__name__}, error={str(e)[:200]}, duration_ms={duration_ms:.0f}")
        return json.dumps({"status": "fail", "action_name": command, "result": f"Unexpected error: {type(e).__name__}: {str(e)}", "comment": "Internal error"})
    

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
@ckan_agent.tool_plain
def get_ckan_action_names() -> Dict[str,str]:
    """Lists all avalable CKAN actions by action name

    Returns:
        List[str]: List of names of CKAN actions
    """
    return get_ckan_actions()

# @agent.tool_plain
# @research_agent.tool_plain
@ckan_agent.tool_plain
def get_ckan_action_details(action: str) -> FuncSignature:
    """Returns the doc string of a CKAN action by action name

    Returns:
        FuncSignature: Doc string of the CKAN action
    """
    return get_ckan_action(action=action)


# @ckan_agent.tool_plain
# def get_action_info(action_key: str) -> dict:
#     """Get the doc string of an action

#     Args:
#         action_key (str): the str the acton is named after

#     Returns:
#         dict: a dictionary containing the doc string and the arguments definitions needed to run the action
#     """
#     from ckan.logic.action.get import help_show

#     func_model = FuncSignature(doc=help_show({}, {"name": action_key}))
#     return func_model.model_dump()




#@agent.tool
@rag_agent.tool
@ckan_agent.tool
def run_action(ctx: RunContext[Deps], action_name: str, parameters: Dict) -> Any:
    """Run CKAN actions with smart parameter filling and metrics.
    
    Returns ONLY metrics for ckan_agent validation (no data sent to LLM).
    Actual data fetching is done separately by ckan_run().

    Args:
        ctx (RunContext[Deps]): Instance of Agent dependencys at runtime
        action_name (str): Name of the action to run
        parameters (Dict): Dict of Parameters to be passed to the action

    Returns:
        Dict: Validation result with metrics only (no actual data)
    """
    # Track what parameters were auto-added
    merged_parameters = merge_with_smart_defaults(action_name, parameters)
    params_added = {k: v for k, v in merged_parameters.items() if k not in parameters}
    
    user = CKANmodel.User.get(user_reference=ctx.deps.user_id)
    context = {
        "user": user.name,
        "auth_user_obj": user,
        "model": CKANmodel,
        "session": CKANmodel.Session,
        "ignore_auth": False,
    }
    
    try:
        # Execute CKAN action
        response = toolkit.get_action(action_name)(context, merged_parameters)
        
        # Measure response (but don't process it)
        json_str = json.dumps(response)
        response_size_bytes = len(json_str)
        estimated_tokens = response_size_bytes // 4
        
        # Count items in response
        items_count = 1
        if isinstance(response, list):
            items_count = len(response)
        elif isinstance(response, dict):
            if 'results' in response and isinstance(response['results'], list):
                items_count = len(response['results'])
            elif 'count' in response:
                items_count = response['count']
        
        # Return ONLY metrics (no data)
        return {
            'success': True,
            'action_name': action_name,
            'parameters_used': merged_parameters,
            'parameters_auto_added': params_added if params_added else None,
            'metrics': {
                'response_size_bytes': response_size_bytes,
                'estimated_tokens': estimated_tokens,
                'items_returned': items_count
            }
        }
        
    except Exception as e:
        # Return error with details
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'parameters_attempted': merged_parameters
        }

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
        resource_url (str): The download url of the CKAN resource
        ssl_verify (bool): Whether to verify SSL certificates. Defaults to config value.

    Returns:
        TextResource: The raw string content of the file retrieved
    """
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
    response = requests.post(
        api_url, headers=headers, data=json.dumps(data), verify=False
    )

    if response.status_code == 200:
        return response.json()["embeddings"]
    else:
        return {"error": response.status_code, "message": response.text}


@rag_agent.tool
async def rag_search(
    ctx: RunContext[Deps], search_query: List[str], limit: int = 3
) -> List[RagHit]:
    """Vector rag serach using Milvus vector store

    Args:
        ctx (RunContext[Deps]): Instance of Agent dependencys at runtime, passed in by agent framework by default
        search_query (List[str]): A list of strings or which to do the vector search with.
        limit (int, optional): Limit for amount of Hits to be returned for the serach. Defaults to 3.

    Returns:
        List[RagHit]: List of RagHit instances as a result of rag search. the object provided a distance attribute with the metrics of similarity and an entity attribute containing the meta data of the vector entity in store.
    """
    if not ctx.deps.milvus_client or not ctx.deps.embeddings:
        return "The Milvus Client was not setup properly, no rag_search supported in the moment."
    else:
        query_vectors = await get_embedding(
            search_query,
            model=ctx.deps.embedding_model,
            api_url=ctx.deps.embeddings,
            vector_dim=ctx.deps.vector_dim,
        )
        num_results = 0
        hits = []
        filter_ids = []
        while num_results < limit:
            log.debug(f"{search_query} filtered by: {filter_ids}")
            search_res = ctx.deps.milvus_client.search(
                collection_name=ctx.deps.collection_name,
                data=query_vectors,
                search_params={"metric_type": "COSINE", "params": {"level": 1}},
                limit=6,
                filter_params={"ids": filter_ids} if filter_ids else None,
                filter="id not in {ids}" if filter_ids else None,
                output_fields=list(VectorMeta.__fields__.keys()),
                consistency_level="Bounded",
            )
            if search_res:
                for i in range(len(query_vectors)):
                    hit = [RagHit(**item) for item in search_res[i]]
                    hits += hit
                    filter_ids += list(set(hit.id for hit in hits))
                    # log.debug(hits)
                distinct_sources = list(set(hit.entity.source for hit in hits))
                num_results = len(distinct_sources)
                log.debug(
                    f"Rag search for:{search_query} with limit: {limit} returned {num_results} results."
                )
        return hits
        





@agent.tool
@research_agent.tool
async def literature_search(
    ctx: RunContext[Deps], search_question: str, num_results: int = 5
) -> list[str]:
    start_time = datetime.now(timezone.utc)
    log.info(f"literature_search starting: query='{search_question[:100]}...', num_results={num_results}")
    
    for attempt in range(config.MAX_RETRIES_LITERATURE_SEARCH):
        try:
            log.debug(f"literature_search attempt {attempt+1}/{config.MAX_RETRIES_LITERATURE_SEARCH}")
            r = await asyncio.wait_for(
                rag_agent.run(
                    f"Search for documents using this question:{search_question}. You must return {num_results} results",
                    deps=ctx.deps,
                    usage_limits=UsageLimits(
                        request_limit=config.REQUEST_LIMIT_LITERATURE_SEARCH,
                        total_tokens_limit=config.MAX_TOKENS_LITERATURE_SEARCH
                    ),
                ),
                timeout=config.LITERATURE_SEARCH_TIMEOUT
            )
            
            # Track usage metrics
            usage = r.usage()
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            log.info(f"literature_search completed: attempt={attempt+1}, "
                    f"tokens=[request:{usage.request_tokens}, response:{usage.response_tokens}, total:{usage.total_tokens}], "
                    f"duration_ms={duration_ms:.0f}")
            
            return r.output.model_dump_json()
            
        except asyncio.TimeoutError:
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            log.warning(f"literature_search timeout on attempt {attempt+1}/3, duration_ms={duration_ms:.0f}")
            if attempt == 2:  # Last attempt
                log.error(f"literature_search all retries timed out after {duration_ms:.0f}ms")
                raise RuntimeError("All literature_search retries timed out")
            continue
            
        except UsageLimitExceeded as e:
            log.error(f"literature_search usage limit exceeded on attempt {attempt+1}: {e}")
            return json.dumps({"answer": "", "error": [f"Token limit exceeded: {str(e)}"]})
            
        except ModelHTTPError as e:
            log.error(f"literature_search API error on attempt {attempt+1}: status={e.status_code if hasattr(e, 'status_code') else 'unknown'}")
            if attempt == 2:
                return json.dumps({"answer": "", "error": [f"API error: {str(e)}"]})
            continue
            
        except UnexpectedModelBehavior as e:
            log.error(f"literature_search model behavior error on attempt {attempt+1}: {str(e)[:200]}")
            if attempt == 2:
                return json.dumps({"answer": "", "error": [f"Model output validation failed: {str(e)}"]})
            continue
            
        except Exception as e:
            log.error(f"literature_search unexpected error on attempt {attempt+1}: error_type={type(e).__name__}, error={str(e)[:200]}")
            if attempt == 2:
                raise RuntimeError(f"All literature_search retries failed: {type(e).__name__}: {str(e)}")
            continue
    
    # Should not reach here, but just in case
    raise RuntimeError("All literature_search retries exhausted")

@agent.tool_plain
@research_agent.tool_plain
async def literature_analyse(doc: TextResource, question: str, ssl_verify: bool = None) -> list[str]:
    """
    Analyze a document to answer a question.
    
    Args:
        doc: TextResource with document URL
        question: Question to answer
        ssl_verify: Whether to verify SSL certificates. Defaults to config value.
    
    Returns:
        JSON string with analysis results
    """
    # Read SSL verification from config if not explicitly provided
    if ssl_verify is None:
        ssl_verify = toolkit.config.get("ckanext.chat.ssl_verify", True)
    
    start_time = datetime.now(timezone.utc)
    log.info(f"literature_analyse starting: doc_url='{doc.url}', question='{question[:100]}...'")
    
    try:
        doc=await get_resource_file_contents(resource_url=str(doc.url),ssl_verify=ssl_verify)
        log.debug(f"literature_analyse loaded document: length={doc.length} chars")
    except Exception as e:
        log.error(f"literature_analyse failed to load document: error={str(e)[:200]}")
        return json.dumps({"answer": "", "source": str(doc.url), "error": [f"Failed to load document: {str(e)}"]})
    
    prompt = (
        f"Analyze the provided TextResource to determine whether it contains an answer to the question below.\n\n"
        f"**Question:** {question}\n\n"
    )
    
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
        log.info(f"literature_analyse completed: "
                f"tokens=[request:{usage.request_tokens}, response:{usage.response_tokens}, total:{usage.total_tokens}], "
                f"duration_ms={duration_ms:.0f}")
        
        return r.output.model_dump_json()
        
    except asyncio.TimeoutError:
        duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        log.error(f"literature_analyse timeout after {duration_ms:.0f}ms, limit=120000ms")
        return json.dumps({"answer": "", "source": str(doc.url), "error": ["Analysis timeout after 120 seconds"]})
        
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


def get_user_token(user_id: str) -> Optional[str]:
    user = CKANmodel.User.get(user_reference=user_id)
    context = {
        "user": user.name,
        "auth_user_obj": user,
        "model": CKANmodel,
        "session": CKANmodel.Session,
        "ignore_auth": False,
    }
    parameters = {"user": user.name, "name": "chat_agent"}
    try:
        response = toolkit.get_action("api_token_create")(context, parameters)
    except Exception as e:
        return e
    if "token" in response.keys():
        token = response["token"].decode("utf-8")
        return token
    return None
