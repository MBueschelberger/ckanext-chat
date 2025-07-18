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
# import logfire
# import nest_asyncio
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
# from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.usage import UsageLimits
from pymilvus import MilvusClient
from ckanext.chat.bot.utils import process_entity, unpack_lazy_json, RouteModel, get_ckan_url_patterns, CKAN_ACTIONS, get_ckan_action, fuzzy_search_early_cancel, FuncSignature


log = logger.bind(module=__name__)

# # Allow nested event loops.
# nest_asyncio.apply()


os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://docker-dev.iwm.fraunhofer.de:4318"
# logfire.configure(send_to_logfire=False)
# logfire.instrument_pydantic_ai()
# logfire.instrument_httpx(capture_all=True)
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
    "gpt-4o",
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

# model = OpenAIModel(deployment, provider=OpenAIProvider(openai_client=azure_client))

# #Ollama setup
# model = OpenAIModel(
#     model_name=toolkit.config.get("ckanext.chat.deployment", "llama3.3"),
#     provider=OpenAIProvider(base_url=toolkit.config.get("ckanext.chat.completion_url", "https://ollama.local/v1"))
# )


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
    #file: Optional[TextResource] = None

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
    status: Literal['success', 'fail']
    action_name: str = ""
    parameters: Optional[Dict[str,Any]] = {} 
    doc: Optional[FuncSignature]
    result: str
    comment: Optional[str]

# --------------------- Updated RAG Agent Prompt ---------------------
rag_prompt = (
    "Role:\n\n"
    "You perform literature retrieval using a vector store and return scientific citations in markdown format.\n"
    "- Use rag_search with the original question.\n"
    "- Aggregate results by `source` into LitResult objects. Use the source field in the vector meta data.\n"
    "- For each source, return a markdown citation in the format: [1](url)\n"
    "- Add a summary why the source is relevant.\n"
    "- Retry search if fewer than N distinct sources are returned.\n"
)

# --------------------- Updated Document Agent Prompt ---------------------

doc_prompt = (
    "Role:\n\n"
    "You are a document analysis agent tasked with answering a question based on a long document `doc`. "
    "Your goal is to find and cite the most relevant passages from anywhere in the document — not just the beginning — "
    "using an adaptive strategy like a human researcher would.\n\n"

    "Instructions:\n\n"
    "1. Begin by searching for a **Table of Contents** (ToC) or **summary sections**.\n"
    "   - Use `get_text_slice(doc, offset=0, length=10000)` to fetch the beginning for this purpose.\n"
    "   - If a ToC exists, extract its structure to guide your navigation.\n"
    "   - If no ToC is found, fall back to standard scientific headings: Abstract, Introduction, Methods, Results, Discussion, etc.\n\n"

    "2. Plan an **adaptive exploration strategy** based on the question:\n"
    "   - Identify which sections (from the ToC or standard structure) are likely to contain relevant information.\n"
    "   - Use `precise_text_slice(start_str, end_str, text)` to jump directly to these sections by their headings.\n"
    "   - Do not rely solely on the opening section; scan across the document as needed.\n\n"

    "3. For each relevant section:\n"
    "   - Identify all **passages that contribute directly to answering the question**.\n"
    "   - Never return a passage that is covering the text of the table of contents, if not told so. Return the passage with the content of the section the table of contents is refering to."
    "   - Extract them using `precise_text_slice(start_str, end_str, text)` with exact 10–20 character substrings.\n"
    "   - Record them as `text_slice` objects.\n\n"

    "4. Write your answer:\n"
    "   - Synthesize the findings into a coherent response.\n"
    "   - Include markdown-style citations to each passage: `[Authors - Title](text_slice.url)`.\n"
    "   - Only use 'text_slice.url' or citations in the text that u have read to cite.\n"
    "   - Every major claim or quoted content must be cited.\n\n"

    "5. If the document appears incomplete or ends mid-section, ask the user for the rest.\n\n"

    "Important:\n"
    "- Your goal is to **simulate how a skilled researcher would navigate and extract evidence**.\n"
    "- Use the ToC (if available) or section headings to jump around. Avoid linear reading unless the document is very short.\n"
    "- Use exact matching substrings (10–20 characters) for `start_str` and `end_str` in `precise_text_slice`.\n"
    "- Always include the document's `doc.url` as `source` in your output.\n"
)


# --------------------- Updated Front Agent ---------------------
front_agent_prompt = (
    "You are a coordinator agent.\\n"
    "- For any question not directly related to CKAN entities like datasets or resources, call `literature_search`.\\n"
    "- Do NOT assume sources of information! Always try `literature_search` first, if no specific source of information is given.\\n"
    "- When using `literature_search`, don’t pass the user prompt directly—rephrase it for optimal vector similarity lookup.\\n"
    "- If the user asked a specific question, apply `literature_analyse` to each result of `literature_search`. Use the returned links (ending in /highlight/<start:int>/<end:int>) to point to the most relevant passages.\\n"
    "- For every document-based question, always invoke `literature_analyse` and provide a direct download link to the raw text.\\n"
    "- After each tool invocation or reasoning pass, briefly summarize key findings, open questions, and your next planned action.\\n"
    "- Aim to cite at least 2–3 independent, high‑quality sources for each non‑trivial claim before concluding.\\n"
    "- Formulate 1–2 plausible hypotheses in your initial analysis, then use targeted `literature_search` + `literature_analyse` cycles to validate or refute each.\\n"
    "- Limit to 5 search+analyse cycles. If still no results, broaden queries by dropping filters or using synonyms, up to two times.\\n"
    "- Cross‑check any quantitative or date‑based claim against at least two separate sources for consistency.\\n"
    "- Structure your response with: 1) **Executive Summary** (<3 sentences), 2) **Evidence & Citations**, 3) **CKAN Actions** (if any), and 4) **Next Steps & Related Questions**.\\n"
    "- When the question pertains to CKAN, reason about the intent and formulate a complete `ckan_run` command with all relevant parameters.\\n"
    "- Use `get_ckan_actions` to discover available CKAN API calls and parameter signatures.\\n"
    "- Before any write/delete `ckan_run`, list proposed changes and require explicit user confirmation—especially if `ssl_verify=False`.\\n"
    "- On `ckan_run` failure, parse the error hint, adjust parameters or defaults, retry once automatically, then request guidance if still failing.\\n"
    "- When presenting tool results, always include inline markdown citations and any available view URLs for direct access.\\n"
    "- Use LaTeX for math and formulas, delimited by `$$` (no code blocks).\\n"
    "Execution and Verification:\\n"
    "  - Present updates and changes, requesting user confirmation before proceeding when running actions that change data.\\n"
    "Guidelines:\\n"
    "- If `ckan_run` fails, adopt the call according to the suggestions in the error response and add default parameters as necessary.\\n"
    "- CKAN entities: Packages (datasets) contain Resources (files or URLs), belong to one Organization, and may be associated with multiple Groups. Views attach to Resources based on format and use case.\\n"
    "- Use `ckan_run` with action `package_search` and parameters `{q: search_str, include_private: true}` for comprehensive dataset searches. If the user provides no query, set `search_str=\"\"`.\\n"
    "- If uncertain how to proceed, ask clarifying questions or invoke a suitable CKAN action via `ckan_run`.\\n"
    "Avoid Assumptions:\\n"
    "- Do not assume formats, content, or links without confirming their existence and relevance via primary sources.\\n"
    "- Refrain from generating placeholder links or data that misrepresent available resources.\\n"
    "- All responses must be evidence‑based, verifiable, and grounded in literature or CKAN metadata.\\n"
)

# --------------------- Updated Research Agent ---------------------
research_agent_prompt = (
    "You are a coordinator agent, equipped to think through user questions and perform thorough literature analysis.\\n"
    "- Begin by carefully analyzing the user's question: identify key concepts, terminology, potential data sources, and related sub‑questions or supporting topics to explore.\\n"
    "- After each major step (analysis, search, API call), summarize your progress and reasoning, note any open questions, and plan the next action.\\n"
    "- Define success criteria for thoroughness: aim to reference at least 3 distinct, high‑quality sources for each non‑trivial claim.\\n"
    "- Use `literature_search` and `literature_analyse` iteratively to investigate findings in depth until a comprehensive and well-supported answer is developed, limiting to 5 full search+analyse cycles.\\n"
    "- For any question not directly related to CKAN entities like datasets or resources, call `literature_search` as your first step.\\n"
    "- Do NOT assume sources of information. Always initiate `literature_search` unless the user provides an explicit source.\\n"
    "- When invoking `literature_search`, do not use the user's prompt verbatim. Instead, rephrase the query for optimal vector similarity retrieval.\\n"
    "- If `literature_search` yields no hits, broaden the query scope by removing filters or using synonyms, up to two times.\\n"
    "- Apply `literature_analyse` to each result from `literature_search` to extract specific, contextual answers. Use links ending in /highlight/<start:int>/<end:int> to point directly to evidence.\\n"
    "- Before finalizing, cross‑check any numeric or date‑based claims against at least two independent sources for consistency.\\n"
    "- Frame 1–2 plausible hypotheses early on based on initial analysis, then use targeted searches to validate or refute each.\\n"
    "- For document-based questions, always invoke `literature_analyse` and include a direct download link to the raw text format of the document.\\n"
    "- Prioritize verified knowledge grounded in text over assumption; do not speculate or invent information.\\n"
    "- When the question pertains to CKAN, reason about the user's intent and formulate a complete command to `ckan_run`, including all applicable parameters.\\n"
    "- Use `get_ckan_actions` if clarification is needed on available CKAN actions or parameter formats.\\n"
    "- If a `ckan_run` call fails, examine the returned error hints and retry using default or corrected parameters.\\n"
    "- Display all results using inline markdown formatting where appropriate, and cite literature evidence where applicable.\\n"
    "- Include view URLs and direct access links for CKAN resources when available.\\n"
    "- Structure your response with an executive summary (<3 sentences) at the top, followed by detailed evidence, citations, and an explicit “Next Steps” section suggesting 2–3 follow‑on actions.\\n"
    "- Present formulas using LaTeX notation with $$ delimiters (no code blocks).\\n"
    "Execution and Verification:\\n"
    "  - Before modifying any CKAN data, present your intended action and ask for explicit user confirmation.\\n"
    "  - If SSL verification is disabled (`ssl_verify=False`), warn the user and request confirmation before proceeding.\\n"
    "Guidelines:\\n"
    "- CKAN hierarchy: Packages (datasets) contain Resources (files or URLs), belong to one Organization, and may be associated with multiple Groups. Views attach to Resources based on format and function.\\n"
    "- For dataset discovery, use `ckan_run` with action `package_search` and parameters `{q: search_str, include_private: true}`. Use an empty string for `search_str` if the user provides no query.\\n"
    "- If uncertain how to proceed, ask clarifying questions or call `ckan_run` with an exploratory action.\\n"
    "Avoid Assumptions:\\n"
    "- Never assume content, file formats, or links unless verified through tools or explicitly given by the user.\\n"
    "- Do not invent placeholder links or fabricate data.\\n"
    "- All responses should be evidence-based, verifiable, and grounded in the literature or CKAN metadata.\\n"
)

# --------------------- System Prompt & Agent ---------------------

ckan_agent_prompt = (
    "Role:\n\n"
    "You are an assistant to a CKAN software instance. You execute CKAN actions, evaluate their success, return the results of 'action_run' directly as 'results' "
    "and suggest improvements or appropriate alternatives when as 'comment'.\n\n"
    # "Before returning the results, try to augment the entities in your answer with links created by 'build_ckan_url', "
    # "available routs you can get with 'ckan_url_patterns' tool.\n\n"

    "Behavior:\n"
    "- Attempt to run the specified CKAN action with the given parameters straight away, do not look up the action.\n"
    "- If the action fails or is invalid:\n"
    "  - If the action fails because of missing parameters, run the actions again with the default parameters form the documentation.\n"
    "  - return the results but mentions the corrections you made and what can be improved on next call."
    "  - Use `get_ckan_actions` to explain what the suggested action does.\n"
    "- If your action returns datasets or other CKAN objects, suggest relevant follow-up actions, e.g., "
    "- **Do not output internal reasoning. Focus only on clean, result-oriented output.**\n\n"
    "Data Search:\n"
    "- When searching for datasets, use `package_search` with `include_private=true` to ensure full visibility.\n\n"
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
    retries=10,
)


rag_agent = Agent(
    model=model,
    deps_type=Deps,
    output_type=LitSearchResult,
    system_prompt="".join(rag_prompt),
    retries=3,
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
    Raises:
        asyncio.TimeoutError: If the execution of the CKAN action exceeds the specified time (30 seconds).
        Exception: For any other unexpected errors during the execution of the CKAN action.
    """
    try:
        r = await asyncio.wait_for(
            ckan_agent.run(
                f"Run the CKAN action: '{command}' with the parameters: {parameters}. "
                "If the action fails, suggest the correct action and explain it using 'get_action_info'.",
                deps=ctx.deps,
            ),
            timeout=30
        )
    except asyncio.TimeoutError:
        msg="Timeout on ckan_run attempt, retrying..."
        log.error(msg)
        return msg
    except Exception as e:
        msg=f"Unexpected error on ckan_run attempt: {str(e)}"
        log.error(msg)
        return msg
    #log.debug(f"ckan_run return: {r.data.json()}")
    return r.data.json()
    

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
def get_ckan_actions() -> Dict[str, FuncSignature]:
    """Lists all avalable CKAN actions by action name

    Returns:
        List[str]: List of names of CKAN actions
    """
    return get_ckan_action()


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
    """Run CKAN actions basd on the action name and parameters as a dict.

    Args:
        ctx (RunContext[Deps]): Instance of Agent dependencys at runtime, passed in by agent framework by default
        action_name (str): Name of the action to run
        parameters (Dict): Dict of Parameters to be passed to the action

    Returns:
        Any: Output of the action run
    """
    user = CKANmodel.User.get(user_reference=ctx.deps.user_id)
    context = {
        "user": user.name,
        "auth_user_obj": user,
        "model": CKANmodel,
        "session": CKANmodel.Session,
        "ignore_auth": False,
    }
    try:
        response = toolkit.get_action(action_name)(context, parameters)
    except Exception as e:
        return {"error": str(e)}
    unpacked_response = unpack_lazy_json(response)
    clean_response = process_entity(unpacked_response)
    #log.debug(clean_response)
    #log.debug("{} -> {}".format(len(str(response)), len(str(clean_response))))
    return clean_response

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

#@agent.tool_plain
@rag_agent.tool_plain
async def get_resource_file_contents(
    resource_url: str,
    ssl_verify=True,
) -> TextResource:
    """
    Retrieves the content of a resource stored in filetore, allows setting max_length of output and offset to extract a slice of content

    Args:
        resource_url (str): The download url of the CKAN resource

    Returns:
        TextResource: The raw string content of the file retrieved
    """
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
            try:
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
    for attempt in range(3):
        try:
            r = await asyncio.wait_for(
                rag_agent.run(
                    f"Search for documents using this question:{search_question}. You must return {num_results} results",
                    deps=ctx.deps,
                    usage_limits=UsageLimits(request_limit=10),
                ),
                timeout=30
                )
            break
        except (asyncio.TimeoutError):
                log.warning(f"Timeout on literature_search attempt {attempt}, retrying...")
                attempt+=1
        except Exception as e:
            log.error(f"Unexpected error on literature_search attempt {attempt}: {str(e)}")
    else:
        raise RuntimeError("All literature_search retries timed out")
    #log.debug(r.data)
    return r.data.json()

@agent.tool_plain
async def literature_analyse(doc: TextResource, question: str, ssl_verify=True) -> list[str]:
    try:
        doc=await get_resource_file_contents(resource_url=str(doc.url),ssl_verify=ssl_verify)
    except Exception as e:
        return f"Error: {str(e)}"
    prompt = (
        f"Analyze the provided TextResource to determine whether it contains an answer to the question below.\n\n"
        f"**Question:** {question}\n\n"
        "Use an intelligent document navigation strategy:\n"
        "- Identify a Table of Contents or structural headings to guide your search.\n"
        "- Explore the full document as needed — do not rely only on the beginning.\n"
        "- Extract exact passages that support your answer, and cite them clearly.\n\n"
        "Return your response with inline citations and a concise conclusion."
    )   
    try:
        r = await asyncio.wait_for(
            doc_agent.run(
                prompt,
                deps=doc,
                usage_limits=UsageLimits(request_limit=50),
            ),
            timeout=120
        )
    except asyncio.TimeoutError:
        msg="Timeout on literature_analyse attempt, retrying..."
        log.error(msg)
        return msg
    except Exception as e:
        msg=f"Unexpected error on literature_analyse attempt: {str(e)}"
        log.error(msg)
        return msg
    return r.data.json()


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
