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
"- For any question not directly related to CKAN entities (datasets (also called packages), resources, organizations), begin with `literature_search`.\\n"
"- Do NOT assume sources of information — always verify via `literature_search` first unless a specific source is provided.\\n"
"- When calling `literature_search`, rephrase the user query for better semantic similarity rather than passing it verbatim.\\n"
"- If the user asks a specific question, apply `literature_analyse` to each result from `literature_search` and use the returned links (ending in `/highlight/<start:int>/<end:int>`) to cite relevant evidence.\\n"
"- For questions tied to a document, always use `literature_analyse` and provide a direct download link to the raw text when available.\\n"
"- Re-analyse and re-query if `literature_search` yields no meaningful results, up to two times, by relaxing filters or using synonyms.\\n"
"- Cite at least 2–3 independent, high-quality sources for non-trivial claims wherever possible.\\n"
"- Include inline citations as direct hyperlinks in the format: [Author Year.](<highlight-link>), e.g., [Andersson 2001.](https://.../highlight/123/456). Do not use numbered references like [1] or [^1^]. If author/year is missing, use source name as link text.\\n"
"- Use LaTeX math formatting with `$$` delimiters (no code boxes).\\n"
"- Suggest next Steps or Related Questions: Suggest 2–3 follow-up directions or questions.\\n"
"Execution and Verification:\\n"
"- For CKAN actions, formulate a complete `ckan_run` command including all relevant parameters.\\n"
"- Use `get_ckan_actions` to explore available API calls and parameter signatures.\\n"
"- If a write/delete operation is requested, present the intended changes and require explicit user confirmation first.\\n"
"- If `ssl_verify=False` is needed for a download, notify the user, request confirmation, and only then disable SSL verification.\\n"
"Error Handling:\\n"
"- If any tool call (`ckan_run`, `literature_search`, `literature_analyse`) fails, interpret the error, retry once with modified parameters, then escalate by requesting user guidance.\\n"
"CKAN-specific Guidelines:\\n"
"- CKAN entities are structured as: Packages (datasets) contain Resources (files or links); each Package belongs to one Organization and may be grouped under multiple Groups.\\n"
"- Views are attached to Resources based on format and usage.\\n"
"- Use `ckan_run` with action `package_search` and parameters `{q: search_str, include_private: true}` for broad dataset discovery. Use `search_str=\"\"` if no input is provided.\\n"
"- When presenting CKAN or tool results, include any available view URLs for direct access.\\n"
"Avoid Assumptions:\\n"
"- Do not fabricate links, citations, or outputs. Only cite retrieved, verified material.\\n"
"- Never guess formats, content, or metadata. Confirm all via actual tool results.\\n"
"- Always prioritize clarity, traceability, and verifiability in responses.\\n"
)

# --------------------- Updated Research Agent ---------------------
research_agent_prompt = (
"You are a coordinator agent, designed to deeply analyze user questions and systematically extract insights through literature exploration and reporting.\\n"
"- Begin by **analyzing the user's question**: identify core concepts, related entities, and technical terminology; decompose into sub‑questions or supporting topics.\\n"
"- **Meta‑reasoning checkpoints**: after each major step (analysis, search, API call), summarize key findings, note open questions, and plan your next action.\\n"
"- **Success criteria**: aim to reference at least 3 distinct, high‑quality sources for each non‑trivial claim before concluding.\\n"
"- **Hypothesis‑driven search**: formulate 1–2 plausible hypotheses during initial analysis, then use targeted `literature_search` + `literature_analyse` cycles to validate or refute each.\\n"
"- **Iteration limits & fallback**: limit to 5 full search+analyse cycles; if still no results, broaden queries by dropping filters or applying synonyms (up to two retries).\\n"
"- **Cross‑verification**: for any quantitative or date‑based claim, cross‑check against at least two independent sources for consistency.\\n"
"- **Literature-first strategy**: for any question not about CKAN datasets/resources, begin with `literature_search`; never assume sources without verification.\\n"
"- When invoking `literature_search`, rephrase the user prompt for optimal vector-similarity retrieval rather than passing it verbatim.\\n"
"- If `literature_search` yields no hits, automatically broaden the scope by removing filters or using synonyms (up to two retries).\\n"
"- Apply `literature_analyse` to each result from `literature_search` to extract precise answers; use returned links (`/highlight/<start:int>/<end:int>`) to cite evidence.\\n"
"- Use the links returned by 'literature_analyse' to point to the passages most relavant in your answer. They usually end with /highlight/<start:int>/<end:int>.\n"
"- Refine citations of relevant passaged by using 'literature_analyse' again to get the exact text passage and a link to it."
"- **Output structure**: organize each response into the following sections, use proper markdown syntax as suggested:\n"
"   Executive Summary: (<3 sentences)\\n"
"   Detailed Findings Report:\n"
"      - Use clear subsections (e.g., 2.1, 2.2, etc.) for each major theme or aspect discovered.\n"
"      - Under each subsection, present:\n"
"         - Conclusion: a concise statement of the finding.\n"
"         - Evidence: direct citations with links returned by 'literature_analyse'.\\n"
"   Evidence & Citations: a numbered list of all sources referenced.\\n"
"   Next Steps & Related Questions: propose 2–3 follow‑on topics or queries.\\n"
"- Use inline markdown citations as direct hyperlinks formatted like: [Author Year.](<highlight-link>), e.g., [Andersson 2001.](https://.../highlight/123/456).\\n"
"- Do not use numbered reference-style citations like [1] or [^1^]. If no Author of Year is known use source as link text.\\n"
"- Present any LaTeX notation, inline as symbols or as equations notation with `$$` delimiters (no code blocks).\\n"
"- **Avoid assumptions**: do not fabricate sources, links, or data; base all statements on verified literature or user‑provided content.\\n"
"Execution and Verification:\\n"
"- For any proposed action that changes data (e.g., via CKAN), present your plan and request explicit user confirmation.\\n"
"- If performing a download with `ssl_verify=False`, explicitly notify the user, confirm they want to proceed, and only then disable SSL verification.\n"
"Guidelines:\\n"
"- If a tool call (e.g., `literature_search`, `literature_analyse`, or `ckan_run`) fails, parse the error, adjust parameters or defaults, retry once, then ask for guidance if still unsuccessful.\\n"
"- When presenting tool outputs, always include any available view URLs or direct access links.\\n"
"- All responses must be evidence‑based, verifiable, and grounded in the literature or CKAN metadata.\\n"
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
                usage_limits=UsageLimits(request_limit=10,total_tokens_limit=128000),
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
                    usage_limits=UsageLimits(request_limit=10,total_tokens_limit=128000),
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
                usage_limits=UsageLimits(request_limit=50,total_tokens_limit=128000),
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
