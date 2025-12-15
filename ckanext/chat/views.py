import asyncio
import multiprocessing as mp
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
from pydantic_ai.messages import TextPart
from pydantic_ai.usage import UsageLimits


# from ckanext.chat.bot.agent import (Deps, async_agent_response,
#                                     exception_to_model_response,
#                                     user_input_to_model_request)
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

def ask():
    #logger.debug(request.form)
    user_input = request.form.get("text")
    history = request.form.get("history", "")
    research= request.form.get("reserach", False)
    max_retries = 3
    attempt = 0
    tkuser = toolkit.current_user
    debug = bool(strtobool(os.environ.get("DEBUG", "false")))
    # If they're not a logged in user, don't allow them to see content
    if tkuser.name is None:
        return {"success": False, "msg": "Must be logged in to view site"}
    while attempt < max_retries:
        try:
            response = asyncio.run(
                async_agent_response(user_input, history, user_id=tkuser.id, research=research),
                debug=debug,
            )
            # Now response is guaranteed to have new_messages() if no exception occurred.
            # Ensure new_messages() is awaited in the sync wrapper if it's async
            messages = response.new_messages()
            # for msg in messages:
            #    logger.debug(msg)
            # remove empty text responses parts
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

def async_agent_response(prompt: str, history: str, user_id: str, research: bool = False) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(_agent_worker(prompt, history, user_id, research))
    finally:
        loop.close()

async def _agent_worker(prompt: str, history: str, user_id: str, research: bool = False) -> Any:
    from loguru import logger
    from ckanext.chat.bot.agent import (
        Deps, agent, research_agent, convert_to_model_messages
    )
    from ckanext.chat.bot.utils import init_dynamic_models, dynamic_models_initialized

    logger = logger.bind(process="worker", user_id=user_id)
    logger.debug(f"Worker starting for {user_id}")

    if not dynamic_models_initialized:
        init_dynamic_models()

    deps = Deps(user_id=user_id)
    msg_history = convert_to_model_messages(history)

    if research:
        r = research_agent.run(
            user_prompt=prompt,
            message_history=msg_history,
            deps=deps,
            usage_limits=UsageLimits(request_limit=10,total_tokens_limit=1000000),
        )
    else:
        r = agent.run(
            user_prompt=prompt,
            message_history=msg_history,
            deps=deps,
            usage_limits=UsageLimits(request_limit=3,total_tokens_limit=200000),
        )

    logger.debug(f"Worker done, result: {r}")
    await logger.complete()
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
