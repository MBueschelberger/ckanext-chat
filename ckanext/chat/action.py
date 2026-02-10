from typing import Any

import ckan.plugins as p
import ckan.plugins.toolkit as toolkit

if toolkit.check_ckan_version("2.10"):
    from ckan.types import Context
else:

    class Context(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)


log = __import__("logging").getLogger(__name__)


def chat_submit(context: Context, data_dict: dict[str, Any]) -> dict[str, Any]:
    """Send a chat message"""
    toolkit.check_access("chat_submit", context, data_dict)
    return True


def get_actions():
    actions = {
        "chat_submit": chat_submit,
    }
    return actions
