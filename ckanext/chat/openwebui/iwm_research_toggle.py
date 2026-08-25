"""
title: IWM Research Mode
author: Matthias Büschelberger
version: 1.0.0
required_open_webui_version: 0.6.15
"""

from pydantic import BaseModel

class Filter:
    toggle = True
    icon_url = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'><circle cx='11' cy='11' r='8'/><path d='m21 21-4.3-4.3'/><path d='m11 8v6'/><path d='m8 11h6'/></svg>"

    class Valves(BaseModel):
        pass

    def __init__(self):
        self.valves = self.Valves()

    async def inlet(self, body: dict, __user__: dict = None) -> dict:
        body.setdefault("metadata", {})
        body["metadata"]["research"] = True
        return body
