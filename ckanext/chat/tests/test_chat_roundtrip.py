#!/usr/bin/env python3
"""
Round-trip integration test for ckanext-chat's OpenAI-compatible endpoint.

Sends natural language instructions to /chat/v1/chat/completions and verifies
that the agent performs real CKAN CRUD operations via MCP/ckan_agent.

Analogous to ckanext-mcp's test_crud_lifecycle.py but driven by LLM
instructions instead of direct JSON-RPC calls.

Usage:
    # Set env vars or pass via CLI
    export CKAN_URL=http://localhost:80
    export CKAN_API_TOKEN=<your-sysadmin-api-token>

    python tests/test_chat_roundtrip.py

    # Or with args:
    python tests/test_chat_roundtrip.py --url http://localhost:80 --token <token>

    # Cleanup only (remove test artifacts):
    python tests/test_chat_roundtrip.py --cleanup
"""

import argparse
import json
import os
import re
import sys
import time
import uuid

import requests

# ---------------------------------------------------------------------------

CHAT_ENDPOINT = "/chat/v1/chat/completions"
CKAN_API = "/api/3/action"
TEST_PREFIX = "chat-roundtrip-test"
UNIQUE_SUFFIX = uuid.uuid4().hex[:8]
ORG_NAME = f"{TEST_PREFIX}-org-{UNIQUE_SUFFIX}"
DATASET_NAME = f"{TEST_PREFIX}-ds-{UNIQUE_SUFFIX}"
RESOURCE_NAME = f"{TEST_PREFIX}-res-{UNIQUE_SUFFIX}.csv"
PATCHED_TITLE = f"Patched Dataset Title {UNIQUE_SUFFIX}"
TAG_NAME = f"test-tag-{UNIQUE_SUFFIX}"


class ChatRoundtripTest:
    def __init__(self, base_url: str, api_token: str, verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.verbose = verbose
        self.chat_url = f"{self.base_url}{CHAT_ENDPOINT}"
        self.api_url = f"{self.base_url}{CKAN_API}"
        self.history = []
        self.results = []

    # -- helpers --

    def _chat(self, user_msg: str, timeout: int = 120) -> str:
        """Send a message to chat completions endpoint and return assistant reply."""
        self.history.append({"role": "user", "content": user_msg})
        payload = {
            "model": "default",
            "messages": self.history,
        }
        if self.verbose:
            print(f"\n  >>> {user_msg[:120]}...")

        resp = requests.post(
            self.chat_url,
            json=payload,
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        assistant_msg = data["choices"][0]["message"]["content"]
        self.history.append({"role": "assistant", "content": assistant_msg})

        if self.verbose:
            print(f"  <<< {assistant_msg[:200]}...")

        return assistant_msg

    def _ckan_get(self, action: str, params: dict = None) -> dict:
        """Direct CKAN API call for verification."""
        resp = requests.get(
            f"{self.api_url}/{action}",
            params=params or {},
            headers={"Authorization": self.api_token},
            timeout=30,
        )
        return resp.json()

    def _ckan_post(self, action: str, data: dict) -> dict:
        """Direct CKAN API POST for cleanup."""
        resp = requests.post(
            f"{self.api_url}/{action}",
            json=data,
            headers={"Authorization": self.api_token},
            timeout=30,
        )
        return resp.json()

    def _check(self, name: str, passed: bool, detail: str = ""):
        status = "PASS" if passed else "FAIL"
        self.results.append((name, passed, detail))
        marker = "\033[92m✓\033[0m" if passed else "\033[91m✗\033[0m"
        print(f"  {marker} {name}" + (f" — {detail}" if detail and not passed else ""))

    def _reset_history(self):
        """Start a fresh conversation (no accumulated context)."""
        self.history = []

    # -- test steps --

    def step_1_create_org(self):
        """Ask the agent to create an organization."""
        print("\n[Step 1] Create organization")
        reply = self._chat(
            f"Create a new CKAN organization with the name '{ORG_NAME}' "
            f"and title 'Chat Roundtrip Test Org'. "
            f"Description should be 'Created by chat roundtrip test'. "
            f"Tell me the organization id when done."
        )

        # Verify via direct API
        result = self._ckan_get("organization_show", {"id": ORG_NAME})
        if result.get("success"):
            org = result["result"]
            self._check("org exists", org["name"] == ORG_NAME)
            self._check("org title", org["title"] == "Chat Roundtrip Test Org",
                        f"got: {org.get('title')}")
            return org["id"]
        else:
            self._check("org exists", False, f"API error: {result.get('error', {}).get('message', 'unknown')}")
            return None

    def step_2_create_dataset(self, org_id: str):
        """Ask the agent to create a dataset in the org."""
        print("\n[Step 2] Create dataset")
        reply = self._chat(
            f"Create a new dataset with name '{DATASET_NAME}', "
            f"title 'Chat Roundtrip Test Dataset', "
            f"notes 'Initial description for roundtrip test', "
            f"in the organization with id '{org_id}'. "
            f"Tell me the dataset id when done."
        )

        result = self._ckan_get("package_show", {"id": DATASET_NAME})
        if result.get("success"):
            pkg = result["result"]
            self._check("dataset exists", pkg["name"] == DATASET_NAME)
            self._check("dataset in org", pkg["owner_org"] == org_id,
                        f"got owner_org: {pkg.get('owner_org')}")
            self._check("dataset notes", "roundtrip" in pkg.get("notes", "").lower(),
                        f"got: {pkg.get('notes', '')[:60]}")
            return pkg["id"]
        else:
            self._check("dataset exists", False, f"API error: {result.get('error', {}).get('message', 'unknown')}")
            return None

    def step_3_add_resource(self, pkg_id: str):
        """Ask the agent to add a resource to the dataset."""
        print("\n[Step 3] Add resource")
        reply = self._chat(
            f"Add a resource to dataset '{DATASET_NAME}' (id: {pkg_id}). "
            f"Resource name: '{RESOURCE_NAME}', "
            f"url: 'https://example.com/testdata.csv', "
            f"format: 'CSV', "
            f"description: 'Test CSV resource'. "
            f"Tell me the resource id when done."
        )

        result = self._ckan_get("package_show", {"id": pkg_id})
        if result.get("success"):
            pkg = result["result"]
            resources = pkg.get("resources", [])
            self._check("has resources", len(resources) > 0,
                        f"num_resources: {len(resources)}")
            if resources:
                res = resources[-1]
                self._check("resource format", res.get("format", "").upper() == "CSV",
                            f"got: {res.get('format')}")
                return res["id"]
        else:
            self._check("has resources", False, "could not fetch dataset")
        return None

    def step_4_patch_dataset(self, pkg_id: str):
        """Ask the agent to update the dataset title and add a tag."""
        print("\n[Step 4] Patch dataset")
        self._reset_history()
        reply = self._chat(
            f"Update the dataset with id '{pkg_id}': "
            f"change the title to '{PATCHED_TITLE}' "
            f"and add a tag '{TAG_NAME}'. "
            f"Use package_patch. Confirm what changed."
        )

        result = self._ckan_get("package_show", {"id": pkg_id})
        if result.get("success"):
            pkg = result["result"]
            self._check("title patched", pkg["title"] == PATCHED_TITLE,
                        f"got: {pkg.get('title')}")
            tag_names = [t["name"] for t in pkg.get("tags", [])]
            self._check("tag added", TAG_NAME in tag_names,
                        f"tags: {tag_names}")
        else:
            self._check("title patched", False, "could not fetch dataset")

    def step_5_search_datasets(self):
        """Ask the agent to search for our test dataset."""
        print("\n[Step 5] Search datasets")
        self._reset_history()
        reply = self._chat(
            f"Search for datasets with the tag '{TAG_NAME}'. "
            f"How many results are there? List the dataset names."
        )

        lower_reply = reply.lower()
        self._check("search found dataset",
                     DATASET_NAME in lower_reply or PATCHED_TITLE.lower() in lower_reply,
                     f"reply mentions test dataset: "
                     f"name={'yes' if DATASET_NAME in lower_reply else 'no'}, "
                     f"title={'yes' if PATCHED_TITLE.lower() in lower_reply else 'no'}")

    def step_6_list_org_datasets(self, org_id: str):
        """Ask the agent to list datasets in our test org."""
        print("\n[Step 6] List org datasets")
        self._reset_history()
        reply = self._chat(
            f"List all datasets in the organization '{ORG_NAME}'. "
            f"Tell me the names and titles."
        )

        lower_reply = reply.lower()
        self._check("org listing has dataset",
                     DATASET_NAME in lower_reply or PATCHED_TITLE.lower() in lower_reply,
                     "agent mentions our test dataset in response")

    def step_7_show_resource_details(self, res_id: str):
        """Ask agent to show resource details."""
        print("\n[Step 7] Show resource details")
        self._reset_history()
        reply = self._chat(
            f"Show me the details of the resource with id '{res_id}'. "
            f"What format is it? What is the URL?"
        )

        lower_reply = reply.lower()
        self._check("resource format mentioned",
                     "csv" in lower_reply,
                     "agent mentions CSV format")
        self._check("resource url mentioned",
                     "example.com" in lower_reply,
                     "agent mentions example.com URL")

    # -- cleanup --

    def cleanup(self):
        """Remove all test artifacts via direct CKAN API."""
        print("\n[Cleanup] Removing test artifacts...")

        # Delete dataset (also deletes resources)
        r = self._ckan_post("package_delete", {"id": DATASET_NAME})
        if r.get("success"):
            print(f"  Deleted dataset: {DATASET_NAME}")
            # Purge to fully remove
            self._ckan_post("dataset_purge", {"id": DATASET_NAME})
        else:
            print(f"  Dataset not found or already deleted: {DATASET_NAME}")

        # Delete org
        r = self._ckan_post("organization_delete", {"id": ORG_NAME})
        if r.get("success"):
            print(f"  Deleted organization: {ORG_NAME}")
            self._ckan_post("organization_purge", {"id": ORG_NAME})
        else:
            print(f"  Org not found or already deleted: {ORG_NAME}")

    def cleanup_all_test_artifacts(self):
        """Find and remove ALL chat-roundtrip-test artifacts (from previous runs)."""
        print("\n[Cleanup] Searching for all test artifacts...")

        # Find test datasets
        r = self._ckan_get("package_search", {"q": f"name:{TEST_PREFIX}*", "rows": 100})
        if r.get("success"):
            for pkg in r["result"].get("results", []):
                if pkg["name"].startswith(TEST_PREFIX):
                    self._ckan_post("package_delete", {"id": pkg["id"]})
                    self._ckan_post("dataset_purge", {"id": pkg["id"]})
                    print(f"  Purged dataset: {pkg['name']}")

        # Find test orgs
        r = self._ckan_get("organization_list", {"all_fields": True})
        if r.get("success"):
            for org in r["result"]:
                name = org["name"] if isinstance(org, dict) else org
                if isinstance(name, str) and name.startswith(TEST_PREFIX):
                    self._ckan_post("organization_delete", {"id": name})
                    self._ckan_post("organization_purge", {"id": name})
                    print(f"  Purged organization: {name}")

    # -- runner --

    def run(self):
        print(f"{'='*60}")
        print(f"Chat Completions Round-Trip Test")
        print(f"{'='*60}")
        print(f"  Endpoint: {self.chat_url}")
        print(f"  Org:      {ORG_NAME}")
        print(f"  Dataset:  {DATASET_NAME}")
        print(f"  Resource: {RESOURCE_NAME}")
        print(f"{'='*60}")

        try:
            org_id = self.step_1_create_org()
            if not org_id:
                print("\n  ABORT: org creation failed, cannot continue")
                return False

            pkg_id = self.step_2_create_dataset(org_id)
            if not pkg_id:
                print("\n  ABORT: dataset creation failed, cannot continue")
                return False

            res_id = self.step_3_add_resource(pkg_id)

            self.step_4_patch_dataset(pkg_id)

            self.step_5_search_datasets()

            self.step_6_list_org_datasets(org_id)

            if res_id:
                self.step_7_show_resource_details(res_id)

        finally:
            self.cleanup()

        # Summary
        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        failed = total - passed

        print(f"\n{'='*60}")
        print(f"Results: {passed}/{total} passed", end="")
        if failed:
            print(f", \033[91m{failed} FAILED\033[0m")
            for name, p, detail in self.results:
                if not p:
                    print(f"  \033[91m✗ {name}: {detail}\033[0m")
        else:
            print(f" \033[92m— ALL PASSED\033[0m")
        print(f"{'='*60}")

        return failed == 0


def _get_or_create_token(base_url: str, user: str, password: str, token: str = "") -> str:
    """Return an API token — use provided one, or create via CKAN API."""
    if token:
        return token

    if not user or not password:
        print("ERROR: Provide --token, or both --user and --password")
        sys.exit(1)

    # Create token via CKAN API with basic auth
    resp = requests.post(
        f"{base_url.rstrip('/')}/api/3/action/api_token_create",
        json={"user": user, "name": f"chat-roundtrip-{uuid.uuid4().hex[:8]}"},
        auth=(user, password),
        timeout=30,
    )
    if resp.status_code == 200 and resp.json().get("success"):
        tok = resp.json()["result"]["token"]
        print(f"  Created API token for user '{user}'")
        return tok

    # Fallback: try using password as the API key directly
    resp2 = requests.post(
        f"{base_url.rstrip('/')}/api/3/action/api_token_create",
        json={"user": user, "name": f"chat-roundtrip-{uuid.uuid4().hex[:8]}"},
        headers={"Authorization": password},
        timeout=30,
    )
    if resp2.status_code == 200 and resp2.json().get("success"):
        tok = resp2.json()["result"]["token"]
        print(f"  Created API token for user '{user}'")
        return tok

    print(f"ERROR: Could not create API token: {resp.text[:200]}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Chat completions round-trip test")
    parser.add_argument("--url", default=os.environ.get("CKAN_URL", "http://localhost:80"),
                        help="CKAN base URL (default: $CKAN_URL or http://localhost:80)")
    parser.add_argument("--token", default=os.environ.get("CKAN_API_TOKEN", ""),
                        help="CKAN API token (default: $CKAN_API_TOKEN)")
    parser.add_argument("--user", "-u", default=os.environ.get("CKAN_USER", ""),
                        help="CKAN username (default: $CKAN_USER)")
    parser.add_argument("--password", "-p", default=os.environ.get("CKAN_PASSWORD", ""),
                        help="CKAN password (default: $CKAN_PASSWORD)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print chat messages")
    parser.add_argument("--cleanup", action="store_true",
                        help="Only cleanup test artifacts from previous runs")
    args = parser.parse_args()

    if not args.token and not args.user:
        print("ERROR: Provide --token or --user/--password")
        sys.exit(1)

    api_token = _get_or_create_token(args.url, args.user, args.password, args.token)
    test = ChatRoundtripTest(args.url, api_token, verbose=args.verbose)

    if args.cleanup:
        test.cleanup_all_test_artifacts()
        return

    success = test.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
