#!/usr/bin/env python3
"""
Round-trip integration test for document ingestion, RAG literature search,
and follow-up document analysis via the literature_analyse agent.

Uploads 3 markdown documents about blueberry harvesting machines, waits for
the aiembeddings pipeline to embed them into Milvus, then tests literature
search and document analysis through the chat endpoint.

Requires: running CKAN instance with ckanext-chat and ckanext-aiembeddings.

Usage:
    export CKAN_URL=http://localhost:80
    export CKAN_API_TOKEN=<your-sysadmin-api-token>

    python tests/test_literature_roundtrip.py
    python tests/test_literature_roundtrip.py --url http://localhost:80 --token <token> --verbose
    python tests/test_literature_roundtrip.py --cleanup
"""

import argparse
import json
import os
import re
import sys
import time
import uuid

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_chat_roundtrip import ChatRoundtripTest, _get_or_create_token

# ---------------------------------------------------------------------------

TEST_PREFIX = "lit-roundtrip-test"
UNIQUE_SUFFIX = uuid.uuid4().hex[:8]
ORG_NAME = f"{TEST_PREFIX}-org-{UNIQUE_SUFFIX}"
GROUP_NAME = f"{TEST_PREFIX}-grp-{UNIQUE_SUFFIX}"
GROUP_TITLE = "Heidelbeerernte Studien"

# ---------------------------------------------------------------------------
# Dummy documents — blueberry harvesting machines in southern regions
# ---------------------------------------------------------------------------

DOC_SUEDSCHWARZWALD = """\
# Maschinelle Heidelbeerernte im Südschwarzwald: Feldstudie 2024

## Zusammenfassung

Im Rahmen einer einjährigen Feldstudie wurde die maschinelle Ernte von Kulturheidelbeeren
(Vaccinium corymbosum) im Südschwarzwald evaluiert. Der Einsatz der Erntemaschine BerryMaster 3000
des Herstellers AgriTech Freiburg GmbH erzielte an den Versuchsstandorten bei Todtnau-Muggenbrunn
einen durchschnittlichen Ertrag von 2,8 Tonnen pro Hektar, verglichen mit 1,9 Tonnen pro Hektar
bei konventioneller Handernte.

## Versuchsstandort und Methodik

Die Studie wurde auf drei Versuchsfeldern in der Gemeinde Todtnau durchgeführt, auf einer Höhenlage
von 850 bis 1020 Metern über dem Meeresspiegel. Die Gesamtanbaufläche umfasste 4,7 Hektar mit
Heidelbeersträuchern der Sorten Bluecrop, Duke und Patriot.

Die Besonderheit des Standorts liegt in der Hanglage mit Neigungen von 8 bis 15 Grad. Die
BerryMaster 3000 wurde eigens mit einer Steilhang-Adaptation ausgestattet, bestehend aus einem
hydraulischen Niveauausgleich und breiteren Kettenfahrwerken. Die Anpassungskosten beliefen sich
auf zusätzliche 12.500 Euro.

## Ergebnisse

Der maschinelle Ernteprozess erstreckte sich über den Zeitraum Juli bis August 2024. Folgende
Kernmesswerte wurden dokumentiert:

- **Durchschnittlicher Ertrag**: 2,8 t/ha (maschinell) vs. 1,9 t/ha (Handernte)
- **Fruchtqualität**: 94% unbeschädigte Früchte bei maschineller Ernte
- **Erntegeschwindigkeit**: 0,35 ha/Stunde (BerryMaster 3000) vs. 0,08 ha/Stunde (Handernte)
- **Personalaufwand**: 2 Personen (Maschinenführer + Sortierung) vs. 12 Personen
- **Kraftstoffverbrauch**: 18,5 Liter Diesel pro Hektar

Die Feldleiterin Dr. Margarete Huber vom Landwirtschaftlichen Technologiezentrum Augustenberg
bewertet die Ergebnisse als vielversprechend: "Die Steilhang-Adaptation der BerryMaster 3000
ermöglicht erstmals einen wirtschaftlichen Maschineneinsatz in den typischen Hanglagen des
Südschwarzwalds."

## Wirtschaftliche Bewertung

Die Investitionskosten für die BerryMaster 3000 inklusive Steilhang-Adaptation betragen
197.500 Euro. Bei einer jährlichen Einsatzfläche von 15 Hektar und einer Kosteneinsparung
von 3.200 Euro pro Hektar gegenüber Handernte ergibt sich eine Amortisationszeit von
4,1 Jahren.

## Fazit

Die maschinelle Heidelbeerernte im Südschwarzwald ist trotz der anspruchsvollen Topografie
wirtschaftlich darstellbar. Die Steilhang-Adaptation ist ein entscheidender Erfolgsfaktor
für den Einsatz in Mittelgebirgslagen.
"""

DOC_SUEDSCHWEDEN = """\
# Pilotprojekt Maschinelle Heidelbeerernte in Südschweden 2025

## Zusammenfassung

Das Pilotprojekt "Nordic Blueberry Harvest" evaluierte den großflächigen Einsatz der
Erntemaschine NordicBerry X1 der Firma ScanHarvest AB (Göteborg) in der Region Småland,
Südschweden. Auf einer Versuchsfläche von 12,3 Hektar bei Växjö wurde ein durchschnittlicher
Ertrag von 3,4 Tonnen pro Hektar erzielt, was den bisherigen Regionsrekord für maschinelle
Heidelbeerernte darstellt.

## Projektrahmen und Standort

Das Projekt wurde von der Universität Linnaeus (Växjö) in Kooperation mit dem
schwedischen Landwirtschaftsamt (Jordbruksverket) und drei lokalen Anbaubetrieben
durchgeführt. Die Versuchsflächen befinden sich im Flachland südlich von Växjö auf
einer Höhe von 160 bis 190 Metern.

Der Erntezeitraum in Südschweden liegt mit Juni bis Juli etwa vier Wochen früher als
im Südschwarzwald, bedingt durch die längeren Tageslichtstunden und die Sortenauswahl
(überwiegend Duke und Draper, angepasst an skandinavische Bedingungen).

## Die NordicBerry X1

Die NordicBerry X1 ist eine GPS-gesteuerte Erntemaschine der neuesten Generation mit
folgenden technischen Merkmalen:

- **Arbeitsbreite**: 3,2 Meter (vs. 2,4 Meter bei BerryMaster 3000)
- **GPS-RTK-Steuerung**: Zentimetergenau, automatische Reihenführung
- **Elektrischer Antrieb**: Hybrid-System mit 48V-Batterie und Diesel-Generator
- **Schüttelsystem**: Frequenzgesteuertes Vibrationsverfahren (patentiert)

Der Projektleiter Professor Erik Lindqvist von der Universität Linnaeus betont:
"Die GPS-gesteuerte Reihenführung der NordicBerry X1 reduziert Überfahrschäden
auf ein Minimum und ermöglicht den Einsatz auch bei eingeschränkter Sicht."

## Ergebnisse

- **Durchschnittlicher Ertrag**: 3,4 t/ha (maschinell) vs. 2,1 t/ha (Handernte)
- **Fruchtqualität**: 91% unbeschädigte Früchte
- **Erntegeschwindigkeit**: 0,48 ha/Stunde
- **Personalaufwand**: 1 Person (autonomer Betrieb mit Fernüberwachung)
- **Energieverbrauch**: 12,3 Liter Diesel-Äquivalent pro Hektar

Bemerkenswert ist der Vergleich zwischen den drei Partnerbetrieben. Der Betrieb
Svensson Bär AB erzielte mit 3,8 t/ha den höchsten Einzelertrag, zurückzuführen
auf optimale Bewässerung und die Sorte Draper, die besonders maschinenerntegerecht
wächst.

## Ökonomische Analyse

Die Anschaffungskosten der NordicBerry X1 betragen 210.000 Euro. Bei einer
jährlichen Einsatzfläche von 25 Hektar und einer Kosteneinsparung von 2.800 Euro
pro Hektar amortisiert sich die Investition in 3,0 Jahren. Der niedrigere
Personalaufwand (1 vs. 2 Personen) ist dabei der zentrale Kostenvorteil.

## Fazit

Das Pilotprojekt belegt, dass GPS-gesteuerte Erntemaschinen im skandinavischen
Flachland Spitzenerträge bei gleichzeitig reduzierten Betriebskosten erzielen können.
Die NordicBerry X1 setzt neue Maßstäbe für die automatisierte Beerenernte.
"""

DOC_VERGLEICH = """\
# Technischer Vergleich: BerryMaster 3000 vs. NordicBerry X1

## Einleitung

Die maschinelle Ernte von Kulturheidelbeeren hat in den letzten Jahren bedeutende
Fortschritte gemacht. Zwei Maschinen dominieren den europäischen Markt: Die
BerryMaster 3000 von AgriTech Freiburg GmbH und die NordicBerry X1 von
ScanHarvest AB (Göteborg). Dieser Bericht vergleicht beide Systeme anhand der
Daten aus der Südschwarzwald-Feldstudie 2024 und dem Småland-Pilotprojekt 2025.

## Technische Gegenüberstellung

| Merkmal | BerryMaster 3000 | NordicBerry X1 |
|---------|-------------------|----------------|
| Hersteller | AgriTech Freiburg GmbH | ScanHarvest AB, Göteborg |
| Arbeitsbreite | 2,4 m | 3,2 m |
| Antrieb | Diesel (68 PS) | Hybrid Diesel-Elektrisch |
| Steuerung | Manuell + Assistenz | GPS-RTK autonom |
| Gewicht | 4.200 kg | 5.100 kg |
| Anschaffungspreis | 185.000 EUR | 210.000 EUR |
| Steilhang-Option | Ja (Aufpreis 12.500 EUR) | Nein |
| Max. Hangneigung | 18 Grad | 5 Grad |

## Ertragsvergleich

Die Ernteergebnisse unterscheiden sich signifikant, wobei standortbedingte Faktoren
berücksichtigt werden müssen:

- **BerryMaster 3000 (Südschwarzwald)**: 2,8 t/ha bei 94% Fruchtqualität
- **NordicBerry X1 (Südschweden)**: 3,4 t/ha bei 91% Fruchtqualität

Der Ertragsunterschied von 0,6 t/ha zugunsten der NordicBerry X1 ist primär auf
die günstigeren Flachlandbedingungen und die größere Arbeitsbreite zurückzuführen,
nicht auf eine generelle Überlegenheit der Maschine.

## Kosten-Nutzen-Analyse

Die Return-on-Investment-Berechnung zeigt folgendes Bild:

### BerryMaster 3000
- Investition: 197.500 EUR (inkl. Steilhang-Adaptation)
- Einsparung: 3.200 EUR/ha/Jahr
- Typische Einsatzfläche: 15 ha/Jahr
- **ROI: 4,1 Jahre**

### NordicBerry X1
- Investition: 210.000 EUR
- Einsparung: 2.800 EUR/ha/Jahr
- Typische Einsatzfläche: 25 ha/Jahr
- **ROI: 3,0 Jahre**

Der schnellere ROI der NordicBerry X1 resultiert aus der größeren Einsatzfläche,
die durch den höheren Automatisierungsgrad und die größere Arbeitsbreite möglich wird.

## Empfehlung

Die Wahl der Erntemaschine sollte standortabhängig erfolgen:

1. **Für Hanglagen und Mittelgebirge** (Neigung > 5 Grad): BerryMaster 3000 mit
   Steilhang-Adaptation — einzige Maschine mit zertifizierter Hanglagentauglichkeit.

2. **Für Flachlandanbau** (Neigung < 5 Grad): NordicBerry X1 — überlegene
   Automatisierung, höherer Durchsatz, schnellerer ROI bei großen Flächen.

3. **Für gemischte Betriebe**: Kombination beider Maschinen oder Prüfung der
   angekündigten BerryMaster 4000 (erwartet Q2 2026), die ebenfalls GPS-Steuerung
   bieten soll.

## Ausblick

Der europäische Markt für Heidelbeer-Erntemaschinen wächst jährlich um
geschätzte 15%. Mit der zunehmenden Automatisierung und dem Fachkräftemangel
in der Landwirtschaft wird die maschinelle Ernte in den kommenden Jahren zur
Standardmethode avancieren. Die Forschungsgruppe von Professor Lindqvist
plant bereits eine Folgestudie mit der NordicBerry X2 für 2026.
"""

DOCUMENTS = [
    ("heidelbeer_ernte_suedschwarzwald_2024.md",
     "Feldstudie Heidelbeerernte Südschwarzwald 2024",
     DOC_SUEDSCHWARZWALD,
     f"{TEST_PREFIX}-schwarzwald-{UNIQUE_SUFFIX}"),
    ("heidelbeer_ernte_suedschweden_2025.md",
     "Pilotprojekt Heidelbeerernte Südschweden 2025",
     DOC_SUEDSCHWEDEN,
     f"{TEST_PREFIX}-schweden-{UNIQUE_SUFFIX}"),
    ("heidelbeer_maschinentechnik_vergleich.md",
     "Technischer Vergleich Heidelbeer-Erntemaschinen",
     DOC_VERGLEICH,
     f"{TEST_PREFIX}-vergleich-{UNIQUE_SUFFIX}"),
]


class LiteratureRoundtripTest(ChatRoundtripTest):

    # -- override: parse SSE events from /chat/ask/stream properly --

    def _chat_upload(self, user_msg, file_bytes, filename,
                     content_type="text/csv", timeout=300):
        """Send file upload to /chat/ask/stream, parse SSE response."""
        url = f"{self.base_url}/chat/ask/stream"
        if self.verbose:
            print(f"\n  >>> [upload: {filename}] {user_msg[:120]}...")

        resp = requests.post(
            url,
            data={"text": user_msg},
            files={"upload": (filename, file_bytes, content_type)},
            headers={"Authorization": f"Bearer {self.api_token}"},
            timeout=timeout,
        )
        resp.raise_for_status()

        assistant_text = ""
        for block in resp.text.split("\n\n"):
            lines = block.strip().split("\n")
            event_type = None
            data_str = None
            for line in lines:
                if line.startswith("event: "):
                    event_type = line[7:].strip()
                elif line.startswith("data: "):
                    data_str = line[6:]

            if event_type == "status" and data_str and self.verbose:
                try:
                    msg = json.loads(data_str).get("message", "")
                    print(f"  [status] {msg}")
                except json.JSONDecodeError:
                    pass

            elif event_type == "done" and data_str:
                try:
                    payload = json.loads(data_str)
                    for message in reversed(payload.get("response", [])):
                        for part in message.get("parts", []):
                            if (part.get("part_kind") == "text"
                                    and part.get("content")):
                                assistant_text = part["content"]
                                break
                        if assistant_text:
                            break
                except json.JSONDecodeError:
                    pass

        if self.verbose and assistant_text:
            print(f"  <<< {assistant_text}")

        return assistant_text

    # -- test steps ------------------------------------------------------------

    def step_1_create_org(self):
        """Ask the agent to create an organization."""
        print("\n[Step 1] Create organization")

        self._chat(
            f"Create a new CKAN organization with name '{ORG_NAME}' "
            f"and title 'Literature Roundtrip Test Org'. "
            f"Tell me the organization id when done."
        )

        result = self._ckan_get("organization_show", {"id": ORG_NAME})
        if not result.get("success"):
            self._check("org exists", False,
                        f"API error: {result.get('error', {}).get('message', 'unknown')}")
            return None
        org_id = result["result"]["id"]
        self._check("org exists", result["result"]["name"] == ORG_NAME)
        return org_id

    def step_1b_create_group(self):
        """Ask the agent to create a CKAN group."""
        print("\n[Step 1b] Create group")

        self._chat(
            f"Create a new CKAN group with name '{GROUP_NAME}' "
            f"and title 'Heidelbeerernte Studien'. "
            f"Tell me the group id when done."
        )

        result = self._ckan_get("group_show", {"id": GROUP_NAME})
        if not result.get("success"):
            self._check("group exists", False,
                        f"API error: {result.get('error', {}).get('message', 'unknown')}")
            return False
        self._check("group exists", result["result"]["name"] == GROUP_NAME)
        return True

    def step_1c_create_datasets_in_group(self, org_id):
        """Create one dataset per document, each assigned to the group."""
        print("\n[Step 1c] Create datasets and assign to group")

        pkg_ids = {}
        for filename, title, content, ds_name in DOCUMENTS:
            self._chat(
                f"Create a dataset with name '{ds_name}', "
                f"title '{title}', "
                f"notes 'Test document for literature roundtrip', "
                f"in the organization with id '{org_id}', "
                f"and add it to the group '{GROUP_NAME}'. "
                f"Tell me the dataset id when done."
            )

            result = self._ckan_get("package_show", {"id": ds_name})
            if not result.get("success"):
                self._check(f"dataset {ds_name} exists", False,
                            f"API error: {result.get('error', {}).get('message', 'unknown')}")
                return None
            pkg_ids[ds_name] = result["result"]["id"]
            self._check(f"dataset {ds_name} exists",
                        result["result"]["name"] == ds_name)

        result = self._ckan_get("group_show",
                                {"id": GROUP_NAME, "include_datasets": True})
        if result.get("success"):
            grp_pkg_names = [p["name"] if isinstance(p, dict) else p
                             for p in result["result"].get("packages", [])]
            expected_names = [ds_name for _, _, _, ds_name in DOCUMENTS]
            assigned = all(n in grp_pkg_names for n in expected_names)
            self._check("all datasets assigned to group", assigned,
                        f"expected {expected_names}, got {grp_pkg_names}")

        return pkg_ids

    def step_2_upload_documents(self, pkg_ids):
        """Upload each markdown document to its own dataset."""
        print("\n[Step 2] Upload markdown documents (one per dataset)")

        all_ok = True
        for filename, title, content, ds_name in DOCUMENTS:
            pid = pkg_ids[ds_name]
            self._chat_upload(
                f"Upload the attached file as a new resource to dataset '{pid}'. "
                f"Resource name: '{filename}', format: 'markdown', "
                f"description: '{title}'. Tell me the resource id when done.",
                file_bytes=content.encode("utf-8"),
                filename=filename,
                content_type="text/markdown",
            )

            result = self._ckan_get("package_show", {"id": pid})
            if result.get("success"):
                resources = result["result"].get("resources", [])
                md_count = sum(
                    1 for r in resources
                    if r.get("format", "").lower() in ("markdown", "md")
                )
                ok = md_count >= 1
                self._check(f"resource uploaded to {ds_name}", ok,
                            f"{md_count} markdown resources in {len(resources)} total")
                if not ok:
                    all_ok = False
            else:
                self._check(f"resource uploaded to {ds_name}", False,
                            "could not fetch dataset")
                all_ok = False

        return all_ok

    def step_3_wait_for_embeddings(self, pkg_ids):
        """Poll until aiembeddings has produced .chunks + .embedding resources for all datasets."""
        print("\n[Step 3] Waiting for embedding pipeline...")

        max_attempts = 24
        poll_interval = 15
        expected_per_dataset = 3  # 1 md + 1 chunks + 1 embedding

        for attempt in range(1, max_attempts + 1):
            all_ready = True
            for ds_name, pid in pkg_ids.items():
                result = self._ckan_get("package_show", {"id": pid})
                if result.get("success"):
                    total = len(result["result"].get("resources", []))
                    if self.verbose:
                        formats = {}
                        for r in result["result"].get("resources", []):
                            fmt = r.get("format", "?").lower()
                            formats[fmt] = formats.get(fmt, 0) + 1
                        print(f"  Poll {attempt}/{max_attempts} [{ds_name}]: "
                              f"{total} resources — {formats}")
                    if total < expected_per_dataset:
                        all_ready = False
                else:
                    all_ready = False

            if all_ready:
                self._check("embedding pipeline complete", True,
                            f"all {len(pkg_ids)} datasets ready")
                return True

            time.sleep(poll_interval)

        for ds_name, pid in pkg_ids.items():
            result = self._ckan_get("package_show", {"id": pid})
            total = len(result["result"].get("resources", [])) if result.get("success") else 0
            self._check(f"embeddings for {ds_name}",
                        total >= expected_per_dataset,
                        f"{total}/{expected_per_dataset} resources")

        return False

    def step_4_literature_search(self):
        """Search for the uploaded documents via literature_search.

        Also checks that the agent used find_relevant_groups and selected
        the test group automatically (group-aware search without explicit mention).
        """
        print("\n[Step 4] Literature search via chat")
        self._reset_history()

        reply = self._chat(
            "Suche in der Literatur nach maschineller Heidelbeerernte. "
            "Welche Studien und Ergebnisse gibt es dazu?",
            timeout=300,
        )

        # Check content quality (clean reply)
        clean = re.sub(r"\[status\].*?\[/status\]", "", reply, flags=re.DOTALL).lower()

        found_heidelbeer = "heidelbeer" in clean
        found_machine = any(
            m in clean for m in ["berrymaster", "nordicberry", "erntemaschine"]
        )
        found_region = any(
            r in clean for r in ["schwarzwald", "schweden", "småland", "växjö", "todtnau"]
        )

        self._check(
            "search mentions blueberry harvest",
            found_heidelbeer,
            f"heidelbeer={'yes' if found_heidelbeer else 'no'}",
        )
        self._check(
            "search mentions machines or regions",
            found_machine or found_region,
            f"machine={'yes' if found_machine else 'no'}, "
            f"region={'yes' if found_region else 'no'}",
        )

        # Check that find_relevant_groups was called and picked the test group
        found_group_selected = GROUP_NAME in reply
        self._check(
            "auto group selection used test group",
            found_group_selected,
            f"group '{GROUP_NAME}' {'found' if found_group_selected else 'not found'} in status output",
        )

    def step_4b_literature_search_with_group(self):
        """Search with explicit group reference to test group-aware filtering."""
        print("\n[Step 4b] Literature search with group filter")
        self._reset_history()

        reply = self._chat(
            f"Suche in der Literatur in der Gruppe '{GROUP_TITLE}' nach maschineller "
            f"Heidelbeerernte. Welche Studien und Ergebnisse gibt es dazu?",
            timeout=300,
        )

        clean = re.sub(r"\[status\].*?\[/status\]", "", reply, flags=re.DOTALL).lower()

        found_heidelbeer = "heidelbeer" in clean
        found_machine = any(
            m in clean for m in ["berrymaster", "nordicberry", "erntemaschine"]
        )
        found_region = any(
            r in clean for r in ["schwarzwald", "schweden", "småland", "växjö", "todtnau"]
        )

        self._check(
            "group search mentions blueberry harvest",
            found_heidelbeer,
            f"heidelbeer={'yes' if found_heidelbeer else 'no'}",
        )
        self._check(
            "group search mentions machines or regions",
            found_machine or found_region,
            f"machine={'yes' if found_machine else 'no'}, "
            f"region={'yes' if found_region else 'no'}",
        )

    def step_5_follow_up_analysis(self):
        """Follow-up in same conversation triggering literature_analyse."""
        print("\n[Step 5] Follow-up document analysis (literature_analyse)")

        reply = self._chat(
            "Analysiere das Dokument über die Heidelbeerernte in Südschweden genauer. "
            "Welche konkreten Ernteerträge wurden mit der NordicBerry X1 erzielt, "
            "und welcher Betrieb hatte den höchsten Einzelertrag?",
            timeout=300,
        )

        clean = re.sub(r"\[status\].*?\[/status\]", "", reply, flags=re.DOTALL).lower()

        found_yield = "3,4" in clean or "3.4" in clean
        found_svensson = "svensson" in clean
        found_lindqvist = "lindqvist" in clean
        found_nordic = "nordicberry" in clean
        found_peak = "3,8" in clean or "3.8" in clean

        specific_facts = sum([
            found_yield, found_svensson, found_lindqvist,
            found_nordic, found_peak,
        ])

        self._check(
            "analysis contains specific document facts",
            specific_facts >= 2,
            f"yield={'yes' if found_yield else 'no'}, "
            f"svensson={'yes' if found_svensson else 'no'}, "
            f"lindqvist={'yes' if found_lindqvist else 'no'}, "
            f"nordicberry={'yes' if found_nordic else 'no'}, "
            f"peak_yield={'yes' if found_peak else 'no'}",
        )

    # -- cleanup ---------------------------------------------------------------

    def cleanup(self):
        """Remove test artifacts created by this run."""
        print("\n[Cleanup] Removing test artifacts...")

        for _, _, _, ds_name in DOCUMENTS:
            r = self._ckan_post("package_delete", {"id": ds_name})
            if r.get("success"):
                print(f"  Deleted dataset: {ds_name}")
                self._ckan_post("dataset_purge", {"id": ds_name})
            else:
                print(f"  Dataset not found or already deleted: {ds_name}")

        r = self._ckan_post("group_delete", {"id": GROUP_NAME})
        if r.get("success"):
            print(f"  Deleted group: {GROUP_NAME}")
            self._ckan_post("group_purge", {"id": GROUP_NAME})
        else:
            print(f"  Group not found or already deleted: {GROUP_NAME}")

        r = self._ckan_post("organization_delete", {"id": ORG_NAME})
        if r.get("success"):
            print(f"  Deleted organization: {ORG_NAME}")
            self._ckan_post("organization_purge", {"id": ORG_NAME})
        else:
            print(f"  Org not found or already deleted: {ORG_NAME}")

    def cleanup_all_test_artifacts(self):
        """Find and remove ALL lit-roundtrip-test artifacts from any prior run."""
        print("\n[Cleanup] Searching for all literature test artifacts...")

        r = self._ckan_get("package_search",
                           {"q": f"name:{TEST_PREFIX}*", "rows": 100})
        if r.get("success"):
            for pkg in r["result"].get("results", []):
                if pkg["name"].startswith(TEST_PREFIX):
                    self._ckan_post("package_delete", {"id": pkg["id"]})
                    self._ckan_post("dataset_purge", {"id": pkg["id"]})
                    print(f"  Purged dataset: {pkg['name']}")

        r = self._ckan_get("group_list", {"all_fields": True})
        if r.get("success"):
            for grp in r["result"]:
                name = grp["name"] if isinstance(grp, dict) else grp
                if isinstance(name, str) and name.startswith(TEST_PREFIX):
                    self._ckan_post("group_delete", {"id": name})
                    self._ckan_post("group_purge", {"id": name})
                    print(f"  Purged group: {name}")

        r = self._ckan_get("organization_list", {"all_fields": True})
        if r.get("success"):
            for org in r["result"]:
                name = org["name"] if isinstance(org, dict) else org
                if isinstance(name, str) and name.startswith(TEST_PREFIX):
                    self._ckan_post("organization_delete", {"id": name})
                    self._ckan_post("organization_purge", {"id": name})
                    print(f"  Purged organization: {name}")

    # -- runner ----------------------------------------------------------------

    def run(self):
        print(f"{'=' * 60}")
        print("Literature Round-Trip Test")
        print(f"{'=' * 60}")
        ds_names = [ds for _, _, _, ds in DOCUMENTS]
        print(f"  Endpoint:  {self.chat_url}")
        print(f"  Org:       {ORG_NAME}")
        print(f"  Group:     {GROUP_NAME}")
        print(f"  Datasets:  {', '.join(ds_names)}")
        print(f"  Documents: {len(DOCUMENTS)}")
        print(f"{'=' * 60}")

        try:
            org_id = self.step_1_create_org()
            if not org_id:
                print("\n  ABORT: org creation failed, cannot continue")
                return False

            group_ok = self.step_1b_create_group()
            if not group_ok:
                print("\n  ABORT: group creation failed, cannot continue")
                return False

            pkg_ids = self.step_1c_create_datasets_in_group(org_id)
            if not pkg_ids:
                print("\n  ABORT: dataset creation failed, cannot continue")
                return False

            uploaded = self.step_2_upload_documents(pkg_ids)
            if not uploaded:
                print("\n  ABORT: document upload failed, cannot continue")
                return False

            embeddings_ready = self.step_3_wait_for_embeddings(pkg_ids)
            if not embeddings_ready:
                print("\n  ABORT: embedding pipeline did not finish, cannot continue")
                return False

            self.step_4_literature_search()

            self.step_4b_literature_search_with_group()

            self.step_5_follow_up_analysis()

        finally:
            self.cleanup()

        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        failed = total - passed

        print(f"\n{'=' * 60}")
        print(f"Results: {passed}/{total} passed", end="")
        if failed:
            print(f", \033[91m{failed} FAILED\033[0m")
            for name, p, detail in self.results:
                if not p:
                    print(f"  \033[91m✗ {name}: {detail}\033[0m")
        else:
            print(" \033[92m— ALL PASSED\033[0m")
        print(f"{'=' * 60}")

        return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Literature round-trip test (RAG search + document analysis)")
    parser.add_argument("--url",
                        default=os.environ.get("CKAN_URL", "http://localhost:80"),
                        help="CKAN base URL (default: $CKAN_URL or http://localhost:80)")
    parser.add_argument("--token",
                        default=os.environ.get("CKAN_API_TOKEN", ""),
                        help="CKAN API token (default: $CKAN_API_TOKEN)")
    parser.add_argument("--user", "-u",
                        default=os.environ.get("CKAN_USER", ""),
                        help="CKAN username (default: $CKAN_USER)")
    parser.add_argument("--password", "-p",
                        default=os.environ.get("CKAN_PASSWORD", ""),
                        help="CKAN password (default: $CKAN_PASSWORD)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print chat messages and polling details")
    parser.add_argument("--cleanup", action="store_true",
                        help="Only cleanup test artifacts from previous runs")
    args = parser.parse_args()

    if not args.token and not args.user:
        print("ERROR: Provide --token or --user/--password")
        sys.exit(1)

    api_token = _get_or_create_token(args.url, args.user, args.password, args.token)
    test = LiteratureRoundtripTest(args.url, api_token, verbose=args.verbose)

    if args.cleanup:
        test.cleanup_all_test_artifacts()
        return

    success = test.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
