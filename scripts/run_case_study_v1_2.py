#!/usr/bin/env python3
"""
Chapter 4 case study (Section 4.5): the Russia-Ukraine war article run for Mercedes-Benz Group AG through the
pipeline of the Chapter 5 evaluation, with the product search and alternative sourcing agents enabled.

    python -m scripts.run_case_study_v1_2 article     fetch the archived input article and save it with its hash
    python -m scripts.run_case_study_v1_2 preflight   environment, pipeline files, graph size and article; no API call
    python -m scripts.run_case_study_v1_2 launch      start `run` as a detached process, log in logs/
    python -m scripts.run_case_study_v1_2 run         one crew run; repeated only after an API or database error
    python -m scripts.run_case_study_v1_2 figures     thesis figures drawn from the saved outputs of the run (final/)

Input. The article is the Encyclopaedia Britannica entry "Russia-Ukraine War" at the URL in main.py
(https://www.britannica.com/event/2022-Russian-invasion-of-Ukraine), taken from the Internet Archive capture of
12 June 2025, 13:31:24 UTC. The capture carries the news item shown in the thesis figure of the article. The text
passed to the crew is the entry's title, section headings and body paragraphs, in page order.

Configuration. As in scripts/run_evaluation.py: the .env of this repository is loaded with override=True before any
pipeline module is imported, OPENAI_MODEL_NAME is gpt-4o, no temperature is sent by the crew, and the graph must
hold 6,318 nodes and 22,862 relationships (dataset/supplychainKG_clean.csv). DISABLE_PRODUCT_AGENTS is 0, so the
crew runs its six tasks (Disruption Monitor, KG Query, Product Search, Risk Manager, CSCO, Alternative Sourcing).
The visualisation agent is not a task of the crew; after the crew, agents/visualization_agent.py is executed on the
crew's outputs (it calls the NetworkX visualisation tool and no model).

Every API call is logged with the served model; the attempt stops if a served model differs from the snapshot
expected for the requested model. Every tool call is logged with its arguments and output. An API or graph-database
error stops the attempt, which is recorded with its calls and cost, and the run is started again after a pause.

Output: evaluation/results/case_study_v1_2/
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)
for _var in ("MODEL", "MODEL_NAME", "OPENAI_MODEL_NAME"):
    os.environ.pop(_var, None)
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
os.environ["DISABLE_PRODUCT_AGENTS"] = "0"
os.environ["CREWAI_STORAGE_DIR"] = "supply_chain_case_study_v1_2"
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

RUN_ID = "CASE_RU_UA_MERC"
COMPANY = "Mercedes-Benz Group AG"
OUT = ROOT / "evaluation" / "results" / "case_study_v1_2"
LOG_DIR = ROOT / "logs"
LOCK = OUT / "RUNNING.lock"
TEMP_KG = ROOT / "output" / "temp" / "kg_payloads"

SOURCE_URL = "https://www.britannica.com/event/2022-Russian-invasion-of-Ukraine"
CAPTURE = "20250612133124"
ARCHIVE_URL = f"https://web.archive.org/web/{CAPTURE}id_/{SOURCE_URL}"

PIPELINE_PATHS = ("crew.py", "main.py", "agents", "tools", "config", "offline", "dataset",
                  "evaluation/ground_truth_generator.py", ":(exclude,glob)**/*.md")
ENVIRONMENT = {"crewai": "1.9.3", "openai": "1.83.0", "neo4j": "6.1.0"}  # requirements-lock.txt
GRAPH_SIZE = {"nodes": 6318, "relationships": 22862}
SNAPSHOTS = {  # ENVIRONMENT.md
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-2024-08-06": "gpt-4o-2024-08-06",
    "text-embedding-3-small": "text-embedding-3-small",
}
# USD per million tokens: input, cached input, output. gpt-4o-2024-08-06 as in scripts/run_evaluation.py;
# text-embedding-3-small from platform.openai.com/docs/pricing.
PRICES = {"gpt-4o-2024-08-06": (2.50, 1.25, 10.00), "text-embedding-3-small": (0.02, 0.02, 0.0)}
MAX_USD = 3.00
MAX_ATTEMPTS = 5
BACKOFF = (60, 120, 240, 480)
EXIT_DONE, EXIT_TRANSIENT, EXIT_STOP = 0, 75, 77

TASK_KEYS = {
    "analyze_disruptions_task": "1_disruption_monitor",
    "kg_query_task": "2_kg_query",
    "product_search_task": "3_product_search",
    "risk_manager_task": "4_risk_manager",
    "decision_making_task": "5_csco",
    "sourcing_task": "6_alternative_sourcing",
}


# ---------------------------------------------------------------------------------------------------- article

class _ArticleText(HTMLParser):
    """Title (h1), section headings (h2.h1, h2.h2) and body paragraphs (p.topic-paragraph), in page order."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self._buf, self._tag, self._depth = [], [], None, 0

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if self._tag:
            if tag == self._tag:
                self._depth += 1
            return
        if (tag == "h1") or (tag == "h2" and cls in ("h1", "h2")) or (tag == "p" and "topic-paragraph" in cls.split()):
            self._tag, self._depth, self._buf = tag, 1, []

    def handle_endtag(self, tag):
        if self._tag and tag == self._tag:
            self._depth -= 1
            if self._depth == 0:
                text = " ".join("".join(self._buf).split())
                if text:
                    self.blocks.append(text)
                self._tag = None

    def handle_data(self, data):
        if self._tag:
            self._buf.append(data)


def cmd_article(_args):
    target = OUT / "input"
    target.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "Mozilla/5.0 (case-study reproduction)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read()
    html = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    (target / f"britannica_{CAPTURE}.html.gz").write_bytes(gzip.compress(html))
    parser = _ArticleText()
    parser.feed(html.decode("utf-8", errors="replace"))
    text = "\n\n".join(parser.blocks)
    (target / "article.txt").write_text(text, encoding="utf-8")
    page = html.decode("utf-8", errors="replace")
    meta = {
        "source_url": SOURCE_URL, "archive_url": ARCHIVE_URL, "capture_utc": CAPTURE,
        "retrieved": datetime.now().isoformat(timespec="seconds"),
        "html_sha256": hashlib.sha256(html).hexdigest(),
        "article_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "blocks": len(parser.blocks), "words": len(text.split()), "characters": len(text),
        "first_block": parser.blocks[0] if parser.blocks else None,
        "news_item_in_capture": "Russian drone and missile attack, one of the biggest in the war" in page,
    }
    (target / "article_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def load_article():
    path = OUT / "input" / "article.txt"
    if not path.exists():
        raise SystemExit("No article; run the `article` command first.")
    meta = json.loads((OUT / "input" / "article_meta.json").read_text())
    text = path.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != meta["article_sha256"]:
        raise SystemExit("article.txt does not match article_meta.json")
    return text, meta


# ---------------------------------------------------------------------------------------------------- checks

def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def provenance():
    import crewai
    import neo4j
    import openai
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "pipeline_files_modified": git("status", "--porcelain", "--", *PIPELINE_PATHS).splitlines(),
        "python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
        "crewai": crewai.__version__, "openai": openai.__version__, "neo4j": neo4j.__version__,
    }


def check_provenance():
    prov = provenance()
    wrong = {k: prov[k] for k, v in ENVIRONMENT.items() if prov[k] != v}
    if wrong:
        raise SystemExit(f"Environment differs from requirements-lock.txt: {wrong}")
    if prov["pipeline_files_modified"]:
        raise SystemExit("Pipeline files are modified:\n" + "\n".join(prov["pipeline_files_modified"]))
    return prov


def graph_state():
    """Counts read with the driver directly, so that no pipeline module (and no embedding call) is involved."""
    from neo4j import GraphDatabase
    uri, user, pwd = os.environ["NEO4J_URI"], os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ["NEO4J_PASSWORD"]
    db = os.environ.get("NEO4J_DATABASE", "neo4j")
    with GraphDatabase.driver(uri, auth=(user, pwd)) as driver, driver.session(database=db) as s:
        out = {
            "nodes": s.run("MATCH (n) RETURN count(n) AS c").single()["c"],
            "relationships": s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"],
            "company_nodes": s.run("MATCH (c:Company) RETURN count(c) AS c").single()["c"],
            "companies_to_embed": s.run(
                "MATCH (c:Company) WHERE c.embedding IS NULL AND c.unstructured_data IS NOT NULL "
                "RETURN count(c) AS c").single()["c"],
            "monitored_company_present": s.run(
                "MATCH (c:Company {name: $n}) RETURN count(c) AS c", n=COMPANY).single()["c"] == 1,
            "vector_index": [r["name"] for r in s.run("SHOW INDEXES YIELD name, type WHERE type = 'VECTOR' RETURN name")],
        }
    return out


def cmd_preflight(_args):
    prov = check_provenance()
    g = graph_state()
    _, meta = load_article()
    ok = (g["nodes"], g["relationships"]) == (GRAPH_SIZE["nodes"], GRAPH_SIZE["relationships"])
    report = {"provenance": prov, "graph": g, "graph_size_as_expected": ok, "article": meta,
              "api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
              "neo4j_credentials_present": all(os.environ.get(v) for v in ("NEO4J_URI", "NEO4J_PASSWORD"))}
    print(json.dumps(report, indent=1))
    if not ok:
        raise SystemExit("Graph size differs from the cleaned network.")


# ---------------------------------------------------------------------------------------------------- logging

class SnapshotMismatch(BaseException):
    """Raised from the call log; a BaseException so that no agent or tool handler can absorb it."""


class BudgetStop(BaseException):
    pass


CALLS: list = []
TOOL_CALLS: list = []
API_ERRORS: list = []
GRAPH_ERRORS: list = []


def usd(calls):
    total, unpriced = 0.0, set()
    for c in calls:
        price = PRICES.get(c.get("served_model"))
        if not price:
            unpriced.add(c.get("served_model"))
            continue
        pin, cached, pout = c.get("prompt_tokens") or 0, c.get("cached_tokens") or 0, c.get("completion_tokens") or 0
        total += ((pin - cached) * price[0] + cached * price[1] + pout * price[2]) / 1_000_000
    return round(total, 6), sorted(m for m in unpriced if m)


def cost(calls):
    total, unpriced = usd(calls)
    return {"calls": len(calls),
            "by_served_model": {m: sum(1 for c in calls if c.get("served_model") == m)
                                for m in sorted({c.get("served_model") or "none" for c in calls})},
            "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
            "cached_tokens": sum(c.get("cached_tokens") or 0 for c in calls),
            "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
            "usd": total, "unpriced_models": unpriced}


def install_call_log(attempt_dir, caller):
    calls_file = attempt_dir / "calls.jsonl"

    def record(api, kwargs, response):
        usage = getattr(response, "usage", None)
        prompt = (getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)) if usage else None
        completion = (getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)) if usage else None
        pdet = (getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_tokens_details", None)) if usage else None
        choices = getattr(response, "choices", None)
        finish = getattr(choices[0], "finish_reason", None) if choices else None
        served = getattr(response, "model", None)
        messages = kwargs.get("messages") or kwargs.get("input")
        entry = {
            "n": len(CALLS) + 1, "api": api,
            "caller": "embeddings" if api == "embeddings" else caller(messages, kwargs.get("instructions")),
            "requested_model": kwargs.get("model"),
            "temperature": kwargs.get("temperature", "not sent (API default)"),
            "served_model": served, "prompt_tokens": prompt,
            "cached_tokens": (getattr(pdet, "cached_tokens", 0) or 0) if pdet else 0,
            "completion_tokens": completion, "finish_reason": finish,
            "truncated": finish in ("length", "max_output_tokens"),
            "tool_calls": [t.function.name for t in (getattr(choices[0].message, "tool_calls", None) or [])]
            if choices and getattr(choices[0], "message", None) else [],
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        CALLS.append(entry)
        with calls_file.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        expected = SNAPSHOTS.get(kwargs.get("model"))
        if not served or served != expected:
            raise SnapshotMismatch(f"served {served} for {kwargs.get('model')}; expected {expected}")
        spent, _ = usd(CALLS)
        if spent > MAX_USD:
            raise BudgetStop(f"spent ${spent} > ${MAX_USD}")

    def wrap(cls, api, is_async):
        original = cls.create

        def failed(kwargs, exc):
            API_ERRORS.append({"api": api, "model": kwargs.get("model"), "time": datetime.now().isoformat(timespec="seconds"),
                               "error": f"{type(exc).__module__}.{type(exc).__name__}: {exc}"[:500]})

        if is_async:
            async def create(self, *args, **kwargs):
                try:
                    response = await original(self, *args, **kwargs)
                except Exception as exc:
                    failed(kwargs, exc)
                    raise
                record(api, kwargs, response)
                return response
        else:
            def create(self, *args, **kwargs):
                try:
                    response = original(self, *args, **kwargs)
                except Exception as exc:
                    failed(kwargs, exc)
                    raise
                record(api, kwargs, response)
                return response
        cls.create = create

    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.embeddings import AsyncEmbeddings, Embeddings
    from openai.resources.responses import AsyncResponses, Responses
    wrap(Completions, "chat.completions", False)
    wrap(AsyncCompletions, "chat.completions", True)
    wrap(Responses, "responses", False)
    wrap(AsyncResponses, "responses", True)
    wrap(Embeddings, "embeddings", False)
    wrap(AsyncEmbeddings, "embeddings", True)


def install_tool_log(attempt_dir):
    from tools.crewai_tool_wrapper import CrewAIToolWrapper
    original = CrewAIToolWrapper._run
    tools_file = attempt_dir / "tool_calls.jsonl"

    def _run(self, *args, **kwargs):
        started = datetime.now().isoformat(timespec="seconds")
        try:
            out = original(self, *args, **kwargs)
            error = None
        except Exception as exc:
            out, error = None, f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry = {"n": len(TOOL_CALLS) + 1, "tool": self.name, "args": list(args), "kwargs": kwargs,
                     "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
                     "error": error, "output": out}
            TOOL_CALLS.append(entry)
            with tools_file.open("a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        return out

    CrewAIToolWrapper._run = _run


def install_graph_watch():
    from langchain_community.graphs import Neo4jGraph
    original = Neo4jGraph.query

    def query(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:
            GRAPH_ERRORS.append(f"{type(exc).__module__}.{type(exc).__name__}: {exc}"[:500])
            raise
    Neo4jGraph.query = query


def classify(exc):
    if isinstance(exc, (SnapshotMismatch, BudgetStop, SystemExit)):
        return "stop"
    chain, seen, e = [], set(), exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append(e)
        e = e.__cause__ or e.__context__
    text = " ".join(f"{type(x).__module__}.{type(x).__name__}: {x}" for x in chain)
    if "insufficient_quota" in text:
        return "stop"
    import httpx
    import openai
    from neo4j import exceptions as neo
    for x in chain:
        if isinstance(x, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
            return "stop"
        if isinstance(x, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                          openai.InternalServerError, httpx.TransportError, ConnectionError, TimeoutError,
                          neo.ServiceUnavailable, neo.SessionExpired, neo.TransientError)):
            return "transient"
    markers = ("Error code: 429", "Error code: 500", "Error code: 502", "Error code: 503", "Error code: 504",
               "Connection error", "Request timed out", "ServiceUnavailable", "SessionExpired")
    return "transient" if any(m in text for m in markers) else None


# ---------------------------------------------------------------------------------------------------- one attempt

def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")


def cmd_one(args):
    attempt = args.attempt
    attempt_dir = OUT / "attempts" / f"attempt_{attempt}"
    if attempt_dir.exists():
        shutil.rmtree(attempt_dir)
    attempt_dir.mkdir(parents=True)
    text, article_meta = load_article()
    prov = check_provenance()
    g = graph_state()
    if (g["nodes"], g["relationships"]) != (GRAPH_SIZE["nodes"], GRAPH_SIZE["relationships"]):
        raise SystemExit(f"Graph has {g['nodes']} nodes and {g['relationships']} relationships; expected {GRAPH_SIZE}")

    import scripts.pipeline_runner as base
    install_call_log(attempt_dir, base.caller)
    install_graph_watch()
    install_tool_log(attempt_dir)

    started = datetime.now()
    record = {"run_id": RUN_ID, "attempt": attempt, "monitored_company": COMPANY, "model": "gpt-4o",
              "expected_snapshot": SNAPSHOTS["gpt-4o"], "started": started.isoformat(timespec="seconds"),
              "provenance": prov, "graph_before": g, "article": article_meta,
              "disable_product_agents": os.environ["DISABLE_PRODUCT_AGENTS"]}
    tasks_out = {}
    try:
        for stale in TEMP_KG.glob(f"{RUN_ID}_*"):
            stale.unlink()
        from crew import SupplyChainCrew
        crew_obj = SupplyChainCrew(company_name=COMPANY)
        crew = crew_obj.crew()
        record["tasks"] = [t.name for t in crew.tasks]
        record["agents"] = [a.role for a in crew.agents]
        t0 = time.time()
        crew.kickoff(inputs={"extracted_text": text, "company_name": COMPANY, "scenario_id": RUN_ID})
        record["crew_runtime_seconds"] = round(time.time() - t0, 1)
        if GRAPH_ERRORS:
            raise ConnectionError("graph query failed during the run: " + GRAPH_ERRORS[0])
        for task in crew.tasks:
            key = TASK_KEYS.get(task.name, task.name)
            raw = getattr(task.output, "raw", None) if task.output is not None else None
            parsed = base.extract_dict(task.output) if task.output is not None else None
            tasks_out[key] = {"task": task.name, "agent": task.agent.role if task.agent else None,
                              "raw": raw, "parsed_json": parsed}
            (attempt_dir / "agent_outputs").mkdir(exist_ok=True)
            (attempt_dir / "agent_outputs" / f"{key}.txt").write_text(str(raw), encoding="utf-8")
            if parsed is not None:
                write_json(attempt_dir / "agent_outputs" / f"{key}.json", parsed)
    except KeyboardInterrupt:
        raise
    except BaseException as exc:  # noqa: BLE001 - every outcome is recorded
        kind = classify(exc)
        record.update({"status": "stopped", "kind": kind or "pipeline_error", "error": f"{type(exc).__name__}: {exc}"[:3000],
                       "traceback": traceback.format_exc(), "api_errors": API_ERRORS, "graph_errors": GRAPH_ERRORS,
                       "cost": cost(CALLS), "finished": datetime.now().isoformat(timespec="seconds")})
        write_json(attempt_dir / "attempt.json", record)
        print(f"ATTEMPT {attempt} STOPPED ({record['kind']}): {exc}", flush=True)
        sys.exit(EXIT_TRANSIENT if kind == "transient" else EXIT_STOP)

    # Tool outputs saved by the pipeline on disk.
    tool_dir = attempt_dir / "tool_outputs"
    tool_dir.mkdir(exist_ok=True)
    kg_payload = {}
    meta_path = TEMP_KG / f"{RUN_ID}_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        shutil.copy(meta_path, tool_dir / "kg_meta.json")
        if meta.get("kg_payload_path") and Path(meta["kg_payload_path"]).exists():
            shutil.copy(meta["kg_payload_path"], tool_dir / "kg_payload.json")
            kg_payload = json.loads(Path(meta["kg_payload_path"]).read_text())
    risk_path = TEMP_KG / f"{RUN_ID}_risk.json"
    if risk_path.exists():
        shutil.copy(risk_path, tool_dir / "risk_assessment.json")

    # Visualisation agent on the crew's outputs (no model call).
    viz = {"executed": False}
    try:
        from agents.visualization_agent import VisualizationAgent
        cfg = dict(crew_obj.agents_config["visualization_agent"])
        agent = VisualizationAgent(config=cfg, tools=[], verbose=False)
        viz_dir = attempt_dir / "visualisation"
        viz_dir.mkdir(exist_ok=True)
        product = tasks_out.get("3_product_search", {})
        here = os.getcwd()
        os.chdir(viz_dir)
        try:
            result = agent.execute({
                "product_details": product.get("parsed_json") or product.get("raw") or {},
                "kg_results": kg_payload,
                "disruption_analysis": tasks_out.get("1_disruption_monitor", {}).get("parsed_json") or {},
                "company_name": COMPANY,
            })
        finally:
            os.chdir(here)
        viz = {"executed": True, "product_map_source": "product_details.chain_map"
               if isinstance(product.get("parsed_json"), dict) and product["parsed_json"].get("chain_map")
               else "kg_results (fallback in VisualizationAgent.execute)",
               **{k: v for k, v in (result or {}).items() if k != "visualization_html"}}
    except Exception as exc:  # noqa: BLE001
        viz = {"executed": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    write_json(attempt_dir / "visualisation" / "visualisation_result.json", viz)

    record.update({
        "status": "completed", "finished": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round((datetime.now() - started).total_seconds(), 1),
        "calls": CALLS, "cost": cost(CALLS), "served_models": sorted({c["served_model"] for c in CALLS}),
        "truncated_calls": [c for c in CALLS if c.get("truncated")],
        "api_errors_within_run": API_ERRORS, "tool_calls": [{k: v for k, v in t.items() if k != "output"} for t in TOOL_CALLS],
        "kg_chain_counts": {f"tier_{t}": len(kg_payload.get(f"tier_{t}") or []) for t in range(1, 5)},
        "visualisation": viz, "agent_outputs": tasks_out,
    })
    write_json(attempt_dir / "attempt.json", record)
    print(f"ATTEMPT {attempt} COMPLETED: {record['runtime_seconds']}s, ${record['cost']['usd']}, "
          f"served {record['served_models']}, chains {record['kg_chain_counts']}", flush=True)
    sys.exit(EXIT_DONE)


# ---------------------------------------------------------------------------------------------------- orchestration

def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def cmd_run(_args):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "run.json").exists():
        raise SystemExit(f"{OUT / 'run.json'} exists; the case study has been run.")
    if LOCK.exists():
        pid = int(LOCK.read_text().split()[0] or 0)
        if _alive(pid):
            raise SystemExit(f"A run with id {RUN_ID} is in progress (pid {pid}).")
    LOCK.write_text(f"{os.getpid()} {datetime.now().isoformat(timespec='seconds')}\n")
    attempts = []
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"===== attempt {attempt} {datetime.now().isoformat(timespec='seconds')} =====", flush=True)
            code = subprocess.run([sys.executable, "-m", "scripts.run_case_study_v1_2", "one", "--attempt", str(attempt)],
                                  cwd=ROOT).returncode
            rec_path = OUT / "attempts" / f"attempt_{attempt}" / "attempt.json"
            rec = json.loads(rec_path.read_text()) if rec_path.exists() else {}
            attempts.append({"attempt": attempt, "exit_code": code, "status": rec.get("status"), "kind": rec.get("kind"),
                             "error": rec.get("error"), "cost": rec.get("cost")})
            if code == EXIT_DONE and rec.get("status") == "completed":
                final = OUT / "final"
                if final.exists():
                    shutil.rmtree(final)
                shutil.copytree(rec_path.parent, final)
                summary = {k: rec[k] for k in ("run_id", "monitored_company", "model", "expected_snapshot", "served_models",
                                               "started", "finished", "runtime_seconds", "crew_runtime_seconds", "cost",
                                               "kg_chain_counts", "visualisation", "tasks", "agents", "provenance",
                                               "graph_before", "article") if k in rec}
                summary["attempts"] = attempts
                summary["total_usd_all_attempts"] = round(sum((a.get("cost") or {}).get("usd", 0) for a in attempts), 6)
                summary["final_attempt_folder"] = f"attempts/attempt_{attempt} (copied to final/)"
                write_json(OUT / "run.json", summary)
                print(f"RUN COMPLETED after {attempt} attempt(s); total ${summary['total_usd_all_attempts']}", flush=True)
                return
            if code == EXIT_TRANSIENT and attempt < MAX_ATTEMPTS:
                wait = BACKOFF[min(attempt - 1, len(BACKOFF) - 1)]
                print(f"API or database error; retry in {wait}s", flush=True)
                time.sleep(wait)
                continue
            print(f"RUN STOPPED at attempt {attempt} (exit {code}); see {rec_path}", flush=True)
            write_json(OUT / "stopped.json", {"attempts": attempts})
            return
    finally:
        LOCK.unlink(missing_ok=True)


def cmd_launch(_args):
    LOG_DIR.mkdir(exist_ok=True)
    if LOCK.exists() and _alive(int(LOCK.read_text().split()[0] or 0)):
        raise SystemExit(f"A run with id {RUN_ID} is in progress.")
    log = LOG_DIR / f"case_study_v1_2_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    with log.open("a") as f:
        proc = subprocess.Popen([sys.executable, "-m", "scripts.run_case_study_v1_2", "run"], cwd=ROOT, stdout=f,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, start_new_session=True)
    print(f"started pid {proc.pid}; log {log}")


# ---------------------------------------------------------------------------------------------------- figures

MONO = "DejaVu Sans Mono"


def _strip_fences(text):
    lines = [ln for ln in text.strip().splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines)


def _wrap(text, width):
    import textwrap
    out = []
    for line in text.splitlines():
        if len(line) <= width:
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        out += textwrap.wrap(line, width=width, subsequent_indent=" " * (indent + 2), break_long_words=True,
                             replace_whitespace=False, drop_whitespace=True) or [""]
    return out


def _text_figure(path, text, columns=1, fontsize=6.4):
    import matplotlib.pyplot as plt
    from thesis_figures import style
    style.apply()
    width_in = style.TEXT_WIDTH
    char_w = 0.6 * fontsize / 72
    chars = int((width_in / columns - 0.12) / char_w)
    lines = _wrap(text, chars)
    per_col = -(-len(lines) // columns)
    fig = plt.figure(figsize=(width_in, 30))
    texts = [fig.text(0, 0, "\n".join(lines[c * per_col:(c + 1) * per_col]), family=MONO, fontsize=fontsize,
                      va="top", ha="left", linespacing=1.18, color=style.COLOURS["text"],
                      transform=fig.dpi_scale_trans) for c in range(columns)]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    height_in = max(t.get_window_extent(renderer).height for t in texts) / fig.dpi + 0.16
    fig.set_size_inches(width_in, height_in)
    for c, t in enumerate(texts):
        t.set_position((0.08 + c * width_in / columns, height_in - 0.08))
    fig.patches.append(plt.Rectangle((0.002, 0.002), 0.996, 0.996, transform=fig.transFigure, fill=False,
                                     edgecolor="#BBBBBB", linewidth=0.6))
    fig.savefig(path, bbox_inches=None)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches=None)
    plt.close(fig)


def cmd_figures(_args):
    import matplotlib.pyplot as plt
    from thesis_figures import style
    import scripts.pipeline_runner as base
    final = OUT / "final"
    figs = OUT / "figures"
    figs.mkdir(exist_ok=True)
    ao = final / "agent_outputs"

    dm = json.loads((ao / "1_disruption_monitor.json").read_text())
    _text_figure(figs / "casestudy-dm-agent.pdf", json.dumps(dm, indent=2, ensure_ascii=False))

    kg_meta = base.extract_dict((ao / "2_kg_query.txt").read_text())
    for key in ("kg_payload_path", "message"):
        if isinstance(kg_meta.get(key), str):
            kg_meta[key] = kg_meta[key].replace(str(ROOT) + "/", "")
    payload = json.loads((final / "tool_outputs" / "kg_payload.json").read_text())
    tier2 = [" -> ".join(f"{n['company']} ({n['country']})" for n in chain) for chain in payload.get("tier_2") or []]
    kg_text = ("KG Query Agent answer (metadata returned by build_and_save_kg)\n"
               + json.dumps(kg_meta, indent=1, ensure_ascii=False)
               + f"\n\nTier-2 paths in the saved payload ({len(tier2)})\n" + "\n".join(tier2))
    _text_figure(figs / "casestudy-kg-agent.pdf", kg_text)

    _text_figure(figs / "casestudy-product-search.pdf", _strip_fences((ao / "3_product_search.txt").read_text()))

    viz = json.loads((final / "visualisation" / "visualisation_result.json").read_text())
    png = final / "visualisation" / "visualizations" / "network_plots" / Path(viz["visualization_file_path"]).with_suffix(".png").name
    img = plt.imread(png)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(style.TEXT_WIDTH, style.TEXT_WIDTH * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, interpolation="lanczos")
    ax.set_axis_off()
    fig.savefig(figs / "casestudy-network-visualizer.pdf", dpi=300, bbox_inches=None)
    plt.close(fig)

    risk = json.loads((final / "tool_outputs" / "risk_assessment.json").read_text())
    ranked = sorted(risk["supplier_risk_scores"].items(), key=lambda kv: -kv[1])
    style.apply()
    fig, ax = plt.subplots(figsize=(style.TEXT_WIDTH, 2.5))
    xs = range(1, len(ranked) + 1)
    colours = [style.COLOURS["system"] if s >= 0.45 else style.COLOURS["baseline"] for _, s in ranked]
    ax.bar(xs, [s for _, s in ranked], width=0.8, color=colours, linewidth=0)
    ax.axhline(0.60, color=style.COLOURS["text"], linewidth=0.6, linestyle="--")
    ax.axhline(0.45, color=style.COLOURS["text"], linewidth=0.6, linestyle=":")
    ax.text(len(ranked) + 0.5, 0.60, "HIGH 0.60", fontsize=7, va="bottom", ha="right")
    ax.text(len(ranked) + 0.5, 0.45, "MEDIUM 0.45", fontsize=7, va="bottom", ha="right")
    for name in ("Shell PLC", "Siemens AG", "Johnson Matthey PLC"):
        rank = next(i for i, (n, _) in enumerate(ranked, 1) if n == name)
        score = ranked[rank - 1][1]
        ax.annotate(f"{name}\nrank {rank}, {score:.4f}", xy=(rank, score), xytext=(rank + (3 if rank < 60 else -3), score + 0.12),
                    fontsize=7, ha="left" if rank < 60 else "right",
                    arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": style.COLOURS["text"]})
    ax.set_xlim(0, len(ranked) + 1)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel(f"Tier-1 suppliers ranked by risk score (n = {len(ranked)})")
    ax.set_ylabel("Risk score")
    style.save(fig, figs / "casestudy-risk-manager.pdf")

    _text_figure(figs / "casestudy-csco-agent.pdf", (ao / "5_csco.txt").read_text().strip(), columns=2, fontsize=6.0)
    _text_figure(figs / "casestudy-alternative-sourcing.pdf", (ao / "6_alternative_sourcing.txt").read_text().strip(),
                 columns=2, fontsize=6.0)
    print("\n".join(sorted(p.name for p in figs.glob("*.pdf"))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("article", "preflight", "launch", "run", "one", "figures"))
    p.add_argument("--attempt", type=int, default=1)
    args = p.parse_args()
    {"article": cmd_article, "preflight": cmd_preflight, "launch": cmd_launch, "run": cmd_run, "one": cmd_one,
     "figures": cmd_figures}[args.command](args)


if __name__ == "__main__":
    main()
