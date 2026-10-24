#!/usr/bin/env python3
"""
The 630 evaluation runs of Chapter 5 on the 30 scenarios.

    python scripts/run_evaluation.py plan   [--models ...] [--conditions ...]
    python scripts/run_evaluation.py run    [--models ...] [--conditions ...] [--workers N] [--max-usd X]
    python scripts/run_evaluation.py status

Conditions (the stored articles are used unchanged)
    original    evaluation/scenarios/multi_company_scenarios.json; gpt-4o, gpt-4.1 and gpt-5-mini, five runs each
    noise       evaluation/results/noise_sensitivity/noisy_articles, mild, moderate and high; gpt-4o, one run each
    paraphrase  evaluation/results/prompt_sensitivity/paraphrases, three per article; gpt-4o, one run each

Every run is a separate process with a new crew, product agents disabled, the model given by OPENAI_MODEL_NAME and no
temperature sent. Every API call is logged with the served snapshot, tokens, cached tokens, reasoning tokens and finish
reason; the process stops if a served snapshot differs from ENVIRONMENT.md. Each agent is checked on its own answer
(scripts/pipeline_runner.compare) against the scenario references (evaluation/ground_truth_final) on the cleaned
network. Needs an OpenAI API key in .env and the Neo4j graph of dataset/supplychainKG_clean.csv.

Completion rule
    A run whose pipeline finishes is recorded and scored, whatever its output. A run in which the pipeline raises an
    error of its own (not an API or database error) is recorded as completed with success false and is scored as wrong
    on every agent; it is not run again. The workers then pause for inspection unless --continue-after-pipeline-error
    is given.
    An API error (rate limit, server, connection, timeout) or a graph-database error is not an outcome of the system:
    the attempt is logged, with its calls and cost, and the run is started again after a pause. After eight such
    attempts the worker stops and the run stays pending. An insufficient-quota error, an authentication or model
    error, a snapshot mismatch or a change in the graph stops every worker; the run stays pending. Retries made inside
    the OpenAI client and API errors absorbed by CrewAI are counted in the record. Nothing is killed on a timer:
    runtime is recorded and judged against the 600-second budget when the runs are scored. `run` resumes from the runs
    already recorded; a run whose pipeline finished but was not scored is scored from its saved output.

Output
    evaluation/results/v1_2_{set}/{condition}/{model}/{scenario}__{variant}__r{k}.json   record (paths as counts and
        per-tier agreement with the ground truth), committed
    evaluation/results/v1_2_{set}/full/...json.gz      complete agent outputs, not committed
    evaluation/results/v1_2_{set}/attempts/...jsonl    attempts stopped by an API or database error
    evaluation/results/v1_2_{set}/logs/...log          pipeline console output, not committed
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ("gpt-4o", "gpt-4.1", "gpt-5-mini")
CONDITIONS = ("original", "noise", "paraphrase")
NOISE_LEVELS = ("mild", "moderate", "high")
RUNS_PER_SCENARIO = 5

# OpenAI list prices, USD per million tokens: input, cached input, output (platform.openai.com/docs/pricing,
# standard tier, checked 13 October 2026).
PRICES = {
    "gpt-4o-2024-08-06": (2.50, 1.25, 10.00),
    "gpt-4.1-2025-04-14": (2.00, 0.50, 8.00),
    "gpt-5-mini-2025-08-07": (0.25, 0.025, 2.00),
}
ENVIRONMENT = {"crewai": "1.9.3", "openai": "1.83.0", "neo4j": "6.1.0"}  # requirements-lock.txt
GRAPH_SIZE = {"nodes": 6318, "relationships": 22862}  # dataset/supplychainKG_clean.csv as loaded
PIPELINE_PATHS = ("crew.py", "agents", "tools", "config", "offline", "analysis", "dataset",
                  "evaluation/ground_truth_final", "evaluation/scenarios", "evaluation/results/noise_sensitivity/noisy_articles",
                  "evaluation/results/prompt_sensitivity/paraphrases", "scripts/run_evaluation.py",
                  "scripts/pipeline_runner.py", "scripts/generate_ground_truth_v1_2.py")

EXIT_DONE, EXIT_TRANSIENT, EXIT_QUOTA, EXIT_SNAPSHOT = 0, 75, 76, 77
MAX_TRANSIENT_ATTEMPTS = 8
BACKOFF = (60, 120, 240, 480, 900, 900, 900)


# ---------------------------------------------------------------------------------------------------- plan

def scenarios():
    return {s["scenario_id"]: s for s in json.loads(
        (ROOT / "evaluation" / "scenarios" / "multi_company_scenarios.json").read_text())}


def article(sid, condition, variant):
    if condition == "original":
        return scenarios()[sid]["news_article"]
    if condition == "noise":
        path = ROOT / "evaluation" / "results" / "noise_sensitivity" / "noisy_articles" / f"{sid}_noisy.json"
        return json.loads(path.read_text())["noisy_versions"][variant]
    path = ROOT / "evaluation" / "results" / "prompt_sensitivity" / "paraphrases" / f"{sid}_paraphrases.json"
    return json.loads(path.read_text())["paraphrases"][int(variant[1:]) - 1]


def plan(run_set, models=MODELS, conditions=CONDITIONS):
    ids = list(scenarios())
    out = []
    for model in models:
        for sid in ids:
            if "original" in conditions:
                out += [("original", model, sid, "article", k) for k in range(1, RUNS_PER_SCENARIO + 1)]
            if model == "gpt-4o" and "noise" in conditions:
                out += [("noise", model, sid, level, 1) for level in NOISE_LEVELS]
            if model == "gpt-4o" and "paraphrase" in conditions:
                out += [("paraphrase", model, sid, f"p{i}", 1) for i in (1, 2, 3)]
    return out


def run_id(run):
    condition, model, sid, variant, k = run
    return f"{condition}/{model}/{sid}__{variant}__r{k}"


def parse_run_id(rid):
    condition, model, rest = rid.split("/")
    sid, variant, k = rest.split("__")
    return condition, model, sid, variant, int(k[1:])


def out_dir(run_set):
    return ROOT / "evaluation" / "results" / f"v1_2_{run_set}"


def record_path(run_set, rid):
    return out_dir(run_set) / f"{rid}.json"


def side_path(run_set, kind, rid, suffix):
    return out_dir(run_set) / kind / f"{rid.replace('/', '__')}{suffix}"


def recorded(run_set, rid):
    path = record_path(run_set, rid)
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "completed"
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------------- provenance

def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def provenance():
    import crewai
    import neo4j
    import openai
    return {
        "commit": git("rev-parse", "HEAD"),
        "tags_at_commit": git("tag", "--points-at", "HEAD").split(),
        "pipeline_files_modified": git("status", "--porcelain", "--", *PIPELINE_PATHS).splitlines(),
        "python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
        "crewai": crewai.__version__, "openai": openai.__version__, "neo4j": neo4j.__version__,
    }


def check_environment(run_set):
    prov = provenance()
    wrong = {k: prov[k] for k, v in ENVIRONMENT.items() if prov[k] != v}
    if wrong:
        raise SystemExit(f"Environment differs from requirements-lock.txt: {wrong} ({prov['executable']})")
    if run_set == "full":
        if prov["pipeline_files_modified"]:
            raise SystemExit("Pipeline files are modified; commit them before the full runs:\n  "
                             + "\n  ".join(prov["pipeline_files_modified"]))
    return prov


# ---------------------------------------------------------------------------------------------------- one run

class SnapshotMismatch(BaseException):
    """Raised from the call log; a BaseException so that no agent or tool handler can absorb it."""


GRAPH_ERRORS: list = []
API_ERRORS: list = []
CLIENT_RETRIES: list = []


class RetryCount(logging.Handler):
    def emit(self, rec):
        if "Retrying request" in rec.getMessage():
            CLIENT_RETRIES.append(datetime.now().isoformat(timespec="seconds"))


def install_call_log(base):
    def record(api, kwargs, response):
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) if usage else None
        completion = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) if usage else None
        pdet = (getattr(usage, "prompt_tokens_details", None) or getattr(usage, "input_tokens_details", None)) if usage else None
        cdet = (getattr(usage, "completion_tokens_details", None) or getattr(usage, "output_tokens_details", None)) if usage else None
        choices = getattr(response, "choices", None)
        if choices:
            finish = getattr(choices[0], "finish_reason", None)
        else:
            incomplete = getattr(response, "incomplete_details", None)
            finish = getattr(incomplete, "reason", None) or getattr(response, "status", None)
        served = getattr(response, "model", None)
        entry = {
            "api": api,
            "caller": base.caller(kwargs.get("messages") or kwargs.get("input"), kwargs.get("instructions")),
            "requested_model": kwargs.get("model"),
            "temperature": kwargs.get("temperature", "not sent (API default)"),
            "served_model": served,
            "prompt_tokens": prompt,
            "cached_tokens": (getattr(pdet, "cached_tokens", 0) or 0) if pdet else 0,
            "completion_tokens": completion,
            "reasoning_tokens": (getattr(cdet, "reasoning_tokens", 0) or 0) if cdet else 0,
            "finish_reason": finish,
            "truncated": finish in ("length", "max_output_tokens"),
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        base.CALLS.append(entry)
        expected = base.SNAPSHOTS.get(kwargs.get("model"))
        if not served or served != expected:
            raise SnapshotMismatch(f"served {served} for {kwargs.get('model')}; ENVIRONMENT.md gives {expected}")

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
    from openai.resources.responses import AsyncResponses, Responses
    wrap(Completions, "chat.completions", False)
    wrap(AsyncCompletions, "chat.completions", True)
    wrap(Responses, "responses", False)
    wrap(AsyncResponses, "responses", True)
    client_log = logging.getLogger("openai._base_client")
    client_log.setLevel(logging.INFO)
    client_log.addHandler(RetryCount())


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


def check_graph():
    from tools.neo4j_setup import graph
    if graph is None:
        raise ConnectionError("Neo4j graph not available")
    nodes = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
    rels = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    if (nodes, rels) != (GRAPH_SIZE["nodes"], GRAPH_SIZE["relationships"]):
        raise SystemExit(f"Graph has {nodes} nodes and {rels} relationships; expected {GRAPH_SIZE}")
    return {"nodes": nodes, "relationships": rels}


def cost(calls):
    usd, missing = 0.0, set()
    for c in calls:
        price = PRICES.get(c.get("served_model"))
        if not price:
            missing.add(c.get("served_model"))
            continue
        pin, cached, pout = c.get("prompt_tokens") or 0, c.get("cached_tokens") or 0, c.get("completion_tokens") or 0
        usd += ((pin - cached) * price[0] + cached * price[1] + pout * price[2]) / 1_000_000
    return {
        "calls": len(calls),
        "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
        "cached_tokens": sum(c.get("cached_tokens") or 0 for c in calls),
        "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
        "reasoning_tokens": sum(c.get("reasoning_tokens") or 0 for c in calls),
        "usd": round(usd, 6),
        "unpriced_models": sorted(m for m in missing if m),
    }


def classify(exc):
    """'quota', 'transient' (API or database), 'snapshot', or None for an error of the pipeline itself."""
    if isinstance(exc, SnapshotMismatch):
        return "snapshot"
    chain, seen = [], set()
    e = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append(e)
        e = e.__cause__ or e.__context__
    text = " ".join(f"{type(x).__module__}.{type(x).__name__}: {x}" for x in chain)
    if "insufficient_quota" in text:
        return "quota"
    import httpx
    import openai
    from neo4j import exceptions as neo
    for x in chain:
        if isinstance(x, (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)):
            return "environment"
        if isinstance(x, (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                          openai.InternalServerError, httpx.TransportError, ConnectionError, TimeoutError,
                          neo.ServiceUnavailable, neo.SessionExpired, neo.TransientError)):
            return "transient"
        if isinstance(x, openai.APIStatusError) and getattr(x, "status_code", 0) in (408, 409, 429, 500, 502, 503, 504):
            return "transient"
    markers = ("Error code: 429", "Error code: 500", "Error code: 502", "Error code: 503", "Error code: 504",
               "Connection error", "Request timed out", "ServiceUnavailable", "SessionExpired")
    return "transient" if any(m in text for m in markers) else None


def path_agreement(base, kg, ref):
    a, b = base.path_sets(kg or {}), base.path_sets(ref or {})
    out = {}
    for tier in a:
        out[tier] = {"tp": len(a[tier] & b[tier]), "fp": len(a[tier] - b[tier]), "fn": len(b[tier] - a[tier])}
    keys = sorted("|".join(p) for tier in a for p in a[tier])
    return out, hashlib.sha256("\n".join(keys).encode()).hexdigest()


def one(run_set, rid, worker):
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    condition, model, sid, variant, k = parse_run_id(rid)
    for var in ("MODEL", "MODEL_NAME", "OPENAI_MODEL_NAME"):
        os.environ.pop(var, None)
    os.environ["OPENAI_MODEL_NAME"] = model
    os.environ["DISABLE_PRODUCT_AGENTS"] = "1"
    os.environ["CREWAI_STORAGE_DIR"] = f"supply_chain_evaluation_w{worker}"
    os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")

    prov = check_environment(run_set)
    import scripts.pipeline_runner as base
    import scripts.pipeline_runner as v12
    from offline import kg as offline_kg, names
    from scripts.generate_ground_truth_v1_2 import ground_truth
    install_call_log(base)
    install_graph_watch()

    started = datetime.now().isoformat(timespec="seconds")
    scenario = dict(scenarios()[sid])
    scenario["news_article"] = article(sid, condition, variant)
    gt = json.loads((ROOT / "evaluation" / "ground_truth_final" / f"{sid}.json").read_text())
    record = {
        "run_id": rid, "condition": condition, "model": model, "scenario_id": sid, "variant": variant, "run_index": k,
        "monitored_company": scenario["monitored_company"],
        "article_sha256": hashlib.sha256(scenario["news_article"].encode()).hexdigest(),
        "ground_truth": "v1.2", "graph_file": "dataset/supplychainKG_clean.csv", "started": started,
        "provenance": prov,
    }

    full = side_path(run_set, "full", rid, ".json.gz")
    if full.exists():
        with gzip.open(full, "rt") as f:
            run = json.load(f)
        record.update(run.get("pre_run") or {})
        print(f"Scoring the saved output of {rid}; the pipeline is not run again.", flush=True)
    else:
        try:
            record["graph"] = check_graph()
            GRAPH_ERRORS.clear()
            API_ERRORS.clear()
            CLIENT_RETRIES.clear()
            run = base.run_one(scenario)
            if GRAPH_ERRORS:
                raise ConnectionError("graph query failed during the run: " + GRAPH_ERRORS[0])
        except KeyboardInterrupt:
            raise
        except BaseException as exc:  # noqa: BLE001 - every outcome is recorded
            kind = "environment" if isinstance(exc, SystemExit) else classify(exc)
            if kind in ("quota", "transient", "snapshot", "environment"):
                side = side_path(run_set, "attempts", rid, ".jsonl")
                side.parent.mkdir(parents=True, exist_ok=True)
                with side.open("a") as f:
                    f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "kind": kind,
                                        "error": str(exc)[:2000], "graph_errors": GRAPH_ERRORS[:5],
                                        "api_errors": API_ERRORS[:10], "client_retries": len(CLIENT_RETRIES),
                                        "calls": list(base.CALLS), "cost": cost(base.CALLS)}, default=str) + "\n")
                print(f"ATTEMPT STOPPED ({kind}): {exc}", flush=True)
                sys.exit({"quota": EXIT_QUOTA, "transient": EXIT_TRANSIENT}.get(kind, EXIT_SNAPSHOT))
            run = {"scenario_id": sid, "success": False, "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(), "calls": list(base.CALLS), "agent_outputs": {},
                   "runtime_seconds": None}
        else:
            run["success"], run["error"] = True, None
        run["api_errors"], run["client_retries"] = list(API_ERRORS), len(CLIENT_RETRIES)
        run["pre_run"] = {"graph": record.get("graph"), "provenance": prov, "started": started}
        run["truncated_calls"] = [c for c in run.get("calls") or [] if c.get("truncated")]
        full.parent.mkdir(parents=True, exist_ok=True)
        tmp = full.with_name(full.name + ".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(run, f, default=str)
        tmp.replace(full)

    attempts_file = side_path(run_set, "attempts", rid, ".jsonl")
    attempts = [json.loads(line) for line in attempts_file.read_text().splitlines()] if attempts_file.exists() else []
    record.update({
        "status": "completed", "finished": datetime.now().isoformat(timespec="seconds"),
        "success": run["success"], "error": run.get("error"), "traceback": run.get("traceback"),
        "runtime_seconds": run.get("runtime_seconds"),
        "calls": run.get("calls") or [], "truncated_calls": len(run.get("truncated_calls") or []),
        "cost": cost(run.get("calls") or []),
        "earlier_attempts": [{k2: a[k2] for k2 in ("time", "kind", "error", "cost")} for a in attempts],
        "earlier_attempts_usd": round(sum(a["cost"]["usd"] for a in attempts), 6),
        "api_errors_within_run": run.get("api_errors") or [],
        "client_retries_within_run": run.get("client_retries") or 0,
        "agent_answers": run.get("agent_answers"),
    })

    if run["success"]:
        not_objects = [k for k, v in run["agent_outputs"].items() if v is not None and not isinstance(v, dict)]
        if not_objects:
            run = dict(run, agent_outputs={k: (v if isinstance(v, dict) else None)
                                           for k, v in run["agent_outputs"].items()})
        record["answers_not_json_objects"] = not_objects
        net = offline_kg.load()
        resolver = names.Resolver(net)
        from evaluation.ground_truth_generator import GroundTruthGenerator
        GroundTruthGenerator._network, GroundTruthGenerator._resolver = net, resolver
        meta_path = base.TEMP_KG / f"{sid}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        dm = run["agent_outputs"].get("disruption_analysis") or {}
        req = meta.get("requested") or {}
        cond = ground_truth(net, resolver, scenario, {
            "type": dm.get("type") or "None", "ongoing_disruption": bool(meta.get("ongoing_disruption")),
            "scope": req.get("scope", "country"), "countries": req.get("countries") or [],
            "industries": req.get("industries") or [], "companies": req.get("companies") or [],
            "basis": "the values the pipeline passed to the path tool"})
        kg = run["agent_outputs"].get("kg_results")
        agreement, digest = path_agreement(base, kg if isinstance(kg, dict) else {}, gt["kg_results"])
        record.update({
            "kg_tool_called": bool(meta),
            "kg_tool_metadata": meta,
            "comparison": v12.compare(sid, run, gt, cond, meta, resolver),
            "paths_vs_ground_truth": agreement,
            "path_set_sha256": digest,
        })
        record["agent_outputs"] = v12.slim(run)["agent_outputs"]
    else:
        record["agent_outputs"] = {}

    path = record_path(run_set, rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str))
    tmp.replace(path)
    c = record.get("comparison") or {}
    print(f"DONE {rid} success={record['success']} "
          + (f"DM={c['disruption_monitor']['pass']} KG={c['kg_query']['pass']} Risk={c['risk_manager']['pass']} "
             f"CSCO={c['csco']['pass']} e2e={c['system']['end_to_end']} " if c else "")
          + f"{record['runtime_seconds']}s ${record['cost']['usd']}", flush=True)
    sys.exit(EXIT_DONE)


# ---------------------------------------------------------------------------------------------------- orchestration

STOP = threading.Event()
LOCK = threading.Lock()


def spent(run_set):
    total = 0.0
    root = out_dir(run_set)
    for p in root.rglob("*.json"):
        if "full" in p.parts:
            continue
        try:
            total += json.loads(p.read_text()).get("cost", {}).get("usd", 0)
        except Exception:
            pass
    for p in (root / "attempts").glob("*.jsonl") if (root / "attempts").exists() else []:
        for line in p.read_text().splitlines():
            total += json.loads(line)["cost"]["usd"]
    return round(total, 4)


def worker_loop(run_set, worker, queue, max_usd, report, pause_on_error):
    for rid in queue:
        if STOP.is_set():
            return
        if recorded(run_set, rid):
            continue
        transient = 0
        while not STOP.is_set():
            with LOCK:
                if spent(run_set) >= max_usd:
                    report.append(f"budget of ${max_usd} reached before {rid}")
                    STOP.set()
                    return
            log = side_path(run_set, "logs", rid, ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a") as f:
                f.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} worker {worker} =====\n")
                f.flush()
                code = subprocess.run([sys.executable, str(Path(__file__).resolve()), "one", "--set", run_set,
                                       "--run", rid, "--worker", str(worker)],
                                      cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
            tail = log.read_text().splitlines()[-1:] or [""]
            if code == EXIT_DONE and recorded(run_set, rid):
                print(f"[w{worker}] {tail[0]}", flush=True)
                if pause_on_error and not json.loads(record_path(run_set, rid).read_text()).get("success"):
                    report.append(f"{rid}: recorded with a pipeline error (scored as wrong); workers paused for "
                                  "inspection")
                    STOP.set()
                    return
                break
            if code == EXIT_TRANSIENT:
                transient += 1
                if transient >= MAX_TRANSIENT_ATTEMPTS:
                    report.append(f"{rid}: {transient} attempts stopped by API or database errors; left pending")
                    STOP.set()
                    return
                wait = BACKOFF[min(transient - 1, len(BACKOFF) - 1)]
                print(f"[w{worker}] {rid}: API or database error, attempt {transient}; retry in {wait}s", flush=True)
                STOP.wait(wait)
                continue
            if code == EXIT_QUOTA:
                report.append(f"{rid}: insufficient quota; all workers stopped, run left pending")
            elif code == EXIT_SNAPSHOT:
                report.append(f"{rid}: snapshot or environment check failed ({tail[0]}); all workers stopped")
            else:
                report.append(f"{rid}: runner exited with code {code} ({tail[0]}); all workers stopped")
            STOP.set()
            return


def cmd_run(args):
    runs = [run_id(r) for r in plan(args.set, args.models, args.conditions)]
    check_environment(args.set)
    by_sid = {}
    for rid in runs:
        by_sid.setdefault(parse_run_id(rid)[2], []).append(rid)
    queues = [[] for _ in range(args.workers)]
    for i, sid in enumerate(sorted(by_sid)):
        queues[i % args.workers] += by_sid[sid]
    pending = [r for r in runs if not recorded(args.set, r)]
    print(f"{len(runs)} runs in set, {len(pending)} pending, {args.workers} workers, spent so far ${spent(args.set)}",
          flush=True)
    report = []
    threads = [threading.Thread(target=worker_loop,
                                args=(args.set, w, q, args.max_usd, report, not args.continue_after_pipeline_error))
               for w, q in enumerate(queues)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    status(args.set, runs)
    for line in report:
        print("STOPPED:", line)
    sys.exit(1 if report else 0)


def status(run_set, runs):
    done = [r for r in runs if recorded(run_set, r)]
    failed = []
    for r in done:
        rec = json.loads(record_path(run_set, r).read_text())
        if not rec.get("success"):
            failed.append(r)
    print(f"{len(done)} of {len(runs)} runs recorded; {len(failed)} recorded with a pipeline error "
          f"(scored as wrong); {len(runs) - len(done)} pending; total spent ${spent(run_set)}")
    for r in failed:
        print("  pipeline error:", r)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("plan", "run", "status", "one"))
    p.add_argument("--set", default="full", choices=("full",), help=argparse.SUPPRESS)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-usd", type=float, default=10.0)
    p.add_argument("--continue-after-pipeline-error", action="store_true")
    p.add_argument("--run")
    p.add_argument("--worker", type=int, default=0)
    args = p.parse_args()
    if args.command == "one":
        one(args.set, args.run, args.worker)
    runs = [run_id(r) for r in plan(args.set, args.models, args.conditions)]
    if args.command == "plan":
        for r in runs:
            print(r)
        print(f"{len(runs)} runs")
    elif args.command == "status":
        status(args.set, runs)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
