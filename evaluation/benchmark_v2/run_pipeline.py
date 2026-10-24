"""Run the primary system on the relevance benchmark (Chapter 4 and Appendix C.6).

    python -m evaluation.benchmark_v2.run_pipeline [--shard K/N] [--repeat R] [--items ID ...]

The pipeline of Chapter 4 on GPT-4o (gpt-4o-2024-08-06) without the Product Search, Network Visualizer and Alternative
Sourcing Agents, on the cleaned network, with every API call logged; the run stops if a served snapshot differs from
ENVIRONMENT.md. The pipeline receives the article and the monitored firm. Its run identifier, which two agents pass to
their tools, is an opaque hash of the item id, so no prompt carries the item type. The items run are the 1,117 items
of the test set. Needs an OpenAI API key in .env and the Neo4j graph of dataset/supplychainKG_clean.csv.

Each run is saved to evaluation/results/benchmark_v2/test/primary/r<R>/<item>.json and is not repeated once saved.
A run that ends with a system error is saved as failed and is scored as wrong on every metric. A run interrupted by
the API or the graph database before the system answered (rate limit, exhausted quota, connection loss, timeout,
server error, a failed graph query) has no system output; it is not saved, and the item is run again after a pause.
The graph is checked before every run (reachable, with the node and relationship counts of the cleaned network), and
the script stops if it is not. After eight such interruptions in a row the script stops, without saving the item, so
that it can be resumed later. The repeats of an item share its run identifier and so its temporary files; a lock file
stops a second run of the item while one is in progress.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback

from evaluation.benchmark_v2.test_set import in_set
from evaluation.benchmark_v2.specify import OUT
from offline import kg as offline_kg, names

ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
import scripts.pipeline_runner as base  # noqa: E402
from scripts.run_evaluation import GRAPH_ERRORS, check_graph, install_graph_watch  # noqa: E402

RESULTS = ROOT / "evaluation" / "results" / "benchmark_v2"
INFRASTRUCTURE = re.compile(r"RateLimit|rate limit|insufficient_quota|exceeded your current quota|APIConnection|"
                            r"Connection error|APITimeout|timed out|InternalServerError|Service Unavailable|"
                            r"Bad Gateway|\b50[0234]\b", re.I)
TIERS = [f"tier_{t}" for t in range(1, 5)]
MAX_INTERRUPTIONS = 8


def run_id(item_id):
    return "run_" + hashlib.sha256(f"benchmark_v2:{item_id}".encode()).hexdigest()[:12]


def slim(run):
    out = json.loads(json.dumps(run, default=str))
    ao = out.get("agent_outputs") or {}
    if isinstance(ao.get("kg_results"), dict):
        ao["path_counts"] = [len(ao["kg_results"].get(t) or []) for t in TIERS]
    return out


def run_locked(scenario, item_id):
    """One run of an item, run again after an API or graph interruption. The item's temporary files are read before
    its lock is released."""
    rid = scenario["scenario_id"]
    pause, interruptions = 60, 0
    while True:
        try:
            check_graph()
            GRAPH_ERRORS.clear()
            run = base.run_one(scenario)
            if GRAPH_ERRORS:
                raise ConnectionError("graph query failed during the run: " + GRAPH_ERRORS[0])
            run.update(success=True, error=None)
            break
        except SystemExit:
            raise
        except Exception as exc:
            text = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, ConnectionError) or INFRASTRUCTURE.search(text):
                interruptions += 1
                if interruptions > MAX_INTERRUPTIONS:
                    raise SystemExit(f"{item_id}: {MAX_INTERRUPTIONS} API interruptions in a row ({text[:160]}); "
                                     f"stopped without saving the item")
                print(f"  {item_id}: API interruption ({text[:160]}); retrying in {pause}s", flush=True)
                time.sleep(pause)
                pause = min(pause * 2, 1800)
                continue
            run = {"scenario_id": rid, "success": False, "error": text, "traceback": traceback.format_exc(),
                   "calls": list(base.CALLS), "cost": base.cost_of(base.CALLS), "agent_outputs": {}}
            break
    meta_path = base.TEMP_KG / f"{rid}_meta.json"
    run["kg_tool_metadata"] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", default="1/1")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--items", nargs="*")
    args = ap.parse_args()
    k, n = map(int, args.shard.split("/"))

    for var in ("MODEL", "MODEL_NAME", "OPENAI_MODEL_NAME"):
        os.environ.pop(var, None)
    os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
    os.environ["DISABLE_PRODUCT_AGENTS"] = "1"
    base.install_wrappers()
    install_graph_watch()
    print(f"graph: {check_graph()}", flush=True)
    net = offline_kg.load()
    from evaluation.ground_truth_generator import GroundTruthGenerator
    GroundTruthGenerator._network, GroundTruthGenerator._resolver = net, names.Resolver(net)

    folder = OUT / "test"
    specs = {s["item_id"]: s for s in json.loads((folder / "specs.json").read_text())["items"]}
    todo = [i for i in in_set() if not args.items or i in args.items][k - 1::n]
    out = RESULTS / "test" / "primary" / f"r{args.repeat}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(todo)} items in shard {args.shard}", flush=True)
    for item_id in todo:
        path = out / f"{item_id}.json"
        if path.exists() or path.with_suffix(".failed.json").exists():
            continue
        rid = run_id(item_id)
        text = json.loads((folder / "articles" / f"{item_id}.json").read_text())["article"]
        if not text:
            raise SystemExit(f"{item_id}: no article text")
        scenario = {"scenario_id": rid, "monitored_company": specs[item_id]["firm"], "news_article": text}
        lock = base.TEMP_KG / f"{rid}.lock"  # repeats of an item share its run id and its temporary files
        base.TEMP_KG.mkdir(parents=True, exist_ok=True)
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            raise SystemExit(f"{item_id}: another run of this item is in progress ({lock}); stopped without saving")
        try:
            run = run_locked(scenario, item_id)
        finally:
            lock.unlink()
        run.update(item_id=item_id, run_id=rid, item_type=specs[item_id]["item_type"], firm=specs[item_id]["firm"],
                   set="test", repeat=args.repeat)
        target = path if run["success"] else path.with_suffix(".failed.json")
        target.write_text(json.dumps(slim(run), indent=1, default=str))
        counts = [len((run["agent_outputs"].get("kg_results") or {}).get(t) or []) for t in TIERS]
        print(f"  {item_id}: {'ok' if run['success'] else 'SYSTEM ERROR ' + run['error'][:120]} paths={counts} "
              f"${run['cost']['usd']} {run.get('runtime_seconds')}s", flush=True)


if __name__ == "__main__":
    main()
