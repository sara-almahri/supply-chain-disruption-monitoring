"""The comparison of Chapter 4 (Appendix C.6): the primary run's monitor output with the disruption treated as under
way and its scope set to the whole country.

    python -m evaluation.benchmark_v2.run_comparison

For each saved primary run (repeat 1) the path query of the pipeline (build_disrupted_supply_chains, the function the
path tool calls) is run once more with the arguments the pipeline passed to the tool, or the Disruption Monitor's
involved entities if the tool was not called, and with only those two arguments overridden. Name resolution, the path
definition and the graph are those of the primary run; no model is called. The monitor lists no entity for an article
it judges to report no disruption under way (config/tasks.yaml), so for such an article the query matches nothing and
the comparison differs from the primary system through the scope alone. A primary run that failed has no monitor
output and its comparison is failed too. Needs the Neo4j graph; the graph is checked before every query, and a failed
graph query stops the script without saving the item. Saves evaluation/results/benchmark_v2/test/comparison/<item>.json.
"""
import argparse
import json
import sys

from evaluation.benchmark_v2.run_pipeline import RESULTS, ROOT, TIERS

sys.path.insert(0, str(ROOT))
from scripts.run_evaluation import GRAPH_ERRORS, check_graph, install_graph_watch  # noqa: E402
from tools.full_supply_chain_path_tool import build_disrupted_supply_chains  # noqa: E402


def arguments(run):
    requested = (run.get("kg_tool_metadata") or {}).get("requested")
    if requested:
        return {k: requested.get(k) or [] for k in ("countries", "industries", "companies")}, "path tool arguments"
    involved = ((run["agent_outputs"].get("disruption_analysis") or {}).get("involved") or {})
    return {k: involved.get(k) or [] for k in ("countries", "industries", "companies")}, "monitor output"


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    primary, out = RESULTS / "test" / "primary" / "r1", RESULTS / "test" / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    install_graph_watch()
    print(f"graph: {check_graph()}", flush=True)
    for path in sorted(primary.glob("*.json")):
        item_id = path.name.split(".")[0]
        target = out / f"{item_id}.json"
        if target.exists():
            continue
        run = json.loads(path.read_text())
        record = {"item_id": item_id, "item_type": run["item_type"], "firm": run["firm"],
                  "primary_success": run["success"]}
        if run["success"]:
            given, source = arguments(run)
            check_graph()
            GRAPH_ERRORS.clear()
            kg = build_disrupted_supply_chains(monitored_company=run["firm"], disrupted_countries=given["countries"],
                                               disrupted_industries=given["industries"],
                                               disrupted_companies=given["companies"], scope="country")
            if GRAPH_ERRORS:
                raise SystemExit(f"{item_id}: graph query failed ({GRAPH_ERRORS[0]}); stopped without saving")
            counts = [len(kg.get(t) or []) for t in TIERS]
            record |= {"success": True, "arguments": given, "arguments_from": source, "ongoing_disruption": True,
                       "scope": "country", "resolution": kg.get("resolution"), "path_counts": counts,
                       "alert": sum(counts) > 0}
        else:
            record |= {"success": False, "error": "primary run failed"}
        target.write_text(json.dumps(record, indent=1, default=str))
        print(f"  {item_id}: {record.get('path_counts', 'failed')}", flush=True)


if __name__ == "__main__":
    main()
