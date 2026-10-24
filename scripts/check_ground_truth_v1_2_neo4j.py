#!/usr/bin/env python3
"""The references of the 30 scenarios of Chapter 5 against the Neo4j path query of the pipeline.

    python -m scripts.check_ground_truth_v1_2_neo4j

The references are computed in Python over the network file (offline/). The pipeline's path tool
(tools/full_supply_chain_path_tool.build_disrupted_supply_chains) runs a Cypher query on the Neo4j graph loaded from the
same file. For each scenario the tool is called with the specification of the ground truth (countries, industries,
companies and scope), and its paths are compared with the ground-truth paths tier by tier, a path being the sequence of
its companies. Name resolution (offline/names.py) is shared by both; the traversal and the disruption test are not.
Risk scores are not compared: the pipeline's risk tool and the ground truth call the same function (offline/risk.py).
Needs the Neo4j graph of dataset/supplychainKG_clean.csv. No model is called. Writes logs/neo4j_paths_check.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env", override=True)  # before crewai, whose load_dotenv() finds the .env above the venv

from scripts.run_evaluation import check_graph  # noqa: E402
from tools.full_supply_chain_path_tool import build_disrupted_supply_chains  # noqa: E402

GT = ROOT / "evaluation" / "ground_truth_final"
OUT = ROOT / "logs" / "neo4j_paths_check.json"
TIERS = ("tier_1", "tier_2", "tier_3", "tier_4")


def paths(result, tier):
    return {tuple(node["company"] for node in path) for path in result.get(tier) or []}


def main():
    graph = check_graph()
    rows = {}
    for f in sorted(GT.glob("*_SC*.json")):
        gt = json.loads(f.read_text())
        involved = gt["disruption_analysis"]["involved"]
        tool = build_disrupted_supply_chains(gt["kg_results"]["monitored_company"], involved["countries"],
                                             involved["companies"], involved["industries"],
                                             gt["disruption_analysis"]["scope"])
        tiers = {}
        for t in TIERS:
            g, n = paths(gt["kg_results"], t), paths(tool, t)
            tiers[t] = {"ground_truth": len(g), "neo4j": len(n), "only_ground_truth": len(g - n),
                        "only_neo4j": len(n - g)}
        rows[f.stem] = {"identical": all(v["only_ground_truth"] == v["only_neo4j"] == 0 for v in tiers.values()),
                        "tiers": tiers}
        print(f"{f.stem}: {'identical' if rows[f.stem]['identical'] else 'DIFFERENT'} "
              f"{[tiers[t]['ground_truth'] for t in TIERS]}", flush=True)
    out = {"graph": graph, "scenarios": len(rows), "identical": sum(r["identical"] for r in rows.values()),
           "paths_compared": sum(v["ground_truth"] for r in rows.values() for v in r["tiers"].values()),
           "rows": rows}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"{out['identical']} of {out['scenarios']} scenarios identical; {out['paths_compared']} ground-truth paths")


if __name__ == "__main__":
    main()
