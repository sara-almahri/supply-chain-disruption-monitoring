#!/usr/bin/env python3
"""Write the references of the 30 scenarios of Chapter 5 (evaluation/ground_truth_final).

    python -m scripts.generate_ground_truth_v1_2

The references are computed on the supply network (dataset/supplychainKG_clean.csv) from the article specifications
of evaluation/scenarios/original_templates_v1_2.json. A path is disrupted when its last company is a named disrupted
company, or is located in a disrupted country and, under scope "industries", operates in a disrupted industry; paths
contain no company twice. Risk scores, levels and actions follow the risk function of Equation 4.1. The expected CSCO
decision for every scored Tier-1 supplier is stored in "expected_decisions", apart from the risk assessment the
pipeline produces.

Writes evaluation/ground_truth_final/{scenario}.json and index.json.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.metrics import alert, expected_action  # noqa: E402
from evaluation.ground_truth_generator import GroundTruthGenerator  # noqa: E402
from offline import kg, names  # noqa: E402

OUT = ROOT / "evaluation" / "ground_truth_final"
SPECS = ROOT / "evaluation" / "scenarios" / "original_templates_v1_2.json"
SCENARIOS = ROOT / "evaluation" / "scenarios" / "multi_company_scenarios.json"
TIERS = ("tier_1", "tier_2", "tier_3", "tier_4")


def ground_truth(net, resolver, scenario: dict, spec: dict) -> dict:
    firm = scenario["monitored_company"]
    da = {
        "ongoing_disruption": spec["ongoing_disruption"],
        "type": spec["type"] if spec["ongoing_disruption"] else "None",
        "scope": spec["scope"],
        "involved": {k: sorted(spec[k]) for k in ("countries", "industries", "companies")},
    }
    resolution = resolver.disruption(spec["countries"], spec["industries"], spec["companies"], scope=spec["scope"])
    match = resolution.pop("match")
    if spec["ongoing_disruption"]:
        paths = kg.disrupted_chains(net, firm, match["countries"], match["companies"], mode="simple",
                                    industries=match["industries"])
    else:
        paths = kg.disrupted_chains(net, firm, [], [], mode="simple")
    paths["resolution"] = resolution
    risk = GroundTruthGenerator(monitored_company=firm)._calculate_risk_assessment(paths, da, copy.deepcopy(scenario))
    for k in ("generated_at", "methodology", "recommendation"):
        risk.pop(k, None)
    scores = risk.get("supplier_risk_scores") or {}
    return {
        "disruption_analysis": da,
        "kg_results": paths,
        "risk_assessment": risk,
        "expected_decisions": {s: expected_action(v) for s, v in sorted(scores.items())},
        "scenario_metadata": {"scenario_id": scenario["scenario_id"], "label": alert(paths)},
    }


def main():
    net = kg.load()
    GroundTruthGenerator._network = net
    GroundTruthGenerator._resolver = resolver = names.Resolver(net)
    specs = json.loads(SPECS.read_text())["templates"]
    scenarios = json.loads(SCENARIOS.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    labels = {}
    for sc in scenarios:
        sid = sc["scenario_id"]
        gt = ground_truth(net, resolver, sc, specs[sid.split("_")[1]])
        (OUT / f"{sid}.json").write_text(json.dumps(gt, indent=2, default=str))
        labels[sid] = gt["scenario_metadata"]["label"]
        print(sid, [len(gt["kg_results"][t]) for t in TIERS], "suppliers",
              len(gt["risk_assessment"].get("supplier_risk_scores") or {}), "label", labels[sid], flush=True)
    (OUT / "index.json").write_text(json.dumps({"scenarios": sorted(labels)}, indent=1))
    print(f"{sum(labels.values())} positive and {len(labels) - sum(labels.values())} negative scenarios")


if __name__ == "__main__":
    main()
