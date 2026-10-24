"""Recompute the references of Chapters 4 and 5 from their specifications and compare them with the released files.

    python -m offline.check_references [--set scenarios|benchmark|all]

For every item, the label, the disrupted paths of each tier, the Tier-1 risk scores and the expected CSCO actions are
computed on the released network (dataset/supplychainKG_clean.csv) with NetworkX, without Neo4j or a language model,
and compared with the stored reference: the 30 scenarios of Chapter 5 (evaluation/ground_truth_final) and the 1,117
items of the relevance benchmark of Chapter 4 (evaluation/benchmark_v2/items/test/ground_truth_final). Nothing is written.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.benchmark_v2.ground_truth import compute  # noqa: E402
from evaluation.benchmark_v2.test_set import in_set  # noqa: E402
from evaluation.ground_truth_generator import GroundTruthGenerator  # noqa: E402
from offline import kg, names  # noqa: E402
from scripts.generate_ground_truth_v1_2 import SCENARIOS, SPECS, ground_truth  # noqa: E402

TIERS = ("tier_1", "tier_2", "tier_3", "tier_4")
TOL = 1e-4


def paths(gt):
    return {t: sorted(tuple((n["company"], n["country"], n["industry"]) for n in p) for p in gt["kg_results"].get(t, []))
            for t in TIERS}


def differences(new, ref):
    """Fields in which a recomputed reference differs from the stored one; empty if they agree."""
    out = []
    if new["scenario_metadata"]["label"] != ref["scenario_metadata"]["label"]:
        out.append("label")
    if paths(new) != paths(ref):
        out.append("paths")
    s0 = ref["risk_assessment"].get("supplier_risk_scores") or {}
    s1 = new["risk_assessment"].get("supplier_risk_scores") or {}
    if set(s0) != set(s1) or any(abs(s0[k] - s1[k]) > TOL for k in s0):
        out.append("risk scores")
    if new["expected_decisions"] != ref["expected_decisions"]:
        out.append("actions")
    return out


def check_scenarios(net, resolver):
    specs = json.loads(SPECS.read_text())["templates"]
    folder = ROOT / "evaluation" / "ground_truth_final"
    result = {}
    for sc in json.loads(SCENARIOS.read_text()):
        sid = sc["scenario_id"]
        new = ground_truth(net, resolver, sc, specs[sid.split("_")[1]])
        result[sid] = differences(new, json.loads((folder / f"{sid}.json").read_text()))
    return result


def check_benchmark(net, resolver):
    folder = ROOT / "evaluation" / "benchmark_v2" / "items" / "test"
    specs = {s["item_id"]: s for s in json.loads((folder / "specs.json").read_text())["items"]}
    result = {}
    for i in in_set():
        new, _ = compute(net, resolver, specs[i])
        result[i] = differences(new, json.loads((folder / "ground_truth_final" / f"{i}.json").read_text()))
    return result


def report(name, result):
    bad = {k: v for k, v in result.items() if v}
    print(f"{name}: {len(result) - len(bad)} of {len(result)} references reproduced"
          + ("" if not bad else f"; differences in {bad}"))
    return not bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=("scenarios", "benchmark", "all"), default="all")
    args = ap.parse_args()
    logging.disable(logging.INFO)
    net = kg.load()
    GroundTruthGenerator._network = net
    GroundTruthGenerator._resolver = resolver = names.Resolver(net)
    ok = True
    if args.set in ("scenarios", "all"):
        ok &= report("Chapter 5 scenarios", check_scenarios(net, resolver))
    if args.set in ("benchmark", "all"):
        ok &= report("Chapter 4 relevance benchmark", check_benchmark(net, resolver))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
