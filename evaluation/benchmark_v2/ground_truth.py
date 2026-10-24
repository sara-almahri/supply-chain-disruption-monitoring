"""Ground truth of the relevance benchmark, computed from each item's specification (Section 4.4.1, Appendix C.6).

    python -m evaluation.benchmark_v2.ground_truth

The function is scripts/generate_ground_truth_v1_2.ground_truth, unchanged: the label, the disrupted paths (no company
twice), the Tier-1 risk scores and the expected CSCO actions. It runs before any article exists. Each item is checked
against the property its type requires, and the script stops if one fails:
  P   positive, and the lowest disrupted tier is the intended tier;
  N1  negative (no ongoing disruption);
  N2  negative, in a country in none of the three networks;
  N3  negative, and positive at the intended tier were the same disruption ongoing;
  N4  negative, and positive were the same disruption not confined to its industry (scope "country").
Writes evaluation/benchmark_v2/items/test/ground_truth_final/<item>.json and ground_truth_summary.json. No model is called.
"""
import argparse
import json
from collections import Counter

from evaluation.benchmark_v2.specify import OUT, Frame
from evaluation.ground_truth_generator import GroundTruthGenerator
from offline import kg, names
from scripts.generate_ground_truth_v1_2 import ground_truth


def lowest_tier(gt):
    paths = gt["kg_results"]
    return next((t for t in range(1, kg.MAX_TIER + 1) if paths.get(f"tier_{t}")), None)


def compute(net, resolver, spec):
    scenario = {"scenario_id": spec["item_id"], "monitored_company": spec["firm"]}
    s = {k: spec[k] for k in ("ongoing_disruption", "type", "scope", "countries", "industries", "companies")}
    return ground_truth(net, resolver, scenario, s), s


def check(net, resolver, fr, spec, gt):
    t = spec["item_type"]
    label = gt["scenario_metadata"]["label"]
    if t == "P":
        return label and lowest_tier(gt) == spec["intended_tier"]
    if t in ("N1", "N2") and label:
        return False
    if t == "N2":
        return spec["countries"][0] in fr.outside
    if t == "N3":
        alt = dict(spec, ongoing_disruption=True)
        g2, _ = compute(net, resolver, alt)
        return not label and g2["scenario_metadata"]["label"] and lowest_tier(g2) == spec["intended_tier"]
    if t == "N4":
        alt = dict(spec, scope="country")
        g2, _ = compute(net, resolver, alt)
        return not label and bool(g2["scenario_metadata"]["label"])
    return not label


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    folder = OUT / "test"
    specs = json.loads((folder / "specs.json").read_text())["items"]
    net = kg.load()
    GroundTruthGenerator._resolver = resolver = names.Resolver(net)
    fr = Frame(net)
    out = folder / "ground_truth_final"
    out.mkdir(exist_ok=True)
    failed, summary = [], Counter()
    for spec in specs:
        gt, _ = compute(net, resolver, spec)
        if not check(net, resolver, fr, spec, gt):
            failed.append(spec["item_id"])
            continue
        meta = gt["scenario_metadata"]
        meta.update(item_type=spec["item_type"], lowest_tier=lowest_tier(gt))
        (out / f"{spec['item_id']}.json").write_text(json.dumps(gt, indent=1, ensure_ascii=False))
        summary[f"{spec['item_type']}|{'positive' if meta['label'] else 'negative'}"] += 1
    report = {"items": len(specs), "written": len(specs) - len(failed), "failed_checks": failed,
              "labels_by_type": dict(sorted(summary.items()))}
    (folder / "ground_truth_summary.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    if failed:
        raise SystemExit(f"{len(failed)} items fail the check of their type")


if __name__ == "__main__":
    main()
