"""Scores of the relevance benchmark (Chapter 4, Tables 4.5 to 4.7; definitions in analysis/METRICS.md).

    python -m evaluation.benchmark_v2.score

Scores the 1,117 items of the test set from the stored runs, and refuses to score while a run is missing. Detection of
the primary system and of the comparison: precision, recall, specificity, F1 and MCC with 95% bootstrap intervals
stratified by item type (10,000 resamples), recall on the positives and specificity per negative type with exact
intervals, and the always-alert baseline. A failed run is scored as wrong. Agent-level metrics on the positive items
use the decision functions of scripts/score_v1_2.py. Lower prevalence: observed at 10% and 1% by drawing the positives
at random (1,000 draws), with a stratified bootstrap interval, and projected at 0.1% and 0.01% for the stated mixture
of negative types, with intervals from the Jeffreys posteriors of sensitivity and specificity. Run-to-run variation
from the five further runs on the items of items/test/repeat_sample.json. Every metric comes from analysis/metrics.py.
Writes evaluation/results/benchmark_v2/test/scores.json. No model is called.
"""
import argparse
import json
import math
import statistics
from collections import Counter

import numpy as np

from analysis.metrics import (DEFAULT_BOOT, DEFAULT_SEED, Confusion, alert, bootstrap, clopper_pearson,
                              constant_baseline, entity_keys, normalize, path_node_set, project, subsample_at_prevalence)
from evaluation.benchmark_v2.test_set import in_set
from evaluation.benchmark_v2.run_pipeline import RESULTS, TIERS
from evaluation.benchmark_v2.specify import OUT
from offline import kg as offline_kg, names
from scripts.score_v1_2 import decisions, mcc_block

METRICS = ("precision", "recall", "specificity", "f1", "mcc")
NEGATIVES = ("N1", "N2", "N3", "N4")
MIXTURES = {"monitoring stream": {"N1": 0.70, "N4": 0.15, "N2": 0.10, "N3": 0.05}}
OBSERVED, PROJECTED = (0.10, 0.01), (0.001, 0.0001)
SERVED = "gpt-4o-2024-08-06"


def clean(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def interval(k, n):
    return {"k": k, "n": n, "rate": k / n if n else None, "exact_95": list(clopper_pearson(k, n))}


def detection(units):
    """units: (item type, label, predicted alert) for every item."""
    conf = Confusion.from_labels([u[1] for u in units], [u[2] for u in units])
    out = {"counts": conf.as_dict()}
    strata = [u[0] for u in units]
    for m in METRICS:
        out[m] = bootstrap(units, lambda us, m=m: getattr(Confusion.from_labels([u[1] for u in us],
                                                                              [u[2] for u in us]), m), strata=strata)
    by_type = {}
    for t in sorted({u[0] for u in units}):
        of = [u for u in units if u[0] == t]
        correct = sum(u[1] == u[2] for u in of)
        by_type[t] = {"measure": "recall" if t == "P" else "specificity", **interval(correct, len(of))}
    out["by_type"] = by_type
    pos = sum(u[1] for u in units)
    out["baselines"] = {"always_alert": constant_baseline(pos, len(units) - pos, True).as_dict()}
    return out


def bootstrap_at(positives, negatives, k, statistic):
    """Stratified bootstrap of a benchmark with k positives: k positives drawn with replacement and the negatives
    resampled within each type, so the interval covers both which positives are drawn and sampling uncertainty."""
    rng = np.random.default_rng(DEFAULT_SEED)
    groups = {}
    for u in negatives:
        groups.setdefault(u[0], []).append(u)
    values, undefined = [], 0
    for _ in range(DEFAULT_BOOT):
        p = [positives[i] for i in rng.choice(len(positives), size=k, replace=True)]
        n = [g[i] for g in groups.values() for i in rng.choice(len(g), size=len(g), replace=True)]
        v = statistic(p, n)
        if math.isnan(v):
            undefined += 1
        else:
            values.append(v)
    lo, hi = np.quantile(values, [0.025, 0.975]) if values else (math.nan, math.nan)
    return {"lo": float(lo), "hi": float(hi), "n_boot": DEFAULT_BOOT, "n_undefined": undefined}


def prevalence(units):
    positives = [u for u in units if u[1]]
    negatives = [u for u in units if not u[1]]

    def stat(m):
        return lambda p, n: getattr(Confusion.from_labels([u[1] for u in p + n], [u[2] for u in p + n]), m)

    observed = {}
    for pi in OBSERVED:
        k = round(pi * len(negatives) / (1 - pi))
        observed[str(pi)] = ({m: {**subsample_at_prevalence(positives, negatives, pi, stat(m)),
                                  "bootstrap": bootstrap_at(positives, negatives, k, stat(m))}
                              for m in ("precision", "f1", "mcc")} if 1 <= k <= len(positives)
                             else f"not computable: needs {k} positives with {len(negatives)} negatives")
    tp = sum(u[2] for u in positives)
    tn = {t: sum(not u[2] for u in negatives if u[0] == t) for t in NEGATIVES}
    n = {t: sum(1 for u in negatives if u[0] == t) for t in NEGATIVES}
    projected = {name: project(tp, len(positives), tn, n, w, PROJECTED) for name, w in MIXTURES.items()}
    return {"observed": observed, "projected": projected, "mixtures": MIXTURES}


def context(net, resolver, firms):
    return {
        "resolver": resolver,
        "vocab": {"countries": entity_keys("countries", {c for cs in net.countries.values() for c in cs}, resolver),
                  "industries": entity_keys("industries", set(net.industry.values()), resolver),
                  "companies": entity_keys("companies", set(net.industry), resolver)},
        "universe": {f: sum(1 for t in range(1, 5) for _ in offline_kg.paths(net, f, t, "simple")) for f in firms},
        "tier1": {f: sorted({normalize(s) for s in net.suppliers.get(f, [])}) for f in firms},
    }


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    folder = OUT / "test"
    specs = {s["item_id"]: s for s in json.loads((folder / "specs.json").read_text())["items"]}
    items = in_set()
    base = RESULTS / "test"
    missing = [i for i in items if not (base / "primary" / "r1" / f"{i}.json").exists()
               and not (base / "primary" / "r1" / f"{i}.failed.json").exists()]
    missing += [f"comparison {i}" for i in items if not (base / "comparison" / f"{i}.json").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} runs missing, for example {missing[:5]}; nothing is scored")

    net = offline_kg.load()
    resolver = names.Resolver(net)
    ctx = context(net, resolver, sorted({specs[i]["firm"] for i in items}))
    ctx["firm_of"] = {i: specs[i]["firm"] for i in items}
    primary, comparison, dec_rows, gts, calls, ongoing = [], [], [], {}, Counter(), []
    cost = {"P": [], "N": []}
    runtime = []
    for i in items:
        t = specs[i]["item_type"]
        gt_raw = json.loads((folder / "ground_truth_final" / f"{i}.json").read_text())
        label = bool(gt_raw["scenario_metadata"]["label"])
        path = base / "primary" / "r1" / f"{i}.json"
        run = json.loads((path if path.exists() else path.with_suffix(".failed.json")).read_text())
        ao = run.get("agent_outputs") or {}
        primary.append((t, label, alert(ao.get("kg_results")) if run["success"] else not label))
        comp = json.loads((base / "comparison" / f"{i}.json").read_text())
        comparison.append((t, label, comp["alert"] if comp["success"] else not label))
        da = ao.get("disruption_analysis") if isinstance(ao.get("disruption_analysis"), dict) else {}
        ongoing.append((t, gt_raw["disruption_analysis"]["ongoing_disruption"],
                        da.get("ongoing_disruption") is True if run["success"]
                        else not gt_raw["disruption_analysis"]["ongoing_disruption"]))
        calls.update(c.get("served_model") for c in run.get("calls") or [])
        cost["P" if t == "P" else "N"].append(run["cost"]["usd"])
        runtime.append(run.get("runtime_seconds") or 0.0)
        if label:
            gts[i] = {"raw": gt_raw, "positive": True,
                      "path_keys": {path_node_set(c) for tier in TIERS for c in gt_raw["kg_results"].get(tier) or []}}
            dec_rows.append((i, 1, decisions({k: v for k, v in ao.items() if isinstance(v, dict)}, gts[i], ctx, i)))

    failed = [u for u, i in zip(primary, items) if (base / "primary" / "r1" / f"{i}.failed.json").exists()]
    out = {
        "set": "test", "items": len(items), "by_type": dict(Counter(specs[i]["item_type"] for i in items)),
        "positives": sum(u[1] for u in primary), "failed_runs": len(failed),
        "served_models": dict(calls),
        "cost_usd": {"per_positive": statistics.mean(cost["P"]) if cost["P"] else None,
                     "per_negative": statistics.mean(cost["N"]) if cost["N"] else None,
                     "total": sum(cost["P"]) + sum(cost["N"])},
        "runtime_seconds": {"mean": statistics.mean(runtime), "max": max(runtime)},
        "primary": detection(primary),
        "comparison": detection(comparison),
        "ongoing_decision": Confusion.from_labels([u[1] for u in ongoing], [u[2] for u in ongoing]).as_dict(),
        "prevalence": prevalence(primary),
        "agents_on_positives": mcc_block(dec_rows, gts) if dec_rows else None,
    }
    out["comparison_minus_primary"] = {t: {"primary_correct": sum(u[1] == u[2] for u in primary if u[0] == t),
                                           "comparison_correct": sum(u[1] == u[2] for u in comparison if u[0] == t)}
                                       for t in sorted(out["by_type"])}
    repeat = folder / "repeat_sample.json"
    if repeat.exists():
        chosen = json.loads(repeat.read_text())["items"]
        runs = {}
        for r in range(1, 7):
            d = base / "primary" / f"r{r}"
            if all((d / f"{i}.json").exists() or (d / f"{i}.failed.json").exists() for i in chosen):
                runs[r] = {}
                for i in chosen:
                    p = d / f"{i}.json"
                    rec = json.loads((p if p.exists() else p.with_suffix(".failed.json")).read_text())
                    label = bool(json.loads((folder / "ground_truth_final" / f"{i}.json").read_text())
                                 ["scenario_metadata"]["label"])
                    runs[r][i] = (specs[i]["item_type"], label,
                                  alert((rec.get("agent_outputs") or {}).get("kg_results")) if rec["success"]
                                  else not label)
        if len(runs) == 6:
            per = {r: Confusion.from_labels([u[1] for u in v.values()], [u[2] for u in v.values()]) for r, v in runs.items()}
            out["repeats"] = {
                "items": len(chosen), "runs": 6,
                **{m: {"per_run": [getattr(c, m) for c in per.values()],
                       "mean": statistics.mean([getattr(c, m) for c in per.values()]),
                       "sd": statistics.stdev([getattr(c, m) for c in per.values()])} for m in METRICS},
                "same_decision_in_all_runs": sum(len({runs[r][i][2] for r in runs}) == 1 for i in chosen)}
        else:
            out["repeats"] = {"complete_runs": sorted(runs), "status": "incomplete; not scored"}
    target = base / "scores.json"
    target.write_text(json.dumps(clean(out), indent=1, default=str))
    p, c = out["primary"], out["comparison"]
    print(f"{len(items)} items, {out['positives']} positive, {len(failed)} failed runs, served {dict(calls)}")
    for name, block in (("primary", p), ("comparison", c)):
        print(name, {m: round(block[m]["estimate"], 3) for m in METRICS},
              {t: f"{v['k']}/{v['n']}" for t, v in block["by_type"].items()})
    print("wrote", target)


if __name__ == "__main__":
    main()
