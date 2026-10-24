#!/usr/bin/env python3
"""
Score the 630 runs of Chapter 5 on the 30 scenarios against their references, and the agent decisions of Table 5.4.

    python scripts/score_v1_2.py

Reads the run records and the complete agent outputs (evaluation/results/v1_2_full/) and the scenario references
(evaluation/ground_truth_final/), and writes evaluation/results/analysis/v1_2_full_scores.json and v1_2_full_runs.csv.
No model is called.

Per run: the Disruption Monitor's type, ongoing decision, scope and entity F1, the path precision, recall and F1, the
validity of each hand-off, and the latency, cost and tokens against the declared budgets of 600 seconds, $0.20 and
30,000 tokens. Per condition and model: the agent decisions of Table 5.4, pooled over the 30 scenarios of each run
(units of decision in analysis/METRICS.md), with the mean and standard deviation of MCC over runs and a 95% bootstrap
interval over articles (10,000 resamples, seed 20261011). The three firm scenarios of an article share its text and
almost all of their network, so an article is resampled with all its scenarios and runs, and detection is resampled
within positive and negative articles. Agent correctness on the received input, error propagation, decision
correctness, constrained success, pass^k and the paired differences are computed from these scores by
scripts/ch5_analysis_v1_2.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import evaluation_harness as h  # noqa: E402
from analysis.metrics import (Confusion, MulticlassConfusion, alert, bootstrap, entity_keys, expected_action,  # noqa: E402
                              normalize, path_node_set, risk_level)
from offline import kg as offline_kg, names  # noqa: E402

GT_DIR = ROOT / "evaluation" / "ground_truth_final"
ANALYSIS = ROOT / "evaluation" / "results" / "analysis"
TIERS = ("tier_1", "tier_2", "tier_3", "tier_4")
BUDGET = {"latency_s": 600, "cost_usd": 0.20, "tokens": 30000}
RISK_LEVELS = ("NONE", "LOW", "MEDIUM", "HIGH")
ACTIONS = ("MAINTAIN_STANDARD", "INCREASE_MONITORING", "REPLACE_SUPPLIER")
ENTITY_KINDS = ("countries", "industries", "companies")


# ---------------------------------------------------------------------------------------------------- inputs

def safe_dict(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            pass
    return {}


def eval_dm_type(dm, gt):
    def norm(t):
        return t.lower().strip().replace("labour", "labor")
    return norm(gt.get("disruption_analysis", {}).get("type", "")) == norm(dm.get("type", dm.get("disruption_type", "")))


def eval_dm_entity_f1(dm, gt):
    gi = gt.get("disruption_analysis", {}).get("involved", {})
    pi = dm.get("involved", {})
    ge, pe = set(), set()
    for c in ENTITY_KINDS:
        ge.update(e.lower().strip() for e in gi.get(c, []))
        pe.update(e.lower().strip() for e in pi.get(c, dm.get(f"affected_{c}", [])) if isinstance(e, str))
    if not ge and not pe:
        return 1.0
    if not pe:
        return 0.0
    tp = len(ge & pe)
    p, r = tp / len(pe), (tp / len(ge) if ge else 0)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def load_gt():
    gts = {}
    for p in sorted(GT_DIR.glob("*_SC0*.json")):
        g = json.loads(p.read_text())
        harness_gt = copy.deepcopy(g)
        harness_gt["chief_supply_chain_output"] = {
            "decisions": {s: {"action": a} for s, a in (g.get("expected_decisions") or {}).items()}}
        gts[p.stem] = {"raw": g, "harness": harness_gt,
                       "positive": any(g["kg_results"].get(t) for t in TIERS),
                       "path_keys": {path_node_set(c) for t in TIERS for c in g["kg_results"].get(t) or []}}
    return gts


def runs_of(run_set):
    root = ROOT / "evaluation" / "results" / f"v1_2_{run_set}"
    for p in sorted(root.glob("*/*/*.json")):
        rec = json.loads(p.read_text())
        if rec.get("status") != "completed":
            continue
        full = root / "full" / f"{rec['run_id'].replace('/', '__')}.json.gz"
        with gzip.open(full, "rt") as f:
            run = json.load(f)
        yield rec, run


# ---------------------------------------------------------------------------------------------------- per run

def csco_actions(ao):
    cs = ao.get("chief_supply_chain_output")
    cs = h.safe_json_parse(cs) if isinstance(cs, str) else cs
    dec = cs.get("decisions") if isinstance(cs, dict) else None
    if not isinstance(dec, dict):
        return {}
    return {normalize(k): (v.get("action") if isinstance(v, dict) else None) for k, v in dec.items()}


def risk_scores(ao):
    ra = h.safe_json_parse(ao.get("risk_assessment")) or {}
    sc = ra.get("supplier_risk_scores") if isinstance(ra, dict) else None
    return {normalize(k): v for k, v in (sc or {}).items() if isinstance(v, (int, float))}


def harness_scores(ao, gt, rec):
    """The per-run measures of one run on one agent-output dictionary."""
    hg = gt["harness"]
    kgp = h.eval_kg_query(ao, hg)
    inter = h.eval_interactions(ao, hg)
    da = safe_dict(ao.get("disruption_analysis"))
    gda = gt["raw"]["disruption_analysis"]
    ongoing_ok = da.get("ongoing_disruption") is gda["ongoing_disruption"]
    scope_ok = da.get("scope") == gda["scope"] if gda["ongoing_disruption"] else True
    cost = rec["cost"]
    tokens = (cost.get("prompt_tokens") or 0) + (cost.get("completion_tokens") or 0)
    return {
        "dm_type": 1.0 if eval_dm_type(da, gt["raw"]) else 0.0, "dm_entity_f1": eval_dm_entity_f1(da, gt["raw"]),
        "dm_ongoing": 1.0 if ongoing_ok else 0.0, "dm_scope": 1.0 if scope_ok else 0.0,
        "kg_precision": kgp.precision, "kg_recall": kgp.recall, "kg_f1": kgp.f1, "kg_detail": kgp.detail,
        "hvr": inter.hvr, "handoffs": inter.hvr_details,
        "within_latency": (rec.get("runtime_seconds") or 0) <= BUDGET["latency_s"],
        "within_cost": cost["usd"] <= BUDGET["cost_usd"], "within_tokens": tokens <= BUDGET["tokens"],
        "latency_s": rec.get("runtime_seconds") or 0.0, "cost_usd": cost["usd"], "tokens": tokens,
    }


def decisions(ao, gt, ctx, sid):
    """The METRICS.md units of decision of one run: confusion matrices for MCC."""
    raw = gt["raw"]
    firm = ctx["firm_of"][sid]
    da = safe_dict(ao.get("disruption_analysis"))
    inv = da.get("involved") or {}
    gda = raw["disruption_analysis"]
    out = {"type": (normalize(gda["type"]), normalize(da.get("type") or "")),
           "scope": (gda["scope"], da.get("scope"))}
    for kind in ENTITY_KINDS:
        p = entity_keys(kind, inv.get(kind) or [], ctx["resolver"])
        g = entity_keys(kind, gda["involved"][kind], ctx["resolver"])
        out[kind] = Confusion.from_sets(p, g, ctx["vocab"][kind])
    kg = ao.get("kg_results") if isinstance(ao.get("kg_results"), dict) else {}
    pred_paths = {path_node_set(c) for t in TIERS for c in kg.get(t) or []}
    gold = gt["path_keys"]
    tp = len(pred_paths & gold)
    out["paths"] = Confusion.from_matches(tp, len(pred_paths) - tp, len(gold) - tp, ctx["universe"][firm])
    out["alert"] = (gt["positive"], alert(kg))
    tier1 = ctx["tier1"][firm]
    gold_scores = {normalize(k): v for k, v in raw["risk_assessment"]["supplier_risk_scores"].items()}
    pred_scores = risk_scores(ao)
    acts = csco_actions(ao)
    out["risk_level"] = MulticlassConfusion.from_labels(
        [risk_level(gold_scores.get(s)) for s in tier1], [risk_level(pred_scores.get(s)) for s in tier1], RISK_LEVELS)
    out["action"] = MulticlassConfusion.from_labels(
        [expected_action(gold_scores.get(s)) for s in tier1],
        [acts.get(s) if acts.get(s) in ACTIONS else "MAINTAIN_STANDARD" for s in tier1], ACTIONS)
    out["unknown_actions"] = sorted(a for a in acts.values() if a not in ACTIONS)
    return out


# ---------------------------------------------------------------------------------------------------- aggregation

def mean(v):
    return float(statistics.mean(v)) if v else float("nan")


def sd(v):
    return float(statistics.stdev(v)) if len(v) > 1 else 0.0


def summarise(rows):
    """rows: per-run score dicts with 'scenario'. Thesis aggregation: scenario mean of runs, then across scenarios."""
    by_sid = defaultdict(list)
    for r in rows:
        by_sid[r["scenario"]].append(r)
    keys = ("dm_type", "dm_entity_f1", "dm_ongoing", "dm_scope", "kg_precision", "kg_recall", "kg_f1", "hvr",
            "latency_s", "cost_usd", "tokens")
    out = {"n_scenarios": len(by_sid), "n_runs": len(rows)}
    for k in keys:
        per = [mean([r[k] for r in runs]) for runs in by_sid.values()]
        out[k] = {"mean": mean(per), "sd": sd(per), "within_scenario_sd": mean([sd([r[k] for r in runs])
                                                                               for runs in by_sid.values()])}
    boundaries = sorted({b for r in rows for b in r["handoffs"]})
    out["handoffs"] = {
        b: {"runs_valid": sum(1 for r in rows if r["handoffs"].get(b)),
            "scenarios_valid_on_every_run": sum(1 for runs in by_sid.values() if all(r["handoffs"].get(b) for r in runs))}
        for b in boundaries}
    out["scenarios_all_handoffs_valid_every_run"] = sum(
        1 for runs in by_sid.values() if all(all(r["handoffs"].values()) for r in runs))
    lat = [r["latency_s"] for r in rows]
    tok = [r["tokens"] for r in rows]
    out["latency_runs"] = {"mean": mean(lat), "sd": sd(lat), "min": min(lat), "max": max(lat)}
    out["tokens_runs"] = {"mean": mean(tok), "sd": sd(tok), "min": min(tok), "max": max(tok),
                          "over_budget": sum(t > BUDGET["tokens"] for t in tok)}
    out["cost_runs"] = {"mean": mean([r["cost_usd"] for r in rows]), "max": max(r["cost_usd"] for r in rows),
                        "total": sum(r["cost_usd"] for r in rows)}
    out["budget"] = {"latency_violations": sum(not r["within_latency"] for r in rows),
                     "cost_violations": sum(not r["within_cost"] for r in rows),
                     "token_violations": sum(not r["within_tokens"] for r in rows)}
    return out


def mcc_block(dec_rows, gts):
    """dec_rows: (scenario, run index, decisions). Per run pooled over scenarios; mean and sd over runs; bootstrap."""
    by_sid = defaultdict(list)
    for sid, k, d in dec_rows:
        by_sid[sid].append((k, d))
    sids = sorted(by_sid)
    run_ids = sorted({k for _, k, _ in dec_rows})

    def pooled(units, field):
        per_run = []
        for k in run_ids:
            items = [d[field] for _, runs in units for kk, d in runs if kk == k]
            if not items:
                continue
            if field in ("type", "scope", "alert"):
                labels = sorted({x for pair in items for x in pair}, key=str)
                if field == "alert":
                    conf = Confusion.from_labels([t for t, _ in items], [p for _, p in items])
                else:
                    conf = MulticlassConfusion.from_labels([t for t, _ in items], [p for _, p in items], labels)
            else:
                conf = items[0]
                for x in items[1:]:
                    conf = conf + x
            per_run.append(conf)
        return per_run

    def stat(field, attr):
        def f(units):
            vals = [getattr(c, attr) for c in pooled(units, field)]
            vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
            return mean(vals) if vals else float("nan")
        return f

    articles = defaultdict(list)
    for sid in sids:
        articles[sid.split("_")[-1]].append((sid, by_sid[sid]))
    units = [u for art in sorted(articles) for u in articles[art]]
    art_units = [articles[art] for art in sorted(articles)]
    art_strata = [gts[members[0][0]]["positive"] for members in art_units]
    assert all(len({gts[sid]["positive"] for sid, _ in m}) == 1 for m in art_units)

    def over_articles(f):
        return lambda chosen: f([u for members in chosen for u in members])
    out = {}
    for field in ("alert", "type", "scope", *ENTITY_KINDS, "paths", "risk_level", "action"):
        confs = pooled(units, field)
        block = {"per_run_mcc": [c.mcc for c in confs], "mcc_mean": mean([c.mcc for c in confs]),
                 "mcc_sd": sd([c.mcc for c in confs]), "accuracy_mean": mean([c.accuracy for c in confs])}
        if isinstance(confs[0], Confusion):
            block.update({k: mean([getattr(c, k) for c in confs]) for k in ("precision", "recall", "f1", "specificity")})
            block["pooled_counts_run1"] = confs[0].as_dict()
        else:
            block["labels"] = list(confs[0].labels)
            block["matrix_run1"] = [list(r) for r in confs[0].matrix]
        block["bootstrap_mcc"] = bootstrap(art_units, over_articles(stat(field, "mcc")),
                                           strata=art_strata if field == "alert" else None)
        out[field] = block
    # Baselines on the same decisions
    pos = sum(1 for sid in sids if gts[sid]["positive"])
    out["baselines"] = {
        "always_alert": Confusion(pos, len(sids) - pos, 0, 0).as_dict(),
        "always_none_risk_level_accuracy": mean([
            sum(c.matrix[0]) / c.n if c.n else float("nan") for c in pooled(units, "risk_level")]),
        "always_maintain_accuracy": mean([
            sum(c.matrix[0]) / c.n if c.n else float("nan") for c in pooled(units, "action")]),
        "always_none_and_always_maintain_mcc": 0.0,
    }
    return out


# ---------------------------------------------------------------------------------------------------- main

def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    gts = load_gt()
    net = offline_kg.load()
    resolver = names.Resolver(net)
    scenarios = {s["scenario_id"]: s for s in json.loads(
        (ROOT / "evaluation" / "scenarios" / "multi_company_scenarios.json").read_text())}
    firms = sorted({s["monitored_company"] for s in scenarios.values()})
    ctx = {
        "resolver": resolver,
        "firm_of": {sid: s["monitored_company"] for sid, s in scenarios.items()},
        "vocab": {"countries": entity_keys("countries", {c for cs in net.countries.values() for c in cs}, resolver),
                  "industries": entity_keys("industries", set(net.industry.values()), resolver),
                  "companies": entity_keys("companies", set(net.industry), resolver)},
        "universe": {f: sum(1 for t in range(1, 5) for _ in offline_kg.paths(net, f, t, "simple")) for f in firms},
        "tier1": {f: sorted({normalize(s) for s in net.suppliers.get(f, [])}) for f in firms},
    }

    rows = defaultdict(list)        # (condition, model) -> per-run scores
    dec = defaultdict(list)         # (condition, model) -> (scenario, run key, decisions)
    para = defaultdict(list)
    integrity = Counter()
    for rec, run in runs_of("full"):
        sid, model = rec["scenario_id"], rec["model"]
        cond = f"noise-{rec['variant']}" if rec["condition"] == "noise" else rec["condition"]
        gt = gts[sid]
        ao = run.get("agent_outputs") or {}
        ao = {k: v for k, v in ao.items() if isinstance(v, dict)}
        integrity["runs"] += 1
        integrity["pipeline_errors"] += int(not rec.get("success"))
        s = harness_scores(ao, gt, rec)
        s.update({"run_id": rec["run_id"], "scenario": sid, "condition": cond, "model": model,
                  "variant": rec["variant"], "run_index": rec["run_index"], "success": rec["success"]})
        rows[(cond, model)].append(s)
        d = decisions(ao, gt, ctx, sid)
        integrity["unknown_actions"] += len(d["unknown_actions"])
        harness_counts = dict(x.split("=") for x in s["kg_detail"].split()[:3])
        p = d["paths"]
        if (int(harness_counts["TP"]), int(harness_counts["FP"]), int(harness_counts["FN"])) != (p.tp, p.fp, p.fn):
            integrity["path_counts_differ_from_harness"] += 1
        run_key = rec["variant"] if cond == "paraphrase" else rec["run_index"]
        dec[(cond, model)].append((sid, run_key, d))
        if cond == "paraphrase":
            da = safe_dict(ao.get("disruption_analysis"))
            inv = da.get("involved") or {}
            para[sid].append((normalize(da.get("type") or ""), da.get("scope"),
                              tuple(tuple(sorted(entity_keys(k, inv.get(k) or [], resolver))) for k in ENTITY_KINDS)))

    report = {
        "set": "full", "ground_truth": "evaluation/ground_truth_final", "network": "dataset/supplychainKG_clean.csv",
        "settings": {"path_matching": "exact node sets", "budgets": BUDGET},
        "path_universe": ctx["universe"],
        "integrity": dict(integrity),
        "results": {},
    }
    for key in sorted(rows):
        cond, model = key
        block = {"summary": summarise(rows[key]), "mcc": mcc_block(dec[key], gts)}
        report["results"][f"{cond}/{model}"] = block
    # Paraphrase agreement (Disruption Monitor), as reported in the thesis: the same type, and the same entity sets,
    # across the three paraphrases of an article.
    complete = {sid: v for sid, v in para.items() if len(v) == 3}
    report["paraphrase_agreement"] = {
        "scenarios": len(complete),
        "same_type": sum(1 for v in complete.values() if len({x[0] for x in v}) == 1),
        "same_scope": sum(1 for v in complete.values() if len({x[1] for x in v}) == 1),
        "same_entity_sets": sum(1 for v in complete.values() if len({x[2] for x in v}) == 1),
    }

    out_json = ANALYSIS / "v1_2_full_scores.json"
    out_json.write_text(json.dumps(report, indent=1, default=str))
    with (ANALYSIS / "v1_2_full_runs.csv").open("w", newline="") as f:
        cols = [c for c in next(iter(rows.values()))[0] if c not in ("handoffs", "kg_detail")]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows):
            for r in sorted(rows[key], key=lambda r: r["run_id"]):
                w.writerow(r)
    print(f"Wrote {out_json}")
    for name, block in report["results"].items():
        s, m = block["summary"], block["mcc"]
        print(f"{name}: {s['n_runs']} runs, {s['n_scenarios']} scenarios; detection MCC {m['alert']['mcc_mean']:.3f}; "
              f"path MCC {m['paths']['mcc_mean']:.3f}; risk level MCC {m['risk_level']['mcc_mean']:.3f}; "
              f"action MCC {m['action']['mcc_mean']:.3f}")


if __name__ == "__main__":
    main()
