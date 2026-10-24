#!/usr/bin/env python3
"""
Chapter 5 measures of the 630 runs, from the stored run records and outputs. No model is called.

    python scripts/ch5_analysis_v1_2.py

Reads evaluation/results/v1_2_full/ (run records and complete outputs), the scenario references and the per-run scores of
scripts/score_v1_2.py, and writes evaluation/results/analysis/v1_2_ch5_analysis.json.

Agent level. An agent is correct on a run when its output equals the correct output for the input it received:
  - Disruption Monitor (input: the article): ongoing decision, type, scope and the resolved country, industry and
    company sets all equal to the reference;
  - Knowledge Graph Query: the Disruption Monitor's values are passed to the path tool unchanged and the paths equal
    those the tool returns for them;
  - Risk Manager: the answer is the risk tool's result and its scores equal the offline scores of the paths received;
  - CSCO: every action follows the decision rule from the scores received, every quoted score is the score received,
    and no supplier outside the assessment is listed.
The first agent that is not correct on its input is the origin of a failing run.

System level. A run decides correctly when the CSCO action for every Tier-1 supplier of the monitored firm equals the
reference action (an unlisted supplier is MAINTAIN_STANDARD) and no other supplier receives an action. It succeeds
under constraints when it also stays within 600 s, $0.20 and 30,000 tokens.

Repeatability. For c_i successes in K = 5 runs of scenario i, pass^k = mean_i C(c_i, k) / C(K, k) (Yao et al., 2025).

Comparisons. Differences between conditions or models are paired by scenario: d_i is the difference of the scenario
means, and the 95% interval is the percentile interval of mean(d) over 10,000 bootstrap resamples of articles
(seed 20261018). The three firm scenarios of an article share its text, its variants and almost all of their network,
so an article is resampled with all its scenarios and runs. Rates carry the same article bootstrap interval.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from analysis.metrics import entity_keys, expected_action, normalize  # noqa: E402
from offline import kg as offline_kg, names  # noqa: E402

RUNS = ROOT / "evaluation" / "results" / "v1_2_full"
GT_DIR = ROOT / "evaluation" / "ground_truth_final"
ANALYSIS = ROOT / "evaluation" / "results" / "analysis"
ACTIONS = ("MAINTAIN_STANDARD", "INCREASE_MONITORING", "REPLACE_SUPPLIER")
KINDS = ("countries", "industries", "companies")
MODELS = ("gpt-4o", "gpt-4.1", "gpt-5-mini")
K = 5
N_BOOT, SEED = 10_000, 20261018


def parse(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return {}
    return x if isinstance(x, dict) else {}


# ---------------------------------------------------------------------------------------------------- per run

def run_measures(rec, ao, gt, tier1, resolver, score_row):
    c = rec["comparison"]
    da = parse(ao.get("disruption_analysis"))
    gda = gt["disruption_analysis"]
    inv = da.get("involved") or {}
    fields = {
        "ongoing": da.get("ongoing_disruption") is gda["ongoing_disruption"],
        "type": normalize(da.get("type") or "") == normalize(gda["type"]),
        "scope": (da.get("scope") == gda["scope"]) if gda["ongoing_disruption"] else True,
        **{k: entity_keys(k, inv.get(k) or [], resolver) == entity_keys(k, gda["involved"][k], resolver)
           for k in KINDS},
    }
    ok = {"dm": all(fields.values()), "kg": bool(c["kg_query"]["pass"]), "risk": bool(c["risk_manager"]["pass"]),
          "csco": bool(c["csco"]["pass"])}
    origin = next((a for a in ("dm", "kg", "risk", "csco") if not ok[a]), "none")

    cs = parse(ao.get("chief_supply_chain_output"))
    listed = {normalize(k): (v.get("action") if isinstance(v, dict) else None)
              for k, v in (cs.get("decisions") or {}).items()}
    gold = {normalize(k): v for k, v in gt["risk_assessment"]["supplier_risk_scores"].items()}
    action_errors = [s for s in tier1 if (listed.get(s) if listed.get(s) in ACTIONS else "MAINTAIN_STANDARD")
                     != expected_action(gold.get(s))]
    outside = [s for s, a in listed.items() if s not in tier1 and a != "MAINTAIN_STANDARD"]
    decision_ok = not action_errors and not outside
    budget_ok = bool(score_row["within_latency"]) and bool(score_row["within_cost"]) and bool(score_row["within_tokens"])

    kq = c["kg_query"]
    rm = c["risk_manager"]
    csv_in = c["csco"]["vs_assessment_received"]
    return {
        "scenario": rec["scenario_id"], "run_index": rec["run_index"], "variant": rec["variant"],
        "fields": fields, "ok": ok, "origin": origin,
        "paths_equal_reference": bool(kq["vs_ground_truth"]["identical"]),
        "scores_equal_reference": bool(rm["vs_ground_truth"]["identical"]),
        "decision_ok": decision_ok, "action_errors": len(action_errors), "outside_actions": len(outside),
        "budget_ok": budget_ok, "constrained": decision_ok and budget_ok,
        "within": {k: bool(score_row[f"within_{k}"]) for k in ("latency", "cost", "tokens")},
        "fidelity": {"kg_relay": all(kq["relay"].values()), "risk_answer_is_tool_result": bool(rm["answer_equals_tool_file"]),
                     "csco_quotes": not csv_in["score_quote_errors"],
                     "csco_known_suppliers": not csv_in["suppliers_not_in_assessment"]},
        "csco_errors_on_input": [{"supplier": e["supplier"], "expected": e["expected"], "csco": e["csco"],
                                  "score": e["score"], "listed": e["listed"]} for e in csv_in["action_errors"]],
        "unresolved": {k: v for k, v in (kq.get("unresolved") or {}).items() if v},
        "hvr": score_row["hvr"], "tokens": score_row["tokens"], "latency_s": score_row["latency_s"],
        "cost_usd": score_row["cost_usd"], "dm_entity_f1": score_row["dm_entity_f1"], "kg_f1": score_row["kg_f1"],
        "dm_tuple": (normalize(da.get("type") or ""), da.get("scope"),
                     tuple(tuple(sorted(entity_keys(k, inv.get(k) or [], resolver))) for k in KINDS)),
    }


# ---------------------------------------------------------------------------------------------------- statistics

RNG_INDEX = None


def boot_index(n):
    global RNG_INDEX
    if RNG_INDEX is None or RNG_INDEX.shape[1] != n:
        RNG_INDEX = np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))
    return RNG_INDEX


def article(sid):
    return sid.split("_")[-1]


def ci(values, sids):
    """Percentile interval of the mean over scenarios, resampling articles with all their scenarios."""
    arts = sorted({article(s) for s in sids})
    v = np.asarray(values, dtype=float)
    sums = np.array([v[[i for i, s in enumerate(sids) if article(s) == a]].sum() for a in arts])
    counts = np.array([sum(article(s) == a for s in sids) for a in arts], dtype=float)
    idx = boot_index(len(arts))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def by_scenario(rows, key):
    d = defaultdict(list)
    for r in rows:
        d[r["scenario"]].append(float(key(r)))
    return {s: st.mean(v) for s, v in sorted(d.items())}


def rate(rows, key):
    per = by_scenario(rows, key)
    return {"runs": int(sum(bool(key(r)) for r in rows)), "n": len(rows), "rate": st.mean(per.values()),
            "ci95": ci(list(per.values()), list(per))}


def paired(rows_b, rows_a, key, scenarios=None):
    """Mean over scenarios of (mean of b) - (mean of a), with an article bootstrap interval."""
    b, a = by_scenario(rows_b, key), by_scenario(rows_a, key)
    sids = sorted(set(a) & set(b) if scenarios is None else scenarios)
    d = [b[s] - a[s] for s in sids]
    return {"diff": st.mean(d), "ci95": ci(d, sids), "n_scenarios": len(sids)}


def pass_k(rows, key):
    c = defaultdict(int)
    n = defaultdict(int)
    for r in rows:
        c[r["scenario"]] += int(bool(key(r)))
        n[r["scenario"]] += 1
    assert set(n.values()) == {K}
    cs = [c[s] for s in sorted(c)]
    return {"pass^k": {k: sum(math.comb(x, k) / math.comb(K, k) for x in cs) / len(cs) for k in range(1, K + 1)},
            "successes_per_scenario": {i: cs.count(i) for i in range(K + 1)},
            "never": sorted(s for s in c if c[s] == 0), "varying": sorted(s for s in c if 0 < c[s] < K)}


def agreement(groups):
    return {"type": sum(len({t[0] for t in g}) == 1 for g in groups),
            "scope": sum(len({t[1] for t in g}) == 1 for g in groups),
            "entities": sum(len({t[2] for t in g}) == 1 for g in groups), "n": len(groups)}


# ---------------------------------------------------------------------------------------------------- blocks

def agent_block(rows):
    out = {"n_runs": len(rows),
           "correct_on_input": {a: rate(rows, lambda r, a=a: r["ok"][a]) for a in ("dm", "kg", "risk", "csco")},
           "dm_fields": {f: int(sum(r["fields"][f] for r in rows)) for f in rows[0]["fields"]},
           "origin": dict(Counter(r["origin"] for r in rows)),
           "csco_errors_on_input": [{"run": f"{r['scenario']}/r{r['run_index']}", "errors": r["csco_errors_on_input"],
                                     "dm_scope_ok": r["fields"]["scope"], "dm_ok": r["ok"]["dm"]}
                                    for r in rows if r["csco_errors_on_input"]]}
    # all agents correct on their input implies the reference decisions; checked, not assumed
    out["all_correct_but_decision_wrong"] = sum(1 for r in rows if r["origin"] == "none" and not r["decision_ok"])
    return out


def propagation_block(rows):
    wrong = [r for r in rows if not r["ok"]["dm"]]
    kinds = Counter()
    for r in wrong:
        kinds[tuple(f for f, v in r["fields"].items() if not v)] += 1
    by_kind = defaultdict(lambda: {"runs": 0, "paths_wrong": 0, "decisions_wrong": 0})
    for r in wrong:
        key = "+".join(f for f, v in r["fields"].items() if not v)
        by_kind[key]["runs"] += 1
        by_kind[key]["paths_wrong"] += int(not r["paths_equal_reference"])
        by_kind[key]["decisions_wrong"] += int(not r["decision_ok"])
    # each field: runs in which it is wrong, and in how many of those the paths / decisions are wrong
    per_field = {}
    for f in rows[0]["fields"]:
        rs = [r for r in wrong if not r["fields"][f]]
        per_field[f] = {"runs": len(rs), "paths_wrong": sum(not r["paths_equal_reference"] for r in rs),
                        "decisions_wrong": sum(not r["decision_ok"] for r in rs)}
    changed = [r for r in wrong if not r["paths_equal_reference"]]
    return {
        "two_stage": {"dm_wrong": len(wrong), "paths_changed": len(changed),
                      "decisions_changed_after_paths_changed": sum(not r["decision_ok"] for r in changed),
                      "decisions_changed_with_reference_paths": sum(
                          1 for r in rows if r["paths_equal_reference"] and not r["decision_ok"])},
        "decision_wrong_origin": dict(Counter(r["origin"] for r in rows if not r["decision_ok"])),
        "field_errors_by_scenario": {f: dict(sorted(Counter(r["scenario"] for r in rows if not r["fields"][f]).items()))
                                     for f in rows[0]["fields"]},
        "dm_wrong": len(wrong),
        "dm_wrong_paths_wrong": sum(not r["paths_equal_reference"] for r in wrong),
        "dm_wrong_decisions_wrong": sum(not r["decision_ok"] for r in wrong),
        "dm_wrong_decisions_right": sum(r["decision_ok"] for r in wrong),
        "dm_right_paths_wrong": sum(1 for r in rows if r["ok"]["dm"] and not r["paths_equal_reference"]),
        "paths_wrong": sum(not r["paths_equal_reference"] for r in rows),
        "paths_wrong_scope_right": sum(1 for r in rows if not r["paths_equal_reference"] and r["fields"]["scope"]),
        "paths_right_decisions_wrong": sum(1 for r in rows if r["paths_equal_reference"] and not r["decision_ok"]),
        "paths_wrong_decisions_right": sum(1 for r in rows if not r["paths_equal_reference"] and r["decision_ok"]),
        "by_error_combination": dict(sorted(by_kind.items(), key=lambda kv: -kv[1]["runs"])),
        "by_field": per_field,
    }


def interaction_block(rows):
    return {"hvr_all_valid": sum(r["hvr"] == 1.0 for r in rows), "n": len(rows),
            "fidelity": {k: sum(r["fidelity"][k] for r in rows) for k in rows[0]["fidelity"]},
            "runs_with_unresolved_names": sum(bool(r["unresolved"]) for r in rows),
            "unresolved_names": dict(Counter(n for r in rows for v in r["unresolved"].values() for n in v))}


def system_block(rows, gt):
    tok = defaultdict(list)
    for r in rows:
        tok[r["scenario"]].append(r["tokens"])
    n_scored = {s: len(gt[s]["risk_assessment"]["supplier_risk_scores"]) for s in tok}
    xs = [n_scored[r["scenario"]] for r in rows]
    ys = [r["tokens"] for r in rows]
    over = Counter(r["scenario"] for r in rows if not r["within"]["tokens"])
    budget_only = [r for r in rows if r["decision_ok"] and not r["budget_ok"]]
    return {
        "decision_ok": rate(rows, lambda r: r["decision_ok"]),
        "constrained": rate(rows, lambda r: r["constrained"]),
        "violations": {k: sum(not r["within"][k] for r in rows) for k in ("latency", "cost", "tokens")},
        "decision_ok_but_over_budget": len(budget_only),
        "decision_ok_but_over_budget_tokens": [min((r["tokens"] for r in budget_only), default=None),
                                               max((r["tokens"] for r in budget_only), default=None)],
        "decision_ok_but_over_budget_by_scenario": dict(Counter(r["scenario"] for r in budget_only)),
        "over_token_budget_by_scenario": dict(sorted(over.items())),
        "tokens": {"mean": st.mean(ys), "min": min(ys), "max": max(ys),
                   "within_scenario_sd": st.mean(st.pstdev(v) for v in tok.values()),
                   "between_scenario_sd": st.pstdev([st.mean(v) for v in tok.values()]),
                   "corr_with_reference_scored_suppliers": float(np.corrcoef(xs, ys)[0, 1])},
        "latency_s": {"mean": st.mean(r["latency_s"] for r in rows), "min": min(r["latency_s"] for r in rows),
                      "max": max(r["latency_s"] for r in rows)},
        "cost_usd": {"mean": st.mean(r["cost_usd"] for r in rows), "max": max(r["cost_usd"] for r in rows),
                     "total": sum(r["cost_usd"] for r in rows)},
    }


# ---------------------------------------------------------------------------------------------------- main

def main():
    net = offline_kg.load()
    resolver = names.Resolver(net)
    gt = {p.stem: json.loads(p.read_text()) for p in sorted(GT_DIR.glob("*_SC0*.json"))}
    scen = {s["scenario_id"]: s for s in json.loads(
        (ROOT / "evaluation" / "scenarios" / "multi_company_scenarios.json").read_text())}
    tier1 = {f: {normalize(s) for s in net.suppliers.get(f, [])} for f in {s["monitored_company"] for s in scen.values()}}
    score_rows = {}
    with (ANALYSIS / "v1_2_full_runs.csv").open() as f:
        for r in csv.DictReader(f):
            score_rows[r["run_id"]] = {k: (v == "True") if v in ("True", "False") else
                                       (float(v) if k in ("hvr", "tokens", "latency_s", "cost_usd", "dm_entity_f1",
                                                          "kg_f1") else v) for k, v in r.items()}

    rows = defaultdict(list)
    for p in sorted(RUNS.glob("*/*/*.json")):
        rec = json.loads(p.read_text())
        assert rec["status"] == "completed" and rec["success"], rec["run_id"]
        with gzip.open(RUNS / "full" / f"{rec['run_id'].replace('/', '__')}.json.gz", "rt") as f:
            ao = json.load(f).get("agent_outputs") or {}
        cond = f"noise-{rec['variant']}" if rec["condition"] == "noise" else rec["condition"]
        sid = rec["scenario_id"]
        rows[(cond, rec["model"])].append(
            run_measures(rec, ao, gt[sid], tier1[scen[sid]["monitored_company"]], resolver, score_rows[rec["run_id"]]))

    report = {"n_runs": sum(len(v) for v in rows.values()), "bootstrap": {"resamples": N_BOOT, "seed": SEED},
              "models": {}, "comparisons": {}}
    for model in MODELS:
        R = rows[("original", model)]
        report["models"][model] = {
            "agent": agent_block(R), "propagation": propagation_block(R), "interaction": interaction_block(R),
            "system": system_block(R, gt),
            "repeatability": {k: pass_k(R, f) for k, f in (("dm_ok", lambda r: r["ok"]["dm"]),
                                                            ("decision_ok", lambda r: r["decision_ok"]),
                                                            ("constrained", lambda r: r["constrained"]))},
        }

    base = rows[("original", "gpt-4o")]
    metrics = {"dm_ok": lambda r: r["ok"]["dm"], "type": lambda r: r["fields"]["type"],
               "scope": lambda r: r["fields"]["scope"],
               "entities": lambda r: all(r["fields"][k] for k in KINDS), "dm_entity_f1": lambda r: r["dm_entity_f1"],
               "decision_ok": lambda r: r["decision_ok"], "constrained": lambda r: r["constrained"]}
    for model in MODELS[1:]:
        report["comparisons"][f"{model} - gpt-4o"] = {m: paired(rows[("original", model)], base, f)
                                                      for m, f in metrics.items()}

    # Article variants (GPT-4o): paired against the mean of the five runs on the original article.
    noisy_text = {}
    for sid in scen:
        nf = json.loads((ROOT / f"evaluation/results/noise_sensitivity/noisy_articles/{sid}_noisy.json").read_text())
        noisy_text[sid] = nf["noisy_versions"]
    variants = {}
    for cond in ("paraphrase", "noise-mild", "noise-moderate", "noise-high"):
        R = rows[(cond, "gpt-4o")]
        excluded = []
        if cond.startswith("noise-"):
            level = cond.split("-", 1)[1]
            for sid in sorted(scen):
                names_lost = [c for c in gt[sid]["disruption_analysis"]["involved"]["companies"] + gt[sid][
                    "disruption_analysis"]["involved"]["countries"]
                              if c.lower() in scen[sid]["news_article"].lower()
                              and c.lower() not in noisy_text[sid][level].lower()]
                if names_lost:
                    excluded.append({"scenario": sid, "names_no_longer_stated": names_lost})
        keep = sorted({r["scenario"] for r in R} - {e["scenario"] for e in excluded})
        Rk = [r for r in R if r["scenario"] in keep]
        variants[cond] = {"n_runs": len(Rk), "excluded": excluded,
                          "rates": {m: rate(Rk, f) for m, f in metrics.items() if m != "dm_entity_f1"},
                          "paired_vs_original": {m: paired(Rk, base, f, keep) for m, f in metrics.items()},
                          "violations": {k: sum(not r["within"][k] for r in Rk) for k in ("latency", "cost", "tokens")},
                          "latency_max_s": max(r["latency_s"] for r in Rk)}
    report["variants"] = variants

    # Agreement of the three paraphrases against the agreement of three runs on the same article (all ten triples).
    para = defaultdict(list)
    for r in rows[("paraphrase", "gpt-4o")]:
        para[r["scenario"]].append(r["dm_tuple"])
    orig = defaultdict(dict)
    for r in base:
        orig[r["scenario"]][r["run_index"]] = r["dm_tuple"]
    triples = [agreement([[orig[s][i] for i in idx] for s in sorted(orig)]) for idx in combinations(range(1, K + 1), 3)]
    report["paraphrase_agreement"] = {
        "paraphrases": agreement(list(para.values())),
        "three_runs_same_article_mean": {k: st.mean(t[k] for t in triples) for k in ("type", "scope", "entities")},
        "three_runs_same_article_range": {k: [min(t[k] for t in triples), max(t[k] for t in triples)]
                                          for k in ("type", "scope", "entities")},
    }

    report["reference"] = {
        "scenarios": len(gt), "positives": sum(bool(g["risk_assessment"]["supplier_risk_scores"]) for g in gt.values()),
        "types": dict(Counter(g["disruption_analysis"]["type"] for g in gt.values())),
        "scope": dict(Counter(g["disruption_analysis"]["scope"] for g in gt.values())),
        "scored_suppliers_positive": [min(len(g["risk_assessment"]["supplier_risk_scores"]) for g in gt.values()
                                          if g["risk_assessment"]["supplier_risk_scores"]),
                                      max(len(g["risk_assessment"]["supplier_risk_scores"]) for g in gt.values())],
        "reference_actions": dict(Counter(a for g in gt.values() for a in g["expected_decisions"].values())),
    }
    out = ANALYSIS / "v1_2_ch5_analysis.json"
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"Wrote {out}")
    for model in MODELS:
        m = report["models"][model]
        print(model, "on input", {a: v["runs"] for a, v in m["agent"]["correct_on_input"].items()},
              "origin", m["agent"]["origin"], "decision", m["system"]["decision_ok"]["runs"],
              "constrained", m["system"]["constrained"]["runs"])


if __name__ == "__main__":
    main()
