"""Scores of the Chapter 3 prompt variants and other models (Section 3.4.2 and Table 3.4).

    python chapter3/score_reruns.py

Scores every arm of rerun_extraction.py, and the ten stored GPT-4o runs, against the reference annotation
(chapter3/reference_final): precision, recall and F1 as reported in the thesis (distinct items pooled over documents;
mean and population standard deviation over runs), and MCC and F1 on the decisions of score_ch3.py with 95% bootstrap
intervals over documents. Each arm is compared with the prompt as used on the same documents by the difference in
mean MCC and F1, with a paired bootstrap over documents. "comparison_documents" excludes the two demonstration
documents and is the basis of Table 3.4; "all_documents" includes them, for the arms without demonstrations.

Writes chapter3/results/reruns_scores.json. No model is called.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rerun_extraction import ARMS, OUT, REPO, RUNS  # noqa: E402
from score_ch3 import (DEFAULT_BOOT, DEFAULT_SEED, NODE_TYPES, REL_TYPES, doc_counts, keep,  # noqa: E402
                       mcc_arrays, parse_file, thesis_prf)

ROWS = ["NER overall"] + [f"NER {t}" for t in NODE_TYPES] + ["RE overall"] + [f"RE {t}" for t in REL_TYPES]


def load_runs(run_dirs, docs):
    runs, missing = [], []
    for rd in run_dirs:
        run = {}
        for d in docs:
            path = next((rd / n for n in (f"Nodes_Rels_{d}.txt", f"Nodes_Rels_{d}..txt") if (rd / n).exists()), None)
            if path is None:
                missing.append(f"{rd.name}/{d}")
            run[d] = keep(*parse_file(path)) if path else ([], [])
        runs.append(run)
    return runs, missing


def counts_array(docs, gt, runs):
    """{row: run x doc x (tp, fp, fn, tn)}"""
    c = [[doc_counts(*gt[d], *run[d]) for d in docs] for run in runs]
    return {row: np.array([[[x[row].tp, x[row].fp, x[row].fn, x[row].tn] for x in run] for run in c], float)
            for row in ROWS}


def stats(a, idx):
    """Mean over runs of MCC and F1 pooled over the documents idx (with repeats)."""
    s = a[:, idx, :].sum(axis=1)
    tp, fp, fn, tn = s.T
    mcc = mcc_arrays(tp, fp, fn, tn)
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return (float(np.nanmean(mcc)) if not np.all(np.isnan(mcc)) else math.nan,
            float(np.nanmean(f1)) if not np.all(np.isnan(f1)) else math.nan)


def interval(values):
    v = np.array([x for x in values if not math.isnan(x)])
    return [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))] if len(v) else [math.nan, math.nan]


def evaluate(docs, gt, runs, base=None):
    a = counts_array(docs, gt, runs)
    b = counts_array(docs, gt, base) if base is not None else None
    rng = np.random.default_rng(DEFAULT_SEED)
    draws = [rng.integers(0, len(docs), len(docs)) for _ in range(DEFAULT_BOOT)]
    thesis = thesis_prf(docs, gt, runs)
    out = {}
    for row in ROWS:
        point = stats(a[row], np.arange(len(docs)))
        boot = [stats(a[row], i) for i in draws]
        s = a[row].sum(axis=1)
        per_run = mcc_arrays(*s.T)
        res = {"thesis_prf": thesis[row], "mcc": point[0],
               "mcc_sd": float(np.nanstd(per_run)) if not np.all(np.isnan(per_run)) else math.nan,
               "mcc_95": interval([x[0] for x in boot]),
               "f1_decisions": point[1], "f1_decisions_95": interval([x[1] for x in boot])}
        if b is not None:
            bp = stats(b[row], np.arange(len(docs)))
            diffs = [(x[0] - y[0], x[1] - y[1]) for x, y in zip(boot, (stats(b[row], i) for i in draws))]
            res["minus_as_used"] = {"mcc": point[0] - bp[0], "mcc_95": interval([d[0] for d in diffs]),
                                    "f1": point[1] - bp[1], "f1_95": interval([d[1] for d in diffs])}
        out[row] = res
    return out


def main():
    argparse.ArgumentParser(description=__doc__.split("\n")[0]).parse_args()
    gt_dir = REPO / "chapter3" / "reference_final"
    all_docs = sorted(f.name[len("GT_Nodes_Rels_"):-len(".txt")] for f in gt_dir.glob("GT_Nodes_Rels_*.txt"))
    gt = {d: keep(*parse_file(gt_dir / f"GT_Nodes_Rels_{d}.txt")) for d in all_docs}
    demo = json.loads((OUT / "demonstration_documents.json").read_text())
    scored = [d for d in all_docs if d not in demo]
    stored = sorted((p for p in (REPO / "chapter3" / "stored_runs" / "runs").iterdir() if p.name.startswith("run_")),
                    key=lambda p: int(p.name.split("_")[1]))
    calls = [json.loads(line) for line in (OUT / "calls.jsonl").open()]
    arms = {"stored_runs": stored} | {a: [OUT / a / f"run_{k}" for k in range(1, RUNS + 1)] for a in ARMS}

    report = {"reference": "chapter3/reference_final", "demonstration_documents": demo,
              "settings": {"bootstrap": f"{DEFAULT_BOOT} resamples over documents, seed {DEFAULT_SEED}, paired across arms",
                           "sd": "population, over runs"},
              "arms": {}}
    loaded = {}
    for docs_name, docs in (("comparison_documents", scored), ("all_documents", all_docs)):
        for arm, dirs in arms.items():
            if arm == "demonstrations" and docs_name == "all_documents":
                continue
            loaded[arm, docs_name] = load_runs(dirs, docs)
    for (arm, docs_name), (runs, missing) in loaded.items():
        docs = scored if docs_name == "comparison_documents" else all_docs
        base = loaded["as_used", docs_name][0] if arm != "as_used" else None
        entry = report["arms"].setdefault(arm, {})
        entry[docs_name] = evaluate(docs, gt, runs, base)
        entry[docs_name + "_outputs_missing"] = missing
        if arm in ARMS:
            mine = [c for c in calls if c["arm"] == arm]
            entry["calls"] = {"n": len(mine), "served_models": dict(Counter(c["served_model"] for c in mine)),
                              "finish_reasons": dict(Counter(c["finish_reason"] for c in mine)),
                              "usd": round(sum(c["usd"] for c in mine), 4),
                              "model": ARMS[arm][0], "prompt": ARMS[arm][1], "temperature": ARMS[arm][2]}
            entry["outputs_with_no_items"] = sum(1 for run in runs for d in docs if not run[d][0] and not run[d][1])
    path = OUT.parent / "reruns_scores.json"
    path.write_text(json.dumps(report, indent=1))
    print("Wrote", path)
    for arm, e in report["arms"].items():
        r = e["comparison_documents"]
        line = f"{arm:15s}"
        for row in ("NER overall", "RE overall"):
            t = r[row]["thesis_prf"]["f1"]
            line += f"  {row} F1 {t['mean']:.3f}±{t['sd']:.3f} MCC {r[row]['mcc']:.3f} [{r[row]['mcc_95'][0]:.3f}, {r[row]['mcc_95'][1]:.3f}]"
            if "minus_as_used" in r[row]:
                m = r[row]["minus_as_used"]
                line += f" Δ{m['mcc']:+.3f} [{m['mcc_95'][0]:+.3f}, {m['mcc_95'][1]:+.3f}]"
        print(line, "| missing", len(e["comparison_documents_outputs_missing"]), "| finish", e.get("calls", {}).get("finish_reasons"))


if __name__ == "__main__":
    main()
