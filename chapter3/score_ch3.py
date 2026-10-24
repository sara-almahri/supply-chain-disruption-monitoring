"""Chapter 3 entity and relation extraction scores (Section 3.4 and Table 3.1).

    python chapter3/score_ch3.py [--source chapter3/stored_runs]

Scores the ten stored GPT-4o runs (chapter3/stored_runs/runs/run_1 ... run_10) against the reference annotation
(chapter3/reference_final). Precision, recall and F1 pool the distinct items of all documents into one set per run, as
in the thesis; MCC uses the units of decision of analysis/METRICS.md (Section 3), with a 95% bootstrap interval over
documents and a chance baseline for every type.

Units of decision:
  entities   (document, candidate entity, entity type); candidates are the distinct names in the reference or the
             output of that document
  relations  (document, ordered pair of candidate nodes, relation type); candidate nodes are the distinct (name, type)
             pairs in the reference or the output of that document, including relation end points

Chance baseline: for each type, a classifier that assigns the type at random to the candidates of each document, as
often as the system does in that document. Its true-positive count is hypergeometric, so the null distribution of
MCC is drawn exactly, without a permutation loop.

Writes chapter3/results/ch3_scores.json. No model is called.
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from analysis.metrics import DEFAULT_BOOT, DEFAULT_SEED, Confusion, bootstrap  # noqa: E402

ALLOWED_RELS = {"produces", "suppliesto", "locatedin", "owns"}
DROP_TYPES = {"organization", "brand"}
NODE_TYPES = ["company", "product", "location", "material", "person", "mine"]
REL_TYPES = ["locatedin", "owns", "produces", "suppliesto"]
REFERENCE = REPO / "chapter3" / "reference_final"
STORED_RUNS = REPO / "chapter3" / "stored_runs"

def normalize(text):
    return text.strip().lower()


def parse_file(path):
    """Nodes and relationships of an output or reference file."""
    nodes, rels, section = [], [], None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("nodes:"):
                section = "nodes"
                continue
            if line.lower().startswith("relationships:"):
                section = "relationships"
                continue
            if section == "nodes":
                m = re.match(r"Node ID:\s*(.*?),\s*Type:\s*(.*)", line, re.IGNORECASE)
                if m:
                    nodes.append((normalize(m.group(1)), normalize(m.group(2))))
            elif section == "relationships":
                m = re.match(r"Source:\s*(.*?),\s*Source Type:\s*(.*?),\s*Relationship:\s*(.*?),"
                             r"\s*Target:\s*(.*?),\s*Target Type:\s*(.*)", line, re.IGNORECASE)
                if m:
                    rels.append(tuple(normalize(m.group(i)) for i in range(1, 6)))
    return nodes, rels


def keep(nodes, rels):
    return [n for n in nodes if n[1] not in DROP_TYPES], [r for r in rels if r[2] in ALLOWED_RELS]


def load(source, gt_dir):
    """Ground truth and the ten runs, per document: {doc: (nodes, rels)} and [{doc: (nodes, rels)}, ...]."""
    gt, runs = {}, []
    docs = sorted(f[len("GT_Nodes_Rels_"):-len(".txt")] for f in os.listdir(gt_dir)
                  if f.startswith("GT_Nodes_Rels_") and f.endswith(".txt"))
    for d in docs:
        gt[d] = keep(*parse_file(gt_dir / f"GT_Nodes_Rels_{d}.txt"))
    run_dirs = sorted((p for p in (source / "runs").iterdir() if p.name.startswith("run_")),
                      key=lambda p: int(p.name.split("_")[1]))
    missing = defaultdict(list)
    for rd in run_dirs:
        run = {}
        for d in docs:
            names = [f"Nodes_Rels_{d}.txt", f"Nodes_Rels_{d}..txt"]
            path = next((rd / n for n in names if (rd / n).exists()), None)
            if path is None:
                missing[rd.name].append(d)
                run[d] = ([], [])
            else:
                run[d] = keep(*parse_file(path))
        runs.append(run)
    return docs, gt, runs, dict(missing)


# ---------------------------------------------------------------------------------------------------------------------
# Precision, recall and F1 as reported in the thesis (distinct items pooled over documents)
# ---------------------------------------------------------------------------------------------------------------------
def prf(g, o):
    tp, fp, fn = len(g & o), len(o - g), len(g - o)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def thesis_prf(docs, gt, runs):
    g_nodes = {n for d in docs for n in gt[d][0]}
    g_rels = {r for d in docs for r in gt[d][1]}
    rows = defaultdict(list)
    for run in runs:
        o_nodes = {n for d in docs for n in run[d][0]}
        o_rels = {r for d in docs for r in run[d][1]}
        rows["NER overall"].append(prf(g_nodes, o_nodes))
        rows["RE overall"].append(prf(g_rels, o_rels))
        for t in NODE_TYPES:
            rows[f"NER {t}"].append(prf({n for n in g_nodes if n[1] == t}, {n for n in o_nodes if n[1] == t}))
        for t in REL_TYPES:
            rows[f"RE {t}"].append(prf({r for r in g_rels if r[2] == t}, {r for r in o_rels if r[2] == t}))
    out = {}
    for k, vals in rows.items():
        cols = list(zip(*vals))
        out[k] = {m: {"mean": statistics.mean(c), "sd": statistics.pstdev(c)} for m, c in zip(("p", "r", "f1"), cols)}
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Decisions of METRICS.md, Section 3
# ---------------------------------------------------------------------------------------------------------------------
def doc_counts(g_nodes, g_rels, o_nodes, o_rels):
    """{row: (tp, fp, fn, tn, n_predicted, n_positive, n_decisions)} for one document and one run."""
    g_nodes, o_nodes, g_rels, o_rels = set(g_nodes), set(o_nodes), set(g_rels), set(o_rels)
    out = {}
    names = {n for n, _ in g_nodes | o_nodes}
    for t in NODE_TYPES:
        g = {n for n, tt in g_nodes if tt == t}
        o = {n for n, tt in o_nodes if tt == t}
        out[f"NER {t}"] = Confusion.from_matches(len(g & o), len(o - g), len(g - o), len(names))
    nodes = g_nodes | o_nodes | {(r[0], r[1]) for r in g_rels | o_rels} | {(r[3], r[4]) for r in g_rels | o_rels}
    pairs = len(nodes) * (len(nodes) - 1)
    for t in REL_TYPES:
        g = {r for r in g_rels if r[2] == t}
        o = {r for r in o_rels if r[2] == t}
        loops = sum(1 for r in g | o if (r[0], r[1]) == (r[3], r[4]))
        out[f"RE {t}"] = Confusion.from_matches(len(g & o), len(o - g), len(g - o), pairs + loops)
    out["NER overall"] = sum((out[f"NER {t}"] for t in NODE_TYPES), Confusion())
    out["RE overall"] = sum((out[f"RE {t}"] for t in REL_TYPES), Confusion())
    return out


def mcc_arrays(tp, fp, fn, tn):
    pos, neg, pp, pn = tp + fn, tn + fp, tp + fp, tn + fn
    den = np.sqrt(pp * pos * neg * pn)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(den > 0, (tp * tn - fp * fn) / np.where(den > 0, den, 1), 0.0)
    return np.where((pos == 0) | (neg == 0), np.nan, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=STORED_RUNS, help="folder holding runs/run_<k>")
    args = ap.parse_args()
    gt_dir = REFERENCE
    docs, gt, runs, missing = load(args.source, gt_dir)

    thesis = thesis_prf(docs, gt, runs)

    counts = [[doc_counts(*gt[d], *run[d]) for d in docs] for run in runs]  # [run][doc][row]
    rows = list(counts[0][0])
    arr = {row: np.array([[[c[row].tp, c[row].fp, c[row].fn, c[row].tn] for c in run] for run in counts], float)
           for row in rows}  # run x doc x 4

    def pooled_mcc(row, doc_idx):
        s = arr[row][:, doc_idx, :].sum(axis=1)  # run x 4
        return float(np.nanmean(mcc_arrays(*s.T))) if not np.all(np.isnan(mcc_arrays(*s.T))) else math.nan

    results = {}
    rng = np.random.default_rng(DEFAULT_SEED)
    all_idx = np.arange(len(docs))
    for row in rows:
        per_run = [sum((counts[r][d][row] for d in range(len(docs))), Confusion()) for r in range(len(runs))]
        stat = lambda units, row=row: pooled_mcc(row, np.array(units))
        boot = bootstrap(list(all_idx), stat)
        res = {
            "mcc": {"mean": statistics.mean(c.mcc for c in per_run), "sd": statistics.pstdev(c.mcc for c in per_run),
                    "bootstrap_95": [boot["lo"], boot["hi"]], "bootstrap_undefined": boot["n_undefined"]},
            "same_decisions": {m: {"mean": statistics.mean(getattr(c, m) for c in per_run),
                                   "sd": statistics.pstdev(getattr(c, m) for c in per_run)}
                               for m in ("precision", "recall", "f1", "specificity")},
            "mean_counts": {k: statistics.mean(getattr(c, k) for c in per_run) for k in ("tp", "fp", "fn", "tn")},
            "prevalence": statistics.mean(c.prevalence for c in per_run),
        }
        # Chance baseline: per run and document, TP ~ Hypergeometric(positives, negatives, predicted).
        if not row.endswith("overall"):
            a = arr[row]
            pos, neg = a[..., 0] + a[..., 2], a[..., 3] + a[..., 1]
            pred = a[..., 0] + a[..., 1]
            tp = rng.hypergeometric(pos.astype(np.int64)[..., None].repeat(DEFAULT_BOOT, -1),
                                    neg.astype(np.int64)[..., None].repeat(DEFAULT_BOOT, -1),
                                    pred.astype(np.int64)[..., None].repeat(DEFAULT_BOOT, -1))
            fp, fn = pred[..., None] - tp, pos[..., None] - tp
            tn = neg[..., None] - fp
            null = np.nanmean(mcc_arrays(*(x.sum(axis=1) for x in (tp, fp, fn, tn))), axis=0)  # mean over runs
            f1_null = np.mean(2 * tp.sum(1) / (2 * tp.sum(1) + fp.sum(1) + fn.sum(1)), axis=0)
            obs = res["mcc"]["mean"]
            res["chance"] = {"mcc_mean": float(np.mean(null)), "mcc_97_5": float(np.quantile(null, 0.975)),
                             "mcc_max": float(np.max(null)), "f1_mean": float(np.mean(f1_null)),
                             "f1_97_5": float(np.quantile(f1_null, 0.975)),
                             "p_value": float((1 + np.sum(null >= obs)) / (1 + DEFAULT_BOOT))}
        results[row] = res

    g_mentions = Counter(n[1] for d in docs for n in gt[d][0]) + Counter(r[2] for d in docs for r in gt[d][1])
    g_distinct = Counter(n[1] for n in {n for d in docs for n in gt[d][0]}) + \
        Counter(r[2] for r in {r for d in docs for r in gt[d][1]})
    out_counts = {k: statistics.mean(Counter(n[1] for d in docs for n in run[d][0])[k] if k in NODE_TYPES
                                     else Counter(r[2] for d in docs for r in run[d][1])[k] for run in runs)
                  for k in NODE_TYPES + REL_TYPES}
    distinct_out = defaultdict(list)
    for run in runs:
        nodes = {n for d in docs for n in run[d][0]}
        rels = {r for d in docs for r in run[d][1]}
        c = Counter(n[1] for n in nodes) + Counter(r[2] for r in rels)
        for k in NODE_TYPES + REL_TYPES:
            distinct_out[k].append(c[k])
        distinct_out["entities"].append(len(nodes))
        distinct_out["relations"].append(len(rels))
    g_distinct["entities"] = sum(g_distinct[k] for k in NODE_TYPES)
    g_distinct["relations"] = sum(g_distinct[k] for k in REL_TYPES)
    report = {
        "reference": str(gt_dir.relative_to(REPO)), "runs": len(runs),
        "distinct_counts": {"note": "distinct items pooled over documents, as compared by the reported precision and "
                                    "recall: ground truth = TP + FN, output = TP + FP",
                            "ground_truth": {k: g_distinct[k] for k in NODE_TYPES + REL_TYPES + ["entities", "relations"]},
                            "output": {k: {"mean": statistics.mean(v), "sd": statistics.pstdev(v)}
                                       for k, v in distinct_out.items()}},
        "documents": docs, "runs": len(runs), "outputs_not_read": missing,
        "ground_truth_counts": {"annotated_mentions": dict(g_mentions), "distinct_pooled": dict(g_distinct)},
        "mean_output_mentions_per_run": out_counts,
        "thesis_prf": thesis, "results": results,
        "settings": {"bootstrap": f"{DEFAULT_BOOT} resamples over documents, seed {DEFAULT_SEED}",
                     "chance_draws": DEFAULT_BOOT, "sd": "population, as in the thesis tables"},
    }
    path = REPO / "chapter3" / "results" / "ch3_scores.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print("Wrote", path)
    print("Outputs not read:", missing or "none")
    for row in rows:
        r, t = results[row], thesis[row]
        ch = r.get("chance")
        print(f"{row:16s} F1 {t['f1']['mean']:.3f}±{t['f1']['sd']:.3f}  MCC {r['mcc']['mean']:.3f}±{r['mcc']['sd']:.3f} "
              f"[{r['mcc']['bootstrap_95'][0]:.3f}, {r['mcc']['bootstrap_95'][1]:.3f}]  "
              f"F1 same decisions {r['same_decisions']['f1']['mean']:.3f}"
              + (f"  chance: MCC 97.5% {ch['mcc_97_5']:.3f}, F1 {ch['f1_mean']:.3f}, p {ch['p_value']:.4f}" if ch else ""))


if __name__ == "__main__":
    main()
