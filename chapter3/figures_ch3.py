"""Chapter 3 count figures, drawn from chapter3/results/ch3_scores.json (score_ch3.py).

  ground-truth-entities.pdf, ground-truth-relations.pdf    fig:ch3-distribution
  extracted-entities.pdf, extracted-relations.pdf          fig:ch3-average-extraction

Counts are distinct items pooled over the evaluation documents, the items compared by the reported precision and recall, so
every bar equals TP + FN (ground truth) or the mean of TP + FP over the ten runs (extracted) in the results table.
Usage: python chapter3/figures_ch3.py [--out DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from thesis_figures import style as fs  # noqa: E402

ENTITY_TYPES = [("company", "Company"), ("product", "Product"), ("location", "Location"), ("material", "Material"),
                ("person", "Person"), ("mine", "Mine")]
RELATION_TYPES = [("locatedin", "locatedIn"), ("owns", "owns"), ("produces", "produces"), ("suppliesto", "suppliesTo")]


def ground_truth(counts, types, path):
    fig, ax = fs.plt.subplots(figsize=(fs.HALF_WIDTH, fs.HEIGHT))
    vals = [counts["ground_truth"][k] for k, _ in types]
    bars = ax.bar(range(len(types)), vals, width=0.62, color=fs.COLOURS["ground_truth"])
    ax.set_xticks(range(len(types)), [lab for _, lab in types], rotation=30, ha="right")
    ax.set_ylabel("Distinct items")
    ax.set_ylim(0, max(vals) * 1.15)
    fs.label_bars(ax, bars, vals)
    return fs.save(fig, path)


def extracted(counts, types, path):
    fig, ax = fs.plt.subplots(figsize=(fs.HALF_WIDTH, fs.HEIGHT))
    x = np.arange(len(types))
    w = 0.38
    g = [counts["ground_truth"][k] for k, _ in types]
    m = [counts["output"][k]["mean"] for k, _ in types]
    s = [counts["output"][k]["sd"] for k, _ in types]
    b1 = ax.bar(x - w / 2, g, w, color=fs.COLOURS["ground_truth"], label="Ground truth")
    b2 = ax.bar(x + w / 2, m, w, yerr=s, color=fs.COLOURS["system"], label="Extracted (mean of 10 runs)",
                error_kw={"elinewidth": 0.6, "ecolor": fs.COLOURS["text"]})
    ax.set_xticks(x, [lab for _, lab in types], rotation=30, ha="right")
    ax.set_ylabel("Distinct items")
    ax.set_ylim(0, max(g + m) * 1.15)
    fs.label_bars(ax, b1, g)
    for bar, v, sd in zip(b2, m, s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + sd + 0.012 * ax.get_ylim()[1], f"{v:.0f}",
                ha="center", va="bottom", fontsize=7, color=fs.COLOURS["text"])
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.2, columnspacing=1.0)
    return fs.save(fig, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "chapter3" / "figures")
    args = ap.parse_args()
    counts = json.loads((REPO / "chapter3" / "results" / "ch3_scores.json").read_text())["distinct_counts"]
    fs.apply()
    for p in (ground_truth(counts, ENTITY_TYPES, args.out / "ground-truth-entities"),
              ground_truth(counts, RELATION_TYPES, args.out / "ground-truth-relations"),
              extracted(counts, ENTITY_TYPES, args.out / "extracted-entities"),
              extracted(counts, RELATION_TYPES, args.out / "extracted-relations")):
        print("Wrote", p)


if __name__ == "__main__":
    main()
