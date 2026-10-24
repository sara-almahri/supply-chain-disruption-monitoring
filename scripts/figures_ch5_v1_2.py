"""The Chapter 5 data figures (Figures 5.4 and 5.5), drawn from the stored analysis into thesis_figures/out/ch5.

    PYTHONDONTWRITEBYTECODE=1 python scripts/figures_ch5_v1_2.py

Reads evaluation/results/analysis/v1_2_ch5_analysis.json (scripts/ch5_analysis_v1_2.py) and writes (.pdf and .png):
  - repeatability: pass^k for k = 1..5 of decision correctness and of constrained success, per model;
  - input-variation: paired differences (variant minus original article, GPT-4o) with 95% article-bootstrap intervals.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from thesis_figures import style as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

THESIS_FIGURES = ROOT / "thesis_figures" / "out" / "ch5"
MODELS = (("gpt-4o", "GPT-4o"), ("gpt-4.1", "gpt-4.1"), ("gpt-5-mini", "gpt-5-mini"))
MARKERS = {"gpt-4o": "o", "gpt-4.1": "s", "gpt-5-mini": "^"}


def repeatability(a):
    fig, axes = plt.subplots(1, 2, figsize=(fs.TEXT_WIDTH, fs.HEIGHT), sharey=True)
    for ax, (key, title) in zip(axes, (("decision_ok", "Decisions correct"),
                                       ("constrained", "Decisions correct within budget"))):
        for model, label in MODELS:
            pk = a["models"][model]["repeatability"][key]["pass^k"]
            ks = [1, 2, 3, 4, 5]
            ax.plot(ks, [pk[str(k)] for k in ks], marker=MARKERS[model], ms=4, lw=1.2,
                    color=fs.COLOURS[model], label=label)
        ax.set_title(title)
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xlabel("$k$ (runs that must all succeed)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel(r"pass$^k$")
    axes[1].legend(loc="lower left")
    return fs.save(fig, THESIS_FIGURES / "repeatability")


def input_variation(a):
    measures = (("type", "Type correct"), ("scope", "Scope correct"), ("entities", "Entity sets correct"),
                ("decision_ok", "Decisions correct"))
    conds = (("paraphrase", "Paraphrase"), ("noise-mild", "Noise, mild"), ("noise-moderate", "Noise, moderate"),
             ("noise-high", "Noise, high"))
    colours = (fs.COLOURS["system"], fs.COLOURS["gpt-4.1"], fs.COLOURS["second"], fs.COLOURS["gpt-5-mini"])
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH, 2.6))
    n_m, n_c = len(measures), len(conds)
    for j, ((cond, clabel), colour) in enumerate(zip(conds, colours)):
        for i, (m, _) in enumerate(measures):
            d = a["variants"][cond]["paired_vs_original"][m]
            y = (n_m - 1 - i) + (n_c - 1 - j - (n_c - 1) / 2) * 0.17
            ax.errorbar(d["diff"], y, xerr=[[d["diff"] - d["ci95"][0]], [d["ci95"][1] - d["diff"]]], fmt="o", ms=3.5,
                        color=colour, lw=1, label=clabel if i == 0 else None)
    ax.axvline(0, color=fs.COLOURS["text"], lw=0.6)
    ax.set_yticks(range(n_m)[::-1], [label for _, label in measures])
    ax.set_xlabel("Difference from the original articles (rate, paired by scenario)")
    ax.grid(axis="x", lw=0.4, color="#E5E5E5")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return fs.save(fig, THESIS_FIGURES / "input-variation")


def main():
    fs.apply()
    a = json.loads((ROOT / "evaluation/results/analysis/v1_2_ch5_analysis.json").read_text())
    for path in (repeatability(a), input_variation(a)):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
