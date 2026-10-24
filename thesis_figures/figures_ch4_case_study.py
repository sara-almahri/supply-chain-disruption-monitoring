"""Chapter 4 case-study figures from the saved outputs of the Russia--Ukraine run.

    PYTHONDONTWRITEBYTECODE=1 python3 thesis_figures/figures_ch4_case_study.py

Reads evaluation/results/case_study_v1_2/final/tool_outputs/{kg_payload,risk_assessment}.json and writes
casestudy-tier2-paths and casestudy-risk-manager (.pdf and .png) to thesis_figures/out/ch4.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from thesis_figures import style as fs  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

RUN = ROOT / "evaluation/results/case_study_v1_2/final/tool_outputs"
OUT = ROOT / "thesis_figures" / "out" / "ch4"
DISRUPTED = "#CC3311"
SHORT = {"United Kingdom": "UK"}
NEUTRAL = "#33415C"


def node(ax, x, y, name, sub, colour, width=2.95):
    box = FancyBboxPatch((x - width / 2, y - 0.3), width, 0.6, boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=0.8, edgecolor=colour, facecolor="white", zorder=3)
    ax.add_patch(box)
    ax.text(x, y + 0.09, name, ha="center", va="center", fontsize=8, color=fs.COLOURS["text"], zorder=4)
    ax.text(x, y - 0.14, sub, ha="center", va="center", fontsize=6.5, color="#555555", style="italic", zorder=4)


def tier2_paths(payload):
    chains = payload["tier_2"]
    tier1 = list(dict.fromkeys(c[1]["company"] for c in chains))
    tier2 = list(dict.fromkeys(c[2]["company"] for c in chains))
    info = {n["company"]: n for c in chains for n in c}
    focal = chains[0][0]
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH, 2.9))
    xs = (0.0, 3.75, 7.5)
    y1 = {n: 2.0 - i for i, n in enumerate(tier1)}
    y2 = {n: 2.0 - i for i, n in enumerate(tier2)}
    for c in chains:
        a, b = c[1]["company"], c[2]["company"]
        ax.plot([xs[0] + 1.48, xs[1] - 1.48], [0, y1[a]], color="#999999", lw=0.7, zorder=1)
        ax.plot([xs[1] + 1.48, xs[2] - 1.48], [y1[a], y2[b]], color=DISRUPTED, lw=1.0, zorder=1)
    node(ax, xs[0], 0, focal["company"], f"{focal['country']}, monitored firm", NEUTRAL)
    for n, y in y1.items():
        node(ax, xs[1], y, n, f"{SHORT.get(info[n]['country'], info[n]['country'])}, {info[n]['industry']}", NEUTRAL)
    for n, y in y2.items():
        node(ax, xs[2], y, n, f"{info[n]['country']}, {info[n]['industry']}", DISRUPTED)
    for x, label in zip(xs, ("Tier 0", "Tier 1", "Tier 2 (disrupted)")):
        ax.text(x, 2.75, label, ha="center", va="center", fontsize=8.5, weight="bold", color=fs.COLOURS["text"])
    ax.set_xlim(-1.55, 9.05)
    ax.set_ylim(-2.45, 3.0)
    ax.axis("off")
    return fs.save(fig, OUT / "casestudy-tier2-paths")


def risk_scores(risk):
    scores = sorted(risk["supplier_risk_scores"].items(), key=lambda kv: -kv[1])
    values = [v for _, v in scores]
    medium = [n for n, v in scores if 0.45 <= v < 0.60]
    fig, ax = plt.subplots(figsize=(fs.TEXT_WIDTH, 2.3))
    colours = [fs.COLOURS["system"] if 0.45 <= v else fs.COLOURS["baseline"] for v in values]
    ax.bar(range(1, len(values) + 1), values, width=0.8, color=colours, linewidth=0)
    for level, label in ((0.60, "HIGH"), (0.45, "MEDIUM")):
        ax.axhline(level, color=fs.COLOURS["text"], lw=0.6, ls=(0, (4, 3)))
        ax.text(len(values) + 0.5, level + 0.012, f"{label} ({level:.2f})", ha="right", va="bottom", fontsize=7.5)
    ax.annotate(f"{len(medium)} MEDIUM", xy=(len(medium) + 0.6, 0.50), xytext=(10, 0.52), fontsize=7.5,
                color=fs.COLOURS["system"], va="center", arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.COLOURS["system"]))
    ax.text(48, 0.36, f"{len(values) - len(medium)} LOW", fontsize=7.5, color="#777777", ha="center")
    ax.set_xlim(0, len(values) + 1)
    ax.set_ylim(0, 0.70)
    ax.set_xlabel(f"Tier-1 suppliers ranked by risk score ($n={len(values)}$)")
    ax.set_ylabel("Risk score")
    ax.grid(axis="y", lw=0.4, color="#E5E5E5")
    ax.set_axisbelow(True)
    return fs.save(fig, OUT / "casestudy-risk-manager")


def main():
    fs.apply()
    payload = json.loads((RUN / "kg_payload.json").read_text())
    risk = json.loads((RUN / "risk_assessment.json").read_text())
    assert len(payload["tier_2"]) == 6 and len(risk["supplier_risk_scores"]) == 91
    for path in (tier2_paths(payload), risk_scores(risk)):
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
