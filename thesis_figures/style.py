"""One style for every data figure of the thesis.

Figures are drawn at their printed size (text width 160 mm, A4 with 30 mm and 20 mm margins), so that lettering in a
figure is the size of the surrounding text. The typeface is Computer Modern, as in the thesis. Colours come from Paul
Tol's colour-blind-safe palette and keep one meaning throughout: ground truth, the evaluated system (GPT-4o), the
other models, a second measure, and baselines.
"""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)  # font timestamps of the embedded Computer Modern
import matplotlib.pyplot as plt  # noqa: E402

TEXT_WIDTH = 6.30          # inches
HALF_WIDTH = 0.48 * TEXT_WIDTH
HEIGHT = 2.35              # inches, for a half-width panel

COLOURS = {
    "ground_truth": "#33415C",
    "system": "#4477AA",
    "gpt-4o": "#4477AA",
    "gpt-4.1": "#66CCEE",
    "gpt-5-mini": "#CCBB44",
    "second": "#228833",
    "baseline": "#BBBBBB",
    "text": "#222222",
}


def apply():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.linewidth": 0.6, "axes.edgecolor": COLOURS["text"], "axes.labelcolor": COLOURS["text"],
        "xtick.color": COLOURS["text"], "ytick.color": COLOURS["text"],
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E5E5E5", "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "errorbar.capsize": 2,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
    })


def label_bars(ax, bars, values, fmt="{:.0f}", offset=0.012):
    top = ax.get_ylim()[1]
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset * top, fmt.format(v),
                ha="center", va="bottom", fontsize=7, color=COLOURS["text"])


def save(fig, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)
    return path.with_suffix(".pdf")
