"""Shared style for the Results-chapter figures.

The LaTeX caption is the title, so these figures have none. Spines stay on.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from sim.paths import PROJECT_ROOT

RESULTS_FIGURES = PROJECT_ROOT / "results" / "figures"

MEASURED = "#000000"
MODEL = "#C55A11"
LISTED = "#1F4E79"
THRESHOLD = "#7F7F7F"

# Same channel-count colours in every literature figure.
CHANNEL_COLOURS = {
    1: "#7F7F7F",
    7: "#9DC3E6",
    8: "#1F4E79",
    21: "#C55A11",
    45: "#7B2C2C",
}

SINGLE = (6.5, 4.5)
TWO_PANEL = (6.5, 6.0)


def apply_style():
    """Font sizes, no grid. Does not hide the top or right spine."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 9,
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
    })


def channel_colour(c):
    """Colour for a literature channel count. Unknown counts use the last colour."""
    if c in CHANNEL_COLOURS:
        return CHANNEL_COLOURS[c]
    return CHANNEL_COLOURS[45]


def panel_label(ax, text):
    """Site label in the top-left corner, inside the axes."""
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        color="black",
        zorder=6,
    )


def save_figure(fig, name):
    """Write results/figures/<name>.pdf and a 300 dpi PNG. Returns both paths."""
    RESULTS_FIGURES.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    png = RESULTS_FIGURES / f"{stem}.png"
    pdf = RESULTS_FIGURES / f"{stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    return png, pdf
