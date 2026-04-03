"""
Centralized plotting configuration for journal-quality figures.

Color palette based on user-defined primary pair:
  Blue  #1a80bb  |  Yellow/Gold  #f2c45f

All figures:
  - No titles (captions will be in LaTeX)
  - Saved as both PDF and PNG (300 DPI)
  - Professional axis labels, tick sizes
  - Consistent font family (serif for journal)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import numpy as np

# ============================================================
# COLOR PALETTE
# ============================================================

# Primary pair (from user sample)
BLUE = "#1a80bb"
GOLD = "#f2c45f"

# Extended palette for multi-item figures (up to 6 experiments)
# Chosen to be visually distinct while harmonizing with the primary pair
CORAL  = "#e05a4f"   # warm red-coral
GREEN  = "#5bae6a"   # muted green
PURPLE = "#8e6bbf"   # soft purple
GRAY   = "#7f8c8d"   # neutral gray

# Ordered palette: first 2 match the sample, rest extend naturally
PALETTE_6 = [BLUE, GOLD, CORAL, GREEN, PURPLE, GRAY]

# For 2-item figures (the most common: e.g. train vs val)
PALETTE_2 = [BLUE, GOLD]

# For 3-item figures
PALETTE_3 = [BLUE, GOLD, CORAL]

# For 4-item figures
PALETTE_4 = [BLUE, GOLD, CORAL, GREEN]

# For 5-item figures
PALETTE_5 = [BLUE, GOLD, CORAL, GREEN, PURPLE]

# Mapping experiment names to consistent colors
EXPERIMENT_COLORS = {
    "random":            GRAY,
    "imagenet":          GOLD,
    "ssl_casia":         BLUE,
    "ssl_ekush":         GREEN,
    "supervised_casia":  CORAL,
    "ssl_combined":      PURPLE,
}

# Label fraction line styles for distinguishing within same color
FRACTION_MARKERS = {
    0.01: "v",
    0.05: "^",
    0.10: "s",
    0.25: "D",
    0.50: "o",
    1.00: "P",
}

# Sequential colormaps for heatmaps (confusion matrix)
CMAP_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_seq", ["#ffffff", "#c6ddf0", "#6baed6", BLUE, "#0a4f7a"], N=256
)

# Diverging colormap
CMAP_DIVERGING = "RdYlBu_r"


def get_palette(n):
    """Return a list of n colors from the palette."""
    palettes = {1: [BLUE], 2: PALETTE_2, 3: PALETTE_3, 4: PALETTE_4,
                5: PALETTE_5, 6: PALETTE_6}
    if n <= 6:
        return palettes.get(n, PALETTE_6[:n])
    # For >6, cycle through palette
    return [PALETTE_6[i % 6] for i in range(n)]


def get_experiment_color(name):
    """Get consistent color for an experiment name."""
    for key, color in EXPERIMENT_COLORS.items():
        if key in name.lower():
            return color
    return GRAY


# ============================================================
# MATPLOTLIB RCPARAMS (journal quality)
# ============================================================

JOURNAL_RC = {
    # Font
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,

    # Axes
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Ticks
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",

    # Legend
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc",

    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 5,

    # Figure
    "figure.dpi": 300,
    "figure.figsize": (6, 4),

    # Saving
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "savefig.transparent": False,
}


def apply_journal_style():
    """Apply journal-quality matplotlib settings globally."""
    plt.rcParams.update(JOURNAL_RC)


def save_figure(fig, path, close=True):
    """Save figure as both PDF and PNG. No title."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Save PNG
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.05)

    # Save PDF
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)

    if close:
        plt.close(fig)

    print(f"[FIG] Saved: {png_path.name} + {pdf_path.name}")


# ============================================================
# COMMON FIGURE HELPERS
# ============================================================

def create_figure(width=6, height=4, nrows=1, ncols=1, **kwargs):
    """Create a figure with journal styling applied."""
    apply_journal_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), **kwargs)
    return fig, axes


def add_grid(ax, axis="y", alpha=0.3):
    """Add subtle grid lines."""
    ax.grid(True, axis=axis, alpha=alpha, linewidth=0.5, linestyle="--", color="#cccccc")
    ax.set_axisbelow(True)
