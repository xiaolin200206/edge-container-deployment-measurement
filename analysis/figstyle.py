"""Shared figure style for the manuscript figure set.

Palette: categorical slots 1-2 of the validated reference palette
(blue #2a78d6 / orange #eb6834). Validated all-pairs, light surface:
CVD dE 24.7 (protan), normal-vision dE 33.6, contrast >= 3:1 - all PASS.
Diverging arm for signed deltas: blue <-> red with a neutral gray midpoint.
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- roles -------------------------------------------------------------
BARE      = "#2a78d6"   # categorical slot 1 - bare-metal
CONT      = "#eb6834"   # categorical slot 2 - containerised
NEG       = "#e34948"   # diverging warm pole
POS       = "#2a78d6"   # diverging cool pole
MID       = "#f0efec"   # diverging neutral midpoint
INK       = "#0b0b0b"   # text primary
INK2      = "#52514e"   # text secondary
MUTED     = "#8a8a84"   # recessive grid / annotation
SURFACE   = "#ffffff"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Column widths for Elsevier single/double column, in inches.
COL1, COL2 = 3.54, 7.20


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
        "font.size": 7.5, "axes.labelsize": 7.5, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        # recessive axes: only left and bottom, hairline
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
        "lines.linewidth": 1.4, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.handlelength": 1.4,
        "legend.columnspacing": 1.2, "legend.handletextpad": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, editable text
    })


def grid_y(ax):
    """Recessive horizontal grid, behind the marks."""
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)


def save(fig, stem, outdir=None):
    """Write both vector and raster copies into <repo>/figures by default."""
    out = Path(outdir) if outdir else Path(__file__).resolve().parent.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)
    return str(out / f"{stem}.pdf")
