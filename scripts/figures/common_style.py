import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE   = "#0173B2"   # bare-metal / primary
RED    = "#D55E00"   # docker / secondary
GREEN  = "#029E73"
ORANGE = "#DE8F05"
GREY   = "#555555"

def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "0.8",
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
