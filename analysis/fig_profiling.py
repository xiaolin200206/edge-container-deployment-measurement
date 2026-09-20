"""Figures derived from the four profiling runs."""
import os
from pathlib import Path
import json, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import *

ROOT = os.environ.get("REPO_ROOT", str(Path(__file__).resolve().parent.parent))
PR = f"{ROOT}/data/profiling_runs"
N = json.load(open(Path(ROOT) / "out" / "numbers.json"))
use_style()


def load(run):
    src = Path(PR) / run / "basil_data.csv.gz"
    if not src.exists():
        src = Path(PR) / run / "basil_data.csv"
    df = pd.read_csv(src)
    t = pd.to_datetime(df.Timestamp, format="%H:%M:%S.%f")
    e = (t - t.iloc[0]).dt.total_seconds().to_numpy()
    df["el"] = np.where(e < 0, e + 86400, e)
    return df


# ---------------------------------------------------------------- Fig: traces
def fig_traces():
    fig, axes = plt.subplots(2, 1, figsize=(COL2, 4.1), sharex=True,
                             gridspec_kw={"hspace": 0.18})
    for run, col, lab in [("bare_metal_A", BARE, "Native"),
                          ("docker_A", CONT, "Containerised")]:
        d = load(run)
        d = d[d.el <= 10800]
        x = d.el / 3600
        axes[0].plot(x, d.Temp_C, color=col, lw=0.25, alpha=0.35)
        axes[0].plot(x, d.Temp_C.rolling(600, min_periods=1).mean(),
                     color=col, lw=1.5, label=lab)
        axes[1].plot(x, d["CPU_%"], color=col, lw=0.12, alpha=0.13)
        axes[1].plot(x, d["CPU_%"].rolling(600, min_periods=1).mean(),
                     color=col, lw=1.5, label=lab)

    for ax in axes:
        grid_y(ax)
        ax.axvspan(0, 600 / 3600, color=MUTED, alpha=0.10, lw=0)
        ax.set_xlim(0, 3)
    axes[0].axhline(82, color=NEG, lw=0.9, ls=(0, (4, 3)))
    axes[0].text(2.97, 82.8, "82 °C throttling threshold", ha="right",
                 va="bottom", color=NEG, fontsize=6.5)
    axes[0].text(600 / 3600 + 0.03, 45, "warm-up\n(excluded)", fontsize=6,
                 color=INK2, va="bottom")
    axes[0].set_ylabel("SoC temperature (°C)")
    axes[0].set_ylim(40, 88)
    axes[1].set_ylabel("CPU utilisation (%)")
    axes[1].set_xlabel("Elapsed time (h)")
    axes[1].set_ylim(0, 100)
    axes[0].legend(loc="lower right", ncol=2)
    # direct labels on the traces themselves
    return save(fig, "fig_traces")


# ------------------------------------------------- Fig: per-run condition plot
def fig_runs():
    metrics = [("latency_ms_mean", "Inference latency", "ms"),
               ("cpu_pct_mean", "CPU utilisation", "%"),
               ("cyclic_peak_temp_c_mean", "Cyclic peak temp.", "\u00b0C"),
               ("power_w_mean", "Node input power", "W"),
               ("throughput_fps_effective", "Effective throughput", "frame s$^{-1}$")]
    fig, axes = plt.subplots(1, 5, figsize=(COL2, 2.35))
    fig.subplots_adjust(wspace=0.85)
    runs = N["per_run"]
    for ax, (key, lab, unit) in zip(axes, metrics):
        allv = [v[key] for v in runs.values()]
        lo, hi = min(allv), max(allv)
        pad = max((hi - lo) * 0.28, (hi + lo) * 0.002)
        for i, (cond, col) in enumerate([("bare-metal", BARE), ("container", CONT)]):
            vals = [v[key] for v in runs.values() if v["condition"] == cond]
            for j, v in enumerate(vals):
                ax.plot(i, v, ["o", "s"][j], color=col, ms=5, mew=0.8,
                        mec="white", zorder=3)
            m = float(np.mean(vals))
            ax.plot([i - 0.28, i + 0.28], [m, m], color=col, lw=2.2, zorder=2)
        grid_y(ax)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Native", "Container"], fontsize=6.8,
                           rotation=28, ha="right", rotation_mode="anchor")
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax.set_title(lab, fontsize=7.2, pad=5)
        ax.set_ylabel(unit, labelpad=1.5, fontsize=7)
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=5, mec="white",
                    label="Replicate A (night)"),
         plt.Line2D([], [], marker="s", ls="", color=MUTED, ms=5, mec="white",
                    label="Replicate B (day)"),
         plt.Line2D([], [], color=MUTED, lw=2.2, label="Condition mean")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.18))
    return save(fig, "fig_runs")


# ------------------------------------------ Fig: the normalisation reversal
def fig_normalised():
    d = N["conditions"]
    items = [("power_w_mean", "Node input\npower", "W", 1),
             ("throughput_fps_effective", "Effective\nthroughput", "frame s$^{-1}$", 1),
             ("energy_per_inference_J", "Energy per\ninference", "J", 1)]
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.5))
    fig.subplots_adjust(wspace=0.42)
    for ax, (key, lab, unit, _) in zip(axes, items):
        b, c = d["bare-metal"][key], d["container"][key]
        eb, ec = d["bare-metal"][key + "_halfrange"], d["container"][key + "_halfrange"]
        ax.bar([0], [b], 0.56, color=BARE, yerr=[eb], capsize=2.5,
               error_kw=dict(lw=0.8, ecolor=INK2), zorder=3)
        ax.bar([1], [c], 0.56, color=CONT, yerr=[ec], capsize=2.5,
               error_kw=dict(lw=0.8, ecolor=INK2), zorder=3)
        pct = 100 * (c / b - 1)
        top = max(b + eb, c + ec)
        ax.set_ylim(0, top * 1.32)
        # direct label: the delta is the point of the figure
        ax.annotate(f"{pct:+.1f}%", xy=(0.5, top * 1.15), ha="center",
                    fontsize=9, fontweight="bold", color=INK)
        for i, v in enumerate((b, c)):
            ax.text(i, v * 0.5, f"{v:.3g}", ha="center", va="center",
                    color="white", fontsize=7, fontweight="bold")
        grid_y(ax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Native", "Containerised"], fontsize=6.8)
        ax.set_title(lab, fontsize=7.5, pad=4)
        ax.set_ylabel(unit, labelpad=2)
    axes[2].set_title("Energy per\ninference", fontsize=7.5, pad=4,
                      fontweight="bold")
    return save(fig, "fig_normalised")


if __name__ == "__main__":
    for f in (fig_traces, fig_runs, fig_normalised):
        print("wrote", f())
