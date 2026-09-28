#!/usr/bin/env python3
"""Figures for the TSUSC manuscript, from out/numbers_v2.json and the raw runs.

Print figures for IEEEtran: single column 3.5 in, double column 7.16 in, 8 pt
text. Condition colours are the first three slots of a validated categorical
palette (all-pairs CVD dE >= 9.2); the third slot is below 3:1 contrast, so
every condition is also direct-labelled and carries a distinct marker shape.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def csv_path(p):
    """Return p, or p + '.gz' if only the compressed copy exists (the repository
    stores the large per-frame logs gzipped; pandas reads either)."""
    p = Path(p)
    if p.exists():
        return p
    gz = p.with_name(p.name + ".gz")
    return gz if gz.exists() else p

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RUNS_ROOT", HERE.parent / "runs"))
OUT = Path(os.environ.get("OUT_DIR", HERE.parent / "out"))
FIG = Path(os.environ.get("FIGURE_DIR", HERE.parent / "figures"))
FIG.mkdir(parents=True, exist_ok=True)
N = json.loads((OUT / "numbers_v2.json").read_text())

COL1, COL2 = 3.5, 7.16
INK, INK2, MUTED, GRID = "#1a1a19", "#52514e", "#8a8984", "#e4e3df"
CC = {"native": "#2a78d6", "matched": "#eb6834", "legacy": "#1baf7a"}
MK = {"native": "o", "matched": "s", "legacy": "^"}
LAB = {"native": "A  native", "matched": "B  matched container", "legacy": "C  legacy container"}
SHORT = {"native": "A", "matched": "B", "legacy": "C"}
ORDER = ["native", "matched", "legacy"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def post_warmup(label, cols):
    """Frames after the warm-up, timed from the run's first cycle event, as in
    analyse_v2.py."""
    df = pd.read_csv(csv_path(ROOT / label / "basil_data.csv"), usecols=["Timestamp"] + cols)
    ev = pd.read_csv(ROOT / label / "cycle_events.csv", usecols=["Timestamp"])
    t0 = pd.to_datetime(ev.Timestamp.iloc[0], format="%H:%M:%S.%f")
    t = pd.to_datetime(df.Timestamp, format="%H:%M:%S.%f")
    s = (t - t0).dt.total_seconds().to_numpy()
    s = np.where(s < -43200, s + 86400, s)
    return df[s >= 600]


# ------------------------------------------------------------ Fig 1: design
def fig_design():
    fig, ax = plt.subplots(figsize=(COL2, 2.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")

    def box(x, y, w, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    fc="white", ec=color, lw=1.1))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
                fontsize=7.8, fontweight="bold", color=INK)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 7.2 - i * 3.6, ln, ha="center", va="top",
                    fontsize=7, color=INK2)

    w, h, y = 26, 22, 14
    xs = [1, 37, 73]
    box(xs[0], y, w, h, "A  native", ["host process", "Python 3.13.5 (Debian)",
        "glibc 2.41", "ONNX Runtime 1.29.0"], CC["native"])
    box(xs[1], y, w, h, "B  matched container", ["Docker, debian:trixie-slim",
        "Python 3.13.5 (Debian)", "glibc 2.41", "ONNX Runtime 1.29.0"], CC["matched"])
    box(xs[2], y, w, h, "C  legacy container", ["Docker, python:3.9-slim",
        "Python 3.9.23", "glibc 2.31", "ONNX Runtime 1.19.2"], CC["legacy"])

    def arrow(x0, x1, label, sub):
        ax.annotate("", xy=(x1, y + h / 2), xytext=(x0, y + h / 2),
                    arrowprops=dict(arrowstyle="-|>", lw=1.0, color=INK))
        ax.text((x0 + x1) / 2, y + h / 2 + 1.6, label, ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color=INK)
        ax.text((x0 + x1) / 2, y + h / 2 - 2.2, sub, ha="center", va="top",
                fontsize=6.6, color=INK2)
    arrow(xs[0] + w + 1.2, xs[1] - 1.2, "B − A", "execution\nmode only")
    arrow(xs[1] + w + 1.2, xs[2] - 1.2, "C − B", "software\nstack only")

    ax.text(50, 9.6,
            "A and B: same interpreter and glibc versions; compiled extension modules of "
            "onnxruntime, numpy, OpenCV and psutil byte-identical (SHA-256)",
            ha="center", va="center", fontsize=6.8, color=INK2, style="italic")
    ax.text(50, 3.4,
            "Same board, cooler, camera, model file (SHA-256), inference code and instrumentation in all three\n"
            "3 blocks × 3 conditions, order randomised within block  ·  3 h per run, 60 s / 15 s duty cycle",
            ha="center", va="center", fontsize=6.8, color=INK)
    save(fig, "fig1_design")


# ------------------------------------------------------------ Fig 2: latency ECDF
def fig_latency():
    fig, ax = plt.subplots(figsize=(COL1, 2.25))
    for c in ORDER:
        x = np.concatenate([post_warmup(f"{c}_{i}", ["Latency_ms"]).Latency_ms.to_numpy()
                            for i in (1, 2, 3)])
        x = np.sort(x)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, color=CC[c], lw=1.4, label=LAB[c])
        for q, ls in ((0.5, ":"), (0.95, "--")):
            v = np.quantile(x, q)
            if c == "matched":   # hollow, so A's marker stays visible where they coincide
                ax.plot([v], [q], marker=MK[c], ms=5.5, mfc="white", mec=CC[c], mew=1.0, zorder=4)
            else:
                ax.plot([v], [q], marker=MK[c], ms=3.8, color=CC[c], mec="white", mew=0.4, zorder=5)
    ax.set_xlim(10, 50); ax.set_ylim(0, 1.01)
    ax.set_xlabel("Per-inference latency (ms)")
    ax.set_ylabel("Cumulative fraction of frames")
    ax.axhline(0.5, color=MUTED, lw=0.5, ls=":"); ax.axhline(0.95, color=MUTED, lw=0.5, ls=":")
    ax.text(10.4, 0.51, "median", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    ax.text(10.4, 0.955, "p95", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    # direct labels
    ax.text(13.2, 0.30, "A", color=CC["native"], fontsize=8, fontweight="bold")
    ax.text(20.5, 0.24, "B", color=CC["matched"], fontsize=8, fontweight="bold")
    ax.text(22.6, 0.36, "C", color=CC["legacy"], fontsize=8, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, handlelength=1.6, borderaxespad=0.2,
              bbox_to_anchor=(1.0, 0.08))
    save(fig, "fig2_latency")


# ------------------------------------------------------------ Fig 3: decomposition
ROWS = [("latency_ms_mean", "Latency, mean"),
        ("latency_ms_p95", "Latency, p95"),
        ("latency_ms_p99", "Latency, p99"),
        ("cpu_pct_frame_mean", "CPU utilisation"),
        ("cpu_ms_per_inference", "CPU time/inference"),
        ("power_w_active", "Power, active phase"),
        ("power_w_sleep", "Power, sleep phase"),
        ("energy_per_inference_above_floor_mJ", "Energy/inf., above floor"),
        ("energy_per_inference_cycle_mJ", "Energy/inf., cycle")]


def fig_decomposition():
    E = N["effects"]; C = N["conditions"]
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.35), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    for ax, key, title, base in ((axes[0], "B_minus_A", "B − A   containerisation, stack held constant", "native"),
                                 (axes[1], "C_minus_B", "C − B   stale stack, execution mode held constant", "matched")):
        for i, (m, lab) in enumerate(ROWS):
            e = E[m][key]
            b = e["mean"] * 100 / e["pct_of_base"]
            mu, lo, hi = (e["pct_of_base"], 100 * e["ci95"][0] / b, 100 * e["ci95"][1] / b)
            yy = len(ROWS) - 1 - i
            ax.plot([lo, hi], [yy, yy], color=INK, lw=1.2, solid_capstyle="round")
            filled = e["ci_excludes_zero"]
            ax.plot([mu], [yy], marker="o", ms=5.2, color=INK if filled else "white",
                    mec=INK, mew=1.0, zorder=3)
            ax.annotate(f"{mu:+.1f}%".replace("-", "\u2212"), (hi, yy), xytext=(5, 0),
                        textcoords="offset points", va="center", ha="left",
                        fontsize=6.6, color=INK2,
                        bbox=dict(fc="white", ec="none", pad=0.4))
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(title, loc="left", fontsize=7.8, pad=4)
        ax.set_xlabel("Difference, % of reference condition (95% CI)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(ROWS)))
    axes[0].set_yticklabels([lab for _, lab in ROWS][::-1])
    axes[0].set_xlim(-22, 32); axes[1].set_xlim(-30, 108)
    axes[0].text(0.02, 0.02, "● CI excludes 0\n○ CI includes 0",
                 transform=axes[0].transAxes, ha="left", va="bottom", fontsize=6.6, color=INK2)
    save(fig, "fig3_decomposition")


# ------------------------------------------------------------ Fig 4: energy budget
def fig_energy():
    R = N["runs"]
    fig, ax = plt.subplots(figsize=(COL1, 2.2))
    ax.set_axisbelow(True)
    x = np.arange(3)
    base_v, work_v = [], []
    for c in ORDER:
        base_v.append([R[f"{c}_{i}"]["baseline_floor_per_inference_mJ"] for i in (1, 2, 3)])
        work_v.append([R[f"{c}_{i}"]["energy_per_inference_above_floor_mJ"] for i in (1, 2, 3)])
    for j, c in enumerate(ORDER):
        b, w = np.mean(base_v[j]), np.mean(work_v[j])
        ax.bar(j, b, width=0.56, color="#d9d8d3", edgecolor="white", lw=1.0)
        ax.bar(j, w, bottom=b, width=0.56, color=CC[c], edgecolor="white", lw=1.0)
        ax.text(j, b / 2, f"{b:.0f}", ha="center", va="center", fontsize=7, color=INK)
        ax.text(j, b + w / 2, f"{w:.0f}", ha="center", va="center", fontsize=7,
                color="white", fontweight="bold")
        ax.text(j, b + w + 14, f"{np.mean(np.add(base_v[j], work_v[j])):.1f} mJ",
                ha="center", va="bottom", fontsize=7, color=INK)
        for k in range(3):
            ax.plot(j + 0.36 + 0.05 * k, base_v[j][k] + work_v[j][k], marker=MK[c], ms=3.4,
                    color=CC[c], mec="white", mew=0.4, ls="none")
    ax.set_xticks(x); ax.set_xticklabels(["A\nnative", "B\nmatched", "C\nlegacy"])
    ax.set_ylabel("Energy per inference over\none duty cycle (mJ)")
    ax.set_ylim(0, 900)
    ax.set_xlim(-0.5, 2.7)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_energy")


# ------------------------------------------------------------ Fig 5: thermal + fan by block
def fig_thermal():
    R = N["runs"]
    blocks = N["design"]["blocks"]
    fig, axes = plt.subplots(1, 2, figsize=(COL1, 2.2), gridspec_kw={"wspace": 0.75})
    for ax, m, ylab in ((axes[0], "cyclic_peak_c_mean", "Mean cyclic peak SoC\ntemperature (°C)"),
                        (axes[1], "fan_rpm_mean_active", "Mean fan speed,\nactive phase (rpm)")):
        for bi, (b, info) in enumerate(blocks.items()):  # noqa: B007
            for c in ORDER:
                lab = [l for l in info["runs"] if l.startswith(c)][0]
                off = {"native": -0.18, "matched": 0.0, "legacy": 0.18}[c]
                ax.plot(bi + off, R[lab][m], marker=MK[c], ms=5, color=CC[c],
                        mec="white", mew=0.6, ls="none", label=SHORT[c] if bi == 0 else None)
        ax.set_xticks(range(3))
        ax.set_xticklabels([str(b) for b in blocks])
        ax.set_xlabel("Block")
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
        ax.set_xlim(-0.5, 2.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, ["A  native", "B  matched", "C  legacy"], loc="upper center", ncol=3,
               frameon=False, bbox_to_anchor=(0.52, 1.07), handletextpad=0.2, columnspacing=1.0)
    save(fig, "fig5_thermal")


if __name__ == "__main__":
    fig_design(); fig_latency(); fig_decomposition(); fig_energy(); fig_thermal()
