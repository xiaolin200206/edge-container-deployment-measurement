"""Node architecture and measurement chain.

Draws what Section 3.4 argues in prose: which nodes of the power chain the
supply module instruments, and which it does not.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import *

use_style()
PWR, SIG, INSTR, GOLD = "#eaf2fd", "#fdeee7", "#fff6d9", "#b8901f"


def box(ax, x, y, w, h, label, sub=None, fc="#f7f7f5", ec=MUTED, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.12,rounding_size=0.3",
                                fc=fc, ec=ec, lw=0.9, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, label, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=5.8, color=INK2, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)


def arrow(ax, x0, y0, x1, y1, label=None, color=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, lw=1.0, color=color,
                                 ls=ls, zorder=2, shrinkA=0, shrinkB=0))
    if label:
        ax.text((x0 + x1) / 2, max(y0, y1) + 1.1, label, ha="center",
                va="bottom", fontsize=5.9, color=INK)


def fig_system():
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL2, 4.6),
                               gridspec_kw={"height_ratios": [1, 1], "hspace": 0.34})
    for ax in (a, b):
        ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")

    # ---------------- (a) power chain -------------------------------------
    a.text(0, 33, "(a) Power chain and instrumented nodes", fontsize=7.6,
           fontweight="bold", color=INK, va="top")
    TOP, BOT, H = 19, 4, 8
    box(a, 1, TOP, 17, H, "Mains adapter", "USB-PD", PWR, BARE)
    box(a, 27, TOP, 18, H, "IP2368", "PD buck–boost", PWR, BARE)
    box(a, 54, TOP, 18, H, "4S 21700 pack", "BQ4050 gauge", PWR, BARE)
    box(a, 27, BOT, 18, H, "5 V buck", "6 A output", PWR, BARE)
    box(a, 54, BOT, 18, H, "Raspberry Pi 5", "8 GB, passive", SIG, CONT)
    box(a, 80, 11.5, 18, H, "MCU 0x2D", "I²C aggregator", INSTR, GOLD)

    arrow(a, 18, TOP + H / 2, 27, TOP + H / 2, "15 V")
    arrow(a, 45, TOP + H / 2, 54, TOP + H / 2, "16.8 V")
    arrow(a, 36, TOP, 36, BOT + H)
    arrow(a, 45, BOT + H / 2, 54, BOT + H / 2)
    a.text(46.5, BOT + H / 2 + 1.1, "5 V", ha="left", va="bottom", fontsize=5.9, color=INK)
    for x0, y0 in ((72, TOP + H / 2), (72, BOT + H / 2)):
        a.add_patch(FancyArrowPatch((x0, y0), (80, 15.5), arrowstyle="-",
                                    lw=0.7, color=GOLD, ls=(0, (3, 2)), zorder=1))
    a.text(89, 9.0, "1 Hz telemetry", ha="center", va="top", fontsize=5.9, color=INK2)

    # instrumented nodes
    for x, y, t in ((22.5, TOP + H / 2, "VBUS instrumented\nBus_V, Bus_P"),
                    (63, TOP, "pack instrumented\nBat_V, Bat_I, Bat_Pct")):
        a.plot([x], [y], "o", ms=5.0, mfc=CONT, mec="white", mew=1.0, zorder=5)
    a.text(22.5, TOP - 1.0, "VBUS instrumented\nBus_V, Bus_P", ha="center", va="top",
           fontsize=5.8, color=CONT, fontweight="bold")
    a.text(63, TOP - 1.0, "pack instrumented\nBat_V, Bat_I, Bat_Pct", ha="center",
           va="top", fontsize=5.8, color=CONT, fontweight="bold")
    # the uninstrumented node is the point of the panel
    a.plot([51.0], [BOT + H / 2], "x", ms=6.5, mew=1.8, color=NEG, zorder=5)
    a.text(51.0, BOT - 1.0, "5 V rail NOT instrumented", ha="center", va="top",
           fontsize=6.0, color=NEG, fontweight="bold")

    # ---------------- (b) inference and alerting --------------------------
    b.text(0, 33, "(b) Inference and alerting path", fontsize=7.6,
           fontweight="bold", color=INK, va="top")
    ROW, H2 = 15, 8
    b.add_patch(FancyBboxPatch((22, 0.5), 69, 25.5,
                               boxstyle="round,pad=0.5,rounding_size=0.5",
                               fc="none", ec=CONT, lw=1.1, ls=(0, (5, 3)), zorder=1))
    b.text(56.5, 27.6, "container boundary \u2014 the whole inference process "
                       "(condition C; absent in condition A)",
           ha="center", fontsize=6.0, color=CONT, fontweight="bold")

    box(b, 1, ROW, 17, H2, "Camera", "V4L2 640×480", SIG, CONT)
    box(b, 24, ROW, 18, H2, "Pre-process", "224×224", "#f7f7f5", MUTED)
    box(b, 48, ROW, 18, H2, "ONNX Runtime", "MobileNetV2", SIG, CONT)
    box(b, 72, ROW, 18, H2, "Temporal filter", "3 of 5, τ = 0.70", "#f7f7f5", MUTED)
    box(b, 48, 1.5, 18, 7, "Per-frame CSV", "latency, CPU, temp", INSTR, GOLD)
    box(b, 72, 1.5, 18, 7, "HTTPS webhook", "< 2.5 s end-to-end", "#f7f7f5", MUTED)

    arrow(b, 18, ROW + H2 / 2, 24, ROW + H2 / 2)
    arrow(b, 42, ROW + H2 / 2, 48, ROW + H2 / 2)
    arrow(b, 66, ROW + H2 / 2, 72, ROW + H2 / 2)
    arrow(b, 57, ROW, 57, 8.5)
    arrow(b, 81, ROW, 81, 8.5)
    arrow(b, 90, ROW + H2 / 2, 96, ROW + H2 / 2)
    b.text(96.5, ROW + H2 / 2, "alert", ha="left", va="center", fontsize=6.0, color=INK)
    b.set_xlim(0, 104)

    return save(fig, "fig_system")


if __name__ == "__main__":
    print("wrote", fig_system())
