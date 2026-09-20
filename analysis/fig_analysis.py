"""Figures for the attribution argument, architecture comparison, OOD behaviour
and alerting-filter sensitivity."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figstyle import *

ROOT = str(Path(__file__).resolve().parent.parent)
N = json.load(open(Path(__file__).resolve().parent / "numbers.json"))
use_style()
N_VAL = 340          # held-out validation images per condition (20% of 1698)
PP_PER_IMAGE = 100 / N_VAL


# ------------------------------------------------- Fig: the attribution problem
def fig_attribution():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(COL2, 2.75),
                                 gridspec_kw={"width_ratios": [1, 1.28], "wspace": 0.10})
    for ax in (a0, a1):
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(ax, x, y, w, h, fc, ec, lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.10,rounding_size=0.22",
                                    fc=fc, ec=ec, lw=lw, zorder=2))

    # ---- (a) what was compared -------------------------------------------
    a0.text(0, 9.7, "(a) What the four runs compared", fontsize=7.6,
            fontweight="bold", color=INK, va="top")
    specs = [(0.2, "Condition A", "bare-metal", ["Python 3.13", "ORT 1.29.0", "Debian 13"],
              "#eaf2fd", BARE),
             (5.5, "Condition C", "containerised", ["Python 3.9", "ORT (unpinned)", "Debian 11"],
              "#fdeee7", CONT)]
    for x, tag, mode, lines, fc, ec in specs:
        box(a0, x, 5.5, 4.3, 3.1, fc, ec)
        a0.text(x + 2.15, 8.15, tag, ha="center", fontsize=6.9, fontweight="bold", color=INK)
        a0.text(x + 2.15, 7.55, mode, ha="center", fontsize=6.6, color=INK)
        for k, ln in enumerate(lines):
            a0.text(x + 2.15, 7.0 - k * 0.52, ln, ha="center", fontsize=6.3, color=INK2)
    a0.add_patch(FancyArrowPatch((4.65, 7.0), (5.35, 7.0), arrowstyle="<->",
                                 mutation_scale=8, lw=1.0, color=INK2))
    a0.text(5.0, 4.85,
            "measured Δ = +45.6 % latency, −9.2 pp CPU,"
            "\n−3.7 °C peak, −0.47 W",
            ha="center", va="top", fontsize=6.6, color=INK)
    box(a0, 0.2, 0.7, 9.6, 2.6, "#f7f7f5", MUTED, lw=0.8)
    a0.text(5.0, 3.02, "three factors co-vary", ha="center", fontsize=6.8,
            fontweight="bold", color=INK, va="top")
    for k, t in enumerate(["execution mode", "interpreter + runtime version", "base image"]):
        a0.text(5.0, 2.22 - k * 0.45, t, ha="center", fontsize=6.3, color=INK2)
    a0.text(5.0, 0.90, "Δ is not attributable to any one of them",
            ha="center", fontsize=6.5, color=INK, style="italic")

    # ---- (b) the design that decomposes it -------------------------------
    a1.text(0, 9.7, "(b) Design required for attribution", fontsize=7.6,
            fontweight="bold", color=INK, va="top")
    cols = [2.35, 4.35, 5.95, 7.60]
    a1.text(0.32, 8.55, "run", fontsize=6.2, color=MUTED)
    for cx, h in zip(cols, ["execution", "Python", "ORT", "base image"]):
        a1.text(cx, 8.55, h, ha="center", fontsize=6.2, color=MUTED)
    rows = [("A", "bare-metal", "3.13", "1.29.0", "Debian 13", "#eaf2fd", BARE),
            ("B", "container", "3.13", "1.29.0", "Debian 13", "#eaf2fd", BARE),
            ("C", "container", "3.9", "1.19.2", "Debian 11", "#fdeee7", CONT)]
    ys = [6.95, 4.35, 1.75]
    for (tag, ex, py, ort, base, fc, ec), y in zip(rows, ys):
        box(a1, 0.15, y, 8.5, 1.35, fc, ec)
        a1.text(0.60, y + 0.75, tag, fontsize=7, fontweight="bold", color=INK, va="center")
        for cx, t in zip(cols, [ex, py, ort, base]):
            a1.text(cx, y + 0.68, t, ha="center", fontsize=6.3, color=INK, va="center")
    for y0, y1, lab in ((ys[1] + 1.35, ys[0], "B−A\ncontainerisation"),
                        (ys[2] + 1.35, ys[1], "C−B\nruntime version")):
        a1.annotate("", xy=(8.95, y1), xytext=(8.95, y0),
                    arrowprops=dict(arrowstyle="<->", lw=0.9, color=INK2))
        a1.text(9.15, (y0 + y1) / 2, lab, fontsize=6.2, color=INK, va="center")
    a1.text(0.15, 1.05, "not performed here; specified so the comparison can be decomposed",
            fontsize=6.2, color=INK2, style="italic", va="top")
    a1.set_xlim(0, 11.6)
    return save(fig, "fig_attribution")


# --------------------------------------------- Fig: architecture delta vs noise
def fig_architectures():
    rows = sorted(N["architectures"]["rows"], key=lambda r: r["delta_pp"])
    # RUN_INFO.txt records that the row labelled ViT-Tiny in the released CSVs is
    # torchvision vit_b_16 (85.8 M parameters). Relabelled here accordingly.
    ren = {"ViT-Tiny": "ViT-B/16"}
    names = [ren.get(r["model"], r["model"]) for r in rows]
    d = np.array([r["delta_pp"] for r in rows])
    tol = 2 * PP_PER_IMAGE + 1e-9

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL2, 2.8), sharey=True,
                                   gridspec_kw={"width_ratios": [1, 2.2], "wspace": 0.06})
    y = np.arange(len(d))
    for ax in (axL, axR):
        for k, al in ((2, 0.16), (1, 0.26)):
            ax.axvspan(-k * PP_PER_IMAGE, k * PP_PER_IMAGE, color=MUTED, alpha=al, lw=0, zorder=0)
        ax.axvline(0, color=INK2, lw=0.8, zorder=1)
        cols = [POS if v > 0 else (NEG if v < 0 else MUTED) for v in d]
        ax.barh(y, d, 0.62, color=cols, zorder=3)
        ax.set_axisbelow(True); ax.xaxis.grid(True); ax.yaxis.grid(False)

    axL.set_xlim(-32, -5)
    axR.set_xlim(-1.6, 1.6)
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(left=False)
    axL.set_yticks(y); axL.set_yticklabels(names, fontsize=7)

    # axis-break marks
    for ax, xpos in ((axL, 1.0), (axR, 0.0)):
        for yy in (0, 1):
            ax.plot([xpos], [yy], transform=ax.transAxes, marker=[(-1, -.4), (1, .4)],
                    ms=5, color=MUTED, mew=0.9, ls="", clip_on=False)

    for i, v in enumerate(d):
        if v < -5:      # outlier panel: label sits inside, to the right of the bar end
            axL.text(v - 0.6, i, f"{v:+.2f}", va="center", ha="right",
                     fontsize=6.4, color=INK)
        else:
            off = 0.10 * (1 if v >= 0 else -1)
            axR.text(v + off, i, f"{v:+.2f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=6.4, color=INK)

    # the released CSVs carry accuracy to four decimals, so allow for rounding
    n_in = int((np.abs(d) <= 2 * PP_PER_IMAGE + 0.012).sum())
    n_small = int((np.abs(d) <= 4 * PP_PER_IMAGE + 0.012).sum())
    axR.text(-1.52, len(d) - 0.35,
             f"{n_in} of {len(d)} architectures differ by \u2264 2 validation\n"
             f"images (\u00b1{2*PP_PER_IMAGE:.2f} pp); {n_small} by \u2264 4",
             fontsize=6.8, color=INK, va="top")
    axL.text(-31.5, 2.6, "one condition failed to converge\nin these two runs",
             fontsize=6.4, color=INK2, va="bottom", style="italic")
    axR.text(-0.12, len(d) + 0.15, "favours Real+Proxy", ha="right", fontsize=6.6, color=NEG)
    axR.text(0.12, len(d) + 0.15, "favours Real-Only", ha="left", fontsize=6.6, color=POS)
    fig.supxlabel("Accuracy difference, Real-Only $-$ Real+Proxy (percentage points)",
                  fontsize=7.5, y=-0.02)
    axR.annotate("\u00b11 / \u00b12 validation\nimages", xy=(0.45, 1.0),
                 xytext=(1.05, 2.1), fontsize=6.3, color=INK2, ha="center",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=MUTED))
    return save(fig, "fig_architectures")


# ------------------------------------------------------- Fig: OOD event timeline
def fig_ood():
    df = pd.read_csv(f"{ROOT}/data/supplementary_session/basil_data.csv.gz")
    nb = np.where(df.Predicted_Class.str.lower() != "background")[0]
    ev = np.split(nb, np.where(np.diff(nb) > 30)[0] + 1)
    last = int(nb.max())
    quiescent = len(df) - last - 1

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(COL2, 3.0),
                                  gridspec_kw={"height_ratios": [1, 1.35], "hspace": 0.75})

    # (a) the whole session, at session scale
    ax.set_xlim(0, len(df)); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0, 0.30), len(df), 0.40, fc="#eaf2fd", ec=BARE, lw=0.7))
    ax.text(len(df) * 0.52, 0.50,
            f"{quiescent:,} consecutive frames (3 h 32 min) of an unattended empty scene\n"
            f"with zero non-Background predictions",
            ha="center", va="center", fontsize=7, color=INK)
    ax.plot([0, last], [0.82, 0.82], color=CONT, lw=2.2, solid_capstyle="butt")
    ax.annotate(f"all {len(nb)} non-Background frames fall here:\n"
                f"frames 87\u20131327, a 43 s window at session start",
                xy=(last, 0.80), xytext=(len(df) * 0.10, 0.02),
                fontsize=6.6, color=CONT, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", lw=0.8, color=CONT))
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Frame index", labelpad=1)
    ax.xaxis.set_major_formatter(lambda v, p: f"{int(v/1000)}k")
    ax.set_title("(a) Session-scale behaviour", fontsize=7.6, pad=4, loc="left")

    # (b) the 43 s window, at event scale
    lo, hi = 0, 1500
    ax2.set_xlim(lo, hi)
    for i, e in enumerate(ev):
        c = df.Confidence.iloc[e]
        ax2.plot(e, c, "o", color=CONT, ms=3.4, mew=0.5, mec="white", zorder=3)
        below = (i == 1)            # Event 2 sits just under the tau line
        dy = -7 if below else 8
        va = "top" if below else "bottom"
        yref = float(c.min()) if below else float(c.max())
        ax2.annotate(f"Event {i+1}\n{len(e)} frame{'s' if len(e) > 1 else ''}",
                     xy=(e[len(e)//2], yref), xytext=(0, dy), va=va,
                     textcoords="offset points", ha="center", fontsize=6.3, color=INK)
    ax2.axhline(0.70, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax2.text(780, 0.715, "deployed threshold $\\tau$ = 0.70", ha="left",
             fontsize=6.4, color=INK2)
    ax2.set_ylim(0.40, 1.16)
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Prediction confidence")
    grid_y(ax2)
    ax2.set_title("(b) The four intrusion events, all assigned to Disease, none to Healthy",
                  fontsize=7.6, pad=9, loc="left")
    return save(fig, "fig_ood")


# --------------------------------------------- Fig: threshold/window sensitivity
def simulate(df, threshold, window, ratio=0.6):
    d = df.copy()
    d["cc"] = np.where(d.Confidence >= threshold, d.Predicted_Class, "unconfirmed")
    min_votes = int(np.ceil(window * ratio))
    classes = d.cc.tolist()
    raw = smoothed = 0
    last = None
    for i in range(len(classes)):
        if i > 0 and classes[i] != classes[i - 1]:
            raw += 1
        w = classes[max(0, i - window + 1):i + 1]
        if len(w) < window:
            continue
        vals, counts = np.unique(w, return_counts=True)
        j = int(np.argmax(counts))
        top, cnt = vals[j], counts[j]
        conf = top if (top != "unconfirmed" and cnt >= min_votes) else None
        if conf is not None and conf != last:
            if last is not None:
                smoothed += 1
            last = conf
    return raw, smoothed


def fig_threshold():
    df = pd.read_csv(f"{ROOT}/data/field_log/basil_data.csv.gz")
    taus, wins = [0.3, 0.5, 0.7], [3, 5, 7]
    supp = np.zeros((len(taus), len(wins)))
    trans = np.zeros_like(supp)
    for i, t in enumerate(taus):
        for j, w in enumerate(wins):
            raw, sm = simulate(df, t, w)
            supp[i, j] = 100 * (1 - sm / raw) if raw else np.nan
            trans[i, j] = sm
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    im = ax.imshow(supp, cmap=cmap, vmin=supp.min(), vmax=100, aspect="auto")
    for i in range(len(taus)):
        for j in range(len(wins)):
            v = supp[i, j]
            # choose ink vs white from the actual mapped cell luminance
            r, g, b, _ = cmap((v - supp.min()) / (100 - supp.min()))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, f"{v:.1f}%\n{int(trans[i,j])} trans.", ha="center", va="center",
                    fontsize=6.8, color="white" if lum < 0.55 else INK)
    ax.set_xticks(range(len(wins))); ax.set_xticklabels(wins)
    ax.set_yticks(range(len(taus))); ax.set_yticklabels([f"{t:.1f}" for t in taus])
    ax.set_xlabel("Temporal window (frames)")
    ax.set_ylabel("Confidence threshold $\\tau$")
    ax.set_title("Flip suppression on the field log", fontsize=7.6, pad=5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=6.5)
    cb.set_label("Flip suppression (%)", fontsize=6.8)
    ax.add_patch(plt.Rectangle((1 - .5, 2 - .5), 1, 1, fc="none", ec=CONT, lw=1.8))
    ax.text(1, 1.62, "deployed configuration", ha="center", va="bottom",
            fontsize=6.3, color=CONT, fontweight="bold")
    return save(fig, "fig_threshold")


if __name__ == "__main__":
    for f in (fig_attribution, fig_architectures, fig_ood, fig_threshold):
        print("wrote", f())
