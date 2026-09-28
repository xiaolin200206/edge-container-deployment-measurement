import numpy as np
from common_style import *
apply_style()

# ---------------- Figure 4: metric comparison ----------------
# Real Only values derived from the Fig. 8 confusion matrix (macro-averaged),
# so that Fig. 4, Fig. 8 and Section 5.2 are mutually consistent:
#   accuracy 227/229 = 99.13; macro P = 99.18; macro R = 99.11; macro F1 = 99.13
# Real + Proxy values retained from the original figure (verify against your
# classification report and edit here if needed).
labels = ["Accuracy", "Precision", "Recall", "F1-score"]
real_only  = [99.13, 99.18, 99.11, 99.13]
real_proxy = [97.82, 97.70, 98.00, 97.80]

fig, ax = plt.subplots(figsize=(8.2, 4.6))
x = np.arange(2); w = 0.19
colors = [BLUE, "#56B4E9", ORANGE, RED]
for i, (lab, c) in enumerate(zip(labels, colors)):
    vals = [real_only[i], real_proxy[i]]
    b = ax.bar(x + (i-1.5)*w, vals, w*0.92, color=c, edgecolor="black", lw=0.6, label=lab)
    for r, v in zip(b, vals):
        ax.text(r.get_x()+r.get_width()/2, v+0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["Real Only", "Real + Proxy"], fontsize=12)
ax.set_ylabel("Score (%)"); ax.set_ylim(95, 100.6)
ax.yaxis.grid(True, ls=":", lw=0.6, color="0.8"); ax.set_axisbelow(True)
ax.legend(loc="upper right", ncol=2)
fig.savefig("Figure_4.png"); plt.close(fig)

# ---------------- Figure 5: threshold calibration ----------------
# Values digitised from the original Figure 5 (same underlying sweep);
# verify the acceptance-rate points against the threshold sweep output.
tau        = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
accuracy   = [99.13, 99.13, 99.13, 99.56, 99.56, 99.56, 100.0, 100.0, 100.0, 100.0]
acceptance = [100.0, 99.5, 98.4, 97.9, 97.0, 97.0, 96.5, 95.1, 94.8, 91.7]

fig, ax = plt.subplots(figsize=(8.2, 4.8))
ax2 = ax.twinx()
l1, = ax.plot(tau, accuracy, "-o", color=BLUE, lw=2, ms=6, label="Accuracy (accepted predictions)")
l2, = ax2.plot(tau, acceptance, "--s", color=RED, lw=2, ms=6, label="Acceptance rate")
ax.axvline(0.70, color="0.5", ls=":", lw=1.4)
ax.annotate("Deployed\n(τ = 0.70)", xy=(0.70, 99.56), xytext=(0.715, 96.6),
            fontsize=10, ha="left",
            arrowprops=dict(arrowstyle="->", color="0.4", lw=1.1))
ax.set_xlabel("Confidence threshold (τ)")
ax.set_ylabel("Classification accuracy (%)", color=BLUE)
ax2.set_ylabel("Acceptance rate (%)", color=RED)
ax.tick_params(axis="y", colors=BLUE); ax2.tick_params(axis="y", colors=RED)
ax.set_ylim(95, 101); ax2.set_ylim(90, 101)
ax2.spines["right"].set_visible(True)
ax.yaxis.grid(True, ls=":", lw=0.6, color="0.85"); ax.set_axisbelow(True)
ax.legend(handles=[l1, l2], loc="lower left")
fig.savefig("Figure_5.png"); plt.close(fig)

# ---------------- Figure 8: confusion matrix ----------------
cm = np.array([[75, 0, 0],
               [ 0,73, 2],
               [ 0, 0,79]])
classes = ["Background", "Healthy", "Disease"]
fig, ax = plt.subplots(figsize=(6.0, 5.2))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
for i in range(3):
    for j in range(3):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                color="white" if cm[i, j] > cm.max()*0.55 else "#1a1a1a",
                fontweight="bold" if (i == 1 and j == 2) else "normal")
ax.set_xticks(range(3)); ax.set_xticklabels(classes)
ax.set_yticks(range(3)); ax.set_yticklabels(classes)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.spines[:].set_visible(False)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.outline.set_visible(False)
fig.savefig("Figure_8.png"); plt.close(fig)
print("done")
