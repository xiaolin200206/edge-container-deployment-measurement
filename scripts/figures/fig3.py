from common_style import *
import matplotlib.patches as mp
apply_style()

fig, ax = plt.subplots(figsize=(10.5, 4.2))
ax.set_xlim(0, 10.5); ax.set_ylim(0, 4.2); ax.axis("off")

def box(x, y, w, h, fc="white", ec="0.25", lw=1.3, r=0.08):
    ax.add_patch(mp.FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={r}", fc=fc, ec=ec, lw=lw))

def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#1f4e79", shrinkA=0, shrinkB=0))

# Stage 1
box(0.25, 0.35, 2.9, 3.1)
ax.text(1.70, 3.75, "Edge hardware", ha="center", fontsize=13, fontweight="bold")
ax.text(1.70, 2.70, "Raspberry Pi 5 (8 GB)", ha="center", fontsize=11.5, fontweight="bold")
ax.text(1.70, 2.22, "Camera Module 3", ha="center", fontsize=10.5)
ax.text(1.70, 1.86, "(Picamera2 / libcamera)", ha="center", fontsize=9.5, color="0.35")
ax.text(1.70, 1.05, "RGB frame capture", ha="center", fontsize=10, style="italic", color="0.3")

# Stage 2
box(3.85, 0.35, 3.15, 3.1, ec="#1f4e79", lw=1.6)
ax.text(5.425, 3.75, "Docker container environment", ha="center", fontsize=13, fontweight="bold")
steps = [(2.85, "Image preprocessing",   "Resize 224 × 224 · ImageNet normalisation"),
         (1.90, "ONNX Runtime inference","MobileNetV2, ARM64 CPU"),
         (0.95, "Temporal confirmation", "3-of-5 window · τ ≥ 0.70")]
for yc, title, sub in steps:
    box(4.05, yc-0.275, 2.75, 0.55, fc="#2e8b57", ec="none")
    ax.text(5.425, yc, title, ha="center", va="center", fontsize=10.5,
            color="white", fontweight="bold")
    ax.text(5.425, yc-0.50, sub, ha="center", va="center", fontsize=8.4, color="0.3")

# Stage 3
box(7.85, 0.35, 2.4, 3.1)
ax.text(9.05, 3.75, "Real-time alerting", ha="center", fontsize=13, fontweight="bold")
ax.text(9.05, 2.70, "Telegram Bot API", ha="center", fontsize=11.5, fontweight="bold")
ax.text(9.05, 2.22, "Asynchronous webhook", ha="center", fontsize=10.5)
ax.text(9.05, 1.86, "(HTTPS POST)", ha="center", fontsize=9.5, color="0.35")
ax.text(9.05, 1.05, "End-to-end latency < 2.5 s", ha="center", fontsize=10,
        style="italic", color="0.3")

arrow(3.22, 1.9, 3.80, 1.9)
arrow(7.06, 1.9, 7.80, 1.9)
fig.savefig("Figure_3.png")
print("done")
