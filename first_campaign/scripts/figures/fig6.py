import pandas as pd, numpy as np
from common_style import *
apply_style()

BASE = "data/profiling_runs"

def mono_seconds(ts):
    t = pd.to_datetime(ts, format="%H:%M:%S.%f")
    s = t.dt.hour*3600 + t.dt.minute*60 + t.dt.second + t.dt.microsecond/1e6
    wraps = (s.diff() < -1000).cumsum().fillna(0)
    return s + wraps*86400

def load(run):
    df = pd.read_csv(f"{BASE}/{run}/basil_data.csv.gz")
    df.columns = [c.strip() for c in df.columns]
    df["t"] = mono_seconds(df.Timestamp); df["t"] -= df.t.iloc[0]
    df = df[df.t <= 10800]
    g = df.groupby(df.t.astype(int)).agg(temp=("Temp_C","mean"), cpu=("CPU_%","mean"))
    g = g.reindex(range(0,10801))          # keep sleep gaps as NaN-filled seconds
    g["cpu_s"]  = g.cpu.rolling(60, min_periods=10, center=True).mean()
    return g.reset_index().rename(columns={"index":"t"})

bare = load("bare_metal_A")
dock = load("docker_A")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.4), sharex=True,
                               gridspec_kw=dict(hspace=0.12))
for ax in (ax1, ax2):
    ax.axvspan(0, 10, color="0.88", alpha=0.6, zorder=0)

ax1.plot(bare.t/60, bare.temp, color=RED,  lw=0.6, label="Native (bare-metal), night run")
ax1.plot(dock.t/60, dock.temp, color=BLUE, lw=0.6, label="Docker container, night run")
ax1.axhline(82, color="k", ls="--", lw=1.2, label="Throttling threshold (82 °C)")
ax1.set_ylabel("SoC temperature (°C)")
ax1.set_ylim(48, 86)
ax1.text(5, 49.5, "warm-up\n(excluded)", fontsize=8.5, color="0.35", ha="center", va="bottom")
ax1.legend(loc="center right", bbox_to_anchor=(1.0, 0.78))

ax2.plot(bare.t/60, bare.cpu, color=RED,  lw=0.4, alpha=0.22)
ax2.plot(dock.t/60, dock.cpu, color=BLUE, lw=0.4, alpha=0.22)
ax2.plot(bare.t/60, bare.cpu_s, color=RED,  lw=1.6, label="Native (60 s rolling mean)")
ax2.plot(dock.t/60, dock.cpu_s, color=BLUE, lw=1.6, label="Docker (60 s rolling mean)")
ax2.set_ylabel("CPU utilisation (%)")
ax2.set_xlabel("Elapsed time (minutes)")
ax2.set_ylim(0, 100); ax2.set_xlim(0, 180)
ax2.legend(loc="upper right")

fig.align_ylabels((ax1, ax2))
fig.savefig("Figure_6.png")
print("ok")
