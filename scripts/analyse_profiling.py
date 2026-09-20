#!/usr/bin/env python3
"""
Reproduces Tables 3 and 4 of the manuscript from the raw profiling logs.

Usage:
    python3 scripts/analyse_profiling.py

Reads the four profiling runs in data/profiling_runs/ and prints:
  - per-run aggregates (manuscript Table 3)
  - condition means with half-range (manuscript Table 4)
  - the day/night ambient-effect check reported in Section 4.3

Protocol notes (see manuscript Section 3.7):
  - every run is truncated to the first 10 800 s (3 h) so that all four are
    equal length; the Docker A run overran and is truncated here
  - the first 600 s of each run is discarded as a thermal warm-up period
  - cyclic peak temperatures are taken from the CYCLE_SLEEP_START events,
    i.e. the temperature recorded at the end of each 60 s active period
"""

import os
import sys
import pandas as pd

RUNS = {
    "Bare-metal A (night)": ("data/profiling_runs/bare_metal_A", "bare"),
    "Bare-metal B (day)":   ("data/profiling_runs/bare_metal_B", "bare"),
    "Docker A (night)":     ("data/profiling_runs/docker_A",     "docker"),
    "Docker B (day)":       ("data/profiling_runs/docker_B",     "docker"),
}

RUN_SECONDS = 10_800   # 3 h
WARMUP_SECONDS = 600   # discarded


def monotonic_seconds(timestamps):
    """Convert HH:MM:SS.mmm strings to seconds, repairing midnight wrap-around."""
    t = pd.to_datetime(timestamps, format="%H:%M:%S.%f")
    s = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second + t.dt.microsecond / 1e6
    wraps = (s.diff() < -1000).cumsum().fillna(0)
    return s + wraps * 86400


def _frames_path(path):
    """Per-frame log, stored gzipped in the repository to stay within file-size limits."""
    gz = os.path.join(path, "basil_data.csv.gz")
    plain = os.path.join(path, "basil_data.csv")
    return gz if os.path.exists(gz) else plain


def load_run(path):
    frames = pd.read_csv(_frames_path(path))
    frames.columns = [c.strip() for c in frames.columns]
    frames["t"] = monotonic_seconds(frames.Timestamp)
    frames["t"] -= frames.t.iloc[0]
    frames = frames[frames.t <= RUN_SECONDS]

    events = pd.read_csv(os.path.join(path, "cycle_events.csv"))
    events.columns = [c.strip() for c in events.columns]
    events["t"] = monotonic_seconds(events.Timestamp)
    events["t"] -= events.t.iloc[0]
    events = events[events.t <= RUN_SECONDS]
    return frames, events


def summarise(frames, events):
    warm = frames[frames.t > WARMUP_SECONDS]
    peaks = events.loc[
        (events.Event == "CYCLE_SLEEP_START") & (events.t > WARMUP_SECONDS), "Temp_C"
    ].astype(float)
    all_peaks = events.loc[events.Event == "CYCLE_SLEEP_START", "Temp_C"].astype(float)
    return {
        "frames": len(frames),
        "cycles": len(all_peaks),
        "latency_ms": warm.Latency_ms.mean(),
        "cpu_pct": warm["CPU_%"].mean(),
        "ram_pct": warm["RAM_%"].mean(),
        "temp_mean_c": warm.Temp_C.mean(),
        "peak_temp_c": peaks.mean(),
        "peak_temp_max_c": all_peaks.max(),
        "power_w": warm.Bus_P_mW.mean() / 1000.0,
    }


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    results = {}
    for label, (path, condition) in RUNS.items():
        if not os.path.isdir(path):
            sys.exit(f"missing run directory: {path}")
        frames, events = load_run(path)
        summary = summarise(frames, events)
        summary["condition"] = condition
        results[label] = summary

    table = pd.DataFrame(results).T

    print("\n=== Table 3: per-run results (3 h, 144 cycles, warm-up excluded) ===\n")
    print(
        table[["frames", "cycles", "latency_ms", "cpu_pct", "ram_pct",
               "peak_temp_c", "power_w"]].round(2).to_string()
    )

    print("\n=== Table 4: condition means (n = 2, +/- half-range) ===\n")
    metrics = ["latency_ms", "cpu_pct", "ram_pct", "temp_mean_c", "peak_temp_c", "power_w"]
    for metric in metrics:
        bare = table.loc[table.condition == "bare", metric].astype(float)
        dock = table.loc[table.condition == "docker", metric].astype(float)
        b_mid, b_half = bare.mean(), (bare.max() - bare.min()) / 2
        d_mid, d_half = dock.mean(), (dock.max() - dock.min()) / 2
        delta = d_mid - b_mid
        pct = delta / b_mid * 100
        print(f"{metric:14s} bare {b_mid:7.2f} +/- {b_half:4.2f}   "
              f"docker {d_mid:7.2f} +/- {d_half:4.2f}   "
              f"delta {delta:+7.2f} ({pct:+5.1f}%)")

    print("\n=== Section 4.3 day/night ambient check ===\n")
    for condition in ("bare", "docker"):
        night = [k for k, v in RUNS.items() if v[1] == condition and "night" in k][0]
        day = [k for k, v in RUNS.items() if v[1] == condition and "day" in k][0]
        n_t = float(table.loc[night, "temp_mean_c"])
        d_t = float(table.loc[day, "temp_mean_c"])
        print(f"{condition:7s} night {n_t:5.2f} C   day {d_t:5.2f} C   "
              f"ambient effect {d_t - n_t:+.2f} C")
    print()


if __name__ == "__main__":
    main()
