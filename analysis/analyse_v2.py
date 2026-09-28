#!/usr/bin/env python3
"""Three-condition profiling analysis (second measurement campaign).

Reads runs/<label>/{basil_data,power_log,cycle_events}.csv, writes
out/numbers_v2.json. Every quantity stated in the manuscript is emitted here.

Design: randomised complete block design, three conditions x three blocks.
  A native   host interpreter and runtime
  B matched  container, software stack pinned to the host (fingerprint-verified)
  C legacy   container, as-deployed stack (Python 3.9, ORT 1.19.2, glibc 2.31)
Effects are estimated from within-block differences (B-A, C-B, C-A), with a
95% t interval on 2 degrees of freedom. The first 600 s of every run are
excluded as thermal warm-up, as in the first campaign.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd


def csv_path(p):
    """Return p, or p + '.gz' if only the compressed copy exists (the repository
    stores the large per-frame logs gzipped; pandas reads either)."""
    p = Path(p)
    if p.exists():
        return p
    gz = p.with_name(p.name + ".gz")
    return gz if gz.exists() else p

ROOT = Path(os.environ.get("RUNS_ROOT", Path(__file__).resolve().parent.parent / "runs"))
OUT = Path(os.environ.get("OUT_DIR", Path(__file__).resolve().parent.parent / "out"))
WARMUP = 600.0
# planned session periods; actual start times are in each RUN_INFO.txt (block 1's
# native run was repeated in the evening, see the paper's deviations section)
BLOCKS = {1: ("planned night", ["legacy_1", "matched_1", "native_1"]),
          2: ("planned day",   ["legacy_2", "native_2", "matched_2"]),
          3: ("planned night", ["matched_3", "legacy_3", "native_3"])}
FAST_MS = 17.5    # boundary between the two latency modes of A and B
SETTLE = 6.0      # s into the sleep phase after which the governor has stepped down
COND = {"native": "A", "matched": "B", "legacy": "C"}
T975_DF2 = 4.302652729911275


def cycles(ev: pd.DataFrame, t0_clock: pd.Timestamp) -> list[tuple[float, float]]:
    t = pd.to_datetime(ev.Timestamp, format="%H:%M:%S.%f")
    e = (t - t0_clock).dt.total_seconds().to_numpy()
    e = e + np.cumsum(np.r_[0, (np.diff(e) < -43200)]) * 86400.0
    e = np.where(e < -43200, e + 86400, e)
    starts = e[ev.Event.eq("CYCLE_ACTIVE_START").to_numpy()]
    sleeps = e[ev.Event.eq("CYCLE_SLEEP_START").to_numpy()]
    out = []
    for s in starts:
        nx = sleeps[sleeps > s]
        if len(nx):
            out.append((s, nx[0]))
    return out


def run_stats(label: str) -> dict:
    d = ROOT / label
    df = pd.read_csv(csv_path(d / "basil_data.csv"))
    pw = pd.read_csv(csv_path(d / "power_log.csv"))
    ev = pd.read_csv(d / "cycle_events.csv")
    t0 = pd.to_datetime(ev.Timestamp.iloc[0], format="%H:%M:%S.%f")

    def rel(col):
        t = pd.to_datetime(col, format="%H:%M:%S.%f")
        s = (t - t0).dt.total_seconds().to_numpy()
        s = np.where(s < -43200, s + 86400, s)
        return s
    df["el"] = rel(df.Timestamp)
    pw["el"] = rel(pw.Timestamp)
    cyc = cycles(ev, t0)
    cyc_w = [(a, b) for a, b in cyc if a >= WARMUP]
    active_s = sum(b - a for a, b in cyc_w)

    f = df[df.el >= WARMUP]
    lat = f.Latency_ms.astype(float)
    p = pw[pw.el >= WARMUP].copy()
    pa, ps = p[p.Phase == "active"], p[p.Phase == "sleep"]

    # per-cycle peak temperature from the 1 Hz log (active phase of each cycle)
    peaks = []
    for a, b in cyc_w:
        seg = p[(p.el >= a) & (p.el <= b + 2)]
        if len(seg):
            peaks.append(seg.Temp_C.max())
    peaks = np.array(peaks)

    frames = int(len(f))
    fps = frames / active_s
    P_act = pa.Bus_P_mW.mean() / 1000.0
    P_slp = ps.Bus_P_mW.mean() / 1000.0
    # duty-cycle energy, per cycle, from the measured phase powers and durations
    act_len = np.mean([b - a for a, b in cyc_w])
    slp_len = np.mean([c[0] - cyc_w[i][1] for i, c in enumerate(cyc_w[1:])])
    E_cycle = P_act * act_len + P_slp * slp_len
    frames_per_cycle = frames / len(cyc_w)

    # settled floor: sleep-phase samples after the ondemand governor has left
    # the maximum frequency (it holds 2400 MHz for about 5 s after load stops)
    nxt = [c[0] for c in cyc_w[1:]] + [cyc_w[-1][1] + slp_len]
    fl = np.concatenate([ps[(ps.el >= b + SETTLE) & (ps.el < n)].Bus_P_mW.to_numpy()
                         for (a, b), n in zip(cyc_w, nxt)])
    hold = np.concatenate([ps[(ps.el >= b) & (ps.el < b + SETTLE)].Bus_P_mW.to_numpy()
                           for a, b in cyc_w])
    P_floor = fl.mean() / 1000.0
    P_hold = hold.mean() / 1000.0
    T_cyc = act_len + slp_len
    base_floor = P_floor * T_cyc / frames_per_cycle
    E_inc = E_cycle / frames_per_cycle - base_floor

    # whole-run supply and temperature extremes (including warm-up)
    bat_w = (pw.Bat_I_mA.astype(float) * pw.Bat_V_mV.astype(float) / 1e6)

    capped = int((p.Cap_MHz.astype(float) < 2400).sum())
    uv = int((p.UV.astype(float) > 0).sum())
    thr_flags = sorted(p.Throttled.astype(str).unique().tolist())

    fan_pwm_frac = (p.Fan_PWM.astype(int).value_counts(normalize=True)
                    .sort_index().round(4).to_dict())

    return {
        "label": label,
        "condition": label.split("_")[0],
        "frames": frames,
        "frames_total": int(len(df)),
        "cycles_analysed": len(cyc_w),
        "active_s": float(active_s),
        "fps": float(fps),
        "latency_ms_mean": float(lat.mean()),
        "latency_ms_sd": float(lat.std()),
        "latency_ms_median": float(lat.median()),
        "latency_ms_p95": float(lat.quantile(0.95)),
        "latency_ms_p99": float(lat.quantile(0.99)),
        "cpu_pct_frame_mean": float(f["CPU_%"].astype(float).mean()),
        "cpu_pct_1hz_active": float(pd.to_numeric(pa["CPU_%"], errors="coerce").mean()),
        "cpu_pct_1hz_sleep": float(pd.to_numeric(ps["CPU_%"], errors="coerce").mean()),
        "ram_pct_mean": float(f["RAM_%"].astype(float).mean()),
        "temp_c_mean_active": float(pa.Temp_C.mean()),
        "temp_c_mean_all": float(p.Temp_C.mean()),
        "temp_c_max": float(p.Temp_C.max()),
        "cyclic_peak_c_mean": float(peaks.mean()),
        "cyclic_peak_c_max": float(peaks.max()),
        "cpu_mhz_active_mean": float(pa.CPU_MHz.astype(float).mean()),
        "cap_below_hw_max_samples": capped,
        "undervolt_samples": uv,
        "throttle_flags": thr_flags,
        "fan_rpm_mean": float(p.Fan_RPM.astype(float).mean()),
        "fan_rpm_mean_active": float(pa.Fan_RPM.astype(float).mean()),
        "fan_pwm_time_fraction": {str(k): v for k, v in fan_pwm_frac.items()},
        "bus_v_mean": float(p.Bus_V_mV.mean() / 1000),
        "power_w_active": float(P_act),
        "power_w_sleep": float(P_slp),
        "power_w_overall": float(p.Bus_P_mW.mean() / 1000),
        "bus_i_vs_p_over_v_pct": float(100 * (p.Bus_I_mA.mean()
                                       - (p.Bus_P_mW / p.Bus_V_mV * 1000).mean())
                                       / p.Bus_I_mA.mean()),
        "bat_i_ma_mean": float(p.Bat_I_mA.mean()),
        "bat_i_ma_min": int(p.Bat_I_mA.min()), "bat_i_ma_max": int(p.Bat_I_mA.max()),
        "bat_pct_min": int(p.Bat_Pct.min()), "bat_pct_max": int(p.Bat_Pct.max()),
        "active_len_s": float(act_len),
        "sleep_len_s": float(slp_len),
        "energy_per_inference_gross_mJ": float(1000 * P_act / fps),
        "energy_per_inference_net_mJ": float(1000 * (P_act - P_slp) / fps),
        "energy_per_cycle_J": float(E_cycle),
        "energy_per_inference_cycle_mJ": float(1000 * E_cycle / frames_per_cycle),
        "power_w_sleep_floor": float(P_floor),
        "power_w_sleep_hold": float(P_hold),
        "baseline_floor_per_inference_mJ": float(1000 * base_floor),
        "energy_per_inference_above_floor_mJ": float(1000 * E_inc),
        "cpu_ms_per_inference": float(f["CPU_%"].astype(float).mean() / 100 * 4 * 1000 / fps),
        "bat_i_ma_mean_whole": float(pw.Bat_I_mA.mean()),
        "bat_i_ma_min_whole": int(pw.Bat_I_mA.min()), "bat_i_ma_max_whole": int(pw.Bat_I_mA.max()),
        "bat_charge_w_mean_whole": float(bat_w.mean()),
        "temp_c_max_whole_1hz": float(pw.Temp_C.max()),
        "temp_c_max_whole_frame": float(df.Temp_C.astype(float).max()),
        "latency_ms_min": float(lat.min()),
        "latency_ms_p01": float(lat.quantile(0.01)),
        "fast_mode_pct": float(100 * (lat < FAST_MS).mean()),
    }


METRICS = ["fps", "latency_ms_mean", "latency_ms_median", "latency_ms_p95", "latency_ms_p99",
           "latency_ms_sd", "cpu_pct_frame_mean", "cpu_pct_1hz_active", "ram_pct_mean",
           "temp_c_mean_active", "cyclic_peak_c_mean", "temp_c_max", "fan_rpm_mean",
           "fan_rpm_mean_active", "cpu_mhz_active_mean", "power_w_active", "power_w_sleep",
           "power_w_overall", "energy_per_inference_gross_mJ", "energy_per_inference_net_mJ",
           "energy_per_cycle_J", "energy_per_inference_cycle_mJ", "power_w_sleep_floor",
           "power_w_sleep_hold", "baseline_floor_per_inference_mJ",
           "energy_per_inference_above_floor_mJ", "cpu_ms_per_inference", "fast_mode_pct",
           "latency_ms_p01"]


def effects(rs: dict) -> dict:
    by = {b: {rs[l]["condition"]: rs[l] for l in labels} for b, (_, labels) in BLOCKS.items()}
    out = {}
    for m in METRICS:
        e = {}
        for name, hi, lo in (("B_minus_A", "matched", "native"),
                             ("C_minus_B", "legacy", "matched"),
                             ("C_minus_A", "legacy", "native")):
            d = np.array([by[b][hi][m] - by[b][lo][m] for b in BLOCKS])
            base = np.mean([by[b][lo][m] for b in BLOCKS])
            mean = d.mean()
            half = T975_DF2 * d.std(ddof=1) / np.sqrt(3)
            e[name] = {
                "per_block": [float(x) for x in d],
                "mean": float(mean),
                "ci95": [float(mean - half), float(mean + half)],
                "pct_of_base": float(100 * mean / base) if base else None,
                "same_sign_all_blocks": bool(np.all(d > 0) or np.all(d < 0)),
                "ci_excludes_zero": bool((mean - half) > 0 or (mean + half) < 0),
            }
        out[m] = e
    return out


def cond_means(rs: dict) -> dict:
    out = {}
    for c in ("native", "matched", "legacy"):
        mem = [v for v in rs.values() if v["condition"] == c]
        out[c] = {m: {"mean": float(np.mean([x[m] for x in mem])),
                      "sd": float(np.std([x[m] for x in mem], ddof=1))}
                  for m in METRICS}
    return out


def main():
    labels = [l for _, ls in BLOCKS.values() for l in ls]
    rs = {l: run_stats(l) for l in labels}
    res = {"runs": rs, "conditions": cond_means(rs), "effects": effects(rs),
           "design": {"warmup_s": WARMUP, "settle_s": SETTLE, "fast_mode_ms": FAST_MS,
                      "blocks": {b: {"planned_period": p, "runs": ls}
                                                     for b, (p, ls) in BLOCKS.items()},
                      "t_975_df2": T975_DF2}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers_v2.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / 'numbers_v2.json'}")


if __name__ == "__main__":
    main()
