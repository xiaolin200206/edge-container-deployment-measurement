#!/usr/bin/env python3
"""Additional quantities derived from the existing nine runs (no new measurements).

Reads the same logs as analyse_v2.py and writes out/extras_v2.json:

* background processor load: 1 Hz CPU utilisation in the settled part of the
  sleep phase, when the measured process sleeps with the camera open, and the
  workload CPU time per inference above that background;
* the share of active-phase wall time spent inside the timed inference call,
  an upper bound on CPU time inside the call (four cores busy for the whole
  call), and the inference-only rate that the call duration would allow;
* frame-interval statistics, to show that the loop was camera-bound;
* a per-cycle energy account split into active and sleep phase;
* the drift of cyclic peak temperature and active-phase fan speed over each
  run (first vs last 20 analysed cycles and a least-squares slope);
* SoC temperature and fan speed late in the sleep phase, as a proxy for the
  thermal environment of each run (ambient temperature was not logged);
* within-block effects for all of the above (same estimator as analyse_v2.py),
  plus two-block (blocks 2 and 3, camera pinned in every run) estimates.
"""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
import pandas as pd

import analyse_v2 as A

OUT = A.OUT
LATE_SLEEP = 3.0   # last seconds of each sleep phase used for the thermal proxy
EDGE = 20          # cycles at the start / end of the analysed period
T975_DF1 = 12.706204736174698


def run_extras(label: str) -> dict:
    d = A.ROOT / label
    df = pd.read_csv(A.csv_path(d / "basil_data.csv"))
    pw = pd.read_csv(A.csv_path(d / "power_log.csv"))
    ev = pd.read_csv(d / "cycle_events.csv")
    t0 = pd.to_datetime(ev.Timestamp.iloc[0], format="%H:%M:%S.%f")

    def rel(col):
        t = pd.to_datetime(col, format="%H:%M:%S.%f")
        s = (t - t0).dt.total_seconds().to_numpy()
        s = s + np.cumsum(np.r_[0, (np.diff(s) < -43200)]) * 86400.0
        return np.where(s < -43200, s + 86400, s)

    df["el"] = rel(df.Timestamp)
    pw["el"] = rel(pw.Timestamp)
    cyc = [(a, b) for a, b in A.cycles(ev, t0) if a >= A.WARMUP]
    act_len = np.mean([b - a for a, b in cyc])
    slp_len = np.mean([c[0] - cyc[i][1] for i, c in enumerate(cyc[1:])])
    nxt = [c[0] for c in cyc[1:]] + [cyc[-1][1] + slp_len]

    f = df[df.el >= A.WARMUP]
    p = pw[pw.el >= A.WARMUP].copy()
    p["cpu"] = pd.to_numeric(p["CPU_%"], errors="coerce")
    pa, ps = p[p.Phase == "active"], p[p.Phase == "sleep"]

    frames = len(f)
    active_s = sum(b - a for a, b in cyc)
    fps = frames / active_s
    lat = f.Latency_ms.astype(float)

    # --- processor load: background (settled sleep) vs active phase
    settled = pd.concat([ps[(ps.el >= b + A.SETTLE) & (ps.el < n)] for (a, b), n in zip(cyc, nxt)])
    cpu_bg = float(settled.cpu.mean())
    cpu_act_1hz = float(pa.cpu.mean())
    cpu_frame = float(f["CPU_%"].astype(float).mean())
    ms_per_inf = lambda pct: pct / 100 * 4 * 1000 / fps
    cpu_ms_total = ms_per_inf(cpu_frame)
    cpu_ms_bg = ms_per_inf(cpu_bg)
    cpu_ms_work = cpu_ms_total - cpu_ms_bg
    call_bound = 4 * float(lat.mean())          # four cores busy for the whole call

    # --- frame intervals within active phases (camera-bound check)
    gaps = []
    for (a, b) in cyc:
        t = f[(f.el >= a) & (f.el <= b)].el.to_numpy()
        if len(t) > 1:
            gaps.append(np.diff(t))
    gaps = np.concatenate(gaps) * 1000

    # --- per-cycle energy split
    P_act = pa.Bus_P_mW.mean() / 1000
    P_slp = ps.Bus_P_mW.mean() / 1000
    E_act = P_act * act_len
    E_slp = P_slp * slp_len
    fpc = frames / len(cyc)

    # --- drift over the run
    peaks, fans, tmid = [], [], []
    for a, b in cyc:
        seg = p[(p.el >= a) & (p.el <= b + 2)]
        peaks.append(seg.Temp_C.max())
        fans.append(pa[(pa.el >= a) & (pa.el <= b)].Fan_RPM.astype(float).mean())
        tmid.append((a + b) / 2 / 3600)
    peaks, fans, tmid = map(np.asarray, (peaks, fans, tmid))
    slope_peak = float(np.polyfit(tmid, peaks, 1)[0])
    slope_fan = float(np.polyfit(tmid, fans, 1)[0])

    # --- thermal proxy late in each sleep phase
    late = pd.concat([ps[(ps.el >= n - LATE_SLEEP) & (ps.el < n)] for (a, b), n in zip(cyc, nxt)])

    return {
        "label": label, "condition": label.split("_")[0],
        "cpu_pct_background": cpu_bg,
        "cpu_pct_active_1hz": cpu_act_1hz,
        "cpu_ms_per_inference": cpu_ms_total,
        "cpu_ms_background_per_inference": cpu_ms_bg,
        "cpu_ms_workload_per_inference": cpu_ms_work,
        "cpu_ms_call_upper_bound": call_bound,
        "cpu_ms_outside_call_lower_bound": cpu_ms_work - call_bound,
        "call_share_of_active_pct": float(100 * fps * lat.mean() / 1000),
        "inference_only_rate_per_s": float(1000 / lat.mean()),
        "frame_interval_ms_mean": float(gaps.mean()),
        "frame_interval_ms_sd": float(gaps.std()),
        "frame_interval_ms_p99": float(np.percentile(gaps, 99)),
        "frame_interval_over_100ms_pct": float(100 * (gaps > 100).mean()),
        "cycle_s": float(act_len + slp_len),
        "inferences_per_cycle": float(fpc),
        "energy_active_J_per_cycle": float(E_act),
        "energy_sleep_J_per_cycle": float(E_slp),
        "energy_cycle_J": float(E_act + E_slp),
        "sleep_share_of_cycle_energy_pct": float(100 * E_slp / (E_act + E_slp)),
        "energy_sleep_mJ_per_inference": float(1000 * E_slp / fpc),
        "energy_active_mJ_per_inference": float(1000 * E_act / fpc),
        "peak_first20_c": float(peaks[:EDGE].mean()),
        "peak_last20_c": float(peaks[-EDGE:].mean()),
        "peak_slope_c_per_h": slope_peak,
        "fan_first20_rpm": float(fans[:EDGE].mean()),
        "fan_last20_rpm": float(fans[-EDGE:].mean()),
        "fan_slope_rpm_per_h": slope_fan,
        "late_sleep_temp_c": float(late.Temp_C.mean()),
        "late_sleep_fan_rpm": float(late.Fan_RPM.astype(float).mean()),
    }


def effects(rs, metrics, blocks):
    by = {b: {rs[l]["condition"]: rs[l] for l in A.BLOCKS[b][1]} for b in blocks}
    t = A.T975_DF2 if len(blocks) == 3 else T975_DF1
    out = {}
    for m in metrics:
        e = {}
        for name, hi, lo in (("B_minus_A", "matched", "native"),
                             ("C_minus_B", "legacy", "matched")):
            dd = np.array([by[b][hi][m] - by[b][lo][m] for b in blocks])
            base = np.mean([by[b][lo][m] for b in blocks])
            half = t * dd.std(ddof=1) / np.sqrt(len(dd))
            e[name] = {"per_block": dd.tolist(), "mean": float(dd.mean()),
                       "ci95": [float(dd.mean() - half), float(dd.mean() + half)],
                       "pct_of_base": float(100 * dd.mean() / base) if base else None,
                       "ci_excludes_zero": bool(dd.mean() - half > 0 or dd.mean() + half < 0)}
        out[m] = e
    return out


def main():
    labels = [l for _, ls in A.BLOCKS.values() for l in ls]
    rs = {l: run_extras(l) for l in labels}
    metrics = [k for k in next(iter(rs.values())) if k not in ("label", "condition")]
    cm = {c: {m: {"mean": float(np.mean([v[m] for v in rs.values() if v["condition"] == c])),
                  "sd": float(np.std([v[m] for v in rs.values() if v["condition"] == c], ddof=1))}
              for m in metrics} for c in ("native", "matched", "legacy")}
    # two-block sensitivity for the main quantities of analyse_v2 as well
    main_rs = json.loads((OUT / "numbers_v2.json").read_text())["runs"]
    two = effects(main_rs, ["power_w_active", "power_w_sleep_floor",
                            "energy_per_inference_above_floor_mJ",
                            "energy_per_inference_cycle_mJ", "cpu_ms_per_inference",
                            "latency_ms_mean"], [2, 3])
    res = {"runs": rs, "conditions": cm,
           "effects": effects(rs, metrics, [1, 2, 3]),
           "effects_blocks_2_3": effects(rs, metrics, [2, 3]),
           "main_effects_blocks_2_3": two}
    (OUT / "extras_v2.json").write_text(json.dumps(res, indent=2))
    print(f"wrote {OUT / 'extras_v2.json'}")


if __name__ == "__main__":
    main()
