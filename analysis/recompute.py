"""Recompute every quantity reported in the manuscript from the released raw logs.

Writes numbers.json. Nothing in the manuscript may state a figure that does not
appear here, so that every reported value is traceable to the deposited data.
"""
import json, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PR = ROOT / "data/profiling_runs"
WARMUP, RUNEND = 600.0, 10800.0
RUNS = {"bare_metal_A": ("bare-metal", "night"), "bare_metal_B": ("bare-metal", "day"),
        "docker_A": ("container", "night"), "docker_B": ("container", "day")}


def elapsed(ts, t0):
    t = pd.to_datetime(ts, format="%H:%M:%S.%f")
    e = (t - t0).dt.total_seconds().to_numpy()
    return np.where(e < 0, e + 86400.0, e)


def run_stats(name):
    df = pd.read_csv(PR / name / "basil_data.csv")
    ev = pd.read_csv(PR / name / "cycle_events.csv")
    t0 = pd.to_datetime(df.Timestamp.iloc[0], format="%H:%M:%S.%f")
    df["el"] = elapsed(df.Timestamp, t0)
    ev["el"] = elapsed(ev.Timestamp, t0)
    w = df[(df.el >= WARMUP) & (df.el <= RUNEND)]
    ew = ev[(ev.el >= WARMUP) & (ev.el <= RUNEND)]

    # Active-window duration: telemetry is logged only while inference runs,
    # so effective throughput is measured against summed active time.
    starts = ev[ev.Event == "CYCLE_ACTIVE_START"].el.to_numpy()
    sleeps = ev[ev.Event == "CYCLE_SLEEP_START"].el.to_numpy()
    n = min(len(starts), len(sleeps))
    spans = [(s, e) for s, e in zip(starts[:n], sleeps[:n])
             if e > WARMUP and s < RUNEND]
    active_s = sum(min(e, RUNEND) - max(s, WARMUP) for s, e in spans)

    lat, cpu, pw = w.Latency_ms, w["CPU_%"], w.Bus_P_mW / 1000.0
    fps_eff = len(w) / active_s
    # Cyclic peak temperature: SoC temperature recorded at each active->sleep
    # transition, i.e. the peak of each duty cycle.
    peaks = ew[ew.Event == "CYCLE_SLEEP_START"].Temp_C

    return {
        "condition": RUNS[name][0], "time_of_day": RUNS[name][1],
        "n_frames": int(len(w)), "active_seconds": round(active_s, 1),
        "latency_ms_mean": round(lat.mean(), 2), "latency_ms_sd": round(lat.std(), 2),
        "latency_ms_median": round(lat.median(), 2),
        "latency_ms_p95": round(lat.quantile(.95), 2),
        "latency_ms_p99": round(lat.quantile(.99), 2),
        "cpu_pct_mean": round(cpu.mean(), 2), "cpu_pct_sd": round(cpu.std(), 2),
        "ram_pct_mean": round(w["RAM_%"].mean(), 2),
        "temp_c_mean": round(w.Temp_C.mean(), 2),
        "temp_c_max_instant": round(w.Temp_C.max(), 2),
        "cyclic_peak_temp_c_mean": round(peaks.mean(), 2),
        "cyclic_peak_temp_c_max": round(peaks.max(), 2),
        "n_cycles": int(len(peaks)),
        "power_w_mean": round(pw.mean(), 3), "power_w_sd": round(pw.std(), 3),
        "throughput_fps_effective": round(fps_eff, 2),
        "energy_per_inference_J": round(pw.mean() / fps_eff, 4),
        # Supply validation
        "bus_v_mean": round(w.Bus_V_mV.mean() / 1000, 3),
        "bus_v_sd": round(w.Bus_V_mV.std() / 1000, 4),
        "bat_i_ma_mean": round(w.Bat_I_mA.mean(), 2),
        "bat_i_ma_min": int(w.Bat_I_mA.min()), "bat_i_ma_max": int(w.Bat_I_mA.max()),
        "bat_pct_min": int(w.Bat_Pct.min()), "bat_pct_max": int(w.Bat_Pct.max()),
        "corr_busP_batI": round(pw.corr(w.Bat_I_mA), 3),
        "throttle_flags": sorted(w.Throttled.astype(str).unique().tolist()),
    }


def condition_summary(rs):
    out = {}
    for cond in ("bare-metal", "container"):
        members = [v for v in rs.values() if v["condition"] == cond]
        d = {}
        for k in ("latency_ms_mean", "cpu_pct_mean", "ram_pct_mean", "temp_c_mean",
                  "cyclic_peak_temp_c_mean", "power_w_mean",
                  "throughput_fps_effective", "energy_per_inference_J"):
            vals = [m[k] for m in members]
            d[k] = round(float(np.mean(vals)), 4)
            d[k + "_halfrange"] = round((max(vals) - min(vals)) / 2, 4)
        out[cond] = d
    b, c = out["bare-metal"], out["container"]
    out["delta"] = {
        "latency_ms_abs": round(c["latency_ms_mean"] - b["latency_ms_mean"], 2),
        "latency_pct": round(100 * (c["latency_ms_mean"] / b["latency_ms_mean"] - 1), 1),
        "cpu_pp": round(c["cpu_pct_mean"] - b["cpu_pct_mean"], 2),
        "ram_pp": round(c["ram_pct_mean"] - b["ram_pct_mean"], 2),
        "cyclic_peak_temp_c": round(c["cyclic_peak_temp_c_mean"] - b["cyclic_peak_temp_c_mean"], 2),
        "power_w_abs": round(c["power_w_mean"] - b["power_w_mean"], 3),
        "power_pct_of_system_input": round(100 * (c["power_w_mean"] / b["power_w_mean"] - 1), 1),
        "throughput_fps_abs": round(c["throughput_fps_effective"] - b["throughput_fps_effective"], 2),
        "throughput_pct": round(100 * (c["throughput_fps_effective"] / b["throughput_fps_effective"] - 1), 1),
        "energy_per_inference_pct": round(100 * (c["energy_per_inference_J"] / b["energy_per_inference_J"] - 1), 1),
    }
    return out


def ood_session():
    df = pd.read_csv(ROOT / "data/supplementary_session/basil_data.csv.gz")
    col = "Predicted_Class"
    counts = df[col].value_counts().to_dict()
    nb = np.where(df[col].str.lower() != "background")[0]
    # Frames within one intrusion of a novel object are not independent
    # observations; group them into events separated by >30 frames (~1 s).
    ev = np.split(nb, np.where(np.diff(nb) > 30)[0] + 1) if len(nb) else []
    return {
        "n_frames": int(len(df)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "pct_background": round(100 * counts.get("Background", 0) / len(df), 4),
        "n_nonbackground_frames": int(len(nb)),
        "n_distinct_events": len(ev),
        "event_frame_lengths": [int(len(e)) for e in ev],
        "event_classes": [sorted(df[col].iloc[e].unique().tolist()) for e in ev],
        "confidence_min": round(float(df.Confidence.iloc[nb].min()), 3) if len(nb) else None,
        "confidence_max": round(float(df.Confidence.iloc[nb].max()), 3) if len(nb) else None,
        "latency_ms_mean": round(float(df.Latency_ms.mean()), 2),
        "latency_ms_sd": round(float(df.Latency_ms.std()), 2),
        "latency_ms_max": round(float(df.Latency_ms.max()), 2),
        "cpu_pct_mean": round(float(df["CPU_%"].mean()), 2),
        "temp_c_mean": round(float(df.Temp_C.mean()), 2),
        "temp_c_max": round(float(df.Temp_C.max()), 2),
    }


def field_log():
    df = pd.read_csv(ROOT / "data/field_log/basil_data.csv.gz")
    col, conf = "Predicted_Class", df.Confidence
    low = conf < 0.5
    idx = np.where(low)[0]
    clusters = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []
    sizes = [len(c) for c in clusters]
    flip = df[col].ne(df[col].shift())
    flip.iloc[0] = False
    return {
        "n_frames": int(len(df)),
        "class_counts": {k: int(v) for k, v in df[col].value_counts().items()},
        "n_low_conf": int(low.sum()),
        "pct_low_conf": round(100 * low.mean(), 2),
        "pct_isolated_single_frame": round(100 * sum(1 for s in sizes if s == 1) / len(sizes), 1) if sizes else None,
        "pct_clusters_ge3": round(100 * sum(1 for s in sizes if s >= 3) / len(sizes), 1) if sizes else None,
        "max_cluster_len": int(max(sizes)) if sizes else 0,
        "flip_rate_low_conf_pct": round(100 * flip[low].mean(), 1),
        "flip_rate_high_conf_pct": round(100 * flip[~low].mean(), 1),
        "corr_conf_temp": round(conf.corr(df.Temp_C), 3),
        "corr_conf_cpu": round(conf.corr(df["CPU_%"]), 3),
        "corr_conf_latency": round(conf.corr(df.Latency_ms), 3),
    }


def architectures():
    base = ROOT / "basil_experiments/02_baseline_comparison/results"
    ro = pd.read_csv(base / "baseline_comparison_REAL_ONLY_final.csv")
    rp = pd.read_csv(base / "baseline_comparison_REAL_PLUS_PROXY_BALANCED_final.csv")
    m = ro.merge(rp, on="Model", suffixes=("_real", "_proxy"))
    m["delta_pp"] = (m.Accuracy_real - m.Accuracy_proxy) * 100
    return {
        "n_architectures": int(len(m)),
        "rows": [{"model": r.Model,
                  "real_only_acc": round(r.Accuracy_real * 100, 2),
                  "balanced_acc": round(r.Accuracy_proxy * 100, 2),
                  "delta_pp": round(r.delta_pp, 2)} for r in m.itertuples()],
        "n_favours_real": int((m.delta_pp > 0.001).sum()),
        "n_no_difference": int((m.delta_pp.abs() <= 0.001).sum()),
        "n_favours_proxy": int((m.delta_pp < -0.001).sum()),
        "median_abs_delta_pp": round(float(m.delta_pp.abs().median()), 2),
        "delta_pp_range": [round(float(m.delta_pp.min()), 2), round(float(m.delta_pp.max()), 2)],
    }


if __name__ == "__main__":
    rs = {k: run_stats(k) for k in RUNS}
    out = {"per_run": rs, "conditions": condition_summary(rs),
           "supplementary_session": ood_session(), "field_log": field_log(),
           "architectures": architectures()}
    (Path(__file__).resolve().parent / "numbers.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["conditions"], indent=2))
    print("\nOOD:", json.dumps(out["supplementary_session"], indent=2))
