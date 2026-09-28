"""Recompute every quantity reported in the manuscript from the released raw logs.

Writes numbers.json. Nothing in the manuscript may state a figure that does not
appear here, so that every reported value is traceable to the deposited data.
"""
import json, math, os
import numpy as np, pandas as pd
from pathlib import Path

# Repository root: the parent of the directory holding this script, so a clone runs
# as-is from anywhere. Override with REPO_ROOT if the data lives elsewhere.
ROOT = Path(os.environ.get("REPO_ROOT", Path(__file__).resolve().parent.parent))
OUT = ROOT / "out"
PR = ROOT / "data/profiling_runs"
WARMUP, RUNEND = 600.0, 10800.0
RUNS = {"bare_metal_A": ("bare-metal", "night"), "bare_metal_B": ("bare-metal", "day"),
        "docker_A": ("container", "night"), "docker_B": ("container", "day")}


def elapsed(ts, t0):
    t = pd.to_datetime(ts, format="%H:%M:%S.%f")
    e = (t - t0).dt.total_seconds().to_numpy()
    return np.where(e < 0, e + 86400.0, e)


def run_stats(name):
    # The deposit ships gzipped CSVs; a working copy may be uncompressed. Accept either.
    d = PR / name
    src = d / "basil_data.csv.gz"
    if not src.exists():
        src = d / "basil_data.csv"
    df = pd.read_csv(src)
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
    # Loop period: the profiling loop is acquisition-bound, so inference is only
    # a fraction of it. Gaps above 1 s are duty-cycle sleeps and are excluded.
    gaps = np.diff(np.sort(w.el.to_numpy()))
    gaps = gaps[(gaps > 0) & (gaps < 1.0)]
    loop_ms = float(np.median(gaps) * 1000)
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
        "loop_period_ms_median": round(loop_ms, 1),
        "inference_share_of_loop_pct": round(100 * lat.mean() / loop_ms, 1),
    }


def condition_summary(rs):
    out = {}
    for cond in ("bare-metal", "container"):
        members = [v for v in rs.values() if v["condition"] == cond]
        d = {}
        for k in ("latency_ms_mean", "latency_ms_median", "latency_ms_p95",
                  "latency_ms_p99", "cpu_pct_mean", "ram_pct_mean", "temp_c_mean",
                  "cyclic_peak_temp_c_mean", "power_w_mean",
                  "throughput_fps_effective", "energy_per_inference_J",
                  "loop_period_ms_median", "inference_share_of_loop_pct"):
            vals = [m[k] for m in members]
            d[k] = round(float(np.mean(vals)), 4)
            d[k + "_halfrange"] = round((max(vals) - min(vals)) / 2, 4)
        out[cond] = d
    b, c = out["bare-metal"], out["container"]
    # Which effects actually replicate: an effect replicates only if both
    # native runs sit on the same side of both containerised runs.
    rep = {}
    for k in ("latency_ms_mean", "latency_ms_p95", "latency_ms_p99", "cpu_pct_mean",
              "ram_pct_mean", "cyclic_peak_temp_c_mean", "power_w_mean",
              "throughput_fps_effective", "energy_per_inference_J"):
        bv = [v[k] for v in rs.values() if v["condition"] == "bare-metal"]
        cv = [v[k] for v in rs.values() if v["condition"] == "container"]
        rep[k] = bool(max(bv) < min(cv) or min(bv) > max(cv))
    out["replicates_across_both_pairs"] = rep
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
        "latency_p95_ms_abs": round(c["latency_ms_p95"] - b["latency_ms_p95"], 2),
        "latency_p99_ms_abs": round(c["latency_ms_p99"] - b["latency_ms_p99"], 2),
        "latency_p99_pct": round(100 * (c["latency_ms_p99"] / b["latency_ms_p99"] - 1), 1),
        # Five frames of confirmation cost five times the MEAN difference, not a
        # function of the tail: the accumulation time is a sum, not a maximum.
        "five_frame_confirmation_penalty_ms": round(5 * (c["latency_ms_mean"] - b["latency_ms_mean"]), 1),
        "loop_period_ms_abs": round(c["loop_period_ms_median"] - b["loop_period_ms_median"], 1),
        "inference_share_of_loop_pp": round(c["inference_share_of_loop_pct"] - b["inference_share_of_loop_pct"], 1),
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
        "confirmed_class_counts": ({k: int(v) for k, v in df.Confirmed_Class.value_counts().items()}
                                   if "Confirmed_Class" in df else None),
        # Zero events in n trials: the 95% one-sided upper bound is the rule of
        # three, 3/n, and it assumes independence the frames do not have.
        "rule_of_three_upper_bound_per_frame": float(f"{3/len(df):.2g}"),
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
    ev = pd.read_csv(ROOT / "data/field_log/cycle_events.csv")
    col, conf = "Predicted_Class", df.Confidence
    # Wall-clock span covers three duty cycles including two 15 s sleeps, so the
    # effective frame rate must be computed against active time, not wall clock.
    t0 = pd.to_datetime(ev.Timestamp.iloc[0], format="%H:%M:%S.%f")
    def el(ts):
        e = (pd.to_datetime(ts, format="%H:%M:%S.%f") - t0).dt.total_seconds()
        return np.where(e < 0, e + 86400, e)
    ev["el"] = el(ev.Timestamp)
    starts = ev[ev.Event == "CYCLE_ACTIVE_START"].el.to_numpy()
    sleeps = ev[ev.Event == "CYCLE_SLEEP_START"].el.to_numpy()
    dt = pd.to_datetime(df.Timestamp, format="%H:%M:%S.%f")
    last = float(np.where((dt - t0).dt.total_seconds() < 0,
                          (dt - t0).dt.total_seconds() + 86400,
                          (dt - t0).dt.total_seconds()).max())
    spans = [(s_, e_) for s_, e_ in zip(starts, list(sleeps) + [last])]
    active_s = float(sum(e_ - s_ for s_, e_ in spans))
    peaks = ev[ev.Event == "CYCLE_SLEEP_START"].Temp_C
    low = conf < 0.5
    idx = np.where(low)[0]
    clusters = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []
    sizes = [len(c) for c in clusters]
    flip = df[col].ne(df[col].shift())
    flip.iloc[0] = False
    return {
        "n_frames": int(len(df)),
        "wall_clock_s": round(last, 1),
        "active_inference_s": round(active_s, 1),
        "n_duty_cycles": int(len(starts)),
        "throughput_fps_effective": round(len(df) / active_s, 1),
        # In-situ thermal behaviour. Reported because it measures directly the
        # laboratory-to-field gap that Section 5.1 would otherwise have to assume.
        "temp_c_mean": round(float(df.Temp_C.mean()), 1),
        "temp_c_max": round(float(df.Temp_C.max()), 1),
        "temp_c_at_start": round(float(df.Temp_C.iloc[0]), 1),
        "cycle_peak_temps_c": [round(float(x), 1) for x in peaks],
        "pct_frames_above_70c": round(100 * float((df.Temp_C > 70).mean()), 1),
        "n_frames_above_75c": int((df.Temp_C > 75).sum()),
        "margin_to_throttle_c": round(82 - float(df.Temp_C.max()), 1),
        "cpu_pct_mean": round(float(df["CPU_%"].mean()), 1),
        "latency_ms_mean": round(float(df.Latency_ms.mean()), 2),
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
    d = m.delta_pp.to_numpy()
    N_VAL = 340
    pos, neg = int((d > 0.01).sum()), int((d < -0.01).sum())
    k, n = pos, pos + neg
    # Two-sided exact sign test on the non-tied architectures.
    pmf = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    p_sign = min(1.0, 2 * min(sum(pmf[:k + 1]), sum(pmf[k:])))
    # Two-sided exact Wilcoxon signed-rank test, zeros dropped (Wilcoxon's own
    # handling), ranks averaged over ties, null distribution enumerated in full.
    # The released CSVs carry accuracy to four decimals, so deltas are exact to
    # two decimal places; round before ranking so that float noise does not
    # break genuine ties in |delta|.
    dr = np.round(d, 4)
    nz = dr[np.abs(dr) > 0.01]
    order = np.argsort(np.abs(nz))
    ranks = np.empty(len(nz))
    absv = np.abs(nz)[order]
    i = 0
    while i < len(absv):
        j = i
        while j + 1 < len(absv) and absv[j + 1] == absv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    w_plus = float(ranks[nz > 0].sum())
    nn = len(nz)
    dist = {0.0: 1}
    for r in ranks:
        nxt = {}
        for tot, cnt in dist.items():
            nxt[tot] = nxt.get(tot, 0) + cnt
            nxt[tot + r] = nxt.get(tot + r, 0) + cnt
        dist = nxt
    total = 2 ** nn
    mu = ranks.sum() / 2.0
    dev = abs(w_plus - mu)
    p_w = sum(c for t, c in dist.items() if abs(t - mu) >= dev - 1e-9) / total
    return {
        "wilcoxon_signed_rank_p_two_sided": round(min(1.0, p_w), 3),
        "wilcoxon_n_nonzero": int(nn),
        "wilcoxon_w_plus": w_plus,
        "n_architectures": int(len(m)),
        "median_delta_pp": round(float(np.median(d)), 2),
        "mean_delta_pp": round(float(d.mean()), 2),
        "n_favours_real_only": pos, "n_favours_proxy": neg,
        "n_tied": int(len(d) - pos - neg),
        "sign_test_p_two_sided": round(p_sign, 3),
        "validation_set_size": N_VAL,
        "pp_per_validation_image": round(100 / N_VAL, 3),
        "delta_in_validation_images": {
            ("ViT-B/16" if r.Model == "ViT-Tiny" else r.Model): round(r.delta_pp / (100 / N_VAL), 1)
            for r in m.itertuples()},
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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT / 'numbers.json'}")
    print(json.dumps(out["conditions"], indent=2))
    print("\nOOD:", json.dumps(out["supplementary_session"], indent=2))
