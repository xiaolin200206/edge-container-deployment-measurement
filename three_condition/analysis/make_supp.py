#!/usr/bin/env python3
"""Generate the supplemental-material tables (LaTeX) from the raw runs and
out/numbers_v2.json. Run after analyse_v2.py. Writes out/supp_tables.tex."""
from __future__ import annotations
import json, os, re
from pathlib import Path
import pandas as pd


def csv_path(p):
    """Return p, or p + '.gz' if only the compressed copy exists (the repository
    stores the large per-frame logs gzipped; pandas reads either)."""
    p = Path(p)
    if p.exists():
        return p
    gz = p.with_name(p.name + ".gz")
    return gz if gz.exists() else p

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("RUNS_ROOT", HERE.parent / "runs"))
OUT = Path(os.environ.get("OUT_DIR", HERE.parent / "out"))
BASE = ROOT.parent
N = json.loads((OUT / "numbers_v2.json").read_text())
R, E = N["runs"], N["effects"]
BLOCKS = N["design"]["blocks"]
ORDER = [l for b in BLOCKS.values() for l in b["runs"]]


def tt(s):
    return "\\texttt{" + s.replace("_", "\\_") + "}"


def num(x, d):
    s = f"{x:.{d}f}"
    return s.replace("-", "\\textminus ")


def runinfo(label):
    txt = (ROOT / label / "RUN_INFO.txt").read_text()
    d = {}
    for line in txt.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


out = []
A = out.append

# S1 run schedule and integrity
A(r"\begin{table*}[!htbp]\centering\scriptsize")
A(r"\caption{Run schedule and integrity. Start times are local (UTC+8). Frames: all inferences in the run; analysed: after the 600\,s warm-up. "
  r"Battery current: mean (min, max) over the whole run, positive into the pack. Cap/UV: 1\,Hz samples with the CPU frequency capped below 2400\,MHz / under-voltage alarm set.}")
A(r"\label{tab:s-runs}\setlength{\tabcolsep}{3.2pt}")
A(r"\begin{tabular}{llllrrrrrrr}\toprule")
A(r"Block & Run & Condition & Start & Frames & Analysed & Rate (frame/s) & Active / sleep (s) & Battery I (mA) & Bus V (V) & Cap / UV\\\midrule")
for b, info in BLOCKS.items():
    for lab in info["runs"]:
        r = R[lab]; ri = runinfo(lab)
        start = re.sub(r"\.\d+", "", ri["Start"]).replace(" (+0800)", "")
        tot = len(pd.read_csv(csv_path(ROOT / lab / "basil_data.csv"), usecols=["Timestamp"]))
        A(f"{b} & {tt(lab)} & {r['condition']} & {start} & {tot:,} & {r['frames']:,} & {r['fps']:.3f} & "
          f"{r['active_len_s']:.2f} / {r['sleep_len_s']:.2f} & {r['bat_i_ma_mean_whole']:.1f} ({num(r['bat_i_ma_min_whole'], 0)}, {num(r['bat_i_ma_max_whole'], 0)}) & "
          f"{r['bus_v_mean']:.3f} & {r['cap_below_hw_max_samples']} / {r['undervolt_samples']}\\\\")
    A(r"\addlinespace")
A(r"\bottomrule\end{tabular}\end{table*}")

# S2 per-run results
COLS = [("latency_ms_mean", "Lat. mean", 2), ("latency_ms_median", "median", 2),
        ("latency_ms_p95", "p95", 2), ("latency_ms_p99", "p99", 2), ("latency_ms_sd", "SD", 2),
        ("cpu_pct_frame_mean", "CPU \\%", 2), ("ram_pct_mean", "RAM \\%", 2),
        ("power_w_active", "$P_\\mathrm{act}$ W", 3), ("power_w_sleep", "$P_\\mathrm{slp}$ W", 3),
        ("power_w_sleep_floor", "$P_\\mathrm{floor}$ W", 3),
        ("baseline_floor_per_inference_mJ", "$E_\\mathrm{base}$", 1),
        ("energy_per_inference_above_floor_mJ", "$E_\\mathrm{inc}$", 1),
        ("energy_per_inference_cycle_mJ", "$E_\\mathrm{cyc}$", 1),
        ("energy_per_cycle_J", "J/cycle", 1),
        ("temp_c_mean_active", "$T$ act.", 2), ("cyclic_peak_c_mean", "$T$ peak", 2),
        ("temp_c_max", "$T$ max", 1), ("fan_rpm_mean_active", "Fan rpm", 0),
        ("cpu_mhz_active_mean", "MHz", 0)]
A(r"\begin{sidewaystable*}[p]\centering\scriptsize\setlength{\tabcolsep}{1.9pt}")
A(r"\caption{Per-run results after warm-up. Latency in ms; CPU ms: CPU time per inference; $P_\mathrm{floor}$: settled sleep-phase floor; energies per inference in mJ, Eqs.~(1)--(3) of the paper; temperatures in $^\circ$C (active-phase mean, mean cyclic peak, maximum); fan and frequency are active-phase means.}")
A(r"\label{tab:s-perrun}")
A(r"\begin{tabular}{l" + "r" * len(COLS) + r"}\toprule")
A("Run & " + " & ".join(c[1] for c in COLS) + r"\\\midrule")
for lab in sorted(ORDER, key=lambda l: (["native", "matched", "legacy"].index(R[l]["condition"]), l)):
    A(tt(lab) + " & " + " & ".join(num(R[lab][k], d) for k, _, d in COLS) + r"\\")
A(r"\bottomrule\end{tabular}\end{sidewaystable*}")

# S3 per-block effects
KEYS = [("latency_ms_mean", "Latency, mean (ms)", 3), ("latency_ms_p95", "Latency, p95 (ms)", 3),
        ("latency_ms_p99", "Latency, p99 (ms)", 3), ("cpu_pct_frame_mean", "CPU utilisation (pp)", 3),
        ("power_w_active", "Power, active (W)", 3), ("power_w_sleep", "Power, sleep (W)", 3),
        ("energy_per_inference_above_floor_mJ", "Energy/inf., above floor (mJ)", 2),
        ("cpu_ms_per_inference", "CPU time/inference (ms)", 3),
        ("energy_per_inference_cycle_mJ", "Energy/inf., cycle (mJ)", 2),
        ("cyclic_peak_c_mean", "Cyclic peak ($^\\circ$C)", 3), ("fan_rpm_mean_active", "Fan, active (rpm)", 1)]
A(r"\begin{table*}[!htbp]\centering\scriptsize")
A(r"\caption{Within-block differences. Each effect in the paper is the mean of the three block differences shown; the 95\% interval uses $t_{0.975,2}=4.303$.}")
A(r"\label{tab:s-blocks}\setlength{\tabcolsep}{3pt}")
A(r"\begin{tabular}{l rrr r rrr r rrr r}\toprule")
A(r" & \multicolumn{4}{c}{$B-A$} & \multicolumn{4}{c}{$C-B$} & \multicolumn{4}{c}{$C-A$}\\")
A(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}")
A(r"Quantity & blk 1 & blk 2 & blk 3 & mean & blk 1 & blk 2 & blk 3 & mean & blk 1 & blk 2 & blk 3 & mean\\\midrule")
for k, lab, d in KEYS:
    cells = []
    for ek in ("B_minus_A", "C_minus_B", "C_minus_A"):
        e = E[k][ek]
        cells += [num(v, d) for v in e["per_block"]] + [num(e["mean"], d)]
    A(lab + " & " + " & ".join(cells) + r"\\")
A(r"\bottomrule\end{tabular}\end{table*}")

# S4 fingerprints
fps = {k: json.loads((BASE / f"fp_{k}.json").read_text()) for k in ("host", "matched", "legacy")}
A(r"\begin{table*}[!htbp]\centering\scriptsize")
A(r"\caption{Environment fingerprints: SHA-256 of the compiled extension module imported by each numerical package, and the ONNX Runtime build string. Host and matched container are identical in every field.}")
A(r"\label{tab:s-fp}")
A(r"\begin{tabular}{lll}\toprule Package & Host = matched container & Legacy container\\\midrule")
for pkg in ("onnxruntime", "numpy", "opencv-python-headless", "psutil"):
    h, l = fps["host"]["packages"][pkg], fps["legacy"]["packages"][pkg]
    assert fps["matched"]["packages"][pkg] == h
    A(f"{pkg} & {h['version']} & {l['version']}\\\\")
    A(f" & \\texttt{{{h['sha256'][:32]}}} & \\texttt{{{l['sha256'][:32]}}}\\\\")
    A(f" & \\texttt{{{h['sha256'][32:]}}} & \\texttt{{{l['sha256'][32:]}}}\\\\")
A(f"smbus2 (pure Python) & {fps['host']['packages']['smbus2']['version']} & {fps['legacy']['packages']['smbus2']['version']}\\\\")
A(f"Python / glibc & {fps['host']['python']} / {fps['host']['glibc']} & {fps['legacy']['python']} / {fps['legacy']['glibc']}\\\\")
hb = fps["host"]["ort_build_info"].replace("ORT Build Info: ", "")
lb = fps["legacy"]["ort_build_info"].replace("ORT Build Info: ", "")
A(r"\bottomrule\end{tabular}")
A(r"\par\medskip\raggedright\scriptsize ONNX Runtime build string, host and matched: \path{" + hb + r"}\par")
A(r"ONNX Runtime build string, legacy: \path{" + lb + r"}\par")
A(r"\end{table*}")

(OUT / "supp_tables.tex").write_text("\n".join(out) + "\n")
print("wrote", OUT / "supp_tables.tex")
