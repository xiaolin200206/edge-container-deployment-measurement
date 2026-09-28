#!/usr/bin/env python3
"""Unattended runner for the nine-run schedule.

Handles the parts that are easy to get wrong at 3 a.m.: waiting for the board
to cool between runs, refusing to start while the pack is charging, choosing
the right invocation for each condition, verifying each run before moving on,
and resuming where it left off.

    python3 orchestrate.py --block 1     # the three runs of block 1
    python3 orchestrate.py --resume      # continue wherever it stopped
    python3 orchestrate.py --all         # all nine, waiting for the right
                                         # time of day between blocks

ALWAYS start it inside tmux. If the SSH session drops, anything not in tmux
dies with it and you lose the run in progress:

    tmux new -s runs
    python3 orchestrate.py --block 1
    # detach with Ctrl-b then d ; reattach later with: tmux attach -t runs

It stops at the first failure rather than pressing on, because a broken run
early in a block makes everything after it harder to interpret.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE / "run_plan.json"
RUNS = HERE / "runs"
LOG = HERE / "orchestrator.log"

VENV_PY = os.environ.get("VENV_PY", str(Path.home() / "venv/bin/python"))
COOL_C = float(os.environ.get("COOL_C", 50.0))
COOL_MAX_MIN = float(os.environ.get("COOL_MAX_MIN", 60.0))
DURATION = float(os.environ.get("DURATION", 10800.0))

# Start windows, local time. A block only begins inside its window; runs that
# follow within a block just continue, which is the point of a block.
WINDOWS = {"night": (20, 6), "day": (8, 16)}


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def temp_c() -> float:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        return float("nan")


def in_window(period: str) -> bool:
    lo, hi = WINDOWS[period]
    h = datetime.now().hour
    return (lo <= h or h < hi) if lo > hi else (lo <= h < hi)


def wait_for_window(period: str) -> None:
    if in_window(period):
        return
    lo, hi = WINDOWS[period]
    log(f"waiting for the {period} window ({lo:02d}:00-{hi:02d}:00) to open")
    while not in_window(period):
        time.sleep(300)
    log(f"{period} window open")


def wait_cool() -> bool:
    t = temp_c()
    if t <= COOL_C:
        log(f"board at {t:.1f} C, no cooling needed")
        return True
    log(f"board at {t:.1f} C, waiting for <= {COOL_C:.0f} C")
    deadline = time.time() + COOL_MAX_MIN * 60
    while time.time() < deadline:
        time.sleep(60)
        t = temp_c()
        if t <= COOL_C:
            log(f"cooled to {t:.1f} C")
            return True
    log(f"STILL {t:.1f} C after {COOL_MAX_MIN:.0f} min — something is wrong "
        f"(ambient too high, or a process is loading the CPU)")
    return False


BATT_MAX_MA = float(os.environ.get("BATT_MAX_MA", 50.0))
BATT_MAX_H = float(os.environ.get("BATT_MAX_H", 5.0))


def battery() -> tuple[int, int] | None:
    """(percent, current mA) via profile_run.py --selftest, or None if unreadable."""
    try:
        out = subprocess.run([VENV_PY, str(HERE / "profile_run.py"), "--selftest"],
                             capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"Battery:\s*(\d+)%\s+current\s+([+-]?\d+)\s*mA", out)
        return (int(m.group(1)), int(m.group(2))) if m else None
    except Exception:
        return None


def wait_float() -> bool:
    """Wait until the pack stops bulk-charging, the way wait_cool waits for heat.

    The charger stays connected throughout, as in the first round. That is only
    sound once the pack is floating; a pack still taking charge current puts it
    into Bus_P_mW. profile_run.py refuses to start in that state, so wait here
    instead of letting the run abort.
    """
    b = battery()
    if b is None:
        log("could not read the battery — check the UPS with --selftest")
        return False
    pct, ma = b
    if ma <= BATT_MAX_MA:
        log(f"battery {pct}% at {ma:+d} mA, floating")
        return True
    log(f"battery {pct}% charging at {ma:+d} mA — waiting for it to float "
        f"(<= {BATT_MAX_MA:+.0f} mA)")
    deadline = time.time() + BATT_MAX_H * 3600
    last = 0.0
    while time.time() < deadline:
        time.sleep(120)
        b = battery()
        if b is None:
            continue
        pct, ma = b
        if ma <= BATT_MAX_MA:
            log(f"battery floating: {pct}% at {ma:+d} mA")
            return True
        if time.time() - last > 1800:
            log(f"     still charging: {pct}% at {ma:+d} mA")
            last = time.time()
    log(f"battery still charging after {BATT_MAX_H:.0f} h — charger or pack fault?")
    return False


def done(label: str) -> bool:
    info = RUNS / label / "RUN_INFO.txt"
    return info.exists() and "Complete         : yes" in info.read_text()


def command_for(condition: str, replicate: int, label: str) -> list[str]:
    out_host = RUNS / label
    if condition == "native":
        return [VENV_PY, str(HERE / "profile_run.py"),
                "--condition", "native", "--replicate", str(replicate),
                "--duration", str(DURATION), "--fps", str(FPS), "--out", str(out_host)]
    # infer:matched is debian-slim with the distribution python3 in a venv, so
    # call the venv interpreter explicitly rather than relying on PATH.
    image, py = {"matched": ("infer:matched", "/venv/bin/python"),
                 "legacy": ("infer:legacy", "python3")}[condition]
    return ["docker", "run", "--rm",
            "--device", "/dev/video0", "--device", "/dev/i2c-1",
            "-v", "/etc/localtime:/etc/localtime:ro",
            "-v", f"{RUNS}:/out", image,
            py, "profile_run.py",
            "--condition", condition, "--replicate", str(replicate),
            "--duration", str(DURATION), "--fps", str(FPS), "--out", f"/out/{label}"]


FPS = float(os.environ.get("FPS", 15.0))


def pin_camera(label: str) -> bool:
    """Stop the camera trading frame rate for exposure, and record its state.

    exposure_dynamic_framerate defaults to 0 on this camera but was found at 1,
    so it cannot be trusted to stay off: set it before every run, from the
    host, identically for all three conditions.
    """
    if shutil.which("v4l2-ctl") is None:
        log("FAIL: v4l2-ctl missing (sudo apt install v4l-utils)")
        return False
    subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "-c",
                    "exposure_dynamic_framerate=0"], capture_output=True)
    out = subprocess.run(["v4l2-ctl", "-d", "/dev/video0", "--list-ctrls"],
                         capture_output=True, text=True).stdout
    (RUNS / f"{label}.camera.txt").write_text(out)
    ok = "exposure_dynamic_framerate" not in out or \
         any("exposure_dynamic_framerate" in ln and "value=0" in ln for ln in out.splitlines())
    log(f"camera pinned: dynamic framerate {'off' if ok else 'STILL ON'}")
    return ok


def verify(label: str) -> bool:
    d = RUNS / label
    info = d / "RUN_INFO.txt"
    if not info.exists():
        log(f"FAIL {label}: no RUN_INFO.txt")
        return False
    txt = info.read_text()
    if "Complete         : yes" not in txt:
        log(f"FAIL {label}: run did not reach full duration")
        return False
    data, power = d / "basil_data.csv", d / "power_log.csv"
    if not data.exists() or not power.exists():
        log(f"FAIL {label}: missing csv")
        return False
    n_data = sum(1 for _ in data.open()) - 1
    lines = power.read_text().splitlines()
    n_power = len(lines) - 1
    n_sleep = sum(1 for ln in lines[1:] if ",sleep," in ln)
    empty_bus = sum(1 for ln in lines[1:5000] if ln.split(",")[2] == "")
    log(f"     {label}: {n_data} frames, {n_power} power rows, {n_sleep} in sleep")
    ok = True
    expected = 144 * 60 * FPS * DURATION / 10800.0
    if not (0.93 * expected <= n_data <= 1.05 * expected):
        log(f"FAIL {label}: {n_data} frames, expected ~{expected:.0f} at {FPS:.0f} frame/s "
            f"— the camera rate changed during the run"); ok = False
    if n_power < 0.8 * DURATION:
        log(f"FAIL {label}: only {n_power} power rows, expected ~{int(DURATION)}"); ok = False
    if n_sleep < 500:
        log(f"FAIL {label}: only {n_sleep} sleep-phase rows"); ok = False
    if empty_bus > 100:
        log(f"FAIL {label}: {empty_bus} empty Bus_ readings in the first 5000 "
            f"rows — I2C dropped out"); ok = False
    if not (d / "provenance.json").exists():
        log(f"FAIL {label}: no provenance.json"); ok = False
    return ok


def preflight() -> bool:
    ok = True
    if not Path(VENV_PY).exists():
        log(f"FAIL: no host interpreter at {VENV_PY} (set VENV_PY)"); ok = False
    if shutil.which("docker") is None:
        log("FAIL: docker not on PATH"); ok = False
    else:
        have = subprocess.run(["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
                              capture_output=True, text=True).stdout
        for img in ("infer:matched", "infer:legacy"):
            if img not in have:
                log(f"FAIL: image {img} not built"); ok = False
    if not Path("/dev/video0").exists():
        log("FAIL: /dev/video0 missing"); ok = False
    if not PLAN.exists():
        log(f"FAIL: {PLAN} missing"); ok = False
    free_gb = shutil.disk_usage(HERE).free / 1e9
    if free_gb < 3:
        log(f"FAIL: only {free_gb:.1f} GB free, need ~3"); ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--block", type=int, choices=[1, 2, 3])
    g.add_argument("--resume", action="store_true")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--ignore-window", action="store_true",
                    help="start regardless of time of day (records the real time anyway)")
    a = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    plan = json.loads(PLAN.read_text())
    log("=" * 64)
    log(f"orchestrator start: {'block %d' % a.block if a.block else ('all' if a.all else 'resume')}")

    if not preflight():
        sys.exit("preflight failed — fix the above and rerun")

    if a.block:
        todo = [r for r in plan if r["block"] == f"Block {a.block}"]
    else:
        todo = plan

    pending = [r for r in todo if not done(r["label"])]
    for r in todo:
        if done(r["label"]):
            log(f"skip {r['label']} (already complete)")
    if not pending:
        log("nothing to do — every run in scope is already complete")
        return
    log(f"{len(pending)} run(s) to go: " + ", ".join(r["label"] for r in pending))
    log(f"estimated {len(pending) * (DURATION / 3600 + 0.5):.1f} h including cooling")

    current_block = None
    for r in pending:
        label, cond, rep = r["label"], r["condition"], r["run"]
        if r["block"] != current_block:
            current_block = r["block"]
            if not a.ignore_window:
                wait_for_window(r["period"])
        log("-" * 64)
        log(f"run {r['run']}/9  {current_block}  {r['period']}  {label}")

        if not wait_float():
            sys.exit("aborting: battery never reached float")
        if not wait_cool():
            sys.exit("aborting: board would not cool down")
        if not pin_camera(label):
            sys.exit("aborting: could not pin the camera frame rate")

        d = RUNS / label
        if d.exists() and any(d.iterdir()):
            log(f"{d} exists and is not empty — removing an incomplete attempt")
            shutil.rmtree(d)

        cmd = command_for(cond, rep, label)
        log("$ " + " ".join(cmd))
        t0 = time.time()
        proc = subprocess.run(cmd)
        log(f"exited {proc.returncode} after {(time.time()-t0)/3600:.2f} h")
        if proc.returncode != 0:
            sys.exit(f"aborting: {label} exited {proc.returncode}")
        if not verify(label):
            sys.exit(f"aborting: {label} failed verification — see above")
        log(f"OK {label}")

    log("=" * 64)
    remaining = [r["label"] for r in plan if not done(r["label"])]
    if remaining:
        log(f"block finished. still outstanding: {', '.join(remaining)}")
    else:
        log("ALL NINE RUNS COMPLETE")
        log("now: tar czf profiling_v2.tar.gz runs/   and send it to Claude")


if __name__ == "__main__":
    main()
