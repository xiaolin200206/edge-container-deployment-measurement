#!/usr/bin/env python3
"""Three-condition profiling harness.

Runs the deployed inference loop under a fixed duty cycle for a fixed duration
and records three things:

  basil_data.csv    one row per inference frame (as in the first round of runs)
  power_log.csv     supply telemetry at 1 Hz across the WHOLE run, active and
                    sleep alike. The first round logged nothing during sleep,
                    so no idle baseline could be subtracted and per-inference
                    energy carried an unknown offset. This file fixes that.
  cycle_events.csv  duty-cycle transitions
  provenance.json   full software stack, written by provenance.py
  RUN_INFO.txt      human-readable summary

Usage
-----
    # check the I2C wiring before committing three hours to a run
    python3 profile_run.py --selftest

    # a real run
    python3 profile_run.py --condition native --replicate 1 --out runs/native_1

Conditions
----------
    native   host interpreter and runtime
    matched  container built from Dockerfile.matched  (versions pinned to host)
    legacy   container built from Dockerfile.legacy   (as originally deployed)

The condition name is recorded in the manifest but does not change behaviour:
what differs between conditions is the environment this script runs inside.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = str(HERE / "basil_mobilenet.onnx")
INFERENCE_SIZE = 224
CLASS_NAMES = ["background", "basil_healthy", "disease"]


# --------------------------------------------------------------- UPS telemetry
class UPS:
    """Waveshare UPS HAT (E), I2C address 0x2D.

    Register map (Waveshare wiki, "UPS HAT (E) Register"), all 16-bit
    little-endian:

        0x10  Type-C VBUS voltage   mV   unsigned
        0x12  Type-C VBUS current   mA   unsigned
        0x14  Type-C VBUS power     mW   unsigned
        0x20  battery total voltage mV   unsigned
        0x22  battery current       mA   SIGNED (+ charging, - discharging)
        0x24  battery percentage    %    unsigned

    Note that 0x12 exists. The first round of runs did not read it, and the
    manuscript stated the module exposes no bus-current register. That was
    wrong. Reading it here allows P = V x I to be checked against the module's
    own power register, which is the only cross-check available without an
    external meter.
    """

    ADDR = 0x2D

    def __init__(self, bus: int = 1):
        from smbus2 import SMBus  # imported late so --selftest can report cleanly
        self.bus = SMBus(bus)

    def _u16(self, reg: int) -> int:
        d = self.bus.read_i2c_block_data(self.ADDR, reg, 2)
        return d[0] | (d[1] << 8)

    def _s16(self, reg: int) -> int:
        v = self._u16(reg)
        return v - 65536 if v > 32767 else v

    def read(self) -> dict:
        return {
            "bus_v_mv": self._u16(0x10),
            "bus_i_ma": self._u16(0x12),
            "bus_p_mw": self._u16(0x14),
            "bat_v_mv": self._u16(0x20),
            "bat_i_ma": self._s16(0x22),
            "bat_pct": self._u16(0x24),
        }


EMPTY_UPS = {k: "" for k in
             ("bus_v_mv", "bus_i_ma", "bus_p_mw", "bat_v_mv", "bat_i_ma", "bat_pct")}


# ------------------------------------------------------------- system monitors
def cpu_temp() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return float("nan")


def throttled_state() -> str:
    """Populated by vcgencmd, which the container images do not ship.

    In the first round this silently logged 'Unknown' for every containerised
    frame while reading 'No' natively, and the manuscript reported "no
    throttling" for all four runs. Keep the field, but the run script now also
    records, in RUN_INFO.txt, whether vcgencmd was available at all.
    """
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"],
                             capture_output=True, text=True, timeout=2).stdout
        val = out.strip().split("=")[-1]
        return "No" if val in ("0x0", "0") else val
    except Exception:
        return "Unknown"


def _hwmon(name: str) -> Path | None:
    """Find a hwmon directory by its reported name (e.g. rpi_volt, pwmfan)."""
    try:
        for d in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
            try:
                if (d / "name").read_text().strip() == name:
                    return d
            except Exception:
                continue
    except Exception:
        pass
    return None


def _read_int(p: Path | None) -> int | None:
    try:
        return int(p.read_text().strip()) if p is not None else None
    except Exception:
        return None


CPUFREQ = Path("/sys/devices/system/cpu/cpu0/cpufreq")
HW_VOLT = _hwmon("rpi_volt")
HW_FAN = _hwmon("pwmfan")
CPU_HW_MAX = _read_int(CPUFREQ / "cpuinfo_max_freq")


def uv_alarm() -> int | None:
    """Under-voltage alarm from the rpi_volt hwmon (1 = supply sagged)."""
    if HW_VOLT is None:
        return None
    for f in sorted(HW_VOLT.glob("in*_lcrit_alarm")):
        v = _read_int(f)
        if v is not None:
            return v
    return None


def platform_state() -> dict:
    """Throttle-relevant platform state, read from /sys in EVERY condition.

    Pi 5 exposes no firmware get_throttled node, and vcgencmd is absent from the
    images, so the first harness could only ever log 'Unknown' inside a
    container. Everything below lives in /sys, which Docker mounts read-only
    into containers by default, so all three conditions are instrumented from
    the same source:

      cpu_mhz      current CPU frequency
      cap_mhz      current frequency ceiling; the kernel's thermal framework
                   lowers this when it throttles, so cap < hw max means capped
      uv           under-voltage alarm
      fan_rpm/pwm  the fan, if one is fitted — it changes both the thermal
                   result and the power draw, so it must be on the record
    """
    cur = _read_int(CPUFREQ / "scaling_cur_freq")
    cap = _read_int(CPUFREQ / "scaling_max_freq")
    fan_rpm = _read_int(HW_FAN / "fan1_input") if HW_FAN else None
    fan_pwm = _read_int(HW_FAN / "pwm1") if HW_FAN else None
    return {
        "cpu_mhz": cur // 1000 if cur else None,
        "cap_mhz": cap // 1000 if cap else None,
        "uv": uv_alarm(),
        "fan_rpm": fan_rpm,
        "fan_pwm": fan_pwm,
    }


def throttle_flag(st: dict) -> str:
    capped = (st["cap_mhz"] is not None and CPU_HW_MAX is not None
              and st["cap_mhz"] * 1000 < CPU_HW_MAX)
    uv = st["uv"] == 1
    if st["cap_mhz"] is None and st["uv"] is None:
        return "Unknown"
    return "No" if not (capped or uv) else "+".join(
        x for x, on in (("capped", capped), ("undervolt", uv)) if on)


def throttle_source() -> str:
    parts = []
    parts.append("cpufreq" if (CPUFREQ / "scaling_max_freq").exists() else "no-cpufreq")
    parts.append("rpi_volt" if HW_VOLT else "no-rpi_volt")
    parts.append("pwmfan" if HW_FAN else "no-fan-hwmon")
    return "sysfs " + "+".join(parts) + f" (hw max {CPU_HW_MAX//1000 if CPU_HW_MAX else '?'} MHz)"


def read_throttle() -> str:
    return throttle_flag(platform_state())


def have_vcgencmd() -> bool:
    try:
        subprocess.run(["vcgencmd", "get_throttled"],
                       capture_output=True, timeout=2)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ the sampler
class PowerSampler(threading.Thread):
    """1 Hz supply telemetry for the whole run, including sleep phases."""

    def __init__(self, ups: UPS | None, path: Path, state: dict):
        super().__init__(daemon=True)
        self.ups, self.path, self.state = ups, path, state
        self.stop_flag = threading.Event()
        self.latest = dict(EMPTY_UPS)
        self.throttled = "Unknown"
        self.pstate = {"cpu_mhz": None, "cap_mhz": None, "uv": None,
                       "fan_rpm": None, "fan_pwm": None}
        self._last_cpu = None

    def _cpu_pct(self) -> str:
        """CPU busy % over the last sampling interval, from raw cpu_times.

        psutil.cpu_percent(interval=None) keeps one module-level 'last call'
        state; the main loop also calls it once per frame, so sharing it would
        chop this thread's window into fragments. cpu_times() is stateless.
        """
        import psutil
        t = psutil.cpu_times()
        idle = t.idle + getattr(t, "iowait", 0.0)
        total = sum(t)
        if self._last_cpu is None:
            self._last_cpu = (idle, total)
            return ""
        di, dt = idle - self._last_cpu[0], total - self._last_cpu[1]
        self._last_cpu = (idle, total)
        return f"{100.0 * (1 - di / dt):.1f}" if dt > 0 else ""

    def run(self):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Timestamp", "Phase", "Bus_V_mV", "Bus_I_mA", "Bus_P_mW",
                        "Bat_V_mV", "Bat_I_mA", "Bat_Pct", "Temp_C", "CPU_%",
                        "Throttled", "CPU_MHz", "Cap_MHz", "UV", "Fan_RPM", "Fan_PWM"])
            next_t = time.time()
            while not self.stop_flag.is_set():
                if self.ups is not None:
                    try:
                        self.latest = self.ups.read()
                    except Exception:
                        self.latest = dict(EMPTY_UPS)
                r = self.latest
                self.pstate = platform_state()
                self.throttled = throttle_flag(self.pstate)
                ps = {k: ("" if v is None else v) for k, v in self.pstate.items()}
                w.writerow([
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    self.state.get("phase", "?"),
                    r["bus_v_mv"], r["bus_i_ma"], r["bus_p_mw"],
                    r["bat_v_mv"], r["bat_i_ma"], r["bat_pct"],
                    f"{cpu_temp():.1f}",
                    self._cpu_pct(),
                    self.throttled,
                    ps["cpu_mhz"], ps["cap_mhz"], ps["uv"], ps["fan_rpm"], ps["fan_pwm"],
                ])
                f.flush()
                next_t += 1.0
                time.sleep(max(0.0, next_t - time.time()))


# ------------------------------------------------------------------- inference
class Runner:
    def __init__(self, args):
        import cv2
        import onnxruntime as ort
        self.cv2 = cv2
        self.args = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)

        self.session = ort.InferenceSession(MODEL_PATH,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        dummy = np.zeros((1, 3, INFERENCE_SIZE, INFERENCE_SIZE), dtype=np.float32)
        self.session.run([self.output_name], {self.input_name: dummy})

        self.is_rgb_input = False
        self.cap = cv2.VideoCapture(args.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # The loop is camera-bound, so the camera's frame rate sets the CPU load,
        # temperature and power of the whole run. With auto exposure and
        # exposure_dynamic_framerate on, the camera halves its rate when the
        # room gets dark — the first block ran at 15 frame/s in the dark and a
        # run that straddled the lights coming on jumped to 30 mid-run. Request
        # a fixed rate; the orchestrator also turns dynamic framerate off.
        self.cap.set(cv2.CAP_PROP_FPS, float(args.fps))
        self.fps_reported = self.cap.get(cv2.CAP_PROP_FPS)
        if not self.cap.isOpened():
            sys.exit("camera did not open")

    # preprocessing is byte-for-byte the deployed path: Resize(256) ->
    # CenterCrop(224) -> RGB -> ImageNet normalise. Do not "clean this up":
    # any change here changes the measured latency.
    def preprocess(self, frame):
        cv2 = self.cv2
        h, w = frame.shape[:2]
        if h < w:
            new_h, new_w = 256, int(w * (256 / h))
        else:
            new_w, new_h = 256, int(h * (256 / w))
        img = cv2.resize(frame, (new_w, new_h))
        sy, sx = (new_h - 224) // 2, (new_w - 224) // 2
        img = img[sy:sy + 224, sx:sx + 224]
        if not self.is_rgb_input:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        return np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    def infer(self, frame):
        tensor = self.preprocess(frame)
        t0 = time.time()
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0]
        latency = (time.time() - t0) * 1000
        logits = raw[0]
        e = np.exp(logits - np.max(logits))
        probs = e / e.sum()
        i = int(np.argmax(probs))
        return CLASS_NAMES[i], float(probs[i]), latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["native", "matched", "legacy"])
    ap.add_argument("--replicate", type=int, default=1)
    ap.add_argument("--duration", type=float, default=10800.0, help="seconds")
    ap.add_argument("--active", type=float, default=60.0)
    ap.add_argument("--sleep", type=float, default=15.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--fps", type=float, default=15.0,
                    help="camera frame rate to request (all runs use 15)")
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--no-ups", action="store_true",
                    help="run without supply telemetry (not for real runs)")
    ap.add_argument("--allow-charging", action="store_true",
                    help="start even if the pack is bulk-charging (invalidates power data)")
    ap.add_argument("--selftest", action="store_true",
                    help="read the UPS once, print it, and exit")
    a = ap.parse_args()

    if a.selftest:
        try:
            u = UPS(a.i2c_bus)
            r = u.read()
        except Exception as e:
            sys.exit(f"UPS read FAILED: {e}\n"
                     f"Check: i2c enabled, module present, "
                     f"`i2cdetect -y {a.i2c_bus}` shows 2d, smbus2 installed.")
        implied = 1000 * r["bus_p_mw"] / r["bus_v_mv"] if r["bus_v_mv"] else float("nan")
        print(json.dumps(r, indent=2))
        print(f"\nimplied bus current from P/V : {implied:8.1f} mA")
        print(f"bus current register (0x12)  : {r['bus_i_ma']:8.1f} mA")
        print(f"\nBattery: {r['bat_pct']}%  current {r['bat_i_ma']:+d} mA "
              f"({'CHARGING - do not start a run' if r['bat_i_ma'] > 50 else 'float/idle - fine'})")
        print("\nSanity checks:")
        ok = True
        for label, cond, hint in [
            ("bus voltage 4.5-21 V", 4500 <= r["bus_v_mv"] <= 21000,
             "a value near 0 or above 21000 means the byte order is wrong"),
            ("battery voltage 10-18 V", 10000 <= r["bat_v_mv"] <= 18000,
             "4S pack should read about 16800 mV when full"),
            ("battery percent 0-100", 0 <= r["bat_pct"] <= 100, ""),
            ("battery not bulk-charging (|I| <= 50 mA)", abs(r["bat_i_ma"]) <= 50,
             "the pack is still charging; wait for it to float before running"),
            ("P/V agrees with 0x12 within 15%",
             abs(implied - r["bus_i_ma"]) <= 0.15 * max(implied, 1),
             "large disagreement means one register is being misread"),
        ]:
            mark = "PASS" if cond else "FAIL"
            if not cond:
                ok = False
            print(f"  [{mark}] {label}" + (f"   <- {hint}" if not cond and hint else ""))
        sys.exit(0 if ok else 1)

    if not a.condition:
        sys.exit("--condition is required for a real run")
    if a.out is None:
        a.out = f"runs/{a.condition}_{a.replicate}"

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        sys.exit(f"{out} is not empty; refusing to overwrite a previous run")

    ups = None
    if not a.no_ups:
        try:
            ups = UPS(a.i2c_bus)
            ups.read()
        except Exception as e:
            sys.exit(f"UPS unavailable ({e}). Run --selftest, or pass --no-ups "
                     f"if you really intend a run with no power data.")

    # The charging input stays connected, as in the first round of runs. That is
    # only sound while the pack is floating: if it is still bulk-charging, the
    # charge current lands in Bus_P_mW and the whole run is unusable. Check
    # before committing three hours, not after.
    if ups is not None:
        b = ups.read()
        if b["bat_i_ma"] > 50 and not a.allow_charging:
            sys.exit(
                f"battery is charging at {b['bat_i_ma']} mA ({b['bat_pct']}%).\n"
                f"Bus_P_mW would include the charge current and the run would be\n"
                f"unusable. Leave it on the charger until the current settles near\n"
                f"zero (the first round floated at -15..+20 mA at 93-94%), then\n"
                f"start. Override with --allow-charging only if you know why.")
        state_txt = ("CHARGING - allowed by --allow-charging, power data invalid"
                     if b["bat_i_ma"] > 50 else "floating")
        print(f"battery {b['bat_pct']}% at {b['bat_i_ma']:+d} mA ({state_txt})")
        bat_start = f"{b['bat_pct']}% at {b['bat_i_ma']:+d} mA"
    else:
        bat_start = "n/a"

    # provenance first: if this fails, nothing else is worth recording
    sys.path.insert(0, str(HERE))
    try:
        from provenance import dump_provenance
        prov = dump_provenance(MODEL_PATH, out_path=str(out / "provenance.json"))
    except Exception as e:
        sys.exit(f"provenance capture failed: {e}")

    runner = Runner(a)
    state = {"phase": "active"}
    sampler = PowerSampler(ups, out / "power_log.csv", state)
    sampler.start()

    data_path = out / "basil_data.csv"
    ev_path = out / "cycle_events.csv"
    with open(data_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "Timestamp", "Latency_ms", "FPS", "CPU_%", "RAM_%", "Temp_C",
            "Throttled", "Predicted_Class", "Confidence",
            "Bus_V_mV", "Bus_I_mA", "Bus_P_mW", "Bat_V_mV", "Bat_I_mA", "Bat_Pct"])
    with open(ev_path, "w", newline="") as f:
        csv.writer(f).writerow(["Timestamp", "Event", "Temp_C", "Note"])

    def log_event(name, note=""):
        with open(ev_path, "a", newline="") as f:
            csv.writer(f).writerow([datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                    name, f"{cpu_temp():.1f}", note])

    import psutil
    t_start = time.time()
    log_event("SYSTEM_START", f"{a.condition} r{a.replicate} "
                              f"cycle {a.active:.0f}/{a.sleep:.0f}")
    log_event("CYCLE_ACTIVE_START")
    cycle_start = time.time()
    active = True
    fps_start, fps_cnt, fps = time.time(), 0, 0.0
    frames = 0
    active_s, active_t0 = 0.0, time.time()

    print(f"running {a.condition} replicate {a.replicate} for "
          f"{a.duration/3600:.2f} h -> {out}")
    try:
        while time.time() - t_start < a.duration:
            now = time.time()
            if active and now - cycle_start > a.active:
                active_s += now - active_t0
                log_event("CYCLE_SLEEP_START")
                active, cycle_start, state["phase"] = False, now, "sleep"
                continue
            if not active:
                if now - cycle_start > a.sleep:
                    log_event("CYCLE_ACTIVE_START")
                    active, cycle_start, state["phase"] = True, now, "active"
                    active_t0 = now
                    fps_start, fps_cnt = time.time(), 0
                else:
                    time.sleep(0.2)
                continue

            if frames % 200 == 0:
                gc.collect()
            ok, frame = runner.cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            cls, conf, lat = runner.infer(frame)
            frames += 1
            fps_cnt += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_cnt / (time.time() - fps_start)
                fps_start, fps_cnt = time.time(), 0

            r = sampler.latest
            with open(data_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    f"{lat:.1f}", f"{fps:.1f}",
                    f"{psutil.cpu_percent(interval=None):.1f}",
                    f"{psutil.virtual_memory().percent:.1f}",
                    f"{cpu_temp():.1f}", sampler.throttled, cls, f"{conf:.4f}",
                    r["bus_v_mv"], r["bus_i_ma"], r["bus_p_mw"],
                    r["bat_v_mv"], r["bat_i_ma"], r["bat_pct"]])
    except KeyboardInterrupt:
        log_event("INTERRUPTED", "operator stopped the run")
        print("\ninterrupted — partial run, do not use it as a replicate")
    finally:
        if active:
            active_s += time.time() - active_t0
        log_event("SYSTEM_STOP", f"{frames} frames")
        sampler.stop_flag.set()
        sampler.join(timeout=3)
        runner.cap.release()

    elapsed = time.time() - t_start
    try:
        e = ups.read() if ups else None
        bat_end = f"{e['bat_pct']}% at {e['bat_i_ma']:+d} mA" if e else "n/a"
    except Exception:
        bat_end = "unreadable"
    (out / "RUN_INFO.txt").write_text(
        f"Condition        : {a.condition}\n"
        f"Replicate        : {a.replicate}\n"
        f"Start            : {datetime.fromtimestamp(t_start)} ({time.strftime('%z')})\n"
        f"Duration         : {elapsed:.1f} s planned {a.duration:.0f} s\n"
        f"Duty cycle       : {a.active:.0f} s active / {a.sleep:.0f} s sleep\n"
        f"Frames           : {frames}\n"
        f"Camera           : USB V4L2 index {a.camera_index}, 640x480, requested {a.fps:.0f} fps, driver reports {runner.fps_reported:.1f}\n"
        f"Measured rate    : {frames / max(1.0, active_s):.2f} frame/s over {active_s:.0f} s active\n"
        f"Supply telemetry : {'UPS HAT (E) @ 0x2D, I2C bus %d, 1 Hz' % a.i2c_bus if ups else 'NONE'}\n"
        f"Throttle source  : {throttle_source()}  (1 Hz, identical in all conditions)\n"
        f"Fan hwmon        : {'present - fan RPM/PWM logged in power_log.csv' if HW_FAN else 'absent'}\n"
        f"Power log        : power_log.csv covers active AND sleep phases\n"
        f"Battery at start : {bat_start}\n"
        f"Battery at end   : {bat_end}\n"
        f"Provenance       : provenance.json\n"
        f"Complete         : {'yes' if elapsed >= a.duration - 5 else 'NO - DISCARD'}\n"
    )
    print(f"done: {frames} frames in {elapsed:.0f} s -> {out}")


if __name__ == "__main__":
    main()
