"""
Environment provenance capture for the profiling runs.

Call dump_provenance() once at the start of every run, bare-metal and
containerised alike, and write the result next to the telemetry CSV.

This is the piece that was missing from the first round of runs: because the
software stack was never recorded per run, the bare-metal and containerised
conditions silently differed in Python, onnxruntime, glibc and numpy, and the
difference only surfaced later from RUN_INFO.txt. Emitting this file makes that
class of confound impossible to repeat, and it is also the artefact a reviewer
will look for.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _sh(cmd: str) -> str | None:
    """Run a shell command, returning stripped stdout or None if unavailable."""
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _sha256(path: str) -> str | None:
    """Hash the model file so both conditions can be proven identical."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _in_container() -> bool:
    if Path("/.dockerenv").exists():
        return True
    try:
        return "docker" in Path("/proc/1/cgroup").read_text()
    except Exception:
        return False


def _cpu_quota() -> str | None:
    """cgroup v2 CPU limit. Must be 'max' in both conditions, or the container
    was given fewer CPU shares than the host and the comparison is invalid."""
    for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            return Path(p).read_text().strip()
        except Exception:
            continue
    return None


def dump_provenance(model_path: str, session=None, out_path: str = "provenance.json") -> dict:
    """Capture the full software and hardware context of this run."""
    import numpy
    import onnxruntime as ort

    prov: dict = {
        # --- the three variables that were confounded the first time ---
        "execution_mode": "container" if _in_container() else "bare-metal",
        "python_version": sys.version.split()[0],
        "onnxruntime_version": ort.__version__,
        # --- the rest of the software stack ---
        "libc": "-".join(x for x in platform.libc_ver() if x) or None,
        "numpy_version": numpy.__version__,
        "os_release": _sh("cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2"),
        "kernel": platform.release(),
        # --- threading: the most likely mechanism behind the latency gap ---
        "ort_providers": ort.get_available_providers(),
        "os_cpu_count": os.cpu_count(),
        "sched_affinity": len(os.sched_getaffinity(0)),
        "cgroup_cpu_max": _cpu_quota(),
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        # --- the model must be byte-identical across conditions ---
        "model_sha256": _sha256(model_path),
        # --- platform state that must match ---
        "cpu_governor": _sh(
            "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
        ),
        "throttled_before_run": _sh("vcgencmd get_throttled"),
    }

    try:
        import cv2
        prov["opencv_version"] = cv2.__version__
    except Exception:
        prov["opencv_version"] = None

    # The actual thread counts ORT resolved, not the defaults you assumed.
    if session is not None:
        try:
            so = session.get_session_options()
            prov["ort_intra_op_num_threads"] = so.intra_op_num_threads
            prov["ort_inter_op_num_threads"] = so.inter_op_num_threads
        except Exception:
            pass

    Path(out_path).write_text(json.dumps(prov, indent=2, sort_keys=True))
    return prov


if __name__ == "__main__":
    print(json.dumps(dump_provenance("basil_mobilenet.onnx"), indent=2))
