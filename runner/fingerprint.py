#!/usr/bin/env python3
"""Fingerprint the inference stack so condition B can be shown to match A.

Comparing version strings is not enough. The host resolves wheels through
piwheels as well as PyPI, while the container sees PyPI only, so the same
version number can be a different build — a different compiler, a different
BLAS, different vectorisation. That difference would land squarely in B - A and
reintroduce the confound this experiment exists to remove, while every version
check reported agreement.

This prints, for each package the inference path uses:

    version       what pip records
    wheel tag     the platform tag of the wheel that was installed
                  (manylinux_* = PyPI, linux_aarch64 = locally built/piwheels)
    binary        sha256 of the compiled extension actually imported

Run it on the host and inside each image and diff the output:

    ~/venv/bin/python fingerprint.py > fp_host.json
    docker run --rm -v "$PWD:/w" infer:matched python3 /w/fingerprint.py > fp_matched.json
    docker run --rm -v "$PWD:/w" infer:legacy  python3 /w/fingerprint.py > fp_legacy.json
    python3 fingerprint.py --diff fp_host.json fp_matched.json

For condition B the host and matched fingerprints must agree on every field.
For condition C they are expected to differ — that difference is what C - B
measures.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

PKGS = ["onnxruntime", "numpy", "opencv-python-headless", "psutil", "smbus2"]

# The compiled extension that carries the actual numerical work for each
# package. Hashing the import target is stronger evidence than any metadata.
BINARY_OF = {
    "onnxruntime": ("onnxruntime.capi.onnxruntime_pybind11_state", None),
    "numpy": ("numpy.core._multiarray_umath", "numpy._core._multiarray_umath"),
    "opencv-python-headless": ("cv2", None),
    "psutil": ("psutil._psutil_linux", None),
}


def sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def wheel_tag(dist: str) -> str | None:
    try:
        from importlib.metadata import distribution
        d = distribution(dist)
        txt = d.read_text("WHEEL")
        if not txt:
            return None
        tags = [ln.split(":", 1)[1].strip()
                for ln in txt.splitlines() if ln.startswith("Tag:")]
        return ",".join(tags) or None
    except Exception:
        return None


def version_of(dist: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def binary_hash(dist: str) -> dict:
    spec = BINARY_OF.get(dist)
    if not spec:
        return {}
    import importlib
    for modname in spec:
        if not modname:
            continue
        try:
            m = importlib.import_module(modname)
        except Exception:
            continue
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = Path(f)
        if p.suffix in (".so", ".pyd") or ".so" in p.name:
            return {"binary": p.name, "sha256": sha256(p)}
        # cv2 is a package whose extension sits beside __init__.py
        for cand in sorted(p.parent.glob("*.so")):
            return {"binary": cand.name, "sha256": sha256(cand)}
    return {}


def collect() -> dict:
    out = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "glibc": ".".join(platform.libc_ver()[1].split(".")[:2]) or None,
        "packages": {},
    }
    for dist in PKGS:
        rec = {"version": version_of(dist), "wheel_tag": wheel_tag(dist)}
        rec.update(binary_hash(dist))
        out["packages"][dist] = rec
    try:
        import onnxruntime as ort
        out["ort_build_info"] = ort.get_build_info()
        out["ort_providers"] = ort.get_available_providers()
    except Exception:
        pass
    return out


def diff(a_path: str, b_path: str) -> int:
    a = json.loads(Path(a_path).read_text())
    b = json.loads(Path(b_path).read_text())
    problems = []

    def cmp(label, x, y):
        if x != y:
            problems.append(f"  {label}\n      {a_path}: {x}\n      {b_path}: {y}")

    for k in ("python", "machine", "glibc"):
        cmp(k, a.get(k), b.get(k))
    cmp("ort_build_info", a.get("ort_build_info"), b.get("ort_build_info"))
    for dist in PKGS:
        pa, pb = a["packages"].get(dist, {}), b["packages"].get(dist, {})
        for field in ("version", "wheel_tag", "sha256"):
            cmp(f"{dist}.{field}", pa.get(field), pb.get(field))

    if not problems:
        print(f"IDENTICAL: {a_path} and {b_path} agree on every field.")
        print("Condition B is genuinely matched to condition A.")
        return 0
    print(f"{len(problems)} DIFFERENCE(S) between {a_path} and {b_path}:\n")
    print("\n".join(problems))
    print("\nIf this is host vs matched, the two environments are NOT the same "
          "and B - A would measure the difference above as well as\n"
          "containerisation. Fix it before running. If this is host vs legacy, "
          "the differences are expected and are what C - B measures.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--diff":
        sys.exit(diff(sys.argv[2], sys.argv[3]))
    print(json.dumps(collect(), indent=2, sort_keys=True))
