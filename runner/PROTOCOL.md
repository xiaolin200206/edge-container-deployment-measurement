# Three-condition run protocol

This is the procedure used for the nine runs in `runs/` at the repository root. It describes what
was done, including the changes made during the campaign (listed at the end).

## Purpose

Three conditions, measured in a randomised complete block design:

| Condition | Execution | Software stack |
|---|---|---|
| A `native`  | host process | host: Python 3.13.5 (Debian), glibc 2.41, ONNX Runtime 1.29.0 |
| B `matched` | Docker, `debian:trixie-slim` | same versions as the host (generated from it) |
| C `legacy`  | Docker, `python:3.9-slim-bullseye` | Python 3.9.23, glibc 2.31, ONNX Runtime 1.19.2 |

`B − A` isolates the execution mode; `C − B` isolates the software stack.

## Part 0 — preparation (once, before any run)

1. Copy `profile_run.py`, `provenance.py`, `fingerprint.py`,
   `make_matched_dockerfile.py`, `Dockerfile.legacy` and the model
   (`basil_mobilenet.onnx`, from the repository root) into one directory on the
   Raspberry Pi.
2. Build the host environment (condition A) in a virtual environment on the
   distribution interpreter and install the packages. Do not change the host
   after this point.
3. Generate condition B's image definition from the running host interpreter:
   `~/venv/bin/python make_matched_dockerfile.py > Dockerfile.matched`.
   The script reads the interpreter and package versions and emits pins; it
   refuses to write a file if a required package is missing.
4. Build both images:
   `docker build -f Dockerfile.matched -t infer:matched .` and
   `docker build -f Dockerfile.legacy -t infer:legacy .`
5. Fingerprint the host and both images with `fingerprint.py` and compare.
   Host and matched must be identical in every field (Python and glibc
   versions, ONNX Runtime build string, SHA-256 of each compiled extension
   module). The fingerprints used for the campaign are `fp_*.json` at the repository root.
6. Check the supply telemetry: `profile_run.py --selftest` must pass (bus
   voltage in range, bus current register `0x12` agreeing with power/voltage).
7. Keep the charger connected. A run may start only when the battery is
   floating (charge current ≤ +50 mA); `profile_run.py` refuses otherwise and
   records the battery state at both ends of the run.
8. Fix the camera and the scene for the whole campaign.
9. Stop all systemd timers for the duration of the campaign; do no other work
   on the node while runs are in progress.
10. Rehearse each condition for ten minutes and check that the per-frame log,
    the 1 Hz power log (with `sleep` rows) and `provenance.json` are written.

## Part 1 — schedule

Nine runs of three hours, three blocks, within-block order randomised with
seed 20260927 (`run_plan.json`):

| run | block | condition | label |
|---|---|---|---|
| 1 | 1 | legacy  | `legacy_1`  |
| 2 | 1 | matched | `matched_1` |
| 3 | 1 | native  | `native_1`  |
| 4 | 2 | legacy  | `legacy_2`  |
| 5 | 2 | native  | `native_2`  |
| 6 | 2 | matched | `matched_2` |
| 7 | 3 | matched | `matched_3` |
| 8 | 3 | legacy  | `legacy_3`  |
| 9 | 3 | native  | `native_3`  |

## Part 2 — running

`orchestrate.py --block N` runs a block unattended inside `tmux`. Before each
run it waits for the battery to float and for the SoC to cool to ≤ 50 °C,
disables the camera's dynamic frame rate and saves the camera controls
(`runs/<label>.camera.txt`), launches the run with the right command for the
condition (host venv, or `docker run` with the camera and I2C devices and the
host time zone mounted), and afterwards checks that `RUN_INFO.txt` reports
completion and that the frame count is within 0.93–1.05 × 129,600. It stops at
the first failure. Progress is written to `orchestrator.log`.

Each run lasts 10,800 s under a 60 s active / 15 s sleep duty cycle, with the
camera requested at 15 frame/s. The first 600 s are treated as warm-up in the
analysis.

## Part 3 — after each run

Check `RUN_INFO.txt` (`Complete : yes`), the frame count, and that
`power_log.csv` has about 10,800 rows including sleep-phase rows. A run that
fails is deleted and repeated; partial runs are never analysed.

## Changes made during the campaign

- **Camera frame rate.** The first two runs (`legacy_1`, `matched_1`) were
  started without an explicit frame-rate request and without disabling the
  camera's dynamic frame rate; both held 15.0 frame/s in a dark room. The
  original `native_1` changed frame rate when the room lights came on and was
  stopped and discarded. From `legacy_2` onward the orchestrator disabled
  dynamic frame rate and the harness requested 15 frame/s.
- **Harness update and image rebuild.** To add the frame-rate request (and a
  measured-rate line in `RUN_INFO.txt`), `profile_run.py` and `orchestrate.py`
  were updated before `legacy_2`, and both images were rebuilt. The harness is
  copied in the images' last layer; the fingerprints were not repeated after
  the rebuild, but each run's `provenance.json` records the interpreter, glibc
  and library versions.
- **Block 1 timing.** The replacement `native_1` ran on the evening of
  27 September (20:19–23:19), after block 2.
- **Legacy image.** The first campaign's image no longer existed; condition C
  was rebuilt from its definition with the versions pinned (see the comments in
  `Dockerfile.legacy`). `libglib2.0-0` is no longer available from the Debian 11
  archive and was omitted; the headless OpenCV build does not need it.
