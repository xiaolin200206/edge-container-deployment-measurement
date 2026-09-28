# Container overhead or stale dependencies? — edge inference on a Raspberry Pi 5

Data, run kit and analysis for

> Lin Ding Shan, "Container Overhead or Stale Dependencies? Decomposing the
> Latency and Energy Cost of Containerised Inference on an Edge Node"
> (submitted to *IEEE Transactions on Sustainable Computing*).

Comparisons of containerised and native execution on edge devices usually
change two things at once: the execution mode and the software stack inside
the image. This study separates them with three conditions on the same
Raspberry Pi 5 running a duty-cycled MobileNetV2 classifier with ONNX Runtime:

| Condition | Execution | Stack |
|---|---|---|
| A `native`  | host process | Python 3.13.5 (Debian), glibc 2.41, ONNX Runtime 1.29.0 |
| B `matched` | Docker, `debian:trixie-slim` | same versions as the host; compiled extension modules byte-identical |
| C `legacy`  | Docker, `python:3.9-slim-bullseye` | Python 3.9.23, glibc 2.31, ONNX Runtime 1.19.2 |

- **B − A** isolates the execution mode (container vs. native, same stack).
- **C − B** isolates the software stack (legacy vs. current, both in Docker).

Nine three-hour runs, randomised complete block design (3 blocks × 3
conditions), 60 s active / 15 s sleep duty cycle, camera-bound at 15 frame/s,
with node input power sampled at 1 Hz through both phases.

**Main results.** Containerisation with the stack held constant: +1.30 ms
(+6.7 %) mean inference latency, +1.9 % CPU time per inference, no resolvable
power or temperature change. Legacy stack with the execution mode held
constant: +41.5 % mean latency, −6.7 % active power, −16.3 % energy per
inference above the idle floor. The idle floor is 67–71 % of the energy per
inference over the duty cycle.

## Contents

```
├── runs/<label>/            # nine runs: native_1..3, matched_1..3, legacy_1..3
│   ├── basil_data.csv.gz    #   per-frame log (latency, CPU, temperature, supply)
│   ├── power_log.csv        #   1 Hz log, active AND sleep phases (power, battery,
│   │                        #   CPU MHz / cap, under-voltage, fan rpm / PWM)
│   ├── cycle_events.csv     #   duty-cycle transitions
│   ├── RUN_INFO.txt         #   run summary (see note on "Replicate" below)
│   └── provenance.json      #   kernel, governor, cgroup, versions, model hash
├── runs/<label>.camera.txt  # camera controls saved before each pinned run
├── fp_host.json, fp_matched.json, fp_legacy.json   # environment fingerprints
├── run_plan.json, orchestrator.log                  # schedule and run log
├── runner/                  # harness, orchestrator, Dockerfiles, fingerprinting,
│                            # PROTOCOL.md (procedure as executed)
├── analysis/
│   ├── analyse_v2.py        # every quantity -> out/numbers_v2.json
│   ├── make_tex.py          # LaTeX macros + Table 2 -> out/
│   ├── make_supp.py         # supplemental tables -> out/
│   └── figures_v2.py        # Figs. 1-5 -> figures/
├── out/                     # generated numbers and tables
├── figures/                 # generated figures (PDF + PNG)
├── manuscript/              # paper and supplement sources, build.sh, PDFs
├── basil_mobilenet.onnx     # the model used in every run (SHA-256 in provenance.json)
└── first_campaign/          # the earlier two-condition campaign (see below)
```

## Reproducing the paper

Requirements: Python 3 with `pandas`, `numpy`, `matplotlib`; for the PDFs, a
TeX Live installation with `IEEEtran`.

```bash
sh manuscript/build.sh
```

This runs the four analysis scripts in order and compiles
`manuscript/main.pdf` and `manuscript/supplement.pdf`. The paper takes every
measured result from `out/numbers_v2.tex` and `out/table_results.tex`; the
first-campaign values are read from `first_campaign/analysis/numbers.json`.
The analysis reads the gzipped per-frame logs directly.

To run the analysis alone:

```bash
python3 analysis/analyse_v2.py && python3 analysis/make_tex.py \
  && python3 analysis/make_supp.py && python3 analysis/figures_v2.py
```

## Notes on the data

- **Warm-up.** The first 600 s of each run, timed from the first cycle event,
  are excluded; 136 complete cycles per run are analysed.
- **Settled floor.** The `ondemand` governor holds 2400 MHz for about 5 s after
  each active phase. The energy split uses the mean power of sleep-phase samples
  taken at least 6 s into the phase.
- **Camera.** `legacy_1` and `matched_1` ran before the camera's dynamic frame
  rate was disabled; both held 15.0 frame/s. The original `native_1` was
  discarded when its frame rate changed; the replacement ran on the evening of
  27 September, after block 2.
- **Harness and images.** The harness was updated before `legacy_2` (frame-rate
  request and a measured-rate line in `RUN_INFO.txt`) and both images were
  rebuilt. The files in `runner/` are the updated versions used from `legacy_2`
  onward; the earlier version was not retained.
- **Replicate field.** The `Replicate` line in `RUN_INFO.txt` holds the run's
  position in the nine-run schedule (1-9), not the replicate within the
  condition; the directory label identifies the replicate.
- **Power** is node input power at the UPS HAT (E)'s Type-C input, not the SoC
  rail.
- **Legacy image.** Rebuilt from the first campaign's definition with pinned
  versions (the original image no longer existed); `libglib2.0-0` was omitted
  because the Debian 11 archive no longer serves it.
- **Line endings.** Files under `runs/` are stored byte-exact
  (`.gitattributes`: `runs/** -text`), so they match the Zenodo deposit.

## Running the kit

See `runner/PROTOCOL.md`. The Dockerfiles copy `basil_mobilenet.onnx`; copy it
from the repository root into `runner/` before building.

## First campaign

`first_campaign/` holds the earlier two-condition campaign (native vs. the
legacy container, two runs each), the greenhouse field log, the classifier
training experiments and the edge application. Its contrast is reproduced by
C − A here and decomposed by this study. Commands in `first_campaign/README.md`
run from inside that directory. The original layout of those materials, as
first released, is preserved at tag `v1.0-caee`.

## Citation

Please cite the data deposits together with this repository (release
`v2.0-tsusc`):

- three-condition campaign (version 2.0.0): https://doi.org/10.5281/zenodo.23007169
- greenhouse imagery and first campaign (version 1.0.0): https://doi.org/10.5281/zenodo.22857312

## Licence

Code: Apache License 2.0 (`LICENSE`). Data: CC BY 4.0 (see the Zenodo records).
