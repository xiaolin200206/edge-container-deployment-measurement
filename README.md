# Unified AgTech Edge AI Engine

**Real-time agricultural disease monitoring on resource-constrained edge devices.**

Edge AI platform for commercial greenhouse deployment with zero cloud dependency, developed during
an internship at Urban Farm Tech (Jan–Apr 2026) and running on a Raspberry Pi 5.

The repository hosts two related systems built on a shared deployment stack:

| | Task | Model | Status |
| --- | --- | --- | --- |
| **A. Disease classification** | 3-class whole-frame classification (Background / Healthy / Disease) with temporal debouncing and Telegram alerting | MobileNetV2 → ONNX | Materials for the manuscript described below |
| **B. Pest detection** | 5-class object detection with bounding boxes (Fungal, Leaf Damage, Mealybugs, Miner, Mite) | YOLOv8s → ONNX | Separate line of work |

Both share the same duty-cycle scheduler, ONNX Runtime inference path, CSV telemetry logging and
Telegram alerting layer.

> **Revision notice.** This release corrects several claims made in the previous version of the
> manuscript materials, most importantly the attribution of the bare-metal versus container
> comparison. No logged value has been altered; what changed is what is claimed from the logs.
> **See [`CHANGELOG.md`](CHANGELOG.md) for each correction and the evidence that prompted it.**

---

# 1. Disease classification system (manuscript materials)

Code, raw telemetry and analysis for *"Deployment measurements of a containerised edge vision node
on a Raspberry Pi 5: thermal, power and latency characteristics, and the attribution of container
overhead."*

## Contents

```
├── classification.py                 # edge inference + logging + alerting loop
├── provenance.py                     # per-run environment manifest  (NEW)
├── Dockerfile                        # the image used in the measured runs (historical)
├── Dockerfile.matched                # condition B: versions pinned to the host  (NEW)
├── Dockerfile.legacy                 # condition C: the measured image, pinned   (NEW)
├── Mobilenet.ipynb                   # training / ONNX export notebook
├── DEPLOYMENT_GUIDE.md               # provisioning SOP for a fresh Raspberry Pi
├── new_sd_card_setup.md              # SD-card level setup notes
├── CHANGELOG.md                      # corrections in this release  (NEW)
├── analysis/                         # authoritative analysis for the revision  (NEW)
│   ├── recompute.py                  #   every reported quantity, from the logs only
│   ├── fig_system.py                 #   Fig. 1  architecture + measurement chain
│   ├── fig_profiling.py              #   Figs. 2–4
│   ├── fig_analysis.py               #   Figs. 5–8
│   ├── make_dataset_figure.py        #   Fig. 9  from the image archives
│   ├── figstyle.py                   #   shared figure style
│   └── numbers.json                  #   the authoritative value of every quantity
├── figures/                          # regenerated manuscript figures  (NEW)
├── manuscript/                       # revised manuscript + supplement  (NEW)
├── basil_experiments/                # training, ablation and offline analysis scripts
├── scripts/
│   └── analyse_profiling.py          # earlier profiling analysis (superseded)
└── data/
    ├── profiling_runs/               # the four controlled runs
    │   ├── bare_metal_A/             #   night, native execution
    │   ├── bare_metal_B/             #   day,   native execution
    │   ├── docker_A/                 #   night, containerised
    │   └── docker_B/                 #   day,   containerised
    ├── field_log/                    # the live greenhouse session
    └── supplementary_session/        # extended duty-cycle session
```

## Reproducing the manuscript

Every table and figure is regenerated from the deposited logs:

```bash
python analysis/recompute.py            # writes analysis/numbers.json
python analysis/fig_system.py           # Fig. 1        -> figures/
python analysis/fig_profiling.py        # Figs. 2, 3, 4 -> figures/
python analysis/fig_analysis.py         # Figs. 5-8     -> figures/
python analysis/make_dataset_figure.py  # Fig. 9  (needs the image archives)
```

`recompute.py` reads only `data/` and `basil_experiments/02_baseline_comparison/results/`, and
emits every quantity stated in the manuscript. No reported figure exists outside that chain.

Requirements: `pandas`, `numpy`, `matplotlib`.

## Data

### Hardware profiling runs

Four independent three-hour runs under a 60 s active / 15 s sleep duty cycle — two native and two
containerised — interleaved across day and night so that ambient temperature is balanced between
conditions rather than confounded with them. All four runs used identical hardware, the same USB
(V4L2) camera, the same ONNX model file and the same inference script.

**The two conditions also differed in software environment, and this bounds what the comparison
can attribute.** The native runs executed under Python 3.13 with onnxruntime 1.29.0 on Debian 13;
the containerised runs under `python:3.9-slim-bullseye` — Python 3.9, Debian 11 — with
onnxruntime resolved from an unpinned requirement at image-build time — determined from package
metadata to have been **1.19.2**, against the host's **1.29.0**, eighteen releases and some 23
months apart. Execution mode, interpreter version, inference-runtime version and base image
therefore co-vary. The manuscript reports the
measured differences as properties of the two configurations as built, states the attribution
bound explicitly, and specifies the three-condition design that would decompose it. See
`CHANGELOG.md` §1 and Supplementary S8.

Each run directory contains `basil_data.csv.gz` (per-frame telemetry, gzipped to stay within
GitHub file-size limits), `cycle_events.csv` (duty-cycle transitions with the temperature at each
boundary) and `RUN_INFO.txt` (conditions, start time, hardware, OS and runtime versions,
throttling status). `pandas.read_csv` opens the gzipped files directly.

Telemetry rows exist **only during active periods**; the 15 s sleep intervals appear as gaps. No
idle power baseline is therefore available from these runs, and per-inference energy figures carry
the supply module's constant offset. Future runs should log supply telemetry through the sleep
phase as well.

Supply telemetry is read over I2C from the UPS HAT (E). Its `Bus_*` registers report the
**Type-C connector** — the total DC power the Pi and the HAT together draw from the mains PD
adapter — not the 5 V rail feeding the board, which the module does not instrument. The 15.29 V
bus reading is a negotiated USB-PD 15 V contract. The reported figures are therefore **node input
power**, inclusive of two cascaded conversion stages and HAT housekeeping, and absolute
differences between conditions are the meaningful quantity. See `CHANGELOG.md` §3 and
Supplementary S3.

### Field log

The live greenhouse session: 5,989 frames over 181 s of continuous in-situ operation. Used for the
alerting-filter sensitivity analysis and the confidence-structure analysis.

### Supplementary session

3 h 33 min, 313,011 frames, under a 180 s / 45 s duty cycle. Execution mode was not recorded for
this session — it predates the `provenance.py` manifest — and its frame-level confidence threshold
was more permissive than the deployed τ = 0.70. Reported as a separate operating point, not pooled
with the profiling runs.

## Environment capture

`provenance.py` writes a JSON manifest per run recording execution mode, interpreter version,
onnxruntime version, glibc, NumPy and OpenCV versions, the ONNX Runtime provider list and resolved
intra/inter-op thread counts, `os.cpu_count()` and scheduler affinity, the cgroup CPU quota,
`OMP_NUM_THREADS`, the CPU frequency governor, the SHA-256 digest of the model file, and host
throttling status.

```python
from provenance import dump_provenance
prov = dump_provenance("basil_mobilenet.onnx", session=ort_session)
```

Call it once at the start of every run and deposit the output alongside the telemetry. Its absence
is what made the attribution in `CHANGELOG.md` §1 impossible to resolve after the fact.

## Container definitions

```bash
docker build -f Dockerfile.matched -t infer:matched .   # condition B
docker build -f Dockerfile.legacy  -t infer:legacy  .   # condition C
```

`Dockerfile.matched` pins every dependency to the host's versions so that a container-versus-host
comparison isolates containerisation. `Dockerfile.legacy` reproduces the measured image with the
versions stated explicitly. The original unpinned `Dockerfile` is retained unchanged as the
historical artefact of the runs reported in the manuscript — **do not use it for new measurements.**

## Citation

If you use this data or code, please cite the archived data deposit (https://doi.org/10.5281/zenodo.22857312)
together with this repository (https://github.com/xiaolin200206/edge-container-deployment-measurement). The deposit holds the image dataset and the complete
deployment telemetry; this repository holds the analysis code, the container definitions and the
edge application. Together they reproduce every figure and table in the manuscript.
