# Greenhouse imagery and deployment telemetry for a containerised edge vision node

Research data supporting the study *"Deployment measurements of a containerised edge vision node on
a Raspberry Pi 5: thermal, power and latency characteristics, and the attribution of container
overhead"*, submitted to *Computers and Electrical Engineering*.

Analysis code, container definitions and the edge application that consume these files are at
<https://github.com/xiaolin200206/edge-container-deployment-measurement>.

---

## Contents

| File | Size | Contents |
|---|---|---|
| `Background.zip` | 1.2 GB | 640 images — distant outdoor scenes from beyond the greenhouse |
| `Basil_healthy.zip` | 308 MB | 498 images — healthy basil foliage, site-collected |
| `Real_disease.zip` | 1.9 GB | 560 images — foliar disease symptoms observed on site |
| `telemetry.zip` | 6.9 MB | deployment logs; see **Telemetry** below |

The cross-domain proxy imagery used in the study's secondary training condition is **not** part of
this record. See **The proxy set is not deposited** below.

---

## Image data

Collected over three weeks in an enclosed commercial greenhouse cultivating basil
(*Ocimum basilicum*) hydroponically in Cyberjaya, Selangor, Malaysia, under natural and highly
variable illumination. The three site-collected archives are original camera captures made in situ,
**hand-held** during walkthroughs. Note that the deployed node uses a **fixed** camera at a
different working distance and framing; the classifier is therefore trained on hand-held close-range
imagery and deployed against a fixed viewpoint.

### The three classes are not constructed symmetrically

- **Background** consists of scenes **outside** the greenhouse — buildings, hard standing, open
  ground and surrounding vegetation. It is distinguished from the other two classes principally by
  **scene scale and context**, not by object identity. It is *not* a generic "everything else" class.
- **Healthy** is close-range basil foliage in good condition.
- **Disease** aggregates **six** visually distinct symptom and pest categories collected on site,
  preserved as subdirectories: fungal infection, leaf curl, senescent withering, mealybugs, leaf
  miner and mite damage. It therefore spans a far wider range of appearances than Healthy.

This asymmetry is the basis of a conjecture in the accompanying study about where novel inputs are
routed: if scene-scale cues dominate the learned representation, a novel object presented close to
the lens cannot resemble a distant outdoor view, so the Background region would be unavailable to
it and the decision would fall between the two close-range classes. Section 5.3 of the manuscript
states why that conjecture is not established by the four events observed.

### The two training conditions

The classifier distinguishes three classes: Background, Healthy and Disease. Two conditions are
compared in the accompanying study:

- **Real-Only** — the Disease class is drawn entirely from `Real_disease`.
- **Real+Proxy** — the Disease class is sample-size-matched at 560 images, half from
  `Real_disease` and half from a cross-domain proxy set that is not deposited here (see below).

Matching the sample size isolates the effect of data composition from the confounding effect of
dataset size.

### The proxy set is not deposited

The 280 cross-domain disease images used in the Real+Proxy condition were obtained in March 2026
from a publicly available plant-disease image collection. **The specific source collection was not
recorded at the time, and could not be identified afterwards.** Because its licence is therefore
unknown, the images are not redistributed here: this record cannot grant rights the depositor
cannot establish.

The consequence is stated plainly, in this record and in the accompanying manuscript: **the
Real+Proxy condition is not exactly reproducible from this deposit.** The Real-Only condition is,
in full. What the proxy set contributed to the study can still be characterised, because it is the
property the study actually tested: the images came from a different acquisition regime and differ
from the site imagery in resolution and capture conditions as well as in symptom morphology, and
all images are resized to 224 × 224 during training, which affects the two sources asymmetrically
— the site captures are substantially downsampled, while the proxy images were close to their
native resolution.

A replication of the comparison does not need these particular images. It needs *any* set of
cross-domain disease imagery of matched size, and the study's finding is that at this dataset scale
no effect of such padding is detectable in either direction (Section 4.6 of the manuscript). The
released training scripts read whatever is placed in a `Proxy_disease/` directory, so the
comparison can be re-run against a proxy set of the replicator's choosing.

---

## Telemetry

`telemetry.zip` contains the deployment logs in three groups. Gzipped CSVs open directly with
`pandas.read_csv`; no manual decompression is needed.

### `profiling_runs/`

Four controlled runs under a 60 s active / 15 s sleep duty cycle — two executing natively on the
host, two inside a Docker container — interleaved across day and night so that ambient temperature
is balanced between conditions rather than confounded with them. Each is analysed over a
three-hour window. Three of the logs end there; `docker_A/` continues to 6 h 25 min and is
deposited unaltered, with the analysis using its first three hours so that all four runs
contribute an equal span.

`Throttled` is populated by `vcgencmd`, which the container image does not ship: it reads `No` in
the two native runs and `Unknown` in the two containerised ones, which means uninstrumented, not
clear.

| Directory | Condition | Time of day |
|---|---|---|
| `bare_metal_A/` | native | night |
| `bare_metal_B/` | native | day |
| `docker_A/` | containerised | night |
| `docker_B/` | containerised | day |

Each directory contains:

- `basil_data.csv.gz` — per-frame telemetry, gzipped. Columns: `Timestamp`, `Latency_ms`, `FPS`, `CPU_%`,
  `RAM_%`, `Temp_C`, `Throttled`, `Predicted_Class`, `Confidence`, `Bus_V_mV`, `Bus_P_mW`,
  `Bat_V_mV`, `Bat_I_mA`, `Bat_Pct`.
- `cycle_events.csv` — duty-cycle transitions. Columns: `Timestamp`, `Event`, `Temp_C`, `Note`.
- `RUN_INFO.txt` — run conditions, start time, hardware, operating system and runtime versions,
  throttling status.

### `field_log/`

The live greenhouse session: 5,989 frames spanning 181.8 s of wall clock, of which 151.8 s across
three duty cycles were active inference (approximately 39.5 frame s⁻¹). SoC temperature reaches
77.7 °C in this log, 4.3 °C below the throttling threshold — considerably hotter than any of the
profiling runs.

### `supplementary_session/`

An extended continuous-inference session: 313,011 frames over 3 h 33 min under a 180 s / 45 s duty
cycle. Execution mode was not recorded for this session, and its frame-level confidence threshold
was more permissive than the deployed τ = 0.70.

### Reading the telemetry correctly

Three properties of these logs determine how they should be used.

1. **Rows exist only during active periods.** The 15 s sleep intervals appear as gaps in the
   timestamps. No idle power baseline can be derived from these runs.
2. **`Bus_*` is the Type-C supply input, not the board's 5 V rail.** The supply module reports
   telemetry from two devices behind one microcontroller: the bus registers come from a
   bidirectional USB Power Delivery controller and measure the total DC power drawn from the mains
   adapter by the Raspberry Pi and the module together; the battery registers come from a
   four-series lithium-ion fuel gauge. The module provides **no** register for the 5 V rail. The
   15.29 V bus reading is a negotiated Power Delivery contract. These are node input power figures,
   inclusive of two cascaded conversion stages and module housekeeping, and are not comparable with
   SoC-level power benchmarks.
3. **The unit of analysis is the run, not the frame.** Frames within a run share thermal state,
   illumination and scene content and are strongly autocorrelated. The effective sample size is two
   per condition, not the ~100,000 rows each run contains.

### Important note on the profiling comparison

The two profiling conditions differed in interpreter, inference-runtime and base-image version as
well as in execution mode:

| | Native runs | Containerised runs |
|---|---|---|
| Operating system | Debian 13 (trixie), glibc 2.41 | Debian 11 (bullseye), glibc 2.31 |
| Python | 3.13 | 3.9 |
| ONNX Runtime | 1.29.0 (released 2026-08-17) | 1.19.2 (released 2024-09-04) |

Eighteen releases and roughly 23 months separate the two runtimes. **The measured differences are
properties of the two configurations as built and are not attributable to containerisation alone.**
The accompanying study reports the comparison as bounded and specifies the three-condition design
that would decompose it; the repository's `CHANGELOG.md` records this and every other correction
made relative to earlier versions of these materials.

---

## Reproducing the published results

```bash
git clone https://github.com/xiaolin200206/edge-container-deployment-measurement
cd edge-container-deployment-measurement
unzip /path/to/telemetry.zip          # restores data/

python analysis/recompute.py          # every quantity stated in the manuscript
python analysis/fig_system.py         # Fig. 1
python analysis/fig_profiling.py      # Figs. 2-4
python analysis/fig_analysis.py       # Figs. 5-8
```

`recompute.py` reads only the deposited logs and emits every value reported in the manuscript, so
no published quantity exists outside the traceable chain from raw telemetry.

The dataset figure additionally needs the image archives:

```bash
python analysis/make_dataset_figure.py \
    --background Background/ --healthy Basil_healthy/ \
    --disease Real_disease/
```

The script also accepts a `--proxy` directory, which adds a fourth column. It is omitted here
because the proxy imagery is not part of this record.

Requirements: `pandas`, `numpy`, `matplotlib`.

---

## Licence

The site-collected image archives (`Background.zip`, `Basil_healthy.zip`, `Real_disease.zip`) and
`telemetry.zip` are released under **CC BY 4.0**.

No third-party imagery is redistributed in this record. Every image deposited here was captured
by the author at the study site.

## Citation

Please cite this record together with the accompanying manuscript and the code repository.
