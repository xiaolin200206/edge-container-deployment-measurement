# Supplementary material

**Deployment measurements of a containerised edge vision node on a Raspberry Pi 5: thermal, power and latency characteristics, and the attribution of container overhead**

Every table in this document is generated directly from the deposited raw logs by the released analysis scripts. Section S8 gives the reproduction commands.

---

## S1. Complete per-run profiling results

Four independent three-hour runs under the 60 s / 15 s duty cycle. The first ten minutes of each run are excluded as thermal warm-up, leaving 136 complete duty cycles and approximately 8,166 s of active inference per run. Telemetry rows exist only during active periods; sleep-phase power was not logged (Section 5.4 of the main text).

| Quantity | Native A (night) | Native B (day) | Container A (night) | Container B (day) |
|---|---|---|---|---|
| Frames analysed | 105,629 | 101,692 | 91,073 | 102,467 |
| Active inference time (s) | 8166.6 | 8166.4 | 8166.2 | 8165.4 |
| Duty cycles analysed | 136 | 136 | 136 | 136 |
| Latency mean (ms) | 18.13 | 18.28 | 26.44 | 26.57 |
| Latency SD (ms) | 6.16 | 6.24 | 3.62 | 4.09 |
| Latency median (ms) | 15.20 | 15.00 | 24.90 | 25.20 |
| Latency p95 (ms) | 34.7 | 33.6 | 34.0 | 36.4 |
| Latency p99 (ms) | 37.3 | 38.9 | 41.9 | 44.1 |
| CPU mean (%) | 49.81 | 48.23 | 36.79 | 42.78 |
| CPU SD (%) | 6.80 | 7.86 | 6.14 | 6.66 |
| Memory mean (%) | 7.70 | 7.61 | 7.94 | 7.81 |
| SoC temperature mean (deg C) | 63.64 | 64.73 | 60.07 | 61.20 |
| Cyclic peak temperature mean (deg C) | 64.61 | 65.72 | 60.63 | 62.28 |
| Cyclic peak temperature max (deg C) | 68.3 | 68.3 | 63.4 | 65.0 |
| Instantaneous max temperature (deg C) | 70.0 | 71.0 | 64.5 | 66.1 |
| Node input power mean (W) | 10.169 | 10.225 | 9.457 | 9.999 |
| Node input power SD (W) | 2.233 | 2.249 | 2.343 | 2.377 |
| Effective throughput (frame/s) | 12.93 | 12.45 | 11.15 | 12.55 |
| Energy per inference (J) | 0.7862 | 0.8211 | 0.8480 | 0.7968 |

The unit of statistical analysis in the main text is the run, not the frame. Frames within a run share thermal state, illumination and scene content and are strongly autocorrelated, so the effective sample size per condition is two, not the ~100,000 rows the logs contain.

## S2. Software environment of each condition

The two profiling conditions differed in four respects simultaneously. This table is the basis of the attribution bound stated in Section 4.3 of the main text.

| Component | Native runs (A, B) | Containerised runs (A, B) |
|---|---|---|
| Execution | host process | Docker container, no CPU or memory limit applied |
| Base image / OS | Debian 13 (trixie) | `python:3.9-slim-bullseye` (Debian 11) |
| C library | glibc 2.41 | glibc 2.31 |
| Python | 3.13 | 3.9 |
| ONNX Runtime | 1.29.0, released 2026-08-17 | 1.19.2, released 2024-09-04 (determined; see below) |
| NumPy | 2.x series | 1.x series |
| Kernel | 6.18.34-rpt-rpi-2712 | shared with host |
| Device access | direct | `--device /dev/video0`, `--device /dev/i2c-1` |
| Inference script | identical across all four runs | identical across all four runs |
| Model file | identical across all four runs | identical across all four runs |
| Camera | USB V4L2, 640 x 480 | USB V4L2, 640 x 480 |

**Determination of the container's ONNX Runtime version.** The version resolved at image-build time was not recorded, and the storage media have since been reimaged, so it cannot be read back from the original image. It is nonetheless determined, not estimated, by package metadata:

1. `Dockerfile` installs `onnxruntime` with no version constraint, so `pip` resolves the newest release with a compatible wheel.
2. The base image is `python:3.9-slim-bullseye` for `linux/arm64`, so a compatible wheel must carry the `cp39` interpreter tag and an `aarch64` Linux platform tag.
3. Querying the package index shows that the last release publishing such a wheel is **1.19.2**, uploaded 2024-09-04 (`onnxruntime-1.19.2-cp39-cp39-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`). **No release after 1.19.2 publishes a `cp39` `aarch64` wheel at all.**
4. That wheel's `manylinux_2_27` floor requires glibc ≥ 2.27; Debian 11 supplies 2.31, so the constraint is satisfied and no older release would be preferred.

Exactly one release therefore satisfies the constraints, and the containerised runs executed ONNX Runtime 1.19.2. The host ran 1.29.0, released 2026-08-17 — the day before the first profiling run. **Eighteen intervening releases and approximately 23 months separate them.** `Dockerfile.legacy` pins 1.19.2 on this basis and the reasoning is recorded in its header comment.

Environments for the runs reported here were otherwise reconstructed from the deposited run records.

**These runs predate the per-run manifest routine.** The release includes `provenance.py`, which writes a JSON manifest per run capturing execution mode, interpreter version, inference-runtime version, C library version, NumPy and OpenCV versions, the ONNX Runtime provider list, `os.cpu_count()`, the scheduler affinity size, the cgroup CPU quota, `OMP_NUM_THREADS`, the resolved intra- and inter-op thread counts, the CPU frequency governor, the SHA-256 digest of the model file and the host throttling status. Any replication of this protocol should emit it.

## S3. Measurement chain and supply validation

### S3.1 What the supply module actually instruments

The Waveshare UPS HAT (E) exposes telemetry over I2C from a single microcontroller that aggregates two distinct measurement devices. The mapping matters, because the naming convention in the logged columns is inherited from a different generation of the hardware.

| Logged column | Physical node | Source device |
|---|---|---|
| `Bus_V_mV`, `Bus_P_mW` | **Type-C connector (VBUS)** — total DC power drawn from the mains Power Delivery adapter by the Raspberry Pi and the supply module together | bidirectional USB-PD buck–boost controller |
| `Bat_V_mV`, `Bat_I_mA`, `Bat_Pct` | the four-series 21700 lithium-ion pack | four-series lithium-ion fuel gauge |

**The module provides no register reporting the 5 V rail that feeds the board.** The Raspberry Pi's isolated consumption was therefore not measured, and the reported figures are node input power at the supply connector.

Three observations follow, and they explain readings that would otherwise look anomalous:

- **The 15.29 V bus is a negotiated USB Power Delivery contract**, one of the fixed voltages the controller supports. That is why it is both non-standard for a single-board computer and stable to 0.17 %.
- **The 16.79 V pack reading is 4.198 V per cell across four series cells**, essentially the charge ceiling. The 1.5 V difference between the two readings is the conversion step between adapter and pack, not a measurement discrepancy.
- **A node input figure of 10.20 W is consistent with a Raspberry Pi 5 under sustained CPU inference.** At 80–90 % end-to-end efficiency across the two cascaded conversion stages, it corresponds to roughly 8–9 W at the board's 5 V rail, which is an ordinary figure for this workload with active cooling. The efficiency is assumed rather than measured; the manufacturer publishes no efficiency curve.

**A note on the column naming.** The manufacturer's documentation for this module carries a section heading referring to an INA219 current-sense device, inherited from earlier modules in the same product family in which the monitored bus genuinely *is* the battery. This module contains no such device, and its bus registers are the Type-C input. Anyone reusing logging code or column conventions across that product family should check the register map rather than the heading; the `Bus_*` naming in the deposited logs originates from that lineage.

### S3.2 Validation of the bus-power measurement

The module's charging input was connected throughout every run. These figures establish that the battery acted as a float load rather than a variable one, so that bus power reflects live node consumption rather than charging current.

| Quantity | Native A (night) | Native B (day) | Container A (night) | Container B (day) |
|---|---|---|---|---|
| Bus voltage mean (V) | 15.291 | 15.288 | 15.289 | 15.289 |
| Bus voltage SD (V) | 0.0227 | 0.0223 | 0.0264 | 0.0261 |
| Battery current mean (mA) | -1.45 | -1.47 | -0.79 | -1.51 |
| Battery current min (mA) | -15 | -14 | -13 | -15 |
| Battery current max (mA) | 20 | 19 | 19 | 20 |
| Battery charge start (%) | 93 | 93 | 93 | 93 |
| Battery charge end (%) | 94 | 94 | 94 | 94 |
| Correlation, bus power vs battery current | -0.073 | -0.093 | -0.046 | -0.041 |

Across all four runs the battery state of charge moved only from 93 % to 94 % and no charge cycle occurred. At the measured pack voltage the worst-case instantaneous battery contribution is 0.34 W and the mean contribution 0.025 W. Bus voltage was held at 15.29 V with a standard deviation of 0.17 % and was indistinguishable across conditions. The pack was full and floating, and the converter carried essentially the entire load from the adapter.

### S3.3 Bounds on the instrument

The Power Delivery controller is a charge-management device rather than a precision metrology part; the manufacturer documents neither the resolution nor the accuracy tolerance of its voltage, current and power registers. Absolute values should therefore be treated as uncalibrated. Because both conditions were measured through the identical chain on identical hardware, any systematic error is common to them and does not affect the between-condition difference that this study reports.

Bus current was not logged alongside bus power, so the direction of power flow is inferred from the experimental configuration — adapter attached, battery at float — rather than read directly. The module's port is bidirectional, and a repeat run should log bus current as well and verify that bus power equals the product of bus voltage and bus current. The reported state of charge is a model-based estimate from the fuel gauge, dependent on a configured design capacity, and should not be treated as a calibrated quantity; it reads 93–94 % while the cells sit at their charge ceiling.

An external in-line meter cross-check, and per-rail readings from the board's own power-management IC, would close the energy budget properly and are identified as future work.

## S4. Complete architecture comparison

Real-Only versus sample-size-matched Real + Proxy, eleven architectures, one training run per architecture per condition. The final column expresses each difference in units of validation images, the validation partition containing approximately 340 images so that one image is 0.294 percentage points.

| architecture | params (M) | model size (MB) | Real-Only acc. (%) | Real+Proxy acc. (%) | difference (pp) | difference (validation images) |
|---|---|---|---|---|---|---|
| MobileNetV2 | 2.23 | 8.91 | 98.24 | 97.35 | +0.89 | +3.0 |
| EfficientNet-B1 | 6.52 | 26.07 | 99.41 | 98.82 | +0.59 | +2.0 |
| EfficientNet-B0 | 4.01 | 16.05 | 99.12 | 98.53 | +0.59 | +2.0 |
| MobileNetV3-Large | 4.21 | 16.82 | 97.94 | 97.94 | +0.00 | +0.0 |
| MobileNetV3-Small | 1.52 | 6.08 | 98.24 | 98.24 | +0.00 | +0.0 |
| ShuffleNetV2-0.5x | 0.34 | 1.38 | 97.35 | 97.35 | +0.00 | +0.0 |
| RegNet-Y400MF | 3.90 | 15.62 | 98.82 | 99.12 | -0.30 | -1.0 |
| ResNet18 | 11.18 | 44.71 | 97.94 | 98.53 | -0.59 | -2.0 |
| DenseNet-121 | 6.96 | 27.83 | 97.65 | 98.82 | -1.17 | -4.0 |
| ViT-B/16 | 85.80 | 343.20 | 81.18 | 88.53 | -7.35 | -25.0 |
| SqueezeNet | 0.72 | 2.90 | 64.71 | 94.41 | -29.70 | -101.0 |

Three properties of this protocol bound any conclusion drawn from it:

1. The 80/20 split was drawn without a fixed generator seed, so the two conditions were evaluated on **different validation partitions**. The exact validation indices are not recoverable from the deposited artefacts.
2. Each architecture was trained **once per condition**. No seed repetition was performed and no variance estimate accompanies any difference.
3. Differences of the magnitude observed correspond to whole numbers of validation images — between +3 and −4 for nine of the eleven architectures.

SqueezeNet (−29.70 pp) and ViT-B/16 (−7.35 pp) are runs in which one condition failed to converge to the accuracy the architecture reaches in the other; we read them as optimisation failures rather than as data effects. The row labelled *ViT-Tiny* in the released result files is `torchvision`'s `vit_b_16` (85.8 M parameters) and is relabelled ViT-B/16 throughout.

## S5. Complete alerting-filter sensitivity grid

The filter gates each frame on confidence threshold τ, substituting an `unconfirmed` token below it, then confirms a class only when it holds at least ⌈0.6W⌉ of the W positions in the window. Replayed against the field log (5,989 frames, 181 s).

| tau | W | min votes | raw transitions | confirmed transitions | flip suppression (%) |
|---|---|---|---|---|---|
| 0.3 | 3 | 2 | 310 | 151 | 51.3 |
| 0.3 | 5 | 3 | 310 | 110 | 64.5 |
| 0.3 | 7 | 5 | 310 | 74 | 76.1 |
| 0.5 | 3 | 2 | 362 | 130 | 64.1 |
| 0.5 | 5 | 3 | 362 | 96 | 73.5 |
| 0.5 | 7 | 5 | 362 | 72 | 80.1 |
| 0.7 | 3 | 2 | 624 | 62 | 90.1 |
| 0.7 | 5 | 3 | 624 | 58 | 90.7 |
| 0.7 | 7 | 5 | 624 | 46 | 92.6 |

The raw-transition count differs between thresholds because the confidence gate alters the raw class sequence: below the threshold a frame contributes `unconfirmed`, which itself constitutes a transition. Flip suppression is therefore computed against the raw sequence at the same threshold, not against a common baseline.

The deployed configuration is τ = 0.70, W = 5. Across the grid, raising the threshold from 0.3 to 0.7 at fixed W = 5 improves suppression by 26.2 percentage points; extending the window from 3 to 7 at fixed τ = 0.7 improves it by 2.5.

## S6. Out-of-distribution event detail

Supplementary session, 313,011 frames over 3 h 33 min. Frames separated by more than 30 frames (approximately one second) are treated as distinct events.

| event | frame range | frames | elapsed from session start (s) | assigned class | confidence range |
|---|---|---|---|---|---|
| 1 | 87–109 | 19 | 5.4 | Disease | 0.575–0.993 |
| 2 | 469–472 | 2 | 16.6 | Disease | 0.522–0.579 |
| 3 | 1252–1252 | 1 | 40.3 | Disease | 0.528–0.528 |
| 4 | 1320–1327 | 8 | 42.3 | Disease | 0.566–0.930 |

All four events fall within the first 43 s of a 3 h 33 min session. The remaining 311,683 consecutive frames contain no non-Background prediction of any kind. The frame-level confidence threshold active in this session was more permissive than the deployed τ = 0.70, with confirmations appearing down to 0.52.

Background in this dataset consists of distant outdoor scenes from beyond the greenhouse rather than generic non-plant content, so it is distinguished from the two plant classes by scene scale and context; Section 5.3 of the main text argues that this makes it geometrically unavailable to a close-range novel object. The event count, not the frame count, is the effective sample size for any claim about class routing. Under a null hypothesis in which novel inputs distribute at random between the two plant classes, four events all falling in one class occurs with probability 2⁻³ = 0.125 for a specified class, or 0.0625 for one nominated in advance. The main text reports this as an observation rather than as an established structural property, and Section 5.3 specifies the controlled probe that would test it.

## S7. Field-log confidence analysis

Live greenhouse deployment, 5,989 frames over 181 s at approximately 33 frame s⁻¹.

| Quantity | Value |
|---|---|
| Frames | 5,989 |
| Predicted Healthy | 4,189 |
| Predicted Background | 1,769 |
| Predicted Disease | 31 |
| Frames below confidence 0.5 | 92 (1.54 %) |
| Low-confidence frames occurring as isolated single frames | 72.3 % |
| Low-confidence frames occurring in runs of three or more | 9.2 % |
| Longest run of consecutive low-confidence frames | 4 |
| Class-flip rate, low-confidence frames | 50.0 % |
| Class-flip rate, high-confidence frames | 4.5 % |
| Correlation, confidence vs SoC temperature | −0.050 |
| Correlation, confidence vs CPU utilisation | +0.015 |
| Correlation, confidence vs inference latency | −0.023 |

The absence of correlation with system telemetry indicates that prediction instability on this log is not attributable to thermal state or processor load.

## S8. Protocol for the three-condition attribution design

The comparison reported in the main text cannot separate containerisation from the runtime version carried in the image. The following design decomposes it. It is specified here so that the measurement can be completed by the present authors or by others, and the released container definitions implement conditions B and C directly.

**Conditions.** Three, not two:

| Condition | Execution | Python | ONNX Runtime | Base image |
|---|---|---|---|---|
| A | host process | 3.13 | 1.29.0 | host (Debian 13) |
| B | container | 3.13 | 1.29.0 | `python:3.13-slim-trixie` |
| C | container | 3.9 | 1.19.2 | `python:3.9-slim-bullseye` |

Condition B must pin NumPy, OpenCV and every other dependency to the versions `pip freeze` reports on the host, not merely the interpreter and the inference runtime.

**Contrasts.** B − A isolates the cost of containerisation with the software stack held constant. C − B isolates the cost of the image's runtime version with execution mode held constant. C − A is the quantity reported in the present work, now decomposed.

**Replication and analysis.** At least three runs per condition, ideally four so that day and night are balanced two-and-two within each condition. Run order randomised rather than alternated. Per-run means as the unit of analysis, with confidence intervals computed across runs, not across frames. Latency reported as median, p95 and p99 in addition to the mean, since the present data show the mean and the tail behaving differently.

**Controls to verify and record per run.** cgroup CPU quota unrestricted and identical (`cpu.max` = `max`); CPU frequency governor identical; SHA-256 digest of the model file identical; host throttling status before and after; the resolved ONNX Runtime intra- and inter-op thread counts, which differ between runtime versions and are a candidate mechanism for the observed latency and utilisation pattern.

**Additional instrumentation recommended.** Log supply telemetry through the sleep phase as well as the active phase, so that an idle baseline can be subtracted and workload-attributable energy reported without the module's constant offset. Log bus *current* alongside bus power and verify that power equals voltage times current. If board-level rather than node input power is required, three options exist, in increasing order of effort: read the board's own power-management IC per-rail (`vcgencmd pmic_read_adc` on a Raspberry Pi 5 reports per-rail voltage and current and costs nothing to add); place an in-line meter on the 5 V path between the module and the board; or instrument the rail with a dedicated current-sense amplifier. The first of these should be adopted as standard in any repeat run, since it closes the energy budget between node input and board consumption and requires no additional hardware.

**A dedicated threading probe.** Run conditions A and B with the intra-op thread count forced to 1, 2 and 4. If the version-attributable gap collapses under forced thread counts, the mechanism is the runtime's default thread resolution; if it persists, the mechanism lies in the compiled kernels. Either outcome is reportable and both are cheap.

## S9. Reproduction

The deposited archive (https://doi.org/10.5281/zenodo.22854130) contains the raw logs and the image dataset; the analysis code and container definitions are at https://github.com/xiaolin200206/unified-agtech-engine. Every table and figure in the main text and in this supplement is regenerated by:

```
python analysis/recompute.py           # writes numbers.json; all reported quantities
python analysis/fig_system.py          # Fig. 1  (architecture and measurement chain)
python analysis/fig_profiling.py       # Figs. 2, 3, 4
python analysis/fig_analysis.py        # Figs. 5, 6, 7, 8
python analysis/make_dataset_figure.py # Fig. 9  (requires the image archives)
```

The dataset figure is the only one requiring input beyond the telemetry logs. It is generated from
the deposited class archives with deterministic sampling:

```
python analysis/make_dataset_figure.py \
    --background <background/> --healthy <healthy/> \
    --disease <real_disease/> [--proxy <proxy_disease/>]
```

The `--proxy` column is optional and should be omitted where the source licence of the
cross-domain imagery does not permit redistribution; in that case the proxy sources are cited in
the deposit README rather than included.

`recompute.py` reads only the deposited logs and emits every quantity stated in the main text, so that no reported figure exists outside the traceable chain from raw telemetry. Container images are built with:

```
docker build -f Dockerfile.matched -t infer:matched .   # condition B
docker build -f Dockerfile.legacy  -t infer:legacy  .   # condition C
```

`provenance.py` should be invoked once at the start of every run and its output deposited alongside the telemetry.
