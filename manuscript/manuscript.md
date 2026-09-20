# Deployment measurements of a containerised edge vision node on a Raspberry Pi 5: thermal, power and latency characteristics, and the attribution of container overhead

**Abstract**

Container runtimes increasingly package device software on single-board computers at the edge, and the resulting overhead is usually assumed rather than measured. We report an instrumented case study of a duty-cycled MobileNetV2 vision node in a commercial greenhouse. Four interleaved three-hour runs, two native and two containerised, were logged per frame for latency, utilisation and SoC temperature, and per second for node input power. The containerised configuration showed 45.6 % higher mean inference latency, 9.24 percentage points lower processor utilisation, 3.71 °C lower cyclic peak temperature and 0.469 W lower input power. Three qualifications carry the contribution. The power advantage does not survive normalisation: 6.6 % fewer inferences were completed, and energy per inference was not detectably better. The latency difference costs 41.5 ms across a five-frame confirmation window, against an end-to-end alert path under 2.5 s. And the conditions differed in interpreter, runtime and base-image version as well as execution mode, package metadata fixing the container at ONNX Runtime 1.19.2 against the host's 1.29.0, so the result is reported as bounded, with the three-condition design that would decompose it. A field session reaching 77.7 °C indicates that the bench regime understates deployed thermal load. Telemetry and code are released.

**Keywords:** edge computing; containerisation; deployment measurement; thermal characterisation; single-board computers; energy per inference; measurement reproducibility; Internet of Things

---

## 1. Introduction

A continuously operating vision inference node (a camera, a single-board computer, a model and an alerting path) is now a standard building block of Internet-of-Things deployments (Ray, 2017; O’Grady et al., 2019) in settings where connectivity is intermittent and the value of an observation decays within hours. Whether such a node works in the field is determined less by the accuracy of its classifier than by three properties of the deployment stack around it: whether it remains thermally stable under sustained inference, what it costs to isolate and maintain its device software on constrained hardware, and how far its behaviour under laboratory evaluation predicts its behaviour in situ.

The second of these has become a practical concern as container runtimes migrate from the cloud end of the cloud-to-thing continuum onto the devices themselves, where the resource envelope that makes containerisation cheap no longer holds. Practitioners adopt containers on single-board computers (Merkel, 2014; Mahmud and Toosi, 2021) for reasons that have nothing to do with performance: reproducible dependency resolution, atomic over-the-air update, rollback. They accept an unquantified performance penalty in exchange. The literature that would quantify that penalty is thin for this class of device, and the figures that do circulate are rarely accompanied by enough environment detail to be portable between studies.

This paper reports what one such node actually does when it is built, deployed and instrumented. The measurement platform is a Raspberry Pi 5 running a MobileNetV2 classifier exported to ONNX Runtime under a 60 s / 15 s duty cycle, deployed in a commercial greenhouse cultivating basil (*Ocimum basilicum*). Greenhouse cultivation of high-value herbaceous crops is a setting in which foliar disease can decimate a yield within days if undetected (Ferentinos, 2018; Mohanty et al., 2016), and in which convolutional classifiers reach high accuracy on curated datasets while transferring poorly to the field (Barbedo, 2018; Kamilaris and Prenafeta-Boldú, 2018; Picon et al., 2019). The application setting is nonetheless incidental to the measurement: it was chosen because it exercises the failure modes of interest, namely a passively cooled enclosure at elevated ambient temperature, intermittent connectivity, and an open visual world that the training set does not span. The results are stated in terms transferable to other single-board vision workloads.

The contribution is empirical and methodological rather than algorithmic. No new architecture or training procedure is proposed. Specifically:

1. **A measured characterisation of the deployment trade.** Four interleaved three-hour profiling runs quantify the differences in latency, processor load, temperature and board-level power between native and containerised execution of an identical inference script, with day/night interleaving so that ambient temperature is balanced between conditions rather than confounded with them (Section 4.1).

2. **Normalisation by completed work, which reverses one of the headline results.** Reporting power at a fixed duty cycle conceals a throughput difference. Once energy per inference is computed, the containerised configuration is marginally *less* efficient, not more (Section 4.2). We argue that per-inference energy, not instantaneous power, is the quantity this class of study should report.

3. **An explicit treatment of attribution.** In the configuration measured here the container image carried a different interpreter and inference-runtime version from the host, through a build practice common enough that we would expect the same confound wherever a container-overhead comparison does not report both sides' versions. Execution mode, runtime version and base image co-vary, and the measured difference is not attributable to any one of them. Section 4.3 states this as a bounded result, quantifies what the comparison does and does not support, and specifies the three-condition design required to decompose it.

4. **Two further deployment characteristics.** The value of padding a small field dataset with cross-domain proxy imagery is shown not to generalise across eleven lightweight architectures (Section 4.6), and the routing of out-of-distribution inputs through a closed-set three-class classifier is characterised at two clearly separated levels of evidential strength (Section 4.7).

All per-frame telemetry, per-cycle event markers, analysis code and environment manifests are released so that the profiling protocol can be reused as a test bed for other edge inference workloads.

## 2. Related work

### 2.1. Edge inference on single-board computers

Localising inference on the sensing device reduces latency, removes a dependence on uplink bandwidth and keeps imagery on-premises (Chen and Ran, 2019; Zamora-Izquierdo et al., 2019; Li et al., 2018). Lightweight architectures designed for this envelope are routinely deployed on accelerator-equipped edge boards: MobileNet and its successors (Howard et al., 2017; Sandler et al., 2018), SqueezeNet (Iandola et al., 2016), EfficientNet (Tan and Le, 2019) and the ShuffleNet family. Procurement cost and power draw nonetheless limit those boards in multi-node installations. CPU-only single-board computers remain the cost-effective substrate for dense deployments (Upton and Halfacree, 2014), and the current Raspberry Pi generation is capable enough that the binding constraint is thermal and software-operational rather than arithmetic.

Surveys of edge-enabled agriculture (Sapna et al., 2026) and of edge AI more generally (Singh and Gill, 2023) converge on a common observation: sustained on-device operation, rather than model accuracy, is the dominant unresolved constraint, and deployment-stage characteristics such as thermal behaviour, memory pressure, energy budget and update mechanics are reported far less consistently than classification metrics. Work at the microcontroller end of the spectrum reports practical deployment trade-offs for TinyML platforms (Gookyi et al., 2024), and studies of resource-limited agricultural settings document the connectivity constraints such deployments face (Nawaz and Babar, 2025). Closest to the thermal component of the present work, Benoit-Cattin et al. (2020) characterise the effect of thermal throttling on long-term visual inference on a CPU-based edge device; we extend that line to the current hardware generation and add the containerisation and power dimensions. Quantifying those characteristics for a current-generation single-board computer, under a workload representative of actual operation, is the gap this study addresses.

### 2.2. Container overhead on constrained devices

Containers are the industry standard for cloud software delivery (Merkel, 2014; Pahl et al., 2019), and the same engineering arguments of environmental isolation, reproducible builds and continuous deployment apply with greater force at the edge, where physical access to a device is expensive. Virtualisation layers were long considered too resource-intensive for constrained IoT hardware, and most edge deployments consequently run natively. Early comparisons of hypervisors against lightweight virtualisation established the broad ordering (Morabito et al., 2015), and work on minimal base images and container deployment on Raspberry Pi hardware narrowed the gap considerably (Bellavista and Zanni, 2017; Mahmud and Toosi, 2021). Comparative analyses of container engines now report startup time, memory footprint, image size and computational overhead across engines including those designed for embedded targets (Baresi et al., 2024), the energy cost of running workloads inside a container against outside one has been measured directly (Santos et al., 2018), and the comparison has recently been extended to microVMs on edge devices (Lee et al., 2026).

What remains scarce is measurement on current single-board hardware under a sustained, thermally relevant workload, with the software environment reported in enough detail to be compared against. Section 4.3 argues that this second condition is not a documentation nicety: without it, published overhead figures are not portable, and the present study is itself an instance of the problem.

### 2.3. Reproducibility of performance measurement

Performance figures obtained from a managed language runtime depend on the version of that runtime. For ARM64 inference in particular, successive releases of ONNX Runtime (ONNX Runtime developers, 2021) have introduced substantial kernel-level optimisation, so a comparison that does not hold the runtime version fixed measures the runtime as much as it measures anything else. Recent work on benchmark methodology for embedded AI makes the same point from the metrological side, arguing that energy benchmarks for edge devices require the measurement protocol itself to be specified before the numbers are comparable (Apicella et al., 2026). The same applies to the C library version supplied by the base image and to the numerical libraries beneath the inference engine. Practices developed in empirical software engineering address this directly: pinned dependency manifests, per-run environment capture, archival deposit of both data and code, and Section 3.9 describes the manifest scheme adopted here in response to the limitation identified in this work.

### 2.4. Temporal filtering for event alerting

Single-frame positive classifications from a continuously operating camera are susceptible to transient glare, moving shadows and partial occlusion. Low-overhead temporal filtering bridges raw per-frame inference and a practical notification protocol (Shafique et al., 2020) so that an operator receives only persistent detections. The filter used here is a confidence gate followed by a majority vote over a sliding window; Section 4.5 reports its sensitivity across the threshold–window grid, replayed against a recorded field log.

## 3. Materials and methods

### 3.1. Node architecture and physical deployment

The edge node was built on a Raspberry Pi 5 (8 GB variant) with passive cooling. The deployment site is an enclosed commercial greenhouse in Cyberjaya, Selangor, Malaysia, cultivating basil hydroponically. The location has a tropical rainforest climate (Köppen Af) with an annual mean temperature of 26.0 °C, monthly mean maxima approaching 30 °C, and a year-round mean relative humidity of 84 %, conditions materially warmer and more humid than the climate-controlled laboratory bench on which the profiling runs of Section 4.1 were performed, which bears on the interpretation of the thermal result (Section 5.1).

**Two distinct capture regimes should be distinguished.** The training imagery of Section 3.8 was captured **hand-held**, close to the foliage, during walkthroughs of the greenhouse. The deployed node instead uses a **fixed** Raspberry Pi Camera Module 3 through the libcamera stack, mounted inside the greenhouse, positioned to observe a section of the basil canopy, and housed in an enclosure that protects the electronics while exposing the lens and permitting convective dissipation. The classifier is thus trained on hand-held close-range imagery and deployed against a fixed viewpoint; this mismatch is one component of the gap between held-out and in-situ behaviour discussed in Section 4.8.

Fig. 1 gives the node architecture, the power chain and the points at which it is instrumented. The controlled profiling reported in Section 4.1 was performed subsequently in a laboratory setting using a USB (V4L2) camera at 640 × 480 and a Waveshare UPS HAT (E) supply module, which provides a regulated supply and reports bus voltage, bus power, battery voltage and battery current over I2C. Because the acquisition path differs between the two settings, absolute latency figures from the profiling runs are not directly comparable with the live deployment; the comparison *between* profiling conditions is internally valid because all four runs used identical acquisition hardware.

### 3.2. Classifier and inference runtime

MobileNetV2 (Sandler et al., 2018) was selected for its inverted-residual, depthwise-separable design and consequently low multiply–accumulate count. The model was implemented in PyTorch (Paszke et al., 2019), initialised from ImageNet weights and fully fine-tuned with the backbone unfrozen, using Adam at an initial learning rate of 1 × 10⁻⁴, batch size 16, up to 20 epochs with early stopping on validation loss (patience 5) and a ReduceLROnPlateau schedule. Training used an aggressive online augmentation pipeline (random horizontal and vertical flips, random resized crop, affine rotation ±15°, colour jitter ±30 % brightness and contrast and ±20 % saturation, Gaussian blur, random erasing) to simulate the dynamic illumination of the greenhouse.

For deployment the weights were exported to ONNX, converting the dynamic computation graph to a static one and enabling the ONNX Runtime engine to apply graph-level optimisation. Export was executed with `dynamo=False` to avoid introducing operators unsupported on ARM64. The exported model is under 15 MB. Only the ONNX Runtime CPU execution provider was evaluated; alternative engines were not benchmarked (Section 5.4).

The classifier distinguishes three classes: Background, Healthy and Disease. Dataset composition and the training-data comparison protocol are given in Section 3.8.

### 3.3. Containerisation and the software environment

The inference process was encapsulated in a Docker container built from `python:3.9-slim-bullseye`, with `opencv-python-headless` specified explicitly to avoid pulling the full OpenCV dependency tree. The camera was made available by passing the video device into the container at launch (`--device /dev/video0`), and for the profiling runs the I2C bus was passed in additionally (`--device /dev/i2c-1`) so that the containerised process could read supply telemetry. No CPU quota, memory limit or CPU-set restriction was applied to the container.

**The two conditions differed in more than execution mode, and this is central to the interpretation of the results.** Table 1 states the software environment of each condition as recorded in the run manifests. The native condition executed under the host operating system's Python 3.13 with ONNX Runtime 1.29.0 on Debian 13 (glibc 2.41); the containerised condition executed under Python 3.9 on Debian 11 (glibc 2.31), with ONNX Runtime installed from an unpinned requirement and therefore resolved at image-build time.

The resolved version was not recorded at the time, and the SD cards have since been reimaged, so it cannot be read back from the original image. It is nonetheless determined by package metadata, not estimated. ONNX Runtime publishes its last `cp39` wheel for `aarch64` Linux at version **1.19.2** (uploaded 2024-09-04); no release after it publishes a `cp39` `aarch64` wheel at all, and that wheel's `manylinux_2_27` floor is satisfied by the base image's glibc 2.31. On `python:3.9-slim-bullseye` for `linux/arm64`, `pip install onnxruntime` could therefore only have resolved to 1.19.2. The host's 1.29.0 was released on 2026-08-17, the day before the first profiling run. The two conditions consequently executed inference runtimes separated by eighteen intervening releases and approximately 23 months of development, over which substantial ARM64 kernel optimisation was introduced. Section 4.3 treats the consequences.

**Table 1.** Software environment of the two profiling conditions, as recorded in the released run manifests. Execution mode, interpreter version, inference-runtime version and base image co-vary between conditions.

| Component | Native condition | Containerised condition |
|---|---|---|
| Execution | host process | Docker container |
| Operating system | Debian 13 (trixie) | `python:3.9-slim-bullseye` (Debian 11) |
| C library | glibc 2.41 | glibc 2.31 |
| Python | 3.13 | 3.9 |
| ONNX Runtime | 1.29.0 (released 2026-08-17) | 1.19.2 (released 2024-09-04); see Section 3.3 |
| Kernel | 6.18.34-rpt-rpi-2712 | shared with host |
| Inference script | identical | identical |
| Model file | identical | identical |
| Camera | USB V4L2, 640 × 480 | USB V4L2, 640 × 480 (passed through) |

### 3.4. Instrumentation and the measurement chain

Per-frame telemetry (inference latency, processor utilisation, memory utilisation, SoC temperature, throttling status, predicted class and confidence) was written to disk for every frame during active periods, together with per-cycle event markers at each active-to-sleep and sleep-to-active transition. Supply telemetry (bus voltage, bus power, battery voltage, battery current, battery state of charge) was sampled once per second from the UPS module over I2C.

Several properties of this chain bound the interpretation of the power figures. They are reported here rather than in the limitations, because they determine which quantity is meaningful.

*The instrumented node is the supply input to the whole assembly* (Fig. 1a). The supply module reports telemetry over I2C from a single microcontroller that aggregates two distinct measurement devices. The registers logged here as bus voltage and bus power originate from the module's bidirectional USB Power Delivery buck–boost controller and correspond to the **Type-C connector**: that is, to the total DC power drawn from the mains adapter by the Raspberry Pi and the supply module together. The module exposes no bus-current register, so current on that side of the converter is not logged. The registers logged as battery voltage, current and state of charge originate from a separate four-series lithium-ion fuel gauge and describe the battery pack. **The module provides no instrumentation of the 5 V rail that feeds the board**, so the Pi's isolated consumption was not measured.

Throughout all four runs the adapter had negotiated a 15 V Power Delivery contract, which is why the bus voltage is both non-standard for a single-board computer and extremely stable (15.29 V, SD 0.026 V, 0.17 %); the pack sat at 16.79 V, or 4.198 V per cell, essentially at its charge ceiling. The 1.5 V difference between the two readings is the conversion step between them, not a measurement discrepancy.

The reported figures are therefore **node input power**: the Raspberry Pi's own consumption together with the losses of two cascaded conversion stages and the supply module's housekeeping current. Assuming 80–90 % end-to-end conversion efficiency, the mean native figure of 10.20 W corresponds to roughly 8–9 W at the board's 5 V rail; that efficiency is assumed, not measured, the manufacturer publishing no efficiency curve. Two consequences follow for interpretation. A workload-independent offset from conversion loss and housekeeping is common to both conditions, so the **absolute difference** between conditions is the meaningful quantity and is reported as primary throughout; the corresponding percentage is computed against a denominator inflated by that offset and understates the workload-attributable difference. And the figures are not comparable with published SoC-level power benchmarks, which measure a different node.

We report node input power rather than board power without apology: it is the quantity a deployment actually pays for at the socket, and it is the quantity an installer sizing a supply or a solar budget needs. It is simply not the quantity a processor benchmark reports, and we distinguish them explicitly.

*The charging input remained connected, and the battery did not contribute.* Because the module's charging input was connected for the duration of every run, battery charging current could in principle have contaminated the bus-power measurement. Per-run telemetry excludes this. Across all four runs the battery state of charge moved only from 93 % to 94 % and no charge cycle occurred; battery current had a mean of −1.5 mA across a range of −15 to +20 mA, corresponding to a mean contribution of 0.025 W and a worst-case instantaneous 0.34 W at the measured pack voltage. Battery current showed no meaningful correlation with bus power (r between −0.04 and −0.09 across runs). The pack was full and floating, and the converter carried essentially the entire load from the adapter; the charging input acted as a constant-voltage supply rather than as a variable load, and its contribution is common to both conditions.

*Accuracy of the telemetry is unspecified.* The Power Delivery controller is a charge-management device, not a precision metrology part, and the manufacturer documents neither the resolution nor the accuracy tolerance of its voltage, current and power registers. Values are reported as read from the module's onboard telemetry. Because both conditions were measured through the identical chain on identical hardware, any systematic error is common to them and does not affect the between-condition difference, which is the quantity this study reports; it does bound the absolute figures, and an external meter cross-check is identified as future work.

### 3.5. Profiling protocol

Continuous inference on a passively cooled single-board computer drives the SoC toward its thermal limit, so a duty cycle was adopted as the standard operating condition for all controlled comparisons: 60 s of active inference followed by 15 s of resource-release sleep.

Four independent runs were conducted, two native and two containerised, each analysed over a three-hour window comprising 144 complete duty cycles. Three of the four logs terminate at that point. The `docker_A` log continues for 6 h 25 min; its first three hours are analysed and the remainder discarded, so that all four runs contribute an equal span and an equal number of cycles. The full log is deposited unaltered. The runs were interleaved across day and night, with each condition contributing one night-time and one daytime run, so that any ambient temperature effect is balanced between conditions rather than confounded with them. All four runs used identical hardware, the same USB camera, the same ONNX model file and the same inference script. The first ten minutes of each run were excluded from aggregate statistics as a thermal warm-up period, leaving 136 complete cycles and approximately 8,166 s of active inference per run. Throttling status was also queried on the host shell before and after every run, including the containerised runs, and reported no undervoltage or frequency-capping event. The per-frame throttling field is a separate matter: it is populated by `vcgencmd`, which the container image does not ship, so that field is instrumented only in the native runs (Section 4.1).

**The unit of statistical analysis is the run, not the frame.** Frames within a run share thermal state, illumination and scene content and are strongly autocorrelated; treating the ~100,000 frames of a run as independent observations would be pseudoreplication in the sense of Hurlbert (1984). Per-run means are therefore computed first, and condition-level quantities are reported as the mean of two run means with the half-range as a dispersion measure. With n = 2 per condition this supports a statement about the direction and approximate magnitude of each effect and about replicate agreement, but it does not support significance testing, and none is claimed.

### 3.6. Supplementary continuous-inference session

A separate indoor session was recorded to characterise behaviour under longer uninterrupted inference periods and to observe naturalistic out-of-distribution inputs at scale. This session ran the deployed pipeline for 3 h 33 min under a 180 s / 45 s duty cycle, preserving the 4:1 active-to-sleep ratio but with longer active periods, and logging 313,011 inference frames. Because the duty cycle differs from the controlled protocol, its telemetry is reported as a separate operating point (Section 4.4) and is not pooled with the profiling runs. Two properties of this session are bounded by its log, not asserted: its execution mode was not recorded, the log predating the manifest scheme of Section 3.9; and its frame-level confidence threshold was lower than the deployed τ = 0.70, with confirmations appearing at confidences down to 0.52. During the session the camera was not directed at foliage, so the predicted-class distribution reflects behaviour under a largely empty scene rather than classification accuracy on plant material.

### 3.7. Alerting filter

The deployed pipeline (Fig. 1b) executes a continuous loop: frame acquisition; downsampling to 224 × 224 and ImageNet normalisation; a forward pass through the ONNX Runtime session yielding class probabilities; temporal filtering; and, on confirmation, an asynchronous HTTPS POST to a messaging API that delivers a payload of timestamp, class, confidence, inference latency and SoC temperature to a designated mobile device.

The temporal filter is a confidence gate followed by a majority vote. A frame contributes its predicted class only if its confidence meets a threshold τ; otherwise it contributes an `unconfirmed` token. A class is confirmed at frame *t* only if it holds at least ⌈0.6 W⌉ of the W positions in the window ending at *t*. The deployed configuration is τ = 0.70, W = 5, i.e. three of five frames. Section 4.5 reports the sensitivity of this filter across τ ∈ {0.3, 0.5, 0.7} and W ∈ {3, 5, 7}, replayed against the recorded field log.

Because a transmission is generated only on confirmation and not on a schedule, network activity is sparse and its duty cycle is independent of frame rate. The single-node deployment uses a standard HTTPS webhook; contention among many nodes on shared wireless infrastructure is outside the scope of a single-node study (Section 5.4).

### 3.8. Dataset and the training-data comparison

Imagery was collected over three weeks in the commercial greenhouse under highly variable natural lighting; Fig. 2 shows representative examples of each class. The primary dataset comprises three classes: Background (640 images), Healthy (498) and Disease (560). The three classes are not constructed symmetrically, and the asymmetry matters for Section 5.3. **Background** consists of scenes **outside** the greenhouse: buildings, hard standing, open ground and vegetation on the surrounding site, which is to say distant outdoor views and not close-range plant material. **Healthy** is close-range basil foliage in good condition. **Disease** aggregates six visually distinct symptom and pest categories collected on site: fungal infection, leaf curl, senescent withering, mealybugs, leaf miner and mite damage. Background is distinguished from the other two classes principally by **scene scale and context** rather than by object identity, while Healthy and Disease are both close-range foliage separated by appearance, and Disease spans a far wider range of appearances than Healthy does. Padding a small field dataset with imagery from a visually similar public source is common practice (Chen et al., 2020; Too et al., 2019), and carries a known risk: models trained on curated laboratory imagery learn background correlates instead of pathology and fail on field complexity (Sladojevic et al., 2016; Picon et al., 2019; Geirhos et al., 2020). A secondary condition, *Real + Proxy*, replaces half of the Disease class with cross-domain imagery obtained in March 2026 from a publicly available plant-disease image collection, which exhibits visible morphological and spectral domain shift relative to the site imagery. The identity of that source collection was not recorded at the time and could not be established afterwards; the consequences are set out at the end of this section and in Section 5.4.

To separate the effect of data composition from the confounding effect of dataset size, the Real + Proxy condition was resampled to match the Real-Only Disease count exactly, at 560 images, 280 site-collected and 280 proxy-sourced, rather than using the unbalanced union. The comparison was extended across eleven lightweight CNN and transformer architectures. Three properties of this protocol constrain the strength of any conclusion and are stated here rather than in the limitations:

- The 80/20 train/validation split was drawn **without a fixed generator seed**, so the two conditions were evaluated on different validation partitions. The exact validation indices are not recoverable.
- Each architecture was trained **once per condition**; no seed repetition was performed, so no variance estimate accompanies any reported difference.
- The validation partition contains approximately 340 images, so one image corresponds to 0.29 percentage points of accuracy.
- The proxy imagery is **not redistributed** with this work. Because its source collection and therefore its licence could not be established, it cannot be deposited under the licence applied to the site-collected archives. The Real-Only condition is exactly reproducible from the deposit; the Real + Proxy condition is not.

Section 4.6 interprets the results within these constraints. We note that the comparison as a question does not depend on these particular images: it requires cross-domain disease imagery of matched sample size, and the released training scripts read whatever is placed in the proxy directory, so a replicator can re-run it against a proxy set of their own choosing. What cannot be reproduced is this exact run. The row labelled *ViT-Tiny* in the released result files is `torchvision`'s `vit_b_16` (85.8 M parameters); it is reported here as ViT-B/16.

The weights of the model deployed in the live session were not preserved; the released MobileNetV2 checkpoints are from the architecture comparison and are separately trained. Validation accuracies obtained from the originally deployed model are reported as historical context and are not reproducible from the release.

### 3.9. Released artefacts and environment capture

All per-frame telemetry, per-cycle event markers, analysis code, training scripts and model artefacts are deposited in a public archive with a persistent identifier. Prompted by the attribution limitation identified in Section 4.3, the release additionally provides: version-pinned container definitions for both the environment-matched and the as-measured image; and an environment-capture routine that writes a per-run manifest recording execution mode, interpreter version, inference-runtime version, C library version, numerical library versions, the resolved ONNX Runtime thread counts, the cgroup CPU quota, the CPU frequency governor, the SHA-256 digest of the model file and the host throttling status. Runs reported here predate that routine; their environments were reconstructed from the deposited run records and are given in Table 1.

## 4. Results

### 4.1. Native versus containerised execution

Per-run results are given in Table 2 and condition-level summaries in Table 3. Representative longitudinal traces for the night-time pair appear in Fig. 3, and the per-run distribution underlying each condition mean in Fig. 4.

**Table 2.** Per-run profiling results. Three hours per run, 136 complete duty cycles after exclusion of the first ten minutes as warm-up, approximately 8,166 s of active inference per run.

| Quantity | Native A (night) | Native B (day) | Container A (night) | Container B (day) |
|---|---|---|---|---|
| Frames analysed | 105,629 | 101,692 | 91,073 | 102,467 |
| Inference latency, mean (ms) | 18.13 | 18.28 | 26.44 | 26.57 |
| Inference latency, SD (ms) | 6.16 | 6.24 | 3.62 | 4.09 |
| Inference latency, p95 (ms) | 34.7 | 33.6 | 34.0 | 36.4 |
| Inference latency, p99 (ms) | 37.3 | 38.9 | 41.9 | 44.1 |
| CPU utilisation, mean (%) | 49.81 | 48.23 | 36.79 | 42.78 |
| Memory utilisation, mean (%) | 7.70 | 7.61 | 7.94 | 7.81 |
| SoC temperature, mean (°C) | 63.64 | 64.73 | 60.07 | 61.20 |
| Cyclic peak temperature, mean (°C) | 64.61 | 65.72 | 60.63 | 62.28 |
| Cyclic peak temperature, max (°C) | 68.3 | 68.3 | 63.4 | 65.0 |
| Instantaneous maximum temperature (°C) | 70.0 | 71.0 | 64.5 | 66.1 |
| Node input power, mean (W) | 10.169 | 10.225 | 9.457 | 9.999 |
| Effective throughput (frame s⁻¹) | 12.93 | 12.45 | 11.15 | 12.55 |
| Energy per inference (J) | 0.786 | 0.821 | 0.848 | 0.797 |
| Throttling flag reported | none | none | not instrumented | not instrumented |

**Table 3.** Condition means (n = 2 runs per condition; dispersion is the half-range). Differences are stated as containerised minus native. Percentage differences in node input power are computed against the instrumented supply node and understate the workload-attributable difference (Section 3.4).

| Quantity | Native | Containerised | Difference |
|---|---|---|---|
| Inference latency, mean (ms) | 18.21 ± 0.08 | 26.51 ± 0.07 | +8.30 ms (+45.6 %) |
| Inference latency, p95 (ms) | 34.2 ± 0.6 | 35.2 ± 1.2 | +1.0 ms (+3.0 %) |
| Inference latency, p99 (ms) | 38.1 ± 0.8 | 43.0 ± 1.1 | +4.9 ms (+12.9 %) |
| CPU utilisation (%) | 49.02 ± 0.79 | 39.79 ± 3.00 | −9.24 pp |
| Memory utilisation (%) | 7.66 ± 0.05 | 7.88 ± 0.07 | +0.22 pp |
| Cyclic peak temperature (°C) | 65.17 ± 0.56 | 61.46 ± 0.83 | −3.71 °C |
| Node input power (W) | 10.197 ± 0.028 | 9.728 ± 0.271 | −0.469 W |
| Effective throughput (frame s⁻¹) | 12.69 ± 0.24 | 11.85 ± 0.70 | −0.84 (−6.6 %) |
| Energy per inference (J) | 0.804 ± 0.018 | 0.822 ± 0.026 | +0.018 J (+2.3 %) |
| Capture-to-capture loop period, median (ms) | 76.0 ± 0.0 | 83.0 ± 7.0 | +7.0 ms |
| Inference as share of loop period (%) | 23.95 ± 0.05 | 32.2 ± 2.8 | +8.3 pp |

Not every effect separates cleanly at n = 2. Taking an effect to replicate only when both native runs fall on the same side of both containerised runs, six of the nine quantities in Table 3 do so: mean latency, p99 latency, processor utilisation, memory utilisation, cyclic peak temperature and node input power. Three do not. The p95 latencies interleave (native 34.7 and 33.6 ms; containerised 34.0 and 36.4 ms), as do effective throughput (12.93 and 12.45 against 11.15 and 12.55 frame s⁻¹) and energy per inference (0.786 and 0.821 against 0.848 and 0.797 J). The condition differences reported for those three quantities rest on run-to-run spread of the same order as the difference itself and are reported here as unresolved rather than as effects. Memory overhead was negligible at 0.22 percentage points, consistent with the use of a minimal base image.

Two features support the internal validity of the comparison. First, day/night interleaving permits the ambient contribution to be estimated directly: within each condition the daytime run recorded a mean SoC temperature approximately 1.1 °C above its night-time counterpart (native 63.64 vs 64.73 °C; containerised 60.07 vs 61.20 °C). The 3.71 °C cyclic-peak difference attributed to condition is therefore roughly three times the observed ambient effect, and the effect direction holds within each time-of-day stratum considered separately. Second, replicate agreement was close in the native condition (latency 18.13 and 18.28 ms; power 10.169 and 10.225 W). The containerised condition was more variable, particularly in processor utilisation (36.79 % and 42.78 %) and throughput (11.15 and 12.55 frame s⁻¹); the containerised CPU and throughput figures should accordingly be read as the less precisely estimated of the two.

The throttling flag requires care. The native runs polled `vcgencmd get_throttled` and recorded no undervoltage or frequency-capping event. The container image does not ship `vcgencmd`, so the same field logged `Unknown` for every containerised frame; the containerised runs are therefore uninstrumented for throttling, not clear of it. Temperature is instrumented identically in both, and it is the stronger evidence here: the maximum cyclic peak observed across all four runs was 68.3 °C and the maximum instantaneous per-frame temperature 71.0 °C, both well below the 82 °C threshold at which the Raspberry Pi 5 begins capping clocks. On that basis neither condition operated in a thermally limited regime during profiling, so the 3.71 °C difference reflects steady-state thermal load and should be read as additional headroom rather than as the difference between a throttled and an unthrottled node. Section 4.8 reports a deployment regime in which that headroom is largely consumed.

**The profiling loop is acquisition-bound.** Inference does not occupy the whole of the capture-to-capture interval. Taking the median gap between consecutive frame timestamps as the loop period, the native runs cycled at 76.0 ms and the containerised runs at 76.0 and 90.0 ms, against mean inference times of 18.2 and 26.5 ms. Inference thus accounts for roughly a quarter of the native loop and a third of the containerised one; the remainder is frame acquisition, colour conversion, telemetry sampling and the logging write. Two consequences follow. The effective throughput figures in Table 3 are properties of the whole loop and not of the classifier, which is part of why a 45.6 % difference in inference time yields only a 6.6 % difference in throughput; the loop period itself also differed between the two containerised runs, which is one source of the unresolved throughput spread noted above. And the node, as profiled, spent most of its time waiting on the camera, which bears on the comparison's external validity (Section 5.4).

**A mean difference is not a tail difference.** The 45.6 % gap in mean latency shrinks at the 95th percentile, where the two conditions fall within about 1 ms of each other (34.2 ms native, 35.2 ms containerised), although as noted above the individual runs interleave there. At the 99th percentile the gap reopens to 4.9 ms. The native distribution is faster in its typical case but substantially wider (SD 6.2 ms against 3.9 ms); the containerised distribution is slower but tighter.

The consequence for the alert path is a sum rather than a maximum. Confirming a class takes W consecutive frames, so at the deployed W = 5 the accumulated cost of the latency difference is five times the mean difference, 41.5 ms, and the percentile behaviour is irrelevant to it. Section 5.1 places that figure against the rest of the path.

### 4.2. Normalisation by completed work

Reporting power at a fixed duty cycle implicitly assumes that both conditions perform the same amount of work in the same wall-clock time. They do not. Over an identical 8,166 s of active inference, the containerised condition completed 6.6 % fewer inferences (Table 3).

Fig. 5 shows the consequence. The containerised configuration drew 0.469 W less at the supply, but also delivered 0.84 fewer inferences per second; dividing one by the other, energy per inference was 0.822 J containerised against 0.804 J native — **2.3 % higher, not lower**. The apparent power advantage is an artefact of reduced throughput, and reverses under normalisation.

The magnitude of the reversal is small and, at n = 2 per condition with a half-range of 0.018–0.026 J, it is not distinguishable from replicate spread; the defensible statement is that per-inference energy is *not detectably different* between conditions, and specifically that the measured power reduction does not constitute evidence of improved efficiency. That statement is nonetheless the opposite of what the unnormalised figure implies, and we take it as a general point: comparisons of this kind should report energy per unit of completed work alongside instantaneous power, because a configuration that is slower at fixed duty cycle will appear more power-efficient for that reason alone.

### 4.3. What the comparison can and cannot attribute

The differences in Section 4.1 are between two configurations that differ in four respects simultaneously: execution mode, interpreter version, inference-runtime version and base image (Table 1). Fig. 6a states this structure.

The consequence is a bound on the claim. The measurement supports the statement that *this containerised configuration*, as built and as a practitioner following common practice would build it, exhibited the reported differences relative to *this host configuration*. It does not support the statement that containerisation costs 45.6 % in inference latency on this class of hardware.

The size of the confound is not a matter of conjecture. As established in Section 3.3, package metadata determines that the container's inference runtime was ONNX Runtime 1.19.2, the last release publishing a `cp39` `aarch64` wheel, while the host ran 1.29.0, released the day before the first run. Eighteen releases and roughly 23 months separate them, a period over which the ARM64 kernels beneath a convolutional forward pass were substantially reworked. A latency difference of the magnitude observed is well within the range such a gap could produce on its own, and the older C library compounds it.

Three observations in the data are consistent with a runtime-attributable component, though none is decisive. The containerised condition was simultaneously slower per frame and lower in processor utilisation, which is the signature of a differently parallelised or differently vectorised kernel rather than of added indirection — supervisory overhead would be expected to raise processor utilisation, not lower it by 9.24 percentage points. The containerised latency distribution was markedly tighter (Table 2), again more consistent with a different implementation than with a constant added cost. And memory overhead, which is the component of containerisation cost most directly attributable to the mechanism itself, was negligible.

We therefore report the measurement as bounded and specify the design that would decompose it (Fig. 6b). Three conditions are required, not two:

- **A** — native execution, host interpreter and runtime;
- **B** — containerised execution, with interpreter, inference runtime, numerical libraries and base image **pinned to the host's versions**;
- **C** — containerised execution with the as-deployed image.

Then B − A isolates the cost of containerisation with the software stack held constant, C − B isolates the cost of the image's runtime version with execution mode held constant, and C − A is the quantity measured here, now decomposed. Replication at n ≥ 3 per condition with run-level analysis, randomised run order and per-run environment manifests would additionally support interval estimation, which the present design does not.

This is stated at length because we believe the problem generalises. A container image is a frozen dependency resolution; unless its versions are pinned to the host's, a container-versus-host comparison measures the dependency resolution as much as the isolation mechanism. Published overhead figures that do not report the interpreter and runtime versions of both sides are, on this argument, not portable between studies. The environment-capture routine described in Section 3.9 is released as the minimum instrumentation required to make such figures comparable.

### 4.4. Behaviour under extended continuous inference

The supplementary session provides a complementary operating point with longer uninterrupted active periods: 3 h 33 min across 313,011 frames under a 180 s / 45 s duty cycle. Inference latency averaged 18.20 ms (SD 5.65 ms) with a maximum of 94.8 ms during transient system load. SoC temperature remained stable throughout (mean 67.0 °C, maximum 71.0 °C), never approaching the throttling threshold, and mean processor utilisation was 72.9 %, leaving headroom for additional onboard processing. Thermal cycling was shallower than in Fig. 3, consistent with longer active periods driving the SoC closer to quasi-steady state before each sleep interval.

Thermal headroom is therefore preserved under longer uninterrupted inference periods on this hardware and in this enclosure.

### 4.5. Alerting filter sensitivity

The confidence-gate and majority-vote filter was replayed against the recorded field log (5,989 frames, 151.8 s of active inference across three duty cycles) across the threshold–window grid. Fig. 7 reports flip suppression, the proportion of raw frame-to-frame class transitions that do not survive to a confirmed-class transition, together with the resulting confirmed-transition count.

Suppression rises monotonically in both parameters, from 51.3 % at τ = 0.3, W = 3 to 92.6 % at τ = 0.7, W = 7. The deployed configuration (τ = 0.70, W = 5) achieves 90.7 % suppression, reducing 624 raw transitions to 58 confirmed ones. The gradient across the grid is steeper in τ than in W: raising the threshold from 0.3 to 0.7 at fixed W = 5 improves suppression by 26.2 percentage points, while extending the window from 3 to 7 at fixed τ = 0.7 improves it by 2.5. Confidence gating, not temporal extension, does most of the work on this log, and extending the window past five frames buys little at the cost of added confirmation latency.

### 4.6. Training-data composition across architectures

Fig. 8 reports the accuracy difference between the Real-Only and sample-size-matched Real + Proxy conditions for each of the eleven architectures. Three architectures favour Real-Only, three are exactly tied, and five favour Real + Proxy.

All eleven are retained in the analysis below. Two of them sit far from the rest: SqueezeNet at −29.70 pp and ViT-B/16 at −7.35 pp, in both cases because the Real-Only run reached an accuracy well below what the same architecture reached under Real + Proxy. Both are plausibly optimisation failures at a single seed and not data effects — SqueezeNet's Real-Only accuracy of 64.71 % against 94.41 % under Real + Proxy is a larger swing than any data-composition effect at this scale could produce — but a single unseeded run offers no way to establish that, and discarding the two runs that most strongly favour one condition would bias the summary toward the other. They are kept, and the summary statistics are chosen to be insensitive to them.

On that basis the central tendency is zero. The median difference across the eleven architectures is exactly 0.00 pp; the mean, −3.37 pp, is a property of the two extreme runs and not of the set. Of the eight architectures that differ at all, three favour Real-Only and five favour Real + Proxy, which a two-sided exact sign test does not distinguish from chance (p = 0.727); an exact Wilcoxon signed-rank test over the same eight, which weights magnitude as well as direction, gives p = 0.383. Magnitudes point the same way: the median absolute difference is 0.59 pp, and on a validation partition of approximately 340 images one image is worth 0.29 pp, so the typical architecture separates the two conditions by two images. Seven of eleven differ by two validation images or fewer, nine of eleven by four or fewer. Since the two conditions were evaluated on different validation partitions (Section 3.8) and neither was repeated across seeds, differences of that size are not distinguishable from the variability the protocol itself introduces.

The conclusion is negative and is stated as such: under this protocol, at this dataset scale, no effect of cross-domain proxy augmentation is detectable in either direction across eleven architectures, and the apparent advantage of site-collected data observed on the deployed architecture does not replicate. The result is robust to how the two extreme runs are treated — excluding them leaves the remaining nine spanning +0.89 to −1.17 pp, which supports the same conclusion — but it does not need that exclusion to hold. For a practitioner assembling a training set for a comparable deployment, the implication is that padding a small field dataset with visually similar imagery from a public collection can be neither justified nor dismissed on the basis of a single-architecture comparison at this scale. It is a configuration choice that requires a seeded, repeated evaluation to resolve, and the two outlying runs are a reminder that at n = 1 per cell a single bad optimisation trajectory is indistinguishable from a treatment effect.

### 4.7. Out-of-distribution routing

The supplementary session (Section 3.6) supports two claims about the behaviour of the closed-set three-class classifier when the scene contains material absent from the training distribution. They rest on very different amounts of evidence and are separated accordingly (Fig. 9).

**A large-sample claim about stability.** Of 313,011 frames, 312,981 (99.99 %) were classified Background. All 30 non-Background frames occurred within the first 43 s of the session, between frame 87 and frame 1,327. The remaining **311,683 consecutive frames, 3 h 32 min of an unattended, quiescent scene, produced no non-Background prediction of any kind**, at a frame-level confidence threshold more permissive than the deployed one. With no event observed in that stretch, the rule of three places the upper bound of a 95 % confidence interval on the per-frame false-positive rate at 3/311,683, or approximately 1 × 10⁻⁵. Applying the deployed confirmation filter removes a further four of the 30 non-Background frames, leaving 26 confirmed. On this evidence the node does not generate spurious alerts when the observed scene is static.

**A small-sample observation about routing.** The 30 non-Background frames are not 30 independent observations. Grouping frames separated by more than 30 frames (approximately one second) into distinct events yields **four intrusion events**, of 19, 2, 1 and 8 frames, corresponding to non-plant objects entering the field of view during the period when the session was being set up. All four were assigned to Disease, at confidences from 0.52 to 0.99. Under a null hypothesis in which a novel input falls into one or other of the two plant classes with equal probability, four events landing in the same class, whichever of the two it is, occurs with probability 0.125, which is not a result. Two further cautions apply. Healthy was not predicted on a single frame anywhere in the session, so the absence of Healthy among the four events is not informative about routing: it is consistent with the four events preferring Disease and equally consistent with a decision boundary that placed nothing at all in Healthy on this scene. And the four events are not independent of one another in the way the null assumes, since all four arose from objects present during the same short set-up period and may have been the same object. The pattern is suggestive of asymmetric routing, and would be consistent with the geometry expected of a closed-set softmax over three visually homogeneous classes (Hendrycks and Gimpel, 2017), in which one class occupies the residual region of feature space. **Four events do not establish it**, and we do not claim otherwise. Section 5.3 sets out what would.

A separate record from the live field deployment is consistent with the same pattern: the two alerts reproduced in Fig. 10, delivered during an earlier run on the day of the field session (Section 4.8), were both triggered by a metal tool edge transiently entering the field of view and were assigned to Disease. The alerting path executed correctly on an input the classifier misjudged, which illustrates that notification reliability and classification reliability are separate properties of the node.

### 4.8. Field log characteristics

The node is designed as an **on-demand expert-assistance tool rather than an autonomous 24-hour monitoring service**. Continuous unattended inference through static night-time conditions, when no agronomist is present to act on an alert, offers little operational value against its sustained power and thermal cost. The intended operating pattern is a supervised scouting session, during which the node observes a fixed section of canopy while the agronomist is on site or scheduled to visit. The duty cycle, the alerting design and the measurements reported here should all be read against that operating pattern rather than against a continuously operating installation.

The live greenhouse deployment comprises 5,989 frames spanning 181.8 s of wall-clock time, of which 151.8 s across three duty cycles were active inference, giving an effective rate of approximately 39.5 frame s⁻¹ under the 60 s / 15 s duty cycle. Whenever a Disease classification passed the confidence threshold and the confirmation window, the webhook executed reliably, delivering the payload to the designated device with an end-to-end latency averaging under 2.5 s.

Fig. 10 reproduces two such alerts. They are **not** drawn from the deposited log: they were delivered at 16:46 and 16:47 on the same day, roughly two and a half hours before the 19:11 session, during an earlier run whose telemetry was not retained. The two records are distinguishable on their face — the alerts report 38.2 ms and 14.8 ms of inference latency, and the deposited log's fastest frame is 15.1 ms — and we state the separation rather than presenting the figure as an excerpt of the log. The values visible in the figure are therefore read from the alert payloads themselves and are the only quantities in this article not emitted by the released analysis script. Both alerts were triggered by a metal tool edge transiently entering the field of view, the same out-of-distribution object class analysed in Section 4.7, and were confirmed at 0.73 and 0.71 confidence against the deployed threshold of 0.70. The alerting path executed exactly as designed on an input the classifier misjudged, which illustrates directly that notification reliability and classification reliability are separate properties of the node.

**The deployed node runs far hotter than the profiled one.** This is the single largest difference between the bench measurements of Section 4.1 and the field, and it runs against them. From a cold start at 46.9 °C, SoC temperature rose to a first-cycle peak of 70.5 °C and a second-cycle peak of 76.5 °C, with an instantaneous maximum of 77.7 °C and a session mean of 70.0 °C; 55.5 % of frames were recorded above 70 °C and 813 above 75 °C. Mean processor utilisation was 94.1 %, against 49.0 % native and 39.8 % containerised during profiling. The peak sits 4.3 °C below the 82 °C threshold at which the Raspberry Pi 5 caps clocks, and the trajectory across the three cycles was still rising when the session ended, so a longer session would plausibly have reached it. The profiling runs, which cycled 60 s of inference against 30 s of sleep in a controlled enclosure and peaked at 68.3 °C, do not characterise this regime. Whatever thermal headroom containerisation returns at the bench is small relative to the margin a deployed session actually consumes, and Section 5.1 treats that as the governing constraint rather than as a footnote.

In the absence of frame-level ground truth for this session, prediction confidence was used to characterise the temporal structure of likely ambiguous frames. Of all frames, 92 (1.54 %) fell below a confidence of 0.5. Of these, 72.3 % occurred as isolated single-frame events and 9.2 % in sustained runs of three or more consecutive frames, with a maximum run of four; low confidence on this log is therefore driven more often by transient single-frame noise than by sustained scene conditions. Low-confidence frames exhibited a class-flip rate of 50.0 % against 4.5 % among high-confidence frames, an elevenfold difference, indicating that confidence is a meaningful indicator of prediction instability and not noise uncorrelated with model behaviour. Correlation between confidence and system telemetry was negligible (temperature r = −0.050; processor utilisation r = 0.015; latency r = −0.023), so this instability is not attributable to thermal state or system load and more plausibly reflects intrinsic visual ambiguity in the captured frames.

Manual review of this session indicated an operational accuracy of approximately 88.5 %. The figure was obtained post hoc by a single annotator scoring predictions against the visible content of sampled frames, without independent verification and without a pre-registered sampling protocol. It is reported to indicate the magnitude of the gap between held-out validation accuracy and in-situ behaviour, not to certify field performance, and Section 5.4 treats it accordingly.

## 5. Discussion

### 5.1. Where latency sits in the critical path

The profiling results revise, rather than confirm, the intuition that motivates containerised edge deployment. Containerisation is usually defended on operational grounds alone, on reproducibility, dependency isolation and over-the-air updatability, with the implicit assumption that those benefits are paid for in performance. The measurements here complicate that accounting in both directions. The containerised configuration ran 3.71 °C cooler at cyclic peak and 9.24 percentage points lighter on the processor, while being 45.6 % slower per inference in the mean; and the power advantage that appears alongside the thermal one does not survive normalisation by work completed.

Whether the latency cost matters depends on where latency sits in the critical path, and for the alerting architecture described here it is not close. Confirming a class takes five consecutive frames, so the accumulated cost of an 8.3 ms difference in mean inference time is 41.5 ms. The alert then leaves the node over an HTTPS round trip that took under 2.5 s end to end in the deployed session. The inference-time difference is thus about 1.7 % of the path an operator actually waits through, and it is a sum over five frames, not anything governed by the tail, so the percentile behaviour in Table 3 does not enter. Duty-cycled monitoring is in this respect close to an ideal workload for containerisation: it is throughput-tolerant, so it can spend mean latency without spending meaningful confirmation latency.

The conclusion would invert for a latency-critical task such as closed-loop actuation or high-frame-rate tracking, where per-frame inference time is the binding constraint. It would also invert for a throughput-limited task, where the 6.6 % reduction in completed inferences would be a direct loss of capability rather than a spare margin.

What the thermal margin is worth can be read off the field log instead of argued in the abstract, and the reading is not comfortable. Neither profiling condition came close to the throttling threshold: mean cyclic peaks of 65.2 °C native and 61.5 °C containerised, and a hottest single cycle of 68.3 °C, against a limit of 82 °C. The deployed session behaved quite differently. It reached 77.7 °C within three duty cycles from a 46.9 °C cold start, held a mean of 70.0 °C, and was still climbing cycle over cycle when it ended, leaving 4.3 °C of margin (Section 4.8). The bench does not characterise the field, and the reason is visible in the numbers: the deployed node ran at 94.1 % processor utilisation against 49.0 % native during profiling, in an enclosed greenhouse at a tropical ambient instead of on a climate-controlled bench, in an enclosure that restricts convection.

Placed against that 4.3 °C, a 3.71 °C difference between conditions is not a rounding error. It is of the same order as the entire remaining margin, and that is what would make it consequential if it transferred. Two things prevent us from claiming that it does. The field session was run in one configuration only, so no containerised field counterpart exists to compare it against; and the profiling difference was measured at 49 % utilisation, under shallow cycling and at bench ambient, which the deployed regime does not resemble. The honest statement is a conditional one: if a difference of this order survives into the deployed regime, it is large relative to the margin that regime actually has, which makes the question worth answering directly. Measuring both conditions in situ, over sessions long enough to reach quasi-steady state, is the obvious next experiment and a cheap one.

### 5.2. Why container-overhead figures are not portable

Section 4.3 reported that the comparison performed here cannot attribute its result to containerisation alone. We think this is worth more than a limitation paragraph, because the mechanism that produced it is not specific to this study.

A container image is a frozen dependency resolution. The common practice that produced the configuration measured here, which is to select a base image by interpreter version, install the inference runtime from an unpinned requirement and ship it, resolves the runtime to whatever version was current and compatible at build time. The host, meanwhile, tracks its distribution. The two drift apart, and they drift apart in exactly the component whose performance is being measured. For ARM64 inference this is not a marginal effect: the kernels beneath a convolutional forward pass have been substantially reworked across recent ONNX Runtime releases, and a comparison that does not hold that version fixed is partly a measurement of the release cycle.

Three practical consequences follow for work in this area. First, a container-versus-host comparison requires the three-condition design of Fig. 6b, not two conditions; the intermediate condition, in which the container carries the host's exact software stack, is the one that isolates the mechanism of interest. Second, an overhead figure is only interpretable if the interpreter, inference-runtime, numerical-library and C-library versions of *both* sides are reported; we would encourage this as a reporting norm, and the manifest routine released with this paper is offered as one way to satisfy it at no cost to the experimenter. Third, and this is the practitioner-facing implication, the version pinned inside an image is a performance-relevant configuration choice, not a packaging detail. A deployment that pins its runtime for reproducibility, as it should, also pins whatever performance that runtime had on the day the image was built, and an image rebuilt a year later is not the same artefact.

### 5.3. Where novel inputs go, and why Background was not available to them

The routing observation in Section 4.7 is easy to overstate and we have tried not to. What the data support is that in four intrusions by non-plant objects, all four were assigned to Disease and none to Healthy or Background, at confidences spanning most of the usable range. What four events cannot do is separate a structural explanation from chance.

A conjecture is nonetheless available, and it follows from how the three classes were constructed rather than from any property of the architecture. The classes are not symmetric (Section 3.8). **Background is not a generic "everything else" class**: it consists of distant outdoor scenes from beyond the greenhouse (buildings, hard standing, open ground), and is distinguished from the other two classes principally by scene scale and context. Healthy and Disease are both close-range foliage, separated by appearance.

The conjecture is that for a novel object presented close to the lens, which is what a tool edge, a hand or a piece of fabric entering the frame amounts to, scene scale dominates the representation. If it does, such an object cannot resemble a distant outdoor view, the Background region is effectively unavailable to it however unfamiliar it is, and the decision falls between the two close-range classes. Of those, Disease has the broader training support, aggregating six visually distinct symptom and pest categories against Healthy's single compact appearance. A closed-set softmax must place every input somewhere, and on this account the somewhere is Disease.

We are aware that this argument leans on two criteria at once and that they are in tension. Background is excluded on grounds of scene scale, while Disease is preferred on grounds of appearance breadth — yet Background, spanning buildings, hard standing, open ground and site vegetation, is no less heterogeneous in appearance than Disease is. Applying the breadth criterion consistently would make Background a candidate for absorbing novelty rather than the class excluded from it. The argument therefore holds only if scene-scale cues dominate appearance-breadth cues in the learned feature space, and nothing here establishes that they do. We state it as a conjecture for that reason, not out of modesty.

Two observations are at least consistent with it. The held-out confusion matrix of the deployed model contains only two errors and both are Healthy samples assigned to Disease, the same direction. And the classifier is trained on hand-held close-range imagery but deployed against a fixed camera (Section 3.1), so the deployed viewpoint is already at the edge of the training distribution before any novel object appears. Neither observation discriminates between the scene-scale account and the breadth account.

We stress that this is a conjecture consistent with four events, not a demonstration. Establishing it requires a controlled probe, not an observational one: a set of N distinct novel object classes, each presented multiple times under varied illumination, framing and working distance, with the class assignment recorded per presentation, and with distance deliberately varied so that the scene-scale hypothesis above can be tested directly — the prediction is that the same object should become assignable to Background as it recedes. That design would yield an event count adequate to the claim, and it is cheap: it requires no greenhouse access and no retraining.

The accompanying mitigation is not a matter of tuning. Neither a higher confidence threshold nor a longer confirmation window can change which class a confidently misclassified novel object is routed into; both suppress transient instability, which Section 4.8 shows is a real and separable phenomenon, but they are the wrong instrument for this failure. An explicit rejection mechanism addresses it directly: a maximum-softmax-probability baseline (Hendrycks and Gimpel, 2017) or an energy-based score, with a dedicated reject outcome. A complementary and cheaper measure follows from the analysis above: a Background class collected at the working distance and framing of the deployed camera, rather than as distant scenery, would give novel close-range objects somewhere to go.

### 5.4. Limitations

*Attribution of the profiling comparison.* The two profiling conditions differed in interpreter, inference-runtime and base-image versions as well as in execution mode (Table 1, Section 4.3). The reported differences are properties of the two configurations as measured and are not attributable to containerisation alone. The three-condition design in Fig. 6b is specified but was not performed. The container's runtime version is determined from package metadata and not read from the image, the storage media having since been reimaged; the determination is exact, because exactly one release satisfies the interpreter and C library constraints, but it is an inference from metadata and is stated as such.

*Statistical strength of the profiling comparison.* Two runs per condition bound run-to-run spread but do not support significance testing or interval estimation, and none is claimed. Six of the nine reported quantities separate the conditions cleanly in the sense that both runs of one condition fall on the same side of both runs of the other; three do not, and are reported as unresolved, not as effects (Section 4.1). The containerised condition showed appreciable spread in processor utilisation (36.79 % and 42.78 %) and throughput (11.15 and 12.55 frame s⁻¹); those two figures in particular should be treated as indicative.

*Measurement node and instrument.* Power was measured at the Type-C supply input to the whole assembly, not at the board's 5 V rail, which the supply module does not instrument (Section 3.4). The figures include two cascaded conversion stages and the module's housekeeping current; the implied board-level figure of 8–9 W rests on an assumed 80–90 % efficiency that was not measured. Absolute differences between conditions are meaningful; percentages are computed against an inflated denominator. The module's Power Delivery controller is a charge-management device whose telemetry resolution and accuracy are undocumented, so absolute values should be treated as uncalibrated; the between-condition difference is unaffected, as both conditions were measured through the identical chain. Bus current was not logged alongside bus power, so the direction of flow is inferred from the experimental configuration and not read directly. Sleep-phase power was not logged, so an idle baseline cannot be subtracted, and the per-inference energy figures in Table 3 carry the same offset.

*The profiled regime is not the deployed regime.* This bounds the external validity of Section 4.1 more than any other limitation here. The profiling loop is acquisition-bound: inference occupies 24 % of the native loop period and 32 % of the containerised one, the rest being frame acquisition, colour conversion, telemetry sampling and logging, so a difference in inference time is diluted roughly fourfold before it reaches throughput. Profiling also used a USB V4L2 camera at 640 × 480, where the deployment uses a Camera Module 3 through libcamera, and ran on a climate-controlled bench at about 49 % processor utilisation, where the field session ran at 94.1 % and 26 °C warmer at the SoC (Section 4.8). Effects measured in the profiling regime should not be assumed to transfer to the deployed one in magnitude, and the thermal effect in particular was not measured in situ in both conditions.

*Single platform and single inference engine.* One board, one enclosure, one camera path and one execution provider were evaluated. Alternative edge-inference engines were not benchmarked, and the results should not be extrapolated to accelerator-equipped boards.

*Field evidence.* All field data originate from a single commercial greenhouse, and the in-situ deployment was brief: 151.8 s of active inference across three duty cycles, yielding 5,989 frames. This is sufficient to verify the end-to-end alerting path and to supply a densely sampled record for the filter analysis, but it is not an extended field trial. The 88.5 % operational accuracy figure rests on single-annotator post-hoc review without a pre-registered sampling protocol and should be read as indicative only. Alert-level precision and event-level recall would require frame-level ground truth for the full log and are not reported. The two alerts reproduced in Fig. 10 come from an earlier run on the same day whose telemetry was not retained, so they cannot be cross-referenced against the deposited log; they evidence that the transport path executed, and nothing more.

*Training-data comparison.* The split was drawn without a fixed generator seed, so the two conditions were evaluated on different validation partitions; each architecture was trained once per condition; and the validation partition is small enough that one image is 0.29 percentage points. The result is reported as an absence of detectable effect under this protocol, not as evidence of equivalence.

*Provenance of the proxy imagery.* The public collection from which the 280 proxy images were drawn was not recorded, so it cannot be cited and its licence cannot be established. The images are consequently not deposited, and the Real + Proxy condition cannot be reproduced exactly; the Real-Only condition can. This is a failure of record-keeping on our part and we report it as such. It weakens the comparison asymmetrically: the finding is an absence of detectable effect, so the missing provenance bears on a positive claim about proxy augmentation that this study does not make. It would matter a great deal to a study that did.

*Out-of-distribution routing.* Four intrusion events, all within a 43 s window at the start of one session, support an observation and not a structural claim. The explanation offered in Section 5.3 is consistent with those four events and with the construction of the classes, but it was not tested; the controlled probe specified there would test it.

*Train–deploy viewpoint mismatch.* The classifier is trained on hand-held close-range imagery and deployed against a fixed camera at a different working distance and framing (Section 3.1). No measurement isolates the contribution of that mismatch to the gap between held-out and in-situ behaviour.

*Model provenance.* The weights of the originally deployed model were not preserved. Validation accuracies attributed to it are historical and not reproducible from the released artefacts; the released checkpoints are separately trained.

*Operational constraints.* Passing the video device into the container monopolises it, preventing a second containerised consumer of the same camera. The node depends on existing wireless infrastructure, and multi-node contention on shared infrastructure is outside the scope of a single-node study.

## 6. Conclusions

This paper reported a measured case study of a containerised edge vision node on a Raspberry Pi 5, under a duty cycle representative of continuous monitoring, with per-frame compute telemetry and board-level supply instrumentation.

Relative to native execution, the containerised configuration exhibited 45.6 % higher mean inference latency, 9.24 percentage points lower processor utilisation, 3.71 °C lower mean cyclic peak temperature and 0.469 W lower node input power. Six of the nine reported quantities separate the conditions cleanly across both interleaved pairs; three, including effective throughput and energy per inference, do not, and are reported as unresolved. Three qualifications determine how the numbers should be used. The power advantage does not survive normalisation: the containerised configuration completed 6.6 % fewer inferences, and energy per inference was not detectably better, so a comparison of this kind should report energy per unit of completed work alongside instantaneous power. The latency cost is small where it lands: five frames of confirmation accumulate 41.5 ms of it, against an end-to-end alert path under 2.5 s. And the comparison cannot attribute its result to containerisation alone, because the container carried a different interpreter and inference-runtime version from the host; we state that bound explicitly and specify the three-condition design that decomposes it.

We take the third of these to be the most transferable result. A container image freezes a dependency resolution, and unless its versions are pinned to the host's, a container-versus-host measurement is partly a measurement of the release cycle of the inference runtime. Overhead figures reported without the software versions of both sides are not portable between studies, and the per-run environment manifest released with this work is offered as the minimum instrumentation that makes them so.

Two further deployment characteristics were reported. The benefit of augmenting site-collected training data with cross-domain proxy imagery did not replicate across eleven lightweight architectures: nine of eleven differed by four validation images or fewer, which under a protocol that did not fix the split seed or repeat across seeds is not distinguishable from procedural variability. And in 3 h 33 min of continuous inference the node produced no spurious non-Background prediction across 311,683 consecutive frames of a quiescent scene, while the four intrusions by novel non-plant objects that did occur were all routed to one class — an observation that motivates an explicit rejection mechanism, and a controlled probe with enough distinct novel classes to test it properly.

One result cuts against the rest and is worth isolating. The profiling runs were thermally comfortable, their hottest cyclic peak reaching 68.3 °C against an 82 °C limit, but the deployed session reached 77.7 °C within three duty cycles and was still climbing, leaving 4.3 °C of margin at 94.1 % processor utilisation. A 3.71 °C difference between conditions is of the same order as that entire remaining margin, and that is why it would matter operationally — but the field session was run in one configuration only, so whether the difference transfers is untested. It is the measurement we would make next.

Priority future work follows directly from the limitations: the three-condition profiling design with replication sufficient for interval estimation; sleep-phase power logging so that an idle baseline can be subtracted; a controlled out-of-distribution probe; a seeded, repeated training-data comparison; in-situ thermal measurement of both execution conditions over sessions long enough to reach quasi-steady state; and extension of the in-situ deployment to the full duration of a scouting session across multiple facilities.

## Data and code availability

All research data underlying this article are deposited in a public repository under a persistent
identifier: the site-collected greenhouse image dataset from which the classifier was trained, and the
complete deployment telemetry on which every measurement reported here rests, namely per-frame logs and
per-cycle event markers from the four controlled profiling runs, the live field session and the extended
continuous-inference session. The deposit is cited as [dataset] and is available at
https://doi.org/10.5281/zenodo.22857312. The cross-domain proxy imagery used in the secondary training
condition is not included, for the reason given in Section 3.8; every image in the deposit was captured
by the author at the study site.

The analysis code, container definitions, environment-capture routine and the edge inference
application are available at https://github.com/xiaolin200206/edge-container-deployment-measurement. Every table and figure in this article and its supplementary
material is regenerated from the deposited telemetry by the released scripts; `analysis/recompute.py`
emits every quantity stated in the text. The single exception is Fig. 10, a screen capture of two
delivered alerts whose originating run was not retained; the values it shows are read from the alert
payloads and are identified as such in Section 4.8.

## CRediT authorship contribution statement

**Lin Ding Shan:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review & editing, Visualization.

## Declaration of competing interest

The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the author used Claude (Anthropic) to assist in drafting and editing the manuscript text, and in writing the analysis and figure-generation scripts released with this work. All analyses were independently re-run and verified by the author, and every reported quantity is reproducible from the deposited data by the released scripts. The author reviewed and edited all content and takes full responsibility for the content of the publication.

---

## References

Apicella, A., Arpaia, P., Capobianco, S., Caputo, E., Cioffi, A., Esposito, A., Isgrò, F., Manzo, M., Moccaldi, N., Pau, D., Toscano, R., 2026. Energy consumption assessment in embedded AI: metrological improvements of benchmarks for edge devices. Computer Standards & Interfaces 97, 104095.

Barbedo, J.G.A., 2018. Factors influencing the use of deep learning for plant disease recognition. Biosystems Engineering 172, 84–91.

Baresi, L., Quattrocchi, G., Rasi, N., 2024. A qualitative and quantitative analysis of container engines. Journal of Systems and Software 210, 111965.

Bellavista, P., Zanni, A., 2017. Feasibility of fog computing deployment based on Docker containerization over Raspberry Pi, in: Proceedings of the 18th International Conference on Distributed Computing and Networking (ICDCN), Pune, India, pp. 1–10.

Benoit-Cattin, T., Velasco-Montero, D., Fernández-Berni, J., 2020. Impact of thermal throttling on long-term visual inference in a CPU-based edge device. Electronics 9 (12), 2106.

Chen, J., Chen, J., Zhang, D., Sun, Y., Nanehkaran, Y.A., 2020. Using deep transfer learning for image-based plant disease identification. Computers and Electronics in Agriculture 173, 105393.

Chen, J., Ran, X., 2019. Deep learning with edge computing: a review. Proceedings of the IEEE 107 (8), 1655–1674.

Ferentinos, K.P., 2018. Deep learning models for plant disease detection and diagnosis. Computers and Electronics in Agriculture 145, 311–318.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., Wichmann, F.A., 2020. Shortcut learning in deep neural networks. Nature Machine Intelligence 2, 665–673.

Gookyi, D.A.N., Wulnye, F.A., Arthur, E.A.E., Ahiadormey, R.K., Agyemang, J.O., Agyekum, K.O.B.O., Gyaang, R., 2024. TinyML for smart agriculture: comparative analysis of TinyML platforms and practical deployment for maize leaf disease identification. Smart Agricultural Technology 8, 100490.

Hendrycks, D., Gimpel, K., 2017. A baseline for detecting misclassified and out-of-distribution examples in neural networks, in: Proceedings of the International Conference on Learning Representations (ICLR), Toulon, France.

Howard, A.G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., Adam, H., 2017. MobileNets: efficient convolutional neural networks for mobile vision applications. arXiv:1704.04861.

Hurlbert, S.H., 1984. Pseudoreplication and the design of ecological field experiments. Ecological Monographs 54 (2), 187–211.

Iandola, F.N., Han, S., Moskewicz, M.W., Ashraf, K., Dally, W.J., Keutzer, K., 2016. SqueezeNet: AlexNet-level accuracy with 50× fewer parameters and <0.5 MB model size. arXiv:1602.07360.

Kamilaris, A., Prenafeta-Boldú, F.X., 2018. Deep learning in agriculture: a survey. Computers and Electronics in Agriculture 147, 70–90.

Lee, J., Choi, H., Tak, B., 2026. Performance analysis of microVMs and containers for edge computing: a focus on file and network I/O. Future Generation Computer Systems 176, 108013.

Li, H., Ota, K., Dong, M., 2018. Learning IoT in edge: deep learning for the Internet of Things with edge computing. IEEE Network 32 (1), 96–101.

Mahmud, R., Toosi, A.N., 2021. Con-Pi: a distributed container-based edge and fog computing framework for Raspberry Pis. arXiv:2101.03533.

Merkel, D., 2014. Docker: lightweight Linux containers for consistent development and deployment. Linux Journal 2014 (239), 2.

Mohanty, S.P., Hughes, D.P., Salathé, M., 2016. Using deep learning for image-based plant disease detection. Frontiers in Plant Science 7, 1419.

Morabito, R., Kjällman, J., Komu, M., 2015. Hypervisors vs. lightweight virtualization: a performance comparison, in: Proceedings of the IEEE International Conference on Cloud Engineering (IC2E), Tempe, AZ, USA, pp. 386–393.

Nawaz, M., Babar, M.I.K., 2025. IoT and AI for smart agriculture in resource-constrained environments: challenges, opportunities and solutions. Discover Internet of Things 5 (1), 24.

O’Grady, M.J., Langton, D., O’Hare, G.M.P., 2019. Edge computing: a tractable model for smart agriculture? Artificial Intelligence in Agriculture 3, 42–51.

ONNX Runtime developers, 2021. ONNX Runtime. https://onnxruntime.ai/ (accessed [date]; versions 1.19.2 and 1.29.0 used in this work).

Pahl, C., Brogi, A., Soldani, J., Jamshidi, P., 2019. Cloud container technologies: a state-of-the-art review. IEEE Transactions on Cloud Computing 7 (3), 677–692.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., et al., 2019. PyTorch: an imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems 32.

Picon, A., Alvarez-Gila, A., Seitz, M., Ortiz-Barredo, A., Echazarra, J., Johannes, A., 2019. Deep convolutional neural networks for mobile capture device-based crop disease classification in the wild. Computers and Electronics in Agriculture 161, 280–290.

Ray, P.P., 2017. Internet of things for smart agriculture: technologies, practices and future direction. Journal of Ambient Intelligence and Smart Environments 9 (4), 395–420.

Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.-C., 2018. MobileNetV2: inverted residuals and linear bottlenecks, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Salt Lake City, UT, USA, pp. 4510–4520.

Santos, E.A., McLean, C., Solinas, C., Hindle, A., 2018. How does docker affect energy consumption? Evaluating workloads in and out of Docker containers. Journal of Systems and Software 146, 14–25.

Sapna, Gauttam, H., Chauhan, V., Pattanaik, K.K., Trivedi, A., Ghosh, H., 2026. A comprehensive review of edge computing empowered smart agriculture: trends, opportunities and future directions. Computers and Electronics in Agriculture 241, 111252.

Shafique, M., Theocharides, T., Bouganis, C.-S., Hanif, M.A., Khalid, F., Hafiz, R., Rehman, S., 2020. Robust machine learning systems: challenges, current trends, perspectives, and the road ahead. IEEE Design & Test 37 (2), 30–57.

Singh, R., Gill, S.S., 2023. Edge AI: a survey. Internet of Things and Cyber-Physical Systems 3, 71–92.

Sladojevic, S., Arsenovic, M., Anderla, A., Culibrk, D., Stefanovic, D., 2016. Deep neural networks based recognition of plant diseases by leaf image classification. Computational Intelligence and Neuroscience 2016, 3289801.

Tan, M., Le, Q., 2019. EfficientNet: rethinking model scaling for convolutional neural networks, in: Proceedings of the International Conference on Machine Learning (ICML), Long Beach, CA, USA, pp. 6105–6114.

Too, E.C., Li, Y., Njuki, S., Li, Y., 2019. A comparative study of fine-tuning deep learning models for plant disease identification. Computers and Electronics in Agriculture 161, 272–279.

Upton, E., Halfacree, G., 2014. Raspberry Pi User Guide, 2nd ed. John Wiley & Sons, Chichester, UK.

Zamora-Izquierdo, M.A., Santa, J., Martínez, J.A., Martínez, V., Skarmeta, A.F., 2019. Smart farming IoT platform based on edge and cloud computing. Biosystems Engineering 177, 4–17.


## Figure captions

**Fig. 1.** Node architecture and measurement chain. (a) The power chain from the mains Power Delivery adapter through the supply module's bidirectional buck–boost stage, the four-series battery pack and the separate 5 V conversion stage to the board, with the two instrumented nodes marked. The 5 V rail that feeds the Raspberry Pi is not instrumented by the module, which is why the reported figures are node input power rather than board power (Section 3.4). (b) The inference and alerting path, with the container boundary indicated: in the containerised condition the entire inference process (acquisition, preprocessing, inference, temporal filtering, telemetry logging and the webhook call) executes inside the container, with the camera and the I2C bus passed through.

**Fig. 2.** Representative imagery from the dataset, sampled deterministically from the deposited class archives by the released script. Background (left) consists of distant outdoor scenes from beyond the greenhouse and is distinguished from the other classes by scene scale and context rather than by object identity; Healthy (centre) and Disease (right) are both close-range foliage. The Disease class aggregates six visually distinct symptom and pest categories (fungal infection, leaf curl, senescent withering, mealybugs, leaf miner and mite damage), and spans a correspondingly wider range of appearances than Healthy. This asymmetry is the basis of the account in Section 5.3. Training imagery was captured hand-held; the deployed node uses a fixed camera (Section 3.1). The cross-domain proxy set used in the Real + Proxy condition is not shown, and is not part of the deposit (Section 3.8).

**Fig. 3.** Representative longitudinal traces from one profiling run of each condition (night-time pair), generated from the released per-frame telemetry. Top: SoC temperature over the three-hour analysis window under the 60 s / 15 s duty cycle, with the 82 °C throttling threshold indicated; the first ten minutes (shaded) are excluded from aggregate statistics as thermal warm-up. The deployed field session reached 77.7 °C on this axis (Section 4.8). Bottom: per-frame processor utilisation (light traces) with rolling means (bold). Neither condition approaches the throttling threshold, and the containerised trace sits consistently below the native one in both quantities.

**Fig. 4.** Per-run values underlying each condition mean, for the five primary quantities. Circles denote night-time replicates and squares daytime replicates; horizontal bars denote condition means. Where the two runs of one condition both fall outside the range of the other, the effect separates the conditions at this sample size; where they interleave, as in throughput, it does not. The wider containerised spread in processor utilisation and throughput is visible directly.

**Fig. 5.** Node input power, effective throughput and energy per inference, by condition. Error bars denote the half-range across two runs. The containerised configuration draws less power but also completes fewer inferences; normalised by work done, its energy per inference is marginally higher, so the apparent power advantage does not survive normalisation.

**Fig. 6.** The attribution problem. (a) The four profiling runs compared two configurations differing in execution mode, interpreter version, inference-runtime version and base image simultaneously; the measured difference is not attributable to any one of them. (b) The three-condition design that decomposes it: B − A isolates containerisation with the software stack held constant, C − B isolates the image's runtime version with execution mode held constant. Condition B was not performed in this study and is specified for replication.

**Fig. 7.** Sensitivity of the alerting filter across the confidence-threshold and temporal-window grid, replayed against the recorded field log (5,989 frames, 151.8 s of active inference). Each cell reports flip suppression and the resulting confirmed-transition count. The deployed configuration (τ = 0.70, W = 5) is outlined. Suppression is governed more strongly by the confidence threshold than by the window length.

**Fig. 8.** Accuracy difference between the Real-Only and sample-size-matched Real + Proxy training conditions across eleven architectures. Shaded bands denote ±1 and ±2 validation images on a partition of approximately 340 images. Note the axis break. SqueezeNet and ViT-B/16 are runs in which the Real-Only condition reached an accuracy far below its Real + Proxy counterpart; both are retained in the analysis, which uses the median and a sign test for that reason. The other nine architectures all fall within four validation images of zero, and the median across all eleven is 0.00 pp.

**Fig. 9.** Out-of-distribution behaviour in the 3 h 33 min continuous session. (a) At session scale, 311,683 consecutive frames of a quiescent scene produced no non-Background prediction; every non-Background frame falls in a 43 s window at session start. (b) Within that window, the thirty non-Background frames resolve into four distinct intrusion events, all assigned to Disease, at confidences spanning 0.52 to 0.99. Healthy was not predicted on any frame of the session, so the absence of Healthy here is uninformative about routing (Section 4.7). The deployed confidence threshold is indicated.

**Fig. 10.** Two alerts delivered to a mobile device by the deployed node, reproduced as a screen capture from the operator's message history and cropped to the message content. Each payload carries the confirmed frame together with the SoC temperature, prediction confidence and per-frame inference latency at the moment of confirmation. These alerts come from a run earlier on the same day as the deposited field log, not from that log itself (Section 4.8), so the values shown are read from the payloads and not from the released telemetry; the overlay on the upper frame was cut off by the preceding message in the original capture. Both alerts were triggered by a metal tool edge transiently entering the field of view, at confidences of 0.73 and 0.71 against the deployed threshold of 0.70. They are instances of the out-of-distribution routing characterised in Section 4.7, and they document that the transport layer executes correctly and within the reported latency budget independently of whether the classification it carries is correct.
