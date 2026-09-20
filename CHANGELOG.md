# Corrections and additions

This release revises the manuscript materials. Several corrections affect claims made in the
previous version; each is listed with the evidence that prompted it. Raw telemetry is unchanged —
no logged value has been altered. What changed is what is claimed from it, and what is now
computed from it.

Every quantity stated in the revised manuscript is emitted by `analysis/recompute.py`, which reads
only the deposited logs. Nothing is reported that does not appear in `analysis/numbers.json`.

---

## 1. The bare-metal versus container comparison is no longer attributed to containerisation alone

**What was claimed.** The previous manuscript stated, in the profiling protocol section, that
"the only difference between conditions was whether the script executed natively or inside the
container". The repository README repeated it.

**Why it was withdrawn.** The `RUN_INFO.txt` files deposited with the four runs record that the
native runs executed under Python 3.13 with onnxruntime 1.29.0 on Debian 13 (glibc 2.41), while
the containerised runs executed under `python:3.9-slim-bullseye` — Python 3.9, Debian 11, glibc
2.31 — with onnxruntime installed from an **unpinned** requirement in `Dockerfile`. Execution
mode, interpreter version, inference-runtime version and base image therefore co-vary, and the
measured difference is not attributable to any one of them.

**The size of the gap is determined, not estimated.** The version pip resolved was never recorded
and the storage media have since been reimaged, but package metadata fixes it exactly. The last
onnxruntime release publishing a `cp39` `aarch64` Linux wheel is **1.19.2** (uploaded 2024-09-04);
no later release publishes one at all, and its `manylinux_2_27` floor is satisfied by Debian 11's
glibc 2.31. Exactly one release therefore fits, so the containerised runs ran 1.19.2. The host ran
1.29.0, released 2026-08-17 — the day before the first profiling run. **Eighteen intervening
releases and approximately 23 months separate them.** `Dockerfile.legacy` pins 1.19.2 on this
basis and records the reasoning in its header.

**What the revision says instead.** The measured differences are reported unchanged, as properties
of the two configurations as built. The attribution bound is stated explicitly (main text Section
4.3), the software environment of both conditions is tabulated (Table 1), and the three-condition
design that would decompose the result is specified (Fig. 4b and Supplementary S8). The bound is
treated as a methodological contribution rather than only as a limitation: a container image is a
frozen dependency resolution, and an overhead figure reported without the runtime versions of both
sides is not portable between studies.

## 2. The proposed mechanism for the thermal and CPU differences was incorrect and has been removed

**What was claimed.** That the reduced processor load and temperature arose because the minimal
`python-slim` image "removes operating-system background services that are present on the
general-purpose host".

**Why it was withdrawn.** A container shares the host kernel. Host daemons continue to run
regardless of the contents of the container image, so a slim image cannot remove them. The claim
is not supported by any mechanism.

**What the revision says instead.** No mechanism is asserted. Three observations consistent with a
runtime-attributable component are reported — the containerised condition was simultaneously
slower and *lower* in processor utilisation, its latency distribution was tighter, and memory
overhead was negligible — and a threading probe that would test the hypothesis directly is
specified in Supplementary S8.

## 3. The power result is reported as an absolute difference, and normalised by work completed

**What is new.** Two quantities absent from the previous version are now computed from the
existing logs:

- **Effective throughput.** Over an identical 8,166 s of active inference, the containerised
  condition completed 6.6 % fewer inferences (12.69 against 11.85 frame s⁻¹).
- **Energy per inference.** 0.804 J native against 0.822 J containerised — **2.3 % higher**, not
  lower.

**Consequence.** The previously reported 4.6 % power reduction reflects reduced work rather than
improved efficiency, and does not survive normalisation. The revision reports the absolute
difference (−0.469 W) as primary and states that per-inference energy is not detectably different
between conditions.

**Measurement node identified precisely, and the quantity renamed.** The previous version
described the power figures only as board-level. They are not. The UPS HAT (E) aggregates two
measurement devices behind one microcontroller: the `Bus_*` registers come from a bidirectional
USB Power Delivery buck-boost controller and report the **Type-C connector** — the total DC power
the Pi and the HAT together draw from the mains adapter — while the `Bat_*` registers come from a
four-series lithium-ion fuel gauge. **No register reports the 5 V rail feeding the board.**

This explains two readings that otherwise look anomalous. The 15.29 V bus is a negotiated USB-PD
15 V contract, which is why it is stable to 0.17 %; the 16.79 V pack is 4.198 V per cell across
four series cells. The 1.5 V difference between them is the conversion step, not an error.

The quantity is therefore renamed **node input power** throughout. At 80–90 % end-to-end
efficiency across the two cascaded conversion stages it corresponds to roughly 8–9 W at the
board's 5 V rail — an ordinary figure for a Pi 5 under sustained inference with active cooling,
and higher than a naive 5–7 W assumption would suggest. That efficiency is assumed, not measured.

**Note on the column naming.** Waveshare's documentation for this module carries a section heading
referring to an INA219, inherited from the (B)/(C)/(D) modules in which the monitored bus genuinely
*is* the battery. This module contains no INA219. The `Bus_*` naming in the deposited logs comes
from that lineage and is the likely origin of the misreading.

**Instrument bounds added.** The PD controller is a charge-management device, not a precision
metrology part, and its telemetry resolution and accuracy are undocumented; absolute values are
reported as uncalibrated. Bus *current* was not logged alongside bus power, so flow direction is
inferred from the configuration rather than read. Reported state of charge is a model-based fuel
gauge estimate and is not calibrated. Between-condition differences are unaffected, both conditions
having been measured through the identical chain.

**Supply validated.** The charging input was connected throughout every run, so a validation of
the bus-power measurement is now reported: across all four runs the battery moved only from 93 %
to 94 %, no charge cycle occurred, battery current averaged −1.5 mA (range −15 to +20 mA, at most
0.34 W instantaneous), and bus power showed no meaningful correlation with battery current
(r between −0.04 and −0.09). Bus voltage held at 15.29 V ± 0.17 %.

## 4. Latency is reported as a distribution, not only as a mean

**What is new.** The 45.6 % difference in mean latency is not reproduced in the tail. At the 95th
percentile the two conditions lie within 1 ms of one another (34.2 ms native, 35.2 ms
containerised); the native distribution is faster typically but substantially wider (SD 6.2 ms
against 3.9 ms). Median, p95 and p99 are now reported per run (Supplementary S1).

**Consequence.** The argument that the latency cost is acceptable for a debounced alerting path is
now supported by the quantity that actually governs confirmation time, rather than by the mean.

## 5. The architecture comparison is restated as an absence of detectable effect

**What was claimed.** That a fixed random seed (42) governed the comparison, and that the
direction of the effect across eleven architectures was informative — three supporting, three
neutral, five reversing.

**Why it was withdrawn.** `basil_experiments/02_baseline_comparison/results/RUN_INFO.txt` records
that the 80/20 split was drawn "WITHOUT a fixed generator seed", so the two conditions were
evaluated on **different validation partitions**, and the exact indices are not recoverable. Each
architecture was trained once per condition, with no seed repetition and therefore no variance
estimate. The validation partition contains approximately 340 images, so one image is 0.294
percentage points.

**What the revision says instead.** Differences are expressed in units of validation images, where
they resolve to whole numbers: nine of the eleven architectures fall between +3 and −4 images of
zero, and seven within ±2. Under a protocol that fixes neither the split nor the seed, no effect is
distinguishable from procedural variability in either direction. The conclusion is stated as an
absence of detectable effect, not as evidence of equivalence, and the seeded repeated comparison
that would resolve it is identified as future work.

**Relabelling.** The row recorded as *ViT-Tiny* in the released result files is `torchvision`'s
`vit_b_16` (85.8 M parameters), as noted in the same `RUN_INFO.txt`. It is reported as ViT-B/16
throughout. SqueezeNet (−29.70 pp) and ViT-B/16 (−7.35 pp) are treated as runs in which one
condition failed to converge, not as data effects.

## 6. The out-of-distribution result is split into two claims of different evidential strength

**What was claimed.** A "large-sample characterisation" of a residual-class failure mode "across
313,011 frames", establishing that the Disease class functions as a residual category for novel
input.

**Why it was withdrawn.** The 30 non-Background frames in that session are not 30 independent
observations. Grouped at a one-second separation they resolve into **four intrusion events** of
19, 2, 1 and 8 frames — and all four fall between frame 87 and frame 1,327, a **43-second window
at the very start** of a 3 h 33 min session. The effective sample size for a claim about class
routing is four, not 313,011.

**What the revision says instead.** Two separate claims:

- **Large-sample, strong.** The remaining 311,683 consecutive frames — 3 h 32 min of a quiescent
  scene — produced no non-Background prediction of any kind, at a threshold more permissive than
  the deployed one. The node does not generate spurious alerts when the scene is static.
- **Small-sample, suggestive.** Four intrusion events were all assigned to Disease and none to
  Healthy (p = 0.0625 under a random-routing null). This is reported as an observation consistent
  with residual-region geometry, explicitly not as an established structural property, and the
  controlled probe that would test it is specified.

## 7. Alerting-filter figures recomputed

Raw-transition counts are now reported per threshold, because the confidence gate alters the raw
class sequence and the baseline is therefore not common across thresholds (624 raw transitions at
τ = 0.7 against 310 at τ = 0.3). The deployed configuration reduces 624 raw transitions to 58
confirmed ones, a 90.7 % suppression. Minor numerical differences from the previously reported
table (for example 151 rather than 152 confirmed transitions at τ = 0.3, W = 3, and a 50.0 %
rather than 51.1 % low-confidence flip rate) arise from recomputation against the released log and
are the values the released script now reproduces.

## 8. Statistical unit of analysis corrected

Per-run means are computed first, and condition-level quantities are the mean of two run means
with the half-range as dispersion. Frames within a run share thermal state, illumination and scene
content and are strongly autocorrelated; treating ~100,000 frames as independent observations
would be pseudoreplication. The effective sample size is **two per condition**, which supports a
statement about effect direction and replicate agreement but not significance testing. None is
claimed.

---

## 9. Figure set revised

A new **Fig. 1** presents the node architecture and measurement chain as a block diagram,
replacing the deployment photograph of the previous version. It makes visible what Section 3.4
argues in prose: which nodes of the power chain the supply module instruments, and that the 5 V
rail feeding the board is not among them. For an electrical-engineering readership this carries
more information than a photograph of the enclosure.

All remaining figures are renumbered accordingly, and every data figure is regenerated from the
deposited logs. The dataset figure is produced by a new script, `make_dataset_figure.py`, with
deterministic sampling from the class archives, and takes the proxy column as optional so that it
can be omitted where redistribution rights do not permit it. The illustrative out-of-distribution
photographs of the previous version are not reinstated: the event-level analysis in Fig. 8 carries
that claim, and the photographs were the weaker evidence that supported the overstatement
corrected in §6 above.

## 10. Reference list restored and verified

The intermediate draft of this revision rewrote the related-work sections as prose and lost the
citations and reference list in the process. They are restored: 38 entries, every one cited in the
text and every in-text citation resolving to an entry.

Seven references are new to this revision, covering the container-measurement literature the
attribution argument engages with, the statistical-unit argument of Section 3.5, and the inference
runtime itself. Three had their bibliographic details verified against Crossref and the publisher
of record after an initial draft attributed them incorrectly:

- *A qualitative and quantitative analysis of container engines* is by **Baresi, Quattrocchi and
  Rasi** (J. Syst. Softw. 210, 111965, 2024).
- *Impact of thermal throttling on long-term visual inference in a CPU-based edge device* is by
  **Benoit-Cattin, Velasco-Montero and Fernández-Berni**, and was published in *Electronics*
  9 (12), 2106, 2020 — not only as the arXiv preprint (arXiv:2010.06291) it is often cited as.
- *How does docker affect energy consumption?* is by **Santos, McLean, Solinas and Hindle**
  (J. Syst. Softw. 146, 14–25, 2018); note the lowercase "docker" in the published title.

## New artefacts in this release

| File | Purpose |
|---|---|
| `provenance.py` | Per-run environment manifest: execution mode, interpreter, onnxruntime, glibc, NumPy, OpenCV, resolved ORT thread counts, cgroup CPU quota, CPU governor, model SHA-256, throttling status. The instrumentation whose absence caused correction 1. |
| `Dockerfile.matched` | Condition B — container with every dependency pinned to the host's versions, so that B − A isolates containerisation. |
| `Dockerfile.legacy` | Condition C — the as-measured image, with the versions now pinned explicitly so the condition is reproducible. |
| `analysis/recompute.py` | Emits every quantity stated in the manuscript, from the deposited logs only. |
| `analysis/fig_system.py` | Fig. 1, the architecture and measurement-chain diagram. |
| `analysis/fig_profiling.py`, `analysis/fig_analysis.py`, `analysis/figstyle.py` | Regenerate the eight data figures. |
| `analysis/make_dataset_figure.py` | Fig. 9, from the deposited image archives; proxy column optional. |
| `analysis/numbers.json` | The authoritative value of every reported quantity. |
| `manuscript/` | Revised manuscript and supplementary material. |

The original `Dockerfile` is retained unchanged as the historical artefact of the measured runs.
`scripts/analyse_profiling.py` and `basil_experiments/03_analysis/` are retained; where their
output differs from `analysis/recompute.py`, the latter is authoritative for the revised
manuscript.
