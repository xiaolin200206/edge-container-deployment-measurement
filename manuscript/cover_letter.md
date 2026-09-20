Lin Ding Shan
Faculty of Computer Science and Data Science
UCSI University
Cheras 56000, Kuala Lumpur, Malaysia
1002475487@ucsiuniversity.edu.my

[Date]

The Editor-in-Chief
*Computers and Electrical Engineering*
Elsevier

**Re: Submission of an original research paper — "Deployment measurements of a containerised edge vision node on a Raspberry Pi 5: thermal, power and latency characteristics, and the attribution of container overhead"**

Dear Editor,

Please find enclosed our manuscript for consideration as an original research paper in *Computers and Electrical Engineering*.

**What the paper reports.** We instrumented a continuously operating vision inference node — a duty-cycled MobileNetV2 classifier under ONNX Runtime on a Raspberry Pi 5, deployed in a commercial greenhouse — and measured what containerising it actually costs. Four interleaved three-hour runs, instrumented per frame for latency, processor load and SoC temperature and once per second for node input power over I2C, give the trade in full: containerised execution was 45.6 % slower per inference in the mean, but ran 3.71 °C cooler at cyclic peak, 9.24 percentage points lighter on the processor, and drew 0.469 W less at the supply.

**Why the numbers do not mean what they appear to.** Three findings qualify that result, and they are the contribution:

1. **The power advantage does not survive normalisation.** The containerised configuration also completed 6.6 % fewer inferences per unit of active time. Normalised by work done, energy per inference was 2.3 % *higher*, not lower. A configuration that is slower at a fixed duty cycle will appear more power-efficient for that reason alone, and we argue that studies of this kind should report energy per unit of completed work alongside instantaneous power.

2. **The mean latency difference is not a tail difference.** At the 95th percentile the two conditions lie within 1 ms of each other. Since the end-to-end alert path of this class of node is gated by a multi-frame confirmation window rather than by a single inference, the trade is close to free in the quantity that actually governs response time — a conclusion the mean alone would not support.

3. **The comparison cannot be attributed to containerisation alone, and we say so.** The container carried a different interpreter and inference-runtime version from the host. We establish from package metadata that it could only have run ONNX Runtime 1.19.2 against the host's 1.29.0 — eighteen releases and some 23 months of ARM64 kernel optimisation apart. We therefore report the measurement as bounded, tabulate both software environments, and specify the three-condition factorial design that would decompose it.

We have chosen to foreground this last point rather than bury it, because we believe the problem generalises beyond our own measurement. A container image is a frozen dependency resolution; unless its versions are pinned to the host's, a container-versus-host comparison measures the dependency resolution as much as the isolation mechanism. On that argument, published overhead figures that do not report the interpreter and runtime versions of both sides are not portable between studies. We release a per-run environment-capture routine as the minimum instrumentation that would make them so.

**Fit to the journal.** The paper sits squarely in the integration of computer technology and computational techniques with electrical and information systems that defines the journal's scope: it combines direct electrical instrumentation of a deployed embedded node with the applied artificial intelligence running on it and the communication layer that carries its alerts. We submit it as a **case study**, an article type the journal explicitly accepts, and we note that its contribution is empirical and methodological rather than algorithmic — we propose no new architecture and make no claim to one. What we offer instead is a measurement that has been carried out, released in full, and reported with its limits made explicit.

**Reproducibility.** All per-frame telemetry, per-cycle event markers and the greenhouse image dataset are deposited at https://doi.org/10.5281/zenodo.22854130 under CC BY 4.0, and the analysis code, container definitions and edge application are at https://github.com/xiaolin200206/unified-agtech-engine. Every table and figure in the manuscript and its supplement is regenerated from the deposited logs by a single released script; no reported quantity exists outside that chain.

The manuscript is original, has not been published previously, and is not under consideration elsewhere. The author declares no competing interests.

We would be grateful for your consideration and look forward to the reviewers' comments.

Yours sincerely,

Lin Ding Shan
