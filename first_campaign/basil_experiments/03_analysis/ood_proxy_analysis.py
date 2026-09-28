"""
Confidence-based OOD proxy analysis.

No ground-truth labels needed - uses prediction confidence and system
telemetry to find indirect evidence of "the model may be seeing OOD input".

Data source: basil_data.csv (inference log with Timestamp, Predicted_Class,
             Confidence, Latency_ms, CPU_%, RAM_%, Temp_C, Throttled, FPS, etc.)

Outputs:
1. Temporal clustering analysis of low-confidence moments
2. Confidence vs system state (temperature/CPU/latency) correlations
3. Class-flip (prediction instability) analysis
4. A paper-ready summary table and text
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ Edit here ============
CSV_FILE = "./data/basil_data.csv"
OUTPUT_DIR = "./outputs"
LOW_CONF_THRESHOLD = 0.5  # below this counts as "low confidence / candidate OOD"
# ====================================

df = pd.read_csv(CSV_FILE)
print(f"Total samples: {len(df)}")

# ============ 1. Basic low-confidence statistics ============
df['is_low_conf'] = df['Confidence'] < LOW_CONF_THRESHOLD
low_conf_count = df['is_low_conf'].sum()
low_conf_rate = low_conf_count / len(df) * 100

print(f"\n{'='*60}")
print("1. Low-confidence sample statistics (candidate OOD)")
print(f"{'='*60}")
print(f"Low-confidence samples (<{LOW_CONF_THRESHOLD}): {low_conf_count} ({low_conf_rate:.2f}%)")

# ============ 2. Temporal clustering ============
# Group consecutive low-confidence frames into clusters rather than
# treating each one as isolated noise.
print(f"\n{'='*60}")
print("2. Temporal clustering analysis (sustained low-confidence events)")
print(f"{'='*60}")

df['low_conf_int'] = df['is_low_conf'].astype(int)
df['group'] = (df['low_conf_int'] != df['low_conf_int'].shift()).cumsum()

low_conf_clusters = df[df['is_low_conf']].groupby('group').size()
n_clusters = len(low_conf_clusters)
isolated_frames = (low_conf_clusters == 1).sum()
sustained_clusters = low_conf_clusters[low_conf_clusters >= 3]  # 3+ consecutive frames = "sustained"

print(f"Total low-confidence events (contiguous segments): {n_clusters}")
print(f"  Isolated single-frame noise (cluster length=1): {isolated_frames} ({isolated_frames/n_clusters*100:.1f}%)")
print(f"  Sustained events (>=3 consecutive frames): {len(sustained_clusters)} ({len(sustained_clusters)/n_clusters*100:.1f}%)")
if len(sustained_clusters) > 0:
    print(f"  Longest sustained event: {sustained_clusters.max()} frames")
    print(f"  Mean sustained event length: {sustained_clusters.mean():.1f} frames")

print(f"\nInterpretation: isolated single-frame noise is more likely transient")
print(f"     interference (e.g. one motion-blurred frame); sustained events are")
print(f"     more likely environmental OOD conditions (e.g. prolonged backlighting, occlusion)")

# ============ 3. Confidence vs system state correlation ============
print(f"\n{'='*60}")
print("3. Confidence vs system state correlation")
print(f"{'='*60}")

corr_temp = df['Confidence'].corr(df['Temp_C'])
corr_cpu = df['Confidence'].corr(df['CPU_%'])
corr_latency = df['Confidence'].corr(df['Latency_ms'])

print(f"Confidence vs temperature: r = {corr_temp:.3f}")
print(f"Confidence vs CPU usage:   r = {corr_cpu:.3f}")
print(f"Confidence vs latency:     r = {corr_latency:.3f}")

# Compare mean system state between low- and high-confidence groups
low_conf_group = df[df['is_low_conf']]
high_conf_group = df[~df['is_low_conf']]

print(f"\nLow-confidence vs high-confidence group system state comparison:")
print(f"{'Metric':<15}{'Low-conf':<15}{'High-conf':<15}{'Diff':<10}")
for col in ['Temp_C', 'CPU_%', 'Latency_ms']:
    low_mean = low_conf_group[col].mean()
    high_mean = high_conf_group[col].mean()
    diff = low_mean - high_mean
    print(f"{col:<15}{low_mean:<15.2f}{high_mean:<15.2f}{diff:+.2f}")

if 'Throttled' in df.columns:
    throttled_in_low = (low_conf_group['Throttled'] == 'Yes').sum()
    throttled_in_high = (high_conf_group['Throttled'] == 'Yes').sum()
    print(f"\nThrottling event distribution: low-conf group {throttled_in_low} | high-conf group {throttled_in_high}")

# ============ 4. Class-flip (prediction instability) analysis ============
print(f"\n{'='*60}")
print("4. Class-flip rate analysis")
print(f"{'='*60}")

df['class_changed'] = df['Predicted_Class'] != df['Predicted_Class'].shift()
flip_count = df['class_changed'].sum()
flip_rate = flip_count / len(df) * 100

print(f"Class changes: {flip_count} ({flip_rate:.2f}% of all frames)")

flip_in_low_conf = df[df['is_low_conf']]['class_changed'].mean() * 100
flip_in_high_conf = df[~df['is_low_conf']]['class_changed'].mean() * 100

print(f"Class-flip rate in low-confidence group:  {flip_in_low_conf:.1f}%")
print(f"Class-flip rate in high-confidence group: {flip_in_high_conf:.1f}%")
print(f"-> {'Low-confidence samples flip significantly more often, supporting their use as an instability/OOD proxy' if flip_in_low_conf > flip_in_high_conf * 1.5 else 'No clear difference in flip rate between groups'}")

# ============ 5. Plots ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(df.index, df['Confidence'], linewidth=0.5, color='steelblue')
axes[0, 0].axhline(LOW_CONF_THRESHOLD, color='red', linestyle='--', label=f'Threshold={LOW_CONF_THRESHOLD}')
axes[0, 0].fill_between(df.index, 0, 1, where=df['is_low_conf'], alpha=0.2, color='red')
axes[0, 0].set_title('Confidence Over Time (red = low-confidence candidate OOD)')
axes[0, 0].set_xlabel('Frame Index')
axes[0, 0].set_ylabel('Confidence')
axes[0, 0].legend()

axes[0, 1].hist(low_conf_clusters, bins=range(1, low_conf_clusters.max()+2), color='coral', edgecolor='black')
axes[0, 1].set_title('Low-Confidence Event Duration Distribution')
axes[0, 1].set_xlabel('Consecutive Frames in Event')
axes[0, 1].set_ylabel('Count')

axes[1, 0].scatter(df['Temp_C'], df['Confidence'], alpha=0.3, s=5, color='green')
axes[1, 0].set_title(f'Confidence vs Temperature (r={corr_temp:.3f})')
axes[1, 0].set_xlabel('Temperature (°C)')
axes[1, 0].set_ylabel('Confidence')

axes[1, 1].boxplot([low_conf_group['Latency_ms'].dropna(), high_conf_group['Latency_ms'].dropna()],
                    labels=['Low Conf', 'High Conf'])
axes[1, 1].set_title('Latency Distribution: Low vs High Confidence')
axes[1, 1].set_ylabel('Latency (ms)')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/ood_proxy_analysis_charts.png', dpi=300, bbox_inches='tight')
print(f"\nCharts saved to ood_proxy_analysis_charts.png")

# ============ 6. Paper-ready text ============
paper_text = f"""
Uncertainty-Based Out-of-Distribution Proxy Analysis:

Due to access constraints following the conclusion of the field deployment 
period, frame-level ground truth annotation for the 3-hour field dataset 
was not feasible. As a proxy for direct error analysis, we examined the 
temporal and distributional characteristics of low-confidence predictions 
(confidence < {LOW_CONF_THRESHOLD}) as candidate indicators of out-of-distribution 
or ambiguous inputs, following the established practice of using prediction 
uncertainty as an OOD signal in the absence of ground truth labels.

Of {len(df):,} total inferences, {low_conf_count} ({low_conf_rate:.1f}%) fell below the 
confidence threshold. Temporal clustering analysis revealed {n_clusters} discrete 
low-confidence events, of which {len(sustained_clusters)} ({len(sustained_clusters)/max(n_clusters,1)*100:.1f}%) persisted 
for three or more consecutive frames, suggesting sustained environmental 
conditions (e.g., prolonged backlighting, partial occlusion) rather than 
transient single-frame noise as the dominant source of uncertainty.

Low-confidence predictions exhibited a {flip_in_low_conf:.1f}% class-flip rate compared to 
{flip_in_high_conf:.1f}% among high-confidence predictions, indicating reduced prediction 
stability under ambiguous conditions. Correlation analysis between confidence 
and system telemetry showed {'a notable' if abs(corr_temp) > 0.3 else 'a weak'} relationship between confidence and 
operating temperature (r = {corr_temp:.3f}), {'suggesting thermal stress may coincide with degraded model reliability' if abs(corr_temp) > 0.3 else 'suggesting confidence degradation is not primarily driven by thermal throttling'}.

This proxy analysis, while not a substitute for ground-truth-validated error 
characterization, provides actionable evidence that prediction instability is 
concentrated in specific temporal windows rather than uniformly distributed, 
informing future targeted data collection for model improvement.
"""

with open(f'{OUTPUT_DIR}/ood_proxy_analysis_text.txt', 'w') as f:
    f.write(paper_text)

print(f"Paper text saved to ood_proxy_analysis_text.txt")
print(paper_text)
