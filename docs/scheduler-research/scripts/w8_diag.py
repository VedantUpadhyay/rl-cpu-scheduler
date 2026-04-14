"""Diagnostic: burst duration distribution shape for trace training split."""
import sys
import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")

from schedsim.trace_sampler import TraceEpisodeSampler
from schedsim.env import BURST_P95

TRAIN_PATH = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/docs/scheduler-research/scripts/trace_train.csv"

print("Loading training sampler ...")
sampler = TraceEpisodeSampler(TRAIN_PATH)
bursts = sampler._bursts  # raw float32 array
n = len(bursts)
print(f"Total valid-duration tasks: {n:,}")
print()

# 1. Bucket distribution
p25 = float(np.percentile(bursts, 25))
p50 = float(np.percentile(bursts, 50))
p75 = float(np.percentile(bursts, 75))
p95 = float(np.percentile(bursts, 95))
p99 = float(np.percentile(bursts, 99))
bmax = float(bursts.max())

buckets = [
    (0,   p25,  "[0, p25)"),
    (p25, p50,  "[p25, p50)"),
    (p50, p75,  "[p50, p75)"),
    (p75, p95,  "[p75, p95)"),
    (p95, p99,  "[p95, p99)"),
    (p99, bmax, "[p99, max]"),
]

print("=== 1. Burst duration distribution (training split) ===")
print(f"  p25={p25:.1f}s  p50={p50:.1f}s  p75={p75:.1f}s  p95={p95:.1f}s  p99={p99:.1f}s  max={bmax:.1f}s")
print()
print(f"  {'Bucket':<15}  {'Count':>10}  {'%':>7}  {'Range':>20}")
print("  " + "-" * 58)

prev_lo = 0.0
for lo, hi, label in buckets:
    if label == "[p99, max]":
        count = int(np.sum(bursts >= lo))
    else:
        count = int(np.sum((bursts >= lo) & (bursts < hi)))
    pct = 100.0 * count / n
    print(f"  {label:<15}  {count:>10,}  {pct:>6.2f}%  [{lo:.1f}s, {hi:.1f}s)")

print()

# 2. % of 1000-episode samples containing burst > 0.90 normalized
print("=== 2. Episodes with at least one normalized burst > 0.90 (n=1000) ===")
rng = np.random.default_rng(42)
clip_ceiling = 0.90 * BURST_P95
n_episodes = 1000
n_above = 0
for _ in range(n_episodes):
    ep = sampler.sample_episode(rng)
    if any(t["burst_ms"] > clip_ceiling for t in ep):
        n_above += 1
pct_above = 100.0 * n_above / n_episodes
print(f"  Clip ceiling (0.90 * BURST_P95): {clip_ceiling:.1f}s")
print(f"  Episodes with ≥1 burst > 0.90 norm: {n_above}/{n_episodes}  ({pct_above:.1f}%)")
print()

# 3. p95/p50 ratio
ratio = p95 / p50
print("=== 3. Heavy-tail ratio: p95 / p50 ===")
print(f"  p95 = {p95:.1f}s")
print(f"  p50 = {p50:.1f}s")
print(f"  p95/p50 = {ratio:.2f}x")
