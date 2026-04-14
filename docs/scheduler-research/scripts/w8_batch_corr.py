"""Step 2a: Within-batch burst correlation — 1-second windows vs random sampling."""
import csv
import sys
import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")
from schedsim.trace_sampler import TraceEpisodeSampler

TRAIN_PATH = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/docs/scheduler-research/scripts/trace_train.csv"

print(f"Loading {TRAIN_PATH} ...")
# Load all (start_time, duration) pairs
start_times = []
durations   = []

with open(TRAIN_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            st = int(row["start_time"])
            d  = int(row["end_time"]) - st
            if d > 0:
                start_times.append(st)
                durations.append(d)
        except (ValueError, KeyError, TypeError):
            pass

start_times = np.array(start_times, dtype=np.int64)
durations   = np.array(durations,   dtype=np.float32)
sort_idx    = np.argsort(start_times, kind="stable")
start_times = start_times[sort_idx]
durations   = durations[sort_idx]
n = len(start_times)
print(f"Loaded {n:,} valid-duration tasks.\n")

rng = np.random.default_rng(42)

# -----------------------------------------------------------------------
# 1-second window sampling
# -----------------------------------------------------------------------
print("=== 1-second time-window burst_ratio analysis (n=1,000 windows) ===")
WINDOW   = 1        # seconds
MIN_TASKS = 5
N_SAMPLE  = 1_000

T_min = int(start_times[0])
T_max = int(start_times[-1]) - WINDOW

ratios_window = []
attempts = 0
while len(ratios_window) < N_SAMPLE:
    T = int(rng.integers(T_min, T_max))
    lo = int(np.searchsorted(start_times, T,          side="left"))
    hi = int(np.searchsorted(start_times, T + WINDOW, side="right"))
    if hi - lo >= MIN_TASKS:
        d_win = durations[lo:hi]
        ratio = float(d_win.max()) / float(d_win.min())
        ratios_window.append(ratio)
    attempts += 1

ratios_window = np.array(ratios_window)
print(f"  Attempts to get {N_SAMPLE} valid windows: {attempts:,}")
print(f"  p25 burst_ratio : {np.percentile(ratios_window, 25):.2f}x")
print(f"  p50 burst_ratio : {np.percentile(ratios_window, 50):.2f}x")
print(f"  p75 burst_ratio : {np.percentile(ratios_window, 75):.2f}x")
print(f"  p95 burst_ratio : {np.percentile(ratios_window, 95):.2f}x")
print(f"  % > 10x         : {100*np.mean(ratios_window > 10):.1f}%")
print(f"  % < 3x          : {100*np.mean(ratios_window < 3):.1f}%")
print()

# -----------------------------------------------------------------------
# Random 5-task sampling
# -----------------------------------------------------------------------
print("=== Random 5-task sampling burst_ratio analysis (n=1,000 episodes) ===")
N_PROC = 5
sampler = TraceEpisodeSampler(TRAIN_PATH)
all_bursts = sampler._bursts  # float32 array

ratios_random = []
rng2 = np.random.default_rng(43)
for _ in range(N_SAMPLE):
    indices = rng2.integers(0, len(all_bursts), size=N_PROC)
    d_ep    = all_bursts[indices]
    ratio   = float(d_ep.max()) / float(d_ep.min())
    ratios_random.append(ratio)

ratios_random = np.array(ratios_random)
print(f"  p25 burst_ratio : {np.percentile(ratios_random, 25):.2f}x")
print(f"  p50 burst_ratio : {np.percentile(ratios_random, 50):.2f}x")
print(f"  p75 burst_ratio : {np.percentile(ratios_random, 75):.2f}x")
print(f"  p95 burst_ratio : {np.percentile(ratios_random, 95):.2f}x")
print(f"  % > 10x         : {100*np.mean(ratios_random > 10):.1f}%")
print(f"  % < 3x          : {100*np.mean(ratios_random < 3):.1f}%")
print()

# -----------------------------------------------------------------------
# Summary table
# -----------------------------------------------------------------------
print("=== Final comparison ===")
print(f"  {'Method':<22} | {'p50 ratio':>10} | {'>10x%':>6} | {'<3x%':>5}")
print("  " + "-" * 52)
p50_r = np.percentile(ratios_random, 50)
p50_w = np.percentile(ratios_window, 50)
gt10_r = 100*np.mean(ratios_random > 10)
gt10_w = 100*np.mean(ratios_window > 10)
lt3_r  = 100*np.mean(ratios_random < 3)
lt3_w  = 100*np.mean(ratios_window < 3)
print(f"  {'Random 5-task':<22} | {p50_r:>10.2f}x | {gt10_r:>5.1f}% | {lt3_r:>4.1f}%")
print(f"  {'1-second window':<22} | {p50_w:>10.2f}x | {gt10_w:>5.1f}% | {lt3_w:>4.1f}%")
