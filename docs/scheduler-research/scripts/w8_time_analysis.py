"""Step 1: Analyze trace time structure — inter-arrival times and window density."""
import csv
import sys
import numpy as np

TRAIN_PATH = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/docs/scheduler-research/scripts/trace_train.csv"

print(f"Loading {TRAIN_PATH} ...")
start_times = []

with open(TRAIN_PATH, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            st = int(row["start_time"])
            et = int(row["end_time"])
            if et > st:           # valid-duration tasks only
                start_times.append(st)
        except (ValueError, KeyError, TypeError):
            pass

start_times = np.array(start_times, dtype=np.int64)
start_times.sort()
n = len(start_times)
print(f"Valid-duration tasks loaded: {n:,}")
print()

# Column used
print("=== Column used for task arrival time ===")
print("  Column: start_time")
print(f"  Range : {start_times[0]:,}s  to  {start_times[-1]:,}s")
print(f"  Span  : {(start_times[-1] - start_times[0]):,}s  ≈  {(start_times[-1] - start_times[0])/3600:.1f} hours")
print()

# Inter-arrival times
print("=== Inter-arrival times (consecutive tasks by start_time) ===")
iats = np.diff(start_times)   # n-1 values
print(f"  Count : {len(iats):,}")
print(f"  Min   : {iats.min():.0f}s")
print(f"  p5    : {np.percentile(iats, 5):.2f}s")
print(f"  p25   : {np.percentile(iats, 25):.4f}s")
print(f"  p50   : {np.percentile(iats, 50):.4f}s")
print(f"  p75   : {np.percentile(iats, 75):.4f}s")
print(f"  p95   : {np.percentile(iats, 95):.2f}s")
print(f"  Max   : {iats.max():.0f}s")
print(f"  % == 0: {100*np.sum(iats==0)/len(iats):.1f}%  (tasks with identical start_time)")
print()

# Tasks per time window
trace_span = int(start_times[-1] - start_times[0])

for window in [10, 60]:
    # Sample 10,000 random windows and count tasks in each
    rng = np.random.default_rng(42)
    sample_starts = rng.integers(start_times[0], start_times[-1] - window, size=10_000)
    counts = []
    for T in sample_starts:
        lo = np.searchsorted(start_times, T, side="left")
        hi = np.searchsorted(start_times, T + window, side="right")
        counts.append(hi - lo)
    counts = np.array(counts)
    print(f"=== Tasks in random {window}s windows (n=10,000 samples) ===")
    print(f"  Mean  : {counts.mean():.2f}")
    print(f"  Median: {np.median(counts):.0f}")
    print(f"  p25   : {np.percentile(counts, 25):.0f}")
    print(f"  p75   : {np.percentile(counts, 75):.0f}")
    print(f"  p95   : {np.percentile(counts, 95):.0f}")
    print(f"  % with ≥5  tasks: {100*np.mean(counts >= 5):.1f}%")
    print(f"  % with ≥8  tasks: {100*np.mean(counts >= 8):.1f}%")
    print(f"  % with ≥15 tasks: {100*np.mean(counts >= 15):.1f}%")
    print()

# Also check 30s windows
for window in [30, 120, 300]:
    rng = np.random.default_rng(42)
    sample_starts = rng.integers(start_times[0], start_times[-1] - window, size=10_000)
    counts = []
    for T in sample_starts:
        lo = np.searchsorted(start_times, T, side="left")
        hi = np.searchsorted(start_times, T + window, side="right")
        counts.append(hi - lo)
    counts = np.array(counts)
    print(f"=== Tasks in random {window}s windows (n=10,000 samples) ===")
    print(f"  Mean  : {counts.mean():.2f}")
    print(f"  % with ≥8  tasks: {100*np.mean(counts >= 8):.1f}%")
    print(f"  % with ≥15 tasks: {100*np.mean(counts >= 15):.1f}%")
    print()
