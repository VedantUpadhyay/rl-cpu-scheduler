"""Filter trace splits to duration ≤ p75=47s, then re-run burst_ratio analysis."""
import csv
import sys
import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")

TRAIN_IN  = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/docs/scheduler-research/scripts/trace_train.csv"
TEST_IN   = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/docs/scheduler-research/scripts/trace_test.csv"
TRAIN_OUT = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
TEST_OUT  = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_test_filtered.csv"
P75_THRESHOLD = 47.0   # seconds, from prior analysis

# ---------------------------------------------------------------------------
# STEP 1 — Filter and save
# ---------------------------------------------------------------------------

def filter_csv(src, dst, threshold):
    kept = 0
    total = 0
    durations = []
    with open(src, newline="") as f_in, open(dst, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            total += 1
            try:
                d = int(row["end_time"]) - int(row["start_time"])
                if 0 < d <= threshold:
                    writer.writerow(row)
                    durations.append(float(d))
                    kept += 1
            except (ValueError, KeyError, TypeError):
                pass
    return total, kept, np.array(durations, dtype=np.float32)

print("Filtering training split ...")
train_total, train_kept, train_durs = filter_csv(TRAIN_IN, TRAIN_OUT, P75_THRESHOLD)
print(f"  Before: {train_total:,} rows")
print(f"  After : {train_kept:,} rows  ({100*train_kept/train_total:.1f}% retained)")

print("\nFiltering test split ...")
test_total, test_kept, test_durs = filter_csv(TEST_IN, TEST_OUT, P75_THRESHOLD)
print(f"  Before: {test_total:,} rows")
print(f"  After : {test_kept:,} rows  ({100*test_kept/test_total:.1f}% retained)")

print()
print("=== Filtered training duration distribution ===")
for label, pct in [("min",0),("p25",25),("p50",50),("p75",75),("p95",95),("max",100)]:
    v = np.percentile(train_durs, pct) if pct > 0 else train_durs.min()
    v = train_durs.max() if pct == 100 else v
    print(f"  {label:>4}: {v:.1f}s")

BURST_P95_new    = float(np.percentile(train_durs, 95))
BURST_MEDIAN_new = float(np.percentile(train_durs, 50))
tier0 = BURST_MEDIAN_new * 0.05
tier1 = BURST_MEDIAN_new * 0.20
tier2 = BURST_MEDIAN_new * 0.80

print()
print(f"  New BURST_P95    = {BURST_P95_new:.1f}s")
print(f"  New BURST_MEDIAN = {BURST_MEDIAN_new:.1f}s")
print(f"  New quantum tiers:")
print(f"    tier0 = {BURST_MEDIAN_new:.1f} × 0.05 = {tier0:.3f}s")
print(f"    tier1 = {BURST_MEDIAN_new:.1f} × 0.20 = {tier1:.3f}s")
print(f"    tier2 = {BURST_MEDIAN_new:.1f} × 0.80 = {tier2:.3f}s")

# ---------------------------------------------------------------------------
# STEP 2 — Burst ratio on filtered random episodes
# ---------------------------------------------------------------------------

print()
print("=== burst_ratio on 1,000 filtered random 5-task episodes ===")

N_PROC   = 5
N_SAMPLE = 1_000
rng = np.random.default_rng(42)

ratios = []
for _ in range(N_SAMPLE):
    indices = rng.integers(0, len(train_durs), size=N_PROC)
    d_ep    = train_durs[indices]
    ratios.append(float(d_ep.max()) / float(d_ep.min()))

ratios = np.array(ratios)
print(f"  p25 burst_ratio : {np.percentile(ratios, 25):.2f}x")
print(f"  p50 burst_ratio : {np.percentile(ratios, 50):.2f}x")
print(f"  p75 burst_ratio : {np.percentile(ratios, 75):.2f}x")
print(f"  p95 burst_ratio : {np.percentile(ratios, 95):.2f}x")
print(f"  % > 10x         : {100*np.mean(ratios > 10):.1f}%")
print(f"  % < 3x          : {100*np.mean(ratios < 3):.1f}%")

print()
print("=== Comparison ===")
print(f"  {'Method':<22} | {'p50 ratio':>10} | {'>10x%':>6} | {'<3x%':>5}")
print("  " + "-" * 52)
print(f"  {'Unfiltered random':<22} | {'74.00x':>10} | {'89.7%':>6} | {'0.4%':>5}")
p50_f  = np.percentile(ratios, 50)
gt10_f = 100*np.mean(ratios > 10)
lt3_f  = 100*np.mean(ratios < 3)
print(f"  {'Filtered random':<22} | {p50_f:>10.2f}x | {gt10_f:>5.1f}% | {lt3_f:>4.1f}%")
