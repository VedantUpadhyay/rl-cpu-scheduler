"""Prepare Alibaba 2017 batch_task trace: add headers, shuffle, split 80/20."""
import csv, random

SRC   = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/batch_task.csv"
TRAIN = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/docs/scheduler-research/scripts/trace_train.csv"
TEST  = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/docs/scheduler-research/scripts/trace_test.csv"
COLS  = ["start_time", "end_time", "task_type", "inst_num",
         "task_name", "status", "plan_cpu", "plan_mem"]

print(f"Reading {SRC} ...")
rows = []
with open(SRC, newline="") as f:
    for line in f:
        rows.append(line.rstrip("\n"))

print(f"Total rows: {len(rows):,}")
random.seed(42)
random.shuffle(rows)

split = int(len(rows) * 0.8)
train_rows = rows[:split]
test_rows  = rows[split:]

header = ",".join(COLS)

with open(TRAIN, "w") as f:
    f.write(header + "\n")
    for r in train_rows:
        f.write(r + "\n")

with open(TEST, "w") as f:
    f.write(header + "\n")
    for r in test_rows:
        f.write(r + "\n")

print(f"Train: {len(train_rows):,} rows → {TRAIN}")
print(f"Test : {len(test_rows):,} rows  → {TEST}")
print("Done.")
