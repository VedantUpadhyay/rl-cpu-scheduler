"""TraceEpisodeSampler — sample 5-task episodes from Alibaba 2018 batch_task trace.

Replaces the synthetic burst sampler (U[1, 60]ms) with real cluster task durations.

Design
------
- Loads from a CSV file (trace_train.csv or trace_test.csv) written by the
  data-preparation step.  Expects a header row with columns:
      task_name, instance_num, job_name, task_type, status,
      start_time, end_time, plan_cpu, plan_mem
- Filters to rows where end_time > start_time (duration > 0).
- Stores raw durations as float32 numpy array (in seconds, matching trace units).
- sample_episode(rng) draws 5 tasks independently at random and assigns
  them synthetic arrival slots {0, 2, 5, 8, 10} (shuffled per episode).

Units
-----
- burst_ms: raw trace duration in seconds (despite the "_ms" naming convention
  inherited from the toy environment; the env's time unit has shifted to seconds).
- arrival_ms: synthetic arrival offsets in the same unit as burst_ms.
- Normalisation (burst / BURST_P95) is applied by the environment, not here.

Replacement note
----------------
  If fewer than 5 valid-duration rows exist in the loaded split,
  sample_episode samples with replacement.
"""
from __future__ import annotations

import csv

import numpy as np


ARRIVAL_SLOTS: tuple[float, ...] = (0.0, 2.0, 5.0, 8.0, 10.0)
N_PROC = 5


class TraceEpisodeSampler:
    """Sample 5-task episodes from a pre-split Alibaba batch_task CSV."""

    def __init__(self, csv_path: str) -> None:
        self._path   = csv_path
        self._bursts = self._load(csv_path)   # float32 array of raw durations (s)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample_episode(self, rng: np.random.Generator) -> list[dict]:
        """Return a list of 5 task dicts with keys arrival_ms and burst_ms.

        Parameters
        ----------
        rng : numpy.random.Generator
            Caller-owned RNG (numpy new-style Generator, e.g. np.random.default_rng(42)).

        Returns
        -------
        list of 5 dicts, each with:
            arrival_ms : float — synthetic arrival offset (s)
            burst_ms   : float — raw task duration from trace (s)
        """
        n = len(self._bursts)
        indices  = rng.integers(0, n, size=N_PROC)
        bursts   = self._bursts[indices]
        arrivals = rng.permutation(np.array(ARRIVAL_SLOTS, dtype=np.float64))
        return [
            {"arrival_ms": float(arr), "burst_ms": float(bst)}
            for arr, bst in zip(arrivals, bursts)
        ]

    def __len__(self) -> int:
        return int(len(self._bursts))

    def __repr__(self) -> str:
        return f"TraceEpisodeSampler(path={self._path!r}, n_tasks={len(self)})"

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _load(self, path: str) -> np.ndarray:
        """Load CSV and return float32 array of positive-duration values."""
        bursts: list[float] = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    d = int(row["end_time"]) - int(row["start_time"])
                    if d > 0:
                        bursts.append(float(d))
                except (ValueError, KeyError, TypeError):
                    pass
        if not bursts:
            raise ValueError(
                f"No valid (duration > 0) rows found in {path!r}. "
                "Check that the file has a header row and end_time/start_time columns."
            )
        return np.array(bursts, dtype=np.float32)


# ---------------------------------------------------------------------------
# Standalone demo / unit test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    TRAIN_PATH = "/tmp/trace_train.csv"

    print(f"Loading {TRAIN_PATH} ...")
    try:
        sampler = TraceEpisodeSampler(TRAIN_PATH)
    except FileNotFoundError:
        print(f"ERROR: {TRAIN_PATH} not found. Run the data-prep step first.")
        sys.exit(1)

    print(f"Loaded {len(sampler):,} valid-duration tasks from training split.")
    print()

    rng = np.random.default_rng(42)
    ep  = sampler.sample_episode(rng)

    print("Sample episode (5 tasks):")
    print(f"{'PID':>4}  {'arrival_ms':>12}  {'burst_ms':>12}")
    print("-" * 34)
    for i, t in enumerate(ep):
        print(f"  P{i}  {t['arrival_ms']:>12.1f}  {t['burst_ms']:>12.1f}")

    all_pos = all(t["burst_ms"] > 0 for t in ep)
    print()
    print(f"All burst_ms > 0: {all_pos}")
    assert all_pos, "FAIL: found burst_ms <= 0"
    print("Unit test PASSED.")
