"""Baseline evaluation for N-scaling experiment.

Evaluates MLFQ, Round Robin, and CFS-lite at N ∈ {5, 10, 20, 50}
using 500 episodes per seed × 5 seeds = 2500 episodes per (N, policy).

Results are saved to:
  docs/scheduler-research/results/n_scaling/baselines.json

VA-DQN reference numbers are read from summary.md in the same directory
and compared in the printed table.

Usage:
  python eval_baselines_n_scaling.py
  python eval_baselines_n_scaling.py --trace_path /data/alibaba2018/
  python eval_baselines_n_scaling.py --n_list 5          # smoke test
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJ    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJ)
sys.path.insert(0, _SCRIPTS)

from project_config import TRACE_PATH as _CFG_TRACE, get_agent_dir
from schedsim.env    import SchedEnv, N_QUANTUM_TIERS
from schedsim.process import Process
from w9_train        import _valid_actions, _make_procs, N_QT

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline N-scaling evaluation")
    p.add_argument("--trace_path", type=str, default=None,
                   help="Trace file or directory (default: project_config.TRACE_PATH)")
    p.add_argument("--n_list", type=int, nargs="+", default=[5, 10, 20, 50],
                   help="N values to evaluate (default: 5 10 20 50)")
    p.add_argument("--n_eval", type=int, default=500,
                   help="Episodes per seed (default: 500)")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 999],
                   help="Seeds (default: 42 123 456 789 999)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Trace sampler (replicates TraceEpisodeSamplerN from train.py)
# ---------------------------------------------------------------------------

_BURST_P95_FILT = 36.0


class TraceEpisodeSamplerN:
    """Sample N-task episodes for any N >= 1. Arrival slots: 2*i seconds."""

    def __init__(self, trace_file: str, n_processes: int) -> None:
        self.n_processes   = n_processes
        self._arrive_slots = tuple(i * 2 for i in range(n_processes))
        records = []
        with open(trace_file, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    dur = float(row["end_time"]) - float(row["start_time"])
                    cpu = float(row["plan_cpu"])
                    mem = float(row["plan_mem"])
                    if 0 < dur <= _BURST_P95_FILT * 1.1:
                        records.append((dur, cpu, mem))
                except (ValueError, TypeError, KeyError):
                    pass
        self._data = np.array(records, dtype=np.float32)

    def sample_episode(self, rng: np.random.Generator) -> list[dict]:
        idx   = rng.integers(0, len(self._data), size=self.n_processes)
        tasks = self._data[idx]
        order = rng.permutation(self.n_processes)
        slots = self._arrive_slots
        return [
            {
                "burst_ms":   float(tasks[i, 0]),
                "arrival_ms": float(slots[order[k]]),
                "plan_cpu":   float(tasks[i, 1]),
                "plan_mem":   float(tasks[i, 2]),
            }
            for k, i in enumerate(range(self.n_processes))
        ]


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

QUANTUM_TIERS    = (0.5, 2.0, 8.0)
MLFQ_AGE_THRESH  = 50.0   # seconds without execution → promote to Q0


def policy_rr(env: SchedEnv, _tasks, ep_state: dict) -> int:
    """Round Robin: cycle through runnable processes by PID, tier 0."""
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0
    rr_idx   = ep_state.get("rr_idx", 0)
    chosen   = sorted(runnable, key=lambda p: p.pid)[rr_idx % len(runnable)]
    ep_state["rr_idx"] = rr_idx + 1
    return chosen.pid * N_QT + 0


def policy_cfs(env: SchedEnv, _tasks, ep_state: dict) -> int:
    """CFS-lite: always run the process with lowest accumulated CPU time."""
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0
    vrt    = ep_state.setdefault("vruntime", {p.pid: 0.0 for p in env.processes})
    chosen = min(runnable, key=lambda p: vrt[p.pid])
    vrt[chosen.pid] += QUANTUM_TIERS[0]
    return chosen.pid * N_QT + 0


def policy_mlfq(env: SchedEnv, _tasks, ep_state: dict) -> int:
    """MLFQ: 3 queues mapped to quantum tiers 0/1/2.

    Demotion: process uses its full quantum → move to next queue.
    Aging:    time_since_last_execution > MLFQ_AGE_THRESH → promote to Q0.
    Within a queue: FIFO by arrival time.
    """
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0

    queues   = ep_state.setdefault("mlfq_queue",    {p.pid: 0   for p in env.processes})
    prev_rem = ep_state.setdefault("prev_remaining", {p.pid: p.burst_length for p in env.processes})
    last_pid = ep_state.get("last_pid")

    # Demotion: if the previous process used its full quantum, demote it
    if last_pid is not None:
        prev_p = next((p for p in env.processes if p.pid == last_pid), None)
        if prev_p is not None and not prev_p.is_complete:
            tier     = queues.get(prev_p.pid, 0)
            consumed = prev_rem.get(prev_p.pid, 0.0) - prev_p.remaining_burst
            if consumed >= QUANTUM_TIERS[tier] - 1e-6 and tier < 2:
                queues[prev_p.pid] = tier + 1

    # Aging: starving processes jump back to Q0
    for p in runnable:
        if p.time_since_last_execution > MLFQ_AGE_THRESH:
            queues[p.pid] = 0

    # Select from highest non-empty queue
    for level in range(3):
        candidates = [p for p in runnable if queues.get(p.pid, 0) == level]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda p: p.arrival_time)
        ep_state["last_pid"]              = chosen.pid
        prev_rem[chosen.pid]              = chosen.remaining_burst
        return chosen.pid * N_QT + level

    chosen = runnable[0]
    return chosen.pid * N_QT + 0


POLICIES = [
    ("MLFQ",         policy_mlfq),
    ("Round Robin",  policy_rr),
    ("CFS-lite",     policy_cfs),
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_policy(policy_fn, sampler: TraceEpisodeSamplerN,
               n_eval: int, seed: int) -> dict:
    """Run n_eval episodes; return mct_mean and mct_std."""
    rng  = np.random.default_rng(seed)
    mcts = []

    for _ in range(n_eval):
        tasks = sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()

        ep_state = {}
        done     = False

        while not done:
            action = policy_fn(env, tasks, ep_state)
            _, _, done, info = env.step(action)

        mct = info.get("mean_completion_time_so_far") or 0.0
        mcts.append(mct)

    return {
        "mct_mean": float(np.mean(mcts)),
        "mct_std":  float(np.std(mcts)),
    }


# ---------------------------------------------------------------------------
# Load VA-DQN reference from summary.md
# ---------------------------------------------------------------------------

def _load_vadqn_reference(summary_path: str) -> dict[int, float]:
    """Parse the summary table to extract VA-DQN mean MCT per N."""
    vadqn = {}
    try:
        with open(summary_path) as f:
            for line in f:
                # Match table rows like: |  5  |  21.67  | ...
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 4:
                    continue
                try:
                    n   = int(parts[1])
                    mct = float(parts[2])
                    if n in (5, 10, 20, 50):
                        vadqn[n] = mct
                except (ValueError, IndexError):
                    pass
    except FileNotFoundError:
        pass
    return vadqn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve trace file
    if args.trace_path is not None:
        tp = args.trace_path
        trace_file = os.path.join(tp, "trace_train_filtered.csv") if os.path.isdir(tp) else tp
    else:
        trace_file = _CFG_TRACE

    out_dir = get_agent_dir("n_scaling")
    out_path = os.path.join(out_dir, "baselines.json")

    summary_path = os.path.join(out_dir, "summary.md")
    vadqn_ref    = _load_vadqn_reference(summary_path)

    print(f"Trace:     {trace_file}")
    print(f"Output:    {out_path}")
    print(f"N values:  {args.n_list}")
    print(f"Seeds:     {args.seeds}  ×  {args.n_eval} episodes each")
    print()

    all_results: dict = {}

    for N in args.n_list:
        print(f"{'='*60}")
        print(f"N = {N}")
        print(f"{'='*60}")

        sampler = TraceEpisodeSamplerN(trace_file, n_processes=N)

        all_results[N] = {}

        for pol_name, pol_fn in POLICIES:
            seed_mcts = []
            t0 = time.time()
            for seed in args.seeds:
                res = run_policy(pol_fn, sampler, args.n_eval, seed)
                seed_mcts.append(res["mct_mean"])

            mean_across_seeds = float(np.mean(seed_mcts))
            std_across_seeds  = float(np.std(seed_mcts))
            elapsed           = time.time() - t0

            all_results[N][pol_name] = {
                "mct_mean": mean_across_seeds,
                "mct_std":  std_across_seeds,
                "per_seed": {str(s): float(m) for s, m in zip(args.seeds, seed_mcts)},
            }
            print(f"  {pol_name:<14}  MCT={mean_across_seeds:.2f}±{std_across_seeds:.2f}s"
                  f"  ({elapsed:.1f}s wall)")

        print()

    # Save JSON
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {out_path}\n")

    # Print summary table
    print(f"{'N':>4}  {'Policy':<14}  {'Mean MCT':>10}  {'Std':>7}  {'vs VA-DQN':>10}")
    print("-" * 54)
    for N in args.n_list:
        vadqn_mct = vadqn_ref.get(N)
        for pol_name, _ in POLICIES:
            res = all_results[N][pol_name]
            m   = res["mct_mean"]
            s   = res["mct_std"]
            if vadqn_mct is not None:
                # Positive = baseline is slower than VA-DQN (VA-DQN wins)
                pct = (m - vadqn_mct) / vadqn_mct * 100.0
                vs  = f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"
            else:
                vs = "n/a"
            print(f"{N:>4}  {pol_name:<14}  {m:>9.2f}s  {s:>6.2f}s  {vs:>10}")
        if vadqn_mct is not None:
            print(f"{'':>4}  {'VA-DQN':<14}  {vadqn_mct:>9.2f}s  {'':>7}  {'(ref)':>10}")
        print()


if __name__ == "__main__":
    main()
