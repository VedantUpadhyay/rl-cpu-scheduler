"""Greedy (ε=0) agent evaluation for N-scaling experiment.

Loads saved W15 checkpoints and evaluates them with pure greedy action
selection on 500 fresh episodes per seed, then averages across 5 seeds.

Expected checkpoint layout (written by train.py on Nautilus):
  docs/scheduler-research/results/n_scaling/n{N}_seed{SEED}/final.pt

Compares against baselines loaded from baselines.json (written by
eval_baselines_n_scaling.py).  VA-DQN training MCT from summary.md is
shown for reference so the ε gap is visible.

Usage:
  python eval_agent_n_scaling.py                     # all N
  python eval_agent_n_scaling.py --n_list 5          # smoke test
  python eval_agent_n_scaling.py --trace_path /data/alibaba2018/
"""
from __future__ import annotations
import argparse
import csv
import json
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

from project_config  import TRACE_PATH as _CFG_TRACE, get_agent_dir
from schedsim.env    import SchedEnv
from schedsim.process import Process
from w9_train         import _valid_actions, _make_procs, N_QT
from w15_network_torch import W15Trainer, device

import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Greedy agent evaluation for N-scaling")
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
# Trace sampler (identical to train.py)
# ---------------------------------------------------------------------------

_BURST_P95_FILT = 36.0
D_CAND          = 7
AFI             = 6   # arrived_flag index


class TraceEpisodeSamplerN:
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
# State encoder (identical to train.py)
# ---------------------------------------------------------------------------

from w9_train import (
    _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem, WAIT_NORM,
)


def _encode_state(env: SchedEnv, tasks: list[dict],
                  n_processes: int) -> np.ndarray:
    vec = np.zeros(n_processes * D_CAND, dtype=np.float32)
    for p in env.processes:
        i   = p.pid
        off = i * D_CAND
        if p.arrival_time <= env.current_time and not p.is_complete:
            tq = env.current_time - p.arrival_time
            vec[off + 0] = _norm_time_log(tq)
            vec[off + 1] = p.wait_time / WAIT_NORM
            vec[off + 2] = _norm_time_log(p.time_since_last_execution)
            vec[off + 3] = _urgency_norm(p)
            vec[off + 4] = _norm_cpu(tasks[i]["plan_cpu"])
            vec[off + 5] = _norm_mem(tasks[i]["plan_mem"])
            vec[off + 6] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_checkpoint(n_scaling_dir: str, N: int, seed: int) -> str | None:
    """Return path to checkpoint for (N, seed), or None if not found."""
    run_dir = os.path.join(n_scaling_dir, f"n{N}_seed{seed}")
    for name in ("final.pt", "checkpoint.pt"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Single-checkpoint greedy evaluation
# ---------------------------------------------------------------------------

def eval_checkpoint(ckpt_path: str, N: int,
                    sampler: TraceEpisodeSamplerN,
                    n_eval: int, seed: int) -> dict:
    """Load checkpoint, run n_eval greedy episodes, return mct_mean/std."""
    trainer = W15Trainer(n_processes=N)
    trainer.load(ckpt_path)
    trainer.epsilon = 0.0        # pure greedy
    trainer.online.eval()

    rng  = np.random.default_rng(seed)
    mcts = []

    with torch.no_grad():
        for _ in range(n_eval):
            tasks = sampler.sample_episode(rng)
            procs = _make_procs(tasks)
            env   = SchedEnv(procs)
            env.reset()
            sv   = _encode_state(env, tasks, N)
            done = False

            while not done:
                valid  = _valid_actions(env)
                action = trainer.select_action(sv, valid, omega_s=0.5)
                _, _, done, info = env.step(action)
                sv = _encode_state(env, tasks, N)

            mct = info.get("mean_completion_time_so_far") or 0.0
            mcts.append(mct)

    return {
        "mct_mean": float(np.mean(mcts)),
        "mct_std":  float(np.std(mcts)),
    }


# ---------------------------------------------------------------------------
# Load comparison data
# ---------------------------------------------------------------------------

def _load_baselines(n_scaling_dir: str) -> dict:
    p = os.path.join(n_scaling_dir, "baselines.json")
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        raw = json.load(f)
    # keys are strings in JSON
    return {int(k): v for k, v in raw.items()}


def _load_training_mct(n_scaling_dir: str) -> dict[int, float]:
    """Parse summary.md for VA-DQN training MCT per N."""
    ref = {}
    p = os.path.join(n_scaling_dir, "summary.md")
    try:
        with open(p) as f:
            for line in f:
                parts = [x.strip() for x in line.split("|")]
                if len(parts) < 4:
                    continue
                try:
                    n   = int(parts[1])
                    mct = float(parts[2])
                    if n in (5, 10, 20, 50):
                        ref[n] = mct
                except (ValueError, IndexError):
                    pass
    except FileNotFoundError:
        pass
    return ref


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.trace_path is not None:
        tp = args.trace_path
        trace_file = os.path.join(tp, "trace_train_filtered.csv") if os.path.isdir(tp) else tp
    else:
        trace_file = _CFG_TRACE

    n_scaling_dir = get_agent_dir("n_scaling")
    out_path      = os.path.join(n_scaling_dir, "agent_eval.json")

    baselines   = _load_baselines(n_scaling_dir)
    training_mct = _load_training_mct(n_scaling_dir)

    print(f"Trace:     {trace_file}")
    print(f"Output:    {out_path}")
    print(f"Device:    {device}")
    print(f"N values:  {args.n_list}")
    print(f"Seeds:     {args.seeds}  ×  {args.n_eval} episodes each")
    print()

    all_results: dict = {}

    for N in args.n_list:
        print(f"{'='*60}")
        print(f"N = {N}")
        print(f"{'='*60}")

        sampler = TraceEpisodeSamplerN(trace_file, n_processes=N)

        seed_mcts   = []
        missing     = []

        for seed in args.seeds:
            ckpt = _find_checkpoint(n_scaling_dir, N, seed)
            if ckpt is None:
                print(f"  MISSING  n{N}_seed{seed}  (no final.pt found)")
                missing.append(seed)
                continue

            t0  = time.time()
            res = eval_checkpoint(ckpt, N, sampler, args.n_eval, seed)
            elapsed = time.time() - t0
            seed_mcts.append(res["mct_mean"])
            print(f"  seed={seed}  MCT={res['mct_mean']:.2f}±{res['mct_std']:.2f}s"
                  f"  ckpt={os.path.basename(ckpt)}  ({elapsed:.1f}s)")

        if not seed_mcts:
            print(f"  No checkpoints found for N={N} — skipping.\n")
            print(f"  Expected path: {n_scaling_dir}/n{N}_seed<SEED>/final.pt\n")
            continue

        mean_mct = float(np.mean(seed_mcts))
        std_mct  = float(np.std(seed_mcts))

        all_results[N] = {
            "mct_mean":     mean_mct,
            "mct_std":      std_mct,
            "per_seed":     {str(s): float(m) for s, m in zip(
                                [s for s in args.seeds if s not in missing],
                                seed_mcts)},
            "seeds_missing": missing,
            "n_eval":        args.n_eval,
        }
        print(f"  → greedy mean MCT = {mean_mct:.2f} ± {std_mct:.2f}s"
              f"  (training ref: {training_mct.get(N, 'n/a')}s)\n")

    # Save
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved → {out_path}\n")

    # Summary comparison table
    evaluated = [N for N in args.n_list if N in all_results]
    if not evaluated:
        print("No results to display — check checkpoint paths above.")
        return

    print(f"{'N':>4}  {'Policy':<18}  {'Mean MCT':>10}  {'Std':>7}  {'vs MLFQ':>9}")
    print("-" * 58)

    for N in evaluated:
        agent_mct = all_results[N]["mct_mean"]
        agent_std = all_results[N]["mct_std"]
        bl        = baselines.get(N, {})
        mlfq_mct  = bl.get("MLFQ",        {}).get("mct_mean")
        rr_mct    = bl.get("Round Robin",  {}).get("mct_mean")
        cfs_mct   = bl.get("CFS-lite",     {}).get("mct_mean")
        train_mct = training_mct.get(N)

        def _vs(baseline):
            if baseline is None:
                return "n/a"
            pct = (agent_mct - baseline) / baseline * 100.0
            return f"{pct:+.1f}%"

        print(f"{N:>4}  {'VA-DQN (greedy)':<18}  {agent_mct:>9.2f}s  {agent_std:>6.2f}s  {_vs(mlfq_mct):>9}")
        if train_mct is not None:
            print(f"{'':>4}  {'VA-DQN (training)':<18}  {train_mct:>9.2f}s  {'':>7}  {_vs(mlfq_mct):>9}  ← ε=0.05 ref")
        if mlfq_mct  is not None: print(f"{'':>4}  {'MLFQ':<18}  {mlfq_mct:>9.2f}s  {bl['MLFQ']['mct_std']:>6.2f}s")
        if rr_mct    is not None: print(f"{'':>4}  {'Round Robin':<18}  {rr_mct:>9.2f}s  {bl['Round Robin']['mct_std']:>6.2f}s")
        if cfs_mct   is not None: print(f"{'':>4}  {'CFS-lite':<18}  {cfs_mct:>9.2f}s  {bl['CFS-lite']['mct_std']:>6.2f}s")
        print()


if __name__ == "__main__":
    main()
