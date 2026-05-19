"""Fairness evaluation (starvation + VRFI) for N-scaling experiment.

Evaluates VA-DQN (greedy, ε=0), MLFQ, and Round Robin at
N ∈ {5, 10, 20, 50} using checkpoints already stored locally.

Metrics match the main paper (benchmark_vs_mlfq.py):

  Starvation (per episode, binary):
    Episode is "starved" if any task has slowdown > 3 × median slowdown,
    where slowdown = turnaround_time / burst_length.
    Reported as % of episodes with at least one starved task.

  VLR (value loss rate) per completed task:
    VLR_i = (V_i(0) − V_i(wait_i)) / max(wait_i, 1.0)
    V_i(d) = base_value_i × max(floor_i, exp(−d / tau_i))
    tau_i, floor_i, base_value_i sampled by env.reset() each episode.

  VRFI (value-rate fairness index):
    Accumulated across all tasks in all episodes:
    VRFI = 1 − CV(VLR)  where CV = std(VLR) / mean(VLR)
    Higher is better (1.0 = perfect equality of value loss rates).

Usage:
  python eval_fairness_n_scaling.py               # all N
  python eval_fairness_n_scaling.py --n_list 5    # smoke test
  python eval_fairness_n_scaling.py --trace_path /data/alibaba2018/
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

from project_config   import TRACE_PATH as _CFG_TRACE, get_agent_dir
from schedsim.env     import SchedEnv
from schedsim.process import Process
from w9_train         import (
    _valid_actions, _make_procs, N_QT,
    _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem, WAIT_NORM,
)
from w15_network_torch import W15Trainer, device

import torch

STARVATION_SLOW = 3.0   # slowdown multiplier — matches benchmark_vs_mlfq.py

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fairness evaluation for N-scaling")
    p.add_argument("--trace_path", type=str, default=None)
    p.add_argument("--n_list",  type=int, nargs="+", default=[5, 10, 20, 50])
    p.add_argument("--n_eval",  type=int, default=500)
    p.add_argument("--seeds",   type=int, nargs="+",
                   default=[42, 123, 456, 789, 999])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Trace sampler
# ---------------------------------------------------------------------------

_BURST_P95_FILT = 36.0
D_CAND          = 7


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
# State encoder (identical to train.py / eval_agent_n_scaling.py)
# ---------------------------------------------------------------------------

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
# Fairness metrics extracted from one completed episode
# ---------------------------------------------------------------------------

def _episode_fairness(env: SchedEnv) -> dict:
    """Compute starvation flag and per-task VLR for a completed episode.

    Returns:
      starved  : int  — 1 if any task's slowdown > 3 × median, else 0
      vlrs     : list[float]  — one VLR per completed task
    """
    completed = [p for p in env.processes if p.is_complete]
    if not completed:
        return {"starved": 0, "vlrs": []}

    # Starvation: slowdown = turnaround / burst_length
    turnarounds = [p.completion_time - p.arrival_time for p in completed]
    bursts      = [p.burst_length                     for p in completed]
    slowdowns   = [t / max(b, 1e-6) for t, b in zip(turnarounds, bursts)]
    med_slow    = float(np.median(slowdowns))
    starved     = int(any(s > STARVATION_SLOW * med_slow for s in slowdowns))

    # VLR per task — uses value-curve params set by env.reset()
    vlrs = []
    for p in completed:
        delay = p.wait_time
        if p.tau > 0:
            v_delivered = p.base_value * max(p.floor, math.exp(-delay / p.tau))
        else:
            v_delivered = p.base_value
        vlr = (p.base_value - v_delivered) / max(delay, 1.0)
        vlrs.append(vlr)

    return {"starved": starved, "vlrs": vlrs}


def _aggregate(starved_list: list[int], all_vlrs: list[float]) -> dict:
    """Aggregate episode results into summary statistics."""
    starve_pct = float(np.sum(starved_list)) / max(len(starved_list), 1) * 100.0

    if len(all_vlrs) > 1:
        a  = np.array(all_vlrs, dtype=np.float64)
        cv = float(np.std(a) / (np.mean(a) + 1e-12))
    else:
        cv = 0.0
    vrfi = 1.0 - cv

    return {
        "starve_pct": starve_pct,
        "vrfi":       vrfi,
        "n_episodes": len(starved_list),
        "n_tasks":    len(all_vlrs),
    }


# ---------------------------------------------------------------------------
# Baseline policies (same as eval_baselines_n_scaling.py)
# ---------------------------------------------------------------------------

QUANTUM_TIERS   = (0.5, 2.0, 8.0)
MLFQ_AGE_THRESH = 50.0


def policy_rr(env, _tasks, ep_state):
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0
    rr_idx   = ep_state.get("rr_idx", 0)
    chosen   = sorted(runnable, key=lambda p: p.pid)[rr_idx % len(runnable)]
    ep_state["rr_idx"] = rr_idx + 1
    return chosen.pid * N_QT + 0


def policy_mlfq(env, _tasks, ep_state):
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0
    queues   = ep_state.setdefault("mlfq_queue",    {p.pid: 0   for p in env.processes})
    prev_rem = ep_state.setdefault("prev_remaining", {p.pid: p.burst_length for p in env.processes})
    last_pid = ep_state.get("last_pid")

    if last_pid is not None:
        prev_p = next((p for p in env.processes if p.pid == last_pid), None)
        if prev_p is not None and not prev_p.is_complete:
            tier     = queues.get(prev_p.pid, 0)
            consumed = prev_rem.get(prev_p.pid, 0.0) - prev_p.remaining_burst
            if consumed >= QUANTUM_TIERS[tier] - 1e-6 and tier < 2:
                queues[prev_p.pid] = tier + 1

    for p in runnable:
        if p.time_since_last_execution > MLFQ_AGE_THRESH:
            queues[p.pid] = 0

    for level in range(3):
        candidates = [p for p in runnable if queues.get(p.pid, 0) == level]
        if not candidates:
            continue
        chosen = min(candidates, key=lambda p: p.arrival_time)
        ep_state["last_pid"]  = chosen.pid
        prev_rem[chosen.pid]  = chosen.remaining_burst
        return chosen.pid * N_QT + level

    return runnable[0].pid * N_QT + 0


# ---------------------------------------------------------------------------
# Generic evaluation loop for one (policy, N, seed-list) combination
# ---------------------------------------------------------------------------

def _run_baseline(policy_fn, sampler, n_eval, seeds):
    starved_list, all_vlrs = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for _ in range(n_eval):
            tasks = sampler.sample_episode(rng)
            procs = _make_procs(tasks)
            env   = SchedEnv(procs)
            env.reset()
            ep_state = {}
            done     = False
            while not done:
                _, _, done, _ = env.step(policy_fn(env, tasks, ep_state))
            ep = _episode_fairness(env)
            starved_list.append(ep["starved"])
            all_vlrs.extend(ep["vlrs"])
    return _aggregate(starved_list, all_vlrs)


def _run_agent(trainer: W15Trainer, N: int, sampler, n_eval, seeds):
    starved_list, all_vlrs = [], []
    with torch.no_grad():
        for seed in seeds:
            rng = np.random.default_rng(seed)
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
                    _, _, done, _ = env.step(action)
                    sv = _encode_state(env, tasks, N)
                ep = _episode_fairness(env)
                starved_list.append(ep["starved"])
                all_vlrs.extend(ep["vlrs"])
    return _aggregate(starved_list, all_vlrs)


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
    out_path      = os.path.join(n_scaling_dir, "fairness_eval.json")

    print(f"Trace:  {trace_file}")
    print(f"Output: {out_path}")
    print(f"N values: {args.n_list}  |  seeds: {args.seeds}  |  {args.n_eval} eps/seed")
    print()

    all_results: dict = {}

    for N in args.n_list:
        print(f"{'='*60}")
        print(f"N = {N}")
        print(f"{'='*60}")

        sampler = TraceEpisodeSamplerN(trace_file, n_processes=N)

        # --- VA-DQN (greedy) — average across seeds ---
        ag_starved, ag_vlrs = [], []
        missing = []
        for seed in args.seeds:
            ckpt_dir = os.path.join(n_scaling_dir, f"n{N}_seed{seed}")
            ckpt     = next(
                (os.path.join(ckpt_dir, f) for f in ("final.pt", "checkpoint.pt")
                 if os.path.isfile(os.path.join(ckpt_dir, f))),
                None,
            )
            if ckpt is None:
                print(f"  MISSING checkpoint: {ckpt_dir}/final.pt — skipping seed {seed}")
                missing.append(seed)
                continue

            trainer         = W15Trainer(n_processes=N)
            trainer.load(ckpt)
            trainer.epsilon = 0.0
            trainer.online.eval()

            t0 = time.time()
            res = _run_agent(trainer, N, sampler, args.n_eval, [seed])
            print(f"  VA-DQN seed={seed}  starve={res['starve_pct']:.1f}%"
                  f"  VRFI={res['vrfi']:.3f}  ({time.time()-t0:.1f}s)")
            ag_starved.extend([res["starve_pct"]])
            ag_vlrs.append(res["vrfi"])

        if missing:
            print(f"  ({len(missing)} seeds missing — averages over remaining seeds)")

        # Aggregate agent across seeds by re-running as one block to get
        # correct global VRFI (not mean-of-means).  Reuse loaded checkpoints.
        agent_seeds = [s for s in args.seeds if s not in missing]
        if agent_seeds:
            # Load all seed checkpoints and run together for global VRFI
            all_starved_ag, all_vlrs_ag = [], []
            with torch.no_grad():
                for seed in agent_seeds:
                    ckpt_dir = os.path.join(n_scaling_dir, f"n{N}_seed{seed}")
                    ckpt     = os.path.join(ckpt_dir, "final.pt")
                    trainer  = W15Trainer(n_processes=N)
                    trainer.load(ckpt)
                    trainer.epsilon = 0.0
                    trainer.online.eval()
                    rng = np.random.default_rng(seed)
                    for _ in range(args.n_eval):
                        tasks = sampler.sample_episode(rng)
                        procs = _make_procs(tasks)
                        env   = SchedEnv(procs)
                        env.reset()
                        sv   = _encode_state(env, tasks, N)
                        done = False
                        while not done:
                            valid  = _valid_actions(env)
                            action = trainer.select_action(sv, valid, omega_s=0.5)
                            _, _, done, _ = env.step(action)
                            sv = _encode_state(env, tasks, N)
                        ep = _episode_fairness(env)
                        all_starved_ag.append(ep["starved"])
                        all_vlrs_ag.extend(ep["vlrs"])
            agent_agg = _aggregate(all_starved_ag, all_vlrs_ag)
        else:
            agent_agg = {"starve_pct": float("nan"), "vrfi": float("nan"),
                         "n_episodes": 0, "n_tasks": 0}

        # --- Baselines ---
        t0 = time.time()
        mlfq_res = _run_baseline(policy_mlfq, sampler, args.n_eval, args.seeds)
        print(f"  MLFQ         starve={mlfq_res['starve_pct']:.1f}%  VRFI={mlfq_res['vrfi']:.3f}"
              f"  ({time.time()-t0:.1f}s)")

        t0 = time.time()
        rr_res   = _run_baseline(policy_rr,   sampler, args.n_eval, args.seeds)
        print(f"  Round Robin  starve={rr_res['starve_pct']:.1f}%  VRFI={rr_res['vrfi']:.3f}"
              f"  ({time.time()-t0:.1f}s)")

        all_results[N] = {
            "VA-DQN":       agent_agg,
            "MLFQ":         mlfq_res,
            "Round Robin":  rr_res,
        }
        print()

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved → {out_path}\n")

    # Summary table
    print(f"{'N':>4}  {'Policy':<14}  {'Starve%':>9}  {'VRFI':>7}")
    print("-" * 42)
    for N in args.n_list:
        if N not in all_results:
            continue
        for pol in ("VA-DQN", "MLFQ", "Round Robin"):
            r = all_results[N].get(pol, {})
            s = r.get("starve_pct", float("nan"))
            v = r.get("vrfi",       float("nan"))
            print(f"{N:>4}  {pol:<14}  {s:>8.1f}%  {v:>7.3f}")
        print()


if __name__ == "__main__":
    main()
