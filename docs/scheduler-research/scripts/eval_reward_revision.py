"""Evaluation script for the reward-revision experiment.

Loads TransformerSchedulerNet checkpoints from the reward-revision Nautilus
run, evaluates each (lambda, threshold, seed) combination at greedy ε=0,
and produces starvation-rate and MCT grids for comparison against MLFQ.

Expected checkpoint layout (after kubectl cp from reward-revision-out-pvc):
  docs/scheduler-research/results/reward_revision/
    lambda0.01_thresh50.0_seed42/final.pt
    lambda0.01_thresh50.0_seed123/final.pt
    ...  (27 total)

MLFQ baseline: loaded from baselines.json (N=50) if present; otherwise
run fresh.  Fallback hard-coded constants match eval_baselines_n_scaling.py.

Goal: find (lambda, threshold) where
  starvation < MLFQ 56.0%  AND  MCT within 10% of MLFQ 163.94s.

Usage:
  python eval_reward_revision.py
  python eval_reward_revision.py --n_eval 100          # quick smoke test
  python eval_reward_revision.py --trace_path /data/alibaba2018/
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

from project_config  import TRACE_PATH as _CFG_TRACE, get_agent_dir
from schedsim.env    import SchedEnv
from w9_train        import (
    _valid_actions, _make_procs, N_QT,
    _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem, WAIT_NORM,
)
from transformer_agent import TransformerTrainer, D_CAND
from train             import TraceEpisodeSamplerN

import torch

# ---------------------------------------------------------------------------
# Experiment grid (must match nautilus_reward_revision.yaml bash arrays)
# ---------------------------------------------------------------------------
LAMBDAS     = ["0.01", "0.05", "0.10"]   # string keys so dir names match exactly
THRESHOLDS  = ["50.0", "100.0", "200.0"]
SEEDS       = [42, 123, 456]
N           = 50

# MLFQ reference values at N=50 (from eval_baselines_n_scaling.py / fairness eval)
MLFQ_MCT_REF    = 163.94   # seconds
MLFQ_STARVE_REF = 56.0     # percent
MCT_BUDGET_FRAC = 0.10     # "within 10%" → MCT ≤ MLFQ × 1.10

STARVATION_SLOW = 3.0      # slowdown multiplier threshold

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reward-revision experiment evaluation")
    p.add_argument("--trace_path", type=str, default=None,
                   help="Trace file or directory (default: project_config.TRACE_PATH)")
    p.add_argument("--n_eval", type=int, default=500,
                   help="Episodes per (lambda, threshold, seed) (default: 500)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# State encoder — identical signature to transformer_train.py
# ---------------------------------------------------------------------------

def _encode_state(env: SchedEnv, tasks: list[dict], n_processes: int) -> np.ndarray:
    norm_fns = {
        "time":      _norm_time_log,
        "wait_norm": WAIT_NORM,
        "urgency":   _urgency_norm,
        "cpu":       _norm_cpu,
        "mem":       _norm_mem,
    }
    vec = np.zeros(n_processes * D_CAND, dtype=np.float32)
    for p in env.processes:
        i   = p.pid
        off = i * D_CAND
        if p.arrival_time <= env.current_time and not p.is_complete:
            tq = env.current_time - p.arrival_time
            vec[off + 0] = norm_fns["time"](tq)
            vec[off + 1] = p.wait_time / norm_fns["wait_norm"]
            vec[off + 2] = norm_fns["time"](p.time_since_last_execution)
            vec[off + 3] = norm_fns["urgency"](p)
            vec[off + 4] = norm_fns["cpu"](tasks[i]["plan_cpu"])
            vec[off + 5] = norm_fns["mem"](tasks[i]["plan_mem"])
            vec[off + 6] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Fairness metrics (identical to eval_fairness_n_scaling.py)
# ---------------------------------------------------------------------------

def _episode_metrics(env: SchedEnv) -> dict:
    """Return MCT, starvation flag, and per-task VLR for one completed episode."""
    completed = [p for p in env.processes if p.is_complete]
    if not completed:
        return {"mct": 0.0, "starved": 0, "vlrs": []}

    # MCT
    mct = sum(p.completion_time - p.arrival_time for p in completed) / len(completed)

    # Starvation: any task slowdown > 3 × median slowdown
    turnarounds = [p.completion_time - p.arrival_time for p in completed]
    bursts      = [p.burst_length                     for p in completed]
    slowdowns   = [t / max(b, 1e-6) for t, b in zip(turnarounds, bursts)]
    med_slow    = float(np.median(slowdowns))
    starved     = int(any(s > STARVATION_SLOW * med_slow for s in slowdowns))

    # VLR per task
    vlrs = []
    for p in completed:
        delay = p.wait_time
        if p.tau > 0:
            v_delivered = p.base_value * max(p.floor, math.exp(-delay / p.tau))
        else:
            v_delivered = p.base_value
        vlrs.append((p.base_value - v_delivered) / max(delay, 1.0))

    return {"mct": float(mct), "starved": starved, "vlrs": vlrs}


def _aggregate(ep_metrics: list[dict]) -> dict:
    mcts        = [e["mct"]     for e in ep_metrics]
    starved     = [e["starved"] for e in ep_metrics]
    all_vlrs    = [v for e in ep_metrics for v in e["vlrs"]]

    starve_pct = float(np.mean(starved)) * 100.0
    mct_mean   = float(np.mean(mcts))
    mct_std    = float(np.std(mcts))

    if len(all_vlrs) > 1:
        a  = np.array(all_vlrs, dtype=np.float64)
        cv = float(np.std(a) / (np.mean(a) + 1e-12))
    else:
        cv = 0.0
    vrfi = 1.0 - cv

    return {
        "mct_mean":    mct_mean,
        "mct_std":     mct_std,
        "starve_pct":  starve_pct,
        "vrfi":        vrfi,
        "n_episodes":  len(ep_metrics),
        "n_tasks":     len(all_vlrs),
    }


# ---------------------------------------------------------------------------
# Transformer agent greedy evaluation
# ---------------------------------------------------------------------------

def eval_transformer(ckpt_path: str, sampler: TraceEpisodeSamplerN,
                     n_eval: int, seed: int) -> list[dict]:
    trainer = TransformerTrainer()
    trainer.load(ckpt_path)
    trainer.epsilon = 0.0
    trainer.online.eval()

    rng     = np.random.default_rng(seed)
    results = []

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
                _, _, done, _ = env.step(action)
                sv = _encode_state(env, tasks, N)
            results.append(_episode_metrics(env))

    return results


# ---------------------------------------------------------------------------
# MLFQ baseline
# ---------------------------------------------------------------------------

MLFQ_AGE_THRESH = 50.0
QUANTUM_TIERS   = (0.5, 2.0, 8.0)


def _policy_mlfq(env: SchedEnv, ep_state: dict) -> int:
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0
    queues   = ep_state.setdefault("queues",    {p.pid: 0   for p in env.processes})
    prev_rem = ep_state.setdefault("prev_rem",  {p.pid: p.burst_length for p in env.processes})
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
        ep_state["last_pid"] = chosen.pid
        prev_rem[chosen.pid] = chosen.remaining_burst
        return chosen.pid * N_QT + level

    return runnable[0].pid * N_QT + 0


def eval_mlfq(sampler: TraceEpisodeSamplerN, n_eval: int,
              seeds: list[int]) -> list[dict]:
    results = []
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
                _, _, done, _ = env.step(_policy_mlfq(env, ep_state))
            results.append(_episode_metrics(env))
    return results


def _load_mlfq_baseline(rr_dir: str) -> dict | None:
    """Try to load MLFQ N=50 numbers from existing baselines.json."""
    p = os.path.join(rr_dir, "..", "n_scaling", "baselines.json")
    p = os.path.normpath(p)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        entry = data.get("50", data.get(50, {}))
        mlfq  = entry.get("MLFQ")
        if mlfq and "mct_mean" in mlfq:
            return mlfq
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _load_mlfq_fairness(rr_dir: str) -> dict | None:
    """Try to load MLFQ N=50 starvation/VRFI from existing fairness_eval.json."""
    p = os.path.join(rr_dir, "..", "n_scaling", "fairness_eval.json")
    p = os.path.normpath(p)
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as f:
            data = json.load(f)
        entry = data.get("50", data.get(50, {}))
        return entry.get("MLFQ")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


# ---------------------------------------------------------------------------
# Grid printing helpers
# ---------------------------------------------------------------------------

def _print_grid(title: str, values: dict[tuple, str], lambdas: list[str],
                thresholds: list[str], footer: str) -> None:
    col_w = 8
    header = f"{'λ\\threshold':<12}" + "".join(f"{t:>{col_w}}" for t in thresholds)
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for lam in lambdas:
        row = f"{lam:<12}"
        for thr in thresholds:
            row += f"{values.get((lam, thr), 'n/a'):>{col_w}}"
        print(row)
    print(f"{footer}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    if args.trace_path is not None:
        tp = args.trace_path
        trace_file = os.path.join(tp, "trace_train_filtered.csv") \
                     if os.path.isdir(tp) else tp
    else:
        trace_file = _CFG_TRACE

    rr_dir = get_agent_dir("reward_revision")
    out_path = os.path.join(rr_dir, "summary.json")

    print(f"Trace      : {trace_file}")
    print(f"Results dir: {rr_dir}")
    print(f"Output     : {out_path}")
    print(f"Episodes   : {args.n_eval} per (lambda, threshold, seed)")
    print(f"Seeds      : {SEEDS}")
    print()

    sampler = TraceEpisodeSamplerN(trace_file, n_processes=N)

    # ── MLFQ baseline ─────────────────────────────────────────────────────
    mlfq_bl_mct    = _load_mlfq_baseline(rr_dir)
    mlfq_bl_fair   = _load_mlfq_fairness(rr_dir)

    if mlfq_bl_mct and mlfq_bl_fair:
        mlfq_mct    = mlfq_bl_mct["mct_mean"]
        mlfq_starve = mlfq_bl_fair["starve_pct"]
        mlfq_vrfi   = mlfq_bl_fair["vrfi"]
        mlfq_source = "cached"
        print(f"MLFQ baseline (N=50): MCT={mlfq_mct:.2f}s  "
              f"starve={mlfq_starve:.1f}%  VRFI={mlfq_vrfi:.3f}  [from cache]")
    else:
        print("Running MLFQ baseline (N=50) fresh …")
        t0 = time.time()
        mlfq_eps  = eval_mlfq(sampler, args.n_eval, SEEDS)
        mlfq_agg  = _aggregate(mlfq_eps)
        mlfq_mct    = mlfq_agg["mct_mean"]
        mlfq_starve = mlfq_agg["starve_pct"]
        mlfq_vrfi   = mlfq_agg["vrfi"]
        mlfq_source = "fresh"
        print(f"MLFQ baseline (N=50): MCT={mlfq_mct:.2f}s  "
              f"starve={mlfq_starve:.1f}%  VRFI={mlfq_vrfi:.3f}  ({time.time()-t0:.1f}s)")
    print()

    mct_ceiling = mlfq_mct * (1.0 + MCT_BUDGET_FRAC)

    # ── Agent evaluation ──────────────────────────────────────────────────
    all_results: dict = {}

    for lam in LAMBDAS:
        for thr in THRESHOLDS:
            cell_eps: list[dict] = []
            seed_rows: list[dict] = []

            for seed in SEEDS:
                tag  = f"lambda{lam}_thresh{thr}_seed{seed}"
                ckpt = os.path.join(rr_dir, tag, "final.pt")

                if not os.path.isfile(ckpt):
                    print(f"  MISSING  {tag}/final.pt — skipping")
                    seed_rows.append({"seed": seed, "status": "missing"})
                    continue

                t0  = time.time()
                eps = eval_transformer(ckpt, sampler, args.n_eval, seed)
                agg = _aggregate(eps)
                cell_eps.extend(eps)

                print(f"  λ={lam}  thresh={thr}  seed={seed}"
                      f"  MCT={agg['mct_mean']:.2f}s"
                      f"  starve={agg['starve_pct']:.1f}%"
                      f"  VRFI={agg['vrfi']:.3f}"
                      f"  ({time.time()-t0:.1f}s)")
                seed_rows.append({
                    "seed":       seed,
                    "status":     "ok",
                    "mct_mean":   agg["mct_mean"],
                    "mct_std":    agg["mct_std"],
                    "starve_pct": agg["starve_pct"],
                    "vrfi":       agg["vrfi"],
                })

            if cell_eps:
                cell_agg = _aggregate(cell_eps)
                meets_goal = (
                    cell_agg["starve_pct"] < mlfq_starve
                    and cell_agg["mct_mean"] <= mct_ceiling
                )
                all_results[(lam, thr)] = {
                    "lambda":      float(lam),
                    "threshold":   float(thr),
                    "mct_mean":    cell_agg["mct_mean"],
                    "mct_std":     cell_agg["mct_std"],
                    "starve_pct":  cell_agg["starve_pct"],
                    "vrfi":        cell_agg["vrfi"],
                    "n_episodes":  cell_agg["n_episodes"],
                    "meets_goal":  meets_goal,
                    "per_seed":    seed_rows,
                }
                marker = " *** GOAL MET ***" if meets_goal else ""
                print(f"  → λ={lam}  thresh={thr}  "
                      f"MCT={cell_agg['mct_mean']:.2f}s  "
                      f"starve={cell_agg['starve_pct']:.1f}%{marker}")
            print()

    # ── Grids ─────────────────────────────────────────────────────────────
    starve_vals: dict[tuple, str] = {}
    mct_vals:    dict[tuple, str] = {}

    for lam in LAMBDAS:
        for thr in THRESHOLDS:
            key = (lam, thr)
            if key not in all_results:
                starve_vals[key] = "n/a"
                mct_vals[key]    = "n/a"
                continue
            r  = all_results[key]
            s  = r["starve_pct"]
            m  = r["mct_mean"]
            vs = f"{s:.1f}%"
            vm = f"{(m - mlfq_mct) / mlfq_mct * 100:+.1f}%"
            # Flag cells that meet the full goal
            if r["meets_goal"]:
                vs += "*"
                vm += "*"
            starve_vals[key] = vs
            mct_vals[key]    = vm

    _print_grid(
        "Starvation rate  (* = meets goal: starve < MLFQ and MCT within 10%)",
        starve_vals, LAMBDAS, THRESHOLDS,
        f"MLFQ: {mlfq_starve:.1f}%",
    )

    _print_grid(
        "MCT vs MLFQ  (positive = worse than MLFQ)",
        mct_vals, LAMBDAS, THRESHOLDS,
        f"MLFQ: {mlfq_mct:.2f}s  (ceiling: {mct_ceiling:.2f}s = +10%)",
    )

    # ── Goal summary ──────────────────────────────────────────────────────
    print()
    goal_cells = [(lam, thr) for (lam, thr), r in all_results.items()
                  if r.get("meets_goal")]
    if goal_cells:
        print("Goal met by:")
        for lam, thr in sorted(goal_cells):
            r = all_results[(lam, thr)]
            print(f"  λ={lam}  thresh={thr}  "
                  f"MCT={r['mct_mean']:.2f}s ({(r['mct_mean']-mlfq_mct)/mlfq_mct*100:+.1f}%)  "
                  f"starve={r['starve_pct']:.1f}%  "
                  f"VRFI={r['vrfi']:.3f}")
    else:
        print("No (lambda, threshold) combination met the full goal "
              f"(starve < {mlfq_starve:.1f}% AND MCT ≤ {mct_ceiling:.2f}s).")
        # Show nearest: best starvation within MCT budget
        candidates = [(lam, thr) for (lam, thr), r in all_results.items()
                      if r["mct_mean"] <= mct_ceiling]
        if candidates:
            best = min(candidates, key=lambda k: all_results[k]["starve_pct"])
            r    = all_results[best]
            print(f"Best starvation within MCT budget: "
                  f"λ={best[0]}  thresh={best[1]}  "
                  f"starve={r['starve_pct']:.1f}%  MCT={r['mct_mean']:.2f}s")

    # ── Save summary.json ─────────────────────────────────────────────────
    summary = {
        "n_processes":   N,
        "n_eval":        args.n_eval,
        "seeds":         SEEDS,
        "mlfq_baseline": {
            "source":     mlfq_source,
            "mct_mean":   mlfq_mct,
            "starve_pct": mlfq_starve,
            "vrfi":       mlfq_vrfi,
        },
        "mct_ceiling":   mct_ceiling,
        "goal_cells":    [{"lambda": lam, "threshold": thr}
                          for lam, thr in goal_cells],
        "results": {
            f"lambda{lam}_thresh{thr}": all_results.get((lam, thr), {"status": "missing"})
            for lam in LAMBDAS for thr in THRESHOLDS
        },
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
