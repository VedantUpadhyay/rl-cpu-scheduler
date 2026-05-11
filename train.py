"""RL CPU Scheduler — top-level training entry point.

Usage:
    python train.py --n_processes 5  --n_episodes 10    --seed 42
    python train.py --n_processes 10 --n_episodes 20000 --seed 123 \\
                    --trace_path /data/alibaba2018/ \\
                    --output_dir /outputs/n10_seed123/

N_PROCESSES flows as:
  CLI --n_processes
       └─▶ W15Trainer(n_processes)  ─▶ W15OmegaDQN.n_processes / n_actions
       └─▶ SchedEnv(procs)          ─▶  env.n_processes / n_actions (inferred)
       └─▶ OmegaReplayBuffer(state_dim = N * D_CAND)
       └─▶ TraceEpisodeSamplerN(n_processes)

Trace file convention:
  --trace_path may be a directory (train.py appends trace_train_filtered.csv)
  or a full file path.  Defaults to project_config.TRACE_PATH.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — project root + scripts dir both on sys.path
# ---------------------------------------------------------------------------
_PROJ    = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(_PROJ, "docs", "scheduler-research", "scripts")
sys.path.insert(0, _PROJ)
sys.path.insert(0, _SCRIPTS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL CPU Scheduler training")
    p.add_argument("--n_processes", type=int, default=5,
                   help="Number of processes in the scheduling window (default: 5)")
    p.add_argument("--n_episodes",  type=int, default=10_000,
                   help="Training episodes (default: 10 000)")
    p.add_argument("--seed",        type=int, default=42,
                   help="RNG seed (default: 42)")
    p.add_argument("--trace_path",  type=str, default=None,
                   help="Path to trace file or directory containing "
                        "trace_train_filtered.csv  (default: project_config.TRACE_PATH)")
    p.add_argument("--output_dir",  type=str, default=None,
                   help="Directory for checkpoint and results JSON "
                        "(default: results/train_n<N>_seed<SEED>/)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Generalised trace sampler — works for any N >= 1
# ---------------------------------------------------------------------------

_BURST_P95_FILT = 36.0   # matches w9_train.BURST_P95_FILT


class TraceEpisodeSamplerN:
    """Sample N-task episodes from the Alibaba trace for any N >= 1.

    Arrival slots are evenly spaced 2 s apart: (0, 2, 4, ..., 2*(N-1)).
    For N == 5 this matches TraceEpisodeSampler5._ARRIVE_SLOTS = (0,2,5,8,10)
    in spirit (same spread), with the difference that slots are uniform
    rather than the original hand-picked values.
    """

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
        print(f"  TraceEpisodeSamplerN(N={n_processes}): {len(self._data):,} tasks loaded.")

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
# Per-process state encoder (7-dim W15 features)
# ---------------------------------------------------------------------------

D_CAND = 7   # features per slot — fixed by W15 architecture
AFI    = 6   # arrived_flag index


def _encode_state(env, tasks: list[dict], n_processes: int,
                  norm_fns: dict) -> np.ndarray:
    """Build N×7 = state_dim float32 vector from SchedEnv."""
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
            vec[off + 6] = 1.0          # arrived_flag
    return vec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    N    = args.n_processes

    # --- Late imports (network module uses N_PROCESSES as default only) ----
    import torch
    from project_config import TRACE_PATH as _CFG_TRACE, get_agent_dir
    from schedsim.env    import SchedEnv
    from w15_network_torch import W15Trainer, device
    from w9_train import (
        _make_procs, _valid_actions,
        N_QT, _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem,
        WAIT_NORM,
    )
    from w14_omega import OmegaReplayBuffer
    from ablation_multiseed import (
        LAMBDA_START, LAMBDA_END,
        BUF_CAPACITY, BATCH_SIZE, TARGET_UPDATE_FREQ, WARMUP,
    )

    STATE_DIM = N * D_CAND
    N_ACTIONS  = N * N_QT

    # --- Resolve trace file path -------------------------------------------
    if args.trace_path is not None:
        tp = args.trace_path
        # Accept both a directory and a direct file path
        if os.path.isdir(tp):
            trace_file = os.path.join(tp, "trace_train_filtered.csv")
        else:
            trace_file = tp
    else:
        trace_file = _CFG_TRACE

    # --- Resolve output directory ------------------------------------------
    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        out_dir = get_agent_dir(f"train_n{N}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"n_processes={N}  state_dim={STATE_DIM}  n_actions={N_ACTIONS}")
    print(f"n_episodes={args.n_episodes}  seed={args.seed}  device={device}")
    print(f"trace_file={trace_file}")
    print(f"output_dir={out_dir}")

    # --- Sampler -----------------------------------------------------------
    sampler = TraceEpisodeSamplerN(trace_file, n_processes=N)

    norm_fns = {
        "time":      _norm_time_log,
        "wait_norm": WAIT_NORM,
        "urgency":   _urgency_norm,
        "cpu":       _norm_cpu,
        "mem":       _norm_mem,
    }

    # --- Seed --------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- Agent + replay buffer — both parameterised by N ------------------
    agent  = W15Trainer(n_processes=N)
    buffer = OmegaReplayBuffer(capacity=BUF_CAPACITY, state_dim=STATE_DIM)

    total_transitions = 0
    ep_mcts: list[float] = []
    print(f"\nTraining ({args.n_episodes} episodes)…")

    for ep in range(1, args.n_episodes + 1):
        agent.lambda_ent = (LAMBDA_START
                            - (LAMBDA_START - LAMBDA_END) * ep / args.n_episodes)
        omega_s = random.random()

        tasks = sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()
        sv    = _encode_state(env, tasks, N, norm_fns)
        done  = False
        ep_loss_sum = 0.0
        ep_loss_n   = 0

        while not done:
            valid  = _valid_actions(env)
            action = agent.select_action(sv, valid, omega_s)
            _, reward, done, info = env.step(action)
            sv_next = _encode_state(env, tasks, N, norm_fns)

            buffer.store(sv, action, float(reward), 0.0, sv_next, done, omega_s)
            total_transitions += 1

            if total_transitions >= WARMUP and len(buffer) >= BATCH_SIZE:
                s_b, a_b, rv_b, rs_b, ns_b, d_b, om_b = buffer.sample(BATCH_SIZE)
                loss, _ent = agent.update(s_b, a_b, rv_b, rs_b, ns_b, d_b, om_b)
                ep_loss_sum += loss
                ep_loss_n   += 1

            sv = sv_next

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()
        agent.decay_epsilon(ep, n_eps=args.n_episodes)

        mct = info.get("mean_completion_time_so_far") or 0.0
        ep_mcts.append(mct)

        log_every = max(1, args.n_episodes // 10)
        if ep % log_every == 0 or ep == args.n_episodes:
            mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n else float("nan")
            mean_mct  = float(np.mean(ep_mcts[-100:]))
            print(f"  ep {ep:>6} | loss={mean_loss:.4f} | eps={agent.epsilon:.3f} "
                  f"| MCT={mean_mct:.2f}s | transitions={total_transitions}")

    # --- Save checkpoint ---------------------------------------------------
    ckpt_path = os.path.join(out_dir, "final.pt")
    agent.save(ckpt_path)
    print(f"\nCheckpoint → {ckpt_path}")

    # --- Save results summary ----------------------------------------------
    results = {
        "n_processes":       N,
        "n_episodes":        args.n_episodes,
        "seed":              args.seed,
        "total_transitions": total_transitions,
        "final_mct_mean":    float(np.mean(ep_mcts[-100:])) if ep_mcts else None,
        "final_mct_std":     float(np.std(ep_mcts[-100:]))  if ep_mcts else None,
    }
    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results    → {results_path}")

    # --- Q-value smoke check -----------------------------------------------
    s_check = np.zeros(STATE_DIM, dtype=np.float32)
    s_check[AFI] = 1.0   # process 0 arrived_flag = 1
    s_t = torch.from_numpy(s_check).float().unsqueeze(0).to(device)
    o_t = torch.tensor([0.5], dtype=torch.float32, device=device)
    with torch.no_grad():
        q = agent.online(s_t, o_t).squeeze(0).cpu().numpy()
    valid_q = q[:N_QT]
    assert q.shape == (N_ACTIONS,), f"expected ({N_ACTIONS},), got {q.shape}"
    assert np.all(np.isfinite(valid_q)), "NaN/Inf in valid-action Q-values"
    print(f"\nQ-values (p0, omega=0.5): {valid_q}  shape={q.shape}  ✓")


if __name__ == "__main__":
    main()
