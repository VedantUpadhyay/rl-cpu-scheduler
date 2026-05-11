"""RL CPU Scheduler — top-level training entry point.

Usage:
    python train.py --n_processes 5 --n_episodes 10 --seed 42

N_PROCESSES flows as:
  CLI --n_processes
       └─▶ W15Trainer(n_processes)  ─▶ W15OmegaDQN.n_processes / n_actions
       └─▶ SchedEnv(procs)          ─▶  env.n_processes / n_actions  (inferred from list length)
       └─▶ OmegaReplayBuffer(state_dim = N * D_CAND)
"""
from __future__ import annotations
import argparse
import math
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
    return p.parse_args()


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
    from project_config import TRACE_PATH, N_PROCESSES as _CFG_DEFAULT
    from schedsim.env    import SchedEnv, N_QUANTUM_TIERS
    from schedsim.process import Process
    from w15_network_torch import W15Trainer, device
    from w9_train import (
        TraceEpisodeSampler5, _make_procs, _valid_actions,
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

    print(f"n_processes={N}  state_dim={STATE_DIM}  n_actions={N_ACTIONS}")
    print(f"n_episodes={args.n_episodes}  seed={args.seed}  device={device}")

    # Sampler — TraceEpisodeSampler5 has exactly 5 fixed arrival slots.
    # For N == 5 use it directly; other values require a generalized sampler.
    if N != 5:
        raise NotImplementedError(
            f"TraceEpisodeSampler5 encodes exactly 5 arrival slots; "
            f"a generalized sampler for N={N} is not yet implemented. "
            f"Use --n_processes 5 for now."
        )
    sampler = TraceEpisodeSampler5(TRACE_PATH)

    # Helpers for state encoding
    norm_fns = {
        "time":      _norm_time_log,
        "wait_norm": WAIT_NORM,
        "urgency":   _urgency_norm,
        "cpu":       _norm_cpu,
        "mem":       _norm_mem,
    }

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Agent + replay buffer — both parameterised by N
    agent  = W15Trainer(n_processes=N)
    buffer = OmegaReplayBuffer(capacity=BUF_CAPACITY, state_dim=STATE_DIM)

    total_transitions = 0
    print(f"\nTraining ({args.n_episodes} episodes)…")

    for ep in range(1, args.n_episodes + 1):
        # Anneal entropy coefficient
        agent.lambda_ent = (LAMBDA_START
                            - (LAMBDA_START - LAMBDA_END) * ep / args.n_episodes)
        omega_s = random.random()

        tasks = sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)          # env.n_processes == N (asserted internally)
        env.reset()
        sv    = _encode_state(env, tasks, N, norm_fns)
        done  = False
        ep_loss_sum = 0.0
        ep_loss_n   = 0

        while not done:
            valid  = _valid_actions(env)
            action = agent.select_action(sv, valid, omega_s)
            _, reward, done, _ = env.step(action)
            sv_next = _encode_state(env, tasks, N, norm_fns)

            # Use env step-reward as value-delta proxy; starvation component = 0
            buffer.store(sv, action, float(reward), 0.0, sv_next, done, omega_s)
            total_transitions += 1

            if (total_transitions >= WARMUP and len(buffer) >= BATCH_SIZE):
                s_b, a_b, rv_b, rs_b, ns_b, d_b, om_b = buffer.sample(BATCH_SIZE)
                loss, _ent = agent.update(s_b, a_b, rv_b, rs_b, ns_b, d_b, om_b)
                ep_loss_sum += loss
                ep_loss_n   += 1

            sv = sv_next

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()
        agent.decay_epsilon(ep, n_eps=args.n_episodes)

        log_every = max(1, args.n_episodes // 5)
        if ep % log_every == 0 or ep == args.n_episodes:
            mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n else float("nan")
            print(f"  ep {ep:>5} | loss={mean_loss:.4f} | eps={agent.epsilon:.3f} "
                  f"| transitions={total_transitions}")

    # -----------------------------------------------------------------------
    # Q-value smoke check — run a forward pass with a synthetic valid state
    # -----------------------------------------------------------------------
    print("\n--- Q-value smoke check ---")
    # Build a state where process 0 is valid (arrived_flag = 1) and others are not
    s_check = np.zeros(STATE_DIM, dtype=np.float32)
    s_check[AFI] = 1.0                 # process 0 arrived_flag = 1

    s_t  = torch.from_numpy(s_check).float().unsqueeze(0).to(device)
    o_t  = torch.tensor([0.5], dtype=torch.float32, device=device)
    with torch.no_grad():
        q = agent.online(s_t, o_t).squeeze(0).cpu().numpy()

    valid_q = q[:N_QT]                 # first N_QT actions (process 0, all quanta)
    print(f"  n_processes={N}  state_dim={STATE_DIM}  n_actions={N_ACTIONS}")
    print(f"  Q-values for process-0 actions (quanta 0,1,2): {valid_q}")
    print(f"  Q-values shape: {q.shape}  finite: {np.all(np.isfinite(valid_q))}")
    assert q.shape == (N_ACTIONS,), f"expected ({N_ACTIONS},), got {q.shape}"
    assert np.all(np.isfinite(valid_q)), "NaN/Inf in valid-action Q-values"
    print("Smoke test PASSED.")


if __name__ == "__main__":
    main()
