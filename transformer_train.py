"""Transformer scheduler training entry point.

Drop-in replacement for train.py using TransformerSchedulerNet.
Training loop is identical to train.py; differences are noted below.

Differences from train.py:
  - Agent: TransformerTrainer (transformer_agent.py) instead of W15Trainer
  - Buffer: internal to TransformerTrainer; states zero-padded to N_MAX*7
  - N mode: --fixed_n (default 5) or --variable_n (samples {5,10,20,50} each ep)
  - omega_s: fixed scalar via --omega_s (default 0.5) instead of random per episode
  - Checkpoints: saved every 2000 episodes + final.pt
  - results.json: same format + n_mode and omega_s fields; per-N MCT for variable mode
  - reward_lambda / wait_threshold: optional starvation penalty added to env reward
      reward = value_delta - lambda * sum(max(0, wait_i - threshold) for i in queue)
      When reward_lambda=0.0 (default), behavior is identical to the original.

Usage:
  # Fixed N — comparable to train.py results:
  python transformer_train.py --fixed_n 5 --n_episodes 20000 --seed 42

  # Variable N curriculum:
  python transformer_train.py --variable_n --n_episodes 20000 --seed 42

  # With starvation penalty:
  python transformer_train.py --fixed_n 5 --n_episodes 20000 --seed 42 \\
    --reward_lambda 0.05 --wait_threshold 100.0

  # Smoke test:
  python transformer_train.py --fixed_n 5 --n_episodes 100 --seed 42 \\
    --trace_path /path/to/trace --output_dir /tmp/smoke_test/
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJ    = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(_PROJ, "docs", "scheduler-research", "scripts")
sys.path.insert(0, _PROJ)
sys.path.insert(0, _SCRIPTS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transformer RL scheduler training")

    # N mode — mutually exclusive
    n_group = p.add_mutually_exclusive_group()
    n_group.add_argument("--fixed_n", type=int, default=None,
                         metavar="N",
                         help="Fixed scheduling window size (default: 5)")
    n_group.add_argument("--variable_n", action="store_true",
                         help="Sample N from {5,10,20,50} each episode")

    # Shared with train.py
    p.add_argument("--n_episodes",  type=int, default=20_000)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--trace_path",  type=str, default=None,
                   help="Trace file or directory containing "
                        "trace_train_filtered.csv")
    p.add_argument("--output_dir",  type=str, default=None)

    # Transformer-specific
    p.add_argument("--omega_s", type=float, default=0.5,
                   help="Fixed omega conditioning scalar in [0,1] (default: 0.5)")

    # Reward shaping
    p.add_argument("--reward_lambda", type=float, default=0.0,
                   help="Weight of starvation penalty term (0.0 = original reward)")
    p.add_argument("--wait_threshold", type=float, default=100.0,
                   help="Wait-time threshold in seconds for starvation penalty (default: 100.0)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # Resolve N mode
    if args.variable_n:
        n_mode = "variable"
        default_n = 5   # used only for output_dir naming when variable
    else:
        n_mode = "fixed"
        default_n = args.fixed_n if args.fixed_n is not None else 5

    # Late imports
    import torch
    from project_config       import TRACE_PATH as _CFG_TRACE, get_agent_dir
    from schedsim.env         import SchedEnv
    from w9_train             import (
        _make_procs, _valid_actions,
        _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem, WAIT_NORM,
    )
    from ablation_multiseed   import (
        LAMBDA_START, LAMBDA_END,
        BATCH_SIZE, TARGET_UPDATE_FREQ, WARMUP,
    )
    from transformer_agent    import (
        TransformerTrainer, N_OPTIONS, D_CAND, AFI,
        device,
    )
    from train import TraceEpisodeSamplerN, _encode_state

    norm_fns = {
        "time":      _norm_time_log,
        "wait_norm": WAIT_NORM,
        "urgency":   _urgency_norm,
        "cpu":       _norm_cpu,
        "mem":       _norm_mem,
    }

    # --- Resolve paths -------------------------------------------------------
    if args.trace_path is not None:
        tp = args.trace_path
        trace_file = os.path.join(tp, "trace_train_filtered.csv") \
                     if os.path.isdir(tp) else tp
    else:
        trace_file = _CFG_TRACE

    if args.output_dir is not None:
        out_dir = args.output_dir
    else:
        tag = "variable" if n_mode == "variable" else f"n{default_n}"
        out_dir = get_agent_dir(f"transformer_{tag}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Print config --------------------------------------------------------
    print(f"Agent      : TransformerSchedulerNet")
    print(f"N mode     : {n_mode}"
          + (f"  (options={N_OPTIONS})" if n_mode == "variable"
             else f"  (N={default_n})"))
    print(f"omega_s    : {args.omega_s} (fixed)")
    print(f"reward_lambda : {args.reward_lambda}  wait_threshold : {args.wait_threshold}s")
    print(f"n_episodes : {args.n_episodes}  seed={args.seed}  device={device}")
    print(f"trace_file : {trace_file}")
    print(f"output_dir : {out_dir}")

    # --- Build samplers (one per N to avoid reloading the CSV) ---------------
    ns_to_build = N_OPTIONS if n_mode == "variable" else [default_n]
    print(f"\nLoading trace for N={ns_to_build} …")
    samplers = {n: TraceEpisodeSamplerN(trace_file, n_processes=n)
                for n in ns_to_build}

    # --- Seed ----------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # --- Agent (buffer is internal to trainer) --------------------------------
    trainer = TransformerTrainer()
    print(f"\nTransformerSchedulerNet parameters: {trainer.n_params():,}")

    # --- Training loop -------------------------------------------------------
    total_transitions = 0
    ep_mcts: list[float] = []
    ep_n_log: list[int]  = []        # track which N was used each episode

    print(f"\nTraining ({args.n_episodes} episodes) …")

    for ep in range(1, args.n_episodes + 1):
        # Lambda entropy decay (same schedule as train.py)
        trainer.online.training = True
        trainer.lambda_ent = (LAMBDA_START
                              - (LAMBDA_START - LAMBDA_END) * ep / args.n_episodes)

        # Resolve N for this episode
        N_ep = TransformerTrainer.sample_n(N_OPTIONS) if n_mode == "variable" \
               else default_n
        sampler = samplers[N_ep]

        # Sample episode
        tasks = sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()
        sv    = _encode_state(env, tasks, N_ep, norm_fns)
        done  = False
        ep_loss_sum = 0.0
        ep_loss_n   = 0

        while not done:
            valid  = _valid_actions(env)
            action = trainer.select_action(sv, valid, args.omega_s)
            _, value_delta, done, info = env.step(action)
            sv_next = _encode_state(env, tasks, N_ep, norm_fns)

            # Optional starvation penalty over the post-step queue.
            # When reward_lambda=0.0 this branch is skipped entirely,
            # preserving identical behavior to the original reward.
            if args.reward_lambda > 0.0:
                queue = env._get_runnable()
                penalty = args.reward_lambda * sum(
                    max(0.0, p.wait_time - args.wait_threshold)
                    for p in queue
                )
                reward = value_delta - penalty
            else:
                reward = value_delta

            # store() pads sv / sv_next to STATE_DIM_MAX internally
            trainer.store(sv, action, float(reward), 0.0, sv_next, done,
                          args.omega_s)
            total_transitions += 1

            if total_transitions >= WARMUP and len(trainer) >= BATCH_SIZE:
                s_b, a_b, rv_b, rs_b, ns_b, d_b, om_b = trainer.sample(BATCH_SIZE)
                loss, _ent = trainer.update(s_b, a_b, rv_b, rs_b, ns_b, d_b, om_b)
                ep_loss_sum += loss
                ep_loss_n   += 1

            sv = sv_next

        if ep % TARGET_UPDATE_FREQ == 0:
            trainer.update_target()
        trainer.decay_epsilon(ep, n_eps=args.n_episodes)

        mct = info.get("mean_completion_time_so_far") or 0.0
        ep_mcts.append(mct)
        ep_n_log.append(N_ep)

        # --- Periodic checkpoint (every 2000 episodes) ---------------------
        if ep % 2000 == 0:
            ckpt = os.path.join(out_dir, f"transformer_checkpoint_{ep}.pt")
            trainer.save(ckpt)

        # --- Logging -------------------------------------------------------
        log_every = max(1, args.n_episodes // 10)
        if ep % log_every == 0 or ep == args.n_episodes:
            mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n else float("nan")
            mean_mct  = float(np.mean(ep_mcts[-100:]))
            n_tag     = f"N={N_ep}" if n_mode == "fixed" else \
                        f"N~{np.mean(ep_n_log[-100:]):.0f}"
            print(f"  ep {ep:>6} | loss={mean_loss:.4f} | eps={trainer.epsilon:.3f}"
                  f" | MCT={mean_mct:.2f}s | {n_tag}"
                  f" | trans={total_transitions}")

    # --- Final checkpoint ----------------------------------------------------
    final_ckpt = os.path.join(out_dir, "final.pt")
    trainer.save(final_ckpt)
    print(f"\nCheckpoint → {final_ckpt}")

    # --- Results JSON (same format as train.py + transformer fields) ---------
    results: dict = {
        "n_processes":       "variable" if n_mode == "variable" else default_n,
        "n_mode":            n_mode,
        "omega_s":           args.omega_s,
        "reward_lambda":     args.reward_lambda,
        "wait_threshold":    args.wait_threshold,
        "n_episodes":        args.n_episodes,
        "seed":              args.seed,
        "total_transitions": total_transitions,
        "final_mct_mean":    float(np.mean(ep_mcts[-100:])) if ep_mcts else None,
        "final_mct_std":     float(np.std(ep_mcts[-100:]))  if ep_mcts else None,
    }

    # Per-N breakdown for variable mode
    if n_mode == "variable":
        for n_val in N_OPTIONS:
            mcts_n = [m for m, n in zip(ep_mcts, ep_n_log) if n == n_val]
            tail   = mcts_n[-100:] if len(mcts_n) >= 100 else mcts_n
            results[f"final_mct_mean_n{n_val}"] = \
                float(np.mean(tail)) if tail else None
            results[f"final_mct_std_n{n_val}"] = \
                float(np.std(tail))  if tail else None

    results_path = os.path.join(out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results    → {results_path}")

    # --- Q-value sanity check ------------------------------------------------
    N_check   = default_n if n_mode == "fixed" else 5
    sd_check  = N_check * D_CAND
    s_check   = np.zeros(sd_check, dtype=np.float32)
    s_check[AFI] = 1.0   # process 0 arrived_flag = 1
    s_t = torch.from_numpy(s_check).float().unsqueeze(0).to(device)
    o_t = torch.tensor([args.omega_s], dtype=torch.float32, device=device)
    trainer.online.eval()
    with torch.no_grad():
        q = trainer.online(s_t, o_t).squeeze(0).cpu().numpy()
    assert q.shape == (N_check * 3,), f"Q shape mismatch: {q.shape}"
    assert np.all(np.isfinite(q[:3])), "NaN/Inf in valid Q-values"
    print(f"\nQ-values (p0 quanta, omega={args.omega_s}): {q[:3]}  ✓")


if __name__ == "__main__":
    main()
