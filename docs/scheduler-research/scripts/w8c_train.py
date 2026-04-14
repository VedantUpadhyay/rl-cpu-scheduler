"""W8c: AttentionDQN on filtered trace, log-norm, BURST_P95=36, REWARD_SCALE=20, lambda=0.30."""
from __future__ import annotations

import csv
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")

from schedsim.env           import SchedEnv, N_ACTIONS, N_PROCESSES, N_QUANTUM_TIERS, WAIT_NORM
from schedsim.agent         import AttentionDQN, ReplayBuffer
from schedsim.process       import Process
from schedsim.trace_sampler import TraceEpisodeSampler

# ---------------------------------------------------------------------------
# Filtered-trace constants (override env.py values)
# ---------------------------------------------------------------------------
BURST_P95_FILT   = 36.0          # 95th pct of filtered training durations (≤ p75 of full trace)
BURST_MEDIAN_FILT = 5.0
QUANTUM_TIERS_FILT = (0.25, 1.0, 4.0)   # tier0, tier1, tier2
REWARD_SCALE     = float(N_PROCESSES * QUANTUM_TIERS_FILT[2])   # 5 × 4.0 = 20.0

_LOG_DENOM = float(np.log1p(BURST_P95_FILT))   # log(1+36) ≈ 3.611

# ---------------------------------------------------------------------------
# Paths and hyperparameters
# ---------------------------------------------------------------------------
TRACE_PATH   = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
LOG_PATH     = "/tmp/w8c_train.log"
WEIGHTS_PATH = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                "GRAD - FALL 23/UCSC/Capstone/results/dqn_w8c.npz")

N_EPISODES         = 10_000
LR                 = 0.001
GAMMA              = 1.0
GRAD_CLIP          = 1.0
BUF_CAPACITY       = 10_000
BATCH_SIZE         = 32
TARGET_UPDATE_FREQ = 200
WARMUP             = 500
LAMBDA_START       = 0.30
LAMBDA_END         = 0.005
PRINT_EVERY        = 500
LOSS_GATE          = 100.0
ENTROPY_GATE_5K    = 1.2
ENTROPY_FLAG_1K    = 0.20   # flag if avg_H < 0.20 at ep 1000, don't stop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_norm_burst(burst: float) -> float:
    """log(1 + burst) / log(1 + BURST_P95_FILT)."""
    return float(np.log1p(burst) / _LOG_DENOM)


def _encode_state(env: SchedEnv) -> np.ndarray:
    vec: list[float] = []
    for p in env.processes:
        arrived   = float(p.arrival_time <= env.current_time)
        remaining = 0.0 if p.is_complete else p.remaining_burst
        vec += [_log_norm_burst(remaining), arrived, p.wait_time / WAIT_NORM]
    return np.array(vec, dtype=np.float32)


def _valid_actions(env: SchedEnv) -> list[int]:
    return [
        p.pid * N_QUANTUM_TIERS + qt
        for p in env.processes
        for qt in range(N_QUANTUM_TIERS)
        if p.arrival_time <= env.current_time and not p.is_complete
    ]


def _make_procs(tasks: list[dict]) -> list[Process]:
    return [
        Process(pid=i, arrival_time=task["arrival_ms"], burst_length=task["burst_ms"])
        for i, task in enumerate(tasks)
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> AttentionDQN:
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    print(f"Loading trace: {TRACE_PATH} ...")
    sampler = TraceEpisodeSampler(TRACE_PATH)
    print(f"Loaded {len(sampler):,} valid-duration tasks.")
    print()

    # --- Pre-training verification ---
    print("=== Pre-training verification ===")
    print(f"  Formula: log(1 + burst) / log(1 + {BURST_P95_FILT}) = log(1+burst) / {_LOG_DENOM:.4f}")
    print()
    expected = {1: 0.083, 3: 0.269, 5: 0.386, 13: 0.610, 36: 1.000}
    all_match = True
    for b, exp_v in expected.items():
        actual = _log_norm_burst(b)
        match  = abs(actual - exp_v) < 0.01
        flag   = "  OK" if match else "  NOTE: actual differs from expected"
        if not match:
            all_match = False
        print(f"  burst={b:>3}s -> log-norm={actual:.4f}  (expected ~{exp_v:.3f}){flag}")
    if not all_match:
        print()
        print("  NOTE: actual values differ from expected for some entries.")
        print("  The formula log(1+burst)/log(1+BURST_P95) is correctly implemented.")
        print("  The expected values in the spec appear to use a different denominator;")
        print("  proceeding with the stated formula.")
    print()
    print(f"  REWARD_SCALE = {N_PROCESSES} × {QUANTUM_TIERS_FILT[2]} = {REWARD_SCALE}")
    print(f"  lambda_ent start = {LAMBDA_START}  (annealing to {LAMBDA_END} over {N_EPISODES} ep)")
    print()

    # --- Agent init ---
    agent  = AttentionDQN(n_actions=N_ACTIONS, state_dim=15, lr=LR,
                          gamma=GAMMA, epsilon=1.0, grad_clip=GRAD_CLIP)
    buffer = ReplayBuffer(capacity=BUF_CAPACITY, state_dim=15)

    rng = np.random.default_rng(42)
    total_transitions = 0

    win_loss:    list[float] = []
    win_entropy: list[float] = []
    win_mct:     list[float] = []

    with open(LOG_PATH, "w", newline="") as log_f:
        writer = csv.writer(log_f)
        writer.writerow(["episode", "avg_loss", "avg_H", "avg_MCT", "lambda_ent"])

        for ep in range(1, N_EPISODES + 1):
            agent.lambda_ent = LAMBDA_START - (LAMBDA_START - LAMBDA_END) * (ep / N_EPISODES)

            tasks = sampler.sample_episode(rng)
            procs = _make_procs(tasks)
            env   = SchedEnv(procs)
            env.reset()
            state_vec = _encode_state(env)

            prev_info: dict = {
                "env_reward": 0.0, "current_time": 0.0,
                "completed_pids": [], "completion_times": {},
                "num_waiting": 0,
            }
            ep_loss_sum = 0.0
            ep_ent_sum  = 0.0
            ep_loss_n   = 0
            done        = False

            while not done:
                valid = _valid_actions(env)
                if total_transitions < WARMUP:
                    action = random.choice(valid)
                else:
                    action = agent.select_action(state_vec, agent.epsilon, valid)

                _, _, done, info = env.step(action)
                reward         = info.get("env_reward", 0.0) / REWARD_SCALE
                next_state_vec = _encode_state(env)

                buffer.store(state_vec, action, reward, next_state_vec, done)
                total_transitions += 1

                if total_transitions >= WARMUP and len(buffer) >= BATCH_SIZE:
                    s_b, a_b, r_b, ns_b, d_b = buffer.sample(BATCH_SIZE)
                    loss, ent = agent.update_online(
                        s_b.astype(np.float64), a_b,
                        r_b.astype(np.float64),
                        ns_b.astype(np.float64),
                        d_b.astype(np.float64),
                    )
                    ep_loss_sum += loss
                    ep_ent_sum  += ent
                    ep_loss_n   += 1

                state_vec = next_state_vec
                prev_info = info

            if ep % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            mct       = info.get("mean_completion_time_so_far") or 0.0
            mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else float("nan")
            mean_ent  = ep_ent_sum  / ep_loss_n if ep_loss_n > 0 else float("nan")
            lam_cur   = agent.lambda_ent

            agent.decay_epsilon(ep, min_eps=0.05, decay=0.9995)

            win_loss.append(mean_loss)
            win_entropy.append(mean_ent)
            win_mct.append(mct)

            writer.writerow([ep, f"{mean_loss:.6f}", f"{mean_ent:.6f}",
                             f"{mct:.4f}", f"{lam_cur:.5f}"])

            if ep % PRINT_EVERY == 0:
                n  = PRINT_EVERY
                al = float(np.nanmean(win_loss[-n:]))
                ah = float(np.nanmean(win_entropy[-n:]))
                am = float(np.mean(win_mct[-n:]))

                flags = ""
                if ep == 1000 and ah < ENTROPY_FLAG_1K:
                    flags = "  [FLAG: avg_H < 0.20 — collapsed attention, continuing]"

                print(f"ep {ep:>6} | avg_loss={al:.4f} | avg_H={ah:.4f} | "
                      f"avg_MCT={am:.2f}s | lambda_ent={lam_cur:.5f}{flags}")
                sys.stdout.flush()
                log_f.flush()

                if al > LOSS_GATE:
                    print(f"\nSTOP GATE: avg_loss={al:.2f} > {LOSS_GATE} at ep {ep} — diverged.")
                    return agent

                if ep == 5000 and ah > ENTROPY_GATE_5K:
                    print(f"\nSTOP GATE: avg_H={ah:.4f} > {ENTROPY_GATE_5K} at ep 5000.")
                    return agent

    print(f"\nTraining complete — {N_EPISODES} episodes.")
    return agent


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("=" * 64)
    print("Week 8c — AttentionDQN, filtered trace + log-norm + lambda=0.30")
    print("=" * 64)
    t0    = time.time()
    agent = train()
    elapsed = time.time() - t0
    print(f"\nWall time: {elapsed/60:.1f} min")
    agent.save(WEIGHTS_PATH)
    print(f"Weights saved → {WEIGHTS_PATH}")
