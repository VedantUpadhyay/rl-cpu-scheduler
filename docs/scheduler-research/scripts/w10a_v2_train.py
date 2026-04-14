"""Week 10A-v2 — W9 + PBRS, alpha=0.05 (reduced from 0.20)."""
from __future__ import annotations
import csv, os, random, sys, time
import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")
sys.path.insert(0, "/tmp")

from schedsim.env    import SchedEnv, N_PROCESSES, N_QUANTUM_TIERS
from schedsim.agent  import AdamOptimizer, ReplayBuffer
from schedsim.process import Process
from w9_train import (
    AttentionDQN9, TraceEpisodeSampler5,
    _encode_state, _valid_actions, _make_procs,
    D_CAND, N_QT, N_ACTIONS,
    _LOG_DENOM, _norm_burst,
    REWARD_SCALE,
)

PBRS_ALPHA = 0.05   # reduced from 0.20
PBRS_GAMMA = 0.99

TRACE_PATH   = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
LOG_PATH     = "/tmp/w10a_v2_train.log"
WEIGHTS_PATH = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                "GRAD - FALL 23/UCSC/Capstone/results/dqn_w10a_v2.npz")

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

PRINT_AT    = {500, 1000, 2000, 5000, 10000}
LOSS_GATE   = 100.0
ENT_GATE_5K = 1.2


def phi(env: SchedEnv) -> float:
    total = sum(
        _norm_burst(p.remaining_burst)
        for p in env.processes
        if not p.is_complete and p.arrival_time <= env.current_time
    )
    return -PBRS_ALPHA * total


def verify_pbrs() -> None:
    print("=" * 60)
    print(f"PBRS Verification (alpha={PBRS_ALPHA})")
    print("=" * 60)
    norms_s  = [0.20, 0.50, 0.80]
    bursts_s = [float(np.expm1(n * _LOG_DENOM)) for n in norms_s]
    phi_s    = -PBRS_ALPHA * sum(norms_s)
    print(f"\n  Initial state — burst_norms: {norms_s}")
    print(f"  Actual bursts (s): {[f'{b:.3f}' for b in bursts_s]}")
    print(f"  Phi(s) = -{PBRS_ALPHA} × {sum(norms_s):.2f} = {phi_s:.4f}")

    # Scenario A: shortest completes with tier2 (4.0s)
    qt_a    = 4.0
    rem_a0  = max(0.0, bursts_s[0] - qt_a)
    norm_a0 = _norm_burst(rem_a0) if rem_a0 > 0 else 0.0
    phi_sp_a  = -PBRS_ALPHA * (norm_a0 + norms_s[1] + norms_s[2])
    shaping_a = PBRS_GAMMA * phi_sp_a - phi_s
    print(f"\n  Scenario A — run SHORTEST (burst={bursts_s[0]:.3f}s) with tier2 ({qt_a}s) → completes")
    print(f"  Phi(s') = -{PBRS_ALPHA} × {norm_a0 + norms_s[1] + norms_s[2]:.4f} = {phi_sp_a:.4f}")
    print(f"  Shaping = {PBRS_GAMMA} × ({phi_sp_a:.4f}) - ({phi_s:.4f}) = {shaping_a:+.4f}")

    # Scenario B: longest runs with tier0 (0.25s)
    qt_b    = 0.25
    rem_b2  = max(0.0, bursts_s[2] - qt_b)
    norm_b2 = _norm_burst(rem_b2)
    phi_sp_b  = -PBRS_ALPHA * (norms_s[0] + norms_s[1] + norm_b2)
    shaping_b = PBRS_GAMMA * phi_sp_b - phi_s
    print(f"\n  Scenario B — run LONGEST (burst={bursts_s[2]:.3f}s) with tier0 ({qt_b}s)"
          f" → burst_norm {norms_s[2]:.4f}→{norm_b2:.4f}")
    print(f"  Phi(s') = -{PBRS_ALPHA} × {norms_s[0] + norms_s[1] + norm_b2:.4f} = {phi_sp_b:.4f}")
    print(f"  Shaping = {PBRS_GAMMA} × ({phi_sp_b:.4f}) - ({phi_s:.4f}) = {shaping_b:+.4f}")

    print(f"\n  Shaping bonus: shortest completion = {shaping_a:+.4f}, "
          f"longest run = {shaping_b:+.4f}")
    print(f"  Shortest favoured? {shaping_a > shaping_b}  (required: True)")
    assert shaping_a > shaping_b, "PBRS direction check FAILED"
    print("=" * 60)


def train() -> AttentionDQN9:
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    verify_pbrs()
    print()

    print(f"Loading trace: {TRACE_PATH}")
    sampler = TraceEpisodeSampler5(TRACE_PATH)
    print()

    agent  = AttentionDQN9(lr=LR, gamma=GAMMA, grad_clip=GRAD_CLIP)
    buffer = ReplayBuffer(capacity=BUF_CAPACITY, state_dim=N_PROCESSES * D_CAND)

    rng = np.random.default_rng(42)
    total_transitions = 0
    win_loss:    list[float] = []
    win_entropy: list[float] = []
    win_mct:     list[float] = []

    with open(LOG_PATH, "w", newline="") as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow(["episode", "avg_loss", "avg_H", "avg_MCT", "lambda_ent"])

        for ep in range(1, N_EPISODES + 1):
            agent.lambda_ent = LAMBDA_START - (LAMBDA_START - LAMBDA_END) * (ep / N_EPISODES)

            tasks = sampler.sample_episode(rng)
            procs = _make_procs(tasks)
            env   = SchedEnv(procs)
            env.reset()
            sv    = _encode_state(env, tasks)

            ep_loss_sum = 0.0
            ep_ent_sum  = 0.0
            ep_loss_n   = 0
            done        = False

            while not done:
                valid     = _valid_actions(env)
                if total_transitions < WARMUP:
                    action = random.choice(valid)
                else:
                    action = agent.select_action(sv, agent.epsilon, valid)

                phi_s_val   = phi(env)
                _, _, done, info = env.step(action)
                phi_sp_val  = phi(env)

                r_original  = info.get("env_reward", 0.0) / REWARD_SCALE
                shaping     = PBRS_GAMMA * phi_sp_val - phi_s_val
                reward      = r_original + shaping

                sv_next = _encode_state(env, tasks)
                buffer.store(sv, action, reward, sv_next, done)
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

                sv = sv_next

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

            log_writer.writerow([ep, f"{mean_loss:.6f}", f"{mean_ent:.6f}",
                                 f"{mct:.4f}", f"{lam_cur:.5f}"])

            if ep in PRINT_AT:
                n  = min(ep, 500)
                al = float(np.nanmean(win_loss[-n:]))
                ah = float(np.nanmean(win_entropy[-n:]))
                am = float(np.mean(win_mct[-n:]))
                print(f"ep {ep:>6} | avg_loss={al:.4f} | avg_H={ah:.4f} | "
                      f"avg_MCT={am:.2f}s | lambda_ent={lam_cur:.5f}")
                sys.stdout.flush(); log_f.flush()

                if al > LOSS_GATE:
                    print(f"\nSTOP GATE: avg_loss={al:.2f} > {LOSS_GATE} at ep {ep}")
                    return agent
                if ep == 5000 and ah > ENT_GATE_5K:
                    print(f"\nSTOP GATE: avg_H={ah:.4f} > {ENT_GATE_5K} at ep 5000")
                    return agent

    print(f"\nTraining complete — {N_EPISODES} episodes.")
    return agent


if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    print("=" * 64)
    print(f"Week 10A-v2 — AttentionDQN9 + PBRS (alpha={PBRS_ALPHA}, gamma={PBRS_GAMMA})")
    print("=" * 64)
    t0    = time.time()
    agent = train()
    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
    agent.save(WEIGHTS_PATH)
    print(f"Weights saved → {WEIGHTS_PATH}")
