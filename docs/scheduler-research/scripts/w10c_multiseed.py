"""W10C Multi-Seed Training and Evaluation.

Runs the paper's best agent (W10C: true remaining burst, value-delta reward,
2-head attention) with seeds 42, 123, 456 to establish statistical reliability.

Uses original AttentionDQN10C (full backprop) from Project/docs/ClaudeCode/w10c_train.py.
Only the weight initialization seed and the episode-sampling RNG are varied.

Each seed: 10,000 training episodes, then 100 eval episodes (epsilon=0).
Reports MCT and SRPT agreement per seed and mean ± std across seeds.
"""
from __future__ import annotations
import os, random, sys, time
import numpy as np

_PROJECT_ROOT = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                 "GRAD - FALL 23/UCSC/Capstone")
_SCRIPTS_DIR  = os.path.join(_PROJECT_ROOT, "Project", "docs", "ClaudeCode")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, "/tmp")

from schedsim.env    import SchedEnv, N_PROCESSES
from schedsim.agent  import ReplayBuffer
from w9_train import (
    TraceEpisodeSampler5,
    _encode_state, _valid_actions, _make_procs,
    D_CAND, N_QT, N_ACTIONS,
    REWARD_SCALE,
)
from w10c_train import (
    AttentionDQN10C,
    D_ATTN, N_HEADS, D_HEAD, D_V_TOT,
    N_EPISODES, LR, GAMMA, GRAD_CLIP,
    BUF_CAPACITY, BATCH_SIZE, TARGET_UPDATE_FREQ, WARMUP,
    LAMBDA_START, LAMBDA_END,
    LOSS_GATE, ENT_GATE_5K,
)

TRACE_TRAIN = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
SEEDS       = [42, 123, 456]
N_EVAL      = 100


# ---------------------------------------------------------------------------
# Weight re-initializer — overwrite all weights in-place with a new seed
# ---------------------------------------------------------------------------

def reinit_weights(agent: AttentionDQN10C, seed: int) -> None:
    """Replace all trainable weights in-place using He initialization with given seed."""
    rng = np.random.default_rng(seed)

    def he(fan_in: int, *shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float64) * np.sqrt(2.0 / fan_in)

    for h in range(N_HEADS):
        agent.W_Q[h][:] = he(D_CAND,   D_CAND, D_HEAD)
        agent.b_Q[h][:] = 0.0
        agent.W_K[h][:] = he(D_CAND,   D_CAND, D_HEAD)
        agent.b_K[h][:] = 0.0
        agent.W_V[h][:] = he(D_CAND,   D_CAND, D_HEAD)
        agent.b_V[h][:] = 0.0

    agent.W_O[:] = he(D_V_TOT, D_V_TOT, D_V_TOT)
    agent.b_O[:] = 0.0

    D_MLP_IN = D_V_TOT + D_CAND + 1  # 22
    agent._W[0][:] = he(D_MLP_IN, D_MLP_IN, 64)
    agent._b[0][:] = 0.0
    agent._W[1][:] = he(64,       64,        32)
    agent._b[1][:] = 0.0
    agent._W[2][:] = he(32,       32,         1)
    agent._b[2][:] = 0.0

    agent.update_target()   # sync target network to new weights


# ---------------------------------------------------------------------------
# Training for one seed
# ---------------------------------------------------------------------------

def train_one_seed(seed: int, sampler) -> AttentionDQN10C:
    """Train W10C for 10k episodes with given seed. Returns trained agent."""
    print(f"\n{'='*60}")
    print(f"Training seed={seed}")
    print(f"{'='*60}")

    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    agent = AttentionDQN10C(lr=LR, gamma=GAMMA, grad_clip=GRAD_CLIP)
    reinit_weights(agent, seed)

    buffer = ReplayBuffer(capacity=BUF_CAPACITY, state_dim=N_PROCESSES * D_CAND)

    total_transitions = 0
    win_loss: list[float] = []
    win_ent:  list[float] = []
    win_mct:  list[float] = []

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
            valid = _valid_actions(env)
            if total_transitions < WARMUP:
                action = random.choice(valid)
            else:
                action = agent.select_action(sv, agent.epsilon, valid)

            _, _, done, info = env.step(action)
            reward  = info.get("env_reward", 0.0) / REWARD_SCALE
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

        mct      = info.get("mean_completion_time_so_far") or 0.0
        avg_loss = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else float("nan")
        avg_ent  = ep_ent_sum  / ep_loss_n if ep_loss_n > 0 else float("nan")

        agent.decay_epsilon(ep, min_eps=0.05, decay=0.9995)
        win_mct.append(mct)
        win_loss.append(avg_loss)
        win_ent.append(avg_ent)

        if ep in {500, 2000, 5000, 10000}:
            n  = min(ep, 500)
            al = float(np.nanmean(win_loss[-n:]))
            ah = float(np.nanmean(win_ent[-n:]))
            am = float(np.mean(win_mct[-n:]))
            lam = agent.lambda_ent
            print(f"ep {ep:>6} | avg_loss={al:.4f} | avg_H={ah:.4f} | "
                  f"avg_MCT={am:.2f}s | lambda_ent={lam:.5f}")
            sys.stdout.flush()

            if al > LOSS_GATE:
                print(f"STOP GATE: avg_loss={al:.2f} > {LOSS_GATE}")
                break
            if ep == 5000 and ah > ENT_GATE_5K:
                print(f"STOP GATE: avg_H={ah:.4f} > {ENT_GATE_5K}")
                break

    ckpt_path = os.path.join(RESULTS_DIR, f"w10c_seed{seed}_final.npz")
    agent.save(ckpt_path)
    print(f"Saved → {ckpt_path}")

    return agent


# ---------------------------------------------------------------------------
# Evaluation: 100 episodes, epsilon=0
# ---------------------------------------------------------------------------

def eval_agent(agent: AttentionDQN10C, sampler, eval_seed: int = 999) -> dict:
    """Run N_EVAL episodes with epsilon=0, return MCT and SRPT agreement stats."""
    rng  = np.random.default_rng(eval_seed)
    mcts = []
    srpt_fracs = []

    for _ in range(N_EVAL):
        tasks = sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()
        sv   = _encode_state(env, tasks)
        done = False

        agree_n = 0
        total_n = 0

        while not done:
            valid    = _valid_actions(env)
            runnable = [p for p in env.processes
                        if p.arrival_time <= env.current_time and not p.is_complete]
            if runnable:
                srpt_pid = min(runnable, key=lambda p: p.remaining_burst).pid
            else:
                srpt_pid = -1

            action     = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
            chosen_pid = action // N_QT

            if srpt_pid >= 0:
                agree_n += int(chosen_pid == srpt_pid)
                total_n += 1

            _, _, done, info = env.step(action)
            sv = _encode_state(env, tasks)

        mcts.append(info.get("mean_completion_time_so_far") or 0.0)
        srpt_fracs.append(agree_n / total_n if total_n > 0 else 0.0)

    return {
        "mct_mean":  float(np.mean(mcts)),
        "mct_std":   float(np.std(mcts)),
        "srpt_mean": float(np.mean(srpt_fracs)) * 100.0,
        "srpt_std":  float(np.std(srpt_fracs))  * 100.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> tuple[float, float, float, float]:
    print("=" * 60)
    print("W10C Multi-Seed Experiment (seeds: 42, 123, 456)")
    print("=" * 60)

    if not os.path.isfile(TRACE_TRAIN):
        print(f"ERROR: trace not found at {TRACE_TRAIN}")
        sys.exit(1)

    sampler = TraceEpisodeSampler5(TRACE_TRAIN)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    per_seed: list[dict] = []

    for seed in SEEDS:
        t0    = time.time()
        agent = train_one_seed(seed, sampler)
        elapsed = (time.time() - t0) / 60.0

        print(f"\nEvaluating seed={seed} ({N_EVAL} episodes, epsilon=0)...")
        ev = eval_agent(agent, sampler, eval_seed=999)

        per_seed.append({
            "seed":      seed,
            "mct_mean":  ev["mct_mean"],
            "mct_std":   ev["mct_std"],
            "srpt_mean": ev["srpt_mean"],
            "srpt_std":  ev["srpt_std"],
            "wall_min":  elapsed,
        })

        print(f"  Seed {seed}: MCT = {ev['mct_mean']:.2f} ± {ev['mct_std']:.2f} s, "
              f"SRPT = {ev['srpt_mean']:.1f} ± {ev['srpt_std']:.1f}%")

    # Summary table
    print(f"\n{'='*60}")
    print("Per-Seed Results (100 eval episodes each, epsilon=0)")
    print(f"{'='*60}")
    print(f"{'Seed':>6} | {'MCT (s)':>12} | {'SRPT%':>10} | {'Wall (min)':>10}")
    print("-" * 48)
    for r in per_seed:
        print(f"{r['seed']:>6} | {r['mct_mean']:>12.2f} | "
              f"{r['srpt_mean']:>10.1f} | {r['wall_min']:>10.1f}")

    mcts  = np.array([r["mct_mean"]  for r in per_seed])
    srpts = np.array([r["srpt_mean"] for r in per_seed])

    mct_mean  = float(mcts.mean());  mct_std  = float(mcts.std())
    srpt_mean = float(srpts.mean()); srpt_std = float(srpts.std())

    print(f"\nAcross {len(SEEDS)} seeds (mean ± std):")
    print(f"  MCT           : {mct_mean:.2f} ± {mct_std:.2f} s")
    print(f"  SRPT agreement: {srpt_mean:.1f} ± {srpt_std:.1f}%")

    sentence = (f"Results are reported as mean \\pm standard deviation across "
                f"3 independent training seeds: "
                f"MCT = {mct_mean:.2f} \\pm {mct_std:.2f} s, "
                f"SRPT agreement = {srpt_mean:.1f} \\pm {srpt_std:.1f}\\%.")

    print(f"\nPaper sentence for §5.20:")
    print(f'  "{sentence}"')

    print(f"\n{'='*60}")
    print("Experiment complete.")
    print(f"{'='*60}")

    return mct_mean, mct_std, srpt_mean, srpt_std


if __name__ == "__main__":
    main()
