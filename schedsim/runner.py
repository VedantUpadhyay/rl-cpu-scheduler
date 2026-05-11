"""Training loop: connects env + agent + reward.

Week 3: QLearningAgent replaced by DQNAgent (numpy MLP) with experience
replay and a target network.  State is encoded as a 15-dim continuous
vector; the Q-table is replaced by a 15 → 64 → 32 → 15 network trained
with Adam and gradient clipping ±1.0.

Process set is still randomised each episode (Week 2 carry-over).
Coverage (distinct discrete states visited) is still tracked for
comparison with Week 2 results.
"""
from __future__ import annotations

import csv
import os
import random

import numpy as np

try:
    from schedsim.env       import SchedEnv, N_ACTIONS, N_PROCESSES, N_QUANTUM_TIERS, BURST_P95, WAIT_NORM
    from schedsim.agent     import (DQNAgent, ActionConditionedDQN,
                                    SumPoolingDQN, AttentionDQN,
                                    ReplayBuffer, QLearningAgent)
    from schedsim.process   import (Process,
                                    generate_random_processes,
                                    generate_fixed_processes,
                                    generate_indist_processes)
    from schedsim.reward    import compute_reward
    from schedsim.baselines import RoundRobin, FCFS
except ImportError:
    from env       import SchedEnv, N_ACTIONS, N_PROCESSES, N_QUANTUM_TIERS, BURST_P95, WAIT_NORM  # type: ignore
    from agent     import (DQNAgent, ActionConditionedDQN,                    # type: ignore
                           SumPoolingDQN, AttentionDQN,
                           ReplayBuffer, QLearningAgent)
    from process   import (Process,                                            # type: ignore
                           generate_random_processes,
                           generate_fixed_processes,
                           generate_indist_processes)
    from reward    import compute_reward                                       # type: ignore
    from baselines import RoundRobin, FCFS                                     # type: ignore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_PATH    = os.path.join("results", "training_log_w3.csv")
_LOG_PATH_W5 = os.path.join("results", "training_log_w5.csv")
_LOG_PATH_W6 = os.path.join("results", "training_log_w6.csv")
_LOG_PATH_W7 = os.path.join("results", "training_log_w7.csv")

# In-distribution test set benchmarks (spec v2 §3a)
_INDIST_SRPT = 26.4
_INDIST_RR   = 34.4

# OOD (Week 1) test set benchmarks (spec v2 §3b)
_OOD_SRPT    = 28.4
_OOD_RR      = 36.4

# Warmup transitions before first gradient update
_WARMUP = 500

# Replay buffer capacity and batch size
_BUF_CAPACITY = 10_000
_BATCH_SIZE   = 32

# Target network hard-copy interval (episodes)
_TARGET_UPDATE_FREQ = 200

# Loss divergence threshold — training halts with WARNING if exceeded
_LOSS_DIVERGENCE_THRESHOLD = 1000.0


# ---------------------------------------------------------------------------
# SRPT oracle (for baseline comparison)
# ---------------------------------------------------------------------------

class SRPT:
    """Shortest Remaining Processing Time — preemptive oracle.

    Always runs the runnable process with minimum remaining_burst.
    Uses the 1 ms quantum (tier 0) to maximise preemption opportunity.
    """

    def reset(self) -> None:
        pass

    def select_action(self, env: SchedEnv) -> int:
        runnable = [
            p for p in env.processes
            if p.arrival_time <= env.current_time and not p.is_complete
        ]
        if not runnable:
            raise RuntimeError("SRPT: no runnable processes")
        best = min(runnable, key=lambda p: p.remaining_burst)
        return best.pid * N_QUANTUM_TIERS + 0   # quantum_tier=0 → 1 ms


# ---------------------------------------------------------------------------
# State encoding: env → 15-dim continuous vector
# ---------------------------------------------------------------------------

def _encode_state(env: SchedEnv) -> np.ndarray:
    """Encode live env state as a 15-dim float32 vector.

    Per process (3 features × 5 processes = 15):
        remaining_burst / BURST_P95   (0 if complete)
        arrived_flag                  (1.0 if arrival_time <= current_time)
        wait_time / WAIT_NORM
    """
    vec: list[float] = []
    for p in env.processes:
        arrived   = float(p.arrival_time <= env.current_time)
        remaining = 0.0 if p.is_complete else p.remaining_burst
        vec += [remaining / BURST_P95, arrived, p.wait_time / WAIT_NORM]
    return np.array(vec, dtype=np.float32)


# ---------------------------------------------------------------------------
# Valid-action helpers
# ---------------------------------------------------------------------------

def _valid_actions_env(env: SchedEnv) -> list[int]:
    """Action indices for currently runnable processes."""
    return [
        p.pid * N_QUANTUM_TIERS + qt
        for p in env.processes
        for qt in range(N_QUANTUM_TIERS)
        if p.arrival_time <= env.current_time and not p.is_complete
    ]


def _valid_actions_disc(state: tuple[int, ...]) -> list[int]:
    """Actions for runnable processes from discrete state (rb_bin not 0 or 6)."""
    return [
        pid * N_QUANTUM_TIERS + qt
        for pid in range(N_PROCESSES)
        for qt in range(N_QUANTUM_TIERS)
        if state[pid] not in (0, 6)
    ]


# ---------------------------------------------------------------------------
# Training (Week 3 — DQN)
# ---------------------------------------------------------------------------

def train(
    n_episodes:   int   = 10_000,
    lr:           float = 0.001,
    gamma:        float = 1.0,
    init_epsilon: float = 1.0,
    eps_min:      float = 0.05,
    eps_decay:    float = 0.9995,
    log_path:     str   = _LOG_PATH_W5,
    print_every:  int   = 500,
    agent_class          = None,
) -> ActionConditionedDQN:
    """Train on randomised episodes (new process set per episode).

    agent_class defaults to ActionConditionedDQN (Week 5).
    Pass DQNAgent to reproduce Week 3 training.

    Reward   : dense area-integral  R = -n_active * q_actual.
    Warmup   : first _WARMUP transitions use random actions (no updates).
    Updates  : one gradient step per env step after warmup.
    Target   : hard copy online → target every _TARGET_UPDATE_FREQ episodes.
    Coverage : distinct discrete states visited, logged per episode.
    Loss     : mean batch loss per episode, logged in CSV.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if agent_class is None:
        agent_class = ActionConditionedDQN

    agent = agent_class(
        n_actions  = N_ACTIONS,
        state_dim  = 15,
        lr         = lr,
        gamma      = gamma,
        epsilon    = init_epsilon,
        grad_clip  = 1.0,
    )
    buffer = ReplayBuffer(capacity=_BUF_CAPACITY, state_dim=15)

    visited_states: set[tuple[int, ...]] = set()
    total_transitions: int = 0

    win_reward:   list[float] = []
    win_mct:      list[float] = []
    win_loss:     list[float] = []
    win_entropy:  list[float] = []

    # True when update_online returns (loss, entropy) tuple (e.g. AttentionDQN)
    _returns_entropy = isinstance(agent, AttentionDQN)

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "total_reward", "mean_completion_time",
            "epsilon", "steps", "coverage_states", "mean_loss", "mean_entropy",
        ])

        for ep in range(1, n_episodes + 1):
            # Anneal lambda_ent linearly: 0.10 → 0.005 over n_episodes
            if _returns_entropy:
                agent.lambda_ent = 0.10 - (0.10 - 0.005) * (ep / n_episodes)

            # Fresh random process set every episode
            procs = generate_random_processes()
            env   = SchedEnv(procs)
            env.reset()
            state_vec = _encode_state(env)

            # Discrete state for coverage tracking only
            visited_states.add(env._discretize_state())

            prev_info: dict = {
                "env_reward": 0.0, "current_time": 0.0,
                "completed_pids": [], "completion_times": {},
                "num_waiting": 0,
            }
            ep_reward     = 0.0
            ep_loss_sum   = 0.0
            ep_loss_n     = 0
            ep_ent_sum    = 0.0
            done          = False

            while not done:
                valid = _valid_actions_env(env)

                # Random action during warmup, masked epsilon-greedy after
                if total_transitions < _WARMUP:
                    action = random.choice(valid)
                else:
                    action = agent.select_action(state_vec, agent.epsilon, valid)

                _, _, done, info = env.step(action)
                reward         = compute_reward(info, prev_info)
                next_state_vec = _encode_state(env)

                # Track discrete coverage
                visited_states.add(env._discretize_state())

                buffer.store(state_vec, action, reward, next_state_vec, done)
                total_transitions += 1

                # Gradient update (after warmup, once buffer has enough)
                if total_transitions >= _WARMUP and len(buffer) >= _BATCH_SIZE:
                    s_b, a_b, r_b, ns_b, d_b = buffer.sample(_BATCH_SIZE)
                    result = agent.update_online(
                        s_b.astype(np.float64),
                        a_b,
                        r_b.astype(np.float64),
                        ns_b.astype(np.float64),
                        d_b.astype(np.float64),
                    )
                    if _returns_entropy:
                        loss, ent = result
                        ep_ent_sum += ent
                    else:
                        loss = result
                    ep_loss_sum += loss
                    ep_loss_n   += 1

                ep_reward  += reward
                state_vec   = next_state_vec
                prev_info   = info

            # Target network hard copy every _TARGET_UPDATE_FREQ episodes
            if ep % _TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            mct        = info.get("mean_completion_time_so_far") or 0.0
            mean_loss  = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else ""
            mean_ent   = ep_ent_sum  / ep_loss_n if (ep_loss_n > 0 and _returns_entropy) else ""
            agent.decay_epsilon(ep, min_eps=eps_min, decay=eps_decay)
            coverage   = len(visited_states)

            writer.writerow([
                ep,
                f"{ep_reward:.4f}",
                f"{mct:.4f}",
                f"{agent.epsilon:.6f}",
                env.step_count,
                coverage,
                f"{mean_loss:.6f}" if mean_loss != "" else "",
                f"{mean_ent:.6f}"  if mean_ent  != "" else "",
            ])

            win_reward.append(ep_reward)
            win_mct.append(mct)
            if mean_loss != "":
                win_loss.append(float(mean_loss))
            if mean_ent != "":
                win_entropy.append(float(mean_ent))

            if ep % print_every == 0:
                n  = print_every
                ar = sum(win_reward[-n:]) / n
                am = sum(win_mct[-n:])    / n
                al = (sum(win_loss[-n:]) / min(len(win_loss), n)
                      if win_loss else float("nan"))
                ae = (sum(win_entropy[-n:]) / min(len(win_entropy), n)
                      if win_entropy else None)

                if ae is not None:
                    lam_cur = agent.lambda_ent if isinstance(agent, AttentionDQN) else 0.0
                    ent_str = f" | avg_H={ae:.4f} | lambda_ent={lam_cur:.4f}"
                else:
                    ent_str = ""
                print(
                    f"ep {ep:>6} | eps={agent.epsilon:.4f} | "
                    f"avg_reward={ar:>9.2f} | avg_MCT={am:>6.2f}ms | "
                    f"coverage={coverage:>5} ({coverage/16807*100:.1f}%) | "
                    f"avg_loss={al:.4f}{ent_str}"
                )

                if al > _LOSS_DIVERGENCE_THRESHOLD:
                    print(
                        f"\nWARNING: avg_loss={al:.1f} exceeds threshold "
                        f"{_LOSS_DIVERGENCE_THRESHOLD} at ep {ep} — "
                        f"training diverged, stopping early."
                    )
                    return agent

                # Attention entropy gate: H > 1.2 at ep 5000 → stop
                if _returns_entropy and ae is not None and ep == 5000:
                    if ae > 1.2:
                        print(
                            f"\nSTOP GATE: avg_H={ae:.4f} > 1.2 at ep 5000 — "
                            f"attention not sharpening.  Stopping training."
                        )
                        return agent

    print(f"\nTraining complete — log: {log_path}")
    return agent


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_on(
    agent:      DQNAgent,
    procs:      list[Process],
    label:      str,
    srpt_opt:   float,
    rr_base:    float,
    n_episodes: int = 100,
) -> float:
    """Greedy evaluation on a fixed process set."""
    env  = SchedEnv(procs)
    mcts: list[float] = []

    for _ in range(n_episodes):
        env.reset()
        state_vec = _encode_state(env)
        done      = False
        while not done:
            valid  = _valid_actions_env(env)
            action = agent.select_action(state_vec, epsilon=0.0, valid_actions=valid)
            _, _, done, info = env.step(action)
            state_vec = _encode_state(env)
        mcts.append(info["mean_completion_time_so_far"])

    mean_mct = sum(mcts) / len(mcts)
    std_mct  = (sum((x - mean_mct) ** 2 for x in mcts) / len(mcts)) ** 0.5

    print()
    print("=" * 58)
    print(f"Evaluation — {label}")
    print("=" * 58)
    print(f"  Episodes        : {n_episodes}")
    print(f"  Mean MCT        : {mean_mct:.2f}ms  (std={std_mct:.2f})")
    print(f"  SRPT optimal    : {srpt_opt}ms")
    print(f"  RR(5ms) base    : {rr_base}ms")
    print(f"  Beat RR?        : {'YES' if mean_mct < rr_base else 'NO'}")
    print(f"  Gap to SRPT     : {mean_mct - srpt_opt:+.2f}ms")
    print("=" * 58)
    return mean_mct


def evaluate_indist(agent: DQNAgent, n_episodes: int = 100) -> float:
    """Greedy evaluation on the in-distribution test set."""
    return _evaluate_on(
        agent, generate_indist_processes(),
        label      = "in-distribution set  (arr: 0,2,5,8,10ms)",
        srpt_opt   = _INDIST_SRPT,
        rr_base    = _INDIST_RR,
        n_episodes = n_episodes,
    )


def evaluate_ood(agent: DQNAgent, n_episodes: int = 100) -> float:
    """Greedy evaluation on the Week 1 OOD set."""
    return _evaluate_on(
        agent, generate_fixed_processes(),
        label      = "OOD set  (Week 1 — three arr at t=0)",
        srpt_opt   = _OOD_SRPT,
        rr_base    = _OOD_RR,
        n_episodes = n_episodes,
    )


# ---------------------------------------------------------------------------
# Random evaluation + SRPT baseline (null hypothesis test)
# ---------------------------------------------------------------------------

def _make_eval_seeds(n: int, master_seed: int = 42) -> list[int]:
    """Generate n integer seeds from a fixed master seed."""
    return np.random.RandomState(master_seed).randint(0, 2**31, size=n).tolist()


def evaluate_random(agent: DQNAgent, n_episodes: int = 500) -> dict:
    """Greedy evaluation on fresh random process sets.

    Each episode uses a seed from a reproducible sequence (master_seed=42)
    so results are directly comparable with run_srpt_baseline().

    Returns dict with mean, std, min, max, p5, p95.
    """
    seeds = _make_eval_seeds(n_episodes)
    mcts: list[float] = []

    for seed in seeds:
        procs = generate_random_processes(seed=seed)
        env   = SchedEnv(procs)
        env.reset()
        state_vec = _encode_state(env)
        done      = False
        while not done:
            valid  = _valid_actions_env(env)
            action = agent.select_action(state_vec, epsilon=0.0, valid_actions=valid)
            _, _, done, info = env.step(action)
            state_vec = _encode_state(env)
        mcts.append(info["mean_completion_time_so_far"])

    arr = np.array(mcts)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "p5":   float(np.percentile(arr, 5)),
        "p95":  float(np.percentile(arr, 95)),
    }


def run_srpt_baseline(n_episodes: int = 500) -> dict:
    """SRPT oracle on the same seed sequence as evaluate_random().

    Identical process sets (same master_seed=42) isolate agent variance
    from process-set variance in the std comparison.

    Returns dict with mean, std, min, max, p5, p95.
    """
    seeds = _make_eval_seeds(n_episodes)
    srpt  = SRPT()
    mcts: list[float] = []

    for seed in seeds:
        procs = generate_random_processes(seed=seed)
        env   = SchedEnv(procs)
        srpt.reset()
        env.reset()
        done = False
        while not done:
            action = srpt.select_action(env)
            _, _, done, info = env.step(action)
        mcts.append(info["mean_completion_time_so_far"])

    arr = np.array(mcts)
    return {
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "min":  float(arr.min()),
        "max":  float(arr.max()),
        "p5":   float(np.percentile(arr, 5)),
        "p95":  float(np.percentile(arr, 95)),
    }


def _print_random_stats(label: str, stats: dict) -> None:
    print()
    print("=" * 58)
    print(f"Random eval — {label}")
    print("=" * 58)
    print(f"  Mean MCT : {stats['mean']:.2f}ms")
    print(f"  Std      : {stats['std']:.2f}ms")
    print(f"  Min      : {stats['min']:.2f}ms")
    print(f"  Max      : {stats['max']:.2f}ms")
    print(f"  P5       : {stats['p5']:.2f}ms")
    print(f"  P95      : {stats['p95']:.2f}ms")
    print("=" * 58)


# ---------------------------------------------------------------------------
# Baseline comparison (updated for DQNAgent)
# ---------------------------------------------------------------------------

def _run_baseline_episode(env: SchedEnv, policy) -> float:
    policy.reset()
    env.reset()
    done = False
    while not done:
        action = policy.select_action(env)
        _, _, done, info = env.step(action)
    return info["mean_completion_time_so_far"]


def compare_baselines(agent: DQNAgent, n_episodes: int = 100) -> None:
    env  = SchedEnv(generate_fixed_processes())
    rr   = RoundRobin()
    fcfs = FCFS()

    results: dict[str, list[float]] = {"RL Agent": [], "Round Robin": [], "FCFS": []}

    for _ in range(n_episodes):
        env.reset()
        state_vec = _encode_state(env)
        done      = False
        while not done:
            valid  = _valid_actions_env(env)
            action = agent.select_action(state_vec, epsilon=0.0, valid_actions=valid)
            _, _, done, info = env.step(action)
            state_vec = _encode_state(env)
        results["RL Agent"].append(info["mean_completion_time_so_far"])
        results["Round Robin"].append(_run_baseline_episode(env, rr))
        results["FCFS"].append(_run_baseline_episode(env, fcfs))

    def _stats(mcts):
        n = len(mcts); m = sum(mcts) / n
        return m, (sum((x - m) ** 2 for x in mcts) / n) ** 0.5, min(mcts), max(mcts)

    stats  = {name: _stats(v) for name, v in results.items()}
    col_w  = [16, 22, 8, 8, 8]
    hdr    = (f"{'Policy':<{col_w[0]}} | {'Mean MCT (ms)':>{col_w[1]}} | "
              f"{'Std':>{col_w[2]}} | {'Min':>{col_w[3]}} | {'Max':>{col_w[4]}}")
    sep    = "-" * len(hdr)
    print(); print(sep); print(hdr); print(sep)
    for name, (mean, std, lo, hi) in stats.items():
        print(f"{name:<{col_w[0]}} | {mean:>{col_w[1]}.2f} | "
              f"{std:>{col_w[2]}.2f} | {lo:>{col_w[3]}.2f} | {hi:>{col_w[4]}.2f}")
    print(sep)
    rl, rr_m = stats["RL Agent"][0], stats["Round Robin"][0]
    result = "PASSED" if rl < rr_m else "FAILED"
    print(f"\nSuccess criterion (RL < RR): {result}  "
          f"({rl:.2f}ms vs {rr_m:.2f}ms)")


# ---------------------------------------------------------------------------
# Policy analysis (Week 4 table)
# ---------------------------------------------------------------------------

def run_policy_analysis(agent, n_decisions: int = 10_000) -> dict:
    """Collect greedy decisions vs SRPT oracle; return analysis dict.

    Returns a dict with:
        by_n_active : dict{n_active → {total, disagree, qt_counts}}
        overall     : {total, agree, disagree}
    """
    from collections import defaultdict, Counter

    QUANTUM_MS = {0: 1, 1: 5, 2: 20}
    srpt       = SRPT()
    records: list[dict] = []

    ep = 0
    while len(records) < n_decisions:
        procs = generate_random_processes()
        env   = SchedEnv(procs)
        env.reset()
        state_vec = _encode_state(env)
        srpt.reset()
        done = False

        while not done:
            valid  = _valid_actions_env(env)
            action = agent.select_action(state_vec, epsilon=0.0, valid_actions=valid)
            dqn_pid = action // N_QUANTUM_TIERS
            dqn_qt  = action %  N_QUANTUM_TIERS

            srpt_action = srpt.select_action(env)
            srpt_pid    = srpt_action // N_QUANTUM_TIERS

            n_active = sum(
                1 for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete
            )

            records.append({
                "n_active":  n_active,
                "dqn_qt":    dqn_qt,
                "agree_pid": dqn_pid == srpt_pid,
            })

            _, _, done, _ = env.step(action)
            state_vec = _encode_state(env)
        ep += 1

    # Aggregate by n_active
    by_n: dict = defaultdict(lambda: {"total": 0, "disagree": 0,
                                      "qt": Counter()})
    for r in records:
        n = r["n_active"]
        by_n[n]["total"]   += 1
        by_n[n]["qt"][r["dqn_qt"]] += 1
        if not r["agree_pid"]:
            by_n[n]["disagree"] += 1

    total    = len(records)
    agree    = sum(r["agree_pid"] for r in records)
    disagree = total - agree

    print()
    print("=" * 64)
    print(f"Policy analysis — {total} decisions across {ep} episodes")
    print("=" * 64)
    print(f"  Overall SRPT agreement : {agree}/{total}  ({100*agree/total:.1f}%)")
    print()
    print(f"  {'n_act':>5}  {'agree%':>7}  {'1ms%':>7}  {'5ms%':>7}  {'20ms%':>7}  {'n':>6}")
    print("  " + "-" * 46)
    for n in sorted(by_n):
        tot = by_n[n]["total"]
        dis = by_n[n]["disagree"]
        qt  = by_n[n]["qt"]
        agr = 100 * (tot - dis) / tot
        q0  = 100 * qt[0] / tot
        q1  = 100 * qt[1] / tot
        q2  = 100 * qt[2] / tot
        print(f"  {n:>5}  {agr:>6.1f}%  {q0:>6.1f}%  {q1:>6.1f}%  {q2:>6.1f}%  {tot:>6}")

    return {"by_n": dict(by_n), "total": total,
            "agree": agree, "disagree": disagree}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    import time

    w_path_w5 = os.path.join("results", "dqn_w5.npz")
    w_path_w6 = os.path.join("results", "dqn_w6.npz")
    w_path_w7 = os.path.join("results", "dqn_w7.npz")

    print("=" * 64)
    print("Week 7 Training — AttentionDQN (lambda_ent annealed 0.10→0.005)")
    print("=" * 64)
    t0    = time.time()
    agent = train(n_episodes=10_000, agent_class=AttentionDQN,
                  log_path=_LOG_PATH_W7)
    elapsed = time.time() - t0
    print(f"\nTraining wall time: {elapsed/60:.1f} min")
    save_path = w_path_w7.replace(".npz", "_anneal.npz")
    agent.save(save_path)
    print(f"Weights saved → {save_path}")

    # ================================================================
    # STEP 5 — Full evaluation
    # ================================================================

    # --- a) Random evaluation -------------------------------------------
    print("\nRunning evaluate_random (n=500) ...")
    rl_stats = evaluate_random(agent, n_episodes=500)
    _print_random_stats("AttentionDQN (Week 7, annealed lambda_ent)", rl_stats)

    # --- b) SRPT calibration -------------------------------------------
    print("\nRunning run_srpt_baseline (n=500) ...")
    srpt_stats = run_srpt_baseline(n_episodes=500)
    _print_random_stats("SRPT oracle", srpt_stats)

    # --- c) Policy analysis (10,000 greedy decisions) ------------------
    print("\nCollecting 10,000 greedy decisions for policy analysis ...")
    policy_results = run_policy_analysis(agent, n_decisions=10_000)

    # --- d) Identity bias probe ----------------------------------------
    # P0 complete, P1 10ms remaining, P2 complete, P3 not arrived, P4 3ms
    probe_state = np.array([
        0.0,               1.0, 0.0,   # P0 complete  (remaining=0 → invalid)
        10.0/BURST_P95,    1.0, 0.0,   # P1 10s remaining
        0.0,               1.0, 0.0,   # P2 complete  (remaining=0 → invalid)
        50.0/BURST_P95,    0.0, 0.0,   # P3 not arrived (arrived=0 → invalid)
        3.0/BURST_P95,     1.0, 0.0,   # P4 3s remaining
    ], dtype=np.float64)

    # P1/long = action 1*3+2=5;  P4/long = action 4*3+2=14
    q_p1_long = float(agent.forward_batch(probe_state[None], np.array([5]))[0, 0])
    q_p4_long = float(agent.forward_batch(probe_state[None], np.array([14]))[0, 0])
    delta_bias = abs(q_p1_long - q_p4_long)
    p4_wins    = q_p4_long > q_p1_long

    print()
    print("=" * 58)
    print("Identity bias probe — P3 not arrived, P4 3ms vs P1 10ms")
    print("=" * 58)
    print(f"  Q(P1/long) = {q_p1_long:+.5f}")
    print(f"  Q(P4/long) = {q_p4_long:+.5f}")
    print(f"  |Delta|    = {delta_bias:.5f}")
    print(f"  P4 preferred (SRPT correct)? {p4_wins}")
    print(f"  Direction correct          : {p4_wins}")

    # --- e) Comparative permutation invariance test (W7 vs W5) --------
    # State A: P0 candidate (3ms,wait=0), P1 comp (10ms,wait=5ms),
    #          P2 comp (15ms,wait=3ms), P3/P4 not arrived.
    # State B: P4 candidate (3ms,wait=0), P2 comp (10ms,wait=5ms),
    #          P1 comp (15ms,wait=3ms), P0/P3 not arrived.
    # Same competitor multiset, different PIDs — W7 must be invariant.

    cmp_A = np.zeros(15, dtype=np.float64)
    cmp_A[0*3 : 0*3+3] = [3.0/BURST_P95,  1.0, 0.0/WAIT_NORM]   # P0 candidate
    cmp_A[1*3 : 1*3+3] = [10.0/BURST_P95, 1.0, 5.0/WAIT_NORM]   # P1 comp
    cmp_A[2*3 : 2*3+3] = [15.0/BURST_P95, 1.0, 3.0/WAIT_NORM]   # P2 comp

    cmp_B = np.zeros(15, dtype=np.float64)
    cmp_B[4*3 : 4*3+3] = [3.0/BURST_P95,  1.0, 0.0/WAIT_NORM]   # P4 candidate
    cmp_B[2*3 : 2*3+3] = [10.0/BURST_P95, 1.0, 5.0/WAIT_NORM]   # P2 comp (10s→P2)
    cmp_B[1*3 : 1*3+3] = [15.0/BURST_P95, 1.0, 3.0/WAIT_NORM]   # P1 comp (15s→P1)

    Q_w7_A = float(agent.forward_batch(cmp_A[None], np.array([0*3+0]))[0, 0])
    Q_w7_B = float(agent.forward_batch(cmp_B[None], np.array([4*3+0]))[0, 0])

    print()
    print("=" * 64)
    print("Comparative permutation invariance test (W7 vs W5)")
    print("=" * 64)
    print()
    print("  Week 7 AttentionDQN (trained):")
    print(f"    Q(P0 candidate, state_A) = {Q_w7_A:.10f}")
    print(f"    Q(P4 candidate, state_B) = {Q_w7_B:.10f}")
    print(f"    |Delta|                  = {abs(Q_w7_A - Q_w7_B):.10f}")
    print(f"    Invariant (|delta|<1e-8) : {abs(Q_w7_A - Q_w7_B) < 1e-8}")

    w5_exists = os.path.isfile(w_path_w5)
    if w5_exists:
        agent_w5 = ActionConditionedDQN()
        agent_w5.load(w_path_w5)
        Xcmp_A_w5 = agent_w5._build_input_batch(cmp_A[None], np.array([0*3+0]))
        Xcmp_B_w5 = agent_w5._build_input_batch(cmp_B[None], np.array([4*3+0]))
        Q_w5_A = float(agent_w5.forward_batch(Xcmp_A_w5)[0, 0])
        Q_w5_B = float(agent_w5.forward_batch(Xcmp_B_w5)[0, 0])
        print()
        print("  Week 5 ActionConditionedDQN (loaded from dqn_w5.npz):")
        print(f"    Q(P0 candidate, state_A) = {Q_w5_A:.10f}")
        print(f"    Q(P4 candidate, state_B) = {Q_w5_B:.10f}")
        print(f"    |Delta|                  = {abs(Q_w5_A - Q_w5_B):.10f}")
        print(f"    Invariant (|delta|<1e-8) : {abs(Q_w5_A - Q_w5_B) < 1e-8}")
        print()
        print("  W5 |Delta| > 0 confirms residual positional bias that")
        print("  W7 attention eliminates.")
    else:
        print()
        print(f"  (W5 weights not found at {w_path_w5} — skipping W5 comparison)")
    print("=" * 64)
