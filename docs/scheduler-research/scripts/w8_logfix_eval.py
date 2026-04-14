"""W8b full evaluation: AttentionDQN (log-norm trained) vs SRPT vs RR on trace test set."""
from __future__ import annotations

import sys
import random
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")

from schedsim.env           import SchedEnv, N_ACTIONS, N_PROCESSES, N_QUANTUM_TIERS, BURST_P95, WAIT_NORM, QUANTUM_TIERS
from schedsim.agent         import AttentionDQN
from schedsim.process       import Process
from schedsim.baselines     import RoundRobin
from schedsim.trace_sampler import TraceEpisodeSampler

WEIGHTS_PATH = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/results/dqn_w8_logfix.npz"
TEST_PATH    = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone/docs/scheduler-research/scripts/trace_test.csv"
MASTER_SEED  = 42
N_EVAL       = 500
N_DECISIONS  = 10_000

_LOG_BURST_DENOM = float(np.log1p(BURST_P95))


def _log_norm_burst(burst: float) -> float:
    return float(np.log1p(burst) / _LOG_BURST_DENOM)


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


def _make_procs(tasks):
    return [Process(pid=i, arrival_time=t["arrival_ms"], burst_length=t["burst_ms"])
            for i, t in enumerate(tasks)]


def _make_seeds(n, master=MASTER_SEED):
    return np.random.RandomState(master).randint(0, 2**31, size=n).tolist()


class SRPT:
    def reset(self): pass
    def select_action(self, env):
        runnable = [p for p in env.processes
                    if p.arrival_time <= env.current_time and not p.is_complete]
        best = min(runnable, key=lambda p: p.remaining_burst)
        return best.pid * N_QUANTUM_TIERS + 0


def _run_agent(agent, procs):
    env = SchedEnv(procs)
    env.reset()
    sv = _encode_state(env)
    done = False
    while not done:
        valid = _valid_actions(env)
        action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
        _, _, done, info = env.step(action)
        sv = _encode_state(env)
    return info["mean_completion_time_so_far"]


def _run_policy(policy, procs):
    env = SchedEnv(procs)
    policy.reset()
    env.reset()
    done = False
    while not done:
        action = policy.select_action(env)
        _, _, done, info = env.step(action)
    return info["mean_completion_time_so_far"]


def _stats(arr):
    return {"mean": float(arr.mean()), "std": float(arr.std()),
            "min": float(arr.min()), "max": float(arr.max()),
            "p5": float(np.percentile(arr, 5)), "p95": float(np.percentile(arr, 95))}


def _print_stats(label, s):
    print(f"\n{'='*58}")
    print(f"  {label}")
    print(f"{'='*58}")
    print(f"  Mean : {s['mean']:.4f}s")
    print(f"  Std  : {s['std']:.4f}s")
    print(f"  Min  : {s['min']:.4f}s")
    print(f"  Max  : {s['max']:.4f}s")
    print(f"  P5   : {s['p5']:.4f}s")
    print(f"  P95  : {s['p95']:.4f}s")
    print(f"{'='*58}")


if __name__ == "__main__":
    random.seed(MASTER_SEED)
    np.random.seed(MASTER_SEED)

    print("Loading test sampler ...")
    sampler = TraceEpisodeSampler(TEST_PATH)
    print(f"  {len(sampler):,} valid-duration tasks in test split.")

    print("\nLoading W8b agent (log-norm) ...")
    agent = AttentionDQN(n_actions=N_ACTIONS, state_dim=15, lr=0.001,
                         gamma=1.0, epsilon=0.0, grad_clip=1.0)
    agent.load(WEIGHTS_PATH)
    agent.epsilon = 0.0
    print(f"  Weights loaded from {WEIGHTS_PATH}")

    srpt = SRPT()
    rr   = RoundRobin()

    seeds = _make_seeds(N_EVAL)
    all_procs = []
    for seed in seeds:
        rng   = np.random.default_rng(seed)
        tasks = sampler.sample_episode(rng)
        all_procs.append(_make_procs(tasks))

    # a) Agent
    print(f"\nRunning agent evaluation (n={N_EVAL}) ...")
    agent_mcts  = np.array([_run_agent(agent, p) for p in all_procs])
    agent_stats = _stats(agent_mcts)
    _print_stats("a) W8b AttentionDQN (log-norm) — trace test episodes", agent_stats)

    # b) SRPT
    print(f"\nRunning SRPT oracle (n={N_EVAL}) ...")
    srpt_mcts  = np.array([_run_policy(srpt, p) for p in all_procs])
    srpt_stats = _stats(srpt_mcts)
    _print_stats("b) SRPT oracle — trace test episodes", srpt_stats)

    # c) Round Robin
    print(f"\nRunning Round Robin (n={N_EVAL}, quantum={QUANTUM_TIERS[1]}s) ...")
    rr_mcts  = np.array([_run_policy(rr, p) for p in all_procs])
    rr_stats = _stats(rr_mcts)
    _print_stats(f"c) Round Robin (tier1={QUANTUM_TIERS[1]}s) — trace test episodes", rr_stats)

    # Summary table
    print("\n")
    print(f"{'Agent':<32} | {'Mean MCT':>10} | {'Std':>8} | {'vs RR':>9} | {'vs SRPT':>9}")
    print("-" * 78)
    print(f"{'W8b AttentionDQN (log-norm)':<32} | {agent_stats['mean']:>10.4f}s | {agent_stats['std']:>8.4f}s | {agent_stats['mean']-rr_stats['mean']:>+9.4f}s | {agent_stats['mean']-srpt_stats['mean']:>+9.4f}s")
    print(f"{'SRPT oracle (trace)':<32} | {srpt_stats['mean']:>10.4f}s | {srpt_stats['std']:>8.4f}s | {'---':>9} | {'0.0':>9}")
    print(f"{'Round Robin (trace)':<32} | {rr_stats['mean']:>10.4f}s | {rr_stats['std']:>8.4f}s | {'base':>9} | {'---':>9}")

    # d) Policy analysis
    print(f"\n\nRunning policy analysis ({N_DECISIONS:,} greedy decisions) ...")
    records = []
    ep = 0
    rng_pa = np.random.default_rng(MASTER_SEED + 1)

    while len(records) < N_DECISIONS:
        tasks = sampler.sample_episode(rng_pa)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()
        sv = _encode_state(env)
        srpt.reset()
        done = False

        while not done:
            valid  = _valid_actions(env)
            action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
            dqn_pid = action // N_QUANTUM_TIERS
            srpt_pid = srpt.select_action(env) // N_QUANTUM_TIERS
            n_active = sum(1 for p in env.processes
                           if p.arrival_time <= env.current_time and not p.is_complete)
            records.append({"n_active": n_active, "dqn_qt": action % N_QUANTUM_TIERS,
                            "agree_pid": dqn_pid == srpt_pid})
            _, _, done, _ = env.step(action)
            sv = _encode_state(env)
        ep += 1

    by_n = defaultdict(lambda: {"total": 0, "disagree": 0, "qt": Counter()})
    for r in records:
        n = r["n_active"]
        by_n[n]["total"] += 1
        by_n[n]["qt"][r["dqn_qt"]] += 1
        if not r["agree_pid"]:
            by_n[n]["disagree"] += 1

    total = len(records)
    agree = sum(r["agree_pid"] for r in records)

    print(f"\n{'='*64}")
    print(f"d) Policy analysis — {total:,} decisions across {ep} episodes")
    print(f"{'='*64}")
    print(f"  Overall SRPT agreement : {agree}/{total}  ({100*agree/total:.1f}%)")
    print()
    print(f"  {'n_act':>5}  {'agree%':>7}  {'tier0%':>7}  {'tier1%':>7}  {'tier2%':>7}  {'n':>6}")
    print("  " + "-" * 48)
    for n in sorted(by_n):
        tot = by_n[n]["total"]
        dis = by_n[n]["disagree"]
        qt  = by_n[n]["qt"]
        agr = 100 * (tot - dis) / tot
        print(f"  {n:>5}  {agr:>6.1f}%  {100*qt[0]/tot:>6.1f}%  {100*qt[1]/tot:>6.1f}%  {100*qt[2]/tot:>6.1f}%  {tot:>6}")
    print(f"{'='*64}")

    # e) Identity bias probe — log-normalized inputs
    # Candidate P0: remaining=0.05 norm → burst = exp(0.05 * log(398)) - 1 ≈ 1.35s actual
    # But user spec says 0.05 and 0.33 as normalized values directly
    probe = np.zeros(15, dtype=np.float64)
    probe[0*3 : 0*3+3] = [0.05, 1.0, 0.0]   # P0 candidate (0.05 log-norm)
    probe[1*3 : 1*3+3] = [0.33, 1.0, 0.0]   # P1 competitor (0.33 log-norm)

    # Actual burst equivalents under log-norm
    cand_burst = float(np.expm1(0.05 * _LOG_BURST_DENOM))
    comp_burst = float(np.expm1(0.33 * _LOG_BURST_DENOM))

    Q_cand = float(agent.forward_batch(probe[None], np.array([0*3+2]))[0, 0])
    Q_comp = float(agent.forward_batch(probe[None], np.array([1*3+2]))[0, 0])
    delta  = abs(Q_cand - Q_comp)
    cand_preferred = Q_cand > Q_comp

    print(f"\n{'='*58}")
    print("e) Identity bias probe (log-norm inputs)")
    print(f"   Candidate P0: log-norm=0.05 -> {cand_burst:.2f}s actual")
    print(f"   Competitor P1: log-norm=0.33 -> {comp_burst:.2f}s actual")
    print(f"{'='*58}")
    print(f"  Q(candidate/tier2) = {Q_cand:+.6f}")
    print(f"  Q(competitor/tier2)= {Q_comp:+.6f}")
    print(f"  |Delta|            = {delta:.6f}")
    print(f"  Candidate preferred (SRPT-correct)? {cand_preferred}")
    print(f"{'='*58}")
