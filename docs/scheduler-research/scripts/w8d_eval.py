"""W8d full evaluation: filtered trace, log-norm, all diagnostics."""
from __future__ import annotations

import sys, random
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")

from schedsim.env           import SchedEnv, N_ACTIONS, N_PROCESSES, N_QUANTUM_TIERS, WAIT_NORM
from schedsim.agent         import AttentionDQN
from schedsim.process       import Process
from schedsim.baselines     import RoundRobin
from schedsim.trace_sampler import TraceEpisodeSampler

WEIGHTS_PATH = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                "GRAD - FALL 23/UCSC/Capstone/results/dqn_w8d.npz")
TEST_PATH    = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_test_filtered.csv"
MASTER_SEED  = 42
N_EVAL       = 500
N_DECISIONS  = 10_000

# Filtered-trace constants
BURST_P95_FILT    = 36.0
_LOG_DENOM        = float(np.log1p(BURST_P95_FILT))
QUANTUM_TIERS_W8D = (0.25, 1.0, 4.0)


def _log_norm_burst(burst: float) -> float:
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
        return min(runnable, key=lambda p: p.remaining_burst).pid * N_QUANTUM_TIERS + 0


class RoundRobinTier1:
    """RR with tier1 = 1.0s quantum (PID-ascending cycle)."""
    def __init__(self): self._last_pid = -1
    def reset(self):    self._last_pid = -1
    def select_action(self, env):
        ready = [p for p in env.processes
                 if p.arrival_time <= env.current_time and not p.is_complete]
        pids  = sorted(p.pid for p in ready)
        nxt   = next((p for p in pids if p > self._last_pid), pids[0])
        self._last_pid = nxt
        return nxt * N_QUANTUM_TIERS + 1   # tier1 = 1.0s


def _run_agent(agent, procs):
    env = SchedEnv(procs); env.reset()
    sv = _encode_state(env); done = False
    while not done:
        valid = _valid_actions(env)
        action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
        _, _, done, info = env.step(action)
        sv = _encode_state(env)
    return info["mean_completion_time_so_far"]


def _run_policy(policy, procs):
    env = SchedEnv(procs); policy.reset(); env.reset(); done = False
    while not done:
        action = policy.select_action(env)
        _, _, done, info = env.step(action)
    return info["mean_completion_time_so_far"]


def _stats(arr):
    return dict(mean=float(arr.mean()), std=float(arr.std()),
                min=float(arr.min()), max=float(arr.max()),
                p5=float(np.percentile(arr,5)), p95=float(np.percentile(arr,95)))


def _print_stats(label, s):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for k in ("mean","std","min","max","p5","p95"):
        print(f"  {k:>4} : {s[k]:.4f}s")
    print(f"{'='*60}")


def _get_attention_weights(agent, state_vec, action):
    """Return (attn_weights[4], comp_bursts[4], comp_valid[4])."""
    state_b  = state_vec[None].astype(np.float64)
    action_b = np.array([action], dtype=np.int32)
    batch    = 1
    pids     = (action_b // agent.N_QT).astype(np.int32)
    s3d      = state_b.reshape(batch, N_PROCESSES, 3)
    cand_enc = s3d[np.arange(batch), pids]
    comp_encs, comp_valid = agent._build_competitor_data(state_b, pids)
    _, cache = agent._attention_forward(cand_enc, comp_encs, comp_valid)
    weights    = cache["weights"][0]          # (4,)
    comp_valid = comp_valid[0]                # (4,)
    comp_bursts = comp_encs[0, :, 0]         # (4,) log-norm values
    return weights, comp_bursts, comp_valid


if __name__ == "__main__":
    random.seed(MASTER_SEED); np.random.seed(MASTER_SEED)

    print("Loading test sampler ...")
    sampler = TraceEpisodeSampler(TEST_PATH)
    print(f"  {len(sampler):,} valid-duration tasks in filtered test split.")

    print("\nLoading W8d agent ...")
    agent = AttentionDQN(n_actions=N_ACTIONS, state_dim=15, lr=0.001,
                         gamma=1.0, epsilon=0.0, grad_clip=1.0)
    agent.load(WEIGHTS_PATH)
    agent.epsilon = 0.0
    print(f"  Loaded {WEIGHTS_PATH}")

    srpt = SRPT()
    rr   = RoundRobinTier1()

    # Pre-generate 500 shared episode process sets
    seeds = _make_seeds(N_EVAL)
    all_procs = []
    for seed in seeds:
        rng   = np.random.default_rng(seed)
        tasks = sampler.sample_episode(rng)
        all_procs.append(_make_procs(tasks))

    # ---- a) Agent ----
    print(f"\nRunning agent evaluation (n={N_EVAL}) ...")
    agent_mcts  = np.array([_run_agent(agent, p) for p in all_procs])
    agent_stats = _stats(agent_mcts)
    _print_stats("a) W8d AttentionDQN (filtered, log-norm) — trace test", agent_stats)

    # ---- b) SRPT ----
    print(f"\nRunning SRPT oracle (n={N_EVAL}) ...")
    srpt_mcts  = np.array([_run_policy(srpt, p) for p in all_procs])
    srpt_stats = _stats(srpt_mcts)
    _print_stats("b) SRPT oracle — filtered trace test", srpt_stats)

    # ---- c) Round Robin ----
    print(f"\nRunning Round Robin (n={N_EVAL}, quantum={QUANTUM_TIERS_W8D[1]}s) ...")
    rr_mcts  = np.array([_run_policy(rr, p) for p in all_procs])
    rr_stats = _stats(rr_mcts)
    _print_stats(f"c) Round Robin (tier1={QUANTUM_TIERS_W8D[1]}s) — filtered trace test", rr_stats)

    # ---- Summary table ----
    print("\n")
    hdr = f"{'Agent':<28} | {'Mean MCT':>10} | {'Std':>8} | {'vs RR':>9} | {'vs SRPT':>9}"
    sep = "-" * len(hdr)
    print(hdr); print(sep)
    print(f"{'SRPT oracle (filtered)':<28} | {srpt_stats['mean']:>10.4f}s | {srpt_stats['std']:>8.4f}s | {'---':>9} | {'0.0':>9}")
    print(f"{'W8d AttentionDQN':<28} | {agent_stats['mean']:>10.4f}s | {agent_stats['std']:>8.4f}s | {agent_stats['mean']-rr_stats['mean']:>+9.4f}s | {agent_stats['mean']-srpt_stats['mean']:>+9.4f}s")
    print(f"{'Round Robin (filtered)':<28} | {rr_stats['mean']:>10.4f}s | {rr_stats['std']:>8.4f}s | {'base':>9} | {'---':>9}")

    # ---- d) Policy analysis ----
    print(f"\n\nRunning policy analysis ({N_DECISIONS:,} greedy decisions) ...")
    records = []
    ep      = 0
    rng_pa  = np.random.default_rng(MASTER_SEED + 1)
    while len(records) < N_DECISIONS:
        tasks = sampler.sample_episode(rng_pa)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs); env.reset(); sv = _encode_state(env)
        srpt.reset(); done = False
        while not done:
            valid  = _valid_actions(env)
            action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
            srpt_pid = srpt.select_action(env) // N_QUANTUM_TIERS
            n_active = sum(1 for p in env.processes
                           if p.arrival_time <= env.current_time and not p.is_complete)
            records.append({"n_active": n_active,
                            "dqn_qt": action % N_QUANTUM_TIERS,
                            "agree_pid": action // N_QUANTUM_TIERS == srpt_pid})
            _, _, done, _ = env.step(action)
            sv = _encode_state(env)
        ep += 1

    by_n = defaultdict(lambda: {"total":0,"disagree":0,"qt":Counter()})
    for r in records:
        n = r["n_active"]
        by_n[n]["total"]   += 1
        by_n[n]["qt"][r["dqn_qt"]] += 1
        if not r["agree_pid"]: by_n[n]["disagree"] += 1
    total = len(records); agree = sum(r["agree_pid"] for r in records)

    print(f"\n{'='*64}")
    print(f"d) Policy analysis — {total:,} decisions across {ep} episodes")
    print(f"{'='*64}")
    print(f"  Overall SRPT agreement : {agree}/{total}  ({100*agree/total:.1f}%)")
    print(); print(f"  {'n_act':>5}  {'agree%':>7}  {'tier0%':>7}  {'tier1%':>7}  {'tier2%':>7}  {'n':>6}")
    print("  " + "-" * 48)
    for n in sorted(by_n):
        tot = by_n[n]["total"]; dis = by_n[n]["disagree"]; qt = by_n[n]["qt"]
        print(f"  {n:>5}  {100*(tot-dis)/tot:>6.1f}%  {100*qt[0]/tot:>6.1f}%  {100*qt[1]/tot:>6.1f}%  {100*qt[2]/tot:>6.1f}%  {tot:>6}")
    print(f"{'='*64}")

    # ---- e) Identity bias probe ----
    probe = np.zeros(15, dtype=np.float64)
    probe[0*3:0*3+3] = [0.20, 1.0, 0.0]   # P0 candidate
    probe[1*3:1*3+3] = [0.60, 1.0, 0.0]   # P1 competitor
    cand_burst = float(np.expm1(0.20 * _LOG_DENOM))
    comp_burst = float(np.expm1(0.60 * _LOG_DENOM))
    Q_cand = float(agent.forward_batch(probe[None], np.array([0*3+2]))[0, 0])
    Q_comp = float(agent.forward_batch(probe[None], np.array([1*3+2]))[0, 0])
    delta  = abs(Q_cand - Q_comp)
    print(f"\n{'='*60}")
    print("e) Identity bias probe")
    print(f"   Candidate P0: log-norm=0.20 -> {cand_burst:.2f}s actual")
    print(f"   Competitor P1: log-norm=0.60 -> {comp_burst:.2f}s actual")
    print(f"{'='*60}")
    print(f"  Q(candidate/tier2) = {Q_cand:+.6f}")
    print(f"  Q(competitor/tier2)= {Q_comp:+.6f}")
    print(f"  |Delta|            = {delta:.6f}")
    print(f"  Candidate preferred (SRPT-correct)? {Q_cand > Q_comp}")
    print(f"{'='*60}")

    # ---- f) Attention diagnostic — n_active >= 3 ----
    print(f"\n\nRunning attention diagnostic (n_active >= 3) ...")
    attn_records = []
    rng_attn = np.random.default_rng(MASTER_SEED + 2)
    target_n = 1000
    while len(attn_records) < target_n:
        tasks = sampler.sample_episode(rng_attn)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs); env.reset(); sv = _encode_state(env)
        done  = False
        while not done:
            valid  = _valid_actions(env)
            n_active = sum(1 for p in env.processes
                           if p.arrival_time <= env.current_time and not p.is_complete)
            action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
            if n_active >= 3:
                weights, comp_bursts, comp_valid_mask = _get_attention_weights(agent, sv, action)
                valid_mask = comp_valid_mask
                if valid_mask.sum() >= 2:
                    valid_weights = weights.copy(); valid_weights[~valid_mask] = -1.0
                    top_idx  = int(np.argmax(valid_weights))
                    valid_burst_vals = [(comp_bursts[j], j) for j in range(4) if valid_mask[j]]
                    shortest_idx = min(valid_burst_vals, key=lambda x: x[0])[1]
                    longest_idx  = max(valid_burst_vals, key=lambda x: x[0])[1]
                    attn_records.append({
                        "top_idx": top_idx,
                        "shortest_idx": shortest_idx,
                        "longest_idx":  longest_idx,
                        "top_on_shortest": top_idx == shortest_idx,
                        "top_on_longest":  top_idx == longest_idx,
                        "weight_shortest": float(weights[shortest_idx]),
                        "weight_longest":  float(weights[longest_idx]),
                    })
            _, _, done, _ = env.step(action)
            sv = _encode_state(env)

    n_attn = len(attn_records)
    top_shortest = sum(r["top_on_shortest"] for r in attn_records)
    top_longest  = sum(r["top_on_longest"]  for r in attn_records)
    mean_w_short = np.mean([r["weight_shortest"] for r in attn_records])
    mean_w_long  = np.mean([r["weight_longest"]  for r in attn_records])

    print(f"\n{'='*60}")
    print(f"f) Attention diagnostic — n_active >= 3  (n={n_attn} decisions)")
    print(f"{'='*60}")
    print(f"  Highest-attention = shortest competitor : {top_shortest}/{n_attn}  ({100*top_shortest/n_attn:.1f}%)")
    print(f"  Highest-attention = longest  competitor : {top_longest}/{n_attn}  ({100*top_longest/n_attn:.1f}%)")
    print(f"  Mean attention weight on shortest       : {mean_w_short:.4f}")
    print(f"  Mean attention weight on longest        : {mean_w_long:.4f}")
    print(f"{'='*60}")
