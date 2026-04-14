"""Week 10A-v2 evaluation — PBRS alpha=0.05, filtered test split."""
from __future__ import annotations
import csv, random, sys
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")
sys.path.insert(0, "/tmp")

from schedsim.env    import SchedEnv, N_PROCESSES
from schedsim.process import Process
from w9_train import (
    AttentionDQN9, TraceEpisodeSampler5,
    _encode_state, _valid_actions, _make_procs,
    D_CAND, N_QT, _LOG_DENOM,
)

WEIGHTS_PATH = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                "GRAD - FALL 23/UCSC/Capstone/results/dqn_w10a_v2.npz")
TEST_PATH    = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_test_filtered.csv"
MASTER_SEED  = 42
N_EVAL       = 500
N_DECISIONS  = 10_000


class SRPT:
    def reset(self): pass
    def select_action(self, env):
        runnable = [p for p in env.processes
                    if p.arrival_time <= env.current_time and not p.is_complete]
        return min(runnable, key=lambda p: p.remaining_burst).pid * N_QT + 0


class RoundRobinTier1:
    def __init__(self): self._last_pid = -1
    def reset(self):    self._last_pid = -1
    def select_action(self, env):
        ready = [p for p in env.processes
                 if p.arrival_time <= env.current_time and not p.is_complete]
        pids  = sorted(p.pid for p in ready)
        nxt   = next((p for p in pids if p > self._last_pid), pids[0])
        self._last_pid = nxt
        return nxt * N_QT + 1


def _run_agent(agent, procs, tasks):
    env = SchedEnv(procs); env.reset()
    sv = _encode_state(env, tasks); done = False
    while not done:
        valid  = _valid_actions(env)
        action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
        _, _, done, info = env.step(action)
        sv = _encode_state(env, tasks)
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
                p5=float(np.percentile(arr, 5)), p95=float(np.percentile(arr, 95)))


def _print_stats(label, s):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for k in ("mean","std","min","max","p5","p95"):
        print(f"  {k:>4} : {s[k]:.4f}s")
    print(f"{'='*60}")


def _get_attention_weights(agent, sv, action):
    states  = sv[None].astype(np.float64)
    actions = np.array([action], dtype=np.int32)
    pids    = (actions // N_QT).astype(np.int32)
    s3d     = states.reshape(1, N_PROCESSES, D_CAND)
    cand_enc = s3d[np.arange(1), pids]
    comp_encs, comp_valid = agent._build_competitor_data(states, pids)
    _, cache = agent._attention_forward(cand_enc, comp_encs, comp_valid)
    return cache["weights"][0], comp_encs[0, :, 0], comp_valid[0]


if __name__ == "__main__":
    random.seed(MASTER_SEED); np.random.seed(MASTER_SEED)

    print("Loading test sampler ...")
    sampler = TraceEpisodeSampler5(TEST_PATH)

    print(f"\nLoading W10A-v2 agent ...")
    agent = AttentionDQN9(lr=0.001, gamma=1.0, grad_clip=1.0)
    agent.load(WEIGHTS_PATH)
    agent.epsilon = 0.0

    srpt = SRPT()
    rr   = RoundRobinTier1()

    seeds = np.random.RandomState(MASTER_SEED).randint(0, 2**31, size=N_EVAL).tolist()
    all_eps = []
    for seed in seeds:
        rng   = np.random.default_rng(seed)
        tasks = sampler.sample_episode(rng)
        all_eps.append((tasks, _make_procs(tasks)))

    # a) Agent
    print(f"\nRunning agent evaluation (n={N_EVAL}) ...")
    agent_mcts = np.array([_run_agent(agent, procs, tasks) for tasks, procs in all_eps])
    _print_stats("a) W10A-v2 AttentionDQN+PBRS α=0.05 — trace test", _stats(agent_mcts))

    # b) SRPT
    print(f"\nRunning SRPT oracle (n={N_EVAL}) ...")
    srpt_mcts = np.array([_run_policy(srpt, procs) for _, procs in all_eps])
    _print_stats("b) SRPT oracle — filtered trace test", _stats(srpt_mcts))

    # c) RR
    print(f"\nRunning Round Robin (n={N_EVAL}) ...")
    rr_mcts = np.array([_run_policy(rr, procs) for _, procs in all_eps])
    _print_stats("c) Round Robin (tier1=1.0s) — filtered trace test", _stats(rr_mcts))

    # Summary table
    as_ = _stats(agent_mcts); ss = _stats(srpt_mcts); rs = _stats(rr_mcts)
    print("\n")
    hdr = f"{'Agent':<30} | {'Mean MCT':>9} | {'Std':>8} | {'vs RR':>8} | {'vs SRPT':>9}"
    print(hdr); print("-"*len(hdr))
    print(f"{'SRPT oracle':<30} | {ss['mean']:>9.4f}s | {ss['std']:>8.4f}s | {'---':>8} | {'0.0':>9}")
    print(f"{'W10A-v2 PBRS α=0.05':<30} | {as_['mean']:>9.4f}s | {as_['std']:>8.4f}s | {as_['mean']-rs['mean']:>+8.4f}s | {as_['mean']-ss['mean']:>+9.4f}s")
    print(f"{'W9 (5-ft, no PBRS)':<30} | {'19.2234s':>9} | {'11.5843s':>8} | {'-2.0870s':>8} | {'+4.0254s':>9}")
    print(f"{'W10A PBRS α=0.20':<30} | {'22.1468s':>9} | {'12.4767s':>8} | {'+0.8364s':>8} | {'+6.9488s':>9}")
    print(f"{'Round Robin':<30} | {rs['mean']:>9.4f}s | {rs['std']:>8.4f}s | {'base':>8} | {'---':>9}")

    # d) Policy analysis
    print(f"\n\nRunning policy analysis ({N_DECISIONS:,} decisions) ...")
    records = []; rng_pa = np.random.default_rng(MASTER_SEED + 1)
    srpt2 = SRPT(); ep = 0
    while len(records) < N_DECISIONS:
        tasks = sampler.sample_episode(rng_pa)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs); env.reset()
        sv    = _encode_state(env, tasks)
        srpt2.reset(); done = False
        while not done:
            valid  = _valid_actions(env)
            action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
            srpt_pid = srpt2.select_action(env) // N_QT
            n_active = sum(1 for p in env.processes
                           if p.arrival_time <= env.current_time and not p.is_complete)
            records.append({"n_active": n_active, "dqn_qt": action % N_QT,
                            "agree_pid": action // N_QT == srpt_pid})
            _, _, done, _ = env.step(action)
            sv = _encode_state(env, tasks)
        ep += 1

    by_n = defaultdict(lambda: {"total": 0, "disagree": 0, "qt": Counter()})
    for r in records:
        n = r["n_active"]
        by_n[n]["total"] += 1; by_n[n]["qt"][r["dqn_qt"]] += 1
        if not r["agree_pid"]: by_n[n]["disagree"] += 1
    total = len(records); agree = sum(r["agree_pid"] for r in records)

    print(f"\n{'='*64}")
    print(f"d) Policy analysis — {total:,} decisions across {ep} episodes")
    print(f"{'='*64}")
    print(f"  Overall SRPT agreement : {agree}/{total}  ({100*agree/total:.1f}%)")
    print(); print(f"  {'n_act':>5}  {'agree%':>7}  {'tier0%':>7}  {'tier1%':>7}  {'tier2%':>7}  {'n':>6}")
    print("  " + "-" * 50)
    for n in sorted(by_n):
        tot = by_n[n]["total"]; dis = by_n[n]["disagree"]; qt = by_n[n]["qt"]
        t0pct = 100*qt[0]/tot
        flag  = " ← FLAG" if n >= 2 and t0pct > 95 else ""
        print(f"  {n:>5}  {100*(tot-dis)/tot:>6.1f}%  "
              f"{t0pct:>6.1f}%  {100*qt[1]/tot:>6.1f}%  {100*qt[2]/tot:>6.1f}%  {tot:>6}{flag}")
    print(f"{'='*64}")

    # e) Attention diagnostic
    print(f"\n\nRunning attention diagnostic (n_active >= 3) ...")
    attn_records = []; rng_attn = np.random.default_rng(MASTER_SEED + 2)
    while len(attn_records) < 1000:
        tasks = sampler.sample_episode(rng_attn)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs); env.reset()
        sv    = _encode_state(env, tasks); done = False
        while not done:
            valid  = _valid_actions(env)
            n_active = sum(1 for p in env.processes
                           if p.arrival_time <= env.current_time and not p.is_complete)
            action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
            if n_active >= 3:
                weights, comp_bursts, comp_vm = _get_attention_weights(agent, sv, action)
                if comp_vm.sum() >= 2:
                    valid_w = weights.copy(); valid_w[~comp_vm] = -1.0
                    top_idx = int(np.argmax(valid_w))
                    vb = [(comp_bursts[j], j) for j in range(4) if comp_vm[j]]
                    shortest_idx = min(vb, key=lambda x: x[0])[1]
                    longest_idx  = max(vb, key=lambda x: x[0])[1]
                    attn_records.append({
                        "top_on_shortest": top_idx == shortest_idx,
                        "top_on_longest":  top_idx == longest_idx,
                        "w_shortest": float(weights[shortest_idx]),
                        "w_longest":  float(weights[longest_idx]),
                    })
            _, _, done, _ = env.step(action)
            sv = _encode_state(env, tasks)

    n_a = len(attn_records)
    ts  = sum(r["top_on_shortest"] for r in attn_records)
    tl  = sum(r["top_on_longest"]  for r in attn_records)
    mws = np.mean([r["w_shortest"] for r in attn_records])
    mwl = np.mean([r["w_longest"]  for r in attn_records])

    print(f"\n{'='*60}")
    print(f"e) Attention diagnostic — n_active >= 3  (n={n_a} decisions)")
    print(f"{'='*60}")
    print(f"  Highest-attention = shortest competitor : {ts}/{n_a}  ({100*ts/n_a:.1f}%)")
    print(f"  Highest-attention = longest  competitor : {tl}/{n_a}  ({100*tl/n_a:.1f}%)")
    print(f"  Mean attention weight on shortest       : {mws:.4f}")
    print(f"  Mean attention weight on longest        : {mwl:.4f}")
    print(f"{'='*60}")

    # f) Identity bias probe
    probe = np.zeros(N_PROCESSES * D_CAND, dtype=np.float64)
    probe[0*D_CAND:0*D_CAND+D_CAND] = [0.20, 1.0, 0.0, 0.125, 0.339]
    probe[1*D_CAND:1*D_CAND+D_CAND] = [0.60, 1.0, 0.0, 0.125, 0.339]
    cand_burst = float(np.expm1(0.20 * _LOG_DENOM))
    comp_burst = float(np.expm1(0.60 * _LOG_DENOM))
    Q_cand = float(agent.forward_batch(probe[None], np.array([0*N_QT+2]))[0, 0])
    Q_comp = float(agent.forward_batch(probe[None], np.array([1*N_QT+2]))[0, 0])
    delta  = abs(Q_cand - Q_comp)

    print(f"\n{'='*60}")
    print("f) Identity bias probe")
    print(f"   Candidate P0: burst_norm=0.20 → {cand_burst:.2f}s, cpu=0.125, mem=0.339")
    print(f"   Competitor P1: burst_norm=0.60 → {comp_burst:.2f}s, cpu=0.125, mem=0.339")
    print(f"{'='*60}")
    print(f"  Q(candidate/tier2) = {Q_cand:+.6f}")
    print(f"  Q(competitor/tier2)= {Q_comp:+.6f}")
    print(f"  |Delta|            = {delta:.6f}")
    print(f"  Candidate preferred (SRPT-correct)? {Q_cand > Q_comp}")
    print(f"{'='*60}")
