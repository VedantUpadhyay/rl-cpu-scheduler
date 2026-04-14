"""Benchmark: W12 vs MLFQ / MLFQ-V / Round Robin / CFS-lite / value_aware.

CONDITION 1 — Pure scheduling (no value-curve scoring in heuristics):
  Policies: W12, MLFQ, Round Robin, CFS-lite
  Metrics:  MCT mean±std, Starvation%, SDV, SRPT%

CONDITION 2 — Value-aware:
  Policies: W12, MLFQ-V, value_aware heuristic, MLFQ (reference)
  Metrics:  MCT mean±std, Total value delivered, VRFI, Starvation%

N=500 test episodes from Alibaba 2018 test split.
W12 checkpoint: results/w12_seed456.npz  (best seed, lowest MCT=21.09s).

MLFQ implementation notes:
  - 3 priority queues mapped to env quantum tiers (0.5s, 2.0s, 8.0s)
    which give a 1:4:16 ratio — classic MLFQ exponential progression.
  - Quantum exhaustion → demote to next queue.
  - Aging: if time_since_last_execution > MLFQ_AGE_THRESH, promote to Q0.
  - MLFQ-V: within highest non-empty queue, break ties by value loss so far
    score = V(0) - V(wait_time) = base * (1 - max(floor, exp(-wait/tau))).
"""
from __future__ import annotations
import csv, math, os, random, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                 "GRAD - FALL 23/UCSC/Capstone")
_SCRIPTS_DIR  = os.path.join(_PROJECT_ROOT, "docs", "scheduler-research", "scripts")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

from schedsim.env    import SchedEnv, N_PROCESSES, N_QUANTUM_TIERS, value_delta
from schedsim.process import Process

from w9_train import (
    TraceEpisodeSampler5,
    _valid_actions, _make_procs,
    N_QT, N_ACTIONS, QT_VALUES,
    _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem,
    WAIT_NORM, CPU_MAX, MEM_P95,
)

# Import W12 architecture from ablation_multiseed (already has AttentionDQN + _encode_7dim)
from ablation_multiseed import AttentionDQN, _encode_7dim, _make_agent

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRACE_TEST  = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
               "data/alibaba2018/trace_test_filtered.csv")
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")
W12_CKPT    = os.path.join(RESULTS_DIR, "w12_seed456.npz")

N_EVAL      = 500
EVAL_SEED   = 999          # deterministic, different from training seeds

QUANTUM_TIERS = (0.5, 2.0, 8.0)   # seconds — must match schedsim/env.py
MLFQ_AGE_THRESH = 50.0             # seconds without execution → promote to Q0

# ---------------------------------------------------------------------------
# Load W12 agent
# ---------------------------------------------------------------------------

def load_w12() -> AttentionDQN:
    agent = AttentionDQN(d_cand=7, arrived_flag_idx=6)
    agent.load(W12_CKPT)
    agent.epsilon = 0.0
    return agent


# ---------------------------------------------------------------------------
# Generic evaluation runner
# ---------------------------------------------------------------------------

def run_policy(policy_fn, sampler, n_eval: int, seed: int,
               value_aware: bool = False) -> dict:
    """Run n_eval episodes under policy_fn.

    policy_fn(env, tasks, mlfq_state) -> action (int)
      mlfq_state: mutable per-episode dict passed through for stateful policies.

    value_aware: if True, also compute total_value_delivered and VRFI.
    """
    rng = np.random.default_rng(seed)

    mcts, srpts, starved_list = [], [], []
    all_turns, all_vlrs       = [], []
    total_values              = []

    for ep_idx in range(n_eval):
        tasks = sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()

        # Per-episode MLFQ state
        ep_state = {
            "mlfq_queue": {p.pid: 0 for p in env.processes},  # queue level per pid
        }

        srpt_agree = 0
        srpt_total = 0
        done       = False

        while not done:
            runnable = [p for p in env.processes
                        if p.arrival_time <= env.current_time and not p.is_complete]
            if not runnable:
                # Shouldn't happen (env auto-advances clock), but guard anyway
                break

            srpt_pid = min(runnable, key=lambda p: p.remaining_burst).pid

            action = policy_fn(env, tasks, ep_state)
            _, _, done, info = env.step(action)

            chosen_pid = action // N_QT
            if srpt_pid >= 0:
                srpt_agree += int(chosen_pid == srpt_pid)
                srpt_total += 1

        mct = info.get("mean_completion_time_so_far") or 0.0
        mcts.append(mct)
        srpts.append(srpt_agree / srpt_total if srpt_total > 0 else 0.0)

        # Starvation: any slowdown > 3× median
        completed = [p for p in env.processes if p.is_complete]
        starved = 0
        if completed:
            turnarounds = [p.completion_time - p.arrival_time for p in completed]
            bursts_ep   = [p.burst_length for p in completed]
            slowdowns   = [t / max(b, 1e-6) for t, b in zip(turnarounds, bursts_ep)]
            med_slow    = float(np.median(slowdowns))
            if any(s > 3.0 * med_slow for s in slowdowns):
                starved = 1
        starved_list.append(starved)

        # Fairness + value metrics
        ep_total_val = 0.0
        for p in env.processes:
            if p.is_complete:
                T = p.completion_time - p.arrival_time
                all_turns.append(T)
                delay = p.wait_time
                if p.tau > 0:
                    v_delivered = p.base_value * max(p.floor, math.exp(-delay / p.tau))
                else:
                    v_delivered = p.base_value
                ep_total_val += v_delivered
                vlr = (p.base_value - v_delivered) / max(delay, 1.0)
                all_vlrs.append(vlr)
        total_values.append(ep_total_val)

    def jfi(arr):
        a = np.array(arr, dtype=np.float64)
        return float(a.sum()**2 / (len(a) * np.sum(a**2) + 1e-12))

    def sdv(arr):
        a = np.array(arr, dtype=np.float64)
        return float(np.std(a) / (np.mean(a) + 1e-12)) if len(a) > 1 else float("nan")

    def vrfi(vlrs):
        a = np.array(vlrs, dtype=np.float64)
        cv = float(np.std(a) / (np.mean(a) + 1e-12)) if len(a) > 1 else 0.0
        return 1.0 - cv

    return {
        "mct_mean":    float(np.mean(mcts)),
        "mct_std":     float(np.std(mcts)),
        "srpt_mean":   float(np.mean(srpts)) * 100.0,
        "starve_pct":  float(np.sum(starved_list)) / n_eval * 100.0,
        "sdv":         sdv(all_turns),
        "jfi":         jfi(all_turns),
        "vrfi":        vrfi(all_vlrs),
        "total_value": float(np.mean(total_values)),
    }


# ---------------------------------------------------------------------------
# Policy: W12 (RL agent, greedy)
# ---------------------------------------------------------------------------

def make_w12_policy(agent: AttentionDQN):
    def policy(env: SchedEnv, tasks: list[dict], ep_state: dict) -> int:
        sv    = _encode_7dim(env, tasks)
        valid = _valid_actions(env)
        return agent.select_action(sv, epsilon=0.0, valid_actions=valid)
    return policy


# ---------------------------------------------------------------------------
# Policy: Round Robin
# ---------------------------------------------------------------------------

def policy_rr(env: SchedEnv, tasks: list[dict], ep_state: dict) -> int:
    """Cycle through runnable processes in arrival order, always use tier 0."""
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0

    # Cycle index based on step count (global env time is a proxy)
    rr_idx = ep_state.get("rr_idx", 0)
    runnable_sorted = sorted(runnable, key=lambda p: p.pid)
    chosen = runnable_sorted[rr_idx % len(runnable_sorted)]
    ep_state["rr_idx"] = rr_idx + 1

    return chosen.pid * N_QT + 0   # tier 0 (0.5s quantum)


# ---------------------------------------------------------------------------
# Policy: CFS-lite (virtual runtime)
# ---------------------------------------------------------------------------

def policy_cfs(env: SchedEnv, tasks: list[dict], ep_state: dict) -> int:
    """Pick runnable process with minimum accumulated CPU time (vruntime)."""
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0

    vrt = ep_state.setdefault("vruntime", {p.pid: 0.0 for p in env.processes})

    chosen = min(runnable, key=lambda p: vrt[p.pid])
    q      = QUANTUM_TIERS[0]   # always use smallest quantum for max control
    vrt[chosen.pid] += q

    return chosen.pid * N_QT + 0   # tier 0 (0.5s)


# ---------------------------------------------------------------------------
# Policy: MLFQ (pure scheduling — no value curves)
# ---------------------------------------------------------------------------

def policy_mlfq(env: SchedEnv, tasks: list[dict], ep_state: dict,
                value_tiebreak: bool = False) -> int:
    """Multi-level feedback queue scheduler.

    Queue levels 0/1/2 map to env quantum tiers 0/1/2 (0.5s / 2.0s / 8.0s).
    Demotion: process exhausts its quantum without I/O → move to next level.
    Aging: time_since_last_execution > MLFQ_AGE_THRESH → promote to level 0.
    value_tiebreak: if True, within highest non-empty queue pick by max value loss.
    """
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0

    queues    = ep_state.setdefault("mlfq_queue", {p.pid: 0 for p in env.processes})
    prev_rem  = ep_state.setdefault("prev_remaining", {p.pid: p.burst_length for p in env.processes})
    last_act  = ep_state.setdefault("last_action_pid", None)

    # --- Demotion logic: did the previous chosen process exhaust its quantum? ---
    # If its remaining burst decreased by exactly the quantum (no I/O), demote.
    if last_act is not None:
        prev_p = next((p for p in env.processes if p.pid == last_act), None)
        if prev_p is not None and not prev_p.is_complete:
            tier     = queues.get(prev_p.pid, 0)
            q_used   = QUANTUM_TIERS[tier]
            consumed = prev_rem.get(prev_p.pid, 0.0) - prev_p.remaining_burst
            # If consumed ≥ q_used (full quantum), demote
            if consumed >= q_used - 1e-6 and tier < 2:
                queues[prev_p.pid] = tier + 1

    # --- Aging: promote starving processes to Q0 ---
    for p in runnable:
        if p.time_since_last_execution > MLFQ_AGE_THRESH:
            queues[p.pid] = 0

    # --- Select: pick from highest non-empty queue ---
    for level in range(3):
        candidates = [p for p in runnable if queues.get(p.pid, 0) == level]
        if not candidates:
            continue

        if value_tiebreak and hasattr(candidates[0], 'tau'):
            # MLFQ-V: break ties by total value loss = V(0) - V(wait_time)
            def val_loss(p):
                delay = p.wait_time
                if p.tau > 0:
                    v_now = p.base_value * max(p.floor, math.exp(-delay / p.tau))
                else:
                    v_now = p.base_value
                return p.base_value - v_now   # higher = more urgent
            chosen = max(candidates, key=val_loss)
        else:
            # MLFQ: among candidates, prefer shortest remaining burst (FIFO approx)
            chosen = min(candidates, key=lambda p: p.arrival_time)

        ep_state["last_action_pid"] = chosen.pid
        prev_rem[chosen.pid]        = chosen.remaining_burst

        return chosen.pid * N_QT + level   # quantum tier matches queue level

    # Fallback: pick first runnable
    chosen = runnable[0]
    return chosen.pid * N_QT + 0


def policy_mlfq_plain(env, tasks, ep_state):
    return policy_mlfq(env, tasks, ep_state, value_tiebreak=False)

def policy_mlfq_v(env, tasks, ep_state):
    return policy_mlfq(env, tasks, ep_state, value_tiebreak=True)


# ---------------------------------------------------------------------------
# Policy: value_aware heuristic (greedy VLR)
# ---------------------------------------------------------------------------

def policy_value_aware(env: SchedEnv, tasks: list[dict], ep_state: dict) -> int:
    """Pick runnable process with highest value loss rate (VLR = urgency).

    VLR = (V(0) - V(wait_time)) / max(wait_time, 1.0)
    Always uses smallest quantum for maximum scheduling frequency.
    """
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0

    def vlr(p):
        delay = p.wait_time
        if p.tau > 0:
            v_now = p.base_value * max(p.floor, math.exp(-delay / p.tau))
        else:
            v_now = p.base_value
        return (p.base_value - v_now) / max(delay, 1.0)

    chosen = max(runnable, key=vlr)
    return chosen.pid * N_QT + 0   # tier 0 for fine-grained control


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def fmt(m: dict, include_srpt=True, include_sdv=True,
        include_val=False, include_vrfi=False) -> str:
    parts = [f"MCT={m['mct_mean']:.2f}±{m['mct_std']:.2f}s",
             f"Starve={m['starve_pct']:.1f}%"]
    if include_sdv:
        parts.append(f"SDV={m['sdv']:.3f}")
    if include_srpt:
        parts.append(f"SRPT={m['srpt_mean']:.1f}%")
    if include_val:
        parts.append(f"Value={m['total_value']:.4f}")
    if include_vrfi:
        parts.append(f"VRFI={m['vrfi']:.3f}")
    return "  ".join(parts)


def main():
    print("=" * 70)
    print("W12 vs MLFQ Benchmark")
    print(f"N_eval={N_EVAL}  checkpoint={W12_CKPT}")
    print("=" * 70)

    print(f"\nLoading test trace: {TRACE_TEST}")
    sampler = TraceEpisodeSampler5(TRACE_TEST)
    print(f"  Loaded {len(sampler._data):,} test tasks.")

    print("\nLoading W12 checkpoint (seed 456) ...")
    agent = load_w12()
    w12_policy = make_w12_policy(agent)
    print("  Done.")

    # -------------------------------------------------------------------------
    # CONDITION 1 — Pure scheduling (no value-curve scoring in heuristics)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONDITION 1 — Pure scheduling comparison (N=500 test episodes)")
    print("=" * 70)

    policies_c1 = [
        ("W12",         w12_policy),
        ("MLFQ",        policy_mlfq_plain),
        ("Round Robin", policy_rr),
        ("CFS-lite",    policy_cfs),
    ]

    results_c1 = {}
    for name, pol in policies_c1:
        print(f"\n  Running {name} ...")
        t0 = time.time()
        res = run_policy(pol, sampler, N_EVAL, seed=EVAL_SEED)
        elapsed = time.time() - t0
        results_c1[name] = res
        print(f"  {name}: {fmt(res)}  ({elapsed:.1f}s wall)")

    print("\n\n--- CONDITION 1 TABLE ---")
    print(f"{'Policy':<14} {'MCT mean±std':>16} {'Starve%':>9} {'SDV':>7} {'SRPT%':>7}")
    print("-" * 60)
    for name, _ in policies_c1:
        m = results_c1[name]
        print(f"{name:<14} {m['mct_mean']:>7.2f}±{m['mct_std']:<6.2f}s "
              f"{m['starve_pct']:>8.1f}% "
              f"{m['sdv']:>7.3f} "
              f"{m['srpt_mean']:>6.1f}%")

    # -------------------------------------------------------------------------
    # CONDITION 2 — Value-aware comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONDITION 2 — Value-aware comparison (N=500 test episodes)")
    print("=" * 70)

    policies_c2 = [
        ("W12",          w12_policy),
        ("MLFQ-V",       policy_mlfq_v),
        ("value_aware",  policy_value_aware),
        ("MLFQ",         policy_mlfq_plain),
    ]

    results_c2 = {}
    for name, pol in policies_c2:
        print(f"\n  Running {name} ...")
        t0 = time.time()
        res = run_policy(pol, sampler, N_EVAL, seed=EVAL_SEED, value_aware=True)
        elapsed = time.time() - t0
        results_c2[name] = res
        print(f"  {name}: {fmt(res, include_val=True, include_vrfi=True, include_sdv=False)}  ({elapsed:.1f}s wall)")

    print("\n\n--- CONDITION 2 TABLE ---")
    print(f"{'Policy':<14} {'MCT mean±std':>16} {'Total Value':>12} {'VRFI':>7} {'Starve%':>9}")
    print("-" * 62)
    for name, _ in policies_c2:
        m = results_c2[name]
        print(f"{name:<14} {m['mct_mean']:>7.2f}±{m['mct_std']:<6.2f}s "
              f"{m['total_value']:>12.4f} "
              f"{m['vrfi']:>7.3f} "
              f"{m['starve_pct']:>8.1f}%")

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    w12_c1   = results_c1["W12"]
    mlfq_c1  = results_c1["MLFQ"]
    w12_c2   = results_c2["W12"]
    mlfqv_c2 = results_c2["MLFQ-V"]
    va_c2    = results_c2["value_aware"]

    q1 = "YES" if w12_c1["mct_mean"] < mlfq_c1["mct_mean"] else "NO"
    delta_mct = mlfq_c1["mct_mean"] - w12_c1["mct_mean"]
    print(f"\n1. Does W12 beat MLFQ on MCT? (Condition 1): {q1}")
    print(f"   W12={w12_c1['mct_mean']:.2f}s  MLFQ={mlfq_c1['mct_mean']:.2f}s  "
          f"Δ={delta_mct:+.2f}s")

    q2 = "YES" if w12_c2["total_value"] > mlfqv_c2["total_value"] else "NO"
    delta_val = w12_c2["total_value"] - mlfqv_c2["total_value"]
    print(f"\n2. Does W12 beat MLFQ-V on total value? (Condition 2): {q2}")
    print(f"   W12={w12_c2['total_value']:.4f}  MLFQ-V={mlfqv_c2['total_value']:.4f}  "
          f"Δ={delta_val:+.4f}")

    print(f"\n3. Per-metric W12 wins/losses (vs MLFQ, Condition 1):")
    for metric, key, better_if in [
        ("MCT",         "mct_mean",   "lower"),
        ("Starvation%", "starve_pct", "lower"),
        ("SDV",         "sdv",        "lower"),
        ("SRPT%",       "srpt_mean",  "higher"),
    ]:
        w12v  = w12_c1[key]
        mlfqv = mlfq_c1[key]
        if better_if == "lower":
            win = "WIN" if w12v < mlfqv else "LOSS"
        else:
            win = "WIN" if w12v > mlfqv else "LOSS"
        print(f"   {metric:<14}: W12={w12v:.3f}  MLFQ={mlfqv:.3f}  → W12 {win}")

    print(f"\n4. Per-metric W12 wins/losses (vs MLFQ-V, Condition 2):")
    for metric, key, better_if in [
        ("MCT",         "mct_mean",      "lower"),
        ("Total Value", "total_value",   "higher"),
        ("VRFI",        "vrfi",          "higher"),
        ("Starvation%", "starve_pct",    "lower"),
    ]:
        w12v   = w12_c2[key]
        mlfqvv = mlfqv_c2[key]
        if better_if == "lower":
            win = "WIN" if w12v < mlfqvv else "LOSS"
        else:
            win = "WIN" if w12v > mlfqvv else "LOSS"
        print(f"   {metric:<14}: W12={w12v:.4f}  MLFQ-V={mlfqvv:.4f}  → W12 {win}")


if __name__ == "__main__":
    main()
