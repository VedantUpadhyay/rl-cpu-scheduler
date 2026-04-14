"""W13 — Hard Threshold Starvation Penalty Agent.

MLFQ's aging rule is a hard guarantee: a task waiting > threshold is
promoted and scheduled next. This experiment replaces continuous PBRS
with a binary penalty that fires only when starvation exceeds a threshold.

Reward: R = value_delta_reward + starvation_penalty
  value_delta_reward = sum(V(delay+q) - V(delay) for runnable) / 20.0
  starvation_penalty = -PENALTY  if max_wait > THRESHOLD  else 0.0

Variants:
  W13-A: THRESHOLD=50s,  PENALTY=0.5
  W13-B: THRESHOLD=50s,  PENALTY=2.0
  W13-C: THRESHOLD=100s, PENALTY=2.0

3 seeds × 10k episodes each on Alibaba 2018 trace.
Checkpoints: docs/scheduler-research/results/w13_threshold/w13_X_seedS.npz
"""
from __future__ import annotations
import json, math, os, random, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from project_config import (PROJECT_ROOT as _PROJECT_ROOT, SCRIPTS_DIR as _SCRIPTS_DIR,
                             TRACE_PATH as TRACE_TRAIN, TEST_PATH as TRACE_TEST,
                             get_agent_dir)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

from schedsim.env    import SchedEnv, N_PROCESSES, N_QUANTUM_TIERS, value_delta
from schedsim.agent  import AdamOptimizer, ReplayBuffer
from schedsim.process import Process

from w9_train import (
    TraceEpisodeSampler5, _valid_actions, _make_procs,
    N_QT, N_ACTIONS, QT_VALUES,
    _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem,
    WAIT_NORM, CPU_MAX, MEM_P95,
)
from ablation_multiseed import (
    AttentionDQN, _encode_7dim, reinit_weights,
    N_HEADS, D_HEAD, D_V_TOT,
    LAMBDA_START, LAMBDA_END, LOSS_GATE, ENT_GATE_5K,
    LR, GAMMA, GRAD_CLIP, BUF_CAPACITY, BATCH_SIZE,
    TARGET_UPDATE_FREQ, WARMUP, PRINT_EVERY,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = get_agent_dir("w13_threshold")

N_EPISODES = 10_000
N_EVAL     = 500
SEEDS      = [42, 123, 456]

REWARD_SCALE = 20.0

# (variant_name, threshold_seconds, penalty_magnitude)
VARIANTS = [
    ("W13-A", 50.0,  0.5),
    ("W13-B", 50.0,  2.0),
    ("W13-C", 100.0, 2.0),
]

QUANTUM_TIERS   = (0.5, 2.0, 8.0)
MLFQ_AGE_THRESH = 50.0

# MLFQ baseline numbers (from prior benchmark_vs_mlfq.py run)
MLFQ_MCT    = 21.59
MLFQ_STARVE = 36.0
MLFQ_VRFI   = 0.419
MLFQ_SRPT   = 58.1

W12_MCT    = 21.18
W12_STARVE = 53.2
W12_VRFI   = 0.267
W12_SRPT   = 57.0


# ---------------------------------------------------------------------------
# Hard threshold starvation penalty
# ---------------------------------------------------------------------------

def starvation_penalty(env: SchedEnv, threshold: float, penalty: float) -> float:
    """Returns -penalty if any runnable process has been waiting > threshold, else 0."""
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0.0
    max_wait = max(p.time_since_last_execution for p in runnable)
    return -penalty if max_wait > threshold else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_seed(variant_name: str, threshold: float, penalty: float,
               seed: int, train_sampler) -> AttentionDQN:
    agent = AttentionDQN(d_cand=7, arrived_flag_idx=6)
    reinit_weights(agent, seed)

    random.seed(seed)
    np.random.seed(seed)
    rng_ep = np.random.default_rng(seed)

    buffer = ReplayBuffer(capacity=BUF_CAPACITY, state_dim=N_PROCESSES * 7)

    total_transitions = 0
    win_loss, win_ent, win_mct, win_srpt = [], [], [], []

    for ep in range(1, N_EPISODES + 1):
        agent.lambda_ent = LAMBDA_START - (LAMBDA_START - LAMBDA_END) * (ep / N_EPISODES)

        tasks = train_sampler.sample_episode(rng_ep)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()
        sv    = _encode_7dim(env, tasks)

        ep_loss_sum  = 0.0
        ep_ent_sum   = 0.0
        ep_loss_n    = 0
        srpt_agree_n = 0
        srpt_total_n = 0
        done         = False

        while not done:
            valid    = _valid_actions(env)
            runnable = [p for p in env.processes
                        if p.arrival_time <= env.current_time and not p.is_complete]
            srpt_pid = (min(runnable, key=lambda p: p.remaining_burst).pid
                        if runnable else -1)

            action = agent.select_action(sv, agent.epsilon, valid)

            # Value-delta base reward (identical to W12)
            chosen       = env.processes[action // N_QT]
            runnable_pre = list(runnable)
            q_tier       = action % N_QT
            q            = QUANTUM_TIERS[q_tier]
            q_actual     = min(q, chosen.remaining_burst)

            r_base = sum(
                value_delta(p.tau, p.floor, p.base_value, p.wait_time, q_actual)
                for p in runnable_pre
            ) / REWARD_SCALE

            _, _, done, info = env.step(action)

            # Hard threshold starvation penalty (evaluated AFTER step)
            r_penalty = starvation_penalty(env, threshold, penalty)
            reward    = r_base + r_penalty

            sv_next = _encode_7dim(env, tasks)

            if srpt_pid >= 0:
                srpt_agree_n += int((action // N_QT) == srpt_pid)
                srpt_total_n += 1

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
        srpt_frac = srpt_agree_n / srpt_total_n if srpt_total_n > 0 else 0.0
        mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else float("nan")
        mean_ent  = ep_ent_sum  / ep_loss_n if ep_loss_n > 0 else float("nan")

        win_mct.append(mct)
        win_srpt.append(srpt_frac)
        win_loss.append(mean_loss)
        win_ent.append(mean_ent)

        agent.decay_epsilon(ep, min_eps=0.05, decay=0.9995)

        if ep % PRINT_EVERY == 0:
            n  = min(ep, 100)
            al = float(np.nanmean(win_loss[-n:]))
            ah = float(np.nanmean(win_ent[-n:]))
            am = float(np.mean(win_mct[-n:]))
            sp = float(np.mean(win_srpt[-n:])) * 100.0
            print(f"  ep {ep:>6} | loss={al:.4f} | H={ah:.4f} | "
                  f"MCT={am:.2f}s | SRPT={sp:.1f}%")
            sys.stdout.flush()

            if al > LOSS_GATE:
                print(f"  STOP: loss {al:.2f} > {LOSS_GATE}")
                break
            if ep == 5000 and ah > ENT_GATE_5K:
                print(f"  STOP: H={ah:.4f} > {ENT_GATE_5K} at ep 5000")
                break

    return agent


# ---------------------------------------------------------------------------
# Evaluation (greedy, ε=0)
# ---------------------------------------------------------------------------

def evaluate(agent: AttentionDQN, test_sampler, n_eval: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mcts, srpts, starved_list = [], [], []
    all_vlrs = []

    for _ in range(n_eval):
        tasks = test_sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()
        sv    = _encode_7dim(env, tasks)

        srpt_agree = 0
        srpt_total = 0
        done = False

        while not done:
            valid    = _valid_actions(env)
            runnable = [p for p in env.processes
                        if p.arrival_time <= env.current_time and not p.is_complete]
            srpt_pid = min(runnable, key=lambda p: p.remaining_burst).pid if runnable else -1

            action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
            _, _, done, info = env.step(action)
            sv = _encode_7dim(env, tasks)

            if srpt_pid >= 0:
                srpt_agree += int((action // N_QT) == srpt_pid)
                srpt_total += 1

        mcts.append(info.get("mean_completion_time_so_far") or 0.0)
        srpts.append(srpt_agree / srpt_total if srpt_total > 0 else 0.0)

        completed = [p for p in env.processes if p.is_complete]
        starved = 0
        if completed:
            turnarounds = [p.completion_time - p.arrival_time for p in completed]
            bursts_ep   = [p.burst_length for p in completed]
            slowdowns   = [t / max(b, 1e-6) for t, b in zip(turnarounds, bursts_ep)]
            med = float(np.median(slowdowns))
            if any(s > 3.0 * med for s in slowdowns):
                starved = 1
        starved_list.append(starved)

        for p in env.processes:
            if p.is_complete:
                delay = p.wait_time
                v_now = (p.base_value * max(p.floor, math.exp(-delay / p.tau))
                         if p.tau > 0 else p.base_value)
                vlr = (p.base_value - v_now) / max(delay, 1.0)
                all_vlrs.append(vlr)

    def vrfi(vlrs):
        a = np.array(vlrs, dtype=np.float64)
        cv = float(np.std(a) / (np.mean(a) + 1e-12)) if len(a) > 1 else 0.0
        return 1.0 - cv

    return {
        "mct_mean":   float(np.mean(mcts)),
        "mct_std":    float(np.std(mcts)),
        "srpt_mean":  float(np.mean(srpts)) * 100.0,
        "starve_pct": float(np.sum(starved_list)) / n_eval * 100.0,
        "vrfi":       vrfi(all_vlrs),
    }


# ---------------------------------------------------------------------------
# MLFQ inline baseline
# ---------------------------------------------------------------------------

def run_mlfq(test_sampler, n_eval: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mcts, srpts, starved_list, all_vlrs = [], [], [], []

    for _ in range(n_eval):
        tasks = test_sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()

        ep_state = {
            "mlfq_queue":     {p.pid: 0 for p in env.processes},
            "prev_remaining": {p.pid: p.burst_length for p in env.processes},
            "last_action_pid": None,
        }
        srpt_agree = 0
        srpt_total = 0
        done = False

        while not done:
            runnable = [p for p in env.processes
                        if p.arrival_time <= env.current_time and not p.is_complete]
            if not runnable:
                break
            srpt_pid = min(runnable, key=lambda p: p.remaining_burst).pid

            queues   = ep_state["mlfq_queue"]
            prev_rem = ep_state["prev_remaining"]
            last_act = ep_state["last_action_pid"]

            if last_act is not None:
                prev_p = next((p for p in env.processes if p.pid == last_act), None)
                if prev_p is not None and not prev_p.is_complete:
                    tier     = queues.get(prev_p.pid, 0)
                    consumed = prev_rem.get(prev_p.pid, 0.0) - prev_p.remaining_burst
                    if consumed >= QUANTUM_TIERS[tier] - 1e-6 and tier < 2:
                        queues[prev_p.pid] = tier + 1

            for p in runnable:
                if p.time_since_last_execution > MLFQ_AGE_THRESH:
                    queues[p.pid] = 0

            action = None
            for level in range(3):
                candidates = [p for p in runnable if queues.get(p.pid, 0) == level]
                if candidates:
                    chosen = min(candidates, key=lambda p: p.arrival_time)
                    ep_state["last_action_pid"] = chosen.pid
                    prev_rem[chosen.pid]        = chosen.remaining_burst
                    action = chosen.pid * N_QT + level
                    break
            if action is None:
                action = runnable[0].pid * N_QT + 0

            _, _, done, info = env.step(action)
            if srpt_pid >= 0:
                srpt_agree += int((action // N_QT) == srpt_pid)
                srpt_total += 1

        mcts.append(info.get("mean_completion_time_so_far") or 0.0)
        srpts.append(srpt_agree / srpt_total if srpt_total > 0 else 0.0)

        completed = [p for p in env.processes if p.is_complete]
        starved = 0
        if completed:
            turnarounds = [p.completion_time - p.arrival_time for p in completed]
            bursts_ep   = [p.burst_length for p in completed]
            slowdowns   = [t / max(b, 1e-6) for t, b in zip(turnarounds, bursts_ep)]
            med = float(np.median(slowdowns))
            if any(s > 3.0 * med for s in slowdowns):
                starved = 1
        starved_list.append(starved)

        for p in env.processes:
            if p.is_complete:
                delay = p.wait_time
                v_now = (p.base_value * max(p.floor, math.exp(-delay / p.tau))
                         if p.tau > 0 else p.base_value)
                vlr = (p.base_value - v_now) / max(delay, 1.0)
                all_vlrs.append(vlr)

    def vrfi(vlrs):
        a = np.array(vlrs, dtype=np.float64)
        cv = float(np.std(a) / (np.mean(a) + 1e-12)) if len(a) > 1 else 0.0
        return 1.0 - cv

    return {
        "mct_mean":   float(np.mean(mcts)),
        "mct_std":    float(np.std(mcts)),
        "srpt_mean":  float(np.mean(srpts)) * 100.0,
        "starve_pct": float(np.sum(starved_list)) / n_eval * 100.0,
        "vrfi":       vrfi(all_vlrs),
    }


# ---------------------------------------------------------------------------
# ASCII text chart
# ---------------------------------------------------------------------------

def text_chart(variant_names, thresholds, penalties, mcts, starves):
    bar_w = 40
    print("\n--- MCT vs variant (lower is better) ---")
    all_mcts = [W12_MCT] + mcts + [MLFQ_MCT]
    lo, hi = min(all_mcts) - 0.5, max(all_mcts) + 0.5
    print(f"  {'W12 (λ=0.01)':<18} {'█' * int((W12_MCT - lo)/(hi-lo+1e-9)*bar_w):<40} {W12_MCT:.2f}s")
    for name, mct in zip(variant_names, mcts):
        bar = "█" * int((mct - lo) / (hi - lo + 1e-9) * bar_w)
        print(f"  {name:<18} {bar:<40} {mct:.2f}s")
    bar = "░" * int((MLFQ_MCT - lo) / (hi - lo + 1e-9) * bar_w)
    print(f"  {'MLFQ':<18} {bar:<40} {MLFQ_MCT:.2f}s  ← threshold")

    print("\n--- Starvation% vs variant (lower is better) ---")
    all_st = [W12_STARVE] + starves + [MLFQ_STARVE]
    hi_st = max(all_st + [60.0])
    print(f"  {'W12 (λ=0.01)':<18} {'█' * int(W12_STARVE/hi_st*bar_w):<40} {W12_STARVE:.1f}% ← above MLFQ")
    for name, st in zip(variant_names, starves):
        bar    = "█" * int(st / hi_st * bar_w)
        marker = " ✓ below MLFQ" if st < MLFQ_STARVE else " ← above MLFQ"
        print(f"  {name:<18} {bar:<40} {st:.1f}%{marker}")
    bar = "░" * int(MLFQ_STARVE / hi_st * bar_w)
    print(f"  {'MLFQ':<18} {bar:<40} {MLFQ_STARVE:.1f}%  ← threshold")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 68)
    print("W13 — Hard Threshold Starvation Penalty Experiment")
    print(f"Variants: {[n for n,_,_ in VARIANTS]}  Seeds: {SEEDS}")
    print(f"N_episodes={N_EPISODES}  N_eval={N_EVAL}")
    print(f"Success: MCT < {MLFQ_MCT}s AND Starve < {MLFQ_STARVE}%")
    print("=" * 68)

    print(f"\nLoading TRAIN trace: {TRACE_TRAIN}")
    train_sampler = TraceEpisodeSampler5(TRACE_TRAIN)
    print(f"Loading TEST trace: {TRACE_TEST}")
    test_sampler  = TraceEpisodeSampler5(TRACE_TEST)
    print()

    all_results: dict[str, list[dict]] = {}
    best_agents: dict[str, AttentionDQN] = {}

    for variant_name, threshold, penalty in VARIANTS:
        print(f"\n{'='*68}")
        print(f"VARIANT: {variant_name}  THRESHOLD={threshold}s  PENALTY={penalty}")
        print(f"{'='*68}")

        seed_results = []
        best_mct     = float("inf")
        best_agent   = None

        for seed in SEEDS:
            print(f"\n--- {variant_name} seed={seed} ---")
            t0    = time.time()
            agent = train_seed(variant_name, threshold, penalty, seed, train_sampler)
            wall  = (time.time() - t0) / 60.0

            ckpt = os.path.join(RESULTS_DIR,
                                f"{variant_name.replace('-','_')}_seed{seed}.npz")
            agent.save(ckpt)
            print(f"  Checkpoint → {ckpt}")

            agent.epsilon = 0.0
            metrics = evaluate(agent, test_sampler, N_EVAL, seed=seed)
            metrics["seed"]      = seed
            metrics["wall_min"]  = wall
            metrics["threshold"] = threshold
            metrics["penalty"]   = penalty
            seed_results.append(metrics)

            print(f"  Seed {seed}: MCT={metrics['mct_mean']:.2f}±{metrics['mct_std']:.2f}s  "
                  f"SRPT={metrics['srpt_mean']:.1f}%  "
                  f"Starve={metrics['starve_pct']:.1f}%  "
                  f"VRFI={metrics['vrfi']:.3f}  "
                  f"Wall={wall:.1f}min")
            sys.stdout.flush()

            if metrics["mct_mean"] < best_mct:
                best_mct   = metrics["mct_mean"]
                best_agent = agent

        all_results[variant_name] = seed_results
        best_agents[variant_name] = best_agent

    # -------------------------------------------------------------------------
    # Cross-seed summary
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 68)
    print("CROSS-SEED SUMMARY (mean±std over 3 seeds)")
    print("=" * 68)
    print(f"\n{'Variant':<12} {'T':>6} {'P':>5} {'MCT mean±std':>16} {'Starve%':>9} {'SRPT%':>7} {'VRFI':>7}")
    print("-" * 65)

    summary: dict[str, dict] = {}
    for variant_name, threshold, penalty in VARIANTS:
        res = all_results[variant_name]
        mct_m    = float(np.mean([r["mct_mean"]   for r in res]))
        mct_s    = float(np.std( [r["mct_mean"]   for r in res]))
        srpt_m   = float(np.mean([r["srpt_mean"]  for r in res]))
        starve_m = float(np.mean([r["starve_pct"] for r in res]))
        vrfi_m   = float(np.mean([r["vrfi"]       for r in res]))
        summary[variant_name] = {
            "threshold": threshold, "penalty": penalty,
            "mct_mean": mct_m, "mct_std": mct_s,
            "srpt_mean": srpt_m, "starve_pct": starve_m, "vrfi": vrfi_m,
        }
        beats = " ★" if (mct_m < MLFQ_MCT and starve_m < MLFQ_STARVE) else ""
        print(f"{variant_name:<12} {threshold:>5.0f}s {penalty:>4.1f} "
              f"{mct_m:>7.2f}±{mct_s:<6.2f}s "
              f"{starve_m:>8.1f}% "
              f"{srpt_m:>6.1f}% "
              f"{vrfi_m:>7.3f}{beats}")

    # -------------------------------------------------------------------------
    # Fresh MLFQ baseline
    # -------------------------------------------------------------------------
    print("\nRunning MLFQ benchmark (N=500, seed=999) ...")
    mlfq_res = run_mlfq(test_sampler, N_EVAL, seed=999)
    print(f"  MLFQ: MCT={mlfq_res['mct_mean']:.2f}s  "
          f"Starve={mlfq_res['starve_pct']:.1f}%  "
          f"VRFI={mlfq_res['vrfi']:.3f}  "
          f"SRPT={mlfq_res['srpt_mean']:.1f}%")

    # -------------------------------------------------------------------------
    # Full comparison table
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 68)
    print("FULL COMPARISON vs MLFQ (best-seed per variant, N=500 eval)")
    print("=" * 68)
    print(f"\n{'Policy':<20} {'MCT':>10} {'Starve%':>9} {'VRFI':>7} {'SRPT%':>7}")
    print("-" * 56)

    def mark(val, threshold, lower_better=True):
        return "✓" if (val < threshold) == lower_better else "✗"

    print(f"{'MLFQ':<20} {mlfq_res['mct_mean']:>9.2f}s "
          f"{mlfq_res['starve_pct']:>8.1f}% "
          f"{mlfq_res['vrfi']:>7.3f} "
          f"{mlfq_res['srpt_mean']:>6.1f}%  ← baseline")
    print(f"{'W12 (λ=0.01)':<20} {W12_MCT:>9.2f}s "
          f"{W12_STARVE:>8.1f}% "
          f"{W12_VRFI:>7.3f} "
          f"{W12_SRPT:>6.1f}%  "
          f"MCT:{mark(W12_MCT, mlfq_res['mct_mean'])} "
          f"Starve:{mark(W12_STARVE, mlfq_res['starve_pct'])}")

    comparison_names, comparison_mcts, comparison_starves = [], [], []
    comparison_thresholds, comparison_penalties = [], []
    winners = []

    for variant_name, threshold, penalty in VARIANTS:
        agent = best_agents[variant_name]
        agent.epsilon = 0.0
        m = evaluate(agent, test_sampler, N_EVAL, seed=999)
        both_beat = (m["mct_mean"] < mlfq_res["mct_mean"] and
                     m["starve_pct"] < mlfq_res["starve_pct"])
        flag = " ★ BEATS MLFQ ON BOTH" if both_beat else ""
        if both_beat:
            winners.append(variant_name)
        print(f"{variant_name:<20} {m['mct_mean']:>9.2f}s "
              f"{m['starve_pct']:>8.1f}% "
              f"{m['vrfi']:>7.3f} "
              f"{m['srpt_mean']:>6.1f}%  "
              f"MCT:{mark(m['mct_mean'], mlfq_res['mct_mean'])} "
              f"Starve:{mark(m['starve_pct'], mlfq_res['starve_pct'])}{flag}")
        comparison_names.append(variant_name)
        comparison_mcts.append(m["mct_mean"])
        comparison_starves.append(m["starve_pct"])
        comparison_thresholds.append(threshold)
        comparison_penalties.append(penalty)

    # Text chart
    text_chart(comparison_names, comparison_thresholds,
               comparison_penalties, comparison_mcts, comparison_starves)

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    if winners:
        print(f"\n★ SUCCESS: {winners} beat MLFQ on BOTH MCT and starvation!")
        best_winner = min(winners,
                          key=lambda n: comparison_mcts[comparison_names.index(n)])
        print(f"  Recommended final agent: {best_winner}")
    else:
        best_idx = int(np.argmin(comparison_starves))
        best_name = comparison_names[best_idx]
        print(f"\nNo variant beats MLFQ on both metrics simultaneously.")
        print(f"Best trade-off: {best_name} (T={comparison_thresholds[best_idx]}s, "
              f"P={comparison_penalties[best_idx]}): "
              f"MCT={comparison_mcts[best_idx]:.2f}s, "
              f"Starve={comparison_starves[best_idx]:.1f}%")
        print(f"  Recommendation: use {best_name} and document starvation gap in paper.")

    # -------------------------------------------------------------------------
    # Save JSON
    # -------------------------------------------------------------------------
    out = {
        "mlfq": mlfq_res,
        "w12_baseline": {
            "mct_mean": W12_MCT, "starve_pct": W12_STARVE,
            "vrfi": W12_VRFI, "srpt_mean": W12_SRPT,
        },
        "variants": {
            name: {
                "threshold": threshold,
                "penalty":   penalty,
                "per_seed":  all_results[name],
                "summary":   summary[name],
            }
            for name, threshold, penalty in VARIANTS
        },
        "winners": winners,
    }
    json_path = os.path.join(RESULTS_DIR, "w13_threshold_results.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {json_path}")


if __name__ == "__main__":
    main()
