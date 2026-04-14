"""W12 PBRS Lambda Tuning — find starvation threshold without MCT regression.

Trains 4 variants of W12 (identical architecture, different PBRS lambda):
  W12-λ1: λ=0.05
  W12-λ2: λ=0.10
  W12-λ3: λ=0.25
  W12-λ4: λ=0.50
  (W12 baseline: λ=0.01, already trained — loaded from results/w12_seed456.npz)

3 seeds × 10,000 episodes each on Alibaba 2018 training trace.
Eval: N=500 test episodes at ε=0.
Checkpoints: docs/scheduler-research/results/w12_lambda_tuning/w12_lN_seedS.npz

MLFQ baseline (from prior benchmark):
  MCT=21.59s  Starve=36.0%  VRFI=0.420  SRPT=58.1%

Target: find λ where starvation < 36.0% AND MCT < 21.59s.
"""
from __future__ import annotations
import csv, json, math, os, random, sys, time
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
RESULTS_DIR  = get_agent_dir("w12_lambda_tuning")
MAIN_RESULTS = os.path.join(_PROJECT_ROOT, "results")   # for w12_seed456.npz

N_EPISODES = 10_000
N_EVAL     = 500
SEEDS      = [42, 123, 456]

REWARD_SCALE_W12 = 20.0   # same as W12

# Lambda variants (W12 baseline λ=0.01 already trained)
LAMBDA_VARIANTS = [
    ("W12-λ1", 0.05),
    ("W12-λ2", 0.10),
    ("W12-λ3", 0.25),
    ("W12-λ4", 0.50),
]

# MLFQ baseline numbers from prior benchmark run
MLFQ_MCT    = 21.59
MLFQ_STARVE = 36.0
MLFQ_VRFI   = 0.420
MLFQ_SRPT   = 58.1

MLFQ_AGE_THRESH = 50.0
QUANTUM_TIERS   = (0.5, 2.0, 8.0)


# ---------------------------------------------------------------------------
# PBRS potential
# ---------------------------------------------------------------------------

def phi(env: SchedEnv, pbrs_lambda: float) -> float:
    runnable = [p for p in env.processes
                if p.arrival_time <= env.current_time and not p.is_complete]
    if not runnable:
        return 0.0
    return -pbrs_lambda * max(p.time_since_last_execution for p in runnable)


# ---------------------------------------------------------------------------
# Training loop — W12 variant with configurable lambda
# ---------------------------------------------------------------------------

def train_seed(variant_name: str, pbrs_lambda: float,
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

        # Pre-step PBRS potential
        phi_s = phi(env, pbrs_lambda)

        while not done:
            valid   = _valid_actions(env)
            runnable = [p for p in env.processes
                        if p.arrival_time <= env.current_time and not p.is_complete]
            srpt_pid = (min(runnable, key=lambda p: p.remaining_burst).pid
                        if runnable else -1)

            action = agent.select_action(sv, agent.epsilon, valid)

            # Value-delta base reward
            chosen = env.processes[action // N_QT]
            runnable_pre = list(runnable)
            q_tier = action % N_QT
            q = QUANTUM_TIERS[q_tier]
            q_actual = min(q, chosen.remaining_burst)

            # Compute value delta before step
            r_base = sum(
                value_delta(p.tau, p.floor, p.base_value, p.wait_time, q_actual)
                for p in runnable_pre
            ) / REWARD_SCALE_W12

            _, _, done, info = env.step(action)

            # PBRS shaping
            phi_s_next = phi(env, pbrs_lambda)
            reward = r_base + phi_s_next - phi_s
            phi_s  = phi_s_next

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
# Evaluation (identical to ablation_multiseed.evaluate_agent for W12)
# ---------------------------------------------------------------------------

def evaluate(agent: AttentionDQN, test_sampler, n_eval: int,
             seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mcts, srpts, starved_list = [], [], []
    all_turns, all_vlrs = [], []

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
                all_turns.append(p.completion_time - p.arrival_time)
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
# MLFQ policy (inline — from benchmark_vs_mlfq.py)
# ---------------------------------------------------------------------------

def run_mlfq(test_sampler, n_eval: int, seed: int) -> dict:
    """Run MLFQ policy to get fresh baseline on same RNG seed."""
    rng = np.random.default_rng(seed)
    mcts, srpts, starved_list = [], [], []
    all_turns, all_vlrs = [], []

    for _ in range(n_eval):
        tasks = test_sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()

        ep_state = {
            "mlfq_queue":    {p.pid: 0 for p in env.processes},
            "prev_remaining":{p.pid: p.burst_length for p in env.processes},
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

            # MLFQ action
            queues   = ep_state["mlfq_queue"]
            prev_rem = ep_state["prev_remaining"]
            last_act = ep_state["last_action_pid"]

            if last_act is not None:
                prev_p = next((p for p in env.processes if p.pid == last_act), None)
                if prev_p is not None and not prev_p.is_complete:
                    tier = queues.get(prev_p.pid, 0)
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
                all_turns.append(p.completion_time - p.arrival_time)
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

def text_chart(lambdas: list[float], mcts: list[float],
               starves: list[float]) -> None:
    """Print simple text bar charts for MCT and Starvation% vs lambda."""
    print("\n--- MCT vs λ (lower is better) ---")
    mct_min, mct_max = min(mcts + [MLFQ_MCT]) - 0.5, max(mcts + [MLFQ_MCT]) + 0.5
    bar_w = 40
    for lam, mct in zip([0.01] + lambdas, [21.18] + mcts):
        frac = (mct - mct_min) / (mct_max - mct_min + 1e-9)
        bar  = "█" * int(frac * bar_w)
        label = f"λ={lam:.2f}"
        print(f"  {label:<10} {bar:<40} {mct:.2f}s")
    frac = (MLFQ_MCT - mct_min) / (mct_max - mct_min + 1e-9)
    bar  = "░" * int(frac * bar_w)
    print(f"  {'MLFQ':<10} {bar:<40} {MLFQ_MCT:.2f}s  ← threshold")

    print("\n--- Starvation% vs λ (lower is better) ---")
    st_min, st_max = 0.0, max(starves + [MLFQ_STARVE, 60.0])
    for lam, st in zip([0.01] + lambdas, [53.2] + starves):
        frac = st / (st_max + 1e-9)
        bar  = "█" * int(frac * bar_w)
        label = f"λ={lam:.2f}"
        marker = " ← above MLFQ" if st > MLFQ_STARVE else " ✓ below MLFQ"
        print(f"  {label:<10} {bar:<40} {st:.1f}%{marker}")
    frac = MLFQ_STARVE / (st_max + 1e-9)
    bar  = "░" * int(frac * bar_w)
    print(f"  {'MLFQ':<10} {bar:<40} {MLFQ_STARVE:.1f}%  ← threshold")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 68)
    print("W12 PBRS Lambda Tuning Experiment")
    print(f"Variants: {[n for n,_ in LAMBDA_VARIANTS]}  Seeds: {SEEDS}")
    print(f"N_episodes={N_EPISODES}  N_eval={N_EVAL}")
    print("=" * 68)

    print(f"\nLoading TRAIN trace: {TRACE_TRAIN}")
    train_sampler = TraceEpisodeSampler5(TRACE_TRAIN)
    print(f"Loading TEST trace: {TRACE_TEST}")
    test_sampler  = TraceEpisodeSampler5(TRACE_TEST)
    print()

    # {variant_name: [seed_result_dict, ...]}
    all_results: dict[str, list[dict]] = {}
    best_agents: dict[str, AttentionDQN] = {}  # best seed per variant

    for variant_name, lam in LAMBDA_VARIANTS:
        print(f"\n{'='*68}")
        print(f"VARIANT: {variant_name}  λ={lam}")
        print(f"{'='*68}")

        seed_results = []
        best_mct     = float("inf")
        best_agent   = None

        for seed in SEEDS:
            print(f"\n--- {variant_name} seed={seed} ---")
            t0    = time.time()
            agent = train_seed(variant_name, lam, seed, train_sampler)
            wall  = (time.time() - t0) / 60.0

            ckpt = os.path.join(RESULTS_DIR, f"{variant_name.replace('-','_').replace('λ','l')}_seed{seed}.npz")
            agent.save(ckpt)
            print(f"  Checkpoint → {ckpt}")

            agent.epsilon = 0.0
            metrics = evaluate(agent, test_sampler, N_EVAL, seed=seed)
            metrics["seed"]     = seed
            metrics["wall_min"] = wall
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
    print("CROSS-SEED SUMMARY")
    print("=" * 68)
    print(f"\n{'Variant':<12} {'MCT mean±std':>16} {'Starve%':>9} {'SRPT%':>7} {'VRFI':>7}")
    print("-" * 57)

    summary_means: dict[str, dict] = {}
    for variant_name, lam in LAMBDA_VARIANTS:
        res = all_results[variant_name]
        mct_m    = float(np.mean([r["mct_mean"]   for r in res]))
        mct_s    = float(np.std( [r["mct_mean"]   for r in res]))
        srpt_m   = float(np.mean([r["srpt_mean"]  for r in res]))
        starve_m = float(np.mean([r["starve_pct"] for r in res]))
        vrfi_m   = float(np.mean([r["vrfi"]       for r in res]))
        summary_means[variant_name] = {
            "lambda": lam, "mct_mean": mct_m, "mct_std": mct_s,
            "srpt_mean": srpt_m, "starve_pct": starve_m, "vrfi": vrfi_m,
        }
        print(f"{variant_name:<12} {mct_m:>7.2f}±{mct_s:<6.2f}s "
              f"{starve_m:>8.1f}% "
              f"{srpt_m:>6.1f}% "
              f"{vrfi_m:>7.3f}")

    # -------------------------------------------------------------------------
    # Re-run MLFQ on same test RNG for fair comparison
    # -------------------------------------------------------------------------
    print("\nRunning MLFQ benchmark (N=500, fresh eval) ...")
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
    print(f"\n{'Policy':<16} {'MCT':>10} {'Starve%':>9} {'VRFI':>7} {'SRPT%':>7}")
    print("-" * 52)

    def mark(val, threshold, lower_better=True):
        if lower_better:
            return "✓" if val < threshold else "✗"
        return "✓" if val > threshold else "✗"

    # W12 baseline (λ=0.01, seed 456, from prior run)
    print(f"{'MLFQ':<16} {mlfq_res['mct_mean']:>9.2f}s "
          f"{mlfq_res['starve_pct']:>8.1f}% "
          f"{mlfq_res['vrfi']:>7.3f} "
          f"{mlfq_res['srpt_mean']:>6.1f}%  ← baseline")

    # W12 baseline from benchmark (λ=0.01)
    w12_base_mct    = 21.18
    w12_base_starve = 53.2
    w12_base_vrfi   = 0.267
    w12_base_srpt   = 57.0
    mct_m_mark   = mark(w12_base_mct,    mlfq_res["mct_mean"])
    starve_m_mark = mark(w12_base_starve, mlfq_res["starve_pct"])
    print(f"{'W12 (λ=0.01)':<16} {w12_base_mct:>9.2f}s "
          f"{w12_base_starve:>8.1f}% "
          f"{w12_base_vrfi:>7.3f} "
          f"{w12_base_srpt:>6.1f}%  MCT:{mct_m_mark} Starve:{starve_m_mark}")

    # Eval best agent per variant
    comparison_lambdas = []
    comparison_mcts    = []
    comparison_starves = []

    for variant_name, lam in LAMBDA_VARIANTS:
        agent = best_agents[variant_name]
        agent.epsilon = 0.0
        m = evaluate(agent, test_sampler, N_EVAL, seed=999)
        mct_mark    = mark(m["mct_mean"],   mlfq_res["mct_mean"])
        starve_mark = mark(m["starve_pct"], mlfq_res["starve_pct"])
        both_beat   = (m["mct_mean"] < mlfq_res["mct_mean"] and
                       m["starve_pct"] < mlfq_res["starve_pct"])
        flag        = " ★ BEATS MLFQ ON BOTH" if both_beat else ""
        print(f"{variant_name:<16} {m['mct_mean']:>9.2f}s "
              f"{m['starve_pct']:>8.1f}% "
              f"{m['vrfi']:>7.3f} "
              f"{m['srpt_mean']:>6.1f}%  MCT:{mct_mark} Starve:{starve_mark}{flag}")
        comparison_lambdas.append(lam)
        comparison_mcts.append(m["mct_mean"])
        comparison_starves.append(m["starve_pct"])

    # -------------------------------------------------------------------------
    # Text chart
    # -------------------------------------------------------------------------
    text_chart(comparison_lambdas, comparison_mcts, comparison_starves)

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)

    winners = []
    for (variant_name, lam), mct, starve in zip(
            LAMBDA_VARIANTS, comparison_mcts, comparison_starves):
        if mct < mlfq_res["mct_mean"] and starve < mlfq_res["starve_pct"]:
            winners.append((variant_name, lam, mct, starve))

    if winners:
        print(f"\nλ values that beat MLFQ on BOTH MCT and starvation:")
        for vn, lam, mct, st in winners:
            print(f"  {vn} (λ={lam}): MCT={mct:.2f}s, Starve={st:.1f}%")
        # Recommend: lowest lambda that beats MLFQ (least shaping = most conservative)
        rec = winners[0]
        print(f"\nRecommended λ: {rec[1]} ({rec[0]}) — lowest lambda that beats MLFQ on both metrics")
        print(f"  MCT={rec[2]:.2f}s vs MLFQ {mlfq_res['mct_mean']:.2f}s  "
              f"(Δ={mlfq_res['mct_mean']-rec[2]:+.2f}s)")
        print(f"  Starve={rec[3]:.1f}% vs MLFQ {mlfq_res['starve_pct']:.1f}%")
    else:
        print("\nNo λ variant beats MLFQ on BOTH metrics simultaneously.")
        # Find best trade-off: minimize (mct/mlfq_mct + starve/mlfq_starve)
        scores = [(m / mlfq_res["mct_mean"] + s / mlfq_res["starve_pct"],
                   vn, lam, m, s)
                  for (vn, lam), m, s in zip(LAMBDA_VARIANTS, comparison_mcts, comparison_starves)]
        best = min(scores, key=lambda x: x[0])
        print(f"\nBest trade-off: {best[1]} (λ={best[2]}): MCT={best[3]:.2f}s, Starve={best[4]:.1f}%")
        print("  Recommendation: use this λ and note the MCT/starvation trade-off in paper.")

    # -------------------------------------------------------------------------
    # Save full results JSON
    # -------------------------------------------------------------------------
    json_path = os.path.join(RESULTS_DIR, "lambda_tuning_results.json")
    out = {
        "mlfq": mlfq_res,
        "w12_baseline": {
            "lambda": 0.01,
            "mct_mean": w12_base_mct,
            "starve_pct": w12_base_starve,
            "vrfi": w12_base_vrfi,
            "srpt_mean": w12_base_srpt,
        },
        "variants": {
            vn: {
                "lambda": lam,
                "per_seed": all_results[vn],
                "summary": summary_means[vn],
            }
            for vn, lam in LAMBDA_VARIANTS
        },
        "winners": [{"name": vn, "lambda": lam, "mct": mct, "starve": st}
                    for vn, lam, mct, st in winners],
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved → {json_path}")


if __name__ == "__main__":
    main()
