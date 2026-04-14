"""W12 Evaluation Script — checkpoint eval + full fairness suite.

Loads W12 checkpoints (ep5000, ep10000), runs 100 evaluation episodes each,
reports MCT, composite reward, starvation, SRPT agreement, then runs full
fairness suite (JFI, SDV, VRFI) on the best checkpoint.

Comparison table: W10C | W11b | W12-ep5k | W12-ep10k
"""
from __future__ import annotations
import csv, math, os, random, sys
import numpy as np

_PROJECT_ROOT = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                 "GRAD - FALL 23/UCSC/Capstone")
_SCRIPTS_DIR  = os.path.join(_PROJECT_ROOT, "docs", "scheduler-research", "scripts")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, "/tmp")

from schedsim.env    import SchedEnv, N_PROCESSES, value_delta
from schedsim.process import Process
from w9_train import (
    TraceEpisodeSampler5,
    _valid_actions, _make_procs,
    N_QT, BURST_P95_FILT,
)
from w12_train import (
    AttentionDQN12,
    _encode_state, _make_random_tasks,
    PBRS_LAMBDA, REWARD_SCALE,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRACE_PATH   = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
RESULTS_DIR  = os.path.join(_PROJECT_ROOT, "results")
CKPT_EP5000  = os.path.join(RESULTS_DIR, "w12_ep5000.npz")
CKPT_EP10000 = os.path.join(RESULTS_DIR, "w12_ep10000.npz")
N_EVAL       = 100

# Reference baselines (hardcoded from prior runs)
BASELINE_MCT    = 18.21   # W10C
BASELINE_STARVE = 0.0
BASELINE_SRPT   = 71.9
BASELINE_VRFI   = 0.712

W11B_MCT    = 62.74
W11B_STARVE = 13.0
W11B_SRPT   = 36.2
W11B_VRFI   = 0.685


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _starvation_flag(env: SchedEnv) -> int:
    """1 if any completed process has slowdown > 3x group median, else 0."""
    completed = [p for p in env.processes if p.is_complete]
    if not completed:
        return 0
    turnarounds = [p.completion_time - p.arrival_time for p in completed]
    bursts_ep   = [p.burst_length for p in completed]
    slowdowns   = [t / max(b, 1e-6) for t, b in zip(turnarounds, bursts_ep)]
    med_slow    = float(np.median(slowdowns))
    return int(any(s > 3.0 * med_slow for s in slowdowns))


def _run_episode(
    agent:   AttentionDQN12,
    rng:     np.random.Generator,
    sampler,
) -> dict:
    """Run one eval episode (epsilon=0). Returns metrics dict."""
    if sampler is not None:
        tasks = sampler.sample_episode(rng)
    else:
        tasks = _make_random_tasks(rng)

    procs = _make_procs(tasks)
    env   = SchedEnv(procs)
    env.reset()
    sv    = _encode_state(env, tasks)

    ep_reward_sum = 0.0
    ep_step_n     = 0
    srpt_agree_n  = 0
    srpt_total_n  = 0
    done          = False

    while not done:
        valid = _valid_actions(env)

        runnable_procs = [
            p for p in env.processes
            if p.arrival_time <= env.current_time and not p.is_complete
        ]
        srpt_pid = (min(runnable_procs, key=lambda p: p.remaining_burst).pid
                    if runnable_procs else -1)

        # Pre-step snapshot for reward
        phi_s = (-PBRS_LAMBDA * max(p.time_since_last_execution for p in runnable_procs)
                 if runnable_procs else 0.0)
        runnable_snapshot = [
            (p.wait_time, p.tau, p.floor, p.base_value) for p in runnable_procs
        ]

        action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
        chosen_proc = env.processes[action // N_QT]
        rb_before   = chosen_proc.remaining_burst

        _, _, done, info = env.step(action)

        q_actual = rb_before - chosen_proc.remaining_burst
        r_base = sum(
            value_delta(tau, floor, base_val, wait_before, q_actual)
            for (wait_before, tau, floor, base_val) in runnable_snapshot
        ) / REWARD_SCALE

        runnable_after = [
            p for p in env.processes
            if p.arrival_time <= env.current_time and not p.is_complete
        ]
        phi_s_next = (-PBRS_LAMBDA * max(p.time_since_last_execution for p in runnable_after)
                      if runnable_after else 0.0)
        reward = r_base + (phi_s_next - phi_s)

        sv_next = _encode_state(env, tasks)

        if srpt_pid >= 0:
            chosen_pid = action // N_QT
            srpt_agree_n += int(chosen_pid == srpt_pid)
            srpt_total_n += 1

        ep_reward_sum += reward
        ep_step_n     += 1
        sv = sv_next

    mct     = info.get("mean_completion_time_so_far") or 0.0
    starved = _starvation_flag(env)
    srpt_f  = srpt_agree_n / srpt_total_n if srpt_total_n > 0 else 0.0
    reward_m = ep_reward_sum / ep_step_n if ep_step_n > 0 else 0.0

    # Per-process metrics for fairness suite
    turnarounds = []
    vrfi_vlrs   = []
    mean_tau    = 0.0
    for p in env.processes:
        if p.is_complete:
            T = p.completion_time - p.arrival_time
            turnarounds.append(T)
            delay = p.wait_time
            if p.tau > 0.0:
                v_now = p.base_value * max(p.floor, math.exp(-delay / p.tau))
            else:
                v_now = p.base_value
            vlr = (p.base_value - v_now) / max(delay, 1.0)
            vrfi_vlrs.append(vlr)
        mean_tau += p.tau
    mean_tau /= N_PROCESSES

    return {
        "mct":          mct,
        "reward":       reward_m,
        "starved":      starved,
        "srpt_frac":    srpt_f,
        "turnarounds":  turnarounds,
        "vrfi_vlrs":    vrfi_vlrs,
        "mean_tau":     mean_tau,
        "mean_floor":   sum(p.floor for p in env.processes) / N_PROCESSES,
    }


def evaluate_checkpoint(label: str, path: str, sampler) -> dict:
    """Load checkpoint, run N_EVAL episodes, return aggregated metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"  Path: {path}")
    if not os.path.isfile(path):
        print(f"  [MISSING] checkpoint not found — skipping.")
        return {}

    agent = AttentionDQN12()
    agent.load(path)
    agent.epsilon = 0.0

    rng = np.random.default_rng(0)

    mcts         = []
    rewards      = []
    starved_list = []
    srpt_list    = []
    all_turns    = []
    all_vlrs     = []
    steep_turns  = []
    smooth_turns = []

    for i in range(N_EVAL):
        ep = _run_episode(agent, rng, sampler)
        mcts.append(ep["mct"])
        rewards.append(ep["reward"])
        starved_list.append(ep["starved"])
        srpt_list.append(ep["srpt_frac"])
        all_turns.extend(ep["turnarounds"])
        all_vlrs.extend(ep["vrfi_vlrs"])

        if ep["mean_floor"] > 0.1:
            steep_turns.extend(ep["turnarounds"])
        else:
            smooth_turns.extend(ep["turnarounds"])

    def jfi(arr):
        if not arr:
            return float("nan")
        a = np.array(arr, dtype=np.float64)
        return float(a.sum() ** 2 / (len(a) * np.sum(a ** 2) + 1e-12))

    def sdv(arr):
        if not arr or len(arr) < 2:
            return float("nan")
        a = np.array(arr, dtype=np.float64)
        return float(np.std(a) / (np.mean(a) + 1e-12))

    def vrfi(vlrs):
        if not vlrs or len(vlrs) < 2:
            return float("nan")
        a = np.array(vlrs, dtype=np.float64)
        cv = float(np.std(a) / (np.mean(a) + 1e-12))
        return 1.0 - cv

    mct_mean = float(np.mean(mcts))
    mct_std  = float(np.std(mcts))
    rew_mean = float(np.mean(rewards))
    starve_n = int(np.sum(starved_list))
    starve_p = starve_n / N_EVAL * 100.0
    srpt_pct = float(np.mean(srpt_list)) * 100.0

    print(f"\n  Results ({N_EVAL} eval episodes, epsilon=0):")
    print(f"    MCT          : {mct_mean:.4f} ± {mct_std:.4f} s")
    print(f"    Reward mean  : {rew_mean:.6f}")
    print(f"    Starvation   : {starve_n}/{N_EVAL} ({starve_p:.1f}%)")
    print(f"    SRPT agree   : {srpt_pct:.2f}%")

    return {
        "label":        label,
        "mct_mean":     mct_mean,
        "mct_std":      mct_std,
        "rew_mean":     rew_mean,
        "starve_n":     starve_n,
        "starve_pct":   starve_p,
        "srpt_pct":     srpt_pct,
        "all_turns":    all_turns,
        "all_vlrs":     all_vlrs,
        "steep_turns":  steep_turns,
        "smooth_turns": smooth_turns,
        "jfi_all":      jfi(all_turns),
        "sdv_all":      sdv(all_turns),
        "jfi_steep":    jfi(steep_turns),
        "sdv_steep":    sdv(steep_turns),
        "jfi_smooth":   jfi(smooth_turns),
        "sdv_smooth":   sdv(smooth_turns),
        "vrfi":         vrfi(all_vlrs),
    }


def print_fairness_suite(res: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Full Fairness Suite — {res['label']}")
    print(f"{'='*60}")
    print(f"\n  Jain's Fairness Index (JFI) on turnaround times:")
    print(f"    All episodes   : {res['jfi_all']:.6f}")
    print(f"    Steep-tau eps  : {res['jfi_steep']:.6f}  (floor=0.2, tau~U(600,1200))")
    print(f"    Smooth-tau eps : {res['jfi_smooth']:.6f}  (floor=0.0, tau~U(600,1000))")
    print(f"\n  Coefficient of Variation (SDV = std/mean) on turnaround times:")
    print(f"    All episodes   : {res['sdv_all']:.6f}")
    print(f"    Steep-tau eps  : {res['sdv_steep']:.6f}")
    print(f"    Smooth-tau eps : {res['sdv_smooth']:.6f}")
    print(f"\n  Value-Rate Fairness Index (VRFI = 1 - CV(VLR)):")
    print(f"    All episodes   : {res['vrfi']:.6f}")
    print(f"\n  Starvation count (slowdown > 3x median): {res['starve_n']}/{N_EVAL}")


def print_comparison_table(results: list[dict]) -> None:
    print(f"\n{'='*60}")
    print("Comparison Table")
    print(f"{'='*60}")
    header = (f"{'Agent':<22} {'MCT(s)':>10} {'Starve%':>9} "
              f"{'SRPT%':>8} {'VRFI':>8} {'Reward':>10}")
    print(header)
    print("-" * 70)
    print(f"{'W10C baseline':<22} {BASELINE_MCT:>10.2f} {BASELINE_STARVE:>9.1f} "
          f"{BASELINE_SRPT:>8.1f} {BASELINE_VRFI:>8.3f} {'N/A':>10}")
    print(f"{'W11b (ct bonus)':<22} {W11B_MCT:>10.2f} {W11B_STARVE:>9.1f} "
          f"{W11B_SRPT:>8.1f} {W11B_VRFI:>8.3f} {'N/A':>10}")
    for r in results:
        if not r:
            continue
        print(f"{r['label']:<22} {r['mct_mean']:>10.4f} {r['starve_pct']:>9.1f} "
              f"{r['srpt_pct']:>8.2f} {r['vrfi']:>8.3f} {r['rew_mean']:>10.6f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(0); np.random.seed(0)

    print("=" * 60)
    print("W12 Evaluation — checkpoint eval + fairness suite")
    print("=" * 60)

    use_trace = os.path.isfile(TRACE_PATH)
    if use_trace:
        print(f"\nLoading trace: {TRACE_PATH}")
        sampler = TraceEpisodeSampler5(TRACE_PATH)
    else:
        sampler = None
        print(f"\nTrace not found at {TRACE_PATH} — using random process generation.")

    res5k  = evaluate_checkpoint("W12 ep5000",  CKPT_EP5000,  sampler)
    res10k = evaluate_checkpoint("W12 ep10000", CKPT_EP10000, sampler)

    results = [r for r in [res5k, res10k] if r]

    print_comparison_table(results)

    if results:
        best = min(results, key=lambda r: r["mct_mean"])
        print(f"Best checkpoint (lowest MCT): {best['label']}")
        print_fairness_suite(best)

        others = [r for r in results if r["label"] != best["label"]]
        for other in others:
            print(f"\n  Fairness suite also computed for {other['label']}:")
            print(f"    JFI(all)={other['jfi_all']:.6f}  SDV(all)={other['sdv_all']:.6f}"
                  f"  VRFI={other['vrfi']:.6f}")
    else:
        print("\nNo checkpoint results available — please run training first.")

    print(f"\n{'='*60}")
    print("Evaluation complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
