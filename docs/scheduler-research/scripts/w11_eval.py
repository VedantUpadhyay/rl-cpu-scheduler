"""W11 Evaluation Script — Week 11 checkpoint evaluation and fairness suite.

Loads W11 checkpoints (ep5000, ep10000), runs 100 evaluation episodes each,
reports MCT, composite reward, starvation, SRPT agreement, then runs full
fairness suite (JFI, SDV, VRFI) on the best checkpoint.
"""
from __future__ import annotations
import csv, math, os, random, sys
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PROJECT_ROOT = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                 "GRAD - FALL 23/UCSC/Capstone")
_SCRIPTS_DIR  = os.path.join(_PROJECT_ROOT, "docs", "scheduler-research", "scripts")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, "/tmp")

from schedsim.env    import SchedEnv, N_PROCESSES, N_QUANTUM_TIERS
from schedsim.process import Process
from w9_train import (
    TraceEpisodeSampler5,
    _encode_state, _valid_actions, _make_procs,
    D_CAND, N_QT, N_ACTIONS, QT_VALUES,
    _norm_cpu, _norm_mem,
    BURST_P95_FILT, WAIT_NORM, CPU_MAX, MEM_P95,
    REWARD_SCALE,
)
from w10c_train import AttentionDQN10C, _make_random_tasks

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRACE_PATH    = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
RESULTS_DIR   = os.path.join(_PROJECT_ROOT, "results")
CKPT_EP5000   = os.path.join(RESULTS_DIR, "w11d_ep5000.npz")
CKPT_EP10000  = os.path.join(RESULTS_DIR, "w11d_ep10000.npz")
N_EVAL        = 100

# Reference baselines (hardcoded)
BASELINE_MCT    = 18.21   # W10C original, seconds
BASELINE_STARVE = 0.0     # W10C original, %
BASELINE_SRPT   = 71.9    # W10C original, %
# W11 / W11b reference (hardcoded from prior runs)
W11_MCT    = 57.83
W11_STARVE = 26.0
W11_SRPT   = 38.5
W11B_MCT    = 62.74
W11B_STARVE = 13.0
W11B_SRPT   = 36.2
W11C_MCT    = 71.64
W11C_STARVE = 35.0
W11C_SRPT   = 25.5

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
    agent: AttentionDQN10C,
    rng:   np.random.Generator,
    sampler,
) -> dict:
    """Run one evaluation episode (epsilon=0). Returns metrics dict."""
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
        srpt_pid = min(runnable_procs, key=lambda p: p.remaining_burst).pid if runnable_procs else -1

        action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
        _, _, done, info = env.step(action)
        reward  = info.get("env_reward", 0.0) / REWARD_SCALE
        sv_next = _encode_state(env, tasks)

        if srpt_pid >= 0:
            chosen_pid = action // N_QT
            srpt_agree_n += int(chosen_pid == srpt_pid)
            srpt_total_n += 1

        ep_reward_sum += reward
        ep_step_n     += 1
        sv = sv_next

    mct      = info.get("mean_completion_time_so_far") or 0.0
    starved  = _starvation_flag(env)
    srpt_f   = srpt_agree_n / srpt_total_n if srpt_total_n > 0 else 0.0
    reward_m = ep_reward_sum / ep_step_n if ep_step_n > 0 else 0.0

    # Collect per-process turnaround times and value-curve info
    turnarounds  = []
    vrfi_vlrs    = []
    mean_tau     = 0.0
    for p in env.processes:
        if p.is_complete:
            T = p.completion_time - p.arrival_time
            turnarounds.append(T)
            # VLR: (base_value - V(delay)) / max(delay, 1)
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
        # tau floor: 0.2 = steep, 0.0 = smooth; track mean floor for classification
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

    agent = AttentionDQN10C()
    agent.load(path)
    agent.epsilon = 0.0

    rng = np.random.default_rng(0)

    mcts         = []
    rewards      = []
    starved_list = []
    srpt_list    = []
    all_turns    = []     # all turnaround times across episodes
    all_vlrs     = []     # all VLR values
    steep_turns  = []     # turnarounds for steep-curve episodes (floor~0.2)
    smooth_turns = []     # turnarounds for smooth-curve episodes (floor~0.0)

    for i in range(N_EVAL):
        ep = _run_episode(agent, rng, sampler)
        mcts.append(ep["mct"])
        rewards.append(ep["reward"])
        starved_list.append(ep["starved"])
        srpt_list.append(ep["srpt_frac"])
        all_turns.extend(ep["turnarounds"])
        all_vlrs.extend(ep["vrfi_vlrs"])

        # Classify episode: floor > 0.1 → steep, else smooth
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
    """Print full fairness suite for one checkpoint."""
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
    """Print comparison table against W10C baseline."""
    print(f"\n{'='*60}")
    print("Comparison Table")
    print(f"{'='*60}")
    header = f"{'Agent':<20} {'MCT(s)':>10} {'Starve%':>9} {'SRPT%':>8} {'Reward':>10}"
    print(header)
    print("-" * 60)
    # Baseline rows
    print(f"{'W10C baseline':<20} {BASELINE_MCT:>10.2f} {BASELINE_STARVE:>9.1f} "
          f"{BASELINE_SRPT:>8.1f} {'N/A':>10}")
    print(f"{'W11 (equal wts)':<20} {W11_MCT:>10.2f} {W11_STARVE:>9.1f} "
          f"{W11_SRPT:>8.1f} {'N/A':>10}")
    print(f"{'W11b (CT bonus)':<20} {W11B_MCT:>10.2f} {W11B_STARVE:>9.1f} "
          f"{W11B_SRPT:>8.1f} {'N/A':>10}")
    print(f"{'W11c (noisy burst)':<20} {W11C_MCT:>10.2f} {W11C_STARVE:>9.1f} "
          f"{W11C_SRPT:>8.1f} {'N/A':>10}")
    for r in results:
        if not r:
            continue
        print(f"{r['label']:<20} {r['mct_mean']:>10.4f} {r['starve_pct']:>9.1f} "
              f"{r['srpt_pct']:>8.2f} {r['rew_mean']:>10.6f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(0); np.random.seed(0)

    print("=" * 60)
    print("W11 Evaluation — checkpoint eval + fairness suite")
    print("=" * 60)

    # Load trace sampler if available
    use_trace = os.path.isfile(TRACE_PATH)
    if use_trace:
        print(f"\nLoading trace: {TRACE_PATH}")
        sampler = TraceEpisodeSampler5(TRACE_PATH)
    else:
        sampler = None
        print(f"\nTrace not found at {TRACE_PATH} — using random process generation.")

    # Evaluate checkpoints
    res5k  = evaluate_checkpoint("W11d ep5000",  CKPT_EP5000,  sampler)
    res10k = evaluate_checkpoint("W11d ep10000", CKPT_EP10000, sampler)

    results = [r for r in [res5k, res10k] if r]

    # Comparison table
    print_comparison_table(results)

    # Best checkpoint = lowest MCT (best scheduling efficiency)
    if results:
        best = min(results, key=lambda r: r["mct_mean"])
        print(f"Best checkpoint (lowest MCT): {best['label']}")
        print_fairness_suite(best)

        # Also print fairness for the other checkpoint if both exist
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
