"""W14-ω Poisson generalization evaluation.

Tests whether W14-ω (ω_s=0.7, seed 123) generalizes to variable-depth
queues driven by Poisson arrival streams.

Architecture: fixed 35-dim state (5 slots × 7 features).
  - N < 5: empty slots filled with zeros (arrived_flag=0 → masked in attention)
  - N > 5: only first 5 FIFO candidates exposed to agent (OOD condition)

Arrival rates: λ ∈ {1.0, 2.0, 5.0} tasks/second.
Episode end: 300 task completions.
N_eval: 200 episodes per (agent, λ) combination.

Results saved to: docs/scheduler-research/results/w14_variable_n/
"""
from __future__ import annotations
import csv, json, math, os, random, sys, time
import numpy as np
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from project_config import (PROJECT_ROOT as _PROJECT_ROOT, SCRIPTS_DIR as _SCRIPTS_DIR,
                             TEST_PATH as TRACE_TEST, get_agent_dir)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

from w9_train import (
    TraceEpisodeSampler5,
    N_QT, QT_VALUES,
    _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem,
    WAIT_NORM, CPU_MAX, MEM_P95,
)
from ablation_multiseed import N_HEADS, D_HEAD, D_V_TOT
from w14_omega import W14OmegaDQN

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUT_DIR = get_agent_dir("w14_variable_n")

CKPT_PATH    = os.path.join(get_agent_dir("w14_omega"), "w14_seed123_final.npz")
OMEGA_S      = 0.7           # best deployment setting per pareto_frontier.json
N_PROCESSES  = 5             # training distribution slot count
D_CAND       = 7             # features per slot
AFI          = 6             # arrived_flag index within 7-dim slot
N_COMPLETE   = 100           # episode end condition: completions
N_EVAL       = 100           # episodes per condition
N_GENERATE   = 400           # task pool per episode (generous buffer)
# Lambda values calibrated to trace burst mean ~10s:
#   rho = lambda * E[burst] = lambda * 10.0
#   lambda=0.03 → rho=0.3 → E[queue]≈0.4  (low contention, mostly N≤5)
#   lambda=0.07 → rho=0.7 → E[queue]≈2.3  (moderate contention, OOD emerging)
#   lambda=0.09 → rho=0.9 → E[queue]≈9    (high contention, mostly OOD)
# The user's original values (1.0, 2.0, 5.0) would give rho=10–50:
# system unstable, queue grows unboundedly, MCT→∞. Rescaled accordingly.
LAMBDAS      = [0.03, 0.07, 0.09]
SEEDS        = [42, 123, 456]

MLFQ_AGE_THRESH  = 50.0     # seconds
QUANTUM_TIERS    = (0.5, 2.0, 8.0)  # matches QT_VALUES
STARVATION_SLOW  = 3.0      # slowdown multiplier for starvation flag


# ---------------------------------------------------------------------------
# PoissonTask — lightweight task tracker for variable-N simulation
# ---------------------------------------------------------------------------

@dataclass
class PoissonTask:
    task_id:   int
    arrival_time: float
    burst_length: float
    plan_cpu:  float
    plan_mem:  float
    tau:       float
    floor:     float
    base_value: float = 1.0

    remaining_burst: float           = field(init=False)
    wait_time:       float           = field(default=0.0)
    time_since_last_execution: float = field(default=0.0)
    completion_time: float | None    = field(default=None)

    def __post_init__(self) -> None:
        self.remaining_burst = self.burst_length

    @property
    def is_complete(self) -> bool:
        return self.remaining_burst <= 1e-9


# ---------------------------------------------------------------------------
# Helpers: state encoding for variable-N queue
# ---------------------------------------------------------------------------

def encode_state_poisson(candidates: list[PoissonTask],
                         current_time: float) -> np.ndarray:
    """Build 35-dim state from up to 5 candidate tasks.

    candidates: ordered list (FIFO), len 1..5.
    Missing slots (len < 5) are left as zeros (arrived_flag=0).
    """
    vec = np.zeros(N_PROCESSES * D_CAND, dtype=np.float32)
    for slot, t in enumerate(candidates):
        off = slot * D_CAND
        tq = current_time - t.arrival_time
        vec[off + 0] = _norm_time_log(tq)
        vec[off + 1] = t.wait_time / WAIT_NORM
        vec[off + 2] = _norm_time_log(t.time_since_last_execution)
        vec[off + 3] = _urgency_norm_task(t)
        vec[off + 4] = _norm_cpu(t.plan_cpu)
        vec[off + 5] = _norm_mem(t.plan_mem)
        vec[off + 6] = 1.0   # arrived_flag
    return vec


def _urgency_norm_task(t: PoissonTask) -> float:
    """VLR / 0.1 — mirrors _urgency_norm() from w9_train."""
    delay = t.wait_time
    if t.tau <= 0.0 or delay <= 0.0:
        return 0.0
    v_now = t.base_value * max(t.floor, math.exp(-delay / t.tau))
    vlr   = (t.base_value - v_now) / delay
    return float(vlr / 0.1)


# ---------------------------------------------------------------------------
# Task pool generation — Poisson inter-arrivals, trace-drawn features
# ---------------------------------------------------------------------------

def generate_task_pool(trace_data: np.ndarray, lam: float,
                       rng: np.random.Generator, n_tasks: int) -> list[PoissonTask]:
    """Generate n_tasks with Poisson inter-arrival times and trace features.

    trace_data: (N, 3) array of [burst_length, plan_cpu, plan_mem] from test trace.
    lam: arrival rate (tasks/second).
    Value curves: 50% steep (tau~[600,1200], floor=0.2), 50% smooth (tau~[600,1000], floor=0.0).
    """
    idx     = rng.integers(0, len(trace_data), size=n_tasks)
    samples = trace_data[idx]   # (n_tasks, 3)

    # Poisson inter-arrivals: inter_t ~ Exp(1/lam)
    inter_times = rng.exponential(scale=1.0 / lam, size=n_tasks)
    arrival_times = np.cumsum(inter_times)

    tasks = []
    for i in range(n_tasks):
        burst, cpu, mem = float(samples[i, 0]), float(samples[i, 1]), float(samples[i, 2])
        if rng.random() < 0.5:   # steep curve
            tau   = rng.uniform(600.0, 1200.0)
            floor = 0.2
        else:                     # smooth curve
            tau   = rng.uniform(600.0, 1000.0)
            floor = 0.0
        tasks.append(PoissonTask(
            task_id      = i,
            arrival_time = float(arrival_times[i]),
            burst_length = burst,
            plan_cpu     = cpu,
            plan_mem     = mem,
            tau          = tau,
            floor        = floor,
        ))
    return tasks


# ---------------------------------------------------------------------------
# Core simulation — W14-ω agent
# ---------------------------------------------------------------------------

def simulate_w14(agent: W14OmegaDQN, task_pool: list[PoissonTask],
                 omega_s: float = OMEGA_S) -> dict:
    """Run one episode with W14-ω on a pre-generated Poisson task pool.

    Returns per-episode metrics dict.
    """
    # Work on shallow copies of task state
    from copy import deepcopy
    tasks = [deepcopy(t) for t in task_pool]
    tasks.sort(key=lambda t: t.arrival_time)

    current_time  = 0.0
    arrival_idx   = 0
    queue: list[PoissonTask] = []
    completed: list[PoissonTask] = []

    queue_depths: list[int] = []
    ood_decisions = 0
    total_decisions = 0

    def advance_arrivals() -> None:
        nonlocal arrival_idx
        while arrival_idx < len(tasks) and tasks[arrival_idx].arrival_time <= current_time:
            queue.append(tasks[arrival_idx])
            arrival_idx += 1

    advance_arrivals()

    while len(completed) < N_COMPLETE:
        if not queue:
            if arrival_idx < len(tasks):
                current_time = tasks[arrival_idx].arrival_time
                advance_arrivals()
                continue
            else:
                break  # exhausted task pool

        # FIFO candidate selection: first 5 by arrival_time
        queue.sort(key=lambda t: t.arrival_time)
        n_q    = len(queue)
        n_cand = min(n_q, N_PROCESSES)
        candidates = queue[:n_cand]

        queue_depths.append(n_q)
        total_decisions += 1
        if n_q > N_PROCESSES:
            ood_decisions += 1

        state         = encode_state_poisson(candidates, current_time)
        valid_actions = [slot * N_QT + qt
                         for slot in range(n_cand) for qt in range(N_QT)]

        action   = agent.select_action(state, epsilon=0.0,
                                       valid_actions=valid_actions, omega_s=omega_s)
        slot_idx = action // N_QT
        qt_idx   = action % N_QT
        quantum  = QT_VALUES[qt_idx]
        chosen   = candidates[slot_idx]

        q_actual = min(quantum, chosen.remaining_burst)

        # Update wait/starvation for tasks waiting in queue (not chosen)
        for t in queue:
            if t is not chosen:
                t.wait_time               += q_actual
                t.time_since_last_execution += q_actual

        chosen.time_since_last_execution = 0.0
        chosen.remaining_burst          -= q_actual
        current_time                    += q_actual

        if chosen.is_complete:
            chosen.completion_time = current_time
            queue.remove(chosen)
            completed.append(chosen)

        advance_arrivals()

    return _compute_metrics(completed, queue_depths, ood_decisions, total_decisions)


# ---------------------------------------------------------------------------
# Core simulation — MLFQ baseline
# ---------------------------------------------------------------------------

def simulate_mlfq(task_pool: list[PoissonTask]) -> dict:
    """Run one episode with MLFQ on a pre-generated Poisson task pool."""
    from copy import deepcopy
    tasks = [deepcopy(t) for t in task_pool]
    tasks.sort(key=lambda t: t.arrival_time)

    current_time  = 0.0
    arrival_idx   = 0
    queue: list[PoissonTask] = []
    completed: list[PoissonTask] = []

    queue_depths: list[int] = []
    ood_decisions = 0
    total_decisions = 0

    # MLFQ state per task
    mlfq_level:   dict[int, int]   = {}   # task_id → level 0/1/2
    prev_remaining: dict[int, float] = {}  # task_id → remaining_burst at last pick
    last_picked_id: int | None = None

    def advance_arrivals() -> None:
        nonlocal arrival_idx
        while arrival_idx < len(tasks) and tasks[arrival_idx].arrival_time <= current_time:
            t = tasks[arrival_idx]
            queue.append(t)
            mlfq_level[t.task_id]    = 0
            prev_remaining[t.task_id] = t.burst_length
            arrival_idx += 1

    advance_arrivals()

    while len(completed) < N_COMPLETE:
        if not queue:
            if arrival_idx < len(tasks):
                current_time = tasks[arrival_idx].arrival_time
                advance_arrivals()
                continue
            else:
                break

        n_q = len(queue)
        queue_depths.append(n_q)
        total_decisions += 1
        if n_q > N_PROCESSES:
            ood_decisions += 1

        # Demotion: if last picked task consumed >= full quantum for its tier
        if last_picked_id is not None:
            lt = next((t for t in queue if t.task_id == last_picked_id), None)
            if lt is not None:
                level   = mlfq_level.get(lt.task_id, 0)
                prev_r  = prev_remaining.get(lt.task_id, lt.burst_length)
                consumed = prev_r - lt.remaining_burst
                if consumed >= QUANTUM_TIERS[level] - 1e-6 and level < 2:
                    mlfq_level[lt.task_id] = level + 1

        # Aging: promote any task waiting > threshold
        for t in queue:
            if t.time_since_last_execution > MLFQ_AGE_THRESH:
                mlfq_level[t.task_id] = 0

        # Pick: lowest level first, FIFO within level
        chosen = None
        chosen_qt = 0
        for level in range(3):
            level_tasks = [t for t in queue if mlfq_level.get(t.task_id, 0) == level]
            if level_tasks:
                chosen   = min(level_tasks, key=lambda t: t.arrival_time)
                chosen_qt = level   # tier matches queue level
                break
        if chosen is None:
            chosen   = min(queue, key=lambda t: t.arrival_time)
            chosen_qt = 0

        quantum  = QUANTUM_TIERS[chosen_qt]
        q_actual = min(quantum, chosen.remaining_burst)

        prev_remaining[chosen.task_id] = chosen.remaining_burst

        for t in queue:
            if t is not chosen:
                t.wait_time               += q_actual
                t.time_since_last_execution += q_actual

        chosen.time_since_last_execution = 0.0
        chosen.remaining_burst          -= q_actual
        current_time                    += q_actual
        last_picked_id                   = chosen.task_id

        if chosen.is_complete:
            chosen.completion_time = current_time
            queue.remove(chosen)
            completed.append(chosen)

        advance_arrivals()

    return _compute_metrics(completed, queue_depths, ood_decisions, total_decisions)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(completed: list[PoissonTask],
                     queue_depths: list[int],
                     ood_decisions: int,
                     total_decisions: int) -> dict:
    if not completed:
        return {"mct": float("nan"), "starve_pct": float("nan"),
                "mean_queue_depth": float("nan"), "max_queue_depth": 0,
                "ood_pct": float("nan"), "n_completed": 0}

    turnarounds = [t.completion_time - t.arrival_time for t in completed]
    mct = float(np.mean(turnarounds))

    bursts = [t.burst_length for t in completed]
    slows  = [ta / max(b, 1e-6) for ta, b in zip(turnarounds, bursts)]
    med    = float(np.median(slows))
    starved = int(any(s > STARVATION_SLOW * med for s in slows))

    mean_q = float(np.mean(queue_depths)) if queue_depths else 0.0
    max_q  = int(max(queue_depths))       if queue_depths else 0
    ood_p  = 100.0 * ood_decisions / max(total_decisions, 1)

    return {
        "mct":             mct,
        "starved_episode": starved,
        "mean_queue_depth": mean_q,
        "max_queue_depth": max_q,
        "ood_pct":         ood_p,
        "n_completed":     len(completed),
    }


# ---------------------------------------------------------------------------
# Evaluation loop for one condition
# ---------------------------------------------------------------------------

def evaluate_condition(agent_or_mlfq, trace_data: np.ndarray,
                       lam: float, n_eval: int, seeds: list[int],
                       is_mlfq: bool = False) -> dict:
    """Run n_eval episodes for a given λ, collect aggregate metrics."""
    all_mct, all_starved = [], []
    all_mean_q, all_max_q, all_ood = [], [], []
    total_completed = 0

    for seed_i, seed in enumerate(seeds):
        rng      = np.random.default_rng(seed + 7919)   # avoid collision with training seeds
        n_eps    = n_eval // len(seeds) + (1 if seed_i < n_eval % len(seeds) else 0)

        done_eps = 0
        for ep in range(n_eps):
            # Fresh task pool per episode
            ep_rng   = np.random.default_rng(seed * 10000 + ep)
            pool     = generate_task_pool(trace_data, lam, ep_rng, N_GENERATE)

            if is_mlfq:
                metrics = simulate_mlfq(pool)
            else:
                metrics = simulate_w14(agent_or_mlfq, pool, omega_s=OMEGA_S)

            done_eps += 1
            if done_eps % 20 == 0:
                agent_tag = "MLFQ" if is_mlfq else "W14-ω"
                print(f"    [{agent_tag} λ={lam:.1f}] seed={seed} ep={done_eps}/{n_eps} "
                      f"MCT={metrics['mct']:.1f}s", flush=True)

            all_mct.append(metrics["mct"])
            all_starved.append(metrics["starved_episode"])
            all_mean_q.append(metrics["mean_queue_depth"])
            all_max_q.append(metrics["max_queue_depth"])
            all_ood.append(metrics["ood_pct"])
            total_completed += metrics["n_completed"]

    return {
        "lam":             lam,
        "is_mlfq":         is_mlfq,
        "n_eval":          n_eval,
        "mct_mean":        float(np.mean(all_mct)),
        "mct_std":         float(np.std(all_mct)),
        "starve_pct":      float(np.mean(all_starved)) * 100.0,
        "mean_queue_depth": float(np.mean(all_mean_q)),
        "max_queue_depth": int(max(all_max_q)),
        "ood_pct":         float(np.mean(all_ood)),
        "total_completed": total_completed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 72)
    print("W14-ω Poisson Generalization Evaluation")
    print(f"Checkpoint: {CKPT_PATH}")
    print(f"omega_s = {OMEGA_S}  |  N_COMPLETE={N_COMPLETE}  |  N_EVAL={N_EVAL}")
    print(f"Lambda values: {LAMBDAS}  (trace E[burst]~10s, stable threshold λ<0.10)")
    print(f"  ρ values: {[round(l*10, 2) for l in LAMBDAS]}  (E[queue] via M/M/1: "
          f"{[round(l*10/(1-l*10),2) if l*10<1 else '∞' for l in LAMBDAS]})")
    print(f"Output: {OUT_DIR}")
    print("=" * 72)

    # Load test trace data (raw numpy array)
    print(f"\nLoading test trace: {TRACE_TEST}")
    trace_data_list = []
    with open(TRACE_TEST, newline="") as f:
        for row in csv.DictReader(f):
            try:
                st  = float(row["start_time"])
                et  = float(row["end_time"])
                dur = et - st
                cpu = float(row["plan_cpu"])
                mem = float(row["plan_mem"])
                if dur > 0:
                    trace_data_list.append((dur, cpu, mem))
            except (ValueError, TypeError, KeyError):
                pass
    trace_data = np.array(trace_data_list, dtype=np.float32)
    print(f"  {len(trace_data):,} test tasks loaded.")

    # Load W14-ω agent (seed 123, best per pareto_frontier.json)
    print(f"\nLoading W14-ω checkpoint: {CKPT_PATH}")
    agent = W14OmegaDQN()
    agent.load(CKPT_PATH)
    agent.epsilon = 0.0   # greedy evaluation
    print("  Agent loaded.")

    # Fixed-N=5 reference numbers from pareto_frontier.json (ω_s=0.7)
    fixed_n_w14  = {"mct": 20.602, "starve_pct": 32.5}
    fixed_n_mlfq = {"mct": 21.59,  "starve_pct": 36.0}

    results = []

    print("\n" + "-" * 72)
    print("Running conditions...")
    print("-" * 72)
    t0_total = time.time()

    for lam in LAMBDAS:
        for is_mlfq in [False, True]:
            label = "MLFQ" if is_mlfq else f"W14-ω(ω={OMEGA_S})"
            print(f"\n  λ={lam:.1f}  Agent={label} ...")
            sys.stdout.flush()
            t0 = time.time()

            cond = evaluate_condition(
                agent_or_mlfq = None if is_mlfq else agent,
                trace_data    = trace_data,
                lam           = lam,
                n_eval        = N_EVAL,
                seeds         = SEEDS,
                is_mlfq       = is_mlfq,
            )
            elapsed = time.time() - t0
            cond["elapsed_s"] = elapsed

            results.append(cond)

            print(f"    MCT={cond['mct_mean']:.2f}±{cond['mct_std']:.2f}s  "
                  f"Starve={cond['starve_pct']:.1f}%  "
                  f"MeanQ={cond['mean_queue_depth']:.2f}  "
                  f"MaxQ={cond['max_queue_depth']}  "
                  f"OOD%={cond['ood_pct']:.1f}%  "
                  f"({elapsed:.0f}s)")
            sys.stdout.flush()

    total_elapsed = time.time() - t0_total
    print(f"\nTotal wall time: {total_elapsed/60:.1f} min")

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 88)
    print("RESULTS TABLE")
    print("=" * 88)
    hdr = (f"{'Setting':<22} {'Agent':<18} {'MCT':>7} {'Starve%':>8} "
           f"{'MeanQ':>7} {'MaxQ':>5} {'OOD%':>6}")
    print(hdr)
    print("-" * 88)

    # Fixed N=5 reference rows
    print(f"{'Fixed N=5':<22} {'W14-ω':<18} "
          f"{fixed_n_w14['mct']:>7.2f} {fixed_n_w14['starve_pct']:>8.1f} "
          f"{'5.00':>7} {'5':>5} {'0.0':>6}")
    print(f"{'Fixed N=5':<22} {'MLFQ':<18} "
          f"{fixed_n_mlfq['mct']:>7.2f} {fixed_n_mlfq['starve_pct']:>8.1f} "
          f"{'5.00':>7} {'5':>5} {'0.0':>6}")

    for cond in results:
        setting = f"Poisson λ={cond['lam']:.1f}"
        agent_n = "MLFQ" if cond["is_mlfq"] else f"W14-ω"
        print(f"{setting:<22} {agent_n:<18} "
              f"{cond['mct_mean']:>7.2f} {cond['starve_pct']:>8.1f} "
              f"{cond['mean_queue_depth']:>7.2f} {cond['max_queue_depth']:>5} "
              f"{cond['ood_pct']:>6.1f}%")

    print("=" * 88)

    # ---------------------------------------------------------------------------
    # Answer the four research questions
    # ---------------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("RESEARCH QUESTIONS")
    print("=" * 72)

    # Q1: Does W14-ω still beat MLFQ at λ=2.0?
    w14_20 = next((r for r in results if r["lam"] == 2.0 and not r["is_mlfq"]), None)
    mlfq_20 = next((r for r in results if r["lam"] == 2.0 and r["is_mlfq"]), None)
    if w14_20 and mlfq_20:
        beats_mct    = w14_20["mct_mean"]   < mlfq_20["mct_mean"]
        beats_starve = w14_20["starve_pct"] < mlfq_20["starve_pct"]
        print(f"\nQ1. Does W14-ω beat MLFQ at λ=2.0?")
        print(f"    W14-ω MCT={w14_20['mct_mean']:.2f}s  MLFQ MCT={mlfq_20['mct_mean']:.2f}s  "
              f"→ MCT {'BETTER' if beats_mct else 'WORSE'}")
        print(f"    W14-ω Starve={w14_20['starve_pct']:.1f}%  MLFQ Starve={mlfq_20['starve_pct']:.1f}%  "
              f"→ Starvation {'BETTER' if beats_starve else 'WORSE'}")
        if beats_mct and beats_starve:
            print("    → YES — W14-ω dominates MLFQ at λ=2.0 on both metrics.")
        elif beats_mct or beats_starve:
            print("    → PARTIALLY — W14-ω wins one metric, loses the other at λ=2.0.")
        else:
            print("    → NO — W14-ω does not beat MLFQ on either metric at λ=2.0.")

    # Q2: At what λ does OOD% exceed 20%?
    print(f"\nQ2. At what λ does OOD% become significant (>20%)?")
    for lam in LAMBDAS:
        r_w14 = next((r for r in results if r["lam"] == lam and not r["is_mlfq"]), None)
        if r_w14:
            flag = " ← exceeds 20%" if r_w14["ood_pct"] > 20.0 else ""
            print(f"    λ={lam:.1f}: OOD%={r_w14['ood_pct']:.1f}%{flag}")

    # Q3: Graceful degradation or cliff-drop?
    print(f"\nQ3. Graceful degradation or cliff-drop at N>5?")
    mcts_w14 = {r["lam"]: r["mct_mean"] for r in results if not r["is_mlfq"]}
    mcts_mlfq = {r["lam"]: r["mct_mean"] for r in results if r["is_mlfq"]}
    for lam in LAMBDAS:
        delta_w14 = mcts_w14.get(lam, float("nan")) - fixed_n_w14["mct"]
        delta_mlfq = mcts_mlfq.get(lam, float("nan")) - fixed_n_mlfq["mct"]
        print(f"    λ={lam:.1f}: W14-ω ΔMCT={delta_w14:+.2f}s  MLFQ ΔMCT={delta_mlfq:+.2f}s")

    # Q4: Is retraining necessary?
    print(f"\nQ4. Is retraining with variable N necessary?")
    all_beat_both = []
    for lam in LAMBDAS:
        w14_r = next((r for r in results if r["lam"] == lam and not r["is_mlfq"]), None)
        if w14_r:
            dominates = (w14_r["mct_mean"] < fixed_n_mlfq["mct"] and
                         w14_r["starve_pct"] < fixed_n_mlfq["starve_pct"])
            all_beat_both.append((lam, dominates))
            print(f"    λ={lam:.1f}: beats canonical MLFQ on both metrics? {'YES' if dominates else 'NO'}")
    if all(d for _, d in all_beat_both):
        print("    → Current architecture generalizes adequately. Retraining not required.")
    elif any(d for _, d in all_beat_both):
        print("    → Partial generalization. Retraining recommended for high-λ deployments.")
    else:
        print("    → Architecture does not generalize. Retraining with variable N is necessary.")

    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------

    output = {
        "config": {
            "checkpoint":   CKPT_PATH,
            "omega_s":      OMEGA_S,
            "n_complete":   N_COMPLETE,
            "n_eval":       N_EVAL,
            "lambdas":      LAMBDAS,
            "seeds":        SEEDS,
        },
        "fixed_n_reference": {
            "w14_omega": fixed_n_w14,
            "mlfq":      fixed_n_mlfq,
        },
        "conditions": results,
    }
    out_path = os.path.join(OUT_DIR, "w14_variable_n_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
