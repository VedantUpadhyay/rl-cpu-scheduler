"""W15 — Variable-N W14-ω retrained with Poisson arrival environment.

Training changes vs W14-ω:
  1. Poisson arrival stream: λ ~ Uniform(0.01, 0.08) per episode, 300 completions.
  2. Variable window: N<=5 all tasks; N>5 shows 5 longest-waiting (most-starved visible).
  3. State normalization: wait_time / 200.0 (was 500.0 in W14).

Architecture: identical to W14-ω (35-dim state, 2-head attention DQN, FiLM omega).
Training: 40,000 episodes × 3 seeds.
All output goes to LOG_FILE. No stdout. Runs fully unattended.
"""
from __future__ import annotations
import argparse
import csv, json, math, os, random, sys, time
import numpy as np
import torch
from dataclasses import dataclass, field
from copy import deepcopy

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from project_config import (PROJECT_ROOT as _PROJECT_ROOT, SCRIPTS_DIR as _SCRIPTS_DIR,
                             TRACE_PATH as TRACE_TRAIN, TEST_PATH as TRACE_TEST,
                             get_agent_dir)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

from schedsim.env   import SchedEnv, value_delta

from w9_train import (
    TraceEpisodeSampler5, _make_procs,
    N_QT, N_ACTIONS, QT_VALUES,
    _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem,
    WAIT_NORM, CPU_MAX, MEM_P95,
)
from ablation_multiseed import (
    N_HEADS, D_HEAD, D_V_TOT,
    LAMBDA_START, LAMBDA_END, LOSS_GATE, ENT_GATE_5K,
    LR, GAMMA, GRAD_CLIP, BUF_CAPACITY, BATCH_SIZE,
    TARGET_UPDATE_FREQ, WARMUP,
)
TARGET_UPDATE_FREQ = 5   # override ablation's 200; ~1k gradient steps between syncs
ENT_GATE_5K = float("inf")  # disable: Poisson max-H=log(6)≈1.79, gate's 1.2 is too strict
from w14_omega import (
    OmegaReplayBuffer,
    STARVATION_THRESHOLD, REWARD_SCALE, QUANTUM_TIERS,
    MLFQ_AGE_THRESH, GRAD_RATIO_WARN,
    run_mlfq,
)
from w15_network_torch import W15Trainer, device

# Seed from CLI: --seed 42 | 123 | 456  (default 42)
_parser = argparse.ArgumentParser()
_parser.add_argument("--seed", type=int, default=42)
_args, _ = _parser.parse_known_args()
SEEDS = [_args.seed]

# ---------------------------------------------------------------------------
# Output — all to LOG_FILE, nothing to stdout
# ---------------------------------------------------------------------------
OUT_DIR  = get_agent_dir("w15_variable_n")
LOG_FILE = os.path.join(OUT_DIR, f"w15_seed{_args.seed}_train.log")
os.makedirs(OUT_DIR, exist_ok=True)

_log_fh = open(LOG_FILE, "w", buffering=1)

def log(*args, **kwargs) -> None:
    kwargs.pop("file", None)
    kwargs.pop("flush", None)
    print(*args, **kwargs, file=_log_fh, flush=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PROCESSES       = 5
D_CAND            = 7
AFI               = 6        # arrived_flag index

W15_WAIT_NORM     = 200.0    # was 500.0 in W14

N_EPISODES        = 20_000
UPDATE_FREQ       = 2        # update network every N transitions (reduces backward overhead)
LOG_EVERY         = 1_000
CKPT_EVERY        = 1_000    # checkpoint every 1000 eps (all seeds, parallel run)

N_COMPLETE_TRAIN  = 300      # completions per training episode
N_GENERATE_TRAIN  = 600      # task pool buffer (>= N_COMPLETE_TRAIN)
LAM_MIN           = 0.01
LAM_MAX           = 0.08

STARVATION_SLOW   = 3.0

# Evaluation
OMEGA_EVAL        = 0.7
N_EVAL_FIXED      = 200
N_EVAL_POISSON    = 200
N_COMPLETE_EVAL   = 100      # completions per eval episode
N_GENERATE_EVAL   = 400
EVAL_LAMBDAS      = [0.03, 0.05, 0.07, 0.09]   # ρ ≈ {0.3, 0.5, 0.7, 0.9}

# Prior W14-ω results for comparison table
W14_FIXED_N5  = {"mct": 20.60, "starve_pct": 32.5, "ood_pct": 0.0}
MLFQ_FIXED_N5 = {"mct": 21.59, "starve_pct": 36.0, "ood_pct": 0.0}
W14_POISSON   = {
    0.03: {"mct": 14.58, "starve_pct": 100.0, "ood_pct":  0.2},
    0.07: {"mct": 31.77, "starve_pct": 100.0, "ood_pct": 13.8},
    0.09: {"mct": 58.19, "starve_pct": 100.0, "ood_pct": 38.0},
}
MLFQ_POISSON  = {
    0.03: {"mct": 14.62, "starve_pct": 100.0, "ood_pct":  0.2},
    0.07: {"mct": 31.00, "starve_pct":  99.0, "ood_pct": 17.9},
    0.09: {"mct": 51.73, "starve_pct":  83.0, "ood_pct": 46.8},
}


# ---------------------------------------------------------------------------
# PoissonTask dataclass
# ---------------------------------------------------------------------------

@dataclass
class PoissonTask:
    task_id:      int
    arrival_time: float
    burst_length: float
    plan_cpu:     float
    plan_mem:     float
    tau:          float
    floor:        float
    base_value:   float = 1.0

    remaining_burst:           float        = field(init=False)
    wait_time:                 float        = field(default=0.0)
    time_since_last_execution: float        = field(default=0.0)
    completion_time:           float | None = field(default=None)

    def __post_init__(self) -> None:
        self.remaining_burst = self.burst_length

    @property
    def is_complete(self) -> bool:
        return self.remaining_burst <= 1e-9


# ---------------------------------------------------------------------------
# State encoding — W15 norms (wait_time / 200.0)
# ---------------------------------------------------------------------------

def _urgency_norm_task(t: PoissonTask) -> float:
    delay = t.wait_time
    if t.tau <= 0.0 or delay <= 0.0:
        return 0.0
    v_now = t.base_value * max(t.floor, math.exp(-delay / t.tau))
    vlr   = (t.base_value - v_now) / delay
    return float(vlr / 0.1)


def encode_state_w15(candidates: list[PoissonTask], current_time: float) -> np.ndarray:
    """35-dim state from up to 5 Poisson candidates using W15 norms."""
    vec = np.zeros(N_PROCESSES * D_CAND, dtype=np.float32)
    for slot, t in enumerate(candidates[:N_PROCESSES]):
        off = slot * D_CAND
        tq  = current_time - t.arrival_time
        vec[off + 0] = _norm_time_log(tq)
        vec[off + 1] = t.wait_time / W15_WAIT_NORM      # W15 change: 200.0
        vec[off + 2] = _norm_time_log(t.time_since_last_execution)
        vec[off + 3] = _urgency_norm_task(t)
        vec[off + 4] = _norm_cpu(t.plan_cpu)
        vec[off + 5] = _norm_mem(t.plan_mem)
        vec[off + 6] = 1.0
    return vec


def encode_state_w15_fixed(env, tasks) -> np.ndarray:
    """Fixed-N=5 state encoding for SchedEnv using W15 norms (200.0)."""
    vec = np.zeros(N_PROCESSES * D_CAND, dtype=np.float32)
    for pid in range(N_PROCESSES):
        p   = env.processes[pid]
        off = pid * D_CAND
        if p.arrival_time <= env.current_time and not p.is_complete:
            tq = env.current_time - p.arrival_time
            vec[off + 0] = _norm_time_log(tq)
            vec[off + 1] = p.wait_time / W15_WAIT_NORM  # W15 change: 200.0
            vec[off + 2] = _norm_time_log(p.time_since_last_execution)
            vec[off + 3] = _urgency_norm_task(p)
            vec[off + 4] = _norm_cpu(tasks[pid]['plan_cpu'])
            vec[off + 5] = _norm_mem(tasks[pid]['plan_mem'])
            vec[off + 6] = 1.0
    return vec


def select_window(queue: list[PoissonTask]) -> list[PoissonTask]:
    """Select up to 5 tasks to present to agent.
    N <= 5: all tasks.
    N > 5: 5 with longest wait_time (most-starved first; CHANGE 2).
    """
    if len(queue) <= N_PROCESSES:
        return list(queue)
    return sorted(queue, key=lambda t: t.wait_time, reverse=True)[:N_PROCESSES]


# ---------------------------------------------------------------------------
# Task pool generation
# ---------------------------------------------------------------------------

def generate_task_pool(trace_data: np.ndarray, lam: float,
                       rng: np.random.Generator, n_tasks: int) -> list[PoissonTask]:
    idx     = rng.integers(0, len(trace_data), size=n_tasks)
    samples = trace_data[idx]
    inter   = rng.exponential(scale=1.0 / lam, size=n_tasks)
    arrivals = np.cumsum(inter)

    tasks = []
    for i in range(n_tasks):
        burst, cpu, mem = float(samples[i, 0]), float(samples[i, 1]), float(samples[i, 2])
        if rng.random() < 0.5:
            tau, floor = rng.uniform(600.0, 1200.0), 0.2
        else:
            tau, floor = rng.uniform(600.0, 1000.0), 0.0
        tasks.append(PoissonTask(
            task_id=i, arrival_time=float(arrivals[i]),
            burst_length=burst, plan_cpu=cpu, plan_mem=mem,
            tau=tau, floor=floor,
        ))
    return tasks


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

def compute_rewards_poisson(queue: list[PoissonTask], chosen: PoissonTask,
                            q_actual: float) -> tuple[float, float]:
    r_vd = sum(
        value_delta(t.tau, t.floor, t.base_value, t.wait_time, q_actual)
        for t in queue
    ) / REWARD_SCALE

    waiting = [t for t in queue if t is not chosen]
    n_wait  = max(len(waiting), 1)
    r_ss = -sum(
        max(0.0, t.time_since_last_execution - STARVATION_THRESHOLD) / STARVATION_THRESHOLD
        for t in waiting
    ) / n_wait

    return r_vd, r_ss


# ---------------------------------------------------------------------------
# Omega sampling schedule (W15: 8000-ep boundary)
# ---------------------------------------------------------------------------

def sample_omega_w15(ep: int) -> float:
    if ep < 8000:
        return random.uniform(0.0, 1.0)
    u = random.uniform(0.0, 1.0)
    return math.sin(u * math.pi / 2) ** 2


# ---------------------------------------------------------------------------
# Training episode (Poisson, variable window, inline updates)
# ---------------------------------------------------------------------------

def run_train_episode(agent: W15Trainer, trace_data: np.ndarray,
                      lam: float, omega_s: float, rng: np.random.Generator,
                      buffer: OmegaReplayBuffer,
                      total_transitions: list) -> tuple[dict, list, list]:
    """One Poisson training episode. Returns (metrics, losses, entropies)."""
    pool = generate_task_pool(trace_data, lam, rng, N_GENERATE_TRAIN)
    pool.sort(key=lambda t: t.arrival_time)

    current_time = 0.0
    arrival_idx  = 0
    queue: list[PoissonTask] = []
    completed: list[PoissonTask] = []
    queue_depths: list[int] = []
    losses: list[float] = []
    ents:   list[float] = []

    def advance_arrivals() -> None:
        nonlocal arrival_idx
        while arrival_idx < len(pool) and pool[arrival_idx].arrival_time <= current_time:
            queue.append(pool[arrival_idx])
            arrival_idx += 1

    advance_arrivals()

    while len(completed) < N_COMPLETE_TRAIN:
        if not queue:
            if arrival_idx < len(pool):
                current_time = pool[arrival_idx].arrival_time
                advance_arrivals()
                continue
            else:
                break

        n_q = len(queue)
        queue_depths.append(n_q)

        candidates    = select_window(queue)
        n_cand        = len(candidates)
        state         = encode_state_w15(candidates, current_time)
        valid_actions = [slot * N_QT + qt
                         for slot in range(n_cand) for qt in range(N_QT)]

        action   = agent.select_action(state, valid_actions, omega_s)
        slot_idx = action // N_QT
        qt_idx   = action % N_QT
        quantum  = QUANTUM_TIERS[qt_idx]
        chosen   = candidates[slot_idx]
        q_actual = min(quantum, chosen.remaining_burst)

        r_vd, r_ss = compute_rewards_poisson(queue, chosen, q_actual)

        for t in queue:
            if t is not chosen:
                t.wait_time                += q_actual
                t.time_since_last_execution += q_actual
        chosen.time_since_last_execution = 0.0
        chosen.remaining_burst           -= q_actual
        current_time                     += q_actual

        if chosen.is_complete:
            chosen.completion_time = current_time
            queue.remove(chosen)
            completed.append(chosen)

        advance_arrivals()

        next_cands = select_window(queue) if queue else []
        sv_next    = encode_state_w15(next_cands, current_time)
        done       = (len(completed) >= N_COMPLETE_TRAIN)

        buffer.store(state, action, r_vd, r_ss, sv_next, done, omega_s)
        total_transitions[0] += 1

        if (total_transitions[0] >= WARMUP
                and len(buffer) >= BATCH_SIZE
                and total_transitions[0] % UPDATE_FREQ == 0):
            s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b = buffer.sample(BATCH_SIZE)
            loss, ent = agent.update(s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b)
            losses.append(loss)
            ents.append(ent)

    if completed:
        turnarounds = [t.completion_time - t.arrival_time for t in completed]
        mct    = float(np.mean(turnarounds))
        bursts = [t.burst_length for t in completed]
        slows  = [ta / max(b, 1e-6) for ta, b in zip(turnarounds, bursts)]
        med    = float(np.median(slows))
        starved = int(any(s > STARVATION_SLOW * med for s in slows))
    else:
        mct, starved = float("nan"), 0

    mean_q = float(np.mean(queue_depths)) if queue_depths else 0.0
    metrics = {"mct": mct, "n_completed": len(completed),
               "mean_queue_depth": mean_q, "starved": starved}
    return metrics, losses, ents


# ---------------------------------------------------------------------------
# Training loop — one seed
# ---------------------------------------------------------------------------

def train_seed_w15(seed: int, train_data: np.ndarray,
                   ckpt_every: int = CKPT_EVERY) -> W15Trainer:
    log(f"\n{'='*72}")
    log(f"Training seed={seed}  |  N_EPISODES={N_EPISODES}")
    log(f"λ~Uniform({LAM_MIN},{LAM_MAX})  |  N_COMPLETE_TRAIN={N_COMPLETE_TRAIN}")
    log(f"Omega schedule: Uniform 0-8000, Beta(0.5,0.5) 8000+")
    log(f"{'='*72}")

    torch.manual_seed(seed)
    agent = W15Trainer()
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    buffer = OmegaReplayBuffer(capacity=BUF_CAPACITY, state_dim=N_PROCESSES * D_CAND)

    total_transitions = [0]
    win_mct, win_starv, win_loss, win_ent, win_q, win_lam = [], [], [], [], [], []

    t_seed_start = time.time()

    for ep in range(1, N_EPISODES + 1):
        agent.lambda_ent = LAMBDA_START - (LAMBDA_START - LAMBDA_END) * (ep / N_EPISODES)
        omega_s = sample_omega_w15(ep)
        lam_ep  = float(rng.uniform(LAM_MIN, LAM_MAX))

        ep_metrics, losses, ents = run_train_episode(
            agent, train_data, lam_ep, omega_s, rng, buffer, total_transitions)

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        mct_val = ep_metrics["mct"] if not math.isnan(ep_metrics["mct"]) else 0.0
        win_mct.append(mct_val)
        win_starv.append(ep_metrics["starved"])
        win_q.append(ep_metrics["mean_queue_depth"])
        win_lam.append(lam_ep)

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        mean_ent  = float(np.mean(ents))   if ents   else float("nan")
        win_loss.append(mean_loss)
        win_ent.append(mean_ent)

        agent.decay_epsilon(ep, n_eps=N_EPISODES)

        if ep % LOG_EVERY == 0:
            n  = min(ep, 100)
            am = float(np.mean(win_mct[-n:]))
            st = float(np.mean(win_starv[-n:])) * 100.0
            al = float(np.nanmean(win_loss[-n:]))
            ah = float(np.nanmean(win_ent[-n:]))
            aq = float(np.mean(win_q[-n:]))
            al_lm = float(np.mean(win_lam[-n:]))

            grad_ratio = float("nan")
            if len(buffer) >= BATCH_SIZE:
                s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b = buffer.sample(BATCH_SIZE)
                _, _, ratio = agent.compute_grad_norms(
                    s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b)
                grad_ratio = ratio

            elapsed_min = (time.time() - t_seed_start) / 60
            log(f"  ep {ep:>6} | λ={al_lm:.3f} | ω={omega_s:.2f} | "
                f"loss={al:.4f} | H={ah:.4f} | MCT={am:.2f}s | "
                f"Starv={st:.1f}% | MeanQ={aq:.2f} | "
                f"grad_ratio={grad_ratio:.2f} | {elapsed_min:.1f}min")

            if not math.isnan(al) and al > LOSS_GATE:
                log(f"  STOP: loss {al:.2f} > {LOSS_GATE}")
                break
            if ep == 5000 and not math.isnan(ah) and ah > ENT_GATE_5K:
                log(f"  STOP: H={ah:.4f} > {ENT_GATE_5K} at ep 5000")
                break

        if ep % ckpt_every == 0:
            ckpt = os.path.join(OUT_DIR, f"w15_seed{seed}_ep{ep}.pt")
            agent.save(ckpt)
            log(f"  [ckpt] → {ckpt}")

    final_path = os.path.join(OUT_DIR, f"w15_seed{seed}_final.pt")
    agent.save(final_path)
    log(f"  [final] → {final_path}")
    log(f"  Seed {seed} done. ({(time.time()-t_seed_start)/60:.1f} min)")
    return agent


# ---------------------------------------------------------------------------
# Evaluation 1: Fixed N=5 (SchedEnv, W15 state norms)
# ---------------------------------------------------------------------------

def eval_fixed_n5_w15(agent: W15Trainer, test_sampler,
                      omega_s: float, n_eval: int, seed: int) -> dict:
    from w9_train import _valid_actions
    rng = np.random.default_rng(seed)
    mcts, starved_list = [], []

    for _ in range(n_eval):
        tasks = test_sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()
        sv   = encode_state_w15_fixed(env, tasks)
        done = False

        while not done:
            valid  = _valid_actions(env)
            action = agent.select_action(sv, valid, omega_s)
            _, _, done, info = env.step(action)
            sv = encode_state_w15_fixed(env, tasks)

        mcts.append(info.get("mean_completion_time_so_far") or 0.0)
        completed = [p for p in env.processes if p.is_complete]
        starved = 0
        if completed:
            turns  = [p.completion_time - p.arrival_time for p in completed]
            bursts = [p.burst_length for p in completed]
            slows  = [ta / max(b, 1e-6) for ta, b in zip(turns, bursts)]
            med    = float(np.median(slows))
            if any(s > STARVATION_SLOW * med for s in slows):
                starved = 1
        starved_list.append(starved)

    return {
        "mct_mean":   float(np.mean(mcts)),
        "mct_std":    float(np.std(mcts)),
        "starve_pct": float(np.mean(starved_list)) * 100.0,
        "ood_pct":    0.0,
    }


def eval_mlfq_fixed_n5(test_sampler, n_eval: int, seed: int) -> dict:
    result = run_mlfq(test_sampler, n_eval, seed)
    return {"mct_mean": result["mct_mean"], "mct_std": result["mct_std"],
            "starve_pct": result["starve_pct"], "ood_pct": 0.0}


# ---------------------------------------------------------------------------
# Evaluation 2: Poisson sweep
# ---------------------------------------------------------------------------

def _poisson_metrics(completed: list[PoissonTask], queue_depths: list[int],
                     ood_dec: int, total_dec: int) -> dict:
    if not completed:
        return {"mct": float("nan"), "starved_episode": 0,
                "mean_queue_depth": 0.0, "max_queue_depth": 0,
                "ood_pct": 0.0, "n_completed": 0}
    turnarounds = [t.completion_time - t.arrival_time for t in completed]
    mct    = float(np.mean(turnarounds))
    bursts = [t.burst_length for t in completed]
    slows  = [ta / max(b, 1e-6) for ta, b in zip(turnarounds, bursts)]
    med    = float(np.median(slows))
    starved = int(any(s > STARVATION_SLOW * med for s in slows))
    mean_q  = float(np.mean(queue_depths)) if queue_depths else 0.0
    max_q   = int(max(queue_depths)) if queue_depths else 0
    ood_p   = 100.0 * ood_dec / max(total_dec, 1)
    return {"mct": mct, "starved_episode": starved,
            "mean_queue_depth": mean_q, "max_queue_depth": max_q,
            "ood_pct": ood_p, "n_completed": len(completed)}


def simulate_w15_ep(agent: W15Trainer, task_pool: list[PoissonTask],
                    omega_s: float) -> dict:
    tasks = [deepcopy(t) for t in task_pool]
    tasks.sort(key=lambda t: t.arrival_time)

    current_time = 0.0
    arrival_idx  = 0
    queue:     list[PoissonTask] = []
    completed: list[PoissonTask] = []
    queue_depths: list[int] = []
    ood_dec = 0
    total_dec = 0

    def adv() -> None:
        nonlocal arrival_idx
        while arrival_idx < len(tasks) and tasks[arrival_idx].arrival_time <= current_time:
            queue.append(tasks[arrival_idx])
            arrival_idx += 1

    adv()

    while len(completed) < N_COMPLETE_EVAL:
        if not queue:
            if arrival_idx < len(tasks):
                current_time = tasks[arrival_idx].arrival_time
                adv()
                continue
            else:
                break

        n_q = len(queue)
        queue_depths.append(n_q)
        total_dec += 1
        if n_q > N_PROCESSES:
            ood_dec += 1

        candidates    = select_window(queue)
        n_cand        = len(candidates)
        state         = encode_state_w15(candidates, current_time)
        valid_actions = [s * N_QT + qt
                         for s in range(n_cand) for qt in range(N_QT)]

        action   = agent.select_action(state, valid_actions, omega_s)
        slot_idx = action // N_QT
        qt_idx   = action % N_QT
        quantum  = QUANTUM_TIERS[qt_idx]
        chosen   = candidates[slot_idx]
        q_actual = min(quantum, chosen.remaining_burst)

        for t in queue:
            if t is not chosen:
                t.wait_time                += q_actual
                t.time_since_last_execution += q_actual
        chosen.time_since_last_execution = 0.0
        chosen.remaining_burst          -= q_actual
        current_time                    += q_actual

        if chosen.is_complete:
            chosen.completion_time = current_time
            queue.remove(chosen)
            completed.append(chosen)

        adv()

    return _poisson_metrics(completed, queue_depths, ood_dec, total_dec)


def simulate_mlfq_ep(task_pool: list[PoissonTask]) -> dict:
    tasks = [deepcopy(t) for t in task_pool]
    tasks.sort(key=lambda t: t.arrival_time)

    current_time = 0.0
    arrival_idx  = 0
    queue:     list[PoissonTask] = []
    completed: list[PoissonTask] = []
    queue_depths: list[int] = []
    ood_dec = 0
    total_dec = 0

    mlfq_level:    dict[int, int]   = {}
    prev_remaining: dict[int, float] = {}
    last_id: int | None = None

    def adv() -> None:
        nonlocal arrival_idx
        while arrival_idx < len(tasks) and tasks[arrival_idx].arrival_time <= current_time:
            t = tasks[arrival_idx]
            queue.append(t)
            mlfq_level[t.task_id]    = 0
            prev_remaining[t.task_id] = t.burst_length
            arrival_idx += 1

    adv()

    while len(completed) < N_COMPLETE_EVAL:
        if not queue:
            if arrival_idx < len(tasks):
                current_time = tasks[arrival_idx].arrival_time
                adv()
                continue
            else:
                break

        n_q = len(queue)
        queue_depths.append(n_q)
        total_dec += 1
        if n_q > N_PROCESSES:
            ood_dec += 1

        # Demotion of last picked
        if last_id is not None:
            lt = next((t for t in queue if t.task_id == last_id), None)
            if lt is not None:
                lvl      = mlfq_level.get(lt.task_id, 0)
                consumed = prev_remaining.get(lt.task_id, lt.burst_length) - lt.remaining_burst
                if consumed >= QUANTUM_TIERS[lvl] - 1e-6 and lvl < 2:
                    mlfq_level[lt.task_id] = lvl + 1

        # Aging
        for t in queue:
            if t.time_since_last_execution > MLFQ_AGE_THRESH:
                mlfq_level[t.task_id] = 0

        # Pick
        chosen = None
        chosen_qt = 0
        for lvl in range(3):
            cands = [t for t in queue if mlfq_level.get(t.task_id, 0) == lvl]
            if cands:
                chosen    = min(cands, key=lambda t: t.arrival_time)
                chosen_qt = lvl
                break
        if chosen is None:
            chosen    = min(queue, key=lambda t: t.arrival_time)
            chosen_qt = 0

        quantum  = QUANTUM_TIERS[chosen_qt]
        q_actual = min(quantum, chosen.remaining_burst)
        prev_remaining[chosen.task_id] = chosen.remaining_burst

        for t in queue:
            if t is not chosen:
                t.wait_time                += q_actual
                t.time_since_last_execution += q_actual
        chosen.time_since_last_execution = 0.0
        chosen.remaining_burst          -= q_actual
        current_time                    += q_actual
        last_id                          = chosen.task_id

        if chosen.is_complete:
            chosen.completion_time = current_time
            queue.remove(chosen)
            completed.append(chosen)

        adv()

    return _poisson_metrics(completed, queue_depths, ood_dec, total_dec)


def eval_poisson_condition(agent_or_none, trace_data: np.ndarray,
                           lam: float, n_eval: int,
                           seeds: list[int], is_mlfq: bool) -> dict:
    all_mct, all_starved, all_mean_q, all_max_q, all_ood = [], [], [], [], []

    for seed_i, seed in enumerate(seeds):
        n_eps = n_eval // len(seeds) + (1 if seed_i < n_eval % len(seeds) else 0)
        for ep in range(n_eps):
            ep_rng = np.random.default_rng(seed * 10000 + ep + 77777)
            pool   = generate_task_pool(trace_data, lam, ep_rng, N_GENERATE_EVAL)
            if is_mlfq:
                m = simulate_mlfq_ep(pool)
            else:
                m = simulate_w15_ep(agent_or_none, pool, omega_s=OMEGA_EVAL)
            all_mct.append(m["mct"])
            all_starved.append(m["starved_episode"])
            all_mean_q.append(m["mean_queue_depth"])
            all_max_q.append(m["max_queue_depth"])
            all_ood.append(m["ood_pct"])

    return {
        "lam":              lam,
        "is_mlfq":          is_mlfq,
        "mct_mean":         float(np.nanmean(all_mct)),
        "mct_std":          float(np.nanstd(all_mct)),
        "starve_pct":       float(np.mean(all_starved)) * 100.0,
        "mean_queue_depth": float(np.mean(all_mean_q)),
        "max_queue_depth":  int(max(all_max_q)) if all_max_q else 0,
        "ood_pct":          float(np.mean(all_ood)),
    }


# ---------------------------------------------------------------------------
# Comparison table builder
# ---------------------------------------------------------------------------

def build_comparison_table(w15_fixed: dict, mlfq_fixed_live: dict,
                            poisson_results: dict) -> str:
    lines = []
    lines.append("W15 vs W14-ω vs MLFQ — Comparison Table")
    lines.append("=" * 80)
    hdr = f"{'Setting':<24} {'Agent':<10} {'MCT':>8} {'Starve%':>8} {'OOD%':>7}"
    lines.append(hdr)
    lines.append("-" * 80)

    # Fixed N=5
    lines.append(f"{'Fixed N=5':<24} {'MLFQ':<10} "
                 f"{MLFQ_FIXED_N5['mct']:>8.2f} {MLFQ_FIXED_N5['starve_pct']:>8.1f} {'0.0':>7}")
    lines.append(f"{'Fixed N=5':<24} {'W14-ω':<10} "
                 f"{W14_FIXED_N5['mct']:>8.2f} {W14_FIXED_N5['starve_pct']:>8.1f} {'0.0':>7}")
    lines.append(f"{'Fixed N=5':<24} {'W15':<10} "
                 f"{w15_fixed['mct_mean']:>8.2f} {w15_fixed['starve_pct']:>8.1f} {'0.0':>7}")
    lines.append("")

    for lam in EVAL_LAMBDAS:
        rho  = lam * 10.0
        key  = lam
        setting = f"rho={rho:.1f} (lam={lam:.2f})"
        mlfq_r = poisson_results[key]["mlfq"]
        w15_r  = poisson_results[key]["w15"]
        w14_r  = W14_POISSON.get(lam)

        lines.append(f"{setting:<24} {'MLFQ':<10} "
                     f"{mlfq_r['mct_mean']:>8.2f} {mlfq_r['starve_pct']:>8.1f} "
                     f"{mlfq_r['ood_pct']:>7.1f}")
        if w14_r:
            lines.append(f"{setting:<24} {'W14-omega':<10} "
                         f"{w14_r['mct']:>8.2f} {w14_r['starve_pct']:>8.1f} "
                         f"{w14_r['ood_pct']:>7.1f}")
        lines.append(f"{setting:<24} {'W15':<10} "
                     f"{w15_r['mct_mean']:>8.2f} {w15_r['starve_pct']:>8.1f} "
                     f"{w15_r['ood_pct']:>7.1f}")
        lines.append("")

    lines.append("=" * 80)
    lines.append("SUCCESS CRITERIA:")

    w15_rho07 = poisson_results.get(0.07, {}).get("w15", {}).get("mct_mean", 999.0)
    w15_rho09 = poisson_results.get(0.09, {}).get("w15", {}).get("mct_mean", 999.0)
    lines.append(f"  rho=0.7  W15 MCT < MLFQ 31.00s: "
                 f"{'PASS' if w15_rho07 < 31.00 else 'FAIL'}  (W15={w15_rho07:.2f}s)")
    lines.append(f"  rho=0.9  W15 MCT < MLFQ 51.73s: "
                 f"{'PASS' if w15_rho09 < 51.73 else 'FAIL'}  (W15={w15_rho09:.2f}s)")
    lines.append(f"  Fixed    W15 MCT < MLFQ 21.59s: "
                 f"{'PASS' if w15_fixed['mct_mean'] < 21.59 else 'FAIL'}  "
                 f"(W15={w15_fixed['mct_mean']:.2f}s)")
    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    log(f"W15 Variable-N Training  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"OUT_DIR:   {OUT_DIR}")
    log(f"LOG_FILE:  {LOG_FILE}")
    log(f"Seeds:     {SEEDS}")
    log(f"Episodes:  {N_EPISODES}")
    log(f"λ:         Uniform({LAM_MIN},{LAM_MAX})")
    log(f"N_COMPLETE_TRAIN: {N_COMPLETE_TRAIN}")
    log(f"W15_WAIT_NORM: {W15_WAIT_NORM} (W14 was 500.0)")
    log(f"Window: N<=5 all; N>5 longest-wait-first 5")

    # Load traces
    log(f"\nLoading training trace: {TRACE_TRAIN}")
    train_list = []
    with open(TRACE_TRAIN, newline="") as f:
        for row in csv.DictReader(f):
            try:
                dur = float(row["end_time"]) - float(row["start_time"])
                cpu = float(row["plan_cpu"])
                mem = float(row["plan_mem"])
                if dur > 0:
                    train_list.append((dur, cpu, mem))
            except (ValueError, TypeError, KeyError):
                pass
    train_data = np.array(train_list, dtype=np.float32)
    log(f"  {len(train_data):,} training tasks.")

    log(f"\nLoading test trace: {TRACE_TEST}")
    test_list = []
    with open(TRACE_TEST, newline="") as f:
        for row in csv.DictReader(f):
            try:
                dur = float(row["end_time"]) - float(row["start_time"])
                cpu = float(row["plan_cpu"])
                mem = float(row["plan_mem"])
                if dur > 0:
                    test_list.append((dur, cpu, mem))
            except (ValueError, TypeError, KeyError):
                pass
    test_data = np.array(test_list, dtype=np.float32)
    log(f"  {len(test_data):,} test tasks.")

    log(f"\nLoading fixed-N5 test sampler...")
    test_sampler = TraceEpisodeSampler5(TRACE_TEST)
    log(f"  Done.")

    # -----------------------------------------------------------------------
    # Training — single seed (parallel runs: one process per seed)
    trained_agents: dict[int, W15Trainer] = {}
    for seed in SEEDS:
        agent = train_seed_w15(seed, train_data, ckpt_every=CKPT_EVERY)
        trained_agents[seed] = agent

    log(f"\n{'='*72}")
    log(f"ALL SEEDS COMPLETE. Starting evaluation...")
    log(f"{'='*72}")

    best_agent = trained_agents[SEEDS[0]]
    best_agent.epsilon = 0.0

    # -----------------------------------------------------------------------
    # Eval 1: Fixed N=5
    # -----------------------------------------------------------------------
    log(f"\n--- EVAL 1: Fixed N=5 (N_EVAL={N_EVAL_FIXED}, omega={OMEGA_EVAL}) ---")

    log(f"  W15 (omega={OMEGA_EVAL})...")
    t0 = time.time()
    w15_fixed = eval_fixed_n5_w15(best_agent, test_sampler, OMEGA_EVAL, N_EVAL_FIXED, SEEDS[0])
    log(f"    MCT={w15_fixed['mct_mean']:.2f}+/-{w15_fixed['mct_std']:.2f}s  "
        f"Starve={w15_fixed['starve_pct']:.1f}%  ({time.time()-t0:.0f}s)")

    log(f"  MLFQ (live)...")
    t0 = time.time()
    mlfq_fixed = eval_mlfq_fixed_n5(test_sampler, N_EVAL_FIXED, SEEDS[0])
    log(f"    MCT={mlfq_fixed['mct_mean']:.2f}+/-{mlfq_fixed['mct_std']:.2f}s  "
        f"Starve={mlfq_fixed['starve_pct']:.1f}%  ({time.time()-t0:.0f}s)")

    # -----------------------------------------------------------------------
    # Eval 2: Poisson sweep
    # -----------------------------------------------------------------------
    log(f"\n--- EVAL 2: Poisson sweep (N_EVAL={N_EVAL_POISSON} each) ---")
    poisson_results: dict[float, dict] = {}

    for lam in EVAL_LAMBDAS:
        rho = lam * 10.0
        log(f"\n  lam={lam:.2f} (rho~{rho:.1f}):")

        log(f"    W15...")
        t0 = time.time()
        w15_p = eval_poisson_condition(best_agent, test_data, lam,
                                       N_EVAL_POISSON, SEEDS, is_mlfq=False)
        log(f"      MCT={w15_p['mct_mean']:.2f}+/-{w15_p['mct_std']:.2f}s  "
            f"Starve={w15_p['starve_pct']:.1f}%  "
            f"MeanQ={w15_p['mean_queue_depth']:.2f}  "
            f"OOD%={w15_p['ood_pct']:.1f}%  ({time.time()-t0:.0f}s)")

        log(f"    MLFQ...")
        t0 = time.time()
        mlfq_p = eval_poisson_condition(None, test_data, lam,
                                        N_EVAL_POISSON, SEEDS, is_mlfq=True)
        log(f"      MCT={mlfq_p['mct_mean']:.2f}+/-{mlfq_p['mct_std']:.2f}s  "
            f"Starve={mlfq_p['starve_pct']:.1f}%  "
            f"MeanQ={mlfq_p['mean_queue_depth']:.2f}  "
            f"OOD%={mlfq_p['ood_pct']:.1f}%  ({time.time()-t0:.0f}s)")

        poisson_results[lam] = {"w15": w15_p, "mlfq": mlfq_p}

    # -----------------------------------------------------------------------
    # Save eval results JSON
    # -----------------------------------------------------------------------
    eval_output = {
        "config": {
            "seeds": SEEDS, "n_episodes": N_EPISODES,
            "n_complete_train": N_COMPLETE_TRAIN,
            "lam_range": [LAM_MIN, LAM_MAX],
            "w15_wait_norm": W15_WAIT_NORM,
            "omega_eval": OMEGA_EVAL,
            "eval_lambdas": EVAL_LAMBDAS,
        },
        "fixed_n5": {
            "w15":         w15_fixed,
            "mlfq_live":   mlfq_fixed,
            "w14_prior":   W14_FIXED_N5,
            "mlfq_prior":  MLFQ_FIXED_N5,
        },
        "poisson": {str(lam): poisson_results[lam] for lam in EVAL_LAMBDAS},
        "w14_prior_poisson": {str(k): v for k, v in W14_POISSON.items()},
        "mlfq_prior_poisson": {str(k): v for k, v in MLFQ_POISSON.items()},
    }
    eval_path = os.path.join(OUT_DIR, "w15_eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2)
    log(f"\nEval results → {eval_path}")

    # -----------------------------------------------------------------------
    # Comparison table
    # -----------------------------------------------------------------------
    table_str = build_comparison_table(w15_fixed, mlfq_fixed, poisson_results)
    table_path = os.path.join(OUT_DIR, "w15_comparison_table.txt")
    with open(table_path, "w") as f:
        f.write(table_str)
    log(f"Comparison table → {table_path}")
    log("\n" + table_str)

    # -----------------------------------------------------------------------
    # Final
    # -----------------------------------------------------------------------
    total_h = (time.time() - t_start) / 3600
    log(f"\nTotal wall time: {total_h:.2f} hours")
    log(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log("TRAINING COMPLETE")

    _log_fh.close()


if __name__ == "__main__":
    main()
