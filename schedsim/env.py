"""Scheduler environment: state, step, reset.

State  : tuple[int, ...] of length 5 — one rb_bin per process (values 0-6).
Action : int in [0, 14]; action = process_idx * 3 + quantum_tier_idx.
Reward : value-curve sum over active processes / NEW_REWARD_SCALE.
"""
from __future__ import annotations

import math
import random

try:
    from schedsim.process import Process
except ImportError:
    from process import Process  # type: ignore[no-redef]  # running as script


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTUM_TIERS: tuple[float, ...] = (0.5, 2.0, 8.0)   # seconds — short, medium, long
N_PROCESSES    = 5
N_QUANTUM_TIERS = 3
N_ACTIONS      = N_PROCESSES * N_QUANTUM_TIERS  # 15

# Burst normalization constants derived from Alibaba 2018 batch_task trace (training split).
BURST_P95: float = 397.0           # 95th-percentile duration, seconds
WAIT_NORM: float = BURST_P95 * N_PROCESSES  # max plausible wait ≈ 1985.0 s

NEW_REWARD_SCALE: float = 20.0    # max plausible |reward| per step ≈ 5 * 4.0


def value_delta(tau: float, floor: float, base_value: float,
                delay: float, q: float) -> float:
    """Incremental value lost by extending a process's delay by q seconds.

    V(d) = base_value * max(floor, exp(-d / tau))
    Returns V(delay + q) - V(delay)  ≤ 0.
    Handles tau == 0 gracefully (flat curve → returns 0.0).
    """
    if tau <= 0.0:
        return 0.0
    v_before = base_value * max(floor, math.exp(-delay / tau))
    v_after  = base_value * max(floor, math.exp(-(delay + q) / tau))
    return v_after - v_before


def compute_step_reward(
    chosen:       Process,
    all_runnable: list,
    current_time: float,
    q_actual:     float,
    weights:      list | None = None,
) -> float:
    """Composite step reward using only observable features (no burst time).

    Components (all normalized to approx [0, 1]):
      w2 · starvation  — avg time_since_last_execution across runnable
      w3 · wait        — avg accumulated wait_time across runnable
      w4 · queue_age   — avg time since arrival across runnable
      w5 · urgency     — value loss rate of chosen process / 0.1

    Default weights: [0.6, 0.1, 0.1, 0.1, 0.1] (w1 reserved for
    completion penalty applied externally in the training loop;
    w2-w5 are fairness secondary pressure).
    """
    if weights is None:
        weights = [0.6, 0.1, 0.1, 0.1, 0.1]
    _w1, w2, w3, w4, w5 = weights

    n    = max(len(all_runnable), 1)
    NORM = 500.0

    starvation   = sum(p.time_since_last_execution for p in all_runnable) / (n * NORM)
    wait_penalty = sum(p.wait_time                 for p in all_runnable) / (n * NORM)
    queue_age    = sum(current_time - p.arrival_time for p in all_runnable) / (n * NORM)

    delay        = chosen.wait_time
    v0           = chosen.base_value
    v_now        = (v0 * max(chosen.floor, math.exp(-delay / chosen.tau))
                    if chosen.tau > 0.0 else v0)
    urgency      = (v0 - v_now) / max(delay, 1.0)
    urgency_norm = urgency / 0.1

    return -(w2 * starvation + w3 * wait_penalty + w4 * queue_age + w5 * urgency_norm)


# rb_bin breakpoints (upper-exclusive) scaled to trace-second units.
# bin 0 = complete, bin 6 = not yet arrived; bins 1-5 = active tiers.
_RB_BREAKPOINTS = (33, 100, 200, 397)  # seconds (≈ old 5,15,30,60 ms × BURST_P95/60)


def _rb_bin(remaining: float, arrived: bool, complete: bool) -> int:
    """Map a process's live status to a discrete bin in {0 … 6}."""
    if complete:
        return 0
    if not arrived:
        return 6
    # arrived and not complete — remaining > 0
    for level, threshold in enumerate(_RB_BREAKPOINTS, start=1):
        if remaining < threshold:
            return level
    return 5   # remaining >= 60 ms


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SchedEnv:
    C_INVALID = 250.0   # N * max_burst = 5 * 50
    MAX_STEPS  = 500

    def __init__(self, process_templates: list[Process]) -> None:
        assert len(process_templates) > 0, "process_templates must be non-empty"
        self._templates    = process_templates
        self.n_processes   = len(process_templates)
        self.n_actions     = self.n_processes * N_QUANTUM_TIERS
        self.processes:    list[Process] = []
        self.current_time: float = 0.0
        self.step_count:   int   = 0
        self.completed_pids: list[int] = []
        self._step_reward: float = 0.0   # cached for _info(); reset each step

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[int, ...]:
        """Reset to episode start. Returns initial discrete state."""
        self.processes = [
            Process(
                pid          = t.pid,
                arrival_time = t.arrival_time,
                burst_length = t.burst_length,
            )
            for t in self._templates
        ]
        # Sample value-curve parameters per episode (calibrated to trace delays)
        for p in self.processes:
            if random.random() < 0.5:   # steep curve
                p.tau   = random.uniform(600.0, 1200.0)
                p.floor = 0.2
            else:                        # smooth curve
                p.tau   = random.uniform(600.0, 1000.0)
                p.floor = 0.0
            p.base_value = 1.0
        self.current_time   = 0.0
        self.step_count     = 0
        self.completed_pids = []
        self._step_reward   = 0.0
        self._maybe_advance_clock()
        return self._discretize_state()

    def step(self, action: int) -> tuple[tuple[int, ...], float, bool, dict]:
        """Execute one scheduling decision.

        Returns
        -------
        next_state : discrete state tuple
        reward     : float
        done       : True when all processes complete
        info       : dict with current_time, completed_pids,
                     mean_completion_time_so_far
        """
        assert 0 <= action < self.n_actions, (
            f"action {action} out of range [0, {self.n_actions})"
        )

        process_idx   = action // N_QUANTUM_TIERS
        quantum_tier  = action %  N_QUANTUM_TIERS
        quantum       = QUANTUM_TIERS[quantum_tier]
        chosen        = self.processes[process_idx]
        runnable      = self._get_runnable()

        # --- Invalid action (process not runnable) ----------------------
        if chosen not in runnable:
            self.step_count += 1
            return self._discretize_state(), -self.C_INVALID, self._is_done(), self._info()

        # --- Valid action -----------------------------------------------
        q_actual = min(float(quantum), chosen.remaining_burst)

        # Update starvation tracking before modifying state
        chosen.time_since_last_execution = 0.0
        for p in runnable:
            if p is not chosen:
                p.time_since_last_execution += q_actual

        reward = compute_step_reward(chosen, runnable, self.current_time, q_actual)
        self._step_reward = reward

        # Run chosen process
        chosen.remaining_burst -= q_actual

        # Accumulate wait for every other currently-runnable process
        for p in runnable:
            if p is not chosen:
                p.wait_time += q_actual

        # Advance simulation clock
        self.current_time += q_actual

        # Handle completion
        if chosen.is_complete:
            chosen.completion_time = self.current_time
            self.completed_pids.append(chosen.pid)

        # Idle-gap handling: if nothing is runnable, jump to next arrival
        self._maybe_advance_clock()

        done = self._is_done()
        self.step_count += 1
        return self._discretize_state(), reward, done, self._info()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_runnable(self) -> list[Process]:
        """Processes that have arrived and still have remaining burst."""
        return [
            p for p in self.processes
            if p.arrival_time <= self.current_time and not p.is_complete
        ]

    def _discretize_state(self) -> tuple[int, ...]:
        """Map continuous process states to the discrete rb_bin 5-tuple."""
        return tuple(
            _rb_bin(
                p.remaining_burst,
                arrived = p.arrival_time <= self.current_time,
                complete = p.is_complete,
            )
            for p in self.processes
        )

    def _maybe_advance_clock(self) -> None:
        """Advance clock to next arrival when no processes are runnable.

        This is env bookkeeping — not an agent action — so it emits no reward
        (n_active = 0 during any true idle gap, so the area integral is 0).
        """
        while not self._get_runnable():
            pending = [
                p for p in self.processes
                if not p.is_complete and p.arrival_time > self.current_time
            ]
            if not pending:
                break
            self.current_time = min(p.arrival_time for p in pending)

    def _is_done(self) -> bool:
        return all(p.is_complete for p in self.processes)

    def _info(self) -> dict:
        completed = [p for p in self.processes if p.is_complete]
        mct = (
            sum(p.completion_time - p.arrival_time for p in completed) / len(completed)
            if completed else None
        )
        return {
            "current_time":                self.current_time,
            "completed_pids":              list(self.completed_pids),
            "mean_completion_time_so_far": mct,
            # Fields required by reward.compute_reward
            "completion_times": {p.pid: p.completion_time for p in completed},
            "num_waiting":      len(self._get_runnable()),
            "env_reward":       self._step_reward,
        }


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    random.seed(42)

    # Fixed process set from spec
    PROCESSES = [
        Process(pid=0, arrival_time=0,  burst_length=4),
        Process(pid=1, arrival_time=0,  burst_length=25),
        Process(pid=2, arrival_time=5,  burst_length=2),
        Process(pid=3, arrival_time=0,  burst_length=50),
        Process(pid=4, arrival_time=10, burst_length=8),
    ]

    QTIER_LABEL = {0: "short(1ms) ", 1: "med(5ms)  ", 2: "long(20ms)"}

    env   = SchedEnv(PROCESSES)
    state = env.reset()

    print(f"Initial state: {state}")
    print(f"  (rb_bins for P0..P4; 0=done, 1-5=active tier, 6=not arrived)")
    print()
    print(f"{'Step':>4}  {'Action':<22}  {'Reward':>8}  {'State':<27}  {'t':>6}  Done")
    print("-" * 85)

    done         = False
    total_reward = 0.0

    while not done:
        if env.step_count >= env.MAX_STEPS:
            print("MAX_STEPS reached — aborting.")
            sys.exit(1)

        # Masked random: sample only from valid (runnable) actions
        valid_actions = [
            idx * N_QUANTUM_TIERS + qt
            for idx in range(N_PROCESSES)
            for qt in range(N_QUANTUM_TIERS)
            if state[idx] not in (0, 6)
        ]
        if not valid_actions:
            print("No valid actions — env bug.")
            sys.exit(1)

        action = random.choice(valid_actions)
        pid    = action // N_QUANTUM_TIERS
        qt     = action %  N_QUANTUM_TIERS
        label  = f"P{pid} / {QTIER_LABEL[qt]}"

        next_state, reward, done, info = env.step(action)
        total_reward += reward

        print(
            f"{env.step_count:>4}  {label:<22}  {reward:>8.1f}"
            f"  {str(next_state):<27}  {info['current_time']:>6.1f}  {done}"
        )
        state = next_state

    print("-" * 85)
    mct = info["mean_completion_time_so_far"]
    print(f"Steps: {env.step_count} | Final t = {info['current_time']:.1f}ms")
    print(f"Completion order : PIDs {info['completed_pids']}")
    print(f"Mean turnaround  : {mct:.1f}ms")
    print(f"Total reward     : {total_reward:.1f}")
    print()
    print(f"Targets  —  SRPT optimal: 28.4ms  |  RR(5ms) baseline: 37.4ms")
    print(f"Gap to SRPT: {mct - 28.4:+.1f}ms  |  Beat RR? {'YES' if mct < 37.4 else 'NO'}")
