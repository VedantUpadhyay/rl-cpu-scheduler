"""Deterministic baseline schedulers.

Both classes expose the same episode interface used by the RL agent:
  policy.reset()             — called once per episode before env.reset()
  policy.select_action(env)  — returns an int action in [0, N_ACTIONS)

They operate on the live SchedEnv object (reading public attributes
env.processes and env.current_time) rather than on the discrete state
tuple, because their decision rules are defined over continuous process
state (arrival time, remaining burst) rather than discretised bins.
"""
from __future__ import annotations

try:
    from schedsim.env     import N_QUANTUM_TIERS
    from schedsim.process import Process
except ImportError:
    from env     import N_QUANTUM_TIERS  # type: ignore[no-redef]
    from process import Process          # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _runnable(env) -> list[Process]:
    """Processes that have arrived and still have remaining burst > 0."""
    return [
        p for p in env.processes
        if p.arrival_time <= env.current_time and not p.is_complete
    ]


# ---------------------------------------------------------------------------
# Round Robin (5 ms quantum)
# ---------------------------------------------------------------------------

class RoundRobin:
    """Cycles through runnable processes in PID order, 5 ms quantum each.

    After serving process i, the next pick is the lowest PID > i among
    currently runnable processes; if none exists, wrap to the lowest PID.
    This is equivalent to maintaining a circular queue sorted by PID.
    """

    def __init__(self) -> None:
        self._last_pid: int = -1

    def reset(self) -> None:
        self._last_pid = -1

    def select_action(self, env) -> int:
        ready = _runnable(env)
        if not ready:
            raise RuntimeError("RoundRobin: no runnable processes")

        pids = sorted(p.pid for p in ready)

        # First pid strictly greater than last served; wrap on exhaustion
        next_pid = next((p for p in pids if p > self._last_pid), pids[0])
        self._last_pid = next_pid

        # quantum_tier 1 → 5 ms (medium)
        return next_pid * N_QUANTUM_TIERS + 1


# ---------------------------------------------------------------------------
# First-Come First-Served (non-preemptive)
# ---------------------------------------------------------------------------

class FCFS:
    """Runs the earliest-arrived process to completion before switching.

    Ties in arrival time are broken by PID (lower PID first).
    Uses the 20 ms quantum to minimise decision steps; the env caps
    it at remaining_burst automatically, so the schedule is identical
    to true non-preemptive FCFS regardless of quantum size.
    """

    def __init__(self) -> None:
        self._current_pid: int | None = None

    def reset(self) -> None:
        self._current_pid = None

    def select_action(self, env) -> int:
        ready = _runnable(env)
        if not ready:
            raise RuntimeError("FCFS: no runnable processes")

        # Keep running the current process until it completes
        if self._current_pid is not None:
            still_here = next(
                (p for p in ready if p.pid == self._current_pid), None
            )
            if still_here is not None:
                return self._current_pid * N_QUANTUM_TIERS + 2  # long (20 ms)

        # Current process finished (or first step): pick earliest arrival
        chosen = min(ready, key=lambda p: (p.arrival_time, p.pid))
        self._current_pid = chosen.pid
        return self._current_pid * N_QUANTUM_TIERS + 2           # long (20 ms)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    try:
        from schedsim.env     import SchedEnv
        from schedsim.process import Process as _P
    except ImportError:
        from env     import SchedEnv   # type: ignore[assignment]
        from process import Process as _P  # type: ignore[assignment]

    PROCS = [
        _P(pid=0, arrival_time=0,  burst_length=4),
        _P(pid=1, arrival_time=0,  burst_length=25),
        _P(pid=2, arrival_time=5,  burst_length=2),
        _P(pid=3, arrival_time=0,  burst_length=50),
        _P(pid=4, arrival_time=10, burst_length=8),
    ]

    def _run_episode(env: SchedEnv, policy) -> float:
        policy.reset()
        env.reset()
        done = False
        while not done:
            action = policy.select_action(env)
            _, _, done, info = env.step(action)
        return info["mean_completion_time_so_far"]

    env = SchedEnv(PROCS)

    rr_mct   = _run_episode(env, RoundRobin())
    fcfs_mct = _run_episode(env, FCFS())

    print(f"Round Robin (5ms)  MCT = {rr_mct:.2f}ms")
    print(f"FCFS               MCT = {fcfs_mct:.2f}ms")
    print(f"SRPT optimal             28.40ms")
    print(f"RR beats FCFS?     {'YES' if rr_mct < fcfs_mct else 'NO'}")
