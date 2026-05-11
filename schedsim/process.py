"""Process data class and factory functions."""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field


@dataclass
class Process:
    pid:              int
    arrival_time:     float
    burst_length:     float
    remaining_burst:  float = field(default=0.0)
    wait_time:        float = field(default=0.0)
    completion_time:  float | None = field(default=None)
    # Value-curve reward attributes — sampled per episode in env.reset()
    tau:              float = field(default=0.0)
    floor:            float = field(default=0.0)
    base_value:       float = field(default=1.0)
    # Starvation tracking — time elapsed since this process last received CPU
    # Resets to 0 when the process is scheduled; increments while it waits.
    time_since_last_execution: float = field(default=0.0)

    def __post_init__(self) -> None:
        # remaining_burst mirrors burst_length at construction unless caller
        # explicitly provides a different value (used in mid-episode snapshots).
        if self.remaining_burst == 0.0:
            self.remaining_burst = self.burst_length

    @property
    def is_complete(self) -> bool:
        return self.remaining_burst <= 0.0

    @property
    def turnaround_time(self) -> float | None:
        if self.completion_time is None:
            return None
        return self.completion_time - self.arrival_time


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def generate_fixed_processes() -> list[Process]:
    """Week 1 OOD fixed set — three simultaneous arrivals at t=0."""
    return [
        Process(pid=0, arrival_time=0,  burst_length=4),
        Process(pid=1, arrival_time=0,  burst_length=25),
        Process(pid=2, arrival_time=5,  burst_length=2),
        Process(pid=3, arrival_time=0,  burst_length=50),
        Process(pid=4, arrival_time=10, burst_length=8),
    ]


def generate_indist_processes() -> list[Process]:
    """In-distribution test set — one process per arrival slot."""
    return [
        Process(pid=0, arrival_time=0,  burst_length=4),
        Process(pid=1, arrival_time=2,  burst_length=25),
        Process(pid=2, arrival_time=5,  burst_length=2),
        Process(pid=3, arrival_time=8,  burst_length=50),
        Process(pid=4, arrival_time=10, burst_length=8),
    ]


def generate_random_processes(
    n:             int              = 5,
    arrival_slots: list[int]        = [0, 2, 5, 8, 10],
    burst_range:   tuple[float, float] = (1.0, 60.0),
    seed:          int | None       = None,
) -> list[Process]:
    """Sample n processes with staggered arrivals and random bursts.

    arrival_time : drawn WITHOUT replacement from arrival_slots
                   → exactly one process per slot, guaranteed stagger
    burst_length : drawn from U[burst_range[0], burst_range[1]]

    When seed=None the global random state is used (caller controls
    reproducibility via random.seed()).  Passing an explicit seed creates
    an isolated RNG that does not affect global state.
    """
    assert len(arrival_slots) >= n, "need at least n arrival slots"
    rng      = _random.Random(seed) if seed is not None else _random
    arrivals = rng.sample(arrival_slots, k=n)
    return [
        Process(
            pid          = i,
            arrival_time = float(arrivals[i]),
            burst_length = rng.uniform(burst_range[0], burst_range[1]),
        )
        for i in range(n)
    ]


if __name__ == "__main__":
    import random
    random.seed(0)

    print("=== Fixed (OOD) process set ===")
    for p in generate_fixed_processes():
        print(f"  P{p.pid}: arr={p.arrival_time:>4.0f}ms  burst={p.burst_length:>5.1f}ms")

    print("\n=== In-distribution process set ===")
    for p in generate_indist_processes():
        print(f"  P{p.pid}: arr={p.arrival_time:>4.0f}ms  burst={p.burst_length:>5.1f}ms")

    print("\n=== 3 random process sets (seed=None, global random.seed(0)) ===")
    for trial in range(3):
        procs = generate_random_processes()
        label = f"Trial {trial+1}"
        descs = [f"P{p.pid}(arr={p.arrival_time:.0f} b={p.burst_length:.1f})"
                 for p in procs]
        print(f"  {label}: {', '.join(descs)}")
