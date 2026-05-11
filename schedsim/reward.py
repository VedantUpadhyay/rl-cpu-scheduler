"""Isolated, swappable reward function for the RL scheduler.

Week 2 amendment: reward is now purely the dense area-integral signal
computed by the environment:

    R(t) = -n_active(t) × q_actual

where n_active = processes that have arrived and are not yet complete,
and q_actual = min(quantum, remaining_burst) = CPU time actually used.

Week 3 amendment: reward is normalised by REWARD_SCALE = 100.0.

    R_norm(t) = (-n_active(t) × q_actual) / REWARD_SCALE

Rationale: with N=5 processes and max quantum=20ms, the maximum raw
per-step reward magnitude is 5 × 20 = 100, so dividing by 100 keeps
each step in [-1, 0].  Over ~150 steps per episode, cumulative Q-value
targets stay in roughly [-150, 0], a range the network can represent
without numerical divergence.

The function reads 'env_reward' from the info dict, which env.step()
populates with the exact value it computes internally.  This keeps the
reward computation in one place (the env) and reward.py as the
authoritative interface for the training loop.

Expected info dict schema (provided by env.step())
---------------------------------------------------
{
    'env_reward':   float,   # -n_active * q_actual for this step
    ...                      # other fields unused by this function
}
"""
from __future__ import annotations

# Maximum possible |reward| per step: N_PROCESSES * max_quantum = 5 * 20
REWARD_SCALE: float = 100.0


def compute_reward(
    info:      dict,
    prev_info: dict,
    config:    dict | None = None,   # reserved for future amendments
) -> float:
    """Return the normalised dense area-integral reward for this step.

    Parameters
    ----------
    info      : info dict returned by env.step() after the step.
    prev_info : info dict from the previous step (unused; retained for
                interface compatibility with callers).
    config    : reserved; currently ignored.

    Returns
    -------
    float in [-1.0, 0.0]  (raw env_reward divided by REWARD_SCALE=100.0)
    """
    return info.get("env_reward", 0.0) / REWARD_SCALE


# ---------------------------------------------------------------------------
# Standalone unit tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    BLANK = {"env_reward": 0.0}

    def _run(label: str, info: dict) -> None:
        r = compute_reward(info, BLANK)
        print(f"  {label:<45}  reward = {r:+.4f}")

    print(f"REWARD_SCALE = {REWARD_SCALE}  (raw rewards divided by this constant)")
    print()

    # --- Scenario A: 3 active processes, 5ms quantum consumed -----------
    print("Scenario A: n_active=3, q_actual=5ms")
    info_a = {"env_reward": -3 * 5.0}
    _run("n=3, q=5ms  →  raw=-15.0  normalised=-0.15", info_a)
    expected_a = -15.0 / REWARD_SCALE
    assert abs(compute_reward(info_a, BLANK) - expected_a) < 1e-9
    print()

    # --- Scenario B: 1 active process (last job), 4ms burst remaining ---
    print("Scenario B: n_active=1, q_actual=4ms (last job finishing)")
    info_b = {"env_reward": -1 * 4.0}
    _run("n=1, q=4ms  →  raw=-4.0  normalised=-0.04", info_b)
    assert abs(compute_reward(info_b, BLANK) - (-4.0 / REWARD_SCALE)) < 1e-9
    print()

    # --- Scenario C: idle advance (no process ran) ----------------------
    print("Scenario C: idle clock advance (n_active=0 → env_reward=0)")
    info_c = {"env_reward": 0.0}
    _run("n=0, q=any  →  raw=0.0   normalised=0.0", info_c)
    assert abs(compute_reward(info_c, BLANK) - 0.0) < 1e-9
    print()

    # --- Scenario D: worst-case per-step reward --------------------------
    print("Scenario D: worst-case  n_active=5, q_actual=20ms")
    info_d = {"env_reward": -5 * 20.0}
    _run("n=5, q=20ms →  raw=-100.0  normalised=-1.0", info_d)
    assert abs(compute_reward(info_d, BLANK) - (-1.0)) < 1e-9
    print()

    # --- Scenario E: load-neutrality check ------------------------------
    print("Scenario E: load-neutrality — same n_active, same q_actual")
    info_light = {"env_reward": -2 * 5.0}
    info_heavy = {"env_reward": -2 * 5.0}
    _run("light load (env_reward=-10)", info_light)
    _run("heavy load (env_reward=-10)", info_heavy)
    assert compute_reward(info_light, BLANK) == compute_reward(info_heavy, BLANK)
    print("  Load-neutral: PASS")
    print()

    print("All assertions passed.")
