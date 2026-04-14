"""Week 9 verification — normalization check + permutation invariance unit test.
No training. Does not modify agent.py.
"""
from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PROCESSES = 5
N_QT        = 3
N_ACTIONS   = N_PROCESSES * N_QT   # 15
D_CAND      = 5   # per-process feature dim (was 3)
D_ATTN      = 16  # attention head dim     (was 8)
D_V         = 16  # value dim              (was 8)
# a6: [remaining_norm, arrived_flag, wait_norm, cpu_norm, mem_norm, qt/2.0]
D_A6        = 6   # candidate+quantum vec  (was 4)
D_MLP_IN    = D_V + D_A6  # 22              (was 12)

BURST_P95_FILT = 36.0
_LOG_DENOM     = float(np.log1p(BURST_P95_FILT))
WAIT_NORM      = BURST_P95_FILT * N_PROCESSES   # 180.0
CPU_MAX        = 800.0
MEM_P95        = 0.59

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def norm_burst(burst: float) -> float:
    return float(np.log1p(burst) / _LOG_DENOM)

def norm_cpu(plan_cpu: float) -> float:
    return float(plan_cpu / CPU_MAX)

def norm_mem(plan_mem: float) -> float:
    return float(min(plan_mem / MEM_P95, 1.0))

# ---------------------------------------------------------------------------
# Tiny Week 9 AttentionDQN (weights only — no training machinery)
# ---------------------------------------------------------------------------

class AttentionDQN9:
    """Week 9: 5-dim per-process encoding, d_attn=16, a6=6-dim."""

    def __init__(self, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        def he(fan_in: int, *shape: int) -> np.ndarray:
            return rng.standard_normal(shape).astype(np.float64) * np.sqrt(2.0 / fan_in)

        # Attention projections: 5 → 16
        self.W_Q = he(D_CAND, D_CAND, D_ATTN);  self.b_Q = np.zeros(D_ATTN)
        self.W_K = he(D_CAND, D_CAND, D_ATTN);  self.b_K = np.zeros(D_ATTN)
        self.W_V = he(D_CAND, D_CAND, D_V);      self.b_V = np.zeros(D_V)

        # MLP: 22 → 64 → 32 → 1
        self._W = [he(D_MLP_IN, D_MLP_IN, 64),
                   he(64,       64,        32),
                   he(32,       32,         1)]
        self._b = [np.zeros(64), np.zeros(32), np.zeros(1)]

    def _param_count(self) -> int:
        attn = (D_CAND*D_ATTN + D_ATTN) * 3   # W_Q,b_Q  W_K,b_K  W_V,b_V
        mlp  = (D_MLP_IN*64 + 64) + (64*32 + 32) + (32*1 + 1)
        return attn + mlp

    def _build_competitor_data(
        self, states: np.ndarray, pids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return comp_encs (batch,4,5) and comp_valid (batch,4) bool."""
        batch    = states.shape[0]
        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        all_pids = np.tile(np.arange(N_PROCESSES), (batch, 1))
        mask     = all_pids != pids[:, None]
        comp_idx = all_pids[mask].reshape(batch, 4)
        bidx     = np.arange(batch)[:, None]
        comp_encs  = s3d[bidx, comp_idx]                           # (batch,4,5)
        # valid: arrived_flag (index 1) > 0.5 AND remaining (index 0) > 1e-6
        comp_valid = (comp_encs[:, :, 1] > 0.5) & (comp_encs[:, :, 0] > 1e-6)
        return comp_encs, comp_valid

    def _attention_forward(
        self,
        cand_enc:   np.ndarray,  # (batch, 5)
        comp_encs:  np.ndarray,  # (batch, 4, 5)
        comp_valid: np.ndarray,  # (batch, 4) bool
    ) -> tuple[np.ndarray, dict]:
        batch     = cand_enc.shape[0]
        comp_flat = comp_encs.reshape(batch * 4, D_CAND)

        q     = cand_enc @ self.W_Q + self.b_Q                    # (batch, 16)
        K_flat = comp_flat @ self.W_K + self.b_K
        V_flat = comp_flat @ self.W_V + self.b_V
        K = K_flat.reshape(batch, 4, D_ATTN)
        V = V_flat.reshape(batch, 4, D_V)

        scores = np.einsum('bi,bji->bj', q, K) / np.sqrt(D_ATTN)  # (batch, 4)

        invalid       = ~comp_valid
        scores_masked = scores.copy()
        scores_masked[invalid] = -1e9
        scores_shifted = scores_masked - scores_masked.max(axis=1, keepdims=True)
        exp_s          = np.exp(scores_shifted)
        exp_s[invalid] = 0.0
        denom          = exp_s.sum(axis=1, keepdims=True) + 1e-10
        weights        = exp_s / denom

        context = np.einsum('bj,bjd->bd', weights, V)             # (batch, 16)
        return context, dict(weights=weights)

    def forward(self, state: np.ndarray, action: int) -> float:
        """Single Q-value for (state, action)."""
        states  = state[None].astype(np.float64)   # (1, 25)
        actions = np.array([action], dtype=np.int32)
        batch   = 1
        pids    = actions // N_QT
        qts_raw = np.array([0.25, 1.0, 4.0])[actions % N_QT]     # (1,)

        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        cand_enc = s3d[np.arange(batch), pids]                    # (1, 5)

        comp_encs, comp_valid = self._build_competitor_data(states, pids)
        context, _ = self._attention_forward(cand_enc, comp_encs, comp_valid)

        a6 = np.column_stack([cand_enc, qts_raw[:, None] / 2.0])  # (1, 6)
        x  = np.concatenate([context, a6], axis=1)                # (1, 22)

        z1 = x  @ self._W[0] + self._b[0]
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ self._W[1] + self._b[1]
        h2 = np.maximum(0.0, z2)
        z3 = h2 @ self._W[2] + self._b[2]
        return float(z3[0, 0])


# ---------------------------------------------------------------------------
# State builder: 25-dim vector
# Per-process slot (i*5 : i*5+5):
#   [remaining_norm, arrived_flag, wait_norm, cpu_norm, mem_norm]
# ---------------------------------------------------------------------------

def make_state(
    processes: list[dict],       # list of dicts with keys: pid, burst, cpu, mem, arrived, wait
    n: int = N_PROCESSES,
) -> np.ndarray:
    """Build 25-dim state from process list. Missing PIDs get all-zeros."""
    state = np.zeros(n * D_CAND, dtype=np.float64)
    for p in processes:
        i = p["pid"]
        state[i*D_CAND + 0] = norm_burst(p.get("burst", 0.0))
        state[i*D_CAND + 1] = 1.0 if p.get("arrived", True) else 0.0
        state[i*D_CAND + 2] = p.get("wait", 0.0) / WAIT_NORM
        state[i*D_CAND + 3] = norm_cpu(p.get("cpu", 0.0))
        state[i*D_CAND + 4] = norm_mem(p.get("mem", 0.0))
    return state


# ===========================================================================
# STEP 4 — Normalization verification
# ===========================================================================
print("=" * 60)
print("STEP 4 — Normalization verification")
print("=" * 60)
print()
print("plan_cpu normalization (CPU_MAX=800):")
for cpu in [50, 100, 400, 800]:
    print(f"  plan_cpu={cpu:>4}  →  cpu_norm = {norm_cpu(cpu):.4f}")

print()
print("plan_mem normalization (MEM_P95=0.59):")
for mem in [0.20, 0.39, 0.59]:
    print(f"  plan_mem={mem:.2f}  →  mem_norm = {norm_mem(mem):.4f}")

print()
print("log-burst normalization (BURST_P95=36s, LOG_DENOM={:.4f}):".format(_LOG_DENOM))
for burst in [3.0, 10.0, 15.0, 36.0]:
    print(f"  burst={burst:>5.1f}s  →  burst_norm = {norm_burst(burst):.4f}")


# ===========================================================================
# STEP 3 — Parameter count
# ===========================================================================
agent = AttentionDQN9(seed=42)
n_params = agent._param_count()
print()
print("=" * 60)
print("STEP 3 — Architecture parameter count")
print("=" * 60)
attn_params = (D_CAND*D_ATTN + D_ATTN) * 3
mlp_params  = (D_MLP_IN*64 + 64) + (64*32 + 32) + (32*1 + 1)
print(f"  Attention projections (3 × [5×16 + 16]): {attn_params}")
print(f"  MLP (22×64+64) + (64×32+32) + (32×1+1): {mlp_params}")
print(f"  Total: {n_params} parameters")
print(f"  (W8c had 3,041 parameters)")


# ===========================================================================
# STEP 5 — Permutation invariance unit tests
# ===========================================================================
print()
print("=" * 60)
print("STEP 5 — Permutation invariance unit tests")
print("=" * 60)

# --- Test 1: Q_A must equal Q_B ---
# State A: P0 is candidate, P1 and P2 are active competitors
state_A = make_state([
    dict(pid=0, burst=3.0,  cpu=100, mem=0.30, arrived=True, wait=0.0),  # candidate
    dict(pid=1, burst=10.0, cpu=50,  mem=0.20, arrived=True, wait=0.0),  # competitor
    dict(pid=2, burst=15.0, cpu=100, mem=0.39, arrived=True, wait=0.0),  # competitor
    # P3, P4: all zeros (not arrived)
])

# State B: P4 is candidate (same features as P0 in A)
#          P2 has the features of P1 in A (burst=10, cpu=50, mem=0.20)
#          P1 has the features of P2 in A (burst=15, cpu=100, mem=0.39)
state_B = make_state([
    dict(pid=1, burst=15.0, cpu=100, mem=0.39, arrived=True, wait=0.0),  # competitor
    dict(pid=2, burst=10.0, cpu=50,  mem=0.20, arrived=True, wait=0.0),  # competitor
    dict(pid=4, burst=3.0,  cpu=100, mem=0.30, arrived=True, wait=0.0),  # candidate
    # P0, P3: all zeros
])

qt = 2  # tier2 = 4.0s
action_A = 0 * N_QT + qt   # P0, tier2
action_B = 4 * N_QT + qt   # P4, tier2

Q_A = agent.forward(state_A, action_A)
Q_B = agent.forward(state_B, action_B)
delta_AB = abs(Q_A - Q_B)

print()
print("Test 1 — SYMMETRY: same candidates/competitors, different PIDs")
print(f"  Q(P0/tier2 in State A) = {Q_A:.12f}")
print(f"  Q(P4/tier2 in State B) = {Q_B:.12f}")
print(f"  |Q_A - Q_B|            = {delta_AB:.2e}")
print(f"  PASS: {delta_AB < 1e-10}")

# --- Test 2 (negative): change P2 burst from 15s → 11s in State B ---
state_B_neg = make_state([
    dict(pid=1, burst=11.0, cpu=100, mem=0.39, arrived=True, wait=0.0),  # changed!
    dict(pid=2, burst=10.0, cpu=50,  mem=0.20, arrived=True, wait=0.0),
    dict(pid=4, burst=3.0,  cpu=100, mem=0.30, arrived=True, wait=0.0),
])

Q_B_neg = agent.forward(state_B_neg, action_B)
delta_neg = abs(Q_A - Q_B_neg)

print()
print("Test 2 — NEGATIVE: P1 burst changed 15s → 11s in State B")
print(f"  Q(P0/tier2 in State A)          = {Q_A:.12f}")
print(f"  Q(P4/tier2 in State B modified) = {Q_B_neg:.12f}")
print(f"  |Delta|                         = {delta_neg:.6f}")
print(f"  PASS (must differ): {delta_neg > 1e-6}")

print()
print("=" * 60)
print("All verification complete.")
print("=" * 60)
