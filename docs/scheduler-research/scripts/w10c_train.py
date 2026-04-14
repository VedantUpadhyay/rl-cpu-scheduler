"""Week 11d — W11d: 6-dim state (true burst + 4 fairness + arrived_flag), rebalanced reward.
2-head attention (d_head=8 each), output projection W_O:16→16.
Reward: w1=0.6 CT bonus + w2-w5=0.1 fairness. Checkpointing at ep1000/2000/5000/10000.
"""
from __future__ import annotations
import csv, os, random, sys, time
import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")
sys.path.insert(0, "/tmp")

from schedsim.env    import SchedEnv, N_PROCESSES, N_QUANTUM_TIERS
from schedsim.agent  import AdamOptimizer, ReplayBuffer
from schedsim.process import Process
from w9_train import (
    TraceEpisodeSampler5,
    _encode_state, _valid_actions, _make_procs,
    D_CAND, N_QT, N_ACTIONS, QT_VALUES,
    _LOG_DENOM, _norm_cpu, _norm_mem,
    BURST_P95_FILT, WAIT_NORM, CPU_MAX, MEM_P95,
    REWARD_SCALE,
)

# ---------------------------------------------------------------------------
# Multi-head attention constants
# ---------------------------------------------------------------------------
D_ATTN  = 16          # total attention dim (same as W9)
N_HEADS = 2
D_HEAD  = D_ATTN // N_HEADS   # 8 per head
D_V     = D_HEAD               # 8 per head (value dim per head)
D_V_TOT = D_ATTN               # 16 after concat
D_A6    = D_CAND + 1           # candidate enc (8) + quantum tier (1) = 9
D_MLP_IN = D_V_TOT + D_A6     # 25

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
TRACE_PATH   = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
LOG_PATH     = "/tmp/w11d_train.log"
WEIGHTS_PATH = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                "GRAD - FALL 23/UCSC/Capstone/results/w11d_final.npz")
RESULTS_DIR  = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                "GRAD - FALL 23/UCSC/Capstone/results")

N_EPISODES         = 10_000
LR                 = 0.001
GAMMA              = 1.0
GRAD_CLIP          = 1.0
BUF_CAPACITY       = 10_000
BATCH_SIZE         = 32
TARGET_UPDATE_FREQ = 200
WARMUP             = 500
LAMBDA_START       = 0.30
LAMBDA_END         = 0.005

PRINT_AT       = set(range(500, N_EPISODES + 1, 500))
CHECKPOINT_AT  = {1000, 2000, 5000, 10000}
LOSS_GATE      = 100.0
ENT_GATE_5K    = 1.2


# ---------------------------------------------------------------------------
# AttentionDQN10C — 2-head attention
# ---------------------------------------------------------------------------

class AttentionDQN10C:
    """
    Week 11d: 6-dim state (true burst + 4 fairness + arrived_flag), 2-head attention (d_head=8),
    output projection W_O: 16→16, MLP 23→64→32→1.

    Parameter count:
      Head projections (QKV per head):  2 × 3 × (6×8 + 8) = 336
      Output projection W_O (16×16+16):                    = 272
      MLP (23×64+64 + 64×32+32 + 32×1+1):                  = 3,649
      Total: 4,257
    """

    def __init__(self, lr: float = 0.001, gamma: float = 1.0,
                 grad_clip: float = 1.0, lambda_ent: float = 0.10) -> None:
        self.gamma      = gamma
        self.grad_clip  = grad_clip
        self.lambda_ent = lambda_ent
        self.epsilon    = 1.0

        rng = np.random.default_rng(42)
        def he(fan_in: int, *shape: int) -> np.ndarray:
            return rng.standard_normal(shape).astype(np.float64) * np.sqrt(2.0 / fan_in)

        # 2 heads: each has W_Q, b_Q, W_K, b_K, W_V, b_V
        # Shape: W_Q[h]: (D_CAND, D_HEAD) = (6, 8)
        self.W_Q = [he(D_CAND, D_CAND, D_HEAD) for _ in range(N_HEADS)]
        self.b_Q = [np.zeros(D_HEAD)            for _ in range(N_HEADS)]
        self.W_K = [he(D_CAND, D_CAND, D_HEAD) for _ in range(N_HEADS)]
        self.b_K = [np.zeros(D_HEAD)            for _ in range(N_HEADS)]
        self.W_V = [he(D_CAND, D_CAND, D_HEAD) for _ in range(N_HEADS)]
        self.b_V = [np.zeros(D_HEAD)            for _ in range(N_HEADS)]

        # Output projection: concat(16) → 16
        self.W_O = he(D_V_TOT, D_V_TOT, D_V_TOT)
        self.b_O = np.zeros(D_V_TOT)

        # MLP: 22 → 64 → 32 → 1
        self._W = [he(D_MLP_IN, D_MLP_IN, 64),
                   he(64,       64,        32),
                   he(32,       32,         1)]
        self._b = [np.zeros(64), np.zeros(32), np.zeros(1)]

        # Target network
        self._copy_target()
        self._opt = AdamOptimizer(lr=lr)

    def _copy_target(self) -> None:
        self._tW_Q = [w.copy() for w in self.W_Q]
        self._tb_Q = [b.copy() for b in self.b_Q]
        self._tW_K = [w.copy() for w in self.W_K]
        self._tb_K = [b.copy() for b in self.b_K]
        self._tW_V = [w.copy() for w in self.W_V]
        self._tb_V = [b.copy() for b in self.b_V]
        self._tW_O = self.W_O.copy(); self._tb_O = self.b_O.copy()
        self._tW   = [w.copy() for w in self._W]
        self._tb   = [b.copy() for b in self._b]

    def update_target(self) -> None:
        self._copy_target()

    def param_count(self) -> int:
        total = 0
        for h in range(N_HEADS):
            total += self.W_Q[h].size + self.b_Q[h].size
            total += self.W_K[h].size + self.b_K[h].size
            total += self.W_V[h].size + self.b_V[h].size
        total += self.W_O.size + self.b_O.size
        for W, b in zip(self._W, self._b):
            total += W.size + b.size
        return total

    def _build_competitor_data(
        self, states: np.ndarray, pids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return comp_encs (batch,4,6) and comp_valid (batch,4) bool."""
        batch    = states.shape[0]
        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        all_pids = np.tile(np.arange(N_PROCESSES), (batch, 1))
        mask     = all_pids != pids[:, None]
        comp_idx = all_pids[mask].reshape(batch, 4)
        bidx     = np.arange(batch)[:, None]
        comp_encs  = s3d[bidx, comp_idx]
        comp_valid = (comp_encs[:, :, 5] > 0.5)                    # arrived_flag at off+5
        return comp_encs, comp_valid

    def _head_forward(
        self,
        h: int,
        cand_enc:   np.ndarray,  # (batch, 6)
        comp_flat:  np.ndarray,  # (batch*4, 6)
        comp_valid: np.ndarray,  # (batch, 4) bool
        use_target: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Forward for one attention head. Returns context_h (batch, D_HEAD) and cache."""
        WQ = (self._tW_Q if use_target else self.W_Q)[h]
        bQ = (self._tb_Q if use_target else self.b_Q)[h]
        WK = (self._tW_K if use_target else self.W_K)[h]
        bK = (self._tb_K if use_target else self.b_K)[h]
        WV = (self._tW_V if use_target else self.W_V)[h]
        bV = (self._tb_V if use_target else self.b_V)[h]

        batch = cand_enc.shape[0]
        invalid = ~comp_valid

        q      = cand_enc @ WQ + bQ                                   # (batch, D_HEAD)
        K_flat = comp_flat @ WK + bK                                   # (batch*4, D_HEAD)
        V_flat = comp_flat @ WV + bV
        K = K_flat.reshape(batch, 4, D_HEAD)
        V = V_flat.reshape(batch, 4, D_HEAD)

        scores = np.einsum('bi,bji->bj', q, K) / np.sqrt(D_HEAD)      # (batch, 4)

        scores_masked = scores.copy()
        scores_masked[invalid] = -1e9
        scores_shifted = scores_masked - scores_masked.max(axis=1, keepdims=True)
        exp_s = np.exp(scores_shifted)
        exp_s[invalid] = 0.0
        denom   = exp_s.sum(axis=1, keepdims=True) + 1e-10
        weights = exp_s / denom                                         # (batch, 4)

        context_h = np.einsum('bj,bjd->bd', weights, V)               # (batch, D_HEAD)

        cache = dict(q=q, K=K, V=V, K_flat=K_flat, V_flat=V_flat,
                     weights=weights, invalid=invalid)
        return context_h, cache

    def _attention_forward(
        self,
        cand_enc:   np.ndarray,  # (batch, 6)
        comp_encs:  np.ndarray,  # (batch, 4, 6)
        comp_valid: np.ndarray,  # (batch, 4) bool
        use_target: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Multi-head forward. Returns context_out (batch, 16) and full cache."""
        WO = self._tW_O if use_target else self.W_O
        bO = self._tb_O if use_target else self.b_O

        batch     = cand_enc.shape[0]
        comp_flat = comp_encs.reshape(batch * 4, D_CAND)

        head_contexts = []
        head_caches   = []
        for h in range(N_HEADS):
            ctx_h, c_h = self._head_forward(h, cand_enc, comp_flat, comp_valid, use_target)
            head_contexts.append(ctx_h)
            head_caches.append(c_h)

        context_cat = np.concatenate(head_contexts, axis=1)           # (batch, 16)
        context_out = context_cat @ WO + bO                           # (batch, 16)

        cache = dict(
            cand_enc=cand_enc, comp_encs=comp_encs, comp_flat=comp_flat,
            comp_valid=comp_valid,
            head_caches=head_caches,
            context_cat=context_cat,
        )
        return context_out, cache

    def forward_batch(
        self,
        states:     np.ndarray,  # (batch, 30)
        actions:    np.ndarray,  # (batch,) int
        use_target: bool = False,
    ) -> np.ndarray:             # (batch, 1)
        batch    = states.shape[0]
        pids     = (actions // N_QT).astype(np.int32)
        qts      = QT_VALUES[actions % N_QT]

        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        cand_enc = s3d[np.arange(batch), pids]
        comp_encs, comp_valid = self._build_competitor_data(states, pids)
        context_out, _ = self._attention_forward(cand_enc, comp_encs, comp_valid, use_target)

        a6    = np.column_stack([cand_enc, qts / 2.0])                # (batch, 7)
        x_mlp = np.concatenate([context_out, a6], axis=1)            # (batch, 23)

        W, b = (self._tW, self._tb) if use_target else (self._W, self._b)
        z1 = x_mlp @ W[0] + b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1    @ W[1] + b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2    @ W[2] + b[2]
        return z3

    def select_action(
        self,
        state:         np.ndarray,
        epsilon:       float,
        valid_actions: list[int],
    ) -> int:
        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))
        states_b  = np.tile(state, (len(valid_actions), 1)).astype(np.float64)
        actions_b = np.array(valid_actions, dtype=np.int32)
        q_vals    = self.forward_batch(states_b, actions_b).flatten()
        return valid_actions[int(np.argmax(q_vals))]

    def update_online(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        rewards:     np.ndarray,
        next_states: np.ndarray,
        dones:       np.ndarray,
    ) -> tuple[float, float]:
        batch = states.shape[0]
        pids  = (actions // N_QT).astype(np.int32)
        qts   = QT_VALUES[actions % N_QT]

        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        cand_enc = s3d[np.arange(batch), pids]
        comp_encs, comp_valid = self._build_competitor_data(states, pids)

        # Forward (online)
        context_out, attn_cache = self._attention_forward(cand_enc, comp_encs, comp_valid)
        a6    = np.column_stack([cand_enc, qts / 2.0])
        x_mlp = np.concatenate([context_out, a6], axis=1)

        z1 = x_mlp @ self._W[0] + self._b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1    @ self._W[1] + self._b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2    @ self._W[2] + self._b[2]
        q_pred = z3.flatten()

        # Entropy: average across heads
        valid_f = comp_valid.astype(np.float64)
        H_heads = []
        for h in range(N_HEADS):
            w_h = attn_cache['head_caches'][h]['weights']
            ent_h = -(w_h * np.log(w_h + 1e-8) * valid_f).sum(axis=1)
            H_heads.append(float(np.mean(ent_h)))
        H_batch = float(np.mean(H_heads))

        # Target Q
        ns          = np.asarray(next_states, dtype=np.float64)
        all_acts    = np.tile(np.arange(N_ACTIONS), batch)
        all_ns      = np.repeat(ns, N_ACTIONS, axis=0)
        Q_next_flat = self.forward_batch(all_ns, all_acts, use_target=True).flatten()
        Q_next_mat  = Q_next_flat.reshape(batch, N_ACTIONS)

        valid_mask = np.zeros((batch, N_ACTIONS), dtype=bool)
        for pid in range(N_PROCESSES):
            runnable = (ns[:, pid * D_CAND + 5] > 0.5)              # arrived_flag at off+5
            for qt in range(N_QT):
                valid_mask[:, pid * N_QT + qt] = runnable

        Q_next_mat[~valid_mask] = -np.inf
        all_invalid = ~np.any(valid_mask, axis=1)
        max_q_next  = np.where(all_invalid, 0.0, Q_next_mat.max(axis=1))
        targets = rewards + self.gamma * max_q_next * (1.0 - dones)

        # Loss
        delta = q_pred - targets
        L_td  = float(np.mean(delta ** 2))
        loss  = L_td - self.lambda_ent * H_batch

        # ---- Backward ----

        # MLP backward
        dz3    = (2.0 * delta / batch).reshape(-1, 1)
        dW2    = h2.T @ dz3;     db2 = dz3.sum(axis=0)
        dh2    = dz3 @ self._W[2].T
        dz2    = dh2 * (h2 > 0).astype(np.float64)
        dW1    = h1.T @ dz2;     db1 = dz2.sum(axis=0)
        dh1    = dz2 @ self._W[1].T
        dz1    = dh1 * (h1 > 0).astype(np.float64)
        dW0    = x_mlp.T @ dz1;  db0 = dz1.sum(axis=0)
        dx_mlp = dz1 @ self._W[0].T                                   # (batch, 23)

        d_context_out = dx_mlp[:, :D_V_TOT]                          # (batch, 16)

        # W_O backward
        context_cat = attn_cache['context_cat']
        comp_flat   = attn_cache['comp_flat']
        dW_O = context_cat.T @ d_context_out                         # (16, 16)
        db_O = d_context_out.sum(axis=0)
        d_context_cat = d_context_out @ self.W_O.T                   # (batch, 16)

        # Split gradient to heads
        d_ctx_heads = np.split(d_context_cat, N_HEADS, axis=1)       # each (batch, 8)

        # Per-head backward
        head_grad_lists = []  # list of [dWQ, dbQ, dWK, dbK, dWV, dbV] per head
        for h in range(N_HEADS):
            hc       = attn_cache['head_caches'][h]
            q_h      = hc['q']
            K_h      = hc['K']
            V_h      = hc['V']
            weights_h = hc['weights']
            invalid_h = hc['invalid']
            d_ctx_h  = d_ctx_heads[h]                                 # (batch, D_HEAD)

            d_weights_h = np.einsum('bd,bjd->bj', d_ctx_h, V_h)     # (batch, 4)
            dV_h        = weights_h[:, :, None] * d_ctx_h[:, None, :] # (batch, 4, D_HEAD)
            dV_h[invalid_h] = 0.0

            if self.lambda_ent > 0:
                d_ent_h   = (self.lambda_ent / (batch * N_HEADS)) * (
                    np.log(weights_h + 1e-8) + weights_h / (weights_h + 1e-8)
                ) * valid_f
                d_weights_h = d_weights_h + d_ent_h

            wdotdw            = (weights_h * d_weights_h).sum(axis=1, keepdims=True)
            d_scores_masked_h = weights_h * (d_weights_h - wdotdw)
            d_scores_masked_h[invalid_h] = 0.0
            d_raw_scores_h    = d_scores_masked_h / np.sqrt(D_HEAD)

            dq_h    = np.einsum('bj,bji->bi', d_raw_scores_h, K_h)   # (batch, D_HEAD)
            dK_h    = d_raw_scores_h[:, :, None] * q_h[:, None, :]   # (batch, 4, D_HEAD)
            dK_h[invalid_h] = 0.0

            dW_Qh   = cand_enc.T @ dq_h                               # (6, D_HEAD)
            db_Qh   = dq_h.sum(axis=0)
            dK_flat = dK_h.reshape(batch * 4, D_HEAD)
            dW_Kh   = comp_flat.T @ dK_flat                           # (6, D_HEAD)
            db_Kh   = dK_flat.sum(axis=0)
            dV_flat = dV_h.reshape(batch * 4, D_HEAD)
            dW_Vh   = comp_flat.T @ dV_flat                           # (6, D_HEAD)
            db_Vh   = dV_flat.sum(axis=0)

            head_grad_lists.append([dW_Qh, db_Qh, dW_Kh, db_Kh, dW_Vh, db_Vh])

        # Assemble all grads in same order as all_params below
        all_grads = [
            dW0, db0, dW1, db1, dW2, db2,
        ]
        for h in range(N_HEADS):
            dWQ, dbQ, dWK, dbK, dWV, dbV = head_grad_lists[h]
            all_grads.extend([dWQ, dbQ, dWK, dbK, dWV, dbV])
        all_grads.extend([dW_O, db_O])

        # Grad norm clip
        global_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in all_grads)))
        if global_norm > self.grad_clip:
            scale     = self.grad_clip / global_norm
            all_grads = [g * scale for g in all_grads]

        all_params = [
            self._W[0], self._b[0], self._W[1], self._b[1], self._W[2], self._b[2],
        ]
        for h in range(N_HEADS):
            all_params.extend([
                self.W_Q[h], self.b_Q[h],
                self.W_K[h], self.b_K[h],
                self.W_V[h], self.b_V[h],
            ])
        all_params.extend([self.W_O, self.b_O])

        self._opt.step(all_params, all_grads)
        return loss, H_batch

    def decay_epsilon(self, ep: int, min_eps: float = 0.05, decay: float = 0.9995) -> float:
        self.epsilon = max(min_eps, 1.0 * (decay ** ep))
        return self.epsilon

    def save(self, path: str) -> None:
        kw: dict = {}
        for h in range(N_HEADS):
            kw[f'W_Q{h}'] = self.W_Q[h]; kw[f'b_Q{h}'] = self.b_Q[h]
            kw[f'W_K{h}'] = self.W_K[h]; kw[f'b_K{h}'] = self.b_K[h]
            kw[f'W_V{h}'] = self.W_V[h]; kw[f'b_V{h}'] = self.b_V[h]
        kw['W_O'] = self.W_O; kw['b_O'] = self.b_O
        for i, (W, b) in enumerate(zip(self._W, self._b)):
            kw[f'W{i}'] = W; kw[f'b{i}'] = b
        np.savez(path, **kw)

    def load(self, path: str) -> None:
        d = np.load(path)
        self.W_Q = [d[f'W_Q{h}'] for h in range(N_HEADS)]
        self.b_Q = [d[f'b_Q{h}'] for h in range(N_HEADS)]
        self.W_K = [d[f'W_K{h}'] for h in range(N_HEADS)]
        self.b_K = [d[f'b_K{h}'] for h in range(N_HEADS)]
        self.W_V = [d[f'W_V{h}'] for h in range(N_HEADS)]
        self.b_V = [d[f'b_V{h}'] for h in range(N_HEADS)]
        self.W_O = d['W_O']; self.b_O = d['b_O']
        self._W  = [d[f'W{i}'] for i in range(3)]
        self._b  = [d[f'b{i}'] for i in range(3)]
        self.update_target()


# ---------------------------------------------------------------------------
# Permutation invariance unit test
# ---------------------------------------------------------------------------

def run_perm_test() -> None:
    print("=" * 60)
    print("Permutation Invariance Unit Test")
    print("=" * 60)
    agent = AttentionDQN10C()

    # State slots: [burst_norm, wait, last_exec, urgency, time_in_queue, arrived_flag]
    # Build state A: candidate=P0, P1 competitor, P2 competitor; P3,P4 zeros (not active)
    sv_A = np.zeros(N_PROCESSES * D_CAND, dtype=np.float64)
    sv_A[0*D_CAND:1*D_CAND] = [0.30, 0.10, 0.0, 0.0, 0.20, 1.0]
    sv_A[1*D_CAND:2*D_CAND] = [0.60, 0.20, 0.0, 0.0, 0.30, 1.0]
    sv_A[2*D_CAND:3*D_CAND] = [0.75, 0.30, 0.0, 0.0, 0.40, 1.0]
    # P3, P4: zeros (not active)
    action_A = 0 * N_QT + 1   # candidate=P0, tier1

    # Build state B: candidate=P4 (same features as P0 in A)
    # P1 gets P2's original features, P2 gets P1's original features (swapped)
    sv_B = np.zeros(N_PROCESSES * D_CAND, dtype=np.float64)
    # P0: zeros (not active)
    sv_B[1*D_CAND:2*D_CAND] = [0.75, 0.30, 0.0, 0.0, 0.40, 1.0]
    sv_B[2*D_CAND:3*D_CAND] = [0.60, 0.20, 0.0, 0.0, 0.30, 1.0]
    # P3: zeros
    sv_B[4*D_CAND:5*D_CAND] = [0.30, 0.10, 0.0, 0.0, 0.20, 1.0]
    action_B = 4 * N_QT + 1   # candidate=P4, tier1

    Q_A = float(agent.forward_batch(sv_A[None], np.array([action_A]))[0, 0])
    Q_B = float(agent.forward_batch(sv_B[None], np.array([action_B]))[0, 0])
    diff = abs(Q_A - Q_B)
    result = "PASS" if diff < 1e-10 else "FAIL"
    print(f"  Q(state_A, P0/tier1) = {Q_A:.10f}")
    print(f"  Q(state_B, P4/tier1) = {Q_B:.10f}")
    print(f"  |Q_A - Q_B|          = {diff:.2e}")
    print(f"  Result: {result}")

    # Verify sensitivity: same candidate but different competitor burst
    sv_C = sv_A.copy()
    sv_C[1*D_CAND:2*D_CAND] = [0.90, 0.20, 0.0, 0.0, 0.30, 1.0]
    Q_C  = float(agent.forward_batch(sv_C[None], np.array([action_A]))[0, 0])
    diff2 = abs(Q_A - Q_C)
    result2 = "PASS" if diff2 > 1e-6 else "FAIL"
    print(f"\n  Sensitivity check (competitor burst 10→20): |Delta| = {diff2:.6f}")
    print(f"  Result: {result2}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _make_random_tasks(rng: np.random.Generator) -> list[dict]:
    """Fallback: generate 5 random tasks when trace file is unavailable."""
    from w9_train import BURST_P95_FILT
    _ARRIVE_SLOTS = (0, 2, 5, 8, 10)
    order = rng.permutation(N_PROCESSES)
    bursts = rng.uniform(1.0, BURST_P95_FILT, size=N_PROCESSES)
    cpus   = rng.uniform(50.0, 800.0, size=N_PROCESSES)
    mems   = rng.uniform(0.05, 0.59,  size=N_PROCESSES)
    return [
        {
            "burst_ms":   float(bursts[i]),
            "arrival_ms": float(_ARRIVE_SLOTS[order[k]]),
            "plan_cpu":   float(cpus[i]),
            "plan_mem":   float(mems[i]),
        }
        for k, i in enumerate(range(N_PROCESSES))
    ]


def train() -> AttentionDQN10C:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load trace or fall back to random generation
    use_trace = os.path.isfile(TRACE_PATH)
    if use_trace:
        print(f"Loading trace: {TRACE_PATH}")
        sampler = TraceEpisodeSampler5(TRACE_PATH)
        print()
    else:
        sampler = None
        print(f"Trace not found at {TRACE_PATH} — using random process generation.\n")

    agent = AttentionDQN10C(lr=LR, gamma=GAMMA, grad_clip=GRAD_CLIP)

    # Parameter count
    n_head  = N_HEADS * 3 * (D_CAND * D_HEAD + D_HEAD)
    n_wo    = D_V_TOT * D_V_TOT + D_V_TOT
    n_mlp   = (D_MLP_IN*64+64) + (64*32+32) + (32*1+1)
    n_total = n_head + n_wo + n_mlp
    actual  = agent.param_count()
    print(f"Parameter count: {actual} total  (head_proj={n_head}, W_O={n_wo}, mlp={n_mlp})")
    print(f"  [Formula check: {n_head} + {n_wo} + {n_mlp} = {n_total}]")
    print()

    buffer = ReplayBuffer(capacity=BUF_CAPACITY, state_dim=N_PROCESSES * D_CAND)

    rng = np.random.default_rng(42)
    total_transitions = 0
    win_loss:      list[float] = []
    win_entropy:   list[float] = []
    win_mct:       list[float] = []
    win_reward:    list[float] = []
    win_starve:    list[int]   = []
    win_srpt:      list[float] = []

    with open(LOG_PATH, "w", newline="") as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow([
            "episode", "avg_loss", "avg_H", "avg_MCT",
            "ep_reward", "starved", "srpt_agree", "lambda_ent",
        ])

        for ep in range(1, N_EPISODES + 1):
            agent.lambda_ent = LAMBDA_START - (LAMBDA_START - LAMBDA_END) * (ep / N_EPISODES)

            if sampler is not None:
                tasks = sampler.sample_episode(rng)
            else:
                tasks = _make_random_tasks(rng)

            procs = _make_procs(tasks)
            env   = SchedEnv(procs)
            env.reset()
            sv    = _encode_state(env, tasks)

            ep_loss_sum   = 0.0
            ep_ent_sum    = 0.0
            ep_loss_n     = 0
            ep_reward_sum = 0.0
            ep_step_n     = 0
            srpt_agree_n  = 0
            srpt_total_n  = 0
            done          = False

            while not done:
                valid = _valid_actions(env)

                # --- SRPT oracle: pid with minimum remaining burst among runnable ---
                runnable_procs = [
                    p for p in env.processes
                    if p.arrival_time <= env.current_time and not p.is_complete
                ]
                if runnable_procs:
                    srpt_pid = min(runnable_procs, key=lambda p: p.remaining_burst).pid
                else:
                    srpt_pid = -1

                if total_transitions < WARMUP:
                    action = random.choice(valid)
                else:
                    action = agent.select_action(sv, agent.epsilon, valid)

                chosen_proc = env.processes[action // N_QT]
                _, _, done, info = env.step(action)
                reward  = info.get("env_reward", 0.0) / REWARD_SCALE

                # CT bonus (w1=0.6): penalize completion time, scaled to ~[-1, 0]
                if chosen_proc.is_complete:
                    reward += -0.6 * chosen_proc.completion_time / 60.0

                sv_next = _encode_state(env, tasks)

                # Track SRPT agreement
                if srpt_pid >= 0:
                    chosen_pid = action // N_QT
                    srpt_agree_n += int(chosen_pid == srpt_pid)
                    srpt_total_n += 1

                ep_reward_sum += reward
                ep_step_n     += 1

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

            # Checkpoint
            if ep in CHECKPOINT_AT:
                ckpt_path = os.path.join(RESULTS_DIR, f"w11d_ep{ep}.npz")
                agent.save(ckpt_path)
                print(f"  [Checkpoint saved → {ckpt_path}]")

            mct       = info.get("mean_completion_time_so_far") or 0.0
            mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else float("nan")
            mean_ent  = ep_ent_sum  / ep_loss_n if ep_loss_n > 0 else float("nan")
            ep_reward_mean = ep_reward_sum / ep_step_n if ep_step_n > 0 else 0.0
            srpt_frac = srpt_agree_n / srpt_total_n if srpt_total_n > 0 else 0.0
            lam_cur   = agent.lambda_ent

            # Starvation: check if any process had slowdown > 3x group median
            completed = [p for p in env.processes if p.is_complete]
            starved_ep = 0
            if completed:
                turnarounds = [p.completion_time - p.arrival_time for p in completed]
                bursts_ep   = [p.burst_length for p in completed]
                slowdowns   = [t / max(b, 1e-6) for t, b in zip(turnarounds, bursts_ep)]
                med_slow    = float(np.median(slowdowns))
                if any(s > 3.0 * med_slow for s in slowdowns):
                    starved_ep = 1

            agent.decay_epsilon(ep, min_eps=0.05, decay=0.9995)
            win_loss.append(mean_loss)
            win_entropy.append(mean_ent)
            win_mct.append(mct)
            win_reward.append(ep_reward_mean)
            win_starve.append(starved_ep)
            win_srpt.append(srpt_frac)

            log_writer.writerow([
                ep, f"{mean_loss:.6f}", f"{mean_ent:.6f}", f"{mct:.4f}",
                f"{ep_reward_mean:.6f}", starved_ep, f"{srpt_frac:.4f}",
                f"{lam_cur:.5f}",
            ])

            if ep in PRINT_AT:
                n   = min(ep, 100)
                al  = float(np.nanmean(win_loss[-n:]))
                ah  = float(np.nanmean(win_entropy[-n:]))
                am  = float(np.mean(win_mct[-n:]))
                ar  = float(np.mean(win_reward[-n:]))
                sc  = int(np.sum(win_starve[-n:]))
                sp  = float(np.mean(win_srpt[-n:])) * 100.0
                print(
                    f"ep {ep:>6} | MCT={am:.2f}s | reward={ar:.4f} | "
                    f"starve={sc}/100 | SRPT={sp:.1f}% | "
                    f"loss={al:.4f} | H={ah:.4f} | lam={lam_cur:.5f}"
                )
                sys.stdout.flush(); log_f.flush()

                if al > LOSS_GATE:
                    print(f"\nSTOP GATE: avg_loss={al:.2f} > {LOSS_GATE} at ep {ep}")
                    return agent
                if ep == 5000 and ah > ENT_GATE_5K:
                    print(f"\nSTOP GATE: avg_H={ah:.4f} > {ENT_GATE_5K} at ep 5000")
                    return agent

    print(f"\nTraining complete — {N_EPISODES} episodes.")
    return agent


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    print("=" * 64)
    print("Week 11d — W11d: 6-dim state (true burst + fairness), CT w1=0.6 + fairness w2-5=0.1")
    print("=" * 64)

    run_perm_test()
    print()

    t0    = time.time()
    agent = train()
    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
    agent.save(WEIGHTS_PATH)
    print(f"Weights saved → {WEIGHTS_PATH}")
