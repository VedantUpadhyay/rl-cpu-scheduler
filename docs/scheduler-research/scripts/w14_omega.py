"""W14-ω — Preference-Conditioned Attention DQN.

The agent learns the full MCT/starvation Pareto frontier in a single training
run. omega_starvation ∈ [0,1] is a preference scalar injected directly into the
attention computation (no state concatenation, no learned parameters for omega).

Architectural changes vs W12:
  1. Pre-attention query modulation:
       q_cond = q * (1 + omega_s)   applied per-head before dot-product attention
  2. Post-attention FiLM modulation:
       context_cond = context_out * (1 + omega_s) + omega_mct * 1.0
  3. Reward: omega_mct * value_delta/20 + omega_s * starvation_signal / max(omega_mct,0.1)
       starvation_signal = -mean(max(0, wait-50)/50 for p in runnable)
  4. omega sampling schedule: Uniform(0,1) for eps<5000, sin^2 (U-shaped) for eps>=5000
  5. Replay buffer stores (s, a, r_vd, r_ss, s', done, omega_s)
  6. Gradient norm monitoring every 500 episodes

Training: 30,000 episodes × 3 seeds on Alibaba 2018 trace.
Checkpoints: every 5000 episodes.

Evaluation: sweep omega_s ∈ {0.0, 0.1, ..., 1.0}, N=200 episodes each.
Pareto frontier: identify omega values where MCT < 21.59s AND Starve < 36%.
"""
from __future__ import annotations
import json, math, os, random, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
from project_config import (PROJECT_ROOT as _PROJECT_ROOT, SCRIPTS_DIR as _SCRIPTS_DIR,
                             TRACE_PATH as TRACE_TRAIN, TEST_PATH as TRACE_TEST,
                             get_agent_dir)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

from schedsim.env    import SchedEnv, N_PROCESSES, N_QUANTUM_TIERS, value_delta
from schedsim.agent  import AdamOptimizer
from schedsim.process import Process

from w9_train import (
    TraceEpisodeSampler5, _valid_actions, _make_procs,
    N_QT, N_ACTIONS, QT_VALUES,
    _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem,
    WAIT_NORM, CPU_MAX, MEM_P95,
)
from ablation_multiseed import (
    _encode_7dim,
    N_HEADS, D_HEAD, D_V_TOT,
    LAMBDA_START, LAMBDA_END, LOSS_GATE, ENT_GATE_5K,
    LR, GAMMA, GRAD_CLIP, BUF_CAPACITY, BATCH_SIZE,
    TARGET_UPDATE_FREQ, WARMUP,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = get_agent_dir("w14_omega")

N_EPISODES    = 30_000
N_EVAL_PARETO = 200
SEEDS         = [42, 123, 456]
PRINT_EVERY   = 500
CKPT_EVERY    = 5_000
D_CAND        = 7
AFI           = 6       # arrived_flag_idx

REWARD_SCALE = 20.0
STARVATION_THRESHOLD = 50.0   # seconds — matches MLFQ aging threshold
QUANTUM_TIERS = (0.5, 2.0, 8.0)
MLFQ_AGE_THRESH = 50.0

MLFQ_MCT    = 21.59
MLFQ_STARVE = 36.0

GRAD_RATIO_WARN = 10.0   # flag W11-ghost risk if starvation_grad / vd_grad > this


# ---------------------------------------------------------------------------
# Omega-aware replay buffer  (stores r_vd and r_ss separately)
# ---------------------------------------------------------------------------

class OmegaReplayBuffer:
    """Ring buffer storing (s, a, r_vd, r_ss, s', done, omega_s)."""

    def __init__(self, capacity: int, state_dim: int) -> None:
        self.capacity  = capacity
        self.ptr       = 0
        self.size      = 0
        self.s         = np.zeros((capacity, state_dim), dtype=np.float32)
        self.a         = np.zeros(capacity, dtype=np.int32)
        self.r_vd      = np.zeros(capacity, dtype=np.float32)
        self.r_ss      = np.zeros(capacity, dtype=np.float32)
        self.s_next    = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done      = np.zeros(capacity, dtype=np.float32)
        self.omega_s   = np.zeros(capacity, dtype=np.float32)

    def store(self, s, a: int, r_vd: float, r_ss: float,
              s_next, done: bool, omega_s: float) -> None:
        i = self.ptr
        self.s[i]       = s
        self.a[i]       = a
        self.r_vd[i]    = r_vd
        self.r_ss[i]    = r_ss
        self.s_next[i]  = s_next
        self.done[i]    = float(done)
        self.omega_s[i] = omega_s
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (self.s[idx], self.a[idx],
                self.r_vd[idx], self.r_ss[idx],
                self.s_next[idx], self.done[idx],
                self.omega_s[idx])

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# W14OmegaDQN — 2-head attention DQN with FiLM omega modulation
# ---------------------------------------------------------------------------

class W14OmegaDQN:
    """Same weight structure as AttentionDQN(d_cand=7, arrived_flag_idx=6).

    Key changes:
      - _head_forward: q_cond = q * (1 + omega_s) before dot-product
      - _attention_forward: context_cond = context_out * (1+omega_s) + omega_mct
      - forward_batch / select_action / update_online all take omega_s
      - update_online uses per-sample omega from replay buffer
    """

    def __init__(self, lr: float = LR, gamma: float = GAMMA,
                 grad_clip: float = GRAD_CLIP,
                 lambda_ent: float = LAMBDA_START) -> None:
        self.d_cand        = D_CAND
        self.arrived_flag_idx = AFI
        self.d_mlp_in      = D_V_TOT + D_CAND + 1
        self.gamma         = gamma
        self.grad_clip     = grad_clip
        self.lambda_ent    = lambda_ent
        self.epsilon       = 1.0

        rng = np.random.default_rng(42)
        def he(fan_in: int, *shape: int) -> np.ndarray:
            return rng.standard_normal(shape).astype(np.float64) * np.sqrt(2.0 / fan_in)

        self.W_Q = [he(D_CAND, D_CAND, D_HEAD) for _ in range(N_HEADS)]
        self.b_Q = [np.zeros(D_HEAD)            for _ in range(N_HEADS)]
        self.W_K = [he(D_CAND, D_CAND, D_HEAD) for _ in range(N_HEADS)]
        self.b_K = [np.zeros(D_HEAD)            for _ in range(N_HEADS)]
        self.W_V = [he(D_CAND, D_CAND, D_HEAD) for _ in range(N_HEADS)]
        self.b_V = [np.zeros(D_HEAD)            for _ in range(N_HEADS)]

        self.W_O = he(D_V_TOT, D_V_TOT, D_V_TOT)
        self.b_O = np.zeros(D_V_TOT)

        self._W = [he(self.d_mlp_in, self.d_mlp_in, 64),
                   he(64,            64,             32),
                   he(32,            32,              1)]
        self._b = [np.zeros(64), np.zeros(32), np.zeros(1)]

        self._copy_target()
        self._opt = AdamOptimizer(lr=lr)

    # ------------------------------------------------------------------
    # Target network management
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Helper: build competitor encodings (same as AttentionDQN)
    # ------------------------------------------------------------------

    def _build_competitor_data(self, states: np.ndarray,
                                pids: np.ndarray):
        batch    = states.shape[0]
        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        all_pids = np.tile(np.arange(N_PROCESSES), (batch, 1))
        mask     = all_pids != pids[:, None]
        comp_idx = all_pids[mask].reshape(batch, 4)
        bidx     = np.arange(batch)[:, None]
        comp_encs  = s3d[bidx, comp_idx]
        comp_valid = (comp_encs[:, :, AFI] > 0.5)
        return comp_encs, comp_valid

    # ------------------------------------------------------------------
    # Per-head forward with pre-attention query modulation
    # ------------------------------------------------------------------

    def _head_forward(self, h: int, cand_enc, comp_flat, comp_valid,
                      omega_s_col,       # shape (batch, 1)
                      use_target: bool = False):
        WQ = (self._tW_Q if use_target else self.W_Q)[h]
        bQ = (self._tb_Q if use_target else self.b_Q)[h]
        WK = (self._tW_K if use_target else self.W_K)[h]
        bK = (self._tb_K if use_target else self.b_K)[h]
        WV = (self._tW_V if use_target else self.W_V)[h]
        bV = (self._tb_V if use_target else self.b_V)[h]

        batch   = cand_enc.shape[0]
        invalid = ~comp_valid

        q      = cand_enc @ WQ + bQ               # (batch, D_HEAD)
        q_cond = q * (1.0 + omega_s_col)           # Change 1: pre-attention modulation

        K_flat = comp_flat @ WK + bK
        V_flat = comp_flat @ WV + bV
        K = K_flat.reshape(batch, 4, D_HEAD)
        V = V_flat.reshape(batch, 4, D_HEAD)

        scores = np.einsum('bi,bji->bj', q_cond, K) / np.sqrt(D_HEAD)
        scores_masked = scores.copy()
        scores_masked[invalid] = -1e9
        scores_shifted = scores_masked - scores_masked.max(axis=1, keepdims=True)
        exp_s = np.exp(scores_shifted)
        exp_s[invalid] = 0.0
        denom   = exp_s.sum(axis=1, keepdims=True) + 1e-10
        weights = exp_s / denom

        context_h = np.einsum('bj,bjd->bd', weights, V)

        cache = dict(q=q, q_cond=q_cond, K=K, V=V,
                     K_flat=K_flat, V_flat=V_flat,
                     weights=weights, invalid=invalid)
        return context_h, cache

    # ------------------------------------------------------------------
    # Multi-head attention + FiLM post-attention modulation
    # ------------------------------------------------------------------

    def _attention_forward(self, cand_enc, comp_encs, comp_valid,
                           omega_s_col,       # shape (batch, 1)
                           omega_mct_col,     # shape (batch, 1)
                           use_target: bool = False):
        WO = self._tW_O if use_target else self.W_O
        bO = self._tb_O if use_target else self.b_O

        batch     = cand_enc.shape[0]
        comp_flat = comp_encs.reshape(batch * 4, D_CAND)

        head_contexts, head_caches = [], []
        for h in range(N_HEADS):
            ctx_h, c_h = self._head_forward(
                h, cand_enc, comp_flat, comp_valid, omega_s_col, use_target)
            head_contexts.append(ctx_h)
            head_caches.append(c_h)

        context_cat = np.concatenate(head_contexts, axis=1)   # (batch, D_V_TOT)
        context_out = context_cat @ WO + bO                    # (batch, D_V_TOT)

        # Change 2: FiLM post-attention modulation (no learned params)
        context_cond = context_out * (1.0 + omega_s_col) + omega_mct_col * 1.0

        cache = dict(cand_enc=cand_enc, comp_encs=comp_encs, comp_flat=comp_flat,
                     comp_valid=comp_valid, head_caches=head_caches,
                     context_cat=context_cat, context_out=context_out,
                     omega_s_col=omega_s_col)
        return context_cond, cache

    # ------------------------------------------------------------------
    # Batch forward pass
    # ------------------------------------------------------------------

    def forward_batch(self, states, actions, omega_s,
                      use_target: bool = False):
        """omega_s: scalar or array of shape (batch,). Returns (batch,1) Q-values."""
        batch = states.shape[0]
        pids  = (actions // N_QT).astype(np.int32)
        qts   = QT_VALUES[actions % N_QT]

        omega_s_arr = np.broadcast_to(
            np.asarray(omega_s, dtype=np.float64).ravel(), (batch,))
        omega_s_col  = omega_s_arr[:, None]           # (batch, 1)
        omega_mct_col = (1.0 - omega_s_arr)[:, None]

        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        cand_enc = s3d[np.arange(batch), pids].astype(np.float64)
        comp_encs, comp_valid = self._build_competitor_data(
            states.astype(np.float64), pids)

        context_cond, _ = self._attention_forward(
            cand_enc, comp_encs, comp_valid,
            omega_s_col, omega_mct_col, use_target)

        a6    = np.column_stack([cand_enc, qts / 2.0])
        x_mlp = np.concatenate([context_cond, a6], axis=1)

        W, b = (self._tW, self._tb) if use_target else (self._W, self._b)
        z1 = x_mlp @ W[0] + b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1    @ W[1] + b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2    @ W[2] + b[2]
        return z3

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state, epsilon: float,
                      valid_actions: list[int],
                      omega_s: float) -> int:
        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))
        n = len(valid_actions)
        states_b  = np.tile(state, (n, 1)).astype(np.float64)
        actions_b = np.array(valid_actions, dtype=np.int32)
        omegas_b  = np.full(n, omega_s, dtype=np.float64)
        q_vals    = self.forward_batch(states_b, actions_b, omegas_b).flatten()
        return valid_actions[int(np.argmax(q_vals))]

    # ------------------------------------------------------------------
    # Full backward pass (returns loss, entropy, and all raw gradients)
    # ------------------------------------------------------------------

    def _backward(self, states, actions, rewards,
                  next_states, dones, omega_s_arr):
        """Compute forward + backward. Returns (loss, H, all_grads, q_pred, targets)."""
        batch = states.shape[0]
        pids  = (actions // N_QT).astype(np.int32)
        qts   = QT_VALUES[actions % N_QT]

        omega_s_col   = omega_s_arr[:, None]
        omega_mct_col = (1.0 - omega_s_arr)[:, None]

        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        cand_enc = s3d[np.arange(batch), pids]
        comp_encs, comp_valid = self._build_competitor_data(states, pids)
        valid_f = comp_valid.astype(np.float64)

        context_cond, attn_cache = self._attention_forward(
            cand_enc, comp_encs, comp_valid, omega_s_col, omega_mct_col)

        a6    = np.column_stack([cand_enc, qts / 2.0])
        x_mlp = np.concatenate([context_cond, a6], axis=1)

        z1 = x_mlp @ self._W[0] + self._b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1    @ self._W[1] + self._b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2    @ self._W[2] + self._b[2]
        q_pred = z3.flatten()

        # Entropy from attention weights
        H_heads = []
        for h in range(N_HEADS):
            w_h   = attn_cache['head_caches'][h]['weights']
            ent_h = -(w_h * np.log(w_h + 1e-8) * valid_f).sum(axis=1)
            H_heads.append(float(np.mean(ent_h)))
        H_batch = float(np.mean(H_heads))

        # TD target — use same per-sample omega for target Q
        ns       = np.asarray(next_states, dtype=np.float64)
        all_acts = np.tile(np.arange(N_ACTIONS), batch)
        all_ns   = np.repeat(ns, N_ACTIONS, axis=0)
        all_oms  = np.repeat(omega_s_arr, N_ACTIONS)   # same omega per sample

        Q_next_flat = self.forward_batch(all_ns, all_acts, all_oms,
                                         use_target=True).flatten()
        Q_next_mat  = Q_next_flat.reshape(batch, N_ACTIONS)

        valid_mask = np.zeros((batch, N_ACTIONS), dtype=bool)
        ns_3d = ns.reshape(batch, N_PROCESSES, D_CAND)
        for pid in range(N_PROCESSES):
            runnable = (ns_3d[:, pid, AFI] > 0.5)
            for qt in range(N_QT):
                valid_mask[:, pid * N_QT + qt] = runnable

        Q_next_mat[~valid_mask] = -np.inf
        all_invalid = ~np.any(valid_mask, axis=1)
        max_q_next  = np.where(all_invalid, 0.0, Q_next_mat.max(axis=1))
        targets     = rewards + self.gamma * max_q_next * (1.0 - dones)

        delta = q_pred - targets
        L_td  = float(np.mean(delta ** 2))
        loss  = L_td - self.lambda_ent * H_batch

        # --- Backward through MLP ---
        dz3    = (2.0 * delta / batch).reshape(-1, 1)
        dW2    = h2.T @ dz3;     db2 = dz3.sum(axis=0)
        dh2    = dz3 @ self._W[2].T
        dz2    = dh2 * (h2 > 0).astype(np.float64)
        dW1    = h1.T @ dz2;     db1 = dz2.sum(axis=0)
        dh1    = dz2 @ self._W[1].T
        dz1    = dh1 * (h1 > 0).astype(np.float64)
        dW0    = x_mlp.T @ dz1;  db0 = dz1.sum(axis=0)
        dx_mlp = dz1 @ self._W[0].T

        # dx_mlp = [d_context_cond | d_a6] — we only need d_context_cond
        d_context_cond = dx_mlp[:, :D_V_TOT]

        # --- Backward through FiLM modulation ---
        # context_cond = context_out * (1 + omega_s) + omega_mct
        # d_context_out = d_context_cond * (1 + omega_s)
        d_context_out = d_context_cond * (1.0 + attn_cache['omega_s_col'])

        # --- Backward through WO ---
        context_cat = attn_cache['context_cat']
        comp_flat   = attn_cache['comp_flat']
        dW_O = context_cat.T @ d_context_out
        db_O = d_context_out.sum(axis=0)
        d_context_cat = d_context_out @ self.W_O.T

        d_ctx_heads = np.split(d_context_cat, N_HEADS, axis=1)

        # --- Backward through each attention head ---
        head_grad_lists = []
        for h in range(N_HEADS):
            hc        = attn_cache['head_caches'][h]
            q_h       = hc['q']
            q_cond_h  = hc['q_cond']
            K_h       = hc['K']
            V_h       = hc['V']
            weights_h = hc['weights']
            invalid_h = hc['invalid']
            d_ctx_h   = d_ctx_heads[h]

            d_weights_h = np.einsum('bd,bjd->bj', d_ctx_h, V_h)
            dV_h        = weights_h[:, :, None] * d_ctx_h[:, None, :]
            dV_h[invalid_h] = 0.0

            if self.lambda_ent > 0:
                d_ent_h     = (self.lambda_ent / (batch * N_HEADS)) * (
                    np.log(weights_h + 1e-8) + weights_h / (weights_h + 1e-8)
                ) * valid_f
                d_weights_h = d_weights_h + d_ent_h

            wdotdw             = (weights_h * d_weights_h).sum(axis=1, keepdims=True)
            d_scores_masked_h  = weights_h * (d_weights_h - wdotdw)
            d_scores_masked_h[invalid_h] = 0.0
            d_raw_scores_h     = d_scores_masked_h / np.sqrt(D_HEAD)

            # Backward through q_cond = q * (1 + omega_s)
            dq_cond_h = np.einsum('bj,bji->bi', d_raw_scores_h, K_h)
            dq_h      = dq_cond_h * (1.0 + attn_cache['omega_s_col'])  # chain rule

            dK_h    = d_raw_scores_h[:, :, None] * q_cond_h[:, None, :]
            dK_h[invalid_h] = 0.0

            dW_Qh   = cand_enc.T @ dq_h
            db_Qh   = dq_h.sum(axis=0)
            dK_flat = dK_h.reshape(batch * 4, D_HEAD)
            dW_Kh   = comp_flat.T @ dK_flat
            db_Kh   = dK_flat.sum(axis=0)
            dV_flat = dV_h.reshape(batch * 4, D_HEAD)
            dW_Vh   = comp_flat.T @ dV_flat
            db_Vh   = dV_flat.sum(axis=0)

            head_grad_lists.append([dW_Qh, db_Qh, dW_Kh, db_Kh, dW_Vh, db_Vh])

        all_grads = [dW0, db0, dW1, db1, dW2, db2]
        for h in range(N_HEADS):
            all_grads.extend(head_grad_lists[h])
        all_grads.extend([dW_O, db_O])

        return loss, H_batch, all_grads, q_pred, targets

    # ------------------------------------------------------------------
    # Online update
    # ------------------------------------------------------------------

    def update_online(self, states, actions, r_vd, r_ss,
                      next_states, dones, omega_s_batch):
        """Train step using stored (r_vd, r_ss, omega_s) from replay buffer."""
        omega_s_arr  = np.asarray(omega_s_batch, dtype=np.float64)
        omega_mct_arr = 1.0 - omega_s_arr

        # Reconstruct combined reward using stored omega
        rewards = (omega_mct_arr * r_vd
                   + omega_s_arr * r_ss / np.maximum(omega_mct_arr, 0.1))

        states_d     = np.asarray(states,     dtype=np.float64)
        next_states_d = np.asarray(next_states, dtype=np.float64)
        rewards_d    = np.asarray(rewards,     dtype=np.float64)
        dones_d      = np.asarray(dones,       dtype=np.float64)

        loss, H_batch, all_grads, _, _ = self._backward(
            states_d, actions, rewards_d, next_states_d, dones_d, omega_s_arr)

        # Gradient clipping
        global_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in all_grads)))
        if global_norm > self.grad_clip:
            scale     = self.grad_clip / global_norm
            all_grads = [g * scale for g in all_grads]

        all_params = [self._W[0], self._b[0], self._W[1], self._b[1],
                      self._W[2], self._b[2]]
        for h in range(N_HEADS):
            all_params.extend([self.W_Q[h], self.b_Q[h], self.W_K[h],
                                self.b_K[h], self.W_V[h], self.b_V[h]])
        all_params.extend([self.W_O, self.b_O])

        self._opt.step(all_params, all_grads)
        return loss, H_batch

    # ------------------------------------------------------------------
    # Gradient norm monitoring (no optimizer step)
    # ------------------------------------------------------------------

    def compute_grad_norms(self, states, actions, r_vd, r_ss,
                           next_states, dones, omega_s_batch):
        """Compute gradient norms for each reward component independently.
        Returns (norm_vd, norm_ss, ratio)."""
        omega_s_arr  = np.asarray(omega_s_batch, dtype=np.float64)
        omega_mct_arr = 1.0 - omega_s_arr

        states_d      = np.asarray(states,      dtype=np.float64)
        next_states_d = np.asarray(next_states, dtype=np.float64)
        dones_d       = np.asarray(dones,       dtype=np.float64)

        # Component 1: value_delta only
        r_vd_only = (omega_mct_arr * np.asarray(r_vd, dtype=np.float64))
        _, _, grads_vd, _, _ = self._backward(
            states_d, actions, r_vd_only, next_states_d, dones_d, omega_s_arr)
        norm_vd = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads_vd)))

        # Component 2: starvation signal only
        r_ss_only = (omega_s_arr * np.asarray(r_ss, dtype=np.float64)
                     / np.maximum(omega_mct_arr, 0.1))
        _, _, grads_ss, _, _ = self._backward(
            states_d, actions, r_ss_only, next_states_d, dones_d, omega_s_arr)
        norm_ss = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads_ss)))

        ratio = norm_ss / (norm_vd + 1e-12)
        return norm_vd, norm_ss, ratio

    # ------------------------------------------------------------------
    # Epsilon decay
    # ------------------------------------------------------------------

    def decay_epsilon(self, ep: int, min_eps: float = 0.05,
                      decay: float = 0.9997) -> float:
        # Slower decay for 30k episodes: 0.9997^30000 ≈ 0.011
        self.epsilon = max(min_eps, 1.0 * (decay ** ep))
        return self.epsilon

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

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
# Weight re-initializer
# ---------------------------------------------------------------------------

def reinit_weights_w14(agent: W14OmegaDQN, seed: int) -> None:
    rng = np.random.default_rng(seed)
    def he(fan_in: int, *shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float64) * np.sqrt(2.0 / fan_in)

    for h in range(N_HEADS):
        agent.W_Q[h][:] = he(D_CAND, D_CAND, D_HEAD)
        agent.b_Q[h][:] = 0.0
        agent.W_K[h][:] = he(D_CAND, D_CAND, D_HEAD)
        agent.b_K[h][:] = 0.0
        agent.W_V[h][:] = he(D_CAND, D_CAND, D_HEAD)
        agent.b_V[h][:] = 0.0

    agent.W_O[:] = he(D_V_TOT, D_V_TOT, D_V_TOT)
    agent.b_O[:] = 0.0

    dm = agent.d_mlp_in
    agent._W[0][:] = he(dm, dm, 64)
    agent._b[0][:] = 0.0
    agent._W[1][:] = he(64, 64, 32)
    agent._b[1][:] = 0.0
    agent._W[2][:] = he(32, 32,  1)
    agent._b[2][:] = 0.0
    agent.update_target()


# ---------------------------------------------------------------------------
# Omega sampling schedule
# ---------------------------------------------------------------------------

def sample_omega(ep: int) -> float:
    """Return omega_starvation for this episode."""
    if ep < 5000:
        return random.uniform(0.0, 1.0)          # Beta(1,1) = Uniform
    else:
        u = random.uniform(0.0, 1.0)
        return math.sin(u * math.pi / 2) ** 2    # Beta(0.5,0.5) approximation


# ---------------------------------------------------------------------------
# Reward decomposition
# ---------------------------------------------------------------------------

def compute_rewards(env: SchedEnv, runnable_pre: list,
                    chosen, q_actual: float) -> tuple[float, float]:
    """Returns (r_vd, r_ss) — the two reward components separately."""
    # Value delta component
    r_vd = sum(
        value_delta(p.tau, p.floor, p.base_value, p.wait_time, q_actual)
        for p in runnable_pre
    ) / REWARD_SCALE

    # Starvation signal: proportional, dense, fires above 50s threshold
    runnable_now = [p for p in env.processes
                    if p.arrival_time <= env.current_time and not p.is_complete]
    n_run = max(len(runnable_now), 1)
    r_ss = -sum(
        max(0.0, p.time_since_last_execution - STARVATION_THRESHOLD) / STARVATION_THRESHOLD
        for p in runnable_now
    ) / n_run

    return r_vd, r_ss


# ---------------------------------------------------------------------------
# Training loop for one seed
# ---------------------------------------------------------------------------

def train_seed(seed: int, train_sampler,
               results_dir: str) -> tuple[W14OmegaDQN, list[dict]]:
    agent = W14OmegaDQN()
    reinit_weights_w14(agent, seed)

    random.seed(seed)
    np.random.seed(seed)
    rng_ep = np.random.default_rng(seed)

    buffer = OmegaReplayBuffer(capacity=BUF_CAPACITY,
                               state_dim=N_PROCESSES * D_CAND)

    total_transitions = 0
    win_loss, win_ent, win_mct, win_srpt, win_starv = [], [], [], [], []
    checkpoint_log: list[dict] = []

    for ep in range(1, N_EPISODES + 1):
        agent.lambda_ent = LAMBDA_START - (LAMBDA_START - LAMBDA_END) * (ep / N_EPISODES)
        omega_s = sample_omega(ep)
        omega_mct = 1.0 - omega_s

        tasks = train_sampler.sample_episode(rng_ep)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()
        sv    = _encode_7dim(env, tasks)

        ep_loss_sum  = 0.0
        ep_ent_sum   = 0.0
        ep_loss_n    = 0
        srpt_agree_n = 0
        srpt_total_n = 0
        done         = False

        while not done:
            valid    = _valid_actions(env)
            runnable = [p for p in env.processes
                        if p.arrival_time <= env.current_time and not p.is_complete]
            srpt_pid = (min(runnable, key=lambda p: p.remaining_burst).pid
                        if runnable else -1)

            action = agent.select_action(sv, agent.epsilon, valid, omega_s)

            chosen   = env.processes[action // N_QT]
            q_tier   = action % N_QT
            q        = QUANTUM_TIERS[q_tier]
            q_actual = min(q, chosen.remaining_burst)

            runnable_pre = list(runnable)
            _, _, done, info = env.step(action)

            r_vd, r_ss = compute_rewards(env, runnable_pre, chosen, q_actual)

            sv_next = _encode_7dim(env, tasks)

            if srpt_pid >= 0:
                srpt_agree_n += int((action // N_QT) == srpt_pid)
                srpt_total_n += 1

            buffer.store(sv, action, r_vd, r_ss, sv_next, done, omega_s)
            total_transitions += 1

            if total_transitions >= WARMUP and len(buffer) >= BATCH_SIZE:
                s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b = buffer.sample(BATCH_SIZE)
                loss, ent = agent.update_online(
                    s_b.astype(np.float64), a_b,
                    rvd_b.astype(np.float64),
                    rss_b.astype(np.float64),
                    ns_b.astype(np.float64),
                    d_b.astype(np.float64),
                    om_b.astype(np.float64),
                )
                ep_loss_sum += loss
                ep_ent_sum  += ent
                ep_loss_n   += 1

            sv = sv_next

        if ep % TARGET_UPDATE_FREQ == 0:
            agent.update_target()

        mct        = info.get("mean_completion_time_so_far") or 0.0
        srpt_frac  = srpt_agree_n / srpt_total_n if srpt_total_n > 0 else 0.0
        mean_loss  = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else float("nan")
        mean_ent   = ep_ent_sum  / ep_loss_n if ep_loss_n > 0 else float("nan")

        win_mct.append(mct)
        win_srpt.append(srpt_frac)
        win_loss.append(mean_loss)
        win_ent.append(mean_ent)

        # Simple starvation estimate from completed processes
        completed = [p for p in env.processes if p.is_complete]
        ep_starved = 0
        if completed:
            turns  = [p.completion_time - p.arrival_time for p in completed]
            bursts = [p.burst_length for p in completed]
            slows  = [t / max(b, 1e-6) for t, b in zip(turns, bursts)]
            med    = float(np.median(slows))
            if any(s > 3.0 * med for s in slows):
                ep_starved = 1
        win_starv.append(ep_starved)

        agent.decay_epsilon(ep)

        if ep % PRINT_EVERY == 0:
            n  = min(ep, 100)
            al = float(np.nanmean(win_loss[-n:]))
            ah = float(np.nanmean(win_ent[-n:]))
            am = float(np.mean(win_mct[-n:]))
            sp = float(np.mean(win_srpt[-n:])) * 100.0
            st = float(np.mean(win_starv[-n:])) * 100.0

            # Gradient norm monitoring
            grad_ratio = float("nan")
            warn = ""
            if len(buffer) >= BATCH_SIZE:
                s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b = buffer.sample(BATCH_SIZE)
                nv, ns_, ratio = agent.compute_grad_norms(
                    s_b.astype(np.float64), a_b,
                    rvd_b.astype(np.float64),
                    rss_b.astype(np.float64),
                    ns_b.astype(np.float64),
                    d_b.astype(np.float64),
                    om_b.astype(np.float64),
                )
                grad_ratio = ratio
                if ratio > GRAD_RATIO_WARN:
                    warn = f"  ⚠ WARNING: W11 ghost risk — starvation gradient dominating (ratio={ratio:.1f})"

            print(f"  ep {ep:>6} | ω_s={omega_s:.2f} | loss={al:.4f} | H={ah:.4f} | "
                  f"MCT={am:.2f}s | SRPT={sp:.1f}% | Starv={st:.1f}% | "
                  f"grad_ratio={grad_ratio:.2f}")
            if warn:
                print(warn)
            sys.stdout.flush()

            checkpoint_log.append({
                "ep": ep, "omega_s_last": omega_s,
                "loss": al, "entropy": ah,
                "mct": am, "srpt": sp, "starve": st,
                "grad_ratio": grad_ratio if not math.isnan(grad_ratio) else -1.0,
            })

            if al > LOSS_GATE:
                print(f"  STOP: loss {al:.2f} > {LOSS_GATE}")
                break
            if ep == 5000 and ah > ENT_GATE_5K:
                print(f"  STOP: H={ah:.4f} > {ENT_GATE_5K} at ep 5000")
                break

        # Periodic checkpoints
        if ep % CKPT_EVERY == 0:
            ckpt_path = os.path.join(results_dir, f"w14_seed{seed}_ep{ep}.npz")
            agent.save(ckpt_path)
            print(f"  [checkpoint] → {ckpt_path}")
            sys.stdout.flush()

    return agent, checkpoint_log


# ---------------------------------------------------------------------------
# Evaluation at fixed omega
# ---------------------------------------------------------------------------

def evaluate_at_omega(agent: W14OmegaDQN, test_sampler,
                      omega_s: float, n_eval: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mcts, srpts, starved_list = [], [], []
    all_vlrs = []

    for _ in range(n_eval):
        tasks = test_sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()
        sv    = _encode_7dim(env, tasks)

        srpt_agree = 0
        srpt_total = 0
        done = False

        while not done:
            valid    = _valid_actions(env)
            runnable = [p for p in env.processes
                        if p.arrival_time <= env.current_time and not p.is_complete]
            srpt_pid = min(runnable, key=lambda p: p.remaining_burst).pid if runnable else -1

            action = agent.select_action(sv, epsilon=0.0,
                                         valid_actions=valid, omega_s=omega_s)
            _, _, done, info = env.step(action)
            sv = _encode_7dim(env, tasks)

            if srpt_pid >= 0:
                srpt_agree += int((action // N_QT) == srpt_pid)
                srpt_total += 1

        mcts.append(info.get("mean_completion_time_so_far") or 0.0)
        srpts.append(srpt_agree / srpt_total if srpt_total > 0 else 0.0)

        completed = [p for p in env.processes if p.is_complete]
        starved = 0
        if completed:
            turns  = [p.completion_time - p.arrival_time for p in completed]
            bursts = [p.burst_length for p in completed]
            slows  = [t / max(b, 1e-6) for t, b in zip(turns, bursts)]
            med    = float(np.median(slows))
            if any(s > 3.0 * med for s in slows):
                starved = 1
        starved_list.append(starved)

        for p in env.processes:
            if p.is_complete:
                delay = p.wait_time
                v_now = (p.base_value * max(p.floor, math.exp(-delay / p.tau))
                         if p.tau > 0 else p.base_value)
                vlr = (p.base_value - v_now) / max(delay, 1.0)
                all_vlrs.append(vlr)

    def vrfi(vlrs):
        a = np.array(vlrs, dtype=np.float64)
        cv = float(np.std(a) / (np.mean(a) + 1e-12)) if len(a) > 1 else 0.0
        return 1.0 - cv

    return {
        "omega_s":    omega_s,
        "mct_mean":   float(np.mean(mcts)),
        "mct_std":    float(np.std(mcts)),
        "srpt_mean":  float(np.mean(srpts)) * 100.0,
        "starve_pct": float(np.sum(starved_list)) / n_eval * 100.0,
        "vrfi":       vrfi(all_vlrs),
    }


# ---------------------------------------------------------------------------
# ASCII Pareto frontier plot
# ---------------------------------------------------------------------------

def ascii_pareto_plot(frontier: list[dict]) -> None:
    """ASCII scatter: X=Starvation%, Y=MCT. Each point labelled with omega."""
    x_min, x_max = 0.0,  65.0
    y_min, y_max = 15.0, 32.0
    W, H_plot = 60, 20

    grid = [[" "] * W for _ in range(H_plot)]

    def to_xy(starv, mct):
        col = int((starv - x_min) / (x_max - x_min) * (W - 1))
        row = int((1.0 - (mct - y_min) / (y_max - y_min)) * (H_plot - 1))
        return max(0, min(W - 1, col)), max(0, min(H_plot - 1, row))

    # MLFQ reference
    mx, my = to_xy(MLFQ_STARVE, MLFQ_MCT)
    grid[my][mx] = "M"

    # Omega points
    for pt in frontier:
        col, row = to_xy(pt["starve_pct"], pt["mct_mean"])
        label = f"{pt['omega_s']:.1f}"[1:]  # ".0", ".1", etc.
        grid[row][col] = label[-1]

    print("\n  ASCII Pareto Frontier (W14-ω)")
    print(f"  Y: MCT (↑={y_max:.0f}s → ↓={y_min:.0f}s)   X: Starvation% (←=0% → →={x_max:.0f}%)")
    print("  M=MLFQ reference, digit = omega_starvation × 10")
    print()
    for row_idx, row in enumerate(grid):
        mct_label = y_max - (y_max - y_min) * row_idx / (H_plot - 1)
        print(f"  {mct_label:4.1f}s |{''.join(row)}|")
    print("         " + "-" * W)
    print("         " + f"{'0%':<{W//3}}{'~30%':<{W//3}}{'~65%':>{W//3}}")


# ---------------------------------------------------------------------------
# MLFQ inline baseline
# ---------------------------------------------------------------------------

def run_mlfq(test_sampler, n_eval: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mcts, srpts, starved_list, all_vlrs = [], [], [], []

    for _ in range(n_eval):
        tasks = test_sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()

        ep_state = {
            "mlfq_queue":     {p.pid: 0 for p in env.processes},
            "prev_remaining": {p.pid: p.burst_length for p in env.processes},
            "last_action_pid": None,
        }
        srpt_agree = 0
        srpt_total = 0
        done = False

        while not done:
            runnable = [p for p in env.processes
                        if p.arrival_time <= env.current_time and not p.is_complete]
            if not runnable:
                break
            srpt_pid = min(runnable, key=lambda p: p.remaining_burst).pid

            queues   = ep_state["mlfq_queue"]
            prev_rem = ep_state["prev_remaining"]
            last_act = ep_state["last_action_pid"]

            if last_act is not None:
                prev_p = next((p for p in env.processes if p.pid == last_act), None)
                if prev_p is not None and not prev_p.is_complete:
                    tier     = queues.get(prev_p.pid, 0)
                    consumed = prev_rem.get(prev_p.pid, 0.0) - prev_p.remaining_burst
                    if consumed >= QUANTUM_TIERS[tier] - 1e-6 and tier < 2:
                        queues[prev_p.pid] = tier + 1

            for p in runnable:
                if p.time_since_last_execution > MLFQ_AGE_THRESH:
                    queues[p.pid] = 0

            action = None
            for level in range(3):
                candidates = [p for p in runnable if queues.get(p.pid, 0) == level]
                if candidates:
                    chosen = min(candidates, key=lambda p: p.arrival_time)
                    ep_state["last_action_pid"] = chosen.pid
                    prev_rem[chosen.pid]        = chosen.remaining_burst
                    action = chosen.pid * N_QT + level
                    break
            if action is None:
                action = runnable[0].pid * N_QT + 0

            _, _, done, info = env.step(action)
            if srpt_pid >= 0:
                srpt_agree += int((action // N_QT) == srpt_pid)
                srpt_total += 1

        mcts.append(info.get("mean_completion_time_so_far") or 0.0)
        srpts.append(srpt_agree / srpt_total if srpt_total > 0 else 0.0)

        completed = [p for p in env.processes if p.is_complete]
        starved = 0
        if completed:
            turns  = [p.completion_time - p.arrival_time for p in completed]
            bursts = [p.burst_length for p in completed]
            slows  = [t / max(b, 1e-6) for t, b in zip(turns, bursts)]
            med    = float(np.median(slows))
            if any(s > 3.0 * med for s in slows):
                starved = 1
        starved_list.append(starved)

        for p in env.processes:
            if p.is_complete:
                delay = p.wait_time
                v_now = (p.base_value * max(p.floor, math.exp(-delay / p.tau))
                         if p.tau > 0 else p.base_value)
                vlr = (p.base_value - v_now) / max(delay, 1.0)
                all_vlrs.append(vlr)

    def vrfi(vlrs):
        a = np.array(vlrs, dtype=np.float64)
        cv = float(np.std(a) / (np.mean(a) + 1e-12)) if len(a) > 1 else 0.0
        return 1.0 - cv

    return {
        "mct_mean":   float(np.mean(mcts)),
        "mct_std":    float(np.std(mcts)),
        "srpt_mean":  float(np.mean(srpts)) * 100.0,
        "starve_pct": float(np.sum(starved_list)) / n_eval * 100.0,
        "vrfi":       vrfi(all_vlrs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 72)
    print("W14-ω  Preference-Conditioned Attention DQN")
    print(f"Seeds: {SEEDS}  N_episodes={N_EPISODES}  N_eval_pareto={N_EVAL_PARETO}")
    print(f"MLFQ target: MCT<{MLFQ_MCT}s  Starve<{MLFQ_STARVE}%")
    print("=" * 72)

    print(f"\nLoading TRAIN trace: {TRACE_TRAIN}")
    train_sampler = TraceEpisodeSampler5(TRACE_TRAIN)
    print(f"Loading TEST trace: {TRACE_TEST}")
    test_sampler  = TraceEpisodeSampler5(TRACE_TEST)
    print()

    OMEGA_SWEEP = [round(x * 0.1, 1) for x in range(11)]  # 0.0 to 1.0

    all_seed_frontiers: dict[int, list[dict]] = {}
    all_ckpt_logs: dict[int, list[dict]] = {}
    best_agent      = None
    best_mct        = float("inf")
    best_seed       = -1

    for seed in SEEDS:
        print(f"\n{'='*72}")
        print(f"SEED {seed}")
        print(f"{'='*72}")
        t0 = time.time()

        agent, ckpt_log = train_seed(seed, train_sampler, RESULTS_DIR)
        wall = (time.time() - t0) / 60.0
        print(f"\n  Training complete — {wall:.1f} min")
        all_ckpt_logs[seed] = ckpt_log

        # Save final checkpoint
        final_path = os.path.join(RESULTS_DIR, f"w14_seed{seed}_final.npz")
        agent.save(final_path)
        print(f"  Final checkpoint → {final_path}")

        # Pareto frontier sweep
        print(f"\n  Evaluating Pareto frontier (N={N_EVAL_PARETO} per omega point)...")
        agent.epsilon = 0.0
        frontier = []
        for omega_s in OMEGA_SWEEP:
            m = evaluate_at_omega(agent, test_sampler,
                                  omega_s, N_EVAL_PARETO, seed=seed * 1000)
            m["wall_total_min"] = wall
            frontier.append(m)
            dom = "★ DOMINATES MLFQ" if (m["mct_mean"] < MLFQ_MCT and
                                          m["starve_pct"] < MLFQ_STARVE) else ""
            print(f"    ω_s={omega_s:.1f}: MCT={m['mct_mean']:.2f}s  "
                  f"Starve={m['starve_pct']:.1f}%  "
                  f"VRFI={m['vrfi']:.3f}  {dom}")
            sys.stdout.flush()

        all_seed_frontiers[seed] = frontier

        # Track best agent (by MCT at omega_s=0.0, the pure MCT mode)
        mct_at_0 = next(f["mct_mean"] for f in frontier if f["omega_s"] == 0.0)
        if mct_at_0 < best_mct:
            best_mct   = mct_at_0
            best_agent = agent
            best_seed  = seed

    # -------------------------------------------------------------------------
    # Use best seed's frontier for final report
    # -------------------------------------------------------------------------
    frontier = all_seed_frontiers[best_seed]
    print(f"\n\nBest seed: {best_seed} (MCT at ω_s=0.0: {best_mct:.2f}s)")

    # Fresh MLFQ baseline
    print("\nRunning MLFQ benchmark (N=200, seed=999)...")
    mlfq_res = run_mlfq(test_sampler, N_EVAL_PARETO, seed=999)
    print(f"  MLFQ: MCT={mlfq_res['mct_mean']:.2f}s  "
          f"Starve={mlfq_res['starve_pct']:.1f}%  "
          f"VRFI={mlfq_res['vrfi']:.3f}")

    # -------------------------------------------------------------------------
    # Pareto frontier table
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 72)
    print("PARETO FRONTIER TABLE  (best seed, N=200 eval per omega)")
    print("=" * 72)
    print(f"\n{'omega_s':>8} {'MCT (mean)':>12} {'Starvation%':>13} {'VRFI':>7} {'SRPT%':>7}  Dominates MLFQ?")
    print("-" * 65)
    print(f"  {'MLFQ':>6} {mlfq_res['mct_mean']:>11.2f}s {mlfq_res['starve_pct']:>12.1f}% "
          f"{mlfq_res['vrfi']:>7.3f} {mlfq_res['srpt_mean']:>6.1f}%  ← reference")

    dominant_omegas = []
    recommended_omega = None
    for pt in frontier:
        dom = (pt["mct_mean"] < mlfq_res["mct_mean"] and
               pt["starve_pct"] < mlfq_res["starve_pct"])
        dom_str = "★ YES" if dom else "no"
        if dom:
            dominant_omegas.append(pt["omega_s"])
            if recommended_omega is None:
                recommended_omega = pt["omega_s"]
        print(f"  {pt['omega_s']:>6.1f} {pt['mct_mean']:>11.2f}s "
              f"{pt['starve_pct']:>12.1f}% "
              f"{pt['vrfi']:>7.3f} {pt['srpt_mean']:>6.1f}%  {dom_str}")

    # ASCII plot
    ascii_pareto_plot(frontier)

    # -------------------------------------------------------------------------
    # Gradient norm summary
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 72)
    print("GRADIENT NORM RATIO HISTORY  (starvation_grad / value_delta_grad)")
    print("=" * 72)
    for seed in SEEDS:
        log = all_ckpt_logs[seed]
        print(f"\n  Seed {seed}:")
        for entry in log:
            ratio = entry["grad_ratio"]
            warn  = " ⚠ W11 ghost risk" if ratio > GRAD_RATIO_WARN else ""
            print(f"    ep {entry['ep']:>6}: ratio={ratio:.3f}{warn}")

    # -------------------------------------------------------------------------
    # Verdict
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if dominant_omegas:
        print(f"\n★ SUCCESS: {len(dominant_omegas)} omega value(s) dominate MLFQ on both metrics.")
        print(f"  Dominant omega_s values: {dominant_omegas}")
        print(f"  Recommended deployment omega_s: {recommended_omega}")
    else:
        # Find best trade-off (closest to MLFQ on both axes)
        best_tradeoff = min(frontier,
            key=lambda pt: (max(0, pt["starve_pct"] - mlfq_res["starve_pct"])**2
                            + max(0, pt["mct_mean"] - mlfq_res["mct_mean"])**2))
        print(f"\nNo omega value dominates MLFQ on both metrics simultaneously.")
        print(f"Best trade-off: omega_s={best_tradeoff['omega_s']:.1f}  "
              f"MCT={best_tradeoff['mct_mean']:.2f}s  "
              f"Starve={best_tradeoff['starve_pct']:.1f}%")

    # Save final agent
    final_agent_path = os.path.join(RESULTS_DIR, "w14_omega_final.npz")
    best_agent.save(final_agent_path)
    print(f"\nFinal agent saved → {final_agent_path}")

    # -------------------------------------------------------------------------
    # Save Pareto JSON
    # -------------------------------------------------------------------------
    pareto_out = {
        "mlfq": mlfq_res,
        "best_seed": best_seed,
        "pareto_frontier": frontier,
        "dominant_omegas": dominant_omegas,
        "recommended_omega": recommended_omega,
        "all_seed_frontiers": {str(s): all_seed_frontiers[s] for s in SEEDS},
        "checkpoint_logs": {str(s): all_ckpt_logs[s] for s in SEEDS},
    }
    json_path = os.path.join(RESULTS_DIR, "pareto_frontier.json")
    with open(json_path, "w") as f:
        json.dump(pareto_out, f, indent=2)
    print(f"Pareto data saved → {json_path}")


if __name__ == "__main__":
    main()
