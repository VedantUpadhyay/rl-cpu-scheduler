"""W11 / W11b / W11c / W11d / W12 Multi-Seed Ablation — 3 seeds × 10 k episodes each.

Agents and configurations (all import helpers from local scripts/w9_train.py):

  W11   D_CAND=7  no burst; 7 obs features; env_reward/RS + CT-bonus(w1=0.2)
  W11b  D_CAND=7  no burst; 7 obs features; env_reward/RS + CT-bonus(w1=0.6)
  W11c  D_CAND=8  no burst; 7 obs + noisy burst; env_reward/RS + CT-bonus(w1=0.6)
  W11d  D_CAND=6  true burst; 6 features; env_reward/RS + CT-bonus(w1=0.6)
  W12   D_CAND=7  no burst; 7 obs features; value-delta/20 + PBRS(lam=0.01)

Seeds: 42, 123, 456  (each run re-inits weights, resets episode RNG)
Eval:  N=500 episodes from test split (epsilon=0)

Results:
  Per-seed: MCT mean±std, SRPT%, starvation%
  Cross-seed: mean±std for MCT and SRPT
  Fairness suite on best checkpoint of W12 (JFI, SDV, VRFI)
  2×2 ablation table printed at end
"""
from __future__ import annotations
import csv, math, os, random, sys, time
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — use local scripts/ w9_train (D_CAND=6) for helpers
# ---------------------------------------------------------------------------
_PROJECT_ROOT = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                 "GRAD - FALL 23/UCSC/Capstone")
_SCRIPTS_DIR  = os.path.join(_PROJECT_ROOT, "docs", "scheduler-research", "scripts")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPTS_DIR)

from schedsim.env    import SchedEnv, N_PROCESSES, N_QUANTUM_TIERS, value_delta
from schedsim.agent  import AdamOptimizer, ReplayBuffer
from schedsim.process import Process

# Import helpers from scripts/w9_train (D_CAND=6, WAIT_NORM=500)
from w9_train import (
    TraceEpisodeSampler5,
    _valid_actions, _make_procs,
    N_QT, N_ACTIONS, QT_VALUES,
    _norm_burst, _norm_time_log, _urgency_norm, _norm_cpu, _norm_mem,
    BURST_P95_FILT, WAIT_NORM, CPU_MAX, MEM_P95,
    REWARD_SCALE,   # = 1.0 in scripts/w9_train
)

# ---------------------------------------------------------------------------
# Architecture constants (shared across all agents)
# ---------------------------------------------------------------------------
N_HEADS  = 2
D_HEAD   = 8
D_V_TOT  = N_HEADS * D_HEAD   # 16

PBRS_LAMBDA  = 0.01
REWARD_SCALE_W12 = 20.0   # W12 divides value-delta by 20 (same as Project/w10c)

CT_NORM = 60.0   # normalises completion-time bonus: -W_CT * ct / CT_NORM

# ---------------------------------------------------------------------------
# Training hyperparameters (identical to W10C)
# ---------------------------------------------------------------------------
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
LOSS_GATE          = 100.0
ENT_GATE_5K        = 1.2
PRINT_EVERY        = 500

SEEDS  = [42, 123, 456]
N_EVAL = 500

TRACE_TRAIN = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
TRACE_TEST  = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_test_filtered.csv"
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "results")

# ---------------------------------------------------------------------------
# State encoders
# ---------------------------------------------------------------------------
_TIME_LOG_DENOM = float(np.log1p(500.0))

def _encode_7dim(env: SchedEnv, task_meta: list[dict]) -> np.ndarray:
    """7-dim observable state (no burst) — used by W11, W11b, W12.
    Features per process: [tq_norm, wait_norm, last_exec_norm, urgency_norm,
                           cpu_norm, mem_norm, arrived_flag]
    arrived_flag at position 6.
    """
    D = 7
    vec = np.zeros(N_PROCESSES * D, dtype=np.float32)
    for p in env.processes:
        i   = p.pid
        off = i * D
        if not p.is_complete and p.arrival_time <= env.current_time:
            tq = env.current_time - p.arrival_time
            vec[off + 0] = _norm_time_log(tq)
            vec[off + 1] = p.wait_time / WAIT_NORM
            vec[off + 2] = _norm_time_log(p.time_since_last_execution)
            vec[off + 3] = _urgency_norm(p)
            vec[off + 4] = _norm_cpu(task_meta[i]["plan_cpu"])
            vec[off + 5] = _norm_mem(task_meta[i]["plan_mem"])
            vec[off + 6] = 1.0
    return vec


def _encode_8dim(env: SchedEnv, task_meta: list[dict],
                 rng: np.random.Generator) -> np.ndarray:
    """8-dim state with noisy burst estimate — used by W11c.
    Features: [tq_norm, wait_norm, last_exec_norm, urgency_norm,
               cpu_norm, mem_norm, arrived_flag, noisy_burst_norm]
    arrived_flag at position 6, noisy_burst at position 7.
    noise ~ N(0, 0.3) applied multiplicatively, floor at 0.1.
    """
    D = 8
    vec = np.zeros(N_PROCESSES * D, dtype=np.float32)
    for p in env.processes:
        i   = p.pid
        off = i * D
        if not p.is_complete and p.arrival_time <= env.current_time:
            tq = env.current_time - p.arrival_time
            vec[off + 0] = _norm_time_log(tq)
            vec[off + 1] = p.wait_time / WAIT_NORM
            vec[off + 2] = _norm_time_log(p.time_since_last_execution)
            vec[off + 3] = _urgency_norm(p)
            vec[off + 4] = _norm_cpu(task_meta[i]["plan_cpu"])
            vec[off + 5] = _norm_mem(task_meta[i]["plan_mem"])
            vec[off + 6] = 1.0
            # noisy burst estimate: true * (1 + N(0,0.3)), floored at 0.1 s
            noisy = max(0.1, p.remaining_burst * (1.0 + rng.normal(0.0, 0.3)))
            vec[off + 7] = _norm_burst(noisy)
    return vec


def _encode_6dim(env: SchedEnv, task_meta: list[dict]) -> np.ndarray:
    """6-dim state with true burst — used by W11d (same as scripts/w9_train).
    Features: [burst_norm, wait_norm, last_exec_norm, urgency_norm,
               tq_norm, arrived_flag]
    arrived_flag at position 5.
    """
    D = 6
    vec = np.zeros(N_PROCESSES * D, dtype=np.float32)
    for p in env.processes:
        i   = p.pid
        off = i * D
        if not p.is_complete and p.arrival_time <= env.current_time:
            vec[off + 0] = _norm_burst(p.remaining_burst)
            vec[off + 1] = p.wait_time / WAIT_NORM
            vec[off + 2] = _norm_time_log(p.time_since_last_execution)
            vec[off + 3] = _urgency_norm(p)
            vec[off + 4] = _norm_time_log(env.current_time - p.arrival_time)
            vec[off + 5] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Generic 2-head AttentionDQN — parametric in D_CAND and arrived_flag_idx
# ---------------------------------------------------------------------------

class AttentionDQN:
    """2-head attention DQN, parametric in state dimension.

    Parameters
    ----------
    d_cand : int
        Per-process feature dimension (6, 7, or 8).
    arrived_flag_idx : int
        Which offset within a process's feature slice contains the validity flag.
        Valid (runnable) iff comp_encs[:,:, arrived_flag_idx] > 0.5.
    """

    def __init__(self, d_cand: int, arrived_flag_idx: int,
                 lr: float = 0.001, gamma: float = 1.0,
                 grad_clip: float = 1.0, lambda_ent: float = 0.10) -> None:
        self.d_cand          = d_cand
        self.arrived_flag_idx = arrived_flag_idx
        self.d_mlp_in        = D_V_TOT + d_cand + 1   # context + cand_enc + qt
        self.gamma           = gamma
        self.grad_clip       = grad_clip
        self.lambda_ent      = lambda_ent
        self.epsilon         = 1.0

        rng = np.random.default_rng(42)
        def he(fan_in: int, *shape: int) -> np.ndarray:
            return rng.standard_normal(shape).astype(np.float64) * np.sqrt(2.0 / fan_in)

        self.W_Q = [he(d_cand, d_cand, D_HEAD) for _ in range(N_HEADS)]
        self.b_Q = [np.zeros(D_HEAD)            for _ in range(N_HEADS)]
        self.W_K = [he(d_cand, d_cand, D_HEAD) for _ in range(N_HEADS)]
        self.b_K = [np.zeros(D_HEAD)            for _ in range(N_HEADS)]
        self.W_V = [he(d_cand, d_cand, D_HEAD) for _ in range(N_HEADS)]
        self.b_V = [np.zeros(D_HEAD)            for _ in range(N_HEADS)]

        self.W_O = he(D_V_TOT, D_V_TOT, D_V_TOT)
        self.b_O = np.zeros(D_V_TOT)

        self._W = [he(self.d_mlp_in, self.d_mlp_in, 64),
                   he(64,            64,             32),
                   he(32,            32,              1)]
        self._b = [np.zeros(64), np.zeros(32), np.zeros(1)]

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
        n = 0
        for h in range(N_HEADS):
            n += self.W_Q[h].size + self.b_Q[h].size
            n += self.W_K[h].size + self.b_K[h].size
            n += self.W_V[h].size + self.b_V[h].size
        n += self.W_O.size + self.b_O.size
        for W, b in zip(self._W, self._b):
            n += W.size + b.size
        return n

    def _build_competitor_data(
        self, states: np.ndarray, pids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        dc       = self.d_cand
        afi      = self.arrived_flag_idx
        batch    = states.shape[0]
        s3d      = states.reshape(batch, N_PROCESSES, dc)
        all_pids = np.tile(np.arange(N_PROCESSES), (batch, 1))
        mask     = all_pids != pids[:, None]
        comp_idx = all_pids[mask].reshape(batch, 4)
        bidx     = np.arange(batch)[:, None]
        comp_encs  = s3d[bidx, comp_idx]
        comp_valid = (comp_encs[:, :, afi] > 0.5)
        return comp_encs, comp_valid

    def _head_forward(self, h: int, cand_enc, comp_flat, comp_valid,
                      use_target: bool = False) -> tuple:
        WQ = (self._tW_Q if use_target else self.W_Q)[h]
        bQ = (self._tb_Q if use_target else self.b_Q)[h]
        WK = (self._tW_K if use_target else self.W_K)[h]
        bK = (self._tb_K if use_target else self.b_K)[h]
        WV = (self._tW_V if use_target else self.W_V)[h]
        bV = (self._tb_V if use_target else self.b_V)[h]

        batch   = cand_enc.shape[0]
        invalid = ~comp_valid

        q      = cand_enc @ WQ + bQ
        K_flat = comp_flat @ WK + bK
        V_flat = comp_flat @ WV + bV
        K = K_flat.reshape(batch, 4, D_HEAD)
        V = V_flat.reshape(batch, 4, D_HEAD)

        scores = np.einsum('bi,bji->bj', q, K) / np.sqrt(D_HEAD)
        scores_masked = scores.copy()
        scores_masked[invalid] = -1e9
        scores_shifted = scores_masked - scores_masked.max(axis=1, keepdims=True)
        exp_s = np.exp(scores_shifted)
        exp_s[invalid] = 0.0
        denom   = exp_s.sum(axis=1, keepdims=True) + 1e-10
        weights = exp_s / denom

        context_h = np.einsum('bj,bjd->bd', weights, V)
        cache = dict(q=q, K=K, V=V, K_flat=K_flat, V_flat=V_flat,
                     weights=weights, invalid=invalid)
        return context_h, cache

    def _attention_forward(self, cand_enc, comp_encs, comp_valid,
                           use_target: bool = False) -> tuple:
        WO = self._tW_O if use_target else self.W_O
        bO = self._tb_O if use_target else self.b_O

        batch     = cand_enc.shape[0]
        comp_flat = comp_encs.reshape(batch * 4, self.d_cand)

        head_contexts, head_caches = [], []
        for h in range(N_HEADS):
            ctx_h, c_h = self._head_forward(h, cand_enc, comp_flat, comp_valid, use_target)
            head_contexts.append(ctx_h)
            head_caches.append(c_h)

        context_cat = np.concatenate(head_contexts, axis=1)
        context_out = context_cat @ WO + bO

        cache = dict(cand_enc=cand_enc, comp_encs=comp_encs, comp_flat=comp_flat,
                     comp_valid=comp_valid, head_caches=head_caches,
                     context_cat=context_cat)
        return context_out, cache

    def forward_batch(self, states, actions, use_target: bool = False):
        dc    = self.d_cand
        batch = states.shape[0]
        pids  = (actions // N_QT).astype(np.int32)
        qts   = QT_VALUES[actions % N_QT]

        s3d      = states.reshape(batch, N_PROCESSES, dc)
        cand_enc = s3d[np.arange(batch), pids]
        comp_encs, comp_valid = self._build_competitor_data(states, pids)
        context_out, _ = self._attention_forward(cand_enc, comp_encs, comp_valid, use_target)

        a6    = np.column_stack([cand_enc, qts / 2.0])
        x_mlp = np.concatenate([context_out, a6], axis=1)

        W, b = (self._tW, self._tb) if use_target else (self._W, self._b)
        z1 = x_mlp @ W[0] + b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1    @ W[1] + b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2    @ W[2] + b[2]
        return z3

    def select_action(self, state, epsilon: float, valid_actions: list[int]) -> int:
        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))
        states_b  = np.tile(state, (len(valid_actions), 1)).astype(np.float64)
        actions_b = np.array(valid_actions, dtype=np.int32)
        q_vals    = self.forward_batch(states_b, actions_b).flatten()
        return valid_actions[int(np.argmax(q_vals))]

    def update_online(self, states, actions, rewards, next_states, dones):
        dc    = self.d_cand
        batch = states.shape[0]
        pids  = (actions // N_QT).astype(np.int32)
        qts   = QT_VALUES[actions % N_QT]

        s3d      = states.reshape(batch, N_PROCESSES, dc)
        cand_enc = s3d[np.arange(batch), pids]
        comp_encs, comp_valid = self._build_competitor_data(states, pids)

        context_out, attn_cache = self._attention_forward(cand_enc, comp_encs, comp_valid)
        a6    = np.column_stack([cand_enc, qts / 2.0])
        x_mlp = np.concatenate([context_out, a6], axis=1)

        z1 = x_mlp @ self._W[0] + self._b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1    @ self._W[1] + self._b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2    @ self._W[2] + self._b[2]
        q_pred = z3.flatten()

        valid_f = comp_valid.astype(np.float64)
        H_heads = []
        for h in range(N_HEADS):
            w_h   = attn_cache['head_caches'][h]['weights']
            ent_h = -(w_h * np.log(w_h + 1e-8) * valid_f).sum(axis=1)
            H_heads.append(float(np.mean(ent_h)))
        H_batch = float(np.mean(H_heads))

        ns         = np.asarray(next_states, dtype=np.float64)
        all_acts   = np.tile(np.arange(N_ACTIONS), batch)
        all_ns     = np.repeat(ns, N_ACTIONS, axis=0)
        Q_next_flat = self.forward_batch(all_ns, all_acts, use_target=True).flatten()
        Q_next_mat  = Q_next_flat.reshape(batch, N_ACTIONS)

        afi = self.arrived_flag_idx
        valid_mask = np.zeros((batch, N_ACTIONS), dtype=bool)
        for pid in range(N_PROCESSES):
            runnable = (ns[:, pid * dc + afi] > 0.5)
            for qt in range(N_QT):
                valid_mask[:, pid * N_QT + qt] = runnable

        Q_next_mat[~valid_mask] = -np.inf
        all_invalid = ~np.any(valid_mask, axis=1)
        max_q_next  = np.where(all_invalid, 0.0, Q_next_mat.max(axis=1))
        targets = rewards + self.gamma * max_q_next * (1.0 - dones)

        delta = q_pred - targets
        L_td  = float(np.mean(delta ** 2))
        loss  = L_td - self.lambda_ent * H_batch

        # ----- Backward -----
        dz3    = (2.0 * delta / batch).reshape(-1, 1)
        dW2    = h2.T @ dz3;     db2 = dz3.sum(axis=0)
        dh2    = dz3 @ self._W[2].T
        dz2    = dh2 * (h2 > 0).astype(np.float64)
        dW1    = h1.T @ dz2;     db1 = dz2.sum(axis=0)
        dh1    = dz2 @ self._W[1].T
        dz1    = dh1 * (h1 > 0).astype(np.float64)
        dW0    = x_mlp.T @ dz1;  db0 = dz1.sum(axis=0)
        dx_mlp = dz1 @ self._W[0].T

        d_context_out = dx_mlp[:, :D_V_TOT]

        context_cat = attn_cache['context_cat']
        comp_flat   = attn_cache['comp_flat']
        dW_O = context_cat.T @ d_context_out
        db_O = d_context_out.sum(axis=0)
        d_context_cat = d_context_out @ self.W_O.T

        d_ctx_heads = np.split(d_context_cat, N_HEADS, axis=1)

        head_grad_lists = []
        for h in range(N_HEADS):
            hc        = attn_cache['head_caches'][h]
            q_h       = hc['q']
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

            wdotdw            = (weights_h * d_weights_h).sum(axis=1, keepdims=True)
            d_scores_masked_h = weights_h * (d_weights_h - wdotdw)
            d_scores_masked_h[invalid_h] = 0.0
            d_raw_scores_h    = d_scores_masked_h / np.sqrt(D_HEAD)

            dq_h    = np.einsum('bj,bji->bi', d_raw_scores_h, K_h)
            dK_h    = d_raw_scores_h[:, :, None] * q_h[:, None, :]
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

    def decay_epsilon(self, ep: int, min_eps: float = 0.05,
                      decay: float = 0.9995) -> float:
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
# Weight re-initializer (for multi-seed)
# ---------------------------------------------------------------------------

def reinit_weights(agent: AttentionDQN, seed: int) -> None:
    """Replace all trainable weights in-place using He init with given seed."""
    dc  = agent.d_cand
    rng = np.random.default_rng(seed)

    def he(fan_in: int, *shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float64) * np.sqrt(2.0 / fan_in)

    for h in range(N_HEADS):
        agent.W_Q[h][:] = he(dc,      dc,      D_HEAD)
        agent.b_Q[h][:] = 0.0
        agent.W_K[h][:] = he(dc,      dc,      D_HEAD)
        agent.b_K[h][:] = 0.0
        agent.W_V[h][:] = he(dc,      dc,      D_HEAD)
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
# Agent configuration registry
# ---------------------------------------------------------------------------

def _make_agent(name: str) -> tuple:
    """Return (agent, d_cand, arrived_flag_idx, encode_fn, w_ct, reward_type)."""
    if name == "W11":
        agent = AttentionDQN(d_cand=7, arrived_flag_idx=6)
        return agent, 7, 6, _encode_7dim, 0.2, "composite"
    elif name == "W11b":
        agent = AttentionDQN(d_cand=7, arrived_flag_idx=6)
        return agent, 7, 6, _encode_7dim, 0.6, "composite"
    elif name == "W11c":
        agent = AttentionDQN(d_cand=8, arrived_flag_idx=6)
        return agent, 8, 6, None, 0.6, "composite_noisy"
    elif name == "W11d":
        agent = AttentionDQN(d_cand=6, arrived_flag_idx=5)
        return agent, 6, 5, _encode_6dim, 0.6, "composite"
    elif name == "W12":
        agent = AttentionDQN(d_cand=7, arrived_flag_idx=6)
        return agent, 7, 6, _encode_7dim, 0.0, "pbrs"
    else:
        raise ValueError(f"Unknown agent: {name}")


# ---------------------------------------------------------------------------
# Training loop (one seed)
# ---------------------------------------------------------------------------

def train_seed(agent_name: str, seed: int, sampler) -> tuple:
    """Train one agent for one seed.  Returns (agent, final_mct_100ep)."""

    agent, dc, afi, encode_fn, w_ct, reward_type = _make_agent(agent_name)
    reinit_weights(agent, seed)

    random.seed(seed)
    np.random.seed(seed)
    rng_ep = np.random.default_rng(seed)

    buffer = ReplayBuffer(capacity=BUF_CAPACITY, state_dim=N_PROCESSES * dc)

    total_transitions = 0
    win_loss: list[float] = []
    win_ent:  list[float] = []
    win_mct:  list[float] = []
    win_srpt: list[float] = []

    for ep in range(1, N_EPISODES + 1):
        agent.lambda_ent = LAMBDA_START - (LAMBDA_START - LAMBDA_END) * (ep / N_EPISODES)

        tasks = sampler.sample_episode(rng_ep)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()

        # State encoding
        if reward_type == "composite_noisy":
            sv = _encode_8dim(env, tasks, rng_ep)
        else:
            sv = encode_fn(env, tasks)

        ep_loss_sum  = 0.0
        ep_ent_sum   = 0.0
        ep_loss_n    = 0
        srpt_agree_n = 0
        srpt_total_n = 0
        done         = False

        # For W12 PBRS: track pre-step potential
        if reward_type == "pbrs":
            runnable_pre = [p for p in env.processes
                            if p.arrival_time <= env.current_time and not p.is_complete]
            phi_s = (-PBRS_LAMBDA * max(p.time_since_last_execution
                                        for p in runnable_pre)
                     if runnable_pre else 0.0)
            runnable_snapshot: list[tuple] = [
                (p.wait_time, p.tau, p.floor, p.base_value) for p in runnable_pre
            ]
            chosen_rb_before: float = 0.0

        while not done:
            valid = _valid_actions(env)

            runnable_procs = [p for p in env.processes
                              if p.arrival_time <= env.current_time and not p.is_complete]
            srpt_pid = (min(runnable_procs, key=lambda p: p.remaining_burst).pid
                        if runnable_procs else -1)

            if reward_type == "pbrs":
                runnable_snapshot = [
                    (p.wait_time, p.tau, p.floor, p.base_value) for p in runnable_procs
                ]

            if total_transitions < WARMUP:
                action = random.choice(valid)
            else:
                action = agent.select_action(sv, agent.epsilon, valid)

            chosen_proc = env.processes[action // N_QT]

            if reward_type == "pbrs":
                chosen_rb_before = chosen_proc.remaining_burst

            _, _, done, info = env.step(action)

            # ---- Reward computation ----
            if reward_type in ("composite", "composite_noisy"):
                # env_reward = compute_step_reward (fairness, default weights)
                reward = info.get("env_reward", 0.0) / REWARD_SCALE
                if chosen_proc.is_complete:
                    reward += -w_ct * chosen_proc.completion_time / CT_NORM

            else:  # pbrs (W12)
                q_actual = chosen_rb_before - chosen_proc.remaining_burst
                r_base = sum(
                    value_delta(tau, floor, base_val, wait_before, q_actual)
                    for (wait_before, tau, floor, base_val) in runnable_snapshot
                ) / REWARD_SCALE_W12

                runnable_after = [p for p in env.processes
                                  if p.arrival_time <= env.current_time and not p.is_complete]
                phi_s_next = (-PBRS_LAMBDA * max(p.time_since_last_execution
                                                  for p in runnable_after)
                              if runnable_after else 0.0)
                reward = r_base + (phi_s_next - phi_s)
                phi_s  = phi_s_next

            # ---- Next state ----
            if reward_type == "composite_noisy":
                sv_next = _encode_8dim(env, tasks, rng_ep)
            else:
                sv_next = encode_fn(env, tasks)

            # SRPT tracking
            if srpt_pid >= 0:
                srpt_agree_n += int((action // N_QT) == srpt_pid)
                srpt_total_n += 1

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

        mct       = info.get("mean_completion_time_so_far") or 0.0
        srpt_frac = srpt_agree_n / srpt_total_n if srpt_total_n > 0 else 0.0
        mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else float("nan")
        mean_ent  = ep_ent_sum  / ep_loss_n if ep_loss_n > 0 else float("nan")
        lam_cur   = agent.lambda_ent

        win_mct.append(mct)
        win_srpt.append(srpt_frac)
        win_loss.append(mean_loss)
        win_ent.append(mean_ent)

        agent.decay_epsilon(ep, min_eps=0.05, decay=0.9995)

        if ep % PRINT_EVERY == 0:
            n  = min(ep, 100)
            al = float(np.nanmean(win_loss[-n:]))
            ah = float(np.nanmean(win_ent[-n:]))
            am = float(np.mean(win_mct[-n:]))
            sp = float(np.mean(win_srpt[-n:])) * 100.0
            print(f"  ep {ep:>6} | avg_loss={al:.4f} | avg_H={ah:.4f} | "
                  f"avg_MCT={am:.2f}s | SRPT={sp:.1f}% | lam={lam_cur:.5f}")
            sys.stdout.flush()

            if al > LOSS_GATE:
                print(f"  STOP GATE: loss {al:.2f} > {LOSS_GATE}")
                break
            if ep == 5000 and ah > ENT_GATE_5K:
                print(f"  STOP GATE: H={ah:.4f} > {ENT_GATE_5K} at ep 5000")
                break

    return agent


# ---------------------------------------------------------------------------
# Evaluation (one seed, N episodes, epsilon=0)
# ---------------------------------------------------------------------------

def evaluate_agent(agent_name: str, agent: AttentionDQN, sampler,
                   n_eval: int, master_seed: int = 0) -> dict:
    """Run n_eval greedy episodes from sampler. Returns metrics dict."""
    _, dc, afi, encode_fn, _, reward_type = _make_agent(agent_name)

    rng   = np.random.default_rng(master_seed)
    mcts  = []
    srpts = []
    starved_list = []
    all_turns = []
    all_vlrs  = []

    for _ in range(n_eval):
        tasks = sampler.sample_episode(rng)
        procs = _make_procs(tasks)
        env   = SchedEnv(procs)
        env.reset()

        if reward_type == "composite_noisy":
            sv = _encode_8dim(env, tasks, rng)
        else:
            sv = encode_fn(env, tasks)

        srpt_agree = 0
        srpt_total = 0
        done = False

        while not done:
            valid = _valid_actions(env)
            runnable_procs = [p for p in env.processes
                              if p.arrival_time <= env.current_time and not p.is_complete]
            srpt_pid = (min(runnable_procs, key=lambda p: p.remaining_burst).pid
                        if runnable_procs else -1)

            action = agent.select_action(sv, epsilon=0.0, valid_actions=valid)
            _, _, done, info = env.step(action)

            if reward_type == "composite_noisy":
                sv = _encode_8dim(env, tasks, rng)
            else:
                sv = encode_fn(env, tasks)

            if srpt_pid >= 0:
                srpt_agree += int((action // N_QT) == srpt_pid)
                srpt_total += 1

        mct = info.get("mean_completion_time_so_far") or 0.0
        mcts.append(mct)
        srpts.append(srpt_agree / srpt_total if srpt_total > 0 else 0.0)

        # Starvation flag
        completed = [p for p in env.processes if p.is_complete]
        starved   = 0
        if completed:
            turnarounds = [p.completion_time - p.arrival_time for p in completed]
            bursts_ep   = [p.burst_length for p in completed]
            slowdowns   = [t / max(b, 1e-6) for t, b in zip(turnarounds, bursts_ep)]
            med_slow    = float(np.median(slowdowns))
            if any(s > 3.0 * med_slow for s in slowdowns):
                starved = 1
        starved_list.append(starved)

        # Fairness metrics
        for p in env.processes:
            if p.is_complete:
                T = p.completion_time - p.arrival_time
                all_turns.append(T)
                delay = p.wait_time
                if p.tau > 0:
                    v_now = p.base_value * max(p.floor, math.exp(-delay / p.tau))
                else:
                    v_now = p.base_value
                vlr = (p.base_value - v_now) / max(delay, 1.0)
                all_vlrs.append(vlr)

    def jfi(arr):
        a = np.array(arr, dtype=np.float64)
        return float(a.sum() ** 2 / (len(a) * np.sum(a ** 2) + 1e-12))

    def sdv(arr):
        a = np.array(arr, dtype=np.float64)
        return float(np.std(a) / (np.mean(a) + 1e-12)) if len(a) > 1 else float("nan")

    def vrfi(vlrs):
        a = np.array(vlrs, dtype=np.float64)
        cv = float(np.std(a) / (np.mean(a) + 1e-12)) if len(a) > 1 else 0.0
        return 1.0 - cv

    mct_mean  = float(np.mean(mcts))
    mct_std   = float(np.std(mcts))
    srpt_mean = float(np.mean(srpts)) * 100.0
    srpt_std  = float(np.std(srpts)) * 100.0
    starve_n  = int(np.sum(starved_list))
    starve_pct = starve_n / n_eval * 100.0

    return {
        "mct_mean": mct_mean, "mct_std": mct_std,
        "srpt_mean": srpt_mean, "srpt_std": srpt_std,
        "starve_n": starve_n, "starve_pct": starve_pct,
        "jfi": jfi(all_turns),
        "sdv": sdv(all_turns),
        "vrfi": vrfi(all_vlrs),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

AGENTS = ["W11", "W11b", "W11c", "W11d", "W12"]

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 64)
    print("Ablation Multi-Seed Experiment  (W11 / W11b / W11c / W11d / W12)")
    print(f"Seeds: {SEEDS}    N_episodes: {N_EPISODES}    N_eval: {N_EVAL}")
    print("=" * 64)

    # Load trace samplers
    print(f"\nLoading TRAIN trace: {TRACE_TRAIN}")
    train_sampler = TraceEpisodeSampler5(TRACE_TRAIN)
    print(f"Loading TEST trace:  {TRACE_TEST}")
    test_sampler  = TraceEpisodeSampler5(TRACE_TEST)
    print()

    # Accumulate results: {agent_name: [seed_result_dict, ...]}
    all_results: dict[str, list[dict]] = {a: [] for a in AGENTS}

    for agent_name in AGENTS:
        print(f"\n{'='*64}")
        print(f"AGENT: {agent_name}")
        print(f"{'='*64}")

        _, dc, afi, _, w_ct, rtype = _make_agent(agent_name)
        print(f"  D_CAND={dc}  arrived_flag_idx={afi}  "
              f"reward={rtype}  W_CT={w_ct}")

        for seed in SEEDS:
            print(f"\n--- {agent_name} seed={seed} ---")
            t0 = time.time()
            agent = train_seed(agent_name, seed, train_sampler)
            wall  = (time.time() - t0) / 60.0

            # Save checkpoint
            ckpt_path = os.path.join(RESULTS_DIR, f"{agent_name.lower()}_seed{seed}.npz")
            agent.save(ckpt_path)
            print(f"  Checkpoint → {ckpt_path}")

            # Evaluate on test split
            print(f"  Evaluating {agent_name} seed={seed} "
                  f"(N={N_EVAL} test episodes, epsilon=0)...")
            agent.epsilon = 0.0
            metrics = evaluate_agent(agent_name, agent, test_sampler, N_EVAL,
                                     master_seed=seed)
            metrics["seed"] = seed
            metrics["wall_min"] = wall
            all_results[agent_name].append(metrics)

            print(f"  Seed {seed}: MCT={metrics['mct_mean']:.2f}±{metrics['mct_std']:.2f}s  "
                  f"SRPT={metrics['srpt_mean']:.1f}%  "
                  f"Starve={metrics['starve_pct']:.1f}%  "
                  f"Wall={wall:.1f}min")
            sys.stdout.flush()

    # -------------------------------------------------------------------------
    # Per-agent cross-seed summary
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 64)
    print("Per-Agent Cross-Seed Results")
    print("=" * 64)

    summary: dict[str, dict] = {}

    for agent_name in AGENTS:
        res = all_results[agent_name]
        mct_vals  = [r["mct_mean"]  for r in res]
        srpt_vals = [r["srpt_mean"] for r in res]
        mct_m  = float(np.mean(mct_vals))
        mct_s  = float(np.std(mct_vals))
        srpt_m = float(np.mean(srpt_vals))
        srpt_s = float(np.std(srpt_vals))
        summary[agent_name] = {"mct_mean": mct_m, "mct_std": mct_s,
                                "srpt_mean": srpt_m, "srpt_std": srpt_s}

        print(f"\n{agent_name}:")
        header = f"  {'Seed':>6} | {'MCT (s)':>12} | {'SRPT%':>8} | {'Starve%':>8} | {'Wall(min)':>9}"
        print(header)
        print("  " + "-" * 55)
        for r in res:
            print(f"  {r['seed']:>6} | {r['mct_mean']:>12.2f} | "
                  f"{r['srpt_mean']:>8.1f} | {r['starve_pct']:>8.1f} | "
                  f"{r['wall_min']:>9.1f}")
        print(f"  Mean±std: MCT={mct_m:.2f}±{mct_s:.2f}s  "
              f"SRPT={srpt_m:.1f}±{srpt_s:.1f}%")

    # -------------------------------------------------------------------------
    # Updated 2×2 ablation table
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 64)
    print("Updated 2×2 Ablation Table (2018-trace, mean±std across 3 seeds)")
    print("=" * 64)

    W10C_MCT_ORIG  = 17.23   # original checkpoint N=1500
    W10C_SRPT_ORIG = 71.9

    print(f"\n{'Factor removed':<30} {'Agent':<8} {'MCT (mean±std)':<22} {'SRPT (mean±std)'}")
    print("-" * 80)
    print(f"{'Neither (baseline)':<30} {'W10C':<8} "
          f"{'17.23±11.38s*':<22} {'71.9%*'}")

    W12_s  = summary.get("W12",  {})
    W11d_s = summary.get("W11d", {})
    W11b_s = summary.get("W11b", {})

    for factor, name, s in [
        ("Burst only (value-delta kept)", "W12",  W12_s),
        ("Reward only (burst kept)",      "W11d", W11d_s),
        ("Both removed",                  "W11b", W11b_s),
    ]:
        mct  = f"{s['mct_mean']:.2f}±{s['mct_std']:.2f}s"
        srpt = f"{s['srpt_mean']:.1f}±{s['srpt_std']:.1f}%"
        print(f"{factor:<30} {name:<8} {mct:<22} {srpt}")

    print("\n* W10C: original checkpoint, N=1500 eval episodes")

    # -------------------------------------------------------------------------
    # Full comparison table (all 5 ablation agents)
    # -------------------------------------------------------------------------
    print("\n\n" + "=" * 64)
    print("Full Ablation Comparison Table")
    print("=" * 64)
    print(f"\n{'Agent':<8} {'MCT mean±std':>18} {'SRPT%':>12} {'Description'}")
    print("-" * 72)
    print(f"{'W10C':<8} {'17.23±11.38s*':>18} {'71.9%*':>12}  "
          f"oracle burst + value-delta (N=1500)")
    for aname in AGENTS:
        s = summary[aname]
        _, dc, _, _, w_ct, rtype = _make_agent(aname)
        desc = {
            "W11":  f"no burst, 7-dim, equal composite (W_CT={0.2})",
            "W11b": f"no burst, 7-dim, CT-dominant composite (W_CT={0.6})",
            "W11c": f"no burst+noisy, 8-dim, CT-dominant composite (W_CT={0.6})",
            "W11d": f"true burst, 6-dim, CT-dominant composite (W_CT={0.6})",
            "W12":  f"no burst, 7-dim, value-delta+PBRS",
        }[aname]
        print(f"{aname:<8} {s['mct_mean']:>8.2f}±{s['mct_std']:>5.2f}s "
              f"{s['srpt_mean']:>10.1f}±{s['srpt_std']:>4.1f}%  {desc}")

    # -------------------------------------------------------------------------
    # Save JSON summary
    # -------------------------------------------------------------------------
    import json
    out_path = os.path.join(RESULTS_DIR, "ablation_multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "per_seed": {a: all_results[a] for a in AGENTS},
                "summary":  summary,
            },
            f, indent=2
        )
    print(f"\nResults saved → {out_path}")

    print("\n" + "=" * 64)
    print("Ablation experiment complete.")
    print("=" * 64)


if __name__ == "__main__":
    main()
