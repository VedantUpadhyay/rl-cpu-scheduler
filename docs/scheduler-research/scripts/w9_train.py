"""Week 9 — AttentionDQN with 6-dim per-process state (true burst + 4 fairness + arrived).
Architecture: d_attn=16, a7=7-dim (D_CAND+qt), MLP 23->64->32->1.
"""
from __future__ import annotations
import csv, math, os, random, sys, time
import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")

from schedsim.env    import SchedEnv, N_PROCESSES, N_QUANTUM_TIERS
from schedsim.agent  import AdamOptimizer, ReplayBuffer
from schedsim.process import Process

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
D_CAND   = 6    # per-process feature dim (6: burst + 4 fairness components + arrived_flag)
D_ATTN   = 16   # attention head dim
D_V      = 16   # value dim
D_A6     = D_CAND + 1   # candidate enc (6) + quantum tier (1) = 7
D_MLP_IN = D_V + D_A6   # 23

N_QT      = N_QUANTUM_TIERS   # 3
N_ACTIONS = N_PROCESSES * N_QT   # 15
QT_VALUES = np.array([0.25, 1.0, 4.0])

# ---------------------------------------------------------------------------
# Normalisation constants (filtered trace)
# ---------------------------------------------------------------------------
BURST_P95_FILT = 36.0
_LOG_DENOM     = float(np.log1p(BURST_P95_FILT))   # kept for sampler filter
TIME_NORM      = 500.0                               # normalization for time features
_TIME_LOG_DENOM = float(np.log1p(TIME_NORM))        # ≈ 6.2146
WAIT_NORM      = TIME_NORM                           # 500.0 (updated from 180.0)
CPU_MAX        = 800.0
MEM_P95        = 0.59
REWARD_SCALE   = 1.0    # new reward is already normalized; no additional scaling

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
TRACE_PATH   = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
LOG_PATH     = "/tmp/w9_train.log"
WEIGHTS_PATH = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                "GRAD - FALL 23/UCSC/Capstone/results/dqn_w9.npz")

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

PRINT_AT   = {500, 1000, 2000, 5000, 10000}
LOSS_GATE  = 100.0
ENT_GATE_5K = 1.2
ENT_FLAG_1K = 0.20


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_burst(b: float) -> float:
    """Kept for trace sampler filter and permutation tests."""
    return float(np.log1p(b) / _LOG_DENOM)

def _norm_time_log(t: float) -> float:
    """Log-normalize a time duration against TIME_NORM=500s reference."""
    return float(np.log1p(max(t, 0.0)) / _TIME_LOG_DENOM)

def _urgency_norm(p) -> float:
    """Value loss rate (VLR) normalized by 0.1. Zero for tau<=0 or delay<=0."""
    delay = p.wait_time
    if p.tau <= 0.0 or delay <= 0.0:
        return 0.0
    v0    = p.base_value
    v_now = v0 * max(p.floor, math.exp(-delay / p.tau))
    vlr   = (v0 - v_now) / delay
    return float(vlr / 0.1)

def _norm_cpu(c: float) -> float:
    return float(c / CPU_MAX)

def _norm_mem(m: float) -> float:
    return float(min(m / MEM_P95, 1.0))


# ---------------------------------------------------------------------------
# Trace sampler — loads burst, plan_cpu, plan_mem
# ---------------------------------------------------------------------------

class TraceEpisodeSampler5:
    """Sample 5-task episodes from trace; returns burst + plan_cpu + plan_mem."""
    _ARRIVE_SLOTS = (0, 2, 5, 8, 10)

    def __init__(self, path: str) -> None:
        records = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    st = float(row["start_time"])
                    et = float(row["end_time"])
                    dur = et - st
                    cpu = float(row["plan_cpu"])
                    mem = float(row["plan_mem"])
                    if dur > 0 and dur <= BURST_P95_FILT * 1.1:
                        records.append((dur, cpu, mem))
                except (ValueError, TypeError):
                    pass
        self._data = np.array(records, dtype=np.float32)  # (N, 3)
        print(f"  TraceEpisodeSampler5: {len(self._data):,} tasks loaded.")

    def __len__(self) -> int:
        return len(self._data)

    def sample_episode(self, rng: np.random.Generator) -> list[dict]:
        idx    = rng.integers(0, len(self._data), size=N_PROCESSES)
        tasks  = self._data[idx]                 # (5, 3)
        order  = rng.permutation(N_PROCESSES)
        slots  = self._ARRIVE_SLOTS
        return [
            {
                "burst_ms":    float(tasks[i, 0]),
                "arrival_ms":  float(slots[order[k]]),
                "plan_cpu":    float(tasks[i, 1]),
                "plan_mem":    float(tasks[i, 2]),
            }
            for k, i in enumerate(range(N_PROCESSES))
        ]


# ---------------------------------------------------------------------------
# State encoding — 30-dim  (6 features × 5 processes)
# Per-process slot i*6 : (i+1)*6 :
#   off+0  remaining_burst_norm    log1p(remaining_burst) / log1p(36.0)  [TRUE burst]
#   off+1  wait_norm               wait_time / 500.0
#   off+2  time_since_last_exec_norm  log1p(time_since_last_execution) / log1p(500)
#   off+3  urgency_norm            VLR / 0.1  (value loss rate)
#   off+4  time_in_queue_norm      log1p(current_time - arrival_time) / log1p(500)
#   off+5  arrived_flag            float(arrival_time <= current_time)
# Completed or not-yet-arrived processes: all zeros (arrived_flag=0).
# ---------------------------------------------------------------------------

def _encode_state(env: SchedEnv, task_meta: list[dict]) -> np.ndarray:
    vec = np.zeros(N_PROCESSES * D_CAND, dtype=np.float32)
    for p in env.processes:
        i   = p.pid
        off = i * D_CAND
        if p.is_complete:
            pass  # all zeros — arrived_flag=0 marks as invalid for attention
        else:
            arrived = p.arrival_time <= env.current_time
            if arrived:
                vec[off + 0] = float(np.log1p(max(p.remaining_burst, 0.0)) / _LOG_DENOM)
                vec[off + 1] = p.wait_time / WAIT_NORM
                vec[off + 2] = _norm_time_log(p.time_since_last_execution)
                vec[off + 3] = _urgency_norm(p)
                vec[off + 4] = _norm_time_log(env.current_time - p.arrival_time)
                vec[off + 5] = 1.0
    return vec


def _valid_actions(env: SchedEnv) -> list[int]:
    return [
        p.pid * N_QT + qt
        for p in env.processes
        for qt in range(N_QT)
        if p.arrival_time <= env.current_time and not p.is_complete
    ]


def _make_procs(tasks: list[dict]) -> list[Process]:
    return [
        Process(pid=i, arrival_time=task["arrival_ms"], burst_length=task["burst_ms"])
        for i, task in enumerate(tasks)
    ]


# ---------------------------------------------------------------------------
# AttentionDQN9 — full training-capable implementation
# ---------------------------------------------------------------------------

class AttentionDQN9:
    """Week 9: 5-dim per-process state, d_attn=16, a6=6, MLP 22→64→32→1."""

    def __init__(self, lr: float = 0.001, gamma: float = 1.0,
                 grad_clip: float = 1.0, lambda_ent: float = 0.10) -> None:
        self.gamma      = gamma
        self.grad_clip  = grad_clip
        self.lambda_ent = lambda_ent
        self.epsilon    = 1.0

        rng = np.random.default_rng(42)
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

        # Target network copies
        self._copy_attn_target()
        self._tW = [w.copy() for w in self._W]
        self._tb = [b.copy() for b in self._b]

        self._opt = AdamOptimizer(lr=lr)

    def _copy_attn_target(self) -> None:
        self._tW_Q = self.W_Q.copy(); self._tb_Q = self.b_Q.copy()
        self._tW_K = self.W_K.copy(); self._tb_K = self.b_K.copy()
        self._tW_V = self.W_V.copy(); self._tb_V = self.b_V.copy()

    def update_target(self) -> None:
        self._copy_attn_target()
        self._tW = [w.copy() for w in self._W]
        self._tb = [b.copy() for b in self._b]

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
        comp_encs  = s3d[bidx, comp_idx]                           # (batch,4,6)
        comp_valid = (comp_encs[:, :, 5] > 0.5)                    # arrived_flag at off+5
        return comp_encs, comp_valid

    def _attention_forward(
        self,
        cand_enc:   np.ndarray,  # (batch, 6)
        comp_encs:  np.ndarray,  # (batch, 4, 6)
        comp_valid: np.ndarray,  # (batch, 4) bool
        use_target: bool = False,
    ) -> tuple[np.ndarray, dict]:
        WQ = self._tW_Q if use_target else self.W_Q
        bQ = self._tb_Q if use_target else self.b_Q
        WK = self._tW_K if use_target else self.W_K
        bK = self._tb_K if use_target else self.b_K
        WV = self._tW_V if use_target else self.W_V
        bV = self._tb_V if use_target else self.b_V

        batch     = cand_enc.shape[0]
        comp_flat = comp_encs.reshape(batch * 4, D_CAND)

        q      = cand_enc @ WQ + bQ                                 # (batch, 16)
        K_flat = comp_flat @ WK + bK
        V_flat = comp_flat @ WV + bV
        K = K_flat.reshape(batch, 4, D_ATTN)
        V = V_flat.reshape(batch, 4, D_V)

        scores = np.einsum('bi,bji->bj', q, K) / np.sqrt(D_ATTN)   # (batch, 4)

        invalid       = ~comp_valid
        scores_masked = scores.copy()
        scores_masked[invalid] = -1e9
        scores_shifted = scores_masked - scores_masked.max(axis=1, keepdims=True)
        exp_s          = np.exp(scores_shifted)
        exp_s[invalid] = 0.0
        denom          = exp_s.sum(axis=1, keepdims=True) + 1e-10
        weights        = exp_s / denom

        context = np.einsum('bj,bjd->bd', weights, V)               # (batch, 16)
        cache = dict(
            cand_enc=cand_enc, comp_encs=comp_encs, comp_flat=comp_flat,
            comp_valid=comp_valid, invalid=invalid,
            q=q, K=K, V=V, scores=scores, weights=weights, context=context,
        )
        return context, cache

    def forward_batch(
        self,
        states:     np.ndarray,  # (batch, 30)
        actions:    np.ndarray,  # (batch,) int
        use_target: bool = False,
    ) -> np.ndarray:             # (batch, 1)
        batch    = states.shape[0]
        pids     = (actions // N_QT).astype(np.int32)
        qts      = QT_VALUES[actions % N_QT]                        # (batch,)

        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        cand_enc = s3d[np.arange(batch), pids]                      # (batch, 6)
        comp_encs, comp_valid = self._build_competitor_data(states, pids)
        context, _ = self._attention_forward(cand_enc, comp_encs, comp_valid, use_target)

        a6 = np.column_stack([cand_enc, qts / 2.0])                 # (batch, 7)
        x  = np.concatenate([context, a6], axis=1)                  # (batch, 23)

        W, b = (self._tW, self._tb) if use_target else (self._W, self._b)
        z1 = x  @ W[0] + b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1 @ W[1] + b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2 @ W[2] + b[2]
        return z3                                                    # (batch, 1)

    def select_action(
        self,
        state:         np.ndarray,  # (25,)
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
        states:      np.ndarray,  # (batch, 35)
        actions:     np.ndarray,  # (batch,) int
        rewards:     np.ndarray,  # (batch,)
        next_states: np.ndarray,  # (batch, 30)
        dones:       np.ndarray,  # (batch,) float
    ) -> tuple[float, float]:
        batch = states.shape[0]
        pids  = (actions // N_QT).astype(np.int32)
        qts   = QT_VALUES[actions % N_QT]

        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        cand_enc = s3d[np.arange(batch), pids]                      # (batch, 6)
        comp_encs, comp_valid = self._build_competitor_data(states, pids)

        # Forward (online)
        context, attn_cache = self._attention_forward(cand_enc, comp_encs, comp_valid)
        a6    = np.column_stack([cand_enc, qts / 2.0])              # (batch, 7)
        x_mlp = np.concatenate([context, a6], axis=1)               # (batch, 23)

        z1 = x_mlp @ self._W[0] + self._b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1    @ self._W[1] + self._b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2    @ self._W[2] + self._b[2]
        q_pred = z3.flatten()

        # Entropy
        weights  = attn_cache['weights']
        valid_f  = comp_valid.astype(np.float64)
        ent_per  = -(weights * np.log(weights + 1e-8) * valid_f).sum(axis=1)
        H_batch  = float(np.mean(ent_per))

        # Target Q (all 15 actions, target network)
        ns          = np.asarray(next_states, dtype=np.float64)
        all_acts    = np.tile(np.arange(N_ACTIONS), batch)
        all_ns      = np.repeat(ns, N_ACTIONS, axis=0)
        Q_next_flat = self.forward_batch(all_ns, all_acts, use_target=True).flatten()
        Q_next_mat  = Q_next_flat.reshape(batch, N_ACTIONS)

        # Valid-action mask in next state (arrived_flag at off+5)
        valid_mask = np.zeros((batch, N_ACTIONS), dtype=bool)
        for pid in range(N_PROCESSES):
            runnable = (ns[:, pid * D_CAND + 5] > 0.5)
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

        # MLP backward
        dz3    = (2.0 * delta / batch).reshape(-1, 1)
        dW2    = h2.T @ dz3;     db2 = dz3.sum(axis=0)
        dh2    = dz3 @ self._W[2].T
        dz2    = dh2 * (h2 > 0).astype(np.float64)
        dW1    = h1.T @ dz2;     db1 = dz2.sum(axis=0)
        dh1    = dz2 @ self._W[1].T
        dz1    = dh1 * (h1 > 0).astype(np.float64)
        dW0    = x_mlp.T @ dz1;  db0 = dz1.sum(axis=0)
        dx_mlp = dz1 @ self._W[0].T                                 # (batch, 22)

        d_context = dx_mlp[:, :D_V]                                 # (batch, 16)

        # Attention backward
        q_attn    = attn_cache['q']
        K         = attn_cache['K']
        V         = attn_cache['V']
        invalid   = attn_cache['invalid']
        comp_flat = attn_cache['comp_flat']

        d_weights = np.einsum('bd,bjd->bj', d_context, V)           # (batch, 4)
        dV        = weights[:, :, None] * d_context[:, None, :]     # (batch, 4, 16)
        dV[invalid] = 0.0

        if self.lambda_ent > 0:
            d_ent     = (self.lambda_ent / batch) * (
                np.log(weights + 1e-8) + weights / (weights + 1e-8)
            ) * valid_f
            d_weights = d_weights + d_ent

        wdotdw          = (weights * d_weights).sum(axis=1, keepdims=True)
        d_scores_masked = weights * (d_weights - wdotdw)
        d_scores_masked[invalid] = 0.0
        d_raw_scores = d_scores_masked / np.sqrt(D_ATTN)

        dq = np.einsum('bj,bji->bi', d_raw_scores, K)               # (batch, 16)
        dK = d_raw_scores[:, :, None] * q_attn[:, None, :]          # (batch, 4, 16)
        dK[invalid] = 0.0

        dW_Q  = cand_enc.T @ dq                                     # (5, 16)
        db_Q  = dq.sum(axis=0)
        dK_flat = dK.reshape(batch * 4, D_ATTN)
        dW_K  = comp_flat.T @ dK_flat                               # (5, 16)
        db_K  = dK_flat.sum(axis=0)
        dV_flat = dV.reshape(batch * 4, D_V)
        dW_V  = comp_flat.T @ dV_flat                               # (5, 16)
        db_V  = dV_flat.sum(axis=0)

        # Grad norm clipping
        all_grads = [dW0, db0, dW1, db1, dW2, db2,
                     dW_Q, db_Q, dW_K, db_K, dW_V, db_V]
        global_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in all_grads)))
        if global_norm > self.grad_clip:
            scale     = self.grad_clip / global_norm
            all_grads = [g * scale for g in all_grads]

        all_params = [self._W[0], self._b[0], self._W[1], self._b[1],
                      self._W[2], self._b[2],
                      self.W_Q, self.b_Q, self.W_K, self.b_K, self.W_V, self.b_V]
        self._opt.step(all_params, all_grads)
        return loss, H_batch

    def decay_epsilon(self, ep: int, min_eps: float = 0.05, decay: float = 0.9995) -> float:
        self.epsilon = max(min_eps, 1.0 * (decay ** ep))
        return self.epsilon

    def save(self, path: str) -> None:
        np.savez(path,
                 W_Q=self.W_Q, b_Q=self.b_Q,
                 W_K=self.W_K, b_K=self.b_K,
                 W_V=self.W_V, b_V=self.b_V,
                 W0=self._W[0], b0=self._b[0],
                 W1=self._W[1], b1=self._b[1],
                 W2=self._W[2], b2=self._b[2])

    def load(self, path: str) -> None:
        d = np.load(path)
        self.W_Q = d['W_Q']; self.b_Q = d['b_Q']
        self.W_K = d['W_K']; self.b_K = d['b_K']
        self.W_V = d['W_V']; self.b_V = d['b_V']
        self._W  = [d['W0'], d['W1'], d['W2']]
        self._b  = [d['b0'], d['b1'], d['b2']]
        self.update_target()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> AttentionDQN9:
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    print(f"Loading trace: {TRACE_PATH}")
    sampler = TraceEpisodeSampler5(TRACE_PATH)

    # Verify normalization
    print("\nNormalisation check:")
    for cpu in [50, 100, 400, 800]:
        print(f"  plan_cpu={cpu:>4}  →  {_norm_cpu(cpu):.4f}")
    for mem in [0.20, 0.39, 0.59]:
        print(f"  plan_mem={mem:.2f}  →  {_norm_mem(mem):.4f}")

    agent  = AttentionDQN9(lr=LR, gamma=GAMMA, grad_clip=GRAD_CLIP)
    buffer = ReplayBuffer(capacity=BUF_CAPACITY, state_dim=N_PROCESSES * D_CAND)

    attn_params = (D_CAND * D_ATTN + D_ATTN) * 3
    mlp_params  = (D_MLP_IN*64 + 64) + (64*32 + 32) + (32*1 + 1)
    print(f"\nParameter count: {attn_params + mlp_params} total "
          f"(attn={attn_params}, mlp={mlp_params})\n")

    rng = np.random.default_rng(42)
    total_transitions = 0
    win_loss:    list[float] = []
    win_entropy: list[float] = []
    win_mct:     list[float] = []

    with open(LOG_PATH, "w", newline="") as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow(["episode", "avg_loss", "avg_H", "avg_MCT", "lambda_ent"])

        for ep in range(1, N_EPISODES + 1):
            agent.lambda_ent = LAMBDA_START - (LAMBDA_START - LAMBDA_END) * (ep / N_EPISODES)

            tasks = sampler.sample_episode(rng)
            procs = _make_procs(tasks)
            env   = SchedEnv(procs)
            env.reset()
            sv    = _encode_state(env, tasks)

            ep_loss_sum = 0.0
            ep_ent_sum  = 0.0
            ep_loss_n   = 0
            done        = False

            while not done:
                valid = _valid_actions(env)
                if total_transitions < WARMUP:
                    action = random.choice(valid)
                else:
                    action = agent.select_action(sv, agent.epsilon, valid)

                _, _, done, info = env.step(action)
                reward   = info.get("env_reward", 0.0) / REWARD_SCALE
                sv_next  = _encode_state(env, tasks)

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
            mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else float("nan")
            mean_ent  = ep_ent_sum  / ep_loss_n if ep_loss_n > 0 else float("nan")
            lam_cur   = agent.lambda_ent

            agent.decay_epsilon(ep, min_eps=0.05, decay=0.9995)

            win_loss.append(mean_loss)
            win_entropy.append(mean_ent)
            win_mct.append(mct)

            log_writer.writerow([ep, f"{mean_loss:.6f}", f"{mean_ent:.6f}",
                                 f"{mct:.4f}", f"{lam_cur:.5f}"])

            if ep in PRINT_AT:
                n  = ep
                al = float(np.nanmean(win_loss[-min(n, 500):]))
                ah = float(np.nanmean(win_entropy[-min(n, 500):]))
                am = float(np.mean(win_mct[-min(n, 500):]))

                print(f"ep {ep:>6} | avg_loss={al:.4f} | avg_H={ah:.4f} | "
                      f"avg_MCT={am:.2f}s | lambda_ent={lam_cur:.5f}")
                sys.stdout.flush()
                log_f.flush()

                if al > LOSS_GATE:
                    print(f"\nSTOP GATE: avg_loss={al:.2f} > {LOSS_GATE} at ep {ep}")
                    return agent
                if ep == 5000 and ah > ENT_GATE_5K:
                    print(f"\nSTOP GATE: avg_H={ah:.4f} > {ENT_GATE_5K} at ep 5000")
                    return agent
                if ep == 1000 and ah < ENT_FLAG_1K:
                    print(f"\nFLAG: avg_H={ah:.4f} < {ENT_FLAG_1K} at ep 1000 (attention may be collapsing)")

    print(f"\nTraining complete — {N_EPISODES} episodes.")
    return agent


if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    print("=" * 64)
    print("Week 9 — AttentionDQN, 5-dim state (+ plan_cpu, plan_mem)")
    print("=" * 64)
    t0    = time.time()
    agent = train()
    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
    agent.save(WEIGHTS_PATH)
    print(f"Weights saved → {WEIGHTS_PATH}")
