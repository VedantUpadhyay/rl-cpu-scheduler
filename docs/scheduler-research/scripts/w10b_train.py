"""Week 10B — W9 + learnable attention temperature (log_beta).
No PBRS. Only change from W9: fixed 1/sqrt(d) replaced by exp(log_beta).
"""
from __future__ import annotations
import csv, math, os, random, sys, time
import numpy as np

sys.path.insert(0, "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/GRAD - FALL 23/UCSC/Capstone")
sys.path.insert(0, "/tmp")

from schedsim.env    import SchedEnv, N_PROCESSES, N_QUANTUM_TIERS
from schedsim.agent  import AdamOptimizer, ReplayBuffer
from schedsim.process import Process
from w9_train import (
    TraceEpisodeSampler5,
    _encode_state, _valid_actions, _make_procs,
    D_CAND, D_ATTN, D_V, D_A6, D_MLP_IN, N_QT, N_ACTIONS, QT_VALUES,
    _LOG_DENOM, WAIT_NORM, _norm_burst, _norm_cpu, _norm_mem,
    REWARD_SCALE,
)

# ---------------------------------------------------------------------------
# Paths and hyperparameters
# ---------------------------------------------------------------------------
TRACE_PATH   = "/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv"
LOG_PATH     = "/tmp/w10b_train.log"
WEIGHTS_PATH = ("/Users/vedantupadhyay/Library/CloudStorage/OneDrive-Personal/"
                "GRAD - FALL 23/UCSC/Capstone/results/dqn_w10b.npz")

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
PRINT_AT           = {500, 1000, 2000, 5000, 10000}
LOSS_GATE          = 100.0
ENT_GATE_5K        = 1.2

# Initial log_beta: matches fixed 1/sqrt(16) = 0.25
LOG_BETA_INIT = float(math.log(1.0 / math.sqrt(D_ATTN)))   # ≈ -1.3863


# ---------------------------------------------------------------------------
# AttentionDQN10B — W9 + learnable beta = exp(log_beta)
# ---------------------------------------------------------------------------

class AttentionDQN10B:
    """Week 10B: learnable attention temperature beta = exp(log_beta).
    Initialised to 1/sqrt(16) = 0.25 (same as W9 fixed scaling).
    3,874 parameters (3,873 + 1).
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

        # Learnable log_beta — stored as (1,) array for Adam
        self.log_beta = np.array([LOG_BETA_INIT], dtype=np.float64)

        # Attention projections: 5 → 16
        self.W_Q = he(D_CAND, D_CAND, D_ATTN);  self.b_Q = np.zeros(D_ATTN)
        self.W_K = he(D_CAND, D_CAND, D_ATTN);  self.b_K = np.zeros(D_ATTN)
        self.W_V = he(D_CAND, D_CAND, D_V);      self.b_V = np.zeros(D_V)

        # MLP: 22 → 64 → 32 → 1
        self._W = [he(D_MLP_IN, D_MLP_IN, 64),
                   he(64,       64,        32),
                   he(32,       32,         1)]
        self._b = [np.zeros(64), np.zeros(32), np.zeros(1)]

        self._copy_attn_target()
        self._tW = [w.copy() for w in self._W]
        self._tb = [b.copy() for b in self._b]

        self._opt = AdamOptimizer(lr=lr)

    def beta(self) -> float:
        return float(np.exp(self.log_beta[0]))

    def _copy_attn_target(self) -> None:
        self._tW_Q = self.W_Q.copy(); self._tb_Q = self.b_Q.copy()
        self._tW_K = self.W_K.copy(); self._tb_K = self.b_K.copy()
        self._tW_V = self.W_V.copy(); self._tb_V = self.b_V.copy()
        self._t_log_beta = self.log_beta.copy()

    def update_target(self) -> None:
        self._copy_attn_target()
        self._tW = [w.copy() for w in self._W]
        self._tb = [b.copy() for b in self._b]

    def _build_competitor_data(
        self, states: np.ndarray, pids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        batch    = states.shape[0]
        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        all_pids = np.tile(np.arange(N_PROCESSES), (batch, 1))
        mask     = all_pids != pids[:, None]
        comp_idx = all_pids[mask].reshape(batch, 4)
        bidx     = np.arange(batch)[:, None]
        comp_encs  = s3d[bidx, comp_idx]
        comp_valid = (comp_encs[:, :, 1] > 0.5) & (comp_encs[:, :, 0] > 1e-6)
        return comp_encs, comp_valid

    def _attention_forward(
        self,
        cand_enc:   np.ndarray,  # (batch, 5)
        comp_encs:  np.ndarray,  # (batch, 4, 5)
        comp_valid: np.ndarray,  # (batch, 4) bool
        use_target: bool = False,
    ) -> tuple[np.ndarray, dict]:
        WQ = self._tW_Q if use_target else self.W_Q
        bQ = self._tb_Q if use_target else self.b_Q
        WK = self._tW_K if use_target else self.W_K
        bK = self._tb_K if use_target else self.b_K
        WV = self._tW_V if use_target else self.W_V
        bV = self._tb_V if use_target else self.b_V
        lb = self._t_log_beta if use_target else self.log_beta
        beta_val = float(np.exp(lb[0]))

        batch     = cand_enc.shape[0]
        comp_flat = comp_encs.reshape(batch * 4, D_CAND)

        q      = cand_enc @ WQ + bQ
        K_flat = comp_flat @ WK + bK
        V_flat = comp_flat @ WV + bV
        K = K_flat.reshape(batch, 4, D_ATTN)
        V = V_flat.reshape(batch, 4, D_V)

        # Learnable scaling: beta * (q · K_j)
        raw_dot = np.einsum('bi,bji->bj', q, K)                    # (batch, 4)
        scores  = beta_val * raw_dot                                # (batch, 4)

        invalid       = ~comp_valid
        scores_masked = scores.copy()
        scores_masked[invalid] = -1e9
        scores_shifted = scores_masked - scores_masked.max(axis=1, keepdims=True)
        exp_s          = np.exp(scores_shifted)
        exp_s[invalid] = 0.0
        denom          = exp_s.sum(axis=1, keepdims=True) + 1e-10
        weights        = exp_s / denom

        context = np.einsum('bj,bjd->bd', weights, V)
        cache = dict(
            cand_enc=cand_enc, comp_encs=comp_encs, comp_flat=comp_flat,
            comp_valid=comp_valid, invalid=invalid,
            q=q, K=K, V=V, raw_dot=raw_dot, scores=scores,
            weights=weights, context=context,
        )
        return context, cache

    def forward_batch(
        self,
        states:     np.ndarray,
        actions:    np.ndarray,
        use_target: bool = False,
    ) -> np.ndarray:
        batch    = states.shape[0]
        pids     = (actions // N_QT).astype(np.int32)
        qts      = QT_VALUES[actions % N_QT]

        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        cand_enc = s3d[np.arange(batch), pids]
        comp_encs, comp_valid = self._build_competitor_data(states, pids)
        context, _ = self._attention_forward(cand_enc, comp_encs, comp_valid, use_target)

        a6 = np.column_stack([cand_enc, qts / 2.0])
        x  = np.concatenate([context, a6], axis=1)

        W, b = (self._tW, self._tb) if use_target else (self._W, self._b)
        z1 = x  @ W[0] + b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1 @ W[1] + b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2 @ W[2] + b[2]
        return z3

    def select_action(self, state, epsilon, valid_actions):
        if np.random.random() < epsilon:
            return int(np.random.choice(valid_actions))
        states_b  = np.tile(state, (len(valid_actions), 1)).astype(np.float64)
        actions_b = np.array(valid_actions, dtype=np.int32)
        q_vals    = self.forward_batch(states_b, actions_b).flatten()
        return valid_actions[int(np.argmax(q_vals))]

    def update_online(self, states, actions, rewards, next_states, dones):
        batch    = states.shape[0]
        pids     = (actions // N_QT).astype(np.int32)
        qts      = QT_VALUES[actions % N_QT]
        beta_val = float(np.exp(self.log_beta[0]))

        s3d      = states.reshape(batch, N_PROCESSES, D_CAND)
        cand_enc = s3d[np.arange(batch), pids]
        comp_encs, comp_valid = self._build_competitor_data(states, pids)

        context, attn_cache = self._attention_forward(cand_enc, comp_encs, comp_valid)
        a6    = np.column_stack([cand_enc, qts / 2.0])
        x_mlp = np.concatenate([context, a6], axis=1)

        z1 = x_mlp @ self._W[0] + self._b[0]; h1 = np.maximum(0.0, z1)
        z2 = h1    @ self._W[1] + self._b[1]; h2 = np.maximum(0.0, z2)
        z3 = h2    @ self._W[2] + self._b[2]
        q_pred = z3.flatten()

        weights  = attn_cache['weights']
        valid_f  = comp_valid.astype(np.float64)
        ent_per  = -(weights * np.log(weights + 1e-8) * valid_f).sum(axis=1)
        H_batch  = float(np.mean(ent_per))

        # Target Q
        ns          = np.asarray(next_states, dtype=np.float64)
        all_acts    = np.tile(np.arange(N_ACTIONS), batch)
        all_ns      = np.repeat(ns, N_ACTIONS, axis=0)
        Q_next_flat = self.forward_batch(all_ns, all_acts, use_target=True).flatten()
        Q_next_mat  = Q_next_flat.reshape(batch, N_ACTIONS)

        valid_mask = np.zeros((batch, N_ACTIONS), dtype=bool)
        for pid in range(N_PROCESSES):
            runnable = (ns[:, pid * D_CAND + 1] > 0.5) & (ns[:, pid * D_CAND + 0] > 1e-6)
            for qt in range(N_QT):
                valid_mask[:, pid * N_QT + qt] = runnable

        Q_next_mat[~valid_mask] = -np.inf
        all_invalid = ~np.any(valid_mask, axis=1)
        max_q_next  = np.where(all_invalid, 0.0, Q_next_mat.max(axis=1))
        targets = rewards + self.gamma * max_q_next * (1.0 - dones)

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
        dx_mlp = dz1 @ self._W[0].T

        d_context = dx_mlp[:, :D_V]

        # Attention backward
        q_attn    = attn_cache['q']
        K         = attn_cache['K']
        V         = attn_cache['V']
        raw_dot   = attn_cache['raw_dot']
        invalid   = attn_cache['invalid']
        comp_flat = attn_cache['comp_flat']

        d_weights = np.einsum('bd,bjd->bj', d_context, V)
        dV        = weights[:, :, None] * d_context[:, None, :]
        dV[invalid] = 0.0

        if self.lambda_ent > 0:
            d_ent     = (self.lambda_ent / batch) * (
                np.log(weights + 1e-8) + weights / (weights + 1e-8)
            ) * valid_f
            d_weights = d_weights + d_ent

        wdotdw          = (weights * d_weights).sum(axis=1, keepdims=True)
        d_scores_masked = weights * (d_weights - wdotdw)
        d_scores_masked[invalid] = 0.0

        # d_raw_scores = d_scores_masked * beta  (since scores = beta * raw_dot)
        # But we need d_scores w.r.t. raw_dot → multiply by beta
        # And gradient for log_beta: d_loss/d_beta = sum(d_scores_masked * raw_dot)
        #                            d_loss/d_log_beta = d_loss/d_beta * beta
        d_beta       = float(np.sum(d_scores_masked * raw_dot))
        d_log_beta   = np.array([d_beta * beta_val], dtype=np.float64)

        d_raw_scores = d_scores_masked * beta_val                   # (batch, 4)

        dq = np.einsum('bj,bji->bi', d_raw_scores, K)
        dK = d_raw_scores[:, :, None] * q_attn[:, None, :]
        dK[invalid] = 0.0

        dW_Q  = cand_enc.T @ dq
        db_Q  = dq.sum(axis=0)
        dK_flat = dK.reshape(batch * 4, D_ATTN)
        dW_K  = comp_flat.T @ dK_flat
        db_K  = dK_flat.sum(axis=0)
        dV_flat = dV.reshape(batch * 4, D_V)
        dW_V  = comp_flat.T @ dV_flat
        db_V  = dV_flat.sum(axis=0)

        # Grad norm clipping — include log_beta gradient
        all_grads = [dW0, db0, dW1, db1, dW2, db2,
                     dW_Q, db_Q, dW_K, db_K, dW_V, db_V, d_log_beta]
        global_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in all_grads)))
        if global_norm > self.grad_clip:
            scale     = self.grad_clip / global_norm
            all_grads = [g * scale for g in all_grads]

        all_params = [self._W[0], self._b[0], self._W[1], self._b[1],
                      self._W[2], self._b[2],
                      self.W_Q, self.b_Q, self.W_K, self.b_K, self.W_V, self.b_V,
                      self.log_beta]
        self._opt.step(all_params, all_grads)
        return loss, H_batch

    def decay_epsilon(self, ep, min_eps=0.05, decay=0.9995):
        self.epsilon = max(min_eps, 1.0 * (decay ** ep))
        return self.epsilon

    def save(self, path: str) -> None:
        np.savez(path,
                 log_beta=self.log_beta,
                 W_Q=self.W_Q, b_Q=self.b_Q,
                 W_K=self.W_K, b_K=self.b_K,
                 W_V=self.W_V, b_V=self.b_V,
                 W0=self._W[0], b0=self._b[0],
                 W1=self._W[1], b1=self._b[1],
                 W2=self._W[2], b2=self._b[2])

    def load(self, path: str) -> None:
        d = np.load(path)
        self.log_beta = d['log_beta']
        self.W_Q = d['W_Q']; self.b_Q = d['b_Q']
        self.W_K = d['W_K']; self.b_K = d['b_K']
        self.W_V = d['W_V']; self.b_V = d['b_V']
        self._W  = [d['W0'], d['W1'], d['W2']]
        self._b  = [d['b0'], d['b1'], d['b2']]
        self.update_target()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> AttentionDQN10B:
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    print(f"Loading trace: {TRACE_PATH}")
    sampler = TraceEpisodeSampler5(TRACE_PATH)

    agent  = AttentionDQN10B(lr=LR, gamma=GAMMA, grad_clip=GRAD_CLIP)
    buffer = ReplayBuffer(capacity=BUF_CAPACITY, state_dim=N_PROCESSES * D_CAND)

    attn_p = (D_CAND * D_ATTN + D_ATTN) * 3
    mlp_p  = (D_MLP_IN*64 + 64) + (64*32 + 32) + (32*1 + 1)
    total_p = attn_p + mlp_p + 1  # +1 for log_beta
    print(f"\nParameter count: {total_p} total (attn={attn_p}, mlp={mlp_p}, log_beta=1)")
    print(f"Initial log_beta = {agent.log_beta[0]:.6f}")
    print(f"Initial beta     = {agent.beta():.6f}  (target: {1.0/math.sqrt(D_ATTN):.6f})")
    print()

    rng = np.random.default_rng(42)
    total_transitions = 0
    win_loss:    list[float] = []
    win_entropy: list[float] = []
    win_mct:     list[float] = []

    with open(LOG_PATH, "w", newline="") as log_f:
        log_writer = csv.writer(log_f)
        log_writer.writerow(["episode", "avg_loss", "avg_H", "avg_MCT", "lambda_ent", "beta"])

        for ep in range(1, N_EPISODES + 1):
            agent.lambda_ent = LAMBDA_START - (LAMBDA_START - LAMBDA_END) * (ep / N_EPISODES)

            tasks = sampler.sample_episode(rng)
            procs = _make_procs(tasks)
            env   = SchedEnv(procs); env.reset()
            sv    = _encode_state(env, tasks)

            ep_loss_sum = 0.0; ep_ent_sum = 0.0; ep_loss_n = 0
            done = False

            while not done:
                valid = _valid_actions(env)
                if total_transitions < WARMUP:
                    action = random.choice(valid)
                else:
                    action = agent.select_action(sv, agent.epsilon, valid)

                _, _, done, info = env.step(action)
                reward  = info.get("env_reward", 0.0) / REWARD_SCALE
                sv_next = _encode_state(env, tasks)

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
                    ep_loss_sum += loss; ep_ent_sum += ent; ep_loss_n += 1

                sv = sv_next

            if ep % TARGET_UPDATE_FREQ == 0:
                agent.update_target()

            mct       = info.get("mean_completion_time_so_far") or 0.0
            mean_loss = ep_loss_sum / ep_loss_n if ep_loss_n > 0 else float("nan")
            mean_ent  = ep_ent_sum  / ep_loss_n if ep_loss_n > 0 else float("nan")
            lam_cur   = agent.lambda_ent
            beta_cur  = agent.beta()

            agent.decay_epsilon(ep, min_eps=0.05, decay=0.9995)
            win_loss.append(mean_loss); win_entropy.append(mean_ent); win_mct.append(mct)

            log_writer.writerow([ep, f"{mean_loss:.6f}", f"{mean_ent:.6f}",
                                 f"{mct:.4f}", f"{lam_cur:.5f}", f"{beta_cur:.6f}"])

            if ep in PRINT_AT:
                n  = min(ep, 500)
                al = float(np.nanmean(win_loss[-n:]))
                ah = float(np.nanmean(win_entropy[-n:]))
                am = float(np.mean(win_mct[-n:]))
                print(f"ep {ep:>6} | avg_loss={al:.4f} | avg_H={ah:.4f} | "
                      f"avg_MCT={am:.2f}s | lambda_ent={lam_cur:.5f} | beta={beta_cur:.4f}")
                sys.stdout.flush(); log_f.flush()

                if al > LOSS_GATE:
                    print(f"\nSTOP GATE: avg_loss={al:.2f} > {LOSS_GATE} at ep {ep}")
                    return agent
                if ep == 5000 and ah > ENT_GATE_5K:
                    print(f"\nSTOP GATE: avg_H={ah:.4f} > {ENT_GATE_5K} at ep 5000")
                    return agent

    print(f"\nTraining complete — {N_EPISODES} episodes.")
    print(f"Final log_beta = {agent.log_beta[0]:.6f}")
    print(f"Final beta     = {agent.beta():.6f}  (init was {1.0/math.sqrt(D_ATTN):.4f})")
    return agent


if __name__ == "__main__":
    random.seed(42); np.random.seed(42)
    print("=" * 64)
    print("Week 10B — AttentionDQN + learnable beta (no PBRS)")
    print("=" * 64)
    t0    = time.time()
    agent = train()
    print(f"\nWall time: {(time.time()-t0)/60:.1f} min")
    agent.save(WEIGHTS_PATH)
    print(f"Weights saved → {WEIGHTS_PATH}")
