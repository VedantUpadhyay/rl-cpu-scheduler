"""Transformer-based RL scheduler agent.

Architecture
------------
  per-task encoder : Linear(7, 64) → ReLU → Linear(64, 64)
  transformer      : 2 layers, 4 heads, d_model=64, ffn=256, dropout=0.1
                     key_padding_mask derived from arrived_flag (index 6)
  global context   : mean-pool over valid (non-padded) task tokens  → (64,)
  FiLM omega cond  : gamma, beta = Linear(1, 64); ctx = gamma*ctx + beta
  scoring head     : [task_enc(64) ‖ global_ctx(64)] → Linear(128, 3)
  output           : (batch, N*3) Q-values; invalid positions masked to −1e9

Variable N
----------
The network infers N from the state tensor shape: N = state.shape[-1] // 7.
For the replay buffer, states are zero-padded to N_MAX*7 = 350 dims.
Arrived_flag=0 at padded positions acts as a natural attention mask.

Training
--------
Double DQN.  Same replay buffer (OmegaReplayBuffer), hyperparameters
(BUF_CAPACITY, BATCH_SIZE, WARMUP, TARGET_UPDATE_FREQ, LAMBDA_START/END),
and epsilon schedule as the existing W15 training pipeline.

Variable-N training
-------------------
TransformerTrainer.sample_n() draws N uniformly from N_OPTIONS each episode.
States are zero-padded to N_MAX * D_CAND before buffer storage so the buffer
has a fixed state_dim regardless of episode N.

Usage
-----
  from transformer_agent import TransformerTrainer
  trainer = TransformerTrainer()                  # default N_MAX=50
  action  = trainer.select_action(sv, valid, omega_s=0.5)
  loss, _ = trainer.update(s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b)

Standalone smoke test:
  python transformer_agent.py
"""
from __future__ import annotations
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
D_CAND  = 7     # features per task slot (matches train.py / w15_network_torch.py)
AFI     = 6     # arrived_flag index within each 7-dim slot
N_QT    = 3     # quantum tiers: 0.5 s, 2.0 s, 8.0 s
N_MAX   = 50    # largest N supported; defines replay buffer state_dim
N_OPTIONS = [5, 10, 20, 50]   # for variable-N curriculum sampling

STATE_DIM_MAX = N_MAX * D_CAND   # 350

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FFN_DIM = 256
DROPOUT = 0.1

LR        = 1e-4
GAMMA     = 0.99
GRAD_CLIP = 1.0

# Hyperparams imported from the existing training pipeline
_SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "docs", "scheduler-research", "scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS)

from ablation_multiseed import (
    LAMBDA_START, LAMBDA_END,
    BUF_CAPACITY, BATCH_SIZE, TARGET_UPDATE_FREQ, WARMUP,
)
from w14_omega import OmegaReplayBuffer


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class TransformerSchedulerNet(nn.Module):
    """Transformer DQN with FiLM omega conditioning.

    Accepts any N at forward time — N is inferred from state tensor width.
    Arrived_flag (index 6 of each 7-dim slot) serves as the validity mask
    for both attention and output action masking.
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_layers: int = N_LAYERS, ffn_dim: int = FFN_DIM,
                 dropout: float = DROPOUT, n_qt: int = N_QT) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_qt    = n_qt

        # Per-task encoder: Linear(7,64) → ReLU → Linear(64,64)
        self.task_encoder = nn.Sequential(
            nn.Linear(D_CAND, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Transformer encoder (batch_first so input is (B, N, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = n_heads,
            dim_feedforward = ffn_dim,
            dropout        = dropout,
            batch_first    = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # FiLM conditioning on omega_s: gamma and beta both (1 → d_model)
        self.film_gamma = nn.Linear(1, d_model)
        self.film_beta  = nn.Linear(1, d_model)

        # Per-task scoring: [task_enc ‖ global_ctx] → n_qt Q-values
        self.scoring_head = nn.Linear(d_model * 2, n_qt)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor,
                omega_s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:   (batch, N*7) — N inferred at runtime from tensor width
            omega_s: (batch,)     — starvation preference in [0, 1]
        Returns:
            q_vals:  (batch, N*3) — invalid actions masked to −1e9
        """
        batch = state.shape[0]
        n     = state.shape[1] // D_CAND

        # (batch, N*7) → (batch, N, 7)
        encs = state.view(batch, n, D_CAND)

        # Arrived mask from arrived_flag (index AFI=6)
        arrived   = encs[:, :, AFI] > 0.5          # (batch, N) bool

        # key_padding_mask: True = position is IGNORED by attention
        # Use ~arrived so that not-arrived/completed tasks are masked out.
        # Guard against all-masked batches (would produce NaN in softmax).
        pad_mask = ~arrived                          # (batch, N)
        all_pad  = pad_mask.all(dim=1, keepdim=True) # (batch, 1) safety
        safe_mask = pad_mask & ~all_pad              # at least one slot open

        # --- Per-task encoding ---
        task_emb = self.task_encoder(encs)           # (batch, N, 64)

        # --- Transformer ---
        trans_out = self.transformer(
            task_emb,
            src_key_padding_mask=safe_mask,          # (batch, N)
        )                                             # (batch, N, 64)

        # --- Global context: mean-pool over valid (arrived) tokens ---
        valid_f     = arrived.unsqueeze(-1).float()  # (batch, N, 1)
        valid_count = valid_f.sum(dim=1).clamp(min=1.0)   # (batch, 1)
        global_ctx  = (trans_out * valid_f).sum(dim=1) / valid_count  # (batch, 64)

        # --- FiLM conditioning ---
        os_    = omega_s.unsqueeze(1)                # (batch, 1)
        gamma  = self.film_gamma(os_)                # (batch, 64)
        beta   = self.film_beta(os_)                 # (batch, 64)
        global_ctx = gamma * global_ctx + beta        # (batch, 64)

        # --- Per-task scoring ---
        ctx_exp  = global_ctx.unsqueeze(1).expand(batch, n, self.d_model)
        combined = torch.cat([trans_out, ctx_exp], dim=-1)  # (batch, N, 128)
        q_tasks  = self.scoring_head(combined)              # (batch, N, 3)

        # Flatten to (batch, N*3)
        q_flat = q_tasks.reshape(batch, n * self.n_qt)

        # Mask invalid actions (not arrived or complete)
        inv_mask = (~arrived).unsqueeze(-1) \
                             .expand(batch, n, self.n_qt) \
                             .reshape(batch, n * self.n_qt)
        q_flat = q_flat.masked_fill(inv_mask, -1e9)

        return q_flat

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# State padding helper
# ---------------------------------------------------------------------------

def _pad_state(state_np: np.ndarray, target_dim: int = STATE_DIM_MAX) -> np.ndarray:
    """Zero-pad a state vector to target_dim.

    Padded positions have arrived_flag=0, so the transformer masks them out
    automatically. This is safe because the env state encoder also writes
    zeros for non-arrived / complete processes.
    """
    if len(state_np) == target_dim:
        return state_np
    out = np.zeros(target_dim, dtype=np.float32)
    out[:len(state_np)] = state_np
    return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class TransformerTrainer:
    """Wraps online + target TransformerSchedulerNet with Double DQN training.

    Public API mirrors W15Trainer:
      select_action(state_np, valid_actions, omega_s) -> int
      update(s, a, rvd, rss, ns, d, om)              -> (loss, entropy)
      update_target()
      decay_epsilon(ep, n_eps)
      save(path) / load(path)

    Variable-N:
      sample_n() returns N drawn uniformly from N_OPTIONS.
      States passed to select_action / update may be any multiple of 7.
      Internally they are zero-padded to STATE_DIM_MAX for buffer storage.
      Forward pass uses actual N from state tensor width so both small and
      large episodes run at their true cost.
    """

    def __init__(self, n_max: int = N_MAX,
                 lr: float = LR, gamma: float = GAMMA,
                 grad_clip: float = GRAD_CLIP) -> None:
        self.n_max     = n_max
        self.n_actions_max = n_max * N_QT          # 150 for N_MAX=50
        self.gamma     = gamma
        self.grad_clip = grad_clip
        self.epsilon   = 1.0
        self.lambda_ent = LAMBDA_START

        self.online = TransformerSchedulerNet().to(device)
        self.target = TransformerSchedulerNet().to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer = optim.Adam(self.online.parameters(), lr=lr)

        # Replay buffer with fixed padded state_dim
        self._buf = OmegaReplayBuffer(
            capacity  = BUF_CAPACITY,
            state_dim = STATE_DIM_MAX,
        )

    # ------------------------------------------------------------------
    # Variable-N episode sampling
    # ------------------------------------------------------------------

    @staticmethod
    def sample_n(options: list[int] = N_OPTIONS) -> int:
        """Draw N uniformly from N_OPTIONS for curriculum training."""
        return random.choice(options)

    # ------------------------------------------------------------------
    # Buffer interface (handles padding transparently)
    # ------------------------------------------------------------------

    def store(self, sv: np.ndarray, action: int, r_vd: float,
              r_ss: float, sv_next: np.ndarray, done: bool,
              omega_s: float) -> None:
        """Store transition, zero-padding states to STATE_DIM_MAX."""
        self._buf.store(
            _pad_state(sv),
            action,
            r_vd, r_ss,
            _pad_state(sv_next),
            done,
            omega_s,
        )

    def __len__(self) -> int:
        return len(self._buf)

    def sample(self, batch_size: int):
        return self._buf.sample(batch_size)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state_np: np.ndarray,
                      valid_actions: list[int], omega_s: float) -> int:
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            # Use actual N from state (not padded) for the forward pass
            s = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
            o = torch.tensor([omega_s], dtype=torch.float32, device=device)
            q = self.online(s, o).squeeze(0).cpu().numpy()
        mask = np.full(len(q), -np.inf)
        for a in valid_actions:
            mask[a] = q[a]
        return int(np.argmax(mask))

    # ------------------------------------------------------------------
    # Training step (Double DQN)
    # ------------------------------------------------------------------

    def update(self, s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b):
        """Double DQN update. All inputs are NumPy arrays (padded to N_MAX*7)."""
        s    = torch.from_numpy(s_b).float().to(device)    # (B, 350)
        ns   = torch.from_numpy(ns_b).float().to(device)
        a    = torch.from_numpy(a_b).long().to(device)
        d    = torch.from_numpy(d_b).float().to(device)
        om   = torch.from_numpy(om_b).float().to(device)
        om_mct = 1.0 - om

        r_vd = torch.from_numpy(rvd_b).float().to(device)
        r_ss = torch.from_numpy(rss_b).float().to(device)
        r    = om_mct * r_vd + om * r_ss

        # Q(s, a) from online network
        q_online = self.online(s, om)                          # (B, N_MAX*3)
        q_sa     = q_online.gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN: online selects, target evaluates
        with torch.no_grad():
            best_a   = self.online(ns, om).argmax(dim=1)
            q_next   = self.target(ns, om).gather(
                           1, best_a.unsqueeze(1)).squeeze(1)
            td_target = (r + self.gamma * q_next * (1.0 - d)).clamp(-50.0, 50.0)

        # Entropy bonus over valid Q-values
        probs   = torch.softmax(q_online, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        loss = nn.functional.smooth_l1_loss(q_sa, td_target) \
               - self.lambda_ent * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item()), float(entropy.item())

    # ------------------------------------------------------------------
    # Target network, epsilon, save / load
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self, ep: int, n_eps: int = 40_000,
                      eps_min: float = 0.05) -> float:
        self.epsilon = max(eps_min, 1.0 - (1.0 - eps_min) * ep / n_eps)
        return self.epsilon

    def save(self, path: str) -> None:
        torch.save({
            "online":     self.online.state_dict(),
            "target":     self.target.state_dict(),
            "epsilon":    self.epsilon,
            "lambda_ent": self.lambda_ent,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon    = ckpt.get("epsilon",    0.05)
        self.lambda_ent = ckpt.get("lambda_ent", LAMBDA_END)
        self.target.eval()

    def n_params(self) -> int:
        return self.online.n_params()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    print(f"Device: {device}")
    print(f"Hyperparams: BUF_CAPACITY={BUF_CAPACITY}  BATCH_SIZE={BATCH_SIZE}"
          f"  WARMUP={WARMUP}  TARGET_UPDATE_FREQ={TARGET_UPDATE_FREQ}")
    print()

    net = TransformerSchedulerNet().to(device)
    n_p = net.n_params()
    print(f"TransformerSchedulerNet parameters: {n_p:,}")
    print()

    # Forward pass for N = 5, 10, 20 — confirms variable-N support
    for N in [5, 10, 20]:
        batch = 4
        state_dim = N * D_CAND
        s  = torch.randn(batch, state_dim, device=device)
        # Set arrived_flag=1 for all tasks so attention isn't fully masked
        s[:, AFI::D_CAND] = 1.0
        om = torch.rand(batch, device=device)

        with torch.no_grad():
            q = net(s, om)

        expected = (batch, N * N_QT)
        assert q.shape == expected, \
            f"N={N}: expected {expected}, got {tuple(q.shape)}"
        assert torch.all(torch.isfinite(q)), f"N={N}: NaN/Inf in output"
        print(f"  N={N:>2}  input=({batch},{state_dim:>3})  "
              f"output={tuple(q.shape)}  "
              f"Q range=[{q.min():.2f}, {q.max():.2f}]  ✓")

    print()

    # Trainer smoke test — one update step with padded states
    print("Trainer smoke test (one Double DQN update) ...")
    trainer = TransformerTrainer()

    N_ep   = 5
    sd_ep  = N_ep * D_CAND
    B      = BATCH_SIZE

    # Build a fake batch with padded states (as the buffer would return)
    s_b   = np.zeros((B, STATE_DIM_MAX), dtype=np.float32)
    ns_b  = np.zeros((B, STATE_DIM_MAX), dtype=np.float32)
    # Mark first N_ep*D_CAND dims as arrived
    s_b[:,  AFI:N_ep * D_CAND:D_CAND] = 1.0
    ns_b[:, AFI:N_ep * D_CAND:D_CAND] = 1.0

    a_b   = np.random.randint(0, N_ep * N_QT, B).astype(np.int64)
    rvd_b = np.random.randn(B).astype(np.float32)
    rss_b = np.random.randn(B).astype(np.float32)
    d_b   = np.zeros(B, dtype=np.float32)
    om_b  = np.random.rand(B).astype(np.float32)

    loss, ent = trainer.update(s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b)
    print(f"  loss={loss:.4f}  entropy={ent:.4f}  ✓")

    # select_action with actual-N state (not padded)
    sv    = np.zeros(sd_ep, dtype=np.float32)
    sv[AFI::D_CAND] = 1.0    # all tasks arrived
    valid = list(range(N_ep * N_QT))
    act   = trainer.select_action(sv, valid, omega_s=0.5)
    assert 0 <= act < N_ep * N_QT, f"action {act} out of range"
    print(f"  select_action (N=5, greedy): action={act}  ✓")

    # sample_n
    ns_drawn = [TransformerTrainer.sample_n() for _ in range(1000)]
    assert all(n in N_OPTIONS for n in ns_drawn)
    counts = {n: ns_drawn.count(n) for n in N_OPTIONS}
    print(f"  sample_n distribution (1000 draws): {counts}  ✓")

    print()
    print("All smoke tests passed.")
    print(f"\nSummary:")
    print(f"  Architecture : Transformer DQN, FiLM omega conditioning")
    print(f"  Parameters   : {n_p:,}")
    print(f"  d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, ffn={FFN_DIM}")
    print(f"  State dim    : variable (N×{D_CAND}), padded to {STATE_DIM_MAX} in buffer")
    print(f"  Action dim   : variable (N×{N_QT}), up to {N_MAX * N_QT} in buffer")
    print(f"  N options    : {N_OPTIONS}")
