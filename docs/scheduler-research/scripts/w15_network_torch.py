"""W15 PyTorch network — W15OmegaDQN and W15Trainer.

Replaces NumPy W14OmegaDQN with PyTorch for GPU acceleration.
Environment simulation, reward, and replay buffer remain in NumPy.
Only network forward pass, loss, backprop, and optimizer use PyTorch.

Architecture matches W14OmegaDQN exactly:
  - 35-dim state (5 slots × 7 features)
  - 2-head attention DQN
  - FiLM omega conditioning:
      pre-attention:  Q_cond = Q * (1 + omega_s)
      post-attention: context_cond = context * (1 + omega_s) + omega_mct
  - MLP: [cand_enc(7) | context_cond(16) | omega_s(1)] → 64 → 32 → 1
  - Output: (batch, N_PROCESSES * N_QT) Q-values, one per action

Note: N_QT=3 (0.5s, 2.0s, 8.0s quanta). The user spec wrote N_QT=7 —
that was a typo. N_ACTIONS = 5 × 3 = 15, matching W14OmegaDQN.
"""
from __future__ import annotations
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# MPS (Apple Silicon GPU) is ~19x SLOWER than CPU for batch_size=32.
# This model is tiny (4,369 params). CPU is the right device.
# Switch to CUDA if available (large-batch Colab use case).
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Dimensions — must match w15_colab.py / w9_train.py exactly
# ---------------------------------------------------------------------------
N_PROCESSES = 5
D_CAND      = 7          # features per slot
D_HEAD      = 8
N_HEADS     = 2
D_V_TOT     = N_HEADS * D_HEAD    # 16
N_QT        = 3          # quantum tiers: 0.5s, 2.0s, 8.0s
N_ACTIONS   = N_PROCESSES * N_QT  # 15
D_MLP_IN    = D_CAND + D_V_TOT + 1  # 7 + 16 + 1 = 24  (omega_s as extra feature)
AFI         = 6          # arrived_flag index in 7-dim feature vector

LR               = 1e-4
GAMMA            = 0.99
GRAD_CLIP        = 1.0
LAMBDA_ENT_START = 0.01
LAMBDA_ENT_END   = 0.001


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class W15OmegaDQN(nn.Module):
    """2-head attention DQN with FiLM omega conditioning (PyTorch).

    Identical semantics to W14OmegaDQN (NumPy) but uses autograd.
    Output shape: (batch, N_PROCESSES * N_QT) = (batch, 15).
    Invalid actions (arrived_flag == 0) are masked to -1e9.
    """

    def __init__(self, n_processes: int = N_PROCESSES) -> None:
        super().__init__()
        self.n_processes = n_processes
        self.n_actions   = n_processes * N_QT

        self.W_Q = nn.ModuleList([nn.Linear(D_CAND, D_HEAD) for _ in range(N_HEADS)])
        self.W_K = nn.ModuleList([nn.Linear(D_CAND, D_HEAD) for _ in range(N_HEADS)])
        self.W_V = nn.ModuleList([nn.Linear(D_CAND, D_HEAD) for _ in range(N_HEADS)])
        self.W_O = nn.Linear(D_V_TOT, D_V_TOT)

        self.mlp = nn.Sequential(
            nn.Linear(D_MLP_IN, 64), nn.ReLU(),
            nn.Linear(64, 32),       nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.scale = D_HEAD ** -0.5
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor, omega_s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:   (batch, 35)  float32
            omega_s: (batch,)     float32, starvation preference in [0, 1]
        Returns:
            q_vals:  (batch, 15)  float32
        """
        batch     = state.shape[0]
        omega_mct = 1.0 - omega_s                                        # (batch,)
        encs      = state.view(batch, self.n_processes, D_CAND)          # (batch, N, 7)
        valid     = encs[:, :, AFI] > 0.5                                # (batch, N) bool

        # Multi-head attention with FiLM pre-attention Q modulation
        head_outputs = []
        for h in range(N_HEADS):
            Q = self.W_Q[h](encs)   # (batch, 5, D_HEAD)
            K = self.W_K[h](encs)
            V = self.W_V[h](encs)

            # FiLM pre-attention: Q_cond = Q * (1 + omega_s)
            Q_cond = Q * (1.0 + omega_s.view(batch, 1, 1))

            # Scaled dot-product attention: (batch, 5, 5)
            scores = torch.bmm(Q_cond, K.transpose(1, 2)) * self.scale
            # Mask invalid key positions
            scores = scores.masked_fill(~valid.unsqueeze(1), -1e9)
            attn   = torch.softmax(scores, dim=-1)       # (batch, 5, 5)
            head_outputs.append(torch.bmm(attn, V))      # (batch, 5, D_HEAD)

        context_cat  = torch.cat(head_outputs, dim=-1)   # (batch, 5, 16)
        context_out  = self.W_O(context_cat)              # (batch, 5, 16)

        # FiLM post-attention modulation
        os_  = omega_s.view(batch, 1, 1)
        om_  = omega_mct.view(batch, 1, 1)
        context_cond = context_out * (1.0 + os_) + om_   # (batch, 5, 16)

        # MLP input: [cand_enc(7) | context_cond(16) | omega_s(1)] = 24
        omega_feat = omega_s.view(batch, 1, 1).expand(batch, self.n_processes, 1)
        mlp_in     = torch.cat([encs, context_cond, omega_feat], dim=-1)  # (batch, N, 24)

        # Per-candidate score → (batch, N)
        cand_scores = self.mlp(mlp_in).squeeze(-1)   # (batch, N)

        # Expand to (batch, N*N_QT): same Q-value for all quanta of each task
        q_vals = (cand_scores.unsqueeze(-1)
                             .expand(batch, self.n_processes, N_QT)
                             .reshape(batch, self.n_actions))

        # Mask invalid actions
        invalid_mask = (~valid.unsqueeze(-1)
                              .expand(batch, self.n_processes, N_QT)
                              .reshape(batch, self.n_actions))
        q_vals = q_vals.masked_fill(invalid_mask, -1e9)
        return q_vals


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class W15Trainer:
    """Wraps online + target W15OmegaDQN with PyTorch Adam optimizer.

    Public API matches W14OmegaDQN where possible:
      select_action(state_np, valid_actions, omega_s) -> int
      update(s, a, rvd, rss, ns, d, om) -> (loss, entropy)
      compute_grad_norms(s, a, rvd, rss, ns, d, om) -> (norm_vd, norm_ss, ratio)
      update_target()
      decay_epsilon(ep, n_eps)
      save(path) / load(path)   — .pt extension (not .npz)
    """

    def __init__(self, n_processes: int = N_PROCESSES,
                 lr: float = LR, gamma: float = GAMMA,
                 grad_clip: float = GRAD_CLIP) -> None:
        self.n_processes = n_processes
        self.n_actions   = n_processes * N_QT
        self.online = W15OmegaDQN(n_processes).to(device)
        self.target = W15OmegaDQN(n_processes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.optimizer  = optim.Adam(self.online.parameters(), lr=lr)
        self.gamma      = gamma
        self.grad_clip  = grad_clip
        self.epsilon    = 1.0
        self.lambda_ent = LAMBDA_ENT_START

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state_np: np.ndarray,
                      valid_actions: list[int], omega_s: float) -> int:
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        with torch.no_grad():
            s = torch.from_numpy(state_np).float().unsqueeze(0).to(device)
            o = torch.tensor([omega_s], dtype=torch.float32, device=device)
            q = self.online(s, o).squeeze(0).cpu().numpy()
        mask = np.full(self.n_actions, -np.inf)
        for a in valid_actions:
            mask[a] = q[a]
        return int(np.argmax(mask))

    # ------------------------------------------------------------------
    # Training step (Double DQN)
    # ------------------------------------------------------------------

    def update(self, s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b):
        """Single Double-DQN update step. All inputs are NumPy arrays."""
        s   = torch.from_numpy(s_b).float().to(device)
        ns  = torch.from_numpy(ns_b).float().to(device)
        a   = torch.from_numpy(a_b).long().to(device)
        d   = torch.from_numpy(d_b).float().to(device)
        om  = torch.from_numpy(om_b).float().to(device)
        om_mct = 1.0 - om

        r_vd = torch.from_numpy(rvd_b).float().to(device)
        r_ss = torch.from_numpy(rss_b).float().to(device)
        r    = om_mct * r_vd + om * r_ss

        # Q(s, a) from online network
        q_online = self.online(s, om)
        q_sa     = q_online.gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN: online selects action, target evaluates
        with torch.no_grad():
            best_a   = self.online(ns, om).argmax(dim=1)
            q_next   = self.target(ns, om).gather(1, best_a.unsqueeze(1)).squeeze(1)
            td_target = (r + self.gamma * q_next * (1.0 - d)).clamp(-50.0, 50.0)

        # Entropy bonus (encourages exploration of valid actions)
        probs   = torch.softmax(q_online, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        loss = nn.functional.smooth_l1_loss(q_sa, td_target) - self.lambda_ent * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        return float(loss.item()), float(entropy.item())

    # ------------------------------------------------------------------
    # Gradient norm monitoring (W11-ghost risk detection)
    # ------------------------------------------------------------------

    def compute_grad_norms(self, s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b):
        """Returns (norm_vd, norm_ss, ratio). Does NOT step optimizer."""
        s   = torch.from_numpy(s_b).float().to(device)
        ns  = torch.from_numpy(ns_b).float().to(device)
        a   = torch.from_numpy(a_b).long().to(device)
        d   = torch.from_numpy(d_b).float().to(device)
        om  = torch.from_numpy(om_b).float().to(device)
        om_mct = 1.0 - om

        r_vd = torch.from_numpy(rvd_b).float().to(device)
        r_ss = torch.from_numpy(rss_b).float().to(device)

        q_sa = self.online(s, om).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_a = self.online(ns, om).argmax(dim=1)
            q_next = self.target(ns, om).gather(1, best_a.unsqueeze(1)).squeeze(1)

        def _norm(loss_t):
            self.optimizer.zero_grad()
            loss_t.backward(retain_graph=True)
            return sum(
                p.grad.norm().item() ** 2
                for p in self.online.parameters() if p.grad is not None
            ) ** 0.5

        t_vd   = (om_mct * r_vd + self.gamma * q_next * (1.0 - d)).clamp(-50.0, 50.0).detach()
        norm_vd = _norm(nn.functional.mse_loss(q_sa, t_vd))

        t_ss   = (om * r_ss + self.gamma * q_next * (1.0 - d)).clamp(-50.0, 50.0).detach()
        norm_ss = _norm(nn.functional.mse_loss(q_sa, t_ss))

        self.optimizer.zero_grad()
        ratio = norm_ss / max(norm_vd, 1e-8)
        return norm_vd, norm_ss, ratio

    # ------------------------------------------------------------------
    # Target network, epsilon, save / load
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def decay_epsilon(self, ep: int, n_eps: int = 40_000,
                      eps_min: float = 0.05) -> float:
        """Linear decay: reaches eps_min at ep == n_eps."""
        self.epsilon = max(eps_min, 1.0 - (1.0 - eps_min) * ep / n_eps)
        return self.epsilon

    def save(self, path: str) -> None:
        pt = path.replace(".npz", ".pt") if path.endswith(".npz") else path
        torch.save({
            "online":     self.online.state_dict(),
            "target":     self.target.state_dict(),
            "epsilon":    self.epsilon,
            "lambda_ent": self.lambda_ent,
        }, pt)

    def load(self, path: str) -> None:
        pt   = path.replace(".npz", ".pt") if path.endswith(".npz") else path
        ckpt = torch.load(pt, map_location=device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.epsilon    = ckpt.get("epsilon",    0.05)
        self.lambda_ent = ckpt.get("lambda_ent", LAMBDA_ENT_END)
        self.target.eval()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.online.parameters())


# ---------------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device: {device}")
    t = W15Trainer()
    n = t.n_params()
    print(f"W15OmegaDQN parameters: {n:,}  (W14 reference: 4,369)")

    batch = 32
    s  = torch.randn(batch, N_PROCESSES * D_CAND, device=device)
    om = torch.rand(batch, device=device)
    q  = t.online(s, om)
    print(f"Forward pass: output shape {tuple(q.shape)}")

    s_b   = np.random.randn(batch, N_PROCESSES * D_CAND).astype(np.float32)
    a_b   = np.random.randint(0, N_ACTIONS, batch).astype(np.int32)
    rvd_b = np.random.randn(batch).astype(np.float32)
    rss_b = np.random.randn(batch).astype(np.float32)
    ns_b  = np.random.randn(batch, N_PROCESSES * D_CAND).astype(np.float32)
    d_b   = np.zeros(batch, dtype=np.float32)
    om_b  = np.random.rand(batch).astype(np.float32)

    loss, ent = t.update(s_b, a_b, rvd_b, rss_b, ns_b, d_b, om_b)
    print(f"Update: loss={loss:.4f}  entropy={ent:.4f}")
    print("Smoke test passed.")
