"""Q-learning agent (Week 1/2) and DQN components (Weeks 3–7).

Week 3 additions
----------------
AdamOptimizer  — numpy-only Adam; parameters updated in-place.
DQNAgent       — 15→64→32→15 MLP with He init, experience replay,
                 target network; replaces QLearningAgent for Week 3.
ReplayBuffer   — circular buffer storing (s, a, r, s', done).

Week 5 addition
---------------
ActionConditionedDQN — 19→64→32→1; input = [s_masked(15) ‖ a4(4)].
                 Achieves partial permutation invariance (candidate
                 PID-invariant when competitor set is at the same
                 PID positions).

Week 6 addition
---------------
SumPoolingDQN  — 7→64→32→1; input = [competitor_pool(3) ‖ a4(4)].
                 competitor_pool = element-wise sum of arrived,
                 non-complete competitor feature encodings.
                 Sum-pooling is permutation-invariant by commutativity,
                 achieving full permutation invariance over both candidate
                 and competitor PIDs.

Week 7 addition
---------------
AttentionDQN   — dot-product attention + MLP.
                 Attention: W_Q/W_K/W_V (3×8 each) produce query from
                 candidate and keys/values from each competitor.
                 context = Σ_i w_i·v_i where w = softmax(q·k/√8, masked).
                 MLP: [context(8) ‖ a4(4)] = 12 → 64(ReLU) → 32(ReLU) → 1.
                 Full permutation invariance via softmax commutativity.
                 Total: 3,041 parameters.

QLearningAgent is retained for backward compatibility with Week 1/2
runner.py code and saved Q-tables.
"""
from __future__ import annotations

import os
import random

import numpy as np


# ---------------------------------------------------------------------------
# Adam optimiser (numpy-only)
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """Vanilla Adam (Kingma & Ba 2015).

    Usage
    -----
    opt = AdamOptimizer(lr=0.001)
    # inside training loop:
    opt.step(params=[W1, b1, W2, b2, ...], grads=[dW1, db1, dW2, db2, ...])
    # params are updated in-place; grads are read-only.
    """

    def __init__(
        self,
        lr:      float = 0.001,
        beta1:   float = 0.9,
        beta2:   float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.epsilon = epsilon

        self._m: list[np.ndarray] = []   # first-moment estimates
        self._v: list[np.ndarray] = []   # second-moment estimates
        self._t: int = 0                  # global step counter

    def reset(self) -> None:
        """Clear moment history and step counter (e.g. between runs)."""
        self._m.clear()
        self._v.clear()
        self._t = 0

    def step(
        self,
        params: list[np.ndarray],
        grads:  list[np.ndarray],
    ) -> None:
        """Update each param array in-place using its corresponding gradient.

        Parameters
        ----------
        params : list of numpy arrays (modified in-place)
        grads  : list of numpy arrays, same length and shapes as params
        """
        # Initialise moment arrays on first call
        if not self._m:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]

        self._t += 1
        t = self._t

        b1t = self.beta1 ** t
        b2t = self.beta2 ** t

        for i, (p, g) in enumerate(zip(params, grads)):
            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * (g * g)

            m_hat = self._m[i] / (1.0 - b1t)
            v_hat = self._v[i] / (1.0 - b2t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Circular experience-replay buffer.

    Stores transitions (state, action, reward, next_state, done).
    States are stored as flat float32 vectors (continuous DQN inputs).
    """

    def __init__(self, capacity: int = 10_000, state_dim: int = 15) -> None:
        self.capacity  = capacity
        self.state_dim = state_dim
        self._ptr      = 0    # write pointer
        self._size     = 0    # number of valid entries

        self._states      = np.zeros((capacity, state_dim), dtype=np.float32)
        self._next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self._actions     = np.zeros(capacity, dtype=np.int32)
        self._rewards     = np.zeros(capacity, dtype=np.float32)
        self._dones       = np.zeros(capacity, dtype=np.float32)

    def store(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        """Add one transition. Overwrites oldest entry when full."""
        i = self._ptr
        self._states[i]      = state
        self._next_states[i] = next_state
        self._actions[i]     = action
        self._rewards[i]     = reward
        self._dones[i]       = float(done)
        self._ptr  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Sample a random mini-batch without replacement.

        Returns
        -------
        states, actions, rewards, next_states, dones
        """
        idx = np.random.choice(self._size, size=batch_size, replace=False)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx],
            self._dones[idx],
        )

    def __len__(self) -> int:
        return self._size


# ---------------------------------------------------------------------------
# DQN agent (Week 3)
# ---------------------------------------------------------------------------

class DQNAgent:
    """Numpy-only DQN: 15 → 64(ReLU) → 32(ReLU) → 15(linear).

    Parameters
    ----------
    n_actions  : number of actions (15)
    state_dim  : length of continuous state vector (15)
    lr         : Adam learning rate
    gamma      : discount factor (1.0 = undiscounted episodic)
    epsilon    : initial exploration probability
    grad_clip  : symmetric gradient clipping threshold (±grad_clip)
    """

    LAYER_SIZES = (15, 64, 32, 15)

    def __init__(
        self,
        n_actions:  int   = 15,
        state_dim:  int   = 15,
        lr:         float = 0.001,
        gamma:      float = 1.0,
        epsilon:    float = 1.0,
        grad_clip:  float = 1.0,
    ) -> None:
        self.n_actions      = n_actions
        self.state_dim      = state_dim
        self.gamma          = gamma
        self.epsilon        = epsilon
        self._init_epsilon  = epsilon
        self.grad_clip      = grad_clip

        # --- Online network weights (He init: std = sqrt(2/fan_in)) --------
        self._W, self._b = self._init_weights()

        # --- Target network (hard copy; updated every N episodes) -----------
        self._tW, self._tb = self._copy_weights(self._W, self._b)

        # --- Optimiser (online network only) --------------------------------
        self._opt = AdamOptimizer(lr=lr)

    # ------------------------------------------------------------------
    # Weight initialisation helpers
    # ------------------------------------------------------------------

    def _init_weights(
        self,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """He initialisation for all layers."""
        sizes = self.LAYER_SIZES
        W, b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            std = np.sqrt(2.0 / fan_in)
            W.append(np.random.randn(fan_in, fan_out).astype(np.float64) * std)
            b.append(np.zeros(fan_out, dtype=np.float64))
        return W, b

    @staticmethod
    def _copy_weights(
        W: list[np.ndarray], b: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return [w.copy() for w in W], [bi.copy() for bi in b]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: np.ndarray,
        use_target: bool = False,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Compute Q-values for a single state vector.

        Parameters
        ----------
        x          : 1-D array of length state_dim
        use_target : if True, run through target network

        Returns
        -------
        q_values : 1-D array of length n_actions
        cache    : list of pre-activation arrays (needed for backprop)
        """
        W = self._tW if use_target else self._W
        b = self._tb if use_target else self._b

        activations: list[np.ndarray] = [x]
        h = x.astype(np.float64)

        for i, (w, bi) in enumerate(zip(W, b)):
            z = h @ w + bi
            if i < len(W) - 1:      # hidden layers: ReLU
                h = np.maximum(0.0, z)
            else:                    # output layer: linear
                h = z
            activations.append(h)

        return activations[-1], activations

    def forward_batch(
        self,
        X: np.ndarray,
        use_target: bool = False,
    ) -> np.ndarray:
        """Compute Q-values for a batch of states.

        Parameters
        ----------
        X : 2-D array of shape (batch, state_dim)

        Returns
        -------
        Q : 2-D array of shape (batch, n_actions)
        """
        W = self._tW if use_target else self._W
        b = self._tb if use_target else self._b

        h = X.astype(np.float64)
        for i, (w, bi) in enumerate(zip(W, b)):
            z = h @ w + bi
            h = np.maximum(0.0, z) if i < len(W) - 1 else z
        return h

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state:         np.ndarray,
        epsilon:       float,
        valid_actions: list[int],
    ) -> int:
        """Masked epsilon-greedy action selection.

        Parameters
        ----------
        state         : continuous state vector
        epsilon       : exploration probability
        valid_actions : list of action indices that are currently legal
        """
        if not valid_actions:
            raise RuntimeError("No valid actions — env bug")
        if random.random() < epsilon:
            return random.choice(valid_actions)
        q_values, _ = self.forward(state)
        return max(valid_actions, key=lambda a: q_values[a])

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update_online(
        self,
        states:      np.ndarray,   # (batch, state_dim)
        actions:     np.ndarray,   # (batch,)  int32
        rewards:     np.ndarray,   # (batch,)  float
        next_states: np.ndarray,   # (batch, state_dim)
        dones:       np.ndarray,   # (batch,)  float  0/1
    ) -> float:
        """One gradient step on the online network using a sampled batch.

        Target  y_i = r_i + γ · max_a Q_target(s'_i, a)   [non-terminal]
                y_i = r_i                                   [terminal]

        Loss    L = mean( (y_i − Q_online(s_i)[a_i])² )

        Returns mean loss (scalar).
        """
        batch = len(states)

        # --- Targets from target network (no gradient) --------------------
        q_next = self.forward_batch(next_states, use_target=True)   # (B, A)
        max_q_next = q_next.max(axis=1)                              # (B,)
        targets = rewards + self.gamma * max_q_next * (1.0 - dones) # (B,)

        # --- Forward pass through online network --------------------------
        q_online = self.forward_batch(states, use_target=False)      # (B, A)

        # Predicted Q for the selected action
        q_pred = q_online[np.arange(batch), actions]                 # (B,)

        # --- Loss and output-layer delta ----------------------------------
        delta = q_pred - targets                                      # (B,)
        loss  = float(np.mean(delta ** 2))

        # --- Backprop (manual; 3-layer network) ---------------------------
        # Output layer gradient (dL/dz_out = 2*delta/B per sample, linear act)
        dout = 2.0 * delta / batch                                   # (B,)

        # Sparse: only the chosen action's Q-value contributes
        dQ = np.zeros_like(q_online)                                 # (B, A)
        dQ[np.arange(batch), actions] = dout

        # Layer 3 (hidden2 → output): linear
        W2, b2 = self._W[2], self._b[2]
        # cache hidden2 activations
        h1 = np.maximum(0.0, states.astype(np.float64) @ self._W[0] + self._b[0])
        h2 = np.maximum(0.0, h1 @ self._W[1] + self._b[1])

        dW2 = h2.T @ dQ                                              # (32, 15)
        db2 = dQ.sum(axis=0)                                         # (15,)

        # Layer 2 (hidden1 → hidden2): ReLU
        dh2 = dQ @ W2.T                                              # (B, 32)
        dz2 = dh2 * (h2 > 0).astype(np.float64)                     # ReLU grad

        dW1 = h1.T @ dz2                                             # (64, 32)
        db1 = dz2.sum(axis=0)                                        # (32,)

        # Layer 1 (input → hidden1): ReLU
        dh1 = dz2 @ self._W[1].T                                     # (B, 64)
        dz1 = dh1 * (h1 > 0).astype(np.float64)                     # ReLU grad

        dW0 = states.astype(np.float64).T @ dz1                     # (15, 64)
        db0 = dz1.sum(axis=0)                                        # (64,)

        # --- Gradient clipping (global norm) ------------------------------
        # global_norm = sqrt( sum ||g_i||^2 )
        # if global_norm > grad_clip: scale all gradients by grad_clip/global_norm
        grads = [dW0, db0, dW1, db1, dW2, db2]
        global_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if global_norm > self.grad_clip:
            scale = self.grad_clip / global_norm
            grads = [g * scale for g in grads]

        # --- Adam step (online network weights only) ----------------------
        params = [self._W[0], self._b[0], self._W[1], self._b[1],
                  self._W[2], self._b[2]]
        self._opt.step(params, grads)

        return loss

    # ------------------------------------------------------------------
    # Target network
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        """Hard copy online → target network."""
        self._tW, self._tb = self._copy_weights(self._W, self._b)

    # ------------------------------------------------------------------
    # Epsilon decay (same formula as QLearningAgent)
    # ------------------------------------------------------------------

    def decay_epsilon(
        self,
        episode:  int,
        min_eps:  float = 0.05,
        decay:    float = 0.9995,
    ) -> float:
        self.epsilon = max(min_eps, self._init_epsilon * (decay ** episode))
        return self.epsilon

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save online network weights to .npz file."""
        np.savez(
            path,
            W0=self._W[0], b0=self._b[0],
            W1=self._W[1], b1=self._b[1],
            W2=self._W[2], b2=self._b[2],
        )

    def load(self, path: str) -> None:
        """Load online network weights from .npz file."""
        data = np.load(path)
        self._W = [data["W0"], data["W1"], data["W2"]]
        self._b = [data["b0"], data["b1"], data["b2"]]
        self.update_target()   # sync target


# ---------------------------------------------------------------------------
# Action-conditioned DQN (Week 5) — partial permutation invariance
# ---------------------------------------------------------------------------

class ActionConditionedDQN:
    """Week 5: action-conditioned DQN for partial permutation invariance.

    Architecture  : 19 → 64(ReLU) → 32(ReLU) → 1(linear)  [3,393 params]
    Input         : [s_masked(15) ‖ a4(4)]
    Output        : Q_scalar

    Masking rule (spec amendment):
        s_masked = s15.copy()
        s_masked[pid*3 : pid*3+3] = 0.0   ← zero candidate's own slot
        input_19 = concat(s_masked, a4)

    Rationale: a4 carries the candidate's features; zeroing its slot in
    s_masked separates competitor context (s_masked) from candidate
    description (a4), preventing the network from latching onto the
    PID-indexed copy of the candidate's features.

    Invariance property: Q is invariant to the candidate's PID when
    the competitor context (s_masked after zeroing) is identical across
    states — which holds whenever all non-candidate processes are at
    the same PID positions in both states.  Full permutation invariance
    (arbitrary reordering of all 5 processes) requires attention over
    the state context; that is left for a future week.

    To select an action: enumerate valid actions, build (n_valid × 19)
    batch in one call, run one forward pass, return argmax.
    """

    LAYER_SIZES = (19, 64, 32, 1)
    N_PROCESSES = 5
    N_QT        = 3
    N_ACTIONS   = 15

    def __init__(
        self,
        n_actions:  int   = 15,
        state_dim:  int   = 15,
        lr:         float = 0.001,
        gamma:      float = 1.0,
        epsilon:    float = 1.0,
        grad_clip:  float = 1.0,
    ) -> None:
        self.n_actions     = n_actions
        self.state_dim     = state_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self._init_epsilon = epsilon
        self.grad_clip     = grad_clip

        self._W, self._b   = self._init_weights()
        self._tW, self._tb = self._copy_weights(self._W, self._b)
        self._opt          = AdamOptimizer(lr=lr)

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        sizes = self.LAYER_SIZES
        W, b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            std = np.sqrt(2.0 / fan_in)
            W.append(np.random.randn(fan_in, fan_out).astype(np.float64) * std)
            b.append(np.zeros(fan_out, dtype=np.float64))
        return W, b

    @staticmethod
    def _copy_weights(
        W: list[np.ndarray], b: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return [w.copy() for w in W], [bi.copy() for bi in b]

    # ------------------------------------------------------------------
    # Input construction: [s_masked ‖ a4]
    # ------------------------------------------------------------------

    def _build_input_batch(
        self,
        states:  np.ndarray,   # (batch, 15)
        actions: np.ndarray,   # (batch,)  int
    ) -> np.ndarray:
        """Build (batch, 19) input matrix with candidate-slot masking.

        s_masked[pid*3 : pid*3+3] = 0.0  (candidate's slot zeroed)
        a4 = [remaining/BURST_P95, arrived, wait/WAIT_NORM, qt/2.0]  from original s
        """
        batch = len(states)
        s     = np.asarray(states,  dtype=np.float64)
        a     = np.asarray(actions, dtype=np.int32)
        pids  = a // self.N_QT
        qts   = a %  self.N_QT
        idx   = np.arange(batch)

        # Masked state: zero the candidate process's three features
        s_masked = s.copy()
        s_masked[idx, pids * 3]     = 0.0
        s_masked[idx, pids * 3 + 1] = 0.0
        s_masked[idx, pids * 3 + 2] = 0.0

        # Action features: read from original s (before masking)
        a4 = np.stack([
            s[idx, pids * 3 + 0],   # remaining_burst / 60.0
            s[idx, pids * 3 + 1],   # arrived_flag
            s[idx, pids * 3 + 2],   # wait_time / 300.0
            qts / 2.0,              # quantum tier normalised (0, 0.5, 1.0)
        ], axis=1)                  # (batch, 4)

        return np.concatenate([s_masked, a4], axis=1)  # (batch, 19)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        X:          np.ndarray,
        use_target: bool = False,
    ) -> np.ndarray:
        """X: (batch, 19) → (batch, 1)."""
        W = self._tW if use_target else self._W
        b = self._tb if use_target else self._b
        h = X.astype(np.float64)
        for i, (w, bi) in enumerate(zip(W, b)):
            z = h @ w + bi
            h = np.maximum(0.0, z) if i < len(W) - 1 else z
        return h   # (batch, 1)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state:         np.ndarray,
        epsilon:       float,
        valid_actions: list[int],
    ) -> int:
        """Masked epsilon-greedy via one batched forward pass."""
        if not valid_actions:
            raise RuntimeError("No valid actions — env bug")
        if random.random() < epsilon:
            return random.choice(valid_actions)

        n_valid    = len(valid_actions)
        states_rep = np.tile(state, (n_valid, 1))           # (n_valid, 15)
        X          = self._build_input_batch(
            states_rep, np.array(valid_actions, dtype=np.int32)
        )                                                    # (n_valid, 19)
        Q          = self.forward_batch(X).flatten()         # (n_valid,)
        return valid_actions[int(np.argmax(Q))]

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update_online(
        self,
        states:      np.ndarray,   # (batch, 15)
        actions:     np.ndarray,   # (batch,)  int32
        rewards:     np.ndarray,   # (batch,)
        next_states: np.ndarray,   # (batch, 15)
        dones:       np.ndarray,   # (batch,)  float 0/1
    ) -> float:
        """One gradient step on the online network.

        Target y_i = r_i + γ · max_{a valid} Q_target(s'_i, a)   [non-terminal]
               y_i = r_i                                           [terminal]

        Q_next computed via (batch × N_ACTIONS = 480) forward pass through
        the target network in one call; invalid actions masked to -inf.
        """
        batch = len(states)
        N     = self.N_ACTIONS
        ns    = np.asarray(next_states, dtype=np.float64)

        # --- Q_pred: online network forward (save activations for backprop) ---
        X_pred = self._build_input_batch(states, actions)   # (batch, 19)
        h0     = X_pred.astype(np.float64)
        z1     = h0 @ self._W[0] + self._b[0]              # (batch, 64)
        h1     = np.maximum(0.0, z1)
        z2     = h1 @ self._W[1] + self._b[1]              # (batch, 32)
        h2     = np.maximum(0.0, z2)
        z3     = h2 @ self._W[2] + self._b[2]              # (batch, 1)
        q_pred = z3.flatten()                               # (batch,)

        # --- Q_next: target network, all N_ACTIONS per next-state sample ---
        all_actions = np.tile(np.arange(N), batch)          # (batch*N,)
        all_next    = np.repeat(ns, N, axis=0)              # (batch*N, 15)
        X_next      = self._build_input_batch(all_next, all_actions)  # (batch*N, 19)
        Q_next_mat  = (
            self.forward_batch(X_next, use_target=True)
                .flatten()
                .reshape(batch, N)
        )                                                    # (batch, N)

        # Valid-action mask: arrived AND remaining > 0
        valid_mask = np.zeros((batch, N), dtype=bool)
        for pid in range(self.N_PROCESSES):
            runnable = (ns[:, pid * 3 + 1] > 0.5) & (ns[:, pid * 3 + 0] > 1e-6)
            for qt in range(self.N_QT):
                valid_mask[:, pid * self.N_QT + qt] = runnable

        Q_next_mat[~valid_mask] = -np.inf
        all_invalid = ~np.any(valid_mask, axis=1)           # fully terminal rows
        max_q_next  = np.where(all_invalid, 0.0, Q_next_mat.max(axis=1))

        targets = rewards + self.gamma * max_q_next * (1.0 - dones)  # (batch,)

        # --- Loss + backprop -------------------------------------------
        delta = q_pred - targets                            # (batch,)
        loss  = float(np.mean(delta ** 2))

        dz3 = (2.0 * delta / batch).reshape(-1, 1)         # (batch, 1)

        dW2 = h2.T @ dz3                                    # (32, 1)
        db2 = dz3.sum(axis=0)                               # (1,)
        dh2 = dz3 @ self._W[2].T                            # (batch, 32)
        dz2 = dh2 * (h2 > 0).astype(np.float64)

        dW1 = h1.T @ dz2                                    # (64, 32)
        db1 = dz2.sum(axis=0)                               # (32,)
        dh1 = dz2 @ self._W[1].T                            # (batch, 64)
        dz1 = dh1 * (h1 > 0).astype(np.float64)

        dW0 = h0.T @ dz1                                    # (19, 64)
        db0 = dz1.sum(axis=0)                               # (64,)

        # Global norm clipping
        grads = [dW0, db0, dW1, db1, dW2, db2]
        global_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if global_norm > self.grad_clip:
            scale = self.grad_clip / global_norm
            grads = [g * scale for g in grads]

        params = [self._W[0], self._b[0], self._W[1], self._b[1],
                  self._W[2], self._b[2]]
        self._opt.step(params, grads)
        return loss

    # ------------------------------------------------------------------
    # Target network / epsilon / persistence
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        self._tW, self._tb = self._copy_weights(self._W, self._b)

    def decay_epsilon(
        self,
        episode:  int,
        min_eps:  float = 0.05,
        decay:    float = 0.9995,
    ) -> float:
        self.epsilon = max(min_eps, self._init_epsilon * (decay ** episode))
        return self.epsilon

    def save(self, path: str) -> None:
        np.savez(
            path,
            W0=self._W[0], b0=self._b[0],
            W1=self._W[1], b1=self._b[1],
            W2=self._W[2], b2=self._b[2],
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self._W = [data["W0"], data["W1"], data["W2"]]
        self._b = [data["b0"], data["b1"], data["b2"]]
        self.update_target()


# ---------------------------------------------------------------------------
# Sum-pooling DQN (Week 6) — full permutation invariance
# ---------------------------------------------------------------------------

class SumPoolingDQN:
    """Week 6: sum-pooling DQN for full permutation invariance.

    Architecture  : 7 → 64(ReLU) → 32(ReLU) → 1(linear)  [2,625 params]
    Input         : [competitor_pool(3) ‖ a4(4)]
    Output        : Q_scalar

    Input construction for action (pid, qt):
        competitor_pool = Σ_{i ≠ pid, arrived, not-complete}
                            [remaining_i/BURST_P95, arrived_flag_i, wait_i/WAIT_NORM]
        a4              = [remaining_pid/BURST_P95, arrived_flag_pid,
                           wait_pid/WAIT_NORM, qt/2.0]
        input_7         = concat(competitor_pool, a4)

    Invariance property: sum-pooling is commutative.  Permuting competitor
    PIDs (assigning the same burst magnitudes to different PIDs) leaves
    competitor_pool unchanged.  Combined with a4 which describes only the
    candidate (not its PID position), Q is invariant to ALL permutations of
    the 5-process set — both candidate PID and competitor PIDs simultaneously.

    This is strictly stronger than Week 5 (ActionConditionedDQN), which
    was invariant to candidate PID but not to competitor PID ordering.

    Information cost: the network sees only the SUM of competitor features,
    not individual magnitudes.  It cannot distinguish one competitor at 10ms
    from two competitors summing to 10ms.  For n_active=2 (one competitor)
    this is lossless; for n_active≥3 this introduces a fundamental ceiling
    below 100% SRPT agreement (see Week 6 spec).
    """

    LAYER_SIZES = (7, 64, 32, 1)
    N_PROCESSES = 5
    N_QT        = 3
    N_ACTIONS   = 15

    def __init__(
        self,
        n_actions:  int   = 15,
        state_dim:  int   = 15,
        lr:         float = 0.001,
        gamma:      float = 1.0,
        epsilon:    float = 1.0,
        grad_clip:  float = 1.0,
    ) -> None:
        self.n_actions     = n_actions
        self.state_dim     = state_dim
        self.gamma         = gamma
        self.epsilon       = epsilon
        self._init_epsilon = epsilon
        self.grad_clip     = grad_clip

        self._W, self._b   = self._init_weights()
        self._tW, self._tb = self._copy_weights(self._W, self._b)
        self._opt          = AdamOptimizer(lr=lr)

    # ------------------------------------------------------------------
    # Weight helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        sizes = self.LAYER_SIZES
        W, b = [], []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            std = np.sqrt(2.0 / fan_in)
            W.append(np.random.randn(fan_in, fan_out).astype(np.float64) * std)
            b.append(np.zeros(fan_out, dtype=np.float64))
        return W, b

    @staticmethod
    def _copy_weights(
        W: list[np.ndarray], b: list[np.ndarray]
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return [w.copy() for w in W], [bi.copy() for bi in b]

    # ------------------------------------------------------------------
    # Input construction: [competitor_pool ‖ a4]
    # ------------------------------------------------------------------

    def _build_input_batch(
        self,
        states:  np.ndarray,   # (batch, 15)
        actions: np.ndarray,   # (batch,)  int
    ) -> np.ndarray:
        """Build (batch, 7) input matrix via sum-pooling over competitors.

        competitor_pool[j] = Σ_{i ≠ pid, arrived, not-complete} s[i, j]
                             where j ∈ {remaining/BURST_P95, arrived_flag, wait/WAIT_NORM}

        a4 = [remaining_pid/BURST_P95, arrived_flag_pid, wait_pid/WAIT_NORM, qt/2.0]
        """
        batch = len(states)
        s     = np.asarray(states,  dtype=np.float64)
        a     = np.asarray(actions, dtype=np.int32)
        pids  = a // self.N_QT
        qts   = a %  self.N_QT
        idx   = np.arange(batch)

        # Reshape to (batch, N_PROCESSES, 3) for per-process access
        s_3d = s.reshape(batch, self.N_PROCESSES, 3)

        # Competitor mask: process i is a valid competitor if:
        #   arrived    (feature[1] > 0.5)
        #   not complete (feature[0] > 1e-6, i.e. remaining_burst > 0)
        #   not the candidate (i ≠ pid)
        arrived   = s_3d[:, :, 1] > 0.5    # (batch, 5)
        runnable  = s_3d[:, :, 0] > 1e-6   # (batch, 5)
        comp_mask = arrived & runnable      # (batch, 5)
        comp_mask[idx, pids] = False        # exclude candidate PID

        # Sum-pool: (batch, 3)
        competitor_pool = (s_3d * comp_mask[:, :, None]).sum(axis=1)

        # Candidate features + quantum tier: (batch, 4)
        a4 = np.stack([
            s_3d[idx, pids, 0],   # remaining / 60.0
            s_3d[idx, pids, 1],   # arrived_flag
            s_3d[idx, pids, 2],   # wait / 300.0
            qts / 2.0,            # quantum tier normalised (0, 0.5, 1.0)
        ], axis=1)

        return np.concatenate([competitor_pool, a4], axis=1)  # (batch, 7)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        X:          np.ndarray,
        use_target: bool = False,
    ) -> np.ndarray:
        """X: (batch, 7) → (batch, 1)."""
        W = self._tW if use_target else self._W
        b = self._tb if use_target else self._b
        h = X.astype(np.float64)
        for i, (w, bi) in enumerate(zip(W, b)):
            z = h @ w + bi
            h = np.maximum(0.0, z) if i < len(W) - 1 else z
        return h   # (batch, 1)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state:         np.ndarray,
        epsilon:       float,
        valid_actions: list[int],
    ) -> int:
        """Masked epsilon-greedy via one batched forward pass."""
        if not valid_actions:
            raise RuntimeError("No valid actions — env bug")
        if random.random() < epsilon:
            return random.choice(valid_actions)

        n_valid    = len(valid_actions)
        states_rep = np.tile(state, (n_valid, 1))           # (n_valid, 15)
        X          = self._build_input_batch(
            states_rep, np.array(valid_actions, dtype=np.int32)
        )                                                    # (n_valid, 7)
        Q          = self.forward_batch(X).flatten()         # (n_valid,)
        return valid_actions[int(np.argmax(Q))]

    # ------------------------------------------------------------------
    # Learning update
    # ------------------------------------------------------------------

    def update_online(
        self,
        states:      np.ndarray,   # (batch, 15)
        actions:     np.ndarray,   # (batch,)  int32
        rewards:     np.ndarray,   # (batch,)
        next_states: np.ndarray,   # (batch, 15)
        dones:       np.ndarray,   # (batch,)  float 0/1
    ) -> float:
        """One gradient step on the online network.

        Target y_i = r_i + γ · max_{a valid} Q_target(s'_i, a)   [non-terminal]
               y_i = r_i                                           [terminal]

        Q_next computed via (batch × N_ACTIONS = 480) forward pass through
        the target network in one call; invalid actions masked to -inf.
        """
        batch = len(states)
        N     = self.N_ACTIONS
        ns    = np.asarray(next_states, dtype=np.float64)

        # --- Q_pred: online network forward (save activations for backprop) ---
        X_pred = self._build_input_batch(states, actions)   # (batch, 7)
        h0     = X_pred.astype(np.float64)
        z1     = h0 @ self._W[0] + self._b[0]              # (batch, 64)
        h1     = np.maximum(0.0, z1)
        z2     = h1 @ self._W[1] + self._b[1]              # (batch, 32)
        h2     = np.maximum(0.0, z2)
        z3     = h2 @ self._W[2] + self._b[2]              # (batch, 1)
        q_pred = z3.flatten()                               # (batch,)

        # --- Q_next: target network, all N_ACTIONS per next-state sample ---
        all_actions = np.tile(np.arange(N), batch)          # (batch*N,)
        all_next    = np.repeat(ns, N, axis=0)              # (batch*N, 15)
        X_next      = self._build_input_batch(all_next, all_actions)  # (batch*N, 7)
        Q_next_mat  = (
            self.forward_batch(X_next, use_target=True)
                .flatten()
                .reshape(batch, N)
        )                                                    # (batch, N)

        # Valid-action mask: arrived AND remaining > 0
        valid_mask = np.zeros((batch, N), dtype=bool)
        for pid in range(self.N_PROCESSES):
            runnable = (ns[:, pid * 3 + 1] > 0.5) & (ns[:, pid * 3 + 0] > 1e-6)
            for qt in range(self.N_QT):
                valid_mask[:, pid * self.N_QT + qt] = runnable

        Q_next_mat[~valid_mask] = -np.inf
        all_invalid = ~np.any(valid_mask, axis=1)           # fully terminal rows
        max_q_next  = np.where(all_invalid, 0.0, Q_next_mat.max(axis=1))

        targets = rewards + self.gamma * max_q_next * (1.0 - dones)  # (batch,)

        # --- Loss + backprop -------------------------------------------
        delta = q_pred - targets                            # (batch,)
        loss  = float(np.mean(delta ** 2))

        dz3 = (2.0 * delta / batch).reshape(-1, 1)         # (batch, 1)

        dW2 = h2.T @ dz3                                    # (32, 1)
        db2 = dz3.sum(axis=0)                               # (1,)
        dh2 = dz3 @ self._W[2].T                            # (batch, 32)
        dz2 = dh2 * (h2 > 0).astype(np.float64)

        dW1 = h1.T @ dz2                                    # (64, 32)
        db1 = dz2.sum(axis=0)                               # (32,)
        dh1 = dz2 @ self._W[1].T                            # (batch, 64)
        dz1 = dh1 * (h1 > 0).astype(np.float64)

        dW0 = h0.T @ dz1                                    # (7, 64)
        db0 = dz1.sum(axis=0)                               # (64,)

        # Global norm clipping
        grads = [dW0, db0, dW1, db1, dW2, db2]
        global_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        if global_norm > self.grad_clip:
            scale = self.grad_clip / global_norm
            grads = [g * scale for g in grads]

        params = [self._W[0], self._b[0], self._W[1], self._b[1],
                  self._W[2], self._b[2]]
        self._opt.step(params, grads)
        return loss

    # ------------------------------------------------------------------
    # Target network / epsilon / persistence
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        self._tW, self._tb = self._copy_weights(self._W, self._b)

    def decay_epsilon(
        self,
        episode:  int,
        min_eps:  float = 0.05,
        decay:    float = 0.9995,
    ) -> float:
        self.epsilon = max(min_eps, self._init_epsilon * (decay ** episode))
        return self.epsilon

    def save(self, path: str) -> None:
        np.savez(
            path,
            W0=self._W[0], b0=self._b[0],
            W1=self._W[1], b1=self._b[1],
            W2=self._W[2], b2=self._b[2],
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self._W = [data["W0"], data["W1"], data["W2"]]
        self._b = [data["b0"], data["b1"], data["b2"]]
        self.update_target()


# ---------------------------------------------------------------------------
# Attention-based DQN (Week 7) — full permutation invariance via attention
# ---------------------------------------------------------------------------

class AttentionDQN:
    """Week 7: dot-product attention over competitor set.

    Architecture:
        Attention projections (3 → 8 each):
            W_Q, b_Q  — query from candidate encoding
            W_K, b_K  — keys  from each competitor encoding
            W_V, b_V  — values from each competitor encoding
        Attention:
            q      = cand_enc @ W_Q + b_Q           (batch, 8)
            K[j]   = comp_enc_j @ W_K + b_K         (batch, 4, 8)
            V[j]   = comp_enc_j @ W_V + b_V         (batch, 4, 8)
            score_j = (q · K[j]) / sqrt(8)          (batch, 4)
            w      = masked_softmax(scores)          (batch, 4)
            context= Σ_j w_j · V[j]                 (batch, 8)
        MLP: [context(8) ‖ a4(4)] = 12 → 64(ReLU) → 32(ReLU) → 1(linear)
        Total: 3,041 parameters

    Permutation invariance: softmax over competitor set is commutative —
    permuting which PID holds which burst value leaves context unchanged.
    Combined with candidate-only a4, Q is invariant to ALL permutations.
    """

    D_ATTN      = 8
    D_V         = 8
    D_CAND      = 3
    N_PROCESSES = 5
    N_QT        = 3
    N_ACTIONS   = 15
    QT_VALUES   = np.array([1.0, 5.0, 20.0])

    def __init__(
        self,
        n_actions:   int   = 15,    # ignored — architecture is fixed
        state_dim:   int   = 15,    # ignored — architecture is fixed
        gamma:       float = 0.99,
        lr:          float = 3e-4,
        epsilon:     float = 1.0,
        buffer_size: int   = 50_000,
        batch_size:  int   = 256,
        target_freq: int   = 500,
        grad_clip:   float = 10.0,
        lambda_ent:  float = 0.10,  # entropy regularisation coefficient
    ) -> None:
        self.gamma         = gamma
        self.lr            = lr
        self.epsilon       = epsilon
        self._init_epsilon = epsilon
        self.batch_size    = batch_size
        self.target_freq   = target_freq
        self.grad_clip     = grad_clip
        self.lambda_ent    = lambda_ent

        self.replay_buffer = ReplayBuffer(buffer_size, state_dim=15)

        rng = np.random.default_rng(0)

        def he(fan_in: int, *shape: int) -> np.ndarray:
            return rng.standard_normal(shape).astype(np.float64) * np.sqrt(2.0 / fan_in)

        # Attention projections: 3 → D_ATTN (or D_V for W_V)
        self.W_Q = he(3, 3, self.D_ATTN); self.b_Q = np.zeros(self.D_ATTN)
        self.W_K = he(3, 3, self.D_ATTN); self.b_K = np.zeros(self.D_ATTN)
        self.W_V = he(3, 3, self.D_V);    self.b_V = np.zeros(self.D_V)

        # MLP: 12 → 64 → 32 → 1
        self._W = [
            he(12, 12, 64),
            he(64, 64, 32),
            he(32, 32,  1),
        ]
        self._b = [
            np.zeros(64),
            np.zeros(32),
            np.zeros(1),
        ]

        # Target network copies
        self._copy_attn_target()
        self._tW, self._tb = self._copy_mlp_weights(self._W, self._b)

        self._opt = AdamOptimizer(lr=lr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _copy_attn_target(self) -> None:
        self._tW_Q = self.W_Q.copy(); self._tb_Q = self.b_Q.copy()
        self._tW_K = self.W_K.copy(); self._tb_K = self.b_K.copy()
        self._tW_V = self.W_V.copy(); self._tb_V = self.b_V.copy()

    @staticmethod
    def _copy_mlp_weights(W, b):
        return [w.copy() for w in W], [bi.copy() for bi in b]

    def _build_competitor_data(
        self,
        states: np.ndarray,  # (batch, 15)
        pids:   np.ndarray,  # (batch,)  int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return comp_encs (batch,4,3) and comp_valid (batch,4) bool."""
        batch    = states.shape[0]
        s3d      = states.reshape(batch, self.N_PROCESSES, 3)
        all_pids = np.tile(np.arange(self.N_PROCESSES), (batch, 1))  # (batch, 5)
        mask     = all_pids != pids[:, None]                          # (batch, 5)
        comp_idx = all_pids[mask].reshape(batch, 4)                   # (batch, 4)
        bidx     = np.arange(batch)[:, None]                          # (batch, 1)
        comp_encs  = s3d[bidx, comp_idx]                              # (batch, 4, 3)
        comp_valid = (comp_encs[:, :, 1] > 0.5) & (comp_encs[:, :, 0] > 1e-6)
        return comp_encs, comp_valid

    def _attention_forward(
        self,
        cand_enc:   np.ndarray,  # (batch, 3)
        comp_encs:  np.ndarray,  # (batch, 4, 3)
        comp_valid: np.ndarray,  # (batch, 4) bool
        use_target: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Compute attention context.  Returns (context, cache)."""
        WQ = self._tW_Q if use_target else self.W_Q
        bQ = self._tb_Q if use_target else self.b_Q
        WK = self._tW_K if use_target else self.W_K
        bK = self._tb_K if use_target else self.b_K
        WV = self._tW_V if use_target else self.W_V
        bV = self._tb_V if use_target else self.b_V

        batch     = cand_enc.shape[0]
        comp_flat = comp_encs.reshape(batch * 4, 3)          # (batch*4, 3)

        # Query from candidate
        q = cand_enc @ WQ + bQ                               # (batch, D_ATTN)

        # Keys and Values from all competitors
        K_flat = comp_flat @ WK + bK                         # (batch*4, D_ATTN)
        V_flat = comp_flat @ WV + bV                         # (batch*4, D_V)
        K = K_flat.reshape(batch, 4, self.D_ATTN)            # (batch, 4, D_ATTN)
        V = V_flat.reshape(batch, 4, self.D_V)               # (batch, 4, D_V)

        # Dot-product scores: score_j = (q · k_j) / sqrt(D_ATTN)
        scores = np.einsum('bi,bji->bj', q, K) / np.sqrt(self.D_ATTN)  # (batch, 4)

        # Masked softmax: invalid slots → -1e9 before exp, 0 after exp
        invalid       = ~comp_valid                           # (batch, 4)
        scores_masked = scores.copy()
        scores_masked[invalid] = -1e9
        scores_shifted = scores_masked - scores_masked.max(axis=1, keepdims=True)
        exp_s          = np.exp(scores_shifted)
        exp_s[invalid] = 0.0                                 # explicitly zero invalid
        denom          = exp_s.sum(axis=1, keepdims=True) + 1e-10
        weights        = exp_s / denom                       # (batch, 4)

        # Attended context
        context = np.einsum('bj,bjd->bd', weights, V)        # (batch, D_V)

        cache = dict(
            cand_enc=cand_enc, comp_encs=comp_encs, comp_flat=comp_flat,
            comp_valid=comp_valid, invalid=invalid,
            q=q, K=K, V=V, scores=scores, weights=weights, context=context,
        )
        return context, cache

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward_batch(
        self,
        states:     np.ndarray,  # (batch, 15)
        actions:    np.ndarray,  # (batch,) int
        use_target: bool = False,
    ) -> np.ndarray:             # (batch, 1)
        batch    = states.shape[0]
        pids     = (actions // self.N_QT).astype(np.int32)
        qts      = self.QT_VALUES[actions % self.N_QT]       # (batch,)

        s3d      = states.reshape(batch, self.N_PROCESSES, 3)
        cand_enc = s3d[np.arange(batch), pids]               # (batch, 3)
        comp_encs, comp_valid = self._build_competitor_data(states, pids)

        context, _ = self._attention_forward(
            cand_enc, comp_encs, comp_valid, use_target
        )

        a4    = np.column_stack([cand_enc, qts / 2.0])       # (batch, 4)
        x     = np.concatenate([context, a4], axis=1)        # (batch, 12)

        W, b  = (self._tW, self._tb) if use_target else (self._W, self._b)
        z1 = x  @ W[0] + b[0]
        h1 = np.maximum(0.0, z1)
        z2 = h1 @ W[1] + b[1]
        h2 = np.maximum(0.0, z2)
        z3 = h2 @ W[2] + b[2]
        return z3                                             # (batch, 1)

    def select_action(
        self,
        state:         np.ndarray,  # (15,)
        epsilon:       float,
        valid_actions: list[int],
    ) -> int:
        if np.random.random() < epsilon or not valid_actions:
            return int(np.random.choice(valid_actions))
        states_b  = np.tile(state, (len(valid_actions), 1))
        actions_b = np.array(valid_actions, dtype=np.int32)
        q_vals    = self.forward_batch(states_b, actions_b).flatten()
        return valid_actions[int(np.argmax(q_vals))]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update_online(
        self,
        states:      np.ndarray,  # (batch, 15)
        actions:     np.ndarray,  # (batch,) int
        rewards:     np.ndarray,  # (batch,)
        next_states: np.ndarray,  # (batch, 15)
        dones:       np.ndarray,  # (batch,) float 0/1
    ) -> tuple[float, float]:
        """One gradient step.  Returns (loss, attention_entropy)."""
        batch = states.shape[0]
        N     = self.N_ACTIONS
        pids  = (actions // self.N_QT).astype(np.int32)
        qts   = self.QT_VALUES[actions % self.N_QT]

        s3d      = states.reshape(batch, self.N_PROCESSES, 3)
        cand_enc = s3d[np.arange(batch), pids]               # (batch, 3)
        comp_encs, comp_valid = self._build_competitor_data(states, pids)

        # --- Forward pass (online network) ---
        context, attn_cache = self._attention_forward(cand_enc, comp_encs, comp_valid)

        a4    = np.column_stack([cand_enc, qts / 2.0])       # (batch, 4)
        x_mlp = np.concatenate([context, a4], axis=1)        # (batch, 12)

        z1 = x_mlp @ self._W[0] + self._b[0]
        h1 = np.maximum(0.0, z1)
        z2 = h1    @ self._W[1] + self._b[1]
        h2 = np.maximum(0.0, z2)
        z3 = h2    @ self._W[2] + self._b[2]
        q_pred = z3.flatten()

        # Attention entropy: H = -mean_batch Σ_j w_j log(w_j + 1e-8)
        weights = attn_cache['weights']                       # (batch, 4)
        valid_f = comp_valid.astype(np.float64)
        ent_per = -(weights * np.log(weights + 1e-8) * valid_f).sum(axis=1)
        H_batch = float(np.mean(ent_per))
        entropy = H_batch

        # --- Target Q (target network, all actions) ---
        ns          = np.asarray(next_states, dtype=np.float64)
        all_acts    = np.tile(np.arange(N), batch)
        all_ns      = np.repeat(ns, N, axis=0)
        Q_next_flat = self.forward_batch(all_ns, all_acts, use_target=True).flatten()
        Q_next_mat  = Q_next_flat.reshape(batch, N)

        valid_mask = np.zeros((batch, N), dtype=bool)
        for pid in range(self.N_PROCESSES):
            runnable = (ns[:, pid * 3 + 1] > 0.5) & (ns[:, pid * 3 + 0] > 1e-6)
            for qt in range(self.N_QT):
                valid_mask[:, pid * self.N_QT + qt] = runnable

        Q_next_mat[~valid_mask] = -np.inf
        all_invalid = ~np.any(valid_mask, axis=1)
        max_q_next  = np.where(all_invalid, 0.0, Q_next_mat.max(axis=1))

        targets = rewards + self.gamma * max_q_next * (1.0 - dones)

        # --- Loss: L_total = L_td - lambda_ent * H  (maximise entropy) ---
        delta = q_pred - targets
        L_td  = float(np.mean(delta ** 2))
        loss  = L_td - self.lambda_ent * H_batch

        # --- MLP backward ---
        dz3 = (2.0 * delta / batch).reshape(-1, 1)

        dW2   = h2.T @ dz3
        db2   = dz3.sum(axis=0)
        dh2   = dz3 @ self._W[2].T
        dz2   = dh2 * (h2 > 0).astype(np.float64)

        dW1   = h1.T @ dz2
        db1   = dz2.sum(axis=0)
        dh1   = dz2 @ self._W[1].T
        dz1   = dh1 * (h1 > 0).astype(np.float64)

        dW0   = x_mlp.T @ dz1
        db0   = dz1.sum(axis=0)
        dx_mlp = dz1 @ self._W[0].T                          # (batch, 12)

        # Gradient w.r.t. context (first 8 dims of dx_mlp)
        d_context = dx_mlp[:, :self.D_V]                     # (batch, D_V)

        # --- Attention backward ---
        q_attn    = attn_cache['q']        # (batch, D_ATTN)
        K         = attn_cache['K']        # (batch, 4, D_ATTN)
        V         = attn_cache['V']        # (batch, 4, D_V)
        invalid   = attn_cache['invalid']  # (batch, 4) bool
        comp_flat = attn_cache['comp_flat']  # (batch*4, 3)

        # Backprop through context = einsum('bj,bjd->bd', weights, V)
        d_weights = np.einsum('bd,bjd->bj', d_context, V)    # (batch, 4)
        dV        = weights[:, :, None] * d_context[:, None, :]  # (batch, 4, D_V)
        dV[invalid] = 0.0

        # Entropy regularisation gradient:
        # d(-lambda*H)/d_w_bj = lambda * (log(w_bj+1e-8) + w_bj/(w_bj+1e-8)) * valid / batch
        if self.lambda_ent > 0:
            d_ent = (self.lambda_ent / batch) * (
                np.log(weights + 1e-8) + weights / (weights + 1e-8)
            ) * valid_f
            d_weights = d_weights + d_ent

        # Backprop through softmax (standard formula)
        wdotdw          = (weights * d_weights).sum(axis=1, keepdims=True)
        d_scores_masked = weights * (d_weights - wdotdw)      # (batch, 4)
        d_scores_masked[invalid] = 0.0
        d_raw_scores = d_scores_masked / np.sqrt(self.D_ATTN) # (batch, 4)

        # Backprop through scores = einsum('bi,bji->bj', q, K)
        dq = np.einsum('bj,bji->bi', d_raw_scores, K)         # (batch, D_ATTN)
        dK = d_raw_scores[:, :, None] * q_attn[:, None, :]    # (batch, 4, D_ATTN)
        dK[invalid] = 0.0

        # Backprop through projections
        dW_Q  = cand_enc.T   @ dq                                       # (3, D_ATTN)
        db_Q  = dq.sum(axis=0)                                          # (D_ATTN,)

        dK_flat = dK.reshape(batch * 4, self.D_ATTN)
        dW_K  = comp_flat.T  @ dK_flat                                  # (3, D_ATTN)
        db_K  = dK_flat.sum(axis=0)                                     # (D_ATTN,)

        dV_flat = dV.reshape(batch * 4, self.D_V)
        dW_V  = comp_flat.T  @ dV_flat                                  # (3, D_V)
        db_V  = dV_flat.sum(axis=0)                                     # (D_V,)

        # --- Global norm clipping ---
        all_grads = [dW0, db0, dW1, db1, dW2, db2,
                     dW_Q, db_Q, dW_K, db_K, dW_V, db_V]
        global_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in all_grads)))
        if global_norm > self.grad_clip:
            scale     = self.grad_clip / global_norm
            all_grads = [g * scale for g in all_grads]

        # --- Parameter update ---
        all_params = [self._W[0], self._b[0], self._W[1], self._b[1],
                      self._W[2], self._b[2],
                      self.W_Q, self.b_Q, self.W_K, self.b_K, self.W_V, self.b_V]
        self._opt.step(all_params, all_grads)

        return loss, entropy

    def gradient_check(
        self,
        states:  np.ndarray,  # (batch, 15)
        actions: np.ndarray,  # (batch,) int
        targets: np.ndarray,  # (batch,) float
        eps:     float = 1e-5,
    ) -> tuple[float, str, bool]:
        """Finite-difference check on ALL parameters.

        Returns (max_rel_err, worst_param_name, pass_bool).
        PASS criterion: max_rel_err < 1e-4.
        """
        batch = states.shape[0]
        pids  = (actions // self.N_QT).astype(np.int32)
        qts   = self.QT_VALUES[actions % self.N_QT]

        s3d      = states.reshape(batch, self.N_PROCESSES, 3)
        cand_enc = s3d[np.arange(batch), pids]
        comp_encs, comp_valid = self._build_competitor_data(states, pids)

        # Analytical gradients — forward + backward (same as update_online)
        context, attn_cache = self._attention_forward(cand_enc, comp_encs, comp_valid)
        a4    = np.column_stack([cand_enc, qts / 2.0])
        x_mlp = np.concatenate([context, a4], axis=1)

        z1 = x_mlp @ self._W[0] + self._b[0]
        h1 = np.maximum(0.0, z1)
        z2 = h1    @ self._W[1] + self._b[1]
        h2 = np.maximum(0.0, z2)
        z3 = h2    @ self._W[2] + self._b[2]
        q_pred = z3.flatten()

        delta = q_pred - targets
        dz3   = (2.0 * delta / batch).reshape(-1, 1)

        dW2   = h2.T @ dz3
        db2   = dz3.sum(axis=0)
        dh2   = dz3 @ self._W[2].T
        dz2   = dh2 * (h2 > 0).astype(np.float64)

        dW1   = h1.T @ dz2
        db1   = dz2.sum(axis=0)
        dh1   = dz2 @ self._W[1].T
        dz1   = dh1 * (h1 > 0).astype(np.float64)

        dW0   = x_mlp.T @ dz1
        db0   = dz1.sum(axis=0)
        dx_mlp = dz1 @ self._W[0].T

        d_context = dx_mlp[:, :self.D_V]

        q_attn    = attn_cache['q']
        K         = attn_cache['K']
        V         = attn_cache['V']
        invalid   = attn_cache['invalid']
        comp_flat = attn_cache['comp_flat']

        gc_weights = attn_cache['weights']                  # (batch, 4)
        d_weights = np.einsum('bd,bjd->bj', d_context, V)
        dV        = gc_weights[:, :, None] * d_context[:, None, :]  # (batch, 4, D_V)
        dV[invalid] = 0.0

        wdotdw          = (gc_weights * d_weights).sum(axis=1, keepdims=True)
        d_scores_masked = gc_weights * (d_weights - wdotdw)
        d_scores_masked[invalid] = 0.0
        d_raw_scores = d_scores_masked / np.sqrt(self.D_ATTN)

        dq = np.einsum('bj,bji->bi', d_raw_scores, K)
        dK = d_raw_scores[:, :, None] * q_attn[:, None, :]
        dK[invalid] = 0.0

        dW_Q  = cand_enc.T   @ dq
        db_Q  = dq.sum(axis=0)
        dK_flat = dK.reshape(batch * 4, self.D_ATTN)
        dW_K  = comp_flat.T  @ dK_flat
        db_K  = dK_flat.sum(axis=0)
        dV_flat = dV.reshape(batch * 4, self.D_V)
        dW_V  = comp_flat.T  @ dV_flat
        db_V  = dV_flat.sum(axis=0)

        analytic = {
            'W_Q': (self.W_Q, dW_Q), 'b_Q': (self.b_Q, db_Q),
            'W_K': (self.W_K, dW_K), 'b_K': (self.b_K, db_K),
            'W_V': (self.W_V, dW_V), 'b_V': (self.b_V, db_V),
            'W0':  (self._W[0], dW0), 'b0': (self._b[0], db0),
            'W1':  (self._W[1], dW1), 'b1': (self._b[1], db1),
            'W2':  (self._W[2], dW2), 'b2': (self._b[2], db2),
        }

        def _loss(s, a, t):
            return float(np.mean((self.forward_batch(s, a).flatten() - t) ** 2))

        max_rel_err = 0.0
        worst_name  = ''
        for name, (param, grad_a) in analytic.items():
            for idx in range(param.size):
                orig = param.flat[idx]
                param.flat[idx] = orig + eps
                lp = _loss(states, actions, targets)
                param.flat[idx] = orig - eps
                lm = _loss(states, actions, targets)
                param.flat[idx] = orig
                num_g = (lp - lm) / (2.0 * eps)
                ana_g = float(grad_a.flat[idx])
                abs_err = abs(ana_g - num_g)
                # When both gradients are negligible (< 1e-6), floating-point
                # residuals dominate.  Use absolute error in that regime.
                magnitude = abs(ana_g) + abs(num_g)
                if magnitude < 1e-6:
                    eff_err = abs_err  # absolute check; 1e-8 << 1e-4 passes
                else:
                    eff_err = abs_err / (magnitude + 1e-8)
                if eff_err > max_rel_err:
                    max_rel_err = eff_err
                    worst_name  = f"{name}[{idx}]"

        return max_rel_err, worst_name, max_rel_err < 1e-4

    # ------------------------------------------------------------------
    # Target network / epsilon / persistence
    # ------------------------------------------------------------------

    def update_target(self) -> None:
        self._copy_attn_target()
        self._tW, self._tb = self._copy_mlp_weights(self._W, self._b)

    def decay_epsilon(
        self,
        episode:  int,
        min_eps:  float = 0.05,
        decay:    float = 0.9995,
    ) -> float:
        self.epsilon = max(min_eps, self._init_epsilon * (decay ** episode))
        return self.epsilon

    def save(self, path: str) -> None:
        np.savez(
            path,
            W_Q=self.W_Q, b_Q=self.b_Q,
            W_K=self.W_K, b_K=self.b_K,
            W_V=self.W_V, b_V=self.b_V,
            W0=self._W[0], b0=self._b[0],
            W1=self._W[1], b1=self._b[1],
            W2=self._W[2], b2=self._b[2],
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        self.W_Q = data['W_Q']; self.b_Q = data['b_Q']
        self.W_K = data['W_K']; self.b_K = data['b_K']
        self.W_V = data['W_V']; self.b_V = data['b_V']
        self._W  = [data['W0'], data['W1'], data['W2']]
        self._b  = [data['b0'], data['b1'], data['b2']]
        self.update_target()


# ---------------------------------------------------------------------------
# Tabular Q-learning agent (Week 1/2 — retained for backward compatibility)
# ---------------------------------------------------------------------------

class QLearningAgent:
    def __init__(
        self,
        n_actions:  int,
        state_bins:  tuple[int, ...],
        alpha:       float = 0.1,
        gamma:       float = 0.99,
        epsilon:     float = 1.0,
    ) -> None:
        self.n_actions   = n_actions
        self.state_bins  = state_bins
        self.alpha       = alpha
        self.gamma       = gamma
        self.epsilon     = epsilon
        self._init_epsilon = epsilon

        self.q_table = np.zeros(state_bins + (n_actions,), dtype=np.float64)

    def select_action(self, state: tuple[int, ...], epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state:      tuple[int, ...],
        action:     int,
        reward:     float,
        next_state: tuple[int, ...],
        done:       bool,
    ) -> float:
        current_q = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * float(np.max(self.q_table[next_state]))
        td_error = target - current_q
        self.q_table[state][action] += self.alpha * td_error
        return td_error

    def decay_epsilon(
        self,
        episode:  int,
        min_eps:  float = 0.05,
        decay:    float = 0.995,
    ) -> float:
        self.epsilon = max(min_eps, self._init_epsilon * (decay ** episode))
        return self.epsilon

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        loaded = np.load(path)
        if loaded.shape != self.q_table.shape:
            raise ValueError(
                f"Shape mismatch on load: expected {self.q_table.shape}, "
                f"got {loaded.shape}"
            )
        self.q_table = loaded


# ---------------------------------------------------------------------------
# Standalone unit tests / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Normalization constants (must match env.py BURST_P95 and WAIT_NORM)
    _BP = 397.0    # BURST_P95
    _WN = 1985.0   # WAIT_NORM = BURST_P95 * N_PROCESSES

    random.seed(7)
    np.random.seed(7)

    # ======================================================================
    # SECTION 1: AdamOptimizer unit test
    # ======================================================================
    print("=" * 64)
    print("AdamOptimizer — unit test")
    print("=" * 64)

    # --- 1a: Single scalar weight, manual verification --------------------
    #   w=1.0, g=0.5, lr=0.001, β1=0.9, β2=0.999, ε=1e-8, t=1
    #   m1     = 0.9·0   + 0.1·0.5   = 0.05
    #   v1     = 0.999·0 + 0.001·0.25 = 0.00025
    #   m_hat  = 0.05   / (1 - 0.9^1)   = 0.5
    #   v_hat  = 0.00025/ (1 - 0.999^1) = 0.25
    #   Δw     = 0.001 · 0.5 / (√0.25 + 1e-8) ≈ 0.001
    #   w_new  = 1.0 - 0.001 = 0.999
    print()
    print("1a. Scalar weight — one step")
    LR, B1, B2, EPS = 0.001, 0.9, 0.999, 1e-8
    opt = AdamOptimizer(lr=LR, beta1=B1, beta2=B2, epsilon=EPS)
    w   = np.array([1.0])
    g   = np.array([0.5])
    opt.step([w], [g])

    m1       = (1 - B1) * 0.5
    v1       = (1 - B2) * 0.25
    m_hat    = m1 / (1 - B1**1)
    v_hat    = v1 / (1 - B2**1)
    expected = 1.0 - LR * m_hat / (np.sqrt(v_hat) + EPS)
    match    = abs(w[0] - expected) < 1e-12

    print(f"  w_new    (computed) = {w[0]:.15f}")
    print(f"  w_new    (expected) = {expected:.15f}")
    print(f"  m1={m1:.6f}  v1={v1:.8f}  m_hat={m_hat:.6f}  v_hat={v_hat:.6f}")
    print(f"  Δw expected         ≈ {LR * m_hat / (np.sqrt(v_hat) + EPS):.6f}")
    print(f"  Match (tol=1e-12)   : {match}")
    assert match, "FAIL: scalar weight mismatch"

    # --- 1b: Two-step test — verify moment accumulation ------------------
    print()
    print("1b. Two-step test — moment accumulation")
    opt2 = AdamOptimizer(lr=LR, beta1=B1, beta2=B2, epsilon=EPS)
    w2   = np.array([2.0])
    g1   = np.array([0.3])
    g2   = np.array([0.7])
    opt2.step([w2], [g1])
    w_after_step1 = float(w2[0])
    opt2.step([w2], [g2])
    w_after_step2 = float(w2[0])

    # Manual step 1
    m_s1 = (1 - B1) * 0.3
    v_s1 = (1 - B2) * 0.3**2
    mh1  = m_s1 / (1 - B1**1)
    vh1  = v_s1 / (1 - B2**1)
    exp1 = 2.0 - LR * mh1 / (np.sqrt(vh1) + EPS)
    # Manual step 2
    m_s2 = B1 * m_s1 + (1 - B1) * 0.7
    v_s2 = B2 * v_s1 + (1 - B2) * 0.7**2
    mh2  = m_s2 / (1 - B1**2)
    vh2  = v_s2 / (1 - B2**2)
    exp2 = exp1 - LR * mh2 / (np.sqrt(vh2) + EPS)

    match1 = abs(w_after_step1 - exp1) < 1e-12
    match2 = abs(w_after_step2 - exp2) < 1e-12
    print(f"  After step 1: computed={w_after_step1:.15f}  expected={exp1:.15f}  match={match1}")
    print(f"  After step 2: computed={w_after_step2:.15f}  expected={exp2:.15f}  match={match2}")
    assert match1 and match2, "FAIL: two-step moment accumulation"

    # --- 1c: Multi-array param test (simulate W + b) ---------------------
    print()
    print("1c. Multi-array test (W matrix + bias vector)")
    opt3 = AdamOptimizer(lr=LR)
    W = np.ones((3, 4), dtype=np.float64)
    b = np.zeros(4,      dtype=np.float64)
    dW = np.full((3, 4), 0.1)
    db = np.full(4,      0.2)
    opt3.step([W, b], [dW, db])
    print(f"  W shape after step : {W.shape}  (all ~{W[0,0]:.6f} — was 1.0)")
    print(f"  b shape after step : {b.shape}  (all ~{b[0]:.6f} — was 0.0)")
    assert W.shape == (3, 4) and b.shape == (4,), "FAIL: shapes changed"
    assert abs(W[0, 0] - W[2, 3]) < 1e-14, "FAIL: W elements should be equal"
    print("  Shape and uniformity: OK")

    print()
    print("AdamOptimizer: ALL ASSERTIONS PASSED")

    # ======================================================================
    # SECTION 2: DQNAgent forward pass
    # ======================================================================
    print()
    print("=" * 64)
    print("DQNAgent — forward pass unit test")
    print("=" * 64)

    np.random.seed(42)
    agent_dqn = DQNAgent(n_actions=15, state_dim=15)

    x = np.random.rand(15).astype(np.float64)
    q_vals, cache = agent_dqn.forward(x)

    print(f"  Input shape        : {x.shape}")
    print(f"  Output shape       : {q_vals.shape}")
    print(f"  Output dtype       : {q_vals.dtype}")
    print(f"  Q-value range      : [{q_vals.min():.4f}, {q_vals.max():.4f}]")
    print(f"  Cache layers       : {len(cache)}")
    assert q_vals.shape == (15,),    "FAIL: output shape != (15,)"
    assert q_vals.dtype == np.float64, "FAIL: dtype != float64"
    assert len(cache) == 4,          "FAIL: cache should have 4 entries (input + 3 layers)"
    print("  DQNAgent forward pass: OK")

    # ======================================================================
    # SECTION 3: ReplayBuffer
    # ======================================================================
    print()
    print("=" * 64)
    print("ReplayBuffer — unit test")
    print("=" * 64)

    buf = ReplayBuffer(capacity=10, state_dim=15)
    for i in range(5):
        s  = np.random.rand(15).astype(np.float32)
        ns = np.random.rand(15).astype(np.float32)
        done = (i == 4)
        buf.store(s, i, float(-i * 2), ns, done)

    print(f"  Buffer size after 5 stores: {len(buf)}")
    assert len(buf) == 5, "FAIL: buffer size"

    s_b, a_b, r_b, ns_b, d_b = buf.sample(3)
    print(f"  Sample shapes: s={s_b.shape}  a={a_b.shape}  r={r_b.shape}  "
          f"ns={ns_b.shape}  d={d_b.shape}")
    assert s_b.shape  == (3, 15), "FAIL: states shape"
    assert a_b.shape  == (3,),    "FAIL: actions shape"
    assert r_b.shape  == (3,),    "FAIL: rewards shape"
    assert ns_b.shape == (3, 15), "FAIL: next_states shape"
    assert d_b.shape  == (3,),    "FAIL: dones shape"

    # Verify terminal entry: done=1.0 for the last stored transition (i=4)
    # Find it in the buffer
    terminal_idx = np.where(d_b == 1.0)[0]
    print(f"  Terminal entries in sample : {len(terminal_idx)} "
          f"(may be 0 — random sample, terminal=last transition)")
    print(f"  All dones in buffer        : {buf._dones[:5]}")
    assert buf._dones[4] == 1.0, "FAIL: terminal done flag not set"

    # Circular overwrite test
    buf2 = ReplayBuffer(capacity=3, state_dim=2)
    for i in range(5):
        buf2.store(np.array([float(i), float(i)]), i, 0.0,
                   np.zeros(2), False)
    print(f"  After 5 stores into cap=3: size={len(buf2)}  ptr={buf2._ptr}")
    assert len(buf2) == 3, "FAIL: circular buffer size should be capped at 3"

    print()
    print("ReplayBuffer: ALL ASSERTIONS PASSED")

    # --- Terminal state target test ---------------------------------------
    print()
    print("3d. Terminal state target test")
    #   Transition 1: (s1, a=0, r=-5.0, s2, done=False)  → bootstrap
    #   Transition 2: (s2, a=1, r=-3.0, s3, done=True)   → no bootstrap
    np.random.seed(0)
    agent_t = DQNAgent(n_actions=15, state_dim=15, gamma=1.0)
    s1 = np.random.rand(15).astype(np.float32)
    s2 = np.random.rand(15).astype(np.float32)
    s3 = np.random.rand(15).astype(np.float32)

    buf_t = ReplayBuffer(capacity=10, state_dim=15)
    buf_t.store(s1, 0, -5.0, s2, done=False)
    buf_t.store(s2, 1, -3.0, s3, done=True)

    # Retrieve both transitions (deterministic: only 2 entries, sample both)
    states_t      = np.stack([s1, s2])
    actions_t     = np.array([0, 1], dtype=np.int32)
    rewards_t     = np.array([-5.0, -3.0], dtype=np.float32)
    next_states_t = np.stack([s2, s3])
    dones_t       = np.array([0.0, 1.0], dtype=np.float32)

    # Compute targets using the same formula as update_online
    q_next_t      = agent_t.forward_batch(next_states_t, use_target=True)
    max_q_next_t  = q_next_t.max(axis=1)
    targets_t     = rewards_t + 1.0 * max_q_next_t * (1.0 - dones_t)

    non_term_target = float(targets_t[0])
    term_target     = float(targets_t[1])

    # What Q(s3) max is (for transparency)
    q_s3_max = float(agent_t.forward_batch(s3[None], use_target=True).max())

    print(f"  Q(s3) max (target net)       : {q_s3_max:+.6f}")
    print(f"  Non-terminal target (r + γ·V): {non_term_target:+.6f}  "
          f"(= -5.0 + {q_next_t[0].max():+.6f})")
    print(f"  Terminal target (r only)     : {term_target:+.6f}  (must == -3.0)")
    print(f"  done flag applied correctly  : {abs(term_target - (-3.0)) < 1e-6}")

    assert abs(term_target - (-3.0)) < 1e-6, (
        f"FAIL: terminal target = {term_target:.8f}, expected -3.0  "
        f"(done mask not applied — Q(s3)={q_s3_max:.4f} was incorrectly added)"
    )
    assert non_term_target != -5.0, "FAIL: non-terminal target should include bootstrap"
    print("  Terminal state target test: PASS")

    # ======================================================================
    # SECTION 4: QLearningAgent (unchanged, smoke test)
    # ======================================================================
    print()
    print("=" * 64)
    print("QLearningAgent — smoke test (backward compat)")
    print("=" * 64)
    random.seed(7); np.random.seed(7)
    STATE_BINS = (7, 7, 7, 7, 7)
    qa = QLearningAgent(n_actions=15, state_bins=STATE_BINS)
    s  = (1, 3, 0, 4, 6)
    qa.q_table[s][6] = 99.0
    greedy = qa.select_action(s, epsilon=0.0)
    print(f"  Greedy on hot state: action={greedy}  expected=6  ok={greedy==6}")
    assert greedy == 6
    ep_at_min = next(ep for ep in range(1, 10_001)
                     if 1.0 * (0.9995**ep) <= 0.05)
    print(f"  Epsilon hits 0.05 at ep {ep_at_min} (decay=0.9995)")
    print("QLearningAgent smoke test: OK")

    # ======================================================================
    # SECTION 5: ActionConditionedDQN — unit tests
    # ======================================================================
    print()
    print("=" * 64)
    print("ActionConditionedDQN — unit tests")
    print("=" * 64)

    np.random.seed(99)
    agent_ac = ActionConditionedDQN()

    # --- 5a: forward pass shape ------------------------------------------
    X_test = np.random.rand(1, 19)
    q_out  = agent_ac.forward_batch(X_test)
    assert q_out.shape == (1, 1), f"FAIL: expected (1,1), got {q_out.shape}"
    print(f"  Forward pass shape : {q_out.shape}  (expected (1,1)) — OK")

    # --- 5b: _build_input_batch masking ----------------------------------
    s_test = np.random.rand(15)
    s_test[0 * 3 : 0 * 3 + 3] = [0.4, 1.0, 0.1]   # set P0 explicitly
    a_test = np.array([0 * 3 + 2], dtype=np.int32)  # action = P0 / long (qt=2)
    X_mask = agent_ac._build_input_batch(s_test[None], a_test)

    # Candidate slot (P0 = [0:3]) should be zeroed in s_masked
    assert X_mask[0, 0] == 0.0 and X_mask[0, 1] == 0.0 and X_mask[0, 2] == 0.0, \
        "FAIL: candidate slot not zeroed in s_masked"
    # a4 should carry P0's original features + qt/2
    assert abs(X_mask[0, 15] - 0.4) < 1e-10, "FAIL: a4[0] (remaining)"
    assert abs(X_mask[0, 16] - 1.0) < 1e-10, "FAIL: a4[1] (arrived)"
    assert abs(X_mask[0, 17] - 0.1) < 1e-10, "FAIL: a4[2] (wait)"
    assert abs(X_mask[0, 18] - 1.0) < 1e-10, "FAIL: a4[3] (qt=2 → 2/2=1.0)"
    print("  _build_input_batch masking         : OK")

    # --- 5c: permutation invariance test ---------------------------------
    # Construct state_A and state_B:
    #   state_A : P0=[0.4,1,0.1], P1=[0.3,1,0.05], P2=P3=P4=[0,0,0]
    #   state_B : P4=[0.4,1,0.1], P1=[0.3,1,0.05], P0=P2=P3=[0,0,0]
    #
    # After masking:
    #   s_masked_A (P0 zeroed) = [0,0,0, 0.3,1,0.05, 0,0,0, 0,0,0, 0,0,0]
    #   s_masked_B (P4 zeroed) = [0,0,0, 0.3,1,0.05, 0,0,0, 0,0,0, 0,0,0]
    #   → IDENTICAL  (P1 competitor occupies slot [3:6] in both)
    #   a4 = [0.4, 1.0, 0.1, 1.0] for both
    #   → full input vectors are identical → Q-values must be equal

    state_A = np.zeros(15)
    state_A[0 * 3 : 0 * 3 + 3] = [0.4, 1.0, 0.1]   # P0 = candidate
    state_A[1 * 3 : 1 * 3 + 3] = [0.3, 1.0, 0.05]  # P1 = competitor

    state_B = np.zeros(15)
    state_B[4 * 3 : 4 * 3 + 3] = [0.4, 1.0, 0.1]   # P4 = candidate
    state_B[1 * 3 : 1 * 3 + 3] = [0.3, 1.0, 0.05]  # P1 = competitor

    X_A = agent_ac._build_input_batch(
        state_A[None], np.array([0 * 3 + 2])        # P0 / long
    )
    X_B = agent_ac._build_input_batch(
        state_B[None], np.array([4 * 3 + 2])        # P4 / long
    )

    print()
    print("  Permutation invariance: P0 vs P4 as candidate, P1 as competitor")
    print(f"  s_masked_A : {X_A[0, :15].round(4)}")
    print(f"  s_masked_B : {X_B[0, :15].round(4)}")
    print(f"  a4_A       : {X_A[0, 15:].round(4)}")
    print(f"  a4_B       : {X_B[0, 15:].round(4)}")

    inputs_match = bool(np.allclose(X_A, X_B))
    print(f"  Input vectors identical : {inputs_match}")
    assert inputs_match, "FAIL: input vectors should be identical after masking"

    Q_A = float(agent_ac.forward_batch(X_A)[0, 0])
    Q_B = float(agent_ac.forward_batch(X_B)[0, 0])
    q_match = abs(Q_A - Q_B) < 1e-10
    print(f"  Q(P0/long, state_A) = {Q_A:.10f}")
    print(f"  Q(P4/long, state_B) = {Q_B:.10f}")
    print(f"  Q-values identical  : {q_match}")
    assert q_match, f"FAIL: Q-values should be identical, got |delta|={abs(Q_A-Q_B)}"
    print("  Permutation invariance test : PASS")

    # --- 5d: negative test — competitor at different PID → NOT invariant ---
    state_C = np.zeros(15)
    state_C[0 * 3 : 0 * 3 + 3] = [0.4, 1.0, 0.1]   # P0 = candidate
    state_C[2 * 3 : 2 * 3 + 3] = [0.3, 1.0, 0.05]  # P2 = competitor (≠ P1)

    X_C = agent_ac._build_input_batch(
        state_C[None], np.array([0 * 3 + 2])
    )
    # s_masked_C has P2 at slot [6:9], s_masked_A has P1 at slot [3:6]
    # → different → NOT invariant (expected)
    not_same = not np.allclose(X_A, X_C)
    print()
    print(f"  Negative test (competitor P1 vs P2) → inputs differ : {not_same}")
    assert not_same, "FAIL: should differ when competitor is at different PID"
    print("  Non-invariance for different competitor positions : confirmed (expected)")

    # --- 5e: parameter count -------------------------------------------
    total = sum(w.size for w in agent_ac._W) + sum(b.size for b in agent_ac._b)
    print()
    print(f"  Parameter count : {total}  (target ≈3,393)")
    assert total == 3393, f"FAIL: expected 3393, got {total}"

    print()
    print("ActionConditionedDQN: ALL ASSERTIONS PASSED")

    # ======================================================================
    # SECTION 6: SumPoolingDQN — unit tests (Week 6)
    # ======================================================================
    print()
    print("=" * 64)
    print("SumPoolingDQN — unit tests (Week 6)")
    print("=" * 64)

    np.random.seed(77)
    agent_sp = SumPoolingDQN()

    # --- 6a: parameter count -------------------------------------------
    total_sp = sum(w.size for w in agent_sp._W) + sum(b.size for b in agent_sp._b)
    print(f"  Parameter count : {total_sp}  (target 2,625)")
    # 7*64+64 + 64*32+32 + 32*1+1 = 512 + 2080 + 33 = 2625
    assert total_sp == 2625, f"FAIL: expected 2625, got {total_sp}"
    print("  Parameter count : OK")

    # --- 6b: forward pass shape -----------------------------------------
    X_sp = np.random.rand(4, 7)
    q_sp = agent_sp.forward_batch(X_sp)
    assert q_sp.shape == (4, 1), f"FAIL: expected (4,1), got {q_sp.shape}"
    print(f"  Forward pass shape : {q_sp.shape}  (expected (4,1)) — OK")

    # --- 6c: _build_input_batch sanity ----------------------------------
    # State: P0 arrived+runnable (0.05, 1.0, 0.0), P1 arrived+runnable (0.2, 1.0, 0.01)
    # Action: P0/qt=0.  competitor_pool should equal P1's features only.
    s_sanity = np.zeros(15)
    s_sanity[0*3 : 0*3+3] = [0.05, 1.0, 0.0]
    s_sanity[1*3 : 1*3+3] = [0.20, 1.0, 0.01]
    X_sanity = agent_sp._build_input_batch(s_sanity[None], np.array([0]))
    assert abs(X_sanity[0, 0] - 0.20) < 1e-10, "FAIL: pool[0] should be P1 remaining"
    assert abs(X_sanity[0, 1] - 1.0)  < 1e-10, "FAIL: pool[1] should be 1.0"
    assert abs(X_sanity[0, 2] - 0.01) < 1e-10, "FAIL: pool[2] should be P1 wait"
    assert abs(X_sanity[0, 3] - 0.05) < 1e-10, "FAIL: a4[0] should be P0 remaining"
    assert abs(X_sanity[0, 6] - 0.0)  < 1e-10, "FAIL: a4[3] qt=0 → 0/2=0.0"
    print("  _build_input_batch sanity             : OK")

    # ----------------------------------------------------------------
    # UNIT TEST 1 — Candidate invariance (same as Week 5 test)
    # P0 as candidate (burst=3ms), P1 competitor (burst=10ms)  →  action P0/qt=2
    # P4 as candidate (burst=3ms), P1 competitor (burst=10ms)  →  action P4/qt=2
    # Competitor pool must be equal (P1 in both cases): Q-values must be equal.
    # ----------------------------------------------------------------
    print()
    print("  Unit Test 1 — Candidate invariance (P0 vs P4, same competitor P1)")

    st1_A = np.zeros(15)
    st1_A[0*3 : 0*3+3] = [3.0/_BP, 1.0, 0.0]   # P0 = candidate, 3s
    st1_A[1*3 : 1*3+3] = [10.0/_BP, 1.0, 0.0]  # P1 = competitor, 10s

    st1_B = np.zeros(15)
    st1_B[4*3 : 4*3+3] = [3.0/_BP, 1.0, 0.0]   # P4 = candidate, 3s
    st1_B[1*3 : 1*3+3] = [10.0/_BP, 1.0, 0.0]  # P1 = competitor, 10s

    X1_A = agent_sp._build_input_batch(st1_A[None], np.array([0*3+2]))  # P0/long
    X1_B = agent_sp._build_input_batch(st1_B[None], np.array([4*3+2]))  # P4/long

    pool1_A = X1_A[0, :3]; pool1_B = X1_B[0, :3]
    a4_1_A  = X1_A[0, 3:]; a4_1_B  = X1_B[0, 3:]
    print(f"    competitor_pool A : {pool1_A.round(6)}")
    print(f"    competitor_pool B : {pool1_B.round(6)}")
    print(f"    a4 A              : {a4_1_A.round(6)}")
    print(f"    a4 B              : {a4_1_B.round(6)}")

    inputs1_match = bool(np.allclose(X1_A, X1_B))
    print(f"    Input vectors identical : {inputs1_match}")
    assert inputs1_match, "FAIL: UT1 inputs should be identical"

    Q1_A = float(agent_sp.forward_batch(X1_A)[0, 0])
    Q1_B = float(agent_sp.forward_batch(X1_B)[0, 0])
    q1_match = abs(Q1_A - Q1_B) < 1e-10
    print(f"    Q(P0/long) = {Q1_A:.10f}")
    print(f"    Q(P4/long) = {Q1_B:.10f}")
    print(f"    Q-values identical : {q1_match}")
    assert q1_match, f"FAIL: UT1 Q-values differ by {abs(Q1_A-Q1_B)}"
    print("  Unit Test 1 : PASS")

    # ----------------------------------------------------------------
    # UNIT TEST 2 — Full permutation invariance (new for Week 6)
    # State A: P0 candidate (3ms, w=0), P1 comp (10ms, w=5ms),
    #          P2 comp (15ms, w=3ms)
    # State B: P4 candidate (3ms, w=0), P2 comp (10ms, w=5ms),
    #          P1 comp (15ms, w=3ms)
    # Competitors are permuted across PIDs — competitor_pool must be equal.
    # ----------------------------------------------------------------
    print()
    print("  Unit Test 2 — Full permutation invariance")
    print("    (candidate AND competitor PIDs permuted simultaneously)")

    st2_A = np.zeros(15)
    st2_A[0*3 : 0*3+3] = [3.0/_BP,  1.0, 0.0/_WN]   # P0 candidate 3s
    st2_A[1*3 : 1*3+3] = [10.0/_BP, 1.0, 5.0/_WN]   # P1 comp 10s wait=5s
    st2_A[2*3 : 2*3+3] = [15.0/_BP, 1.0, 3.0/_WN]   # P2 comp 15s wait=3s

    st2_B = np.zeros(15)
    st2_B[4*3 : 4*3+3] = [3.0/_BP,  1.0, 0.0/_WN]   # P4 candidate 3s
    st2_B[2*3 : 2*3+3] = [10.0/_BP, 1.0, 5.0/_WN]   # P2 comp 10s wait=5s
    st2_B[1*3 : 1*3+3] = [15.0/_BP, 1.0, 3.0/_WN]   # P1 comp 15s wait=3s

    X2_A = agent_sp._build_input_batch(st2_A[None], np.array([0*3+0]))  # P0/1ms
    X2_B = agent_sp._build_input_batch(st2_B[None], np.array([4*3+0]))  # P4/1ms

    pool2_A = X2_A[0, :3]; pool2_B = X2_B[0, :3]
    a4_2_A  = X2_A[0, 3:]; a4_2_B  = X2_B[0, 3:]
    print(f"    competitor_pool A : {pool2_A.round(6)}")
    print(f"    competitor_pool B : {pool2_B.round(6)}")
    print(f"    a4 A              : {a4_2_A.round(6)}")
    print(f"    a4 B              : {a4_2_B.round(6)}")

    inputs2_match = bool(np.allclose(X2_A, X2_B))
    print(f"    Input vectors identical : {inputs2_match}")
    assert inputs2_match, "FAIL: UT2 inputs should be identical after sum-pooling"

    Q2_A = float(agent_sp.forward_batch(X2_A)[0, 0])
    Q2_B = float(agent_sp.forward_batch(X2_B)[0, 0])
    q2_match = abs(Q2_A - Q2_B) < 1e-10
    print(f"    Q(P0/1ms, state_A) = {Q2_A:.10f}")
    print(f"    Q(P4/1ms, state_B) = {Q2_B:.10f}")
    print(f"    Q-values identical : {q2_match}")
    assert q2_match, f"FAIL: UT2 Q-values differ by {abs(Q2_A-Q2_B)}"
    print("  Unit Test 2 : PASS")

    # ----------------------------------------------------------------
    # UNIT TEST 3 — Negative test: modify State B (P2 burst = 11ms)
    # competitor_pool changes → Q must differ from State A
    # ----------------------------------------------------------------
    print()
    print("  Unit Test 3 — Negative test: P2 burst 10ms → 11ms in State B")

    st2_B_mod = st2_B.copy()
    st2_B_mod[2*3] = 11.0 / _BP   # P2 competitor now 11s (was 10s)

    X2_B_mod = agent_sp._build_input_batch(st2_B_mod[None], np.array([4*3+0]))
    pool2_B_mod = X2_B_mod[0, :3]
    print(f"    competitor_pool A     : {pool2_A.round(6)}")
    print(f"    competitor_pool B_mod : {pool2_B_mod.round(6)}")

    inputs_differ = not np.allclose(X2_A, X2_B_mod)
    print(f"    Input vectors differ  : {inputs_differ}  (expected True)")
    assert inputs_differ, "FAIL: UT3 inputs should differ when burst changes"

    Q2_B_mod = float(agent_sp.forward_batch(X2_B_mod)[0, 0])
    q_differs = abs(Q2_A - Q2_B_mod) > 1e-10
    print(f"    Q(P0/1ms, state_A)     = {Q2_A:.10f}")
    print(f"    Q(P4/1ms, state_B_mod) = {Q2_B_mod:.10f}")
    print(f"    Q-values differ        : {q_differs}  (expected True)")
    assert q_differs, "FAIL: UT3 Q-values should differ"
    print("  Unit Test 3 : PASS")

    print()
    print("SumPoolingDQN: ALL ASSERTIONS PASSED")

    # ======================================================================
    # SECTION 7: AttentionDQN — gradient check + unit tests (Week 7)
    # ======================================================================
    print()
    print("=" * 64)
    print("AttentionDQN — gradient check + unit tests (Week 7)")
    print("=" * 64)

    np.random.seed(42)
    agent_at = AttentionDQN()

    # --- 7a: parameter count --------------------------------------------
    attn_params_count = (
        agent_at.W_Q.size + agent_at.b_Q.size +
        agent_at.W_K.size + agent_at.b_K.size +
        agent_at.W_V.size + agent_at.b_V.size
    )
    mlp_params_count = (
        sum(w.size for w in agent_at._W) +
        sum(b.size for b in agent_at._b)
    )
    total_at = attn_params_count + mlp_params_count
    print(f"  Attention params : {attn_params_count}  (target 96)")
    print(f"  MLP params       : {mlp_params_count}   (target 2,945)")
    print(f"  Total params     : {total_at}  (target 3,041)")
    assert attn_params_count == 96,   f"FAIL: attn params {attn_params_count} != 96"
    assert mlp_params_count  == 2945, f"FAIL: mlp params {mlp_params_count} != 2945"
    assert total_at          == 3041, f"FAIL: total {total_at} != 3041"
    print("  Parameter count  : OK")

    # --- 7b: forward pass shape -----------------------------------------
    np.random.seed(42)
    s_rand  = np.random.rand(8, 15)
    a_rand  = np.random.randint(0, 15, size=8)
    q_at    = agent_at.forward_batch(s_rand, a_rand)
    assert q_at.shape == (8, 1), f"FAIL: expected (8,1), got {q_at.shape}"
    print(f"  Forward pass shape : {q_at.shape}  (expected (8,1)) — OK")

    # -------------------------------------------------------------------
    # NON-NEGOTIABLE GATE: Finite-difference gradient check
    # -------------------------------------------------------------------
    print()
    print("  --- Gradient check (HARD GATE: max_rel_err < 1e-4) ---")
    np.random.seed(13)

    # Build a small realistic batch (batch=4) with some valid competitors
    gc_batch = 4
    gc_states = np.zeros((gc_batch, 15))
    # P0 candidate (arrived, runnable), P1 & P2 competitors (arrived, runnable)
    gc_states[:, 0*3 : 0*3+3] = [0.1, 1.0, 0.05]   # P0
    gc_states[:, 1*3 : 1*3+3] = [0.2, 1.0, 0.03]   # P1
    gc_states[:, 2*3 : 2*3+3] = [0.3, 1.0, 0.02]   # P2
    # Add small per-sample noise so samples are not identical
    gc_states += np.random.rand(gc_batch, 15) * 0.01

    gc_actions  = np.array([0, 1, 2, 0], dtype=np.int32)  # P0/1ms, P0/5ms, P0/20ms, P0/1ms
    gc_targets  = np.random.randn(gc_batch) * 2.0

    max_err, worst, passed = agent_at.gradient_check(gc_states, gc_actions, gc_targets)

    print(f"  Max relative error : {max_err:.2e}")
    print(f"  Worst parameter    : {worst}")
    print(f"  PASS (< 1e-4)      : {passed}")

    if not passed:
        print()
        print("  *** GRADIENT CHECK FAILED — STOPPING ***")
        print(f"  max_rel_err = {max_err:.2e}  (threshold 1e-4)")
        print(f"  Worst param : {worst}")
        raise AssertionError(
            f"AttentionDQN gradient check FAILED: max_rel_err={max_err:.2e}, "
            f"worst param={worst}"
        )
    print("  Gradient check : PASS")

    # ----------------------------------------------------------------
    # UNIT TEST 1 — Candidate invariance
    # P0 candidate (3ms), P1 competitor (10ms)  →  action P0/long
    # P4 candidate (3ms), P1 competitor (10ms)  →  action P4/long
    # Same candidate features, same competitor set → same Q.
    # ----------------------------------------------------------------
    print()
    print("  Unit Test 1 — Candidate invariance (P0 vs P4, same competitor P1)")

    np.random.seed(7)
    agent_at2 = AttentionDQN()

    ut1_A = np.zeros(15)
    ut1_A[0*3 : 0*3+3] = [3.0/_BP, 1.0, 0.0]    # P0 = candidate, 3s
    ut1_A[1*3 : 1*3+3] = [10.0/_BP, 1.0, 0.0]   # P1 = competitor, 10s

    ut1_B = np.zeros(15)
    ut1_B[4*3 : 4*3+3] = [3.0/_BP, 1.0, 0.0]    # P4 = candidate, 3s
    ut1_B[1*3 : 1*3+3] = [10.0/_BP, 1.0, 0.0]   # P1 = competitor, 10s

    Q_ut1_A = float(agent_at2.forward_batch(
        ut1_A[None], np.array([0*3+2])           # P0/long
    )[0, 0])
    Q_ut1_B = float(agent_at2.forward_batch(
        ut1_B[None], np.array([4*3+2])           # P4/long
    )[0, 0])

    print(f"    Q(P0/long, state_A) = {Q_ut1_A:.10f}")
    print(f"    Q(P4/long, state_B) = {Q_ut1_B:.10f}")
    ut1_match = abs(Q_ut1_A - Q_ut1_B) < 1e-10
    print(f"    Q-values identical  : {ut1_match}")
    assert ut1_match, f"FAIL: UT1 |delta|={abs(Q_ut1_A - Q_ut1_B):.2e}"
    print("  Unit Test 1 : PASS")

    # ----------------------------------------------------------------
    # UNIT TEST 2 — Full permutation invariance
    # State A: P0 candidate (3ms), P1 comp (10ms,w=5ms), P2 comp (15ms,w=3ms)
    # State B: P4 candidate (3ms), P2 comp (10ms,w=5ms), P1 comp (15ms,w=3ms)
    # Competitors permuted across PIDs — same multiset → same context → same Q.
    # ----------------------------------------------------------------
    print()
    print("  Unit Test 2 — Full permutation invariance")
    print("    (candidate AND competitor PIDs permuted simultaneously)")

    ut2_A = np.zeros(15)
    ut2_A[0*3 : 0*3+3] = [3.0/_BP,  1.0, 0.0/_WN]  # P0 candidate 3s
    ut2_A[1*3 : 1*3+3] = [10.0/_BP, 1.0, 5.0/_WN]  # P1 comp 10s wait=5s
    ut2_A[2*3 : 2*3+3] = [15.0/_BP, 1.0, 3.0/_WN]  # P2 comp 15s wait=3s

    ut2_B = np.zeros(15)
    ut2_B[4*3 : 4*3+3] = [3.0/_BP,  1.0, 0.0/_WN]  # P4 candidate 3s
    ut2_B[2*3 : 2*3+3] = [10.0/_BP, 1.0, 5.0/_WN]  # P2 comp 10s wait=5s  ← permuted
    ut2_B[1*3 : 1*3+3] = [15.0/_BP, 1.0, 3.0/_WN]  # P1 comp 15s wait=3s  ← permuted

    Q_ut2_A = float(agent_at2.forward_batch(
        ut2_A[None], np.array([0*3+0])           # P0/1ms
    )[0, 0])
    Q_ut2_B = float(agent_at2.forward_batch(
        ut2_B[None], np.array([4*3+0])           # P4/1ms
    )[0, 0])

    print(f"    Q(P0/1ms, state_A) = {Q_ut2_A:.10f}")
    print(f"    Q(P4/1ms, state_B) = {Q_ut2_B:.10f}")
    ut2_match = abs(Q_ut2_A - Q_ut2_B) < 1e-10
    print(f"    Q-values identical : {ut2_match}")
    assert ut2_match, f"FAIL: UT2 |delta|={abs(Q_ut2_A - Q_ut2_B):.2e}"
    print("  Unit Test 2 : PASS")

    # ----------------------------------------------------------------
    # UNIT TEST 3 — Negative test
    # Modify state_B: P2 burst 10ms → 11ms.
    # Competitor multiset changes → context changes → Q must differ.
    # ----------------------------------------------------------------
    print()
    print("  Unit Test 3 — Negative test: P2 burst 10ms → 11ms in State B")

    ut2_B_mod = ut2_B.copy()
    ut2_B_mod[2*3] = 11.0 / _BP   # P2 competitor now 11s

    Q_ut2_B_mod = float(agent_at2.forward_batch(
        ut2_B_mod[None], np.array([4*3+0])       # P4/1ms
    )[0, 0])

    q_differs = abs(Q_ut2_A - Q_ut2_B_mod) > 1e-10
    print(f"    Q(P0/1ms, state_A)     = {Q_ut2_A:.10f}")
    print(f"    Q(P4/1ms, state_B_mod) = {Q_ut2_B_mod:.10f}")
    print(f"    Q-values differ        : {q_differs}  (expected True)")
    assert q_differs, "FAIL: UT3 Q-values should differ after burst change"
    print("  Unit Test 3 : PASS")

    print()
    print("AttentionDQN: GRADIENT CHECK PASSED + ALL UNIT TESTS PASSED")
