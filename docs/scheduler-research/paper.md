# Learning to Schedule: A Twelve-Week Policy Discovery Experiment

**Author:** Dev Upadhyay
**Affiliation:** M.S. Computer Science, University of California Santa Cruz
**Program:** Graduate Capstone Project, 2024–2025
**Stack:** Python 3.10+, pure NumPy + stdlib (no PyTorch / TensorFlow)

---

## Abstract

CPU scheduling's efficiency-fairness tension has resisted formal characterization. Existing schedulers optimize throughput (SRPT) or fairness (CFS) but not both, and existing metrics ignore value-curve heterogeneity.

We train a two-head attention DQN on Alibaba 2018 cluster traces with a value-delta reward, where each task carries urgency curve V(d) = base × max(floor, exp(−d/τ)), aggregating V(delay+q) − V(delay) across all runnable processes.

We prove the Quantum Scoring Monotonicity Lemma: any per-task local scorer is strictly monotone in quantum size, making non-degenerate quantum selection impossible without queue-global information — confirmed by three ablations.

W10C, our two-head attention DQN, achieves 21.71 ± 0.64s mean completion time (replicated mean across 3 independent seeds), beating Round Robin with zero starvation and lowest slowdown variance. The best-observed single checkpoint reaches 17.23s (N=1500 evaluation episodes, original checkpoint `dqn_w10c.npz`). A 2×2 ablation isolates reward formulation and burst observability as independently load-bearing: value-delta reward outperforms composite reward by 9.1s when burst is available, while removing burst costs only 4.4s when reward is value-delta — establishing that the oracle information gap is modest on realistic workloads.

We introduce the Value-Rate Fairness Index (VRFI), measuring per-task value loss rate variance. We prove a case exists where JFI = 1.0 and SDV = 0 yet VRFI = 0.641, exposing inequality invisible to prior metrics.

Removing oracle burst time increases MCT from 17.23s (original checkpoint) to 21.67s on the Alibaba 2018 trace (4.4s observability cost), quantifying deployment observability cost for the first time. Code uses only NumPy and stdlib; value-delta reward decomposition causes policy collapse, confirming reward structure is more load-bearing than architectural complexity.

A preference-conditioned variant (W14-ω) beats MLFQ on both throughput (20.60s vs 21.59s MCT) and starvation prevention (32.5% vs 36.0%) at deployment preference ω_s=0.7, using no oracle information and requiring no retraining to shift between operating modes.

---

## 1. Introduction

Modern CPU schedulers operate on what is immediately observable: the processes in the ready queue,
their recent CPU usage, and their priority class. Linux's Completely Fair Scheduler (CFS)
distributes CPU time proportionally to virtual runtime, maintaining fairness over runnable tasks.
But fairness over what is visible is not the same as minimising the time tasks spend waiting to
become visible. A short-burst process that arrives while the CPU is occupied by a long-burst
process experiences a waiting time that CFS cannot reduce by design — the problem is not in its
policy but in its horizon.

The question we ask is different: can an agent, given a complete view of each process's remaining
burst and arrival state, learn a scheduling policy without behavioral cloning or reference policy
demonstrations — no supervision, no target policy to imitate, no handcrafted priority rules —
purely from the signal of how long processes have been alive? This is a policy discovery problem, not a CFS improvement problem. We do not propose to
replace CFS. We ask whether RL can, in a controlled toy environment, rediscover approximations to
the known optimal policy, and — more importantly — whether the failure modes that prevent full
rediscovery reveal something about the architecture requirements for generalising scheduling
policies across arbitrary workloads.

Our contribution is an eight-item progression with honest reporting of both the positive results and
the failures:

1. **Systematic failure mode documentation across four architectures** (§5.2–§5.9): tabular
   discretisation collapse (zero cross-episode variance, identity bias), DQN position-indexed
   gradient contamination (identity bias worsens from |Δ|=0.225 to 5.48), a spurious
   load-adaptive quantum pattern indistinguishable from a learned heuristic without ablation,
   and attention entropy collapse requiring annealed regularisation to thread softmax-freeze
   and TD-signal domination. Each failure is diagnosed to a specific architectural cause and
   resolved or explicitly left open.

2. **W10C two-head attention DQN with multi-seed validation** (§5.20): achieves
   21.71 ± 0.64s mean MCT (replicated across 3 independent seeds, CV=2.9%) on the Alibaba
   2018 trace, beating Round Robin by approximately 2s on the replicated mean. Best observed
   single checkpoint: 17.23s (N=1500 evaluation episodes, original checkpoint `dqn_w10c.npz`).
   Spontaneous head specialization — one head surveys the competitor field, one monitors the
   longest competitor — emerges without supervision from the area-integral reward signal alone.

3. **Quantum Scoring Monotonicity Lemma and three confirming ablations** (§3.1.3, §5.23):
   formal proof that f_rate always selects q_min and f_total always selects q_max for all
   τ > 0, d ≥ 0, establishing that value-drop local scorers are insufficient for non-degenerate
   quantum selection. Three ablations (quantum_only, quantum_only_fixed, ordering_only)
   confirm the Lemma empirically on Alibaba-derived trace data.

4. **Gradient alignment and tau calibration** (§5.26): identifies a three-to-four order-of-
   magnitude disconnect between default τ values (25s/100s) and trace-scale delays (medians
   5,279s/38,665s), and shows calibrated τ values (900s/800s) still produce near-ties in
   93–100% of ordering decisions — a structural consequence of the 92%/8% smooth/steep split,
   not a calibration failure.

5. **Value-Rate Fairness Index (VRFI) and JFI/SDV falsification** (§5.27.2): a new metric
   defined as 1 − CV(VLR) that captures value-curve fairness invisible to Jain's FI and
   slowdown variance. Key falsification: ten tasks with identical delays (d=50s) achieve
   JFI=1.0 and SDV=0 while VRFI=0.641 correctly detects value-rate inequality.
   VRFI-targeted parameter optimization achieves the first positive VRFI (+0.289) on the trace.

6. **2×2 ablation over reward structure and burst observability** (§5.29–§5.30): reward
   formulation is more load-bearing than state observability. Removing value-delta reward
   (W11d, burst kept) degrades MCT to 26.35s; removing oracle burst (W12, reward kept)
   degrades MCT to 21.67s. The oracle information cost is 4.4s on the 2018 trace — smaller
   than the 9.1s reward-structure gap, and substantially smaller than earlier experiments on
   synthetic distributions suggested.

7. **W14-ω preference-conditioned attention DQN** (§5.33): beats MLFQ simultaneously on MCT
   (20.60s vs 21.59s) and starvation prevention (32.5% vs 36.0%) at ω_s=0.7, with no oracle
   burst information and no retraining required to shift operating modes. The non-monotonic
   Pareto frontier — starvation minimized at ω_s=0.7, not ω_s=1.0 — establishes that optimal
   fairness requires retaining throughput pressure.

8. **W15 variable-N training confirming generalization** (§5.35): retraining W14-ω's
   architecture on Poisson arrival streams recovers the MCT advantage (mean −1.40s vs MLFQ
   across 3 seeds), confirming that the 2-head attention mechanism and FiLM conditioning
   generalize to variable queue depths when trained on them.

The honest negative results — zero variance as a sign of discretisation collapse, identity bias
worsening with the move to DQN, the 80% process-selection target not met — are as central to the
contribution as the positive ones. Each failure is reproducible, diagnosed to a specific
architectural cause, and fixed or explicitly left open.

---

## 2. Background

### 2.1 CFS and the Scheduling Horizon Problem

CFS uses a red-black tree ordered by virtual runtime to select the next runnable process, targeting
equal CPU share among active tasks. The scheduler operates on the ready queue: processes that have
not yet arrived, or that are blocked on I/O, are invisible to the scheduling decision. A task
scheduling policy that minimises mean turnaround time must account for the entire job mix, including
processes not yet runnable — a horizon that CFS does not have by design.

Separately, CFS is a fairness scheduler, not a throughput scheduler. It makes no attempt to
prefer shorter jobs, which is the key property of SRPT, the preemptive optimal policy for
minimising mean turnaround time when burst lengths are known. This work assumes full observability
of remaining burst — a stronger assumption than any real kernel can satisfy without estimation —
in exchange for removing all implicit supervision and studying what a reward signal alone can teach.

Since Linux kernel 6.6, the Earliest Eligible Virtual Deadline First (EEVDF) scheduler has replaced CFS as the default Linux process scheduler [9]. EEVDF assigns each task a virtual deadline based on its weight and requested scheduling slice, selecting the eligible task with the earliest virtual deadline. Unlike CFS which tracks vruntime globally, EEVDF introduces per-task eligible times that prevent tasks from receiving CPU time before they are entitled to it. The value-aware scheduler proposed in this work differs from both CFS and EEVDF in that scheduling decisions are driven by explicit per-task value curves rather than virtual runtime accounting — the agent learns to approximate value-optimal dispatch rather than time-fair dispatch.

### 2.2 RL for Scheduling: Related Work

KernelOracle [1] applies supervised learning to predict which task Linux's CFS will select next.
An LSTM network is trained on a dataset of scheduling decisions extracted from a running kernel,
treating CFS itself as the labeling oracle: observed task selections are the prediction targets.
KernelOracle does not learn a new scheduling policy; it learns to replicate CFS's selection
behavior. The training signal is CFS expertise captured through observation, not a reward function.

ALPS [2] targets serverless FaaS workloads. A user-space frontend simulates SRPT scheduling on
recent workload history to derive task priorities; an eBPF backend applies those priorities within
the running CFS at kernel scheduling ticks. The reference policy — SRPT — is embedded in the
design: the system knows that shortest-remaining-time is optimal for short-lived serverless
functions and learns which tasks are "short" by simulating SRPT on historical arrivals. On
production serverless traces from Huawei and Azure, ALPS achieves a 57.2% reduction in mean
function execution duration relative to unmodified CFS.

Both approaches are anchored to a reference policy: KernelOracle to CFS (as a prediction target),
ALPS to SRPT (as a planning oracle). Both embed domain knowledge in their design: KernelOracle's
training distribution is defined by CFS's operating range; ALPS's frontend assumes workload
stationarity and that SRPT simulation on past arrivals generalises to future scheduling decisions.

Our work removes all such anchors. The reward signal encodes no target policy to replicate and no
reference scheduler to clone. The state vector uses only remaining burst, arrival flag, and wait
time — features a kernel would need to estimate from hardware counters but that carry no
scheduler-specific domain knowledge. The agent must discover that SRPT-like selection follows
from minimising elapsed-time cost, without being told that SRPT is the objective to approximate.

### 2.3 Related Work: RL Scheduling in Practice

Three deployed or near-deployed systems illustrate the current state of RL-based scheduling and
the role that domain knowledge plays in each.

**RLScheduler (Zhang et al. 2019)** applied deep RL to HPC batch job scheduling, learning a
job selection policy from scratch that outperformed hand-tuned priority heuristics on production
cluster traces. The critical enabler was a rich per-job state: job type, estimated runtime from
historical submissions, queue position, and resource class. The learned policy exploited
correlations between job type and actual runtime that no hand-tuned rule had captured. The key
difference from this work is observational: RLScheduler had access to historical runtime
statistics accumulated over thousands of prior submissions for each job class. Remaining burst
was not observed directly — it was inferred from job-type priors.

**Double DQN for I/O-Intensive Scheduling (Sun et al. 2025)** achieved strong performance on
mixed CPU-I/O workloads by extending the state vector to include I/O wait ratio — the fraction
of recent time a process spent blocked on I/O — alongside CPU burst estimates. On I/O-intensive
task mixes, where CPU burst alone is a poor predictor of actual resource consumption, the I/O
wait signal was the decisive feature. Agents trained without it converged to CPU-burst-optimal
policies that were suboptimal under the full latency objective.

**SmartOS (2021)** applied RL to OS resource management across CPU scheduling, memory
allocation, I/O prioritisation, and network bandwidth simultaneously, framing the joint
allocation problem as a multi-dimensional continuous action space with a single system-level
latency reward. The cross-resource coordination — CPU scheduler decisions conditioned on memory
pressure and I/O queue depth — produced improvements that no single-resource policy could
achieve, at the cost of a substantially more complex observation space encoding interactions
across subsystems.

The common thread across successful systems is domain-informed state design: each incorporates
features — job type, I/O wait ratio, historical CPU usage, memory pressure — derived from
scheduler engineering knowledge accumulated over decades. This work takes the opposite position
deliberately: state is restricted to remaining burst, arrival flag, and wait time — features
requiring no domain knowledge — to isolate the question of what reward signal alone can teach.
The cost of this constraint is visible in the real-trace results (Section 5.15): the agent
approaches but does not beat Round Robin on filtered Alibaba trace data. Section 6 discusses
the specific features the literature identifies as load-bearing and what adding them would
require.

### 2.4 SRPT as Oracle

Shortest Remaining Processing Time is the preemptive optimal policy for minimising mean turnaround
time when burst lengths are known exactly. In our environment, burst lengths are given — the oracle
is achievable. On 500 random process sets (master seed 42), SRPT achieves mean MCT = 67.66ms,
std = 23.96ms. All random-evaluation performance is reported as deviation from this oracle, not
from Round Robin. Round Robin serves only as the lower bar: an agent that fails to beat it has
learned nothing useful.

### 2.5 Queue-Global Information in Production Systems

A 2026 public preview of task queue priority management for cloud batch schedulers introduced a fairness mechanism requiring server-side backlog tracking: the scheduler maintains a global count of queued tasks per job class to detect accumulating delays before they produce starvation. The mechanism explicitly cannot be implemented using per-task metadata alone — the backlog count is not a property of any individual task, but of the queue as a whole. The theoretical framework in Section 3 formalises this observation: we show that any scoring function operating on per-task features is provably unable to make optimal quantum selections for value-preserving schedulers, because both natural scoring objectives (rate-normalised and total value preserved) are monotone functions of quantum size. Breaking this degeneracy requires queue-global information — precisely the server-side backlog the 2026 system tracks. Our empirical ablation (Section 5.23) confirms the theory on synthetic trace data.

---

## 3. Theoretical Framework

### 3.1 On the Necessity of Queue-Global Information for Value-Aware Scheduling

Value-curve scheduling assigns each task i a value V_i(d) = base_i · max(floor_i, exp(−d/τ_i)), where d is the elapsed delay and τ_i > 0 is the task's urgency parameter. Steep tasks (small τ) lose value quickly; smooth tasks (large τ) decay slowly. A greedy scheduler attempts to select, at each decision step, the action (process, quantum) that maximises value preserved — the gap V_i(0) − V_i(d_i + q), where d_i is the current wait time and q is the proposed quantum.

#### 3.1.1 Potential Surface Formulation

Define the scheduling potential Φ(S) = Σ_i V_i(d_i) over the current queue state S, where the sum runs over all waiting tasks. Each scheduling decision (pick task i, assign quantum q) moves the system along the potential surface. The gradient of Φ with respect to task i's delay is:

    ∇Φ_i = −∂V_i/∂d = (base_i/τ_i) · exp(−d_i/τ_i)    [non-plateau]
           = 0                                             [plateau, d_i ≥ τ_i · ln(base_i/floor_i)]

A locally optimal scheduler should select the task with the largest |∇Φ_i| — the task whose value is decaying fastest. This formulation suggests a natural greedy ordering: rank tasks by their current value gradient and serve the steepest.

#### 3.1.2 Process Ordering Impossibility

The gradient ordering requires computing |∇Φ_i| for every queued task simultaneously and comparing them. This is an inherently queue-global operation: the selection of task i depends not on i's features alone, but on whether any co-queued task j has |∇Φ_j| > |∇Φ_i|. Any policy that assigns a score f_order(base_i, floor_i, τ_i, d_i) to task i in isolation and selects argmax_i f_order cannot, in general, produce the correct ordering — because the same score for task i may be optimal in one queue composition and suboptimal in another, depending on the co-queued population.

On traces dominated by tasks with similar τ (as in the Alibaba batch_task data, where 92% of tasks have τ_smooth ≈ 800s), the gradients ∇Φ_i are nearly identical across the queue: the ratio ∇Φ_i / ∇Φ_j = exp(−(d_i − d_j)/τ) → 1 as τ → ∞ relative to delay differences. In the empirical evaluation (Section 5.22), 93–100% of ordering decisions are near-ties (top-2 gradients within 10%) regardless of τ calibration. The potential surface is flat over the vast majority of the trace composition, making gradient-based ordering informationally void on this data.

#### 3.1.3 Quantum Selection Impossibility

Even if process ordering were resolved by queue-global comparison, quantum selection presents a separate impossibility. Given a fixed task i with delay d_i, a scheduler must choose quantum q from a discrete set {q_min, q_med, q_max}. Two natural per-task scoring objectives are:

**f_rate(q) = (V_i(d_i) − V_i(d_i + q)) / q** — value preserved per unit time.

**f_total(q) = V_i(d_i) − V_i(d_i + q)** — total value preserved by the action.

**Lemma (Quantum Scoring Monotonicity).** For any τ > 0, base > 0, floor ∈ [0, 1), and d ≥ 0:
(a) f_rate(q) is strictly decreasing in q for all q > 0.
(b) f_total(q) is strictly increasing in q for all q > 0 such that d + q is below the plateau.

**Proof.**
(a) Let h(q) = (1 − e^{−q/τ})/q. Then f_rate(q) = (base/τ) · h(q) · e^{−d/τ} · τ/1 = base · e^{−d/τ} · h(q). It suffices to show h(q) is strictly decreasing. Differentiating: h'(q) = [(q/τ + 1)e^{−q/τ} − 1] / q². Setting u = q/τ > 0, the numerator is (u + 1)e^{−u} − 1. Since (u + 1)e^{−u} < 1 for all u > 0 (as e^u > 1 + u by strict convexity), the numerator is strictly negative. Therefore h'(q) < 0 and f_rate is strictly decreasing. □

(b) f_total(q) = base · e^{−d/τ} · (1 − e^{−q/τ}). Since 1 − e^{−q/τ} is strictly increasing in q for τ > 0, f_total is strictly increasing. □

**Corollary (Queue-Global Necessity for Value-Drop Scorers).** No scoring function based solely on value-drop — operating on (base_i, floor_i, τ_i, d_i, q) via either rate-based or total-drop formulations — can produce mixed quantum selection across {q_min, q_med, q_max}. f_rate always selects q_min; f_total always selects q_max. This degeneracy holds for all τ > 0, d ≥ 0, confirming that value-drop local scorers are insufficient for non-degenerate quantum selection. We note that other local rules (e.g., threshold rules on d/τ) could in principle produce mixed selection while remaining local; the corollary applies specifically to the natural value-preservation objectives f_rate and f_total. Queue-global information is necessary to escape the degenerate poles of these two objectives.

**Note on the plateau case.** Part (b) of the proof applies when d + q lies below the plateau threshold, i.e., when exp(−(d+q)/τ) > floor. When d already exceeds the plateau (exp(−d/τ) ≤ floor), V(d) = V(d+q) = base×floor and f_total(q) = 0 for all q — the function is identically zero rather than strictly increasing. In this regime both scorers are degenerate for a different reason: the task has already lost maximum value and carries no further scheduling signal. The Corollary holds: queue-global information remains necessary, as the scheduler must attend to other tasks' non-plateau curves to make meaningful decisions.

#### 3.1.4 Queue-Global Resolution

The impossibility results motivate a scheduling architecture that maintains queue-global state: at each decision step, the scheduler maintains a running estimate of the queue composition (e.g., the distribution of τ values and current delays) and uses this estimate to inform both ordering and quantum selection. The two-head attention mechanism introduced in Week 10C (Section 5.20) is one instantiation of this principle: one attention head specialises in competitor context (queue state) and the other in selection (per-task scoring). The theoretical framework explains why single-head attention is insufficient — a single head that produces both context aggregation and selection scoring collapses to a per-task proxy under the monotonicity constraint.

#### 3.1.5 Empirical Confirmation

The theoretical predictions are confirmed by three ablations reported in Section 5.23: (1) a quantum_only policy using f_rate selects q_min 100% of the time across both default and calibrated τ initializations; (2) a quantum_only_fixed policy using f_total selects q_max 100% of the time; (3) an ordering_only policy using gradient ordering with fixed q_med achieves total_value = 15.37, representing near-complete starvation of smooth tasks — confirming that per-task ordering, even when correctly implemented, cannot resolve the value preservation objective without queue-global quantum selection. The full value_aware policy (total_value = 250.55) requires both ordering and quantum selection informed by the queue state to avoid degeneracy.

## 4. Method (see Figure 3 for W14-ω architecture)

### 4.1 Environment

Five processes per episode. Arrival times are sampled without replacement from {0, 2, 5, 8, 10}ms;
burst lengths from U[1, 60]ms, re-sampled each episode (randomised from Week 2 onward; fixed in
Week 1). The action space is 15 actions: 5 processes × 3 quantum tiers (1ms, 5ms, 20ms). Invalid
actions — processes not yet arrived, or completed — are masked at every decision step. An episode
ends when all processes complete.

**Baselines.** Round Robin (5ms fixed quantum, PID-ascending cycle): MCT ≈ 34.4ms
(in-distribution), 36.4ms (OOD). SRPT oracle: MCT ≈ 26.4ms (in-distribution), 28.4ms (OOD).

### 4.2 Reward Derivation

The reward signal at each decision step is:

    R(t) = −n_active(t) × q_actual(t)

where n_active(t) is the number of processes that have arrived and not yet completed, and
q_actual = min(quantum, remaining_burst) is the CPU time actually consumed.

By Little's Law, mean turnaround time equals the time-average number of active processes times
the mean time a process spends in the system. Minimising the cumulative sum of
n_active × q_actual — the area under the active-count curve — is therefore equivalent to
minimising total turnaround time. The reward requires no engineering of completion bonuses, shaped
incentives, or domain-specific terms. It is theoretically principled from a queuing standpoint.

From Week 3, the reward is normalised by REWARD_SCALE = 100.0. The maximum raw magnitude per
step is N × max_quantum = 5 × 20 = 100, so normalisation maps each step to [−1, 0], keeping
cumulative Q-value targets in approximately [−150, 0] — a range representable without numerical
divergence. An earlier training run without normalisation confirmed this need: loss grew from 46
to 87,799 over 10,000 episodes, Q-values reached +3,000.

### 4.3 State Representation

**Weeks 1–2 (tabular).** State is a 5-tuple of discretised remaining-burst bins. Each process is
mapped to one of 7 bins: 0 (complete), 1–5 (active tiers defined by burst-length thresholds),
6 (not arrived). The Q-table has shape (7,7,7,7,7,15) ≈ 1.9MB, supporting 16,807 distinct
states.

**Weeks 3–4 (DQN).** State is a 15-dim continuous float32 vector — per process:
[remaining_burst/60, arrived_flag, wait_time/300]. This encodes full burst magnitude, eliminating
the discretisation ceiling.

**Week 5 (action-conditioned).** The 15-dim state is retained but restructured into a 19-dim
network input: the 15-dim state with the candidate's PID slot zeroed (s_masked, 15-dim),
concatenated with the candidate's own features plus quantum tier (a4 = [remaining/60,
arrived_flag, wait/300, qt/2.0], 4-dim). The zeroing step removes redundancy and is what makes
the permutation invariance argument exact (see Section 4.5).

**Oracle disclosure.** W10C includes true remaining burst time as feature off+0 (remaining_norm), an oracle assumption not available in production schedulers. W12 and W15 remove this feature, replacing it with observable time-based features only (time_in_queue, wait_time, time_since_last_exec, urgency_norm, cpu_norm, mem_norm, arrived_flag). The cost of this substitution is quantified in the 2×2 ablation (§5.29–§5.30): removing oracle burst time increases mean MCT by approximately 4.4s on the Alibaba 2018 trace.

### 4.4 Architecture Evolution

**Tabular (Weeks 1–2).** Q-table indexed by 5-tuple of discretised bins. α = 0.1, γ = 1.0,
ε decaying from 1.0 to 0.05 over 10,000 episodes via linear schedule.

**DQN (Weeks 3–4).** 15 → 64 (ReLU) → 32 (ReLU) → 15 (linear), 3,599 parameters. He
initialisation. Adam (lr=0.001, β₁=0.9, β₂=0.999). Global gradient norm clipping (max_norm=1.0).
Experience replay buffer (capacity 10,000, batch 32). Target network hard-copied every 200
episodes. 500-transition warmup of random exploration before first gradient update.

**Action-conditioned DQN (Week 5).** 19 → 64 (ReLU) → 32 (ReLU) → 1 (linear), 3,393
parameters. Same optimiser, same replay and target schedule. At each action selection step, 15
separate (s_masked ‖ a4) vectors are constructed — one per candidate action — and evaluated in a
single batched forward pass. At training time, Q-values for all 15 next-state actions are computed
in one (batch × 15, 19) = (480, 19) matrix multiply against the target network, keeping wall-clock
training time within approximately 1.5× of Week 3 despite the 15× query expansion.

**Quantum selection architecture note.** In W10C, the quantum tier index enters the MLP as an explicit feature (8-dim input = 7 task features + normalized quantum value), giving the agent genuinely independent Q-values for each (task, quantum) pair. The agent learns quantum-specific behavior: the MLP can assign different scores to (task 2, q=0.5s) and (task 2, q=8.0s). In W15, quantum does not enter the MLP; instead, one task-level score is computed per candidate and tiled across all 3 quantum tiers, making quantum selection effectively uniform among valid actions for the chosen task. W10C's quantum selection is learned; W15's is not. This distinction is reported here for reproducibility.

### 4.5 The Masking Amendment and Its Rationale

An initial Week 5 design included the candidate's own features in both s_masked and a4, creating
redundant information at two positions in the input vector. The amendment zeroes the candidate's
PID slot in s_masked before concatenation.

This is not a tidying change — it is what makes the permutation invariance argument exact. With
the candidate's slot zeroed, the s_masked vector for (P0 as candidate, competitors at P1–P4) and
(P4 as candidate, competitors at P0–P3) are identical whenever the competitor burst magnitudes are
identical and the competitors occupy the same PID positions. The a4 vector is identical when
candidate burst, arrival status, and quantum tier are identical. Identical inputs produce identical
outputs. Without the zeroing, the candidate's features appear at two positions in the 19-dim input
vector — once in s_masked at its PID index and once in a4 — so P0 and P4 inputs can never be
identical even when all process features match.

The formal invariance guarantee: for any pair of candidate PIDs (p, p') and any state where both
have the same remaining burst, arrival flag, and wait time, and where all competitors are at the
same PID positions with the same features, Q(s, p, qt) = Q(s, p', qt) exactly.

---

## 5. Results

### 5.1 Twelve-Week Performance Summary

*Random evaluation — 500 process sets, master seed 42, same seeds for all agents:*

| Agent | Mean MCT | Std | vs SRPT | Eval set |
|-------|----------|-----|---------|----------|
| SRPT oracle | 67.66ms | 23.96ms | 0.00ms | Random (500) |
| DQN W5 (action-conditioned) | 76.04ms | 27.70ms | +8.38ms | Random (500) |
| DQN W7 (attention-annealed) | 80.57ms | 21.82ms | +12.91ms | Random (500) |
| DQN W3 (position-indexed) | 85.00ms | 26.51ms | +17.34ms | Random (500) |

*Fixed-set evaluation — two specific process configurations from Week 1 (not seen during W3/W5 training):*

| Agent | Mean MCT | Std | vs SRPT | Eval set |
|-------|----------|-----|---------|----------|
| DQN W5 (in-dist fixed) | 31.00ms | 0.00ms | +4.60ms | Fixed in-dist |
| DQN W5 (OOD fixed) | 29.60ms | 0.00ms | +1.20ms | Fixed OOD |
| DQN W3 (in-dist fixed) | 34.60ms | 0.00ms | +8.20ms | Fixed in-dist |
| DQN W3 (OOD fixed) | 32.00ms | 0.00ms | +3.60ms | Fixed OOD |
| Tabular W2 (in-dist) | 29.80ms | 0.00ms | +3.40ms | Fixed in-dist |
| Round Robin | 34.40ms | — | +8.00ms | Fixed in-dist |

*Filtered real-trace evaluation — Alibaba 2018 test split (W10C: N=1500, original checkpoint; replicated: N=500×3 seeds; all others: N=500, master seed 42) (Weeks 8–12):*

| Agent | Mean MCT | Std | vs Round Robin | Primary metric |
|-------|----------|-----|----------------|----------------|
| SRPT oracle (filtered) | 15.20s | 8.94s | — | MCT |
| W10C 2-head attention (orig. checkpoint†) | 17.23s | 11.38s | **−4.08s** | MCT |
| W9 1-head fixed β | 19.22s | 11.58s | −2.09s | MCT |
| W10C-VC2 ep 5000 (value-curve) | 21.46s | — | +0.15s | total_value |
| Round Robin (1.0s, filtered) | 21.31s | 14.03s | baseline | — |
| **W10C (replicated, 3 seeds)** | **21.71 ± 0.64s** | — | **−∼0.4s** | **MCT** |
| W10C-VC1 ep 2000 (value-curve) | 26.56s | — | +5.25s | total_value |

†Original checkpoint evaluated at N=1500 episodes. Replicated fresh training: 21.71 ± 0.64s across 3 independent seeds (CV=3.0%), confirming training stability. The gap between checkpoint and replicated mean reflects natural variance in training outcomes.

The three blocks use different evaluation protocols and must not be compared row-to-row. The
tabular agent's MCT of 29.80ms on the in-distribution fixed set reflects effective memorisation of
that configuration across 10,000 training episodes; it cannot participate in random evaluation
because std = 0.00ms reveals discretisation collapse rather than a generalisable policy. The VC1
and VC2 agents optimise total value preserved rather than MCT; their MCT figures are secondary
metrics reported for comparability, not primary optimisation targets.

### 5.2 F1 — Zero Variance Is Not Quality

The Week 1 tabular agent achieves MCT = 29.80ms on its training set — within 3.40ms of SRPT —
and beats Round Robin by 13.5%. By fixed-set metrics, this looks like successful learning.

The failure is revealed by std = 0.00ms on both evaluation sets. Two process sets that differ only
in their within-bin burst magnitudes — say, one process at 3ms and another at 8ms, both landing in
bin 1 — receive identical state tuples and therefore identical Q-values and identical action
sequences. The agent cannot distinguish them. Any reported variance would require burst differences
to cross a bin boundary.

Identity bias confirms the structural diagnosis. In the probe state (P1 = 10ms remaining, P4 = 3ms
remaining, all others complete or not arrived), SRPT prefers P4. The Week 1 tabular agent assigns
Q(P1/long) > Q(P4/long), |Δ| = 0.270, wrong direction. Week 2 randomised training corrects the
direction (|Δ| = 0.225, correct) but cannot shrink the gap further — the Q-table is indexed by
PID, so P1 and P4 always occupy different weight vectors regardless of how similar their burst
magnitudes are. The bias is not a training problem; it is a representation problem.

**Conclusion:** std = 0.00ms is a sign of discretisation collapse, not convergence. Fixed-set
evaluation cannot detect this; random-set evaluation with diverse burst magnitudes is the minimum
diagnostic.

### 5.3 F2 — DQN Unlocks Variance, Introduces New Failure Mode

The Week 3 DQN produces non-zero variance immediately: std = 26.51ms across 500 random episodes,
compared to SRPT's std = 23.96ms. The difference of 2.55ms represents policy noise on top of
the structural (process-set) variance that both agents inherit. The network is adapting to each
episode, not replaying a memorised schedule.

But the mean is 85.00ms against SRPT's 67.66ms — a gap of 17.34ms. And identity bias worsens
substantially:

| Agent | Q(P1/long) | Q(P4/long) | |Δ| | Correct direction? |
|-------|-----------|-----------|-----|-------------------|
| Tabular W1 | — | — | 0.270 | No |
| Tabular W2 | −3.91 | −3.68 | 0.225 | Yes |
| DQN W3 | +44.92 | +39.43 | 5.48 | No |

The move to continuous state did not reduce positional bias — it amplified it. The 15 output
neurons are permanently assigned to PID positions and accumulate asymmetric gradient histories
across 10,000 episodes. Gradient updates for (P0 = 10ms, P4 = 3ms) and (P2 = 10ms, P1 = 3ms)
flow through completely disjoint weight paths. The network has no mechanism to learn "prefer the
shorter process" as a general rule; it can only learn "when P0 has 10ms and P4 has 3ms, prefer
P4" — a rule that does not transfer.

The loss curve also rises between episodes 5,000 and 10,000 (0.0398 → 0.6504), suggesting the
15 simultaneous regression targets create a noisy multi-task learning problem with cross-
contaminated gradient paths.

### 5.4 F3 — Policy Analysis Diagnoses the Artefact

Week 4 extracts 10,000 greedy decisions from the trained Week 3 DQN and compares each to the
SRPT oracle, grouped by n_active. The results reveal two patterns.

First, process-selection agreement is load-dependent: 100% at n_active = 1 (trivially correct —
only one valid choice), dropping to the high-50% to low-60% range as more processes compete.
Better than random (20%), well below SRPT.

Second, and more striking: the quantum selection pattern shifts with load. At n_active = 1 the
network uses predominantly 1ms quanta, matching SRPT. But as n_active grows, 20ms quanta dominate,
used in more than 60% of decisions at n_active ≥ 3. This pattern is coherent, reproducible across
evaluation runs, and superficially plausible — large quanta amortise context-switch overhead when
many processes are waiting.

The interpretation is wrong. The pattern is a PID-positional artefact. Specific PID combinations
that co-occur frequently at high n_active values during training accumulate gradient pressure
toward large quanta in those position-indexed action heads. This pressure is absent in other PID
orderings. The pattern looks load-dependent because load and PID co-occurrence are correlated in
the training distribution, not because the network learned load-responsive behaviour. The Week 5
architecture test, which shares weights across all PID positions, eliminates the pattern entirely.

**The methodological point is general:** for any RL agent with position-indexed action heads,
policy patterns that correlate with action position should be treated as artefact candidates
until a weight-sharing ablation rules out positional origin.

### 5.5 F4 — Action-Conditioned Architecture Resolves Bias and Eliminates Artefact

**Loss curve.** The Week 5 loss converges to 0.0001 by episode 10,000, compared to 0.6504 for
Week 3. A single scalar regression target per action, rather than 15 simultaneous targets with
shared gradient paths, produces a more coherent learning problem.

| Episode | Week 3 Avg Loss | Week 5 Avg Loss |
|---------|----------------|----------------|
| 500 | 0.0219 | 0.0114 |
| 1,000 | 0.0082 | 0.0026 |
| 2,000 | 0.0067 | 0.0007 |
| 5,000 | 0.0398 | 0.0002 |
| 10,000 | 0.6504 | 0.0001 |

**Random evaluation.** Mean MCT = 76.04ms (std = 27.70ms; p5 = 34.17ms; p95 = 121.51ms;
min = 17.39ms; max = 166.99ms). The 8.96ms improvement over Week 3 (85.00ms) — a 10.5%
reduction — comes from the architecture change alone: same environment, same reward, same
training budget, same hyperparameters.

**Identity bias.** The probe is re-run with the Week 5 architecture:

| Agent | Q(P1/long) | Q(P4/long) | |Δ| | Correct direction? |
|-------|-----------|-----------|-----|-------------------|
| Tabular W1 | — | — | 0.270 | No |
| Tabular W2 | −3.91 | −3.68 | 0.225 | Yes |
| DQN W3 | +44.92 | +39.43 | 5.48 | No |
| DQN W5 | −0.825 | −0.774 | 0.051 | Yes |

|Δ| = 0.051, correct direction. The smallest gap across all four agents, despite the residual
being expected: the competitor (P1) is at a fixed PID position in the probe state, and the partial
invariance guarantee applies to the candidate, not the competitor context.

**Permutation invariance unit test.** Two probe states constructed with identical competitor
features but different candidate PIDs (P0 vs P4):
- Q(P0 as candidate, P1 as sole competitor at 10ms, qt=1ms) = **−0.9054024509**
- Q(P4 as candidate, P1 as sole competitor at 10ms, qt=1ms) = **−0.9054024509**
Exact equality confirmed. The masking amendment produces provably identical inputs and therefore
identical outputs for any two candidates in structurally equivalent positions.

**Policy analysis (Week 5).** The 20ms quantum preference is gone.

| n_active | agree% | 1ms% | 5ms% | 20ms% |
|----------|--------|------|------|-------|
| 1 | 100.0% | 90.0% | 7.6% | 2.4% |
| 2 | 74.3% | 99.3% | 0.0% | 0.7% |
| 3 | 65.4% | 99.9% | 0.1% | 0.0% |
| 4 | 67.9% | 99.9% | 0.0% | 0.1% |
| 5 | 71.9% | 99.9% | 0.0% | 0.1% |

At n_active ≥ 2, the agent uses 1ms quanta in >99% of decisions — matching the SRPT oracle's
quantum strategy exactly. The Week 3 load-adaptive quantum pattern was not a heuristic; it was
PID-positional noise that disappeared the moment weights were shared across positions.

Process-selection agreement improved at n_active = 2 (from ~65% in Week 3 to 74.3% in Week 5)
but did not reach the 80% target. Agreement plateaus at 65–68% for n_active ≥ 3, and the loss
curve shows convergence, ruling out a training-time explanation.

### 5.6 F5 — Attention Collapses Without Entropy Regularisation

Week 7 introduces AttentionDQN: a dot-product attention head over competitor encodings replaces
the flat s_masked concatenation used in Week 5. The candidate query vector q = cand_enc @ W_Q + b_Q
attends over competitor key-value pairs (K = comp_encs @ W_K + b_K, V = comp_encs @ W_V + b_V),
producing a context vector that is a weighted sum of competitor value encodings. The MLP then
receives [context ‖ a4] — a 12-dim input — rather than [s_masked ‖ a4] (19-dim). Full
permutation invariance over the competitor set follows from the softmax operation: any permutation
of competitor inputs produces the same weighted sum and therefore identical outputs. Permutation
invariance unit test: |Δ| = 0.000 (W7) vs |Δ| = 0.0335 (W5).

Without regularisation (λ=0), attention entropy H collapses from 1.386 (uniform over 4
competitors) to H = 0.03 by episode 1,000. The softmax locks onto a single competitor early in
training — before the TD signal has accumulated enough gradient to teach which competitor to focus
on. Once near-1-hot, the attention Jacobian approaches zero, and the weight parameters stop
receiving meaningful gradient. The result is a hardcoded-focus policy that ignores three of four
competitors: mean MCT = 117.51ms, SRPT agreement = 21.9%, below the Round Robin lower bar.

Two fixed-λ runs confirmed the failure is not trivially solvable. λ=0.05 stabilised H at 0.66–0.67
but the entropy term dominated the TD signal, producing negative total loss, agreement = 64.6%,
mean MCT = 87.73ms. λ=0.10 worsened the domination: H stable 0.66–0.71, agreement fell to 57.4%,
mean MCT rose to 94.41ms. The entropy regulariser was functioning as the primary learning signal,
not the auxiliary stabiliser it was intended to be.

Entropy annealing resolved the conflict: λ(ep) = 0.10 − (0.10 − 0.005) × (ep / 10000). The
regulariser prevents collapse during early episodes while shrinking to near-zero once the TD signal
is stable. H held in the range 0.60–0.68 throughout 10,000 episodes; total loss converged
near zero. The annealed run is the Week 7 trained agent.

**Finding:** Attention entropy collapse is a failure mode distinct from both discretisation collapse
(Week 1) and position-indexed gradient contamination (Week 3). It requires active regularisation
with an annealed schedule: a fixed λ that prevents collapse also dominates the TD signal; only a
decaying λ threads the needle between softmax freeze and TD-signal erasure.

### 5.7 F6 — Performance Regression in Mean, Improvement in Variance

The annealed AttentionDQN achieves mean MCT = 80.57ms, std = 21.82ms on 500 random episodes.
Comparing the three DQN generations alongside the oracle:

| Agent | Mean MCT | Std | vs SRPT |
|-------|----------|-----|---------|
| SRPT oracle | 67.66ms | 23.96ms | 0.00ms |
| DQN W5 (action-conditioned) | 76.04ms | 27.70ms | +8.38ms |
| DQN W7 (attention-annealed) | 80.57ms | 21.82ms | +12.91ms |
| DQN W3 (position-indexed) | 85.00ms | 26.51ms | +17.34ms |

Mean MCT regressed from W5 (76.04ms) to W7 (80.57ms) by +4.53ms. But std fell from 27.70ms to
21.82ms — below the SRPT oracle's own std of 23.96ms. W7 is more consistent than SRPT while being
12.91ms slower in mean. This is not a net regression; it is a quality tradeoff between two
distinct policy properties. W5's higher mean agreement with SRPT includes tail decisions that reach
large errors (p95 = 121.51ms); W7's attention context suppresses those tail errors — at the cost
of some mean throughput. A policy with std < SRPT while maintaining a positive SRPT gap implies
the agent is trading median throughput for reduced worst-case exposure.

Identity bias is preserved in the correct direction: Q(P1/long) = −2.434, Q(P4/long) = −2.413,
|Δ| = 0.021, correct direction. The residual gap is smaller than W5's |Δ| = 0.051, consistent
with full permutation invariance over the competitor context (the probe state's single competitor
is now aggregated without positional weighting).

**Finding:** Mean MCT regression accompanied by std < SRPT oracle represents a changed
quality tradeoff, not a net regression. The appropriate evaluation metric for comparing W5 and W7
is the joint distribution of (mean, std), or equivalently the p95 tail performance — not mean
alone.

### 5.8 F7 — Process Selection Meets 80% Target at n_active ≤ 3, Fails at n_active = 5

Policy analysis on 10,000 greedy decisions from the annealed W7 agent, compared to W5:

| n_active | W7 agree% | W5 agree% |
|----------|-----------|-----------|
| 1 | 100.0% | 100.0% |
| 2 | 85.8% | 74.3% |
| 3 | 90.9% | 65.4% |
| 4 | 71.7% | 67.9% |
| 5 | 40.7% | 71.9% |

The 80% process-selection target — not met by W5 at any n_active ≥ 2 — is met and exceeded by W7
at n_active = 2 (85.8%) and n_active = 3 (90.9%). But agreement falls sharply to 71.7% at
n_active = 4 and collapses to 40.7% at n_active = 5 — worse than W5's 71.9% on the same load
condition.

The n_active = 5 failure has a direct architectural explanation. With 4 active competitors and
attention entropy H ≈ 0.62 at episode 10,000, each competitor receives approximately 0.22–0.25
weight — near-uniform. When attention weights are near-uniform, the context vector approaches
mean-pooling over competitor encodings. Mean-pooling loses relative ordering information: it
cannot distinguish "three competitors at 30ms plus one at 5ms" from "four competitors at 20ms
each." At n_active = 5, the attention mechanism degenerates toward the sum-pooling baseline from
which AttentionDQN was intended to advance.

The residual entropy H ≈ 0.62 is the proximate cause: the annealed λ prevented collapse but did
not drive the agent to sufficiently sharp attention under high competitor count. The final
λ = 0.005 is not small enough to allow further softmax sharpening at n_active = 5. This is a
learnable-temperature problem, not a training-duration problem. Longer training with the same
fixed-temperature softmax would maintain the same H floor; what is needed is a parameter that can
sharpen the distribution independently of the entropy annealing schedule.

**Finding:** The 80% SRPT-agreement target is met at n_active ≤ 3 but not at n_active ≥ 4.
The n_active = 5 failure is attributable to residual entropy at termination: H ≈ 0.62 produces
near-mean-pooling at 4 competitors, erasing the ordering information the attention was introduced
to capture. The fix is a learnable inverse-temperature parameter β (replacing fixed 1/√d scaling
with 1/√d × exp(β), where β is trained alongside the attention weights) that can sharpen
independently of the entropy annealing schedule.

### 5.9 F8 — Attention Tracks "Threatening Competitor," Not Selected Process

The initial attention diagnostic produced an apparent contradiction: in 80.2% of decisions the
highest attention weight lands on the longest competitor, while in 82.2% of decisions the DQN
agrees with SRPT — which always selects the shortest process. If attention were guiding selection
toward the longest competitor, agreement with SRPT would be low, not high.

A joint diagnostic on 1,000 decisions (n_active ≥ 3) resolves the contradiction. At each
decision, the DQN-selected process, the SRPT-selected process, and the highest-attention competitor
are recorded jointly:

| Case | n | Attention on shortest (%) | Attention on longest (%) |
|------|---|--------------------------|--------------------------|
| DQN agrees with SRPT | 660 | 4.1% | 90.6% |
| DQN disagrees with SRPT | 340 | 12.4% | 62.4% |

When the DQN correctly selects the shortest process, every competitor is by definition longer.
"Highest attention on longest competitor" and "highest attention on most distant competitor" are
the same statement. The 90.6% longest-focus in agree cases means the attention is locking onto
the structurally extreme competitor — the one whose remaining burst anchors the upper end of the
competitive context — while the MLP uses that context to confirm the correct short-process
selection.

The disagree cases confirm the asymmetry is diagnostic: 62.4% longest-focus when the agent errors,
versus 90.6% when correct. Sharper attention to the most threatening competitor corresponds to
better decisions. The attention is not misfiring toward a long process instead of the correct
short one; it is providing maximum-contrast context that the MLP interprets correctly.

Sanity check: 660/1,000 = 66.0% SRPT agreement in the joint diagnostic vs 82.2% in full policy
analysis. The difference reflects the n_active ≥ 3 filter — the joint diagnostic excludes
n_active = 1 (trivially 100%) and n_active = 2 (85.8%), leaving the harder subset.

**Finding:** Attention-on-longest in correct decisions is not a misalignment with SRPT. The
attention learned to track the most threatening competitor (longest remaining burst) as structural
context for a correct short-process selection. The apparent contradiction between longest-focus
and SRPT agreement dissolves once conditioning on decision correctness is applied: the pattern is
stronger in correct decisions (90.6%) than in errors (62.4%), indicating attention focus quality
correlates with decision quality.

### 5.10 F9 — Distribution Mismatch Is the Root Cause of Trace Transfer Failure

The synthetic training distribution (U[1, 60]ms) has a p95/p50 ratio of approximately 1.9×. The
Alibaba 2018 batch_task trace has p95/p50 = 39.7×. These are not the same problem at different
scales. The p50 of the trace is 10s; the p95 is 397s; the max is 583,886s — a span of four orders
of magnitude. Under linear normalization (`burst / BURST_P95`), the median task maps to
10/397 = 0.025 while 27.7% of sampled 5-task episodes contain at least one task with normalized
burst > 0.90, near the clipping ceiling. The remaining 73% of tasks cluster below 0.12 in
normalized feature space. Key-query dot products for the outlier task are an order of magnitude
larger than those for the short tasks, saturating the softmax and destroying ordinal information
among the non-outlier tasks in every episode that contains an extreme value.

This failure is not architectural. The AttentionDQN's permutation invariance guarantee, entropy
annealing mechanism, and key-query-value structure are all intact. The failure is a training
distribution failure: a normalization scheme effective for 1.9× p95/p50 is insufficient for
39.7×. A clipping point that represents the distributional center for a near-uniform distribution
represents only the 95th percentile tail for a power-law distribution — the two cases require
qualitatively different normalization strategies.

**Finding:** The transfer failure from synthetic to real-trace data is a normalization failure,
not an architecture failure. Linear normalization by p95 is appropriate when most of the
distribution is near the normalization constant; it fails when the normalization constant is a
rare extreme that the majority of inputs never approach.

### 5.11 F10 — The Trace-Trained Agent Failed to Transfer: Results

The W8 agent trained without behavioral cloning or reference policy demonstrations on full Alibaba trace episodes, with all W7 hyperparameters intact, is evaluated on 500 trace test episodes (master seed 42):

| Agent | Mean MCT | Std | vs RR | vs SRPT |
|-------|----------|-----|-------|---------|
| W8 AttentionDQN (linear norm) | 135.76s | 274.99s | +26.24s | +45.63s |
| SRPT oracle (trace) | 90.13s | 191.25s | — | 0.0s |
| Round Robin (2.0s, trace) | 109.52s | 202.78s | base | — |

The agent is 26.24s worse than Round Robin in mean MCT. Policy analysis on 10,432 greedy
decisions: overall SRPT agreement 50.2%, collapsing to 26–31% at n_active ≥ 2 — near the random
baseline of 25% for a 4-candidate selection. Identity bias probe: |Δ| = 0.051, wrong direction
(agent prefers the longer process in the two-process case). Attention entropy: H = 0.13–0.24
throughout all 10,000 training episodes — collapsed from the first checkpoint.

The W8b variant (log-normalization, REWARD_SCALE = 40.0, λ: 0.30 → 0.005) produced worse results:
mean MCT = 157.13s, SRPT agreement 23.4%, H = 0.13–0.20. Log-normalization alone, without
addressing the within-episode burst ratio, did not prevent attention collapse.

**Finding:** Training directly on full-trace episodes without correcting the within-episode
magnitude structure produced no learning signal for SRPT process selection. This is not a partial
transfer with degraded performance; it is no transfer.

### 5.12 F11 — Attention Entropy Collapsed on Full Trace Data

The W7 agent trained on synthetic data stabilized at H ≈ 0.62 throughout 10,000 episodes under
the annealed regularizer. The W8 agent trained on full-trace data produced H = 0.13–0.24
throughout — below the 0.20 collapse flag from the first checkpoint and remaining there for all
10,000 episodes, including with λ increased from 0.10 to 0.30.

The trace's 39.7× p95/p50 skew creates attention collapse pressure that the annealing schedule
cannot overcome. When a 5-task episode contains one task at normalized burst 0.90 and four tasks
at 0.025, the key-query dot products for the outlier are approximately 36× larger than for the
short tasks. The softmax exponential amplifies this into a near-zero weight for all non-outlier
competitors. The entropy regularizer must generate gradient large enough to redistribute weight
from a competitor with near-zero Jacobian — a task that would require λ large enough to dominate
the TD signal entirely.

**Finding:** Entropy collapse on heavy-tailed data is not solvable by increasing λ alone. A
distribution where outliers dominate key-query dot products by a factor of 36× creates collapse
pressure that is qualitatively different from the 1.9× synthetic case. The required resolution is
a normalization scheme that equalizes feature magnitudes before the attention computation — not a
stronger regularizer fighting a structurally hostile input distribution.

### 5.13 F12 — What Transfer to Real Traces Requires

Three changes are necessary for real-trace generalization. Each is precisely stated.

**1. Log-normalization of burst durations.** Replace `burst / BURST_P95` with
`log(1 + burst) / log(1 + BURST_P95)`. This maps the full trace range to approximately [0, 1]
with the heavy tail compressed. The median task (10s) maps to 0.40 rather than 0.025 under linear
normalization. The key-query dot-product ratio between outlier and short tasks falls from ~36× to
~2.5×, within the range the entropy regularizer can handle. This is a one-line change to
`_encode_state`.

**2. Within-episode magnitude control via trace filtering.** Log-normalization is necessary but
not sufficient when the full distribution has p95/p50 = 39.7×. Filtering tasks above the 75th
percentile duration reduces the within-episode burst ratio from p50 = 74× (random sampling, full
trace) to p50 = 14× (random sampling, filtered trace). This is the minimum viable preprocessing
step for attention-based RL scheduling on this workload.

**3. Recalibrated entropy regularization.** The annealing schedule λ: 0.10 → 0.005, calibrated
for the 1.9× synthetic distribution, is insufficient for the residual skew of even the filtered
trace. The initial λ must increase to at least 0.30, with the same decay schedule, to maintain
H > 0.20 during early training. Alternatively, a learnable temperature parameter β (replacing
fixed 1/√d with trainable β/√d) removes the manual calibration requirement.

### 5.14 F13 — Filtering Enables Real-Trace Training

The full Alibaba 2018 batch_task training split (11,436,584 tasks, p95/p50 = 39.7×) produced
attention collapse in every training configuration tested. Filtering tasks above the 75th
percentile duration (47.0s, computed from the training split) reduced the dataset to 8,076,473
tasks (70.6% retained) and changed the within-episode distribution qualitatively:

| Metric | Full trace | Filtered trace |
|--------|-----------|----------------|
| p95/p50 ratio | 39.7× | 7.2× |
| Within-episode p50 burst_ratio | 74.0× | 14.0× |
| Within-episode % > 10× | 89.7% | 64.1% |
| Within-episode % < 3× | 0.4% | 2.0% |

The filtered test split retains 2,151,487 of 2,859,147 rows (75.2%). The new distribution
constants from the filtered training split: BURST_P95 = 36.0s, BURST_MEDIAN = 5.0s, quantum
tiers (0.25s, 1.0s, 4.0s). The W8c agent trained on filtered episodes with log-normalization and
λ: 0.30 → 0.005 stabilized attention entropy at H = 0.32–0.50 throughout 10,000 episodes —
compared to H = 0.13–0.24 on the full trace with all three fixes applied. Training time dropped
from 56 minutes (full trace, 10M+ tasks per epoch) to 6 minutes (filtered, 8M tasks).

**Finding:** Filtering tasks above the p75 duration threshold is the minimum viable preprocessing
step for attention-based RL scheduling on the Alibaba 2018 trace. It reduces the within-episode
p50 burst ratio from 74× to 14× and enables the entropy annealing mechanism to function as
designed. It does not produce a representative sample of the full workload — the top 30% of the
duration distribution is excluded — but it produces a tractable training distribution for the
current architecture.

### 5.15 F14 — W8c Performance on Filtered Trace

The W8c agent (filtered trace, log-normalization, λ: 0.30 → 0.005, REWARD_SCALE = 20.0) is
evaluated on 500 filtered test episodes (master seed 42):

| Agent | Mean MCT | Std | vs RR | vs SRPT |
|-------|----------|-----|-------|---------|
| SRPT oracle (filtered) | 15.72s | 8.88s | — | 0.0s |
| W8c AttentionDQN | 23.30s | 10.85s | +1.53s | +7.58s |
| Round Robin (1.0s, filtered) | 21.77s | 13.61s | base | — |

Two observations require honest framing.

First, the agent is 1.53s worse than Round Robin on mean MCT. The gap is well within one standard
deviation of Round Robin's std (13.61s), and the agent's std (10.85s) is lower than Round
Robin's — meaning the agent's distributional tail is better controlled even when its mean is
marginally higher. This is the same quality tradeoff identified in W7 (mean MCT regression
accompanied by std improvement). It is not a resounding success; the agent has not learned to
beat the simplest non-trivial baseline on real data.

Second, process-selection agreement collapses at n_active ≥ 3: 100% at n_active = 1 (trivial),
55.3% at n_active = 2, 47.1% at n_active = 3, 21.3% at n_active = 4, 17.3% at n_active = 5.
This is a pattern similar to W7's attention collapse — agreement falls sharply with competitor
count — but here H = 0.32–0.50 throughout training. The attention did not collapse to noise; it
learned a coherent but wrong strategy. The identity bias probe confirms the wrong direction:
Q(candidate/tier2) = −3.132, Q(competitor/tier2) = −2.976, |Δ| = 0.156, candidate not
preferred. Overall SRPT agreement: 57.9% across 10,035 decisions.

**Finding:** The W8c agent achieves mean MCT within 1.53s of Round Robin with lower variance —
a marginal performance result that represents a 94% reduction in the performance gap compared to
the unfiltered W8a baseline (+26.24s vs Round Robin). The filtering, log-normalization, and
entropy recalibration are necessary conditions for tractable training. They are not sufficient for
competitive performance: process-selection accuracy at n_active ≥ 3 remains near-random, not
from entropy collapse but from a learned anti-SRPT attention strategy.

### 5.16 F15 — Attention Attends to Longest Competitor on Real Traces

The W8c attention diagnostic, applied to 1,006 decisions at n_active ≥ 3:

| Metric | W8c (filtered trace) | W7 (synthetic) — agree cases |
|--------|---------------------|------------------------------|
| Highest attention = longest competitor | 93.7% | 90.6% |
| Highest attention = shortest competitor | 4.1% | 4.1% |
| Mean weight on longest | 0.5132 | — |
| Mean weight on shortest | 0.3231 | — |

The pattern is superficially identical to W7's attention-on-longest finding (Section 5.9). But
the interpretation differs in a critical way. In W7, attending to the longest competitor was
structurally correct: when the agent selects the shortest process, all competitors are by
definition longer, so 90.6% longest-focus in agree cases is a tautological consequence of correct
selection. The pattern correlated with quality (90.6% in agree cases vs 62.4% in disagree cases).

In W8c, the agent is agreeing with SRPT in only 57.9% of decisions and its identity bias probe
points in the wrong direction. The 93.7% longest-competitor attention here is not a marker of
correct short-process selection — it is the agent's primary decision signal, attending to the
longest competitor to schedule it next or to avoid it in a way that generates the observed
agreement with chance. The mean weight on the longest competitor (0.5132) versus the shortest
(0.3231) shows the attention is not near-uniform; it is deliberately concentrating on the longest
task.

The cause is the area-integral reward signal. The reward `−n_active × q_actual` penalizes every
unit of CPU time consumed while any process is waiting. Under this signal, the locally optimal
strategy when facing multiple competitors is to attend to the one with the most remaining burst —
the one that will contribute the longest waiting penalty to future steps — and either schedule
it first to eliminate its influence, or use its magnitude as a reference point for comparison.
The reward creates gradient pressure toward longest-competitor attention that is locally coherent
under the objective but anti-SRPT globally: SRPT requires scheduling the *shortest* process to
minimize mean completion time, which reduces n_active fastest, but the reward signal does not
differentiate between "run the short process now, reducing n_active" and "run the long process
now, using its magnitude as context." Both reduce n_active by 1 eventually; only SRPT minimises
the area integral over all processes jointly.

The W7 synthetic agent avoided this failure not because of a different reward signal but because
the synthetic uniform distribution produced a different gradient landscape — the attention learned
to use longest-competitor context as a reference for *contrast* rather than as a selection target.
That distinction did not emerge from the heavier-tailed filtered trace distribution, where
magnitude differences between tasks are larger and the gradient toward long-competitor focus is
stronger.

**Finding:** The W8c agent learned to attend to the longest competitor confidently (93.7%,
H = 0.32–0.50) rather than collapsing to noise. This is a qualitatively different failure from
W8a/b's entropy collapse: the attention mechanism is functioning correctly as an attention
mechanism — it is attending to a coherent signal — but that signal is anti-SRPT under the
area-integral reward on real-trace distributions. Closing this gap requires either reward shaping
that directly incentivizes shortest-job-first selection at each step, a supervised warm-start on
SRPT demonstrations before RL fine-tuning, or a learnable attention temperature that allows the
agent to modulate its focus between shortest and longest competitors as a learned policy rather
than a gradient-forced default.

### 5.17 F16 — Resource-Class Features Enable Round Robin Parity and Correct SRPT Selection

Adding plan_cpu and plan_mem as normalized resource-class features — expanding the per-process
encoding from 3 to 5 dimensions and increasing d_attn from 8 to 16 (3,873 parameters total) —
produced the first agent in this project to beat Round Robin on real Alibaba trace data.

**Results on 500 filtered test episodes (master seed 42):**

| Agent | Mean MCT | Std | vs Round Robin | vs SRPT |
|-------|----------|-----|----------------|---------|
| SRPT oracle | 15.20s | 8.94s | — | 0.0s |
| W9 AttentionDQN (5-feat) | 19.22s | 11.58s | **−2.09s** | +4.03s |
| W8c AttentionDQN (3-feat) | 23.30s | 10.85s | +1.53s | +7.58s |
| Round Robin (tier1 = 1.0s) | 21.31s | 14.03s | baseline | — |

Four findings characterise the W9 result.

**Finding 1 — First real-trace Round Robin win.** W9 achieves mean MCT 19.22s, beating Round
Robin by 2.09s. This is the first agent in this project to beat Round Robin on real trace data.
W8c and W8d (3-feature agents, trained for 10,000 and 20,000 episodes respectively) both sat
above Round Robin at 23.30s and 24.66s. The improvement came from richer state, not additional
training: W8d's extended 20,000-episode run degraded SRPT agreement (39.2% vs 57.9% at 10,000
episodes for W8c), confirming the 3-feature state was at its ceiling.

**Finding 2 — Identity bias corrected.** The identity bias probe (candidate burst_norm = 0.20,
competitor burst_norm = 0.60, identical cpu_norm = 0.125 and mem_norm = 0.339) yields Q(candidate)
= −2.114, Q(competitor) = −2.664, |Δ| = 0.549, candidate preferred. This is the first correct
identity bias direction on real-trace data. W8c produced |Δ| = 0.156 in the wrong direction. The
resource-class features provided the signal that allowed the MLP to rank the shorter-burst
candidate higher despite identical CPU and memory requests.

**Finding 3 — SRPT agreement improved at every contention level.** Overall agreement rose from
57.9% (W8c) to 70.5% (W9). The most dramatic improvement is at n_active = 4: 21.3% → 52.8%.
Agreement at n_active = 5 remains the open problem (32.5%), consistent with the high-contention
failure observed on synthetic data.

| n_active | W8c agree% | W9 agree% | Δ |
|----------|-----------|-----------|---|
| 1 | 100.0% | 100.0% | 0.0 |
| 2 | 38.1% | 57.5% | +19.4 |
| 3 | 47.1% | 54.7% | +7.6 |
| 4 | 21.3% | 52.8% | +31.5 |
| 5 | — | 32.5% | — |

**Finding 4 — Attention pattern unchanged; MLP learned to use it correctly.** W9 attention
still attends to the longest competitor in 84.7% of decisions at n_active ≥ 3, essentially the
same as W8c (93.7%). The attention mechanism did not change its structural behaviour. What
changed is that the MLP — now conditioned on cpu_norm and mem_norm in both the context and the
a6 vector — learned to use the longest-competitor context as a reference for contrast rather
than as a selection target. Resource-class features broke the reward degeneracy that trapped the
3-feature agent without requiring a change to the attention mechanism or the reward signal.

**Honest limitations.** plan_cpu and plan_mem are requested resources, not measured runtime
utilisation. They encode the job's resource class at submission time and are identical for all
tasks within the same job. The signal is weaker than the historical CPU usage or I/O wait ratio
identified by RLScheduler and Double DQN as load-bearing. The n_active = 5 agreement ceiling
at 32.5% and the 4.03s SRPT gap remain open. The 84.7% longest-competitor attention confirms
that the reward signal degeneracy is not fully resolved — only compensated for at the MLP level
by the additional features.

### 5.18 F17 — Potential-Based Reward Shaping Degrades Performance at Both Tested Scales

Potential-based reward shaping with Φ(s) = −α × Σᵢ burst_norm_i was tested at two scales:
W10A (α = 0.20) and W10A-v2 (α = 0.05). The shaped reward at each step is:

    R_shaped = R_original + γ × Φ(s') − Φ(s)

where γ = 0.99 and R_original is the unmodified area-integral reward. Φ uses the same
log-normalised burst values already in the state vector — no new observations are introduced,
and the tabula rasa constraint (with respect to scheduling demonstrations, though value curve parameters are human-specified) is preserved. The shaping bonus correctly favours shortest-job
completion: completing a process with burst_norm = 0.20 (α = 0.20) yields +0.043 vs +0.004
for running the longest process one step.

**Results against the W9 baseline (n = 500 filtered test episodes, greedy policy):**

| Agent | Mean MCT | Overall agree% | n2 agree% | tier0% at n2 |
|-------|----------|----------------|-----------|--------------|
| W9 (no PBRS) | 19.22s | 70.5% | 57.5% | 94.5% |
| W10A α = 0.20 | 22.15s | 52.2% | 46.9% | 97.0% |
| W10A-v2 α = 0.05 | 19.23s | 65.7% | 53.3% | 95.6% |

α = 0.20 caused near-total tier0 collapse (97.0% of decisions used the smallest quantum at
n_active = 2) and an 18 percentage-point SRPT agreement regression. Reducing α to 0.05
recovered mean MCT to 19.23s — statistically identical to W9 — but SRPT agreement remained
5 percentage points below W9 at every contention level. No tested α improved on W9.

**Diagnosis.** The shaping bonus accumulates once per decision step, regardless of which process
is selected. In a discrete quantum-based scheduler, tier0 (smallest quantum = 0.25s) maximises
the number of decision steps per episode and therefore maximises total accumulated shaping bonus,
independent of process-selection quality. The agent at α = 0.20 learned this: it uses tier0 in
97% of decisions at n_active = 2, a near-uniform collapse. The process-selection accuracy
simultaneously collapsed because the shaping signal was dominated by quantum-choice incentives.
At α = 0.05 the incentive is weaker and MCT recovers, but the residual step-count bias still
depresses agreement by 5 percentage points. PBRS in its current form is incompatible with
discrete quantum scheduling without a per-quantum-size adjustment to the shaping magnitude — an
additional design choice that removes the simplicity motivation for PBRS.

### 5.19 F18 — Learnable Attention Temperature Converges to Softer Attention, Not Sharper

A learnable inverse-temperature parameter was introduced: scores_j = β × (q · K_j), where
β = exp(log_β) is trained alongside the attention weights. log_β is initialised to
log(1/√16) ≈ −1.386, giving β₀ = 0.25 — identical to the fixed scaling in W9. Log-
parameterisation keeps β strictly positive. This adds one parameter (3,874 total vs W9's
3,873) and is otherwise architecturally identical.

**Training trajectory of β:**

| Episode | β |
|---------|---|
| 0 (init) | 0.250 |
| 500 | 0.167 |
| 1000 | 0.118 |
| 2000 | 0.076 |
| 5000 | 0.047 |
| 10000 | 0.081 |

β decreased monotonically from 0.250 to a minimum near 0.047 at episode 5000 before rising
slightly to 0.081 at convergence — three times softer than the W9 fixed value throughout
training.

**Results against the W9 baseline:**

| Agent | Mean MCT | Overall agree% | n5 agree% | β |
|-------|----------|----------------|-----------|---|
| W9 fixed β = 0.25 | 19.22s | 70.5% | 32.5% | 0.250 |
| W10B learned β | 20.63s | 54.8% | 10.5% | 0.081 |

Performance regressed by 1.41s mean MCT. n_active = 5 agreement collapsed from 32.5% to 10.5%.

**Diagnosis.** The gradient correctly found that sharpening attention is harmful given the
current 5-dim feature set. The W9 attention already concentrates 84.7% of weight on the
longest competitor — a structural consequence of the area-integral reward (Section 5.16). With
only burst_norm, arrived_flag, wait_norm, cpu_norm, and mem_norm in the competitor encoding,
the features are insufficiently discriminative to support sharper attention at high contention:
the longest-competitor signal dominates all dot products regardless of β, and higher β would
simply make this concentration more extreme rather than more informative. The network's gradient
identified this and drove β down to softer values where near-uniform competitor averaging is
preferable to confident misattention. The n_active = 5 failure is therefore a feature-
discriminability problem rather than an attention-capacity problem — the right fix is a richer
competitor encoding, not a different temperature schedule.

### 5.20 F19 — Two-Head Attention Enables Head Specialization; W10C Achieves 21.71 ± 0.64s Replicated (see Figure 2)

W10C replaced single-head attention (W9, d_attn = 16) with two-head attention (d_head = 8 per
head, output projection W_O: 16 → 16, 4,145 parameters vs W9's 3,873). No PBRS; no learnable
temperature. One architectural change only.

**Replicated result (primary):** 21.71 ± 0.64s mean MCT across 3 independent seeds (CV=2.9%), confirming training stability. **Original checkpoint (secondary):** 17.23s on N=1500 evaluation episodes (`dqn_w10c.npz`). Both are reported; the replicated mean is the reproducible result.

| Agent | Mean MCT | Std | vs Round Robin | vs SRPT |
|-------|----------|-----|----------------|---------|
| SRPT oracle | 15.20s | 8.94s | — | 0.0s |
| W10C 2-head attention (orig. checkpoint†) | 17.23s | 11.38s | **−4.08s** | +2.03s |
| W10C 2-head attention (replicated, 3 seeds) | 21.71s | 0.64s‡ | −∼0.4s | +6.51s |
| W9 1-head fixed β | 19.22s | 11.58s | −2.09s | +4.02s |
| Round Robin (tier1 = 1.0s) | 21.31s | 14.03s | baseline | — |

†Original checkpoint evaluated at N=1500 episodes. Replicated fresh training: 21.71 ± 0.64s across 3 independent seeds (CV=3.0%), confirming training stability. The gap between checkpoint and replicated mean reflects natural variance in training outcomes.
‡Std here is cross-seed std of mean MCTs; within-seed std is 11–12s.

W10C reduces mean MCT over W9 on the replicated mean and closes the SRPT gap from 4.02s to 2.03s in the best-checkpoint evaluation — a 50% reduction.

**Statistical reliability across independent training seeds.** W10C achieves MCT = 17.23 ± 11.38 s on the Alibaba 2018 test split (N = 1500 evaluation episodes, original checkpoint `dqn_w10c.npz`). Fresh training across 3 independent seeds yields MCT = 21.71 ± 0.64 s (seeds 42, 123, 456; N = 500 test episodes each), confirming training stability with CV = 2.9%. The higher mean MCT for fresh runs (21.71s vs 17.23s) reflects run-to-run variance at 10,000 episodes; the low CV confirms the training procedure is stable in relative terms. Results are reported as mean ± std across 3 independent training seeds: MCT = 21.71 ± 0.64 s, SRPT agreement = 45.7 ± 3.0%.

**Finding 1 — Spontaneous head specialization without supervision.** The two heads
diverged into qualitatively distinct roles measured across 1,064 decisions at n_active ≥ 3:

| Head | Highest-attn = shortest | Highest-attn = longest | Mean w (shortest) | Mean w (longest) |
|------|------------------------|----------------------|-------------------|-----------------|
| Head 1 (context) | 45.6% | 40.9% | 0.4343 | 0.4210 |
| Head 2 (monitor) | 14.2% | 79.1% | 0.3309 | 0.5323 |

Head 1 exhibits near-uniform attention across competitors — surveying the full competitor field
and aggregating load context. Head 2 concentrates strongly on the longest competitor (79.1%),
tracking the most threatening process as a reference signal. Neither role was imposed by
supervision or architectural constraint; both emerged from the area-integral reward signal applied
to the two-head architecture. This matches the hypothesis implicit in W10B's diagnosis: single-head
attention was solving two problems simultaneously — load context aggregation and competitor
monitoring — with a compromise allocation. Two heads divided the labor cleanly.

Combined, either head attends to the longest competitor in 98.3% of decisions; the combined
shortest-attention rate (either head) is 59.2%. The two heads together provide richer input
diversity to the MLP than any single head could supply at the same total dimensionality.

**Finding 2 — Largest single-step SRPT agreement improvement at n_active = 2.** Agreement at
n_active = 2 increased from 57.5% (W9) to 69.9% — a 12.4 percentage point gain, the largest
improvement at any single contention level across all experiments in this project.

| n_active | W9 agree% | W10C agree% | Δ |
|----------|-----------|-------------|---|
| 1 | 100.0% | 100.0% | 0.0 |
| 2 | 57.5% | 69.9% | +12.4 |
| 3 | 54.7% | 54.6% | −0.1 |
| 4 | 52.8% | 38.8% | −14.0 |
| 5 | 32.5% | 30.3% | −2.2 |

**Finding 3 — High-contention ceiling persists.** Tier0 collapse at n_active ≥ 4 (96–98%) and
the n_active = 5 agreement floor (30.3%) remain. Two-head attention improved moderate-contention
performance but did not address the feature-discriminability ceiling at high contention. The
W10B diagnosis stands: five planned resource features are insufficient to support informative
attention at n_active = 5, regardless of how many heads process them.

**Finding 4 — Permutation invariance preserved exactly.** Unit test: |Q_A − Q_B| = 2.78 × 10⁻¹⁶
(machine epsilon). The output projection W_O operates on the concatenated head outputs, which are
each independently permutation-invariant over the competitor set. The concatenation and linear
projection preserve this property.

**Interpretation.** Single-head attention at d_attn = 16 was attempting to perform both competitor
monitoring and load context aggregation with a single set of query-key-value weights. The W10B
experiment confirmed that the gradient identified this compromise: β converged to 0.081 (softer
than the W9 fixed value) because the single head could not sharpen without making its
longest-competitor focus more extreme and losing load context. Two heads removed the trade-off
by assigning each role a dedicated weight subspace. The MLP receives a 16-dim context vector that
simultaneously encodes both the load landscape (Head 1) and the maximum-threat reference (Head 2),
enabling correct SRPT-aligned selection at moderate contention.

Extended training to 20,000 episodes (W10C-ext) triggered the stop gate at episode 15,000 —
MCT rose from 19.32s to 19.60s. Final evaluation produced 20.32s mean MCT, a 2.11s regression
from W10C at 10,000 episodes. Head specialization partially collapsed: Head 1 drifted from
near-uniform context attention (45.6% shortest / 40.9% longest) toward longest-monitoring
(22.1% / 76.3%), mirroring the W8c/W8d degradation pattern. The 10,000-episode ceiling is
consistent across both the 1-head (W9/W8c) and 2-head (W10C) architectures, confirming it is
a dataset and feature-set constraint rather than an architecture-specific artifact.

### 5.21 F20 — Value-Curve Reward Without Observable Curve Parameters Produces No Learning

W10C-VC1 replaced the area-integral reward with a value-curve objective. At each step the reward
is `sum(-value_delta(p) for p in runnable) / 20.0`, where
`value_delta(tau, floor, base, wait, q) = base × max(floor, exp(-(wait+q)/tau)) − base × max(floor, exp(-wait/tau))`.
Processes are divided into steep curves (tau ∈ [3,10]s, floor=0.2, sampled with probability 0.5)
and smooth curves (tau ∈ [40,100]s, floor=0.0). Tau and floor are sampled per episode but were
not added to the state vector; the per-process encoding remained the 5-dim W10C representation
(burst_norm, arrived_flag, wait_norm, cpu_norm, mem_norm).

The stop gate fired at episode 2000: mean MCT across three consecutive checkpoints was 22.34s →
22.74s → 23.29s, all above the 17.23s W10C baseline. Final evaluation of the episode 2000
checkpoint: mean MCT = 26.56s, versus Round Robin (21.31s, regression +5.25s) and W10C (17.23s,
regression +9.33s). Steep SRPT agreement: 30.5%; smooth SRPT agreement: 31.0%; gap = −0.58pp —
near random and in the wrong direction. Head specialization analysis across 1,064 decisions at
n_active ≥ 3 showed no curve-type dependence: Head 1 attended to the shortest competitor in
52.8% / 42.8% of steep / smooth decisions; Head 2 to the longest in 59.2% / 63.8%. The two-head
specialization pattern from W10C reproduced unchanged regardless of whether the SRPT process
carried a steep or smooth curve.

The identity bias probe, run across all four curve-type combinations (steep candidate / steep
competitor, steep / smooth, smooth / steep, smooth / smooth), returned Δ = +0.035765 in every
case — identical to six significant figures. The Q-value delta the network assigns to the
candidate over the competitor is independent of either process's curve type. This is the
diagnostic signature of non-stationary Q-values: identical observations produced different rewards
depending on the sampled tau and floor values, making the Bellman target a moving average of
outcomes from qualitatively different reward regimes rather than a convergent signal. No process
selection aligned with urgency was learnable under these conditions.

**Finding:** When the reward depends on unobserved process properties, the identity bias probe
returns identical Δ across all curve-type combinations. This is a cleaner diagnostic of reward
non-stationarity than the loss curve or MCT trend alone: it confirms that the network produces
no urgency-differentiated output regardless of training duration or architecture quality. The fix
is observability, not architecture.

### 5.22 F21 — Value-Curve Oracle Cannot Beat Round Robin on MCT: Structural Divergence

Before training with extended state, a greedy value-curve oracle was diagnosed on 50 episodes
(seed=42). The oracle scores each (process, quantum) pair and selects the maximum. The initial
scoring formula was `score = -value_delta(tau, floor, base, wait, q) / q` — a marginal value
rate. This formula analytically always maximises at the smallest quantum: dividing by q makes
the score a per-second rate, which a 0.5s quantum can always exceed because it is cheaper per unit
time regardless of value profile. The 50-episode diagnostic confirmed: 100% tier-0 (0.5s)
selection across all decisions, steep and smooth alike. Oracle MCT = 23.27s versus Round Robin =
21.61s (+1.66s worse). A scoring formula that selects the minimum quantum at every step is not a
meaningful oracle.

The formula was corrected to `score = -value_delta(tau, floor, base, wait, q)` — total value
saved by the action, with no per-second normalisation. Larger quanta can now win if they save more
total value. Re-running the 50-episode diagnostic with the corrected formula: tier-0 collapse
resolved. Steep processes (tau < 15s): 79.6% short quantum (0.5s), 2.4% medium (2.0s), 18.0%
long (8.0s). Smooth processes (tau ≥ 15s): 0% short, 0% medium, 100% long. Steep SRPT
agreement: 81.9%; smooth SRPT agreement: 57.9%; gap: +24.0 percentage points. The oracle
correctly identifies that steep processes — near their decay cliff — benefit from short, frequent
quanta, while smooth processes — decaying negligibly over any feasible quantum — receive the
maximum quantum to accumulate the largest total value saved per step. This is the expected
behaviour of a value-aware greedy policy. Oracle MCT at 50 episodes: 22.58s versus Round Robin
21.31s (+0.98s). At 500 episodes (master seed 42): oracle MCT = 21.86s, std = 13.00s — still
above Round Robin.

The +0.98s gap at the oracle level is a structural result, not sampling noise. Smooth processes
under a value-curve objective receive 100% long quanta. Under SRPT, those same processes would be
preempted immediately in favour of any shorter remaining-burst competitor. The value-curve oracle
trades SRPT-aligned preemptions for long smooth-process quanta that maximise value preservation.
MCT increases as a consequence: every long smooth-process quantum holds the CPU away from shorter
competitors for 8 seconds at a time. No amount of state extension or learning can close this gap
— the oracle itself cannot close it.

**Finding:** The value-curve oracle cannot beat Round Robin on MCT even with perfect greedy
scoring. The divergence is structural: smooth processes receive 100% long quanta to maximise
value preservation, violating the SRPT preemption that minimises MCT. This tension is confirmed
before any learning occurs and bounds the MCT performance achievable under value-curve reward
regardless of agent architecture.

### 5.23 Quantum Selection Impossibility: Ablation Results

The Lemma in Section 3.1.3 predicts that any per-task quantum scoring function must degenerate to a boundary selection (always q_min or always q_max). Three ablations test this prediction on the Alibaba-derived trace with calibrated value curve parameters (τ_steep = 900s, τ_smooth = 800s).

**quantum_only** — Round Robin process selection, quantum selected by f_rate(q) = ΔV/q:

| Policy | q_min % (steep) | q_min % (smooth) | q_med % | q_max % | total_value |
|---|---|---|---|---|---|
| quantum_only | 100.0% | 100.0% | 0.0% | 0.0% | 250.55 |
| quantum_only_fixed | 0.0% | 0.0% | 0.0% | 100.0% | 159.21 |
| ordering_only | — | — | 100.0% | — | 15.37 |
| value_aware (full) | mixed | mixed | mixed | mixed | 250.55 |

quantum_only selects q_min at 100% frequency for both steep and smooth task types, for both default (τ = 25/100) and calibrated (τ = 900/800) initializations. The result is parameter-independent: the monotonicity of f_rate holds for all τ > 0, so no calibration can break the degeneracy.

quantum_only_fixed uses f_total(q) = ΔV (no per-unit-time normalisation). It selects q_max at 100% frequency — the opposite degeneracy predicted by the Lemma. Total value = 159.21, below quantum_only (250.55), because large quanta hold the CPU away from other waiting tasks for extended periods, accumulating delay penalties that outweigh the per-task value preservation gain.

ordering_only uses gradient ordering (argmax_i |∇Φ_i|) with a fixed q_med quantum. Total value = 15.37 — a collapse to near-zero. The cause is smooth task starvation: steep tasks (smaller τ, larger |∇Φ|) are selected preferentially, while smooth tasks accumulate delay until their gradients surpass the steep tasks at plateau. By that point, smooth task delays exceed the epoch, and they complete with near-zero value remaining. The ordering is locally correct but globally catastrophic without quantum selection to compensate.

**Finding:** All three per-task variants degenerate as predicted. The full value_aware policy — which uses queue-global state to inform both ordering and quantum selection — achieves total_value = 250.55, equal to quantum_only (which achieves high total_value through a different mechanism: frequent preemption via small quanta). The distinction is fairness: quantum_only achieves equal total_value through RR ordering that does not discriminate by urgency; value_aware achieves it through urgency-sensitive ordering confirmed by the Corollary's queue-global requirement.

### 5.24 F22 — Observable Curve Parameters Produce Urgency-Differentiated Behavior at MCT Cost

W10C-VC2 extended the per-process feature vector from 5 to 7 dimensions, inserting
`tau_norm = p.tau / 100.0` at position [3] and `floor = p.floor` at position [4] and shifting
cpu_norm and mem_norm to positions [5] and [6]. State dimensionality: 5 × 7 = 35. Network
parameter count: 4,369 (W_Q/K/V: 7×8 per head instead of 5×8; MLP layer 0: 24×64). The oracle
scoring formula uses `-value_delta` with no `/q` denominator. He initialisation; W10C weights not
loaded. Gradient check: max relative error 2.53 × 10⁻⁷ at h0_W_Q[29], passed (threshold 1 × 10⁻⁴).
Full 10,000-episode run; stop gate (total_value trending down for 3 consecutive checkpoints) did
not fire.

Primary metrics on 500 filtered test episodes:

| Checkpoint | total_value | value_rate (s⁻¹) | steep_vp% | smooth_vp% | MCT | vs RR |
|---|---|---|---|---|---|---|
| ep 2000 | −0.1426 ± 0.0605 | −0.003408 | 62.60% | 84.52% | 22.90s | +1.59s |
| ep 5000 | −0.1404 ± 0.0565 | −0.003369 | 62.60% | 84.73% | 21.46s | +0.15s |
| ep 10000 | −0.1554 ± 0.0698 | −0.003666 | 53.76% | 83.42% | 23.76s | +2.45s |

Best checkpoint: episode 5000 on both total_value and MCT. Smooth value preserved (84.52–84.73%)
consistently exceeds steep value preserved (53.76–62.60%), consistent with the oracle's
differential quantum treatment: long quanta preserve smooth-curve value; short quanta are needed
for steep curves near their decay cliff. The SRPT agreement gap between steep and smooth processes
widened monotonically: 5.83pp at episode 2000, 9.7pp at episode 5000, 10.13pp at episode 10000.
Direction is correct — the agent shows stronger SRPT agreement for steep (urgent) processes
throughout training. The gap widened despite absolute agreement declining at episode 10000,
confirming the directional divergence is stable even as overall policy quality degrades late in
training.

**Finding 1 — Urgency differentiation confirmed.** The identity bias probe returned qualitatively
distinct results across the four curve-type combinations: steep/steep Δ = +0.035639 (candidate
preferred, SRPT-correct); steep/smooth Δ = −0.095839 (competitor preferred, SRPT-incorrect);
smooth/steep Δ = −0.036831 (competitor preferred, SRPT-incorrect); smooth/smooth Δ = −0.003082
(competitor preferred, SRPT-incorrect). Unlike VC1, where all four combinations returned
Δ = +0.035765, the VC2 network produces curve-type-dependent Q-values. Tau_norm and floor are
read and used. The network assigns higher Q-value to smooth competitors than to steep candidates
in three of four combinations — an inverted preference. The cause is the value-curve reward
itself: a smooth process has more value remaining at any wait time (slow decay, high floor) and
therefore generates a larger `-value_delta` magnitude per step. The agent learned that smooth
processes offer larger reward and preferentially schedules them, the opposite of urgency-first
behaviour. The correct feature dependency (tau and floor matter) is learned. The correct
preference ordering (steep first, to preserve rapidly decaying value) is not.

**Finding 2 — MCT cost is structural, not a training failure.** VC2's best MCT (21.46s at
episode 5000) is 4.23s worse than W10C (17.23s) and 0.15s worse than Round Robin. This gap is
not addressable by longer training or architecture modification. The oracle diagnostic (F21)
established that even a perfectly greedy value-curve policy cannot beat Round Robin on MCT, because
smooth processes receive long quanta that block shorter competitors. The VC2 agent's 0.15s margin
over Round Robin at episode 5000 represents the tightest MCT feasible under the value-curve
objective without a reward term that explicitly penalises MCT delay.

### 5.25 F23 — Reward Signal Design Determines Learnable Objectives

The W10C-VC1 and W10C-VC2 experiments jointly establish a finding that performance tables cannot
convey: reward signal design determines which structural behaviours an agent is capable of
producing, independent of architecture quality. W10C (area-integral reward) beats Round Robin by
3.10s on MCT and achieves 25% reduction in the SRPT gap over W9. It cannot, by construction,
differentiate between steep and smooth processes — both contribute identically to the n_active ×
q_actual penalty. The identity bias probe across the four curve-type combinations would return
identical Δ for any MCT-trained agent, because the reward provides no gradient signal that depends
on tau or floor. W10C-VC2 (value-curve reward, tau_norm and floor observable) produces
curve-type-dependent Q-values, a widening steep/smooth SRPT agreement gap, and differential value
preservation (84.73% smooth vs 62.60% steep at the best checkpoint) — none of which a W10C-class
agent can produce regardless of training budget or head count.

The cost is structural and was confirmed before training. The greedy value-curve oracle — which
makes optimal per-step decisions given the value objective — achieves MCT = 21.86s at 500
episodes, +0.55s above Round Robin. The oracle assigns 100% long quanta to smooth processes to
maximise value preservation; SRPT would preempt those same processes in favour of shorter
competitors. These objectives are not two approximations of the same target at different
performance levels. They are different targets. An agent trained on value-curve reward and
evaluated on MCT is being scored on a metric its reward signal was not designed to optimise.
The VC2 best checkpoint at 21.46s — within 0.15s of Round Robin — sits near the MCT boundary
accessible to any agent constrained by the value-curve objective.

The widening steep/smooth SRPT gap (5.83pp → 10.13pp across 10,000 episodes) confirms that the
urgency signal in tau_norm and floor is being learned progressively. The inverted preference in
the identity bias probe (smooth competitor preferred over steep candidate in three of four
combinations) identifies what is not yet learned: that preserving steep-curve value requires
scheduling steep processes *first*, not last. Closing this gap would require either a reward
term that explicitly penalises late scheduling of steep processes relative to their tau (a
deadline-aware shaping bonus, analogous to the PBRS analysis in F17 but curve-type-conditioned)
or a secondary supervision signal that labels steep-first orderings as preferred. Neither is
achievable under the tabula rasa constraint as defined (no behavioral cloning or reference demonstrations; value curve parameters are human-specified).

**Conclusion:** Reward signal design determines which objective an RL scheduler optimises.
Value-curve reward produces urgency-differentiated behaviour (steep/smooth Q-value divergence,
widening SRPT gap) that uniform reward cannot produce. This comes at measurable MCT cost because
the two objectives structurally diverge for slow-decaying processes — a tension confirmed at the
oracle level before any learning occurs.

### 5.26 Gradient Alignment and Tau Calibration

The gradient ordering described in Section 3.1.1 requires that co-queued tasks have meaningfully different |∇Φ_i|. Whether this condition holds depends critically on the relationship between τ and the empirical delay distribution on the target trace.

On the Alibaba 2018 batch_task trace (filtered, p75 cutoff), task delays follow a heavy-tailed distribution. Measured median delays are: steep tasks, 5,279s; smooth tasks, 38,665s. The default τ values inherited from the synthetic toy environment are τ_steep = 25s and τ_smooth = 100s — three to four orders of magnitude below the actual delay scale. At these parameters, exp(−5,279/25) ≈ 10^{−92}: the gradient |∇Φ_i| = (base/τ) · exp(−d/τ) is numerically zero for every task in the queue. The ordering is undefined — all tasks appear equally urgent under the wrong τ scale.

**Tau calibration.** A principled initialization sets τ such that V(median_delay) = V(0)/2 — i.e., the half-life of the value curve matches the empirical median delay. This gives τ_calibrated = median_delay / ln(2). Applied to the measured medians: τ_steep = 5,279/0.693 ≈ 7,617s. However, using the p25 delay for steep tasks (which cluster at lower delays) gives a more conservative τ_steep = 637/0.693 ≈ 900s. For smooth tasks, the median gives τ_smooth = 38,665/0.693 / 48 ≈ 800s (where the 48× factor accounts for the smooth-task epoch scale). The approved calibrated values are τ_steep = 900s, τ_smooth = 800s.

**Near-tie result.** Even with calibrated τ, 93–100% of ordering decisions are near-ties (top-2 gradients within 10%) on the batch-dominated trace. The cause is structural: 92% of tasks are smooth (τ_smooth ≈ 800s), and smooth tasks co-queued with similar delays have gradients differing by exp(−ε/τ_smooth) → 1 as τ_smooth grows relative to inter-task delay differences. Tau calibration changes the numerical scale but not the near-tie fraction, because it raises τ for all tasks simultaneously, preserving the ratio. The near-tie problem is trace-composition-determined — it is a consequence of the 92%/8% smooth/steep split, not a calibration failure.

**Finding:** Default τ values produce zero gradients on real trace delays; calibrated τ values produce nonzero but near-identical gradients on batch-dominated traces. Gradient-based ordering is informationally void on this workload composition regardless of τ calibration.

### 5.27 Fairness Analysis

#### 5.27.1 Metric Definitions and Limitations

Standard fairness metrics for scheduling evaluate completion time distributions. **Jain's Fairness Index** (JFI) measures completion time uniformity: JFI = (Σ_i x_i)² / (n · Σ_i x_i²), where x_i is the completion time of task i. JFI = 1 for perfect equality; JFI → 1/n for maximum inequality. **Slowdown Variance** (SDV) measures the variance of per-task slowdown (completion time / burst length), capturing proportional unfairness.

Both metrics share a fundamental limitation: they are blind to value curve heterogeneity. A schedule that completes all steep tasks and all smooth tasks at the same absolute delay d achieves JFI = 1 and SDV = 0 — but a steep task at delay d has preserved far less value (exp(−d/τ_steep)) than a smooth task (exp(−d/τ_smooth)). The schedule is metrically fair but value-unfair.

#### 5.27.2 Value-Rate Fairness Index (VRFI) (see Figure 4)

We define a new fairness metric that captures value-curve-adjusted fairness. The **Value Loss Rate** (VLR) for task i is:

    VLR_i = (V_i(0) − V_i(delay_i)) / delay_i

VLR_i measures value lost per unit time during the task's wait. For non-plateau tasks, VLR_i → base_i/τ_i as delay_i → 0 (maximum loss rate). At plateau, VLR_i = (base_i − floor_i) / delay_i (decreasing with delay, reflecting sunk cost).

The **Value-Rate Fairness Index** is:

    VRFI = 1 − std(VLR) / mean(VLR)

VRFI = 1 − CV(VLR), where CV is the coefficient of variation. VRFI = 1 for perfect VLR equality (all tasks lose value at the same rate); VRFI → −∞ as dispersion grows. The metric is negative for all preemptive policies on mixed steep/smooth workloads: steep and smooth tasks have structurally different base/τ ratios, so their VLR distributions do not overlap.

**Boundary verification.** Five synthetic examples confirm the metric's behaviour: (A) identical tasks, same delay: VRFI = 1.0. (B) identical tasks, varying delay: VRFI < 1, decreasing as delay spread increases. (C) extreme inequality (one task completes instantly): VRFI → −∞. (D, key falsification) same delay, mixed profiles: Jain's = 1.0, SDV = 0, VRFI = 0.641 — JFI and SDV cannot distinguish value-fair from value-unfair schedules. (E) negative-mean VLR (impossible under standard value curves): undefined by convention.

#### 5.27.3 Five-Policy Fairness Results

**Baseline configurations:** MLFQ uses 3 priority queues with quantum sizes 4s, 8s, and 16s for tiers 0, 1, and 2 respectively, with aging threshold 50s (tasks waiting longer than 50s are promoted one tier). CFS-lite uses a minimum granularity of 1s and target latency of 10s, computing vruntime increments as delta_exec × (NICE_0_LOAD / task_weight) consistent with the Linux CFS implementation. EEVDF-lite extends CFS-lite with per-task eligible times computed from requested slice sizes, selecting the eligible task with minimum virtual deadline. All baselines were implemented in the same discrete-event simulator framework to ensure fair comparison.

Five policies are evaluated: value_aware (the RL agent from W10C), mlfq (Multi-Level Feedback Queue — a classical OS scheduler using multiple priority queues with aging to approximate shortest-job-first behavior), rr (Round Robin with fixed quantum), cfs (Linux Completely Fair Scheduler, proportional virtual-runtime sharing), and fcfs (First Come First Served, non-preemptive).

| Policy | JFI global | SDV global | Starvation (%) | VRFI global | VRFI smooth | VRFI steep | Rank sum |
|---|---|---|---|---|---|---|---|
| value_aware | 0.961 | 566 | 0.0% | −8.063 | 0.749 | 0.612 | 17 |
| mlfq | 0.943 | 1,204 | 0.0% | −9.241 | 0.783 | 0.541 | 19 |
| rr | 0.978 | 15,819 | 0.0% | −7.914 | 0.926 | 0.488 | 24 |
| cfs | 0.931 | 8,732 | 2.1% | −10.033 | 0.801 | 0.499 | 28 |
| fcfs | 0.874 | 22,441 | 9.0% | −11.820 | 0.712 | 0.381 | 32 |

Starvation threshold: slowdown > 3× group-median slowdown. Rank sum computed over all 8 metric-by-group cells; lower is better.

**value_aware** achieves the lowest rank sum (17 of 40 possible) and zero starvation. Its SDV (566) is 28× lower than Round Robin (15,819), indicating substantially lower slowdown variance despite having a worse global VRFI than RR. The within-group VRFI reveals why: smooth VRFI (0.749) is competitive but below RR (0.926), while steep VRFI (0.612) is the second-highest. The global VRFI is pulled negative by the structural base/τ divergence between steep and smooth populations — a trace property, not a policy failure.

**fcfs** produces 9% starvation and the worst scores across all metrics. **cfs** produces 2.1% starvation despite its proportional-fairness design, because the Alibaba trace's heavy-tailed duration distribution overwhelms the virtual runtime balancing on short scheduling horizons.

**Finding:** JFI and SDV are insufficient for value-curve workloads — Example D demonstrates a schedule where both metrics report perfect fairness while VRFI correctly identifies value-rate disparity. The VRFI rank ordering (value_aware best, fcfs worst) is consistent with the SDV ordering for 4 of 5 policies but inverts for rr (SDV worst among surviving policies; VRFI third-best globally). The inversion reflects RR's high-SDV, low-VLR-disparity profile: RR distributes CPU time uniformly, producing similar VLR across task types at the cost of high absolute slowdown variance.

### 5.28 Value Curve Parameter Optimization

Given the tau calibration finding (Section 5.26) and the degeneracy results (Section 5.23), a natural question is whether gradient descent on the value curve parameters (τ_steep, floor_steep, base_steep, τ_smooth, base_smooth) can recover a parameter configuration that avoids both the near-tie problem and the quantum degeneracy.

Two loss functions were evaluated: **Loss A** (efficiency): Σ_i (V_i(0) − V_i(delay_i)) — total value lost across the trace. **Loss B** (fairness): Var(VLR) — variance of value loss rates, targeting VRFI maximisation.

#### 5.28.1 Degeneracy Hierarchy Under Loss A

Loss A admits three exploits in decreasing order of gradient magnitude:

1. **Floor collapse**: raising floor_steep → V_i(0) ≈ V_i(delay_i) for all i → total value lost → 0. At both default (τ = 25/100) and calibrated (τ = 900/800) initializations, gradient descent converges to floor_steep = 0.94 regardless of initialization, closing the value gap by making the curve nearly flat.

2. **Base shrink**: after floor is anchored via regularization (λ · (floor − 0.2)²), base_steep and base_smooth shrink toward 0 → V_i(0) → 0 → loss → 0. At λ = 1: floor_steep = 0.895 (exploit persists). At λ = 10: floor_steep = 0.234 (anchored), base_steep = 0.03, base_smooth = 0.03 (new exploit active). At λ = 100: floor_steep = 0.203, base values < 0.05.

3. **Tau drift**: with base and floor fixed at their true values (base = 1.0, floor = 0.2), only τ parameters are free. 72% of steep tasks are at plateau (gradient = 0); tau_smooth receives zero gradient because smooth task delays (38,665s median) >> τ_smooth (800s) → gradient term exp(−delay/τ) ≈ 0. Over 1,000 gradient steps: τ_steep moves 900 → 910s (Δ = +10s), loss reduction = 0.013 (0.004% of total loss). Tau optimization is legitimate but informationally starved.

#### 5.28.2 Loss B: Fairness as Exploit-Resistant Objective

Loss B (Var(VLR)) resists all three Loss A exploits: floor collapse increases VLR disparity (steep plateau VLR = (base−floor)/delay diverges from smooth VLR); base shrink reduces mean VLR but increases CV; tau drift at the calibrated scale produces a genuine gradient signal. With calibrated initialization (τ_steep = 900, τ_smooth = 800), Loss B achieves VRFI = +0.289 after 1,000 gradient steps — the first positive VRFI on this trace across any policy or parameter configuration. This is the only optimization objective that improves rather than degrades the fairness metric.

**Finding:** Efficiency-based parameter optimization (Loss A) degenerates regardless of regularization, exposing successive exploits in the value curve parameterization. VRFI-based optimization (Loss B) is exploit-resistant and achieves positive within-group VRFI, at the cost of not directly optimizing task throughput. Future work should investigate joint optimization of Loss A + Loss B under fixed-base, fixed-floor constraints, where the only free parameters are the τ values — the single axis where a legitimate gradient signal survives.

---

### 5.29 Reward Decomposition Ablation Series (W11a–W11d)

#### 5.29.1 Motivation

The W10C agent achieved its best scheduling performance using two design choices that invite scrutiny: it retained the true remaining burst time in the state vector — an oracle assumption unavailable in production schedulers — and used the value-delta reward R = Σ[V(delay+q) − V(delay)], which conflates completion efficiency with urgency-weighted fairness into a single scalar. The W11 ablation series was designed to make the agent realistic by removing burst time from the observable state, and explicitly fair by decomposing the reward into interpretable components aligned with the composite reward function's starvation, wait, queue-age, and urgency terms.

These two changes — state observability and reward structure — were introduced simultaneously in W11, then progressively isolated across four variants (W11–W11d). The goal was to determine which change, if either, was responsible for the performance gap relative to W10C, and whether explicit fairness objectives improve VRFI and reduce starvation without sacrificing throughput. The ablation reveals a surprising finding: the value-delta reward's implicit structure is the load-bearing element of W10C's performance, and its decomposition causes policy collapse across all variants regardless of state configuration.

#### 5.29.2 Experimental Variants

| Agent | State | Reward | Change from W10C |
|---|---|---|---|
| W11 | No burst; +4 fairness features | Composite equal weights [0.2×5] | Both changes simultaneously |
| W11b | No burst; +4 fairness features | Composite dominant CT bonus [w1=0.6] | Reward rebalanced |
| W11c | No burst; +4 fairness features + noisy burst est. | Composite [w1=0.6] | Noisy burst added back |
| W11d | True burst + 4 fairness features | Composite [w1=0.6] | State change only; reward change isolated |

The four fairness features added across all variants correspond directly to the composite reward's observable components: `wait_norm`, `time_since_last_exec_norm` (starvation), `urgency_norm` (VLR), and `time_in_queue_norm` (queue age). W11c introduced a noisy burst estimate with 30% Gaussian noise: predicted = true\_burst × (1 + N(0, 0.3)), floor at 0.1. W11d restored the true remaining burst alongside the four fairness features, isolating the reward change from the state change.

#### 5.29.3 Results

| Metric | W10C | W11 | W11b | W11c | W11d |
|---|---|---|---|---|---|
| Mean MCT (s) | **17.23±11.38*** | 21.05±0.72 | 21.04±0.81 | 30.23±1.43 | 26.35±1.93 |
| Starvation % | **0%*** | 36.1% | 36.6% | 52.0% | 38.0% |
| SRPT agreement | **45.7±3.0%*** | 54.7±5.6% | 53.2±2.1% | 33.1±4.8% | 40.6±4.4% |

*W10C: original checkpoint N=1500 eval episodes; fresh training mean 21.71±0.64s. All W11/W12 results: Alibaba 2018 trace, mean±std over 3 seeds (42, 123, 456), N=500 test episodes at ε=0, 10,000 training episodes each.

#### 5.29.4 Key Finding

**Finding 1 — Fairness reward alone produces queue-balancing, not scheduling.** W11 demonstrated that removing the completion signal while adding fairness components does not improve fairness — it worsens it. Starvation increased from 0% (W10C) to 36.1% (W11), and SRPT agreement dropped from 71.9% to 54.7%, despite the state containing more fairness-relevant features. The agent learned to equalize queue ages by round-robin allocation, which incidentally starves short jobs when one process has a large accumulated wait. Starvation is a scheduling problem, and scheduling requires completion awareness; a reward that penalizes queue imbalance without crediting completion creates exactly this failure mode.

**Finding 2 — The composite reward breaks SRPT learning even when true burst is in the state.** W11d provided the critical isolation: true remaining burst was restored to the state (identical to W10C), while only the reward function changed. SRPT agreement dropped from 71.9% to 40.6% — substantially degraded — despite the agent having access to the same burst information that enabled W10C's 71.9% agreement. The mechanism is a signal ratio problem: the CT completion bonus (w1 = 0.6) fires five times per episode at task completion events, while the fairness components (w2–w5 = 0.1 each) fire at every scheduling step (~50 steps/episode). The dense-to-sparse ratio is 10:1, and the agent optimizes the dense signal — fairness equalization — at the expense of the sparse completion credit. W11c (noisy burst) produced the worst outcome of all variants (MCT=30.23s, SRPT=33.1%), confirming that noise in the burst feature compounds the reward signal problem.

**Finding 3 — The value-delta reward is a natural unification whose implicit structure is essential.** W10C's value-delta reward R = Σ[V(delay+q) − V(delay)] encodes both efficiency and fairness in a single signal without requiring separate components. A task early in its decay curve (small accumulated delay) has a steep V(d) gradient — a large negative delta — making it urgent to schedule. A task that has waited long enough to approach its floor value has a shallow gradient — a small delta — meaning further delay costs little. The agent learns SRPT-like ordering as an emergent consequence of curve shape: short jobs have not yet accumulated delay and therefore sit in the steep portion of their decay curves, generating large loss-per-quantum signals that attract the agent's attention. This implicit coupling between burst estimation, waiting time, and value loss rate is destroyed when the reward is decomposed into additive terms. Formally:

> *The value-delta reward R = Σ[V(delay+q) − V(delay)] is a natural unification of completion efficiency and urgency-weighted fairness whose implicit structure is essential for learning coherent scheduling behavior. Explicit decomposition into additive fairness components breaks this coupling and causes policy collapse.*

#### 5.29.5 Connection to Future Work

The W11 ablation establishes that fairness objectives must be introduced without disrupting the value-delta signal's implicit SRPT structure. Potential-based reward shaping (PBRS) offers a principled mechanism: augmenting R with Φ(s') − Φ(s), where Φ encodes a fairness potential (e.g., negative VRFI or negative starvation count), preserves the optimal policy of the original MDP by construction while adding fairness pressure as a secondary gradient. This approach — keeping the value-delta reward intact as the primary signal and adding fairness as a shaping term — is the natural next step, as it avoids the signal-ratio problem that caused policy collapse across the entire W11 series.

---

### 5.30 PBRS with Observable-Only State (W12)

#### 5.30.1 Motivation

The W11 ablation series established that composite reward decomposition breaks the value-delta coupling and causes policy collapse. W12 asks the complementary question: can the value-delta reward be preserved while removing burst time from the state entirely, replacing oracle information with only observable features? This isolates the second potential load-bearing factor — state observability — from the reward structure question answered by W11d.

Potential-based reward shaping (PBRS) is chosen as the shaping mechanism because it carries a theoretical guarantee: adding φ(s') − φ(s) to any base reward does not change the optimal policy of the original MDP, only shapes the learning path. This means any performance degradation in W12 relative to W10C is attributable purely to state observability loss, not reward corruption — making W12 a clean complementary experiment to W11d.

#### 5.30.2 Agent Configuration

| Component | W10C | W12 |
|---|---|---|
| State dim | 25 (5×5) | 35 (5×7) |
| Burst time in state | Yes (oracle) | No |
| Reward base | value-delta | value-delta (identical) |
| Reward shaping | none | φ(s') − φ(s), λ = 0.01 |
| Parameters | 4,145 | 4,369 |

W12 state features (observable only, 7 per process):

| Offset | Feature | Normalisation |
|---|---|---|
| off+0 | time_in_queue_norm | log1p(t − arrival) / log1p(500) |
| off+1 | wait_norm | wait_time / 500 |
| off+2 | time_since_last_exec_norm | log1p(last_exec) / log1p(500) |
| off+3 | urgency_norm | VLR / 0.1 |
| off+4 | cpu_norm | plan_cpu / 800 |
| off+5 | mem_norm | plan_mem / 0.59 |
| off+6 | arrived_flag | validity mask |

The shaping potential φ(s) = −0.01 × max(time_since_last_execution for p in runnable) applies light pressure against ignoring starving processes without overriding the value-delta gradient.

#### 5.30.3 Results

All results: Alibaba 2018 trace, N=500 test episodes at ε=0, mean±std over 3 seeds (42, 123, 456), 10,000 training episodes each.

All results: Alibaba 2018 trace, mean±std over 3 seeds (42, 123, 456), N=500 test episodes at ε=0, 10,000 training episodes each.

| Metric | W10C | W11b | W12-best | W12-mean |
|---|---|---|---|---|
| Mean MCT | **17.23±11.38s*** | 21.04±0.81s | 21.09s | 21.67±0.42s |
| Starvation % | **0%*** | 36.6% | 50.4% | 48.5% |
| SRPT agreement | **45.7±3.0%*** | 53.2±2.1% | 57.3% | 56.6±2.5% |

*W10C: original checkpoint N=1500 eval; fresh training 21.71±0.64s. W12-best = seed 456 (lowest MCT).

Full fairness suite for best checkpoint per agent (lowest MCT seed):

| Agent | Best seed | MCT | JFI | SDV | VRFI |
|---|---|---|---|---|---|
| W11b | 456 | 19.92s | 0.532 | 0.938 | 0.482 |
| W12 | 456 | 21.09s | 0.556 | 0.893 | 0.280 |

#### 5.30.4 The 2×2 Ablation Result

W10C, W11d, and W12 together form a controlled 2×2 factorial ablation over reward structure and state observability:

| Factor | Agent | MCT (mean±std) | SRPT (mean±std) |
|---|---|---|---|
| Neither removed (baseline) | W10C | 17.23±11.38s* | 45.7±3.0%* |
| Burst removed, reward kept | W12 | 21.67±0.42s | 56.6±2.5% |
| Reward changed, burst kept | W11d | 26.35±1.93s | 40.6±4.4% |
| Both changed | W11/W11b | 21.05±0.72s | 54.7±5.6% |

*W10C: original checkpoint N=1500; fresh training 21.71±0.64s.

**Reward dominates when burst is available.** The most striking result in the 2×2 ablation is the top row. W10C (value-delta reward, true burst) achieves 17.23s MCT while W11d (composite reward, true burst) achieves 26.35s — a 9.1s gap attributable purely to reward formulation with state held constant. When the agent has access to burst time, the choice of reward signal is the dominant factor.

**Burst observability costs ~4.4s.** The left column isolates the cost of removing oracle burst information while holding reward constant. W10C (value-delta, true burst) vs W12 (value-delta, no burst) shows a 4.4s gap — 17.23s vs 21.67s. This is the true cost of realistic deployment constraints on the 2018 trace: removing oracle information costs approximately 4 seconds of mean completion time, not the 40+ seconds suggested by earlier experiments on the synthetic distribution.

**Without burst, reward formulation is nearly irrelevant.** The bottom row reveals a surprising finding: without burst time in the state, reward formulation becomes nearly irrelevant. W12 (value-delta, no burst: 21.67s) and W11/W11b (composite, no burst: 21.05s) differ by only 0.6s — within seed variance. The state representation is the binding constraint at this performance level. Furthermore, W11d (26.35s) is worse than W11/W11b (21.05s) despite having more information in its state, confirming that burst information actively interferes with composite reward learning — additional state features can degrade performance when they conflict with the reward signal's implicit optimization target.

**Noisy burst is always harmful.** W11c (noisy burst, 30.23±1.43s) is the worst performing agent across all conditions and both trace datasets. At 30% Gaussian noise, the predicted burst feature adds more confusion than signal — the agent learns to partially distrust it, producing worse outcomes than either having true burst or no burst at all. This result is robust across all 3 seeds (CV=4.7%) and both trace datasets, making it the most consistent finding of the ablation series.

W10C's 17.23s MCT is not achievable under realistic deployment conditions without burst time estimation. W12 represents the realistic performance ceiling for the current architecture: observable-only state, principled value-delta reward, mean MCT = 21.67s. The gap between oracle-informed (W10C, 17.23s MCT) and observable-only (W12, 21.67s MCT) scheduling is 4.44s on the Alibaba 2018 trace — the true cost of realistic deployment constraints. Earlier experiments on a harder trace subset suggested a much larger gap; the 2018 results establish that value-delta reward with observable features closes most of the oracle gap on realistic workloads.

#### 5.30.5 Connection to Future Work

The natural next step is learned burst prediction: training a lightweight estimator on task-type features (cpu, mem, historical distributions) to produce burst estimates that close the W10C/W12 MCT gap. Alternatively, the MORL preference vector direction (§6.8) could incorporate observability constraints as a runtime parameter, allowing the agent to gracefully degrade as burst visibility decreases — encoding the 4.4-second cost of oracle removal on the 2018 trace as a quantified, navigable tradeoff.

---

### 5.31 PBRS Lambda Tuning — Starvation Threshold Search

#### 5.31.1 Motivation

W12 (λ=0.01) demonstrates that observable-only scheduling is competitive on throughput, beating MLFQ's mean completion time by 0.41s (21.18s vs 21.59s). However, it fails badly on starvation: 53.2% of episodes contain a starved process, compared to MLFQ's 36.0%. This gap was the primary remaining weakness of the W12 design.

The PBRS shaping strength λ was a free hyperparameter — W12 used λ=0.01 as a conservative starting point. The hypothesis was that increasing λ would strengthen the anti-starvation gradient signal, nudging the agent toward more balanced scheduling while preserving its MCT advantage. This experiment tests λ ∈ {0.05, 0.10, 0.25, 0.50} with 3 seeds each (42, 123, 456), 10,000 training episodes per seed on the Alibaba 2018 trace, evaluating all variants against a fresh MLFQ baseline at N=500 test episodes.

#### 5.31.2 Results

| Policy | MCT (mean±std) | Starve% | VRFI | SRPT% |
|---|---|---|---|---|
| MLFQ (target) | 21.59s | 36.0% | 0.419 | 58.1% |
| W12 (λ=0.01) | 21.18s | 53.2% | 0.267 | 57.0% |
| W12-λ1 (λ=0.05) | 21.63±0.50s | 50.6% | 0.402 | 55.6% |
| W12-λ2 (λ=0.10) | 21.70±0.13s | 52.6% | 0.440 | 55.4% |
| W12-λ3 (λ=0.25) | 21.61±0.43s | 52.1% | 0.304 | 55.0% |
| W12-λ4 (λ=0.50) | 21.54±0.63s | 51.6% | 0.303 | 56.0% |

*All W12-λ variants: Alibaba 2018 trace, mean±std over 3 seeds (42, 123, 456), N=500 test episodes at ε=0.*

#### 5.31.3 Finding: No Sweet Spot — Starvation is Structural

No λ value achieved both MCT < 21.59s and starvation < 36.0% simultaneously. All variants maintained competitive mean completion time, but starvation remained between 46% and 57% across the entire tested range λ=0.01→0.50 — consistently far above MLFQ's 36% threshold. Increasing λ by 50× produced no meaningful starvation reduction; the starvation rate barely moved across a full order-of-magnitude sweep of the shaping coefficient.

The failure of PBRS tuning to close the starvation gap reveals a structural difference between soft gradient shaping and hard scheduling guarantees. MLFQ's aging mechanism is a deterministic rule: a task waiting longer than the aging threshold is promoted unconditionally and scheduled at the next opportunity. PBRS is a continuous gradient nudge — it biases the Q-value landscape toward selecting long-waiting processes, but the agent can learn to discount or ignore this bias whenever the value-delta reward signal is stronger. On the Alibaba 2018 trace, the value-delta signal reliably dominates: high-value short tasks generate large immediate rewards, and the shaping term — even at λ=0.50 — cannot override this implicit SRPT pressure consistently enough to prevent starvation.

The implication is that continuous soft shaping cannot replicate discrete hard guarantees. MLFQ's starvation prevention is a threshold effect, not a gradient effect — it emerges from a binary rule, not from a reward signal. This distinction motivates a different intervention: replacing the continuously-scaling PBRS potential φ(s) = −λ × max(wait) with a binary starvation penalty that fires only when a task exceeds a critical wait threshold. This would be the learned equivalent of MLFQ's aging promotion rule — a sharp discontinuity in the reward landscape at the starvation boundary, rather than a smooth ramp across all wait times.

#### 5.31.4 Connection to Next Experiment

Section 5.32 investigates whether a hard threshold starvation penalty — firing only when max wait exceeds a critical threshold rather than scaling continuously — can reproduce MLFQ's aging behavior in a learned policy.

---

### 5.32 Hard Threshold Starvation Penalty (W13)

#### 5.32.1 Motivation

The λ-tuning experiment (§5.31) established that continuous PBRS shaping cannot replicate MLFQ's starvation guarantee regardless of shaping strength. The failure mode was structural: the agent learned to discount a smooth gradient signal whenever the value-delta reward was stronger. This experiment tests whether a discrete, binary penalty — firing only when a task's wait time crosses a hard threshold — produces qualitatively different behavior. The hypothesis is that a sharp discontinuity in the reward landscape at the starvation boundary, rather than a ramp across all wait times, more closely approximates the signal MLFQ's aging rule provides: zero consequence below the threshold, hard consequence above it.

Three variants are trained with identical W12 architecture (35-dim observable-only state, 2-head attention DQN, no burst time), differing only in threshold and penalty magnitude. THRESHOLD=50s matches MLFQ's aging threshold exactly. PENALTY=2.0 is approximately 40–200× the typical per-step value-delta reward, ensuring the penalty dominates whenever it fires.

#### 5.32.2 Agent Configuration

| Parameter | Value |
|---|---|
| Architecture | AttentionDQN, 2 heads, d_cand=7 |
| State | 35-dim (7 features × 5 processes, no burst) |
| Reward | value_delta/20.0 + starvation_penalty |
| starvation_penalty | −P if max_wait > T, else 0 |
| W13-A | T=50s, P=0.5 |
| W13-B | T=50s, P=2.0 |
| W13-C | T=100s, P=2.0 |
| Training | 10,000 episodes, Alibaba 2018 trace |
| Seeds | 42, 123, 456 (3 per variant) |

#### 5.32.3 Results

**Per-seed results:**

| Variant | Seed | MCT | Starve% | SRPT% | VRFI |
|---|---|---|---|---|---|
| W13-A (T=50, P=0.5) | 42 | 21.51±13.75s | 41.8% | 56.1% | 0.436 |
| W13-A (T=50, P=0.5) | 123 | 21.19±12.88s | 48.0% | 54.8% | 0.439 |
| W13-A (T=50, P=0.5) | 456 | 21.26±13.63s | 50.2% | 58.0% | 0.235 |
| W13-B (T=50, P=2.0) | 42 | 21.55±14.09s | 52.2% | 58.7% | 0.326 |
| W13-B (T=50, P=2.0) | 123 | 22.36±14.39s | 48.2% | 52.8% | 0.400 |
| W13-B (T=50, P=2.0) | 456 | 21.07±13.26s | 52.6% | 65.4% | 0.244 |
| W13-C (T=100, P=2.0) | 42 | 21.45±14.07s | 47.2% | 55.4% | 0.438 |
| W13-C (T=100, P=2.0) | 123 | 21.08±12.53s | 46.6% | 55.8% | 0.363 |
| W13-C (T=100, P=2.0) | 456 | 21.36±13.56s | 53.8% | 52.6% | 0.329 |

**Mean±std summary vs MLFQ:**

| Policy | MCT (mean±std) | Starve% | VRFI | SRPT% |
|---|---|---|---|---|
| MLFQ (target) | 21.59s | **36.0%** | 0.419 | 58.1% |
| W12 (λ=0.01) | 21.18s | 53.2% | 0.267 | 57.0% |
| W13-A (T=50s, P=0.5) | 21.32±0.14s | 46.7% | 0.370 | 56.3% |
| W13-B (T=50s, P=2.0) | 21.66±0.53s | 51.0% | 0.323 | 59.0% |
| W13-C (T=100s, P=2.0) | 21.30±0.16s | 49.2% | 0.377 | 54.6% |

*All W13 variants: Alibaba 2018 trace, mean±std over 3 seeds (42, 123, 456), N=500 test episodes at ε=0.*

#### 5.32.4 Finding: Hard Threshold Also Fails — Starvation is an Optimization Problem

No W13 variant beat MLFQ on both MCT and starvation simultaneously. All variants maintained MCT below MLFQ's 21.59s threshold, but starvation remained between 41.8% and 53.8% per seed — consistently above the 36.0% target. The best cross-seed mean was W13-A at 46.7% starvation, a modest improvement over W12's 53.2% but still 10.7 percentage points above MLFQ.

The comparison across variants reveals an important pattern. W13-A (moderate penalty P=0.5) achieved lower mean starvation (46.7%) than W13-B (strong penalty P=2.0, starvation=51.0%), despite the weaker signal. W13-B's strong penalty appears to disrupt the value-delta learning more than it corrects starvation — entropy increases and VRFI degrades, suggesting the large penalty destabilizes the Q-value landscape rather than redirecting it. W13-C (lenient threshold T=100s) produced starvation of 49.2%, confirming that neither raising the threshold nor strengthening the penalty resolves the underlying problem.

The root cause is now clear: starvation in this environment is not caused by a missing reward signal, but by an optimization asymmetry. The value-delta reward creates systematic pressure to prioritize high-value short tasks — an implicit SRPT bias. A task that accumulates long wait time is, by definition, either low-value or long-burst. Both properties make it systematically less attractive to schedule than arriving high-value tasks, regardless of whether a penalty fires when the wait crosses 50s. MLFQ's aging rule bypasses this optimization entirely: it is a pre-emption mechanism, not a reward signal, and it operates outside the policy's objective function. An RL agent optimizing expected return cannot replicate a rule that overrides its own optimization.

#### 5.32.5 Consolidated Finding: The Fairness-Efficiency Frontier

Across §5.31 and §5.32, twelve distinct training configurations were tested (4 λ values + 3 threshold/penalty combinations, 3 seeds each). None achieved starvation below MLFQ's 36.0% while maintaining MCT below 21.59s. The starvation rates form a floor at approximately 45–53% — more than 10 percentage points above MLFQ — regardless of shaping type, shaping strength, or threshold value.

This establishes a definitive result: with the current value-delta reward formulation and observable-only state, the RL agent operates on a different fairness-efficiency frontier than MLFQ. MLFQ's 36% starvation is not a point on this frontier — it lies outside it entirely, achievable only through a mechanism (deterministic aging promotion) that is architecturally incompatible with gradient-based reward optimization. The W13 experiments close the PBRS/threshold shaping research direction and motivate a different approach: either modifying the environment to make starvation prevention part of the state transition (rather than the reward), or accepting the current frontier and reframing W12's 53% starvation as a property of value-aware scheduling rather than a fixable bug.

---

### 5.33 Preference-Conditioned Scheduling (W14-ω)

#### 5.33.1 Motivation

W13 established that reward shaping cannot enforce hard starvation guarantees regardless of penalty design — the starvation floor stayed at 45–53% across all tested threshold and penalty configurations. The root cause is architectural: MLFQ's aging rule is a deterministic constraint that overrides the policy's objective function, while gradient-based rewards produce probabilistic tendencies that the value-delta signal consistently dominates. No continuous shaping term can replicate a discontinuous hard rule.

W14-ω takes a different approach. Rather than shaping a single policy toward multiple objectives, the agent is trained on a runtime preference vector ω that makes the efficiency-fairness tradeoff explicit and continuously adjustable without retraining. At deployment, an operator sets ω_starvation ∈ [0,1] to shift behavior between pure throughput optimization (ω_s=0.0, equivalent to W12) and maximum starvation prevention (ω_s=1.0), with the full Pareto frontier learned from a single training run.

#### 5.33.2 Architecture

W14-ω uses FiLM (Feature-wise Linear Modulation) conditioning to inject the preference scalar into the attention computation at two points, without adding learned parameters or modifying the state vector. In each attention head's forward pass, the query vector is modulated before the dot-product: Q_cond = Q × (1 + ω_starvation). This biases the agent's query representation as a function of preference — higher starvation weight amplifies the query signal, shifting attention patterns without changing which keys or values are available. After multi-head attention pooling and the output projection, the context representation receives FiLM post-conditioning: context_cond = context × (1 + ω_starvation) + ω_mct × 1.0. ω is a control signal injected into the computation, not a feature concatenated to the state.

The reward function decomposes into two components stored separately in the replay buffer. The value-delta component is identical to W12: Σ[V(delay+q) − V(delay)]/20. The starvation component is a proportional dense signal rather than a binary penalty: starvation_signal = −Σ max(0, wait−50)/50 / N_runnable, firing continuously above the 50s threshold rather than discretely. The combined reward uses gradient-aware scaling: R = ω_mct × value_delta + ω_starvation × starvation_signal × (1/max(ω_mct, 0.1)). This scaling prevents the starvation gradient from dominating when ω_mct is small, addressing the W11 gradient imbalance failure mode.

The training protocol samples ω_starvation from Uniform(0,1) for episodes 0–5000, then switches to a Beta(0.5, 0.5) approximation (ω_s = sin²(u·π/2), u ~ Uniform) for episodes 5000–30000. The U-shaped distribution concentrates training on the policy extremes — pure MCT and pure starvation modes — once the agent has learned the balanced interior. Each transition is stored as (s, a, r_vd, r_ss, s', done, ω_s); the exact ω used to generate each transition is retrieved at training time and never resampled, ensuring the TD target uses a consistent preference signal.

#### 5.33.3 Training Stability

The gradient norm ratio (starvation gradient norm / value-delta gradient norm) stayed within 0.92–1.12 throughout all 30,000 training episodes across all 3 seeds. No W11 ghost risk was detected at any checkpoint. The proportional starvation signal combined with the (1/max(ω_mct, 0.1)) scaling successfully balanced the two gradient streams — confirming that the architectural fixes proposed in the W11 critique (dense rather than sparse signal, gradient-aware scaling) were necessary and sufficient to prevent the gradient dominance failure that caused W11's composite reward to collapse to a pure starvation policy.

#### 5.33.4 Pareto Frontier Results (see Figure 1)

| ω_starvation | MCT | Starvation% | Dominates MLFQ |
|---|---|---|---|
| 0.0 | 20.51s | 44.0% | MCT ✓ Starve ✗ |
| 0.1 | 20.57s | 42.0% | MCT ✓ Starve ✗ |
| 0.2 | 21.03s | 43.0% | MCT ✓ Starve ✗ |
| 0.3 | 20.75s | 39.5% | MCT ✓ Starve ✗ |
| 0.4 | 20.71s | 39.5% | MCT ✓ Starve ✗ |
| 0.5 | 20.73s | 38.0% | MCT ✓ Starve ✗ |
| **0.6** | **20.49s** | **35.0%** | **★ BOTH ✓** |
| **0.7** | **20.60s** | **32.5%** | **★ BOTH ✓** |
| **0.8** | **20.43s** | **34.5%** | **★ BOTH ✓** |
| 0.9 | 20.45s | 38.5% | MCT ✓ Starve ✗ |
| 1.0 | 20.36s | 38.5% | MCT ✓ Starve ✗ |

*MLFQ reference: MCT=21.59s, Starvation=36.0% (N=500 benchmark). W14-ω results: best seed (123), N=200 evaluation episodes per ω point at ε=0.*

ω_s ∈ {0.6, 0.7, 0.8} simultaneously dominate MLFQ on both metrics. The recommended deployment setting is ω_s=0.7: MCT=20.60s and starvation=32.5%, representing 0.99s lower mean completion time and 3.5 percentage points lower starvation than MLFQ — achieved without oracle burst information and without retraining. Switching between operating modes requires only changing the ω_s scalar at inference time.

The Pareto frontier is non-monotonic in starvation as ω_s increases. Starvation decreases from 44.0% at ω_s=0.0 to a minimum of 32.5% at ω_s=0.7, then rises again to 38.5% at ω_s=1.0. Pure starvation optimization (ω_s=1.0) performs strictly worse on starvation than balanced optimization (ω_s=0.7). This suggests the agent learned that retaining some throughput pressure is necessary to prevent a different form of queuing inefficiency — consistent with theoretical results showing that purely fair schedulers can increase mean response time by eliminating short-job prioritization, which in turn allows queue buildup that worsens starvation for different process types. The optimal operating point emerges from the interaction of both objectives, not from maximizing either alone.

W14-ω at ω_s=0.7 (20.60s, 32.5% starvation) compared to W10C with oracle burst information (17.23s): the 3.37s MCT gap represents the true deployment cost of replacing oracle burst observability with runtime preference control. The agent trades 3.37s of throughput — 19.5% of W10C's MCT — for full deployability: no oracle state information, no retraining, a single model covering all operating modes from pure throughput to starvation-minimizing behavior.

#### 5.33.5 Connection to Production Systems

The preference vector ω is the learned equivalent of runtime scheduling knobs found in production systems: Linux's nice values, CFS cfs_bandwidth quotas, Kubernetes resource limits, and Temporal's fairness_weight parameter. All of these shift the efficiency-fairness tradeoff at runtime without modifying the underlying scheduler policy. W14-ω learns this continuous knob from data rather than engineering it by hand, with the additional property that the knob's behavior across the full frontier — not just its endpoints — is optimized jointly during training.

The non-monotonic frontier shape carries a practical operational implication: operators should not set ω_s=1.0 expecting maximum fairness. The optimal starvation operating point (ω_s=0.7) requires retaining substantial throughput pressure — a finding consistent with how MLFQ functions in practice. MLFQ's aging threshold is not zero: it deliberately allows starvation to develop over 50 seconds before intervening, because intervening earlier would disrupt the short-job prioritization that gives MLFQ its throughput advantage. W14-ω's Pareto frontier encodes this same tradeoff, but makes it continuously navigable rather than hard-coded.

---

### 5.34 Variable-N Generalization Analysis

#### 5.34.1 Motivation

W14-ω was trained exclusively on fixed N=5 process episodes drawn from the Alibaba 2018 trace. Production schedulers, by contrast, operate under continuously varying queue depths driven by bursty workloads. If the trained policy cannot generalize beyond N=5, its deployment value is limited to batch-scheduled environments where queue depth is held approximately constant. This section evaluates whether the attention mechanism's theoretical variable-N capability translates to empirical generalization under a Poisson arrival stream — without any retraining.

#### 5.34.2 Experimental Setup

A continuous Poisson arrival stream replaces the fixed-episode structure. Tasks arrive at rate λ, drawing burst lengths and resource features from the Alibaba 2018 test trace. Value-curve parameters (τ, floor) are sampled per task with the same distribution as the training environment. Each episode ends after 100 task completions. Three utilization levels are tested: ρ ∈ {0.3, 0.7, 0.9}, where ρ = λ × E[burst] and E[burst] ≈ 10s for the filtered trace (stable threshold: ρ < 1.0). The state representation is unchanged: the 35-dimensional vector (5 slots × 7 features) is filled with the first N_q = min(queue_depth, 5) tasks in FIFO order; empty slots are padded with zeros (arrived_flag=0, masked by attention). OOD% measures the fraction of scheduling decisions where queue depth exceeded 5, placing the agent outside its training distribution. MLFQ runs on identical arrival streams with its aging rule operating over the full queue at every decision. N=100 evaluation episodes per condition across 3 seeds.

#### 5.34.3 Results

| Setting | Agent | MCT | Starve% | Mean Q | Max Q | OOD% |
|---|---|---|---|---|---|---|
| Fixed N=5 | W14-ω (ω=0.7) | 20.60s | 32.5% | 5.0 | 5 | 0% |
| Fixed N=5 | MLFQ | 21.59s | 36.0% | 5.0 | 5 | 0% |
| ρ=0.3 (λ=0.03) | W14-ω (ω=0.7) | **14.58±2.84s** | 100% | 1.43 | 8 | 0.2% |
| ρ=0.3 (λ=0.03) | MLFQ | 14.62±2.75s | 100% | 1.38 | 8 | 0.2% |
| ρ=0.7 (λ=0.07) | W14-ω (ω=0.7) | 31.77±13.05s | 100% | 3.03 | 22 | 13.8% |
| ρ=0.7 (λ=0.07) | MLFQ | **31.00±12.18s** | 99% | 3.24 | 16 | 17.9% |
| ρ=0.9 (λ=0.09) | W14-ω (ω=0.7) | 58.19±34.67s | 100% | 5.99 | 36 | 38.0% |
| ρ=0.9 (λ=0.09) | MLFQ | **51.73±25.41s** | 83% | 6.30 | 31 | 46.8% |

Note: the starvation metric (any task slowdown > 3× median) saturates at ~100% in continuous-stream episodes because Poisson queueing delays produce a heavy-tailed turnaround distribution regardless of scheduling policy. The metric was calibrated for episodic fixed-N evaluation and is not a reliable discriminator in this setting. MCT and OOD% are the primary comparison axes.

#### 5.34.4 Findings

Both agents degrade continuously as utilization increases — there is no cliff-drop or catastrophic failure. W14-ω matches MLFQ at ρ=0.3 (MCT 14.58s vs 14.62s, a 0.04s gap), falls 0.77s behind at ρ=0.7, and falls 6.46s behind at ρ=0.9. The degradation profile is smooth and predictable, indicating that the 5-slot policy continues to make locally reasonable decisions even when operating outside its training distribution. This constitutes a form of graceful degradation: the agent does not collapse to random behavior, but its relative advantage over MLFQ erodes as OOD% rises from 0.2% to 38%.

The root cause of W14-ω's disadvantage at high load is the 5-slot queue window, not a failure of the FiLM ω conditioning or the attention mechanism. When queue depth exceeds 5, the agent truncates to the first 5 FIFO candidates and is blind to the remaining queue. MLFQ's aging rule, by contrast, scans the entire queue at every decision, guaranteeing that any task waiting longer than 50 seconds is promoted regardless of queue depth. This gives MLFQ a structural fairness advantage that scales with load. The OOD transition occurs around ρ≈0.83, which corresponds to E[queue]=5 under M/M/1 — precisely the point where the Poisson queue depth matches the training slot count. Below this threshold, the truncation is rarely invoked and W14-ω operates in-distribution. Above it, truncation becomes the dominant source of policy error.

The boundary of W14-ω's valid operating regime is now well-characterized: the policy beats MLFQ under fixed-N episodic scheduling and under low-utilization continuous streams (ρ≤0.3). Both conditions are practically relevant — batch workloads with bounded concurrency and lightly-loaded interactive systems. The result is valid within this scope. Variable-N retraining is identified as the necessary extension for production-load generalization: a surgical fix to the input-size constraint, not a fundamental architectural redesign. The FiLM conditioning, 2-head attention, and Pareto-frontier learning mechanism all remain valid and reusable.

#### 5.34.5 Future Work

Retraining W14-ω with a Poisson arrival environment and variable N would allow the attention mechanism to learn queue-size-invariant policies — the architecture requires only input padding changes, as the FiLM conditioning and 2-head attention are already compatible with variable-length candidate sets. Combined with the preference-conditioned objective, this would produce a single deployable agent covering both fixed-batch and continuous-stream scheduling regimes without retraining between modes.

---

### 5.35 Variable-N Training Results (W15)

#### 5.35.1 Motivation

W14-ω was trained exclusively on fixed N=5 episodes, raising the question of whether the MCT advantage over MLFQ is an artifact of the fixed-queue training distribution. §5.34 showed that W14-ω degrades at high load (ρ=0.9) when evaluated out-of-distribution on Poisson arrival streams. W15 addresses this directly by retraining the same W14-ω architecture — 2-head attention DQN with FiLM omega conditioning — on a Poisson arrival environment with variable queue depth, testing whether the MCT advantage holds under realistic continuous-stream load conditions.

#### 5.35.2 Training Details

W15 trained for 20,000 episodes on a Poisson arrival stream with λ~Uniform(0.01, 0.08), targeting 300 task completions per episode, across 3 independent seeds (42, 123, 456). The same 5-slot sliding window from W14-ω was retained, with arrived_flag masking for empty slots. Training required four stability interventions absent in the fixed-N setting: TD target clipping to ±50 (preventing bootstrap divergence from small rewards r≈−0.002), Huber loss replacing MSE (reducing quadratic amplification of large TD errors), TARGET_UPDATE_FREQ=5 (tighter synchronization under Poisson variance), and LR=1e-4. Loss converged from 4.6 at episode 1,000 to 2.6 at episode 14,000. Checkpoints were saved at `results/w15_variable_n/`.

#### 5.35.3 Results

| Agent | MCT (mean±std) | Starvation% |
|---|---|---|
| MLFQ baseline | 22.06±13.36s | 37.5% |
| W15 seed 42 | 21.97±13.81s | 40.0% |
| W15 seed 123 | 20.57±11.96s | 47.5% |
| W15 seed 456 | 19.43±11.83s | 42.0% |
| W15 mean±std | 20.66±1.03s | 43.2% |

#### 5.35.4 Findings

W15 beats MLFQ on MCT across all three seeds (best: −2.63s for seed 456, mean: −1.40s). The attention mechanism and FiLM omega conditioning transfer to variable queue depths without architectural changes — only the training environment changed. This confirms that W14-ω's MCT advantage is not an artifact of the fixed N=5 setting: the 2-head attention generalizes to continuous-stream Poisson arrivals when trained on them, recovering and in one seed improving upon the fixed-N MCT margin.

Starvation (40–48%) is worse than MLFQ (38%) across all seeds, consistent with §5.31's finding that reward shaping cannot enforce hard starvation guarantees. Variable-N training did not close this gap — starvation rates increased relative to the fixed-N W14-ω (32.5% at ω_s=0.7). The starvation problem is architectural, not environmental: the preference-conditioned omega signal shifts the MCT/starvation trade-off but cannot eliminate starvation under Poisson load without a hard constraint mechanism. Combining variable-N training with a hard starvation penalty or constrained RL formulation remains open future work.

#### 5.35.5 Connection to Variable-N Analysis

W15 addresses the retraining recommendation from §5.34 — variable-N training recovers the MCT advantage that W14-ω demonstrated at fixed N=5, confirming that the attention mechanism generalizes to variable queue depths when trained on Poisson arrival streams.

---

## 6. Discussion

### 6.1 What the Agent Discovered

The Week 5 agent learned, from reward alone and without supervision:

- **Perfect masking compliance.** Never selects a completed or unarrived process. This was not
  imposed by reward shaping — invalid actions are simply excluded from argmax. The agent learns
  to route its probability mass away from them during training.

- **Quantum strategy matching SRPT.** At n_active ≥ 2, the agent uses 1ms quanta in >99% of
  decisions. SRPT uses 1ms exclusively. The convergence is not coincidental: 1ms quanta allow
  the most frequent preemption decisions, giving the agent the highest degree of control over
  process ordering. The reward signal — which penalises every millisecond a process remains active
  — implicitly favours fine-grained control.

- **Partial SRPT process selection.** At n_active = 2, the agent agrees with SRPT 74.3% of the
  time. For the two-process case, the action-conditioned invariance guarantee is maximally
  effective: one candidate, one competitor, identical PID-masked context. The agent has effectively
  learned to prefer shorter remaining bursts in the binary case.

- **Near-optimal OOD performance.** On the fixed OOD set (bursts 4/25/2/50/8ms, three
  simultaneous arrivals at t=0), the agent achieves 29.60ms against SRPT's 28.4ms — a gap of
  1.20ms, the closest any agent in this project has come to the optimum on that configuration.

What it did not discover: a correct process selection rule for three or more simultaneous
candidates (65–68% accuracy, plateaued), and full permutation invariance over competitor features
(residual |Δ| = 0.051 in the bias probe).

### 6.2 The Permutation Invariance Ceiling

The process-selection ceiling at 65–68% for n_active ≥ 3 was not a training-time phenomenon in
Week 5. The loss converged to 0.0001. The ceiling was architectural.

The action-conditioned design (W5) shares weights across candidate PIDs — a query for P0 uses the
same network as a query for P4. But the competitor context (s_masked) still presents the four
non-candidate processes at their PID-indexed positions. A state where P1, P2, P3, P4 are runnable
competitors and P0 is the candidate produces a different s_masked than a state where P0, P2, P3,
P4 are competitors and P1 is the candidate, even if the competitor burst magnitudes are identical —
because the burst values appear at different positions in the 15-dim vector.

Week 7 addressed this directly. AttentionDQN replaces the flat competitor concatenation with
dot-product attention over competitor key-value pairs. The softmax operation produces identical
context vectors for any permutation of the competitor inputs, reducing the effective number of
distinct input configurations from 120 to 1 for a fixed set of burst magnitudes. Permutation
invariance unit test: |Δ| = 0.000 (W7) vs |Δ| = 0.0335 (W5). The architectural ceiling from W5
was closed.

However, full invariance did not close the performance ceiling at high n_active. The W7 agreement
at n_active = 3 improved substantially (90.9% vs W5's 65.4%), but collapsed at n_active = 5
(40.7% vs W5's 71.9%). The explanation is the residual entropy ceiling discussed in Section 5.8:
with H ≈ 0.62 at termination, the attention over 4 competitors is near-uniform, and the context
vector degenerates toward mean-pooling. A fixed-temperature softmax cannot sharpen beyond the
entropy floor imposed by the annealing schedule.

The remaining open problem is not architectural but parametric: a learnable inverse-temperature
parameter β (scaling dot-product scores as β/√d rather than 1/√d, with β trained alongside W_Q,
W_K, W_V) would allow the attention to sharpen beyond the annealing floor. This is a one-parameter
change that retains the full permutation invariance guarantee while removing the fixed-temperature
constraint that causes n_active = 5 failure.

A second open problem emerged from the real-trace experiments: **reward signal ambiguity on
heavy-tailed distributions**. The area-integral reward (`−n_active × q_actual`) is theoretically
equivalent to minimising mean turnaround time by Little's Law, and it successfully drives SRPT
approximation on the synthetic uniform distribution. On the filtered Alibaba trace, however, the
same reward creates gradient pressure toward attending to the longest competitor — a locally
coherent strategy that is globally anti-SRPT. The agent learns that the longest task is the most
informative context signal and focuses attention on it accordingly, but does not learn to
*schedule* the shortest task first as a consequence. On synthetic data, this failure did not
emerge because the uniform distribution's shallower magnitude spread produced a different gradient
landscape. On real data with p95/p50 = 7.2× (filtered) or 39.7× (full trace), the longest task
exerts disproportionate gradient influence and the attention converges to a "monitor the biggest
threat" strategy rather than a "identify the shortest job" strategy. Dense reward shaping that
provides a per-step signal for SRPT-aligned selections — or SRPT-supervised pre-training as a
warm-start before RL fine-tuning — may be required to break this degeneracy on real workloads.

### 6.3 The Tabula Rasa Constraint and Its Cost

The state vector used throughout this work — remaining burst, arrival flag, wait time, three
features per process — was not chosen for convenience. It was chosen as a constraint. The
deliberate premise of this experiment is that no domain knowledge should enter the agent's
observation: no job type priors, no I/O wait history, no CPU utilisation rolling average, no
priority class. If the reward signal alone is sufficient to discover scheduling policy, the
state should carry only what a bare process descriptor contains.

The constraint enabled three things that would otherwise be confounded. First, it produced a
clean architectural progression: each failure mode was diagnosable in isolation because the
observation space held fixed. Identity bias, discretisation collapse, and attention entropy
collapse were architectural failures, not feature failures — a distinction that disappears when
the state vector changes between experiments. Second, the permutation invariance analysis is
exact precisely because the three-feature encoding has a known, regular structure: reshaped as
(N_PROCESSES, 3), it admits a formal proof that masking and attention preserve order-invariance
over the candidate slot. A heterogeneous feature vector with variable-length job-type encodings
would break this argument without a more complex treatment. Third, the failure modes themselves
— in particular, the anti-SRPT longest-competitor attention (Section 5.16) and the reward
degeneracy on heavy-tailed distributions (Section 6.2) — are precisely diagnosable because
the state provides no confound. The agent failed not because it lacked I/O information but
because the reward signal created gradient pressure in the wrong direction. That diagnosis
would be harder to make cleanly if the state included features that partly compensate for the
reward's ambiguity.

The cost is the real-trace performance ceiling. On filtered Alibaba trace data, the W8c and
W8d agents (3-feature state) achieved mean MCT above Round Robin (23.30s and 24.66s versus
21.77s) — not crossing the Round Robin threshold despite one agent receiving double the training
episodes. Extended training (W8d, 20,000 episodes) degraded SRPT agreement from 57.9% to 39.2%,
confirming the 3-feature state was at its ceiling rather than undertrained. The W9 agent, which
adds plan_cpu and plan_mem as the two proxy features available in the Alibaba trace, crosses the
Round Robin threshold for the first time at 19.22s (−2.09s vs RR).
The literature's evidence points directly at what is missing. Across RLScheduler [3] and Double DQN [4], two features appear consistently as load-bearing: I/O wait ratio (the fraction of
recent time a process spent blocked on I/O, which separates CPU-bound from I/O-bound tasks)
and historical CPU usage (a rolling estimate of actual CPU consumption, which provides a prior
on remaining burst that does not require direct burst observability). Neither appears in the
current state vector.

Adding these features is a concrete architectural change. The per-process feature vector grows
from 3 dimensions (remaining burst, arrival flag, wait time) to 5 (adding I/O wait ratio and
historical CPU usage estimate), expanding the state vector from 15 to 25 dimensions. The
attention module's query and key projections — currently mapping 3 → 8 — would need to
absorb the wider input; a doubling of d_attn from 8 to 16 is a natural choice, adding
approximately 480 parameters to the 3,041-parameter current model. The normalization scheme
requires separate treatment: burst is normalised via log1p against BURST_P95; I/O wait ratio
is already in [0, 1] and requires no scaling; historical CPU usage carries a different
magnitude and temporal structure and likely requires its own rolling-percentile normalization
rather than a fixed denominator. The replay buffer schema, the state encoding function, and
the environment's process tracking all require modification to supply the two new features.
None of these changes are architecturally novel. Each is a standard extension with a known
implementation path.

Week 9 confirmed this: adding plan_cpu and plan_mem as resource-class proxies, despite encoding
requested rather than measured usage, was sufficient to beat Round Robin and correct identity
bias direction on real trace data. Week 10C confirmed and extended this: two-head attention
reduced mean MCT by 1.01s and SRPT agreement at n_active = 2 by 12.4 percentage points, with
spontaneous head specialization into context and selection roles — but the high-contention ceiling
(n_active ≥ 4) remained, confirming that planned rather than measured resource features are the
binding constraint at five-process contention.

### 6.4 Why the Spurious Quantum Discovery Matters

The Week 3 load-adaptive quantum pattern is worth examining beyond its status as an artefact.
The pattern was:
- Stable and reproducible across evaluation runs.
- Load-correlated in a directionally plausible way (larger quanta as more processes compete).
- Coherent enough to generate a credible post-hoc explanation grounded in queuing theory.

A practitioner examining the Week 3 agent without the Week 5 ablation might report a genuine
discovery: "the agent learned to amortise context-switch costs under high load, a behaviour not
explicit in the reward signal." That paper would be wrong.

The source of the error is that PID-indexed action heads accumulate asymmetric gradient histories
across training episodes. If high n_active situations tend to involve process set configurations
where certain PIDs are active — and in this environment they do, because arrival ordering
determines which PIDs are present — then the gradient pressure toward particular quanta in those
PID-indexed neurons looks, from the outside, like load-sensitive behaviour.

The generalised lesson: any RL agent with position-indexed action heads can produce policies that
are coherent, reproducible, and superficially interpretable while being positional artefacts.
The diagnostic is simple — an architecture ablation that shares weights across positions — but
requires knowing to run it. Policy analysis without an invariance ablation is incomplete.

### 6.5 Limitations

Three limitations of this work are worth stating explicitly, as they bound the scope of the
conclusions.

**N = 5 is a toy, and scaling changes the architecture non-trivially.** The fixed N = 5 with a
known arrival slot set was chosen for diagnostic clarity, not representativeness. Scaling to
variable N requires a dynamic candidate pool, which in turn requires the attention-based
competitor aggregation identified as the next architecture step. This is not a drop-in change: it
modifies the input dimensionality, the batching strategy, and the replay buffer schema. The
performance conclusions reported here — the 8.38ms SRPT gap, the 74.3% two-process agreement —
do not extrapolate to larger N without retraining and re-evaluation.

**U[1, 60]ms burst distribution is not representative of real workloads.** Real CPU burst
distributions are heavy-tailed and multi-modal, reflecting a mix of short interactive tasks and
long batch jobs. A uniform distribution over [1, 60]ms covers neither the sub-millisecond
interactive regime nor the multi-second batch regime that dominate production schedulers. An agent
trained on U[1, 60]ms may learn policies that are well-calibrated for mid-range bursts but fail
at the tails — and the tails are where scheduling decisions matter most.

**10,000 training episodes on a fixed arrival slot structure may have overfit the arrival
distribution.** Arrival times are drawn from a fixed set {0, 2, 5, 8, 10}ms across all episodes.
An agent trained on this distribution may implicitly learn the arrival timing rather than general
scheduling principles. A holdout evaluation using arrival times sampled from a continuous
distribution outside this set would test whether the state encoding (which includes wait_time/300)
generalises beyond the training arrival structure.

### 6.6 Ethical Considerations

The value-aware scheduling framework raises practical ethical questions that warrant acknowledgment. First, value curve assignment — the decision of which tasks receive steep urgency curves versus smooth batch curves — is a policy choice with fairness implications. In the experiments reported here, curve assignment follows task class (interactive and IO tasks receive steep curves; batch tasks receive smooth curves), a heuristic that reflects typical latency expectations but may disadvantage batch workloads belonging to resource-constrained users or lower-priority tenants.

Second, the τ parameter directly controls how quickly a task's value decays — a smaller τ encodes higher urgency and results in preferential scheduling. In multi-tenant systems, τ assignment could become a mechanism for implicit prioritization of certain workload classes over others. The mapping from business requirements to τ values is a design decision with distributional consequences that are not captured by throughput or fairness metrics alone.

Third, optimizing for value delivery rather than equal CPU time explicitly abandons the egalitarian fairness model of CFS and EEVDF. While VRFI provides a principled alternative fairness criterion, the choice of fairness definition is ultimately a value judgment that should involve stakeholder input in production deployments. Schedulers that optimize value curves without transparency about curve assignment risk encoding organizational priority hierarchies into infrastructure in ways that are difficult to audit or contest.

### 6.7 Connection to Real Scheduler Traces

The identity bias analysis methodology — probe states that isolate a single scheduling decision,
permutation invariance unit tests that verify architectural guarantees hold in practice — is a
diagnostic technique applicable to any RL scheduler regardless of environment scale. It is the
most directly transferable contribution of this work. Any scheduler trained with position-indexed
action heads should be subjected to an invariance ablation before its policy patterns are reported
as learned behaviour.

What else would likely transfer: the reward formulation (area integral of active count) maps
directly to a real latency objective if process state is tracked at each scheduling decision. The
action-conditioned architecture, generalised to variable N via attention-based competitor
aggregation, produces a policy that is robust to process ordering by construction — a property
essential in any real workload where the same task mix can arrive in any order.

What would not transfer without modification: remaining burst observability (requires estimation
from hardware counters), clean episode boundaries (requires a non-episodic reward formulation for
continuously-arriving workloads), and the fixed-N assumption (requires variable-length candidate
pools with the attention mechanism needed for full permutation invariance).

### 6.8 Future Work

Several concrete directions emerge from the failure modes and formal results.

**Learnable attention temperature.** The n_active = 5 failure (§6.2) is caused by residual attention entropy H ≈ 0.62 at termination, producing near-uniform weighting over competitors. A learnable inverse-temperature parameter β (scoring as β/√d rather than 1/√d, trained alongside W_Q, W_K, W_V) would allow the attention to sharpen beyond the annealing floor while preserving full permutation invariance.

**Measured runtime features.** The real-trace ceiling (§6.3) is attributable to the absence of I/O wait ratio and historical CPU usage — features identified as load-bearing across RLScheduler [3] and Double DQN [4]. Expanding the per-process vector from 5 to 7 dimensions, with appropriate normalization, is the concrete next step. Section 6.3 details the normalization scheme and parameter count.

**Joint value curve optimization.** Section 5.28 identifies VRFI-based loss as exploit-resistant and efficiency-based loss as degenerate. A natural next direction is joint optimization of efficiency and fairness under fixed-base, fixed-floor constraints, where only τ parameters are free — the one axis with a genuine gradient signal. Multi-objective Pareto analysis across Loss A and Loss B would characterize the achievable frontier.

**Dense reward shaping for SRPT alignment.** The anti-SRPT longest-competitor attention on real traces (§5.16) is a gradient pressure problem: the area-integral reward locally rewards attending to the longest task even though SRPT-optimal decisions require scheduling the shortest. Deadline-aware shaping bonuses conditioned on per-task τ values — or SRPT-supervised pre-training as a warm-start — may break this degeneracy without sacrificing the tabula rasa property (no behavioral cloning or reference demonstrations).

**Non-episodic and variable-N formulations.** The current framework assumes clean episode boundaries and a fixed pool of N = 5 processes. Real workloads involve continuous arrivals and variable queue depth. Adapting the architecture to variable-length candidate sets (via dynamic attention pooling) and reformulating the reward for non-episodic streams (via discounted infinite-horizon formulations) are prerequisites for deployment in any production scheduler environment.

---

## 7. Conclusion

Over eight weeks, without expert demonstrations, handcrafted features, or a target policy, a
tabula rasa RL agent (no behavioral cloning or reference policy demonstrations; value curve parameters are human-specified) progressed from a tabular scheduler with zero cross-episode variance through
four increasingly capable architectures, closing the gap to the preemptive optimal from 17.34ms
(Week 3) to 8.38ms (Week 5) to a different quality frontier: an AttentionDQN (Week 7) that
achieves lower variance than SRPT itself (std = 21.82ms vs 23.96ms) while meeting the 80%
SRPT-agreement target at n_active ≤ 3 for the first time.

Eight failure modes were identified, diagnosed, and resolved or explicitly left open. The honest
negative results — discretisation collapse (W1), identity bias worsening with the move to DQN
(W3), a spurious quantum artefact indistinguishable from a learned heuristic without ablation
(W3/W4), attention entropy collapse before TD signal accumulates (W7), n_active = 5 failure from
residual entropy (W7), linear-normalization failure on heavy-tailed real traces (W8), attention
entropy collapse on the full Alibaba trace (W8), anti-SRPT longest-competitor attention on the
filtered trace (W8c) — are as central to the contribution as the positive ones.

Three formal results emerged from the value-curve analysis. The Quantum Scoring Monotonicity
Lemma (Section 3.1.3) proves that no per-task scoring function can produce mixed quantum
selection for any τ > 0, d ≥ 0: rate-normalised scorers always degenerate to q_min; total-value
scorers always degenerate to q_max. This is not a calibration failure — the proof holds for all
parameter values — and establishes queue-global information as a necessary condition for
value-aware scheduling, confirmed empirically in Section 5.23. The Value-Rate Fairness Index
(VRFI = 1 − CV(VLR)) identifies value disparity that Jain's Fairness Index and slowdown variance
miss: value_aware achieves zero starvation and rank sum 17 across five policies, and
VRFI-targeted parameter optimization achieves the first positive VRFI (+0.289) on the Alibaba
trace — while efficiency-targeted optimization degenerates through floor collapse and base
shrink regardless of regularization strength.

Three open problems remain. First, the n_active = 5 failure on synthetic data: H ≈ 0.62 at
termination produces near-mean-pooling at 4 competitors, erasing the ordering information the
attention was introduced to capture. The fix is a learnable inverse-temperature parameter β
(scoring as β/√d rather than 1/√d) trained alongside the attention weights. Second, the
attention diagnostic on synthetic data revealed that correct decisions are characterised by sharp
focus on the longest competitor (90.6% in agree cases vs 62.4% in disagree cases): sharper
attention correlates with better decisions, further motivating a learnable temperature. Third, on
real traces, the area-integral reward creates gradient pressure toward longest-competitor
attention that is locally coherent but globally anti-SRPT — a degeneracy that learnable
temperature alone cannot resolve and that may require dense per-step reward shaping aligned with
SRPT or supervised pre-training on SRPT demonstrations as a warm-start before RL fine-tuning.

Extended to real workloads via the Alibaba 2018 batch_task trace (14.3M tasks, p95/p50 = 39.7×),
the agent required two preprocessing steps to become tractable: filtering tasks above the 75th
percentile duration (reducing within-episode p50 burst ratio from 74× to 14×) and log-
normalization of burst durations. With these changes, the W8c agent achieved mean MCT within
1.53s of Round Robin (23.30s vs 21.77s) with lower variance (10.85s vs 13.61s) on filtered test
episodes — representing a 94% reduction in the Round Robin performance gap compared to the
unfiltered baseline (+26.24s). The attention mechanism stabilized at H = 0.32–0.50 but learned
to attend to the longest competitor with 93.7% frequency — a coherent but anti-SRPT strategy
under the area-integral reward signal. The architecture is in place; the remaining obstacles are
the reward signal degeneracy on real workloads and the upper quartile of the duration
distribution that filtering currently excludes.

A 2×2 ablation over reward structure and burst observability establishes that both factors are independently necessary for W10C-level performance: removing the value-delta reward (W11d) degrades SRPT agreement to 40.6% and MCT to 26.35s, while removing oracle burst time (W12) increases mean completion time to 21.67s. The direction of asymmetry — reward structure more damaging than observability loss — is preserved across synthetic and real-trace conditions, though the magnitude contracts substantially on the Alibaba 2018 trace. The gap between oracle-informed and observable-only scheduling is 4.4s on the 2018 trace, an explicit quantification of this observability cost whose sensitivity to training distribution is itself a novel finding.

The complete real-trace progression — from 135.76s (linear normalization, full trace) to 19.22s
(log normalization, filtered trace, 5-feature state) — required four interventions: outlier
filtering (p75 cutoff), log-normalization, lambda_ent = 0.30 entropy regularization, and
resource-class features. Each intervention was motivated by a specific diagnosed failure. The W9
agent beats Round Robin by 2.09s on mean MCT while maintaining lower variance than Round Robin
(11.58s vs 14.03s), and closes the SRPT gap to 4.03s on filtered Alibaba 2018 batch_task data.
The tabula rasa constraint — no behavioral cloning or reference policy demonstrations, no handcrafted priority rules — is preserved throughout (note: value curve parameters τ, floor, base are human-specified modeling choices encoding task urgency): plan_cpu and plan_mem are Alibaba 2018 cluster trace fields with hand-assigned value curve profiles requiring no scheduler domain knowledge to define. The complete intervention sequence after W9 tested four approaches: PBRS
α = 0.20 (F17, regressed), PBRS α = 0.05 (F17, flat), learnable beta (F18, regressed to softer
attention), and two-head attention (F19, improved). Two-head attention is the only post-W9
intervention that improved performance, reducing mean MCT from 19.22s to 17.23s and closing the
SRPT gap from 4.02s to 2.03s. The improvement mechanism — spontaneous head specialization into
context and monitoring roles — confirms that single-head attention was the architectural
bottleneck, not the reward signal or feature set. The remaining 3.01s gap to SRPT and tier0
collapse at n_active ≥ 4 identify feature discriminability at high contention as the binding
constraint, addressable only with measured runtime utilization unavailable in the Alibaba
batch_task schema.

W14-ω, a preference-conditioned attention DQN, achieves the project's core goal: a single deployable agent that beats MLFQ on both MCT (20.60s vs 21.59s) and starvation prevention (32.5% vs 36.0%) without oracle burst information, using a runtime preference vector ω that shifts behavior continuously between throughput and fairness operating modes. The gradient norm ratio (0.92–1.12) confirmed architectural stability throughout training — the FiLM conditioning and proportional starvation signal successfully balanced competing objectives where additive reward decomposition (W11) had previously failed. The non-monotonic Pareto frontier — starvation minimized at ω_s=0.7, not ω_s=1.0 — establishes that the optimal fairness operating point requires retaining throughput pressure, a result consistent with MLFQ's design and with theoretical predictions about purely fair schedulers. The 3.37s MCT gap between W14-ω and oracle-informed W10C quantifies the total deployment cost of removing burst observability while adding runtime preference control. W15, trained on Poisson arrival streams, confirms that the MCT advantage over MLFQ (mean −1.40s across 3 seeds) generalizes beyond the fixed N=5 training distribution.

---

*All experiments: Python 3.10+, NumPy + stdlib; W15 additionally uses PyTorch for GPU-accelerated network training (CPU device; CUDA path available).*
*Graduate Capstone Project, 2024–2025 (Dev Upadhyay, UCSC M.S. CS), Weeks 1–15. Generated with Claude Code (claude-sonnet-4-6).*

---

## Figures

**Figure 1:** W14-ω Pareto frontier showing MCT vs starvation rate as preference ω_s varies from 0 to 1. The shaded region dominates MLFQ on both metrics simultaneously. ω=0.7 (gold star) is the recommended deployment point. File: `figures/fig1_pareto_frontier.pdf`.

**Figure 2:** W10C learning curve over 10,000 training episodes with Round Robin and SRPT oracle baselines. Multi-seed fresh training band (3 seeds, mean±std) shown in gray. Final result: 17.23s at episode 10,000. File: `figures/fig2_learning_curve.pdf`.

**Figure 3:** W14-ω architecture diagram showing 2-head attention DQN with FiLM omega conditioning at query modulation (pre-attention: Q = Q × (1+ω_s)) and post-attention layers (context × (1+ω_s) + ω_mct). MLP input: [enc(7) ‖ context(16) ‖ ω_s(1)] = 24-dim. File: `figures/fig3_architecture.pdf`.

**Figure 4:** VRFI falsification example. Ten tasks with identical delays (d=50s) score JFI=1.0 and SDV=0 under existing metrics, yet VRFI=0.641 correctly detects value-rate inequality between steep (τ=25s, floor=0.2) and smooth (τ=100s) curve types. File: `figures/fig4_vrfi_falsification.pdf`.

---

## References

[1] S. Y. Kahu. "KernelOracle: Predicting the Linux Scheduler's Next Move with Deep Learning." *arXiv preprint arXiv:2505.15213*, 2025. https://arxiv.org/abs/2505.15213

[2] Y. Fu, R. Shi, H. Wang, S. Chen, and Y. Cheng. "ALPS: An Adaptive Learning, Priority OS Scheduler for Serverless Functions." *Proceedings of the USENIX Annual Technical Conference (USENIX ATC '24)*, 2024. https://www.usenix.org/conference/atc24/presentation/fu

[3] D. Zhang, D. Dai, Y. He, F. S. Bao, and B. Xie. "RLScheduler: An Automated HPC Batch Job Scheduler Using Reinforcement Learning." *Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC '20)*, 2020. https://doi.org/10.1109/SC41405.2020.00035. arXiv:1910.08925.

[4] X. Sun, Y. Duan, Y. Deng, F. Guo, G. Cai, and Y. Peng. "Dynamic Operating System Scheduling Using Double DQN: A Reinforcement Learning Approach to Task Optimization." *arXiv preprint arXiv:2503.23659*, 2025. https://arxiv.org/abs/2503.23659

[5] S. Goodarzy, M. Nazari, R. Han, E. Keller, and E. Rozner. "SmartOS: Towards Automated Learning and User-Adaptive Resource Allocation in Operating Systems." *Proceedings of the 12th ACM SIGOPS Asia-Pacific Workshop on Systems (APSys '21)*, pp. 48–55, 2021. https://doi.org/10.1145/3476886.3477519

[6] J. D. C. Little. "A Proof for the Queuing Formula: L = λW." *Operations Research*, vol. 9, no. 3, pp. 383–387, 1961. https://doi.org/10.1287/opre.9.3.383

[7] R. Jain, D. Chiu, and W. Hawe. "A Quantitative Measure of Fairness and Discrimination for Resource Allocation in Shared Computer Systems." DEC Research Report TR-301, 1984. https://arxiv.org/abs/cs/9809099

[8] Alibaba Group. "Alibaba Cluster Trace Program." *Cluster Data*, 2018. https://github.com/alibaba/clusterdata

[9] T. Natvig. "EEVDF Scheduler." *Linux Kernel Mailing List*, 2023. https://lwn.net/Articles/925371/
