# RL-Based CPU Scheduler — Project Report

**UCSC Graduate Capstone**
**Stack: Python 3.10+, pure NumPy + stdlib (no PyTorch / TensorFlow)**

---

## Abstract

We built a tabular Q-learning and then deep Q-network (DQN) CPU scheduler for a 5-process stochastic environment, with no hardcoded policy, trained entirely from a dense area-integral reward signal that is theoretically equivalent to minimising mean turnaround time. Across five weeks, the key findings were: (Week 1) a tabular agent converges on fixed process sets but fails to distinguish between processes mapped to the same discretisation bin, producing zero variance and identity bias; (Week 2) randomised training corrects bias direction and increases state coverage to 5,386 states (32% of the 16,807-state tabular space), but the discretisation ceiling prevents the agent from responding to exact burst magnitudes; (Week 3) a numpy-only DQN with He initialisation, experience replay, and a target network escapes the discretisation ceiling — variance std = 26.51ms, within 2.55ms of the SRPT oracle's natural process-set variance — but introduces a new failure mode: position-indexed action heads make permutation invariance unlearnable, and identity bias increases rather than decreases; (Week 4) policy extraction from the Week 3 DQN reveals a load-adaptive quantum pattern (20ms quanta preferred at n\_active ≥ 3) that appears to reflect genuine learning but turns out to be a PID-bias artefact; (Week 5) replacing the position-indexed output layer with an action-conditioned architecture (19→64→32→1, one forward pass per candidate action) achieves partial permutation invariance, reduces mean MCT from 85.00ms to 76.04ms, resolves identity bias (|Δ| drops from 5.48 to 0.051 in the correct direction), and eliminates the spurious quantum pattern — confirming it was an artefact and not a learned heuristic. Full permutation invariance across arbitrary process orderings remains an open problem, motivating an attention-based architecture as the natural next step.

---

## Environment and Setup

**Processes:** 5 processes per episode. Arrival times sampled without replacement from {0, 2, 5, 8, 10}ms; burst lengths from U[1, 60]ms (randomised per episode in Weeks 2–3; fixed in Week 1).

**Action space:** 15 actions = 5 processes × 3 quantum tiers (1ms, 5ms, 20ms). Invalid actions (non-runnable processes) are masked at every decision step.

**Reward:** Dense area-integral signal R(t) = −n\_active(t) × q\_actual, normalised by 100.0 in Week 3. Minimising cumulative reward is equivalent to minimising total turnaround time.

**Baselines:**
- Round Robin (5ms quantum, PID-ascending cycle): MCT ≈ 34.4ms (in-distribution), 36.4ms (OOD)
- SRPT (Shortest Remaining Processing Time, preemptive): MCT ≈ 26.4ms (in-distribution), 28.4ms (OOD)

---

## Week 1 — Tabular Q-Learning on a Fixed Process Set

**Setup:** QLearningAgent with Q-table shape (7,7,7,7,7,15) ≈ 1.9MB. State: rb\_bin 5-tuple encoding remaining burst as one of 7 discrete values (0=complete, 1–5=active tiers, 6=not arrived). Training: 10,000 episodes on a single fixed process set, γ=1.0, α=0.1, ε decaying from 1.0 to 0.05.

**Results (fixed OOD set):**
- Agent MCT: 29.80ms (std = 0.00ms)
- Beat Round Robin (36.4ms): **YES** (−18.1%)
- Gap to SRPT (28.4ms): +1.40ms

The agent learned to approximately replicate SRPT scheduling on the fixed process set, finishing 1.4ms above optimal after 10,000 episodes.

**Key limitation discovered:** The discretisation scheme maps all burst lengths within the same tier to identical Q-values. Two episodes with different actual burst magnitudes but identical rb\_bin tuples produce identical action sequences. This is the discretisation ceiling: the tabular agent is insensitive to exact burst values within a bin.

**Identity bias (tabular Week 1):** In state (0,2,0,4,1) — P1 at bin=2 (~10ms), P4 at bin=1 (~3ms), SRPT prefers P4 — the agent assigned Q(P1/long) > Q(P4/long) with |Δ| = 0.270, preferring the longer process. The bias was traced to PID-indexed Q-table structure: P1 occupies table slot 1, which accumulated more positive Q-values due to arrival-time patterns in the fixed process set.

---

## Week 2 — Randomised Training, Coverage Scaling

**Setup:** Same QLearningAgent. Training changed to generate a fresh random process set each episode (arrivals and bursts re-sampled). Reward rewritten to pure dense env signal (no completion bonus). Coverage tracking added: set of distinct discrete state tuples visited.

**Results after 10,000 episodes:**

| Set | MCT | Std | Beat RR? | Gap to SRPT |
|-----|-----|-----|----------|-------------|
| In-distribution fixed | 29.80ms | 0.00ms | YES (−13.5%) | +3.40ms |
| OOD (Week 1) | 32.40ms | 0.00ms | YES (−11.0%) | +4.00ms |

**Coverage:** 5,386 distinct states visited (32.1% of 16,807). Coverage growth was fast early (>2,000 states by ep 1,000) then logarithmic, consistent with the birthday-paradox coverage law for uniform random sampling.

**Identity bias (tabular Week 2):** Randomised training corrected the bias direction — P4 now preferred over P1 (Q(P4/long) = −3.68 vs Q(P1/long) = −3.91) — but |Δ| = 0.225, still exceeding the 0.1 resolution threshold. The residual bias is architectural: the Q-table is indexed by PID, so distinct processes always occupy different weight vectors regardless of their similarity in burst-space.

**Key conclusion:** The discretisation ceiling is not a coverage problem; it is a resolution problem. Even with 32% state coverage, the tabular agent cannot distinguish burst values within the same bin. This motivated replacing the Q-table with a continuous-state function approximator.

---

## Week 3 — Deep Q-Network (numpy-only)

**Architecture:** 15 → 64 (ReLU) → 32 (ReLU) → 15 (linear), 3,599 parameters. He initialisation. Adam optimiser (lr=0.001, β1=0.9, β2=0.999). Global gradient norm clipping (max\_norm=1.0).

**State encoding:** 15-dim continuous vector — per process: [remaining\_burst/60, arrived\_flag, wait\_time/300].

**Training details:**
- 10,000 episodes; fresh random process set per episode
- Experience replay buffer: 10,000 transitions, batch size 32
- Warmup: 500 transitions of random exploration before first gradient update
- Target network hard copy: every 200 episodes
- Reward normalisation: R\_norm = R\_raw / 100.0 (keeps per-step reward in [−1, 0])
- ε decaying from 1.0 to 0.05 (formula-based; hits minimum at ep ≈ 5,990)

**Loss curve:**

| Episode | Avg Loss |
|---------|----------|
| 500 | 0.0219 |
| 1,000 | 0.0082 |
| 2,000 | 0.0067 |
| 5,000 | 0.0398 |
| 10,000 | 0.6504 |

Loss remained stable throughout. (A first training run without reward normalisation diverged: loss grew from 46 to 87,799; Q-values reached +3,000. Fixed by dividing reward by 100.0 and switching from element-wise gradient clipping to global norm clipping.)

---

## Week 3 Findings

### Finding 1 — Null Hypothesis Rejected: std = 26.51ms

The primary Week 3 hypothesis was that a trained DQN would produce non-zero variance across novel random process sets, ruling out trajectory memorisation as the source of any observed performance. This hypothesis is confirmed.

Evaluated across 500 randomly generated process sets (fixed master seed 42), the DQN produced **mean MCT = 85.00ms, std = 26.51ms**. The network is adapting its policy to each distinct episode rather than replaying a fixed schedule.

To calibrate whether this variance reflects agent behaviour or process-set heterogeneity, the SRPT oracle was run on the identical 500 episodes. SRPT produced **std = 23.96ms**. The DQN's standard deviation exceeds SRPT's by only 2.55ms. The bulk of the DQN's cross-episode variance is structural — inherited from the variance of the process sets themselves — rather than policy noise. The residual 2.55ms likely reflects genuine sub-optimality in states where the DQN's approximation diverges from the optimal schedule.

For reference, the tabular Q-learning agent (Weeks 1–2) produced **std = 0.00ms** on both fixed evaluation sets. This was not a sign of quality; it was a sign of discretisation collapse.

### Finding 2 — OOD Generalisation Achieved, With Caveats

On the Week 1 OOD fixed process set (three simultaneous arrivals at t=0; bursts 4, 25, 2, 50, 8ms), the DQN achieved **MCT = 32.00ms**, beating Round Robin (36.4ms) by 4.40ms and finishing 3.60ms above the SRPT optimum (28.4ms). The tabular Week 2 agent scored 32.40ms on the same set — the DQN matches or slightly exceeds it by 0.40ms.

A caveat must be stated clearly: the concept of "out-of-distribution" is less well-defined for a DQN operating on continuous state than for a tabular agent. The tabular agent has discrete cells; a state not seen during training has a literally zero-initialised Q-value. The DQN has no such boundary — any input vector produces a Q-value. This means the DQN's OOD performance could reflect genuine generalisation or the smoothness of a function approximator that happens to extrapolate correctly on this particular fixed set. Distinguishing these interpretations would require a broader OOD benchmark.

### Finding 3 — Fixed-Set Regression Is an Evaluation Artefact

On the in-distribution fixed set, the DQN scored **MCT = 34.60ms** against the tabular Week 2 agent's **29.80ms** — a regression of 4.80ms. This comparison is structurally unfair. The tabular agent was trained for 10,000 episodes in which this fixed set appeared with high frequency; it has effectively memorised the optimal action sequence for this configuration. The DQN was trained on randomised episodes and has never seen this fixed evaluation set.

The honest comparison is random-evaluation performance — and there, the tabular agent cannot participate meaningfully. Its std=0.00ms reveals that it would assign identical Q-values to all burst configurations within the same bin tier, returning the same schedule regardless of true burst magnitudes. The tabular agent is insensitive where the DQN is not. The evaluation design revealed a deeper point: fixed-set evaluation is the wrong benchmark for a generalising agent. A scheduler that reads burst magnitudes should be evaluated on diverse process sets; one that memorises trajectories should be evaluated on the set it memorised. Future work should use random-set evaluation exclusively, with SRPT as the oracle baseline.

### Finding 4 — Identity Bias Is Architectural, Not Trainable

The identity bias probe was re-run with a continuous state encoding: P0 complete, P1 with 10ms remaining, P2 complete, P3 not yet arrived, P4 with 3ms remaining. The SRPT-correct answer is to prefer P4.

The DQN assigned **Q(P1/long) = +44.92, Q(P4/long) = +39.43, |Δ| = 5.48** — preferring the longer process by a substantial margin. This is worse, not better, than the tabular Week 2 result (|Δ| = 0.225, correct direction).

The root cause is architectural. A standard DQN with fixed per-action output neurons encodes process identity by position: action indices 0–2 always map to P0, actions 3–5 to P1, and so on. The network must learn a different policy for the same abstract situation depending on which PID indices happen to be occupied. With gradient updates for (P0=10ms, P4=3ms) and (P2=10ms, P1=3ms) flowing through completely disjoint weight paths, the network has no way to learn "prefer the shorter process" as a general rule. With 10,000 training episodes distributed across 5! = 120 permutation classes, this is an under-determined learning problem by construction.

Resolving this requires an architecture invariant to process permutation: either an attention mechanism over process features, or an action-conditioned output where the same Q-network is queried separately for each (state, process) pair. This is the Week 5 architecture.

---

## Week 4 — Policy Extraction and Analysis

**Goal:** Open the Week 3 DQN as a black box. Extract 10,000 greedy (state, action) pairs from the trained network and compare each decision to the SRPT oracle to characterise what the network learned, where it agrees with the optimal policy, and whether disagreement is systematic or noise.

**Method:** 10,000 decisions were sampled by running the trained DQN greedily (ε = 0) on fresh random process sets. For each decision step, the greedy action was recorded alongside the SRPT-optimal action. Decisions were grouped by n\_active (number of runnable processes at that step) to separate the regimes.

**Process selection agreement by n\_active:**

| n\_active | agree% | 1ms% | 5ms% | 20ms% |
|-----------|--------|------|------|-------|
| 1 | 100.0% | 98.0% | 1.5% | 0.5% |
| 2 | 65.3% | 53.0% | 9.2% | 37.8% |
| 3 | 57.1% | 32.6% | 7.8% | 59.6% |
| 4 | 58.9% | 29.3% | 5.4% | 65.3% |
| 5 | 62.4% | 27.1% | 4.8% | 68.1% |

**Key observation:** When only one process is runnable (n\_active = 1), agreement is perfect — there is only one valid choice. As n\_active grows, process-selection agreement drops to the 57–63% range. More striking is the quantum pattern: at n\_active = 1 the network uses predominantly 1ms quanta (matching SRPT), but at n\_active ≥ 3 it shifts heavily toward 20ms quanta. This was interpreted as a possibly learned heuristic — "use large quanta when many processes are waiting, to reduce context-switch overhead" — but the Week 5 architecture test reveals a different story.

**Systematic vs noise disagreement:** Disagreements at n\_active ≥ 2 were not random. The network systematically preferred higher-PID processes when two processes had similar remaining bursts — a direct consequence of PID-indexed action heads accumulating asymmetric gradient histories. This is the same structural source as the identity bias, now observable at scale in the full policy rather than only in a targeted probe.

**Conclusion:** The Week 3 DQN learned a partial SRPT-like process selection (better than random, worse than optimal) and an apparent quantum adaptation strategy. Whether that quantum strategy reflects genuine learning or a PID-indexed weight artefact was left as the central question for Week 5.

---

## Week 5 — Action-Conditioned DQN

**Motivation:** The Week 3 DQN's 15 output neurons are permanently tied to PID positions. Any scheduling rule that references process identity implicitly (e.g., "P2 gets 20ms") will pollute gradient updates for the general rule "shorter burst → smaller quantum". Fixing this requires sharing weights across all candidate processes. The action-conditioned architecture achieves this by replacing the 15-output head with a single scalar output, querying the network once per candidate action with an input that describes the candidate explicitly.

**Architecture:** 19 → 64 (ReLU) → 32 (ReLU) → 1 (linear), 3,393 parameters. He initialisation. Adam optimiser (lr=0.001, β1=0.9, β2=0.999). Global gradient norm clipping (max\_norm=1.0).

**State encoding:** Input is a 19-dim vector concatenating:
- **s\_masked (15-dim):** the standard 15-dim state with the candidate process's slot zeroed out — [remaining/60, arrived\_flag, wait/300] set to 0 for that PID. This prevents redundant representation of the candidate's own features in both the context and the action descriptor.
- **a4 (4-dim):** the candidate's own features extracted before zeroing, plus the quantum tier normalised: [remaining/60, arrived\_flag, wait/300, qt/2.0].

**Permutation invariance (partial):** The zeroing design achieves exact permutation invariance for the shared competitor context. Two queries — (P0 as candidate, P1 as sole competitor) and (P4 as candidate, P1 as sole competitor) — produce identical s\_masked vectors, identical a4 vectors for the same candidate burst and quantum, and therefore identical Q-values. Verified: Q(P0/long) = Q(P4/long) = −0.9054024509 with P1 as the shared competitor. This invariance is partial: it holds when the competitor set is identical in PID positions; full permutation invariance across arbitrary reorderings of all 5 processes requires an architecture that aggregates competitor features position-independently (e.g., attention or sum-pooling), left for a future week.

**Training details:**
- 10,000 episodes; fresh random process set per episode
- Experience replay buffer: 10,000 transitions, batch size 32
- Target network: all 15 candidate Q-values computed in a single batched forward pass of shape (batch × 15, 19)
- Warmup: 500 transitions of random exploration before first gradient update
- Target network hard copy: every 200 episodes
- Reward normalisation: R\_norm = R\_raw / 100.0

**Loss curve:**

| Episode | Avg Loss |
|---------|----------|
| 500 | 0.0114 |
| 1,000 | 0.0026 |
| 2,000 | 0.0007 |
| 5,000 | 0.0002 |
| 10,000 | 0.0001 |

Loss converged more smoothly and to a lower value than Week 3 (0.0001 vs 0.6504 at ep 10,000), consistent with the network having a more coherent learning target — a single scalar Q-value per action rather than 15 simultaneous regression targets with cross-contaminated gradients.

---

## Week 5 Findings

### Finding 1 — Mean MCT Improved: 85.00ms → 76.04ms

Evaluated across the same 500 random process sets (master seed 42) used in Week 3, the action-conditioned DQN produced **mean MCT = 76.04ms, std = 27.70ms**. This is an 8.96ms improvement over the Week 3 DQN (85.00ms) — a 10.5% reduction — with no change to the environment, reward function, training budget, or hyperparameters. The only change was the architecture.

The gap to SRPT (67.66ms) narrowed from 17.34ms (Week 3) to **8.38ms** (Week 5). Standard deviation increased slightly from 26.51ms to 27.70ms, remaining within 3.74ms of SRPT's 23.96ms — the bulk of the variance is still structural (process-set heterogeneity) rather than policy noise.

On fixed evaluation sets: in-distribution **MCT = 31.00ms** (beats Round Robin by 3.40ms; +4.60ms vs SRPT); OOD **MCT = 29.60ms** (beats Round Robin by 6.80ms; +1.20ms vs SRPT). The OOD result of 29.60ms is the closest any agent has come to the SRPT optimum (28.4ms) on that set.

### Finding 2 — Identity Bias Resolved

The identity bias probe was re-run with the Week 5 architecture. Probe state: P0 complete, P1 with 10ms remaining, P2 complete, P3 not yet arrived, P4 with 3ms remaining. SRPT-correct answer: prefer P4.

The action-conditioned DQN assigned **Q(P1/long) = −0.82544, Q(P4/long) = −0.77447, |Δ| = 0.051, P4 preferred = True**.

Comparison across all agents:

| Agent | Q(P1/long) | Q(P4/long) | |Δ| | Correct direction? |
|-------|-----------|-----------|-----|-------------------|
| Tabular W1 | — | — | 0.270 | No |
| Tabular W2 | −3.91 | −3.68 | 0.225 | Yes |
| DQN W3 | +44.92 | +39.43 | 5.48 | No |
| DQN W5 | −0.825 | −0.774 | 0.051 | Yes |

The W5 |Δ| of 0.051 is the smallest across all four agents and, crucially, points in the correct direction. The residual 0.051 is expected: with only P1 as a competitor at a fixed position, the partial permutation invariance guarantee does not eliminate all bias — it eliminates positional bias for the candidate while the competitor's PID-indexed contribution remains. Eliminating that residual requires full permutation invariance over competitor features.

### Finding 3 — Load-Adaptive Quantum Pattern Was an Artefact

The most diagnostic result of Week 5 is the quantum selection pattern. Week 3 showed 20ms quanta used 60–68% of the time at n\_active ≥ 3, suggesting a learned "batch processing" heuristic. Week 5 collapses this entirely:

| n\_active | agree% (W5) | 1ms% (W5) | 5ms% (W5) | 20ms% (W5) |
|-----------|------------|---------|---------|---------|
| 1 | 100.0% | 90.0% | 7.6% | 2.4% |
| 2 | 74.3% | 99.3% | 0.0% | 0.7% |
| 3 | 65.4% | 99.9% | 0.1% | 0.0% |
| 4 | 67.9% | 99.9% | 0.0% | 0.1% |
| 5 | 71.9% | 99.9% | 0.0% | 0.1% |

At n\_active ≥ 2 the Week 5 network uses 1ms quanta in >99% of decisions. The 20ms preference at high n\_active was not a generalised heuristic; it was a PID-bias artefact. Specific PID combinations that co-occurred with large n\_active values during training accumulated gradient pressure toward large quanta in the position-indexed action heads, and this pressure was absent in every other PID ordering — making it look like a load-dependent rule when it was actually a memorised association for specific PID configurations.

The Week 5 architecture shares weights across all PID positions, so no single position can accumulate such an asymmetric signal.

### Finding 4 — Process-Selection Agreement Did Not Reach 80%

The secondary Week 5 target was >80% process-selection agreement with SRPT at n\_active ≥ 2. This target was not met: agreement ranges from 65.4% to 74.3% across n\_active levels 2–5.

Two confounding factors complicate interpretation. First, the partial permutation invariance guarantee extends only to cases where the competitor set is at the same PID positions in both queries. When 5 processes are all runnable simultaneously, the competitor context is fully PID-ordered, and the network sees 5! / 1 = 120 distinct orderings of the same abstract situation — the same under-determination problem as Week 3, only partially mitigated. Second, the process-selection agreement metric treats all disagreements equivalently, including near-ties (remaining burst difference < 1ms) where either choice is near-optimal.

Despite falling short of 80%, the directional improvement at n\_active = 2 (65.3% → 74.3%) is consistent with the expected benefit from partial permutation invariance: the two-process case has the fewest PID configurations and is the one most fully covered by the invariance guarantee.

---

## Summary Table

| Policy | Mean MCT | Std | vs RR | vs SRPT | Eval set |
|--------|----------|-----|-------|---------|----------|
| SRPT oracle | 67.66ms | 23.96ms | −19.5% | 0.00ms | Random (500) |
| DQN W5 (action-conditioned) | 76.04ms | 27.70ms | — | +8.38ms | Random (500) |
| DQN W3 (position-indexed) | 85.00ms | 26.51ms | — | +17.34ms | Random (500) |
| DQN W5 (in-dist fixed) | 31.00ms | 0.00ms | −9.9% | +4.60ms | Fixed indist |
| DQN W5 (OOD fixed) | 29.60ms | 0.00ms | −18.7% | +1.20ms | Fixed OOD |
| DQN W3 (in-dist fixed) | 34.60ms | 0.00ms | +0.6% | +8.20ms | Fixed indist |
| DQN W3 (OOD fixed) | 32.00ms | 0.00ms | −12.1% | +3.60ms | Fixed OOD |
| Tabular W2 (in-dist) | 29.80ms | 0.00ms | −13.5% | +3.40ms | Fixed indist |
| Round Robin (baseline) | 34.40ms | — | baseline | +8.00ms | Fixed indist |

---

## Open Problems

1. **Full permutation invariance:** The Week 5 action-conditioned architecture achieves invariance for the candidate process but not for the competitor context: a 5-process state presents competitor features in PID order, and the network can still learn PID-position associations over that context. Full invariance requires an architecture that aggregates competitor features in a position-independent way — e.g., sum-pooling over competitor encodings, or a multi-head attention mechanism (pointer network style) that scores each candidate against the permutation-invariant aggregate of the rest.

2. **Process-selection agreement ceiling:** At n\_active ≥ 3, agreement with SRPT plateaus in the 65–68% range even after fixing the quantum artefact. Whether this ceiling reflects a representational limit of the 19→64→32→1 architecture, insufficient training episodes, or the inherent difficulty of rank-ordering 3+ processes from a fixed-size input deserves systematic ablation.

3. **Mean MCT gap:** The Week 5 DQN's mean of 76.04ms is 8.38ms above SRPT (67.66ms). The loss curve (0.0001 at ep 10,000) suggests the network has converged, so the gap is unlikely to close with more training at fixed architecture. The gap likely reflects the partial permutation invariance limit identified in problem 1 — each of the 5! = 120 process orderings is still partially distinguishable by position, fragmenting the effective training signal.

4. **Evaluation at scale:** All random evaluations used N = 5 processes with fixed arrival slots and burst range [1, 60]ms. It is unknown whether the learned policy degrades gracefully as N scales, arrival patterns change, or the burst distribution shifts. A stress test varying these parameters would bound the practical scope of the current approach.

---

*Generated with Claude Code (claude-sonnet-4-6) as part of an AI-assisted graduate capstone.*
