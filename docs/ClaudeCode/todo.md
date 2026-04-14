# Pre-Submission TODO List

_2026-03-19_

---

## BLOCKER — Must fix before arXiv submission

0. **Hard threshold starvation penalty (§5.32 experiment)**
   - λ tuning (§5.31) proved continuous PBRS cannot close the starvation gap
   - Hypothesis: binary penalty firing only when max_wait > threshold T
     replicates MLFQ's aging promotion rule as a learned behavior
   - Reward: R = value_delta + α × 𝟙[max_wait > T]  (negative bonus at threshold)
   - Sweep: T ∈ {30s, 50s, 100s} × α ∈ {−0.5, −1.0, −2.0}, 3 seeds each
   - Target: starvation < 36% AND MCT < 21.59s (beat MLFQ on both)
   - Write §5.32 in paper.md once results are in

1. **Multi-seed experiments**
   - Run W10C with seeds 42, 123, 456 (minimum 3 seeds)
   - Report mean ± std of MCT and SRPT agreement across seeds
   - Add results to §5.20 with error bars
   - This addresses the single most likely reviewer objection
   - Estimated time: 20–30 minutes

2. **Add figures (paper currently has zero)**
   - (a) AttentionDQN architecture diagram
   - (b) Learning curve for W9/W10C (MCT vs episodes)
   - (c) Attention weight heatmap showing head specialization
       (head 1 = context/survey, head 2 = longest-monitor)
   - (d) Alibaba burst distribution before/after filtering
   - (e) VRFI comparison bar chart across 5 policies
   - Estimated time: 1–2 hours

3. **Add author name and affiliation**
   - Currently only "UCSC Graduate Capstone" listed
   - Add full name and program
   - Quick fix, 2 minutes

4. **Resolve SCHED citation [4]**
   - Currently marked [citation needed] in §2.3 and §6.3
   - Either find the correct citation or remove dependency
   - Quick fix once located

---

## SIGNIFICANT — Strengthens paper considerably

5. **Restructure §5 from chronological to thematic**
   - Current: Week-by-week lab notebook format
   - Target: Three thematic groups:
     - (a) Synthetic environment results (W1–W7)
     - (b) Real-trace transfer results (W8–W10C)
     - (c) Value-curve theory and fairness (§5.21–§5.30)
   - Move weekly narrative to appendix
   - Addresses "lab notebook not a paper" objection
   - Estimated time: 2–3 hours

6. **Rewrite abstract to 200 words**
   - Currently done but verify word count is actually ≤200
   - Check it leads with problem not chronology
   - Quick verification pass

7. **Add EEVDF to related work §2**
   - One paragraph: EEVDF replaced CFS as default Linux
     scheduler since kernel 6.6
   - Explain how it relates to this paper's CFS framing
   - Quick fix, 15 minutes

8. **Specify MLFQ and CFS baseline configurations in §5.27.3**
   - MLFQ: queue count, quantum sizes, aging parameters
   - CFS: virtual runtime granularity
   - Essential for reproducibility
   - Quick fix, 10 minutes

9. **Address plateau edge case in Corollary §3.1.3**
   - When d_i is at or past plateau, f_total is zero
     and monotonicity claim is vacuous
   - Add a brief note acknowledging this case
   - Quick fix, 5 minutes

10. **Fix section numbering gaps in §5**
    - §5.23 appears after §5.21, skipping §5.22 in
      linear order
    - Renumber all subsections sequentially
    - Quick fix, 10 minutes

11. **Add ethical considerations paragraph**
    - Who sets value curves in practice?
    - Fairness implications of τ assignment
    - Which tasks get steep vs smooth curves?
    - Quick fix, 15 minutes

---

## FUTURE WORK — Next research directions (post-submission)

12. **Option B — PBRS with λ tuning**
    - W12 used λ=0.01, too weak to prevent starvation
    - Try λ=0.05, 0.1, 0.5 to find starvation threshold
    - May close the gap between W10C and W12

13. **Preference-Conditioned Agent (ω in State)**
    - Current situation: separate trained agents for each
      objective. W10C optimizes value-delta. W11 optimizes
      fairness. Switching objectives requires retraining.
    - Proposed: add preference vector ω = [w1..w5] directly
      to the state vector at every step. Agent learns a single
      policy π(s, ω) conditioned on the current preference.
    - Training: sample ω randomly each episode from a Dirichlet
      distribution. Agent sees all preference combinations
      during training and learns the full Pareto frontier.
    - Inference: change behavior by changing ω at runtime.
      No retraining, no model swap.
    - Example operating modes (no retraining needed):
      - Batch window:     ω = [0, 0.25, 0.25, 0.25, 0.25]
      - Interactive peak: ω = [1.0, 0, 0, 0, 0]
      - Balanced:         ω = [0.6, 0.1, 0.1, 0.1, 0.1]
    - Connection to theory: the Pareto frontier (§5.22) shows
      two non-dominated policies — W10C and VC oracle. A
      preference-conditioned agent learns the entire frontier
      between them in one training run.
    - Connection to Temporal: their `fairness_weight` parameter
      lets operators dial between efficiency and fairness at
      runtime. This is the RL equivalent — same concept,
      learned rather than hand-tuned.
    - Implementation: expand state from 35-dim to 40-dim
      (add 5 weight dimensions). Sample ω ~ Dirichlet(α=1)
      each episode. Everything else identical to W12.
    - Expected impact: single deployable model covering all
      operating modes. Closes the Pareto gap from §5.22.
      Strongest practical contribution of the entire project.
    - Risk: medium. Training slower (larger state, more
      diverse episodes). Convergence needs ~3× more episodes.
      But no architectural changes required beyond state
      expansion.

14. **Learned burst prediction**
    - Train lightweight estimator on task-type features
    - Use predicted burst to close W10C/W12 gap (41.4s)
    - More realistic than oracle, more informative than nothing

15. **Neural value functions**
    - Replace fixed exponential V(d) = base × exp(−d/τ)
    - Small neural network learns curve shape from data
    - Removes last human prior from the pipeline

16. **Full transformer architecture**
    - W10C is a restricted transformer (2-head attention)
    - Scale to full transformer jointly modeling:
      value curves, queue state, scheduling history
    - Natural extension of current architecture

17. **LLM-based policy discovery**
    - Zero-shot: prompt open-source LLM with queue state
    - Fine-tuned: train on scheduling trajectories
    - Question: does LLM discover heuristics humans haven't?
    - Strictly open-source models (Llama, Mistral, Phi-3)

18. **Temporal connection — systems paper**
    - Temporal's fairness key mechanism is production
      evidence of queue-global necessity
    - Write short systems paper: our theory formalizes
      why their design is correct
    - Target venue: OSDI/EuroSys workshop

19. **Value curve optimization with regularization**
    - Loss A needs: fix base=1.0, floor=0.2, optimize τ only
    - Loss B (VRFI) already works with calibrated init
    - Run the fully constrained Loss A experiment

20. **Multi-seed W12 and W11 series**
    - W11/W12 also single seed
    - 3 seeds each would complete the statistical picture

---

## COMPLETED — For reference

- [x] Quantum degeneracy proof and ablations (§3.1.3, §5.23)
- [x] Gradient alignment / tau calibration (§5.26)
- [x] VRFI definition and JFI/SDV falsification (§5.27.2)
- [x] Five-policy fairness table (§5.27.3)
- [x] Curve optimization Loss A/B (§5.28)
- [x] W11 ablation series documented (§5.29)
- [x] W12 PBRS documented (§5.30)
- [x] 2×2 ablation table in paper (§5.30.4)
- [x] References section with 9 entries
- [x] Abstract rewritten to 200 words
- [x] 15 contributions in introduction
- [x] Cross-references verified
- [x] Conclusion updated with W11/W12 findings
- [x] README.md in docs/ClaudeCode
- [x] All scripts moved from /tmp to docs/ClaudeCode
- [x] Multi-seed ablation rerun on Alibaba 2018 trace (items 12/20) — W11/W11b/W11c/W11d/W12, 3 seeds × 10k episodes; results in results/ablation_multiseed_results.json; §5.29.3 and §5.30.4 updated with mean±std
- [x] MLFQ benchmark (item 8) — W12 (seed 456) vs MLFQ/RR/CFS-lite (Condition 1) and MLFQ-V/value_aware (Condition 2), N=500 episodes; W12 beats MLFQ on MCT (+0.41s) but loses on starvation (53.2% vs 36.0%); script at docs/scheduler-research/scripts/benchmark_vs_mlfq.py
- [x] PBRS λ tuning experiment (item 12/P3) — W12-λ1/λ2/λ3/λ4 (λ=0.05/0.10/0.25/0.50), 3 seeds × 10k episodes; no λ beats MLFQ on both MCT and starvation simultaneously; best trade-off W12-λ4 (λ=0.50): MCT=21.54s, Starve=51.6%; results in docs/scheduler-research/results/w12_lambda_tuning/lambda_tuning_results.json
- [x] Variable-N generalization eval (§5.34) — W14-ω (ω=0.7, seed 123) evaluated on Poisson arrival streams at ρ∈{0.3,0.7,0.9}; W14-ω ties MLFQ at ρ=0.3 (14.58 vs 14.62s), loses at ρ=0.7 (+0.77s), loses significantly at ρ=0.9 (+6.46s); OOD% exceeds 20% at ρ≈0.9; root cause: 5-slot window truncation, not architectural failure; results in docs/scheduler-research/results/w14_variable_n/
- [x] W15 variable-N training (§5.35) — W14-ω architecture retrained on Poisson arrivals (λ~U(0.01,0.08), 300 completions/ep, 20k eps, 3 seeds); beats MLFQ on MCT across all seeds (mean −1.40s); starvation 40–48% worse than MLFQ (38%); stability fixes: TD clamp ±50, Huber loss, TARGET_UPDATE_FREQ=5, LR=1e-4; checkpoints at results/w15_variable_n/; PyTorch (W15Trainer) replaces NumPy network

---

## PIVOT DIRECTIONS — Capstone to Production/Research Paper

P1. **Drop Tabula Rasa — Embrace Domain Knowledge**
    - Current approach: purely tabula rasa, no domain features
    - Proposed change: add `io_wait_ratio` and `historical_cpu_usage`
      as standard state features alongside existing observable ones
    - Rationale: W11/W12 ablation proved burst time (domain
      knowledge) is load-bearing. Pretending to avoid domain
      knowledge while using `task_class` and `plan_cpu` is inconsistent.
    - Reframe contribution as: RL augments scheduler heuristics,
      not replaces them from scratch.
    - Expected impact: MCT improvement, more honest framing,
      stronger practical relevance claim.
    - Risk: low — features are standard, well-defined, observable.

P2. **Variable-N Retraining (Poisson Arrival Stream)**
    - Evaluation done (§5.34): W14-ω ties MLFQ at ρ=0.3 but loses
      at ρ≥0.7. Root cause: 5-slot window truncation, not
      architectural failure. FiLM and attention mechanism are sound.
    - Proposed: retrain W14-ω with Poisson arrival stream (same
      λ range: ρ∈{0.3,0.7,0.9}), variable N at each decision.
    - Architecture change: only input padding logic changes.
      FiLM omega conditioning, 2-head attention, and Pareto-frontier
      training protocol all remain identical.
    - Expected impact: close the 6.46s MCT gap at ρ=0.9;
      agent learns queue-size-invariant policy; single model
      valid across batch and continuous-stream regimes.
    - Evaluation script: w14_variable_n_eval.py already written —
      reuse for post-training comparison.
    - Risk: low — no architectural changes required.

P3. **Hybrid Reward — Value-Delta + PBRS Starvation Floor**
    - Current: W12 used λ=0.01 PBRS, too weak
    - Proposed: keep value-delta as dominant signal, add PBRS
      specifically as a starvation floor:
        φ(s) = −λ × max(time_since_last_execution)
    - Tune λ ∈ {0.05, 0.1, 0.5} to find threshold where
      starvation drops below 5% without MCT regression.
    - Rationale: PBRS has theoretical guarantee — does not
      change optimal policy, only shapes learning path.
      W12 proved the mechanism works directionally (SRPT
      agreement recovered to 46% vs W11's 36%).
    - Risk: low — one hyperparameter sweep, ~1 hour training.

P4. **Publish Diagnostics as Standalone Library**
    - Tools to release independently of the paper:
      - (a) Identity Bias Probe — detects degenerate Q-value
            collapse in any set-input RL agent. General tool,
            not CPU-scheduling specific.
      - (b) VRFI Metric — value-rate fairness index for any
            scheduler operating on tasks with explicit value
            curves. Computable from task logs alone.
      - (c) Permutation Invariance Test — unit test for any
            attention-based RL agent. Verifies policy is
            independent of input ordering.
    - Library name suggestion: `schedeval`
    - Release as: pip-installable Python package on PyPI
    - Target citation: independent of the main paper,
      cited by anyone building value-aware or RL schedulers.
    - Risk: low — code already exists, needs packaging only.
