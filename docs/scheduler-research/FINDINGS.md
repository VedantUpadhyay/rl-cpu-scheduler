# Research Findings

---

## N-Scaling Experiment (May 11, 2026)

**Setup:** W15OmegaDQN trained for N ∈ {5, 10, 20, 50} processes × 5 seeds = 20 GPU runs on Nautilus.  
20 000 episodes per run. Full results: `results/n_scaling/`

### Results

| N  | Mean MCT (s) | Std (s) | CV (%) | MCT / task (s) |
|:--:|:------------:|:-------:|:------:|:--------------:|
|  5 |   21.67      |  1.44   |  6.6   |  4.33          |
| 10 |   39.02      |  2.33   |  6.0   |  3.90          |
| 20 |   74.48      |  1.67   |  2.2   |  3.72          |
| 50 |  176.91      |  5.69   |  3.2   |  3.54          |

### Key findings

- **Per-task MCT improves with N:** 4.33 s → 3.54 s (−18 %) as queue grows from 5 to 50.  
  Larger scheduling windows expose more scheduling opportunities per step.
- **Seed variance stays low:** CV ≤ 6.6 % at N=5 and falls to ≤ 3.2 % for N ≥ 20, confirming reliable convergence.
- **Architecture generalises at runtime:** W15 attention weights (Q/K/V: Linear(7, 8)) work for any N without architectural changes — N is purely a runtime parameter.
