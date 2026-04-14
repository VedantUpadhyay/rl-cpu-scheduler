# RL-Based CPU Scheduler — Project Context

## Project

RL-based CPU scheduler — tabula rasa, no hardcoded policy.
Goal: Learn scheduling policy from scratch using RL on process traces.

## Week 1 Scope

Toy environment only: 5 processes, fixed arrivals, tabular Q-learning, variable quantum.

## Constraints

- Language: Python 3.10+
- No ML frameworks (no PyTorch, no TF) — pure numpy + stdlib only
- All modules must be independently testable
- No premature abstraction — keep it flat and readable
- Every component must have a `__main__` block that demos it standalone

## Architecture

Do not deviate from this structure:

```
schedsim/
    env.py        # scheduler environment (state, step, reset)
    agent.py      # tabular Q-learning agent
    process.py    # process data class
    reward.py     # reward function (isolated, swappable)
    runner.py     # training loop + logging
tests/
    test_env.py
    test_agent.py
results/
    (csv logs go here)
```

## Success Criteria (Week 1)

- Agent beats Round Robin on mean completion time after 10k episodes
- No process starves (all complete within episode)
- Training curve is logged and plottable

---

# Claude Code Project Rules

## File Storage — CRITICAL
Never write project files to /tmp.
/tmp is cleared on reboot and is not backed up.

All files must go to permanent locations:

| File type | Permanent location |
|---|---|
| Training scripts (.py) | docs/scheduler-research/scripts/ |
| Checkpoints (.npz) | docs/scheduler-research/results/<agent_name>/ |
| Results (.json, .csv) | docs/scheduler-research/results/<agent_name>/ |
| Training logs (.log) | docs/scheduler-research/results/<agent_name>/ |
| Evaluation outputs | docs/scheduler-research/results/<agent_name>/ |
| Trace data | ~/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/ |
| Paper | docs/scheduler-research/paper.md |
| TODO | docs/ClaudeCode/todo.md |
| README | docs/ClaudeCode/README.md |

## Trace Paths
```
TRACE_PATH = ~/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_train_filtered.csv
TEST_PATH  = ~/Library/CloudStorage/OneDrive-Personal/data/alibaba2018/trace_test_filtered.csv
```

## Script conventions
- Every new script must use `get_agent_dir()` from `project_config.py` for output paths
- Never hardcode /tmp in any path
- Always create destination directory before writing: `os.makedirs(OUT_DIR, exist_ok=True)`
- Import shared config at top: `from project_config import TRACE_PATH, TEST_PATH, get_agent_dir`

## Agent naming convention
```
W10C → w10c/
W11 series → w11/, w11b/, w11c/, w11d/
W12 series → w12/, w12_lambda_tuning/
W13 series → w13_threshold/
W14 → w14_omega/
Future agents → w15/, w16/ etc.
```
