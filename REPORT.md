# System Understanding Report

## 0) Executive Summary
- **What the simulator is:** `schedsim` is a Python 3 discrete-event CPU scheduler implemented in `schedsim/simulator.py`, driven by JSON workloads (`schedsim/io.py`) and pluggable policies (`schedsim/policies.py`). It replays textbook-style arrivals/bursts and records execution timelines plus metrics (`schedsim/metrics.py`).
- **What policies exist:** First-Come-First-Serve, Round Robin, Shortest Job First, Oracle SJF, Predicted SJF (with ML hook), and `cfs_lite`—a simplified Completely Fair Scheduler analog. All derive from the `Policy` base class and register in `POLICIES` (`schedsim/policies.py`).
- **What the CLI does:** `python3 -m schedsim run <workload.json> [flags]` (`schedsim/cli.py`) loads a workload, instantiates the simulator, and prints metrics (text table or JSON). Optional flags expose per-task timelines and raw simulation traces.
- **What verify_poc.py guarantees:** `verify_poc.py` runs a curated suite of CLI invocations plus a micro-simulation to ensure outputs are sane, deterministic under seeds, and that specific invariants (non-negative latencies, RR dispatch count ≥ FCFS, predicted SJF with σ=0 matches oracle) hold before any demo.
- **What “Linux-inspired” means:** The `CFSLite` policy (`schedsim/policies.py`) mimics Linux’s Completely Fair Scheduler by tracking `vruntime`, scaling it by inverse weight (analogous to `nice`), enforcing a uniform slice (`slice_ms`), and always selecting the smallest `vruntime`. It drops Linux’s red-black tree and load tracking but keeps the vruntime fairness intuition.

## 1) Repository Map (file-by-file)
- `README.md`: Minimal run instructions reminding engineers to execute `python verify_poc.py` and expect `DEMO READY`.
- `pyproject.toml`: (implicit) Defines packaging metadata if published; not central to runtime but enforces Python dependency declarations.
- `schedsim/__init__.py`, `__main__.py`: Package glue; `__main__.py` delegates to `schedsim.cli.main`, enabling `python3 -m schedsim` entry point.
- `schedsim/process.py`: Declares the `Task` dataclass and `Workload` wrapper. Handles invariants (non-negative arrivals, positive bursts), tracks dynamic state (`remaining`, `start`, `finish`, waiting accumulator), and exposes helpers (`mark_ready`, `mark_dispatch`, `mark_complete`, `fresh_tasks`). Every other module consumes `Task` objects.
- `schedsim/policies.py`: Base `Policy` plus six concrete policies and the `noisy_predictor`. Each policy stores its own ready structure (deque, heap, list) and implements `pick_next`, optionally `time_slice`/`on_preempted`. `POLICIES` registers them for lookup.
- `schedsim/simulator.py`: Core discrete-event loop. Accepts `List[Task]` and a `Policy`, maintains arrival pointer (`pending`/`future_idx`), running task, time slice budget, context-switch counter, and timeline snapshots. Returns `(completed_tasks, dispatch_count, makespan, timeline)`.
- `schedsim/metrics.py`: Aggregates per-task statistics (avg/p95 wait, turnaround, response, dispatches, makespan). Includes `_nearest_rank` percentile helper and `format_metrics` for CLI text output.
- `schedsim/io.py`: Loads workload JSON, validates required fields, applies default weights/predictions, and constructs `Task` instances. Raises `TestcaseError` for malformed files.
- `schedsim/cli.py`: Argument parsing (`build_parser`), policy-specific kwargs, pretty-print utilities, JSON normalization, and the `handle_run` command invoked by `main`. Acts as a façade over IO → Simulator → Metrics.
- `schedsim/metrics.py`: Already described but also consumed by CLI.
- `schedsim/cli.py`: Already described; also prints optional timelines.
- `schedsim/__pycache__`: Build artifacts (ignored for docs).
- `tests/basic.json`, `tests/demo.json`, `tests/weighted.json`: Example workloads used by CLI and verifier. `basic` stresses mixed arrivals; `demo` drives timeline demo; `weighted` highlights different weights for `cfs_lite`.
- `tests/test_task.py`: Micro-unit test ensuring a new `Task` starts with `remaining == burst`.
- `verify_poc.py`: Automation harness. Defines `CommandSpec`, runs CLI commands via `subprocess`, validates JSON payloads, compares policies, and prints `DEMO READY` or raises `VerificationError`.

### ASCII Data Flow Overview
```
JSON Workload ──▶ schedsim.io.load_testcase ──▶ Workload/Task objects
                                                      │
                                                      ▼
                 CLI args ──▶ schedsim.cli.handle_run ──▶ Simulator.run
                                                             │
                                                             ▼
 Policy.ready queue ⇄ schedsim.simulator.simulate ⇄ Task state updates
                                                             │
                                                             ▼
                           schedsim.metrics.compute_metrics ──▶ CLI/JSON output
```

## 2) Data Model
- **Fields (see `schedsim/process.py:8-65`):**
  - `pid` (str, required) unique identifier.
  - `arrival` (int, required) absolute time; must be ≥0.
  - `burst` (int, required) total service time; must be >0.
  - `remaining` (int, derived) initialized to `burst`, decremented during simulation.
  - `start` (float|None) first dispatch timestamp, set in `mark_dispatch`.
  - `finish` (float|None) completion timestamp, set in `mark_complete`.
  - `weight` (int, default 1024) used only by `CFSLite` to modulate vruntime advance.
  - `vruntime` (float, default 0.0) relative progress metric for `CFSLite`.
  - `predicted` (float|None) optional ML prediction; set either via workload JSON or `PredictedSJF.on_arrival`.
  - Internal trackers: `_waiting_time`, `_last_ready` accumulate queueing delay exposed via `waiting_time` property, though metrics recompute waits directly.
  - `Workload.metadata` (dict) allows scenario annotations; currently unused by simulator but preserved for extension.
- **Required vs optional:** `pid`, `arrival`, `burst` are mandatory per loader checks. `weight`, `predicted`, metadata default if omitted. `start`/`finish` may stay `None` if a task never ran (should not happen under successful simulations).
- **Start/finish/makespan definitions:** `start` is when the task first leaves ready queue (`mark_dispatch`). `finish` is `mark_complete` time. `makespan` is computed in `simulate` as `max(task.finish) - start_time`, where `start_time` is first arrival; idle gaps before first task are excluded but idle gaps between tasks extend makespan because `t` continues advancing.
- **Dispatches vs context switches:** `simulate` increments `context_switches` every time a task is picked (`running = policy.pick_next ...; context_switches += 1`). Metrics report this value as `dispatches`. This counts both initial dispatch and re-dispatch after preemption, but does not differentiate between voluntary yield vs forced preemption; it is not a context-switch duration model.
- **Seeds & randomness:** Only `PredictedSJF` introduces randomness via `noisy_predictor` (`math.exp(rng.gauss(0, sigma))`). `--seed` from CLI becomes predictor RNG seed, ensuring deterministic predictions per workload. The simulator itself is deterministic beyond these predictions.

## 3) Simulator Semantics (core scheduling loop)
`schedsim/simulator.py:10-112` implements a discrete-event simulation with three key structures: `pending` (arrival-sorted list), the policy-managed ready set, and an optional `running` task.

**Event lists**
- `pending`: tasks sorted by arrival; `future_idx` points to next unseen arrival.
- `ready`: entirely owned by the policy; simulator interacts through `policy` hooks only.
- `timeline`: list of snapshots (`{"time": t, "running": pid|None, "ready": [pids]}`) appended whenever the system state changes (start, finish, arrival, idle advance).

**Loop semantics**
1. Initialize `t` to the first arrival and release all tasks whose arrival ≤ `t` via `release_arrivals`, which also calls `task.mark_ready` then `policy.on_arrival`.
2. Snapshot initial state.
3. Repeat until `completed` == `total`:
   - If no `running` task: ask policy for `pick_next` if `policy.has_ready()`. Dispatch sets `task.start` (if first time), increments `context_switches`, sets `slice_budget = policy.time_slice(...)`, and snapshots.
   - If still nothing ready but arrivals remain: jump `t` forward to next arrival, release arrivals, snapshot, and continue (modeling idle CPU).
   - Run `running` for `run_amount = min(slice_budget, task.remaining)` by advancing `t`, decrementing `task.remaining`, and calling `policy.on_run_completed` (for vruntime updates). `release_arrivals(t)` handles arrivals during execution.
   - If `task.remaining <= EPS`: call `mark_complete`, `policy.on_task_done`, append to `completed`, clear `running`.
   - Else if slice budget exhausted: treat as preemption (`mark_ready`, `policy.on_preempted` requeues task).
4. After exiting, compute makespan relative to first arrival; return completed tasks, dispatch count, makespan, and timeline.

**Pseudocode close to implementation**
```
pending = sort_by_arrival(tasks)
t = pending[0].arrival
release_arrivals(t)
snapshot()
while completed < total:
    if running is None:
        if policy.has_ready():
            running = policy.pick_next(t, policy.ready, None)
            running.mark_dispatch(t)
            context_switches += 1
            slice_budget = max(EPS, policy.time_slice(t, running))
            snapshot()
        elif future arrivals remain:
            t = pending[future_idx].arrival
            release_arrivals(t)
            snapshot()
            continue
        else:
            break
    run_amount = min(slice_budget, running.remaining)
    t += run_amount
    running.remaining -= run_amount
    policy.on_run_completed(running, run_amount)
    slice_budget -= run_amount
    release_arrivals(t)
    snapshot()
    if running.remaining <= EPS:
        running.mark_complete(t)
        policy.on_task_done(...)
        completed.append(running)
        running = None
        slice_budget = 0
    elif slice_budget <= EPS:
        running.mark_ready(t)
        policy.on_preempted(running, t)
        running = None
        slice_budget = 0
```

**Timeline semantics:** Each snapshot records the logical time, which PID (if any) is running, and the ready queue state as reported by `policy.ready_state()`. CLI’s `--print-timeline` renders these entries (`schedsim/cli.py:_print_event_timeline`).

## 4) Policy Interface and Implementations

### Base Interface (`schedsim/policies.py:12-64`)
- `on_arrival(task, t)`: default enqueues into `self.ready` deque. Simulator calls it whenever a task becomes ready.
- `has_ready()`: whether policy can supply a task.
- `pick_next(t, ready, running)`: must return next task to run.
- `time_slice(t, task)`: default equals `task.remaining` (non-preemptive). RR/CFS override to enforce quanta.
- `on_run_completed(task, duration)`: hook after each execution chunk (used by `CFSLite` to advance `vruntime`).
- `on_task_done(task, t)`: completion hook (unused but available for logging/stats).
- `on_preempted(task, t)`: default reuses `on_arrival` to requeue.
- `ready_state()`: string list for timeline debugging.

### Concrete Policies
| Policy | Scheduling Rule | Preemption | Data Structure | Complexity | Behavior |
| --- | --- | --- | --- | --- | --- |
| `FirstComeFirstServe` | FIFO order based on arrival enqueue (`deque`). | No; `time_slice = remaining`. | `collections.deque`. | O(1) enqueue/dequeue. | Maximizes throughput but poor latency for late arrivals; zero starvation risk. |
| `RoundRobin(quantum)` | Cycles through ready queue; each run capped at `quantum`. | Yes; `time_slice = min(quantum, remaining)` and `on_preempted` appends to tail. | `deque`. | O(1). | Fair for bursty mixes; higher dispatch count leading to context-switch overhead. |
| `ShortestJobFirst` | Selects smallest `remaining` at arrival time using min-heap. | No (non-preemptive). | `heapq` keyed by remaining, tie-broken via counter. | O(log n) per enqueue/dequeue. | Minimizes average wait but long jobs can starve if steady stream of short arrivals. |
| `OracleSJF` | Same as SJF but conceptually uses perfect knowledge (identical implementation). | No. | `heapq`. | O(log n). | Baseline for predictive policies. |
| `PredictedSJF` | On arrival, compute/consume `task.predicted` (existing or via predictor) and order by predicted burst. | No. | `heapq` keyed by predicted length. | O(log n) plus predictor cost. | Behavior depends on predictor accuracy; low σ approximates Oracle; noisy predictions risk mis-ordering, increasing wait time but still starvation-prone for mispredicted long jobs. |
| `CFSLite(slice_ms)` | Tracks per-task `vruntime`, always running task with minimal `vruntime`. After each slice, increments by `duration * (1024 / weight)` so heavier weights (smaller numbers) run proportionally longer. | Yes; `time_slice = min(slice_ms, remaining)` and preemptions reinsert into array list. Selection scans list for min vruntime. | Python list + linear `min` search (O(n)). | Approximates Linux CFS fairness; mitigates starvation by design but less precise due to linear search and fixed slice. |

**Oracle CLI mode:** Selecting `--policy oracle_sjf` routes through `POLICY_CHOICES` aliasing to `oracle_sjf`. CLI encloses display name vs policy key.

## 5) CFS-lite Deep Dive
- **Vruntime update:** `on_run_completed` multiplies elapsed `duration` by `(1024.0 / weight)` and adds to `task.vruntime`. Thus a task with weight 2048 accumulates vruntime at half the rate of weight 1024, approximating proportional CPU shares.
- **Weight meaning:** Equivalent to Linux’s load weight (scaled by 1024). Heavier `weight` ⇒ smaller vruntime increment ⇒ more CPU share. Default weight 1024 yields 1× rate.
- **Slice/quantum selection:** `slice_ms` (default 3 ms) mimics Linux’s `sched_slice`: a target runtime per entity proportional to total load. Here it is a fixed constant; tasks run at most this long before reevaluation, enforcing periodic fairness.
- **Missing features vs Linux CFS:** No red-black tree (O(log n) selection); no per-CPU run queue, load balancing, or `nice` translation table; no decay/invruntime shift to prevent wraparound; no sleeping fairness adjustments.
- **Mapping Table:**
  - Linux `se.vruntime` ⇨ `Task.vruntime`
  - Linux load weight (`se.load.weight`) ⇨ `Task.weight`
  - Linux `sched_slice` ⇨ `CFSLite.slice_ms`
  - Linux RB-tree ordered by vruntime ⇨ Python list scanned for min
  - Linux CFS clock ⇨ simulator time `t`

## 6) ML Hook (PredictedSJF + noisy_predictor)
- **Prediction site:** `PredictedSJF.on_arrival` checks `task.predicted`. If absent, it calls `self.predict_fn(task)` and caches the float on the task so subsequent clones reuse it.
- **Predictor interface:** Callable `Task -> float`. Default built via `noisy_predictor(sigma, seed)` returning log-normal noise scale of true burst (`burst * exp(N(0, sigma))`). `sigma` tunes variance; `seed` seeds the RNG for reproducibility.
- **Replacing with real model:** Provide `PredictedSJF(predictor=my_model)`. A RandomForest wrapper would implement:
  ```python
  def predict_task(task: Task) -> float:
      features = [task.arrival, task.weight, task.burst, task.metadata.get("type", 0)]
      return max(EPS, rf.predict([features])[0])
  ```
  That function respects the interface and can ingest richer metadata.

## 7) CLI Contract
- **Commands:** Only `run` subcommand (`schedsim/cli.py:18-53`). Syntax: `python3 -m schedsim run tests/basic.json --policy rr --quantum 2 --format text`.
- **Flags:**
  - `--policy {fcfs, rr, sjf, predicted_sjf, cfs, oracle_sjf}` (`cfs` alias resolves to `cfs_lite`).
  - `--quantum <float>`: Round Robin quantum (default 4.0 when omitted).
  - `--slice <float>`: `cfs_lite` time slice (default 3.0 ms).
  - `--sigma <float>` and `--seed <int>`: forwarded to `PredictedSJF`.
  - `--format {text,json}`: choose pretty output vs machine output.
  - `--timeline`: dumps per-task intervals.
  - `--print-timeline`: dumps simulator event snapshots.
- **Example JSON output:**
  ```json
  {
    "workload": "basic",
    "policy": "fcfs",
    "metrics": {
      "avg_waiting": 8.75,
      "p95_waiting": 18.0,
      "avg_turnaround": 15.25,
      "p95_turnaround": 23.0,
      "avg_response": 8.75,
      "dispatches": 4,
      "makespan": 26.0,
      "completed_tasks": 4
    }
  }
  ```
- **Timeline flag interaction:** `--format json` suppresses timeline printing (per `handle_run`, only JSON payload is printed). To view timeline simultaneously, run twice: once with `--format text --print-timeline`, once with JSON for automation. This avoids mixing structured output with debug text.
- **`--seed` behavior:** Provided seed feeds predictor RNG ensuring reproducible scheduling order when noise exists. Other policies ignore the seed flag.

## 8) Metrics Definitions (`schedsim/metrics.py`)
- **Waiting time:** `start - arrival` (time before first CPU slice). Equivalent to `response` here because there is no think time.
- **Turnaround time:** `finish - arrival` (total system time). Always ≥ burst because it includes waiting.
- **Response time:** Same calculation as waiting (first response). Kept separate for clarity/extensions.
- **P95 calculation:** `_nearest_rank` sorts values, computes `ceil(0.95 * n)`, and picks that rank (1-indexed). For small samples this equals the max when `n ≤ 20`.
- **Makespan:** Provided by simulator (last finish - first arrival). Idle periods between tasks increase makespan; idle before first arrival does not because simulation starts at earliest arrival.
- **Dispatches:** Count returned by simulator (`context_switches`), effectively number of times CPU begins executing a (possibly repeated) task.
- **Corner cases:** Empty workload yields zeros for all metrics but retains dispatch count passed in. If CPU idles due to no ready tasks, waiting time for yet-to-arrive tasks is unaffected because arrival time gating prevents negative waits.
- **Comparability:** Dispatch counts between preemptive and non-preemptive policies differ; only relative comparisons (e.g., RR vs FCFS) are meaningful. Wait/turnaround/responses are comparable because simulator semantics are identical across policies.

## 9) Verification Harness (`verify_poc.py`)
- **Purpose:** Ensure regressions are caught before demos by exercising CLI, verifying metric invariants, and confirming deterministic relationships between policies.
- **Commands executed:** Nine `CommandSpec`s cover `basic` and `demo` workloads with FCFS, SJF, RR, CFS-lite, Predicted SJF (σ=0 and σ=0.35 seeded), Oracle SJF, plus timeline-enabled runs for demo scenarios. Each command uses `python <sys.executable> -m schedsim run ... --format json ...`.
- **Invariants checked:**
  - `verify_metrics`: makespan must exceed last arrival, latency metrics non-negative, avg turnaround ≥ avg waiting, dispatches ≥ number of tasks, makespan ≥ total burst duration.
  - `run_micro_test`: Direct `simulate` call ensures recording of `start`/`finish`, monotonically increasing times, turnaround ≥ burst, dispatch count ≥ completed tasks, and makespan ≥ last finish.
  - `compare_rr_vs_fcfs`: Round Robin dispatches must be at least FCFS dispatches on same workload, reflecting extra preemptions.
  - `compare_predicted_vs_oracle`: With `sigma=0`, Predicted SJF predictions equal actual burst, so metrics must match Oracle SJF exactly; any drift signals bug in prediction caching or ordering.
- **RR ≥ FCFS check nuance:** Could fail if RR quantum ≥ sum of bursts (i.e., no preemptions). Here quantum=2 against varied bursts, so RR necessarily has ≥ dispatches; future workloads should maintain that relationship or adjust check.
- **Deterministic micro-test:** Uses direct Task list and `FirstComeFirstServe` to ensure simulator invariants independent of CLI/IO.
- **Limitations:** Harness does not exhaustively test all edge cases (e.g., zero-length idle gaps, huge weights, timeline serialization). Policies other than RR/Predicted might regress silently if invariants do not capture their behavior.

## 10) “20% Professor Demo” Script
1. **Show readiness:** `python3 verify_poc.py` → highlight successive command checks and final `DEMO READY` banner. Emphasize this means policies, metrics, and CLI interoperate.
2. **Inspect workload:** `cat tests/demo.json` → explain arrival/burst fields.
3. **Run FCFS text report:** `python3 -m schedsim run tests/demo.json --policy fcfs --format text --timeline` → point to formatted metrics and per-task start/finish list.
4. **Contrast with RR:** `python3 -m schedsim run tests/demo.json --policy rr --quantum 2 --format text` → discuss increased dispatches but lower tail latency.
5. **Preview Linux-inspired mode:** `python3 -m schedsim run tests/demo.json --policy cfs --slice 2 --print-timeline --format text` → scroll event timeline showing vruntime-driven fairness.

**Speaking notes:**
- Highlight JSON-to-simulator pipeline and how policies swap via flag.
- Emphasize metrics table readability (avg vs P95) for grading fairness.
- Point at timeline output to demonstrate debugging visibility.
- Mention ML-ready `predicted_sjf` for research extensions.
- Close by reiterating `verify_poc.py` as regression safety net.

## 11) Extension Plan: Online Learning / RL
- **Current state:** Policies are offline/deterministic—no feedback loop except `PredictedSJF` predictions at arrival time. Simulator never adapts weights mid-run.
- **Needed hooks:**
  1. Add `Policy.on_dispatch(task, t)` and `Policy.on_complete(task, t)` so learners observe runtime outcomes.
  2. Extend `pick_next` signature (already receives `t`, ready set, running task) to also pass richer global state (e.g., total load, recent wait stats) or expose via new `SimulatorContext` object.
  3. Record reward as negative waiting time or slowdown per decision. Overhead budget can be measured by counting policy invocations and limiting per-call compute (hook instrumentation).
- **RLPolicy skeleton:**
  ```python
  class RLPolicy(Policy):
      def __init__(self, learner):
          super().__init__()
          self.learner = learner
      def pick_next(self, t, ready, running):
          state = self._encode_state(t, ready)
          action = self.learner.select_action(state)
          return ready[action]
      def on_run_completed(self, task, duration):
          reward = -task.waiting_time
          self.learner.update(reward)
  ```
  Actions could be “choose task index” or “set slice length.” Simple contextual bandit framing (state = task features, action = index) allows online adaptation without full value iteration. An initial incremental improvement is to learn weights for CFSLite by updating `task.weight` based on observed wait ratios.

## 12) Known Limitations / Technical Debt
- **Simulator fidelity:** No CPU/cache warmup, zero cost context switches, no I/O blocking, and single-core assumption. Real Linux adds per-core run queues and scheduling classes.
- **Performance:** Policies like `CFSLite` scan the ready list linearly; large workloads would benefit from trees/heaps.
- **Metrics coverage:** Lacks jitter, slowdown, or throughput metrics; dispatch count alone underestimates scheduler overhead.
- **IO validation:** JSON loader lacks schema enforcement for metadata/predicted fields beyond basic typing.
- **Closer-to-Linux upgrades:** Would need RB-tree implementation, `nice` to weight mapping, periodic load balancing, and sleeping fairness updates. Could validate accuracy by capturing real process traces via eBPF (`sched_switch` events) and replaying them to compare waiting/turnaround distributions.

