# TODO_MASTER_S150
**Updated:** 2026-03-21  
**Run 1 status:** Active — study `window_opt_1774109563`, Trial 8+

---

## P0 — WATCHER --force-step flag

**Problem:** WATCHER freshness check skips Step 1 if `optimal_window_config.json`
exists from a prior run. Every resume requires manually deleting the file first.
This has blocked 6+ launch attempts across S148-S150.

**Fix:** Add `--force-step N` flag to `watcher_agent.py` that bypasses freshness
check for the specified step. Alternatively, add `--force` flag to `sweep_run1.sh`
that deletes the output before launching.

**Files:** `agents/watcher_agent.py`, `sweep_run1.sh`, `sweep_run2.sh` etc.

---

## P1 — sweep_run2.sh

**Design:** After Run 1 completes all 50 trials:
1. Read `optimal_window_config.json` best params from Run 1
2. Create new Optuna study (fresh, no add_trials — scores are range-specific)
3. `enqueue_trial()` with Run 1 best params as trial 0 (warm-start)
4. Launch with `seed_start` auto-advanced by coverage tracker
5. Save new study name to `logs/sweep_run2_study_name.txt`

**Note:** `add_trials()` from prior study NOT viable — scores are seed-range-specific
and would poison the TPE model with non-comparable evidence.

---

## P2 — Remove legacy dict-list coordinator parser

After one full production run confirms slim_v1 stable, remove the legacy
path from `_dispatch_to_worker()` in `persistent_worker_coordinator.py`.

**Gate:** Run 1 completes 50 trials with no legacy-path fallbacks observed in logs.

---

## P3 — GPU utilization measurement Pass 3/4

Confirm 22x improvement is sustained across full hybrid pass, not just rolling
average artifact. Use `rocm-smi --showuse` at 1s intervals during Pass 3/4.

```bash
ssh rrig6600 "while true; do rocm-smi --showuse 2>/dev/null | grep 'GPU use'; echo '---'; sleep 1; done"
```

---

## Backlog (unchanged from S149)

- S110 root cleanup (884 files in project root)
- sklearn warnings in Step 5
- Remove dead CSV writer from `coordinator.py`
- Regression diagnostics gate → set to True
- Chapter 13 wire-up: `dispatch_selfplay()`, `dispatch_learning_loop()`
- Selfplay NN two-part fix (remove forbidden guard + y-normalization in `inner_episode_trainer.py`)
- Walk-forward simulation
- vmap batching for selfplay (deferred until NN stabilized)

---

## Completed This Session (S149+S150)

- [x] S149-B: Direct Device(gpu_id) — AMD 4-worker ceiling removed (46x throughput gain)
- [x] Manifest success_condition fix (bidirectional_survivors.json removed)
- [x] Per-trial NPZ checkpoint (eliminate end-of-run data loss)
- [x] NPZ checkpoint wiring fix (strategy on BayesianOptimization not WindowOptimizer)
- [x] NPZ checkpoint order fix (strategy assigned before _survivor_accumulator)
- [x] slim_v1 IPC serialization (22x Pass 3/4 throughput gain, TB 5-round review)
- [x] Run 1 fresh start with clean study, survivors safe from Trial 1
