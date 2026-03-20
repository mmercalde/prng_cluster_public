# S149 Chat Prompt — PRNG Distributed Analysis System

## Session Priority

**P1 — Start here:**
1. Monitor / triage production sweep Run 1 (launched end of S148)
2. Q0 zero-survivor branch live verification (TB P2 item from S147)
3. Chapter 13 wire-up — `dispatch_selfplay()` + `dispatch_learning_loop()` into WATCHER

---

## Current System State (end of S148)

**All commits synced at `<commit_after_s148_push>` on both remotes.**

### What was completed in S148

**Empirical threshold calibration — first production sweep under empirical threshold governance.**

Two calibration scripts (`ca_d3_threshold_calibration.py`, `ca_d3_window_calibration.py`)
ran on Zeus RTX 3080 Ti against real `sieve_gpu_worker.py` IPC path. Replaced all
synthetic-era threshold defaults across 5 files + manifest.

| Experiment | Key finding |
|---|---|
| Skip sweep (window=8, skip=0/3/5/10/20) | Zero-noise floor: 0.30–0.40 depending on skip |
| Window sweep (skip=5, window=8/10/12/16) | W12+T0.30 = ~1 false survivor per 200k |

**S148 Run-1 ruling:** `window_size` promoted 8 → 12. Retain `threshold=0.30`.
Preserve Optuna bounds `[0.30, 0.75]`.

### Patches applied in S148

| Patch script | Files | Checks |
|---|---|---|
| `apply_s148_threshold_calibration.py` | PWC, window_optimizer, distributed_config, baselines/, THRESHOLD_GOVERNANCE.md | 15/15 ✓ |
| `apply_s148_manifest_update.py` | agent_manifests/window_optimizer.json (informational) | 2/2 ✓ |
| `apply_s148_w12_promotion.py` | distributed_config, agent_manifests, baselines/ | 10/10 ✓ |

### Run 1 configuration — final (as launched)

| Parameter | Value |
|---|---|
| `window_size` | **12** |
| `threshold` (fwd/rev) | **0.30** |
| Optuna bounds (fwd/rev) | **[0.30, 0.75]** |
| `max_seeds` | 1,073,741,824 |
| `window_trials` | 50 |
| `use_persistent_workers` | true |
| Q0 hybrid gate | active (S147) |
| Q1 Step 1 timeout | infinite (S147) |
| Q2 hybrid strategy | `balanced_hybrid` only (S147) |

### Expected Run 1 performance
- ~5 false fwd survivors/trial (vs ~1,360,000 at old W8/T0.25)
- ~250 false fwd survivors total across 50 trials (vs ~68,000,000)
- Per-trial time estimate: ~2.5–3 hrs (Q2 single strategy) → ~125–150 hrs total
- Q0 gate may fire if `balanced_hybrid` fwd survivors = 0 on some trials

---

## Infrastructure

**Zeus:** `rzeus`, `~/distributed_prng_analysis/`, `~/venvs/torch/bin/activate`
**Rigs:** `rrig6600` (192.168.3.120), `rrig6600b` (192.168.3.154), `rrig6600c` (192.168.3.162)
**Dashboard:** `45.32.131.224:5002`
**Git:** dual-push `origin` + `public` always

### Monitor Run 1
```bash
tail -f logs/sweep_run1_production.log
# or
open http://45.32.131.224:5002
```

### Resume if crashed
```bash
bash sweep_run1.sh --resume
```

### Kill all workers
```bash
ssh rzeus "pkill -f 'watcher_agent.py'; pkill -f 'window_optimizer.py'"
ssh rrig6600 "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600b "pkill -f sieve_gpu_worker 2>/dev/null"
ssh rrig6600c "pkill -f sieve_gpu_worker 2>/dev/null"
```

### After Run 1 completes — commit NPZ
```bash
git add -f bidirectional_survivors_all.npz bidirectional_survivors_binary.npz
git commit -m "data(s149-r1): sweep run 1 complete — W12/T0.30 empirical governance"
git push origin main && git push public main
```

---

## P1 TODO (S149 order)

1. **Run 1 monitor / triage** — Check log, dashboard. If crashed: diagnose + resume.
   If completed: commit NPZ, record survivor count.
2. **Q0 zero-survivor branch live verification** — Monkeypatch `run_trial_persistent()`
   so `java_lcg_hybrid` returns `[]`, exercise the skip-Pass-4 branch live without GPUs.
3. **Chapter 13 wire-up** — `dispatch_selfplay()` + `dispatch_learning_loop()` into WATCHER
   post-Step-6 exit path. This is the critical path to closing the autonomous feedback loop.

## Backlog (priority order)
- phase2_threshold calibration (S148 deferred — hybrid passes, currently 0.50)
- Selfplay NN two-part fix (remove forbidden guard + y-normalization in inner_episode_trainer.py)
- S110 root cleanup (884 files)
- sklearn warnings in Step 5
- Remove CSV writer from coordinator.py
- Regression diagnostics gate → True
- S103 Part 2
- Phase 9B.3 (deferred)
- sweep_run2.sh, sweep_run3.sh, sweep_run4.sh

---

## NPZ Accumulator Status (as of S147)
- `bidirectional_survivors_all.npz` — 666 seeds, 22 fields
- `bidirectional_survivors_binary.npz` — 666 seeds (Step 2-6 input)
- Coverage: 0→510M (prior) + 560M→610M (S146 preprod)
- Run 1 range: 0 → 1,073,741,824 (fresh full range, W12/T0.30)

---

## Key architectural notes for S149

### Why Q0 branch verification matters
In S147, hybrid forward found 4 survivors so the gate didn't fire. TB flagged the
skip-Pass-4 code path as source-verified only. A monkeypatched live run would close
this gap without consuming GPU time. Use the existing `verify_q0_hybrid_gate.py` as
the harness base — add mock patch forcing fwd hybrid result to `[]`.

### Chapter 13 wire-up scope
`dispatch_selfplay()` and `dispatch_learning_loop()` stubs exist in `agents/watcher_agent.py`.
They are not triggered after Step 6 completes. Wire them into the post-Step-6 exit path
in `agents/watcher_agent.py`. The Chapter 13 orchestrator (`chapter_13_orchestrator.py`)
exists and is standalone — needs to be called by WATCHER, not invoked manually.
Cold-start gate: ≥15 real draws + ≥10 episodes + ≥1 promoted policy (Strategy Advisor).
