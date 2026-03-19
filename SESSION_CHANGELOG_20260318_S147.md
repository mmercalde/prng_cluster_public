# SESSION CHANGELOG — March 18, 2026 (S147)

**Focus:** Documentation updates, production sweep launch, Q0/Q1/Q2 TB ruling patches
**Outcome:** 5 chapter docs updated, sweep launched and diagnosed, 3 critical patches
deployed and verified 19/19 on live hardware. Both remotes synced at `bf2549b`.

---

## Summary

Extended session covering documentation updates from S146, production sweep Run 1
launch and failure diagnosis, Team Beta ruling on hybrid sieve runtime architecture,
and deployment of three patches addressing the root causes. All patches verified with
a live harness (29/29 mock tests) and live hardware verification (19/19 on Zeus).

---

## Documentation Updates (P1-1)

Applied S146 PWC invariants to 5 chapter files via `apply_s146_doc_updates.py`.
All 5 patches applied cleanly (258 insertions). Committed `9ca2671` + `c797045`.

| File | Change |
|------|--------|
| `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` | PWC architecture invariants section |
| `CHAPTER_1_WINDOW_OPTIMIZER.md` | Hybrid kernel arg tail invariants |
| `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | PWC execution path + hybrid routing |
| `CHAPTER_12_WATCHER_AGENT.md` | Step 1 PWC dispatch path |
| `COMPLETE_OPERATING_GUIDE_v2_0.md` | PWC operating procedures section |

Also added `STEP1_EXECUTION_FLOW_AND_PRUNING_S147.md` — full flowchart documentation
of both execution paths with pruning gates. Committed `0e01dcb`.

---

## Dashboard Fixes

### trial_stats persistence (progress_display.py)
**Bug:** `ProgressWriter.__init__()` never initialized `self.trial_stats`. Each trial
creates a fresh `PersistentWorkerCoordinator` → fresh `ProgressWriter` → `_write()`
with `trial_stats: {}`, wiping previous trial's survivor counts from dashboard.

**Fix:** On `__init__`, read back existing `trial_stats` from `/tmp/cluster_progress.json`
before first write. Initialized `self.trial_stats = {}` explicitly.
Commit: `60393a0`

### GPU clock display (web_dashboard.py)
**Bug:** `gpu_monitor.py` import fails silently → `gpu_stats = {}` → all clocks show 0 MHz.

**Fix:** Replaced dead clock polling with seeds/s per GPU (node throughput ÷ GPU count).
AVG CLOCK column → AVG SEEDS/GPU. Mini activity bars driven by active/idle state.
Workers page GPU cards show active/idle status instead of duplicate seeds/s.
Commit: `60393a0`

---

## Production Sweep Run 1 — Launch and Failure Diagnosis

### Launch
Launched `sweep_run1.sh` (1B seeds, 50 trials, seed_start=610M from coverage tracker).
Actual range: 610M → 1.68B (coverage tracker added to manifest seed_start).

### Failure
WATCHER escalated after 120 minutes — Step 1 timed out before Trial 1 completed.

**Root cause investigation:**
- S145 commit `2e6081a` intended to give Step 1 infinite timeout by omitting it from
  `step_timeout_overrides={0: 1, 5: 360}`
- But `get_step_timeout_minutes(1)` returns the default `120` when Step 1 is not in
  the dict — `120 * 60 = 7200s` is passed to `_run_step_streaming()`, which only
  treats `<= 0` as infinite
- Step 1 was silently getting 120 minutes despite the S145 intent

**Why 120 minutes was insufficient:**
- Each trial runs 4 sieve passes: constant fwd, constant rev, hybrid fwd, hybrid rev
- Passes 1+2: ~30 minutes total (fast)
- Passes 3+4: ~8-10 hours (hybrid × 5 strategies × 1B seeds)
- Total per trial: ~9-11 hours

**Additional finding — hybrid pass runtime:**
The hybrid forward pass with all 5 strategies across 1B seeds = 5B evaluations.
No pruning gate existed between hybrid forward (Pass 3) and hybrid reverse (Pass 4).

---

## Team Beta Ruling Request and Rulings

Submitted `TB_RULING_REQUEST_HYBRID_RUNTIME_S147.md` covering Q0-Q4.

| Question | Ruling |
|----------|--------|
| Q0: Hybrid forward zero-survivor gate | Option A — implement both paths |
| Q1: WATCHER timeout | Option B — per-step override `{1: 720}` (revised to `{1: 0}`) |
| Q2: Strategy reduction | Option C — configurable, balanced_hybrid for full scan |
| Q3: Constant-bidi threshold gate | Option B — reject |
| Q4: Staged architecture | Option C — staged (rejected by Michael) |

**Michael's override:** Q4 staged architecture rejected. Current single-study approach
retained. Problem is throughput not study design.

---

## Patches Deployed — Q0/Q1/Q2

All applied via `apply_s147_q0_q1_q2.py`. TB-approved full-mode only (Q0 and Q2
are interdependent — Q0A references `_hybrid_strategies` introduced by Q2).

### Q0A — PWC hybrid forward gate
**File:** `persistent_worker_coordinator.py`
**Fix:** Added `if not fwd_h_survivors:` gate between Pass 3 and Pass 4.
Skip not prune — constant-skip results preserved. `bidirectional_variable = set()`.

### Q0B — Legacy coordinator hybrid forward gate
**File:** `window_optimizer_integration_final.py`
**Fix:** Added `if not forward_records_hybrid:` gate. Same skip-not-prune behavior.

### Q1 — Step 1 infinite timeout
**File:** `agents/watcher_agent.py`
**Fix:** Changed `step_timeout_overrides={0: 1, 5: 360}` to `{0: 1, 1: 0, 5: 360}`.
`0 * 60 = 0 → _run_step_streaming` receives 0 → S145 guard fires → `float('inf')`.

### Q2 — Single strategy for full-range hybrid scan
**File:** `persistent_worker_coordinator.py`
**Fix:** Load `balanced_hybrid` strategy via `get_strategy("balanced_hybrid")` before
hybrid passes. Pass `strategies=_hybrid_strategies` to BOTH forward AND reverse
hybrid `run_sieve_pass()` calls. Fallback to all strategies on import error.
Uses `pwc.logger` (not `self.logger` — `run_trial_persistent` is standalone function).
**Impact:** 5x work reduction for hybrid passes.

**Apply order:** Q2 first (introduces `_hybrid_strategies`), then Q0A (uses it in
reverse hybrid call), then Q0B, then Q1.

Commit: `667ece5`

---

## Verification

### Mock harness (test_s147_patches_harness.py)
29/29 pass — logic verification of all three patches.

### Live hardware verification (verify_q0_hybrid_gate.py)
19/19 pass on Zeus with real GPU workers. Live run confirmed:
- All 4 sieve passes executed correctly
- Constant-skip: 1,427 fwd → 1,460 rev → 5 bidirectional survivors
- Hybrid (single strategy): 4 fwd → 1 rev → 0 bidirectional variable
- Q0 gate correctly did not fire (fwd hybrid > 0)
- Q2 single strategy passed to both forward and reverse workers
- Q1 timeout fix confirmed in watcher source

Commit: `bf2549b`

---

## ser8 Cleanup

Cleaned `~/Downloads/CONCISE_OPERATING_GUIDE_v1.0/` — removed ~150 stale files
(session changelogs, proposals, bak files, old chapter versions). Retained 34
canonical reference files.

---

## Key Architectural Findings

### Step 1 execution flow documented
Both paths (original coordinator and PWC) now have matching pruning gates:
- Gate 1: constant-skip forward = 0 → prune trial (existed)
- Gate 2: hybrid forward = 0 → skip hybrid reverse (new Q0)

### GitHub sync clarified
`agents/watcher_agent.py` (3,084 lines, canonical) vs root `watcher_agent.py`
(1,863 lines, stale artifact). Public repo `agents/` subdir is current.
Project knowledge file was stale root copy — needs replacing.

### Sweep timing
With Q2 (1 strategy) instead of 5: hybrid passes ~5x faster.
Estimated per-trial time: ~2.5-3 hours (vs 9-11 hours with all 5 strategies).
WATCHER Step 1 now has infinite timeout (Q1 fix).

---

## Git Commits (S147)

| Commit | Description |
|--------|-------------|
| `9ca2671` | docs(S147): apply script for S146 PWC invariants |
| `c797045` | docs(S147): commit actual patched chapter file content |
| `60393a0` | fix(dashboard): trial_stats persistence + GPU display fixes |
| `0e01dcb` | docs(S147): Step 1 execution flow and pruning — both paths |
| `667ece5` | fix(S147): Q0 hybrid gate + Q1 timeout + Q2 single strategy |
| `bf2549b` | test(S147): Q0/Q1/Q2 verification scripts — 19/19 pass on Zeus |

Both remotes synced at `bf2549b`.

---

## Files Created This Session

| File | Purpose |
|------|---------|
| `apply_s146_doc_updates.py` | Chapter doc patch script |
| `apply_s147_trial_stats_fix.py` | Dashboard trial_stats fix |
| `apply_s147_gpu_display_fix.py` | Dashboard GPU clock fix |
| `apply_s147_gpu_card_fix.py` | Dashboard worker card cleanup |
| `apply_s147_q0_q1_q2.py` | Q0/Q1/Q2 patch script (TB-approved) |
| `test_s147_patches_harness.py` | Mock harness 29/29 pass |
| `verify_q0_hybrid_gate.py` | Live hardware verifier 19/19 pass |
| `docs/STEP1_EXECUTION_FLOW_AND_PRUNING_S147.md` | Step 1 flowchart documentation |
| `docs/TB_RULING_REQUEST_HYBRID_RUNTIME_S147.md` | TB ruling request |

---

**END OF SESSION S147**
