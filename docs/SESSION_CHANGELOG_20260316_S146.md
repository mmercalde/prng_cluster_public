# SESSION CHANGELOG — March 16, 2026 (S146)

**Focus:** Persistent Worker Coordinator (PWC) Debug, Hybrid Sieve Fix, Production Sweep Launch
**Outcome:** All 4 sieve passes (forward, reverse, forward hybrid, reverse hybrid) validated on live hardware. PWC fully operational.

---

## Summary

Extended debugging session focused on making the persistent worker path (`persistent_worker_coordinator.py` + `sieve_gpu_worker.py`) fully functional across all sieve modes. Seven bugs were found and fixed across two files. All fixes verified by Team Beta architectural review and internal harness testing. All 4 sieve passes validated on live hardware (3 trials, 10M seeds).

---

## Bugs Found & Fixed

### Bug 1: `cudaErrorDevicesUnavailable` on Zeus local sieve jobs
**File:** `persistent_worker_coordinator.py`
**Root cause:** PWC dispatched Zeus local sieve jobs with no concurrency control. Zeus has 2 GPUs; without a semaphore, dozens of `sieve_filter.py` subprocesses fired simultaneously.
**Fix:** Added `_localhost_semaphore = threading.Semaphore(2)` mirroring `coordinator.py` line 269. Semaphore acquired/released in `_run_once()` around `_dispatch_local_sieve()`.
**Commit:** `723721a`

### Bug 2: Web dashboard not updating (ProgressWriter missing)
**File:** `persistent_worker_coordinator.py`
**Root cause:** PWC never initialized `ProgressWriter` — coordinator.py wrote `/tmp/cluster_progress.json` but PWC did not.
**Fix:** Added `ProgressWriter` initialization in `startup()`, `update_step()` before each sieve pass, `update_progress()` after each pass, `log_gpu_result()` after each chunk, `finish()` in `shutdown()`.
**Commit:** `723721a`, `cf1d1dc`

### Bug 3: PWC job format mismatch (CRITICAL — from S145, completed S146)
**File:** `persistent_worker_coordinator.py`
**Root cause:** Job dict sent wrong field names to `sieve_gpu_worker.py` (`threshold` vs `min_match_threshold`, missing `skip_range`, `offset`, `sessions`, `prng_families`).
**Fix:** `fix_pwc_job_format.py` — 6 field name corrections.
**Commit:** `0cb2364`

### Bug 4: `java_lcg_hybrid_reverse` kernel signature mismatch (CRITICAL)
**File:** `sieve_gpu_worker.py`
**Root cause:** Hybrid branch used single combined kernel_args for both forward and reverse hybrid. Forward kernel signature: `...threshold, a, c`. Reverse kernel signature: `...threshold, offset` (a,c hardcoded inside kernel). Worker sent `a, c` for both → crash on reverse hybrid.
**Fix:** Split hybrid elif into forward and reverse branches with correct arg tails.
**Team Beta:** Identified by Team Beta architectural review of live `prng_registry.py` kernel signatures.
**Commit:** `cf1d1dc`

### Bug 5: Wrong threshold for hybrid
**File:** `sieve_gpu_worker.py`
**Root cause:** Hybrid kernel and post-filter used `min_match_threshold` instead of `phase2_threshold`. Original `sieve_filter.py` uses `phase2_threshold` for single-phase hybrid families.
**Fix:** `hybrid_threshold = coerce_threshold(phase2_raw, threshold) if phase2_raw is not None else threshold`. Post-filter changed from `if rate >= threshold:` to `if rate >= hybrid_threshold:`.
**Team Beta:** Post-filter bug caught by Team Beta second review pass.
**Commit:** `cf1d1dc`

### Bug 6: Base threshold not coerced
**File:** `sieve_gpu_worker.py`
**Root cause:** `threshold = job.get('min_match_threshold', 0.25)` — string inputs could pass through without coercion.
**Fix:** `threshold = coerce_threshold(job.get('min_match_threshold', None), 0.25)`
**Commit:** `cf1d1dc`

### Bug 7: `custom_params` parsed but never applied
**File:** `sieve_gpu_worker.py`
**Root cause:** Per-family custom params were parsed from job but `default_params` was re-assigned from config without merging.
**Fix:** `default_params = dict(config.get("default_params", {}))` then `if custom_params: default_params.update(custom_params)`.
**Commit:** `cf1d1dc`

### Bug 8: Raw Python ints in constant-skip kernel_args
**File:** `sieve_gpu_worker.py`
**Root cause:** `n_seeds`, `k`, `skip_min`, `skip_max` passed as raw Python ints — ROCm/CuPy can be pickier than expected.
**Fix:** Explicit `cp.int32()` casts for all scalar kernel args.
**Commit:** `cf1d1dc`

### Bug 9: No defensive count clamp on survivor extraction
**File:** `sieve_gpu_worker.py`
**Root cause:** `count = int(survivor_count_gpu[0].get())` — bad kernel write could corrupt count and explode Python extraction.
**Fix:** `count = min(int(survivor_count_gpu[0].get()), n_seeds)` for both hybrid and non-hybrid paths.
**Commit:** `cf1d1dc`

### Bug 10: Strategy dict missing required StrategyConfig fields
**File:** `persistent_worker_coordinator.py`, `sieve_gpu_worker.py`
**Root cause:** Both files built strategy dicts with only `max_consecutive_misses` and `skip_tolerance`. `sieve_filter.py` calls `StrategyConfig(**s)` which requires all 6 fields: `name`, `max_consecutive_misses`, `skip_tolerance`, `enable_reseed_search`, `skip_learning_rate`, `breakpoint_threshold`.
**Fix:** Changed to `s.to_dict() if hasattr(s, 'to_dict') else s` — sends full StrategyConfig dict.
**Commit:** `7e4ae02`

### Bug 11: Inline `import time` shadowing module-level import
**File:** `persistent_worker_coordinator.py`
**Root cause:** Retry block contained `import time; time.sleep(1)` which made `time` a local variable, causing `UnboundLocalError: local variable 'time' referenced before assignment` on all chunks.
**Fix:** Removed inline `import time` — module-level import at line 87 already available.
**Commit:** `e0f5b96`

### Bug 12: `log_gpu_result` never called — dashboard shows 0 seeds/sec
**File:** `persistent_worker_coordinator.py`
**Root cause:** `coordinator.py` calls `log_gpu_result()` after every completed chunk to drive per-node throughput on dashboard. PWC never called it.
**Fix:** Added `t0 = time.time()` before dispatch, `elapsed = time.time() - t0` after, then `self._progress_writer.log_gpu_result(hostname, gpu_id, gpu_type, seeds_in_chunk, elapsed)` after each successful chunk.
**Commit:** `cf1d1dc`

---

## Validation Results

**Test:** 3 trials, 10M seeds, `--seed-start 500000000`, `--test-both-modes`, `--use-persistent-workers`

| Pass | Status | Survivors |
|------|--------|-----------|
| java_lcg forward | ✅ | 43, 2 |
| java_lcg_reverse | ✅ | 40, 0 |
| java_lcg_hybrid forward | ✅ | 671, 201 |
| java_lcg_hybrid_reverse | ✅ | 62, 10 |

Zero ❌ errors. Zero crashes. Zero tracebacks across all chunks.

---

## Git Commits (S146)

| Commit | Description |
|--------|-------------|
| `4b1c975` | fix(persistent_workers): JOB_TIMEOUT_S=600, worker_pool_size=4 |
| `0cb2364` | fix(pwc): job dict field names — CRITICAL FIX |
| `723721a` | fix(pwc): localhost semaphore + ProgressWriter — mirrors coordinator.py |
| `cf1d1dc` | fix(workers): hybrid kernel sigs, thresholds, dashboard throughput — TB verified |
| `e0f5b96` | fix(pwc): remove inline import time shadowing module-level import |
| `7e4ae02` | fix(hybrid): send full StrategyConfig dict — sieve_filter.py requires all 6 fields |

Both remotes synced at `7e4ae02`.

---

## Documentation Updates Required

The following chapters describe the Step 1 sieve execution architecture and need updating to reflect the persistent worker path:

| Document | What Needs Updating |
|----------|-------------------|
| `CHAPTER_1_WINDOW_OPTIMIZER.md` | Add PWC section — persistent worker mode, `--use-persistent-workers` flag, `sieve_gpu_worker.py` as execution backend |
| `CHAPTER_2_BIDIRECTIONAL_SIEVE.md` | Update sieve execution path — coordinator.py → sieve_filter.py is now coordinator.py → PWC → sieve_gpu_worker.py |
| `CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md` | Add persistent worker architecture section — PWC, worker lifecycle, `sieve_gpu_worker.py`, hybrid routing |
| `CHAPTER_12_WATCHER_AGENT.md` | Update Step 1 execution path description — PWC as new default |
| `COMPLETE_OPERATING_GUIDE_v2_0.md` | Add persistent worker operating procedures, new CLI flags, monitoring |

**Invariants to document:**
- Forward hybrid kernel: `...threshold, a, c`
- Reverse hybrid kernel: `...threshold, offset` (a,c hardcoded)
- Hybrid uses `phase2_threshold` not `min_match_threshold`
- Strategies must be sent as full `StrategyConfig` dicts (6 fields)
- `worker_pool_size=4` validated stable (not 8)
- `JOB_TIMEOUT_S=600` (not 300)

---

## Web Dashboard Status

ProgressWriter is now integrated into PWC — `log_gpu_result()` called after every chunk. Dashboard should show live per-node throughput. **Needs verification** with a live run since the dashboard was showing 500 errors earlier in the session due to stale progress file.

---

## Key File Changes

| File | Changes |
|------|---------|
| `persistent_worker_coordinator.py` | +localhost semaphore, +ProgressWriter, +log_gpu_result, +full strategy dict, -inline import time |
| `sieve_gpu_worker.py` | +hybrid kernel sig split, +phase2_threshold, +coerce_threshold, +custom_params, +cp.int32 casts, +count clamp, +full strategy dict |

---

## P1 TODO (carry to S147)

1. **Verify web dashboard live throughput** with production run
2. **Launch production sweep Run 1** — `bash sweep_run1.sh` (clean start)
3. **Chapter documentation updates** (5 documents listed above)
4. **Chapter 13 wire-up** (critical path — autonomous feedback loop)
5. **S110 root cleanup** (884 files)

---

## Architecture Invariants Added (S146)

- PWC must call `log_gpu_result()` after every successful chunk for dashboard throughput
- `_localhost_semaphore = threading.Semaphore(2)` required for Zeus local sieve dispatch
- Hybrid forward kernel: `...threshold, unsigned long long a, unsigned long long c`
- Hybrid reverse kernel: `...threshold, int offset` (a,c hardcoded inside kernel)
- Hybrid uses `phase2_threshold` for both kernel and post-filter
- Strategies sent as full `StrategyConfig.to_dict()` — all 6 fields required by `sieve_filter.py`
- `worker_pool_size=4` (not 8) — validated stable envelope
- `JOB_TIMEOUT_S=600` (not 300)

---

**END OF SESSION S146**
