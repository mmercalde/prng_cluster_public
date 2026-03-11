# SESSION CHANGELOG — S135
**Date:** 2026-03-10
**Session:** S135
**Author:** Team Alpha (Claude)

---

## Summary

Fixed 9 bugs in the persistent worker engine. All 14/14 chunks now pass clean across all 4 sieve passes (forward, reverse, hybrid-forward, hybrid-reverse). Pipeline is fully operational end-to-end with persistent workers.

---

## Bugs Fixed

### Bug 1 — `dataset_path` never passed to workers
- **Files:** `persistent_worker_coordinator.py`, `window_optimizer_integration_final.py`
- **Symptom:** All 14 chunks failed: `FileNotFoundError: ''`
- **Fix:** Added `dataset_path` to `run_sieve_pass()` signature and all call sites in `run_trial_persistent()`, and propagated through integration call site.

### Bug 2 — Worker result nested under `"result"` key
- **File:** `persistent_worker_coordinator.py`
- **Symptom:** `TypeError: unhashable type: 'dict'`
- **Fix:** `_dispatch_to_worker` now unwraps `result["result"]` and extracts `seed`, `match_rate`, `skip_sequence`, `strategy_id` into flat parallel lists.

### Bug 3 — SSH pipe binary mode / large JSON truncation
- **File:** `persistent_worker_coordinator.py`
- **Symptom:** `Expecting value: line 1 column 8193` — JSON truncated at 8192 bytes
- **Fix:** Switched to `Popen(bufsize=0)` binary mode. stdin writes use `.encode()`, stdout reads decode bytes explicitly.

### Bug 4 — Local sieve (Zeus) result format mismatch
- **File:** `persistent_worker_coordinator.py`
- **Symptom:** Chunk 12 (Zeus local path) always "unknown"
- **Fix:** `_dispatch_local_sieve` now accepts `{"success": true, ...}` format from `sieve_filter.py`.

### Bug 5 — Empty pipe response not detected
- **File:** `persistent_worker_coordinator.py`
- **Symptom:** Chunk 13 tail chunk — `Expecting value: line 1 column 1 (char 0)`
- **Fix:** Explicit empty-line check after decode — marks worker dead, returns structured error.

### Bug 6 — `TestResult` pruned constructor invalid fields
- **File:** `window_optimizer_integration_final.py`
- **Symptom:** `TypeError: TestResult.__init__() got an unexpected keyword argument 'bidirectional_constant'`
- **Fix:** Pruned return passes only `config`, `forward_count=0`, `reverse_count=0`, `bidirectional_count=0`, `iteration`.

### Bug 7 — `skip_range` string format crashes NPZ conversion
- **File:** `convert_survivors_to_binary.py`
- **Symptom:** `ValueError: invalid literal for int() with base 10: '5-56'`
- **Fix:** Added `_parse_skip_range()` helper handling string `'min-max'`, list `[min,max]`, or plain int.

### Bug 8 — SSH banner consuming heartbeat line
- **File:** `persistent_worker_coordinator.py`
- **Symptom:** Chunk 0 always JSON parse error on heartbeat
- **Fix:** Added `-q` to SSH command. Drain loop reads until line contains `"status"` and `"ready"`, discarding banner noise.

### Bug 9 — Two chunks sharing same worker handle concurrently
- **File:** `persistent_worker_coordinator.py`
- **Symptom:** Chunk 0 and Chunk 13 both fail — 14 chunks, 13 workers, `workers[0]` and `workers[13]` are same handle, interleaving stdin/stdout across threads
- **Fix:** Added `dispatch_lock: threading.Lock` to `WorkerHandle` dataclass. Entire write/read cycle inside `with handle.dispatch_lock:`.

---

## Test Results (Final)

```
Trial 1 (W8_O43):
  Forward pass:      14/14 chunks ✅  2,737 survivors
  Reverse pass:      14/14 chunks ✅  2,737 survivors
  Hybrid-fwd pass:   14/14 chunks ✅  2,737 survivors
  Hybrid-rev pass:   14/14 chunks ✅  2,737 survivors

Trial 2 (W452_O36):
  Forward pass:      14/14 chunks ✅  0 survivors (expected)

Worker pool:         12/12 alive
Chunk failures:      0/14 ✅  (was 2/14 pre-fix)
```

---

## Files Modified

| File | Changes |
|------|---------|
| `persistent_worker_coordinator.py` | Bugs 1–5, 8–9 |
| `window_optimizer_integration_final.py` | Bugs 1, 6 |
| `convert_survivors_to_binary.py` | Bug 7 |
| `test_persistent_worker_harness.py` | T22 binary stdin fix |

---

## Active TODOs (carry-forward)

1. S110 root cleanup — 884 files in project root
2. sklearn warnings in Step 5
3. Remove CSV writer from `coordinator.py`
4. Regression diagnostics gate → set to `True`
5. S103 Part 2
6. Phase 9B.3 (deferred)
7. Z10×Z10×Z10 kernel update (TB proposal needed)
8. Telegram alert for GPU quarantine
9. Per-node seed limit for rig-6600c
10. TRSE Step 0 CLI args fix
11. Gate 1 threshold (50 survivor minimum blocks test runs)
12. Low variance warning — 3 unique match_rate values (Step 1 integration version issue)
