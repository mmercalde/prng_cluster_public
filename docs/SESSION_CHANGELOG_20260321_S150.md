# SESSION CHANGELOG — S150
**Date:** 2026-03-21  
**Commit:** `65461c9`  
**Status:** Run 1 active (study `window_opt_1774109563`, Trial 8+ running)

---

## Summary

S150 implemented and deployed the slim_v1 IPC serialization fix — TB approved
Option A. Pass 3/4 (hybrid) throughput improved 22x on AMD rigs. Per-trial NPZ
checkpoint confirmed working (708 seeds accumulated after Trial 7). Both
correctness goals met: survivors saved, rigs running fast.

---

## Fix Deployed: slim_v1 IPC Serialization

**Commit:** `65461c9`  
**Files:** `sieve_gpu_worker.py`, `persistent_worker_coordinator.py`  
**Patch:** `apply_s150_slim_v1_ipc.py` (12/12 verified on fresh clone)

**Problem:**  
Workers returned results as lists of Python dicts — one dict per survivor with
6-7 fields. At 1,400 survivors × ~150 bytes/dict = ~210KB JSON per chunk.
Pass 3/4 throughput: ~3,774 s/s per rig (down from ~84k s/s burst). 22x
degradation caused entirely by dict serialization overhead.

**Fix — slim_v1 parallel arrays:**  
Worker now returns flat parallel arrays instead of list-of-dicts:
```json
{
  "format": "slim_v1",
  "seeds": [...],
  "match_rates": [...],
  "strategy_ids": [...],   
  "skip_sequences": [...]  
}
```
`strategy_ids` and `skip_sequences` included for hybrid passes only, driven
from job context (not survivor content — TB ruling to handle zero-survivor
hybrid chunks correctly).

**Coordinator:** Accepts both slim_v1 fast path and legacy dict-list fallback
for rollout safety. Hybrid enforcement: if job is hybrid and arrays missing →
clean error, not silent corruption. Length assertion on all arrays.

**Compact separators:** `json.dumps(obj, separators=(',', ':'))` in `_emit()`.

**TB ruling iterations:** 5 rounds of TB review. Key corrections:
- Nested result bug (fixed: slim_v1 fields at top level of run_sieve_job return)
- verify() mode-aware (fixed: --coordinator-only/--worker-only check only relevant file)
- Compact separators (fixed: `(',', ':')` not `(',', ': ')`)
- List multiplication bug (fixed: `[[] for _ in survivors]` not `[[]] * n`)
- Hybrid zero-survivor edge case (fixed: `_is_hybrid` from job context not survivor content)

**Deploy sequence:**
1. `--coordinator-only` on Zeus (takes effect immediately, old workers still work)
2. `--worker-only` on Zeus + commit + scp to rigs (workers respawn per trial)

---

## Measured Results

| Metric | Before | After | Gain |
|---|---|---|---|
| Pass 3/4 rig throughput | ~3,774 s/s | ~84,000 s/s | **22x** |
| Cluster total (hybrid pass) | ~73,000 s/s | ~299,135 s/s | **4x** |
| Pass 1/2 throughput | ~2,000,000 s/s | ~2,000,000 s/s | unchanged |
| NPZ checkpoint | end-of-run only | per-trial | data safe |
| NPZ accumulator | 666 seeds | 708 seeds | growing |

---

## Run 1 Status

| Item | Value |
|---|---|
| Study | `window_opt_1774109563` |
| Seed range | 660,000,000 → 1,733,741,824 |
| Trials completed | 7 (Trial 8 running) |
| Best so far | W5_O41 — 701 bidirectional |
| NPZ accumulator | 708 seeds |
| Checkpoint | Active — fires after every survivor trial |

---

## Architecture Invariants Added S150

- **[S150]** slim_v1 IPC format — parallel arrays, JSON-line outer protocol preserved
- **[S150]** `_is_hybrid` driven from job context (`prng_type`/`skip_mode`), not survivor content
- **[S150]** Coordinator enforces hybrid arrays present for hybrid jobs (clean error on violation)
- **[S150]** Legacy dict-list parser preserved in coordinator for rollout safety
- **[S150]** Rigs have no git — deploy worker updates via scp from Zeus

---

## S151 Priority List

1. **P0** — `--force-step` flag for WATCHER (freshness skip blocking every resume/restart)
2. **P1** — `sweep_run2.sh` with `enqueue_trial()` warm-start from Run 1 best params
3. **P2** — Remove legacy dict-list parser from coordinator (after one full production run proves slim_v1 stable)
4. **P3** — Measure GPU utilization during Pass 3/4 with 1s rocm-smi polling (confirm 22x improvement is sustained)
5. **Backlog** — S110 root cleanup, sklearn warnings, CSV writer removal, Chapter 13 wire-up, selfplay NN fix
