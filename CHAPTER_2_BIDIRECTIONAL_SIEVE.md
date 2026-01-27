
---

## 14. Inter-Chunk GPU Cleanup (Added 2026-01-26)

### Problem Identified

Step 1 forward sieves process seeds in chunks (~19K seeds/chunk). With large seed spaces (e.g., 500K seeds = 26 chunks), VRAM fragmentation accumulated without cleanup, causing intermittent GPU hangs:
```
Error: HW Exception by GPU node-11... reason: GPU Hang
```

### Root Cause

| Step | Chunks/Invocation | Cleanup Frequency | Result |
|------|-------------------|-------------------|--------|
| Step 1 | ~26 | Once at exit | **GPU hangs** |
| Step 2.5/3 | 1 | Every invocation | Stable |

### Fix Applied

Added inter-chunk cleanup to both forward sieve loops in `sieve_filter.py` (lines 230, 385):
```python
if chunk_start + chunk_size < seed_end:
    _best_effort_gpu_cleanup()
```

Also added `gc.collect()` to `_best_effort_gpu_cleanup()`.

### Validation

- 20/20 benchmark trials: 0 GPU hangs
- All 26 GPUs healthy post-run
- Performance overhead: <5%
