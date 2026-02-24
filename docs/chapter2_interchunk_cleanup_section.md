## Section to add to CHAPTER_2_BIDIRECTIONAL_SIEVE.md

### 14. Inter-Chunk GPU Cleanup (Added 2026-01-26)

#### Problem Identified

Step 1 forward sieves process seeds in chunks (~19K seeds/chunk). With large seed spaces (e.g., 500K seeds = 26 chunks), VRAM fragmentation accumulated without cleanup, causing intermittent GPU hangs:

```
Error: HW Exception by GPU node-11... reason: GPU Hang
```

#### Root Cause

| Step | Chunks/Invocation | Cleanup Frequency | Result |
|------|-------------------|-------------------|--------|
| Step 1 | ~26 | Once at exit | **GPU hangs** |
| Step 2.5/3 | 1 | Every invocation | Stable |

`sieve_filter.py` only called `_best_effort_gpu_cleanup()` at script exit (line 679), not between chunks.

#### Fix Applied

Added inter-chunk cleanup to both forward sieve loops in `sieve_filter.py`:

```python
# Lines 230 and 385 (inside chunk loops)
# Inter-chunk cleanup (skip final chunk - Team Beta 2026-01-26)
if chunk_start + chunk_size < seed_end:
    _best_effort_gpu_cleanup()
```

Also added `gc.collect()` to `_best_effort_gpu_cleanup()` for consistency with other workers.

#### Validation Results

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| GPU hangs (20 trials) | 1 | 0 |
| GPUs healthy post-run | 25/26 | 26/26 |
| Performance overhead | - | <5% |

#### Files Modified

- `sieve_filter.py` - Lines 64 (gc.collect), 230, 385 (inter-chunk cleanup)
- Backup: `sieve_filter_backup_20260126_182352.py`
