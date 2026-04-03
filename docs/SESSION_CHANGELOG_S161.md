# Session S161 — PWC TCP Transport Benchmark
**Date:** 2026-04-03 (opening)
**Team:** Team Alpha
**Status:** IN PROGRESS

---

## Context

S161 was deferred from S159 pending resolution of the AMD GPU crash instability. S160 identified and patched the root cause (`_best_effort_gpu_cleanup()` missing from persistent worker loop). With stability confirmed, S161 proceeds.

---

## What We Know Going In

### Stability
- `_best_effort_gpu_cleanup()` after each chunk eliminates GPU memory accumulation crashes
- All 3 AMD rigs stable under sustained load with the S160-v5 patch
- The same patch **must be applied to the PWC TCP worker** before benchmarking

### ZMQ Baseline (for comparison)
- Aggregate throughput: ~800,000 sps (26 GPUs)
- Per-GPU throughput: ~31,000 sps
- Cross-pass lease coordination overhead: unacceptable for production
- Inter-pass cleanup stall: ~26 minutes between forward and reverse sieve
- GPU utilization: 0-49% (massively underutilized due to ZMQ overhead)

### PWC TCP Architecture
- Direct TCP sockets between coordinator and workers — no ZMQ broker
- No SQLite lease management overhead
- No systemd-run worker lifecycle — workers managed directly via TCP connection state
- Inter-chunk cleanup must be explicit (same lesson as ZMQ)
- Cross-pass sequencing must be coordinator-controlled (no shared job queue)

---

## S161 Goals

1. **Apply S160-v5 cleanup patch to PWC TCP worker** — prerequisite before any benchmark
2. **Run PWC TCP acceptance test** — confirm all 3 AMD rigs stable under TCP transport
3. **Benchmark throughput** — establish sps/GPU baseline vs ZMQ's 31K sps/GPU
4. **Validate cross-pass sequencing** — confirm coordinator blocks next pass until current pass drains
5. **TB review** — submit benchmark results with comparison to ZMQ and S130 PWC SSH

---

## First Action Item

Before launching any run, apply inter-chunk cleanup to the PWC TCP worker execution loop.

Locate the worker's chunk processing loop in `persistent_worker_coordinator.py` and add after result delivery:

```python
# S161 (TB-approval pending): inter-chunk GPU cleanup — same fix as S160-v5
try:
    from sieve_filter import _best_effort_gpu_cleanup
    _best_effort_gpu_cleanup()
except Exception:
    pass
```

This requires TB approval before deployment — submit as S161 first patch.

---

## Success Criteria

| Metric | ZMQ Baseline | Target |
|--------|-------------|--------|
| Aggregate sps | ~800,000 | >2,000,000 |
| Per-GPU sps | ~31,000 | >80,000 |
| GPU crashes | 0 (patched) | 0 |
| Cross-pass stall | ~26 min | <60 sec |
| GPU utilization | 0–49% | >70% |

---

## Pending TB Items From S160
- S160 cross-cutting finding: PWC cleanup gap (proposal submitted)
- TB ruling on inter-chunk cleanup in PWC TCP worker loop
