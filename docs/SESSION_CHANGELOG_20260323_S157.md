# SESSION CHANGELOG — S157
**Date:** 2026-03-23  
**Session:** S157  
**Author:** Claude (Team Alpha Lead Dev)  
**Status:** Complete (reconstructed from git log and session history)

---

## Summary

Short handoff session. Deployed per-worker CUPY_CACHE_DIR fix to resolve
the 8-worker CuPy kernel cache race condition on AMD rigs. Also deployed
zero-survivor NPZ conversion guard.

---

## Root Cause Addressed

CuPy's `RawKernel` crashes when multiple worker processes simultaneously
compile kernels to the shared `~/.cupy/kernel_cache/` directory. With 8
workers per rig all initializing at startup, write conflicts in the shared
cache caused crashes — particularly on rrig6600c.

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `c06e4e4` | fix(s157): per-worker CUPY_CACHE_DIR — prevents 8-worker cache race |
| `c93e680` | docs(s157): add S157 chat prompt and handoff |

---

## Patches Deployed

### Per-worker CUPY_CACHE_DIR (`c06e4e4`)
- `persistent_worker_coordinator.py` — each worker gets isolated cache path
- `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_{gpu_id}` per worker
- GPU 0 → `/tmp/cupy_cache_gpu_0`
- GPU 7 → `/tmp/cupy_cache_gpu_7`
- Eliminates shared kernel cache write race
- Cache lives in `/tmp` — wiped on reboot for clean state

---

## Invariants
- `bidirectional_survivors_binary.npz` git-tracked ✅
- Dual-push enforced ✅
- seed_cap_amd = 2,000,000 ✅
