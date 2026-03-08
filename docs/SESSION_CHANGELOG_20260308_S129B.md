# Session Changelog — S129B / S129B-A
**Date:** 2026-03-08
**Commits:** `006623c` (pre-patch save), `24a7d94` (S129B-A applied)
**Tags:** `pre-s129b-a` at `006623c`, `s129b-a-applied` at `24a7d94`
**Branch:** `s129b-a-chunk-fix` — merged to main, both remotes
**Status:** CLOSED

---

## Summary

S129B was the design and approval session for the S130 persistent worker feature. S129B-A was a targeted patch to fix incorrect GPU throughput parameters in `gpu_optimizer.py` and `coordinator.py` that were causing severe chunk size miscalculation — chunks were being sized at 19k–40k seeds when the hardware could handle 787k–2.2M seeds per chunk.

---

## S129B-A Patch — GPU Parameter Fix

### Problem

`gpu_optimizer.py` contained wildly incorrect measured values from early infrastructure probing:
- RTX 3080 Ti: `seeds_per_second = 29,000` (actual: ~2,210,000)
- RX 6600: `seeds_per_second = 5,000` (actual: ~787,950)

These values fed the chunk sizing formula in `coordinator.py`, producing chunks far too small and causing excessive job fragmentation — 100+ jobs where far fewer were needed.

Additionally, `coordinator.py`'s CLI defaults for `--seed-cap-nvidia` (40k) and `--seed-cap-amd` (19k) were holdovers from early capacity probing and did not reflect measured hardware capability.

### Files Changed

**`gpu_optimizer.py`** (2 sites):
| Field | Before | After |
|-------|--------|-------|
| RTX 3080 Ti `seeds_per_second` | 29,000 | 2,210,000 |
| RTX 3080 Ti `scaling_factor` | 6.0 | 2.80 |
| RX 6600 `seeds_per_second` | 5,000 | 787,950 |

**`coordinator.py`** (3 sites):
| Parameter | Before | After |
|-----------|--------|-------|
| `--seed-cap-nvidia` CLI default | 40,000 | 5,000,000 |
| `--seed-cap-amd` CLI default | 19,000 | 2,000,000 |
| Chunk sizing formula | `total_seeds // 100` hardcoded | `seed_cap_amd` / `seed_cap_nvidia` fallback chain |

### Patch Diff Summary
- 2 files changed, 15 insertions, 9 deletions

### Validation — Phase C Re-run

**Command:**
```bash
python3 coordinator.py --method residue_sieve \
  --seed-cap-amd 2000000 --seed-cap-nvidia 5000000 \
  -s 200000000 daily3.json
```

**Result:**
- Duration: 240.3s
- Aggregate sps: **832,300 sps**
- Jobs: 100/100, 0 failures
- vs S128 baseline (849,469 sps): within normal variance ✅

Log saved: `/tmp/phase_c_rerun_s129ba.log` on rzeus

---

## S129B — Persistent Worker Design (Approved, Not Yet Built)

S129B produced the design and Team Beta approval for S130. Key architectural decisions made:

- `window_optimizer.json` only needs updating (not `scorer_meta.json`) — Step 2 has zero residue_sieve involvement, code-verified
- Gate placement in `execute_gpu_job()` must be **above** the semaphore acquisition block, not between local/remote branches
- Four hard gates established: fault tolerance parity, manifest wiring, GPU-clean invariant, additive routing only
- Proposal document: `PROPOSAL_PERSISTENT_WORKERS_S129B_v1_0.md`

---

## Architecture Findings (Documented for S130)

- WATCHER step mapping confirmed: Step 1 = `window_optimizer.py`, Step 2 = `run_scorer_meta_optimizer.sh`
- Residue sieve call chain is Step 1 only: `window_optimizer.py` → `window_optimizer_integration_final.py` → `MultiGPUCoordinator`
- `execute_gpu_job()` semaphore structure confirmed — persistent path must bypass entirely
- `sieve_gpu_worker.py` validated: ~758k sps/card, 8-GPU concurrent smoke test clean

---

## Carry-Forward Items

- S128 changelog (written this session, retroactively)
- Seed cap patch to `window_optimizer_integration_final.py` (4 sites) — not yet implemented
- `apply_caps.py` with final measured values — not yet run
- rrig6600c i5-8400T CPU throughput deficit (~50%) — known hardware limitation, documented
