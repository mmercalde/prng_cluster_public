# SESSION CHANGELOG — March 22, 2026 (S152)

**Focus:** slim_v1 IPC crash investigation, NPZ checkpoint fixes, production stability verification
**Outcome:** Pre-slim_v1 stable on 26/26 GPUs. NPZ checkpoint confirmed working. slim_v1 parked pending fix. i5-9400 CPU upgrade ordered for rrig6600c.

---

## Summary

Long debugging session focused on two parallel tracks: (1) diagnosing why slim_v1 crashes rrig6600c and (2) fixing the NPZ checkpoint pipeline so survivors are actually saved. Pre-slim_v1 is confirmed stable at 26/26 GPUs. Three bugs found and fixed in the checkpoint pipeline. slim_v1 remains parked — root causes identified but not yet fully resolved.

---

## Bugs Found & Fixed

### Bug 1: NPZ checkpoint accumulator wiring (CRITICAL — silently broken since S149)
**File:** `window_optimizer.py`
**Root cause:** `_survivor_accumulator` was set on the `BayesianOptimization` instance, but `OptunaBayesianSearch.search()` reads `getattr(self, '_survivor_accumulator', None)` where `self` = `OptunaBayesianSearch` — a different object. Accumulator was always `None` → S149-CKPT never fired → survivors never written mid-run.
**Fix:** `apply_s152_accumulator_wiring_fix.py` — copies `_survivor_accumulator` from `BayesianOptimization` to `OptunaBayesianSearch` before delegation.
**Commit:** `2c3ae24`

### Bug 2: NPZ checkpoint tmp filename (numpy auto-appends .npz)
**File:** `window_optimizer_bayesian.py`
**Root cause:** `numpy.savez_compressed(_tmp, ...)` auto-appends `.npz` when the filename doesn't end in `.npz`. The tmp file was named `bidirectional_survivors_all.npz.ckpt.tmp` — numpy wrote `bidirectional_survivors_all.npz.ckpt.tmp.npz`. Then `os.replace()` tried to rename the non-`.npz` path → `FileNotFoundError`.
**Fix:** `apply_s152_ckpt_tmp_fix.py` — tmp filenames changed to end in `.npz`: `bidirectional_survivors_all.ckpt.tmp.npz`
**Commit:** `636abc0`

### Bug 3: Per-worker CUPY_CACHE_DIR (slim_v1 race condition mitigation)
**File:** `persistent_worker_coordinator.py`
**Root cause:** slim_v1 (S149-B) removed `ROCR_VISIBLE_DEVICES` per-worker GPU masking. All 8 workers on each rig now see all GPUs and bind via `cp.cuda.Device(gpu_id)`. When 8 workers simultaneously compile `cp.RawKernel()` on first job, they race on the shared `~/.cupy/kernel_cache/` directory. rrig6600c's slower i5-8400T CPU (1.70GHz vs 2.80/2.90GHz on other rigs) makes compilation take longer, causing all 8 workers to be in the cache-write window simultaneously — deterministic crash.
**Fix:** `apply_s152_cupy_cache_dir.py` — per-worker `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_N` injected into spawn env. Each worker gets isolated cache, no shared writes.
**TB ruling:** Confirmed as TB first-choice fix. CUPY_CACHE_IN_MEMORY=1 not reliable on ROCm/hipcc backend.
**Commit:** `636abc0` (included in same push)

---

## slim_v1 Status — PARKED

slim_v1 (`sieve_gpu_worker.py` S150 + S149-B) delivers 22x throughput gain on hybrid passes but has two unresolved issues:

### Issue 1: Kernel out-of-bounds write (TB confirmed — latent bug)
**File:** `prng_registry.py`
**Root cause:** All 22 hybrid kernels write to `skip_sequences[pos * k + i]` where `pos = atomicAdd(survivor_count, 1)`. Buffer is allocated to `_max_surv * k` but guard is `if (pos < n_seeds)` — wrong bound. If survivors > `_max_surv`, kernel writes beyond buffer → GPU memory corruption.
**Status:** Not causing production crashes under normal operation (typical survivors 40-300 per chunk, `_max_surv`=5000). Only triggered by degenerate configs (window_size=2 → 340k survivors/chunk). Fix identified: inject `#define MAX_SURVIVORS {_max_surv}` via kernel source prepend in `sieve_gpu_worker.py`, replace guards in `prng_registry.py`. Not yet implemented — deferred to slim_v1 fix sprint.

### Issue 2: window_optimizer -9 kill with large IPC payload
**Root cause (most likely per TB):** slim_v1 parallel arrays + degenerate config (window_size=2) → 340k survivors/chunk → single massive JSON dict during `json.loads()` on Zeus → peak RSS spike → SIGKILL.
**Status:** Not confirmed. Per-worker cache fix eliminated rrig6600c crash but exposed this second crash on Zeus. Clean test with sane config (window_size ≥ 8, FT ≥ 0.30) never completed — killed externally before a valid trial ran.
**Fix path:** Either chunk IPC payloads (send survivors in batches), add minimum window_size floor in Optuna bounds, or confirm with controlled test.

### Recommended next steps for slim_v1:
1. Fix kernel bounds guard in `prng_registry.py` (`#define MAX_SURVIVORS` approach)
2. Add minimum window_size floor (e.g. ≥ 6) in Optuna search bounds to prevent degenerate configs
3. Run controlled 3-trial test with slim_v1 + cache fix + sane config
4. Monitor Zeus RSS during trial to confirm no OOM

---

## Hardware

### rrig6600c CPU upgrade ordered
- **Current:** Intel i5-8400T @ 1.70GHz (35W TDP) — significantly slower than other rigs
- **Replacement:** Intel i5-9400 @ 2.90GHz (65W TDP) — matches rrig6600b
- **Motherboard:** Biostar TB360-BTC Pro 2.0 — officially supports 9th gen, no BIOS flash needed
- **Cost:** $59 eBay, arriving in ~2-4 days
- **Impact:** Eliminates CPU timing differential that makes slim_v1 cache race deterministic on rrig6600c

---

## Production Verification

### 3-trial verification run (pre-slim_v1)
- **26/26 GPUs active** including rrig6600c ✅
- **S149-CKPT confirmed working:** `Trial 2: NPZ checkpoint written (676 total, +2 new seeds)` ✅
- **NPZ tmp filename fix confirmed working** ✅
- **Accumulator wiring fix confirmed working** ✅
- **rrig6600c stable** throughout all 3 trials ✅

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `ae6afb2` | slim_v1 VRAM fix + incremental flush + force-step + IPC confirm + sweep_test + coverage reset |
| `12a8b6b` | SSH keepalive hardening — ServerAliveCountMax=10, ConnectTimeout=10 |
| `701c501` | Heartbeat timeout 30s→90s |
| `2c3ae24` | Wire _survivor_accumulator BayesianOptimization→OptunaBayesianSearch |
| `636abc0` | NPZ checkpoint tmp filename fix + per-worker CUPY_CACHE_DIR |

---

## Architecture Invariants Added S152

- **[S152]** slim_v1 VRAM: `skip_sequences_gpu` capped at `_max_surv * k` (not `n_seeds * k`)
- **[S152]** Incremental NPZ flush after every `PRNG_FLUSH_EVERY` new bidi survivors
- **[S152]** `--force-step N` bypasses freshness check; `sweep_run1.sh --resume` auto-adds `--force-step 1`
- **[S152]** SSH spawn: `ServerAliveCountMax=10`, `ConnectTimeout=10`
- **[S152]** `WORKER_HEARTBEAT_TIMEOUT_S = 90` (was 30)
- **[S152]** `_survivor_accumulator` wired through `BayesianOptimization→OptunaBayesianSearch`
- **[S152]** NPZ tmp filenames end in `.npz` to prevent numpy auto-append collision
- **[S152]** Per-worker `CUPY_CACHE_DIR=/tmp/cupy_cache_gpu_N` in coordinator spawn env
- **[S152]** `[slim_v1]` DEBUG / `[legacy-ipc]` WARN log lines in coordinator

---

## Next Session (S153)

**Priority order:**
1. Reset manifest to 50 trials and launch production Run 1 (660M→1.07B)
2. slim_v1 fix sprint: kernel bounds guard + minimum window_size floor + controlled test
3. Install i5-9400 on rrig6600c when it arrives
4. S110 root cleanup (884 files in project root) — deferred
5. sklearn warnings in Step 5 — deferred
