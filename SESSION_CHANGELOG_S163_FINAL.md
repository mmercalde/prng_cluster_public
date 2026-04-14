# SESSION CHANGELOG — S163 FINAL
**Date:** 2026-04-12 through 2026-04-13
**Session:** S163
**Focus:** TCP-PWC Restoration + Seed Cap Ladder Testing
**Author:** Claude (Team Alpha Lead Dev)
**Status:** CLOSED — All fixes committed, 300K seed cap validated
**HEAD:** `0bfc391`

---

## 🏆 Session Achievements

1. Restored full 26/26 GPU operation after rrig6600b clone broke TCP-PWC
2. Fixed 6 root causes preventing workers from connecting
3. Discovered and fixed JSON serialization hang (2.6M records → multi-hour freeze)
4. Discovered and fixed missing `set_limit()` in TCP-PWC path
5. Validated seed cap ladder: 100K → 250K → 300K ✅ stable

---

## Fixes This Session

### Fix 1 — `window_optimizer_integration_final.py`
**Commit:** `526ae93`
`run_trial_persistent()` always defaulted to `"ssh"` transport. Added:
`pwc_transport = getattr(coordinator, 'pwc_transport', 'tcp')`

### Fix 2 — `persistent/pwc_worker_service.py`
**Commits:** `a42e71c`, `c0615f3`
- Restored S161 v2 two-phase startup (`online→init→ready`, lazy ROCm import)
- Added `sock.settimeout(None)` after handshake — `create_connection(timeout=10)`
  was timing out ALL subsequent reads including the blocking `_wait_for_init()` recv

### Fix 3 — `persistent/pwc_transport_tcp.py`
**Commit:** `aacb0b1`
Increased `HEARTBEAT_TIMEOUT_S` from 60s to 300s. Workers go silent after
sending `online` while waiting for `init` — 60s heartbeat was reclaiming them.

### Fix 4 — `persistent_worker_coordinator.py`
**Commit:** `b632a26`
TCP launch script passed `--gpu-id` and `--worker-id` as CLI args but
`pwc_worker_service.py __main__` reads ONLY env vars (`PWC_GPU_ID`, `PWC_WORKER_ID`).
All 8 workers per rig registered as the same `worker_id`, overwriting each other.
Only 1 worker per rig (3 total) survived. Fixed by adding env var exports to launch script.

### Fix 5 — `sieve_filter.py`
**Commit:** `9b3c443`
Added `set_limit()` call at module import time in TCP-PWC path. The 256MB pool
cap (`PRNG_CUPY_POOL_LIMIT_MB`) was only enforced in `sieve_gpu_worker.py` (SSH
path). TCP-PWC workers import `sieve_filter` directly — the cap was never applied,
causing pool to grow unbounded across trials and triggering GPU page faults.

### Fix 6 — `persistent_worker_coordinator.py`
**Commit:** `0bfc391`
Skipped large JSON result writes when survivor count > 100K. At 250K seed cap
with 2.6M survivors, `json.dump()` of the full result dict took 2h44m and hung
the process. These JSON files are archival only — Steps 2-6 use NPZ files.
Now writes a lightweight summary JSON instead.

### Fix 7 — Data file missing on rrig6600b
`daily3.json` missing after public repo clone. Fixed:
`scp rzeus:~/distributed_prng_analysis/daily3.json rrig6600b:~/distributed_prng_analysis/`

---

## Seed Cap Ladder Results

| Seed Cap | Result | Notes |
|----------|--------|-------|
| 10K | ✅ Stable | Verification run |
| 100K | ✅ Stable | S162 victory baseline |
| 250K | ✅ Stable | After fresh reboot |
| 300K | ✅ Stable | After fresh reboot, all 26 GPUs held |
| 500K | ⚠️ Crashes | rrig6600 crashed — memory accumulation |

**Confirmed production ceiling: 300K seed cap**

**Important:** 300K requires fresh rig reboot between runs. Without reboot,
CuPy memory pool accumulates state across trials and triggers GPU page faults
at 300K+. The `set_limit()` fix caps new allocations but does not flush
existing pool state. A startup pool flush fix is needed for next session.

---

## Key Learnings

1. **`pwc_worker_service.py __main__` reads ONLY env vars** — `PWC_GPU_ID`,
   `PWC_WORKER_ID`, `PWC_HOST`, `PWC_PORT`. CLI args are completely ignored.
   Launch script MUST export these env vars.

2. **`create_connection(timeout=N)` sets socket timeout for ALL reads** —
   always call `sock.settimeout(None)` after handshake.

3. **`HEARTBEAT_TIMEOUT_S=300s`** — do not reduce. Workers are silent between
   `online` and `init` messages.

4. **`set_limit()` must be called in BOTH code paths** — `sieve_gpu_worker.py`
   (SSH-PWC) and `sieve_filter.py` (TCP-PWC).

5. **Large JSON writes block for hours** — skip when survivors > 100K. Steps
   2-6 use NPZ files, not these JSON files.

6. **After rig rebuild from public repo**: SCP `daily3.json` and `trse_context.json`
   manually — data files are not in the public repo.

7. **Memory accumulation between runs** — rigs need full reboot between
   production runs at 300K+ until startup pool flush is implemented.

8. **GPU page fault signature** `GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF` +
   `Faulty UTCL2 client ID: unknown (0x1ff)` + `WALKER_ERROR: 0x7` indicates
   complete GPU virtual address space corruption — caused by unbounded CuPy
   memory pool growth across trials.

---

## Throughput Benchmarks (This Session)

| Seed Cap | Seeds/sec | Notes |
|----------|-----------|-------|
| 10K | ~537K | TCP-PWC working (3/24 workers initially) |
| 10K | ~1.7M | After env var fix (24/24 workers) |
| 100K | ~1.7M | Stable |
| 250K | ~7-15M | Stable after set_limit + JSON fix |
| 300K | ~12M | Stable, all 26 GPUs |

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `526ae93` | fix(s163): pass pwc_transport to run_trial_persistent |
| `a42e71c` | fix(s163): restore pwc_worker_service v2 two-phase startup |
| `c0615f3` | fix(s163): set socket to blocking after handshake |
| `2b182ee` | fix(s163): restore pwc_transport_tcp.py to S162 victory baseline |
| `aacb0b1` | fix(s163): increase TCP heartbeat timeout 60s→300s |
| `b632a26` | fix(s163): add PWC_GPU_ID and PWC_WORKER_ID env vars to TCP launch script |
| `0250e06` | docs(s163): session changelog — 6 root causes fixed, 26/26 GPUs restored |
| `9b3c443` | fix(s163): add set_limit() to sieve_filter.py + skip large JSON writes |
| `0bfc391` | fix(s163): 300K seed cap validated stable |

---

## Next Session Priorities

1. **Startup pool flush** — add `cp.get_default_memory_pool().free_all_blocks()`
   at worker startup (before first job, single call, no race condition). Eliminates
   need for rig reboot between runs.

2. **Production Step 1 run** — 300K seed cap, 50-200 trials, full seed space
   coverage across 4 quarters.

3. **Fix NPZ accumulator IndexError** — `index 4050 is out of bounds for axis 0
   with size 0` — fires when run produces 0 survivors. Low priority but noisy.

4. **Run Step 2** — Scorer Meta-Optimizer with accumulated survivors from
   `bidirectional_survivors_all.npz` (4,724 seeds).

5. **Zeus `cudaErrorDevicesUnavailable`** — GPU P8 idle state between chunks.
   TB proposal needed.

---

## Infrastructure State (End of S163)

| Component | State |
|-----------|-------|
| Zeus HEAD | `0bfc391` |
| Public repo HEAD | `0bfc391` |
| rrig6600 amdgpu | `6.12.12` DKMS ✅ pinned |
| rrig6600b amdgpu | `6.12.12` DKMS ✅ pinned |
| rrig6600c amdgpu | `6.12.12` DKMS ✅ pinned |
| TCP-PWC transport | ✅ 26/26 GPUs |
| Confirmed seed cap | 300K (fresh reboot required) |
| bidirectional_survivors_all.npz | ✅ 4,724 seeds |
| HEARTBEAT_TIMEOUT_S | ✅ 300s |
| PWC_GPU_ID env var | ✅ In launch script |
| set_limit() TCP path | ✅ sieve_filter.py |
| JSON write cap | ✅ 100K threshold |

---

*Session S163 — 2026-04-12 through 2026-04-13 — Team Alpha*
