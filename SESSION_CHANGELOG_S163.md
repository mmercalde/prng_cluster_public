# SESSION CHANGELOG — S163
**Date:** 2026-04-12
**Session:** S163
**Focus:** TCP-PWC Full 26-GPU Restoration After rrig6600b Clone
**Author:** Claude (Team Alpha Lead Dev)
**Status:** VICTORY — 26/26 GPUs active, 537,408 seeds/sec confirmed

---

## Summary

S163 restored full 26-GPU TCP-PWC operation after the cluster was destabilized by
cloning rrig6600b from the public repo (which lacked several critical private-repo
changes). Six root causes were identified and fixed across four files.

---

## Root Causes Found & Fixed

### Fix 1 — `window_optimizer_integration_final.py`
**Commit:** `526ae93`
**Problem:** `run_trial_persistent()` was called without `pwc_transport` parameter,
so it always defaulted to `"ssh"` regardless of `--pwc-transport tcp` flag.
**Fix:** Added `pwc_transport = getattr(coordinator, 'pwc_transport', 'tcp')` to
the `run_trial_persistent()` call.

### Fix 2 — `persistent/pwc_worker_service.py`
**Commit:** `a42e71c` → `c0615f3`
**Problem:** The S161 v2 two-phase startup (`online→init→ready` with lazy ROCm
import) was never in the public repo. All rigs had the old v1 single-phase worker
that imported sieve_filter at startup BEFORE connecting, causing the coordinator's
`_tcp_wait_online` timeout to expire before workers connected.
**Fix 1:** Rebuilt v2 worker with two-phase protocol:
  - Connect to Zeus (milliseconds, no ROCm import)
  - Send `{"message_type": "online"}`
  - Wait for `{"command": "init"}` from coordinator
  - Import sieve_filter (ROCm warmup, ~2s on warm cache)
  - Send `{"message_type": "ready"}`
  - Enter job loop
**Fix 2 (`c0615f3`):** Added `sock.settimeout(None)` after handshake — `create_connection(timeout=10)`
sets socket timeout for ALL reads, causing workers to timeout after 10s waiting for
`init` command.

### Fix 3 — `persistent/pwc_transport_tcp.py`
**Commit:** `aacb0b1`
**Problem:** `HEARTBEAT_TIMEOUT_S = 60.0` — the heartbeat monitor reclaimed workers
that had been silent for 60s while waiting for the `init` broadcast. Workers connect,
send `online`, then go silent waiting for `init`. With 24 workers launching over ~14s
plus coordinator overhead, workers could be reclaimed before `init` was broadcast.
**Fix:** Increased `HEARTBEAT_TIMEOUT_S` from `60.0` to `300.0`.

### Fix 4 — `persistent_worker_coordinator.py`
**Commit:** `b632a26`
**Problem:** The TCP launch script passed `--gpu-id` and `--worker-id` as CLI
arguments to `pwc_worker_service.py`, but the worker's `__main__` entry point reads
ONLY from environment variables (`PWC_GPU_ID`, `PWC_WORKER_ID`). CLI args were
completely ignored. All 8 workers per rig defaulted to `gpu_id=0` and
`worker_id=rig-6600:gpu0`, causing each new connection to overwrite the previous
one in the coordinator's `_workers` dict. Only 1 worker per rig (3 total) survived.
**Fix:** Added `export PWC_GPU_ID=` and `export PWC_WORKER_ID=` to the launch script
in `_tcp_launch_workers()`.

### Fix 5 — Data file missing on rrig6600b
**Problem:** `daily3.json` was missing on rrig6600b because the public repo clone
does not include data files.
**Fix:** `scp rzeus:~/distributed_prng_analysis/daily3.json rrig6600b:~/distributed_prng_analysis/daily3.json`

---

## Key Learnings — NEVER FORGET

1. **`pwc_worker_service.py` entry point uses ENV VARS only** — `PWC_GPU_ID`,
   `PWC_HOST`, `PWC_PORT`, `PWC_WORKER_ID`, `PWC_USE_ROCM`. CLI args `--gpu-id`
   etc. are parsed but NOT used in `__main__`. The launch script MUST export these
   env vars.

2. **`create_connection(timeout=N)` sets socket timeout for ALL subsequent reads** —
   after handshake, always call `sock.settimeout(None)` to make the socket blocking
   before entering the `_wait_for_init()` blocking read.

3. **`HEARTBEAT_TIMEOUT_S` must be > online_wait_timeout + init_broadcast_time** —
   workers go silent after sending `online` while waiting for `init`. Default 60s
   was too short. 300s gives ample headroom.

4. **The public repo does NOT contain data files** — `daily3.json`, `trse_context.json`
   and other data files must be manually SCP'd to any rig rebuilt from the public repo.

5. **S161 v2 `pwc_worker_service.py` was never in the public repo** — it was written
   directly and committed to the private repo only. Always verify the public repo
   has all critical files after any architectural change.

6. **Rig cloning from public repo will be missing** all files that were only
   committed to the private repo. After any rig rebuild, always run a full file
   sync from Zeus.

---

## Files Modified This Session

| File | Commits | Change |
|------|---------|--------|
| `window_optimizer_integration_final.py` | `526ae93` | Pass `pwc_transport` to `run_trial_persistent()` |
| `persistent/pwc_worker_service.py` | `a42e71c`, `c0615f3` | Restore v2 two-phase startup + socket blocking fix |
| `persistent/pwc_transport_tcp.py` | `45cb6a6`, `aacb0b1` | Restore S162 baseline + increase heartbeat timeout 60→300s |
| `persistent_worker_coordinator.py` | `526ae93`, `b632a26` | TCP online timeout 30→60s + PWC_GPU_ID/PWC_WORKER_ID env vars |

---

## Post-Session Deployment Checklist (for any future rig rebuild)

```bash
# 1. Data files
scp rzeus:~/distributed_prng_analysis/daily3.json <rig>:~/distributed_prng_analysis/
scp rzeus:~/distributed_prng_analysis/trse_context.json <rig>:~/distributed_prng_analysis/

# 2. Verify pwc_worker_service.py has v2 two-phase protocol
ssh <rig> "grep 'sent online\|waiting for init\|sent ready' ~/distributed_prng_analysis/persistent/pwc_worker_service.py"

# 3. Verify HEARTBEAT_TIMEOUT_S = 300.0
ssh <rig> "grep HEARTBEAT_TIMEOUT_S ~/distributed_prng_analysis/persistent/pwc_transport_tcp.py"

# 4. Verify PWC_GPU_ID in launch script
ssh rzeus "grep PWC_GPU_ID ~/distributed_prng_analysis/persistent_worker_coordinator.py"
```

---

## Final State

| Component | Status |
|-----------|--------|
| TCP-PWC transport | ✅ 26/26 GPUs active |
| Throughput | ✅ 537,408 seeds/sec |
| rrig6600 (.120) | ✅ Active, 146K s/s |
| rrig6600b (.154) | ✅ Active, 264K s/s |
| rrig6600c (.162) | ✅ Active, 125K s/s |
| Zeus RTX 3080Ti | ✅ Active, 1,071 s/s |
| daily3.json | ✅ All 3 rigs |
| HEARTBEAT_TIMEOUT_S | ✅ 300s |
| PWC_GPU_ID env var | ✅ In launch script |
| pwc_worker_service v2 | ✅ Two-phase protocol |

---

## Commits This Session

| Commit | Description |
|--------|-------------|
| `526ae93` | fix(s163): pass pwc_transport to run_trial_persistent + tcp_wait_online 30→60s |
| `45cb6a6` | fix(s163): restore pwc_transport_tcp.py to S162 victory baseline |
| `a42e71c` | fix(s163): restore pwc_worker_service v2 two-phase startup |
| `c0615f3` | fix(s163): set socket to blocking after handshake |
| `aacb0b1` | fix(s163): increase TCP heartbeat timeout 60s→300s |
| `b632a26` | fix(s163): add PWC_GPU_ID and PWC_WORKER_ID env vars to TCP launch script |

---

*Session S163 — 2026-04-12 — Team Alpha*
