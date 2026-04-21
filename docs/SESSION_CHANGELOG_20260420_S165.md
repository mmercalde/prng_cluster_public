# SESSION CHANGELOG — S165
**Date:** 2026-04-20
**Session:** S165
**Engineer:** Team Alpha (Michael + Claude)
**HEAD:** b851511 (start) → pending commit
**Status:** COMPLETE — stock DPM baseline validated, Zeus TCP workers deployed, HSA debug trap fix applied

---

## Session Summary

Long and complex session. Multiple crash cycles traced to the Kaspa DPM profile
applied in S164. Reverted to stock `high` DPM — clean run confirmed. Zeus TCP
persistent workers deployed and partially working. HSA_ENABLE_DEBUG_TRAP=0 fix
applied. Session ended with a clean 2-trial run producing 17,721 bidirectional
survivors.

---

## Major Findings

### 1. Kaspa DPM Profile (2250MHz/-150mV) Was Causing All Crashes

**Root cause confirmed:** The 2250MHz/-150mV Kaspa mining profile applied in S164
was the primary source of today's crash cascade. Specifically:

- Michael's Kaspa mining profile was tuned for **efficiency** — minimum power at
  the edge of stability for microsecond mining kernels
- Our sieve kernels run for **seconds** — sustained voltage droop under long kernels
  pushed cards below minimum stable voltage
- rrig6600b has different card variants (100W PwrCap vs 135-145W on rrig6600) —
  these cards have a lower stable operating point and failed first
- Card `0000:09:00.0` on rrig6600b had repeated unrecoverable ASIC reset failures
  (-22) requiring multiple power cycles

**Fix:** Reverted all 3 rigs to stock `high` DPM performance level. Disabled
`amdgpu-dpm-pin.service` on all 3 rigs.

**Validation:** Clean 2-trial run, zero netconsole errors, 17,721 survivors.

### 2. KFD Debug Trap NULL Pointer Dereference

**Root cause:** `HSA_ENABLE_DEBUG_TRAP` was not disabled. Under concurrent GPU
reset + worker init, KFD's debug trap activation called `kq_acquire_packet_buffer`
on a NULL queue pointer:

```
kq_acquire_packet_buffer+0x1c [amdgpu]
pm_send_set_resources [amdgpu]
kfd_dbg_trap_activate [amdgpu]
kfd_ioctl_set_debug_trap [amdgpu]
BUG: kernel NULL pointer dereference, address: 0000000000000030
```

This cascaded across all 3 rigs simultaneously, causing Python workers to block
for 368+ seconds and triggering forced reboots.

**Fix:** Added `HSA_ENABLE_DEBUG_TRAP=0` to `ROCM_ENV_VARS` in
`persistent_worker_coordinator.py` (line 125). Propagated to all remote workers
via SSH launch script. KFD debug trap is not needed for compute workloads.

### 3. Zeus TCP Persistent Workers — Partially Working

**S165-ZEUS-TCP patch deployed** — `_tcp_launch_workers()` now also launches
2 local `pwc_worker_service` processes on Zeus (CUDA mode, connecting to
`127.0.0.1:5600`). Workers confirmed connecting:

```
[PWC-TCP] Zeus GPU0 local worker launched — PID=8382
[PWC-TCP] Zeus GPU1 local worker launched — PID=8383
[zeus_gpu0] handshake complete, connected to 127.0.0.1:5600
[zeus_gpu1] handshake complete, connected to 127.0.0.1:5600
```

**Remaining issue:** `isinstance(wh, WorkerNode)` guard in `_run_once()` at
line 1057 still routes Zeus WorkerNode objects to `_dispatch_local_sieve()`
subprocess path instead of `_dispatch_to_tcp()`. The sed patch was applied
but did not take effect in time for the production run.

**Impact on 3080Ti throughput — CRITICAL:**

When Zeus TCP workers are on the subprocess path AND AMD rigs are active,
the RTX 3080Ti GPUs contribute almost nothing:

| State | Zeus throughput | Note |
|-------|----------------|------|
| Zeus alone (TCP workers) | ~582,000 s/s | 291K per 3080Ti |
| Zeus alone (subprocess) | ~10,000 s/s | subprocess overhead |
| Zeus + 24 AMD (subprocess) | ~8,000 s/s | AMD consumes all chunks |

The 3080Ti GPUs are **monsters** — 291,000 s/s each when properly utilized.
When paired with the AMD rigs and on the subprocess path, they drop to ~4,000 s/s
because the coordinator dispatches chunks to AMD workers faster than Zeus can
process them via subprocess. The AMD rigs starve Zeus of work.

**Fix needed next session:** Ensure `isinstance(wh, WorkerNode)` guard is
removed from `_run_once()` so Zeus WorkerNodes route through TCP dispatch.
Verify with grep:
```bash
ssh rzeus "grep -n 'isinstance(wh, WorkerNode)' \
  ~/distributed_prng_analysis/persistent_worker_coordinator.py"
```
Should return nothing. If present, run:
```bash
ssh rzeus "sed -i 's/if self._tcp_transport is not None and not isinstance(wh, WorkerNode):/if self._tcp_transport is not None:/' \
  ~/distributed_prng_analysis/persistent_worker_coordinator.py"
```

### 4. DPM Harness — Designed, Not Yet Built

**Finding:** Stock `high` DPM is stable for short runs (2 trials) but dynamic
frequency scaling between trials causes instability in longer runs (5-50 trials).
`manual` DPM with fixed clock/voltage is required for production.

**Why Kaspa profile was wrong:**
- Mining goal: minimum power, efficiency, edge of stability
- Our goal: maximum stability, sustained compute, zero crashes
- -150mV was the efficiency floor for microsecond mining kernels
- Our second-long kernels push the voltage droop beyond the stability floor

**Correct approach — stability first:**
- Use **absolute voltage** (not offset) — HiveOS uses absolute mV values
- Start at **900mV** (generous, not efficiency-tuned)
- Core: 2100-2200MHz (below boost ceiling, above minimum)
- Memory: per-rig variant (1000MHz for rrig6600, 875MHz for rrig6600b/c)
- Never reduce voltage looking for efficiency — stability is the only goal

**Live clock snapshot under load (captured during clean run):**

| Rig | SCLK range | MCLK | PwrCap | Temp |
|-----|-----------|------|--------|------|
| rrig6600 | 2639-2755MHz | 1000MHz | 135-145W | 48-55°C |
| rrig6600b | 2585-2740MHz | 875MHz | 100W | 48-55°C |
| rrig6600c | 2595-2750MHz | 875MHz | 100W | 52-54°C |

**Proposed production DPM profile (to be validated by harness):**

| Rig | Core | Voltage | Mem | Mode |
|-----|------|---------|-----|------|
| rrig6600 | 2200MHz | 900mV absolute | 950MHz | manual |
| rrig6600b | 2100MHz | 900mV absolute | 875MHz | manual |
| rrig6600c | 2100MHz | 900mV absolute | 875MHz | manual |

**Harness design:** Run all 8 GPUs per rig simultaneously under real sieve
workload. Apply profile. Run 30 minutes. Watch netconsole. Zero faults = PASS.
Same methodology as HiveOS mining tuning — whole rig under real load, not
isolated GPU testing. Test rrig6600b first (most unstable card variant).

### 5. Sudoers NOPASSWD Persistence Issue

**Finding:** `/etc/sudoers.d/michael-cluster` gets wiped on hard crash-forced
reboots because the filesystem doesn't sync before power cycle. Survives
intentional reboots.

**Fix applied:** Always run `sudo sync` after writing critical config files.
File now named `michael-cluster` (not `michael-nopasswd`) on all 4 nodes.

### 6. VPS NAT Rule — Duplicate Entry Fixed

Old rule forwarded `45.32.131.224:5002 → 192.168.3.127:5000` (wrong port).
New rule: `45.32.131.224:5002 → 192.168.3.127:5002` (correct).
Saved to `/etc/iptables/rules.v4` on VPS.

---

## Clean Run Results

**Config:** 2 trials, `seed_cap_amd=100000`, stock `high` DPM, `HSA_ENABLE_DEBUG_TRAP=0`

```
Bidirectional survivors: 17,721
Forward survivors: 3,434,053
Reverse survivors: 3,435,984
Total seeds: 1,073,741,824
Elapsed: ~14 minutes
26/26 GPUs active
Netconsole: SILENT throughout ✅
```

---

## Files Changed This Session

| File | Change |
|------|--------|
| `persistent_worker_coordinator.py` | S165-ZEUS-TCP: Zeus local persistent workers in TCP mode |
| `persistent_worker_coordinator.py` | HSA_ENABLE_DEBUG_TRAP=0 added to ROCM_ENV_VARS (line 125) |

---

## Cluster State — End of Session

| Component | State |
|-----------|-------|
| Zeus | Rebooted clean, nvidia-gpu-policy DEFAULT ✅ |
| rrig6600 | Stock `high` DPM, amdgpu-dpm-pin DISABLED ✅ |
| rrig6600b | Stock `high` DPM, amdgpu-dpm-pin DISABLED ✅ |
| rrig6600c | Stock `high` DPM, amdgpu-dpm-pin DISABLED ✅ |
| Sudoers NOPASSWD | All 4 nodes ✅ |
| HSA_ENABLE_DEBUG_TRAP=0 | Deployed to Zeus coordinator ✅ |
| Zeus TCP workers | Launching but routing issue remains ⚠️ |
| NPZ | 17,721 new survivors ready for Step 2 |

---

## P1 Next Session

1. **Fix Zeus TCP routing** — verify/apply `isinstance` guard removal
2. **Run NPZ conversion** — `convert_survivors_to_binary.py`
3. **Run Step 2** — Scorer Meta-Optimizer with new survivors
4. **Build DPM harness** — validate manual 900mV profile on rrig6600b first
5. **Production run** — 5-10 trials with validated DPM profile

---

## TODO Carry-Forward

- DPM harness — design complete, implementation pending
- Zeus 3080Ti full utilization — `isinstance` guard fix
- crash_forensic_daemon.py — 3 bugs (false-DOWN, log-caching, dmesg empty)
- Selfplay NN fix in `inner_episode_trainer.py`
- S110 root cleanup (884 files)

