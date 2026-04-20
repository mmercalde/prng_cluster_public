# SESSION CHANGELOG — S164
**Date:** 2026-04-19
**Session:** S164
**Focus:** Cluster hardening, crash forensics, Kaspa OC profile, Zeus compute mode fix
**Author:** Claude (Team Alpha Lead Dev)
**Status:** CLOSED
**HEAD at session start:** `884b424`
**HEAD at session end:** `884b424` (no code commits this session — infrastructure only)

---

## Session Achievements

1. rrig6600 card0 PCIe riser reseat — 16.0 GT/s Gen4 confirmed healthy
2. NPZ v3 schema validated end-to-end against Steps 2–6 via survivor_loader sandbox
3. Identified and fixed web_dashboard.py port mismatch (5000→5002) + SSH stdin detach
4. Deep crash forensics — full journalctl + netconsole analysis of ring_reset_failure
5. Identified amdgpu HDMI audio suspend as reset cascade trigger — deployed audio=0
6. Identified dm_suspend (Display Core) as secondary hang path — deployed dc=0
7. Deployed full Kaspa kHeavyHash OC profile to all 24 AMD GPUs (2250MHz / -150mV)
8. Updated DPM pin service to apply Kaspa profile at boot — survives reboots
9. Deployed lockup_timeout=30000 + gpu_recovery=1 to all 3 rigs
10. Identified and permanently fixed Zeus EXCLUSIVE_PROCESS bug — 40-session recurring issue closed
11. Sudoers NOPASSWD deployed on Zeus + all 3 rigs

---

## Crash Analysis — ring_reset_failure

### What Happened
GPU `0000:19:00.0` (card7) on rrig6600 experienced a KFD compute queue preemption
failure at 17:39:22 during Trial 1 of the S164 production run.

### Complete Event Timeline
| Time | Event |
|------|-------|
| 17:39:22 | KFD queue preemption failed, doorbell 0x80004008 — attempt 1 |
| 17:39:26 | KFD preemption failed — attempt 2. 14 vital buffers evicted |
| 17:39:30 | SMU 0xFFFFFFFF. Ring buffer OOB. PSP suspend -22. MODE1 reset failed (-121) |
| 17:39:40 | SDMA0 timeout (seq 146→150). sdma0 reset test -110 ETIMEDOUT |
| 17:39:43 | qcm fence wait timeout. "cp might be in unrecoverable state" |
| 17:41:56 | 10 kworker threads blocked >122s. dm_suspend hang in reset path |
| 17:43:13 | Second capture — workers 8/8/8 all rigs. Full recovery in 3.5 min |

### Root Cause
KFD-initiated reset (not DRM timeout) — one HIP kernel on card7 entered a
non-preemptable state. No ring timeout ever fired because KFD got there first.
Contributing factors: 2750MHz clock above mining-proven stable point, Display Core
active during reset path (dm_suspend hang), HDMI audio component causing PSP suspend
failure, lockup_timeout disabled (-1).

### Comparison to S163 GCVM crashes
This was a softer failure — MODE1 reset attempt (recoverable), not fatal
`device lost from bus!`. Workers auto-recovered in 3.5 minutes without reboot.

---

## Infrastructure Changes This Session

### AMD Rigs — All 3 (rrig6600, rrig6600b, rrig6600c)

**amdgpu.conf** (updated):
```
options amdgpu gfxoff=0 audio=0
```
`audio=0` added — prevents amdgpu from registering HDMI audio components.
Eliminates `failed to suspend display audio` cascade during GPU reset.

**GRUB** (updated on all 3 rigs):
```
Added:   amdgpu.dc=0              # Disable Display Core — headless nodes
Added:   amdgpu.lockup_timeout=30000  # 30s ring timeout for long HIP kernels
Added:   amdgpu.gpu_recovery=1    # Enable GPU recovery on reset failure
Removed: amdgpu.sched_jobs=256    # Not taking effect (kernel showed 32), removed
```

**amdgpu-dpm-pin.service** (updated):
Full Kaspa kHeavyHash OC profile applied at boot to all 8 cards per rig:
- SCLK: 2250MHz (state 1 custom OD) — proven mining-stable on Gigabyte Navi 23
- VDD: -150mV offset (~850mV) — matches HiveOS Kaspa profile
- MCLK: 1000MHz — compute bound, memory unchanged
- DPM: manual

**Verification — all 3 rigs confirmed:**
```
lockup_timeout=30000  ✅
gpu_recovery=1        ✅
dc=0                  ✅
gfxoff=0              ✅
audio=0               ✅
SCLK: 2250MHz × 8    ✅
OD_VDDGFX: -150mV    ✅
No AMD ALSA cards     ✅
DPM service: active   ✅
```

### Zeus

**nvidia-gpu-policy.service** (fixed):
Changed `EXCLUSIVE_PROCESS` → `DEFAULT`. This service was silently overriding
`nvidia-compute-mode.service` on every boot since S125b — the root cause of
40+ sessions of "Zeus compute mode reverted to Exclusive_Process after reboot."

Architecture invariant from TODO_MASTER_S132:
> Zeus GPU compute mode: DEFAULT (never Exclusive_Process) — EXCLUSIVE_PROCESS
> breaks n_parallel>1 (confirmed S125b)

**Verification:**
```
Default, Enabled    # GPU 0 ✅
Default, Enabled    # GPU 1 ✅
Survives reboot     ✅
sudo without password from ser8 ✅
```

### ser8

**web_dashboard.py** (fixed):
- Port 5000 → 5002 in `__main__` block (fuser kill, banner, app.run)
- Root cause: monitor_all.sh linked to 5002, dashboard bound to 5000 — never matched

**monitor_all.sh** (fixed):
- Added `</dev/null` before `&` in SSH dashboard launch
- Root cause: `nohup ... &` without stdin detach gets killed when SSH connection closes

---

## Kaspa Profile Research Summary

Our sieve workload characteristics vs mining algorithms:
- **NOT** Ethash — memory bandwidth bound, short kernels
- **CLOSEST TO** kHeavyHash (Kaspa) — pure integer arithmetic, fully compute bound,
  long sustained kernel runs, minimal memory pressure
- **Also similar to** Autolykos v2 (Ergo) — iterative integer arithmetic on candidates

Mining community lesson: ETH mining used ~1100MHz core (memory bound — core irrelevant).
Kaspa mining used 2250MHz core / 850mV (compute bound — core clock matters).
Our sieve = compute bound → Kaspa profile is the correct reference.

Prior crash at 2750MHz / stock voltage was ~500MHz above proven-stable compute clock.

---

## NPZ Validation Results

Synthetic NPZ with exact 22-field schema (matching live `bidirectional_survivors_all.npz`):
- `survivor_loader.py` v2.0 detection: NPZ v3 ✅
- Step 2 required fields: NONE missing ✅
- Steps 3–5 required fields: NONE missing ✅
- All arrays same length (n=20,914): ✅
- Dict mode (legacy steps): all 22 fields reconstructed ✅

Current accumulator: **20,914 seeds** in `bidirectional_survivors_all.npz`
Ready for Step 2 — Scorer Meta-Optimizer.

---

## Files Delivered This Session

| File | Destination | Purpose |
|------|-------------|---------|
| `preseed_s164_study.py` | Zeus `~/distributed_prng_analysis/` | Optuna warm-start with S162 victory config |
| `apply_kaspa_dpm_profile.sh` | All 3 rigs | One-shot Kaspa OC profile application |
| `update_dpm_pin_service.sh` | All 3 rigs | DPM pin service update (boot-persistent) |
| `update_grub_compute.sh` | All 3 rigs | GRUB compute params update |
| `web_dashboard.py` | Zeus `~/distributed_prng_analysis/` | Port fix |
| `monitor_all.sh` | ser8 `~/` | SSH stdin detach fix |

---

## Known Issues — NOT Resolved This Session

| Issue | Priority | Notes |
|-------|----------|-------|
| Crash dump directory — 876 entries | Medium | Needs retention cap + purge script |
| UFW port 5002 not open on Zeus | Medium | Dashboard unreachable from browser |
| `web_dashboard.py` + `monitor_all.sh` uncommitted | Low | Deployed but not in git |
| crash_forensic_daemon.py — 3 bugs | Low | false-DOWN, log-caching, dmesg empty |
| Step 2 not yet run | P1 | 20,914 survivors waiting |
| Missing 2,851 survivors from S163-P2 run | Low | Recovery pending |
| Zeus `_localhost_semaphore=2` | Deferred | TB proposal needed |

---

## Git — Commit and Dual-Push

```bash
scp ~/Downloads/SESSION_CHANGELOG_20260419_S164.md \
    rzeus:~/distributed_prng_analysis/docs/

ssh rzeus "cd ~/distributed_prng_analysis && \
  source ~/venvs/torch/bin/activate && \
  git add docs/SESSION_CHANGELOG_20260419_S164.md && \
  git commit -m 'docs(s164): session changelog — cluster hardening + Kaspa OC profile

Infrastructure changes (no code commits this session):
- amdgpu audio=0 on all 3 rigs — eliminates HDMI suspend cascade
- amdgpu.dc=0 GRUB — disables Display Core, removes dm_suspend hang
- amdgpu.lockup_timeout=30000 + gpu_recovery=1 on all 3 rigs
- Kaspa kHeavyHash OC profile: 2250MHz/-150mV on all 24 AMD GPUs
- DPM pin service updated — Kaspa profile survives reboots
- Zeus nvidia-gpu-policy.service: EXCLUSIVE_PROCESS→DEFAULT (40-session bug closed)
- Sudoers NOPASSWD: Zeus + all 3 rigs
- web_dashboard.py: port 5000→5002
- monitor_all.sh: SSH stdin detach fix

Crash analysis: KFD queue preemption failure on card7 (rrig6600).
Root cause: 2750MHz above Kaspa-proven 2250MHz stable clock.
Workers auto-recovered in 3.5min. No reboot required.

NPZ validation: 20,914 seeds, 22 fields, all Steps 2-6 compatible.
Ready for Step 2 — Scorer Meta-Optimizer.' && \
  git push origin main && git push public main"
```

---

*Session S164 — Team Alpha (Claude)*
*Cluster hardening complete. Kaspa OC profile deployed. Zeus bug closed.*
*20,914 survivors ready for Step 2.*
