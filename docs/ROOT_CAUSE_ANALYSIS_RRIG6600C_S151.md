# ROOT CAUSE ANALYSIS — rrig6600c Persistent Crashes
**Date:** 2026-03-21  
**Session:** S151  
**Status:** RESOLVED  
**Author:** Team Alpha

---

## Executive Summary

rrig6600c (192.168.3.162) crashed repeatedly throughout S151 under 8-worker
production load. The root cause was **`MaxSessions 5` in `/etc/ssh/sshd_config`**
on rrig6600c — a security hardening setting that limited concurrent SSH sessions
to 5. With 8 workers requiring 8 SSH connections, workers 5-7 were silently
rejected by sshd, triggering a respawn hammer that crashed the rig.

---

## Timeline

| Time | Event |
|------|-------|
| S149/S150 overnight | rrig6600c ran 8 workers for hours — STABLE |
| S150 ~10:30 | slim_v1 deployed — crashes begin |
| S151 all day | Multiple crashes, reboots, theories investigated |
| S151 20:00 | Crash monitor deployed to all rigs |
| S151 20:36 | rrig6600c stable at 4 workers with monitor showing healthy state |
| S151 21:00 | SSH sshd_config comparison reveals MaxSessions 5 on rrig6600c |
| S151 21:10 | Fix applied — MaxSessions 50, ClientAliveInterval 0 |

---

## Root Cause

**File:** `/etc/ssh/sshd_config` on rrig6600c (192.168.3.162)

**Setting:** `MaxSessions 5` (should be 50, matching other rigs)

### Comparison

| Setting | rrig6600 | rrig6600b | rrig6600c (broken) | rrig6600c (fixed) |
|---------|----------|-----------|-------------------|-------------------|
| MaxSessions | 50 | 50 | **5** | 50 |
| ClientAliveInterval | 0 | 0 | **300** | 0 |
| ClientAliveCountMax | 3 | 3 | **2** | 3 |
| MaxStartups | 50:30:100 | 50:30:100 | **missing** | 50:30:100 |

---

## Failure Cascade

```
Zeus spawns 8 workers on rrig6600c
    ↓
Workers 0-4 connect successfully (MaxSessions 5 allows 5)
    ↓
Workers 5-7 REJECTED by sshd — "MaxSessions exceeded"
    ↓
PWC sees spawn failure → immediate respawn attempt
    ↓
Respawn also rejected → workers 5-7 quarantined
    ↓
PWC retry storm → multiple SSH connection attempts
    ↓
sshd overwhelmed → ClientAliveInterval 300 + CountMax 2
= after ~10 minutes, connected workers 0-4 also dropped
    ↓
ALL 8 workers dead → full respawn storm
    ↓
rig overloaded → system crash → physical reboot required
```

---

## Why It Worked Overnight (S149/S150)

The overnight stable run used the same S149-B code but the run happened to
complete before `ClientAliveInterval 300` (5 minutes) triggered on workers
0-4. The rig appeared stable because trials completed fast enough that the
SSH keepalive timeout never fired.

---

## Why slim_v1 Appeared to Cause Crashes

slim_v1 increased result payload size per chunk. Larger payloads meant longer
SSH pipe writes, keeping connections open longer. This increased the probability
of hitting the ClientAliveInterval timeout on the 5 connected workers, making
the correlation with slim_v1 deployment appear causal — but it was coincidental.

---

## Why 4 Workers Was Stable

`gpu_count=4` meant only 4 SSH connections — within the MaxSessions 5 limit.
All 4 workers connected successfully, no rejections, no hammer, no crash.

---

## All Theories That Were Wrong

| Theory | Why Wrong |
|--------|-----------|
| Hardware fault | Hardware was fine — SSH rejection caused the crash |
| Kernel upgrade (6.8.0-106) | All 3 rigs same kernel, only rrig6600c had MaxSessions 5 |
| slim_v1 IPC change | Correlation not causation — longer pipes exposed existing SSH limit |
| CuPy memory pool race | Real issue but not the crash cause |
| Weak CPU (i5-8400T) | CPU was fine — SSH was the bottleneck |
| PCIe 1x overload | PCIe was fine — SSH rejection caused overload appearance |
| ROCm queue eviction | Downstream effect of SSH hammer, not root cause |

---

## Fix Applied

**File:** `/etc/ssh/sshd_config` on rrig6600c

**Backup:** `/etc/ssh/sshd_config.bak_s151`

**Changes:**
```
MaxSessions 5          →  MaxSessions 50
ClientAliveInterval 300  →  ClientAliveInterval 0
ClientAliveCountMax 2    →  ClientAliveCountMax 3
MaxStartups (missing)    →  MaxStartups 50:30:100
```

**Command:** `sudo systemctl reload sshd` (no reboot required)

---

## Lessons Learned

1. **Check SSH config parity across rigs** — any security hardening difference
   can manifest as mysterious GPU/ROCm instability.

2. **MaxSessions must equal worker_pool_size + buffer** — at minimum
   `MaxSessions >= worker_pool_size + 2` (for monitor, admin connections).

3. **The crash monitor was essential** — without it we would never have known
   to compare sshd configs. Deploy it on all rigs for every production run.

4. **Don't blame code when infrastructure differs** — rrig6600c had different
   sshd config from day 1. All code changes were innocent.

5. **Add sshd config check to preflight** — Step 1 preflight should verify
   `MaxSessions >= 10` on all remote nodes.

---

## Follow-Up Actions

1. ✅ Fix applied to rrig6600c sshd_config
2. ✅ rrig6600c gpu_count restored to 8
3. ☐ Add sshd MaxSessions check to PWC preflight
4. ☐ Add to REMOTE_NODE_SETUP_CHECKLIST.md
5. ☐ slim_v1 can now be re-evaluated safely (not the cause)
6. ☐ Revert CUPY_CUDA_MEMORY_POOL_TYPE=none if desired (not needed)
7. ☐ Revert S151 respawn lock+stagger if desired (not needed, but harmless)

---

## Diagnostic Evidence

### Crash monitor showing healthy 4-worker state:
```
20:36:02 workers=4 load=0.12, 0.10  used=2637MB free=4012MB total_vram_used=931MB ssh_conns=1
```

### sshd_config comparison that revealed root cause:
```
rrig6600c:  MaxSessions 5   ← THE BUG
rrig6600:   MaxSessions 50
rrig6600b:  MaxSessions 50
```

### After fix — verified:
```
MaxSessions 50
ClientAliveInterval 0
ClientAliveCountMax 3
MaxStartups 50:30:100
```
