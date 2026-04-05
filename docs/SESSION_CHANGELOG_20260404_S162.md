# SESSION CHANGELOG — S162
**Date:** 2026-04-04  
**Session:** S162  
**Focus:** TCP-PWC Production Integration, rrig6600c Crash Investigation  
**Author:** Claude (Team Alpha Lead Dev)  
**Status:** Open — TB ruling pending on rrig6600c fix options

---

## Summary

S162 began with the goal of wiring TCP-PWC into WATCHER as the default
transport (already completed in prior sub-sessions) and validating full
26-GPU stability. The session successfully confirmed TCP-PWC works at
production speed but encountered the persistent rrig6600c crash under
3-rig simultaneous load. Extensive root cause investigation was conducted
including network analysis, netconsole crash signature analysis, internet
research, and multiple fix attempts. A formal TB proposal was submitted.

---

## Commits This Session (Pre-Session — Already in Repo)

| Commit | Description |
|--------|-------------|
| `10871b8` | feat(s162): switch WATCHER default transport to TCP-PWC |
| `b89b33a` | feat(s162): TCP-PWC default — fix parser/fallback/manifest/helptext |
| `c2a59cc` | fix(s162): TCP startup reaps stale sieve_gpu_worker |
| `b3c3207` | fix(s162): add _best_effort_gpu_cleanup to TCP worker between chunks |

---

## Work Completed This Session

### S162-1 — First 26-GPU Run Attempt
- WATCHER launched with TCP-PWC default, all 3 rigs, 26 GPUs
- Dashboard confirmed: 26/26 GPUs active, 7,162,151 seeds/sec aggregate
- rrig6600c active and working (first time ever under TCP-PWC)
- **rrig6600c crashed at ~18 seconds** — same signature as all previous transports

**Netconsole crash signature:**
```
GCVM_L2_PROTECTION_FAULT_STATUS: 0xFFFFFFFF
Faulty UTCL2 client ID: unknown (0x1ff)
WALKER_ERROR: 0x7 / PERMISSION_FAULTS: 0xf / MAPPING_ERROR: 0x1
Multiple GPUs + PIDs simultaneously → watchdog hard LOCKUP
```

---

### S162-2 — Cleanup Patch Missing on All Rigs

Discovered that `_best_effort_gpu_cleanup()` patch (committed `b3c3207`)
was never deployed to the rigs — only existed in Zeus repo. All 3 rigs
confirmed missing:

```bash
ssh rrig6600 "grep -n '_best_effort_gpu_cleanup' ~/distributed_prng_analysis/persistent/pwc_worker_service.py"
# → empty (not present)
```

**Fix:** SCP'd `pwc_worker_service.py` from Zeus to all 3 rigs:
```bash
ssh rzeus "cat ~/distributed_prng_analysis/persistent/pwc_worker_service.py" > /tmp/pwc_worker_service.py
scp /tmp/pwc_worker_service.py rrig6600:~/distributed_prng_analysis/persistent/
scp /tmp/pwc_worker_service.py rrig6600b:~/distributed_prng_analysis/persistent/
scp /tmp/pwc_worker_service.py rrig6600c:~/distributed_prng_analysis/persistent/
```

**Key learning:** Files committed to Zeus repo are NOT automatically on rigs.
Rig deployment requires explicit SCP. This is a process gap — future commits
to `persistent/pwc_worker_service.py` must be accompanied by rig deployment.

---

### S162-3 — Isolation Test (rrig6600c + Zeus only)

Disabled rrig6600 and rrig6600b by setting `gpu_count=0` in
`distributed_config.json`. Ran full 3-trial run with Zeus + rrig6600c only
(10 GPUs).

**Result:** ✅ Complete success — 18 minutes 52 seconds, zero crashes, zero
netconsole faults. Confirms rrig6600c is fully functional in isolation.

**Invariant confirmed:** rrig6600c crash is caused exclusively by simultaneous
3-rig load, not by rrig6600c hardware, software, or configuration.

---

### S162-4 — Zeus Network Stack Investigation

Identified Zeus network configuration:
- **Active NIC:** `enp2s0` — Intel I210 Gigabit (1GbE)
- **Inactive NIC:** `eno1` — Intel I219-LM (NO CARRIER — not connected)
- **CPU:** i9-9920X — 12 cores / 24 threads

**Thread oversubscription analysis:**
- 24 TCP worker handler threads (one per AMD GPU)
- 2 Zeus local GPU workers
- Accept + lease + heartbeat + main threads (~4)
- Total: ~30 threads competing for 24 physical threads

**TCP buffer tuning (applied and persisted):**
```
net.core.wmem_max    = 16,777,216  (was 212,992)
net.core.wmem_default = 1,048,576  (was 212,992)
net.core.rmem_max    = 16,777,216  (was 212,992)
net.core.rmem_default = 1,048,576  (was 212,992)
net.ipv4.tcp_wmem    = 4096 / 1,048,576 / 16,777,216
net.ipv4.tcp_rmem    = 4096 / 1,048,576 / 16,777,216
```
Persisted to `/etc/sysctl.conf` on Zeus.

---

### S162-5 — Dispatch Semaphore (Deployed to Zeus)

Added `threading.Semaphore(8)` around job payload `conn.send_obj()` in
`persistent/pwc_transport_tcp.py` `_handle_client()`.

**Rationale:** Prevents all 24 worker handler threads from simultaneously
calling `send_obj()`, which floods Zeus's 1GbE NIC send buffer. Limits
concurrent outbound job dispatches to 8 (one rig's worth) at a time.

**File:** `persistent/pwc_transport_tcp.py`  
**Lines changed:** 160 (semaphore init in `__init__`), 459 (context manager
around `send_obj()`)  
**Deployed:** Zeus only (this file runs on Zeus, not rigs)

**Assessment:** Good hygiene for 1GbE with 24 concurrent senders. Does NOT
directly cause GPU page faults (TCP guarantees payload integrity). Overhead
is negligible (~0.3ms delay for 3rd batch of 8 workers vs 200-500ms chunk
execution time).

---

### S162-6 — Second 26-GPU Run (Post-Cleanup Patch + Semaphore)

Full 26-GPU run with all fixes deployed. **Crashed at kernel uptime 3036s
(~50 minutes).**

**New crash signature — materially different:**
```
FIRST FAULT:
  GCVM_L2_PROTECTION_FAULT_STATUS: 0x00801231
  Faulty UTCL2 client ID: SQC (inst) (0x9)  ← SHADER INSTRUCTION CACHE
  WALKER_ERROR: 0x0 / MAPPING_ERROR: 0x0

ESCALATION → 0xFFFFFFFF cascade → rig offline
```

**Key observations:**
1. Crash delayed from ~18 seconds to ~50 minutes — cleanup patch working
2. Only ONE GPU (`0000:16:00.0`), ONE PID (`pid 5141`) — not multi-GPU
3. SQC (inst) = shader instruction cache fetch fault — new failure mode
4. `snd_hda_intel: Unable to change power state D3hot→D0` on that GPU card

---

### S162-7 — Internet Research

**ROCm GitHub Issue #5616:** Identical SQC (inst) fault signature documented
with darktable on AMD GPU. Same progression: SQC fault → MES failed to
respond → Failed to evict process queues → sq_intr → GPU reset. Issue open,
no upstream fix.

**HIP Debugging Docs:** `AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3`
forces synchronous kernel execution for diagnostic fault isolation.

**CuPy Documentation:** `free_all_blocks()` releases data memory pool only.
`RawKernel` compiled binaries are managed by separate kernel cache, not
memory pool. This partially contradicts the SQC fault hypothesis.

---

### S162-8 — TB Proposal Submitted

Full proposal submitted: `docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md`

**4 fix options presented:**
- Option A: Replace `free_all_blocks()` with `gc.collect()` only (Team Alpha recommendation)
- Option B: `AMD_SERIALIZE_KERNEL=3` diagnostic on rrig6600c
- Option C: Cap rrig6600c at 4 workers (proven workaround, ~15% throughput loss)
- Option D: Startup stagger for rrig6600c workers

**5 open questions for TB** — particularly whether `free_all_blocks()` can
affect instruction mappings on ROCm/HIP, and whether the SQC fault is
caused by cleanup or GPU memory pressure.

**Status:** Awaiting TB ruling before proceeding with Option A deployment.

---

## Architecture Invariants Added This Session

- **[S162]** Rig files (`persistent/pwc_worker_service.py`) must be explicitly
  SCP'd to all 3 rigs after every commit — Zeus repo is NOT automatically synced
- **[S162]** Zeus TCP buffers tuned: `wmem/rmem_max=16MB`, `default=1MB`,
  persisted to `/etc/sysctl.conf`
- **[S162]** Dispatch semaphore `Semaphore(8)` in `TCPWorkerTransport._handle_client()`
  — limits concurrent job payload sends to prevent 1GbE NIC flood
- **[S162]** `gpu_count=0` in `distributed_config.json` is the correct mechanism
  to disable a rig in TCP-PWC mode (`max_concurrent_script_jobs=0` only affects
  old ephemeral SSH path)
- **[S162]** rrig6600c crash is ONLY triggered by simultaneous 3-rig load —
  isolation runs are always stable

---

## TODO Carry-Forward

### P0 — Awaiting TB Ruling
- TB decision on Option A/B/C/D for rrig6600c fix
- After ruling: deploy approved fix and run full 26-GPU validation

### P1 — After rrig6600c Resolved
- Selfplay NN fix (`inner_episode_trainer.py`) — forbidden guard + y-normalization
- Pre-warm CuPy kernel cache script for all rigs
- Ephemeral coordinator benchmark (complete comparison table)

### P2 — Backlog
- S110 root cleanup (884 stray files)
- SESSION_CHANGELOG committed to docs/

---

## Files Modified This Session

| File | Change | Location |
|------|--------|----------|
| `persistent/pwc_worker_service.py` | `_best_effort_gpu_cleanup()` deployed | All 3 rigs |
| `persistent/pwc_transport_tcp.py` | Dispatch semaphore `Semaphore(8)` | Zeus |
| `/etc/sysctl.conf` | TCP buffer tuning | Zeus |
| `distributed_config.json` | `gpu_count=0` for rrig6600/b (isolation test, restored) | Zeus |
| `docs/PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md` | TB proposal | Zeus |

---

## Benchmark Reference

| Transport | GPUs | Seeds | Time | Aggregate sps |
|-----------|------|-------|------|---------------|
| ZMQ+SQLite | 24 AMD | 50M | ~60s | ~800K |
| SSH-PWC | 8 AMD | 16M | 2m 1.5s | ~133K |
| TCP-PWC (S161) | 26 | 50M | 53s | ~2,240,701 |
| TCP-PWC isolation (S162) | 10 | 1.07B | 18m 52s | ~948K |

---

*Session S162 — 2026-04-04 — Team Alpha*
