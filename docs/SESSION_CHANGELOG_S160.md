# Session S160 Changelog
**Date:** 2026-04-03
**Team:** Team Alpha
**Status:** COMPLETE — stability fix confirmed, moving to S161

---

## Summary

S160 goal was to validate the S159G ROCm env var + iommu=pt fix via TB acceptance test and stability run. The env var fix was confirmed operational but insufficient — GPU crashes continued. Root cause was identified and patched during this session.

---

## Commits This Session

| Hash | Description |
|------|-------------|
| `55b2645` | feat(s160): coordinator worker env-invariant check v1 |
| `4fcfa51` | fix(s160-v2): TB-approved env check — systemd MainPID, 10s settle, diagnostics |
| `ecc2a88` | fix(s160-v3): add XDG_RUNTIME_DIR to systemctl --user SSH calls |
| `815cd06` | fix(s160-v4): cgroup-based PID discovery, D-Bus-free |
| `7e1a62e` | fix(s160-v5): TB-approved inter-chunk GPU cleanup in ZMQ worker |

---

## Env-Invariant Check — v1 through v4

### Problem
TB-approved env-invariant check (verify all 5 ROCm protective vars in live worker environ before chunk dispatch) failed across 4 iterations.

### Root Cause (v1–v3)
`systemctl --user` requires a D-Bus session bus. SSH exec sessions (non-interactive, non-login) have no D-Bus session — `XDG_RUNTIME_DIR` alone is insufficient. The bus socket must also be present and the session active.

**Proof:** `bash -c "systemctl --user show ..."` returns `Failed to connect to bus: No such file or directory` in all SSH exec contexts regardless of `XDG_RUNTIME_DIR`.

### Fix (v4 — `815cd06`)
Replaced `systemctl --user` with cgroup-based PID discovery — scans `/proc/*/cgroup` on the remote host for the unit name. Every process launched under a systemd unit has that unit's name in its cgroup entry. D-Bus-free, works in all SSH contexts on Ubuntu 22.04 cgroup v2.

Combined PID discovery and environ read into a single SSH call.

### Acceptance Test Result
```
[EnvCheck] PASS on 192.168.3.120: MainPID=100024 all ROCm protective vars confirmed.
[EnvCheck] PASS on 192.168.3.154: MainPID=8938 all ROCm protective vars confirmed.
[EnvCheck] PASS on 192.168.3.162: MainPID=7221 all ROCm protective vars confirmed.
[EnvCheck] All AMD rigs passed env-invariant check. Proceeding to chunk dispatch.
```

---

## GPU Crash Investigation

### Crash Signature
```
GCVM_L2_PROTECTION_FAULT_STATUS:0xFFFFFFFF
WALKER_ERROR: 0x7 / PERMISSION_FAULTS: 0xf / MAPPING_ERROR: 0x1
```
`0xFFFFFFFF` is not a normal page fault value — it indicates PCIe read returned all ones, meaning the device stopped responding on the bus mid-operation. GPU entered unrecoverable state requiring physical riser power cycle to recover (fan at 100%, OS-invisible until power cycled).

### Crash Pattern
- rrig6600 (.120) and rrig6600c (.162): crashed repeatedly under load
- rrig6600b (.154): never crashed across entire session
- Crashes were workload-triggered and time-delayed (intermittent)
- Crash moved between rigs depending on which was under active load

### Key Diagnostic Question
"Why did this never fail with the original ephemeral coordinator?"

The ephemeral coordinator spawns a fresh worker process per chunk — process death between chunks provided implicit GPU memory cleanup. Persistent ZMQ workers keep the ROCm/HIP context alive indefinitely across all chunks with no cleanup between executions.

### Root Cause (Leading Runtime Hypothesis — TB language)
`_best_effort_gpu_cleanup()` exists in `sieve_filter.py` (TB-approved 2026-01-26, called at lines 232, 387, 693) but was **never called** by `zmq_sqlite_worker.py`. GPU memory allocator state, ROCm HIP context state, and CuPy memory pools accumulated across hundreds of chunks. After sufficient accumulation the GPU memory controller hit an unrecoverable fault state.

### Fix (v5 — `7e1a62e`)
Added `_best_effort_gpu_cleanup()` call to the ZMQ worker main loop after each chunk result is sent and before next job fetch. Placement is deliberate — cleanup cannot delay or suppress result delivery for a completed chunk.

```python
_send_result(result)

# S160-v5 (TB-approved): best-effort GPU cleanup between chunks.
try:
    from sieve_filter import _best_effort_gpu_cleanup
    _best_effort_gpu_cleanup()
except Exception:
    pass
```

### Acceptance Test Result
Full bidirectional trial (java_lcg forward + reverse + java_lcg_hybrid forward) completed with:
- Zero GPU crashes
- Zero netconsole fault events from any rig
- All 3 AMD rigs active throughout
- rrig6600c (.162) ran past all previous crash thresholds

---

## Cross-Cutting Finding — PWC Transport

The same missing cleanup gap applies to `persistent_worker_coordinator.py`. The S156 PWC SSH failures exhibited identical characteristics: intermittent, workload-triggered, resolved by falling back to the ephemeral coordinator. The S156 cleanup patches addressed pre-spawn zombie cleanup (correct) but not inter-chunk GPU memory accumulation (the actual mechanism).

**Recommendation for S161:** Apply `_best_effort_gpu_cleanup()` to the PWC TCP worker execution loop before benchmarking. Same placement: after result delivery, before next job fetch.

---

## ZMQ Architecture Limitations Confirmed

1. **Throughput:** ZMQ delivers ~31K sps/GPU vs S130 persistent workers at ~115K sps/GPU. The ZMQ broker + SQLite lease cycle costs approximately 73% of per-GPU throughput.

2. **Cross-pass lease coordination:** The coordinator dispatches all sieve passes into a shared job queue simultaneously. Workers finishing one pass cannot prioritize the next — chunks from the next pass expire while workers drain the current pass. All 537 hybrid chunks failed via lease exhaustion during trial 2 (not a crash — workers were alive, finishing the reverse pass).

3. **Cleanup overhead:** `_best_effort_gpu_cleanup()` between chunks introduces a cross-pass stall of ~26 minutes when transitioning between forward and reverse sieve. This is the cleanup running on workers that hold leases against the next pass's chunks.

---

## Pending Items for S161

- [ ] Apply inter-chunk cleanup to PWC TCP worker loop
- [ ] PWC TCP transport benchmark (deferred from S159)
- [ ] S160 session changelog committed to docs/ ← this file
- [ ] Normalize rrig6600 .bashrc (remove PS1 guard before activate)
- [ ] Update TODO_MASTER

---

## Architecture Invariants (Unchanged)
- seed_cap_amd: 2,000,000 / seed_cap_nvidia: 5,000,000
- ZMQ ports: job=5557, result=5558
- Zeus semaphore: 2
- Required ROCm vars: `HSA_ENABLE_SDMA=0`, `HSA_ENABLE_RUNTIME_POWER_MGMT=0`, `AMDGPU_NO_POWER_PROFILE=1`, `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `ROCR_VISIBLE_DEVICES=<gpu_id>`
