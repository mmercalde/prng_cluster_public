# SESSION CHANGELOG — S153
**Date:** 2026-03-23  
**Commits:** None — diagnostic session only  
**Status:** CLOSED — java_lcg_reverse kernel arg mismatch and IPC path ruled out as crash causes

---

## Summary

S153 was a pure diagnostic session continuing the rrig6600c crash investigation
from S152. The S152 hypothesis was that a `java_lcg_reverse` kernel argument
mismatch in slim_v1 was causing crashes on rrig6600c. Two diagnostic stages were
run to test this. Both passed — ROCm silently tolerates the wrong kernel args and
the full IPC simulation with a reverse job completed successfully. The crash cause
remained unknown at end of S153, handing off to S154.

---

## Diagnostic Work

### Context — S152 Hypothesis
`sieve_gpu_worker.py` slim_v1: in the `elif family_name in ('java_lcg', 'java_lcg_reverse'):` 
branch, both families receive `uint64(a), uint64(c)` args. But `java_lcg_reverse`
kernel only expects `int offset` — `a` and `c` are hardcoded inside the kernel.
This sends 2 extra unexpected args to the reverse kernel.

S152 theory: this mismatch was crashing rrig6600c specifically because its
incomplete amdgpu library installation (fixed S152 via `apt-get install`) made
it less tolerant of the mismatch than rrig6600/rrig6600b.

### Stage 8 — Reverse Kernel Argument Test
**Script:** `slim_v1_diag_rrig6600c_v2.py --only 8`

**Stage 8A — Correct args (offset only):**
```
GPU 0-7: ✅ correct args — passed
Stage 8A PASSED — correct args work on all 8 GPUs
```

**Stage 8B — Wrong args (a, c, offset — current slim_v1 bug):**
```
GPU 0-7: ⚠️  wrong args — DID NOT CRASH (ROCm tolerated it)
Stage 8B: All GPUs survived wrong args — ROCm tolerated the mismatch
```

**Conclusion:** Kernel arg mismatch is a real code bug but NOT the crash cause.
ROCm on rrig6600c tolerates the extra arguments silently. The fix patch
`apply_s153_java_lcg_reverse_args_fix.py` is still correct code hygiene but
will not fix the production crash.

### Stage 9 — Full IPC Simulation (Reverse Pass)
**Script:** `slim_v1_diag_rrig6600c_v2.py --only 9`

Spawned 8 actual persistent worker subprocesses and sent each a real
`java_lcg_reverse` job through the full IPC path:

```
GPU 0-7: ✅ worker ready
All 8 workers alive — sending REVERSE pass job...
GPU 0-7: ✅ reverse job done in 3.1-4.1s — fmt=slim_v1 survivors=0
Stage 9 PASSED — All 8 reverse jobs completed successfully
```

**Conclusion:** Full reverse pass IPC simulation passes on rrig6600c. The crash
does not occur in the diagnostic environment — it only occurs in production
under sustained multi-hundred-chunk load.

---

## Key Insight from S153

The diagnostic environment passes all stages but production crashes. The critical
difference:
- **Diagnostic:** 1 reverse job per worker
- **Production:** ~21 reverse jobs per worker sequentially over 5+ minutes

This points to a **resource accumulation** problem — something grows with each
job until the system crashes. This became the S154 investigation focus.

---

## Handoff to S154

- java_lcg_reverse kernel arg mismatch: ruled out as crash cause, still a code bug
- IPC path: ruled out as crash cause
- Hypothesis going into S154: memory accumulation over many sequential jobs
- Next step: run production with persistent crash monitor log to capture kern.log evidence

---

## Architecture Invariants Added S153

- **[S153]** `slim_v1_diag_rrig6600c_v2.py` adds Stage 8 (kernel arg test) and Stage 9 (reverse IPC simulation)
- **[S153]** Crash monitor MUST write to `~/rig_crash_monitor_persistent.log` with `>>` append — `/tmp` is wiped on reboot
