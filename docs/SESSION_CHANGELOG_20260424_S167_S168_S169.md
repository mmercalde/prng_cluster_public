# Session Changelog — S167 / S168 / S169
**Date:** 2026-04-24
**Branch:** s167-clean
**HEAD:** b6cabe9
**Author:** Team Alpha (Claude) — reviewed by Team Beta (Michael)

---

## Session Summary

This session covered three major workstreams:

1. **S167** — WATCHER warm-start leak fix
2. **S168A-DIAG** — Passive startup hammer telemetry
3. **S168 + S169** — Startup jitter + per-worker pacing anti-hammer

---

## S167 — WATCHER Warm-Start Leak Fix

### Problem
WATCHER was unconditionally calling `get_best_step1_params()` from `step1_trial_history` DB and injecting historical best params into `final_params` on every Step 1 run. These params flowed through to the CLI as `--warm-start-window N --warm-start-offset N ...` even on fresh runs with no `study_name` provided.

This caused two problems:
- Fresh runs were silently converted into historically warm-started runs
- The forced first-trial workload from DB-selected historical configs was hammering AMD rigs on trial 0
- `TypeError: int() argument must be a string... not 'NoneType'` crash when session_idx conversion failed

### Root Cause
`agents/watcher_agent.py` — S140b block (lines ~1441-1455) called `get_best_step1_params()` unconditionally and wrote results into `final_params`. These are NOT S114-S116 resume args. They are later warm-start baggage introduced in S144/S166 that was never properly separated from the resume mechanism.

### Fix (agents/watcher_agent.py only)
Three surgical changes:

1. **Removed S140b warm-start DB injection block** — `get_best_step1_params()` no longer called during Step 1 execution
2. **Restored full `_INTERNAL_ONLY_PARAMS`** — all `warm_start_*` fields stripped from CLI
3. **Added None/empty string guard in CLI builder** — prevents `--study-name` with empty value or `--warm-start-window None`

### Verification
```
EXEC CMD: python3 window_optimizer.py ... --pwc-transport tcp --min-workers 24
```
No `--warm-start-*` args. No empty `--study-name`. Clean. ✅

### Resume Behavior (S114-S116 preserved)
Resume still works via explicit `study_name` in `--params`. WATCHER S145-R1 logic auto-sets `resume_study=True` when `study_name` is non-empty.

### Team Beta Ruling
- S166 commits `66462fb → 55262a7 → 3edfdf9` REJECTED
- S167 WATCHER-only fix APPROVED
- Root cause: S166 mixed resume, warm-start, and WATCHER passthrough into one tangled path

---

## S168A-DIAG — Passive Startup Hammer Telemetry

### Problem
rrig6600c (`192.168.3.162`) was crashing repeatedly with `gfxhub page fault` + `SQC (inst)` signature. Initial theories (hardware, cwsr, initramfs) were all incorrect.

### Diagnostic Patch
Added passive telemetry to `persistent/pwc_transport_tcp.py` (env-gated: `PRNG_PWC_STARTUP_DIAG=1`):
- Logs `job_assign` events with timestamp, worker_id, host, gpu_id
- Outputs to `logs/pwc_startup_diag_simple.jsonl`
- Zero behavior change — logging only

### Key Finding
```
All 26 GPUs received their first real job within a 129ms window
```

Specifically — all rrig6600c GPUs hit simultaneously:
- gpu2: t+0.001s
- gpu7: t+0.007s
- gpu5: t+0.023s
- gpu3: t+0.023s
- gpu4: t+0.096s
- gpu0: t+0.110s
- gpu1: t+0.115s
- gpu6: t+0.129s

**This is the startup hammer.** Zero stagger between ROCm warmup and full workload dispatch.

Michael identified this correctly from the beginning. The diagnostic data confirmed it.

---

## S168 — Startup Jitter (First-Assignment De-sync)

### Problem
129ms synchronized first-wave burst causes ROCm initialization spike on last-initialized rig.

### Fix (`persistent/pwc_transport_tcp.py`)
CRC32-based deterministic delay on each worker's first job assignment only:
```python
delay = (zlib.crc32(worker_id.encode("utf-8")) % (slots + 1)) / 1000.0
```
- Spreads first jobs across configurable window (e.g. 3 seconds)
- Same worker always gets same delay slot (deterministic, not random)
- Only affects first job — sustained throughput unaffected
- Env-gated: `PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=3` (default off)
- On worker disconnect, jitter resets — reconnected workers jitter again

---

## S169 — Per-Worker Pacing (Steady-State Burst Smoothing)

### Problem
After S168, Trial 2 crashes still occurred. Diagnostic data showed ~20+ assignments to rrig6600c in a 0.5-second window during sustained operation — not startup. The per-worker dispatch rate was unconstrained.

### Fix (`persistent/pwc_transport_tcp.py`)
Minimum gap enforcement between consecutive assignments to same worker:
```python
min_gap = float(os.environ.get("PRNG_PWC_PER_WORKER_MIN_GAP_SEC", "0") or 0)
if min_gap > 0:
    wait = min_gap - (now - last)
    if wait > 0:
        time.sleep(wait)
```
- 20ms gap per worker = smooth dispatch, no micro-bursts
- Does NOT reduce worker count or limit sustained throughput
- Env-gated: `PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.02` (default off)

---

## Test Results

| Run | Config | Result | Netconsole | Notes |
|-----|--------|--------|------------|-------|
| s168a_diag | 200k, 2 trials, diag only | Trial 1 ✅ Trial 2 ❌ | rrig6600c page fault at t+30s | Confirmed 129ms burst |
| s169_test | 200k, 2 trials, jitter+pacing | Trial 1 ✅ Trial 2 ✅ | rrig6600 SMU warnings (non-fatal) | First clean 2-trial run |

---

## Stability Test Plan (Tomorrow)

### Command Template
```bash
CAP=100000
TRIALS=5
ssh rzeus "cd ~/distributed_prng_analysis && \
  rm -f logs/pwc_startup_diag_simple.jsonl optimal_window_config.json && \
  truncate -s 0 logs/netconsole_all_rigs.log && \
  source ~/venvs/torch/bin/activate && \
  PRNG_PWC_STARTUP_DIAG=1 \
  PRNG_PWC_FIRST_ASSIGN_JITTER_SEC=3 \
  PRNG_PWC_PER_WORKER_MIN_GAP_SEC=0.02 \
  PYTHONPATH=. nohup python3 agents/watcher_agent.py \
  --run-pipeline --start-step 1 --end-step 1 --force-step 1 \
  --params '{\"min_workers\": 24, \"seed_cap_amd\": '$CAP', \"window_trials\": '$TRIALS'}' \
  > logs/stability_cap_${CAP}_t${TRIALS}_$(date +%H%M).log 2>&1 & echo PID: \$!"
```

### Test Matrix
```
CAP=100000 TRIALS=5
CAP=150000 TRIALS=5
CAP=200000 TRIALS=5
```

### Pass/Fail Criteria
```
PASS = all trials complete, netconsole clean
WARN = completes but netconsole has amdgpu/KFD faults
FAIL = any rig crash/reset/unreachable
```

### Track Both
- GPU faults (netconsole)
- Transport failures (script write failed on rrig6600)

---

## Cluster State

| Component | State |
|-----------|-------|
| Zeus HEAD | b6cabe9 (s167-clean branch) |
| rrig6600 | cwsr_enable=0 mcbp=0 ✅ |
| rrig6600b | stock ✅ |
| rrig6600c | stock (cwsr causes faster crashes on this rig) ✅ |
| S167 | Deployed, committed ✅ |
| S168A-DIAG | Deployed, committed ✅ |
| S168 jitter | Deployed, committed, default OFF ✅ |
| S169 pacing | Deployed, committed, default OFF ✅ |
| Accumulator | ~20,916 seeds (NPZ v3.1) |

---

## Key Learnings

1. **Michael's instinct was correct from the start** — startup hammering was always the root cause. The diagnostic data confirmed what he said days ago.

2. **Team Alpha/Beta structure is essential** — Team Beta found in minutes what Team Alpha couldn't see while defending its own code.

3. **S166 warm-start overreach** — mixing resume, warm-start, and WATCHER passthrough into one tangled path caused rig crashes by changing first-trial workload character.

4. **Two independent failure modes exist:**
   - Startup synchronization (S168 mitigated)
   - Steady-state assignment pressure (S169 mitigated)
   - System-level contention (I/O / ROCm / kernel) — still under investigation

5. **cwsr_enable=0 behaves differently per rig** — works on rrig6600, causes faster crashes on rrig6600c. Do not apply universally.

---

## Pending

- Stability curve test (tomorrow)
- DPM harness — 900mV/2100-2200MHz per rig (P1, deferred)
- BoTorch dual-GPU (after DPM harness)
- rrig6600 script write failed investigation
- S103 Part 2
