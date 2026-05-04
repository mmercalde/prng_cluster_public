# SESSION CHANGELOG — S170
**Date:** 2026-04-25 (Saturday, PDT)
**Branch:** s167-clean
**HEAD at session start:** a6cd55e
**Theme:** Cluster crash investigation transformed from hardware-suspicion to controlled workload-shape mapping. Multiple structural bugs found in `--config-file` mode and patched. Crash root cause narrowed to dispatch pressure under sustained multi-GPU load. Decisive cooldown experiment in progress at session end.

---

## Headline outcomes

1. **Hardware is not the root cause.** Same rigs are stable while mining, stable at `worker_pool_size=6`, and fully recover after reboot. Crashes are load-shape and time-under-load dependent.
2. **Two structural bugs in `--config-file` mode** identified, patched, and verified (PARITY-1 by Team Alpha, PARITY-2 by Team Beta).
3. **Two production patches** added during the crash investigation (buffer fix, live GPU probe).
4. **Stability boundary mapped:** `pool8` crashes at ~10 min, `pool6` runs 5 trials clean. Per-GPU throughput is anomalously similar between pool6 and pool8 — indicates a saturating shared resource.
5. **Decisive experiment underway:** `pool8 + cooldown500ms` testing whether dispatch pressure (not compute pressure) is the root cause.

---

## Final state at session end

| Component | State |
|:---|:---|
| Branch | s167-clean |
| All 3 AMD rigs | DPM peak-lock systemd service active from boot ✅ |
| amdgpu-dkms 6.12.12 | Pinned via apt-mark hold ✅ |
| Live `optimal_window_config.json` | Reconstructed S162 victory config staged |
| Patches deployed | S170-PARITY-1, S170-PARITY-2, S170-LIVE-GPU-PROBE, 512→2048 buffer fix, S170-WORKER-COOLDOWN |
| Currently running | `pool8 + cooldown500ms` decisive experiment |
| Netconsole | Clean across all post-patch runs (zero GCVM/SMU/qcm/device-lost faults during stable runs) |

---

## Phase 1 — Eliminate obvious software bugs

### Patch A — 512 → 2048 buffer fix (`prng_registry.py`)

**Issue:** Unsafe local array sizing / stride issues created illegal memory access risk in PRNG kernel buffers.

**Fix:** Increased buffer size from 512 to 2048 across affected kernel paths.

**Result:** System still crashes → **NOT root cause**, but eliminated a known correctness risk.

### Patch B — S170-LIVE-GPU-PROBE

**Issue:** Coordinator assumed static GPU count from config, would assign jobs to GPUs that weren't actually visible after a crash had degraded the rig.

**Fix:**
- Coordinator dynamically detects real GPU count at startup
- Workers validate `Device(0)` before signaling READY
- Prevents `GPU_COUNT_MISMATCH` between coordinator dispatch table and worker reality

**Result:** Eliminated GPU_COUNT_MISMATCH issue. System still crashes → **NOT root cause**, but eliminated a class of misdispatch errors.

---

## Phase 2 — Validate workload vs topology

### Test 1 — Remove rrig6600b (2-rig configuration)

**Hypothesis:** Cross-rig contention causing crashes.

**Result:** Still crashes.

**Conclusion:** Not a cross-rig contention issue. Ruled out network / inter-rig coordinator pressure.

### Test 2 — `worker_pool_size = 6`

**Result:**
- ✅ 5/5 trials completed
- ✅ No `hipError`
- ✅ No GCVM faults
- ✅ Netconsole clean

**Critical insight:** Reducing active GPUs per rig from 8 to 6 stabilizes the system completely.

### Anomaly discovered — throughput scaling

```
pool8 throughput ≈ pool6 throughput  (per-rig seeds/sec roughly equal)
```

This is **not normal scaling**. With 33% more active GPUs (8 vs 6), per-rig throughput should rise proportionally. Instead it's flat. This is the signature of a **saturating shared resource** — adding the 7th and 8th GPUs adds dispatch contention without adding effective throughput.

Interpretation: per-GPU throughput drops when more GPUs are active. The bottleneck is upstream of the kernel work.

---

## Phase 3 — Test dispatch pressure hypothesis

### Patch C — S170-WORKER-COOLDOWN

**New env var:** `PRNG_PWC_PER_WORKER_COOLDOWN_MS`

**Behavior:** Inserts a configurable cooldown sleep on each worker after every chunk completion, before requesting the next job.

**Purpose:** Smooth out dispatch cadence to test whether the saturating resource is dispatch-pressure / kernel-launch-rate / ROCm-context-churn.

### Test 3 — `pool8 + cooldown 50ms`

**Result:** ❌ Still crashes at ~10 minutes.

**Conclusion:** 50ms smoothing is insufficient. Either the bottleneck has a much higher pressure threshold, or 50ms doesn't materially change the dispatch shape.

### Test 4 (CURRENT) — `pool8 + cooldown 500ms`

**Status:** Running at session end.

**Purpose:** Heavy dispatch throttling. Tests whether sustained slow cadence prevents ROCm / KFD / SMU subsystem destabilization at full pool size.

**This is the decisive experiment** of the session. Outcome determines next investigation branch.

---

## What we now KNOW (proven, not hypothesis)

1. **Hardware is not fundamentally broken**
   - Same rigs run stable while mining
   - Same rigs run stable at pool6
   - Same rigs fully recover after reboot — no permanent fault

2. **Crashes are load-shape dependent**
   - pool6 → stable across 5 trials
   - pool8 → crash within ~10 minutes

3. **Crashes are time-under-load dependent**
   - ~10 minute runtime is the consistent failure threshold
   - Not an instantaneous fault — accumulates

4. **Failure pattern is consistent**
   - SMU errors → GPU reset → `device lost from bus` → probe failure → coordinator marks GPU dead
   - Same signature across rigs and runs

5. **Reboot fully clears the issue**
   - Not a permanent hardware fault
   - Runtime / system state issue (driver buffers, KFD queues, SMU state, or similar)

6. **`pool8` and `pool6` produce similar aggregate throughput**
   - Per-GPU throughput drops when more GPUs are active
   - Smoking gun for a saturating shared resource

---

## Best current root-cause model

```
TFM (functional mimicry) workload creates sustained multi-GPU pressure
  → ROCm / KFD / SMU / PCIe subsystem destabilizes
  → GPU drops off bus
```

**Key difference vs mining workloads:**

| Mining | TFM (our workload) |
|:---|:---|
| Static kernel | Rapid chunk dispatch |
| Steady load | Frequent kernel launches |
| Single ROCm context | Multi-process ROCm contexts |
| Stable memory pattern | Variable memory patterns |

**Most important insight:** This is **NOT "too much compute."** This is **too much dispatch + concurrency + churn.** The GPUs themselves can sustain the work; the driver / scheduling subsystem cannot sustain the launch rate.

---

## Earlier S170 runs (prior to crash investigation)

### Run 1 — yesterday-style 100k cap, S168/S169 jitter
- rrig6600c crashed at 08:53:56 with GCVM_L2_PROTECTION_FAULT cascade
- rrig6600 GPU0 crashed at 09:08:33 with SMU response 0xFFFFFFFF → device lost from bus
- **Result:** FAIL (matched yesterday's failure pattern). Now understood as the same dispatch-pressure crash signature.

### Run 2 — post-reboot, DPM peak-lock active, 100k/2-trial Optuna
- Watcher PID 2627, log `stability_cap_100000_t2_dpm_peak_jitter_1734.log`
- All 26 workers spawned cleanly
- 53 minutes processing, completed all 10,737 chunks of trial 1
- Optuna trial 0 sampled too-permissive thresholds → 292,671,006 survivors
- Step 1 SIGKILL'd (code -9) at 18:28:59 due to result-write blowup
- Watcher escalated: missing `optimal_window_config.json`
- **Netconsole clean throughout** — survived because the survivor explosion throttled GPU dispatch
- **Result:** Hardware PASS, pipeline FAIL/ESCALATED on artifact validation

### Process lifecycle gap discovered during cleanup
- After Run 2's coordinator was SIGKILL'd at 18:28:59, **26 worker processes survived for ~2 hours**
  - rrig6600: 8 orphans (PIDs 2054, 2151, 2251, 2364, 2483, 2590, 2697, 2804)
  - rrig6600b: 8 orphans
  - rrig6600c: 8 orphans
  - Zeus: 2 orphans (PIDs 2844, 2845)
- Workers held GPU state and TCP listeners on port 5600
- The 60-retry × 5-second cycle = ~5 minute timeout before workers self-exit (per their own logs: `coordinator gone after session (60 refused) — exiting cleanly`)
- The "stale workers killed (pwc + sieve_gpu)" defensive code at next coordinator startup is incidental cleanup, not a designed lifecycle policy
- **Recommendation:** worker lifecycle should be hardened so a coordinator SIGKILL leads to worker exit within 30 seconds. Future TB workstream.

### Run 3 — first victory config attempt (failed structurally)
- Launched at 20:13 with reconstructed S162 victory config + `--config-file` mode
- Workers never spawned on rigs
- Errors: `Authentication timeout`, `No existing session` (legacy SSH transport)
- **Diagnosis:** `--config-file` mode silently downgraded to legacy SSH path despite `--pwc-transport tcp --use-persistent-workers` on CLI. Killed.

### Run 4 — post S170-PARITY-1 patch
- All 26 workers spawned via TCP-PWC ✅
- BUT: chunks were 2,000,000 seeds each, not the 100,000 we passed via `--seed-cap-amd`
- 5 second runtime, 0 survivors
- **Diagnosis:** `--config-file` mode was also missing seed_cap, worker_pool_size, min_workers attributes. Led to PARITY-2.

### Final config-mode validation (post both PARITY patches)
- 100,000-seed chunks confirmed
- Full 1.07B seed scan completed
- **0 survivors against the reconstructed S162 victory config** — likely data drift in `daily3.json` since April 5

---

## Structural bugs in `--config-file` mode (PARITY-1, PARITY-2)

### Bug 1 — Transport flag silent downgrade (S170-PARITY-1, by Team Alpha)

`run_with_config()` did not propagate `use_persistent_workers` or `pwc_transport` onto the coordinator instance. Integration code's `getattr(coordinator, 'use_persistent_workers', False)` defaulted to False → silent fall-through to legacy SSH job distribution, even with `--pwc-transport tcp` on CLI.

**Fix:** 2-line addition mirroring Bayesian path lines 614-616.

### Bug 2 — Execution parameter silent override (S170-PARITY-2, by Team Beta)

`run_with_config()` also did not propagate `seed_cap_amd`, `seed_cap_nvidia`, `worker_pool_size`, or `pwc_min_workers`. Integration code's `getattr(coordinator, 'seed_cap_amd', 2_000_000)` defaulted to 2M chunks regardless of CLI value.

**Fix:** 4 attribute assignments mirroring Bayesian path lines 617-625.

### Implications beyond S170

Per TB: **any prior use of `--config-file` mode for distributed runs may have been silently using the legacy SSH path with default chunking.** Past results from `--config-file` mode runs should be reviewed with this knowledge.

---

## Reconstructed S162 victory config

The original S162 victory `optimal_window_config.json` could not be located:
- Live root file: deleted by `rm -f` lineage
- Git history: never committed
- NPZ embedded metadata: zero-shell (separate bug)
- `bidirectional_survivors.json`: empty (post-S162 overwrite)

Reconstructed from `TODO_MASTER_S163.md` documented values:

```json
{
  "window_size": 6,
  "offset": 64,
  "sessions": ["evening"],
  "skip_min": 3,
  "skip_max": 37,
  "prng_type": "java_lcg",
  "forward_threshold": 0.68,
  "reverse_threshold": 0.70,
  "test_both_modes": false
}
```

TB approved as best available source of truth. Final config-mode run produced 0 survivors vs. S162's documented 887 — likely data drift in `daily3.json`.

---

## NPZ data integrity bug (separate workstream)

`s170_npz_audit.py` (deployed but not wired to production) confirmed:
- April 18 `bidirectional_survivors_binary.npz` had 20,916 rows but every metadata field zero-filled
- `meta.json` claimed 10,766 survivors (mismatch with NPZ row count)
- Final run's empty-survivor NPZ has only `seeds` array — schema fields missing

Diagnosis: `convert_survivors_to_binary.py` writes a degenerate NPZ when `survivors_list == []` (only the `seeds` array). For non-empty inputs, metadata fields can default to 0 if upstream JSON didn't populate them.

**TB ruling:** Document, do not fix in S170. Separate proposal.

---

## Cluster operations

### DPM peak-lock systemd service deployed
- Files: `/usr/local/sbin/s170_dpm_peak_lock.sh` + `/etc/systemd/system/s170-dpm-peak-lock.service`
- All 3 AMD rigs, persistent across reboots
- Sets `manual` perf level + DPM peak sclk index + COMPUTE profile at boot
- Verified active after every reboot

### Full VBIOS archive captured
- All 24 GPU ROMs in `~/vbios_backups/` on each rig
- rrig6600c card1 confirmed L03 outlier (1 of 8), 0.052% diff from L04, all metadata-only
- TB ruled L03→L04 flash deferred — STRICT_DEVMEM kernel block requires `iomem=relaxed`, vetoed

---

## Decision tree pending cooldown500ms outcome

### If cooldown500 PASSES (no crash)
- **Root cause confirmed:** unbounded dispatch pressure
- **Fix path:** design proper backpressure system (token bucket / adaptive scheduler)
- Cooldown is a knob, not a fix — production needs adaptive throttling that responds to GPU saturation signals

### If cooldown500 FAILS (still crashes)
- Root cause shifts to one of:
  - Memory footprint (2048 buffers may still be insufficient)
  - ROCm kernel resource limits (queue depth, KFD context count)
  - PCIe Gen1 interconnect saturation under sustained multi-GPU load
- Next experiments:
  - `pool7` — find exact stability boundary
  - Larger chunk sizes (fewer dispatches, more compute per dispatch)
  - Kernel memory pressure profiling

---

## Files produced this session

| File | Status |
|:---|:---|
| `optimal_window_config.s162_victory.json` | Reconstructed config, staged on Zeus |
| `s170_npz_audit.py` | Manual post-run integrity check, deployed but not wired |
| `s170_optuna_replay/patch_optuna_replay.py` | Built but **HELD** by TB pending priority |
| `s170_optuna_replay/replay_inspect.py` | Built but **HELD** by TB |
| `s170_config_mode_parity/patch_config_mode_parity.py` | **DEPLOYED** as PARITY-1 |
| `s170_parity_2/patch_config_mode_parity_2.py` | **REDUNDANT** — Team Beta deployed equivalent independently |
| `prng_registry.py` (512→2048 buffer fix) | **DEPLOYED** in production |
| Live GPU probe patch (S170-LIVE-GPU-PROBE) | **DEPLOYED** in production |
| Worker cooldown env var (S170-WORKER-COOLDOWN) | **DEPLOYED** in production |
| `SESSION_CHANGELOG_20260425_S170.md` | This file |

---

## Workflow lessons logged

1. **Read TB's tense.** Past tense ("what we actually fixed") = patch already deployed. Don't rebuild.
2. **Verify log timestamps before drawing conclusions.** Worker logs from earlier in the session are NOT current state.
3. **`--config-file` mode is a known-narrow path.** Future use should explicitly verify chunk size + transport in the log within first 30 seconds of launch before assuming intent matches execution.
4. **Orphan worker check is mandatory pre-launch.** Add to standard pre-launch sanity block on all rigs + Zeus.
5. **The "hardware is broken" hypothesis is exhausted.** Future investigations should default to "code-path / dispatch / load-shape" framing per TB's repeated correction this session.
6. **Workflow rule reaffirmed:** Claude (Team Alpha) NEVER commits to git from sandbox. Files only. Michael handles dual-push.

---

## Next steps (post-S170)

### Immediately (depends on cooldown500ms outcome)
- **PASS:** Design backpressure system proposal (TB review required)
- **FAIL:** Run pool7 boundary test, then chunk-size scaling test

### TB Priority Queue
1. **Backpressure / adaptive dispatch scheduler** — pending cooldown500 outcome
2. **Survivor-rate kill gate / cap** — coordinator-side change, prevent 292M survivor blowup. Separate proposal.
3. **Optuna trial replay logger** — held during S170, may now be appropriate. Patch built and AST-verified at `/mnt/user-data/outputs/s170_optuna_replay/`.
4. **Runtime pressure instrumentation** — chunks/s, survivors/s, result bytes/s, dispatch queue depth, per-worker inflight, GPU busy.
5. **Worker lifecycle hardening** — investigate the 60-retry timeout, consider process group / setsid wrapper.
6. **NPZ data integrity fix** — converter writes degenerate NPZ on empty input; metadata fields can be zero-filled. Separate proposal.
7. **Data drift investigation** — is the S162 victory config still reproducible? Need git-archive of `daily3.json` as-of April 5 for direct comparison.

### Held / not currently active
- BoTorch dual-GPU (deferred per project memory)
- DPM harness 900mV/2100-2200MHz (deferred — DPM lock service deployed at default peak)

---

## Bottom line

We are no longer debugging randomly. We are now precisely mapping the **stability boundary vs workload shape**:

```
pool6: stable
pool8 + cooldown50ms: crash at ~10 min
pool8 + cooldown500ms: ← decisive experiment in progress
```

This is the right level of resolution to design a real fix.

---

## Git operations (for Michael to execute on Zeus)

Per workflow rule, Claude does NOT commit. Michael should:
1. SCP this file to Zeus: `~/distributed_prng_analysis/docs/SESSION_CHANGELOG_20260425_S170.md`
2. Stage all S170 deliverables under `docs/`
3. Commit with message: `docs(s170): session changelog — config-mode parity, GPU probe, worker cooldown, dispatch pressure investigation`
4. Dual-push: `git push origin main && git push public main`

---

**END OF SESSION S170**
