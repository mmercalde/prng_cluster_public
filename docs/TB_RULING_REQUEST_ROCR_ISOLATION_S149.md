# TB Ruling Request — ROCR_VISIBLE_DEVICES Worker Isolation Fix
**Session:** S148 (discovered during Run 1 monitoring)  
**Author:** Team Alpha  
**Date:** 2026-03-19  
**Priority:** P1 — blocks AMD rig throughput for all future production sweeps  

---

## Background

During Run 1 production sweep monitoring (S148), AMD rig throughput was
observed at ~1,050 seeds/sec per rig vs the S130 soak test baseline of
~787,000 seeds/sec per GPU. `rocm-smi --showuse` confirmed **0% GPU
utilization** on all 8 GPUs while 4 persistent workers were alive and
connected. CPU was 99% idle. Workers had only accumulated ~39 seconds of
CPU time after 49 minutes of runtime — **1.3% utilization**.

The root cause was traced to an IPC starvation pattern (coordinator
dispatch latency) combined with a hard cap of `worker_pool_size=4` per
rig. The pool size cap was established in S146 because any value above 4
caused rig crashes. This ruling request addresses the crash root cause.

---

## Investigation

A source-analysis harness (`test_s149_rocr_isolation_harness.py`) was
built against the live public repo and run to verify the theory. All
15/15 checks passed.

### What the harness found

**1. `ROCR_VISIBLE_DEVICES` is absent from the worker spawn environment.**

`_spawn_worker()` in `persistent_worker_coordinator.py` constructs the
worker environment as:

```python
rocm_env = " ".join(ROCM_ENV_VARS + [
    f"CUDA_VISIBLE_DEVICES={gpu_id}",
    f"HIP_VISIBLE_DEVICES={gpu_id}",
])
```

`ROCM_ENV_VARS` contains `HSA_OVERRIDE_GFX_VERSION`, `HSA_ENABLE_SDMA`,
`ROCM_PATH`, `HIP_PATH`, `LD_LIBRARY_PATH`, `PATH`, `CUPY_CACHE_DIR`.
`ROCR_VISIBLE_DEVICES` is not present in either location.

**2. `sieve_gpu_worker.py` hardcodes `cp.cuda.Device(0)` everywhere.**

Three occurrences confirmed at lines 161, 359, 361:

```python
device = cp.cuda.Device(0)          # run_sieve_job
with cp.cuda.Device(0):             # startup warmup
    cp.cuda.Device(0).synchronize()
```

The code comment at line 143 states the design intent explicitly:
> *"Always uses device 0 (ROCR_VISIBLE_DEVICES has isolated the GPU)"*

The design is correct — the worker is supposed to always use `Device(0)`
because `ROCR_VISIBLE_DEVICES` remaps device 0 to the assigned physical
GPU at the HSA runtime level. The problem is the spawner never sets
`ROCR_VISIBLE_DEVICES`.

**3. Crash pattern is consistent with the theory.**

`HIP_VISIBLE_DEVICES` provides partial mitigation. It works reliably for
GPUs 0–3 on the RX 6600 ROCm version in use. For GPUs 4–7,
`HIP_VISIBLE_DEVICES` remapping is inconsistent — workers assigned to
those GPUs fall back to initializing `Device(0)` on an already-active
physical GPU, producing an HSA context conflict and crashing the worker
process. This exactly explains why `worker_pool_size=4` is stable and
anything above crashes.

### The fix

One line added to `_spawn_worker()` in
`persistent_worker_coordinator.py`:

```python
# Current
rocm_env = " ".join(ROCM_ENV_VARS + [
    f"CUDA_VISIBLE_DEVICES={gpu_id}",
    f"HIP_VISIBLE_DEVICES={gpu_id}",
])

# Fixed
rocm_env = " ".join(ROCM_ENV_VARS + [
    f"CUDA_VISIBLE_DEVICES={gpu_id}",
    f"HIP_VISIBLE_DEVICES={gpu_id}",
    f"ROCR_VISIBLE_DEVICES={gpu_id}",   # ADD — HSA-level isolation
])
```

No changes to `sieve_gpu_worker.py`. The `Device(0)` hardcoding is
correct by design and should not be changed.

---

## Impact

Current state with `worker_pool_size=4` and no ROCR isolation:
- 4 workers per rig, GPUs 0–3 only, GPUs 4–7 idle
- ~1,050 seeds/sec per rig observed in Run 1
- Zeus (2× RTX 3080 Ti) contributing ~90% of total throughput

Expected state after fix with `worker_pool_size=8`:
- 8 workers per rig, all 8 GPUs active
- ~6M+ seeds/sec per rig (8 × ~787k baseline)
- 3 rigs × 6M = ~18M seeds/sec vs Zeus's ~28k seeds/sec
- Rigs become the dominant compute resource as originally designed
- Per-trial time: hours → minutes

---

## Questions for Team Beta

**Q1 — Fix correctness**  
Is adding `ROCR_VISIBLE_DEVICES={gpu_id}` to the spawn env the correct
and sufficient fix for HSA-level GPU isolation on the RX 6600 / ROCm
stack in use? Are there any known interactions with `HSA_ENABLE_SDMA=0`
or `HSA_OVERRIDE_GFX_VERSION=10.3.0` that could make this unsafe?

**Q2 — Worker pool size after fix**  
After applying the ROCR fix, is it safe to raise `worker_pool_size` from
4 to 8 (one worker per GPU per rig)? Should this be validated via the
existing soak test pattern (`sweep_preprod.sh`) before restoring
production manifest, or is a targeted `verify_rocr_workers.py` harness
on a single rig sufficient?

**Q3 — Spawn stagger**  
Current stagger is `ROCM_SPAWN_STAGGER_S = 4.0s` — 4 workers × 4s =
16s init per rig. With 8 workers × 4s = 32s. Is the 4s stagger still
necessary with proper ROCR isolation, or can it be reduced to lower
trial startup overhead?

**Q4 — IPC starvation (separate issue)**  
Even with 8 workers per rig, each worker still dispatches one job at a
time (sequential send→wait→receive). GPU utilization per worker will
remain low (~1%) due to SSH round-trip latency between jobs. The ROCR
fix gets all 8 GPUs active, multiplying throughput 8×. Is further
pipelining (pre-fetching next job while current job executes) within
scope for S149, or should it be deferred until ROCR fix is validated?

---

## Harness

`test_s149_rocr_isolation_harness.py` — source-analysis only, no GPU
hardware required, runs against live repo. 15/15 checks. Verifies:
- ROCR_VISIBLE_DEVICES absent from ROCM_ENV_VARS and _spawn_worker
- HIP_VISIBLE_DEVICES present (partial mitigation)
- Device(0) hardcoded 3× in sieve_gpu_worker.py
- Design intent comment confirms ROCR was always required
- Fix not yet applied in live codebase

Run with:
```bash
cd ~/distributed_prng_analysis
python3 test_s149_rocr_isolation_harness.py
```

---

## Proposed Implementation (pending TB approval)

1. Apply `ROCR_VISIBLE_DEVICES={gpu_id}` fix to `_spawn_worker()`
2. Run `test_s149_rocr_isolation_harness.py` — verify 15/15 still pass
3. Run single-rig smoke test: spawn 8 workers on `rrig6600`, confirm
   all 8 heartbeat without crash
4. Run `sweep_preprod.sh` (50M seeds, 5 trials) with `worker_pool_size=8`
5. Observe `rocm-smi --showuse` — expect GPU% > 0 on all 8 GPUs
6. If preprod passes: restore production manifest and relaunch

**This fix does not touch any sieve kernel, ML model, NPZ format,
bidirectional logic, or WATCHER orchestration. Blast radius is limited
to the worker spawn environment string in one function.**
