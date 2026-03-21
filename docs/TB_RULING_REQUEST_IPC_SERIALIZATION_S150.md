# TB Ruling Request — IPC Result Serialization Optimization
**Session:** S149 (for implementation in S150)  
**Author:** Team Alpha  
**Date:** 2026-03-21  
**Priority:** P1 — throughput bottleneck on high-survivor trials

---

## Background

S149-B fixed the AMD 4-worker ceiling. Post-fix cluster throughput is:

| Condition | Throughput | Notes |
|---|---|---|
| Low-survivor trial (Pass 1) | ~1,990,000 s/s | GPU-bound, fast serialization |
| High-survivor trial (Pass 3/4) | ~73,000 s/s | Serialization-bound |
| Ratio | **27x degradation** | Purely from result payload size |

The GPU hardware is not the bottleneck — Pass 1 proves the cluster can sustain
~2M s/s. The degradation on high-survivor passes is caused by result payload
serialization overhead in the worker→coordinator IPC path.

---

## Root Cause

Each worker currently returns full survivor records over stdout as JSON:

```json
[
  {
    "seed": 123456789,
    "forward_match_rate": 0.72,
    "reverse_match_rate": 0.68,
    "score": 0.70,
    "window_size": 12,
    "offset": 5,
    "skip_min": 10,
    "skip_max": 53,
    "trial_number": 7,
    "prng_type": "java_lcg",
    "prng_base": "java_lcg",
    "skip_mode": "constant",
    "forward_count": 470904,
    "reverse_count": 469893,
    "bidirectional_count": 2437,
    "forward_only_count": 468467,
    "reverse_only_count": 467456,
    "intersection_count": 2437,
    "intersection_ratio": 0.0052,
    "survivor_overlap_ratio": 0.0052,
    "intersection_weight": 0.0026,
    "sessions": "evening",
    "skip_range": "10-53"
  },
  ...
]
```

A chunk returning 1,400 survivors sends ~1,400 × ~500 bytes = **700KB of JSON**
over a loopback SSH pipe per chunk result. At 537 chunks per pass, that's
potentially **375MB of JSON** serialized, transmitted, and parsed per pass.

The GPU computes its chunk in ~50ms. The JSON serialization + SSH write +
Zeus read + JSON parse takes several seconds per high-survivor chunk. The GPU
sits idle during this entire period.

---

## Proposed Options

### Option A — Slim payload: return seeds + match rates only (recommended)

Workers return only the numerically essential fields:

```json
{
  "status": "ok",
  "seeds": [123456789, 987654321, ...],
  "forward_match_rates": [0.72, 0.81, ...],
  "reverse_match_rates": [0.68, 0.74, ...]
}
```

Zeus reconstructs the full survivor record from these 3 arrays plus the
known job parameters (window_size, offset, etc. from the job dict).

**Payload reduction:** ~500 bytes/survivor → ~24 bytes/survivor (~20x)  
**Risk:** Coordinator must reconstruct metadata fields from job context  
**Files:** `sieve_gpu_worker.py` (output format), `persistent_worker_coordinator.py` (result parsing)  
**Backward compatibility:** `convert_survivors_to_binary.py` and NPZ schema unchanged — reconstruction happens before accumulator write

### Option B — Binary IPC: numpy over pipe

Workers write results as compressed numpy arrays to stdout binary stream.
Zeus reads with `np.load()` from a BytesIO wrapper.

```python
# Worker side
import numpy as np, io, sys
buf = io.BytesIO()
np.savez_compressed(buf, seeds=seed_array, fmr=fmr_array, rmr=rmr_array)
sys.stdout.buffer.write(buf.getvalue())
```

**Payload reduction:** ~20x vs JSON (numpy binary is much denser)  
**Risk:** Binary pipe requires careful framing (length prefix or delimiter) to avoid read/parse errors; more complex than Option A  
**Files:** `sieve_gpu_worker.py`, `persistent_worker_coordinator.py`

### Option C — Shared memory segment

Worker writes results to a POSIX shared memory segment (`/dev/shm`), sends
Zeus only the shm name and array shape. Zero copy, maximum throughput.

**Payload reduction:** ~1000x (only metadata over pipe)  
**Risk:** Requires shared memory setup/teardown per chunk; complex cleanup on worker crash; SSH tunnel complicates shm access for remote rigs (shm is local to the rig, not accessible from Zeus directly) — **this option may not be viable for remote rigs**  
**Files:** `sieve_gpu_worker.py`, `persistent_worker_coordinator.py`

---

## Questions for Team Beta

**Q1 — Option A viability**  
Is the slim payload approach (seeds + match rates only, reconstruct metadata
from job context) architecturally sound? The job dict already contains
`window_size`, `offset`, `skip_min`, `skip_max`, `session`, `trial_number`,
`prng_type`, `skip_mode`. Are there any survivor metadata fields that cannot
be reconstructed from the job dict + the 3 returned arrays?

**Q2 — Option B framing**  
If binary IPC is preferred, what is the recommended framing strategy for
numpy binary over unbuffered pipes? Options:
- Length-prefix header (4-byte int before each payload)
- Fixed-size sentinel delimiter
- One numpy file per result line (newline-terminated base64)

**Q3 — Option C viability for remote rigs**  
Is shared memory viable for the AMD rigs given that Zeus dispatches jobs
over SSH? The worker runs on rrig6600 (192.168.3.120) and Zeus reads results
over the SSH stdout pipe — there is no direct shm access from Zeus to the
rig's `/dev/shm`. Does this rule out Option C entirely for remote workers?

**Q4 — Scope boundary**  
Should this change touch only the IPC layer (`sieve_gpu_worker.py` output
format + `persistent_worker_coordinator.py` result parsing), or does the
NPZ accumulator write in `window_optimizer_integration_final.py` also need
to change to accept the new slim format?

**Q5 — Match rate precision**  
The current JSON includes `forward_match_rate` and `reverse_match_rate` as
float64. For the slim payload, is float32 precision sufficient for downstream
ML scoring in Steps 2-6, or must these fields remain float64?

---

## Current Measured Baseline (post S149-B)

| Pass | Survivor density | Throughput | Bottleneck |
|---|---|---|---|
| Forward constant (Pass 1) | Low (0-50/chunk) | ~2,000,000 s/s | GPU-bound |
| Reverse constant (Pass 2) | Low | ~2,000,000 s/s | GPU-bound |
| Forward hybrid (Pass 3) | High (500-2000/chunk) | ~73,000 s/s | Serialization |
| Reverse hybrid (Pass 4) | High | ~73,000 s/s | Serialization |

Target after fix: all 4 passes sustained >500,000 s/s.

---

## Files in Scope

| File | Expected change |
|---|---|
| `sieve_gpu_worker.py` | Output format — slim payload or binary |
| `persistent_worker_coordinator.py` | Result parsing — reconstruct from slim payload |
| `window_optimizer_integration_final.py` | Possibly none if reconstruction happens in coordinator |

**Out of scope:** NPZ schema, Steps 2-6, ML models, Optuna, WATCHER,
bidirectional sieve logic, kernel code.
