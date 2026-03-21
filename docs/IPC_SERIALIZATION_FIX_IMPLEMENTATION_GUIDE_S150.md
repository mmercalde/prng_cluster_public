# IPC Serialization Fix — Implementation Guide
**Session:** S149 → S150  
**Author:** Team Alpha  
**Date:** 2026-03-21  
**TB Ruling:** Approved Option A / slim_v1  
**Status:** Design complete, implementation pending S150

---

## 1. The Problem — Plain English

When the GPU finds survivors in a chunk, the worker must send those survivors
back to Zeus so they can be collected. Right now it sends them like this:

```
Worker → Zeus (over SSH pipe):
{"status":"ok","result":{"survivors":[
  {"seed":123456789,"family":"java_lcg","match_rate":0.72,"matches":9,"total":12,"best_skip":5},
  {"seed":987654321,"family":"java_lcg","match_rate":0.68,"matches":8,"total":12,"best_skip":3},
  ... (1400 more entries) ...
]}}
```

Each survivor is a Python dict with 6-7 fields. Sending 1,400 survivors means
Zeus is receiving and parsing a JSON string that looks like:

```
~1,400 survivors × ~150 bytes each = ~210,000 bytes (210KB) per chunk result
```

At 537 chunks per pass, a high-survivor pass can generate:
```
537 chunks × 210KB = ~113MB of JSON per pass
```

The GPU computes its chunk in ~50 milliseconds. But Python's JSON serializer
takes several seconds to turn 1,400 dicts into a string, write it to the pipe,
have Zeus read it, and parse it back into Python objects. During those seconds,
the GPU sits completely idle waiting for the next job.

**This is why the dashboard shows:**
- Pass 1 (few survivors): ~2,000,000 seeds/sec — GPU never waits
- Pass 3/4 (many survivors): ~73,000 seeds/sec — GPU mostly waits

**The 27x slowdown is entirely caused by sending too much data per result.**

---

## 2. What We Are NOT Changing

Before describing the fix, it is critical to understand what stays the same:

1. **The outer IPC protocol** — One JSON line in (job), one JSON line out
   (result). This does not change. Zeus still writes one line to worker stdin,
   reads one line from worker stdout.

2. **The coordinator's output** — `_dispatch_to_worker()` currently returns:
   ```python
   {
     "status": "ok",
     "job_id": "...",
     "survivors": [123456789, 987654321, ...],      # list of ints
     "match_rates": [0.72, 0.68, ...],              # list of floats
     "skip_sequences": [[5,3,7,...], [2,8,1,...]], # list of lists
     "strategy_ids": [0, 1, 0, ...]                # list of ints
   }
   ```
   This does not change. Everything downstream of the coordinator
   (window_optimizer_integration_final.py, NPZ accumulator, Steps 2-6)
   remains completely untouched.

3. **The GPU kernels** — sieve_gpu_worker.py still computes the same arrays
   on GPU. The kernel code does not change.

4. **The NPZ schema** — bidirectional_survivors_all.npz format unchanged.

5. **Steps 2-6** — No changes anywhere in the pipeline beyond the coordinator.

---

## 3. What Changes — Exactly Two Functions

### Change 1: sieve_gpu_worker.py — `run_sieve_job()` output format

**Current behavior:**
The worker builds a list of dicts (one dict per survivor) and serializes the
whole thing as a JSON array inside the result:

```python
# Current — slow
survivors_out.append({
    'seed': int(seed),
    'family': family_name,
    'match_rate': float(rate),
    'matches': int(rate * k),
    'total': k,
    'strategy_id': int(sid),
    'skip_sequence': ss,
})
# ...
return {
    'job_id': job_id,
    'success': True,
    'survivors': all_survivors,  # list of dicts — expensive!
    ...
}
```

**New behavior (slim_v1):**
Instead of a list of dicts, return parallel arrays — one array per field:

```python
# New — fast
seeds_out.append(int(seed))
match_rates_out.append(float(rate))
strategy_ids_out.append(int(sid))
skip_sequences_out.append(ss)
# ...
return {
    'job_id': job_id,
    'success': True,
    'result': {
        'format': 'slim_v1',
        'seeds': seeds_out,               # [123456789, 987654321, ...]
        'match_rates': match_rates_out,   # [0.72, 0.68, ...]
        'strategy_ids': strategy_ids_out, # [0, 1, 0, ...]  (hybrid only)
        'skip_sequences': skip_seqs_out,  # [[5,3],[2,8],...] (hybrid only)
    },
    ...
}
```

**Why this is faster:**
- Old: 1,400 dicts × 7 keys × string field names = 1,400 × ~150 chars
- New: 4 flat arrays × 1,400 numbers = 1,400 × ~10 chars
- Reduction: ~15x smaller payload
- JSON serialization of flat arrays is ~10x faster than nested dicts

**Backward compatibility in the worker:**
No legacy path needed in the worker — the coordinator handles both formats.

---

### Change 2: persistent_worker_coordinator.py — `_dispatch_to_worker()` parsing

**Current behavior:**
The coordinator parses the list-of-dicts format:

```python
raw_survivors = inner.get("survivors", [])
survivors   = [s["seed"]       if isinstance(s, dict) else int(s) for s in raw_survivors]
match_rates = [s["match_rate"] if isinstance(s, dict) else 0.5     for s in raw_survivors]
skip_seqs   = [s.get("skip_sequence", []) if isinstance(s, dict) else [] for s in raw_survivors]
strat_ids   = [s.get("strategy_id",    0) if isinstance(s, dict) else 0  for s in raw_survivors]
```

**New behavior — accept both formats:**

```python
inner = result.get("result", {})

if inner.get("format") == "slim_v1":
    # New fast path — parallel arrays
    survivors   = [int(s) for s in inner.get("seeds", [])]
    match_rates = inner.get("match_rates", [])
    skip_seqs   = inner.get("skip_sequences", [[] for _ in survivors])
    strat_ids   = inner.get("strategy_ids",   [0  for _ in survivors])
else:
    # Legacy path — list of dicts (old worker format)
    raw_survivors = inner.get("survivors", [])
    survivors   = [s["seed"]       if isinstance(s, dict) else int(s) for s in raw_survivors]
    match_rates = [s["match_rate"] if isinstance(s, dict) else 0.5     for s in raw_survivors]
    skip_seqs   = [s.get("skip_sequence", []) if isinstance(s, dict) else [] for s in raw_survivors]
    strat_ids   = [s.get("strategy_id",    0) if isinstance(s, dict) else 0  for s in raw_survivors]
```

**Why two paths:**
During rollout, if something goes wrong with the new worker format, you can
revert just the worker to the old format. The coordinator will automatically
fall back to the legacy path. No other code needs to change.

Once slim_v1 is proven stable (one full production run), the legacy path
can be removed in a later session.

---

## 4. Implementation Steps — S150

The patch must be applied in this exact order to avoid breaking the running system:

### Step 1: Apply coordinator patch first
Update `_dispatch_to_worker()` in `persistent_worker_coordinator.py` to accept
both formats. **Deploy this first.** Old workers still work (legacy path).
New workers will also work (slim_v1 path).

```
Commit: fix(s150): coordinator accepts slim_v1 and legacy worker result formats
```

### Step 2: Verify coordinator patch against live run
Run one full trial with the coordinator patch but OLD worker. Confirm:
- All 4 passes complete
- Survivors accumulate correctly in NPZ
- No coordinator errors

### Step 3: Apply worker patch
Update `run_sieve_job()` in `sieve_gpu_worker.py` to emit slim_v1 format.

```
Commit: fix(s150): worker emits slim_v1 parallel arrays — reduce IPC payload 15x
```

### Step 4: Deploy to all rigs simultaneously
The coordinator is on Zeus. The worker is on all 3 rigs. After committing:

```bash
# Update all rigs
ssh rrig6600 "cd ~/distributed_prng_analysis && git pull"
ssh rrig6600b "cd ~/distributed_prng_analysis && git pull"
ssh rrig6600c "cd ~/distributed_prng_analysis && git pull"
```

Workers are respawned per trial — the new worker code takes effect on the
next trial's worker spawn. No manual restart needed.

### Step 5: Measure Pass 3/4 throughput
After the first high-survivor trial completes, compare dashboard throughput:
- Before fix: ~73,000 s/s on Pass 3/4
- Expected after fix: >500,000 s/s on Pass 3/4
- If still below 200,000 s/s → escalate to TB for Option B (binary)

---

## 5. What Could Go Wrong and How to Recover

### Risk 1: slim_v1 coordinator patch breaks legacy workers
**Symptom:** `[SAVE] Trial N: 0 survivors` on all trials after coordinator patch
**Cause:** Legacy path fallback not working
**Recovery:** `git revert` the coordinator patch, run preprod to confirm recovery

### Risk 2: Worker emits slim_v1 but coordinator doesn't parse it
**Symptom:** All survivors disappear after worker patch
**Cause:** Worker patch deployed before coordinator patch
**Recovery:** Always deploy coordinator patch FIRST — this is why Step 1 comes before Step 3

### Risk 3: skip_sequences shape mismatch on hybrid passes
**Symptom:** NPZ accumulator has wrong skip_mode distribution
**Cause:** skip_sequences array not padded correctly for variable-length sequences
**Recovery:** Check `len(skip_seqs[i]) == k` for all i in hybrid pass results

### Risk 4: strategy_ids missing on constant-skip passes
**Symptom:** All strategy_ids are 0 on constant passes
**Note:** This is correct — constant-skip passes don't use strategy_ids.
The coordinator already handles this (defaults to 0).

---

## 6. Files Changed Summary

| File | Change | Lines affected |
|---|---|---|
| `sieve_gpu_worker.py` | Output format: list-of-dicts → slim_v1 parallel arrays | ~30 lines in `run_sieve_job()` |
| `persistent_worker_coordinator.py` | Parse: accept slim_v1 + legacy fallback | ~15 lines in `_dispatch_to_worker()` |

**Zero changes to:**
- `window_optimizer_integration_final.py`
- `window_optimizer_bayesian.py`
- `window_optimizer.py`
- NPZ accumulator
- `convert_survivors_to_binary.py`
- `survivor_loader.py`
- All Steps 2-6
- All ML models
- WATCHER

---

## 7. Expected Outcome

| Metric | Before | After |
|---|---|---|
| Pass 1/2 throughput | ~2,000,000 s/s | ~2,000,000 s/s (unchanged) |
| Pass 3/4 throughput | ~73,000 s/s | >500,000 s/s (estimated) |
| Per-trial time (50 trials) | ~2.5 hrs/trial avg | ~30 min/trial avg |
| Full Run 1 (50 trials) | ~5 days | ~25 hours |
| Full 4-run sweep | ~20 days | ~4 days |
