# GPU THROUGHPUT INVESTIGATION & OPTIMIZATION PLAN
**Version:** 1.0  
**Created:** 2026-03-07 (S126)  
**Author:** Team Alpha  
**Status:** PLANNED — Pending execution after 200-trial production run  

---

## Background & Motivation

Post-S125b smoke test revealed significant underutilization of cluster throughput capacity:

| GPU | Measured Seeds/Sec | Current Job Size | Job Duration | GPU Idle Time |
|-----|-------------------|-----------------|--------------|---------------|
| RTX 3080 Ti (Zeus gpu0) | ~33,627 | 19,000–40,000 | ~1.2s | HIGH — starved |
| RTX 3080 Ti (Zeus gpu1) | ~31,825 | 19,000–40,000 | ~1.2s | HIGH — starved |
| RX 6600 (AMD rigs) | ~11,500 | 19,000 | ~1.6s | MODERATE |

**Root problem:** `seed_cap_nvidia=40,000` and `seed_cap_amd=19,000` were set conservatively during ROCm instability work (S96B) and the CUDA Exclusive_Process fix (S125b). Both caps are now known to be far below hardware capacity. `gpu_optimizer.py` performance profiles are also stale — RX 6600 is listed at 5,000 seeds/sec but measured at ~11,500.

**Key AMD constraint discovered:** ROCm documentation explicitly states "Radeon GPUs do not support large amounts of simultaneous, parallel workloads — it is not recommended to exceed 2 simultaneous compute workloads." We are running **8 workers per rig**. Current stability is maintained by `PYTORCH_HIP_ALLOC_CONF` + 80% VRAM limiting from S96B. This makes Phase B (multi-worker) testing critical — single-card ceilings will differ from full-rig ceilings.

**Estimated throughput gain:** 5–6x overall cluster throughput from config changes alone. No code refactoring required.

---

## Current State (Baseline — S125b)

### Confirmed Measured Values
```
RTX 3080 Ti:  ~27,000–33,627 seeds/sec per card  (smoke test, 100k seed jobs)
RX 6600:      ~11,000–13,800 seeds/sec per card  (smoke test, 100k seed jobs)
```

### Current Config Values (Stale)
```python
# coordinator.py line 233
seed_cap_nvidia  = 40,000   # set S125b — historically conservative
seed_cap_amd     = 19,000   # set S96B — ROCm instability era
seed_cap_default = 19,000

# gpu_optimizer.py
"RTX 3080 Ti": seeds_per_second = 29,000   # close to real, acceptable
"RX 6600":     seeds_per_second =  5,000   # WRONG — measured 11,500 (2.3x understated)
"RTX 3080 Ti": scaling_factor   =    6.0   # WRONG — real ratio is ~2.9x not 6x
"RX 6600":     scaling_factor   =    1.0   # baseline (stale absolute value)
```

### Theoretical Max Throughput (With Corrected Caps)
```
2× RTX 3080 Ti:  ~33,000 × 2  =  ~66,000 seeds/sec
24× RX 6600:     ~11,500 × 24 = ~276,000 seeds/sec
─────────────────────────────────────────────────────
Total:                          ~342,000 seeds/sec

vs. current effective:          ~54,000 seeds/sec (AMD bottleneck only)
Projected improvement:          ~6.3x
```

---

## Investigation Plan

### Phase A — Single Card Isolated Ceiling (One card, no concurrent workers)

**Purpose:** Find the true hardware ceiling per card type in isolation. This is the theoretical max before multi-worker contention is factored in.

**Procedure:** Run one sieve job at each seed count step. Wait for completion. Record seeds/sec and peak VRAM. Stop at first OOM or crash.

| Step | RTX 3080 Ti Seeds | RX 6600 Seeds | Expected RTX Time | Expected AMD Time |
|------|------------------|---------------|-------------------|-------------------|
| A1 | 100,000 ✅ known safe | 100,000 ✅ known safe | ~3s | ~8.5s |
| A2 | 500,000 | 250,000 | ~15s | ~22s |
| A3 | 1,000,000 | 500,000 | ~30s | ~43s |
| A4 | 2,000,000 | 1,000,000 | ~60s | ~87s |
| A5 | 5,000,000 | 2,000,000 | ~150s | ~174s |

**Stop condition:** OOM error, kernel crash, or throughput degradation >20% from prior step (indicates memory pressure).

**Monitoring commands:**
```bash
# RTX — monitor VRAM during job
watch -n 1 "nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader"

# AMD — monitor VRAM during job
ssh rrig6600 "watch -n 1 'rocm-smi --showmeminfo vram'"
```

**Run commands:**
```bash
# Single RTX card (Zeus)
CUDA_VISIBLE_DEVICES=0 python3 sieve_filter.py --job-file /tmp/probe_A2_rtx.json --gpu-id 0

# Single RX 6600 (one rig, one card only — ROCR_VISIBLE_DEVICES isolates it)
ssh rrig6600 "source ~/rocm_env/bin/activate && \
  ROCR_VISIBLE_DEVICES=0 python3 ~/distributed_prng_analysis/sieve_filter.py \
  --job-file /tmp/probe_A2_amd.json --gpu-id 0"
```

---

### Phase B — Full Rig Concurrent Ceiling (All workers on one rig simultaneously)

**Purpose:** Find the real production ceiling when all workers compete for the same VRAM bus and ROCm runtime. This is the number that actually matters for setting `seed_cap_*`.

**Why this differs from Phase A:** ROCm's documented "do not exceed 2 simultaneous workloads" warning means the rig ceiling is NOT simply (card ceiling × 8). Bandwidth contention, VRAM fragmentation across 8 workers, and HSA runtime overhead all impose a lower limit.

**Procedure:** Launch all 8 AMD workers simultaneously (or both RTX workers) with identical job sizes. Record aggregate throughput and any errors.

| Step | RTX per card | AMD per card | Notes |
|------|-------------|--------------|-------|
| B1 | 100,000 ✅ | 100,000 ✅ | Baseline concurrent — known safe |
| B2 | 300,000 | 200,000 | First concurrent stress test |
| B3 | 750,000 | 400,000 | VRAM contention probe |
| B4 | Phase A ceiling × 0.5 | Phase A ceiling × 0.5 | Half-ceiling concurrent test |
| B5 | Phase A ceiling × 0.75 | Phase A ceiling × 0.75 | Find concurrent cliff |

**Stop condition:** Any single worker OOM, ROCm HangDetected, or HIP error.

**Key metric to capture:** Gap between Phase A ceiling and Phase B ceiling. This gap represents the multi-worker ROCm tax. Expected: Phase B ceiling = 50–70% of Phase A ceiling for AMD based on ROCm documentation.

---

### Phase C — Throughput Stability Test

**Purpose:** Confirm the chosen caps remain stable over sustained operation (not just a single job).

**Procedure:** Run 50 consecutive jobs at the selected cap (Phase B ceiling × 0.85) on all 26 GPUs simultaneously via the normal WATCHER pipeline. Monitor for any degradation, OOM, or rig dropout over time.

**This is the gate before updating production config.**

---

## Changes Required After Probing

All changes are **single-line or small constant edits** in exactly two files. No refactoring.

### File 1: `coordinator.py` — line 233

```python
# BEFORE (current stale values):
seed_cap_nvidia: int = 40000, seed_cap_amd: int = 19000, seed_cap_default: int = 19000,

# AFTER (replace with Phase B ceiling × 0.85):
seed_cap_nvidia: int = <TBD_from_Phase_B>, seed_cap_amd: int = <TBD_from_Phase_B>, seed_cap_default: int = <TBD_from_Phase_B>,
```

### File 2: `gpu_optimizer.py` — performance profiles

```python
# BEFORE (current stale values):
"RTX 3080 Ti": {
    "seeds_per_second": 29000,
    "scaling_factor": 6.0,
},
"RX 6600": {
    "seeds_per_second": 5000,
    "scaling_factor": 1.0,
},

# AFTER (replace with measured values from Phase A):
"RTX 3080 Ti": {
    "seeds_per_second": <TBD_from_Phase_A>,   # measured ~33,000
    "scaling_factor": <TBD_from_Phase_A / Phase_A_AMD>,  # expected ~2.9
},
"RX 6600": {
    "seeds_per_second": <TBD_from_Phase_A>,   # measured ~11,500
    "scaling_factor": 1.0,   # remains baseline
},
```

**Note on scaling_factor:** After updating `seeds_per_second` for both card types, recalculate `scaling_factor` as `RTX_sps / RX6600_sps`. Current 6.0x is derived from stale baselines. Expected new value: ~2.9x.

---

## Execution Sequence

```
1. ✅ Complete smoke test (S125b) — confirms all 26 GPUs operational
2. ✅ Complete 200-trial production run — grow survivor pool first
3. → Phase A probing (isolated single-card tests, both GPU types)
4. → Phase B probing (full concurrent rig tests)
5. → Phase C stability test at chosen caps
6. → Update coordinator.py + gpu_optimizer.py constants
7. → Commit both files to both repos
8. → Add new caps to Architecture Invariants in TODO_MASTER
9. → Run one full WATCHER pipeline cycle to validate end-to-end
```

**Sequence rationale:** The 200-trial production run comes FIRST because it uses the current (known-safe) caps. Throughput optimization is not needed for that run — correctness of survivors matters more than speed at that stage. Throughput work then pays off on all subsequent runs.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RX 6600 OOM at high seed counts | Medium | Low — just find the ceiling | Step-ladder probe stops safely |
| ROCm HangDetected on concurrent overload | Medium | Medium — requires rig reboot | Phase B starts conservatively at B1 |
| RTX OOM (12GB VRAM) | Low | Low | RTX has far more headroom |
| Phase B ceiling much lower than Phase A | Medium | Low — just use a lower cap | Expected and planned for |
| Probe disrupts active 200-trial run | N/A | High | Probes run AFTER 200-trial run completes |

---

## Success Criteria

- [ ] Phase A ceiling measured for both RTX 3080 Ti and RX 6600
- [ ] Phase B ceiling measured (all workers concurrent) for both GPU types
- [ ] Gap between Phase A and Phase B quantified (the "ROCm multi-worker tax")
- [ ] `coordinator.py` `seed_cap_*` updated to Phase B ceiling × 0.85
- [ ] `gpu_optimizer.py` `seeds_per_second` and `scaling_factor` updated to measured values
- [ ] Both files committed to both repos
- [ ] Full WATCHER pipeline run validates no regressions
- [ ] Measured end-to-end throughput improvement documented in session changelog

---

## Expected Outcome

| Metric | Before | After (estimated) |
|--------|--------|-------------------|
| RTX job size | 40,000 seeds | 500k–2M seeds |
| AMD job size | 19,000 seeds | 200k–500k seeds |
| RTX job duration | ~1.2s | 15–60s (stays fed) |
| AMD job duration | ~1.6s | 17–43s |
| Cluster seeds/sec | ~54,000 | ~250,000–342,000 |
| Overall speedup | baseline | **~5–6x** |

All gains from config constant changes only. Zero code refactoring.

---

## Files To Be Modified (Summary)

| File | Location | Change Type | Lines Affected |
|------|----------|-------------|----------------|
| `coordinator.py` | `~/distributed_prng_analysis/` | Constant update | Line 233 |
| `gpu_optimizer.py` | `~/distributed_prng_analysis/` | Constant update | Lines 17, 18, 35, 36 |

---

*Created S126 — 2026-03-07 | Status: PLANNED*  
*Execute after: smoke test complete + 200-trial production run complete*
