# Step 1 Quick Benchmark Report
**Generated:** 2026-02-01 15:59:08
**Cluster:** Zeus (2 GPU) + rig-6600 (8 GPU) + rig-6600b (8 GPU) + rig-6600c (8 GPU) = 26 GPUs

---


## Test 1: Seed Count — Single Sieve Each

| seed_count | Time (s) | Exit | GPU Errors | Status |
|-----------|---------|------|------------|--------|
| 10000 | 6 | 0 | 0 | ✅ PASS |
| 50000 | 7 | 0 | 0 | ✅ PASS |
| 100000 | 12 | 0 | 0 | ✅ PASS |
| 500000 | 34 | 0 | 0 | ✅ PASS |
| 1000000 | 36 | 0 | 0 | ✅ PASS |
| 5000000 | 64 | 0 | 0 | ✅ PASS |

## Test 2: Concurrency — 500K Seeds Each

| max-concurrent | Time (s) | Exit | GPU Errors | Status |
|---------------|---------|------|------------|--------|
| 8 | 34 | 0 | 0 | ✅ PASS |
| 16 | 35 | 0 | 0 | ✅ PASS |
| 26 | 35 | 0 | 0 | ✅ PASS |

## Test 3: Rapid Fire — 10 Sieves, No Cooldown

| Sieve # | Time (s) | Exit | GPU Errors | Status |
|---------|---------|------|------------|--------|
| 1 | 35 | 0 | 0 | ✅ |
| 2 | 32 | 0 | 0 | ✅ |
| 3 | 34 | 0 | 0 | ✅ |
| 4 | 32 | 0 | 0 | ✅ |
| 5 | 32 | 0 | 0 | ✅ |
| 6 | 32 | 0 | 0 | ✅ |
| 7 | 32 | 0 | 0 | ✅ |
| 8 | 33 | 0 | 0 | ✅ |
| 9 | 32 | 0 | 0 | ✅ |
| 10 | 32 | 0 | 0 | ✅ |

**✅ All 10 consecutive sieves passed.**

## Test 4: Recovery Check

**✅ No residual GPU anomalies after 30s recovery.**

---

## Configuration Recommendation

```bash
# Fill in optimal values from results above:
opt_seed_count=           # Highest PASS from Test 1
max_concurrent=           # Highest PASS from Test 2
inter_trial_cooldown=     # If Test 3 failed: required; if passed: optional
```

## Raw Logs

- `benchmark_logs/test1_memory.log` — Seed count memory snapshots
- `benchmark_logs/test2_memory.log` — Concurrency memory snapshots
- `benchmark_logs/test3_memory.log` — Stress test memory snapshots
- `benchmark_logs/test4_memory.log` — Recovery snapshots

---
*Generated: 2026-02-01 16:12:28*
