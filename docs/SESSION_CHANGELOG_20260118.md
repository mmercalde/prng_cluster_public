# SESSION CHANGELOG - 2026-01-18

## ROCm Stability Envelope Validation

### Summary
Identified true ROCm instability cause as host memory pressure during Step 2.5 data loading. Validated 12-GPU concurrency on RX 6600 rigs. Established sample_size=450 as optimal operating point.

### Key Findings

| Previous Assumption | Validated Reality |
|--------------------|-------------------|
| ROCm can't handle high concurrency | Full 12-GPU concurrency is stable |
| Weak CPUs cause contention | i5-9400/i5-8400 are sufficient |
| HIP init collision is dominant failure | Memory pressure during load is the cause |
| Reduce GPU count for stability | Reduce sample_size instead |

### Benchmark Results (100 trials total, 100% success rate)

| Sample Size | Throughput | Status |
|-------------|------------|--------|
| 350 | 14.98 trials/min | ✅ Stable |
| **450** | **15.41 trials/min** | ✅ **Optimal** |
| 550 | 14.66 trials/min | ✅ Stable |
| 650 | 13.13 trials/min | ✅ Stable |
| 750 | 12.45 trials/min | ✅ Stable |

### Performance Improvement
- Old: 5000 samples @ 4 concurrent = ~3.4 trials/min
- New: 450 samples @ 12 concurrent = 15.41 trials/min
- **Factor: 4.5× faster**

### Configuration Changes Applied

**distributed_config.json:**
```json
{
  "hostname": "192.168.3.120",
  "max_concurrent_script_jobs": 12
},
{
  "hostname": "192.168.3.154", 
  "max_concurrent_script_jobs": 12
}
```

**run_scorer_meta_optimizer.sh:**
```bash
--sample-size 450
```

**Both rigs - kernel parameter:**
```bash
amdgpu.ppfeaturemask=0xffff7fff  # GFXOFF disabled
```

### Files Modified
- `distributed_config.json` - concurrency settings
- `run_scorer_meta_optimizer.sh` - sample size
- `benchmark_sample_sizes_v2.sh` - new benchmark script with diagnostics

### Documentation Updated
- CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md - Section 8.4 (ROCm Stability Envelope)
- CHAPTER_3_SCORER_META_OPTIMIZER.md - Section 9.4 (Resource Scaling)
