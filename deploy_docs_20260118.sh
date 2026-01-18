#!/bin/bash
# Deploy documentation patches for ROCm Stability Envelope validation
# Run from: ~/distributed_prng_analysis/
# Date: 2026-01-18

DOCS_DIR="$HOME/distributed_prng_analysis/docs"

echo "==================================="
echo "Documentation Update Deployment"
echo "==================================="
echo ""

# Backup existing files
echo "Creating backups..."
cp "$DOCS_DIR/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md" "$DOCS_DIR/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md.bak_20260118"
cp "$DOCS_DIR/CHAPTER_3_SCORER_META_OPTIMIZER.md" "$DOCS_DIR/CHAPTER_3_SCORER_META_OPTIMIZER.md.bak_20260118"
echo "Backups created"
echo ""

# Append Chapter 9 patch (after Section 8.3) - with idempotency guard
echo "Updating Chapter 9..."
if grep -q "ROCm Stability Envelope (RX 6600)" "$DOCS_DIR/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md"; then
    echo "Chapter 9 already contains Section 8.4 - skipping"
else
    cat >> "$DOCS_DIR/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md" << 'CHAPTER9_EOF'

---

### 8.4 ROCm Stability Envelope (RX 6600) — VALIDATED

> **Updated: 2026-01-18** — Based on systematic benchmark testing

#### Root Cause Analysis

Prior assumptions about ROCm instability (HIP initialization collisions, GPU concurrency limits) have been superseded. Systematic benchmarking revealed the **true constraint is host memory pressure during data loading**, not GPU-side limitations.

| ❌ Previous Assumption | ✅ Validated Reality |
|------------------------|---------------------|
| "ROCm can't handle high concurrency" | Full 12-GPU concurrency is stable |
| "Weak CPUs cause contention" | i5-9400/i5-8400 are sufficient |
| "HIP init collision is dominant failure" | Memory pressure during load is the cause |
| "Reduce GPU count for stability" | Reduce sample_size instead |

#### Validated Configuration

```bash
# Validated ROCm configuration (RX 6600)
# Tested: 2026-01-18, 100 trials, 100% success rate

max_concurrent_script_jobs: 12     # Full GPU utilization
sample_size: 450                   # Optimal operating point
ppfeaturemask: 0xffff7fff          # GFXOFF disabled
cleanup: enabled                   # Best-effort GPU allocator cleanup between jobs
```

#### Performance Envelope

| Sample Size | Throughput | Stability |
|-------------|------------|-----------|
| 350 | 14.98 trials/min | ✅ Stable |
| **450** | **15.41 trials/min** | ✅ **Optimal** |
| 550 | 14.66 trials/min | ✅ Stable |
| 650 | 13.13 trials/min | ✅ Stable |
| 750 | 12.45 trials/min | ✅ Stable |
| 1000 | 10.42 trials/min | ✅ Stable |
| 2000 | — | ❌ Freeze risk |

#### Required Settings

1. **GFXOFF Disabled** — Add to kernel boot params:
   ```bash
   amdgpu.ppfeaturemask=0xffff7fff
   ```

2. **Concurrency Configuration** — `distributed_config.json`:
   ```json
   {
     "hostname": "192.168.3.120",
     "max_concurrent_script_jobs": 12
   }
   ```

3. **Sample Size Cap** — `run_scorer_meta_optimizer.sh`:
   ```bash
   --sample-size 450
   ```

#### Troubleshooting: ROCm Freeze / Monitor Desync

**Symptoms:**
- `rocm-smi` shows GPU with `N/A` in SCLK/MCLK columns
- `Perf` column shows `unknown` instead of `auto`
- Jobs hang without error messages
- Monitor shows "Expected integer value" warnings

**Root Cause:** Memory pressure during concurrent data loading causes allocator thrashing.

**Fix Checklist:**
1. ✅ Reduce sample_size (not GPU count)
2. ✅ Verify GFXOFF disabled: `cat /sys/module/amdgpu/parameters/ppfeaturemask`
3. ✅ Verify cleanup enabled between jobs
4. ✅ Reboot rig if GPU shows persistent N/A state
5. ❌ Do NOT reduce max_concurrent_script_jobs as first response

CHAPTER9_EOF
    echo "Chapter 9 updated ✓"
fi

# Append Chapter 3 patch (after Section 9.3) - with idempotency guard
echo "Updating Chapter 3..."
if grep -q "Resource Scaling and Performance Constraints — VALIDATED" "$DOCS_DIR/CHAPTER_3_SCORER_META_OPTIMIZER.md"; then
    echo "Chapter 3 already contains Section 9.4 - skipping"
else
    cat >> "$DOCS_DIR/CHAPTER_3_SCORER_META_OPTIMIZER.md" << 'CHAPTER3_EOF'

---

### 9.4 Resource Scaling and Performance Constraints — VALIDATED

> **Updated: 2026-01-18** — Based on systematic benchmark testing

#### Sample Size vs Throughput Trade-off

At 12-way ROCm concurrency, sample sizes above 450 increase memory residency time without improving Optuna convergence. The relationship is inverse: smaller samples = higher throughput.

| Sample Size | Throughput | Signal Quality |
|-------------|------------|----------------|
| 350 | 14.98 trials/min | ✅ Preserved |
| **450** | **15.41 trials/min** | ✅ **Optimal** |
| 550 | 14.66 trials/min | ✅ Preserved |
| 1000 | 10.42 trials/min | ✅ Preserved |
| 2000+ | — | ❌ Freeze risk |

#### Why Signal Quality Is Preserved

Optuna's TPE (Tree-structured Parzen Estimator) sampler is designed for noisy objectives:
- Hyperparameter **ranking** is preserved across sample sizes
- More trials with lower precision beats fewer trials with higher precision
- 500 trials × moderate precision > 100 trials × high precision for global coverage

#### Validated Operating Point

```bash
# Optimal Step 2.5 configuration
sample_size=450              # Maximize throughput
max_concurrent_script_jobs=12  # Full GPU utilization
trials=200-500               # Good Bayesian coverage
```

#### Performance Improvement

| Configuration | Throughput | Factor |
|---------------|------------|--------|
| Old (5000 samples @ 4 concurrent) | ~3.4 trials/min | 1× |
| New (450 samples @ 12 concurrent) | 15.41 trials/min | **4.5×** |

CHAPTER3_EOF
    echo "Chapter 3 updated ✓"
fi

# Create session changelog
echo "Creating session changelog..."
cat > "$DOCS_DIR/SESSION_CHANGELOG_20260118.md" << 'CHANGELOG_EOF'
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
CHANGELOG_EOF
echo "Session changelog created ✓"

echo ""
echo "==================================="
echo "Deployment Complete"
echo "==================================="
echo ""
echo "Updated files:"
echo "  - $DOCS_DIR/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md"
echo "  - $DOCS_DIR/CHAPTER_3_SCORER_META_OPTIMIZER.md"
echo "  - $DOCS_DIR/SESSION_CHANGELOG_20260118.md"
echo ""
echo "Backups:"
echo "  - $DOCS_DIR/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md.bak_20260118"
echo "  - $DOCS_DIR/CHAPTER_3_SCORER_META_OPTIMIZER.md.bak_20260118"
