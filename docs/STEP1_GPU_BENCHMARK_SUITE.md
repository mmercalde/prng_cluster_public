# Step 1 GPU Benchmark Suite
## Determining Optimal Settings for coordinator.py (Seed-Based Jobs)

**Version:** 1.0.0  
**Date:** February 1, 2026  
**Purpose:** Systematic benchmarks to prevent GPU overload during Step 1 (Window Optimizer + Direct Sieve)  
**Modeled on:** Step 2/3 benchmark methodology (January 18, 2026) that established `sample_size=450` @ `max_concurrent=12`

---

## Background

### The Gap

Steps 2-3 (`scripts_coordinator.py`) have validated protections from systematic benchmarking:

| Protection | Step 2/3 Value | Step 1 Status |
|-----------|----------------|---------------|
| Stagger delay | 2.0s | **NONE** |
| Max concurrent jobs | 12 per node | **Unbounded** (--max-concurrent 26) |
| Inter-job cooldown | Built into scripts_coordinator | **NONE between trials** |
| HIP cache cleanup | `_best_effort_gpu_cleanup()` between jobs | Inter-chunk only (Jan 26 fix) |
| VRAM fraction limit | 80% (6.4GB of 8GB) | **Not applied in sieve path** |
| Host memory monitoring | Validated at sample_size=450 | **Never tested** |

### What Step 1 Actually Does

Each Window Optimizer trial executes:
```
Trial N:
  1. Forward sieve → coordinator.py dispatches seed-range jobs to all 26 GPUs
  2. Reverse sieve → coordinator.py dispatches seed-range jobs to all 26 GPUs
  3. Compute intersection (CPU-side)
  [NO COOLDOWN, NO CLEANUP, NO HEALTH CHECK]
Trial N+1:
  1. Forward sieve → immediately dispatches to all 26 GPUs again
  ...
```

With 20 trials, that's **40 full-cluster sieve operations** back-to-back with zero recovery time.

### What We're Looking For

Analogous to how Step 2/3 benchmarks found `sample_size=450` as the sweet spot, we need to find:

1. **Optimal seed_count per trial** — how many seeds per sieve before GPU stress accumulates
2. **Optimal max-concurrent** — whether 26-GPU concurrency is safe for seed jobs or needs limiting
3. **Multi-trial stability ceiling** — how many back-to-back trials before GPU degradation
4. **Optimal inter-trial cooldown** — time between trials for GPU recovery

---

## Prerequisites

### Monitoring Setup (Run on Zeus)

Open 4 terminal tabs before starting benchmarks:

**Tab 1 — rig-6600 GPU Monitor:**
```bash
ssh 192.168.3.120 "watch -n 2 rocm-smi"
```

**Tab 2 — rig-6600b GPU Monitor:**
```bash
ssh 192.168.3.154 "watch -n 2 rocm-smi"
```

**Tab 3 — rig-6600c GPU Monitor:**
```bash
ssh 192.168.3.162 "watch -n 2 rocm-smi"
```

**Tab 4 — Host Memory Monitor (all nodes):**
```bash
watch -n 5 'echo "=== Zeus ===" && free -m | head -2 && echo "=== rig-6600 ===" && ssh 192.168.3.120 "free -m | head -2" && echo "=== rig-6600b ===" && ssh 192.168.3.154 "free -m | head -2" && echo "=== rig-6600c ===" && ssh 192.168.3.162 "free -m | head -2"'
```

### GPU Health Check Script

Save this on Zeus as `~/distributed_prng_analysis/gpu_health_check.sh`:

```bash
#!/bin/bash
# gpu_health_check.sh — Snapshot all GPU states across cluster
# Usage: ./gpu_health_check.sh [label]
# Example: ./gpu_health_check.sh "pre_benchmark_1"

LABEL="${1:-snapshot}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTDIR="benchmark_logs"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/gpu_health_${LABEL}_${TIMESTAMP}.txt"

echo "=== GPU Health Check: $LABEL ===" | tee "$OUTFILE"
echo "Timestamp: $(date)" | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

# Zeus (CUDA)
echo "--- Zeus (CUDA) ---" | tee -a "$OUTFILE"
nvidia-smi --query-gpu=index,temperature.gpu,power.draw,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1 | tee -a "$OUTFILE"
echo "" | tee -a "$OUTFILE"

# ROCm rigs
for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
    HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
    echo "--- $HOSTNAME ($node) ---" | tee -a "$OUTFILE"
    ssh $node "rocm-smi 2>&1" | tee -a "$OUTFILE"
    echo "" | tee -a "$OUTFILE"
    
    # Check for N/A sensors or unknown perf states
    ERRORS=$(ssh $node "rocm-smi 2>&1" | grep -c -E "N/A|unknown")
    if [ "$ERRORS" -gt 0 ]; then
        echo "⚠️  WARNING: $HOSTNAME has $ERRORS sensor anomalies!" | tee -a "$OUTFILE"
    fi
    echo "" | tee -a "$OUTFILE"
done

# Host memory
echo "--- Host Memory ---" | tee -a "$OUTFILE"
echo "Zeus:" | tee -a "$OUTFILE"
free -m | head -2 | tee -a "$OUTFILE"
for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
    HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
    echo "$HOSTNAME:" | tee -a "$OUTFILE"
    ssh $node "free -m | head -2" 2>&1 | tee -a "$OUTFILE"
done

echo "" | tee -a "$OUTFILE"
echo "Saved to: $OUTFILE"
```

```bash
chmod +x gpu_health_check.sh
```

---

## Benchmark 1: Seed Count Scaling

**Purpose:** Find optimal `seed_count` per sieve operation — analogous to finding `sample_size=450` for Step 2.5.

**Hypothesis:** Larger seed counts increase per-GPU memory residency time and accumulated VRAM fragmentation. There's a sweet spot that balances throughput with GPU health.

**Test Matrix:**

| Test | seed_count | trials | max-concurrent | Expected Runtime |
|------|-----------|--------|----------------|-----------------|
| 1A | 10,000 | 3 | 26 | ~1 min |
| 1B | 50,000 | 3 | 26 | ~2 min |
| 1C | 100,000 | 3 | 26 | ~3 min |
| 1D | 500,000 | 3 | 26 | ~5 min |
| 1E | 1,000,000 | 3 | 26 | ~8 min |
| 1F | 5,000,000 | 3 | 26 | ~15 min |
| 1G | 10,000,000 | 3 | 26 | ~25 min |

**Script:** Save as `~/distributed_prng_analysis/benchmark_1_seed_scaling.sh`

```bash
#!/bin/bash
# Benchmark 1: Seed Count Scaling
# Tests different seed_count values with fixed 3 trials to find throughput sweet spot.
# Run from ~/distributed_prng_analysis on Zeus.

cd ~/distributed_prng_analysis

OUTDIR="benchmark_logs/bench1_seed_scaling"
mkdir -p "$OUTDIR"

SEED_COUNTS=(10000 50000 100000 500000 1000000 5000000 10000000)
TRIALS=3
MAX_CONCURRENT=26
PRNG="java_lcg"
LOTTERY="daily3.json"

echo "=========================================="
echo "BENCHMARK 1: SEED COUNT SCALING"
echo "Trials per test: $TRIALS"
echo "Max concurrent: $MAX_CONCURRENT"
echo "PRNG: $PRNG"
echo "=========================================="

# Pre-benchmark health check
./gpu_health_check.sh "bench1_pre"

for SEEDS in "${SEED_COUNTS[@]}"; do
    LABEL="seeds_${SEEDS}"
    LOGFILE="$OUTDIR/${LABEL}.log"
    
    echo ""
    echo "=========================================="
    echo "Testing seed_count=$SEEDS"
    echo "=========================================="
    
    # Pre-test health check
    ./gpu_health_check.sh "bench1_pre_${LABEL}"
    
    # Record host memory before
    echo "--- Host Memory Before ---" >> "$LOGFILE"
    free -m >> "$LOGFILE" 2>&1
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        ssh $node "free -m" >> "$LOGFILE" 2>&1
    done
    
    # Run the test
    START=$(date +%s)
    
    python3 coordinator.py "$LOTTERY" \
        --optimize-window \
        --prng-type "$PRNG" \
        --opt-strategy bayesian \
        --opt-iterations "$TRIALS" \
        --opt-seed-count "$SEEDS" \
        --max-concurrent "$MAX_CONCURRENT" \
        --resume-policy restart \
        2>&1 | tee "$OUTDIR/${LABEL}_output.log"
    
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    # Record host memory after
    echo "--- Host Memory After ---" >> "$LOGFILE"
    free -m >> "$LOGFILE" 2>&1
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        ssh $node "free -m" >> "$LOGFILE" 2>&1
    done
    
    # Post-test health check
    ./gpu_health_check.sh "bench1_post_${LABEL}"
    
    # Summary
    THROUGHPUT=$(echo "scale=2; $SEEDS * $TRIALS * 2 / $ELAPSED" | bc 2>/dev/null || echo "N/A")
    
    echo "---"
    echo "seed_count=$SEEDS | time=${ELAPSED}s | exit=$EXIT_CODE | throughput=${THROUGHPUT} seeds/sec"
    echo "seed_count=$SEEDS | time=${ELAPSED}s | exit=$EXIT_CODE | throughput=${THROUGHPUT} seeds/sec" >> "$OUTDIR/summary.txt"
    echo "---"
    
    # 30-second cooldown between tests to let GPUs fully recover
    echo "Cooling down 30s..."
    sleep 30
done

# Post-benchmark health check
./gpu_health_check.sh "bench1_post"

echo ""
echo "=========================================="
echo "BENCHMARK 1 COMPLETE"
echo "Results in: $OUTDIR/"
echo "Summary: $OUTDIR/summary.txt"
echo "=========================================="
```

**What to Record:**

| seed_count | Wall Time (s) | Exit Code | GPU Errors? | Host Mem Delta | Throughput (seeds/sec) |
|-----------|--------------|-----------|-------------|----------------|----------------------|
| 10,000 | | | | | |
| 50,000 | | | | | |
| 100,000 | | | | | |
| 500,000 | | | | | |
| 1,000,000 | | | | | |
| 5,000,000 | | | | | |
| 10,000,000 | | | | | |

**Success Criteria:** No GPU sensor anomalies (N/A, unknown), exit code 0, host memory stable.

---

## Benchmark 2: Concurrency Scaling

**Purpose:** Verify whether full 26-GPU concurrency is safe for seed-range jobs, or if we need node-level limits like Steps 2-3.

**Hypothesis:** Seed-range jobs (CuPy/sieve) have a different memory profile than script jobs (PyTorch/ML). The 26-GPU ceiling may or may not be appropriate.

**Test Matrix:**

| Test | max-concurrent | seed_count | trials | Notes |
|------|---------------|-----------|--------|-------|
| 2A | 8 | 500,000 | 5 | Conservative (1 rig worth) |
| 2B | 16 | 500,000 | 5 | Moderate (2 rigs) |
| 2C | 20 | 500,000 | 5 | Near-full |
| 2D | 26 | 500,000 | 5 | Full cluster |

**Script:** Save as `~/distributed_prng_analysis/benchmark_2_concurrency.sh`

```bash
#!/bin/bash
# Benchmark 2: Concurrency Scaling
# Tests different --max-concurrent values with fixed seed count.

cd ~/distributed_prng_analysis

OUTDIR="benchmark_logs/bench2_concurrency"
mkdir -p "$OUTDIR"

CONCURRENCY_LEVELS=(8 16 20 26)
SEEDS=500000
TRIALS=5
PRNG="java_lcg"
LOTTERY="daily3.json"

echo "=========================================="
echo "BENCHMARK 2: CONCURRENCY SCALING"
echo "Seed count: $SEEDS"
echo "Trials per test: $TRIALS"
echo "=========================================="

./gpu_health_check.sh "bench2_pre"

for CONC in "${CONCURRENCY_LEVELS[@]}"; do
    LABEL="conc_${CONC}"
    
    echo ""
    echo "=========================================="
    echo "Testing max-concurrent=$CONC"
    echo "=========================================="
    
    ./gpu_health_check.sh "bench2_pre_${LABEL}"
    
    START=$(date +%s)
    
    python3 coordinator.py "$LOTTERY" \
        --optimize-window \
        --prng-type "$PRNG" \
        --opt-strategy bayesian \
        --opt-iterations "$TRIALS" \
        --opt-seed-count "$SEEDS" \
        --max-concurrent "$CONC" \
        --resume-policy restart \
        2>&1 | tee "$OUTDIR/${LABEL}_output.log"
    
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    ./gpu_health_check.sh "bench2_post_${LABEL}"
    
    THROUGHPUT=$(echo "scale=2; $SEEDS * $TRIALS * 2 / $ELAPSED" | bc 2>/dev/null || echo "N/A")
    
    echo "max-concurrent=$CONC | time=${ELAPSED}s | exit=$EXIT_CODE | throughput=${THROUGHPUT} seeds/sec"
    echo "max-concurrent=$CONC | time=${ELAPSED}s | exit=$EXIT_CODE | throughput=${THROUGHPUT} seeds/sec" >> "$OUTDIR/summary.txt"
    
    # 30-second cooldown
    echo "Cooling down 30s..."
    sleep 30
done

./gpu_health_check.sh "bench2_post"

echo ""
echo "=========================================="
echo "BENCHMARK 2 COMPLETE"
echo "Results in: $OUTDIR/summary.txt"
echo "=========================================="
```

**What to Record:**

| max-concurrent | Wall Time (s) | Exit Code | GPU Errors? | Throughput (seeds/sec) |
|---------------|--------------|-----------|-------------|----------------------|
| 8 | | | | |
| 16 | | | | |
| 20 | | | | |
| 26 | | | | |

**Key insight to watch for:** If throughput doesn't scale linearly (e.g., 26 GPUs isn't ~3.25× faster than 8 GPUs), that suggests contention — likely host memory pressure on the mining rigs, same as the Step 2/3 root cause.

---

## Benchmark 3: Multi-Trial Stress Test

**Purpose:** Find the stability ceiling — how many consecutive Window Optimizer trials can run before GPU degradation appears. This is the "how hard can we push it" test.

**Hypothesis:** Without inter-trial cleanup, VRAM fragmentation and HIP allocator state accumulate across trials. At some point, a GPU will report N/A sensors or unknown perf state.

**Test Matrix:**

| Test | trials | seed_count | Notes |
|------|--------|-----------|-------|
| 3A | 5 | 500,000 | Baseline |
| 3B | 10 | 500,000 | Light stress |
| 3C | 20 | 500,000 | Moderate (your failed test) |
| 3D | 30 | 500,000 | Heavy |
| 3E | 50 | 500,000 | Endurance |

**Important:** Run these sequentially. If 3C fails (20 trials), don't bother with 3D/3E — we've found the ceiling.

**Script:** Save as `~/distributed_prng_analysis/benchmark_3_stress.sh`

```bash
#!/bin/bash
# Benchmark 3: Multi-Trial Stress Test
# Runs increasing numbers of consecutive trials to find stability ceiling.
# STOP if any test shows GPU sensor anomalies.

cd ~/distributed_prng_analysis

OUTDIR="benchmark_logs/bench3_stress"
mkdir -p "$OUTDIR"

TRIAL_COUNTS=(5 10 20 30 50)
SEEDS=500000
MAX_CONCURRENT=26
PRNG="java_lcg"
LOTTERY="daily3.json"

echo "=========================================="
echo "BENCHMARK 3: MULTI-TRIAL STRESS TEST"
echo "Seed count: $SEEDS"
echo "Max concurrent: $MAX_CONCURRENT"
echo "=========================================="

./gpu_health_check.sh "bench3_pre"

for TRIALS in "${TRIAL_COUNTS[@]}"; do
    LABEL="trials_${TRIALS}"
    
    echo ""
    echo "=========================================="
    echo "Testing $TRIALS consecutive trials"
    echo "=========================================="
    echo "⚠️  Watch GPU monitors closely!"
    echo "⚠️  If you see N/A sensors or unknown perf, Ctrl+C and note the trial count."
    echo ""
    
    ./gpu_health_check.sh "bench3_pre_${LABEL}"
    
    START=$(date +%s)
    
    python3 coordinator.py "$LOTTERY" \
        --optimize-window \
        --prng-type "$PRNG" \
        --opt-strategy bayesian \
        --opt-iterations "$TRIALS" \
        --opt-seed-count "$SEEDS" \
        --max-concurrent "$MAX_CONCURRENT" \
        --resume-policy restart \
        2>&1 | tee "$OUTDIR/${LABEL}_output.log"
    
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    ./gpu_health_check.sh "bench3_post_${LABEL}"
    
    # Check for GPU anomalies
    GPU_ERRORS=0
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        ERRORS=$(ssh $node "rocm-smi 2>&1" | grep -c -E "N/A|unknown")
        GPU_ERRORS=$((GPU_ERRORS + ERRORS))
    done
    
    STATUS="PASS"
    if [ "$GPU_ERRORS" -gt 0 ]; then
        STATUS="FAIL ($GPU_ERRORS anomalies)"
    fi
    if [ "$EXIT_CODE" -ne 0 ]; then
        STATUS="CRASH (exit=$EXIT_CODE)"
    fi
    
    echo "trials=$TRIALS | time=${ELAPSED}s | status=$STATUS | gpu_errors=$GPU_ERRORS"
    echo "trials=$TRIALS | time=${ELAPSED}s | status=$STATUS | gpu_errors=$GPU_ERRORS" >> "$OUTDIR/summary.txt"
    
    # If we got errors, STOP — no point testing higher
    if [ "$GPU_ERRORS" -gt 0 ] || [ "$EXIT_CODE" -ne 0 ]; then
        echo ""
        echo "🛑 STOPPING: GPU anomalies detected at $TRIALS trials."
        echo "🛑 Stability ceiling found: $((TRIALS - 1)) trials or fewer."
        echo ""
        # Reboot suggestion
        echo "Recommended: Reboot affected rigs before continuing benchmarks."
        break
    fi
    
    # 60-second cooldown between stress levels (longer cooldown for stress test)
    echo "Cooling down 60s..."
    sleep 60
done

./gpu_health_check.sh "bench3_post"

echo ""
echo "=========================================="
echo "BENCHMARK 3 COMPLETE"
echo "Results in: $OUTDIR/summary.txt"
echo "=========================================="
```

**What to Record:**

| Trials | Wall Time (s) | Status | GPU Anomalies | Notes |
|--------|--------------|--------|---------------|-------|
| 5 | | | | |
| 10 | | | | |
| 20 | | | | |
| 30 | | | | |
| 50 | | | | |

**Note about rig-6600b GPU[4]:** This GPU always shows anomalies due to the broken SMU. Exclude it from your error count — look for NEW anomalies on other GPUs.

---

## Benchmark 4: Single-Sieve Memory Profile

**Purpose:** Deep-dive into exactly what happens to GPU VRAM and host RAM during a single sieve operation. This is the "microscope" test — fewer trials, more granular measurement.

**Why separate from Benchmark 1:** Benchmarks 1-3 measure outcomes (did it crash?). Benchmark 4 measures the mechanism (what's happening to memory?).

**Script:** Save as `~/distributed_prng_analysis/benchmark_4_memory_profile.sh`

```bash
#!/bin/bash
# Benchmark 4: Single-Sieve Memory Profile
# Runs a SINGLE forward sieve at different seed counts while capturing
# detailed memory snapshots before, during, and after.

cd ~/distributed_prng_analysis

OUTDIR="benchmark_logs/bench4_memory"
mkdir -p "$OUTDIR"

SEED_COUNTS=(50000 500000 5000000 10000000)
MAX_CONCURRENT=26
PRNG="java_lcg"
LOTTERY="daily3.json"

echo "=========================================="
echo "BENCHMARK 4: SINGLE-SIEVE MEMORY PROFILE"
echo "=========================================="

capture_memory() {
    local LABEL=$1
    local OUTFILE=$2
    echo "--- Memory snapshot: $LABEL ---" >> "$OUTFILE"
    echo "Timestamp: $(date +%H:%M:%S)" >> "$OUTFILE"
    
    echo "Zeus:" >> "$OUTFILE"
    free -m >> "$OUTFILE" 2>&1
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader >> "$OUTFILE" 2>&1
    
    for node in 192.168.3.120 192.168.3.154 192.168.3.162; do
        HOSTNAME=$(ssh $node hostname 2>/dev/null || echo "$node")
        echo "$HOSTNAME host RAM:" >> "$OUTFILE"
        ssh $node "free -m | head -2" >> "$OUTFILE" 2>&1
        echo "$HOSTNAME VRAM:" >> "$OUTFILE"
        ssh $node "rocm-smi --showmeminfo vram 2>/dev/null || rocm-smi 2>&1" >> "$OUTFILE" 2>&1
    done
    echo "" >> "$OUTFILE"
}

for SEEDS in "${SEED_COUNTS[@]}"; do
    LABEL="mem_seeds_${SEEDS}"
    MEMLOG="$OUTDIR/${LABEL}_memory.log"
    
    echo ""
    echo "=========================================="
    echo "Memory profiling: seed_count=$SEEDS (single forward sieve)"
    echo "=========================================="
    
    # Capture BEFORE
    capture_memory "BEFORE" "$MEMLOG"
    
    # Run single forward sieve (NOT window optimizer — just one sieve pass)
    START=$(date +%s)
    
    python3 coordinator.py "$LOTTERY" \
        --resume-policy restart \
        --max-concurrent "$MAX_CONCURRENT" \
        --method residue_sieve \
        --prng-type "$PRNG" \
        --window-size 512 \
        --skip-max 20 \
        --seeds "$SEEDS" \
        2>&1 | tee "$OUTDIR/${LABEL}_output.log"
    
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    
    # Capture AFTER (immediately — before cleanup has time to run)
    capture_memory "AFTER" "$MEMLOG"
    
    # Wait 10 seconds, capture RECOVERED state
    sleep 10
    capture_memory "RECOVERED (10s)" "$MEMLOG"
    
    echo "seed_count=$SEEDS | time=${ELAPSED}s | exit=$EXIT_CODE"
    echo "seed_count=$SEEDS | time=${ELAPSED}s | exit=$EXIT_CODE" >> "$OUTDIR/summary.txt"
    
    # 30-second cooldown
    sleep 30
done

echo ""
echo "=========================================="
echo "BENCHMARK 4 COMPLETE"
echo "Memory logs in: $OUTDIR/"
echo "=========================================="
```

**What to Record:**

For each seed_count, compare BEFORE vs AFTER vs RECOVERED:

| seed_count | Host RAM Used (Before→After→Recovered) | VRAM Used (Before→After→Recovered) | Recovery Clean? |
|-----------|---------------------------------------|-----------------------------------|-----------------|
| 50,000 | | | |
| 500,000 | | | |
| 5,000,000 | | | |
| 10,000,000 | | | |

---

## Execution Order

Run the benchmarks in this specific order:

1. **Benchmark 4 first** — Memory profiling with single sieves. Lowest risk, gives us the memory baseline.
2. **Benchmark 1 second** — Seed count scaling. Uses Benchmark 4 results to understand why certain seed counts stress GPUs.
3. **Benchmark 2 third** — Concurrency scaling. By now we know the optimal seed count.
4. **Benchmark 3 last** — Stress test. Uses optimal seed_count and max-concurrent from Benchmarks 1-2.

**Total estimated time:** 2-4 hours depending on how far you push Benchmark 3.

**Between-benchmark procedure:**
1. Run `./gpu_health_check.sh "between_benchmarks"`
2. If any GPU shows anomalies → reboot that rig before continuing
3. Allow 2-minute cooldown between benchmarks

---

## Analysis Framework

After running all benchmarks, fill in this summary to determine optimal Step 1 settings:

### Optimal Configuration (to be determined)

```bash
# Step 1 optimal configuration
# (fill in after benchmarks)
opt_seed_count=?????           # From Benchmark 1 (sweet spot)
max_concurrent=??              # From Benchmark 2 (safe ceiling)
max_consecutive_trials=??      # From Benchmark 3 (stability ceiling)
inter_trial_cooldown=??s       # Derived from Benchmark 3 results
```

### Decision Matrix

**seed_count:** Pick the highest value from Benchmark 1 where:
- ✅ Exit code 0
- ✅ Zero GPU anomalies post-test
- ✅ Host memory delta < 500MB
- ✅ Throughput still scales (not plateaued)

**max-concurrent:** Pick the highest value from Benchmark 2 where:
- ✅ Throughput scales roughly linearly
- ✅ Zero GPU anomalies
- ✅ No host memory pressure on mining rigs

**max_consecutive_trials:** From Benchmark 3:
- ✅ Stability ceiling is the LAST trial count that passed cleanly
- ✅ If 50 trials passes, the ceiling is very high (may not need inter-trial cooldown)
- ✅ If 20 trials fails (matching your observed error), that confirms the need for inter-trial cleanup

### What to Implement Based on Results

After benchmarks, we'll need to add protections to one or both of:

1. **`window_optimizer.py`** (or `window_optimizer_integration_final.py`):
   - Inter-trial cooldown (`time.sleep(N)`)
   - Inter-trial GPU cleanup call (SSH `_best_effort_gpu_cleanup` on all nodes)
   - GPU health check between trials (abort if anomalies detected)

2. **`coordinator.py`**:
   - Respect a new `--inter-trial-delay` parameter
   - Add `--max-seed-count-per-trial` safety cap
   - Optional: per-node concurrency limits for seed jobs (if Benchmark 2 shows issues)

---

## Quick Reference: Deploy All Scripts

```bash
# From ser8, copy to Zeus:
scp STEP1_GPU_BENCHMARK_SUITE.md rzeus:~/distributed_prng_analysis/docs/

# On Zeus, create the scripts:
cd ~/distributed_prng_analysis
# Create gpu_health_check.sh (from section above)
# Create benchmark_1_seed_scaling.sh
# Create benchmark_2_concurrency.sh
# Create benchmark_3_stress.sh
# Create benchmark_4_memory_profile.sh
chmod +x gpu_health_check.sh benchmark_*.sh
```

---

*End of Step 1 GPU Benchmark Suite v1.0.0*
