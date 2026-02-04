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
