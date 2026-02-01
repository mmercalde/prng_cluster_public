#!/bin/bash
# verify_cluster.sh
# Verify all 26 GPUs are online and accessible
# Run from ~/distributed_prng_analysis on Zeus

echo "=========================================="
echo "  CLUSTER VERIFICATION - 26 GPU Check"
echo "=========================================="
echo ""

# Node configuration
declare -A NODES
NODES["localhost"]="2:CUDA"
NODES["192.168.3.120"]="8:ROCm"
NODES["192.168.3.154"]="8:ROCm"
NODES["192.168.3.162"]="8:ROCm"

TOTAL_EXPECTED=26
TOTAL_FOUND=0

echo "=== Node Connectivity ==="
for host in localhost 192.168.3.120 192.168.3.154 192.168.3.162; do
    IFS=':' read -r expected_gpus backend <<< "${NODES[$host]}"
    
    if [ "$host" == "localhost" ]; then
        hostname_result=$(hostname)
        uptime_result=$(uptime -p)
        echo "✅ $host ($hostname_result) - $uptime_result"
    else
        if ssh -o ConnectTimeout=3 "$host" "echo OK" >/dev/null 2>&1; then
            hostname_result=$(ssh "$host" "hostname")
            uptime_result=$(ssh "$host" "uptime -p")
            echo "✅ $host ($hostname_result) - $uptime_result"
        else
            echo "❌ $host - UNREACHABLE"
        fi
    fi
done

echo ""
echo "=== GPU Detection ==="

# Zeus (localhost) - CUDA
echo "--- Zeus (localhost) - Expected: 2 GPUs ---"
gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
echo "  Found: $gpu_count GPUs"
TOTAL_FOUND=$((TOTAL_FOUND + gpu_count))

# ROCm rigs
for host in 192.168.3.120 192.168.3.154 192.168.3.162; do
    IFS=':' read -r expected_gpus backend <<< "${NODES[$host]}"
    hostname_result=$(ssh -o ConnectTimeout=3 "$host" "hostname" 2>/dev/null || echo "unknown")
    
    echo "--- $hostname_result ($host) - Expected: $expected_gpus GPUs ---"
    
    gpu_count=$(ssh -o ConnectTimeout=10 "$host" \
        "source ~/rocm_env/bin/activate && python3 -c 'import cupy; print(cupy.cuda.runtime.getDeviceCount())'" \
        2>/dev/null || echo "0")
    
    if [ "$gpu_count" -gt 0 ]; then
        echo "  Found: $gpu_count GPUs ✅"
        TOTAL_FOUND=$((TOTAL_FOUND + gpu_count))
    else
        echo "  Found: 0 GPUs ❌"
    fi
done

echo ""
echo "=========================================="
echo "  SUMMARY"
echo "=========================================="
echo "  Expected: $TOTAL_EXPECTED GPUs"
echo "  Found:    $TOTAL_FOUND GPUs"
if [ "$TOTAL_FOUND" -eq "$TOTAL_EXPECTED" ]; then
    echo "  Status:   ✅ ALL GPUS ONLINE"
else
    echo "  Status:   ⚠️  MISSING $((TOTAL_EXPECTED - TOTAL_FOUND)) GPUs"
fi
echo "=========================================="
echo ""

if [ "$TOTAL_FOUND" -eq "$TOTAL_EXPECTED" ]; then
    echo "Cluster ready! Run a test:"
    echo "  python3 coordinator.py test_xorshift32_hybrid.json --method residue_sieve --prng-type xorshift32_hybrid --seeds 10000 --hybrid"
fi
