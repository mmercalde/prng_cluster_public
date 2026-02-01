#!/bin/bash
# deploy_to_rigs.sh
# Deploy updated files to all 3 mining rigs
# Run from ~/distributed_prng_analysis on Zeus

set -e

echo "=== Deploying to All Mining Rigs ==="
echo "Target: rig-6600 (192.168.3.120), rig-6600b (192.168.3.154), rig-6600c (192.168.3.162)"
echo ""

# Core files to deploy
CORE_FILES=(
    "distributed_worker.py"
    "sieve_filter.py"
    "enhanced_gpu_model_id.py"
    "survivor_scorer.py"
    "reinforcement_engine.py"
    "scorer_trial_worker.py"
    "anti_overfit_trial_worker.py"
    "prng_registry.py"
    "hybrid_strategy.py"
    "adaptive_thresholds.py"
    "distributed_config.json"
)

# All three rigs
HOSTS=(
    "192.168.3.120"
    "192.168.3.154"
    "192.168.3.162"
)

HOSTNAMES=(
    "rig-6600"
    "rig-6600b"
    "rig-6600c"
)

# Deploy to each rig
for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    name="${HOSTNAMES[$i]}"
    
    echo "=== Deploying to $name ($host) ==="
    
    # Test connectivity first
    if ! ssh -o ConnectTimeout=5 "$host" "echo OK" >/dev/null 2>&1; then
        echo "  ❌ Cannot connect to $host - skipping"
        continue
    fi
    
    # Create directory if needed (for new rig)
    ssh "$host" "mkdir -p ~/distributed_prng_analysis"
    
    # Deploy each file
    for file in "${CORE_FILES[@]}"; do
        if [ -f "$file" ]; then
            scp -q "$file" "$host:~/distributed_prng_analysis/"
            echo "  ✅ $file"
        else
            echo "  ⏭️  $file not found"
        fi
    done
    
    # Deploy modules directory
    if [ -d "modules" ]; then
        scp -rq modules "$host:~/distributed_prng_analysis/"
        echo "  ✅ modules/"
    fi
    
    echo ""
done

echo "=== Verifying Checksums ==="
for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    name="${HOSTNAMES[$i]}"
    
    echo "$name ($host):"
    ssh "$host" "cd ~/distributed_prng_analysis && sha256sum distributed_worker.py sieve_filter.py 2>/dev/null | head -2" || echo "  Could not verify"
done

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next: Test GPU connectivity"
echo "  ssh 192.168.3.120 'source ~/rocm_env/bin/activate && python3 -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\"'"
echo "  ssh 192.168.3.154 'source ~/rocm_env/bin/activate && python3 -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\"'"
echo "  ssh 192.168.3.162 'source ~/rocm_env/bin/activate && python3 -c \"import cupy; print(cupy.cuda.runtime.getDeviceCount())\"'"
