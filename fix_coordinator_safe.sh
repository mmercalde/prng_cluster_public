#!/bin/bash
cd ~/distributed_prng_analysis

echo "🔧 Backing up coordinator.py..."
cp coordinator.py coordinator.py.backup_$(date +%s)

# Find the CORRECT pattern
LINE=$(grep -n "if 'script' in job_data:" coordinator.py | tail -1 | cut -d: -f1)

if [ -z "$LINE" ]; then
    echo "❌ ERROR: Could not find 'if 'script' in job_data:' pattern"
    exit 1
fi

echo "✅ Found script block at line: $LINE"

# Find the END of the block (next elif, else, or de-indented line)
END_LINE=$(awk -v start=$LINE '
    NR > start && /^        elif|^        else|^    [^ ]/ {print NR; exit}
' coordinator.py)

if [ -z "$END_LINE" ]; then
    echo "❌ ERROR: Could not find end of script block"
    exit 1
fi

# Calculate lines to delete
LINES_TO_DELETE=$((END_LINE - LINE))
echo "📏 Will replace $LINES_TO_DELETE lines (line $LINE to $((END_LINE-1)))"

# Create replacement with CORRECT GPU ID (0, not worker.gpu_id)
cat > /tmp/replacement.txt << 'EEOF'
        if 'script' in job_data:
            # Route through distributed_worker.py to use ThreadPool
            job_file = f"job_{job.job_id}.json"
            job_path = os.path.join(node.script_path, job_file)
            with open(job_path, 'w') as f:
                json.dump(job_data, f)
            
            worker_script = "distributed_worker.py"
            mining_flag = "--mining-mode" if job_data.get('mining_mode', False) else ""
            worker_args = f"{job_file} --gpu-id 0 {mining_flag}"
            
            cmd_str = (
                f"source {activate_path} && "
                f"CUDA_VISIBLE_DEVICES={worker.gpu_id} "
                f"python -u {worker_script} {worker_args}"
            ).strip()
EEOF

# Use Python for atomic replacement (safer than sed)
python3 << PYEOF
with open('coordinator.py', 'r') as f:
    lines = f.readlines()

with open('/tmp/replacement.txt', 'r') as f:
    replacement = f.readlines()

# Replace lines atomically
new_lines = lines[:$LINE-1] + replacement + lines[$END_LINE-1:]

with open('coordinator.py', 'w') as f:
    f.writelines(new_lines)

print("✅ Replacement complete")
PYEOF

# Verify
echo ""
echo "🔍 Verifying fix..."

if grep -q 'worker_script = "distributed_worker.py"' coordinator.py; then
    echo "✅ Fix verified: distributed_worker.py routing detected"
else
    echo "❌ Fix verification failed!"
    echo "Restoring backup..."
    cp coordinator.py.backup_* coordinator.py
    exit 1
fi

# Syntax check
echo ""
echo "🔍 Running syntax check..."
python3 -m py_compile coordinator.py

if [ $? -eq 0 ]; then
    echo "✅ Syntax check passed!"
    echo ""
    echo "🚀 Ready to test:"
    echo "   bash run_scorer_meta_optimizer.sh 6"
else
    echo "❌ Syntax error! Restoring backup..."
    cp coordinator.py.backup_* coordinator.py
    exit 1
fi

# Cleanup
rm /tmp/replacement.txt
