#!/bin/bash
# deploy_npz_v3_fix.sh - Master deployment for NPZ metadata preservation
# Team Beta ruling: January 23, 2026
#
# This fixes the silent data loss where 14/47 ML features were 0.0
# because NPZ v2.0 only saved 3 arrays instead of 22 fields.
#
# USAGE (from ser8):
#   scp patch_npz_v3.sh patch_survivor_loader_v2.sh deploy_npz_v3_fix.sh rzeus:~/distributed_prng_analysis/
#   ssh rzeus "cd ~/distributed_prng_analysis && bash deploy_npz_v3_fix.sh"

set -e
cd ~/distributed_prng_analysis

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  NPZ v3.0 METADATA FIX - MASTER DEPLOYMENT                   ║"
echo "║  Team Beta ruling: January 23, 2026                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Fixes: 14/47 ML features silently zeroed due to missing     ║"
echo "║         metadata in NPZ format.                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Patch converter
echo "[1/4] Patching convert_survivors_to_binary.py..."
if [ -f patch_npz_v3.sh ]; then
    bash patch_npz_v3.sh
else
    echo "ERROR: patch_npz_v3.sh not found!"
    exit 1
fi

echo ""

# Step 2: Patch loader
echo "[2/4] Patching utils/survivor_loader.py..."
if [ -f patch_survivor_loader_v2.sh ]; then
    bash patch_survivor_loader_v2.sh
else
    echo "ERROR: patch_survivor_loader_v2.sh not found!"
    exit 1
fi

echo ""

# Step 3: Regenerate NPZ
echo "[3/4] Regenerating NPZ with full metadata..."
if [ ! -f bidirectional_survivors.json ]; then
    echo "ERROR: bidirectional_survivors.json not found!"
    exit 1
fi

python3 convert_survivors_to_binary.py bidirectional_survivors.json

echo ""

# Step 4: Verify
echo "[4/4] Verifying NPZ v3.0..."
python3 << 'VERIFY'
import numpy as np

data = np.load('bidirectional_survivors_binary.npz')
keys = sorted(data.keys())

print(f"Arrays in NPZ: {len(keys)}")
print(f"Expected:      22")
print()

# Check critical metadata fields
critical = ['seeds', 'skip_min', 'skip_max', 'forward_count', 'bidirectional_count']
missing = [k for k in critical if k not in keys]

if missing:
    print(f"❌ MISSING CRITICAL FIELDS: {missing}")
    exit(1)

print("✅ All critical metadata fields present")
print()

# Sample verification
print("Sample values (first survivor):")
print(f"  seed:               {data['seeds'][0]}")
print(f"  skip_min:           {data['skip_min'][0]}")
print(f"  skip_max:           {data['skip_max'][0]}")
print(f"  forward_count:      {data['forward_count'][0]}")
print(f"  bidirectional_count:{data['bidirectional_count'][0]}")

# Check non-zero
nonzero = sum(1 for k in ['skip_min', 'skip_max', 'forward_count'] if data[k].sum() > 0)
if nonzero >= 2:
    print()
    print("✅ Metadata fields contain real data (not all zeros)")
else:
    print()
    print("⚠️ WARNING: Some metadata fields may be all zeros")
VERIFY

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DEPLOYMENT COMPLETE                                         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "NEXT STEPS:"
echo ""
echo "1. Copy updated NPZ to remote rigs:"
echo "   scp bidirectional_survivors_binary.npz 192.168.3.120:~/distributed_prng_analysis/"
echo "   scp bidirectional_survivors_binary.npz 192.168.3.154:~/distributed_prng_analysis/"
echo ""
echo "2. Copy updated survivor_loader.py to remote rigs:"
echo "   scp utils/survivor_loader.py 192.168.3.120:~/distributed_prng_analysis/utils/"
echo "   scp utils/survivor_loader.py 192.168.3.154:~/distributed_prng_analysis/utils/"
echo ""
echo "3. Test Step 3:"
echo "   PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 3"
echo ""
echo "4. Git commit:"
echo "   git add convert_survivors_to_binary.py utils/survivor_loader.py"
echo "   git commit -m 'fix(npz): preserve all 22 metadata fields (v3.0)'"
echo ""
