#!/bin/bash
# patch_rig6600c_comprehensive.sh
# Patches ALL remaining files that need rig-6600c support
# Run from ~/distributed_prng_analysis on Zeus
# NOTE: First 7 files were already patched by patch_rocm_prelude.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PATCHED=0
SKIPPED=0
FAILED=0

backup_and_patch() {
    local file="$1"
    local desc="$2"
    if [ -f "$file" ]; then
        cp "$file" "${file}.bak_${TIMESTAMP}"
        echo "  📄 $file — $desc"
        return 0
    else
        echo "  ⏭️  $file — not found, skipping"
        SKIPPED=$((SKIPPED + 1))
        return 1
    fi
}

verify_patch() {
    local file="$1"
    if grep -q "rig-6600c" "$file"; then
        echo "     ✅ Verified"
        PATCHED=$((PATCHED + 1))
    else
        echo "     ❌ FAILED — rig-6600c not found after patch"
        FAILED=$((FAILED + 1))
    fi
}

echo "=============================================="
echo "  Comprehensive rig-6600c Patch"
echo "  Timestamp: $TIMESTAMP"
echo "=============================================="
echo ""

# -----------------------------------------------
# 1. full_scoring_worker.py (lines 71, 80)
# -----------------------------------------------
echo "=== 1. full_scoring_worker.py ==="
if backup_and_patch "full_scoring_worker.py" "ROCm hostname check"; then
    sed -i 's/if HOST in \["rig-6600", "rig-6600b"\]/if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]/g' full_scoring_worker.py
    verify_patch full_scoring_worker.py
fi

# -----------------------------------------------
# 2. reverse_sieve_filter.py (line 55)
# -----------------------------------------------
echo "=== 2. reverse_sieve_filter.py ==="
if backup_and_patch "reverse_sieve_filter.py" "ROCm hostname check"; then
    sed -i 's/if HOST in \["rig-6600", "rig-6600b"\]/if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]/g' reverse_sieve_filter.py
    verify_patch reverse_sieve_filter.py
fi

# -----------------------------------------------
# 3. reverse_sieve_filter_INTEGRATED.py (line 16)
# -----------------------------------------------
echo "=== 3. reverse_sieve_filter_INTEGRATED.py ==="
if backup_and_patch "reverse_sieve_filter_INTEGRATED.py" "ROCm hostname check"; then
    sed -i 's/if HOST in \["rig-6600", "rig-6600b"\]/if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]/g' reverse_sieve_filter_INTEGRATED.py
    verify_patch reverse_sieve_filter_INTEGRATED.py
fi

# -----------------------------------------------
# 4. sieve_filter_INTEGRATED.py (line 38)
# -----------------------------------------------
echo "=== 4. sieve_filter_INTEGRATED.py ==="
if backup_and_patch "sieve_filter_INTEGRATED.py" "ROCm hostname check"; then
    sed -i 's/if HOST in \["rig-6600", "rig-6600b"\]/if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]/g' sieve_filter_INTEGRATED.py
    verify_patch sieve_filter_INTEGRATED.py
fi

# -----------------------------------------------
# 5. aggregate_reinforcement_shards.py (line 20)
# -----------------------------------------------
echo "=== 5. aggregate_reinforcement_shards.py ==="
if backup_and_patch "aggregate_reinforcement_shards.py" "ROCm hostname check"; then
    sed -i 's/if HOST in \["rig-6600", "rig-6600b"\]/if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]/g' aggregate_reinforcement_shards.py
    verify_patch aggregate_reinforcement_shards.py
fi

# -----------------------------------------------
# 6. test_gpu_capability.py (lines 22, 71)
# -----------------------------------------------
echo "=== 6. test_gpu_capability.py ==="
if backup_and_patch "test_gpu_capability.py" "ROCm hostname check"; then
    sed -i 's/if HOST in \["rig-6600", "rig-6600b"\]/if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]/g' test_gpu_capability.py
    verify_patch test_gpu_capability.py
fi

# -----------------------------------------------
# 7. test_max_window.py (line 15)
# -----------------------------------------------
echo "=== 7. test_max_window.py ==="
if backup_and_patch "test_max_window.py" "ROCm hostname check"; then
    sed -i 's/if HOST in \["rig-6600", "rig-6600b"\]/if HOST in ["rig-6600", "rig-6600b", "rig-6600c"]/g' test_max_window.py
    verify_patch test_max_window.py
fi

# -----------------------------------------------
# 8. scripts_coordinator.py (line 79)
#    ROCM_HOSTNAMES = ['192.168.3.120', '192.168.3.154', 'rig-6600', 'rig-6600b']
# -----------------------------------------------
echo "=== 8. scripts_coordinator.py ==="
if backup_and_patch "scripts_coordinator.py" "ROCM_HOSTNAMES list"; then
    sed -i "s/ROCM_HOSTNAMES = \['192.168.3.120', '192.168.3.154', 'rig-6600', 'rig-6600b'\]/ROCM_HOSTNAMES = ['192.168.3.120', '192.168.3.154', '192.168.3.162', 'rig-6600', 'rig-6600b', 'rig-6600c']/" scripts_coordinator.py
    verify_patch scripts_coordinator.py
fi

# -----------------------------------------------
# 9. agents/agent_core.py (line 416)
#    'nodes': ['zeus', 'rig-6600', 'rig-6600b']
# -----------------------------------------------
echo "=== 9. agents/agent_core.py ==="
if backup_and_patch "agents/agent_core.py" "nodes list"; then
    sed -i "s/'nodes': \['zeus', 'rig-6600', 'rig-6600b'\]/'nodes': ['zeus', 'rig-6600', 'rig-6600b', 'rig-6600c']/" agents/agent_core.py
    verify_patch agents/agent_core.py
fi

# -----------------------------------------------
# 10. integration/metadata_writer.py (line 352)
#     "nodes": ["zeus", "rig-6600", "rig-6600b"]
# -----------------------------------------------
echo "=== 10. integration/metadata_writer.py ==="
if backup_and_patch "integration/metadata_writer.py" "nodes list"; then
    sed -i 's/"nodes": \["zeus", "rig-6600", "rig-6600b"\]/"nodes": ["zeus", "rig-6600", "rig-6600b", "rig-6600c"]/' integration/metadata_writer.py
    verify_patch integration/metadata_writer.py
fi

# -----------------------------------------------
# 11. distributed_worker.py (line 346)
#     if hostname in ["rig-6600", "rig-6600b", "rig-6600xt"]:
# -----------------------------------------------
echo "=== 11. distributed_worker.py (line 346 — secondary pattern) ==="
if [ -f "distributed_worker.py" ]; then
    if grep -q '"rig-6600", "rig-6600b", "rig-6600xt"' distributed_worker.py; then
        sed -i 's/"rig-6600", "rig-6600b", "rig-6600xt"/"rig-6600", "rig-6600b", "rig-6600c", "rig-6600xt"/' distributed_worker.py
        echo "     ✅ Secondary pattern patched"
        PATCHED=$((PATCHED + 1))
    else
        echo "     ⏭️  Pattern not found (may already include rig-6600c)"
    fi
else
    echo "  ⏭️  distributed_worker.py not found"
fi

# -----------------------------------------------
# 12. chunk_size_config.py
#     Needs new rig-6600c entry + hostname mapping + node list
# -----------------------------------------------
echo "=== 12. chunk_size_config.py ==="
if backup_and_patch "chunk_size_config.py" "chunk config for rig-6600c"; then
    # Add rig-6600c config block after rig-6600b block
    sed -i '/"rig-6600b": {/,/},/{
        /},/a\    # rig-6600c - 8GB RAM, 8 GPUs (same specs as rig-6600b)\
    "rig-6600c": {\
        "chunk_size": 5000,\
        "max_concurrent": 8,\
        "ram_gb": 8,\
        "gpu_count": 8\
    },
    }' chunk_size_config.py

    # Add hostname mapping
    sed -i 's/"rig-6600b": "rig-6600b",/"rig-6600b": "rig-6600b",\n    "192.168.3.162": "rig-6600c",\n    "rig-6600c": "rig-6600c",/' chunk_size_config.py

    # Add to node list
    sed -i 's/nodes = \["zeus", "rig-6600", "rig-6600b"\]/nodes = ["zeus", "rig-6600", "rig-6600b", "rig-6600c"]/' chunk_size_config.py

    verify_patch chunk_size_config.py
fi

# -----------------------------------------------
# 13. web_dashboard.py (line 925)
#     Add rig-6600c row to HTML table
# -----------------------------------------------
echo "=== 13. web_dashboard.py ==="
if backup_and_patch "web_dashboard.py" "dashboard HTML table"; then
    sed -i '/<td>rig-6600b (192.168.3.154)<\/td>/,/<\/tr>/{
        /<\/tr>/a\                <tr>\n                <td>rig-6600c (192.168.3.162)</td>\n                <td>8× RX 6600</td>\n                <td>ROCm</td>\n                <td>Worker Node 3</td>\n                </tr>
    }' web_dashboard.py
    verify_patch web_dashboard.py
fi

# -----------------------------------------------
# 14. step3_tarball_helpers.sh (line 134)
#     local rig_names=("rig-6600" "rig-6600b")
# -----------------------------------------------
echo "=== 14. step3_tarball_helpers.sh ==="
if backup_and_patch "step3_tarball_helpers.sh" "rig_names array"; then
    sed -i 's/rig_names=("rig-6600" "rig-6600b")/rig_names=("rig-6600" "rig-6600b" "rig-6600c")/' step3_tarball_helpers.sh
    verify_patch step3_tarball_helpers.sh
fi

# -----------------------------------------------
# 15. ramdisk_preload_fixed.sh (line 22 — comment only)
# -----------------------------------------------
echo "=== 15. ramdisk_preload_fixed.sh ==="
if backup_and_patch "ramdisk_preload_fixed.sh" "comment update"; then
    sed -i 's/#   - 192.168.3.154 (rig-6600b)/#   - 192.168.3.154 (rig-6600b)\n#   - 192.168.3.162 (rig-6600c)/' ramdisk_preload_fixed.sh
    verify_patch ramdisk_preload_fixed.sh
fi

echo ""
echo "=============================================="
echo "  PATCH SUMMARY"
echo "=============================================="
echo "  ✅ Patched: $PATCHED"
echo "  ⏭️  Skipped: $SKIPPED"
echo "  ❌ Failed:  $FAILED"
echo "  Backups:   .bak_${TIMESTAMP}"
echo "=============================================="
echo ""

if [ "$FAILED" -gt 0 ]; then
    echo "⚠️  Some patches failed! Review manually."
    exit 1
else
    echo "All patches applied successfully."
    echo ""
    echo "Verify with:"
    echo "  grep -rn 'rig-6600b' --include='*.py' --include='*.sh' | grep -v .bak_ | grep -v __pycache__ | grep -v rig-6600c | grep -v full_scoring_results | grep -v results_ | grep -v backup | grep -v _TEST_ | grep -v _backup_"
fi
