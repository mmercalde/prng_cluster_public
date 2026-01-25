#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# GPU Failure Debug Script for rig-6600
# ════════════════════════════════════════════════════════════════════════════════
# Version: 1.0.0
# Date: January 25, 2026
#
# Purpose: Capture detailed error information from GPU failures on rig-6600
# before the retry mechanism masks the error.
#
# Usage (from Zeus):
#   bash debug_gpu_failures.sh
#
# Output:
#   debug_gpu_failures_YYYYMMDD_HHMMSS.log
#
# ════════════════════════════════════════════════════════════════════════════════

set -euo pipefail

LOG_FILE="debug_gpu_failures_$(date +%Y%m%d_%H%M%S).log"
RIG_6600="192.168.3.120"
RIG_6600B="192.168.3.154"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "═══════════════════════════════════════════════════════════════════════════════"
log "GPU Failure Debug Script - Starting"
log "═══════════════════════════════════════════════════════════════════════════════"

# ────────────────────────────────────────────────────────────────────────────────
# 1. Check memory status on both rigs
# ────────────────────────────────────────────────────────────────────────────────
log ""
log "═══ MEMORY STATUS ═══"

log ""
log "--- rig-6600 ($RIG_6600) ---"
ssh "$RIG_6600" "free -h" 2>&1 | tee -a "$LOG_FILE" || log "ERROR: Could not check rig-6600 memory"

log ""
log "--- rig-6600b ($RIG_6600B) ---"
ssh "$RIG_6600B" "free -h" 2>&1 | tee -a "$LOG_FILE" || log "ERROR: Could not check rig-6600b memory"

# ────────────────────────────────────────────────────────────────────────────────
# 2. Check GPU health via rocm-smi
# ────────────────────────────────────────────────────────────────────────────────
log ""
log "═══ GPU HEALTH (rocm-smi) ═══"

log ""
log "--- rig-6600 ($RIG_6600) ---"
ssh "$RIG_6600" "source ~/rocm_env/bin/activate && rocm-smi" 2>&1 | tee -a "$LOG_FILE" || log "ERROR: Could not check rig-6600 GPUs"

log ""
log "--- rig-6600b ($RIG_6600B) ---"
ssh "$RIG_6600B" "source ~/rocm_env/bin/activate && rocm-smi" 2>&1 | tee -a "$LOG_FILE" || log "ERROR: Could not check rig-6600b GPUs"

# ────────────────────────────────────────────────────────────────────────────────
# 3. Check for OOM killer activity in kernel logs
# ────────────────────────────────────────────────────────────────────────────────
log ""
log "═══ OOM KILLER ACTIVITY (last 50 lines) ═══"

log ""
log "--- rig-6600 ($RIG_6600) ---"
ssh "$RIG_6600" "dmesg | grep -iE 'oom|kill|memory' | tail -20" 2>&1 | tee -a "$LOG_FILE" || log "No OOM activity found"

log ""
log "--- rig-6600b ($RIG_6600B) ---"
ssh "$RIG_6600B" "dmesg | grep -iE 'oom|kill|memory' | tail -20" 2>&1 | tee -a "$LOG_FILE" || log "No OOM activity found"

# ────────────────────────────────────────────────────────────────────────────────
# 4. Check ROCm/HIP specific errors
# ────────────────────────────────────────────────────────────────────────────────
log ""
log "═══ ROCM/HIP ERRORS ═══"

log ""
log "--- rig-6600 ($RIG_6600) ---"
ssh "$RIG_6600" "dmesg | grep -iE 'amdgpu|rocm|hip|hsa' | tail -20" 2>&1 | tee -a "$LOG_FILE" || log "No ROCm errors found"

log ""
log "--- rig-6600b ($RIG_6600B) ---"
ssh "$RIG_6600B" "dmesg | grep -iE 'amdgpu|rocm|hip|hsa' | tail -20" 2>&1 | tee -a "$LOG_FILE" || log "No ROCm errors found"

# ────────────────────────────────────────────────────────────────────────────────
# 5. Check ramdisk status
# ────────────────────────────────────────────────────────────────────────────────
log ""
log "═══ RAMDISK STATUS ═══"

log ""
log "--- localhost ---"
ls -la /dev/shm/prng/step3/ 2>&1 | tee -a "$LOG_FILE" || log "Ramdisk not found on localhost"

log ""
log "--- rig-6600 ($RIG_6600) ---"
ssh "$RIG_6600" "ls -la /dev/shm/prng/step3/" 2>&1 | tee -a "$LOG_FILE" || log "Ramdisk not found on rig-6600"

log ""
log "--- rig-6600b ($RIG_6600B) ---"
ssh "$RIG_6600B" "ls -la /dev/shm/prng/step3/" 2>&1 | tee -a "$LOG_FILE" || log "Ramdisk not found on rig-6600b"

# ────────────────────────────────────────────────────────────────────────────────
# 6. Check HIP cache size
# ────────────────────────────────────────────────────────────────────────────────
log ""
log "═══ HIP CACHE SIZE ═══"

log ""
log "--- rig-6600 ($RIG_6600) ---"
ssh "$RIG_6600" "du -sh ~/.cache/hip_* ~/.cache/cupy 2>/dev/null || echo 'No HIP cache found'" 2>&1 | tee -a "$LOG_FILE"

log ""
log "--- rig-6600b ($RIG_6600B) ---"
ssh "$RIG_6600B" "du -sh ~/.cache/hip_* ~/.cache/cupy 2>/dev/null || echo 'No HIP cache found'" 2>&1 | tee -a "$LOG_FILE"

# ────────────────────────────────────────────────────────────────────────────────
# 7. GPU-specific tests for GPU2 and GPU4 on rig-6600
# ────────────────────────────────────────────────────────────────────────────────
log ""
log "═══ GPU2 AND GPU4 SPECIFIC TESTS (rig-6600) ═══"

log ""
log "Testing GPU2 visibility..."
ssh "$RIG_6600" "source ~/rocm_env/bin/activate && \
    HIP_VISIBLE_DEVICES=2 python3 -c 'import torch; print(f\"GPU2: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT AVAILABLE\"}\")'" \
    2>&1 | tee -a "$LOG_FILE" || log "ERROR: GPU2 test failed"

log ""
log "Testing GPU4 visibility..."
ssh "$RIG_6600" "source ~/rocm_env/bin/activate && \
    HIP_VISIBLE_DEVICES=4 python3 -c 'import torch; print(f\"GPU4: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT AVAILABLE\"}\")'" \
    2>&1 | tee -a "$LOG_FILE" || log "ERROR: GPU4 test failed"

# ────────────────────────────────────────────────────────────────────────────────
# 8. Summary
# ────────────────────────────────────────────────────────────────────────────────
log ""
log "═══════════════════════════════════════════════════════════════════════════════"
log "Debug script complete. Output saved to: $LOG_FILE"
log "═══════════════════════════════════════════════════════════════════════════════"

echo ""
echo "Quick summary:"
echo "  - Memory: Check if available RAM < 4GB on either rig"
echo "  - GPUs: Look for N/A or 'unknown' in rocm-smi output"
echo "  - OOM: Any 'Killed' or 'Out of memory' in dmesg"
echo "  - Ramdisk: Should have 2 files (train_history.json, holdout_history.json)"
echo "  - HIP cache: Large cache size (>1GB) may indicate memory pressure"
echo ""
echo "Full details in: $LOG_FILE"
