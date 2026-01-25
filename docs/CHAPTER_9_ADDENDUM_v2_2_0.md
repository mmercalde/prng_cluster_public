# CHAPTER 9 ADDENDUM: Diagnostic Tools & Ramdisk v2.1.0

**Version:** 2.2.0  
**Date:** January 25, 2026  
**New Files:** `debug_gpu_failures.sh`, `ramdisk_preload.sh` v2.1.0

---

## INSERT INTO SECTION 8 (Operations)

---

## 8.6 GPU Diagnostic Battery (v2.2.0)

### 8.6.1 Overview

The `debug_gpu_failures.sh` script provides comprehensive cluster diagnostics for troubleshooting GPU failures.

**File:** `debug_gpu_failures.sh`

**Usage:**
```bash
cd ~/distributed_prng_analysis
bash debug_gpu_failures.sh
# Output: debug_gpu_failures_YYYYMMDD_HHMMSS.log
```

### 8.6.2 Checks Performed

| Check | Command | Looking For |
|-------|---------|-------------|
| Memory status | `free -h` | Available RAM < 4GB = OOM risk |
| GPU health | `rocm-smi` | N/A, "unknown", or missing GPUs |
| OOM killer | `dmesg \| grep oom` | "Killed" or "Out of memory" messages |
| ROCm/HIP errors | `dmesg \| grep amdgpu` | Driver or HIP initialization failures |
| Ramdisk status | `ls /dev/shm/prng/step3/` | Missing files |
| HIP cache size | `du -sh ~/.cache/hip_*` | Large cache = memory pressure |

### 8.6.3 Interpreting Results

| Pattern in Output | Meaning | Action |
|-------------------|---------|--------|
| `Killed` in dmesg | OOM killer terminated process | Reduce `chunk_size` or concurrent workers |
| GPU shows `N/A` | GPU not responding | Reboot node |
| GPU shows `unknown` | SMU communication failure | Reboot node or reseat GPU |
| Ramdisk missing | Files not preloaded | Run `ramdisk_preload.sh` |
| HIP cache > 1GB | Memory pressure | Clear cache with `rm -rf ~/.cache/hip_*` |

### 8.6.4 Sample Output (Healthy Cluster)

```
[2026-01-25 11:15:07] ═══ MEMORY STATUS ═══
--- rig-6600 (192.168.3.120) ---
Mem:           7.7Gi       2.5Gi       4.0Gi       7.0Mi       1.1Gi       4.9Gi

[2026-01-25 11:15:07] ═══ GPU HEALTH (rocm-smi) ═══
--- rig-6600 (192.168.3.120) ---
Device  Temp    Power  SCLK    MCLK   Perf
0       44.0°C  4.0W   700Mhz  96Mhz  auto  ← All healthy
...
11      39.0°C  4.0W   700Mhz  96Mhz  auto
```

### 8.6.5 Sample Output (Problem Detected)

```
--- rig-6600b (192.168.3.154) ---
Device 10  Temp: N/A  Power: N/A  SCLK: N/A  Perf: unknown  ← DEAD GPU

Action: Reboot rig-6600b to recover GPU[10]
```

---

## 8.7 Ramdisk Preload v2.1.0 (Updated)

### 8.7.1 Bug Fix

**Issue:** v2.0.0 created `.ready` sentinel marker even when file copy failed, causing Step 3 to believe ramdisk was populated when files were missing.

**Root Cause:** 
```bash
# v2.0.0 BUG: .ready created regardless of copy success
scp "$f" "$NODE:$RAMDISK_DIR/" 2>/dev/null  # May fail silently
ssh "$NODE" "touch $RAMDISK_SENTINEL"       # Created anyway!
```

**Fix in v2.1.0:**
```bash
# v2.1.0 FIX: Only create .ready if ALL files copied
if [ $copied -eq $expected_count ]; then
    ssh "$NODE" "touch $RAMDISK_SENTINEL"
    echo "    ✓ Preloaded ($copied files)"
else
    echo "    ❌ INCOMPLETE: Only $copied/$expected_count files copied"
    ssh "$NODE" "rm -f $RAMDISK_SENTINEL"  # Remove stale sentinel
fi
```

### 8.7.2 New Features in v2.1.0

| Feature | v2.0.0 | v2.1.0 |
|---------|--------|--------|
| Source file validation | Silent failure | Fails loudly before copying |
| Copy count verification | Not checked | All files must copy successfully |
| Stale sentinel cleanup | Not removed | Removed on incomplete copy |
| Error messaging | Unclear | Explicit "INCOMPLETE" message |

### 8.7.3 Updated Preload Behavior

```
[11:16:30] ════════════════════════════════════════════════════════════════
[11:16:30] Ramdisk Preload - Step 3
[11:16:30] ════════════════════════════════════════════════════════════════
[11:16:30] Files to preload: train_history.json holdout_history.json ...
[11:16:30] 
[11:16:30] --- localhost ---
[11:16:30]   Missing on localhost: /dev/shm/prng/step3/train_history.json
[11:16:30]   Populating ramdisk...
[11:16:30]   ✓ Preloaded (4 files)    ← Only if ALL 4 copied
[11:16:30] 
[11:16:30] --- 192.168.3.120 ---
[11:16:30]   ✓ Already loaded (verified)  ← Checks actual files, not just .ready
```

### 8.7.4 Standalone Preload Script

For manual ramdisk population (e.g., after node reboot):

**File:** `ramdisk_preload_fixed.sh`

```bash
# Usage
bash ramdisk_preload_fixed.sh 3  # Preload for Step 3

# Output
[11:16:32] Final verification:
[11:16:32]   ✅ localhost: All files present
[11:16:33]   ✅ 192.168.3.120: All files present
[11:16:33]   ✅ 192.168.3.154: All files present
```

---

## Version History Update

| Version | Date | Changes |
|---------|------|---------|
| 2.2.0 | 2026-01-25 | Added `debug_gpu_failures.sh`, ramdisk v2.1.0 |
| 2.1.0 | 2026-01-24 | Memory-safe configuration, OOM fixes |
| 2.0.0 | 2026-01-22 | Ramdisk deployment (Section 8.5) |
| 1.0.0 | 2025-12-01 | Initial infrastructure documentation |
