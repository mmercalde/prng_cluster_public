#!/usr/bin/env bash
# ============================================================================
# fix_gpu_rigs.sh — Deploy permanent GPU fixes to remote mining rigs
#
# Fixes:
#   1. udev rule: auto-set perf=high when GPUs are detected
#   2. GRUB: disable GFXOFF (prevents soft lockup crashes)
#
# Usage:
#   bash fix_gpu_rigs.sh              # Deploy to all rigs
#   bash fix_gpu_rigs.sh 192.168.3.154  # Deploy to one rig
#
# Requires: SSH access + sudo on each rig
# Reboot required after for GFXOFF change to take effect
# ============================================================================

set -euo pipefail

# Default rigs
ALL_RIGS=("192.168.3.120" "192.168.3.154" "192.168.3.162")

# If specific rig passed as argument, use only that
if [[ $# -gt 0 ]]; then
    RIGS=("$@")
else
    RIGS=("${ALL_RIGS[@]}")
fi

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✅ $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

for RIG in "${RIGS[@]}"; do
    echo ""
    echo "============================================================"
    echo "  Deploying GPU fixes to: $RIG"
    echo "============================================================"

    # Test SSH connectivity
    if ! ssh -o ConnectTimeout=5 "$RIG" "echo ok" &>/dev/null; then
        fail "$RIG — SSH unreachable, skipping"
        continue
    fi

    ssh -t "$RIG" bash -s << 'REMOTE_SCRIPT'
        set -euo pipefail

        GREEN='\033[0;32m'
        RED='\033[0;31m'
        YELLOW='\033[1;33m'
        NC='\033[0m'

        echo ""
        echo "--- Fix 1: udev rule (perf=high on GPU detect) ---"
        UDEV_FILE="/etc/udev/rules.d/99-amdgpu-perf.rules"
        UDEV_RULE='ACTION=="add", SUBSYSTEM=="drm", DRIVERS=="amdgpu", ATTR{device/power_dpm_force_performance_level}="high"'

        if [ -f "$UDEV_FILE" ] && grep -q "power_dpm_force_performance_level" "$UDEV_FILE"; then
            echo -e "${YELLOW}⚠️  udev rule already exists — skipping${NC}"
        else
            echo "$UDEV_RULE" | sudo tee "$UDEV_FILE" > /dev/null
            sudo udevadm control --reload-rules
            echo -e "${GREEN}✅ udev rule deployed${NC}"
        fi

        echo ""
        echo "--- Fix 2: GFXOFF disable (kernel parameter) ---"
        if grep -q "amdgpu.gfxoff=0" /proc/cmdline; then
            echo -e "${GREEN}✅ GFXOFF already disabled in running kernel${NC}"
        elif grep -q "amdgpu.gfxoff=0" /etc/default/grub; then
            echo -e "${YELLOW}⚠️  GFXOFF in GRUB but not active — needs reboot${NC}"
        else
            # Add to GRUB
            sudo cp /etc/default/grub /etc/default/grub.bak
            if grep -q 'GRUB_CMDLINE_LINUX_DEFAULT=".*"' /etc/default/grub; then
                sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 amdgpu.gfxoff=0"/' /etc/default/grub
            else
                echo 'GRUB_CMDLINE_LINUX_DEFAULT="amdgpu.gfxoff=0"' | sudo tee -a /etc/default/grub > /dev/null
            fi
            sudo update-grub 2>/dev/null || sudo grub-mkconfig -o /boot/grub/grub.cfg 2>/dev/null
            echo -e "${GREEN}✅ GFXOFF added to GRUB — reboot required${NC}"
        fi

        echo ""
        echo "--- Fix 3: Set perf=high now (immediate) ---"
        FIXED=0
        for f in /sys/class/drm/card*/device/power_dpm_force_performance_level; do
            CURRENT=$(cat "$f" 2>/dev/null || echo "unknown")
            if [ "$CURRENT" != "high" ]; then
                echo high | sudo tee "$f" > /dev/null 2>&1 && FIXED=$((FIXED+1))
            fi
        done
        COUNT=$(cat /sys/class/drm/card*/device/power_dpm_force_performance_level 2>/dev/null | grep -c high || echo 0)
        echo -e "${GREEN}✅ $COUNT GPUs now at perf=high ($FIXED changed)${NC}"

        echo ""
        echo "--- Verification ---"
        echo "  Kernel GFXOFF: $(grep -q 'amdgpu.gfxoff=0' /proc/cmdline && echo 'ACTIVE' || echo 'PENDING REBOOT')"
        echo "  udev rule:     $([ -f /etc/udev/rules.d/99-amdgpu-perf.rules ] && echo 'INSTALLED' || echo 'MISSING')"
        echo "  GPU perf:      $(cat /sys/class/drm/card*/device/power_dpm_force_performance_level 2>/dev/null | sort | uniq -c | tr '\n' ' ')"
REMOTE_SCRIPT

    ok "$RIG — complete"
done

echo ""
echo "============================================================"
echo "  DONE — Reboot rigs for GFXOFF to take effect"
echo "  udev rule + perf=high are active immediately"
echo "============================================================"
