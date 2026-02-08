#!/usr/bin/env bash
# ============================================================================
# run_soak_c.sh — Soak Test C: Full Autonomous Loop
# Version: 1.0.0
# Date: 2026-02-06
#
# Performs all pre-flight checks, backups, bootstrap, and launches daemons.
# Uses tmux for daemon management so terminals persist if SSH drops.
#
# Usage:
#   ./run_soak_c.sh              # Full pre-flight + launch
#   ./run_soak_c.sh --status     # Check running soak status
#   ./run_soak_c.sh --stop       # Stop daemons + show results
#   ./run_soak_c.sh --cleanup    # Restore production configs
# ============================================================================

set -euo pipefail

WORKDIR="$HOME/distributed_prng_analysis"
TMUX_SESSION="soakc"
LOG_DIR="$WORKDIR/logs/soak"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/soakC_${TIMESTAMP}.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

ok()   { echo -e "  ${GREEN}✅ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
fail() { echo -e "  ${RED}❌ $1${NC}"; }
info() { echo -e "  ${CYAN}ℹ️  $1${NC}"; }
header() { echo -e "\n${CYAN}═══ $1 ═══${NC}"; }

cd "$WORKDIR"

# ============================================================================
# --status: Check running soak test
# ============================================================================
if [[ "${1:-}" == "--status" ]]; then
    header "SOAK C STATUS"

    # Check if tmux session exists
    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        ok "tmux session '$TMUX_SESSION' is running"
        echo ""
        tmux list-windows -t "$TMUX_SESSION" 2>/dev/null | sed 's/^/    /'
    else
        fail "tmux session '$TMUX_SESSION' not found — soak not running"
        exit 1
    fi

    echo ""

    # Find the most recent soak log
    LATEST_LOG=$(ls -t "$LOG_DIR"/soakC_*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        echo -e "  Log: $LATEST_LOG"
        echo ""
        CYCLES=$(grep -c 'CHAPTER 13 CYCLE' "$LATEST_LOG" 2>/dev/null || true)
        AUTO=$(grep -c 'Auto-approv' "$LATEST_LOG" 2>/dev/null || true)
        ESCALATED=$(grep -c 'ESCALATE' "$LATEST_LOG" 2>/dev/null || true)
        ERRORS=$(grep -c 'Traceback' "$LATEST_LOG" 2>/dev/null || true)

        echo "  Cycles completed:  $CYCLES"
        echo "  Auto-approved:     $AUTO"
        echo "  Escalated:         $ESCALATED"
        echo "  Errors:            $ERRORS"
        echo ""

        # Pass/fail assessment
        if [[ "$CYCLES" -gt 10 && "$ESCALATED" -eq 0 ]]; then
            ok "PASS CRITERIA MET (Cycles > 10, Escalated = 0)"
        elif [[ "$CYCLES" -gt 0 && "$ESCALATED" -eq 0 ]]; then
            info "On track — $CYCLES cycles so far, no escalations"
        elif [[ "$ESCALATED" -gt 0 ]]; then
            fail "ESCALATIONS DETECTED — patches may not be working"
        else
            info "No cycles yet — still warming up"
        fi

        echo ""
        echo "  Last 5 log lines:"
        tail -5 "$LATEST_LOG" | sed 's/^/    /'
    else
        warn "No soak log files found in $LOG_DIR"
    fi
    exit 0
fi

# ============================================================================
# --stop: Stop daemons and show results
# ============================================================================
if [[ "${1:-}" == "--stop" ]]; then
    header "STOPPING SOAK C"

    if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
        tmux kill-session -t "$TMUX_SESSION"
        ok "tmux session '$TMUX_SESSION' killed"
    else
        warn "tmux session '$TMUX_SESSION' not found"
    fi

    # Also kill any stragglers
    pkill -f "synthetic_draw_injector.py" 2>/dev/null && info "Killed injector process" || true
    pkill -f "chapter_13_orchestrator.py" 2>/dev/null && info "Killed orchestrator process" || true

    echo ""
    header "FINAL RESULTS"

    LATEST_LOG=$(ls -t "$LOG_DIR"/soakC_*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        CYCLES=$(grep -c 'CHAPTER 13 CYCLE' "$LATEST_LOG" 2>/dev/null || true)
        AUTO=$(grep -c 'Auto-approv' "$LATEST_LOG" 2>/dev/null || true)
        ESCALATED=$(grep -c 'ESCALATE' "$LATEST_LOG" 2>/dev/null || true)
        ERRORS=$(grep -c 'Traceback' "$LATEST_LOG" 2>/dev/null || true)

        echo ""
        echo "  Log file:          $LATEST_LOG"
        echo "  Cycles completed:  $CYCLES"
        echo "  Auto-approved:     $AUTO"
        echo "  Escalated:         $ESCALATED"
        echo "  Errors:            $ERRORS"
        echo ""

        if [[ "$CYCLES" -gt 10 && "$ESCALATED" -eq 0 && "$ERRORS" -lt 5 ]]; then
            echo -e "  ${GREEN}╔══════════════════════════════════╗${NC}"
            echo -e "  ${GREEN}║   SOAK C: PASSED ✅              ║${NC}"
            echo -e "  ${GREEN}║   Phase 7 ready for certification ║${NC}"
            echo -e "  ${GREEN}╚══════════════════════════════════╝${NC}"
        elif [[ "$ESCALATED" -gt 0 ]]; then
            echo -e "  ${RED}╔══════════════════════════════════╗${NC}"
            echo -e "  ${RED}║   SOAK C: FAILED ❌              ║${NC}"
            echo -e "  ${RED}║   Escalations detected           ║${NC}"
            echo -e "  ${RED}╚══════════════════════════════════╝${NC}"
        else
            echo -e "  ${YELLOW}╔══════════════════════════════════╗${NC}"
            echo -e "  ${YELLOW}║   SOAK C: INCONCLUSIVE ⚠️        ║${NC}"
            echo -e "  ${YELLOW}║   Cycles: $CYCLES (need >10)          ║${NC}"
            echo -e "  ${YELLOW}╚══════════════════════════════════╝${NC}"
        fi
    else
        warn "No soak log files found"
    fi

    echo ""
    info "Run './run_soak_c.sh --cleanup' to restore production configs"
    exit 0
fi

# ============================================================================
# --cleanup: Restore production configs
# ============================================================================
if [[ "${1:-}" == "--cleanup" ]]; then
    header "RESTORING PRODUCTION CONFIGS"

    for f in lottery_history.json optimal_window_config.json watcher_policies.json; do
        backup="${f}.pre_soakC"
        if [[ -f "$backup" ]]; then
            cp "$backup" "$f"
            ok "Restored $f from $backup"
        else
            warn "No backup found: $backup"
        fi
    done

    echo ""

    # Force production mode regardless of backup state
    python3 -c "
import json
with open('watcher_policies.json') as f: p = json.load(f)
p['test_mode'] = False
p['auto_approve_in_test_mode'] = False
p['skip_escalation_in_test_mode'] = False
with open('watcher_policies.json','w') as f: json.dump(p,f,indent=2)
"
    ok "Forced test_mode=false in watcher_policies.json"
    
    # Clean stale pipeline files and halt flags
    rm -f /tmp/agent_halt watcher_halt.flag new_draw.flag pending_approval.json
    ok "Cleared halt/lock/flag files"
    
    echo ""
    # Optionally revert code patches
    read -p "  Revert code patches? (y/N): " REVERT
    if [[ "${REVERT,,}" == "y" ]]; then
        python3 patch_soak_c_integration_v1.py --revert
        ok "Code patches reverted"
    else
        info "Keeping code patches (safe for production — only active in test_mode)"
    fi

    rm -f pending_approval.json new_draw.flag
    ok "Cleaned up state files"

    echo ""
    ok "Production configs restored. System ready for normal operation."
    exit 0
fi

# ============================================================================
# MAIN: Pre-flight + Launch
# ============================================================================

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║      SOAK TEST C — Full Autonomous Loop         ║${NC}"
echo -e "${CYAN}║      Pre-flight Checks + Launch                  ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"

# --- Check if already running ---
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    fail "Soak C appears to already be running (tmux session '$TMUX_SESSION' exists)"
    info "Use './run_soak_c.sh --status' to check or '--stop' to halt"
    exit 1
fi

# ============================================================================
# STEP 1: Verify patches
# ============================================================================
header "STEP 1: Verify Patches"

PATCH_OUTPUT=$(python3 patch_soak_c_integration_v1.py --check 2>&1)
echo "$PATCH_OUTPUT" | sed 's/^/  /'

if echo "$PATCH_OUTPUT" | grep -q "Not applied"; then
    warn "Patches not applied — applying now..."
    python3 patch_soak_c_integration_v1.py --apply
    ok "Patches applied"
elif echo "$PATCH_OUTPUT" | grep -q "Unknown state"; then
    warn "Patch state unclear — re-applying to be safe..."
    python3 patch_soak_c_integration_v1.py --apply
fi

# ============================================================================
# STEP 2: Verify policy flags
# ============================================================================
header "STEP 2: Verify Policy Flags"

POLICY_CHECK=$(python3 << 'PYCHECK'
import json, sys

with open("watcher_policies.json") as f:
    p = json.load(f)

flags = {
    "test_mode": p.get("test_mode", False),
    "auto_approve_in_test_mode": p.get("auto_approve_in_test_mode", False),
    "skip_escalation_in_test_mode": p.get("skip_escalation_in_test_mode", False),
}

all_good = True
for flag, val in flags.items():
    status = "✅" if val else "❌"
    print(f"  {status} {flag}: {val}")
    if not val:
        all_good = False

if not all_good:
    # Fix missing flags
    p["test_mode"] = True
    p["auto_approve_in_test_mode"] = True
    p["skip_escalation_in_test_mode"] = True
    with open("watcher_policies.json", "w") as f:
        json.dump(p, f, indent=2)
    print("  ⚠️  Fixed — flags set to true")
    sys.exit(1)
else:
    sys.exit(0)
PYCHECK
)
echo "$POLICY_CHECK"

if [[ $? -ne 0 ]]; then
    ok "Policy flags auto-corrected"
fi

# ============================================================================
# STEP 3: Backup production data
# ============================================================================
header "STEP 3: Backup Production Data"

for f in lottery_history.json optimal_window_config.json watcher_policies.json; do
    backup="${f}.pre_soakC"
    if [[ -f "$f" ]]; then
        if [[ -f "$backup" ]]; then
            info "$backup already exists — keeping existing backup"
        else
            cp "$f" "$backup"
            ok "Backed up $f → $backup"
        fi
    else
        warn "$f not found — will be created by bootstrap"
    fi
done

# ============================================================================
# STEP 4: Clean state
# ============================================================================
header "STEP 4: Clean State"

rm -f pending_approval.json new_draw.flag
ok "Removed pending_approval.json and new_draw.flag"

# Check for stale lock files
STALE_LOCKS=$(find . -name "*.lock" -mmin +60 2>/dev/null | wc -l)
if [[ "$STALE_LOCKS" -gt 0 ]]; then
    find . -name "*.lock" -mmin +60 -delete
    warn "Removed $STALE_LOCKS stale lock file(s)"
else
    ok "No stale lock files"
fi

# ============================================================================
# STEP 5: Bootstrap synthetic history
# ============================================================================
header "STEP 5: Bootstrap Synthetic History"

python3 << 'BOOTSTRAP'
import json
from datetime import datetime, timedelta
from prng_registry import get_cpu_reference

TRUE_SEED = 12345
PRNG_TYPE = "java_lcg"
NUM_DRAWS = 5000
MOD = 1000

prng = get_cpu_reference(PRNG_TYPE)
raw = prng(TRUE_SEED, NUM_DRAWS)
vals = [v % MOD for v in raw]

draws = []
date = datetime(2024, 1, 1)

for i, v in enumerate(vals):
    digits = [(v // 100) % 10, (v // 10) % 10, v % 10]
    session = "midday" if i % 2 == 0 else "evening"
    draws.append({
        "date": date.strftime("%Y-%m-%d"),
        "session": session,
        "draw": digits,
        "value": v,
        "draw_source": "synthetic_bootstrap",
        "true_seed": TRUE_SEED
    })
    if session == "evening":
        date += timedelta(days=1)

with open("lottery_history.json", "w") as f:
    json.dump({"draws": draws}, f, indent=2)

with open("optimal_window_config.json", "w") as f:
    json.dump({
        "prng_type": PRNG_TYPE,
        "mod": MOD,
        "bootstrap": True
    }, f, indent=2)

print(f"  ✅ Bootstrap complete: {len(draws)} synthetic draws (seed {TRUE_SEED}, {PRNG_TYPE})")
BOOTSTRAP

# ============================================================================
# STEP 6: Pre-launch snapshot
# ============================================================================
header "STEP 6: Pre-Launch Snapshot"

DRAW_COUNT=$(python3 -c "import json; d=json.load(open('lottery_history.json')); print(len(d['draws']))")
POLICY_COUNT=$(ls policy_history/ 2>/dev/null | wc -l)
DISK_FREE=$(df -h /home/michael --output=avail | tail -1 | tr -d ' ')

echo "  Draws in history:    $DRAW_COUNT"
echo "  Policy candidates:   $POLICY_COUNT"
echo "  Disk available:      $DISK_FREE"
date > "$LOG_DIR/soakC_start_timestamp.txt" 2>/dev/null || true

# ============================================================================
# STEP 7: Launch daemons in tmux
# ============================================================================
header "STEP 7: Launch Daemons"

mkdir -p "$LOG_DIR"

# Create tmux session with injector in first window
tmux new-session -d -s "$TMUX_SESSION" -n "injector" \
    "cd $WORKDIR && PYTHONPATH=. python3 synthetic_draw_injector.py --daemon --interval 60; echo 'INJECTOR EXITED - press enter'; read"

# Create orchestrator in second window
tmux new-window -t "$TMUX_SESSION" -n "orchestrator" \
    "cd $WORKDIR && PYTHONPATH=. python3 chapter_13_orchestrator.py --daemon --auto-start-llm 2>&1 | tee $LOG_DIR/soakC_${TIMESTAMP}.log; echo 'ORCHESTRATOR EXITED - press enter'; read"

# Create a status window
tmux new-window -t "$TMUX_SESSION" -n "monitor" \
    "cd $WORKDIR && echo 'Soak C Monitor — updates every 60s' && echo ''; while true; do echo \"[\$(date '+%H:%M:%S')] Cycles: \$(grep -c 'CHAPTER 13 CYCLE' $LOG_DIR/soakC_${TIMESTAMP}.log 2>/dev/null || true) | Auto-approved: \$(grep -c 'Auto-approv' $LOG_DIR/soakC_${TIMESTAMP}.log 2>/dev/null || true) | Escalated: \$(grep -c 'ESCALATE' $LOG_DIR/soakC_${TIMESTAMP}.log 2>/dev/null || true) | Errors: \$(grep -ci 'error' $LOG_DIR/soakC_${TIMESTAMP}.log 2>/dev/null || true)\"; sleep 60; done"

ok "tmux session '$TMUX_SESSION' created with 3 windows:"
echo "    0: injector    — synthetic_draw_injector.py (60s interval)"
echo "    1: orchestrator — chapter_13_orchestrator.py (logging to $LOG_DIR/)"
echo "    2: monitor     — live cycle counter (60s refresh)"

# ============================================================================
# DONE
# ============================================================================
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   SOAK C LAUNCHED — Let run 1-2 hours            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Attach to tmux:     tmux attach -t $TMUX_SESSION"
echo "  Check status:       ./run_soak_c.sh --status"
echo "  Stop + results:     ./run_soak_c.sh --stop"
echo "  Restore configs:    ./run_soak_c.sh --cleanup"
echo ""
echo "  Pass criteria: Cycles > 10, Escalated = 0, Errors < 5"
echo ""
info "Log file: $LOG_DIR/soakC_${TIMESTAMP}.log"
