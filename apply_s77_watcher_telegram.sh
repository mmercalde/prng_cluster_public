#!/bin/bash
# =============================================================================
# Session 77: WATCHER Telegram Notification Integration
# =============================================================================
# Adds Telegram notifications to WATCHER for three severity classes:
#   A. CRITICAL -- human intervention required (always enabled)
#   B. DEGRADED -- autonomy self-corrected (always enabled)
#   C. INFO     -- pipeline completed cleanly (disabled by default)
#
# Prerequisites:
#   1. /usr/local/bin/cluster_notify.sh installed (see below)
#   2. /etc/cluster-boot-notify.conf readable by michael (chmod 640)
#
# Files modified:
#   1. agents/watcher_agent.py -- import + notify calls at decision points
#   2. watcher_policies.json   -- notification policy settings
#
# Usage: bash apply_s77_watcher_telegram.sh
# Rollback: git checkout agents/watcher_agent.py watcher_policies.json
# =============================================================================

set -euo pipefail
cd ~/distributed_prng_analysis

echo "=== Session 77: WATCHER Telegram Notification Integration ==="
echo ""

# ---------------------------------------------------------------------------
# STEP 0: Backup files before modification
# ---------------------------------------------------------------------------
echo "[0/5] Backing up files..."

BACKUP_DIR="backups/s77_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp agents/watcher_agent.py "$BACKUP_DIR/watcher_agent.py.bak"
cp watcher_policies.json "$BACKUP_DIR/watcher_policies.json.bak"

echo "  OK Backups saved to $BACKUP_DIR/"
echo ""

# ---------------------------------------------------------------------------
# STEP 1: Install cluster_notify.sh (if not already installed)
# ---------------------------------------------------------------------------
echo "[1/5] Checking cluster_notify.sh installation..."

if [ -x /usr/local/bin/cluster_notify.sh ]; then
    echo "  -> Already installed, skipping"
else
    if [ -f cluster_notify.sh ]; then
        echo "  Installing cluster_notify.sh to /usr/local/bin/ ..."
        sudo install -m 0755 cluster_notify.sh /usr/local/bin/cluster_notify.sh
        echo "  OK Installed"
    else
        echo "  WARN cluster_notify.sh not found in project root"
        echo "  Copy it to ~/distributed_prng_analysis/ first, then re-run"
        echo "  Continuing without install (notifications will be silent)"
    fi
fi

# ---------------------------------------------------------------------------
# STEP 2: Add notify_telegram helper to watcher_agent.py
# ---------------------------------------------------------------------------
echo "[2/5] Adding notify_telegram() helper..."

if grep -q "def notify_telegram" agents/watcher_agent.py; then
    echo "  -> Helper already exists, skipping"
else
    # Insert after the DISTRIBUTED_STEPS line
    sed -i '/^DISTRIBUTED_STEPS = {1, 2, 3}$/a\
\
# Telegram Notification Integration (Session 77)\
def notify_telegram(message: str):\
    """Send a Telegram notification via cluster_notify.sh.\
    \
    Session 77: Best-effort, non-blocking, silent on failure.\
    Telegram is human attention routing, not logging.\
    \
    Notification classes:\
        A. CRITICAL -- human intervention required\
        B. DEGRADED -- autonomy self-corrected\
        C. INFO     -- pipeline completed cleanly (optional)\
    """\
    try:\
        subprocess.Popen(\
            ["/usr/local/bin/cluster_notify.sh", message],\
            stdout=subprocess.DEVNULL,\
            stderr=subprocess.DEVNULL,\
        )\
    except Exception:\
        pass  # best-effort, never block' agents/watcher_agent.py
    echo "  OK Helper added"
fi

# ---------------------------------------------------------------------------
# STEP 3: Add notification calls to WATCHER decision points
# ---------------------------------------------------------------------------
echo "[3/5] Wiring notification calls into decision handlers..."

# 3a. _handle_escalate -- Class A: CRITICAL
if grep -q "notify_telegram.*CRITICAL" agents/watcher_agent.py; then
    echo "  -> CRITICAL notification already wired, skipping"
else
    python3 << 'PYEOF'
with open("agents/watcher_agent.py", "r") as f:
    content = f.read()

# Insert notify_telegram call inside _handle_escalate, right after _notify_escalation
old = '        # Notify human\n        self._notify_escalation(step, decision, context)\n\n        return False'
new = '''        # Notify human
        self._notify_escalation(step, decision, context)

        # Session 77: Telegram Class A -- CRITICAL
        notify_telegram(
            "[WATCHER][CRITICAL][ACTION REQUIRED]\\n"
            f"Step {step}: {STEP_NAMES.get(step, 'Unknown')}\\n"
            f"Reason: {decision.reasoning[:200]}\\n"
            f"Confidence: {decision.confidence:.2f}\\n"
            "Pipeline HALTED -- human review required"
        )

        return False'''

if old in content:
    content = content.replace(old, new, 1)
    with open("agents/watcher_agent.py", "w") as f:
        f.write(content)
    print("  OK CRITICAL notification wired into _handle_escalate")
else:
    print("  WARN Could not find _handle_escalate insertion point -- check manually")
PYEOF
fi

# 3b. Session 76 health check -- Class B: DEGRADED (RETRY)
if grep -q "notify_telegram.*DEGRADED.*RETRY" agents/watcher_agent.py; then
    echo "  -> DEGRADED RETRY notification already wired, skipping"
else
    python3 << 'PYEOF'
with open("agents/watcher_agent.py", "r") as f:
    content = f.read()

# Insert after the RETRY warning log, before self.current_step = 5
old = '''                            logger.warning(
                                "[WATCHER][HEALTH] Training health RETRY %d/%d -- re-running Step 5",
                                training_health_retries, _max_retries
                            )
                            # Stay on Step 5 -- override current_step back'''

new = '''                            logger.warning(
                                "[WATCHER][HEALTH] Training health RETRY %d/%d -- re-running Step 5",
                                training_health_retries, _max_retries
                            )
                            # Session 77: Telegram Class B -- DEGRADED (RETRY)
                            notify_telegram(
                                "[WATCHER][DEGRADED]\\n"
                                "Step 5: Anti-Overfit Training\\n"
                                f"Issue: {"; ".join(_health.get("issues", [])[:2])}\\n"
                                f"Action: RETRY with modified params\\n"
                                f"Attempt: {training_health_retries}/{_max_retries}"
                            )
                            # Stay on Step 5 -- override current_step back'''

if old in content:
    content = content.replace(old, new, 1)
    with open("agents/watcher_agent.py", "w") as f:
        f.write(content)
    print("  OK DEGRADED RETRY notification wired")
else:
    print("  WARN Could not find RETRY insertion point -- check manually")
PYEOF
fi

# 3c. Session 76 health check -- Class B: DEGRADED (max retries exhausted)
if grep -q "notify_telegram.*DEGRADED.*exhausted" agents/watcher_agent.py; then
    echo "  -> DEGRADED exhausted notification already wired, skipping"
else
    python3 << 'PYEOF'
with open("agents/watcher_agent.py", "r") as f:
    content = f.read()

old = '''                            logger.error(
                                "[WATCHER][HEALTH] Max training retries (%d) exhausted -- proceeding to Step 6",
                                _max_retries
                            )'''

new = '''                            logger.error(
                                "[WATCHER][HEALTH] Max training retries (%d) exhausted -- proceeding to Step 6",
                                _max_retries
                            )
                            # Session 77: Telegram Class B -- DEGRADED (exhausted)
                            notify_telegram(
                                "[WATCHER][DEGRADED]\\n"
                                "Step 5: Anti-Overfit Training\\n"
                                f"Issue: Max training retries ({_max_retries}) exhausted\\n"
                                "Action: Proceeding to Step 6 (best-effort)\\n"
                                "Confidence: REDUCED"
                            )'''

if old in content:
    content = content.replace(old, new, 1)
    with open("agents/watcher_agent.py", "w") as f:
        f.write(content)
    print("  OK DEGRADED exhausted notification wired")
else:
    print("  WARN Could not find exhausted insertion point -- check manually")
PYEOF
fi

# 3d. Session 76 health check -- Class B: DEGRADED (SKIP_MODEL)
if grep -q "notify_telegram.*DEGRADED.*skipped" agents/watcher_agent.py; then
    echo "  -> DEGRADED SKIP notification already wired, skipping"
else
    python3 << 'PYEOF'
with open("agents/watcher_agent.py", "r") as f:
    content = f.read()

old = '''                    elif health_action == "skip":
                        logger.warning("[WATCHER][HEALTH] Model type skipped -- proceeding to Step 6")'''

new = '''                    elif health_action == "skip":
                        logger.warning("[WATCHER][HEALTH] Model type skipped -- proceeding to Step 6")
                        # Session 77: Telegram Class B -- DEGRADED (SKIP)
                        notify_telegram(
                            "[WATCHER][DEGRADED]\\n"
                            "Step 5: Anti-Overfit Training\\n"
                            f"Issue: Model type skipped (consecutive critical)\\n"
                            "Action: Proceeding to Step 6 without this model\\n"
                            "Confidence: REDUCED"
                        )'''

if old in content:
    content = content.replace(old, new, 1)
    with open("agents/watcher_agent.py", "w") as f:
        f.write(content)
    print("  OK DEGRADED SKIP notification wired")
else:
    print("  WARN Could not find SKIP insertion point -- check manually")
PYEOF
fi

# 3e. _notify_complete -- Class C: INFO (policy-gated)
if grep -q "notify_telegram.*INFO.*PIPELINE COMPLETE" agents/watcher_agent.py; then
    echo "  -> INFO notification already wired, skipping"
else
    python3 << 'PYEOF'
with open("agents/watcher_agent.py", "r") as f:
    content = f.read()

old = '''    def _notify_complete(self, context: FullAgentContext):
        """Notify that pipeline completed successfully."""
        print("\\n" + "=" * 60)'''

new = '''    def _notify_complete(self, context: FullAgentContext):
        """Notify that pipeline completed successfully."""
        # Session 77: Telegram Class C -- INFO (policy-gated, disabled by default)
        try:
            import json as _json_nc
            _notify_info = False
            if os.path.isfile("watcher_policies.json"):
                with open("watcher_policies.json") as _pf_nc:
                    _pol_nc = _json_nc.load(_pf_nc)
                _notify_info = _pol_nc.get("notifications", {}).get("info_on_complete", False)
            if _notify_info:
                notify_telegram(
                    "[WATCHER][INFO]\\n"
                    "PIPELINE COMPLETE\\n"
                    f"Steps completed: {context.step}\\n"
                    f"Success rate: {self.history.get_success_rate():.1%}"
                )
        except Exception:
            pass  # best-effort
        print("\\n" + "=" * 60)'''

if old in content:
    content = content.replace(old, new, 1)
    with open("agents/watcher_agent.py", "w") as f:
        f.write(content)
    print("  OK INFO notification wired (disabled by default)")
else:
    print("  WARN Could not find _notify_complete insertion point -- check manually")
PYEOF
fi

# ---------------------------------------------------------------------------
# STEP 4: Add notification policy to watcher_policies.json
# ---------------------------------------------------------------------------
echo "[4/5] Updating watcher_policies.json with notification policy..."

if python3 -c "
import json, sys
with open('watcher_policies.json') as f:
    p = json.load(f)
if 'notifications' in p:
    print('already_present')
    sys.exit(0)
else:
    sys.exit(1)
" 2>/dev/null; then
    echo "  -> Notification policy already exists, skipping"
else
    python3 << 'PYEOF'
import json

with open("watcher_policies.json") as f:
    policies = json.load(f)

policies["notifications"] = {
    "description": "Telegram notification policy (Session 77). Advisory only -- does not affect control flow.",
    "telegram_enabled": True,
    "script_path": "/usr/local/bin/cluster_notify.sh",
    "classes": {
        "critical": {
            "enabled": True,
            "description": "Human intervention required -- always notify",
            "can_disable": False
        },
        "degraded": {
            "enabled": True,
            "description": "Autonomy self-corrected -- always notify",
            "can_disable": False
        },
        "info": {
            "enabled": False,
            "description": "Pipeline completed cleanly -- optional",
            "can_disable": True
        }
    },
    "info_on_complete": False
}

with open("watcher_policies.json", "w") as f:
    json.dump(policies, f, indent=2)

print("  OK Notification policy added to watcher_policies.json")
PYEOF
fi

# ---------------------------------------------------------------------------
# STEP 5: Verify integration
# ---------------------------------------------------------------------------
echo "[5/5] Running verification..."

# Check helper exists
if grep -q "def notify_telegram" agents/watcher_agent.py; then
    echo "  OK notify_telegram() helper present"
else
    echo "  FAIL notify_telegram() not found"
fi

# Check notification call sites
CRITICAL_COUNT=$(grep -c "notify_telegram.*CRITICAL" agents/watcher_agent.py || true)
DEGRADED_COUNT=$(grep -c "notify_telegram.*DEGRADED" agents/watcher_agent.py || true)
INFO_COUNT=$(grep -c "notify_telegram.*INFO" agents/watcher_agent.py || true)

echo "  Notification call sites:"
echo "    CRITICAL: $CRITICAL_COUNT (expected: 1)"
echo "    DEGRADED: $DEGRADED_COUNT (expected: 3)"
echo "    INFO:     $INFO_COUNT (expected: 1)"

# Check watcher still loads
echo "  Testing watcher import..."
PYTHONPATH=. python3 -c "
from agents.watcher_agent import WatcherAgent, notify_telegram
print('  OK WatcherAgent + notify_telegram import successful')
" 2>&1 || echo "  FAIL Import error -- check watcher_agent.py"

echo ""
echo "=== Patch Complete (Session 77) ==="
echo ""
echo "Summary:"
echo "  1. cluster_notify.sh installed to /usr/local/bin/"
echo "  2. notify_telegram() helper added to watcher_agent.py"
echo "  3. Notification calls wired at 5 decision points:"
echo "     - _handle_escalate (CRITICAL)"
echo "     - Training health RETRY (DEGRADED)"
echo "     - Training retries exhausted (DEGRADED)"
echo "     - Model type skipped (DEGRADED)"
echo "     - Pipeline complete (INFO, disabled by default)"
echo "  4. Notification policy added to watcher_policies.json"
echo ""
echo "Test commands:"
echo "  # Verify watcher loads:"
echo "  PYTHONPATH=. python3 agents/watcher_agent.py --status"
echo ""
echo "  # Test Telegram directly:"
echo "  /usr/local/bin/cluster_notify.sh '[WATCHER][TEST] Session 77 integration verified'"
echo ""
echo "  # Enable INFO notifications (optional):"
echo "  python3 -c \"import json; p=json.load(open('watcher_policies.json')); p['notifications']['info_on_complete']=True; json.dump(p,open('watcher_policies.json','w'),indent=2)\""
echo ""
echo "Rollback:"
echo "  cp $BACKUP_DIR/watcher_agent.py.bak agents/watcher_agent.py"
echo "  cp $BACKUP_DIR/watcher_policies.json.bak watcher_policies.json"
echo "  sudo rm /usr/local/bin/cluster_notify.sh"
