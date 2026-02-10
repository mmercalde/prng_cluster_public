#!/bin/bash
# =============================================================================
# Session 76: RETRY Param-Threading Patch (v2 -- Team Beta fixes applied)
# =============================================================================
# Wires check_training_health() into run_pipeline() between Step 5 and Step 6.
# When health returns RETRY, re-runs Step 5 with modified params from
# get_retry_params_suggestions(). Respects max_training_retries policy.
#
# Team Beta fixes (v2):
#   1. ASCII-only -- no Unicode emojis/arrows in injected code
#   2. Single health check call -- cached dict passed to both helpers
#   3. Centralized _get_max_training_retries() helper
#
# Files modified:
#   1. agents/watcher_agent.py  - Import + run_pipeline hook + helper methods
#   2. watcher_policies.json    - Add max_training_retries param
#
# Usage: bash apply_s76_retry_threading.sh
# Rollback: git checkout agents/watcher_agent.py watcher_policies.json
# =============================================================================

set -euo pipefail
cd ~/distributed_prng_analysis

echo "=== Session 76: RETRY Param-Threading Patch (v2) ==="
echo ""

# ---------------------------------------------------------------------------
# STEP 0: Backup files before modification
# ---------------------------------------------------------------------------
echo "[0/5] Backing up files..."

BACKUP_DIR="backups/s76_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

cp agents/watcher_agent.py "$BACKUP_DIR/watcher_agent.py.bak"
cp watcher_policies.json "$BACKUP_DIR/watcher_policies.json.bak"

echo "  OK Backups saved to $BACKUP_DIR/"
echo "  Rollback: cp $BACKUP_DIR/*.bak . && mv watcher_agent.py.bak agents/watcher_agent.py && mv watcher_policies.json.bak watcher_policies.json"
echo ""

# ---------------------------------------------------------------------------
# STEP 1: Add import for training_health_check at top of watcher_agent.py
# ---------------------------------------------------------------------------
echo "[1/5] Adding training_health_check import..."

if grep -q "from training_health_check import" agents/watcher_agent.py; then
    echo "  -> Import already exists, skipping"
else
    # Insert after the gpu_cleanup import block (line with GPU_CLEANUP_AVAILABLE = False)
    sed -i '/^GPU_CLEANUP_AVAILABLE = False$/a\
\
# Training Health Check Integration (Session 76 -- RETRY param-threading)\
try:\
    from training_health_check import check_training_health, reset_skip_registry, get_retry_params_suggestions\
    TRAINING_HEALTH_AVAILABLE = True\
except ImportError:\
    TRAINING_HEALTH_AVAILABLE = False\
    check_training_health = None' agents/watcher_agent.py
    echo "  OK Import added"
fi

# ---------------------------------------------------------------------------
# STEP 2: Add _get_max_training_retries() centralized policy helper
# ---------------------------------------------------------------------------
echo "[2/5] Adding _get_max_training_retries() helper..."

if grep -q "_get_max_training_retries" agents/watcher_agent.py; then
    echo "  -> Helper already exists, skipping"
else
    # Insert before _run_step_streaming
    sed -i '/def _run_step_streaming/i\
    def _get_max_training_retries(self) -> int:\
        """\
        Read max_retries from watcher_policies.json (centralized).\
        Session 76: Single source of truth for training retry limit.\
        Path: training_diagnostics.severity_thresholds.critical.max_retries\
        """\
        try:\
            if os.path.isfile("watcher_policies.json"):\
                with open("watcher_policies.json") as pf:\
                    policies = json.load(pf)\
                td = policies.get("training_diagnostics", {})\
                st = td.get("severity_thresholds", {}).get("critical", {})\
                return st.get("max_retries", 2)\
        except Exception as e:\
            logger.debug("Could not read max_training_retries from policy: %s", e)\
        return 2\
' agents/watcher_agent.py
    echo "  OK Helper added"
fi

# ---------------------------------------------------------------------------
# STEP 3: Add _handle_training_health() and _build_retry_params() methods
#          Both accept a pre-fetched health dict (no double-call)
# ---------------------------------------------------------------------------
echo "[3/5] Adding _handle_training_health() + _build_retry_params() methods..."

if grep -q "_handle_training_health" agents/watcher_agent.py; then
    echo "  -> Methods already exist, skipping"
else
    # Insert before _get_max_training_retries (which we just added above _run_step_streaming)
    sed -i '/def _get_max_training_retries/i\
    def _handle_training_health(self, health: Dict[str, Any]) -> str:\
        """\
        Post-Step-5 training health check with RETRY param-threading.\
        \
        Session 76: Maps check_training_health() result to pipeline action.\
        Accepts pre-fetched health dict (caller caches -- no double call).\
        \
        Args:\
            health: Result dict from check_training_health()\
        \
        Returns:\
            "proceed" - continue to Step 6\
            "retry"   - re-run Step 5 with modified params (caller handles)\
            "skip"    - model skipped, continue to Step 6 anyway\
        """\
        action = health.get("action", "PROCEED")\
        model_type = health.get("model_type", "unknown")\
        severity = health.get("severity", "absent")\
        issues = health.get("issues", [])\
        \
        # -- PROCEED / PROCEED_WITH_NOTE --\
        if action in ("PROCEED", "PROCEED_WITH_NOTE"):\
            if severity != "critical":\
                reset_skip_registry(model_type)\
            if action == "PROCEED_WITH_NOTE":\
                logger.warning(\
                    "[WATCHER][HEALTH] Training health NOTE (%s): %s",\
                    model_type, "; ".join(issues[:3])\
                )\
            else:\
                logger.info("[WATCHER][HEALTH] Training health OK (%s) -- proceeding to Step 6", model_type)\
            return "proceed"\
        \
        # -- SKIP_MODEL --\
        if action == "SKIP_MODEL":\
            consecutive = health.get("consecutive_critical", "?")\
            logger.warning(\
                "[WATCHER][HEALTH] Training health SKIP_MODEL (%s): %s consecutive critical -- skipping",\
                model_type, consecutive\
            )\
            return "skip"\
        \
        # -- RETRY --\
        if action == "RETRY":\
            logger.warning(\
                "[WATCHER][HEALTH] Training health CRITICAL (%s): %s -- requesting RETRY",\
                model_type, "; ".join(issues[:3])\
            )\
            return "retry"\
        \
        # Unknown action -> proceed (best-effort)\
        logger.warning("[WATCHER][HEALTH] Unknown health action: %s -- proceeding", action)\
        return "proceed"\
\
    def _build_retry_params(self, health: Dict[str, Any], original_params: Dict[str, Any] = None) -> Dict[str, Any]:\
        """\
        Build modified Step 5 params for retry based on diagnostics.\
        Session 76: Uses get_retry_params_suggestions() from training_health_check.\
        Accepts pre-fetched health dict (same instance as _handle_training_health).\
        """\
        suggestions = get_retry_params_suggestions(health)\
        \
        retry_params = dict(original_params or {})\
        \
        # Apply suggestions to retry params\
        if suggestions.get("model_type"):\
            retry_params["model_type"] = suggestions["model_type"]\
            logger.info("[WATCHER][RETRY] Switching model_type to %s", suggestions["model_type"])\
        \
        if suggestions.get("increase_regularization"):\
            current = retry_params.get("dropout", 0.3)\
            retry_params["dropout"] = min(current + 0.1, 0.7)\
            logger.info("[WATCHER][RETRY] Increasing dropout to %.2f", retry_params["dropout"])\
        \
        if suggestions.get("normalize_features"):\
            retry_params["normalize_features"] = True\
            logger.info("[WATCHER][RETRY] Enabling feature normalization")\
        \
        if suggestions.get("use_leaky_relu"):\
            retry_params["use_leaky_relu"] = True\
            logger.info("[WATCHER][RETRY] Enabling LeakyReLU activation")\
        \
        logger.info("[WATCHER][RETRY] Modified params: %s", {k: v for k, v in retry_params.items() if k != "reason"})\
        return retry_params\
' agents/watcher_agent.py
    echo "  OK Methods added"
fi

# ---------------------------------------------------------------------------
# STEP 4: Wire health check into run_pipeline() between Step 5 and Step 6
# ---------------------------------------------------------------------------
echo "[4/5] Wiring health check into run_pipeline()..."

if grep -q "training_health_retries" agents/watcher_agent.py; then
    echo "  -> Pipeline wiring already exists, skipping"
else
    # First, add the training_health_retries counter init near the other retry inits
    sed -i 's/self.running = True/self.running = True\n        training_health_retries = 0  # Session 76: RETRY param-threading counter/' agents/watcher_agent.py

    # Now replace the "Small delay between steps" section with the health check hook
    python3 << 'PYEOF'
with open("agents/watcher_agent.py", "r") as f:
    content = f.read()

old_block = """                # Small delay between steps
                time.sleep(1)"""

new_block = """                # -- Session 76: Post-Step-5 training health check ----------
                if (step == 5
                        and decision.recommended_action == "proceed"
                        and TRAINING_HEALTH_AVAILABLE):

                    # Single health check call -- cached for both helpers
                    _health = check_training_health()
                    _max_retries = self._get_max_training_retries()
                    health_action = self._handle_training_health(_health)

                    if health_action == "retry":
                        if training_health_retries < _max_retries:
                            training_health_retries += 1
                            retry_params = self._build_retry_params(_health, params)
                            logger.warning(
                                "[WATCHER][HEALTH] Training health RETRY %d/%d -- re-running Step 5",
                                training_health_retries, _max_retries
                            )
                            # Stay on Step 5 -- override current_step back
                            # (_handle_proceed already advanced to 6)
                            self.current_step = 5
                            params = retry_params
                            time.sleep(1)
                            continue
                        else:
                            logger.error(
                                "[WATCHER][HEALTH] Max training retries (%d) exhausted -- proceeding to Step 6",
                                _max_retries
                            )
                    elif health_action == "skip":
                        logger.warning("[WATCHER][HEALTH] Model type skipped -- proceeding to Step 6")
                    else:
                        # Reset only after successful health PROCEED
                        training_health_retries = 0

                # Small delay between steps
                time.sleep(1)"""

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    with open("agents/watcher_agent.py", "w") as f:
        f.write(content)
    print("  OK Pipeline wiring applied")
else:
    print("  FAIL Could not find insertion point -- check manually")
    exit(1)
PYEOF
fi

# ---------------------------------------------------------------------------
# STEP 5: Add max_training_retries to watcher_policies.json
# ---------------------------------------------------------------------------
echo "[5/5] Updating watcher_policies.json..."

if python3 -c "
import json, sys
with open('watcher_policies.json') as f:
    p = json.load(f)
td = p.get('training_diagnostics', {})
st = td.get('severity_thresholds', {}).get('critical', {})
if 'max_retries' in st:
    print('already_present')
    sys.exit(0)
else:
    sys.exit(1)
" 2>/dev/null; then
    echo "  -> max_retries already in policies, skipping"
else
    python3 << 'PYEOF'
import json

with open("watcher_policies.json") as f:
    policies = json.load(f)

# Ensure training_diagnostics section exists
if "training_diagnostics" not in policies:
    policies["training_diagnostics"] = {
        "enabled": True,
        "description": "Post-Step-5 training health check before proceeding to Step 6",
        "severity_thresholds": {
            "ok": {"action": "PROCEED", "log_level": "info"},
            "warning": {"action": "PROCEED_WITH_NOTE", "log_level": "warning"},
            "critical": {
                "action": "RETRY_OR_SKIP",
                "log_level": "error",
                "max_retries": 2,
                "description": "Training fundamentally broken -- retry with different config or skip model type"
            }
        },
        "metric_bounds": {
            "nn_dead_neuron_pct": {"warning": 25.0, "critical": 50.0},
            "nn_gradient_spread_ratio": {"warning": 100.0, "critical": 1000.0},
            "overfit_ratio": {"warning": 1.3, "critical": 1.5},
            "early_stop_ratio": {"warning": 0.3, "critical": 0.15},
            "unused_feature_pct": {"warning": 40.0, "critical": 70.0}
        },
        "model_skip_rules": {
            "max_consecutive_critical": 3,
            "skip_duration_hours": 24
        }
    }
else:
    # Just ensure max_retries is in critical threshold
    td = policies["training_diagnostics"]
    if "severity_thresholds" not in td:
        td["severity_thresholds"] = {}
    if "critical" not in td["severity_thresholds"]:
        td["severity_thresholds"]["critical"] = {}
    td["severity_thresholds"]["critical"]["max_retries"] = 2
    td["severity_thresholds"]["critical"]["description"] = (
        "Training fundamentally broken -- retry with different config or skip model type"
    )

with open("watcher_policies.json", "w") as f:
    json.dump(policies, f, indent=2)

print("  OK watcher_policies.json updated with max_retries=2")
PYEOF
fi

echo ""
echo "=== Patch Complete (v2 -- Team Beta fixes) ==="
echo ""
echo "Summary of changes:"
echo "  1. agents/watcher_agent.py:"
echo "     - Added import: training_health_check (check, reset, get_retry_params)"
echo "     - Added helper: _get_max_training_retries() -- centralized policy read"
echo "     - Added method: _handle_training_health(health) -- accepts cached dict"
echo "     - Added method: _build_retry_params(health, params) -- accepts cached dict"
echo "     - Added hook in run_pipeline(): single check_training_health() call,"
echo "       result passed to both helpers (no double-call)"
echo "     - Training-specific retry counter (separate from step retry_counts)"
echo ""
echo "  2. watcher_policies.json:"
echo "     - Added max_retries=2 to training_diagnostics.severity_thresholds.critical"
echo ""
echo "Team Beta fixes applied (v2):"
echo "  [1] ASCII-only -- no Unicode emojis/arrows in injected code"
echo "  [2] Single health check call -- cached dict passed to both helpers"
echo "  [3] Centralized _get_max_training_retries() policy helper"
echo ""
echo "Behavior:"
echo "  Step 5 PROCEED -> check_training_health() ->"
echo "    PROCEED       -> reset skip registry, continue to Step 6"
echo "    PROCEED_NOTE  -> log warning, continue to Step 6"
echo "    RETRY         -> re-run Step 5 with modified params (up to 2x)"
echo "    SKIP_MODEL    -> log skip, continue to Step 6"
echo "    (max retries) -> log exhaustion, continue to Step 6"
echo ""
echo "Test commands:"
echo "  # Dry test -- check watcher still starts:"
echo "  PYTHONPATH=. python3 agents/watcher_agent.py --status"
echo ""
echo "  # Full test -- run Steps 5-6 with diagnostics enabled:"
echo "  PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \\"
echo "    --start-step 5 --end-step 6 \\"
echo "    --params '{\"trials\":3,\"max_seeds\":5000,\"enable_diagnostics\":true}'"
echo ""
echo "Rollback (from backup):"
echo "  cp $BACKUP_DIR/watcher_agent.py.bak agents/watcher_agent.py"
echo "  cp $BACKUP_DIR/watcher_policies.json.bak watcher_policies.json"
echo ""
echo "Rollback (from git):"
echo "  git checkout agents/watcher_agent.py watcher_policies.json"
