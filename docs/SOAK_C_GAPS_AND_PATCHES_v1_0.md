# Soak C Integration Gaps & Proposed Patches

**Document:** SOAK_C_GAPS_AND_PATCHES_v1_0.md  
**Version:** 1.0.0  
**Date:** 2026-02-05 (Session 60)  
**Status:** Gaps Documented, Patches Proposed  

---

## 1. Executive Summary

Soak C (full autonomous loop test) was attempted but could not complete autonomously due to integration gaps between:
- `chapter_13_orchestrator.py` (detects draws, runs diagnostics, fires triggers)
- `chapter_13_acceptance.py` (validates proposals, gates execution)
- `watcher_agent.py` (executes pipeline steps)

The system successfully:
- ✅ Detected synthetic draws
- ✅ Generated diagnostics
- ✅ Fired appropriate triggers (hit_rate_collapse)
- ✅ Created approval requests

The system failed to:
- ❌ Honor `auto_approve_in_test_mode` flag
- ❌ Honor `skip_escalation_in_test_mode` flag
- ❌ Execute learning loop autonomously (required manual `--approve`)
- ❌ Start LLM server within timeout (60s exceeded)

---

## 2. Correct Soak C Architecture (What We Learned)

### 2.1 Component Roles

| Component | Role | Daemon Command |
|-----------|------|----------------|
| `synthetic_draw_injector.py` | Injects draws every 60s | `--daemon --interval 60` |
| `chapter_13_orchestrator.py` | Detects draws → diagnostics → triggers → approval request | `--daemon --auto-start-llm` |
| `watcher_agent.py` | Executes pipeline steps when approved | `--daemon` (watches `watcher_requests/`) |

### 2.2 Intended Flow

```
synthetic_draw_injector.py
        │ creates new_draw.flag + appends to lottery_history.json
        ▼
chapter_13_orchestrator.py (daemon)
        │ detects fingerprint change
        │ runs chapter_13_diagnostics.py
        │ evaluates triggers (chapter_13_triggers.py)
        │ gets LLM analysis (optional)
        │ validates proposal (chapter_13_acceptance.py)
        ▼
    ┌───────────────────────────────────────┐
    │ IF test_mode + auto_approve:          │
    │   → Auto-approve, execute immediately │
    │ ELSE:                                 │
    │   → Create pending_approval.json      │
    │   → Wait for human --approve          │
    └───────────────────────────────────────┘
        │
        ▼ (if approved)
watcher_agent.py --run-pipeline --start-step 3 --end-step 6
        │ executes learning loop
        ▼
chapter_13_orchestrator.py
        │ clears pending_approval.json
        │ continues monitoring for next draw
```

### 2.3 What Actually Happened

```
synthetic_draw_injector.py ✅
        │
        ▼
chapter_13_orchestrator.py ✅
        │ diagnostics generated ✅
        │ triggers fired ✅
        │ LLM analysis attempted (timeout) ⚠️
        │ proposal validated → ESCALATED ❌
        ▼
pending_approval.json created ❌ (should have auto-approved)
        │
        ▼
BLOCKED — waiting for manual --approve
```

---

## 3. Gap Analysis

### Gap 1: Acceptance Engine Ignores Test Mode Flags

**File:** `chapter_13_acceptance.py`  
**Lines:** 470-482

**Current Code:**
```python
# If any escalation reasons, escalate
if escalation_reasons:
    return self._create_decision(
        ValidationResult.ESCALATE,
        "Mandatory escalation",
        violations,
        proposal_id,
        timestamp,
        escalation_reasons=escalation_reasons
    )
```

**Problem:** No check for `skip_escalation_in_test_mode` before escalating.

**Impact:** Every cycle hits mandatory escalation due to `consecutive_misses >= 3`, even in test mode.

---

### Gap 2: No Auto-Approve Logic in Acceptance Engine

**File:** `chapter_13_acceptance.py`

**Problem:** The `auto_approve_in_test_mode` flag exists in `watcher_policies.json` but is never read or acted upon.

**Impact:** Even if escalation is skipped, there's no path to automatic approval.

---

### Gap 3: Orchestrator Cannot Execute Pipeline Directly

**File:** `chapter_13_orchestrator.py`  
**Line:** (in `--approve` handler)

**Current Behavior:**
```
ERROR: No WatcherAgent reference - cannot execute pipeline
   Run learning loop via watcher_agent.py instead:
   python3 watcher_agent.py --run-pipeline --start-step 3 --end-step 6
```

**Problem:** Orchestrator creates approval requests but cannot execute them. Requires separate WATCHER invocation.

**Impact:** True autonomous operation requires WATCHER daemon to poll for approved requests, but WATCHER daemon watches `results/` not `pending_approval.json`.

---

### Gap 4: LLM Server Startup Timeout

**File:** `chapter_13_orchestrator.py` / `llm_lifecycle.py`

**Current Behavior:**
```
Failed to start LLM server: Command '['bash', 'llm_services/start_llm_servers.sh']' 
timed out after 60 seconds
```

**Problem:** LLM server (DeepSeek-R1-14B with 32K context) takes >60s to load on Zeus.

**Impact:** Falls back to heuristic analysis (acceptable for test), but logs error.

---

### Gap 5: WATCHER Daemon Watches Wrong Location

**File:** `watcher_agent.py`

**Current Behavior:**
```
Starting daemon mode, watching: results
```

**Problem:** WATCHER daemon watches `results/` for step output files, not `pending_approval.json` or `watcher_requests/`.

**Impact:** WATCHER daemon doesn't pick up Chapter 13 approval requests. The two systems are disconnected.

---

## 4. Proposed Patches

### Patch 1: Honor `skip_escalation_in_test_mode` in Acceptance Engine

**File:** `chapter_13_acceptance.py`  
**Location:** Around line 470

**Patch:**
```python
# === PATCH: Check test mode before escalating ===
# If any escalation reasons, escalate (unless test mode skip enabled)
if escalation_reasons:
    # Check if we should skip escalation in test mode
    policies = self._load_policies()
    test_mode = policies.get('test_mode', False)
    skip_escalation = policies.get('skip_escalation_in_test_mode', False)
    
    if test_mode and skip_escalation:
        logger.info(
            "Skipping escalation in test mode: %s", 
            ", ".join(escalation_reasons)
        )
        # Continue to acceptance criteria instead of returning
    else:
        return self._create_decision(
            ValidationResult.ESCALATE,
            "Mandatory escalation",
            violations,
            proposal_id,
            timestamp,
            escalation_reasons=escalation_reasons
        )
# === END PATCH ===
```

---

### Patch 2: Add Auto-Approve Logic in Acceptance Engine

**File:** `chapter_13_acceptance.py`  
**Location:** After escalation check, before acceptance criteria

**Patch:**
```python
# === PATCH: Auto-approve in test mode ===
policies = self._load_policies()
test_mode = policies.get('test_mode', False)
auto_approve = policies.get('auto_approve_in_test_mode', False)

if test_mode and auto_approve:
    logger.info("Auto-approving in test mode")
    return self._create_decision(
        ValidationResult.ACCEPT,
        "Auto-approved (test_mode + auto_approve_in_test_mode)",
        violations=[],
        proposal_id=proposal_id,
        timestamp=timestamp,
    )
# === END PATCH ===
```

---

### Patch 3: Extend LLM Startup Timeout

**File:** `chapter_13_orchestrator.py` or `llm_lifecycle.py`

**Current:** `timeout=60`

**Patch:**
```python
# Increase timeout for large model loading
LLM_STARTUP_TIMEOUT = 180  # 3 minutes for DeepSeek-R1-14B
```

Or add to `watcher_policies.json`:
```json
{
  "llm_startup_timeout_seconds": 180
}
```

---

### Patch 4: Orchestrator Executes Pipeline via Subprocess

**File:** `chapter_13_orchestrator.py`  
**Location:** In approve handler

**Patch:**
```python
# === PATCH: Execute pipeline via subprocess if no WatcherAgent ===
def _execute_learning_loop_standalone(self, steps: List[int]):
    """Execute learning loop via watcher_agent subprocess."""
    import subprocess
    
    cmd = [
        sys.executable, 
        "agents/watcher_agent.py",
        "--run-pipeline",
        "--start-step", str(min(steps)),
        "--end-step", str(max(steps)),
    ]
    
    logger.info("Executing learning loop via subprocess: %s", " ".join(cmd))
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    result = subprocess.run(
        cmd,
        cwd=self.project_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,  # 1 hour max
    )
    
    if result.returncode != 0:
        logger.error("Learning loop failed: %s", result.stderr)
        return False
    
    logger.info("Learning loop completed successfully")
    return True
# === END PATCH ===
```

---

### Patch 5: WATCHER Daemon Polls Chapter 13 Requests

**File:** `watcher_agent.py`  
**Location:** In daemon loop

**Patch:**
```python
# === PATCH: Check for Chapter 13 approval in daemon loop ===
def _check_chapter_13_approved(self):
    """Check if Chapter 13 has an auto-approved request ready."""
    approval_file = Path("pending_approval.json")
    if not approval_file.exists():
        return None
    
    with open(approval_file) as f:
        approval = json.load(f)
    
    # Check if auto-approved (status == "approved" or test_mode conditions met)
    if approval.get("status") == "approved":
        return approval
    
    # Check test mode auto-approve
    policies = self._load_policies()
    if (policies.get("test_mode") and 
        policies.get("auto_approve_in_test_mode")):
        approval["status"] = "auto_approved"
        return approval
    
    return None

# In daemon loop:
def _daemon_iteration(self):
    # Existing: scan watcher_requests/
    self._scan_watcher_requests()
    
    # NEW: check Chapter 13 approvals
    ch13_approval = self._check_chapter_13_approved()
    if ch13_approval:
        logger.info("Processing Chapter 13 approved request")
        self._execute_learning_loop(ch13_approval.get("steps", [3, 5, 6]))
        self._archive_approval(ch13_approval)
# === END PATCH ===
```

---

## 5. Consolidated Patch File

**File:** `patch_soak_c_integration_v1.py`

```python
#!/usr/bin/env python3
"""
Soak C Integration Patches

Applies patches to enable true autonomous operation in test mode:
1. chapter_13_acceptance.py — Honor skip_escalation_in_test_mode
2. chapter_13_acceptance.py — Honor auto_approve_in_test_mode
3. chapter_13_orchestrator.py — Execute pipeline via subprocess
4. llm_lifecycle.py — Extend startup timeout

Usage:
    python3 patch_soak_c_integration_v1.py --apply
    python3 patch_soak_c_integration_v1.py --revert
    python3 patch_soak_c_integration_v1.py --check
"""

import argparse
import shutil
import re
from pathlib import Path
from datetime import datetime

PATCHES = {
    "chapter_13_acceptance.py": {
        "backup_suffix": ".pre_soakc_patch",
        "patches": [
            {
                "name": "skip_escalation_in_test_mode",
                "find": '''        # If any escalation reasons, escalate
        if escalation_reasons:
            return self._create_decision(
                ValidationResult.ESCALATE,
                "Mandatory escalation",''',
                "replace": '''        # If any escalation reasons, escalate (unless test mode)
        if escalation_reasons:
            # PATCH: Check test mode before escalating
            _policies = self._load_policies() if hasattr(self, '_load_policies') else {}
            _test_mode = _policies.get('test_mode', False)
            _skip_esc = _policies.get('skip_escalation_in_test_mode', False)
            
            if not (_test_mode and _skip_esc):
                return self._create_decision(
                    ValidationResult.ESCALATE,
                    "Mandatory escalation",'''
            },
            {
                "name": "auto_approve_in_test_mode", 
                "find": '''        # =====================================================================
        # AUTOMATIC ACCEPTANCE (Section 13.2)
        # =====================================================================''',
                "replace": '''        # =====================================================================
        # PATCH: AUTO-APPROVE IN TEST MODE
        # =====================================================================
        _policies = self._load_policies() if hasattr(self, '_load_policies') else {}
        if _policies.get('test_mode') and _policies.get('auto_approve_in_test_mode'):
            logger.info("Auto-approving proposal in test mode")
            return self._create_decision(
                ValidationResult.ACCEPT,
                "Auto-approved (test_mode + auto_approve_in_test_mode)",
                violations=[],
                proposal_id=proposal_id,
                timestamp=timestamp,
            )
        
        # =====================================================================
        # AUTOMATIC ACCEPTANCE (Section 13.2)
        # ====================================================================='''
            }
        ]
    }
}


def apply_patches():
    """Apply all Soak C integration patches."""
    for filename, config in PATCHES.items():
        filepath = Path(filename)
        if not filepath.exists():
            print(f"❌ File not found: {filename}")
            continue
        
        # Backup
        backup_path = Path(f"{filename}{config['backup_suffix']}")
        if not backup_path.exists():
            shutil.copy(filepath, backup_path)
            print(f"📦 Backed up: {filename} → {backup_path}")
        
        # Read content
        content = filepath.read_text()
        
        # Apply patches
        for patch in config["patches"]:
            if patch["find"] in content:
                content = content.replace(patch["find"], patch["replace"])
                print(f"✅ Applied: {patch['name']} to {filename}")
            elif patch["replace"] in content:
                print(f"⚠️  Already applied: {patch['name']} to {filename}")
            else:
                print(f"❌ Could not find target for: {patch['name']} in {filename}")
        
        # Write
        filepath.write_text(content)
    
    print("\n✅ Patches applied. Run Soak C again.")


def revert_patches():
    """Revert all Soak C integration patches."""
    for filename, config in PATCHES.items():
        backup_path = Path(f"{filename}{config['backup_suffix']}")
        filepath = Path(filename)
        
        if backup_path.exists():
            shutil.copy(backup_path, filepath)
            print(f"✅ Reverted: {filename} from {backup_path}")
        else:
            print(f"⚠️  No backup found for: {filename}")
    
    print("\n✅ Patches reverted.")


def check_patches():
    """Check if patches are applied."""
    for filename, config in PATCHES.items():
        filepath = Path(filename)
        if not filepath.exists():
            print(f"❌ File not found: {filename}")
            continue
        
        content = filepath.read_text()
        
        for patch in config["patches"]:
            if patch["replace"] in content:
                print(f"✅ Applied: {patch['name']} in {filename}")
            elif patch["find"] in content:
                print(f"⬜ Not applied: {patch['name']} in {filename}")
            else:
                print(f"❓ Unknown state: {patch['name']} in {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soak C Integration Patches")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true", help="Apply patches")
    group.add_argument("--revert", action="store_true", help="Revert patches")
    group.add_argument("--check", action="store_true", help="Check patch status")
    
    args = parser.parse_args()
    
    if args.apply:
        apply_patches()
    elif args.revert:
        revert_patches()
    elif args.check:
        check_patches()
```

---

## 6. Soak C Retry Procedure (After Patches)

### Step 1: Apply Patches
```bash
cd ~/distributed_prng_analysis
python3 patch_soak_c_integration_v1.py --apply
```

### Step 2: Verify Policies
```bash
grep -E "test_mode|auto_approve|skip_escalation" watcher_policies.json
```

Expected:
```
"test_mode": true,
"auto_approve_in_test_mode": true,
"skip_escalation_in_test_mode": true
```

### Step 3: Clean State
```bash
rm -f pending_approval.json new_draw.flag
cp lottery_history.json.pre_soakC lottery_history.json
```

### Step 4: Re-run Bootstrap
```bash
python3 << 'EOF'
# (same bootstrap script as before)
EOF
```

### Step 5: Start Daemons (2 terminals)

**Terminal 1:**
```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 synthetic_draw_injector.py --daemon --interval 60
```

**Terminal 2:**
```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 chapter_13_orchestrator.py --daemon --auto-start-llm \
    |& tee logs/soak/soakC_orchestrator_$(date +%Y%m%d_%H%M%S).log
```

### Step 6: Verify Autonomous Operation

After 2-3 cycles, check:
```bash
# Should see "Auto-approving proposal in test mode"
grep -i "auto-approv" logs/soak/soakC_orchestrator_*.log

# Should NOT see "pending_approval"
grep -c "pending_approval" logs/soak/soakC_orchestrator_*.log
```

### Step 7: Let Run for 1-2 Hours

Monitor for:
- Cycles completing without human intervention
- No error accumulation
- No memory leaks (RSS stable)

### Step 8: Verify Results
```bash
echo "Cycles: $(grep -c 'CHAPTER 13 CYCLE' logs/soak/soakC_orchestrator_*.log)"
echo "Auto-approved: $(grep -c 'Auto-approv' logs/soak/soakC_orchestrator_*.log)"
echo "Escalated: $(grep -c 'ESCALATE\|pending_approval' logs/soak/soakC_orchestrator_*.log)"
echo "Errors: $(grep -ci 'error\|exception\|traceback' logs/soak/soakC_orchestrator_*.log)"
```

**Pass criteria:**
- Cycles > 0
- Auto-approved == Cycles (or close)
- Escalated == 0 (or only LLM timeout warnings)
- Errors < 5 (LLM timeout is expected)

---

## 7. Post-Soak C Cleanup

```bash
# Restore production configs
cp lottery_history.json.pre_soakC lottery_history.json
cp optimal_window_config.json.pre_soakC optimal_window_config.json
cp watcher_policies.json.pre_soakC watcher_policies.json

# Optionally revert patches (or keep for future testing)
python3 patch_soak_c_integration_v1.py --revert
```

---

## 8. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-05 | Initial gap analysis and patches |

---

**END OF DOCUMENT**
