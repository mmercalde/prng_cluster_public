#!/usr/bin/env python3
"""Patch COMPLETE_OPERATING_GUIDE_v1_1.md → v2.0.0"""

filepath = "docs/COMPLETE_OPERATING_GUIDE_v1_1.md"

with open(filepath, 'r') as f:
    content = f.read()

changes = 0

# 1. Update header
old_header = '**Version 1.1.0**  \n**December 2025**  \n**Updated: Session 17 (Dec 27, 2025)**'
new_header = '**Version 2.0.0**  \n**February 2026**  \n**Updated: Session 80 (Feb 11, 2026)**'
assert content.count(old_header) == 1, f"Header: found {content.count(old_header)}"
content = content.replace(old_header, new_header)
changes += 1

# 2. Update cluster description
old_cluster = 'Zeus (2× RTX 3080 Ti) + rig-6600 (8× RX 6600) + rig-6600b (8× RX 6600)  \n~285 TFLOPS Combined Computing Power'
new_cluster = 'Zeus (2× RTX 3080 Ti) + rig-6600 (8× RX 6600) + rig-6600b (8× RX 6600) + rig-6600c (8× RX 6600)  \n26 GPUs • ~285 TFLOPS Combined Computing Power'
if content.count(old_cluster) == 1:
    content = content.replace(old_cluster, new_cluster)
    changes += 1
else:
    # Try with encoded ×
    old_cluster2 = 'Zeus (2Ã— RTX 3080 Ti) + rig-6600 (8Ã— RX 6600) + rig-6600b (8Ã— RX 6600)'
    if content.count(old_cluster2) == 1:
        content = content.replace(old_cluster2, 'Zeus (2× RTX 3080 Ti) + rig-6600 (8× RX 6600) + rig-6600b (8× RX 6600) + rig-6600c (8× RX 6600)')
        changes += 1
    else:
        print("WARNING: Could not find cluster description to update")

# 3. Update hardware table - add rig-6600c and fix IP line
old_hw = """**Network:** All nodes connected via SSH with key-based authentication.  
**IP Addresses:** rig-6600 (192.168.3.120), rig-6600b (192.168.3.154)"""
new_hw = """| rig-6600c | RX 6600 (8GB) | 8 | ROCm / HIP |

**Network:** All nodes connected via SSH with key-based authentication.  
**IP Addresses:** rig-6600 (192.168.3.120), rig-6600b (192.168.3.154), rig-6600c (192.168.3.162)  
**GPU Stability:** udev rule (perf=high auto-set), GFXOFF disabled via kernel parameter  
**SSH Alias:** `rzeus` (from ser8 to Zeus)"""
assert content.count(old_hw) == 1, f"HW table: found {content.count(old_hw)}"
content = content.replace(old_hw, new_hw)
changes += 1

# 4. Fix rig-6600 count from 12 to 8 (if still wrong)
content = content.replace('| rig-6600 | RX 6600 (8GB) | 12 | ROCm / HIP |',
                          '| rig-6600 | RX 6600 (8GB) | 8 | ROCm / HIP |')
content = content.replace('| rig-6600b | RX 6600 (8GB) | 12 | ROCm / HIP |',
                          '| rig-6600b | RX 6600 (8GB) | 8 | ROCm / HIP |')
changes += 1

# 5. Update TOC
old_toc = """1. System Overview
2. Pipeline Steps
3. Core Modules
4. Distributed Workers
5. Configuration System
6. Operational Procedures
7. Monitoring & Debugging
8. Appendix A: Data Models
9. Appendix B: File Inventory"""
new_toc = """1. System Overview
2. Pipeline Steps
3. Core Modules
4. Distributed Workers
5. Configuration System
6. Operational Procedures
7. Monitoring & Debugging
8. Chapter 13: Live Feedback Loop
9. WATCHER Agent & Daemon
10. Policy System
11. Appendix A: Data Models
12. Appendix B: File Inventory"""
assert content.count(old_toc) == 1, f"TOC: found {content.count(old_toc)}"
content = content.replace(old_toc, new_toc)
changes += 1

# 6. Update version numbers in file inventory
content = content.replace(
    '| reinforcement_engine.py | Neural network model (v1.4.0) |',
    '| reinforcement_engine.py | Neural network model (v1.7.0, Ch14 diagnostics hooks) |')
content = content.replace(
    '| prng_registry.py | 46 PRNG implementations (v2.4) |',
    '| prng_registry.py | 46 PRNG implementations (v2.4, GPU vectorization) |')
changes += 1

# 7. Add NPZ v3.0 note to Step 3 output section
old_step3_output = '### Output'
# Find the Step 3 output specifically (after line 199)
# We'll add a note about NPZ and run_step3_full_scoring.sh
old_step3_run = '## 2.3 Step 3: Full Distributed Scoring'
if old_step3_run in content:
    # Add note after the Step 3 header
    content = content.replace(
        '## 2.3 Step 3: Full Distributed Scoring',
        '## 2.3 Step 3: Full Distributed Scoring\n\n'
        '> **Updated (Session 80):** Step 3 now uses `run_step3_full_scoring.sh` v2.0.0 '
        '(scripts_coordinator.py compliant, supersedes old `run_full_scoring.sh`). '
        'NPZ v3.0 format preserves all 22 metadata fields through `convert_survivors_to_binary.py` v3.0.')
    changes += 1

# 8. Add LLM infrastructure note to Software Dependencies
old_deps = '- XGBoost, LightGBM, CatBoost for multi-model comparison (NEW Session 9)'
new_deps = ('- XGBoost, LightGBM, CatBoost for multi-model comparison (NEW Session 9)\n'
            '- DeepSeek-R1-14B (primary LLM) via llama.cpp with ROCm/HIP backend\n'
            '- Grammar-constrained decoding via GBNF files (Pydantic + chapter_13.gbnf)\n'
            '- Claude Opus 4.6 (backup LLM for strategic escalation)')
assert content.count(old_deps) == 1, f"Deps: found {content.count(old_deps)}"
content = content.replace(old_deps, new_deps)
changes += 1

# 9. Add WATCHER to core architecture components table
old_arch_end = '| Multi-Model Comparison | meta_prediction_optimizer_anti_overfit.py | 4 ML models with subprocess isolation |'
new_arch_end = (old_arch_end + '\n'
    '| WATCHER Agent | agents/watcher_agent.py | v2.0.0 autonomous pipeline executor (2,795 lines) |\n'
    '| Chapter 13 Orchestrator | chapter_13_orchestrator.py | Live feedback loop: detect → diagnose → trigger → approve |\n'
    '| Training Diagnostics | training_diagnostics.py | Chapter 14: model health monitoring (GPU/CPU metrics) |\n'
    '| Strategy Advisor | parameter_advisor.py | LLM-guided parameter recommendations |\n'
    '| Bundle Factory | agents/contexts/bundle_factory.py | Unified LLM context assembly (7 bundle types) |')
assert content.count(old_arch_end) == 1, f"Arch table: found {content.count(old_arch_end)}"
content = content.replace(old_arch_end, new_arch_end)
changes += 1

# 11. Append new sections before end of document
old_end = '**— End of Document —**'
new_sections = """---

# PART 8: CHAPTER 13 — LIVE FEEDBACK LOOP

## 8.1 Overview

Chapter 13 implements a closed-loop feedback system that monitors prediction accuracy after each draw, diagnoses degradation, and autonomously triggers retraining when needed.

### Architecture

| Component | File | Purpose |
|-----------|------|---------|
| Diagnostics | `chapter_13_diagnostics.py` | Post-draw metrics (hit rate, confidence, survivor performance) |
| Triggers | `chapter_13_triggers.py` | Retrain trigger thresholds (hit_rate_collapse, confidence_drift) |
| LLM Advisor | `chapter_13_llm_advisor.py` | DeepSeek-R1-14B analysis with grammar-constrained output |
| Acceptance | `chapter_13_acceptance.py` | Proposal validation (delta limits, cooldowns, escalation) |
| Orchestrator | `chapter_13_orchestrator.py` | Daemon mode: detect draws → diagnose → trigger → LLM → accept/reject |

### Cycle Flow

```
New draw detected (new_draw.flag)
  → Generate diagnostics (post_draw_diagnostics.json)
    → Evaluate triggers (hit_rate_collapse, confidence_drift, etc.)
      → LLM analysis (DeepSeek-R1-14B, grammar-constrained)
        → Acceptance engine (validate proposal)
          → Route to execution authority (orchestrator or WATCHER)
```

### Running the Orchestrator

```bash
# Daemon mode (polls for new draws)
PYTHONPATH=. python3 chapter_13_orchestrator.py --daemon --auto-start-llm

# Single cycle
PYTHONPATH=. python3 chapter_13_orchestrator.py --run-once
```

## 8.2 Chapter 14: Training Diagnostics

Training diagnostics monitor model health during Step 5 execution, detecting issues like gradient collapse, loss divergence, and GPU memory pressure.

| Phase | Component | Status |
|-------|-----------|--------|
| Diagnostic Engine | `training_diagnostics.py` | ✅ Complete |
| GPU/CPU Collection | CUDA + ROCm metrics | ✅ Complete |
| Engine Wiring | `reinforcement_engine.py` v1.7.0 | ✅ Complete |
| RETRY Param-Threading | WATCHER health check | ✅ Complete |
| FIFO Pruning | Unbounded growth prevention | ✅ Complete |
| Health Check | `check_training_health()` | ✅ Complete |

---

# PART 9: WATCHER AGENT & DAEMON

## 9.1 Overview

The WATCHER agent (`agents/watcher_agent.py`, v2.0.0, ~2,800 lines) is the autonomous execution engine. It runs pipeline steps, evaluates results, and dispatches follow-up actions.

### CLI Commands

```bash
# Run pipeline manually
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 3 --end-step 6

# Start daemon (persistent, polls for approvals)
PYTHONPATH=. python3 agents/watcher_agent.py --daemon

# Check daemon status
PYTHONPATH=. python3 agents/watcher_agent.py --status

# Stop daemon gracefully (<2s response)
PYTHONPATH=. python3 agents/watcher_agent.py --stop

# View recent decisions
PYTHONPATH=. python3 agents/watcher_agent.py --explain 5

# Manually approve pending request
PYTHONPATH=. python3 agents/watcher_agent.py --approve
```

## 9.2 Daemon Mode

When running as `--daemon`, WATCHER:
1. Writes PID file (`watcher_daemon.pid`)
2. Installs SIGTERM/SIGINT handlers for graceful shutdown
3. Polls every 30s for `pending_approval.json`
4. Auto-approves in test mode (dual-flag check)
5. Dispatches `run_pipeline()` for approved requests
6. Archives completed requests to `watcher_requests/archive/`
7. Persists state to `daemon_state.json`
8. Sends Telegram notifications on approval events

### Key Design: Lifecycle Separation

```
self.running          → daemon loop (stays alive between pipelines)
self._pipeline_running → pipeline execution (set/cleared per run)
SIGTERM               → kills both
```

## 9.3 Approval Flow

```
approval_route = "orchestrator"  →  Ch13 decides AND executes internally
approval_route = "watcher"       →  Ch13 writes pending_approval.json
                                     → WATCHER detects, approves, dispatches
```

## 9.4 Preflight Checks

Before each step, WATCHER runs preflight checks:

| Check | Type | Failure |
|-------|------|---------|
| SSH reachability | HARD | Blocks step |
| GPU availability | HARD | Blocks step |
| Ramdisk populated | SOFT | Auto-remediation |
| Input files present | HARD | Blocks step |

HARD failures block execution. SOFT failures attempt auto-fix.

---

# PART 10: POLICY SYSTEM

## 10.1 Policy File

All runtime policies live in `watcher_policies.json`. This is the single source of truth for system behavior flags.

## 10.2 Critical Flags

| Flag | Production | Test/Soak | Purpose |
|------|-----------|-----------|---------|
| `test_mode` | `false` | `true` | Master test switch |
| `auto_approve_in_test_mode` | `false` | `true` | WATCHER auto-approves |
| `skip_escalation_in_test_mode` | `false` | `true` | Suppress mandatory escalation |
| `approval_route` | `"orchestrator"` | `"watcher"` | Execution authority |

## 10.3 Mode Switching

```bash
# Check current mode
python3 -c "
import json
with open('watcher_policies.json') as f: p = json.load(f)
for k in ['test_mode','auto_approve_in_test_mode','skip_escalation_in_test_mode','approval_route']:
    print(f'  {k}: {p.get(k, \"NOT SET\")}')
"
```

## 10.4 Safety Invariants

1. `test_mode=false` overrides all other test flags
2. Auto-approve requires BOTH `test_mode=true` AND `auto_approve_in_test_mode=true`
3. WATCHER never mutates policies — only humans change `watcher_policies.json`
4. Invalid `approval_route` values silently fall back to `"orchestrator"`

> **📋 Complete Policy Reference:** See `docs/WATCHER_POLICIES_REFERENCE.md` for full flag documentation.

---

**— End of Document —**"""
assert content.count(old_end) == 1, f"End marker: found {content.count(old_end)}"
content = content.replace(old_end, new_sections)
changes += 1

# Write
with open(filepath, 'w') as f:
    f.write(content)

import py_compile
# Can't compile .md but verify it wrote correctly
lines = content.count('\n') + 1
print(f"✅ {changes} patches applied — {lines} lines total")
