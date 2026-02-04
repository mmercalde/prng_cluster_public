#!/usr/bin/env python3
"""
Documentation Staleness Fixer — Session 60
==========================================
Updates 6 critically stale documents to reflect Phase 7 completion (Sessions 57-59).

Run on Zeus:
    cd ~/distributed_prng_analysis
    python3 update_stale_docs_v1.py

What it does:
    1. CHAPTER_13_IMPLEMENTATION_PROGRESS — v2.0 → v3.0 (Phase 7 COMPLETE)
    2. CHAPTER_13_SECTION_19_UPDATED     — Phase 7 checkboxes checked
    3. README.md                         — Recent updates, cluster, roadmap
    4. CANONICAL_PIPELINE                — Autonomy section, convergence metrics
    5. CHAPTER_12_WATCHER_AGENT          — Dispatch, bundle factory, lifecycle
    6. CHAPTER_10_AUTONOMOUS_AGENT       — New components, version bump

Safety:
    - Creates .bak backup of every file before modifying
    - Dry-run mode: python3 update_stale_docs_v1.py --dry-run
    - Copies updated files to ~/doc_updates_staging/ for scp to ser8
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path.home() / "distributed_prng_analysis"
DOCS_DIR = PROJECT_ROOT / "docs"
STAGING_DIR = Path.home() / "doc_updates_staging"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Track results
results = {"patched": [], "created": [], "skipped": [], "errors": []}


def find_file(name, search_dirs=None):
    """Find a file in project root or docs/."""
    if search_dirs is None:
        search_dirs = [DOCS_DIR, PROJECT_ROOT]
    for d in search_dirs:
        path = d / name
        if path.exists():
            return path
    return None


def backup_file(path):
    """Create timestamped backup."""
    bak = Path(str(path) + f".bak_{TIMESTAMP}")
    shutil.copy2(path, bak)
    return bak


def safe_replace(content, old, new, label=""):
    """Replace text, return (new_content, success)."""
    if old in content:
        return content.replace(old, new, 1), True
    return content, False


def write_result(path, content, dry_run=False):
    """Write content to file and staging."""
    if dry_run:
        print(f"  [DRY RUN] Would write {len(content)} chars to {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    # Also copy to staging
    staging = STAGING_DIR / path.name
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, staging)


# ═══════════════════════════════════════════════════════════════
# FILE 1: CHAPTER_13_IMPLEMENTATION_PROGRESS  v2.0 → v3.0
# ═══════════════════════════════════════════════════════════════
def patch_ch13_progress(dry_run=False):
    """Full rewrite — too many changes for surgical patches."""
    label = "CHAPTER_13_IMPLEMENTATION_PROGRESS"
    print(f"\n{'='*60}")
    print(f"[1/6] {label}")
    print(f"{'='*60}")

    # Find the v2.0 file
    old_path = find_file("CHAPTER_13_IMPLEMENTATION_PROGRESS_v2_0.md")
    if old_path:
        backup_file(old_path)
        print(f"  Backed up: {old_path}")

    # New path — create v3.0
    new_path = DOCS_DIR / "CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_0.md"

    content = r"""# Chapter 13 Implementation Progress

**Last Updated:** 2026-02-03
**Document Version:** 3.0.0
**Status:** ✅ ALL PHASES COMPLETE — Full Autonomous Operation Achieved
**Team Beta Endorsement:** ✅ Approved (Phase 7 verified Session 59)

---

## ⚠️ Documentation Sync Notice (2026-02-03)

**Previous issue (2026-01-30):** Section 19 checklist showed unchecked boxes despite code being complete since January 12.

**Resolution (2026-02-03):** Phase 7 WATCHER Integration now COMPLETE (Sessions 57-59). All dispatch functions wired, end-to-end test passed. This document is the authoritative progress tracker.

**Lesson Learned:** When code is completed, update BOTH the progress tracker AND the original chapter checklist within the same session.

---

## Overall Progress

| Phase | Status | Owner | Completion | Verified |
|-------|--------|-------|------------|----------|
| 1. Draw Ingestion | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 2. Diagnostics Engine | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 3. Retrain Triggers | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 4. LLM Integration | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 5. Acceptance Engine | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| 6. Chapter 13 Orchestration | ✅ Complete | Claude | 2026-01-12 | 2026-01-30 |
| **7. WATCHER Integration** | **✅ Complete** | **Team Alpha+Beta** | **2026-02-03** | **2026-02-03** |
| 8. Selfplay Integration | ✅ Complete | Team Beta | 2026-01-30 | 2026-01-30 |
| 9A. Chapter 13 ↔ Selfplay Hooks | ✅ Complete | Team Beta | 2026-01-30 | 2026-01-30 |
| 9B.1 Policy Transform Module | ✅ Complete | Claude | 2026-01-30 | 2026-01-30 |
| 9B.2 Policy-Conditioned Mode | ✅ Complete | Claude | 2026-01-30 | 2026-01-30 |
| 9B.3 Policy Proposal Heuristics | 🔲 Future | TBD | — | — |

**Legend:** 🔲 Not Started | 🟡 In Progress | ✅ Complete | ❌ Blocked/Missing

---

## Files Inventory (Verified 2026-02-03)

### Chapter 13 Core Files

| File | Size | Created | Updated | Purpose |
|------|------|---------|---------|---------|
| `chapter_13_diagnostics.py` | 39KB | Jan 12 | Jan 29 | Diagnostics engine |
| `chapter_13_llm_advisor.py` | 23KB | Jan 12 | Jan 12 | LLM analysis module |
| `chapter_13_triggers.py` | 36KB | Jan 12 | Jan 29 | Retrain trigger logic |
| `chapter_13_acceptance.py` | 41KB | Jan 12 | Jan 29 | Proposal validation |
| `chapter_13_orchestrator.py` | 23KB | Jan 12 | Jan 12 | Main orchestrator |
| `llm_proposal_schema.py` | 14KB | Jan 12 | Jan 12 | Pydantic models |
| `chapter_13.gbnf` | 2.9KB | Jan 12 | Jan 12 | LLM grammar constraint |
| `draw_ingestion_daemon.py` | 22KB | Jan 12 | Jan 12 | Draw monitoring |
| `synthetic_draw_injector.py` | 20KB | Jan 12 | Jan 12 | Test mode injection |
| `watcher_policies.json` | 4.7KB | Jan 12 | Jan 29 | Policy thresholds |

**Total:** ~226KB of Chapter 13 code

### Phase 7 Files (WATCHER Integration — Sessions 57-59)

| File | Size | Created | Purpose |
|------|------|---------|---------|
| `agents/watcher_dispatch.py` | ~30KB | Feb 02 | Dispatch functions (selfplay, learning loop, request processing) |
| `agents/contexts/bundle_factory.py` | ~32KB | Feb 02 | Step awareness bundle assembly engine |
| `llm_services/llm_lifecycle.py` | ~8KB | Feb 01 | LLM lifecycle management (stop/restart around GPU phases) |
| `agent_grammars/*.gbnf` | ~6KB | Feb 01 | Fixed v1.1 GBNF grammar files (4 files) |
| `docs/ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` | ~10KB | Feb 02 | Bundle factory specification |

### Phase 9B Files (Selfplay)

| File | Size | Created | Purpose |
|------|------|---------|---------|
| `selfplay_orchestrator.py` | 43KB | Jan 29 | Main selfplay loop (v1.1.0) |
| `policy_transform.py` | 36KB | Jan 30 | Transform engine (v1.0.0) |
| `policy_conditioned_episode.py` | 25KB | Jan 30 | Episode conditioning (v1.0.0) |
| `inner_episode_trainer.py` | — | Jan 29 | Tree model trainer |
| `modules/learning_telemetry.py` | — | Jan 29 | Telemetry system |

---

## ✅ Phase 7: WATCHER Integration (COMPLETE)

**Completed:** Sessions 57-59 (2026-02-01 through 2026-02-03)

### What Was Built

| Function | File | Purpose | Status |
|----------|------|---------|--------|
| `dispatch_selfplay()` | `agents/watcher_dispatch.py` | Spawn selfplay_orchestrator.py | ✅ Verified |
| `dispatch_learning_loop()` | `agents/watcher_dispatch.py` | Run Steps 3→5→6 | ✅ Verified |
| `process_chapter_13_request()` | `agents/watcher_dispatch.py` | Handle watcher_requests/*.json | ✅ Verified |
| `build_step_awareness_bundle()` | `agents/contexts/bundle_factory.py` | Unified LLM context assembly | ✅ Verified |
| LLM Lifecycle Management | `llm_services/llm_lifecycle.py` | Stop/restart LLM around GPU phases | ✅ Verified |

### Integration Flow (WIRED AND VERIFIED)

```
Chapter 13 Triggers                WATCHER                    Execution
──────────────────                 ───────                    ─────────
request_selfplay()
        │
        └──► watcher_requests/*.json
                    │
                    └──► process_chapter_13_request()  ✅ WIRED
                              │
                              ▼
                         validate_request()
                              │
                              ▼ (if APPROVE)
                    dispatch_selfplay()  ✅ WIRED
                              │
                              └──────────────────────► selfplay_orchestrator.py
```

### D5 End-to-End Test (Session 59 — Clean Pass)

```
Pre-validation: real LLM (4s response, not instant heuristic)
LLM stop: "confirmed stopped — GPU VRAM freed"
Selfplay: rc=0, candidate emitted (58s)
LLM restart: "healthy after 3.2s"
Post-eval: grammar-constrained JSON — real structured output
Archive: COMPLETED — zero warnings, zero heuristic fallbacks
```

### Five Integration Bugs Found & Fixed

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | Lifecycle dead code | `self.llm_lifecycle` never set in `__init__` | Added initialization block |
| 2 | API mismatch | `.start()` / `.stop(string)` not real methods | → `.ensure_running()` / `.stop()` |
| 3 | Router always None | `GrammarType` import poisoned entire import | Removed dead import |
| 4 | Grammar 400 errors | `agent_grammars/` had broken v1.0 GBNF | Copied fixed v1.1 from `grammars/` |
| 5 | Try 1 private API | `_call_primary_with_grammar()` missing config | Gate to public API for `watcher_decision.gbnf` only |

---

## ✅ What Works Today (Full Autonomous Operation)

### Can Run via WATCHER (ALL MODES)

```bash
# Pipeline Steps 1-6
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 6

# Dispatch selfplay
PYTHONPATH=. python3 agents/watcher_agent.py --dispatch-selfplay

# Dispatch learning loop
PYTHONPATH=. python3 agents/watcher_agent.py --dispatch-learning-loop steps_3_5_6

# Process Chapter 13 requests
PYTHONPATH=. python3 agents/watcher_agent.py --process-requests

# Daemon mode (monitor + auto-dispatch)
PYTHONPATH=. python3 agents/watcher_agent.py --daemon
```

### Autonomous Loop (VERIFIED)

```
Chapter 13 Triggers → watcher_requests/ → WATCHER → Selfplay
       ↑                                              ↓
       └────────── Diagnostics ← Reality ←───────────┘
```

No human in the loop for routine decisions.

---

## Critical Design Invariants

### Chapter 13 Invariant
**Chapter 13 v1 does not alter model weights directly. All learning occurs through controlled re-execution of Step 5 with expanded labels.**

### Selfplay Invariant
**GPU sieving work MUST use coordinator.py / scripts_coordinator.py. Direct SSH to rigs for GPU work is FORBIDDEN.**

### Learning Authority Invariant
**Learning is statistical (tree models + bandit). Verification is deterministic (Chapter 13). LLM is advisory only. Telemetry is observational only.**

### Policy Transform Invariant
**`apply_policy()` is pure functional: stateless, deterministic, never fabricates data. Same inputs always produce same outputs.**

### Dispatch Guardrails (NEW — Phase 7)
**Guardrail #1:** Single context entry point — dispatch calls `build_llm_context()`, nothing else.
**Guardrail #2:** No baked-in token assumptions — bundle_factory owns prompt structure.

### Documentation Sync Invariant
**When code is completed, update BOTH the progress tracker AND the original chapter checklist within the same session.**

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-12 | 1.0.0 | Initial document, Phases 1-6 code complete |
| 2026-01-18 | 1.1.0 | Added Phase 7 testing framework |
| 2026-01-23 | 1.2.0 | NPZ v3.0 integration notes |
| 2026-01-27 | 1.3.0 | GPU stability improvements |
| 2026-01-29 | 1.5.0 | Phase 8 Selfplay architecture approved |
| 2026-01-30 | 1.6.0 | Phase 8 COMPLETE — Zeus integration verified |
| 2026-01-30 | 1.7.0 | Phase 9A COMPLETE — Hooks verified |
| 2026-01-30 | 1.8.0 | Phase 9B.1 COMPLETE — Policy Transform Module |
| 2026-01-30 | 1.9.0 | Phase 9B.2 COMPLETE — Integration verified |
| 2026-01-30 | 2.0.0 | Documentation audit — Identified Phase 7 as actual gap |
| **2026-02-03** | **3.0.0** | **Phase 7 COMPLETE — Full autonomous operation achieved** |

---

## Next Steps

1. **Soak Testing** (Optional) — Run daemon mode for extended periods, verify stability
2. **Phase 9B.3** (Deferred) — Automatic policy proposal heuristics
3. **Parameter Advisor** (Deferred) — LLM-advised parameter recommendations for Steps 4-6
4. **`--save-all-models` flag** — Save all 4 models in Step 5 for post-hoc AI analysis

---

*Update this document as implementation progresses.*
"""

    write_result(new_path, content, dry_run)
    results["created"].append(str(new_path))
    print(f"  ✅ Created: {new_path}")

    # If old v2.0 exists, mark it superseded
    if old_path and not dry_run:
        print(f"  ℹ️  Old v2.0 remains at {old_path} (backed up, can be removed)")
        print(f"     To remove: rm {old_path}")

    return True


# ═══════════════════════════════════════════════════════════════
# FILE 2: CHAPTER_13_SECTION_19_UPDATED
# ═══════════════════════════════════════════════════════════════
def patch_section_19(dry_run=False):
    label = "CHAPTER_13_SECTION_19_UPDATED"
    print(f"\n{'='*60}")
    print(f"[2/6] {label}")
    print(f"{'='*60}")

    path = find_file("CHAPTER_13_SECTION_19_UPDATED.md")
    if not path:
        print(f"  ❌ File not found — will create fresh")
        path = DOCS_DIR / "CHAPTER_13_SECTION_19_UPDATED.md"
    else:
        backup_file(path)
        print(f"  Backed up: {path}")

    content = r"""# CHAPTER 13 — Section 19 (UPDATED)

**Last Verified:** 2026-02-03
**Status:** ALL PHASES COMPLETE — Full Autonomous Operation

---

## 19. Implementation Checklist

### Phase 1: Draw Ingestion ✅ COMPLETE (2026-01-12)

- [x] `draw_ingestion_daemon.py` — Monitors for new draws (22KB)
- [x] `synthetic_draw_injector.py` — Test mode draw generation (20KB)
  - Reads PRNG type from `optimal_window_config.json` (no hardcoding)
  - Uses `prng_registry.py` (same as Steps 1-6)
  - Modes: manual (`--inject-one`), daemon (`--daemon --interval 60`), flag-triggered
- [x] Append-only history updates
- [x] Fingerprint change detection
- [x] `watcher_policies.json` — Includes test_mode and synthetic_injection settings (4.7KB, updated Jan 29)

### Phase 2: Diagnostics Engine ✅ COMPLETE (2026-01-12, updated 2026-01-29)

- [x] `chapter_13_diagnostics.py` — Core diagnostic generator (39KB)
- [x] Prediction vs reality comparison
- [x] Confidence calibration metrics
- [x] Survivor performance tracking
- [x] Feature drift detection
- [x] Generate `post_draw_diagnostics.json`
- [x] Create `diagnostics_history/` archival

### Phase 3: LLM Integration ✅ COMPLETE (2026-01-12)

- [x] `chapter_13_llm_advisor.py` — LLM analysis module (23KB)
- [x] `llm_proposal_schema.py` — Pydantic model for proposals (14KB)
- [x] `chapter_13.gbnf` — Grammar constraint (2.9KB)
- [x] System/user prompt templates
- [x] Integration with existing LLM infrastructure

### Phase 4: WATCHER Policies ✅ COMPLETE (2026-01-12, updated 2026-01-29)

- [x] `chapter_13_acceptance.py` — Acceptance/rejection rules (41KB)
- [x] `chapter_13_triggers.py` — Retrain trigger thresholds (36KB)
- [x] Cooldown enforcement
- [x] Escalation handlers

### Phase 5: Orchestration ✅ COMPLETE (2026-01-12)

- [x] `chapter_13_orchestrator.py` — Main orchestrator (23KB)
- [x] Partial rerun logic (Steps 3→5→6)
- [x] Full rerun trigger (Steps 1→6)
- [x] Decision logging
- [x] Audit trail

### Phase 6: Testing ✅ COMPLETE (2026-02-03)

- [x] Synthetic draw injection (module exists)
- [x] Proposal validation tests (in acceptance.py)
- [x] End-to-end convergence monitoring (via D5 integration test)
- [x] Divergence detection tests (via acceptance engine)
- [x] Live integration testing (Session 59 — D5 clean pass)

### Phase 7: WATCHER Integration ✅ COMPLETE (2026-02-03, Sessions 57-59)

**Full autonomous loop verified — no human in the loop for routine decisions.**

- [x] `dispatch_selfplay()` in `agents/watcher_dispatch.py` (Session 58)
- [x] `dispatch_learning_loop()` in `agents/watcher_dispatch.py` (Session 58)
- [x] `process_chapter_13_request()` in `agents/watcher_dispatch.py` (Session 58)
- [x] `build_step_awareness_bundle()` in `agents/contexts/bundle_factory.py` (Session 58)
- [x] LLM lifecycle management in `llm_services/llm_lifecycle.py` (Session 57)
- [x] Wire Chapter 13 orchestrator into WATCHER daemon (Session 58)
- [x] Move `chapter_13.gbnf` to `agent_grammars/` directory (Session 57)
- [x] Fix v1.0 → v1.1 GBNF grammar files (Session 59)
- [x] Integration tests: WATCHER → Chapter 13 → Selfplay (Session 59, D5 clean pass)
- [x] Five integration bugs found and fixed (Session 59)

---

## Files Summary (Verified 2026-02-03)

| File | Size | Phase | Status |
|------|------|-------|--------|
| `draw_ingestion_daemon.py` | 22KB | 1 | ✅ |
| `synthetic_draw_injector.py` | 20KB | 1 | ✅ |
| `watcher_policies.json` | 4.7KB | 1,4 | ✅ |
| `chapter_13_diagnostics.py` | 39KB | 2 | ✅ |
| `chapter_13_llm_advisor.py` | 23KB | 3 | ✅ |
| `llm_proposal_schema.py` | 14KB | 3 | ✅ |
| `chapter_13.gbnf` | 2.9KB | 3 | ✅ |
| `chapter_13_acceptance.py` | 41KB | 4,5 | ✅ |
| `chapter_13_triggers.py` | 36KB | 4 | ✅ |
| `chapter_13_orchestrator.py` | 23KB | 5 | ✅ |
| `agents/watcher_dispatch.py` | ~30KB | 7 | ✅ |
| `agents/contexts/bundle_factory.py` | ~32KB | 7 | ✅ |
| `llm_services/llm_lifecycle.py` | ~8KB | 7 | ✅ |
| `agent_grammars/*.gbnf` | ~6KB | 7 | ✅ |

**Total Chapter 13 + Phase 7 Code:** ~300KB+ across 14+ files

---

## Autonomous Loop (VERIFIED)

```
Chapter 13 Triggers → watcher_requests/ → WATCHER → Selfplay
       ↑                                              ↓
       └────────── Diagnostics ← Reality ←───────────┘
```

---

*This section replaces the original Section 19 in CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md*
"""

    write_result(path, content, dry_run)
    results["patched"].append(str(path))
    print(f"  ✅ Updated: {path}")
    return True


# ═══════════════════════════════════════════════════════════════
# FILE 3: README.md
# ═══════════════════════════════════════════════════════════════
def patch_readme(dry_run=False):
    label = "README.md"
    print(f"\n{'='*60}")
    print(f"[3/6] {label}")
    print(f"{'='*60}")

    path = find_file("README.md", [PROJECT_ROOT])
    if not path:
        print(f"  ❌ README.md not found at project root")
        results["errors"].append(label)
        return False

    backup_file(path)
    print(f"  Backed up: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    patches_applied = 0

    # Patch 1: Title — remove "Seed Reconstruction", emphasize functional mimicry
    content, ok = safe_replace(
        content,
        "# 🎲 Distributed PRNG Analysis & Seed Reconstruction System",
        "# 🎲 Distributed PRNG Analysis & Functional Mimicry System"
    )
    if ok: patches_applied += 1

    # Patch 2: Subtitle
    content, ok = safe_replace(
        content,
        "**Multi-GPU Cluster • AI Agent Architecture • ML Scoring • Optuna Meta-Optimization • Reinforcement Engine**",
        "**Multi-GPU Cluster • Autonomous AI Agents • ML Scoring • Selfplay Learning • Optuna Meta-Optimization**"
    )
    if ok: patches_applied += 1

    # Patch 3: Recent Updates section — replace entire block
    old_updates = """## 🆕 Recent Updates (December 24, 2025)

### Session 15: Step 6 Confidence Fix + Lineage
- ✅ **CRITICAL FIX:** Confidence scores now differentiated (was all 1.0)
- ✅ Raw scores preserved for automation cross-run comparability
- ✅ Parent run ID lineage from Step 5 → Step 6
- ✅ Score statistics for monitoring (min/max/mean/std/unique)

### Session 14: Step 6 Restoration v2.2
- ✅ GlobalStateTracker module (14 global features, GPU-neutral)
- ✅ Type-tolerant intersection (handles int and dict survivors)
- ✅ Model loading from sidecar with feature schema

### Session 11-12: Multi-Model Architecture
- ✅ Subprocess isolation for OpenCL/CUDA conflict
- ✅ 4 ML models: XGBoost, LightGBM, CatBoost, Neural Net
- ✅ Model checkpoint persistence

📄 See `CURRENT_Status.txt` for detailed session history."""

    new_updates = """## 🆕 Recent Updates (February 2026)

### Sessions 57-59: Phase 7 WATCHER Integration — COMPLETE ✅
- ✅ **FULL AUTONOMY ACHIEVED** — End-to-end Chapter 13 → WATCHER → Selfplay loop
- ✅ Dispatch module (`watcher_dispatch.py`) — selfplay, learning loop, request processing
- ✅ Bundle factory (`bundle_factory.py`) — unified LLM context assembly (7 bundle types)
- ✅ LLM lifecycle management — automatic stop/restart around GPU-intensive phases
- ✅ 5 integration bugs found and fixed in D5 end-to-end testing
- ✅ Grammar-constrained LLM evaluation with real DeepSeek-R1 responses

### Session 56: Selfplay Validation + LLM Infrastructure
- ✅ Selfplay system validated — 8 episodes, 3 candidates, policy-conditioned mode
- ✅ LLM infrastructure upgraded — 8K → 32K context windows
- ✅ LLM lifecycle management deployed

### Sessions 53-55: Chapter 13 + Selfplay Architecture
- ✅ Policy transform module — stateless, deterministic, pure functional
- ✅ Policy-conditioned episodes — filter, weight, mask transforms
- ✅ Authority contract ratified: Chapter 13 decides, WATCHER executes, selfplay explores

📄 See `docs/SESSION_CHANGELOG_*.md` for detailed session history."""

    content, ok = safe_replace(content, old_updates, new_updates)
    if ok: patches_applied += 1

    # Patch 4: Dual-LLM diagram → update model names
    content, ok = safe_replace(
        content,
        "│   GPU0: ORCHESTRATOR    │    │   GPU1: MATH SPECIALIST │",
        "│   GPU0: PRIMARY LLM     │    │   GPU1: SPECIALIST LLM  │"
    )
    if ok: patches_applied += 1

    content, ok = safe_replace(
        content,
        "│   Qwen2.5-Coder-14B     │    │   Qwen2.5-Math-7B       │",
        "│   DeepSeek-R1-14B        │    │   (Configurable)         │"
    )
    if ok: patches_applied += 1

    # Patch 5: Cluster table — fix GPU counts and add rig-6600c
    old_cluster = """| Node | GPUs | Type | Purpose |
|------|------|------|---------|
| Zeus (Primary) | 2× RTX 3080 Ti | CUDA | Orchestration, LLM hosting, job generation |
| rig-6600 | 8× RX 6600 | ROCm | Worker Node 1 |
| rig-6600b | 8× RX 6600 | ROCm | Worker Node 2 |
| **Total** | **26 GPUs** | | **~285 TFLOPS** |"""

    new_cluster = """| Node | GPUs | Type | Purpose |
|------|------|------|---------|
| Zeus (Primary) | 2× RTX 3080 Ti | CUDA | Orchestration, LLM hosting, job generation |
| rig-6600 | 12× RX 6600 | ROCm | Worker Node 1 |
| rig-6600b | 12× RX 6600 | ROCm | Worker Node 2 |
| rig-6600c | (deploying) | ROCm | Worker Node 3 (192.168.3.162) |
| **Total** | **26+ GPUs** | | **~285 TFLOPS** |"""

    content, ok = safe_replace(content, old_cluster, new_cluster)
    if ok: patches_applied += 1

    # Patch 6: PRNG count 44 → 46
    content, ok = safe_replace(content,
        "**44 PRNG Algorithms** across 11 families",
        "**46 PRNG Algorithms** across 11+ families"
    )
    if ok: patches_applied += 1

    # Patch 7: Project structure — add new directories
    old_structure = """├── agents/                    # AI Agent implementations
│   ├── agent_core.py          # BaseAgent class
│   └── __init__.py"""

    new_structure = """├── agents/                    # AI Agent implementations
│   ├── watcher_agent.py       # Main WATCHER orchestrator
│   ├── watcher_dispatch.py    # Selfplay/learning dispatch
│   └── contexts/
│       └── bundle_factory.py  # LLM context assembly
│
├── agent_grammars/            # GBNF grammar constraints
│   ├── chapter_13.gbnf
│   ├── watcher_decision.gbnf
│   └── agent_decision.gbnf"""

    content, ok = safe_replace(content, old_structure, new_structure)
    if ok: patches_applied += 1

    # Patch 8: Add selfplay files to structure
    old_prng_line = """├── prng_registry.py           # 44 PRNG kernels"""
    new_prng_line = """├── prng_registry.py           # 46 PRNG kernels
├── selfplay_orchestrator.py   # Selfplay learning loop
├── policy_transform.py        # Pure-functional policy transforms
├── policy_conditioned_episode.py  # Policy-conditioned episodes"""

    content, ok = safe_replace(content, old_prng_line, new_prng_line)
    if ok: patches_applied += 1

    # Patch 9: Roadmap — check completed items, add new ones
    old_roadmap = """- [x] 26-GPU distributed architecture
- [x] 44 PRNG algorithms (forward + reverse)
- [x] 6-step pipeline
- [x] Dual-LLM infrastructure
- [x] Agent manifests
- [x] Schema v1.0.4 with agent_metadata
- [ ] Watcher Agent (autonomous pipeline)
- [ ] optuna_agent_bridge.py (cross-run learning)
- [ ] WebUI for visualization"""

    new_roadmap = """- [x] 26-GPU distributed architecture
- [x] 46 PRNG algorithms (forward + reverse)
- [x] 6-step pipeline with autonomous execution
- [x] LLM infrastructure (DeepSeek-R1-14B, 32K context)
- [x] Agent manifests (6 pipeline steps)
- [x] WATCHER Agent — autonomous pipeline orchestration
- [x] Chapter 13 — live feedback loop (10 files, ~226KB)
- [x] Selfplay system — policy-conditioned reinforcement learning
- [x] Full WATCHER dispatch — Chapter 13 → Selfplay → Learning loop
- [x] Bundle factory — unified LLM context assembly
- [x] LLM lifecycle management (stop/restart around GPU phases)
- [ ] Parameter advisor (LLM-advised recommendations for Steps 4-6)
- [ ] Phase 9B.3 (automatic policy proposal heuristics)
- [ ] WebUI for visualization"""

    content, ok = safe_replace(content, old_roadmap, new_roadmap)
    if ok: patches_applied += 1

    # Patch 10: Bottom tagline
    content, ok = safe_replace(
        content,
        "*Distributed PRNG Analysis System — Functional mimicry through ML-enhanced pattern detection*",
        "*Distributed PRNG Analysis System — Functional mimicry through ML-enhanced pattern detection and autonomous selfplay learning*"
    )
    if ok: patches_applied += 1

    write_result(path, content, dry_run)
    results["patched"].append(str(path))
    print(f"  ✅ Applied {patches_applied} patches to: {path}")
    return True


# ═══════════════════════════════════════════════════════════════
# FILE 4: CANONICAL_PIPELINE
# ═══════════════════════════════════════════════════════════════
def patch_canonical(dry_run=False):
    label = "CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE"
    print(f"\n{'='*60}")
    print(f"[4/6] {label}")
    print(f"{'='*60}")

    path = find_file("CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md")
    if not path:
        # File might not exist on Zeus — create it in docs/
        print(f"  ⚠️  File not found on Zeus — creating in docs/")
        path = DOCS_DIR / "CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md"

        # Read from project files if available, otherwise we'll need the content
        # Since we can't access /mnt/project from Zeus, we create from template
        results["skipped"].append(f"{label} (not found, needs manual copy from project)")
        print(f"  ⚠️  SKIPPED: Copy from project files, then re-run this script")
        print(f"     scp ser8:~/Downloads/{label}.md rzeus:~/distributed_prng_analysis/docs/")
        return False

    backup_file(path)
    print(f"  Backed up: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    patches_applied = 0

    # Patch 1: Section 8 — Remove "Request human approval" from Chapter 13 description
    content, ok = safe_replace(
        content,
        "- Request human approval before action (v1)",
        "- Execute via WATCHER dispatch (autonomous LLM evaluation, grammar-constrained)"
    )
    if ok: patches_applied += 1

    # Patch 2: Section 9 — Fix convergence metrics (functional mimicry, not seed ranking)
    old_convergence = """## 9. Convergence Expectations (Test Mode)

| Metric | Target |
|------|-------|
| True seed in top‑100 | ≤ 20 draws |
| True seed in top‑20 | ≤ 50 draws |
| Confidence trend | Increasing |
| Diagnostics | Stable or improving |

Failure to converge is **information**, not error."""

    new_convergence = """## 9. Convergence Expectations (Test Mode)

**Functional Mimicry Paradigm:** This system learns output patterns to predict future draws. Seeds generate candidate sequences for heuristic extraction. Success is measured by hit rate and confidence calibration, NOT seed ranking.

| Metric | Target |
|------|-------|
| Hit rate improvement | Increases after N draws |
| Confidence calibration | Correlation > 0.7 |
| Confidence trend | Increasing |
| Diagnostics | Stable or improving |

Failure to converge is **information**, not error."""

    content, ok = safe_replace(content, old_convergence, new_convergence)
    if ok: patches_applied += 1

    # Patch 3: Section 10 — Add Selfplay to learning table
    old_learns = """| Component | Learns? |
|--------|--------|
| Sieves (Steps 1–3) | ❌ |
| ML Model (Step 5) | ✅ |
| Prediction ranking | ❌ |
| Chapter 13 triggers | ❌ |
| LLM advisor | ❌ |

Only **Step 5** updates weights."""

    new_learns = """| Component | Learns? |
|--------|--------|
| Sieves (Steps 1–3) | ❌ |
| ML Model (Step 5) | ✅ |
| Selfplay (tree models + bandit) | ✅ |
| Prediction ranking | ❌ |
| Chapter 13 triggers | ❌ |
| LLM advisor | ❌ (advisory only) |

**Step 5** updates model weights. **Selfplay** learns via tree models and bandit algorithms. Both are statistical, evidence-driven, and auditable."""

    content, ok = safe_replace(content, old_learns, new_learns)
    if ok: patches_applied += 1

    # Patch 4: Section 11 — Update autonomy boundary
    old_autonomy = """## 11. Autonomy Boundary

Current (v1):
- LLM proposes
- Human approves
- WATCHER executes

Future:
- Automatic execution after confidence threshold
- PRNG switching allowed only after exhaustive failure"""

    new_autonomy = """## 11. Autonomy Boundary

Current (v2 — Phase 7 Complete):
- Chapter 13 triggers retrain/selfplay requests
- WATCHER evaluates via grammar-constrained LLM (DeepSeek-R1)
- WATCHER dispatches selfplay or learning loop autonomously
- LLM lifecycle managed (stop before GPU phase, restart after)
- All decisions audited in `watcher_requests/` archive

Authority separation:
- **Chapter 13 decides** (ground truth, promotion/rejection)
- **WATCHER enforces** (dispatch, safety, halt flag)
- **Selfplay explores** (never self-promotes)

Future:
- Phase 9B.3: Automatic policy proposal heuristics
- Parameter advisor: LLM-recommended tuning for Steps 4-6"""

    content, ok = safe_replace(content, old_autonomy, new_autonomy)
    if ok: patches_applied += 1

    write_result(path, content, dry_run)
    results["patched"].append(str(path))
    print(f"  ✅ Applied {patches_applied} patches to: {path}")
    return True


# ═══════════════════════════════════════════════════════════════
# FILE 5: CHAPTER_12_WATCHER_AGENT
# ═══════════════════════════════════════════════════════════════
def patch_ch12_watcher(dry_run=False):
    label = "CHAPTER_12_WATCHER_AGENT"
    print(f"\n{'='*60}")
    print(f"[5/6] {label}")
    print(f"{'='*60}")

    path = find_file("CHAPTER_12_WATCHER_AGENT.md")
    if not path:
        print(f"  ❌ File not found")
        results["errors"].append(label)
        return False

    backup_file(path)
    print(f"  Backed up: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    patches_applied = 0

    # Patch 1: Version header
    content, ok = safe_replace(
        content,
        "**Version:** 1.1.0  \n**Date:** January 9, 2026",
        "**Version:** 1.4.0  \n**Date:** February 3, 2026"
    )
    if ok: patches_applied += 1

    # Patch 2: Add dispatch + lifecycle + bundle factory to Key Features table
    old_features = """| Feature | Description |
|---------|-------------|
| **Heuristic Evaluation** | Rule-based decisions without LLM |
| **LLM Evaluation** | GBNF-constrained LLM decisions (optional) |
| **Fingerprint Registry** | Track dataset + PRNG combinations |
| **PRNG Priority Order** | Systematic PRNG family testing |
| **Safety Controls** | Kill switch, max retries, escalation |"""

    new_features = """| Feature | Description |
|---------|-------------|
| **Heuristic Evaluation** | Rule-based decisions without LLM |
| **LLM Evaluation** | GBNF-constrained LLM decisions (optional) |
| **Fingerprint Registry** | Track dataset + PRNG combinations |
| **PRNG Priority Order** | Systematic PRNG family testing |
| **Safety Controls** | Kill switch, max retries, escalation |
| **Dispatch Module** | Selfplay, learning loop, request processing (v1.4.0) |
| **Bundle Factory** | Unified LLM context assembly — 7 bundle types (v1.4.0) |
| **LLM Lifecycle** | Auto stop/restart around GPU-intensive dispatch phases (v1.4.0) |
| **Chapter 13 Integration** | Process watcher_requests/, dispatch retrain/selfplay (v1.4.0) |"""

    content, ok = safe_replace(content, old_features, new_features)
    if ok: patches_applied += 1

    # Patch 3: Add new files to File Locations table
    old_files = """| File | Purpose |
|------|---------|
| `agents/watcher_agent.py` | Main WATCHER implementation |
| `agents/watcher_registry_hooks.py` | Registry integration hooks |
| `agents/fingerprint_registry.py` | Dataset fingerprint tracking |
| `agent_manifests/*.json` | Step configurations |
| `watcher_history.json` | Decision history log |
| `watcher_decisions.jsonl` | Detailed decision audit |"""

    new_files = """| File | Purpose |
|------|---------|
| `agents/watcher_agent.py` | Main WATCHER implementation |
| `agents/watcher_dispatch.py` | Dispatch functions (selfplay, learning loop, requests) |
| `agents/contexts/bundle_factory.py` | Step awareness bundle assembly engine |
| `agents/watcher_registry_hooks.py` | Registry integration hooks |
| `agents/fingerprint_registry.py` | Dataset fingerprint tracking |
| `llm_services/llm_lifecycle.py` | LLM lifecycle management (stop/restart) |
| `agent_manifests/*.json` | Step configurations |
| `agent_grammars/*.gbnf` | GBNF grammar constraint files (v1.1) |
| `watcher_history.json` | Decision history log |
| `watcher_decisions.jsonl` | Detailed decision audit |
| `watcher_requests/` | Chapter 13 request queue (JSON files) |"""

    content, ok = safe_replace(content, old_files, new_files)
    if ok: patches_applied += 1

    # Patch 4: Append dispatch section before "END OF CHAPTER 12"
    # (if the end marker exists)
    dispatch_addendum = """
---

## 11. Dispatch Module (v1.4.0 — Session 58)

### 11.1 Overview

The dispatch module (`agents/watcher_dispatch.py`) extends WATCHER with Chapter 13 integration:

| Function | Purpose |
|----------|---------|
| `dispatch_selfplay()` | Spawn selfplay_orchestrator.py with LLM lifecycle management |
| `dispatch_learning_loop()` | Run Steps 3→5→6 re-execution |
| `process_chapter_13_request()` | Handle watcher_requests/*.json approval/dispatch |

### 11.2 Architecture

Uses method binding pattern — standalone module bound to WatcherAgent instance at import time, avoiding inheritance/MRO complexity.

### 11.3 Guardrails

- **Guardrail #1:** All dispatch calls use `build_llm_context()` from bundle_factory — zero inline prompt construction
- **Guardrail #2:** No hardcoded token counts — bundle_factory owns prompt structure
- **Authority:** Selfplay cannot self-promote. Chapter 13 decides. WATCHER enforces.
- **LLM Lifecycle:** Automatic stop before GPU dispatch, restart after
- **Halt Flag:** Checked at entry of every dispatch function AND between steps

### 11.4 Bundle Factory

The bundle factory (`agents/contexts/bundle_factory.py`) provides unified context assembly:

| Bundle Type | Purpose |
|-------------|---------|
| step_1 through step_6 | Pipeline step evaluation context |
| chapter_13 | Post-selfplay/learning evaluation context |

Self-test: `python3 agents/contexts/bundle_factory.py` (verifies all 7 bundles)

### 11.5 LLM Lifecycle Manager

`llm_services/llm_lifecycle.py` manages LLM server state around GPU-intensive operations:

```
dispatch_selfplay():
    llm_lifecycle.stop()          # Free GPU VRAM
    run selfplay_orchestrator.py  # Uses GPUs
    llm_lifecycle.ensure_running() # Restart for evaluation
    evaluate results via LLM       # Grammar-constrained
```

### 11.6 Evaluation Path

```
_evaluate_step_via_bundle(prompt, grammar_name)
    │
    ├─ Try 1: LLM Router (public API) — watcher_decision.gbnf only
    ├─ Try 2: HTTP Direct — all grammars via inline content
    └─ Try 3: Heuristic Fallback — proceed, confidence=0.50 (emergency only)
```

"""

    # Try to insert before END marker
    if "**END OF CHAPTER 12**" in content:
        content, ok = safe_replace(
            content,
            "**END OF CHAPTER 12**",
            dispatch_addendum + "**END OF CHAPTER 12**"
        )
        if ok: patches_applied += 1
    elif content.rstrip().endswith("---"):
        # Append at end
        content = content.rstrip() + "\n" + dispatch_addendum
        patches_applied += 1

    write_result(path, content, dry_run)
    results["patched"].append(str(path))
    print(f"  ✅ Applied {patches_applied} patches to: {path}")
    return True


# ═══════════════════════════════════════════════════════════════
# FILE 6: CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK
# ═══════════════════════════════════════════════════════════════
def patch_ch10_agent(dry_run=False):
    label = "CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK"
    print(f"\n{'='*60}")
    print(f"[6/6] {label}")
    print(f"{'='*60}")

    path = find_file("CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md")
    if not path:
        print(f"  ❌ File not found")
        results["errors"].append(label)
        return False

    backup_file(path)
    print(f"  Backed up: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    patches_applied = 0

    # Patch 1: Version and date
    content, ok = safe_replace(
        content,
        "**Version:** 3.0.0 (Validated Working System)  \n**Date:** January 8, 2026",
        "**Version:** 3.1.0 (Full Autonomous Operation)  \n**Date:** February 3, 2026"
    )
    if ok: patches_applied += 1

    # Patch 2: Status line
    content, ok = safe_replace(
        content,
        "**Status:** ✅ LLM Integration Verified Working",
        "**Status:** ✅ Full Autonomous Operation — Phase 7 Complete"
    )
    if ok: patches_applied += 1

    # Patch 3: Component table — add new components
    old_components = """| Component | File | Status |
|-----------|------|--------|
| Watcher Agent | `agents/watcher_agent.py` | ✅ v1.1.0 Working |
| LLM Router | `llm_services/llm_router.py` | ✅ v2.0.0 Working |
| Grammar Loader | `llm_services/grammar_loader.py` | ✅ v1.0.0 Working |
| Server Startup | `llm_services/start_llm_servers.sh` | ✅ v2.0.0 Working |
| Step Contexts | `agents/contexts/*.py` | ✅ All 6 steps |
| Doctrine | `agents/doctrine.py` | ✅ v3.2.0 Working |
| Prompt Builder | `agents/prompt_builder.py` | ✅ v3.2.0 Working |"""

    new_components = """| Component | File | Status |
|-----------|------|--------|
| Watcher Agent | `agents/watcher_agent.py` | ✅ v1.4.0 Working |
| Watcher Dispatch | `agents/watcher_dispatch.py` | ✅ v1.0.0 Working (Session 58) |
| Bundle Factory | `agents/contexts/bundle_factory.py` | ✅ v1.0.0 Working (Session 58) |
| LLM Lifecycle | `llm_services/llm_lifecycle.py` | ✅ v1.0.0 Working (Session 57) |
| LLM Router | `llm_services/llm_router.py` | ✅ v2.0.0 Working |
| Grammar Loader | `llm_services/grammar_loader.py` | ✅ v1.0.0 Working |
| Server Startup | `llm_services/start_llm_servers.sh` | ✅ v2.1.0 Working |
| Step Contexts | `agents/contexts/*.py` | ✅ All 6 steps + Chapter 13 |
| Doctrine | `agents/doctrine.py` | ✅ v3.2.0 Working |
| Prompt Builder | `agents/prompt_builder.py` | ✅ v3.2.0 Working |
| GBNF Grammars | `agent_grammars/*.gbnf` | ✅ v1.1 (4 files, fixed Session 59) |"""

    content, ok = safe_replace(content, old_components, new_components)
    if ok: patches_applied += 1

    # Patch 4: Architecture diagram version
    content, ok = safe_replace(
        content,
        "AUTONOMOUS AGENT FRAMEWORK v3.0.0",
        "AUTONOMOUS AGENT FRAMEWORK v3.1.0"
    )
    if ok: patches_applied += 1

    # Patch 5: File inventory at bottom
    old_inventory = """| File | Purpose |
|------|---------|
| `agents/watcher_agent.py` | Main orchestrator |
| `agents/doctrine.py` | Decision framework |
| `agents/contexts/*.py` | Step-specific contexts |
| `llm_services/llm_router.py` | LLM routing logic |
| `llm_services/grammar_loader.py` | Grammar management |
| `llm_services/start_llm_servers.sh` | Server startup |
| `agent_manifests/*.json` | Step configurations |"""

    new_inventory = """| File | Purpose |
|------|---------|
| `agents/watcher_agent.py` | Main orchestrator |
| `agents/watcher_dispatch.py` | Dispatch: selfplay, learning loop, Ch13 requests |
| `agents/contexts/bundle_factory.py` | Unified LLM context assembly (7 bundle types) |
| `agents/doctrine.py` | Decision framework |
| `agents/contexts/*.py` | Step-specific contexts |
| `llm_services/llm_router.py` | LLM routing logic |
| `llm_services/llm_lifecycle.py` | LLM server lifecycle management |
| `llm_services/grammar_loader.py` | Grammar management |
| `llm_services/start_llm_servers.sh` | Server startup (v2.1.0) |
| `agent_manifests/*.json` | Step configurations |
| `agent_grammars/*.gbnf` | Grammar constraint files (v1.1) |"""

    content, ok = safe_replace(content, old_inventory, new_inventory)
    if ok: patches_applied += 1

    write_result(path, content, dry_run)
    results["patched"].append(str(path))
    print(f"  ✅ Applied {patches_applied} patches to: {path}")
    return True


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Fix stale documentation (Session 60)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    args = parser.parse_args()

    print("=" * 60)
    print("  DOCUMENTATION STALENESS FIXER — Session 60")
    print("  Phase 7 WATCHER Integration: Complete")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print(f"  Project: {PROJECT_ROOT}")
    print(f"  Docs: {DOCS_DIR}")
    print(f"  Staging: {STAGING_DIR}")
    print("=" * 60)

    # Verify we're in the right place
    if not PROJECT_ROOT.exists():
        print(f"\n❌ Project root not found: {PROJECT_ROOT}")
        print("   Run this from Zeus with ~/distributed_prng_analysis/ present.")
        sys.exit(1)

    # Ensure docs dir exists
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # Run all patches
    patch_ch13_progress(args.dry_run)
    patch_section_19(args.dry_run)
    patch_readme(args.dry_run)
    patch_canonical(args.dry_run)
    patch_ch12_watcher(args.dry_run)
    patch_ch10_agent(args.dry_run)

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Created:  {len(results['created'])} files")
    for f in results['created']:
        print(f"    + {f}")
    print(f"  Patched:  {len(results['patched'])} files")
    for f in results['patched']:
        print(f"    ~ {f}")
    print(f"  Skipped:  {len(results['skipped'])} files")
    for f in results['skipped']:
        print(f"    ? {f}")
    print(f"  Errors:   {len(results['errors'])} files")
    for f in results['errors']:
        print(f"    ✗ {f}")

    if not args.dry_run:
        print(f"\n  📁 Staging directory: {STAGING_DIR}")
        print(f"     Contains copies of all updated files for scp to ser8:")
        print(f"")
        print(f"     scp {STAGING_DIR}/*.md ser8:~/Downloads/CONCISE_OPERATING_GUIDE_v1.0/")
        print(f"")
        print(f"  Git commands:")
        print(f"     cd ~/distributed_prng_analysis")
        print(f"     git add docs/CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_0.md")
        print(f"     git add docs/CHAPTER_13_SECTION_19_UPDATED.md")
        print(f"     git add README.md")
        if find_file("CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md"):
            print(f"     git add docs/CANONICAL_PIPELINE_AND_CH13_WITH_STARTUP_COMPLETE.md")
        if find_file("CHAPTER_12_WATCHER_AGENT.md"):
            print(f"     git add docs/CHAPTER_12_WATCHER_AGENT.md")
        if find_file("CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md"):
            print(f"     git add docs/CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md")
        print(f'     git commit -m "docs: Update 6 stale documents for Phase 7 completion (Session 60)"')
        print(f"     git push origin main")

        # Also suggest removing old v2 if it exists
        old_v2 = find_file("CHAPTER_13_IMPLEMENTATION_PROGRESS_v2_0.md")
        if old_v2:
            print(f"\n  Optional cleanup:")
            print(f"     rm {old_v2}  # Superseded by v3.0")
            print(f"     git add -u && git commit -m 'docs: Remove superseded v2.0 progress tracker'")

    print(f"\n{'='*60}")
    print("  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
