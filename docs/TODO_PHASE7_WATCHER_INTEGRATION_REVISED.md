# TODO: Phase 7 — WATCHER Integration (REVISED)

**Created:** 2026-01-30  
**Revised:** 2026-01-30 (after documentation audit)  
**Status:** Planning  
**Goal:** Complete autonomous operation pipeline

---

## ⚠️ Revision Notice

**Original TODO was over-scoped.** Documentation audit revealed Phases 1-6 of Chapter 13 are COMPLETE (code exists since January 12).

**Actual Gap:** WATCHER does not yet dispatch Chapter 13 or Selfplay.

---

## What EXISTS (No Work Needed)

| Component | File | Size | Status |
|-----------|------|------|--------|
| Diagnostics Engine | `chapter_13_diagnostics.py` | 39KB | ✅ Complete |
| LLM Advisor | `chapter_13_llm_advisor.py` | 23KB | ✅ Complete |
| Triggers Engine | `chapter_13_triggers.py` | 36KB | ✅ Complete |
| Acceptance Engine | `chapter_13_acceptance.py` | 41KB | ✅ Complete |
| Orchestrator | `chapter_13_orchestrator.py` | 23KB | ✅ Complete |
| Proposal Schema | `llm_proposal_schema.py` | 14KB | ✅ Complete |
| GBNF Grammar | `chapter_13.gbnf` | 2.9KB | ✅ Complete |
| Draw Ingestion | `draw_ingestion_daemon.py` | 22KB | ✅ Complete |
| Synthetic Injector | `synthetic_draw_injector.py` | 20KB | ✅ Complete |
| Policies Config | `watcher_policies.json` | 4.7KB | ✅ Complete |
| Selfplay Orchestrator | `selfplay_orchestrator.py` | 43KB | ✅ Complete |
| Policy Transform | `policy_transform.py` | 36KB | ✅ Complete |
| Policy Conditioning | `policy_conditioned_episode.py` | 25KB | ✅ Complete |

**Total existing code:** ~370KB

---

## What's MISSING (Actual Work)

### Part A: Selfplay Validation Testing (30-60 min)

| # | Task | Command | Status |
|---|------|---------|--------|
| A1 | Run multi-episode selfplay | `python3 selfplay_orchestrator.py --survivors survivors_with_scores.json --episodes 5 --policy-conditioned` | 🔲 |
| A2 | Verify candidate emission | `cat learned_policy_candidate.json` | 🔲 |
| A3 | Verify policy history archive | `ls -la policy_history/` | 🔲 |
| A4 | Test with active policy | Create test policy, re-run | 🔲 |
| A5 | Verify telemetry health | `cat telemetry/learning_health_latest.json` | 🔲 |

### Part B: WATCHER Dispatch Functions (60-90 min)

| # | Task | File | Lines | Status |
|---|------|------|-------|--------|
| B1 | Add `dispatch_selfplay()` | `agents/watcher_agent.py` | ~50 | 🔲 |
| B2 | Add `dispatch_learning_loop()` | `agents/watcher_agent.py` | ~40 | 🔲 |
| B3 | Add `process_chapter_13_request()` | `agents/watcher_agent.py` | ~60 | 🔲 |
| B4 | Wire to WATCHER daemon mode | `agents/watcher_agent.py` | ~30 | 🔲 |

**B1 Details — `dispatch_selfplay()`:**
```python
def dispatch_selfplay(self, request: dict) -> bool:
    """Execute selfplay_orchestrator.py with policy conditioning."""
    cmd = [
        "python3", "selfplay_orchestrator.py",
        "--survivors", "survivors_with_scores.json",
        "--episodes", str(request.get("episodes", 5)),
        "--policy-conditioned",
        "--project-root", self.project_root,
    ]
    # subprocess.run with logging
```

**B2 Details — `dispatch_learning_loop()`:**
```python
def dispatch_learning_loop(self, scope: str = "steps_3_5_6") -> bool:
    """Execute partial pipeline rerun (Steps 3→5→6)."""
    if scope == "steps_3_5_6":
        self._run_step(3)  # Refresh labels
        self._run_step(5)  # Retrain model
        self._run_step(6)  # Generate predictions
    elif scope == "full":
        self._run_pipeline(1, 6)
```

**B3 Details — `process_chapter_13_request()`:**
```python
def process_chapter_13_request(self, request_path: str) -> str:
    """Process a request from watcher_requests/."""
    request = load_json(request_path)
    
    if request["request_type"] == "selfplay_retrain":
        if self._validate_selfplay_request(request):
            return self.dispatch_selfplay(request)
        return "REJECTED"
    
    elif request["request_type"] == "learning_loop":
        return self.dispatch_learning_loop(request["scope"])
```

### Part C: File Organization (5 min)

| # | Task | Action | Status |
|---|------|--------|--------|
| C1 | Move GBNF grammar | `mkdir -p agent_grammars && mv chapter_13.gbnf agent_grammars/` | 🔲 |

### Part D: Integration Testing (60 min)

| # | Task | Action | Status |
|---|------|--------|--------|
| D1 | Test selfplay dispatch | `python3 agents/watcher_agent.py --dispatch-selfplay --dry-run` | 🔲 |
| D2 | Test learning loop dispatch | `python3 agents/watcher_agent.py --dispatch-learning-loop steps_3_5_6 --dry-run` | 🔲 |
| D3 | Test request processing | Create mock request, process via WATCHER | 🔲 |
| D4 | End-to-end: Chapter 13 → WATCHER → Selfplay | Full flow test | 🔲 |

---

## Estimated Effort

| Part | Tasks | Time | Lines |
|------|-------|------|-------|
| A: Selfplay Testing | 5 | 30-60 min | 0 |
| B: WATCHER Dispatch | 4 | 60-90 min | ~180 |
| C: File Organization | 1 | 5 min | 0 |
| D: Integration Testing | 4 | 60 min | 0 |
| **Total** | **14** | **~3 hours** | **~180** |

---

## Dependency Chain

```
Part A (Selfplay Testing)
       ↓
Part B (WATCHER Dispatch Functions)
       ↓
Part C (File Organization)
       ↓
Part D (Integration Testing)
```

---

## Success Criteria

### Part A Complete When:
- [ ] Selfplay runs 5+ episodes without error
- [ ] Candidates emitted to `learned_policy_candidate.json`
- [ ] Telemetry health file updated

### Part B Complete When:
- [ ] `dispatch_selfplay()` spawns selfplay_orchestrator.py
- [ ] `dispatch_learning_loop()` runs Steps 3→5→6
- [ ] `process_chapter_13_request()` handles watcher_requests/*.json

### Part D Complete When:
- [ ] End-to-end flow works without human intervention
- [ ] Audit trail shows WATCHER → Selfplay dispatch
- [ ] Request → Approval → Execution cycle verified

### Full Autonomy Achieved When:
```
Chapter 13 Triggers → watcher_requests/ → WATCHER → Selfplay/Learning Loop
       ↑                                                        ↓
       └──────────────── Diagnostics ← Reality ←────────────────┘
```

No human in the loop for routine decisions.

---

## Files Modified Summary

| File | Action | Location |
|------|--------|----------|
| `agents/watcher_agent.py` | MODIFY | Add dispatch functions |
| `chapter_13.gbnf` | MOVE | → `agent_grammars/chapter_13.gbnf` |

**Note:** No new files needed. Only ~180 lines of additions to existing WATCHER.

---

## What Was REMOVED from Original TODO

| Original Task | Reason Removed |
|---------------|----------------|
| B1: `llm_proposal_schema.py` | Already exists (14KB) |
| B2: `chapter_13.gbnf` | Already exists (2.9KB) |
| B3: Parameter vocabulary extraction | Already in advisor |
| B4-B6: Diagnostics engine | Already exists (39KB) |
| B7-B10: LLM Advisor | Already exists (23KB) |
| B11: `validate_proposal()` | Already in acceptance.py |
| B12: `apply_parameter_changes()` | Already in acceptance.py |

**Original estimate:** 27 tasks, 630 lines, 2-3 sessions  
**Revised estimate:** 14 tasks, 180 lines, 1 session

---

**END OF REVISED TODO**
