# TODO: Phase 7 — WATCHER Integration (REVISED v2)

**Created:** 2026-01-30  
**Revised:** 2026-02-02 (Parts A & C complete, Part B0 added)  
**Status:** Parts A & C Complete → Part B0 In Progress → Parts B & D Remain  
**Goal:** Complete autonomous operation pipeline  
**Authority:** Joint Alpha + Beta sequencing decision (Session 57)

---

## ⚠️ Revision Notice (v2 — 2026-02-02)

**Sequencing decision locked:** Bundle factory + Addendum A spec FIRST, then wire dispatch against it.

**Rationale (Team Beta):**
1. Dispatch wiring should call a stable abstraction from day one
2. That abstraction is `build_step_awareness_bundle()`
3. Today it wraps existing static context builders
4. Tomorrow it adds structured retrieval
5. Dispatch code never changes

**Required guardrails for dispatch code:**
- **Guardrail #1:** Single context entry point — dispatch calls `build_llm_context()`, nothing else
- **Guardrail #2:** No baked-in token assumptions — bundle_factory owns prompt structure

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
| LLM Lifecycle Manager | `llm_services/llm_lifecycle.py` | ~8KB | ✅ Complete (Session 57) |
| 4 GBNF Grammar Files | `agent_grammars/*.gbnf` | ~6KB | ✅ Complete (Session 57) |
| Context Window 32K | `llm_services/configs/*.json` | — | ✅ Complete (Session 57) |

**Total existing code:** ~400KB+

---

## What's MISSING (Actual Work)

### Part A: Selfplay Validation Testing — ✅ COMPLETE (2026-02-01)

| # | Task | Command | Status | Evidence |
|---|------|---------|--------|----------|
| A1 | Run multi-episode selfplay | `python3 selfplay_orchestrator.py --survivors survivors_with_scores.json --episodes 5 --policy-conditioned` | ✅ | 8 episodes (5+3), zero crashes |
| A2 | Verify candidate emission | `cat learned_policy_candidate.json` | ✅ | Schema v1.1.0, 3 candidates emitted |
| A3 | Verify policy history archive | `ls -la policy_history/` | ✅ | 3 files accumulated |
| A4 | Test with active policy | Create test policy, re-run | ✅ | Filter: 75,396→47,614; Weight: 46,715 adjusted |
| A5 | Verify telemetry health | `cat telemetry/learning_health_latest.json` | ✅ | 38 models tracked, JSON valid |

**Git:** Commit `c0f5d32`

---

### Part B0: Bundle Factory + Spec — 🔶 IN PROGRESS (2026-02-02)

**Prerequisite for Part B.** Team Alpha + Beta joint decision: spec + skeleton first, then wire dispatch against it.

| # | Task | File | Lines | Status |
|---|------|------|-------|--------|
| B0.1 | `bundle_factory.py` — Pydantic models + assembly engine | `agents/contexts/bundle_factory.py` | ~900 | ✅ Built |
| B0.2 | Addendum A — Step Awareness Bundle spec | `docs/ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` | ~300 | ✅ Built |
| B0.3 | Self-test verification on Zeus | `python3 agents/contexts/bundle_factory.py` | — | 🔲 |
| B0.4 | Git commit + docs update | — | — | 🔲 |

**B0.1 Details — `bundle_factory.py`:**
- `StepAwarenessBundle` Pydantic model (immutable, frozen)
- `TokenBudget` with tiered enforcement (Tier 0/1/2)
- `ProvenanceRecord` with SHA256 hashing
- Step mission statements for all 6 steps + Chapter 13
- Grammar name mapping to `agent_grammars/*.gbnf`
- Per-step guardrails (hard constraints injected into prompt)
- `build_step_awareness_bundle()` — main assembly function
- `render_prompt_from_bundle()` — tiered prompt renderer
- `build_llm_context()` — convenience wrapper (Guardrail #1)
- Retrieval stubs for Track 2 (empty returns, zero dispatch rework when filled)

**B0.2 Details — Addendum A:**
- A1: Bundle Object Model
- A2: Token Budget Policy (default 32K)
- A3: Retrieval Source Hierarchy (what's included, what's excluded)
- A4: Assembly Ownership (controller-only, no LLM pulling)
- A5: Determinism + Immutability Guarantees
- A6: Provenance Requirements
- A7: Step-by-Step Minimum Bundle Contents (Steps 1-6 + Ch.13)
- A8: API Reference
- A9-A10: Version history, explicit exclusions

---

### Part B: WATCHER Dispatch Functions (60-90 min)

**REVISED:** All dispatch functions use `build_llm_context()` per Guardrail #1.

| # | Task | File | Lines | Status |
|---|------|------|-------|--------|
| B1 | Add `dispatch_selfplay()` | `agents/watcher_agent.py` | ~60 | 🔲 |
| B2 | Add `dispatch_learning_loop()` | `agents/watcher_agent.py` | ~50 | 🔲 |
| B3 | Add `process_chapter_13_request()` | `agents/watcher_agent.py` | ~70 | 🔲 |
| B4 | Wire to WATCHER daemon mode | `agents/watcher_agent.py` | ~30 | 🔲 |

**B1 Details — `dispatch_selfplay()`:**
```python
def dispatch_selfplay(self, request: dict) -> bool:
    """Execute selfplay_orchestrator.py with policy conditioning.
    
    Uses build_llm_context() for any pre/post LLM evaluation.
    Manages LLM lifecycle: stop LLM before GPU dispatch, restart after.
    """
    # Pre-dispatch: LLM lifecycle management
    if self.llm_lifecycle:
        self.llm_lifecycle.stop("selfplay dispatch — freeing VRAM")
    
    cmd = [
        "python3", "selfplay_orchestrator.py",
        "--survivors", "survivors_with_scores.json",
        "--episodes", str(request.get("episodes", 5)),
        "--policy-conditioned",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    # Post-dispatch: Evaluate candidate if emitted
    if result.returncode == 0:
        candidate_path = "learned_policy_candidate.json"
        if os.path.exists(candidate_path):
            with open(candidate_path) as f:
                candidate = json.load(f)
            
            # Use bundle factory for LLM evaluation (Guardrail #1)
            prompt, grammar, bundle = build_llm_context(
                step_id=13, is_chapter_13=True,
                results={"candidate": candidate, "source": "selfplay"},
                state_paths=[candidate_path]
            )
            # ... LLM evaluation of candidate
    
    # Post-dispatch: Restart LLM
    if self.llm_lifecycle:
        self.llm_lifecycle.start()
    
    return result.returncode == 0
```

**B2 Details — `dispatch_learning_loop()`:**
```python
def dispatch_learning_loop(self, scope: str = "steps_3_5_6") -> bool:
    """Execute partial pipeline rerun (Steps 3→5→6 or full).
    
    Uses build_llm_context() for per-step evaluation.
    Guardrail #1: no inline context assembly.
    """
    if scope == "steps_3_5_6":
        for step in [3, 5, 6]:
            results = self.run_step(step)
            if results is None:
                return False
            
            # Evaluate using bundle factory (Guardrail #1)
            prompt, grammar, bundle = build_llm_context(
                step_id=step, results=results,
                manifest_path=os.path.join(self.config.manifests_dir, STEP_MANIFESTS[step])
            )
            decision = self._evaluate_step_via_bundle(prompt, grammar)
            if decision.recommended_action != "proceed":
                return False
    elif scope == "full":
        self.run_pipeline(1, 6)
    return True
```

**B3 Details — `process_chapter_13_request()`:**
```python
def process_chapter_13_request(self, request_path: str) -> str:
    """Process a request from watcher_requests/.
    
    Routes to dispatch_selfplay() or dispatch_learning_loop()
    based on request_type. Uses build_llm_context() for evaluation.
    """
    with open(request_path) as f:
        request = json.load(f)
    
    request_type = request.get("request_type")
    
    if request_type == "selfplay_retrain":
        # Validate request using LLM (via bundle factory)
        prompt, grammar, bundle = build_llm_context(
            step_id=13, is_chapter_13=True,
            results=request
        )
        # Acceptance check...
        if self._validate_selfplay_request(request):
            success = self.dispatch_selfplay(request)
            return "COMPLETED" if success else "FAILED"
        return "REJECTED"
    
    elif request_type == "learning_loop":
        success = self.dispatch_learning_loop(request.get("scope", "steps_3_5_6"))
        return "COMPLETED" if success else "FAILED"
    
    else:
        logger.warning(f"Unknown request type: {request_type}")
        return "UNKNOWN_TYPE"
```

**B4 Details — Daemon wiring:**
- Add `watcher_requests/` directory scanning to `run_daemon()`
- Add `--dispatch-selfplay` and `--dispatch-learning-loop` CLI args
- Wire `process_chapter_13_request()` to daemon loop
- Add `--dry-run` flag for testing dispatch without execution

---

### Part C: File Organization — ✅ COMPLETE (2026-01-30)

| # | Task | Action | Status | Evidence |
|---|------|--------|--------|----------|
| C1 | Move GBNF grammar | `mkdir -p agent_grammars && mv chapter_13.gbnf agent_grammars/` | ✅ | Commit 22abd7b |

---

### Part D: Integration Testing (60 min)

| # | Task | Action | Status |
|---|------|--------|--------|
| D1 | Test bundle factory self-test | `cd ~/distributed_prng_analysis && python3 agents/contexts/bundle_factory.py` | 🔲 |
| D2 | Test selfplay dispatch | `python3 agents/watcher_agent.py --dispatch-selfplay --dry-run` | 🔲 |
| D3 | Test learning loop dispatch | `python3 agents/watcher_agent.py --dispatch-learning-loop steps_3_5_6 --dry-run` | 🔲 |
| D4 | Test request processing | Create mock request, process via WATCHER | 🔲 |
| D5 | End-to-end: Chapter 13 → WATCHER → Selfplay | Full flow test | 🔲 |

---

## Estimated Effort (Remaining)

| Part | Tasks | Time | Lines | Status |
|------|-------|------|-------|--------|
| ~~A: Selfplay Testing~~ | ~~5~~ | ~~30-60 min~~ | ~~0~~ | ✅ Complete |
| **B0: Bundle Factory** | **4** | **60-90 min** | **~1200** | **🔶 In Progress** |
| **B: WATCHER Dispatch** | **4** | **60-90 min** | **~210** | **🔲 Next** |
| ~~C: File Organization~~ | ~~1~~ | ~~5 min~~ | ~~0~~ | ✅ Complete |
| **D: Integration Testing** | **5** | **60 min** | **0** | **🔲 After B** |
| **Remaining** | **13** | **~3-4 hours** | **~1410** |  |

---

## Dependency Chain (Updated)

```
Part A (Selfplay Testing) ✅ DONE
       ↓
Part B0 (Bundle Factory + Spec) ← CURRENT SESSION
       ↓
Part B (WATCHER Dispatch Functions) ← NEXT SESSION
       ↓
Part C (File Organization) ✅ DONE
       ↓
Part D (Integration Testing)
```

**Critical path:** B0 → B → D. Nothing else blocks.

---

## Success Criteria

### Part A Complete When:
- [x] Selfplay runs 5+ episodes without error
- [x] Candidates emitted to `learned_policy_candidate.json`
- [x] Telemetry health file updated

### Part B0 Complete When:
- [x] `bundle_factory.py` builds bundles for all 6 steps + Chapter 13
- [x] Addendum A spec locked and versioned
- [ ] Self-test passes on Zeus (all 7 bundle types)
- [ ] Git committed with both files

### Part B Complete When:
- [ ] `dispatch_selfplay()` spawns selfplay_orchestrator.py
- [ ] `dispatch_learning_loop()` runs Steps 3→5→6
- [ ] `process_chapter_13_request()` handles watcher_requests/*.json
- [ ] All dispatch functions use `build_llm_context()` (Guardrail #1 verified)
- [ ] No baked-in token assumptions (Guardrail #2 verified)

### Part D Complete When:
- [ ] Bundle factory self-test passes on Zeus
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

## Files Created/Modified Summary

| File | Action | Location | Part |
|------|--------|----------|------|
| `agents/contexts/bundle_factory.py` | CREATE | `agents/contexts/` | B0 |
| `docs/ADDENDUM_A_STEP_AWARENESS_BUNDLES_v1_0.md` | CREATE | `docs/` | B0 |
| `agents/watcher_agent.py` | MODIFY | `agents/` | B |
| `docs/TODO_PHASE7_WATCHER_INTEGRATION_REVISED.md` | UPDATE | `docs/` | — |

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
**Revised (v1) estimate:** 14 tasks, 180 lines, 1 session  
**Revised (v2) estimate:** 13 tasks, ~1410 lines, 2 sessions (bundle factory adds scope but pays for itself via clean dispatch wiring)

---

## What Is Explicitly NOT Being Built This Phase

Per Team Beta lock (2026-02-02):

| Item | Status | When |
|------|--------|------|
| Vector database / embeddings | ❌ Out of scope | Not planned |
| GPU-resident retrieval services | ❌ Out of scope | Not planned |
| Raw decision log ingestion | ❌ Out of scope | Track 2 (structured only) |
| Authority changes | ❌ Out of scope | Not planned |
| Parameter advisor (Item B) | ⏳ Deferred | Steps 4-6 in production |
| Phase 9B.3 auto-heuristics | ⏳ Deferred | 9B.2 validated first |
| Track 2 retrieval enhancement | ⏳ After dispatch wired | Fills stubs in bundle_factory |

---

**END OF REVISED TODO (v2)**
