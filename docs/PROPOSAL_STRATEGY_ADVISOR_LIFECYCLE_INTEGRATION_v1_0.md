# PROPOSAL: Strategy Advisor — LLM Lifecycle Integration & Heuristic Demotion

**Version:** 1.0.0
**Date:** 2026-02-08
**Status:** PROPOSED — Pending Team Beta Review
**Author:** Claude (Session 67)
**Scope:** `parameter_advisor.py`, `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md`, `watcher_dispatch.py`
**Impact:** Strategy Advisor decision quality; LLM utilization; heuristic fallback policy
**Triggered by:** Integration gap discovery during WATCHER wiring session

---

## Approval Status

| Team | Status | Date |
|------|--------|------|
| Team Alpha | ☐ Proposed | 2026-02-08 |
| Team Beta | ☐ Pending | |
| Michael | ☐ Pending | |

---

## 1. Gap Discovery

During Session 67 WATCHER wiring for the Strategy Advisor, the following integration gap was identified:

**`parameter_advisor.py` (Session 66, ~620 lines) has no awareness of `llm_lifecycle.py` (Session 57, ~380 lines).**

The advisor treats LLM availability as uncertain and implements heuristic fallback as a first-class, co-equal operating mode. This is architecturally incorrect — the lifecycle manager was specifically built to guarantee LLM availability on demand via `ensure_running()`, which starts the server in ~3 seconds from cold.

### Root Cause

`CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md` was authored 2026-02-03. `llm_lifecycle.py` was completed and verified in Sessions 57–59 (2026-02-01 through 2026-02-03). The contract never references the lifecycle manager because both were being developed concurrently, and the contract was finalized before lifecycle verification completed.

This is the same class of integration gap as Session 63's `search_strategy` visibility fix — a component built bottom-up that the governance layer didn't account for.

### Impact

The Strategy Advisor — the component designed to interpret complex, high-dimensional diagnostic data and make PhD-level analytical decisions — silently degrades to simple threshold logic (`if score < 0.3 then RETRAIN`) whenever the LLM server happens to not be running. The system generates extraordinarily rich data from 46 PRNG algorithms, 62+ ML features, survivor pattern analysis, and regime detection. Threshold heuristics discard nearly all of that signal.

With the lifecycle manager available, there is no legitimate reason for the LLM to be unavailable when the advisor needs it. The advisor runs **before** GPU-heavy dispatch, so VRAM is free. The lifecycle can start the server in 3 seconds.

---

## 2. Current Behavior (Incorrect)

```
parameter_advisor.py → generate_recommendation()
    ├── Try LLM path
    │     ├── Build prompt via advisor_bundle.py
    │     ├── Call LLM via llm_router.py
    │     └── If LLM unavailable → fall through
    │
    └── Heuristic path (TREATED AS EQUAL)
          ├── Compute metrics
          ├── Apply threshold rules
          └── Return recommendation
```

No lifecycle management. No escalation. No degraded-state logging. The system silently makes worse decisions without anyone knowing.

---

## 3. Proposed Behavior (Correct)

```
parameter_advisor.py → generate_recommendation()
    │
    ├── Step 1: llm_lifecycle.ensure_running()     ← NEW
    │     └── Server starts in ~3 seconds if cold
    │
    ├── Step 2: Build prompt via advisor_bundle.py
    │
    ├── Step 3: Call DeepSeek via llm_router.py with grammar constraint
    │     ├── Confident response → validate bounds → return
    │     └── Low confidence / ESCALATE → Step 4
    │
    ├── Step 4: Escalate to Claude via llm_router.py  ← NEW
    │     ├── escalation_triggers: UNCERTAIN, LOW_CONFIDENCE,
    │     │   ESCALATE_TO_BACKUP, REQUIRES_DEEP_ANALYSIS
    │     ├── Response → validate bounds → return
    │     └── Claude unreachable → Step 5
    │
    └── Step 5: Heuristic fallback (EMERGENCY ONLY)   ← DEMOTED
          ├── Log WARNING: "DEGRADED_MODE — both LLMs unreachable"
          ├── Apply threshold rules
          └── Tag recommendation: {"mode": "heuristic_degraded"}
```

---

## 4. Specific Code Changes

### 4.1 `parameter_advisor.py` — Lifecycle Integration

**Location:** `generate_recommendation()` method, before LLM call

```python
# BEFORE (current — no lifecycle awareness)
def generate_recommendation(self, state_dir, force=False):
    # ... gate check ...
    metrics = self._compute_metrics(state_dir)
    
    # Try LLM
    try:
        recommendation = self._generate_llm_recommendation(metrics)
    except Exception:
        recommendation = self._generate_heuristic_recommendation(metrics)
    
    return recommendation
```

```python
# AFTER (proposed — lifecycle-aware, escalation-capable)
def generate_recommendation(self, state_dir, force=False):
    # ... gate check ...
    metrics = self._compute_metrics(state_dir)
    
    # Step 1: Ensure LLM is available
    lifecycle = self._get_lifecycle_manager()
    if lifecycle:
        lifecycle.ensure_running()
    
    # Step 2-3: Try DeepSeek (primary)
    try:
        recommendation = self._generate_llm_recommendation(metrics)
        if recommendation.get("focus_confidence", 0) < 0.3:
            logger.info("DeepSeek low confidence (%.2f) — escalating to Claude",
                        recommendation["focus_confidence"])
            raise EscalationRequired("Low confidence triggers backup LLM")
        return recommendation
    except EscalationRequired:
        # Step 4: Escalate to Claude (backup)
        try:
            recommendation = self._generate_llm_recommendation(
                metrics, use_backup=True
            )
            recommendation["advisor_model"] = "claude_backup"
            return recommendation
        except Exception as e:
            logger.warning("Claude escalation failed: %s", e)
    except Exception as e:
        logger.warning("DeepSeek analysis failed: %s — attempting escalation", e)
        # Step 4: Try Claude before giving up
        try:
            recommendation = self._generate_llm_recommendation(
                metrics, use_backup=True
            )
            recommendation["advisor_model"] = "claude_backup"
            return recommendation
        except Exception as e2:
            logger.warning("Claude escalation also failed: %s", e2)
    
    # Step 5: Heuristic — EMERGENCY ONLY
    logger.warning("DEGRADED_MODE — both LLMs unreachable. "
                    "Using heuristic fallback. Decision quality reduced.")
    recommendation = self._generate_heuristic_recommendation(metrics)
    recommendation["mode"] = "heuristic_degraded"
    recommendation["degraded_reason"] = "both_llms_unreachable"
    return recommendation
```

**Estimated change:** ~40 lines modified in `parameter_advisor.py`

### 4.2 `parameter_advisor.py` — Lifecycle Helper

```python
def _get_lifecycle_manager(self):
    """Get LLM lifecycle manager (lazy import to avoid circular deps)."""
    try:
        from llm_services.llm_lifecycle import get_lifecycle_manager
        return get_lifecycle_manager()
    except ImportError:
        logger.debug("llm_lifecycle not available — LLM may not be manageable")
        return None
```

**Estimated change:** ~8 lines added

### 4.3 `parameter_advisor.py` — Backup LLM Support

Add `use_backup` parameter to `_generate_llm_recommendation()`:

```python
def _generate_llm_recommendation(self, metrics, use_backup=False):
    """Generate recommendation via LLM (primary or backup)."""
    bundle = self._build_advisor_bundle(metrics)
    
    if use_backup:
        # Route to Claude via llm_router escalation
        response = self.llm_router.evaluate(
            prompt=bundle,
            grammar_file="strategy_advisor.gbnf",
            force_backup=True
        )
    else:
        response = self.llm_router.evaluate(
            prompt=bundle,
            grammar_file="strategy_advisor.gbnf"
        )
    
    return self._parse_and_validate(response)
```

**Estimated change:** ~15 lines modified

### 4.4 `watcher_dispatch.py` — Advisor Call Before Selfplay

**Location:** `process_chapter_13_request()`, line ~462, after validation passes, before `dispatch_selfplay()`

```python
    # ── Strategy Advisor enrichment (before dispatch) ────────
    if request_type == "selfplay_retrain":
        try:
            from parameter_advisor import StrategyAdvisor
            advisor = StrategyAdvisor()
            rec = advisor.generate_recommendation(
                state_dir=os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), ".."
                )
            )
            if rec and rec.get("selfplay_overrides"):
                overrides = rec["selfplay_overrides"]
                logger.info(f"[{run_id}] Strategy Advisor recommends: "
                            f"focus={rec.get('focus_area')}, "
                            f"action={rec.get('recommended_action')}, "
                            f"mode={rec.get('mode', 'llm')}")
                # Merge overrides into request
                request.setdefault("selfplay_overrides", {}).update(overrides)
            elif rec:
                logger.info(f"[{run_id}] Strategy Advisor: no selfplay overrides "
                            f"(action={rec.get('recommended_action')})")
        except Exception as e:
            logger.warning(f"[{run_id}] Strategy Advisor failed: {e} — "
                           f"proceeding without strategic guidance")
```

**Estimated change:** ~25 lines added

### 4.5 `watcher_dispatch.py` — Pass Overrides to Orchestrator

**Location:** `dispatch_selfplay()`, line ~143, where `cmd` list is built

```python
    # Current
    cmd = [
        sys.executable, "selfplay_orchestrator.py",
        "--survivors", survivors_file,
        "--episodes", str(episodes),
        "--policy-conditioned",
    ]
    
    # Proposed addition
    overrides = request.get("selfplay_overrides", {})
    if overrides.get("min_fitness_threshold"):
        cmd.extend(["--min-fitness", str(overrides["min_fitness_threshold"])])
    if overrides.get("max_episodes"):
        episodes = overrides["max_episodes"]  # Override from advisor
        cmd[cmd.index("--episodes") + 1] = str(episodes)
```

**Estimated change:** ~10 lines added

---

## 5. Contract Update

Add the following section to `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md` after Section 8 (Implementation Plan):

### Section 8.5: LLM Lifecycle Dependency

```
The Strategy Advisor MUST use llm_lifecycle.ensure_running() before 
any LLM analysis call. The lifecycle manager (llm_services/llm_lifecycle.py, 
Session 57) guarantees LLM availability in ~3 seconds from cold state.

Decision hierarchy:
  1. DeepSeek (primary) — routine analysis, grammar-constrained
  2. Claude (backup) — escalation on low confidence or DeepSeek failure
  3. Heuristic (emergency) — ONLY when both LLMs unreachable
     MUST log as DEGRADED_MODE warning
     MUST tag recommendation with mode=heuristic_degraded

The advisor runs BEFORE GPU-heavy dispatch phases, so VRAM is available 
for the LLM server. There is no legitimate VRAM contention at the 
advisor's decision point.

Lifecycle stop() is NOT called by the advisor. The subsequent 
dispatch_selfplay() call handles LLM shutdown when GPU VRAM is needed 
for compute.
```

---

## 6. Questions for Team Beta

### Q1: Lifecycle Stop Responsibility

Should the advisor call `lifecycle.stop()` after obtaining its recommendation, or leave the server running?

**Recommendation:** Leave it running. The WATCHER's `dispatch_selfplay()` already calls `lifecycle.stop()` before spawning the selfplay orchestrator (line 133 of `watcher_dispatch.py`). Adding a stop/restart cycle between advisor and dispatch wastes 3 seconds for no benefit.

### Q2: Confidence Threshold for Escalation

The proposal uses `focus_confidence < 0.3` as the escalation trigger from DeepSeek to Claude. Is this threshold correct?

**Context:** The `strategy_advisor.gbnf` grammar allows `focus_confidence` values from 0.0 to 1.0. Below 0.3 means DeepSeek is essentially uncertain about which focus area applies. Above 0.3, DeepSeek's classification is likely usable even if not maximally confident.

### Q3: Claude Backup Method

The `llm_router.py` already has escalation triggers (`UNCERTAIN`, `LOW_CONFIDENCE`, `ESCALATE_TO_BACKUP`, `REQUIRES_DEEP_ANALYSIS`) and is configured to route to Claude Code CLI. Should the advisor:

- **Option A:** Use `llm_router.evaluate(force_backup=True)` — explicit escalation
- **Option B:** Return an `ESCALATE_TO_BACKUP` signal and let the router handle it
- **Option C:** Write a request to `watcher_requests/` for human-triggered Claude consultation

**Recommendation:** Option A for automated operation. Option C remains available via the existing `ESCALATE` action in the recommendation schema for truly ambiguous situations.

### Q4: Heuristic Degraded Mode — Halt or Continue?

When both LLMs are unreachable and the advisor falls back to heuristics, should dispatch:

- **Option A:** Proceed with heuristic recommendation (current implicit behavior)
- **Option B:** Log warning and proceed, but tag the dispatch as degraded
- **Option C:** Halt dispatch entirely and wait for LLM recovery

**Recommendation:** Option B. Stopping dispatch entirely would stall the pipeline. But the degraded tag lets downstream components (Chapter 13, WATCHER logs) know the decision was made without LLM intelligence.

---

## 7. Files Affected

| File | Change Type | Est. Lines |
|------|------------|------------|
| `parameter_advisor.py` | Modify | ~65 |
| `agents/watcher_dispatch.py` | Add | ~35 |
| `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md` | Add Section 8.5 | ~25 |
| `CHAPTER_13_IMPLEMENTATION_PROGRESS_v3_3.md` | Update | ~10 |

**Total estimated effort:** ~135 lines, ~1 hour implementation + testing

---

## 8. Verification Plan

### Test 1: Lifecycle Integration

```bash
# Ensure LLM is stopped
pkill -f 'llama-server' 2>/dev/null; sleep 3

# Run advisor — should auto-start LLM
PYTHONPATH=. python3 parameter_advisor.py --state-dir . --force --verbose

# Expected: "LLM server healthy after ~3s" then grammar-constrained recommendation
# NOT: silent heuristic fallback
```

### Test 2: Degraded Mode Logging

```bash
# Block LLM startup (rename binary temporarily)
mv llama.cpp/llama-server llama.cpp/llama-server.bak

# Run advisor — should log DEGRADED_MODE
PYTHONPATH=. python3 parameter_advisor.py --state-dir . --force --verbose

# Expected: WARNING "DEGRADED_MODE — both LLMs unreachable"
# Expected: recommendation contains "mode": "heuristic_degraded"

# Restore
mv llama.cpp/llama-server.bak llama.cpp/llama-server
```

### Test 3: End-to-End Dispatch

```bash
# Start LLM, create test request
./llm_services/start_llm_servers.sh

cat > watcher_requests/test_advisor_e2e.json << 'EOF'
{
    "request_type": "selfplay_retrain",
    "source": "advisor_integration_test",
    "episodes": 3,
    "reason": "Test advisor enrichment before dispatch"
}
EOF

PYTHONPATH=. python3 agents/watcher_agent.py --process-requests

# Expected: Logs show "Strategy Advisor recommends: focus=X, action=Y"
# Expected: Selfplay dispatched with enriched parameters
```

---

## 9. Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| `ensure_running()` adds 3s latency | Certain | Negligible | One-time cost per cycle; dwarfed by minutes-long selfplay |
| Claude backup unavailable (no internet) | Low | Low | Heuristic degraded mode still functions; logged for review |
| Escalation creates loop (DeepSeek→Claude→DeepSeek) | None | N/A | Code uses explicit `use_backup=True`, no re-routing |
| Lifecycle import fails on older Zeus install | Very Low | Low | Lazy import with try/except; degrades gracefully |

---

## 10. Precedent

This proposal follows the same pattern as `PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md` (Session 63):

- **Discovery:** Integration gap between execution layer and governance layer
- **Root cause:** Concurrent development without cross-referencing
- **Fix:** Surgical — wire existing components together, update contract documentation
- **Scope:** No new architecture, no new dependencies, no execution code changes

---

## 11. Summary

The Strategy Advisor was built to make intelligent decisions about complex data. With the lifecycle manager already operational, there is no reason it should ever silently degrade to threshold logic. This proposal wires two existing, tested components together and establishes the correct decision hierarchy: DeepSeek first, Claude backup, heuristic emergency-only.

**One-sentence version:** Make the advisor use the LLM that's already there instead of pretending it might not be.

---

**END OF PROPOSAL**
