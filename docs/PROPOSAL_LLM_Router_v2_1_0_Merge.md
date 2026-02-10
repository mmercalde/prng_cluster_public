# PROPOSAL: LLM Router v2.1.0 — API Restoration + Missing Method

**Document Version:** 1.0  
**Date:** January 9, 2026  
**Author:** Claude (AI Assistant) + Michael  
**Status:** DRAFT - Pending Team Beta Review  
**Target:** `llm_services/llm_router.py`  
**Priority:** CRITICAL — Blocks Watcher Agent LLM evaluation

---

## Executive Summary

The Watcher Agent's `--run-pipeline` with LLM evaluation is broken due to a missing method (`evaluate_watcher_decision()`) that was never implemented, combined with a January 7 refactor that stripped essential API methods from the LLM Router.

This proposal merges the best of both versions:
- **Keep v2.0.0:** DeepSeek-R1-14B primary + Claude Opus backup (faster, simpler routing)
- **Restore v1.0.5:** Full method API needed by agent framework
- **Add new:** `evaluate_watcher_decision()` — the method watcher actually needs

---

## Problem Statement

### Symptom
```
$ PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2

Parse errors: ['Direct parse: Expecting value: line 1 column 1 (char 0)']
Action: ESCALATE
Reason: Partial parse from malformed response
```

### Root Cause Analysis

| Issue | Details |
|-------|---------|
| **Missing method** | `watcher_agent.py` line 359 calls `self.llm_router.evaluate_watcher_decision()` — this method **never existed** |
| **Stripped API** | Jan 7 refactor (commit `07bfd79`) removed `orchestrate()`, `calculate()`, `generate_json()` — needed by agent framework |
| **No integration test** | Jan 7 validated with "A/B testing: 5/5 sections" but didn't test watcher integration |

### Timeline

```
Dec 6, 2025:  v1.0.5 deployed — full API (orchestrate, calculate, generate_json)
              watcher_agent.py written expecting router.evaluate_decision()
              
Dec 7, 2025:  Watcher tested with evaluate_decision() — worked
              (Method existed in development, never committed?)

Jan 3, 2026:  Watcher tested with --evaluate (heuristic mode) — passed
              LLM evaluation path not tested

Jan 7, 2026:  v2.0.0 refactor — stripped to 186 lines
              Kept: route(), _call_primary(), _call_backup()
              Removed: orchestrate(), calculate(), generate_json(), logging, etc.
              Reason: "Parsing was fragile" + simplified architecture

Jan 9, 2026:  --run-pipeline with LLM fails
              evaluate_watcher_decision() not found
              Root cause identified
```

---

## Proposed Solution

### v2.1.0 Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM ROUTER v2.1.0 — MERGED                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  FROM v2.0.0 (Keep):                FROM v1.0.5 (Restore):              │
│  ├─ DeepSeek-R1-14B primary         ├─ orchestrate()                    │
│  ├─ Claude Opus backup              ├─ calculate()                      │
│  ├─ Escalation triggers             ├─ generate_json()                  │
│  ├─ route() core method             ├─ set_agent()                      │
│  ├─ _call_primary()                 ├─ _setup_logging()                 │
│  ├─ _call_backup()                  ├─ _add_agent_header()              │
│  ├─ health_check()                  ├─ _check_context_reset()           │
│  ├─ get_llm_metadata()              └─ _log_request()                   │
│  └─ LLMMetrics dataclass                                                │
│                                                                         │
│  NEW (Add):                                                             │
│  └─ evaluate_watcher_decision()  ◄── What watcher actually needs        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Method Inventory

| Method | Source | Purpose | Used By |
|--------|--------|---------|---------|
| `route()` | v2.0.0 | Core routing with escalation | All |
| `_call_primary()` | v2.0.0 | DeepSeek-R1 via llama.cpp | Internal |
| `_call_backup()` | v2.0.0 | Claude Opus via CLI | Internal |
| `orchestrate()` | v1.0.5 | Force primary for planning | Agents |
| `calculate()` | v1.0.5 | Force math tasks | Step 4, 5 |
| `generate_json()` | v1.0.5 | JSON generation + parsing | Agents |
| `evaluate_watcher_decision()` | **NEW** | Watcher decision JSON | Watcher |
| `set_agent()` | v1.0.5 | Agent identity headers | Logging |
| `_setup_logging()` | v1.0.5 | Rotating log files | Debug |
| `_add_agent_header()` | v1.0.5 | Prompt headers | Tracing |
| `_check_context_reset()` | v1.0.5 | KV cache management | Long runs |
| `_log_request()` | v1.0.5 | Request/response logging | Audit |
| `health_check()` | v2.0.0 | Endpoint health | Monitoring |
| `get_llm_metadata()` | v2.0.0 | Schema metadata | agent_metadata |
| `reset_metrics()` | v2.0.0 | Clear metrics | New runs |

### New Method: `evaluate_watcher_decision()`

```python
def evaluate_watcher_decision(
    self, 
    prompt: str, 
    step_id: str = None, 
    agent: str = None,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Evaluate watcher decision with guaranteed JSON output.
    
    Added: v2.1.0 (January 9, 2026)
    Reason: watcher_agent.py line 359 requires this method
    
    Args:
        prompt: Full context prompt from watcher
        step_id: Pipeline step identifier (e.g., "step1_window_optimizer")
        agent: Agent name for logging (e.g., "watcher_step1_v2")
        temperature: LLM temperature (default 0.3 for consistency)
    
    Returns:
        Dict with keys: decision, confidence, reasoning, checks
        
    Raises:
        json.JSONDecodeError: If response cannot be parsed as JSON
    """
    # Add agent header
    if agent:
        self.set_agent(agent)
    
    # Wrap prompt with JSON instruction
    json_prompt = f"""<step: {step_id or 'unknown'}>

{prompt}

IMPORTANT: Respond with ONLY valid JSON. No markdown, no explanation.
Required format:
{{
    "decision": "proceed" | "retry" | "escalate",
    "confidence": 0.0-1.0,
    "reasoning": "explanation string",
    "checks": {{"used_rates": true/false, "mentioned_data_source": true/false}}
}}"""

    # Get response with low temperature for consistency
    response = self.route(json_prompt, temperature=temperature)
    
    # Parse JSON with fallback cleaning
    return self._parse_json_response(response)


def _parse_json_response(self, response: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response with cleaning fallbacks.
    
    Handles:
    - Raw JSON
    - Markdown code blocks (```json ... ```)
    - Leading/trailing whitespace
    - Embedded JSON in text
    """
    import re
    
    response = response.strip()
    
    # Try direct parse first
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    if "```" in response:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
    
    # Try extracting first {...} block
    match = re.search(r'\{[\s\S]*\}', response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # All parsing failed
    raise json.JSONDecodeError(
        f"Could not parse JSON from response: {response[:200]}...",
        response, 0
    )
```

---

## File Changes

| File | Action | Lines |
|------|--------|-------|
| `llm_services/llm_router.py` | Replace | ~350 (was 186) |
| `llm_services/llm_router_v2.0.0_backup.py` | Create | Backup current before merge |

### Estimated Line Count

```
v2.0.0 core (keep):          ~120 lines
v1.0.5 methods (restore):    ~100 lines
New methods (add):           ~80 lines
Imports/docstrings:          ~50 lines
─────────────────────────────────────────
Total v2.1.0:                ~350 lines
```

---

## Testing Plan

### Unit Tests

```bash
# 1. Health check
python3 -c "
from llm_services.llm_router import get_router
router = get_router()
print(router.health_check())
"

# 2. Basic route
python3 -c "
from llm_services.llm_router import get_router
router = get_router()
print(router.route('What is 2+2?')[:100])
"

# 3. evaluate_watcher_decision (NEW)
python3 -c "
from llm_services.llm_router import get_router
router = get_router()
result = router.evaluate_watcher_decision(
    'Survivors: 101592, confidence: 0.95',
    step_id='step1_window_optimizer',
    agent='test'
)
print(result)
"

# 4. generate_json (RESTORED)
python3 -c "
from llm_services.llm_router import get_router
router = get_router()
result = router.generate_json('A person with name Alice and age 30')
print(result)
"
```

### Integration Test

```bash
# Watcher with LLM (the failing case)
python3 -m agents.watcher_agent --clear-halt
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2
```

### Regression Test

```bash
# Ensure heuristic mode still works
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 2 --end-step 2 --no-llm
```

---

## Rollback Plan

If v2.1.0 causes issues:

```bash
# Restore v2.0.0
cp llm_services/llm_router_v2.0.0_backup.py llm_services/llm_router.py

# Use heuristic mode until fixed
python3 agents/watcher_agent.py --run-pipeline --no-llm
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| JSON parsing still fragile | Medium | Medium | Multiple fallback parsers |
| DeepSeek-R1 format incompatible | Low | High | Test before deploy |
| Claude backup integration breaks | Low | Medium | Backup preserved in v2.0.0 |
| Performance regression | Low | Low | Same core routing logic |

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Claude (AI) | 2026-01-09 | ✓ |
| Co-Author | Michael | 2026-01-09 | |
| Team Beta Review | | | |
| Final Approval | | | |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-07 | DeepSeek-R1 + Claude Opus (stripped API) |
| 2.1.0 | 2026-01-09 | Restore v1.0.5 API + add evaluate_watcher_decision() |

---

## Next Steps

1. **Team Beta Review** — Approve this proposal
2. **Create backup** — `cp llm_router.py llm_router_v2.0.0_backup.py`
3. **Implement merge** — Create v2.1.0 with all methods
4. **Test** — Run unit + integration tests
5. **Deploy** — Replace llm_router.py
6. **Verify** — Watcher with LLM works

---

**End of Proposal**
