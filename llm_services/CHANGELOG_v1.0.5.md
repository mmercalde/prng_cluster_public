# Schema v1.0.5 - Team Beta Review Fixes

**Review Date:** December 1, 2025  
**Reviewer:** Team Beta (ChatGPT Technical AI Reviewer)  
**Status:** All 7 recommendations implemented  

---

## Summary of Changes (v1.0.4 → v1.0.5)

| Fix # | Issue | Solution | Files Modified |
|-------|-------|----------|----------------|
| 1 | KV cache bloat risk | Model-specific max_tokens (2048/512) | `llm_router.py`, `llm_server_config.json` |
| 2 | Long-run slowdowns | Context reset at 14K tokens | `llm_router.py` |
| 3 | LLM grounding | Agent identity headers `<agent: NAME>` | `llm_router.py` |
| 4 | Incomplete routing | 16 new PRNG-specific keywords | `llm_server_config.json` |
| 5 | Edge case handling | `[MATH_DELEGATE]` delegation trigger | `llm_router.py` |
| 6 | Debug difficulty | Rotating log files (10MB × 5 backups) | `llm_router.py` |
| 7 | Override mechanism | Human review trigger detection | `llm_router.py`, `llm_server_config.json` |

---

## Detailed Implementation

### Fix 1: Max Token Guardrails

**Before (v1.0.4):**
```python
max_tokens = 2048  # Same for both models
```

**After (v1.0.5):**
```python
self.max_tokens = {
    "orchestrator": 2048,  # Needs longer for code/planning
    "math": 512            # Math answers are concise
}
```

**Config addition:**
```json
"orchestrator": { "max_tokens": 2048 },
"math": { "max_tokens": 512 }
```

---

### Fix 2: Context Reset Trigger

**New method in `llm_router.py`:**
```python
def _check_context_reset(self):
    if self.metrics.context_tokens_estimate > self.context_reset_threshold:
        self.reset_context()

def reset_context(self):
    self.metrics.context_tokens_estimate = 0
```

**Config:**
```json
"routing": { "context_reset_threshold": 14000 }
```

---

### Fix 3: Agent Identity Headers

**New method:**
```python
def set_agent(self, agent_name: str):
    self.current_agent = agent_name

def _add_agent_header(self, prompt: str) -> str:
    if self.current_agent:
        return f"<agent: {self.current_agent}>\n{prompt}"
    return prompt
```

**Usage:**
```python
router.set_agent("scorer_meta_optimizer")
router.route("Analyze survivor distribution...")
# Prompt becomes: "<agent: scorer_meta_optimizer>\nAnalyze survivor distribution..."
```

---

### Fix 4: Expanded Routing Keywords

**16 new keywords added:**
```python
# v1.0.5 additions
"forward sieve", "reverse sieve", "bidirectional",
"skip interval", "gap-aware", "state reconstruction",
"temper", "twist", "pcg output", "xoroshiro", "lcg step",
"survivor scoring", "residue filter", "window size",
"survival rate", "hybrid sieve"
```

---

### Fix 5: Delegation Mechanism

**Detection:**
```python
MATH_DELEGATE_TRIGGER = "[MATH_DELEGATE]"

def _check_delegation(self, response: str) -> bool:
    return self.MATH_DELEGATE_TRIGGER in response
```

**Auto-rerouting:**
```python
if endpoint_name == "orchestrator" and self._check_delegation(content):
    # Re-route to math specialist
    return self.route(prompt, force_endpoint="math", ...)
```

---

### Fix 6: Rotating Disk-Based Logging

**Directory structure:**
```
logs/llm/
├── orchestrator/
│   └── requests.log    # 10MB rotating, 5 backups
└── math/
    └── requests.log    # 10MB rotating, 5 backups
```

**Log format:**
```json
{
    "timestamp": "2025-12-01T15:32:00.123456",
    "agent": "scorer_meta_optimizer",
    "prompt_preview": "Analyze the following...",
    "response_preview": "The statistical analysis shows...",
    "tokens": 342,
    "latency_ms": 1250,
    "prompt_length": 456,
    "response_length": 892
}
```

---

### Fix 7: Human Override Trigger

**Trigger phrases:**
```python
HUMAN_OVERRIDE_TRIGGERS = [
    "HUMAN_REVIEW_REQUIRED",
    "FLAG_FOR_REVIEW",
    "REQUIRES_HUMAN_VERIFICATION",
    "UNCERTAIN_RESULT",
    "LOW_CONFIDENCE_WARNING"
]
```

**Detection and schema update:**
```python
def _check_human_override(self, response: str) -> bool:
    for trigger in self.HUMAN_OVERRIDE_TRIGGERS:
        if trigger in response:
            self.metrics.human_override_requested = True
            self.metrics.override_reason = trigger
            return True
    return False
```

**Check method for agents:**
```python
if router.is_human_override_requested():
    # Halt autonomous execution
    coordinator.enter_safe_hold_mode()
```

---

## Updated Schema Fields

**`agent_metadata.llm_metadata` now includes:**
```json
{
    "orchestrator_model": "Qwen2.5-Coder-14B-Instruct-Q4_K_M",
    "math_model": "Qwen2.5-Math-7B-Instruct-Q5_K_M",
    "orchestrator_calls": 3,
    "math_calls": 7,
    "total_tokens_generated": 2847,
    "llm_reasoning_trace": [...],
    "llm_decision": "math_heavy_analysis",
    "human_override_requested": false,
    "override_reason": null
}
```

---

## Testing Commands

```bash
# Test agent identity
python -m llm_services.llm_router --query "Calculate seed entropy" --agent "window_optimizer"

# Verify logging
tail -f logs/llm/math/requests.log

# Check human override
python -c "
from llm_services import get_router
router = get_router()
router.route('This result has LOW_CONFIDENCE_WARNING')
print(f'Override requested: {router.is_human_override_requested()}')
"
```

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Claude (AI) | 2025-12-01 | ✓ |
| Team Beta Review | ChatGPT | 2025-12-01 | ✓ |
| Technical Review | | | |
| Final Approval | | | |
