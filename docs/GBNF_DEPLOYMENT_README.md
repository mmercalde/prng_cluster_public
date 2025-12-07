# GBNF Grammar Implementation - Deployment Guide

**Version:** 1.0.0  
**Date:** December 6, 2025  
**Status:** Ready for Deployment  
**Effort:** 30 minutes  

---

## What This Is

GBNF (GGML BNF) grammars constrain LLM output to valid JSON structures. This implementation:

- **Forces valid JSON** - No more parsing errors
- **Constrains field names** - Only known parameters allowed
- **Limits enum values** - Actions can only be "proceed", "retry", or "escalate"
- **Works with any llama.cpp model** - Qwen, Phi-4, Mistral, etc.

## What This Is NOT

- ❌ Does not fix semantic reasoning
- ❌ Does not prevent terminology drift in `reasoning` field
- ❌ Does not make Qwen understand "window = ordered list"
- ❌ Is not a substitute for supervisory model

**This is a safety layer, not a solution. Implement this FIRST, then add supervisory model.**

---

## Files Created

```
outputs/
├── grammars/
│   ├── agent_decision.gbnf       # Agent evaluation responses
│   ├── sieve_analysis.gbnf       # Sieve result interpretation  
│   ├── parameter_adjustment.gbnf # Parameter change suggestions
│   └── json_generic.gbnf         # Fallback for any JSON
│
└── llm_services/
    ├── grammar_loader.py         # Grammar loading utility
    └── llm_router_patch.py       # Shows changes for llm_router.py
```

---

## Deployment Steps

### Step 1: Copy Files to Zeus

```bash
# On Zeus
cd ~/distributed_prng_analysis

# Create directories
mkdir -p grammars
mkdir -p llm_services

# Copy grammar files (from wherever you downloaded them)
cp /path/to/outputs/grammars/*.gbnf grammars/
cp /path/to/outputs/llm_services/*.py llm_services/

# Verify
ls -la grammars/
ls -la llm_services/
```

### Step 2: Test Grammar Loading

```bash
cd ~/distributed_prng_analysis
python3 llm_services/grammar_loader.py
```

Expected output:
```
============================================================
GBNF Grammar Loader - Test
============================================================

Grammar directory: /home/michael/distributed_prng_analysis/grammars
Available grammars: ['agent_decision', 'sieve_analysis', 'parameter_adjustment', 'json_generic']
✅ agent_decision: 52 lines loaded
✅ sieve_analysis: 48 lines loaded
✅ parameter_adjustment: 89 lines loaded
✅ json_generic: 32 lines loaded

--- Auto-selection test ---
  'Evaluate the sieve results and decide whether to...' → agent_decision
  'Analyze the bidirectional survivors from the for...' → sieve_analysis
  'Suggest parameter adjustments for window_size...' → parameter_adjustment
  'Generate a summary report...' → json_generic
```

### Step 3: Modify llm_router.py

Open your existing `llm_services/llm_router.py` and apply these changes:

**Add import at top:**
```python
from llm_services.grammar_loader import (
    get_grammar,
    get_grammar_for_prompt,
    GrammarType
)
```

**Modify your completion call to include grammar:**
```python
# Find where you call the LLM and add grammar parameter
payload = {
    "prompt": prompt,
    "n_predict": max_tokens,
    "temperature": temperature,
    "grammar": get_grammar_for_prompt(prompt)  # ADD THIS LINE
}
```

See `llm_router_patch.py` for complete examples.

### Step 4: Test with LLM Server

```bash
# Start your LLM server (if not running)
./scripts/start_llm_servers.sh

# Test grammar constraint
curl -X POST http://localhost:8080/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Evaluate: survivors=5000. Success if >1000. Respond with decision JSON.",
    "n_predict": 500,
    "grammar": "'"$(cat grammars/agent_decision.gbnf)"'"
  }'
```

Output will be forced into this structure:
```json
{
  "success_condition_met": true,
  "confidence": 0.92,
  "reasoning": "5000 survivors exceeds threshold of 1000",
  "recommended_action": "proceed"
}
```

### Step 5: Verify in Agent Code

Test that agents receive properly structured responses:

```python
# test_grammar_integration.py
from llm_services.llm_router import LLMRouter
import json

router = LLMRouter()

prompt = """
Evaluate these window optimization results:
- Forward survivors: 3184
- Reverse survivors: 3295  
- Bidirectional survivors: 5
- Success condition: bidirectional >= 3

Provide your decision.
"""

# This should now return valid JSON every time
response = router.route_with_grammar(prompt, grammar_type="agent_decision")
decision = json.loads(response)

print(f"Success: {decision['success_condition_met']}")
print(f"Action: {decision['recommended_action']}")
print(f"Confidence: {decision['confidence']}")
```

---

## Grammar Reference

### agent_decision.gbnf

Forces this structure:
```json
{
  "success_condition_met": true|false,
  "confidence": 0.00-1.00,
  "reasoning": "string",
  "recommended_action": "proceed"|"retry"|"escalate",
  "suggested_param_adjustments": {...},  // optional
  "warnings": [...]                       // optional
}
```

### sieve_analysis.gbnf

Forces this structure:
```json
{
  "analysis_type": "forward"|"reverse"|"bidirectional"|"hybrid",
  "prng_type": "<any of 46 PRNG types>",
  "survivor_assessment": "high_confidence"|"medium_confidence"|"low_confidence"|"no_signal"|"needs_retry",
  "forward_survivors": integer,
  "reverse_survivors": integer,
  "bidirectional_survivors": integer,
  "match_rate": 0.00-1.00,
  "recommended_window_size": integer,
  "recommended_threshold": 0.00-1.00,
  "interpretation": "string",
  "next_step": "proceed_to_scoring"|"retry_with_adjustments"|"expand_search"|"reduce_threshold"|"escalate"|"complete"
}
```

### parameter_adjustment.gbnf

Forces this structure:
```json
{
  "pipeline_step": 1-6,
  "step_name": "window_optimizer"|"scorer_meta_optimizer"|...,
  "adjustments": {
    "<known_param>": value,
    ...
  },
  "rationale": "string"
}
```

Known parameters include: `window_size`, `offset`, `threshold`, `skip_min`, `skip_max`, `seed_count`, `trials`, `k_folds`, `learning_rate`, etc.

---

## Troubleshooting

### Grammar not loading
```bash
# Check file exists and is readable
cat grammars/agent_decision.gbnf | head -20
```

### LLM ignoring grammar
- Verify grammar is being sent in request
- Check llama.cpp version supports grammar (requires recent build)
- Try simpler grammar (json_generic.gbnf) first

### Invalid JSON still produced
- Grammar may be malformed - validate syntax
- Model may be truncating output - increase max_tokens
- Check for model-specific issues

### Performance impact
- Grammar parsing adds ~10-50ms overhead
- Negligible compared to inference time
- Cache grammars in memory (grammar_loader does this)

---

## Next Steps After Deployment

1. ✅ Deploy grammar files
2. ✅ Test grammar loading
3. ✅ Modify llm_router.py
4. ✅ Verify agent responses are valid JSON
5. ⏳ Proceed to supervisory model implementation (Phase 2)

**Remember: Grammar ensures valid structure, supervisory model ensures valid semantics.**

---

## Files Checklist

```
[ ] grammars/agent_decision.gbnf        → ~/distributed_prng_analysis/grammars/
[ ] grammars/sieve_analysis.gbnf        → ~/distributed_prng_analysis/grammars/
[ ] grammars/parameter_adjustment.gbnf  → ~/distributed_prng_analysis/grammars/
[ ] grammars/json_generic.gbnf          → ~/distributed_prng_analysis/grammars/
[ ] llm_services/grammar_loader.py      → ~/distributed_prng_analysis/llm_services/
[ ] llm_services/llm_router_patch.py    → Reference for modifying llm_router.py
```
