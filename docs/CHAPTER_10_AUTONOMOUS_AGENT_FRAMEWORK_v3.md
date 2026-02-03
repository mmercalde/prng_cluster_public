# Chapter 10: Autonomous Agent Framework

**Version:** 3.1.0 (Full Autonomous Operation)  
**Date:** February 3, 2026  
**Status:** ✅ Full Autonomous Operation — Phase 7 Complete  
**Autonomy:** ~85%  

---

## 1. Executive Summary

The Autonomous Agent Framework is **fully implemented and working**. This chapter documents the existing codebase, validated through live testing on January 8, 2026.

### 1.1 Verified Working Components

| Component | File | Status |
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
| GBNF Grammars | `agent_grammars/*.gbnf` | ✅ v1.1 (4 files, fixed Session 59) |

### 1.2 Live Test Results (January 8, 2026)

```
$ PYTHONPATH=. python3 agents/watcher_agent.py --evaluate optimal_window_config.json

Step: 1 - Window Optimizer
Success: True
Confidence: 0.95
Action: proceed
Reasoning: The evaluation shows bidirectional_count, forward_count, and reverse_count 
           all at 801, meeting the success condition.
Parse method: llm_http_extracted
```

---

## 2. Architecture Overview

### 2.1 System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS AGENT FRAMEWORK v3.1.0                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  Pipeline Step  │───►│  Watcher Agent  │───►│   LLM Router    │        │
│  │   Completes     │    │  (Orchestrator) │    │   (v2.0.0)      │        │
│  └─────────────────┘    └────────┬────────┘    └────────┬────────┘        │
│                                  │                      │                  │
│                                  ▼                      ▼                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐        │
│  │  Step Context   │    │    Doctrine     │    │ DeepSeek-R1-14B │        │
│  │ (Metrics Only)  │    │ (Decision Rules)│    │   (Primary)     │        │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘        │
│                                                         │                  │
│                                  ┌──────────────────────┘                  │
│                                  ▼                                          │
│                         ┌─────────────────┐                                │
│                         │ Claude Opus 4.5 │                                │
│                         │    (Backup)     │                                │
│                         └─────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 File Locations

```
distributed_prng_analysis/
├── agents/
│   ├── watcher_agent.py           # Main orchestrator (v1.1.0)
│   ├── doctrine.py                # Decision framework rules
│   ├── prompt_builder.py          # Prompt assembly
│   ├── agent_decision.py          # Decision models
│   ├── agent_core.py              # Base agent class
│   ├── fingerprint_registry.py    # Dataset+PRNG tracking
│   ├── full_agent_context.py      # Context builder
│   │
│   ├── contexts/                  # Step-specific contexts
│   │   ├── base_agent_context.py
│   │   ├── window_optimizer_context.py      # Step 1
│   │   ├── scorer_meta_context.py           # Step 2.5
│   │   ├── full_scoring_context.py          # Step 3
│   │   ├── ml_meta_context.py               # Step 4
│   │   ├── anti_overfit_context.py          # Step 5
│   │   └── prediction_context.py            # Step 6
│   │
│   ├── manifest/                  # Manifest loading
│   ├── parameters/                # Parameter context
│   ├── history/                   # Analysis history
│   ├── runtime/                   # Runtime context
│   ├── safety/                    # Kill switch
│   └── step_runner/               # Step execution
│
├── llm_services/
│   ├── llm_router.py              # Primary + Backup routing (v2.0.0)
│   ├── grammar_loader.py          # GBNF grammar management
│   ├── llm_server_config.json     # Server configuration
│   └── start_llm_servers.sh       # Server startup script
│
└── agent_manifests/               # Step configurations
    ├── window_optimizer.json
    ├── scorer_meta.json
    ├── full_scoring.json
    ├── ml_meta.json
    ├── reinforcement.json
    └── prediction.json
```

---

## 3. LLM Architecture

### 3.1 Primary + Backup Design

| Role | Model | Speed | Access |
|------|-------|-------|--------|
| **Primary** | DeepSeek-R1-14B | 51 tok/s | llama.cpp port 8080 |
| **Backup** | Claude Opus 4.5 | 38 tok/s | Claude Code CLI |

### 3.2 Server Configuration

**File:** `llm_services/llm_server_config.json`

```json
{
    "schema_version": "2.0.0",
    "architecture": "primary_plus_backup",
    
    "primary": {
        "name": "DeepSeek-R1-14B",
        "model_file": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        "port": 8080,
        "context_length": 8192,
        "n_gpu_layers": 99
    },
    
    "backup": {
        "name": "Claude Opus 4.5",
        "type": "external_api",
        "access_method": "claude_code_cli"
    },
    
    "routing": {
        "escalation_triggers": [
            "UNCERTAIN", "LOW_CONFIDENCE", 
            "ESCALATE_TO_BACKUP", "REQUIRES_DEEP_ANALYSIS"
        ]
    }
}
```

### 3.3 Server Startup

**File:** `llm_services/start_llm_servers.sh`

```bash
# Start server
cd ~/distributed_prng_analysis
./llm_services/start_llm_servers.sh

# Output:
# ✅ Primary model found: DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf
# ✅ Primary (DeepSeek-R1-14B) (port 8080): HEALTHY
# ✅ Claude Code CLI: AVAILABLE

# Stop server
pkill -f 'llama-server.*port 8080'
```

### 3.4 LLM Router

**File:** `llm_services/llm_router.py` (v2.0.0)

```python
class LLMRouter:
    """Routes requests to primary LLM with backup escalation."""
    
    def route(self, prompt: str, force_backup: bool = False) -> str:
        """Route request to primary, escalate to backup if needed."""
        
        if force_backup:
            return self._call_backup(prompt)
        
        response = self._call_primary(prompt)
        
        if self._should_escalate(response):
            return self._call_backup(prompt)
        
        return response
```

---

## 4. Watcher Agent

### 4.1 CLI Reference

```bash
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/watcher_agent.py --help

Options:
  --daemon              Run as daemon, watching for results
  --run-pipeline        Run the full pipeline
  --start-step N        Starting step for pipeline (1-6)
  --end-step N          Ending step for pipeline (1-6)
  --evaluate FILE       Evaluate a single result file
  --status              Show watcher status
  --clear-halt          Clear the halt file to resume
  --halt                Create halt file to stop watcher
  --no-llm              Disable LLM evaluation, use heuristics only
  --no-grammar          Disable grammar-constrained decoding
  --threshold N         Auto-proceed confidence threshold (default: 0.70)
```

### 4.2 Evaluation Modes

| Mode | Command | Description |
|------|---------|-------------|
| Single file | `--evaluate file.json` | Evaluate one output |
| Pipeline | `--run-pipeline` | Run full pipeline |
| Daemon | `--daemon` | Watch for new outputs |
| Heuristic only | `--no-llm` | Skip LLM, use rules |

### 4.3 Decision Flow

```python
# In watcher_agent.py
def _evaluate_with_llm(self, context):
    """Evaluation hierarchy:
    1. Try LLMRouter with grammar (if available)
    2. Try HTTP direct call
    3. Fall back to heuristic
    """
    
    # Try grammar-constrained evaluation
    if self.llm_router and self.config.use_grammar:
        return self._evaluate_with_router(context)
    
    # Try HTTP fallback
    return self._evaluate_with_http(context)
```

### 4.4 Decision Actions

| Action | Confidence | Behavior |
|--------|------------|----------|
| PROCEED | ≥ 0.70 | Trigger next step automatically |
| RETRY | 0.50 - 0.70 | Re-run current step with adjustments |
| ESCALATE | < 0.50 | Alert human for review |

---

## 5. Step Contexts

### 5.1 Design Philosophy

> **"LLM does the reasoning, not this file."**
> — window_optimizer_context.py header

Step contexts are responsible for:
- ✅ Extract raw metrics from results
- ✅ Compute derived metrics (rates, ratios)
- ✅ Load threshold priors from config
- ✅ Package data for LLM evaluation

NOT responsible for:
- ❌ Semantic interpretation ("good", "bad")
- ❌ Decision making (proceed/retry/escalate)

### 5.2 Available Contexts

| Context | Step | Key Metrics |
|---------|------|-------------|
| `window_optimizer_context.py` | 1 | bidirectional_count, forward_count, rates |
| `scorer_meta_context.py` | 2.5 | best_accuracy, completed_trials |
| `full_scoring_context.py` | 3 | scored_count, coverage_pct |
| `ml_meta_context.py` | 4 | best_r2, validation_r2, model comparisons |
| `anti_overfit_context.py` | 5 | test_r2, overfit_ratio, signal_quality |
| `prediction_context.py` | 6 | prediction_confidence, pool_coverage |

### 5.3 Context Structure

**File:** `agents/contexts/window_optimizer_context.py` (v4.0.0)

```python
class WindowOptimizerContext(BaseAgentContext):
    """
    Specialized context for Step 1: Window Optimizer.
    
    REFACTORED RESPONSIBILITIES:
    - Extract raw metrics from results
    - Compute derived metrics (rates, ratios)
    - Load threshold priors from config
    - Package for LLM evaluation
    """
    
    agent_name: str = "window_optimizer_agent"
    pipeline_step: int = 1
    step_id: str = "step_1_window_optimizer"
```

---

## 6. Doctrine (Decision Framework)

### 6.1 Purpose

The doctrine defines consistent decision-making rules across all steps.

**File:** `agents/doctrine.py` (v3.2.0)

```python
def get_doctrine() -> Dict[str, Any]:
    return {
        "version": "1.0.0",
        "purpose": "PRNG pattern analysis through systematic pipeline execution",
        
        "decision_framework": {
            "proceed": {
                "conditions": [
                    "success_condition_met = true",
                    "confidence >= 0.70",
                    "no critical anomalies detected"
                ]
            },
            "retry": {
                "conditions": [
                    "success_condition_met = false",
                    "retries_remaining > 0",
                    "adjustable parameters available"
                ]
            },
            "escalate": {
                "conditions": [
                    "confidence < 0.50",
                    "retries_exhausted",
                    "critical anomaly detected"
                ]
            }
        }
    }
```

---

## 7. Grammar System

### 7.1 Grammar Loader

**File:** `llm_services/grammar_loader.py` (v1.0.0)

```python
class GrammarType(str, Enum):
    """Available grammar types for constrained decoding."""
    AGENT_DECISION = "agent_decision"
    SIEVE_ANALYSIS = "sieve_analysis"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    JSON_GENERIC = "json_generic"
```

### 7.2 Grammar File Location

Grammar files should be in: `agent_grammars/` (referenced but needs creation)

```
agent_grammars/
├── agent_decision.gbnf       # Proceed/retry/escalate decisions
├── sieve_analysis.gbnf       # Sieve result interpretation
├── parameter_adjustment.gbnf # Parameter change suggestions
└── json_generic.gbnf         # Fallback for any valid JSON
```

### 7.3 Decision Grammar

```gbnf
root ::= "{" ws "\"decision\"" ws ":" ws decision-value ws "," ws 
         "\"confidence\"" ws ":" ws number ws "," ws 
         "\"reasoning\"" ws ":" ws string ws "}"

decision-value ::= "\"proceed\"" | "\"retry\"" | "\"escalate\""
string ::= "\"" ([^"\\] | "\\" .)* "\""
number ::= "0." [0-9]+
ws ::= [ \t\n]*
```

---

## 8. Usage Examples

### 8.1 Evaluate Single Step Output

```bash
cd ~/distributed_prng_analysis

# Start LLM server
./llm_services/start_llm_servers.sh

# Evaluate Step 1 output
PYTHONPATH=. python3 agents/watcher_agent.py --evaluate optimal_window_config.json

# Output:
# Step: 1 - Window Optimizer
# Success: True
# Confidence: 0.95
# Action: proceed
```

### 8.2 Run Without LLM (Heuristic Only)

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --evaluate optimal_window_config.json --no-llm

# Output:
# Using heuristic evaluation (LLM unavailable)
# Action: proceed
# Confidence: 0.95
```

### 8.3 Run Full Pipeline

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 6
```

### 8.4 Run as Daemon

```bash
PYTHONPATH=. python3 agents/watcher_agent.py --daemon
```

---

## 9. Troubleshooting

### 9.1 "No module named 'agents'"

```bash
# Always run from project root with PYTHONPATH
cd ~/distributed_prng_analysis
PYTHONPATH=. python3 agents/watcher_agent.py --help
```

### 9.2 "Connection refused" on port 8080

```bash
# Start the LLM server first
./llm_services/start_llm_servers.sh

# Verify health
curl http://localhost:8080/health
```

### 9.3 LLM Returns Invalid JSON

The system has multiple fallbacks:
1. Grammar-constrained evaluation (if LLMRouter available)
2. HTTP direct call with JSON extraction
3. Heuristic fallback (always works)

---

## 10. Known Gaps

| Gap | Impact | Priority |
|-----|--------|----------|
| `agent_grammars/` directory missing | Grammar files referenced but not created | Medium |
| On-demand LLM lifecycle | Server stays running vs start/stop per query | Low |
| MAIN_MISSION in doctrine.py | No global mission context | Low |

---

## 11. Version History

```
Version 3.0.0 - January 8, 2026
- VALIDATED: Live testing confirmed all components working
- DOCUMENTED: Existing codebase structure
- REMOVED: Duplicate code proposals (test_watcher_clean.py)

Version 2.0.0 - January 7, 2026
- Updated to DeepSeek-R1-14B primary + Claude backup
- A/B testing results documented

Version 1.0.0 - December 2025
- Initial Dual-LLM architecture (Qwen models)
```

---

## 12. References

| File | Purpose |
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
| `agent_grammars/*.gbnf` | Grammar constraint files (v1.1) |

---


### Parameter Bounds Authority

**Critical:** Agent manifests (`agent_manifests/*.json`) define parameter metadata and documentation only.
Runtime bounds for Optuna search are defined exclusively in `distributed_config.json`.

| Source | Role |
|--------|------|
| `distributed_config.json` | **Authoritative** - Runtime search bounds |
| `agent_manifests/*.json` | Documentation only - NOT runtime bounds |
| `baselines/baseline_window_thresholds.json` | Recovery baseline |

See `docs/THRESHOLD_GOVERNANCE.md` for full governance model.

---

**END OF CHAPTER 10**
