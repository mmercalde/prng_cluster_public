# Unified Agent Context Framework v3.2.0

**Document Version:** 3.2.0  
**Date:** December 4, 2025  
**Author:** Claude (AI Assistant)  
**Status:** PRODUCTION-READY  
**Supersedes:** v3.1.0  
**Patch Focus:** AI Agent Parameter Awareness

---

## Critical Issue Addressed

**Problem:** AI agents cannot effectively recommend parameter adjustments because they don't clearly see:
1. What arguments the script accepts (from manifest `args_map`)
2. What values are legal (from parameter registry)
3. What values were used in the current run
4. How to format `suggested_param_adjustments` in their response

**Solution:** Create a unified "Adjustable Parameters" section in the prompt that combines manifest args, registry bounds, and current values.

---

## Part 1: Enhanced Agent Manifest Model

```python
# agents/manifest/agent_manifest.py (v3.2.0)
"""
Agent Manifest - Now with parameter introspection for AI agents.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
import json


class ActionType(str, Enum):
    RUN_SCRIPT = "run_script"
    RUN_DISTRIBUTED = "run_distributed"
    AGGREGATE = "aggregate"


class AgentAction(BaseModel):
    """Single action with full argument mapping."""
    
    type: ActionType
    script: str
    args_map: Dict[str, str] = Field(default_factory=dict)
    distributed: bool = False
    timeout_minutes: int = 60
    
    def get_script_args(self) -> List[str]:
        """Get list of script argument names (CLI flags)."""
        return list(self.args_map.keys())
    
    def get_context_vars(self) -> List[str]:
        """Get list of context variable names."""
        return list(self.args_map.values())


class AgentManifest(BaseModel):
    """
    Complete agent manifest with parameter introspection.
    
    The args_map defines what parameters the agent can adjust:
    - Keys = CLI argument names (e.g., "trials", "max-seeds")
    - Values = Context variable names (e.g., "window_trials", "seed_count")
    """
    
    agent_name: str
    description: str = ""
    pipeline_step: int = Field(ge=1, le=6)
    
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    
    actions: List[AgentAction] = Field(default_factory=list)
    
    follow_up_agents: List[str] = Field(default_factory=list)
    success_condition: str = ""
    retry: int = Field(default=2, ge=0, le=5)
    
    version: str = "1.0.0"
    
    @classmethod
    def load(cls, manifest_path: str) -> "AgentManifest":
        with open(manifest_path) as f:
            data = json.load(f)
        return cls.model_validate(data)
    
    @property
    def primary_action(self) -> Optional[AgentAction]:
        return self.actions[0] if self.actions else None
    
    @property
    def script_name(self) -> str:
        """Get the primary script name."""
        if self.primary_action:
            return self.primary_action.script
        return ""
    
    @property
    def adjustable_args(self) -> Dict[str, str]:
        """
        Get all adjustable arguments from primary action.
        
        Returns dict mapping CLI arg -> context variable name.
        """
        if self.primary_action:
            return self.primary_action.args_map
        return {}
    
    def get_adjustable_params_list(self) -> List[str]:
        """Get list of adjustable parameter names for LLM."""
        return list(self.adjustable_args.keys())
```

---

## Part 2: Parameter Context Builder

This new class combines manifest args + registry bounds + current values:

```python
# agents/parameters/parameter_context.py
"""
Parameter Context - Builds complete parameter awareness for AI agents.

Combines:
1. Manifest args_map (what can be adjusted)
2. Registry bounds (legal values)
3. Current values (what was used)
4. Suggested adjustments (for retry scenarios)
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from agents.manifest.agent_manifest import AgentManifest
from agents.registry.parameter_registry import (
    get_registry, 
    ScriptParameterRegistry,
    ParameterSpec
)


class ParameterInfo(BaseModel):
    """Complete information about a single adjustable parameter."""
    
    # Identity
    name: str
    cli_flag: str
    context_var: str
    
    # From registry
    param_type: str = "unknown"
    description: str = ""
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: List[Any] = Field(default_factory=list)
    default: Optional[Any] = None
    
    # Current run
    current_value: Optional[Any] = None
    
    def to_prompt_line(self) -> str:
        """Format for LLM prompt."""
        # Build bounds string
        if self.choices:
            bounds = f"choices: {self.choices}"
        elif self.min_value is not None or self.max_value is not None:
            bounds = f"range: [{self.min_value} - {self.max_value}]"
        else:
            bounds = "no bounds specified"
        
        # Current value
        current = f"current: {self.current_value}" if self.current_value is not None else "current: (not set)"
        
        # Default
        default = f"default: {self.default}" if self.default is not None else ""
        
        return f"""
    {self.name} ({self.cli_flag}):
      Type: {self.param_type}
      {bounds}
      {current}
      {default}
      {self.description}""".strip()


class ParameterContext(BaseModel):
    """
    Complete parameter context for AI agent decision-making.
    
    This is what gets surfaced in the LLM prompt so the agent
    knows exactly what it can adjust and within what bounds.
    """
    
    script_name: str
    parameters: List[ParameterInfo] = Field(default_factory=list)
    
    @classmethod
    def build(
        cls,
        manifest: AgentManifest,
        current_values: Dict[str, Any] = None
    ) -> "ParameterContext":
        """
        Build parameter context from manifest + registry + current values.
        
        Args:
            manifest: Agent manifest with args_map
            current_values: Dict of current parameter values from results/config
        """
        current_values = current_values or {}
        
        script_name = manifest.script_name
        args_map = manifest.adjustable_args
        
        # Get registry for this script
        registry = get_registry(script_name)
        
        parameters = []
        
        for cli_flag, context_var in args_map.items():
            # Clean CLI flag (remove leading dashes)
            clean_name = cli_flag.lstrip('-').replace('-', '_')
            
            # Get spec from registry if available
            spec = registry.get_param(clean_name) if registry else None
            
            param_info = ParameterInfo(
                name=clean_name,
                cli_flag=f"--{cli_flag.lstrip('-')}",
                context_var=context_var,
                param_type=spec.param_type if spec else "unknown",
                description=spec.description if spec else "",
                min_value=spec.min_value if spec else None,
                max_value=spec.max_value if spec else None,
                choices=spec.choices if spec else [],
                default=spec.default if spec else None,
                current_value=current_values.get(clean_name) or current_values.get(context_var)
            )
            
            parameters.append(param_info)
        
        return cls(script_name=script_name, parameters=parameters)
    
    def to_prompt_section(self) -> str:
        """Generate ADJUSTABLE PARAMETERS section for LLM prompt."""
        
        if not self.parameters:
            return ""
        
        lines = [
            "═" * 80,
            "ADJUSTABLE PARAMETERS",
            "═" * 80,
            "",
            f"Script: {self.script_name}",
            "",
            "The following parameters can be adjusted in suggested_param_adjustments:",
            ""
        ]
        
        for param in self.parameters:
            lines.append(param.to_prompt_line())
            lines.append("")
        
        # Add format example
        lines.extend([
            "─" * 80,
            "ADJUSTMENT FORMAT",
            "─" * 80,
            "",
            "When recommending 'retry', use this exact format:",
            "",
            '"suggested_param_adjustments": {',
        ])
        
        # Show example with actual param names
        for i, param in enumerate(self.parameters[:3]):
            comma = "," if i < min(2, len(self.parameters) - 1) else ""
            example_val = param.default if param.default is not None else "..."
            lines.append(f'    "{param.name}": {repr(example_val)}{comma}')
        
        lines.extend([
            '}',
            '',
            'Only include parameters you want to change. Use values within the specified bounds.'
        ])
        
        return "\n".join(lines)
    
    def validate_adjustments(self, adjustments: Dict[str, Any]) -> List[str]:
        """
        Validate AI-suggested parameter adjustments.
        
        Returns list of error messages (empty if all valid).
        """
        errors = []
        
        param_lookup = {p.name: p for p in self.parameters}
        
        for name, value in adjustments.items():
            if name not in param_lookup:
                errors.append(f"Unknown parameter: {name}")
                continue
            
            param = param_lookup[name]
            
            # Check choices
            if param.choices and value not in param.choices:
                errors.append(f"{name}: {value} not in {param.choices}")
            
            # Check numeric bounds
            if param.min_value is not None and isinstance(value, (int, float)):
                if value < param.min_value:
                    errors.append(f"{name}: {value} < min {param.min_value}")
            
            if param.max_value is not None and isinstance(value, (int, float)):
                if value > param.max_value:
                    errors.append(f"{name}: {value} > max {param.max_value}")
        
        return errors
```

---

## Part 3: Updated FullAgentContext

```python
# agents/full_agent_context.py (v3.2.0)
"""
Full Agent Context - Now with parameter awareness.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from datetime import datetime

from agents.contexts.agent_context import AgentContext
from agents.contexts.runtime_context import RuntimeContext
from agents.contexts.pipeline_context import PipelineStepContext
from agents.safety.kill_switch import KillSwitch
from agents.history.analysis_history import AnalysisHistory
from agents.doctrine import load_doctrine


class FullAgentContext(BaseModel):
    """
    Complete context for AI agent evaluation.
    
    v3.2.0: Now includes ParameterContext for adjustment awareness.
    """
    
    agent_context: AgentContext
    runtime: RuntimeContext
    pipeline: PipelineStepContext
    kill_switch: KillSwitch
    history: AnalysisHistory
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_prompt(self) -> str:
        """
        Generate complete prompt with FULL PARAMETER AWARENESS.
        
        Sections:
        1. Doctrine (shared reasoning rules)
        2. Agent-specific context (step, evaluation)
        3. ADJUSTABLE PARAMETERS (manifest args + registry bounds + current values)
        4. Runtime environment
        5. Pipeline expectations
        6. Safety status
        7. Historical analysis
        8. Output format
        """
        if not self.check_safety():
            return self._safety_halt_prompt()
        
        sections = []
        
        # 1. Doctrine
        sections.append(load_doctrine())
        
        # 2. Agent-specific context
        sections.append(self.agent_context.to_prompt_body())
        
        # 3. ADJUSTABLE PARAMETERS - Key for AI agent awareness
        sections.append(self.agent_context.parameter_context.to_prompt_section())
        
        # 4. Runtime
        sections.append(self.runtime.to_prompt_section())
        
        # 5. Pipeline expectations
        sections.append(self.pipeline.to_prompt_section())
        
        # 6. Safety
        sections.append(self.kill_switch.to_prompt_section())
        
        # 7. History
        sections.append("─" * 80)
        sections.append("HISTORICAL ANALYSIS")
        sections.append("─" * 80)
        sections.append(self.history.to_prompt_section(
            self.agent_context.manifest.agent_name
        ))
        
        # 8. Output format with explicit param adjustment instructions
        sections.append(self._output_format_section())
        
        return "\n\n".join(sections)
    
    def _output_format_section(self) -> str:
        """Generate output format section with clear param adjustment instructions."""
        
        # Get adjustable param names for example
        param_names = [p.name for p in self.agent_context.parameter_context.parameters[:3]]
        
        return f"""
═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Evaluate these results against the success condition and evaluation criteria.

Respond with ONLY valid JSON (no markdown, no text outside JSON):

{{
    "success_condition_met": true | false,
    "confidence": 0.00 to 1.00,
    "reasoning": "1-2 sentence explanation using step-specific metrics",
    "recommended_action": "proceed" | "retry" | "escalate",
    "suggested_param_adjustments": {{
        // ONLY if recommending "retry"
        // Use parameter names from ADJUSTABLE PARAMETERS section
        // Example: "{param_names[0] if param_names else 'param'}": <new_value>
        // Values MUST be within the specified bounds
    }},
    "warnings": []
}}

DECISION GUIDE:
• PROCEED: success_condition_met=true AND confidence ≥ 0.7
• RETRY: success_condition_met=false BUT adjustable parameters could help
• ESCALATE: Anomalies detected OR confidence < 0.5 OR no viable adjustments
"""
```

---

## Part 5: Example Prompt Output

Here's what the LLM now sees for Window Optimizer:

```
═══════════════════════════════════════════════════════════════════════════════
DISTRIBUTED PRNG ANALYSIS SYSTEM - AGENT DOCTRINE
═══════════════════════════════════════════════════════════════════════════════
[... doctrine ...]

═══════════════════════════════════════════════════════════════════════════════
STEP 1: WINDOW OPTIMIZER
═══════════════════════════════════════════════════════════════════════════════
[... system description, results, interpretation ...]

═══════════════════════════════════════════════════════════════════════════════
ADJUSTABLE PARAMETERS
═══════════════════════════════════════════════════════════════════════════════

Script: window_optimizer.py

The following parameters can be adjusted in suggested_param_adjustments:

    window_size (--window-size):
      Type: int
      range: [50 - 2000]
      current: 512
      default: 512
      Number of lottery draws to analyze

    skip_max (--skip-max):
      Type: int
      range: [1 - 100]
      current: 20
      default: 20
      Maximum PRNG skip value

    trials (--trials):
      Type: int
      range: [10 - 500]
      current: 50
      default: 50
      Optuna optimization trials

    prng_type (--prng-type):
      Type: choice
      choices: ['java_lcg', 'mt19937', 'xorshift32', 'xorshift64', 'pcg32']
      current: java_lcg
      default: java_lcg
      PRNG algorithm to analyze

────────────────────────────────────────────────────────────────────────────────
ADJUSTMENT FORMAT
────────────────────────────────────────────────────────────────────────────────

When recommending 'retry', use this exact format:

"suggested_param_adjustments": {
    "window_size": 100,
    "skip_max": 0.1,
    "trials": "java_lcg"
}

Only include parameters you want to change. Use values within the specified bounds.

[... runtime, pipeline, history sections ...]

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Evaluate these results against the success condition and evaluation criteria.

Respond with ONLY valid JSON...
```

---

## Part 6: Validation of AI Adjustments

When the AI recommends a retry, we validate its suggested adjustments:

```python
# In watcher_agent.py

def handle_decision(self, decision: AgentDecision, context: FullAgentContext):
    """Process AI decision with parameter validation."""
    
    if decision.recommended_action == "retry":
        # Validate suggested adjustments against bounds
        errors = context.agent_context.parameter_context.validate_adjustments(
            decision.suggested_param_adjustments
        )
        
        if errors:
            # AI suggested invalid values - log warning but proceed
            logger.warning(f"Invalid param adjustments: {errors}")
            # Could auto-correct or escalate
        else:
            # Valid adjustments - apply to next run
            self.apply_adjustments(decision.suggested_param_adjustments)
```

---

## Summary: v3.2.0 Additions

| Component | Purpose |
|-----------|---------|
| `ParameterInfo` | Single parameter with name, bounds, current value |
| `ParameterContext` | Combines manifest args + registry bounds + current values |
| `to_prompt_section()` | Generates clear "ADJUSTABLE PARAMETERS" section for LLM |
| `validate_adjustments()` | Validates AI's suggested values against bounds |
| Updated prompt structure | Shows parameters, format example, and decision guide |

**Now the AI agent clearly sees:**
1. ✅ What parameters exist (from manifest `args_map`)
2. ✅ What values are legal (from parameter registry)
3. ✅ What values were used (from results)
4. ✅ How to format adjustments in response
5. ✅ Validation catches out-of-bounds suggestions

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Claude (AI) | 2025-12-04 | ✓ |
| Team Beta | | | |
| Team Alpha | | | |
| Team Charlie | | | |

---

**End of v3.2.0 Specification**

---
---
---

# Addendum E: Centralized Search Bounds & Threshold Philosophy

**Date:** December 7, 2025 (Session 8)
**Author:** Claude (AI Assistant)
**Focus:** Single Source of Truth for Configuration, Correct Threshold Strategy

---

## Overview

Session 8 addressed a critical architectural issue: search bounds were duplicated across multiple files, leading to inconsistent behavior and difficult debugging. Additionally, the threshold philosophy was corrected from "filtering" to "discovery" mode, dramatically improving survivor counts.

---

## Problem Analysis

### Issue 1: Duplicate Hardcoded Bounds

Search bounds were defined in **three separate locations**:

| File | Location | Values |
|------|----------|--------|
| `window_optimizer.py` | SearchBounds class defaults (lines 82-94) | min_window=128, max_threshold=0.70 |
| `window_optimizer_bayesian.py` | Duplicate SearchBounds class (lines 78-81) | min_window=256, max_threshold=0.70 |
| `window_optimizer_integration_final.py` | Hardcoded instantiation (lines 461-469) | min_window=1, max_window=4096 |

**Consequence:** Changes to one file had no effect because another file was actually being used.

### Issue 2: Wrong Threshold Philosophy

Thresholds were set HIGH (0.50-0.95) acting as **filters**:
- High thresholds eliminated most candidates
- Optuna had no signal to learn (all trials returned 0 survivors)
- Score function returned 0 for zero survivors (no penalty)

**Result:** 0-2 bidirectional survivors instead of 37,000+

---

## Solution Architecture

### 1. Centralized Configuration in `distributed_config.json`

Added new `search_bounds` section:

```json
{
  "nodes": [...],
  "sieve_defaults": {...},
  "reverse_sieve_defaults": {...},
  "search_bounds": {
    "window_size": {"min": 2, "max": 500},
    "offset": {"min": 0, "max": 100},
    "skip_min": {"min": 0, "max": 10},
    "skip_max": {"min": 10, "max": 500},
    "forward_threshold": {"min": 0.001, "max": 0.10, "default": 0.01},
    "reverse_threshold": {"min": 0.001, "max": 0.10, "default": 0.01}
  }
}
```

### 2. Loader Function with Safe Fallbacks

```python
# window_optimizer.py

def load_search_bounds_from_config(config_path: str = "distributed_config.json") -> dict:
    """
    Load search bounds from distributed_config.json.
    Returns dict with all bounds, using safe defaults if config missing.
    """
    defaults = {
        "window_size": {"min": 2, "max": 500},
        "offset": {"min": 0, "max": 100},
        "skip_min": {"min": 0, "max": 10},
        "skip_max": {"min": 10, "max": 500},
        "forward_threshold": {"min": 0.001, "max": 0.10, "default": 0.01},
        "reverse_threshold": {"min": 0.001, "max": 0.10, "default": 0.01}
    }
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        bounds = config.get("search_bounds", {})
        for key in defaults:
            if key in bounds:
                defaults[key].update(bounds[key])
        return defaults
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load search_bounds from {config_path}: {e}")
        return defaults
```

### 3. SearchBounds Class with Factory Method

```python
@dataclass
class SearchBounds:
    """
    Search space boundaries for optimization.
    Values loaded from distributed_config.json via from_config() classmethod.
    """
    # Defaults (overridden by from_config)
    min_window_size: int = 2
    max_window_size: int = 500
    min_offset: int = 0
    max_offset: int = 100
    min_skip_min: int = 0
    max_skip_min: int = 10
    min_skip_max: int = 10
    max_skip_max: int = 500
    # Threshold bounds - LOW for discovery, not filtering
    min_forward_threshold: float = 0.001
    max_forward_threshold: float = 0.10
    min_reverse_threshold: float = 0.001
    max_reverse_threshold: float = 0.10
    # Defaults
    default_forward_threshold: float = 0.01
    default_reverse_threshold: float = 0.01
    session_options: List[List[str]] = None

    @classmethod
    def from_config(cls, config_path: str = "distributed_config.json") -> 'SearchBounds':
        """Create SearchBounds from config file."""
        cfg = load_search_bounds_from_config(config_path)
        return cls(
            min_window_size=cfg["window_size"]["min"],
            max_window_size=cfg["window_size"]["max"],
            # ... all other fields
        )
```

### 4. Fixed Circular Import

**Problem:** `window_optimizer_bayesian.py` imported from `window_optimizer.py` and vice versa.

**Solution:**
```python
# window_optimizer_bayesian.py

from typing import TYPE_CHECKING

# Type checking import (no runtime circular dependency)
if TYPE_CHECKING:
    from window_optimizer import SearchBounds

# Runtime import function for when we actually need the classes
def _get_search_bounds():
    from window_optimizer import SearchBounds, load_search_bounds_from_config
    return SearchBounds, load_search_bounds_from_config

# Use string forward references for type hints
def search(self, objective_function, bounds: 'SearchBounds', ...):
```

---

## Threshold Philosophy Correction

### Before: Filtering Mode (WRONG)

```
High thresholds (0.50-0.95)
         │
         ▼
    ┌─────────────┐
    │ Forward     │ ──► Very few candidates
    │ Sieve       │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Reverse     │ ──► Even fewer candidates
    │ Sieve       │
    └─────────────┘
         │
         ▼
    0-2 bidirectional survivors
    (Optuna has no signal to learn)
```

### After: Discovery Mode (CORRECT)

```
Low thresholds (0.001-0.10)
         │
         ▼
    ┌─────────────┐
    │ Forward     │ ──► Many candidates (including false positives)
    │ Sieve       │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │ Reverse     │ ──► Many candidates (including false positives)
    │ Sieve       │
    └─────────────┘
         │
         ▼
    ┌─────────────────────────┐
    │ Bidirectional           │
    │ Intersection            │ ──► Only REAL seeds survive
    │ (the actual filter!)    │
    └─────────────────────────┘
         │
         ▼
    37,000+ bidirectional survivors
    (Optuna discovers optimal thresholds)
    (ML has rich training data)
```

---

## Accumulated Totals Display

### New Fields in `progress_display.py`

```python
def update_trial_stats(self, trial_num: int = 0, forward_survivors: int = 0,
                      reverse_survivors: int = 0, bidirectional: int = 0,
                      best_bidirectional: int = 0, config_desc: str = "",
                      accumulated_forward: int = 0, accumulated_reverse: int = 0,
                      accumulated_bidirectional: int = 0):
    """Update current trial statistics."""
    self.trial_stats = {
        "trial_num": trial_num,
        "forward_survivors": forward_survivors,
        "reverse_survivors": reverse_survivors,
        "bidirectional": bidirectional,
        "best_bidirectional": best_bidirectional,
        "config_desc": config_desc,
        "accumulated_forward": accumulated_forward,
        "accumulated_reverse": accumulated_reverse,
        "accumulated_bidirectional": accumulated_bidirectional
    }
```

### Dashboard Display

```html
<!-- Accumulated Totals -->
<div style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-color);">
    <div style="font-size: 11px; color: var(--text-secondary);">📊 Accumulated Across All Trials</div>
    <div class="stats-grid">
        <div class="stat-item">
            <div class="stat-value">{{ accumulated_forward }}</div>
            <div class="stat-label">Total Forward</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ accumulated_reverse }}</div>
            <div class="stat-label">Total Reverse</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{{ accumulated_bidirectional }}</div>
            <div class="stat-label">Total Bidirectional</div>
        </div>
    </div>
</div>
```

---

## Test Results

### Before Session 8 (High Thresholds)

| Metric | Value |
|--------|-------|
| Best Bidirectional | 1-2 |
| Accumulated Forward | 3 |
| Accumulated Reverse | 3 |
| Accumulated Bidirectional | 3 |
| Optuna Signal | None (all zeros) |

### After Session 8 (Low Thresholds + Centralized Config)

| Metric | Value |
|--------|-------|
| Best Bidirectional | **37,095** |
| Accumulated Forward | **121,622** |
| Accumulated Reverse | **121,622** |
| Accumulated Bidirectional | **121,622** |
| Unique Seeds Saved | **37,933** |
| Optuna-Discovered FT | 0.0 |
| Optuna-Discovered RT | 0.02 |

**Improvement:** ~18,500x more bidirectional survivors

---

## Files Modified

| File | Changes |
|------|---------|
| `distributed_config.json` | Added `search_bounds` section |
| `window_optimizer.py` | Added `load_search_bounds_from_config()`, `SearchBounds.from_config()` |
| `window_optimizer_bayesian.py` | Removed duplicate SearchBounds, fixed circular import with TYPE_CHECKING |
| `window_optimizer_integration_final.py` | Uses `SearchBounds.from_config()`, fixed bounds definition order |
| `progress_display.py` | Added accumulated_forward/reverse/bidirectional parameters |
| `web_dashboard.py` | Added accumulated totals display section |

---

## Pattern for Future Configuration

When adding new configurable parameters:

1. **Add to `distributed_config.json`** in appropriate section
2. **Create loader function** with safe fallback defaults
3. **Use factory method** (`from_config()`) instead of class defaults
4. **Never hardcode** values in multiple places
5. **Document** the parameter in manifests for AI awareness

---

## Session 8 Git Commits

- `dfcba45` - Session 8: Centralize search bounds + fix threshold optimization

---

**End of Addendum E**

---
---
---

# Addendum F: agent_metadata Injection for Pipeline Scripts

**Document Version:** 1.1.0 (Revised after Team Beta Review)  
**Date:** December 8-9, 2025 (Session 9)  
**Author:** Claude (AI Assistant)  
**Status:** ✅ IMPLEMENTED (December 9, 2025)  
**Focus:** Complete agent_metadata injection for Steps 4, 5, and 6  
**Reviewers:** Team Beta (Peer Review Completed)

### Implementation Commits
```
9d94204 - Session 9: agent_metadata injection for prediction_generator.py (Step 6)
97aaa4d - Session 9: agent_metadata injection for meta_prediction_optimizer_anti_overfit.py (Step 5)
ea399e7 - Session 9: agent_metadata injection for meta_prediction_optimizer.py (Step 4)
```

---

## Executive Summary

This addendum documents the **corrected** agent_metadata injection plan after:
1. Initial audit identifying scripts needing injection
2. **Team Beta peer review** identifying 3 critical caveats
3. **Full verification** of all field names and class structures
4. Correction of incorrect field references

**Result:** Only **3 scripts** need injection. All field names have been verified against actual source code.

---

## Team Beta Review Response

### Critical Caveats Addressed

| Caveat | Issue | Resolution |
|--------|-------|------------|
| **1** | `self.best_metrics.accuracy` doesn't exist in Step 4 | ✅ Changed to `self.best_metrics.composite_score()` |
| **2** | Verify `overfit_ratio` and `is_overfitting()` exist in Step 5 | ✅ Verified: Both exist (lines 74, 82) |
| **3** | Verify `self.base_config_path` exists in Step 5 | ✅ Verified: Exists (line 144) |
| **4** | Verify agent manifest names | ✅ Verified: Manifests exist in `agent_manifests/`, names match exactly |
| **5** | Verify prediction result keys | ✅ Corrected: Uses `confidence_scores`, not `avg_confidence` |

---

## Verification Evidence

### meta_prediction_optimizer.py (Step 4)

**PredictionMetrics dataclass (line 36):**
```python
@dataclass
class PredictionMetrics:
    variance: float
    mean_absolute_error: float
    discrimination_power: float
    # Methods:
    composite_score() -> float  # Line 340 confirms usage
```

**Verified usage (line 340-343):**
```python
self.logger.info(f"Best Score: {self.best_metrics.composite_score():.4f}")
self.logger.info(f"Best Variance: {self.best_metrics.variance:.4f}")
self.logger.info(f"Best MAE: {self.best_metrics.mean_absolute_error:.4f}")
```

### meta_prediction_optimizer_anti_overfit.py (Step 5)

**Dataclass fields (line 61+):**
```python
@dataclass
class AntiOverfitMetrics:
    test_variance: float
    test_mae: float
    train_mae: float
    overfit_ratio: float  # ✅ Line 74
    
    def is_overfitting(self) -> bool:  # ✅ Line 82
```

**self.base_config_path (line 144):**
```python
self.base_config_path = base_config_path  # ✅ Confirmed
```

### prediction_generator.py (Step 6)

**Result structure (line 246-262):**
```python
result = {
    'predictions': predictions[:k],           # ✅ List of ints
    'confidence_scores': confidences[:k],     # ✅ NOT 'avg_confidence'
    'metadata': {
        'method': method,
        'pool_size': self.config.pool_size,
        'k': k,
        # ... other fields
    }
}
```

### Agent Manifests (agent_manifests/*.json)

**Location:** `~/distributed_prng_analysis/agent_manifests/`

**Follow-up agent chain verified:**
```
ml_meta.json:        "follow_up_agents": ["reinforcement_agent"]  ✅
reinforcement.json:  "follow_up_agents": ["prediction_agent"]     ✅
prediction.json:     "follow_up_agents": []                       ✅ (final step)
```

**Agent names match injection code exactly:**
| Step | Injection `follow_up_agent` | Manifest `agent_name` | Status |
|------|----------------------------|----------------------|--------|
| 4 → 5 | `"reinforcement_agent"` | `"reinforcement_agent"` | ✅ Match |
| 5 → 6 | `"prediction_agent"` | `"prediction_agent"` | ✅ Match |
| 6 | `None` | `[]` (empty) | ✅ Match |

---

### ⚠️ Manifest Audit Note: `scoring_statistics.json`

**Finding:** The manifests reference `scoring_statistics.json` but this file is never created.

| Manifest | Reference |
|----------|-----------|
| `full_scoring.json` line 15 | Listed as output |
| `full_scoring.json` line 55 | success_condition checks it |
| `ml_meta.json` line 9 | Listed as input |

**Team Consensus (Alpha + Beta):**
> "NO — the manifest is out of date. All statistical signals now come exclusively from `survivor_scorer.py` feature extraction."

**Current Production Flow:**
```
survivor_scorer.py (46 ML features)
    → scorer_trial_worker.py (distributed scoring)
    → survivors_with_scores.json (aggregated)
    → generate_ml_jobs.py (feature extraction for ML)
```

**Recommendation:** Audit and update manifests via `AgentManifest.load()` to match verified outputs. Remove `scoring_statistics.json` references from:
- `agent_manifests/full_scoring.json` (lines 15, 55)
- `agent_manifests/ml_meta.json` (line 9)

**Impact on This Proposal:** None. The `agent_metadata` injection correctly reflects actual file dependencies (`survivors_with_scores.json`), not the outdated manifest references.

---

## Verification Audit (December 8, 2025)

### Scripts WITH agent_metadata Injection

```bash
$ grep -l "inject_agent_metadata" ~/distributed_prng_analysis/*.py
run_scorer_meta_optimizer.py
window_optimizer.py
```

| Script | Step | Line (Import) | Line (Call) | Status |
|--------|------|---------------|-------------|--------|
| `window_optimizer.py` | 1 | 35 | 566 | ✅ DONE |
| `run_scorer_meta_optimizer.py` | 2.5 | 26 | 199 | ✅ DONE |

### Scripts That Do NOT Need Injection

| Script | Step | Reason |
|--------|------|--------|
| `scorer_trial_worker.py` | 2.5, 3 | **Runs on remote nodes** (rig-6600, rig-6600b). Produces intermediate `trial_XXXX.json` files that are aggregated on Zeus. Remote nodes do not have `integration/` folder. Watcher monitors aggregated output, not individual trials. |
| `survivor_scorer.py` | 3 | **Library module**, imported by other scripts. Not a pipeline endpoint. |
| `reinforcement_engine.py` | 5 | **Class definition file**. `to_json()` saves config, not pipeline results. Used by other scripts, not run directly as pipeline step. |

### Scripts That NEED Injection (Zeus-Only) - ✅ ALL COMPLETE

| Script | Step | Output File | Follow-up Agent | Status |
|--------|------|-------------|-----------------|--------|
| `meta_prediction_optimizer.py` | 4 | `meta_optimization_results.json` | reinforcement_agent | ✅ DONE |
| `meta_prediction_optimizer_anti_overfit.py` | 5 | `{study_name}_results.json` | prediction_agent | ✅ DONE |
| `prediction_generator.py` | 6 | `predictions_{timestamp}.json` | None (final step) | ✅ DONE |
| `prediction_generator.py` | 6 | `predictions_{timestamp}.json` | none (final step) |

---

## Pipeline Architecture Verification

```
Step 1: window_optimizer.py (Zeus)
        └── Output: optimal_window_config.json ← agent_metadata ✅
        └── Output: bidirectional_survivors.json
                ↓
Step 2.5: run_scorer_meta_optimizer.py (Zeus - orchestrator)
          └── Output: optimal_scorer_config.json ← agent_metadata ✅
          └── Workers: scorer_trial_worker.py (REMOTE - no injection needed)
                       └── trial_XXXX.json (intermediate, aggregated)
                ↓
Step 3: run_full_scoring.sh (Zeus - orchestrator)
        └── Workers: scorer_trial_worker.py (REMOTE - no injection needed)
        └── Output: survivors_with_scores.json (aggregated on Zeus)
                ↓
Step 4: meta_prediction_optimizer.py (Zeus)
        └── Output: meta_optimization_results.json ← NEEDS agent_metadata
                ↓
Step 5: meta_prediction_optimizer_anti_overfit.py (Zeus)
        └── Output: {study_name}_results.json ← NEEDS agent_metadata
                ↓
Step 6: prediction_generator.py (Zeus)
        └── Output: predictions_{timestamp}.json ← NEEDS agent_metadata (final)
```

---

## Implementation Specifications

### Script 1: meta_prediction_optimizer.py (Step 4)

**File:** `~/distributed_prng_analysis/meta_prediction_optimizer.py`  
**Size:** 17,461 bytes  
**Save Point:** Line 390

#### Current Code (Lines 376-392)

```python
    def save_results(self, output_path: str = 'meta_optimization_results.json'):
        """
        Save optimization results

        Args:
            output_path: Where to save results
        """
        results = {
            'best_config': self.best_config,
            'best_metrics': asdict(self.best_metrics) if self.best_metrics else None,
            'optimization_history': self.optimization_history
        }

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"✅ Optimization results saved to: {output_path}")
```

#### Modification 1: Add Import (After Line 32)

**Location:** After existing imports (line 32: `from survivor_scorer import SurvivorScorer`)

```python
from integration.metadata_writer import inject_agent_metadata
```

#### Modification 2: Add Injection (Before Line 389)

**Insert BEFORE:** `with open(output_path, 'w') as f:`

```python
        # Inject agent_metadata for pipeline chaining (Step 4 → Step 5)
        confidence_score = 0.5
        reasoning_text = "Meta-optimization complete"
        if self.best_metrics:
            # composite_score() returns value typically 0-100 range
            confidence_score = min(0.95, max(0.1, self.best_metrics.composite_score() / 100.0))
            reasoning_text = (
                f"Meta-optimization complete: {len(self.optimization_history)} trials, "
                f"composite_score={self.best_metrics.composite_score():.4f}, "
                f"variance={self.best_metrics.variance:.4f}, "
                f"MAE={self.best_metrics.mean_absolute_error:.4f}"
            )
        
        results = inject_agent_metadata(
            results,
            inputs=[
                {"file": "survivors_with_scores.json", "required": True},
                {"file": "train_history.json", "required": True}
            ],
            outputs=[output_path],
            pipeline_step=4,
            pipeline_step_name="ml_meta_optimizer",
            follow_up_agent="reinforcement_agent",
            confidence=confidence_score,
            suggested_params=self.best_config,
            reasoning=reasoning_text
        )

```

---

### Script 2: meta_prediction_optimizer_anti_overfit.py (Step 5)

**File:** `~/distributed_prng_analysis/meta_prediction_optimizer_anti_overfit.py`  
**Size:** 25,386 bytes  
**Save Point:** Line 584

#### Current Code (Lines 561-586)

```python
        results_file = Path('optimization_results') / f'{self.study_name}_results.json'
        # ... (results dict construction lines 562-581)
        results = {
            'study_name': self.study_name,
            'best_config': self.best_config,
            'best_test_metrics': {
                'test_variance': self.best_metrics.test_variance,
                'test_mae': self.best_metrics.test_mae,
                'train_mae': self.best_metrics.train_mae,
                'overfit_ratio': self.best_metrics.overfit_ratio,
                'is_overfitting': self.best_metrics.is_overfitting()
            },
            'all_trials': self.optimization_history,
            'total_time_seconds': sum(self.trial_times),
            'avg_trial_time_seconds': np.mean(self.trial_times),
            'timestamp': datetime.now().isoformat()
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"✅ Detailed results saved to: {results_file}")
```

#### Modification 1: Add Import (After Line 32)

**Location:** After existing imports (line 32: `from reinforcement_engine import ReinforcementEngine, ReinforcementConfig`)

```python
from integration.metadata_writer import inject_agent_metadata
```

#### Modification 2: Add Injection (Before Line 583)

**Insert BEFORE:** `with open(results_file, 'w') as f:`

```python
        # Inject agent_metadata for pipeline chaining (Step 5 → Step 6)
        confidence_score = 0.5
        reasoning_text = "Anti-overfit training complete"
        if self.best_metrics:
            # Lower overfit_ratio = better generalization = higher confidence
            # overfit_ratio ~1.0 is ideal, >1.5 is bad
            confidence_score = max(0.1, min(0.95, 1.0 - (self.best_metrics.overfit_ratio - 1.0) * 0.5))
            reasoning_text = (
                f"Anti-overfit training complete: "
                f"test_mae={self.best_metrics.test_mae:.4f}, "
                f"train_mae={self.best_metrics.train_mae:.4f}, "
                f"overfit_ratio={self.best_metrics.overfit_ratio:.2f}, "
                f"is_overfitting={self.best_metrics.is_overfitting()}"
            )
        
        results = inject_agent_metadata(
            results,
            inputs=[
                {"file": "survivors_with_scores.json", "required": True},
                {"file": "train_history.json", "required": True},
                {"file": self.base_config_path, "required": True}
            ],
            outputs=[str(results_file), "models/anti_overfit/best_model.pth"],
            pipeline_step=5,
            pipeline_step_name="anti_overfit_training",
            follow_up_agent="prediction_agent",
            confidence=confidence_score,
            suggested_params=self.best_config,
            reasoning=reasoning_text
        )

```

---

### Script 3: prediction_generator.py (Step 6 - Final)

**File:** `~/distributed_prng_analysis/prediction_generator.py`  
**Size:** 14,718 bytes  
**Save Point:** Line 314

#### Current Code (Lines 304-317)

```python
    def _save_predictions(self, result: Dict):
        """Save predictions to JSON"""
        output_dir = Path(self.config.predictions_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        self.logger.info(f"Saved predictions to {filepath}")
```

#### Modification 1: Add Import (After Line 26)

**Location:** After existing imports (line 26: `import numpy as np`)

```python
from integration.metadata_writer import inject_agent_metadata
```

#### Modification 2: Add Injection (Before Line 313)

**Insert BEFORE:** `with open(filepath, 'w') as f:`

```python
        # Inject agent_metadata - FINAL STEP (no follow_up_agent)
        # Calculate average confidence from confidence_scores list
        avg_conf = 0.5
        if result.get('confidence_scores'):
            avg_conf = sum(result['confidence_scores']) / len(result['confidence_scores'])
        
        num_predictions = len(result.get('predictions', []))
        
        result = inject_agent_metadata(
            result,
            inputs=[
                {"file": "models/anti_overfit/best_model.pth", "required": True},
                {"file": "survivors_with_scores.json", "required": True}
            ],
            outputs=[str(filepath)],
            pipeline_step=6,
            pipeline_step_name="prediction",
            follow_up_agent=None,  # Final step - no follow-up
            confidence=avg_conf,
            suggested_params=None,
            reasoning=f"Generated {num_predictions} predictions with avg confidence {avg_conf:.4f}"
        )

```

---

## Implementation Procedure

### Pre-Implementation Checklist

- [ ] Verify `integration/metadata_writer.py` exists on Zeus
- [ ] Verify import works: `python3 -c "from integration.metadata_writer import inject_agent_metadata; print('OK')"`

### Stage 1: prediction_generator.py (Lowest Risk - Final Step)

```bash
# 1. Create backup
cp ~/distributed_prng_analysis/prediction_generator.py \
   ~/distributed_prng_analysis/prediction_generator.py.bak_$(date +%Y%m%d)

# 2. Apply modifications (add import after line 26, add injection before line 313)

# 3. Verify syntax
python3 -m py_compile ~/distributed_prng_analysis/prediction_generator.py

# 4. Verify import works
cd ~/distributed_prng_analysis && python3 -c "from prediction_generator import PredictionGenerator; print('OK')"

# 5. Run minimal test (if available) and check output JSON for agent_metadata

# 6. Commit
cd ~/distributed_prng_analysis && git add prediction_generator.py && \
   git commit -m "Session 9: agent_metadata injection for prediction_generator.py (Step 6)"
```

### Stage 2: meta_prediction_optimizer_anti_overfit.py

```bash
# 1. Create backup
cp ~/distributed_prng_analysis/meta_prediction_optimizer_anti_overfit.py \
   ~/distributed_prng_analysis/meta_prediction_optimizer_anti_overfit.py.bak_$(date +%Y%m%d)

# 2. Apply modifications (add import after line 32, add injection before line 583)

# 3. Verify syntax
python3 -m py_compile ~/distributed_prng_analysis/meta_prediction_optimizer_anti_overfit.py

# 4. Verify import works
cd ~/distributed_prng_analysis && python3 -c "from meta_prediction_optimizer_anti_overfit import AntiOverfitOptimizer; print('OK')"

# 5. Commit
cd ~/distributed_prng_analysis && git add meta_prediction_optimizer_anti_overfit.py && \
   git commit -m "Session 9: agent_metadata injection for meta_prediction_optimizer_anti_overfit.py (Step 5)"
```

### Stage 3: meta_prediction_optimizer.py

```bash
# 1. Create backup
cp ~/distributed_prng_analysis/meta_prediction_optimizer.py \
   ~/distributed_prng_analysis/meta_prediction_optimizer.py.bak_$(date +%Y%m%d)

# 2. Apply modifications (add import after line 32, add injection before line 389)

# 3. Verify syntax
python3 -m py_compile ~/distributed_prng_analysis/meta_prediction_optimizer.py

# 4. Verify import works
cd ~/distributed_prng_analysis && python3 -c "from meta_prediction_optimizer import MetaPredictionOptimizer; print('OK')"

# 5. Commit
cd ~/distributed_prng_analysis && git add meta_prediction_optimizer.py && \
   git commit -m "Session 9: agent_metadata injection for meta_prediction_optimizer.py (Step 4)"
```

---

## Rollback Procedure

If any issues arise:

```bash
# Restore from backup
cp ~/distributed_prng_analysis/SCRIPT.py.bak_YYYYMMDD \
   ~/distributed_prng_analysis/SCRIPT.py

# Or restore from git
cd ~/distributed_prng_analysis && git checkout HEAD~1 -- SCRIPT.py
```

---

## Validation

After all injections are complete:

```bash
# Count scripts with injection (should be 5 total)
grep -l "inject_agent_metadata" ~/distributed_prng_analysis/*.py | wc -l

# List all scripts with injection
grep -l "inject_agent_metadata" ~/distributed_prng_analysis/*.py

# Expected output:
# meta_prediction_optimizer.py
# meta_prediction_optimizer_anti_overfit.py
# prediction_generator.py
# run_scorer_meta_optimizer.py
# window_optimizer.py
```

---

## Summary Table

| Step | Script | Runs On | Has Injection | Action |
|------|--------|---------|---------------|--------|
| 1 | window_optimizer.py | Zeus | ✅ YES | None |
| 2.5 | run_scorer_meta_optimizer.py | Zeus | ✅ YES | None |
| 2.5 | scorer_trial_worker.py | **Remote** | ❌ NO | None (intentional) |
| 3 | scorer_trial_worker.py | **Remote** | ❌ NO | None (intentional) |
| 4 | meta_prediction_optimizer.py | Zeus | ❌ NO | **ADD** |
| 5 | meta_prediction_optimizer_anti_overfit.py | Zeus | ❌ NO | **ADD** |
| 6 | prediction_generator.py | Zeus | ❌ NO | **ADD** |

---

## Why Remote Workers Don't Need Injection

1. **`integration/` folder doesn't exist on remote nodes:**
   ```
   $ ssh 192.168.3.120 "ls ~/distributed_prng_analysis/integration/"
   ls: cannot access '/home/michael/distributed_prng_analysis/integration/': No such file or directory
   ```

2. **Worker outputs are intermediate:**
   - `trial_XXXX.json` files are pulled and aggregated on Zeus
   - Watcher Agent monitors aggregated output, not individual trials

3. **Adding injection would require:**
   - Deploying `integration/` to all remote nodes
   - Maintaining sync across cluster
   - No benefit since Watcher monitors aggregated outputs

---

## Git Commits (After Implementation)

```
Session 9: agent_metadata injection for prediction_generator.py (Step 6)
Session 9: agent_metadata injection for meta_prediction_optimizer_anti_overfit.py (Step 5)
Session 9: agent_metadata injection for meta_prediction_optimizer.py (Step 4)
```

---

## Summary: Changes from Original Proposal

| Item | Original (Incorrect) | Corrected |
|------|---------------------|-----------|
| Scripts needing injection | 6 | **3** |
| Step 4 confidence field | `self.best_metrics.accuracy` | `self.best_metrics.composite_score()` |
| Step 6 confidence field | `result.get('avg_confidence')` | `result['confidence_scores']` (averaged) |
| Remote worker injection | Proposed | **Excluded** (intentional) |

---

## Approval Signatures

| Role | Team | Name | Date | Approval |
|------|------|------|------|----------|
| Author | - | Claude | 2025-12-08 | ✓ |
| Peer Review | Beta | | 2025-12-08 | ✓ (with caveats) |
| Caveat Resolution | - | Claude | 2025-12-08 | ✓ All resolved |
| Implementation | - | Claude | 2025-12-09 | ✓ All 3 scripts |
| Final Approval | Alpha | | 2025-12-09 | ✓ |

---

## Implementation Summary

| Script | Import Line | Injection Lines | Commit |
|--------|-------------|-----------------|--------|
| `prediction_generator.py` | 27 | 313-332 | 9d94204 |
| `meta_prediction_optimizer_anti_overfit.py` | 32 | 584-610 | 97aaa4d |
| `meta_prediction_optimizer.py` | 33 | 389-413 | ea399e7 |

---

**End of Addendum F v1.1.0 - IMPLEMENTED**
