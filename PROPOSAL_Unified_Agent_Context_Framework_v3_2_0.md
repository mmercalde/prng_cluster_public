PROPOSAL_Unified_Agent_Context_Framework_v3_2_0.md
PROPOSAL_Unified_Agent_Context_Framework
_v3_2_0.md
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
            spec = None
            if registry:
                spec = registry.get_param(clean_name)
                if not spec:
                    # Try with original name
                    spec = registry.get_param(cli_flag)

            # Build parameter info
            param_info = ParameterInfo(
                name=clean_name,
                cli_flag=cli_flag,
                context_var=context_var,
                current_value=current_values.get(clean_name) or current_values.get(context_var)
            )

            # Enrich from registry spec
            if spec:
                param_info.param_type = spec.param_type.value
                param_info.description = spec.description
                param_info.min_value = spec.min_value
                param_info.max_value = spec.max_value
                param_info.choices = spec.choices
                param_info.default = spec.default

            parameters.append(param_info)

        return cls(
            script_name=script_name,
            parameters=parameters
        )

    def to_prompt_section(self) -> str:
        """
        Generate the ADJUSTABLE PARAMETERS section for LLM prompt.

        This is the key section that enables AI parameter adjustment.
        """
        if not self.parameters:
            return "No adjustable parameters available for this step."

        lines = [
            "════════════════════════════════════════════════════════════════════════════════",
            "ADJUSTABLE PARAMETERS",
            "════════════════════════════════════════════════════════════════════════════════",
            "",
            f"Script: {self.script_name}",
            "",
            "You may suggest adjustments to these parameters in your response:",
            ""
        ]

        for param in self.parameters:
            lines.append(param.to_prompt_line())
            lines.append("")

        lines.extend([
            "────────────────────────────────────────────────────────────────────────────────",
            "FORMAT FOR SUGGESTED ADJUSTMENTS:",
            '  "suggested_param_adjustments": {',
            '    "parameter_name": new_value,',
            '    ...',
            '  }',
            "────────────────────────────────────────────────────────────────────────────────"
        ])

        return "\n".join(lines)

    def validate_adjustments(self, adjustments: Dict[str, Any]) -> List[str]:
        """
        Validate suggested adjustments against parameter bounds.

        Returns list of error messages (empty if all valid).
        """
        errors = []

        for param_name, value in adjustments.items():
            # Find parameter spec
            param = next((p for p in self.parameters if p.name == param_name), None)

            if not param:
                errors.append(f"Unknown parameter: {param_name}")
                continue

            # Check choices
            if param.choices and value not in param.choices:
                errors.append(f"{param_name}: {value} not in {param.choices}")
                continue

            # Check numeric bounds
            if param.min_value is not None and value < param.min_value:
                errors.append(f"{param_name}: {value} below minimum {param.min_value}")
            if param.max_value is not None and value > param.max_value:
                errors.append(f"{param_name}: {value} above maximum {param.max_value}")

        return errors
```

---

## Part 3: Step-Specific Agent Contexts

Each pipeline step has its own context class that inherits from a base:

```python
# agents/contexts/base_context.py
"""
Base Agent Context - Common functionality for all pipeline steps.
"""

from pydantic import BaseModel, Field, computed_field
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

from agents.manifest.agent_manifest import AgentManifest
from agents.parameters.parameter_context import ParameterContext


class BaseAgentContext(BaseModel):
    """
    Base context for all pipeline step agents.

    Provides:
    - Manifest loading and access
    - Parameter context building
    - Common prompt generation
    """

    # Core identity
    step_name: str
    manifest: AgentManifest

    # Results from the step
    results: Dict[str, Any] = Field(default_factory=dict)

    # System description for this step
    system_description: str = ""

    # Cache for parameter context
    _parameter_context: Optional[ParameterContext] = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def parameter_context(self) -> ParameterContext:
        """Lazy-load parameter context."""
        if self._parameter_context is None:
            # Extract current values from results
            current_values = self._extract_current_params()
            self._parameter_context = ParameterContext.build(
                manifest=self.manifest,
                current_values=current_values
            )
        return self._parameter_context

    def _extract_current_params(self) -> Dict[str, Any]:
        """
        Extract current parameter values from results.

        Override in subclasses for step-specific extraction.
        """
        # Try common locations
        params = {}

        # From config section
        if "config" in self.results:
            params.update(self.results["config"])

        # From best_config
        if "best_config" in self.results:
            params.update(self.results["best_config"])

        # From parameters section
        if "parameters" in self.results:
            params.update(self.results["parameters"])

        # From analysis_parameters
        if "analysis_parameters" in self.results:
            params.update(self.results["analysis_parameters"])

        return params

    def to_prompt_body(self) -> str:
        """
        Generate the agent-specific prompt body.

        Called by FullAgentContext.to_prompt() after doctrine.
        """
        # Format results
        results_str = json.dumps(self.results, indent=2, default=str)
        if len(results_str) > 4000:
            results_str = results_str[:4000] + "\n... (truncated)"

        return f"""
════════════════════════════════════════════════════════════════════════════════
STEP {self.manifest.pipeline_step}: {self.step_name.upper()}
════════════════════════════════════════════════════════════════════════════════

{self.system_description}

────────────────────────────────────────────────────────────────────────────────
MANIFEST CONTRACT
────────────────────────────────────────────────────────────────────────────────

Agent: {self.manifest.agent_name}
Script: {self.manifest.script_name}
Success Condition: {self.manifest.success_condition}
Retries Allowed: {self.manifest.retry}
Follow-up Agents: {', '.join(self.manifest.follow_up_agents) or 'None'}

────────────────────────────────────────────────────────────────────────────────
CURRENT RESULTS
────────────────────────────────────────────────────────────────────────────────

{results_str}

{self.parameter_context.to_prompt_section()}
"""
```

---

## Part 4: Full Agent Context with Doctrine

```python
# agents/contexts/full_context.py
"""
Full Agent Context - Combines doctrine, step context, and decision format.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

from agents.contexts.base_context import BaseAgentContext


# The doctrine that appears at the top of every agent prompt
AGENT_DOCTRINE = '''
════════════════════════════════════════════════════════════════════════════════
PRNG ANALYSIS SYSTEM - AGENT DOCTRINE
════════════════════════════════════════════════════════════════════════════════

TERMINOLOGY (USE ONLY THESE DEFINITIONS):
─────────────────────────────────────────
• window: An ordered list of N lottery draw results used for validation
• sieve: A GPU kernel that eliminates non-matching seed candidates
• forward sieve: Tests seeds against draws in chronological order
• reverse sieve: Tests seeds against draws in reverse chronological order
• survivor: A seed candidate that passes the sieve filter
• bidirectional survivor: A seed that survives BOTH forward and reverse sieves
• skip: The number of PRNG iterations between consecutive draws
• threshold: Minimum match rate required for a seed to survive

DECISION FRAMEWORK:
───────────────────
1. ALWAYS respond with valid JSON matching the schema below
2. Base decisions on SUCCESS CONDITION in manifest contract
3. If success condition met → "proceed"
4. If not met but retries available → "retry" with parameter adjustments
5. If not met and no retries → "escalate" with explanation

YOUR ROLE:
──────────
You are an autonomous agent in a 6-step pipeline. Your job is to:
1. Analyze the results from your assigned step
2. Determine if success criteria are met
3. Recommend next action (proceed/retry/escalate)
4. Suggest parameter adjustments if retry needed
'''


class FullAgentContext(BaseModel):
    """
    Complete context for agent decision-making.

    Combines:
    - Doctrine (terminology, decision framework)
    - Step-specific context (results, parameters)
    - Decision format specification
    """

    agent_context: BaseAgentContext
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    run_id: Optional[str] = None

    def to_prompt(self) -> str:
        """Generate the complete prompt for the LLM."""
        return f"""
{AGENT_DOCTRINE}

{self.agent_context.to_prompt_body()}

════════════════════════════════════════════════════════════════════════════════
RESPOND WITH VALID JSON
════════════════════════════════════════════════════════════════════════════════

{{
  "success_condition_met": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Your analysis...",
  "recommended_action": "proceed" | "retry" | "escalate",
  "suggested_param_adjustments": {{}},  // if retry
  "warnings": []  // optional
}}

Respond with ONLY valid JSON, no markdown, no explanation outside JSON.
"""
```

---

## Part 5: Example Usage

```python
# Example: Creating context for Window Optimizer evaluation

from agents.manifest.agent_manifest import AgentManifest
from agents.contexts.base_context import BaseAgentContext
from agents.contexts.full_context import FullAgentContext

# Load manifest
manifest = AgentManifest.load("agent_manifests/window_optimizer.json")

# Create step context with results
step_context = BaseAgentContext(
    step_name="Window Optimizer",
    manifest=manifest,
    results={
        "best_config": {
            "window_size": 512,
            "skip_min": 0,
            "skip_max": 50,
            "forward_threshold": 0.72,
            "reverse_threshold": 0.81
        },
        "forward_survivors": 3184,
        "reverse_survivors": 3295,
        "bidirectional_survivors": 5,
        "best_score": 0.847
    },
    system_description="""
    The Window Optimizer uses Bayesian optimization (Optuna TPE) to find
    optimal window parameters for the forward and reverse sieves. It
    searches for the configuration that maximizes bidirectional survivor
    overlap while minimizing false positives.
    """
)

# Create full context
full_context = FullAgentContext(
    agent_context=step_context,
    run_id="step1_20251206_143022_abc123"
)

# Generate prompt for LLM
prompt = full_context.to_prompt()

# Send to LLM router
from llm_services.llm_router import get_router

router = get_router()
response = router.orchestrate(prompt, agent="window_optimizer_agent")
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

## Summary: v3.2.0 Core Additions

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

## Project Implementation Phases

This section defines the overall implementation roadmap for the PRNG Analysis Agent System.

### Phase Overview

| Phase | Component | Status | Session | Evidence |
|-------|-----------|--------|---------|----------|
| 1 | Schema v1.0.3/v1.0.4 | ✅ DONE | 1 | metadata_writer.py, llm_metadata fields |
| 2 | Dual-LLM Infrastructure | ✅ DONE | 1 | Qwen2.5-Coder-14B + Qwen2.5-Math-7B |
| 3 | Universal Agent v1.0 (BaseAgent) | ✅ DONE | 1 | agents/agent_core.py |
| 4 | Watcher Agent (Pipeline Manager) | ✅ DONE | 2 | agents/watcher_agent.py |
| 5 | Pydantic Context Framework | ✅ DONE | 1-4 | v3.2.0 + Addendums A-D |
| 6 | Web Dashboard | ✅ DONE | 5 | web_dashboard.py, gpu_monitor.py |

### Ongoing Work

| Item | Status | Description |
|------|--------|-------------|
| agent_metadata Injection | ⏳ IN PROGRESS | Injecting metadata into pipeline steps 3-6 |
| GBNF Phase 2: Supervisory LLM | 📝 PENDING | Semantic validation (Phi-4 or Claude API) |
| End-to-end Pipeline Test | 📝 PENDING | Full 26-GPU cluster validation |

### Phase Details

**Phase 1: Schema v1.0.3/v1.0.4**
- Extended JSON schema with `agent_metadata` block
- Added `llm_metadata` for dual-LLM routing
- Implemented metadata_writer.py for injection

**Phase 2: Dual-LLM Infrastructure**
- Deployed Qwen2.5-Coder-14B (port 8080) for code/technical tasks
- Deployed Qwen2.5-Math-7B (port 8081) for mathematical analysis
- Created LLM Router for intelligent task routing

**Phase 3: Universal Agent v1.0**
- Created BaseAgent class in agents/agent_core.py
- 6 specialized agent classes inheriting from BaseAgent
- Consistent interface across all pipeline steps

**Phase 4: Watcher Agent (Pipeline Manager)**
- Orchestrates full 6-step pipeline via `run_pipeline()`
- Evaluates results and makes proceed/retry/escalate decisions
- Supports daemon mode for continuous monitoring
- Integrates with LLM Router for decision-making

**Phase 5: Pydantic Context Framework**
- v3.2.0 specification with parameter introspection
- Addendum A: Step 1 threshold implementation
- Addendum B: Steps 2-6 manifest implementation (58 parameters)
- Addendum C: GBNF grammar for structural constraints
- Addendum D: Web dashboard implementation

**Phase 6: Web Dashboard**
- HiveOS-inspired Flask dashboard
- Interactive Plotly charts (scatter, convergence, distribution, importance)
- Live trial data with survivor counts
- GPU clock speed monitoring (NVIDIA + AMD)
- Optuna study persistence and visualization

### GBNF Two-Phase LLM Solution

Separate from the project phases, the LLM terminology drift issue is being addressed in two phases:

| LLM Phase | Component | Status |
|-----------|-----------|--------|
| Phase 1 | GBNF Grammar (Structural) | ✅ COMPLETE |
| Phase 2 | Supervisory Model (Semantic) | 📝 PENDING |

**Phase 1** ensures JSON structure is valid. **Phase 2** will add semantic validation via Phi-4 14B locally or Claude API.

---
---
---

# v3.2.0 Addendum A: Step 1 Threshold Implementation (December 5-6, 2025)

## Implementation Completed

The v3.2.0 proposal has been partially implemented for Step 1 (Window Optimizer) as a proof-of-concept. This addendum documents what was done and serves as a pattern for other steps.

---

## Problem Solved

**Original Issue:** Sieve thresholds were hardcoded to 0.01 (1%), causing:
- 50,000 false positive survivors (should be ~1-10)
- AI agents couldn't adjust thresholds
- Forward/reverse sieves used same threshold (should differ: 0.72 vs 0.81)

**Root Cause:** Violated the configurability principle:
> "Nothing should be hardcoded - all parameters must remain adjustable for ML and AI applications."

---

## Implementation Details

### 1. window_optimizer.py Changes

Added CLI arguments for threshold control:

```python
parser.add_argument('--forward-threshold', type=float, default=0.72,
                    help='Forward sieve threshold (default: 0.72 from Optuna)')
parser.add_argument('--reverse-threshold', type=float, default=0.81,
                    help='Reverse sieve threshold (default: 0.81 from Optuna)')
```

Modified Optuna objective to optimize thresholds:

```python
def objective(trial):
    # ... existing parameters ...
    forward_threshold = trial.suggest_float('forward_threshold', 0.50, 0.95)
    reverse_threshold = trial.suggest_float('reverse_threshold', 0.60, 0.98)
```

### 2. Manifest Update (window_optimizer.json v1.3.0)

Added `parameter_bounds` section:

```json
{
  "version": "1.3.0",
  "parameter_bounds": {
    "forward_threshold": {
      "type": "float",
      "min": 0.50,
      "max": 0.95,
      "default": 0.72,
      "description": "Minimum match rate for forward sieve survivors",
      "optimized_by": "Optuna TPE",
      "effect": "Higher = fewer survivors, more precision; Lower = more survivors, more recall"
    },
    "reverse_threshold": {
      "type": "float",
      "min": 0.60,
      "max": 0.98,
      "default": 0.81,
      "description": "Minimum match rate for reverse sieve survivors",
      "optimized_by": "Optuna TPE",
      "effect": "Higher = fewer survivors, more precision; Lower = more survivors, more recall"
    }
  }
}
```

### 3. Complete Manifest Example

```json
{
  "agent_name": "window_optimizer_agent",
  "description": "Optimizes window parameters using Bayesian optimization (Optuna TPE)",
  "pipeline_step": 1,
  "version": "1.3.0",
  "inputs": [
    "lottery_draws.json"
  ],
  "outputs": [
    "optimal_window_config.json",
    "bidirectional_survivors.json"
  ],
  "actions": [
    {
      "type": "run_script",
      "script": "window_optimizer.py",
      "args_map": {
        "--window-size": "window_size",
        "--offset": "offset",
        "--skip-min": "skip_min",
        "--skip-max": "skip_max",
        "--forward-threshold": "forward_threshold",
        "--reverse-threshold": "reverse_threshold",
        "--trials": "window_trials",
        "--seeds": "seed_count",
        "--strategy": "search_strategy",
        "--test-both-modes": "test_both_modes"
      },
      "distributed": true,
      "timeout_minutes": 120
    }
  ],
  "follow_up_agents": ["scorer_meta_agent"],
  "success_condition": "bidirectional_survivors >= 1 AND bidirectional_survivors <= 1000",
  "retry": 2,
  "parameter_bounds": {
    "window_size": {
      "type": "int",
      "min": 128,
      "max": 4096,
      "default": 512,
      "description": "Number of draws in validation window"
    },
    "offset": {
      "type": "int",
      "min": 0,
      "max": 2000,
      "default": 0,
      "description": "Starting position offset in draw history"
    },
    "skip_min": {
      "type": "int",
      "min": 0,
      "max": 50,
      "default": 0,
      "description": "Minimum PRNG iterations between draws"
    },
    "skip_max": {
      "type": "int",
      "min": 20,
      "max": 500,
      "default": 100,
      "description": "Maximum PRNG iterations between draws"
    },
    "forward_threshold": {
      "type": "float",
      "min": 0.50,
      "max": 0.95,
      "default": 0.72,
      "description": "Minimum match rate for forward sieve"
    },
    "reverse_threshold": {
      "type": "float",
      "min": 0.60,
      "max": 0.98,
      "default": 0.81,
      "description": "Minimum match rate for reverse sieve"
    },
    "window_trials": {
      "type": "int",
      "min": 10,
      "max": 200,
      "default": 50,
      "description": "Number of Optuna optimization trials"
    },
    "seed_count": {
      "type": "int",
      "min": 1000000,
      "max": 5000000000,
      "default": 50000000,
      "description": "Number of seed candidates to test"
    },
    "search_strategy": {
      "type": "choice",
      "choices": ["bayesian", "random", "grid"],
      "default": "bayesian",
      "description": "Optuna search strategy"
    },
    "test_both_modes": {
      "type": "bool",
      "default": false,
      "description": "Test both forward-first and reverse-first modes"
    }
  }
}
```

---

## Git Commits

- `bd95d6a` - Make thresholds configurable for ML/AI optimization
- `eec22ea` - Fix chunk sizing (19K min) and progress display type hints
- `b45c8cc` - Update CURRENT_Status.txt with Session 3

---

## Status

| Component | Status |
|-----------|--------|
| Step 1 (Window Optimizer) | ✅ Complete with parameter_bounds |
| Steps 2-6 | ✅ Complete (See Addendum B) |
| Pydantic Framework Integration | ✅ Ready (ParameterContext supports this) |

---

**End of v3.2.0 Addendum A**

---
---
---

# v3.2.0 Addendum B: Steps 2-6 Manifest Implementation (December 6, 2025)

## Implementation Completed

Following the Step 1 pattern established in Addendum A, all remaining pipeline steps now have complete manifests with `parameter_bounds` sections. This completes TODO #1 from instructions.txt.

---

## Files Created

| File | Step | Description | Parameters |
|------|------|-------------|------------|
| `window_optimizer.json` | 1 | Bayesian window optimization | 10 params |
| `scorer_meta.json` | 2 | Distributed scorer meta-optimizer | 6 params |
| `full_scoring.json` | 3 | Full survivor scoring | 5 params |
| `ml_meta.json` | 4 | ML architecture optimizer | 13 params |
| `reinforcement.json` | 5 | K-fold anti-overfit training | 13 params |
| `prediction.json` | 6 | Final prediction generation | 11 params |

**Total: 58 configurable parameters across 6 pipeline steps**

---

## Manifest Schema (v1.3.0)

Each manifest includes the standard fields plus the new `parameter_bounds` section:

```json
{
  "agent_name": "step_name_agent",
  "description": "What this step does",
  "pipeline_step": 1-6,
  "version": "1.3.0",
  "inputs": ["required_files"],
  "outputs": ["produced_files"],
  "actions": [{...}],
  "follow_up_agents": ["next_agent"],
  "success_condition": "validation expression",
  "retry": 2,
  "parameter_bounds": {
    "param_name": {
      "type": "int|float|bool|choice",
      "min": 0,
      "max": 100,
      "default": 50,
      "choices": ["a", "b", "c"],
      "description": "What this parameter does",
      "optimized_by": "Optuna TPE|Manual|Fixed",
      "effect": "How changing this affects behavior"
    }
  }
}
```

---

## Parameter Summary by Step

### Step 1: Window Optimizer
*(Implemented in Addendum A)*

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| window_size | int | 128-4096 | 512 |
| offset | int | 0-2000 | 0 |
| skip_min | int | 0-50 | 0 |
| skip_max | int | 20-500 | 100 |
| forward_threshold | float | 0.50-0.95 | 0.72 |
| reverse_threshold | float | 0.60-0.98 | 0.81 |
| window_trials | int | 10-200 | 50 |
| seed_count | int | 1M-5B | 50M |
| search_strategy | choice | bayesian/random/grid | bayesian |
| test_both_modes | bool | true/false | false |

### Step 2: Scorer Meta

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| chunk_size | int | 100-10000 | 1000 |
| scorer_trials | int | 10-500 | 100 |
| match_weight | float | 0.1-2.0 | 1.0 |
| residue_weight | float | 0.0-1.5 | 0.5 |
| temporal_weight | float | 0.0-1.5 | 0.3 |
| min_score_threshold | float | 0.01-0.50 | 0.10 |

### Step 3: Full Scoring

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| chunk_size | int | 500-20000 | 5000 |
| batch_size | int | 64-1024 | 256 |
| feature_extraction_depth | int | 46-128 | 64 |
| score_precision | int | 4-8 | 6 |
| parallel_workers | int | 1-26 | 26 |

### Step 4: ML Meta

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| ml_trials | int | 20-500 | 100 |
| timeout_hours | float | 1.0-48.0 | 8.0 |
| parallel_jobs | int | 1-4 | 2 |
| min_layers | int | 1-5 | 2 |
| max_layers | int | 3-10 | 6 |
| min_neurons | int | 16-128 | 32 |
| max_neurons | int | 128-1024 | 512 |
| learning_rate_min | float | 1e-6 - 1e-4 | 1e-5 |
| learning_rate_max | float | 1e-3 - 1e-1 | 1e-2 |
| dropout_min | float | 0.0-0.3 | 0.1 |
| dropout_max | float | 0.3-0.7 | 0.5 |
| weight_decay_min | float | 1e-7 - 1e-5 | 1e-6 |
| weight_decay_max | float | 1e-4 - 1e-2 | 1e-3 |

### Step 5: Reinforcement

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| k_folds | int | 3-15 | 5 |
| trials_per_fold | int | 10-100 | 50 |
| epochs_min | int | 10-50 | 20 |
| epochs_max | int | 50-500 | 200 |
| early_stopping_patience | int | 5-50 | 20 |
| batch_size | int | 16-256 | 64 |
| learning_rate | float | 1e-5 - 1e-2 | 1e-3 |
| lr_scheduler_factor | float | 0.1-0.9 | 0.5 |
| lr_scheduler_patience | int | 3-20 | 10 |
| gradient_clip_norm | float | 0.5-5.0 | 1.0 |
| validation_split | float | 0.10-0.30 | 0.20 |
| reinforce_weight | float | 0.0-1.0 | 0.5 |
| temporal_window | int | 10-200 | 50 |

### Step 6: Prediction

| Parameter | Type | Range | Default |
|-----------|------|-------|---------|
| pool_size | int | 5-100 | 20 |
| confidence_threshold | float | 0.50-0.95 | 0.70 |
| ensemble_mode | choice | single/weighted/voting/stacking | weighted |
| temporal_decay | float | 0.90-1.0 | 0.98 |
| drift_sensitivity | float | 0.0-1.0 | 0.3 |
| forward_weight | float | 0.3-0.7 | 0.5 |
| bidirectional_bonus | float | 1.0-3.0 | 1.5 |
| recency_window | int | 10-100 | 30 |
| min_survivor_history | int | 5-50 | 10 |
| skip_mode_preference | choice | constant/variable/adaptive/both | adaptive |
| explanation_verbosity | choice | minimal/standard/detailed/debug | standard |

---

## Deployment Instructions

```bash
# On Zeus, from the distributed_prng_analysis directory:
cd ~/distributed_prng_analysis

# Create directory if not exists
mkdir -p agent_manifests

# Extract downloaded archive
tar -xzvf agent_manifests_v1.3.0.tar.gz

# Verify
ls -la agent_manifests/
```

Expected output:
```
window_optimizer.json
scorer_meta.json
full_scoring.json
ml_meta.json
reinforcement.json
prediction.json
```

---

## Integration with ParameterContext

These manifests work with the `ParameterContext` class from the main proposal:

```python
from agents.parameters.parameter_context import ParameterContext
from agents.manifest.agent_manifest import AgentManifest

# Load manifest
manifest = AgentManifest.load('agent_manifests/reinforcement.json')

# Build parameter context with current values
context = ParameterContext.build(
    manifest=manifest,
    current_values={'k_folds': 5, 'learning_rate': 0.001}
)

# Generate prompt section for AI agent
prompt_section = context.to_prompt_section()

# Validate AI's suggested adjustments
errors = context.validate_adjustments({
    'k_folds': 10,           # valid
    'learning_rate': 0.1     # invalid - above max 0.01
})
```

---

## Git Commit Message

```
feat(manifests): Add parameter_bounds to Steps 2-6 (v1.3.0)

Following Step 1 pattern from Addendum A:
- scorer_meta.json: 6 parameters
- full_scoring.json: 5 parameters
- ml_meta.json: 13 parameters
- reinforcement.json: 13 parameters
- prediction.json: 11 parameters

Completes TODO #1 from instructions.txt
Reference: PROPOSAL v3.2.0 Addendum B
```

---

## Status

| Component | Status |
|-----------|--------|
| Step 1 (Window Optimizer) | ✅ Complete (Addendum A) |
| Step 2 (Scorer Meta) | ✅ Manifest Complete |
| Step 3 (Full Scoring) | ✅ Manifest Complete |
| Step 4 (ML Meta) | ✅ Manifest Complete |
| Step 5 (Reinforcement) | ✅ Manifest Complete |
| Step 6 (Prediction) | ✅ Manifest Complete |
| ParameterContext Integration | ✅ Ready |

---

**End of v3.2.0 Addendum B**

---
---
---

# v3.2.0 Addendum C: GBNF Grammar Implementation (December 6, 2025)

## Problem Identified

**Team Beta Report:** Qwen2.5-Coder-14B and Qwen2.5-Math-7B cannot maintain custom PRNG terminology definitions despite explicit system prompts.

**Symptoms:**
- "window" (ordered draw list) → interpreted as segment tree/sliding window
- "sieve" (GPU seed elimination kernel) → interpreted as prime sieve/Eratosthenes
- "forward/reverse" → interpreted as neural network passes
- "survivor" → interpreted as survival analysis

**Root Cause:** 7-14B parameter models lack capacity to override strongly embedded terminology priors from pretraining. This affects 70-80% of prompts and breaks multi-turn consistency.

**Impact on Agent System:**
- Agents make incorrect decisions based on misinterpreted results
- Parameter suggestions don't align with actual system parameters
- Pipeline automation blocked by unreliable LLM reasoning

---

## Solution: Two-Phase Approach

### Phase 1: GBNF Grammar (Structural Constraints) - ✅ COMPLETE

GBNF (GGML BNF) grammars constrain LLM output to valid JSON structures at the output layer.

**What GBNF Does:**
| Layer | Controls | Does NOT Control |
|-------|----------|------------------|
| Output Structure | Forces valid JSON syntax | ❌ Semantic reasoning |
| Field Names | Ensures correct keys exist | ❌ Value correctness |
| Enum Values | Limits to proceed/retry/escalate | ❌ Decision quality |

**Files Deployed:**
```
grammars/
├── agent_decision.gbnf       # Agent evaluation responses
├── sieve_analysis.gbnf       # Sieve result interpretation
├── parameter_adjustment.gbnf # Parameter change suggestions
└── json_generic.gbnf         # Fallback for any valid JSON
```

### Phase 2: Supervisory Model (Semantic Validation) - 📝 PENDING

A larger model validates reasoning before actions are taken.

**Options Under Consideration:**
1. **Phi-4 14B** - Test locally first (~9GB VRAM)
2. **Claude API** - Guaranteed solution (~$6/month)

**Architecture:**
```
User Prompt
    ↓
Supervisory Model (validates terminology, logic)
    ↓
Qwen-Coder (code generation) OR Qwen-Math (calculations)
    ↓
GBNF Grammar Filter (ensures valid JSON)
    ↓
Validated Output
```

---

## Phase 1 Implementation Details

### Grammar Files

**agent_decision.gbnf** - Forces agent evaluation structure:
```gbnf
root ::= "{" ws success-kv "," ws confidence-kv "," ws reasoning-kv "," ws action-kv ws "}"

success-kv ::= "\"success_condition_met\"" ws ":" ws boolean
confidence-kv ::= "\"confidence\"" ws ":" ws number
reasoning-kv ::= "\"reasoning\"" ws ":" ws string
action-kv ::= "\"recommended_action\"" ws ":" ws action-enum

action-enum ::= "\"proceed\"" | "\"retry\"" | "\"escalate\""
boolean ::= "true" | "false"
number ::= [0-9] "." [0-9] [0-9]
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape
escape ::= ["\\/bfnrt]
ws ::= [ \t\n]*
```

**Output Structure Guaranteed:**
```json
{
  "success_condition_met": true,
  "confidence": 0.85,
  "reasoning": "...",
  "recommended_action": "proceed"
}
```

### LLM Router v1.1.0 Changes

**New Parameters in `route()` method:**
```python
def route(self, prompt: str,
          ...
          grammar: Optional[str] = None,           # Raw grammar string
          grammar_type: Optional[GrammarType] = None,  # Grammar type enum
          auto_grammar: Optional[bool] = None) -> str:  # Override auto-select
```

**New Convenience Methods:**
```python
# Structured output with guaranteed JSON format
decision = router.evaluate_decision(prompt)      # → Dict
analysis = router.analyze_sieve(prompt)          # → Dict
params = router.suggest_parameters(prompt)       # → Dict
```

**New CLI Options:**
```bash
python3 llm_router.py --list-grammars
python3 llm_router.py --query "..." --grammar agent_decision
python3 llm_router.py --query "..." --no-grammar
```

**New Metrics:**
- `grammar_enforced_calls` - Count of grammar-constrained requests
- Grammar name included in trace entries

### Testing Results

```bash
# Health check
$ python3 llm_services/llm_router.py --health
LLM Server Health Status (v1.1.0):
  ✅ orchestrator: HEALTHY
  ✅ math: HEALTHY
  ✅ grammars_available: HEALTHY

# Grammar test
$ python3 llm_services/llm_router.py \
  --query "Evaluate: bidirectional=5, threshold=3. JSON decision:" \
  --grammar agent_decision --endpoint orchestrator

Response:
{"success_condition_met":true,"confidence":0.85,"reasoning":"...","recommended_action":"proceed"}
```

### Observed Limitation

During testing, the model returned:
```json
{
  "success_condition_met": false,
  "reasoning": "5 is less than 3..."
}
```

This confirms:
- ✅ Grammar enforcement works (valid JSON produced)
- ❌ Semantic reasoning broken (5 < 3 is obviously wrong)
- ✅ Phase 2 (Supervisory Model) is necessary

---

## Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `grammars/agent_decision.gbnf` | Created | Agent decision grammar |
| `grammars/sieve_analysis.gbnf` | Created | Sieve analysis grammar |
| `grammars/parameter_adjustment.gbnf` | Created | Parameter grammar |
| `grammars/json_generic.gbnf` | Created | Generic JSON grammar |
| `llm_services/grammar_loader.py` | Created | Grammar loading utility |
| `llm_services/llm_router.py` | Modified | v1.0.5 → v1.1.0 |
| `llm_services/llm_router_v1.0.5_backup.py` | Created | Backup |

---

## Integration with Agent System

### Before (v1.0.5)
```python
response = router.route(prompt)
# Could return invalid JSON, unparseable output
decision = json.loads(response)  # Might fail
```

### After (v1.1.0)
```python
# Option 1: Explicit grammar type
response = router.route(prompt, grammar_type="agent_decision")
decision = json.loads(response)  # Guaranteed valid JSON

# Option 2: Convenience method
decision = router.evaluate_decision(prompt)  # Returns Dict directly
```

### Watcher Agent Integration

```python
# In watcher_agent.py
def evaluate_step_results(self, results: Dict) -> AgentDecision:
    prompt = self.build_evaluation_prompt(results)

    # Use grammar-enforced evaluation
    decision_dict = self.router.evaluate_decision(
        prompt,
        agent=self.agent_name
    )

    # Guaranteed structure, but verify logic in Phase 2
    return AgentDecision(**decision_dict)
```

---

## Status

| Component | Status |
|-----------|--------|
| GBNF Grammar Files | ✅ Deployed |
| Grammar Loader | ✅ Deployed |
| LLM Router v1.1.0 | ✅ Deployed |
| Structural Enforcement | ✅ Working |
| Semantic Reasoning | ❌ Still broken (expected) |
| Phase 2 (Supervisory) | 📝 Pending team decision |

---

## Next Steps (Phase 2)

1. **Option A: Test Phi-4 14B Locally**
   - Download microsoft/phi-4 Q4_K_M (~9GB)
   - Test terminology retention
   - If successful, deploy as supervisory model

2. **Option B: Claude API Integration**
   - Create anthropic client wrapper
   - Route semantic decisions to Claude
   - Keep Qwen for code/math
   - ~$6/month operational cost

**Team Recommendation:** Test Phi-4 first (zero cost), Claude as guaranteed fallback.

---

**End of v3.2.0 Addendum C**
---

# Addendum D: Web Dashboard & Real-Time Monitoring

**Date:** December 6-7, 2025 (Session 5)
**Author:** Claude (AI Assistant)
**Focus:** HiveOS-Inspired Dashboard, Interactive Visualizations, Live Trial Data

---

## Overview

Session 5 addressed TODO #2 (Progress Display & Visualization) by creating a comprehensive web-based dashboard that replaces the tmux-based monitoring approach. The new system provides real-time cluster monitoring, interactive Optuna visualizations, and live trial data display.

---

## Major Components Implemented

### 1. Flask Web Dashboard (`web_dashboard.py`)

**Location:** `~/distributed_prng_analysis/web_dashboard.py`
**Access:** `http://192.168.3.127:5000`

**Features:**
- HiveOS-inspired dark theme design
- Multi-route architecture with 5 functional tabs
- Auto-refresh (2 seconds) on active pages
- Real-time GPU clock speed monitoring

**Routes:**
| Route | Purpose |
|-------|---------|
| `/` | Overview - Progress, worker summary, live trial data |
| `/workers` | Detailed per-GPU stats with clock speeds |
| `/stats` | Hardware summary, cluster totals |
| `/plots` | Interactive Optuna visualizations |
| `/settings` | Configuration options |

### 2. Interactive Plotly Charts

Replaced static matplotlib charts with interactive Plotly visualizations:

| Chart | Description |
|-------|-------------|
| Parameter Optimization Scatter | X/Y parameter plot with score coloring |
| Trial Convergence | Score progression with best-so-far line |
| Score Distribution | Histogram of trial scores |
| Parameter Importance | Horizontal bar chart of parameter influence |

**Features:**
- Hover tooltips showing detailed trial data
- Zoom/pan capabilities
- Full-screen view option for each chart
- Study selector dropdown for multiple Optuna studies

### 3. Live Trial Data Card

New dashboard component showing real-time optimization progress:

```
┌─────────────────────────────────────────────┐
│ Live Trial Data                    Trial #5 │
│ Window[500] Skip[1-10] Offset[0]            │
│                                             │
│  Forward      Reverse     Bidir    Best     │
│  12,345       8,901      1,234    2,567     │
│  Survivors    Survivors   Match   So Far    │
└─────────────────────────────────────────────┘
```

### 4. GPU Clock Speed Monitoring

**File:** `~/distributed_prng_analysis/gpu_monitor.py`

Monitors real GPU activity across the cluster:

| Node | Method | Metrics |
|------|--------|---------|
| Zeus (localhost) | `nvidia-smi` | Clock (MHz), Temp, Utilization |
| rig-6600 | `rocm-smi --showclocks` | SCLK (shader clock) |
| rig-6600b | `rocm-smi --showclocks` | SCLK (shader clock) |

**Display:**
- Overview page shows average clock per worker
- Workers page shows per-GPU clock speeds
- Color coding: Green (>1000 MHz), Orange (>100 MHz), Gray (idle)

### 5. ProgressWriter Enhancement

**File:** `~/distributed_prng_analysis/progress_display.py`

Added `update_trial_stats()` method to track:
- Trial number
- Forward survivors count
- Reverse survivors count
- Bidirectional survivors count
- Best bidirectional so far
- Configuration description

---

## Implementation Details

### Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    web_dashboard.py                         │
├─────────────────────────────────────────────────────────────┤
│  Flask Routes     │  Chart Generators  │  Data Sources      │
│  ─────────────    │  ────────────────  │  ────────────      │
│  /                │  generate_heatmap  │  cluster_progress  │
│  /workers         │  _plotly()         │  .json             │
│  /stats           │  generate_         │                    │
│  /plots           │  convergence_      │  gpu_monitor.py    │
│  /settings        │  plotly()          │                    │
│  /plot/<type>     │  generate_         │  optuna_studies/   │
│  /api/progress    │  distribution_     │  *.db              │
│  /health          │  plotly()          │                    │
│                   │  generate_         │                    │
│                   │  importance_       │                    │
│                   │  plotly()          │                    │
└─────────────────────────────────────────────────────────────┘
```

### Optuna Study Persistence Fix

**Problem:** Window optimizer studies were created in-memory and lost after completion.

**Solution:** Added persistent storage to `window_optimizer_bayesian.py`:

```python
# Before (in-memory, lost after run)
study = optuna.create_study(direction='maximize', sampler=sampler)

# After (persisted to disk)
study_name = f"window_opt_{int(time.time())}"
storage_path = f"sqlite:///optuna_studies/{study_name}.db"
study = optuna.create_study(
    study_name=study_name,
    storage=storage_path,
    direction='maximize',
    sampler=sampler
)
```

### Parameter Importance Implementation

Uses Optuna's built-in importance calculation:

```python
from optuna.importance import get_param_importances

def generate_importance_plotly(study_name):
    study = optuna.load_study(study_name=study_name, storage=storage)
    importances = get_param_importances(study)  # Requires 4+ trials
    # Generate horizontal bar chart...
```

---

## Files Modified/Created

### New Files
| File | Purpose |
|------|---------|
| `web_dashboard.py` | Main Flask dashboard (~1650 lines) |
| `gpu_monitor.py` | GPU clock/stats monitoring |

### Modified Files
| File | Changes |
|------|---------|
| `progress_display.py` | Added `update_trial_stats()` method |
| `window_optimizer_integration_final.py` | Added trial stats updates |
| `window_optimizer_bayesian.py` | Added Optuna study persistence |
| `coordinator.py` | Auto-starts web dashboard |

---

## Usage

### Start Dashboard
```bash
python3 ~/distributed_prng_analysis/web_dashboard.py &
```

### Access Points
- Local: `http://localhost:5000`
- Network: `http://192.168.3.127:5000`

### Run Optimization with Dashboard
```bash
python3 window_optimizer.py --strategy bayesian --trials 15 \
    --max-seeds 1000000 --prng-type java_lcg \
    --lottery-file synthetic_lottery.json
```

---

## TODO #2 Status

| Item | Status |
|------|--------|
| Fix tmux auto-attach | ✅ Replaced with web dashboard |
| Live GPU utilization | ✅ Clock speed monitoring |
| Optuna trial progress | ✅ Interactive Plotly charts |
| Survivor counts live | ✅ Live Trial Data card |
| ETA estimation | ✅ In summary bar |
| Seaborn heatmap | ✅ Replaced with Plotly interactive |
| Parameter Importance | ✅ Implemented |

---

## Known Issues / Future Work

1. **AMD rocm-smi parsing:** Some GPUs may not report (device visibility varies)
2. **Settings page:** Currently UI-only, not functional
3. **Historical data:** No persistence between dashboard restarts
4. **WebSocket:** Currently uses meta-refresh (2s), could upgrade to WebSocket

---

## Session 5 Git Commits

*(To be committed)*
- `web_dashboard.py` - HiveOS-inspired dashboard
- `gpu_monitor.py` - GPU clock monitoring
- `progress_display.py` - Trial stats updates
- `window_optimizer_bayesian.py` - Study persistence
- `window_optimizer_integration_final.py` - Dashboard integration

---

**End of v3.2.0 Addendum D**
