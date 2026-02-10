# PROPOSAL: Modular Step Automation Framework v1.1
## Focus: Step 1 Window Optimizer | Architecture: All Steps

**Date:** January 2, 2026  
**Author:** Claude (Architecture Assistant)  
**Status:** DRAFT - Awaiting Team Beta Approval  
**Priority:** HIGH - Core Autonomy Infrastructure  

---

## 0. Project Goal

### What We Are Building

A **distributed PRNG analysis and functional mimicry system** capable of:

1. **Reverse-engineering unknown PRNGs** - Given only output sequences (e.g., lottery draws), identify the underlying pseudo-random number generator algorithm and its parameters

2. **Seed reconstruction** - Brute-force and intelligently sieve candidate seeds using GPU-accelerated forward/reverse filtering across a 26-GPU cluster

3. **Functional mimicry** - Train ML models to predict future outputs by learning patterns in PRNG behavior, even when the exact algorithm remains uncertain

4. **Autonomous operation** - Execute a 6-step analysis pipeline with minimal human intervention, using PhD-level AI reasoning to evaluate results and make informed decisions

### Why Automation Matters

The pipeline involves complex statistical analysis at each step. Parameters like window sizes, thresholds, and PRNG hypotheses require expert-level judgment to tune correctly. Without automation:

- Human must manually review each step's output
- Human must have PhD-level statistics knowledge to interpret results
- Human must decide parameter adjustments for retries
- Process is slow and error-prone

**With automation:**

- LLM-Math (Qwen2.5-Math-7B) provides PhD-level statistical evaluation
- WATCHER agent orchestrates execution and decision-making
- Human only intervenes on true edge cases (ESCALATE)
- System learns from failures via fingerprint registry (no infinite retry loops)

### The 6-Step Pipeline

```
Step 1: Window Optimizer      → Find optimal temporal windows + generate survivors
Step 2.5: Scorer Meta         → Optimize scoring parameters
Step 3: Full Scoring          → Score all survivors with ML features
Step 4: Adaptive Meta         → Capacity planning for training
Step 5: Anti-Overfit Training → Train ML model with K-Fold validation
Step 6: Prediction Generator  → Generate predictions from trained model
```

**Target autonomy: 85%+** with human oversight only for edge cases.

---

## 1. Executive Summary

### 1.1 Vision

Build a **single modular framework** that automates any pipeline step through configuration, not code changes. Implement and validate with Step 1, then extend to Steps 2-6 by adding config files only.

### 1.2 Two Operating Modes

| Mode | Trigger | Intelligence |
|------|---------|--------------|
| **Human-Directed** | `--run-step 1 --params {...}` | Human reviews results, decides next action |
| **PhD-Autonomous** | `--run-step 1 --autonomous` | LLM-Math evaluates results, makes informed decisions |

### 1.3 Design Principle

```
ONE CODEBASE + PER-STEP CONFIG = ALL STEPS AUTOMATED
```

| To Automate New Step | Code Changes | Config Changes |
|----------------------|--------------|----------------|
| Step 1 | Build framework | Create step1 configs |
| Step 2 | None | Create step2 configs |
| Step 3 | None | Create step3 configs |
| Step 4 | None | Create step4 configs |
| Step 5 | None | Create step5 configs |
| Step 6 | None | Create step6 configs |

---

## 2. Current Infrastructure

### 2.1 What We Have ✅

| Component | Status | Location |
|-----------|--------|----------|
| Dual-LLM (Qwen) | ✅ Operational | GPU0: 8080, GPU1: 8081 |
| LLM Router | ✅ Complete | `llm_services/llm_router.py` |
| GBNF Grammar Support | ✅ Complete | `llm_services/grammars/` |
| Agent Manifests (6 steps) | ✅ Complete | `agent_manifests/*.json` |
| Watcher Agent (skeleton) | ✅ Partial | `agents/watcher_agent.py` |
| Pydantic Context Framework | ✅ Complete | `agents/contexts/` |
| Fingerprint Registry | ✅ Complete | `agents/fingerprint_registry.py` |
| Registry Hooks | ✅ Complete | `agents/watcher_registry_hooks.py` |

### 2.2 Step 1 Manifest (Verified)

**File:** `agent_manifests/window_optimizer.json`

```json
{
  "pipeline_step": 1,
  "script": "window_optimizer.py",
  "args_map": {
    "lottery-file": "lottery_file",
    "trials": "window_trials",
    "prng-type": "prng_type",
    "max-seeds": "seed_count"
  },
  "outputs": ["bidirectional_survivors.json", "optimal_window_config.json"],
  "success_condition": "optimal_window_config.json exists AND survivors > 0",
  "success_metrics": {
    "min_bidirectional_survivors": 1
  },
  "default_params": {
    "lottery_file": "synthetic_lottery.json",
    "strategy": "bayesian",
    "trials": 50,
    "prng_type": "java_lcg"
  },
  "retry": 2,
  "timeout_minutes": 240,
  "follow_up_agents": ["scorer_meta_agent"]
}
```

---

## 3. What's Missing

### 3.1 Generic Step Runner

A reusable execution engine that:
- Loads any step manifest
- Builds subprocess command
- Executes with timeout
- Validates outputs
- Extracts metrics

### 3.2 LLM Evaluation Layer

PhD-level intelligence that:
- Evaluates step results statistically
- Recommends PROCEED / RETRY / ESCALATE
- Suggests parameter adjustments
- Works via config (prompts + grammars), not hardcoded logic

### 3.3 Step-Specific Configurations

| Config Type | Purpose | One Per Step |
|-------------|---------|--------------|
| Manifest | Script, args, outputs | ✅ Already exists |
| Eval Prompt | PhD-level evaluation template | 🆕 Needed |
| Metrics Extractor | What to extract from outputs | 🆕 Needed (in manifest) |

---

## 4. Proposed Architecture

### 4.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MODULAR STEP RUNNER                                  │
│                    (Generic Code - All Steps)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      STEP CONFIGURATIONS                             │    │
│  │                      (Per-Step Files)                                │    │
│  │                                                                      │    │
│  │   agent_manifests/          agent_prompts/       agent_grammars/    │    │
│  │   ├── window_optimizer.json ├── step1_eval.txt   └── step_decision  │    │
│  │   ├── scorer_meta.json      ├── step2_eval.txt        .gbnf         │    │
│  │   ├── full_scoring.json     ├── step3_eval.txt                      │    │
│  │   ├── ml_meta.json          ├── step4_eval.txt                      │    │
│  │   ├── reinforcement.json    ├── step5_eval.txt                      │    │
│  │   └── prediction.json       └── step6_eval.txt                      │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      GENERIC COMPONENTS                              │    │
│  │                      (Reusable Code)                                 │    │
│  │                                                                      │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │   │   Manifest   │  │   Command    │  │    Step      │              │    │
│  │   │   Loader     │  │   Builder    │  │   Executor   │              │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  │                                                                      │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │    │
│  │   │   Output     │  │   Metrics    │  │     LLM      │              │    │
│  │   │  Validator   │  │  Extractor   │  │  Evaluator   │              │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘              │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      DECISION ENGINE                                 │    │
│  │                                                                      │    │
│  │   Human-Directed Mode          PhD-Autonomous Mode                  │    │
│  │   ──────────────────           ─────────────────────                │    │
│  │   Execute → Report             Execute → LLM Evaluate → Decide      │    │
│  │   Human decides next           LLM decides next                     │    │
│  │                                (GBNF-constrained)                   │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            WATCHER AGENT                                     │
│                                                                              │
│   run_step(step=1, params={...}, mode="human"|"autonomous")                 │
│   run_pipeline(start=1, end=6, mode="autonomous")                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 File Structure

```
agents/
├── watcher_agent.py                    # Main orchestrator (wire in StepRunner)
├── fingerprint_registry.py             # ✅ Exists
├── watcher_registry_hooks.py           # ✅ Exists
│
└── step_runner/                        # 🆕 NEW MODULE
    ├── __init__.py                     # Public API
    ├── models.py                       # Pydantic models (StepManifest, StepResult, etc.)
    ├── manifest_loader.py              # Load JSON → StepManifest
    ├── command_builder.py              # StepManifest + params → command list
    ├── step_executor.py                # Run subprocess, capture output
    ├── output_validator.py             # Check files exist, extract metrics
    ├── metrics_extractor.py            # Step-specific metric extraction
    ├── llm_evaluator.py                # Generic LLM evaluation via config
    └── runner.py                       # Main StepRunner class (ties it together)

agent_prompts/                          # 🆕 NEW DIRECTORY
├── step1_eval.txt                      # Step 1 PhD evaluation prompt
├── step2_eval.txt                      # Step 2.5 PhD evaluation prompt
├── step3_eval.txt                      # Step 3 PhD evaluation prompt
├── step4_eval.txt                      # Step 4 PhD evaluation prompt
├── step5_eval.txt                      # Step 5 PhD evaluation prompt
└── step6_eval.txt                      # Step 6 PhD evaluation prompt

agent_grammars/                         # 🆕 NEW DIRECTORY (or extend existing)
└── step_decision.gbnf                  # Shared decision grammar (all steps)
```

---

## 5. Execution Flow

### 5.1 Human-Directed Mode

```
User: python3 watcher_agent.py --run-step 1 --lottery-file synthetic.json

┌─────────────────────────────────────────────────────────────────────┐
│ 1. Load Manifest                                                     │
│    → agent_manifests/window_optimizer.json                          │
├─────────────────────────────────────────────────────────────────────┤
│ 2. Build Command                                                     │
│    → ["python3", "window_optimizer.py", "--lottery-file", ...]      │
├─────────────────────────────────────────────────────────────────────┤
│ 3. Execute (subprocess)                                              │
│    → Wait for completion, capture stdout/stderr                      │
├─────────────────────────────────────────────────────────────────────┤
│ 4. Validate Outputs                                                  │
│    → Check: bidirectional_survivors.json exists? survivors > 0?     │
├─────────────────────────────────────────────────────────────────────┤
│ 5. Extract Metrics                                                   │
│    → survivor_count: 45,000                                         │
│    → forward_survivors: 52,000                                      │
│    → reverse_survivors: 48,000                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 6. Report to Human                                                   │
│                                                                      │
│    ╔════════════════════════════════════════════════════════════╗   │
│    ║ STEP 1 COMPLETE: Window Optimizer                          ║   │
│    ╠════════════════════════════════════════════════════════════╣   │
│    ║ Status: ✅ SUCCESS                                         ║   │
│    ║ Duration: 47 minutes                                       ║   │
│    ║ Survivors: 45,000                                          ║   │
│    ║ Config: optimal_window_config.json                         ║   │
│    ╠════════════════════════════════════════════════════════════╣   │
│    ║ Next: Run --run-step 2 to continue                         ║   │
│    ╚════════════════════════════════════════════════════════════╝   │
│                                                                      │
│ 7. Human Reviews, Decides Next Action                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 PhD-Autonomous Mode

```
User: python3 watcher_agent.py --run-step 1 --autonomous

┌─────────────────────────────────────────────────────────────────────┐
│ 1-5. Same as Human-Directed (Load → Execute → Validate → Metrics)   │
├─────────────────────────────────────────────────────────────────────┤
│ 6. LLM Evaluation (PhD-Level)                                        │
│                                                                      │
│    ┌─────────────────────────────────────────────────────────────┐  │
│    │ Load: agent_prompts/step1_eval.txt                          │  │
│    │                                                              │  │
│    │ Prompt to LLM-Math:                                         │  │
│    │ ─────────────────────────────────────────────────────────── │  │
│    │ You are a PhD statistician analyzing PRNG sieve results.    │  │
│    │                                                              │  │
│    │ ## Results                                                   │  │
│    │ - Survivors: 45,000                                         │  │
│    │ - Seeds tested: 10,000,000                                  │  │
│    │ - Survival rate: 0.45%                                      │  │
│    │ - PRNG: java_lcg                                            │  │
│    │ - Window: 768                                               │  │
│    │ - Thresholds: fwd=0.72, rev=0.81                            │  │
│    │                                                              │  │
│    │ ## Evaluate                                                  │  │
│    │ 1. Is survival rate statistically reasonable?               │  │
│    │ 2. Sufficient statistical power for ML training?            │  │
│    │ 3. Thresholds appropriately tuned?                          │  │
│    │ ─────────────────────────────────────────────────────────── │  │
│    │                                                              │  │
│    │ LLM Response (GBNF-constrained):                            │  │
│    │ {                                                           │  │
│    │   "assessment": "ACCEPTABLE",                               │  │
│    │   "confidence": 0.82,                                       │  │
│    │   "reasoning": "0.45% survival from 10M seeds yields        │  │
│    │                 45K candidates - sufficient for ML.         │  │
│    │                 Threshold balance appears reasonable.",      │  │
│    │   "recommendation": "PROCEED",                              │  │
│    │   "parameter_adjustments": {}                               │  │
│    │ }                                                           │  │
│    └─────────────────────────────────────────────────────────────┘  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ 7. Execute Decision                                                  │
│                                                                      │
│    Decision: PROCEED                                                 │
│    → Auto-trigger Step 2.5 (Scorer Meta-Optimizer)                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Step 1 Specific Configurations

### 6.1 Evaluation Prompt

**File:** `agent_prompts/step1_eval.txt`

```
You are a PhD statistician specializing in PRNG analysis and reverse-engineering.

## Step 1 Results: Window Optimizer

### Execution Metrics
- Duration: {duration_seconds} seconds
- Exit code: {exit_code}

### Survivor Statistics  
- Bidirectional survivors: {survivor_count:,}
- Forward-only survivors: {forward_survivors:,}
- Reverse-only survivors: {reverse_survivors:,}
- Intersection ratio: {intersection_ratio:.2%}

### Search Parameters
- PRNG type: {prng_type}
- Seeds tested: {seed_count:,}
- Survival rate: {survival_rate:.4%}
- Window size: {window_size}
- Forward threshold: {forward_threshold}
- Reverse threshold: {reverse_threshold}
- Skip mode: {skip_mode}

## Your Evaluation

Analyze these results as a PhD statistician would:

1. **Statistical Validity**: Is the survival rate reasonable for {prng_type}? 
   - Expected range for LCG-family: 0.01% - 0.5%
   - Expected range for XOR-family: 0.05% - 1.0%
   
2. **ML Readiness**: Are there sufficient survivors for robust ML training?
   - Minimum recommended: 10,000 survivors
   - Optimal: 50,000+ survivors
   
3. **Threshold Balance**: Are forward/reverse thresholds appropriately tuned?
   - Too loose (< 0.70): High false positive rate
   - Too tight (> 0.90): May miss true positives
   
4. **PRNG Hypothesis**: Does the survival pattern suggest correct PRNG family?

Based on your analysis, provide your assessment and recommendation.
```

### 6.2 Metrics Extraction Config

**Added to:** `agent_manifests/window_optimizer.json`

```json
{
  "metrics_extraction": {
    "from_output_files": {
      "bidirectional_survivors.json": {
        "survivor_count": "len(data)",
        "forward_survivors": "len([s for s in data if s.get('direction') == 'forward'])",
        "reverse_survivors": "len([s for s in data if s.get('direction') == 'reverse'])"
      },
      "optimal_window_config.json": {
        "window_size": "data.get('window_size')",
        "forward_threshold": "data.get('forward_threshold')",
        "reverse_threshold": "data.get('reverse_threshold')",
        "skip_mode": "data.get('skip_mode', 'constant')"
      }
    },
    "computed": {
      "survival_rate": "survivor_count / seed_count",
      "intersection_ratio": "survivor_count / max(forward_survivors, 1)"
    }
  },
  "evaluation": {
    "prompt_file": "agent_prompts/step1_eval.txt",
    "grammar_file": "agent_grammars/step_decision.gbnf",
    "llm_model": "math"
  }
}
```

### 6.3 Decision Grammar (Shared)

**File:** `agent_grammars/step_decision.gbnf`

```gbnf
root ::= "{" ws members ws "}"

members ::= assessment-member "," ws confidence-member "," ws reasoning-member "," ws recommendation-member "," ws adjustments-member

assessment-member ::= "\"assessment\"" ws ":" ws assessment-value
assessment-value ::= "\"OPTIMAL\"" | "\"ACCEPTABLE\"" | "\"SUBOPTIMAL\"" | "\"FAILED\""

confidence-member ::= "\"confidence\"" ws ":" ws number

reasoning-member ::= "\"reasoning\"" ws ":" ws string

recommendation-member ::= "\"recommendation\"" ws ":" ws recommendation-value
recommendation-value ::= "\"PROCEED\"" | "\"RETRY\"" | "\"CHANGE_PRNG\"" | "\"ADJUST_THRESHOLD\"" | "\"ESCALATE\""

adjustments-member ::= "\"parameter_adjustments\"" ws ":" ws "{" ws param-list? ws "}"

param-list ::= param-item ("," ws param-item)*
param-item ::= string ws ":" ws (string | number | "true" | "false")

ws ::= [ \t\n\r]*
number ::= "-"? [0-9]+ ("." [0-9]+)?
string ::= "\"" ([^"\\] | "\\" .)* "\""
```

---

## 7. Implementation Plan

### Phase 1: Core Models (~100 lines)

**File:** `agents/step_runner/models.py`

```python
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
from enum import Enum

class AssessmentLevel(str, Enum):
    OPTIMAL = "OPTIMAL"
    ACCEPTABLE = "ACCEPTABLE"
    SUBOPTIMAL = "SUBOPTIMAL"
    FAILED = "FAILED"

class RecommendationAction(str, Enum):
    PROCEED = "PROCEED"
    RETRY = "RETRY"
    CHANGE_PRNG = "CHANGE_PRNG"
    ADJUST_THRESHOLD = "ADJUST_THRESHOLD"
    ESCALATE = "ESCALATE"

class StepManifest(BaseModel):
    """Typed representation of agent manifest."""
    agent_name: str
    pipeline_step: int
    script: str
    args_map: Dict[str, str]
    outputs: List[str]
    success_condition: str
    success_metrics: Dict[str, Any] = {}
    default_params: Dict[str, Any] = {}
    metrics_extraction: Dict[str, Any] = {}
    evaluation: Dict[str, str] = {}
    retry: int = 2
    timeout_minutes: int = 240
    follow_up_agents: List[str] = []

class StepResult(BaseModel):
    """Result of step execution."""
    step: int
    step_name: str
    success: bool
    exit_code: int
    duration_seconds: int
    outputs_found: Dict[str, bool]
    metrics: Dict[str, Any]
    error_message: Optional[str] = None

class StepDecision(BaseModel):
    """LLM evaluation decision."""
    assessment: AssessmentLevel
    confidence: float
    reasoning: str
    recommendation: RecommendationAction
    parameter_adjustments: Dict[str, Any] = {}

class RunMode(str, Enum):
    HUMAN = "human"
    AUTONOMOUS = "autonomous"
```

### Phase 2: Manifest Loader + Command Builder (~80 lines)

**File:** `agents/step_runner/manifest_loader.py`

```python
MANIFEST_MAP = {
    1: "window_optimizer.json",
    2: "scorer_meta.json",
    3: "full_scoring.json",
    4: "ml_meta.json",
    5: "reinforcement.json",
    6: "prediction.json"
}

def load_manifest(step: int, manifest_dir: str = "agent_manifests") -> StepManifest:
    """Load and parse manifest for given step."""
    ...
```

**File:** `agents/step_runner/command_builder.py`

```python
def build_command(manifest: StepManifest, params: Dict[str, Any]) -> List[str]:
    """Build subprocess command from manifest and params."""
    cmd = ["python3", manifest.script]
    
    for cli_arg, param_key in manifest.args_map.items():
        if param_key in params and params[param_key] is not None:
            value = params[param_key]
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{cli_arg}")
            else:
                cmd.extend([f"--{cli_arg}", str(value)])
    
    return cmd
```

### Phase 3: Step Executor (~100 lines)

**File:** `agents/step_runner/step_executor.py`

```python
def execute_step(
    command: List[str],
    timeout_minutes: int,
    work_dir: Path,
    step: int,
    step_name: str
) -> StepResult:
    """Execute step and capture result."""
    ...
```

### Phase 4: Output Validator + Metrics Extractor (~120 lines)

**File:** `agents/step_runner/output_validator.py`

```python
def validate_outputs(manifest: StepManifest, work_dir: Path) -> Dict[str, bool]:
    """Check which output files exist."""
    ...
```

**File:** `agents/step_runner/metrics_extractor.py`

```python
def extract_metrics(manifest: StepManifest, work_dir: Path, params: Dict) -> Dict[str, Any]:
    """Extract metrics from output files based on manifest config."""
    ...
```

### Phase 5: LLM Evaluator (~100 lines)

**File:** `agents/step_runner/llm_evaluator.py`

```python
class LLMEvaluator:
    """Generic LLM evaluation - works for any step via config."""
    
    def __init__(self, llm_router):
        self.router = llm_router
    
    def evaluate(
        self,
        step: int,
        metrics: Dict[str, Any],
        eval_config: Dict[str, str]
    ) -> StepDecision:
        """Evaluate step results using configured prompt + grammar."""
        
        # Load prompt template
        prompt_template = Path(eval_config["prompt_file"]).read_text()
        
        # Format with metrics
        prompt = prompt_template.format(**metrics)
        
        # Load grammar
        grammar_path = eval_config.get("grammar_file", "agent_grammars/step_decision.gbnf")
        
        # Query LLM
        if eval_config.get("llm_model") == "math":
            response = self.router.calculate(prompt, grammar=grammar_path)
        else:
            response = self.router.orchestrate(prompt, grammar=grammar_path)
        
        return StepDecision(**json.loads(response))
```

### Phase 6: Main Runner (~150 lines)

**File:** `agents/step_runner/runner.py`

```python
class StepRunner:
    """
    Generic step runner - works for any step via configuration.
    
    Usage:
        runner = StepRunner(llm_router)
        
        # Human-directed
        result = runner.run(step=1, params={...}, mode=RunMode.HUMAN)
        
        # Autonomous
        result, decision = runner.run(step=1, params={...}, mode=RunMode.AUTONOMOUS)
    """
    
    def __init__(self, llm_router=None, work_dir: str = "."):
        self.llm_router = llm_router
        self.work_dir = Path(work_dir)
        self.evaluator = LLMEvaluator(llm_router) if llm_router else None
    
    def run(
        self,
        step: int,
        params: Dict[str, Any],
        mode: RunMode = RunMode.HUMAN
    ) -> Tuple[StepResult, Optional[StepDecision]]:
        """
        Run a pipeline step.
        
        Args:
            step: Pipeline step number (1-6)
            params: Runtime parameters
            mode: HUMAN (report only) or AUTONOMOUS (LLM decides)
        
        Returns:
            (StepResult, StepDecision) - Decision only in AUTONOMOUS mode
        """
        # 1. Load manifest
        manifest = load_manifest(step)
        
        # 2. Merge default params
        full_params = {**manifest.default_params, **params}
        
        # 3. Build command
        command = build_command(manifest, full_params)
        logger.info(f"Executing: {' '.join(command)}")
        
        # 4. Execute
        result = execute_step(
            command=command,
            timeout_minutes=manifest.timeout_minutes,
            work_dir=self.work_dir,
            step=step,
            step_name=manifest.agent_name
        )
        
        # 5. Validate outputs
        result.outputs_found = validate_outputs(manifest, self.work_dir)
        
        # 6. Extract metrics
        result.metrics = extract_metrics(manifest, self.work_dir, full_params)
        
        # 7. Update success based on validation
        outputs_ok = all(result.outputs_found.values())
        result.success = result.exit_code == 0 and outputs_ok
        
        # 8. LLM Evaluation (autonomous mode only)
        decision = None
        if mode == RunMode.AUTONOMOUS and self.evaluator:
            eval_config = manifest.evaluation
            if eval_config:
                decision = self.evaluator.evaluate(step, result.metrics, eval_config)
        
        return result, decision
```

### Phase 7: WATCHER Integration (~80 lines)

**Added to:** `agents/watcher_agent.py`

```python
from step_runner import StepRunner, RunMode

class WatcherAgent:
    
    def __init__(self, ...):
        ...
        self.step_runner = StepRunner(llm_router=self.llm_router)
    
    def run_step(
        self,
        step: int,
        params: Dict[str, Any] = None,
        autonomous: bool = False
    ) -> StepResult:
        """
        Run a single pipeline step.
        
        Args:
            step: Step number (1-6)
            params: Runtime parameters (merged with manifest defaults)
            autonomous: If True, LLM evaluates and decides next action
        
        Returns:
            StepResult with metrics and status
        """
        mode = RunMode.AUTONOMOUS if autonomous else RunMode.HUMAN
        params = params or {}
        
        result, decision = self.step_runner.run(step, params, mode)
        
        # Report result
        self._report_result(result)
        
        # Handle decision in autonomous mode
        if decision:
            self._handle_decision(step, result, decision)
        
        return result
    
    def _handle_decision(self, step: int, result: StepResult, decision: StepDecision):
        """Execute the LLM's decision."""
        
        if decision.recommendation == RecommendationAction.PROCEED:
            next_step = self._get_next_step(step)
            if next_step:
                logger.info(f"AUTONOMOUS: Proceeding to Step {next_step}")
                self.run_step(next_step, autonomous=True)
                
        elif decision.recommendation == RecommendationAction.RETRY:
            if result.retry_count < manifest.retry:
                new_params = {**result.params, **decision.parameter_adjustments}
                logger.info(f"AUTONOMOUS: Retrying with {decision.parameter_adjustments}")
                self.run_step(step, new_params, autonomous=True)
                
        elif decision.recommendation == RecommendationAction.ESCALATE:
            logger.warning(f"AUTONOMOUS: Escalating to human - {decision.reasoning}")
            self._notify_human(step, result, decision)
```

---

## 8. CLI Interface

```bash
# Human-directed: Run Step 1, report results
python3 watcher_agent.py --run-step 1 \
    --lottery-file synthetic_lottery.json \
    --prng-type java_lcg

# Autonomous: Run Step 1, LLM evaluates and decides
python3 watcher_agent.py --run-step 1 \
    --lottery-file synthetic_lottery.json \
    --prng-type java_lcg \
    --autonomous

# Autonomous pipeline: Run Steps 1-6 with LLM decision-making
python3 watcher_agent.py --run-pipeline \
    --start-step 1 \
    --end-step 6 \
    --autonomous

# Check step status
python3 watcher_agent.py --status --step 1
```

---

## 9. Implementation Summary

| Phase | Component | Lines | Focus |
|-------|-----------|-------|-------|
| 1 | Pydantic Models | ~100 | All steps |
| 2 | Manifest Loader + Command Builder | ~80 | All steps |
| 3 | Step Executor | ~100 | All steps |
| 4 | Output Validator + Metrics | ~120 | All steps |
| 5 | LLM Evaluator | ~100 | All steps |
| 6 | Main Runner | ~150 | All steps |
| 7 | WATCHER Integration | ~80 | All steps |
| **Subtotal (Code)** | | **~730** | **Reusable** |
| | | | |
| Config | Step 1 Eval Prompt | ~40 | Step 1 only |
| Config | Decision Grammar | ~30 | All steps |
| **Total** | | **~800** | |

---

## 10. Extending to Remaining Steps

After Step 1 is working:

| Step | Work Required |
|------|---------------|
| Step 2.5 | Create `agent_prompts/step2_eval.txt` (~40 lines) |
| Step 3 | Create `agent_prompts/step3_eval.txt` (~40 lines) |
| Step 4 | Create `agent_prompts/step4_eval.txt` (~40 lines) |
| Step 5 | Create `agent_prompts/step5_eval.txt` + registry hook (~60 lines) |
| Step 6 | Create `agent_prompts/step6_eval.txt` (~40 lines) |

**Total for remaining steps: ~220 lines of config (no new code)**

---

## 11. Success Criteria

| Criterion | Target |
|-----------|--------|
| Step 1 runs in human-directed mode | ✅ |
| Step 1 runs in autonomous mode | ✅ |
| LLM provides PhD-level evaluation | ✅ |
| Decision is GBNF-constrained | ✅ |
| Framework is reusable for Steps 2-6 | ✅ |
| No hardcoded step logic in code | ✅ |

---

## 12. Approval Request

**Team Beta:** Please confirm:

1. ✅ / ❌ Architecture is modular and extensible
2. ✅ / ❌ Focus on Step 1 with clear path to remaining steps
3. ✅ / ❌ Two modes (human-directed / autonomous) are appropriate
4. ✅ / ❌ PhD-level evaluation via LLM is the right approach
5. ✅ / ❌ Ready to begin implementation

---

**END OF PROPOSAL v1.1**

---

## 13. Team Beta Clarifications (APPROVED)

**Status:** ✅ APPROVED with clarifications  
**Date:** January 2, 2026

### Clarification 1: Registry Check Before LLM-Suggested Actions

**Concern:** When LLM suggests RETRY or CHANGE_PRNG, registry might already know this will fail.

**Resolution:** WATCHER always checks registry before executing ANY action:

```python
def _handle_decision(self, step: int, result: StepResult, decision: StepDecision):
    """Execute the LLM's decision - WITH REGISTRY CHECK."""
    
    if decision.recommendation == RecommendationAction.RETRY:
        # ALWAYS check registry first
        new_params = {**result.params, **decision.parameter_adjustments}
        fingerprint = new_params.get("fingerprint")
        prng_type = new_params.get("prng_type")
        
        if fingerprint and prng_type:
            registry_check = self.registry_hooks.pre_run_check(fingerprint, prng_type)
            if registry_check.action == "SKIP_PRNG":
                logger.warning(f"Registry blocked LLM suggestion: {registry_check.reason}")
                # Use registry's suggestion instead
                new_params["prng_type"] = registry_check.suggested_prng
        
        self.run_step(step, new_params, autonomous=True)
```

**Rule:** Registry has veto power over LLM suggestions.

---

### Clarification 2: Retry Authority Hierarchy

**Concern:** Retries can come from manifest, LLM, or WATCHER policy. Who wins?

**Resolution:** Clear hierarchy documented:

```
┌─────────────────────────────────────────────────────────────┐
│                    RETRY AUTHORITY                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. MANIFEST defines maximum retry count (hard cap)         │
│     └── "retry": 2 → never exceed 2 retries                 │
│                                                              │
│  2. REGISTRY can block retries (veto power)                 │
│     └── "Already tried this" → skip, don't count as retry   │
│                                                              │
│  3. WATCHER enforces retry count (authority)                │
│     └── Tracks retry_count per step execution               │
│                                                              │
│  4. LLM may suggest retry (advisor only)                    │
│     └── WATCHER decides whether to honor suggestion         │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│  FINAL RULE:                                                 │
│  Manifest = Cap, Registry = Veto, WATCHER = Authority,      │
│  LLM = Advisor                                               │
└─────────────────────────────────────────────────────────────┘
```

---

### Clarification 3: Metrics Extractor Fails Loudly

**Concern:** Silent metric failures could propagate bad data to LLM decisions.

**Resolution:** Metrics extractor returns structured errors:

```python
@dataclass
class MetricsResult:
    success: bool
    metrics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

def extract_metrics(manifest: StepManifest, work_dir: Path, params: Dict) -> MetricsResult:
    """Extract metrics with structured error handling."""
    metrics = {}
    errors = []
    warnings = []
    
    for output_file, extractors in manifest.metrics_extraction.get("from_output_files", {}).items():
        file_path = work_dir / output_file
        
        if not file_path.exists():
            errors.append(f"Metrics file not found: {output_file}")
            continue
        
        try:
            with open(file_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in {output_file}: {e}")
            continue
        
        for metric_name, expression in extractors.items():
            try:
                # Safe eval with limited scope
                value = safe_eval(expression, {"data": data, "len": len})
                metrics[metric_name] = value
            except Exception as e:
                errors.append(f"Failed to extract {metric_name}: {e}")
                metrics[metric_name] = None  # Explicit null, not silent skip
    
    return MetricsResult(
        success=len(errors) == 0,
        metrics=metrics,
        errors=errors,
        warnings=warnings
    )
```

**Rule:** If metrics extraction fails:
- Step marked as `FAILED_METRICS_EXTRACTION`
- LLM evaluation skipped (cannot evaluate without metrics)
- WATCHER escalates to human

---

### Clarification 4: Registry Consulted Pre-Run AND Post-Run

**Resolution:** Explicit registry integration points:

```python
class StepRunner:
    
    def run(self, step: int, params: Dict, mode: RunMode) -> Tuple[StepResult, Optional[StepDecision]]:
        
        # ══════════════════════════════════════════════════════════
        # PRE-RUN: Registry check (for steps that use fingerprints)
        # ══════════════════════════════════════════════════════════
        if step == 5 and "fingerprint" in params:
            pre_check = self.registry_hooks.pre_run_check(
                params["fingerprint"], 
                params["prng_type"]
            )
            if pre_check.action in ("SKIP_PRNG", "REJECT_DATA_WINDOW", "BLOCK"):
                return self._handle_pre_run_block(pre_check)
        
        # ... execute step ...
        
        # ══════════════════════════════════════════════════════════
        # POST-RUN: Record to registry (for steps that emit fingerprints)
        # ══════════════════════════════════════════════════════════
        if step == 5 and result.metrics.get("fingerprint"):
            self.registry_hooks.post_run_record(
                fingerprint=result.metrics["fingerprint"],
                prng_type=params["prng_type"],
                outcome=self._map_exit_code_to_outcome(result.exit_code),
                sidecar_path=work_dir / "models/reinforcement/best_model.meta.json"
            )
        
        return result, decision
```

**Rule:** Registry is always consulted:
- **Pre-run:** Can this combination proceed?
- **Post-run:** Record outcome for future decisions

---

### Summary of Authority Model

| Component | Authority | Scope |
|-----------|-----------|-------|
| **Step Scripts** | Compute facts | Emit results, no decisions |
| **StepRunner** | Execute | Run commands, validate outputs |
| **Metrics Extractor** | Extract facts | Fail loudly on errors |
| **Registry** | Memory + Veto | Block known-bad combinations |
| **LLM** | Advisor | Suggest, never enforce |
| **WATCHER** | Orchestrator | Final decision authority |
| **Manifest** | Configuration | Define caps and rules |
| **Human** | Escalation target | Handle ESCALATE decisions |

---

## 14. Implementation Ready

**Team Beta Status:** ✅ APPROVED

**Clarifications:** ✅ All 4 addressed above

**Next Steps:**
1. Implement Phase 1: Core Models
2. Implement Phase 2: Manifest Loader + Command Builder
3. Implement Phase 3: Step Executor
4. Continue through phases...

---

**END OF PROPOSAL v1.1 (WITH TEAM BETA CLARIFICATIONS)**
