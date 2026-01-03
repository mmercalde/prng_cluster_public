#!/usr/bin/env python3
"""
Step Runner Models - Pydantic models for step automation
=========================================================
Version: 1.0.0
Date: 2026-01-02

Provides typed models for:
- StepManifest: Parsed agent manifest
- StepResult: Execution result with metrics
- StepDecision: LLM evaluation decision
- MetricsResult: Extracted metrics with error handling
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# ENUMS
# =============================================================================

class AssessmentLevel(str, Enum):
    """LLM assessment of step results."""
    OPTIMAL = "OPTIMAL"           # Results exceed expectations
    ACCEPTABLE = "ACCEPTABLE"     # Results meet minimum thresholds
    SUBOPTIMAL = "SUBOPTIMAL"     # Results below expectations but recoverable
    FAILED = "FAILED"             # Results indicate fundamental failure


class RecommendationAction(str, Enum):
    """LLM recommended action after evaluation."""
    PROCEED = "PROCEED"                   # Continue to next step
    RETRY = "RETRY"                       # Retry same step with adjustments
    CHANGE_PRNG = "CHANGE_PRNG"           # Try different PRNG hypothesis
    ADJUST_THRESHOLD = "ADJUST_THRESHOLD" # Adjust sieve thresholds
    EXPAND_WINDOW = "EXPAND_WINDOW"       # Increase window/holdout size
    ESCALATE = "ESCALATE"                 # Require human intervention


class RunMode(str, Enum):
    """Step execution mode."""
    HUMAN = "human"           # Execute, report, human decides
    AUTONOMOUS = "autonomous" # Execute, LLM evaluates, LLM decides


class StepStatus(str, Enum):
    """Step execution status."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    FAILED_METRICS = "FAILED_METRICS_EXTRACTION"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"
    BLOCKED = "BLOCKED"


# =============================================================================
# MANIFEST MODEL
# =============================================================================

class EvaluationConfig(BaseModel):
    """LLM evaluation configuration from manifest."""
    prompt_file: str = ""
    grammar_file: str = "agent_grammars/step_decision.gbnf"
    llm_model: str = "math"  # "math" or "general"


class MetricsExtractionConfig(BaseModel):
    """Metrics extraction configuration from manifest."""
    from_output_files: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    computed: Dict[str, str] = Field(default_factory=dict)


class ActionConfig(BaseModel):
    """
    Configuration for a single action within a step.
    
    Steps can have multiple actions executed sequentially.
    """
    type: str = "run_script"  # "run_script" or "run_distributed"
    script: str
    args_map: Dict[str, str] = Field(default_factory=dict)
    distributed: bool = False
    timeout_minutes: int = 240
    
    class Config:
        extra = "allow"


class StepManifest(BaseModel):
    """
    Typed representation of agent manifest JSON.
    
    Loaded from agent_manifests/*.json files.
    Supports multi-action steps where actions are executed sequentially.
    """
    agent_name: str
    description: str = ""
    pipeline_step: int
    version: str = "1.0.0"
    
    # Multi-action support
    actions: List[ActionConfig] = Field(default_factory=list)
    
    # Execution config (legacy single-action, extracted from actions[0])
    script: str = ""  # Extracted from actions[0].script if not present
    args_map: Dict[str, str] = Field(default_factory=dict)
    outputs: List[str] = Field(default_factory=list)
    
    # Success criteria
    success_condition: str = ""
    success_metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Defaults and bounds
    default_params: Dict[str, Any] = Field(default_factory=dict)
    parameter_bounds: Dict[str, Any] = Field(default_factory=dict)
    
    # Metrics and evaluation
    metrics_extraction: MetricsExtractionConfig = Field(default_factory=MetricsExtractionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # Execution control
    retry: int = 2
    timeout_minutes: int = 240
    distributed: bool = False
    
    # Pipeline flow
    follow_up_agents: List[str] = Field(default_factory=list)
    
    @property
    def is_multi_action(self) -> bool:
        """Check if step has multiple actions."""
        return len(self.actions) > 1
    
    @property
    def action_count(self) -> int:
        """Number of actions in this step."""
        return len(self.actions) if self.actions else 1
    
    def get_action(self, index: int) -> Optional[ActionConfig]:
        """Get action by index."""
        if self.actions and 0 <= index < len(self.actions):
            return self.actions[index]
        return None
    
    class Config:
        extra = "allow"  # Allow extra fields from manifest


# =============================================================================
# RESULT MODELS
# =============================================================================

@dataclass
class MetricsResult:
    """
    Result of metrics extraction with structured error handling.
    
    Fails loudly - errors prevent LLM evaluation.
    """
    success: bool
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def has_metric(self, name: str) -> bool:
        """Check if a metric was successfully extracted."""
        return name in self.metrics and self.metrics[name] is not None


@dataclass
class ActionResult:
    """
    Result of a single action within a step.
    
    WATCHER uses this for granular visibility into multi-action steps.
    """
    action_index: int
    action_type: str  # "run_script" or "run_distributed"
    script: str
    success: bool
    exit_code: int
    duration_seconds: int
    
    # Execution details
    command: List[str] = field(default_factory=list)
    stdout_tail: str = ""
    stderr_tail: str = ""
    error_message: Optional[str] = None
    
    # For distributed actions
    distributed: bool = False
    jobs_completed: int = 0
    jobs_failed: int = 0


@dataclass
class StepResult:
    """
    Result of step execution.
    
    Contains execution status, outputs validation, and extracted metrics.
    For multi-action steps, tracks per-action results for WATCHER visibility.
    """
    step: int
    step_name: str
    status: StepStatus
    exit_code: int
    duration_seconds: int
    
    # Output validation
    outputs_found: Dict[str, bool] = field(default_factory=dict)
    
    # Metrics (populated by metrics extractor)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metrics_errors: List[str] = field(default_factory=list)
    
    # Execution details
    command: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    stdout_tail: str = ""  # Last N chars of stdout
    stderr_tail: str = ""  # Last N chars of stderr
    error_message: Optional[str] = None
    
    # Retry tracking
    retry_count: int = 0
    
    # Multi-action tracking
    action_results: List[ActionResult] = field(default_factory=list)
    failed_action_index: Optional[int] = None
    total_actions: int = 1
    
    # Timestamps
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @property
    def success(self) -> bool:
        """Overall success based on status."""
        return self.status == StepStatus.SUCCESS
    
    @property
    def actions_completed(self) -> int:
        """Number of actions that completed successfully."""
        return sum(1 for a in self.action_results if a.success)
    
    @property
    def is_multi_action(self) -> bool:
        """Whether this step has multiple actions."""
        return self.total_actions > 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step": self.step,
            "step_name": self.step_name,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "outputs_found": self.outputs_found,
            "metrics": self.metrics,
            "metrics_errors": self.metrics_errors,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# DECISION MODELS
# =============================================================================

class StepDecision(BaseModel):
    """
    LLM evaluation decision.
    
    Structured response from LLM, constrained by GBNF grammar.
    """
    assessment: AssessmentLevel
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    recommendation: RecommendationAction
    parameter_adjustments: Dict[str, Any] = Field(default_factory=dict)
    
    def should_proceed(self) -> bool:
        """Check if recommendation is to proceed."""
        return self.recommendation == RecommendationAction.PROCEED
    
    def should_retry(self) -> bool:
        """Check if recommendation is to retry."""
        return self.recommendation in (
            RecommendationAction.RETRY,
            RecommendationAction.CHANGE_PRNG,
            RecommendationAction.ADJUST_THRESHOLD,
            RecommendationAction.EXPAND_WINDOW
        )
    
    def requires_human(self) -> bool:
        """Check if human intervention is required."""
        return self.recommendation == RecommendationAction.ESCALATE


@dataclass 
class PreRunCheck:
    """
    Result of pre-run validation.
    
    Returned by registry or policy checks before step execution.
    """
    can_proceed: bool
    reason: str
    suggested_params: Dict[str, Any] = field(default_factory=dict)
    blocked_by: Optional[str] = None  # "registry", "policy", "validation"


# =============================================================================
# STEP MAPPING
# =============================================================================

# Maps step numbers to manifest filenames
STEP_MANIFEST_MAP: Dict[int, str] = {
    1: "window_optimizer.json",
    2: "scorer_meta.json",       # Step 2.5
    3: "full_scoring.json",
    4: "ml_meta.json",
    5: "reinforcement.json",
    6: "prediction.json"
}

# Maps step numbers to display names
STEP_DISPLAY_NAMES: Dict[int, str] = {
    1: "Window Optimizer",
    2: "Scorer Meta-Optimizer",
    3: "Full Scoring",
    4: "Adaptive Meta-Optimizer", 
    5: "Anti-Overfit Training",
    6: "Prediction Generator"
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_manifest_filename(step: int) -> str:
    """Get manifest filename for step number."""
    if step not in STEP_MANIFEST_MAP:
        raise ValueError(f"Invalid step number: {step}. Valid steps: {list(STEP_MANIFEST_MAP.keys())}")
    return STEP_MANIFEST_MAP[step]


def get_step_display_name(step: int) -> str:
    """Get human-readable name for step."""
    return STEP_DISPLAY_NAMES.get(step, f"Step {step}")
