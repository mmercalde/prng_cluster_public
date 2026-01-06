"""
Watcher Decision Schema v1.0
Universal output schema for LLM evaluation across all pipeline steps.

Team Beta Approved: 2026-01-04
Decision vocabulary: {proceed, retry, escalate} - 3 VERBS ONLY
Nuance expressed via retry_reason, suggested_params, warnings
"""
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Any, Optional, List, Literal
from enum import Enum


class RetryReason(str, Enum):
    """
    Why retry was recommended.
    Expresses tighten/widen/investigate without expanding verb vocabulary.
    """
    tighten = "tighten"          # Thresholds too loose, narrow search
    widen = "widen"              # Thresholds too tight, expand search  
    rerun = "rerun"              # Transient failure, same params
    investigate = "investigate"  # Needs analysis before proceeding


class ReasoningChecks(BaseModel):
    """
    Self-report: did LLM follow the reasoning rules?
    Used by CI tests to catch template echoing.
    """
    used_rates: bool = Field(
        ..., 
        description="Used rates/ratios, not just absolute counts"
    )
    mentioned_data_source: bool = Field(
        ..., 
        description="Referenced data_source.type in reasoning"
    )
    avoided_absolute_only: bool = Field(
        ..., 
        description="Did not base decision on counts alone"
    )


class WatcherDecision(BaseModel):
    """
    Universal output schema for all pipeline steps.
    
    Team Beta mandate:
    - 3 verbs only: proceed, retry, escalate
    - Nuance via retry_reason + suggested_params + warnings
    - Must include reasoning checks for CI validation
    """
    decision: Literal["proceed", "retry", "escalate"]
    retry_reason: Optional[RetryReason] = Field(
        None,
        description="Required if decision=retry"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = Field(
        ..., 
        min_length=20, 
        max_length=500,
        description="Explanation of decision - must reference rates and data_source"
    )
    primary_signal: str = Field(
        ..., 
        description="The metric that drove the decision (e.g., 'bidirectional_rate')"
    )
    suggested_params: Optional[Dict[str, Any]] = Field(
        None,
        description="Parameter adjustments if retry/escalate"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-blocking concerns to surface"
    )
    checks: ReasoningChecks
    
    @model_validator(mode='after')
    def validate_retry_has_reason(self):
        """Ensure retry decisions include a reason."""
        if self.decision == "retry" and self.retry_reason is None:
            raise ValueError("retry_reason is required when decision='retry'")
        return self
    
    class Config:
        use_enum_values = True
