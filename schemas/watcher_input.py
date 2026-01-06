"""
Watcher Input Schema v1.0
Universal input schema for LLM evaluation across all pipeline steps.

Team Beta Approved: 2026-01-04
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Literal


class DataSource(BaseModel):
    """Data provenance - REQUIRED for rate calibration."""
    type: Literal["synthetic", "real", "hybrid"]
    generator: Optional[str] = None      # If synthetic: which generator
    notes: Optional[str] = None          # If real: "Daily 3 CA" etc.
    version: Optional[str] = None        # Generator version if applicable


class PRNGHypothesis(BaseModel):
    """Current PRNG hypothesis being tested."""
    prng_type: str = "java_lcg"
    mod: int = 1000
    skip_mode: Literal["constant", "variable"] = "constant"
    skip_min: Optional[int] = None
    skip_max: Optional[int] = None


class WindowConfig(BaseModel):
    """Window configuration for train/holdout split."""
    train_draws: int
    holdout_draws: int
    train_start: int = 1


class RunContext(BaseModel):
    """Context for the current pipeline run."""
    pipeline_step: int = Field(..., ge=1, le=6)
    step_id: str
    prng_hypothesis: PRNGHypothesis
    data_source: DataSource
    window: Optional[WindowConfig] = None
    run_id: Optional[str] = None
    parent_run_id: Optional[str] = None


class WatcherInput(BaseModel):
    """
    Universal input schema for all pipeline steps.
    
    Contains:
    - context: What step, what data source, what hypothesis
    - raw_metrics: Unprocessed counts from the step
    - derived_metrics: Computed rates/ratios (non-semantic)
    - threshold_priors: From distributed_config.json (optional)
    """
    context: RunContext
    raw_metrics: Dict[str, Any]
    derived_metrics: Dict[str, float]
    threshold_priors: Optional[Dict[str, Any]] = None
    
    class Config:
        extra = "allow"  # Allow additional fields for step-specific data
