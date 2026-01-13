#!/usr/bin/env python3
"""
LLM Proposal Schema — Chapter 13 Phase 4
Models for structured LLM advisory proposals

VERSION: 1.0.0
DATE: 2026-01-12

These schemas define the contract between:
- LLM output (constrained by GBNF grammar)
- WATCHER validation (acceptance/rejection engine)
- Audit trail (decision logging)

The LLM is ADVISORY ONLY. These proposals are validated by WATCHER
before any action is taken.

NOTE: Uses dataclasses for compatibility. No external dependencies.
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json


# =============================================================================
# ENUMS
# =============================================================================

class FailureMode(str, Enum):
    """Classified failure modes the LLM can identify."""
    CALIBRATION_DRIFT = "calibration_drift"
    FEATURE_RELEVANCE = "feature_relevance"
    WINDOW_MISALIGNMENT = "window_misalignment"
    RANDOM_VARIANCE = "random_variance"
    REGIME_SHIFT = "regime_shift"
    MODEL_OVERFIT = "model_overfit"
    DATA_QUALITY = "data_quality"
    NONE_DETECTED = "none_detected"


class RecommendedAction(str, Enum):
    """Actions the LLM can recommend."""
    RETRAIN = "RETRAIN"           # Run learning loop (Steps 3→5→6)
    WAIT = "WAIT"                 # No action, continue observing
    ESCALATE = "ESCALATE"         # Human review required
    FULL_RESET = "FULL_RESET"     # Full pipeline (Steps 1→6)


class RetrainScope(str, Enum):
    """Scope of retrain operation."""
    STEPS_3_5_6 = "steps_3_5_6"   # Learning loop
    STEPS_5_6 = "steps_5_6"       # Model + predict only
    STEP_6_ONLY = "step_6_only"   # Prediction refresh only
    FULL_PIPELINE = "full_pipeline"  # Steps 1→6


class RiskLevel(str, Enum):
    """Risk assessment for proposed changes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# PARAMETER PROPOSAL
# =============================================================================

@dataclass
class ParameterProposal:
    """
    A single parameter adjustment proposal from the LLM.
    
    Example:
        {
            "parameter": "confidence_threshold",
            "current_value": 0.7,
            "proposed_value": 0.65,
            "delta": "-0.05",
            "confidence": 0.82,
            "rationale": "Underconfidence pattern over last 8 draws"
        }
    """
    parameter: str
    proposed_value: float
    delta: str
    confidence: float
    rationale: str
    current_value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterProposal':
        return cls(**data)


# =============================================================================
# FULL LLM PROPOSAL
# =============================================================================

@dataclass
class LLMProposal:
    """
    Complete LLM advisory proposal for Chapter 13.
    
    This is the primary output schema that the LLM produces.
    It is constrained by chapter_13.gbnf grammar to ensure
    valid structure even with open-ended LLM generation.
    
    IMPORTANT: This proposal is ADVISORY ONLY.
    WATCHER validates against policies before any execution.
    """
    
    # Analysis
    analysis_summary: str
    failure_mode: FailureMode
    confidence: float
    
    # Recommendation
    recommended_action: RecommendedAction
    risk_level: RiskLevel
    requires_human_review: bool
    
    # Optional fields
    retrain_scope: Optional[RetrainScope] = None
    parameter_proposals: List[ParameterProposal] = field(default_factory=list)
    alternative_hypothesis: Optional[str] = None
    
    # Metadata (populated by system, not LLM)
    generated_at: Optional[str] = None
    diagnostics_fingerprint: Optional[str] = None
    model_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "analysis_summary": self.analysis_summary,
            "failure_mode": self.failure_mode.value if isinstance(self.failure_mode, Enum) else self.failure_mode,
            "confidence": self.confidence,
            "recommended_action": self.recommended_action.value if isinstance(self.recommended_action, Enum) else self.recommended_action,
            "retrain_scope": self.retrain_scope.value if self.retrain_scope and isinstance(self.retrain_scope, Enum) else self.retrain_scope,
            "parameter_proposals": [p.to_dict() if hasattr(p, 'to_dict') else p for p in self.parameter_proposals],
            "risk_level": self.risk_level.value if isinstance(self.risk_level, Enum) else self.risk_level,
            "requires_human_review": self.requires_human_review,
            "alternative_hypothesis": self.alternative_hypothesis,
            "generated_at": self.generated_at,
            "diagnostics_fingerprint": self.diagnostics_fingerprint,
            "model_id": self.model_id
        }
    
    def model_dump(self) -> Dict[str, Any]:
        """Pydantic-compatible alias for to_dict()."""
        return self.to_dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMProposal':
        """Create from dict."""
        # Convert string enums back to enum types
        failure_mode = data.get("failure_mode", "none_detected")
        if isinstance(failure_mode, str):
            failure_mode = FailureMode(failure_mode)
        
        action = data.get("recommended_action", "WAIT")
        if isinstance(action, str):
            action = RecommendedAction(action)
        
        risk = data.get("risk_level", "low")
        if isinstance(risk, str):
            risk = RiskLevel(risk)
        
        scope = data.get("retrain_scope")
        if scope and isinstance(scope, str):
            scope = RetrainScope(scope)
        
        # Convert parameter proposals
        proposals = []
        for p in data.get("parameter_proposals", []):
            if isinstance(p, dict):
                proposals.append(ParameterProposal.from_dict(p))
            else:
                proposals.append(p)
        
        return cls(
            analysis_summary=data.get("analysis_summary", ""),
            failure_mode=failure_mode,
            confidence=float(data.get("confidence", 0.0)),
            recommended_action=action,
            retrain_scope=scope,
            parameter_proposals=proposals,
            risk_level=risk,
            requires_human_review=bool(data.get("requires_human_review", False)),
            alternative_hypothesis=data.get("alternative_hypothesis"),
            generated_at=data.get("generated_at"),
            diagnostics_fingerprint=data.get("diagnostics_fingerprint"),
            model_id=data.get("model_id")
        )
    
    def validate_against_policies(self, policies: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate proposal against WATCHER policies.
        
        Returns:
            Dict with 'valid' bool and 'violations' list
        """
        violations = []
        acceptance = policies.get("acceptance_rules", {})
        
        # Check max parameter delta
        max_delta = acceptance.get("max_parameter_delta", 0.3)
        for prop in self.parameter_proposals:
            # Parse delta to check magnitude
            try:
                delta_str = prop.delta.replace('+', '').replace('*', '')
                delta_val = abs(float(delta_str))
                if prop.current_value and prop.current_value != 0:
                    relative_delta = delta_val / abs(prop.current_value)
                    if relative_delta > max_delta:
                        violations.append(
                            f"Parameter '{prop.parameter}' delta {relative_delta:.2%} "
                            f"exceeds max {max_delta:.0%}"
                        )
            except ValueError:
                pass  # Non-numeric delta, skip check
        
        # Check max parameters per proposal
        max_params = acceptance.get("max_parameters_per_proposal", 3)
        if len(self.parameter_proposals) > max_params:
            violations.append(
                f"Too many parameters ({len(self.parameter_proposals)}) > max ({max_params})"
            )
        
        return {
            "valid": len(violations) == 0,
            "violations": violations
        }


# =============================================================================
# DIAGNOSTICS SUMMARY (Input to LLM)
# =============================================================================

@dataclass
class DiagnosticsSummary:
    """
    Summarized diagnostics for LLM consumption.
    
    This is a simplified view of post_draw_diagnostics.json
    formatted for efficient LLM context usage.
    """
    
    # Identity
    draw_id: str
    draw_timestamp: str
    data_fingerprint: str
    
    # Key metrics
    exact_hits: int
    pool_size: int
    hit_rate: float
    
    # Confidence
    mean_confidence: float
    confidence_correlation: float
    overconfident: bool
    underconfident: bool
    
    # Health
    consecutive_misses: int
    model_stability: str
    window_decay: float
    survivor_churn: float
    
    # Flags
    summary_flags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_diagnostics(cls, diagnostics: Dict[str, Any]) -> 'DiagnosticsSummary':
        """Create summary from full diagnostics dict."""
        pv = diagnostics.get("prediction_validation", {})
        cc = diagnostics.get("confidence_calibration", {})
        ph = diagnostics.get("pipeline_health", {})
        
        pool_size = pv.get("pool_size", 1)
        exact_hits = pv.get("exact_hits", 0)
        
        return cls(
            draw_id=diagnostics.get("draw_id", "unknown"),
            draw_timestamp=diagnostics.get("draw_timestamp", ""),
            data_fingerprint=diagnostics.get("data_fingerprint", ""),
            exact_hits=exact_hits,
            pool_size=pool_size,
            hit_rate=exact_hits / max(pool_size, 1),
            mean_confidence=cc.get("mean_confidence", 0.0),
            confidence_correlation=cc.get("predicted_vs_actual_correlation", 0.0),
            overconfident=cc.get("overconfidence_detected", False),
            underconfident=cc.get("underconfidence_detected", False),
            consecutive_misses=ph.get("consecutive_misses", 0),
            model_stability=ph.get("model_stability", "unknown"),
            window_decay=ph.get("window_decay", 0.0),
            survivor_churn=ph.get("survivor_churn", 0.0),
            summary_flags=diagnostics.get("summary_flags", [])
        )


# =============================================================================
# RUN HISTORY ENTRY (For LLM context)
# =============================================================================

@dataclass
class RunHistoryEntry:
    """
    Single run history entry for LLM context.
    
    Used to provide the LLM with recent run summaries.
    """
    run_id: str
    timestamp: str
    hit_rate: float
    confidence: float
    action_taken: str
    outcome: str


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_llm_response_to_proposal(
    response_text: str,
    diagnostics_fingerprint: str = None,
    model_id: str = None
) -> LLMProposal:
    """
    Parse LLM response text into LLMProposal.
    
    Expects JSON response (enforced by GBNF grammar).
    
    Args:
        response_text: Raw LLM response (should be valid JSON)
        diagnostics_fingerprint: Optional fingerprint to attach
        model_id: Optional model identifier
    
    Returns:
        LLMProposal instance
    
    Raises:
        ValueError: If response cannot be parsed
    """
    # Clean response (remove markdown code blocks if present)
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM response: {e}")
    
    # Add metadata
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    if diagnostics_fingerprint:
        data["diagnostics_fingerprint"] = diagnostics_fingerprint
    if model_id:
        data["model_id"] = model_id
    
    return LLMProposal.from_dict(data)


def create_empty_proposal(reason: str = "No analysis performed") -> LLMProposal:
    """Create a neutral/empty proposal when LLM is unavailable."""
    return LLMProposal(
        analysis_summary=reason,
        failure_mode=FailureMode.NONE_DETECTED,
        confidence=0.0,
        recommended_action=RecommendedAction.WAIT,
        retrain_scope=None,
        parameter_proposals=[],
        risk_level=RiskLevel.LOW,
        requires_human_review=False,
        alternative_hypothesis=None,
        generated_at=datetime.now(timezone.utc).isoformat()
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'FailureMode',
    'RecommendedAction',
    'RetrainScope',
    'RiskLevel',
    # Models
    'ParameterProposal',
    'LLMProposal',
    'DiagnosticsSummary',
    'RunHistoryEntry',
    # Functions
    'parse_llm_response_to_proposal',
    'create_empty_proposal'
]
