#!/usr/bin/env python3
"""
Pydantic models for training diagnostics LLM analysis output.

Version: 1.0.0
Date: 2026-02-11
Chapter: 14 Phase 7 (LLM Integration)
Spec: CHAPTER_14_TRAINING_DIAGNOSTICS.md Section 8.3

Mirrors diagnostics_analysis.gbnf exactly.
Used to validate and parse LLM responses after grammar-constrained decoding.

Usage:
    from diagnostics_analysis_schema import DiagnosticsAnalysis

    # Parse LLM response
    analysis = DiagnosticsAnalysis.model_validate_json(raw_json)
    print(analysis.focus_area)         # e.g. FocusArea.FEATURE_RELEVANCE
    print(analysis.root_cause)         # Root cause explanation
    print(analysis.model_recommendations)  # Per-model verdicts
"""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DiagnosticsFocusArea(str, Enum):
    """Focus area classification for training diagnostics.

    Maps to Strategy Advisor focus areas but limited to
    diagnostics-relevant subset.
    """
    MODEL_DIVERSITY = "MODEL_DIVERSITY"
    FEATURE_RELEVANCE = "FEATURE_RELEVANCE"
    POOL_PRECISION = "POOL_PRECISION"
    CONFIDENCE_CALIBRATION = "CONFIDENCE_CALIBRATION"
    REGIME_SHIFT = "REGIME_SHIFT"


class DiagnosticsModelType(str, Enum):
    """Model types supported by training diagnostics."""
    NEURAL_NET = "neural_net"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"


class DiagnosticsModelVerdict(str, Enum):
    """Per-model verdict from LLM analysis.

    viable:        Model is working well, continue using.
    fixable:       Model has issues but diagnostics suggest specific fixes.
    skip:          Model is not suitable for this data/regime.
    not_evaluated: No diagnostics data was available for this model type.
    """
    VIABLE = "viable"
    FIXABLE = "fixable"
    SKIP = "skip"
    NOT_EVALUATED = "not_evaluated"


class DiagnosticsModelRecommendation(BaseModel):
    """Per-model-type recommendation from LLM analysis."""
    model_config = ConfigDict(extra="forbid")

    model_type: DiagnosticsModelType
    verdict: DiagnosticsModelVerdict
    rationale: str = Field(..., max_length=500)


class DiagnosticsParameterProposal(BaseModel):
    """Parameter change proposal from LLM analysis.

    WATCHER validates these against watcher_policies.json bounds
    before applying. LLM proposals are advisory only.
    """
    model_config = ConfigDict(extra="forbid")

    parameter: str = Field(..., max_length=100)
    current_value: Optional[float] = None
    proposed_value: float
    rationale: str = Field(..., max_length=300)


class DiagnosticsAnalysis(BaseModel):
    """LLM-produced analysis of training diagnostics.

    This is the OUTPUT of LLM inference, not the input.
    The LLM receives a DiagnosticsBundle prompt and produces
    this structured response via GBNF-constrained decoding.

    Invariant: Grammar-constrained output means all fields are
    guaranteed present and type-correct. Pydantic adds semantic
    validation (unique models, confidence range, max lengths).

    extra="forbid" prevents silent acceptance of unknown fields
    if grammar evolves but schema does not (drift protection).
    """
    model_config = ConfigDict(extra="forbid")

    focus_area: DiagnosticsFocusArea
    root_cause: str = Field(..., max_length=500)
    root_cause_confidence: float = Field(..., ge=0.0, le=1.0)
    model_recommendations: List[DiagnosticsModelRecommendation] = Field(
        ..., min_length=1, max_length=4
    )
    parameter_proposals: List[DiagnosticsParameterProposal] = Field(
        default_factory=list, max_length=5
    )
    selfplay_guidance: str = Field(..., max_length=500)
    requires_human_review: bool = False

    @field_validator('model_recommendations')
    @classmethod
    def validate_unique_models(cls, v):
        """Ensure no duplicate model types in recommendations."""
        model_types = [r.model_type for r in v]
        if len(model_types) != len(set(model_types)):
            raise ValueError("Duplicate model types in recommendations")
        return v


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def parse_diagnostics_response(raw_json: str) -> Optional[DiagnosticsAnalysis]:
    """Parse raw LLM JSON response into validated DiagnosticsAnalysis.

    Handles common LLM quirks:
    - Strips markdown code fences
    - Strips leading/trailing whitespace

    Args:
        raw_json: Raw string from LLM response.

    Returns:
        DiagnosticsAnalysis if valid, None if parsing fails.
    """
    cleaned = raw_json.strip()

    # Strip markdown fences if present (grammar should prevent this, but be safe)
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        return DiagnosticsAnalysis.model_validate_json(cleaned)
    except Exception:
        return None


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    # Validate against the example from Chapter 14 Section 8.6
    example = {
        "focus_area": "FEATURE_RELEVANCE",
        "root_cause": (
            "Neural net feature gradient spread of 12847x indicates severe "
            "input scaling imbalance. forward_count dominates gradient computation "
            "while normalized features are effectively invisible."
        ),
        "root_cause_confidence": 0.85,
        "model_recommendations": [
            {
                "model_type": "neural_net",
                "verdict": "fixable",
                "rationale": "Add BatchNorm or StandardScaler. Replace ReLU with LeakyReLU."
            },
            {
                "model_type": "catboost",
                "verdict": "viable",
                "rationale": "CatBoost MSE 1.77e-9 is excellent. No issues detected."
            },
            {
                "model_type": "xgboost",
                "verdict": "viable",
                "rationale": "XGBoost MSE 9.32e-9 is strong."
            },
            {
                "model_type": "lightgbm",
                "verdict": "viable",
                "rationale": "LightGBM MSE 1.06e-8 is competitive."
            }
        ],
        "parameter_proposals": [
            {
                "parameter": "normalize_features",
                "current_value": None,
                "proposed_value": 1,
                "rationale": "Add StandardScaler to preprocessing pipeline"
            },
            {
                "parameter": "nn_activation",
                "current_value": None,
                "proposed_value": 0,
                "rationale": "Switch from ReLU to LeakyReLU(0.01) to prevent dead neurons"
            }
        ],
        "selfplay_guidance": (
            "Next selfplay episode should focus FEATURE_RELEVANCE: "
            "test normalized vs unnormalized features with catboost as baseline."
        ),
        "requires_human_review": False
    }

    import json
    raw = json.dumps(example)
    result = parse_diagnostics_response(raw)

    if result:
        print("✅ Schema validation passed")
        print(f"   focus_area: {result.focus_area.value}")
        print(f"   root_cause_confidence: {result.root_cause_confidence}")
        print(f"   model_recommendations: {len(result.model_recommendations)}")
        print(f"   parameter_proposals: {len(result.parameter_proposals)}")
        print(f"   requires_human_review: {result.requires_human_review}")
    else:
        print("❌ Schema validation FAILED")
        exit(1)

    # Test duplicate model detection
    bad_example = example.copy()
    bad_example["model_recommendations"] = [
        {"model_type": "catboost", "verdict": "viable", "rationale": "test"},
        {"model_type": "catboost", "verdict": "skip", "rationale": "duplicate"},
    ]
    raw_bad = json.dumps(bad_example)
    bad_result = parse_diagnostics_response(raw_bad)
    if bad_result is None:
        print("✅ Duplicate model detection works")
    else:
        print("❌ Duplicate model detection FAILED")
        exit(1)

    print("\n✅ All schema tests passed")
