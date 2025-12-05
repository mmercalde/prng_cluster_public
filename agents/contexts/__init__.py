"""
Specialized Agent Contexts Module - Domain expertise for each pipeline step.

Exports:
    BaseAgentContext - Abstract base class
    WindowOptimizerContext - Step 1
    ScorerMetaContext - Step 2
    FullScoringContext - Step 3
    MLMetaContext - Step 4
    AntiOverfitContext - Step 5
    PredictionContext - Step 6
    
    EvaluationResult - Result of metric evaluation
    get_context_for_step - Factory function
"""

from .base_agent_context import BaseAgentContext, EvaluationResult
from .window_optimizer_context import WindowOptimizerContext, create_window_optimizer_context
from .scorer_meta_context import ScorerMetaContext, create_scorer_meta_context
from .full_scoring_context import FullScoringContext, create_full_scoring_context
from .ml_meta_context import MLMetaContext, create_ml_meta_context
from .anti_overfit_context import AntiOverfitContext, create_anti_overfit_context
from .prediction_context import PredictionContext, create_prediction_context

from typing import Dict, Any, Optional


# Context class mapping by step
CONTEXT_CLASSES = {
    1: WindowOptimizerContext,
    2: ScorerMetaContext,
    3: FullScoringContext,
    4: MLMetaContext,
    5: AntiOverfitContext,
    6: PredictionContext
}

# Factory functions mapping by step
CONTEXT_FACTORIES = {
    1: create_window_optimizer_context,
    2: create_scorer_meta_context,
    3: create_full_scoring_context,
    4: create_ml_meta_context,
    5: create_anti_overfit_context,
    6: create_prediction_context
}


def get_context_for_step(
    step: int,
    results: Dict[str, Any],
    run_number: int = 1,
    manifest_path: Optional[str] = None
) -> BaseAgentContext:
    """
    Factory function to create the appropriate context for a pipeline step.
    
    Args:
        step: Pipeline step number (1-6)
        results: Results dict to evaluate
        run_number: Current run number
        manifest_path: Optional path to agent manifest
        
    Returns:
        Specialized context for the step
        
    Raises:
        ValueError: If step number is invalid
    """
    if step not in CONTEXT_FACTORIES:
        raise ValueError(f"Invalid step number: {step}. Must be 1-6.")
    
    factory = CONTEXT_FACTORIES[step]
    return factory(results, run_number, manifest_path)


def get_context_class(step: int) -> type:
    """Get the context class for a pipeline step."""
    if step not in CONTEXT_CLASSES:
        raise ValueError(f"Invalid step number: {step}. Must be 1-6.")
    return CONTEXT_CLASSES[step]
