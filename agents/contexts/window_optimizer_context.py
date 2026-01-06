#!/usr/bin/env python3
"""
Window Optimizer Context - Step 1 specialized agent context.

REFACTORED: 2026-01-04 (Team Beta Approved)
- Thresholds loaded from distributed_config.json
- No semantic interpretation - raw + derived metrics only
- LLM does the reasoning, not this file

Version: 4.0.0
"""
from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path
from agents.contexts.base_agent_context import BaseAgentContext, EvaluationResult

# Import the new metrics extractor
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.metrics_extractor import (
    extract_step1_derived_metrics,
    get_step_thresholds,
    STEP_IDS
)


class WindowOptimizerContext(BaseAgentContext):
    """
    Specialized context for Step 1: Window Optimizer.
    
    REFACTORED RESPONSIBILITIES:
    - Extract raw metrics from results
    - Compute derived metrics (rates, ratios)
    - Load threshold priors from config
    - Package for LLM evaluation
    
    NOT RESPONSIBLE FOR:
    - Semantic interpretation ("good", "bad", "consider narrowing")
    - Decision making (proceed/retry/escalate)
    - That's the LLM's job now
    """
    
    agent_name: str = "window_optimizer_agent"
    pipeline_step: int = 1
    step_id: str = "step_1_window_optimizer"
    data_source_type: str = "synthetic"
    manifest_path: Optional[str] = None
    
    _threshold_priors: Optional[Dict[str, Any]] = None
    
    class Config:
        underscore_attrs_are_private = True
    
    def get_key_metrics(self) -> List[str]:
        """Key metrics for window optimization."""
        return [
            "seeds_tested",
            "bidirectional_count",
            "forward_count",
            "reverse_count",
            "best_window_size",
            "best_skip",
            "optimization_score",
            "execution_time_seconds"
        ]
    
    def get_threshold_priors(self) -> Dict[str, Any]:
        """
        Load thresholds from distributed_config.json.
        
        These are PRIORS for the LLM, not absolute rules.
        """
        if self._threshold_priors is None:
            self._threshold_priors = get_step_thresholds(
                self.step_id,
                self.data_source_type
            )
        return self._threshold_priors
    
    def get_raw_metrics(self) -> Dict[str, Any]:
        """
        Extract raw metrics from results.
        
        NO INTERPRETATION - just the facts.
        """
        return {
            "seeds_tested": self.results.get("seed_count", 
                           self.results.get("seeds_tested", 0)),
            "forward_count": self.results.get("forward_count", 0),
            "reverse_count": self.results.get("reverse_count", 0),
            "bidirectional_count": self.results.get("bidirectional_count", 0),
            "window_size": self.results.get("window_size",
                          self.results.get("best_window_size", 0)),
            "skip_min": self.results.get("skip_min", 0),
            "skip_max": self.results.get("skip_max", 0),
            "optimization_score": self.results.get("optimization_score", 0),
            "runtime_seconds": self.results.get("execution_time_seconds",
                              self.results.get("runtime_seconds", 0)),
        }
    
    def get_derived_metrics(self) -> Dict[str, float]:
        """
        Compute derived metrics (rates, ratios).
        
        NO INTERPRETATION - just math.
        """
        raw = self.get_raw_metrics()
        return extract_step1_derived_metrics(raw)
    
    def get_overall_success(self) -> Tuple[bool, float]:
        """
        Determine overall success using rate-based thresholds.
        
        This is the HEURISTIC FALLBACK when LLM is unavailable.
        Uses config-driven thresholds, not hardcoded values.
        """
        derived = self.get_derived_metrics()
        priors = self.get_threshold_priors()
        
        bi_rate = derived.get("bidirectional_rate", 1.0)
        overlap = derived.get("overlap_ratio", 0.0)
        
        # Get thresholds from config
        bi_thresholds = priors.get("bidirectional_rate", {})
        overlap_thresholds = priors.get("overlap_ratio", {})
        
        good_max = bi_thresholds.get("good_max", 0.02)
        warn_max = bi_thresholds.get("warn_max", 0.10)
        fail_max = bi_thresholds.get("fail_max", 0.30)
        overlap_good = overlap_thresholds.get("good_min", 0.25)
        
        # Zero survivors = critical failure
        if self.results.get("bidirectional_count", 0) == 0:
            return (False, 0.1)
        
        # Rate-based evaluation
        if bi_rate <= good_max and overlap >= overlap_good:
            return (True, 0.95)
        elif bi_rate <= warn_max:
            return (True, 0.75)
        elif bi_rate <= fail_max:
            return (False, 0.55)
        else:
            return (False, 0.30)
    
    def get_heuristic_decision(self) -> Dict[str, Any]:
        """
        Deterministic heuristic fallback when LLM unavailable.
        
        Returns a decision dict compatible with WatcherDecision schema.
        Conservative bias: when in doubt, retry > escalate > proceed.
        """
        derived = self.get_derived_metrics()
        priors = self.get_threshold_priors()
        success, confidence = self.get_overall_success()
        
        bi_rate = derived.get("bidirectional_rate", 1.0)
        bi_thresholds = priors.get("bidirectional_rate", {})
        
        good_max = bi_thresholds.get("good_max", 0.02)
        warn_max = bi_thresholds.get("warn_max", 0.10)
        
        # Zero survivors
        if self.results.get("bidirectional_count", 0) == 0:
            return {
                "decision": "escalate",
                "retry_reason": None,
                "confidence": 0.90,
                "reasoning": f"Zero bidirectional survivors for {self.data_source_type} data - PRNG hypothesis may be incorrect",
                "primary_signal": "bidirectional_count",
                "suggested_params": {"window_size": "increase", "skip_max": "increase"},
                "warnings": ["zero_survivors"],
                "checks": {
                    "used_rates": True,
                    "mentioned_data_source": True,
                    "avoided_absolute_only": True
                }
            }
        
        # Good range
        if bi_rate <= good_max:
            return {
                "decision": "proceed",
                "retry_reason": None,
                "confidence": confidence,
                "reasoning": f"bidirectional_rate {bi_rate:.4f} within good range for {self.data_source_type} data",
                "primary_signal": "bidirectional_rate",
                "suggested_params": None,
                "warnings": [],
                "checks": {
                    "used_rates": True,
                    "mentioned_data_source": True,
                    "avoided_absolute_only": True
                }
            }
        
        # Warning range - retry with tighten
        if bi_rate <= warn_max:
            return {
                "decision": "retry",
                "retry_reason": "tighten",
                "confidence": confidence,
                "reasoning": f"bidirectional_rate {bi_rate:.4f} in warning range for {self.data_source_type} - thresholds may be too loose",
                "primary_signal": "bidirectional_rate",
                "suggested_params": {
                    "forward_threshold": 0.005,
                    "reverse_threshold": 0.005
                },
                "warnings": ["rate_in_warning_range"],
                "checks": {
                    "used_rates": True,
                    "mentioned_data_source": True,
                    "avoided_absolute_only": True
                }
            }
        
        # Fail range - escalate
        return {
            "decision": "escalate",
            "retry_reason": None,
            "confidence": confidence,
            "reasoning": f"bidirectional_rate {bi_rate:.4f} exceeds fail threshold for {self.data_source_type} data",
            "primary_signal": "bidirectional_rate",
            "suggested_params": None,
            "warnings": ["rate_exceeds_fail_threshold"],
            "checks": {
                "used_rates": True,
                "mentioned_data_source": True,
                "avoided_absolute_only": True
            }
        }
    
    # ════════════════════════════════════════════════════════════════════════
    # DEPRECATED METHODS - Kept for backward compatibility
    # ════════════════════════════════════════════════════════════════════════
    
    def get_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """DEPRECATED: Use get_threshold_priors() instead."""
        import warnings
        warnings.warn(
            "get_thresholds() is deprecated. Use get_threshold_priors() which loads from config.",
            DeprecationWarning
        )
        return self.get_threshold_priors()
    
    def interpret_results(self) -> str:
        """
        DEPRECATED: LLM now does interpretation.
        
        Returns raw metrics summary for backward compatibility.
        """
        import warnings
        warnings.warn(
            "interpret_results() is deprecated. LLM now handles interpretation.",
            DeprecationWarning
        )
        raw = self.get_raw_metrics()
        derived = self.get_derived_metrics()
        return (
            f"seeds_tested={raw['seeds_tested']}, "
            f"bidirectional_count={raw['bidirectional_count']}, "
            f"bidirectional_rate={derived['bidirectional_rate']:.4f}, "
            f"overlap_ratio={derived['overlap_ratio']:.4f}"
        )
    
    def get_retry_suggestions(self) -> List[Dict[str, Any]]:
        """DEPRECATED: Use get_heuristic_decision()['suggested_params'] instead."""
        import warnings
        warnings.warn(
            "get_retry_suggestions() is deprecated. Use get_heuristic_decision() instead.",
            DeprecationWarning
        )
        decision = self.get_heuristic_decision()
        params = decision.get("suggested_params") or {}
        return [{"param": k, "suggestion": v} for k, v in params.items()]


# ════════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION (for backward compatibility)
# ════════════════════════════════════════════════════════════════════════════════

def create_window_optimizer_context(
    results: Dict[str, Any],
    run_number: int = 1,
    manifest_path: str = None,
    data_source_type: str = "synthetic"
) -> WindowOptimizerContext:
    """
    Factory function to create window optimizer context.
    
    Args:
        results: Results from window_optimizer.py
        run_number: Current run number (for retry tracking)
        manifest_path: Path to manifest (optional, for metadata)
        data_source_type: "synthetic", "real", or "hybrid"
    
    Returns:
        Configured WindowOptimizerContext instance
    """
    ctx = WindowOptimizerContext(
        results=results,
        run_number=run_number,
        data_source_type=data_source_type
    )
    
    if manifest_path:
        ctx.manifest_path = manifest_path
    
    return ctx
