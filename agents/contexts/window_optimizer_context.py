#!/usr/bin/env python3
"""
Window Optimizer Context - Step 1 specialized agent context.

Provides domain expertise for evaluating window optimization results,
including bidirectional survivor analysis and PRNG parameter tuning.

Version: 3.2.0
"""

from typing import Dict, Any, List, Tuple
import math
import json
from pathlib import Path
from agents.contexts.base_agent_context import BaseAgentContext, EvaluationResult


class WindowOptimizerContext(BaseAgentContext):
    """
    Specialized context for Step 1: Window Optimizer.
    
    Key focus areas:
    - Bidirectional survivor count (forward AND reverse matches)
    - Window size optimization
    - PRNG parameter tuning (skip values)
    - Forward/reverse balance
    """
    
    agent_name: str = "window_optimizer_agent"
    pipeline_step: int = 1
    
    def get_key_metrics(self) -> List[str]:
        """Key metrics for window optimization."""
        return [
            "bidirectional_count",
            "forward_count",
            "reverse_count",
            "best_window_size",
            "best_skip",
            "optimization_score",
            "execution_time_seconds"
        ]
    
    def get_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Evaluation thresholds for window optimizer."""
        return {
            "bidirectional_count": {
                "excellent": {"min": 1, "max": 10},
                "good": {"min": 11, "max": 100},
                "acceptable": {"min": 101, "max": 1000},
                "poor": {"min": 1001, "max": 10000},
                "fail": {"min": 10001, "max": None}
            },
            "forward_count": {
                "excellent": {"min": 1, "max": 100},
                "good": {"min": 101, "max": 1000},
                "acceptable": {"min": 1001, "max": 10000},
                "poor": {"min": 10001, "max": 100000},
                "fail": {"min": 100001, "max": None}
            },
            "reverse_count": {
                "excellent": {"min": 1, "max": 100},
                "good": {"min": 101, "max": 1000},
                "acceptable": {"min": 1001, "max": 10000},
                "poor": {"min": 10001, "max": 100000},
                "fail": {"min": 100001, "max": None}
            },
            "optimization_score": {
                "excellent": {"min": 0.9, "max": 1.0},
                "good": {"min": 0.7, "max": 0.9},
                "acceptable": {"min": 0.5, "max": 0.7},
                "poor": {"min": 0.3, "max": 0.5},
                "fail": {"min": 0.0, "max": 0.3}
            }
        }
    

    def get_overall_success(self) -> Tuple[bool, float]:
        """
        Determine overall success and confidence for Step 1.
        
        Reads evaluation_params from manifest for configurable confidence formula.
        Returns (success, confidence) tuple.
        """
        # Extract survivor counts - check both top-level and nested locations
        bi = self.results.get("bidirectional_count", 0)
        if bi == 0:
            # Try nested location (best_result)
            best_result = self.results.get("best_result", {})
            bi = best_result.get("bidirectional_count", 0)
        
        bi = int(bi or 0)
        
        # Load evaluation params from manifest (with defaults)
        eval_params = {
            "confidence_formula": "bit_length",
            "confidence_base": 0.7,
            "confidence_divisor": 64.0,
            "confidence_cap": 0.95,
            "zero_survivors_confidence": 0.5
        }
        
        # Try to load from manifest
        manifest_path = Path("agent_manifests/window_optimizer.json")
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                eval_params.update(manifest.get("evaluation_params", {}))
            except Exception:
                pass  # Use defaults
        
        # Zero survivors = failure
        if bi == 0:
            return False, eval_params["zero_survivors_confidence"]
        
        # Calculate confidence based on formula
        formula = eval_params["confidence_formula"]
        base = eval_params["confidence_base"]
        cap = eval_params["confidence_cap"]
        divisor = eval_params["confidence_divisor"]
        
        if formula == "bit_length":
            confidence = min(cap, base + (bi.bit_length() / divisor))
        elif formula == "log10":
            confidence = min(cap, base + (math.log10(bi + 1) / 10.0))
        else:
            # Default to bit_length
            confidence = min(cap, base + (bi.bit_length() / divisor))
        
        return True, confidence

    def interpret_results(self) -> str:
        """Interpret window optimization results."""
        bi_count = self.results.get("bidirectional_count", 0)
        fwd_count = self.results.get("forward_count", 0)
        rev_count = self.results.get("reverse_count", 0)
        window = self.results.get("best_window_size", "unknown")
        skip = self.results.get("best_skip", "unknown")
        
        # Zero survivors is critical failure
        if bi_count == 0:
            return "CRITICAL: Zero bidirectional survivors. PRNG hypothesis may be incorrect or parameters need major adjustment."
        
        # Evaluate balance
        if fwd_count > 0 and rev_count > 0:
            ratio = min(fwd_count, rev_count) / max(fwd_count, rev_count)
        else:
            ratio = 0
        
        interpretation_parts = []
        
        # Bidirectional analysis
        if bi_count <= 10:
            interpretation_parts.append(f"Excellent signal: Only {bi_count} bidirectional survivors - highly constrained search space.")
        elif bi_count <= 100:
            interpretation_parts.append(f"Good signal: {bi_count} bidirectional survivors - manageable for ML scoring.")
        elif bi_count <= 1000:
            interpretation_parts.append(f"Acceptable: {bi_count} bidirectional survivors - scoring will take longer but feasible.")
        else:
            interpretation_parts.append(f"High survivor count ({bi_count}) - consider adjusting parameters to narrow search space.")
        
        # Balance analysis
        if ratio > 0.8:
            interpretation_parts.append(f"Forward/reverse balance is good (ratio={ratio:.2f}).")
        elif ratio > 0.5:
            interpretation_parts.append(f"Moderate forward/reverse imbalance (ratio={ratio:.2f}).")
        else:
            interpretation_parts.append(f"Significant forward/reverse imbalance (ratio={ratio:.2f}) - may indicate parameter issues.")
        
        # Optimal parameters
        interpretation_parts.append(f"Optimal window={window}, skip={skip}.")
        
        return " ".join(interpretation_parts)
    
    def get_retry_suggestions(self) -> List[Dict[str, Any]]:
        """Suggest parameter adjustments for retry."""
        suggestions = []
        bi_count = self.results.get("bidirectional_count", 0)
        fwd_count = self.results.get("forward_count", 0)
        current_window = self.results.get("best_window_size", 512)
        current_trials = self.results.get("config", {}).get("trials", 50)
        
        # Zero survivors - major adjustments needed
        if bi_count == 0:
            suggestions.append({
                "param": "window_size",
                "suggestion": min(current_window * 2, 2000),
                "reason": "Zero survivors - try larger window to capture more draws"
            })
            suggestions.append({
                "param": "skip_max",
                "suggestion": 50,
                "reason": "Zero survivors - expand skip search range"
            })
            return suggestions
        
        # Too many survivors - narrow search
        if bi_count > 1000:
            suggestions.append({
                "param": "window_size",
                "suggestion": max(current_window // 2, 50),
                "reason": f"Too many survivors ({bi_count}) - try smaller window"
            })
            suggestions.append({
                "param": "trials",
                "suggestion": min(current_trials * 2, 200),
                "reason": "Increase optimization trials to find better parameters"
            })
        
        # Forward/reverse imbalance
        if fwd_count > 0 and bi_count > 0:
            if fwd_count > bi_count * 100:
                suggestions.append({
                    "param": "skip_max",
                    "suggestion": 30,
                    "reason": "High forward count relative to bidirectional - adjust skip range"
                })
        
        # Low optimization score
        opt_score = self.results.get("optimization_score", 1.0)
        if opt_score < 0.5:
            suggestions.append({
                "param": "trials",
                "suggestion": min(current_trials * 2, 300),
                "reason": f"Low optimization score ({opt_score:.2f}) - more trials may help"
            })
        
        return suggestions


def create_window_optimizer_context(
    results: Dict[str, Any],
    run_number: int = 1,
    manifest_path: str = None
) -> WindowOptimizerContext:
    """Factory function to create window optimizer context."""
    ctx = WindowOptimizerContext(
        run_number=run_number,
        results=results
    )
    
    if manifest_path:
        ctx.load_manifest(manifest_path)
    
    return ctx
