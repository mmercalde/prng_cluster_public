#!/usr/bin/env python3
"""
Base Agent Context - Abstract base class for specialized agent contexts.

All six pipeline agent contexts inherit from this base, ensuring
consistent interface and shared functionality.

Version: 3.2.0
"""

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from agents.manifest import AgentManifest
from agents.parameters import ParameterContext
from agents.history import AnalysisHistory
from agents.runtime import RuntimeContext
from agents.safety import KillSwitch
from agents.pipeline import PipelineStepContext


class EvaluationResult(BaseModel):
    """Result of evaluating step output against criteria."""
    
    metric_name: str
    value: Any
    threshold_met: bool
    rating: str = ""  # excellent, good, acceptable, poor, fail
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "value": self.value,
            "passed": self.threshold_met,
            "rating": self.rating,
            "message": self.message
        }


class BaseAgentContext(BaseModel, ABC):
    """
    Abstract base class for all agent contexts.
    
    Each pipeline step has a specialized context that inherits from this,
    providing step-specific evaluation logic and domain expertise.
    """
    
    # Core identifiers
    agent_name: str
    pipeline_step: int = Field(ge=1, le=6)
    run_number: int = 1
    run_id: str = ""
    
    # Loaded manifest
    manifest: Optional[AgentManifest] = None
    
    # Parameter context (from Sub-Phase 1)
    parameter_context: Optional[ParameterContext] = None
    
    # Results to evaluate
    results: Dict[str, Any] = Field(default_factory=dict)
    
    # Evaluation outcomes
    evaluations: List[EvaluationResult] = Field(default_factory=list)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Generate run_id if not provided
        if not self.run_id:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"step{self.pipeline_step}_{timestamp}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ABSTRACT METHODS - Must be implemented by each specialized context
    # ═══════════════════════════════════════════════════════════════════════════
    
    @abstractmethod
    def get_key_metrics(self) -> List[str]:
        """
        Return list of key metric names for this step.
        
        These are the metrics the AI should focus on when evaluating.
        """
        pass
    
    @abstractmethod
    def get_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """
        Return evaluation thresholds for this step.
        
        Format:
        {
            "metric_name": {
                "excellent": {"min": x, "max": y},
                "good": {"min": x, "max": y},
                "acceptable": {"min": x, "max": y},
                "fail": {"min": x, "max": y}
            }
        }
        """
        pass
    
    @abstractmethod
    def interpret_results(self) -> str:
        """
        Provide step-specific interpretation of results.
        
        Returns a brief explanation of what the results mean
        in the context of this pipeline step.
        """
        pass
    
    @abstractmethod
    def get_retry_suggestions(self) -> List[Dict[str, Any]]:
        """
        Suggest parameter adjustments if retry is needed.
        
        Based on current results, what parameters should be adjusted?
        Returns list of {"param": name, "suggestion": value, "reason": why}
        """
        pass
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SHARED METHODS - Available to all specialized contexts
    # ═══════════════════════════════════════════════════════════════════════════
    
    def evaluate_metric(
        self, 
        metric_name: str, 
        value: Any
    ) -> EvaluationResult:
        """Evaluate a single metric against thresholds."""
        thresholds = self.get_thresholds().get(metric_name, {})
        
        if not thresholds:
            return EvaluationResult(
                metric_name=metric_name,
                value=value,
                threshold_met=True,
                rating="unknown",
                message="No thresholds defined"
            )
        
        # Check each rating level
        for rating in ["excellent", "good", "acceptable", "poor", "fail"]:
            bounds = thresholds.get(rating, {})
            if not bounds:
                continue
            
            min_val = bounds.get("min")
            max_val = bounds.get("max")
            
            in_range = True
            if min_val is not None and value < min_val:
                in_range = False
            if max_val is not None and value > max_val:
                in_range = False
            
            if in_range:
                return EvaluationResult(
                    metric_name=metric_name,
                    value=value,
                    threshold_met=(rating != "fail"),
                    rating=rating,
                    message=f"{metric_name}={value} is {rating}"
                )
        
        # No matching range found
        return EvaluationResult(
            metric_name=metric_name,
            value=value,
            threshold_met=False,
            rating="unknown",
            message=f"{metric_name}={value} outside all defined ranges"
        )
    
    def evaluate_all_metrics(self) -> List[EvaluationResult]:
        """Evaluate all key metrics from results."""
        self.evaluations = []
        
        for metric in self.get_key_metrics():
            if metric in self.results:
                value = self.results[metric]
                result = self.evaluate_metric(metric, value)
                self.evaluations.append(result)
        
        return self.evaluations
    
    def get_overall_success(self) -> Tuple[bool, float]:
        """
        Determine overall success and confidence.
        
        Returns (success, confidence) tuple.
        """
        if not self.evaluations:
            self.evaluate_all_metrics()
        
        if not self.evaluations:
            return True, 0.5  # No metrics to evaluate
        
        # Count passed/failed
        passed = sum(1 for e in self.evaluations if e.threshold_met)
        total = len(self.evaluations)
        
        # Calculate confidence based on ratings
        rating_scores = {
            "excellent": 1.0,
            "good": 0.85,
            "acceptable": 0.7,
            "poor": 0.4,
            "fail": 0.0,
            "unknown": 0.5
        }
        
        if total > 0:
            confidence = sum(rating_scores.get(e.rating, 0.5) for e in self.evaluations) / total
        else:
            confidence = 0.5
        
        success = passed == total and all(e.rating != "fail" for e in self.evaluations)
        
        return success, round(confidence, 2)
    
    def to_context_dict(self) -> Dict[str, Any]:
        """
        Generate agent context as clean dict for LLM.
        
        Hybrid JSON approach - combines all relevant context.
        """
        # Evaluate if not done
        if not self.evaluations:
            self.evaluate_all_metrics()
        
        success, confidence = self.get_overall_success()
        
        context = {
            "agent": self.agent_name,
            "step": self.pipeline_step,
            "run_number": self.run_number,
            "run_id": self.run_id,
            "key_metrics": self.get_key_metrics(),
            "results_summary": {
                metric: self.results.get(metric)
                for metric in self.get_key_metrics()
                if metric in self.results
            },
            "evaluations": [e.to_dict() for e in self.evaluations],
            "overall": {
                "success": success,
                "confidence": confidence
            },
            "interpretation": self.interpret_results(),
            "thresholds": self.get_thresholds()
        }
        
        # Add retry suggestions if not successful
        if not success:
            context["retry_suggestions"] = self.get_retry_suggestions()
        
        return context
    
    def load_manifest(self, manifest_path: str) -> "BaseAgentContext":
        """Load manifest and build parameter context."""
        self.manifest = AgentManifest.load(manifest_path)
        self.parameter_context = ParameterContext.build(
            self.manifest,
            self.results.get("config", {})
        )
        return self
    
    def set_results(self, results: Dict[str, Any]) -> "BaseAgentContext":
        """Set results to evaluate."""
        self.results = results
        self.evaluations = []  # Clear cached evaluations
        return self
