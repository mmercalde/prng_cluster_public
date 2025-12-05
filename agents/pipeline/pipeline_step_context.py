#!/usr/bin/env python3
"""
Pipeline Step Context - Expected inputs, outputs, and behavior per step.

Provides pipeline-level context for AI agents to understand
what each step expects and produces.

Version: 3.2.0
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum


class StepStatus(str, Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStep(BaseModel):
    """Definition of a single pipeline step."""
    
    step_number: int = Field(ge=1, le=6)
    name: str
    description: str = ""
    
    # Expected I/O
    required_inputs: List[str] = Field(default_factory=list)
    optional_inputs: List[str] = Field(default_factory=list)
    expected_outputs: List[str] = Field(default_factory=list)
    
    # Timing expectations
    typical_duration_minutes: int = 30
    max_duration_minutes: int = 120
    
    # Resource requirements
    requires_gpu: bool = True
    min_gpu_memory_mb: int = 4000
    distributed_capable: bool = False
    
    # Dependencies
    depends_on: List[int] = Field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return {
            "step": self.step_number,
            "name": self.name,
            "description": self.description,
            "inputs": self.required_inputs,
            "outputs": self.expected_outputs,
            "typical_minutes": self.typical_duration_minutes,
            "requires_gpu": self.requires_gpu,
            "distributed": self.distributed_capable
        }


# Pre-defined pipeline steps
PIPELINE_STEPS = {
    1: PipelineStep(
        step_number=1,
        name="Window Optimizer",
        description="Find optimal analysis window and PRNG parameters using Bayesian optimization",
        required_inputs=["lottery_data.json"],
        expected_outputs=["optimal_window_config.json", "bidirectional_survivors.json"],
        typical_duration_minutes=15,
        max_duration_minutes=60,
        requires_gpu=True,
        min_gpu_memory_mb=4000,
        distributed_capable=False,
        depends_on=[]
    ),
    2: PipelineStep(
        step_number=2,
        name="Scorer Meta-Optimizer",
        description="Optimize ML scorer hyperparameters across distributed GPUs",
        required_inputs=["optimal_window_config.json", "bidirectional_survivors.json"],
        expected_outputs=["best_scorer_config.json", "scorer_optimization_results.json"],
        typical_duration_minutes=45,
        max_duration_minutes=180,
        requires_gpu=True,
        min_gpu_memory_mb=4000,
        distributed_capable=True,
        depends_on=[1]
    ),
    3: PipelineStep(
        step_number=3,
        name="Full Scoring",
        description="Score all survivors using optimized configuration",
        required_inputs=["best_scorer_config.json", "bidirectional_survivors.json"],
        expected_outputs=["scored_survivors.json", "feature_vectors.npy"],
        typical_duration_minutes=30,
        max_duration_minutes=120,
        requires_gpu=True,
        min_gpu_memory_mb=6000,
        distributed_capable=True,
        depends_on=[2]
    ),
    4: PipelineStep(
        step_number=4,
        name="ML Meta-Optimizer",
        description="Optimize neural network architecture for prediction",
        required_inputs=["scored_survivors.json", "feature_vectors.npy"],
        expected_outputs=["optimal_architecture.json", "ml_optimization_results.json"],
        typical_duration_minutes=60,
        max_duration_minutes=240,
        requires_gpu=True,
        min_gpu_memory_mb=8000,
        distributed_capable=True,
        depends_on=[3]
    ),
    5: PipelineStep(
        step_number=5,
        name="Anti-Overfit Training",
        description="Train final model with overfitting prevention",
        required_inputs=["optimal_architecture.json", "feature_vectors.npy"],
        expected_outputs=["trained_model.pt", "training_metrics.json"],
        typical_duration_minutes=90,
        max_duration_minutes=360,
        requires_gpu=True,
        min_gpu_memory_mb=8000,
        distributed_capable=True,
        depends_on=[4]
    ),
    6: PipelineStep(
        step_number=6,
        name="Prediction Generator",
        description="Generate prediction pools using trained model",
        required_inputs=["trained_model.pt", "lottery_data.json"],
        expected_outputs=["prediction_pool.json", "prediction_analysis.json"],
        typical_duration_minutes=15,
        max_duration_minutes=60,
        requires_gpu=True,
        min_gpu_memory_mb=4000,
        distributed_capable=False,
        depends_on=[5]
    )
}


class PipelineStepContext(BaseModel):
    """
    Context about current pipeline position and expectations.
    
    Helps AI agents understand where they are in the pipeline
    and what's expected at each step.
    """
    
    current_step: int = Field(ge=1, le=6)
    total_steps: int = 6
    
    # Step statuses
    step_statuses: Dict[int, StepStatus] = Field(default_factory=dict)
    
    # Current step info
    current_step_info: Optional[PipelineStep] = None
    
    # Validation results
    inputs_available: List[str] = Field(default_factory=list)
    inputs_missing: List[str] = Field(default_factory=list)
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Initialize step info
        if self.current_step_info is None:
            self.current_step_info = PIPELINE_STEPS.get(self.current_step)
        
        # Initialize statuses if empty
        if not self.step_statuses:
            for i in range(1, self.total_steps + 1):
                if i < self.current_step:
                    self.step_statuses[i] = StepStatus.COMPLETED
                elif i == self.current_step:
                    self.step_statuses[i] = StepStatus.RUNNING
                else:
                    self.step_statuses[i] = StepStatus.PENDING
    
    def validate_inputs(self, available_files: List[str]) -> bool:
        """
        Validate that required inputs are available.
        
        Args:
            available_files: List of available file paths/names
            
        Returns:
            True if all required inputs available
        """
        if not self.current_step_info:
            return True
        
        # Normalize file names (strip paths)
        available = set(f.split('/')[-1] for f in available_files)
        
        self.inputs_available = []
        self.inputs_missing = []
        
        for inp in self.current_step_info.required_inputs:
            inp_name = inp.split('/')[-1]
            if inp_name in available:
                self.inputs_available.append(inp)
            else:
                self.inputs_missing.append(inp)
        
        return len(self.inputs_missing) == 0
    
    def can_proceed_to_next(self) -> bool:
        """Check if can proceed to next step."""
        if self.current_step >= self.total_steps:
            return False
        
        return self.step_statuses.get(self.current_step) == StepStatus.COMPLETED
    
    def get_next_step(self) -> Optional[int]:
        """Get next step number if available."""
        if self.can_proceed_to_next():
            return self.current_step + 1
        return None
    
    def mark_completed(self):
        """Mark current step as completed."""
        self.step_statuses[self.current_step] = StepStatus.COMPLETED
    
    def mark_failed(self):
        """Mark current step as failed."""
        self.step_statuses[self.current_step] = StepStatus.FAILED
    
    def advance_to_next(self) -> bool:
        """
        Advance to next step if possible.
        
        Returns True if advanced, False otherwise.
        """
        next_step = self.get_next_step()
        if next_step:
            self.current_step = next_step
            self.current_step_info = PIPELINE_STEPS.get(next_step)
            self.step_statuses[next_step] = StepStatus.RUNNING
            return True
        return False
    
    def get_step_info(self, step: int) -> Optional[PipelineStep]:
        """Get info for a specific step."""
        return PIPELINE_STEPS.get(step)
    
    def to_context_dict(self) -> Dict[str, Any]:
        """
        Generate pipeline context as clean dict for LLM.
        
        Hybrid JSON approach - data only.
        """
        current = self.current_step_info.to_dict() if self.current_step_info else {}
        
        # Build progress summary
        progress = []
        for i in range(1, self.total_steps + 1):
            status = self.step_statuses.get(i, StepStatus.PENDING)
            step_info = PIPELINE_STEPS.get(i)
            progress.append({
                "step": i,
                "name": step_info.name if step_info else f"Step {i}",
                "status": status.value
            })
        
        result = {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_step_info": current,
            "progress": progress
        }
        
        # Add input validation if done
        if self.inputs_missing:
            result["inputs_missing"] = self.inputs_missing
        
        return result
    
    @classmethod
    def for_step(cls, step: int) -> "PipelineStepContext":
        """Create context for a specific step."""
        return cls(current_step=step)


# Convenience functions
def get_step_info(step: int) -> Optional[PipelineStep]:
    """Get pipeline step information."""
    return PIPELINE_STEPS.get(step)


def get_pipeline_overview() -> List[Dict[str, Any]]:
    """Get overview of all pipeline steps."""
    return [step.to_dict() for step in PIPELINE_STEPS.values()]
