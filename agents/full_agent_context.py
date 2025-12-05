#!/usr/bin/env python3
"""
Full Agent Context - Complete context assembly for LLM evaluation.

Combines all framework components into a single, cohesive context
that provides the AI agent with everything needed for decision-making.

Version: 3.2.0
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from agents.manifest import AgentManifest
from agents.parameters import ParameterContext
from agents.history import AnalysisHistory, RunRecord
from agents.runtime import RuntimeContext, detect_runtime
from agents.safety import KillSwitch
from agents.pipeline import PipelineStepContext
from agents.contexts import BaseAgentContext, get_context_for_step
from agents.doctrine import get_doctrine, get_doctrine_summary


class FullAgentContext(BaseModel):
    """
    Complete context assembly for AI agent decision-making.
    
    Combines:
    - Manifest (agent contract)
    - Parameters (adjustable with bounds)
    - Specialized context (step-specific evaluation)
    - History (trend analysis)
    - Runtime (GPU/compute info)
    - Safety (kill switch status)
    - Pipeline (position and expectations)
    - Doctrine (decision rules)
    """
    
    # Core identifiers
    step: int = Field(ge=1, le=6)
    run_number: int = 1
    run_id: str = ""
    
    # Component contexts
    manifest: Optional[AgentManifest] = None
    parameters: Optional[ParameterContext] = None
    agent_context: Optional[BaseAgentContext] = None
    history: Optional[AnalysisHistory] = None
    runtime: Optional[RuntimeContext] = None
    safety: Optional[KillSwitch] = None
    pipeline: Optional[PipelineStepContext] = None
    
    # Results being evaluated
    results: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Generate run_id if not provided
        if not self.run_id:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"step{self.step}_run{self.run_number}_{timestamp}"
    
    @classmethod
    def build(
        cls,
        step: int,
        results: Dict[str, Any],
        run_number: int = 1,
        manifest_path: Optional[str] = None,
        history: Optional[AnalysisHistory] = None,
        detect_hardware: bool = True
    ) -> "FullAgentContext":
        """
        Factory method to build complete context.
        
        Args:
            step: Pipeline step number (1-6)
            results: Results dict to evaluate
            run_number: Current run number
            manifest_path: Optional path to agent manifest
            history: Optional existing history object
            detect_hardware: Whether to detect GPU hardware
            
        Returns:
            Fully assembled FullAgentContext
        """
        ctx = cls(
            step=step,
            run_number=run_number,
            results=results
        )
        
        # Load manifest if path provided
        if manifest_path:
            ctx.manifest = AgentManifest.load(manifest_path)
            ctx.parameters = ParameterContext.build(
                ctx.manifest,
                results.get("config", {})
            )
        
        # Create specialized agent context
        ctx.agent_context = get_context_for_step(
            step=step,
            results=results,
            run_number=run_number,
            manifest_path=manifest_path
        )
        
        # Set up history
        ctx.history = history or AnalysisHistory()
        
        # Detect runtime if requested
        if detect_hardware:
            ctx.runtime = detect_runtime()
        else:
            ctx.runtime = RuntimeContext()
        
        # Initialize safety
        ctx.safety = KillSwitch()
        ctx.safety.check_all()
        
        # Set up pipeline context
        ctx.pipeline = PipelineStepContext.for_step(step)
        
        return ctx
    
    def is_safe(self) -> bool:
        """Check if safe to proceed with evaluation."""
        if self.safety:
            return self.safety.is_safe()
        return True
    
    def to_context_dict(self) -> Dict[str, Any]:
        """
        Generate complete context as structured dict.
        
        This is the hybrid JSON approach - all data in clean structure.
        """
        context = {
            "meta": {
                "run_id": self.run_id,
                "step": self.step,
                "run_number": self.run_number,
                "created_at": self.created_at.isoformat()
            },
            "doctrine": get_doctrine(),
            "results": self.results
        }
        
        # Add manifest if available
        if self.manifest:
            context["manifest"] = {
                "agent_name": self.manifest.agent_name,
                "pipeline_step": self.manifest.pipeline_step,
                "success_condition": self.manifest.success_condition,
                "inputs": self.manifest.inputs,
                "outputs": self.manifest.outputs,
                "max_retries": self.manifest.retry
            }
        
        # Add parameters if available
        if self.parameters:
            context["parameters"] = self.parameters.to_context_dict()
        
        # Add specialized agent context (with evaluations)
        if self.agent_context:
            context["evaluation"] = self.agent_context.to_context_dict()
        
        # Add history
        if self.history:
            agent_name = self.manifest.agent_name if self.manifest else None
            context["history"] = self.history.to_context_dict(agent_name)
        
        # Add runtime
        if self.runtime:
            context["runtime"] = self.runtime.to_context_dict()
        
        # Add safety
        if self.safety:
            context["safety"] = self.safety.to_context_dict()
        
        # Add pipeline
        if self.pipeline:
            context["pipeline"] = self.pipeline.to_context_dict()
        
        return context
    
    def to_llm_prompt(self) -> str:
        """
        Generate complete prompt for LLM evaluation.
        
        Assembles all context into a structured prompt with:
        1. Doctrine summary
        2. Full context as JSON
        3. Task instruction
        4. Output format specification
        """
        # Check safety first
        if not self.is_safe():
            return self._safety_halt_prompt()
        
        # Get full context
        context = self.to_context_dict()
        
        # Build prompt
        prompt_parts = []
        
        # 1. Doctrine summary (brief rules)
        prompt_parts.append("DOCTRINE:")
        prompt_parts.append(get_doctrine_summary())
        prompt_parts.append("")
        
        # 2. Full context as JSON
        prompt_parts.append("CONTEXT:")
        prompt_parts.append(json.dumps(context, indent=2, default=str))
        prompt_parts.append("")
        
        # 3. Task instruction
        prompt_parts.append("TASK:")
        prompt_parts.append("Evaluate the results against the success condition and thresholds.")
        prompt_parts.append("Determine if the pipeline should proceed, retry, or escalate.")
        prompt_parts.append("")
        
        # 4. Output format
        prompt_parts.append("OUTPUT FORMAT:")
        prompt_parts.append(self._get_output_format())
        
        return "\n".join(prompt_parts)
    
    def _safety_halt_prompt(self) -> str:
        """Generate prompt when safety halt is triggered."""
        safety_ctx = self.safety.to_context_dict() if self.safety else {}
        
        return f"""SAFETY HALT TRIGGERED

The kill switch has been activated. Pipeline execution is paused.

Safety Status:
{json.dumps(safety_ctx, indent=2)}

Required Response:
{{
    "success_condition_met": false,
    "confidence": 0.0,
    "reasoning": "Safety halt triggered - human review required",
    "recommended_action": "escalate",
    "warnings": ["Kill switch activated"]
}}"""
    
    def _get_output_format(self) -> str:
        """Get the expected output format specification."""
        # Get adjustable param names for example
        param_names = []
        if self.parameters:
            params = self.parameters.to_context_dict().get("adjustable_parameters", [])
            param_names = [p.get("name", "param") for p in params[:2]]
        
        example_params = ", ".join(f'"{p}": <value>' for p in param_names) if param_names else '"param": <value>'
        
        return f"""Respond with ONLY valid JSON (no markdown, no extra text):

{{
    "success_condition_met": true | false,
    "confidence": 0.00 to 1.00,
    "reasoning": "Brief explanation using metrics from evaluation",
    "recommended_action": "proceed" | "retry" | "escalate",
    "suggested_param_adjustments": {{
        {example_params}
    }},
    "warnings": ["any concerns or anomalies"]
}}

NOTES:
- suggested_param_adjustments only required if action is "retry"
- Use parameter names and bounds from CONTEXT.parameters
- warnings array can be empty []"""
    
    def record_to_history(self, decision: Dict[str, Any]):
        """
        Record this run to history after decision is made.
        
        Args:
            decision: The AI's decision dict
        """
        if not self.history:
            self.history = AnalysisHistory()
        
        record = RunRecord(
            run_id=self.run_id,
            run_number=self.run_number,
            timestamp=datetime.utcnow(),
            agent_name=self.manifest.agent_name if self.manifest else f"step_{self.step}_agent",
            pipeline_step=self.step,
            success=decision.get("success_condition_met", False),
            confidence=decision.get("confidence", 0.0),
            execution_time_seconds=self.results.get("execution_time_seconds", 0),
            metrics={k: v for k, v in self.results.items() if isinstance(v, (int, float))},
            action_taken=decision.get("recommended_action", ""),
            param_adjustments=decision.get("suggested_param_adjustments", {})
        )
        
        self.history.add_run(record)
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get a quick summary of the evaluation."""
        if self.agent_context:
            success, confidence = self.agent_context.get_overall_success()
            return {
                "step": self.step,
                "success": success,
                "confidence": confidence,
                "interpretation": self.agent_context.interpret_results()
            }
        return {
            "step": self.step,
            "success": None,
            "confidence": None,
            "interpretation": "No evaluation performed"
        }


def build_full_context(
    step: int,
    results: Dict[str, Any],
    run_number: int = 1,
    manifest_path: Optional[str] = None,
    history_path: Optional[str] = None
) -> FullAgentContext:
    """
    Convenience function to build full context.
    
    Args:
        step: Pipeline step (1-6)
        results: Results to evaluate
        run_number: Current run number
        manifest_path: Path to agent manifest JSON
        history_path: Path to history JSON file
        
    Returns:
        Complete FullAgentContext ready for LLM
    """
    # Load history if path provided
    history = None
    if history_path:
        from agents.history import load_history
        history = load_history(history_path)
    
    return FullAgentContext.build(
        step=step,
        results=results,
        run_number=run_number,
        manifest_path=manifest_path,
        history=history
    )
