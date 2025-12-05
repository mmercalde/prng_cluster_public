"""
Pydantic Agent Context Framework v3.2.0

This package provides the complete context infrastructure for autonomous
AI agent operation in the distributed PRNG analysis pipeline.

Modules:
    manifest/       - Pydantic manifest models
    parameters/     - Parameter awareness for AI agents
    registry/       - Script parameter bounds
    history/        - Multi-run analysis history
    runtime/        - GPU/compute detection
    safety/         - Kill switch controls
    pipeline/       - Pipeline step expectations
    contexts/       - Specialized agent contexts
    doctrine        - Shared reasoning rules
    full_agent_context - Complete context assembly
    watcher_agent   - Autonomous pipeline monitoring

Key exports:
    WatcherAgent    - Autonomous pipeline monitoring (main entry point)
    FullAgentContext - Complete LLM-ready context
    build_full_context - Convenience factory function
    AgentManifest   - Load and validate agent manifests
    AgentDecision   - Validated LLM decision response
    get_context_for_step - Factory for specialized agent contexts
"""

__version__ = "3.2.0"

# Phase 6: Watcher Agent (main entry point for autonomy)
from agents.watcher_agent import WatcherAgent, WatcherConfig

# Sub-Phase 4: Full Context Integration
from agents.full_agent_context import FullAgentContext, build_full_context
from agents.doctrine import get_doctrine, get_doctrine_summary, validate_decision_against_doctrine

# Sub-Phase 1: Core Infrastructure
from agents.manifest import AgentManifest
from agents.agent_decision import AgentDecision, parse_llm_response
from agents.parameters import ParameterContext
from agents.prompt_builder import build_evaluation_prompt, get_thresholds_for_step

# Sub-Phase 2: Team Beta Infrastructure
from agents.history import AnalysisHistory, RunRecord, load_history
from agents.runtime import RuntimeContext, detect_runtime, get_gpu_summary
from agents.safety import KillSwitch, check_safety, create_halt, clear_halt
from agents.pipeline import PipelineStepContext, get_step_info, get_pipeline_overview

# Sub-Phase 3: Specialized Agent Contexts
from agents.contexts import (
    BaseAgentContext,
    WindowOptimizerContext,
    ScorerMetaContext,
    FullScoringContext,
    MLMetaContext,
    AntiOverfitContext,
    PredictionContext,
    get_context_for_step,
    EvaluationResult
)
