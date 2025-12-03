"""
LLM Services - Dual-Model Architecture for Distributed PRNG Analysis
Schema Version: 1.0.5

v1.0.5 Features (Team Beta Review):
- Model-specific max_tokens (2048 orchestrator, 512 math)
- Context reset trigger at 14K tokens
- Agent identity headers in prompts
- Expanded PRNG routing keywords
- Orchestrator → Math delegation mechanism
- Rotating disk-based logging
- Human override trigger detection

Quick Start:
    from llm_services import get_router
    
    router = get_router()
    
    # Set agent identity for better LLM grounding
    router.set_agent("scorer_meta_optimizer")
    
    # Auto-routing based on content
    response = router.route("Calculate the match rate for seed 12345")
    
    # Force specific endpoint
    plan = router.orchestrate("Create execution plan for Step 3")
    result = router.calculate("What is 6364136223846793005 mod 2^32?")
    
    # Check for human override request
    if router.is_human_override_requested():
        print("LLM requested human review!")
    
    # Get metadata for schema injection
    metadata = router.get_llm_metadata()
"""

from .llm_router import (
    LLMRouter, 
    get_router, 
    reset_router, 
    LLMMetrics
)

__all__ = [
    "LLMRouter",
    "get_router", 
    "reset_router",
    "LLMMetrics"
]

__version__ = "1.0.5"
__schema_version__ = "1.0.5"
__review_status__ = "Team Beta approved"
