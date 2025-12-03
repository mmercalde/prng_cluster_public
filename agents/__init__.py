"""
Agents Package - Universal Agent Architecture v1.1
"""
from .agent_core import BaseAgent, load_agent, run_agent_pipeline, AgentResult

__all__ = ['BaseAgent', 'load_agent', 'run_agent_pipeline', 'AgentResult']
__version__ = '1.1.0'
