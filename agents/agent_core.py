#!/usr/bin/env python3
"""
Agent Core - Base Agent Class for PRNG Analysis Pipeline
Version: 1.1.0 (Universal Agent Architecture)

This module provides the foundational BaseAgent class that all pipeline
agents inherit from. It integrates with the dual-LLM infrastructure
and provides standardized methods for:
- Manifest loading and validation
- LLM-powered reasoning (think/calculate/plan)
- Pipeline execution with metadata tracking
- Success criteria evaluation
- Agent chaining via follow_up_agent

Based on: combined_agent_framework.md, PROPOSAL_Schema_v1_0_4
"""

import json
import os
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import LLM router (graceful degradation if not available)
try:
    from llm_services import get_router
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM services not available - agents will run without AI reasoning")


@dataclass
class AgentResult:
    """Standardized result from agent execution"""
    success: bool
    confidence: float
    outputs: List[str]
    suggested_params: Dict[str, Any]
    reasoning: str
    follow_up_agent: Optional[str]
    execution_time_seconds: float
    llm_metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseAgent:
    """
    Base class for all pipeline agents.
    
    Agents are declarative - their behavior is defined in JSON manifests.
    Agents do not run code directly - they ask the coordinator to run jobs.
    Agents evaluate success and choose the next agent to run.
    
    Lifecycle:
        1. __init__() - Load manifest, initialize LLM
        2. prepare() - Validate inputs exist, check prerequisites
        3. run() - Execute the agent's primary action
        4. evaluate() - Check success criteria, compute confidence
        5. next() - Determine and return follow-up agent
    """
    
    def __init__(self, manifest_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize agent from manifest.
        
        Args:
            manifest_path: Path to agent JSON manifest
            config: Optional runtime configuration overrides
        """
        self.manifest_path = Path(manifest_path)
        self.manifest = self._load_manifest(manifest_path)
        self.config = config or {}
        self.run_id = self._generate_run_id()
        self.start_time = None
        self.end_time = None
        
        # Initialize LLM router if available
        self.llm = get_router() if LLM_AVAILABLE else None
        
        # Results storage
        self.result: Optional[AgentResult] = None
        
        logger.info(f"Initialized agent: {self.manifest['agent_name']} (run_id: {self.run_id})")
    
    def _load_manifest(self, path: str) -> Dict[str, Any]:
        """Load and validate agent manifest"""
        with open(path, 'r') as f:
            manifest = json.load(f)
        
        # Validate required fields
        required = ['agent_name', 'description', 'inputs', 'outputs', 'actions']
        missing = [f for f in required if f not in manifest]
        if missing:
            raise ValueError(f"Manifest missing required fields: {missing}")
        
        return manifest
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID for this execution"""
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        hash_suffix = hashlib.md5(f"{timestamp}{os.getpid()}".encode()).hexdigest()[:8]
        step = self.manifest.get('pipeline_step', 0)
        return f"step{step}_{timestamp}_{hash_suffix}"
    
    # =========================================================================
    # LLM Integration Methods
    # =========================================================================
    
    def think(self, context: str) -> str:
        """
        Let LLM reason about current state and next action.
        Auto-routes to appropriate model based on content.
        
        Args:
            context: Current state description for LLM to reason about
            
        Returns:
            LLM's reasoning response
        """
        if not self.llm:
            logger.warning("LLM not available - returning empty reasoning")
            return ""
        
        return self.llm.route(context)
    
    def calculate(self, math_query: str) -> str:
        """
        Explicit math query to Math Specialist LLM.
        
        Args:
            math_query: Mathematical question (PRNG states, residues, etc.)
            
        Returns:
            Math specialist's response
        """
        if not self.llm:
            logger.warning("LLM not available - returning empty calculation")
            return ""
        
        return self.llm.calculate(math_query)
    
    def plan(self, task_description: str) -> Dict[str, Any]:
        """
        Ask Orchestrator LLM to create execution plan.
        
        Args:
            task_description: Description of task to plan
            
        Returns:
            JSON execution plan with steps, parameters, expected_outputs
        """
        if not self.llm:
            logger.warning("LLM not available - returning empty plan")
            return {"steps": [], "parameters": {}, "expected_outputs": []}
        
        prompt = f"""
Create a JSON execution plan for the following task:
{task_description}

Return ONLY valid JSON with these keys:
- steps: List of execution steps
- parameters: Dict of recommended parameters
- expected_outputs: List of expected output files
- success_criteria: How to verify success
- estimated_runtime_minutes: Rough time estimate

Your response must be valid JSON only, no markdown or explanation.
"""
        response = self.llm.orchestrate(prompt)
        
        # Try to parse JSON from response
        try:
            # Handle potential markdown code blocks
            if '```json' in response:
                response = response.split('```json')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM plan response as JSON: {response[:200]}")
            return {"steps": [], "parameters": {}, "expected_outputs": [], "error": "Parse failed"}
    
    # =========================================================================
    # Lifecycle Methods (Override in subclasses)
    # =========================================================================
    
    def prepare(self) -> bool:
        """
        Validate inputs exist and prerequisites are met.
        
        Returns:
            True if ready to run, False otherwise
        """
        logger.info(f"[{self.manifest['agent_name']}] Preparing...")
        
        # Check input files exist
        inputs_spec = self.manifest.get('inputs', [])
        
        # Handle dict format: {"required": [...], "optional": [...]}
        if isinstance(inputs_spec, dict):
            required_inputs = inputs_spec.get('required', [])
        else:
            # Handle list format: ["file1", "file2"] - all required
            required_inputs = inputs_spec
        
        for input_spec in required_inputs:
            if isinstance(input_spec, dict):
                filepath = input_spec.get('file', '')
            else:
                filepath = input_spec
            if filepath and not Path(filepath).exists():
                logger.error(f"Required input missing: {filepath}")
                return False
        logger.info(f"[{self.manifest['agent_name']}] Preparation complete")
        return True
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the agent's primary action.
        Returns:
            Raw execution result dict
        """
        self.start_time = datetime.now(timezone.utc)
        logger.info(f"[{self.manifest['agent_name']}] Running...")
        
        results = {}
        
        for action in self.manifest.get('actions', []):
            action_type = action.get('type', 'run_script')
            action_name = action.get('name', 'unnamed')
            
            if action_type in ('run_script', 'subprocess'):
                # Handle command_template (subprocess) or script (run_script)
                if 'command_template' in action:
                    # Build full command from template
                    params = {**self.manifest.get('default_params', {}), **self.config}
                    cmd_str = action['command_template'].format(**params)
                    cmd_parts = cmd_str.split()
                    logger.info(f"  Executing: {cmd_str}")
                else:
                    # Legacy run_script format
                    script = action.get('script', '')
                    args = self._build_args(action.get('args_map', {}))
                    cmd_parts = ['python3', script] + args
                    logger.info(f"  Executing: python3 {script} {' '.join(args)}")
                
                try:
                    result = subprocess.run(
                        cmd_parts,
                        capture_output=True,
                        text=True,
                        timeout=action.get('timeout', 3600)
                    )
                    results[action_name] = {
                        'returncode': result.returncode,
                        'stdout': result.stdout[-2000:] if result.stdout else '',
                        'stderr': result.stderr[-2000:] if result.stderr else ''
                    }
                except subprocess.TimeoutExpired:
                    results[action_name] = {'returncode': -1, 'error': 'Timeout'}
                except Exception as e:
                    results[action_name] = {'returncode': -1, 'error': str(e)}
        
        self.end_time = datetime.now(timezone.utc)
        return results

    def evaluate(self, run_result: Dict[str, Any]) -> AgentResult:
        """
        Evaluate run results and compute confidence.
        
        Args:
            run_result: Raw result from run()
            
        Returns:
            AgentResult with success, confidence, and metadata
        """
        logger.info(f"[{self.manifest['agent_name']}] Evaluating results...")
        
        # Check success criteria
        success_condition = self.manifest.get('success_condition', '')
        success = self._check_success_condition(success_condition)
        
        # Check outputs exist
        outputs_exist = all(
            Path(out).exists() for out in self.manifest.get('outputs', [])
        )
        
        # Compute confidence (can be enhanced with LLM reasoning)
        confidence = 0.0
        if success and outputs_exist:
            confidence = 0.85  # Base confidence
            
            # Boost confidence if LLM validates (graceful degradation if servers down)
            if self.llm:
                try:
                    validation = self.think(f"""
                    Agent {self.manifest['agent_name']} completed with outputs: {self.manifest.get('outputs', [])}
                    Success condition '{success_condition}' evaluated to: {success}
                    On a scale of 0.0 to 1.0, what confidence should we have in these results?
                    Reply with just a number between 0.0 and 1.0.
                    """)
                    llm_confidence = float(validation.strip())
                    confidence = (confidence + llm_confidence) / 2
                except (ConnectionError, ValueError, Exception) as e:
                    logger.warning(f"LLM evaluation skipped (servers not running): {type(e).__name__}")
        
        # Build reasoning
        reasoning = f"Agent completed. Success: {success}, Outputs exist: {outputs_exist}"
        
        # Get execution time
        exec_time = 0.0
        if self.start_time and self.end_time:
            exec_time = (self.end_time - self.start_time).total_seconds()
        
        # Build result
        self.result = AgentResult(
            success=success and outputs_exist,
            confidence=confidence,
            outputs=self.manifest.get('outputs', []),
            suggested_params=self.manifest.get('default_suggested_params', {}),
            reasoning=reasoning,
            follow_up_agent=self.manifest.get('follow_up_agents', [None])[0] if success else None,
            execution_time_seconds=exec_time,
            llm_metadata=self.llm.get_llm_metadata() if self.llm else {}
        )
        
        return self.result
    
    def next(self) -> Optional[str]:
        """
        Determine the next agent to run.
        
        Returns:
            Name of follow-up agent, or None if pipeline complete
        """
        if not self.result:
            return None
        
        if self.result.success and self.result.confidence >= 0.8:
            return self.result.follow_up_agent
        
        return None  # Requires human review
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _build_args(self, args_map: Dict[str, str]) -> List[str]:
        """Build command-line arguments from args_map"""
        args = []
        for flag, config_key in args_map.items():
            value = self.config.get(config_key, self.manifest.get('default_params', {}).get(config_key))
            if value is not None:
                args.extend([f'--{flag}', str(value)])
        return args
    
    def _check_success_condition(self, condition: str) -> bool:
        """Evaluate success condition string"""
        if not condition:
            return True
        
        # Simple file existence check
        if 'exists' in condition:
            # Parse "filename.json exists" format
            parts = condition.split()
            if len(parts) >= 1:
                return Path(parts[0]).exists()
        
        return True
    
    def get_agent_metadata(self) -> Dict[str, Any]:
        """
        Build agent_metadata dict for schema v1.0.4 compliance.
        
        Returns:
            Dict ready to inject into result JSON
        """
        inputs = []
        for inp in self.manifest.get('inputs', []):
            if isinstance(inp, dict):
                filepath = inp.get('file', '')
            else:
                filepath = inp
            
            if filepath and Path(filepath).exists():
                with open(filepath, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                inputs.append({
                    'file': filepath,
                    'hash': f'sha256:{file_hash[:12]}',
                    'required': True
                })
        
        return {
            'inputs': inputs,
            'outputs': self.manifest.get('outputs', []),
            'parent_run_id': self.config.get('parent_run_id'),
            'pipeline_step': self.manifest.get('pipeline_step', 0),
            'pipeline_step_name': self.manifest.get('pipeline_step_name', self.manifest['agent_name']),
            'follow_up_agent': self.result.follow_up_agent if self.result else None,
            'confidence': self.result.confidence if self.result else 0.0,
            'suggested_params': self.result.suggested_params if self.result else {},
            'reasoning': self.result.reasoning if self.result else '',
            'success_criteria_met': self.result.success if self.result else False,
            'retry_count': 0,
            'cluster_resources': {
                'nodes': ['zeus', 'rig-6600', 'rig-6600b', 'rig-6600c'],
                'total_gpus': 26,
                'total_tflops': 285.69
            },
            'llm_metadata': self.result.llm_metadata if self.result else {}
        }
    
    def finalize_run(self, result_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inject agent_metadata into result before saving.
        
        Args:
            result_dict: The result dict to enhance
            
        Returns:
            Enhanced result dict with agent_metadata
        """
        result_dict['agent_metadata'] = self.get_agent_metadata()
        
        # Reset LLM metrics for next run
        if self.llm:
            self.llm.reset_metrics()
        
        return result_dict


# =============================================================================
# Agent Factory
# =============================================================================

def load_agent(manifest_path: str, config: Optional[Dict[str, Any]] = None) -> BaseAgent:
    """
    Factory function to load an agent from manifest.
    
    Args:
        manifest_path: Path to agent manifest JSON
        config: Optional runtime configuration
        
    Returns:
        Initialized BaseAgent instance
    """
    return BaseAgent(manifest_path, config)


def run_agent_pipeline(start_agent: str, manifests_dir: str = 'agent_manifests',
                       config: Optional[Dict[str, Any]] = None,
                       max_steps: int = 10,
                       confidence_threshold: float = 0.8) -> List[AgentResult]:
    """
    Run a full agent pipeline starting from specified agent.
    
    Args:
        start_agent: Name of starting agent (without .json)
        manifests_dir: Directory containing agent manifests
        config: Runtime configuration
        max_steps: Maximum pipeline steps (safety limit)
        confidence_threshold: Minimum confidence to auto-proceed
        
    Returns:
        List of AgentResults from each step
    """
    results = []
    current_agent = start_agent
    step = 0
    
    while current_agent and step < max_steps:
        manifest_path = Path(manifests_dir) / f"{current_agent}.json"
        
        if not manifest_path.exists():
            logger.error(f"Agent manifest not found: {manifest_path}")
            break
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PIPELINE STEP {step + 1}: {current_agent}")
        logger.info(f"{'='*60}")
        
        # Load and run agent
        agent = load_agent(str(manifest_path), config)
        
        if not agent.prepare():
            logger.error(f"Agent {current_agent} failed preparation")
            break
        
        run_result = agent.run()
        agent_result = agent.evaluate(run_result)
        results.append(agent_result)
        
        # Check if we should proceed
        if not agent_result.success:
            logger.warning(f"Agent {current_agent} failed - stopping pipeline")
            break
        
        if agent_result.confidence < confidence_threshold:
            logger.warning(f"Confidence {agent_result.confidence:.2f} below threshold {confidence_threshold}")
            logger.warning("Human review required - stopping pipeline")
            break
        
        # Get next agent
        current_agent = agent.next()
        step += 1
        
        # Update config with suggested params for next step
        if config is None:
            config = {}
        config.update(agent_result.suggested_params)
        config['parent_run_id'] = agent.run_id
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PIPELINE COMPLETE: {len(results)} steps executed")
    logger.info(f"{'='*60}")
    
    return results


if __name__ == '__main__':
    # Test agent loading
    print("Agent Core v1.1.0 - Universal Agent Architecture")
    print("=" * 60)
    print(f"LLM Available: {LLM_AVAILABLE}")
    print("\nUsage:")
    print("  from agents.agent_core import load_agent, run_agent_pipeline")
    print("  agent = load_agent('agent_manifests/window_optimizer.json')")
    print("  results = run_agent_pipeline('window_optimizer')")
