# Universal Agent Architecture Proposal v1.0

**Document Version:** 1.0  
**Date:** December 2, 2025  
**Author:** Claude (AI Assistant)  
**Status:** DRAFT - Pending Review  
**Compatibility:** Schema v1.0.3+, Dual-LLM Infrastructure  

---

## Executive Summary

This proposal defines a **Universal Agent Architecture** for the Distributed PRNG Analysis System. The architecture implements a single `BaseAgent` class that all 6 pipeline agents inherit from, ensuring consistent coding patterns while allowing step-specific execution logic.

**Key Principles:**
- One base pattern, six implementations (like Step 2.5 → Step 3 reuse)
- Agents are **config-aware** and can **modify parameters** via LLM consultation
- Full compatibility with existing `combined_agent_framework.md` proposal
- Integration with deployed Dual-LLM infrastructure (Qwen2.5-Coder-14B + Qwen2.5-Math-7B)

**Estimated Effort:**
- Base infrastructure: ~4 hours
- Per-agent implementation: ~30-60 minutes each
- Total: ~8-10 hours

---

## Part 1: Architecture Overview

### 1.1 Design Goals

| Goal | Solution |
|------|----------|
| Consistent coding pattern | `BaseAgent` class with inheritance |
| Config awareness | `config_manifests/` with tunable parameter definitions |
| Autonomous adjustment | LLM consultation when success criteria fail |
| Two execution modes | Seed-based (Steps 1-2) vs Script-based (Steps 2.5-6) |
| Backward compatible | Extends existing agent proposal, no breaking changes |

### 1.2 File Structure

```
distributed_prng_analysis/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py              # THE PATTERN (90% shared logic)
│   ├── agent_manager.py           # Orchestrates pipeline execution
│   │
│   ├── step1_window_agent.py      # Seed-based execution
│   ├── step2_sieve_agent.py       # Seed-based execution
│   ├── step3_scorer_agent.py      # Script-based execution
│   ├── step4_ml_agent.py          # Script-based execution
│   ├── step5_training_agent.py    # Script-based execution
│   └── step6_predict_agent.py     # Script-based execution
│
├── agent_manifests/               # What each step DOES
│   ├── step1_window_optimizer.json
│   ├── step2_scorer_meta.json
│   ├── step3_full_scoring.json
│   ├── step4_ml_meta.json
│   ├── step5_anti_overfit.json
│   └── step6_prediction.json
│
├── config_manifests/              # What each step CAN TUNE
│   ├── step1_window_optimizer.json
│   ├── step2_scorer_meta.json
│   ├── step3_full_scoring.json
│   ├── step4_ml_meta.json
│   ├── step5_anti_overfit.json
│   └── step6_prediction.json
│
├── configs/                       # CURRENT parameter values
│   ├── window_config.json
│   ├── scorer_config.json
│   ├── ml_config.json
│   ├── training_config.json
│   └── prediction_config.json
│
├── integration/                   # Already deployed ✅
│   ├── metadata_writer.py
│   └── ...
│
└── llm_services/                  # Already deployed ✅
    ├── llm_router.py
    └── ...
```

### 1.3 Execution Modes

| Steps | Mode | Execution Method | Primary Script |
|-------|------|------------------|----------------|
| 1 | Seed-based | Direct sieve_filter.py calls | window_optimizer.py |
| 2 | Seed-based | Direct sieve_filter.py calls | bidirectional sieve |
| 2.5/3 | Script-based | coordinator.py jobs | scorer_trial_worker.py |
| 4 | Script-based | coordinator.py jobs | adaptive_meta_optimizer.py |
| 5 | Script-based | coordinator.py jobs | meta_prediction_optimizer_anti_overfit.py |
| 6 | Local | Direct model inference | reinforcement_engine.py |

---

## Part 2: Base Agent Specification

### 2.1 BaseAgent Class

```python
#!/usr/bin/env python3
"""
Base Agent - Universal Pattern for All Pipeline Agents
======================================================

All 6 pipeline agents inherit from this class.
90% of logic is shared; only execute() differs per step.

Location: agents/base_agent.py
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from llm_services.llm_router import get_router
from integration.metadata_writer import inject_agent_metadata


class BaseAgent(ABC):
    """
    Universal agent base class.
    
    Provides:
    - Manifest loading (agent + config)
    - LLM integration
    - Success criteria evaluation
    - Config modification loop
    - Metadata injection
    - Standard run loop
    """
    
    def __init__(self, step: int, config_override: Optional[Dict] = None):
        """
        Initialize agent for a specific pipeline step.
        
        Args:
            step: Pipeline step number (1-6)
            config_override: Optional config overrides (for testing/automation)
        """
        self.step = step
        self.logger = self._setup_logging()
        
        # Load manifests
        self.agent_manifest = self._load_agent_manifest()
        self.config_manifest = self._load_config_manifest()
        self.current_config = self._load_current_config()
        
        # Apply overrides
        if config_override:
            self.current_config.update(config_override)
        
        # Initialize LLM router
        self.llm = get_router()
        self.llm.set_context(f"Step {step}: {self.agent_manifest['name']}")
        
        # Track execution state
        self.run_id = None
        self.parent_run_id = None
        self.attempt = 0
        self.results = None
    
    # =========================================================================
    # ABSTRACT METHOD - Must be implemented by each agent
    # =========================================================================
    
    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the step-specific logic.
        
        This is the ONLY method that differs between agents.
        Seed-based agents call sieve_filter directly.
        Script-based agents submit jobs to coordinator.
        
        Returns:
            Dict containing execution results
        """
        pass
    
    # =========================================================================
    # STANDARD METHODS - Identical across all agents
    # =========================================================================
    
    def prepare(self) -> bool:
        """
        Validate prerequisites before execution.
        
        Checks:
        - Required input files exist
        - Config is valid
        - Resources available
        
        Returns:
            True if ready to execute, False otherwise
        """
        self.logger.info(f"Preparing Step {self.step}: {self.agent_manifest['name']}")
        
        # Check required inputs
        for input_spec in self.agent_manifest.get('inputs', []):
            input_file = input_spec if isinstance(input_spec, str) else input_spec.get('file')
            required = input_spec.get('required', True) if isinstance(input_spec, dict) else True
            
            if required and not Path(input_file).exists():
                self.logger.error(f"Required input missing: {input_file}")
                return False
        
        # Validate config against manifest
        validation = self._validate_config()
        if not validation['valid']:
            self.logger.warning(f"Config validation warnings: {validation['warnings']}")
        
        self.run_id = self._generate_run_id()
        self.logger.info(f"Run ID: {self.run_id}")
        
        return True
    
    def evaluate(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate results against success criteria.
        
        Args:
            results: Output from execute()
            
        Returns:
            Dict with 'success', 'metrics', 'failures' keys
        """
        criteria = self.config_manifest.get('success_criteria', {})
        evaluation = {
            'success': True,
            'metrics': {},
            'failures': []
        }
        
        # Check each criterion
        for criterion, threshold in criteria.items():
            actual_value = results.get(criterion)
            
            if actual_value is None:
                continue
            
            evaluation['metrics'][criterion] = {
                'actual': actual_value,
                'threshold': threshold
            }
            
            # Evaluate based on criterion type
            if criterion.startswith('min_'):
                if actual_value < threshold:
                    evaluation['success'] = False
                    evaluation['failures'].append(
                        f"{criterion}: {actual_value} < {threshold}"
                    )
            elif criterion.startswith('max_'):
                if actual_value > threshold:
                    evaluation['success'] = False
                    evaluation['failures'].append(
                        f"{criterion}: {actual_value} > {threshold}"
                    )
        
        # Calculate confidence score
        evaluation['confidence'] = self._calculate_confidence(results, evaluation)
        
        return evaluation
    
    def adjust(self, results: Dict, evaluation: Dict) -> Dict[str, Any]:
        """
        Consult LLM for parameter adjustments when criteria not met.
        
        Args:
            results: Execution results
            evaluation: Evaluation results with failures
            
        Returns:
            Dict with 'action', 'changes', 'reasoning' keys
        """
        # Build prompt with full context
        prompt = self._build_adjustment_prompt(results, evaluation)
        
        # Consult orchestrator LLM
        self.logger.info("Consulting LLM for parameter adjustment...")
        response = self.llm.orchestrate(prompt)
        
        try:
            adjustment = json.loads(response)
        except json.JSONDecodeError:
            self.logger.error(f"LLM returned invalid JSON: {response}")
            return {
                'action': 'escalate',
                'changes': {},
                'reasoning': 'LLM response parsing failed'
            }
        
        self.logger.info(f"LLM recommendation: {adjustment['action']}")
        self.logger.info(f"Reasoning: {adjustment.get('reasoning', 'None provided')}")
        
        return adjustment
    
    def apply_changes(self, changes: Dict[str, Any]) -> bool:
        """
        Apply parameter changes to current config.
        
        Args:
            changes: Dict of parameter -> new_value
            
        Returns:
            True if changes applied successfully
        """
        for param, new_value in changes.items():
            # Validate against config manifest
            param_spec = self.config_manifest.get('parameters', {}).get(param)
            
            if param_spec:
                # Check range constraints
                if 'range' in param_spec:
                    min_val, max_val = param_spec['range']
                    if not (min_val <= new_value <= max_val):
                        self.logger.warning(
                            f"Value {new_value} for {param} outside range [{min_val}, {max_val}]"
                        )
                        continue
            
            self.logger.info(f"Adjusting {param}: {self.current_config.get(param)} → {new_value}")
            self.current_config[param] = new_value
        
        # Save updated config
        self._save_current_config()
        
        return True
    
    def finalize(self, results: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Finalize results with metadata injection.
        
        Args:
            results: Execution results
            evaluation: Evaluation results
            
        Returns:
            Results with agent_metadata injected
        """
        # Determine next agent
        if evaluation['success']:
            follow_up = self.agent_manifest.get('follow_up_agent')
        else:
            follow_up = None  # Will require manual intervention
        
        # Inject metadata
        results = inject_agent_metadata(
            results,
            inputs=[inp if isinstance(inp, str) else inp.get('file') 
                    for inp in self.agent_manifest.get('inputs', [])],
            outputs=self.agent_manifest.get('outputs', []),
            parent_run_id=self.parent_run_id,
            pipeline_step=self.step,
            follow_up_agent=follow_up,
            confidence=evaluation.get('confidence', 0.5),
            success_criteria_met=evaluation['success'],
            suggested_params=self.current_config if evaluation['success'] else None,
            reasoning=f"Step {self.step} {'completed successfully' if evaluation['success'] else 'failed criteria'}"
        )
        
        # Add LLM metadata if available
        if hasattr(self.llm, 'get_llm_metadata'):
            results['agent_metadata']['llm_metadata'] = self.llm.get_llm_metadata()
        
        # Save results
        output_path = self._get_output_path()
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def trigger_next(self) -> Optional[str]:
        """
        Trigger the next agent in the pipeline.
        
        Returns:
            Name of next agent triggered, or None
        """
        follow_up = self.agent_manifest.get('follow_up_agent')
        
        if not follow_up:
            self.logger.info("Pipeline complete - no follow-up agent")
            return None
        
        self.logger.info(f"Triggering next agent: {follow_up}")
        
        # In autonomous mode, this would trigger the next agent
        # For now, just return the name
        return follow_up
    
    # =========================================================================
    # MAIN RUN LOOP - Standard execution pattern
    # =========================================================================
    
    def run(self, 
            max_retries: int = 3, 
            parent_run_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main execution loop - IDENTICAL for all agents.
        
        Args:
            max_retries: Maximum adjustment attempts before escalation
            parent_run_id: Run ID of parent step (for lineage tracking)
            
        Returns:
            Final results dict with metadata
        """
        self.parent_run_id = parent_run_id
        
        # Phase 1: Preparation
        if not self.prepare():
            return {
                'success': False,
                'error': 'Preparation failed',
                'step': self.step
            }
        
        # Phase 2: Execute → Evaluate → Adjust loop
        for self.attempt in range(max_retries):
            self.logger.info(f"Attempt {self.attempt + 1}/{max_retries}")
            
            # Execute step-specific logic
            results = self.execute()
            self.results = results
            
            # Evaluate against success criteria
            evaluation = self.evaluate(results)
            
            if evaluation['success']:
                # Success! Finalize and optionally trigger next
                self.logger.info(f"Step {self.step} PASSED (confidence: {evaluation['confidence']:.2f})")
                final_results = self.finalize(results, evaluation)
                self.trigger_next()
                return final_results
            
            # Criteria not met - consult LLM for adjustments
            self.logger.warning(f"Step {self.step} FAILED criteria: {evaluation['failures']}")
            
            adjustment = self.adjust(results, evaluation)
            
            if adjustment['action'] == 'modify_config':
                self.apply_changes(adjustment['changes'])
                continue  # Retry with new config
            
            elif adjustment['action'] == 'proceed':
                # LLM says proceed despite failures
                self.logger.info("LLM recommends proceeding despite failures")
                final_results = self.finalize(results, evaluation)
                self.trigger_next()
                return final_results
            
            elif adjustment['action'] == 'escalate':
                # Immediate escalation
                return self._escalate(results, evaluation, adjustment['reasoning'])
        
        # Max retries exceeded
        return self._escalate(
            results, 
            evaluation, 
            f"Max retries ({max_retries}) exceeded"
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for this agent."""
        logger = logging.getLogger(f"Agent_Step{self.step}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'[Step {self.step}] %(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_agent_manifest(self) -> Dict[str, Any]:
        """Load agent manifest JSON."""
        manifest_path = Path(f"agent_manifests/step{self.step}_*.json")
        matches = list(Path("agent_manifests").glob(f"step{self.step}_*.json"))
        
        if not matches:
            raise FileNotFoundError(f"No agent manifest found for step {self.step}")
        
        with open(matches[0]) as f:
            return json.load(f)
    
    def _load_config_manifest(self) -> Dict[str, Any]:
        """Load config manifest JSON."""
        manifest_path = Path(f"config_manifests/step{self.step}_*.json")
        matches = list(Path("config_manifests").glob(f"step{self.step}_*.json"))
        
        if not matches:
            raise FileNotFoundError(f"No config manifest found for step {self.step}")
        
        with open(matches[0]) as f:
            return json.load(f)
    
    def _load_current_config(self) -> Dict[str, Any]:
        """Load current configuration."""
        config_file = self.agent_manifest.get('config_file')
        
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                return json.load(f)
        
        # Return defaults from config manifest
        defaults = {}
        for param, spec in self.config_manifest.get('parameters', {}).items():
            if 'default' in spec:
                defaults[param] = spec['default']
        
        return defaults
    
    def _save_current_config(self):
        """Save current configuration to disk."""
        config_file = self.agent_manifest.get('config_file')
        
        if config_file:
            with open(config_file, 'w') as f:
                json.dump(self.current_config, f, indent=2)
    
    def _validate_config(self) -> Dict[str, Any]:
        """Validate current config against manifest."""
        result = {'valid': True, 'warnings': []}
        
        for param, spec in self.config_manifest.get('parameters', {}).items():
            value = self.current_config.get(param)
            
            if value is None and spec.get('required', False):
                result['warnings'].append(f"Required parameter missing: {param}")
            
            if value is not None and 'range' in spec:
                min_val, max_val = spec['range']
                if not (min_val <= value <= max_val):
                    result['warnings'].append(
                        f"{param}={value} outside range [{min_val}, {max_val}]"
                    )
        
        return result
    
    def _generate_run_id(self) -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"step{self.step}_{timestamp}"
    
    def _get_output_path(self) -> Path:
        """Get output path for results."""
        outputs = self.agent_manifest.get('outputs', [])
        if outputs:
            return Path(outputs[0])
        return Path(f"results/step{self.step}_{self.run_id}.json")
    
    def _calculate_confidence(self, results: Dict, evaluation: Dict) -> float:
        """Calculate confidence score based on results and evaluation."""
        if not evaluation['success']:
            return 0.3  # Low confidence if criteria failed
        
        # Calculate based on how much results exceed thresholds
        scores = []
        for criterion, metrics in evaluation['metrics'].items():
            actual = metrics['actual']
            threshold = metrics['threshold']
            
            if criterion.startswith('min_'):
                if threshold > 0:
                    scores.append(min(actual / threshold, 2.0))  # Cap at 2x threshold
            elif criterion.startswith('max_'):
                if actual > 0:
                    scores.append(min(threshold / actual, 2.0))
        
        if scores:
            return min(sum(scores) / len(scores) / 2, 1.0)  # Normalize to [0, 1]
        
        return 0.7  # Default confidence
    
    def _build_adjustment_prompt(self, results: Dict, evaluation: Dict) -> str:
        """Build prompt for LLM adjustment consultation."""
        template = self.config_manifest.get('llm_prompt_template', '')
        
        prompt = f"""
You are an AI agent assistant for a PRNG analysis pipeline.

## Current Step
Step {self.step}: {self.agent_manifest['name']}

## Config Manifest (What Can Be Tuned)
{json.dumps(self.config_manifest.get('parameters', {}), indent=2)}

## Current Configuration
{json.dumps(self.current_config, indent=2)}

## Execution Results
{json.dumps(results, indent=2, default=str)}

## Evaluation (FAILED)
Failures: {evaluation['failures']}
Metrics: {json.dumps(evaluation['metrics'], indent=2)}

## Adjustment Rules
{json.dumps(self.config_manifest.get('adjustment_rules', []), indent=2)}

## Question
{template if template else 'Based on the failures and adjustment rules, what parameters should be modified?'}

## Response Format
Respond with ONLY valid JSON in this exact format:
{{
  "action": "modify_config" | "proceed" | "escalate",
  "changes": {{"parameter_name": new_value, ...}},
  "reasoning": "Brief explanation of your decision"
}}
"""
        return prompt
    
    def _escalate(self, results: Dict, evaluation: Dict, reason: str) -> Dict[str, Any]:
        """Handle escalation to human review."""
        self.logger.error(f"ESCALATING Step {self.step}: {reason}")
        
        results['escalated'] = True
        results['escalation_reason'] = reason
        results['evaluation'] = evaluation
        
        return self.finalize(results, evaluation)
```

---

## Part 3: Agent Manifests

### 3.1 Step 1 - Window Optimizer

```json
{
  "step": 1,
  "name": "window_optimizer",
  "description": "Bayesian window optimization with bidirectional forward/reverse sieve",
  "execution_mode": "seed_based",
  
  "script": "window_optimizer.py",
  "config_file": "configs/window_config.json",
  "config_manifest": "config_manifests/step1_window_optimizer.json",
  
  "inputs": [
    {"file": "lottery_history.json", "required": true},
    {"file": "prng_registry.py", "required": true}
  ],
  
  "outputs": [
    "bidirectional_survivors.json",
    "forward_survivors.json",
    "reverse_survivors.json",
    "optimal_window_config.json",
    "train_history.json",
    "holdout_history.json"
  ],
  
  "follow_up_agent": "step2_scorer_meta",
  "retry": 3
}
```

### 3.2 Step 2/2.5 - Scorer Meta Optimizer

```json
{
  "step": 2,
  "name": "scorer_meta_optimizer",
  "description": "Distributed Optuna meta-optimization for scorer parameters (26-GPU PULL architecture)",
  "execution_mode": "script_based",
  
  "script": "generate_scorer_jobs.py",
  "worker_script": "scorer_trial_worker.py",
  "config_file": "configs/scorer_config.json",
  "config_manifest": "config_manifests/step2_scorer_meta.json",
  
  "inputs": [
    {"file": "bidirectional_survivors.json", "required": true},
    {"file": "train_history.json", "required": true},
    {"file": "holdout_history.json", "required": true}
  ],
  
  "outputs": [
    "optimal_scorer_config.json",
    "aggregated_scorer_results.json",
    "optuna_studies/scorer_meta_opt_*.db"
  ],
  
  "follow_up_agent": "step3_full_scoring",
  "retry": 2
}
```

### 3.3 Step 3 - Full Scoring

```json
{
  "step": 3,
  "name": "full_scoring",
  "description": "Score all survivors using optimized parameters (26-GPU distributed)",
  "execution_mode": "script_based",
  
  "script": "generate_full_scoring_jobs.py",
  "worker_script": "survivor_scorer.py",
  "config_file": "configs/scorer_config.json",
  "config_manifest": "config_manifests/step3_full_scoring.json",
  
  "inputs": [
    {"file": "optimal_scorer_config.json", "required": true},
    {"file": "bidirectional_survivors.json", "required": true},
    {"file": "train_history.json", "required": true}
  ],
  
  "outputs": [
    "survivors_with_scores.json"
  ],
  
  "follow_up_agent": "step4_ml_meta",
  "retry": 2
}
```

### 3.4 Step 4 - ML Meta Optimizer

```json
{
  "step": 4,
  "name": "ml_meta_optimizer",
  "description": "Adaptive ML architecture optimization based on survivor distribution",
  "execution_mode": "script_based",
  
  "script": "adaptive_meta_optimizer.py",
  "config_file": "configs/ml_config.json",
  "config_manifest": "config_manifests/step4_ml_meta.json",
  
  "inputs": [
    {"file": "survivors_with_scores.json", "required": true}
  ],
  
  "outputs": [
    "reinforcement_engine_config.json",
    "best_model_architecture.json"
  ],
  
  "follow_up_agent": "step5_anti_overfit",
  "retry": 2
}
```

### 3.5 Step 5 - Anti-Overfit Training

```json
{
  "step": 5,
  "name": "anti_overfit_training",
  "description": "K-fold cross-validated model training with anti-overfitting measures (26-GPU)",
  "execution_mode": "script_based",
  
  "script": "meta_prediction_optimizer_anti_overfit.py",
  "config_file": "configs/training_config.json",
  "config_manifest": "config_manifests/step5_anti_overfit.json",
  
  "inputs": [
    {"file": "survivors_with_scores.json", "required": true},
    {"file": "reinforcement_engine_config.json", "required": true},
    {"file": "train_history.json", "required": true}
  ],
  
  "outputs": [
    "models/anti_overfit/best_model.pth",
    "training_metrics.json"
  ],
  
  "follow_up_agent": "step6_prediction",
  "retry": 2
}
```

### 3.6 Step 6 - Prediction

```json
{
  "step": 6,
  "name": "prediction",
  "description": "Generate prediction pools using trained model",
  "execution_mode": "local",
  
  "script": "prediction_generator.py",
  "config_file": "configs/prediction_config.json",
  "config_manifest": "config_manifests/step6_prediction.json",
  
  "inputs": [
    {"file": "models/anti_overfit/best_model.pth", "required": true},
    {"file": "survivors_with_scores.json", "required": true},
    {"file": "holdout_history.json", "required": true}
  ],
  
  "outputs": [
    "prediction_pool.json",
    "prediction_confidence_map.json"
  ],
  
  "follow_up_agent": null,
  "retry": 1
}
```

---

## Part 4: Config Manifests

### 4.1 Step 1 - Window Optimizer Config Manifest

```json
{
  "step": 1,
  "name": "window_optimizer",
  
  "parameters": {
    "window_size": {
      "type": "int",
      "range": [512, 4096],
      "default": 1024,
      "description": "Number of draws to analyze in each window",
      "effect": "Larger windows = more survivors but slower execution"
    },
    "offset": {
      "type": "int",
      "range": [0, 1000],
      "default": 0,
      "description": "Starting offset in draw history",
      "effect": "Non-zero offset skips early draws"
    },
    "skip_min": {
      "type": "int",
      "range": [0, 100],
      "default": 0,
      "description": "Minimum skip value to test",
      "effect": "Lower bound of skip search space"
    },
    "skip_max": {
      "type": "int",
      "range": [1, 200],
      "default": 50,
      "description": "Maximum skip value to test",
      "effect": "Upper bound of skip search space"
    },
    "threshold": {
      "type": "float",
      "range": [0.05, 0.5],
      "default": 0.15,
      "description": "Minimum match rate to keep survivor",
      "effect": "Lower = more lenient filtering, more survivors"
    },
    "prng_type": {
      "type": "categorical",
      "options": [
        "java_lcg", "java_lcg_hybrid",
        "mt19937", "mt19937_hybrid",
        "xorshift32", "xorshift32_hybrid",
        "xorshift64", "xorshift64_hybrid",
        "pcg32", "pcg32_hybrid",
        "lcg32", "lcg32_hybrid"
      ],
      "default": "java_lcg",
      "description": "PRNG algorithm to test",
      "effect": "Different PRNGs have different state spaces"
    },
    "test_both_modes": {
      "type": "bool",
      "default": true,
      "description": "Test both constant and variable skip modes",
      "effect": "True = more comprehensive but 2x slower"
    },
    "trials": {
      "type": "int",
      "range": [10, 200],
      "default": 50,
      "description": "Number of Optuna optimization trials",
      "effect": "More trials = better optimization but slower"
    },
    "max_seeds": {
      "type": "int",
      "range": [1000000, 1000000000],
      "default": 10000000,
      "description": "Maximum seeds to search",
      "effect": "More seeds = better coverage but slower"
    }
  },
  
  "success_criteria": {
    "min_survivors": 100,
    "min_bidirectional_survivors": 50,
    "min_match_rate": 0.1,
    "max_runtime_minutes": 120
  },
  
  "adjustment_rules": [
    {
      "condition": "survivors < 50",
      "action": "increase window_size by 50%, decrease threshold by 25%",
      "reasoning": "Too few survivors indicates overly strict filtering"
    },
    {
      "condition": "survivors > 50000",
      "action": "decrease window_size by 25%, increase threshold by 25%",
      "reasoning": "Too many survivors indicates insufficient filtering"
    },
    {
      "condition": "bidirectional_survivors < 10",
      "action": "increase skip_max by 50%",
      "reasoning": "Few bidirectional matches suggests skip range too narrow"
    }
  ],
  
  "llm_prompt_template": "Given {survivors} total survivors and {bidirectional_survivors} bidirectional survivors with match_rate={match_rate:.3f} using window_size={window_size} and threshold={threshold}, should parameters be adjusted?"
}
```

### 4.2 Step 2/2.5 - Scorer Meta Config Manifest

```json
{
  "step": 2,
  "name": "scorer_meta_optimizer",
  
  "parameters": {
    "residue_mod_1": {
      "type": "int",
      "range": [3, 100],
      "default": 15,
      "description": "First residue modulus for pattern analysis",
      "effect": "Affects residue lane distribution detection"
    },
    "residue_mod_2": {
      "type": "int",
      "range": [3, 100],
      "default": 67,
      "description": "Second residue modulus for pattern analysis",
      "effect": "Different mods capture different periodicity"
    },
    "temporal_window_size": {
      "type": "int",
      "range": [50, 500],
      "default": 150,
      "description": "Window size for temporal pattern analysis",
      "effect": "Larger windows detect longer-term patterns"
    },
    "n_trials": {
      "type": "int",
      "range": [50, 500],
      "default": 100,
      "description": "Number of Optuna trials for meta-optimization",
      "effect": "More trials = better parameter search"
    },
    "timeout_minutes": {
      "type": "int",
      "range": [30, 480],
      "default": 120,
      "description": "Maximum optimization time",
      "effect": "Limits total runtime"
    }
  },
  
  "success_criteria": {
    "min_accuracy": 0.6,
    "min_completed_trials": 50,
    "max_failed_trials_pct": 0.1
  },
  
  "adjustment_rules": [
    {
      "condition": "accuracy < 0.5",
      "action": "increase n_trials by 50%",
      "reasoning": "Low accuracy may need more exploration"
    },
    {
      "condition": "failed_trials_pct > 0.2",
      "action": "increase timeout_minutes by 50%",
      "reasoning": "Many failures suggest timeout too aggressive"
    }
  ],
  
  "llm_prompt_template": "Scorer optimization achieved accuracy={accuracy:.3f} with {completed_trials} completed trials. Best params: residue_mod_1={residue_mod_1}, residue_mod_2={residue_mod_2}. Should we adjust the search space?"
}
```

### 4.3 Step 5 - Anti-Overfit Config Manifest

```json
{
  "step": 5,
  "name": "anti_overfit_training",
  
  "parameters": {
    "n_layers": {
      "type": "int",
      "range": [2, 4],
      "default": 3,
      "description": "Number of hidden layers in neural network",
      "effect": "More layers = more capacity but risk of overfitting"
    },
    "hidden_size_max": {
      "type": "int",
      "range": [64, 256],
      "default": 128,
      "description": "Maximum hidden layer size",
      "effect": "Larger layers = more capacity"
    },
    "dropout": {
      "type": "float",
      "range": [0.2, 0.5],
      "default": 0.3,
      "description": "Dropout rate for regularization",
      "effect": "Higher dropout = more regularization"
    },
    "learning_rate": {
      "type": "float",
      "range": [0.00001, 0.001],
      "default": 0.0001,
      "log_scale": true,
      "description": "Learning rate for optimizer",
      "effect": "Lower LR = slower but more stable training"
    },
    "batch_size": {
      "type": "categorical",
      "options": [64, 128, 256],
      "default": 128,
      "description": "Training batch size",
      "effect": "Larger batches = more stable gradients"
    },
    "epochs": {
      "type": "int",
      "range": [50, 150],
      "default": 100,
      "description": "Maximum training epochs",
      "effect": "More epochs = longer training"
    },
    "early_stopping_patience": {
      "type": "int",
      "range": [5, 15],
      "default": 10,
      "description": "Epochs to wait before early stopping",
      "effect": "Higher patience = more training before stopping"
    },
    "k_folds": {
      "type": "int",
      "range": [3, 10],
      "default": 5,
      "description": "Number of cross-validation folds",
      "effect": "More folds = more robust but slower"
    },
    "weight_decay": {
      "type": "float",
      "range": [0.00001, 0.01],
      "default": 0.0001,
      "log_scale": true,
      "description": "L2 regularization strength",
      "effect": "Higher = more regularization"
    },
    "optimizer": {
      "type": "categorical",
      "options": ["adam", "adamw"],
      "default": "adamw",
      "description": "Optimizer algorithm",
      "effect": "AdamW has built-in weight decay"
    },
    "loss": {
      "type": "categorical",
      "options": ["mse", "huber"],
      "default": "huber",
      "description": "Loss function",
      "effect": "Huber is more robust to outliers"
    }
  },
  
  "success_criteria": {
    "min_val_score": 0.6,
    "max_overfit_ratio": 1.5,
    "min_variance_consistency": 0.7,
    "max_p_value": 0.05
  },
  
  "adjustment_rules": [
    {
      "condition": "overfit_ratio > 1.5",
      "action": "increase dropout by 0.1, increase weight_decay by 10x",
      "reasoning": "High overfit ratio indicates need for more regularization"
    },
    {
      "condition": "val_score < 0.5",
      "action": "increase hidden_size_max by 50%, increase epochs by 25%",
      "reasoning": "Low validation score suggests underfitting"
    },
    {
      "condition": "variance_consistency < 0.6",
      "action": "increase k_folds by 2",
      "reasoning": "Low consistency suggests unstable training"
    }
  ],
  
  "llm_prompt_template": "Training achieved val_score={val_score:.3f} with overfit_ratio={overfit_ratio:.2f}. Architecture: {n_layers} layers, dropout={dropout}, lr={learning_rate}. Is the model overfitting or underfitting?"
}
```

### 4.4 Step 6 - Prediction Config Manifest

```json
{
  "step": 6,
  "name": "prediction",
  
  "parameters": {
    "pool_size": {
      "type": "int",
      "range": [1, 20],
      "default": 5,
      "description": "Number of predictions to generate",
      "effect": "Larger pool = more options but lower individual confidence"
    },
    "confidence_threshold": {
      "type": "float",
      "range": [0.5, 0.95],
      "default": 0.7,
      "description": "Minimum confidence to include in pool",
      "effect": "Higher threshold = fewer but more confident predictions"
    },
    "use_ensemble": {
      "type": "bool",
      "default": true,
      "description": "Use ensemble of top models",
      "effect": "Ensemble improves robustness"
    },
    "temporal_weight": {
      "type": "float",
      "range": [0.0, 1.0],
      "default": 0.3,
      "description": "Weight for temporal pattern features",
      "effect": "Higher = more emphasis on recent patterns"
    }
  },
  
  "success_criteria": {
    "min_pool_size": 3,
    "min_average_confidence": 0.6
  },
  
  "adjustment_rules": [
    {
      "condition": "pool_size < 3",
      "action": "decrease confidence_threshold by 0.1",
      "reasoning": "Too few predictions, need to lower threshold"
    }
  ],
  
  "llm_prompt_template": "Generated {pool_size} predictions with average confidence {avg_confidence:.2f}. Top prediction: {top_prediction} (confidence: {top_confidence:.2f}). Is this prediction pool adequate?"
}
```

---

## Part 5: Implementation Plan

### Phase 1: Infrastructure (Day 1)

| Task | Time | Output |
|------|------|--------|
| Create `agents/` directory structure | 15 min | Directory tree |
| Implement `base_agent.py` | 2 hours | Base class |
| Implement `agent_manager.py` | 1 hour | Orchestration |
| Create all 6 agent manifests | 30 min | 6 JSON files |
| Create all 6 config manifests | 1 hour | 6 JSON files |

### Phase 2: Agent Implementations (Day 2)

| Task | Time | Output |
|------|------|--------|
| `step1_window_agent.py` (seed-based) | 1 hour | Step 1 agent |
| `step2_sieve_agent.py` (seed-based) | 30 min | Step 2 agent |
| `step3_scorer_agent.py` (script-based) | 30 min | Step 3 agent |
| `step4_ml_agent.py` (script-based) | 30 min | Step 4 agent |
| `step5_training_agent.py` (script-based) | 30 min | Step 5 agent |
| `step6_predict_agent.py` (local) | 30 min | Step 6 agent |

### Phase 3: Testing & Integration (Day 3)

| Task | Time | Output |
|------|------|--------|
| Unit tests for base_agent | 1 hour | Test suite |
| Integration test: Step 1 → 2 | 1 hour | Verified handoff |
| Integration test: Full pipeline | 2 hours | End-to-end test |
| Documentation | 1 hour | README, examples |

---

## Part 6: Compatibility Verification

### Original Proposal Compliance

| Original Rule | This Proposal | ✓/✗ |
|---------------|---------------|-----|
| Agents don't run code directly | `execute()` delegates to scripts/coordinator | ✅ |
| Agents are declarative (JSON) | `agent_manifests/` + `config_manifests/` | ✅ |
| Agents evaluate success | `evaluate()` method + `success_criteria` | ✅ |
| Agents choose next agent | `trigger_next()` + `follow_up_agent` | ✅ |

### Schema v1.0.3 Compliance

| Field | Supported | ✓/✗ |
|-------|-----------|-----|
| `inputs` | Via `inject_agent_metadata()` | ✅ |
| `outputs` | Via `inject_agent_metadata()` | ✅ |
| `parent_run_id` | Passed through `run()` | ✅ |
| `pipeline_step` | From agent step number | ✅ |
| `follow_up_agent` | From manifest | ✅ |
| `confidence` | Calculated in `evaluate()` | ✅ |
| `success_criteria_met` | From `evaluate()` | ✅ |
| `suggested_params` | Current config on success | ✅ |

### Dual-LLM Integration

| Feature | Implementation | ✓/✗ |
|---------|----------------|-----|
| LLM Router | `get_router()` in `__init__` | ✅ |
| Orchestrator queries | `self.llm.orchestrate()` in `adjust()` | ✅ |
| Math queries | Available via `self.llm.calculate()` | ✅ |
| LLM metadata | Captured in `finalize()` | ✅ |

---

## Appendix A: Example Agent Implementation

### step3_scorer_agent.py (Script-Based Example)

```python
#!/usr/bin/env python3
"""
Step 3 Agent: Full Scoring (Script-Based)
=========================================

Scores all survivors using optimized parameters via 26-GPU coordinator.
"""

from agents.base_agent import BaseAgent
from coordinator import MultiGPUCoordinator
import json


class FullScoringAgent(BaseAgent):
    """Step 3: Distributed full scoring agent."""
    
    def __init__(self, config_override=None):
        super().__init__(step=3, config_override=config_override)
        self.coordinator = None
    
    def execute(self):
        """
        Execute full scoring via distributed coordinator.
        
        This is the ONLY method that differs from base class.
        """
        # Initialize coordinator
        self.coordinator = MultiGPUCoordinator()
        
        # Load optimal scorer config from Step 2
        with open('optimal_scorer_config.json') as f:
            scorer_params = json.load(f)
        
        # Generate scoring jobs
        jobs = self._generate_scoring_jobs(scorer_params)
        
        self.logger.info(f"Generated {len(jobs)} scoring jobs")
        
        # Execute via coordinator (26-GPU cluster)
        job_results = self.coordinator.execute_jobs(
            jobs=jobs,
            timeout=self.current_config.get('timeout_minutes', 120) * 60
        )
        
        # Aggregate results
        results = self._aggregate_results(job_results)
        
        return results
    
    def _generate_scoring_jobs(self, scorer_params):
        """Generate distributed scoring jobs."""
        # Load survivors
        with open('bidirectional_survivors.json') as f:
            survivors = json.load(f)
        
        # Split across 26 GPUs
        chunk_size = len(survivors) // 26 + 1
        jobs = []
        
        for i, chunk_start in enumerate(range(0, len(survivors), chunk_size)):
            chunk = survivors[chunk_start:chunk_start + chunk_size]
            
            jobs.append({
                'job_id': f'scoring_chunk_{i}',
                'script': 'survivor_scorer.py',
                'args': [
                    f'chunk_{i}.json',
                    'train_history.json',
                    json.dumps(scorer_params)
                ],
                'chunk_data': chunk
            })
        
        return jobs
    
    def _aggregate_results(self, job_results):
        """Aggregate results from all GPU jobs."""
        all_scores = []
        failed_jobs = 0
        
        for result in job_results:
            if result.get('success'):
                all_scores.extend(result.get('scores', []))
            else:
                failed_jobs += 1
        
        return {
            'total_scored': len(all_scores),
            'failed_jobs': failed_jobs,
            'scores': all_scores,
            'average_score': sum(s['score'] for s in all_scores) / len(all_scores) if all_scores else 0
        }


# Entry point
if __name__ == '__main__':
    agent = FullScoringAgent()
    result = agent.run()
    print(json.dumps(result, indent=2, default=str))
```

---

## Approval

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Author | Claude (AI) | 2025-12-02 | ✓ |
| Technical Review | | | |
| Team Lead | | | |
| Final Approval | | | |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-02 | Initial proposal with full architecture |

---

**End of Proposal**
