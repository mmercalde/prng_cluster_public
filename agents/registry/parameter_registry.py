#!/usr/bin/env python3
"""
Parameter Registry - Script parameter bounds for AI agents.

Enables LLM introspection of legal parameter bounds, allowing
AI agents to make informed decisions about parameter adjustments.

Version: 3.2.0
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class ParamType(str, Enum):
    """Parameter data types."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOL = "bool"
    LIST = "list"
    CHOICE = "choice"


class ParameterSpec(BaseModel):
    """
    Specification for a single configurable parameter.
    
    Defines the name, type, bounds, and description of a parameter
    that can be adjusted by AI agents.
    """
    
    name: str = Field(
        ...,
        description="Parameter name (matches CLI arg without dashes)"
    )
    
    param_type: ParamType = Field(
        default=ParamType.STRING,
        description="Data type of the parameter"
    )
    
    description: str = Field(
        default="",
        description="Human-readable description"
    )
    
    # Bounds for numeric types
    min_value: Optional[float] = Field(
        default=None,
        description="Minimum allowed value (for int/float)"
    )
    
    max_value: Optional[float] = Field(
        default=None,
        description="Maximum allowed value (for int/float)"
    )
    
    # Choices for enum types
    choices: List[Any] = Field(
        default_factory=list,
        description="Allowed values (for choice type)"
    )
    
    # Default value
    default: Optional[Any] = Field(
        default=None,
        description="Default value if not specified"
    )
    
    # Metadata
    required: bool = Field(
        default=False,
        description="Whether this parameter is required"
    )
    
    affects_runtime: bool = Field(
        default=False,
        description="Whether changing this affects runtime significantly"
    )
    
    cli_flag: str = Field(
        default="",
        description="CLI flag format (e.g., '--trials')"
    )
    
    def validate_value(self, value: Any) -> tuple[bool, Optional[str]]:
        """
        Check if value is within legal bounds.
        
        Args:
            value: Value to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if value is None and not self.required:
            return True, None
        
        if value is None and self.required:
            return False, f"{self.name} is required"
        
        if self.param_type == ParamType.INT:
            if not isinstance(value, int):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    return False, f"{self.name} must be an integer"
            
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.name}={value} below minimum {self.min_value}"
            
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.name}={value} above maximum {self.max_value}"
        
        elif self.param_type == ParamType.FLOAT:
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    return False, f"{self.name} must be a number"
            
            if self.min_value is not None and value < self.min_value:
                return False, f"{self.name}={value} below minimum {self.min_value}"
            
            if self.max_value is not None and value > self.max_value:
                return False, f"{self.name}={value} above maximum {self.max_value}"
        
        elif self.param_type == ParamType.CHOICE:
            if self.choices and value not in self.choices:
                return False, f"{self.name}={value} not in allowed choices {self.choices}"
        
        elif self.param_type == ParamType.BOOL:
            if not isinstance(value, bool):
                if str(value).lower() not in ('true', 'false', '0', '1', 'yes', 'no'):
                    return False, f"{self.name} must be a boolean"
        
        return True, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Return parameter spec as clean dict for LLM context."""
        result = {
            "name": self.name,
            "type": self.param_type.value,
        }
        
        if self.cli_flag:
            result["cli_flag"] = self.cli_flag
        
        if self.min_value is not None:
            result["min"] = self.min_value
        
        if self.max_value is not None:
            result["max"] = self.max_value
        
        if self.choices:
            result["choices"] = self.choices
        
        if self.default is not None:
            result["default"] = self.default
        
        if self.description:
            result["description"] = self.description
        
        if self.affects_runtime:
            result["affects_runtime"] = True
        
        return result
    
    def to_prompt_line(self) -> str:
        """DEPRECATED: Use to_dict() instead."""
        import json
        return json.dumps(self.to_dict())


class ScriptParameterRegistry(BaseModel):
    """
    Registry of parameters for a specific script.
    
    Contains all configurable parameters that an AI agent
    can adjust when recommending retries.
    """
    
    script_name: str = Field(
        ...,
        description="Name of the script file"
    )
    
    description: str = Field(
        default="",
        description="Description of what the script does"
    )
    
    parameters: List[ParameterSpec] = Field(
        default_factory=list,
        description="List of configurable parameters"
    )
    
    def get_param(self, name: str) -> Optional[ParameterSpec]:
        """
        Get parameter spec by name.
        
        Args:
            name: Parameter name (with or without dashes)
            
        Returns:
            ParameterSpec or None
        """
        # Normalize name
        clean_name = name.lstrip('-').replace('-', '_')
        
        for p in self.parameters:
            p_clean = p.name.lstrip('-').replace('-', '_')
            if p_clean == clean_name:
                return p
        
        return None
    
    def validate_adjustments(self, adjustments: Dict[str, Any]) -> List[str]:
        """
        Validate a set of parameter adjustments.
        
        Args:
            adjustments: Dict of param_name -> new_value
            
        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        
        for name, value in adjustments.items():
            spec = self.get_param(name)
            
            if spec is None:
                errors.append(f"Unknown parameter: {name}")
                continue
            
            is_valid, error = spec.validate_value(value)
            if not is_valid:
                errors.append(error)
        
        return errors
    
    def to_context_dict(self) -> Dict[str, Any]:
        """Return registry as clean dict for LLM context."""
        return {
            "script": self.script_name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters]
        }
    
    def to_prompt_section(self) -> str:
        """DEPRECATED: Use to_context_dict() instead."""
        import json
        return json.dumps(self.to_context_dict(), indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# PRE-DEFINED REGISTRIES FOR ALL PIPELINE SCRIPTS
# ════════════════════════════════════════════════════════════════════════════════

PARAMETER_REGISTRIES: Dict[str, ScriptParameterRegistry] = {
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Window Optimizer
    # ══════════════════════════════════════════════════════════════════════
    "window_optimizer.py": ScriptParameterRegistry(
        script_name="window_optimizer.py",
        description="Bayesian window optimization for bidirectional sieve analysis",
        parameters=[
            ParameterSpec(
                name="window_size",
                cli_flag="--window-size",
                param_type=ParamType.INT,
                description="Number of lottery draws to analyze",
                min_value=50,
                max_value=2000,
                default=512
            ),
            ParameterSpec(
                name="offset",
                cli_flag="--offset",
                param_type=ParamType.INT,
                description="Starting position in dataset",
                min_value=0,
                max_value=500,
                default=0
            ),
            ParameterSpec(
                name="skip_min",
                cli_flag="--skip-min",
                param_type=ParamType.INT,
                description="Minimum PRNG skip value to test",
                min_value=0,
                max_value=50,
                default=0
            ),
            ParameterSpec(
                name="skip_max",
                cli_flag="--skip-max",
                param_type=ParamType.INT,
                description="Maximum PRNG skip value to test",
                min_value=1,
                max_value=100,
                default=20
            ),
            ParameterSpec(
                name="trials",
                cli_flag="--trials",
                param_type=ParamType.INT,
                description="Number of Optuna optimization trials",
                min_value=3,
                max_value=500,
                default=50,
                affects_runtime=True
            ),
            ParameterSpec(
                name="max_seeds",
                cli_flag="--max-seeds",
                param_type=ParamType.INT,
                description="Maximum seeds to test per trial",
                min_value=100000,
                max_value=10000000000,
                default=10000000,
                affects_runtime=True
            ),
            ParameterSpec(
                name="prng_type",
                cli_flag="--prng-type",
                param_type=ParamType.CHOICE,
                description="PRNG algorithm to analyze",
                choices=["java_lcg", "mt19937", "xorshift32", "xorshift64", "pcg32", "lcg64"],
                default="java_lcg"
            ),
            ParameterSpec(
                name="strategy",
                cli_flag="--strategy",
                param_type=ParamType.CHOICE,
                description="Optimization search strategy",
                choices=["bayesian", "grid", "random", "evolutionary"],
                default="bayesian"
            ),
            ParameterSpec(
                name="test_both_modes",
                cli_flag="--test-both-modes",
                param_type=ParamType.BOOL,
                description="Test both constant and variable skip modes",
                default=False,
                affects_runtime=True
            ),
        ]
    ),
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Scorer Meta Optimizer
    # ══════════════════════════════════════════════════════════════════════
    "scorer_meta_optimizer.py": ScriptParameterRegistry(
        script_name="scorer_meta_optimizer.py",
        description="Distributed Optuna optimization for scoring parameters",
        parameters=[
            ParameterSpec(
                name="n_trials",
                cli_flag="--n-trials",
                param_type=ParamType.INT,
                description="Number of Optuna trials",
                min_value=10,
                max_value=500,
                default=100,
                affects_runtime=True
            ),
            ParameterSpec(
                name="threshold_min",
                cli_flag="--threshold-min",
                param_type=ParamType.FLOAT,
                description="Minimum threshold to search",
                min_value=0.001,
                max_value=0.05,
                default=0.005
            ),
            ParameterSpec(
                name="threshold_max",
                cli_flag="--threshold-max",
                param_type=ParamType.FLOAT,
                description="Maximum threshold to search",
                min_value=0.01,
                max_value=0.2,
                default=0.1
            ),
            ParameterSpec(
                name="k_folds",
                cli_flag="--k-folds",
                param_type=ParamType.CHOICE,
                description="Cross-validation folds",
                choices=[3, 5, 7, 10],
                default=5
            ),
            ParameterSpec(
                name="distributed",
                cli_flag="--distributed",
                param_type=ParamType.BOOL,
                description="Run in distributed mode across cluster",
                default=True
            ),
        ]
    ),
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Full Scoring
    # ══════════════════════════════════════════════════════════════════════
    "full_scoring.py": ScriptParameterRegistry(
        script_name="full_scoring.py",
        description="Score all survivors using optimized parameters",
        parameters=[
            ParameterSpec(
                name="batch_size",
                cli_flag="--batch-size",
                param_type=ParamType.INT,
                description="Batch size for GPU processing",
                min_value=100,
                max_value=10000,
                default=1000
            ),
            ParameterSpec(
                name="distributed",
                cli_flag="--distributed",
                param_type=ParamType.BOOL,
                description="Run in distributed mode",
                default=True
            ),
        ]
    ),
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: ML Meta Optimizer / Adaptive Meta Optimizer
    # ══════════════════════════════════════════════════════════════════════
    "adaptive_meta_optimizer.py": ScriptParameterRegistry(
        script_name="adaptive_meta_optimizer.py",
        description="Optimize ML model architecture using Optuna",
        parameters=[
            ParameterSpec(
                name="trials",
                cli_flag="--trials",
                param_type=ParamType.INT,
                description="Architecture search trials",
                min_value=10,
                max_value=200,
                default=50,
                affects_runtime=True
            ),
            ParameterSpec(
                name="min_layers",
                cli_flag="--min-layers",
                param_type=ParamType.INT,
                description="Minimum hidden layers",
                min_value=1,
                max_value=4,
                default=2
            ),
            ParameterSpec(
                name="max_layers",
                cli_flag="--max-layers",
                param_type=ParamType.INT,
                description="Maximum hidden layers",
                min_value=2,
                max_value=8,
                default=5
            ),
            ParameterSpec(
                name="mode",
                cli_flag="--mode",
                param_type=ParamType.CHOICE,
                description="Optimization mode",
                choices=["quick", "full", "deep"],
                default="full"
            ),
        ]
    ),
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 5: Anti-Overfit Training
    # ══════════════════════════════════════════════════════════════════════
    "meta_prediction_optimizer_anti_overfit.py": ScriptParameterRegistry(
        script_name="meta_prediction_optimizer_anti_overfit.py",
        description="K-fold anti-overfit model training",
        parameters=[
            ParameterSpec(
                name="trials",
                cli_flag="--trials",
                param_type=ParamType.INT,
                description="Optuna trials for training optimization",
                min_value=10,
                max_value=200,
                default=50,
                affects_runtime=True
            ),
            ParameterSpec(
                name="k_folds",
                cli_flag="--k-folds",
                param_type=ParamType.INT,
                description="Number of cross-validation folds",
                min_value=3,
                max_value=10,
                default=5
            ),
            ParameterSpec(
                name="epochs",
                cli_flag="--epochs",
                param_type=ParamType.INT,
                description="Training epochs per fold",
                min_value=10,
                max_value=500,
                default=100,
                affects_runtime=True
            ),
            ParameterSpec(
                name="dropout_min",
                cli_flag="--dropout-min",
                param_type=ParamType.FLOAT,
                description="Minimum dropout rate to search",
                min_value=0.0,
                max_value=0.5,
                default=0.1
            ),
            ParameterSpec(
                name="dropout_max",
                cli_flag="--dropout-max",
                param_type=ParamType.FLOAT,
                description="Maximum dropout rate to search",
                min_value=0.2,
                max_value=0.7,
                default=0.5
            ),
            ParameterSpec(
                name="distributed",
                cli_flag="--distributed",
                param_type=ParamType.BOOL,
                description="Run in distributed mode across cluster",
                default=True
            ),
        ]
    ),
    
    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: Prediction
    # ══════════════════════════════════════════════════════════════════════
    "prediction_generator.py": ScriptParameterRegistry(
        script_name="prediction_generator.py",
        description="Generate prediction pool from trained model",
        parameters=[
            ParameterSpec(
                name="pool_size",
                cli_flag="--pool-size",
                param_type=ParamType.INT,
                description="Number of predictions to generate",
                min_value=10,
                max_value=1000,
                default=200
            ),
            ParameterSpec(
                name="confidence_threshold",
                cli_flag="--confidence-threshold",
                param_type=ParamType.FLOAT,
                description="Minimum confidence for inclusion",
                min_value=0.0,
                max_value=1.0,
                default=0.5
            ),
            ParameterSpec(
                name="diversity_weight",
                cli_flag="--diversity-weight",
                param_type=ParamType.FLOAT,
                description="Weight for prediction diversity",
                min_value=0.0,
                max_value=1.0,
                default=0.3
            ),
        ]
    ),
}


# ════════════════════════════════════════════════════════════════════════════════
# LOOKUP FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def get_registry(script_name: str) -> Optional[ScriptParameterRegistry]:
    """
    Get parameter registry for a script.
    
    Args:
        script_name: Name of the script (with or without .py)
        
    Returns:
        ScriptParameterRegistry or None
    """
    # Normalize name
    if not script_name.endswith('.py'):
        script_name = f"{script_name}.py"
    
    return PARAMETER_REGISTRIES.get(script_name)


def get_registry_for_agent(agent_name: str) -> Optional[ScriptParameterRegistry]:
    """
    Get parameter registry based on agent name.
    
    Maps agent names to their corresponding scripts.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        ScriptParameterRegistry or None
    """
    # Normalize
    agent_name = agent_name.lower().strip()
    
    # Direct mapping
    agent_to_script = {
        # Step 1
        "window_optimizer_agent": "window_optimizer.py",
        "window_optimizer": "window_optimizer.py",
        
        # Step 2
        "scorer_meta_agent": "scorer_meta_optimizer.py",
        "scorer_meta_optimizer": "scorer_meta_optimizer.py",
        "scorer_meta": "scorer_meta_optimizer.py",
        
        # Step 3
        "full_scoring_agent": "full_scoring.py",
        "full_scoring": "full_scoring.py",
        
        # Step 4
        "ml_meta_agent": "adaptive_meta_optimizer.py",
        "ml_meta_optimizer": "adaptive_meta_optimizer.py",
        "ml_meta": "adaptive_meta_optimizer.py",
        "adaptive_meta_optimizer": "adaptive_meta_optimizer.py",
        
        # Step 5
        "anti_overfit_agent": "meta_prediction_optimizer_anti_overfit.py",
        "reinforcement_agent": "meta_prediction_optimizer_anti_overfit.py",
        "anti_overfit": "meta_prediction_optimizer_anti_overfit.py",
        
        # Step 6
        "prediction_agent": "prediction_generator.py",
        "prediction": "prediction_generator.py",
    }
    
    script = agent_to_script.get(agent_name)
    if script:
        return get_registry(script)
    
    # Fuzzy match
    for key, script in agent_to_script.items():
        if key in agent_name or agent_name in key:
            return get_registry(script)
    
    return None


def list_all_registries() -> List[str]:
    """Get list of all registered script names."""
    return list(PARAMETER_REGISTRIES.keys())
