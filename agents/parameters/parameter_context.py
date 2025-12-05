#!/usr/bin/env python3
"""
Parameter Context - Builds complete parameter awareness for AI agents.

Combines:
1. Manifest args_map (what can be adjusted)
2. Registry bounds (legal values)
3. Current values (what was used)
4. Suggested adjustments (for retry scenarios)

This is the key component that enables AI agents to understand
what parameters they can modify and within what bounds.

Version: 3.2.0
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.manifest.agent_manifest import AgentManifest

from agents.registry.parameter_registry import (
    get_registry,
    ScriptParameterRegistry,
    ParameterSpec,
    ParamType
)


class ParameterInfo(BaseModel):
    """
    Complete information about a single adjustable parameter.
    
    This combines information from:
    - Manifest (what it's called, CLI flag)
    - Registry (type, bounds, description)
    - Current run (what value was used)
    """
    
    # Identity
    name: str = Field(
        ...,
        description="Normalized parameter name"
    )
    
    cli_flag: str = Field(
        default="",
        description="CLI flag format (e.g., '--trials')"
    )
    
    context_var: str = Field(
        default="",
        description="Context variable name from manifest"
    )
    
    # From registry
    param_type: str = Field(
        default="unknown",
        description="Data type (int, float, choice, etc.)"
    )
    
    description: str = Field(
        default="",
        description="Human-readable description"
    )
    
    min_value: Optional[float] = Field(
        default=None,
        description="Minimum allowed value"
    )
    
    max_value: Optional[float] = Field(
        default=None,
        description="Maximum allowed value"
    )
    
    choices: List[Any] = Field(
        default_factory=list,
        description="Allowed values for choice type"
    )
    
    default: Optional[Any] = Field(
        default=None,
        description="Default value"
    )
    
    # Current run
    current_value: Optional[Any] = Field(
        default=None,
        description="Value used in current run"
    )
    
    # Metadata
    affects_runtime: bool = Field(
        default=False,
        description="Whether changing this affects runtime significantly"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Return parameter info as clean dict for LLM context."""
        result = {
            "name": self.name,
            "type": self.param_type,
        }
        
        if self.cli_flag:
            result["cli_flag"] = self.cli_flag
        
        if self.min_value is not None:
            result["min"] = self.min_value
        
        if self.max_value is not None:
            result["max"] = self.max_value
        
        if self.choices:
            result["choices"] = self.choices
        
        if self.current_value is not None:
            result["current"] = self.current_value
        
        if self.default is not None:
            result["default"] = self.default
        
        if self.description:
            result["description"] = self.description
        
        return result
    
    def to_prompt_lines(self) -> List[str]:
        """DEPRECATED: Use to_dict() instead."""
        import json
        return [json.dumps(self.to_dict())]
    
    def to_prompt_block(self) -> str:
        """DEPRECATED: Use to_dict() instead."""
        import json
        return json.dumps(self.to_dict())


class ParameterContext(BaseModel):
    """
    Complete parameter context for AI agent decision-making.
    
    This is what gets surfaced in the LLM prompt so the agent
    knows exactly what it can adjust and within what bounds.
    """
    
    script_name: str = Field(
        default="",
        description="Name of the script being configured"
    )
    
    parameters: List[ParameterInfo] = Field(
        default_factory=list,
        description="List of adjustable parameters"
    )
    
    @classmethod
    def build(
        cls,
        manifest: "AgentManifest",
        current_values: Optional[Dict[str, Any]] = None
    ) -> "ParameterContext":
        """
        Build parameter context from manifest + registry + current values.
        
        Args:
            manifest: Agent manifest with args_map
            current_values: Dict of current parameter values from results/config
            
        Returns:
            ParameterContext with all parameters populated
        """
        current_values = current_values or {}
        
        script_name = manifest.script_name
        args_map = manifest.adjustable_args
        
        # Get registry for this script
        registry = get_registry(script_name)
        
        parameters = []
        
        for cli_flag, context_var in args_map.items():
            # Normalize name (remove dashes, use underscores)
            clean_name = cli_flag.lstrip('-').replace('-', '_')
            
            # Get spec from registry if available
            spec: Optional[ParameterSpec] = None
            if registry:
                spec = registry.get_param(clean_name)
                if not spec:
                    spec = registry.get_param(cli_flag)
            
            # Try to find current value
            current = None
            for key in [clean_name, context_var, cli_flag, cli_flag.lstrip('-')]:
                if key in current_values:
                    current = current_values[key]
                    break
            
            # Build parameter info
            param_info = ParameterInfo(
                name=clean_name,
                cli_flag=f"--{clean_name.replace('_', '-')}",
                context_var=context_var,
                current_value=current
            )
            
            # Enrich from registry spec
            if spec:
                param_info.param_type = spec.param_type.value
                param_info.description = spec.description
                param_info.min_value = spec.min_value
                param_info.max_value = spec.max_value
                param_info.choices = spec.choices
                param_info.default = spec.default
                param_info.affects_runtime = spec.affects_runtime
            
            parameters.append(param_info)
        
        return cls(
            script_name=script_name,
            parameters=parameters
        )
    
    @classmethod
    def from_results(
        cls,
        manifest: "AgentManifest",
        results: Dict[str, Any]
    ) -> "ParameterContext":
        """
        Build from manifest and results JSON.
        
        Extracts current values from common locations in results.
        """
        # Extract current values from various locations
        current_values = {}
        
        # Try different sections
        for section in ['config', 'best_config', 'parameters', 'analysis_parameters', 'settings']:
            if section in results and isinstance(results[section], dict):
                current_values.update(results[section])
        
        # Also check top-level
        for key, value in results.items():
            if not isinstance(value, (dict, list)) and key not in current_values:
                current_values[key] = value
        
        return cls.build(manifest, current_values)
    
    def get_param(self, name: str) -> Optional[ParameterInfo]:
        """Get parameter by name."""
        name_clean = name.lstrip('-').replace('-', '_').lower()
        for p in self.parameters:
            if p.name.lower() == name_clean:
                return p
        return None
    
    def validate_adjustments(self, adjustments: Dict[str, Any]) -> List[str]:
        """
        Validate proposed parameter adjustments against bounds.
        
        Args:
            adjustments: Dict of param_name -> new_value
            
        Returns:
            List of validation errors (empty if all valid)
        """
        errors = []
        
        for name, value in adjustments.items():
            param = self.get_param(name)
            
            if param is None:
                errors.append(f"Unknown parameter: {name}")
                continue
            
            # Check type
            if param.param_type == "int":
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    errors.append(f"{name} must be an integer, got {type(value).__name__}")
                    continue
            
            elif param.param_type == "float":
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    errors.append(f"{name} must be a number, got {type(value).__name__}")
                    continue
            
            # Check bounds
            if param.min_value is not None and value < param.min_value:
                errors.append(f"{name}={value} below minimum {param.min_value}")
            
            if param.max_value is not None and value > param.max_value:
                errors.append(f"{name}={value} above maximum {param.max_value}")
            
            # Check choices
            if param.choices and value not in param.choices:
                errors.append(f"{name}={value} not in allowed choices {param.choices}")
        
        return errors
    
    def to_context_dict(self) -> Dict[str, Any]:
        """
        Generate parameter context as clean JSON dict.
        
        This is what gets sent to the LLM - no prose, just data.
        """
        return {
            "script": self.script_name,
            "adjustable_parameters": [p.to_dict() for p in self.parameters]
        }
    
    def to_prompt_section(self) -> str:
        """DEPRECATED: Use to_context_dict() instead."""
        import json
        return json.dumps(self.to_context_dict(), indent=2)
    
    def get_adjustable_names(self) -> List[str]:
        """Get list of all adjustable parameter names."""
        return [p.name for p in self.parameters]
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get dict of current parameter values."""
        return {
            p.name: p.current_value 
            for p in self.parameters 
            if p.current_value is not None
        }
