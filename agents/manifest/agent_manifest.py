#!/usr/bin/env python3
"""
Agent Manifest - Pydantic model for agent configuration.

All fields accessed via attribute syntax, not dictionary.

Example:
    manifest = AgentManifest.load("agent_manifests/window_optimizer.json")
    print(manifest.agent_name)      # ✓ Correct
    print(manifest.get("agent_name"))  # ✗ Wrong - will fail

Version: 3.2.0
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path
import json


class ActionType(str, Enum):
    """Types of actions an agent can perform."""
    RUN_SCRIPT = "run_script"
    RUN_DISTRIBUTED = "run_distributed"
    AGGREGATE = "aggregate"
    COMMAND = "command"
    SUBPROCESS = "subprocess"  # Added for compatibility with existing manifests


class AgentAction(BaseModel):
    """
    Single action definition within an agent manifest.
    
    The args_map defines the mapping from CLI arguments to context variables,
    enabling AI agents to understand what parameters they can adjust.
    """
    
    # Allow extra fields for forward compatibility
    model_config = {"extra": "allow"}
    
    name: str = Field(
        default="",
        description="Action name/identifier"
    )
    
    type: ActionType = Field(
        default=ActionType.RUN_SCRIPT,
        description="Type of action to execute"
    )
    
    script: str = Field(
        default="",
        description="Script filename to execute"
    )
    
    command_template: str = Field(
        default="",
        description="Command template with {variable} placeholders"
    )
    
    args_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of CLI arg names to context variable names"
    )
    
    distributed: bool = Field(
        default=False,
        description="Whether to run distributed across cluster"
    )
    
    timeout_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Maximum execution time in minutes"
    )
    
    working_dir: str = Field(
        default=".",
        description="Working directory for script execution"
    )
    
    def get_script_args(self) -> List[str]:
        """Get list of script argument names (CLI flags)."""
        return list(self.args_map.keys())
    
    def get_context_vars(self) -> List[str]:
        """Get list of context variable names."""
        return list(self.args_map.values())
    
    def build_command(self, context: Dict[str, Any]) -> str:
        """
        Build command string from template and context.
        
        Args:
            context: Dict of variable names to values
            
        Returns:
            Formatted command string
        """
        if self.command_template:
            return self.command_template.format(**context)
        
        # Build from script + args_map
        parts = [f"python3 {self.script}"]
        for cli_arg, var_name in self.args_map.items():
            if var_name in context:
                value = context[var_name]
                if isinstance(value, bool):
                    if value:
                        parts.append(f"--{cli_arg}")
                else:
                    parts.append(f"--{cli_arg} {value}")
        
        return " ".join(parts)


class AgentManifest(BaseModel):
    """
    Complete agent manifest as Pydantic model.
    
    This model represents the contract for an agent:
    - What it does (description)
    - What it needs (inputs)
    - What it produces (outputs)
    - How to run it (actions)
    - What comes next (follow_up_agents)
    - How to judge success (success_condition)
    
    Access fields via attributes:
        manifest.agent_name  ✓
        manifest.get("agent_name")  ✗ WRONG
    """
    
    # Allow extra fields for forward compatibility with manifest versions
    model_config = {"extra": "allow"}
    
    # ══════════════════════════════════════════════════════════════════════
    # IDENTITY
    # ══════════════════════════════════════════════════════════════════════
    
    agent_name: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the agent"
    )
    
    description: str = Field(
        default="",
        description="Human-readable description of agent purpose"
    )
    
    pipeline_step: int = Field(
        ...,
        ge=1,
        le=6,
        description="Position in the 6-step pipeline (1-6)"
    )
    
    # ══════════════════════════════════════════════════════════════════════
    # I/O CONTRACT
    # ══════════════════════════════════════════════════════════════════════
    
    inputs: Any = Field(
        default_factory=list,
        description="Required input files or parameters (list or dict)"
    )
    
    outputs: List[str] = Field(
        default_factory=list,
        description="Expected output files"
    )
    
    @field_validator('inputs', mode='before')
    @classmethod
    def normalize_inputs(cls, v):
        """Handle both list and dict formats for inputs."""
        if isinstance(v, dict):
            # Extract all inputs from required and optional
            all_inputs = []
            all_inputs.extend(v.get('required', []))
            all_inputs.extend(v.get('optional', []))
            return all_inputs
        return v if v else []
    
    @field_validator('outputs', mode='before')
    @classmethod
    def normalize_outputs(cls, v):
        """Handle both list of strings and list of dicts for outputs."""
        if not v:
            return []
        normalized = []
        for item in v:
            if isinstance(item, dict):
                # Extract file path from dict format (v1.1 schema)
                # Priority: file_pattern > file > name
                path = item.get('file_pattern') or item.get('file') or item.get('name', '')
                if path:
                    normalized.append(path)
            else:
                normalized.append(str(item))
        return normalized
    
    # ══════════════════════════════════════════════════════════════════════
    # ACTIONS
    # ══════════════════════════════════════════════════════════════════════
    
    actions: List[AgentAction] = Field(
        default_factory=list,
        description="Actions the agent can perform"
    )
    
    # ══════════════════════════════════════════════════════════════════════
    # FLOW CONTROL
    # ══════════════════════════════════════════════════════════════════════
    
    follow_up_agents: List[str] = Field(
        default_factory=list,
        description="Agents to trigger after successful completion"
    )
    
    success_condition: Any = Field(
        default="",
        description="Condition to evaluate for success (string or dict)"
    )
    
    @field_validator('success_condition', mode='before')
    @classmethod
    def normalize_success_condition(cls, v):
        """Handle both string and dict formats for success_condition."""
        if isinstance(v, dict):
            # Convert dict to human-readable string
            cond_type = v.get('type', '')
            if cond_type == 'file_exists':
                files = v.get('files', [])
                return f"Files exist: {', '.join(files)}"
            elif cond_type == 'metric_threshold':
                metric = v.get('metric', '')
                threshold = v.get('threshold', 0)
                return f"{metric} >= {threshold}"
            else:
                return str(v)
        return v if v else ""
    
    retry: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum retry attempts on failure"
    )
    
    # ══════════════════════════════════════════════════════════════════════
    # METADATA
    # ══════════════════════════════════════════════════════════════════════
    
    version: str = Field(
        default="1.0.0",
        description="Manifest version"
    )
    
    author: str = Field(
        default="",
        description="Author of the manifest"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorization"
    )
    
    # ══════════════════════════════════════════════════════════════════════
    # VALIDATORS
    # ══════════════════════════════════════════════════════════════════════
    
    @field_validator('agent_name')
    @classmethod
    def validate_agent_name(cls, v: str) -> str:
        """Ensure agent name follows conventions."""
        v = v.strip().lower()
        # Replace spaces with underscores
        v = v.replace(' ', '_').replace('-', '_')
        return v
    
    # ══════════════════════════════════════════════════════════════════════
    # FACTORY METHODS
    # ══════════════════════════════════════════════════════════════════════
    
    @classmethod
    def load(cls, manifest_path: str) -> "AgentManifest":
        """
        Load manifest from JSON file.
        
        Args:
            manifest_path: Path to manifest JSON file
            
        Returns:
            Validated AgentManifest instance
            
        Raises:
            FileNotFoundError: If manifest file doesn't exist
            ValidationError: If manifest fails validation
        """
        path = Path(manifest_path)
        if not path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        with open(path) as f:
            data = json.load(f)
        
        return cls.model_validate(data)
    
    @classmethod
    def load_by_name(
        cls, 
        agent_name: str, 
        manifests_dir: str = "agent_manifests"
    ) -> "AgentManifest":
        """
        Load manifest by agent name.
        
        Searches for:
        1. {manifests_dir}/{agent_name}.json
        2. {manifests_dir}/{agent_name}_agent.json
        3. {manifests_dir}/{agent_name.replace('_agent', '')}.json
        
        Args:
            agent_name: Name of the agent
            manifests_dir: Directory containing manifests
            
        Returns:
            Validated AgentManifest instance
        """
        base_dir = Path(manifests_dir)
        
        # Normalize name
        name = agent_name.lower().strip()
        
        # Try various patterns
        patterns = [
            f"{name}.json",
            f"{name}_agent.json",
            f"{name.replace('_agent', '')}.json",
            f"{name.replace('agent', '').strip('_')}.json",
        ]
        
        for pattern in patterns:
            path = base_dir / pattern
            if path.exists():
                return cls.load(str(path))
        
        raise FileNotFoundError(
            f"No manifest found for agent '{agent_name}' in {manifests_dir}. "
            f"Tried: {patterns}"
        )
    
    # ══════════════════════════════════════════════════════════════════════
    # PROPERTIES
    # ══════════════════════════════════════════════════════════════════════
    
    @property
    def primary_action(self) -> Optional[AgentAction]:
        """Get the first/primary action."""
        return self.actions[0] if self.actions else None
    
    @property
    def script_name(self) -> str:
        """Get the primary script name."""
        if self.primary_action:
            return self.primary_action.script
        return ""
    
    @property
    def adjustable_args(self) -> Dict[str, str]:
        """
        Get all adjustable arguments from primary action.
        
        Returns dict mapping CLI arg name -> context variable name.
        This is what AI agents use to understand what they can modify.
        """
        if self.primary_action:
            return self.primary_action.args_map
        return {}
    
    @property
    def next_agent(self) -> Optional[str]:
        """Get the next agent in the pipeline."""
        return self.follow_up_agents[0] if self.follow_up_agents else None
    
    # ══════════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ══════════════════════════════════════════════════════════════════════
    
    def get_adjustable_params_list(self) -> List[str]:
        """Get list of adjustable parameter names for LLM prompt."""
        return list(self.adjustable_args.keys())
    
    def to_context_dict(self) -> Dict[str, Any]:
        """
        Generate manifest context as clean JSON dict.
        
        Uses Pydantic's model_dump() with selective fields.
        This is what gets sent to the LLM - no prose, just data.
        """
        return {
            "agent_name": self.agent_name,
            "pipeline_step": self.pipeline_step,
            "total_steps": 6,
            "script": self.script_name,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "success_condition": self.success_condition,
            "next_agent": self.next_agent,
            "max_retries": self.retry,
            "adjustable_parameters": list(self.adjustable_args.keys())
        }
    
    def to_prompt_section(self) -> str:
        """
        DEPRECATED: Use to_context_dict() instead.
        
        Kept for backward compatibility only.
        """
        import json
        return json.dumps(self.to_context_dict(), indent=2)
    
    def save(self, path: str):
        """Save manifest to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)


# ════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def load_manifest(path_or_name: str, manifests_dir: str = "agent_manifests") -> AgentManifest:
    """
    Load manifest from path or by name.
    
    Args:
        path_or_name: Either a file path or agent name
        manifests_dir: Directory to search if name provided
        
    Returns:
        Validated AgentManifest
    """
    if path_or_name.endswith('.json') or '/' in path_or_name:
        return AgentManifest.load(path_or_name)
    else:
        return AgentManifest.load_by_name(path_or_name, manifests_dir)
