#!/usr/bin/env python3
"""
Agent Decision - Robust LLM response validation.

Includes:
- Strict action validation (proceed/retry/escalate)
- JSON repair fallback for malformed responses
- Confidence bounds enforcement
- Graceful degradation to safe defaults

Version: 3.2.0
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import json
import re


# Allowed actions - strict validation
ALLOWED_ACTIONS = frozenset({"proceed", "retry", "escalate"})


class AgentDecision(BaseModel):
    """
    Validated LLM decision response.
    
    Strict validation prevents malformed decisions from breaking the pipeline.
    
    Example valid response:
        {
            "success_condition_met": true,
            "confidence": 0.85,
            "reasoning": "Bidirectional count of 47 is within acceptable range",
            "recommended_action": "proceed",
            "suggested_param_adjustments": {},
            "warnings": []
        }
    """
    
    success_condition_met: bool = Field(
        ...,
        description="Whether the manifest's success_condition was satisfied"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in this evaluation (0.0-1.0)"
    )
    
    reasoning: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Brief explanation using step-specific metrics"
    )
    
    # Strict action validation with Literal type
    recommended_action: Literal["proceed", "retry", "escalate"] = Field(
        ...,
        description="What should happen next"
    )
    
    suggested_param_adjustments: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter changes for retry"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="Any concerns or recommendations"
    )
    
    # Metadata (auto-populated)
    parsed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this decision was parsed"
    )
    
    parse_method: str = Field(
        default="direct",
        description="How the response was parsed (direct/markdown/repair/fallback)"
    )
    
    # ══════════════════════════════════════════════════════════════════════
    # VALIDATORS
    # ══════════════════════════════════════════════════════════════════════
    
    @field_validator('recommended_action', mode='before')
    @classmethod
    def validate_action(cls, v):
        """Normalize and validate action."""
        if isinstance(v, str):
            v = v.lower().strip()
            
            # Handle common variants
            action_map = {
                # Proceed variants
                "proceed": "proceed",
                "continue": "proceed",
                "pass": "proceed",
                "ok": "proceed",
                "success": "proceed",
                "approve": "proceed",
                "accept": "proceed",
                "next": "proceed",
                
                # Retry variants
                "retry": "retry",
                "rerun": "retry",
                "adjust": "retry",
                "fail": "retry",
                "redo": "retry",
                "again": "retry",
                "repeat": "retry",
                
                # Escalate variants
                "escalate": "escalate",
                "review": "escalate",
                "human": "escalate",
                "abort": "escalate",
                "stop": "escalate",
                "halt": "escalate",
                "error": "escalate",
                "manual": "escalate",
            }
            
            if v in action_map:
                return action_map[v]
            
            # Partial match
            for key, value in action_map.items():
                if key in v or v in key:
                    return value
        
        raise ValueError(f"Invalid action: {v}. Must be one of: {ALLOWED_ACTIONS}")
    
    @field_validator('confidence', mode='before')
    @classmethod
    def validate_confidence(cls, v):
        """Ensure confidence is within bounds."""
        if isinstance(v, str):
            # Handle percentage format like "85%" or "0.85"
            v = v.replace('%', '').strip()
            v = float(v)
            if v > 1:
                v = v / 100  # Convert percentage to decimal
        
        # Clamp to valid range
        return max(0.0, min(1.0, float(v)))
    
    @field_validator('reasoning', mode='before')
    @classmethod
    def validate_reasoning(cls, v):
        """Ensure reasoning is a string."""
        if v is None:
            return "No reasoning provided"
        return str(v)[:1000]  # Truncate if too long
    
    @field_validator('warnings', mode='before')
    @classmethod
    def validate_warnings(cls, v):
        """Ensure warnings is a list of strings."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(w) for w in v]
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Ensure logical consistency of decision."""
        # If success_condition_met is False but proceeding, add warning
        if not self.success_condition_met and self.recommended_action == "proceed":
            self.warnings.append(
                "Proceeding despite success_condition_met=False"
            )
        
        # If action is retry but no param adjustments, add warning
        if self.recommended_action == "retry" and not self.suggested_param_adjustments:
            self.warnings.append(
                "Retry recommended but no param adjustments suggested"
            )
        
        # If escalating with high confidence, add warning
        if self.recommended_action == "escalate" and self.confidence > 0.8:
            self.warnings.append(
                "Escalating despite high confidence - verify reasoning"
            )
        
        return self
    
    # ══════════════════════════════════════════════════════════════════════
    # DECISION HELPERS
    # ══════════════════════════════════════════════════════════════════════
    
    def should_auto_proceed(self, threshold: float = 0.8) -> bool:
        """
        Check if decision meets criteria for automatic continuation.
        
        Args:
            threshold: Minimum confidence for auto-proceed
            
        Returns:
            True if safe to proceed automatically
        """
        return (
            self.success_condition_met and
            self.recommended_action == "proceed" and
            self.confidence >= threshold and
            len(self.warnings) == 0
        )
    
    def should_retry(self) -> bool:
        """Check if retry is recommended."""
        return self.recommended_action == "retry"
    
    def should_escalate(self) -> bool:
        """Check if human review is needed."""
        return self.recommended_action == "escalate"
    
    def get_severity(self) -> str:
        """
        Get severity level of the decision.
        
        Returns:
            "success", "warning", "failure", or "critical"
        """
        if self.recommended_action == "proceed":
            if self.confidence >= 0.8 and len(self.warnings) == 0:
                return "success"
            return "warning"
        elif self.recommended_action == "retry":
            return "failure"
        else:
            return "critical"
    
    # ══════════════════════════════════════════════════════════════════════
    # SERIALIZATION
    # ══════════════════════════════════════════════════════════════════════
    
    def to_agent_metadata(self) -> Dict[str, Any]:
        """
        Convert to Schema v1.0.4 agent_metadata format.
        
        For inclusion in result JSON files.
        """
        return {
            "success_criteria_met": self.success_condition_met,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "recommended_action": self.recommended_action,
            "suggested_params": self.suggested_param_adjustments,
            "warnings": self.warnings,
            "parsed_at": self.parsed_at.isoformat(),
        }
    
    def to_log_dict(self) -> Dict[str, Any]:
        """Convert to dict for logging."""
        return {
            "success": self.success_condition_met,
            "confidence": f"{self.confidence:.2f}",
            "action": self.recommended_action,
            "warnings": len(self.warnings),
            "adjustments": list(self.suggested_param_adjustments.keys()),
        }
    
    @classmethod
    def output_schema(cls) -> Dict[str, Any]:
        """
        Return the expected output schema for LLM responses.
        
        This is included in prompts so the LLM knows exactly
        what format to respond with.
        """
        return {
            "success_condition_met": "boolean - true if success condition satisfied",
            "confidence": "float 0.0-1.0 - confidence in this evaluation",
            "reasoning": "string - brief explanation using metrics from results",
            "recommended_action": "string - one of: proceed, retry, escalate",
            "suggested_param_adjustments": "object - parameter changes if retry (empty {} otherwise)",
            "warnings": "array of strings - any concerns (empty [] if none)"
        }
    
    @classmethod
    def output_example(cls) -> Dict[str, Any]:
        """
        Return an example valid output for LLM guidance.
        """
        return {
            "success_condition_met": True,
            "confidence": 0.85,
            "reasoning": "Bidirectional count of 47 is within target range 1-1000",
            "recommended_action": "proceed",
            "suggested_param_adjustments": {},
            "warnings": []
        }


# ════════════════════════════════════════════════════════════════════════════════
# JSON REPAIR UTILITIES
# ════════════════════════════════════════════════════════════════════════════════

def repair_json(json_str: str) -> str:
    """
    Attempt to repair common JSON issues from LLM output.
    
    Handles:
    - Markdown code blocks
    - Trailing commas
    - Unquoted keys
    - Single quotes
    - Python booleans (True/False/None)
    - Comments
    
    Args:
        json_str: Potentially malformed JSON string
        
    Returns:
        Repaired JSON string
    """
    # Remove markdown code blocks
    json_str = re.sub(r'^```(?:json)?\s*', '', json_str, flags=re.MULTILINE)
    json_str = re.sub(r'\s*```$', '', json_str, flags=re.MULTILINE)
    
    # Remove single-line comments
    json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
    
    # Remove multi-line comments
    json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
    
    # Fix trailing commas (common LLM error)
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Fix single quotes to double quotes (but not inside strings)
    # Simple approach: replace all single quotes
    json_str = json_str.replace("'", '"')
    
    # Fix Python booleans and None
    json_str = re.sub(r'\bTrue\b', 'true', json_str)
    json_str = re.sub(r'\bFalse\b', 'false', json_str)
    json_str = re.sub(r'\bNone\b', 'null', json_str)
    
    # Fix unquoted keys (simple heuristic)
    # Match word followed by colon that isn't already quoted
    json_str = re.sub(r'(?<=[{,\s])(\w+)(?=\s*:)', r'"\1"', json_str)
    
    # Remove any BOM or invisible characters
    json_str = json_str.strip().strip('\ufeff')
    
    return json_str


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object from text that may contain other content.
    
    Tries:
    1. Markdown code block
    2. First {...} block
    3. Last {...} block (in case of preamble)
    
    Returns:
        Extracted JSON string or None
    """
    # Try markdown code block first
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try to find JSON object
    # Find all {...} blocks
    brace_matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL))
    
    if brace_matches:
        # Prefer larger matches (more likely to be the full response)
        brace_matches.sort(key=lambda m: len(m.group(0)), reverse=True)
        return brace_matches[0].group(0)
    
    return None


# ════════════════════════════════════════════════════════════════════════════════
# MAIN PARSER FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def parse_llm_response(response_text: str) -> AgentDecision:
    """
    Parse LLM response with JSON repair fallback.
    
    Tries multiple strategies in order:
    1. Direct JSON parse
    2. Extract from markdown code block
    3. Extract {...} from text
    4. JSON repair and retry
    5. Field extraction fallback
    6. Safe default (escalate)
    
    Args:
        response_text: Raw LLM response string
        
    Returns:
        Validated AgentDecision (never raises, always returns something)
    """
    errors = []
    
    # Normalize whitespace
    response_text = response_text.strip()
    
    # ══════════════════════════════════════════════════════════════════════
    # Strategy 1: Direct parse
    # ══════════════════════════════════════════════════════════════════════
    try:
        data = json.loads(response_text)
        decision = AgentDecision.model_validate(data)
        decision.parse_method = "direct"
        return decision
    except Exception as e:
        errors.append(f"Direct parse: {e}")
    
    # ══════════════════════════════════════════════════════════════════════
    # Strategy 2: Extract JSON from text
    # ══════════════════════════════════════════════════════════════════════
    json_str = extract_json_from_text(response_text)
    if json_str:
        try:
            data = json.loads(json_str)
            decision = AgentDecision.model_validate(data)
            decision.parse_method = "extracted"
            return decision
        except Exception as e:
            errors.append(f"Extracted parse: {e}")
        
        # ══════════════════════════════════════════════════════════════════
        # Strategy 3: Repair JSON and retry
        # ══════════════════════════════════════════════════════════════════
        try:
            repaired = repair_json(json_str)
            data = json.loads(repaired)
            decision = AgentDecision.model_validate(data)
            decision.parse_method = "repaired"
            decision.warnings.append("LLM response required JSON repair")
            return decision
        except Exception as e:
            errors.append(f"Repaired parse: {e}")
    
    # ══════════════════════════════════════════════════════════════════════
    # Strategy 4: Field extraction fallback
    # ══════════════════════════════════════════════════════════════════════
    try:
        # Look for key fields anywhere in text
        confidence_match = re.search(
            r'"?confidence"?\s*:\s*([\d.]+)', 
            response_text
        )
        action_match = re.search(
            r'"?recommended_action"?\s*:\s*"?(\w+)"?', 
            response_text
        )
        success_match = re.search(
            r'"?success_condition_met"?\s*:\s*(true|false)', 
            response_text, 
            re.IGNORECASE
        )
        reasoning_match = re.search(
            r'"?reasoning"?\s*:\s*"([^"]+)"',
            response_text
        )
        
        # Build decision from extracted fields
        decision = AgentDecision(
            success_condition_met=(
                success_match.group(1).lower() == "true" 
                if success_match else False
            ),
            confidence=(
                float(confidence_match.group(1)) 
                if confidence_match else 0.5
            ),
            reasoning=(
                reasoning_match.group(1) 
                if reasoning_match 
                else f"Partial parse from malformed response"
            ),
            recommended_action=(
                action_match.group(1).lower() 
                if action_match else "escalate"
            ),
            warnings=[
                "LLM response was malformed - used field extraction fallback",
                f"Parse errors: {errors[:2]}"
            ]
        )
        decision.parse_method = "field_extraction"
        return decision
        
    except Exception as e:
        errors.append(f"Field extraction: {e}")
    
    # ══════════════════════════════════════════════════════════════════════
    # Strategy 5: Safe default (always escalate when parsing fails)
    # ══════════════════════════════════════════════════════════════════════
    return AgentDecision(
        success_condition_met=False,
        confidence=0.0,
        reasoning=f"Complete parse failure. Response: {response_text[:200]}...",
        recommended_action="escalate",
        warnings=[
            "Complete parse failure - escalating to human review",
            f"Errors: {errors[:3]}"
        ],
        parse_method="fallback"
    )


# ════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FACTORY
# ════════════════════════════════════════════════════════════════════════════════

def create_decision(
    success: bool,
    confidence: float,
    reasoning: str,
    action: str = "proceed",
    adjustments: Dict[str, Any] = None,
    warnings: List[str] = None
) -> AgentDecision:
    """
    Factory function to create an AgentDecision programmatically.
    
    Args:
        success: Whether success condition was met
        confidence: Confidence level (0.0-1.0)
        reasoning: Explanation
        action: proceed/retry/escalate
        adjustments: Parameter adjustments for retry
        warnings: List of warnings
        
    Returns:
        Validated AgentDecision
    """
    return AgentDecision(
        success_condition_met=success,
        confidence=confidence,
        reasoning=reasoning,
        recommended_action=action,
        suggested_param_adjustments=adjustments or {},
        warnings=warnings or [],
        parse_method="programmatic"
    )
