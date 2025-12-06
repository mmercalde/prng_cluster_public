#!/usr/bin/env python3
"""
LLM Router - Intelligent Request Routing for Dual-Model Architecture
Version: 1.1.0 (Grammar-Constrained Decoding Integration)

Routes requests to appropriate LLM based on task characteristics:
- Math/PRNG tasks → Qwen2.5-Math-7B (GPU1, port 8081)
- Code/orchestration → Qwen2.5-Coder-14B (GPU0, port 8080)

v1.1.0 Changes (GBNF Grammar Integration):
- NEW: Grammar-constrained decoding for structured JSON output
- NEW: Auto-grammar selection based on prompt content
- NEW: Convenience methods for agent decisions, sieve analysis, parameter suggestions
- NEW: GrammarType enum for explicit grammar selection
- All v1.0.5 features preserved

v1.0.5 Features (Preserved):
- Model-specific max_tokens (2048 orchestrator, 512 math)
- Context reset trigger at 14K tokens
- Agent identity headers in prompts
- Expanded routing keywords for full PRNG coverage
- Orchestrator → Math delegation mechanism ([MATH_DELEGATE])
- Rotating disk-based LLM logging
- Human override trigger detection

Part of: Distributed PRNG Analysis System
Schema Version: 1.1.0
"""

import requests
import json
import time
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from enum import Enum


# =============================================================================
# v1.1.0: Grammar Integration Imports
# =============================================================================

class GrammarType(str, Enum):
    """Available grammar types for constrained decoding."""
    AGENT_DECISION = "agent_decision"
    SIEVE_ANALYSIS = "sieve_analysis"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    JSON_GENERIC = "json_generic"
    NONE = "none"  # Disable grammar


# Grammar auto-selection patterns
GRAMMAR_PATTERNS = {
    # Agent decision patterns
    "evaluate": GrammarType.AGENT_DECISION,
    "decision": GrammarType.AGENT_DECISION,
    "success_condition": GrammarType.AGENT_DECISION,
    "proceed": GrammarType.AGENT_DECISION,
    "retry": GrammarType.AGENT_DECISION,
    "escalate": GrammarType.AGENT_DECISION,
    "recommended_action": GrammarType.AGENT_DECISION,
    
    # Sieve analysis patterns
    "sieve": GrammarType.SIEVE_ANALYSIS,
    "survivors": GrammarType.SIEVE_ANALYSIS,
    "forward_survivors": GrammarType.SIEVE_ANALYSIS,
    "reverse_survivors": GrammarType.SIEVE_ANALYSIS,
    "bidirectional": GrammarType.SIEVE_ANALYSIS,
    "match_rate": GrammarType.SIEVE_ANALYSIS,
    
    # Parameter adjustment patterns
    "adjust": GrammarType.PARAMETER_ADJUSTMENT,
    "parameter": GrammarType.PARAMETER_ADJUSTMENT,
    "window_size": GrammarType.PARAMETER_ADJUSTMENT,
    "threshold": GrammarType.PARAMETER_ADJUSTMENT,
    "suggest": GrammarType.PARAMETER_ADJUSTMENT,
    "suggested_param": GrammarType.PARAMETER_ADJUSTMENT,
}


class GrammarLoader:
    """
    Loads and manages GBNF grammar files for constrained LLM output.
    v1.1.0 Addition - Integrated grammar management.
    """
    
    def __init__(self, grammar_dir: Optional[Path] = None):
        if grammar_dir is None:
            # Default: look for grammars/ relative to this file
            grammar_dir = Path(__file__).parent.parent / "grammars"
        
        self.grammar_dir = Path(grammar_dir)
        self._cache: Dict[str, str] = {}
        
        # Create directory if it doesn't exist
        if not self.grammar_dir.exists():
            self.grammar_dir.mkdir(parents=True, exist_ok=True)
    
    def get(self, grammar_type: Union[str, GrammarType]) -> Optional[str]:
        """Load grammar by type name."""
        if isinstance(grammar_type, GrammarType):
            grammar_type = grammar_type.value
        
        if grammar_type == "none" or grammar_type is None:
            return None
        
        # Check cache
        if grammar_type in self._cache:
            return self._cache[grammar_type]
        
        # Load from file
        grammar_file = self.grammar_dir / f"{grammar_type}.gbnf"
        
        if not grammar_file.exists():
            return None
        
        try:
            grammar_content = grammar_file.read_text()
            self._cache[grammar_type] = grammar_content
            return grammar_content
        except Exception:
            return None
    
    def get_for_prompt(self, prompt: str) -> Optional[str]:
        """Auto-select grammar based on prompt content."""
        prompt_lower = prompt.lower()
        
        for keyword, grammar_type in GRAMMAR_PATTERNS.items():
            if keyword in prompt_lower:
                return self.get(grammar_type)
        
        # Default to generic JSON if prompt suggests JSON output
        if "json" in prompt_lower or "respond with" in prompt_lower:
            return self.get(GrammarType.JSON_GENERIC)
        
        return None
    
    def list_available(self) -> List[str]:
        """List all available grammar files."""
        if not self.grammar_dir.exists():
            return []
        return [f.stem for f in self.grammar_dir.glob("*.gbnf")]


# =============================================================================
# Original v1.0.5 Code (Preserved)
# =============================================================================

@dataclass
class LLMMetrics:
    """Track LLM usage for schema metadata injection"""
    orchestrator_calls: int = 0
    math_calls: int = 0
    total_tokens: int = 0
    context_tokens_estimate: int = 0
    trace: List[Dict] = field(default_factory=list)
    human_override_requested: bool = False
    override_reason: Optional[str] = None
    # v1.1.0: Grammar usage tracking
    grammar_enforced_calls: int = 0


class LLMRouter:
    """
    Dual-LLM Router for PRNG Analysis System (v1.1.0)

    Routes requests to specialized models:
    - Orchestrator (Qwen2.5-Coder-14B): Planning, code gen, JSON manipulation
    - Math Specialist (Qwen2.5-Math-7B): PRNG calculations, statistics, residue math

    v1.1.0 Features:
    - GBNF grammar-constrained decoding
    - Auto-grammar selection
    - Structured output convenience methods
    
    v1.0.5 Features (Preserved):
    - Model-specific token limits
    - Context reset at threshold
    - Agent identity headers
    - Expanded PRNG routing keywords
    - Delegation mechanism
    - Rotating log files
    - Human override detection
    """

    # Human override trigger phrases
    HUMAN_OVERRIDE_TRIGGERS = [
        "HUMAN_REVIEW_REQUIRED",
        "FLAG_FOR_REVIEW",
        "REQUIRES_HUMAN_VERIFICATION",
        "UNCERTAIN_RESULT",
        "LOW_CONFIDENCE_WARNING"
    ]

    # Delegation trigger
    MATH_DELEGATE_TRIGGER = "[MATH_DELEGATE]"

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize router with configuration.

        Args:
            config_path: Path to llm_server_config.json
        """
        if config_path is None:
            script_dir = Path(__file__).parent
            config_path = script_dir / "llm_server_config.json"

        if not Path(config_path).exists():
            raise FileNotFoundError(
                f"LLM config not found: {config_path}\n"
                "Please create llm_server_config.json or specify path."
            )

        with open(config_path) as f:
            self.config = json.load(f)

        self.endpoints = {
            "orchestrator": f"http://localhost:{self.config['orchestrator']['port']}/completion",
            "math": f"http://localhost:{self.config['math']['port']}/completion"
        }

        self.health_endpoints = {
            "orchestrator": f"http://localhost:{self.config['orchestrator']['port']}/health",
            "math": f"http://localhost:{self.config['math']['port']}/health"
        }

        # Model-specific max tokens
        self.max_tokens = {
            "orchestrator": self.config.get("orchestrator", {}).get("max_tokens", 2048),
            "math": self.config.get("math", {}).get("max_tokens", 512)
        }

        # Expanded routing keywords
        self.math_patterns = self.config.get("routing", {}).get("math_keywords", [
            "residue", "modulo", "modular", "state =", "seed", "probability",
            "calculate", "compute", "lcg", "xorshift", "prng", "xorshifted",
            "statistical", "confidence", "threshold", "entropy", "bitwise",
            "survivor", "match_rate", "skip", "pcg", "mersenne", "mt19937",
            "chi-squared", "distribution", "variance", "standard deviation",
            "forward sieve", "reverse sieve", "bidirectional",
            "skip interval", "gap-aware", "state reconstruction",
            "temper", "twist", "pcg output", "xoroshiro", "lcg step",
            "survivor scoring", "residue filter", "window size",
            "match rate", "survival rate", "hybrid sieve"
        ])

        self.default_temperature = self.config.get("routing", {}).get("default_temperature", 0.7)
        self.timeout = self.config.get("routing", {}).get("request_timeout_seconds", 120)

        # Context reset threshold
        self.context_reset_threshold = self.config.get("routing", {}).get("context_reset_threshold", 14000)

        self.metrics = LLMMetrics()
        self.current_agent: Optional[str] = None

        # v1.1.0: Initialize grammar loader
        grammar_dir = self.config.get("paths", {}).get("grammar_directory", None)
        self.grammar_loader = GrammarLoader(Path(grammar_dir) if grammar_dir else None)
        
        # v1.1.0: Default grammar behavior
        self.auto_grammar = self.config.get("routing", {}).get("auto_grammar", True)

        # Setup rotating loggers
        self._setup_logging()

    def _setup_logging(self):
        """Setup rotating disk-based logging"""
        log_dir = Path(self.config.get("paths", {}).get("log_directory", "logs/llm"))

        (log_dir / "orchestrator").mkdir(parents=True, exist_ok=True)
        (log_dir / "math").mkdir(parents=True, exist_ok=True)

        self.loggers = {}
        for endpoint in ["orchestrator", "math"]:
            logger = logging.getLogger(f"llm_{endpoint}")
            logger.setLevel(logging.INFO)

            if not logger.handlers:
                handler = RotatingFileHandler(
                    log_dir / endpoint / "requests.log",
                    maxBytes=10*1024*1024,
                    backupCount=5
                )
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s | %(message)s'
                ))
                logger.addHandler(handler)

            self.loggers[endpoint] = logger

    def _log_request(self, endpoint: str, prompt: str, response: str,
                     tokens: int, latency_ms: int, agent: Optional[str],
                     grammar_used: Optional[str] = None):
        """Log request to rotating file"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent or "unknown",
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "grammar": grammar_used  # v1.1.0: Track grammar usage
        }
        self.loggers[endpoint].info(json.dumps(log_entry))

    def set_agent(self, agent_name: str):
        """Set the current agent identity for prompt headers"""
        self.current_agent = agent_name

    def _add_agent_header(self, prompt: str) -> str:
        """Add agent identity header to prompt"""
        if self.current_agent:
            return f"<agent: {self.current_agent}>\n{prompt}"
        return prompt

    def _check_context_reset(self):
        """Check if context reset is needed"""
        if self.metrics.context_tokens_estimate > self.context_reset_threshold:
            self.reset_context()

    def reset_context(self):
        """Reset context to prevent KV cache bloat"""
        self.metrics.context_tokens_estimate = 0

    def _check_human_override(self, response: str) -> bool:
        """Check if response requests human review"""
        for trigger in self.HUMAN_OVERRIDE_TRIGGERS:
            if trigger in response:
                self.metrics.human_override_requested = True
                self.metrics.override_reason = trigger
                return True
        return False

    def _check_delegation(self, response: str) -> bool:
        """Check if orchestrator requests math delegation"""
        return self.MATH_DELEGATE_TRIGGER in response

    def route(self, prompt: str, 
              force_endpoint: Optional[str] = None,
              temperature: Optional[float] = None,
              max_tokens: Optional[int] = None,
              agent: Optional[str] = None,
              grammar: Optional[str] = None,
              grammar_type: Optional[Union[str, GrammarType]] = None,
              auto_grammar: Optional[bool] = None) -> str:
        """
        Route request to appropriate LLM based on content analysis.

        Args:
            prompt: The query to send to the LLM
            force_endpoint: Optional - "orchestrator" or "math" to override auto-routing
            temperature: Optional - Override default temperature (0.0-1.0)
            max_tokens: Optional - Override model-specific max tokens
            agent: Optional - Agent identity for prompt header
            grammar: Optional - Raw GBNF grammar string (v1.1.0)
            grammar_type: Optional - GrammarType enum or string (v1.1.0)
            auto_grammar: Optional - Override auto-grammar setting (v1.1.0)

        Returns:
            LLM response text (guaranteed valid JSON if grammar used)
        """
        # Set agent if provided
        if agent:
            self.set_agent(agent)

        # Check context reset
        self._check_context_reset()

        if temperature is None:
            temperature = self.default_temperature

        # Determine endpoint
        if force_endpoint:
            if force_endpoint not in self.endpoints:
                raise ValueError(f"Unknown endpoint: {force_endpoint}")
            endpoint_name = force_endpoint
        elif self._is_math_task(prompt):
            endpoint_name = "math"
        else:
            endpoint_name = "orchestrator"

        # Use model-specific max_tokens
        if max_tokens is None:
            max_tokens = self.max_tokens[endpoint_name]

        # Add agent header
        full_prompt = self._add_agent_header(prompt)

        endpoint = self.endpoints[endpoint_name]

        # =====================================================================
        # v1.1.0: Grammar Resolution
        # =====================================================================
        grammar_content = None
        grammar_name = None
        
        if grammar is not None:
            # Explicit grammar string provided
            grammar_content = grammar
            grammar_name = "custom"
        elif grammar_type is not None:
            # Grammar type specified
            if isinstance(grammar_type, GrammarType):
                if grammar_type != GrammarType.NONE:
                    grammar_content = self.grammar_loader.get(grammar_type)
                    grammar_name = grammar_type.value
            elif grammar_type != "none":
                grammar_content = self.grammar_loader.get(grammar_type)
                grammar_name = grammar_type
        elif (auto_grammar if auto_grammar is not None else self.auto_grammar):
            # Auto-select based on prompt
            grammar_content = self.grammar_loader.get_for_prompt(prompt)
            if grammar_content:
                # Determine which grammar was selected for logging
                prompt_lower = prompt.lower()
                for keyword, gtype in GRAMMAR_PATTERNS.items():
                    if keyword in prompt_lower:
                        grammar_name = gtype.value
                        break
                else:
                    grammar_name = "json_generic"

        # Build request payload
        payload = {
            "prompt": full_prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["</s>", "<|im_end|>", "<|endoftext|>", "```\n\n"]
        }
        
        # v1.1.0: Add grammar to payload if we have one
        if grammar_content:
            payload["grammar"] = grammar_content
            self.metrics.grammar_enforced_calls += 1

        # Make request
        start_time = time.time()

        try:
            response = requests.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to {endpoint_name} LLM server at {endpoint}\n"
                "Please ensure servers are running: ./scripts/start_llm_servers.sh"
            )

        content = result.get("content", "")
        tokens = result.get("tokens_predicted", len(content.split()))
        latency_ms = int((time.time() - start_time) * 1000)

        # Check for delegation request
        if endpoint_name == "orchestrator" and self._check_delegation(content):
            content = content.replace(self.MATH_DELEGATE_TRIGGER, "").strip()
            delegation_prompt = f"[Delegated from Orchestrator]\n{prompt}"
            return self.route(delegation_prompt, force_endpoint="math",
                            temperature=temperature, agent=agent,
                            grammar=grammar, grammar_type=grammar_type,
                            auto_grammar=auto_grammar)

        # Check for human override request
        self._check_human_override(content)

        # Update metrics
        if endpoint_name == "orchestrator":
            self.metrics.orchestrator_calls += 1
        else:
            self.metrics.math_calls += 1

        self.metrics.total_tokens += tokens
        self.metrics.context_tokens_estimate += tokens + len(prompt.split())

        self.metrics.trace.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": endpoint_name,
            "agent": self.current_agent,
            "query_type": self._classify_query(prompt),
            "tokens": tokens,
            "latency_ms": latency_ms,
            "grammar": grammar_name  # v1.1.0: Track grammar in trace
        })

        # Log to rotating file
        self._log_request(endpoint_name, prompt, content, tokens,
                         latency_ms, self.current_agent, grammar_name)

        return content

    def _is_math_task(self, prompt: str) -> bool:
        """Detect if this is a math/PRNG task based on keywords"""
        prompt_lower = prompt.lower()
        return any(pattern in prompt_lower for pattern in self.math_patterns)

    def _classify_query(self, prompt: str) -> str:
        """Classify query type for tracing/analytics"""
        prompt_lower = prompt.lower()

        if "residue" in prompt_lower or "modulo" in prompt_lower:
            return "residue_analysis"
        elif "forward sieve" in prompt_lower or "reverse sieve" in prompt_lower:
            return "sieve_analysis"
        elif "seed" in prompt_lower or "prng" in prompt_lower:
            return "prng_state"
        elif "confidence" in prompt_lower or "probability" in prompt_lower:
            return "statistical"
        elif "json" in prompt_lower or "config" in prompt_lower:
            return "config_generation"
        elif "plan" in prompt_lower or "next" in prompt_lower:
            return "orchestration"
        elif "code" in prompt_lower or "python" in prompt_lower:
            return "code_generation"
        else:
            return "general"

    def get_llm_metadata(self) -> Dict[str, Any]:
        """
        Get metadata for schema injection into agent_metadata.llm_metadata

        Returns:
            Dict matching schema v1.1.0 llm_metadata specification
        """
        return {
            "orchestrator_model": self.config["orchestrator"]["model"],
            "math_model": self.config["math"]["model"],
            "orchestrator_calls": self.metrics.orchestrator_calls,
            "math_calls": self.metrics.math_calls,
            "total_tokens_generated": self.metrics.total_tokens,
            "grammar_enforced_calls": self.metrics.grammar_enforced_calls,  # v1.1.0
            "llm_reasoning_trace": self.metrics.trace[-10:],
            "llm_decision": self._summarize_decision(),
            "human_override_requested": self.metrics.human_override_requested,
            "override_reason": self.metrics.override_reason
        }

    def _summarize_decision(self) -> str:
        """Summarize LLM usage pattern for metadata"""
        if self.metrics.human_override_requested:
            return "human_review_requested"

        total = self.metrics.orchestrator_calls + self.metrics.math_calls
        if total == 0:
            return "no_llm_calls"

        math_ratio = self.metrics.math_calls / total
        if math_ratio > 0.7:
            return "math_heavy_analysis"
        elif math_ratio < 0.3:
            return "orchestration_focused"
        else:
            return "balanced_routing"

    def is_human_override_requested(self) -> bool:
        """Check if human override was requested"""
        return self.metrics.human_override_requested

    def reset_metrics(self):
        """Reset metrics for new run/step"""
        self.metrics = LLMMetrics()
        self.current_agent = None

    # =========================================================================
    # v1.1.0: Grammar-Constrained Convenience Methods
    # =========================================================================

    def evaluate_decision(self, prompt: str, agent: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Evaluate and return structured agent decision.
        
        Uses agent_decision.gbnf grammar to guarantee output format:
        {
            "success_condition_met": bool,
            "confidence": float,
            "reasoning": str,
            "recommended_action": "proceed"|"retry"|"escalate",
            "suggested_param_adjustments": {...},
            "warnings": [...]
        }
        
        Args:
            prompt: Evaluation prompt
            agent: Agent identity
            **kwargs: Additional route() arguments
        
        Returns:
            Parsed decision dict (guaranteed structure)
        """
        response = self.route(
            prompt, 
            agent=agent,
            grammar_type=GrammarType.AGENT_DECISION,
            **kwargs
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            # Should not happen with grammar, but handle gracefully
            return {
                "success_condition_met": False,
                "confidence": 0.0,
                "reasoning": f"JSON parse error (grammar may have failed): {e}",
                "recommended_action": "escalate",
                "warnings": [f"Parse error: {str(e)}", f"Raw response: {response[:200]}"]
            }

    def analyze_sieve(self, prompt: str, agent: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Analyze sieve results with structured output.
        
        Uses sieve_analysis.gbnf grammar to guarantee output format:
        {
            "analysis_type": str,
            "prng_type": str,
            "survivor_assessment": str,
            "forward_survivors": int,
            "reverse_survivors": int,
            "bidirectional_survivors": int,
            "match_rate": float,
            "recommended_window_size": int,
            "recommended_threshold": float,
            "interpretation": str,
            "next_step": str
        }
        
        Args:
            prompt: Analysis prompt
            agent: Agent identity
            **kwargs: Additional route() arguments
        
        Returns:
            Parsed analysis dict (guaranteed structure)
        """
        response = self.route(
            prompt,
            agent=agent,
            grammar_type=GrammarType.SIEVE_ANALYSIS,
            **kwargs
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            return {
                "analysis_type": "error",
                "prng_type": "unknown",
                "survivor_assessment": "needs_retry",
                "forward_survivors": 0,
                "reverse_survivors": 0,
                "bidirectional_survivors": 0,
                "match_rate": 0.0,
                "recommended_window_size": 512,
                "recommended_threshold": 0.5,
                "interpretation": f"Parse error: {e}",
                "next_step": "escalate"
            }

    def suggest_parameters(self, prompt: str, agent: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Get parameter adjustment suggestions with structured output.
        
        Uses parameter_adjustment.gbnf grammar.
        
        Args:
            prompt: Parameter suggestion prompt
            agent: Agent identity
            **kwargs: Additional route() arguments
        
        Returns:
            Parsed parameter dict
        """
        response = self.route(
            prompt,
            agent=agent,
            grammar_type=GrammarType.PARAMETER_ADJUSTMENT,
            **kwargs
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            return {
                "pipeline_step": 0,
                "step_name": "error",
                "adjustments": {},
                "rationale": f"Parse error: {e}"
            }

    # =========================================================================
    # Original Convenience Methods (Preserved)
    # =========================================================================

    def orchestrate(self, task: str, agent: Optional[str] = None, **kwargs) -> str:
        """Force orchestrator for planning/code tasks."""
        return self.route(task, force_endpoint="orchestrator", agent=agent, **kwargs)

    def calculate(self, math_query: str, agent: Optional[str] = None, **kwargs) -> str:
        """Force math specialist for calculations."""
        return self.route(math_query, force_endpoint="math", agent=agent, **kwargs)

    def generate_json(self, spec: str, agent: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate and parse JSON from orchestrator.
        
        v1.1.0: Now uses json_generic grammar for guaranteed valid JSON.
        """
        prompt = f"""Generate ONLY valid JSON (no markdown, no explanation) for:
{spec}

Respond with raw JSON only."""

        response = self.route(
            prompt, 
            force_endpoint="orchestrator",
            agent=agent, 
            temperature=0.3,
            grammar_type=GrammarType.JSON_GENERIC,
            **kwargs
        )

        # With grammar, response should already be valid JSON
        # But keep cleanup for backward compatibility
        response = response.strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        response = response.strip()

        return json.loads(response)

    def health_check(self) -> Dict[str, bool]:
        """Check both LLM endpoints are responsive."""
        status = {}
        for name, endpoint in self.health_endpoints.items():
            try:
                resp = requests.get(endpoint, timeout=5)
                status[name] = resp.status_code == 200
            except:
                status[name] = False
        
        # v1.1.0: Also report grammar availability
        status["grammars_available"] = len(self.grammar_loader.list_available()) > 0
        
        return status

    def is_healthy(self) -> bool:
        """Quick check if at least one endpoint is available"""
        status = self.health_check()
        return status.get("orchestrator", False) or status.get("math", False)


# =============================================================================
# Singleton Pattern
# =============================================================================

_router_instance: Optional[LLMRouter] = None

def get_router(config_path: Optional[str] = None) -> LLMRouter:
    """Get or create router singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter(config_path)
    return _router_instance


def reset_router():
    """Reset singleton (useful for testing)"""
    global _router_instance
    _router_instance = None


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Router CLI v1.1.0 (Grammar Support)")
    parser.add_argument("--health", action="store_true", help="Check server health")
    parser.add_argument("--query", type=str, help="Send a query")
    parser.add_argument("--endpoint", choices=["orchestrator", "math"], help="Force endpoint")
    parser.add_argument("--agent", type=str, help="Agent identity for prompt header")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--grammar", type=str, 
                       choices=["agent_decision", "sieve_analysis", "parameter_adjustment", "json_generic", "none"],
                       help="Grammar type to use (v1.1.0)")
    parser.add_argument("--no-grammar", action="store_true", help="Disable auto-grammar")
    parser.add_argument("--list-grammars", action="store_true", help="List available grammars")

    args = parser.parse_args()

    try:
        router = LLMRouter(args.config) if args.config else get_router()

        if args.list_grammars:
            print("Available GBNF Grammars:")
            grammars = router.grammar_loader.list_available()
            if grammars:
                for g in grammars:
                    print(f"  ✅ {g}")
            else:
                print("  ❌ No grammar files found")
                print(f"     Expected location: {router.grammar_loader.grammar_dir}")

        elif args.health:
            status = router.health_check()
            print("LLM Server Health Status (v1.1.0):")
            for name, healthy in status.items():
                icon = "✅" if healthy else "❌"
                print(f"  {icon} {name}: {'HEALTHY' if healthy else 'UNAVAILABLE'}")

        elif args.query:
            print(f"Routing query to: {args.endpoint or 'auto'}")
            if args.agent:
                print(f"Agent identity: {args.agent}")
            if args.grammar:
                print(f"Grammar: {args.grammar}")
            elif not args.no_grammar:
                print("Grammar: auto-select")
            
            response = router.route(
                args.query, 
                force_endpoint=args.endpoint, 
                agent=args.agent,
                grammar_type=args.grammar if args.grammar else None,
                auto_grammar=not args.no_grammar
            )
            print("\nResponse:")
            print(response)
            print(f"\nMetrics: {router.metrics.orchestrator_calls} orchestrator, "
                  f"{router.metrics.math_calls} math, "
                  f"{router.metrics.total_tokens} tokens, "
                  f"{router.metrics.grammar_enforced_calls} grammar-enforced")
            if router.is_human_override_requested():
                print(f"⚠️  HUMAN OVERRIDE REQUESTED: {router.metrics.override_reason}")

        else:
            parser.print_help()

    except Exception as e:
        print(f"Error: {e}")
        exit(1)
