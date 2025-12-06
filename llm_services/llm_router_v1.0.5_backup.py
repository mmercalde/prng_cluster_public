#!/usr/bin/env python3
"""
LLM Router - Intelligent Request Routing for Dual-Model Architecture
Version: 1.0.5 (Incorporates Team Beta Review Recommendations)

Routes requests to appropriate LLM based on task characteristics:
- Math/PRNG tasks → Qwen2.5-Math-7B (GPU1, port 8081)
- Code/orchestration → Qwen2.5-Coder-14B (GPU0, port 8080)

v1.0.5 Changes (Team Beta Review):
- Fix 1: Model-specific max_tokens (2048 orchestrator, 512 math)
- Fix 2: Context reset trigger at 14K tokens
- Fix 3: Agent identity headers in prompts
- Fix 4: Expanded routing keywords for full PRNG coverage
- Fix 5: Orchestrator → Math delegation mechanism ([MATH_DELEGATE])
- Fix 6: Rotating disk-based LLM logging
- Fix 7: Human override trigger detection

Part of: Distributed PRNG Analysis System
Schema Version: 1.0.5
"""

import requests
import json
import time
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler


@dataclass
class LLMMetrics:
    """Track LLM usage for schema metadata injection"""
    orchestrator_calls: int = 0
    math_calls: int = 0
    total_tokens: int = 0
    context_tokens_estimate: int = 0  # v1.0.5: Track for reset trigger
    trace: List[Dict] = field(default_factory=list)
    human_override_requested: bool = False  # v1.0.5: Override tracking
    override_reason: Optional[str] = None


class LLMRouter:
    """
    Dual-LLM Router for PRNG Analysis System (v1.0.5)
    
    Routes requests to specialized models:
    - Orchestrator (Qwen2.5-Coder-14B): Planning, code gen, JSON manipulation
    - Math Specialist (Qwen2.5-Math-7B): PRNG calculations, statistics, residue math
    
    v1.0.5 Features:
    - Model-specific token limits
    - Context reset at threshold
    - Agent identity headers
    - Expanded PRNG routing keywords
    - Delegation mechanism
    - Rotating log files
    - Human override detection
    """
    
    # v1.0.5 Fix 7: Human override trigger phrases
    HUMAN_OVERRIDE_TRIGGERS = [
        "HUMAN_REVIEW_REQUIRED",
        "FLAG_FOR_REVIEW",
        "REQUIRES_HUMAN_VERIFICATION",
        "UNCERTAIN_RESULT",
        "LOW_CONFIDENCE_WARNING"
    ]
    
    # v1.0.5 Fix 5: Delegation trigger
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
        
        # v1.0.5 Fix 1: Model-specific max tokens
        self.max_tokens = {
            "orchestrator": self.config.get("orchestrator", {}).get("max_tokens", 2048),
            "math": self.config.get("math", {}).get("max_tokens", 512)
        }
        
        # v1.0.5 Fix 4: Expanded routing keywords
        self.math_patterns = self.config.get("routing", {}).get("math_keywords", [
            # Original keywords
            "residue", "modulo", "modular", "state =", "seed", "probability",
            "calculate", "compute", "lcg", "xorshift", "prng", "xorshifted",
            "statistical", "confidence", "threshold", "entropy", "bitwise",
            "survivor", "match_rate", "skip", "pcg", "mersenne", "mt19937",
            "chi-squared", "distribution", "variance", "standard deviation",
            # v1.0.5 additions (Team Beta recommendation)
            "forward sieve", "reverse sieve", "bidirectional",
            "skip interval", "gap-aware", "state reconstruction",
            "temper", "twist", "pcg output", "xoroshiro", "lcg step",
            "survivor scoring", "residue filter", "window size",
            "match rate", "survival rate", "hybrid sieve"
        ])
        
        self.default_temperature = self.config.get("routing", {}).get("default_temperature", 0.7)
        self.timeout = self.config.get("routing", {}).get("request_timeout_seconds", 120)
        
        # v1.0.5 Fix 2: Context reset threshold
        self.context_reset_threshold = self.config.get("routing", {}).get("context_reset_threshold", 14000)
        
        self.metrics = LLMMetrics()
        self.current_agent: Optional[str] = None  # v1.0.5 Fix 3: Track calling agent
        
        # v1.0.5 Fix 6: Setup rotating loggers
        self._setup_logging()
    
    def _setup_logging(self):
        """v1.0.5 Fix 6: Setup rotating disk-based logging"""
        log_dir = Path(self.config.get("paths", {}).get("log_directory", "logs/llm"))
        
        # Create log directories
        (log_dir / "orchestrator").mkdir(parents=True, exist_ok=True)
        (log_dir / "math").mkdir(parents=True, exist_ok=True)
        
        # Setup rotating handlers (10MB per file, keep 5 backups)
        self.loggers = {}
        for endpoint in ["orchestrator", "math"]:
            logger = logging.getLogger(f"llm_{endpoint}")
            logger.setLevel(logging.INFO)
            
            # Avoid duplicate handlers on re-init
            if not logger.handlers:
                handler = RotatingFileHandler(
                    log_dir / endpoint / "requests.log",
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=5
                )
                handler.setFormatter(logging.Formatter(
                    '%(asctime)s | %(message)s'
                ))
                logger.addHandler(handler)
            
            self.loggers[endpoint] = logger
    
    def _log_request(self, endpoint: str, prompt: str, response: str, 
                     tokens: int, latency_ms: int, agent: Optional[str]):
        """v1.0.5 Fix 6: Log request to rotating file"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent or "unknown",
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "tokens": tokens,
            "latency_ms": latency_ms,
            "prompt_length": len(prompt),
            "response_length": len(response)
        }
        self.loggers[endpoint].info(json.dumps(log_entry))
    
    def set_agent(self, agent_name: str):
        """v1.0.5 Fix 3: Set the current agent identity for prompt headers"""
        self.current_agent = agent_name
    
    def _add_agent_header(self, prompt: str) -> str:
        """v1.0.5 Fix 3: Add agent identity header to prompt"""
        if self.current_agent:
            return f"<agent: {self.current_agent}>\n{prompt}"
        return prompt
    
    def _check_context_reset(self):
        """v1.0.5 Fix 2: Check if context reset is needed"""
        if self.metrics.context_tokens_estimate > self.context_reset_threshold:
            self.reset_context()
    
    def reset_context(self):
        """v1.0.5 Fix 2: Reset context to prevent KV cache bloat"""
        self.metrics.context_tokens_estimate = 0
        # Note: llama.cpp servers are stateless per-request, 
        # but we track this for multi-turn scenarios
    
    def _check_human_override(self, response: str) -> bool:
        """v1.0.5 Fix 7: Check if response requests human review"""
        for trigger in self.HUMAN_OVERRIDE_TRIGGERS:
            if trigger in response:
                self.metrics.human_override_requested = True
                self.metrics.override_reason = trigger
                return True
        return False
    
    def _check_delegation(self, response: str) -> bool:
        """v1.0.5 Fix 5: Check if orchestrator requests math delegation"""
        return self.MATH_DELEGATE_TRIGGER in response
    
    def route(self, prompt: str, force_endpoint: Optional[str] = None,
              temperature: Optional[float] = None, 
              max_tokens: Optional[int] = None,
              agent: Optional[str] = None) -> str:
        """
        Route request to appropriate LLM based on content analysis.
        
        Args:
            prompt: The query to send to the LLM
            force_endpoint: Optional - "orchestrator" or "math" to override auto-routing
            temperature: Optional - Override default temperature (0.0-1.0)
            max_tokens: Optional - Override model-specific max tokens
            agent: Optional - Agent identity for prompt header (v1.0.5)
            
        Returns:
            LLM response text
        """
        # v1.0.5 Fix 3: Set agent if provided
        if agent:
            self.set_agent(agent)
        
        # v1.0.5 Fix 2: Check context reset
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
        
        # v1.0.5 Fix 1: Use model-specific max_tokens
        if max_tokens is None:
            max_tokens = self.max_tokens[endpoint_name]
        
        # v1.0.5 Fix 3: Add agent header
        full_prompt = self._add_agent_header(prompt)
        
        endpoint = self.endpoints[endpoint_name]
        
        # Make request
        start_time = time.time()
        
        try:
            response = requests.post(endpoint, json={
                "prompt": full_prompt,
                "n_predict": max_tokens,
                "temperature": temperature,
                "stop": ["</s>", "<|im_end|>", "<|endoftext|>", "```\n\n"]
            }, timeout=self.timeout)
            
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
        
        # v1.0.5 Fix 5: Check for delegation request
        if endpoint_name == "orchestrator" and self._check_delegation(content):
            # Re-route to math specialist
            content = content.replace(self.MATH_DELEGATE_TRIGGER, "").strip()
            delegation_prompt = f"[Delegated from Orchestrator]\n{prompt}"
            return self.route(delegation_prompt, force_endpoint="math", 
                            temperature=temperature, agent=agent)
        
        # v1.0.5 Fix 7: Check for human override request
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
            "latency_ms": latency_ms
        })
        
        # v1.0.5 Fix 6: Log to rotating file
        self._log_request(endpoint_name, prompt, content, tokens, 
                         latency_ms, self.current_agent)
        
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
            Dict matching schema v1.0.5 llm_metadata specification
        """
        return {
            "orchestrator_model": self.config["orchestrator"]["model"],
            "math_model": self.config["math"]["model"],
            "orchestrator_calls": self.metrics.orchestrator_calls,
            "math_calls": self.metrics.math_calls,
            "total_tokens_generated": self.metrics.total_tokens,
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
        """v1.0.5 Fix 7: Check if human override was requested"""
        return self.metrics.human_override_requested
    
    def reset_metrics(self):
        """Reset metrics for new run/step"""
        self.metrics = LLMMetrics()
        self.current_agent = None
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def orchestrate(self, task: str, agent: Optional[str] = None, **kwargs) -> str:
        """Force orchestrator for planning/code tasks."""
        return self.route(task, force_endpoint="orchestrator", agent=agent, **kwargs)
    
    def calculate(self, math_query: str, agent: Optional[str] = None, **kwargs) -> str:
        """Force math specialist for calculations."""
        return self.route(math_query, force_endpoint="math", agent=agent, **kwargs)
    
    def generate_json(self, spec: str, agent: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Generate and parse JSON from orchestrator."""
        prompt = f"""Generate ONLY valid JSON (no markdown, no explanation) for:
{spec}

Respond with raw JSON only."""
        
        response = self.orchestrate(prompt, agent=agent, temperature=0.3, **kwargs)
        
        # Clean response
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
        return status
    
    def is_healthy(self) -> bool:
        """Quick check if at least one endpoint is available"""
        status = self.health_check()
        return any(status.values())


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
    
    parser = argparse.ArgumentParser(description="LLM Router CLI v1.0.5")
    parser.add_argument("--health", action="store_true", help="Check server health")
    parser.add_argument("--query", type=str, help="Send a query")
    parser.add_argument("--endpoint", choices=["orchestrator", "math"], help="Force endpoint")
    parser.add_argument("--agent", type=str, help="Agent identity for prompt header")
    parser.add_argument("--config", type=str, help="Path to config file")
    
    args = parser.parse_args()
    
    try:
        router = LLMRouter(args.config) if args.config else get_router()
        
        if args.health:
            status = router.health_check()
            print("LLM Server Health Status (v1.0.5):")
            for name, healthy in status.items():
                icon = "✅" if healthy else "❌"
                print(f"  {icon} {name}: {'HEALTHY' if healthy else 'UNAVAILABLE'}")
        
        elif args.query:
            print(f"Routing query to: {args.endpoint or 'auto'}")
            if args.agent:
                print(f"Agent identity: {args.agent}")
            response = router.route(args.query, force_endpoint=args.endpoint, agent=args.agent)
            print("\nResponse:")
            print(response)
            print(f"\nMetrics: {router.metrics.orchestrator_calls} orchestrator, "
                  f"{router.metrics.math_calls} math, "
                  f"{router.metrics.total_tokens} tokens")
            if router.is_human_override_requested():
                print(f"⚠️  HUMAN OVERRIDE REQUESTED: {router.metrics.override_reason}")
        
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
