#!/usr/bin/env python3
"""
LLM Router v2.1.0 - Primary + Backup Architecture with Full Method API

Routes requests to DeepSeek-R1-14B (primary) with Claude Opus 4.5 backup.
Restores v1.0.5 method API while keeping v2.0.0 routing architecture.

Architecture:
    Primary:  DeepSeek-R1-14B (local, port 8080, 51 tok/s)
    Backup:   Claude Opus 4.5 (Claude Code CLI, 38 tok/s)

Changes from v2.0.0:
    - Restored: orchestrate(), calculate(), generate_json()
    - Restored: set_agent(), _add_agent_header(), _setup_logging()
    - Added: evaluate_watcher_decision() with GBNF grammar support
    - Added: evaluate_decision() alias for backward compatibility
    - Added: _call_primary_with_grammar() for grammar-constrained decoding
    - Added: _parse_json_response() for robust JSON extraction

Return Type Contract:
    - route() -> str (raw LLM response)
    - orchestrate() -> str
    - calculate() -> str  
    - generate_json() -> Dict[str, Any]
    - evaluate_watcher_decision() -> Dict[str, Any]
    - evaluate_decision() -> Dict[str, Any] (alias)

Team Beta Reviewed: January 9, 2026
"""

import requests
import json
import subprocess
import time
import os
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
from logging.handlers import RotatingFileHandler


@dataclass
class LLMMetrics:
    """Track LLM usage for schema metadata."""
    primary_calls: int = 0
    backup_calls: int = 0
    escalations: int = 0
    total_tokens: int = 0
    trace: List[Dict] = field(default_factory=list)


# Module-level logger
logger = logging.getLogger("llm_router")


class LLMRouter:
    """
    Routes requests to primary LLM with backup escalation.
    
    v2.1.0: Restores full method API from v1.0.5 while keeping
    v2.0.0 DeepSeek + Claude routing architecture.
    """
    
    def __init__(self, config_path: str = "llm_services/llm_server_config.json"):
        """Initialize router with config."""
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.primary_endpoint = f"http://localhost:{self.config['primary']['port']}/completion"
        self.escalation_triggers = self.config['routing']['escalation_triggers']
        self.metrics = LLMMetrics()
        self.current_agent = None
        
        # Setup logging (restored from v1.0.5)
        self._setup_logging()
    
    # ══════════════════════════════════════════════════════════════════════════
    # Restored from v1.0.5: Logging & Agent Identity
    # ══════════════════════════════════════════════════════════════════════════
    
    def _setup_logging(self):
        """Setup rotating log files (restored from v1.0.5)."""
        log_dir = Path("logs/llm")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        handler = RotatingFileHandler(
            log_dir / "llm_router.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        handler.setFormatter(logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s'
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def set_agent(self, agent_name: str):
        """Set current agent identity for logging (restored from v1.0.5)."""
        self.current_agent = agent_name
    
    def _add_agent_header(self, prompt: str, agent: str = None) -> str:
        """Add agent identity header to prompt (restored from v1.0.5)."""
        agent_name = agent or self.current_agent or "unknown"
        return f"<agent: {agent_name}>\n\n{prompt}"
    
    # ══════════════════════════════════════════════════════════════════════════
    # Core Routing (from v2.0.0)
    # ══════════════════════════════════════════════════════════════════════════
    
    def route(self, prompt: str, force_backup: bool = False,
              temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """
        Route request to primary, escalate to backup if needed.
        
        Returns:
            str: Raw LLM response text (never dict)
        """
        if force_backup:
            return self._call_backup(prompt)
        
        # Try primary first
        response = self._call_primary(prompt, temperature, max_tokens)
        
        # Check for escalation triggers
        if self._should_escalate(response):
            self.metrics.escalations += 1
            self.metrics.trace.append({
                "action": "escalate",
                "reason": "trigger_detected",
                "timestamp": time.time()
            })
            return self._call_backup(prompt)
        
        return response
    
    def _call_primary(self, prompt: str, temperature: float, max_tokens: int) -> str:
        """Call DeepSeek-R1-14B via llama.cpp server."""
        
        # ChatML format for DeepSeek-R1
        formatted = f"<|im_start|>system\nYou are an expert AI assistant for PRNG analysis.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        try:
            response = requests.post(
                self.primary_endpoint,
                json={
                    "prompt": formatted,
                    "n_predict": max_tokens,
                    "temperature": temperature,
                    "stop": self.config['primary']['stop_tokens'],
                    "stream": False
                },
                timeout=self.config['routing']['request_timeout_seconds']
            )
            response.raise_for_status()
            result = response.json()
            
            content = result.get("content", "").strip()
            self.metrics.primary_calls += 1
            self.metrics.total_tokens += result.get("tokens_predicted", len(content.split()))
            
            self.metrics.trace.append({
                "model": "primary",
                "tokens": result.get("tokens_predicted", 0),
                "timestamp": time.time()
            })
            
            return content
            
        except Exception as e:
            # Primary failed, try backup
            self.metrics.trace.append({
                "action": "primary_failed",
                "error": str(e),
                "timestamp": time.time()
            })
            return self._call_backup(prompt)
    
    def _call_backup(self, prompt: str) -> str:
        """Call Claude Opus 4.5 via Claude Code CLI."""
        
        try:
            working_dir = os.path.expanduser(self.config['backup']['working_dir'])
            
            result = subprocess.run(
                ['claude', '--print', '-p', prompt],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=working_dir
            )
            
            if result.returncode != 0:
                raise Exception(f"Claude Code error: {result.stderr}")
            
            content = result.stdout.strip()
            self.metrics.backup_calls += 1
            self.metrics.total_tokens += len(content) // 4  # Rough estimate
            
            self.metrics.trace.append({
                "model": "backup",
                "tokens": len(content) // 4,
                "timestamp": time.time()
            })
            
            return content
            
        except Exception as e:
            return f"[ERROR] Both primary and backup failed: {e}"
    
    def _should_escalate(self, response: str) -> bool:
        """Check if response contains escalation triggers."""
        response_upper = response.upper()
        return any(trigger in response_upper for trigger in self.escalation_triggers)
    
    def _is_primary_available(self) -> bool:
        """Check if primary endpoint is available."""
        try:
            resp = requests.get(
                self.primary_endpoint.replace("/completion", "/health"),
                timeout=2
            )
            return resp.status_code == 200
        except:
            return False
    
    # ══════════════════════════════════════════════════════════════════════════
    # NEW in v2.1.0: Grammar-Constrained Decision Evaluation
    # ══════════════════════════════════════════════════════════════════════════
    
    def _call_primary_with_grammar(
        self, 
        prompt: str, 
        grammar: str = "watcher_decision.gbnf",
        temperature: float = 0.3,
        max_tokens: int = 512
    ) -> str:
        """
        Call primary LLM with GBNF grammar constraint.
        
        Args:
            prompt: Input prompt
            grammar: Grammar filename in grammars/ directory
            temperature: Lower for consistent output (default 0.3)
            max_tokens: Limit for decision responses
            
        Returns:
            str: Grammar-constrained response (should be valid JSON)
        """
        # Load grammar
        grammar_path = Path("grammars") / grammar
        if not grammar_path.exists():
            logger.warning(f"Grammar not found: {grammar_path}, falling back to no grammar")
            return self._call_primary(prompt, temperature, max_tokens)
        
        grammar_content = grammar_path.read_text()
        
        # ChatML format with JSON instruction
        formatted = (
            "<|im_start|>system\n"
            "You are an expert AI assistant. Output valid JSON only, no markdown or explanation.\n"
            "<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        try:
            response = requests.post(
                self.primary_endpoint,
                json={
                    "prompt": formatted,
                    "n_predict": max_tokens,
                    "temperature": temperature,
                    "grammar": grammar_content,
                    "stop": self.config['primary']['stop_tokens'],
                    "stream": False
                },
                timeout=self.config['routing']['request_timeout_seconds']
            )
            response.raise_for_status()
            result = response.json()
            
            content = result.get("content", "").strip()
            self.metrics.primary_calls += 1
            self.metrics.total_tokens += result.get("tokens_predicted", len(content.split()))
            
            self.metrics.trace.append({
                "model": "primary_grammar",
                "grammar": grammar,
                "tokens": result.get("tokens_predicted", 0),
                "timestamp": time.time()
            })
            
            logger.info(f"Grammar-constrained response ({grammar}): {content[:100]}...")
            return content
            
        except Exception as e:
            logger.error(f"Grammar call failed: {e}")
            self.metrics.trace.append({
                "action": "grammar_call_failed",
                "error": str(e),
                "timestamp": time.time()
            })
            # Fallback to backup (no grammar available there)
            return self._call_backup(prompt)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parse JSON with multiple fallback strategies.
        
        SAFETY RULE:
        If parsing fails or backup LLM output is ambiguous,
        default to {"decision": "escalate"}.
        
        Order:
        1. Direct JSON parse
        2. Extract from ```json``` code block
        3. Extract {...} from text
        4. Return safe default (escalate)
        """
        response = response.strip()
        
        # Strategy 1: Direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Strategy 2: Extract from markdown code block
        if "```" in response:
            try:
                # Find content between ```json and ```
                match = re.search(r'```(?:json)?\s*(.*?)```', response, re.DOTALL)
                if match:
                    return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass
        
        # Strategy 3: Extract {...} from text
        try:
            match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        
        # Strategy 4: Try to find nested JSON object
        try:
            # More aggressive: find outermost { }
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(response[start:end+1])
        except json.JSONDecodeError:
            pass
        
        # SAFETY RULE: If parsing fails, default to escalate
        logger.warning(f"JSON parsing failed, returning safe default. Response: {response[:200]}")
        return {
            "decision": "escalate",
            "confidence": 0.3,
            "reasoning": "Failed to parse LLM response - escalating for safety",
            "retry_reason": None,
            "primary_signal": "parse_failure",
            "suggested_params": None,
            "warnings": ["LLM response could not be parsed as JSON"],
            "checks": {
                "used_rates": False,
                "mentioned_data_source": False,
                "avoided_absolute_only": False
            }
        }
    
    def evaluate_watcher_decision(
        self, 
        prompt: str, 
        step_id: str = None, 
        agent: str = None,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Evaluate watcher decision with GBNF grammar constraint.
        
        Primary path (DeepSeek via llama.cpp):
            - Uses grammar=watcher_decision.gbnf for guaranteed valid JSON
            - Near-zero parse failures
            
        Backup path (Claude):
            - Grammar not available
            - Uses _parse_json_response() fallback
            
        Args:
            prompt: Full evaluation prompt with metrics
            step_id: Pipeline step identifier (e.g., "step1_window_optimizer")
            agent: Agent name for logging
            temperature: LLM temperature (default 0.3 for consistency)
        
        Returns:
            Dict with keys: decision, confidence, reasoning, checks, etc.
            Matches watcher_decision.gbnf schema
        """
        # Add agent header for logging
        full_prompt = self._add_agent_header(prompt, agent)
        if step_id:
            full_prompt = f"<step: {step_id}>\n{full_prompt}"
        
        # Primary path: Grammar-constrained decoding
        if self._is_primary_available():
            response = self._call_primary_with_grammar(
                full_prompt,
                grammar="watcher_decision.gbnf",
                temperature=temperature
            )
            
            # Grammar should guarantee valid JSON, but be defensive
            try:
                result = json.loads(response)
                logger.info(f"Grammar-constrained decision: {result.get('decision')} "
                           f"(confidence={result.get('confidence')})")
                return result
            except json.JSONDecodeError:
                logger.warning("Grammar response not valid JSON, using fallback parser")
                return self._parse_json_response(response)
        
        # Backup path: Claude (no grammar available)
        logger.info("Primary unavailable, using backup for decision")
        response = self._call_backup(full_prompt)
        return self._parse_json_response(response)
    
    # Compatibility alias (Team Beta recommendation #4)
    def evaluate_decision(self, prompt: str, agent: str = None, 
                          temperature: float = 0.3) -> Dict[str, Any]:
        """
        Alias for evaluate_watcher_decision().
        
        Provided for backward compatibility with older scripts/manifests
        that may reference this method name.
        """
        return self.evaluate_watcher_decision(prompt, agent=agent, temperature=temperature)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Restored from v1.0.5: High-Level Methods
    # ══════════════════════════════════════════════════════════════════════════
    
    def orchestrate(self, task: str, agent: str = None, **kwargs) -> str:
        """
        Force primary model for planning/code tasks (restored from v1.0.5).
        
        In v2.1.0 (single primary), this uses DeepSeek-R1 with:
        - Lower temperature (0.3)
        - System prompt optimized for planning
        
        Returns:
            str: Raw LLM response
        """
        prompt = self._add_agent_header(task, agent)
        return self.route(prompt, temperature=kwargs.get('temperature', 0.3), **kwargs)
    
    def calculate(self, math_query: str, agent: str = None, **kwargs) -> str:
        """
        Force primary model for math tasks (restored from v1.0.5).
        
        In v2.1.0, this is functionally equivalent to orchestrate() since
        DeepSeek-R1 handles reasoning natively. Kept for API compatibility.
        
        Returns:
            str: Raw LLM response
        """
        return self.orchestrate(math_query, agent=agent, **kwargs)
    
    def generate_json(self, spec: str, agent: str = None, **kwargs) -> Dict[str, Any]:
        """
        Generate and parse JSON from primary model (restored from v1.0.5).
        
        Returns:
            Dict[str, Any]: Parsed JSON object
        """
        prompt = f"""Generate ONLY valid JSON (no markdown, no explanation) for:
{spec}
Respond with raw JSON only."""
        
        response = self.orchestrate(prompt, agent=agent, temperature=0.3, **kwargs)
        return self._parse_json_response(response)
    
    # ══════════════════════════════════════════════════════════════════════════
    # Health & Metadata (from v2.0.0)
    # ══════════════════════════════════════════════════════════════════════════
    
    def health_check(self) -> Dict[str, bool]:
        """Check primary endpoint is responsive."""
        return {
            "primary": self._is_primary_available(),
            "backup": True  # Backup always "available" (CLI-based)
        }
    
    def get_llm_metadata(self) -> Dict[str, Any]:
        """Get metadata for schema injection."""
        return {
            "primary_model": self.config["primary"]["model"],
            "backup_model": self.config["backup"]["model"],
            "primary_calls": self.metrics.primary_calls,
            "backup_calls": self.metrics.backup_calls,
            "escalations": self.metrics.escalations,
            "total_tokens_generated": self.metrics.total_tokens,
            "llm_reasoning_trace": self.metrics.trace[-10:]
        }
    
    def reset_metrics(self):
        """Reset metrics for new run."""
        self.metrics = LLMMetrics()


# ══════════════════════════════════════════════════════════════════════════════
# Singleton Pattern
# ══════════════════════════════════════════════════════════════════════════════

_router_instance = None

def get_router(config_path: str = "llm_services/llm_server_config.json") -> LLMRouter:
    """Get or create router singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter(config_path)
    return _router_instance


def reset_router():
    """Reset the singleton (useful for testing)."""
    global _router_instance
    _router_instance = None


# ══════════════════════════════════════════════════════════════════════════════
# CLI Interface (restored from v1.0.5)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Router v2.1.0")
    parser.add_argument("--health", action="store_true", help="Check server health")
    parser.add_argument("--query", type=str, help="Send a query")
    parser.add_argument("--json", type=str, help="Generate JSON for spec")
    parser.add_argument("--agent", type=str, help="Agent identity")
    parser.add_argument("--config", type=str, help="Config file path")
    
    args = parser.parse_args()
    
    config_path = args.config or "llm_services/llm_server_config.json"
    router = LLMRouter(config_path)
    
    if args.health:
        print(json.dumps(router.health_check(), indent=2))
    
    elif args.query:
        if args.agent:
            router.set_agent(args.agent)
        response = router.route(args.query)
        print(response)
    
    elif args.json:
        result = router.generate_json(args.json, agent=args.agent)
        print(json.dumps(result, indent=2))
    
    else:
        parser.print_help()
