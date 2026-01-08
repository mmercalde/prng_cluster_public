#!/usr/bin/env python3
"""
LLM Router - Primary + Backup Architecture
Version: 2.0.0
Lines: ~120

Routes requests to DeepSeek-R1-14B (primary) with Claude Opus 4.5 backup.
No more dual-model math routing - R1 handles reasoning natively.

Architecture:
    Primary:  DeepSeek-R1-14B (local, port 8080, 51 tok/s)
    Backup:   Claude Opus 4.5 (Claude Code CLI, 38 tok/s)
    
Escalation triggers:
    - UNCERTAIN, LOW_CONFIDENCE, ESCALATE_TO_BACKUP, REQUIRES_DEEP_ANALYSIS
"""

import requests
import json
import subprocess
import time
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMMetrics:
    """Track LLM usage for schema metadata"""
    primary_calls: int = 0
    backup_calls: int = 0
    escalations: int = 0
    total_tokens: int = 0
    trace: List[Dict] = field(default_factory=list)


class LLMRouter:
    """Routes requests to primary LLM with backup escalation."""
    
    def __init__(self, config_path: str = "llm_services/llm_server_config.json"):
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.primary_endpoint = f"http://localhost:{self.config['primary']['port']}/completion"
        self.escalation_triggers = self.config['routing']['escalation_triggers']
        self.metrics = LLMMetrics()
    
    def route(self, prompt: str, force_backup: bool = False,
              temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Route request to primary, escalate to backup if needed."""
        
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
    
    def health_check(self) -> Dict[str, bool]:
        """Check primary endpoint is responsive."""
        try:
            resp = requests.get(
                self.primary_endpoint.replace("/completion", "/health"),
                timeout=5
            )
            return {"primary": resp.status_code == 200, "backup": True}  # Backup always "available"
        except:
            return {"primary": False, "backup": True}
    
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


# Singleton pattern
_router_instance = None

def get_router(config_path: str = "llm_services/llm_server_config.json") -> LLMRouter:
    """Get or create router singleton."""
    global _router_instance
    if _router_instance is None:
        _router_instance = LLMRouter(config_path)
    return _router_instance
