#!/usr/bin/env python3
"""
Test WATCHER Agent - Step 1 (Window Optimizer)
Tests autonomous decision-making with DeepSeek-R1-14B (on-demand)

Usage:
    python3 test_watcher_step1.py --no-llm          # Heuristic only
    python3 test_watcher_step1.py --llm             # With LLM (starts server)
    python3 test_watcher_step1.py --mock-results    # Use mock data
"""

import json
import argparse
import subprocess
import time
import requests
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
from enum import Enum
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

class DecisionAction(str, Enum):
    PROCEED = "proceed"
    RETRY = "retry"
    ESCALATE = "escalate"


@dataclass
class WatcherConfig:
    """Configuration for WATCHER Agent."""
    auto_proceed_threshold: float = 0.70
    escalate_threshold: float = 0.50
    max_retries_per_step: int = 3
    use_llm: bool = False
    llm_port: int = 8080
    llm_startup_timeout: int = 60
    project_dir: str = "/home/michael/distributed_prng_analysis"


@dataclass
class AgentDecision:
    """Decision output from WATCHER."""
    action: DecisionAction
    confidence: float
    reasoning: str
    suggested_params: Optional[Dict] = None
    model_used: str = "heuristic"
    escalated: bool = False


# ============================================================================
# STEP 1 HEURISTICS
# ============================================================================

def evaluate_step1_heuristic(results: Dict[str, Any]) -> AgentDecision:
    """
    Heuristic evaluation for Step 1 (Window Optimizer).
    
    Key metrics:
    - bidirectional_count: Number of bidirectional survivors (primary)
    - survivor_count: Total survivors (secondary)
    - optimization_score: Best trial score
    """
    
    bidirectional = results.get('bidirectional_count', 0)
    survivors = results.get('survivor_count', results.get('forward_count', 0))
    score = results.get('optimization_score', 0)
    
    # Calculate confidence based on bidirectional survivors
    if bidirectional > 10000:
        confidence = 0.95
        action = DecisionAction.PROCEED
        reasoning = f"Excellent: {bidirectional:,} bidirectional survivors found"
    elif bidirectional > 1000:
        confidence = 0.85
        action = DecisionAction.PROCEED
        reasoning = f"Good: {bidirectional:,} bidirectional survivors found"
    elif bidirectional > 100:
        confidence = 0.70
        action = DecisionAction.PROCEED
        reasoning = f"Acceptable: {bidirectional:,} bidirectional survivors, proceed with caution"
    elif bidirectional > 10:
        confidence = 0.55
        action = DecisionAction.RETRY
        reasoning = f"Marginal: Only {bidirectional} bidirectional survivors, consider retry with adjusted thresholds"
    else:
        confidence = 0.30
        action = DecisionAction.ESCALATE
        reasoning = f"Poor: Only {bidirectional} bidirectional survivors, needs human review"
    
    # Adjust for forward-only survivors (fallback signal)
    if survivors > 5000 and bidirectional < 100:
        confidence = min(confidence + 0.10, 0.65)
        reasoning += f" (but {survivors:,} forward survivors available)"
    
    # Suggested params for retry
    suggested = None
    if action == DecisionAction.RETRY:
        suggested = {
            "forward_threshold": max(0.60, results.get('forward_threshold', 0.72) - 0.05),
            "reverse_threshold": max(0.70, results.get('reverse_threshold', 0.81) - 0.05),
            "comment": "Lower thresholds to increase survivor count"
        }
    
    return AgentDecision(
        action=action,
        confidence=confidence,
        reasoning=reasoning,
        suggested_params=suggested,
        model_used="heuristic"
    )


# ============================================================================
# LLM INTEGRATION (ON-DEMAND)
# ============================================================================

class OnDemandLLM:
    """On-demand LLM server management."""
    
    def __init__(self, config: WatcherConfig):
        self.config = config
        self.project_dir = Path(config.project_dir)
        self.port = config.llm_port
        
    def is_running(self) -> bool:
        """Check if LLM server is running."""
        try:
            r = requests.get(f"http://localhost:{self.port}/health", timeout=2)
            return r.status_code == 200
        except:
            return False
    
    def start(self) -> bool:
        """Start LLM server (partitioned across both GPUs)."""
        if self.is_running():
            print("  LLM server already running")
            return True
        
        print("  Starting DeepSeek-R1-14B (partitioned across GPU0 + GPU1)...")
        
        # Use the start script if available, otherwise direct command
        start_script = self.project_dir / "llm_services" / "start_llm_server.sh"
        
        if False:  # Disabled - use direct command
            subprocess.Popen(
                ["bash", str(start_script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            # Direct llama-server command
            model_path = self.project_dir / "models" / "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf"
            subprocess.Popen([
                "/home/michael/llama.cpp/build/bin/llama-server",
                "--model", str(model_path),
                "--port", str(self.port),
                "--ctx-size", "16384",
                "--batch-size", "512",
                "--n-gpu-layers", "99",
                "--tensor-split", "0.5,0.5",
                "--flash-attn", "on"
            ], stdout=open("/tmp/llm_server.log", "w"), stderr=subprocess.STDOUT)
        
        # Wait for startup
        start_time = time.time()
        while time.time() - start_time < self.config.llm_startup_timeout:
            if self.is_running():
                print(f"  ✅ LLM ready in {time.time() - start_time:.1f}s")
                return True
            time.sleep(1)
            print(".", end="", flush=True)
        
        print("\n  ❌ LLM failed to start")
        return False
    
    def stop(self):
        """Stop LLM server to free GPUs."""
        print("  Stopping LLM server...")
        subprocess.run(
            ["pkill", "-f", f"llama-server.*port {self.port}"],
            capture_output=True
        )
        time.sleep(2)
        print("  ✅ GPUs freed")
    
    def query(self, prompt: str) -> str:
        """Query LLM."""
        try:
            response = requests.post(
                f"http://localhost:{self.port}/completion",
                json={
                    "prompt": prompt,
                    "temperature": 0.7,
                    "n_predict": 500,
                    "stop": ["```", "</response>", "\n\n\n"]
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()['content']
        except Exception as e:
            return f'{{"error": "{str(e)}", "decision": "UNCERTAIN"}}'


def build_step1_prompt(results: Dict[str, Any]) -> str:
    """Build LLM prompt for Step 1 evaluation."""
    
    return f"""# MISSION CONTEXT: Step 1 - Window Optimizer

## YOUR ROLE
You are evaluating the results of Bayesian window optimization (Step 1).
Your job is to determine if the optimization found sufficient bidirectional survivors.

## MATHEMATICAL CONTEXT
- Bidirectional survivors have P(false positive) ≈ 10⁻¹¹⁹¹
- Every survivor is mathematically significant
- Threshold philosophy: LOW thresholds (0.001-0.10) for DISCOVERY
- Intersection performs actual filtering

## DECISION CRITERIA
| Metric | PROCEED | RETRY | ESCALATE |
|--------|---------|-------|----------|
| bidirectional_count | > 1000 | 100-1000 | < 100 |
| bidirectional_rate | < 0.02 | 0.02-0.05 | > 0.05 |

## CURRENT RESULTS

```json
{json.dumps(results, indent=2)}
```

## YOUR DECISION

Based on these results, respond with ONLY a JSON object (no markdown, no explanation):
{{"decision": "PROCEED" or "RETRY" or "ESCALATE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""


def evaluate_step1_llm(results: Dict[str, Any], llm: OnDemandLLM) -> AgentDecision:
    """LLM-based evaluation for Step 1."""
    
    prompt = build_step1_prompt(results)
    response = llm.query(prompt)
    
    # Parse response
    try:
        # Clean up response
        response = response.strip()
        # Strip DeepSeek-R1 thinking tags
        if "</think>" in response:
            response = response.split("</think>")[-1].strip()
        if response.startswith("```"):
            response = response.split("```")[1]
            if response.startswith("json"):
                response = response[4:]
        
        # Extract JSON from prose if needed
        import re
        json_match = re.search(r'{[^{}]*"decision"[^{}]*}', response)
        if json_match:
            response = json_match.group()
        data = json.loads(response)
        
        action = DecisionAction(data.get('decision', 'ESCALATE').lower())
        confidence = float(data.get('confidence', 0.5))
        reasoning = data.get('reasoning', 'LLM response')
        
        return AgentDecision(
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            model_used="deepseek_r1_14b"
        )
        
    except Exception as e:
        print(f"  ⚠️ LLM parse error: {e}")
        print(f"  Raw response: {response[:200]}...")
        # Fall back to heuristic
        result = evaluate_step1_heuristic(results)
        result.reasoning = f"[LLM fallback] {result.reasoning}"
        return result


# ============================================================================
# WATCHER AGENT
# ============================================================================

class WatcherAgent:
    """WATCHER Agent for autonomous pipeline decisions."""
    
    def __init__(self, config: WatcherConfig):
        self.config = config
        self.llm = OnDemandLLM(config) if config.use_llm else None
        
    def evaluate_step1(self, results: Dict[str, Any]) -> AgentDecision:
        """Evaluate Step 1 results."""
        
        if self.config.use_llm and self.llm:
            # Start LLM on-demand
            if self.llm.start():
                try:
                    decision = evaluate_step1_llm(results, self.llm)
                finally:
                    # Always free GPUs after query
                    self.llm.stop()
                return decision
            else:
                print("  ⚠️ LLM unavailable, using heuristic")
        
        return evaluate_step1_heuristic(results)
    
    def should_proceed(self, decision: AgentDecision) -> bool:
        """Check if we should auto-proceed."""
        return (
            decision.action == DecisionAction.PROCEED and
            decision.confidence >= self.config.auto_proceed_threshold
        )
    
    def should_escalate(self, decision: AgentDecision) -> bool:
        """Check if we should escalate to human."""
        return (
            decision.action == DecisionAction.ESCALATE or
            decision.confidence < self.config.escalate_threshold
        )


# ============================================================================
# MOCK DATA
# ============================================================================

def get_mock_step1_results(scenario: str = "good") -> Dict[str, Any]:
    """Generate mock Step 1 results for testing."""
    
    scenarios = {
        "excellent": {
            "window_size": 256,
            "offset": 50,
            "skip_min": 0,
            "skip_max": 30,
            "prng_type": "java_lcg",
            "seed_count": 10000000,
            "optimization_score": 15847.0,
            "forward_count": 45231,
            "reverse_count": 38654,
            "bidirectional_count": 15847,
            "forward_threshold": 0.72,
            "reverse_threshold": 0.81,
            "run_id": f"step1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_test",
        },
        "good": {
            "window_size": 256,
            "offset": 50,
            "skip_min": 0,
            "skip_max": 30,
            "prng_type": "java_lcg",
            "seed_count": 10000000,
            "optimization_score": 2847.0,
            "forward_count": 12543,
            "reverse_count": 9876,
            "bidirectional_count": 2847,
            "forward_threshold": 0.72,
            "reverse_threshold": 0.81,
            "run_id": f"step1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_test",
        },
        "marginal": {
            "window_size": 256,
            "offset": 50,
            "skip_min": 0,
            "skip_max": 30,
            "prng_type": "mt19937",
            "seed_count": 10000000,
            "optimization_score": 47.0,
            "forward_count": 3210,
            "reverse_count": 890,
            "bidirectional_count": 47,
            "forward_threshold": 0.72,
            "reverse_threshold": 0.81,
            "run_id": f"step1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_test",
        },
        "poor": {
            "window_size": 256,
            "offset": 50,
            "skip_min": 0,
            "skip_max": 30,
            "prng_type": "xorshift32",
            "seed_count": 10000000,
            "optimization_score": 3.0,
            "forward_count": 1250,
            "reverse_count": 340,
            "bidirectional_count": 3,
            "forward_threshold": 0.72,
            "reverse_threshold": 0.81,
            "run_id": f"step1_{datetime.now().strftime('%Y%m%d_%H%M%S')}_test",
        }
    }
    
    return scenarios.get(scenario, scenarios["good"])


# ============================================================================
# MAIN TEST
# ============================================================================

def run_test(args):
    """Run WATCHER Agent test for Step 1."""
    
    print("=" * 60)
    print("WATCHER AGENT TEST - Step 1 (Window Optimizer)")
    print("=" * 60)
    print()
    
    # Configuration
    config = WatcherConfig(
        use_llm=args.llm,
        project_dir=args.project_dir
    )
    
    print(f"Configuration:")
    print(f"  Mode: {'LLM (DeepSeek-R1-14B)' if config.use_llm else 'Heuristic'}")
    print(f"  Auto-proceed threshold: {config.auto_proceed_threshold}")
    print(f"  Escalate threshold: {config.escalate_threshold}")
    print()
    
    # Initialize WATCHER
    watcher = WatcherAgent(config)
    
    # Test scenarios
    if args.results_file:
        # Load real results
        with open(args.results_file) as f:
            scenarios = {"loaded": json.load(f)}
    else:
        # Use mock data
        scenarios = {
            "excellent": get_mock_step1_results("excellent"),
            "good": get_mock_step1_results("good"),
            "marginal": get_mock_step1_results("marginal"),
            "poor": get_mock_step1_results("poor"),
        }
    
    results_summary = []
    
    for name, results in scenarios.items():
        print(f"\n{'─' * 60}")
        print(f"Testing scenario: {name.upper()}")
        print(f"{'─' * 60}")
        print(f"  Bidirectional survivors: {results.get('bidirectional_count', 0):,}")
        print(f"  Forward survivors: {results.get('forward_count', 0):,}")
        print(f"  PRNG type: {results.get('prng_type', 'unknown')}")
        print()
        
        # Evaluate
        decision = watcher.evaluate_step1(results)
        
        # Display decision
        action_emoji = {
            DecisionAction.PROCEED: "✅",
            DecisionAction.RETRY: "🔄",
            DecisionAction.ESCALATE: "⚠️"
        }
        
        print(f"Decision: {action_emoji[decision.action]} {decision.action.value.upper()}")
        print(f"Confidence: {decision.confidence:.2f}")
        print(f"Model: {decision.model_used}")
        print(f"Reasoning: {decision.reasoning}")
        
        if decision.suggested_params:
            print(f"Suggested params: {json.dumps(decision.suggested_params, indent=2)}")
        
        # Action determination
        if watcher.should_proceed(decision):
            print(f"\n→ ACTION: Auto-proceed to Step 2 (Scorer Meta-Optimizer)")
        elif watcher.should_escalate(decision):
            print(f"\n→ ACTION: ESCALATE - Human review required")
        else:
            print(f"\n→ ACTION: RETRY with adjusted parameters")
        
        results_summary.append({
            "scenario": name,
            "decision": decision.action.value,
            "confidence": decision.confidence,
            "model": decision.model_used
        })
    
    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    
    for r in results_summary:
        status = "✅" if r["decision"] == "proceed" else ("🔄" if r["decision"] == "retry" else "⚠️")
        print(f"  {status} {r['scenario']}: {r['decision']} (conf={r['confidence']:.2f}, model={r['model']})")
    
    print()
    print("Test complete!")
    
    # Save results
    output_file = Path("./watcher_test_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": asdict(config),
            "results": results_summary
        }, f, indent=2)
    print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Test WATCHER Agent Step 1")
    parser.add_argument("--llm", action="store_true", help="Use LLM for decisions (on-demand)")
    parser.add_argument("--no-llm", action="store_true", help="Use heuristic only (default)")
    parser.add_argument("--results-file", type=str, help="Load results from JSON file")
    parser.add_argument("--project-dir", type=str, 
                        default="/home/michael/distributed_prng_analysis",
                        help="Project directory path")
    
    args = parser.parse_args()
    
    # Default to no-llm if neither specified
    if not args.llm:
        args.llm = False
    
    run_test(args)


if __name__ == "__main__":
    main()
