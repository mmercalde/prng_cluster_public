#!/usr/bin/env python3
"""
LLM API Test: Claude vs DeepSeek Cloud
Runs the same prompt against cloud APIs for comparison with local models.

Usage:
    python3 llm_api_test.py --model claude
    python3 llm_api_test.py --model deepseek
    python3 llm_api_test.py --compare

Requires:
    export ANTHROPIC_API_KEY="your-key"
    export DEEPSEEK_API_KEY="your-key"
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

# Configuration
PROMPT_FILE = "/home/michael/distributed_prng_analysis/llm_ab_test_prompt.txt"
RESULTS_DIR = Path("/home/michael/distributed_prng_analysis/llm_test_results")

# API endpoints
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def get_api_key(provider: str) -> str:
    """Get API key from environment."""
    if provider == "claude":
        # Claude Code uses Max subscription, no API key needed
        return None
    elif provider == "deepseek":
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY not set. Run: export DEEPSEEK_API_KEY='your-key'")
        return key
    else:
        raise ValueError(f"Unknown provider: {provider}")

def call_claude_api(prompt: str, api_key: str = None, max_tokens: int = 4096) -> dict:
    """Call Claude via Claude Code CLI (uses Max subscription)."""
    import subprocess
    import tempfile
    
    start_time = time.time()
    
    try:
        # Write prompt to temp file to avoid shell escaping issues
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(prompt)
            prompt_file = f.name
        
        # Call claude CLI with --print flag for non-interactive output
        result = subprocess.run(
            ['claude', '--print', '-p', prompt],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=os.path.expanduser('~/claude_test')  # Use test directory
        )
        
        # Clean up temp file
        os.unlink(prompt_file)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        if result.returncode != 0:
            return {"success": False, "error": f"Claude Code error: {result.stderr}"}
        
        content = result.stdout.strip()
        # Estimate tokens (rough: 1 token ≈ 4 chars)
        output_tokens = len(content) // 4
        
        return {
            "success": True,
            "content": content,
            "tokens_predicted": output_tokens,
            "input_tokens": len(prompt) // 4,
            "elapsed_seconds": elapsed,
            "tokens_per_second": output_tokens / elapsed if elapsed > 0 else 0,
            "model": "claude-opus-4.5 (via Claude Code)",
            "raw_response": {"stdout": content, "stderr": result.stderr}
        }
        
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Claude Code timed out after 5 minutes"}
    except FileNotFoundError:
        return {"success": False, "error": "Claude Code not found. Run: npm install -g @anthropic-ai/claude-code && claude login"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def call_deepseek_api(prompt: str, api_key: str, max_tokens: int = 4096) -> dict:
    """Call DeepSeek API."""
    import requests
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "deepseek-reasoner",  # R1 model
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=300)
        response.raise_for_status()
        result = response.json()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        content = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        output_tokens = usage.get("completion_tokens", len(content.split()))
        input_tokens = usage.get("prompt_tokens", 0)
        
        return {
            "success": True,
            "content": content,
            "tokens_predicted": output_tokens,
            "input_tokens": input_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": output_tokens / elapsed if elapsed > 0 else 0,
            "model": "deepseek-reasoner",
            "raw_response": result
        }
        
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"HTTP Error: {e.response.status_code} - {e.response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def analyze_response(content: str) -> dict:
    """Analyze response quality metrics."""
    # Check for thinking tags (DeepSeek R1 style)
    has_thinking = "<think>" in content and "</think>" in content
    
    if has_thinking:
        think_start = content.find("<think>")
        think_end = content.find("</think>") + len("</think>")
        thinking = content[think_start:think_end]
        answer = content[:think_start] + content[think_end:]
    else:
        thinking = ""
        answer = content
    
    # Count sections addressed
    sections = {
        "state_variables": any(kw in answer.lower() for kw in ["state variable", "invariant", "seed", "survivor"]),
        "information_loss": any(kw in answer.lower() for kw in ["mod 1000", "collision", "information loss", "projection"]),
        "leverage_params": any(kw in answer.lower() for kw in ["threshold", "skip", "window", "parameter"]),
        "experiment_plan": any(kw in answer.lower() for kw in ["experiment", "step", "validate", "test"]),
        "assumptions": any(kw in answer.lower() for kw in ["assumption", "risk", "edge case", "fail"])
    }
    
    return {
        "has_thinking": has_thinking,
        "thinking_length": len(thinking),
        "answer_length": len(answer),
        "total_length": len(content),
        "sections_addressed": sum(sections.values()),
        "section_details": sections,
        "word_count": len(content.split()),
        "paragraph_count": content.count("\n\n") + 1
    }

def run_api_test(model_name: str):
    """Run the API test for a model."""
    print(f"\n{'='*60}")
    print(f"Running API Test: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load prompt
    if not os.path.exists(PROMPT_FILE):
        print(f"ERROR: Prompt file not found: {PROMPT_FILE}")
        return None
    
    with open(PROMPT_FILE, 'r') as f:
        prompt = f.read()
    
    print(f"Prompt loaded: {len(prompt)} characters")
    
    # Get API key (None for Claude Code)
    try:
        api_key = get_api_key(model_name)
    except ValueError as e:
        print(f"ERROR: {e}")
        return None
    
    if model_name == "claude":
        print("Using Claude Code CLI (Max subscription)...")
    else:
        print(f"API key found. Calling {model_name} API...")
    print("-" * 40)
    
    # Call appropriate API
    if model_name == "claude":
        result = call_claude_api(prompt, api_key)
    elif model_name == "deepseek":
        result = call_deepseek_api(prompt, api_key)
    else:
        print(f"ERROR: Unknown model: {model_name}")
        return None
    
    if not result["success"]:
        print(f"ERROR: {result['error']}")
        return None
    
    # Analyze response
    analysis = analyze_response(result["content"])
    
    # Print summary
    print(f"\n{'='*40}")
    print(f"RESULTS: {model_name.upper()} API")
    print(f"{'='*40}")
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Time: {result['elapsed_seconds']:.1f} seconds")
    print(f"Input tokens: {result.get('input_tokens', 'N/A')}")
    print(f"Output tokens: {result['tokens_predicted']}")
    print(f"Speed: {result['tokens_per_second']:.1f} tok/s")
    print(f"Response length: {analysis['total_length']} chars")
    print(f"Has thinking: {analysis['has_thinking']}")
    print(f"Sections addressed: {analysis['sections_addressed']}/5")
    print(f"Section coverage: {analysis['section_details']}")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        "model": f"{model_name}_api",
        "model_version": result.get("model", "unknown"),
        "timestamp": timestamp,
        "prompt_file": PROMPT_FILE,
        "inference": {
            "elapsed_seconds": result["elapsed_seconds"],
            "tokens_predicted": result["tokens_predicted"],
            "input_tokens": result.get("input_tokens", 0),
            "tokens_per_second": result["tokens_per_second"]
        },
        "analysis": analysis,
        "response": result["content"]
    }
    
    output_file = RESULTS_DIR / f"{model_name}_api_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {output_file}")
    
    # Also save raw response as text
    text_file = RESULTS_DIR / f"{model_name}_api_{timestamp}_response.txt"
    with open(text_file, 'w') as f:
        f.write(f"MODEL: {model_name} API ({result.get('model', 'unknown')})\n")
        f.write(f"TIME: {result['elapsed_seconds']:.1f}s @ {result['tokens_per_second']:.1f} tok/s\n")
        f.write(f"{'='*60}\n\n")
        f.write(result["content"])
    print(f"Response saved: {text_file}")
    
    return output

def print_full_comparison():
    """Print comparison of all results (local + API)."""
    print(f"\n{'='*70}")
    print("FULL COMPARISON: Local vs API Models")
    print(f"{'='*70}")
    
    models = ["14b", "32b", "claude_api", "deepseek_api"]
    results = {}
    
    for model in models:
        files = sorted(RESULTS_DIR.glob(f"{model}_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                results[model] = json.load(f)
    
    if len(results) < 2:
        print("Need at least 2 results to compare. Run tests first.")
        return
    
    # Header
    cols = [m for m in models if m in results]
    header = f"{'Metric':<25}" + "".join(f"{m:<15}" for m in cols)
    print(f"\n{header}")
    print("-" * (25 + 15 * len(cols)))
    
    # Metrics
    for key in ["elapsed_seconds", "tokens_predicted", "tokens_per_second"]:
        row = f"{key:<25}"
        for m in cols:
            val = results[m]["inference"].get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:<15.2f}"
            else:
                row += f"{val:<15}"
        print(row)
    
    print("-" * (25 + 15 * len(cols)))
    
    for key in ["sections_addressed", "total_length", "has_thinking"]:
        row = f"{key:<25}"
        for m in cols:
            val = results[m]["analysis"].get(key, "N/A")
            row += f"{str(val):<15}"
        print(row)
    
    # Section coverage
    print("\n" + "="*70)
    print("Section Coverage:")
    print("-" * (25 + 15 * len(cols)))
    
    for section in ["state_variables", "information_loss", "leverage_params", "experiment_plan", "assumptions"]:
        row = f"{section:<25}"
        for m in cols:
            val = "✅" if results[m]["analysis"]["section_details"].get(section) else "❌"
            row += f"{val:<15}"
        print(row)

def main():
    parser = argparse.ArgumentParser(description="LLM API Test: Claude vs DeepSeek")
    parser.add_argument("--model", choices=["claude", "deepseek"], help="API to test")
    parser.add_argument("--compare", action="store_true", help="Show full comparison (local + API)")
    args = parser.parse_args()
    
    if args.compare:
        print_full_comparison()
    elif args.model:
        run_api_test(args.model)
    else:
        print("LLM API Test Harness")
        print("="*40)
        print("\nUsage:")
        print("  1. Set API keys:")
        print("     export ANTHROPIC_API_KEY='your-key'")
        print("     export DEEPSEEK_API_KEY='your-key'")
        print("\n  2. Run tests:")
        print("     python3 llm_api_test.py --model claude")
        print("     python3 llm_api_test.py --model deepseek")
        print("\n  3. Compare all results:")
        print("     python3 llm_api_test.py --compare")

if __name__ == "__main__":
    main()
