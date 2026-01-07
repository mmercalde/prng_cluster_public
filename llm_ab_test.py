#!/usr/bin/env python3
"""
LLM A/B Test Harness: DeepSeek-R1 14B vs 32B
Tests both models with identical prompts and measures quality + speed.

Usage:
    # First, start the model server on port 8080, then run:
    python3 llm_ab_test.py --model 14b
    python3 llm_ab_test.py --model 32b
    
    # Or run full comparison (requires manual server restart between):
    python3 llm_ab_test.py --compare
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
import requests

# Configuration
PROMPT_FILE = "/home/michael/distributed_prng_analysis/llm_ab_test_prompt.txt"
RESULTS_DIR = Path("/home/michael/distributed_prng_analysis/llm_test_results")
ENDPOINT = "http://localhost:8080/completion"

# ChatML formatting for DeepSeek-R1-Distill-Qwen
def format_chatml(prompt: str, system: str = "You are an expert AI assistant specializing in PRNG analysis, machine learning, and distributed systems.") -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def run_inference(prompt: str, max_tokens: int = 4096, temperature: float = 0.7) -> dict:
    """Run inference and return response with timing."""
    formatted = format_chatml(prompt)
    
    start_time = time.time()
    
    try:
        response = requests.post(ENDPOINT, json={
            "prompt": formatted,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["</s>", "<|im_end|>", "<|endoftext|>"],
            "stream": False
        }, timeout=300)  # 5 min timeout for long responses
        
        response.raise_for_status()
        result = response.json()
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        content = result.get("content", "").strip()
        tokens_predicted = result.get("tokens_predicted", len(content.split()))
        
        return {
            "success": True,
            "content": content,
            "tokens_predicted": tokens_predicted,
            "elapsed_seconds": elapsed,
            "tokens_per_second": tokens_predicted / elapsed if elapsed > 0 else 0,
            "raw_response": result
        }
        
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out after 5 minutes"}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to server. Is llama-server running on port 8080?"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def extract_thinking(content: str) -> tuple[str, str]:
    """Separate <think>...</think> from final answer."""
    if "<think>" in content and "</think>" in content:
        think_start = content.find("<think>")
        think_end = content.find("</think>") + len("</think>")
        thinking = content[think_start:think_end]
        answer = content[:think_start] + content[think_end:]
        return thinking.strip(), answer.strip()
    return "", content.strip()

def analyze_response(content: str) -> dict:
    """Analyze response quality metrics."""
    thinking, answer = extract_thinking(content)
    
    # Count sections addressed
    sections = {
        "state_variables": any(kw in answer.lower() for kw in ["state variable", "invariant", "seed", "survivor"]),
        "information_loss": any(kw in answer.lower() for kw in ["mod 1000", "collision", "information loss", "projection"]),
        "leverage_params": any(kw in answer.lower() for kw in ["threshold", "skip", "window", "parameter"]),
        "experiment_plan": any(kw in answer.lower() for kw in ["experiment", "step", "validate", "test"]),
        "assumptions": any(kw in answer.lower() for kw in ["assumption", "risk", "edge case", "fail"])
    }
    
    return {
        "has_thinking": len(thinking) > 0,
        "thinking_length": len(thinking),
        "answer_length": len(answer),
        "total_length": len(content),
        "sections_addressed": sum(sections.values()),
        "section_details": sections,
        "word_count": len(content.split()),
        "paragraph_count": content.count("\n\n") + 1
    }

def run_test(model_name: str):
    """Run the full test for a model."""
    print(f"\n{'='*60}")
    print(f"Running A/B Test: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Load prompt
    if not os.path.exists(PROMPT_FILE):
        print(f"ERROR: Prompt file not found: {PROMPT_FILE}")
        return
    
    with open(PROMPT_FILE, 'r') as f:
        prompt = f.read()
    
    print(f"Prompt loaded: {len(prompt)} characters")
    print(f"Sending to {ENDPOINT}...")
    print("-" * 40)
    
    # Run inference
    result = run_inference(prompt)
    
    if not result["success"]:
        print(f"ERROR: {result['error']}")
        return
    
    # Analyze response
    analysis = analyze_response(result["content"])
    
    # Print summary
    print(f"\n{'='*40}")
    print(f"RESULTS: {model_name.upper()}")
    print(f"{'='*40}")
    print(f"Time: {result['elapsed_seconds']:.1f} seconds")
    print(f"Tokens: {result['tokens_predicted']}")
    print(f"Speed: {result['tokens_per_second']:.1f} tok/s")
    print(f"Response length: {analysis['total_length']} chars")
    print(f"Has thinking: {analysis['has_thinking']}")
    print(f"Sections addressed: {analysis['sections_addressed']}/5")
    print(f"Section coverage: {analysis['section_details']}")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output = {
        "model": model_name,
        "timestamp": timestamp,
        "prompt_file": PROMPT_FILE,
        "inference": {
            "elapsed_seconds": result["elapsed_seconds"],
            "tokens_predicted": result["tokens_predicted"],
            "tokens_per_second": result["tokens_per_second"]
        },
        "analysis": analysis,
        "response": result["content"]
    }
    
    output_file = RESULTS_DIR / f"{model_name}_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {output_file}")
    
    # Also save raw response as text
    text_file = RESULTS_DIR / f"{model_name}_{timestamp}_response.txt"
    with open(text_file, 'w') as f:
        f.write(f"MODEL: {model_name}\n")
        f.write(f"TIME: {result['elapsed_seconds']:.1f}s @ {result['tokens_per_second']:.1f} tok/s\n")
        f.write(f"{'='*60}\n\n")
        f.write(result["content"])
    print(f"Response saved: {text_file}")
    
    return output

def print_comparison():
    """Print comparison of latest results."""
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    results = {}
    for model in ["14b", "32b"]:
        files = sorted(RESULTS_DIR.glob(f"{model}_*.json"), reverse=True)
        if files:
            with open(files[0]) as f:
                results[model] = json.load(f)
    
    if len(results) < 2:
        print("Need results from both models. Run tests first.")
        return
    
    print(f"\n{'Metric':<25} {'14B':<20} {'32B':<20}")
    print("-" * 65)
    
    for key in ["elapsed_seconds", "tokens_predicted", "tokens_per_second"]:
        v14 = results["14b"]["inference"].get(key, "N/A")
        v32 = results["32b"]["inference"].get(key, "N/A")
        if isinstance(v14, float):
            print(f"{key:<25} {v14:<20.2f} {v32:<20.2f}")
        else:
            print(f"{key:<25} {v14:<20} {v32:<20}")
    
    print("-" * 65)
    
    for key in ["sections_addressed", "total_length", "has_thinking"]:
        v14 = results["14b"]["analysis"].get(key, "N/A")
        v32 = results["32b"]["analysis"].get(key, "N/A")
        print(f"{key:<25} {str(v14):<20} {str(v32):<20}")
    
    print("\n" + "="*60)
    print("Section Coverage:")
    print("-" * 65)
    for section in ["state_variables", "information_loss", "leverage_params", "experiment_plan", "assumptions"]:
        v14 = "✅" if results["14b"]["analysis"]["section_details"].get(section) else "❌"
        v32 = "✅" if results["32b"]["analysis"]["section_details"].get(section) else "❌"
        print(f"{section:<25} {v14:<20} {v32:<20}")

def main():
    parser = argparse.ArgumentParser(description="LLM A/B Test Harness")
    parser.add_argument("--model", choices=["14b", "32b"], help="Model to test")
    parser.add_argument("--compare", action="store_true", help="Show comparison of results")
    args = parser.parse_args()
    
    if args.compare:
        print_comparison()
    elif args.model:
        run_test(args.model)
    else:
        print("Usage:")
        print("  1. Start 14B server, then: python3 llm_ab_test.py --model 14b")
        print("  2. Start 32B server, then: python3 llm_ab_test.py --model 32b")
        print("  3. Compare results: python3 llm_ab_test.py --compare")
        print("\nServer commands:")
        print("  14B: ~/llama.cpp/llama-server --model ~/distributed_prng_analysis/models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf --port 8080 --ctx-size 8192 --n-gpu-layers 99")
        print("  32B: ~/llama.cpp/llama-server --model ~/distributed_prng_analysis/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --port 8080 --ctx-size 8192 --n-gpu-layers 99")

if __name__ == "__main__":
    main()
