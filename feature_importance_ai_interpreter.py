#!/usr/bin/env python3
"""
AI-Powered Feature Importance Interpreter
Version: 2.0.0 (2026-01-19)

LLM Configuration:
  Primary:  DeepSeek-R1-14B (localhost:8080, Zeus GPU0+GPU1)
  Fallback: Claude API (via claude CLI or API)
  Final:    Template-based interpretation

Changelog:
  v2.0.0 - Replaced Qwen2.5-Math-7B with DeepSeek-R1-14B + Claude fallback
  v1.0.0 - Original Qwen implementation
"""
import json
import requests
import subprocess
import os
from pathlib import Path
from typing import Optional, Dict, Any


# =============================================================================
# Configuration
# =============================================================================

DEEPSEEK_ENDPOINT = "http://localhost:8080/completion"
DEEPSEEK_TIMEOUT = 120
DEEPSEEK_MAX_TOKENS = 1024
DEEPSEEK_TEMPERATURE = 0.7

# =============================================================================
# LLM Query Functions
# =============================================================================

def check_deepseek_available() -> bool:
    """Check if DeepSeek server is running."""
    try:
        resp = requests.get("http://localhost:8080/health", timeout=5)
        return resp.status_code == 200
    except:
        return False


def check_claude_available() -> bool:
    """Check if Claude CLI is available."""
    result = subprocess.run(["which", "claude"], capture_output=True)
    return result.returncode == 0


def query_deepseek(prompt: str, max_tokens: int = DEEPSEEK_MAX_TOKENS) -> Optional[str]:
    """Query DeepSeek-R1-14B on localhost:8080."""
    try:
        system_prompt = (
            "You are an expert data scientist analyzing machine learning model "
            "feature importance for PRNG (Pseudo-Random Number Generator) pattern "
            "detection. Provide clear, actionable insights."
        )
        
        response = requests.post(
            DEEPSEEK_ENDPOINT,
            json={
                "prompt": f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:",
                "n_predict": max_tokens,
                "temperature": DEEPSEEK_TEMPERATURE,
                "stop": ["</s>", "<|endoftext|>", "User:"]
            },
            timeout=DEEPSEEK_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json().get('content', '').strip()
        else:
            print(f"  DeepSeek returned status {response.status_code}")
            return None
            
    except requests.exceptions.Timeout:
        print("  DeepSeek request timed out")
        return None
    except requests.exceptions.ConnectionError:
        print("  DeepSeek server not reachable")
        return None
    except Exception as e:
        print(f"  DeepSeek error: {e}")
        return None


def query_claude(prompt: str) -> Optional[str]:
    """Query Claude via CLI as fallback."""
    try:
        # Use claude CLI with prompt via stdin
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"  Claude CLI error: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("  Claude request timed out")
        return None
    except FileNotFoundError:
        print("  Claude CLI not found")
        return None
    except Exception as e:
        print(f"  Claude error: {e}")
        return None


def generate_template_interpretation(
    top_features: list,
    bottom_features: list,
    total_features: int,
    drift_data: Optional[Dict] = None
) -> str:
    """Generate template-based interpretation when LLMs unavailable."""
    
    top_10_sum = sum(v for _, v in top_features[:10])
    top_feature_name = top_features[0][0] if top_features else "unknown"
    top_feature_val = top_features[0][1] if top_features else 0
    
    interpretation = []
    interpretation.append("=" * 60)
    interpretation.append("FEATURE IMPORTANCE INTERPRETATION (Template Mode)")
    interpretation.append("=" * 60)
    interpretation.append("")
    
    # Distribution health
    if top_10_sum > 0.95:
        interpretation.append("⚠️  WARNING: Top 10 features = {:.1%} of importance".format(top_10_sum))
        interpretation.append("   This suggests possible circular/leaky features.")
        interpretation.append("   Check that training target is holdout_hits, not score.")
    elif top_10_sum > 0.80:
        interpretation.append("📊 MODERATE concentration: Top 10 = {:.1%}".format(top_10_sum))
        interpretation.append("   Model relies heavily on a few features.")
    else:
        interpretation.append("✅ HEALTHY distribution: Top 10 = {:.1%}".format(top_10_sum))
        interpretation.append("   Importance spread across multiple features.")
    
    interpretation.append("")
    interpretation.append(f"Top feature: {top_feature_name} ({top_feature_val:.1%})")
    interpretation.append(f"Total features: {total_features}")
    
    # Drift analysis
    if drift_data:
        drift_score = drift_data.get('drift_score', 0)
        status = drift_data.get('status', 'unknown')
        interpretation.append("")
        interpretation.append(f"Drift Score: {drift_score:.3f} ({status})")
        
        if drift_score > 0.3:
            interpretation.append("⚠️  High drift detected - consider retraining")
        elif drift_score > 0.15:
            interpretation.append("📊 Moderate drift - monitor closely")
        else:
            interpretation.append("✅ Stable - no action needed")
    
    interpretation.append("")
    interpretation.append("=" * 60)
    interpretation.append("Note: Template interpretation. LLM servers unavailable.")
    interpretation.append("=" * 60)
    
    return "\n".join(interpretation)


def query_llm(prompt: str, max_tokens: int = DEEPSEEK_MAX_TOKENS) -> str:
    """
    Query LLM with automatic fallback chain:
    1. DeepSeek-R1-14B (primary)
    2. Claude CLI (fallback)
    3. Template response (final fallback)
    """
    
    # Try DeepSeek first
    if check_deepseek_available():
        print("  Trying DeepSeek-R1-14B (primary)...")
        result = query_deepseek(prompt, max_tokens)
        if result:
            return result
        print("  DeepSeek failed, trying fallback...")
    else:
        print("  DeepSeek not available, trying fallback...")
    
    # Try Claude
    if check_claude_available():
        print("  Trying Claude (fallback)...")
        result = query_claude(prompt)
        if result:
            return result
        print("  Claude failed, using template...")
    else:
        print("  Claude not available, using template...")
    
    # Return None to signal template needed
    return None


# =============================================================================
# Main Interpretation Function
# =============================================================================

def interpret_with_ai(
    step5_path: str = 'feature_importance_step5.json',
    drift_path: str = 'feature_drift_step4_to_step5.json'
):
    """Use AI to interpret feature importance data."""
    
    # Load data
    step5_data = None
    drift_data = None
    
    if Path(step5_path).exists():
        with open(step5_path) as f:
            step5_data = json.load(f)
    
    if Path(drift_path).exists():
        with open(drift_path) as f:
            drift_data = json.load(f)
    
    if not step5_data:
        print("❌ No feature importance data found.")
        return
    
    # Prepare the data summary
    importance = step5_data.get('feature_importance', {})
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    top_10 = sorted_features[:10]
    bottom_5 = sorted_features[-5:]
    
    # Build prompt
    prompt = f"""Analyze this ML model's feature importance for PRNG pattern detection:

**TOP 10 FEATURES (most important):**
{json.dumps(dict(top_10), indent=2)}

**BOTTOM 5 FEATURES (least important):**
{json.dumps(dict(bottom_5), indent=2)}

**TOTAL FEATURES:** {len(sorted_features)}
"""
    
    if drift_data:
        prompt += f"""
**DRIFT ANALYSIS (Step 4 → Step 5):**
- Drift Score: {drift_data.get('drift_score', 'N/A')}
- Status: {drift_data.get('status', 'N/A')}
- Top Gainers: {drift_data.get('top_gainers', [])}
- Top Losers: {drift_data.get('top_losers', [])}
"""
    
    prompt += """
Please provide:
1. What do the top features tell us about how the PRNG can be predicted?
2. Are there any concerning patterns or feature combinations?
3. What does the drift analysis suggest about model stability?
4. Recommendations for improving prediction accuracy.

Keep your response concise and actionable."""
    
    # Print header
    print("=" * 60)
    print("🤖 AI FEATURE IMPORTANCE INTERPRETATION")
    print("=" * 60)
    print()
    print("LLM Priority:")
    print("  1. DeepSeek-R1-14B (localhost:8080)")
    print("  2. Claude CLI (fallback)")
    print("  3. Template (final fallback)")
    print()
    print("Querying LLM...")
    print()
    
    # Query with fallback chain
    response = query_llm(prompt)
    
    if response:
        print("-" * 60)
        print(response)
        print("-" * 60)
    else:
        # Use template fallback
        template_response = generate_template_interpretation(
            top_10, bottom_5, len(sorted_features), drift_data
        )
        print(template_response)
    
    print()
    print("✅ AI interpretation complete.")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='AI-Powered Feature Importance Interpreter v2.0'
    )
    parser.add_argument(
        '--step5', 
        default='feature_importance_step5.json',
        help='Path to Step 5 feature importance JSON'
    )
    parser.add_argument(
        '--drift',
        default='feature_drift_step4_to_step5.json',
        help='Path to drift analysis JSON'
    )
    parser.add_argument(
        '--test-llm',
        action='store_true',
        help='Test LLM connectivity without interpretation'
    )
    
    args = parser.parse_args()
    
    if args.test_llm:
        print("Testing LLM connectivity...")
        print()
        print(f"DeepSeek-R1-14B: {'✅ Available' if check_deepseek_available() else '❌ Not available'}")
        print(f"Claude CLI:      {'✅ Available' if check_claude_available() else '❌ Not available'}")
    else:
        interpret_with_ai(args.step5, args.drift)
