#!/usr/bin/env python3
"""
AI-Powered Feature Importance Interpreter
Uses local Qwen2.5-Math-7B to interpret feature importance data.
"""

import json
import requests
from pathlib import Path

def query_llm(prompt: str, port: int = 8081, max_tokens: int = 1024) -> str:
    """Query the local LLM server."""
    try:
        response = requests.post(
            f"http://localhost:{port}/completion",
            json={
                "prompt": f"<|im_start|>system\nYou are an expert data scientist analyzing machine learning model feature importance for PRNG (Pseudo-Random Number Generator) pattern detection. Provide clear, actionable insights.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                "n_predict": max_tokens,
                "temperature": 0.7,
                "stop": ["<|im_end|>", "<|im_start|>"]
            },
            timeout=120
        )
        if response.status_code == 200:
            return response.json().get('content', '').strip()
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Error connecting to LLM: {e}"


def interpret_with_ai(step5_path='feature_importance_step5.json',
                      drift_path='feature_drift_step4_to_step5.json'):
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
    
    # Prepare the data summary for the LLM
    importance = step5_data.get('feature_importance', {})
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    top_10 = sorted_features[:10]
    bottom_5 = sorted_features[-5:]
    
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

    print("=" * 60)
    print("🤖 AI FEATURE IMPORTANCE INTERPRETATION")
    print("   Model: Qwen2.5-Math-7B (localhost:8081)")
    print("=" * 60)
    print()
    print("Sending data to AI for analysis...")
    print()
    
    response = query_llm(prompt)
    
    print("-" * 60)
    print(response)
    print("-" * 60)
    print()
    print("✅ AI interpretation complete.")


if __name__ == "__main__":
    interpret_with_ai()
