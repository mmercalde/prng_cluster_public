#!/usr/bin/env python3
"""
Simple chat interface with ChatML formatting.
Does NOT modify llm_router.py - wraps it safely.
"""
import sys
import requests

def chat_chatml(prompt: str, endpoint_url: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """Send prompt with proper ChatML formatting for Qwen2.5"""
    formatted = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    try:
        response = requests.post(endpoint_url, json={
            "prompt": formatted,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["</s>", "<|im_end|>", "<|endoftext|>"]
        }, timeout=120)
        response.raise_for_status()
        return response.json().get("content", "").strip()
    except Exception as e:
        return f"Error: {e}"

def main():
    print("="*50)
    print("  DUAL-LLM CHAT (ChatML)")
    print("="*50)
    
    endpoints = {
        "coder": "http://localhost:8080/completion",
        "math": "http://localhost:8081/completion"
    }
    
    # Direct health checks with retry
    def check_health(port, retries=3):
        import time
        for i in range(retries):
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=5)
                if r.ok:
                    return True
            except:
                pass
            if i < retries - 1:
                time.sleep(1)
        return False
    
    coder_ok = check_health(8080)
    math_ok = check_health(8081)
    print(f"Coder (8080): {'✅' if coder_ok else '❌'}")
    print(f"Math  (8081): {'✅' if math_ok else '❌'}")
    print("\nCommands: /coder /math /auto /quit")
    print("="*50 + "\n")
    
    mode = "auto"
    
    # Math keywords for auto-routing
    math_keywords = ["calculate", "compute", "modulo", "residue", "prng", "seed", 
                     "probability", "equation", "solve", "math", "statistical"]
    
    while True:
        try:
            user_input = input(f"[{mode.upper()}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        
        if not user_input:
            continue
        if user_input == "/quit":
            break
        elif user_input == "/coder":
            mode = "coder"
            print("✅ Coder mode")
            continue
        elif user_input == "/math":
            mode = "math"
            print("✅ Math mode")
            continue
        elif user_input == "/auto":
            mode = "auto"
            print("✅ Auto mode")
            continue
        
        # Determine endpoint - only auto-route if mode is "auto"
        if mode == "coder":
            endpoint = endpoints["coder"]
            used = "coder"
        elif mode == "math":
            endpoint = endpoints["math"]
            used = "math"
        else:  # auto mode
            if any(kw in user_input.lower() for kw in math_keywords):
                endpoint = endpoints["math"]
                used = "math"
            else:
                endpoint = endpoints["coder"]
                used = "coder"
        
        print(f"  → routing to {used}...")
        response = chat_chatml(user_input, endpoint)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    main()
