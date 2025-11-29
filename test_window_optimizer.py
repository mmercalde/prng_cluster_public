#!/usr/bin/env python3
"""
STANDALONE TEST - Window Optimizer
Tests ONE PRNG safely without affecting any existing code
"""

from modules.window_optimizer import test_window_optimizer_standalone
import sys

if __name__ == "__main__":
    prng = sys.argv[1] if len(sys.argv) > 1 else 'lcg32'
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║   STANDALONE WINDOW OPTIMIZER TEST                         ║
║   Testing: {prng:<48} ║
║   Safe: Will NOT modify any existing code or settings     ║
╚════════════════════════════════════════════════════════════╝
    """)
    
    result = test_window_optimizer_standalone(prng)
    
    print(f"""
╔════════════════════════════════════════════════════════════╗
║   TEST COMPLETE                                            ║
║   Optimal window: {result['optimal_window']:<44} ║
║   Signal strength: {result['signal_strength']:<43.2f} ║
║   Tests performed: {result['tests_performed']:<43} ║
╚════════════════════════════════════════════════════════════╝
    """)
