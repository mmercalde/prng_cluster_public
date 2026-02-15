#!/usr/bin/env python3
"""
S87 Task 1: Extract _detect_hit_regression() Method

This script extracts and displays the regression detection method from
chapter_13_orchestrator.py. No analysis, just the actual code.

Usage:
    python3 explain_regression_detection.py

Expected on Zeus:
    ~/distributed_prng_analysis/chapter_13_orchestrator.py
"""

import re
import sys
from pathlib import Path


def extract_method(filepath: Path, method_name: str) -> str:
    """Extract a complete method from a Python file."""
    if not filepath.exists():
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)
    
    content = filepath.read_text()
    
    # Find the method definition
    pattern = rf'(    def {method_name}\(.*?\):.*?)(?=\n    def |\n\nclass |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    
    if not match:
        print(f"ERROR: Method {method_name} not found in {filepath}")
        sys.exit(1)
    
    return match.group(1)


def main():
    print("="*70)
    print("S87 TASK 1: Extract _detect_hit_regression() Method")
    print("="*70)
    print()
    
    orchestrator_path = Path("chapter_13_orchestrator.py")
    
    if not orchestrator_path.exists():
        print(f"ERROR: {orchestrator_path} not found")
        print("This script must be run from ~/distributed_prng_analysis on Zeus")
        sys.exit(1)
    
    print(f"Extracting method from {orchestrator_path}...")
    print()
    
    method_code = extract_method(orchestrator_path, "_detect_hit_regression")
    
    print("-" * 70)
    print("METHOD CODE:")
    print("-" * 70)
    print(method_code)
    print("-" * 70)
    print()
    
    print("INTERPRETATION:")
    print("  The method checks diagnostics for hit_rate regression.")
    print("  Look for: hit_rate_history comparisons in the code above.")
    print("  Trigger condition: recent hit_rate < previous hit_rate")
    print()
    print("SYNTHETIC DATA REQUIREMENTS:")
    print("  diagnostics must contain:")
    print("    - 'hit_rate_history' key (list)")
    print("    - At least 2 entries")
    print("    - Recent entry hit_rate < previous entry hit_rate")
    print()
    print("="*70)
    print("TASK 1 COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
