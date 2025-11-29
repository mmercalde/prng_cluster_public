#!/usr/bin/env python3
"""
Debug script to run the Bayesian optimizer with verbose output
and ensure it creates the output configuration file.
"""

import subprocess
import sys
import json
import os

def run_bayesian_optimizer_debug():
    """Run the Bayesian optimizer with debug output"""
    
    print("=" * 70)
    print("DEBUG: Bayesian Window Optimizer")
    print("=" * 70)
    
    # Check if input file exists
    lottery_file = "synthetic_lottery.json"
    if not os.path.exists(lottery_file):
        print(f"❌ ERROR: {lottery_file} not found!")
        return False
    
    print(f"✅ Input file found: {lottery_file}")
    
    # Check if optimizer script exists
    optimizer_script = "window_optimizer_bayesian.py"
    if not os.path.exists(optimizer_script):
        print(f"❌ ERROR: {optimizer_script} not found!")
        return False
    
    print(f"✅ Optimizer script found: {optimizer_script}")
    
    # Run the optimizer with verbose output
    cmd = [
        "python3", optimizer_script,
        "--lottery-file", lottery_file,
        "--trials", "1",
        "--output-config", "optimal_window_config.json"
    ]
    
    print(f"\n🚀 Running command:")
    print(f"   {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        print(f"\nReturn code: {result.returncode}")
        
        # Check if output file was created
        output_file = "optimal_window_config.json"
        if os.path.exists(output_file):
            print(f"\n✅ Output file created: {output_file}")
            
            # Validate JSON
            try:
                with open(output_file, 'r') as f:
                    config = json.load(f)
                print(f"✅ Valid JSON with {len(config)} keys")
                print("\nConfiguration keys:")
                for key in config.keys():
                    print(f"  - {key}")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON: {e}")
                return False
        else:
            print(f"\n❌ Output file NOT created: {output_file}")
            print("\nPossible issues:")
            print("  1. The optimizer may have crashed before saving")
            print("  2. Permission issues in the directory")
            print("  3. The optimizer logic may not be saving the file")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ ERROR: Command timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

def check_optimizer_source():
    """Check the source code for file writing logic"""
    
    print("\n" + "=" * 70)
    print("CHECKING OPTIMIZER SOURCE CODE")
    print("=" * 70)
    
    optimizer_script = "window_optimizer_bayesian.py"
    
    if not os.path.exists(optimizer_script):
        print(f"❌ Cannot check: {optimizer_script} not found")
        return
    
    try:
        with open(optimizer_script, 'r') as f:
            content = f.read()
        
        # Look for file writing patterns
        patterns = [
            "json.dump",
            "with open",
            "output-config",
            "optimal_window_config"
        ]
        
        print("\nSearching for file writing patterns:")
        for pattern in patterns:
            if pattern in content:
                print(f"  ✅ Found: {pattern}")
                # Show context
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if pattern in line:
                        print(f"     Line {i+1}: {line.strip()}")
            else:
                print(f"  ❌ Not found: {pattern}")
        
    except Exception as e:
        print(f"❌ Error reading source: {e}")

if __name__ == "__main__":
    print("🔍 Bayesian Optimizer Debug Tool\n")
    
    # First check the source code
    check_optimizer_source()
    
    # Then run the optimizer
    print()
    success = run_bayesian_optimizer_debug()
    
    if success:
        print("\n" + "=" * 70)
        print("✅ SUCCESS: Optimizer completed and created output file")
        print("=" * 70)
        sys.exit(0)
    else:
        print("\n" + "=" * 70)
        print("❌ FAILURE: Optimizer did not create output file")
        print("=" * 70)
        print("\nNext steps:")
        print("  1. Check the STDOUT/STDERR above for error messages")
        print("  2. Verify window_optimizer_bayesian.py saves to --output-config")
        print("  3. Check file permissions in current directory")
        print("  4. Try running with --trials 0 for a quick test")
        sys.exit(1)
