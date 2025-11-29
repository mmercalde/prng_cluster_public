#!/usr/bin/env python3
"""
Safe Bayesian Optimizer Patcher
Upgrades window_optimizer.py to use real Optuna-based Bayesian optimization
"""

import os
import sys
import shutil
from datetime import datetime

def backup_file(filepath):
    """Create timestamped backup"""
    if not os.path.exists(filepath):
        print(f"❌ Error: {filepath} not found")
        return None
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{filepath}.bak_{timestamp}"
    shutil.copy2(filepath, backup_path)
    print(f"✅ Backed up: {backup_path}")
    return backup_path

def read_file(filepath):
    """Read file contents"""
    with open(filepath, 'r') as f:
        return f.read()

def write_file(filepath, content):
    """Write file contents"""
    with open(filepath, 'w') as f:
        f.write(content)

def find_class_boundaries(content, class_name):
    """Find start and end of a class definition"""
    lines = content.split('\n')
    start_idx = None
    end_idx = None
    indent_level = None
    
    for i, line in enumerate(lines):
        # Find class definition
        if f"class {class_name}" in line and line.strip().startswith('class'):
            start_idx = i
            # Determine indentation level
            indent_level = len(line) - len(line.lstrip())
            continue
        
        # Find end of class (next class or function at same/lower indent)
        if start_idx is not None and end_idx is None:
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            # Check if we've hit the end (next top-level item or end of file)
            if stripped and not stripped.startswith('#'):
                if current_indent <= indent_level and (stripped.startswith('class ') or stripped.startswith('def ')):
                    end_idx = i
                    break
    
    # If we reached end of file
    if start_idx is not None and end_idx is None:
        end_idx = len(lines)
    
    return start_idx, end_idx

def patch_bayesian_class(content):
    """Patch the BayesianOptimization class"""
    
    # Find the class boundaries
    start_line, end_line = find_class_boundaries(content, "BayesianOptimization")
    
    if start_line is None:
        print("❌ Could not find BayesianOptimization class")
        return None
    
    print(f"📍 Found BayesianOptimization class at lines {start_line+1}-{end_line}")
    
    lines = content.split('\n')
    
    # New Bayesian class implementation
    new_bayesian_class = '''# Try to import real Bayesian optimization
try:
    from window_optimizer_bayesian import OptunaBayesianSearch
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

class BayesianOptimization(SearchStrategy):
    """Bayesian optimization using Optuna TPE"""

    def __init__(self, n_initial=5):
        self.n_initial = n_initial
        if BAYESIAN_AVAILABLE:
            self.optuna_search = OptunaBayesianSearch(n_startup_trials=n_initial, seed=None)

    def search(self, objective_function, bounds, max_iterations, scorer):
        if not BAYESIAN_AVAILABLE:
            print(f"\\n⚠️  Optuna not available, falling back to RandomSearch")
            print(f"   Install with: pip install optuna\\n")
            return RandomSearch().search(objective_function, bounds, max_iterations, scorer)
        
        # Use real Optuna Bayesian optimization
        return self.optuna_search.search(objective_function, bounds, max_iterations, scorer)

    def name(self) -> str:
        return "bayesian_optimization"'''
    
    # Reconstruct the file
    new_lines = lines[:start_line] + [new_bayesian_class] + lines[end_line:]
    new_content = '\n'.join(new_lines)
    
    return new_content

def verify_bayesian_module():
    """Check if window_optimizer_bayesian.py exists"""
    if not os.path.exists('window_optimizer_bayesian.py'):
        print("\n❌ Error: window_optimizer_bayesian.py not found!")
        print("\n📥 Please copy it to ~/distributed_prng_analysis first:")
        print("   The file should be in the same directory as this script")
        return False
    
    print("✅ Found window_optimizer_bayesian.py")
    return True

def test_import():
    """Test if the patched code can be imported"""
    try:
        # Try importing
        import window_optimizer
        from window_optimizer import BayesianOptimization
        print("✅ Import test passed")
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    print("="*80)
    print("BAYESIAN OPTIMIZER PATCHER")
    print("="*80)
    print()
    
    # Check we're in the right directory
    if not os.path.exists('window_optimizer.py'):
        print("❌ Error: window_optimizer.py not found in current directory")
        print("   Please run this script from ~/distributed_prng_analysis")
        sys.exit(1)
    
    print("✅ Found window_optimizer.py")
    
    # Check for Bayesian module
    if not verify_bayesian_module():
        sys.exit(1)
    
    # Read current file
    print("\n📖 Reading window_optimizer.py...")
    original_content = read_file('window_optimizer.py')
    
    # Backup
    print("\n💾 Creating backup...")
    backup_path = backup_file('window_optimizer.py')
    if backup_path is None:
        sys.exit(1)
    
    # Patch
    print("\n🔧 Patching BayesianOptimization class...")
    patched_content = patch_bayesian_class(original_content)
    
    if patched_content is None:
        print("❌ Patching failed")
        sys.exit(1)
    
    # Write patched file
    print("💾 Writing patched file...")
    write_file('window_optimizer.py', patched_content)
    print("✅ Patched window_optimizer.py")
    
    # Test import
    print("\n🧪 Testing import...")
    if not test_import():
        print("\n⚠️  Import test failed! Rolling back...")
        shutil.copy2(backup_path, 'window_optimizer.py')
        print(f"✅ Restored from backup: {backup_path}")
        sys.exit(1)
    
    # Success!
    print("\n" + "="*80)
    print("✅ SUCCESS - Bayesian Optimizer Upgraded!")
    print("="*80)
    print()
    print(f"📦 Changes made:")
    print(f"   - BayesianOptimization class now uses Optuna TPE")
    print(f"   - Falls back to RandomSearch if Optuna unavailable")
    print(f"   - Original backed up to: {backup_path}")
    print()
    print("📋 Next steps:")
    print()
    print("1. Quick import test:")
    print("   python3 -c 'from window_optimizer import BayesianOptimization; print(\"✅ Works!\")'")
    print()
    print("2. Test with real optimization (5 min):")
    print("   python3 coordinator.py daily3.json \\")
    print("     --optimize-window \\")
    print("     --prng-type java_lcg \\")
    print("     --opt-strategy bayesian \\")
    print("     --opt-iterations 5 \\")
    print("     --opt-seed-count 100000000")
    print()
    print("3. Deploy to remote nodes:")
    print("   scp window_optimizer_bayesian.py window_optimizer.py \\")
    print("     michael@192.168.3.120:~/distributed_prng_analysis/")
    print("   scp window_optimizer_bayesian.py window_optimizer.py \\")
    print("     michael@192.168.3.154:~/distributed_prng_analysis/")
    print()
    print("4. Verify results show varying window sizes (not all W512)")
    print()
    print("🆘 If something goes wrong, rollback:")
    print(f"   cp {backup_path} window_optimizer.py")
    print()

if __name__ == "__main__":
    main()
