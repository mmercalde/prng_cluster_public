#!/usr/bin/env python3
"""
fix_localhost_stagger.py
Adds a startup stagger delay for localhost script jobs to prevent CUDA initialization collisions.

THE PROBLEM:
When two localhost GPU jobs start at the exact same millisecond, both try to initialize
CUDA/PyTorch simultaneously, causing "CUDA device busy or unavailable" errors.

THE FIX:
Add a per-GPU stagger delay (gpu_id * 3 seconds) for localhost script jobs.
- GPU 0: starts immediately
- GPU 1: waits 3 seconds before starting
This ensures CUDA contexts initialize sequentially while still allowing parallel execution.
"""

import sys
import shutil
from datetime import datetime

def fix_localhost_stagger(filepath="coordinator.py"):
    # Read the file
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Create backup
    backup_path = f"{filepath}.backup.stagger.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(filepath, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # The code to find (after _increment_node_jobs)
    old_code = '''                # Step 4: Increment active job counter
                self._increment_node_jobs(hostname, analysis_type)

                # Step 5: Process the job (unified handling for both script and seed jobs)'''
    
    # The replacement code with stagger delay
    new_code = '''                # Step 4: Increment active job counter
                self._increment_node_jobs(hostname, analysis_type)

                # Step 4.5: Stagger localhost script jobs to prevent CUDA init collision
                # GPU 0 starts immediately, GPU 1 waits 3s, etc.
                if analysis_type == 'script' and hostname == 'localhost':
                    stagger_delay = worker.gpu_id * 3.0  # 3 seconds per GPU
                    if stagger_delay > 0:
                        self.logger.info(f"[localhost stagger] GPU {worker.gpu_id} waiting {stagger_delay}s before CUDA init")
                        time.sleep(stagger_delay)

                # Step 5: Process the job (unified handling for both script and seed jobs)'''
    
    if old_code not in content:
        print("❌ Could not find the target code block. Manual fix required.")
        print("   Looking for: '# Step 4: Increment active job counter'")
        return False
    
    # Apply the fix
    new_content = content.replace(old_code, new_code)
    
    # Write the fixed file
    with open(filepath, 'w') as f:
        f.write(new_content)
    
    print("✅ Added localhost stagger delay")
    
    # Verify syntax
    print("\n=== Syntax check ===")
    import subprocess
    result = subprocess.run([sys.executable, '-m', 'py_compile', filepath], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Python syntax OK")
    else:
        print(f"❌ Syntax error: {result.stderr}")
        print(f"Restoring backup...")
        shutil.copy(backup_path, filepath)
        return False
    
    # Verify the fix was applied
    with open(filepath, 'r') as f:
        verify_content = f.read()
    
    if '[localhost stagger]' in verify_content:
        print("✅ Stagger code verified in file")
    else:
        print("❌ Stagger code not found - fix may have failed")
        return False
    
    print(f"\n✅ Fix applied successfully!")
    print(f"   Backup saved to: {backup_path}")
    print(f"\n   Localhost GPU stagger:")
    print(f"   - GPU 0: starts immediately")
    print(f"   - GPU 1: waits 3 seconds")
    print(f"\nTest with:")
    print(f"   rm -f scorer_trial_results/trial_*.json")
    print(f"   bash run_scorer_meta_optimizer.sh 6")
    
    return True

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "coordinator.py"
    success = fix_localhost_stagger(filepath)
    sys.exit(0 if success else 1)
