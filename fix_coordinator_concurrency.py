#!/usr/bin/env python3
"""
fix_coordinator_concurrency.py
Fixes the broken _node_max_concurrent initialization in coordinator.py

THE BUG:
Lines 110-120 try to populate _node_max_concurrent by looping over self.nodes,
but self.nodes is EMPTY at that point because load_configuration() hasn't run yet.
This causes unlimited concurrent script jobs on localhost, leading to CUDA conflicts.

THE FIX:
1. Remove the broken loop from __init__ (lines 110-120)
2. Add the initialization at the end of load_configuration() where config is available
"""

import sys
import shutil
from datetime import datetime

def fix_coordinator(filepath="coordinator.py"):
    # Read the file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Create backup
    backup_path = f"{filepath}.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy(filepath, backup_path)
    print(f"✅ Created backup: {backup_path}")
    
    # Find and fix the broken section in __init__
    new_lines = []
    i = 0
    removed_broken_loop = False
    added_fix = False
    
    while i < len(lines):
        line = lines[i]
        
        # PART 1: Remove broken loop from __init__
        # Look for "# Load capacity limits from config" comment
        if '# Load capacity limits from config' in line and not removed_broken_loop:
            # Replace the entire broken block (lines 110-120) with a comment
            new_lines.append('        # v1.7.4.1 - Per-node concurrency limits initialized in load_configuration()\n')
            
            # Skip until we find the line after "_node_active_jobs[hostname] = 0"
            while i < len(lines):
                if 'self._node_active_jobs[hostname] = 0' in lines[i]:
                    i += 1  # Skip this line too
                    break
                i += 1
            
            # Skip any blank line after the removed block
            if i < len(lines) and lines[i].strip() == '':
                i += 1
                
            removed_broken_loop = True
            print("✅ Removed broken loop from __init__")
            continue
        
        # PART 2: Add fix to load_configuration()
        # Look for the end of reverse_sieve_defaults block
        if 'self.reverse_sieve_defaults = config.get("reverse_sieve_defaults"' in line and not added_fix:
            # Add this line
            new_lines.append(line)
            i += 1
            
            # Continue until we find the closing })"
            while i < len(lines):
                new_lines.append(lines[i])
                if lines[i].strip() == '})':
                    i += 1
                    break
                i += 1
            
            # Now add our fix code
            fix_code = '''
        # v1.7.4.1 FIX: Initialize per-node concurrency limits from config
        # (Moved here from __init__ because self.nodes and config are now available)
        for node_config in config.get('nodes', []):
            hostname = node_config['hostname']
            limit = node_config.get('max_concurrent_script_jobs', 99)
            self._node_max_concurrent[hostname] = limit
            self._node_active_jobs[hostname] = 0
            self.logger.info(f"Node {hostname}: max_concurrent_script_jobs={limit}")
'''
            new_lines.append(fix_code)
            added_fix = True
            print("✅ Added initialization to load_configuration()")
            continue
        
        new_lines.append(line)
        i += 1
    
    # Write the fixed file
    with open(filepath, 'w') as f:
        f.writelines(new_lines)
    
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
    
    # Show verification
    print("\n=== Verification ===")
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'for node in self.nodes:' in content and '# Load capacity limits from config' in content:
        print("❌ WARNING: Broken loop may still exist in __init__")
    else:
        print("✅ Broken loop removed from __init__")
    
    if 'v1.7.4.1 FIX: Initialize per-node concurrency' in content:
        print("✅ Fix code added to load_configuration()")
    else:
        print("❌ WARNING: Fix code not found in load_configuration()")
    
    print(f"\n✅ Fix applied successfully!")
    print(f"   Backup saved to: {backup_path}")
    print(f"\nTest with:")
    print(f"   bash run_scorer_meta_optimizer.sh 6")
    
    return True

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "coordinator.py"
    success = fix_coordinator(filepath)
    sys.exit(0 if success else 1)
