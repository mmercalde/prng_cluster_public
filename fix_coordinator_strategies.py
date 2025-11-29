#!/usr/bin/env python3
"""
Add strategy loading to _create_sieve_jobs (copying from _create_reverse_sieve_jobs)
"""

with open('coordinator.py', 'r') as f:
    lines = f.readlines()

# Find line 1614-1615 where it prints "Hybrid mode enabled with 0 strategies"
for i in range(len(lines)):
    if i >= 1613 and i <= 1616 and 'Hybrid mode enabled' in lines[i]:
        print(f"✅ Found hybrid print at line {i+1}: {lines[i].strip()}")
        
        # We need to ADD strategy loading BEFORE this print
        # Insert after "if HYBRID_AVAILABLE:" (line before the print)
        
        insert_pos = i  # Right before the print line
        
        # The code to insert (copied from _create_reverse_sieve_jobs lines 1723-1736)
        indent = '                '
        new_code = [
            f"{indent}# Load strategies for hybrid mode\n",
            f"{indent}try:\n",
            f"{indent}    from hybrid_strategy import get_all_strategies\n",
            f"{indent}    strategies = get_all_strategies()\n",
            f"{indent}    self._sieve_config['strategies'] = [\n",
            f"{indent}        {{\n",
            f"{indent}            'name': s.name,\n",
            f"{indent}            'max_consecutive_misses': s.max_consecutive_misses,\n",
            f"{indent}            'skip_tolerance': s.skip_tolerance,\n",
            f"{indent}            'enable_reseed_search': s.enable_reseed_search,\n",
            f"{indent}            'skip_learning_rate': s.skip_learning_rate,\n",
            f"{indent}            'breakpoint_threshold': s.breakpoint_threshold\n",
            f"{indent}        }}\n",
            f"{indent}        for s in strategies\n",
            f"{indent}    ]\n",
            f"{indent}except ImportError:\n",
            f"{indent}    self._sieve_config['strategies'] = []\n",
        ]
        
        lines[insert_pos:insert_pos] = new_code
        print(f"✅ Inserted strategy loading at line {insert_pos+1}")
        break
else:
    print("❌ Could not find the hybrid print line!")
    exit(1)

with open('coordinator.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed coordinator.py - strategies will now be loaded for forward sieve")

