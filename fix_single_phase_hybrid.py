#!/usr/bin/env python3
"""
Add support for single-phase hybrid PRNGs (like xorshift32_hybrid)
"""

with open('sieve_filter.py', 'r') as f:
    lines = f.readlines()

# Find where it says "if use_hybrid and supports_hybrid:"
for i in range(len(lines)):
    if "if use_hybrid and supports_hybrid:" in lines[i]:
        print(f"✅ Found hybrid block at line {i+1}")
        
        # We need to add logic AFTER the strategies are loaded
        # Find where strategies are loaded (line with "if use_hybrid and strategies:")
        for j in range(i, min(i+30, len(lines))):
            if "if use_hybrid and strategies:" in lines[j]:
                print(f"✅ Found strategies check at line {j+1}")
                
                # Insert new logic right after this line
                indent = '                    '
                new_logic = [
                    lines[j],  # Keep: if use_hybrid and strategies:
                    f"{indent}# Check if this is a single-phase hybrid (already has _hybrid suffix)\n",
                    f"{indent}# vs two-phase hybrid (needs Phase 1 + Phase 2)\n",
                    f"{indent}is_single_phase = family_name.endswith('_hybrid')\n",
                    f"{indent}\n",
                    f"{indent}if is_single_phase:\n",
                    f"{indent}    # Single-phase: Direct hybrid sieve (e.g., xorshift32_hybrid)\n",
                    f"{indent}    print(f\"Testing {{family_name}} in SINGLE-PHASE HYBRID mode...\", file=sys.stderr)\n",
                    f"{indent}    phase2_threshold = coerce_threshold(job.get('phase2_threshold', 'auto'), 0.50)\n",
                    f"{indent}    \n",
                    f"{indent}    phase_start = time.time()\n",
                    f"{indent}    result = sieve.run_hybrid_sieve(\n",
                    f"{indent}        prng_family=family_name,\n",
                    f"{indent}        seed_start=seed_start,\n",
                    f"{indent}        seed_end=seed_end,\n",
                    f"{indent}        residues=draws,\n",
                    f"{indent}        strategies=strategies,\n",
                    f"{indent}        min_match_threshold=phase2_threshold,\n",
                    f"{indent}        offset=offset\n",
                    f"{indent}    )\n",
                    f"{indent}    phase_duration = (time.time() - phase_start) * 1000\n",
                    f"{indent}    \n",
                    f"{indent}    survivors = result.get('survivors', [])\n",
                    f"{indent}    print(f\"  Found {{len(survivors)}} survivors ({{phase_duration:.1f}}ms)\", file=sys.stderr)\n",
                    f"{indent}    \n",
                    f"{indent}    result.update({{\n",
                    f"{indent}        'family': family_name,\n",
                    f"{indent}        'seed_range': {{'start': seed_start, 'end': seed_end}},\n",
                    f"{indent}        'single_phase': {{\n",
                    f"{indent}            'threshold': round(phase2_threshold, 4),\n",
                    f"{indent}            'duration_ms': round(phase_duration, 2),\n",
                    f"{indent}            'strategies_tested': len(strategies)\n",
                    f"{indent}        }}\n",
                    f"{indent}    }})\n",
                    f"{indent}    \n",
                    f"{indent}elif family_config.get('multi_strategy', False):\n",
                    f"{indent}    # Two-phase: mt19937 -> mt19937_hybrid\n",
                ]
                
                # Replace just the "if use_hybrid and strategies:" line
                lines[j:j+1] = new_logic
                
                print(f"✅ Added single-phase hybrid support")
                print(f"   xorshift32_hybrid will use direct hybrid sieve")
                print(f"   mt19937 will still use two-phase approach")
                break
        break

with open('sieve_filter.py', 'w') as f:
    f.writelines(lines)

print("✅ Fixed execute_sieve_job to support single-phase hybrid PRNGs")

