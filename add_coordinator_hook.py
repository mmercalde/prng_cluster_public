#!/usr/bin/env python3
"""Add integration hook to coordinator.py (ADDITIVE - no breaking changes)"""

with open('coordinator.py', 'r') as f:
    lines = f.readlines()

output = []
hooks_added = 0

for i, line in enumerate(lines):
    output.append(line)
    
    # Hook 1: After line ~1290 (execute_truly_parallel_dynamic)
    if i > 1285 and i < 1295 and 'print(f"Results saved: {output_file}")' in line:
        hooks_added += 1
        output.append('\n')
        output.append('        # NEW: Also save in new format (optional, non-breaking)\n')
        output.append('        try:\n')
        output.append('            from integration.coordinator_adapter import save_coordinator_results\n')
        output.append('            save_coordinator_results(\n')
        output.append('                run_id=f"coordinator_{int(time.time())}",\n')
        output.append('                final_results=final_results,\n')
        output.append('                config={\n')
        output.append('                    "prng_type": args.prng_type if hasattr(args, "prng_type") else "unknown",\n')
        output.append('                    "total_seeds": total_seeds,\n')
        output.append('                    "samples": samples,\n')
        output.append('                    "lmax": lmax,\n')
        output.append('                    "grid_size": grid_size,\n')
        output.append('                    "target_file": target_file\n')
        output.append('                },\n')
        output.append('                output_file=output_file\n')
        output.append('            )\n')
        output.append('        except Exception as e:\n')
        output.append('            pass  # Silent fail - old format still works\n')
        output.append('\n')
    
    # Hook 2: After line ~1623 (execute_distributed_analysis)
    if i > 1618 and i < 1628 and 'print(f"""\\n=== ANALYSIS COMPLETE ===' in line:
        hooks_added += 1
        output.append('\n')
        output.append('        # NEW: Also save in new format (optional, non-breaking)\n')
        output.append('        try:\n')
        output.append('            from integration.coordinator_adapter import save_coordinator_results\n')
        output.append('            save_coordinator_results(\n')
        output.append('                run_id=analysis_id,\n')
        output.append('                final_results=final_results,\n')
        output.append('                config={\n')
        output.append('                    "prng_type": getattr(args, "prng_type", "unknown"),\n')
        output.append('                    "total_seeds": total_seeds,\n')
        output.append('                    "samples": samples,\n')
        output.append('                    "lmax": lmax,\n')
        output.append('                    "grid_size": grid_size,\n')
        output.append('                    "target_file": target_file\n')
        output.append('                },\n')
        output.append('                output_file=output_file\n')
        output.append('            )\n')
        output.append('        except Exception as e:\n')
        output.append('            pass  # Silent fail - old format still works\n')
        output.append('\n')

with open('coordinator.py', 'w') as f:
    f.writelines(output)

print(f"✅ Added {hooks_added} integration hooks to coordinator.py")
print("   Hooks are ADDITIVE - old format still works")
print("   New format created as bonus")
print("   Silent failure if adapter has issues")
