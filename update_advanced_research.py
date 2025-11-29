#!/usr/bin/env python3
"""
Script to update modules/advanced_research.py to use new results format
Updates 3 locations that create old format JSON files
"""

import os
import shutil
from datetime import datetime

def update_advanced_research():
    """Update advanced_research.py to use new results_manager system"""
    
    filepath = "modules/advanced_research.py"
    
    # Backup first
    backup_file = f"{filepath}.backup_before_new_format"
    shutil.copy2(filepath, backup_file)
    print(f"✅ Backup created: {backup_file}")
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    changes = []
    
    # PATCH 1: Line 264 - Gap-Aware Analysis
    # Replace: filename = f"gap_aware_analysis_{timestamp}.json"
    # With: Use results_manager
    for i in range(len(lines)):
        if i >= 260 and i <= 270:
            if 'filename = f"gap_aware_analysis_{timestamp}.json"' in lines[i]:
                # Comment out old line
                lines[i] = '                # ' + lines[i].lstrip()
                changes.append(f"Line {i+1}: Commented out old gap_aware filename")
                
                # Insert new format code
                new_code = '''                # NEW FORMAT: Use results_manager
                try:
                    from integration.results_manager import save_advanced_results
                    run_id = f"gap_aware_{timestamp}"
                    filename = save_advanced_results(
                        run_id=run_id,
                        analysis_type="gap_aware",
                        results=results,
                        config={"timestamp": timestamp}
                    )
                    print(f"Results saved in new format: {filename}")
                except ImportError:
                    # Fallback to old format if integration not available
                    filename = f"results/json/gap_aware_analysis_{timestamp}.json"
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
'''
                lines[i] = new_code + lines[i]
                changes.append(f"Line {i+1}: Added new format save code")
    
    # PATCH 2: Line 697 - Historical Analysis
    # Replace: default_output = f"historical_analysis_{timestamp}.json"
    # With: New format default
    for i in range(len(lines)):
        if i >= 693 and i <= 703:
            if 'default_output = f"historical_analysis_{timestamp}.json"' in lines[i]:
                # Replace with new format
                old_line = lines[i]
                lines[i] = '        # NEW FORMAT: Use organized directory structure\n'
                lines[i] += '        default_output = f"results/summaries/historical_analysis_{timestamp}_summary.txt"\n'
                changes.append(f"Line {i+1}: Updated historical_analysis default output to new format")
    
    # PATCH 3: Line 813 - Search Results Export
    # Replace: default_file = f"search_results_{search_id}_{timestamp}.json"
    # With: New format default
    for i in range(len(lines)):
        if i >= 809 and i <= 819:
            if 'default_file = f"search_results_{search_id}_{timestamp}.json"' in lines[i]:
                # Replace with new format
                old_line = lines[i]
                lines[i] = '        # NEW FORMAT: Use organized directory structure\n'
                lines[i] += '        default_file = f"results/json/search_results_{search_id}_{timestamp}.json"\n'
                changes.append(f"Line {i+1}: Updated search_results default output to new format")
    
    # Write updated file
    with open(filepath, 'w') as f:
        f.writelines(lines)
    
    print(f"\n✅ CHANGES MADE:")
    for change in changes:
        print(f"   {change}")
    
    print(f"\n✅ Updated: {filepath}")
    print(f"✅ Backup: {backup_file}")
    
    # Verify it still imports
    try:
        import sys
        sys.path.insert(0, 'modules')
        import advanced_research
        print("✅ Module still imports correctly")
    except Exception as e:
        print(f"❌ Import error: {e}")
        print(f"   Restoring backup...")
        shutil.copy2(backup_file, filepath)
        print(f"   Backup restored")
        return False
    
    return True

if __name__ == "__main__":
    print("=" * 70)
    print("UPDATING ADVANCED_RESEARCH.PY FOR NEW FORMAT")
    print("=" * 70)
    print()
    
    if not os.path.exists("modules/advanced_research.py"):
        print("❌ modules/advanced_research.py not found")
        print("   Run this script from ~/distributed_prng_analysis/")
        exit(1)
    
    print("This will update 3 locations in advanced_research.py:")
    print("  1. Gap-Aware Analysis output (line ~264)")
    print("  2. Historical Analysis output (line ~697)")
    print("  3. Search Results Export output (line ~813)")
    print()
    print("Old format: results/gap_aware_analysis_*.json")
    print("New format: results/json/gap_aware_analysis_*.json")
    print("            results/summaries/historical_analysis_*_summary.txt")
    print()
    
    response = input("Proceed with update? (yes/no): ").strip().lower()
    if response != 'yes':
        print("❌ Update cancelled")
        exit(0)
    
    success = update_advanced_research()
    
    if success:
        print()
        print("=" * 70)
        print("✅ UPDATE COMPLETE")
        print("=" * 70)
        print()
        print("Advanced research functions now use new format:")
        print("  ✅ Gap-Aware Analysis → results/json/")
        print("  ✅ Historical Analysis → results/summaries/")
        print("  ✅ Search Results → results/json/")
        print()
        print("To restore old behavior:")
        print(f"  cp modules/advanced_research.py.backup_before_new_format modules/advanced_research.py")
    else:
        print()
        print("❌ Update failed - backup restored")
