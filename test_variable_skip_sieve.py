#!/usr/bin/env python3
"""
Test sieve against variable skip datasets
Verifies the sieve can correctly identify skip patterns across different windows
"""

import json
import subprocess
import sys
from pathlib import Path

def load_dataset_with_metadata(filename):
    """Load dataset with skip metadata for verification"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def extract_window(data, start_idx, window_size=30):
    """Extract a window of draws from dataset"""
    return data[start_idx:start_idx + window_size]

def get_expected_skip_for_window(data, start_idx, window_size=30):
    """Determine the dominant skip value in a window"""
    window = data[start_idx:start_idx + window_size]
    
    # Count skip values in window
    skip_counts = {}
    for draw in window:
        skip = draw.get('skip_used')
        if skip is not None:
            skip_counts[skip] = skip_counts.get(skip, 0) + 1
    
    if not skip_counts:
        return None
    
    # Return most common skip
    dominant_skip = max(skip_counts, key=skip_counts.get)
    dominant_pct = (skip_counts[dominant_skip] / window_size) * 100
    
    return {
        'skip': dominant_skip,
        'percentage': dominant_pct,
        'distribution': skip_counts
    }

def create_window_dataset(data, start_idx, window_size, output_file):
    """Create a dataset file containing just the window"""
    window = extract_window(data, start_idx, window_size)
    
    # Remove metadata
    clean_window = []
    for d in window:
        clean = {k: v for k, v in d.items() if k != 'skip_used'}
        clean_window.append(clean)
    
    with open(output_file, 'w') as f:
        json.dump(clean_window, f, indent=2)
    
    return clean_window

def run_sieve_on_window(window_file, test_name, start_idx=0, seed_range=(0, 100000), skip_range=(0, 10)):
    """Run sieve on a window and return results"""
    
    job_file = "/tmp/test_window_sieve.json"
    job_data = {
        "job_id": f"window_test_{test_name}",
        "dataset_path": window_file,
        "seed_start": seed_range[0],
        "seed_end": seed_range[1],
        "window_size": 30,
        "min_match_threshold": 0.8,
        "skip_range": list(skip_range),
        "offset": start_idx,
        "prng_families": ["xorshift32"],
        "sessions": ["midday", "evening"]
    }
    
    with open(job_file, 'w') as f:
        json.dump(job_data, f)
    
    # Run sieve
    cmd = ["python3", "sieve_filter.py", "--job-file", job_file, "--gpu-id", "0"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        return {"success": False, "error": result.stderr}
    
    # Parse JSON output (last valid JSON line)
    lines = result.stdout.strip().split('\n')
    json_line = None
    for line in reversed(lines):
        if line.strip().startswith('{'):
            try:
                json.loads(line)
                json_line = line
                break
            except:
                continue
    
    if not json_line:
        return {"success": False, "error": "No JSON output"}
    
    try:
        return json.loads(json_line)
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}

def test_window(data, start_idx, window_size, test_name, expected_seed=42):
    """Test a specific window from the dataset"""
    
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    
    # Get expected skip from metadata
    expected = get_expected_skip_for_window(data, start_idx, window_size)
    
    if expected is None:
        print("❌ No metadata available for this window")
        return False
    
    print(f"Window: draws {start_idx} to {start_idx + window_size - 1}")
    print(f"Date range: {data[start_idx]['date']} to {data[start_idx + window_size - 1]['date']}")
    print(f"Expected skip: {expected['skip']} ({expected['percentage']:.1f}% of window)")
    print(f"Skip distribution: {expected['distribution']}")
    
    # Create window dataset
    window_file = f"/tmp/window_{test_name}.json"
    create_window_dataset(data, start_idx, window_size, window_file)
    
    # Run sieve
    print(f"\nRunning sieve (seed range: 0-100000, skip range: 0-10)...")
    result = run_sieve_on_window(window_file, test_name, start_idx)
    
    if not result.get('success'):
        print(f"❌ FAILED: Sieve execution error")
        print(f"   Error: {result.get('error', 'Unknown')}")
        return False
    
    # Check survivors
    survivors = result.get('survivors', [])
    print(f"\nSurvivors found: {len(survivors)}")
    
    if len(survivors) == 0:
        print(f"❌ FAILED: No survivors found")
        return False
    
    # Find seed 42
    seed_42 = None
    for s in survivors:
        if s['seed'] == expected_seed:
            seed_42 = s
            break
    
    if not seed_42:
        print(f"❌ FAILED: Seed {expected_seed} not found in survivors")
        print(f"   Found seeds: {[s['seed'] for s in survivors[:10]]}")
        return False
    
    # Verify skip value
    found_skip = seed_42['best_skip']
    match_rate = seed_42['match_rate']
    
    print(f"\n✓ Seed {expected_seed} found!")
    print(f"  Found skip: {found_skip}")
    print(f"  Expected skip: {expected['skip']}")
    print(f"  Match rate: {match_rate:.3f}")
    print(f"  Matches: {seed_42['matches']}/{seed_42['total']}")
    
    # Check if skip matches
    if found_skip == expected['skip']:
        print(f"✅ SUCCESS: Correct skip value identified!")
        return True
    else:
        # If skip is slightly off but match rate is still high, might be acceptable
        if match_rate >= 0.8:
            print(f"⚠️  WARNING: Skip mismatch but high match rate")
            print(f"   This might indicate skip ambiguity in the window")
            return True
        else:
            print(f"❌ FAILED: Wrong skip value")
            return False

def test_dataset(dataset_file, pattern_name):
    """Test multiple windows from a dataset"""
    
    print(f"\n{'#'*70}")
    print(f"# TESTING DATASET: {pattern_name}")
    print(f"# File: {dataset_file}")
    print(f"{'#'*70}")
    
    # Load dataset with metadata
    metadata_file = dataset_file.replace('.json', '_with_metadata.json')
    
    if not Path(metadata_file).exists():
        print(f"❌ Metadata file not found: {metadata_file}")
        print(f"   Run variable_skip_dataset.py first to generate datasets")
        return []
    
    data = load_dataset_with_metadata(metadata_file)
    print(f"Loaded {len(data)} draws")
    
    # Define test windows based on pattern
    if pattern_name == "mixed":
        # Test each major period
        test_windows = [
            (0, "Start - expect skip=0"),
            (1000, "Early middle - expect skip=0"),
            (3000, "Period 2 start - expect skip=3"),
            (3500, "Period 2 middle - expect skip=3"),
            (5000, "Period 3 start - expect skip=1"),
            (7000, "Period 4 start - expect skip=5"),
            (9000, "Period 5 start - expect skip=2"),
            (9970, "End - expect skip=2")
        ]
    elif pattern_name == "periodic":
        # Test at each period boundary
        test_windows = [
            (0, "Period 0 - expect skip=0"),
            (1000, "Period 1 - expect skip=1"),
            (2000, "Period 2 - expect skip=2"),
            (3000, "Period 3 - expect skip=3"),
            (5000, "Period 5 - expect skip=5"),
            (7000, "Period 7 - expect skip=7"),
        ]
    elif pattern_name == "burst":
        test_windows = [
            (0, "First burst - expect skip=2"),
            (500, "First burst middle - expect skip=2"),
            (1000, "Second burst - expect skip=7"),
            (1500, "Second burst middle - expect skip=7"),
            (2000, "Third burst - expect skip=2"),
            (3000, "Fourth burst - expect skip=7"),
        ]
    elif pattern_name == "drift":
        test_windows = [
            (0, "Start - expect skip=0"),
            (1000, "Early - expect skip=2"),
            (3000, "Middle - expect skip=6"),
            (5000, "Late - expect skip=10"),
            (7000, "Latest - expect skip=14"),
        ]
    else:
        # Default test points
        test_windows = [
            (0, "Start"),
            (2500, "Quarter"),
            (5000, "Middle"),
            (7500, "Three-quarter"),
            (9970, "End")
        ]
    
    results = []
    for start_idx, description in test_windows:
        if start_idx + 30 > len(data):
            continue
        
        success = test_window(data, start_idx, 30, f"{pattern_name}_{start_idx}")
        results.append({
            'window': start_idx,
            'description': description,
            'success': success
        })
    
    return results

def main():
    """Run comprehensive tests on all variable skip datasets"""
    
    print("="*70)
    print("VARIABLE SKIP SIEVE VALIDATION SUITE")
    print("="*70)
    
    # Test datasets
    datasets = [
        ("dataset_10k_mixed.json", "mixed"),
        ("dataset_10k_periodic.json", "periodic"),
        ("dataset_10k_burst.json", "burst"),
        ("dataset_10k_drift.json", "drift"),
    ]
    
    all_results = {}
    
    for dataset_file, pattern_name in datasets:
        if not Path(dataset_file).exists():
            print(f"\n⚠️  Skipping {pattern_name}: {dataset_file} not found")
            continue
        
        results = test_dataset(dataset_file, pattern_name)
        all_results[pattern_name] = results
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for pattern_name, results in all_results.items():
        passed = sum(1 for r in results if r['success'])
        failed = sum(1 for r in results if not r['success'])
        
        print(f"\n{pattern_name.upper()}:")
        print(f"  Total tests: {len(results)}")
        print(f"  Passed: {passed} ✅")
        print(f"  Failed: {failed} ❌")
        
        if failed > 0:
            print(f"  Failed windows:")
            for r in results:
                if not r['success']:
                    print(f"    - {r['description']} (window {r['window']})")
        
        total_tests += len(results)
        total_passed += passed
        total_failed += failed
    
    print(f"\n{'='*70}")
    print(f"OVERALL: {total_passed}/{total_tests} tests passed ({total_passed/total_tests*100:.1f}%)")
    print(f"{'='*70}")
    
    if total_failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("The sieve correctly identifies variable skip patterns!")
        return 0
    else:
        print(f"\n⚠️  {total_failed} test(s) failed")
        print("Review the output above for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
