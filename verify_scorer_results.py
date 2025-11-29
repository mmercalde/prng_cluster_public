import json
import sys
import os
from typing import List, Dict, Any

def verify_results_file(file_path: str, expected_count: int = 26):
    """
    Loads the aggregated JSON results and performs integrity and validity checks.
    """
    if not os.path.exists(file_path):
        print(f"❌ ERROR: Results file not found at: {file_path}")
        return False

    print(f"🔬 Loading and verifying file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Failed to decode JSON file. Check for corruption. Error: {e}")
        return False

    # --- 1. Integrity and Count Checks ---
    metadata = data.get('metadata', {})
    results_list: List[Dict[str, Any]] = data.get('results', [])
    
    total_jobs = metadata.get('total_jobs', 0)
    successful_jobs = metadata.get('successful_jobs', 0)
    results_count = len(results_list)

    print("-" * 40)
    print("1. Integrity Check")
    print(f"   Metadata Total Jobs:     {total_jobs}")
    print(f"   Metadata Successful Jobs: {successful_jobs}")
    print(f"   Results List Count:      {results_count}")

    if total_jobs != expected_count or results_count != expected_count:
        print(f"⚠️ WARNING: Job count mismatch. Expected {expected_count}, Found {total_jobs} (Metadata) / {results_count} (List).")
        return False
    
    if successful_jobs != expected_count:
        print(f"❌ ERROR: Not all jobs were recorded as successful. Successful: {successful_jobs}/{expected_count}.")
        return False
        
    # --- 2. Metric Validity and Type Checks ---
    print("\n2. Metric and Consistency Check")
    
    accuracy_metrics = []
    consistent = True
    
    for i, job_result in enumerate(results_list):
        trial_id = job_result.get('trial_id', i)
        accuracy = job_result.get('accuracy')
        
        if accuracy is None:
            print(f"❌ ERROR: Trial {trial_id} is missing the 'accuracy' metric.")
            consistent = False
            continue
        
        if not isinstance(accuracy, (int, float)):
            print(f"❌ ERROR: Trial {trial_id} accuracy metric is not a number (Found: {type(accuracy).__name__}).")
            consistent = False
            continue

        if accuracy > 0.000001:  # Allowing small float tolerance
             print(f"⚠️ WARNING: Trial {trial_id} has a positive accuracy score ({accuracy:.6f}). Expecting NegMSE (<= 0).")
             # We won't fail the entire script on this, but it's a strong flag.

        accuracy_metrics.append(accuracy)

    if not consistent:
        return False

    print(f"   Successfully verified {len(accuracy_metrics)} accuracy metrics.")
    print(f"   Best (Least Negative) Accuracy: {max(accuracy_metrics):.6f}")
    print(f"   Worst (Most Negative) Accuracy: {min(accuracy_metrics):.6f}")
    
    print("\n✅ VALIDATION SUCCESSFUL: The aggregated file is structurally sound and metrics are present.")
    return True

if __name__ == "__main__":
    RESULTS_FILE = 'results/scorer_jobs_quick_results_1763520044.json'
    EXPECTED_TRIALS = 26
    
    if verify_results_file(RESULTS_FILE, EXPECTED_TRIALS):
        sys.exit(0)
    else:
        sys.exit(1)
