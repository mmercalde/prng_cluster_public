#!/usr/bin/env python3
"""
Simple Results Manager - Better file naming for analysis results
"""

import os
import json
import shutil
from datetime import datetime

def create_readable_filename(analysis_type, params):
    """Create human-readable filename"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # Clean analysis type
    clean_type = analysis_type.lower().replace(' ', '-').replace('_', '-')
    
    # Add parameters
    param_str = ""
    if 'seeds' in params:
        seeds = params['seeds']
        if seeds >= 1000000:
            param_str += f"{seeds//1000000}M-seeds"
        elif seeds >= 1000:
            param_str += f"{seeds//1000}k-seeds"
        else:
            param_str += f"{seeds}-seeds"
    
    if 'samples' in params and params['samples'] > 0:
        samples = params['samples']
        if samples >= 1000000:
            param_str += f"-{samples//1000000}M-samples"
        elif samples >= 1000:
            param_str += f"-{samples//1000}k-samples"
    
    if 'target_number' in params:
        param_str += f"-target-{params['target_number']}"
    
    return f"{timestamp}_{clean_type}_{param_str}.json"

def analyze_result_status(filepath):
    """Check if analysis was successful"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check for draw matching results
        if 'matching_seeds' in data:
            matches = data.get('matching_seeds', [])
            if matches:
                return f"FOUND-{len(matches)}-MATCHES"
            else:
                return "NO-MATCHES"
        
        # Check metadata
        if 'metadata' in data:
            meta = data['metadata']
            successful = meta.get('successful_jobs', 0)
            total = meta.get('total_jobs', 0)
            
            if successful == total and total > 0:
                return "SUCCESS"
            elif successful == 0:
                return "FAILED"
            else:
                return f"PARTIAL-{successful}of{total}"
        
        return "UNKNOWN"
    except:
        return "ERROR"

def rename_result_file(old_filepath, analysis_type, params):
    """Rename result file to readable format"""
    # Generate new filename
    new_filename = create_readable_filename(analysis_type, params)
    
    # Get status
    status = analyze_result_status(old_filepath)
    
    # Add status to filename
    base_name = new_filename.replace('.json', '')
    new_filename = f"{base_name}_{status}.json"
    
    # Create new path
    results_dir = os.path.dirname(old_filepath)
    new_filepath = os.path.join(results_dir, new_filename)
    
    # Rename file
    if old_filepath != new_filepath:
        shutil.move(old_filepath, new_filepath)
    
    return new_filepath

def migrate_old_files():
    """Convert existing files to new naming scheme"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return
    
    print("Converting existing result files to readable names...")
    
    old_files = [
        ("quick_test", {"seeds": 1000, "samples": 1000}),
        ("standard_analysis", {"seeds": 50000, "samples": 50000}),
        ("production_analysis", {"seeds": 1000000, "samples": 500000}),
        ("deep_pattern_mining", {"seeds": 500000, "samples": 100000}),
        ("standard_correlation_analysis", {"seeds": 50000, "samples": 10000}),
        ("quick_connectivity_test", {"seeds": 0, "samples": 0})
    ]
    
    converted = 0
    
    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
        
        filepath = os.path.join(results_dir, filename)
        
        # Skip if already converted (has date pattern)
        if filename.startswith('20') and '_' in filename[:10]:
            continue
        
        # Try to match with known patterns
        base_name = filename.replace('.json', '')
        
        # Handle draw matching files
        if 'draw_match_' in base_name:
            parts = base_name.split('_')
            if len(parts) >= 3:
                try:
                    target = int(parts[2])
                    params = {"target_number": target, "seeds": 100000}
                    new_filepath = rename_result_file(filepath, "draw-matching", params)
                    print(f"Converted: {filename} -> {os.path.basename(new_filepath)}")
                    converted += 1
                    continue
                except:
                    pass
        
        # Handle other analysis types
        for analysis_type, params in old_files:
            if analysis_type in base_name:
                new_filepath = rename_result_file(filepath, analysis_type, params)
                print(f"Converted: {filename} -> {os.path.basename(new_filepath)}")
                converted += 1
                break
    
    print(f"Converted {converted} files to readable format")
    return converted
