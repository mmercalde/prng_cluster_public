#!/usr/bin/env python3
"""
================================================================================
SIEVE FILTER INTEGRATION ADAPTER
================================================================================

File: integration/sieve_integration.py
Version: 1.0.0
Created: 2025-11-03

PURPOSE:
--------
Adapter to integrate ResultsManager into sieve filters without modifying
the original code extensively.

USAGE:
------
from integration.sieve_integration import save_forward_sieve_results
save_forward_sieve_results(survivors, config, direction='forward')

================================================================================
"""

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from core.results_manager import ResultsManager
    RESULTS_MANAGER_AVAILABLE = True
except ImportError:
    RESULTS_MANAGER_AVAILABLE = False
    print("Warning: ResultsManager not available")


def save_forward_sieve_results(
    survivors: List[Dict[str, Any]],
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    execution_time: Optional[float] = None
) -> bool:
    """Save forward sieve results using ResultsManager."""
    return _save_sieve_results(survivors, config, 'forward', run_id, execution_time)


def save_reverse_sieve_results(
    survivors: List[Dict[str, Any]],
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    execution_time: Optional[float] = None
) -> bool:
    """Save reverse sieve results using ResultsManager."""
    return _save_sieve_results(survivors, config, 'reverse', run_id, execution_time)


def save_bidirectional_sieve_results(
    forward_survivors: List[Dict],
    reverse_survivors: List[Dict],
    intersection: List[Dict],
    config: Dict,
    run_id: Optional[str] = None,
    execution_time: Optional[float] = None
) -> bool:
    """Save bidirectional sieve results using ResultsManager."""
    enhanced_config = config.copy()
    enhanced_config['forward_count'] = len(forward_survivors)
    enhanced_config['reverse_count'] = len(reverse_survivors)
    return _save_sieve_results(intersection, enhanced_config, 'bidirectional', run_id, execution_time)


def _save_sieve_results(
    survivors: List[Dict[str, Any]],
    config: Dict[str, Any],
    direction: str,
    run_id: Optional[str] = None,
    execution_time: Optional[float] = None
) -> bool:
    """Internal function to save sieve results."""
    
    if not RESULTS_MANAGER_AVAILABLE:
        print("ResultsManager not available - skipping new format")
        return False
    
    try:
        # Generate run_id if not provided
        if run_id is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            prng = config.get('prng_type', 'unknown')
            run_id = f"{direction}_sieve_{prng}_{timestamp}"
        
        # Determine analysis type
        analysis_type = f"{direction}_sieve"
        
        # Convert survivors to ResultsManager format
        formatted_survivors = []
        for survivor in survivors:
            formatted_survivor = {
                'seed': survivor.get('seed', 0),
                'matches': survivor.get('matches', survivor.get('match_count', 0)),
                'total_draws': config.get('window_size', 244),
                'match_rate': survivor.get('match_rate', 0.0),
                'skip_length': survivor.get('skip', survivor.get('skip_length', survivor.get('best_skip', 0))),
                'direction': direction
            }
            formatted_survivors.append(formatted_survivor)
        
        # Build results data structure
        results_data = {
            'run_metadata': {
                'timestamp_start': datetime.now().isoformat(),
                'execution_time_seconds': execution_time
            },
            'analysis_parameters': {
                'prng_type': config.get('prng_type', 'unknown'),
                'seed_start': config.get('seed_start', 0),
                'seed_end': config.get('seed_end', 0),
                'total_seeds_tested': config.get('total_seeds', 0),
                'window_size': config.get('window_size', 0),
                'window_offset': config.get('offset', 0),
                'skip_min': config.get('skip_min', 0),
                'skip_max': config.get('skip_max', 0),
                'threshold': config.get('threshold', 0.0),
                'dataset_file': config.get('dataset', 'unknown'),
                'sessions_analyzed': config.get('sessions', [])
            },
            'results_summary': {
                'total_survivors': len(survivors),
                'survival_rate': len(survivors) / config.get('total_seeds', 1) if config.get('total_seeds', 0) > 0 else 0.0,
                'analysis_complete': True
            },
            'survivors': formatted_survivors
        }
        
        # Add direction-specific fields
        if direction == 'forward':
            results_data['results_summary']['forward_survivors'] = len(survivors)
        elif direction == 'reverse':
            results_data['results_summary']['reverse_survivors'] = len(survivors)
        elif direction == 'bidirectional':
            results_data['results_summary']['bidirectional_survivors'] = len(survivors)
            results_data['results_summary']['forward_survivors'] = config.get('forward_count', 0)
            results_data['results_summary']['reverse_survivors'] = config.get('reverse_count', 0)
        
        # Add best seed info if available
        if survivors:
            best = max(survivors, key=lambda s: s.get('matches', s.get('match_count', 0)))
            results_data['results_summary']['best_seed'] = best.get('seed', 0)
            results_data['results_summary']['best_match_count'] = best.get('matches', best.get('match_count', 0))
            results_data['results_summary']['best_match_rate'] = best.get('match_rate', 0.0)
        
        # Initialize ResultsManager and save
        rm = ResultsManager(
            schema_dir=str(Path(__file__).parent.parent / 'schemas'),
            results_dir=str(Path(__file__).parent.parent / 'results')
        )
        rm.save_results(
            analysis_type=analysis_type,
            run_id=run_id,
            data=results_data
        )
        
        print(f"✅ New results format saved: {run_id}")
        return True
        
    except Exception as e:
        print(f"⚠️  New results format failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("Testing sieve integration adapter...")
    
    test_survivors = [
        {'seed': 12345, 'matches': 10, 'match_rate': 0.041, 'best_skip': 7},
        {'seed': 67890, 'matches': 9, 'match_rate': 0.037, 'best_skip': 14}
    ]
    
    test_config = {
        'prng_type': 'java_lcg',
        'seed_start': 0,
        'seed_end': 1000000,
        'total_seeds': 1000000,
        'window_size': 244,
        'offset': 139,
        'skip_min': 3,
        'skip_max': 29,
        'threshold': 0.012,
        'dataset': 'daily3.json'
    }
    
    success = save_forward_sieve_results(test_survivors, test_config)
    
    if success:
        print("\n✅ Adapter test passed!")
    else:
        print("\n❌ Adapter test failed!")
