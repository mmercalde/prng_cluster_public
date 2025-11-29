#!/usr/bin/env python3
"""
Coordinator Integration Adapter
Bridge between coordinator.py and results_manager
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.results_manager import ResultsManager
from datetime import datetime
from typing import Dict, Any

def save_coordinator_results(run_id: str, final_results: Dict[str, Any], 
                            config: Dict[str, Any], output_file: str) -> bool:
    """Save coordinator results in new format"""
    try:
        manager = ResultsManager()
        metadata = final_results.get('metadata', {})
        
        formatted = {
            'run_metadata': {
                'run_id': run_id,
                'analysis_type': 'dataset_correlation',
                'timestamp_start': datetime.now().isoformat(),
                'execution_time_seconds': metadata.get('total_runtime', 0),
                'prng_type': config.get('prng_type', 'unknown'),
                'dataset_file': config.get('target_file', 'unknown')
            },
            'analysis_parameters': {
                'total_seeds_tested': config.get('total_seeds', 0),
                'samples_per_seed': config.get('samples', 0),
                'lmax': config.get('lmax', 0),
                'grid_size': config.get('grid_size', 0)
            },
            'execution_info': {
                'nodes_used': metadata.get('nodes_used', 0),
                'total_gpus': metadata.get('total_gpus', 0)
            },
            'results_summary': {
                'successful_jobs': len(final_results.get('results', [])),
                'failed_jobs': len(final_results.get('failed_jobs', []))
            }
        }
        
        # CORRECT SIGNATURE: save_results(analysis_type, run_id, data)
        paths = manager.save_results(
            analysis_type='dataset_correlation',
            run_id=run_id,
            data=formatted
        )
        
        print(f"✅ New format saved:")
        print(f"   Summary: {paths.get('summary')}")
        print(f"   CSV: {paths.get('csv')}")
        print(f"   JSON: {paths.get('json')}")
        return True
        
    except Exception as e:
        print(f"⚠️ New format failed: {e}")
        import traceback
        print(traceback.format_exc()[:300])
        return False
