#!/usr/bin/env python3
"""
Coordinator Integration Adapter v2.0.0
Bridge between coordinator.py/scripts_coordinator.py and results_manager

Updated: 2025-12-18
- v2.0.0: Added support for scripts_coordinator.py output format
- v2.0.0: normalize_coordinator_results() auto-detects format
- v1.0.0: Added agent_metadata injection via metadata_writer.py
- v1.0.0: Supports Phase 2 agent infrastructure

Supports both:
- coordinator.py (legacy format)
- scripts_coordinator.py v1.4.0 (new format)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.results_manager import ResultsManager
from datetime import datetime
from typing import Dict, Any, Optional

# Import metadata writer for agent infrastructure
try:
    from integration.metadata_writer import (
        inject_agent_metadata,
        get_default_cluster_resources
    )
    METADATA_WRITER_AVAILABLE = True
except ImportError:
    METADATA_WRITER_AVAILABLE = False
    print("Warning: metadata_writer not available - agent_metadata will not be injected")


# =============================================================================
# FORMAT NORMALIZATION (v2.0.0)
# =============================================================================

def normalize_coordinator_results(raw_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize results from either coordinator.py or scripts_coordinator.py.
    
    Auto-detects format based on presence of key fields.
    
    Args:
        raw_results: Results dict from either coordinator
        
    Returns:
        Normalized dict with consistent structure:
        {
            'metadata': {'total_runtime': ..., 'nodes_used': ..., 'total_gpus': ...},
            'results': [list of successful jobs],
            'failed_jobs': [list of failed jobs]
        }
    """
    # Detect format by presence of 'run_id' and 'jobs' (scripts_coordinator.py v1.3+)
    if 'run_id' in raw_results and 'jobs' in raw_results:
        # NEW FORMAT (scripts_coordinator.py v1.3.0+)
        nodes_dict = raw_results.get('nodes', {})
        
        # Calculate total GPUs from nodes
        total_gpus = sum(
            n.get('gpu_count', 0) 
            for n in nodes_dict.values()
        ) if isinstance(nodes_dict, dict) else 0
        
        # Separate successful and failed jobs
        all_jobs = raw_results.get('jobs', [])
        successful_jobs = [j for j in all_jobs if j.get('success', False)]
        failed_jobs = [j for j in all_jobs if not j.get('success', False)]
        
        return {
            'metadata': {
                'total_runtime': raw_results.get('runtime_seconds', 0),
                'nodes_used': len(nodes_dict),
                'total_gpus': total_gpus,
                'run_id': raw_results.get('run_id', 'unknown'),
                'output_dir': raw_results.get('output_dir', ''),
                'coordinator_version': 'scripts_coordinator'
            },
            'results': successful_jobs,
            'failed_jobs': failed_jobs
        }
    
    elif 'metadata' in raw_results:
        # OLD FORMAT (coordinator.py) - pass through with version tag
        normalized = dict(raw_results)
        if 'metadata' in normalized:
            normalized['metadata']['coordinator_version'] = 'coordinator'
        return normalized
    
    else:
        # UNKNOWN FORMAT - create minimal structure
        print(f"Warning: Unknown coordinator result format. Keys: {list(raw_results.keys())}")
        return {
            'metadata': {
                'total_runtime': 0,
                'nodes_used': 0,
                'total_gpus': 0,
                'coordinator_version': 'unknown'
            },
            'results': [],
            'failed_jobs': []
        }


def is_scripts_coordinator_format(raw_results: Dict[str, Any]) -> bool:
    """Check if results are from scripts_coordinator.py"""
    return 'run_id' in raw_results and 'jobs' in raw_results


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def save_coordinator_results(
    run_id: str,
    final_results: Dict[str, Any],
    config: Dict[str, Any],
    output_file: str,
    parent_run_id: Optional[str] = None,
    pipeline_step: Optional[int] = None,
    follow_up_agent: Optional[str] = None
) -> bool:
    """
    Save coordinator results in new format with optional agent_metadata.
    
    Supports both coordinator.py and scripts_coordinator.py output formats.
    
    Args:
        run_id: Unique run identifier
        final_results: Results dict from coordinator (either format)
        config: Configuration dict with analysis parameters
        output_file: Path to output file
        parent_run_id: Optional parent run ID for pipeline lineage
        pipeline_step: Optional pipeline step number (1-6)
        follow_up_agent: Optional next agent to trigger
    
    Returns:
        True if save successful, False otherwise
    """
    try:
        manager = ResultsManager()
        
        # Normalize results to consistent format (v2.0.0)
        normalized = normalize_coordinator_results(final_results)
        metadata = normalized.get('metadata', {})
        
        # Calculate success metrics
        successful_jobs = len(normalized.get('results', []))
        failed_jobs = len(normalized.get('failed_jobs', []))
        total_jobs = successful_jobs + failed_jobs
        success_rate = successful_jobs / total_jobs if total_jobs > 0 else 0.0

        formatted = {
            'run_metadata': {
                'run_id': run_id,
                'analysis_type': 'dataset_correlation',
                'timestamp_start': datetime.now().isoformat(),
                'execution_time_seconds': metadata.get('total_runtime', 0),
                'prng_type': config.get('prng_type', 'unknown'),
                'dataset_file': config.get('target_file', 'unknown'),
                'schema_version': '1.0.3',
                'coordinator_version': metadata.get('coordinator_version', 'unknown')
            },
            'analysis_parameters': {
                'prng_type': config.get('prng_type', 'unknown'),
                'total_seeds_tested': config.get('total_seeds', 0),
                'samples_per_seed': config.get('samples', 0),
                'lmax': config.get('lmax', 0),
                'grid_size': config.get('grid_size', 0),
                'dataset_file': config.get('target_file', 'unknown')
            },
            'results_summary': {
                'total_survivors': successful_jobs,  # Using jobs as proxy
                'survival_rate': success_rate,
                'analysis_complete': failed_jobs == 0
            },
            'performance_metrics': {
                'cluster_nodes_used': metadata.get('nodes_used', 0),
                'total_gpu_count': metadata.get('total_gpus', 0),
                'batches_processed': successful_jobs,
                'batches_failed': failed_jobs
            },
            'data_sources': {
                'dataset_name': config.get('target_file', 'unknown')
            }
        }

        # Inject agent_metadata if available
        if METADATA_WRITER_AVAILABLE:
            # Calculate confidence based on job success rate
            confidence = min(0.95, success_rate) if success_rate > 0 else 0.3
            
            formatted = inject_agent_metadata(
                formatted,
                inputs=[config.get("target_file") or "script_job"] if config.get("target_file") else [],
                outputs=[output_file],
                parent_run_id=parent_run_id,
                pipeline_step=pipeline_step,
                follow_up_agent=follow_up_agent,
                confidence=confidence,
                cluster_resources=get_default_cluster_resources(),
                success_criteria_met=(failed_jobs == 0),
                reasoning=f"Coordinator completed {successful_jobs}/{total_jobs} jobs successfully."
            )

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
        if METADATA_WRITER_AVAILABLE:
            print(f"   agent_metadata: injected ✓")
        return True

    except Exception as e:
        print(f"⚠️ New format failed: {e}")
        import traceback
        print(traceback.format_exc()[:300])
        return False


def save_coordinator_results_with_agent_context(
    run_id: str,
    final_results: Dict[str, Any],
    config: Dict[str, Any],
    output_file: str,
    agent_context: Dict[str, Any]
) -> bool:
    """
    Save coordinator results with full agent context.
    
    Convenience wrapper that unpacks agent_context dict.
    
    Args:
        run_id: Unique run identifier
        final_results: Results dict from coordinator (either format)
        config: Configuration dict
        output_file: Path to output file
        agent_context: Dict with keys:
            - parent_run_id
            - pipeline_step
            - follow_up_agent
            - suggested_params (optional)
    
    Returns:
        True if save successful
    """
    return save_coordinator_results(
        run_id=run_id,
        final_results=final_results,
        config=config,
        output_file=output_file,
        parent_run_id=agent_context.get('parent_run_id'),
        pipeline_step=agent_context.get('pipeline_step'),
        follow_up_agent=agent_context.get('follow_up_agent')
    )


def save_scripts_coordinator_results(
    final_results: Dict[str, Any],
    config: Dict[str, Any],
    output_file: str,
    pipeline_step: Optional[int] = None,
    follow_up_agent: Optional[str] = None
) -> bool:
    """
    Convenience function for scripts_coordinator.py results.
    
    Extracts run_id from results automatically.
    
    Args:
        final_results: Results dict from scripts_coordinator.py
        config: Configuration dict
        output_file: Path to output file
        pipeline_step: Optional pipeline step number (1-6)
        follow_up_agent: Optional next agent to trigger
    
    Returns:
        True if save successful
    """
    run_id = final_results.get('run_id', f"scripts_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    return save_coordinator_results(
        run_id=run_id,
        final_results=final_results,
        config=config,
        output_file=output_file,
        pipeline_step=pipeline_step,
        follow_up_agent=follow_up_agent
    )


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing coordinator_adapter.py v2.0.0...")
    print(f"metadata_writer available: {METADATA_WRITER_AVAILABLE}")
    
    # Test OLD format (coordinator.py)
    print("\n--- Testing OLD format (coordinator.py) ---")
    test_results_old = {
        'metadata': {
            'total_runtime': 123.45,
            'nodes_used': 3,
            'total_gpus': 26
        },
        'results': [{'job_id': 'test_001'}, {'job_id': 'test_002'}],
        'failed_jobs': []
    }
    
    normalized_old = normalize_coordinator_results(test_results_old)
    print(f"  Detected format: {normalized_old['metadata'].get('coordinator_version')}")
    print(f"  Nodes used: {normalized_old['metadata']['nodes_used']}")
    print(f"  Total GPUs: {normalized_old['metadata']['total_gpus']}")
    print(f"  Successful jobs: {len(normalized_old['results'])}")
    print(f"  Failed jobs: {len(normalized_old['failed_jobs'])}")
    
    # Test NEW format (scripts_coordinator.py)
    print("\n--- Testing NEW format (scripts_coordinator.py) ---")
    test_results_new = {
        'status': 'complete',
        'run_id': 'step3_20251218_191950',
        'total_jobs': 36,
        'successful': 36,
        'failed': 0,
        'runtime_seconds': 261.0,
        'timestamp': '2025-12-18T19:24:11',
        'output_dir': 'full_scoring_results/step3_20251218_191950',
        'nodes': {
            'localhost': {'jobs': 12, 'successful': 12, 'failed': 0, 'gpu_count': 2, 'gpu_type': 'RTX 3080 Ti'},
            '192.168.3.120': {'jobs': 12, 'successful': 12, 'failed': 0, 'gpu_count': 12, 'gpu_type': 'RX 6600'},
            '192.168.3.154': {'jobs': 12, 'successful': 12, 'failed': 0, 'gpu_count': 12, 'gpu_type': 'RX 6600'}
        },
        'jobs': [
            {'job_id': 'full_scoring_0000', 'success': True, 'node': 'localhost', 'gpu_id': 0, 'runtime': 14.0},
            {'job_id': 'full_scoring_0001', 'success': True, 'node': '192.168.3.120', 'gpu_id': 0, 'runtime': 17.9},
            {'job_id': 'full_scoring_0002', 'success': False, 'node': '192.168.3.154', 'gpu_id': 0, 'runtime': 16.1, 'error': 'test error'}
        ]
    }
    
    normalized_new = normalize_coordinator_results(test_results_new)
    print(f"  Detected format: {normalized_new['metadata'].get('coordinator_version')}")
    print(f"  Run ID: {normalized_new['metadata'].get('run_id')}")
    print(f"  Nodes used: {normalized_new['metadata']['nodes_used']}")
    print(f"  Total GPUs: {normalized_new['metadata']['total_gpus']}")
    print(f"  Successful jobs: {len(normalized_new['results'])}")
    print(f"  Failed jobs: {len(normalized_new['failed_jobs'])}")
    
    # Test format detection
    print("\n--- Testing format detection ---")
    print(f"  Old format is scripts_coordinator: {is_scripts_coordinator_format(test_results_old)}")
    print(f"  New format is scripts_coordinator: {is_scripts_coordinator_format(test_results_new)}")
    
    print("\n✅ coordinator_adapter.py v2.0.0 loaded successfully")
