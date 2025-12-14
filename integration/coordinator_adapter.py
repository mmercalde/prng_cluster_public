#!/usr/bin/env python3
"""
Coordinator Integration Adapter
Bridge between coordinator.py and results_manager

Updated: 2025-12-01
- Added agent_metadata injection via metadata_writer.py
- Supports Phase 2 agent infrastructure
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
    
    Args:
        run_id: Unique run identifier
        final_results: Results dict from coordinator
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
        metadata = final_results.get('metadata', {})
        
        # Calculate success metrics
        successful_jobs = len(final_results.get('results', []))
        failed_jobs = len(final_results.get('failed_jobs', []))
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
                'schema_version': '1.0.3'
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
        final_results: Results dict from coordinator
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


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing coordinator_adapter.py...")
    print(f"metadata_writer available: {METADATA_WRITER_AVAILABLE}")
    
    # Create test data
    test_results = {
        'metadata': {
            'total_runtime': 123.45,
            'nodes_used': 3,
            'total_gpus': 26
        },
        'results': [{'job_id': 'test_001'}, {'job_id': 'test_002'}],
        'failed_jobs': []
    }
    
    test_config = {
        'prng_type': 'java_lcg',
        'total_seeds': 1000000,
        'samples': 10000,
        'lmax': 32,
        'grid_size': 16,
        'target_file': 'test_known.json'
    }
    
    print("\nTest data prepared. To run full test, uncomment save call.")
    # Uncomment to actually save:
    # success = save_coordinator_results(
    #     run_id='test_coordinator_adapter',
    #     final_results=test_results,
    #     config=test_config,
    #     output_file='results/test_output.json',
    #     pipeline_step=1,
    #     follow_up_agent='scorer_meta_agent'
    # )
    # print(f"Save result: {success}")
    
    print("\n✅ coordinator_adapter.py loaded successfully")
