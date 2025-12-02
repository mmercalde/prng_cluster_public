#!/usr/bin/env python3
"""
================================================================================
SIEVE FILTER INTEGRATION ADAPTER
================================================================================

File: integration/sieve_integration.py
Version: 2.0.0
Created: 2025-11-03
Updated: 2025-12-01

PURPOSE:
--------
Adapter to integrate ResultsManager into sieve filters without modifying
the original code extensively.

UPDATES (v2.0.0):
-----------------
- Added agent_metadata injection via metadata_writer.py
- Supports Phase 2 agent infrastructure
- Bidirectional sieve now sets pipeline_step=1, follow_up_agent="scorer_meta_agent"
- Confidence calculated from survivor count and match rates

USAGE:
------
from integration.sieve_integration import save_forward_sieve_results
save_forward_sieve_results(survivors, config, direction='forward')

# With agent context:
save_bidirectional_sieve_results(
    forward_survivors, reverse_survivors, intersection, config,
    follow_up_agent="scorer_meta_agent"
)

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


def save_forward_sieve_results(
    survivors: List[Dict[str, Any]],
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    execution_time: Optional[float] = None,
    parent_run_id: Optional[str] = None
) -> bool:
    """
    Save forward sieve results using ResultsManager.
    
    Args:
        survivors: List of survivor dicts
        config: Sieve configuration
        run_id: Optional run identifier
        execution_time: Optional execution time in seconds
        parent_run_id: Optional parent run ID for lineage
    """
    return _save_sieve_results(
        survivors, config, 'forward', run_id, execution_time,
        parent_run_id=parent_run_id,
        pipeline_step=None,  # Forward-only is not a complete pipeline step
        follow_up_agent=None
    )


def save_reverse_sieve_results(
    survivors: List[Dict[str, Any]],
    config: Dict[str, Any],
    run_id: Optional[str] = None,
    execution_time: Optional[float] = None,
    parent_run_id: Optional[str] = None
) -> bool:
    """
    Save reverse sieve results using ResultsManager.
    
    Args:
        survivors: List of survivor dicts
        config: Sieve configuration
        run_id: Optional run identifier
        execution_time: Optional execution time in seconds
        parent_run_id: Optional parent run ID for lineage
    """
    return _save_sieve_results(
        survivors, config, 'reverse', run_id, execution_time,
        parent_run_id=parent_run_id,
        pipeline_step=None,  # Reverse-only is not a complete pipeline step
        follow_up_agent=None
    )


def save_bidirectional_sieve_results(
    forward_survivors: List[Dict],
    reverse_survivors: List[Dict],
    intersection: List[Dict],
    config: Dict,
    run_id: Optional[str] = None,
    execution_time: Optional[float] = None,
    parent_run_id: Optional[str] = None,
    follow_up_agent: Optional[str] = "scorer_meta_agent"
) -> bool:
    """
    Save bidirectional sieve results using ResultsManager.
    
    Bidirectional sieve is Pipeline Step 1 - the entry point.
    Default follow_up_agent is "scorer_meta_agent" (Step 2).
    
    Args:
        forward_survivors: Forward sieve survivors
        reverse_survivors: Reverse sieve survivors
        intersection: Bidirectional intersection (survivors passing both)
        config: Sieve configuration
        run_id: Optional run identifier
        execution_time: Optional execution time in seconds
        parent_run_id: Should be None for step 1 (no parent)
        follow_up_agent: Next agent to trigger (default: scorer_meta_agent)
    """
    enhanced_config = config.copy()
    enhanced_config['forward_count'] = len(forward_survivors)
    enhanced_config['reverse_count'] = len(reverse_survivors)
    
    return _save_sieve_results(
        intersection, enhanced_config, 'bidirectional', run_id, execution_time,
        parent_run_id=parent_run_id,  # Should be None for step 1
        pipeline_step=1,  # Bidirectional sieve IS step 1
        follow_up_agent=follow_up_agent
    )


def _save_sieve_results(
    survivors: List[Dict[str, Any]],
    config: Dict[str, Any],
    direction: str,
    run_id: Optional[str] = None,
    execution_time: Optional[float] = None,
    parent_run_id: Optional[str] = None,
    pipeline_step: Optional[int] = None,
    follow_up_agent: Optional[str] = None
) -> bool:
    """
    Internal function to save sieve results with agent_metadata.
    
    Args:
        survivors: List of survivor dicts
        config: Sieve configuration
        direction: 'forward', 'reverse', or 'bidirectional'
        run_id: Optional run identifier
        execution_time: Optional execution time
        parent_run_id: Parent run ID for lineage tracking
        pipeline_step: Pipeline step number (1-6)
        follow_up_agent: Next agent to trigger
    """

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
        total_matches = 0
        max_match_rate = 0.0
        
        for survivor in survivors:
            matches = survivor.get('matches', survivor.get('match_count', 0))
            match_rate = survivor.get('match_rate', 0.0)
            
            total_matches += matches
            max_match_rate = max(max_match_rate, match_rate)
            
            formatted_survivor = {
                'seed': survivor.get('seed', 0),
                'matches': matches,
                'total_draws': config.get('window_size', 244),
                'match_rate': match_rate,
                'skip_length': survivor.get('skip', survivor.get('skip_length', survivor.get('best_skip', 0))),
                'direction': direction
            }
            formatted_survivors.append(formatted_survivor)

        # Calculate survival rate
        total_seeds = config.get('total_seeds', 1)
        survival_rate = len(survivors) / total_seeds if total_seeds > 0 else 0.0

        # Build results data structure
        results_data = {
            'run_metadata': {
                'run_id': run_id,
                'analysis_type': analysis_type,
                'timestamp_start': datetime.now().isoformat(),
                'execution_time_seconds': execution_time,
                'schema_version': '1.0.3'
            },
            'analysis_parameters': {
                'prng_type': config.get('prng_type', 'unknown'),
                'seed_start': config.get('seed_start', 0),
                'seed_end': config.get('seed_end', 0),
                'total_seeds_tested': total_seeds,
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
                'survival_rate': survival_rate,
                'analysis_complete': True
            },
            'data_sources': {
                'dataset_name': config.get('dataset', 'unknown')
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

        # =================================================================
        # INJECT AGENT METADATA (Phase 2)
        # =================================================================
        if METADATA_WRITER_AVAILABLE:
            # Calculate confidence based on results quality
            confidence = _calculate_sieve_confidence(
                survivors=survivors,
                direction=direction,
                max_match_rate=max_match_rate,
                forward_count=config.get('forward_count', 0),
                reverse_count=config.get('reverse_count', 0)
            )
            
            # Determine output files
            output_files = [f"{direction}_survivors.json"]
            if direction == 'bidirectional':
                output_files.append("optimal_window_config.json")
            
            # Build reasoning string
            reasoning = _build_sieve_reasoning(
                direction=direction,
                survivor_count=len(survivors),
                total_seeds=total_seeds,
                max_match_rate=max_match_rate,
                forward_count=config.get('forward_count', 0),
                reverse_count=config.get('reverse_count', 0)
            )
            
            results_data = inject_agent_metadata(
                results_data,
                inputs=[config.get('dataset', 'unknown')],
                outputs=output_files,
                parent_run_id=parent_run_id,
                pipeline_step=pipeline_step,
                follow_up_agent=follow_up_agent,
                confidence=confidence,
                suggested_params={
                    'threshold': config.get('threshold', 0.012),
                    'window_size': config.get('window_size', 244),
                    'skip_range': [config.get('skip_min', 0), config.get('skip_max', 20)]
                } if direction == 'bidirectional' else None,
                reasoning=reasoning,
                success_criteria_met=(len(survivors) > 0),
                cluster_resources=get_default_cluster_resources()
            )

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
        if METADATA_WRITER_AVAILABLE:
            print(f"   agent_metadata: injected ✓")
            if pipeline_step:
                print(f"   pipeline_step: {pipeline_step}")
            if follow_up_agent:
                print(f"   follow_up_agent: {follow_up_agent}")
        return True

    except Exception as e:
        print(f"⚠️  New results format failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def _calculate_sieve_confidence(
    survivors: List[Dict],
    direction: str,
    max_match_rate: float,
    forward_count: int = 0,
    reverse_count: int = 0
) -> float:
    """
    Calculate confidence score for sieve results.
    
    Factors:
    - Having survivors at all (base confidence)
    - Best match rate (higher = more confident)
    - For bidirectional: overlap ratio between forward/reverse
    
    Returns:
        Confidence score 0.0 to 1.0
    """
    if not survivors:
        return 0.3  # Low confidence if no survivors
    
    # Base confidence for having survivors
    confidence = 0.5
    
    # Boost for good match rates
    if max_match_rate > 0.05:
        confidence += 0.2
    elif max_match_rate > 0.02:
        confidence += 0.1
    
    # Boost for bidirectional with good overlap
    if direction == 'bidirectional' and forward_count > 0 and reverse_count > 0:
        # Calculate overlap ratio
        min_count = min(forward_count, reverse_count)
        overlap_ratio = len(survivors) / min_count if min_count > 0 else 0
        
        if overlap_ratio > 0.01:  # At least 1% overlap
            confidence += 0.15
        if overlap_ratio > 0.001:  # At least 0.1% overlap
            confidence += 0.05
    
    # Cap at 0.95 (never 100% confident)
    return min(0.95, confidence)


def _build_sieve_reasoning(
    direction: str,
    survivor_count: int,
    total_seeds: int,
    max_match_rate: float,
    forward_count: int = 0,
    reverse_count: int = 0
) -> str:
    """Build human-readable reasoning string for sieve results."""
    
    survival_rate = survivor_count / total_seeds if total_seeds > 0 else 0
    
    parts = []
    
    if direction == 'bidirectional':
        parts.append(f"Bidirectional sieve found {survivor_count:,} survivors from {total_seeds:,} seeds tested.")
        if forward_count > 0 and reverse_count > 0:
            overlap_pct = (survivor_count / min(forward_count, reverse_count)) * 100 if min(forward_count, reverse_count) > 0 else 0
            parts.append(f"Forward: {forward_count:,}, Reverse: {reverse_count:,}, Overlap: {overlap_pct:.2f}%.")
    else:
        parts.append(f"{direction.capitalize()} sieve found {survivor_count:,} survivors ({survival_rate:.6%} survival rate).")
    
    if max_match_rate > 0:
        parts.append(f"Best match rate: {max_match_rate:.2%}.")
    
    if survivor_count > 0 and direction == 'bidirectional':
        parts.append("Recommend proceeding to scorer meta-optimization.")
    elif survivor_count == 0:
        parts.append("No survivors found - consider adjusting parameters.")
    
    return " ".join(parts)


# =============================================================================
# Module Test
# =============================================================================

if __name__ == '__main__':
    print("Testing sieve integration adapter...")
    print(f"ResultsManager available: {RESULTS_MANAGER_AVAILABLE}")
    print(f"metadata_writer available: {METADATA_WRITER_AVAILABLE}")

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

    print("\n--- Testing forward sieve ---")
    success = save_forward_sieve_results(test_survivors, test_config)

    if success:
        print("\n✅ Forward sieve adapter test passed!")
    else:
        print("\n⚠️  Forward sieve adapter test failed (may be expected if ResultsManager not configured)")

    print("\n--- Testing bidirectional sieve ---")
    # Simulated bidirectional test
    forward = [{'seed': 12345, 'matches': 10, 'match_rate': 0.041}]
    reverse = [{'seed': 12345, 'matches': 8, 'match_rate': 0.033}]
    intersection = [{'seed': 12345, 'matches': 10, 'match_rate': 0.041}]
    
    success2 = save_bidirectional_sieve_results(
        forward, reverse, intersection, test_config,
        follow_up_agent="scorer_meta_agent"
    )
    
    if success2:
        print("\n✅ Bidirectional sieve adapter test passed!")
    else:
        print("\n⚠️  Bidirectional sieve adapter test failed (may be expected if ResultsManager not configured)")

    print("\n✅ sieve_integration.py loaded successfully")
