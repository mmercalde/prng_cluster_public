#!/usr/bin/env python3
"""
aggregate_reinforcement_shards.py - Aggregate distributed reinforcement results
Combines model states from all GPU shards into final unified model
WITH ROCm SUPPORT - Compatible with AMD RX 6600 GPUs
"""

import json
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# ============================================================================
# ROCm environment setup - MUST BE BEFORE TORCH IMPORT
# ============================================================================
import os
import socket
HOST = socket.gethostname()
if HOST in ["rig-6600", "rig-6600b"]:
    os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
    os.environ.setdefault("HSA_ENABLE_SDMA", "0")
os.environ.setdefault("ROCM_PATH", "/opt/rocm")
os.environ.setdefault("HIP_PATH", "/opt/rocm")
# ============================================================================

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_shard_results(results_dir: str, num_shards: int) -> List[Dict[str, Any]]:
    """Load all shard results from disk"""
    results = []
    missing_shards = []

    for shard_id in range(num_shards):
        result_file = Path(results_dir) / f"reinforce_shard_{shard_id}.json"

        if not result_file.exists():
            logger.warning(f"⚠️  Missing shard {shard_id}: {result_file}")
            missing_shards.append(shard_id)
            continue

        try:
            with open(result_file, 'r') as f:
                result = json.load(f)

            if result.get('status') != 'SUCCESS':
                logger.warning(f"⚠️  Shard {shard_id} failed: {result.get('error', 'Unknown')}")
                missing_shards.append(shard_id)
                continue

            results.append(result)
            logger.info(f"✅ Loaded shard {shard_id}: val_loss={result['best_val_loss']:.6f}")

        except Exception as e:
            logger.error(f"❌ Error loading shard {shard_id}: {e}")
            missing_shards.append(shard_id)

    if missing_shards:
        logger.warning(f"⚠️  Missing {len(missing_shards)}/{num_shards} shards: {missing_shards}")

    return results


def aggregate_model_states(shard_results: List[Dict[str, Any]],
                          weighting: str = 'uniform') -> Dict[str, Any]:
    """
    Aggregate model parameters from all shards

    Args:
        shard_results: List of shard result dictionaries
        weighting: 'uniform', 'performance', or 'size'
    """
    logger.info(f"🔄 Aggregating {len(shard_results)} model states...")
    logger.info(f"   Weighting strategy: {weighting}")

    if not shard_results:
        raise ValueError("No shard results to aggregate!")

    # Calculate weights
    if weighting == 'uniform':
        weights = [1.0 / len(shard_results)] * len(shard_results)

    elif weighting == 'performance':
        # Weight by inverse validation loss (better models contribute more)
        val_losses = [r['best_val_loss'] for r in shard_results]
        inv_losses = [1.0 / (loss + 1e-8) for loss in val_losses]
        total = sum(inv_losses)
        weights = [w / total for w in inv_losses]
        logger.info(f"   Performance weights: {[f'{w:.3f}' for w in weights]}")

    elif weighting == 'size':
        # Weight by shard size (more data = more weight)
        sizes = [r['shard_size'] for r in shard_results]
        total = sum(sizes)
        weights = [s / total for s in sizes]
        logger.info(f"   Size weights: {[f'{w:.3f}' for w in weights]}")

    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Get first model state as template
    template_state = shard_results[0]['model_state']
    aggregated_state = {}

    # Average each parameter across all shards
    for param_name in template_state.keys():
        # Stack parameters from all shards
        param_arrays = []
        for result, weight in zip(shard_results, weights):
            param_data = np.array(result['model_state'][param_name])
            param_arrays.append(param_data * weight)

        # Weighted average
        aggregated_param = sum(param_arrays)
        aggregated_state[param_name] = aggregated_param.tolist()

        logger.debug(f"   Aggregated {param_name}: shape={aggregated_param.shape}")

    logger.info(f"✅ Aggregated {len(aggregated_state)} parameters")

    return aggregated_state


def aggregate_metrics(shard_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate training metrics across shards"""
    metrics = {
        'num_shards': len(shard_results),
        'total_survivors': sum(r['shard_size'] for r in shard_results),
        'best_val_losses': [r['best_val_loss'] for r in shard_results],
        'mean_val_loss': np.mean([r['best_val_loss'] for r in shard_results]),
        'std_val_loss': np.std([r['best_val_loss'] for r in shard_results]),
        'min_val_loss': min(r['best_val_loss'] for r in shard_results),
        'max_val_loss': max(r['best_val_loss'] for r in shard_results),
        'best_shard_id': min(shard_results, key=lambda r: r['best_val_loss'])['shard_id'],
        'convergence_epochs': [r['best_epoch'] for r in shard_results],
        'mean_convergence_epoch': np.mean([r['best_epoch'] for r in shard_results])
    }

    return metrics


def save_aggregated_model(model_state: Dict[str, Any],
                          metrics: Dict[str, Any],
                          hyperparams: Dict[str, Any],
                          output_path: str):
    """Save aggregated model in format compatible with ReinforcementEngine"""
    try:
        import torch
    except ImportError:
        logger.error("PyTorch required for model saving")
        sys.exit(1)

    # Convert lists back to tensors
    state_dict = {
        k: torch.tensor(v) for k, v in model_state.items()
    }

    # Create checkpoint compatible with ReinforcementEngine.load_model()
    checkpoint = {
        'model_state_dict': state_dict,
        'aggregation_info': {
            'num_shards': metrics['num_shards'],
            'total_survivors': metrics['total_survivors'],
            'aggregation_method': 'distributed_averaging',
            'mean_val_loss': metrics['mean_val_loss'],
            'min_val_loss': metrics['min_val_loss']
        },
        'metrics': metrics,
        'hyperparams': hyperparams,
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'aggregated': True
    }

    torch.save(checkpoint, output_path)
    logger.info(f"💾 Saved aggregated model to {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Aggregate distributed reinforcement training results (ROCm compatible)'
    )
    parser.add_argument('--results-dir', required=True,
                       help='Directory containing shard result JSONs')
    parser.add_argument('--num-shards', type=int, required=True,
                       help='Expected number of shards')
    parser.add_argument('--weighting', type=str,
                       choices=['uniform', 'performance', 'size'],
                       default='performance',
                       help='Model aggregation weighting strategy')
    parser.add_argument('--output', required=True,
                       help='Output path for aggregated model (.pth)')
    parser.add_argument('--metrics-output', type=str,
                       help='Optional: Save aggregation metrics to JSON')
    parser.add_argument('--min-shards', type=int,
                       help='Minimum shards required (default: all)')

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("REINFORCEMENT SHARD AGGREGATION (ROCm Compatible)")
    logger.info("="*70)
    logger.info(f"Hostname: {HOST}")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Expected shards: {args.num_shards}")
    logger.info(f"Weighting: {args.weighting}")

    # Load shard results
    shard_results = load_shard_results(args.results_dir, args.num_shards)

    # Check minimum shards
    min_shards = args.min_shards or args.num_shards
    if len(shard_results) < min_shards:
        logger.error(f"❌ Insufficient shards: {len(shard_results)}/{min_shards}")
        logger.error("   Cannot aggregate - too many failed shards")
        sys.exit(1)

    logger.info(f"✅ Loaded {len(shard_results)}/{args.num_shards} shards")

    # Aggregate model states
    try:
        aggregated_state = aggregate_model_states(shard_results, args.weighting)
    except Exception as e:
        logger.error(f"❌ Model aggregation failed: {e}")
        sys.exit(1)

    # Aggregate metrics
    metrics = aggregate_metrics(shard_results)

    logger.info("\n" + "="*70)
    logger.info("AGGREGATION SUMMARY")
    logger.info("="*70)
    logger.info(f"Shards: {metrics['num_shards']}")
    logger.info(f"Total survivors: {metrics['total_survivors']}")
    logger.info(f"Mean val loss: {metrics['mean_val_loss']:.6f} ± {metrics['std_val_loss']:.6f}")
    logger.info(f"Best val loss: {metrics['min_val_loss']:.6f} (shard {metrics['best_shard_id']})")
    logger.info(f"Mean convergence: {metrics['mean_convergence_epoch']:.1f} epochs")

    # Get hyperparams from first shard
    hyperparams = shard_results[0].get('hyperparams', {})

    # Save aggregated model
    try:
        save_aggregated_model(
            aggregated_state,
            metrics,
            hyperparams,
            args.output
        )
        logger.info(f"\n✅ Aggregated model saved: {args.output}")
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        sys.exit(1)

    # Save metrics if requested
    if args.metrics_output:
        with open(args.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"📊 Metrics saved: {args.metrics_output}")

    logger.info("\n✅ Aggregation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
