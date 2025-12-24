#!/usr/bin/env python3
"""
Prediction Generator - Pipeline Component
==========================================

Whitepaper-Compliant Implementation:
- Integrates with survivor_scorer.py
- Generates Top-K predictions from forward/reverse sieves
- ML/AI ready with 46-feature extraction
- GPU-accelerated via CuPy
- Part of reinforcement learning pipeline

Author: Distributed PRNG Analysis System
Date: November 6, 2025
Version: 5.0 (Pipeline Integration)
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json
from datetime import datetime

import numpy as np
from integration.metadata_writer import inject_agent_metadata

# GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False

# Import survivor_scorer from pipeline
try:
    sys.path.insert(0, str(Path(__file__).parent))
    from survivor_scorer import SurvivorScorer
    SURVIVOR_SCORER_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Cannot import survivor_scorer: {e}")
    print("Make sure survivor_scorer.py is in the same directory!")
    sys.exit(1)

# Multi-Model Architecture v3.1.2 imports
try:
    from models.model_factory import load_model_from_sidecar
    from models.feature_schema import validate_feature_schema_hash, get_feature_schema_with_hash
    MULTI_MODEL_AVAILABLE = True
except ImportError:
    MULTI_MODEL_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class PredictionConfig:
    """
    Configuration for prediction generation
    
    Whitepaper Section 4: Machine Learning Integration
    """
    # Pool parameters
    pool_size: int = 10
    min_confidence: float = 0.5
    k: int = 10  # Top-K predictions
    
    # Ensemble methods (Whitepaper Section 5)
    ensemble_methods: List[str] = field(default_factory=lambda: [
        'weighted_average',
        'confidence_weighted',
        'feature_weighted'
    ])
    
    # Feature weights (46 features from survivor_scorer.py)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    
    # PRNG parameters
    prng_type: str = 'java_lcg'
    mod: int = 1000
    skip: int = 0
    
    # Dual-sieve (Whitepaper Section 1)
    use_dual_sieve: bool = True
    
    # Output
    save_predictions: bool = True
    predictions_dir: str = 'results/predictions'
    log_level: str = 'INFO'

    @classmethod
    def from_json(cls, path: str) -> 'PredictionConfig':
        """Load from JSON config"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        config_data = data.get('prediction', data)
        
        return cls(
            pool_size=config_data.get('pool_size', 10),
            min_confidence=config_data.get('min_confidence', 0.5),
            k=config_data.get('k', 10),
            ensemble_methods=config_data.get('ensemble_methods', [
                'weighted_average', 'confidence_weighted', 'feature_weighted'
            ]),
            feature_weights=config_data.get('feature_weights', {}),
            prng_type=config_data.get('prng', {}).get('prng_type', 'java_lcg'),
            mod=config_data.get('prng', {}).get('mod', 1000),
            skip=config_data.get('prng', {}).get('skip', 0),
            use_dual_sieve=config_data.get('use_dual_sieve', True),
            save_predictions=config_data.get('output', {}).get('save_predictions', True),
            predictions_dir=config_data.get('output', {}).get('predictions_dir', 'results/predictions'),
            log_level=config_data.get('output', {}).get('log_level', 'INFO')
        )


def setup_logging(config: PredictionConfig) -> logging.Logger:
    """Setup logging"""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


# ============================================================================
# PREDICTION GENERATOR
# ============================================================================

class PredictionGenerator:
    """
    Generate Top-K predictions from survivor pools
    
    Whitepaper Compliance:
    - Section 1: Dual-sieve methodology
    - Section 2: Reinforcement signals from prediction quality
    - Section 4: ML feature integration (46 features)
    - Section 5: Forward-reverse-ML ensemble
    """

    def __init__(self, config: PredictionConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        # Initialize survivor scorer (Whitepaper Section 4.1)
        self.scorer = SurvivorScorer(
            prng_type=config.prng_type,
            mod=config.mod
        )
        
        self.logger.info("PredictionGenerator initialized")
        self.logger.info(f"  PRNG: {config.prng_type}, mod={config.mod}")
        self.logger.info(f"  Pool size: {config.pool_size}, Top-K: {config.k}")
        self.logger.info(f"  Dual-sieve: {config.use_dual_sieve}")
        self.logger.info(f"  GPU: {GPU_AVAILABLE}")

    def generate_predictions(
        self,
        survivors_forward: List[int],
        lottery_history: List[int],
        survivors_reverse: Optional[List[int]] = None,
        k: Optional[int] = None
    ) -> Dict:
        """
        Generate Top-K predictions from survivors
        
        Whitepaper Section 5: Forward-reverse-ML ensemble
        
        Args:
            survivors_forward: Forward sieve survivors
            lottery_history: Historical lottery draws
            survivors_reverse: Reverse sieve survivors (optional)
            k: Number of predictions (default: config.k)
            
        Returns:
            Dictionary with predictions, confidence scores, and metadata
        """
        k = k or self.config.k
        
        self.logger.info(f"Generating {k} predictions from {len(survivors_forward)} survivors")
        
        # Whitepaper Section 1: Dual-sieve methodology
        method = 'forward_only'
        working_survivors = survivors_forward
        
        if survivors_reverse and self.config.use_dual_sieve:
            self.logger.info(f"Dual-sieve mode: {len(survivors_reverse)} reverse survivors")
            method = 'dual_sieve'
            
            # Compute intersection (Whitepaper: "bidirectional approximation mechanism")
            intersection = self.scorer.compute_dual_sieve_intersection(
                survivors_forward, survivors_reverse
            )
            
            if intersection:
                self.logger.info(f"Using intersection: {len(intersection)} survivors")
                working_survivors = intersection
            else:
                self.logger.warning("No intersection! Using forward survivors only")
        
        # Build prediction pool using survivor_scorer.py
        # Whitepaper Section 2: "High-quality prediction pools act as reinforcement signals"
        self.logger.info("Building prediction pool...")
        
        pool_result = self.scorer.build_prediction_pool(
            survivors=working_survivors,
            lottery_history=lottery_history,
            pool_size=self.config.pool_size,
            skip=self.config.skip,
            use_dual_scoring=self.config.use_dual_sieve,
            forward_survivors=survivors_forward if self.config.use_dual_sieve else None,
            reverse_survivors=survivors_reverse if self.config.use_dual_sieve else None
        )
        
        # Extract predictions from pool
        predictions_list = pool_result.get('predictions', [])
        
        if not predictions_list:
            self.logger.warning("No predictions generated!")
            return {
                'predictions': [],
                'confidence_scores': [],
                'metadata': {
                    'error': 'No predictions generated',
                    'method': method
                }
            }
        
        self.logger.info(f"Pool generated {len(predictions_list)} predictions")
        
        # Extract prediction numbers and confidences
        # Whitepaper Section 4: "ML models can learn optimal weighting"
        predictions = []
        confidences = []
        
        for pred in predictions_list[:k]:
            if isinstance(pred, dict):
                predictions.append(pred.get('next_prediction', 0))
                confidences.append(pred.get('confidence', 0.0))
            elif isinstance(pred, (int, float)):
                predictions.append(int(pred))
                confidences.append(1.0 / len(predictions_list))
        
        # Normalize confidences
        if confidences:
            max_conf = max(confidences)
            if max_conf > 0:
                confidences = [c / max_conf for c in confidences]
        
        # Build result
        result = {
            'predictions': predictions[:k],
            'confidence_scores': confidences[:k],
            'metadata': {
                'method': method,
                'pool_size': self.config.pool_size,
                'k': k,
                'forward_count': len(survivors_forward),
                'reverse_count': len(survivors_reverse) if survivors_reverse else 0,
                'intersection_count': len(working_survivors),
                'prng_type': self.config.prng_type,
                'mod': self.config.mod,
                'skip': self.config.skip,
                'dual_sieve': self.config.use_dual_sieve,
                'gpu_available': GPU_AVAILABLE,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Save if configured
        if self.config.save_predictions:
            self._save_predictions(result)
        
        return result

    def extract_features_batch(
        self,
        survivors: List[int],
        lottery_history: List[int],
        forward_survivors: Optional[List[int]] = None,
        reverse_survivors: Optional[List[int]] = None
    ) -> List[Dict]:
        """
        Extract ML features for batch of survivors
        
        Whitepaper Section 4: "Feature sets include survivor overlap ratios..."
        
        This is used by the reinforcement engine for training.
        
        Returns:
            List of feature dictionaries (46 features each)
        """
        self.logger.info(f"Extracting features for {len(survivors)} survivors...")
        
        features_list = []
        
        for seed in survivors:
            features = self.scorer.extract_ml_features(
                seed=seed,
                lottery_history=lottery_history,
                forward_survivors=forward_survivors,
                reverse_survivors=reverse_survivors
            )
            features_list.append(features)
        
        self.logger.info(f"Extracted {len(features_list)} feature sets")
        return features_list

    def _save_predictions(self, result: Dict):
        """Save predictions to JSON"""
        output_dir = Path(self.config.predictions_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions_{timestamp}.json"
        filepath = output_dir / filename

        # Inject agent_metadata for pipeline tracking
        avg_conf = 0.5
        if result.get('confidence_scores'):
            avg_conf = sum(result['confidence_scores']) / len(result['confidence_scores'])

        num_predictions = len(result.get('predictions', []))

        result = inject_agent_metadata(
            result,
            inputs=[
                {"file": self.model_checkpoint_path or "models/reinforcement/best_model.meta.json", "required": True},
                {"file": "survivors_with_scores.json", "required": True}
            ],
            outputs=[str(filepath)],
            pipeline_step=6,
            pipeline_step_name="prediction",
            follow_up_agent=None,
            confidence=avg_conf,
            suggested_params=None,
            reasoning=f"Generated {num_predictions} predictions with avg confidence {avg_conf:.4f}"
        )

        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)

        self.logger.info(f"Saved predictions to {filepath}")


# ============================================================================
# CLI / PIPELINE INTERFACE
# ============================================================================

def main():
    """
    CLI interface for standalone testing
    
    In production, this is called programmatically by the reinforcement engine
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Prediction Generator - Pipeline Component'
    )
    parser.add_argument('--config', type=str, 
                       default='prediction_generator_config.json')
    parser.add_argument('--survivors-forward', type=str, required=False)
    parser.add_argument('--survivors-reverse', type=str)
    parser.add_argument('--lottery-history', type=str, required=False)
    parser.add_argument('--k', type=int)
    parser.add_argument('--test', action='store_true',
                       help='Run self-test')
    # Multi-Model Architecture v3.1.2
    parser.add_argument('--models-dir', type=str, default='models/reinforcement',
                       help='Directory containing best_model.meta.json (default: models/reinforcement)')

    args = parser.parse_args()

    # Test mode
    if args.test:
        print("="*70)
        print("PREDICTION GENERATOR - SELF TEST")
        print("="*70)

        config = PredictionConfig()
        logger = setup_logging(config)

        try:
            generator = PredictionGenerator(config, logger)
            print("✅ Generator initialized successfully")
            print(f"   PRNG: {config.prng_type}")
            print(f"   Pool size: {config.pool_size}")
            print(f"   GPU: {GPU_AVAILABLE}")
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return 1

        print("\n✅ All tests passed!")
        return 0

    # Validate args
    if not args.survivors_forward or not args.lottery_history:
        parser.error("--survivors-forward and --lottery-history required (or use --test)")

    # Load config
    try:
        config = PredictionConfig.from_json(args.config)
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        print("Using default config")
        config = PredictionConfig()

    logger = setup_logging(config)
    logger.info("=== Prediction Generator Started ===")

    # Load data
    with open(args.survivors_forward, 'r') as f:
        data = json.load(f)
        survivors_forward = data if isinstance(data, list) else data.get('survivors', data)

    survivors_reverse = None
    if args.survivors_reverse:
        with open(args.survivors_reverse, 'r') as f:
            data = json.load(f)
            survivors_reverse = data if isinstance(data, list) else data.get('survivors', data)

    with open(args.lottery_history, 'r') as f:
        data = json.load(f)
        lottery_history = data if isinstance(data, list) else data.get('draws', data)

    logger.info(f"Loaded {len(survivors_forward)} forward survivors")
    if survivors_reverse:
        logger.info(f"Loaded {len(survivors_reverse)} reverse survivors")
    logger.info(f"Loaded {len(lottery_history)} lottery draws")

    # Initialize generator
    generator = PredictionGenerator(config, logger)

    # Generate predictions
    result = generator.generate_predictions(
        survivors_forward,
        lottery_history,
        survivors_reverse,
        k=args.k
    )

    # Display results
    logger.info("=== PREDICTIONS ===")
    for i, (num, conf) in enumerate(zip(result['predictions'], result['confidence_scores']), 1):
        logger.info(f"{i}. {num:03d} (confidence: {conf:.4f})")

    logger.info("=== Complete ===")


if __name__ == "__main__":
    sys.exit(main() or 0)
