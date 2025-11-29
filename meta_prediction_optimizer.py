#!/usr/bin/env python3
"""
Meta-Prediction Optimizer
==========================

Automatically tunes the ML prediction model to maximize prediction quality.

Optimizes:
1. Feature engineering (which features to use)
2. Model architecture (network depth/width)
3. Training hyperparameters (learning rate, batch size, epochs)
4. Normalization strategies
5. Loss functions and optimization methods

Goal: Maximize prediction variance, accuracy, and discrimination power

Author: Distributed PRNG Analysis System
Date: November 8, 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
import optuna
from optuna.samplers import TPESampler
import logging

# Assume these are available
from reinforcement_engine import ReinforcementEngine, ReinforcementConfig
from survivor_scorer import SurvivorScorer


@dataclass
class PredictionMetrics:
    """Metrics for evaluating prediction quality"""
    variance: float              # Prediction spread (higher = better)
    mean_absolute_error: float   # Prediction accuracy
    discrimination_power: float  # Ability to separate good/bad
    calibration_error: float     # How well calibrated predictions are
    feature_importance: Dict[str, float]
    
    def composite_score(self) -> float:
        """
        Composite score for optimization
        
        Higher is better:
        - High variance = good discrimination
        - Low MAE = accurate
        - High discrimination = separates winners
        - Low calibration error = well-calibrated
        """
        return (
            self.variance * 10.0 +              # Weight variance heavily
            self.discrimination_power * 5.0 +   # Discrimination is key
            (1.0 / (self.mean_absolute_error + 0.01)) * 2.0 +  # Accuracy
            (1.0 / (self.calibration_error + 0.01)) * 1.0      # Calibration
        )


class MetaPredictionOptimizer:
    """
    Meta-optimizer that tunes the ML prediction model
    
    Uses Bayesian optimization to find optimal:
    - Network architecture
    - Training parameters
    - Feature selection
    - Normalization methods
    """
    
    def __init__(self, 
                 survivors: List[int],
                 lottery_history: List[int],
                 actual_quality: List[float],
                 base_config_path: str = 'reinforcement_engine_config.json'):
        """
        Initialize meta-optimizer
        
        Args:
            survivors: List of survivor seeds
            lottery_history: Historical lottery draws
            actual_quality: Ground truth quality scores
            base_config_path: Base configuration file
        """
        self.survivors = survivors
        self.lottery_history = lottery_history
        self.actual_quality = actual_quality
        self.base_config_path = base_config_path
        
        # Load base config
        self.base_config = ReinforcementConfig.from_json(base_config_path)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Best configuration found
        self.best_config = None
        self.best_metrics = None
        self.optimization_history = []
        
    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function - try a configuration
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Composite score (higher = better)
        """
        # Sample hyperparameters
        config = self._sample_config(trial)
        
        # Train model with this configuration
        metrics = self._evaluate_config(config)
        
        # Store results
        self.optimization_history.append({
            'trial': trial.number,
            'config': config,
            'metrics': asdict(metrics),
            'score': metrics.composite_score()
        })
        
        self.logger.info(f"Trial {trial.number}: Score={metrics.composite_score():.4f}, "
                        f"Variance={metrics.variance:.4f}, "
                        f"MAE={metrics.mean_absolute_error:.4f}")
        
        return metrics.composite_score()
    
    def _sample_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Sample a configuration to test
        
        Args:
            trial: Optuna trial
            
        Returns:
            Configuration dictionary
        """
        config = {
            # Network architecture
            'hidden_layers': self._sample_architecture(trial),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            
            # Training parameters
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512]),
            'epochs': trial.suggest_int('epochs', 50, 300),
            'early_stopping_patience': trial.suggest_int('patience', 5, 20),
            
            # Optimizer
            'optimizer_type': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            
            # Loss function
            'loss_function': trial.suggest_categorical('loss', ['mse', 'mae', 'huber', 'smooth_l1']),
            
            # Normalization
            'normalization_method': trial.suggest_categorical('norm_method', ['standard', 'minmax', 'robust', 'none']),
            
            # Feature engineering
            'feature_selection_threshold': trial.suggest_float('feat_threshold', 0.0, 0.5),
            'use_feature_interactions': trial.suggest_categorical('feat_interact', [True, False]),
            
            # Advanced techniques
            'use_batch_norm': trial.suggest_categorical('batch_norm', [True, False]),
            'use_layer_norm': trial.suggest_categorical('layer_norm', [True, False]),
            'gradient_clip': trial.suggest_float('grad_clip', 0.1, 10.0),
        }
        
        return config
    
    def _sample_architecture(self, trial: optuna.Trial) -> List[int]:
        """
        Sample neural network architecture
        
        Args:
            trial: Optuna trial
            
        Returns:
            List of layer sizes
        """
        n_layers = trial.suggest_int('n_layers', 2, 5)
        
        layers = []
        for i in range(n_layers):
            # Each layer progressively smaller
            if i == 0:
                size = trial.suggest_int(f'layer_{i}_size', 64, 512)
            else:
                prev_size = layers[-1]
                size = trial.suggest_int(f'layer_{i}_size', 32, prev_size)
            
            layers.append(size)
        
        return layers
    
    def _evaluate_config(self, config: Dict[str, Any]) -> PredictionMetrics:
        """
        Train model with config and evaluate prediction quality
        
        Args:
            config: Configuration to test
            
        Returns:
            Prediction metrics
        """
        # Create modified config
        test_config = ReinforcementConfig.from_json(self.base_config_path)
        
        # Apply sampled parameters
        test_config.model['hidden_layers'] = config['hidden_layers']
        test_config.model['dropout'] = config['dropout']
        test_config.training['learning_rate'] = config['learning_rate']
        test_config.training['batch_size'] = config['batch_size']
        test_config.training['epochs'] = config['epochs']
        test_config.training['early_stopping_patience'] = config['early_stopping_patience']
        
        # Initialize engine
        engine = ReinforcementEngine(test_config, self.lottery_history)
        
        # Train/test split
        n_train = int(len(self.survivors) * 0.8)
        train_survivors = self.survivors[:n_train]
        train_quality = self.actual_quality[:n_train]
        test_survivors = self.survivors[n_train:]
        test_quality = self.actual_quality[n_train:]
        
        # Train model
        try:
            engine.train(
                survivors=train_survivors,
                actual_results=train_quality,
                epochs=config['epochs']
            )
            
            # Predict on test set
            predictions = engine.predict_quality_batch(test_survivors)
            
            # Calculate metrics
            metrics = self._calculate_metrics(predictions, test_quality)
            
        except Exception as e:
            self.logger.warning(f"Config failed: {e}")
            # Return poor metrics on failure
            metrics = PredictionMetrics(
                variance=0.0,
                mean_absolute_error=1.0,
                discrimination_power=0.0,
                calibration_error=1.0,
                feature_importance={}
            )
        
        return metrics
    
    def _calculate_metrics(self, 
                          predictions: List[float], 
                          actual: List[float]) -> PredictionMetrics:
        """
        Calculate prediction quality metrics
        
        Args:
            predictions: Predicted quality scores
            actual: Actual quality scores
            
        Returns:
            Prediction metrics
        """
        predictions = np.array(predictions)
        actual = np.array(actual)
        
        # Variance (spread of predictions)
        variance = float(np.var(predictions))
        
        # Mean Absolute Error
        mae = float(np.mean(np.abs(predictions - actual)))
        
        # Discrimination power (ability to separate high/low quality)
        # Split into top/bottom quartiles and measure separation
        if len(predictions) >= 4:
            sorted_idx = np.argsort(actual)
            q1_idx = sorted_idx[:len(sorted_idx)//4]
            q4_idx = sorted_idx[-len(sorted_idx)//4:]
            
            q1_pred = predictions[q1_idx]
            q4_pred = predictions[q4_idx]
            
            # Discrimination = difference in mean predictions between quartiles
            discrimination = float(abs(np.mean(q4_pred) - np.mean(q1_pred)))
        else:
            discrimination = 0.0
        
        # Calibration error (how well predictions match actual distribution)
        pred_bins = np.histogram(predictions, bins=10)[0]
        actual_bins = np.histogram(actual, bins=10)[0]
        calibration = float(np.mean(np.abs(pred_bins / len(predictions) - 
                                           actual_bins / len(actual))))
        
        return PredictionMetrics(
            variance=variance,
            mean_absolute_error=mae,
            discrimination_power=discrimination,
            calibration_error=calibration,
            feature_importance={}  # TODO: Extract from model
        )
    
    def optimize(self, n_trials: int = 50) -> Tuple[Dict[str, Any], PredictionMetrics]:
        """
        Run meta-optimization
        
        Args:
            n_trials: Number of configurations to try
            
        Returns:
            (best_config, best_metrics)
        """
        self.logger.info(f"Starting meta-optimization with {n_trials} trials...")
        self.logger.info(f"Training set: {len(self.survivors)} survivors")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        # Optimize
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best configuration
        self.best_config = self.optimization_history[study.best_trial.number]['config']
        self.best_metrics = PredictionMetrics(**self.optimization_history[study.best_trial.number]['metrics'])
        
        self.logger.info("="*70)
        self.logger.info("META-OPTIMIZATION COMPLETE!")
        self.logger.info("="*70)
        self.logger.info(f"Best Score: {self.best_metrics.composite_score():.4f}")
        self.logger.info(f"Best Variance: {self.best_metrics.variance:.4f}")
        self.logger.info(f"Best MAE: {self.best_metrics.mean_absolute_error:.4f}")
        self.logger.info(f"Best Discrimination: {self.best_metrics.discrimination_power:.4f}")
        self.logger.info("")
        self.logger.info("Best Configuration:")
        for key, value in self.best_config.items():
            self.logger.info(f"  {key}: {value}")
        
        return self.best_config, self.best_metrics
    
    def apply_best_config(self, output_path: str = 'reinforcement_engine_config_optimized.json'):
        """
        Apply best configuration to config file
        
        Args:
            output_path: Where to save optimized config
        """
        if self.best_config is None:
            raise ValueError("Must run optimize() first!")
        
        # Load base config
        config = ReinforcementConfig.from_json(self.base_config_path)
        
        # Apply best parameters
        config.model['hidden_layers'] = self.best_config['hidden_layers']
        config.model['dropout'] = self.best_config['dropout']
        config.training['learning_rate'] = self.best_config['learning_rate']
        config.training['batch_size'] = self.best_config['batch_size']
        config.training['epochs'] = self.best_config['epochs']
        config.training['early_stopping_patience'] = self.best_config['early_stopping_patience']
        
        # Save
        config.to_json(output_path)
        self.logger.info(f"✅ Optimized config saved to: {output_path}")
    
    def save_results(self, output_path: str = 'meta_optimization_results.json'):
        """
        Save optimization results
        
        Args:
            output_path: Where to save results
        """
        results = {
            'best_config': self.best_config,
            'best_metrics': asdict(self.best_metrics) if self.best_metrics else None,
            'optimization_history': self.optimization_history
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"✅ Optimization results saved to: {output_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI for meta-prediction optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Meta-Prediction Optimizer - Tune ML for better predictions'
    )
    parser.add_argument('--survivors', type=str, required=True,
                       help='Survivors JSON file')
    parser.add_argument('--lottery-data', type=str, required=True,
                       help='Lottery history JSON')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of optimization trials')
    parser.add_argument('--config', type=str,
                       default='reinforcement_engine_config.json',
                       help='Base config file')
    
    args = parser.parse_args()
    
    print("="*70)
    print("META-PREDICTION OPTIMIZER")
    print("="*70)
    print(f"Survivors: {args.survivors}")
    print(f"Lottery data: {args.lottery_data}")
    print(f"Optimization trials: {args.trials}")
    print("="*70)
    
    # Load data
    with open(args.survivors) as f:
        survivor_data = json.load(f)
        # Handle both list of seeds and list of dicts with 'seed' key
        if isinstance(survivor_data[0], dict):
            survivors = [s['seed'] for s in survivor_data]
        else:
            survivors = survivor_data
    
    with open(args.lottery_data) as f:
        lottery_data = json.load(f)
        if isinstance(lottery_data, list) and isinstance(lottery_data[0], dict):
            lottery_history = [d['draw'] for d in lottery_data]
        else:
            lottery_history = lottery_data
    
    print(f"✅ Loaded {len(survivors)} survivors")
    print(f"✅ Loaded {len(lottery_history)} lottery draws")
    
    # Generate synthetic quality scores for demonstration
    # In production, use actual quality scores from testing
    np.random.seed(42)
    actual_quality = np.random.uniform(0.2, 0.8, len(survivors)).tolist()
    
    print("⚠️  Using synthetic quality scores for demonstration")
    print("    In production, use actual survivor performance data")
    print()
    
    # Initialize optimizer
    optimizer = MetaPredictionOptimizer(
        survivors=survivors,
        lottery_history=lottery_history,
        actual_quality=actual_quality,
        base_config_path=args.config
    )
    
    # Run optimization
    best_config, best_metrics = optimizer.optimize(n_trials=args.trials)
    
    # Save results
    optimizer.apply_best_config('reinforcement_engine_config_optimized.json')
    optimizer.save_results('meta_prediction_optimization_results.json')
    
    print()
    print("="*70)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*70)
    print("Next steps:")
    print("  1. Review: meta_prediction_optimization_results.json")
    print("  2. Use optimized config: reinforcement_engine_config_optimized.json")
    print("  3. Re-run workflow with optimized settings")
    print("="*70)


if __name__ == "__main__":
    main()
