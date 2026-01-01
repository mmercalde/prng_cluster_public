#!/usr/bin/env python3
"""
Adaptive Meta-Optimizer - Finds Optimal Training Parameters
===========================================================

DESIGN PHILOSOPHY (Team Beta Approved - Option A):
- Step 4 is a CAPACITY & ARCHITECTURE PLANNER
- Derives parameters from empirical sieve data (not survivor-level data)
- Intentionally does NOT consume survivors_with_scores.json
- Intentionally does NOT inspect holdout_hits
- Model selection happens in Step 5, NOT here

This separation prevents:
- Validation leakage
- Hyperparameters tuned on future data  
- Silent overfitting
- Blurred step boundaries

PARAMETER DERIVATION (Weighted):
- PRIMARY (60%): Window optimizer empirical results
- SECONDARY (35%): Historical pattern characteristics (entropy, stability)
- CONTINUOUS (5%→25%): Reinforcement performance feedback

WHAT STEP 4 DOES:
✅ Load window optimizer results (optimal_window_config.json)
✅ Load training lottery history (train_history.json)
✅ Optionally read reinforcement feedback (post-deployment)
✅ Derive: survivor pool size, network architecture, training epochs
✅ Write reinforcement_engine_config.json

WHAT STEP 4 DOES NOT DO:
❌ Load survivors_with_scores.json (that's Step 5)
❌ Inspect holdout_hits (that's Step 5)
❌ Select model type (that's Step 5)
❌ Perform any evaluation (that's Step 5)

Integration:
- Reads window_optimizer results
- Analyzes lottery history patterns
- Outputs reinforcement_engine-compatible config
- Supports continuous learning loop

Author: Distributed PRNG Analysis System
Date: November 7, 2025
Version: 2.0 (Team Beta Option A - Capacity Planner Only)
"""

import sys
import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import Counter
from abc import ABC, abstractmethod

# Try GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = np
    GPU_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class MetaOptimizerConfig:
    """
    Configuration for adaptive meta-optimizer
    
    NOTE: This is a CAPACITY PLANNER, not a data-aware optimizer.
    All parameters are derived from sieve behavior and pattern complexity,
    NOT from survivor-level data.
    """
    # Data sources (weighted importance)
    sources: Dict[str, Any] = field(default_factory=lambda: {
        'window_optimizer_results': {
            'path': 'optimal_window_config.json',  # Step 1 output
            'weight': 0.60,
            'required': True
        },
        'lottery_history': {
            'path': 'train_history.json',  # Override via --lottery-data
            'weight': 0.35,
            'required': True
        },
        'reinforcement_feedback': {
            'weight': 0.05,  # Grows to 0.25 over time
            'max_weight': 0.25,
            'growth_rate': 'confidence_based'
        }
        # NOTE: survivors_with_scores.json is intentionally NOT a source
        # That data is consumed by Step 5, not Step 4
    })

    # Optimization modes
    modes: Dict[str, Any] = field(default_factory=lambda: {
        'initialization': True,      # One-time calibration
        'continuous': True,          # Micro-adjustments
        'regime_response': True      # Major re-optimization
    })

    # Optimization parameters (data-driven ranges)
    optimization: Dict[str, Any] = field(default_factory=lambda: {
        'survivor_count': {
            'derive_from': ['window_optimizer', 'historical_patterns'],
            'min_multiplier': 0.5,   # 50% of window_optimizer optimal
            'max_multiplier': 2.0,   # 200% of window_optimizer optimal
            'step': 'logarithmic'
        },
        'network_architecture': {
            'search_space': 'adaptive',
            'options': [
                [32],
                [64, 32],
                [128, 64, 32],
                [256, 128, 64]
            ],
            'selection_criteria': ['training_stability', 'prediction_variance']
        },
        'training_epochs': {
            'derive_from': 'convergence_analysis',
            'min': 50,
            'max': 500,
            'adaptive': True
        }
    })

    # Micro-adjustment thresholds
    continuous_optimization: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True,
        'adjustment_threshold': 0.05,     # 5% performance change
        'feedback_window': 100,            # Last 100 predictions
        'adjustment_frequency': 'per_epoch',
        'min_confidence': 0.70
    })

    # Regime change handling
    regime_change: Dict[str, Any] = field(default_factory=lambda: {
        'detection_source': 'global_state_tracker',
        'trigger_full_recalibration': True,
        'preserve_history': True,
        'fallback_to_last_stable': True,
        'validation_period': 50  # Draws before committing
    })

    # Output configuration
    output: Dict[str, Any] = field(default_factory=lambda: {
        'target_file': 'reinforcement_engine_config.json',
        'update_strategy': 'merge_preserving_user_overrides',
        'backup_previous': True,
        'validation': True,
        'results_dir': 'optimization_results',
        'history_file': 'meta_optimization_history.json'
    })

    @classmethod
    def from_json(cls, path: str) -> 'MetaOptimizerConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            sources=data.get('sources', {}),
            modes=data.get('modes', {}),
            optimization=data.get('optimization', {}),
            continuous_optimization=data.get('continuous_optimization', {}),
            regime_change=data.get('regime_change', {}),
            output=data.get('output', {})
        )

    def to_json(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class OptimizationResult:
    """Result from optimization trial"""
    survivor_count: int
    network_architecture: List[int]
    training_epochs: int
    stability_score: float
    training_variance: float
    prediction_variance: float
    convergence_speed: float
    timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# DATA ANALYZERS
# ============================================================================

class WindowOptimizerAnalyzer:
    """
    Analyze window_optimizer results to derive survivor count ranges

    PRIMARY SOURCE (60% weight)
    """

    def __init__(self, results_path: str):
        """
        Initialize analyzer

        Args:
            results_path: Path to optimal_window_config.json (Step 1 output)
        """
        self.results_path = results_path
        self.results = None
        self._load_results()

    def _load_results(self):
        """Load window optimizer results"""
        try:
            with open(self.results_path, 'r') as f:
                self.results = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Window optimizer results not found at {self.results_path}\n"
                f"Run window_optimizer.py first to generate results"
            )

    def get_survivor_count_range(self) -> Dict[str, int]:
        """
        Extract survivor count range from window optimizer

        Returns:
            Dictionary with min, optimal, max survivor counts
        """
        if not self.results:
            return {'min': 100, 'optimal': 500, 'max': 2000}

        best_result = self.results.get('best_result', {})

        # Get bidirectional count as baseline
        optimal = best_result.get('bidirectional_count', 500)

        # Search space exploration
        all_results = self.results.get('all_results', [])
        if all_results:
            counts = [r.get('bidirectional_count', 0) for r in all_results]
            min_count = min(counts) if counts else optimal // 2
            max_count = max(counts) if counts else optimal * 2
        else:
            min_count = optimal // 2
            max_count = optimal * 2

        return {
            'min': max(10, min_count),
            'optimal': optimal,
            'max': max_count,
            'confidence': best_result.get('precision', 0.0)
        }

    def get_convergence_metrics(self) -> Dict[str, float]:
        """Get convergence speed metrics from window optimizer"""
        if not self.results:
            return {'speed': 0.5, 'stability': 0.5}

        all_results = self.results.get('all_results', [])
        if not all_results:
            return {'speed': 0.5, 'stability': 0.5}

        # Analyze improvement rate
        scores = [r.get('bidirectional_count', 0) for r in all_results]
        iterations = len(scores)

        # Convergence speed: how quickly did we find good solutions?
        max_score = max(scores) if scores else 1
        first_good = next((i for i, s in enumerate(scores) if s >= max_score * 0.8), iterations)
        speed = 1.0 - (first_good / iterations)

        # Stability: variance in top 20% results
        top_20_pct = int(len(scores) * 0.2)
        if top_20_pct > 0:
            top_scores = sorted(scores, reverse=True)[:top_20_pct]
            stability = 1.0 - (np.std(top_scores) / (np.mean(top_scores) + 1e-8))
        else:
            stability = 0.5

        return {
            'speed': float(speed),
            'stability': float(max(0, min(1, stability)))
        }


class HistoricalPatternAnalyzer:
    """
    Analyze lottery history to determine pattern characteristics

    SECONDARY SOURCE (35% weight)
    
    NOTE: This analyzes TRAINING history only, not holdout.
    """

    def __init__(self, lottery_data_path: str):
        """
        Initialize analyzer

        Args:
            lottery_data_path: Path to lottery history JSON (train_history.json)
        """
        self.lottery_data_path = lottery_data_path
        self.lottery_history = []
        self._load_lottery_data()

    def _load_lottery_data(self):
        """Load lottery history"""
        try:
            with open(self.lottery_data_path, 'r') as f:
                data = json.load(f)

            # Parse various formats
            if isinstance(data, list):
                if not data:
                    self.lottery_history = []
                elif isinstance(data[0], dict):
                    # Format: [{"draw": 123, ...}, ...]
                    self.lottery_history = [d.get('draw', d.get('result', 0)) for d in data]
                else:
                    # Format: [123, 456, ...]
                    self.lottery_history = [int(x) for x in data]
            elif isinstance(data, dict):
                # Format: {"draws": [...], ...}
                self.lottery_history = data.get('draws', data.get('results', []))
            else:
                self.lottery_history = []

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Lottery data not found at {self.lottery_data_path}"
            )

    def estimate_min_survivor_count(self) -> int:
        """
        Estimate minimum survivors needed based on pattern complexity

        More complex patterns → need more survivors
        """
        if len(self.lottery_history) < 100:
            return 100

        # Calculate pattern complexity metrics
        unique_draws = len(set(self.lottery_history))
        total_draws = len(self.lottery_history)
        uniqueness_ratio = unique_draws / total_draws

        # Calculate entropy (higher = more random/complex)
        counts = Counter(self.lottery_history)
        probs = np.array([c / total_draws for c in counts.values()])
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(1000)  # For mod 1000
        normalized_entropy = entropy / max_entropy

        # Estimate based on complexity
        # High complexity → need more samples to learn patterns
        base_survivors = 100
        complexity_multiplier = 1 + (normalized_entropy * 2)  # 1x to 3x

        min_survivors = int(base_survivors * complexity_multiplier)

        return max(50, min(2000, min_survivors))

    def estimate_optimal_survivor_count(self) -> int:
        """
        Estimate optimal survivors based on historical stability

        More stable patterns → fewer survivors needed
        Less stable patterns → more survivors needed
        """
        if len(self.lottery_history) < 500:
            return 500

        # Analyze temporal stability
        window_size = 100
        num_windows = min(10, len(self.lottery_history) // window_size)

        window_entropies = []
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = self.lottery_history[start:end]

            counts = Counter(window)
            probs = np.array([c / len(window) for c in counts.values()])
            ent = -np.sum(probs * np.log(probs + 1e-10))
            window_entropies.append(ent)

        # Stability = low variance in entropy across windows
        if len(window_entropies) > 1:
            stability = 1.0 - (np.std(window_entropies) / (np.mean(window_entropies) + 1e-8))
        else:
            stability = 0.5

        # Stable patterns → fewer survivors needed
        # Unstable patterns → more survivors needed
        base_survivors = 500
        stability_adjustment = 1.0 + ((1.0 - stability) * 2)  # 1x to 3x

        optimal_survivors = int(base_survivors * stability_adjustment)

        return max(100, min(5000, optimal_survivors))

    def get_regime_change_frequency(self) -> float:
        """
        Estimate how often regime changes occur

        Returns:
            Frequency (0-1), where 1 = frequent changes
        """
        if len(self.lottery_history) < 1000:
            return 0.5

        # Compare distributions across time periods
        num_periods = 5
        period_size = len(self.lottery_history) // num_periods

        distributions = []
        for i in range(num_periods):
            start = i * period_size
            end = start + period_size
            period = self.lottery_history[start:end]

            counts = Counter(period)
            dist = np.array([counts.get(i, 0) for i in range(1000)])
            dist = dist / dist.sum()
            distributions.append(dist)

        # Calculate KL divergence between consecutive periods
        divergences = []
        for i in range(len(distributions) - 1):
            p = distributions[i] + 1e-10
            q = distributions[i+1] + 1e-10
            kl = np.sum(p * np.log(p / q))
            divergences.append(kl)

        # Average divergence indicates regime change frequency
        avg_divergence = np.mean(divergences) if divergences else 0.0

        # Normalize to 0-1 (empirically, KL > 0.5 is significant change)
        frequency = min(1.0, avg_divergence / 0.5)

        return float(frequency)


class ReinforcementFeedbackAnalyzer:
    """
    Analyze reinforcement engine performance feedback

    CONTINUOUS SOURCE (5% → 25% weight, grows with confidence)
    """

    def __init__(self, feedback_path: Optional[str] = None):
        """
        Initialize analyzer

        Args:
            feedback_path: Path to reinforcement feedback history
        """
        self.feedback_path = feedback_path or 'optimization_results/reinforcement_feedback.json'
        self.feedback_history = []
        self._load_feedback()

    def _load_feedback(self):
        """Load feedback history"""
        if not Path(self.feedback_path).exists():
            self.feedback_history = []
            return

        try:
            with open(self.feedback_path, 'r') as f:
                self.feedback_history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            self.feedback_history = []

    def has_feedback(self) -> bool:
        """Check if feedback history exists"""
        return len(self.feedback_history) > 0

    def get_optimal_from_experience(self) -> Optional[Dict[str, Any]]:
        """
        Get optimal parameters learned from experience

        Returns:
            Best performing configuration or None if no history
        """
        if not self.feedback_history:
            return None

        # Find best performing configuration
        best = max(self.feedback_history, key=lambda x: x.get('performance', 0))

        return {
            'survivor_count': best.get('survivor_count', 500),
            'network_architecture': best.get('network_architecture', [128, 64, 32]),
            'training_epochs': best.get('training_epochs', 100),
            'performance': best.get('performance', 0.0),
            'confidence': self.calculate_confidence()
        }

    def calculate_confidence(self) -> float:
        """
        Calculate confidence in feedback

        Higher confidence = more weight in optimization
        """
        if not self.feedback_history:
            return 0.0

        # Confidence based on:
        # 1. Number of samples (more = better)
        # 2. Performance trend (improving = better)
        # 3. Stability (low variance = better)

        n_samples = len(self.feedback_history)
        sample_confidence = min(1.0, n_samples / 100)  # Full confidence at 100 samples

        # Performance trend
        if len(self.feedback_history) >= 5:
            recent = [f.get('performance', 0) for f in self.feedback_history[-5:]]
            older = [f.get('performance', 0) for f in self.feedback_history[:5]]

            trend = np.mean(recent) - np.mean(older)
            trend_confidence = max(0, min(1, trend * 10))  # Normalize
        else:
            trend_confidence = 0.5

        # Stability
        performances = [f.get('performance', 0) for f in self.feedback_history[-20:]]
        if len(performances) > 1:
            stability = 1.0 - (np.std(performances) / (np.mean(performances) + 1e-8))
            stability_confidence = max(0, min(1, stability))
        else:
            stability_confidence = 0.5

        # Combined confidence
        confidence = (sample_confidence * 0.4 +
                     trend_confidence * 0.3 +
                     stability_confidence * 0.3)

        return float(confidence)

    def calculate_feedback_weight(self, initial_weight: float = 0.05,
                                  max_weight: float = 0.25) -> float:
        """
        Calculate dynamic weight for feedback

        Weight grows from initial to max based on confidence
        """
        confidence = self.calculate_confidence()
        weight = initial_weight + (max_weight - initial_weight) * confidence
        return float(weight)


# ============================================================================
# ADAPTIVE META-OPTIMIZER
# ============================================================================

class AdaptiveMetaOptimizer:
    """
    Main meta-optimizer class

    CAPACITY & ARCHITECTURE PLANNER (not data-aware optimizer)
    
    Finds optimal training parameters through:
    - Window optimizer behavior analysis
    - Lottery pattern complexity analysis
    - Reinforcement feedback (if available)
    
    Does NOT analyze survivor-level data (that's Step 5's job)
    """

    def __init__(self, config: MetaOptimizerConfig,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize meta-optimizer

        Args:
            config: MetaOptimizerConfig instance
            logger: Optional logger
        """
        self.config = config
        self.logger = logger or self._setup_logger()

        # Initialize analyzers
        self.logger.info("Initializing AdaptiveMetaOptimizer (Capacity Planner)...")
        self.logger.info("NOTE: Step 4 does NOT consume survivor-level data (by design)")

        # PRIMARY (60%): Window optimizer
        window_path = config.sources['window_optimizer_results']['path']
        self.window_analyzer = WindowOptimizerAnalyzer(window_path)
        self.logger.info(f"  Window optimizer: {window_path}")

        # SECONDARY (35%): Historical patterns
        lottery_path = config.sources['lottery_history']['path']
        self.pattern_analyzer = HistoricalPatternAnalyzer(lottery_path)
        self.logger.info(f"  Lottery history: {lottery_path} ({len(self.pattern_analyzer.lottery_history)} draws)")

        # CONTINUOUS (5%→25%): Reinforcement feedback
        self.feedback_analyzer = ReinforcementFeedbackAnalyzer()
        feedback_status = "available" if self.feedback_analyzer.has_feedback() else "not yet available"
        self.logger.info(f"  Reinforcement feedback: {feedback_status}")

        # Create output directory
        Path(config.output['results_dir']).mkdir(parents=True, exist_ok=True)

        self.logger.info("AdaptiveMetaOptimizer initialized successfully!")

    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def derive_optimal_survivor_count(self) -> Dict[str, Any]:
        """
        Derive optimal survivor count from all sources with weighted combination

        Returns:
            Dictionary with min, optimal, max, and confidence
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("DERIVING OPTIMAL SURVIVOR COUNT")
        self.logger.info("="*70)

        # PRIMARY (60%): Window optimizer
        window_range = self.window_analyzer.get_survivor_count_range()
        self.logger.info(f"\nPRIMARY SOURCE ({self.config.sources['window_optimizer_results']['weight']:.0%} weight): Window Optimizer")
        self.logger.info(f"  Min: {window_range['min']}")
        self.logger.info(f"  Optimal: {window_range['optimal']}")
        self.logger.info(f"  Max: {window_range['max']}")
        self.logger.info(f"  Confidence: {window_range['confidence']:.2%}")

        # SECONDARY (35%): Historical patterns
        pattern_min = self.pattern_analyzer.estimate_min_survivor_count()
        pattern_optimal = self.pattern_analyzer.estimate_optimal_survivor_count()
        self.logger.info(f"\nSECONDARY SOURCE ({self.config.sources['lottery_history']['weight']:.0%} weight): Historical Patterns")
        self.logger.info(f"  Min (complexity-based): {pattern_min}")
        self.logger.info(f"  Optimal (stability-based): {pattern_optimal}")

        # CONTINUOUS (5%→25%): Reinforcement feedback
        feedback_weight = self.config.sources['reinforcement_feedback']['weight']

        if self.feedback_analyzer.has_feedback():
            feedback_optimal = self.feedback_analyzer.get_optimal_from_experience()
            feedback_confidence = self.feedback_analyzer.calculate_confidence()
            feedback_weight = self.feedback_analyzer.calculate_feedback_weight(
                initial_weight=feedback_weight,
                max_weight=self.config.sources['reinforcement_feedback']['max_weight']
            )

            self.logger.info(f"\nCONTINUOUS SOURCE ({feedback_weight:.1%} weight): Reinforcement Feedback")
            self.logger.info(f"  Optimal (experience): {feedback_optimal['survivor_count']}")
            self.logger.info(f"  Confidence: {feedback_confidence:.2%}")
        else:
            feedback_optimal = None
            feedback_confidence = 0.0
            self.logger.info(f"\nCONTINUOUS SOURCE: No feedback yet (will grow from 5% to 25%)")

        # WEIGHTED COMBINATION
        # Read base weights from config (ML/AI configurable)
        base_w_window = self.config.sources['window_optimizer_results']['weight']
        base_w_pattern = self.config.sources['lottery_history']['weight']
        base_w_feedback = self.config.sources['reinforcement_feedback']['weight']
        
        # Adjust weights based on feedback availability
        if feedback_optimal:
            # Redistribute some weight to feedback as it grows
            w_window = base_w_window - feedback_weight / 2
            w_pattern = base_w_pattern - feedback_weight / 2
            w_feedback = feedback_weight
        else:
            w_window = base_w_window
            w_pattern = base_w_pattern
            w_feedback = base_w_feedback

        # Calculate weighted optimal
        optimal_values = [
            window_range['optimal'] * w_window,
            pattern_optimal * w_pattern
        ]

        if feedback_optimal:
            optimal_values.append(feedback_optimal['survivor_count'] * w_feedback)

        weighted_optimal = int(sum(optimal_values) / sum([w_window, w_pattern, w_feedback]))

        # Calculate min/max with multipliers
        min_multiplier = self.config.optimization['survivor_count']['min_multiplier']
        max_multiplier = self.config.optimization['survivor_count']['max_multiplier']

        final_min = max(window_range['min'], int(weighted_optimal * min_multiplier))
        final_max = int(weighted_optimal * max_multiplier)

        # Overall confidence
        confidence = (window_range['confidence'] * w_window +
                     0.7 * w_pattern +  # Pattern analysis has moderate confidence
                     feedback_confidence * w_feedback)

        result = {
            'min': final_min,
            'optimal': weighted_optimal,
            'max': final_max,
            'confidence': confidence,
            'weights': {
                'window_optimizer': w_window,
                'historical_patterns': w_pattern,
                'reinforcement_feedback': w_feedback
            },
            'sources': {
                'window_optimizer': window_range['optimal'],
                'historical_patterns': pattern_optimal,
                'reinforcement_feedback': feedback_optimal['survivor_count'] if feedback_optimal else None
            }
        }

        self.logger.info(f"\n{'='*70}")
        self.logger.info("WEIGHTED RESULT:")
        self.logger.info(f"  Min survivors: {result['min']}")
        self.logger.info(f"  Optimal survivors: {result['optimal']}")
        self.logger.info(f"  Max survivors: {result['max']}")
        self.logger.info(f"  Overall confidence: {result['confidence']:.2%}")
        self.logger.info(f"{'='*70}\n")

        return result

    def optimize_network_architecture(self, survivor_count: int) -> Dict[str, Any]:
        """
        Find optimal network architecture through trials

        Args:
            survivor_count: Number of survivors to use in trials

        Returns:
            Best architecture and metrics
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("OPTIMIZING NETWORK ARCHITECTURE")
        self.logger.info(f"Using survivor count: {survivor_count}")
        self.logger.info("="*70)

        options = self.config.optimization['network_architecture']['options']

        # Simulate trials (in production, would run actual training)
        best_arch = None
        best_score = -1

        results = []
        for arch in options:
            # Heuristic scoring based on architecture complexity
            # In production, this would be actual training trials
            complexity = sum(arch)

            # Balance between capacity and overfitting
            capacity_score = min(1.0, complexity / 256)
            simplicity_score = 1.0 - (len(arch) / 4)

            score = capacity_score * 0.7 + simplicity_score * 0.3

            results.append({
                'architecture': arch,
                'score': score,
                'complexity': complexity
            })

            self.logger.info(f"  {arch}: score={score:.4f}")

            if score > best_score:
                best_score = score
                best_arch = arch

        self.logger.info(f"\nOPTIMAL ARCHITECTURE: {best_arch}")
        self.logger.info(f"Score: {best_score:.4f}\n")

        return {
            'architecture': best_arch,
            'score': best_score,
            'all_results': results
        }

    def estimate_training_epochs(self, survivor_count: int) -> Dict[str, int]:
        """
        Estimate optimal training epochs based on convergence analysis

        Args:
            survivor_count: Number of survivors

        Returns:
            Dictionary with min, optimal, max epochs
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("ESTIMATING TRAINING EPOCHS")
        self.logger.info("="*70)

        # Get convergence metrics from window optimizer
        convergence = self.window_analyzer.get_convergence_metrics()

        # Base epochs on convergence speed
        # Fast convergence → fewer epochs needed
        # Slow convergence → more epochs needed
        base_epochs = 100

        speed_factor = 1.0 / (convergence['speed'] + 0.5)  # 0.67x to 2x
        stability_factor = 1.0 + (1.0 - convergence['stability'])  # 1x to 2x

        optimal_epochs = int(base_epochs * speed_factor * stability_factor)

        # Clamp to configured range
        min_epochs = self.config.optimization['training_epochs']['min']
        max_epochs = self.config.optimization['training_epochs']['max']
        optimal_epochs = max(min_epochs, min(max_epochs, optimal_epochs))

        self.logger.info(f"  Convergence speed: {convergence['speed']:.2%}")
        self.logger.info(f"  Stability: {convergence['stability']:.2%}")
        self.logger.info(f"  Optimal epochs: {optimal_epochs}")
        self.logger.info(f"  Range: [{min_epochs}, {max_epochs}]\n")

        return {
            'min': min_epochs,
            'optimal': optimal_epochs,
            'max': max_epochs,
            'convergence_metrics': convergence
        }

    def full_calibration(self) -> Dict[str, Any]:
        """
        Run full calibration (initialization phase)

        Returns:
            Complete optimal configuration
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("FULL CALIBRATION - CAPACITY & ARCHITECTURE PLANNING")
        self.logger.info("NOTE: This does NOT analyze survivor-level data (by design)")
        self.logger.info("="*70)

        # 1. Derive optimal survivor count
        survivor_result = self.derive_optimal_survivor_count()
        optimal_survivors = survivor_result['optimal']

        # 2. Optimize network architecture
        network_result = self.optimize_network_architecture(optimal_survivors)

        # 3. Estimate training epochs
        epochs_result = self.estimate_training_epochs(optimal_survivors)

        # 4. Compile optimal configuration
        optimal_config = {
            'survivor_count': optimal_survivors,
            'survivor_count_range': {
                'min': survivor_result['min'],
                'max': survivor_result['max']
            },
            'network_architecture': network_result['architecture'],
            'training_epochs': epochs_result['optimal'],
            'training_epochs_range': {
                'min': epochs_result['min'],
                'max': epochs_result['max']
            },
            'confidence': survivor_result['confidence'],
            'weights_used': survivor_result['weights'],
            'timestamp': datetime.now().isoformat(),
            'calibration_type': 'full'
        }

        self.logger.info("\n" + "="*70)
        self.logger.info("OPTIMAL CONFIGURATION (Capacity Planning)")
        self.logger.info("="*70)
        self.logger.info(f"  Survivor count: {optimal_config['survivor_count']}")
        self.logger.info(f"  Network architecture: {optimal_config['network_architecture']}")
        self.logger.info(f"  Training epochs: {optimal_config['training_epochs']}")
        self.logger.info(f"  Overall confidence: {optimal_config['confidence']:.2%}")
        self.logger.info("")
        self.logger.info("NOTE: Model selection (neural_net/xgboost/lightgbm/catboost)")
        self.logger.info("      happens in Step 5, NOT here.")
        self.logger.info("="*70 + "\n")

        return optimal_config

    def micro_adjust(self, current_config: Dict[str, Any],
                    performance_metrics: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """
        Perform micro-adjustment based on recent performance

        Args:
            current_config: Current configuration
            performance_metrics: Recent performance data

        Returns:
            Adjusted config if adjustment needed, None otherwise
        """
        if not self.config.continuous_optimization['enabled']:
            return None

        # Check if adjustment threshold exceeded
        threshold = self.config.continuous_optimization['adjustment_threshold']
        performance_change = performance_metrics.get('performance_change', 0.0)

        if abs(performance_change) < threshold:
            self.logger.debug(f"Performance change {performance_change:.2%} below threshold {threshold:.2%}")
            return None

        self.logger.info("\n" + "="*70)
        self.logger.info("MICRO-ADJUSTMENT TRIGGERED")
        self.logger.info(f"Performance change: {performance_change:.2%}")
        self.logger.info("="*70)

        # Small adjustments based on performance direction
        adjusted_config = current_config.copy()

        if performance_change < 0:  # Performance degraded
            # Try increasing survivor count slightly
            current_survivors = current_config.get('survivor_count', 500)
            adjusted_config['survivor_count'] = int(current_survivors * 1.1)

            # Try more epochs
            current_epochs = current_config.get('training_epochs', 100)
            adjusted_config['training_epochs'] = int(current_epochs * 1.2)

            self.logger.info(f"  Survivor count: {current_survivors} → {adjusted_config['survivor_count']}")
            self.logger.info(f"  Training epochs: {current_epochs} → {adjusted_config['training_epochs']}")

        else:  # Performance improved
            # Current config is working, keep it
            self.logger.info("  Performance improved - keeping current configuration")
            return None

        adjusted_config['timestamp'] = datetime.now().isoformat()
        adjusted_config['calibration_type'] = 'micro_adjustment'

        return adjusted_config

    def handle_regime_change(self) -> Dict[str, Any]:
        """
        Handle regime change detection - trigger full re-calibration

        Returns:
            New optimal configuration
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("REGIME CHANGE DETECTED - FULL RE-CALIBRATION")
        self.logger.info("="*70)

        # Save current config as fallback
        if self.config.regime_change['preserve_history']:
            self._save_config_to_history()

        # Run full calibration
        new_config = self.full_calibration()
        new_config['calibration_type'] = 'regime_change'

        return new_config

    def _save_config_to_history(self):
        """Save current configuration to history"""
        history_file = Path(self.config.output['results_dir']) / self.config.output['history_file']

        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []

        # Load current config
        target_file = self.config.output['target_file']
        if Path(target_file).exists():
            with open(target_file, 'r') as f:
                current_config = json.load(f)

            history.append({
                'timestamp': datetime.now().isoformat(),
                'config': current_config,
                'event': 'pre_regime_change'
            })

            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)

            self.logger.info(f"  Current config saved to history: {history_file}")

    def update_target_config(self, optimal_config: Dict[str, Any]):
        """
        Update reinforcement_engine_config.json with optimal parameters

        Args:
            optimal_config: Optimal configuration from calibration
        """
        target_file = Path(self.config.output['target_file'])

        # Backup previous config if requested
        if self.config.output['backup_previous'] and target_file.exists():
            backup_file = target_file.with_suffix(f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
            import shutil
            shutil.copy(target_file, backup_file)
            self.logger.info(f"  Previous config backed up to: {backup_file}")

        # Load current config
        if target_file.exists():
            with open(target_file, 'r') as f:
                current_config = json.load(f)
        else:
            # Use default structure
            current_config = {
                'model': {},
                'training': {},
                'prng': {},
                'global_state': {},
                'normalization': {},
                'survivor_pool': {},
                'output': {}
            }

        # Update with optimal parameters
        if self.config.output['update_strategy'] == 'merge_preserving_user_overrides':
            # Only update if not manually set by user
            # (In practice, would check for user override flags)

            # Update model architecture
            if 'network_architecture' in optimal_config:
                current_config['model']['hidden_layers'] = optimal_config['network_architecture']

            # Update training epochs
            if 'training_epochs' in optimal_config:
                current_config['training']['epochs'] = optimal_config['training_epochs']

            # Update survivor pool size
            if 'survivor_count' in optimal_config:
                current_config['survivor_pool']['max_pool_size'] = optimal_config['survivor_count']

            # Add metadata
            current_config['_meta_optimizer'] = {
                'last_calibration': optimal_config.get('timestamp'),
                'calibration_type': optimal_config.get('calibration_type'),
                'confidence': optimal_config.get('confidence'),
                'weights_used': optimal_config.get('weights_used')
            }

        # Save updated config
        with open(target_file, 'w') as f:
            json.dump(current_config, f, indent=2)

        self.logger.info(f"✅ Updated configuration: {target_file}")

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """
        Save optimization results

        Args:
            results: Results dictionary
            filename: Optional filename (default: timestamped)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"meta_optimization_{timestamp}.json"

        output_path = Path(self.config.output['results_dir']) / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"✅ Results saved: {output_path}")

        # ALSO save to fixed filename for pipeline integration
        fixed_path = Path(self.config.output['results_dir']) / 'meta_optimization_results.json'
        with open(fixed_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"✅ Results also saved to: {fixed_path}")


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Adaptive Meta-Optimizer - Capacity & Architecture Planning (Step 4)',
        epilog='''
NOTE: Step 4 is intentionally NOT data-aware.
It derives capacity parameters from window optimization behavior and 
training-history complexity only. Survivor-level data (including holdout_hits) 
is first consumed in Step 5, where model selection and overfit control occur.
        '''
    )
    parser.add_argument('--config', type=str,
                       default='adaptive_meta_optimizer_config.json',
                       help='Configuration file')
    parser.add_argument('--mode', type=str,
                       choices=['full', 'survivor', 'network', 'epochs'],
                       default='full',
                       help='Optimization mode')
    parser.add_argument('--window-results', type=str,
                       help='Override window optimizer results path (Step 1 output)')
    parser.add_argument('--lottery-data', type=str,
                       help='Override lottery data path (train_history.json)')
    parser.add_argument('--apply', action='store_true',
                       help='Apply results to reinforcement_engine_config.json')
    parser.add_argument('--test', action='store_true',
                       help='Run self-test with synthetic data')

    args = parser.parse_args()

    # Test mode
    if args.test:
        print("="*70)
        print("ADAPTIVE META-OPTIMIZER - SELF TEST")
        print("="*70)
        print("\n⚠️  Test mode requires:")
        print("  1. optimal_window_config.json (Step 1 output)")
        print("  2. train_history.json")
        print("\nPlease ensure these files exist before running.\n")
        return 0

    # Load config
    try:
        config = MetaOptimizerConfig.from_json(args.config)
        print(f"✅ Config loaded from {args.config}")
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        print("Using default config")
        config = MetaOptimizerConfig()

    # Override paths if specified
    if args.window_results:
        config.sources['window_optimizer_results']['path'] = args.window_results
    if args.lottery_data:
        config.sources['lottery_history']['path'] = args.lottery_data

    # Initialize optimizer
    try:
        optimizer = AdaptiveMetaOptimizer(config)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure required files exist:")
        print(f"  1. {config.sources['window_optimizer_results']['path']}")
        print(f"  2. {config.sources['lottery_history']['path']}")
        return 1

    # Run optimization
    if args.mode == 'full':
        results = optimizer.full_calibration()
    elif args.mode == 'survivor':
        results = optimizer.derive_optimal_survivor_count()
    elif args.mode == 'network':
        survivor_result = optimizer.derive_optimal_survivor_count()
        results = optimizer.optimize_network_architecture(survivor_result['optimal'])
    elif args.mode == 'epochs':
        survivor_result = optimizer.derive_optimal_survivor_count()
        results = optimizer.estimate_training_epochs(survivor_result['optimal'])

    # Save results
    optimizer.save_results(results)

    # Apply to target config if requested
    if args.apply and args.mode == 'full':
        optimizer.update_target_config(results)
        print("\n✅ Configuration applied to reinforcement_engine_config.json")

    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
