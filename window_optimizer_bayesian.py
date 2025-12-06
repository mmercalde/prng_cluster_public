#!/usr/bin/env python3
"""
Real Bayesian Optimization for Window Parameter Search
Uses Optuna (Tree-Parzen Estimator) and scikit-learn (Gaussian Processes)
"""

import json
import numpy as np
from typing import Callable, Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import random

# Try importing Optuna (preferred)
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️  Optuna not available - install with: pip install optuna")

# Try importing sklearn for Gaussian Processes (fallback)
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, RBF
    SKLEARN_GP_AVAILABLE = True
except ImportError:
    SKLEARN_GP_AVAILABLE = False
    print("⚠️  scikit-learn GP not available")


# ============================================================================
# DATA STRUCTURES (copied from window_optimizer.py)
# ============================================================================

@dataclass
class WindowConfig:
    """Complete window and skip configuration"""
    window_size: int
    offset: int
    sessions: List[str]
    skip_min: int
    skip_max: int
    forward_threshold: float = 0.72
    reverse_threshold: float = 0.81
    
    def __hash__(self):
        return hash((self.window_size, self.offset, tuple(self.sessions), 
                    self.skip_min, self.skip_max, self.forward_threshold, self.reverse_threshold))
    
    def description(self) -> str:
        sess = '+'.join(self.sessions)
        return f"W{self.window_size}_O{self.offset}_{sess}_S{self.skip_min}-{self.skip_max}_FT{self.forward_threshold}_RT{self.reverse_threshold}"
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_vector(self) -> np.ndarray:
        """Convert to numerical vector for optimization"""
        session_encoding = {
            ('midday', 'evening'): 0,
            ('midday',): 1,
            ('evening',): 2
        }
        sess_tuple = tuple(sorted(self.sessions))
        return np.array([
            self.window_size,
            self.offset,
            session_encoding.get(sess_tuple, 0),
            self.skip_min,
            self.skip_max
        ], dtype=float)


@dataclass
class SearchBounds:
    """Search space boundaries"""
    min_window_size: int = 256
    max_window_size: int = 2048
    min_offset: int = 0
    max_offset: int = 500
    min_skip_min: int = 0
    max_skip_min: int = 3
    min_skip_max: int = 10
    max_skip_max: int = 200
    session_options: List[List[str]] = None
    
    def __post_init__(self):
        if self.session_options is None:
            self.session_options = [
                ['midday', 'evening'],
                ['midday'],
                ['evening']
            ]


@dataclass
class OptimizationResult:
    """Result from testing a window configuration"""
    config: WindowConfig
    forward_count: int
    reverse_count: int
    bidirectional_count: int
    iteration: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'config': self.config.to_dict(),
            'forward_count': self.forward_count,
            'reverse_count': self.reverse_count,
            'bidirectional_count': self.bidirectional_count,
            'iteration': self.iteration
        }


class ResultScorer:
    """Score optimization results (higher is better)"""
    
    def __init__(self, strategy='bidirectional'):
        self.strategy = strategy
    
    def score(self, result: OptimizationResult) -> float:
        """Score a result (higher is better)"""
        if self.strategy == 'bidirectional':
            # Primary: bidirectional survivors
            return result.bidirectional_count
        elif self.strategy == 'balanced':
            # Balance between forward, reverse, and bidirectional
            return (result.forward_count * 0.3 + 
                   result.reverse_count * 0.3 + 
                   result.bidirectional_count * 0.4)
        elif self.strategy == 'conservative':
            # Prefer configurations with consistent forward/reverse
            if result.forward_count == 0 or result.reverse_count == 0:
                return 0
            ratio = min(result.forward_count, result.reverse_count) / max(result.forward_count, result.reverse_count)
            return result.bidirectional_count * ratio
        else:
            return result.bidirectional_count


# ============================================================================
# OPTUNA-BASED BAYESIAN OPTIMIZATION (PREFERRED)
# ============================================================================

class OptunaBayesianSearch:
    """Bayesian optimization using Optuna's TPE sampler"""
    
    def __init__(self, n_startup_trials=5, seed=None):
        """
        Args:
            n_startup_trials: Number of random trials before using TPE
            seed: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna not available. Install with: pip install optuna")
        
        self.n_startup_trials = n_startup_trials
        self.seed = seed
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def search(self, 
               objective_function: Callable,
               bounds: SearchBounds,
               max_iterations: int,
               scorer: ResultScorer) -> Dict:
        """
        Run Bayesian optimization using Optuna
        
        Args:
            objective_function: Function that takes WindowConfig and returns OptimizationResult
            bounds: Search space boundaries
            max_iterations: Number of iterations
            scorer: Function to score results
            
        Returns:
            Dictionary with best config, all results, etc.
        """
        print(f"\n{'='*80}")
        print(f"🎯 BAYESIAN OPTIMIZATION (Optuna TPE)")
        print(f"   Startup trials: {self.n_startup_trials}")
        print(f"   Total trials: {max_iterations}")
        print(f"{'='*80}\n")
        
        # Storage for all results
        all_results = []
        best_result = None
        best_score = float('-inf')
        
        def optuna_objective(trial):
            """Optuna objective function"""
            # Sample parameters from search space
            window_size = trial.suggest_int('window_size', 
                                           bounds.min_window_size, 
                                           bounds.max_window_size)
            offset = trial.suggest_int('offset', 
                                      bounds.min_offset, 
                                      bounds.max_offset)
            session_idx = trial.suggest_int('session_idx', 
                                           0, 
                                           len(bounds.session_options) - 1)
            skip_min = trial.suggest_int('skip_min', 
                                        bounds.min_skip_min, 
                                        bounds.max_skip_min)
            skip_max = trial.suggest_int('skip_max', 
                                        max(skip_min, bounds.min_skip_max),
                                        bounds.max_skip_max)

            # Suggest thresholds (Optuna optimizes these!)
            forward_threshold = trial.suggest_float('forward_threshold',
                                                   bounds.min_forward_threshold,
                                                   bounds.max_forward_threshold)
            reverse_threshold = trial.suggest_float('reverse_threshold',
                                                   bounds.min_reverse_threshold,
                                                   bounds.max_reverse_threshold)
            
            # Create configuration
            config = WindowConfig(
                window_size=window_size,
                offset=offset,
                sessions=bounds.session_options[session_idx],
                skip_min=skip_min,
                skip_max=skip_max,
                forward_threshold=round(forward_threshold, 2),
                reverse_threshold=round(reverse_threshold, 2)
            )
            
            # Evaluate configuration
            result = objective_function(config)
            result.iteration = trial.number
            
            # Store result
            all_results.append(result)
            score = scorer.score(result)
            
            # Track best
            nonlocal best_result, best_score
            if score > best_score:
                best_score = score
                best_result = result
                print(f"✨ NEW BEST [Trial {trial.number + 1}]: {config.description()}")
                print(f"   Score: {score:.2f} (Bidirectional: {result.bidirectional_count})")
                print(f"   Forward: {result.forward_count}, Reverse: {result.reverse_count}\n")
            else:
                print(f"   Trial {trial.number + 1}: {config.description()} → Score: {score:.2f}")
            
            return score
        
        # Create study with TPE sampler
        sampler = TPESampler(
            n_startup_trials=self.n_startup_trials,
            seed=self.seed
        )
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )
        
        # Run optimization
        study.optimize(optuna_objective, n_trials=max_iterations)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"🏆 OPTIMIZATION COMPLETE")
        print(f"   Best score: {best_score:.2f}")
        print(f"   Best config: {best_result.config.description()}")
        print(f"   Bidirectional survivors: {best_result.bidirectional_count}")
        print(f"   📊 Optuna-optimized thresholds:")
        print(f"      Forward threshold: {best_result.config.forward_threshold}")
        print(f"      Reverse threshold: {best_result.config.reverse_threshold}")
        print(f"{'='*80}\n")
        
        return {
            'strategy': 'optuna_bayesian',
            'best_config': best_result.config.to_dict(),
            'best_result': best_result.to_dict(),
            'best_score': best_score,
            'all_results': [r.to_dict() for r in all_results],
            'iterations': len(all_results),
            'optuna_study': {
                'best_trial': study.best_trial.number,
                'best_value': study.best_value,
                'best_params': study.best_params
            }
        }


# ============================================================================
# SKLEARN GAUSSIAN PROCESS OPTIMIZATION (FALLBACK)
# ============================================================================

class GaussianProcessBayesianSearch:
    """Bayesian optimization using sklearn Gaussian Processes"""
    
    def __init__(self, n_initial_points=5, acquisition='ucb', seed=None):
        """
        Args:
            n_initial_points: Number of random points before using GP
            acquisition: 'ucb', 'ei', or 'pi'
            seed: Random seed
        """
        if not SKLEARN_GP_AVAILABLE:
            raise ImportError("scikit-learn not available")
        
        self.n_initial_points = n_initial_points
        self.acquisition = acquisition
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def _config_to_vector(self, config: WindowConfig, bounds: SearchBounds) -> np.ndarray:
        """Convert config to normalized vector [0, 1]"""
        session_idx = bounds.session_options.index(config.sessions)
        return np.array([
            (config.window_size - bounds.min_window_size) / (bounds.max_window_size - bounds.min_window_size),
            (config.offset - bounds.min_offset) / (bounds.max_offset - bounds.min_offset),
            session_idx / max(1, len(bounds.session_options) - 1),
            (config.skip_min - bounds.min_skip_min) / max(1, bounds.max_skip_min - bounds.min_skip_min),
            (config.skip_max - bounds.min_skip_max) / max(1, bounds.max_skip_max - bounds.min_skip_max)
        ])
    
    def _vector_to_config(self, vec: np.ndarray, bounds: SearchBounds) -> WindowConfig:
        """Convert normalized vector to config"""
        window_size = int(np.clip(
            vec[0] * (bounds.max_window_size - bounds.min_window_size) + bounds.min_window_size,
            bounds.min_window_size,
            bounds.max_window_size
        ))
        offset = int(np.clip(
            vec[1] * (bounds.max_offset - bounds.min_offset) + bounds.min_offset,
            bounds.min_offset,
            bounds.max_offset
        ))
        session_idx = int(np.clip(
            vec[2] * len(bounds.session_options),
            0,
            len(bounds.session_options) - 1
        ))
        skip_min = int(np.clip(
            vec[3] * (bounds.max_skip_min - bounds.min_skip_min) + bounds.min_skip_min,
            bounds.min_skip_min,
            bounds.max_skip_min
        ))
        skip_max = int(np.clip(
            vec[4] * (bounds.max_skip_max - bounds.min_skip_max) + bounds.min_skip_max,
            bounds.min_skip_max,
            bounds.max_skip_max
        ))
        
        # Ensure skip_max >= skip_min
        skip_max = max(skip_max, skip_min)
        
        return WindowConfig(
            window_size=window_size,
            offset=offset,
            sessions=bounds.session_options[session_idx],
            skip_min=skip_min,
            skip_max=skip_max,
            forward_threshold=bounds.default_forward_threshold,
            reverse_threshold=bounds.default_reverse_threshold
        )
    
    def _acquisition_function(self, gp, X_train, y_train, X_new, kappa=2.0):
        """Upper Confidence Bound acquisition function"""
        mu, sigma = gp.predict(X_new.reshape(1, -1), return_std=True)
        if self.acquisition == 'ucb':
            return mu + kappa * sigma
        elif self.acquisition == 'ei':
            # Expected Improvement
            best = np.max(y_train)
            z = (mu - best) / (sigma + 1e-9)
            from scipy.stats import norm
            return (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)
        else:  # pi - probability of improvement
            from scipy.stats import norm
            best = np.max(y_train)
            z = (mu - best) / (sigma + 1e-9)
            return norm.cdf(z)
    
    def search(self,
               objective_function: Callable,
               bounds: SearchBounds,
               max_iterations: int,
               scorer: ResultScorer) -> Dict:
        """Run Gaussian Process Bayesian optimization"""
        
        print(f"\n{'='*80}")
        print(f"🎯 BAYESIAN OPTIMIZATION (Gaussian Process)")
        print(f"   Initial random points: {self.n_initial_points}")
        print(f"   Acquisition: {self.acquisition}")
        print(f"{'='*80}\n")
        
        all_results = []
        best_result = None
        best_score = float('-inf')
        
        X_train = []
        y_train = []
        
        # Phase 1: Random initialization
        for i in range(min(self.n_initial_points, max_iterations)):
            # Random config
            config = WindowConfig(
                window_size=random.randint(bounds.min_window_size, bounds.max_window_size),
                offset=random.randint(bounds.min_offset, bounds.max_offset),
                sessions=random.choice(bounds.session_options),
                skip_min=random.randint(bounds.min_skip_min, bounds.max_skip_min),
                skip_max=random.randint(bounds.min_skip_max, bounds.max_skip_max),
                forward_threshold=round(random.uniform(bounds.min_forward_threshold, bounds.max_forward_threshold), 2),
                reverse_threshold=round(random.uniform(bounds.min_reverse_threshold, bounds.max_reverse_threshold), 2)
            )
            config.skip_max = max(config.skip_max, config.skip_min)
            
            result = objective_function(config)
            result.iteration = i
            score = scorer.score(result)
            
            all_results.append(result)
            X_train.append(self._config_to_vector(config, bounds))
            y_train.append(score)
            
            if score > best_score:
                best_score = score
                best_result = result
                print(f"✨ NEW BEST [Init {i + 1}]: {config.description()}")
                print(f"   Score: {score:.2f} (Bidirectional: {result.bidirectional_count})\n")
            else:
                print(f"   Init {i + 1}: {config.description()} → Score: {score:.2f}")
        
        # Phase 2: GP-guided search
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        kernel = Matern(nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        
        for i in range(self.n_initial_points, max_iterations):
            # Fit GP to current data
            gp.fit(X_train, y_train)
            
            # Find best point via acquisition function
            n_candidates = 1000
            X_candidates = np.random.rand(n_candidates, 5)
            
            acquisition_values = []
            for x in X_candidates:
                acq = self._acquisition_function(gp, X_train, y_train, x)
                acquisition_values.append(acq[0] if hasattr(acq, '__len__') else acq)
            
            best_idx = np.argmax(acquisition_values)
            x_next = X_candidates[best_idx]
            
            # Convert to config
            config = self._vector_to_config(x_next, bounds)
            
            # Evaluate
            result = objective_function(config)
            result.iteration = i
            score = scorer.score(result)
            
            all_results.append(result)
            X_train = np.vstack([X_train, x_next])
            y_train = np.append(y_train, score)
            
            if score > best_score:
                best_score = score
                best_result = result
                print(f"✨ NEW BEST [GP {i + 1}]: {config.description()}")
                print(f"   Score: {score:.2f} (Bidirectional: {result.bidirectional_count})\n")
            else:
                print(f"   GP {i + 1}: {config.description()} → Score: {score:.2f}")
        
        print(f"\n{'='*80}")
        print(f"🏆 OPTIMIZATION COMPLETE")
        print(f"   Best score: {best_score:.2f}")
        print(f"   Best config: {best_result.config.description()}")
        print(f"{'='*80}\n")
        
        return {
            'strategy': 'gaussian_process_bayesian',
            'best_config': best_result.config.to_dict(),
            'best_result': best_result.to_dict(),
            'best_score': best_score,
            'all_results': [r.to_dict() for r in all_results],
            'iterations': len(all_results)
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def create_bayesian_optimizer(method='auto', **kwargs):
    """
    Create appropriate Bayesian optimizer
    
    Args:
        method: 'auto', 'optuna', or 'sklearn'
        **kwargs: Additional arguments for the optimizer
        
    Returns:
        Bayesian optimizer instance
    """
    if method == 'auto':
        if OPTUNA_AVAILABLE:
            return OptunaBayesianSearch(**kwargs)
        elif SKLEARN_GP_AVAILABLE:
            return GaussianProcessBayesianSearch(**kwargs)
        else:
            raise ImportError("Neither Optuna nor scikit-learn available for Bayesian optimization")
    elif method == 'optuna':
        return OptunaBayesianSearch(**kwargs)
    elif method == 'sklearn':
        return GaussianProcessBayesianSearch(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    print("🎯 Bayesian Optimization Module")
    print(f"   Optuna available: {OPTUNA_AVAILABLE}")
    print(f"   sklearn GP available: {SKLEARN_GP_AVAILABLE}")
    
    if OPTUNA_AVAILABLE:
        print("\n✅ Recommended: Use Optuna-based Bayesian optimization")
    elif SKLEARN_GP_AVAILABLE:
        print("\n✅ Available: sklearn Gaussian Process optimization")
    else:
        print("\n❌ Install Optuna: pip install optuna")
