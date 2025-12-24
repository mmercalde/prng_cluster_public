#!/usr/bin/env python3
"""
Global State Tracker - System-wide Statistical Pattern Detection
================================================================

Extracted to GPU-neutral module per Team Beta requirement (v2.2).
NO torch/CUDA/OpenCL imports allowed at module level.

Features computed (14 total):
- residue_8_entropy, residue_125_entropy, residue_1000_entropy
- power_of_two_bias
- frequency_bias_ratio
- suspicious_gap_percentage
- regime_change_detected, regime_age
- high_variance_count
- marker_390_variance, marker_804_variance, marker_575_variance
- reseed_probability
- temporal_stability

Usage:
    from models.global_state_tracker import GlobalStateTracker
    
    tracker = GlobalStateTracker(lottery_history, config={'mod': 1000})
    global_features = tracker.get_global_state()
    
    # Or get as numpy array in sorted key order:
    feature_values = tracker.get_feature_values()
    feature_names = tracker.get_feature_names()

Author: PRNG Analysis System
Date: December 2025
Version: 1.0.0
"""

import hashlib
from typing import List, Dict, Any, Optional
from collections import Counter

import numpy as np

# Team Beta Requirement #3: SciPy fallback
# Graceful degradation if scipy not installed
try:
    from scipy.stats import entropy as scipy_entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _numpy_entropy(pk: np.ndarray, qk: Optional[np.ndarray] = None) -> float:
    """
    Numpy-only entropy calculation (fallback when SciPy unavailable).
    
    Computes Shannon entropy: H(p) = -sum(p * log(p))
    Or KL divergence if qk provided: D(p||q) = sum(p * log(p/q))
    """
    pk = np.asarray(pk, dtype=np.float64)
    pk = pk / pk.sum()  # Normalize
    pk = pk[pk > 0]  # Remove zeros to avoid log(0)
    
    if qk is None:
        # Shannon entropy
        return float(-np.sum(pk * np.log(pk)))
    else:
        # KL divergence
        qk = np.asarray(qk, dtype=np.float64)
        qk = qk / qk.sum()
        # Add small epsilon to avoid division by zero
        qk = np.maximum(qk, 1e-10)
        pk = np.maximum(pk, 1e-10)
        return float(np.sum(pk * np.log(pk / qk)))


def entropy(pk: np.ndarray, qk: Optional[np.ndarray] = None) -> float:
    """
    Compute entropy with SciPy if available, else numpy fallback.
    """
    if SCIPY_AVAILABLE:
        return float(scipy_entropy(pk, qk))
    else:
        return _numpy_entropy(pk, qk)


class GlobalStateTracker:
    """
    Track system-wide statistical patterns (14 features).
    
    These features capture anomalies discovered in real lottery data
    that per-seed features cannot detect:
    - Regime changes in PRNG behavior over time
    - Power-of-two bias in certain number ranges
    - Marker number variance (390, 804, 575 showed unusual patterns)
    - Reseed probability indicators
    - Temporal stability across the full history
    """

    def __init__(self, lottery_history: List[int], config: Optional[Dict[str, Any]] = None):
        """
        Initialize GlobalStateTracker.
        
        Args:
            lottery_history: List of historical lottery draws
            config: Optional configuration dict with keys:
                - mod: Modulo value (default: 1000)
                - gap_threshold: Threshold for suspicious gaps (default: 500)
                - window_size: Window size for regime detection (default: 1000)
                - regime_change_threshold: KL divergence threshold (default: 0.15)
                - marker_numbers: List of marker numbers to track (default: [390, 804, 575])
        """
        self.lottery_history = lottery_history
        self.config = config or {'mod': 1000}
        self._cache: Dict[str, float] = {}
        self._history_hash = self._compute_hash(lottery_history)
        self.current_regime_start = 0
        self.regime_history: List[int] = []
        self._initialized = False

    def _compute_hash(self, data: List[int]) -> str:
        """Compute hash of lottery history for cache invalidation."""
        return hashlib.md5(str(data).encode()).hexdigest()

    def get_global_state(self) -> Dict[str, float]:
        """
        Get all 14 global features (cached).
        
        Returns:
            Dictionary mapping feature names to values.
        """
        current_hash = self._compute_hash(self.lottery_history)
        if current_hash != self._history_hash or not self._cache:
            self._cache = self._compute_global_state()
            self._history_hash = current_hash
        return self._cache

    def get_feature_names(self) -> List[str]:
        """
        Get sorted list of global feature names.
        
        Returns:
            List of 14 feature names in lexicographic order.
        """
        return sorted(self.get_global_state().keys())

    def get_feature_values(self) -> np.ndarray:
        """
        Get global feature values as numpy array in sorted key order.
        
        Returns:
            np.ndarray of shape (14,) with feature values.
        """
        state = self.get_global_state()
        return np.array([state[k] for k in sorted(state.keys())], dtype=np.float32)

    def _compute_global_state(self) -> Dict[str, float]:
        """Compute all global features."""
        if len(self.lottery_history) < 100:
            return self._default_state()

        state: Dict[str, float] = {}
        state.update(self._compute_residue_distributions())
        state.update(self._detect_power_of_two_bias())
        state.update(self._detect_frequency_anomalies())
        state.update(self._detect_regime_changes())
        state.update(self._track_marker_numbers())
        state.update(self._compute_temporal_stability())
        return state

    def _default_state(self) -> Dict[str, float]:
        """Return default state when history is too short."""
        return {
            'frequency_bias_ratio': 1.0,
            'high_variance_count': 0.0,
            'marker_390_variance': 0.0,
            'marker_575_variance': 0.0,
            'marker_804_variance': 0.0,
            'power_of_two_bias': 0.0,
            'regime_age': 0.0,
            'regime_change_detected': 0.0,
            'reseed_probability': 0.0,
            'residue_1000_entropy': 0.0,
            'residue_125_entropy': 0.0,
            'residue_8_entropy': 0.0,
            'suspicious_gap_percentage': 0.0,
            'temporal_stability': 1.0,
        }

    def _compute_residue_distributions(self) -> Dict[str, float]:
        """Compute entropy of residue distributions for different moduli."""
        mod = self.config.get('mod', 1000)
        result: Dict[str, float] = {}
        
        for res_mod in [8, 125, 1000]:
            residue_counts = Counter([x % res_mod for x in self.lottery_history])
            total = len(self.lottery_history)
            
            # Build probability distribution
            probs = np.array([residue_counts.get(i, 0) / total for i in range(res_mod)])
            
            # Compute normalized entropy
            ent = entropy(probs + 1e-10)  # Add epsilon to avoid log(0)
            max_entropy = np.log(res_mod) if res_mod > 1 else 1.0
            normalized_entropy = ent / max_entropy if max_entropy > 0 else 0.0
            
            result[f'residue_{res_mod}_entropy'] = float(normalized_entropy)
        
        return result

    def _detect_power_of_two_bias(self) -> Dict[str, float]:
        """Detect bias toward power-of-two numbers."""
        mod = self.config.get('mod', 1000)
        powers_of_two = [2**i for i in range(10) if 2**i < mod]
        
        if not powers_of_two:
            return {'power_of_two_bias': 0.0}
        
        power_two_count = sum(1 for x in self.lottery_history if x in powers_of_two)
        expected_rate = len(powers_of_two) / float(mod)
        actual_rate = power_two_count / len(self.lottery_history) if self.lottery_history else 0
        
        bias = actual_rate / expected_rate if expected_rate > 0 else 1.0
        return {'power_of_two_bias': float(bias)}

    def _detect_frequency_anomalies(self) -> Dict[str, float]:
        """Detect frequency distribution anomalies."""
        freq_counter = Counter(self.lottery_history)
        
        if not freq_counter:
            return {'frequency_bias_ratio': 1.0, 'suspicious_gap_percentage': 0.0}
        
        max_freq = max(freq_counter.values())
        min_freq = min(freq_counter.values())
        ratio = max_freq / min_freq if min_freq > 0 else 1.0
        
        # Check for suspicious gaps
        gap_threshold = self.config.get('gap_threshold', 500)
        mod = self.config.get('mod', 1000)
        
        last_seen: Dict[int, int] = {}
        for i, num in enumerate(self.lottery_history):
            last_seen[num] = i
        
        current_index = len(self.lottery_history) - 1
        suspicious_count = 0
        
        for num in range(mod):
            if num not in last_seen:
                suspicious_count += 1
            elif current_index - last_seen[num] > gap_threshold:
                suspicious_count += 1
        
        suspicious_pct = suspicious_count / float(mod)
        
        return {
            'frequency_bias_ratio': float(ratio),
            'suspicious_gap_percentage': float(suspicious_pct)
        }

    def _detect_regime_changes(self) -> Dict[str, float]:
        """Detect statistical regime changes over time."""
        window_size = self.config.get('window_size', 1000)
        threshold = self.config.get('regime_change_threshold', 0.15)
        mod = self.config.get('mod', 1000)
        
        if len(self.lottery_history) < window_size * 2:
            return {'regime_change_detected': 0.0, 'regime_age': 0.0}
        
        recent = self.lottery_history[-window_size:]
        historical = self.lottery_history[-2*window_size:-window_size]
        
        recent_dist = Counter(recent)
        historical_dist = Counter(historical)
        
        # Compute KL divergence
        divergence = 0.0
        for num in range(mod):
            p = (recent_dist.get(num, 0) + 1) / (window_size + mod)
            q = (historical_dist.get(num, 0) + 1) / (window_size + mod)
            divergence += p * np.log(p / q)
        
        regime_changed = 1.0 if divergence > threshold else 0.0
        
        if regime_changed > 0.5:
            self.current_regime_start = len(self.lottery_history)
        
        regime_age = len(self.lottery_history) - self.current_regime_start
        
        return {
            'regime_change_detected': float(regime_changed),
            'regime_age': float(regime_age)
        }

    def _track_marker_numbers(self) -> Dict[str, float]:
        """Track variance in specific marker numbers discovered in real data."""
        marker_numbers = self.config.get('marker_numbers', [390, 804, 575])
        metrics: Dict[str, float] = {}
        
        for marker in marker_numbers:
            appearances = [i for i, x in enumerate(self.lottery_history) if x == marker]
            
            if len(appearances) < 2:
                metrics[f'marker_{marker}_variance'] = 0.0
                continue
            
            gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
            
            if not gaps:
                metrics[f'marker_{marker}_variance'] = 0.0
                continue
            
            mean_gap = np.mean(gaps)
            std_gap = np.std(gaps)
            cv = std_gap / mean_gap if mean_gap > 0 else 0.0  # Coefficient of variation
            metrics[f'marker_{marker}_variance'] = float(cv)
        
        # Count high variance markers
        high_variance_count = sum(1 for v in metrics.values() if v > 1.0)
        reseed_prob = high_variance_count / len(marker_numbers) if marker_numbers else 0.0
        
        metrics['reseed_probability'] = float(reseed_prob)
        metrics['high_variance_count'] = float(high_variance_count)
        
        return metrics

    def _compute_temporal_stability(self) -> Dict[str, float]:
        """Compute temporal stability of the distribution."""
        window_size = min(100, len(self.lottery_history) // 4)
        
        if len(self.lottery_history) < window_size * 2:
            return {'temporal_stability': 1.0}
        
        windows: List[Counter] = []
        for i in range(4):
            start = len(self.lottery_history) - (i+1) * window_size
            end = len(self.lottery_history) - i * window_size
            if start >= 0:
                windows.append(Counter(self.lottery_history[start:end]))
        
        if len(windows) < 2:
            return {'temporal_stability': 1.0}
        
        mod = self.config.get('mod', 1000)
        overlaps: List[float] = []
        
        for i in range(len(windows)-1):
            common = set(windows[i].keys()) & set(windows[i+1].keys())
            overlap = len(common) / float(mod)
            overlaps.append(overlap)
        
        stability = float(np.mean(overlaps)) if overlaps else 1.0
        return {'temporal_stability': stability}

    def update_history(self, new_draws: List[int]) -> None:
        """
        Update lottery history with new draws (invalidates cache).
        
        Args:
            new_draws: List of new lottery draws to append
        """
        self.lottery_history.extend(new_draws)
        # Cache will be invalidated on next get_global_state() call


# Constants for validation
GLOBAL_FEATURE_COUNT = 14

GLOBAL_FEATURE_NAMES = [
    'frequency_bias_ratio',
    'high_variance_count',
    'marker_390_variance',
    'marker_575_variance',
    'marker_804_variance',
    'power_of_two_bias',
    'regime_age',
    'regime_change_detected',
    'reseed_probability',
    'residue_1000_entropy',
    'residue_125_entropy',
    'residue_8_entropy',
    'suspicious_gap_percentage',
    'temporal_stability',
]


# Self-test
if __name__ == '__main__':
    print("GlobalStateTracker Self-Test")
    print("=" * 50)
    
    # Test with sample data
    test_history = list(range(1000)) * 5  # 5000 draws
    tracker = GlobalStateTracker(test_history, {'mod': 1000})
    
    state = tracker.get_global_state()
    print(f"Features computed: {len(state)}")
    print(f"Expected: {GLOBAL_FEATURE_COUNT}")
    
    assert len(state) == GLOBAL_FEATURE_COUNT, f"Feature count mismatch: {len(state)} != {GLOBAL_FEATURE_COUNT}"
    
    # Verify feature names match
    computed_names = sorted(state.keys())
    assert computed_names == GLOBAL_FEATURE_NAMES, f"Feature names mismatch"
    
    # Test helper methods
    names = tracker.get_feature_names()
    values = tracker.get_feature_values()
    
    assert len(names) == GLOBAL_FEATURE_COUNT
    assert len(values) == GLOBAL_FEATURE_COUNT
    assert values.dtype == np.float32
    
    print(f"\nAll features:")
    for name, value in zip(names, values):
        print(f"  {name}: {value:.6f}")
    
    print(f"\nSciPy available: {SCIPY_AVAILABLE}")
    print("\n✅ All tests passed!")
