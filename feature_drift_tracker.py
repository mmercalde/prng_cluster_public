#!/usr/bin/env python3
"""
Feature Importance Drift Tracker (Phase 3)
==========================================

Tracks feature importance changes over time to detect:
- Model behavior shifts between training runs
- PRNG pattern changes (possible reseeding events)
- Feature degradation or emergence
- Training instability

Design Principle (Addendum G Compliant):
    This module is model-agnostic. It works with importance dictionaries
    from ANY source (Neural Network, XGBoost, etc.)

Author: Distributed PRNG Analysis System
Date: December 9, 2025
Version: 1.0.0
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field

# Setup module logger
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DriftAnalysis:
    """Results of comparing two feature importance snapshots."""
    
    # Timestamps
    current_timestamp: str
    baseline_timestamp: str
    
    # Model versions
    current_version: str
    baseline_version: str
    
    # Raw importance data
    current: Dict[str, float]
    baseline: Dict[str, float]
    
    # Computed deltas
    delta: Dict[str, float]
    
    # Summary metrics
    drift_score: float          # 0-1, higher = more drift
    max_drift_feature: str      # Feature with largest change
    max_drift_value: float      # Magnitude of largest change
    
    # Alerts
    drift_alert: bool           # True if exceeds threshold
    alert_threshold: float      # Threshold used
    
    # Top movers
    top_gainers: List[Tuple[str, float]]   # Features that gained importance
    top_losers: List[Tuple[str, float]]    # Features that lost importance
    
    # Category analysis
    statistical_drift: float    # Change in statistical feature weight
    global_drift: float         # Change in global state feature weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'current_timestamp': str(self.current_timestamp),
            'baseline_timestamp': str(self.baseline_timestamp),
            'current_version': str(self.current_version),
            'baseline_version': str(self.baseline_version),
            'drift_score': float(self.drift_score),
            'max_drift_feature': str(self.max_drift_feature),
            'max_drift_value': float(self.max_drift_value),
            'drift_alert': bool(self.drift_alert),
            'alert_threshold': float(self.alert_threshold),
            'top_gainers': [(str(k), float(v)) for k, v in self.top_gainers],
            'top_losers': [(str(k), float(v)) for k, v in self.top_losers],
            'statistical_drift': float(self.statistical_drift),
            'global_drift': float(self.global_drift),
            'delta': {str(k): float(v) for k, v in self.delta.items()}
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        status = "⚠️ DRIFT ALERT" if self.drift_alert else "✅ Stable"
        lines = [
            f"Feature Importance Drift Analysis",
            f"=" * 40,
            f"Status: {status}",
            f"Drift Score: {self.drift_score:.4f} (threshold: {self.alert_threshold})",
            f"Max Change: {self.max_drift_feature} ({self.max_drift_value:+.4f})",
            f"",
            f"Top Gainers:",
        ]
        for name, val in self.top_gainers[:3]:
            lines.append(f"  ↑ {name}: {val:+.4f}")
        lines.append(f"")
        lines.append(f"Top Losers:")
        for name, val in self.top_losers[:3]:
            lines.append(f"  ↓ {name}: {val:+.4f}")
        lines.append(f"")
        lines.append(f"Category Changes:")
        lines.append(f"  Statistical: {self.statistical_drift:+.4f}")
        lines.append(f"  Global State: {self.global_drift:+.4f}")
        
        return "\n".join(lines)


@dataclass
class DriftHistoryEntry:
    """Single entry in drift history."""
    timestamp: str
    model_version: str
    importance_by_feature: Dict[str, float]
    importance_by_category: Dict[str, float]
    top_5: List[str]
    drift_from_baseline: Optional[float] = None
    drift_from_previous: Optional[float] = None


@dataclass 
class DriftHistory:
    """Complete drift tracking history."""
    
    # Configuration
    drift_threshold: float = 0.15
    
    # Baseline (first recorded or manually set)
    baseline: Optional[Dict[str, float]] = None
    baseline_timestamp: Optional[str] = None
    baseline_version: Optional[str] = None
    
    # History of snapshots
    history: List[DriftHistoryEntry] = field(default_factory=list)
    
    # Alerts
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'drift_threshold': self.drift_threshold,
            'baseline': self.baseline,
            'baseline_timestamp': self.baseline_timestamp,
            'baseline_version': self.baseline_version,
            'history': [asdict(h) for h in self.history],
            'alerts': self.alerts
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DriftHistory":
        """Load from dictionary."""
        history_entries = [
            DriftHistoryEntry(**h) for h in data.get('history', [])
        ]
        return cls(
            drift_threshold=data.get('drift_threshold', 0.15),
            baseline=data.get('baseline'),
            baseline_timestamp=data.get('baseline_timestamp'),
            baseline_version=data.get('baseline_version'),
            history=history_entries,
            alerts=data.get('alerts', [])
        )
    
    def save(self, path: str = 'feature_importance_history.json'):
        """Save history to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Drift history saved to: {path}")
    
    @classmethod
    def load(cls, path: str = 'feature_importance_history.json') -> "DriftHistory":
        """Load history from JSON file."""
        path = Path(path)
        if not path.exists():
            logger.info(f"No existing history at {path}, creating new")
            return cls()
        
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# ============================================================================
# FEATURE CATEGORIES (from feature_importance.py)
# ============================================================================

STATISTICAL_FEATURES = [
    'score', 'confidence', 'exact_matches', 'total_predictions', 'best_offset',
    'residue_8_match_rate', 'residue_8_coherence', 'residue_8_kl_divergence',
    'residue_125_match_rate', 'residue_125_coherence', 'residue_125_kl_divergence',
    'residue_1000_match_rate', 'residue_1000_coherence', 'residue_1000_kl_divergence',
    'temporal_stability_mean', 'temporal_stability_std', 'temporal_stability_min',
    'temporal_stability_max', 'temporal_stability_trend',
    'pred_mean', 'pred_std', 'actual_mean', 'actual_std',
    'lane_agreement_8', 'lane_agreement_125', 'lane_consistency',
    'skip_entropy', 'skip_mean', 'skip_std', 'skip_range',
    'survivor_velocity', 'velocity_acceleration',
    'intersection_weight', 'survivor_overlap_ratio',
    'forward_count', 'reverse_count', 'intersection_count', 'intersection_ratio',
    'pred_min', 'pred_max',
    'residual_mean', 'residual_std', 'residual_abs_mean', 'residual_max_abs',
    'forward_only_count', 'reverse_only_count'
]

GLOBAL_STATE_FEATURES = [
    'residue_8_entropy', 'residue_125_entropy', 'residue_1000_entropy',
    'power_of_two_bias', 'frequency_bias_ratio', 'suspicious_gap_percentage',
    'regime_change_detected', 'regime_age', 'high_variance_count',
    'marker_390_variance', 'marker_804_variance', 'marker_575_variance',
    'reseed_probability', 'temporal_stability'
]


# ============================================================================
# DRIFT TRACKER CLASS
# ============================================================================

class FeatureDriftTracker:
    """
    Tracks feature importance drift over time.
    
    Usage:
        tracker = FeatureDriftTracker()
        
        # After Step 4
        tracker.record(importance_step4, version='step4_trial_42')
        
        # After Step 5
        analysis = tracker.record(importance_step5, version='step5_final')
        
        if analysis.drift_alert:
            print("Warning: Feature importance shifted significantly!")
    """
    
    def __init__(
        self,
        history_path: str = 'feature_importance_history.json',
        drift_threshold: float = 0.15,
        auto_save: bool = True
    ):
        """
        Initialize drift tracker.
        
        Args:
            history_path: Path to save/load history
            drift_threshold: Alert threshold (0-1)
            auto_save: Automatically save after each record
        """
        self.history_path = history_path
        self.drift_threshold = drift_threshold
        self.auto_save = auto_save
        
        # Load existing history or create new
        self.history = DriftHistory.load(history_path)
        self.history.drift_threshold = drift_threshold
        
        logger.info(f"FeatureDriftTracker initialized")
        logger.info(f"  History path: {history_path}")
        logger.info(f"  Drift threshold: {drift_threshold}")
        logger.info(f"  Existing entries: {len(self.history.history)}")
    
    def record(
        self,
        importance: Dict[str, float],
        model_version: str,
        timestamp: Optional[str] = None,
        set_as_baseline: bool = False
    ) -> Optional[DriftAnalysis]:
        """
        Record a new feature importance snapshot and analyze drift.
        
        Args:
            importance: Feature importance dictionary
            model_version: Version string (e.g., 'step4_trial_42')
            timestamp: ISO timestamp (auto-generated if None)
            set_as_baseline: Force this snapshot as new baseline
        
        Returns:
            DriftAnalysis if baseline exists, None otherwise
        """
        timestamp = timestamp or datetime.now().isoformat()
        
        # Compute category weights
        stat_weight = sum(importance.get(f, 0.0) for f in STATISTICAL_FEATURES)
        global_weight = sum(importance.get(f, 0.0) for f in GLOBAL_STATE_FEATURES)
        
        # Get top 5 features
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_5 = [f for f, _ in sorted_features[:5]]
        
        # Create history entry
        entry = DriftHistoryEntry(
            timestamp=timestamp,
            model_version=model_version,
            importance_by_feature=importance,
            importance_by_category={
                'statistical_features': stat_weight,
                'global_state_features': global_weight
            },
            top_5=top_5
        )
        
        # Set baseline if needed
        if set_as_baseline or self.history.baseline is None:
            self.history.baseline = importance
            self.history.baseline_timestamp = timestamp
            self.history.baseline_version = model_version
            logger.info(f"Set baseline: {model_version}")
        
        # Analyze drift
        analysis = None
        if self.history.baseline is not None and not set_as_baseline:
            analysis = self.analyze_drift(
                current=importance,
                baseline=self.history.baseline,
                current_version=model_version,
                baseline_version=self.history.baseline_version,
                current_timestamp=timestamp,
                baseline_timestamp=self.history.baseline_timestamp
            )
            
            entry.drift_from_baseline = analysis.drift_score
            
            # Check for drift from previous
            if self.history.history:
                prev = self.history.history[-1]
                prev_analysis = self.analyze_drift(
                    current=importance,
                    baseline=prev.importance_by_feature,
                    current_version=model_version,
                    baseline_version=prev.model_version,
                    current_timestamp=timestamp,
                    baseline_timestamp=prev.timestamp
                )
                entry.drift_from_previous = prev_analysis.drift_score
            
            # Record alert if triggered
            if analysis.drift_alert:
                alert = {
                    'timestamp': timestamp,
                    'model_version': model_version,
                    'drift_score': analysis.drift_score,
                    'max_drift_feature': analysis.max_drift_feature,
                    'message': f"Drift alert: {analysis.max_drift_feature} changed by {analysis.max_drift_value:+.4f}"
                }
                self.history.alerts.append(alert)
                logger.warning(f"⚠️ DRIFT ALERT: {alert['message']}")
        
        # Add to history
        self.history.history.append(entry)
        
        # Auto-save
        if self.auto_save:
            self.history.save(self.history_path)
        
        # Log summary
        logger.info(f"Recorded: {model_version}")
        logger.info(f"  Top 3: {top_5[:3]}")
        logger.info(f"  Statistical: {stat_weight:.4f}, Global: {global_weight:.4f}")
        if analysis:
            logger.info(f"  Drift score: {analysis.drift_score:.4f}")
        
        return analysis
    
    def analyze_drift(
        self,
        current: Dict[str, float],
        baseline: Dict[str, float],
        current_version: str = 'current',
        baseline_version: str = 'baseline',
        current_timestamp: str = None,
        baseline_timestamp: str = None
    ) -> DriftAnalysis:
        """
        Analyze drift between two importance snapshots.
        
        Args:
            current: Current importance dict
            baseline: Baseline importance dict
            current_version: Current model version
            baseline_version: Baseline model version
            current_timestamp: Current timestamp
            baseline_timestamp: Baseline timestamp
        
        Returns:
            DriftAnalysis object
        """
        current_timestamp = current_timestamp or datetime.now().isoformat()
        baseline_timestamp = baseline_timestamp or 'unknown'
        
        # Compute delta for each feature
        all_features = set(current.keys()) | set(baseline.keys())
        delta = {}
        for feature in all_features:
            curr_val = current.get(feature, 0.0)
            base_val = baseline.get(feature, 0.0)
            delta[feature] = curr_val - base_val
        
        # Compute drift score (mean absolute delta)
        if delta:
            drift_score = np.mean([abs(v) for v in delta.values()])
        else:
            drift_score = 0.0
        
        # Find max drift feature
        if delta:
            max_feature = max(delta.items(), key=lambda x: abs(x[1]))
            max_drift_feature = max_feature[0]
            max_drift_value = max_feature[1]
        else:
            max_drift_feature = 'none'
            max_drift_value = 0.0
        
        # Top gainers and losers
        sorted_delta = sorted(delta.items(), key=lambda x: x[1], reverse=True)
        top_gainers = [(k, v) for k, v in sorted_delta if v > 0][:5]
        top_losers = [(k, v) for k, v in sorted_delta if v < 0][-5:][::-1]
        
        # Category drift
        stat_current = sum(current.get(f, 0.0) for f in STATISTICAL_FEATURES)
        stat_baseline = sum(baseline.get(f, 0.0) for f in STATISTICAL_FEATURES)
        statistical_drift = stat_current - stat_baseline
        
        global_current = sum(current.get(f, 0.0) for f in GLOBAL_STATE_FEATURES)
        global_baseline = sum(baseline.get(f, 0.0) for f in GLOBAL_STATE_FEATURES)
        global_drift = global_current - global_baseline
        
        # Check alert threshold
        drift_alert = drift_score > self.drift_threshold
        
        return DriftAnalysis(
            current_timestamp=current_timestamp,
            baseline_timestamp=baseline_timestamp,
            current_version=current_version,
            baseline_version=baseline_version,
            current=current,
            baseline=baseline,
            delta=delta,
            drift_score=drift_score,
            max_drift_feature=max_drift_feature,
            max_drift_value=max_drift_value,
            drift_alert=drift_alert,
            alert_threshold=self.drift_threshold,
            top_gainers=top_gainers,
            top_losers=top_losers,
            statistical_drift=statistical_drift,
            global_drift=global_drift
        )
    
    def get_baseline(self) -> Optional[Dict[str, float]]:
        """Get current baseline importance."""
        return self.history.baseline
    
    def set_baseline(self, importance: Dict[str, float], version: str):
        """Manually set a new baseline."""
        self.history.baseline = importance
        self.history.baseline_version = version
        self.history.baseline_timestamp = datetime.now().isoformat()
        
        if self.auto_save:
            self.history.save(self.history_path)
        
        logger.info(f"Baseline updated: {version}")
    
    def get_recent_drift_scores(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get drift scores from recent entries."""
        recent = self.history.history[-n:]
        return [
            (h.model_version, h.drift_from_baseline or 0.0)
            for h in recent
        ]
    
    def get_alerts(self, since: str = None) -> List[Dict[str, Any]]:
        """Get drift alerts, optionally filtered by timestamp."""
        if since is None:
            return self.history.alerts
        
        return [a for a in self.history.alerts if a['timestamp'] >= since]
    
    def clear_history(self, keep_baseline: bool = True):
        """Clear history (optionally keeping baseline)."""
        baseline = self.history.baseline if keep_baseline else None
        baseline_ts = self.history.baseline_timestamp if keep_baseline else None
        baseline_ver = self.history.baseline_version if keep_baseline else None
        
        self.history = DriftHistory(
            drift_threshold=self.drift_threshold,
            baseline=baseline,
            baseline_timestamp=baseline_ts,
            baseline_version=baseline_ver
        )
        
        if self.auto_save:
            self.history.save(self.history_path)
        
        logger.info("History cleared")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compare_importance_files(
    current_path: str,
    baseline_path: str,
    threshold: float = 0.15
) -> DriftAnalysis:
    """
    Compare two feature importance JSON files.
    
    Args:
        current_path: Path to current importance file
        baseline_path: Path to baseline importance file
        threshold: Drift alert threshold
    
    Returns:
        DriftAnalysis object
    """
    with open(current_path) as f:
        current_data = json.load(f)
    
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    
    # Extract importance dicts
    current = current_data.get('feature_importance', current_data)
    baseline = baseline_data.get('feature_importance', baseline_data)
    
    # Extract versions
    current_version = current_data.get('model_version', 'current')
    baseline_version = baseline_data.get('model_version', 'baseline')
    
    # Extract timestamps
    current_ts = current_data.get('timestamp', datetime.now().isoformat())
    baseline_ts = baseline_data.get('timestamp', 'unknown')
    
    # Create tracker and analyze
    tracker = FeatureDriftTracker(drift_threshold=threshold, auto_save=False)
    
    return tracker.analyze_drift(
        current=current,
        baseline=baseline,
        current_version=current_version,
        baseline_version=baseline_version,
        current_timestamp=current_ts,
        baseline_timestamp=baseline_ts
    )


def quick_drift_check(
    step4_path: str = 'feature_importance_step4.json',
    step5_path: str = 'feature_importance_step5.json',
    threshold: float = 0.15
) -> DriftAnalysis:
    """
    Quick drift check between Step 4 and Step 5 outputs.
    
    Args:
        step4_path: Path to Step 4 importance file
        step5_path: Path to Step 5 importance file
        threshold: Drift alert threshold
    
    Returns:
        DriftAnalysis object
    """
    return compare_importance_files(
        current_path=step5_path,
        baseline_path=step4_path,
        threshold=threshold
    )


def get_drift_summary_for_agent(analysis: DriftAnalysis) -> Dict[str, Any]:
    """
    Create compact summary suitable for agent_metadata injection.
    
    Args:
        analysis: DriftAnalysis object
    
    Returns:
        Compact summary dict for agents
    """
    return {
        'drift_score': round(float(analysis.drift_score), 4),
        'drift_alert': bool(analysis.drift_alert),
        'max_change_feature': str(analysis.max_drift_feature),
        'max_change_value': round(float(analysis.max_drift_value), 4),
        'top_gainers': [str(f) for f, _ in analysis.top_gainers[:3]],
        'top_losers': [str(f) for f, _ in analysis.top_losers[:3]],
        'statistical_drift': round(float(analysis.statistical_drift), 4),
        'global_drift': round(float(analysis.global_drift), 4),
        'status': 'ALERT' if analysis.drift_alert else 'stable'
    }


# ============================================================================
# MODULE TEST
# ============================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Feature Drift Tracker v1.0.0")
    print("=" * 50)
    
    # Create sample data for testing
    baseline_importance = {
        'lane_agreement_8': 0.15,
        'temporal_stability_mean': 0.12,
        'skip_entropy': 0.09,
        'residue_8_match_rate': 0.08,
        'score': 0.07,
        'confidence': 0.06,
        'pred_std': 0.05,
        'intersection_ratio': 0.04,
        'regime_age': 0.03,
        'reseed_probability': 0.02
    }
    
    # Simulate drift
    current_importance = {
        'lane_agreement_8': 0.12,      # Decreased
        'temporal_stability_mean': 0.14, # Increased
        'skip_entropy': 0.11,           # Increased
        'residue_8_match_rate': 0.06,   # Decreased
        'score': 0.08,                  # Increased
        'confidence': 0.05,             # Decreased
        'pred_std': 0.06,               # Increased
        'intersection_ratio': 0.03,     # Decreased
        'regime_age': 0.04,             # Increased
        'reseed_probability': 0.01      # Decreased
    }
    
    # Test tracker
    tracker = FeatureDriftTracker(
        history_path='/tmp/test_drift_history.json',
        drift_threshold=0.02,
        auto_save=False
    )
    
    # Record baseline
    tracker.record(baseline_importance, 'step4_baseline', set_as_baseline=True)
    
    # Record current and get analysis
    analysis = tracker.record(current_importance, 'step5_final')
    
    if analysis:
        print("\n" + analysis.summary())
        print("\n" + "=" * 50)
        print("Agent Summary:")
        print(json.dumps(get_drift_summary_for_agent(analysis), indent=2))
    
    print("\n✅ Feature Drift Tracker test complete!")
