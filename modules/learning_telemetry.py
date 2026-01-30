#!/usr/bin/env python3
"""
Learning Telemetry Module
=========================
Version: 1.1.1
Date: 2026-01-30
Status: Phase 8 Selfplay Integration

PURPOSE:
    Provides visibility into selfplay learning progress WITHOUT controlling decisions.
    This is the "black box flight recorder" - install it before you fly.

AUTHORITY CONTRACT (from CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md):

    ┌─────────────────────────────────────────────────────────────────┐
    │ Component    │ Access │ Allowed Methods                        │
    ├─────────────────────────────────────────────────────────────────┤
    │ Selfplay     │ WRITE  │ record_inner_episode()                 │
    │              │        │ record_policy_emission()               │
    ├─────────────────────────────────────────────────────────────────┤
    │ Chapter 13   │ WRITE  │ record_promotion() [observational]     │
    │              │ READ   │ get_health_snapshot()                  │
    │              │        │ get_health_warnings()                  │
    │              │        │ get_recent_episodes()                  │
    ├─────────────────────────────────────────────────────────────────┤
    │ WATCHER      │ READ   │ All get_*() methods                    │
    └─────────────────────────────────────────────────────────────────┘

INVARIANT 5 COMPLIANCE:
    "Telemetry may inform diagnostics and human review,
     but MUST NOT be the sole input to promotion,
     execution, or parameter selection."

SINGLE-WRITER MODEL:
    Only the selfplay_orchestrator process writes telemetry.
    Workers return results to the orchestrator; orchestrator records.
    This avoids JSONL interleaving from concurrent processes.
    
    ❌ Workers do NOT call record_*() methods directly
    ✅ Workers return results → Orchestrator calls record_*()

CRITICAL CONSTRAINTS:
    ✅ Append-only writes (JSONL format)
    ✅ Read-only consumption by Chapter 13/WATCHER
    ✅ Single-writer process model (orchestrator only)
    ❌ Does NOT trigger automated actions
    ❌ Does NOT decide anything
    ❌ Cannot be sole input to any decision

OUTPUT FILES:
    - learning_health.jsonl: Append-only telemetry log (versioned records)
    - learning_health_latest.json: Most recent snapshot (for quick reads)

USAGE:
    from modules.learning_telemetry import LearningTelemetry, TelemetryRecord
    
    # ONLY in orchestrator process:
    telemetry = LearningTelemetry(output_dir="telemetry/")
    
    # Selfplay writes (via orchestrator)
    telemetry.record_inner_episode(
        model_type="catboost",
        training_time_ms=142,
        fitness=0.83,
        val_r2=0.9999,
        survivor_count=2340
    )
    
    # Chapter 13 observational write
    telemetry.record_promotion(policy_id="policy_v3")
    
    # Chapter 13/WATCHER reads (observational only)
    health = telemetry.get_health_snapshot()
    warnings = telemetry.get_health_warnings()  # Informational only!
"""

from __future__ import annotations

import json
import os
import socket
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import deque
import math

# =============================================================================
# CONFIGURATION
# =============================================================================

SCHEMA_VERSION = "1.1.1"
DEFAULT_OUTPUT_DIR = "telemetry"
HEALTH_LOG_FILE = "learning_health.jsonl"
LATEST_SNAPSHOT_FILE = "learning_health_latest.json"

# Rolling window for metric calculations
ROLLING_WINDOW_SIZE = 100

# Time-based window for "last hour" calculation
LAST_HOUR_SECONDS = 3600

# Epsilon for numerical stability
EPSILON = 1e-9

# Health thresholds (from CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_5.md)
HEALTH_THRESHOLDS = {
    "inner_episode_throughput": {
        "healthy_min": 50.0,
        "healthy_max": 80.0,
        "warning_below": 30.0,
        "unit": "models/sec"
    },
    "policy_entropy": {
        "healthy_min": 0.2,
        "healthy_max": 0.6,
        "warning_below": 0.1,  # Premature convergence
        "unit": "entropy"
    },
    "recent_reward_trend": {
        "healthy_min": -5.0,
        "warning_below": -10.0,  # Regression
        "unit": "percent"
    },
    "last_promotion_days": {
        "healthy_max": 14,
        "warning_above": 21,  # Stalled
        "unit": "days"
    }
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class InnerEpisodeRecord:
    """Record of a single inner episode (ML training run)."""
    timestamp: str
    model_type: str
    training_time_ms: float
    fitness: float
    val_r2: float
    val_mae: Optional[float] = None
    fold_std: Optional[float] = None
    train_val_gap: Optional[float] = None
    survivor_count: int = 0
    feature_count: int = 0
    policy_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class PolicyRecord:
    """Record of a policy candidate emission."""
    timestamp: str
    policy_id: str
    source: str  # "selfplay", "manual", etc.
    fitness: float
    model_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    promoted: bool = False
    promoted_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TelemetrySnapshot:
    """
    Current health snapshot for observational purposes.
    
    CRITICAL: This is for OBSERVATION ONLY.
    Chapter 13 and WATCHER may READ this but MUST NOT use it
    as the sole input to any automated decision.
    """
    timestamp: str
    schema_version: str
    
    # Throughput metrics
    inner_episode_throughput: float  # models/sec (rolling average)
    training_time_avg_ms: float
    models_trained_total: int
    models_trained_last_hour: int  # Actual time-based count
    
    # Policy metrics
    policy_entropy: Optional[float]  # None if no data yet
    current_best_policy: Optional[str]
    policies_emitted_total: int
    last_promotion_days_ago: Optional[float]
    
    # Reward/fitness metrics
    recent_reward_trend: Optional[float]  # None if insufficient data
    fitness_avg: float
    fitness_std: float
    fitness_best: float
    
    # Data metrics
    survivor_count_avg: int
    
    # Health flags (INFORMATIONAL ONLY - not for automated decisions)
    health_warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# MAIN TELEMETRY CLASS
# =============================================================================

class LearningTelemetry:
    """
    Learning Telemetry System
    
    Provides append-only telemetry recording and read-only health snapshots.
    
    AUTHORITY (see header docstring for full matrix):
        - Selfplay (via orchestrator): record_inner_episode(), record_policy_emission()
        - Chapter 13: record_promotion() [observational], all get_*() methods
        - WATCHER: all get_*() methods
    
    SINGLE-WRITER MODEL:
        This class is designed for use by a SINGLE WRITER PROCESS (the orchestrator).
        Workers should return results to the orchestrator, which then calls record_*().
        This prevents JSONL interleaving from concurrent process writes.
        
        The threading.Lock protects against concurrent threads within that single
        process, NOT against multiple processes.
    """
    
    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        rolling_window: int = ROLLING_WINDOW_SIZE,
        run_id: Optional[str] = None
    ):
        """
        Initialize telemetry system.
        
        Args:
            output_dir: Directory for telemetry files
            rolling_window: Number of records for rolling calculations
            run_id: Optional run identifier for forensics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.health_log_path = self.output_dir / HEALTH_LOG_FILE
        self.latest_snapshot_path = self.output_dir / LATEST_SNAPSHOT_FILE
        
        self.rolling_window = rolling_window
        self._lock = threading.Lock()  # Protects within single process only
        
        # Metadata for forensics
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.host = socket.gethostname()
        self.pid = os.getpid()
        
        # In-memory rolling buffers (for fast metric calculation)
        # Each entry is (timestamp_dt, record)
        self._episode_buffer: deque = deque(maxlen=rolling_window)
        self._fitness_buffer: deque = deque(maxlen=rolling_window)
        self._timing_buffer: deque = deque(maxlen=rolling_window)
        
        # Time-indexed buffer for accurate "last hour" calculation
        self._episode_timestamps: deque = deque(maxlen=rolling_window * 10)
        
        # Counters
        self._models_trained_total = 0
        self._policies_emitted_total = 0
        self._last_promotion_timestamp: Optional[datetime] = None
        self._current_best_policy: Optional[str] = None
        self._current_best_fitness: float = -float('inf')
        
        # Policy tracking for entropy calculation
        self._policy_selections: deque = deque(maxlen=rolling_window)
        
        # Load existing state if available
        self._load_state()
        
        print(f"✅ LearningTelemetry v{SCHEMA_VERSION} initialized")
        print(f"   Output: {self.output_dir}")
        print(f"   Run ID: {self.run_id}")
        print(f"   Host: {self.host} (PID: {self.pid})")
        print(f"   Rolling window: {rolling_window}")
        print(f"   ⚠️  Single-writer model: Only orchestrator should call record_*()")
    
    # =========================================================================
    # WRITE METHODS
    # =========================================================================
    
    def record_inner_episode(
        self,
        model_type: str,
        training_time_ms: float,
        fitness: float,
        val_r2: float,
        survivor_count: int = 0,
        val_mae: Optional[float] = None,
        fold_std: Optional[float] = None,
        train_val_gap: Optional[float] = None,
        feature_count: int = 0,
        policy_id: Optional[str] = None
    ) -> InnerEpisodeRecord:
        """
        Record an inner episode (ML training run).
        
        AUTHORITY: Selfplay (via orchestrator process only)
        
        This is the primary telemetry collection point for selfplay.
        Called after each model training completes.
        
        Args:
            model_type: One of "lightgbm", "xgboost", "catboost"
            training_time_ms: Training duration in milliseconds
            fitness: Computed fitness score
            val_r2: Validation R² score
            survivor_count: Number of survivors used
            val_mae: Optional validation MAE
            fold_std: Optional fold standard deviation
            train_val_gap: Optional train/val gap
            feature_count: Number of features
            policy_id: Optional policy identifier
        
        Returns:
            The recorded InnerEpisodeRecord
        """
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()
        
        record = InnerEpisodeRecord(
            timestamp=timestamp,
            model_type=model_type,
            training_time_ms=training_time_ms,
            fitness=fitness,
            val_r2=val_r2,
            val_mae=val_mae,
            fold_std=fold_std,
            train_val_gap=train_val_gap,
            survivor_count=survivor_count,
            feature_count=feature_count,
            policy_id=policy_id
        )
        
        with self._lock:
            # Update buffers
            self._episode_buffer.append(record)
            self._fitness_buffer.append(fitness)
            self._timing_buffer.append(training_time_ms)
            self._episode_timestamps.append(now)
            self._models_trained_total += 1
            
            if policy_id:
                self._policy_selections.append(policy_id)
            
            # Track best
            if fitness > self._current_best_fitness:
                self._current_best_fitness = fitness
                self._current_best_policy = policy_id
            
            # Append to log (append-only!)
            self._append_to_log("inner_episode", record.to_dict())
            
            # Update latest snapshot
            self._update_snapshot()
        
        return record
    
    def record_policy_emission(
        self,
        policy_id: str,
        fitness: float,
        model_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        source: str = "selfplay"
    ) -> PolicyRecord:
        """
        Record a policy candidate emission.
        
        AUTHORITY: Selfplay (via orchestrator process only)
        
        Called when selfplay emits a learned_policy_candidate.json.
        
        Args:
            policy_id: Unique policy identifier
            fitness: Policy fitness score
            model_type: Model type used
            parameters: Policy parameters
            source: Origin of policy ("selfplay", "manual", etc.)
        
        Returns:
            The recorded PolicyRecord
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        record = PolicyRecord(
            timestamp=timestamp,
            policy_id=policy_id,
            source=source,
            fitness=fitness,
            model_type=model_type,
            parameters=parameters or {}
        )
        
        with self._lock:
            self._policies_emitted_total += 1
            
            self._append_to_log("policy_emission", record.to_dict())
            self._update_snapshot()
        
        return record
    
    def record_promotion(self, policy_id: str, fitness: Optional[float] = None) -> None:
        """
        Record that a policy was promoted by Chapter 13.
        
        AUTHORITY: Chapter 13 ONLY (observational write)
        
        This is an observational record. Telemetry observes the promotion
        but does not decide it. The decision authority belongs to Chapter 13.
        
        Args:
            policy_id: The promoted policy ID
            fitness: Optional fitness at promotion time (for forensic analysis)
        """
        timestamp = datetime.now(timezone.utc)
        
        with self._lock:
            self._last_promotion_timestamp = timestamp
            
            self._append_to_log("promotion", {
                "timestamp": timestamp.isoformat(),
                "policy_id": policy_id,
                "fitness_at_promotion": fitness,  # Forensic: peak or decay?
                "authority": "chapter_13"  # Explicit attribution
            })
            
            self._update_snapshot()
    
    # =========================================================================
    # READ METHODS (Chapter 13 / WATCHER)
    # =========================================================================
    
    def get_health_snapshot(self) -> TelemetrySnapshot:
        """
        Get current health snapshot.
        
        AUTHORITY: Chapter 13, WATCHER (read-only)
        
        CRITICAL: This is for OBSERVATION ONLY.
        Do NOT use as sole input to automated decisions.
        
        Returns:
            Current TelemetrySnapshot
        """
        with self._lock:
            return self._compute_snapshot()
    
    def get_health_warnings(self) -> List[str]:
        """
        Get current health warnings.
        
        AUTHORITY: Chapter 13, WATCHER (read-only)
        
        CRITICAL: These are INFORMATIONAL ONLY.
        They inform human review but MUST NOT trigger automated actions.
        
        Returns:
            List of warning messages (informational, not actionable)
        """
        snapshot = self.get_health_snapshot()
        return snapshot.health_warnings
    
    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Get most recent inner episode records.
        
        AUTHORITY: Chapter 13, WATCHER (read-only)
        
        Args:
            n: Number of records to return
        
        Returns:
            List of episode dictionaries
        """
        with self._lock:
            episodes = list(self._episode_buffer)[-n:]
            return [e.to_dict() for e in episodes]
    
    def get_throughput(self) -> float:
        """
        Get current throughput in models/sec.
        
        AUTHORITY: Chapter 13, WATCHER (read-only)
        
        Returns:
            Rolling average throughput
        """
        with self._lock:
            if not self._timing_buffer:
                return 0.0
            avg_ms = sum(self._timing_buffer) / len(self._timing_buffer)
            if avg_ms <= 0:
                return 0.0
            return 1000.0 / avg_ms
    
    def get_fitness_trend(self) -> Optional[float]:
        """
        Get recent fitness trend as percent change.
        
        AUTHORITY: Chapter 13, WATCHER (read-only)
        
        Returns:
            Percent change in fitness over rolling window, or None if insufficient data
        """
        with self._lock:
            return self._compute_fitness_trend()
    
    def get_policy_entropy(self) -> Optional[float]:
        """
        Get policy selection entropy.
        
        AUTHORITY: Chapter 13, WATCHER (read-only)
        
        Low entropy indicates premature convergence to one policy.
        
        Returns:
            Shannon entropy of policy selections, or None if no data
        """
        with self._lock:
            return self._compute_policy_entropy()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get a simplified health summary for dashboards.
        
        AUTHORITY: Chapter 13, WATCHER (read-only)
        
        CRITICAL: This is INFORMATIONAL ONLY.
        The 'status' field is a convenience label, NOT a decision trigger.
        
        Returns:
            Dict with status, warnings, and key metrics
        """
        snapshot = self.get_health_snapshot()
        warnings = snapshot.health_warnings
        
        # Derive status from warning count (INFORMATIONAL ONLY)
        if len(warnings) == 0:
            status = "healthy"
        elif len(warnings) <= 2:
            status = "warning"
        else:
            status = "degraded"
        
        return {
            "status": status,  # INFORMATIONAL - not for automated decisions
            "warnings": warnings,
            "warning_count": len(warnings),
            "models_trained_total": snapshot.models_trained_total,
            "models_trained_last_hour": snapshot.models_trained_last_hour,
            "throughput": snapshot.inner_episode_throughput,
            "fitness_best": snapshot.fitness_best,
            "policy_entropy": snapshot.policy_entropy,
            "last_promotion_days": snapshot.last_promotion_days_ago,
            "last_updated": snapshot.timestamp
        }
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    def _append_to_log(self, record_type: str, data: Dict[str, Any]) -> None:
        """
        Append record to JSONL log (append-only!).
        
        Each record includes schema version and metadata for forensics.
        """
        record = {
            "schema_version": SCHEMA_VERSION,
            "type": record_type,
            "run_id": self.run_id,
            "host": self.host,
            "pid": self.pid,
            "data": data
        }
        with open(self.health_log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def _update_snapshot(self) -> None:
        """
        Update the latest snapshot file atomically.
        
        Uses write-to-temp + atomic replace to prevent partial reads.
        """
        snapshot = self._compute_snapshot()
        tmp_path = self.latest_snapshot_path.with_suffix('.tmp')
        
        with open(tmp_path, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        
        # Atomic replace (POSIX guarantees this is atomic)
        os.replace(tmp_path, self.latest_snapshot_path)
    
    def _compute_snapshot(self) -> TelemetrySnapshot:
        """Compute current health snapshot."""
        timestamp = datetime.now(timezone.utc)
        
        # Throughput
        throughput = 0.0
        avg_time = 0.0
        if self._timing_buffer:
            avg_time = sum(self._timing_buffer) / len(self._timing_buffer)
            if avg_time > 0:
                throughput = 1000.0 / avg_time
        
        # Models trained in last hour (actual time-based calculation)
        models_last_hour = self._count_models_last_hour(timestamp)
        
        # Policy entropy (None if no data)
        policy_entropy = self._compute_policy_entropy()
        
        # Last promotion days
        last_promotion_days = None
        if self._last_promotion_timestamp:
            delta = timestamp - self._last_promotion_timestamp
            last_promotion_days = delta.total_seconds() / 86400.0
        
        # Fitness metrics
        fitness_trend = self._compute_fitness_trend()
        fitness_avg = 0.0
        fitness_std = 0.0
        fitness_best = 0.0
        if self._fitness_buffer:
            fitness_list = list(self._fitness_buffer)
            fitness_avg = sum(fitness_list) / len(fitness_list)
            fitness_best = max(fitness_list)
            if len(fitness_list) > 1:
                mean = fitness_avg
                variance = sum((x - mean) ** 2 for x in fitness_list) / len(fitness_list)
                fitness_std = math.sqrt(variance)
        
        # Survivor count average
        survivor_avg = 0
        if self._episode_buffer:
            counts = [e.survivor_count for e in self._episode_buffer if e.survivor_count > 0]
            if counts:
                survivor_avg = int(sum(counts) / len(counts))
        
        # Health warnings (INFORMATIONAL ONLY)
        warnings = self._compute_warnings(
            throughput=throughput,
            policy_entropy=policy_entropy,
            fitness_trend=fitness_trend,
            last_promotion_days=last_promotion_days
        )
        
        return TelemetrySnapshot(
            timestamp=timestamp.isoformat(),
            schema_version=SCHEMA_VERSION,
            inner_episode_throughput=round(throughput, 2),
            training_time_avg_ms=round(avg_time, 2),
            models_trained_total=self._models_trained_total,
            models_trained_last_hour=models_last_hour,
            policy_entropy=round(policy_entropy, 4) if policy_entropy is not None else None,
            current_best_policy=self._current_best_policy,
            policies_emitted_total=self._policies_emitted_total,
            last_promotion_days_ago=round(last_promotion_days, 2) if last_promotion_days else None,
            recent_reward_trend=round(fitness_trend, 2) if fitness_trend is not None else None,
            fitness_avg=round(fitness_avg, 4),
            fitness_std=round(fitness_std, 4),
            fitness_best=round(fitness_best, 4),
            survivor_count_avg=survivor_avg,
            health_warnings=warnings
        )
    
    def _count_models_last_hour(self, now: datetime) -> int:
        """
        Count models trained in the actual last hour.
        
        Uses time-indexed buffer for accurate calculation.
        """
        if not self._episode_timestamps:
            return 0
        
        cutoff = now - timedelta(seconds=LAST_HOUR_SECONDS)
        count = sum(1 for ts in self._episode_timestamps if ts >= cutoff)
        return count
    
    def _compute_fitness_trend(self) -> Optional[float]:
        """
        Compute fitness trend as percent change.
        
        Returns None if insufficient data (< 10 samples).
        Uses epsilon for numerical stability.
        """
        if len(self._fitness_buffer) < 10:
            return None  # Insufficient data
        
        fitness_list = list(self._fitness_buffer)
        half = len(fitness_list) // 2
        
        first_half = fitness_list[:half]
        second_half = fitness_list[half:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        # Use epsilon to prevent division explosion
        denominator = max(abs(avg_first), EPSILON)
        
        return ((avg_second - avg_first) / denominator) * 100.0
    
    def _compute_policy_entropy(self) -> Optional[float]:
        """
        Compute Shannon entropy of policy selections.
        
        Returns None if no policy selections have been recorded.
        This is more honest than returning 1.0 ("max diversity").
        """
        if not self._policy_selections:
            return None  # No data yet - entropy undefined
        
        # Count policy frequencies
        policy_counts: Dict[str, int] = {}
        for policy in self._policy_selections:
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
        
        # Single policy = zero entropy (no diversity)
        n_unique = len(policy_counts)
        if n_unique <= 1:
            return 0.0
        
        # Compute entropy
        total = len(self._policy_selections)
        entropy = 0.0
        for count in policy_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize to 0-1 range (max entropy = log2(n_unique_policies))
        max_entropy = math.log2(n_unique)
        entropy = entropy / max_entropy
        
        return entropy
    
    def _compute_warnings(
        self,
        throughput: float,
        policy_entropy: Optional[float],
        fitness_trend: Optional[float],
        last_promotion_days: Optional[float]
    ) -> List[str]:
        """
        Compute health warnings.
        
        CRITICAL: These are INFORMATIONAL ONLY.
        They inform human review but MUST NOT trigger automated actions.
        Per Invariant 5: Telemetry MUST NOT be sole input to any decision.
        
        Naming convention: All warnings are descriptive observations,
        never actionable directives (no "should_halt", "must_retrain", etc.)
        """
        warnings = []
        
        # Throughput warnings
        thresh = HEALTH_THRESHOLDS["inner_episode_throughput"]
        if throughput > 0 and throughput < thresh["warning_below"]:
            warnings.append(
                f"LOW_THROUGHPUT: {throughput:.1f} models/sec "
                f"(warning threshold: {thresh['warning_below']})"
            )
        
        # Policy entropy warnings
        if policy_entropy is not None:
            thresh = HEALTH_THRESHOLDS["policy_entropy"]
            if policy_entropy < thresh["warning_below"]:
                warnings.append(
                    f"PREMATURE_CONVERGENCE: policy_entropy={policy_entropy:.3f} "
                    f"(warning threshold: {thresh['warning_below']})"
                )
        elif self._models_trained_total > 20:
            # Only warn about missing entropy if we have enough training runs
            warnings.append(
                "ENTROPY_UNDEFINED: No policy selections recorded yet"
            )
        
        # Fitness trend warnings
        if fitness_trend is not None:
            thresh = HEALTH_THRESHOLDS["recent_reward_trend"]
            if fitness_trend < thresh["warning_below"]:
                warnings.append(
                    f"FITNESS_REGRESSION: {fitness_trend:+.1f}% trend "
                    f"(warning threshold: {thresh['warning_below']}%)"
                )
        
        # Promotion staleness warnings
        if last_promotion_days is not None:
            thresh = HEALTH_THRESHOLDS["last_promotion_days"]
            if last_promotion_days > thresh["warning_above"]:
                warnings.append(
                    f"STALLED_PROMOTION: {last_promotion_days:.1f} days since last promotion "
                    f"(warning threshold: {thresh['warning_above']} days)"
                )
        
        return warnings
    
    def _load_state(self) -> None:
        """
        Load state from existing snapshot if available.
        
        This allows resuming counts across restarts.
        Note: Rolling buffers are NOT restored (by design - they're ephemeral).
        """
        if not self.latest_snapshot_path.exists():
            return
        
        try:
            with open(self.latest_snapshot_path) as f:
                data = json.load(f)
            
            # Only restore counters, not rolling state
            self._models_trained_total = data.get("models_trained_total", 0)
            self._policies_emitted_total = data.get("policies_emitted_total", 0)
            self._current_best_policy = data.get("current_best_policy")
            
            # Restore last promotion timestamp if available
            last_promo_days = data.get("last_promotion_days_ago")
            if last_promo_days is not None:
                self._last_promotion_timestamp = (
                    datetime.now(timezone.utc) - timedelta(days=last_promo_days)
                )
            
            print(f"   Loaded state: {self._models_trained_total} models trained previously")
        except Exception as e:
            print(f"   Warning: Could not load previous state: {e}")


# =============================================================================
# CLI FOR TESTING
# =============================================================================

def main():
    """Test the telemetry system."""
    import argparse
    import random
    import time
    
    parser = argparse.ArgumentParser(description="Learning Telemetry Test")
    parser.add_argument("--output-dir", default="telemetry", help="Output directory")
    parser.add_argument("--simulate", type=int, default=0, help="Simulate N episodes")
    parser.add_argument("--show-snapshot", action="store_true", help="Show current snapshot")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    args = parser.parse_args()
    
    telemetry = LearningTelemetry(output_dir=args.output_dir, run_id=args.run_id)
    
    if args.show_snapshot:
        snapshot = telemetry.get_health_snapshot()
        print("\n📊 Current Health Snapshot:")
        print(json.dumps(snapshot.to_dict(), indent=2))
        
        warnings = telemetry.get_health_warnings()
        if warnings:
            print("\n⚠️  Health Warnings (INFORMATIONAL ONLY):")
            for w in warnings:
                print(f"   - {w}")
        else:
            print("\n✅ No health warnings")
        return
    
    if args.simulate > 0:
        print(f"\n🎲 Simulating {args.simulate} inner episodes...")
        
        model_types = ["lightgbm", "xgboost", "catboost"]
        policy_ids = [f"policy_v{i}" for i in range(1, 6)]
        
        for i in range(args.simulate):
            model = random.choice(model_types)
            policy = random.choice(policy_ids)
            
            # Simulate training
            training_time = random.uniform(80, 200)
            fitness = random.uniform(0.3, 0.9)
            val_r2 = random.uniform(0.95, 1.0)
            survivors = random.randint(1000, 5000)
            
            record = telemetry.record_inner_episode(
                model_type=model,
                training_time_ms=training_time,
                fitness=fitness,
                val_r2=val_r2,
                survivor_count=survivors,
                policy_id=policy
            )
            
            print(f"   [{i+1}/{args.simulate}] {model}: fitness={fitness:.3f}, "
                  f"time={training_time:.0f}ms")
            
            time.sleep(0.05)  # Small delay
        
        # Show final snapshot
        print("\n📊 Final Health Snapshot:")
        snapshot = telemetry.get_health_snapshot()
        print(json.dumps(snapshot.to_dict(), indent=2))
        
        warnings = telemetry.get_health_warnings()
        if warnings:
            print("\n⚠️  Health Warnings (INFORMATIONAL ONLY):")
            for w in warnings:
                print(f"   - {w}")
        else:
            print("\n✅ No health warnings")


if __name__ == "__main__":
    main()
