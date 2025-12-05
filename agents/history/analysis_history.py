#!/usr/bin/env python3
"""
Analysis History - Multi-run trend analysis and anomaly detection.

Tracks metrics across runs to enable AI agents to:
1. Detect improving/degrading trends
2. Identify anomalies
3. Make informed decisions based on historical context

Version: 3.2.0
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import statistics
import json


class TrendDirection(str, Enum):
    """Direction of metric trend."""
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    INSUFFICIENT_DATA = "insufficient_data"


class AnomalyType(str, Enum):
    """Types of detected anomalies."""
    SPIKE = "spike"
    DROP = "drop"
    PLATEAU = "plateau"
    OSCILLATION = "oscillation"
    NONE = "none"


class RunRecord(BaseModel):
    """Single run record with key metrics."""
    
    run_id: str
    run_number: int
    timestamp: datetime
    agent_name: str
    pipeline_step: int
    
    # Core metrics
    success: bool
    confidence: float = Field(ge=0.0, le=1.0)
    execution_time_seconds: float = 0.0
    
    # Step-specific metrics (flexible dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # Decision info
    action_taken: str = ""  # proceed, retry, escalate
    param_adjustments: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return {
            "run_id": self.run_id,
            "run_number": self.run_number,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "pipeline_step": self.pipeline_step,
            "success": self.success,
            "confidence": self.confidence,
            "execution_time_seconds": self.execution_time_seconds,
            "metrics": self.metrics,
            "action_taken": self.action_taken,
            "param_adjustments": self.param_adjustments
        }


class MetricTrend(BaseModel):
    """Trend analysis for a single metric."""
    
    metric_name: str
    values: List[float] = Field(default_factory=list)
    timestamps: List[datetime] = Field(default_factory=list)
    
    direction: TrendDirection = TrendDirection.INSUFFICIENT_DATA
    slope: float = 0.0
    r_squared: float = 0.0
    
    mean: float = 0.0
    std_dev: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    
    anomaly: AnomalyType = AnomalyType.NONE
    anomaly_indices: List[int] = Field(default_factory=list)
    
    def analyze(self, min_samples: int = 3) -> "MetricTrend":
        """Analyze trend from values."""
        if len(self.values) < min_samples:
            self.direction = TrendDirection.INSUFFICIENT_DATA
            return self
        
        # Basic statistics
        self.mean = statistics.mean(self.values)
        self.std_dev = statistics.stdev(self.values) if len(self.values) > 1 else 0.0
        self.min_value = min(self.values)
        self.max_value = max(self.values)
        
        # Linear regression for trend
        n = len(self.values)
        x = list(range(n))
        
        x_mean = sum(x) / n
        y_mean = self.mean
        
        numerator = sum((x[i] - x_mean) * (self.values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator > 0:
            self.slope = numerator / denominator
            
            # R-squared
            y_pred = [y_mean + self.slope * (x[i] - x_mean) for i in range(n)]
            ss_res = sum((self.values[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((self.values[i] - y_mean) ** 2 for i in range(n))
            self.r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Determine direction
        threshold = 0.01 * self.mean if self.mean != 0 else 0.01
        if abs(self.slope) < threshold:
            self.direction = TrendDirection.STABLE
        elif self.slope > 0:
            self.direction = TrendDirection.IMPROVING
        else:
            self.direction = TrendDirection.DEGRADING
        
        # Detect anomalies
        self._detect_anomalies()
        
        return self
    
    def _detect_anomalies(self, z_threshold: float = 2.0):
        """Detect anomalies using z-score method."""
        if self.std_dev == 0 or len(self.values) < 3:
            self.anomaly = AnomalyType.NONE
            return
        
        self.anomaly_indices = []
        
        for i, val in enumerate(self.values):
            z_score = (val - self.mean) / self.std_dev
            if abs(z_score) > z_threshold:
                self.anomaly_indices.append(i)
                if z_score > 0:
                    self.anomaly = AnomalyType.SPIKE
                else:
                    self.anomaly = AnomalyType.DROP
        
        # Check for oscillation (alternating above/below mean)
        if len(self.values) >= 5:
            above_below = [1 if v > self.mean else -1 for v in self.values]
            changes = sum(1 for i in range(1, len(above_below)) if above_below[i] != above_below[i-1])
            if changes >= len(self.values) * 0.7:
                self.anomaly = AnomalyType.OSCILLATION
        
        # Check for plateau
        if len(self.values) >= 3 and self.std_dev < 0.01 * abs(self.mean):
            self.anomaly = AnomalyType.PLATEAU
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for context."""
        return {
            "metric": self.metric_name,
            "direction": self.direction.value,
            "mean": round(self.mean, 4),
            "std_dev": round(self.std_dev, 4),
            "min": self.min_value,
            "max": self.max_value,
            "samples": len(self.values),
            "anomaly": self.anomaly.value if self.anomaly != AnomalyType.NONE else None
        }


class AnalysisHistory(BaseModel):
    """
    Complete run history with trend analysis.
    
    Maintains rolling history of pipeline runs and provides
    trend analysis for AI agent decision-making.
    """
    
    runs: List[RunRecord] = Field(default_factory=list)
    max_history: int = Field(default=50)
    
    # Cached trends by metric name
    _trends: Dict[str, MetricTrend] = {}
    
    class Config:
        underscore_attrs_are_private = True
    
    def add_run(self, record: RunRecord):
        """Add a run record, maintaining max history."""
        self.runs.append(record)
        
        # Trim to max history
        if len(self.runs) > self.max_history:
            self.runs = self.runs[-self.max_history:]
        
        # Invalidate cached trends
        self._trends = {}
    
    def get_runs_for_agent(self, agent_name: str, limit: int = 10) -> List[RunRecord]:
        """Get recent runs for a specific agent."""
        agent_runs = [r for r in self.runs if r.agent_name == agent_name]
        return agent_runs[-limit:]
    
    def get_runs_for_step(self, step: int, limit: int = 10) -> List[RunRecord]:
        """Get recent runs for a pipeline step."""
        step_runs = [r for r in self.runs if r.pipeline_step == step]
        return step_runs[-limit:]
    
    def analyze_metric(
        self, 
        metric_name: str, 
        agent_name: Optional[str] = None,
        step: Optional[int] = None
    ) -> MetricTrend:
        """Analyze trend for a specific metric."""
        # Filter runs
        runs = self.runs
        if agent_name:
            runs = [r for r in runs if r.agent_name == agent_name]
        if step:
            runs = [r for r in runs if r.pipeline_step == step]
        
        # Extract metric values
        values = []
        timestamps = []
        
        for run in runs:
            if metric_name in run.metrics:
                val = run.metrics[metric_name]
                if isinstance(val, (int, float)):
                    values.append(float(val))
                    timestamps.append(run.timestamp)
            elif metric_name == "confidence":
                values.append(run.confidence)
                timestamps.append(run.timestamp)
            elif metric_name == "execution_time":
                values.append(run.execution_time_seconds)
                timestamps.append(run.timestamp)
        
        trend = MetricTrend(
            metric_name=metric_name,
            values=values,
            timestamps=timestamps
        )
        
        return trend.analyze()
    
    def get_success_rate(self, agent_name: Optional[str] = None, last_n: int = 10) -> float:
        """Calculate success rate for recent runs."""
        runs = self.runs
        if agent_name:
            runs = [r for r in runs if r.agent_name == agent_name]
        
        runs = runs[-last_n:]
        if not runs:
            return 0.0
        
        return sum(1 for r in runs if r.success) / len(runs)
    
    def get_retry_count(self, agent_name: Optional[str] = None, last_n: int = 10) -> int:
        """Count retries in recent runs."""
        runs = self.runs
        if agent_name:
            runs = [r for r in runs if r.agent_name == agent_name]
        
        runs = runs[-last_n:]
        return sum(1 for r in runs if r.action_taken == "retry")
    
    def detect_degradation(
        self, 
        metric_name: str,
        agent_name: Optional[str] = None,
        threshold: float = 0.2
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect if a metric is degrading significantly.
        
        Returns (is_degrading, reason)
        """
        trend = self.analyze_metric(metric_name, agent_name)
        
        if trend.direction == TrendDirection.INSUFFICIENT_DATA:
            return False, None
        
        if trend.direction == TrendDirection.DEGRADING:
            # Check if degradation is significant
            if trend.mean != 0:
                relative_slope = abs(trend.slope) / abs(trend.mean)
                if relative_slope > threshold:
                    return True, f"{metric_name} degrading: slope={trend.slope:.4f}, relative={relative_slope:.2%}"
        
        # Check for anomalies
        if trend.anomaly in [AnomalyType.DROP, AnomalyType.OSCILLATION]:
            return True, f"{metric_name} shows {trend.anomaly.value} anomaly"
        
        return False, None
    
    def to_context_dict(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate history context as clean dict for LLM.
        
        This is the hybrid JSON approach - data only, no prose.
        """
        runs = self.get_runs_for_agent(agent_name, limit=5) if agent_name else self.runs[-5:]
        
        # Key metrics to analyze
        key_metrics = ["confidence", "execution_time"]
        
        # Add step-specific metrics from recent runs
        for run in runs:
            key_metrics.extend(run.metrics.keys())
        key_metrics = list(set(key_metrics))[:6]  # Limit to 6 metrics
        
        # Analyze trends
        trends = {}
        for metric in key_metrics:
            trend = self.analyze_metric(metric, agent_name)
            if trend.direction != TrendDirection.INSUFFICIENT_DATA:
                trends[metric] = trend.to_dict()
        
        # Detect issues
        issues = []
        for metric in key_metrics:
            is_degrading, reason = self.detect_degradation(metric, agent_name)
            if is_degrading:
                issues.append(reason)
        
        return {
            "total_runs": len(self.runs),
            "agent_runs": len(runs) if agent_name else len(self.runs),
            "success_rate": round(self.get_success_rate(agent_name), 2),
            "retry_count": self.get_retry_count(agent_name),
            "recent_runs": [
                {
                    "run_number": r.run_number,
                    "success": r.success,
                    "confidence": r.confidence,
                    "action": r.action_taken
                }
                for r in runs[-3:]
            ],
            "trends": trends,
            "issues": issues if issues else None
        }
    
    def save(self, filepath: str):
        """Save history to JSON file."""
        data = {
            "runs": [r.to_dict() for r in self.runs],
            "max_history": self.max_history
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> "AnalysisHistory":
        """Load history from JSON file."""
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            runs = []
            for r in data.get("runs", []):
                r["timestamp"] = datetime.fromisoformat(r["timestamp"])
                runs.append(RunRecord.model_validate(r))
            
            return cls(
                runs=runs,
                max_history=data.get("max_history", 50)
            )
        except FileNotFoundError:
            return cls()
        except Exception as e:
            print(f"Warning: Could not load history: {e}")
            return cls()


# Convenience function
def load_history(filepath: str = "analysis_history.json") -> AnalysisHistory:
    """Load or create analysis history."""
    return AnalysisHistory.load(filepath)
