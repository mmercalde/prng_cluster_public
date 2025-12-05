"""
Analysis History Module - Multi-run trend analysis.

Exports:
    AnalysisHistory - Complete run history with trends
    RunRecord - Single run record
    MetricTrend - Trend analysis for a metric
    TrendDirection - Enum for trend direction
    AnomalyType - Enum for anomaly types
    load_history - Convenience loader
"""

from .analysis_history import (
    AnalysisHistory,
    RunRecord,
    MetricTrend,
    TrendDirection,
    AnomalyType,
    load_history
)
