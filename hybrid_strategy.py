#!/usr/bin/env python3
"""
Hybrid Variable Skip Strategy Definitions
Multi-strategy testing framework for variable skip detection

This module is completely standalone and optional.
Existing code continues to work without importing this.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import json


@dataclass
class StrategyConfig:
    """Configuration for a variable skip strategy"""
    name: str
    max_consecutive_misses: int  # How many misses before declaring breakpoint
    skip_tolerance: int          # Search window around expected skip (±tolerance)
    enable_reseed_search: bool   # Search for new seed at breakpoints
    skip_learning_rate: float    # How fast to adapt expected skip (0.0-1.0)
    breakpoint_threshold: float  # Match rate drop that indicates breakpoint
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# STRATEGY PRESETS
# ============================================================================

STRATEGY_PRESETS = {
    'strict_continuous': StrategyConfig(
        name='Strict Continuous (No Reseed)',
        max_consecutive_misses=3,
        skip_tolerance=5,
        enable_reseed_search=False,
        skip_learning_rate=0.3,
        breakpoint_threshold=0.5
    ),
    
    'lenient_continuous': StrategyConfig(
        name='Lenient Continuous (Large Gaps OK)',
        max_consecutive_misses=10,
        skip_tolerance=20,
        enable_reseed_search=False,
        skip_learning_rate=0.5,
        breakpoint_threshold=0.3
    ),
    
    'aggressive_reseed': StrategyConfig(
        name='Aggressive Reseed Detection',
        max_consecutive_misses=5,
        skip_tolerance=5,
        enable_reseed_search=True,
        skip_learning_rate=0.2,
        breakpoint_threshold=0.6
    ),
    
    'balanced_hybrid': StrategyConfig(
        name='Balanced Hybrid (Recommended)',
        max_consecutive_misses=7,
        skip_tolerance=10,
        enable_reseed_search=True,
        skip_learning_rate=0.4,
        breakpoint_threshold=0.5
    ),
    
    'extreme_tolerance': StrategyConfig(
        name='Extreme Tolerance (Catch Everything)',
        max_consecutive_misses=20,
        skip_tolerance=50,
        enable_reseed_search=False,
        skip_learning_rate=0.7,
        breakpoint_threshold=0.2
    ),
}


def get_all_strategies() -> List[StrategyConfig]:
    """Get all predefined strategies for testing"""
    return list(STRATEGY_PRESETS.values())


def get_strategy(name: str) -> StrategyConfig:
    """Get specific strategy by name"""
    if name not in STRATEGY_PRESETS:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(STRATEGY_PRESETS.keys())}")
    return STRATEGY_PRESETS[name]


def get_custom_strategy(
    max_misses: int = 7,
    skip_tolerance: int = 10,
    reseed_search: bool = True
) -> StrategyConfig:
    """Create custom strategy with specific parameters"""
    return StrategyConfig(
        name='Custom',
        max_consecutive_misses=max_misses,
        skip_tolerance=skip_tolerance,
        enable_reseed_search=reseed_search,
        skip_learning_rate=0.4,
        breakpoint_threshold=0.5
    )


@dataclass
class StrategyResult:
    """Results from testing a strategy"""
    strategy_name: str
    seed: int
    match_rate: float
    matches: int
    total: int
    skip_pattern: List[int]
    skip_stats: Dict[str, float]
    segments: List[Dict[str, Any]]
    breakpoints_detected: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def rank_strategies(results: List[StrategyResult]) -> List[StrategyResult]:
    """Rank strategies by match rate (best first)"""
    return sorted(results, key=lambda x: x.match_rate, reverse=True)


def analyze_skip_pattern(skip_pattern: List[int]) -> Dict[str, float]:
    """Calculate statistics for skip pattern"""
    if not skip_pattern:
        return {'min': 0, 'max': 0, 'avg': 0, 'variance': 0, 'std_dev': 0}
    
    import statistics
    return {
        'min': min(skip_pattern),
        'max': max(skip_pattern),
        'avg': statistics.mean(skip_pattern),
        'variance': statistics.variance(skip_pattern) if len(skip_pattern) > 1 else 0,
        'std_dev': statistics.stdev(skip_pattern) if len(skip_pattern) > 1 else 0,
    }


if __name__ == '__main__':
    print("Hybrid Strategy Module")
    print("=" * 60)
    print(f"\nAvailable strategies: {len(STRATEGY_PRESETS)}")
    for name, config in STRATEGY_PRESETS.items():
        print(f"\n{name}:")
        print(f"  {config.name}")
        print(f"  Max misses: {config.max_consecutive_misses}")
        print(f"  Skip tolerance: ±{config.skip_tolerance}")
        print(f"  Reseed search: {config.enable_reseed_search}")
