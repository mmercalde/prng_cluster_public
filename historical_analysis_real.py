#!/usr/bin/env python3
"""
Historical Pattern Analysis - Statistical analysis of lottery draw history
Provides comprehensive statistical overview without PRNG reconstruction

This module integrates with the new results system and follows the modular architecture.
Compatible with advanced_search_manager.py and unified_system_working.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict
import statistics
from advanced_search_manager import HistoricalAnalysisConfig

# Ensure proper import path
sys.path.insert(0, str(Path(__file__).parent))
from core.results_manager import ResultsManager


def analyze_draw_frequency(draws: List[int], top_n: int = 10, bottom_n: int = 10) -> Dict[str, Any]:
    """Analyze frequency distribution of draws"""
    counter = Counter(draws)
    total = len(draws)
    
    return {
        'total_draws': total,
        'unique_values': len(counter),
        'most_common': counter.most_common(top_n),
        'least_common': counter.most_common()[-bottom_n:] if len(counter) > 10 else [],
        'frequency_distribution': dict(counter),
        'average_frequency': total / len(counter) if counter else 0
    }


def analyze_gaps(draws: List[int]) -> Dict[str, Any]:
    """Analyze gaps between number appearances"""
    last_seen = {}
    gaps = defaultdict(list)
    
    for idx, draw in enumerate(draws):
        if draw in last_seen:
            gap = idx - last_seen[draw]
            gaps[draw].append(gap)
        last_seen[draw] = idx
    
    gap_stats = {}
    for num, gap_list in gaps.items():
        if gap_list:
            gap_stats[num] = {
                'avg_gap': statistics.mean(gap_list),
                'min_gap': min(gap_list),
                'max_gap': max(gap_list),
                'gap_variance': statistics.variance(gap_list) if len(gap_list) > 1 else 0
            }
    
    return gap_stats


def analyze_streaks(draws: List[int], min_streak: int = 2) -> Dict[str, Any]:
    """Analyze consecutive appearances"""
    streaks = defaultdict(list)
    current_streak = 1
    
    for i in range(1, len(draws)):
        if draws[i] == draws[i-1]:
            current_streak += 1
        else:
            if current_streak >= min_streak:
                streaks[draws[i-1]].append(current_streak)
            current_streak = 1
    
    streak_summary = {}
    for num, streak_list in streaks.items():
        if streak_list:
            streak_summary[num] = {
                'max_streak': max(streak_list),
                'avg_streak': statistics.mean(streak_list),
                'total_streaks': len(streak_list)
            }
    
    return streak_summary


def analyze_temporal_patterns(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze patterns over time (by month, year)"""
    by_month = defaultdict(list)
    by_year = defaultdict(list)
    
    for entry in data:
        date_str = entry.get('date', '')
        draw = entry.get('draw')
        
        if date_str and draw is not None:
            try:
                year = date_str[:4]
                month = date_str[:7]  # YYYY-MM
                by_year[year].append(draw)
                by_month[month].append(draw)
            except:
                continue
    
    monthly_stats = {}
    for month, draws in sorted(by_month.items()):
        if draws:
            monthly_stats[month] = {
                'count': len(draws),
                'mean': statistics.mean(draws),
                'median': statistics.median(draws),
                'std_dev': statistics.stdev(draws) if len(draws) > 1 else 0
            }
    
    yearly_stats = {}
    for year, draws in sorted(by_year.items()):
        if draws:
            yearly_stats[year] = {
                'count': len(draws),
                'mean': statistics.mean(draws),
                'median': statistics.median(draws),
                'std_dev': statistics.stdev(draws) if len(draws) > 1 else 0
            }
    
    return {
        'monthly_stats': monthly_stats,
        'yearly_stats': yearly_stats
    }


def calculate_entropy(draws: List[int]) -> float:
    """Calculate Shannon entropy of draw distribution"""
    counter = Counter(draws)
    total = len(draws)
    entropy = 0.0
    
    for count in counter.values():
        p = count / total
        if p > 0:
            entropy -= p * (p ** 0.5)  # Simplified entropy measure
    
    return entropy


def create_historical_analysis_real(data_file: str, output_file: str, config: HistoricalAnalysisConfig = None) -> str:
    """
    Execute comprehensive historical analysis
    
    Args:
        data_file: Path to lottery data JSON file
        output_file: Output path (used for run_id generation)
        
    Returns:
        search_id: Unique identifier for this analysis
    """
    
    # Initialize config with defaults if not provided
    if config is None:
        config = HistoricalAnalysisConfig(data_file=data_file)

    # Load data
    try:
        with open(data_file, 'r') as f:
            lottery_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load data file: {e}")
    
    # Extract draws and validate
    draws = [entry.get('draw') for entry in lottery_data if entry.get('draw') is not None]
    if not draws:
        raise ValueError("No draw data found in lottery file")
    
    # Generate unique run ID
    timestamp = int(datetime.now().timestamp())
    run_id = f"historical_analysis_{timestamp}"
    
    print(f"\nAnalyzing {len(draws)} historical draws...")
    
    # Perform all analyses
    frequency_analysis = analyze_draw_frequency(draws, config.top_n, config.bottom_n) if config.enable_frequency else {}
    gap_analysis = analyze_gaps(draws) if config.enable_gaps else {}
    streak_analysis = analyze_streaks(draws, config.streak_threshold) if config.enable_streaks else {}
    temporal_analysis = analyze_temporal_patterns(lottery_data) if config.enable_temporal else {}
    entropy_score = calculate_entropy(draws)
    
    # Overall statistics
    overall_stats = {
        'total_draws': len(draws),
        'unique_numbers': len(set(draws)),
        'mean': statistics.mean(draws),
        'median': statistics.median(draws),
        'mode': statistics.mode(draws) if draws else None,
        'std_dev': statistics.stdev(draws) if len(draws) > 1 else 0,
        'min_value': min(draws),
        'max_value': max(draws),
        'entropy': entropy_score
    }
    
    # Format results for results_manager
    formatted_results = {
        'run_metadata': {
            'run_id': run_id,
            'analysis_type': 'historical_analysis',
            'timestamp_start': datetime.now().isoformat(),
            'data_file': data_file,
            'schema_version': '1.0.2'
        },
        'analysis_parameters': {
            'top_n': config.top_n,
            'bottom_n': config.bottom_n,
            'streak_threshold': config.streak_threshold,
            'entropy_window': config.entropy_window,
            'enable_frequency': config.enable_frequency,
            'enable_gaps': config.enable_gaps,
            'enable_streaks': config.enable_streaks,
            'enable_temporal': config.enable_temporal
        },
        'results_summary': {
            'total_draws_analyzed': len(draws),
            'unique_numbers': len(set(draws)),
            'date_range': {
                'earliest': lottery_data[0].get('date') if lottery_data else None,
                'latest': lottery_data[-1].get('date') if lottery_data else None
            },
            'most_frequent_number': frequency_analysis['most_common'][0] if frequency_analysis['most_common'] else None,
            'least_frequent_number': frequency_analysis['least_common'][0] if frequency_analysis['least_common'] else None,
            'entropy_score': entropy_score
        },
        'overall_statistics': overall_stats,
        'frequency_analysis': {
            'total_draws': frequency_analysis['total_draws'],
            'unique_values': frequency_analysis['unique_values'],
            'top_10_most_common': frequency_analysis['most_common'],
            'top_10_least_common': frequency_analysis['least_common'],
            'average_frequency': frequency_analysis['average_frequency']
        },
        'gap_analysis': {
            'numbers_analyzed': len(gap_analysis),
            'sample_gaps': dict(list(gap_analysis.items())[:10])  # Top 10 for brevity
        },
        'streak_analysis': {
            'numbers_with_streaks': len(streak_analysis),
            'longest_streaks': sorted(
                [(num, data['max_streak']) for num, data in streak_analysis.items()],
                key=lambda x: x[1],
                reverse=True
            )[:10]
        },
        'temporal_patterns': temporal_analysis
    }
    
    # Save using new results system
    try:
        manager = ResultsManager()
        
        # Save in new format - results/summaries/, results/csv/, results/json/
        paths = manager.save_results(
            analysis_type='historical_analysis',
            run_id=run_id,
            data=formatted_results
        )
        
        print(f"\n✅ Analysis complete!")
        print(f"   Summary: {paths.get('summary')}")
        print(f"   CSV: {paths.get('csv')}")
        print(f"   JSON: {paths.get('json')}")
        
    except Exception as e:
        # Fallback: save to results/json/ manually if results_manager fails
        print(f"⚠️  Results manager unavailable, using fallback save: {e}")
        fallback_path = Path("results/json") / f"{run_id}.json"
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(fallback_path, 'w') as f:
            json.dump(formatted_results, f, indent=2)
        
        print(f"   Results saved to: {fallback_path}")
    
    return run_id


if __name__ == "__main__":
    # Test execution
    import argparse
    
    parser = argparse.ArgumentParser(description='Historical Pattern Analysis')
    parser.add_argument('--data-file', default='daily3.json', help='Lottery data file')
    parser.add_argument('--output', default='historical_analysis.txt', help='Output file')
    
    args = parser.parse_args()
    
    try:
        search_id = create_historical_analysis_real(args.data_file, args.output)
        print(f"\nAnalysis ID: {search_id}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
