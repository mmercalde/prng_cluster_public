#!/usr/bin/env python3
"""
PRNG Sweep Results Analyzer
============================

Quick analysis tool for interpreting sweep results.

Usage:
    python3 prng_results_analyzer.py prng_sweep_results/sweep_summary.json
"""

import json
import sys
from typing import List, Dict, Any
from collections import defaultdict


def load_results(filepath: str) -> Dict[str, Any]:
    """Load sweep summary JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_bidirectional_survivors(results: Dict[str, Any]):
    """Analyze bidirectional survivor counts"""
    print("\n" + "="*70)
    print("📊 BIDIRECTIONAL SURVIVOR ANALYSIS")
    print("="*70)
    
    prng_results = results['prng_results']
    
    # Group by bidirectional count
    by_count = defaultdict(list)
    for prng_name, data in prng_results.items():
        count = data.get('bidirectional_survivors', 0)
        by_count[count].append(prng_name)
    
    # Print distribution
    print("\nDistribution:")
    for count in sorted(by_count.keys()):
        prngs = by_count[count]
        print(f"  {count:5} survivors: {len(prngs):2} PRNGs - {', '.join(prngs[:3])}{' ...' if len(prngs) > 3 else ''}")
    
    # Identify winners (< 10 survivors)
    winners = []
    for prng_name, data in prng_results.items():
        count = data.get('bidirectional_survivors', 0)
        if count > 0 and count < 10:
            winners.append((prng_name, count, data))
    
    if winners:
        print(f"\n🎯 POTENTIAL WINNERS ({len(winners)} PRNGs with <10 bidirectional survivors):")
        winners.sort(key=lambda x: x[1])
        for prng_name, count, data in winners:
            fwd_match = data.get('forward_top_match_rate', 0) * 100
            rev_match = data.get('reverse_top_match_rate', 0) * 100
            quality = data.get('intersection_quality', 0) * 100
            print(f"  • {prng_name:20} - {count} survivors")
            print(f"    Forward match: {fwd_match:5.2f}%, Reverse match: {rev_match:5.2f}%, Quality: {quality:5.2f}%")
    else:
        print("\n⚠️  No clear winners (<10 bidirectional survivors)")


def analyze_match_rates(results: Dict[str, Any]):
    """Analyze match rate performance"""
    print("\n" + "="*70)
    print("🎯 MATCH RATE ANALYSIS")
    print("="*70)
    
    prng_results = results['prng_results']
    
    # Top by forward match rate
    forward_ranks = []
    reverse_ranks = []
    
    for prng_name, data in prng_results.items():
        fwd = data.get('forward_top_match_rate', 0)
        rev = data.get('reverse_top_match_rate', 0)
        if fwd > 0:
            forward_ranks.append((prng_name, fwd, data))
        if rev > 0:
            reverse_ranks.append((prng_name, rev, data))
    
    forward_ranks.sort(key=lambda x: x[1], reverse=True)
    reverse_ranks.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 - Forward Sieve Match Rates:")
    for i, (prng_name, match_rate, data) in enumerate(forward_ranks[:5], 1):
        surv = data.get('forward_survivors', 0)
        print(f"  {i}. {prng_name:20} - {match_rate*100:5.2f}% ({surv:,} survivors)")
    
    print("\nTop 5 - Reverse Sieve Match Rates:")
    for i, (prng_name, match_rate, data) in enumerate(reverse_ranks[:5], 1):
        surv = data.get('reverse_survivors', 0)
        print(f"  {i}. {prng_name:20} - {match_rate*100:5.2f}% ({surv:,} survivors)")


def analyze_efficiency(results: Dict[str, Any]):
    """Analyze computational efficiency"""
    print("\n" + "="*70)
    print("⚡ EFFICIENCY ANALYSIS")
    print("="*70)
    
    prng_results = results['prng_results']
    
    # Sort by total duration
    by_time = []
    for prng_name, data in prng_results.items():
        duration = data.get('duration_total', 0)
        if duration > 0:
            by_time.append((prng_name, duration))
    
    by_time.sort(key=lambda x: x[1])
    
    print("\nFastest 5 PRNGs:")
    for i, (prng_name, duration) in enumerate(by_time[:5], 1):
        print(f"  {i}. {prng_name:20} - {duration:.1f}s ({duration/60:.1f}min)")
    
    print("\nSlowest 5 PRNGs:")
    for i, (prng_name, duration) in enumerate(by_time[-5:], 1):
        print(f"  {i}. {prng_name:20} - {duration:.1f}s ({duration/60:.1f}min)")
    
    total_time = sum(d for _, d in by_time)
    avg_time = total_time / len(by_time)
    print(f"\nTotal sweep time: {total_time/3600:.1f} hours")
    print(f"Average per PRNG: {avg_time/60:.1f} minutes")


def generate_recommendations(results: Dict[str, Any]):
    """Generate actionable recommendations"""
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)
    
    prng_results = results['prng_results']
    
    # Find best PRNGs
    excellent = []
    good = []
    poor = []
    
    for prng_name, data in prng_results.items():
        bi_surv = data.get('bidirectional_survivors', 0)
        fwd_match = data.get('forward_top_match_rate', 0)
        rev_match = data.get('reverse_top_match_rate', 0)
        quality = data.get('intersection_quality', 0)
        
        if bi_surv > 0:
            score = (fwd_match + rev_match) / 2 * quality
            
            if bi_surv < 10 and (fwd_match > 0.5 or rev_match > 0.5):
                excellent.append((prng_name, bi_surv, score))
            elif bi_surv < 50 and (fwd_match > 0.2 or rev_match > 0.2):
                good.append((prng_name, bi_surv, score))
            else:
                poor.append((prng_name, bi_surv, score))
    
    excellent.sort(key=lambda x: (x[1], -x[2]))
    good.sort(key=lambda x: (x[1], -x[2]))
    
    if excellent:
        print("\n🌟 EXCELLENT CANDIDATES (High confidence):")
        for prng_name, bi_surv, score in excellent:
            print(f"  ✅ {prng_name}")
            print(f"     → {bi_surv} bidirectional survivors")
            print(f"     → Run GPU validation on these seeds")
            print(f"     → Extract ML features")
            print(f"     → Build prediction pool")
    
    if good:
        print("\n👍 GOOD CANDIDATES (Worth investigating):")
        for prng_name, bi_surv, score in good[:5]:
            print(f"  ⚠️  {prng_name}")
            print(f"     → {bi_surv} bidirectional survivors")
            print(f"     → Test hybrid (variable skip) variant")
            print(f"     → Try stricter threshold")
    
    if not excellent and not good:
        print("\n❌ NO STRONG CANDIDATES FOUND")
        print("\nSuggested next steps:")
        print("  1. Re-run with stricter threshold (--threshold 0.05)")
        print("  2. Test hybrid (variable skip) variants")
        print("  3. Test session splitting (--session midday/evening)")
        print("  4. Consider chained PRNG architectures")
    
    # Specific recommendations
    print("\n📋 NEXT ACTIONS:")
    if excellent:
        top = excellent[0][0]
        print(f"  1. Validate {top} seeds with GPU validator")
        print(f"  2. Compare top 3 PRNG candidates")
        print(f"  3. Extract 3-lane coherence features")
        print(f"  4. Begin ML reinforcement training")
    else:
        print(f"  1. Analyze threshold sensitivity")
        print(f"  2. Test hybrid variants of top 10 PRNGs")
        print(f"  3. Investigate session-specific patterns")
        print(f"  4. Review whitepapers for functional mimicry approach")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 prng_results_analyzer.py <sweep_summary.json>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    print("="*70)
    print("PRNG SWEEP RESULTS ANALYZER")
    print("="*70)
    print(f"Loading: {results_file}")
    
    try:
        results = load_results(results_file)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {results_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON file: {results_file}")
        sys.exit(1)
    
    # Metadata
    metadata = results.get('sweep_metadata', {})
    print(f"Timestamp: {metadata.get('timestamp', 'unknown')}")
    print(f"PRNGs tested: {metadata.get('total_prngs_tested', 0)}")
    print(f"Total duration: {metadata.get('total_duration_seconds', 0)/3600:.1f} hours")
    
    # Run analyses
    analyze_bidirectional_survivors(results)
    analyze_match_rates(results)
    analyze_efficiency(results)
    generate_recommendations(results)
    
    print("\n" + "="*70)
    print("✅ Analysis complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
