#!/usr/bin/env python3
"""
Registry Inspector - CLI Tool for Fingerprint Registry Analysis
================================================================

PURPOSE:
    Human-friendly inspection and management of the WATCHER fingerprint registry.
    
USAGE:
    python3 registry_inspector.py --fingerprint f8b09677
    python3 registry_inspector.py --list
    python3 registry_inspector.py --stats
    python3 registry_inspector.py --export registry_dump.json
    python3 registry_inspector.py --prng-summary
    
VERSION: 1.0.0
DATE: January 2, 2026
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fingerprint_registry import FingerprintRegistry


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_header(title: str, width: int = 70) -> None:
    """Print formatted header."""
    print(f"\n╔{'═' * (width-2)}╗")
    print(f"║ {title:<{width-4}} ║")
    print(f"╠{'═' * (width-2)}╣")


def print_footer(width: int = 70) -> None:
    """Print formatted footer."""
    print(f"╚{'═' * (width-2)}╝")


def print_row(key: str, value: str, width: int = 70) -> None:
    """Print formatted row."""
    content = f"{key}: {value}"
    print(f"║ {content:<{width-4}} ║")


def print_separator(width: int = 70) -> None:
    """Print separator line."""
    print(f"╠{'─' * (width-2)}╣")


# =============================================================================
# INSPECTION COMMANDS
# =============================================================================

def cmd_fingerprint(registry: FingerprintRegistry, fingerprint: str, all_prngs: list) -> None:
    """Display detailed info for a fingerprint."""
    entry = registry.get_entry(fingerprint)
    
    if not entry:
        print(f"\n❌ Fingerprint not found: {fingerprint}")
        return
    
    print_header(f"FINGERPRINT: {fingerprint}")
    print_row("First Seen", entry.first_seen)
    print_row("Last Seen", entry.last_seen)
    print_row("Total Attempts", str(entry.total_attempts))
    print_row("Total Failures", str(entry.total_failures))
    print_row("Last Outcome", entry.last_outcome)
    print_row("Last PRNG", entry.last_prng_type)
    
    print_separator()
    print(f"║ {'ATTEMPTS':<66} ║")
    print_separator()
    
    # Get detailed attempts from database
    conn = registry._get_connection()
    attempts = conn.execute("""
        SELECT prng_type, outcome, timestamp, duration_seconds
        FROM attempts WHERE fingerprint = ?
        ORDER BY timestamp
    """, (fingerprint,)).fetchall()
    
    for i, att in enumerate(attempts, 1):
        outcome_icon = "✅" if att["outcome"] == "SUCCESS" else "❌"
        time_str = att["timestamp"][:19] if att["timestamp"] else "?"
        duration = f"{att['duration_seconds']}s" if att["duration_seconds"] else "?"
        print(f"║ #{i:<2} │ {att['prng_type']:<25} │ {outcome_icon} {att['outcome']:<18} │ {duration:<6} ║")
    
    print_separator()
    print(f"║ {'PRNG STATUS':<66} ║")
    print_separator()
    
    tried = set(entry.prng_types_tried)
    untried = [p for p in all_prngs if p not in tried]
    
    print(f"║ Tried ({len(tried)}): {', '.join(sorted(tried)[:5])}{'...' if len(tried) > 5 else '':<30} ║")
    print(f"║ Remaining ({len(untried)}): {', '.join(untried[:5])}{'...' if len(untried) > 5 else '':<25} ║")
    
    # Recommendation
    print_separator()
    if untried:
        print(f"║ 💡 RECOMMENDATION: Try {untried[0]} next{'':<36} ║")
    else:
        print(f"║ ⚠️  ALL PRNGs EXHAUSTED - Consider REJECT_DATA_WINDOW{'':<14} ║")
    
    print_footer()


def cmd_list(registry: FingerprintRegistry, limit: int = 20) -> None:
    """List all fingerprints in registry."""
    conn = registry._get_connection()
    
    rows = conn.execute("""
        SELECT fingerprint, total_attempts, total_failures, last_outcome, last_seen
        FROM fingerprint_summary
        ORDER BY last_seen DESC
        LIMIT ?
    """, (limit,)).fetchall()
    
    if not rows:
        print("\n📭 Registry is empty")
        return
    
    print_header(f"FINGERPRINT REGISTRY ({len(rows)} entries)")
    print(f"║ {'Fingerprint':<12} │ {'Attempts':<8} │ {'Failures':<8} │ {'Last Outcome':<18} │ {'Last Seen':<16} ║")
    print_separator()
    
    for row in rows:
        fp = row["fingerprint"][:10]
        att = str(row["total_attempts"])
        fail = str(row["total_failures"])
        outcome = row["last_outcome"][:16] if row["last_outcome"] else "?"
        seen = row["last_seen"][:16] if row["last_seen"] else "?"
        print(f"║ {fp:<12} │ {att:<8} │ {fail:<8} │ {outcome:<18} │ {seen:<16} ║")
    
    print_footer()


def cmd_stats(registry: FingerprintRegistry) -> None:
    """Display registry statistics."""
    stats = registry.get_stats()
    
    print_header("FINGERPRINT REGISTRY STATISTICS")
    print_row("Schema Version", stats["schema_version"])
    print_row("Total Fingerprints", str(stats["total_fingerprints"]))
    print_row("Total Attempts", str(stats["total_attempts"]))
    print_row("Total Successes", str(stats["total_successes"]))
    print_row("Total Failures", str(stats["total_failures"]))
    print_row("Success Rate", f"{stats['success_rate']*100:.1f}%")
    print_separator()
    print_row("Top Failure Type", stats["top_failure_outcome"] or "N/A")
    print_row("Most Tried Fingerprint", stats["most_tried_fingerprint"] or "N/A")
    print_row("Most Tried Attempts", str(stats["most_tried_attempts"]))
    print_footer()


def cmd_prng_summary(registry: FingerprintRegistry) -> None:
    """Show PRNG-level statistics."""
    conn = registry._get_connection()
    
    rows = conn.execute("""
        SELECT prng_type, 
               COUNT(*) as attempts,
               SUM(CASE WHEN outcome = 'SUCCESS' THEN 1 ELSE 0 END) as successes,
               SUM(CASE WHEN outcome != 'SUCCESS' THEN 1 ELSE 0 END) as failures
        FROM attempts
        GROUP BY prng_type
        ORDER BY attempts DESC
    """).fetchall()
    
    if not rows:
        print("\n📭 No attempts recorded")
        return
    
    print_header("PRNG ATTEMPT SUMMARY")
    print(f"║ {'PRNG Type':<28} │ {'Attempts':<8} │ {'Success':<8} │ {'Fail':<8} │ {'Rate':<8} ║")
    print_separator()
    
    for row in rows:
        prng = row["prng_type"][:26]
        att = str(row["attempts"])
        succ = str(row["successes"])
        fail = str(row["failures"])
        rate = f"{row['successes']/row['attempts']*100:.0f}%" if row["attempts"] > 0 else "N/A"
        print(f"║ {prng:<28} │ {att:<8} │ {succ:<8} │ {fail:<8} │ {rate:<8} ║")
    
    print_footer()


def cmd_failures(registry: FingerprintRegistry, limit: int = 10) -> None:
    """Show recent failures."""
    conn = registry._get_connection()
    
    rows = conn.execute("""
        SELECT fingerprint, prng_type, outcome, timestamp, signal_confidence
        FROM attempts
        WHERE outcome != 'SUCCESS'
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    
    if not rows:
        print("\n✅ No failures recorded")
        return
    
    print_header(f"RECENT FAILURES (Last {len(rows)})")
    print(f"║ {'Fingerprint':<10} │ {'PRNG':<20} │ {'Outcome':<18} │ {'Confidence':<10} ║")
    print_separator()
    
    for row in rows:
        fp = row["fingerprint"][:8]
        prng = row["prng_type"][:18]
        outcome = row["outcome"][:16]
        conf = f"{row['signal_confidence']:.2f}" if row["signal_confidence"] else "N/A"
        print(f"║ {fp:<10} │ {prng:<20} │ {outcome:<18} │ {conf:<10} ║")
    
    print_footer()


def cmd_export(registry: FingerprintRegistry, output_path: str) -> None:
    """Export registry to JSON."""
    registry.export_to_json(output_path)
    print(f"\n✅ Exported to {output_path}")


def cmd_clear(registry: FingerprintRegistry, fingerprint: str) -> None:
    """Clear a fingerprint from registry."""
    print(f"\n⚠️  This will permanently remove fingerprint: {fingerprint}")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() == "yes":
        cleared = registry.clear_fingerprint(fingerprint)
        if cleared:
            print(f"✅ Cleared fingerprint: {fingerprint}")
        else:
            print(f"❌ Fingerprint not found: {fingerprint}")
    else:
        print("Cancelled")


def cmd_expire(registry: FingerprintRegistry, days: int) -> None:
    """Expire old entries."""
    count = registry.expire_old_entries(days)
    print(f"\n✅ Expired {count} entries older than {days} days")


# =============================================================================
# PRNG LIST (from policy)
# =============================================================================

DEFAULT_PRNGS = [
    "xorshift32", "xorshift32_hybrid", "xorshift32_reverse", "xorshift32_hybrid_reverse",
    "xorshift64", "xorshift64_hybrid", "xorshift64_reverse", "xorshift64_hybrid_reverse",
    "xorshift128", "xorshift128_hybrid", "xorshift128_reverse", "xorshift128_hybrid_reverse",
    "pcg32", "pcg32_hybrid", "pcg32_reverse", "pcg32_hybrid_reverse",
    "lcg32", "lcg32_hybrid", "lcg32_reverse", "lcg32_hybrid_reverse",
    "java_lcg", "java_lcg_hybrid", "java_lcg_reverse", "java_lcg_hybrid_reverse",
    "minstd", "minstd_hybrid", "minstd_reverse", "minstd_hybrid_reverse",
    "mt19937", "mt19937_hybrid", "mt19937_reverse", "mt19937_hybrid_reverse",
    "xoshiro256pp", "xoshiro256pp_hybrid", "xoshiro256pp_reverse", "xoshiro256pp_hybrid_reverse",
    "philox4x32", "philox4x32_hybrid", "philox4x32_reverse", "philox4x32_hybrid_reverse",
    "sfc64", "sfc64_hybrid", "sfc64_reverse", "sfc64_hybrid_reverse"
]


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fingerprint Registry Inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --stats                     Show registry statistics
  %(prog)s --list                      List all fingerprints
  %(prog)s --fingerprint f8b09677      Show details for fingerprint
  %(prog)s --prng-summary              Show PRNG-level statistics
  %(prog)s --failures                  Show recent failures
  %(prog)s --export dump.json          Export to JSON
  %(prog)s --clear f8b09677            Clear a fingerprint (force retry)
  %(prog)s --expire 7                  Expire entries older than 7 days
        """
    )
    
    parser.add_argument("--db", type=str, 
                        default="agents/data/fingerprint_registry.db",
                        help="Database path")
    
    # View commands
    parser.add_argument("--stats", action="store_true",
                        help="Show registry statistics")
    parser.add_argument("--list", action="store_true",
                        help="List all fingerprints")
    parser.add_argument("--fingerprint", "-f", type=str,
                        help="Show details for specific fingerprint")
    parser.add_argument("--prng-summary", action="store_true",
                        help="Show PRNG-level statistics")
    parser.add_argument("--failures", action="store_true",
                        help="Show recent failures")
    
    # Management commands
    parser.add_argument("--export", type=str, metavar="FILE",
                        help="Export registry to JSON file")
    parser.add_argument("--clear", type=str, metavar="FINGERPRINT",
                        help="Clear a fingerprint from registry")
    parser.add_argument("--expire", type=int, metavar="DAYS",
                        help="Expire entries older than N days")
    
    # Options
    parser.add_argument("--limit", type=int, default=20,
                        help="Limit for list commands (default: 20)")
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = FingerprintRegistry(args.db)
    
    try:
        if args.stats:
            cmd_stats(registry)
        elif args.list:
            cmd_list(registry, args.limit)
        elif args.fingerprint:
            cmd_fingerprint(registry, args.fingerprint, DEFAULT_PRNGS)
        elif args.prng_summary:
            cmd_prng_summary(registry)
        elif args.failures:
            cmd_failures(registry, args.limit)
        elif args.export:
            cmd_export(registry, args.export)
        elif args.clear:
            cmd_clear(registry, args.clear)
        elif args.expire:
            cmd_expire(registry, args.expire)
        else:
            parser.print_help()
    finally:
        registry.close()


if __name__ == "__main__":
    main()
