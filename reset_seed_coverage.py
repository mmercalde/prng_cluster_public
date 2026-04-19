#!/usr/bin/env python3
"""
reset_seed_coverage.py
Reset the seed coverage tracker for one or all PRNG types.

Usage:
  python3 reset_seed_coverage.py                    # reset all PRNG types
  python3 reset_seed_coverage.py java_lcg           # reset java_lcg only
  python3 reset_seed_coverage.py --show             # show current coverage only

The seed coverage tracker lives in the SQLite database (exhaustive_progress table).
It tracks MAX(seed_range_end) per prng_type so the WATCHER knows where to resume.
Resetting sets seed_start back to 0 for the next run.
"""
import sys
import sqlite3
from pathlib import Path

# Find DB path same way database_system.py does
DB_PATH = Path(__file__).parent / "prng_analysis.db"

def show_coverage(conn):
    cur = conn.cursor()
    cur.execute("""
        SELECT prng_type, MAX(seed_range_end) as max_end, COUNT(*) as rows
        FROM exhaustive_progress
        GROUP BY prng_type
        ORDER BY prng_type
    """)
    rows = cur.fetchall()
    if not rows:
        print("  (no coverage recorded — seed_start will be 0)")
    else:
        print(f"  {'PRNG Type':<20} {'Max seed_end':>20} {'Rows':>6}")
        print(f"  {'-'*20} {'-'*20} {'-'*6}")
        for prng_type, max_end, count in rows:
            print(f"  {prng_type:<20} {max_end:>20,} {count:>6}")

def main():
    args = sys.argv[1:]

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))

    print(f"Database: {DB_PATH}")
    print("\nCurrent coverage:")
    show_coverage(conn)

    if "--show" in args or not args or (len(args) == 1 and args[0] == "--show"):
        if "--show" in args:
            conn.close()
            return

    # Determine what to reset
    prng_filter = [a for a in args if not a.startswith("--")]
    cur = conn.cursor()

    if prng_filter:
        for prng_type in prng_filter:
            cur.execute("DELETE FROM exhaustive_progress WHERE prng_type=?", (prng_type,))
            print(f"\nReset: {prng_type} ({cur.rowcount} rows deleted)")
    else:
        cur.execute("DELETE FROM exhaustive_progress")
        print(f"\nReset ALL coverage ({cur.rowcount} rows deleted)")

    conn.commit()

    print("\nCoverage after reset:")
    show_coverage(conn)
    print("\nNext run will start from seed_start=0")
    conn.close()

if __name__ == "__main__":
    main()
