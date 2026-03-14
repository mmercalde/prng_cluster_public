#!/usr/bin/env python3
"""
apply_s142_wal_mode.py
S142 — WAL mode fix: SQLite concurrent write contention in NP2 partition workers

Problem:
  Multiple forked partition processes call sqlite3.connect(prng_analysis.db)
  simultaneously and attempt writes. SQLite's default rollback journal mode
  allows only one writer at a time — concurrent writers get OperationalError:
  database is locked, silently dropped, leaving ~half trial history rows missing.

Fix:
  Enable WAL (Write-Ahead Logging) journal mode in init_database().
  WAL is a per-file sticky setting — set once, all subsequent connections
  (in any process) automatically use WAL. Writers don't block readers and
  concurrent writers serialise cleanly without errors.

  One line added to database_system.py:init_database() immediately after
  the sqlite3.connect() context manager opens.

File: database_system.py
"""

import shutil
import os
import sys

BASE = os.path.expanduser("~/distributed_prng_analysis")
TARGET = os.path.join(BASE, "database_system.py")

OLD = """    def init_database(self):
        \"\"\"Initialize database tables\"\"\"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''"""

NEW = """    def init_database(self):
        \"\"\"Initialize database tables\"\"\"
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # [S142] WAL: concurrent NP2 writers
            conn.execute('''"""


def main():
    print("=" * 60)
    print("S142 — WAL mode fix: database_system.py")
    print("=" * 60)

    with open(TARGET, 'r') as f:
        src = f.read()

    if OLD not in src:
        print("ERROR: anchor not found.")
        print("First 80 chars:", repr(OLD[:80]))
        sys.exit(1)

    count = src.count(OLD)
    if count > 1:
        print(f"ERROR: anchor appears {count} times — must be unique")
        sys.exit(1)

    # Backup
    bak = TARGET + ".bak_s142_wal"
    if not os.path.exists(bak):
        shutil.copy2(TARGET, bak)
        print(f"Backup: {bak}")
    else:
        print(f"Backup already exists: {bak}")

    patched = src.replace(OLD, NEW, 1)
    with open(TARGET, 'w') as f:
        f.write(patched)

    lines_before = src.count('\n')
    lines_after  = patched.count('\n')
    print(f"Patch applied: {lines_before} → {lines_after} lines (+1)")

    with open(TARGET) as f:
        n = sum(1 for _ in f)
    print(f"Final line count: {n}")

    # Verify WAL pragma is present
    with open(TARGET) as f:
        content = f.read()
    if "PRAGMA journal_mode=WAL" in content:
        print("WAL pragma verified in file ✅")
    else:
        print("ERROR: WAL pragma not found after patch")
        sys.exit(1)

    print("\n✅ Patch complete.")
    print("\nVerification — check WAL is active after next DB connection:")
    print("  ssh rzeus \"cd ~/distributed_prng_analysis && source ~/venvs/torch/bin/activate && \\")
    print("    PYTHONPATH=. python3 -c \\\"")
    print("import sqlite3; conn = sqlite3.connect('prng_analysis.db')")
    print("mode = conn.execute('PRAGMA journal_mode').fetchone()[0]")
    print("print(f'Journal mode: {mode} (should be wal)')\"\"")


if __name__ == "__main__":
    main()
