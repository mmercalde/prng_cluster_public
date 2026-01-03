#!/usr/bin/env python3
"""
Fingerprint Registry - SQLite Backend for WATCHER Loop Prevention
===================================================================

PURPOSE:
    Persistent storage for pipeline execution attempts. Enables WATCHER to:
    - Track which (fingerprint, prng_type) combinations have been tried
    - Prevent infinite retry loops on known-degenerate configurations
    - Detect when a data window has failed multiple PRNG hypotheses

ARCHITECTURE:
    - Fingerprint = data context identity (window + survivors, NOT prng_type)
    - PRNG attempts tracked orthogonally per fingerprint
    - SQLite provides ACID safety, concurrent access, fast queries

APPROVED BY:
    - Team Beta (January 2, 2026)
    - Proposal: WATCHER Fingerprint Registry v1.0.0
    - Storage Memo: SQLite vs JSON

VERSION: 1.0.0
DATE: January 2, 2026
"""

import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RegistryAttempt:
    """Single attempt record."""
    fingerprint: str
    prng_type: str
    outcome: str
    signal_confidence: float
    exit_code: int
    timestamp: str
    run_id: str
    duration_seconds: int
    holdout_draws: int
    training_draws: int
    policy_version: str = "1.0.0"
    watcher_version: str = "1.0.0"


@dataclass 
class RegistryEntry:
    """Summary of all attempts for a fingerprint."""
    fingerprint: str
    first_seen: str
    last_seen: str
    total_attempts: int
    total_failures: int
    prng_types_tried: List[str]
    last_outcome: str
    last_prng_type: str


# =============================================================================
# FINGERPRINT REGISTRY CLASS
# =============================================================================

class FingerprintRegistry:
    """
    SQLite-based registry for tracking pipeline execution attempts.
    
    Usage:
        registry = FingerprintRegistry()
        
        # Before running pipeline
        if registry.is_combination_tried(fingerprint, prng_type):
            print("Skip - already tried")
        
        # After pipeline completes
        registry.record_attempt(fingerprint, prng_type, outcome, sidecar)
        
        # Check exhaustion
        untried = registry.get_untried_prngs(fingerprint, all_prngs)
        if not untried:
            print("All PRNGs exhausted for this fingerprint")
    """
    
    # Schema version for migrations
    SCHEMA_VERSION = "1.0.0"
    
    def __init__(self, path: str = "agents/data/fingerprint_registry.db"):
        """
        Initialize registry with SQLite backend.
        
        Args:
            path: Path to SQLite database file
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        
        self._conn = None
        self._init_database()
        
        logger.info(f"FingerprintRegistry initialized: {self.path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            
            # Enable WAL mode for durability + concurrency
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            
        return self._conn
    
    def _init_database(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        
        # Main attempts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                prng_type TEXT NOT NULL,
                outcome TEXT NOT NULL,
                signal_confidence REAL,
                exit_code INTEGER,
                timestamp TEXT NOT NULL,
                run_id TEXT,
                duration_seconds INTEGER,
                holdout_draws INTEGER,
                training_draws INTEGER,
                policy_version TEXT,
                watcher_version TEXT,
                
                UNIQUE(fingerprint, prng_type)
            )
        """)
        
        # Fingerprint summary table (denormalized for fast lookup)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fingerprint_summary (
                fingerprint TEXT PRIMARY KEY,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                total_attempts INTEGER DEFAULT 0,
                total_failures INTEGER DEFAULT 0,
                last_outcome TEXT,
                last_prng_type TEXT
            )
        """)
        
        # Runs table for batch tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                policy_version TEXT,
                watcher_version TEXT,
                fingerprint TEXT,
                prng_type TEXT,
                outcome TEXT
            )
        """)
        
        # Schema metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        # Store schema version
        conn.execute("""
            INSERT OR REPLACE INTO schema_meta (key, value)
            VALUES ('schema_version', ?)
        """, (self.SCHEMA_VERSION,))
        
        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_fingerprint ON attempts(fingerprint)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_timestamp ON attempts(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_outcome ON attempts(outcome)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_prng ON attempts(prng_type)")
        
        conn.commit()
        logger.debug("Database schema initialized")
    
    # =========================================================================
    # CORE OPERATIONS
    # =========================================================================
    
    def record_attempt(
        self,
        fingerprint: str,
        prng_type: str,
        outcome: str,
        sidecar: Optional[Dict] = None,
        run_id: Optional[str] = None,
        signal_confidence: float = 0.0,
        exit_code: int = 0,
        duration_seconds: int = 0,
        policy_version: str = "1.0.0",
        watcher_version: str = "1.0.0"
    ) -> bool:
        """
        Record a pipeline execution attempt.
        
        Args:
            fingerprint: Data context fingerprint (from sidecar)
            prng_type: PRNG hypothesis tested
            outcome: Result (SUCCESS, DEGENERATE_SIGNAL, SPARSE_SIGNAL, etc.)
            sidecar: Optional full sidecar dict for extracting metadata
            run_id: Unique run identifier
            signal_confidence: Signal confidence from Step 5
            exit_code: Process exit code
            duration_seconds: Execution time
            policy_version: WATCHER policy version
            watcher_version: WATCHER agent version
            
        Returns:
            True if recorded, False if duplicate (already exists)
        """
        conn = self._get_connection()
        timestamp = datetime.now().isoformat()
        
        # Extract additional info from sidecar if provided
        holdout_draws = 0
        training_draws = 0
        if sidecar:
            data_context = sidecar.get("data_context", {})
            holdout_draws = data_context.get("holdout_window", {}).get("draw_count", 0)
            training_draws = data_context.get("training_window", {}).get("draw_count", 0)
            
            signal_quality = sidecar.get("signal_quality", {})
            if signal_confidence == 0.0:
                signal_confidence = signal_quality.get("signal_confidence", 0.0)
            
            if not run_id:
                agent_meta = sidecar.get("agent_metadata", {})
                run_id = agent_meta.get("run_id", f"run_{timestamp}")
        
        if not run_id:
            run_id = f"run_{timestamp}"
        
        try:
            # Insert attempt (UNIQUE constraint prevents duplicates)
            conn.execute("""
                INSERT INTO attempts (
                    fingerprint, prng_type, outcome, signal_confidence,
                    exit_code, timestamp, run_id, duration_seconds,
                    holdout_draws, training_draws, policy_version, watcher_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fingerprint, prng_type, outcome, signal_confidence,
                exit_code, timestamp, run_id, duration_seconds,
                holdout_draws, training_draws, policy_version, watcher_version
            ))
            
            # Update or insert summary
            is_failure = outcome not in ("SUCCESS",)
            
            # Check if summary exists
            row = conn.execute(
                "SELECT * FROM fingerprint_summary WHERE fingerprint = ?",
                (fingerprint,)
            ).fetchone()
            
            if row:
                # Update existing
                conn.execute("""
                    UPDATE fingerprint_summary SET
                        last_seen = ?,
                        total_attempts = total_attempts + 1,
                        total_failures = total_failures + ?,
                        last_outcome = ?,
                        last_prng_type = ?
                    WHERE fingerprint = ?
                """, (timestamp, 1 if is_failure else 0, outcome, prng_type, fingerprint))
            else:
                # Insert new
                conn.execute("""
                    INSERT INTO fingerprint_summary (
                        fingerprint, first_seen, last_seen,
                        total_attempts, total_failures,
                        last_outcome, last_prng_type
                    ) VALUES (?, ?, ?, 1, ?, ?, ?)
                """, (fingerprint, timestamp, timestamp, 1 if is_failure else 0, outcome, prng_type))
            
            # Record run
            conn.execute("""
                INSERT OR REPLACE INTO runs (
                    run_id, started_at, completed_at,
                    policy_version, watcher_version,
                    fingerprint, prng_type, outcome
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (run_id, timestamp, timestamp, policy_version, watcher_version,
                  fingerprint, prng_type, outcome))
            
            conn.commit()
            logger.info(f"Recorded attempt: {fingerprint[:8]}+{prng_type} → {outcome}")
            return True
            
        except sqlite3.IntegrityError:
            # Duplicate - already tried this combination
            logger.warning(f"Duplicate attempt blocked: {fingerprint[:8]}+{prng_type}")
            return False
    
    def is_combination_tried(self, fingerprint: str, prng_type: str) -> bool:
        """
        Check if a (fingerprint, prng_type) combination has been attempted.
        
        Args:
            fingerprint: Data context fingerprint
            prng_type: PRNG hypothesis to check
            
        Returns:
            True if already attempted, False otherwise
        """
        conn = self._get_connection()
        row = conn.execute("""
            SELECT 1 FROM attempts
            WHERE fingerprint = ? AND prng_type = ?
            LIMIT 1
        """, (fingerprint, prng_type)).fetchone()
        
        return row is not None
    
    def get_entry(self, fingerprint: str) -> Optional[RegistryEntry]:
        """
        Get summary entry for a fingerprint.
        
        Args:
            fingerprint: Data context fingerprint
            
        Returns:
            RegistryEntry or None if not found
        """
        conn = self._get_connection()
        
        # Get summary
        row = conn.execute("""
            SELECT * FROM fingerprint_summary WHERE fingerprint = ?
        """, (fingerprint,)).fetchone()
        
        if not row:
            return None
        
        # Get list of tried PRNGs
        prngs = conn.execute("""
            SELECT prng_type FROM attempts WHERE fingerprint = ?
        """, (fingerprint,)).fetchall()
        
        prng_list = [r["prng_type"] for r in prngs]
        
        return RegistryEntry(
            fingerprint=row["fingerprint"],
            first_seen=row["first_seen"],
            last_seen=row["last_seen"],
            total_attempts=row["total_attempts"],
            total_failures=row["total_failures"],
            prng_types_tried=prng_list,
            last_outcome=row["last_outcome"],
            last_prng_type=row["last_prng_type"]
        )
    
    def get_untried_prngs(self, fingerprint: str, all_prngs: List[str]) -> List[str]:
        """
        Get list of PRNGs not yet tried for this fingerprint.
        
        Args:
            fingerprint: Data context fingerprint
            all_prngs: Complete list of available PRNGs
            
        Returns:
            List of untried PRNG types
        """
        conn = self._get_connection()
        
        tried = conn.execute("""
            SELECT prng_type FROM attempts WHERE fingerprint = ?
        """, (fingerprint,)).fetchall()
        
        tried_set = {r["prng_type"] for r in tried}
        
        return [p for p in all_prngs if p not in tried_set]
    
    def get_attempt_count(self, fingerprint: str) -> int:
        """Get number of attempts for a fingerprint."""
        conn = self._get_connection()
        row = conn.execute("""
            SELECT total_attempts FROM fingerprint_summary WHERE fingerprint = ?
        """, (fingerprint,)).fetchone()
        
        return row["total_attempts"] if row else 0
    
    def get_failure_count(self, fingerprint: str) -> int:
        """Get number of failures for a fingerprint."""
        conn = self._get_connection()
        row = conn.execute("""
            SELECT total_failures FROM fingerprint_summary WHERE fingerprint = ?
        """, (fingerprint,)).fetchone()
        
        return row["total_failures"] if row else 0
    
    # =========================================================================
    # MAINTENANCE OPERATIONS
    # =========================================================================
    
    def expire_old_entries(self, ttl_days: int = 7) -> int:
        """
        Remove entries older than TTL.
        
        Args:
            ttl_days: Time-to-live in days
            
        Returns:
            Number of entries removed
        """
        conn = self._get_connection()
        cutoff = (datetime.now() - timedelta(days=ttl_days)).isoformat()
        
        # Get fingerprints to expire
        expired = conn.execute("""
            SELECT fingerprint FROM fingerprint_summary WHERE last_seen < ?
        """, (cutoff,)).fetchall()
        
        expired_fingerprints = [r["fingerprint"] for r in expired]
        
        if not expired_fingerprints:
            return 0
        
        # Delete attempts
        placeholders = ",".join("?" * len(expired_fingerprints))
        conn.execute(f"""
            DELETE FROM attempts WHERE fingerprint IN ({placeholders})
        """, expired_fingerprints)
        
        # Delete summaries
        conn.execute(f"""
            DELETE FROM fingerprint_summary WHERE fingerprint IN ({placeholders})
        """, expired_fingerprints)
        
        conn.commit()
        
        logger.info(f"Expired {len(expired_fingerprints)} fingerprints (TTL={ttl_days} days)")
        return len(expired_fingerprints)
    
    def clear_fingerprint(self, fingerprint: str) -> bool:
        """
        Manually clear a fingerprint from registry (force retry).
        
        Args:
            fingerprint: Fingerprint to clear
            
        Returns:
            True if cleared, False if not found
        """
        conn = self._get_connection()
        
        cursor = conn.execute(
            "DELETE FROM attempts WHERE fingerprint = ?",
            (fingerprint,)
        )
        conn.execute(
            "DELETE FROM fingerprint_summary WHERE fingerprint = ?",
            (fingerprint,)
        )
        
        conn.commit()
        
        if cursor.rowcount > 0:
            logger.info(f"Cleared fingerprint: {fingerprint}")
            return True
        return False
    
    # =========================================================================
    # STATISTICS AND EXPORT
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get global registry statistics."""
        conn = self._get_connection()
        
        total_fingerprints = conn.execute(
            "SELECT COUNT(*) as cnt FROM fingerprint_summary"
        ).fetchone()["cnt"]
        
        total_attempts = conn.execute(
            "SELECT COUNT(*) as cnt FROM attempts"
        ).fetchone()["cnt"]
        
        total_successes = conn.execute(
            "SELECT COUNT(*) as cnt FROM attempts WHERE outcome = 'SUCCESS'"
        ).fetchone()["cnt"]
        
        total_failures = total_attempts - total_successes
        
        # Most common failure outcome
        top_failure = conn.execute("""
            SELECT outcome, COUNT(*) as cnt FROM attempts
            WHERE outcome != 'SUCCESS'
            GROUP BY outcome ORDER BY cnt DESC LIMIT 1
        """).fetchone()
        
        # Most tried fingerprint
        most_tried = conn.execute("""
            SELECT fingerprint, total_attempts FROM fingerprint_summary
            ORDER BY total_attempts DESC LIMIT 1
        """).fetchone()
        
        return {
            "schema_version": self.SCHEMA_VERSION,
            "total_fingerprints": total_fingerprints,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": total_successes / total_attempts if total_attempts > 0 else 0,
            "top_failure_outcome": top_failure["outcome"] if top_failure else None,
            "most_tried_fingerprint": most_tried["fingerprint"] if most_tried else None,
            "most_tried_attempts": most_tried["total_attempts"] if most_tried else 0
        }
    
    def export_to_json(self, output_path: str) -> None:
        """
        Export registry to JSON for audit/sharing.
        
        Args:
            output_path: Path for JSON export
        """
        conn = self._get_connection()
        
        # Get all fingerprints
        fingerprints = conn.execute(
            "SELECT * FROM fingerprint_summary"
        ).fetchall()
        
        export = {
            "schema_version": self.SCHEMA_VERSION,
            "exported_at": datetime.now().isoformat(),
            "stats": self.get_stats(),
            "entries": {}
        }
        
        for fp in fingerprints:
            fingerprint = fp["fingerprint"]
            
            # Get attempts for this fingerprint
            attempts = conn.execute("""
                SELECT * FROM attempts WHERE fingerprint = ? ORDER BY timestamp
            """, (fingerprint,)).fetchall()
            
            export["entries"][fingerprint] = {
                "first_seen": fp["first_seen"],
                "last_seen": fp["last_seen"],
                "total_attempts": fp["total_attempts"],
                "total_failures": fp["total_failures"],
                "last_outcome": fp["last_outcome"],
                "last_prng_type": fp["last_prng_type"],
                "attempts": [dict(a) for a in attempts]
            }
        
        with open(output_path, "w") as f:
            json.dump(export, f, indent=2)
        
        logger.info(f"Exported registry to {output_path}")
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_registry(path: str = "agents/data/fingerprint_registry.db") -> FingerprintRegistry:
    """Get or create a registry instance."""
    return FingerprintRegistry(path)


# =============================================================================
# CLI INTERFACE (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fingerprint Registry CLI")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--export", type=str, help="Export to JSON file")
    parser.add_argument("--check", nargs=2, metavar=("FINGERPRINT", "PRNG"), 
                        help="Check if combination tried")
    parser.add_argument("--clear", type=str, metavar="FINGERPRINT",
                        help="Clear a fingerprint")
    parser.add_argument("--expire", type=int, metavar="DAYS",
                        help="Expire entries older than N days")
    parser.add_argument("--db", type=str, default="agents/data/fingerprint_registry.db",
                        help="Database path")
    
    args = parser.parse_args()
    
    # Configure logging for CLI
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    registry = FingerprintRegistry(args.db)
    
    if args.stats:
        stats = registry.get_stats()
        print("\n╔════════════════════════════════════════════════════════════════╗")
        print("║              FINGERPRINT REGISTRY STATISTICS                    ║")
        print("╠════════════════════════════════════════════════════════════════╣")
        print(f"║ Schema Version:      {stats['schema_version']:<40} ║")
        print(f"║ Total Fingerprints:  {stats['total_fingerprints']:<40} ║")
        print(f"║ Total Attempts:      {stats['total_attempts']:<40} ║")
        print(f"║ Total Successes:     {stats['total_successes']:<40} ║")
        print(f"║ Total Failures:      {stats['total_failures']:<40} ║")
        print(f"║ Success Rate:        {stats['success_rate']*100:.1f}%{'':<37} ║")
        print("╚════════════════════════════════════════════════════════════════╝")
    
    elif args.export:
        registry.export_to_json(args.export)
        print(f"Exported to {args.export}")
    
    elif args.check:
        fingerprint, prng = args.check
        tried = registry.is_combination_tried(fingerprint, prng)
        print(f"{'TRIED' if tried else 'NOT TRIED'}: {fingerprint[:8]}+{prng}")
    
    elif args.clear:
        cleared = registry.clear_fingerprint(args.clear)
        print(f"{'Cleared' if cleared else 'Not found'}: {args.clear}")
    
    elif args.expire:
        count = registry.expire_old_entries(args.expire)
        print(f"Expired {count} entries")
    
    else:
        parser.print_help()
    
    registry.close()
