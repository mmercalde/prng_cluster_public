#!/usr/bin/env python3
"""
WATCHER Registry Hooks - Integration between Watcher Agent and Fingerprint Registry
======================================================================================

PURPOSE:
    Connects the Watcher Agent to the Fingerprint Registry for:
    - Pre-run checks: Prevent retrying known-failed combinations
    - Post-run recording: Log attempt outcomes for future decisions
    - Decision support: Recommend next actions based on history

USAGE:
    from watcher_registry_hooks import WatcherRegistryHooks
    
    hooks = WatcherRegistryHooks()
    
    # Before running pipeline
    decision = hooks.pre_run_check(fingerprint, prng_type)
    if decision.action == "SKIP_PRNG":
        next_prng = decision.suggested_prng
        
    # After pipeline completes
    hooks.post_run_record(fingerprint, prng_type, outcome, sidecar_path)

VERSION: 1.0.0
DATE: January 2, 2026
APPROVED BY: Team Beta
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

# Import the registry
import sys
sys.path.insert(0, str(Path(__file__).parent))
from fingerprint_registry import FingerprintRegistry

# Configure logging
logger = logging.getLogger("WatcherRegistryHooks")


# =============================================================================
# PRNG PRIORITY ORDER (from watcher_policy_v1.0.json)
# =============================================================================

PRNG_PRIORITY_ORDER = [
    # Forward fixed (most common)
    "java_lcg", "mt19937", "xorshift32", "pcg32", "lcg32",
    "minstd", "xorshift64", "xorshift128", "xoshiro256pp", "philox4x32", "sfc64",
    # Forward hybrid
    "java_lcg_hybrid", "mt19937_hybrid", "xorshift32_hybrid", "pcg32_hybrid", "lcg32_hybrid",
    "minstd_hybrid", "xorshift64_hybrid", "xorshift128_hybrid", "xoshiro256pp_hybrid", 
    "philox4x32_hybrid", "sfc64_hybrid",
    # Reverse fixed
    "java_lcg_reverse", "mt19937_reverse", "xorshift32_reverse", "pcg32_reverse", "lcg32_reverse",
    "minstd_reverse", "xorshift64_reverse", "xorshift128_reverse", "xoshiro256pp_reverse",
    "philox4x32_reverse", "sfc64_reverse",
    # Reverse hybrid
    "java_lcg_hybrid_reverse", "mt19937_hybrid_reverse", "xorshift32_hybrid_reverse",
    "pcg32_hybrid_reverse", "lcg32_hybrid_reverse", "minstd_hybrid_reverse",
    "xorshift64_hybrid_reverse", "xorshift128_hybrid_reverse", "xoshiro256pp_hybrid_reverse",
    "philox4x32_hybrid_reverse", "sfc64_hybrid_reverse"
]


# =============================================================================
# POLICY THRESHOLDS
# =============================================================================

@dataclass
class PolicyThresholds:
    """Thresholds from watcher_policy_v1.0.json"""
    MAX_PRNG_ATTEMPTS: int = 5
    MIN_HOLDOUT_DRAWS: int = 500
    MAX_TEMPORAL_GAP: int = 1000
    FINGERPRINT_TTL_DAYS: int = 7
    MIN_SIGNAL_CONFIDENCE: float = 0.3


# =============================================================================
# DECISION DATA CLASSES
# =============================================================================

@dataclass
class PreRunDecision:
    """Decision from pre-run check."""
    action: str  # PROCEED, SKIP_PRNG, REJECT_DATA_WINDOW, EXPAND_HOLDOUT
    reason: str
    fingerprint: str
    prng_type: str
    suggested_prng: Optional[str] = None
    attempt_count: int = 0
    untried_count: int = 0
    requires_human: bool = False


@dataclass
class PostRunResult:
    """Result from post-run recording."""
    recorded: bool
    fingerprint: str
    prng_type: str
    outcome: str
    total_attempts: int
    total_failures: int


# =============================================================================
# WATCHER REGISTRY HOOKS
# =============================================================================

class WatcherRegistryHooks:
    """
    Integration hooks between Watcher Agent and Fingerprint Registry.
    
    Implements the decision rules from watcher_policy_v1.0.json:
    - R1: Skip already-tried combinations
    - R2: Reject exhausted fingerprints
    - R3: Expand small holdouts
    - R4: Shift temporal gaps
    - R5: Halt when all PRNGs exhausted
    - R6: Proceed by default
    """
    
    def __init__(
        self, 
        db_path: str = "agents/data/fingerprint_registry.db",
        policy: PolicyThresholds = None
    ):
        """
        Initialize hooks with registry connection.
        
        Args:
            db_path: Path to SQLite registry database
            policy: Policy thresholds (uses defaults if not provided)
        """
        self.registry = FingerprintRegistry(db_path)
        self.policy = policy or PolicyThresholds()
        self.prng_order = PRNG_PRIORITY_ORDER
        
        logger.info(f"WatcherRegistryHooks initialized (db: {db_path})")
    
    # =========================================================================
    # PRE-RUN CHECK
    # =========================================================================
    
    def pre_run_check(
        self,
        fingerprint: str,
        prng_type: str,
        holdout_draws: Optional[int] = None,
        temporal_gap: Optional[int] = None
    ) -> PreRunDecision:
        """
        Check registry before running pipeline.
        
        Implements decision rules R1-R6 in priority order.
        
        Args:
            fingerprint: Data context fingerprint (v2_data_only)
            prng_type: PRNG hypothesis to test
            holdout_draws: Optional holdout window size (for R3)
            temporal_gap: Optional gap between windows (for R4)
            
        Returns:
            PreRunDecision with action and reasoning
        """
        # Get current state
        entry = self.registry.get_entry(fingerprint)
        untried = self.registry.get_untried_prngs(fingerprint, self.prng_order)
        
        # R1: Skip already-tried combination
        if self.registry.is_combination_tried(fingerprint, prng_type):
            suggested = untried[0] if untried else None
            
            return PreRunDecision(
                action="SKIP_PRNG",
                reason=f"Already tried {prng_type} on fingerprint {fingerprint[:8]}",
                fingerprint=fingerprint,
                prng_type=prng_type,
                suggested_prng=suggested,
                attempt_count=entry.total_attempts if entry else 0,
                untried_count=len(untried)
            )
        
        # R2: Reject exhausted fingerprint
        if entry and entry.total_failures >= self.policy.MAX_PRNG_ATTEMPTS:
            return PreRunDecision(
                action="REJECT_DATA_WINDOW",
                reason=f"Fingerprint {fingerprint[:8]} has failed {entry.total_failures} PRNGs (threshold: {self.policy.MAX_PRNG_ATTEMPTS})",
                fingerprint=fingerprint,
                prng_type=prng_type,
                attempt_count=entry.total_attempts,
                untried_count=len(untried),
                requires_human=True
            )
        
        # R3: Expand small holdout
        if holdout_draws and holdout_draws < self.policy.MIN_HOLDOUT_DRAWS:
            if entry and entry.last_outcome == "SPARSE_SIGNAL":
                return PreRunDecision(
                    action="EXPAND_HOLDOUT",
                    reason=f"Holdout too small ({holdout_draws} < {self.policy.MIN_HOLDOUT_DRAWS}) and previous SPARSE_SIGNAL",
                    fingerprint=fingerprint,
                    prng_type=prng_type,
                    attempt_count=entry.total_attempts if entry else 0,
                    untried_count=len(untried)
                )
        
        # R4: Shift temporal gap
        if temporal_gap and temporal_gap > self.policy.MAX_TEMPORAL_GAP:
            if entry and entry.last_outcome == "DEGENERATE_SIGNAL":
                return PreRunDecision(
                    action="SHIFT_WINDOW",
                    reason=f"Temporal gap too large ({temporal_gap} > {self.policy.MAX_TEMPORAL_GAP}) and previous DEGENERATE_SIGNAL",
                    fingerprint=fingerprint,
                    prng_type=prng_type,
                    attempt_count=entry.total_attempts if entry else 0,
                    untried_count=len(untried)
                )
        
        # R5: Halt if all PRNGs exhausted
        if entry and len(untried) == 0:
            return PreRunDecision(
                action="HALT_ALL_EXHAUSTED",
                reason=f"All {len(self.prng_order)} PRNGs exhausted for fingerprint {fingerprint[:8]}",
                fingerprint=fingerprint,
                prng_type=prng_type,
                attempt_count=entry.total_attempts,
                untried_count=0,
                requires_human=True
            )
        
        # R6: Proceed (default)
        return PreRunDecision(
            action="PROCEED",
            reason="No blocking conditions - ready to run",
            fingerprint=fingerprint,
            prng_type=prng_type,
            attempt_count=entry.total_attempts if entry else 0,
            untried_count=len(untried)
        )
    
    # =========================================================================
    # POST-RUN RECORD
    # =========================================================================
    
    def post_run_record(
        self,
        fingerprint: str,
        prng_type: str,
        outcome: str,
        sidecar_path: Optional[str] = None,
        exit_code: int = 0,
        signal_confidence: float = 0.0,
        duration_seconds: int = 0,
        run_id: Optional[str] = None
    ) -> PostRunResult:
        """
        Record pipeline attempt after completion.
        
        Args:
            fingerprint: Data context fingerprint
            prng_type: PRNG hypothesis tested
            outcome: Result (SUCCESS, DEGENERATE_SIGNAL, SPARSE_SIGNAL, etc.)
            sidecar_path: Optional path to sidecar JSON for metadata extraction
            exit_code: Process exit code
            signal_confidence: Signal confidence from Step 5
            duration_seconds: Execution time
            run_id: Optional run identifier
            
        Returns:
            PostRunResult with recording status
        """
        # Load sidecar if provided
        sidecar = None
        if sidecar_path and Path(sidecar_path).exists():
            try:
                with open(sidecar_path) as f:
                    sidecar = json.load(f)
                    
                # Extract values from sidecar
                signal_quality = sidecar.get("signal_quality", {})
                signal_confidence = signal_quality.get("signal_confidence", signal_confidence)
                
                agent_meta = sidecar.get("agent_metadata", {})
                if not run_id:
                    run_id = agent_meta.get("run_id")
                    
            except Exception as e:
                logger.warning(f"Failed to load sidecar: {e}")
        
        # Record the attempt
        recorded = self.registry.record_attempt(
            fingerprint=fingerprint,
            prng_type=prng_type,
            outcome=outcome,
            sidecar=sidecar,
            run_id=run_id,
            signal_confidence=signal_confidence,
            exit_code=exit_code,
            duration_seconds=duration_seconds
        )
        
        # Get updated stats
        entry = self.registry.get_entry(fingerprint)
        
        return PostRunResult(
            recorded=recorded,
            fingerprint=fingerprint,
            prng_type=prng_type,
            outcome=outcome,
            total_attempts=entry.total_attempts if entry else 1,
            total_failures=entry.total_failures if entry else (0 if outcome == "SUCCESS" else 1)
        )
    
    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================
    
    def get_next_prng(self, fingerprint: str) -> Optional[str]:
        """
        Get next untried PRNG in priority order.
        
        Args:
            fingerprint: Data context fingerprint
            
        Returns:
            Next PRNG to try, or None if all exhausted
        """
        untried = self.registry.get_untried_prngs(fingerprint, self.prng_order)
        return untried[0] if untried else None
    
    def get_fingerprint_status(self, fingerprint: str) -> Dict[str, Any]:
        """
        Get comprehensive status for a fingerprint.
        
        Args:
            fingerprint: Data context fingerprint
            
        Returns:
            Status dict with attempts, failures, recommendations
        """
        entry = self.registry.get_entry(fingerprint)
        untried = self.registry.get_untried_prngs(fingerprint, self.prng_order)
        
        if not entry:
            return {
                "fingerprint": fingerprint,
                "status": "NEW",
                "total_attempts": 0,
                "total_failures": 0,
                "prng_types_tried": [],
                "untried_count": len(self.prng_order),
                "next_prng": self.prng_order[0],
                "recommendation": "Ready for first attempt"
            }
        
        # Determine status
        if entry.total_failures >= self.policy.MAX_PRNG_ATTEMPTS:
            status = "EXHAUSTED"
            recommendation = "Consider rejecting this data window"
        elif len(untried) == 0:
            status = "ALL_TRIED"
            recommendation = "All PRNGs attempted - review results"
        elif entry.last_outcome == "SUCCESS":
            status = "SUCCESS"
            recommendation = "Previous attempt succeeded"
        else:
            status = "IN_PROGRESS"
            recommendation = f"Try {untried[0]} next"
        
        return {
            "fingerprint": fingerprint,
            "status": status,
            "total_attempts": entry.total_attempts,
            "total_failures": entry.total_failures,
            "prng_types_tried": entry.prng_types_tried,
            "untried_count": len(untried),
            "next_prng": untried[0] if untried else None,
            "last_outcome": entry.last_outcome,
            "last_prng": entry.last_prng_type,
            "recommendation": recommendation
        }
    
    def expire_old_entries(self) -> int:
        """Run TTL expiry using policy threshold."""
        return self.registry.expire_old_entries(self.policy.FINGERPRINT_TTL_DAYS)
    
    def close(self):
        """Close registry connection."""
        self.registry.close()


# =============================================================================
# STANDALONE CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="WATCHER Registry Hooks CLI")
    parser.add_argument("--check", nargs=2, metavar=("FINGERPRINT", "PRNG"),
                        help="Pre-run check for fingerprint+prng")
    parser.add_argument("--record", nargs=3, metavar=("FINGERPRINT", "PRNG", "OUTCOME"),
                        help="Record an attempt")
    parser.add_argument("--status", type=str, metavar="FINGERPRINT",
                        help="Get fingerprint status")
    parser.add_argument("--next", type=str, metavar="FINGERPRINT",
                        help="Get next PRNG to try")
    parser.add_argument("--db", type=str, default="agents/data/fingerprint_registry.db",
                        help="Database path")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    hooks = WatcherRegistryHooks(args.db)
    
    try:
        if args.check:
            fingerprint, prng = args.check
            decision = hooks.pre_run_check(fingerprint, prng)
            print(f"\n{'='*60}")
            print(f"PRE-RUN CHECK: {fingerprint[:8]} + {prng}")
            print(f"{'='*60}")
            print(f"Action: {decision.action}")
            print(f"Reason: {decision.reason}")
            if decision.suggested_prng:
                print(f"Suggested: {decision.suggested_prng}")
            print(f"Attempts: {decision.attempt_count}")
            print(f"Untried: {decision.untried_count}")
            if decision.requires_human:
                print(f"⚠️  REQUIRES HUMAN REVIEW")
            print(f"{'='*60}\n")
            
        elif args.record:
            fingerprint, prng, outcome = args.record
            result = hooks.post_run_record(fingerprint, prng, outcome)
            print(f"\n{'='*60}")
            print(f"POST-RUN RECORD: {fingerprint[:8]} + {prng}")
            print(f"{'='*60}")
            print(f"Recorded: {'✅' if result.recorded else '❌ (duplicate)'}")
            print(f"Outcome: {result.outcome}")
            print(f"Total Attempts: {result.total_attempts}")
            print(f"Total Failures: {result.total_failures}")
            print(f"{'='*60}\n")
            
        elif args.status:
            status = hooks.get_fingerprint_status(args.status)
            print(f"\n{'='*60}")
            print(f"FINGERPRINT STATUS: {args.status[:8]}")
            print(f"{'='*60}")
            for k, v in status.items():
                print(f"{k}: {v}")
            print(f"{'='*60}\n")
            
        elif args.next:
            next_prng = hooks.get_next_prng(args.next)
            if next_prng:
                print(f"Next PRNG for {args.next[:8]}: {next_prng}")
            else:
                print(f"All PRNGs exhausted for {args.next[:8]}")
        
        else:
            parser.print_help()
            
    finally:
        hooks.close()


if __name__ == "__main__":
    main()
