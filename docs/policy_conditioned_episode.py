#!/usr/bin/env python3
"""
policy_conditioned_episode.py — Phase 9B.2: Policy-Conditioned Episode Runner

Version: 1.0.0
Created: 2026-01-30
Team Beta Approved: 2026-01-30

This module provides the integration layer between policy_transform.py and
selfplay_orchestrator.py. It handles:
1. Loading the active policy from learned_policy_active.json
2. Applying policy transforms to survivors
3. Tracking policy lineage across episodes
4. Emitting candidates with proper fingerprints

INTEGRATION:
    This module is imported by selfplay_orchestrator.py.
    See PHASE_9B2_INTEGRATION_SPEC.md for hook points.

AUTHORITY BOUNDARIES:
    - Reads learned_policy_active.json (written by Chapter 13)
    - Writes learned_policy_candidate.json (read by Chapter 13)
    - Never promotes policies (that's Chapter 13's job)
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

# Import our Phase 9B.1 module
from policy_transform import (
    apply_policy,
    compute_policy_fingerprint,
    validate_policy_schema,
    create_empty_policy,
    PolicyTransformResult,
    PolicyViolationError,
    PolicyValidationError,
    ABSOLUTE_MIN_SURVIVORS,
)

# =============================================================================
# CONSTANTS
# =============================================================================

VERSION = "1.0.0"

# File paths (relative to project root)
LEARNED_POLICY_ACTIVE_FILE = "learned_policy_active.json"
LEARNED_POLICY_CANDIDATE_FILE = "learned_policy_candidate.json"
POLICY_HISTORY_DIR = "policy_history"

# Logging
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PolicyConditionedResult:
    """Result from running a policy-conditioned episode."""
    
    # Input/output counts
    survivors_before: int
    survivors_after: int
    
    # Policy info
    active_policy_id: Optional[str]
    active_policy_fingerprint: str
    
    # Transform audit
    transform_log: List[str]
    
    # Conditioned survivors (ready for training)
    conditioned_survivors: List[Dict]
    
    # Metrics for telemetry
    transform_time_ms: float = 0.0
    filter_removed: int = 0
    weight_adjusted: int = 0
    mask_features_removed: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "survivors_before": self.survivors_before,
            "survivors_after": self.survivors_after,
            "active_policy_id": self.active_policy_id,
            "active_policy_fingerprint": self.active_policy_fingerprint,
            "transform_log": self.transform_log,
            "transform_time_ms": self.transform_time_ms,
            "filter_removed": self.filter_removed,
            "weight_adjusted": self.weight_adjusted,
            "mask_features_removed": self.mask_features_removed,
        }


@dataclass
class PolicyCandidate:
    """A policy candidate emitted by selfplay for Chapter 13 validation."""
    
    policy_id: str
    parent_policy_id: Optional[str]
    created_at: str
    episode_number: int
    
    # Transforms (proposed for next generation)
    transforms: Dict
    
    # Metrics from this episode's training
    fitness: Optional[float] = None
    fitness_delta: Optional[float] = None  # vs parent
    val_r2: Optional[float] = None
    train_val_gap: Optional[float] = None
    survivor_count_after: Optional[int] = None
    
    # Computed
    fingerprint: Optional[str] = None
    
    def __post_init__(self):
        if self.fingerprint is None:
            self.fingerprint = compute_policy_fingerprint({"transforms": self.transforms})
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# POLICY LOADING
# =============================================================================

def load_active_policy(project_root: str = ".") -> Tuple[Dict, bool]:
    """
    Load the currently active policy from learned_policy_active.json.
    
    Returns:
        (policy_dict, was_loaded)
        
        If no active policy exists, returns (empty_policy, False).
        This is the baseline case for the first episode.
    """
    policy_path = Path(project_root) / LEARNED_POLICY_ACTIVE_FILE
    
    if not policy_path.exists():
        logger.info(f"No active policy found at {policy_path}, using empty baseline")
        return create_empty_policy("baseline_empty"), False
    
    try:
        with open(policy_path, 'r') as f:
            policy = json.load(f)
        
        # Validate schema
        is_valid, errors = validate_policy_schema(policy)
        if not is_valid:
            logger.warning(f"Active policy failed validation: {errors}")
            logger.warning("Falling back to empty baseline")
            return create_empty_policy("baseline_fallback"), False
        
        policy_id = policy.get("policy_id", "unknown")
        logger.info(f"Loaded active policy: {policy_id}")
        
        return policy, True
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse active policy JSON: {e}")
        return create_empty_policy("baseline_parse_error"), False
    except Exception as e:
        logger.error(f"Failed to load active policy: {e}")
        return create_empty_policy("baseline_load_error"), False


def get_parent_fitness(policy: Dict) -> Optional[float]:
    """Extract fitness from parent policy for delta calculation."""
    return policy.get("fitness")


# =============================================================================
# CORE EPISODE CONDITIONING
# =============================================================================

def condition_episode(
    survivors: List[Dict],
    project_root: str = ".",
    *,
    force_policy: Optional[Dict] = None,
    strict_mode: bool = True,
) -> PolicyConditionedResult:
    """
    Condition survivors using the active policy.
    
    This is the main entry point for Phase 9B.2 integration.
    Call this before passing survivors to inner_episode_trainer.
    
    Args:
        survivors: Raw survivors from outer episode / Step 3
        project_root: Path to project root (for finding policy files)
        force_policy: Override active policy (for testing)
        strict_mode: If True, raise on policy errors
    
    Returns:
        PolicyConditionedResult with conditioned survivors and audit info
    
    Example:
        result = condition_episode(raw_survivors, project_root="~/distributed_prng_analysis")
        model = inner_episode_trainer.train(result.conditioned_survivors)
    """
    import time
    start_time = time.time()
    
    # Load active policy (or use override)
    if force_policy is not None:
        active_policy = force_policy
        was_loaded = True
        logger.info("Using forced policy override")
    else:
        active_policy, was_loaded = load_active_policy(project_root)
    
    policy_id = active_policy.get("policy_id")
    
    # Apply transforms
    try:
        transform_result = apply_policy(survivors, active_policy, strict_mode=strict_mode)
    except PolicyViolationError as e:
        logger.error(f"Policy violation during conditioning: {e}")
        if strict_mode:
            raise
        # Fallback: use unconditioned survivors
        logger.warning("Falling back to unconditioned survivors")
        transform_result = PolicyTransformResult(
            survivors=survivors,
            original_count=len(survivors),
            filtered_count=len(survivors),
            transform_log=["FALLBACK: policy violation, using raw survivors"],
            policy_fingerprint=compute_policy_fingerprint(create_empty_policy()),
        )
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    # Log transforms
    for log_entry in transform_result.transform_log:
        logger.debug(f"Transform: {log_entry}")
    
    logger.info(
        f"Episode conditioned: {transform_result.original_count} → "
        f"{transform_result.filtered_count} survivors "
        f"(policy={policy_id}, {elapsed_ms:.1f}ms)"
    )
    
    return PolicyConditionedResult(
        survivors_before=transform_result.original_count,
        survivors_after=transform_result.filtered_count,
        active_policy_id=policy_id,
        active_policy_fingerprint=transform_result.policy_fingerprint,
        transform_log=transform_result.transform_log,
        conditioned_survivors=transform_result.survivors,
        transform_time_ms=elapsed_ms,
        filter_removed=transform_result.filter_removed,
        weight_adjusted=transform_result.weight_adjusted,
        mask_features_removed=transform_result.mask_features_removed,
    )


# =============================================================================
# CANDIDATE EMISSION
# =============================================================================

def create_policy_candidate(
    episode_number: int,
    parent_policy: Dict,
    training_result: Dict,
    proposed_transforms: Optional[Dict] = None,
) -> PolicyCandidate:
    """
    Create a policy candidate from an episode's training results.
    
    Args:
        episode_number: Sequential episode number
        parent_policy: The active policy used for this episode
        training_result: Dict with fitness, val_r2, train_val_gap, etc.
        proposed_transforms: New transforms for next generation
                           (if None, inherits from parent)
    
    Returns:
        PolicyCandidate ready for emission
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Inherit transforms from parent if not specified
    if proposed_transforms is None:
        proposed_transforms = parent_policy.get("transforms", {})
    
    # Calculate fitness delta
    parent_fitness = get_parent_fitness(parent_policy)
    current_fitness = training_result.get("fitness")
    
    fitness_delta = None
    if parent_fitness is not None and current_fitness is not None:
        fitness_delta = current_fitness - parent_fitness
    
    candidate = PolicyCandidate(
        policy_id=f"policy_selfplay_{timestamp}_ep{episode_number:03d}",
        parent_policy_id=parent_policy.get("policy_id"),
        created_at=datetime.now(timezone.utc).isoformat(),
        episode_number=episode_number,
        transforms=proposed_transforms,
        fitness=current_fitness,
        fitness_delta=fitness_delta,
        val_r2=training_result.get("val_r2"),
        train_val_gap=training_result.get("train_val_gap"),
        survivor_count_after=training_result.get("survivor_count"),
    )
    
    logger.info(
        f"Created policy candidate: {candidate.policy_id} "
        f"(fitness={candidate.fitness:.4f}, delta={fitness_delta})"
    )
    
    return candidate


def emit_policy_candidate(
    candidate: PolicyCandidate,
    project_root: str = ".",
    archive: bool = True,
) -> Path:
    """
    Write policy candidate to learned_policy_candidate.json.
    
    This file is read by Chapter 13 for validation and potential promotion.
    
    Args:
        candidate: PolicyCandidate to emit
        project_root: Path to project root
        archive: If True, also save to policy_history/ directory
    
    Returns:
        Path to the emitted candidate file
    """
    candidate_path = Path(project_root) / LEARNED_POLICY_CANDIDATE_FILE
    
    # Write main candidate file
    with open(candidate_path, 'w') as f:
        f.write(candidate.to_json())
    
    logger.info(f"Emitted policy candidate: {candidate_path}")
    
    # Archive for history
    if archive:
        history_dir = Path(project_root) / POLICY_HISTORY_DIR
        history_dir.mkdir(exist_ok=True)
        
        archive_path = history_dir / f"{candidate.policy_id}.json"
        with open(archive_path, 'w') as f:
            f.write(candidate.to_json())
        
        logger.debug(f"Archived policy candidate: {archive_path}")
    
    return candidate_path


# =============================================================================
# POLICY PROPOSAL HEURISTICS (Phase 9B.3 preview)
# =============================================================================

def propose_transform_update(
    current_transforms: Dict,
    training_result: Dict,
    conditioning_result: PolicyConditionedResult,
) -> Dict:
    """
    Propose updated transforms based on training outcomes.
    
    This is a placeholder for Phase 9B.3 (proposal heuristics).
    Currently just returns the current transforms unchanged.
    
    Future heuristics could include:
    - If fitness improved, tighten filter threshold
    - If fitness dropped, loosen filter
    - If overfit detected, add feature masking
    - If survivors too few, lower filter threshold
    
    Args:
        current_transforms: Current policy transforms
        training_result: Dict with fitness, val_r2, etc.
        conditioning_result: Results from condition_episode()
    
    Returns:
        Proposed transforms for next generation
    """
    # Phase 9B.2: No automatic updates, just inherit
    # Phase 9B.3 will add heuristic updates here
    
    return current_transforms.copy()


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def run_conditioned_episode(
    survivors: List[Dict],
    episode_number: int,
    trainer_func,  # Callable[[List[Dict]], Dict]
    project_root: str = ".",
    *,
    policy_conditioned: bool = True,
    emit_candidate: bool = True,
) -> Tuple[Dict, Optional[PolicyCandidate]]:
    """
    High-level helper for running a complete policy-conditioned episode.
    
    This wraps the full workflow:
    1. Load active policy
    2. Condition survivors
    3. Train model
    4. Create candidate
    5. Emit candidate
    
    Args:
        survivors: Raw survivors
        episode_number: Sequential episode number
        trainer_func: Function that takes survivors and returns training result dict
        project_root: Path to project root
        policy_conditioned: If False, skip conditioning (baseline mode)
        emit_candidate: If True, write learned_policy_candidate.json
    
    Returns:
        (training_result, policy_candidate or None)
    
    Example:
        def my_trainer(survivors):
            return inner_episode_trainer.train(survivors, model_type="catboost")
        
        result, candidate = run_conditioned_episode(
            survivors=raw_survivors,
            episode_number=5,
            trainer_func=my_trainer,
            project_root="~/distributed_prng_analysis"
        )
    """
    # Load active policy
    active_policy, _ = load_active_policy(project_root)
    
    # Condition survivors (or skip if disabled)
    if policy_conditioned:
        cond_result = condition_episode(
            survivors,
            project_root=project_root,
        )
        training_survivors = cond_result.conditioned_survivors
    else:
        # Baseline mode: use raw survivors
        cond_result = None
        training_survivors = survivors
        logger.info("Policy conditioning disabled, using raw survivors")
    
    # Train model
    logger.info(f"Training episode {episode_number} with {len(training_survivors)} survivors")
    training_result = trainer_func(training_survivors)
    
    # Add survivor count to result
    training_result["survivor_count"] = len(training_survivors)
    
    # Create candidate
    proposed_transforms = None
    if cond_result is not None:
        proposed_transforms = propose_transform_update(
            active_policy.get("transforms", {}),
            training_result,
            cond_result,
        )
    
    candidate = create_policy_candidate(
        episode_number=episode_number,
        parent_policy=active_policy,
        training_result=training_result,
        proposed_transforms=proposed_transforms,
    )
    
    # Emit candidate
    if emit_candidate:
        emit_policy_candidate(candidate, project_root=project_root)
    
    return training_result, candidate


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI interface for testing and validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Phase 9B.2: Policy-Conditioned Episode Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run unit tests
  python policy_conditioned_episode.py --test
  
  # Load and display active policy
  python policy_conditioned_episode.py --show-active --project-root ~/distributed_prng_analysis
  
  # Dry-run conditioning on test data
  python policy_conditioned_episode.py --dry-run --survivors test_survivors.json
        """
    )
    
    parser.add_argument('--test', action='store_true',
                        help='Run unit tests')
    parser.add_argument('--show-active', action='store_true',
                        help='Show current active policy')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry-run conditioning without training')
    parser.add_argument('--survivors', type=str,
                        help='Survivors JSON file (for --dry-run)')
    parser.add_argument('--project-root', type=str, default='.',
                        help='Project root directory')
    parser.add_argument('--version', action='store_true',
                        help='Show version')
    
    args = parser.parse_args()
    
    if args.version:
        print(f"policy_conditioned_episode.py v{VERSION}")
        return
    
    if args.test:
        run_tests()
        return
    
    if args.show_active:
        policy, was_loaded = load_active_policy(args.project_root)
        if was_loaded:
            print(f"Active Policy: {policy.get('policy_id')}")
            print(f"Fingerprint: {compute_policy_fingerprint(policy)}")
            print(f"\nTransforms:")
            print(json.dumps(policy.get("transforms", {}), indent=2))
        else:
            print("No active policy found (baseline mode)")
        return
    
    if args.dry_run:
        if not args.survivors:
            print("Error: --survivors required with --dry-run")
            return
        
        with open(args.survivors) as f:
            survivors = json.load(f)
        
        result = condition_episode(survivors, args.project_root)
        
        print(f"Conditioning Result:")
        print(f"  Before: {result.survivors_before}")
        print(f"  After: {result.survivors_after}")
        print(f"  Policy: {result.active_policy_id}")
        print(f"  Fingerprint: {result.active_policy_fingerprint}")
        print(f"\nTransform Log:")
        for log in result.transform_log:
            print(f"  {log}")
        return
    
    parser.print_help()


# =============================================================================
# UNIT TESTS
# =============================================================================

def run_tests():
    """Run unit tests for policy_conditioned_episode module."""
    print("=" * 60)
    print("POLICY CONDITIONED EPISODE — UNIT TESTS")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    
    def assert_true(condition, msg):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {msg}")
            passed += 1
        else:
            print(f"  ❌ {msg}")
            failed += 1
    
    def assert_eq(actual, expected, msg):
        nonlocal passed, failed
        if actual == expected:
            print(f"  ✅ {msg}")
            passed += 1
        else:
            print(f"  ❌ {msg}")
            print(f"     Expected: {expected}")
            print(f"     Actual:   {actual}")
            failed += 1
    
    # Test data
    def make_test_survivors(n=100):
        return [
            {
                "seed": 1000 + i,
                "score": 50.0 + i * 0.5,
                "holdout_hits": 0.3 + (i / n) * 0.5,
                "features": {
                    "temporal_stability": 0.5 + (i / n) * 0.4,
                    "residue_coherence": 0.1 + (i / n) * 0.3,
                }
            }
            for i in range(n)
        ]
    
    # === Test 1: Load missing policy returns baseline ===
    print("[Test 1] Load missing policy → baseline")
    policy, was_loaded = load_active_policy("/nonexistent/path")
    assert_true(not was_loaded, "was_loaded = False")
    assert_true("baseline" in policy.get("policy_id", ""), "policy_id contains 'baseline'")
    print()
    
    # === Test 2: Condition with empty policy ===
    print("[Test 2] Condition with empty policy (passthrough)")
    survivors = make_test_survivors(100)
    empty_policy = create_empty_policy("test_empty")
    result = condition_episode(survivors, force_policy=empty_policy)
    assert_eq(result.survivors_before, 100, "survivors_before = 100")
    assert_eq(result.survivors_after, 100, "survivors_after = 100 (passthrough)")
    assert_eq(len(result.conditioned_survivors), 100, "conditioned_survivors length = 100")
    print()
    
    # === Test 3: Condition with filter policy ===
    print("[Test 3] Condition with filter policy")
    survivors = make_test_survivors(100)
    filter_policy = {
        "policy_id": "test_filter",
        "transforms": {
            "filter": {
                "enabled": True,
                "field": "holdout_hits",
                "operator": "gte",
                "threshold": 0.6,
                "min_survivors": 50
            }
        }
    }
    result = condition_episode(survivors, force_policy=filter_policy)
    assert_eq(result.survivors_before, 100, "survivors_before = 100")
    assert_true(result.survivors_after < 100, f"survivors_after < 100 (actual: {result.survivors_after})")
    assert_true(result.survivors_after >= 50, f"survivors_after >= 50 (safety floor)")
    print()
    
    # === Test 4: Create policy candidate ===
    print("[Test 4] Create policy candidate")
    parent_policy = {
        "policy_id": "parent_policy_001",
        "fitness": 0.75,
        "transforms": {"filter": {"enabled": True, "field": "score", "threshold": 0.5}}
    }
    training_result = {
        "fitness": 0.82,
        "val_r2": 0.95,
        "train_val_gap": 0.02,
    }
    candidate = create_policy_candidate(
        episode_number=5,
        parent_policy=parent_policy,
        training_result=training_result,
    )
    assert_true(candidate.policy_id.startswith("policy_selfplay_"), "policy_id prefix correct")
    assert_eq(candidate.parent_policy_id, "parent_policy_001", "parent_policy_id correct")
    assert_eq(candidate.episode_number, 5, "episode_number = 5")
    assert_eq(candidate.fitness, 0.82, "fitness = 0.82")
    assert_true(abs(candidate.fitness_delta - 0.07) < 0.001, f"fitness_delta ≈ 0.07 (actual: {candidate.fitness_delta})")
    assert_true(candidate.fingerprint is not None, "fingerprint computed")
    print()
    
    # === Test 5: Candidate serialization ===
    print("[Test 5] Candidate serialization")
    candidate_dict = candidate.to_dict()
    assert_true("policy_id" in candidate_dict, "policy_id in dict")
    assert_true("fingerprint" in candidate_dict, "fingerprint in dict")
    candidate_json = candidate.to_json()
    assert_true('"policy_id"' in candidate_json, "policy_id in JSON")
    print()
    
    # === Test 6: PolicyConditionedResult ===
    print("[Test 6] PolicyConditionedResult serialization")
    pcr = PolicyConditionedResult(
        survivors_before=100,
        survivors_after=75,
        active_policy_id="test_policy",
        active_policy_fingerprint="abc123",
        transform_log=["filter: 100 → 75"],
        conditioned_survivors=[{"seed": 1}],
        transform_time_ms=42.5,
    )
    pcr_dict = pcr.to_dict()
    assert_eq(pcr_dict["survivors_before"], 100, "survivors_before in dict")
    assert_eq(pcr_dict["transform_time_ms"], 42.5, "transform_time_ms in dict")
    print()
    
    # === Test 7: Transform log propagation ===
    print("[Test 7] Transform log propagation")
    survivors = make_test_survivors(100)
    policy = {
        "policy_id": "test_logging",
        "transforms": {
            "filter": {"enabled": True, "field": "score", "operator": "gte", "threshold": 60.0}
        }
    }
    result = condition_episode(survivors, force_policy=policy)
    assert_true(len(result.transform_log) > 0, "transform_log not empty")
    assert_true(any("filter" in log for log in result.transform_log), "filter logged")
    print()
    
    # === Summary ===
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED")
    else:
        print(f"\n❌ {failed} TESTS FAILED")
    
    return failed == 0


if __name__ == "__main__":
    main()
