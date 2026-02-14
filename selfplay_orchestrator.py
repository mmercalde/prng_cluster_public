#!/usr/bin/env python3
"""
Selfplay Orchestrator
=====================
Version: 1.2.0
Date: 2026-02-13
Status: Phase 9B.2 Policy-Conditioned Learning Integration

PURPOSE:
    Air traffic controller for selfplay learning.
    Schedules work, collects results, emits telemetry, writes candidates.
    
    Phase 9B.2 adds policy-conditioned learning:
    - Loads active policy from learned_policy_active.json
    - Applies transforms (filter/weight/mask/window) to survivors before training
    - Tracks policy lineage across episodes
    - Emits candidates with fingerprints for Chapter 13 validation
    
    Does NOT:
    - Contain ML logic (delegated to inner_episode_trainer)
    - Contain GPU logic (delegated to coordinators)
    - Decide promotion (delegated to Chapter 13)
    - Have authority over ground truth

AUTHORITY CONTRACT (from CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md):
    
    ┌─────────────────────────────────────────────────────────────────┐
    │ "Selfplay explores. Chapter 13 decides. WATCHER enforces."     │
    └─────────────────────────────────────────────────────────────────┘
    
    Selfplay Orchestrator RESPONSIBILITIES:
        ✅ Schedule outer episodes (GPU sieving via coordinators)
        ✅ Schedule inner episodes (CPU ML via inner_episode_trainer)
        ✅ Emit telemetry (via learning_telemetry)
        ✅ Write learned_policy_candidate.json
        ✅ [NEW] Apply policy conditioning to survivors before training
        ✅ [NEW] Track policy lineage (parent_policy_id)
        
    Selfplay Orchestrator MUST NOT:
        ❌ Access live draw outcomes (Chapter 13 only)
        ❌ Promote policies (Chapter 13 only)
        ❌ Modify learned_policy_active.json (Chapter 13 only)
        ❌ Bypass coordinators for GPU work
        ❌ Make promotion decisions

INVARIANT 4 COMPLIANCE (Coordinator Requirement):
    "GPU sieving work MUST use coordinator.py / scripts_coordinator.py.
     Direct SSH to rigs for GPU work is FORBIDDEN."

PHASE 9B.2 INVARIANTS:
    - Policy transforms are pure functional (stateless, deterministic)
    - Transforms never fabricate data (only filter/weight/mask/window)
    - ABSOLUTE_MIN_SURVIVORS = 50 enforced
    - Fingerprints enable semantic deduplication

SINGLE-WRITER MODEL:
    Only this orchestrator writes telemetry.
    Workers return results; orchestrator records.

OUTPUTS:
    - learned_policy_candidate.json: Policy candidates for Chapter 13 review
    - Telemetry records via LearningTelemetry

USAGE:
    # Full selfplay run (baseline mode)
    python3 selfplay_orchestrator.py --config configs/selfplay_config.json
    
    # Policy-conditioned mode (Phase 9B.2)
    python3 selfplay_orchestrator.py --config configs/selfplay_config.json --policy-conditioned
    
    # Single outer+inner episode (testing)
    python3 selfplay_orchestrator.py --single-episode --survivors path/to/survivors.json
    
    # Testing without candidate emission
    python3 selfplay_orchestrator.py --survivors test.json --policy-conditioned --no-emit-candidate
    
    # Dry run (no actual work, just validation)
    python3 selfplay_orchestrator.py --config configs/selfplay_config.json --dry-run

CHANGELOG:
    v1.2.0 (2026-02-13):
        - S83 Phase 8A: Episode diagnostics (Team Beta v2 side-channel architecture)
        - Added episode_diagnostics to EpisodeResult, trend detection
    
    v1.1.0 (2026-01-30):
        - Phase 9B.2: Policy-conditioned learning integration
        - Added --policy-conditioned CLI flag
        - Added --no-emit-candidate CLI flag for testing
        - Integrated condition_episode() before training
        - Added candidate emission with lineage tracking
        - Fitness delta calculation vs parent policy
        - Requires: policy_transform.py, policy_conditioned_episode.py
    
    v1.0.0 (2026-01-30):
        - Initial release
        - Phase 8 Selfplay Integration
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import uuid

# =============================================================================
# IMPORTS (with graceful fallbacks for standalone testing)
# =============================================================================

try:
    from modules.learning_telemetry import LearningTelemetry
    TELEMETRY_AVAILABLE = True
except ImportError:
    try:
        from learning_telemetry import LearningTelemetry
        TELEMETRY_AVAILABLE = True
    except ImportError:
        TELEMETRY_AVAILABLE = False
        print("⚠️  Warning: learning_telemetry not found, telemetry disabled")

try:
    from inner_episode_trainer import InnerEpisodeTrainer, TrainerConfig
    TRAINER_AVAILABLE = True
except ImportError:
    TRAINER_AVAILABLE = False
    print("⚠️  Warning: inner_episode_trainer not found")

# =============================================================================
# PHASE 9B.2: Policy-Conditioned Learning Imports
# =============================================================================

try:
    from policy_conditioned_episode import (
        condition_episode,
        create_policy_candidate as create_9b_candidate,
        emit_policy_candidate,
        load_active_policy,
        PolicyConditionedResult,
        PolicyCandidate as PolicyCandidate9B,
    )
    from policy_transform import (
        compute_policy_fingerprint,
        create_empty_policy,
        PolicyViolationError,
    )
    POLICY_CONDITIONING_AVAILABLE = True
except ImportError:
    POLICY_CONDITIONING_AVAILABLE = False
    print("⚠️  Warning: policy_conditioned_episode not found, Phase 9B.2 disabled")


# =============================================================================
# CONFIGURATION
# =============================================================================

SCHEMA_VERSION = "1.1.0"
DEFAULT_CONFIG_PATH = "configs/selfplay_config.json"
DEFAULT_CANDIDATE_PATH = "learned_policy_candidate.json"
DEFAULT_TELEMETRY_DIR = "telemetry"

# Coordinator scripts (MANDATORY for GPU work)
COORDINATOR_SCRIPT = "coordinator.py"
SCRIPTS_COORDINATOR = "scripts_coordinator.py"

# Minimum fitness threshold to emit a candidate (configurable)
DEFAULT_MIN_FITNESS_THRESHOLD = 0.5


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SelfplayConfig:
    """Configuration for selfplay orchestration."""
    
    # Episode configuration
    max_episodes: int = 10
    min_fitness_threshold: float = DEFAULT_MIN_FITNESS_THRESHOLD
    
    # Inner episode configuration (matches TrainerConfig)
    model_types: List[str] = field(default_factory=lambda: ["lightgbm", "xgboost", "catboost"])
    n_estimators: int = 100
    k_folds: int = 3
    n_jobs: int = -1  # -1 = auto-detect (cpu_count - 2, minimum 2)
    
    # Outer episode configuration (GPU sieving)
    # NOTE: Disabled by default - coordinator integration needs matching interface
    use_coordinator: bool = False  # Set True only when coordinator interface is ready
    coordinator_script: str = COORDINATOR_SCRIPT
    sieve_params: Dict[str, Any] = field(default_factory=dict)
    
    # Paths
    survivors_input: str = "survivors_with_scores.json"
    candidate_output: str = DEFAULT_CANDIDATE_PATH
    telemetry_dir: str = DEFAULT_TELEMETRY_DIR
    project_root: str = "."  # Added for Phase 9B.2
    
    # Policy generation
    policy_prefix: str = "policy"
    
    # Phase 9B.2: Policy-conditioned learning
    policy_conditioned: bool = False  # Enable via --policy-conditioned
    
    # Phase 8A: Training diagnostics capture (Ch14)
    enable_diagnostics: bool = False  # S83_PHASE8A_DIAG
    emit_candidates: bool = True  # Disable via --no-emit-candidate
    
    def __post_init__(self):
        """Auto-detect n_jobs if set to -1."""
        if self.n_jobs == -1 or self.n_jobs is None:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            # Leave ~10% headroom (minimum 1 thread for OS), at least 2 for training
            headroom = max(1, cpu_count // 10)
            self.n_jobs = max(2, cpu_count - headroom)
    
    @classmethod
    def from_file(cls, path: str) -> "SelfplayConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeResult:
    """Result of a single selfplay episode (outer + inner)."""
    
    episode_id: str
    timestamp: str
    
    # Inner episode results
    model_type: str
    fitness: float
    val_r2: float
    val_mae: Optional[float] = None
    fold_std: Optional[float] = None
    train_val_gap: Optional[float] = None
    training_time_ms: float = 0.0
    
    # Data stats
    survivor_count: int = 0
    survivor_count_before: int = 0  # Before conditioning (Phase 9B.2)
    feature_count: int = 0
    
    # Policy info
    policy_id: Optional[str] = None
    is_candidate: bool = False
    
    # Phase 9B.2: Policy conditioning info
    policy_conditioned: bool = False
    active_policy_id: Optional[str] = None
    active_policy_fingerprint: Optional[str] = None
    conditioning_log: List[str] = field(default_factory=list)
    
    # Phase 8A: Episode diagnostics (Ch14)  # S83_PHASE8A_DIAG
    episode_diagnostics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyCandidate:
    """
    A policy candidate for Chapter 13 review.
    
    This is a HYPOTHESIS, not a DECISION.
    Only Chapter 13 can promote this to learned_policy_active.json.
    """
    
    schema_version: str
    policy_id: str
    created_at: str
    source: str  # "selfplay"
    
    # Model info
    model_type: str
    fitness: float
    val_r2: float
    
    # Metrics
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Parameters (what would be applied if promoted)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Provenance
    episode_id: str = ""
    survivors_source: str = ""
    
    # Phase 9B.2: Policy lineage
    parent_policy_id: Optional[str] = None
    fitness_delta: Optional[float] = None
    fingerprint: Optional[str] = None
    transforms: Dict[str, Any] = field(default_factory=dict)
    
    # Status (always "candidate" - Chapter 13 decides promotion)
    status: str = "candidate"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# MAIN ORCHESTRATOR CLASS
# =============================================================================

class SelfplayOrchestrator:
    """
    Selfplay Orchestrator — Air Traffic Controller
    
    Schedules outer episodes (GPU sieving) and inner episodes (CPU ML),
    emits telemetry, and writes policy candidates.
    
    Phase 9B.2 adds policy-conditioned learning:
    - Loads active policy before each episode
    - Applies transforms to survivors
    - Tracks lineage across episodes
    
    AUTHORITY:
        - Schedules work (outer via coordinators, inner via trainer)
        - Emits telemetry (single writer)
        - Writes learned_policy_candidate.json
        
    DOES NOT:
        - Decide promotion (Chapter 13)
        - Access ground truth (Chapter 13)
        - Bypass coordinators for GPU (Invariant 4)
    """
    
    def __init__(
        self,
        config: SelfplayConfig,
        run_id: Optional[str] = None,
        dry_run: bool = False
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: SelfplayConfig instance
            run_id: Optional run identifier
            dry_run: If True, validate but don't execute
        """
        self.config = config
        self.dry_run = dry_run
        self.run_id = run_id or datetime.now(timezone.utc).strftime("selfplay_%Y%m%d_%H%M%S")
        
        # Initialize telemetry (single writer)
        self.telemetry: Optional[LearningTelemetry] = None
        if TELEMETRY_AVAILABLE and not dry_run:
            self.telemetry = LearningTelemetry(
                output_dir=config.telemetry_dir,
                run_id=self.run_id
            )
        
        # Episode tracking
        self.episodes_completed = 0
        self.best_fitness = -float('inf')
        self.best_episode: Optional[EpisodeResult] = None
        
        # Phase 9B.2: Policy tracking
        self.active_policy: Optional[Dict] = None
        self.parent_fitness: Optional[float] = None
        
        # Validate configuration
        self._validate_config()
        
        print(f"\n{'='*60}")
        print(f"🎮 Selfplay Orchestrator v{SCHEMA_VERSION}")
        print(f"{'='*60}")
        print(f"   Run ID: {self.run_id}")
        print(f"   Max episodes: {config.max_episodes}")
        print(f"   Min fitness threshold: {config.min_fitness_threshold}")
        print(f"   Model types: {', '.join(config.model_types)}")
        print(f"   N estimators: {config.n_estimators}")
        print(f"   K-folds: {config.k_folds}")
        print(f"   CPU threads: {config.n_jobs}")
        print(f"   Outer episodes: {'enabled' if config.use_coordinator else 'disabled (inner-only)'}")
        print(f"   Policy-conditioned: {'✅ ENABLED' if config.policy_conditioned else 'disabled (baseline)'}")
        print(f"   Emit candidates: {'enabled' if config.emit_candidates else 'disabled (testing)'}")
        print(f"   Dry run: {dry_run}")
        print(f"{'='*60}\n")
    
    def _validate_config(self) -> None:
        """Validate configuration before running."""
        errors = []
        
        # Phase 9B.2: Check policy conditioning availability
        if self.config.policy_conditioned and not POLICY_CONDITIONING_AVAILABLE:
            print("   ⚠️  Warning: --policy-conditioned requested but modules not found")
            print("      Requires: policy_transform.py, policy_conditioned_episode.py")
            print("      Falling back to baseline mode")
            self.config.policy_conditioned = False
        
        # INVARIANT 4 NOTE: Coordinator is required for GPU work, but inner-only mode is allowed
        if not self.config.use_coordinator:
            print("   ℹ️  Inner-only mode: Outer episodes (GPU sieving) disabled")
            print("      Set use_coordinator=true when coordinator interface is ready")
        
        # Check coordinator exists (only if enabled)
        if self.config.use_coordinator and not self.dry_run:
            coordinator_path = Path(self.config.coordinator_script)
            if not coordinator_path.exists():
                # Try common locations
                alt_paths = [
                    Path(self.config.coordinator_script),
                    Path("scripts") / self.config.coordinator_script,
                    Path.home() / "distributed_prng_analysis" / self.config.coordinator_script,
                ]
                found = False
                for alt in alt_paths:
                    if alt.exists():
                        self.config.coordinator_script = str(alt)
                        found = True
                        break
                if not found:
                    print(f"   ⚠️  Warning: Coordinator script not found at {coordinator_path}")
                    print(f"      Disabling outer episodes (inner-only mode)")
                    self.config.use_coordinator = False
        
        # Check inner trainer
        if not TRAINER_AVAILABLE and not self.dry_run:
            print("   ⚠️  Warning: InnerEpisodeTrainer not available")
        
        # Check survivors file
        survivors_path = Path(self.config.survivors_input)
        if not survivors_path.exists() and not self.dry_run:
            print(f"   ⚠️  Warning: Survivors file not found: {survivors_path}")
        
        if errors:
            for err in errors:
                print(f"❌ {err}")
            raise ValueError("Configuration validation failed")
    
    # =========================================================================
    # PHASE 9B.2: POLICY LOADING
    # =========================================================================
    
    def _load_active_policy(self) -> Tuple[Dict, bool]:
        """
        Load the currently active policy for conditioning.
        
        Phase 9B.2: Loads from learned_policy_active.json.
        If not found, returns an empty baseline policy.
        
        Returns:
            (policy_dict, was_loaded)
        """
        if not POLICY_CONDITIONING_AVAILABLE:
            return {"policy_id": "baseline_unavailable", "transforms": {}}, False
        
        policy, was_loaded = load_active_policy(self.config.project_root)
        
        if was_loaded:
            self.parent_fitness = policy.get("fitness")
            print(f"   📜 Active policy loaded: {policy.get('policy_id')}")
            if self.parent_fitness is not None:
                print(f"      Parent fitness: {self.parent_fitness:.4f}")
        else:
            print(f"   📜 No active policy found, using baseline")
        
        return policy, was_loaded
    
    # =========================================================================
    # MAIN RUN LOOP
    # =========================================================================
    
    def run(self) -> List[EpisodeResult]:
        """
        Run the selfplay loop.
        
        Returns:
            List of EpisodeResult for all completed episodes
        """
        results: List[EpisodeResult] = []
        
        print(f"🚀 Starting selfplay run: {self.run_id}")
        print(f"   Target: {self.config.max_episodes} episodes\n")
        
        if self.dry_run:
            print("   [DRY RUN] Skipping actual execution")
            return results
        
        # Phase 9B.2: Load active policy at start of run
        if self.config.policy_conditioned:
            self.active_policy, _ = self._load_active_policy()
        
        # S83 Phase 8A: Episode diagnostics history  # S83_PHASE8A_DIAG
        diagnostics_history = []

        for episode_num in range(1, self.config.max_episodes + 1):
            print(f"\n{'─'*50}")
            print(f"📍 Episode {episode_num}/{self.config.max_episodes}")
            print(f"{'─'*50}")
            
            try:
                result = self._run_episode(episode_num)
                results.append(result)
                self.episodes_completed += 1
                
                # S83 Phase 8A: Collect episode diagnostics  # S83_PHASE8A_DIAG
                if hasattr(result, 'episode_diagnostics') and result.episode_diagnostics:
                    diagnostics_history.append({
                        'episode': episode_num,
                        'severity': result.episode_diagnostics.get('severity', 'absent'),
                        'best_round_ratio': result.episode_diagnostics.get('best_round_ratio', 0.0),
                        'fitness': result.fitness,
                        'model_type': result.model_type,
                    })
                    diagnostics_history = diagnostics_history[-20:]
                    self._check_episode_training_trend(diagnostics_history)

                # Track best
                if result.fitness > self.best_fitness:
                    self.best_fitness = result.fitness
                    self.best_episode = result
                    print(f"   🏆 New best fitness: {result.fitness:.4f}")
                
                # Check if we should emit a candidate
                if result.fitness >= self.config.min_fitness_threshold and self.config.emit_candidates:
                    self._emit_candidate(result)
                
            except Exception as e:
                print(f"   ❌ Episode {episode_num} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Summary
        self._print_summary(results)
        
        return results
    
    def _run_episode(self, episode_num: int) -> EpisodeResult:
        """
        Run a single selfplay episode (outer + inner).
        
        Phase 9B.2: Applies policy conditioning before inner episode.
        
        Args:
            episode_num: Episode number for logging
            
        Returns:
            EpisodeResult with metrics
        """
        episode_id = f"{self.run_id}_ep{episode_num:03d}"
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Step 1: Outer episode (GPU sieving) - optional
        survivors_path = self.config.survivors_input
        if self._should_run_outer_episode():
            print(f"   📡 Running outer episode (GPU sieving)...")
            survivors_path = self._run_outer_episode(episode_id)
        else:
            print(f"   📁 Using existing survivors: {survivors_path}")
        
        # Step 2: Inner episode (CPU ML training)
        # Phase 9B.2: With optional policy conditioning
        print(f"   🧠 Running inner episode (CPU ML)...")
        result = self._run_inner_episode(
            episode_id=episode_id,
            timestamp=timestamp,
            survivors_path=survivors_path,
            episode_num=episode_num
        )
        
        # Step 3: Record telemetry (single writer model)
        if self.telemetry:
            self.telemetry.record_inner_episode(
                model_type=result.model_type,
                training_time_ms=result.training_time_ms,
                fitness=result.fitness,
                val_r2=result.val_r2,
                val_mae=result.val_mae,
                fold_std=result.fold_std,
                train_val_gap=result.train_val_gap,
                survivor_count=result.survivor_count,
                feature_count=result.feature_count,
                policy_id=result.policy_id
            )
        
        print(f"   ✅ Episode complete: {result.model_type} fitness={result.fitness:.4f}")
        
        return result
    
    def _should_run_outer_episode(self) -> bool:
        """
        Determine if we should run an outer episode.
        
        For now, we only run outer episodes if explicitly configured
        and the coordinator is available.
        """
        if not self.config.use_coordinator:
            return False
        
        coordinator_path = Path(self.config.coordinator_script)
        return coordinator_path.exists()
    
    def _run_outer_episode(self, episode_id: str) -> str:
        """
        Run outer episode via coordinator (GPU sieving).
        
        INVARIANT 4: MUST use coordinator.py / scripts_coordinator.py.
        Direct SSH to rigs is FORBIDDEN.
        
        Args:
            episode_id: Episode identifier
            
        Returns:
            Path to survivors file
        """
        # Build coordinator command
        cmd = [
            "python3",
            self.config.coordinator_script,
            "--episode-id", episode_id,
        ]
        
        # Add any configured sieve params
        for key, value in self.config.sieve_params.items():
            cmd.extend([f"--{key}", str(value)])
        
        print(f"      Running: {' '.join(cmd[:4])}...")
        
        # Execute via coordinator (NOT direct SSH)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                print(f"      ⚠️  Coordinator returned non-zero: {result.returncode}")
                print(f"      stderr: {result.stderr[:200]}")
            
            # Parse output for survivors path
            # Convention: coordinator outputs "SURVIVORS_PATH: /path/to/file.json"
            for line in result.stdout.split('\n'):
                if line.startswith("SURVIVORS_PATH:"):
                    return line.split(":", 1)[1].strip()
            
            # Fallback to default
            return self.config.survivors_input
            
        except subprocess.TimeoutExpired:
            print(f"      ❌ Coordinator timed out")
            return self.config.survivors_input
        except Exception as e:
            print(f"      ❌ Coordinator failed: {e}")
            return self.config.survivors_input
    
    def _run_inner_episode(
        self,
        episode_id: str,
        timestamp: str,
        survivors_path: str,
        episode_num: int
    ) -> EpisodeResult:
        """
        Run inner episode (CPU ML training).
        
        Phase 9B.2: Applies policy conditioning before training if enabled.
        
        Args:
            episode_id: Episode identifier
            timestamp: Episode timestamp
            survivors_path: Path to survivors file
            episode_num: Episode number
            
        Returns:
            EpisodeResult with training metrics
        """
        policy_id = f"{self.config.policy_prefix}_{episode_id}"
        
        if not TRAINER_AVAILABLE:
            # Fallback for testing without trainer
            print(f"      ⚠️  InnerEpisodeTrainer not available, using mock")
            return EpisodeResult(
                episode_id=episode_id,
                timestamp=timestamp,
                model_type="mock",
                fitness=0.5,
                val_r2=0.95,
                policy_id=policy_id
            )
        
        # Load survivors
        try:
            survivors_raw = self._load_survivors_raw(survivors_path)
            survivor_count_before = len(survivors_raw)
        except Exception as e:
            print(f"      ❌ Failed to load survivors: {e}")
            return EpisodeResult(
                episode_id=episode_id,
                timestamp=timestamp,
                model_type="error",
                fitness=0.0,
                val_r2=0.0,
                policy_id=policy_id
            )
        
        # =====================================================================
        # PHASE 9B.2: Apply policy conditioning
        # =====================================================================
        conditioning_result: Optional[PolicyConditionedResult] = None
        active_policy_id = None
        active_policy_fingerprint = None
        conditioning_log = []
        
        if self.config.policy_conditioned and POLICY_CONDITIONING_AVAILABLE:
            try:
                print(f"      🔧 Applying policy conditioning...")
                conditioning_result = condition_episode(
                    survivors_raw,
                    project_root=self.config.project_root,
                    force_policy=self.active_policy,
                )
                
                survivors_raw = conditioning_result.conditioned_survivors
                active_policy_id = conditioning_result.active_policy_id
                active_policy_fingerprint = conditioning_result.active_policy_fingerprint
                conditioning_log = conditioning_result.transform_log
                
                print(f"      📊 Conditioned: {conditioning_result.survivors_before} → {conditioning_result.survivors_after} survivors")
                print(f"         Policy: {active_policy_id}")
                print(f"         Fingerprint: {active_policy_fingerprint}")
                
                # Log each transform
                for log_entry in conditioning_log:
                    if not log_entry.startswith("policy_fingerprint"):
                        print(f"         {log_entry}")
                
            except PolicyViolationError as e:
                print(f"      ⚠️  Policy violation: {e}")
                print(f"      ⚠️  Falling back to unconditioned training")
                conditioning_log = [f"FALLBACK: {e}"]
            except Exception as e:
                print(f"      ⚠️  Conditioning failed: {e}")
                print(f"      ⚠️  Falling back to unconditioned training")
                conditioning_log = [f"ERROR: {e}"]
        else:
            if self.config.policy_conditioned:
                print(f"      ℹ️  Policy conditioning requested but modules unavailable")
        
        # Convert to X, y format for trainer
        X, y, feature_names = self._survivors_to_xy(survivors_raw)
        survivor_count = len(survivors_raw)
        
        # Configure trainer (matches TrainerConfig dataclass)
        trainer_config = TrainerConfig(
            model_types=self.config.model_types,
            n_estimators=self.config.n_estimators,
            k_folds=self.config.k_folds,
            n_jobs=self.config.n_jobs
        )
        
        trainer = InnerEpisodeTrainer(trainer_config)
        
        # Train and select best
        start_time = time.time()
        # S83 Phase 8A: Pass enable_diagnostics from config  # S83_PHASE8A_DIAG
        _enable_diag = self.config.__dict__.get('enable_diagnostics', False)
        best_result, all_results = trainer.train_best(X, y, feature_names, enable_diagnostics=_enable_diag)
        training_time_ms = (time.time() - start_time) * 1000
        
        # Log all model results
        for model_name, result in all_results.items():
            if result.success and result.metrics:
                m = result.metrics
                print(f"      {model_name}: R²={m.val_r2:.4f}, fitness={m.fitness:.4f}")
            else:
                print(f"      {model_name}: FAILED - {result.error}")
        
        # Extract metrics from best result
        if not best_result.success or not best_result.metrics:
            raise ValueError(f"Best model training failed: {best_result.error}")
        
        metrics = best_result.metrics
        
        # S83 Phase 8A: Extract diagnostics from best model  # S83_PHASE8A_DIAG
        _best_diag = getattr(best_result, 'diagnostics', None)

        return EpisodeResult(
            episode_id=episode_id,
            timestamp=timestamp,
            model_type=best_result.model_type,
            fitness=metrics.fitness,
            val_r2=metrics.val_r2,
            val_mae=metrics.val_mae,
            fold_std=metrics.fold_std,
            train_val_gap=metrics.train_val_gap,
            training_time_ms=training_time_ms,
            survivor_count=survivor_count,
            survivor_count_before=survivor_count_before,
            feature_count=len(feature_names),
            policy_id=policy_id,
            # Phase 9B.2 fields
            policy_conditioned=self.config.policy_conditioned,
            active_policy_id=active_policy_id,
            active_policy_fingerprint=active_policy_fingerprint,
            conditioning_log=conditioning_log,
            # Phase 8A  # S83_PHASE8A_DIAG
            episode_diagnostics=_best_diag,
        )
    

    # =====================================================  # S83_PHASE8A_DIAG
    # S83 PHASE 8A: Episode Training Trend Detection
    # =====================================================

    def _check_episode_training_trend(self, diagnostics_history: List[Dict]) -> Optional[str]:
        """
        Detect degrading training quality across episodes.
        Ch14 Section 9.2. OBSERVE-ONLY -- no automatic intervention.
        
        Returns:
            'STABLE', 'WARNING_OVERFIT', 'CRITICAL_OVERFIT', 'DEGRADING', or None.
        """
        if len(diagnostics_history) < 3:
            return None
        
        recent = diagnostics_history[-3:]
        ratios = [d.get('best_round_ratio', 0.0) for d in recent]
        
        if ratios[0] > ratios[1] > ratios[2] and ratios[2] < 0.2:
            print(f"      [DIAG] Training quality DEGRADING: "
                  f"best_round_ratios={[f'{r:.2f}' for r in ratios]}")
            if self.telemetry:
                try:
                    self.telemetry.record_event(
                        event_type='training_quality_degrading',
                        details={
                            'recent_ratios': ratios,
                            'recent_severities': [d.get('severity', 'absent') for d in recent],
                        },
                    )
                except Exception:
                    pass
            return 'DEGRADING'
        
        severities = [d.get('severity', 'ok') for d in recent]
        critical_count = sum(1 for s in severities if s == 'critical')
        
        if critical_count >= 2:
            print(f"      [DIAG] Majority critical episodes: "
                  f"{critical_count}/{len(recent)}")
            return 'CRITICAL_OVERFIT'
        
        warning_count = sum(1 for s in severities if s in ('critical', 'warning'))
        if warning_count >= 2:
            return 'WARNING_OVERFIT'
        
        return 'STABLE'

    def _load_survivors_raw(self, path: str) -> List[Dict]:
        """
        Load survivors from JSON file as raw dicts.
        
        Phase 9B.2: Returns raw survivor dicts for policy conditioning.
        
        Returns:
            List of survivor dicts
        """
        with open(path) as f:
            data = json.load(f)
        
        # Handle both list and dict formats
        if isinstance(data, dict) and "survivors" in data:
            survivors = data["survivors"]
        elif isinstance(data, list):
            survivors = data
        else:
            raise ValueError(f"Unknown survivors format in {path}")
        
        if not survivors:
            raise ValueError(f"Empty survivors file: {path}")
        
        return survivors
    
    def _survivors_to_xy(self, survivors: List[Dict]) -> Tuple[Any, Any, List[str]]:
        """
        Convert survivor dicts to X, y arrays for training.
        
        Args:
            survivors: List of survivor dicts
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        import numpy as np
        
        if not survivors:
            raise ValueError("Empty survivors list")
        
        # Extract features
        first = survivors[0]
        if "features" in first:
            features = first["features"]
        else:
            features = {k: v for k, v in first.items() if k not in ["seed", "score", "label", "holdout_hits"]}
        
        feature_names = sorted(features.keys())
        
        # Build X and y
        X = []
        y = []
        
        for survivor in survivors:
            if "features" in survivor:
                feat = survivor["features"]
            else:
                feat = {k: v for k, v in survivor.items() if k not in ["seed", "score", "label", "holdout_hits"]}
            
            row = [feat.get(name, 0) for name in feature_names]
            X.append(row)
            
            # Target: prefer "holdout_hits" (Phase 9B), then "score", fallback to "label"
            target = survivor.get("holdout_hits", survivor.get("score", survivor.get("label", 0)))
            y.append(target)
        
        return np.array(X), np.array(y), feature_names
    
    def _load_survivors(self, path: str) -> Tuple[Any, Any, List[str], int]:
        """
        Load survivors from JSON file.
        
        Legacy method - kept for compatibility.
        
        Returns:
            Tuple of (X, y, feature_names, count)
        """
        survivors = self._load_survivors_raw(path)
        X, y, feature_names = self._survivors_to_xy(survivors)
        return X, y, feature_names, len(survivors)
    
    # =========================================================================
    # CANDIDATE EMISSION
    # =========================================================================
    
    def _emit_candidate(self, result: EpisodeResult) -> None:
        """
        Emit a policy candidate for Chapter 13 review.
        
        Phase 9B.2: Uses policy_conditioned_episode for emission with lineage.
        
        This writes learned_policy_candidate.json.
        The candidate is a HYPOTHESIS, not a DECISION.
        Only Chapter 13 can promote it.
        
        Args:
            result: The episode result to emit as candidate
        """
        print(f"   📤 Emitting policy candidate: {result.policy_id}")
        
        # Calculate fitness delta
        fitness_delta = None
        if self.parent_fitness is not None:
            fitness_delta = result.fitness - self.parent_fitness
            print(f"      Fitness delta: {fitness_delta:+.4f} (vs parent)")
        
        # Get transforms from active policy
        transforms = {}
        if self.active_policy:
            transforms = self.active_policy.get("transforms", {})
        
        # Compute fingerprint
        fingerprint = None
        if POLICY_CONDITIONING_AVAILABLE:
            fingerprint = compute_policy_fingerprint({"transforms": transforms})
        
        candidate = PolicyCandidate(
            schema_version=SCHEMA_VERSION,
            policy_id=result.policy_id or f"policy_{result.episode_id}",
            created_at=datetime.now(timezone.utc).isoformat(),
            source="selfplay",
            model_type=result.model_type,
            fitness=result.fitness,
            val_r2=result.val_r2,
            metrics={
                "val_mae": result.val_mae,
                "fold_std": result.fold_std,
                "train_val_gap": result.train_val_gap,
                "training_time_ms": result.training_time_ms,
                "survivor_count": result.survivor_count,
                "survivor_count_before": result.survivor_count_before,
                "feature_count": result.feature_count,
            },
            parameters={
                "model_type": result.model_type,
                "k_folds": self.config.k_folds,
                "n_jobs": self.config.n_jobs,
            },
            episode_id=result.episode_id,
            survivors_source=self.config.survivors_input,
            # Phase 9B.2: Lineage tracking
            parent_policy_id=result.active_policy_id,
            fitness_delta=fitness_delta,
            fingerprint=fingerprint,
            transforms=transforms,
            status="candidate"  # Chapter 13 decides promotion
        )
        
        # Write candidate file
        output_path = Path(self.config.candidate_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write
        tmp_path = output_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(candidate.to_dict(), f, indent=2)
        os.replace(tmp_path, output_path)
        
        print(f"   ✅ Candidate written to: {output_path}")
        
        # Phase 9B.2: Also write to policy_history/ for audit
        if POLICY_CONDITIONING_AVAILABLE:
            history_dir = Path(self.config.project_root) / "policy_history"
            history_dir.mkdir(exist_ok=True)
            archive_path = history_dir / f"{candidate.policy_id}.json"
            with open(archive_path, 'w') as f:
                json.dump(candidate.to_dict(), f, indent=2)
            print(f"   📁 Archived to: {archive_path}")
        
        # Record policy emission in telemetry
        if self.telemetry:
            self.telemetry.record_policy_emission(
                policy_id=candidate.policy_id,
                fitness=candidate.fitness,
                model_type=candidate.model_type,
                parameters=candidate.parameters,
                source="selfplay"
            )
        
        # Mark result
        result.is_candidate = True
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    def _print_summary(self, results: List[EpisodeResult]) -> None:
        """Print run summary."""
        print(f"\n{'='*60}")
        print(f"📊 Selfplay Run Summary")
        print(f"{'='*60}")
        print(f"   Run ID: {self.run_id}")
        print(f"   Episodes completed: {self.episodes_completed}")
        print(f"   Best fitness: {self.best_fitness:.4f}")
        
        if self.best_episode:
            print(f"   Best model: {self.best_episode.model_type}")
            print(f"   Best episode: {self.best_episode.episode_id}")
        
        candidates = [r for r in results if r.is_candidate]
        print(f"   Candidates emitted: {len(candidates)}")
        
        # Phase 9B.2: Policy conditioning summary
        if self.config.policy_conditioned:
            print(f"\n   Policy Conditioning:")
            print(f"      Active policy: {self.active_policy.get('policy_id') if self.active_policy else 'baseline'}")
            if self.parent_fitness is not None:
                print(f"      Parent fitness: {self.parent_fitness:.4f}")
            if self.best_episode and self.parent_fitness:
                delta = self.best_fitness - self.parent_fitness
                print(f"      Best delta: {delta:+.4f}")
        
        if self.telemetry:
            summary = self.telemetry.get_health_summary()
            print(f"\n   Telemetry status: {summary['status']}")
            if summary['warnings']:
                print(f"   ⚠️  Warnings: {len(summary['warnings'])}")
        
        print(f"{'='*60}\n")


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Selfplay Orchestrator — Air Traffic Controller for Learning"
    )
    
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to selfplay config JSON (default: {DEFAULT_CONFIG_PATH})"
    )
    
    parser.add_argument(
        "--survivors", "-s",
        help="Override survivors input path"
    )
    
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        help="Override max episodes"
    )
    
    parser.add_argument(
        "--single-episode",
        action="store_true",
        help="Run only a single episode (for testing)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config but don't execute"
    )
    
    parser.add_argument(
        "--run-id",
        help="Override run identifier"
    )
    
    parser.add_argument(
        "--min-fitness",
        type=float,
        help="Override minimum fitness threshold for candidates"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="Override CPU threads for ML training"
    )
    
    # Phase 9B.2: New CLI arguments
    parser.add_argument(
        "--policy-conditioned",
        action="store_true",
        default=False,
        help="Enable policy-conditioned learning (Phase 9B.2)"
    )
    
    parser.add_argument(
        "--no-emit-candidate",
        action="store_true",
        default=False,
        help="Disable candidate emission (for testing)"
    )
    
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (for policy files)"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    config_path = Path(args.config)
    if config_path.exists():
        print(f"📂 Loading config from: {config_path}")
        config = SelfplayConfig.from_file(str(config_path))
    else:
        print(f"📝 Using default config (file not found: {config_path})")
        config = SelfplayConfig()
    
    # Apply CLI overrides
    if args.survivors:
        config.survivors_input = args.survivors
    if args.episodes:
        config.max_episodes = args.episodes
    if args.single_episode:
        config.max_episodes = 1
    if args.min_fitness:
        config.min_fitness_threshold = args.min_fitness
    if args.n_jobs:
        config.n_jobs = args.n_jobs
    
    # Phase 9B.2: Apply policy conditioning overrides
    if args.policy_conditioned:
        config.policy_conditioned = True
    if args.no_emit_candidate:
        config.emit_candidates = False
    if args.project_root:
        config.project_root = args.project_root
    
    # Create and run orchestrator
    orchestrator = SelfplayOrchestrator(
        config=config,
        run_id=args.run_id,
        dry_run=args.dry_run
    )
    
    results = orchestrator.run()
    
    # Exit with appropriate code
    if not results and not args.dry_run:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
