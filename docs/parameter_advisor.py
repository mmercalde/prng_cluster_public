#!/usr/bin/env python3
"""
Strategy Advisor — LLM-Guided Selfplay Strategy Analysis.

Version: 1.1.0
Date: 2026-02-08
Contract: CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md (+ Section 8.5)
Authority: Team Beta

PURPOSE:
    Analyze Chapter 13 diagnostics and selfplay telemetry to produce
    mathematically grounded, auditable recommendations for selfplay focus.
    
    The One-Sentence Rule:
    "Chapter 13 measures. The Advisor interprets. Selfplay explores. WATCHER enforces."

AUTHORITY:
    - MAY: Analyze, classify, recommend, rank
    - NEVER: Execute, modify files, bypass WATCHER, promote policies

ACTIVATION GATE:
    - ≥15 real draws in diagnostics_history/
    - ≥10 selfplay episodes with telemetry
    - ≥1 promoted policy exists

LLM DECISION HIERARCHY (Section 8.5):
    1. DeepSeek (primary) — routine analysis, grammar-constrained
    2. Claude (backup) — escalation on low confidence + risky action
    3. Heuristic (emergency) — ONLY when both LLMs unreachable
       MUST log as DEGRADED_MODE warning

OUTPUTS:
    - strategy_recommendation.json (overwritten each cycle)
    - strategy_history/ (archived recommendations)

CHANGELOG:
    v1.0.0 (2026-02-07): Initial implementation per contract
    v1.1.0 (2026-02-08): Lifecycle integration, escalation chain, heuristic demotion
        - Uses llm_lifecycle.ensure_running() before LLM calls
        - Decision-type gated escalation to Claude backup
        - Heuristic demoted to emergency-only with DEGRADED_MODE tagging
        - _build_recommendation_llm() supports use_backup parameter
        - Exceptions propagate instead of silent heuristic fallthrough
        - Added metadata field to StrategyRecommendation
        - Ref: PROPOSAL_STRATEGY_ADVISOR_LIFECYCLE_INTEGRATION_v1_0.md
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class FocusArea(str, Enum):
    """Focus area classification per Contract Section 4.1."""
    POOL_PRECISION = "POOL_PRECISION"
    POOL_COVERAGE = "POOL_COVERAGE"
    CONFIDENCE_CALIBRATION = "CONFIDENCE_CALIBRATION"
    MODEL_DIVERSITY = "MODEL_DIVERSITY"
    FEATURE_RELEVANCE = "FEATURE_RELEVANCE"
    REGIME_SHIFT = "REGIME_SHIFT"
    STEADY_STATE = "STEADY_STATE"


class AdvisorAction(str, Enum):
    """Recommended action types."""
    RETRAIN = "RETRAIN"
    WAIT = "WAIT"
    ESCALATE = "ESCALATE"
    REFOCUS = "REFOCUS"
    FULL_RESET = "FULL_RESET"


class RetrainScope(str, Enum):
    """Scope of recommended retraining."""
    SELFPLAY_ONLY = "selfplay_only"
    STEPS_5_6 = "steps_5_6"
    STEPS_3_5_6 = "steps_3_5_6"
    FULL_PIPELINE = "full_pipeline"


class RiskLevel(str, Enum):
    """Risk assessment levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FitnessTrend(str, Enum):
    """Fitness trajectory classification."""
    IMPROVING = "improving"
    PLATEAU = "plateau"
    DECLINING = "declining"
    VOLATILE = "volatile"


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ParameterProposal(BaseModel):
    """Single parameter adjustment proposal."""
    parameter: str
    current_value: Optional[float] = None
    proposed_value: float
    delta: str  # e.g., "+0.05", "-10", "*1.2"
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class SelfplayOverrides(BaseModel):
    """Selfplay configuration overrides."""
    max_episodes: int = Field(default=10, ge=1, le=50)
    model_types: List[str] = Field(default_factory=lambda: ["catboost", "lightgbm", "xgboost"])
    min_fitness_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    priority_metrics: List[str] = Field(default_factory=lambda: ["pool_concentration"])
    exploration_ratio: float = Field(default=0.3, ge=0.0, le=1.0)
    search_strategy: Optional[str] = None  # bayesian, random, grid, evolutionary


class PoolStrategy(BaseModel):
    """Per-tier pool guidance."""
    tight_pool_guidance: str = "No change needed"
    balanced_pool_guidance: str = "No change needed"
    wide_pool_guidance: str = "No change needed"


class DiagnosticSummary(BaseModel):
    """Summary of computed diagnostic metrics."""
    hit_at_20: float = Field(default=0.0, ge=0.0, le=1.0)
    hit_at_100: float = Field(default=0.0, ge=0.0, le=1.0)
    hit_at_300: float = Field(default=0.0, ge=0.0, le=1.0)
    calibration_correlation: float = Field(default=0.0, ge=-1.0, le=1.0)
    survivor_churn: float = Field(default=0.0, ge=0.0, le=1.0)
    best_model_type: str = "catboost"
    fitness_trend: FitnessTrend = FitnessTrend.PLATEAU
    draws_since_last_promotion: int = 0


class StrategyRecommendation(BaseModel):
    """Complete strategy recommendation output."""
    schema_version: str = "1.1.0"
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    advisor_model: str = "deepseek-r1-14b"
    draws_analyzed: int = 0
    
    focus_area: FocusArea
    focus_confidence: float = Field(ge=0.0, le=1.0)
    focus_rationale: str
    
    secondary_focus: Optional[FocusArea] = None
    secondary_confidence: Optional[float] = None
    
    recommended_action: AdvisorAction
    retrain_scope: Optional[RetrainScope] = None
    
    selfplay_overrides: SelfplayOverrides = Field(default_factory=SelfplayOverrides)
    parameter_proposals: List[ParameterProposal] = Field(default_factory=list, max_length=5)
    pool_strategy: PoolStrategy = Field(default_factory=PoolStrategy)
    
    risk_level: RiskLevel = RiskLevel.LOW
    requires_human_review: bool = False
    
    diagnostic_summary: DiagnosticSummary = Field(default_factory=DiagnosticSummary)
    alternative_hypothesis: Optional[str] = None
    
    # v1.1.0: Metadata for audit trail and degraded mode tracking
    metadata: Optional[Dict[str, Any]] = None


class ComputedMetrics(BaseModel):
    """Mathematical metrics computed per Section 6."""
    pcs: float = Field(default=0.0, description="Pool Concentration Score")
    cc: float = Field(default=0.0, description="Calibration Correlation")
    fpd: float = Field(default=0.0, description="Fitness Plateau Detection")
    mdi: float = Field(default=0.0, description="Model Diversity Index")
    scs: float = Field(default=0.0, description="Survivor Consistency Score")


# ═══════════════════════════════════════════════════════════════════════════════
# ACTIVATION GATE
# ═══════════════════════════════════════════════════════════════════════════════

class ActivationGate:
    """Checks whether the Advisor has sufficient data to activate.
    
    Per Contract Section 8.4:
    - ≥15 real draws in diagnostics_history/
    - ≥10 selfplay episodes with telemetry
    - ≥1 promoted policy exists
    """
    
    def __init__(self, state_dir: str = "."):
        self.state_dir = Path(state_dir)
        self.diagnostics_dir = self.state_dir / "diagnostics_history"
        self.telemetry_dir = self.state_dir / "telemetry"
        self.policy_history_dir = self.state_dir / "policy_history"
    
    def check(self) -> Tuple[bool, Dict[str, Any]]:
        """Check all activation gates.
        
        Returns:
            (can_activate, gate_status) where gate_status shows each gate's value.
        """
        status = {
            "draws_count": self._count_draws(),
            "draws_required": 15,
            "episodes_count": self._count_episodes(),
            "episodes_required": 10,
            "promoted_policies": self._count_promoted_policies(),
            "policies_required": 1,
        }
        
        can_activate = (
            status["draws_count"] >= status["draws_required"] and
            status["episodes_count"] >= status["episodes_required"] and
            status["promoted_policies"] >= status["policies_required"]
        )
        
        return can_activate, status
    
    def _count_draws(self) -> int:
        """Count diagnostic files in diagnostics_history/."""
        if not self.diagnostics_dir.exists():
            return 0
        return len(list(self.diagnostics_dir.glob("*.json")))
    
    def _count_episodes(self) -> int:
        """Count telemetry files in telemetry/."""
        if not self.telemetry_dir.exists():
            return 0
        return len(list(self.telemetry_dir.glob("episode_*.json")))
    
    def _count_promoted_policies(self) -> int:
        """Count promoted policies in policy_history/."""
        if not self.policy_history_dir.exists():
            return 0
        promoted = 0
        for f in self.policy_history_dir.glob("*.json"):
            try:
                with open(f) as fp:
                    policy = json.load(fp)
                    if policy.get("status") == "promoted":
                        promoted += 1
            except (json.JSONDecodeError, IOError):
                continue
        return promoted


# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL METRICS (Section 6)
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsComputer:
    """Computes mathematical metrics per Contract Section 6."""
    
    @staticmethod
    def compute_pcs(weights: List[float], top_k: int = 20) -> float:
        """Pool Concentration Score (Section 6.1).
        
        PCS = Σ(weight_i for i in Top-K) / Σ(weight_i for all i)
        
        Args:
            weights: List of candidate weights (sorted descending).
            top_k: Number of top candidates.
            
        Returns:
            PCS score (0.0-1.0).
        """
        if not weights:
            return 0.0
        total = sum(weights)
        if total == 0:
            return 0.0
        top_sum = sum(weights[:top_k])
        return min(1.0, top_sum / total)
    
    @staticmethod
    def compute_cc(confidences: List[float], hits: List[bool]) -> float:
        """Calibration Correlation (Section 6.2).
        
        CC = Pearson(confidence_bucket_mean, hit_rate_per_bucket)
        
        Args:
            confidences: List of prediction confidence scores.
            hits: List of boolean hit indicators.
            
        Returns:
            Pearson correlation (-1.0 to 1.0).
        """
        if len(confidences) < 10 or len(confidences) != len(hits):
            return 0.0
        
        # Bucket into deciles
        buckets = {}
        for conf, hit in zip(confidences, hits):
            bucket_idx = min(9, int(conf * 10))
            if bucket_idx not in buckets:
                buckets[bucket_idx] = {"hits": 0, "total": 0}
            buckets[bucket_idx]["total"] += 1
            if hit:
                buckets[bucket_idx]["hits"] += 1
        
        # Need at least 3 buckets for meaningful correlation
        if len(buckets) < 3:
            return 0.0
        
        bucket_means = []
        hit_rates = []
        for idx in sorted(buckets.keys()):
            bucket_means.append((idx + 0.5) / 10)  # Bucket center
            hit_rates.append(buckets[idx]["hits"] / buckets[idx]["total"])
        
        # Pearson correlation
        try:
            corr = np.corrcoef(bucket_means, hit_rates)[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)
        except Exception:
            return 0.0
    
    @staticmethod
    def compute_fpd(fitness_values: List[float], window: int = 10) -> float:
        """Fitness Plateau Detection (Section 6.3).
        
        FPD = slope(fitness_values[-window:]) / std(fitness_values[-window:])
        
        Args:
            fitness_values: List of episode fitness scores.
            window: Number of recent values to analyze.
            
        Returns:
            FPD score (negative = regression, 0 = plateau, positive = improving).
        """
        if len(fitness_values) < window:
            return 0.0
        
        recent = fitness_values[-window:]
        std = np.std(recent)
        if std < 0.001:  # Near-zero variance
            return 0.0
        
        # Simple linear regression slope
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        
        return float(slope / std)
    
    @staticmethod
    def compute_mdi(model_type_counts: Dict[str, int]) -> float:
        """Model Diversity Index (Section 6.4).
        
        MDI = 1 - HHI(model_type_fitness_share)
        HHI = Σ(share_i²)
        
        Args:
            model_type_counts: Dict mapping model type to count of best episodes.
            
        Returns:
            MDI score (0.0-1.0, higher = more diverse).
        """
        if not model_type_counts:
            return 0.0
        
        total = sum(model_type_counts.values())
        if total == 0:
            return 0.0
        
        hhi = sum((count / total) ** 2 for count in model_type_counts.values())
        return 1.0 - hhi
    
    @staticmethod
    def compute_scs(current_survivors: set, previous_survivors: set) -> float:
        """Survivor Consistency Score (Section 6.5).
        
        SCS = |intersection| / |union| (Jaccard similarity)
        
        Args:
            current_survivors: Set of current top survivor IDs.
            previous_survivors: Set of previous top survivor IDs.
            
        Returns:
            Jaccard similarity (0.0-1.0).
        """
        if not current_survivors and not previous_survivors:
            return 1.0  # Both empty = stable
        
        intersection = len(current_survivors & previous_survivors)
        union = len(current_survivors | previous_survivors)
        
        if union == 0:
            return 1.0
        
        return intersection / union


# ═══════════════════════════════════════════════════════════════════════════════
# FOCUS AREA CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class FocusAreaClassifier:
    """Classifies system state into focus areas per Section 4.1."""
    
    # Priority order per Section 4.2
    PRIORITY_ORDER = [
        FocusArea.REGIME_SHIFT,
        FocusArea.POOL_COVERAGE,
        FocusArea.CONFIDENCE_CALIBRATION,
        FocusArea.POOL_PRECISION,
        FocusArea.MODEL_DIVERSITY,
        FocusArea.FEATURE_RELEVANCE,
        FocusArea.STEADY_STATE,
    ]
    
    @classmethod
    def classify(
        cls,
        hit_at_20: float,
        hit_at_100: float,
        hit_at_300: float,
        calibration_correlation: float,
        model_dominance: float,  # Percentage of best episodes by top model
        feature_drift: float,
        window_decay: float,
        survivor_churn: float,
    ) -> Tuple[FocusArea, float, Optional[FocusArea], Optional[float]]:
        """Classify into primary and secondary focus areas.
        
        Returns:
            (primary_focus, primary_confidence, secondary_focus, secondary_confidence)
        """
        candidates = []
        
        # REGIME_SHIFT: window_decay > 0.5 AND survivor_churn > 0.4
        if window_decay > 0.5 and survivor_churn > 0.4:
            candidates.append((FocusArea.REGIME_SHIFT, min(0.95, (window_decay + survivor_churn) / 2)))
        
        # POOL_COVERAGE: Hit@300 < 0.85
        if hit_at_300 < 0.85:
            candidates.append((FocusArea.POOL_COVERAGE, min(0.9, 1.0 - hit_at_300)))
        
        # CONFIDENCE_CALIBRATION: CC < 0.3
        if calibration_correlation < 0.3:
            candidates.append((FocusArea.CONFIDENCE_CALIBRATION, min(0.85, 0.9 - calibration_correlation)))
        
        # POOL_PRECISION: Hit@100 > 0.70 but Hit@20 < 0.10
        if hit_at_100 > 0.70 and hit_at_20 < 0.10:
            candidates.append((FocusArea.POOL_PRECISION, min(0.85, hit_at_100 - hit_at_20)))
        
        # MODEL_DIVERSITY: single model > 80%
        if model_dominance > 0.80:
            candidates.append((FocusArea.MODEL_DIVERSITY, min(0.8, model_dominance)))
        
        # FEATURE_RELEVANCE: feature_drift > 0.3
        if feature_drift > 0.3:
            candidates.append((FocusArea.FEATURE_RELEVANCE, min(0.75, feature_drift)))
        
        # Sort by priority order
        def priority_key(item):
            focus, _ = item
            return cls.PRIORITY_ORDER.index(focus)
        
        candidates.sort(key=priority_key)
        
        if not candidates:
            return FocusArea.STEADY_STATE, 0.7, None, None
        
        primary_focus, primary_conf = candidates[0]
        secondary_focus, secondary_conf = (candidates[1] if len(candidates) > 1 else (None, None))
        
        return primary_focus, primary_conf, secondary_focus, secondary_conf


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

class DiagnosticsLoader:
    """Loads Chapter 13 diagnostics and telemetry data."""
    
    def __init__(self, state_dir: str = "."):
        self.state_dir = Path(state_dir)
    
    def load_recent_diagnostics(self, n: int = 20) -> List[Dict[str, Any]]:
        """Load the N most recent diagnostic files."""
        diag_dir = self.state_dir / "diagnostics_history"
        if not diag_dir.exists():
            return []
        
        files = sorted(diag_dir.glob("*.json"), reverse=True)[:n]
        diagnostics = []
        for f in files:
            try:
                with open(f) as fp:
                    diagnostics.append(json.load(fp))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load diagnostic %s: %s", f, e)
        
        return diagnostics
    
    def load_telemetry(self, n: int = 20) -> List[Dict[str, Any]]:
        """Load the N most recent telemetry files."""
        telemetry_dir = self.state_dir / "telemetry"
        if not telemetry_dir.exists():
            return []
        
        files = sorted(telemetry_dir.glob("episode_*.json"), reverse=True)[:n]
        telemetry = []
        for f in files:
            try:
                with open(f) as fp:
                    telemetry.append(json.load(fp))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load telemetry %s: %s", f, e)
        
        return telemetry
    
    def load_policy_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Load recent policy history (promoted + rejected)."""
        policy_dir = self.state_dir / "policy_history"
        if not policy_dir.exists():
            return []
        
        files = sorted(policy_dir.glob("*.json"), reverse=True)[:n]
        policies = []
        for f in files:
            try:
                with open(f) as fp:
                    policies.append(json.load(fp))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load policy %s: %s", f, e)
        
        return policies
    
    def load_watcher_policies(self) -> Dict[str, Any]:
        """Load watcher_policies.json for bounds validation."""
        policy_path = self.state_dir / "watcher_policies.json"
        if not policy_path.exists():
            return {}
        
        try:
            with open(policy_path) as fp:
                return json.load(fp)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load watcher policies: %s", e)
            return {}


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY ADVISOR
# ═══════════════════════════════════════════════════════════════════════════════

# Risky actions that warrant escalation on low confidence
_RISKY_ACTIONS = {AdvisorAction.RETRAIN, AdvisorAction.ESCALATE, AdvisorAction.FULL_RESET}

# Confidence threshold below which risky actions trigger escalation
_ESCALATION_CONFIDENCE_THRESHOLD = 0.3


class StrategyAdvisor:
    """Main Strategy Advisor class.
    
    Consumes Chapter 13 diagnostics and telemetry to produce
    strategy_recommendation.json for WATCHER consumption.
    
    LLM Decision Hierarchy (Section 8.5):
        1. DeepSeek (primary) — grammar-constrained routine analysis
        2. Claude (backup) — escalation on low confidence + risky action
        3. Heuristic (emergency) — ONLY when both LLMs unreachable
    """
    
    GRAMMAR_FILE = "strategy_advisor.gbnf"
    
    def __init__(self, state_dir: str = ".", llm_router=None):
        """Initialize the advisor.
        
        Args:
            state_dir: Directory containing diagnostics, telemetry, policies.
            llm_router: Optional LLMRouter instance for grammar-constrained calls.
                        If None, advisor will attempt lazy import at analysis time.
        """
        self.state_dir = Path(state_dir)
        self.llm_router = llm_router
        self.loader = DiagnosticsLoader(state_dir)
        self.gate = ActivationGate(state_dir)
        
        # Output paths
        self.recommendation_path = self.state_dir / "strategy_recommendation.json"
        self.history_dir = self.state_dir / "strategy_history"
    
    def can_activate(self) -> Tuple[bool, Dict[str, Any]]:
        """Check if advisor has sufficient data to activate."""
        return self.gate.check()
    
    def _get_llm_router(self):
        """Get LLM router (lazy import to avoid circular deps).
        
        Returns router passed at init, or attempts to import the singleton.
        """
        if self.llm_router:
            return self.llm_router
        try:
            from llm_services.llm_router import get_router
            router = get_router()
            logger.debug("LLM router acquired via lazy import")
            return router
        except ImportError:
            logger.debug("llm_router not importable")
            return None
    
    def _get_lifecycle_manager(self):
        """Get LLM lifecycle manager (lazy import to avoid circular deps).
        
        Returns lifecycle manager or None if not available.
        """
        try:
            from llm_services.llm_lifecycle import get_lifecycle_manager
            return get_lifecycle_manager()
        except ImportError:
            logger.debug("llm_lifecycle not importable")
            return None
    
    def analyze(self, force: bool = False) -> Optional[StrategyRecommendation]:
        """Run strategy analysis with LLM-first decision hierarchy.
        
        Decision chain (Section 8.5):
            1. ensure_running() — guarantee LLM availability
            2. DeepSeek (primary) — grammar-constrained analysis
            3. Claude (backup) — on low confidence + risky action
            4. Heuristic (emergency) — ONLY if both LLMs unreachable
        
        Args:
            force: If True, bypass activation gate (for testing).
            
        Returns:
            StrategyRecommendation if successful, None if gate not passed.
        """
        # Check activation gate
        can_activate, gate_status = self.can_activate()
        if not can_activate and not force:
            logger.warning(
                "Advisor activation gate not passed: %s",
                json.dumps(gate_status, indent=2)
            )
            return None
        
        # Load data
        diagnostics = self.loader.load_recent_diagnostics(20)
        telemetry = self.loader.load_telemetry(20)
        policy_history = self.loader.load_policy_history(10)
        watcher_policies = self.loader.load_watcher_policies()
        
        if not diagnostics:
            logger.error("No diagnostics data available")
            return None
        
        # Compute metrics
        metrics = self._compute_metrics(diagnostics, telemetry)
        
        # Extract diagnostic signals
        signals = self._extract_signals(diagnostics, telemetry, policy_history)
        
        # Classify focus area (heuristic pre-classification for context)
        primary_focus, primary_conf, secondary_focus, secondary_conf = (
            FocusAreaClassifier.classify(
                hit_at_20=signals.get("hit_at_20", 0.0),
                hit_at_100=signals.get("hit_at_100", 0.0),
                hit_at_300=signals.get("hit_at_300", 0.0),
                calibration_correlation=metrics.cc,
                model_dominance=signals.get("model_dominance", 0.0),
                feature_drift=signals.get("feature_drift", 0.0),
                window_decay=signals.get("window_decay", 0.0),
                survivor_churn=metrics.scs,
            )
        )
        
        # ── LLM-first decision hierarchy (Section 8.5) ──────────────────
        recommendation = None
        rec_mode = "unknown"
        
        # Step 1: Ensure LLM is available via lifecycle manager
        llm_router = self._get_llm_router()
        lifecycle = self._get_lifecycle_manager()
        
        if lifecycle:
            try:
                lifecycle.ensure_running()
                logger.info("LLM server confirmed available for advisor analysis")
            except Exception as e:
                logger.warning("Lifecycle ensure_running() failed: %s", e)
        
        # Step 2: Try DeepSeek (primary)
        if llm_router:
            try:
                recommendation = self._build_recommendation_llm(
                    diagnostics, telemetry, policy_history, metrics, signals,
                    primary_focus, primary_conf, secondary_focus, secondary_conf,
                    llm_router=llm_router,
                )
                rec_mode = "deepseek_primary"
                
                # Step 3: Decision-type gated escalation
                # Low confidence on risky actions → escalate to Claude
                if (recommendation.focus_confidence < _ESCALATION_CONFIDENCE_THRESHOLD
                        and recommendation.recommended_action in _RISKY_ACTIONS):
                    logger.info(
                        "DeepSeek low confidence (%.2f) on %s — escalating to Claude",
                        recommendation.focus_confidence,
                        recommendation.recommended_action.value,
                    )
                    recommendation = None  # Clear to trigger escalation
                    
            except Exception as e:
                logger.warning("DeepSeek analysis failed: %s — attempting Claude", e)
        
        # Step 4: Try Claude (backup) if DeepSeek failed or low confidence
        if recommendation is None and llm_router:
            try:
                recommendation = self._build_recommendation_llm(
                    diagnostics, telemetry, policy_history, metrics, signals,
                    primary_focus, primary_conf, secondary_focus, secondary_conf,
                    llm_router=llm_router,
                    use_backup=True,
                )
                rec_mode = "claude_backup"
                logger.info("Claude backup produced recommendation: focus=%s, action=%s",
                           recommendation.focus_area.value,
                           recommendation.recommended_action.value)
            except Exception as e:
                logger.warning("Claude escalation also failed: %s", e)
        
        # Step 5: Heuristic — EMERGENCY ONLY
        if recommendation is None:
            logger.warning(
                "DEGRADED_MODE — both LLMs unreachable. "
                "Using heuristic fallback. Decision quality reduced."
            )
            recommendation = self._build_recommendation_heuristic(
                metrics, signals,
                primary_focus, primary_conf, secondary_focus, secondary_conf,
                len(diagnostics),
            )
            rec_mode = "heuristic_degraded"
        
        # Tag recommendation with mode
        recommendation.advisor_model = rec_mode
        if rec_mode == "heuristic_degraded":
            recommendation.metadata = recommendation.metadata or {}
            recommendation.metadata["degraded_reason"] = "both_llms_unreachable"
        
        # Validate against policy bounds
        recommendation = self._validate_bounds(recommendation, watcher_policies)
        
        # Save recommendation
        self._save_recommendation(recommendation)
        
        return recommendation
    
    def _compute_metrics(
        self,
        diagnostics: List[Dict[str, Any]],
        telemetry: List[Dict[str, Any]],
    ) -> ComputedMetrics:
        """Compute Section 6 mathematical metrics."""
        # PCS from latest diagnostic's pool weights
        weights = []
        if diagnostics and "pool_weights" in diagnostics[0]:
            weights = diagnostics[0]["pool_weights"]
        pcs = MetricsComputer.compute_pcs(weights)
        
        # CC from accumulated confidences/hits
        confidences = []
        hits = []
        for d in diagnostics:
            if "predictions" in d:
                for p in d["predictions"]:
                    confidences.append(p.get("confidence", 0.5))
                    hits.append(p.get("hit", False))
        cc = MetricsComputer.compute_cc(confidences, hits)
        
        # FPD from telemetry fitness values
        fitness_values = [t.get("best_fitness", 0.5) for t in telemetry if "best_fitness" in t]
        fpd = MetricsComputer.compute_fpd(fitness_values)
        
        # MDI from telemetry model type distribution
        model_counts = {}
        for t in telemetry:
            model_type = t.get("best_model_type", "unknown")
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
        mdi = MetricsComputer.compute_mdi(model_counts)
        
        # SCS from consecutive diagnostics
        scs = 0.0
        if len(diagnostics) >= 2:
            current_ids = set(diagnostics[0].get("top_survivors", []))
            previous_ids = set(diagnostics[1].get("top_survivors", []))
            scs = MetricsComputer.compute_scs(current_ids, previous_ids)
        
        return ComputedMetrics(pcs=pcs, cc=cc, fpd=fpd, mdi=mdi, scs=scs)
    
    def _extract_signals(
        self,
        diagnostics: List[Dict[str, Any]],
        telemetry: List[Dict[str, Any]],
        policy_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract diagnostic signals for focus classification."""
        latest = diagnostics[0] if diagnostics else {}
        
        return {
            "hit_at_20": latest.get("hit_at_20", 0.0),
            "hit_at_100": latest.get("hit_at_100", 0.0),
            "hit_at_300": latest.get("hit_at_300", 0.0),
            "model_dominance": self._compute_model_dominance(telemetry),
            "feature_drift": latest.get("feature_drift", 0.0),
            "window_decay": latest.get("window_decay", 0.0),
            "draws_since_last_promotion": self._draws_since_promotion(
                diagnostics, policy_history
            ),
        }
    
    def _compute_model_dominance(self, telemetry: List[Dict[str, Any]]) -> float:
        """Compute fraction of episodes won by the most dominant model type."""
        if not telemetry:
            return 0.0
        
        model_counts = {}
        for t in telemetry:
            model_type = t.get("best_model_type", "unknown")
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
        
        if not model_counts:
            return 0.0
        
        total = sum(model_counts.values())
        max_count = max(model_counts.values())
        
        return max_count / total
    
    def _draws_since_promotion(
        self,
        diagnostics: List[Dict[str, Any]],
        policy_history: List[Dict[str, Any]],
    ) -> int:
        """Count draws since the most recent policy promotion."""
        if not policy_history:
            return len(diagnostics)
        
        promoted = [p for p in policy_history if p.get("status") == "promoted"]
        if not promoted:
            return len(diagnostics)
        
        # Assume policies are sorted by recency
        return min(len(diagnostics), promoted[0].get("draws_since", len(diagnostics)))
    
    def _build_recommendation_heuristic(
        self,
        metrics: ComputedMetrics,
        signals: Dict[str, Any],
        primary_focus: FocusArea,
        primary_conf: float,
        secondary_focus: Optional[FocusArea],
        secondary_conf: Optional[float],
        draws_analyzed: int,
    ) -> StrategyRecommendation:
        """Build recommendation using heuristic rules (emergency fallback only).
        
        NOTE: This method should ONLY be called when both LLMs are unreachable.
        It uses simple threshold logic and discards most of the high-dimensional
        signal from the diagnostic data. Decisions made here are tagged as
        heuristic_degraded for audit purposes.
        """
        # Determine action based on focus area
        if primary_focus == FocusArea.REGIME_SHIFT:
            action = AdvisorAction.FULL_RESET
            scope = RetrainScope.FULL_PIPELINE
            risk = RiskLevel.HIGH
        elif primary_focus in {FocusArea.POOL_COVERAGE, FocusArea.POOL_PRECISION}:
            action = AdvisorAction.RETRAIN
            scope = RetrainScope.STEPS_3_5_6
            risk = RiskLevel.MEDIUM
        elif primary_focus == FocusArea.CONFIDENCE_CALIBRATION:
            action = AdvisorAction.REFOCUS
            scope = RetrainScope.STEPS_5_6
            risk = RiskLevel.MEDIUM
        elif primary_focus == FocusArea.MODEL_DIVERSITY:
            action = AdvisorAction.REFOCUS
            scope = RetrainScope.SELFPLAY_ONLY
            risk = RiskLevel.LOW
        elif primary_focus == FocusArea.STEADY_STATE:
            action = AdvisorAction.WAIT
            scope = None
            risk = RiskLevel.LOW
        else:
            action = AdvisorAction.WAIT
            scope = None
            risk = RiskLevel.MEDIUM
        
        # Build selfplay overrides
        selfplay_overrides = self._get_default_overrides(primary_focus)
        
        # Build pool strategy
        pool_strategy = self._get_pool_strategy(signals)
        
        # Build diagnostic summary
        diagnostic_summary = DiagnosticSummary(
            hit_at_20=signals.get("hit_at_20", 0.0),
            hit_at_100=signals.get("hit_at_100", 0.0),
            hit_at_300=signals.get("hit_at_300", 0.0),
            calibration_correlation=metrics.cc,
            survivor_churn=metrics.scs,
            fitness_trend=self._classify_fitness_trend(metrics.fpd),
            draws_since_last_promotion=signals.get("draws_since_last_promotion", 0),
        )
        
        return StrategyRecommendation(
            focus_area=primary_focus,
            focus_confidence=primary_conf,
            focus_rationale=f"Heuristic classification: {primary_focus.value} "
                           f"(threshold-based, no LLM analysis available)",
            secondary_focus=secondary_focus,
            secondary_confidence=secondary_conf,
            recommended_action=action,
            retrain_scope=scope,
            selfplay_overrides=selfplay_overrides,
            pool_strategy=pool_strategy,
            risk_level=risk,
            requires_human_review=(risk == RiskLevel.HIGH),
            diagnostic_summary=diagnostic_summary,
            draws_analyzed=draws_analyzed,
        )
    
    def _classify_fitness_trend(self, fpd: float) -> FitnessTrend:
        """Classify fitness trend from FPD score."""
        if fpd > 0.5:
            return FitnessTrend.IMPROVING
        elif fpd < -0.5:
            return FitnessTrend.DECLINING
        elif abs(fpd) < 0.1:
            return FitnessTrend.PLATEAU
        else:
            return FitnessTrend.VOLATILE
    
    def _get_default_overrides(self, focus: FocusArea) -> SelfplayOverrides:
        """Get default selfplay overrides for a focus area."""
        if focus == FocusArea.POOL_PRECISION:
            return SelfplayOverrides(
                max_episodes=15,
                model_types=["catboost", "xgboost"],
                min_fitness_threshold=0.55,
                priority_metrics=["pool_concentration", "top_k_accuracy"],
                exploration_ratio=0.2,
            )
        elif focus == FocusArea.POOL_COVERAGE:
            return SelfplayOverrides(
                max_episodes=20,
                model_types=["catboost", "lightgbm", "xgboost", "neural_net"],
                min_fitness_threshold=0.4,
                priority_metrics=["coverage", "diversity"],
                exploration_ratio=0.5,
            )
        elif focus == FocusArea.CONFIDENCE_CALIBRATION:
            return SelfplayOverrides(
                max_episodes=12,
                model_types=["catboost", "lightgbm", "xgboost"],
                min_fitness_threshold=0.5,
                priority_metrics=["fold_stability", "train_val_gap"],
                exploration_ratio=0.35,
            )
        elif focus == FocusArea.MODEL_DIVERSITY:
            return SelfplayOverrides(
                max_episodes=16,
                model_types=["neural_net", "lightgbm", "xgboost"],  # Exclude dominant
                min_fitness_threshold=0.45,
                priority_metrics=["model_diversity", "ensemble_agreement"],
                exploration_ratio=0.4,
            )
        elif focus == FocusArea.STEADY_STATE:
            return SelfplayOverrides(
                max_episodes=5,
                model_types=["catboost", "lightgbm"],
                min_fitness_threshold=0.6,
                priority_metrics=["stability"],
                exploration_ratio=0.15,
            )
        else:
            # REGIME_SHIFT, FEATURE_RELEVANCE — pause selfplay
            return SelfplayOverrides(max_episodes=0)
    
    def _get_pool_strategy(self, signals: Dict[str, Any]) -> PoolStrategy:
        """Generate pool-specific guidance."""
        tight = "No change needed"
        balanced = "No change needed"
        wide = "No change needed"
        
        if signals["hit_at_20"] < 0.05:
            tight = "Increase weight on survivor consistency and model agreement"
        
        if signals["hit_at_100"] < 0.60:
            balanced = "Diversify model types and increase episode count"
        
        if signals["hit_at_300"] < 0.85:
            wide = "Possible regime shift — consider full pipeline rerun"
        
        return PoolStrategy(
            tight_pool_guidance=tight,
            balanced_pool_guidance=balanced,
            wide_pool_guidance=wide,
        )
    
    def _build_recommendation_llm(
        self,
        diagnostics: List[Dict[str, Any]],
        telemetry: List[Dict[str, Any]],
        policy_history: List[Dict[str, Any]],
        metrics: ComputedMetrics,
        signals: Dict[str, Any],
        primary_focus: FocusArea,
        primary_conf: float,
        secondary_focus: Optional[FocusArea],
        secondary_conf: Optional[float],
        llm_router=None,
        use_backup: bool = False,
    ) -> StrategyRecommendation:
        """Build recommendation via LLM with grammar constraint.
        
        v1.1.0: Exceptions propagate to caller for escalation chain handling.
        No internal heuristic fallback — caller owns fallback policy.
        
        Args:
            llm_router: LLM router instance (required).
            use_backup: If True, route to Claude via force_backup=True.
            
        Raises:
            ImportError: If advisor_bundle is not importable.
            Exception: If LLM call fails (caller handles escalation).
        """
        router = llm_router or self.llm_router
        if not router:
            raise RuntimeError("No LLM router available for recommendation")
        
        # Import bundle builder — let ImportError propagate
        from agents.contexts.advisor_bundle import build_advisor_bundle
        prompt = build_advisor_bundle(
            diagnostics=diagnostics,
            telemetry=telemetry,
            policy_history=policy_history,
            metrics=metrics,
            signals=signals,
        )
        
        # Call LLM with grammar constraint — let exceptions propagate
        response = router.evaluate_with_grammar(
            prompt,
            grammar_file=self.GRAMMAR_FILE,
            force_backup=use_backup,
        )
        
        # Parse response JSON
        rec_data = json.loads(response)
        return StrategyRecommendation(**rec_data)
    
    def _validate_bounds(
        self,
        recommendation: StrategyRecommendation,
        watcher_policies: Dict[str, Any],
    ) -> StrategyRecommendation:
        """Validate parameter proposals against policy bounds."""
        if not recommendation.parameter_proposals:
            return recommendation
        
        bounds = watcher_policies.get("parameter_bounds", {})
        valid_proposals = []
        
        for proposal in recommendation.parameter_proposals:
            param = proposal.parameter
            if param not in bounds:
                logger.warning("Parameter %s not in bounds whitelist — rejected", param)
                continue
            
            param_bounds = bounds[param]
            value = proposal.proposed_value
            
            if "min" in param_bounds and value < param_bounds["min"]:
                logger.warning("Parameter %s value %s below min %s — rejected",
                             param, value, param_bounds["min"])
                continue
            
            if "max" in param_bounds and value > param_bounds["max"]:
                logger.warning("Parameter %s value %s above max %s — rejected",
                             param, value, param_bounds["max"])
                continue
            
            valid_proposals.append(proposal)
        
        # Create new recommendation with validated proposals
        return recommendation.model_copy(update={"parameter_proposals": valid_proposals})
    
    def _save_recommendation(self, recommendation: StrategyRecommendation) -> None:
        """Save recommendation to file and archive."""
        # Save current recommendation
        with open(self.recommendation_path, "w") as f:
            f.write(recommendation.model_dump_json(indent=2))
        
        logger.info("Saved strategy recommendation: focus=%s, action=%s, mode=%s",
                   recommendation.focus_area.value,
                   recommendation.recommended_action.value,
                   recommendation.advisor_model)
        
        # Archive to history
        self.history_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archive_path = self.history_dir / f"recommendation_{timestamp}.json"
        shutil.copy(self.recommendation_path, archive_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point for Strategy Advisor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Strategy Advisor — LLM-Guided Selfplay Analysis")
    parser.add_argument("--state-dir", default=".", help="Directory containing diagnostics/telemetry")
    parser.add_argument("--force", action="store_true", help="Bypass activation gate (testing)")
    parser.add_argument("--check-gate", action="store_true", help="Only check activation gate")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    advisor = StrategyAdvisor(state_dir=args.state_dir)
    
    if args.check_gate:
        can_activate, status = advisor.can_activate()
        print(json.dumps({"can_activate": can_activate, **status}, indent=2))
        return 0 if can_activate else 1
    
    recommendation = advisor.analyze(force=args.force)
    
    if recommendation:
        print(recommendation.model_dump_json(indent=2))
        return 0
    else:
        print("Advisor did not produce a recommendation (gate not passed or no data)")
        return 1


if __name__ == "__main__":
    exit(main())
