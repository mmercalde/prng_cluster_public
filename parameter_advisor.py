#!/usr/bin/env python3
"""
Strategy Advisor — LLM-Guided Selfplay Strategy Analysis.

Version: 1.0.0
Date: 2026-02-07
Contract: CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md
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

OUTPUTS:
    - strategy_recommendation.json (overwritten each cycle)
    - strategy_history/ (archived recommendations)
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
    schema_version: str = "1.0.0"
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

class StrategyAdvisor:
    """Main Strategy Advisor class.
    
    Consumes Chapter 13 diagnostics and telemetry to produce
    strategy_recommendation.json for WATCHER consumption.
    """
    
    GRAMMAR_FILE = "strategy_advisor.gbnf"
    
    def __init__(self, state_dir: str = ".", llm_router=None):
        """Initialize the advisor.
        
        Args:
            state_dir: Directory containing diagnostics, telemetry, policies.
            llm_router: Optional LLMRouter instance for grammar-constrained calls.
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
    
    def analyze(self, force: bool = False) -> Optional[StrategyRecommendation]:
        """Run strategy analysis.
        
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
        
        # Classify focus area
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
        
        # Build recommendation (either via LLM or heuristic fallback)
        if self.llm_router:
            recommendation = self._build_recommendation_llm(
                diagnostics, telemetry, policy_history, metrics, signals,
                primary_focus, primary_conf, secondary_focus, secondary_conf
            )
        else:
            recommendation = self._build_recommendation_heuristic(
                metrics, signals,
                primary_focus, primary_conf, secondary_focus, secondary_conf,
                len(diagnostics)
            )
        
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
            current = set(diagnostics[0].get("top_survivors", []))
            previous = set(diagnostics[1].get("top_survivors", []))
            scs = MetricsComputer.compute_scs(current, previous)
        
        return ComputedMetrics(pcs=pcs, cc=cc, fpd=fpd, mdi=mdi, scs=scs)
    
    def _extract_signals(
        self,
        diagnostics: List[Dict[str, Any]],
        telemetry: List[Dict[str, Any]],
        policy_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Extract diagnostic signals from loaded data."""
        signals = {
            "hit_at_20": 0.0,
            "hit_at_100": 0.0,
            "hit_at_300": 0.0,
            "model_dominance": 0.0,
            "feature_drift": 0.0,
            "window_decay": 0.0,
            "best_model_type": "catboost",
            "draws_since_last_promotion": 0,
        }
        
        # Average hit rates from diagnostics
        if diagnostics:
            for key in ["hit_at_20", "hit_at_100", "hit_at_300"]:
                values = [d.get(key, 0.0) for d in diagnostics if key in d]
                if values:
                    signals[key] = sum(values) / len(values)
            
            # Feature drift and window decay from latest
            signals["feature_drift"] = diagnostics[0].get("feature_drift", 0.0)
            signals["window_decay"] = diagnostics[0].get("window_decay", 0.0)
        
        # Model dominance from telemetry
        if telemetry:
            model_counts = {}
            for t in telemetry:
                model_type = t.get("best_model_type", "unknown")
                model_counts[model_type] = model_counts.get(model_type, 0) + 1
            
            if model_counts:
                total = sum(model_counts.values())
                max_count = max(model_counts.values())
                signals["model_dominance"] = max_count / total
                signals["best_model_type"] = max(model_counts, key=model_counts.get)
        
        # Draws since last promotion
        last_promotion_draw = 0
        for p in policy_history:
            if p.get("status") == "promoted":
                last_promotion_draw = max(last_promotion_draw, p.get("draw_number", 0))
        
        current_draw = diagnostics[0].get("draw_number", 0) if diagnostics else 0
        signals["draws_since_last_promotion"] = current_draw - last_promotion_draw
        
        return signals
    
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
        """Build recommendation using heuristic rules (no LLM)."""
        # Determine action based on focus
        if primary_focus == FocusArea.REGIME_SHIFT:
            action = AdvisorAction.FULL_RESET
            retrain_scope = RetrainScope.FULL_PIPELINE
            risk_level = RiskLevel.HIGH
        elif primary_focus == FocusArea.POOL_COVERAGE:
            action = AdvisorAction.RETRAIN
            retrain_scope = RetrainScope.STEPS_3_5_6
            risk_level = RiskLevel.MEDIUM
        elif primary_focus in (FocusArea.POOL_PRECISION, FocusArea.CONFIDENCE_CALIBRATION):
            action = AdvisorAction.REFOCUS
            retrain_scope = RetrainScope.SELFPLAY_ONLY
            risk_level = RiskLevel.LOW
        else:
            action = AdvisorAction.WAIT
            retrain_scope = None
            risk_level = RiskLevel.LOW
        
        # Classify fitness trend
        if metrics.fpd > 0.5:
            fitness_trend = FitnessTrend.IMPROVING
        elif metrics.fpd < -0.3:
            fitness_trend = FitnessTrend.DECLINING
        elif abs(metrics.fpd) < 0.1:
            fitness_trend = FitnessTrend.PLATEAU
        else:
            fitness_trend = FitnessTrend.VOLATILE
        
        # Build selfplay overrides based on focus
        overrides = self._get_focus_overrides(primary_focus)
        
        # Build pool strategy guidance
        pool_strategy = self._get_pool_strategy(signals)
        
        return StrategyRecommendation(
            draws_analyzed=draws_analyzed,
            focus_area=primary_focus,
            focus_confidence=primary_conf,
            focus_rationale=f"Heuristic classification: {primary_focus.value}",
            secondary_focus=secondary_focus,
            secondary_confidence=secondary_conf,
            recommended_action=action,
            retrain_scope=retrain_scope,
            selfplay_overrides=overrides,
            parameter_proposals=[],  # LLM would fill these
            pool_strategy=pool_strategy,
            risk_level=risk_level,
            requires_human_review=risk_level == RiskLevel.HIGH,
            diagnostic_summary=DiagnosticSummary(
                hit_at_20=signals.get("hit_at_20", 0.0),
                hit_at_100=signals.get("hit_at_100", 0.0),
                hit_at_300=signals.get("hit_at_300", 0.0),
                calibration_correlation=metrics.cc,
                survivor_churn=1.0 - metrics.scs,
                best_model_type=signals.get("best_model_type", "catboost"),
                fitness_trend=fitness_trend,
                draws_since_last_promotion=signals.get("draws_since_last_promotion", 0),
            ),
            alternative_hypothesis="Heuristic fallback — no LLM analysis available",
        )
    
    def _get_focus_overrides(self, focus: FocusArea) -> SelfplayOverrides:
        """Get selfplay overrides for a given focus area."""
        if focus == FocusArea.POOL_PRECISION:
            return SelfplayOverrides(
                max_episodes=15,
                model_types=["catboost", "xgboost"],
                min_fitness_threshold=0.55,
                priority_metrics=["pool_concentration", "model_agreement"],
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
    ) -> StrategyRecommendation:
        """Build recommendation via LLM with grammar constraint."""
        # Import bundle builder
        try:
            from agents.contexts.advisor_bundle import build_advisor_bundle
            prompt = build_advisor_bundle(
                diagnostics=diagnostics,
                telemetry=telemetry,
                policy_history=policy_history,
                metrics=metrics,
                signals=signals,
            )
        except ImportError:
            logger.warning("advisor_bundle not available, using heuristic fallback")
            return self._build_recommendation_heuristic(
                metrics, signals, primary_focus, primary_conf,
                secondary_focus, secondary_conf, len(diagnostics)
            )
        
        # Call LLM with grammar constraint
        try:
            response = self.llm_router.evaluate_with_grammar(
                prompt,
                grammar_file=self.GRAMMAR_FILE,
            )
            
            # Parse response JSON
            rec_data = json.loads(response)
            return StrategyRecommendation(**rec_data)
            
        except Exception as e:
            logger.error("LLM call failed, using heuristic fallback: %s", e)
            return self._build_recommendation_heuristic(
                metrics, signals, primary_focus, primary_conf,
                secondary_focus, secondary_conf, len(diagnostics)
            )
    
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
        
        logger.info("Saved strategy recommendation: focus=%s, action=%s",
                   recommendation.focus_area.value, recommendation.recommended_action.value)
        
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
