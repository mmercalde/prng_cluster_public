#!/usr/bin/env python3
"""
Bundle Factory — Step Awareness Bundle Assembly Engine.

Version: 1.1.0
Date: 2026-02-05 (Session 60)
Previous: 1.0.0 (2026-02-01, Session 57)
Spec: PROPOSAL_PHASE7_INFRA_HYBRID_v1.2.0, Addendum A
Authority: Team Alpha + Team Beta joint approval

PURPOSE:
    Single entry point for all LLM context assembly.
    Dispatch code MUST only do this:

        bundle = build_step_awareness_bundle(step_id, run_id, state_paths, budgets)
        prompt = render_prompt_from_bundle(bundle)
        llm_output = call_llm(prompt, grammar=bundle.context.grammar_name)

    No inline context logic. No shortcuts.

DESIGN RULES (non-negotiable):
    1. Controller builds bundle (LLM never pulls files)
    2. Bundle is deterministic (same state â†’ same bundle)
    3. Bundle is immutable (no in-prompt "edits" allowed)
    4. Mission + schema + grammar are always included (Tier 0)
    5. History comes from Chapter 13-curated summaries by default
    6. Per-step token budget enforced
    7. No embeddings / vector DB / GPU resident services

ARCHITECTURE:
    Wraps existing build_full_context() — does NOT replace it.
    Adds: mission statements, grammar resolution, Ch.13 history (stubbed),
    token budget enforcement, provenance hashing.

    Existing flow:
        build_full_context(step, results, ...) â†’ FullAgentContext â†’ to_llm_prompt()

    New flow:
        build_step_awareness_bundle(step_id, ...) â†’ StepAwarenessBundle
            â”œâ”€â”€ internally calls build_full_context() for step eval data
            â”œâ”€â”€ adds mission, schema, grammar, guardrails (Tier 0)
            â”œâ”€â”€ adds inputs summary, manifest expectations (Tier 1)
            â”œâ”€â”€ adds Ch.13 history, trends, incidents (Tier 2, stubbed)
            â””â”€â”€ enforces token budgets

        render_prompt_from_bundle(bundle) â†’ str
            â””â”€â”€ structured prompt with tiered sections

RETRIEVAL LAYER (Track 2 — stubbed):
    _retrieve_recent_outcomes() â†’ [] (will read Chapter 13 summaries)
    _retrieve_trend_summary() â†’ {} (will compute diagnostic deltas)
    _retrieve_open_incidents() â†’ [] (will read watcher_failures.jsonl)

    These stubs return empty data. The prompt renders without them.
    Track 2 fills them in behind the same API — zero dispatch rework.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN MISSION (Global Context for All LLM Calls)
# ═══════════════════════════════════════════════════════════════════════════════

MAIN_MISSION = (
    "You are an evaluator within a distributed PRNG analysis system that uses "
    "functional mimicry — learning surface-level output patterns and statistical "
    "heuristics from PRNG-generated sequences to predict future draws, rather than "
    "attempting to discover or reconstruct actual seeds. The system operates a "
    "6-step pipeline (window optimization → sieve → scoring → ML architecture → "
    "anti-overfit training → prediction generation) across a 26-GPU cluster. "
    "Success is measured by hit rate and confidence calibration on held-out draws. "
    "All evaluations serve this goal: improving the system's ability to learn "
    "exploitable patterns in PRNG output through iterative ML refinement, "
    "selfplay exploration, and autonomous feedback loops."
)

# ═════════════════════════════════════════════════════════════════════════════
# STEP MISSION STATEMENTS
# ═════════════════════════════════════════════════════════════════════════════

STEP_MISSIONS: Dict[int, str] = {
    1: (
        "Step 1 — Window Optimizer: Find optimal window parameters (size, offset, "
        "skip range, thresholds) using Bayesian optimization (Optuna TPE). "
        "Delegates real sieving to coordinator.py across 26 GPUs. "
        "Output: bidirectional_survivors.json + optimal_window_config.json. "
        "Key metric: bidirectional survivor count (higher = more candidates for ML)."
    ),
    2: (
        "Step 2/2.5 — Scorer Meta-Optimizer: Distributed hyperparameter tuning "
        "for the scoring model across 26 GPUs using Optuna. "
        "Optimizes residue modular arithmetic, hidden layers, dropout, learning rate. "
        "Output: optimal_scorer_config.json. "
        "Key metric: scorer accuracy on holdout data."
    ),
    3: (
        "Step 3 — Full Scoring: Distributed 46-feature extraction and scoring "
        "of all survivors across 26 GPUs via scripts_coordinator.py. "
        "Extracts per-seed features (50) + global features (14) = 64 total. "
        "Output: survivors_with_scores.json + NPZ with full feature matrix. "
        "Key metrics: completion rate, feature dimensions, score distribution std. "
        "CRITICAL: holdout label integrity must be preserved (no leakage)."
    ),
    4: (
        "Step 4 — ML Meta-Optimizer: Adaptive neural architecture optimization "
        "and capacity planning. Does NOT consume survivor data directly. "
        "Output: reinforcement_engine_config.json with pool sizing and architecture. "
        "Key metric: validation R² on architecture search."
    ),
    5: (
        "Step 5 — Anti-Overfit Training: K-fold cross-validated model training "
        "with 4 model types (neural_net, xgboost, lightgbm, catboost). "
        "FIRST step to consume survivors_with_scores.json + holdout_hits. "
        "Output: best_model checkpoint + best_model.meta.json sidecar. "
        "Key metrics: val_r2, test_r2, overfit_ratio (<1.5 required). "
        "CRITICAL: fail hard on feature schema mismatch."
    ),
    6: (
        "Step 6 — Prediction Generator: Generate final predictions using "
        "trained model loaded via sidecar metadata (best_model.meta.json). "
        "Validates feature_schema_hash before inference. "
        "Output: prediction_pool.json with confidence-ranked candidates. "
        "Key metrics: prediction confidence, pool size (tight/balanced/wide). "
        "CONTRACT: sidecar-only model loading — no direct checkpoint access."
    ),
}

CHAPTER_13_MISSION = (
    "Chapter 13 — Live Feedback Loop: Monitors for new draws, runs diagnostics, "
    "evaluates retrain triggers, queries LLM advisor for analysis, validates "
    "proposals through acceptance engine, executes approved learning loops. "
    "SOLE AUTHORITY for policy promotion. Selfplay candidates are hypotheses "
    "until Chapter 13 promotes them. Ground truth isolation: only Chapter 13 "
    "accesses live draw outcomes."
)


# ═════════════════════════════════════════════════════════════════════════════
# STEP SCHEMA EXCERPTS (human-readable Pydantic summaries)
# ═════════════════════════════════════════════════════════════════════════════

STEP_SCHEMA_EXCERPTS: Dict[int, str] = {
    1: (
        "WindowOptimizerContext: key_metrics=[survivor_count, bidirectional_count, "
        "forward_count, reverse_count, precision, recall]. "
        "Thresholds: excellent(>10K survivors), good(>5K), acceptable(>1K), fail(<100)."
    ),
    2: (
        "ScorerMetaContext: key_metrics=[best_accuracy, completed_trials, "
        "convergence_rate]. Thresholds: excellent(acc>0.85), good(>0.75), "
        "acceptable(>0.65), fail(<0.50)."
    ),
    3: (
        "FullScoringContext: key_metrics=[completion_rate, survivors_scored, "
        "feature_dimensions, mean_score, score_std, top_candidates]. "
        "Thresholds: completion excellent(100%), fail(<90%). "
        "score_std excellent(0.15-0.30), fail(<0.05 = no signal)."
    ),
    4: (
        "MLMetaContext: key_metrics=[best_r2, validation_r2, architecture_stability]. "
        "Thresholds: excellent(r2>0.7), good(>0.5), fail(<0.2)."
    ),
    5: (
        "AntiOverfitContext: key_metrics=[val_r2, test_r2, overfit_ratio, "
        "feature_importance_entropy]. "
        "Thresholds: overfit_ratio excellent(<1.2), fail(>1.5). "
        "val_r2 excellent(>0.6), fail(<0.1)."
    ),
    6: (
        "PredictionContext: key_metrics=[prediction_count, mean_confidence, "
        "pool_coverage, feature_hash_valid]. "
        "Thresholds: confidence excellent(>0.7), fail(<0.3). "
        "feature_hash_valid must be True."
    ),
}

CHAPTER_13_SCHEMA_EXCERPT = (
    "Chapter13Orchestrator: evaluates TriggerAction=[RETRAIN, WAIT, ESCALATE, FULL_RESET]. "
    "LLM proposal via LLMProposal Pydantic model: recommended_action, confidence, "
    "parameter_proposals[], reasoning. Acceptance engine validates against "
    "watcher_policies.json bounds. Promotion requires Chapter 13 sole authority."
)


# ═════════════════════════════════════════════════════════════════════════════
# STEP-TO-GRAMMAR MAPPING
# ═════════════════════════════════════════════════════════════════════════════

STEP_GRAMMAR_NAMES: Dict[int, str] = {
    1: "agent_decision.gbnf",
    2: "sieve_analysis.gbnf",
    3: "agent_decision.gbnf",
    4: "agent_decision.gbnf",
    5: "agent_decision.gbnf",
    6: "agent_decision.gbnf",
}

CHAPTER_13_GRAMMAR = "chapter_13.gbnf"


# ═════════════════════════════════════════════════════════════════════════════
# STEP GUARDRAILS (per-step reminders injected into prompt)
# ═════════════════════════════════════════════════════════════════════════════

STEP_GUARDRAILS: Dict[int, List[str]] = {
    1: [
        "Low thresholds (0.01-0.15) maximize seed discovery — do not recommend high thresholds.",
        "Bidirectional intersection handles filtering, not individual thresholds.",
    ],
    2: [
        "Scorer accuracy must be measured on holdout data, not training data.",
    ],
    3: [
        "Holdout label integrity is paramount — no information leakage from holdout to training.",
        "Feature count must be exactly 64 (50 per-seed + 14 global) for schema compatibility.",
        "NPZ v3.0 must preserve all 22 metadata fields.",
    ],
    4: [
        "Step 4 does NOT consume survivor data directly — it plans architecture only.",
    ],
    5: [
        "Overfit ratio >1.5 is a hard fail — do not recommend proceeding.",
        "Feature schema hash must match between training and inference.",
        "All 4 model types (neural_net, xgboost, lightgbm, catboost) should be compared.",
    ],
    6: [
        "Model loading is sidecar-only — never load checkpoints directly.",
        "Feature schema hash must be validated before inference.",
        "Prediction pool sizing: tight(100), balanced(500), wide(1000).",
    ],
}

CHAPTER_13_GUARDRAILS = [
    "Selfplay outputs are HYPOTHESES until Chapter 13 promotes them.",
    "Promotion authority is Chapter 13 ONLY — never recommend self-promotion.",
    "Ground truth (live draws) is accessible to Chapter 13 ONLY.",
    "WATCHER executes; Chapter 13 decides; selfplay explores.",
]

# ═══════════════════════════════════════════════════════════════════════════════
# SELFPLAY EVALUATION CONTEXT (step_id=99)
# ═══════════════════════════════════════════════════════════════════════════════

SELFPLAY_EVALUATION_MISSION = (
    "Selfplay Evaluation: Assess the quality and strategic value of selfplay "
    "episode outcomes. Selfplay generates HYPOTHETICAL policy candidates by "
    "exploring parameter variations (feature weights, pool sizing, model "
    "architecture choices) against historical draw data ONLY — it never sees "
    "live ground truth. Evaluate candidates on: fitness score trajectory "
    "(improving across episodes indicates productive exploration), diversity "
    "of parameter space explored (avoid mode collapse into narrow optima), "
    "alignment with the current pipeline's bottleneck (does the candidate "
    "address the weakest link?), and feasibility of integration (can the "
    "proposed parameters be adopted without destabilizing existing models). "
    "Your evaluation informs Chapter 13's promotion decision — you do NOT "
    "promote candidates yourself."
)

SELFPLAY_SCHEMA_EXCERPT = (
    "AgentDecision (shared with Steps 1-6): key_fields=[decision, confidence, reasoning]. "
    "decision enum: proceed (candidate worth promoting to Chapter 13), "
    "retry (selfplay should re-explore with adjusted parameters), "
    "escalate (results anomalous or outside expected bounds — flag for human review). "
    "confidence: 0.0-1.0 in your evaluation. "
    "reasoning: Include assessment of candidate quality (strong/moderate/weak/reject), "
    "exploration breadth (mode collapse risk), fitness trajectory (improving/plateau/declining), "
    "bottleneck alignment, and integration risk. These inform Chapter 13's promotion decision."
)

SELFPLAY_GRAMMAR = "agent_decision.gbnf"  # Reuse existing decision grammar

SELFPLAY_GUARDRAILS = [
    "Selfplay candidates are HYPOTHESES — you evaluate, Chapter 13 promotes.",
    "Selfplay uses historical data ONLY — never assume it validated against live draws.",
    "Mode collapse (all episodes converging to same parameters) is a critical warning.",
    "A candidate that improves one model type but degrades others needs careful weighting.",
    "Exploration breadth matters: a moderate-fitness diverse search beats a high-fitness narrow one.",
    "Do not penalize candidates for low absolute fitness — trajectory matters more than level.",
    "WATCHER executes; Chapter 13 decides; selfplay explores. You evaluate.",
]




# ═════════════════════════════════════════════════════════════════════════════
# CONTRACT REFERENCES
# ═════════════════════════════════════════════════════════════════════════════

AUTHORITY_CONTRACTS = [
    "CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md",
]


# ═════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═════════════════════════════════════════════════════════════════════════════

class TokenBudget(BaseModel):
    """Token budget policy for bundle assembly.

    Tier 0: Mission + schema + grammar + guardrails (always included)
    Tier 1: Step inputs summary + manifest expectations (if available)
    Tier 2: Chapter 13 history + trends + incidents (fill remaining)
    """

    ctx_max_tokens: int = Field(default=32768, description="LLM context window size")
    tier0_reserved_tokens: int = Field(
        default=1200,
        description="Reserved for mission + schema + grammar + guardrails"
    )
    tier1_cap_tokens: int = Field(
        default=3000,
        description="Cap for step inputs + manifest expectations"
    )
    history_cap_tokens: int = Field(
        default=6000,
        description="Cap for Chapter 13 curated history"
    )
    telemetry_cap_tokens: int = Field(
        default=3000,
        description="Cap for trend summaries + incidents"
    )
    generation_reserve_tokens: int = Field(
        default=2000,
        description="Reserved for LLM output generation"
    )

    @property
    def tier2_available(self) -> int:
        """Tokens available for Tier 2 (history + telemetry) after Tier 0+1."""
        used = (
            self.tier0_reserved_tokens
            + self.tier1_cap_tokens
            + self.generation_reserve_tokens
        )
        remaining = self.ctx_max_tokens - used
        return max(0, min(remaining, self.history_cap_tokens + self.telemetry_cap_tokens))


class ProvenanceRecord(BaseModel):
    """Audit trail for bundle source data."""

    path: str
    sha256: str = ""
    size_bytes: int = 0

    @classmethod
    def from_file(cls, filepath: str) -> "ProvenanceRecord":
        """Create provenance record with SHA256 hash of file contents."""
        abs_path = os.path.abspath(filepath)
        if not os.path.isfile(abs_path):
            return cls(path=filepath, sha256="FILE_NOT_FOUND", size_bytes=0)

        try:
            size = os.path.getsize(abs_path)
            h = hashlib.sha256()
            with open(abs_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    h.update(chunk)
            return cls(path=filepath, sha256=h.hexdigest(), size_bytes=size)
        except (IOError, OSError) as e:
            logger.warning("Provenance hash failed for %s: %s", filepath, e)
            return cls(path=filepath, sha256="HASH_ERROR", size_bytes=0)


class OutcomeRecord(BaseModel):
    """Structured outcome from Chapter 13 curated history.

    Team Beta decision: structured outcomes, not free-text narration.
    """

    step: int = 0
    run_id: str = ""
    result: str = ""  # "improved", "degraded", "stable", "failed"
    metric_delta: float = 0.0
    key_metric: str = ""
    timestamp: str = ""


class TrendSummary(BaseModel):
    """Computed trend from recent runs."""

    metric_name: str = ""
    direction: str = ""  # "improving", "declining", "stable", "volatile"
    recent_values: List[float] = Field(default_factory=list)
    window_size: int = 0


class BundleContext(BaseModel):
    """The context payload inside a StepAwarenessBundle.

    Organized by tier priority for token budget enforcement.
    """

    # â”€â”€ Tier 0: Always included â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    mission: str = ""
    schema_excerpt: str = ""
    grammar_name: str = ""
    contracts: List[str] = Field(default_factory=list)
    guardrails: List[str] = Field(default_factory=list)

    # â”€â”€ Tier 1: Include if available â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    inputs_summary: Dict[str, Any] = Field(default_factory=dict)
    evaluation_summary: Dict[str, Any] = Field(default_factory=dict)

    # â”€â”€ Tier 2: Fill remaining tokens (Track 2 retrieval) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    recent_outcomes: List[OutcomeRecord] = Field(default_factory=list)
    trend_summary: List[TrendSummary] = Field(default_factory=list)
    open_incidents: List[str] = Field(default_factory=list)


class StepAwarenessBundle(BaseModel):
    """Immutable, deterministic, token-budgeted context bundle.

    This is the ONLY object that dispatch functions pass to the prompt renderer.
    It is assembled by the controller and never modified by the LLM.
    """

    bundle_version: str = "1.1.0"
    step_id: int
    step_name: str = ""
    run_id: str = ""
    is_chapter_13: bool = False

    context: BundleContext = Field(default_factory=BundleContext)
    budgets: TokenBudget = Field(default_factory=TokenBudget)

    provenance: List[ProvenanceRecord] = Field(default_factory=list)
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    model_config = {"frozen": True}  # Immutable after creation


# ═════════════════════════════════════════════════════════════════════════════
# RETRIEVAL STUBS (Track 2 — will be replaced with structured retriever)
# ═════════════════════════════════════════════════════════════════════════════

def _retrieve_recent_outcomes(step_id: int, max_results: int = 5) -> List[OutcomeRecord]:
    """Retrieve recent Chapter 13-curated outcomes for this step.

    Track 2 implementation will:
        - Read chapter13/summaries/*.json
        - Filter by step_id
        - Return last N outcomes with structured labels
        - Use Chapter 13 evaluation, NOT raw decision logs

    Returns:
        Empty list (Track 2 stub).
    """
    # TODO(Track 2): Implement structured retrieval from Chapter 13 summaries
    return []


def _retrieve_trend_summary(step_id: int) -> List[TrendSummary]:
    """Retrieve computed metric trends for this step.

    Track 2 implementation will:
        - Read run_history.jsonl
        - Compute rolling deltas for key metrics
        - Return direction + recent values

    Returns:
        Empty list (Track 2 stub).
    """
    # TODO(Track 2): Implement trend computation from run_history.jsonl
    return []


def _retrieve_open_incidents(step_id: int) -> List[str]:
    """Retrieve active incidents relevant to this step.

    Track 2 implementation will:
        - Read watcher_failures.jsonl
        - Filter for unresolved incidents
        - Return human-readable summaries

    Returns:
        Empty list (Track 2 stub).
    """
    # TODO(Track 2): Implement incident retrieval from watcher_failures.jsonl
    return []


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def _estimate_tokens(text: str) -> int:
    """Estimate token count using word-based approximation.

    Rule of thumb: ~1.3 tokens per word for structured/technical text.
    This avoids requiring tiktoken as a dependency.

    Args:
        text: String to estimate.

    Returns:
        Approximate token count.
    """
    if not text:
        return 0
    words = len(text.split())
    return int(words * 1.3)


def _truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget.

    Args:
        text: Text to potentially truncate.
        max_tokens: Maximum allowed tokens.

    Returns:
        Original text if within budget, truncated with marker otherwise.
    """
    estimated = _estimate_tokens(text)
    if estimated <= max_tokens:
        return text

    # Approximate word count for target
    target_words = int(max_tokens / 1.3)
    words = text.split()
    if target_words >= len(words):
        return text

    truncated = " ".join(words[:target_words])
    return truncated + "\n[... truncated to fit token budget ...]"


# ═════════════════════════════════════════════════════════════════════════════
# BUNDLE ASSEMBLY
# ═════════════════════════════════════════════════════════════════════════════

def build_step_awareness_bundle(
    step_id: int,
    run_id: str = "",
    results: Optional[Dict[str, Any]] = None,
    run_number: int = 1,
    manifest_path: Optional[str] = None,
    state_paths: Optional[List[str]] = None,
    budgets: Optional[TokenBudget] = None,
    is_chapter_13: bool = False,
) -> StepAwarenessBundle:
    """Build an immutable Step Awareness Bundle for LLM evaluation.

    This is the SINGLE ENTRY POINT for all LLM context assembly.
    Dispatch functions call this, then render_prompt_from_bundle().

    Args:
        step_id: Pipeline step number (1-6) or 13 for Chapter 13.
        run_id: Unique run identifier (auto-generated if empty).
        results: Step output results dict to evaluate.
        run_number: Current run number for this step.
        manifest_path: Path to step's agent manifest JSON.
        state_paths: Additional file paths to include in provenance.
        budgets: Token budget override (uses defaults if None).
        is_chapter_13: True if this is a Chapter 13 evaluation context.

    Returns:
        StepAwarenessBundle — immutable, deterministic, token-budgeted.
    """
    if not run_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"step{step_id}_{timestamp}"

    if budgets is None:
        budgets = TokenBudget()

    if results is None:
        results = {}

    # â”€â”€ Step name resolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    step_names = {
        1: "window_optimizer",
        2: "scorer_meta_optimizer",
        3: "full_scoring",
        4: "ml_meta_optimizer",
        5: "anti_overfit_training",
        6: "prediction_generator",
        13: "chapter_13_feedback",
        99: "selfplay_evaluation",
    }
    step_name = step_names.get(step_id, f"step_{step_id}")

    # â”€â”€ Tier 0: Mission + Schema + Grammar + Guardrails â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if is_chapter_13 or step_id == 13:
        mission = CHAPTER_13_MISSION
        schema_excerpt = CHAPTER_13_SCHEMA_EXCERPT
        grammar_name = CHAPTER_13_GRAMMAR
        guardrails = list(CHAPTER_13_GUARDRAILS)
    elif step_id == 99:  # Selfplay evaluation
        mission = SELFPLAY_EVALUATION_MISSION
        schema_excerpt = SELFPLAY_SCHEMA_EXCERPT
        grammar_name = SELFPLAY_GRAMMAR
        guardrails = list(SELFPLAY_GUARDRAILS)
    else:
        mission = STEP_MISSIONS.get(step_id, f"Step {step_id} — no mission defined.")
        schema_excerpt = STEP_SCHEMA_EXCERPTS.get(step_id, "No schema excerpt available.")
        grammar_name = STEP_GRAMMAR_NAMES.get(step_id, "json_generic.gbnf")
        guardrails = list(STEP_GUARDRAILS.get(step_id, []))

    contracts = list(AUTHORITY_CONTRACTS)

    # â”€â”€ Tier 1: Evaluation data from existing context builders â”€â”€â”€â”€â”€â”€â”€
    inputs_summary = {}
    evaluation_summary = {}

    if is_chapter_13 or step_id == 13:
        # Chapter 13 has its own orchestrator — no step context to build.
        # Extract directly from results dict.
        inputs_summary = {
            k: v for k, v in results.items()
            if not isinstance(v, (list, dict)) or k in (
                "drift_detected", "metric_delta", "trigger_action",
                "retrain_recommended", "confidence"
            )
        }
        evaluation_summary = {
            "success": results.get("success", True),
            "confidence": results.get("confidence", 0.5),
            "interpretation": results.get("interpretation", "Chapter 13 evaluation"),
        }
    elif step_id in range(1, 7):
        # Steps 1-6: Use existing Pydantic context builders
        try:
            from agents.full_agent_context import build_full_context

            context = build_full_context(
                step=step_id,
                results=results,
                run_number=run_number,
                manifest_path=manifest_path,
            )

            # Extract evaluation summary (same dict shape as watcher uses)
            eval_data = context.get_evaluation_summary()
            evaluation_summary = {
                "success": eval_data.get("success", False),
                "confidence": eval_data.get("confidence", 0.5),
                "interpretation": eval_data.get("interpretation", ""),
            }

            # Extract inputs summary (key metrics from results)
            if hasattr(context, "agent_context") and context.agent_context:
                key_metrics = context.agent_context.get_key_metrics()
                inputs_summary = {
                    m: results.get(m) for m in key_metrics if m in results
                }

        except Exception as e:
            logger.warning("Could not build existing context for step %d: %s", step_id, e)
            inputs_summary = {
                k: v for k, v in results.items()
                if not isinstance(v, (list, dict))
            }
    else:
        # Unknown step — extract scalar values from results
        inputs_summary = {
            k: v for k, v in results.items()
            if not isinstance(v, (list, dict))
        }

    # â”€â”€ Tier 2: Retrieval (Track 2 stubs) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    recent_outcomes = _retrieve_recent_outcomes(step_id)
    trend_summary = _retrieve_trend_summary(step_id)
    open_incidents = _retrieve_open_incidents(step_id)

    # â”€â”€ Provenance â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    provenance = []
    if manifest_path and os.path.isfile(manifest_path):
        provenance.append(ProvenanceRecord.from_file(manifest_path))

    if state_paths:
        for sp in state_paths:
            if os.path.isfile(sp):
                provenance.append(ProvenanceRecord.from_file(sp))

    # â”€â”€ Assemble bundle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    bundle_context = BundleContext(
        mission=mission,
        schema_excerpt=schema_excerpt,
        grammar_name=grammar_name,
        contracts=contracts,
        guardrails=guardrails,
        inputs_summary=inputs_summary,
        evaluation_summary=evaluation_summary,
        recent_outcomes=recent_outcomes,
        trend_summary=trend_summary,
        open_incidents=open_incidents,
    )

    bundle = StepAwarenessBundle(
        step_id=step_id,
        step_name=step_name,
        run_id=run_id,
        is_chapter_13=is_chapter_13,
        context=bundle_context,
        budgets=budgets,
        provenance=provenance,
    )

    logger.info(
        "Built awareness bundle: step=%d (%s), run=%s, "
        "tier0=%d guardrails, tier2=%d outcomes",
        step_id,
        step_name,
        run_id,
        len(guardrails),
        len(recent_outcomes),
    )

    return bundle


# ═════════════════════════════════════════════════════════════════════════════
# PROMPT RENDERING
# ═════════════════════════════════════════════════════════════════════════════

def render_prompt_from_bundle(bundle: StepAwarenessBundle) -> str:
    """Render a Step Awareness Bundle into an LLM prompt string.

    Respects token budget tiers:
        Tier 0 (always): Mission, schema, grammar reminder, guardrails
        Tier 1 (if fits): Evaluation data, inputs summary
        Tier 2 (fill remaining): History, trends, incidents

    Args:
        bundle: Assembled StepAwarenessBundle.

    Returns:
        Formatted prompt string ready for LLM consumption.
    """
    ctx = bundle.context
    budgets = bundle.budgets
    sections = []

    # â”€â”€ Tier 0: Always included â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tier0_parts = []

    # Global mission context (v1.1.0)
    tier0_parts.append(f"=== SYSTEM MISSION ===\n{MAIN_MISSION}")

    tier0_parts.append(f"=== STEP MISSION ===\n{ctx.mission}")

    tier0_parts.append(f"=== EVALUATION SCHEMA ===\n{ctx.schema_excerpt}")

    tier0_parts.append(
        f"=== OUTPUT FORMAT ===\n"
        f"Respond using grammar: {ctx.grammar_name}\n"
        f"Your output MUST conform to this grammar exactly."
    )

    if ctx.guardrails:
        guardrail_text = "\n".join(f"- {g}" for g in ctx.guardrails)
        tier0_parts.append(f"=== GUARDRAILS ===\n{guardrail_text}")

    if ctx.contracts:
        contract_text = ", ".join(ctx.contracts)
        tier0_parts.append(
            f"=== AUTHORITY CONTRACTS ===\n"
            f"Active contracts: {contract_text}\n"
            f"You cannot override these. Propose within their bounds only."
        )

    tier0_text = "\n\n".join(tier0_parts)
    sections.append(tier0_text)

    # â”€â”€ Tier 1: Evaluation data (if budget allows) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tier1_parts = []

    if ctx.evaluation_summary:
        eval_json = json.dumps(ctx.evaluation_summary, indent=2, default=str)
        tier1_parts.append(f"=== CURRENT EVALUATION ===\n{eval_json}")

    if ctx.inputs_summary:
        inputs_json = json.dumps(ctx.inputs_summary, indent=2, default=str)
        tier1_parts.append(f"=== STEP OUTPUT METRICS ===\n{inputs_json}")

    if tier1_parts:
        tier1_text = "\n\n".join(tier1_parts)
        tier1_text = _truncate_to_budget(tier1_text, budgets.tier1_cap_tokens)
        sections.append(tier1_text)

    # â”€â”€ Tier 2: History + Trends + Incidents (fill remaining) â”€â”€â”€â”€â”€â”€â”€â”€
    tier2_parts = []

    if ctx.recent_outcomes:
        outcomes_data = [o.model_dump() for o in ctx.recent_outcomes]
        outcomes_json = json.dumps(outcomes_data, indent=2, default=str)
        outcomes_text = f"=== RECENT OUTCOMES (Chapter 13 curated) ===\n{outcomes_json}"
        outcomes_text = _truncate_to_budget(outcomes_text, budgets.history_cap_tokens)
        tier2_parts.append(outcomes_text)

    if ctx.trend_summary:
        trends_data = [t.model_dump() for t in ctx.trend_summary]
        trends_json = json.dumps(trends_data, indent=2, default=str)
        trends_text = f"=== METRIC TRENDS ===\n{trends_json}"
        trends_text = _truncate_to_budget(trends_text, budgets.telemetry_cap_tokens // 2)
        tier2_parts.append(trends_text)

    if ctx.open_incidents:
        incidents_text = "\n".join(f"- {inc}" for inc in ctx.open_incidents)
        incidents_section = f"=== OPEN INCIDENTS ===\n{incidents_text}"
        incidents_section = _truncate_to_budget(
            incidents_section, budgets.telemetry_cap_tokens // 2
        )
        tier2_parts.append(incidents_section)

    if tier2_parts:
        tier2_text = "\n\n".join(tier2_parts)
        sections.append(tier2_text)

    # â”€â”€ Assemble final prompt â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    prompt = "\n\n".join(sections)

    # Final budget check
    total_estimated = _estimate_tokens(prompt)
    available = budgets.ctx_max_tokens - budgets.generation_reserve_tokens
    if total_estimated > available:
        logger.warning(
            "Prompt exceeds budget: ~%d tokens estimated, %d available. Truncating.",
            total_estimated,
            available,
        )
        prompt = _truncate_to_budget(prompt, available)

    logger.debug(
        "Rendered prompt: ~%d estimated tokens (budget: %d)",
        _estimate_tokens(prompt),
        available,
    )

    return prompt


# ═════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: build_llm_context() — Team Beta Guardrail #1
# ═════════════════════════════════════════════════════════════════════════════
# This is the "single context entry point" that dispatch functions call.
# Today it wraps build_step_awareness_bundle + render_prompt_from_bundle.
# Tomorrow (Track 2) it adds retrieval — dispatch code never changes.

def build_llm_context(
    step_id: int,
    run_id: str = "",
    results: Optional[Dict[str, Any]] = None,
    run_number: int = 1,
    manifest_path: Optional[str] = None,
    state_paths: Optional[List[str]] = None,
    is_chapter_13: bool = False,
) -> tuple:
    """Single entry point for LLM context assembly.

    Team Beta Guardrail #1: Dispatch functions call this ONE function.
    No direct context assembly inside dispatch code.

    Args:
        step_id: Pipeline step (1-6) or 13 for Chapter 13.
        run_id: Run identifier.
        results: Step results to evaluate.
        run_number: Current run number.
        manifest_path: Path to agent manifest.
        state_paths: Additional provenance files.
        is_chapter_13: Whether this is a Chapter 13 evaluation.

    Returns:
        (prompt, grammar_name, bundle) tuple.
        - prompt: Rendered prompt string for LLM.
        - grammar_name: Grammar file to use for constrained decoding.
        - bundle: The full StepAwarenessBundle for audit/logging.
    """
    bundle = build_step_awareness_bundle(
        step_id=step_id,
        run_id=run_id,
        results=results,
        run_number=run_number,
        manifest_path=manifest_path,
        state_paths=state_paths,
        is_chapter_13=is_chapter_13,
    )

    prompt = render_prompt_from_bundle(bundle)
    grammar_name = bundle.context.grammar_name

    return prompt, grammar_name, bundle


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Bundle Factory v1.1.0 — Self-Test")
    print("=" * 70)

    # Test 1: Build bundle for each step
    for step in [1, 2, 3, 4, 5, 6]:
        bundle = build_step_awareness_bundle(
            step_id=step,
            results={"completion_rate": 1.0, "survivors_scored": 50000},
        )
        print(f"\n  Step {step} ({bundle.step_name}):")
        print(f"    Mission: {bundle.context.mission[:80]}...")
        print(f"    Grammar: {bundle.context.grammar_name}")
        print(f"    Guardrails: {len(bundle.context.guardrails)}")
        print(f"    Contracts: {bundle.context.contracts}")

    # Test 2: Build Chapter 13 bundle
    ch13_bundle = build_step_awareness_bundle(
        step_id=13,
        is_chapter_13=True,
        results={"drift_detected": True, "metric_delta": -0.05},
    )
    print(f"\n  Chapter 13 ({ch13_bundle.step_name}):")
    print(f"    Mission: {ch13_bundle.context.mission[:80]}...")
    print(f"    Grammar: {ch13_bundle.context.grammar_name}")
    print(f"    Guardrails: {len(ch13_bundle.context.guardrails)}")

    # Test 3: Render prompt
    print("\n  Render test (Step 3):")
    step3_bundle = build_step_awareness_bundle(
        step_id=3,
        results={
            "completion_rate": 0.98,
            "survivors_scored": 48000,
            "survivors_total": 50000,
            "feature_dimensions": 64,
            "mean_score": 0.42,
            "score_std": 0.18,
            "top_candidates": 1200,
        },
    )
    prompt = render_prompt_from_bundle(step3_bundle)
    estimated_tokens = _estimate_tokens(prompt)
    print(f"    Prompt length: {len(prompt)} chars (~{estimated_tokens} tokens)")
    print(f"    Budget: {step3_bundle.budgets.ctx_max_tokens} max")
    print(f"    Within budget: {estimated_tokens < step3_bundle.budgets.ctx_max_tokens}")

    # Test 4: Convenience function
    print("\n  Convenience function test (Step 5):")
    prompt, grammar, bundle = build_llm_context(
        step_id=5,
        results={"val_r2": 0.62, "overfit_ratio": 1.15},
    )
    print(f"    Grammar: {grammar}")
    print(f"    Prompt tokens: ~{_estimate_tokens(prompt)}")
    print(f"    Bundle version: {bundle.bundle_version}")

    # Test 5: Token budget enforcement
    print("\n  Token budget test:")
    tight_budget = TokenBudget(ctx_max_tokens=4096, generation_reserve_tokens=500)
    tight_bundle = build_step_awareness_bundle(
        step_id=1,
        budgets=tight_budget,
        results={"survivor_count": 5000},
    )
    tight_prompt = render_prompt_from_bundle(tight_bundle)
    print(f"    Tight budget (4096): ~{_estimate_tokens(tight_prompt)} tokens")

    # Test 6: Provenance
    print("\n  Provenance test:")
    prov = ProvenanceRecord.from_file("/tmp/nonexistent_file.json")
    print(f"    Missing file: sha256={prov.sha256}")

    # Test 7: Selfplay evaluation bundle (v1.1.0)
    print("\n  Selfplay evaluation test (step_id=99):")
    selfplay_bundle = build_step_awareness_bundle(
        step_id=99,
        results={
            "episodes_completed": 10,
            "candidates_emitted": 3,
            "best_fitness": 0.72,
            "exploration_breadth": 0.65,
            "mode_collapse_detected": False,
        },
    )
    print(f"    Step name: {selfplay_bundle.step_name}")
    print(f"    Mission: {selfplay_bundle.context.mission[:60]}...")
    print(f"    Grammar: {selfplay_bundle.context.grammar_name}")
    print(f"    Guardrails: {len(selfplay_bundle.context.guardrails)}")

    # Test 8: MAIN_MISSION injection (v1.1.0)
    print("\n  MAIN_MISSION injection test:")
    test_prompt = render_prompt_from_bundle(selfplay_bundle)
    assert "=== SYSTEM MISSION ===" in test_prompt, "MAIN_MISSION not found in prompt"
    assert "functional mimicry" in test_prompt, "functional mimicry text not found"
    print(f"    MAIN_MISSION present: OK")
    print(f"    Prompt starts with: {test_prompt[:50]}...")

    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)
