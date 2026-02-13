#!/usr/bin/env python3
"""
Diagnostics LLM Analyzer — Chapter 14 Phase 7.

Version: 1.0.0
Date: 2026-02-11
Chapter: 14 Phase 7 (LLM Integration)
Spec: CHAPTER_14_TRAINING_DIAGNOSTICS.md Sections 8.1, 8.4, 8.5
Pattern: Follows parameter_advisor.py + advisor_bundle.py

PURPOSE:
    When WATCHER or Strategy Advisor detects training health issues
    (severity >= warning), this module asks DeepSeek-R1-14B to
    interpret the diagnostics and produce structured recommendations.

AUTHORITY:
    - MAY: Analyze diagnostics, classify focus, recommend parameters
    - NEVER: Execute changes, modify files, bypass WATCHER policy

CALL CHAIN:
    training_health_check.py (severity >= warning)
        → WATCHER _handle_retry() or Strategy Advisor
            → request_llm_diagnostics_analysis()
                → build_diagnostics_prompt()      (prompt assembly)
                → llm_router.evaluate_with_grammar()  (grammar-constrained)
                → DiagnosticsAnalysis (Pydantic parse)

DESIGN INVARIANT:
    Diagnostics generation is best-effort and non-fatal.
    Failure to produce diagnostics must never fail Step 5,
    block pipeline progression, or alter training outcomes.

Usage:
    from diagnostics_llm_analyzer import request_llm_diagnostics_analysis

    # Full end-to-end (WATCHER calls this)
    analysis = request_llm_diagnostics_analysis(
        diagnostics_path="diagnostics_outputs/training_diagnostics.json",
        tier_comparison_path="diagnostics_outputs/tier_comparison.json",
    )

    if analysis:
        print(analysis.focus_area)          # e.g. FEATURE_RELEVANCE
        print(analysis.root_cause)          # Root cause explanation
        print(analysis.parameter_proposals) # Proposed changes
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS — Paths
# =============================================================================

DIAGNOSTICS_PATH = "diagnostics_outputs/training_diagnostics.json"
DIAGNOSTICS_HISTORY_DIR = "diagnostics_outputs/history/"
DIAGNOSTICS_GRAMMAR = "diagnostics_analysis.gbnf"

# Grammar file location (same directory pattern as strategy_advisor.gbnf)
GRAMMAR_DIR = "grammars"


# =============================================================================
# DIAGNOSTICS MISSION & SCHEMA (Tier 0)
# =============================================================================

DIAGNOSTICS_MISSION = (
    "Training Diagnostics Analyst: Evaluate training health data from Step 5 "
    "across up to 4 model types (neural_net, xgboost, lightgbm, catboost). "
    "Diagnose root causes of model underperformance. Classify into focus areas: "
    "MODEL_DIVERSITY, FEATURE_RELEVANCE, POOL_PRECISION, CONFIDENCE_CALIBRATION, "
    "REGIME_SHIFT. Recommend actionable parameter changes for selfplay exploration. "
    "Your recommendations are PROPOSALS — WATCHER validates against policy bounds. "
    "You do NOT have execution authority."
)

DIAGNOSTICS_SCHEMA_EXCERPT = (
    "DiagnosticsAnalysis: key_fields=[focus_area, root_cause, model_recommendations[], "
    "parameter_proposals[], confidence]. "
    "focus_area enum: MODEL_DIVERSITY, FEATURE_RELEVANCE, POOL_PRECISION, "
    "CONFIDENCE_CALIBRATION, REGIME_SHIFT. "
    "model_recommendations[]: per-model-type verdict (viable/fixable/skip). "
    "parameter_proposals[]: specific changes with rationale. "
    "confidence: 0.0-1.0 in your analysis."
)

DIAGNOSTICS_GUARDRAILS = [
    "You are analyzing training TELEMETRY, not making execution decisions.",
    "All parameter proposals must include specific numeric values, not vague directions.",
    "If multiple model types were diagnosed, compare them — do not analyze in isolation.",
    "Neural net dead neurons > 50% is ALWAYS critical — do not downplay.",
    "Feature gradient spread > 1000x indicates preprocessing failure, not model failure.",
    "Trees outperforming NN on tabular data is EXPECTED, not a defect.",
    "Your focus_area recommendation directly drives selfplay episode planning.",
]


# =============================================================================
# PROMPT BUILDER (Follows advisor_bundle.py pattern)
# =============================================================================

def build_diagnostics_prompt(
    diagnostics_path: str,
    tier_comparison_path: Optional[str] = None,
    history_paths: Optional[List[str]] = None,
    token_budget: int = 8000,
) -> str:
    """Build a structured prompt for LLM diagnostics analysis.

    Follows the tiered architecture from bundle_factory.py:
    - Tier 0: Mission + schema + grammar + guardrails
    - Tier 1: Current diagnostics data + tier comparison
    - Tier 2: Historical diagnostics (trend detection)

    NOT called on every run — only when:
    1. WATCHER severity >= warning, OR
    2. Strategy Advisor scheduled analysis cycle, OR
    3. Chapter 13 requests root cause analysis after hit rate drop

    Args:
        diagnostics_path: Path to training_diagnostics.json
        tier_comparison_path: Path to tier_comparison.json (per-survivor attribution)
        history_paths: Paths to previous diagnostics in history/ dir
        token_budget: Maximum estimated tokens for the prompt.

    Returns:
        Rendered prompt string for LLM.
    """
    sections: List[str] = []

    # ─── TIER 0: Mission + Schema + Guardrails (always included) ─────

    sections.append("=== STEP MISSION ===")
    sections.append(DIAGNOSTICS_MISSION)
    sections.append("")

    sections.append("=== EVALUATION SCHEMA ===")
    sections.append(DIAGNOSTICS_SCHEMA_EXCERPT)
    sections.append("")

    sections.append("=== OUTPUT FORMAT ===")
    sections.append(f"Respond using grammar: {DIAGNOSTICS_GRAMMAR}")
    sections.append("Your output MUST conform to this grammar exactly.")
    sections.append("")

    sections.append("=== GUARDRAILS ===")
    for i, g in enumerate(DIAGNOSTICS_GUARDRAILS, 1):
        sections.append(f"{i}. {g}")
    sections.append("")

    sections.append("=== AUTHORITY CONTRACTS ===")
    sections.append("Active contracts: CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md")
    sections.append("You cannot override these. Propose within their bounds only.")
    sections.append("")

    # ─── TIER 1: Current diagnostics summary ─────────────────────────

    try:
        with open(diagnostics_path) as f:
            current_diag = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error("Cannot read diagnostics file %s: %s", diagnostics_path, e)
        # Return minimal prompt so LLM can at least say "insufficient data"
        sections.append("=== STEP OUTPUT METRICS ===")
        sections.append('{"error": "diagnostics file unavailable"}')
        return "\n".join(sections)

    # Extract key metrics for prompt
    inputs_summary = {
        'model_type': current_diag.get('model_type', 'unknown'),
        'severity': current_diag.get('diagnosis', {}).get('severity', 'unknown'),
        'issues': current_diag.get('diagnosis', {}).get('issues', []),
        'training_rounds_total': current_diag.get('training_rounds', {}).get('total', 0),
        'training_rounds_best': current_diag.get('training_rounds', {}).get('best', 0),
        'top_5_features': list(
            current_diag.get('feature_importance', {}).get('values', {}).keys()
        )[:5],
    }

    # Add NN-specific metrics if present
    nn_data = current_diag.get('nn_specific', {})
    if nn_data:
        inputs_summary['nn_layer_health'] = nn_data.get('layer_health', {})
        inputs_summary['nn_gradient_spread'] = nn_data.get('feature_gradient_spread', 0)

    # Add multi-model comparison data if present (from compare_models run)
    models_data = current_diag.get('models', {})
    if models_data:
        model_summaries = {}
        for mtype, mdata in models_data.items():
            model_summaries[mtype] = {
                'severity': mdata.get('diagnosis', {}).get('severity', 'unknown'),
                'issues': mdata.get('diagnosis', {}).get('issues', []),
                'rounds_total': mdata.get('training_rounds', {}).get('total', 0),
                'rounds_best': mdata.get('training_rounds', {}).get('best', 0),
            }
        inputs_summary['multi_model_comparison'] = model_summaries

    sections.append("=== STEP OUTPUT METRICS ===")
    sections.append(json.dumps(inputs_summary, indent=2, default=str))
    sections.append("")

    # ─── TIER 1: Tier comparison (if available) ──────────────────────

    if tier_comparison_path and os.path.isfile(tier_comparison_path):
        try:
            with open(tier_comparison_path) as f:
                tier_data = json.load(f)

            # Only include top divergent features to save tokens
            divergence = tier_data.get('divergence', {})
            sorted_div = sorted(
                divergence.items(), key=lambda x: abs(x[1]), reverse=True
            )
            evaluation_summary = {
                'tier_comparison_available': True,
                'top_divergent_features': [
                    {'feature': name, 'divergence': round(val, 4)}
                    for name, val in sorted_div[:10]
                ],
                'interpretation': (
                    "Positive divergence = feature concentrated in top tier. "
                    "Negative divergence = feature more important in wide tier. "
                    "Large absolute values indicate structurally different populations."
                ),
            }

            sections.append("=== CURRENT EVALUATION ===")
            sections.append(json.dumps(evaluation_summary, indent=2))
            sections.append("")
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not load tier comparison: %s", e)

    # ─── TIER 2: Historical diagnostics (trend detection) ────────────

    if history_paths:
        recent = []
        for hp in history_paths[-5:]:  # Last 5 runs
            if os.path.isfile(hp):
                try:
                    with open(hp) as f:
                        hist = json.load(f)
                    recent.append({
                        'run_id': os.path.basename(hp).replace('.json', ''),
                        'severity': hist.get('diagnosis', {}).get('severity',
                                    hist.get('watcher_severity', 'unknown')),
                        'model_type': hist.get('model_type', 'unknown'),
                        'archived_at': hist.get('archived_at', ''),
                    })
                except (json.JSONDecodeError, KeyError):
                    continue

        if recent:
            sections.append("=== RECENT HISTORY (last 5 runs) ===")
            sections.append(json.dumps(recent, indent=2))
            sections.append("")

    # ─── Finalize ────────────────────────────────────────────────────

    prompt = "\n".join(sections)

    # Rough token estimate (4 chars per token)
    est_tokens = len(prompt) // 4
    if est_tokens > token_budget:
        logger.warning(
            "Diagnostics prompt ~%d tokens exceeds budget %d, "
            "truncating history section",
            est_tokens, token_budget,
        )
        # Re-build without history to stay in budget
        return build_diagnostics_prompt(
            diagnostics_path=diagnostics_path,
            tier_comparison_path=tier_comparison_path,
            history_paths=None,
            token_budget=token_budget,
        )

    logger.info(
        "Built diagnostics prompt: model=%s, severity=%s, ~%d tokens",
        inputs_summary.get('model_type', 'unknown'),
        inputs_summary.get('severity', 'unknown'),
        est_tokens,
    )
    return prompt


# =============================================================================
# END-TO-END LLM CALL
# =============================================================================

def request_llm_diagnostics_analysis(
    diagnostics_path: str = DIAGNOSTICS_PATH,
    tier_comparison_path: Optional[str] = None,
    llm_router=None,
    timeout: int = 120,
) -> Optional[Any]:
    """Full end-to-end LLM diagnostics analysis.

    1. Build prompt (assemble context with token budgets)
    2. Call LLM with GBNF grammar constraint
    3. Parse and validate response with Pydantic
    4. Return structured DiagnosticsAnalysis

    This is the primary entry point called by WATCHER's _handle_retry()
    or by the Strategy Advisor when diagnostics severity >= warning.

    Best-effort: returns None if any step fails, never raises.

    Args:
        diagnostics_path: Path to training_diagnostics.json
        tier_comparison_path: Path to tier_comparison.json
        llm_router: Optional pre-configured LLMRouter instance.
                    If None, creates a new one.
        timeout: Maximum seconds for LLM call. Default 120.
                 Non-negotiable for daemon safety -- prevents
                 stalled DeepSeek from blocking SIGTERM response.

    Returns:
        DiagnosticsAnalysis or None if LLM call fails.
    """
    try:
        return _request_llm_diagnostics_analysis_inner(
            diagnostics_path, tier_comparison_path, llm_router, timeout
        )
    except Exception as e:
        # Best-effort invariant: never propagate exceptions
        logger.error(
            "Diagnostics LLM analysis failed (non-fatal): %s", e,
            exc_info=True,
        )
        return None


def _request_llm_diagnostics_analysis_inner(
    diagnostics_path: str,
    tier_comparison_path: Optional[str],
    llm_router,
    timeout: int,
) -> Optional[Any]:
    """Inner implementation — may raise on failure."""

    import signal

    # Import schema (deferred to avoid circular imports)
    from diagnostics_analysis_schema import (
        DiagnosticsAnalysis,
        parse_diagnostics_response,
    )

    # Gather history files for trend context
    history_paths = sorted(glob.glob(
        os.path.join(DIAGNOSTICS_HISTORY_DIR, "*.json")
    ))

    # Step 1: Build prompt
    prompt = build_diagnostics_prompt(
        diagnostics_path=diagnostics_path,
        tier_comparison_path=tier_comparison_path,
        history_paths=history_paths,
        token_budget=8000,  # Generous for DeepSeek-R1-14B 32K context
    )

    if not prompt:
        logger.error("Empty prompt — cannot call LLM")
        return None

    # Step 2: Get or create LLM router
    if llm_router is None:
        try:
            from llm_services.llm_router import LLMRouter
            llm_router = LLMRouter()
        except ImportError:
            logger.error("LLMRouter not available — cannot perform diagnostics analysis")
            return None

    # Step 3: Resolve grammar file
    # NOTE: llm_router.evaluate_with_grammar() expects a BARE FILENAME,
    # not a path. The router prepends grammars/ internally.
    grammar_file = DIAGNOSTICS_GRAMMAR  # bare filename
    grammar_full = os.path.join(GRAMMAR_DIR, DIAGNOSTICS_GRAMMAR)
    if not os.path.isfile(grammar_full):
        legacy = os.path.join("agent_grammars", DIAGNOSTICS_GRAMMAR)
        if os.path.isfile(legacy):
            logger.info("Grammar found in legacy location: %s", legacy)
        else:
            logger.warning("Grammar file not found: %s", grammar_full)

    logger.info("Calling LLM with grammar: %s (timeout=%ds)", grammar_file, timeout)

    # Step 4: Call LLM with grammar constraint + timeout enforcement
    # Uses SIGALRM for daemon safety -- if DeepSeek stalls, this ensures
    # we return None rather than blocking the daemon indefinitely.
    raw_response = None
    _prev_handler = None
    try:
        def _timeout_handler(signum, frame):
            raise TimeoutError(
                f"LLM diagnostics call exceeded {timeout}s timeout"
            )

        # SIGALRM only available on Unix (Zeus is Linux)
        if hasattr(signal, 'SIGALRM'):
            _prev_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        raw_response = llm_router.evaluate_with_grammar(
            prompt=prompt,
            grammar_file=grammar_file,
            max_tokens=2000,
        )

    except TimeoutError as te:
        logger.error("LLM diagnostics timed out after %ds: %s", timeout, te)
        return None
    finally:
        # Always restore alarm and handler
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            if _prev_handler is not None:
                signal.signal(signal.SIGALRM, _prev_handler)

    if not raw_response:
        logger.error("LLM returned empty response for diagnostics analysis")
        return None

    # Step 5: Parse with Pydantic
    analysis = parse_diagnostics_response(raw_response)

    if analysis is None:
        logger.error("Failed to parse LLM diagnostics response")
        logger.debug("Raw response (first 500 chars): %s", raw_response[:500])
        return None

    logger.info(
        "LLM diagnostics analysis: focus=%s, root_cause_confidence=%.2f, "
        "models=%d recommendations, params=%d proposals",
        analysis.focus_area.value,
        analysis.root_cause_confidence,
        len(analysis.model_recommendations),
        len(analysis.parameter_proposals),
    )

    # Step 6: Archive analysis
    _archive_analysis(analysis)

    return analysis


# =============================================================================
# ANALYSIS ARCHIVAL
# =============================================================================

def _archive_analysis(analysis) -> None:
    """Archive LLM analysis to diagnostics_outputs/llm_proposals/.

    Best-effort: never raises.
    """
    try:
        archive_dir = Path("diagnostics_outputs/llm_proposals")
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostics_analysis_{timestamp}.json"
        filepath = archive_dir / filename

        with open(filepath, 'w') as f:
            json.dump(analysis.model_dump(), f, indent=2, default=str)

        logger.info("Archived diagnostics analysis to %s", filepath)

    except Exception as e:
        logger.warning("Failed to archive diagnostics analysis: %s", e)


# =============================================================================
# WATCHER INTEGRATION HELPER
# =============================================================================

def get_retry_params_from_analysis(analysis) -> Dict[str, Any]:
    """Extract retry parameters from LLM diagnostics analysis.

    Called by WATCHER's _build_retry_params() when analysis is available.
    Converts LLM parameter proposals into the format expected by
    the pipeline step runner.

    Args:
        analysis: DiagnosticsAnalysis from request_llm_diagnostics_analysis()

    Returns:
        Dict of parameter adjustments, or empty dict if no proposals.
    """
    if analysis is None:
        return {}

    params = {}

    for proposal in analysis.parameter_proposals:
        params[proposal.parameter] = {
            'value': proposal.proposed_value,
            'rationale': proposal.rationale,
            'source': 'llm_diagnostics',
        }

    # Add model-level guidance
    skip_models = []
    fixable_models = []
    for rec in analysis.model_recommendations:
        if rec.verdict.value == "skip":
            skip_models.append(rec.model_type.value)
        elif rec.verdict.value == "fixable":
            fixable_models.append(rec.model_type.value)

    if skip_models:
        params['_skip_models'] = skip_models
    if fixable_models:
        params['_fixable_models'] = fixable_models

    params['_focus_area'] = analysis.focus_area.value
    params['_selfplay_guidance'] = analysis.selfplay_guidance
    params['_requires_human_review'] = analysis.requires_human_review

    return params


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format='%(name)s %(levelname)s: %(message)s')

    print("=" * 60)
    print("Diagnostics LLM Analyzer — Self-Test")
    print("=" * 60)

    # Test 1: Build prompt from synthetic diagnostics
    print("\n--- Test 1: Build prompt from synthetic data ---")

    test_diag = {
        "model_type": "neural_net",
        "diagnosis": {
            "severity": "critical",
            "issues": ["47% dead neurons in fc1", "Feature gradient spread 12847x"]
        },
        "training_rounds": {"total": 200, "best": 23},
        "feature_importance": {
            "values": {
                "intersection_weight": 0.34,
                "skip_entropy": 0.21,
                "lane_agreement_8": 0.15,
                "temporal_stability": 0.12,
                "forward_count": 0.08,
            }
        },
        "nn_specific": {
            "layer_health": {
                "fc1": {"dead_pct": 47, "gradient_norm": 0.0003},
                "fc2": {"dead_pct": 12, "gradient_norm": 0.012},
                "fc3": {"dead_pct": 3, "gradient_norm": 0.45},
            },
            "feature_gradient_spread": 12847,
        }
    }

    # Write temp diagnostics file
    test_dir = Path("/tmp/diag_test")
    test_dir.mkdir(parents=True, exist_ok=True)
    test_path = test_dir / "training_diagnostics.json"
    with open(test_path, 'w') as f:
        json.dump(test_diag, f)

    prompt = build_diagnostics_prompt(str(test_path))
    if prompt and "STEP MISSION" in prompt and "critical" in prompt:
        print("✅ Prompt built successfully")
        print(f"   Length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    else:
        print("❌ Prompt build FAILED")
        sys.exit(1)

    # Test 2: Verify guardrails are in prompt
    print("\n--- Test 2: Verify guardrails in prompt ---")
    missing = []
    for g in DIAGNOSTICS_GUARDRAILS:
        if g not in prompt:
            missing.append(g[:50])
    if not missing:
        print("✅ All 7 guardrails present")
    else:
        print(f"❌ Missing guardrails: {missing}")
        sys.exit(1)

    # Test 3: Verify mission in prompt
    print("\n--- Test 3: Verify mission in prompt ---")
    if "Training Diagnostics Analyst" in prompt:
        print("✅ Mission statement present")
    else:
        print("❌ Mission statement MISSING")
        sys.exit(1)

    # Test 4: Test get_retry_params_from_analysis
    print("\n--- Test 4: get_retry_params_from_analysis ---")
    from diagnostics_analysis_schema import DiagnosticsAnalysis

    mock_analysis = DiagnosticsAnalysis(
        focus_area="FEATURE_RELEVANCE",
        root_cause="Test root cause",
        root_cause_confidence=0.8,
        model_recommendations=[
            {"model_type": "neural_net", "verdict": "fixable", "rationale": "Fix scaling"},
            {"model_type": "catboost", "verdict": "viable", "rationale": "Good"},
        ],
        parameter_proposals=[
            {"parameter": "normalize_features", "proposed_value": 1, "rationale": "Scale inputs"},
        ],
        selfplay_guidance="Focus on feature relevance",
        requires_human_review=False,
    )

    retry_params = get_retry_params_from_analysis(mock_analysis)
    if (retry_params.get('normalize_features', {}).get('value') == 1
        and retry_params.get('_focus_area') == 'FEATURE_RELEVANCE'
        and retry_params.get('_fixable_models') == ['neural_net']):
        print("✅ Retry params extraction correct")
    else:
        print(f"❌ Retry params incorrect: {retry_params}")
        sys.exit(1)

    # Test 5: None-safe
    print("\n--- Test 5: None-safe (analysis=None) ---")
    empty = get_retry_params_from_analysis(None)
    if empty == {}:
        print("✅ None returns empty dict")
    else:
        print(f"❌ Expected empty dict, got: {empty}")
        sys.exit(1)

    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)

    print("\n" + "=" * 60)
    print("✅ All diagnostics LLM analyzer tests passed")
    print("=" * 60)
