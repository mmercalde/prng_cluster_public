#!/usr/bin/env python3
"""
Advisor Bundle — Context Assembly for Strategy Advisor LLM Calls.

Version: 1.0.0
Date: 2026-02-07
Contract: CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md
Parent: bundle_factory.py

PURPOSE:
    Assembles structured prompts for the Strategy Advisor LLM.
    Follows the same tiered architecture as bundle_factory.py:
    - Tier 0: Mission + schema + grammar + guardrails
    - Tier 1: Diagnostic data + telemetry + policy history
    - Tier 2: Computed metrics + signals

USAGE:
    from agents.contexts.advisor_bundle import build_advisor_bundle
    
    prompt = build_advisor_bundle(
        diagnostics=diagnostics_list,
        telemetry=telemetry_list,
        policy_history=policy_list,
        metrics=computed_metrics,
        signals=extracted_signals,
    )
    
    response = llm_router.evaluate_with_grammar(prompt, "strategy_advisor.gbnf")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ═══════════════════════════════════════════════════════════════════════════════
# ADVISOR MISSION & SCHEMA (Tier 0)
# ═══════════════════════════════════════════════════════════════════════════════

ADVISOR_MISSION = """You are a quantitative strategy advisor for a probabilistic PRNG 
analysis system. You analyze diagnostic data to recommend where selfplay exploration 
should focus.

HARD CONSTRAINTS:
- You do NOT execute actions or modify files
- You do NOT guess numbers or predict draws
- You MUST express uncertainty via confidence scores
- You MUST justify every proposal with specific data points
- You MUST classify a focus area from the defined set
- If uncertainty is high (no clear signal), recommend WAIT

FOCUS AREA DEFINITIONS:
- POOL_PRECISION: Hit@100 > 70% but Hit@20 < 10% → Optimize pool concentration
- POOL_COVERAGE: Hit@300 < 85% → Broaden exploration, diversify models
- CONFIDENCE_CALIBRATION: Calibration correlation < 0.3 → Focus on fold stability
- MODEL_DIVERSITY: Single model > 80% of best episodes → Force model rotation
- FEATURE_RELEVANCE: Feature drift > 0.3 → Features may be stale
- REGIME_SHIFT: Window decay > 0.5 AND survivor churn > 0.4 → PAUSE selfplay
- STEADY_STATE: All metrics within bounds → Maintenance mode

PRIORITY ORDER (when multiple apply):
REGIME_SHIFT > POOL_COVERAGE > CONFIDENCE_CALIBRATION > 
POOL_PRECISION > MODEL_DIVERSITY > FEATURE_RELEVANCE > STEADY_STATE

RECOMMENDED ACTIONS:
- RETRAIN: Trigger retraining with specified scope
- WAIT: No action needed, continue monitoring
- ESCALATE: Flag for human review
- REFOCUS: Adjust selfplay focus without full retraining
- FULL_RESET: Complete pipeline rerun (Steps 1→6)"""

ADVISOR_SCHEMA_EXCERPT = """StrategyRecommendation schema:
- focus_area: POOL_PRECISION|POOL_COVERAGE|CONFIDENCE_CALIBRATION|MODEL_DIVERSITY|FEATURE_RELEVANCE|REGIME_SHIFT|STEADY_STATE
- focus_confidence: 0.0-1.0
- focus_rationale: string (MUST cite specific metrics)
- recommended_action: RETRAIN|WAIT|ESCALATE|REFOCUS|FULL_RESET
- retrain_scope: selfplay_only|steps_5_6|steps_3_5_6|full_pipeline|null
- selfplay_overrides: {max_episodes, model_types[], min_fitness_threshold, priority_metrics[], exploration_ratio, search_strategy}
- parameter_proposals: [{parameter, current_value, proposed_value, delta, confidence, rationale}] (0-5 items)
- pool_strategy: {tight_pool_guidance, balanced_pool_guidance, wide_pool_guidance}
- risk_level: low|medium|high
- requires_human_review: boolean
- diagnostic_summary: {hit_at_20, hit_at_100, hit_at_300, calibration_correlation, survivor_churn, best_model_type, fitness_trend, draws_since_last_promotion}
- alternative_hypothesis: string|null"""

ADVISOR_GUARDRAILS = [
    "Every parameter proposal MUST cite specific diagnostic values — 'Increase X' without data is INVALID.",
    "All parameter proposals MUST fall within watcher_policies.json bounds.",
    "NEVER propose changes to: step ordering, feature schema, PRNG algorithms, sieve math, Pydantic schemas.",
    "If multiple focus areas apply, use the priority order to select primary.",
    "REGIME_SHIFT always takes priority — it invalidates all downstream optimization.",
    "If uncertainty is high, recommend WAIT. Speculation without data is forbidden.",
    "The Advisor does NOT execute actions — all outputs are proposals for WATCHER validation.",
]

ADVISOR_GRAMMAR = "strategy_advisor.gbnf"


# ═══════════════════════════════════════════════════════════════════════════════
# METRIC THRESHOLDS (for prompt context)
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_THRESHOLDS = """METRIC INTERPRETATION GUIDE:

Pool Concentration Score (PCS):
  > 0.30 → Strong concentration (good for precision)
  < 0.10 → Diffuse distribution (low confidence, broad exploration)

Calibration Correlation (CC):
  > 0.7 → Well-calibrated (confidence scores are meaningful)
  < 0.3 → Poorly calibrated (confidence is noise)
  < 0.0 → Inversely calibrated (confident predictions are WORSE)

Fitness Plateau Detection (FPD):
  |FPD| < 0.1 → Plateau (exploration not finding better regions)
  FPD > 0.5 → Still improving (continue current strategy)
  FPD < -0.3 → Regression (something is wrong — escalate)

Model Diversity Index (MDI):
  > 0.6 → Good diversity
  < 0.3 → Over-concentration (single model dominance)

Survivor Consistency Score (SCS):
  > 0.7 → Highly stable (same seeds performing)
  < 0.3 → Volatile (pool is churning — may indicate regime shift)

Hit Rate Thresholds:
  Hit@20: excellent(>15%), good(>10%), concerning(<5%)
  Hit@100: excellent(>80%), good(>70%), concerning(<60%)
  Hit@300: excellent(>95%), good(>85%), concerning(<80%)"""


# ═══════════════════════════════════════════════════════════════════════════════
# BUNDLE BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_advisor_bundle(
    diagnostics: List[Dict[str, Any]],
    telemetry: List[Dict[str, Any]],
    policy_history: List[Dict[str, Any]],
    metrics: Any,  # ComputedMetrics from parameter_advisor.py
    signals: Dict[str, Any],
    current_config: Optional[Dict[str, Any]] = None,
    token_budget: int = 8000,
) -> str:
    """Build a structured prompt for the Strategy Advisor LLM.
    
    Args:
        diagnostics: List of Chapter 13 diagnostic records (last N draws).
        telemetry: List of selfplay telemetry records (last N episodes).
        policy_history: List of promoted + rejected policy records.
        metrics: ComputedMetrics object with PCS, CC, FPD, MDI, SCS.
        signals: Dict of extracted diagnostic signals.
        current_config: Current selfplay configuration (optional).
        token_budget: Maximum estimated tokens for the prompt.
        
    Returns:
        Rendered prompt string for LLM.
    """
    sections = []
    
    # ══════════════════════════════════════════════════════════════════════════
    # TIER 0: Mission + Schema + Guardrails (always included)
    # ══════════════════════════════════════════════════════════════════════════
    
    sections.append("=== STRATEGY ADVISOR MISSION ===")
    sections.append(ADVISOR_MISSION)
    sections.append("")
    
    sections.append("=== OUTPUT SCHEMA ===")
    sections.append(ADVISOR_SCHEMA_EXCERPT)
    sections.append("")
    
    sections.append("=== GUARDRAILS ===")
    for i, g in enumerate(ADVISOR_GUARDRAILS, 1):
        sections.append(f"{i}. {g}")
    sections.append("")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1: Computed Metrics + Signals
    # ══════════════════════════════════════════════════════════════════════════
    
    sections.append("=== COMPUTED METRICS (Section 6 Formulas) ===")
    if hasattr(metrics, 'pcs'):
        sections.append(f"Pool Concentration Score (PCS): {metrics.pcs:.4f}")
        sections.append(f"Calibration Correlation (CC): {metrics.cc:.4f}")
        sections.append(f"Fitness Plateau Detection (FPD): {metrics.fpd:.4f}")
        sections.append(f"Model Diversity Index (MDI): {metrics.mdi:.4f}")
        sections.append(f"Survivor Consistency Score (SCS): {metrics.scs:.4f}")
    else:
        # Handle dict-style metrics
        sections.append(f"Pool Concentration Score (PCS): {metrics.get('pcs', 0.0):.4f}")
        sections.append(f"Calibration Correlation (CC): {metrics.get('cc', 0.0):.4f}")
        sections.append(f"Fitness Plateau Detection (FPD): {metrics.get('fpd', 0.0):.4f}")
        sections.append(f"Model Diversity Index (MDI): {metrics.get('mdi', 0.0):.4f}")
        sections.append(f"Survivor Consistency Score (SCS): {metrics.get('scs', 0.0):.4f}")
    sections.append("")
    
    sections.append("=== EXTRACTED SIGNALS ===")
    sections.append(f"Hit@20 (average): {signals.get('hit_at_20', 0.0):.4f}")
    sections.append(f"Hit@100 (average): {signals.get('hit_at_100', 0.0):.4f}")
    sections.append(f"Hit@300 (average): {signals.get('hit_at_300', 0.0):.4f}")
    sections.append(f"Model Dominance: {signals.get('model_dominance', 0.0):.2%}")
    sections.append(f"Best Model Type: {signals.get('best_model_type', 'unknown')}")
    sections.append(f"Feature Drift: {signals.get('feature_drift', 0.0):.4f}")
    sections.append(f"Window Decay: {signals.get('window_decay', 0.0):.4f}")
    sections.append(f"Draws Since Last Promotion: {signals.get('draws_since_last_promotion', 0)}")
    sections.append("")
    
    sections.append(METRIC_THRESHOLDS)
    sections.append("")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1: Diagnostic Data (summarized)
    # ══════════════════════════════════════════════════════════════════════════
    
    sections.append(f"=== DIAGNOSTIC DATA (Last {len(diagnostics)} draws) ===")
    
    # Summarize recent diagnostics (avoid token bloat)
    for i, d in enumerate(diagnostics[:5]):  # Show last 5 in detail
        sections.append(f"Draw {d.get('draw_number', '?')} ({d.get('timestamp', 'unknown')}):")
        sections.append(f"  Hit@20: {d.get('hit_at_20', 0.0):.4f}, Hit@100: {d.get('hit_at_100', 0.0):.4f}, Hit@300: {d.get('hit_at_300', 0.0):.4f}")
        if 'confidence_calibration' in d:
            cc = d['confidence_calibration']
            cc_val = cc.get('mean_confidence', 0.0) if isinstance(cc, dict) else (cc or 0.0)
            sections.append(f"  Confidence calibration: {cc_val:.4f}")
        if 'feature_drift' in d and d['feature_drift'] is not None:
            sections.append(f"  Feature drift: {d['feature_drift']:.4f}")
    
    if len(diagnostics) > 5:
        sections.append(f"  ... and {len(diagnostics) - 5} more diagnostic records")
    sections.append("")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1: Telemetry History (summarized)
    # ══════════════════════════════════════════════════════════════════════════
    
    sections.append(f"=== TELEMETRY HISTORY (Last {len(telemetry)} episodes) ===")
    
    # Model type distribution
    model_counts = {}
    fitness_values = []
    for t in telemetry:
        model_type = t.get('best_model_type', 'unknown')
        model_counts[model_type] = model_counts.get(model_type, 0) + 1
        if 'best_fitness' in t:
            fitness_values.append(t['best_fitness'])
    
    sections.append("Model Type Distribution:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        pct = count / len(telemetry) * 100 if telemetry else 0
        sections.append(f"  {model}: {count} ({pct:.1f}%)")
    
    if fitness_values:
        sections.append(f"Fitness Range: {min(fitness_values):.4f} - {max(fitness_values):.4f}")
        sections.append(f"Fitness Mean: {sum(fitness_values)/len(fitness_values):.4f}")
    
    # Show last 3 episodes in detail
    for i, t in enumerate(telemetry[:3]):
        sections.append(f"Episode {t.get('episode_id', i)}:")
        sections.append(f"  Model: {t.get('best_model_type', '?')}, Fitness: {t.get('best_fitness', 0.0):.4f}")
        if 'val_r2' in t:
            sections.append(f"  Val R²: {t['val_r2']:.4f}")
        if 'fold_stability' in t:
            sections.append(f"  Fold Stability: {t['fold_stability']:.4f}")
    sections.append("")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1: Policy History
    # ══════════════════════════════════════════════════════════════════════════
    
    sections.append(f"=== POLICY HISTORY (Last {len(policy_history)} policies) ===")
    
    promoted = [p for p in policy_history if p.get('status') == 'promoted']
    rejected = [p for p in policy_history if p.get('status') == 'rejected']
    
    sections.append(f"Promoted: {len(promoted)}, Rejected: {len(rejected)}")
    
    for p in policy_history[:5]:
        status = p.get('status', 'unknown')
        reason = p.get('reason', 'no reason provided')
        draw = p.get('draw_number', '?')
        sections.append(f"  Draw {draw}: {status.upper()} — {reason[:80]}")
    sections.append("")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TIER 1: Current Configuration
    # ══════════════════════════════════════════════════════════════════════════
    
    if current_config:
        sections.append("=== CURRENT CONFIGURATION ===")
        for key, value in current_config.items():
            sections.append(f"  {key}: {value}")
        sections.append("")
    
    # ══════════════════════════════════════════════════════════════════════════
    # TASK INSTRUCTIONS
    # ══════════════════════════════════════════════════════════════════════════
    
    sections.append("=== YOUR TASKS ===")
    sections.append("1. Review the computed metrics (PCS, CC, FPD, MDI, SCS)")
    sections.append("2. Classify primary and secondary focus area based on thresholds")
    sections.append("3. Propose selfplay overrides with rationale")
    sections.append("4. Propose 0-5 parameter adjustments (MUST cite data)")
    sections.append("5. Assess risk level and human review necessity")
    sections.append("6. State alternative hypothesis with probability")
    sections.append("")
    sections.append("Respond with ONLY valid JSON matching the strategy_advisor.gbnf grammar.")
    sections.append("Do NOT include any text outside the JSON object.")
    
    # ══════════════════════════════════════════════════════════════════════════
    # ASSEMBLE PROMPT
    # ══════════════════════════════════════════════════════════════════════════
    
    prompt = "\n".join(sections)
    
    # Estimate tokens and truncate if needed
    estimated_tokens = len(prompt.split()) * 1.3
    if estimated_tokens > token_budget:
        # Truncate from the middle (diagnostics/telemetry sections)
        words = prompt.split()
        target_words = int(token_budget / 1.3)
        if target_words < len(words):
            # Keep first 40% and last 20%, truncate middle
            keep_front = int(target_words * 0.4)
            keep_back = int(target_words * 0.2)
            prompt = " ".join(words[:keep_front]) + \
                     "\n\n[... truncated to fit token budget ...]\n\n" + \
                     " ".join(words[-keep_back:])
    
    return prompt


def build_advisor_context(
    state_dir: str = ".",
    diagnostics_count: int = 20,
    telemetry_count: int = 20,
) -> tuple:
    """Convenience function to load data and build advisor bundle.
    
    Args:
        state_dir: Directory containing diagnostics, telemetry, policies.
        diagnostics_count: Number of recent diagnostics to load.
        telemetry_count: Number of recent telemetry records to load.
        
    Returns:
        (prompt, grammar_name) tuple.
    """
    # Import loader from parameter_advisor
    try:
        from parameter_advisor import DiagnosticsLoader, MetricsComputer, StrategyAdvisor
    except ImportError:
        # Try relative import
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from parameter_advisor import DiagnosticsLoader, MetricsComputer, StrategyAdvisor
    
    loader = DiagnosticsLoader(state_dir)
    diagnostics = loader.load_recent_diagnostics(diagnostics_count)
    telemetry = loader.load_telemetry(telemetry_count)
    policy_history = loader.load_policy_history(10)
    
    # Compute metrics (simplified — full computation in StrategyAdvisor)
    metrics = {
        'pcs': 0.0,
        'cc': 0.0,
        'fpd': 0.0,
        'mdi': 0.0,
        'scs': 0.0,
    }
    
    # Extract signals (simplified)
    signals = {
        'hit_at_20': 0.0,
        'hit_at_100': 0.0,
        'hit_at_300': 0.0,
        'model_dominance': 0.0,
        'best_model_type': 'catboost',
        'feature_drift': 0.0,
        'window_decay': 0.0,
        'draws_since_last_promotion': 0,
    }
    
    # Fill from diagnostics if available
    if diagnostics:
        for key in ['hit_at_20', 'hit_at_100', 'hit_at_300']:
            values = [d.get(key, 0.0) for d in diagnostics if key in d]
            if values:
                signals[key] = sum(values) / len(values)
        
        signals['feature_drift'] = diagnostics[0].get('feature_drift', 0.0)
        signals['window_decay'] = diagnostics[0].get('window_decay', 0.0)
    
    prompt = build_advisor_bundle(
        diagnostics=diagnostics,
        telemetry=telemetry,
        policy_history=policy_history,
        metrics=metrics,
        signals=signals,
    )
    
    return prompt, ADVISOR_GRAMMAR


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Advisor Bundle v1.0.0 — Self-Test")
    print("=" * 70)
    
    # Create mock data
    mock_diagnostics = [
        {
            "draw_number": 100,
            "timestamp": "2026-02-07T10:00:00Z",
            "hit_at_20": 0.04,
            "hit_at_100": 0.73,
            "hit_at_300": 0.91,
            "confidence_calibration": 0.42,
            "feature_drift": 0.15,
        },
        {
            "draw_number": 99,
            "timestamp": "2026-02-06T10:00:00Z",
            "hit_at_20": 0.05,
            "hit_at_100": 0.71,
            "hit_at_300": 0.89,
        },
    ]
    
    mock_telemetry = [
        {"episode_id": 1, "best_model_type": "catboost", "best_fitness": 0.72, "val_r2": 0.61},
        {"episode_id": 2, "best_model_type": "catboost", "best_fitness": 0.74, "val_r2": 0.63},
        {"episode_id": 3, "best_model_type": "lightgbm", "best_fitness": 0.68, "val_r2": 0.58},
    ]
    
    mock_policy_history = [
        {"status": "promoted", "reason": "Improved Hit@100 by 5%", "draw_number": 95},
        {"status": "rejected", "reason": "Regression on Hit@20", "draw_number": 98},
    ]
    
    mock_metrics = {
        "pcs": 0.25,
        "cc": 0.42,
        "fpd": 0.15,
        "mdi": 0.55,
        "scs": 0.78,
    }
    
    mock_signals = {
        "hit_at_20": 0.045,
        "hit_at_100": 0.72,
        "hit_at_300": 0.90,
        "model_dominance": 0.67,
        "best_model_type": "catboost",
        "feature_drift": 0.15,
        "window_decay": 0.10,
        "draws_since_last_promotion": 5,
    }
    
    # Build bundle
    prompt = build_advisor_bundle(
        diagnostics=mock_diagnostics,
        telemetry=mock_telemetry,
        policy_history=mock_policy_history,
        metrics=mock_metrics,
        signals=mock_signals,
    )
    
    print(f"\nPrompt length: {len(prompt)} chars")
    print(f"Estimated tokens: {int(len(prompt.split()) * 1.3)}")
    print(f"\n--- First 2000 chars ---")
    print(prompt[:2000])
    print("\n--- Last 500 chars ---")
    print(prompt[-500:])
    
    print("\n" + "=" * 70)
    print("Self-test complete")
    print("=" * 70)
