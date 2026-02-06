#!/usr/bin/env python3
"""
Bundle Factory v1.1.0 Patch — MAIN_MISSION + Selfplay Evaluation Mission

Apply to: agents/contexts/bundle_factory.py on Zeus
Date: 2026-02-05
Authority: Team Alpha recommendation, pending Team Beta approval
Closes: Chapter 10 §10 Known Gap (MAIN_MISSION in doctrine.py → now in bundle_factory.py)
Adds: Selfplay evaluation context for WATCHER/Chapter 13 selfplay outcome evaluation

GRAMMAR ALIGNMENT NOTE:
    SELFPLAY_SCHEMA_EXCERPT describes the agent_decision.gbnf output format
    ({decision, confidence, reasoning}) — NOT custom fields. The evaluation
    criteria (candidate quality, exploration breadth, fitness trajectory,
    bottleneck alignment, integration risk) are instructed to go INSIDE the
    reasoning string, matching the same pattern Steps 1-6 use. No new grammar,
    Pydantic model, or agent manifest is required.

PATCH INSTRUCTIONS:
    1. Add MAIN_MISSION constant after the imports (before STEP_MISSIONS)
    2. Add SELFPLAY_EVALUATION_MISSION, schema, guardrails after CHAPTER_13 constants
    3. Add step_id=99 ("selfplay_evaluation") to step_names dict
    4. Add selfplay handling to Tier 0 resolution in build_step_awareness_bundle()
    5. Prepend MAIN_MISSION in render_prompt_from_bundle()
    6. Update bundle_version to "1.1.0"
    7. Add selfplay bundle to self-test

Total delta: ~85 lines added, 3 lines modified.
No lines removed. No API changes. Fully backward-compatible.
"""

# ═══════════════════════════════════════════════════════════════════════════
# ADDITION 1: MAIN_MISSION constant
# Insert AFTER line 70 (logger = ...) and BEFORE line 72 (STEP MISSION header)
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# ADDITION 2: SELFPLAY_EVALUATION_MISSION + schema + grammar + guardrails
# Insert AFTER CHAPTER_13_GUARDRAILS (line 233) and BEFORE AUTHORITY_CONTRACTS
# ═══════════════════════════════════════════════════════════════════════════

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

SELFPLAY_GRAMMAR = "agent_decision.gbnf"  # Reuse existing; no new grammar needed

SELFPLAY_GUARDRAILS = [
    "Selfplay candidates are HYPOTHESES — you evaluate, Chapter 13 promotes.",
    "Selfplay uses historical data ONLY — never assume it validated against live draws.",
    "Mode collapse (all episodes converging to same parameters) is a critical warning.",
    "A candidate that improves one model type but degrades others needs careful weighting.",
    "Exploration breadth matters: a moderate-fitness diverse search beats a high-fitness narrow one.",
    "Do not penalize candidates for low absolute fitness — trajectory matters more than level.",
    "WATCHER executes; Chapter 13 decides; selfplay explores. You evaluate.",
]


# ═══════════════════════════════════════════════════════════════════════════
# MODIFICATION 1: step_names dict in build_step_awareness_bundle()
# Add entry for selfplay evaluation (step_id=99)
# ═══════════════════════════════════════════════════════════════════════════

# In the step_names dict (around line 526), ADD:
#     99: "selfplay_evaluation",

# The full dict becomes:
UPDATED_STEP_NAMES = {
    1: "window_optimizer",
    2: "scorer_meta_optimizer",
    3: "full_scoring",
    4: "ml_meta_optimizer",
    5: "anti_overfit_training",
    6: "prediction_generator",
    13: "chapter_13_feedback",
    99: "selfplay_evaluation",     # <-- NEW
}


# ═══════════════════════════════════════════════════════════════════════════
# MODIFICATION 2: Tier 0 resolution in build_step_awareness_bundle()
# Add selfplay handling branch BEFORE the generic else clause
# ═══════════════════════════════════════════════════════════════════════════

# Replace the Tier 0 block (lines 537-547) with this expanded version:
"""
    # — Tier 0: Mission + Schema + Grammar + Guardrails ——————————
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
"""


# ═══════════════════════════════════════════════════════════════════════════
# MODIFICATION 3: render_prompt_from_bundle() — prepend MAIN_MISSION
# Insert MAIN_MISSION as the FIRST section, before step-specific mission
# ═══════════════════════════════════════════════════════════════════════════

# In render_prompt_from_bundle() (line 687), REPLACE:
#     tier0_parts.append(f"=== STEP MISSION ===\n{ctx.mission}")
# WITH:
#     tier0_parts.append(f"=== SYSTEM MISSION ===\n{MAIN_MISSION}")
#     tier0_parts.append(f"=== STEP MISSION ===\n{ctx.mission}")

# This adds ~65 tokens to Tier 0. The existing tier0_reserved_tokens=1200
# has headroom (current Tier 0 uses ~400-600 tokens). No budget adjustment needed.


# ═══════════════════════════════════════════════════════════════════════════
# MODIFICATION 4: bundle_version bump
# ═══════════════════════════════════════════════════════════════════════════

# In StepAwarenessBundle class (line 370), change:
#     bundle_version: str = "1.0.0"
# TO:
#     bundle_version: str = "1.1.0"


# ═══════════════════════════════════════════════════════════════════════════
# ADDITION 3: Self-test for selfplay bundle
# Insert after the Chapter 13 test (around line 864)
# ═══════════════════════════════════════════════════════════════════════════

SELFTEST_ADDITION = """
    # Test 2b: Build Selfplay Evaluation bundle
    sp_bundle = build_step_awareness_bundle(
        step_id=99,
        results={
            "episodes_completed": 8,
            "candidates_emitted": 3,
            "best_fitness": 0.72,
            "exploration_breadth": 0.65,
            "mode_collapse_detected": False,
        },
    )
    print(f"\\n  Selfplay Evaluation ({sp_bundle.step_name}):")
    print(f"    Mission: {sp_bundle.context.mission[:80]}...")
    print(f"    Grammar: {sp_bundle.context.grammar_name}")
    print(f"    Guardrails: {len(sp_bundle.context.guardrails)}")

    # Test 2c: Verify MAIN_MISSION appears in rendered prompt
    sp_prompt = render_prompt_from_bundle(sp_bundle)
    assert "SYSTEM MISSION" in sp_prompt, "MAIN_MISSION not prepended!"
    assert "functional mimicry" in sp_prompt, "MAIN_MISSION content missing!"
    print(f"    MAIN_MISSION injected: ✓")
"""


# ═══════════════════════════════════════════════════════════════════════════
# WHAT THE RENDERED PROMPT NOW LOOKS LIKE (example for Step 3)
# ═══════════════════════════════════════════════════════════════════════════

EXAMPLE_RENDERED_PROMPT = """
=== SYSTEM MISSION ===
You are an evaluator within a distributed PRNG analysis system that uses
functional mimicry — learning surface-level output patterns and statistical
heuristics from PRNG-generated sequences to predict future draws, rather than
attempting to discover or reconstruct actual seeds. The system operates a
6-step pipeline (window optimization → sieve → scoring → ML architecture →
anti-overfit training → prediction generation) across a 26-GPU cluster.
Success is measured by hit rate and confidence calibration on held-out draws.
All evaluations serve this goal: improving the system's ability to learn
exploitable patterns in PRNG output through iterative ML refinement,
selfplay exploration, and autonomous feedback loops.

=== STEP MISSION ===
Step 3 — Full Scoring: Distributed 46-feature extraction and scoring
of all survivors across 26 GPUs via scripts_coordinator.py.
[... rest of Step 3 mission ...]

=== EVALUATION SCHEMA ===
[... step schema ...]

=== OUTPUT FORMAT ===
Respond using grammar: agent_decision.gbnf
Your output MUST conform to this grammar exactly.

=== GUARDRAILS ===
[... step guardrails ...]

=== AUTHORITY CONTRACTS ===
Active contracts: CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md
You cannot override these. Propose within their bounds only.

=== CURRENT EVALUATION ===
[... evaluation data ...]

=== STEP OUTPUT METRICS ===
[... metrics ...]
"""


# ═══════════════════════════════════════════════════════════════════════════
# TOKEN BUDGET IMPACT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
#
# MAIN_MISSION text: ~90 words → ~117 tokens (at 1.3 tokens/word)
# Section header "=== SYSTEM MISSION ===\n": ~5 tokens
# Total Tier 0 increase: ~122 tokens
#
# Current tier0_reserved_tokens: 1200
# Current typical Tier 0 usage: ~400-600 tokens
# New typical Tier 0 usage: ~520-720 tokens
# Headroom remaining: ~480-680 tokens
#
# VERDICT: No budget adjustment needed. Tier 0 reserve has ample headroom.
#
# If Chapter 14's DIAGNOSTICS_MISSION is later added (another ~80 tokens),
# total Tier 0 would still fit within 1200.
#
# ═══════════════════════════════════════════════════════════════════════════
# DISPATCH USAGE — HOW TO CALL SELFPLAY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
#
# In watcher_dispatch.py, when evaluating selfplay outcomes:
#
#   prompt, grammar, bundle = build_llm_context(
#       step_id=99,
#       results={
#           "episodes_completed": episode_count,
#           "candidates_emitted": len(candidates),
#           "best_fitness": best_fitness_score,
#           "exploration_breadth": diversity_metric,
#           "mode_collapse_detected": collapse_flag,
#           "parameter_ranges_explored": param_summary,
#       },
#   )
#   llm_output = call_llm(prompt, grammar=grammar)
#
# No new dispatch function needed — existing build_llm_context() handles it
# through step_id=99 routing.
#
# ═══════════════════════════════════════════════════════════════════════════
