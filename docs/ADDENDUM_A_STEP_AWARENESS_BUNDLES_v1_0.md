# Addendum A: Step Awareness Bundle Specification

**Version:** 1.0.0  
**Date:** 2026-02-02  
**Status:** LOCKED — Joint Alpha + Beta Approval  
**Implements:** PROPOSAL_PHASE7_INFRA_HYBRID_v1.2.0  
**Code:** `agents/contexts/bundle_factory.py`

---

## Purpose

This document is the **contract** that Phase 7 dispatch code and future retrieval enhancements code against. It defines the Step Awareness Bundle — the single structured context object passed to LLMs for all pipeline evaluation decisions.

**Governing sentence:**  
*We're aligned: we'll implement `bundle_factory.py` and Addendum A first, wire Phase 7 dispatch against `build_step_awareness_bundle()` immediately, and treat retrieval as an internal enhancement behind that interface. Dispatch logic will never assemble context directly.*

---

## A1. Bundle Object Model

### A1.1 StepAwarenessBundle

The top-level object. Immutable after creation.

```
StepAwarenessBundle
├── bundle_version: str          # "1.0.0"
├── step_id: int                 # 1-6 or 13
├── step_name: str               # "full_scoring", "chapter_13_feedback", etc.
├── run_id: str                  # Unique run identifier
├── is_chapter_13: bool          # True for Chapter 13 evaluations
├── context: BundleContext        # The context payload (see A1.2)
├── budgets: TokenBudget          # Token allocation (see A2)
├── provenance: List[ProvenanceRecord]  # Source file audit trail
└── generated_at: str            # UTC ISO timestamp
```

**Immutability guarantee:** `Config.frozen = True`. Once built, the bundle cannot be modified. The LLM receives it read-only.

### A1.2 BundleContext

The structured context payload, organized by tier priority.

```
BundleContext
├── [Tier 0 — Always included]
│   ├── mission: str                    # Step mission statement
│   ├── schema_excerpt: str             # Pydantic evaluation schema summary
│   ├── grammar_name: str               # GBNF grammar file name
│   ├── contracts: List[str]            # Active authority contract filenames
│   └── guardrails: List[str]           # Per-step hard constraints
│
├── [Tier 1 — Include if available]
│   ├── inputs_summary: Dict[str, Any]  # Key metrics from step results
│   └── evaluation_summary: Dict        # {success, confidence, interpretation}
│
└── [Tier 2 — Fill remaining tokens (Track 2 retrieval)]
    ├── recent_outcomes: List[OutcomeRecord]  # Chapter 13-curated history
    ├── trend_summary: List[TrendSummary]     # Rolling metric deltas
    └── open_incidents: List[str]             # Unresolved watcher failures
```

### A1.3 Supporting Models

| Model | Purpose | Fields |
|-------|---------|--------|
| `TokenBudget` | Token allocation policy | ctx_max_tokens, tier0_reserved, tier1_cap, history_cap, telemetry_cap, generation_reserve |
| `ProvenanceRecord` | Source file audit | path, sha256, size_bytes |
| `OutcomeRecord` | Structured Chapter 13 outcome | step, run_id, result, metric_delta, key_metric, timestamp |
| `TrendSummary` | Rolling metric computation | metric_name, direction, recent_values, window_size |

---

## A2. Token Budget Policy

### A2.1 Default Budget (32K context window)

| Tier | Allocation | Content |
|------|-----------|---------|
| **Tier 0** | 1,200 tokens reserved | Mission + schema + grammar + guardrails + contracts |
| **Tier 1** | 3,000 tokens cap | Step inputs summary + evaluation data |
| **Tier 2** | 9,000 tokens cap | History (6,000) + telemetry (3,000) |
| **Generation** | 2,000 tokens reserved | LLM output generation |
| **Available** | ~16,568 tokens | Headroom for prompt framing |

### A2.2 Budget Enforcement Rules

1. **Tier 0 is non-negotiable.** If Tier 0 exceeds its reservation, it still gets included — other tiers shrink.
2. **Tier 1 is capped.** Content truncated with `[... truncated to fit token budget ...]` marker.
3. **Tier 2 fills remaining.** History first, then telemetry. Each has individual caps.
4. **Generation reserve is sacred.** Prompt must leave room for LLM output.
5. **Token estimation:** `~1.3 tokens per word` approximation (no tiktoken dependency).

### A2.3 Custom Budgets

Dispatch functions may override budgets for specific use cases:

```python
tight_budget = TokenBudget(ctx_max_tokens=4096, generation_reserve_tokens=500)
bundle = build_step_awareness_bundle(step_id=1, budgets=tight_budget, ...)
```

---

## A3. Retrieval Source Hierarchy

### A3.1 Priority Order (v1.0 — static assembly)

| Priority | Source | Available in v1.0 |
|----------|--------|-------------------|
| 1 | Step mission statement (hardcoded) | ✅ Yes |
| 2 | Pydantic evaluation schema excerpt (hardcoded) | ✅ Yes |
| 3 | Grammar name (hardcoded mapping) | ✅ Yes |
| 4 | Authority contracts (hardcoded list) | ✅ Yes |
| 5 | Per-step guardrails (hardcoded) | ✅ Yes |
| 6 | Evaluation data from `build_full_context()` | ✅ Yes |
| 7 | Key metrics from results dict | ✅ Yes |
| 8 | Chapter 13-curated outcome summaries | ❌ Stub (Track 2) |
| 9 | Computed metric trend summaries | ❌ Stub (Track 2) |
| 10 | Open watcher incident summaries | ❌ Stub (Track 2) |

### A3.2 Track 2 Retrieval (future — behind same API)

Track 2 fills stubs 8-10 by reading:

| Stub | Source File | Reader |
|------|-------------|--------|
| `_retrieve_recent_outcomes()` | `chapter13/summaries/*.json` | Structured JSON reader |
| `_retrieve_trend_summary()` | `run_history.jsonl` | Rolling delta computation |
| `_retrieve_open_incidents()` | `watcher_failures.jsonl` | Unresolved filter |

**Contract:** Track 2 changes are INTERNAL to bundle_factory.py. Dispatch code does not change. The three retrieval stubs are the only modification points.

### A3.3 What Is Explicitly Excluded

| Excluded Source | Reason |
|-----------------|--------|
| Raw decision logs | Too noisy, unbounded size |
| Full conversation history | Not deterministic |
| Vector embeddings | No vector DB in architecture |
| GPU-resident state | Ephemeral, not reproducible |
| Live draw outcomes | Chapter 13 exclusive authority |

---

## A4. Assembly Ownership

### A4.1 Controller-Only Assembly

**Rule:** Only the controller (WATCHER dispatch function) builds bundles. The LLM never pulls files, queries databases, or constructs its own context.

```
CORRECT:
    Controller → build_step_awareness_bundle() → render_prompt_from_bundle() → LLM

FORBIDDEN:
    Controller → LLM → "please read file X and evaluate"
    Controller → LLM → "search for recent outcomes in /path/to/logs"
```

### A4.2 Dispatch Code Pattern (Guardrail #1)

Every dispatch function MUST use this exact pattern:

```python
# CORRECT — uses single entry point
prompt, grammar_name, bundle = build_llm_context(
    step_id=step, run_id=run_id, results=results,
    manifest_path=manifest_path
)
llm_output = call_llm(prompt, grammar=grammar_name)

# FORBIDDEN — inline context assembly
prompt = f"You are evaluating step {step}. Results: {json.dumps(results)}"
```

### A4.3 No Baked-In Token Assumptions (Guardrail #2)

Dispatch functions MUST NOT:
- Hard-code token counts
- Fix prompt section ordering
- Inline history blobs
- Assume context window size

All of these are owned by `bundle_factory.py` and its `TokenBudget` model.

---

## A5. Determinism + Immutability Guarantees

### A5.1 Determinism

Given identical inputs `(step_id, results, run_number, manifest_path, state_paths)`:
- The bundle produces identical `context` content
- The prompt renders identically
- Provenance hashes are identical

**Exception:** `generated_at` timestamp and `run_id` (when auto-generated) will differ.

### A5.2 Immutability

`StepAwarenessBundle` is frozen after creation:
- No field modification allowed
- No in-prompt "edits" by the LLM
- No post-assembly context injection

### A5.3 No Side Effects

`build_step_awareness_bundle()` MUST NOT:
- Write files
- Modify global state
- Make network requests (Track 2 retrieval reads local files only)
- Trigger subprocess execution

---

## A6. Provenance Requirements

### A6.1 What Gets Hashed

Every file that contributes to the bundle context is recorded:

| Source | Hashed |
|--------|--------|
| Agent manifest JSON | ✅ SHA256 of file contents |
| State files (e.g., survivors, configs) | ✅ SHA256 of file contents |
| Hardcoded content (missions, schemas) | Not hashed (versioned in code) |

### A6.2 ProvenanceRecord Format

```json
{
    "path": "agent_manifests/full_scoring.json",
    "sha256": "a1b2c3d4e5f6...",
    "size_bytes": 2048
}
```

### A6.3 Missing File Handling

If a file is referenced but does not exist:
- `sha256` is set to `"FILE_NOT_FOUND"`
- `size_bytes` is 0
- Bundle assembly continues (does not fail)

---

## A7. Step-by-Step Minimum Bundle Contents

### A7.1 Steps 1-6 Minimum Contents

Every step bundle MUST include:

| Field | Source | Required |
|-------|--------|----------|
| `mission` | `STEP_MISSIONS[step_id]` | ✅ Always |
| `schema_excerpt` | `STEP_SCHEMA_EXCERPTS[step_id]` | ✅ Always |
| `grammar_name` | `STEP_GRAMMAR_NAMES[step_id]` | ✅ Always |
| `guardrails` | `STEP_GUARDRAILS[step_id]` | ✅ Always (may be empty list) |
| `contracts` | `AUTHORITY_CONTRACTS` | ✅ Always |
| `evaluation_summary` | From `build_full_context()` | ✅ If results provided |
| `inputs_summary` | From `get_key_metrics()` | ✅ If results provided |

### A7.2 Chapter 13 Minimum Contents

| Field | Source | Required |
|-------|--------|----------|
| `mission` | `CHAPTER_13_MISSION` | ✅ Always |
| `schema_excerpt` | `CHAPTER_13_SCHEMA_EXCERPT` | ✅ Always |
| `grammar_name` | `"chapter_13.gbnf"` | ✅ Always |
| `guardrails` | `CHAPTER_13_GUARDRAILS` | ✅ Always |
| `contracts` | `AUTHORITY_CONTRACTS` | ✅ Always |
| `evaluation_summary` | From results dict directly | ✅ If results provided |

### A7.3 Per-Step Grammar Mapping

| Step | Grammar File | Fallback |
|------|-------------|----------|
| 1 — Window Optimizer | `agent_decision.gbnf` | `json_generic.gbnf` |
| 2 — Scorer Meta | `sieve_analysis.gbnf` | `json_generic.gbnf` |
| 3 — Full Scoring | `agent_decision.gbnf` | `json_generic.gbnf` |
| 4 — ML Meta | `agent_decision.gbnf` | `json_generic.gbnf` |
| 5 — Anti-Overfit | `agent_decision.gbnf` | `json_generic.gbnf` |
| 6 — Prediction | `agent_decision.gbnf` | `json_generic.gbnf` |
| 13 — Chapter 13 | `chapter_13.gbnf` | N/A |

### A7.4 Per-Step Key Metrics (for inputs_summary)

| Step | Key Metrics |
|------|-------------|
| 1 | survivor_count, bidirectional_count, forward_count, reverse_count, precision, recall |
| 2 | best_accuracy, completed_trials, convergence_rate |
| 3 | completion_rate, survivors_scored, survivors_total, feature_dimensions, mean_score, score_std, top_candidates |
| 4 | best_r2, validation_r2, architecture_stability |
| 5 | val_r2, test_r2, overfit_ratio, feature_importance_entropy |
| 6 | prediction_count, mean_confidence, pool_coverage, feature_hash_valid |
| 13 | drift_detected, metric_delta, trigger_action, retrain_recommended |

---

## A8. API Reference

### A8.1 Primary Entry Point

```python
def build_step_awareness_bundle(
    step_id: int,                              # 1-6 or 13
    run_id: str = "",                          # Auto-generated if empty
    results: Optional[Dict[str, Any]] = None,  # Step output to evaluate
    run_number: int = 1,                       # Current run number
    manifest_path: Optional[str] = None,       # Agent manifest JSON path
    state_paths: Optional[List[str]] = None,   # Additional provenance files
    budgets: Optional[TokenBudget] = None,     # Override default budgets
    is_chapter_13: bool = False,               # Chapter 13 mode
) -> StepAwarenessBundle:
```

### A8.2 Prompt Renderer

```python
def render_prompt_from_bundle(bundle: StepAwarenessBundle) -> str:
```

### A8.3 Convenience Function (Guardrail #1 Compliance)

```python
def build_llm_context(
    step_id: int,
    run_id: str = "",
    results: Optional[Dict[str, Any]] = None,
    run_number: int = 1,
    manifest_path: Optional[str] = None,
    state_paths: Optional[List[str]] = None,
    is_chapter_13: bool = False,
) -> tuple:  # (prompt, grammar_name, bundle)
```

---

## A9. Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-02-02 | Initial specification. Static assembly, retrieval stubs. |

---

## A10. Out of Scope (Explicitly Excluded)

These items are NOT part of this specification and MUST NOT be added without a new Addendum:

| Item | Status |
|------|--------|
| Vector database / embeddings | ❌ Excluded |
| GPU-resident services | ❌ Excluded |
| Raw decision log ingestion | ❌ Excluded |
| Authority changes | ❌ Excluded |
| Dispatch logic | ❌ Separate (Part B of TODO) |
| Addendum B (Authority Contract) | ⏳ Optional, not blocking |

---

**END OF ADDENDUM A**
