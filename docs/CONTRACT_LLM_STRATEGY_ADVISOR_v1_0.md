# CONTRACT: LLM-Guided Selfplay Strategy Advisor

## Version 1.0 — February 3, 2026

**Status:** PROPOSED  
**Author:** Team Beta  
**Depends On:** Chapter 13 (Live Feedback Loop), Selfplay Contract v1.1, Phase 7 (WATCHER Integration)  
**Implements:** Deferred Item B (`parameter_advisor.py`)  
**Prerequisite:** Chapter 13 accumulates ≥15 real draw diagnostics before activation

---

## The One-Sentence Rule

**Chapter 13 measures. The Advisor interprets. Selfplay explores. WATCHER enforces.**

---

## 1. Purpose

Selfplay currently explores parameter space without strategic direction. It runs N episodes, trains tree models, tracks the best fitness, and emits a candidate. It does not know *why* the last candidate succeeded or failed, *which* proxy rewards are most correlated with actual hit rate, or *where* in parameter space the highest-value opportunities lie.

This contract defines a **Strategy Advisor** — an LLM-powered analytical layer that consumes Chapter 13 diagnostics and telemetry history to produce mathematically grounded, auditable recommendations for selfplay focus. It bridges the gap between "explore everything equally" and "explore what the evidence says matters."

**What this is:** A structured analytical advisor that transforms diagnostic data into actionable selfplay guidance.

**What this is NOT:** A controller, executor, or autonomous decision-maker. All recommendations are proposals that must pass WATCHER validation before affecting selfplay behavior.

---

## 2. Authority Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                      CHAPTER 13                             │
│                 (Governor / Arbiter)                        │
│                                                             │
│  PRODUCES:                                                  │
│    • post_draw_diagnostics.json (per draw)                 │
│    • diagnostics_history/ (accumulated archive)             │
│    • Hit rate metrics (Hit@20, Hit@100, Hit@300)           │
│    • Confidence calibration data                            │
│    • Survivor performance tracking                          │
│                                                             │
│  FEEDS ──────────────────────────────────┐                  │
└──────────────────────────────────────────┼──────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   STRATEGY ADVISOR (LLM)                    │
│                 (Analyst / Recommender)                     │
│                                                             │
│  CONSUMES:                                                  │
│    • Chapter 13 diagnostics (last N draws)                 │
│    • Telemetry history (selfplay episodes)                 │
│    • Pool performance metrics                               │
│    • Policy history (promoted + rejected)                   │
│                                                             │
│  PRODUCES:                                                  │
│    • strategy_recommendation.json (structured proposal)    │
│    • Focus area classification                              │
│    • Parameter adjustment proposals                         │
│    • Selfplay configuration overrides                       │
│                                                             │
│  MAY: Analyze, classify, recommend, rank                   │
│  NEVER: Execute, modify files, bypass WATCHER, promote     │
└─────────────────────────────────────────┬───────────────────┘
                                          │
                                validates │
                                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      WATCHER AGENT                          │
│                 (Validator / Enforcer)                      │
│                                                             │
│  VALIDATES:                                                 │
│    • Proposals within bounds (Section 18 parameter table)  │
│    • Cooldown periods respected                             │
│    • Change limits not exceeded                             │
│    • No frozen parameters modified                          │
│                                                             │
│  EXECUTES:                                                  │
│    • Applies approved selfplay config overrides             │
│    • Dispatches focused selfplay with strategy hints        │
│                                                             │
│  MAY: Accept, reject, or partially apply recommendations   │
│  NEVER: Override Chapter 13, trust advisor without checks  │
└─────────────────────────────────────────────────────────────┘
                                          │
                              dispatches  │
                                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      SELFPLAY                               │
│                 (Explorer / Hypothesis Generator)           │
│                                                             │
│  RECEIVES:                                                  │
│    • Strategy hints (focus area, priority metrics)          │
│    • Optional parameter overrides                           │
│    • Exploration vs exploitation ratio guidance             │
│                                                             │
│  EXECUTES:                                                  │
│    • Episodes with strategy-informed configuration          │
│    • Emits candidates with strategy provenance              │
│                                                             │
│  MAY: Use hints to weight exploration                      │
│  NEVER: Access ground truth, promote policies              │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. What the Advisor Analyzes

### 3.1 Primary Diagnostic Signals

These signals come from Chapter 13's `post_draw_diagnostics.json` after each real draw:

| Signal | Source | What It Reveals |
|--------|--------|-----------------|
| Hit@20 rate (last N draws) | Chapter 13 | Tight pool precision — are top predictions landing? |
| Hit@100 rate (last N draws) | Chapter 13 | Balanced pool coverage — is the system capturing draws? |
| Hit@300 rate (last N draws) | Chapter 13 | Wide pool safety net — baseline coverage health |
| Confidence calibration | Chapter 13 | Are high-confidence predictions more accurate than low? |
| Confidence drift | Chapter 13 | Is calibration stable or degrading over time? |
| Survivor churn | Chapter 13 | Are the same seeds performing or is the pool volatile? |
| Feature drift | Chapter 13 | Are predictive features stable or shifting? |
| Window decay | Chapter 13 | Is the optimal window configuration still valid? |

### 3.2 Telemetry Signals

These come from selfplay's `telemetry/` output:

| Signal | Source | What It Reveals |
|--------|--------|-----------------|
| Fitness distribution by model type | Selfplay telemetry | Which model families produce best results? |
| Validation R² trend | Selfplay telemetry | Is model quality improving or plateauing? |
| Fold stability (std across k-folds) | Selfplay telemetry | Overfitting vs generalization |
| Train/val gap trend | Selfplay telemetry | Is the gap widening (overfit) or narrowing? |
| Episode-over-episode fitness delta | Selfplay telemetry | Diminishing returns detection |
| Policy promotion/rejection history | policy_history/ | What worked, what didn't, and why |

### 3.3 Pool Performance Signals

These are computed from Chapter 13's pool tracking:

| Signal | Computation | Strategic Meaning |
|--------|-------------|-------------------|
| Pool concentration ratio | weight(Top20) / weight(Top100) | How focused are predictions? |
| Pool stability | Jaccard(pool_t, pool_t-1) | Are the same numbers appearing? |
| Lift vs random | Hit@K / (K/1000) | How much better than chance? |
| Miss pattern | Where draws land vs pool boundaries | Near-miss → pool too tight? Far-miss → wrong regime? |
| Weight entropy | -Σ p·log(p) for vote distribution | High = uniform (low confidence), Low = concentrated |

---

## 4. Focus Area Classification

The Advisor classifies the system's current state into one of seven focus areas. Each maps to specific selfplay configuration adjustments.

### 4.1 Focus Area Definitions

| Focus Area | Trigger Condition | Selfplay Directive |
|------------|-------------------|-------------------|
| **POOL_PRECISION** | Hit@100 > 70% but Hit@20 < 10% | Optimize for pool concentration; weight top-survivor agreement |
| **POOL_COVERAGE** | Hit@300 < 80% | Broaden exploration; increase episode count; diversify model types |
| **CONFIDENCE_CALIBRATION** | Calibration correlation < 0.3 | Focus on fold stability and train/val gap minimization |
| **MODEL_DIVERSITY** | Single model type dominates (>80% of best episodes) | Force model rotation; increase diversity in episodes |
| **FEATURE_RELEVANCE** | Feature drift > 0.3 | Signal that features may be stale; recommend learning loop |
| **REGIME_SHIFT** | Window decay > 0.5 AND survivor churn > 0.4 | Full pipeline rerun recommended; selfplay paused |
| **STEADY_STATE** | All metrics within bounds | Reduce episode count; maintenance-mode exploration |

### 4.2 Focus Area Priority (When Multiple Apply)

```
REGIME_SHIFT > POOL_COVERAGE > CONFIDENCE_CALIBRATION > 
POOL_PRECISION > MODEL_DIVERSITY > FEATURE_RELEVANCE > STEADY_STATE
```

Regime shift always takes priority because it invalidates all downstream optimization.

---

## 5. Recommendation Schema

The Advisor outputs a single `strategy_recommendation.json` per analysis cycle.

### 5.1 JSON Schema

```json
{
  "schema_version": "1.0.0",
  "generated_at": "2026-02-03T12:00:00Z",
  "advisor_model": "deepseek-r1-14b",
  "draws_analyzed": 20,
  
  "focus_area": "POOL_PRECISION",
  "focus_confidence": 0.82,
  "focus_rationale": "Hit@100 at 73% but Hit@20 at 4%. Pool is capturing draws but top predictions are unfocused.",
  
  "secondary_focus": "CONFIDENCE_CALIBRATION",
  "secondary_confidence": 0.61,
  
  "recommended_action": "RETRAIN",
  "retrain_scope": "selfplay_only",
  
  "selfplay_overrides": {
    "max_episodes": 15,
    "model_types": ["catboost", "lightgbm"],
    "min_fitness_threshold": 0.55,
    "priority_metrics": ["pool_concentration", "model_agreement"],
    "exploration_ratio": 0.3
  },
  
  "parameter_proposals": [
    {
      "parameter": "n_estimators",
      "current_value": 100,
      "proposed_value": 200,
      "delta": "+100",
      "confidence": 0.75,
      "rationale": "Underfitting signal: fold std decreasing but R² plateau suggests capacity limit"
    },
    {
      "parameter": "k_folds",
      "current_value": 3,
      "proposed_value": 5,
      "delta": "+2",
      "confidence": 0.68,
      "rationale": "Calibration drift correlates with fold count — more folds may stabilize"
    }
  ],
  
  "pool_strategy": {
    "tight_pool_guidance": "Increase weight on survivor consistency (current: 0.15, suggest: 0.25)",
    "balanced_pool_guidance": "Maintain current configuration — performing well",
    "wide_pool_guidance": "No change needed"
  },
  
  "risk_level": "low",
  "requires_human_review": false,
  
  "diagnostic_summary": {
    "hit_at_20": 0.04,
    "hit_at_100": 0.73,
    "hit_at_300": 0.91,
    "calibration_correlation": 0.42,
    "survivor_churn": 0.18,
    "best_model_type": "catboost",
    "fitness_trend": "plateau",
    "draws_since_last_promotion": 8
  },
  
  "alternative_hypothesis": "Random variance — 25% probability. Recommend WAIT if next 3 draws show Hit@20 improvement without intervention."
}
```

### 5.2 GBNF Grammar

The Advisor's output is grammar-constrained via `strategy_advisor.gbnf` to ensure structured, parseable JSON. This grammar extends the existing `chapter_13.gbnf` pattern.

```
# Strategy Advisor Grammar (summary — full file separate)

root ::= recommendation

recommendation ::= "{" ws
    "\"focus_area\"" ws ":" ws focus-area "," ws
    "\"focus_confidence\"" ws ":" ws confidence-value "," ws
    "\"focus_rationale\"" ws ":" ws string "," ws
    "\"recommended_action\"" ws ":" ws advisor-action "," ws
    "\"selfplay_overrides\"" ws ":" ws selfplay-overrides "," ws
    "\"parameter_proposals\"" ws ":" ws parameter-proposals "," ws
    "\"risk_level\"" ws ":" ws risk-level "," ws
    "\"requires_human_review\"" ws ":" ws boolean
    ws "}"

focus-area ::= "\"POOL_PRECISION\"" | "\"POOL_COVERAGE\"" |
               "\"CONFIDENCE_CALIBRATION\"" | "\"MODEL_DIVERSITY\"" |
               "\"FEATURE_RELEVANCE\"" | "\"REGIME_SHIFT\"" |
               "\"STEADY_STATE\""

advisor-action ::= "\"RETRAIN\"" | "\"WAIT\"" | "\"ESCALATE\"" |
                   "\"REFOCUS\"" | "\"FULL_RESET\""
```

---

## 6. Mathematical Framework

### 6.1 Pool Concentration Score

Measures how tightly prediction weight is concentrated in top candidates:

```
PCS = Σ(weight_i for i in Top-K) / Σ(weight_i for all i)

Where:
  K = tight pool size (default 20)
  weight_i = composite score × offset confidence × gap stability
  
Interpretation:
  PCS > 0.30 → Strong concentration (good for precision)
  PCS < 0.10 → Diffuse distribution (low confidence, broad exploration)
```

### 6.2 Calibration Correlation

Measures whether confidence scores predict accuracy:

```
CC = Pearson(confidence_bucket_mean, hit_rate_per_bucket)

Where:
  Bucket confidence into deciles [0.0-0.1, 0.1-0.2, ..., 0.9-1.0]
  For each bucket: hit_rate = hits_in_bucket / predictions_in_bucket
  
Interpretation:
  CC > 0.7 → Well-calibrated (confidence means something)
  CC < 0.3 → Poorly calibrated (confidence is noise)
  CC < 0.0 → Inversely calibrated (confident predictions are WORSE)
```

### 6.3 Fitness Plateau Detection

Detects diminishing returns in selfplay exploration:

```
FPD = slope(fitness_values[-10:]) / std(fitness_values[-10:])

Where:
  fitness_values = last 10 episode fitness scores
  
Interpretation:
  |FPD| < 0.1 → Plateau (exploration is not finding better regions)
  FPD > 0.5  → Still improving (continue current strategy)
  FPD < -0.3 → Regression (something is wrong — escalate)
```

### 6.4 Model Diversity Index

Measures whether selfplay is over-relying on one model family:

```
MDI = 1 - HHI(model_type_fitness_share)

Where:
  HHI = Herfindahl-Hirschman Index = Σ(share_i²)
  share_i = (count of best episodes by model_type_i) / total_best_episodes
  
Interpretation:
  MDI > 0.6 → Good diversity
  MDI < 0.3 → Over-concentration (single model dominance)
```

### 6.5 Survivor Consistency Score

Measures how stable the top survivors are across consecutive runs:

```
SCS = |intersection(top_survivors_t, top_survivors_t-1)| / 
      |union(top_survivors_t, top_survivors_t-1)|

This is the Jaccard similarity coefficient.

Interpretation:
  SCS > 0.7 → Highly stable (same seeds performing)
  SCS < 0.3 → Volatile (pool is churning — may indicate regime shift)
```

---

## 7. Advisor Prompt Template

The Advisor receives a structured prompt assembled by the bundle factory.

```
You are a quantitative strategy advisor for a probabilistic PRNG 
analysis system. You analyze diagnostic data to recommend where 
selfplay exploration should focus.

HARD CONSTRAINTS:
- You do NOT execute actions or modify files
- You do NOT guess numbers or predict draws
- You MUST express uncertainty via confidence scores
- You MUST justify every proposal with specific data points
- You MUST classify a focus area from the defined set
- If uncertainty is high (no clear signal), recommend WAIT

DIAGNOSTIC DATA:
{{ post_draw_diagnostics (last 15-20 draws) }}

TELEMETRY HISTORY:
{{ selfplay episode summaries (last 20 episodes) }}

POOL PERFORMANCE:
{{ pool hit rates, concentration, stability metrics }}

POLICY HISTORY:
{{ last 5 promoted + rejected policies with reasons }}

CURRENT CONFIGURATION:
{{ active selfplay config + parameter values }}

TASKS:
1. Compute and report: PCS, CC, FPD, MDI, SCS (Section 6 formulas)
2. Classify primary and secondary focus area
3. Propose selfplay overrides with rationale
4. Propose parameter adjustments (0-5) with bounds validation
5. Assess risk level and human review necessity
6. State alternative hypothesis with probability

Respond with ONLY valid JSON matching the strategy_advisor.gbnf grammar.
```

---

## 8. Implementation Plan

### 8.1 File Structure

| File | Purpose | Size Est. |
|------|---------|-----------|
| `parameter_advisor.py` | Main advisor module | ~400 lines |
| `strategy_advisor.gbnf` | Grammar constraint | ~80 lines |
| `strategy_recommendation.json` | Output file (overwritten each cycle) | ~2KB |
| `strategy_history/` | Archive of past recommendations | ~2KB each |
| `agents/contexts/advisor_bundle.py` | Bundle factory extension for advisor context | ~150 lines |

### 8.2 Integration Points

```
Chapter 13 Diagnostics
        │
        ▼
parameter_advisor.py
  ├── Loads diagnostics + telemetry + policy history
  ├── Computes mathematical signals (Section 6)
  ├── Builds advisor prompt (Section 7)
  ├── Calls LLM via LLMRouter (grammar-constrained)
  ├── Validates proposal bounds (Section 18 parameter table)
  ├── Writes strategy_recommendation.json
  └── Archives to strategy_history/
        │
        ▼
WATCHER reads strategy_recommendation.json
  ├── Validates against policies
  ├── Applies selfplay_overrides to next dispatch
  └── Logs decision
        │
        ▼
dispatch_selfplay() receives strategy hints
  ├── Passes overrides to selfplay_orchestrator.py
  └── Selfplay runs with informed configuration
```

### 8.3 Phased Rollout

| Phase | What | When | Prerequisite |
|-------|------|------|--------------|
| **Phase A** | Diagnostic collection only — no LLM | After soak test | Chapter 13 Phases 1-2 built |
| **Phase B** | LLM analysis with WAIT-only recommendations | After 15+ real draws | Phase A metrics stable |
| **Phase C** | Full recommendations with WATCHER validation | After 30+ real draws | Phase B proposals reviewed by human |
| **Phase D** | Autonomous strategy-guided selfplay | After 50+ real draws | Phase C proposals consistently reasonable |

### 8.4 Activation Gate

The Advisor MUST NOT activate until:

1. ✅ Chapter 13 has processed ≥15 real draws
2. ✅ `diagnostics_history/` contains ≥15 entries
3. ✅ Selfplay has completed ≥10 episodes with telemetry
4. ✅ At least 1 policy has been promoted (baseline exists)
5. ✅ Soak testing complete (Phase 7 validated)

Without sufficient data, LLM analysis produces noise, not signal.

---

## 9. High-Value Pool Development Strategy

### 9.1 The Pool Optimization Loop

Once Chapter 13 is running with real draws, the system can systematically optimize pool quality:

```
┌──────────────────────────────────────────────────────────┐
│                    POOL OPTIMIZATION LOOP                 │
│                                                           │
│  1. Generate predictions (Step 6)                        │
│     └── Produces Tight/Balanced/Wide pools               │
│                                                           │
│  2. Real draw occurs                                     │
│     └── Chapter 13 records Hit@20, Hit@100, Hit@300     │
│                                                           │
│  3. Advisor analyzes (this contract)                     │
│     └── Which pool tier is underperforming?              │
│     └── Why? (concentration, calibration, features?)     │
│     └── What should selfplay optimize for?               │
│                                                           │
│  4. Strategy-guided selfplay                             │
│     └── Explores with focus on weak pool tier            │
│     └── Emits candidate optimized for that focus         │
│                                                           │
│  5. Chapter 13 validates candidate                       │
│     └── Shadow eval against held-out draws               │
│     └── Promote if improvement, reject if regression     │
│                                                           │
│  6. Updated model generates new predictions              │
│     └── Return to step 1                                 │
└──────────────────────────────────────────────────────────┘
```

### 9.2 Pool-Specific Optimization Strategies

| Pool Tier | If Underperforming | Advisor Recommends |
|-----------|-------------------|-------------------|
| **Tight (Top 20)** | Hit@20 < 5% for 10+ draws | Focus: POOL_PRECISION — increase n_estimators, favor catboost/xgboost, weight survivor consistency and model agreement |
| **Balanced (Top 100)** | Hit@100 < 60% for 5+ draws | Focus: POOL_COVERAGE — diversify model types, increase episode count, lower min_fitness_threshold to explore |
| **Wide (Top 300)** | Hit@300 < 85% for 3+ draws | Focus: REGIME_SHIFT likely — recommend full pipeline rerun (Steps 1→6), window may be stale |

### 9.3 Weighting Strategy Optimization

The current weighting ratio (composite 0.60, offset 0.25, stability 0.15) is static. The Advisor can propose weight adjustments based on which signal dimension is most predictive:

```
If survivor_consistency correlates with hits:
  → Increase stability weight (0.15 → 0.25)
  → Decrease offset weight (0.25 → 0.15)

If composite_score correlates with hits:
  → Maintain current weights
  
If offset_confidence correlates with hits:
  → Increase offset weight (0.25 → 0.35)
  → Decrease composite weight (0.60 → 0.50)
```

These correlations are computed empirically from Chapter 13's accumulated data, not assumed.

---

## 10. External LLM Strategy Consultation

### 10.1 When to Use an External LLM (Claude)

The local LLM (DeepSeek-R1-14B) handles routine per-cycle analysis. For deeper strategic analysis, the system's diagnostics and telemetry can be brought to a top-tier LLM (Claude) for consultation:

| Scenario | Use Local (DeepSeek) | Use External (Claude) |
|----------|---------------------|-----------------------|
| Per-cycle focus classification | ✅ | |
| Routine parameter proposals | ✅ | |
| Multi-draw trend analysis (20+ draws) | | ✅ |
| Strategy pivot evaluation | | ✅ |
| Cross-PRNG-algorithm comparison | | ✅ |
| Feature engineering suggestions | | ✅ |
| Regime shift root cause analysis | | ✅ |

### 10.2 What to Bring to Claude

When consulting Claude for strategy review, provide:

1. **`diagnostics_history/`** — Last 20+ draw diagnostics
2. **`telemetry/`** — Last 20+ selfplay episode summaries  
3. **`policy_history/`** — All promoted and rejected policies
4. **`strategy_history/`** — Last 10 advisor recommendations
5. **Pool performance CSV** — Hit@K by draw, with timestamps
6. **Active configuration** — Current parameter values

This data contains no PII and is purely mathematical/statistical.

### 10.3 What Claude Can Uniquely Provide

- Cross-pattern analysis across PRNG algorithm families
- Identification of non-obvious feature interactions
- Novel fitness function proposals (beyond current proxy rewards)
- Mathematical proofs for concentration bound improvements
- Strategy recommendations that require reasoning beyond structured classification

---

## 11. Invariants

### Invariant 1: Advisory Only
```
The Advisor NEVER executes actions, modifies files, applies 
parameters, or bypasses any authority boundary. Every output 
is a proposal, never a command.
```

### Invariant 2: Evidence Required
```
Every parameter proposal MUST cite specific diagnostic values.
"Increase X" without data reference is INVALID.
```

### Invariant 3: Bounds Enforcement
```
All parameter proposals MUST fall within Section 18 bounds.
Out-of-bounds proposals are silently rejected by WATCHER.
```

### Invariant 4: Frozen Parameters
```
The Advisor MUST NEVER propose changes to: step ordering, 
feature schema, PRNG algorithms, sieve math, Pydantic schemas.
```

### Invariant 5: Data Minimum
```
The Advisor MUST NOT activate with fewer than 15 real draws 
in diagnostics_history/. Insufficient data produces noise.
```

### Invariant 6: Cooldown Respect
```
The Advisor MUST check cooldown_runs and 
parameter_change_limit from watcher_policies.json. 
Proposals that violate cooldown are deferred, not dropped.
```

### Invariant 7: Provenance
```
Every strategy_recommendation.json MUST include:
- draws_analyzed count
- advisor_model identifier
- timestamp
- all computed metrics (Section 6)
Recommendations without provenance are INVALID.
```

---

## 12. Forbidden Actions

### The Advisor MUST NOT:
- Execute code or modify files
- Access live draw outcomes directly (only via Chapter 13 diagnostics)
- Propose changes to frozen components
- Override WATCHER decisions
- Promote policies
- Bypass grammar constraints
- Produce recommendations without data (speculation)
- Recommend specific numbers or draws (this is not a predictor)

### WATCHER MUST NOT:
- Apply advisor recommendations without bounds validation
- Skip cooldown checks
- Apply recommendations from insufficient data (<15 draws)
- Trust advisor output as final (always validate)

---

## 13. Success Metrics

The Advisor's own performance is measured by:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Focus classification accuracy | >70% of recommendations lead to measurable improvement | Before/after comparison over 10 draws |
| Parameter proposal acceptance | >60% accepted by WATCHER | Accepted / total proposed |
| False escalation rate | <20% | Escalations that prove unnecessary |
| Hit rate improvement after strategy change | Positive trend | 5-draw rolling average |
| Time to regime shift detection | <3 draws after shift | Measured against retrospective analysis |

---

## 14. Relationship to Existing Components

| Component | Relationship to Advisor |
|-----------|------------------------|
| `chapter_13.gbnf` | Advisor uses similar grammar pattern; separate grammar file |
| `bundle_factory.py` | Extended with `build_advisor_bundle()` for context assembly |
| `llm_lifecycle.py` | Advisor uses same lifecycle (stop/start around GPU phases) |
| `llm_router.py` | Advisor calls via same router with `strategy_advisor.gbnf` |
| `watcher_policies.json` | Advisor reads bounds + cooldowns; never writes |
| `selfplay_orchestrator.py` | Receives `selfplay_overrides` as CLI args or config overlay |
| `High_Probability_Draw_Pools_Plan.pdf` | Advisor implements pool optimization strategy from this plan |
| Phase 9B.3 (`policy_proposal_heuristics`) | Advisor informs but does not replace heuristic triggers |

---

## 15. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-03 | Initial proposal — defines advisor role, mathematical framework, pool strategy, implementation plan |

---

## 16. Rationale Summary

| Design Choice | Rationale |
|---------------|-----------|
| Advisory only | Prevents runaway optimization without human-auditable checkpoints |
| Grammar-constrained output | Ensures parseable, structured recommendations |
| Minimum 15-draw gate | LLM analysis on sparse data produces noise |
| Phased rollout (A→D) | Builds trust incrementally before autonomous operation |
| Mathematical metrics (Section 6) | Quantitative signals prevent vague LLM recommendations |
| Pool-specific strategy | Different pool tiers have different failure modes and fixes |
| External LLM consultation | Local model handles routine; complex strategy benefits from stronger reasoning |
| Cooldown enforcement | Prevents hyperactive parameter churn from degrading stability |

---

## Appendix A: Quick Reference — When to Use What

```
Q: "Hit@20 is terrible but Hit@100 is fine"
A: Focus = POOL_PRECISION
   → Selfplay: optimize pool concentration, increase n_estimators
   → Models: favor catboost/xgboost (better at ranking)

Q: "Everything is missing, even Hit@300"  
A: Focus = REGIME_SHIFT
   → Selfplay: PAUSE
   → Action: Full pipeline rerun (Steps 1→6), window likely stale

Q: "Confidence scores don't mean anything"
A: Focus = CONFIDENCE_CALIBRATION  
   → Selfplay: increase k_folds, focus on fold stability metric
   → Models: increase diversity (all 3 types)

Q: "CatBoost always wins, other models are noise"
A: Focus = MODEL_DIVERSITY
   → Selfplay: force model rotation, don't early-exit on single type
   → Risk: monoculture overfits to current regime

Q: "Everything is working, metrics are stable"
A: Focus = STEADY_STATE
   → Selfplay: reduce episodes (5→3), maintenance exploration
   → Save compute for when it matters
```

---

**END OF CONTRACT**
