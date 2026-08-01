# STRATEGY_ADVISOR_AUDIT_v1.md

**Audit: the Strategy Advisor and the autonomy control surface — what it emits, what governance
validates, what the application layer applies, what execution receives, and where the chain breaks.**

| | |
|---|---|
| **Authority** | `docs/CLAUDE_CODE_INSTRUCTIONS_STRATEGY_ADVISOR_AUDIT.md` (REV1) |
| **Base** | `main` @ `179f7cd`, VM 101 (`192.168.3.177`), venv `~/venvs/torch`, user `michael` |
| **Date** | 2026-07-31 |
| **Type** | READ-ONLY AUDIT. No code, config, schema, grammar or documentation was modified. No commits. |
| **Status** | Parked reference document. Nothing here authorises implementation. |

> **§4 compliance.** This audit recommends the removal of **nothing**. Items that appear
> unreachable are reported as **declared-but-disconnected**, with the document, commit or comment
> establishing intent cited wherever one was found — and explicitly marked where none was found.
> Removal is a Team Beta ruling, never an audit conclusion.

---

## 0. Two corrections to the brief's premises

Both are stated up front because they change what the rest of the report means. Both are evidenced.

### 0.1 An LLM parameter-application seam **does exist**, is reachable, and applies values

The brief (§2) states *"the application seam does not exist — the only hit for `apply_*` /
`parameter_application` is a docstring."* That is true **of those search terms**, but the seam
exists under different naming and was missed by them.

`agents/watcher_agent.py:1789-1793` iterates LLM parameter proposals, validates each against a
policy whitelist, and **assigns the accepted value into the params dict**:

```python
for _prop in _llm_analysis.parameter_proposals:
    _pname = _prop.parameter
    _pval  = _prop.proposed_value
    if self._is_within_policy_bounds(_pname, _pval):
        retry_params[_pname] = _pval          # ← application
        logger.info('[WATCHER][LLM_DIAG] Applied: %s = %s (%s)', ...)
```

It is reachable, not dead code: `_build_retry_params` is called at
`agents/watcher_agent.py:2111` from the Step-5 training-health RETRY path, and its return value
becomes the next step's params at `agents/watcher_agent.py:2131` (`params = retry_params`).
The producer is the **diagnostics analyzer** (`diagnostics_llm_analyzer.request_llm_diagnostics_analysis`,
imported under a guard at `agents/watcher_agent.py:117-122`; both modules import cleanly in the
venv — verified this session).

**This is a different producer from the Strategy Advisor.** `parameter_advisor.py`'s
`StrategyRecommendation.parameter_proposals` are *not* what this seam consumes. The autonomy layer
has **two LLM proposal producers with different fates**, which §3 below separates.

The seam nevertheless **does not reach execution** — it breaks one hop later, at the step-scoped
manifest filter (§3.3, Break B). The practical consequence is worse than an absent seam: WATCHER
logs `Applied: learning_rate = 0.01` at INFO level for a value that never reaches the training
script.

### 0.2 The `chapter_13_acceptance.py:224` anchor has drifted

The brief cites the broken promotion seam at `chapter_13_acceptance.py:224`. At line 224 of the
file at `179f7cd` there is the `SelfplayCandidate` dataclass field block, not a promotion seam.
Line numbers have moved since that citation was written. The **substance is confirmed by a
different anchor**: `promote_candidate` (`chapter_13_acceptance.py:818`) has **zero callers**
outside its own module (verified by repo-wide search this session). Per the brief's "cite, do not
re-audit" instruction the finding is not re-derived here — only the anchor is corrected so future
readers do not verify against a stale line number.

---

## 1. Chain diagram — emit → validate → apply → execute

Two chains exist. They share governance vocabulary but not code.

### Chain A — Strategy Advisor → selfplay (the §3.4 chain)

```
  EMIT                     VALIDATE                 APPLY                   EXECUTE
┌──────────────┐  ✅   ┌─────────────────┐  ✅  ┌──────────────────┐  ✅ ┌──────────────────┐
│ StrategyAdv- │─────▶│ GBNF (primary   │─────▶│ watcher_dispatch │────▶│ selfplay_        │
│ isor.analyze │      │ path only)      │      │ .py:490 merge    │     │ orchestrator.py  │
│ param_advis- │      │ + Pydantic      │      │ → :150-158 →CLI  │     │ :1182-1187       │
│ or.py:688    │      │ + _clamp        │      │                  │     │ → config         │
└──────────────┘      └─────────────────┘      └──────────────────┘     └──────────────────┘
                                                        │
      2 of 6 SelfplayOverrides fields traverse ─────────┘
      4 of 6 (model_types, priority_metrics, exploration_ratio,
              search_strategy) are read into the dict and never consumed  ⟵ BREAK A
```

**Chain A is PRESENT end-to-end for `max_episodes` and `min_fitness_threshold`.** This is the
audit's principal positive finding and it is what makes §3.4's verdict "yes, partially".

### Chain B — Strategy Advisor → `strategy_recommendation.json` → WATCHER

```
┌──────────────┐      ┌─────────────────────────┐      ┌──────────────┐
│ _save_recom- │ ✅  │ strategy_recommendation │  ❌  │ (no reader)  │
│ mendation    │────▶│ .json  + strategy_hist- │─────▶│              │
│ :1177-1192   │      │ ory/  (live on host)    │      │  ABSENT      │
└──────────────┘      └─────────────────────────┘      └──────────────┘
```

`parameter_advisor.py:629` states the file is written *"for WATCHER consumption."* **No code reads
it.** Exhaustive `git grep` over all tracked files returns the producer, documentation, and the
grammar header comment only — no consumer. Chain A works precisely because
`watcher_dispatch.py:481` calls `advisor.analyze()` **in-memory** and never opens the file. The
file is a write-only audit artifact.

### Chain C — diagnostics analyzer → Step-5 retry params

```
┌──────────────┐  ✅  ┌──────────────────────┐  ✅ ┌──────────────────┐  ❌ ┌─────────────┐
│ diagnostics_ │─────▶│ _is_within_policy_   │────▶│ retry_params[p]  │────▶│ Step-5      │
│ llm_analyzer │      │ bounds :1665-1704    │     │ = v  :1789-1793  │     │ script      │
└──────────────┘      └──────────────────────┘     └──────────────────┘     └─────────────┘
                                                                    ⟵ BREAK B
                            filtered out at watcher_agent.py:1385-1393 (step-scoped
                            allowed_params); all 6 whitelisted params absent from
                            agent_manifests/reinforcement.json default_params
```

### Chain D — Chapter-13 LLMProposal → acceptance engine

```
┌──────────────┐ ✅ ┌────────────────────┐ ✅ ┌──────────────────────────┐ ❌ ┌──────────┐
│ chapter_13_  │───▶│ acceptance_engine  │───▶│ result["outcome"] =      │───▶│ (nothing)│
│ llm_advisor  │    │ .validate_proposal │    │ "pending_approval"       │    │  ABSENT  │
│              │    │ ch13_orch.py:395   │    │ ch13_orch.py:398-441     │    │          │
└──────────────┘    └────────────────────┘    └──────────────────────────┘    └──────────┘
```

Every branch at `chapter_13_orchestrator.py:398-441` terminates in an outcome string or
`request_approval()`. **No branch applies `proposal.parameter_proposals` to any config, manifest or
command.** `record_applied_changes` (`chapter_13_acceptance.py:635`) — the function that would log
such an application — has **zero callers**.

### Hop status summary

| Chain | Emit | Validate | Apply | Execute |
|---|---|---|---|---|
| **A** Advisor → selfplay (`max_episodes`, `min_fitness_threshold`) | PRESENT | PRESENT | PRESENT | **PRESENT** |
| **A′** Advisor → selfplay (other 4 override fields) | PRESENT | PARTIAL | **BROKEN** | ABSENT |
| **B** Advisor → `strategy_recommendation.json` → WATCHER | PRESENT | PRESENT | **ABSENT** | ABSENT |
| **C** diagnostics → Step-5 retry | PRESENT | PRESENT | PRESENT | **BROKEN** |
| **D** Ch13 proposal → acceptance | PRESENT | PRESENT | **ABSENT** | ABSENT |
| **E** Advisor → sieve thresholds | **ABSENT** (no field) | n/a | n/a | n/a |
| **F** Advisor → Step-1 search space | PRESENT (`search_strategy`) | PARTIAL | **BROKEN** | ABSENT |

---

## 2. Emit-surface inventory (§3.1)

### 2.1 `StrategyRecommendation` — `parameter_advisor.py:155-183`

| Field | Type | Bounds | Req. | In loaded grammar? |
|---|---|---|---|---|
| `schema_version` | `str` | — | default `"1.1.0"` | ⚠️ grammar **pins `"1.0.0"`** |
| `generated_at` | `str` | — | default (UTC now) | ❌ absent |
| `advisor_model` | `str` | — | default; overwritten `:811` | ❌ absent |
| `draws_analyzed` | `int` | — | default `0` | ❌ absent |
| `focus_area` | `FocusArea` | 7-value enum | **required** | ✅ 7/7 agree |
| `focus_confidence` | `float` | `ge=0, le=1` | **required** | ✅ |
| `focus_rationale` | `str` | — | **required** | ✅ |
| `secondary_focus` | `FocusArea?` | enum \| null | optional | ✅ |
| `secondary_confidence` | `float?` | — | optional | ✅ |
| `recommended_action` | `AdvisorAction` | 5-value enum | **required** | ✅ 5/5 agree |
| `retrain_scope` | `RetrainScope?` | 4-value enum | optional | ✅ in loaded copy (⚠️ §2.3) |
| `selfplay_overrides` | `SelfplayOverrides` | see 2.2 | default factory | ✅ |
| `parameter_proposals` | `List[ParameterProposal]` | `max_length=5` | default `[]` | ⚠️ grammar imposes **no max** |
| `pool_strategy` | `PoolStrategy` | 3 free strings | default factory | ✅ |
| `risk_level` | `RiskLevel` | 3-value enum | default `low` | ✅ |
| `requires_human_review` | `bool` | — | default `False` | ✅ |
| `diagnostic_summary` | `DiagnosticSummary` | 8 fields | default factory | ✅ |
| `alternative_hypothesis` | `str?` | — | optional | ✅ |
| `metadata` | `Dict?` | — | optional (v1.1.0) | ❌ absent |

**Grammar/Pydantic disagreements (findings):**

- **`schema_version` pinned to a stale value.** Both grammar copies force the literal `"1.0.0"`;
  the Pydantic default is `"1.1.0"` (`parameter_advisor.py:157`). Because the field is an
  unconstrained `str`, validation passes and **every LLM-produced recommendation is stamped
  `1.0.0`** regardless of the model version. Confirmed against the live artifact:
  `strategy_recommendation.json:2` reads `"schema_version": "1.0.0"` while carrying the v1.1.0-only
  `metadata` field. A consumer version-gating on this field would mis-route.
- **`parameter_proposals` cardinality is unenforced at the grammar layer.** Pydantic caps at 5
  (`:173`); the grammar permits unbounded repetition. A 6-proposal emission is a hard
  `ValidationError`, not a graceful truncation.
- **Four Pydantic fields are absent from the grammar** (`generated_at`, `advisor_model`,
  `draws_analyzed`, `metadata`). All carry defaults, so this is benign — the host fills them
  post-parse. Recorded for completeness, not as a defect.

### 2.2 `SelfplayOverrides` — `parameter_advisor.py:126-133`

This is the operative surface for §3.4.

| Field | Type | Pydantic bounds | Clamped? | **Consumed by execution?** |
|---|---|---|---|---|
| `max_episodes` | `int` | `ge=1, le=50` | ✅ `_SELFPLAY_BOUNDS` | ✅ **YES** → `--episodes` |
| `min_fitness_threshold` | `float` | `ge=0.0, le=1.0` | ✅ | ✅ **YES** → `--min-fitness` ⚠️ |
| `model_types` | `List[str]` | none | ❌ | ❌ no |
| `priority_metrics` | `List[str]` | none | ❌ | ❌ no |
| `exploration_ratio` | `float` | `ge=0.0, le=1.0` | ✅ | ❌ no |
| `search_strategy` | `Optional[str]` | **none — free string** | ❌ | ❌ no |

**Findings:**

- **`search_strategy` is emittable and is not type-constrained in Pydantic.** `parameter_advisor.py:133`
  declares `Optional[str] = None` with the four legal values recorded only in a trailing comment
  (`# bayesian, random, grid, evolutionary`). The **grammar** does constrain it to those four
  (`grammars/strategy_advisor.gbnf`, rule `search-strategy`), so on the grammar-constrained path the
  value is safe — but on the two non-grammar paths (§7) any string validates. This directly answers
  the brief's specific question: **yes, the advisor can emit a sampler choice, and it has done so in
  production** — `strategy_recommendation.json:25` on this host contains `"search_strategy": "grid"`,
  emitted 2026-02-08T05:46:01Z by `deepseek_primary`.
- **⚠️ Truthiness bug on `min_fitness_threshold`.** `agents/watcher_dispatch.py:151` gates with
  `if overrides.get("min_fitness_threshold"):`. Pydantic permits `0.0` (`ge=0.0`), and `0.0` is
  falsy — so a legitimate advisory of `0.0` is **silently dropped** and selfplay runs at its own
  default. This is the exact anti-pattern `tfm-project-facts` §2.7 #2 records as already having
  caused one threshold regression (*"`is None` not truthiness — 0.0 is legitimate"*). `max_episodes`
  is not exposed (`ge=1`).
- **Defaults are indistinguishable from advice.** `selfplay_overrides` uses `default_factory`
  (`:172`), so the attribute is always a populated model, and a Pydantic `BaseModel` instance is
  always truthy. The guard `if rec and rec.selfplay_overrides:` (`watcher_dispatch.py:482`)
  therefore always passes, and `max_episodes=10` / `min_fitness_threshold=0.5` are merged into the
  dispatch request **even in `heuristic_degraded` mode when no LLM was reachable**. The dispatcher
  cannot distinguish "the advisor recommends 10 episodes" from "the advisor said nothing and the
  schema default is 10." The merge at `:490` uses `.update()`, so it overrides the caller's own
  `episodes` value at `:153-158`.

### 2.3 `RetrainScope` — `parameter_advisor.py:89-94`

Pydantic: `selfplay_only`, `steps_5_6`, `steps_3_5_6`, `full_pipeline` (4 values).

| Grammar copy | Values | Loaded? |
|---|---|---|
| `grammars/strategy_advisor.gbnf` | those 4 | ✅ **loaded** |
| `strategy_advisor.gbnf` (repo root) | those 4 **+ `steps_0_1` + `step_1_only`** | ❌ never loaded |

The root copy's two extra scopes are the S140b Step-1 autonomy feature (see §6.1). They are
**absent from the Pydantic enum**, so had the root copy been loaded, emitting either would raise
`ValidationError` at `parameter_advisor.py:1131`. The feature is disconnected at two independent
layers.

### 2.4 `ParameterProposal` — `parameter_advisor.py:116-123`

| Field | Type | Bounds |
|---|---|---|
| `parameter` | `str` | none — **any name emittable** |
| `current_value` | `Optional[float]` | none |
| `proposed_value` | **`float`** | none |
| `delta` | `str` | root grammar constrains format; loaded copy does not |
| `confidence` | `float` | `ge=0, le=1` |
| `rationale` | `str` | none |

**Finding — a type/vocabulary impedance mismatch.** `proposed_value` is `float`. The only entry in
`watcher_policies.json`'s `parameter_bounds` is `search_strategy`, whose legal values are the
**strings** `bayesian|random|grid|evolutionary` (`watcher_policies.json:104-115`). A
`ParameterProposal` therefore **cannot represent the one parameter the policy file declares
governable.** The two halves of the governance contract do not share a type.

Second-order: the root grammar constrains `delta` to `("+"|"-"|"*")?[0-9]+(.[0-9]+)?`; the loaded
copy relaxes it to a free `string` (both copies' `parameter-proposal` rule). The tighter constraint
is in the file that is not read.

### 2.5 Sibling emit schemas

| Schema | Location | Parameter surface | Fate |
|---|---|---|---|
| `DiagnosticsAnalysis` | `diagnostics_analysis_schema.py:86-112` | `parameter_proposals: List[DiagnosticsParameterProposal]` (`extra="forbid"`) | Chain C — applied then filtered out |
| `DiagnosticsParameterProposal` | `:72-83` | `parameter`, `current_value`, `proposed_value: float`, `rationale` | as above |
| `LLMProposal` (dataclass) | `llm_proposal_schema.py:105-135` | `parameter_proposals: List[ParameterProposal]` | Chain D — validated, never applied |
| `ParameterProposal` (dataclass) | `llm_proposal_schema.py:71-90` | same 6 fields as 2.4 | as above |

**No advisor schema in the repository contains a field for a sieve threshold.** Confirmed by field
enumeration of all four schemas above. The nearest is `SelfplayOverrides.min_fitness_threshold`,
which is an ML-candidate fitness floor consumed by `selfplay_orchestrator.py:533`
(`result.fitness >= self.config.min_fitness_threshold`) — **not** a forward/reverse sieve match
threshold. The two are unrelated quantities that share a word.

### 2.6 Grammar inventory and directory divergence

Two grammar directories exist with **two independent loaders**:

| Loader | Resolves to | Anchor |
|---|---|---|
| `llm_router.evaluate_with_grammar` | `Path("grammars") / <file>` — **CWD-relative** | `llm_services/llm_router.py:475` |
| `grammar_loader.get_grammar_path` | `<repo>/agent_grammars/` — path-absolute | `llm_services/grammar_loader.py:45-47` |
| `chapter_13_llm_advisor` | `Path("grammars")` — **CWD-relative** | `chapter_13_llm_advisor.py:311-312` |

| Grammar | `grammars/` | `agent_grammars/` | Status |
|---|---|---|---|
| `agent_decision.gbnf` | ✅ | ✅ | **identical** |
| `parameter_adjustment.gbnf` | ✅ | ✅ | **identical** |
| `sieve_analysis.gbnf` | ✅ | ✅ | **identical** |
| `json_generic.gbnf` | ✅ | ✅ | **diverged** (44 diff lines) |
| `chapter_13.gbnf` | ✅ | ✅ | **diverged** — see §6.1 |
| `strategy_advisor.gbnf` | ✅ (loaded) | — (repo root copy) | **diverged**, 170 diff lines — see §6.1 |
| `watcher_decision.gbnf`, `diagnostics_analysis.gbnf` | ✅ | — | single copy |

**Finding — CWD-relative grammar resolution (VIR-6 exposure).** Both `llm_router.py:475` and
`chapter_13_llm_advisor.py:312` resolve grammars relative to the *current working directory*, not
the module path — unlike `grammar_loader.py`, which was explicitly fixed for this in v1.1.0
(`grammar_loader.py:10-11`: *"Fixed GRAMMAR_DIR to use os.path resolution instead of hardcoded
relative path… regardless of CWD"*). The advisor is invoked from `watcher_dispatch.py:476` with
`state_dir` set to the repo root, but the **process CWD is not set by that call**. If WATCHER is
launched from any other directory, `evaluate_with_grammar` raises `FileNotFoundError`
(`llm_router.py:477`) and `chapter_13_llm_advisor` **silently degrades to unconstrained decoding**
(`:313`, `grammar_available` → `False` → `router.route()`). The lesson recorded in
`grammar_loader.py` v1.1.0 was not propagated to the other two loaders.

---

## 3. Declared-vs-enforced — `watcher_policies.json` (§3.2)

Every governance-bearing key, traced to enforcing code or to its absence.

| Key (anchor) | Declares | Enforcing code | Verdict |
|---|---|---|---|
| `v1_approval_required.parameter_application: true` (`:63`) | LLM proposals need approval before application | none — the sole repo reference is a **comment**, `miner/range_miner_coordinator.py:3432` | **DECLARED, NOT IMPLEMENTED** — brief's claim **CONFIRMED** |
| `acceptance_rules.max_parameter_delta: 0.3` (`:41`) | reject >30% changes | `chapter_13_acceptance.py:351`; `llm_proposal_schema.py:212` | **ENFORCED** (Chain D only) |
| `acceptance_rules.max_parameters_per_proposal: 3` (`:43`) | reject >3 params at once | `chapter_13_acceptance.py:352`; `llm_proposal_schema.py:229` | **ENFORCED** (Chain D only) |
| `acceptance_rules.cooldown_runs: 3` (`:45`) | min runs between changes | `chapter_13_acceptance.py:353`; `chapter_13_triggers.py:380` | **ENFORCED** (Chain D only) |
| `parameter_bounds.search_strategy` (`:104-115`) | `type: choice`, 4 choices, default `bayesian` | `parameter_advisor.py:1150` reads the dict but tests **only `min`/`max`** (`:1162-1170`) | **DECLARED, NOT ENFORCED** — see below |
| `strategy_change_cooldown_episodes: 5` (`:116`) | cooldown on strategy switching | **zero references** in any `.py` | **DECLARED, NOT IMPLEMENTED** |
| `escalation.escalate_on_consecutive_rejects: 3` (`:50`) | alert human after 3 rejects | **zero references** in any `.py` | **DECLARED, NOT IMPLEMENTED** |
| `logging.parameter_change_log: "parameter_change_history.json"` (`:91`) | audit log path | key never read; `chapter_13_acceptance.py:56` hardcodes `".parameter_change_history.json"` (**leading dot**) | **DECLARED, NOT WIRED** + filename mismatch |
| `logging.llm_proposals_archive: "llm_proposals/"` (`:90`) | proposal archive dir | **zero references** in any `.py` | **DECLARED, NOT IMPLEMENTED** |
| `logging.decision_log_path`, `diagnostics_archive`, `retrain_log` (`:88-92`) | paths | read by Ch13 modules | **ENFORCED** |
| `selfplay.min_fitness: 0.5` (`:95`) | selfplay fitness floor | **not** the advisor path; advisor supplies its own via CLI (§2.2) | **PARALLEL, UNRECONCILED** |
| `selfplay.auto_promote: false`, `require_human_approval: true` (`:99-100`) | promotion gates | `promote_candidate` (`chapter_13_acceptance.py:818`) has **zero callers** | **MOOT** — gated function unreachable |
| `retrain_triggers.*` (`:17-28`) | retrain conditions | `chapter_13_triggers.py` | **ENFORCED** |
| `regime_shift_triggers.*` (`:30-38`) | full-rerun conditions | `chapter_13_triggers.py` | **ENFORCED** |
| `training_diagnostics.metric_bounds.*` (`:138-164`) | health thresholds | `training_health_check` | **ENFORCED** |
| `training_diagnostics.llm_adjustable_params` | — | **key does not exist in the file**; `agents/watcher_agent.py:1683` reads it with a 6-param hardcoded fallback | **INVERTED** — see below |
| `notifications.*` (`:171-193`) | Telegram policy | `notify_telegram` | **ENFORCED** (self-described advisory) |
| `approval_route: "orchestrator"` (`:194`) | approval routing | `chapter_13_orchestrator.py:406` | **ENFORCED** |

### 3.1 `parameter_bounds.search_strategy` — declared as a choice, enforced as nothing

`_validate_bounds` (`parameter_advisor.py:1141-1175`) is the only consumer of `parameter_bounds`
on the advisor path. It implements a whitelist test plus **numeric `min`/`max` comparisons only**:

```python
if "min" in param_bounds and value < param_bounds["min"]:  ... reject
if "max" in param_bounds and value > param_bounds["max"]:  ... reject
```

`search_strategy`'s bounds object is `{"type": "choice", "choices": [...], "default": "bayesian"}` —
it has neither `min` nor `max`. Therefore **any `proposed_value` whatsoever passes governance** for
the one parameter the policy file governs. Combined with §2.4's type mismatch (`proposed_value` is
`float`, the choices are strings), the enforced behaviour is: a proposal named `search_strategy`
carrying an arbitrary float is *accepted* by the governance layer and then reaches nothing.
`type: "choice"` is declared vocabulary that no code on this path interprets.

### 3.2 The whitelist inversion

`agents/watcher_agent.py:1683` reads `training_diagnostics.llm_adjustable_params` from the policy
file, falling back to a 6-parameter hardcoded default. **That key is absent from
`watcher_policies.json`** (full file read this session). The hardcoded fallback is therefore
*always* the operative whitelist:

`normalize_features`, `nn_activation`, `learning_rate`, `dropout`, `n_estimators`, `max_depth`.

The governance boundary for the only working application seam lives in Python source, not in the
version-controlled policy file that `watcher_policies.json:2` describes as
*"System governance thresholds. Version-controlled by design."* Editing the policy file cannot
widen or narrow it; only editing `watcher_agent.py` can.

---

## 4. Selfplay control verdict (§3.4)

> **Does the Strategy Advisor control selfplay in any operative sense?**

### **Verdict: YES — partially, and narrowly. The premise holds.**

Two of six `SelfplayOverrides` fields traverse the full chain from LLM emission to process
execution. This is a real control surface, not a dormant one, and it is verified at every hop:

| Hop | Evidence | Status |
|---|---|---|
| Advisor invoked on the live dispatch path | `agents/watcher_dispatch.py:473-481` — `request_type == "selfplay_retrain"` → `advisor.analyze()` | ✅ |
| Overrides merged into the dispatch request | `:490` — `request.setdefault("selfplay_overrides", {}).update(overrides)` | ✅ |
| Request routed to the dispatcher | `:508-517` — `dispatch_selfplay(self, request)` | ✅ |
| Overrides translated to CLI args | `:150-158` — `--min-fitness`, and `--episodes` rewritten in place | ✅ |
| CLI args accepted by execution | `selfplay_orchestrator.py:1112-1116`, `:1135-1139` | ✅ |
| CLI args reach live config | `:1182-1187` — `config.max_episodes`, `config.min_fitness_threshold` | ✅ |
| Config governs behaviour | `:504` episode loop bound; `:533` candidate-emission gate | ✅ |

The advisor can therefore lengthen or shorten a selfplay run (1–50 episodes) and raise or lower the
bar for emitting a policy candidate. Both are consequential: `:533` decides whether an episode
produces a candidate at all.

### What it does **not** control

`model_types`, `priority_metrics`, `exploration_ratio` and `search_strategy` are emitted, clamped
(the numeric one), merged into the request dict at `:490`, and then **never read** — `dispatch_selfplay`
consumes exactly two keys (`:151`, `:153`). They are dead dimensions in the §0.5 sense: an
autonomous agent varying `exploration_ratio` would observe outcome changes uncorrelated with its
action and "learn" into a void.

### Bearing on multi-sampler selection

Michael's reasoning was: *if* the advisor controls selfplay, multi-sampler selection becomes a real
requirement rather than a dormant one. **The antecedent is established** — but with a qualification
that matters for how Beta should read it:

The advisor's operative control over selfplay runs through `max_episodes` and
`min_fitness_threshold` — **neither of which is a sampler choice**. `search_strategy` is the field
that would make multi-sampler selection operative, and it is in the disconnected group. So the
premise holds in general (the advisor is not decorative; it steers real runs) while the *specific*
lever multi-sampler selection depends on is one of the four that terminate at
`watcher_dispatch.py:490`. Whether "the advisor controls selfplay" is sufficient warrant for
promoting multi-sampler selection to a live requirement is a scoping judgment reserved for Beta
(§8, Q3) — this audit establishes only that the advisor emits a strategy choice, has done so in
production (`"grid"`), and that nothing receives it.

### Are `propose_transform_update` and the promotion seam the only breaks?

**No.** Within the advisor→selfplay surface this audit identifies breaks independent of both
(neither is re-audited here; both are cited from `tfm-project-facts` §2.5):

1. Four of six override fields unconsumed at `watcher_dispatch.py:150-158` (Break A).
2. `min_fitness_threshold == 0.0` silently dropped by the truthiness guard at `:151`.
3. Schema defaults indistinguishable from advice, including in degraded mode (§2.2).
4. `strategy_recommendation.json` written for a reader that does not exist (Chain B).
5. `validate_selfplay_candidate` (`chapter_13_acceptance.py:716`) — zero callers, so the selfplay
   candidate produced at `selfplay_orchestrator.py:533` is not validated by the acceptance engine
   on any traced path.

---

## 5. The February four-hop chain (§3.5)

`docs/PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md` describes: *advisor recommends "random" →
WATCHER validates → dispatch passes `--strategy random` → window_optimizer uses RandomSampler.*

| Hop | Claim | Status | Evidence |
|---|---|---|---|
| **1** | Advisor recommends a strategy | ✅ **EXISTS** | `parameter_advisor.py:133` field; grammar rule `search-strategy`; **live proof** — `strategy_recommendation.json:25` = `"grid"` |
| **2** | WATCHER validates it | ⚠️ **EXISTS BUT CANNOT SEE IT** | `_is_within_policy_bounds` (`agents/watcher_agent.py:1665-1704`) rejects unknown names by design (`:1692-1693`); `search_strategy` is **not** in its whitelist (`:1683-1690`) → rejected. Independently, `_validate_bounds` (`parameter_advisor.py:1162-1170`) would accept it *unconditionally* for the opposite reason (§3.1). Two validators, neither correct. |
| **3** | Dispatch passes `--strategy` | ⚠️ **EXISTS, NOT ADVISOR-FED** | The flag *is* emitted — `agent_manifests/window_optimizer.json:242` sets `default_params.strategy = "bayesian"`, and `agents/watcher_agent.py:1496-1512` renders it as `--strategy bayesian`. But it is a **static manifest constant**, never an advisor value (§5.1). |
| **4** | window_optimizer uses the sampler | ❌ **BROKEN** — cited, not re-audited | `docs/STRATEGY_ORIGIN_AUDIT.md`. Current state fails closed: `require_supported_strategy` (`window_optimizer.py:532-556`) raises `WINDOW_OPTIMIZER_STRATEGY_UNSUPPORTED` for gutted strategies, per S178 P0 (`ddd2ac8`). |

The proposal predicted the failure at hop 2 (`PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md:227`:
*"`_is_within_policy_bounds()` must recognize it or the proposal is silently dropped"*). **That
prediction is confirmed** — with the addition that hop 3 was never connected to hop 2 either.

### 5.1 A key-name disconnect severs hop 2 → hop 3

The Step-1 manifest uses **two different names for the same concept**:

| Location | Key | Anchor |
|---|---|---|
| `args_map` (CLI ← context) | `"strategy": "search_strategy"` | `agent_manifests/window_optimizer.json:29` |
| `parameter_bounds` | `"search_strategy"` | `:142` |
| `default_params` | **`"strategy"`** | `:242` |

`run_step` filters incoming params against `allowed_params = set(default_params.keys())`
(`agents/watcher_agent.py:1385-1393`). Because `default_params` is keyed `strategy` and every
advisor/policy surface is keyed `search_strategy`, an injected `search_strategy` would be
**discarded** at `:1393` with a DEBUG-level message (`"Skipping param 'search_strategy' — not
declared in step 1 manifest"`). The name that survives the filter (`strategy`) is exactly the one no
advisor surface emits.

The `--strategy` flag that does reach `window_optimizer.py` is produced by the **fallback** branch
at `:1499` (`key.replace("_","-")`), not by the `args_map` reverse lookup — `"strategy"` happens to
be identical as a CLI name. Hop 3 works by coincidence of naming, not by the mapping intended to
carry it.

**Latent collision (not currently reachable):** if both `strategy` and `search_strategy` were ever
present in `final_params`, the loop at `:1496-1512` would emit `--strategy` **twice** (once via
fallback, once via `_param_to_cli`), and argparse would silently take the last. Any future wiring
must reconcile the key names rather than add the second one.

---

## 6. Declared-but-disconnected inventory (§3.6)

Per §4, each item cites the artifact establishing intent, or explicitly records that none was found.

### 6.1 The two S140b Step-1 autonomy features — patched into non-loaded grammar copies

**This is the §2.7 stale-copy regression class, reproduced in the autonomy layer.**

`apply_s140b_trial_history.py` patched two grammars to let an agent request a Step-1 relaunch. Both
patches landed in files **the loaders do not read**:

| Feature | Patched into | Loader actually reads | Present in loaded copy? |
|---|---|---|---|
| `steps_0_1`, `step_1_only` (retrain scope) | `strategy_advisor.gbnf` (repo root) — patch 9, `apply_s140b_trial_history.py:623-647` | `grammars/strategy_advisor.gbnf` via `llm_router.py:475` | ❌ **no** |
| `step1_relaunch` (Ch13 scope) | `agent_grammars/chapter_13.gbnf` — patch 8, `:601-604` | `grammars/chapter_13.gbnf` via `chapter_13_llm_advisor.py:311` | ❌ **no** (verified by diff: line 20 differs by exactly this token) |

**Intent is documented**: `docs/S142_CHAT_PROMPT.md:38` — *"`strategy_advisor.gbnf` — `steps_0_1`,
`step_1_only` scopes"*; and `apply_s140b_trial_history.py:625` — *"add steps_0_1 and step_1_only"*.
Per §0.4 these are deliberately built features, not vestigial artifacts.

Each is disconnected at **three** independent layers:
1. the patched grammar copy is not the loaded one;
2. `parameter_advisor.RetrainScope` (`:89-94`) lacks `steps_0_1`/`step_1_only`, and
   `llm_proposal_schema.RetrainScope` (`:51-56`) lacks `step1_relaunch`;
3. `step1_relaunch` has **zero handlers** in any `.py` file.

So even repairing the copy divergence would not make either feature operative.

### 6.2 Full inventory

| # | Declared item | Declared where | Terminates at | Intent citation |
|---|---|---|---|---|
| 1 | `search_strategy` (advisor emit) | `parameter_advisor.py:133`; grammar `search-strategy` | `watcher_dispatch.py:490` — merged, never read | `PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md`; `docs/STRATEGY_ORIGIN_AUDIT.md` |
| 2 | Sieve thresholds (advisor emit) | — **no field exists** | n/a | `miner/range_miner_coordinator.py:3428-3442` — declares the chokepoint any future application path **must** use; `TODO_SELFPLAY_AND_LLM_AUTONOMY.md` Part B |
| 3 | `model_types` | `parameter_advisor.py:129` | `watcher_dispatch.py:490` | **none found** — no doc/commit/comment located explaining its intended consumer |
| 4 | `priority_metrics` | `:131` | `watcher_dispatch.py:490` | **none found** |
| 5 | `exploration_ratio` | `:132` | `watcher_dispatch.py:490` | clamped by `_SELFPLAY_BOUNDS` (`:581`), implying an intended consumer; **no consumer doc found** |
| 6 | `pool_strategy` (3 guidance strings) | `:136-140` | written to JSON only | `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md` |
| 7 | `parameter_application: true` | `watcher_policies.json:63` | comment only | `range_miner_coordinator.py:3431-3434` — *"DECLARED but NOT BUILT"* |
| 8 | `strategy_change_cooldown_episodes` | `:116` | nothing | **none found** |
| 9 | `escalate_on_consecutive_rejects` | `:50` | nothing | **none found** |
| 10 | `llm_proposals_archive` | `:90` | nothing | **none found** |
| 11 | `parameter_change_log` | `:91` | nothing (name mismatch vs `chapter_13_acceptance.py:56`) | **none found** |
| 12 | `parameter_bounds.type: "choice"` | `:106` | `_validate_bounds` ignores non-min/max | **none found** |
| 13 | `steps_0_1` / `step_1_only` | root `strategy_advisor.gbnf:60-61` | not loaded | `docs/S142_CHAT_PROMPT.md:38` |
| 14 | `step1_relaunch` | `agent_grammars/chapter_13.gbnf:20` | not loaded, no handler | `apply_s140b_trial_history.py:601-604` |
| 15 | `strategy_recommendation.json` consumer | `parameter_advisor.py:629` docstring | no reader | `CONTRACT_LLM_STRATEGY_ADVISOR_v1_0.md:444`; `SESSION_CHANGELOG_20260207_S66.md:203` names the unbuilt function `_apply_strategy_recommendation()` |
| 16 | Six LLM-adjustable Step-5 params | `agents/watcher_agent.py:1683-1690` | dropped at `:1393` (absent from `reinforcement.json` `default_params`, `:173`) | S81 Phase 7 markers, `:1737-1815` |
| 17 | `record_applied_changes` | `chapter_13_acceptance.py:635` | zero callers | governance audit-trail intent, `watcher_policies.json:91` |
| 18 | `validate_selfplay_candidate` | `:716` | zero callers | `watcher_policies.json:99-100` |
| 19 | `promote_candidate` | `:818` | zero callers | `tfm-project-facts` §2.5 (cited, not re-audited) |
| 20 | `_update_strategy_advisor()` | `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md:1719` — attributed to `agents/watcher_agent.py` | **method does not exist** in either WATCHER copy | doc describes it as built |

**Item 20 note:** `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md:1719` tabulates
`_update_strategy_advisor()` as an existing ~30-line method of `agents/watcher_agent.py` that
writes `strategy_recommendation.json`. No such method exists at `179f7cd`. This is documentation
describing an unbuilt component as built — the same class of hazard §0 warns about, in the
opposite direction.

### 6.3 Stale duplicate — `watcher_agent.py`

`watcher_agent.py` (repo root, v1.1.0, 72 KB, last touched by `f83dd8e` *"cleanup: remove .py from
docs/"*) and `agents/watcher_agent.py` (v2.0.0, 128 KB, actively maintained through S167/S168a) are
**both tracked** and differ by 1,386 diff lines. The live file is `agents/watcher_agent.py`; all
§3/§5 anchors in this report are taken from it. Reported for anchor hygiene — future audits reading
the root copy will find a `run_step` at a different line number with different behaviour. Per
`tfm-project-facts` §0.4 and this brief's §4, **no removal is proposed**; a prior Beta ruling
(2026-07-31) already covers leaving known stale duplicates alone.

---

## 7. What the constraints actually enforce (§3.7)

Evidence for Beta's ruling on whether the original restriction still fits. Stated as capability,
not as recommendation.

### 7.1 Three emission paths with three different guarantees

`parameter_advisor.analyze()` (`:742-808`) has a three-tier hierarchy. **The constraints differ per
tier**, which is the central fact for Beta:

| Path | Trigger | Token-level (GBNF) | Schema validation | Range clamp |
|---|---|---|---|---|
| **1. DeepSeek primary** | LLM reachable | ✅ **yes** — `evaluate_with_grammar` → `_call_primary_with_grammar` (`llm_router.py:482-487`) | ✅ `StrategyRecommendation(**rec_data)` (`:1131`) | ✅ `_clamp_llm_recommendation` (`:1128`) |
| **2. Claude backup** | primary down, or low-confidence risky action (`:769-776`) | ❌ **NO** — `_call_backup(prompt)` at `llm_router.py:492` takes **no grammar argument** | ✅ same | ✅ same |
| **3. Heuristic** | both LLMs unreachable (`:798-808`) | n/a — no LLM | n/a — constructed in Python | n/a |

**The grammar constrains only path 1.** On path 2 the model is free-running JSON; the *only* thing
standing between Claude's output and a `StrategyRecommendation` is Pydantic plus the clamp. Path 2
is not exotic — it is entered deliberately whenever DeepSeek returns low confidence on a risky
action (`:769`).

### 7.2 What GBNF actually guarantees on path 1

Token-level constraint during decoding — the model **cannot emit** a non-conforming token:

- **Enum restriction — real and complete** for `focus_area` (7), `recommended_action` (5),
  `retrain_scope` (4), `risk_level` (3), `fitness_trend` (4), `model_type` (4), and **`search_strategy`
  (exactly `bayesian|random|grid|evolutionary`)**.
- **Structural guarantee — real.** Object key sets and ordering are fixed by the `recommendation`
  rule; a hallucinated extra key is unrepresentable.
- **Numeric range — partial and syntactic.** `confidence-value ::= "0"("."[0-9]+)? | "1"(".0")? |
  "0."[0-9]+` genuinely confines confidences to [0,1] by construction. But `integer ::= [0-9]+` is
  **unbounded** — `max_episodes: 99999` is grammatically legal and is caught only downstream by the
  clamp (`_SELFPLAY_BOUNDS`, `:578-582`) and Pydantic (`le=50`).
- **Cardinality — not guaranteed.** `parameter-proposals` permits unbounded repetition against
  Pydantic's `max_length=5` (§2.1).
- **Semantic validity — not guaranteed and not guaranteeable.** The grammar forces `"grid"` to be
  well-formed; it cannot know `GridSearch` is gutted. That is exactly what
  `require_supported_strategy` (`window_optimizer.py:532-556`) now catches at the execution
  boundary — a fail-closed check added by S178 P0 (`ddd2ac8`), independent of the LLM layer.

### 7.3 What Pydantic adds on all LLM paths

- **Type coercion + rejection**: `StrategyRecommendation(**rec_data)` raises `ValidationError` on
  malformed input; the caller treats this as LLM failure and escalates (`:778-779`).
- **Range enforcement**: `ge`/`le` on `focus_confidence`, `max_episodes` (1–50),
  `min_fitness_threshold`, `exploration_ratio`, and the `DiagnosticSummary` fields.
- **Enum re-validation**: independent of the grammar, so path 2 output is still enum-checked.
- **Gaps**: `search_strategy` is an unconstrained `Optional[str]` (§2.2);
  `ParameterProposal.parameter` is an unconstrained `str`; `pool_strategy`'s three fields are free
  text. `DiagnosticsAnalysis` is stricter — it sets `extra="forbid"`
  (`diagnostics_analysis_schema.py:77`), which `StrategyRecommendation` does not.

### 7.4 What the governance layer adds — and does not

- `_clamp_llm_recommendation` (`:585-621`) clamps **three** numeric selfplay fields and tags the
  adjustment for audit. It does not touch `search_strategy`, `model_types` or `priority_metrics`.
- `_validate_bounds` (`:1141-1175`) enforces a **whitelist** on `parameter_proposals` — unknown
  parameter names are rejected (`:1155-1157`), which is the correct default-deny posture. But it
  applies **only to `parameter_proposals`**; `selfplay_overrides` — the fields that actually reach
  execution — bypass it entirely. The one surface with real downstream effect is the one the policy
  layer does not inspect.
- For the single parameter the policy file governs, the check is vacuous (§3.1).

### 7.5 Net position for Beta

For a **grammar-constrained DeepSeek emission**, advisor output is constrained at three independent
layers (token, schema, clamp), with enum restriction that is genuine and total for every enumerated
field including the sampler choice. That is a materially different risk profile from an
unconstrained LLM, and it is the factual basis Michael's position rests on.

Three qualifications belong in the same breath, and Beta should weigh them together:

1. **The strongest constraint is conditional.** The Claude backup path carries **no grammar**
   (`llm_router.py:492`), and it is entered by design on low-confidence risky actions. Any ruling
   that leans on GBNF should say what governs path 2.
2. **Grammar resolution is CWD-relative** (`llm_router.py:475`, `chapter_13_llm_advisor.py:312`).
   The Chapter-13 advisor **silently degrades to unconstrained decoding** when the file is not found
   from the process CWD (`:313-330`); only an INFO log distinguishes the two. The advisor path fails
   loudly instead (`FileNotFoundError`), which is the safer of the two behaviours.
3. **Governance does not inspect the fields that execute.** `selfplay_overrides` reaches
   `selfplay_orchestrator` without passing `_validate_bounds` — bounded today only by Pydantic
   `ge`/`le` and the three-field clamp, both of which live in the advisor's own module rather than
   in the version-controlled policy file.

---

## 8. Open questions Beta must rule on

**Not answered here, by instruction.**

1. **Q1 — Does grammar-constrained + Pydantic-validated emission change the data-pipeline
   restriction?** §7 establishes what is enforced on each of the three paths. Beta rules on whether
   that is sufficient to let advisor output reach pipeline parameters, and if so, on which paths.
2. **Q2 — What governs the ungrammared backup path?** If GBNF is load-bearing for Q1, path 2
   (`llm_router.py:492`) has schema validation but no token constraint. Ruling needed on whether
   path 2 is acceptable as-is, must be grammared, or must be barred from parameter-bearing output.
3. **Q3 — Does §4's verdict promote multi-sampler selection to a live requirement?** The advisor
   controls selfplay via `max_episodes`/`min_fitness_threshold`, but `search_strategy` — the field
   multi-sampler selection depends on — is disconnected. Beta rules whether the general premise
   suffices or whether the specific lever must be live first.
4. **Q4 — Which name is canonical, `strategy` or `search_strategy`?** §5.1: the Step-1 manifest uses
   both for one concept, and the key-name split is what severs hops 2→3. A reconciliation ruling is
   a prerequisite for any wiring, and must address the double-emission collision.
5. **Q5 — Should `strategy_recommendation.json` have a reader, or be reclassified?** Chain B is
   documented as a WATCHER bridge in three contracts but has no consumer, and the named function
   `_apply_strategy_recommendation()` (`SESSION_CHANGELOG_20260207_S66.md:203`) was never built.
6. **Q6 — Is the whitelist inversion (§3.2) intended?** The operative bound for the only working
   application seam is hardcoded in `watcher_agent.py:1683-1690`; the policy file's declared key
   does not exist. Ruling needed on where that boundary should live.
7. **Q7 — Break B: manifest `default_params` or the whitelist?** The six LLM-adjustable Step-5
   params are dropped at `:1393`. Two candidate remediation sites; also whether the misleading
   `Applied:` INFO log (`:1794`) should be corrected before either.
8. **Q8 — The nine declared-but-unimplemented policy keys** (§6.2 items 7–12, 17–19). Beta rules
   per item: build, or formally mark as declared-not-implemented. **No removal is proposed here.**
9. **Q9 — The S140b Step-1 grammar patches (§6.1).** Both landed in non-loaded copies and are
   additionally blocked by their Pydantic enums and (for `step1_relaunch`) by having no handler.
   Beta rules on whether the S140b intent stands and, if so, the order of repair.
10. **Q10 — Grammar-directory consolidation.** Three loaders, two directories, three diverged files.
    Beta rules on canonicalisation; note `grammar_loader.py` v1.1.0 already solved CWD-relative
    resolution and the fix was not propagated.
11. **Q11 — The `0.0` truthiness guard** (`watcher_dispatch.py:151`). Same class as the §2.7 #2
    threshold regression. Beta rules on whether this is a defect to schedule.

---

## 9. Coverage table and completion sentinel

### Verification-integrity controls (VIR-1…6)

- **execution proof** — every claim carries a `file:line` anchor read in this session at `179f7cd`.
  Import availability of `diagnostics_llm_analyzer` / `diagnostics_analysis_schema` verified by
  executing `importlib.util.find_spec` in `~/venvs/torch`. Grammar divergence verified by `diff`.
  Manifest key sets verified by parsing the JSON, not by reading prose.
- **clean control (VIR-2)** — hops verified **correctly connected**, not merely unbroken:
  1. `advisor.analyze()` invoked on the live selfplay dispatch path — `watcher_dispatch.py:473-481`.
  2. Override merge into request — `:490`.
  3. `max_episodes` → `--episodes` rewrite — `:153-158`.
  4. `min_fitness_threshold` → `--min-fitness` — `:151-152` (non-zero values).
  5. Both flags accepted by argparse — `selfplay_orchestrator.py:1112-1116`, `:1135-1139`.
  6. Both reach live config — `:1182-1187`; govern behaviour at `:504` and `:533`.
  7. `_clamp_llm_recommendation` clamps 3 numeric fields and tags for audit — `:585-621`.
  8. `_validate_bounds` default-deny whitelist on unknown parameter names — `:1155-1157`.
  9. `validate_proposal` genuinely invoked — `chapter_13_orchestrator.py:395`.
  10. `max_parameter_delta` / `max_parameters_per_proposal` / `cooldown_runs` enforced —
      `chapter_13_acceptance.py:351-353`.
  11. `_build_retry_params` reachable from the Step-5 retry path — `agents/watcher_agent.py:2111`,
      `:2131`.
  12. `require_supported_strategy` fails closed on unsupported samplers —
      `window_optimizer.py:532-556`.
  13. Three of five grammars **identical** across both directories (`agent_decision`,
      `parameter_adjustment`, `sieve_analysis`) — divergence is not universal.
- **fault-injection (positive) control** — **not applicable to a read-only audit, and none was
  run.** Stated explicitly per §5 rather than omitted. Distinguishing a vacuous detector from a
  correct one at `_validate_bounds` and `_is_within_policy_bounds` would require executing them
  with synthetic proposals; §4 forbids execution of the autonomy stack, so both are reported from
  source reading and are labelled accordingly.
- **completion sentinel (VIR-3)** — see below.
- **unavailable-observer (VIR-5)** — items requiring execution are marked `UNAVAILABLE`, not assumed
  clean.
- **audit claim scope (VIR-6)** — **searched surfaces:** the VM 101 working tree at `179f7cd`
  (tracked and untracked), including `parameter_advisor.py`, `agents/watcher_agent.py`,
  `agents/watcher_dispatch.py`, `chapter_13_*.py`, `llm_proposal_schema.py`,
  `diagnostics_analysis_schema.py`, `selfplay_orchestrator.py`, `window_optimizer.py`,
  `llm_services/`, `agent_manifests/`, `watcher_policies.json`, both grammar directories, and the
  live host artifacts `strategy_recommendation.json` + `strategy_history/`. **Unavailable
  surfaces:** runtime behaviour of every chain (nothing was executed); the LLM server and its actual
  decoding behaviour; deployed copies on the three rigs/CTs; systemd units, cron and host config
  outside the repo tree. **The repository is not the system** — a systemd unit or an uncommitted
  deployed copy could consume `strategy_recommendation.json` without appearing in any search
  performed here.

### Coverage

| § | Required finding | Status | Where |
|---|---|---|---|
| 3.1 | Emit surface — every field, all schemas | **COMPLETE** | §2.1–2.5 |
| 3.1 | GBNF grammars — all, and grammar/Pydantic agreement | **COMPLETE** | §2.1, §2.3, §2.6 |
| 3.1 | Can the advisor emit `search_strategy`? | **COMPLETE — YES**, incl. live proof | §2.2 |
| 3.1 | Can it emit anything reaching Step 1's search space? | **COMPLETE — NO** | §5, §5.1 |
| 3.2 | `watcher_policies.json` declared vs enforced | **COMPLETE** — 19 keys | §3 |
| 3.2 | Confirm/refute `parameter_application` unimplemented | **COMPLETE — CONFIRMED** | §3 row 1 |
| 3.3 | Application surface, producer→consumer, exact break | **COMPLETE** — 4 chains, 2 breaks | §0.1, §1 |
| 3.4 | Selfplay control verdict | **COMPLETE — YES, partial (2/6)** | §4 |
| 3.5 | Four-hop February chain | **COMPLETE** — hops 1✅ 2⚠️ 3⚠️ 4❌ | §5 |
| 3.6 | Declared-but-disconnected inventory | **COMPLETE** — 20 items | §6 |
| 3.7 | What the constraints enforce | **COMPLETE** — 3 paths | §7 |
| 6.8 | Open questions for Beta | **COMPLETE** — 11, unanswered | §8 |
| — | Runtime confirmation of any chain | **UNAVAILABLE** — execution out of scope (§4) | VIR-5 |
| — | Rig/CT deployed copies; systemd/cron surfaces | **UNAVAILABLE** — not inspected | VIR-6 |
| — | Positive/fault-injection control on validators | **UNAVAILABLE** — requires execution | VIR-2 |

### Sentinel

```
STRATEGY_ADVISOR_AUDIT_v1 — COMPLETION SENTINEL

RESULT: PASS

  All required findings §3.1–§3.7 and deliverables §6.1–§6.8 addressed at
  commit 179f7cd on VM 101. Every claim carries a file:line anchor read this
  session. No code, config, schema, grammar or documentation modified. No
  commits. No removal recommended (§4). Beta's open questions left unanswered
  by instruction (§6.8).

  PARTIAL/UNAVAILABLE (VIR-5, not defects in coverage — out of audit scope):
    - runtime behaviour of all four chains: nothing was executed (§4 prohibits)
    - fault-injection control on _validate_bounds / _is_within_policy_bounds
    - deployed copies on rigs/CTs; systemd, cron and host config (VIR-6)

  TWO BRIEF PREMISES CORRECTED, WITH EVIDENCE (§0):
    - an LLM parameter-application seam DOES exist and is reachable
      (agents/watcher_agent.py:1789-1793, called from :2111); it breaks one
      hop later at the step-scoped manifest filter (:1385-1393)
    - the chapter_13_acceptance.py:224 anchor has drifted; the substance is
      confirmed via promote_candidate (:818) having zero callers

  HEADLINE: The Strategy Advisor DOES control selfplay operatively, via 2 of 6
  SelfplayOverrides fields (max_episodes, min_fitness_threshold), verified at
  all 7 hops to selfplay_orchestrator config. search_strategy is emittable,
  grammar-constrained to 4 values, has been emitted in production ("grid",
  strategy_recommendation.json:25), and reaches nothing.

STOP — Team Alpha review.
```

---

*End of `docs/STRATEGY_ADVISOR_AUDIT_v1.md`. Audit only. No changes made. Awaiting Team Alpha
review, then Team Beta ruling on §8.*
