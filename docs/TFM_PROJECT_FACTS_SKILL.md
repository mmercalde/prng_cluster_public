---
name: tfm-project-facts
description: Foundational model, verified as-built facts, superseded-artifact list, and mandatory verification procedure for Michael's TFM (Triangulated Functional Mimicry) distributed PRNG analysis project. Use this skill whenever the conversation touches TFM, the PRNG cluster, RANGE-MINER/S172, selfplay, Chapter 13/14, the bidirectional sieve, survivor pools, the NPZ contract, prediction pools, WATCHER, Zeus/VM101/rrig6600, or any file in the prng_cluster repos — even if the user does not name the project explicitly. Use before making ANY claim that something is missing, broken, unwired, unused, or current, and ALWAYS before proposing to remove, demote or simplify any component.
---

# TFM — Foundations, Verified Facts & Verification Procedure

**Currency:** v25, 2026-08-14. **D6.2 CERTIFIED `18a2419`.** **Attempts 3, 4 AND 5 all RAN and
FAILED** — attempt 3 dirty-tree admission (§2.34), attempt 4 stale lease origin (§2.36), attempt 5
stage-3→4 `worker_admission_timeout` 23/25 (§2.38). Three repairs are CLOSED / CERTIFIED: clean-tree
admission `213bfff` (§2.35), F1 lease-origin + serve-loop instrumentation `2b0d2dc` (§2.36),
attempt-6 remediation `69ff222` (§2.40) and the D6 integration repair `dd03f1d` (§2.42).
**§2.41 IS THE MOST IMPORTANT NEW SECTION: the rigs were running mixed-vintage stale code and
several previously-recorded facts are WITHDRAWN.** Gate-12 attempt 6 is HELD until the parked-fleet
D6 dry run passes; Phase 7 remains HELD.
*(Dated, not commit-pinned: a HEAD pin goes stale the moment anything else lands, and reads as
noise on the first line a session sees. Commit hashes belong where they anchor a certified
artifact.)*

**Three documents carry the facts this skill only points at. Read them; do not paraphrase them:**

| document | authority on |
|---|---|
| `docs/PROJECT_FILE_CATALOG.md` (`1fc05bb`, 803L) | **what documents exist and what question each answers** — the index |
| `docs/PIPELINE_BEHAVIOUR_MODEL.md` (1,603L) | **how the pipeline works and why** — every claim carries a WHY anchor and a WHAT anchor |
| `docs/PHASE6_PREREQS.md` REV5 | **operational fleet state as launched** — measured, and it expires |

**This skill is judgement, rules and pointers.** Where it still states a fact in full, that is
because the target was checked and found weaker, vaguer, or silent on it.

**§0 exists because of a specific failure.** In one session Team Alpha, Team Beta and Claude
Code *independently* recommended removing `skip_min`/`skip_max` from variable-skip search — a
cornerstone of the design — because no document any of them had read explained **why skip
exists**. All three inferred intent from current kernel signatures, which were the defect.
Michael stopped it. **This system has been in development over a year with the reasoning
documented and committed. If a component looks pointless, the explanation exists and has not
been read.**

---

## 0. FOUNDATIONS — read before proposing any change

### 0.1 The hypothesis
California Daily 3 draws are produced by a PRNG. Given observed history, find candidate
**seeds** whose generated sequences are consistent with what was drawn, then rank those
candidates with ML into prediction pools.
Basis: `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md`.

### 0.2 Why bidirectional
Forward: seed `s` survives if match rate ≥ τ_f. For a random seed p = 1/1000 per position, so
survival decays exponentially in window length n. Reverse applies the same test on reversed
indices. For **incorrect** seeds the two are approximately independent:

```
P(survive both) ≈ P(survive forward)²      e^(−cn) → e^(−2cn)
```

**Bidirectional squares the exponent — a catastrophic collapse of noise.** At exact match with
n=50, false survival ≈ 10⁻³⁰⁰: only the true seed survives. That is why the sieve is
bidirectional.

**Reverse kernels iterate the PRNG FORWARD.** Direction comes from reversing residues on the
host (`residues[::-1]`); there is no inverse LCG — most PRNGs aren't invertible without full
state. **Do not flag this as a defect.**

### 0.3 Why thresholds are LOOSE — and must stay tunable
Whitepaper §7, counter-intuitive and misread every time:

> Exact sieves eliminate *all* variance. Survivors = {s*}. No ranking, no gradients, **no
> learning signal.**

Loose thresholds deliberately admit a *manifold* of near-consistent seeds sharing structured
deviations ML can learn to rank. **Loose thresholds are not sloppiness — they are a
mathematical necessity to expose learnable structure.** ML then works in a
survival-conditioned high-signal posterior, not raw PRNG space. This is why thresholds are
Optuna-tuned per direction, and why a threshold silently pinned to a constant is serious.

### 0.4 Why SKIP exists — the physical model
**The part nobody had written down.**

The published draw sequence is **not** an uninterrupted PRNG output stream. Per the
*California State Lottery Daily & SuperLotto Plus Draw Procedures* (eff. 2021-06-09):

- **One automatic pre-test session runs before an automatic Daily draw** on the selected
  equipment (§V: Pre-Test via `[Start Draw Session]`). **Additional pre-test draws run only
  when an anomaly requires them.** Pre-test outputs are generated, verified, certified — and
  **never published.**
  *(Corrected 2026-08-01: previously "two pre-test draws run before every live draw" — an
  Alpha misreading. The "two test draws" language applies to **manual SuperLotto Plus
  equipment**. Only the count was wrong; skip remains physically motivated. Citation
  `UNAVAILABLE` — the PDF is not in the repo.)*
- **Draw equipment is selected per session** by an RNG program, auditor-verified (§II).
  Midday and evening are separate sessions with separate equipment selection.
- Evening draws **D3, D4, Fantasy 5 and Daily Derby together** — other games' outputs sit
  between the Daily 3 values you can see.

**Therefore the observable sequence has real, structural gaps of unknown and varying size.**
Skip models those gaps. It is a physical property of the data source, not a tuning convenience.

| mode | assumption | kernels |
|---|---|---|
| **constant skip** | fixed stride k between observed outputs | 22 kernels, all declare `int skip_min, int skip_max` |
| **variable skip (hybrid)** | stride varies, e.g. `[5,5,3,7,5,5,8,4,5,5]` | 22 kernels, declare `skip_sequences` + `strategy_tolerances` |

**Variable skip is a detector, not a fitting procedure.** It looks for windows where coherent
skip structure appears — the fingerprint glimpse — and produces survivors with *varied* skip
structure so tree/NN models have something to learn from. **The goal was never to reverse
state; it is to extract a fingerprint.**

**`skip_min`/`skip_max` ARE documented — in two readings, at two pipeline stages.** These are
**not contradictory**; they are the same names doing different jobs:

| stage | reading | source |
|---|---|---|
| **input** (*into* the sieve's search) | *"Minimum/Maximum skip value **in pattern**"* — an **element-wise bound** on the discovered sequence. Documented hybrid default `[0,16]` | `docs/instructions.txt:1182-1183`; verbatim in `Cluster_operating_manual.txt:948-949`; present in an older revision, so it predates the current file |
| **output** (sieve → Step-3 scoring) | *"Minimum/Maximum gap that **worked**"* — an ML feature describing what the sieve found. *"Tight skip range = stronger hypothesis"* | `PROPOSAL_ML_Architecture_Remediation_v2_0.md:150-158`; `config_manifests/feature_registry.json:336,345` |

**Two registries currently disagree** — `feature_registry.json` says *"found during"* (output),
`parameter_registry.json:160,166` says *"for sieve search"* (input). One is wrong; correct it in
whatever change settles this.

**Mechanics that correct a common misreading:** `skip_sequences` is an **output, not an input**
(`prng_registry.py:1071`). **No pattern is generated.** `expected_skip = 5` seeds a *greedy
per-draw adaptive search that re-centres on each hit* (`:1047`), and the ancestor file still
carries `// Initial guess` (`prng_registry_pre_registry.py:696`) — a **guess, not a constant**.
`strategy_tolerances` is the half-width of the per-draw **matching** window
(`hybrid_strategy.py:20`), not generation. **No coherence scoring exists** — only `match_rate`.

*(The CA draw-procedures PDF is **not in the repo** — open item.)*

> **Standing rule.** Before recommending removal, demotion or simplification of ANY component,
> find and cite the document explaining why it exists. **Absence of a working implementation
> is not evidence of absent intent.** In a codebase this old, "the code doesn't do X" usually
> means X broke — not that X was never wanted.

### 0.5 Autonomy adjusts parameters, never structure
Whitepaper §9. The loop tightens thresholds over time; WATCHER/LLM governance proposes
*parameter* changes within governed bounds. It does not redesign the sieve.

**Corollary:** every tuned parameter must physically reach the kernel and its effective value
must be observable. A parameter the optimizer tunes but the kernel ignores is a **dead
dimension** — the sampler steers a knob connected to nothing and an autonomous agent would
"learn" into a void. This has now happened four times (§2.7).

### 0.6 The pipeline — TWO NUMBERING SCHEMES, and which one this skill uses

> **⚠ Bare "Step N" anywhere in this skill means the EXECUTABLE scheme** — the one WATCHER runs.
> Confusing the two schemes is *"the single most common error a new session makes"*
> (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). **They agree at 1, 3, 5 and 6 and differ only at 2** —
> which is exactly what makes the mistake so easy, and why v12's unlabelled diagram was a hazard.

| **EXECUTABLE** — `STEP_SCRIPTS`/`STEP_MANIFESTS`, all 7 `agent_manifests/*.json`, `preflight_check.py`, `README.md` | **CONCEPTUAL** — whitepaper, system map §1, chapter titles |
|---|---|
| **0** Regime Segmentation (TRSE) · `trse_step0.py` → `trse_context.json` *(passively read by Step 1)* | — *(no conceptual number)* |
| **1** Window Optimizer (Optuna TPE) · `window_optimizer.py` → `optimal_window_config.json` **+ the certified 22-array NPZ generation**. **The bidirectional sieve runs INSIDE this step** — per trial, `run_bidirectional_test()` → backend cascade → forward ∩ reverse. **[RANGE-MINER (S172) is that engine]** | **1** Window Optimizer (Ch 1)<br>**2** Bidirectional Sieve (Ch 2) |
| **2** Scorer Meta-Optimizer · `run_scorer_meta_optimizer.sh` → `optimal_scorer_config.json` | **2.5** Scorer Meta-Optimizer (Ch 3) |
| **3** Full Scoring (**25 GPUs** — see §2.17) · `run_step3_full_scoring.sh` → `survivors_with_scores.json`, **91-feature vector** per seed | **3** Full Scoring (Ch 4) |
| **4** ML Meta-Optimizer · `adaptive_meta_optimizer.py` → `reinforcement_engine_config.json` (capacity only) | **4** Adaptive Meta-Optimizer (Ch 5) |
| **5** Anti-Overfit Training (**4 model families**) · `meta_prediction_optimizer_anti_overfit.py` → `best_model.*` + `best_model.meta.json` (**89 trained features**) + diagnostics | **5** Anti-Overfit Training (Ch 6) |
| **6** Prediction Generation · `prediction_generator.py` → `next_draw_prediction.json` + **pools 20/100/300** | **6** Prediction Generator (Ch 7) |
| **fb** Live Feedback · `chapter_13_orchestrator.py` — **not a WATCHER step** → ingest draw → grade → (attribute) → decide → relearn (3→5→6) | Chapter 13 |

**There is no executable step whose script is the sieve** (`agents/watcher_agent.py:386-416`).
**The mapping between the two schemes is written down nowhere** — `CHAPTER_3_ALIGNMENT_AUDIT.md` §2
searched for it and reports it as *folklore* (behaviour model §16, **DIVERGENT D1**).

Carriers: the **22-array NPZ survivor contract** (Step 1 → 2 → 3) and the **prediction pool +
coverage/lift score** (Step 6 → Ch 13).

**Full map — manifests, versions, primary outputs and the data-flow diagram:**
`docs/PIPELINE_BEHAVIOUR_MODEL.md` **§1.1** (the two schemes) and **§1.3**, parsed live from
`agents/watcher_agent.py:386-416` and all seven manifests. **Do not retype it from memory.**
Conceptual detail: `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` (binding).

### 0.7 Why RANGE-MINER exists
PWC suffered silent hard resets / `GCVM_L2_PROTECTION_FAULT` on the RX 6600 **XT** rigs at full-fleet
saturation, traced to launch-storm behaviour. After weeks of failed debugging the project
pivoted to RANGE-MINER: **persistent per-GPU daemons**, standalone, producing **all** data the
remaining steps require — *the remaining steps must not be able to tell which engine produced
it.* That is an **interface** contract (the 22 arrays), not "match PWC's values."
**PWC is NOT the authoritative comparator** — Beta retired it from certifying authority
(2026-07-31); it is a flag-selectable, non-certifying diagnostic path.

---

## 1. THE RULE

**Never assert anything is missing, broken, unwired, unused, current or superseded without a
`file:line` anchor obtained in THIS session.** Unverifiable → label **[UNVERIFIED]**.

- **Trace the whole path**: producer → artifact → consumer. Real code with no producer is not
  "wired."
- **Check supersession.** 180+ sessions; existence ≠ current.
- **The repository is NOT the system (VIR-6).** systemd units, cron, host config, deployed
  uncommitted files are invisible to every repo gate. *(Alpha reported "no scraper invoker
  exists" from a clone; an enabled boot-triggered unit was on the host.)*
- **Read the consumer before specifying a change** — every producer, consumer, owner,
  lifecycle transition, path-type requirement, downstream schema dependency. For paths also:
  regular-file vs symlink ownership, compatibility aliases, atomic-replace boundaries,
  cleanup, concurrent-run namespace, restart consumers. *Three briefs in a row were defective
  for skipping this.*
- **§0.4's standing rule** — cite the design document before proposing removal.
- **Cited is not read.** A document named in an audit, brief or changelog has **not** been read
  because its name appears. Open it. *(F6's specification sat in
  `TRSE_INTEGRATION_PLAN_S121.md` — tracked at `643cc30`, cited repeatedly, unopened.)*
- **Gitignored files are invisible to every repo-scoped search.** `.gitignore:41` is `*.json`;
  `agent_manifests/trse.json` — the file **causing** TRSE F1 — had no git history at all. Check
  `git check-ignore` and the filesystem before concluding a config or manifest is absent.
- **The keyword you chose is not the code's vocabulary.** Search the *behaviour* as well as the
  name you expect. The LLM parameter-application seam existed under different naming and was
  reported absent.
- **Before ANY absence claim, enumerate the surfaces:** tracked repo · gitignored files · git
  history including deleted · host state (systemd, cron) · **pre-repository archives on ser8**.
  Name which were searched and which were not. **The project predates its repository** — the
  initial commit is 2025-11-29 and `prng_registry.py` is already in it.
- **A keyword hit is not a finding until the surrounding text is read.** *Four* absence claims
  were falsified in one session. The last — "nobody documented skip semantics" — was made after
  a full-tree grep **that reached the exact line and did not read it**
  (`HYBRID_SKIP_BOUND_AUDIT.md:318` vs `instructions.txt:1182`). Widening the search surface
  does not fix this; only reading the hits does.

### 1.1 SEARCH ORDER — governance trail, then chapter, then code

**Inverted from what comes naturally, and that is the point.** Code-first finds what the code
does and never finds what was decided.

```
1. GOVERNANCE TRAIL  docs/TB_RULING_* · TB_RULING_REQUEST_* · PROPOSAL_* · TEAM_ALPHA_*
2. CHAPTERS          docs/CHAPTER_*  — the knowledge layer, by design
3. CODE              the implementation
```

**The documentation taxonomy** — so neither Alpha nor Claude Code discovers a category by accident
mid-task:

| pattern | answers |
|---|---|
| `TB_RULING_*` · `TB_RULING_REQUEST_*` | **what was decided, and what is still open** |
| `PROPOSAL_*` | what was designed and why |
| `CHAPTER_*` | how a stage works and why it exists |
| `TEAM_ALPHA_*` | what was submitted, and what it was answered with |
| `CLAUDE_CODE_INSTRUCTIONS_*` | what was asked for, and its constraints |
| `SESSION_CHANGELOG_*` | what happened when |
| `PROJECT_FILE_CATALOG.md` | **THE INDEX — READ IT FIRST.** Regenerated `1fc05bb` 2026-08-03: 803 lines, **intent-indexed** (what question each document answers, not what it is called), 562 files accounted for. **§1.1 is the governance trail** with each ruling request paired to its ruling and implementation commit. It carries ★ markers on documents previously misreported — e.g. `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`: *"read this before reporting any Step-2 objective blindness."* |
| `PIPELINE_BEHAVIOUR_MODEL.md` | **HOW THE PIPELINE WORKS AND WHY — the second mandatory first read.** 1,603 lines. **Every claim carries two anchors: a WHY** (chapter / whitepaper / proposal / TB ruling, cited `file:§`) **and a WHAT** (`file:line`, read live). Three markers are load-bearing: **`INCOMPLETE`** (only a code anchor was found — *a statement about that search, never about the repository*), **`DIVERGENT`** (doc and code disagree; **both recorded, neither adjudicated** — §16 is the register), **`GOVERNED`** (already diagnosed or ruled on — **re-reporting one as a new finding is a governance error**). |
| **`apply_s*.py` / `verify_s*.py` (repo root)** | **THE ONE-SHOT PATCH CORPUS — governance lives in code here.** 123 `apply_s*.py` + 4 `verify_s*.py` at HEAD, session-scoped and already applied. **Their docstrings quote TB rulings verbatim.** `apply_s142_partition_runid.py:5-24` records the TB-confirmed root cause of the partition `run_id` collision (*"~50% of COMPLETE trial rows missing, no exception, no print"*); `apply_s142c_remove_worker_writes.py:6-22` records **TB Option A superseding it**. Indexed at catalog §4.8. **Forensic only — never re-execute them.** |
| `docs/BACKLOG.md` | the tracked non-blocking register |

**⚠ The patch corpus is the lesson, not the row.** `PIPELINE_BEHAVIOUR_MODEL.md` §17's own
*"where to look next"* preamble enumerated the surfaces most likely to hold a missing WHY — the
changelog corpus, the unaudited chapters, `instructions.txt`, `Cluster_operating_manual.txt`, the
binaries, and ser8 — **and did not list the patch corpus. I-6's answer was in it anyway**
(behaviour model §17.1, *"What this pass changes about where to look"*). **No taxonomy in this
project named that surface until 2026-08-03.** An enumeration of surfaces is itself a claim that
can be incomplete, and this one was.

**A defect that is known, escalated and mid-remediation is NOT a finding — it is a status.**
Reporting it as new tells the reviewer about its own ruling. *(Alpha nearly submitted the Step-2
objective's dead-signal problem to Beta as a discovery. `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`
had already diagnosed it with live Zeus stats — `bidirectional_selectivity` 98.8% at floor across
6,739 survivors — and the v4.1→v4.2→v4.3 evolution is that ruling process running.)*

**Corollary — EVERY LINE OF THIS CODEBASE HAS BEEN DOCUMENTED.** Michael's statement, and it has
survived every test put to it. **A component with no obvious explanation means the explanation has
not been found yet.** §0.4's standing rule is not advice; it is a description of how this repository
actually is.

### 1.2 A REPORT IS A SNAPSHOT, AND ITS FINDINGS EXPIRE

**Relaying a finding from a dated artifact is MAKING THAT CLAIM YOURSELF**, and it needs a fresh
anchor like any other claim.

**The documents most likely to mislead are the ones that were RIGHT.** A submission arguing for a
fix is accurate about the moment before the fix — and is therefore **the least reliable source on
whether the fix happened.**

*(Alpha read `TEAM_ALPHA_CHAPTER_2_RECOVERY_SUBMISSION.md` — correct when written — and reported
its pre-restore finding as current state, having told Michael hours earlier in the same session
that Chapter 2 was restored and closed. The contradiction was internal and went unnoticed.)*

**An audit performed without the governance trail produces false findings at a high rate.** Before
relaying ANY audit's findings, **re-derive each against the trail.** The audit's anchors prove the
code says what it says; they do **not** prove nobody knew.

---

## 2. SETTLED FACTS

### 2.1 Objective lineage
`score` → `holdout_hits` → **`holdout_quality`**. R² abandoned (0.000155 — zero signal);
**R² is not the objective.** Pipeline objective (S140b): `0.50·hit@20 + 0.30·hit@100 +
0.15·hit@300 + 0.05·pool_coverage`. `evaluate_pools.py` already computes coverage **and lift
vs random** — do not propose building it. Open: selfplay inner loop still uses R²;
`selfplay_orchestrator.py:933` prefers `holdout_hits`.

### 2.2 Feature contract
**91 extracted / 89 trained** (excl. `score`, `confidence`). "~62" is stale by 31. Three
namespaces: **72 survivor-local** (legitimate search space); **14 `global_*`** run-context
(identical for every survivor — filtering can only retain/remove the whole run; random row
folds leak run identity); **5 dead placeholders** with no producer (`skip_mean`, `skip_std`,
`skip_entropy`, `survivor_velocity`, `velocity_acceleration`).

**Three of those five are skip-shape statistics whose producer EXISTS on the GPU.** They are
dead because `extract_survivor_records` (`window_optimizer_integration_final.py:125`) builds each
record as **`{seed, match_rate}` only** — verified live 2026-08-03 — discarding `skip_sequences`.
*(v12 anchored this at `:147`, which is now inside the function body. Cite the symbol.)* The Oct-2025 output spec (`instructions.txt:1230-1245`) declares
`skip_pattern` and `pattern_stats: {mean_skip, variance, std_dev}` per survivor — the literal
ancestor of the three. **Reviving them requires no kernel change**, only that the host stop
discarding the sequence.

### 2.3 The 22-array NPZ contract (frozen)
Only **4 columns carry per-seed information**: `seeds`, `forward_matches`, `reverse_matches`,
`score`. **RANGE-MINER emits exactly 22 arrays, nothing more.**
`forward_matches`/`reverse_matches` are the only independent per-seed sieve signal and are
**absent from the Step-3 merge list** — TB: possibly the most consequential finding in the
trace. Needs a governed schema decision; the miner keeps emitting both regardless.

### 2.4 Attribution
`per_survivor_attribution.py` is real and invoked with seed identity — **implemented, invoked,
unreachable, unconsumed.** Four blockers. Never say "wired" or "not implemented"; both wrong.

### 2.5 Selfplay
A **policy-conditioned evaluation harness**, not a learning system.
`propose_transform_update` is a no-op; promotion seam broken
(`chapter_13_acceptance.py:224`). All Ch13 triggers **defensive**; no opportunity trigger.

### 2.6 Looks-like-a-bug, isn't
Reverse = host-side residue reversal (§0.2). Loose thresholds required (§0.3).
`intersection_count` duplicating `bidirectional_count` is deliberate.
**Step 0's silent failure is ARCHITECTED** — `trse.json` sets `skip_on_fail: true` with a
stated reason, and `TRSE_INTEGRATION_PLAN_S121.md` §2C specifies a **PASSIVE** integration
(*"Step 1 reads `trse_context.json` on its own if present… WATCHER doesn't need to parse or
inject anything"*). Step 0 failing invisibly is the design, which is exactly why F1 went
unnoticed for months. **`serve_timeout=None` is deliberate** — a billion-seed scan exceeds any wall clock; the
bounded clock is on *admission* only (§2.12). **`distributed_config.json`'s bare-metal
addresses are deliberate** — both topologies are retained as profiles (§2.11). **`run_optimization`'s `sampler` and `sampler_metadata` are
REQUIRED and keyword-only with no default** — deliberately, so a caller cannot get TPE by
omission and then report the run as something else; an unlabelled run is not a control.
**Chain D's `pending_approval`** is a valid authority boundary and the **Step-5 `allowed_params` filter**
is a deliberate executable-interface boundary (§2.13).

### 2.7 Recurring defect: tuned parameters don't reach kernels — SEVEN instances

| # | instance | status |
|---|---|---|
| 1 | miner filtered at hardcoded `0.25` | **FIXED** `2be51d5` — single canonical path, per-direction resolution in the parent, effective value read off the executor, parent-side fail-closed provenance |
| 2 | Optuna thresholds dropped above `run_bidirectional_test`; every trial ran `0.30/0.30` | **FIXED** `8a55a68`. Was a **regression**: fixed `3fdf434` (04-30), silently reverted `2389b61` (07-07) by an **out-of-tree working-file overwrite** (§2.7b — *not* "a pre-fix copy", which is what earlier revisions of this skill said) whose message never mentions thresholds. Both routes now use `resolve_directional_threshold()`, `is None` not truthiness (**0.0 is legitimate**). **`3fdf434`'s companion defensive fix — the `run_bidirectional_test` signature defaults `0.01 → 0.50` — was reverted by the same commit and is STILL REVERTED at HEAD**; governed as F3 in `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md:32`, dead path in production (both production callers pass thresholds explicitly) |
| 3 | PWC hybrid filtered at `0.50` | **QUARANTINED** — `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`; PWC non-certifying, so the defect is made loud rather than repaired |
| 4 | **hybrid kernels ignore sampled `skip_min`/`skip_max`; `expected_skip = 5` hardcoded** | **OPEN.** 22/22 constant kernels declare skip bounds; 0/22 hybrid do. Values survive eight hops and **die at `_hybrid_prefix`** (`range_miner_worker.py:177-193`). Anchors `prng_registry.py:1027, :805, :885, :1159`. **Semantics ARE documented (§0.4)** — the "unspecified semantics" premise of `HYBRID_SKIP_BOUND_AUDIT.md:318` is **FALSE**. Decision open; **the output-statistic reading needs no kernel change at all** |
| 5 | forward hybrids ignore `offset` (sampled `window_optimizer_bayesian.py:423`) | **OPEN.** Chapter 2 F-4: `offset` drives **both** the host residue slice **and** the device pre-advance from one payload scalar — coherent only at `skip=0`. Settles Chapter 1 C-2 as an **observed inconsistency, not a repair**; belongs in the future hybrid input-semantics design, **not** a standalone arithmetic patch (TB) |
| 5b | **`recommended_window_size` → Rule A** — the manifest declares `8`, the code reads it into `_rec_ws` (`window_optimizer_bayesian.py:500`) and **never references it**; Rule A uses a hardcoded `32` | **ROOT CAUSE FOUND.** `TRSE_INTEGRATION_PLAN_S121.md` §2C specifies `min(rec_ws * 4, …)` — and `8 × 4 = 32`. **The value is correct; the wiring is missing.** Not "a field of unclear purpose" — a **configurable input frozen at its default by a literal** |
| 6 | **`skip_learning_rate`** configured 0.2–0.7; kernel **hard-adapts at 1.0** | **OPEN**, newly catalogued |
| 7 | **the four staging-capacity controls** (`staging_workers` / `staging_queue_depth` / `staging_deferred_max` / `staging_capacity_timeout`) passed to `run_trial_miner` were **silently swallowed by its `**kwargs` tail** — accepted, never applied; the symptom was a HANG until `serve_timeout`, not an error | **FIXED** in the S172-BP remediation (explicit parameters, full manifest→`CoordinatorConfig` route, gate G10). Beta-accepted for this register 2026-08-06 ("fourth instance of the §2.7 silent-no-op defect class" in Beta's own count of that subclass) |

Fix pattern: **one canonical path** — resolve once in the parent, never reinterpret
downstream, record requested/payload/effective.

### 2.7b `2389b61` — THREE out-of-scope reverts, and dates cannot bound the damage

**⚠ CORRECTED 2026-08-04. Earlier revisions recorded this commit as *"rewritten from a pre-fix
copy."* That is NOT what happened, and the wrong model produces wrong predictions.**

`2389b61` (2026-07-07, *"feat(s172): Phase 0 — shared PRNG_TYPE_ENCODING v3.2"*) was diffed in
full for the first time on **2026-08-04**. Four files, +458/−47. **Two files have zero deletions**
(`tests/test_prng_encoding.py`, `utils/prng_encoding.py`) — no revert can live in them, so every
deletion is in the other two. `utils/survivor_loader.py`'s two hunks are in scope.
**`window_optimizer_integration_final.py`'s six hunks are ALL out of scope** — not one of them
touches `prng_encoding`. That file was overwritten for **no in-scope reason at all**; the encoding
changes the message claims for it (*"inline writer"*) and for `convert_survivors_to_binary.py`
did not land until `66f0425` / `46a3828`, 17–18 days later.

| hunk | out-of-scope revert | added by | at HEAD |
|---|---|---|---|
| H2 | `run_bidirectional_test` defaults `0.50 → 0.01` | `3fdf434` (04-30) | **still reverted** — governed as F3, dead path |
| H3 | `min_workers = getattr(coordinator, 'pwc_min_workers', 1)  # [S174]` | **`ca06f8c` (05-08)** | **still reverted — see §2.11** |
| H4/H6 | the 7 `warm_start_*` params (signature + `_trial_history_ctx`) | `8cb2ada` (04-21) ×6, `a6bc546` (04-22) ×1 | **RESTORED 2026-08-04** |
| H5 | `test_config` call-time threshold resolution | `3fdf434` (04-30) | restored `8a55a68` |
| H1 | S166 clear repositioned, one comment line lost | — | not a revert; region rewritten by D6.2 |

**THE MECHANISM — and why it matters.** The overwrite source was **not any committed revision**.
Every ancestor touching the file (`e8a69f5`, `a6bc546`, `3fdf434`, `ca06f8c`) places the S166 clear
**after** the flush print with three comment lines; `2389b61` writes it **before**, with two. No
ancestor and no `apply_s*.py` produces that arrangement, and the `_trial_history_ctx` DB-lookup
block it introduced **has no git history before it**. `docs/window_optimizer_integration_final.py`
— a stale duplicate last touched `7313a43` (05-03) — is close but still carries the warm-start
params.

> **The pasted copy was an OUT-OF-TREE WORKING FILE that had absorbed some later fixes and not
> others.** That is why the damage is **NON-CONTIGUOUS**, scattered across April and May instead of
> truncating cleanly at one date.

**⚠ CONSEQUENCE — DATE-BASED REASONING ABOUT THIS COMMIT DOES NOT WORK.** You cannot bound its
blast radius by "everything after date X". **Three reverts are known: two were found only by
targeted audit (the threshold fix, four months late; `min_workers`, thirteen months of PWC runs
later) and one by a launch failure** — Step 1 died at `TypeError` three seconds into a 50-trial
soak on 2026-08-04 because `window_optimizer.py:802` passed seven kwargs the signature no longer
accepted. **Nothing establishes that three is the total.** The only surface ever fully checked is
`window_optimizer_integration_final.py`, and it was checked by reading every deleted line, not by
reasoning about dates.

### 2.8 RANGE-MINER Phase 5 as-built (committed, dual-pushed)

> **Companion, not a replacement:** `PIPELINE_BEHAVIOUR_MODEL.md` §3.7 carries what this section
> does **not** — the **per-module role table** (`miner/` × 7), the seed-domain partitioning proof
> obligations, the shared residue-window authority, and **the four `process_sharded` promotion
> criteria** (≥20% median end-to-end gain · identical final arrays · ≤50% host-RAM peak RSS · no
> swap — `PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §17). **It does not carry the D6.1 checkpoint
> mechanics, the authoritative artifact identity, or the P0/P0.5/Q2/§4.3 commits below** — those
> live only here.
**D3.5** shared run finalizer, immutable chain-authenticated generations; owns the root
compatibility **symlinks** and **fails closed if a regular file appears there**.
**D4** `serial_reference` behind a frozen two-backend interface that fails closed.
**D5** `process_sharded` parallelizes **only** spool-local validation; parent alone owns
merge/dedup/intersection. **Available, UNPROMOTED** (~1.6× faster high-survivor at ~2–3× RAM;
~180× slower low-survivor).
**D6** production adapter; miner candidates reach the Step-1 accumulator and a certified
generation; per-direction thresholds reach the kernel with requested/payload/effective
provenance and parent-side fail-closed enforcement; shared session-filtered residue authority.
**D6.1** incremental checkpoint **writes for the first time** — run-isolated
`.s172_checkpoint/<run_id>/`, open-handle write + fsync, six-field transaction identity.
**NON-AUTHORITATIVE four-field snapshot.**
**Phase 6.0** RX 6600 XT (ROCm) and RTX 3080 Ti (CUDA) produce a **byte-identical certified
artifact** — same `artifact_sha256` across the D6 release-grade generation and both 6.0 runs;
22/22 arrays equal; **no GPU reset, no `GCVM_L2_PROTECTION_FAULT`** in the host kernel log.

**Authoritative artifact:** `gen-20260730T002104136270Z-step1_java_lcg_0`, commit `b08c2c5`,
`artifact_sha256 0e0092fe…c4b0` — the **pre-dataset-provenance authoritative generation**.
The ROCm generation is **platform-validation, non-authoritative**.

**Phase 6-P0** `131787d` · **6-P0.5** `d4ff1e4` · **P0.5 Q2 closure** `8600e75` — dataset
authority, see §2.10. **§4.3 admission liveness** `ee0db06` — see §2.12.

**Bounded Phase 6 — CERTIFIED and CLOSED** `d98298c` (TB ruling, 2026-08-02). Wall A: the
complete consumer chain — frozen 22-array bundle → validation → Step-2 load without fallback →
dict conversion → Step-3 chunks → real GPU scorer, with **value-by-value** metadata comparison
(closing the "keys present but values defaulted" class). Wall B: repetition, assembly-backend
equivalence, current CUDA/ROCm equivalence, and **node-assignment independence across two
different ROCm rig pairs** — all five arms reproduced `0e0092fe…c4b0`. **Miner Known-Answer
Transfer Gate:** all four active TFM variants through their real `SieveExecutor.execute` ABI
paths; **eight populations exact-set equal, zero missing / extra / mismatched**; F5–F7 prove
reference independence by rejecting three wrong semantics.

**Certification scope is explicit:** Wall A/B used **constant-skip** generations; **hybrid worker
semantics are covered by the transfer gate, not by a full four-phase Wall-A consumer run.**
The scratch generations are **not** release-grade — future publication still uses
`--release-grade`.

### 2.9 Known-disabled / deliberately off
**S166 in-memory clear — ENABLED** (`f7583bc`, repaired `18a2419`). `_FLUSH_CLEAR_IN_MEMORY = True`;
the checkpoint carries the complete 24-field canonical state and the finalizer is fed the
reconstructed cumulative state, not the truncated stump. **The OOM protection is real for the first
time — and has NEVER RUN AT SCALE. The Phase-7 soak is its first real exercise; RAM across 50
trials is the headline result, not throughput.**

**D6.2 CERTIFIED `18a2419` — `n_parallel == 1` ONLY.** 31/31 gates, 377 assertions, 25/25 mutants.
**Checkpoint recovery and the S166 clear are certified for the single-Optuna-trial path only** —
that path still distributes each sieve trial across the whole fleet; the limit is on Optuna
parallelism, not cluster use. **No claim for `n_parallel > 1`:** not resume, not clearing.
Concurrent partition writers cannot share the member pair; that needs a separate transaction
design. **`resume_checkpoint` + `n_parallel > 1` is refused as the first executable statement of
`optimize_window` (`:1979`)** — above the NP2 block, study creation, the `[NP2-KILL]` SSH and the
fork. **The soak must pin `n_parallel=1`.**

**⚠ Two defects Beta caught at `f7583bc` that the 29 gates did not reach.** Both were execution-path
only, and both are instructive:
1. **The retained Optuna guard compared `int(trial.number) <= int(floor)`** — a **0-based** study
   number against a **1-based record-ordinal** floor. With `k` completed trials the floor is `k`
   and the next legitimate number is `k`, so **every normal resume was rejected.** The gate missed
   it because it used **fabricated values (trial 6 vs floor 5)** instead of the real relationship —
   the VIR-2 vacuous class. *The false name `_resume_trial_floor` is what made the comparison look
   reasonable; it is now `resume_record_ordinal_floor` and seeds the record counter only.*
2. **NP2 ran ~600 lines before the D6.2 context existed**, so a checkpoint resume could drive SSH
   and the whole optimization before being rejected.
**`.s172_checkpoint/<run_id>/` never pruned** — D6.3. **NOT a Phase-7 blocker on measurement:**
census 2026-08-02 shows **25 run directories, 50 files, 266,835 bytes** accumulated over ~2 days —
**~10.7 KB per run.** Directories are per-**run**, not per-trial. *(A brief queued as a read-only
investigation; the growth question it exists to answer is largely answered by this figure.)*
**`process_sharded`** selectable, unpromoted.
**`daily3scraper.service`** — enabled since Sep 2025 with `Restart=always`, target
`run_daily3scraper.py` **never existed**; ENOENT loop every boot. **Now `disable --now`, unit
retained.** Stays disabled until Phase 6-P2 is certified.
**⚠ THAT ENOENT LOOP WAS A SAFETY INTERLOCK, NOT MERELY A DEFECT.** `daily3_scraper.py`
(**Revision 1.5**, located on ser8 and now tracked at repo root) does
`Path(OUTPUT_FILE).write_text(json.dumps(all_draws, indent=2))` — **a full overwrite that never
reads the existing dataset**. With `--recent` (`start_year = end_year = TODAY.year`),
**`daily3_scraper.py --json --recent` replaces 18,068 records with the current year alone.** No
merge, no dedup, no error, **exit status success**. `write_text` is not atomic either.
**Repointing that unit at the real scraper — an obvious, well-intentioned repair — would have
destroyed the canonical dataset on the next boot.** After 6-P2 certifies, activation is a
**one-shot service plus timer**: a terminating scraper under `Restart=always` runs continuously
and hammers the source.
**`daily3_scraper.py` had NEVER been in git history** and is not gitignored — the producer of the
canonical dataset was never under version control. Same class as `agent_manifests/trse.json`.
**`pa_pick3_scraper.py`'s header claims dedup and sorting; the CA original does NEITHER** — do not
infer CA behaviour from the PA descendant.
**`_RusageChildrenSampler` measures the wrong thing and passes by luck.**
`tests/test_s172_phase5_d5_process_sharded.py:2107-2119` reads
`getrusage(RUSAGE_CHILDREN).ru_maxrss` — a **process-lifetime** high-water mark over every child
ever reaped, **not scoped to its `with` block**, despite a docstring saying "any SINGLE reaped
child". Measured: 0 → trivial child 10 MiB → **one torch child 378 MiB** → another trivial child
**still 378 MiB**. `G-RSS` therefore depends on no earlier child exceeding its own ~339 MiB
tree-sum; when the import gate first sat beside `G-NO-GPU`, G-RSS red and **mutant M8 survived**,
deterministically. Contained by ordering. **Any future D5 arm reaping a large child reds it the
same way.** Scope-correct fix (delta from `__enter__`) flagged, not actioned — BACKLOG.
**PWC/ZMQ** retired from certifying authority; PWC hybrid additionally quarantined.
**TRSE F1 is manifest drift, not a design error.** `TRSE_INTEGRATION_PLAN_S121.md` §2B shows
**six** `default_params`; the live `agent_manifests/trse.json` has **seven** — `trse_context`
was added later, and the plan's own §5 places it in **`window_optimizer.json`**, not
`trse.json`. `window_size`/`stride`/`k_clusters` were in the plan from S121, so that half of
the CLI mismatch was baked in at design time.
**⚠ CORRECTED 2026-08-03 (catalog §5.3).** The earlier claim — *"trse.json is the only gitignored
manifest and has no git history"* — is **false as of `93918f5`**. Live: **all 9 files in
`agent_manifests/` match `.gitignore:41` (`*.json`)**; the **7 step manifests are force-added and
tracked**; `trse.json` was force-added at **`93918f5` (2026-08-01)** and has **exactly one commit**.
**`definitions.json` is now the only untracked, ignored manifest** — it carries `schema_version`,
`pipeline_steps`, `sidecar_schema`, `watcher_protocol`, `description`, `updated_at`, and **no
`default_params`. A FRESH CLONE DOES NOT HAVE IT**, so no clone-based reasoning can see the file
that declares the pipeline's own step structure.
**`dataset_provenance/*.json` never pruned** — same class as D6.3, newly found.
**Sampler provenance is unverified** — `run_optimization()` trusts caller-supplied
`sampler_class` / `sampler_module` / `optuna_version` and does not check them against the actual
object. Existing TPE and Random wrappers are correctly labelled, so nothing submitted is
invalidated; **a fail-before-study guard is required before direct use of the neutral core or
registration of another sampler.**
**`process_sharded` import gate — CLOSED** (`e0513ba`).
`tests/test_s172_process_sharded_import_gate.py`, **7 gates + 3 mutants**, wired into D5 via one
`_check` row; **D5 is now 25 gates**. Fresh interpreter, production `_FORBIDDEN_GPU_MODULES`
**read not restated**, both `torch` and `cupy` injected at runtime, mutants exit 3 with
`ShardArtifactError` naming the module.
**The surface is the MINER chain, not the Step-1 host module — and that is not a compromise.**
`window_optimizer_integration_final` **holds cupy at import time** via `sieve_filter.py:51-52`
(module-scope `import cupy`, reached from `…final.py:53`), so importing the Step-1 host and
calling `assert_cpu_only()` **reds on a clean tree**; the brief's literal wording was
unsatisfiable. The gate uses `miner` + `miner.step1_ingress` (AST-derived, transitively all 9
miner modules) — **precisely the chain `assert_cpu_only`'s docstring describes**. `G-HOST-BOUNDARY`
**measures** the exclusion rather than assuming it.
*Corollary: the Step-1 host process legitimately holds a GPU library. Assembly workers are safe
because they are **spawned**, not forked. Making the host GPU-free solves nothing; switching
assembly to `fork` breaks the invariant silently.*
**`quick_test_all_22.sh`** is **differential/liveness evidence only, never known-answer
correctness evidence** (TB). Its output path is now timestamped so the supersession record
survives.
**`random`/`grid`/`evolutionary` strategies** gated at the CLI (signature mismatch) — **not
deleted**. Documented design was four Optuna samplers; `GridSampler` is **unconstructible**
here (7.649 × 10¹⁰ grid points ≈ 7.2 TiB at construction).

### 2.10 Dataset authority — LIVE as of P0.5
- **The pointer manifest is authoritative.** `daily3_current.json` → immutable
  `daily3-<UTC>Z-<sha256[:12]>.json`. **`daily3.json` is now a legacy compatibility alias.**
- **Resolved ONCE at run start and frozen** — manifest identity, absolute path, sha256, size,
  record count. A pointer moving mid-run **cannot alter a run in progress**. `dataset_sha256`
  moved from **per-trial to run scope** (`range_miner_coordinator.py:85`); a mid-study scrape
  used to split a study across two datasets with **no error anywhere**.
- **Dispatch is the absolute immutable path**, never the bare alias. **Fail before first worker
  dispatch.** All three rigs provisioned through one path, digests verified **on target**.
- `DatasetProvisioningError(ResidueError)` — chained, names absolute path **and node**.
- **The `.json` extension is load-bearing**: `.gitignore:41` keeps published artifacts out of
  the clean-tree check at `run_finalizer.py:1589`. **No sidecars** — the digest lives inside the
  manifest. Do not name it `*_config.json` / `schema_*.json`.
- **A local single-GPU run now refuses while any rig is down** — the fail-closed reading;
  a bypass needs governing.
- `dataset_provisioning.json` is gitignored — a fresh clone has no fleet definition.
- **A miner-backed run HARD-FAILS** on a **missing, unreadable, invalid or empty** provisioning
  manifest, before any coordinator construction or dispatch (`8600e75`). Unreadable and invalid
  are decided **in the loader** — a manifest nobody can read establishes nothing for anyone —
  and only *absent* is a caller decision.
- **Status vocabulary, and it is load-bearing:**
  **`UNAVAILABLE`** = a required verification was **attempted and could not complete** → fatal
  for a miner topology. **`NOT_APPLICABLE`** = this path never needed the check → proceed.
  *"We needed it and could not get it" is not "we did not need it."* An **unknown**
  `remote_execution` keeps the over-constrained reading (`UNAVAILABLE`), never the clean one.
- **`remote_execution=False` is a topology statement, NOT a bypass.** It must never become
  Beta's Q1 refinement by the back door: a local run that still drives the **full-fleet**
  coordinator **performs remote execution** and must not declare otherwise. *(v12 wrote "26-GPU"
  here; the fleet is **25** — §2.17. The count was never the point, so it is now stated as a
  property, not a number.)*

### 2.10b Dataset lifecycle (TB rulings 2026-07-30/31)
- Midday and evening use **independently selected equipment** — **no evidentiary basis for
  advancing one PRNG state through interleaved records.** Ordering is normative **within a
  session stream**; combined-container order carries **no PRNG-advance meaning**. The
  chronological-reorder migration was **cancelled**. Combined-session sequential sieve is
  **non-certifying, prohibited by default**; production re-optimization is **per-session**.
- Scraper moves to **append-only immutable versioned files** + an **atomic pointer manifest**
  (not a bare symlink). Version IDs need UTC timestamp **and** content identity.
- **Two walls:** *publication prefix* (history not rewritten — a **record-sequence** check; a
  byte-prefix test is invalid for JSON arrays) and *accumulator input* (**exact
  input-manifest digest match**). **Append-only does NOT make prior scores valid on the next
  version** — adding a draw changes windows, eligibility, gap/skip features, global frequency,
  normalization, any "latest N". Prefix-only merging **not approved**.
- Corrections → **new dataset lineage**, old preserved; scraper stops with
  `CORRECTION_REQUIRED` and may not create the corrected lineage autonomously.
- Generations **chain** (the finalizer merges prior rows) — input identity is a lineage
  invariant, not annotation.

---

### 2.11 Fleet authority — six mechanisms, no single definition (TB fleet ruling)

**There is no single required fleet state today.** Six checks at three granularities on two
address sets; which apply depends on the backend flag and whether the run came via WATCHER or
the CLI. **Three** point at bare metal; P0.5 points at the CT100s; two name no fixed set. The rigs
are booted into Proxmox, so **P0.5 passes and the three bare-metal checks structurally cannot** —
P0.5 is the only mechanism updated for the migration.

> **The six mechanisms in full → `docs/FLEET_STATE_REQUIREMENTS_v1.md` §0 and §5.1.** That table is
> **stronger than the one v12 carried here**: it adds a **blocks?** column and a `file:line` anchor
> per mechanism (`dataset_authority.py:904` · `coordinator.py:502` ·
> `persistent_worker_coordinator.py:864` · `preflight_check.py:293` · `cluster_boot_notify.sh:9-14` ·
> `range_miner_coordinator.py:3715`), plus source-of-truth, trigger and applies-to-which-backend
> rows. **§5.3 and §5.4** carry the divergence analysis and the failure modes **no** mechanism
> covers — including **a dead GPU on a miner run**, which nothing catches.
>
> ⚠ **That document predates the ruling below and does not contain it.** It is the analysis Beta
> ruled *on*. Everything from here down is the ruling and is carried here because the target is
> silent on it.

**⚠ MECHANISM 3 — CORRECTED 2026-08-04. The PWC ready gate does NOT hold at 24, and has not since
2026-07-07.** Both that document's §2.2 and earlier revisions of this section asserted
*"`23 < 24` → `RuntimeError`, run refused."* **That is false at HEAD.** The chain reaches
`coordinator.pwc_min_workers = 24` (`window_optimizer.py:774`) and then **stops**: the line
carrying it into `run_trial_persistent` — `min_workers = getattr(coordinator, 'pwc_min_workers', 1)`,
added by `ca06f8c` (05-08, *"S174: hard ready-gate (TB-approved)"*) — **was deleted by `2389b61`**
(§2.7b) and never restored. `run_trial_persistent` falls back to its own default `min_workers: int = 1`
(`persistent_worker_coordinator.py:1649` → `:1685` → `:268`), so the gate passes at **one** ready
worker and logs `READY GATE PASSED`. **The coordinator-side S174 gate is fully intact; only the
threshold reaching it is wrong.** `docs/FLEET_STATE_REQUIREMENTS_v1.md` §2.2 now carries the
correction in full, with the original analysis retained and marked superseded.

**⚠ It is NOT being restored — owner ruling, 2026-08-04.** The guard existed to confirm **the whole
cluster was being utilised**, in the PWC SSH/TCP era when a crashed worker's share was **picked up
by the remaining workers**, so a run could proceed short-handed and merely take longer. **It was a
UTILISATION check, not a correctness gate.** **RANGE-MINER does not have that shape** — stripes are
claimed per worker against a ledger, not redistributed by slack-picking — so the failure mode the
guard was written against **does not exist on the certifying path**, and PWC is retired from
certifying authority (§0.7). **Not a defect, not restored, not a Phase-7 blocker.** *Do not
re-propose restoring it; do not cite the `24` as an effective threshold.*

**Beta's ruling: none of the six defines the fleet.** The future sole authority is a **frozen,
run-scoped Resolved Execution Set**, created after backend and rig-profile selection but
**before** dataset verification, GPU verification, coordinator construction or dispatch. It
carries backend · rig profile · logical nodes and endpoints · worker/GPU identities ·
local-vs-remote · admission count · dataset-verification targets. **WATCHER and the CLI invoke
the same resolver.** All six become **consumers**. **A partial set must be explicit and frozen
before the run — never inferred from which workers happened to answer**, and unknown miner
workers must not become eligible merely because they connected.

**Both topologies are retained** (TB ruling 3): `.120/.154/.162` is the deliberate bare-metal
profile, `.122/.156/.164` the Proxmox compute endpoints. **The selected profile decides which
endpoints enter the set.** `distributed_config.json`'s bare-metal addresses remain
deliberate — see §4.

**Q1 — local runs.** A local single-GPU run currently **refuses while any rig is down**. The
refinement (verify only the resolved execution set) is approved **in principle** but must come
through the shared resolver, **not** by special-casing P0.5 or weakening `require_fleet`. Until
then the over-constrained behaviour stands.

### 2.12 Admission liveness — the §4.3 hang, repaired `ee0db06`

**The defect:** `assign_stripes`, `_dispatch_pending`, `process_lease_expiry` **and** the stage
advance were all behind one guard, `len(eligible) >= expected_workers` — and `serve_timeout` is
`None` by design. A worker loss crossing the threshold stopped lease expiry from being
processed, so dead workers' stripes stayed `claimed` with expired leases nobody looked at:
**the trial neither completed nor failed.** *The Blocker-3 matrix was unreachable in exactly
the situation it exists for.*

**The repair separates admission from maintenance:**
- **ADMISSION (bounded)** — reaching `expected_workers` is a precondition for *assigning* a
  stage, bounded by **`worker_admission_timeout`, default 180 s** (the PWC readiness window).
  Failure is an explicit `fail_trial` naming run · stage · family · phase · expected · admitted
  · elapsed. **The window re-arms only when `stage_idx` changes**, so worker churn cannot
  extend it.
- **MAINTENANCE (unbounded)** — once a stage is assigned, dispatch, lease expiry and completion
  evaluation run **regardless of the current eligible count**. A shrunken or empty pool is a
  legitimate input the matrix already handles.

**`serve_timeout` stays `None`** — a multi-billion-seed scan exceeds any wall clock, and the
bounded clock is on *admission only*. `expected_workers` is **not** reduced dynamically;
`worker_pool_size` semantics and the Blocker-3 matrix are **unchanged** — the matrix is now
*reachable*, not rewritten.

### 2.12b Admission binding and the freeze retraction (`eff6616`)

**`expected_workers` now comes from the frozen set.** `miner/range_miner_coordinator.py:3693`
→ `_execution_set_expected_workers` → `execution_set.admission_expectation`, returning the
frozen set's **effective** `admission_count`. `context["worker_pool_size"]` keeps its meaning
and is now **the REQUEST**, not a second answer. With no set frozen the context value is
returned unchanged, which is why every pre-existing loopback gate stayed green.

```
admission_count = min( requested worker pool size, count of selected worker identities )
```

**Both counts are recorded and both are in `set_id`**, so a run that asked 8 and was clamped to
2 has a different identity from one that asked 2. *A clamp that overwrites the request is a
clamp nobody can audit.* Cases: 26-GPU/8→8 · local/8→**2** · local/explicit 1→1 · zero,
negative or zero-capacity → **fail at resolution**.

**The defect it closed:** a local two-GPU set waited for the default eight — **six of which the
set itself declared could never connect**, because a worker outside the set is refused
admission. The trial spent its whole 180 s window failing to meet a threshold that was
**unmeetable by construction.**

**RETRACTION — the freeze-after-read property was FALSE as first implemented.** Alpha claimed
the ordering requirement could not be violated. Beta traced it: `active_execution_set()`
incremented `_READS` **only inside `if _ACTIVE is not None`**, so a consumer could read `None`,
take the legacy path, and a freeze could still follow — *the exact sequence claimed impossible*.
**The counter is now unconditional.** *The empty read is not the harmless case; it is THE case
that matters, because a consumer that read `None` behaved as though no fleet authority
existed.* A private `_peek_execution_set()` serves the **resolver owner only** (AST-asserted),
because counting the owner's own check would make it trip the guard it exists to enforce.

### 2.13 Control chains, end to end
*Which knobs actually reach execution. This table exists so a wiring gap is found now, not at
Chapter 13.* **Every row is governed** — none is a new finding.

> **Kept in full deliberately.** `PIPELINE_BEHAVIOUR_MODEL.md` §13.2 lists the same chains but
> **collapses emit→validate→apply→execute into a single state column**, and **omits two rows** —
> *miner kernels → independent reference* and *Advisor → `search_strategy`*. The per-stage marks are
> the point: they say **where** a chain dies. §13.2 is the WHY companion; this is the wiring map.

| chain | emit → validate → apply → execute | state |
|---|---|---|
| per-direction thresholds → kernel | ✅ ✅ ✅ ✅ | **WORKS** (D6 + `8a55a68`) |
| dataset identity → all nodes | ✅ ✅ ✅ ✅ | **WORKS** (P0.5 + Q2 closure) |
| fleet definition → the run | ✅ ✅ ✅ ✅ | **WORKS** — Resolved Execution Set, 34/34 (`63e627f`); admission bound `eff6616` |
| worker loss → failure matrix | ✅ ✅ ✅ ✅ | **WORKS** (`ee0db06`) — was an unbounded hang |
| Advisor → selfplay `max_episodes`, `min_fitness_threshold` | ✅ ✅ ✅ ✅ | **WORKS** |
| Optuna `skip_min`/`skip_max` → hybrid kernel | ✅ ✅ ✅ ✗ | dies at `_hybrid_prefix`. **The approved skip-OUTPUT work does NOT fix this** — see §8 |
| miner kernels → independent reference | ✅ ✅ ✅ ✅ | **WORKS** — transfer gate, 8/8 populations exact-set equal (`d98298c`) |
| Optuna `offset` → forward hybrid | ✅ ✅ ✅ ✗ | dies in kernel args |
| `skip_learning_rate` → kernel | ✅ — — ✗ | kernel hard-adapts at 1.0 |
| Advisor → `search_strategy` | ✅ partial ✗ — | dies in the override dict. **Autonomous application NOT approved** |
| Advisor → `strategy_recommendation.json` → WATCHER | ✅ ✅ ✗ — | **no code reads the file**; the working path is in-memory |
| diagnostics → Step-5 retry params | ✅ ✅ ✅ ✗ | filtered at the step boundary — **deliberate**; reporting fixed `f8b751c` |
| Ch13 proposal → acceptance | ✅ ✅ ✗ — | `pending_approval` is a **valid authority boundary** |
| GPU `skip_sequences` → ML features | ✅ — ✗ — | discarded in `extract_survivor_records` (`…final.py:125`); kills 3 features |

**Reserved authority (human only):** feature engineering · survivor thresholds · sieve
strategy/mathematics · window-optimizer logic · PRNG-family authority · scoring logic ·
meta-optimizer search space · model families · policy authority.

### 2.14 The dataset, MEASURED — invisible to every repo-scoped search

`daily3.json` is gitignored, so **no clone-based audit can see any of this.** Measured on VM101,
2026-08-02.

| fact | value |
|---|---|
| records | **18,068** |
| span | `2000-01-01 evening` → **`2026-02-26 midday`** |
| session values | **exactly** `{evening, midday}` |
| canonical order | **`(date ascending, session: evening BEFORE midday)`** — the stored file **MATCHES** it |
| `2026-02-26` | **midday ONLY — evening ABSENT.** The dataset ends mid-day |
| single-session dates | **1,040 of 9,554** — **1,038 are 2000-2002** (the evening-only era, before CA Daily 3 had a midday draw); 2019: **1**; 2026: **1** |
| staleness | last record `2026-02-26`; the scraper has not run since |

**Three rules this falsifies, two of them already written by Alpha:**
1. **"Both sessions required" completeness is WRONG** — it rejects 1,038 legitimate 2000-2002
   evening-only dates. Those dates are complete.
2. **"The terminal date carries a midday record" is WRONG** — under the bound order **every
   complete date ends with its midday record**, so such a rule defers every complete terminal date
   **forever**. *(Alpha shipped exactly this in 6-P2 REV3; Beta caught it.)* The correct form is a
   post-dedup **session-set** test: `{midday}` defers · `{evening, midday}` and `{evening}` publish
   · anything else fails validation.
3. **`2019-01-25` is evening-only in the modern era** — an anomaly, not a pattern. BACKLOG.

### 2.15 Three-hop parameter route — a new Step-1 parameter dies silently at hop 1

| # | hop | anchor |
|---|---|---|
| 1 | `agent_manifests/window_optimizer.json` → `default_params` (+ `args_map`, `param_docs`) | **WATCHER's step-scoped filter DROPS any key not declared** — `agents/watcher_agent.py:1290-1314`, `if key in declared` |
| 2 | explicit kwarg at the call site | `window_optimizer.py:790-810` |
| 3 | the method signature | `window_optimizer_integration_final.py:1695-1710` |

Adding only hop 3 gives a parameter that exists, accepts a value and **never receives one from
production** — the `Advisor → strategy_recommendation.json → WATCHER` dead-chain shape and the TRSE
F1 manifest drift, in a third place. **Gate the route, not the parameter.**

### 2.16 The record's `trial_number` is NOT `optuna_trial.number`

`trial_counter = {'count': 0}` (`window_optimizer_integration_final.py:2361`) → `+= 1` (`:2382`) →
`trial_number=trial_counter['count']` (`:2399`). A **process-local 1-based ordinal that restarts
every run.** `optuna_trial.number` is study-scoped, 0-based, and reaches only partition routing
(`:2384`) and `result.iteration`.

**They are different quantities, and `trial_number` is part of the replay key
`(seed, trial_number, skip_mode)`.** A guard placed on `trial.number` does **not** close a
replay-key collision. *(Beta's D6.2 addendum §4 was aimed at the wrong counter; D6.2 implements
those checks as specified* **and** *continues the record ordinal from the recovered maximum, which
is what actually closes it.)* `study.enqueue_trial` (`window_optimizer_bayesian.py:725`) is the
S166 warm-start path, so a resumed study really can carry trials numbered below a recovered
maximum.

### 2.17 Fleet state as launched — Phase 7

> **⚠ OPERATIONAL STATE LIVES IN `docs/PHASE6_PREREQS.md` REV5, NOT HERE.** The seven-item
> checklist, its per-item evidence, the gate matrix and the redeploy measurements are all there
> (REV5 items 1–7; §"Code/environment parity"; §"Where these surface"). **It is dated and it
> expires.** What stays below is the part REV5 does **not** carry, plus the durable lessons.

**⚠ CORRECTED — "all operational prerequisites are closed" is WRONG, and v12 said it.** REV5
retracts REV4's phrasing per TB: **the seven checklist items are CLOSED OR OWNER-WAIVED**, which is
*not* the same claim. **Host kernel-log observability was never one of the seven** — which is
precisely how a seven-item sweep reads as complete while a real observability gap stands.
*(Item 1, the second 3080Ti, is **waived by the owner**, not closed: 25 GPUs is owner-mandated.)*

**The frozen execution set — NOT in REV5, which predates the fix and still recommends it:**
```
set_id                    = bea580e764905a0d9485d2688be5841cc95f16e16837c23aced1f634d97f67a8
worker_identity_count     = 25   requested = 25   admission = 25   clamped = False
```
**25 by construction, not by clamp.** `localhost.gpu_count` was corrected **2 → 1** (`f255912`) —
it declared two cards from the old configuration, so the set carried **26 identities admitting 25**,
which *read* like a shortfall. **No execution consequence** (workers launch with explicit
`--gpu-id N`; nothing iterates `gpu_count`). **The bare-metal addresses in that file remain
untouched** (CLAUDE.md §3).
**⚠ Do not stop at "no execution consequence" — REV5 still states it as "the cost is auditability,
not execution", and Beta rejected exactly that framing.** See the closing paragraph of this section.

**⚠ Item 6 is the trap, and the clock is only half of it — the DURABLE part.** Code parity was
never measured until 2026-08-02, and it **failed**. The measurement is in REV5; the lesson is here:
**the rigs are deployment targets, not working copies.** Deployment is `git clone` once, then
targeted `scp` from VM101 (`REMOTE_NODE_SETUP_CHECKLIST.md:127,133,139`) — `rrig6600` carries a
worktree at `8e2f5bf` with 84 dirty entries, and the other two **have no git repository at all.**
**Digest comparison, never `git rev-parse`, is the parity evidence.** Stale modules sat inside the
worker's **executing import closure** (`miner/__init__.py:19`), so *"loaded but not driven"* is not
an acceptance criterion — verify by importing on the target and reading `sys.modules`, and have each
probe print its own `socket.gethostname()` so three machines cannot be one machine answering thrice.

**⚠ Kernel-log observability is NOT established, and the soak launched without it.** CT100 is an
unprivileged LXC — **GPU kernel messages are only visible from the Proxmox hosts** `.121`/`.155`/
`.163`, and **VM101 has no root key auth to them.** Consequence, owner-authorized and
Beta-acknowledged:

> **The "no `GCVM_L2_PROTECTION_FAULT` / no GPU reset" criterion reports `UNAVAILABLE`, NEVER
> `PASS`.** It was not checked. **An inaccessible surface is not a clean one** (VIR-1).

**Substitute detection**, polled and logged as a series: `rocm-smi` device count and per-GPU state
per rig · worker process liveness · repeated lease expiries per identity. These detect **that** a
GPU or worker died on a named rig; they **cannot classify** the fault. Classification comes
afterwards from the Proxmox console, which retains the logs.
*(The binding verbatim TB report language for this exception is in REV5 §"Where these surface" —
use it word for word; do not paraphrase a governance sentence.)*

**Why the risk is judged low:** `GCVM_L2_PROTECTION_FAULT` was a **PWC launch-storm defect**
(~17K kernel launches/trial) that followed the workload across every transport, a code revert and a
package rollback. **RANGE-MINER's persistent per-GPU daemons remove that workload**, and Phase 6.0
produced no reset and no fault on CUDA or ROCm. **The one qualification: PWC was also stable below
saturation and failed ONLY at full-fleet saturation — which is the condition this soak is the first
to meet.**

**Beta's correction to Alpha on the execution set, worth carrying:** Alpha framed the 26-identity
set as an *auditability* problem. **It was not.** Naming 26 eligible identities and letting the
answering population determine which 25 satisfy the threshold **is not an explicit 25-worker set**,
even when the threshold is meetable. *"No execution consequence"* sidestepped the contract.

### 2.17b Chapter status, and the three-lane CRT result

| chapter | state |
|---|---|
| **1** Window Optimizer | audited — **9 of 41 claims accurate** |
| **2** Bidirectional Sieve | **destroyed** at `248e48c` (a 34-line fragment over 709 lines), restored from `d14dcdd`, audited, **extended to 1,463 lines**, closed `ef4b1c6` + content gate `09bbfbf`. **Now the strongest of the three.** |
| **3** Scorer Meta-Optimizer (**Step 2.5, NOT Step 3**) | audited `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` — **55 claims: 17 accurate · 9 stale · 24 false · 5 unverifiable**; §8/§9/§14.2 describe code **deleted at v4.0** |
| **5, 6, 8, 13** | **UNAUDITED** |

**⚠ Chapter numbers are not step numbers.** `STEP_MANIFESTS[2] = "scorer_meta.json"`
(`agents/watcher_agent.py:401`) — WATCHER's **step 2 is the scorer meta-optimizer**, step 3 is
full scoring, and **the bidirectional sieve runs inside Step 1** (`run_bidirectional_test` in
`window_optimizer_integration_final.py`). The conceptual scheme where sieve = 2 and scorer = 2.5
also exists. **Both are in use and they conflict.**

**The three-lane CRT test is DOCUMENTED — `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §6.** It is live
in every kernel (`prng_registry.py:984-986`, `:1042-1044`, `:3146-3148`) as
`(output % 1000) && (output % 8) && (output % 125)`.

**And §6 proves it is EXACTLY EQUIVALENT to `% 1000`.** Because 1000 = 8 × 125 with
gcd(8, 125) = 1, CRT gives agreement mod 1000 ⟺ agreement mod 8 **and** mod 125. Verified two ways:
the CRT argument, and an **exhaustive check over x ∈ [0, 4000) × d ∈ [0, 1000) — zero divergent
cases.** **It is not extra filtering power**, and §6 says the original emphasis was misleading.
`prng_registry.py:773` is the one kernel that does not run it.

### 2.17c Catalog findings — corrections and threads neither party was tracking

From `docs/PROJECT_FILE_CATALOG.md` (`1fc05bb`), which indexed 562 files in one pass.

**⚠ "PHASE 7" IS OVERLOADED — two unrelated milestones share the name.**

| | |
|---|---|
| **Phase 7 (WATCHER)** | dispatch integration, **Feb 2026**, marked **COMPLETE** in Chapters 10, 12, 13 and `TODO_PHASE7_WATCHER_INTEGRATION_REVISED_v3.md` |
| **S172 Phase 7** | the **25-GPU saturation + WATCHER soak** — this project's current milestone |

`SOAK_TEST_PLAN_PHASE7_v1_0.md` (2026-02-03) belongs to the **first**. Its prerequisite is
*"Phase 7 D5 End-to-End (PASSED Session 59)"* and its tests are daemon endurance, queue integrity
and the autonomous loop. **It is NOT the S172 soak plan, and there is no S172-scoped soak plan
other than the brief written for it.**

**Step-map divergences → `PROJECT_FILE_CATALOG.md` §5.1**, which carries all three numbered and with
its own scope caveat: **Step 2**'s manifest-vs-script divergence (*"how a soak hazard reached launch
day"*), **Step 3**'s identical shape (`STEP_SCRIPTS[3] = run_step3_full_scoring.sh` vs three manifest
actions naming none of it), and **Steps 0 and 5 declaring no `actions` at all**, so the two maps
cannot be compared for them. **Retained here because §5.1 does not say it:
`run_step3_full_scoring.sh` has NOT been examined** — structural fact only, not diagnosed.
`trse.json`'s `skip_on_fail: true` carries a stated reason — **that silent-failure behaviour is
ARCHITECTED** (`TRSE_INTEGRATION_PLAN_S121.md` §2C), not a defect.

**The KPI governance chain S176 → S177 → S178 → S179 exists and neither Alpha nor Michael was
tracking it.** **`TB_RULING_S179_IMPLEMENTATION_AUTH.md` is the LIVE AUTHORITY.** Full chain with
line counts, verdicts and the follow-up brief per ruling → **catalog §1.1** (rows `TB_RULING_S176`…
`S179`); **whether the implementation landed → catalog §7 gap 8**, which states it correctly:
*"This is not a claim that it did not land."* Same shape as D3.0-B — a governed requirement whose
completion nobody has checked. **Does not touch a step-1-confined soak.**

**The v4.0 objective was a tautology.** `TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md`: the smoke test
returned **WSI = 0.9997 on trial 1** because the formula's dominant term (w3 ≈ 0.82) was
`quality = fwd*rev` — **the objective measured itself.** Fixed at v4.1; v4.1 then could not
optimise either (`sel_score = 0.0000` on every passing trial, `bidirectional_selectivity` at floor
98.8%), which is the v4.2 ruling. **Lineage: v4.0 → v4.1 → v4.2 → v4.3 → v4.4, all governed.**

**⚠ The v1 catalog's Runtime Data table was wrong in EVERY ROW** — the durable lesson, and the
reason it is still named here. Measurements → **catalog §6.2**, marked ★ **DO-NOT-CARRY-FORWARD**,
which carries all four rows this section listed plus a fifth (manifest count) and the general rule:
**runtime-artifact sizes are not catalogue facts.** *A confidently-stated table in an
authoritative-looking document survived from February to August because nobody measured it.*

### 2.18 D3.0-B — OPEN, and it NARROWS what Phase 6 certified (TB ruling 2026-08-02)

`PHASE6_PREREQS.md` REV3 stated D3.0-B *"must complete before Phase 6 certification."* **No commit
completes it, and the defect is live:**

```python
convert_survivors_to_binary.py:184
encode_prng_type(s.get('prng_type', s.get('prng_base', 'java_lcg')))
```

A record with **neither** `prng_type` **nor** `prng_base` is **fabricated as `'java_lcg'`** instead
of failing closed — **while the canonical resolver already provides the fail-closed behaviour.**

**Beta's ruling: OPEN and REQUIRES COMPLETION.** *Waived* and *superseded* were **rejected** —
REV3 made it mandatory, the defect remains executable, divergent encoding tables persist in
dormant-but-executable writers **and patch scripts**, and no ruling ever removed the prerequisite.
**Beta recorded its own Phase-6 certification as a governance error for omitting it.**

**⚠ THE CERTIFICATION SCOPE IS NARROWER THAN "PHASE 6 IS CERTIFIED":**

> **Phase 6 is certified for the demonstrated miner/finalizer path.** Wall A used the miner
> coordinator, Phase-5 assembly, the D3.5 finalizer, direct 22-array validation and Step-2/Step-3
> consumption — **it never invoked `convert_survivors_to_binary.py`.**
> **Legacy conversion and dormant legacy-writer surfaces are UNCERTIFIED.**

**DO NOT INVOKE THE LEGACY CONVERTER UNTIL D3.0-B CLOSES.** No Wall A/B rerun is required.

**Bounded scope when it is done:** canonical fail-closed resolver replacing missing-identity
defaults · preserve valid `prng_type` precedence and valid `prng_base` fallback · reject records
carrying neither · **remove or hard-retire divergent executable encoding tables, including
rerunnable patch scripts that could reinstall them** · behavioural gates and mutants for missing
identity, unknown identity, and reintroduced `java_lcg` defaulting.

**Does NOT block the miner-backed Phase-7 soak** (the soak does not invoke the legacy writer), and
6-P2 remains independent.

### 2.19 S172-BP staging back-pressure — law, mechanism, and the three-ruling F1 arc

**Beta ruling 2026-08-05 (binding), remediated at `4b1aad6`:**
- **Classification law (Beta D):** a coordinator staging capacity condition is a WAITING /
  INFRASTRUCTURE state, never a worker-stripe fault — it may never enter
  `_handle_stripe_failure_locked`, consume a retry, or fail a trial as a worker failure.
  Permitted terminals: direct `fail_trial` with reasons leading
  `coordinator_staging_capacity_timeout:` (bounded wait, default 600 s, latched, snapshot-
  attributed), `coordinator_staging_capacity_invariant:` (post-sizing overflow, names WHICH
  bound tripped: derived-count / operator-override-count / retained-bytes), or
  `coordinator_staging_sizing:` (derivation failure fails CLOSED — never the smaller
  on-demand fallback).
- **The constant 64 is DELETED.** Bound = `staging_burst_bound_conservative` (Σ per slot of
  max-over-eligible-workers ceil(span/cap)) + margin (= live connections). **116 vs 136 is
  Beta-mandated:** 116 = exact recorded 2026-08-05 assignment (34+14+34+34); 136 =
  conservative four-slot AMD-cap bound. Both logged per stage (`[S172-BP] burst_exact`).
  `staging_deferred_max` survives only as operator override (below-derived WARNs).
- **Pause/resume lives in `_conn_reader_loop`, per connection.** Only `sub_stripe_result`
  gated; ≤1 decoded envelope, in the reader's local; later frames stay on the ordered TCP
  wire (worker `_sendall` has no socket timeout — a full buffer parks its mining thread
  harmlessly). Resume trigger: `_pump_deferred`'s `finally`.
- **Resume credit (F1, three rulings deep — the file to read is the LAST ruling, not the
  first):** one capacity-release event grants at most ONE wake (FIFO-oldest unsignaled);
  the wake RESERVES the observation; only the FIFO head may self-resume via the defensive
  poll and only when no credit is outstanding. The reservation ends at serve-side
  DISPOSITION (admission / deferred / fenced / terminated — via the `dispatch_inbound_result`
  seam wrapping an unchanged `_serve_dispatch`), **never at `inbound.put`** (round-1 defect),
  **matched by exact per-grant `credit_id`, never by socket identity** (round-2 defect
  F1-R2a: an older uncredited same-socket result cleared it), **with the one-result-per-
  reservation barrier BEFORE `recv_msg`, not after** (round-2 defect F1-R2b: a post-decode
  wait left the holder owning two decoded envelopes, breaking the one-envelope bound the
  margin derives from).
- **Lease exemption + resume grace (Beta 6.5: FULLY RATIFIED):** heartbeats are the only
  lease-renewal path and share the ordered stream with results; exemption covers
  coordinator-initiated pause AND a bounded post-resume grace until the first accepted
  heartbeat (`renew_lease`'s own boolean gates the clear); genuine silence still expires.
- **E is REJECTED:** never fix a coordinator capacity defect via seed caps or stripe
  geometry. **Gate 56** of the phase-4 suite changed disposition under D (bound-proof
  retained); its old assertion text is SUPERSEDED — do not cite it.
- **Terminal summary must never raise** (Alpha guard, Beta-approved + gated
  G-SUMMARY-NO-MASK): `bound_in_force` degrades to `None` + `bound_in_force_error` rather
  than masking the primary terminal reason.

**Governance record:** `4b1aad6` was committed and dual-pushed AHEAD of Beta's gate review
at the owner's direction — recorded in the commit message, disclosed in the submission
cover (`42bdbb1`), accepted by Beta as an owner-directed sequence deviation, no rollback,
fix-forward on the hash. Precedent: disclosure-up-front is the accepted handling; the
LAUNCH hold is the operative one and was never breached. Beta ratified: the
gate-per-decoded-frame reading of B.1 ("one held envelope"); the `conn_dropped` clear
(disposition (iv) — necessary because READ-DEADLINE drops emit no eof tuple); the
`dispatch_inbound_result` seam; the G3/G5/G6 bench resequencing (assertion content proven
byte-identical programmatically).

### 2.20 WHAT THE SYSTEM IS AND HOW THE PARTS CONNECT — read this before reasoning about intent

**The authoritative statement** (`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:158-167`):
*"Bidirectional sieving provides exponential noise suppression. Loose thresholds are not a
weakness — they are a mathematical necessity to expose a learnable structure. ML does not guess.
It refines a space already reduced from 2³² to 10⁴."* And the naming rule
(`PIPELINE_BEHAVIOUR_MODEL.md:1094-1095`): **TFM = functional mimicry of PRNG surface output. NOT
seed recovery, NOT state reconstruction.**

**One paragraph:** a GPU sieve scores every candidate seed in a configured range twice — against
the observed window in order, and reversed — keeping seeds that clear both thresholds. Because
the thresholds are deliberately loose, what survives is a **manifold** of near-consistent
candidates, not a shortlist. Each survivor's generator is replayed and characterised into a
feature vector; a model ranks those vectors against a quality label computed on data the sieve
never saw; the top-ranked survivors each vote **their generator's next value** into a prediction
pool. The claim is not that the generator was identified — it is that survivorship plus ranking
beats the k/1000 baseline (`evaluate_pools.py:36-40`).

**THERE ARE TWO LEARNERS. DO NOT COLLAPSE THEM.** (Owner correction 2026-08-08; the
`window_size`/`offset` finding below applies ONLY to the second.)

| learner | stage | learns over | uses window_size / offset? |
|---|---|---|---|
| **Optuna TPE** | Step 1 | window_size, offset, skip range, sessions, thresholds | **YES — these ARE its search dimensions** |
| **Supervised model** | Steps 4–5 | 89 survivor-derived features | **NO** — see below |

- **`window_size` and `offset` do NOT reach the supervised ML; `skip_min`/`skip_max`/`skip_range`
  DO** (features #75/#77/#78 of 89). They are in the 22-array NPZ (arrays 4, 5) but die at the
  merge site: `survivor_scorer.py:774-779` pulls exactly **18 named fields** and neither is among
  them. Verified four ways (NPZ contract, merge list, trained sidecar `feature_count=89` hash
  `7733d30a913545ca`, live artifact) plus Alpha's own read of `:774-779`.
- **`best_offset` (feature #26) is NOT the Optuna offset — it is hardcoded `0.0`**
  (`S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:128`). This is what makes the wrong intuition
  feel confirmed.
- **Skip still influences predictions — through a different route**: sequence regeneration at
  Step 6 (`prediction_generator.py:845`), not the feature vector.
- **Six of the 89 trained features are constant** — the five documented dead placeholders PLUS
  `best_offset`. The "5 dead" figure in circulation undercounts.
- **Training target is `holdout_quality`** — a 50/30/20 composite of CRT match quality,
  distributional coherence and temporal stability (`holdout_quality.py:61-90`), validated by
  K-fold (k=5) plus a structurally separate temporal holdout. **NOT `holdout_hits`** — Chapter 6
  still says otherwise and is stale.
- **A survivor is a scored candidate, not a verdict** (`CHAPTER_2:373-377`). The population is a
  manifold BY DESIGN — near-misses are admitted deliberately, because a population of exactly one
  has no variance and therefore no learning signal (whitepaper §7). **"Most likely seeds"
  describes Step 5's output, not Step 1's.**
- **Data spine, hop 7 — no chapter walks it.** A seed's *relevance* is its `predicted_quality`
  under the model; its *contribution* is literally **the next value its own generator emits**,
  regenerated with **that seed's own skip** (`prediction_generator.py:845-848`, `:530-553`:
  *"Each survivor may have been discovered with a DIFFERENT skip value"*). Relevance decides rank;
  the generator decides what it votes for. This is the only place the skip hypothesis does causal
  work downstream.
- **Pool grading** (`chapter_13_orchestrator.py:306-316`): `0.50·hit@20 + 0.30·hit@100 +
  0.15·hit@300 + 0.05·pool_coverage`. **Unadjudicated mismatch:** this objective weights three
  pool sizes while the live generator emits one.

**Step 1's purpose, established 2026-08-08** (`CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md`):
the earliest authoritative source (`instructions.txt:4368-4371`) states *"The window optimizer is
NOT a sieve itself — it's a meta-tool that finds the BEST window parameters."* Primary deliverable
is the alignment config; survivors are evidence an alignment resolves. The sweep is **global per
`prng_type`, never per-draw** — no source anywhere describes per-draw seed discovery. The PWC
transport pivots (SSH→TCP→ZMQ) were **purely mechanical**; the goal never moved. A regime-shift
rerun **exists and is human-gated**, but enters Step 1 through the same S140 coverage tracker, so
it **sweeps the next uncovered block rather than re-examining the range that produced the current
survivors** — this is where the owner's mental model and the implementation part company.

**The sieve's physical model, established 2026-08-08**
(`CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md`): the kernel (`prng_registry.py:968-980`) assumes
**ONE unbroken 48-bit LCG trajectory** — `state = seed & m`, advance `offset`, advance `skip`,
match. **No machine identity, no A/B RNG branch, no reseed, no date or session input in the draw
loop.** Every real-world gap (pre-test draws, other games, power-cycles) is modelled as *"advance
N more steps."* Per the CA draw procedures the machine is selected **per draw SESSION** by an RNG program
(§II; corrected 2026-08-08 — Alpha's §2.20 and §7 previously said "per draw," which the document
does not support: the draw room is entered, used and re-sealed once per session, §V.8 selects one
"game set," and §VII.2 conducts it with one `[Run LIVE (#) Draw]` press; there are exactly TWO
automatic machines, §VII.5 NOTE 3). Each machine has **two RNGs**, and machines are **powered
down between sessions** — none of
which the model represents. **This is not automatically a defect:** the sieve is a candidate
FILTER, not a state-recovery attack, and selectivity (P(random) ≈ 10⁻⁵⁵⁰) is what carries the
weight. Midday↔evening crossing IS ruled and prohibited; whether one seed may span consecutive
days' power-cycles **within** one session stream — **no evidence found, anywhere**.
**S112 tension, unresolved:** `SESSION_CHANGELOG_20260226_S112.md:169-184` found real data
optimises at **W8** vs W256-1024 synthetic and concluded *"real-world lottery PRNGs operate in
short-lived regimes, not as one continuous seed stream"* — citing the draw procedures as evidence
— yet `window_size.max` remains **50** and no bound moved. **No bound's VALUE has a derivation
anywhere**: `skip_max = 250` (documented default 16, enumerated physical scenarios reach ~20, only
empirical figure S5-56); `offset.max = 100` has **no `_note` and no in-repo rationale** — the
`agent_manifests/window_optimizer.json` block declares `max: 2000` but has **no `args_map` entry
and no CLI route**, so it is inert, and its description *"Time offset from current draw position"*
is **wrong** (host code: head-relative index). Live bound is 100, from
`distributed_config.json` → `window_optimizer.py:74/143/166`.

### 2.21 THE 150-DRAW CONFOUND — read before citing ANY empirical result

**Every empirical result this project has ever produced came from `data[0:150]`.** `offset ≤ 100`
plus `window_size ≤ 50` caps the reachable filtered index at **149** of 18,068 records
(`miner/range_miner_worker.py:648-649`; `distributed_config.json` `search_bounds`). Governed and
already recorded: `DAILY3_CONSUMER_CONTRACT_v1.md` §4.3 — *"The production sieve analyses draws
from March 2000."*

**Consequence, and it is the load-bearing one:** selectivity spread, feature importance, survivor
counts, S112's W8 result, S107's flat-signal finding — all were measured on 2000-2003 draws,
because that is the only window the sieve can reach. **No conclusion drawn from historical trial
data can distinguish "property of the metric/system" from "property of that window."** Treat every
such number as confounded until the window can move.

- The first CA-procedures-governed record sits at filtered index **6,791** (midday) / **7,830**
  (evening); the document is effective 2021-06-09 and governs **3,447 of 18,068 records = 19.1%**
  (measured on VM101, 2026-08-08). The sieve can reach **none** of it.
- **`offset.max = 100` has NO derivation anywhere.** No `_note`, no rationale
  (`window_optimizer.py:74`/`:143`/`:166`). `agent_manifests/window_optimizer.json` declares
  `max: 2000` but has **no `args_map` entry and no CLI route** — inert — and its description
  *"Time offset from current draw position"* is **wrong** (host code: head-relative index from the
  OLDEST end). `config_manifests/parameter_registry.json:38-43` describes it as *"advance seeds by
  offset*(skip+1) before testing"* — **also not what any loader does.**
- **Raising it is NOT a config change.** Chapter 2 **F-4**: one scalar drives BOTH the history
  slice AND the generator pre-advance, *"coherent only at skip = 0"*. Setting offset to ~7,000
  would pre-advance the generator ~7,000 steps as a side effect of choosing which draws to look
  at. Chapter 2 rules F-4 belongs in the hybrid input-semantics design, not an arithmetic patch.
  **The work item is a window-anchor / generator-phase SEPARATION.**
- **Related open question, ruled architecture:** survivors are selected on evidence at index ≤149
  and then vote at `next_idx = len(lottery_history)` ≈ 18,068 (`prediction_generator.py:839`).
  The governing law is `DAILY3_CONSUMER_CONTRACT_v1.md:185` — `offset = train_history_len`,
  *"THIS IS THE LAW (per Team Beta)"*, i.e. the array index IS the generator advance count, so
  any insert/delete/dedup/re-sort silently invalidates `holdout_hits` → `holdout_quality` → the ML
  target. Whether a mimic fitted on a 21-draw window in 2000 still tracks ~18,000 advances later
  is **untested**.

### 2.22 TRIAL-LEVEL vs PER-SEED — a recurring category error, audited 2026-08-08

**`bidirectional_selectivity` is TRIAL-LEVEL, not per-seed.** Computed once per (trial × skip-mode)
as `len(forward_set) / max(len(reverse_set), 1)`
(`window_optimizer_integration_final.py:1783`, hybrid `:1887`), built into `metadata_base` once at
`:1762-1785` and dict-splatted into every survivor record at `:1801`. The code says so:
`:1759` — *"# Trial-level context (same for all seeds in this trial)"*. Only three fields are
re-read per seed, each marked `# v3.0: per-seed`. Miner path identical
(`utils/canonical_records.py:234`, loop `:238`).

**Already governed, NOT a new discovery** (Alpha reported it as a defect; it is status):
`STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md:501` marks it **TRIAL-AGG**, defined `:479-480` as
*"a trial-level scalar stamped identically onto every record of that trial+mode."* Skill §2.3
carries the same fact.

**Verified empirically 2026-08-08:** 5 of 5 held artifacts carry exactly **one distinct value** —
the certified release-grade generation (319 seeds, 1039.5718), two forensics NPZs at 20,949 and
20,916 seeds, and a fixture. Zero trial-groups contain more than one value.

**The S107 arithmetic closes exactly.** After the L2 merge (`run_finalizer.py:714-745`, one row per
seed) each row carries its **winning trial's** value, so `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`'s
98.8%-at-floor over 6,739 survivors measures **trial concentration**, not seed quality: one trial
won ~6,658 rows (`6739 × 0.988 ≈ 6658`), leaving the ~81 reported, and S107's own
`bidirectional_count` max of 6702 is that trial's intersection size.

**The recurring error, and it has a lineage:**
- **S103 (2026-02-21) diagnosed this exact class** — *"all quality fields identical for every seed
  from the same trial — zero signal for ML ranking"* — and fixed it for the match-rate fields.
- **S104 (2026-02-22)** restored seven trial-level statistics while documenting they are *"the
  same value for all seeds from the same trial."*
- **S107, the same day,** made one of them Step 2's **per-seed** quality signal.
- **v4.1's rationale was a category error** (`TB_RULING_REQUEST_STEP2_v4_1_OBJECTIVE.md:112-114`):
  *"min=1.01, max=2.47 … This is NOT flat — there is real variance to optimize against."* That
  spread is **between trials**; the objective needed variance **between seeds**.
- **The approved v4.2 replacement carries the identical error** — `bidirectional_count` is
  `len(both)`, the trial's intersection size, and died the same death
  (`scorer_trial_worker.py:414-415`: *"structurally dead: 79.2% of pool at bc>=11300"*).
- **Both v4.1 and v4.2 were RULED** (catalog §1.1 rows 61-62, `S107_session_log.md`, code comments
  `scorer_trial_worker.py:198`, `:413`) — **neither is an open request.**

**New defect found 2026-08-08:** `feature_registry.json` files `bidirectional_selectivity` under
**`/per_seed_features/`** — a per-seed declaration in the same entry that correctly defines it as a
ratio of trial-level counts. No prior record found.

**Standing rule:** before using ANY survivor-record field to rank or discriminate **seeds**, verify
it is per-seed. The trial-level fields are stamped identically and cannot. Conversely, trial-level
scalars are well-formed **for ranking trials** — a legitimate use that is currently unexploited,
but note that measuring across-trial spread from the historical study is **confounded by §2.21**:
every recorded trial ran inside `data[0:150]`.

### 2.23 CORRECTIONS TO THE 2026-08-07/08 ATTACK-PLAN ANALYSIS

The attack-plan report's Part A/B derivations **stand**; three of its negative conclusions were
**VOID** under the black-box framing and are recorded here so they are not re-inherited:

- **Derived `skip = 9`** (session-scoped, **order-invariant**): the daily inventory is 10 game-draws
  (midday pre-test + live = 2; evening pre-test + live {D3,D4,F5,DD} = 8), so between consecutive
  same-session Daily 3 values lie 9 burned outputs. Evening-only era variant: **7**. H-B variant:
  39. A chronologically combined stream is **not** constant-skip — it alternates `4+p` / `4-p`,
  which **strengthens** the existing prohibition on combined-session sieving.
- **A skip-pinned trial is a DIAGNOSTIC ARM, never a production config.** Alpha originally sold
  `skip_min=9, skip_max=10` on a *"~125× less kernel work per seed"* saving. **That framing was
  backwards:** under whitepaper §7 narrowing skip is a **manifold contraction** — fewer, less
  varied survivors, and variance is what the ML learns from.
- **VOID — "if each digit is its own selection the sieve can never succeed."** Under mimicry the
  predicate asks whether a candidate *emits the published number*; the machine's assembly process
  is not an operand. And the repo had already engaged H-B: `survivor_scorer.py:426-428` — *"Daily 3
  = three independent Z10 draws; score each digit position directly. Additive alongside CRT lanes"*
  (S119, spec `03:00-09r`; also `:616-617`, recorded `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:152`)
  — placed in the **feature vector**, where it adds information, rather than the sieve predicate,
  where it would narrow the manifold.
- **VOID — the two-machine mixture as a fatal objection.** `2^-(n-1)` computes P(the window came
  from one trajectory), but what decides whether the sieve returns survivors is P(∃ a
  parameterisation in the family matching at ≥τ) — only the first depends on how the data was
  produced. `CHAPTER_2:373-377` had already enumerated *"a partial match valid before a reseed
  event"* as a listed survivor category.
- **VOID — "`java_lcg` has no support in the document."** The family is a **substrate for mimicry**,
  chosen empirically, one of **44 `KERNEL_REGISTRY` entries** = 11 base families
  (`xorshift32, pcg32, lcg32, mt19937, xorshift64, java_lcg, minstd, xorshift128, xoshiro256pp,
  philox4x32, sfc64`) × {base, `_reverse`, `_hybrid`, `_hybrid_reverse`}
  (`prng_registry.py:3729`). If it stops fitting, another family is tried. The only real residue:
  if **no** family beats k/1000, that is a fitting failure, never an identification one.
- **WITHDRAWN — "manifold composition already measured."** No such measurement exists. 18 of the 22
  NPZ columns cannot produce one, and the fields that could (`forward_matches`/`reverse_matches`)
  are absent from the Step-3 merge list (§2.3).

**Three owner corrections that govern all of the above (2026-08-08, binding):**
1. **The target is a prediction-pool hit rate of ~65-85%**, not 100% and not a unique seed. No
   survivor must explain the whole series. "No single trajectory accounts for all observations" is
   not an objection to the method — it is a statement about pool composition.
2. **The generator family is a substrate, not a hypothesis about the machine.** We do not care
   which PRNG produced the data if the heuristics can be learned and the surface output mimicked.
3. **A two-source mixture does not break a pool** — some survivors mimic one source, some the
   other, and the pool spans both. That is the manifold behaving as designed.

### 2.24 BOTH GATE-12 PREREQUISITE AMENDMENTS ARE CERTIFIED AND COMMITTED

**S172 staging-capacity amendment — `4dd5535`, Beta CERTIFIED 2026-08-08.**

- **Option C lifecycle (binding):** retain every staged shard **through** the TrialCommit attempt;
  on SUCCESSFUL Phase-5 commit release every trial-owned reservation exactly once and delete the
  files; on FAILED commit **retain everything** so the same event stays retryable (D1.1's
  spool-repair contract depends on it). **Incremental Phase-5 assembly and mid-trial ack are
  explicitly NOT authorized.**
- **Delivery and cleanup are INDEPENDENT durable phases.** Resume is gated on
  `commit_cleanup_status`, never on `commit_delivery_status` — otherwise a crash between
  reservation rows strands the remainder forever (`ack_by_event_id` being idempotent is useless if
  the recovery path never calls it). First pass and recovery run the same code path.
- **The retention bound is DERIVED, never hardcoded.** `staging_high_water_files = None` means
  derive; an operator value below the derived requirement **fails closed before the first
  `StripeAssign`** — a warning is insufficient. The old 512 and the interim 4096 are both gone.
  For the real 16-stripe gate-12 geometry the derivation gives **3,264**; at 32 stripes / 8 workers
  it gave **6,528**. **Never transcribe either — the number comes from the geometry.**
- **The 2026-08-07 wedge was `_run_staging_job`: `while True: … except StagingBackPressure:
  sleep(0.02)` with NO CLOCK.** 50 Hz forever, invisible to a timeout that watched only paused
  connections. Executor waits now register under the **same lock** as the pause registry, so
  "oldest across both classes" is one atomic read.
- **Cohort freeze:** at successful preflight the assignable cohort is frozen **from the same
  `eligible_by_stage` the ceiling was derived over**, enforced at `assign_stripes`, the
  no-eligible-worker guard, **and `_pick_other_worker`** — the retry path was a *second* unguarded
  per-trial eligibility calculation, and fixing only initial assignment would have left the
  invariant holding on the easy path and failing on the harder one. Reconnect signature is exactly
  `backend` + `seed_caps` + `supported_variants`; `supported_variants = None` is kept distinct
  from `[]`.
- **Preflight provenance is mandatory:** if an otherwise-admissible plan cannot be durably
  persisted, **fail closed** (`coordinator_staging_preflight_provenance:`) before any dispatch. A
  failure to persist a *refusal* never masks the primary `coordinator_staging_retention_sizing`.
- **`elapsed_s` is persisted** (Beta R4), `Optional[float] = None` so "not reported" stays
  distinguishable from "measured zero". **It is stripe SERVICE TIME, not fleet throughput —
  concurrent worker intervals overlap. Never reconstruct cluster throughput by summing or
  averaging per-stripe rates; a fleet figure needs an overlap-aware makespan denominator.**
- **CERTIFICATION BOUNDARY:** one active range-miner trial per coordinator process, with
  disconnect/reconnect during that process lifetime. **NOT certified:** concurrent `run_id`s in one
  coordinator; coordinator-process death followed by mid-trial continuation of the same admitted
  run. If the coordinator dies mid-trial, the run is **interrupted/failed**, not resumable.

**S145 seed-domain / coverage-cursor amendment — `a3bb4da`, Beta CERTIFIED 2026-08-09.**

- **The discovery domain terminates at `[0, 2^32)`**, sharing `run_finalizer`'s single
  `SEED_DOMAIN_EXCLUSIVE_MAX` (`:277`) — imported, never restated. No run may begin at, cross, or
  publish outside it. The mathematical 48-bit state does **not** authorize sweeping it.
- **The legacy `exhaustive_progress` tracker is DEAUTHORIZED ENTIRELY** — all 15 rows, including
  rows 1-4. Retained and auditable as telemetry, **zero certified progress**, no longer read by the
  cursor. Certified coverage **restarts at zero**. The ~1.07-billion-seed hole therefore needs no
  repair: the table holding it is no longer the authority.
- **Coverage Ledger v1 is append-only:** bare INSERT, BEFORE UPDATE and BEFORE DELETE triggers,
  and **`PRAGMA recursive_triggers = ON` so `INSERT OR REPLACE` cannot satisfy a conflict by
  deleting.** That pragma is **proven load-bearing**, not asserted: with it OFF a 1,000-seed smoke
  row replaces a billion-seed production row (the legacy incident, reproduced); with it ON the
  DELETE trigger aborts.
- **ONE certification door:** `record_publication(artifact: RunArtifactResult, *, dataset_sha256,
  study_identity)` derives **nine** fields from the witness; only those two come from the caller,
  and both are absent from the frozen D3.5 contract. The raw writer is `_record_certified_interval`,
  out of `__all__`, and a repo AST scan proves no production bypass. **Publication is the evidence
  wall: starting a run is not coverage, receiving results is not coverage, a provisional DB write
  is not coverage.**
- **Coverage identity is `prng_base` + `skip_modes_executed`, with SET CONTAINMENT** —
  `requested_modes ⊆ record.executed_modes`. `{constant}` certified does **not** satisfy a
  `{constant, variable}` request. `_reverse` is a **direction, not a mode**. This resolved the
  `prng_type`/`prng_base` split rather than deferring it.
- **The cursor is FIRST UNCOVERED SEED, not `MAX(seed_range_end)`**, with an explicit `COMPLETE`
  state ⇒ `next_seed_start = None`. There is no numeric `4,294,967,296` next run.
- **Pre-dispatch domain wall on ALL THREE execution paths** — WATCHER, direct Bayesian, and
  `run_with_config`, the last requiring **whole-plan** validation (at `2^30 × 5` the fifth interval
  escapes the domain and the command is refused before iteration 1). `int()` coercions were removed
  so the wall's strict type contract holds. **WATCHER's `max_seeds` fallback was corrected 5M → 10M**
  — it had been validating a plan nobody executes.
- **CERTIFICATION BOUNDARY:** append-only holds **under the production connection contract**
  (ledger-managed connections set the pragma; the repo scan excludes any other certification path).
  It is **NOT** tamper resistance against an external client that disables the pragma.
- **Cursor-zero boundary:** WATCHER auto-overwrites `seed_start` only when `next_seed_start > 0`,
  so an explicit nonzero operator start remains in force. **Nothing claims WATCHER forcibly
  rewrites every run to the first gap.** Supply the first-gap value explicitly.

### 2.25 GATE 12 — ATTEMPT 1 FAILED (`distributed_config_t1_689f3cd9`, 2026-08-09)

**Beta's authorized frozen shape:** `seed_start=0`, `seed_count=2^31`, `miner_stripe_size=2^26` ⇒
**32 macro-stripes/stage**, `java_lcg` + `{constant, variable}`, range-miner, one active trial.
Beta chose 32 over the 25-stripe minimum deliberately: 25 fills the fleet once, **32 fills it and
leaves seven queued**, so the run can show the scheduler and staging operating *while* saturated.

**FAIL, two independent reasons. No 25-GPU saturation claim exists from this run.**

**Reason A — Alpha launch-shape error.** `admission_count = min(requested, selected identities)`,
and for the miner `requested` is **`worker_pool_size`** (`execution_set.py:170-176`), whose manifest
default is **8** (`agent_manifests/window_optimizer.json:262`, CLI route `:38`). Alpha's `--params`
set the seed geometry and **never overrode the pool size**, so the run asked for 8 and got 8 —
internally consistent throughout (`gpus=25 … admission=8` → `expected_workers=8` → 8 frozen
identities). **The correction is `"worker_pool_size": 25`.** Classified by Beta as an operator
error, **not a production defect.**

**Reason B — the four-stage workflow did not complete**, and the root terminal event is
**undiagnosed as of this revision**. Ledger: `(phase 1, done, 32) · (phase 2, done, 26) ·
(phase 2, cancelled, 6)`, 8 distinct workers.

**PARAMETER TRAP, recorded so it is not rediscovered:** the CLI arg `--max-seeds` is mapped in
`args_map` from the param name **`seed_count`**, but `seed_count` is **not** in `default_params`, so
WATCHER's declared-key filter drops it. **The key that works is `max_seeds`** — it is declared, the
S145 wall reads it, and the CLI builder falls back to underscore→hyphen. Passing `seed_count`
silently yields the 1,073,741,824 default (16 stripes, not 32). **Booleans are flag-only:** `true`
emits the flag, `false` **omits it entirely** — that is how `use_persistent_workers: false`
suppresses PWC.

**What the run PROVED (positive production evidence, per Beta §2):**
- retention preflight at the authorized geometry: `mode=derived required=6528 resolved=6528
  stages=4 stripes=32`;
- **staging back-pressure genuinely exercised and held** — `inbound_qsize_high_water=690`,
  `deferred_high_water=247` against `bound_in_force=1110`, **`pause_events=0`,
  `capacity_timeout_terminations=0`, `capacity_invariant_terminations=0`**, 1,844 staging jobs at
  3.055/sec. The 2026-08-05 deadlock class **did not recur under real pressure**;
- **the cohort freeze worked in production:** stage 2 saw `eligible_workers=22` — workers connected
  after preflight — and they were correctly **excluded** from the frozen trial;
- S145 behaved correctly: no publication ⇒ **no coverage advance**, cursor still 0.

**`MinerIngressError … validated=False` is the SYMPTOM, not the defect — for the FOURTH time.**
`validate_threshold_provenance` is called **only** under `if stage_idx >= len(workflow_stages)`
(`miner/range_miner_coordinator.py:6375-6385`), i.e. after all four stages complete. Stages 3-4
never started, so the gate never fired and ingress correctly refused. **Do not patch or weaken that
wall.** Note the record was **populated** this time (`payload`/`effective` present and matching)
versus empty on 2026-08-07 — threshold propagation worked for the stages that ran.

**BINDING FORENSIC FRAME (Beta):** in constant phases 1 and 2, **any stripe failure or lease expiry
fails the trial immediately — no retry** (retry-to-another-worker is for retryable failures in
hybrid phases 3/4). Whole-trial abort cleanup then marks pending stripes cancelled. **One real
failure therefore produces exactly a `26 done / 6 cancelled` tail.** The forensic target is
**the FIRST terminal event in time, worked forward** — never backward from the cancelled rows. And
if a genuine stage-2 failure is found, **immediate termination may be CORRECT behaviour, not a
retry defect**; the question then becomes why that worker failed.

**ALL THREE OPEN ITEMS ARE NOW ANSWERED — see §2.26 (forensics, 2026-08-09).**

**ALPHA TOOLING ERROR:** the concurrency sampler was started *after* the fleet-launch step returned,
so it produced **no in-run rows** — this run carries **no live concurrency evidence even for the 8
workers used**. Any future attempt must start the sampler **before the coordinator can issue the
first `StripeAssign`**, and Beta requires an observation window showing **≥25 distinct in-flight
workers AND queued stripes still available**. "Distinct workers eventually used = 25" is explicitly
insufficient.

### 2.26 GATE-12 FORENSICS — THE COMPUTE LEASE MEASURES QUEUE WAIT, NOT WORKER LIVENESS

Read-only forensic reconstruction of `distributed_config_t1_689f3cd9`, Beta-authorized 2026-08-09.
**Verdict: PRODUCTION DEFECT FOUND.** All three items left open by §2.25 are answered here.

**The first terminal stage-2 event:** compute-lease expiry at **12:47:13.143** on
`…__st1_s5 / rrig6600:gpu4 / attempt 0`. Handled `range_miner_coordinator.py:6367 → :5186 →
:5205 → :5106-5107` (`if phase in (1,2): fail_trial(...)`) `→ :5405 cancel_active_stripes`.
**Per Beta §8 this is CORRECT behaviour, not a retry defect** — phase 2 is constant-mode and the
reassign path is hybrid-only. **The six cancellations are the abort-cleanup footprint of this one
event**, proven without relying on them: `cancel_active_stripes` updates `state` only, so a stripe
killed on this path keeps `claimed_by` **and** its expired `lease_expires_at`; s5/s7/s9 show exactly
that, while s3/s6/s26 carry `stripe_complete_seen=1` + `lease NULL`.

**F-1 (PRIMARY DEFECT) — the lease is stamped at BULK-CLAIM time.** `assign_stripes`
(`:2680-2705`) claims **every** stripe of a stage in one loop with **one `now`** (set once at
`:2671`), stamping each `now + compute_lease_timeout` (`:2695`, 300 s at `:245`). Workers then
execute **serially** (`range_miner_worker.py:1425-1431`). At `stripes_per_worker = 32/8 = 4`, a
worker's last stripe does not begin until **~230-260 s of its own 300 s lease is already gone.**

- **The three that expired were ACTIVELY STREAMING RESULTS** — last shards 12:47:11.338 /
  12:47:12.056 / 12:47:12.607 against a **12:47:05.487** deadline. **Not dead workers.** The
  lease's documented purpose (`:1663-1667`) is reclaiming leases from workers that have *stopped*.
- **Renewal cannot compensate:** `renew_lease` (`:1648-1661`) renews **only
  `msg.current_stripe_id`**, so a queued stripe's lease burns untouched; once current, the
  heartbeat competes with the result stream on one ordered TCP connection (`:6549-6552`). No
  heartbeat renewal landed on s5/s7/s9 at any point (`lease_expires_at` is still
  assign-time + 300 to the microsecond). The §2.19 F2 lease exemption **cannot apply** — it keys on
  `_paused_connections` and the run recorded `pause_events=0`.
- **The clean control is in the same run:** phase 1 is geometrically identical and cleared the lease
  by **+64 s** (4.31 shards/s); phase 2 ran 3.24 shards/s and missed by **−11 s**. **Worker compute
  was 4.5-6.7 s per stripe throughout — the 300 s went to delivery and staging, not GPU work.**
- **Blast radius:** phases 1 and 2 are constant-mode, so **any** stage whose per-worker stripe queue
  takes longer than 300 s to deliver terminates the whole trial with no retry. **A fail-closed
  cliff, not a degradation**, live at any geometry where
  `stripes_per_worker × per-stripe delivery time → 300 s`.
- **THEREFORE: raising `worker_pool_size` to 25 would MASK F-1, not fix it.** It drops
  stripes-per-worker to 1-2 and would very likely have avoided the expiry in this run, but **does
  not remove the coupling.** A gate-12 pass obtained that way would certify a latent cliff.
- **No remedy is proposed.** The candidates — stamp the lease at dispatch rather than at claim;
  renew on any accepted frame from the bound worker rather than heartbeat alone; claim only what a
  worker can start — **differ materially in concurrency properties**, so the choice is Beta's, under
  the §7 owner rule on taking the structurally stronger mechanism.

**F-2 (secondary, observability) — the constant-phase terminal path is SILENT.**
`_handle_stripe_failure_locked:5106-5107` builds a precise reason string and emits **no log
record**; `fail_trial:5342-5348`, `abort_trial:5350-5423` and `cancel_active_stripes:1546-1556`
emit none either, and `trials` has no column for the reason (discarded at `:5406-5407`).
`process_lease_expiry` logs only its two *skip* branches. **Observed consequence: the coordinator
log contains nothing at all between 12:42:05.645 and 12:47:17.448**, and the operator saw only a
downstream `MinerIngressError` about a gate that never ran. **Every fact above had to be recovered
from ledger row shapes** — and only because `cancel_active_stripes` happens not to overwrite
`claimed_by`. The neighbouring capacity-timeout path *does* `logger.error` first (`:6031-6032`), so
this is an inconsistency inside one file.

**§2.25's `Defect 6` item — UNRELATED, disposed.** Two independent proofs: the branch at
`:6127-6141` fires only for connections where `meta["registered"]` is false, and **no stage-2 stripe
existed until 12:42:05.487**. Which worker owned it **cannot be determined** — no `worker_id` was
ever bound and the log line carries no peer address.

**§2.25's `GPU_COUNT_MISMATCH: 0/8` — disposition C, environment/probe defect.** `rocm-smi` is not
on the **non-interactive SSH PATH** on the CT; the identical grep returns **8** via
`/opt/rocm/bin/rocm-smi`. The parsing is correct. `0/8` did not fail the 3/3 preflight because
`preflight_check.py:229` does `checks_passed += 1  # Don't block on GPU warnings`. Secondary: the
probe **cannot distinguish `UNAVAILABLE` from `0`** — its `|| echo 0` reports an unobservable
surface as a definite zero.

**CORRECTIONS TO ALPHA'S OWN GATE-12 EVIDENCE PACKAGE (§3 of that document was wrong):**

1. **Raising the pool to 25 does NOT change the derived retention requirement — it stays 6,528.**
   Computed by importing the production `trial_retention_files_required` (`:613`) with the caps read
   from this run's own ledger; the 8-worker row reproduces the run's logged line exactly. **The
   derivation is a MAX over eligible workers, not a sum** (`:620-624`), and the tightest cap is
   already the AMD one: `ceil(67108864/2000000)=34` constant, `ceil(67108864/1000000)=68` hybrid ⇒
   `32×34=1088` and `32×68=2176` per stage **regardless of pool size**. It would change only if a
   worker advertised a *smaller* cap; none of the 22 that registered does.
2. **The sampler's first row was 12:47:28, not 12:51.**
3. **The sampler's query cannot demonstrate Beta's criterion at all** — under bulk claim `pending`
   never appears, and `claimed_workers` reports **assignment, not occupancy**. A second, independent
   Alpha tooling defect; a corrected block exists in the forensics report §E.3.

**STANDING LESSON:** a terminal decision that leaves no execution record is not observable. The
fail-closed design Beta correctly credits is only auditable here by accident of which columns
`cancel_active_stripes` leaves untouched.

### 2.27 F1/F2 ACTIVE-LEASE SCHEDULER — CERTIFIED AND COMMITTED (`c4e0037`)

Beta CERTIFIED 2026-08-09. **This is the fix for the gate-12 attempt-1 failure (§2.26 F-1).**

**The governing invariant:** *a compute lease exists only for work the coordinator has handed to a
worker presently able to execute it. **Undispatched backlog has NO lease.*** For the current serial
worker: **maximum ONE compute-active claim.**

- **`schedule_pending_stripes` (`:2920`) is the ONLY place a compute lease is created.**
  `assign_stripes` still builds the whole governed geometry — **all 32 rows for gate 12** — but they
  are born `pending / claimed_by NULL / lease_expires_at NULL`. At W=8 the stage opens **8 claimed /
  24 pending**; at W=25, **25 claimed / 7 pending**. `pending` is now a REAL backlog state.
- **The one-active invariant is enforced in SQL** inside `claim_stripe` and **RAISES
  `LeaseInvariantError`** rather than silently refusing — *"a silent refusal would let a bulk-claim
  regression look like correct behaviour."*
- **Renewal: heartbeat OR accepted active-stripe progress**, scoped to the active attempt only.
  **Forbidden to renew:** wrong worker · wrong stripe · stale attempt · invalid result · status
  frame · late result from a prior attempt. The rule is *progress on THIS active attempt renews
  THIS active attempt* — **not** "any traffic from the host keeps everything it owns alive."
- **`StripeComplete` frees the compute slot without waiting for staging** — compute and staging
  genuinely overlap (worker-side send, coordinator-side async staging).
- **Abort cleanup now reaches pending backlog** — no nonterminal/runnable stripe may survive
  termination.
- **The frozen cohort still governs every `pending → claimed`.** Late registrants stay globally
  connected and cannot receive work for an already-frozen trial.
- **The dispatcher needed NO change** — `_dispatch_pending` byte-identical, worker untouched, no
  protocol change: a stripe the scheduler has not claimed cannot be dispatched.
- **Hybrid retry defers PLACEMENT when no alternate is compute-idle** (Alpha position, Beta
  RATIFIED). The terminal decision is untouched; only placement waits.

**R1 — two defects Beta found that the 13/13 suite did not exercise:**

- **A prefix selector was used as an exact selector.** `_handle_stripe_failure_locked` passed a
  complete `stripe_id` into a `LIKE <prefix>%` parameter, so targeting `s1` also matched
  `s10`-`s19`; `placed[0]` could be an **unrelated sibling**, and the handler then reported
  `action="reassigned"` naming a worker that never took the failed stripe — **a false statement in
  the retry contract.** Now two mutually exclusive parameters: `stage_prefix` (LIKE) and keyword-only
  `exact_stripe_id` (identity), `ValueError` if both. **Nothing infers intent from string shape.**
  It survived because the hybrid gate used **two** stripes; the collision needs `s1`/`s10`.
- **Terminal replay parity:** the first durable transition now owns terminal identity permanently;
  already-aborted **and race-lost** paths re-read the durable row and rebuild identity from it,
  including the legacy `reason` prose and the winner's `abort_event_id`.

**R2 — one canonicalization:** `reason = terminal.reason` moved **before** the first/non-first
divergence, so one `event_id` cannot carry two payloads. Previously the first delivery emitted the
*caller's* prose beside canonical `terminal_*` fields.

**Guard supersession (Beta-authorized, ORDER MATTERS):** the byte-identity guards on
`_handle_stripe_failure_locked` were replaced by **ordered decision-semantic** guards — the four
terminal decision tuples in order. **Beta was explicit: fix the defects FIRST, then supersede —
re-pinning to defective source would certify the defects.** The superseding assertion is
**self-protecting**: it asserts `exact_stripe_id=stripe_id` present AND `stripe_prefix=stripe_id`
absent, so a guard re-pinned to defective source fails on its own terms. `admission_binding` B7
received the same treatment (Alpha judgment call, Beta ratified).

**Beta's framing, worth keeping:** **structured fields carry machine truth; prose carries
diagnostic detail; replay changes neither.** Test assertions scraping the legacy `reason` prose for
facts F2 moved into structured fields were re-pointed at `terminal_class`/`run_id` — Beta ratified
that as *"replacing machine decisions inferred from prose with the structured fields F2 was
specifically introduced to provide."*

**CERTIFICATION BOUNDARY:** the one-active check is a **lock-serialized SELECT-then-UPDATE within
one coordinator process — NOT one statement**, and **not** protection against an external writer or
a second coordinator.

### 2.28 PRE-RERUN ITEMS — GPU PROBE (CERTIFIED) AND CONCURRENCY SAMPLER (R2)

**The GPU probe — Beta CERTIFIED.** Root cause, measured on all three CTs: `/opt/rocm/bin` is put on
PATH by **`~/.bashrc:120` and nothing else**, and `~/.bashrc:5-8` is Ubuntu's stock
**non-interactive guard** returning ~112 lines before that export. **`bash -lc` and a bare
non-interactive command see the byte-identical PATH; only `bash -lic` reaches it** — a login shell
was never going to fix it. **Two constructs each manufactured the zero**: `grep -c` printed `0` and
exited 1, then `|| echo 0` printed a second. Also: `ssh` was flattening the argv so the remote shell
re-parsed the pipeline with quoting gone — semantics survived by luck.

Now **three distinguishable outcomes**: a count · **`UNAVAILABLE`** (`gpu_count None`, never `0`) ·
`ERROR`. Binary located, stderr surfaced, render guard prevents `0/8` **or** `None/8` for an
unavailable node. **Advisory gating preserved** (`checks_passed += 1  # Don't block on GPU
warnings`). Measured live: **8/8 on all three rigs.**

**The sampler — three Alpha errors, all corrected.** It must prove Beta's criterion:
**≥25 DISTINCT workers simultaneously compute-active AND queued stripes still available.**
Insufficient: "25 connected" · "25 eventually used" · "32 eventually completed."

- **Semantics (post-F1):** occupancy is `state='claimed'` ONLY — **staging is excluded because
  `StripeComplete` already freed the slot** — and `pending` is the real backlog. The pre-F1 query
  counted `claimed + staging` and never looked at `pending`, i.e. it **overstated occupancy and
  could not see the queue at all.**
- **Atomicity:** a sample is TWO reads. In autocommit they are two independent read transactions and
  **the sample that decides the verdict is the one most likely to be internally inconsistent**,
  because the interesting window is exactly when transitions occur. Fixed with `read_snapshot`
  (`BEGIN DEFERRED … COMMIT`, `isolation_level=None`; `IMMEDIATE` is unavailable on `mode=ro`).
- **VIR-5 on the LEDGER read, not just ESTAB.** Alpha applied VIR-5 to ESTAB — a *context* field —
  and missed the read the criterion is made of. A failed ledger read fell through and was appended
  as a **definite `compute_active=0, queued_pending=0` sample**, while the comment above it claimed
  the opposite. **Rows are now born UNOBSERVED** and become observations only if a read succeeds —
  structurally impossible to fall through, since there is no zero to fall through to. **Gap rule:
  break the window AND annotate** — a window is a claim of *sustained* simultaneity, and an unknown
  interior instant destroys it; across a gap the fleet may have emptied and refilled.
- **A second consequence of the same fall-through:** `runnable = pending + claimed + staging` summed
  a gap to `0+0+0`, so **failed reads started the quiescence timer** and could stop the sampler with
  "run is over" while the run was alive.
- **Two verdicts, never collapsed:** sustained simultaneity AND turnover under full occupancy.
  Turnover is summed **step-wise over consecutive in-window pairs**, so a drain during an occupancy
  dip is not credited and a stage-boundary refill cannot make endpoints lie. Exit codes
  **`0` both · `2` simultaneity failed · `3` turnover failed.**
- **Ordering:** the sampler must arm **before the coordinator can issue the first `StripeAssign`**
  and terminate with the run. In attempt 1 it started *after* the fleet-launch step returned — first
  row 12:47:28 for a run that died at 12:47:17 — producing **no in-run rows at all.**

**Beta's standard, quotable:** *"A saturation verdict computed from an unknown number of missing
samples is not evidence."*

### 2.29 GATE-12 ATTEMPT 2 — THE SHAPE (RAN 2026-08-10; SEE §2.31 FOR THE RESULT)

> **HISTORICAL + STILL-BINDING.** The shape below is the one attempt 2 ran and is **unchanged for
> attempt 3** (Beta §17). The parameter traps are permanent. The *outcome* is §2.31.

Beta chose **32 stripes** over the 25-stripe minimum deliberately: 25 fills the fleet once; **32
fills it and leaves seven queued**, so the run exercises scheduler turnover, completion,
reassignment, staging and back-pressure **under full occupancy**.

```
seed_start        = 0              (explicit; certified first-gap, empty {constant,variable} namespace)
max_seeds         = 2,147,483,648  (2^31) ⇒ 32 macro-stripes per stage   ← THE KEY IS max_seeds
miner_stripe_size = 67,108,864     (2^26)
worker_pool_size  = 25             ← the attempt-1 correction; manifest default 8 was never overridden
test_both_modes   = true           prng_type = java_lcg
window_trials     = 1              n_parallel = 1
use_range_miner   = true           use_persistent_workers = false
```

**PARAMETER TRAPS (both verified against source):**
1. **`max_seeds`, NOT `seed_count`.** `args_map` maps `--max-seeds` from `seed_count`, but
   `seed_count` is **not** in `default_params`, so WATCHER's declared-key filter drops it and the
   run silently falls back to `1,073,741,824` — **16 stripes, not 32.** `max_seeds` is declared, the
   S145 wall reads it, and the CLI builder falls back to underscore→hyphen.
2. **`trials`, NOT `window_trials`, for the CLI.** `args_map` maps `trials`; the underscore fallback
   would emit a **non-existent `--window-trials`.**
3. **Booleans are flag-only:** `true` emits the flag, `false` **omits it entirely** — that is how
   `use_persistent_workers: false` suppresses PWC.

**Standing run conditions:** no mid-run intervention of any kind · a sizing refusal at preflight is
a **legitimate Gate-12 result**, not a reason to shrink the seed count · coordinator process death
means **interrupted, not resumable** · **GPU completion is not completion — only successful
canonical publication is** · fewer than 25 admitted workers ⇒ **no saturation claim.**

### 2.30 THE FIVE-DEFECT PATTERN, AND BETA'S THREE FALSIFIERS

Every defect in the gate-12 sequence had one shape: **an implementation that passes its own gates
because the gate encodes the same assumption the implementation does.**

| defect | untested assumption | why the fixture missed it |
|---|---|---|
| `staging_deferred_max = 64` | bursts stay small | never generated 116 requests |
| `staging_high_water_files = 512` | trials stay short | never ran a whole trial's file count |
| stage-eligibility bound | one worker set serves every stage | had one worker set |
| compute lease at bulk claim | a worker starts work when assigned work | never queued 4 stripes on one serial worker |
| prefix-as-exact selector | no lexical sibling exists | used 2 stripes; needs `s1`/`s10` |

**Beta's correction to Alpha's framing:** the defect is not "the rules are too tight" — it is that
**the same agent writing both the implementation and its fixtures creates correlated blind spots.**
There are **three** independent falsifiers, not one: **independent reviewer reasoning** (Beta
supplied the prefix collision) · **adversarial fixture generation** (needs no fleet access) ·
**real execution.**

**QUEUED WORK — adversarial fixture dimensions (Beta §8), to be briefed after Gate 12 settles.**
*(PARTIALLY DISCHARGED: Defect A's A1–A8 suite generated the **25-worker × four-stage** shape with
**disconnect/reconnect**, `W < N` at a stage boundary, duplicate-socket race and late re-join —
the dimensions attempt 2 died on. The rest of the list is still queued. Attempt 2 is the **sixth**
instance of the pattern, and the seventh was Alpha's own §14 budget: an enforcement gate that
shared the implementation's assumption that `connect()` returns promptly, so it could never
exercise a blocking connect. Beta caught it; A8-B2 now gates it.)*
The dimension list must come from **outside the implementer**: `N = 2, 9, 10, 11, 19, 20, 31, 32` ·
`W < N`, `W = N`, `W > N` · heterogeneous eligibility by stage · queue service time > lease ·
**lexically overlapping identifiers** · late joins · disconnect/reconnect · partial staging ·
idempotent replay with contradictory input. **Every one of the five defects lies on that list.**

**Terminology, per Beta:** an owner-authorized-per-occasion diagnostic run is **not** autonomous
execution. If a delegated-execution amendment is ever proposed it needs Beta's eight-part contract —
frozen hypothesis, hard resource bounds, non-certifying, no authority mutation, no self-healing
relaunch. **Nothing changes unless Beta rules.**

### 2.31 GATE-12 ATTEMPT 2 — RAN AND FAILED (`distributed_config_t1_abc63f71`, 2026-08-10)

**Result: FAILED.** First authoritative terminal event **10:44:16**, `worker_admission_timeout` at
the phase-4 admission boundary: *stage 4 expected 25 eligible workers; **23 admitted after 180.1 s***.

**What completed (measured, ledger + coordinator summary):** stages 1, 2, 3 all **32/32 stripes**,
each over the full `[0, 2^31)` domain = **6,442,450,944 seed-evaluations**, on **25 of a planned 26
GPUs** (Zeus's second 3080Ti unprovisioned). Stage 4 **never assigned** — no phase-4 ledger row, no
phase-4 `derived_bound` line. Survivors: java_lcg **0** · java_lcg_reverse **0** ·
java_lcg_hybrid **44,331**. `staging_jobs_completed=3948`, `capacity_timeout_terminations=0`,
`capacity_invariant_terminations=0`, `pause_events=0`.

**Measured cluster throughput (this run, end-to-end incl. delivery + staging):** stage 1
2^31 seeds in 546 s ≈ **3.93 M seeds/s**; stage 2 in 796 s ≈ **2.70 M seeds/s**. Hybrid is slower
(half batch, double retention, 44k survivors to serialize). **Per-GPU rate is NOT measurable from
this run** — worker logs are stdout capture only (three "Compiled kernel" lines, no timing). Do not
divide the cluster figure by 25 and present it as per-card: the fleet is heterogeneous.

**Exonerated by evidence, not assertion:** NOT hardware (8/8 GPUs post-run on all three rigs, zero
`dmesg` faults, no GCVM_L2) · NOT F1/F2 (it fail-closed correctly and refused stage 4 on 23) · NOT
staging/back-pressure (numbers above).

**The fail-closed cascade worked end to end and is the model outcome:** admission timeout → trial
aborted → `provenance_validated` stays false → `MinerIngressError` → Optuna trial fails → no
`optimal_window_config.json` → WATCHER hard output-validation fails, confidence 0.00, human
escalation. **The pipeline did not reinterpret three good stages as a successful four-stage trial.**

**Beta's corrections to Alpha's forensic report — all three ratified against Alpha:**
1. **"23 proves a connection-liveness gap, not a process death" is NOT ratified.** 23 proves only
   that **two identities were absent from the live registered set**. It cannot distinguish TCP
   failure / worker exit / coordinator close / remote OS kill.
2. **"The worker is per-stage one-shot by design" is FALSE.** It is a **long-lived trial transport
   session** that exits on a transport exception because it had **no reconnect path**. That
   distinction killed the "relaunch the fleet per stage" remedy outright.
3. **"The fleet behaviour is sound" was withdrawn.** The accurate claim is narrower: *the scheduler
   achieved full 25-worker simultaneity **and** real queued-work turnover during at least the early
   qualifying window.* Defect A is itself evidence the fleet transport was **not** sound.

**§23 forensics — the initiating cause of the two lost sessions is UNRESOLVED, and no cause may be
claimed.** Local artifacts are silent (zero coordinator WARN/ERROR 09:47:29→10:44:16 — which is
*why* §15 observability was ordered); kernel ring empty; netconsole empty **but cannot distinguish
"no event" from "not active"**; **rig-side worker logs are UNEXAMINED, not silent.** No TCP idle
timeout is claimed. Beta accepted this disposition and required no further forensics.
**[v25 UPDATE — see §2.41]** The rig-log silence in attempts 4/5 now has a **STRONGLY SUPPORTED /
PRESUMPTIVE** cause: the deployed worker contained no session emitter at all. *"The logging channel
itself was broken"* is **no longer the leading explanation**. The exact historical rig source bytes
remain **NOT PROVEN** — they were never captured — so this is presumptive, not proven.

**The live miner ledger is `/home/michael/miner_staging/miner_ledger.db`** — the same-named file in
the project root is **stale** and will answer run-scoped queries with other runs' history. Schema:
`stripes(run_id, stripe_id, seed_start, seed_count, state, claimed_by, phase, family_name,
survivors_total, …)`.

### 2.32 THE TWO CERTIFIED AMENDMENTS — DEFECT B AND DEFECT A

**DEFECT B — sampler all-qualifying-window turnover. CLOSED / CERTIFIED `f216475`.**
`evaluate()` fed `_turnover` only the **single longest** qualifying window, a false negative whenever
a shorter qualifying window holds real turnover. Now: `_window_turnover(window)` measures each
window (step-wise arithmetic verbatim from the certified `_turnover`, and now also the single source
of `windows_detail`, so census and verdict cannot drift); `_turnover(measurements)` aggregates
**existentially** — `TURNOVER_SATISFIED = EXISTS qualifying window WHERE pending_drained > 0 OR
transitions > 0`. **Widening the aggregation does NOT widen any measured interval** — each
measurement stays inside one window, so an occupancy dip or an UNOBSERVED gap remains uncreditable.
Witness = **earliest qualifying window by start epoch that shows turnover**, labelled; the longest
window is demoted to `CONTEXT ONLY`. Criterion 1 untouched. Suite 49/49 (44 certified + **DB1–DB5**,
renamed from Beta's B1–B5 to avoid shadowing the certified ESTAB B1–B7 — Beta **ratified** the
rename and the additive `windows_detail` schema).

**§20 FORENSIC FINDING (Beta-confirmed, NON-CERTIFYING):** re-run read-only against the preserved
attempt-2 TSV (`sha256 4f69dba7…`), the corrected evaluator finds **real turnover in qualifying
window 1** — 09:25:10/12/14 at `active=25`, `pending 7→6→3` (**drained 4**), `done 0→1→4`
(**transitions 4**). **Attempt 2's banked `VERDICT 2: NOT SATISFIED` was an instrumentation false
negative caused by longest-window aggregation, not a fact about the fleet.** It does **not** rescue
attempt 2 and **carries no credit into attempt 3.**

**DEFECT A — RANGE-MINER transport-session recovery. CLOSED / CERTIFIED `2532803`** (implementation
`acd6f13`, §14 deadline revision `2532803`).

*The defect:* `serve_forever` exited at ONE point for THREE causes, so a permitted worker that lost
its session could never re-register. **A finding Beta ratified beyond the original diagnosis:**
`_dispatch` sat **outside** the inner `try`, so the collapse was really **two** silent exits — an
idle (recv) loss returned 0, a loss mid-result-stream (send) propagated an **uncaught traceback**.

*The certified architecture:*
- **§10 three-way state machine.** Discriminator is `_stop.is_set()` **at the moment the transport
  exception is caught**: set ⇒ a `shutdown` frame or signal already decided; clear ⇒ genuine
  transport loss, recover. `_stop_cause` is **first-writer-wins** so `finally: shutdown()` cannot
  rewrite a signal into a generic stop.
- **§11 NO STALE-WORK REPLAY.** Reconnect is **transport** recovery, never **assignment** recovery.
  The worker returns **idle** and re-sends nothing. F1/F2 alone decides the abandoned assignment:
  constant-phase loss terminal · hybrid first loss = the one certified retry on an alternate ·
  hybrid second loss terminal.
- **§12 one live socket per identity.** A duplicate racing the eviction is a **retryable
  session-establishment condition** — back off, retry the same identity. Never force-replace.
- **§13 frozen cohort remains authority.** `IDENTITY_FIELDS` is an **enumerated allowlist**
  (worker_id, hostname, gpu_id, gpu_name, backend, vram_bytes, capabilities). **NOT
  `dataclasses.asdict()` whole** — `RegisterMessage` carries a per-message `timestamp`, so a whole
  compare fails closed on **every** reconnect and silently re-creates the no-reconnect defect in a
  §13 costume. *(Alpha wrote the defective version; gate **A7's** red-first caught it.)*
- **§14 bound, and its enforcement.** Budget is **positive-finite, cumulative across all episodes**
  (a per-episode reset re-creates the immortal orphan under duplicate-rejection ping-pong), derived
  from `DEFAULT_WORKER_ADMISSION_TIMEOUT` — **read, never redefined.** Enforcement (the R2
  revision): post-backoff re-check before any new attempt · `connect(timeout=remaining)` passes the
  deadline **into `socket.create_connection`** (previously absent, so a black-holed route blocked
  past the clock and the budget was bookkeeping, not a bound) · REGISTER bounded by the residual ·
  **`settimeout(None)` restores blocking BEFORE the session is served**, or the certified read loop
  would misclassify ordinary silence as TRANSPORT_LOSS and loop an idle worker forever.
- **§15 observability.** Worker-side session events (transition-only, no heartbeat noise) and
  coordinator-side `WORKER_DISCONNECTED{worker_id, stage_idx, stage_assigned, identity_evicted,
  eligible_count_after_drop}` / `WORKER_REGISTERED|RECONNECTED{worker_id, registration_generation,
  eligible_count_after_register}`. **An unmeasurable eligible count is `UNOBSERVED`, never `0`** —
  the S4 lesson carried into the coordinator. `_registration_generation` is a **record-keeping
  counter only**; nothing reads it for eligibility.

*Ratified judgement calls:* `close()` shuts down before closing (mirrors coordinator Defect-6 C3) ·
the **send-side** exception surface deliberately excludes `ValueError` (an oversized frame is a
payload-contract violation, not a dead socket) · cumulative-not-per-episode budget.

**Rejected remedies, on the record:** switching Gate 12 to PWC / `use_persistent_workers=true`
(much larger certification surface) · admitting **frozen identities** while disconnected (frozen
cohort = *who MAY work*; live registration = *who CAN work now* — both are kept) · per-stage fleet
relaunch (wrong lifecycle model) · widening `worker_admission_timeout` (waiting longer cannot repair
a nonexistent recovery path).

Suite **29/29** (26 + A8-B1/B2/B3). `A1/RED` restores the **verbatim** pre-fix `serve_forever` body
and reds. **A8-B2's mutant survived its first run** — the verbatim copy resolved `socket` in the
*test module's* globals and escaped the black-hole shim; fixed by rebinding globals to the module
under test. A vacuous gate about an unenforced deadline, caught by the mutation discipline itself.

### 2.33 GATE-12 ATTEMPT 3 — AUTHORIZED (from `2532803`) — SEE §2.34 FOR THE RESULT

**Launch tree is `2532803`. Do NOT launch from a later unreviewed production-code commit.**
Pre-launch mechanical check **already passed** (2026-08-10): HEAD `2532803`, clean tree, phase-4
**63/63**. Beta: if that self-clear held, **no further Beta review is required** before launch.

**Shape is unchanged from §2.29** — `max_seeds = 2147483648` remains the governing key, never
`seed_count`; 32 macro-stripes per stage; four stages java_lcg fwd → rev → hybrid fwd → hybrid rev.

**Pre-launch conditions (Beta §18):** HEAD `2532803` · phase-4 63/63 · GPU truth gate **8/8 on each
of the three remote rigs** · frozen eligible cohort **25** · `worker_pool_size = 25`. **No automatic
downsizing. A GPU-gate refusal is a refusal, not permission to launch with fewer devices.**

**A reconnect during attempt 3 is NOT automatically a failure (Beta §19)** — it is now a certified
recovery path. The evidence must show `WORKER_DISCONNECTED` → `WORKER_RECONNECTED` for the **same**
worker_id with the frozen identity unchanged. A reconnect authorizes **no** replay and **no**
alteration of F1/F2 semantics.

**Completion authority (Beta §21) — all seven, IN THE SAME RUN, nothing composes across attempts:**
truthful GPU preflight PASS · 25-worker frozen admission · `GATE-12 SATURATION VERDICT : SATISFIED`
(both verdicts, via Defect B's corrected existential evaluator) · **all four stages complete** ·
**D3.5 canonical publication succeeds** · S145 publication-bound coverage succeeds · **certified
cursor == 2,147,483,648**. **A successful GPU scan alone is insufficient.**

**Preserve separately for the mathematics (Beta §22):** constant-forward, constant-reverse,
hybrid-forward, hybrid-reverse survivors, and the final **`|F_hybrid ∩ R_hybrid|`** intersection —
the new result of interest. Attempt 2's `hybrid forward = 44,331` carries **no** authority forward.

**Gate 22 — Beta REJECTED permanently allowlisting the Defect-A harness.** The detector reads
`git status --porcelain`, so a **modified tracked** non-allowlisted `.py` trips it too — not only an
untracked one. *(Alpha's own brief wrongly assumed "committed once ⇒ clears forever"; Claude Code
corrected it.)* The answer is never to widen the allowlist: it self-clears on a clean committed tree.

**Governance:** Beta records that the evidence threshold for revisiting the diagnostic-fleet
proposal is now met (two production failures on dimensions no fixture covered). **That does NOT
amend Rule 3.** A separate governance amendment may be submitted if Michael wants one; nothing
changes unless Beta rules.

### 2.34 GATE-12 ATTEMPT 3 — RAN AND FAILED (`distributed_config_t1_d606edbe`, 2026-08-10)

**The compute path worked end to end. Publication refused it.** Launched 17:28:31 from HEAD
`3254a30` (docs-only descendant of the reviewed `2532803`; Beta ratified that as no production-code
lineage violation). Terminal at 19:48:14.

**Conditions 1–4 SATISFIED, and this is the run that proved RANGE-MINER:**

- GPU preflight 8/8 × 3. **25-worker frozen admission held for the entire run** — ESTAB max=25
  **min=25** across all 3,541 samples, **0 UNOBSERVED**.
- `GATE-12 SATURATION VERDICT : SATISFIED`, exit 0. Peak 25 compute-active with 7 queued at
  17:29:58; 78 satisfying samples, 10 qualifying windows.
- **All four stages, 128/128 stripes, full `[0,2^31)`.**
- **Zero disconnects, zero reconnects.** The stage 3→4 boundary that killed attempt 2 crossed
  cleanly. **[v25 CORRECTION — §2.41]** This was originally recorded as "Defect A held by
  *prevention*." **WITHDRAWN.** The rigs had **no Defect-A recovery code deployed at all**, so
  §19 reconnect-crediting was **unsatisfiable by construction**. A clean run is evidence that
  reconnect **was not needed**, NOT that it works. Defect A's 29/29 VM101 certification stands;
  its **fleet deployment was never demonstrated**.
- **Defect B was decisive.** The turnover witness is **window 1** (3 samples); the *longest*
  window shows `drained=0 transitions=0`. The pre-`f216475` longest-window evaluator would have
  false-negatived a second time.

**Terminal event:** `utils.run_finalizer.RunParameterError: repository_tree_clean is False`,
`utils/run_finalizer.py:1592`, via `window_optimizer_integration_final.py:2983`. Step 1 exit 1.
Certified cursor stayed `OPEN` / `covered_seed_count=0`.

**Beta's root-cause ruling, and it corrected Alpha:** Alpha framed this as "a predicate
disagreement, not a defect in the run." **REJECTED.** D3.5 did not discover a new reading of
"clean" — it enforced the already-certified one (the D3.5 prerequisite says `git status
--porcelain` empty; suite gate **F37** rejects `repository_tree_clean=False`). The defect was
entirely **upstream**: launch admission dispatched two hours of fleet compute from a state
publication was predetermined to reject. Classification: **PRE-LAUNCH CLEAN-TREE ADMISSION DEFECT**.

**Mathematics (forensic only — NO certified authority, non-composable, per Beta §14):**
trial `W22_O0_evening_S7-229_FT0.47_RT0.47`. Survivors: constant fwd **0**, constant rev **0**,
hybrid fwd **774**, hybrid rev **6**, and `raw bidirectional candidates = 0` — the
`|F_hybrid ∩ R_hybrid|` intersection is **zero**. `[S172-BP] summary`:
`staging_jobs_completed=5999`, `staging_jobs_per_sec=0.717`, `pause_events=0`,
`capacity_timeout_terminations=0`, `deferred_high_water=1565` vs `bound_in_force=2201`.

**Attempt 3 remains FAILED and IMMUTABLE.** Beta refused "clean the tree now and finalize after
the fact" — that would erase the condition the finalizer correctly detected. Evidence frozen and
hash-verified at `/home/michael/gate12_attempt3_20260810_200824` (13/13); the three residue files
preserved separately at `/home/michael/attempt3_residue_20260811` (3/3) before removal.

**Second, separate defect found in this run — WATCHER FAILURE AUTHORITY (STILL OPEN).** After
`Step 1 failed with code 1`, WATCHER reported `file_exists`, confidence **1.00**, `Step 1 PASSED`,
`Triggering Step 2` — because `optimal_window_config.json` had been written *before* the finalizer
raised. **An explicit subprocess/finalization failure must dominate file-existence heuristics.**
Only `--end-step 1` contained it. **Required before Phase-7 autonomy.** Beta: not to be folded
into the clean-tree repair. Attempt 4 did *not* reproduce it (no stale file existed) — that is
non-reproduction, **not** closure.

### 2.35 CLEAN-TREE ADMISSION REPAIR — CLOSED / CERTIFIED / PRODUCTION-OBSERVED (`213bfff`)

**Two defects, one repair.** (1) `gate12_launch.sh:54` **printed** `git status --porcelain` into
the evidence block and **never tested it** — it printed the reason the run would fail two hours
before it failed. (2) The clean slate renamed `optimal_window_config.json` (ignored) to
`optimal_window_config.json.pregate12_${STAMP}` — a name **no ignore rule matches** — so the
harness dirtied the tree by its own hand, *after* admission and *before* dispatch. Beta caught (2);
Alpha's brief had only (1). It never fired in attempt 3 (no config existed at launch) but *would*
have fired on the next launch.

**The invariant, binding:** ONE predicate → admission (clean) → **launch preparation must preserve
it** → last pre-dispatch assertion → compute → same predicate at D3.5. No launch-harness operation
may create a state D3.5 will reject.

**As built:** `scripts/gate12_cleantree_gate.py` imports `_repository_state` **by identity** from
`window_optimizer_integration_final` — the producer whose boolean the finalizer receives
(`:2972 → :2992`), not a reimplementation, because a second implementation is a second predicate.
`decide(clean)` is **unary** (AST- and behaviourally proven), so the entry listing structurally
cannot influence the verdict; the listing is diagnostic-only. A producer exception →
**UNAVAILABLE, never "clean"**. Rotation destination moved into `logs/`, ignored as a whole
directory — **no filename exception anywhere**. D3.5, the producer and `.gitignore` are
sha256-identical to HEAD.

**Gates 31/31** (C1–C5 + C5A). Beta returned it twice, both for **provenance, not architecture**:
**R1** — RED arms read `HEAD:gate12_launch.sh`, which self-invalidates the moment the repair is
committed; now pinned to `PRE_REPAIR_COMMIT = 3254a306…` and refusing unless both old defect
surfaces are present **in executable lines** (the probes strip comments first, because the repaired
script quotes both surfaces verbatim in its own header comments and a raw-text probe would match
it). **R2** — the C1 fixture was hard-coded while claiming bundle derivation; it now parses
`git_status_porcelain.txt` and **verifies it against the digest the bundle's own `SHA256SUMS.txt`
records for that path**. R3 proved by scratch clone at a post-repair HEAD: 31/31 with the pin,
27/31 with a HEAD-relative mutant.

**Production-observed in attempt 4:** admission PASS, GPU 8/8×3, **pre-dispatch assertion PASS**
(first real firing), rotation landed in ignored `logs/`, tree clean at termination.

**Do not weaken D3.5.** No runtime-residue allowlist, no `.gitignore` exception, no filename
bypass, no weakening of `_repository_state()`. Any future proposal to redefine clean-tree
semantics is a separate contract amendment with its own adversarial evidence.

### 2.36 GATE-12 ATTEMPT 4 — RAN AND FAILED (`distributed_config_t1_c8939b64`, 2026-08-11)

Launched 19:04:14 from `213bfff`, terminal 19:22:53 (~18 min), during **stage 2 / constant
reverse**.

**The error message pointed at the wrong subsystem.** Step 1 saw
`miner.step1_ingress.MinerIngressError` — "no VALIDATED threshold provenance record",
`validated: False`, with `requested`/`payload`/`effective` all agreeing at 0.31 / 0.45. **That is a
secondary symptom.** `provenance_validated` starts False (`:6558`) and flips True only after
`validate_threshold_provenance()` returns; the gate sits behind `if stage_idx >=
len(workflow_stages)` (`:7014`), which stage 2 of 4 never reached. The record is written with
`validated=provenance_validated` on **every** terminal path (`:7132-7134`). **D6 is EXONERATED —
do not modify it.** `validated=False` was truthful and the refusal was correct fail-closed
behaviour.

**P3 was decidable, not inferential:** the validator **NEVER RAN** — zero hits for `threshold
provenance violation`, no `ThresholdProvenanceError`, `terminal_class ≠ TC_THRESHOLD_PROVENANCE`,
and the observed exception was `MinerIngressError` rather than the `raise primary` at `:7062`.

**PRIMARY DEFECT — STALE LOOP-TIME COMPUTE LEASE ORIGIN.** First ERROR in the log (line 151,
19:22:52.014, *preceding* the D6 traceback): `[F1/F2] TRIAL TERMINAL … class=compute_lease_expiry`,
stripe `st1_s30`, worker `zeus-ubuntu-vm:gpu0`, attempt 0. `st1_s30`/`st1_s31` were claimed
19:21:45.667–19:21:47.676 but both carry `lease_expires_at = 19:22:13.373838` — **identical to the
microsecond**, i.e. one shared origin of `19:17:13.373838 + 300.0`. **The iteration's clock was
272.3 s old; both leases were born with ~26 s of 300 s left**, roughly 91% of the budget already
spent, and both produced zero shard rows. `schedule_pending_stripes()` documents that the lease
starts at handoff but accepts a caller `now` (`:2984`, `:3024`) and stamps
`now + compute_lease_timeout` at `claim_stripe` (`:3057-3059`); the serve loop captures
`now = time.time()` once (`:6577`) and passes it down (`:6999`) without refreshing.

**F1 status: NARROWLY REOPENED — lease-origin invariant ONLY.** Attempt 4 *reconfirmed*
one-active-claim (held across all 517 samples), pending/backlog ownership, the expiry matrix, and
**F2 terminal observability (CLOSED / CERTIFIED / production-observed** — class, stripe, worker and
attempt all logged *and* durably persisted).

**Beta prescribed the mechanism** (withdrawing "mechanism is free" after Alpha flagged the shared
timestamp): production stops passing `now=` into `schedule_pending_stripes`;
`claim_now = time.time() if now is None else now` is read **immediately before each
`claim_stripe`**, not once at function entry, so the invariant holds literally even if the
scheduler itself becomes slow. The injected-clock seam is retained for tests. **Do NOT** move or
recapture the serve-loop `now`, change `fail_trial(now=…)`, admission arithmetic, expiry-sweep
timing, or remove `now` generically from shared APIs.

**Six-site audit (required, computed by AST from both sources):** five `fail_trial`
terminal-timestamp paths **unmodified**; one lease seam **modified**. Beta's correction verified
rather than trusted: `process_lease_expiry(run_id, eligible)` takes **two positional args and no
`now`** — Alpha had wrongly listed it as coupled. **Stop-and-report was NOT triggered**, for a
structural reason: the lease was the only site where a *stale* clock computes a *future* deadline
that a *different, fresh* clock later evaluates. Every other consumer compares stale `now` against
a **past** timestamp, so staleness can only make a check fire late, never early.

**Repair status: implemented, 13/13 gates (L1–L7), AWAITING BETA CERTIFICATION. Not committed.**
`miner/range_miner_coordinator.py` +286/−7, exactly three existing defs changed.

**STILL UNKNOWN — the 4.5-minute serve-loop iteration.** No per-iteration timing existed and the
log is empty across the interval; worker logs total 90 bytes each (two kernel compiles plus
`SESSION_END`). **Not recoverable from attempt-4 artifacts.** Beta authorized non-behavioural
instrumentation only; `loop_now_age_max` now measures exactly the quantity that was 272.3 s and is
deliberately kept live **after** the repair so the delay cannot hide behind the fix. The fix makes
the delay harmless, not invisible.

**Worker fault — state the bound precisely.** *Transport/session fault ruled out; no
worker-reported exception observed.* `zeus-ubuntu-vm_gpu0.log` shows `SESSION_END
{"classification": "explicit_shutdown", "assignment_active_at_loss": false, "exc_class": null}` —
a clean shutdown **after** the coordinator terminal. It does **not** reconstruct what s30/s31 did
during their ~26 s leases, or whether they received the assignment at all: **UNAVAILABLE**. Alpha
wrote the broader "worker fault ruled out" and Beta corrected it.

**Independent corroboration of P1 from an unexamined surface:** all 24 rig GPU logs and the Zeus
worker log compiled only `java_lcg` and `java_lcg_reverse` — **no hybrid kernels ever built**,
consistent with stages 3/4 being planned, sized and cohort-frozen at 25 but **never created**
(zero stripe rows, no `derived_bound phase=3/4`).

**Also observed:** saturation SATISFIED again (517 samples, 0 UNOBSERVED) — production evidence,
**not** composable into a Gate-12 PASS. Staging/backpressure exonerated (derived 6528/6528, no
pauses, no capacity timeout, no invariant termination).

**SEPARATE, OPEN, NOT AN ATTEMPT-5 BLOCKER — Optuna raw→canonical quantization.** Optuna sampled
`forward_threshold=0.3057322123717199`, `reverse_threshold=0.4517505335090883`;
`window_optimizer_bayesian.py:560-561` applies `round(…, 2)` at `WindowConfig` construction (the
single-process path, which is what ran — **not** the `…integration_final.py` NP2 partition site,
since `n_parallel=1`). `threshold_provenance["requested"]` is built from the durable trial context,
i.e. **post-quantization**, so the record proves *canonical → payload → effective* and is
**structurally silent on raw → canonical**; `validate_threshold_provenance` never compares the raw
suggestion. **TPE therefore associates objective values with coordinates that were never
executed** — which bears directly on the Optuna study-continuity question. Beta named three
candidate remedies (discrete `step=0.01`; full precision end-to-end; record both
`optuna_suggested` and `canonical_requested`) and ruled **"do not change this yet."** Unconnected
to the attempt-4 failure.

**Evidence frozen:** `/home/michael/gate12_attempt4_20260811_193426` (11/11) and
`…_riglogs` (29/29).

### 2.37 THE FIVE ATTEMPTS, AND WHAT THE PATTERN SAYS

```
Attempt 1  FAILED — bulk lease aging (scheduler/lease architecture)
Attempt 2  FAILED — transport-session recovery gap (stage 3→4, 23/25)
Attempt 3  FAILED — dirty-tree admission; ALL FOUR STAGES COMPLETED FIRST
Attempt 4  FAILED — stale loop-time lease origin, during stage 2
Attempt 5  FAILED — worker_admission_timeout 23/25 at stage 3→4; drain
                    monopolization and reader-exit provenance collapse exposed
D6 dry run STOPPED AT STEP 4 — rigs running mixed-vintage stale code (§2.41)
```

Nothing has failed twice, and the failure keeps moving through different subsystems — but **two of
the five are lease defects** (aging, then stamping). Beta's L5 gate exists precisely so the
attempt-4 repair cannot reintroduce attempt 1's bulk-claiming defect.

**What is proven about RANGE-MINER's compute path:** 25 GPUs simultaneously compute-active with work
queued, real turnover under full occupancy, all four stages across the full `[0,2^31)` domain, no
transport collapse, staging controlled with zero capacity terminations, no GCVM_L2 fault. **The
compute path works.** Every failure since has been in the governance, lifecycle and **deployment**
machinery around it.

**[v25] And the deployment half was never checked at all** until the D6 dry run — see §2.41 and the
meta-lesson in §2.44. Attempts 3/4/5 ran against a rig binary that predates several certified
repairs, which does not invalidate their kernel results but does narrow what they are evidence *of*.

### 2.38 GATE-12 ATTEMPT 5 — RAN AND FAILED (`distributed_config_t1_7e0d020b`, 2026-08-12)

Launched 17:38:44 from `2b0d2dc`, terminal 18:59:53. **All three prelaunch gates passed** — clean-tree
admission, GPU 8/8×3, and the **pre-dispatch assertion in its first production firing**.

**Terminal:** `worker_admission_timeout` — *"stage 3 (family 'java_lcg_hybrid_reverse', phase 4)
expected 25 eligible worker(s); 23 admitted after 180.0s."* Same signature as attempt 2. Phases 1–3
completed 32 stripes each; phase 3 reported **276,439** survivors. Trial
`W12_O49_S10-55_FT0.46_RT0.47`.

**F1 lease-origin held** — no `compute_lease_expiry` anywhere in the run.

**The instrumentation built two days earlier worked on first use.** `[S172-SL] summary`:
`loop_seconds=4856.592 iterations=12300 iteration_max=940.971 drain_max=940.856
loop_now_age_max=940.957 drain_total=3527.128 msg_total=2355.951 msg_max=5.974
unattributed_total=15.388`. **A single iteration lasted 15 min 40.971 s, of which 940.856 s was
inbound drain**; everything outside drain accounts for ~0.115 s of it.

`msg` is nested inside `drain` (R1.2), so `msg_total` vs `drain_total` means two-thirds of drain time
was per-message handling and **no single message blocked** — a cumulative monopolization shape, not
one pathological message.

**Forensic P0–P3 (frozen bundle `/home/michael/gate12_attempt5_20260812_190715`):**

- **P0** — the two missing identities, named from the record: `rrig6600c:gpu2` and `rrig6600b:gpu3`,
  `WORKER_DISCONNECTED` at 18:27:10.537/.539, stepping eligible 25→24→23.
- **P2** — the stalled iteration ran **18:11:28.109 → 18:27:09.080** (START semantics confirmed from
  source). It does **NOT** overlap the admission window — it precedes it by 29 min 44 s — but its
  **end is 1.46 s before the disconnects**. `940-s drain starved phase-4 admission: REFUTED.` What is
  confirmed: the drain **delayed when the serve loop could observe reader termination**.
- **P3** — the discriminating measurement is ESTAB, sampled every 2 s independently of the stalled
  loop: 2420/2420 OBSERVED, no gaps. Both sockets **ESTABLISHED until 18:27:09.113 and gone by
  18:27:11.122**, never passing through 24, with `_drop_conn` logged inside that interval.
  **Alpha wrote "the loss was coordinator-initiated on healthy sockets; worker death, network
  partition and peer-initiated close are ruled out." Beta REJECTED that as an overclaim** — there is a
  ~1.4 s gap between the last ESTAB sample and `_drop_conn` in which a `ConnectionError`, `OSError` or
  decode failure is not excluded. The certified finding: **the coordinator performed the final close
  and eviction; the antecedent cause is UNRESOLVED.**
- **P1** — rig-side session history **UNAVAILABLE**. All 24 rig logs byte-identical, stopped at
  18:02:12, zero session events. Buffering was **refuted by a positive control**, not assumed.

**THE ATTEMPT-5 INITIATING READER-EXIT CAUSE REMAINS UNRESOLVED** and must stay labelled unknown
until a future run reports it directly.

### 2.39 THE TWO DEFECTS ATTEMPT 5 EXPOSED

**SERVE-LOOP DRAIN FAIRNESS / MONOPOLIZATION — CONFIRMED.** `while drained < 256` is a **count**
bound, not a **latency** bound. One drain owned the loop for 940.856 s while the control plane —
deadline handling, stage/admission maintenance, `schedule_pending_stripes`, dispatch, lease expiry,
stage advancement — could not run.

**READER-EXIT CAUSE PROVENANCE COLLAPSE — CONFIRMED.** `_conn_reader_loop` had **nine** semantically
distinct exits all funnelling into one bare `("eof", rawsock, None, None)` with **no reason field and
no log line**. The next layer could say *worker X disappeared* but never *why reader X stopped*. Same
architectural class as the earlier F2 failures: causal information destroyed before the authoritative
observer receives it.

**A structural hazard found during design, worse than either:** `except _queue.Full: break` fires
*because* the queue is at maxsize — then the reader tries to put the eof on **that same full queue**
with a shorter timeout and swallows the failure. For a registered connection nothing reaps it: reader
gone, socket open, worker still counted eligible. **Queue saturation could masquerade not just as
transport failure but as nothing at all.**

### 2.40 ATTEMPT-6 REMEDIATION — CERTIFIED, committed `69ff222`

Three design cycles (R1/R2/R3) and two implementation cycles (R1/R2) before certification. Gates
**78/78**, ten operative gates, four pinned RED arms, seven mutants.

**Part A — reader-exit provenance.** Ten reason constants with `READER_EXIT_UNCLASSIFIED` as the
fail-closed default; `ReaderExit` + `ConnState` carrying a **run-scoped `connection_id` (never
`fileno()`/`id()`** — both are reused in a long process); the reasoned EOF **stays on the same inbound
FIFO** with bounded retry (a separate control queue was **rejected**: it could reap a connection ahead
of envelopes that same reader delivered, hitting the `rawsock not in fs_by_sock` discard and
destroying work the F1-R credit machinery exists to preserve); the silent `timeout=0.5; except: pass`
swallow removed; `_drop_conn` emits **`CONNECTION_CLOSE_INTENT` as its first statement, bound or
not**.

**Three orthogonal facts, never fused, no causation claimed:**
```
CONNECTION_CLOSE_INTENT  = coordinator decision
READER_EXIT              = reader observation
WORKER_DISCONNECTED      = bound-worker identity eviction
```
Any subset may exist honestly depending on chronology. `_drop_conn` mutates identity maps and emits
`WORKER_DISCONNECTED` **before** `shutdown()`, so a coordinator-originated drop carries
`reader_exit_reason=UNOBSERVED` — a later reader event **cannot retroactively join an already-emitted
record**.

**Persistent ingress saturation is a whole-trial infrastructure terminal** (`TC_INBOUND_SATURATION_TIMEOUT`)
over a SimpleQueue emergency channel — **no worker is ever shed** for a coordinator capacity problem.
`S` is **cumulative per connection** and charged **only** from the two real `_queue.Full` paths;
successful puts never reset it. Register-disposition waits, staging pause, the pre-decode barrier and
admission-queue residence are **NOT** ingress saturation: *"did this wait happen because a bounded
queue refused the item?"* Only `inbound` is bounded, so the false terminal is closed **by
construction**.

**Part B — control-plane fairness.** Monotonic drain deadline `D` with 256 retained as secondary
guard; accept-poll folded into the same budget; **the first `get()` clamped by the remaining budget**
(with `D=0.05`/`poll=0.10` the drain could otherwise block 0.10 s on an empty queue). First-frame
REGISTER priority on a bounded admission channel (`D_adm` + `A_max`, deadline tested **only from the
second disposition** so one disposition is the progress floor and `A_max` is a ceiling, never a
guaranteed rate), per-connection fence, eligibility **consumable exactly once per connection**.

**The certified claim is structural, NOT a wall-clock guarantee:**
```
drain contribution <= D + one in-flight message runtime
T_cp = A + D + M_i + K_i      (M_i, K_i are MEASURED runtimes, not bounds)
```
**6.42 s and 6.7 s are attempt-5 counterfactual estimates, not production guarantees.**

**Worker + harness:** `prepare → sentinel → BARRIER → connect → register → serve`;
`scripts/gate12_sentinel_gate.py` verifies **25/25 same-record `SESSION_SENTINEL` + current nonce**
before any REGISTER.

### 2.41 THE STALE-RIG DISCOVERY — WITHDRAWS SEVERAL PRIOR FACTS (2026-08-14)

**The D6 parked-fleet dry run stopped at step 4.** All 25 dispatch lines printed; **all 24 rig
workers died instantly**: `range_miner_worker.py: error: unrecognized arguments: --run-nonce
--session-release-file --release-deadline --sentinel-log-path`.

**The rigs were running mixed-vintage stale code — not any single commit:**
```
prng_registry.py         identical      sieve_gpu_worker.py    identical
miner/__init__.py        identical
range_miner_worker.py    vintage 2026-08-01   (1,524 lines vs 2,178)
range_miner_protocol.py  vintage 2026-07-29
```
Two files **ten days apart**. *"Which commit are the rigs on"* **has no answer** — which is itself the
strongest argument for digest-only parity evidence. The live parity gate later found two more:
`miner/range_miner_coordinator.py` differing on all three rigs (255 KB vs 563 KB) and
`execution_set.py` **absent on all three** → `18 MATCH / 12 MISMATCH / 0 UNAVAILABLE → REFUSED`.

**WITHDRAWN OR NARROWED — do not restate the old forms:**

| claim | disposition |
|---|---|
| Zeus and the rigs run identical code | **WITHDRAWN** — the assumption was never justified |
| Attempts 3/4/5 exercised Defect A in production | **WITHDRAWN** — no recovery code was deployed; 29/29 VM101 certification stands, fleet exercise **NOT ESTABLISHED** |
| Rig `elapsed_s` from attempts 3–5 | **LEGACY / PROTOCOL-CONTAMINATED, not service-time evidence** — the rigs never got `4dd5535` (`float = 0.0` → `Optional[float] = None`), so every rig stripe reported a literal `0.0` instead of omitting the field, collapsing R4's "not reported vs measured zero" on the wire |
| Attempts 4/5 rig-log silence "UNRESOLVED / channel broken" | **stale-deployment cause STRONGLY SUPPORTED / PRESUMPTIVE**; historical rig bytes **NOT PROVEN** |
| Attempt 3's §21 status | **DEFINITIVELY FAILED** — four stages and saturation are valid *mechanical* evidence, but D3.5 refused publication, so **no S145 coverage, no cursor advance**. The miner trial reaching an internal `committed` state **does not override the D3.5 authority wall** |

**NOT affected:** the kernel-bearing files hash identically, so **no survivor count is in question**.
The correct wording is **"not invalidated by this finding,"** never "proven unaffected" — historical
rig source hashes were never captured.

**Attempt 4's and attempt 5's primary terminals are unaffected** — both are coordinator-side
observations.

### 2.42 THE D6 INTEGRATION REPAIR — CERTIFIED, committed `dd03f1d`

**`scripts/gate12_parity_gate.py`** — a fail-closed rig source-parity wall, in `gate12_launch.sh` §0.6
(after the GPU gate, before the clean slate) and D6 STEP 1.5. Expected values are **full 64-hex
SHA256 derived from the local tree at run time**; the 12-char display prefixes appear nowhere as
comparisons, enforced by an AST arm. **`local HEAD` prints as `[CONTEXT ONLY]` and `evaluate()`
provably never reads it** — Git identity must never substitute for content identity. Any mismatch,
missing file, malformed output or unavailable SSH is **REFUSE, never advisory**.

**The governed set is a TEN-file pin — Beta REJECTED narrowing it to a call-graph closure**, because
"does this execution happen to reach this file?" depends on arguments, branch paths, future code,
failure handling and deferred imports, while static project-local reachability is provable. The
correct term is **"statically reachable project-local import/deployment closure"** — `execution_set.py`
is statically reachable but **not executed on today's normal worker path**, and is governed anyway:

```
adaptive_thresholds.py · execution_set.py · hybrid_strategy.py · miner/__init__.py
miner/dataset_authority.py · miner/range_miner_coordinator.py · miner/range_miner_protocol.py
miner/range_miner_worker.py · prng_registry.py · sieve_gpu_worker.py

ACCEPTANCE: 30/30 MATCH · 0 MISMATCH · 0 UNAVAILABLE
```

**Launcher wait-set correction.** `launch_fleet_manual.sh` blocked on its own local worker, which was
parked at the release barrier waiting for a token `gate12_launch.sh` only writes *after* the launcher
returns — a wait cycle that could not complete. The invariant:
```
launcher completion  =  all DISPATCH operations dispositioned
launcher completion !=  all launched worker PROCESSES exited
```
Remote dispatch jobs are now waited individually with status **read, not discarded**; the local worker
is excluded and **asserted alive** before return. Release was **not** moved earlier.

**Two committed ordering gates were found encoding a superseded launch order** — `P2-REFUSAL-PRECEDES-SAMPLER`
(8/9) and `W-ADMISSION-FIRST` (30/31), both already red at `69ff222`, asserting `coordinator < fleet`
which the attempt-6 architecture deliberately inverted. **Neither suite was in the R2 regression
table**, which is how they went unnoticed. Rewritten as **fans rather than chains**: each arm states
what it owns *and what it deliberately does not*.

### 2.43 D6 DISCHARGE REQUIREMENTS — the parked-fleet dry run

**A sentinel record and process liveness are independent facts. A dead worker with a valid sentinel
is not a parked worker.** All of these must hold simultaneously, before cleanup:

```
GPU truth gate                    PASS
source parity                     30/30 MATCH
dispatch                          25 dispositioned
launcher                          returns while the fleet remains parked
remote processes alive + parked   24/24
local process alive + parked       1/1
current-nonce SESSION_SENTINEL    25/25
REGISTER before release              0
release tokens                       0
coordinator / port 5700           absent / unbound
```

Then kill the parked fleet **without release**; confirm no new `run_id`, no coordinator, no token,
zero workers. **The nonce must be fresh every time** — remote logs are truncated by the shell redirect
on each launch *only if the dispatch lands*, so a reused nonce could let a stale log satisfy the gate.
Burned nonces stay burned: `prelaunch-d6-20260814_191543-14495`.

The D6 dry run earns **zero §21 credit**. Attempt 6 starts with a **fresh production nonce** and must
satisfy the full seven-part completion authority **in that single run** — nothing composes from
attempts 1–5 or from the dry run.

### 2.44 META-LESSON — A PRINCIPLE WITH NO ENFORCING GATE IS NOT A CONTROL

**§2.17 already said the rigs are deployment targets, not working copies, and that digest comparison —
never `git rev-parse` — is the parity evidence.** That rule was written down, and **three full
Gate-12 attempts ran without anyone checking.** The rule was not missing. The **enforcement** was.

That is why Beta made parity a **hard prelaunch wall that refuses** rather than a documented
expectation. Apply the same test to every other principle in this skill: *what would fail if this
were violated?* If the answer is "someone would have to notice," it is a hope, not a control.

**The companion lesson, six instances deep:** a gate that passes on a fact it does not actually check
is worse than no gate. Recorded instances — a test asserting `iterations == 2` yields
`iteration_count == 1`; an arm asserting only field *names*; two independent existentials standing in
for a conjunction; `loop_now_age_max >= 0` satisfied by a constructor zero; a declared SSH outcome
exercised only via a local file; and `_mutant_red` crediting *any* exception as detection, so
`MUTANT NOT APPLIED` read as `MUTANT DETECTED` **inside the machinery meant to prove the arms are not
vacuous.** Every one was green. Every one was found by reading the assertion, never the tally.

## 3. SUPERSEDED — in repo, NOT current

> **The full register is `docs/PROJECT_FILE_CATALOG.md` §6.** §6.1 superseded facts and targets —
> **with a *cite-instead* column and a source per row, which v12's bare list here did not have**;
> §6.2 the v1 catalog's Runtime Data table (**wrong in every row**); §6.3 superseded document
> **versions** — 17 lineages, *read the last one only*; §6.4 whole-document staleness warnings.
> **Verified this session: §6.1 carries 11 of v12's 14 entries with equal or better anchoring, and
> four more v12 never had.**

**Retained here — catalog §6 does NOT carry these three:**
`window_optimizer.py:450` **docstring** · **"the writer is unconditionally frozen"** (D6 added one
approved `backend=None` seam) · `feature_importance.py`'s 60-name list being **stale by 31**
(§6.1 names the list; the arithmetic against 91/89 is only here).

**⚠ CORRECTED — v12 listed `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` as a superseded "fragment".
It is NOT, and that entry was actively harmful.** The 34-line fragment at `248e48c` was
destroyed-and-restored; the live file is **1,463 lines, audited, CLOSED at `ef4b1c6`** with content
gate `09bbfbf`, and is **the strongest of the three audited chapters** (§2.17b). **Its §6 carries
the three-lane CRT proof** — the very thing Alpha once claimed was undocumented. Read as written,
v12's line steered a session away from the chapter that answers the question.

**⚠ And do not take the catalog's commit here — `PROJECT_FILE_CATALOG.md` §2 gives Chapter 2's
closure as `81ef3f1`. It is wrong.** Verified this session: `ef4b1c6` is the commit that edits
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (+269) and `CHAPTER_1_WINDOW_OPTIMIZER.md` (+698);
`81ef3f1` adds only the **closure brief** that commissioned the work and touches neither chapter.
*The index is authoritative on what exists, not on every commit it cites — §1.2 applies to it too.*

## 4. FROZEN — reuse, never reimplement
- **`canonical_map_hash()`** (`utils/run_finalizer.py:486`, **exported**) — SHA-256 over canonical
  JSON of `ENCODING_VERSION` + `PRNG_TYPE_ENCODING` + `SKIP_MODE_ENCODING`. **`ENCODING_VERSION`
  alone is insufficient:** `tests/test_prng_encoding.py` pins `len(PRNG_TYPE_ENCODING) == 44`, so
  **renaming a registry key preserves both the count and the version string while renumbering every
  id after it alphabetically.**
- **The three pre-L2 protections**, live order at `utils/run_finalizer.py:1606-1611`:
  `_validate_raw_candidates` (`:665`) · `_validate_candidate_coverage` (`:558`) ·
  `_validate_candidate_identity` (`:634`). **All private, none in `__all__` — import anyway or
  extract; never fork.**
- **`utils/prng_encoding`** — the string↔uint8 codec. Exists because **three divergent hardcoded
  dicts once collapsed unknown/hybrid `prng_type` to `0`**, destroying provenance.
- **`_l2_sort_key` / `_select_l2_winners`** (`utils/run_finalizer.py:690`, `:714`, Ruling D):
  highest **float32** score → lowest `trial_number` → constant-before-variable *within a trial
  only*; same-trial/same-mode collision raises `AccumulatorConsistencyError`. Comparing
  pre-rounding float64 **is the defect this converts away.** Import; never fork.
- **D3.5 finalizer-owned root symlinks** — a regular file there makes `finalize_run` raise
  `PublicationError`.
- **D5 §6.7.A compressed-artifact ban** — scoped to worker *transport* artifacts. The D6.1
  **checkpoint may be compressed**; deliberately separate. Do not harmonize.
- **The 22-array NPZ contract**; the D3.25 four-map ingress contract.
- **`distributed_config.json` bare-metal addresses** (`.120/.154/.162`) — deliberate, they
  match the *default boot target*. `CLAUDE.md` §3: **not a bug, must not be corrected.**

## 5. VERIFICATION INTEGRITY (VIR-1…6) — binding
`docs/VERIFICATION_INTEGRITY_STANDARD.md`. Adopted after three incidents of *a check that was
not checking, presenting as a pass*.
**VIR-1** verification must prove its own execution; silence/truncation/reporter death/an
inaccessible surface is never a pass. **VIR-2** vacuous-capable detectors need execution proof
· **clean control** · **fault-injection (positive) control** · detector independence (*not
interchangeable terms*). **VIR-3** terminate in `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only
`PASS` accepts. **VIR-4** cleanup must not kill its reporter. **VIR-5** unobservable is not
clean. **VIR-6** audit scope must match the claim; declare searched **and** unavailable
surfaces.

> **⚠ VIR-6 ADDENDUM — `docs/` IS A MANDATORY SEARCHED SURFACE.**
> For **any** claim about intent, design rationale, absence, "why does this exist", or whether
> something is known — **`docs/` and the governance trail must appear in the declared searched
> surfaces.** A declaration listing only code surfaces is **INCOMPLETE ON ITS FACE**, and a
> reviewer must reject it on sight.
>
> *This exists because of a specific failure.* Chapter 3's audit declared six searched surfaces —
> gitignored files, `git log`/`git show`/`git log -S`, `git check-ignore`, the live VM101
> filesystem, live Python imports, live execution of `run_trial`. **`docs/` was not among them.**
> The declaration read as rigorous and Alpha approved it. In the same session Alpha claimed the
> three-lane CRT test was undocumented **while `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §6 sat
> committed**, and nearly submitted to Beta a "finding" **Beta had already ruled on** in
> `docs/TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md`.
>
> **The convention treated code as the world and documentation as commentary. It is the reverse
> that is true: the code says what it does; the documents say what was decided.**

Gates should extract and execute the **live source** of the call site (AST), not match text —
`2389b61` reverted a fix by whole-block replacement; a text anchor would have gone green.

Every brief carries:
```
Verification-integrity controls (VIR-1…6):
- execution proof:      - clean control:      - fault-injection control:
- completion sentinel:  - unavailable-observer behavior:
- audit claim scope:    - searched surfaces:  - unavailable surfaces:
- governance trail searched (TB_RULING*, PROPOSAL*, TEAM_ALPHA*):  - chapters searched:
```

## 6. TOPOLOGY (verified 2026-08-01)
Rigs boot **bare-metal by default**; currently in **Proxmox**. `host = rig+1`,
`CT100 worker = host+1`.

| rig | bare-metal | Proxmox host | **CT100 worker (use this)** |
|---|---|---|---|
| rrig6600 | `.120` | `.121` | **`192.168.3.122`** |
| rrig6600b | `.154` | `.155` | **`192.168.3.156`** |
| rrig6600c | `.162` | `.163` | **`192.168.3.164`** |

Key auth works **from VM101**, not ser8. All three CTs: `~/rocm_env`, **8 × RX 6600 XT**, cupy
13.5.1, gfx1032, **no HSA/GFX overrides needed**. **Venvs differ** — VM101 `~/venvs/torch`,
rigs `~/rocm_env`. CT100 is an **unprivileged LXC**: GPU kernel log must be read from the
Proxmox host (`root@.121`). `daily3.json` is **gitignored** — clone alone can't stand up a rig.
**All three rigs are provisioned with the frozen dataset** (P0.5, verified on target).
`dataset_provisioning.json` is also gitignored — a fresh clone has no fleet definition.

## 7. WORKING AGREEMENTS
- **Claude = Team Alpha** (lead dev). **Team Beta** = separate approval authority; rulings
  binding; never impersonate. **Alpha may contest with evidence** — Ruling 20 was withdrawn
  that way — but does not overrule.
- **Never commit or push from the sandbox.** Deliver to `/mnt/user-data/outputs/`; Michael
  downloads to ser8, `scp`s to VM101, commits, dual-pushes.
- **EVERY command must name its host.** ser8 = download target and `scp` source. VM101 = repo,
  all `git`, all rig SSH. *The single most repeated operational failure.*
- **`source ~/venvs/torch/bin/activate` before any test command.** A bare shell yields
  `CuPy not available` / `Optuna not available` — false reds. Watch for `(torch)`.
- **Plans for behaviour-changing work go to Beta before implementation.** P0's procedural
  exception was granted because it was inert; Beta stated it is **not precedent**.
- **AN INHERITED ABSENCE CLAIM IS STILL AN ABSENCE CLAIM — SEARCH BEFORE AMPLIFYING IT.**
  (Owner correction 2026-08-08, binding.) Alpha applies the §0.1 search-first rule to claims it
  originates but NOT to claims it inherits from a Claude Code report, a Beta ruling, or a
  document. Evaluating a finding is not verifying it; interesting-ness substitutes for
  verification. **Before carrying any report's gap into a submission, a design decision, or a
  statement to the owner, run the search yourself.**
  - **A finding scoped to ONE source is not a finding about the repo.** "The document doesn't
    decide it" is a statement about that document. The binding search order (governance trail →
    chapters → code) applies to the inherited claim exactly as it does to an original one.
  - **Four instances in a single session, 2026-08-08:** (a) asserted midday/evening were separate
    machines — the CA procedures §II specify per-draw random equipment selection; (b) claimed the
    seed cursor had "overshot a 2³² target" — Beta had already deauthorized the whole tracker;
    (c) called `offset.max = 100` unexplained scaffolding and then proposed `n − window_size` —
    Chapter 2's **F-4** makes it a structural coupling (one scalar drives both the history slice
    and the generator pre-advance), not a bound edit; (d) carried forward an "unable-to-succeed"
    concern about Daily 3's three-selection spec that the repo had already resolved at
    `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:152` — digit features per S119 / spec
    `03:00-09r`, additive alongside the CRT lanes (`survivor_scorer.py:616-617`, `:426-428`),
    with the kernel deliberately sieving the published value as ONE residue
    (`CHAPTER_2:697` — "mod 1000 | full published value").
  - **Each was caught by the owner, not by Alpha.** If the owner is performing the verification,
    the rule is not being followed.
- **Stage explicitly, never `git add -a`.**
- **OWNER RULE (2026-08-06, binding): when Beta offers multiple acceptable mechanisms,
  Alpha takes the structurally STRONGER one — the one whose properties hold by construction
  rather than by inference. Diff size is NEVER a tiebreaker against correctness-by-
  construction.** Origin: S172-BP round 2 — Alpha chose the smaller acknowledgment design
  over the physical reservation; its two inferred properties each cost a full review round.
  A review round costs days; a bigger diff costs hours.
- **A comment asserting a concurrency property is a claim, not a proof.** Trace the
  interleaving against an unconsumed resource, or demand a gate that drives N waiters
  through ONE release. Concurrency changes need TRANSITION gates (the pause→resume
  boundary), not only state gates — F2 lived exactly in the boundary G-LEASE never crossed.
- **Differential-worktree regression proof:** attribute suite reds under an uncommitted
  diff by running the suite in the SAME environment on the patched tree and on
  `git worktree add <dir> <base>`, then diffing the pass/fail lists. Only the differential
  is chargeable to the change.
- **Committed suites carry a host assumption:** gates inheriting the 16 GiB
  `staging_high_water_bytes` default red on hosts with a smaller free `$TMPDIR`
  (Part B one gate; phase-4 Gate 54). Pre-existing; not regressions; new gates must set an
  explicit high-water.
- **Freshness probe is a content grep, not `head -1`:** revised documents rarely change
  their first line — probe a marker unique to the current revision
  (`grep -c "<current-hash>" <file>`). The one-document/three-locations shape (sandbox
  outputs → ser8 Downloads → VM101 docs) produced two stale-revision events in one day.
- **`grep` eats leading-dash patterns as options** (`grep "--params" f` → rc 1, silent).
  Use `grep -e '--params'`. Same false-negative class as the pgrep self-match rule.
- **Final-state evidence discipline (Beta §6, standing):** the canonical-host run happens
  AFTER the last change and the report/cover are written AFTER that run — evidence must
  describe the final artifact, nothing earlier.

- **Never reuse a filename when uploading to chat.** Same-named uploads deduplicate somewhere
  upstream and arrive as **empty stubs** — six transfers were lost this way in one session
  before the cause was found. Give every upload a unique name. `.diff` and `.png` were 100%
  reliable; repeated `.txt`/`.md` names were not. **The repository is the fallback channel**:
  commit and push, then the reader clones — that path has never failed.
- Session changelog to `docs/`; dual-push `origin` (private) + `public` (mirror) —
  **everything pushed is effectively public.**
- **⚠ A SKILL REVISION LIVES IN THREE PLACES. Committing it updates ONE.**

  | copy | who loads it | how it updates |
  |---|---|---|
  | `docs/TFM_PROJECT_FACTS_SKILL.md` | nobody at runtime — the tracked source | commit + dual-push |
  | `~/.claude/skills/tfm-project-facts/SKILL.md` | **Claude Code**, on invocation | **manual `cp`** |
  | the Settings upload | **new chat sessions**, at session start | **manual re-upload** |

  **Nothing warns you when they diverge, and they diverge silently.** On 2026-08-03 the tracked
  copy reached **v13** while `~/.claude/skills/` still held **v6** (last touched 00:22 that day,
  before the entire day's work) and Settings held **v11**. **Thirteen revisions, and not one had
  reached a runtime copy.** Every correction made that day protected nothing until the copies were
  fixed by hand.

  **A revision is NOT done at the dual-push. It is done when all three are current:**
  1. commit + dual-push the tracked copy;
  2. **back up** the installed copy (`SKILL.md.bak-vN`), then `cp` the tracked copy over it;
  3. **re-upload to Settings**;
  4. **verify in a FRESH session** — *"print the Currency line and the §0.6 heading verbatim"*.
     **The currency line exists to make this drift visible. It is the only signal there is.**

  **A running chat session cannot be updated at all** — its copy is fixed at session start. **Start
  a fresh session after any revision that matters.**
- Prefer **Claude Code on VM101** for as-built questions — live source **and** live host. A
  clone is repo-only (VIR-6); chat-side reasoning is provisional.
- Briefs: one falsifiable question, a defined deliverable, "write the report to
  `docs/<n>.md`". Separate *investigate* from *fix*.
- **Long suites: `python3 -u <suite> | tee /tmp/<name>.log`, or `nohup`. NEVER pipe to `tail`** —
  it buffers and prints nothing until completion, so a live run is indistinguishable from a hang.
  Check for a **descendant** process before concluding a hang; a blocked parent burns no CPU.
- **Phase-4 Gate 22 builds `changed_py` from `git status --porcelain`, which includes UNTRACKED
  files.** Any new test file reds it and propagates to D5's `NR` arm. **Expected during
  development, not a regression, and NOT a reason to widen Gate 22.** Commit the file. *(Arose
  twice; the answer is the same both times.)*
- **Build a `git add` list from the report's "Files changed" section, never from recall.** *(A
  D6.2 stage list omitted hops 1 and 2 of §2.15 — it would have shipped a `resume_checkpoint`
  unreachable from WATCHER, alongside the gate proving otherwise.)*
- **A whole-file JSON rewrite is the `2389b61` mechanism.** Escaped-unicode churn in untouched
  lines means the file was re-serialized, not edited. Diff the **decoded structures**, not the
  text, before committing.

## 8. APPROVED SEQUENCE
```
D6.1 ✅ · Phase 6.0 ✅ · threshold repair ✅ · Ch1 P0+P1/P2 ✅ · Chain C ✅
6-P0 ✅ · 6-P0.5 ✅ · Q2 closure ✅ · §4.3 liveness ✅ · Wall C struck ✅
bounded Phase 6 ✅ CERTIFIED d98298c — **miner/finalizer path ONLY, see §2.18**
Chapters 1 and 2 ✅ ef4b1c6 + content gate 09bbfbf
Resolved Execution Set ✅ 63e627f · admission binding ✅ eff6616
process_sharded import gate ✅ e0513ba — D5 25/25
D6.2 ✅ CERTIFIED 18a2419 — n_parallel == 1 ONLY
Phase 7 prerequisites ✅ closed on measurement; item 1 WAIVED by owner (25 GPUs)
now    S172-BP F1 round-3 fix-forward (token + pre-decode barrier) → Beta delta
       review → commit+dual-push → gate-12 production shape (4-stripe/25-daemon,
       MICHAEL-INITIATED) → then Phase 7 SOAK. Beta HOLDS gate 12 and the soak.
       Soak spec when authorized: 50 trials, ≥5 high + ≥5 low
       survivor, mixed const/hybrid, 25 frozen identities (bea580e76490),
       n_parallel=1 BINDING, serial_reference. FIRST execution ever with the
       S166 clear enabled — the RAM series across 50 trials IS the result.
       GCVM_L2 criterion reports UNAVAILABLE (§2.17).
       ⚠ --start-step 1 --end-step 1 MANDATORY: --end-step DEFAULTS TO 6, and
       STEP_SCRIPTS[2] reaches run_scorer_meta_optimizer.sh, which invokes the
       TB-prohibited converter and mv's a REGULAR FILE onto the D3.5
       finalizer-owned symlink -> PublicationError, hours in, at publication.
       Launch needs `> log 2>&1` (no FileHandler; confirmations go to stderr).
       Abort signal is "STEP 2: Scorer Meta-Optimizer (run #N)" — NOT
       "Triggering Step 2", which is benign and expected on a clean run.
       No Chapter-13 retrain approval during the soak (chapter_13_triggers.py:630
       carries its own STEP_SCRIPTS, unbounded by --end-step).
next   D3.0-B — OPEN, TB requires completion; blocks legacy-writer use only
       6-P2 (scraper — REV4 with Beta; option (a) BINDING)
       D6.3 (retention — non-blocking, ~10.7 KB/run measured)
       NP2 checkpoint transaction design (NEW, separate)
```

**⚠ Sampler-comparison sequencing — a correction TB issued against Alpha.** The certifying
four-phase TPE-vs-random comparison **cannot** be scheduled merely *"after the skip-output
work."* The approved skip-output work retains observed sequences and restores `skip_mean` /
`skip_std` / `skip_entropy`. **It does NOT connect `skip_min`/`skip_max` to the hybrid kernels**
— that is the separate, unresolved **input-bound** interpretation. The comparison must wait
until **either** hybrid search-input bounds have defined effective semantics, **or** the
comparison uses an **explicitly phase-aware search space that does not pretend dead hybrid
dimensions are active.** Skip-output may proceed first; completing it alone **does not remove
the dead-dimension caveat.**

**TPE remains the production default by status quo** — the five-seed run is a valid constant-skip
datapoint and useful directional evidence, **not** a certification of superiority and **not**
authority for autonomous sampler selection.

**Open backlog → `docs/BACKLOG.md` is the register.** It is maintained; this paragraph is not.
*(Chapter 2 restore-and-audit is CLOSED — `ef4b1c6` + content gate `09bbfbf`. The unaudited
chapters are **3, 5, 6, 8, 13**, not "3–7".)* The `java_lcg_cpu` non-zero-skip mismatch
(`survivor_scorer.py:124` / `full_scoring_worker.py:305`) remains a **separate bounded audit, no
fix authorized** — see the Wall C caution below.
**Wall C caution:** `java_lcg_cpu` (`prng_registry.py:170-183`) applies skip **once before
generating**; the kernel applies it **between every draw** (`:987-989`). They agree only at
`skip=0`. **Building the known-answer reference on it would validate the wrong semantics** in
the deliverable meant to catch semantic error. *(Michael reports all 44 PRNGs were validated
through the sieves during pipeline development — constant forward/reverse and hybrid
variable-skip. An inventory is establishing what exists before anything is scoped as new.)*

**Hard Phase-7 prerequisites — SATISFIED.** 6-P0 ✅ · 6-P0.5 ✅ · D6.2 ✅ CERTIFIED `18a2419`.
**D6.3 and 6-P2 are NOT Phase-7 blockers** (TB): D6.3's growth is ~10.7 KB/run measured, and the
miner-backed soak does not invoke the scraper. **D3.0-B is OPEN but blocks only legacy-writer use**
(§2.18). **The soak is HELD by Beta pending the S172-BP F1 round-3 delta (§2.19)** — the earlier
"authorized and launching" state was interrupted by the 2026-08-05 staging incident.
**Optuna:** constant-skip **may resume**; hybrid exploration **non-certifying only**; hybrid
certification **blocked** until skip bounds are live; authoritative studies need provenance
binding.

## 9. SELF-CHECK BEFORE SENDING
1. Claimed something missing/broken/unwired/current? → anchor or **[UNVERIFIED]**.
2. **Proposing to remove, demote or simplify anything? → did I cite the doc explaining why it
   exists (§0.4)?**
3. **Did I READ the grep hits, or only count them?** (§1, fifth corollary)
4. Cited a metric or target? → check §3 **and catalog §6**.
5. Proposing something that already exists? (coverage metric, downstream_score, attribution
   engine).
6. Classified a capability from one module without tracing producer → artifact → consumer?
7. Changing a shared buffer, path or format? → enumerated every consumer?
8. System-scoped claim on repo-scoped evidence? (VIR-6)
9. **Named the host for every command? Included the venv activation?**
10. **Did I write "Step N" without saying which scheme — or read someone else's "Step N" as
    mine?** (§0.6) *The two schemes agree at 1, 3, 5, 6 and differ only at 2.*
11. **Did I search `docs/` and the governance trail — or only the code?** (§1.1, VIR-6 addendum)
12. **Am I relaying a dated document's finding as current state?** (§1.2)
13. **Does this contradict something I said earlier in THIS session?** *(Alpha stated Chapter 2 was
    closed and restored, then hours later reported its content as lost.)*
14. **Writing a rule? Enumerate every case in its input space and state the behaviour for each
    BEFORE submitting.** A rule validated only against the case that motivated it is untested.
    *(6-P2 REV3's terminal-day predicate was written for `{midday}` and never tried against
    `{evening, midday}`, where it defers forever.)*
15. **A ruling that says "decide X" is decided in THAT revision, not the next one.** *(Twice in
    D6.2.)*
16. Long thread? Verification discipline degrades — suggest a fresh session.
17. **Target: any brief closes in ≤3 review rounds.** D6.2 took five, 6-P2 took four, and **every
    round was an Alpha defect, not reviewer padding.** The import gate closed in one — because the
    existing gate was read in full before the brief was written. **Read first, then draft.**
