---
name: tfm-project-facts
description: Foundational model, verified as-built facts, superseded-artifact list, and mandatory verification procedure for Michael's TFM (Triangulated Functional Mimicry) distributed PRNG analysis project. Use this skill whenever the conversation touches TFM, the PRNG cluster, RANGE-MINER/S172, selfplay, Chapter 13/14, the bidirectional sieve, survivor pools, the NPZ contract, prediction pools, WATCHER, Zeus/VM101/rrig6600, or any file in the prng_cluster repos — even if the user does not name the project explicitly. Use before making ANY claim that something is missing, broken, unwired, unused, or current, and ALWAYS before proposing to remove, demote or simplify any component.
---

# TFM — Foundations, Verified Facts & Verification Procedure

**Currency:** v13, HEAD **`1131bb1`** (2026-08-03), clean tree. **D6.2 CERTIFIED `18a2419`.**

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

### 2.7 Recurring defect: tuned parameters don't reach kernels — SIX instances

| # | instance | status |
|---|---|---|
| 1 | miner filtered at hardcoded `0.25` | **FIXED** `2be51d5` — single canonical path, per-direction resolution in the parent, effective value read off the executor, parent-side fail-closed provenance |
| 2 | Optuna thresholds dropped above `run_bidirectional_test`; every trial ran `0.30/0.30` | **FIXED** `8a55a68`. Was a **regression**: fixed `3fdf434` (04-30), silently reverted `2389b61` (07-07) by a stale-copy overwrite whose message never mentions thresholds. Both routes now use `resolve_directional_threshold()`, `is None` not truthiness (**0.0 is legitimate**) |
| 3 | PWC hybrid filtered at `0.50` | **QUARANTINED** — `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`; PWC non-certifying, so the defect is made loud rather than repaired |
| 4 | **hybrid kernels ignore sampled `skip_min`/`skip_max`; `expected_skip = 5` hardcoded** | **OPEN.** 22/22 constant kernels declare skip bounds; 0/22 hybrid do. Values survive eight hops and **die at `_hybrid_prefix`** (`range_miner_worker.py:177-193`). Anchors `prng_registry.py:1027, :805, :885, :1159`. **Semantics ARE documented (§0.4)** — the "unspecified semantics" premise of `HYBRID_SKIP_BOUND_AUDIT.md:318` is **FALSE**. Decision open; **the output-statistic reading needs no kernel change at all** |
| 5 | forward hybrids ignore `offset` (sampled `window_optimizer_bayesian.py:423`) | **OPEN.** Chapter 2 F-4: `offset` drives **both** the host residue slice **and** the device pre-advance from one payload scalar — coherent only at `skip=0`. Settles Chapter 1 C-2 as an **observed inconsistency, not a repair**; belongs in the future hybrid input-semantics design, **not** a standalone arithmetic patch (TB) |
| 5b | **`recommended_window_size` → Rule A** — the manifest declares `8`, the code reads it into `_rec_ws` (`window_optimizer_bayesian.py:500`) and **never references it**; Rule A uses a hardcoded `32` | **ROOT CAUSE FOUND.** `TRSE_INTEGRATION_PLAN_S121.md` §2C specifies `min(rec_ws * 4, …)` — and `8 × 4 = 32`. **The value is correct; the wiring is missing.** Not "a field of unclear purpose" — a **configurable input frozen at its default by a literal** |
| 6 | **`skip_learning_rate`** configured 0.2–0.7; kernel **hard-adapts at 1.0** | **OPEN**, newly catalogued |

Fix pattern: **one canonical path** — resolve once in the parent, never reinterpret
downstream, record requested/payload/effective.

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
- **Stage explicitly, never `git add -a`.**
- **Never reuse a filename when uploading to chat.** Same-named uploads deduplicate somewhere
  upstream and arrive as **empty stubs** — six transfers were lost this way in one session
  before the cause was found. Give every upload a unique name. `.diff` and `.png` were 100%
  reliable; repeated `.txt`/`.md` names were not. **The repository is the fallback channel**:
  commit and push, then the reader clones — that path has never failed.
- Session changelog to `docs/`; dual-push `origin` (private) + `public` (mirror) —
  **everything pushed is effectively public.**
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
now    Phase 7 SOAK — LAUNCHING by owner order. 50 trials, ≥5 high + ≥5 low
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
(§2.18). **The soak is authorized and launching.**
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
