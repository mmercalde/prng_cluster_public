# CLAUDE CODE REPORT — WHAT THE PIPELINE DOES, END TO END (READ-ONLY)

**Date:** 2026-08-08 · **Host:** VM 101 `zeus-ubuntu` (`192.168.3.177`) · **Tree:**
`/home/michael/distributed_prng_analysis` · **HEAD:** `8bbe79e` (verified `git rev-parse`).
**Type:** read-only. Nothing launched, nothing edited, nothing committed. One file written: this one.
**Search order followed:** governance trail → chapters → code (binding).
**Index read first:** `docs/PROJECT_FILE_CATALOG.md`. **Second mandatory read:**
`docs/PIPELINE_BEHAVIOUR_MODEL.md` (1,617 lines).

**Relationship to the two prior reports.** `CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md` establishes
what Step 1 is *for*; `CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md` establishes what the sieve
assumes about the data source. **Neither is restated here.** This report is the whole pipeline, and
where those two already answer something it cites them and moves on.

**Relationship to `PIPELINE_BEHAVIOUR_MODEL.md`.** That document is the existing authority on this
subject and remains so. This report does **not** duplicate it. It (a) answers the brief's specific
question — the one the behaviour model does not resolve — by tracing the feature vector to its
construction site and listing its members, (b) follows one seed end to end, which no document does,
and (c) re-verifies the load-bearing anchors at HEAD, which matters because the behaviour model was
produced at `49c13ad` and three of its anchors have since moved (§10, Observation 1).

---

## 1. THE ONE-PARAGRAPH ANSWER — in the sources' own words

**The most authoritative statement**, and the one a new session should carry, is the whitepaper's
closing (`docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:158-167`):

> Bidirectional sieving provides exponential noise suppression.
> Loose thresholds are not a weakness — they are a mathematical necessity
> to expose a learnable structure.
>
> **ML does not guess. It refines a space already reduced from 2³² to 10⁴.**
>
> That is why this system works.

and the naming rule that governs how the whole thing may be described
(`docs/PIPELINE_BEHAVIOUR_MODEL.md:1094-1095`, itself sourced to `CLAUDE.md`'s header and
Chapter 2 §0.4):

> **TFM = Triangulated Functional Mimicry: functional mimicry of PRNG surface output. It is NOT
> seed recovery and NOT state reconstruction.**

**In one paragraph:** TFM takes a stream of published 3-digit values and asks which candidate seeds
of a chosen PRNG family would have produced something consistent with them. A GPU sieve scores every
candidate seed in a configured range twice — once against the observed window in order, once against
it reversed — and keeps the seeds that clear both thresholds. Because the thresholds are deliberately
loose, what survives is not "the seed" but a **manifold** of near-consistent candidates that share
structured deviations. Each survivor's own generator is then replayed and characterised into a
91-number feature vector, a model is trained to rank those vectors against a quality label computed
on data the sieve never saw, and the top-ranked survivors each vote their generator's *next* value
into a prediction pool. When the real draw lands, the pool is graded and the result decides whether
to retrain. **The claim is not that the generator has been identified; it is that survivorship plus
ranking beats the `k/1000` random baseline** (`evaluate_pools.py:36-40`).

---

## 2. VERDICT ON THE OWNER'S FRAMING — clause by clause

The framing offered for checking:

> *"Step 1 sweeps the seed space and the sieve filters candidates — it is NOT reversing or
> recovering generator state. Those initial steps find the most likely seeds, nothing more.
> The ML then learns from them. The windows and offsets are used later in ML."*

| # | clause | verdict | basis |
|---|---|---|---|
| 1 | *Step 1 sweeps the seed space and the sieve filters candidates* | **CONFIRMED, with one refinement** — it sweeps a **configured range**, not the domain, and the sieve **scores** rather than filters | §5.1 |
| 2 | *it is NOT reversing or recovering generator state* | **CONFIRMED — this is the project's own governing rule**, and it is load-bearing rather than cosmetic | §2.1 |
| 3 | *those initial steps find the most likely seeds, nothing more* | **REFINED — "most likely" overstates it.** By design the survivor population is a manifold, not a ranked shortlist; a survivor is *"a scored candidate, not a verdict"* | §2.2 |
| 4 | *the ML then learns from them* | **CONFIRMED** | §7 |
| 5 | *the windows and offsets are used later in ML* | **REFUTED as stated** — `window_size` and `offset` are carried in the NPZ and **never enter the feature vector**. The skip bounds do | §2.3, §7.2 |

### 2.1 Clause 2 — confirmed, and it is a rule, not a description

Not merely accurate — it is the naming rule the project enforces. `PROPOSAL_Documentation_Paradigm_Correction_v1_2.md`
is its origin; `PIPELINE_BEHAVIOUR_MODEL.md:1102-1103` states why it is load-bearing:

> Chapter 2 §5.6 makes the distinction load-bearing rather than cosmetic: **variable skip is a
> detector looking for coherent windows, not a fitting procedure recovering generator state.**

Mechanically confirmed: there is no modular inverse and no backward recurrence anywhere in the tree —
reverse kernels iterate the generator **forward** against a host-reversed residue array
(`miner/range_miner_worker.py:888`; Chapter 2 §3.2). The clause is right, and right for the reason
the project gives.

**One contradiction worth knowing, already governed (do not re-report as new):** three README files
contradict the rule in their own titles — `docs/proposals/README.md:1` *"Seed **Reconstruction**
System"*, `docs/README.md:11` and `README.md:4` *"Reverse-engineer PRNG behavior"*. Recorded as
**DIVERGENT D16** (`PIPELINE_BEHAVIOUR_MODEL.md:1192`) and `PROJECT_FILE_CATALOG.md` §6.4, *"noted
for Alpha — not fixed."*

### 2.2 Clause 3 — "most likely seeds" is the one phrase to change

The sources are emphatic that this is exactly the misreading to avoid.
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:373-377`:

> **A survivor is a scored candidate, not a verdict.** The sieve reduces 2³²-scale output space to a
> survival-conditioned population with enough internal variance for ML to rank (whitepaper §8).
> Survivors may be the true seed, one of several true seeds, a partial match valid before a reseed
> event, or a near-consistent neighbour admitted on purpose. **Deciding which is Step 3 and Step 5's
> job, not Step 2's.**

The whitepaper explains why admitting non-true seeds is the *point*, not a tolerance
(`BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md:118-131`):

> Exact sieves eliminate *all variance*:
> - Survivors = \({s^*}\) · No ranking · No gradients · **No learning signal**
>
> Looser thresholds produce a *manifold* of near-consistent seeds… These seeds share structured
> deviations that ML can learn to rank.

**So the accurate form of clause 3 is:** *the initial steps find a survival-conditioned population of
candidate seeds — deliberately more than one, deliberately including near-misses — and hand the
ranking problem forward.* "Most likely" describes the *output of Step 5*, not the output of Step 1.

**Also worth stating precisely, because it is the project's own load-bearing claim** — the sweep is
**not** exhaustive over the state space, and validity does not depend on that
(`PIPELINE_BEHAVIOUR_MODEL.md:1161-1167`, §15.4):

> A survivor's validity comes from the mathematics of bidirectional survival — `e^(−cn) → e^(−2cn)` —
> **not** from having searched a large fraction of the domain.

`java_lcg` has a 48-bit internal state; the artifact stores `seeds` as `uint32`, so the certified
sweep covers the **`high16 = 0` stratum, 1 part in 65,536**, and says so in nine frozen sidecar
fields (`seed_domain_contract = "v1.1-stratum"`, `exhaustive_over = "high16=0 stratum only"` —
§15.3). **This is a labelling obligation, not a validity problem, and it is settled.**

### 2.3 Clause 5 — the one clause that is wrong, and how

**This is the brief's central question and it resolves cleanly.** The four alignment parameters split
three ways. Verified live at HEAD, three independent ways (NPZ contract, merge site, trained model
sidecar, live artifact):

| parameter | in the 22-array NPZ? | in the Step-3 feature vector? | in the trained model? |
|---|---|---|---|
| `skip_min` | **YES** — array 7, `int32` | **YES** | **YES** — feature #77 of 89 |
| `skip_max` | **YES** — array 8, `int32` | **YES** | **YES** — #75 |
| `skip_range` | **YES** — array 9, `int32` | **YES** | **YES** — #78 |
| `window_size` | **YES** — array 4, `int32` | **NO** | **NO** |
| `offset` | **YES** — array 5, `int32` | **NO** | **NO** |
| `sessions` | **NO** — never becomes an array | **NO** | **NO** |

**Anchors:**

- **The contract** — `utils/canonical_arrays.py:99-126` (`CANONICAL_ARRAY_CONTRACT`), read live:
  `window_size` and `offset` are arrays 4 and 5. The 24-field record side (`:143`) carries
  `sessions`, and the in-source comment at `:139-141` states its fate explicitly:

  > 24 - 2 = 22: `sessions` and `prng_base` do **NOT** become arrays (they are validated anyway, §4.4)

- **The merge site — this is where `window_size` and `offset` die.**
  `survivor_scorer.py:772-782` merges exactly **18** named fields from `survivor_metadata` into the
  feature dict. Read live this session; the list in full:

  ```python
  survivor_scorer.py:774-779
  for field in ['forward_count', 'reverse_count', 'bidirectional_count',
               'intersection_count', 'intersection_ratio', 'survivor_overlap_ratio',
               'skip_min', 'skip_max', 'skip_range', 'skip_mean', 'skip_std',
               'skip_entropy', 'bidirectional_selectivity', 'survivor_velocity',
               'velocity_acceleration', 'intersection_weight',
               'forward_only_count', 'reverse_only_count']:
  ```

  **`window_size` and `offset` are not in it.** They reach the chunk file — the rectangularizer
  keeps every array whose length matches (`generate_step3_scoring_jobs.py:83-92`) — and are then
  simply not read. `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` §1.4 records them among the
  **nine NPZ arrays that never reach the feature vector**.

- **The trained model — the decisive evidence.** `models/reinforcement/best_model.meta.json`,
  read live: `feature_schema.feature_count = 89`, `feature_schema_hash = 7733d30a913545ca`. Probing
  the list: `skip_min` **PRESENT**, `skip_max` **PRESENT**, `skip_range` **PRESENT**;
  `window_size` **absent**, `offset` **absent**, `sessions` **absent**.

- **The live Step-3 artifact.** `survivors_with_scores.json` (84 records; each record's nested
  `features` dict has **91** keys): `skip_min = 5.0`, `skip_max = 56.0`, `skip_range = 51.0`;
  `window_size`, `offset`, `sessions` **absent**.

**Where the intuition behind the clause probably comes from — and it is a real trap.** There *is* a
feature called **`best_offset`**, and it is #26 of the trained 89. It is **not** the Optuna `offset`
and it is **permanently 0.0**:

```
survivor_scorer.py:373    'best_offset': 0.0                              # sequential path
survivor_scorer.py:707    'best_offset': torch.zeros(batch_size, ...)     # batch GPU path
```

Confirmed 0.0 in the live artifact. **This is documented, not a discovery** —
`docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:128` records it as *"constant `0.0`"*,
`docs/PROGRESS_Feature_Remediation_v1_0.md:147` as *"0 (never computed)"*, and
`docs/PROPOSAL_Feature_Implementation_Remediation_v1_0.md:173-189` proposes computing it. Cite
those; do not re-report it.

**Two qualifications on the half of the clause that *is* true.** The skip bounds do travel — but:

1. **They are trial-constant.** `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:106`: *"`skip_*` are
   trial-constant by construction"* — 1 unique value across all 84 live records, against 2 for the
   other ten NPZ-merged features. Within a trial they carry no per-survivor discrimination.
2. **On the hybrid path they describe a search the kernel never received.** `skip_min`/`skip_max`
   die at `_hybrid_prefix` (`miner/range_miner_worker.py:179-193`) — **GOVERNED**, skill §2.7 #4,
   Chapter 2 §5.4, and covered in
   `CLAUDE_CODE_REPORT_SIEVE_CONTINUITY_MODEL.md`. So in hybrid mode the model is fed a bound that
   did not constrain the pass that produced the survivor.

**And the mechanism that makes the clause *feel* right is real, but it is not a feature.** The
per-survivor **skip** is used at prediction time to regenerate the seed's sequence
(`prediction_generator.py:530-553`, `:789`, `:795`, `:845`) — see §6, hop 7. Skip influences the
prediction **through sequence regeneration**, not through the feature vector. That is a genuinely
different mechanism and worth separating in the skill.

---

## 3. THE TWO NUMBERING SCHEMES

Restated because every section below depends on it, and because
`PIPELINE_BEHAVIOUR_MODEL.md:44-45` calls confusing them *"the single most common error a new
session makes."*

| conceptual (whitepaper, system map §1, chapter titles) | executable (`STEP_SCRIPTS`, all 7 manifests, `preflight_check.py`, `README.md`) |
|---|---|
| 1 Window Optimizer | **0** Regime Segmentation (TRSE) |
| **2 Bidirectional Sieve** | **1** Window Optimizer — **the sieve runs inside this** |
| 2.5 Scorer Meta-Optimizer | **2** Scorer Meta-Optimizer |
| 3 Full Scoring | **3** Full Scoring |
| 4 Adaptive Meta-Optimizer | **4** ML Meta-Optimizer |
| 5 Anti-Overfit Training | **5** Anti-Overfit Training |
| 6 Prediction Generator | **6** Prediction Generator |

**Verified live this session** — `agents/watcher_agent.py:387-394` (`STEP_SCRIPTS`), `:398-405`
(`STEP_MANIFESTS`), `:409-416` (`STEP_NAMES`). **There is no executable step whose script is the
sieve.** The two schemes agree at 1, 3, 5, 6 and differ only at 2 — which is exactly what makes the
mistake easy. **The mapping between them is written down nowhere** (`CHAPTER_3_ALIGNMENT_AUDIT.md`
§2 searched for it and reports it as folklore — **DIVERGENT D1**).

**Bare "Step N" below means the EXECUTABLE scheme.**

---

## 4. STEP BY STEP

Each entry: **consumes → decides → produces → hands forward**, with the owning file. Documented
intent (i), implemented (ii) and what runs today (iii) are separated wherever they differ.

### Step 0 — Regime Segmentation (TRSE) · `trse_step0.py`

- **Consumes:** `daily3.json` (manifest `required_inputs`, `agent_manifests/trse.json`).
- **Decides:** *which temporal regime the sequence is in, how long it has been there, how stable it
  is.* It is a **classifier, not a search** — `TRSE_v1_15_SPEC.md:59-61`: *"This is a classification
  based on signal shape, not a brute-force seed search."*
- **Produces:** `trse_context.json` — `current_regime`, `regime_age`, `regime_stable`, `silhouette`,
  `switch_rate`, `regime_type` (`trse_step0.py:50-56`, `:280-293`).
- **Hands forward:** nothing, actively. The seam is **passive**: Step 1 reads the file if present.
  `TRSE_INTEGRATION_PLAN_S121.md` §2C — *"Step 1 reads `trse_context.json` on its own if present…
  WATCHER doesn't need to parse or inject anything."*
- **Its whole sanctioned influence is one number:** Rule A narrows the `window_size` ceiling. Rules B
  (skip) and C (offset) are **advisory by design and deliberately not applied**
  (`SESSION_CHANGELOG_20260307_S122.md:56`, *"disabled per TB + S121 shuffle test"*). **They are not
  dropped wires** — `TRSE_v1_15_SPEC.md:216-240`, which says they apply, is **SUPERSEDED**
  (**D13**).
- **Failing silently is architected, not accidental** — `trse.json` sets `skip_on_fail: true` with a
  stated reason. A Step-0 failure records `action: proceed` and Step 1 runs on full default bounds.
- **GOVERNED, cite don't re-report:** F1 (WATCHER's Step-0 command is malformed — three
  `default_params` keys the argparse does not define; exit 2, masked by `skip_on_fail`); F2
  (self-perpetuating freshness lock); F6 / `recommended_window_size` (manifest declares `8`, code
  reads it into `_rec_ws` and never uses it, Rule A hardcodes `32` — and `8 × 4 = 32`, so **the
  value is correct and the wiring is missing**, **D14**). Step 0 **has no documenting chapter**
  (`PROJECT_FILE_CATALOG.md` §7 gap 1).

### Step 1 — Window Optimizer · `window_optimizer.py` (+ `_integration_final`, `_bayesian`)

Covered in depth by `CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md`; only the pipeline-relevant
mechanism is restated.

- **Consumes:** the dataset **via the pointer manifest** (`daily3_current.json` → immutable
  `daily3-<UTC>Z-<sha12>.json`), search bounds from `distributed_config.json → search_bounds` merged
  **over** code defaults, `trse_context.json` (optional), ~40 CLI flags.
- **Decides — two things, and conflating them is a known error.** (a) *which alignment resolves* —
  Optuna TPE over **seven** dimensions, all inside the `optuna_objective` closure in
  `window_optimizer_bayesian.py`: `window_size`, `offset`, `session_idx`, `skip_min`, `skip_max`,
  `forward_threshold`, `reverse_threshold` (`:529-550`). The space is **not a rectangle** —
  `skip_max`'s floor is `max(skip_min, bounds.min_skip_max)` (`:542`), so the skip dimensions are
  coupled. (b) *which seeds survive* — per trial, `run_bidirectional_test()` runs real sieves across
  the fleet.
- **Produces — and only one of these matters:**

  | file | what it really is |
  |---|---|
  | **the certified NPZ generation** | **CANONICAL.** 22 arrays + sidecar, `artifact_sha256`, chained lineage |
  | `optimal_window_config.json` | best parameters + `agent_metadata` |
  | `bidirectional_survivors.json` | **post-success summary — generation IDs and sha256s, no seeds** |
  | `forward_survivors.json` / `reverse_survivors.json` | **count-only stubs** |
  | `train_history.json` / `holdout_history.json` | 80/20 split of the draw data |

  The demotion is stated in-source at `window_optimizer_integration_final.py:2999-3001` (read live):
  *"It is **NO LONGER** the canonical Steps 2-6 input… Steps 2-6 consume the canonical NPZ."* The
  finalizer call is `_finalize_run_d3_5(...)` at `:2972` (imported `:2878`).
  The stubs are deliberate — `[S166-ACCUM]` replaced survivor-object retention with counters to stop
  a RAM bomb at fleet scale. **Do not "restore" full retention.**
- **Hands forward:** the 22-array NPZ.
- **The backend cascade — four backends, one certifying.** `run_bidirectional_test`
  (`window_optimizer_integration_final.py:1369`) opens with `getattr`-gated branches:
  `_use_miner` (`:1402`) → **RANGE-MINER, the certifying route**; `_use_pw` (`:1547`) → PWC,
  non-certifying, **hybrid quarantined**; `_use_zmq` (`:1592`) → non-certifying; else the legacy
  coordinator. **Consequence:** *"what does Step 1 do here?"* is unanswerable without knowing which
  flag was set, and most production runs do **not** take the path Chapter 1's diagram draws.
- **Threshold authority:** resolved **once, in the parent**, by `resolve_directional_threshold()`
  (`window_optimizer_integration_final.py:214`), precedence explicit > config > default, `is None`
  as the sole fallback trigger (*"0.0 is a legitimate threshold"*), fail-closed. GOVERNED history:
  fixed `3fdf434`, silently reverted by `2389b61`, repaired `8a55a68`.

### Step 2 — Scorer Meta-Optimizer · `run_scorer_meta_optimizer.sh`

**Chapter 3 is AUDITED and NOT CORRECTED — 55 claims: 17 accurate / 9 stale / 24 false / 5
unverifiable. Read `CHAPTER_3_ALIGNMENT_AUDIT.md` before trusting any specific claim in it.**

- **Consumes:** **seven arrays, by name, from the NPZ only** — `seeds`, `forward_matches`,
  `reverse_matches`, `bidirectional_count`, `intersection_ratio`, `trial_number`, `skip_mode`
  (`CHAPTER_3_ALIGNMENT_AUDIT.md` §4).
- **Decides:** the **scoring hyperparameters Step 3 will use**, judged against the sieve's own
  evidence rather than draw history. Chapter 3 §4.2 draws the boundary: *"Step 2: Find optimal
  SCORING PARAMETERS using SIEVE QUALITY as ground truth. Chapter 13: Compare predictions to real
  draws."*
- **Produces:** `optimal_scorer_config.json` — **eleven scalars, nothing per-seed.**
- **Hands forward:** those eleven scalars (they set `residue_mods`, temporal window sizes etc. that
  Step 3's scorer is constructed with — `survivor_scorer.py:107`, `:109-110`).
- **This step is the strongest evidence for the interface claim.** All seven arrays it reads are in
  the frozen 22 — *"RANGE-MINER's certified artifact satisfies every column this consumer needs,
  with nothing missing and nothing extra required. This consumer genuinely cannot tell which engine
  produced the bundle."* (`PIPELINE_BEHAVIOUR_MODEL.md:410-414`.)
- **GOVERNED, all diagnosed, none authorised for repair:** live objective is **v4.3** while the
  chapter, the module docstring and the "TB FORMULA (final v4.2)" block all advertise v4.2 (**D6**);
  the rewrite landed inside a commit about *moving documentation* (`ca975f8`); the objective is
  structurally blind to 7 of 11 sampled dimensions; and the v4.0 objective **measured itself**
  (`quality = fwd*rev` dominant at w3≈0.82 → WSI 0.9997 on trial 1). **★ Read
  `TB_RULING_REQUEST_STEP2_v4_2_SIGNAL.md` before reporting any Step-2 objective blindness** — Alpha
  nearly submitted it to Beta as a discovery.
- **⚠ Operational hazard, and it is why the soak is bounded:** Step 2's fallback path invokes the
  **TB-prohibited legacy converter** and `mv`s a regular file onto the D3.5 finalizer-owned symlink
  → `PublicationError`. This is why `--start-step 1 --end-step 1` is mandatory for a Step-1-only run
  (`--end-step` **defaults to 6**).

### Step 3 — Full Scoring · `run_step3_full_scoring.sh`

**Chapter 4 is UNAUDITED. Treat its specific numbers as unverified** — it says "47 features"
(§9.8); the live count is 91.

- **Consumes:** the 22-array NPZ, `optimal_scorer_config.json`, `train_history.json`,
  `holdout_history.json` (parsed live from `agent_manifests/full_scoring.json`).
- **Decides:** nothing, in the search sense. It is a **transformation**: regenerate each survivor's
  PRNG sequence from its seed and characterise it. The only judgement it makes is the holdout label.
- **Produces:** `survivors_with_scores.json` + `scoring_statistics.json`.
- **Hands forward:** **91 features per survivor + the holdout label.** Full construction in §7.
- **`offset` for the holdout label is a law, not a parameter** — Chapter 4 §1.4 / Chapter 6 §3.2.1,
  quoting Beta: *"Remove offset as a choice. Keep offset as a law."* `offset = len(train_history)`,
  because a PRNG is a state machine and the holdout is contiguous future data.
- **GOVERNED — the sequential-fallback zero-fill is P0.** The batch path merges **18** metadata
  fields (`survivor_scorer.py:774-779`); the sequential fallback merges only **6**
  (`full_scoring_worker.py:452-455`, read live: `forward_count`, `reverse_count`,
  `bidirectional_count`, `skip_min`, `skip_max`, `skip_range`). On GPU-batch failure the other
  twelve silently become `0.0` while the record keeps all 91 keys and every schema check passes.
  Beta classifies this **P0** and requires gate **F-PARITY**. **Not a RANGE-MINER defect and not
  fixable by it.**
- **GOVERNED — the most consequential seam fact.** `forward_matches` and `reverse_matches`, *the
  only independent per-seed sieve signal*, are transported through the NPZ and the chunk layer and
  are **absent from the merge list**. Beta: *possibly the most consequential finding in the whole
  trace.* **Ruling: do NOT silently add them** — a governed schema decision (Option A or B) is
  required. The miner keeps emitting both regardless.

### Step 4 — Adaptive Meta-Optimizer · `adaptive_meta_optimizer.py`

**Chapter 5 is UNAUDITED.**

- **Consumes:** `optimal_window_config.json` + `train_history.json`.
- **Decides:** **capacity and architecture** — how many survivors to train on, how deep the network,
  how long training runs.
- **Produces:** `reinforcement_engine_config.json` (capacity only).
- **The design principle is a prohibition, and it is the interesting fact about this step:**

  > **Step 4 is intentionally NOT data-aware.** (Chapter 5 §1.2)

  It does not read `survivors_with_scores.json`, does not inspect `holdout_hits`, does not choose a
  model type and does not touch holdout data — because any of those would tune capacity on
  validation data and compromise Step 5's evaluation (§3.2, §3.3). `--survivor-data` and
  `--holdout-history` exist for backward compatibility and are **intentionally ignored** (§7.3).
  Weighting: window-optimizer results 0.60 · history complexity 0.35 · reinforcement feedback 0.05
  → 0.25 cap.
- **GOVERNED:** it reads `best_result.bidirectional_count`, `best_result.precision` and
  `all_results[].bidirectional_count` from `optimal_window_config.json` — **none of which Step 1
  writes** — and silently falls back to `{'min':100,'optimal':500,'max':2000}`. **TB Q4:
  PRESERVE the drift. Do NOT add those keys inside S172.**

### Step 5 — Anti-Overfit Training · `meta_prediction_optimizer_anti_overfit.py`

**Chapter 6 is UNAUDITED and its stated target is SUPERSEDED.**

- **Consumes:** `survivors_with_scores.json`, `train_history.json`,
  `reinforcement_engine_config.json`.
- **Decides:** which of **four model families** best ranks survivors, and whether the result is
  trustworthy enough to predict from.
- **Produces:** `models/reinforcement/best_model.*` + `best_model.meta.json`.
- **Hands forward:** the model **and its sidecar** — the sidecar is the seam (§7.4).
- **Subprocess isolation is mandatory, not stylistic** — `docs/DESIGN_INVARIANT_GPU_ISOLATION.md`:
  GPU-accelerated code must never run in the coordinating process when subprocess isolation is in
  use. Enforced since S72.
- **Target and validation:** see §7.3. In short — target `holdout_quality`, **K-fold CV**, `k_folds`
  default 5.

### Step 6 — Prediction Generator · `prediction_generator.py`

**Chapter 7 is UNAUDITED and its I/O description does not match the live module.**

- **Consumes:** the survivor records, the model, and the sidecar.
- **Decides:** (a) **whether to predict at all** — the abstention gate; (b) the ranking; (c) the
  pool.
- **Produces:** `predictions/next_draw_prediction.json` (canonical, `:899-907`, written `:945-951`)
  plus a **timestamped, explicitly non-contractual** history copy (`:911-913`, `:949`).
- **Sidecar-only loading.** *"Model type is determined ONLY from `best_model.meta.json`. File
  extensions are NEVER used"* (Chapter 7 §3.1), and a feature-schema-hash mismatch is **FATAL** —
  a reordered feature vector produces meaningless predictions from a model that will not complain.
  Enforced at `prediction_generator.py:769-775` (raises unless `require_feature_schema=False`).
- **The abstention gate — verified live.** `:485-510` reads `signal_quality.prediction_allowed` from
  the sidecar; when false it logs `SIGNAL QUALITY GATE BLOCKED` and returns a skip result. The
  in-source contract at `:470-476`: *"Learning steps declare signal quality; execution steps act only
  on declared usable signals; control agents decide recovery."* **Step 6 consumes signal quality and
  does not recompute it.**
- **DIVERGENT — D3, GOVERNED.** Chapter 7 describes `ranked_predictions.json`,
  `prediction_pools.json` and three pools Tight(20)/Balanced(100)/Wide(300). Live: **one** pool
  (`pool_size: int = 20`, `:89`), builder `_build_prediction_pool` (**singular**, `:751`), one
  output. `agent_manifests/prediction.json` **agrees with the code, not the chapter.** Chapter 7 is
  dated Dec-2025 and predates the v6.0 module. This is system-map **Blocker 1** (reader exists, no
  writer exists) — a cause of the attribution blockage, not a new finding.

### Feedback — Chapter 13 · `chapter_13_orchestrator.py` (**not a WATCHER step**)

- **Consumes:** a new draw, signalled by `NEW_DRAW_FLAG = "new_draw.flag"` (`:73`).
- **Decides:** whether to retrain, and which loop to run.
- **Produces:** diagnostics + retrain triggers; writes the S140b downstream score as **annotation
  only**, non-fatal on failure (`:306-316`).
- **Static vs dynamic steps.** Steps **1, 2, 4** run once and re-run only on regime shift; steps
  **3, 5, 6** are the learning loop. *"The system learns by weighting survivors, not by endlessly
  searching new ones"* (Chapter 13 §3.2). **Labels evolve through data accumulation, not mutation**:
  a new draw is appended, Step 3 recomputes the label over expanded history, Step 5 retrains.
- **The design principle:** *the LLM cannot rewrite mathematical logic, invent features, bypass
  validation, mutate control flow or change step ordering; it can interpret diagnostics, detect
  drift, propose parameter adjustments, recommend retraining* (Chapter 13 §2, §11) — enforced by
  Pydantic schemas, GBNF grammars, feature-hash validation and manifest-scoped parameters.

---

## 5. WHAT THE SIEVE ACTUALLY DECIDES

### 5.1 What makes a seed a survivor

**The kernel is a scorer with a threshold, not a set filter.** Each GPU thread owns one seed, walks
the observed window once per skip hypothesis, and keeps a **match rate**
(`prng_registry.py:972-999`, the `java_lcg` constant-skip forward kernel; transcribed in Chapter 2
§2.2):

```c
for (int skip = skip_min; skip <= skip_max; skip++) {
    state = seed & m;
    for (o = 0; o < offset; o++) state = (a*state + c) & m;   // pre-advance
    for (s = 0; s < skip;   s++) state = (a*state + c) & m;   // burn before draw 0
    for (int i = 0; i < k; i++) {
        state = (a*state + c) & m;
        output = (state >> 16) & 0xFFFFFFFF;
        if (three-lane test) matches++;
        for (s = 0; s < skip; s++) state = (a*state + c) & m; // burn between draws
    }
    rate = matches / k;
    if (rate > best_rate) { best_rate = rate; best_skip_val = skip; }
}
if (best_rate >= threshold) emit(seed, best_rate, best_skip_val);
```

Four properties that matter downstream (Chapter 2 §2.2):

1. **The output is a rate, not a boolean.**
2. **Skip is maximised over, not fixed** — which is why a survivor is a **pair**.
3. **Ties resolve to the lowest skip** (`rate > best_rate`, strictly greater).
4. **The rate is float32 against a float32 threshold** — float64 arithmetic puts boundary survivors
   on the wrong side of `>=`.

The three-lane test `(output % 1000) && (output % 8) && (output % 125)` is **exactly equivalent to
`% 1000`** — 1000 = 8 × 125 with gcd(8,125) = 1, proven two ways in Chapter 2 §6 including an
exhaustive check with zero divergent cases. **It is not extra filtering power.**

### 5.2 What "bidirectional" means operationally

```
bidirectional_survivors = forward_survivors ∩ reverse_survivors
```

**A set intersection and nothing more** (Chapter 2 §4.2): *"There is **no joint gate, no
re-verification of the surviving pair, and no combined-rate threshold.** The two passes are
independent scored runs whose seed sets are intersected."* `intersection_count` duplicating
`bidirectional_count` is deliberate.

"Reverse" means the **target data** is reversed, never the generator — the host does
`residues[::-1]` (`miner/range_miner_worker.py:888`) and the reverse kernel runs the same forward
recurrence. **Do not "fix" this.**

**Why it works** (whitepaper §5, `:79-94`): for incorrect seeds forward and reverse survival are
approximately independent, so `P(B) ≈ P(F)²` — *"This **squares the exponent** — a catastrophic
collapse of noise."*

**One genuine divergence, named and unresolved (D10).** Whitepaper §4 (`:57-59`) defines the reverse
predicate with a **negative index** — a backward generator step. The implementation evaluates
`G(s,i)` forward against a reversed array, so for one seed the two passes generate the **identical**
sequence and differ only in what it is compared against. The independence premise on which the
squaring rests is stated about a construction in which they would not be identical. Chapter 2 §3.5
records this deliberately and declines to assert the statistical consequence — *"mathematics is the
whitepaper's side of the boundary."*

### 5.3 What a survivor is evidence *of*

Quoted in full at §2.2. **A scored candidate, not a verdict** — and by construction the population
is a manifold, because a population of exactly one has no variance and therefore no learning signal.

---

## 6. THE DATA SPINE — one seed, end to end

Following one unit of work, with the contract at each hop.

**Hop 1 — a seed is scanned.** The coordinator partitions `[base_start, base_start + total_seeds)`
into contiguous macro-stripes with no gap and no overlap; the worker splits its macro-stripe into
GPU-safe sub-stripes at runtime. Completion is **proved, not assumed**: sub-stripes done == expected
== distinct indices, seed counts sum, survivor counts sum, **and the sub-stripe ranges tile the
parent exactly** (Chapter 2 §8.3).

**Hop 2 — it survives, and becomes a *pair*.** The kernel emits `(seed, best_rate, best_skip_val)`.
Chapter 2 §5.5: **"A survivor is a (seed, skip-hypothesis) pair — not a seed."**

```json
{ "seed": 244139, "skip": 5, "skip_mode": "constant", "match_rate": 0.98 }
```

**Hop 3 — forward ∩ reverse.** Only seeds in both sets continue.

**Hop 4 — it becomes a canonical record, then 22 arrays.** The **24-field**
`CANONICAL_RECORD_FIELDS` (`utils/canonical_arrays.py:143`, duplicated at
`utils/canonical_records.py:115` — *check which one your consumer imports*) is validated, then
**24 − 2 = 22**: `sessions` and `prng_base` are validated but do not become arrays;
`forward_match_rate`/`reverse_match_rate` are **renamed** to `forward_matches`/`reverse_matches`.
The finalizer picks **one record per seed** via the frozen `_l2_sort_key` /`_select_l2_winners`
(`utils/run_finalizer.py:690`, `:714`): highest **float32** score → lowest `trial_number` →
constant-before-variable *within a trial only*; a same-trial/same-mode collision raises
`AccumulatorConsistencyError` because it means the accumulator was fed twice. Ten `_validate_*`
functions run before publication; generations **chain**.

**Contract at this hop:** `CANONICAL_ARRAY_CONTRACT` — 22 names, exact order, exact dtypes
(`utils/canonical_arrays.py:99-126`). **Only 4 of the 22 columns carry per-seed information:**
`seeds`, `forward_matches`, `reverse_matches`, `score`. The rest are trial-level values broadcast
across every survivor of that trial.

**Hop 5 — it is rectangularized into a chunk.** `extract_survivors_full()`
(`generate_step3_scoring_jobs.py:62-102`) transposes arrays into per-survivor dicts, keeping **every**
array whose length matches `n` and renaming `seeds`→`seed`. **All 22 survive into the chunk file.**
Its docstring records why it exists: *"Previous version discarded metadata, causing 14/47 ML features
= 0."* The METADATA-LOSS guardrail at `:95-100` is **weak** — the message says *"Expected 20+"* but
the threshold is `3`, so dropping 19 of 22 fields would pass silently.

**Hop 6 — it becomes 91 numbers.** The seed's own generator is replayed
(`survivor_scorer.py:566-582` batch / `:355` sequential) and compared against `train_history`; 18
metadata fields are merged (`:772-782`); 14 `global_*` are stamped on
(`full_scoring_worker.py:403-405`). **This is where `window_size` and `offset` stop travelling.**
Full composition in §7.1.

**Hop 7 — it is ranked, and it votes.** This is the hop no document walks, and it is the answer to
*what a seed's "relevance" means downstream*. `prediction_generator.py:800-852`, read live:

1. A feature matrix is built **in sidecar order** — `row = [float(features.get(name, 0.0)) for name
   in self.feature_names]` (`:797`). *Order comes from the sidecar, which is why the hash check is
   fatal.*
2. For `neural_net` only, the training scaler is applied (`:812-817`), with an explicit `model_type`
   gate to prevent cross-model normalization.
3. `predicted_quality = self.model.predict(X)` (`:827`) — **this is the seed's relevance: the
   model's predicted `holdout_quality` for it.**
4. Survivors are sorted descending (`:835`) and the top `pool_size` taken.
5. **For each, the seed's sequence is regenerated with its own skip and the value at
   `next_idx = len(lottery_history)` is read off** (`:845-848`):

   ```python
   prediction_generator.py:838   next_idx = len(lottery_history)
   prediction_generator.py:845   seq = self.scorer._generate_sequence(seed, next_idx + 1, skip=survivor_skip)
   prediction_generator.py:848   predictions.append(int(seq[next_idx]))
   ```

   The per-survivor skip comes from `_get_survivor_skip` (`:530-553`), whose docstring states the
   reason plainly: *"Each survivor may have been discovered with a DIFFERENT skip value. Using a
   single global skip will generate wrong predictions."*

**So a survivor's contribution to the prediction is literally the next number its generator would
emit, and its weight is the model's predicted quality.** That is the whole spine, and it is why the
skip hypothesis is part of the survivor's identity rather than search bookkeeping.

**Hop 8 — the pool is graded.** Chapter 13 compares against the real draw and writes
`0.50·hit@20 + 0.30·hit@100 + 0.15·hit@300 + 0.05·pool_coverage`
(`chapter_13_orchestrator.py:306-316`). **Note the mismatch, already flagged and not adjudicated:
this objective weights three pool sizes while the live generator emits one.**

---

## 7. WHAT THE ML LEARNS FROM — the clause the owner most needs verified

### 7.1 The feature vector, at its construction site

**91 extracted / 89 trained.** Verified live four ways this session: the trace report's enumeration,
the merge site, the trained sidecar (`feature_count: 89`), and the live artifact (nested `features`
dict = **91 keys**).

**Primary evidence document:** `docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` §1.3, which
enumerates all 91 by origin with per-feature `file:line`. **Read it rather than re-deriving it.**
Composition:

| category | count | construction site | source |
|---|---|---|---|
| **(a) merged directly from an NPZ array** | **13** | `survivor_scorer.py:772-782`, list at `:774-779` | the sieve |
| **(b) computed in Step 3** | **59** | `survivor_scorer.py:702-731` (27 match/stats) · `:624-667` (9 residue) · `:745-757` (23 battery) | the **regenerated PRNG sequence** vs `train_history` — only NPZ input is `seeds` |
| **(c.1) `global_*` run-context** | **14** | `full_scoring_worker.py:348-349`, merged `:403-405`; names at `models/global_state_tracker.py:366-381` | the **draw history only** |
| **(c.2) dead placeholders** | **5** | requested `survivor_scorer.py:776-778`, zero-filled `:784-791` | **no producer exists** |
| | **91** | | |

**The 13 from the NPZ, in full** (`survivor_scorer.py:774-779`): `forward_count`, `reverse_count`,
`bidirectional_count`, `intersection_count`, `intersection_ratio`, `intersection_weight`,
`bidirectional_selectivity`, `forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`,
**`skip_min`**, **`skip_max`**, **`skip_range`**.

**Three namespaces, and the distinction is architecturally load-bearing** (system map §8):

- **72 survivor-local** — the legitimate search space.
- **14 run-global** — *identical for every survivor in a run*. Filtering by one can only retain or
  remove **the whole run**, and random row folds across runs **leak run identity**. Governance:
  globals *"must not be searchable as ordinary per-survivor filter thresholds"*, and folds over
  multi-run datasets must be **grouped or temporally separated by run** (§8.2).
- **5 permanently zero** — `skip_mean`, `skip_std`, `skip_entropy`, `survivor_velocity`,
  `velocity_acceleration`. Confirmed 0.0 across all 84 live records. **All five are in the trained
  89** (#74, #76, #79, #81, #89 of the sidecar list) — the model trains on five constant columns.
  Three of them are skip-shape statistics whose producer **exists on the GPU** and is discarded at
  `window_optimizer_integration_final.py:125` (`extract_survivor_records`, which reduces each
  survivor to `{seed, match_rate}`).

**A sixth constant feature exists and is separately documented:** `best_offset` (#26) —
`survivor_scorer.py:373` / `:707`, recorded as *"constant `0.0`"* at
`S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:128`. It is **not** counted among "the 5" because it
has a producer; it just produces zero. Net: **six of the 89 trained features carry no information.**

**Nine NPZ arrays never become features** (`S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` §1.4):
`seeds` (join key + regeneration input, not itself a feature), **`forward_matches`**,
**`reverse_matches`**, **`window_size`**, **`offset`**, `trial_number`, `skip_mode`, `prng_type`,
`score`.

### 7.2 Do the alignment parameters travel forward? — the direct answer

**Answered in full at §2.3. Summary: no, except the skip bounds.**

- `skip_min` / `skip_max` / `skip_range` → **feature vector and trained model** (but
  **trial-constant**, and on the hybrid path they describe a bound the kernel never received).
- `window_size` / `offset` → **carried in the NPZ, never read as features.**
- `sessions` → **never becomes an array at all.**
- The Optuna `offset` is **not** the `best_offset` feature; the latter is constant 0.0.
- Skip *does* influence the prediction — but through **sequence regeneration at Step 6**
  (`prediction_generator.py:845`), not through the feature vector.

### 7.3 The target, and how it is validated

**Target: `holdout_quality`.** Verified live at `meta_prediction_optimizer_anti_overfit.py:1481-1482`:

```python
target_field='holdout_quality',
exclude_features=['score', 'confidence', 'holdout_hits', 'holdout_quality']
```

`score` and `confidence` are the two dropped from 91 → **89**. Chapter 6's stated target
`y = holdout_hits` is **SUPERSEDED** (`1cb90aa`; **D5**), and the earlier `y = score` was a
**tautology** — `score` *is* `exact_matches / total × 100`, mathematically the same quantity as
`residue_1000_match_rate`, so the model learned nothing and ignored 60 of 62 features
(Chapter 6 §2.1-§2.3).

**What `holdout_quality` actually is** — a composite of the *same* features recomputed on holdout
data (`holdout_quality.py:61-90`, docstring quoted verbatim, *"Per v1.1 proposal §4, Team Beta
approved"*):

| block | weight | composition |
|---|---|---|
| CRT match quality | **50%** | 40% `residue_1000_match_rate` · 20% `lane_agreement_8` · 20% `lane_agreement_125` · 20% `lane_consistency` |
| Distributional coherence | **30%** | 34% `residue_1000_coherence` · 33% `residue_8_coherence` · 33% `residue_125_coherence` |
| Temporal stability | **20%** | 100% `temporal_stability_mean` |

Computed at `full_scoring_worker.py:408-423` from `holdout_features` — a **77-key** vector built by
the same scorer against the holdout window (verified live: `len(holdout_features) == 77`).

**Validation: K-fold cross-validation**, `k_folds: int = 5` (`:1394`, `:1428`, logged `:1531`),
`from sklearn.model_selection import KFold, TimeSeriesSplit` (`:45`), with the fold count clamped
when survivors are scarce (`:2028-2031`, `_effective_folds = max(2, min(self.k_folds, n_samples))`).
Overfit detection is explicit: `overfit_ratio > 1.5` flags overfitting (`:1355`) and the composite
score applies a 0.5 penalty (`:1361-1362`).

**So there are two independent guards, and they are different things:** the **holdout split** is
temporal and structural (`offset = len(train_history)`, contiguous future data, *a law not a
parameter*), and **K-fold** is applied across survivors within the training set. The holdout is not
a fold.

### 7.4 The sidecar is the seam

`best_model.meta.json` carries `model_type`, the feature schema **and its hash**, metrics,
hyperparameters, `signal_quality` and provenance. Step 6 validates the hash before predicting; a
mismatch is FATAL (Chapter 6 §9, Chapter 7 §4.1). **Authority order for feature names is model
sidecar → versioned extraction manifest → runtime validation, and no diagnostic module may maintain
a manually duplicated feature list** (system map §9) — `feature_importance.py:95-119` omits 31 live
features and the same stale list is duplicated in `feature_drift_tracker.py`.

### 7.5 What actually ran last — dated, and worth stating

The live sidecar and artifact are from **March 2026**, not from the current work:

| fact | value | anchor |
|---|---|---|
| model type | `neural_net` | `best_model.meta.json` `model_type` |
| target | `holdout_quality` | `signal_quality.target_name` |
| **signal status** | **`weak`**, confidence **0.4**, `prediction_allowed: true` | `signal_quality` |
| target variance | **3.999e-06** (std 0.002, mean 0.274) | `signal_quality` |
| survivors trained on | **85** | `data_context.survivor_source.survivor_count` |
| train / holdout | draws 1–14,454 / 14,455–15,454 | `data_context` |
| **R²** | **0.0205** | `training_metrics.r2` |
| trained at | **2026-03-06T21:29:52** | `provenance.started_at` |
| Step-3 artifact | 84 records, 91 features | `survivors_with_scores.json`, mtime 2026-03-11 |

**Read this as "what ran last", not "what the system is."** The signal was declared **weak** and the
gate still allowed prediction; R² was 0.0205 — consistent with the governed finding that **R² was
abandoned as an objective at 0.000155 and is not the objective**
(`PROJECT_FILE_CATALOG.md` §6.1). A falsification test must be run against the live target, not R².

---

## 8. WHERE THE OUTPUTS GO

**The prediction.** `predictions/next_draw_prediction.json` — one pool of `pool_size` (default 20),
each entry a next-draw value voted by one top-ranked survivor, with the model's predicted quality as
its confidence (§6 hop 7).

**"Relevance" of a seed, defined precisely:** its `predicted_quality` under the trained model — i.e.
the model's estimate of how well that seed's regenerated sequence would score on *unseen* data.
Relevance decides **rank**; the seed's *own generator* decides **what it votes for**.

**Pools 20/100/300.** Two lineages, and confusing them is easy:

- **Documented intent** — Chapter 7 §6.2 describes `build_prediction_pools()` slicing at 20/100/300
  into `tight`/`balanced`/`wide` with `--pool-sizes` defaulting to `20,100,300`.
- **What runs** — a **single** pool. `prediction_generator.py:89` `pool_size: int = 20`; the builder
  is `_build_prediction_pool` (**singular**, `:751`). No `build_prediction_pools`, no
  `prediction_pools.json`, no `ranked_predictions.json`, no `--pool-sizes` in the module. The
  manifest agrees with the code. **D3**, and it is system-map **Blocker 1**.
- **A separate backtest lineage** — `build_pools.py` (`--pools` default `"20,100,300"`, `:173`) and
  `evaluate_pools.py:28` (hit **and lift vs random**, `k/1000` baseline, `:36-40`). Their only
  in-repo caller is `backtest_pools.py`. **Nothing in the production chain was ever meant to call
  them** (`PIPELINE_BEHAVIOUR_MODEL.md:1263-1265`).

**⚠ Do not propose building a coverage/lift metric — it exists.** System map §12 item 3 records that
claim as **withdrawn**: *"Wrong — already in `evaluate_pools.py`."*

**The falsification criterion.** `holdout_hits` is the designated criterion, classified
**Classification A — authorized offline outcome-derived supervised label; permitted as a training
target, forbidden as a filter, weight, mask, window or production-time feature** (system map §10). It
is computed on data never seen during sieving and is **not** one of the 91 inputs. The thesis is
falsified if hits show no measurable lift over `k/1000`, or the known-answer control fails (planted
`TRUE_SEED` 12345 must reach `holdout_hits = 1.0` while others sit at ≈0.001), or importance stays
concentrated in the circular pair `residue_1000_match_rate` + `exact_matches` ≈ 100% — the tautology
signature (Chapter 13 §6.1, Chapter 6 §3.6-§3.7).

---

## 9. THE CONTROL LAYER

### 9.1 WATCHER

WATCHER runs steps 0→6 by dispatching `STEP_SCRIPTS[n]` under `STEP_MANIFESTS[n]`
(`agents/watcher_agent.py:387-405`), evaluates each step's outcome, and decides confidence-banded:

| decision | band | authority |
|---|---|---|
| PROCEED | confidence ≥ 0.70 | WATCHER alone |
| RETRY | 0.50–0.70 | WATCHER alone, bounded by `max_retries_per_step` = 3 and `max_total_retries` = 10 |
| ESCALATE | < 0.50, or retries exhausted | **halts; human review required** |

**The five safety invariants** (`WATCHER_POLICIES_REFERENCE.md` §Safety Invariants — the canonical
meaning of every flag in `watcher_policies.json`):

1. **`test_mode=false` overrides everything.**
2. **Auto-approve requires BOTH flags** (`test_mode` *and* `auto_approve_in_test_mode`) — either
   alone does nothing.
3. **`approval_route` is governance** — it decides *who executes*, not *whether to approve*.
4. **Invalid enum values fail safe** — an unknown `approval_route` reverts to `"orchestrator"`.
5. **WATCHER never mutates policies** — only humans change `watcher_policies.json`.

**Parameter flow to a step is a three-hop route, and hop 1 silently drops anything undeclared.**
WATCHER's step-scoped filter is `if key in declared` (`agents/watcher_agent.py:1290-1314`), so a
parameter added only to a method signature exists, accepts a value, and **never receives one from
production**. **Gate the route, not the parameter.**

### 9.2 Chapter 13 triggers — all defensive

Verified live, `chapter_13_triggers.py:60-70`:

```python
N_DRAWS · CONFIDENCE_DRIFT · CONSECUTIVE_MISSES · HIT_RATE_COLLAPSE
REGIME_SHIFT · LLM_PROPOSED · MANUAL · SELFPLAY_RETRAIN
```

Actions (`:73-78`): `LEARNING_LOOP` (3→5→6) · `FULL_PIPELINE` (1→6) · `STEP_6_ONLY` · `SELFPLAY`
(*enum only, WATCHER dispatches*). Priority (`:340-346`):
`REGIME_SHIFT > HIT_RATE_COLLAPSE > CONSECUTIVE_MISSES > CONFIDENCE_DRIFT > N_DRAWS > LLM_PROPOSED`.

**Every trigger is degradation-based. There is no opportunity trigger** — nothing fires on *"a
heuristic shows durable edge — press it"* (system map §6). The accepted nuance, from Michael's own
correction: the *pipeline* does find working configurations; the *feedback loop* is defensive, and
discovery operates at config/window level rather than the per-survivor-heuristic level that
attribution would expose.

### 9.3 Human-gated vs automatic

| automatic | human-gated |
|---|---|
| PROCEED / RETRY within bounds | ESCALATE (halts) |
| trigger evaluation and priority selection | **approving a Chapter-13 retrain** (unless BOTH test-mode flags) |
| dataset freeze, fleet resolution, admission | **applying an LLM parameter proposal** — filtered at the step boundary, **deliberately** |
| grading, downstream-score annotation | **selecting a sampler or a sieve strategy — reserved authority** |

**Reserved authority (human only):** feature engineering · survivor thresholds · sieve
strategy/mathematics · window-optimizer logic · PRNG-family authority · scoring logic ·
meta-optimizer search space · model families · policy authority.

**Two of these are NOT defects** and were mis-reported once already: Chain D's `pending_approval` is
a **valid authority boundary**, and the Step-5 `allowed_params` filter is a **deliberate
executable-interface boundary** — both upheld by Beta after Alpha corrected its own reporting
(`TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md`).

**Autonomy adjusts parameters, never structure** (whitepaper §9, `:147-154`). The corollary is the
project's most repeated defect class: *every tuned parameter must physically reach the kernel and
its effective value must be observable* — otherwise the sampler steers a knob connected to nothing.
**Seven instances catalogued** (skill §2.7).

**Attribution — the exact classification.** *Implemented, invoked, unreachable, unconsumed.* Beta:
*"Algorithmically implemented, partially integrated, operationally disconnected and not yet
behaviorally closed."* **Never say "wired"; never say "not implemented" — both are wrong.** Four
independent blockers (system map §4.2); v1.1's "two cheap unblocks" is **WITHDRAWN**.

**Selfplay is not a learning system** — a **policy-conditioned evaluation harness**
(`TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md`, which **governs how selfplay may be described**).

---

## 10. LOAD-BEARING FACTS FOR A NEW SESSION

Ten, each with why it matters and an anchor. **Items 1–4 are the ones this report adds; 5–10 restate
`PIPELINE_BEHAVIOUR_MODEL.md` §19 because they are still the ones that get broken.**

1. **`window_size` and `offset` do NOT reach the ML. `skip_min`/`skip_max`/`skip_range` do.** —
   *Because "the windows and offsets are used later in ML" is the natural assumption and it is
   wrong; the 18-field merge list is where they stop.* `survivor_scorer.py:774-779`; trained schema
   in `models/reinforcement/best_model.meta.json`.
2. **A seed's contribution to a prediction is the next value its own generator emits, weighted by
   the model's predicted quality — and it is regenerated with that seed's own skip.** — *Because
   this is the only place the skip hypothesis does causal work downstream, and no chapter walks it.*
   `prediction_generator.py:845-848`, `:530-553`.
3. **The training target is `holdout_quality`, a 50/30/20 composite of CRT match, coherence and
   temporal stability — not `holdout_hits`, not `score`, not R².** — *Because Chapter 6 still says
   `holdout_hits` and two of the three earlier targets were tautologies.*
   `holdout_quality.py:61-90`; `meta_prediction_optimizer_anti_overfit.py:1481-1482`.
4. **Six of the 89 trained features are constant.** Five dead placeholders **plus** `best_offset`. —
   *Because the "5 dead" figure is the one in circulation and it undercounts what the model actually
   sees.* `survivor_scorer.py:784-791`, `:373`, `:707`;
   `S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:128`, `:197-224`.
5. **Two step numberings; the mapping is written down nowhere.** Executable Step 1 contains the
   sieve; executable Step 2 is the Scorer Meta-Optimizer; Chapter N ≠ Step N; "Phase 7" names two
   unrelated milestones. — `agents/watcher_agent.py:387-405`; **D1**.
6. **The canonical Step-1 output is the certified NPZ generation, not any `*_survivors.json`.** —
   *Because `bidirectional_survivors.json` looks canonical and contains no seeds.*
   `window_optimizer_integration_final.py:2999-3001`.
7. **Before proposing to remove, demote or simplify ANY component, cite the document explaining why
   it exists.** — *Skip is the canonical case: three parties independently recommended deleting
   `skip_min`/`skip_max`.* Chapter 2 §5.1.
8. **"The code doesn't do X" usually means X broke, not that X was never wanted.** — *Seven
   instances of "tuned parameter never reaches the kernel" are catalogued.* Fix pattern: **one
   canonical path — resolve once in the parent, never reinterpret downstream, record
   requested/payload/effective.**
9. **Loose thresholds, forward-iterating reverse kernels, `intersection_count` duplicating
   `bidirectional_count`, `serve_timeout = None`, the bare-metal addresses in
   `distributed_config.json`, Step 0's silent failure, Chain D's `pending_approval` and the Step-5
   `allowed_params` filter are all NOT defects.** — *Each has been reported as one at least once.*
10. **The repository is not the system, and a report is a snapshot.** `.gitignore:41` is `*.json`, so
    gitignored configs are invisible to repo-scoped searches, and **the shell `grep` here is a ugrep
    wrapper that honours `.gitignore` — use `/bin/grep` for complete `.json` searches.** Re-verify
    any anchor before acting on it. **This report included.**

---

## 11. CONTRADICTIONS ENCOUNTERED

Reported, not resolved, per the brief. **Already-registered divergences are cited by their register
ID and not restated** — the full register is `PIPELINE_BEHAVIOUR_MODEL.md` §16 (D1–D18).

**Encountered and confirmed live this session:**

| # | contradiction | status |
|---|---|---|
| 1 | Chapter 6 §3.3 says `y = holdout_hits`; live target is `holdout_quality` | **D5**, GOVERNED |
| 2 | Chapter 7 describes three pools + two output files; live module emits one pool, one file, and the manifest agrees with the code | **D3**, GOVERNED as Blocker 1 |
| 3 | Chapter 4 §9.8 "47 features" / Chapter 6 §11.2 "62 features"; live is 91/89 | **D4**, GOVERNED |
| 4 | `feature_registry.json` reads `skip_min`/`skip_max` as an **output** statistic; `parameter_registry.json:160,166` reads them as a **search input** | **D12** — one is wrong; correcting it belongs to whichever change settles the semantics |
| 5 | Three READMEs describe the project as "Seed Reconstruction" / "reverse-engineer", contradicting the naming rule | **D16**, noted-not-fixed |
| 6 | The S140b objective weights three pool sizes; the live generator emits one | recorded at `PIPELINE_BEHAVIOUR_MODEL.md:1280-1282`, **explicitly not adjudicated** |

**New observations from this pass** — small, and offered as observations, not findings:

**Observation 1 — three `PIPELINE_BEHAVIOUR_MODEL.md` anchors have moved since it was written.** It
was produced at `49c13ad`; HEAD is `8bbe79e`. Re-verified this session:

| symbol | behaviour model | HEAD `8bbe79e` |
|---|---|---|
| `run_bidirectional_test` | `:1369` | `:1369` ✓ |
| `_use_miner` gate | `:1402` | `:1402` ✓ |
| `_use_pw` gate | `:1535` | **`:1547`** |
| `_use_zmq` gate | `:1580` | **`:1592`** |
| `finalize_run` call | `:2966` | **`:2972`** |
| JSON-demotion note | `:2993-2995` | **`:2999-3001`** |
| file length | 3,076 lines | **3,082 lines** |

Benign and predicted — the document's own §0 says *"Line anchors expire… cite the symbol"*, and the
behaviour model records the identical class as **D15** for Chapter 1. Recorded so the next reader
does not treat a 12-line offset as evidence of a change.

**Observation 2 — the on-disk `survivors_with_scores.json` (84 records) and the trained sidecar
(85 survivors) do not describe the same run.** The sidecar is dated 2026-03-06 and names
`survivors_with_scores.json` as its source; the file's mtime is 2026-03-11. The file was therefore
regenerated **after** the model that cites it was trained. Not a defect claim — both artifacts are
five months stale and neither is on the certifying path — but it means **the live sidecar's feature
schema cannot be assumed to match the live artifact**, and anyone using either as evidence should
say which one and when.

**Observation 3 — the METADATA-LOSS guardrail does not guard what its message claims.**
`generate_step3_scoring_jobs.py:95-100` raises if `len(result[0]) < 3` while its message says
*"Expected 20+"*. Dropping 19 of 22 fields passes silently. **Already recorded** at
`S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md:63` (*"a **weak** guard… It only catches total
collapse, not the partial loss it was written for"*) — re-verified live, repeated here only because
it sits directly on the spine traced in §6 and a reader of that section will ask about it.

---

## 12. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** every `file:line` in this report was obtained this session on VM 101 at HEAD
  `8bbe79e` by `Read`, `sed -n`, `/bin/grep -n`, or `json.load` in a read-only `python3 -c`. No line
  number is recalled. Where a claim originates in a dated document (the behaviour model, the trace
  report, the system map), it was **re-derived against live source** before restatement — the
  feature counts, the merge list, the NPZ contract, `STEP_SCRIPTS`, the backend-cascade gates, the
  Step-5 target and fold count, the Step-6 pool builder, and the Chapter-13 trigger enum were all
  read directly.
- **clean control:** the merge list at `survivor_scorer.py:774-779` is the built-in control for
  every "does not reach the ML" claim — it demonstrably *does* carry `skip_min`/`skip_max`/
  `skip_range` through to the trained schema, so the method distinguishes "carried and consumed"
  from "carried and dropped" rather than reporting everything dropped.
- **fault-injection control:** the absence claims for `window_size` / `offset` / `sessions` were
  tested against a known-present target in the same probe (`skip_min` → PRESENT, value 5.0) and
  across **four independent surfaces** — NPZ contract, merge site, trained sidecar, live artifact.
  A false absence would have had to survive all four. Separately, the first `reseed_probability`-style
  narrow grep pattern was widened before use after an earlier near-miss this session.
- **completion sentinel:** all reads and greps ran to completion; none was truncated or timed out.
  Every grep hit underpinning a claim was opened and read in context, not counted.
- **unavailable-observer behavior:** no pipeline step was executed, no GPU used, no rig contacted,
  no port bound, nothing committed. Claims about *what runs today* rest on **dated on-disk
  artifacts** (March 2026), and are labelled as such in §7.5 — they are not claims about current
  behaviour. The CA draw-procedures PDF remains **not in the repository** and is not relied on here.
- **audit claim scope:** the VM 101 working tree at HEAD `8bbe79e` (tracked + untracked) plus the
  gitignored `distributed_config.json`, `daily3.json`, `survivors_with_scores.json` and
  `models/reinforcement/best_model.meta.json`. **Repo-and-host scoped. NOT cluster-scoped, NOT
  execution-scoped.**
- **searched surfaces:** `docs/PROJECT_FILE_CATALOG.md` (index, first) · `docs/PIPELINE_BEHAVIOUR_MODEL.md`
  (§0–§20) · `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md` ·
  `docs/S172_ATTRIBUTION_AND_FEATURE_TRACE_REPORT.md` · `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` ·
  `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` · `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` ·
  `docs/CHAPTER_3_ALIGNMENT_AUDIT.md` (via the behaviour model's citations) · `docs/BACKLOG.md` ·
  the two prior Claude Code reports; code — `agents/watcher_agent.py`,
  `window_optimizer.py`, `window_optimizer_bayesian.py`, `window_optimizer_integration_final.py`,
  `prng_registry.py`, `miner/range_miner_worker.py`, `utils/canonical_arrays.py`,
  `generate_step3_scoring_jobs.py`, `survivor_scorer.py`, `full_scoring_worker.py`,
  `holdout_quality.py`, `meta_prediction_optimizer_anti_overfit.py`, `prediction_generator.py`,
  `chapter_13_triggers.py`, `chapter_13_orchestrator.py`, `hybrid_strategy.py`,
  `step6_restoration/models/global_state_tracker.py`, `trse_step0.py`.
- **unavailable surfaces:** the three rigs (not contacted) · pipeline execution (none) · the eleven
  unaudited chapters read only where cited · `instructions.txt` and `Cluster_operating_manual.txt`
  (opened only at cited anchors) · the two PDFs and the `.docx` (binary, unread) · the
  `SESSION_CHANGELOG_*` corpus (168 files, sampled not swept) · the `apply_s*.py` patch corpus
  (not opened this pass) · Optuna study DBs and NPZ artifacts (not mined) · host systemd/cron ·
  ser8 pre-repository archives · the public clone.
- **governance trail searched (`TB_RULING*`, `PROPOSAL*`, `TEAM_ALPHA*`):** YES — first, per the
  binding order, primarily via `PROJECT_FILE_CATALOG.md` §1 and the behaviour model's WHY anchors,
  which pair each ruling to its implementation. Load-bearing: `TB_RULING_REQUEST_STEP2_v4_1/v4_2`,
  `TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md`, `TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md`,
  `TB_UPDATE_SELFPLAY_REFRAMING_2026-07-28.md`, `TEAM_ALPHA_D3_0_B_AND_ITEM1_NOTICE.md`,
  `PROPOSAL_S172_RANGE_MINER_v1_4_4/v1_4_5.md`, `TRSE_INTEGRATION_PLAN_S121.md`,
  `PROPOSAL_Feature_Implementation_Remediation_v1_0.md`, `PROGRESS_Feature_Remediation_v1_0.md`.
- **chapters searched:** 1 (§8.3.1, §12.1), 2 (§1–§8, §11), 3 (via its audit), 4 (§1.4, §9.8 via
  citations), 5 (§1.2, §3.2–§3.3, §7.3 via citations), 6 (§2.1–§3.7, §9 via citations), 7 (§1.3,
  §3.1, §4.1, §6.2 via citations), 12 (§3.3, §8, §11.5 via citations), 13 (§2, §3.2, §6.1, §9,
  §11 via citations).
- **termination:** **PASS** (VIR-3). All nine required sections are answered from primary sources
  with anchors. The brief's central question is answered directly and four-ways-verified. One clause
  of the owner's framing is **refuted**, one **refined**, three **confirmed**.

**Nothing was proposed, recommended or implemented, per the brief.**
