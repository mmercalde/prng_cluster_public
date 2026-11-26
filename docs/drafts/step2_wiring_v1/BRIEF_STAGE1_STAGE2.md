# BRIEF: Step 2 settings wiring, Stage 1 (pin) and Stage 2 (safe contract) (2026-11-22)

Design only. No code has been written. Base: worktree `pwc-offset` at `ab2d330`. Every Step 3/5/6 file cited below is byte-identical to the main tree at `41ad509` (sha256 compared this session). Line numbers are for that content.

## Summary

1. **Ruled:** the owner approved the four-stage plan at 19:28. That is Beta's order in `X-STEP2-WIRING_R1_BETA_REVIEW.md` §4. O1 (wire the eleven settings) and O2 (v4.3 objective unchanged, `max_offset` item parked) stand.
2. **Stage 1** pins Step 2's output to the run that made it.
   - The file carries a schema version, the producer run and trial, a config digest, and the identity of every input Step 2 read.
   - Steps 3 and 5 refuse a config that is missing, malformed, or made from different inputs.
   - Today they silently fall back to defaults (`run_step3_full_scoring.sh:167-169`, `generate_step3_scoring_jobs.py:352-355`).
   - Stage 1 **delivers** the values and records them. It switches nothing on: every feature stays byte-identical.
3. **Stage 2** makes Steps 3, 5 and 6 safe for changed feature names before any setting is switched on.
   - Every artifact carries one versioned, ordered feature contract that includes the effective settings.
   - The `holdout_quality` inputs no longer depend on the residue names.
   - Step 5 and Step 6 stop turning a missing feature into 0.0.
4. **Stage 2 is mostly already authorized.** It is the first slice of the system map's P0-A and P0-B (`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:247-253`). Both are recorded OPEN and not started (`TODO_CONSOLIDATED.md:61-62`, `:160-161`). Nothing in this brief is a new finding.
5. **`max_offset`:** delivered to Step 3 and recorded there, but inactive. `best_offset` stays 0 (Chapter 4 B2, `survivor_scorer.py:373`, `:707`).
6. **No constant-skip rows exist in any measured bundle on this box** (8 bundles under `logs/skipsw_session/` checked; all rows are `skip_mode` 1). Every fixture here is therefore partly synthetic: constant-skip and variable-skip rows, built to the 22-key NPZ contract.
7. **Coordination:** agent 3 is coding plan changes C2 and C1 in `~/tfm_worktrees/step2-repair`.
   - C1 edits `run_scorer_meta_optimizer.sh`. Stage 1's producer change touches the same block (`:302-309`), so Stage 1 lands after C1 and is rebased onto it.
   - Stage 1 also absorbs the `_run_context` part of plan C5.
8. **Owner direction, 20:37–20:46 (binding):** the focus is now one thin path from Step 1 to Step 6, with pools scored on unseen draws, then selfplay. Thin does not mean sloppy. This revision applies the direction in four ways:
   - **Configurable and observable:** every threshold, weight, range and feature list this brief introduces is read from one declared source and written into the artifacts (§0).
   - **No intended path deleted:** the config's `prng_type`/`mod` override and the Step 3 `< 46` floor are kept, not removed.
   - **Shortcuts recorded:** every shortcut is listed with its anchor as a TODO (§7).
   - **Step 6 is now on the critical path.** Stage 2 makes Step 6 fail closed, and Step 6's current input is a stub (Chapter 7 B1). With both in place, **the thin path produces no pool until W2 is answered**, so W2 moves to the top of the questions.

**Owner questions**

- **W2 first (thin path).** Should Step 6 rank Step 3's scored rows (`survivors_with_scores.json`, the population Step 5 trained on) instead of the forward∩reverse integer intersection it rebuilds today? Details are under W2 below. Recommendation: yes. Do it as its own small change, S2.7, behind a `--survivor-source scored|intersection` flag:
  - `scored` is the default for the thin path;
  - the `intersection` path is kept;
  - the change ships with tests T2.7.
- **O4 (listed, not decided):** untrack `optimal_scorer_config.json` from git? Stage 1 does not depend on the answer. Its identity check rejects the tracked 2026-03-11 file whatever the answer.
- **W1, the `holdout_quality` inputs.**
  - (a) Keep the label recipe fixed: compute its inputs with a default-settings scorer on the holdout. The label stays what it is today.
  - (b) Key the label to the configured moduli through an explicit slot mapping, and version the target.
  - Option (a) keeps the S111 formula that "Team Beta approved" (`holdout_quality.py:82`). Option (b) changes the label. Beta's first preference is (a).
- **W2, the Step 6 input.**
  - Under Stage 2, Step 6 refuses a row that lacks a model feature. On the WATCHER path, Step 6 rebuilds features for integer seeds with a default scorer. By reading, those rows have no `global_*` or metadata fields, so they would now be refused rather than zero-filled.
  - That path already exits 1 on the stub survivor file (Chapter 7 audit B1), so no working output is lost.
  - Should Step 6 instead read Step 3's scored rows (Beta's suggestion)? That is the Chapter 7 B1 repair; the design is S2.7 and its tests are T2.7.
- **W3, legacy artifacts.** Keep an explicit `--legacy-step2-config` / legacy-contract mode so March-2026-era files stay readable? Or refuse them outright?
- **W4, P0-A scope.** Stage 2 builds P0-A items 1, 2, 4 and 6 for the Step 3/5/6 path only. It leaves item 3 (remove the duplicated static lists) and item 5 (classify each field) for P0-A proper. Is that split acceptable?

## Prior record

**Searches** (`pwc-offset`, read-only, 2026-11-22):
- `/bin/grep -lnE "P0-A|F-PARITY|feature_manifest|FEATURE_CONTRACT|feature contract"` over `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`. Hits: Chapter 4, 6, 11, 13 and 14 audits and chapters, the corrections report, TODO, and Beta R1.
- `/bin/grep -rlnE "feature_schema_hash|validate_feature_schema_hash|feature_manifest" --include=*.py --include=*.json`. Hits: `models/feature_schema.py`, `meta_prediction_optimizer_anti_overfit.py`, `prediction_generator.py`, and `backups/` copies.
- `/bin/grep -lnE "optimal_scorer_config"` over the same governance set (20 files, listed in X-STEP2-WIRING R1). `/bin/grep -rln feature_registry.json --include=*.py`: no reader.
- `git ls-files config_manifests/`. Skip modes in the 8 `logs/skipsw_session/*/survivors.npz`.

**Documents that decide or record the subject:**

| # | Record | What it settles | Label |
|---|---|---|---|
| P1 | System map §13 P0-A/P0-B/P0-C, `:247-253` ("Team Beta, binding"; advisory under the 2026-11-11 owner rule) | Canonical 91/89 manifests, exact name/count/order/hash validation, schema version in every Step 3/5/6 artifact; Step 3 batch/sequential parity, F-PARITY, fail closed | ALREADY RECORDED; OPEN, not started (`TODO_CONSOLIDATED.md:61-62`) |
| P2 | `TFM_PROJECT_FACTS_SKILL.md:673-680`, `:713-719` (§2.10 dataset authority) | Resolve inputs once at run start and freeze; dispatch by immutable path; "append-only does NOT make prior scores valid on the next version"; input identity is a lineage invariant | ALREADY RECORDED; Stage 1 applies it to Step 2's output |
| P3 | `CHAPTER_6_AUDIT.md:219-230` (B5) | Step 6 never calls the hash validator; a renamed feature becomes a silent 0.0 column (`pg:798`) | ALREADY RECORDED |
| P4 | `CHAPTER_13_AUDIT.md:80-105` (B2–B6), `:553-569` (B19) | The relearn loop reads other history files; it runs Step 4 too; diagnostics read the wrong sidecar path and keys | ALREADY RECORDED; Stage 4 |
| P5 | `CHAPTER_4_FULL_SCORING.md:1050-1089` (§9.9) | The 91/89 breakdown; "the count is conditional on the default residue mods"; globals merged twice | ALREADY RECORDED |
| P6 | `CHAPTER_4_FULL_SCORING.md:1454-1459` (§16.1 B2, B4, B5, B6) | `best_offset` always 0; the fallback is not mod-reduced; Step 2 parameters never reach Step 3; the sequential fallback merges 6 of 18 metadata fields | ALREADY RECORDED (B6 governed under P0-B) |
| P7 | `CHAPTER_7_AUDIT.md:74-92`, `:175-192` (B1) | Step 6 works on integer seeds from the forward∩reverse intersection and recomputes their features; its survivor input is a count-only stub | ALREADY RECORDED; W2 |
| P8 | `S111` / `holdout_quality.py:61-125` | The label formula is keyed to `residue_1000/8/125_*`, "Team Beta approved" | ALREADY RECORDED; W1 |
| P9 | `X-STEP2-WIRING_R1_BETA_REVIEW.md` §1–§4 and Finding 1 | The design used here; Beta's positions are advisory | ALREADY RECORDED |
| P10 | `docs/drafts/PLAN_STEP2_REPAIR_20261122.md` C5; `S107_session_log.md:59`, `:70` | `_run_context` block, provenance | ALREADY RECORDED; folded into Stage 1 |
| P11 | Owner rulings O1, O2 (`logs/skipsw_session/OWNER_RULING_step2_O{1,2}_20261122.txt`); 16:39/16:42; 19:28 | Wire; objective unchanged; constant-skip never dropped; no include/remove decision from test runs; four stages approved | Ruled |
| P12 | `logs/skipsw_session/OWNER_DIRECTION_e2e_20261122.txt` (sha `dd1a7bd8…`), with the 20:46 rules relayed by the lead | One thin path, Step 1→6 then selfplay. Correct, tested, fail-closed; nothing hard-coded that may move; small, modular, reversible; keep unused-but-intended paths; every shortcut is a TODO with its anchor | Ruled |
| P13 | `logs/skipsw_session/OWNER_NOTE_step2_v42_terms_20261122.txt` (sha `584cb870…`) | The early S107/S108 removals were probably premature; do not delete the unused v4.2 locals | Ruled; applies by analogy, keep intended paths |

**Labels:**
- The Stage 1 artifact schema and validator: **NEW** as a design, implementing P2 and P10.
- Stage 2: **ALREADY RECORDED** (P1, P3, P6), now scheduled.
- W1 option (b) would **CHANGE A RULING** (P8); it needs the owner.

**Not searched:** host state beyond the main tree, ser8 archives, the private repository, files outside the repository.

## 0. Rules for whoever implements this

- **Owner rules:**
  - no commits, no pushes;
  - no GPU or rigs unless authorized;
  - no main-tree edits;
  - no live ledger;
  - one full test pass at a time;
  - output under `data/`;
  - recorded PIDs only.
- **Never edit agent 3's files in its worktree.** Stage 1 rebases onto C1 once C1 has landed.
- **Yardstick:** at default settings, Stages 1 and 2 must leave every feature value, label, model input and prediction byte-identical. Each stage includes a parity test that proves it. Any hit-rate effect can come only from Stage 3, and is measured only in Stage 4.
- **Never drop a survivor row.** Every test asserts the row count per `skip_mode` in equals the count out.
- **Owner direction, 20:46 (P12):**
  - **Configurable:** every value this brief introduces has one declared source and a default equal to today's behaviour. That covers ranges, label settings, label weights, strictness switches and the skip rule. The value in effect is written into the artifact it shaped.
  - **Fail closed by default.** A switch that relaxes a check is opt-in, and when used it is recorded in the artifact (for example `legacy: true`).
  - **Small and reversible:** one concern per change. A new module for each new rule (`utils/step2_config.py`, `utils/feature_contract.py`). Every call-site edit can be reverted on its own.
  - **No intended path deleted:** paths are kept, and gated or recorded instead.
  - **Every shortcut** gets a `TODO(step2-wiring #n)` comment at its anchor and a row in §7.

## 1. Stage 1: pin Step 2's output to its run

### S1.1 Producer (Step 2)

- **Site:** `run_scorer_meta_optimizer.sh:302-309` writes `best.params` flat to `optimal_scorer_config.json`, with no identity. C1 changes this script, so land after C1.
- **Change:** the file keeps the eleven parameters flat at the top level, so `SurvivorScorer`'s translation (`survivor_scorer.py:97-111`) still reads them. It adds one block, `_run_context`:
  - `schema_version: "step2_config/1"`;
  - `producer` = step 2, study name, study DB path, best trial number and value, objective `v4.3`, the `sample_size` actually run, code commit;
  - `inputs`: path, sha256 and size of each required input in `agent_manifests/scorer_meta.json:6-10` (`bidirectional_survivors_binary.npz`, `train_history.json`, `holdout_history.json`), plus the NPZ's `generation_id` when present;
  - `config_digest` = sha256 of the canonical JSON (sorted keys, no whitespace) of the eleven parameters plus `_run_context` without the digest.
- **Write:** atomic (temporary file, then `os.replace`).
- **Files:** `run_scorer_meta_optimizer.sh` (owner_land; Step 2 launch path) and a new `utils/step2_config.py` (owner_land). Not governed. No hand-landed file is touched.

### S1.2 One validator, used by every consumer

`utils/step2_config.py: load_pinned(path, *, expect_inputs) -> PinnedConfig`. It raises `Step2ConfigError` naming the path and the reason when:

- the file is missing or unparseable;
- `schema_version` is unknown;
- any of the eleven keys is missing, the wrong type, or outside Step 2's declared range (three distinct moduli; `hidden_layers` one of the declared strings);
  - The ranges are not copied into the validator. Step 2's search space is declared once (`generate_scorer_jobs.py:54-66`, lifted into a module-level table that `define_search_space` reads).
  - `_run_context` records the ranges in force, so that if the owner or the ML moves a range, an older config is still judged against the ranges it was made under.
- the digest does not match;
- an input identity does not match what the consumer is about to read.

The legacy mode (W3) is opt-in by flag and records `legacy: true` in every artifact it touches. PRNG family and output `mod` stay under the run's existing input authority (Beta R1 §1). Today the generator lets the config override them (`generate_step3_scoring_jobs.py:205-207`). That path is **kept**, but fails closed: if the config carries `prng_type` or `mod` and the value differs from the run's, the generator exits non-zero naming both values. An equal value is accepted and recorded.

### S1.3 Step 3 consumer

| Site | Today | Stage 1 |
|---|---|---|
| `run_step3_full_scoring.sh:69`, `:167-169` | passes `--config` only if the file exists | always passes it; a missing file fails before Phase 1 |
| `generate_step3_scoring_jobs.py:352-355` | a missing config prints WARNING and uses defaults | `load_pinned(..., expect_inputs={npz: --survivors, train: --train-history, holdout: --holdout-history})`; any error exits non-zero before a job is written |
| `generate_step3_scoring_jobs.py:241-248` | job args carry `prng_type` and `mod` only | adds one argument, `--scorer-config-json '<canonical JSON of the 7 scorer keys + config_digest>'` |
| `full_scoring_worker.py:522-550`, `:344` | no config argument; `scorer_class(prng_type, mod)` | parses and validates `--scorer-config-json` (digest re-checked), and **records** it in each result's `metadata.step2_config` as `{digest, delivered: {7 keys}, applied: false}`. The scorer is still built with defaults. |
| `run_step3_full_scoring.sh:369-408` (aggregation) | merges chunks | refuses mixed `config_digest` across chunks; writes the digest into `survivors_with_scores.json` metadata |

Beta's reasoning for one JSON argument rather than a worker-side file path (R1 §1): the ramdisk preload stages the default filename before option parsing, so the worker could otherwise read a different file.

### S1.4 Step 5 consumer

- `meta_prediction_optimizer_anti_overfit.py:3037-3130`: a new `--step2-config` argument, validated with `load_pinned`. Its digest must equal the digest stamped in `survivors_with_scores.json`. A mismatch exits non-zero before fitting.
- Pass it through the compare-models child argv (`:214-262`).
- Record digest and `_run_context` in `data_context` (`:1498`) and in the sidecar `provenance` (`:2899-2925`).
- The NN values are **not** used yet; that is Stage 3.
- The manifest default (`agent_manifests/reinforcement.json:173-183`) gains `step2_config`. **That is a hand-landed file.**

### S1.5 Step 6

Copy the sidecar's `step2_config_digest` into `next_draw_prediction.json` `agent_metadata`. No check: Step 6 does not read the scored artifact today (P7).

### S1.6 O4

Listed, not decided. With S1.2 in place, the tracked March file fails `schema_version` and the input identity check, so nothing stale can pass silently either way.

### Stage 1 tests

All CPU-only. Fixtures under `data/`. Subprocesses stubbed where a fleet would run.

- **T1.1 producer:** a fake Optuna study with distinctive non-default values for all eleven keys yields a file whose `_run_context` matches the fixture inputs and whose digest recomputes. A mutant that drops one key goes red.
- **T1.2 Step 3 round trip:** all eleven values reach `scoring_jobs.json` args and the worker's `metadata.step2_config`; `applied:false`.
- **T1.3 fail closed.** Each case exits non-zero, and no job file is written:
  - missing file;
  - missing key;
  - out-of-range value;
  - digest tampered;
  - NPZ sha different (the "fresh-looking but unrelated" case);
  - the tracked 2026-03-11 file.
- **T1.4 legacy:** the March file is accepted only with the legacy flag, and `legacy:true` is recorded.
- **T1.5 mixed chunks** at aggregation are refused.
- **T1.6 Step 5:** a digest mismatch between `--step2-config` and the scored artifact exits before fitting. The child argv carries the path.
- **T1.7 parity:** on a 200-row fixture containing **both** `skip_mode` 0 rows (`skip_min == skip_max`) and `skip_mode` 1 rows (`skip_min < skip_max`), Step 3 output `features` and `holdout_quality` are byte-identical before and after Stage 1, and the row count per `skip_mode` is preserved.

**Beta:** the Step 3 launch path and worker are BEFORE-COMMIT classes (CLAUDE.md "Beta reviews"). Send Stage 1 as one review request. Step 5 and Step 6 go in the same request.

## 2. Stage 2: make Steps 3, 5 and 6 safe for changed feature names

### S2.1 The contract (P0-A items 1, 2, 4, 6)

A new `utils/feature_contract.py` builds the contract from the scorer's effective settings, not from a static list:

- `contract_version`;
- the ordered extracted names (91 at defaults);
- the ordered model-input names (extracted minus `score` and `confidence`, 89; today at `:1479-1483`);
- the effective settings (all seven, with `max_offset` marked `active_in_step3: false`);
- `contract_hash` = sha256 over names **and** settings.

The names-only hash today (`mpo:586-588`, `:603`) cannot see a change in temporal settings, because those keep their names (Beta R1 §2). The existing `feature_schema_hash` stays, for reading old sidecars.

### S2.2 Step 3 producer

| Site | Today | Stage 2 |
|---|---|---|
| `full_scoring_worker.py:386-431` batch and `:440-491` fallback | the fallback merges 6 of 18 metadata fields (`:453-456`; Ch4 B6) | the fallback uses the same merge as the batch path (P0-B item 2); every result's key set must equal the contract or the chunk fails |
| `full_scoring_worker.py:425-430`, `:486-491` | a label exception becomes `holdout_quality = 0.0` | an exception fails the chunk. A run with no holdout history is recorded as `holdout_quality: null` with a reason, never a fabricated 0.0. **W1 decides the inputs** (S2.3) |
| `run_step3_full_scoring.sh:369-408`, `:475-478` | floor `< 46` | write `feature_contract` into the output and check every row against it exactly. The `< 46` floor stays as a second, weaker check; a TODO marks it for retirement once P0-A lands |

### S2.3 `holdout_quality` (W1)

The label reads `residue_1000_*`, `residue_8_*` and `residue_125_*` by name (`holdout_quality.py:101-111`). With non-default moduli it would silently lose 40% of the CRT term and all of the coherence term. Beta's isolated check: 0.3447 → 0.0850 on the same vector.

- **Option (a), recommended:** the worker builds a second scorer for the label inputs only.
  - Its settings come from a declared `label_settings` block whose default is today's scorer defaults (`survivor_scorer.py:74-78`). They are not hard-coded at the call site.
  - `compute_holdout_quality` keeps its formula. It requires all seven names it reads, and raises if one is absent instead of using its `_get` default (`holdout_quality.py:92-96`).
  - `holdout_features` stays from the configured scorer and is not a model input.
- **Option (b):** pass an explicit slot mapping and add `target_version`. This changes P8 and needs the owner and Beta.
- **Either option, owner direction "weights configurable":**
  - The eleven label weights (`holdout_quality.py:100-121`: 4 CRT, 3 coherence, 1 temporal, 3 composite) move into a versioned `label_recipe` with exactly today's values as the default.
  - `target_version` plus the recipe digest go into every scored row's metadata and into the Step 5 sidecar.
  - The values do not change, so the TB-approved formula (P8) is not changed. T2.3 proves bit-identity.
  - Beta reviews the move with the rest of Stage 2.

### S2.4 Step 5 consumer

| Site | Today | Stage 2 |
|---|---|---|
| `mpo:546-596` | names inferred from the **first** record, sorted | read `feature_contract` from the artifact; every row must match it exactly, or the run fails naming the row and the key |
| `mpo:651`, `:678` | `features.get(fn, 0.0)` | missing feature → error |
| `mpo:660`, `:686` | missing target → 0.0 | missing or `null` target → error |
| `mpo:1738-1742` | study name keyed by the names hash | keyed by `contract_hash`, so a renamed or re-meant contract never resumes an old study |
| `mpo:2899-2905` sidecar | `feature_schema` (names, hash) | adds `feature_contract` and `step2_config_digest`; schema version raised |

The training scaler and checkpoint are written with the same contract (Beta R1 §2).

### S2.5 Step 6 consumer

| Site | Today | Stage 2 |
|---|---|---|
| `pg:350-353` | `load_model_from_sidecar` without `survivors_file`, so hash checks never run (Ch6 B5) | compare the sidecar's `feature_contract` with the contract of the rows about to be scored, before `predict`; a mismatch exits non-zero |
| `pg:798` | `features.get(name, 0.0)` | a missing name → error naming it |
| `pg:816-824` | NN scaler width mismatch → warning, normalization skipped | → error |
| `pg:263-266`, `:785-790` | integer survivors rebuilt by a default `SurvivorScorer` | unchanged in Stage 2; W2 decides |

### S2.6 `max_offset`

It is delivered (S1.3) and listed in the contract with `active_in_step3: false`. `best_offset` remains the constant 0 it is today (`survivor_scorer.py:373`, `:707`; Ch4 B2). Activating it needs a defined Step 3 alignment rule, which no record specifies; it is not in Stages 1–4. Step 2's own use (`scorer_trial_worker.py:380-389`) is untouched (O2).

### S2.7 Step 6 ranks Step 3's scored rows (W2; thin path)

**Why it is on the thin path:**
- On the WATCHER path Step 6 reads `forward_survivors.json`/`reverse_survivors.json` (`agent_manifests/prediction.json:178-179`). These are count-only stubs, so Step 6 exits 1 (Chapter 7 B1).
- Even with real lists, Step 6 would rank the integer intersection, re-extracting its features with a default scorer (`pg:606-616`, `:785-790`). That is not the population Step 5 trained on.

**Change** (only if the owner answers W2 yes):
- **New argument:** `--survivor-source scored|intersection`. The default is `scored` only once the manifest is changed by hand; until then the code default stays `intersection`, which is today's behaviour.
- **`scored`:**
  - reads `--survivors-scored survivors_with_scores.json`, validates its `feature_contract` against the sidecar (S2.5) and its `step2_config_digest` against the sidecar's (S1.5);
  - uses each row's stored `features`, through the existing dict branch (`pg:782-784`), with no re-extraction;
  - skips the dual-sieve intersection, because Step 3's input was already the bidirectional set (`agent_manifests/full_scoring.json:118`). `compute_dual_sieve_intersection` would coerce dict rows to integers (`ss:128-137`).
  - The pool records `method: scored_rows`.
- **`intersection`:** today's path, kept unchanged for comparison.

**Skip used for the vote** (by reading, this session):
- Step 6 takes a dict row's skip from `skip`, `metadata.skip` or `features.skip_hypothesis`, else the global default (`pg:530-552`). Scored rows carry `skip_min`/`skip_max` in `features` (`full_scoring_worker.py:453-456`; batch merge list `survivor_scorer.py:774-779`) and none of those three keys. So every scored row would vote with the default skip.
- Step 3's label uses a different rule: `skip_best`, then `skip_min`, then 0 (`holdout_quality.py:26-50`).
- Under `scored`, both steps call **one** declared skip resolver whose default is the Step 3 rule. Each pool entry records the skip it used and the rule's name.
- **Shortcut, recorded as TODO #6:** a variable-skip row votes with one skip value (`skip_min`), not its skip sequence.

**Files:** `prediction_generator.py`, `holdout_quality.py` (the shared resolver), and `agent_manifests/prediction.json` (**owner by hand**).

### Stage 2 tests

All fixtures carry **both** constant-skip and variable-skip rows. Each test asserts that row counts per `skip_mode` are preserved.

- **T2.1 contract:** at defaults it reproduces 91/89 names in the current order. Changing one modulus, `temporal_window_size`, `temporal_num_windows` or `min_confidence_threshold` changes `contract_hash`; the last three keep the names, and the hash still changes.
- **T2.2 Step 3 parity:** force the sequential fallback; it emits the same key set and values as the batch path (F-PARITY). A mutant that restores the 6-field merge goes red.
- **T2.3 label:**
  - (a) with a contract built from non-default moduli, `holdout_quality` equals the default-run value to the bit;
  - a holdout row missing `residue_1000_match_rate` raises;
  - a mutant that restores `_get(...,0.0)` goes red.
- **T2.4 Step 5:** these inputs fail before fitting:
  - a scored file whose 2nd row lacks one feature;
  - a renamed row of the same width;
  - a `null` target;
  - a file whose rows mix two contracts.
  At defaults, X and y are byte-identical to today.
- **T2.5 Step 6, both pairings:**
  - an old sidecar with new rows, and a new sidecar with old rows: each exits non-zero naming the first missing feature;
  - NN scaler width mismatch → error;
  - at defaults with dict rows, the ranking on a toy model is identical to today's.
- **T2.6 end-to-end parity at defaults:** Step 3 → Step 5 → Step 6 on the fixture gives byte-identical features, labels and pool to the pre-Stage-2 code. CPU only; tiny trial counts.
- **T2.7 Step 6 `scored`:**
  - On a fixture with both constant-skip and variable-skip rows, every scored row reaches ranking (count per `skip_mode` preserved), and the model input equals the stored `features` in contract order.
  - Each pool entry records its skip and the rule. A constant-skip row's recorded skip equals its `skip_min == skip_max`.
  - A digest or contract mismatch exits non-zero.
  - `--survivor-source intersection` reproduces today's output byte for byte.
- **T2.8 configurability:**
  - Changing `label_settings`, one label weight or one declared Step 2 range changes the recorded digest or version in the artifact. At defaults, every artifact is byte-identical apart from the new identity fields.
  - A mutant that hard-codes a weight at the call site goes red, because the test reads the weight back from the recorded recipe.

**Beta:** BEFORE-COMMIT for the Step 3 worker and aggregation. The Step 5 and Step 6 changes alter the training and inference contract; send them in the same request. `holdout_quality` option (b) would need its own request.

## 3. Landing table

| Change | Files | Landing | Governed? | Beta |
|---|---|---|---|---|
| Stage 1 | `run_scorer_meta_optimizer.sh` (after C1), `utils/step2_config.py` (new), `run_step3_full_scoring.sh`, `generate_step3_scoring_jobs.py`, `full_scoring_worker.py`, `meta_prediction_optimizer_anti_overfit.py`, `prediction_generator.py`, new tests | owner_land | No | Yes, before commit |
| Stage 1 manifest | `agent_manifests/reinforcement.json` (and `full_scoring.json` if a default changes) | **owner by hand** | No | in the same request |
| Stage 2 | `utils/feature_contract.py` (new), `full_scoring_worker.py`, `holdout_quality.py`, `run_step3_full_scoring.sh`, `meta_prediction_optimizer_anti_overfit.py`, `prediction_generator.py`, new tests | owner_land | No | Yes, before commit |
| S2.7 (if W2 = yes) | `prediction_generator.py`, `holdout_quality.py` (shared skip resolver), tests | owner_land | No | Yes, same request as Stage 2 |
| S2.7 manifest | `agent_manifests/prediction.json` (`survivor_source`, `survivors_scored`) | **owner by hand** | No | same request |
| Step 2 range table | `generate_scorer_jobs.py` (lift `:54-66` into a declared table; values unchanged) | owner_land | No | with Stage 1 |

Nothing touches `GOVERNED_FILES` (`scripts/gate12_parity_gate.py:174-185`), `miner/`, or `agents/watcher_agent.py`. Freshness binding in WATCHER stays plan C4.

## 4. Stages 3 and 4 (outline only)

**Stage 3: switch on, behind a flag, with the baseline kept.**
- Step 3: `--apply-step2-scorer-config`, default off. When on, the worker builds `SurvivorScorer(prng_type, mod, config_dict=delivered)` and records `applied: true`. The current defaults stay the baseline configuration (yardstick rule 4).
- Step 5:
  - Fix the NN refit first. Fold evaluation builds `hidden_layers`/`early_stopping_patience`, but refit passes raw `n_layers`/`layer_i`/`patience` (Chapter 6 B3; Beta R1 §3).
  - Then add the Step 2 NN tuple as **one declared starting candidate** (Beta's preferred option): `512_256_128` decoded to `[512, 256, 128]`, evaluated on the same folds.
  - The native sampler (`mpo:2463-2476`) cannot represent that tuple, so the candidate needs its own evaluation path or a versioned study space.
  - Its batch size takes effect only on the mini-batch path; disclose this in the comparison.
- Tests:
  - the candidate wins once and a native trial wins once;
  - the reloaded checkpoint matches the sidecar;
  - there is no duplicate candidate on resume.

**Stage 4: Chapter 13, then measurement.**
- Stubbed-subprocess test that one pinned context reaches 3→5→6 through `agents/watcher_dispatch.py:297-344`, not the contiguous range that also runs Step 4 (Ch13 B6).
- Dataset authority for relearn (Ch13 B2, P2).
- The sidecar reader's path and keys (Ch13 B19).
- §18 table update.
- Then run baseline and switched-on variants against the same unseen draws at the same `k`. Report hit counts, the paired difference and its uncertainty, against `k/1000`.

## 5. What this brief does not do

- No change to the v4.3 objective or to `scorer_trial_worker.py` (agent 3's C2).
- No parameter is removed.
- No survivor or population verdict; no state-recovery framing.
- No WATCHER change.
- No decision on O4 or W1–W4.

## 6. Hit rate

Stages 1 and 2 are built to leave hit rate unchanged: T1.7 and T2.6 prove byte-identity at defaults. Their effect is that a mismatched or renamed input stops the run instead of reaching the pool as zeros.

S2.7 is different: it changes which population Step 6 ranks, from the default-scored intersection to Step 3's scored rows. Under `scored` the pool can differ, so the first thin-path pool should be scored on unseen draws alongside an `intersection` pool when one can be produced. Whether the Step 2 settings themselves change hit rate is unknown until Stage 4 measures it.

## 7. Shortcuts and deferred items (each becomes `TODO(step2-wiring #n)` at its anchor)

| # | Shortcut / deferral | Anchor | Revisit when |
|---|---|---|---|
| 1 | `max_offset` delivered to Step 3 but inactive; `best_offset` is a constant 0 | `survivor_scorer.py:373`, `:707`, `:108` | a Step 3 alignment rule is specified (Ch4 B2) |
| 2 | P0-A items 3 (duplicated static feature lists) and 5 (classify local/global/dead/label) not done | system map `:249-250`; `TODO_CONSOLIDATED.md:61` | P0-A proper (W4) |
| 3 | `< 46` Step 3 output floor kept beside the exact contract check | `run_step3_full_scoring.sh:475-478` | P0-A lands |
| 4 | Run-global features merged twice; the aggregator builds `GlobalStateTracker` with a literal `{'mod': 1000}` | `run_step3_full_scoring.sh:340`; `full_scoring_worker.py:347-349` | the run's `mod` authority is threaded through (Ch4 §9.9) |
| 5 | The `intersection` Step 6 path still rebuilds integer survivors with a default scorer and no `global_*`/metadata fields; it fails closed under S2.5 | `pg:263-266`, `:785-790` | owner decides whether `intersection` stays a comparison path |
| 6 | Variable-skip rows vote with one skip value (`skip_min`), not their sequence | shared resolver; `holdout_quality.py:26-50`; `pg:530-552` | selfplay / a Step 6 sequence-vote design |
| 7 | Constant-skip fixture rows are synthetic; no measured bundle on this box has one | `logs/skipsw_session/*/survivors.npz` (8 checked) | a full search yields constant-skip survivors |
| 8 | Legacy mode for pre-Stage-1 configs and contracts (if W3 keeps it) | `utils/step2_config.py` | the owner retires March-era artifacts |
| 9 | WATCHER freshness still judges Step 2 by mtime, not by `_run_context` identity | `agents/watcher_agent.py:962-994` | plan C4 (hand-landed) |
| 10 | Chapter 13 relearn reads other history files and the wrong sidecar path | Ch13 audit `:80-105`, `:553-569` | Stage 4 |
| 11 | The Step 2 objective ignores 7 of 11 settings and the v4.2 match-data terms (O2, owner note 20:26–20:35) | `scorer_trial_worker.py:380-389`; owner note P13 | a full-search pool exists |
