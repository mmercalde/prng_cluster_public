# Registry repair: the 20 out-of-date parameter-registry entries (owner ruling #7, 2026-11-24)

## Plain summary

The owner ruled at 10:10: "#7 yes fix those but document everything". That means: correct each out-of-date entry in `agents/registry/parameter_registry.py` that has a real equivalent in the script it names, drop the rest, and document each one.

**Result: all 20 are DROPPED and none is corrected.** None of the 20 names a flag that its script accepts with the same meaning.

- **Steps 1, 4, 5 and 6 (14 entries).** The script exists, but the idea lives somewhere other than a command-line flag:
  - the Optuna search range in `distributed_config.json` `search_bounds`;
  - a trial parameter inside the trainer;
  - a config-file key;
  - the operator-only warm-start pin.

  The registry describes command-line flags only (`ParameterSpec.name`: "matches CLI arg without dashes", `parameter_registry.py:34-37`). So none of these can be an entry.
- **Steps 2 and 3 (7 entries).** These name two scripts that never existed, `scorer_meta_optimizer.py` and `full_scoring.py`. Both registries are removed. The table names the live script that does each job today.
- **Step 1 `offset`.** The approved design renames it to `window_anchor` (Brief II naming; CHAPTER_10_AUDIT S1). No `--window-anchor` flag exists yet, so it is **dropped, not renamed**. Nothing was invented.

**What the change does:**
- The registry goes from 29 specs to **9**.
- The 9 good specs are **byte-for-byte unchanged**. So is everything outside `PARAMETER_REGISTRIES`.
- Each dropped entry leaves a dated comment pointing to its row below.
- **Draft diff only**: 188 lines removed, 15 comment lines added, nothing applied.
- `agents/registry/parameter_registry.py` is not governed or hand-landed (`scripts/gate12_parity_gate.py:174`; `scripts/owner_land.py:86-87`). It is still a draft for the owner to land.

**Tests:**
- The repaired file imports under Pydantic, and every kept spec's flag is defined by its script at `41ad509`.
- The main tree's own `test_subphase1.py` passes on a full `41ad509` copy with the repaired registry, as it does on the original. The only difference is "Registry has 5 parameters" instead of 9 for `window_optimizer.py`.
- **87/87** mutants are killed and credited:
  - 6 on the repaired file;
  - 7 on the repair code;
  - the rest of the suite.

**Item 3: options that change results.** Of the **174** unregistered WATCHER step-script options from yesterday's check, **46 rows change results**. Counting the 5 options added identically by two worktrees once, that is **41 distinct options**.

| Step | Rows that change results |
|---|---|
| Step 0 | 2 |
| Step 1 | 15 |
| Step 2 | 6 |
| Step 3 | 8 |
| Step 4 | 1 |
| Step 5 | 12 |
| Step 6 | 2 |

The other 128 rows:

| Class | Rows |
|---|---|
| Paths and output names | 54 |
| Infrastructure | 32 |
| Not run by the step | 15 |
| Logging / test | 14 |
| ids / metadata | 9 |
| Guards | 4 |

**Since the 174:** worktrees added 12 more step-script options; 4 of those change results.

**Nothing removed from any pipeline script.** No commit, no push, no hand-landed file touched. Everything ran on CPU cores 6–7 under `nice -n 15`, at most 2 processes, and no bytecode was written.

## The 20 entries

Registry lines are at `41ad509`. "Live" anchors are at `41ad509` unless noted.

| # | Script (registry key) | Old name / flag (registry line) | What the live script actually accepts | Action | Why |
|---|---|---|---|---|---|
| 1 | `window_optimizer.py` | `window_size` `--window-size` (`:272`) | no such flag (window_optimizer.py argparse :1502-1717). Window size is searched by Optuna over distributed_config.json search_bounds (:72-102), read by load_search_bounds_from_config (window_optimizer.py:67): window_size 6-50, default 12; trial 0 can be pinned with --warm-start-window (:1610), operator-only trial-0 pin; STEP1_EXPLICIT_PIN_KEYS agents/watcher_agent.py:511; Ch12 :1409-1414 'origin, not presence, carries authority' | **DROPPED** | No flag with this meaning; the registry describes CLI flags (ParameterSpec.name 'matches CLI arg without dashes', parameter_registry.py:34-37). The spec's 50-2000 / default 512 also contradicts search_bounds 6-50 / 12. window_size is in FROZEN_PARAMETERS (chapter_13_acceptance.py:72). |
| 2 | `window_optimizer.py` | `offset` `--offset` (`:281`) | no such flag. offset is searched over distributed_config.json search_bounds (:72-102), read by load_search_bounds_from_config (window_optimizer.py:67): 0-100; trial-0 pin --warm-start-offset (:1612), operator-only trial-0 pin; STEP1_EXPLICIT_PIN_KEYS agents/watcher_agent.py:511; Ch12 :1409-1414 'origin, not presence, carries authority' | **DROPPED** | No flag with this meaning. The approved design renames offset to window_anchor (PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md §4.6 :156-161: min 0, no static max, delegated to the resolver), held for Brief II (S172_WINDOW_ANCHOR_BRIEF_I.md:360-361; CHAPTER_10_AUDIT S1). No --window-anchor flag exists, so it is NOT renamed here: registering window_anchor is Brief II's when it gives the anchor a CLI route. |
| 3 | `window_optimizer.py` | `skip_min` `--skip-min` (`:290`) | no such flag. skip_min is searched over distributed_config.json search_bounds (:72-102), read by load_search_bounds_from_config (window_optimizer.py:67): 0-10; trial-0 pin --warm-start-skip-min (:1614), operator-only trial-0 pin; STEP1_EXPLICIT_PIN_KEYS agents/watcher_agent.py:511; Ch12 :1409-1414 'origin, not presence, carries authority' | **DROPPED** | No flag with this meaning; the registry describes CLI flags (ParameterSpec.name 'matches CLI arg without dashes', parameter_registry.py:34-37). |
| 4 | `window_optimizer.py` | `skip_max` `--skip-max` (`:299`) | no such flag. skip_max is searched over distributed_config.json search_bounds (:72-102), read by load_search_bounds_from_config (window_optimizer.py:67): 10-250; trial-0 pin --warm-start-skip-max (:1616), operator-only trial-0 pin; STEP1_EXPLICIT_PIN_KEYS agents/watcher_agent.py:511; Ch12 :1409-1414 'origin, not presence, carries authority' | **DROPPED** | No flag with this meaning; the registry describes CLI flags (ParameterSpec.name 'matches CLI arg without dashes', parameter_registry.py:34-37). The spec's max 100 also contradicts search_bounds 10-250. |
| 5 | `scorer_meta_optimizer.py` | `n_trials` `--n-trials` (`:362`) | run_scorer_meta_optimizer.sh takes the trial count as its positional TRIALS (:184-192; scorer_meta.json arg_style positional, positional_args ['trials']) and passes it to generate_scorer_jobs.py --trials (:76) **Live job:** Step 2 = run_scorer_meta_optimizer.sh (STEP_SCRIPTS, watcher_agent.py:405-413), which runs generate_scorer_jobs.py (:229, :239; also the scorer_meta.json primary action script) and scripts_coordinator.py (:266) to run scorer_trial_worker.py. | **DROPPED** | The named script never existed (git log --all -- scorer_meta_optimizer.py is empty; CHAPTER_10_AUDIT B1). The live equivalent is in another script; registering it there is a new registration (generate_scorer_jobs.py --trials is among the unregistered options), not a correction of this entry. |
| 6 | `scorer_meta_optimizer.py` | `threshold_min` `--threshold-min` (`:372`) | no flag; the scorer's confidence threshold is a hard-coded Optuna range in generate_scorer_jobs.py:60 (min_confidence_threshold 0.05-0.25) **Live job:** Step 2 = run_scorer_meta_optimizer.sh (STEP_SCRIPTS, watcher_agent.py:405-413), which runs generate_scorer_jobs.py (:229, :239; also the scorer_meta.json primary action script) and scripts_coordinator.py (:266) to run scorer_trial_worker.py. | **DROPPED** | The named script never existed (git log --all -- scorer_meta_optimizer.py is empty; CHAPTER_10_AUDIT B1). No flag anywhere with this meaning. |
| 7 | `scorer_meta_optimizer.py` | `threshold_max` `--threshold-max` (`:381`) | no flag; same hard-coded range, generate_scorer_jobs.py:60 **Live job:** Step 2 = run_scorer_meta_optimizer.sh (STEP_SCRIPTS, watcher_agent.py:405-413), which runs generate_scorer_jobs.py (:229, :239; also the scorer_meta.json primary action script) and scripts_coordinator.py (:266) to run scorer_trial_worker.py. | **DROPPED** | The named script never existed (git log --all -- scorer_meta_optimizer.py is empty; CHAPTER_10_AUDIT B1). No flag anywhere with this meaning. |
| 8 | `scorer_meta_optimizer.py` | `k_folds` `--k-folds` (`:390`) | no fold option in generate_scorer_jobs.py or scorer_trial_worker.py (git grep 'fold': no hit) **Live job:** Step 2 = run_scorer_meta_optimizer.sh (STEP_SCRIPTS, watcher_agent.py:405-413), which runs generate_scorer_jobs.py (:229, :239; also the scorer_meta.json primary action script) and scripts_coordinator.py (:266) to run scorer_trial_worker.py. | **DROPPED** | The named script never existed (git log --all -- scorer_meta_optimizer.py is empty; CHAPTER_10_AUDIT B1). No equivalent. |
| 9 | `scorer_meta_optimizer.py` | `distributed` `--distributed` (`:398`) | no switch: Step 2 always dispatches through scripts_coordinator.py (run_scorer_meta_optimizer.sh:266) **Live job:** Step 2 = run_scorer_meta_optimizer.sh (STEP_SCRIPTS, watcher_agent.py:405-413), which runs generate_scorer_jobs.py (:229, :239; also the scorer_meta.json primary action script) and scripts_coordinator.py (:266) to run scorer_trial_worker.py. | **DROPPED** | The named script never existed (git log --all -- scorer_meta_optimizer.py is empty; CHAPTER_10_AUDIT B1). No equivalent; Step 2 is always distributed. |
| 10 | `full_scoring.py` | `batch_size` `--batch-size` (`:415`) | full_scoring_worker.py --batch-size (:536) exists but means 'Progress reporting batch size (default: 100)'; the nearest sizing option is --chunk-size (survivors per job) in generate_step3_scoring_jobs.py (:305) and generate_full_scoring_jobs.py (:28) **Live job:** Step 3 = run_step3_full_scoring.sh, which runs generate_step3_scoring_jobs.py (:160) and scripts_coordinator.py (:226) to run full_scoring_worker.py; the full_scoring.json primary action script is generate_full_scoring_jobs.py, which the wrapper does not run. | **DROPPED** | The named script never existed (git log --all -- full_scoring.py is empty; CHAPTER_10_AUDIT B1). The live --batch-size has a different meaning (progress reporting, default 100) from this spec (GPU batch 100-10000, default 1000), so it is not an equivalent. |
| 11 | `full_scoring.py` | `distributed` `--distributed` (`:424`) | no switch: Step 3 always dispatches through scripts_coordinator.py (run_step3_full_scoring.sh:226) **Live job:** Step 3 = run_step3_full_scoring.sh, which runs generate_step3_scoring_jobs.py (:160) and scripts_coordinator.py (:226) to run full_scoring_worker.py; the full_scoring.json primary action script is generate_full_scoring_jobs.py, which the wrapper does not run. | **DROPPED** | The named script never existed (git log --all -- full_scoring.py is empty; CHAPTER_10_AUDIT B1). No equivalent; Step 3 is always distributed. |
| 12 | `adaptive_meta_optimizer.py` | `trials` `--trials` (`:441`) | no such flag (adaptive_meta_optimizer.py argparse :1101-1114 has --config, --mode, --window-results, --lottery-data, --apply, --test only) | **DROPPED** | No flag with this meaning (also CHAPTER_5_AUDIT :272). The registry's script description 'Optimize ML model architecture using Optuna' is stale too; it is kept unchanged here (outside this ruling). |
| 13 | `adaptive_meta_optimizer.py` | `min_layers` `--min-layers` (`:451`) | no such flag (adaptive_meta_optimizer.py argparse :1101-1114 has --config, --mode, --window-results, --lottery-data, --apply, --test only) | **DROPPED** | No flag with this meaning (CHAPTER_5_AUDIT :272). |
| 14 | `adaptive_meta_optimizer.py` | `max_layers` `--max-layers` (`:460`) | no such flag (adaptive_meta_optimizer.py argparse :1101-1114 has --config, --mode, --window-results, --lottery-data, --apply, --test only) | **DROPPED** | No flag with this meaning (CHAPTER_5_AUDIT :272). |
| 15 | `meta_prediction_optimizer_anti_overfit.py` | `epochs` `--epochs` (`:506`) | no such flag (argparse :3041-3103); epochs is a trial parameter inside the NN path (params.get('epochs', 100), :912) | **DROPPED** | No flag with this meaning. |
| 16 | `meta_prediction_optimizer_anti_overfit.py` | `dropout_min` `--dropout-min` (`:516`) | no range flag; one single-value override --dropout (:3090, 'Override dropout value for NN'), and dropout is otherwise a trial parameter (:909) | **DROPPED** | No flag with this meaning: a range bound is not a single override value. --dropout is one of the six Step-5 LLM whitelist names under the CHAPTER_14_AUDIT S1 ruling (:248); registering it is not part of this repair. |
| 17 | `meta_prediction_optimizer_anti_overfit.py` | `dropout_max` `--dropout-max` (`:525`) | no range flag; see dropout_min (--dropout :3090, trial parameter :909) | **DROPPED** | No flag with this meaning (see row 16). |
| 18 | `meta_prediction_optimizer_anti_overfit.py` | `distributed` `--distributed` (`:534`) | no such flag (argparse :3041-3103) | **DROPPED** | No equivalent. |
| 19 | `prediction_generator.py` | `confidence_threshold` `--confidence-threshold` (`:560`) | no such flag (argparse :965-986); a config-file key: prediction_generator_config.json:9 confidence_threshold 0.7, read at prediction_generator.py:172 (passed with --config) | **DROPPED** | No flag with this meaning; the registry describes CLI flags (ParameterSpec.name 'matches CLI arg without dashes', parameter_registry.py:34-37). The value is set through the --config file. |
| 20 | `prediction_generator.py` | `diversity_weight` `--diversity-weight` (`:569`) | no such flag and no config key in any script (git grep: only agents/contexts/prediction_context.py:144-171, which reads it from results and SUGGESTS changing it) | **DROPPED** | No equivalent anywhere in a pipeline script. Note: prediction_context.py:170-171 still suggests a parameter no script accepts (left unchanged; outside this ruling). |

## Prior record

**Searches:**
- `git grep` at `41ad509` over `docs/` for `parameter[_ ]registry`, `window_anchor`, `scorer_meta_optimizer`, `full_scoring.py`, `--batch-size`, `diversity_weight` and `confidence_threshold`.
- `git log --all -- scorer_meta_optimizer.py full_scoring.py`, which is empty.
- `git grep` over `*.py` and `*.json` for each dropped name.

**Found:**
- **B1:** `CHAPTER_10_AUDIT.md` B1 (`:117-140`): the 20-of-29 count, and the fact that both scripts never existed. This is **ALREADY RECORDED**, and this repair carries it out.
- **S1:** `CHAPTER_10_AUDIT.md` S1 (`:145-155`): `offset` is not yet `window_anchor`.
- **The design:** `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` §4.6 (`:156-161`), held for Brief II (`S172_WINDOW_ANCHOR_BRIEF_I.md:360-361`).
- **Step 4 entries:** `CHAPTER_5_AUDIT.md:272`.
- **Chapter 14 S1 ruling** on the Step-5 LLM fields: `CHAPTER_14_AUDIT.md:248`.
- **The ruling:** `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt`.
- **Yesterday's check:** `docs/drafts/loop_v1/SWITCH_CHECK_20261123.md`.

**Labels:**

| Item | Label |
|---|---|
| The repair itself | carries out an **ALREADY RECORDED** finding (B1) under a new ruling (#7) |
| `offset` → `window_anchor` | **ALREADY RECORDED** (S1, design §4.6); not done here |
| `prediction_context.py:144-171` suggests `diversity_weight`, which no script accepts | **NEW** |
| The 9 kept specs disagree with the code on `mode` choices | **ALREADY RECORDED** (CHAPTER_5_AUDIT `:272`) |
| The 9 kept specs disagree with the code on the `pool_size` default and the `prng_type` choices | **NEW** |
| Item 3 count | **NEW** |
| Changes a ruling | None. The Chapter 14 S1 boundary is untouched: `--dropout` is not registered |

**Not searched:** the host, ser8, the private repository.

## The 9 kept specs (unchanged, as the task says), and where they disagree with the code

Nothing below is changed. These are facts for the owner.

| Spec (registry line) | Live flag | Disagreement |
|---|---|---|
| `window_optimizer.py` `trials` (`:308`) | `window_optimizer.py:1515` | none (default 50 both) |
| `window_optimizer.py` `max_seeds` (`:318`) | `:1521` | none (default 10,000,000 both) |
| `window_optimizer.py` `prng_type` (`:328`) | `:1533` | registry offers 6 choices including `lcg64`; the code takes any `prng_registry` type and has no `choices` (NEW) |
| `window_optimizer.py` `strategy` (`:336`) | `:1502` | same 4 choices; the code's help says only `bayesian` works; code default `None`, registry `bayesian` |
| `window_optimizer.py` `test_both_modes` (`:344`) | `:1573` | none |
| `adaptive_meta_optimizer.py` `mode` (`:469`) | `adaptive_meta_optimizer.py:1104` | registry choices `quick/full/deep`; code `full/survivor/network/epochs`, so an LLM could be told `quick` is legal (ALREADY RECORDED, CHAPTER_5_AUDIT `:272`) |
| `meta_prediction_optimizer_anti_overfit.py` `trials` (`:487`) | `:3045` | none (default 50) |
| `meta_prediction_optimizer_anti_overfit.py` `k_folds` (`:497`) | `:3047` | none (default 5) |
| `prediction_generator.py` `pool_size` (`:551`) | `prediction_generator.py:981` | registry default 200, code default `None` (the value comes from `--config`); and WATCHER sends the manifest's `pool_size` as `--k`, so `--pool-size` is unreachable (ALREADY RECORDED, CHAPTER_7_AUDIT `:105`) |

## The change

**Files**, in `docs/drafts/loop_v1/registry_repair_20261124/`, with `SHA256SUMS`:

| File | sha256 | Content |
|---|---|---|
| `parameter_registry.py.repair.patch` | `6d4870eb666c024da9836c7161bb7db725382709d17e4c00d470e9f3df0b9d5c` | Unified diff against `41ad509:agents/registry/parameter_registry.py` |
| `parameter_registry.py.repaired` | `72d0031569d42ed5c25d0cf5eaa7ef81a0e1827fe7c3728396d5c9e63956d341` | The repaired file |
| `repair_record.json` | `3469d957d6d52f187d05faa701d5b678354bcded6ca1a47f0d03687781d867c7` | Dropped, kept, registries removed, counts |

**How it was made:**
- The diff is generated, not typed. `scripts/diagnostics/switch_check.py` rule R9 (`repair_registry`; sha256 `dc1ce61ec18d471b8cf48d72ddda4a764e9a053d17637fe25d3837df9a6f70ca`) applies the declared decisions in `REGISTRY_REPAIR_DECISIONS_20261124.json` (sha256 `9cdffbb53d152971e1e04d8e7c0747bcc349cc87a8b70dcafe0b22c1cce02f6b`).
- It removes each spec by its exact `ast` span and writes a dated comment in its place.
- It removes a registry whose specs were all dropped.
- It refuses unless:
  - the decisions cover exactly the stale set;
  - every decision has a "why";
  - every other spec is unchanged;
  - every kept flag is defined by its script.
- Only DROPPED is implemented. A CORRECTED decision is refused, because no entry needed one.

**Consumers, checked:**
- `ParameterContext.build` (`agents/parameters/parameter_context.py:181`) already handles `get_registry` returning `None`.
- `get_registry_for_agent` (`:604-660`) still maps the Step 2 and Step 3 agent names to the removed script names. After the repair those lookups return `None`, as they would for any unregistered script. The mapping is **not** changed here, because no correct live key exists in the registry for those steps.
- `test_subphase1.py` (`:70-90`) uses only `window_optimizer.py` `trials` and `max-seeds`, both kept. It passes on a full `41ad509` copy with the repaired file. Logs: `logs/e2e_grader/rr_compat_test_subphase1_orig.log` and `_repaired.log`.

**Sequencing with other drafts (for the lead's merge):**
- **Yesterday's registration patch** (`switch_patches_20261123/`) was made against the unrepaired file. Land the repair first, then regenerate that patch with `switch_check.py --patch-dir`, which re-reads the registry.
- **The step2-repair agent** (TASK_STEP2_WIRING_20261124) will also draft a `ParameterSpec` for its setting into this same file. After this repair, Step 2's registry key `scorer_meta_optimizer.py` no longer exists. A Step 2 setting belongs under the script that defines its flag.

## Tests and mutants

| What | Result | Log |
|---|---|---|
| `tests/test_registry_repair.py` (sha256 `b8ee157b…5ba327`), on the real repaired file: imports under Pydantic in memory; 20 dropped with comments; 9 kept identical field by field; every kept flag in its script's argparse at `41ad509`; nothing outside the dict changed | **5/5** | `logs/e2e_grader/rr_test_run1.log` |
| `tests/test_switch_check.py` (sha256 `409405e7…84011`), fixture repo: adds 3 repair gates (repair, refusals, the flag-check defence) | **12/12** | `logs/e2e_grader/sc_selftest_run6.log` |
| `tests/mutate_e2e_grader.py` (sha256 `712b681b…ba327`): RR1–RR6 on the repaired file, SM17–SM23 on the repair code, plus the existing grader, `compare_settings` and switch-check mutants | **87/87 killed and credited**; all baselines pass; 126 s on 2 workers | `logs/e2e_grader/all_mutants_run6.log` |
| Grader and `compare_settings` self-tests, re-run | 36/36, 9/9 | `logs/e2e_grader/loop_selftest_full_run4.log` |
| `test_subphase1.py` (main tree's), original vs repaired, on isolated `git archive 41ad509` copies | both rc 0, ALL TESTS PASSED; only difference "Registry has 5 parameters" (was 9) | `logs/e2e_grader/rr_compat_test_subphase1_*.log` |

**Two defects fixed during the work, both in my checker:**
- `parse_param_registry` crashed on a registry `description` that is not a literal (found by mutant RR6). It now records the expression, as it already did for spec fields.
- The class gate now reports a wrong stale set under its own tag, not as a crash (found by SM4).

## Item 3: which unregistered step-script options change results

**Rule:** an option changes results if, with the same input files, it changes what is computed: survivors, scores, labels, the trained model, the pool or the grade.

**Not counted:**

| Class | Rows |
|---|---|
| Paths, inputs and output names (P) | 54 |
| Logging, test, dry-run and report-only options (L) | 14 |
| ids, provenance and metadata-only (I) | 9 |
| Infrastructure that is result-neutral by design, such as transport, workers, chunk sizes and devices (F) | 32 |
| Guards that abort or refuse rather than change a value (G) | 4 |
| Options of files the step does not run (N) | 15 |

The N class comes from my checker's own step rule: `reinforcement_engine.py` is only copied to the nodes by Step 2 (`run_scorer_meta_optimizer.sh:259`), and `generate_full_scoring_jobs.py` is not what the Step 3 wrapper runs (`:160`).

**Verified in code, not taken from help text:**
- `--prng-type`, `--mod` and `--holdout-draws` in Step 5 only feed the `data_context` fingerprint (`mpo:1498-1505`).
- `--apply` writes `reinforcement_engine_config.json` (`adaptive_meta_optimizer.py:1171`).
- `--legacy-scoring` switches the scoring implementation (`scorer_trial_worker.py:557`).

**Infrastructure (F) is "result-neutral by design; not verified here".** The RANGE-MINER backend has its own bounded byte-identity certificate (fact file §8). Each option's class and basis is in `REGISTRY_REPAIR_OPTION_CLASSES_20261124.json` (sha256 `ac628851…a7d7`). It was produced by `data/switch_check_trial/classify_options.py`, which refuses if any option is unclassified.

**Of the 174: 46 rows, 41 distinct options, change results.**

**Step 0 (TRSE): 2 rows, 2 distinct options**

- `trse_step0.py:813` `--k-clusters` (main): number of regime clusters; the context narrows Step 1 bounds
- `trse_step0.py:814` `--recommended-window-size` (main): window recommended in the context read by Step 1

**Step 1 (window optimizer): 15 rows, 15 distinct options**

- `window_optimizer.py:1507` `--config-file` (main): switches the procedure: runs the sieve on an existing config and skips the Optuna search
- `window_optimizer.py:1523` `--iterations` (main): number of sieve iterations in config mode
- `window_optimizer.py:1549` `--resume-study` (main): continues an existing Optuna study, so the trial history and the chosen window differ
- `window_optimizer.py:1576` `--enable-pruning` (main): prunes forward_count==0 trials; changes the study's trial set (help says speed)
- `window_optimizer.py:1606` `--seed-start` (main): which seeds are searched
- `window_optimizer.py:1610` `--warm-start-window` (main): enqueues this value as Optuna trial 0 (operator-only pin, STEP1_EXPLICIT_PIN_KEYS)
- `window_optimizer.py:1612` `--warm-start-offset` (main): enqueues this value as Optuna trial 0 (operator-only pin, STEP1_EXPLICIT_PIN_KEYS)
- `window_optimizer.py:1614` `--warm-start-skip-min` (main): enqueues this value as Optuna trial 0 (operator-only pin, STEP1_EXPLICIT_PIN_KEYS)
- `window_optimizer.py:1616` `--warm-start-skip-max` (main): enqueues this value as Optuna trial 0 (operator-only pin, STEP1_EXPLICIT_PIN_KEYS)
- `window_optimizer.py:1618` `--warm-start-fwd-thresh` (main): enqueues this value as Optuna trial 0 (operator-only pin, STEP1_EXPLICIT_PIN_KEYS)
- `window_optimizer.py:1620` `--warm-start-rev-thresh` (main): enqueues this value as Optuna trial 0 (operator-only pin, STEP1_EXPLICIT_PIN_KEYS)
- `window_optimizer.py:1622` `--warm-start-session-idx` (main): enqueues this value as Optuna trial 0 (operator-only pin, STEP1_EXPLICIT_PIN_KEYS)
- `window_optimizer.py:1650` `--deliver-skip-range` (main): RANGE-MINER sieves the trial's own skip range instead of the baseline
- `window_optimizer.py:1735` `--derived-anchor-domain` (anchor-brief2): Optuna samples the anchor over a data-derived domain
- `window_optimizer.py:1743` `--anchor-holdout-guard` (anchor-brief2): caps the anchor so the sieve window ends before the holdout

**Step 2 (scorer meta): 6 rows, 6 distinct options**

- `generate_scorer_jobs.py:76` `--trials` (main): number of Optuna trials for the scorer search
- `generate_scorer_jobs.py:83` `--legacy-scoring` (main): legacy batch_score() instead of vectorized (scorer_trial_worker.py:557); equivalence not verified
- `generate_scorer_jobs.py:85` `--sample-size` (main): number of seeds sampled for training
- `run_scorer_meta_optimizer.sh:42` `--min-collected` (step2-repair): lets a study complete on fewer returned trials (only when trials are lost)
- `run_scorer_meta_optimizer.sh:175` `--sample-size` (main): forwarded to generate_scorer_jobs.py
- `run_scorer_meta_optimizer.sh:179` `--legacy-scoring` (main): forwarded to generate_scorer_jobs.py

**Step 3 (full scoring): 8 rows, 6 distinct options**

- `full_scoring_worker.py:530` `--prng-type` (main): PRNG used to regenerate and score each seed
- `full_scoring_worker.py:532` `--mod` (main): modulus used in scoring
- `full_scoring_worker.py:570` `--per-seed-sieve-features` (ch14-learn, e2e-map): adds two per-seed features (schema v2)
- `full_scoring_worker.py:573` `--fallback-metadata-merge` (ch14-learn, e2e-map): which metadata the sequential fallback merges
- `generate_step3_scoring_jobs.py:309` `--prng-type` (main): PRNG if not in config
- `generate_step3_scoring_jobs.py:311` `--mod` (main): modulus if not in config

**Step 4 (ML meta): 1 rows, 1 distinct options**

- `adaptive_meta_optimizer.py:1112` `--apply` (main): writes the result into reinforcement_engine_config.json (:1171), which later steps read

**Step 5 (anti-overfit training): 12 rows, 9 distinct options**

- `meta_prediction_optimizer_anti_overfit.py:3049` `--test-holdout` (main): share held out as the test set
- `meta_prediction_optimizer_anti_overfit.py:3057` `--model-type` (main): model family trained
- `meta_prediction_optimizer_anti_overfit.py:3060` `--compare-models` (main): trains all families and keeps the best
- `meta_prediction_optimizer_anti_overfit.py:3086` `--normalize-features` (main): StandardScaler before NN
- `meta_prediction_optimizer_anti_overfit.py:3088` `--use-leaky-relu` (main): activation
- `meta_prediction_optimizer_anti_overfit.py:3090` `--dropout` (main): dropout override
- `meta_prediction_optimizer_anti_overfit.py:3128` `--exclude-features` (ch14-learn, e2e-map): training columns
- `meta_prediction_optimizer_anti_overfit.py:3131` `--include-feature-groups` (ch14-learn, e2e-map): training columns
- `meta_prediction_optimizer_anti_overfit.py:3134` `--exclude-feature-groups` (ch14-learn, e2e-map): training columns

**Step 6 (prediction): 2 rows, 2 distinct options**

- `prediction_generator.py:978` `--no-require-schema` (main): allows prediction without the feature schema (zero-filled features)
- `prediction_generator.py:980` `--k` (main): number of top predictions (the pool)

**Added since the 174 (today's worktrees): 4 more rows that change results**

- Step 1: `window_optimizer.py:1748` `--step1-objective` (anchor-brief2): what Optuna maximises (default bidirectional_count = today)
- Step 3: `generate_step3_scoring_jobs.py:338` `--step2-config-mode` (step2-repair): pin applies Step 2's tuned settings (default off = today)
- Step 3: `run_step3_full_scoring.sh:127` `--step2-config-mode` (step2-repair): forwarded Step 2 config mode
- Step 5: `meta_prediction_optimizer_anti_overfit.py:3119` `--step2-config-mode` (step2-repair): pin applies Step 2's tuned settings (default off = today)

**Already reachable through WATCHER:** of these 46 rows, **10** are passed by WATCHER today, because they are declared in their step manifest's `default_params`: Step 0 `trse_step0.py` `--k-clusters`, Step 0 `trse_step0.py` `--recommended-window-size`, Step 1 `window_optimizer.py` `--deliver-skip-range`, Step 1 `window_optimizer.py` `--enable-pruning`, Step 1 `window_optimizer.py` `--resume-study`, Step 1 `window_optimizer.py` `--seed-start`, Step 4 `adaptive_meta_optimizer.py` `--apply`, Step 5 `meta_prediction_optimizer_anti_overfit.py` `--compare-models`, Step 5 `meta_prediction_optimizer_anti_overfit.py` `--model-type`, Step 6 `prediction_generator.py` `--k`. The other 36 are not passed. That includes the seven warm-start pins, which only the operator pin can set. Registering a result-changing option does not by itself make it reachable (`AUTONOMY_SWITCH_MAP_20261123.md` G6, G7).

## Not verified, and housekeeping

- **Infrastructure options (F)** are not proven result-neutral here.
- **`--legacy-scoring`:** its equivalence to the vectorized scorer is not tested.
- **`--min-collected`** changes results only when trials are lost.
- **The repaired registry** is checked by import, field equality, flag existence and the main tree's `test_subphase1.py`. It is not exercised inside a running WATCHER.
- **Bytecode cache in `scripts/diagnostics/`:** `scripts/diagnostics/__pycache__/e2e_grader.cpython-310.pyc` was written on 2026-11-23 at 20:44:39, after my last job of that day ended (19:51). So it was **not written by me**; something else imported my grader from this worktree. I left it in place; the lead decides.
- **How every run here went:**
  - with `PYTHONDONTWRITEBYTECODE=1` and `python3 -B`;
  - under `taskset -c 6,7 nice -n 15`, at most 2 processes;
  - the compatibility copies were extracted from `git archive 41ad509` into my session scratchpad, outside every worktree; that copy wrote no `__pycache__`.
