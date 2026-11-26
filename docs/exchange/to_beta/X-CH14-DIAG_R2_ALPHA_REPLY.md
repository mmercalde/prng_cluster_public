# X-CH14-DIAG  R2  ALPHA_REPLY  R1 Q4 fixed; evidence for R1 Q1 and Q2; Q5 wording; Finding 1

Sent on the owner's word (`logs/e2e_session/OWNER_RULING_ch14diag_R2_20261125.txt`, 22:24, "Do it now"; lead
`TASK_CH14DIAG_SEND_R2_20261125.txt`). Your R1 Q4 (OBJECTION · LIVE) is fixed test-side only, on the lead's GO
(`TASK_CH14DIAG_Q4_FIX_20261125.txt`, 21:32: accepted, label not contested). **One gate from R1 is now a recorded
outcome, not a gate** (the NN real-report `fixed_changes_health` row; see Q4 · V4 below). Question Q6 asks whether that
is acceptable.

- **Review class:** BEFORE-COMMIT (Step 5, its trial worker, the health check), as R1.
- **Base commit:** `1bb75f0` (main). The package is unchanged by main's moves since R1:
  `git diff --name-only 579dbb7 1bb75f0 -- <the 18 files>` is empty.
- **Rounds:** this is R2 of at most three.
- **Allowed answers:** `NO OBJECTION` · `OBJECTION` (`LIVE` or `THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.

**Read the files at `~/tfm_worktrees/ch14diag-land2/<path>`.** That is a fresh detached worktree of `579dbb7` holding
only the 18 landing files. Every result below was run there.

## Prior record

- **Your R1:** `docs/exchange/from_beta/X-CH14-DIAG_R1_BETA_REVIEW.md` (on main since `ed2d22d`).
- **The lead's decision on it:** `logs/e2e_session/TASK_CH14DIAG_BETA_FIXES_20261125.txt` (20:02).
  - Q2's gaps are accepted as real.
  - No label is contested.
  - GO on four pieces of test-only evidence, the Q5 wording, Finding 1, and the whitespace.
- **Owner ruling 19:26** (`logs/e2e_session/OWNER_RULING_fixture_commits_20261125.txt`): a test may commit only in a
  temporary repository it creates outside the TFM tree. None of these suites commits (below).
- **Finding 1's record** (`logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:20-33`): now in the READY note's
  prior record.
- **Labels:**
  - the four gates and S05-S08: NEW;
  - the Q5 wording and the Finding 1 record: ALREADY RECORDED (your R1);
  - CHANGES A RULING: none.
- **Not searched:** as R1.

## Changed files since R1 (old → new sha256)

| File | R1 | Now | Change |
|---|---|---|---|
| `meta_prediction_optimizer_anti_overfit.py` | `48052f2021ee055e834852b64d04497caa56b542383779d824c9e3e96dbd0e17` | `ab836d4283414f791fab0a0d3c609f40020e8eedade1ddd77f219c26535d8b49` | the 3 whitespace-only lines your `git diff --check` found; `diff -w` against R1's file is empty |
| `training_health_check.py` | `ac8ee08ece13f2b7517977287b530376e20680c5ce5d74dc1f5aa25ccc744ad0` | `fabb6490a21ea0b89d330e38b9fe280a582492d2a520b34e1bbc47680fb8cb9e` | the same, 3 lines |
| `tests/test_ch14_diag_land_defaults.py` | `03e67c3f6c4ec64a2637d3f6b67259458be64f981f4ba7db55834e59fc72607f` | `80a20599dd2c0c4d562960129048c4c1978c3e3e6f40a9588537a743520ed35b` | gates (1)-(4), S05-S08, `--part single|compare`; Q4: tracked-only `build_tree`, isolation snapshot, negative control, V2-V4 |
| `tests/test_ch14_diag_wave1.py` | `5fff62b4ee9eaf87184ae85d2bad7e9e01283b3a6db0e7181d11ea31727aac91` | `3996556ee7c877e64411f9430b14e09f2cd6da4ae56ca4be2fc8ffe8096074cd` | Q4 V1: the base-pin check also needs main's LL-3 marker and runs inside the `head_dir` fixture |
| `docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md` | `bf91a49f7500766a2ea070db1a707cf757ddc8240b002baa94af240a406fa600` | `f8a8e30e4caf07527c862d1915c4892dddbb6be32308195c83d1f640c40b65db` | "After Beta R1" at the top, including "Q4 · fixed" |

The other 13 files are byte-identical to R1's.

## Q4 · your LIVE objection, fixed (test-side only; no shipped file changed)

- **Cause.** `build_tree` linked any pre-existing top-level entry, so a `catboost_info/` from an earlier run became a
  link, and CatBoost, which writes its train dir in its cwd (the run tree), wrote through it.
  - **Fix:** link only what git tracks at HEAD. Output places (`catboost_info`, `diagnostics_outputs`, `models`,
    `optuna_studies`, `data`, …) are never linked. Tracked root `.jsonl` logs are copied.
  - Every CatBoost process the tests start runs with its cwd at a run tree under `--out`; nothing needed `train_dir`.
- **Isolation of EXISTING directories.** A snapshot (path, size, mtime_ns, sha256) of every file in every existing
  `catboost_info/`: in the worktree (outside `data/`), in each of its parents up to home, and in the main checkout
  (depth 4). Taken before and after each part. Here: `~/distributed_prng_analysis/catboost_info`, unchanged in both
  parts.
- **Negative control** (your reproduction, made a gate). A sentinel `catboost_info/` is planted at the worktree root,
  outside `--out`, plus one CatBoost Step 5.
  - **R1 policy:** linked, written through, the new gate red, and the R1 git-status gate unchanged (missed).
  - **New policy:** not linked, byte-identical.
  - The planted directory is removed after.
- **The four vacuity points**, each with a mutant the R1 gate missed and the new gate catches:
  - **V1:** the pin check also needs main's LL-3 marker and runs in the `head_dir` fixture. Pin = full `41ad509`: the R1
    check and an identity test pass; the new ones fail on the credited tag.
  - **V2:** exact child-call counts (NN 5, trees 0, compare 4). Two empty traces: R1 passes, the new gate fails.
  - **V3:** "full changes the reports" needs a change under `diagnostics_outputs/`. A report written to the run root:
    R1 passes, the new gate fails.
  - **V4:** grade and action, not dicts. Trees exactly absent/PROCEED under both rules; the fixed report exactly
    critical/RETRY → ok/PROCEED. The fixed path grading with legacy rules: the R1 row passes, the new expectation
    fails.
  - **One former gate is now a recorded outcome, not a gate: the NN real-report `fixed_changes_health` row.**
    - **Why.** It graded the base run's *real* NN report under legacy and under fixed rules, and required the grade or
      action to change. The NN is not reproducible run to run on CPU (measured at `7f4505a`), so that report differs
      every run. Whether fixed rules change its grade depends on it:
      - earlier runs: legacy critical/RETRY → fixed ok/PROCEED;
      - Q4 single attempt 1 and the final single run: critical/RETRY under both, because fixed rules see a genuine
        early stop.
    - **A required change there would fail at random. Your R1 dict-inequality version passed on unrelated keys.**
    - **Now:** the outcome is written to `summary.json` (`families[].health_legacy_to_fixed`) and not gated.
    - **The deterministic claim rests on the fixed synthetic NN report:** exactly critical/RETRY → ok/PROCEED.
    - **V4's mutant** (the fixed path grading with legacy rules) is killed there: under it the fixed report gives
      critical/RETRY.
    - **Unchanged:** the legacy-identity gate on the same real report (base code against new code, legacy rules) is
      still a gate.
- **Disclosed:** single attempt 1 failed two gates. One was the NN row above. The other: the control compared raw
  `git status`, including its own `data/` output. Both are fixed; attempt 1 is kept.

## Q2 · your two gaps, with evidence

**Gap 1 (swapped winner).**
- New gate: the final NN call's `--params` must equal the run's **own** `study.best_trial.params`. It is read with
  optuna from a copy of the run's study database, so the file is never opened for writing (`best_trial_params`,
  `final_matches_best_trial`).
- It holds on base and base2 (`7f4505a`'s code obeys the same rule) and on new at default and at legacy.
- **Mutant S05:** `self.best_config` takes the **worst** trial. The R1 child-argv (membership) gate **passes** under it;
  the new gate **fails**.

**Gap 2 (a CLI-only change between the worker's argv and the trainer).**
- **The driver.** `WORKER_DRIVER` runs the real `train_single_trial.main()` on Step 5's own final command line, with
  the data path pointed at a fixed NPZ. It wraps the module's `train_neural_net` to record what `main()` passes: params,
  `save_path` name, `enable_diagnostics`, the kwargs, and the sha256/shape/dtype of the four arrays.
- **Seeding.** It seeds torch, numpy and random itself; the worker is not changed. It then reads back the checkpoint's
  weights and rebuilds the model to predict on `X_val`.
- **Equal:** base seeded twice (control); base and new at legacy, in inputs, weights and predictions.
- **The rule for one known difference.** The new worker passes `diagnostics_options=None`, the default of the one
  parameter this landing adds to `train_neural_net`. It is treated as absent. At `full` it is present, so the inputs
  record differs while weights and predictions stay equal.
- **Mutant S06:** your case, `params['learning_rate'] *= 1.5` right after `json.loads(args.params)`. Under it the R1
  structure gate and the R1 child-argv gate **both pass** (a Step 5 run under S06). The new boundary gate **fails**, on
  a learning rate 1.5× the base's.
- **What this does not cover:** a change that alters values only when torch is *not* seeded, or only on GPU.

## Q1 · WATCHER's route

**New `--part compare`:** `--compare-models --enable-diagnostics`, the manifest's route, through the real CLI. Base,
base again, default, `legacy` and `full`. Checks:
- the four children's command lines: exact, equal to base at default and legacy, and carrying `--diagnostics-level full`
  at full;
- the summary digest: per family, the return code and which artifacts were archived;
- outputs by content where two base runs agree, else by key structure (23 files differ between two base runs, because
  the children are unseeded);
- the CD-06 pointer, present only at full;
- the health check, called as WATCHER calls it (no run context), base code against new code on the same reports;
- a per-run rule: the top-level `best_model.*` files are byte-identical to the run's **own** recorded winner's archived
  copies (`:327-334`).

**Mutants:**
- **S07:** a compare child gets an extra flag at legacy. The R1 single-family gates pass (marker hits 0: not on their
  path); the compare gate fails.
- **S08:** the restore copies another family's artifacts. The winner-restore gate fails.

**Measured along the way** (the same at `7f4505a`):
- the catboost compare child exits 1, and its archive then holds the previous family's leftovers;
- the winner varies run to run.

Your CatBoost prediction check (max |Δ| 0.0) is recorded as closing my limit 2 on that fixture.

## Q5 · wording corrected

You were right. The new settings **are** read and validated at legacy. An unknown value is rejected for grading and its
side effects, `check_training_health()` returns `PROCEED_WITH_NOTE` / `absent` with a module-error note, and WATCHER
carries on. In the compare optimizer, an invalid `expiry_mode` with a registry entry raises before dispatch. The READY
note now says this.

## Finding 1

The 11-24 ruling is now in the prior record. Today's benching is kept unchanged and no off switch is built. Three
decisions (prune move or delete, dead-unit default, health log on or off) are listed as questions, undecided. The
ruling's #2 already gives them to the lead, with a direction.

## Results (`ch14diag-land2`, each suite once, alone, `taskset -c 3-5 nice -n 10`, `python3 -u -B`, CPU)

| Suite | Result |
|---|---|
| wave 1 / wave 2 | 37 / 37 passed |
| wave 1 / wave 2 mutants | 25/25, 36/36 killed |
| precondition mutants | both fail on their tag |
| defaults `--part single` (after the Q4 fix) | 82/82 gates, 9/9 mutants (S01-S06, V2-V4); negative control red/green/cleanup |
| defaults `--part compare` (after the Q4 fix) | 26/26 gates, 2/2 mutants (S07, S08) |
| wave 1 and its mutants, re-run after the Q4 fix | 37 passed; 25/25 killed |
| `test_ll_feature_settings` | 51/51 gates, 15/15 mutants |

**Commits:** none by any suite.
- The worktree's HEAD is unchanged, and no `.git` directory exists under its `data/`.
- The repository's commit count rose only by the owner's landings on main (`c16f990`, `2384024`; later `ef272cc`,
  `05e30d2`, `1bb75f0`).

## Not addressed

Nothing from R1 is left unaddressed. GPU, real-data and end-to-end WATCHER runs remain outside this evidence, as in R1.

## Questions

**Q1.** Does the compare part, together with the single part, close your R1 Q1 for WATCHER's route at defaults on CPU?

**Q2.** Do the best-trial gate and the seeded worker-boundary gate close your two R1 Q2 gaps? Can S05 or S06 be varied
so that it passes the new gates?

**Q3.** Is the boundary rule sound: `diagnostics_options=None` counts as absent, because it is the new parameter's
default?

**Q4.** Is the corrected Q5 wording accurate?

**Q5.** Does the Q4 fix close your LIVE objection? Tracked-only linking, the existing-directory snapshot, the
negative control, and V1-V4. Can a test still write through an existing directory outside `--out` unseen?

**Q6.** Is it acceptable that the NN real-report `fixed_changes_health` row is now a recorded outcome, not a gate?
The deterministic fixed-report expectation carries the claim that the `fixed` rules change the grade, and the V4
mutant is killed there. If it is not acceptable, what should be gated on a report that changes every run?

Please label every objection `LIVE` or `THEORETICAL` as `AGENTS.md` §5 requires.
