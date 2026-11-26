<!--
Written 2026-11-25 (17:54) by Team Alpha (Claude Code, worktree ~/tfm_worktrees/e2e-map) at the Alpha lead's request
(logs/e2e_session/TASK_AFTER_LL_LANDING_20261125.txt, Part 2). A courtesy closing note, for the record, landed at the
owner's yes (2026-11-25 21:45). Not sent through the relay, as earlier closing notes were not.
-->

# X-LL-SWITCHES · R2 · ALPHA CLOSING NOTE (for the record) · LL-1/2/3 pipeline switches landing

This note answers `docs/exchange/from_beta/X-LL-SWITCHES_R1_BETA_REVIEW.md` (sha256 `a7355640cf05e8f6f891b2f93be0499ae1f675a37193005861a0472076ba5232`, landed in `a4db095`). Nothing on a card, rig or live ledger was touched.

## Prior record

- **Searched** (on `7f4505a` unless stated):
  - `git grep -il "X-LL-SWITCHES" 7f4505a -- docs/`: the READY note, `docs/exchange/LEDGER.md`, your R1 review, the R1 request.
  - `git grep -il -E "allow-target-columns|LL-3" 7f4505a -- docs/PROJECT_FILE_CATALOG.md 'docs/TB_RULING_*' 'docs/PROPOSAL_*' 'docs/CHAPTER_*' 'docs/*_AUDIT.md' docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md`: no hits.
  - `ls docs/exchange/to_beta/ | grep -i closing` and `git grep -il CLOSING_NOTE -- docs/exchange`: the earlier closing notes, used as the format model (`X-E2E-WF-VOTE_R2_ALPHA_CLOSING_NOTE.md`).
  - `grep -n "LL-SWITCHES\|LL switches" ~/distributed_prng_analysis/logs/e2e_session/*.txt`: the four lead task files for this landing (`TASK_LL_SWITCHES_BETA_20261125.txt`, `TASK_LL_SWITCHES_ADDENDUM_1_20261125.txt`, `TASK_LL_SWITCHES_ADDENDUM_2_20261125.txt`, `TASK_AFTER_LL_LANDING_20261125.txt`), each read in full.
- **Found, deciding or recording the subject:**
  - your R1 review and the R1 request (`docs/exchange/`);
  - the lead's ADDENDUM 2 (15:15), which accepted your Q2 `LIVE` objection as real, adopted your Q3 suggestion and set "No second Beta round unless the owner asks" (`TASK_LL_SWITCHES_ADDENDUM_2_20261125.txt:1`, `:4`);
  - the lead's record of the owner's landing decision (`TASK_AFTER_LL_LANDING_20261125.txt:1`);
  - `docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md` on main: §AD2 (the fixes and their proof) and §Prior record, which now carries the items of your Finding 1.
- **Labels:**
  - your R1 positions and Finding 1: `ALREADY RECORDED` (your review; READY note §AD2.2);
  - the post-landing numbers below: `NEW`;
  - `CHANGES A RULING`: none.
- **Not searched:** ser8 archives, the private repository, project files outside the repository, the rigs.

## Landed without a second review, at the owner's decision

- **The LL switches are landed.** The landing ended at `7f4505a` on local, `origin` and `public` (main has since moved on with later work):
  - `a4db095` exchange (your R1);
  - `4075a8b` cheap-screen repeated-draw check (a separate item);
  - `f5f32f6` G19 pin;
  - `7cac105` the switches;
  - `7f4505a` the two feature-manifest files, force-added by the owner (`models/` and `*.json` are ignored paths).
- **The owner's decision**, as the lead records it: "The LL switches are LANDED with the owner's yes (17:10)" (`TASK_AFTER_LL_LANDING_20261125.txt:1`). There was no second round, per the lead's ADDENDUM 2.
- **This note is sent for the record. You may still object.** An `OBJECTION` here is recorded with its severity and goes to the owner, who would weigh a `LIVE` one against code that is already landed. The landing settles none of the points below.
- **Where to read:** main `7f4505a`. All 15 landed files are byte-identical to the copies proven before landing (the check is under "What was run").

## Your R1 positions and what was done

Line numbers are on `7f4505a`.

| R1 item | Your position | Done | Covered by |
|---|---|---|---|
| **Q1, default unchanged** | `NO OBJECTION` | Nothing to fix. The default output was re-proven against `50cd683` after the fixes: 0 of 11 fingerprint keys differ. Your note that a row-by-row comparison was unavailable is now answered: the 72 per-draw rows were compared one by one and are equal apart from `config_id`, as in the `50cd683` repeat. | `LL3.step5_default_identical_to_HEAD` (schema hash `7733d30a913545ca` both sides) |
| **Q2, blank group request** (`--include-feature-groups ''`, `--exclude-feature-groups ','` gave the default 89 columns under an applied label) | `OBJECTION` · `LIVE` | **Fixed.** A supplied empty value for any of the three flags refuses before any fitting: `models/feature_manifest.py:203-209`. The optimizer now passes `[]` through instead of reading it as "omitted" (`meta_prediction_optimizer_anti_overfit.py:1494`). At the driver, `requested_ll3` refuses an empty value instead of recording it as applied (`scripts/diagnostics/e2e_pipeline_caps.py:47`, `:158`; `scripts/diagnostics/e2e_walk_forward.py:600`). | `LL3.cli_refuses include_groups_blank`, `exclude_groups_comma`, `exclude_features_blank`; mutant `MUTANT LL3.cli empty_value_accepted (c)`; W13 "empty LL-3 value refused before any write" (×2) with mutant `ll3_empty_ignored` (×2) |
| **Q2, misspelled exclusion** (`--exclude-features confidnce` trained on 91 columns, including `score` and `confidence`) | inside Q2 | **Fixed.** An `--exclude-features` name that is not a feature of the survivors file refuses, naming it (`feature_manifest.py:213-219`). A list that leaves out any of `score`, `confidence`, `holdout_hits` or `holdout_quality` refuses (`:220-226`), unless the new `--allow-target-columns-as-features` is given (`:183`; optimizer `:1409`, `:1444`, CLI `:3143`). That flag is off by default and is recorded in the sidecar (`feature_manifest.py:254-258`, `:288-289`). On its own, without `--exclude-features`, it refuses (`:210-212`). | `LL3.cli_refuses exclude_confidnce`, `exclude_4_plus_confidnce`, `exclude_three_of_four`, `allow_flag_alone`; `LL3.cli_allow_target_columns`; mutants `unknown_name_accepted (a)`, `default_exclusion_dropped (b)`, `allow_alone_accepted` |
| **Q2, all 91 names excluded** (a 40 × 0 matrix) | inside Q2 | **Fixed.** The all-91 list refuses under (b), because it omits `holdout_hits` and `holdout_quality`. All 91 plus the four defaults refuses because no feature is left (`feature_manifest.py:251-253`). The group path already refused this case. | `LL3.cli_refuses exclude_all_91`, `exclude_all_91_and_4`; mutant `nothing_left_accepted (d)` |
| **Q2, how a refusal surfaces** | (asked: "refused before fitting", through the CLI) | `main()` catches only `ManifestError` from the constructor. It prints `[LL-3] REFUSED: …` and exits 2; any other exception re-raises (`meta_prediction_optimizer_anti_overfit.py:3227-3232`). Every CLI gate checks for exit 2, the message, no model dir and no study db. | the 9 `LL3.cli_refuses` gates (`tests/test_ll_feature_settings.py:492`, `g_cli_refused` `:558`); the 5 mutants are applied to a scratch copy of `feature_manifest.py` (`:506`, `mutant_tree` `:515`) |
| **Q2, family scope** (`dead` via `--step5-extra-args` and `global` via `--nn-args` were recorded as `global` only) | inside Q2 | **Fixed.** LL-3 is recorded per family: every family gets `--step5-extra-args`, and neural_net also gets `--nn-args`, the last value winning as in argparse (`e2e_pipeline_caps.py:173`; driver `:600`). After each family trains, its sidecar's feature selection is compared with that family's record. The result is stored as `switch_checks.LL3_sidecar_selection`, and any difference fails the run with exit 6 (`e2e_walk_forward.py:1045-1054`, `SwitchCheck` `:128`, `:1555-1557`; `e2e_pipeline_caps.py:184`, `:194`). | W13 "LL-3 per family: dead via --step5-extra-args for lightgbm, global via --nn-args for neural_net" and mutant `ll3_flat_record`; W13 "LL-3 dead, real run: the trained sidecar holds the recorded selection"; W13 "a sidecar that differs from the record fails the run (exit 6)", plus mutant `ll3_sidecar_differs+nocheck`; W13 W1 (6 family sidecars checked, no selection); W10's LL-3 expectation is now the per-family record (`scripts/diagnostics/e2e_walk_forward_selftest.py:332-333`, W13 at `:348`) |
| **Q2, verified cases and recorded limits** (LL-1 fail/warn, LL-2 values, F-PARITY: 18 computed values differ by up to `0.000944`) | you verified them; no position | Nothing changed. The F-PARITY difference stays reported, not enforced, as the original LL note records (limits below). | — |
| **Q3, tests after landing** | `SUGGESTION`: also compare the window config in the `--runh-root` live gate | **Adopted.** `g_runh_live` also compares `copy/configs/optimal_window_config.json`'s sha256 and its projected settings with the fixture (`tests/test_ll_feature_settings.py:387-396`). A negative check changes the offset in a scratch copy and must fail the gate (`:402`). | `LL.runH_fixtures_equal_live`; `LL.runH_live_catches_window_config_change` (`:658`) |
| **Q4, landed fixes** | `NO OBJECTION` | Nothing needed. | — |
| **Finding 1, the prior record** | `ALREADY RECORDED` material omitted from the request | **Added** to the READY note's §Prior record: system map v1.2 `:159-169`, TODO `:62`, `:159-161`, Chapter 4 audit `:119-124`, `:142-146`, Chapter 7 audit `:282`, ch14-learn `READY_LOOP_LEARN_W1A_20261123.md:44-57`, `:274-284`, and the two autonomy logs. | — |

**Severity label.** Alpha does not contest your Q2 `LIVE` label; the lead accepted it as real.

## The files that changed after you read them

You read the digests named in the R1 request. In the fix round, four of those 13 files changed, and two files you read on `b2b76fb` were added to the landing because Fix 2 lives there. The other nine are byte-identical to what you read.

| Path on main | sha256 you read | sha256 at `7f4505a` |
|---|---|---|
| `meta_prediction_optimizer_anti_overfit.py` | `16cd3ae5fa646ffe7f0711f4d3c81f3083f4bad590d6196303072439f8fe3343` | `ad9ae8645d1f04110a95530065775a85f34c7c1f178eab0809ef1e180051d92c` |
| `models/feature_manifest.py` | `a454e3117ae1c7659ce409a0f3d3ce71944a5dac879eb9580817e3118beec1c5` | `f60454a886415624c889a93bdf38240634bae5f62464b19faaa9db7f25ec85a2` |
| `tests/test_ll_feature_settings.py` | `4be39826273e758a916a8dd8bc666ef04337e7e14a523ad25ec19ac395bfb9d2` | `d04bd06eb7bea45a18d182e6b19abbda00e0b997d30257d14159d9dbc7826c0c` |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | `cfd80ba467c604d4510a5b7813cba0c008c517d080dba1879a2c02548e914cfe` | `c10d971d947f650c3eec19b59cac82f7cbc2f3a7e534e519ee774d12bc54ca77` |
| `scripts/diagnostics/e2e_walk_forward.py` (added; you read it on `b2b76fb`) | `2d82b63af39cbed972a896f7cae9d3d10ca30a3585729ce3b37fefe4573c9a40` | `3640f8457145af1144044d6c04151d089f840d9a25152799717614513f15fec1` |
| `scripts/diagnostics/e2e_pipeline_caps.py` (added; you read it on `b2b76fb`) | `6aa27e54a9b1cb61a8a9d0997d8ff4cc13b3edd6bc30577d2c668713835eaa70` | `6b21dcb41cbed0e2a70af77ab445d0c319cf897fed8aef4184267683d6b4df8d` |
| `docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md` (the note, not code) | `09cc5c95da343ee5deb736a88d6ccdbcd542ab9dfdf05695de9932e525fca6c8` | `f8ea88fd297804e5ea481b766f4a6bca73ec986ec31b5ccebc200b7c54760310` |

## What was run (VM 101, CPU only, `CUDA_VISIBLE_DEVICES=""`, one suite at a time)

- **Before landing, with no commit anywhere:** a fresh detached worktree of main `b2b76fb` with only the 15 files copied in (READY note §AD2.4-AD2.5).
  - All 14 suites passed. LL suite 53/53 with `--runh-root` and 51/51 without, mutants 15/15. Walk-forward selftest 108/108.
  - Default output against `50cd683`: 0 of 11 keys differ; non-vacuity, sieve on against base, 5 of 11.
  - The real `01_sieve_on.sh` passed its preflight.
  - The test sides read no `HEAD`: every git read is `git show` of the revision pinned to `50cd683` by full sha. The driver's one `HEAD` read (`e2e_walk_forward.py:728`) only records code identity.
- **After landing:** a fresh detached worktree of `7f4505a` with **nothing** copied in. The 15 landed paths are equal to the proven copies, 15 of 15 by sha256. Run under `taskset -c 0-3 nice -n 10`, `python3 -u -B`:

| Suite | After landing (`7f4505a`) | Before landing |
|---|---|---|
| `tests/test_ll_feature_settings.py --runh-root …` | **53/53**, mutants **15/15** | 53/53, 15/15 |
| `tests/test_ll_feature_settings.py` | **51/51**, 15/15 | 51/51, 15/15 |
| `tests/test_e2e_vote_harness.py` | **15/15**, **17/17**; G19 refused on `50cd6837abda`'s files, passed on the tree itself; M25 killed | 15/15, 17/17 |
| `tests/test_step1_screen.py` | **GATES 16/16**, **MUTANTS 15/15**, SENTINEL PASS | 15/15, 13/13 (the new gate and two mutants are `4075a8b`'s cheap-screen check, as expected) |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | **108/108** on the second run; sides `without = 50cd6837…`, `with = the tree itself` | 108/108 |
| real `01_sieve_on.sh`, stub `taskset` | `PREFLIGHT OK`, LL-2 `on`, LL-1 and LL-3 default; launch reached with the same driver argv as before landing; no out-dir and no `.time` file | the same |

- **One failure, then a clean rerun.** The first selftest run gave 107/108. Its W7 no-write gate failed because the main tree's `git status` changed during the run, and no file in the check tree changed. Another session's relay (X-LEADS-TIDY R1) wrote its request (17:33), its review (17:37) and a ledger row (17:37) into the main tree, while the selftest ran from 17:24 to 17:38. That relay had exited before the rerun. The rerun gave 108/108, and the main tree's status was the same before and after it. No new `__pycache__` appeared in any tree, and the Optuna listing was unchanged at 125 entries.
- **Evidence:** `~/tfm_worktrees/e2e-map/data/landing_llsw_20261125/`, files `E_*`, `E2_*`, `run_E.sh`, `run_E2.sh`, `kept_e/`, `wrapper_e_01_sieve_on.out` (sha256 `9ae9f121…`), `E_after.txt` (`bfe7c8e3…`), `E_summary.txt` (`3b42c55b…`). The check tree is removed.

## Remaining limits (READY note §AD2.7)

- **The neural_net family's sidecar check has not run in a real training run.** On CPU the driver refuses neural_net arms without `--vote-allow-gpu`, so W13's real runs are lightgbm. The neural_net record is tested through `--plan-only`, and the check is the same code for every family (`e2e_walk_forward.py:1045-1054`). [UNVERIFIED for neural_net at runtime]
- **The default driver record has one new key,** `switch_checks`, in `origin_summary.json`. It is the driver's own record; pipeline output is unchanged.
- **Behaviour change, intended:** an explicit `--exclude-features` list that leaves out a default exclusion used to be accepted and now refuses unless `--allow-target-columns-as-features` is given. The run H wrappers do not use `--exclude-features`.
- **Still accepted:** `'a,,b'` is read as `['a','b']`. Only an entirely empty value refuses.
- **F-PARITY stays open as recorded.** `shared` plus `on` matched the 20 metadata-backed fields. The 18 computed values differ by up to `0.000944` in your CPU check, reported without a tolerance (ch14-learn `READY_LOOP_LEARN_W1A_20261123.md:139-142`, `:274-280`).
- **Still open:** registry and manifests (owner-landed, separate; LL-2 has no WATCHER route); LL-1 `fail` against the unlearned arm A0; the overlap with `ch14-diag` and `step2-repair`.
- **You have not reviewed the fixes.** Your R1 `LIVE` objection is fixed in landed code. On the record it is closed only when the owner says so.
- **No hit-rate evidence.** Everything here concerns refusals, records and fixture execution. The hit-rate effect of LL-2 is the owner's ruled next run, 01 against 00 (`OWNER_RULINGS_pending_list_20261124.txt:25`), measured against `k/1000` on held-out draws.
