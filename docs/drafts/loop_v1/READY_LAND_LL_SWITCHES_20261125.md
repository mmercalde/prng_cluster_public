# READY · after the variant B landing, and the next landing: LL-1/2/3 pipeline switches (2026-11-25)

> **Latest: §AD2 below** (lead ADDENDUM 2, 15:15): Beta's R1 Q2 `LIVE` objection fixed and its Q3 suggestion adopted, proven in a fresh tree of main `b2b76fb` **with no commit anywhere**. Beta's review itself is §BR1 at the end of this note. No second Beta round unless the owner asks.

## AD2. Lead ADDENDUM 2 (15:15): LL-3 validation, LL-3 per family, the window-config cross-check; proof without a commit

- **Task:** `~/distributed_prng_analysis/logs/e2e_session/TASK_LL_SWITCHES_ADDENDUM_2_20261125.txt`.
- **Status:** nothing committed, anywhere; no scratch clone; nothing pushed or landed. Nothing was written in the main tree: Beta's exchange files wait there for the lead (`M docs/exchange/LEDGER.md`, the R1 request and the R1 review, untracked; `D_after.txt`). No card.
- **Evidence:** `~/tfm_worktrees/e2e-map/data/landing_llsw_20261125/`, files `D_*`, `D2_*`, `*_d_*`, `kept_d/`, `list_LL3.txt`, `copied_LL3.txt`, `run_D.sh`, `pre_ad2/` (prechecks) and `pre_ad2_*` (the four edited files as they were before this addendum).

### AD2.1 Plain summary

- **Beta's Q2 cases are all refused now, before any fitting, through the real CLI.**
  - `--include-feature-groups ''`, `--exclude-feature-groups ','` and `--exclude-features ''` refuse; a supplied empty value is never read as "omitted".
  - `--exclude-features confidnce` refuses, naming `confidnce`.
  - Excluding all 91 names refuses, because the list omits `holdout_hits` and `holdout_quality`. All 91 plus the four defaults refuses because nothing is left to train on.
  - Each refusal exits 2 with `[LL-3] REFUSED: …` and creates no model dir and no study db.
- **Target columns as features need a new explicit flag**, `--allow-target-columns-as-features`. It is off by default and is recorded in the sidecar. Given without `--exclude-features` it refuses.
- **LL-3 is recorded per family.** Beta's case: `dead` via `--step5-extra-args` and `global` via `--nn-args` is now recorded as lightgbm `dead`, neural_net `global`.
  - An empty LL-3 value refuses at the driver's preflight.
  - After training, the driver checks every family's sidecar against the record; any difference fails the run with exit 6.
- **The `--runh-root` live gate also checks the window config** (Beta's Q3 suggestion): its sha256, and the settings the leak gate takes from the fixture.
- **Proof: a fresh detached worktree of main `b2b76fb` with only the 15 landing files copied in, no commit.**
  - All 14 suites pass. The LL suite is 53/53 with `--runh-root` and 51/51 without, with mutants 15/15. The walk-forward selftest is 108/108 (the 98 from before plus 10 W13 gates).
  - Default output is unchanged against `50cd683`: 0 of 11 keys, and the 72 per-draw rows are equal apart from `config_id`. Non-vacuity: 5 of 11.
  - The real `01_sieve_on.sh` passes its preflight.
- **The landing is now 15 files.** The driver `scripts/diagnostics/e2e_walk_forward.py` and the capability module `scripts/diagnostics/e2e_pipeline_caps.py`, both on main since `50cd683`, are now modified too (Fix 2 lives there).

### AD2.2 Beta's positions (X-LL-SWITCHES R1) and what was done

| Question | Beta's position | What was done |
|---|---|---|
| Q1 Default unchanged | NO OBJECTION | Nothing needed. Default output re-proven against `50cd683` (AD2.5). |
| Q2 Switch on | OBJECTION · LIVE ("A blank LL-3 group request can produce default columns under an applied label, and a misspelled exclusion can put two normally excluded columns into training.") | **Fixed** (lead accepted it as real). Fix 1: LL-3 validation in `models/feature_manifest.py` and its call site. Fix 2: the per-family record and the post-training sidecar check in the driver and `e2e_pipeline_caps.py`. Every case Beta named is a CLI gate, with one mutant per check. Not re-reviewed by Beta (lead: no second round unless the owner asks). |
| Q3 Tests after landing | SUGGESTION (also compare the window config in the live gate) | **Adopted** (Fix 3): `g_runh_live` compares sha256 and settings; a negative check changes the offset and must fail. |
| Q4 Landed fixes | NO OBJECTION | Nothing needed. |
| Finding 1 (prior record) | the R1 request omitted governing records | **Added** to this note's §Prior record (system map §7.3, TODO P0-A/P0-B, Ch 4 audit B6/S2, Ch 7 audit X1, the W1A note, the autonomy logs). |

### AD2.3 What changed (e2e-map lines)

1. **`models/feature_manifest.py`** (new file in the landing):
   - `validate_step5_settings` (`:195`) refuses (c) a supplied empty value for any of the three flags (`:203`); the allow flag without `--exclude-features`; (a) an `--exclude-features` name that is not a feature of the survivors file (`:214`; the four default names are allowed, since (b) requires them); and (b) a list that omits any of `score, confidence, holdout_hits, holdout_quality` unless `allow_target_columns` is set (`:221`).
   - `resolve_step5_exclusions` (`:229`) returns today's default untouched when nothing is given. Otherwise it validates first, and (d) refuses an explicit list that leaves no feature (`:251`), as the group path already did.
   - The selection gains `allow_target_columns_as_features` and `requested` (the settings as given, `None` = omitted) at `:258` and `:289`. The default path writes no selection, as before.
2. **`meta_prediction_optimizer_anti_overfit.py`:**
   - constructor `allow_target_columns_as_features=False` (`:1409`, `:1444`);
   - `_load_data` calls the resolver when any setting `is not None`, so `[]` from `''` or `','` reaches it (`:1491-1498`; it used to test truthiness, which turned `[]` into "omitted");
   - CLI `--allow-target-columns-as-features` (`:3143`);
   - `main()` catches only `ManifestError` from the constructor and exits 2 with `[LL-3] REFUSED: …` (`:3206-3232`); any other exception re-raises.
   - The default path is unchanged: in-process `LL3.step5_default_identical_to_HEAD` and the walk-forward fingerprint both match `50cd683`.
3. **`scripts/diagnostics/e2e_pipeline_caps.py`** (**modifies** main's file):
   - `requested_ll3` raises `BadSwitchRequest` on an empty value and records the allow flag (`:158`);
   - `requested_ll3_by_family` (`:173`): every family gets `--step5-extra-args`; neural_net also gets `--nn-args`, last value winning as in argparse;
   - `expected_ll3_selection` / `check_ll3_sidecar` (`:184`, `:194`);
   - the LL-3 capability also requires the new flag (`LL3_BOOL`, `:38`).
4. **`scripts/diagnostics/e2e_walk_forward.py`** (**modifies** main's file):
   - `ll3_request` (`:600`) builds the per-family record and turns an empty value into a refusal before any write;
   - after each family's Step 5 and L3 check, the sidecar is compared with that family's record and stored in `origin_summary.json` `switch_checks.LL3_sidecar_selection` (`:1044-1055`); a difference raises `SwitchCheck`, exit 6 (`:128`, `:1557`; docstring exit codes);
   - three self-test mutant hooks, `ll3_flat_record`, `ll3_empty_ignored` and `ll3_sidecar_differs`, under the existing `E2E_WF_MUTANT` mechanism.
5. **`tests/test_ll_feature_settings.py`:**
   - `CLI_REFUSALS` (`:492`): 9 refusal cases through the real optimizer CLI (lightgbm, CPU, 1 trial, which is single-shot with no Optuna study);
   - `LL3.cli_allow_target_columns`: 91 features trained, flag and request in the sidecar;
   - `CLI_MUTANTS` (`:506`): 5 mutants, one per check, applied to a copy of `feature_manifest.py` in a scratch tree (`mutant_tree`, `:515`);
   - Fix 3 in `g_runh_live` (`:358`, `:392`) and the negative check `g_runh_live_catches_window_change` (`:402`).
6. **`scripts/diagnostics/e2e_walk_forward_selftest.py`:**
   - W10's LL-3 expectation is now the per-family record (`:332`);
   - new **W13** (`w13_ll3_family`, `:348`): Beta's family case via `--plan-only` plus mutant `ll3_flat_record`; the two empty-value refusals plus mutant `ll3_empty_ignored`; W1's 6 family sidecars checked with no selection; a real LL-3 `dead` run with its sidecar checked; mutant `ll3_sidecar_differs` (exit 6); and `+nocheck`, which then exits 0.

### AD2.4 Proof setup (no commit)

- `git -C ~/distributed_prng_analysis worktree add --detach ~/tfm_worktrees/_landing_check_llsw3 b2b76fb`. **HEAD stayed `b2b76fb` throughout; nothing was committed.** The 15 files were copied in as working-tree changes (`list_LL3.txt` `35f0e2cf…`, copy record `copied_LL3.txt` `7f1389bc…`).
- **Nothing in the test sides reads HEAD any more; shown by grep** (`D2_head_grep.txt` `cc3ad9de…`):
  - every git read in the landing files is `git show <pinned rev>:<file>`, at `test_ll_feature_settings.py:588,591` (`--baseline-rev`), `e2e_walk_forward_selftest.py:296,495` (`--caps-without-rev`) and `test_e2e_vote_harness.py:1006` (`CAPS_WITHOUT_REV`), all defaulting to `50cd6837abda2821277fc1cf6cc5f2d0e8f35a9a`;
  - plus `git status` (the selftest's W7 no-write check);
  - the only `HEAD` read is `e2e_walk_forward.py:728`, which records the code identity in the run record and decides nothing;
  - the other `HEAD` hits are comments and gate names.
  - So an uncommitted tree whose working files have the switches is the post-landing state for every test side. W10, W11 and G19 printed "with = the proof tree itself, without = `50cd6837abda`".
- **Base runs:** a second detached worktree at `50cd683` (`_landing_check_llsw3_base`), for the default-output comparison.
- **Prechecks** (e2e-map, then the proof tree, before the full run): the 9 CLI cases, 5 mutants and the allow case (`pre_ad2/precheck_cli.out`); W13 (`pre_ad2/precheck_w13.out`, 9 of 10; the W1 gate cannot pass without a W1 run). The precheck dir in the proof tree was deleted before the suites ran.
- **One false start:** the runner's LL output path had `$v_` (an unset variable under `set -u`). I stopped it by its recorded PID about 20 s into the first suite, deleted that suite's partial `data/e2e_vote_selftest`, fixed the path and restarted. Every result below is from the restarted run.
- **Removal:** both worktrees removed with `git worktree remove --force` after `chmod -R u+w` on their `data/`. `git worktree list` shows neither.

### AD2.5 Results (proof tree = `b2b76fb` + the 15 files, uncommitted)

One suite at a time under `taskset -c 0-5 nice -n 10`, `python3 -u -B`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=""`, venv `~/venvs/torch`. The runner ran under nohup and was polled in the foreground.

| Suite | Result | log sha256 |
|---|---|---|
| `tests/test_ll_feature_settings.py --runh-root …/e2e-map/data/e2e_runH` | **53/53**, mutants **15/15** (new: 9 `LL3.cli_refuses`, `LL3.cli_allow_target_columns`, 5 `MUTANT LL3.cli`, `LL.runH_live_catches_window_config_change`) | `ff3d6e86…` |
| `tests/test_ll_feature_settings.py` (default) | **51/51**, 15/15 | `19558a6c…` |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | **108/108** (W10 12, W11 6, W13 10); sides `without = 50cd6837…; with = the proof tree` | `4015739f…` |
| `tests/test_e2e_vote_harness.py` | 15/15, 17/17; G19 `50cd6837abda` refused, proof tree passes; M25 killed | `51ef091b…` |
| `tests/test_e2e_vote_ll5.py` | 3/3, 3/3 | `f52018e1…` |
| `tests/test_e2e_vote_ordering.py` | 6/6, 3/3 | `65e156ea…` |
| `tests/test_e2e_miss_report.py` | 7/7, 5/5 | `42c60bec…` |
| `tests/test_step1_screen.py` | GATES 15/15, MUTANTS 13/13, PASS | `af5d5a8b…` |
| `tests/test_e2e_grader.py` | 37/37 | `bb5cd569…` |
| `tests/test_compare_settings.py` | 9/9 | `fb08088e…` |
| `tests/test_switch_check.py` | 12/12 | `cdf52adf…` |
| `tests/test_registry_repair.py` | 5/5 | `08fb5c6f…` |
| `tests/mutate_e2e_grader.py` | 89/89 | `13318c69…` |
| `tests/test_window_anchor_brief2.py` | GATES 18/18, MUTANTS 12/12, PASS | `c47c3bd8…` |
| `scripts/diagnostics/e2e_step3_selftest.py` | 18/18 | `4f6e14a3…` |

- **Default output against `50cd683`** (`compare_d_ll_vs_main.json`, the selftest's own `fingerprint()`):
  - base on the proof tree against base on the `50cd683` tree: **0 of 11 keys differ**, outcomes equal;
  - the 72 per-draw rows are equal apart from `config_id`, now checked row by row (`D_per_draw_rows.txt`; rows kept in `kept_d/`);
  - control, two `50cd683` base runs (`compare_d_main_repeat.json`): the same;
  - the two comparison files are byte-identical (`c8a8bd3e…`) because all three base fingerprints are equal;
  - **non-vacuity:** sieve on against base, **5 of 11** (`step3_survivors`, `cv`, `checkpoint`, `winner`, pools).
- **The driver record at default gains one key:** `origin_summary.json` now has `switch_checks.LL3_sidecar_selection` (`{"ok": true, "expected": null, "found": null, "trained_feature_count": 89}`, `kept_d/ll_base_origin_summary.json`). It is the driver's own record, not pipeline output, and the fingerprint does not include it.
- **The real `01_sieve_on.sh`** in the proof tree (stub `taskset`, run H copy symlinked read-only; `wrapper_d_01_sieve_on.out` `cc935096…`):
  - exit 0;
  - `PREFLIGHT OK … {'LL-1': 'default (switch not used)', 'LL-2': {'--per-seed-sieve-features': 'on'}, 'LL-3': 'default (switch not used)'}`;
  - launch reached (the stub's mark holds the full driver argv); no out-dir, no `.time` file.
- **Afterwards** (`D_after.txt` `d52e167a…`):
  - no driver process; no `__pycache__` outside `data/` in either tree or the main tree;
  - Optuna listing unchanged (`optuna_before_d.txt` = `optuna_after_d.txt`, 125 entries);
  - the main tree status was as before (Beta's three exchange entries only);
  - HEADs: proof `b2b76fb`, base `50cd683`, main `b2b76fb`.

### AD2.6 Final landing list (against main `b2b76fb`)

| # | Path | from | N/M | sha256 | on `b2b76fb` |
|---|---|---|---|---|---|
| 1 | `full_scoring_worker.py` | e2e-map | **M** | `3f8d3d056ddc563c3941daf9c7166da968a3348d3668ddbddcddc8c3b1d8e184` | `2c1c963970d9…` |
| 2 | `meta_prediction_optimizer_anti_overfit.py` | e2e-map | **M** | `ad9ae8645d1f04110a95530065775a85f34c7c1f178eab0809ef1e180051d92c` | `dfc53e674d21…` |
| 3 | `prediction_generator.py` | e2e-map (LL-1 to LL-3, not LL-6) | **M** | `3cc11eb1b3add4d4d3139bd5e957ff07f37bc806afe03aaffe88e3cf0c7c8fa2` | `53e9fe7108c5…` |
| 4 | `survivor_scorer.py` | e2e-map | **M** | `efc08ae3fd6d4ac4cd08c1cd045c74fc8726935b80a9fc29f5e567fb42d056ba` | `a53a873f4bba…` |
| 5 | `models/feature_manifest.py` | e2e-map | N, **force-add** | `f60454a886415624c889a93bdf38240634bae5f62464b19faaa9db7f25ec85a2` | — |
| 6 | `models/feature_manifest.json` | e2e-map | N, **force-add** | `b2fabc87eb629ab654a73fd8f713fd2b2eb9827bf3f425b6e16b4139d6b0860f` | — |
| 7 | `tests/test_ll_feature_settings.py` | e2e-map | N | `d04bd06eb7bea45a18d182e6b19abbda00e0b997d30257d14159d9dbc7826c0c` | — |
| 8 | `tests/ll_step3_runner.py` | e2e-map (= ch14-learn's) | N | `a3556cad599257c7d1426c7be315ec26b065184a168329247e205cbb32d04259` | — |
| 9 | `tests/fixtures/runH_step3_feature_names.txt` | e2e-map | N | `7c58df695b7030acebb145f05f543d8331037b4b795d406cdb9cd4420604fb77` | — |
| 10 | `tests/fixtures/runH_winner_feature_schema.json.txt` | e2e-map | N | `5874850e2a505dc2eb3abb850c10b5d9800e273d524d3a6a60e25178158ec03f` | — |
| 11 | `tests/fixtures/runH_leak_inputs.json.txt` | e2e-map | N | `092243696acc7b1145886483c2f04a5c0098b3d4659ae9998073bf6a14ba2b4e` | — |
| 12 | `scripts/diagnostics/e2e_walk_forward_selftest.py` | e2e-map | **M** | `c10d971d947f650c3eec19b59cac82f7cbc2f3a7e534e519ee774d12bc54ca77` | `a6e2a7c42df0…` |
| 13 | `tests/test_e2e_vote_harness.py` | **e2e-vote** | **M** | `585397bdcea6b4d5a7a6770806e1e0080efffd8c5c741aaad282490fa63a0961` | `d8a0074c2e2e…` |
| 14 | `scripts/diagnostics/e2e_walk_forward.py` | e2e-map | **M** (new in the list) | `3640f8457145af1144044d6c04151d089f840d9a25152799717614513f15fec1` | `2d82b63af39c…` |
| 15 | `scripts/diagnostics/e2e_pipeline_caps.py` | e2e-map | **M** (new in the list) | `6b21dcb41cbed0e2a70af77ab445d0c319cf897fed8aef4184267683d6b4df8d` | `6aa27e54a9b1…` |

Plus this note, `docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md` (N), and Beta's R1 exchange files already in the main tree.

- **Modified landed files:** rows 1-4 (pipeline), 12-15 (test, driver and capability files landed in `50cd683` / `f7227e3`). In e2e-map, `git status` shows rows 14-15 as `M` against e2e-map's own WIP base, but e2e-map's copies before this addendum equalled main's byte for byte (`2d82b63a…`, `6aa27e54…`).
- **Changed since the R1 request:** rows 2, 5, 7, 12 (new sha256), plus 14 and 15 (added). Rows 1, 3, 4, 6, 8-11 and 13 are unchanged.
- **Force-add:** rows 5 and 6 only (`.gitignore:162` `models/`, `:41` `*.json`). Check afterwards with `git ls-files models/feature_manifest.*`.
- **Worktree files equal the proven copies:** all 15, checked by sha256 after the proof.

### AD2.7 Limits and open items

- **The NN family's sidecar check did not run in a real run.** On CPU the driver refuses neural_net arms without `--vote-allow-gpu`, so W13's real runs are lightgbm. The NN record is gated through `--plan-only`, and the check is the same code for every family (`e2e_walk_forward.py:1044-1055`). [UNVERIFIED for neural_net at runtime]
- **The default driver record has one new key** (`switch_checks`), above; pipeline output is unchanged.
- **Behaviour change, intended (Fix 1(b)):** an explicit `--exclude-features` list that omits a default exclusion used to be accepted and now refuses unless `--allow-target-columns-as-features` is given. The run H wrappers do not use `--exclude-features`.
- **Not refused:** `'a,,b'` is still read as `['a','b']` by the optimizer's `_ll3_csv`. Only an all-empty value refuses, as Fix 1(c) asks.
- **AD1's `--baseline-rev HEAD` negative check was not repeated:** here HEAD is `b2b76fb`, which has no switches, so the check does not apply. AD1's 35/36 stands as the evidence for the pin.
- **Beta has not reviewed these fixes** (lead: no second round unless the owner asks). R1's `LIVE` objection is fixed in the worktrees, but on the record it is closed only when the lead or owner says so.
- **Still open from earlier sections:** (b) registry and manifests (owner-landed, separate; LL-2 has no WATCHER route); (c) LL-1 `fail` against the unlearned arm; the overlap with `ch14-diag` and `step2-repair`.

## AD1. Lead ADDENDUM 1 (11:15): the post-landing test sides fixed, proven on a commit where HEAD has the switches

- **Task:** `~/distributed_prng_analysis/logs/e2e_session/TASK_LL_SWITCHES_ADDENDUM_1_20261125.txt`.
- **Status:** nothing committed in the project, pushed or landed. Item (b), registry and manifests, was not touched. LL-1 behaviour, item (c), was not changed.
- **Evidence:** `~/tfm_worktrees/e2e-map/data/landing_llsw_20261125/`, files `C_*`, `*_c_*` and `kept_c/`.

### AD1.1 Plain summary

- **The "also found" item is closed.**
  - The without-the-switch sides of **W10, W11 and G19** are pinned to `50cd6837abda2821277fc1cf6cc5f2d0e8f35a9a` (main before the LL landing). Each has a precondition gate that says plainly when that revision has the switch. An override is kept.
  - The with sides default to **the tree under test when it has the switches**. `~/tfm_worktrees/e2e-map` remains only as the fallback.
- **The same fault was in a fourth test, `tests/test_ll_feature_settings.py`.** Its default-identity baseline was `git show HEAD`. After the landing those gates would have compared the new code with itself and passed vacuously, without failing.
  - It is pinned to the same revision (`--baseline-rev`), with a precondition gate `LL.baseline_precondition`.
  - Measured: with `--baseline-rev HEAD` on the post-landing commit, only that gate fails, 35/36, and the identity gates pass. That is exactly the vacuous pass the pin prevents.
- **Item (a) option 1 is applied.** The LL test reads three small fixtures under `tests/fixtures/` (names; the `feature_schema` block; the leak settings with no draws). Each carries the sha256 of the run H file it was cut from.
  - `--runh-root` cross-checks them live.
  - The dataset stays read by path behind `--runh-dataset`, checked against the fixture's sha256.
  - No draw data is in `tests/`.
- **Proof in the real post-landing state.** A throwaway clone at `50cd683` got the 13 landing files and a **local commit**, so HEAD has the switches:
  - **all 14 suites pass**;
  - the LL suite is **37/37** with `--runh-root`, and **36/36** without it;
  - **default output is unchanged** (0 of 11 keys), with non-vacuity at 5 of 11;
  - **the real `01_sieve_on.sh` passes its preflight**;
  - W10, W11 and G19 used the clone itself as the with side, and `50cd683` as the without side.
  - The clone was then deleted.
- **The landing is now 13 files, not 8.** Two of them modify files already on main beyond the four pipeline files: `scripts/diagnostics/e2e_walk_forward_selftest.py` and `tests/test_e2e_vote_harness.py`. **Force-add:** `models/feature_manifest.py`, `models/feature_manifest.json`.

### AD1.2 What changed

1. **`scripts/diagnostics/e2e_walk_forward_selftest.py`** (from e2e-map; **modifies** main's `a6e2a7c42df0…`):
   - `CAPS_WITHOUT_REV = "50cd6837…"` (full sha), the default of `--caps-without-rev` (W10 and W11).
   - `default_caps_with_root()` returns this tree when `e2e_pipeline_caps.capabilities(WT)` has LL-1, LL-2 and LL-3, else `~/tfm_worktrees/e2e-map`. It is the default of `--caps-with-root`, and the chosen sides are printed at start.
   - The precondition gates of W10 and W11 now read `PINNED REVISION <sha> HAS <switch>: the refusal side would test nothing; pass --caps-without-rev a revision without it`.
2. **`tests/test_e2e_vote_harness.py`** (from e2e-vote; **modifies** main's `d8a0074c2e2e…`):
   - G19's without side is `git show $E2E_VOTE_CAPS_WITHOUT_REV:` (default `50cd6837…`), not `HEAD:`.
   - A new precondition check uses the capability module directly, not the mutated `ll1_check`, so mutant M25 is still killed by the refusal assertion.
   - The with side: `E2E_VOTE_CAPS_WITH_ROOT` if set, else this repo when it has LL-1, else e2e-map.
3. **`tests/test_ll_feature_settings.py`** (now landed from e2e-map; ch14-learn's `2916b1f8…` is the source it was edited from):
   - fixtures and `--runh-root` (`g_runh_live`);
   - `g_leak_runH(dataset_path)` with `--runh-dataset`;
   - `--baseline-rev` (default `50cd6837…`) with `LL.baseline_precondition`.
   - The gate names `*_identical_to_HEAD` are kept as they were. They now mean "identical to the pinned pre-LL revision", as the module docstring says.
4. **New fixtures:**
   - `tests/fixtures/runH_step3_feature_names.txt`: 91 names; cut from `data/e2e_runH/step3_full/survivors_with_scores.json`, sha256 `37c8b146…`.
   - `tests/fixtures/runH_winner_feature_schema.json.txt`: the `feature_schema` block, 89 names, hash `7733d30a913545ca`; source sha256 recorded.
   - `tests/fixtures/runH_leak_inputs.json.txt`: window `W12`, anchor 25, `midday`; `train_len` 14,454; the sha256 of the window config, train history and dataset; no draws.
   - The `.txt` endings are not ignored (`git check-ignore` matched only the two `models/` files), so no force-add is needed.

### AD1.3 Proof setup and the commit

- **Clone location:** `~/tfm_worktrees/_scratch_clone_llsw`, **not `/tmp` as the addendum said.**
  - The driver refuses any out-dir under `/tmp` (`FORBIDDEN`, `e2e_walk_forward.py:98`), so the walk-forward selftest and every driver probe would refuse there.
  - CLAUDE.md rule 7 also says never `/tmp`.
- **How it was made:** `git clone ~/distributed_prng_analysis <clone>`, then `git remote remove origin` straight away. `git clone` adds that remote by itself; `git remote -v` then showed nothing. Then `git checkout --detach 50cd6837…`.
- **The 13 files were copied in** (`list_LL2.txt` `3eaef83c…`; copy record `copied_LL2.txt` `7451eb67…`).
- **The commit:**
  - `git add -f models/feature_manifest.py models/feature_manifest.json`, then `git add` of the other 11;
  - one local commit, `a19a320` ("THROWAWAY local proof commit …"), made with a throwaway identity (`throwaway-proof <throwaway@localhost.invalid>`), not the owner's;
  - `git ls-files models/feature_manifest.*` showed both.
  - The commit was allowed by this session's permission check. **This is the only commit this session made, and it existed only in that clone.**
  - CLAUDE.md hard rule 1 says "Never commit or push from an agent sandbox". I took the lead's explicit instruction, for a throwaway clone with no remote, as within that rule's intent. I flag it here for the owner.
- **Base runs:** a worktree of the clone at `50cd683` (`_scratch_clone_llsw_base`) for the main-pipeline base probes.
- **Removal:** the base worktree was removed, then the clone was deleted with `rm -rf` on its exact path. Both paths are gone. Nothing was pushed, and the clone never had a remote.

### AD1.4 Results (post-landing state: HEAD = the local commit with the switches)

Suites ran one at a time under `taskset -c 0-5 nice -n 10`, with `python3 -u -B` and no card; the background runner was polled in the foreground.

| Suite | Result | log sha256 | Before (§Part 3, HEAD without the switches) |
|---|---|---|---|
| `tests/test_ll_feature_settings.py --runh-root …/e2e_runH` | **37/37**, mutants 10/10 (adds `LL.baseline_precondition`, `LL.runH_fixtures_equal_live`) | `d01e72a1…` | 35/35 |
| `tests/test_ll_feature_settings.py` (default, no live check) | **36/36**, 10/10 | `df13bd0c…` | — |
| same, `--baseline-rev HEAD` (negative check) | **35/36**: only `LL.baseline_precondition` fails | `10bae082…` | — |
| `tests/test_e2e_vote_harness.py` | 15/15, 17/17. **G19: without = `50cd6837abda`, with = the clone itself**; M25 killed | `09292f8f…` | 15/15, 17/17 |
| `tests/test_e2e_vote_ll5.py` | 3/3, 3/3 | `5a158478…` | same |
| `tests/test_e2e_vote_ordering.py` | 6/6, 3/3 | `3a29b3d7…` | same |
| `tests/test_e2e_miss_report.py` | 7/7, 5/5 | `342a1f7e…` | same |
| `tests/test_step1_screen.py` | GATES 15/15, MUTANTS 13/13, PASS | `80f402c4…` | same |
| `tests/test_e2e_grader.py` | 37/37 | `9201abd8…` | same |
| `tests/test_compare_settings.py` | 9/9 | `e2754309…` | same |
| `tests/test_switch_check.py` | 12/12 | `16e6041e…` | same |
| `tests/test_registry_repair.py` | 5/5 | `aa59e5e6…` | same |
| `tests/mutate_e2e_grader.py` | 89/89 | `907c4600…` | same |
| `tests/test_window_anchor_brief2.py` | GATES 18/18, MUTANTS 12/12, PASS | `9a2eaaa9…` | same |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | **98/98**; sides printed `without = 50cd6837…; with = <the clone>`; W10 13, W11 6, W12 16 pass | `d923cfe7…` | 98/98 |
| `scripts/diagnostics/e2e_step3_selftest.py` | 18/18 | `db1ac1f4…` | same |

- **Default output against `50cd683`** (`compare_c_ll_vs_main.json`):
  - base on the clone (LL code, LL commit) against base on the clone's `50cd683` worktree: **0 of 11 keys differ**, and the outcomes are equal;
  - the 72 per-draw rows are equal apart from `config_id`;
  - the control (two `50cd683` base runs) differs only in `config_id`;
  - **non-vacuity:** sieve on against base on the clone, **5 of 11 keys differ** (`step3_survivors`, `cv`, `checkpoint`, `winner`, pools).
- **Real `01_sieve_on.sh` in the clone** (stub `taskset`; `wrapper_c_01_sieve_on.out` `17375036…`):
  - exit 0;
  - `PREFLIGHT OK … pipeline switches {'LL-1': 'default (switch not used)', 'LL-2': {'--per-seed-sieve-features': 'on'}, …}`;
  - `started …`;
  - launch reached, with no out-dir and no `.time` file.
- **Afterwards:**
  - no driver process;
  - no `__pycache__` outside `data/`;
  - the clone's tracked state was clean apart from `data/`;
  - no new Optuna study (`optuna_before_c.txt` = `optuna_after_c.txt`);
  - the main tree showed only the closing note from the earlier task.

### AD1.5 Final landing list (against main `50cd683`)

| # | Path | from | N/M | sha256 | on `50cd683` |
|---|---|---|---|---|---|
| 1 | `full_scoring_worker.py` | e2e-map | **M** | `3f8d3d056ddc563c3941daf9c7166da968a3348d3668ddbddcddc8c3b1d8e184` | `2c1c963970d9…` |
| 2 | `meta_prediction_optimizer_anti_overfit.py` | e2e-map | **M** | `16cd3ae5fa646ffe7f0711f4d3c81f3083f4bad590d6196303072439f8fe3343` | `dfc53e674d21…` |
| 3 | `prediction_generator.py` | e2e-map (LL-1 to LL-3, not LL-6) | **M** | `3cc11eb1b3add4d4d3139bd5e957ff07f37bc806afe03aaffe88e3cf0c7c8fa2` | `53e9fe7108c5…` |
| 4 | `survivor_scorer.py` | e2e-map | **M** | `efc08ae3fd6d4ac4cd08c1cd045c74fc8726935b80a9fc29f5e567fb42d056ba` | `a53a873f4bba…` |
| 5 | `models/feature_manifest.py` | e2e-map | N, **force-add** | `a454e3117ae1c7659ce409a0f3d3ce71944a5dac879eb9580817e3118beec1c5` | — |
| 6 | `models/feature_manifest.json` | e2e-map | N, **force-add** | `b2fabc87eb629ab654a73fd8f713fd2b2eb9827bf3f425b6e16b4139d6b0860f` | — |
| 7 | `tests/test_ll_feature_settings.py` | **e2e-map** (edited from ch14-learn's) | N | `4be39826273e758a916a8dd8bc666ef04337e7e14a523ad25ec19ac395bfb9d2` | — |
| 8 | `tests/ll_step3_runner.py` | e2e-map (= ch14-learn's, unchanged) | N | `a3556cad599257c7d1426c7be315ec26b065184a168329247e205cbb32d04259` | — |
| 9 | `tests/fixtures/runH_step3_feature_names.txt` | e2e-map | N | `7c58df695b7030acebb145f05f543d8331037b4b795d406cdb9cd4420604fb77` | — |
| 10 | `tests/fixtures/runH_winner_feature_schema.json.txt` | e2e-map | N | `5874850e2a505dc2eb3abb850c10b5d9800e273d524d3a6a60e25178158ec03f` | — |
| 11 | `tests/fixtures/runH_leak_inputs.json.txt` | e2e-map | N | `092243696acc7b1145886483c2f04a5c0098b3d4659ae9998073bf6a14ba2b4e` | — |
| 12 | `scripts/diagnostics/e2e_walk_forward_selftest.py` | e2e-map | **M** | `cfd80ba467c604d4510a5b7813cba0c008c517d080dba1879a2c02548e914cfe` | `a6e2a7c42df0…` |
| 13 | `tests/test_e2e_vote_harness.py` | **e2e-vote** | **M** | `585397bdcea6b4d5a7a6770806e1e0080efffd8c5c741aaad282490fa63a0961` | `d8a0074c2e2e…` |

Plus this note, `docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md` (N).

- **Modified landed files:** rows 1-4 (pipeline), 12 and 13 (test files landed in `50cd683` and `f7227e3`).
- **Force-add:** rows 5 and 6 only. Check afterwards with `git ls-files models/feature_manifest.*`.
- The worktree files equal the proven copies (sha check: all 13 equal).

### AD1.6 Still open (unchanged by this addendum)

- **Item (b):** registry and manifests are not touched. LL-2 still has no WATCHER route (§(b) below).
- **Item (c):** LL-1 `fail` against the unlearned arm. Behaviour unchanged; options are in §(c) below.
- **After this lands, the walk-forward selftest can stop reading e2e-map.** Its with side becomes the tree itself. The fallback path stays in the code for trees without the switches.
- **The overlap with `ch14-diag` and `step2-repair`** (§Also found) is unchanged.

- **Task:** `~/distributed_prng_analysis/logs/e2e_session/TASK_AFTER_WF_VOTE_LANDING_20261125.txt` (lead, 10:10).
- **Author:** Claude (Team Alpha), worktree `~/tfm_worktrees/e2e-map`.
- **Status:** nothing committed, pushed or landed. No card (`CUDA_VISIBLE_DEVICES=""`).
- **Evidence:**
  - `~/tfm_worktrees/e2e-map/data/landing_llsw_20261125/` (Part 3);
  - `~/tfm_worktrees/e2e-map/data/landing_wfvote_20261125/` (Part 1, the `POST*` files).

  Both are ignored `data/`.

## Prior record

**Searches run** (main `50cd683`):

```
S = docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_*
    docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/ docs/drafts/
git grep -l -e P 50cd683 -- $S   for P in: 'per_seed_sieve_features\|per-seed-sieve' | 'schema_check\|schema-check' |
    'exclude-feature-groups\|include_feature_groups' | feature_manifest | 'LL-1\b\|LL-2\b\|LL-3\b'
/bin/grep -l -e 'LL-1\|LL-2\|LL-3\|per-seed\|schema_check\|feature_manifest' ~/distributed_prng_analysis/logs/e2e_session/*.txt
/bin/grep -l -i registry ~/distributed_prng_analysis/logs/e2e_session/*.txt
```

**Documents found, each read at the hit:**
- `docs/drafts/loop_v1/SWITCH_CHECK_20261123.md`: all six LL options are rows "not registered", each with a proposed `ParameterSpec`.
  - Step 3 rows: `:119-122` (`full_scoring_worker.py:570`, `:573`).
  - Step 5 rows: `:179-184`.
  - Step 6 rows: `:187`, `:201`.
  - Manifest targets: `:472-478` (`agent_manifests/reinforcement.json`, `agent_manifests/prediction.json`); `full_scoring_worker.py` is classed "not a WATCHER step script" (`:458-459`).
- `docs/drafts/loop_v1/switch_patches_20261123/parameter_registry.py.patch` and `manifest_fragments.json`: a **generated draft** that already registers all six options.
  - In the patch: `full_scoring_worker.py` registry at `:2002`; Step 3 harness at `:714`, `:895-898`; `meta_prediction_optimizer_anti_overfit.py` options at `:51-65`; `prediction_generator.py` `schema_check` at `:81`.
  - In the fragments: the four LL-1/LL-3 manifest entries. LL-2 has none.
  - Its status line reads "DRAFT, not applied. agent_manifests/ is HAND_LANDED_DIRS (scripts/owner_land.py): the owner lands it by hand".
- `docs/drafts/loop_v1/AUTONOMY_SWITCH_MAP_20261123.md`: the map of registry → manifest → grammar → apply.
- `docs/drafts/loop_v1/REGISTRY_REPAIR_20261124.md`: owner ruling #7. It drops the 20 stale entries, including the never-existing `full_scoring.py` registry. It is a draft, **not applied on main**: `agents/registry/parameter_registry.py` at `50cd683` differs from `registry_repair_20261124/parameter_registry.py.repaired`.
- **Task logs:**
  - `OWNER_DIRECTION_autonomy_switches_20261123.txt:2` (owner, verbatim): "at some point the code will need to know and control those switches … We also have everything coded for pedantic vocabulary, agent manifest etc".
  - `TASK_SWITCH_REGISTER_ADDENDUM_1.txt:1`: "do NOT create a new, parallel register as the source of truth … the switches must live there".
  - `OWNER_RULINGS_pending_list_20261124.txt:18` (#7) and `:25` (the next run is 01 against 00).
  - `OWNER_GO_wf_sieve_on_block0_20261124.txt`: block 0 with sieve on was already run, from the e2e-map tree.
- `~/tfm_worktrees/ch14-learn/docs/drafts/loop_v1/READY_LOOP_LEARN_W1A_20261123.md` (not on main): the LL-1 to LL-3 as-built record. `:76` says force-add `models/`; `:279` says the production launcher route does not pass the Step 3 flag.

**Added after Beta X-LL-SWITCHES R1 finding 1** (governing records my LL-* searches missed, because they use P0-B / F-PARITY / Option A, not LL names; each read at the hit, main `b2b76fb`, 2026-11-25):
- `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:159-169`, §7.3: defect 2, the sequential fallback merges 6 of 18 metadata fields, with the required **Gate F-PARITY**; defect 3, `forward_matches` / `reverse_matches` never become features, with the TB §8 ruling "do NOT silently add them" and **Option A** (add under a new schema version after leakage and redundancy analysis, then retrain) or Option B. LL-2 is Option A built as a setting, default `off`.
- `docs/TODO_CONSOLIDATED.md:62` (P0-B: exact metadata guard, identical batch and sequential merges, F-PARITY tests, fail closed) and `:159-161` (the `forward_matches` / `reverse_matches` schema decision; P0-A, the canonical 91/89-feature manifests; P0-B).
- `docs/CHAPTER_4_AUDIT.md:119-124` (B6, the fallback's 6 of 18, governed by P0-B) and `:142-146` (S2, per-seed sieve match rates not Step 3 features, held by TB §8).
- `docs/CHAPTER_7_AUDIT.md:282` (X1: the Step 6 schema-hash check never runs; LL-1's `--schema-check` is the setting for it).
- `~/tfm_worktrees/ch14-learn/docs/drafts/loop_v1/READY_LOOP_LEARN_W1A_20261123.md:44-57` (the original LL note's prior record and labels for exactly these items) and `:274-284` (its limits: F-PARITY reported without a tolerance; LL-2 not reachable from WATCHER).
- `logs/e2e_session/OWNER_DIRECTION_autonomy_switches_20261123.txt:2` and `logs/e2e_session/TASK_SWITCH_REGISTER_ADDENDUM_1.txt:1-4`: already cited above and in §(b); the R1 request left them out of its prior record.

**Labels:**

| Item | Label |
|---|---|
| LL-2 sieve switch; shared merge list and F-PARITY; LL-1 manifest and Step 6 schema check | ALREADY RECORDED (system map §7.3 items 2-3, Option A; TODO P0-A, P0-B; Ch 4 audit B6, S2; Ch 7 audit X1); built as settings with today's defaults |
| Landing the LL-1/2/3 pipeline files | ALREADY RECORDED as needed (`OWNER_RULINGS:25`; `READY_LAND_WF_VOTE_20261125.md` §A4.4); this proof is NEW |
| Where the switches must be declared (b) | ALREADY RECORDED (`SWITCH_CHECK_20261123.md`, the `switch_patches_20261123` draft) |
| The LL test's absolute paths (a) | ALREADY RECORDED (`READY_LAND_WF_VOTE_20261125.md` §3(b), §7.9); the options below are NEW |
| LL-1 `fail` against the unlearned arm (c) | ALREADY RECORDED (`READY_LAND_WF_VOTE_20261125.md` §7.3); the options below are NEW |
| CHANGES A RULING | none |

**Not searched:** ser8, the private repository, project files outside the repository, the rigs.

---

## Plain summary

- **Part 1, the commit is complete.** A detached `50cd683` with **nothing copied in** gives exactly the pre-landing numbers:

  | Suite | Result | Pre-landing |
  |---|---|---|
  | `tests/test_e2e_vote_harness.py` | 15/15, 17/17 | same |
  | `tests/test_step1_screen.py` | 15/15, 13/13 | same |
  | walk-forward selftest | **99/99** | same |

  All 33 landed files are byte-identical to the proven copies.
- **Part 2, closing note.** `docs/exchange/to_beta/X-E2E-WF-VOTE_R2_ALPHA_CLOSING_NOTE.md` (sha256 `f19704e6…`) is written and left **uncommitted in the main tree** for the lead. It was sent as earlier closing notes were: placed "for the record", not through the relay.
- **Part 3, the LL switch landing is ready.** On a fresh `50cd683` with only the 8 files:
  - **all 14 suites pass**;
  - **default output is unchanged** against `50cd683` (0 of 11 fingerprint keys differ; the per-draw rows differ only in `config_id`, which also differs between two runs on main);
  - **non-vacuity holds:** per-seed sieve on changes 5 of 11 keys;
  - **`01_sieve_on.sh` now PASSES its preflight in a tree built from main**, the real wrapper, as do W10 and W11.
- **Landing list:** 8 files. **`models/feature_manifest.py` and `models/feature_manifest.json` need `git add -f`** (`.gitignore:162 models/`; the json also matches `:41 *.json`).
- **Reported, not fixed:**
  - (a) the LL test's run H reads;
  - (b) where the switches must be declared. **The draft already exists; `agent_manifests/` is owner-hand-landed; the registry is not governed.** One gap: LL-2 has no WATCHER route at all;
  - (c) LL-1 `fail` and the unlearned arm.

---

## Part 1. Proof that the commit is complete

- **Setup:** `git worktree add --detach ~/tfm_worktrees/_post_landing_check 50cd683`. Nothing copied in: `git status --porcelain` was empty.
- **How the suites ran:** one at a time under `taskset -c 0-5 nice -n 10`, with `python3 -u -B` and no card.

| Suite | Now | log sha256 | Pre-landing (`READY_LAND_WF_VOTE_20261125.md` §A4.2) |
|---|---|---|---|
| `tests/test_e2e_vote_harness.py` | gates 15/15, mutants 17/17 | `c39e828b…` | 15/15, 17/17 |
| `tests/test_step1_screen.py` | GATES 15/15, MUTANTS 13/13, SENTINEL PASS | `4c86199d…` | 15/15, 13/13 |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | **99/99**, 758 s | `5cd348ed…` | 99/99 |

- **Byte check:** for each of the 33 list-B paths, `git show 50cd683:<path>` equals the round-3 copy record `copied_B_r3.txt`: **33 of 33 equal**. The landing also carries this file's sibling, `READY_LAND_WF_VOTE_20261125.md`.
- **Afterwards:**
  - no `__pycache__`;
  - no new Optuna study (`optuna_before_post.txt` = `optuna_after_post.txt`);
  - no stray driver process;
  - the main tree's status was empty at that point;
  - the tree was removed.
- **Logs:** `landing_wfvote_20261125/POST_*.log` and `POSTwf_*.log`; summary JSON in `kept_post/`.

## Part 2. Courtesy closing note to Beta

- **File:** `docs/exchange/to_beta/X-E2E-WF-VOTE_R2_ALPHA_CLOSING_NOTE.md`, untracked in the main tree. Nothing else in the main tree was changed.
- **Precedent:**
  - `X-PWC-STAGE2_R3_ALPHA_CLOSING_NOTE.md:3` ("Not sent through the relay");
  - `X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_stage4_tooling.md:1-5` ("Sent for the record … at the Alpha lead's request");
  - `docs/exchange/LEDGER.md` has no closing-note rows.

  So no relay was run, and the note opens with the same "Sent for the record" comment.
- **Contents:**
  - each R1 position, what was done, and the `file:line` on `50cd683`;
  - the covering gate (W11, W12, G19/M25);
  - the nine changed sha256 (as Beta read them → at `50cd683`);
  - the owner's decision as the lead records it (`TASK_AFTER_WF_VOTE_LANDING_20261125.txt:1-3`, 10:06; no second round);
  - the Part 1 numbers;
  - the remaining limits from §A4.5.
- **Q2 label:** it states that Alpha had contested the `LIVE` label on Q2 and that the owner was not asked to rule.

## Part 3. The LL-1/2/3 landing, proven on a fresh copy of `50cd683`

**Files** (`list_LL.txt` `c17aefc6…`; copy record `copied_LL.txt` `611d9a37…`):

| # | Path | from | N/M on `50cd683` | sha256 |
|---|---|---|---|---|
| 1 | `full_scoring_worker.py` | e2e-map | M | `3f8d3d056ddc…` |
| 2 | `meta_prediction_optimizer_anti_overfit.py` | e2e-map | M | `16cd3ae5fa64…` |
| 3 | `prediction_generator.py` | e2e-map (LL-1 to LL-3; **not** ch14-learn's LL-6 `4f6a26bf…`) | M | `3cc11eb1b3ad…` |
| 4 | `survivor_scorer.py` | e2e-map | M | `efc08ae3fd6d…` |
| 5 | `models/feature_manifest.py` | e2e-map | N, **force-add** | `a454e3117ae1…` |
| 6 | `models/feature_manifest.json` | e2e-map | N, **force-add** | `b2fabc87eb62…` |
| 7 | `tests/test_ll_feature_settings.py` | ch14-learn | N | `2916b1f82f25…` |
| 8 | `tests/ll_step3_runner.py` | ch14-learn | N | `a3556cad5992…` |

- Rows 1-6 are byte-identical in e2e-map and ch14-learn, except `prediction_generator.py`. ch14-learn's LL-6 version needs the ignored, untracked `models/inference_bundle.py` (`READY_LAND_WF_VOTE_20261125.md` §4).
- **None of the four pipeline files changed on main between `fe5ac3b` and `50cd683`**, so the round-1 diffs still apply as they are. Their e2e-map diff against `41ad509` is +199 −23.
- **Where each switch sits** (e2e-map line numbers):
  - **LL-1:** `prediction_generator.py:134` (field), `:197` (`from_json`), `:783-794` (the check), `:999` (`--schema-check`).
  - **LL-2:** `full_scoring_worker.py:312-323` (`score_survivors(…, per_seed_sieve_features='off', fallback_metadata_merge='legacy6')`), `:570`, `:573` (CLI); `survivor_scorer.py:61` (`METADATA_MERGE_FIELDS`), `:86` (`metadata_merge_fields`).
  - **LL-3:** `meta_prediction_optimizer_anti_overfit.py:1406` (constructor), `:3128`, `:3131`, `:3134` (CLI).
- Landing: `git add -f models/feature_manifest.py models/feature_manifest.json`, then check `git ls-files models/feature_manifest.*` shows both (the lost-file lesson, `READY_GRADER_LANDING_20261124.md:111`). `git status` in the check tree did not list them, which confirms they are ignored.

**Setup:** `git worktree add --detach ~/tfm_worktrees/_landing_check_llsw 50cd683`. Two base probes ran on main's own pipeline **before** anything was copied in. Then only the 8 files were copied in. Suites ran one at a time under `taskset -c 0-5 nice -n 10`, with `python3 -u -B` and no card; the background runner was polled in the foreground.

### Results: 14 suites

| Suite | Result | log sha256 | Variant B on `fe5ac3b` (round 3) |
|---|---|---|---|
| `tests/test_ll_feature_settings.py` | **35/35 gates, mutants 10/10**; `LL2.default_bit_identical_to_HEAD` `b00c34c1…` both and `LL2.fallback_default_bit_identical_to_HEAD` `4970ec9d…` both, HEAD = `50cd683` | `e504c331…` | n/a |
| `tests/test_e2e_vote_harness.py` | 15/15, 17/17 | `059a9c4a…` | 15/15, 17/17 |
| `tests/test_e2e_vote_ll5.py` | 3/3, 3/3 | `93a4f0f2…` | same |
| `tests/test_e2e_vote_ordering.py` | 6/6, 3/3 | `7aeee85c…` | same |
| `tests/test_e2e_miss_report.py` | 7/7, 5/5 | `28ac7f06…` | same |
| `tests/test_step1_screen.py` | GATES 15/15, MUTANTS 13/13, PASS | `32bd4749…` | same |
| `tests/test_e2e_grader.py` | 37/37 | `d2a9ff87…` | same |
| `tests/test_compare_settings.py` | 9/9 | `68254c84…` | same |
| `tests/test_switch_check.py` | 12/12 | `a9b159ae…` | same |
| `tests/test_registry_repair.py` | 5/5 | `42065f1f…` | same |
| `tests/mutate_e2e_grader.py` | 89/89 killed and credited | `71de6aee…` | same |
| `tests/test_window_anchor_brief2.py` | GATES 18/18, MUTANTS 12/12, PASS | `1b8f3303…` | same |
| `scripts/diagnostics/e2e_walk_forward_selftest.py` | **98/98** | `c954eb67…` | 99/99 |
| `scripts/diagnostics/e2e_step3_selftest.py` | 18/18 | `fecf9cc3…` | same |

- **Why 98 and not 99:** the W10 direct-harness refusal case applies only when the tree lacks LL-2. Here it prints `INFO W10 this tree has LL-2: the direct-harness refusal case does not apply`.
- All 13 W10 gates and 6 W11 gates pass. W10 on this tree:
  - each of LL-1, LL-2 and LL-3 "present in this tree: the driver plans it and records it applied";
  - the refusal side still runs on `HEAD`'s own files, which lack the switches.
- Logs: `landing_llsw_20261125/LL_*.log` and `LL_summary.txt`.

### `01_sieve_on.sh` passes the preflight in a tree built from main

- **What ran:** the real wrapper `docs/drafts/loop_v1/runH_commands/01_sieve_on.sh` in the check tree, after the 8 files were copied in, with `CORES=0 W=1 STAGE=smoke`. A stub `taskset` recorded its call and ran nothing (`wrapper_01_sieve_on.out` `70e27e38…`).
- **Result:**
  - exit 0;
  - `[DEVELOPMENT] PREFLIGHT OK: …/data/e2e_runH_wf_01_sieve_on_smoke would be written; pipeline switches {'LL-1': 'default (switch not used)', 'LL-2': {'--per-seed-sieve-features': 'on'}, …}`;
  - `started …`;
  - the launch was reached, with no out-dir and no `.time` file.
- `00_base.sh` gave the same (`wrapper_00_base.out` `7892fdb5…`).
- Under variant B on `50cd683`, this same wrapper exits 2 at the preflight (W11, Part 1).
- No driver process ran. The two stub-run logs were deleted before the suites.

### Default output against `50cd683`, with non-vacuity (`compare_default.py`, the selftest's own `fingerprint()`, 11 keys)

| Comparison | Keys differing | Per-draw outcomes | Per-draw grade rows |
|---|---|---|---|
| main base vs main base, repeat (control) | 0 of 11 | equal | 72 rows; only `config_id` differs |
| **LL tree base vs main base** | **0 of 11** | equal | 72 rows; equal apart from `config_id` (the same field that differs in the control) |
| **non-vacuity:** LL tree sieve on vs LL tree base | **5 of 11**: `step3_survivors`, `cv/s42/lightgbm`, `checkpoint/s42/lightgbm`, `winner/s42`, `pools/A1s42_o1_p20` | — | — |

Records: `compare_LL_vs_main.json` and `compare_main_repeat.json`. Both have sha256 `c8a8bd3e…`, which is expected: the file holds only fingerprints, the repeat's equals the LL base's, and both calls used the same sieve-on run as the third argument.

### Every LL case on the LL tree (`probe_LL.json`)

| Case | Exit | Note |
|---|---|---|
| 00 base, 01 sieve on, LL-2 shared, 02, 03 (one ranker), LL-3 `--exclude-features`, LL-3 `--exclude-feature-groups dead` | 0 | |
| LL-3 `--include-feature-groups` | 3 | The first try used the wrong group name `sieve` (my error). Rerun with `per_seed_sieve` (`probe_LL_include_fixed.json`): exit 3 again. LL-3 itself refuses with `ManifestError: the feature-group settings leave no feature to train on` (`models/feature_manifest.py:224`). With LL-2 off, the `per_seed_sieve` group holds no column, so this is fail-closed and correct, not a landing defect |
| LL-1 `schema_check: fail` | 3 | at `o1/vote/A0_o1_p20`; see (c) |

**Afterwards:**
- no driver process;
- no `__pycache__` outside `data/`, and none in `models/` or `tests/`;
- tracked changes were the 4 pipeline files, and untracked the 2 tests;
- no new Optuna study (`optuna_before.txt` = `optuna_after.txt`);
- the main tree showed only the closing note;
- the check tree was removed (`git worktree list`: 0).

---

## Reported, not fixed

### (a) How `tests/test_ll_feature_settings.py` can stop reading `~/tfm_worktrees/e2e-map/data` by absolute path

**What it reads** (`ch14-learn:tests/test_ll_feature_settings.py:34-39`), read only:

| Constant | Path | Size | Used by |
|---|---|---|---|
| `RUNH_STEP3` | e2e-map `data/e2e_runH/step3_full/survivors_with_scores.json` | **92.2 MB** | `g_manifest_rebuild` (`:192-200`): only `sorted(row[0]["features"])`, the 91 names |
| `RUNH_SIDECAR` | e2e-map `data/e2e_runH/step45/winner/best_model.meta.json` | 5.3 KB | `g_manifest_vs_sidecar` (`:203-213`): `feature_schema.feature_names` (89) and `feature_schema_hash` |
| `RUNH_TRAIN` | e2e-map `data/e2e_runH/copy/train_history.json` | 99.6 KB | `g_leak_runH` (`:314-325`): the train-is-prefix check; 14,454 draws |
| `RUNH_DATASET`, `RUNH_WINDOW_CFG` | `~/skipsw_run_thr3030/daily3.json`, `optimal_window_config.json` | 1.38 MB, 1.7 KB | `g_leak_runH`: the window index map against `miner.range_miner_worker.load_residue_window` |

**Options** (none applied):

1. **Small fixtures, following the existing practice of `tests/fixtures/*.txt` with a sha256 pin, as for `e2e_grader_a4d96fe7.py.txt` and `e2e_vote_harness_rev*.py.txt`:**
   - `tests/fixtures/runH_step3_feature_names.txt` (91 names, one per line, in place of the 92 MB file);
   - `tests/fixtures/runH_winner_feature_schema.json` (only the `feature_schema` block of the sidecar).
   - Each records the sha256 of the run H file it was cut from. A json fixture needs `git add -f` (`.gitignore:41`; precedent `81c9386`), or it can be stored as `.json.txt`.
   - The gates read the fixtures. An optional `--runh-root` re-derives them from the live files and fails if they differ, so the live cross-check stays available without being required.
2. **The leak gate:** the train history is the dataset's first 14,454 draws by construction (the run copy builder). The fixture can hold `{train_len, dataset_sha256, window_cfg}`, and the gate rebuilds train from the dataset.
   - The dataset itself lives in a run tree (`~/skipsw_run_thr3030`), not a worktree. The run H wrappers read the same tree (`runH_commands/0*.sh:21`).
   - Keeping that one read by path, through a `--runh-dataset` option with today's path as default, is consistent with how run inputs are read elsewhere.
   - Copying 1.38 MB of draws into `tests/` is the alternative. I would not do it without the owner, as it puts draw data in the repository.
3. **Either way,** `RUNH = Path.home() / "tfm_worktrees/e2e-map/…"` should go. The gates then stop depending on another agent's worktree, which is the same class of fix as the vote repin.

### (b) Where the three switches must be declared, and which files are governed

The owner's direction says the switches live in the existing machinery, not a new list (`OWNER_DIRECTION_autonomy_switches_20261123.txt:2`; `TASK_SWITCH_REGISTER_ADDENDUM_1.txt:1`).

**1. Know: `agents/registry/parameter_registry.py`**
- `ParameterSpec` (`:26`) and `PARAMETER_REGISTRIES` (`full_scoring.py` at `:411`, `meta_prediction_optimizer_anti_overfit.py` at `:483`, `prediction_generator.py` at `:547`); the agent lookup is at `:604-647`.
- **Not governed and not hand-landed:** it is not in `GOVERNED_FILES` (`scripts/gate12_parity_gate.py:174`, 10 files: `adaptive_thresholds.py`, `execution_set.py`, `hybrid_strategy.py`, `miner/*` ×5, `prng_registry.py`, `sieve_gpu_worker.py`), nor in `HAND_LANDED_FILES` or `HAND_LANDED_DIRS` (`scripts/owner_land.py:86-87`).
- **The draft entries already exist,** generated by `switch_check.py`: `switch_patches_20261123/parameter_registry.py.patch`, which includes all six LL options.
- **Two snags:**
  - that patch was generated against the pre-repair registry at `41ad509`, and cites line numbers of drafts since changed (for example `e2e_step3_harness.py:560-574`, now moved). It should be **regenerated** (`scripts/diagnostics/switch_check.py`, on main) after
  - **the registry repair lands** (owner ruling #7, `OWNER_RULINGS:18`; draft `REGISTRY_REPAIR_20261124.md`, not applied). That repair removes the never-existing `full_scoring.py` registry, which is where the Step 3 entries would otherwise go.
- The order: registry repair, then regenerated registration.

**2. Reach: `agent_manifests/*.json` `default_params` / `args_map`**
- **`agent_manifests/` is in `HAND_LANDED_DIRS`: the owner lands it by hand** (`scripts/owner_land.py:87`).
- The draft fragments `switch_patches_20261123/manifest_fragments.json` add:
  - `exclude_features`, `include_feature_groups` and `exclude_feature_groups` to `reinforcement.json`;
  - `schema_check` to `prediction.json`.
- **LL-2 has no fragment, and no route:**
  - Step 3 under WATCHER runs `full_scoring_worker.py` through `agent_manifests/full_scoring.json:37-45` (`run_distributed`, `args_map` only `job-file`, `gpu-id`, `lottery-file`), via the job file that `generate_step3_scoring_jobs.py:262` writes.
  - `run_step3_full_scoring.sh` and the job generator pass no LL-2 flag (ch14-learn `READY_LOOP_LEARN_W1A_20261123.md:279`).
  - **So LL-2 can be declared in the registry, but cannot be reached by WATCHER** until the job generator, the launcher, and possibly `scripts_coordinator.py` carry it. That is pipeline code beyond these 8 files.
  - Today LL-2 is reached only by the direct worker CLI and through the e2e Step 3 harness, which is what `01_sieve_on` uses.
- `agents/watcher_agent.py` is hand-landed (`owner_land.py:86`) if WATCHER itself must change.

**3. Propose, apply, prove.** The grammars, `watcher_policies.json` and the applier are the chain mapped in `AUTONOMY_SWITCH_MAP_20261123.md`. The applier (B12 / LL-16) is held by the owner's earlier choice, recorded there; nothing new here.

### (c) LL-1 `fail` and the unlearned arm

**What happens, measured** (`probe_LL.json`, case `LL1_schema_check`, on the LL tree):
- exit 3, `STEP FAILED … o1/vote/A0_o1_p20: exit 1`;
- the vote log says `ManifestError: [LL-1 schema-check] sidecar has no feature names; sidecar has no feature_schema_hash`;
- the learned arm `A1s42` voted normally before it (36 pools).

**Why:**
- The unlearned arm A0 (`field_sum`) builds a generator with no sidecar (`e2e_vote_harness.py:617-626` `_construct_unlearned`).
- With `schema_check` on, `prediction_generator.py:783-794` checks the rows against `model_meta.feature_schema`, and that is empty for A0.
- `models/feature_manifest.py:240-300` `check_rows_against_schema` records "no feature names" and "no feature_schema_hash", and in `fail` mode raises (`:295`). In `warn` it logs and continues.
- So **one vote config with `schema_check: fail` stops every walk-forward at origin 1**, because the driver always votes A0.

**Options** (none applied; each would be a setting, recorded in the run record):
1. The vote harness applies `schema_check` to learned arms only, and records `schema_check: not applicable (no sidecar)` for A0. This is the narrowest, and matches what LL-1 checks: a sidecar's schema.
2. The driver refuses LL-1 `fail` while A0 is in the run, naming the conflict, before any write. This is honest, but it makes LL-1 `fail` unusable in the walk-forward.
3. `fail` on A0 is downgraded to `warn`. I would not choose this: it is a silent change of setting under one label, which ADDENDUM 1 rules out.

My recommendation is 1, as a flag-selectable default. The owner or lead decides.

---

## Also found (not asked)

- **Other worktrees edit the same pipeline files.** Landing this set changes the base under them:
  - `ch14-diag:meta_prediction_optimizer_anti_overfit.py` (`763e351c…`, diagnostics flags at `:3146`, `:3151` per `SWITCH_CHECK:470-471`);
  - `step2-repair`: all four pipeline files differ from main (`1388f04d…`, `cb01bfff…`, `b24bf0c8…`, `b1d94561…`).

  Those branches will need a re-base, or hunk-by-hunk merges, after this lands (`TASK_LOOP_COMMON.txt`: "several worktrees edit meta_prediction_optimizer_anti_overfit.py … list every hunk"). Not examined beyond the sha.
- **The W11 "with" side still reads `~/tfm_worktrees/e2e-map`.** That is by design (the non-vacuity source). After this landing, main itself has LL-2, so W11's "without" side (`git show HEAD`) would then **have** the switch and its precondition gate would fail.
  - The same holds for W10's and G19's "without" sides.
  - **After this landing, `--caps-without-rev` (and the G19 HEAD copy) must point at a revision without the switches, for example `50cd683`. Otherwise those selftests fail their preconditions on main.** I checked this only by reading `w10_switches` and `w11_wrappers`, which take `HEAD` by default, and `g19_ll1_switch`, which uses `git show HEAD`. It needs a test-only change in the same landing or right after. Not done here; the task said report only for (a)-(c), and this is a test edit the lead should schedule.
- **Probe input errors of mine:** the `sieve` group name, corrected above.

Stopped. No commit, no push, no landing.

---

## BR1. Beta courtesy review X-LL-SWITCHES R1 (lead/owner `TASK_LL_SWITCHES_BETA_20261125.txt`, 14:38)

- **Request:** `docs/exchange/to_beta/X-LL-SWITCHES_R1_ALPHA_REVIEW_REQUEST_ll_switch_landing.md` (main tree, untracked), sha256 `d4bd21ec46f99491b620982cac25ad0cb8637e108e127006288307485a493ded`. Base `b2b76fb`.
- **Reply:** `docs/exchange/from_beta/X-LL-SWITCHES_R1_BETA_REVIEW.md` (main tree, untracked), sha256 `a7355640cf05e8f6f891b2f93be0499ae1f675a37193005861a0472076ba5232`. Read in full; not edited.
- **Relay:** exit 0, `REVIEW RECEIVED`, elapsed 428.7 s; log `logs/beta_relay/X-LL-SWITCHES_R1.log`; output `data/landing_llsw_20261125/D_relay.out`. Ledger row appended by the relay (`docs/exchange/LEDGER.md`, now modified in the main tree). No `THEORETICAL` objection, so no `DEFERRED.md` row.
- **Step 1 of the task (base check):** `data/landing_llsw_20261125/D_base_b2b76fb_check.txt`, sha256 `3df6d1fd43c183e76585bc78becbdf8bc66710a416b882493cc39652de9cb51c`. Nothing in the 13 files or their proof depends on main being `50cd683` rather than `b2b76fb`: the only difference is the closing note, which no suite reads; every landing path has the same blob at both commits or is absent at both; the pin `50cd6837abda…` is an ancestor of `b2b76fb`. By reading, not re-run (a post-landing re-run needs a commit, which the standing rule forbids).
- **Main tree after the relay:** `M docs/exchange/LEDGER.md`, `?? docs/exchange/from_beta/X-LL-SWITCHES_R1_BETA_REVIEW.md`, `?? docs/exchange/to_beta/X-LL-SWITCHES_R1_ALPHA_REVIEW_REQUEST_ll_switch_landing.md`. These are for the owner to land with the switches or on their own.

### Verdict

**One `LIVE` objection (Q2). By `AGENTS.md` §5 and `RELAY_PROTOCOL.md` §2.4 it holds the commit.** The task says fix nothing and no second round, so it goes to the owner as it stands. Q1 and Q4 `NO OBJECTION`; Q3 `SUGGESTION`; one prior-record finding.

### Positions, word for word

| Question | Position | Severity (Beta's words) | My view, one line |
|---|---|---|---|
| Q1 — Default unchanged | NO OBJECTION | — | Agreed. Beta also corrects my request: `prediction_generator.py:228` serializes the dry-run config, not the sidecar; Beta's reading stands. |
| Q2 — Switch on | OBJECTION | LIVE — A blank LL-3 group request can produce default columns under an applied label, and a misspelled exclusion can put two normally excluded columns into training. | Agreed, confirmed by reading: `models/feature_manifest.py:196` treats `''` as no group setting, and `--exclude-features` names are never checked against the data columns; the family-scope point is real too (`e2e_walk_forward.py:583` joins Step 5 and NN args, `e2e_pipeline_caps.py:148` keeps only the last value). The family-scope part sits in two files already on main, outside the 13. |
| Q3 — Tests after landing | SUGGESTION | — | Agreed and small: add the window-config hash and settings to `LL.runH_fixtures_equal_live`. It is test-only and inside this landing's files; not applied (task: fix nothing). |
| Q4 — Landed fixes | NO OBJECTION | — | Agreed. Direct optimizer runs still accept argparse abbreviations, as the closing note already records. |
| Finding 1 (prior record) | — | — | Accepted. My request left out `TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:159-169` (§7.3 Option A, F-PARITY), `TODO_CONSOLIDATED.md:62,159-161`, `CHAPTER_4_AUDIT.md:119-124,142-146`, `CHAPTER_7_AUDIT.md:282`, ch14-learn `READY_LOOP_LEARN_W1A_20261123.md`, and the two autonomy logs that §(b) of this note cites. My searches used LL-* names; those records use P0-B and F-PARITY. |

### Beta's Q2 sub-points, as written (no paraphrase of the position; see the reply file for the reasoning and anchors)

1. "**LL-3 accepts bad requests.**" — `--include-feature-groups ''` or `--exclude-feature-groups ','` gave the default 89 columns with no selection metadata, while the driver's capability function reported the blank include as applied; `--exclude-features confidnce` trained on 91 columns including `score` and `confidence`; listing all 91 names for exclusion gave a 40 × 0 matrix.
2. "**The LL-3 applied record loses family scope.**" — `--exclude-feature-groups dead` in Step 5 args plus `--exclude-feature-groups global` in NN args was recorded as `global` only, while LightGBM receives `dead` and the NN `global`.
3. "**Verified cases and recorded limits.**" — LL-2 and LL-1 behaved as specified in fixtures; the F-PARITY difference (18 computed values up to `0.000944` under `shared` plus `on`) is reported by the test without a tolerance, as recorded before.

### What this leaves for the owner

- **Q2, `LIVE`, holds the commit.** In-scope fixes would be: refuse an empty or blank LL-3 value at the parser or in `resolve_step5_exclusions`; refuse `--exclude-features` names that are not data columns; refuse an explicit exclusion that leaves no column; and record LL-3 per model family in `e2e_pipeline_caps.requested_ll3` / `e2e_walk_forward.pipeline_switches`. The last touches two files on main that are not in the 13. None applied.
- **Q3 suggestion:** not applied.
- **Label contest:** none. I do not contest `LIVE`: with the switches on main, a mistyped LL-3 value in a walk-forward run would train under a record that names a selection it did not apply.

Stopped. No commit, no push, no landing, no second round.
