# READY: registry repair, switch registrations, manifest entries (landing steps 1 to 3)

Task: `logs/e2e_session/TASK_REGISTRY_REPAIR_LAND_20261125.txt` (lead, 2026-11-25 17:55).
Owner ruling: `logs/e2e_session/OWNER_RULING_switch_registration_order_20261125.txt` (17:45). The order is registry repair, then switch registrations, then manifest entries; the grammar change comes later.
Worktree: `~/tfm_worktrees/registry-repair`, detached at main `7f4505a`.

Nothing is committed, pushed or landed. `agent_manifests/` is not edited anywhere except in the throwaway check tree, which has been removed. Nothing was written in `~/distributed_prng_analysis` (snapshot check below).

## After Beta R1 (2026-11-25, `TASK_REG_BETA_Q5_FIXES_20261125.txt`). This is the current state; the sections below are the first proof

**State:**
- **Part 1 (repair) is landed** as `579dbb7` (main = origin = public). `agents/registry/parameter_registry.py` there is `72d00315…`, byte-identical to this worktree's file (`cmp`).
- **Parts 2 and 3** went to Beta as X-REGISTRY-REPAIR R1. Request `docs/exchange/to_beta/X-REGISTRY-REPAIR_R1_ALPHA_REVIEW_REQUEST_registry_parts2_3.md` (`2e8a6857…`); review `docs/exchange/from_beta/X-REGISTRY-REPAIR_R1_BETA_REVIEW.md` (`760e356a…`). Both are uncommitted in the main tree.
- **Beta's positions:** Q1-Q4 NO OBJECTION, Q5 SUGGESTION. No `OBJECTION`, so nothing holds the commit. My summary is `docs/drafts/loop_v1/BETA_R1_REGISTRY_SUMMARY_20261125.md`.
- **The registered registry file is unchanged** (`bea4fcfe…`), and so are the manifest fragments (`2651c094…`) and the owner commands (`990fabb3…`). The Q5 work changed only part 2's checker and tests.

### Beta's three Q5 suggestions, adopted

| # | Beta's point | What changed | Gate | Mutants (killed and credited) |
|---|---|---|---|---|
| 1 | A name registered twice passes every gate: dicts collapse it, and `get_param` returns the first | **Tests:** new gate `A-LL-UNIQUE` in `tests/test_registry_ll_register.py`, which checks the source's own lists (ast) and the loaded Pydantic lists, never a dict. It also checks the total count is repaired + 7. **Checker:** `duplicate_specs()`, and every patch (R8 and R10) now passes through `check_patched()`, which refuses a patched registry that lists a name twice in one script. That includes a duplicate already in the base, which the old dict checks could not see. | `A-LL-UNIQUE`; `A-SCU-DUP`, `A-SCU-PATCHDUP` | LR10, LR11 (Beta's own probe: a duplicate `schema_check` after and before the real one); SM31, SM32 |
| 2 | `A-LL-MANIFEST` checked source strings, not behaviour | **New gate `A-LL-ARGV`:** it applies the exact owner commands to copies of the two manifests at the pinned base `579dbb7`, then builds the Step 5 and Step 6 argv through WATCHER's real `run_step`. It uses the stub boundary of `tests/fixtures/run4_routing_clean_control.py`; the builder is not stubbed. Asserts: the default argv is identical; one value per declared key is dropped before and reaches argv after; the command adds exactly the fragment's keys; once part 3 lands, `HEAD`'s manifest must equal the command's output. The string check is gone. `A-LL-MANIFEST`'s "not yet declared" check now reads the pinned base, so the test stays valid after landing. | `A-LL-ARGV` | LW1 (a None default emitted), LW2 (a False default emitted as a flag), LW3 (the step-scoped filter admits undeclared keys); all on a copy of `agents/watcher_agent.py`, loaded through `E2E_WATCHER_PATH` |
| 3 | `--register` turned a list-valued option into a scalar spec its own validator refuses | The scanner now keeps `nargs`. R10 (`select_register_rows()`, now a pure function) refuses an option with `nargs`, or with an action other than `store`/`store_true`/`store_false`, naming the reason. R8's old path is unchanged. | `A-SCU-NARGS`, `A-SCU-ACTION`, `A-SCU-SELECT` | SM33, SM34, SM35 (the scanner drops `nargs` again) |

**Mutant anchors moved by the refactor.** SM25, SM27 and SM30 (the `sc` target) are updated to the moved code. A static check confirms all 30 `sc` anchors apply exactly once. They were **not run** (see below).

### Suites: what ran, and what was held back

**Rule followed:** no test that runs `git commit`, in any repository, until the owner rules (lead, 19:25). I checked by reading. The only suite in this landing's set that commits is `tests/test_switch_check.py`: its fixture runs `git init`/`add`/`commit` at `:195-197`. The "commit" hits in `tests/test_s172_admission_binding.py` are miner trial commits, not git.

`nproc` = 2 here, so everything ran with `nice -n 15`, one process, `python3 -u -B`, in this worktree. Logs are in `logs/registry_repair_land/F_after_beta/`.

| Ran (no git commit) | Result |
|---|---|
| `tests/test_switch_check_units.py` (new, no repository at all) | **5/5** |
| `tests/test_registry_ll_register.py` | **9/9** (7 earlier + `A-LL-UNIQUE` + `A-LL-ARGV`) |
| `tests/test_registry_repair.py` | **5/5** on the draft, and 5/5 with `E2E_REPAIRED_REGISTRY` = the landed part-1 file |
| `test_subphase1.py` | ALL TESTS PASSED |
| `tests/mutate_e2e_grader.py --target scu` | **5/5** |
| `… --target lr` | **10/10** |
| `… --target lw` | **3/3** |
| `… --target rr` | **6/6** |
| `… --target cs` | **11/11** |
| `… --target grader` | **49/49**; all baseline gates pass in every target |
| `switch_check.py` regeneration (repair + `--register`, read-only git at `7f4505a`) | the repaired file, both patches, `repair_record.json`, both `SHA256SUMS` and the manifest fragments are **byte-identical** to before. Only the generated `.md` changes (timestamp and checker sha). |

| Held back at 19:25, now RUN (fixture root outside every tree, below) | Result |
|---|---|
| `tests/test_switch_check.py`, on the new checker `fe1ae65a…` | **15/15** (the 14 gates plus `test_sc_fixture_root`); fixture at `~/tfm_test_fixtures/switch_check/sc_run_20261125T195817/` |
| `tests/mutate_e2e_grader.py --target sc` (SM1-SM30, on the new checker) | **30/30** killed and credited, baselines pass; fixtures under `~/tfm_test_fixtures/switch_check/mutants_20261125T195829/` |
| `… --target sct` (ST1-ST3, the fixture-root refusal) | **3/3** |

**The part-2 proof on `579dbb7`** (`TASK_REG_FIXTURE_ROOT_20261125`):
- **The tree:** a fresh detached worktree `~/tfm_check_registry_repair_20261125c` with only the part-2 landing files copied in, plus the gitignored `daily3*.json` dataset files, this time **including `daily3.json`**. It is removed now.
- **How:** suites one at a time, `nice -n 15`, `python3 -u -B`. Logs in `logs/registry_repair_land/H_part2_579dbb7/`.

| Suite | Result | Against stage C |
|---|---|---|
| `test_subphase1`, `2`, `4` | pass (`test_subphase1`: "Registry has 5 parameters") | same |
| `test_subphase3`, `test_watcher_agent`, `agents/step_runner/test_step_runner` | red, as on clean main | normalised log diff **0 lines** |
| `test_watcher_followup_repairs` 17/17, `test_watcher_llm_integration` 13 passed, `test_watcher_safety_repairs` 21/21 | pass | same |
| `test_s172_admission_binding` | 11/20 | gate verdicts **identical** |
| `test_s172_run4_routing_patch` | 21/22, NOT-PASS `G-UNPINNED-IDENTICAL` | same |
| `test_s172_phase6_p05_dataset_authority` | **37/37** (stage C: 35/37) | the two earlier reds were my missing `daily3.json`; with it present they pass, which confirms `NOTE_REGISTRY_SUITES_FAILING_ON_MAIN_20261125.md` §6 |
| `test_seed_domain_cursor_amendment` 41/41, `test_step1_search_record_stage1` 39/39 (mutants 33/33) | pass | same |
| `test_registry_repair` 5/5, `test_switch_check` **15/15**, `test_switch_check_units` 5/5, `test_registry_ll_register` 9/9 | pass | new or extended gates |
| `tests/mutate_e2e_grader.py`, **all targets** (`E2E_LR_SOURCE` = the landed file) | **117/117** killed and credited, all baselines pass (grader 49, cs 11, sc 30, rr 6, lr 10, scu 5, lw 3, sct 3) | — |

**No fixture repository was created inside any worktree during the run.**
- `find <check tree> -name .git` (excluding its own) gave **0** before and **0** after.
- Every `sc`/`sct` fixture went to `~/tfm_test_fixtures/switch_check/mutants_20261125T200140/`.
- The only thing created under the check tree's `data/` by these tests was ST3's empty `refuse_probe_*` directory. That is the mutant doing what it should: with `main()`'s refusal disabled, the probe run makes its directory, and the gate catches it. The gate asks for no real gate, so no fixture and no commit can follow.
- The same holds in this worktree: 387 `.git` entries under `data/` before and after the `sc`/`sct` runs. Those 387 are the 129 old fixture repositories, each with 2 worktrees.

**What the checker refactor puts at risk, and what covers it.** The pure functions are covered by the unit suite. The real walk's refusals are now covered end to end again, by `test_switch_check.py` 15/15 and SM24-SM30 on `fe1ae65a…`.

### Open items recorded, no code change

1. **Beta R1 Q2 caution: a string `"false"` for an on/off option.** The registry's BOOL validator accepts the strings `'true'`/`'false'`/`'yes'`/`'1'`/`'0'`/`'no'` (`agents/registry/parameter_registry.py:134-137`). WATCHER's builder treats only a real `bool` as a flag. Any other value becomes `--flag value` (`agents/watcher_agent.py:2416-2427`). So `allow_target_columns_as_features: "false"` would reach the script as `--allow-target-columns-as-features false`, and argparse would refuse the extra word. That is a fail-closed stop.
   - The manifest command writes JSON `false`, so nothing is affected today.
   - **Open:** whatever later hands validated values to WATCHER (G7, the apply step) must turn BOOL values into real booleans first. Tightening `validate_value` instead is shared code and the owner's call.
2. **Beta R1 Q4 condition: `schema_check=fail` stays an operator choice.** After part 3, WATCHER can pass `--schema-check fail` to Step 6 when someone sets it. It stops Step 6, fail-closed, on a model whose sidecar has no `feature_schema`, as it does for the unlearned arm A0 (`READY_LAND_LL_SWITCHES_20261125.md` §(c)).
   - **Open:** `fail` must not become a default, or a requirement of a walk-forward comparison, until it has been run against the intended arms, A0 included.
3. **Fixture commits: ruled 19:26; the suite is now compliant (`TASK_REG_FIXTURE_ROOT_20261125`, lead's GO 20:00).** The ruling is "test may run `git commit` inside a temporary repository that the test itself creates outside the TFM tree" (`logs/e2e_session/OWNER_RULING_fixture_commits_20261125.txt`); the lead adds "and outside every copy or worktree of it".

   **What was wrong.** Before this change, `tests/test_switch_check.py` created its fixtures inside the tree that runs it:
   - the default root was `REPO/data/e2e_grader_selftest/sc_run_<stamp>`;
   - the mutant runner passed `OUT/<id>/work`, also under `REPO/data/`.
   Every run today until 19:25 therefore committed in repositories nested inside a TFM worktree.

   **What changed** (part 2 test files only):
   - **The default fixture root** is `~/tfm_test_fixtures/switch_check/sc_run_<stamp>/`, overridable by `E2E_SELFTEST_DIR` (`tests/test_switch_check.py`, `FIXTURE_BASE`, `WORK`). `~` is not inside any git work tree (checked: `git -C ~ rev-parse` fails).
   - **`check_fixture_root()`** refuses a root under `/tmp`, `/var/tmp` or `/dev/shm` (box rule 7), and a root inside any git work tree. It judges at the nearest existing ancestor with `git rev-parse --show-toplevel`, so it catches the TFM tree, a worktree, a clone, or any other repository. `main()` calls it first and exits 2 with "REFUSED (nothing created)", naming the path and the tree, before anything is created.
   - **New gate `test_sc_fixture_root` (`A-SC-ROOT`):**
     - it refuses three inside-tree paths: this tree's `data/`, beside the test file, and the TFM main tree's `data/`;
     - it refuses the three tmp roots;
     - it accepts the default base and this run's root;
     - it runs the test itself as a subprocess with an inside-tree root and asks for a gate that does not exist, so no fixture can ever be built. It asserts exit 2, "REFUSED", and that nothing was created.
   - **`tests/mutate_e2e_grader.py`:**
     - for the targets whose test commits (`FIXTURE_COMMIT_TARGETS = ("sc", "sct")`) the work dir moves under `~/tfm_test_fixtures/switch_check/<OUT name>/…` (`SC_FIXTURE_ROOT`, overridable by `E2E_SC_FIXTURE_ROOT`);
     - reports and markers stay in `OUT`;
     - the 19:25 guard and `E2E_ALLOW_FIXTURE_COMMIT` **are removed**;
     - a new target `sct` mutates the test file itself, and runs the mutated copy against the real checker.
   - **Mutants:** ST1 (a work-tree root accepted), ST2 (a `/tmp` root accepted), ST3 (`main()` starts with an inside-tree root). All 3 are killed and credited.

   **The 129 old fixture repositories are NOT deleted**, as the task says. They are all in this worktree, from today's runs before 19:25 (`logs/registry_repair_land/G_fixture_root/old_fixture_parents.txt` lists every parent path):

   | Parent directory (under `~/tfm_worktrees/registry-repair/data/e2e_grader_selftest/`) | Repositories | Size of the fixture directories |
   |---|---|---|
   | `mutants_20261125T180347/` | 42 | 38 MB |
   | `mutants_20261125T180412/` | 43 | 39 MB |
   | `mutants_20261125T181910/` | 43 | 39 MB |
   | `sc_run_20261125T180057/` | 1 | 1.1 MB |
   | **Total** | **129** | **≈ 117 MB** (`du -csh` of the 129 `fixture_*` directories: 116M) |

   The whole `data/e2e_grader_selftest/` directory is 246 MB on disk. The rest is mutant copies, logs and reports from the non-committing targets. The fixtures made in the two removed check trees went with them.

   **The new outside root** `~/tfm_test_fixtures/` holds 80 MB from tonight's compliant runs. It is not cleaned automatically; that is for the owner.
4. **Beta R1 Finding 1: the prior-record gap.** These are added here for the record:
   - `logs/e2e_session/TASK_SWITCH_REGISTER_ADDENDUM_1.txt:1-4`;
   - `docs/CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:33-45, 54-57` (the Step 5 reporting boundary; none of its six allowlisted names is declared by part 3);
   - `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md` B3 (`:133-139`, `:158-163`).

**Also land with part 2** (docs; sha256 as they are now):

| File | sha256 |
|---|---|
| `docs/drafts/loop_v1/NOTE_REGISTRY_SUITES_FAILING_ON_MAIN_20261125.md` | `2469f9c7…` |
| `docs/drafts/loop_v1/DESIGN_LL2_WATCHER_ROUTE_20261125.md` | `c6d276db…` |
| `docs/drafts/loop_v1/DRAFT_X-REGISTRY-REPAIR_R1_REQUEST.md` | `2e8a6857…`, the same bytes as the sent request |
| `docs/drafts/loop_v1/BETA_R1_REGISTRY_SUMMARY_20261125.md` | `4508b8b0…` |

The landing list further down carries the final sha256 of every part-2 file.

## Prior record

**Searches run** (worktree at `7f4505a`; `/bin/grep`, because the shell `grep` skips `.json`):

```
/bin/grep -rlE 'parameter_registry|PARAMETER_REGISTRIES|ParameterSpec|registry repair|REGISTRY_REPAIR|switch registration|schema[-_]check|per[-_]seed[-_]sieve[-_]features|fallback[-_]metadata[-_]merge|include[-_]feature[-_]groups|exclude[-_]feature[-_]groups|allow[-_]target[-_]columns' \
   docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md \
   docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange
/bin/grep -rlE 'registry|ParameterSpec|switch regist|manifest entr' ~/distributed_prng_analysis/logs/e2e_session/
git log --all --oneline -- scorer_meta_optimizer.py full_scoring.py          # empty
git diff --stat 41ad509 7f4505a -- <the six registry scripts, wrappers, watcher_agent.py, agent_manifests/>
```

Hits per scope:

| Scope | Hits |
|---|---|
| `PROJECT_FILE_CATALOG`, `TB_RULING_*`, `TB_RULING_REQUEST_*`, `TODO_CONSOLIDATED`, `BACKLOG`, `LEADS` | 0 |
| `PROPOSAL_*` | 7: Unified Agent Context v3.2.x (the registry's design) and Window Anchor v1.0/v1.1 |
| `CHAPTER_*` and `*_AUDIT.md` | 8 and 4 |
| `docs/exchange/` | 4 (X-LL-SWITCHES R1 both ways, X-E2E-WF-VOTE R1 both ways) |

**Read in full, as the task asks:**
- `OWNER_RULINGS_pending_list_20261124.txt` ruling 7;
- `OWNER_DIRECTION_autonomy_switches_20261123.txt`;
- `TASK_SWITCH_REGISTER_ADDENDUM_1.txt`;
- `docs/drafts/loop_v1/REGISTRY_REPAIR_20261124.md`;
- `AUTONOMY_SWITCH_MAP_20261123.md` (§4 G5 to G8);
- `SWITCH_CHECK_20261123.md`;
- the folders `registry_repair_20261124/` and `switch_patches_20261123/`.

**Documents that already decide or record this subject:**

| Document | What it records |
|---|---|
| `CHAPTER_10_AUDIT.md` B1 (`:117-140`) | the 20 of 29 stale specs, and that `scorer_meta_optimizer.py` and `full_scoring.py` never existed |
| `CHAPTER_10_AUDIT.md` S1 (`:145-155`) | `offset` is not yet `window_anchor` |
| `CHAPTER_CORRECTIONS_REPORT_20261111.md:213` | R-30, the same B1 |
| `OWNER_RULINGS_pending_list_20261124.txt` | 10:10: "#7 yes fix those but document everything" |
| `OWNER_RULING_switch_registration_order_20261125.txt` | the order of this landing |
| `REGISTRY_REPAIR_20261124.md` | the 20-row repair table, the 9 kept specs and where they disagree with the code, and the 46/41 count of result-changing options |
| `READY_LAND_LL_SWITCHES_20261125.md` §(b) `:536-560` | the registry is not governed or hand-landed; "registry repair, then regenerated registration"; LL-2 has no WATCHER route |
| `READY_LAND_LL_SWITCHES_20261125.md` §(c) | LL-1 `fail` stops the walk-forward at the unlearned arm A0 |
| `X-LL-SWITCHES_R1_ALPHA_REVIEW_REQUEST_ll_switch_landing.md:40, :108` | the switches are not in the registry or manifests |
| `AUTONOMY_SWITCH_MAP_20261123.md` G5, G6, G7 | the grammar value type; no reach route for the drivers; no apply step |
| `TASK_STEP2_WIRING_20261124.txt:6` | the step2-repair agent will draft its own `ParameterSpec` into this same file |

**Labels for what this note raises:**

| Item | Label |
|---|---|
| The repair (20 dropped, 0 corrected) | **ALREADY RECORDED** (B1; ruling #7; `REGISTRY_REPAIR_20261124.md`). Re-checked here against `7f4505a`. |
| `offset` → `window_anchor` not done | **ALREADY RECORDED** (S1). Brief II landed with no `--window-anchor` flag. |
| The 9 kept specs disagree with the code (`mode`, `prng_type`, `strategy`, `pool_size`) | **ALREADY RECORDED** (`REGISTRY_REPAIR_20261124.md`, "The 9 kept specs"; `CHAPTER_5_AUDIT:272`; `CHAPTER_7_AUDIT:105`). Not changed here. |
| Registering LL-1/2/3, and LL-2 having no route | **ALREADY RECORDED** as the plan (`READY_LAND_LL_SWITCHES` §(b)). Done here as drafts. |
| `allow_target_columns_as_features` (the seventh LL option, added after the 11-23 draft) | **NEW** to the registration drafts |
| `switch_check.py` R10 (`--register`, built on the repaired registry) | **NEW** tool feature |
| The measured argv probe for Steps 5 and 6 | **NEW** evidence |
| Count of the other unregistered step-script options at `7f4505a` (161; 36 change results) | **NEW** count; it updates the 11-24 count of 174 |
| Recommendation of 5 options to register next | **NEW** (recommendation only) |
| WATCHER builds `--k 20 --k 20` for Step 6 (from `k` and from `pool_size`→`--k`) | **ALREADY RECORDED** in substance (`CHAPTER_7_AUDIT:105`, `pool_size` sent as `--k`). Seen in the probe, not changed. |
| Changes a ruling | **None.** Chapter 14 S1 (`--dropout` not registered) is untouched; LLM-visible parameters are unchanged. |

**Not searched:** the host, ser8 archives, the private repository, project files that are not in the repository.

## Plain summary

- **Step 1, registry repair.**
  - The 2026-11-24 draft still holds on main `7f4505a`. Between `41ad509` and `7f4505a`, the registry file and the checker did not change.
  - Each dropped flag is absent from its script at `7f4505a`, re-checked line by line.
  - Brief II added `--derived-anchor-domain` and `--anchor-holdout-guard`, but **no `--window-anchor`**, so `offset` stays dropped and is not renamed.
  - **Result: 20 entries dropped, 0 corrected.** The repaired file is **byte-identical** to the 11-24 draft (sha256 `72d00315…`).
  - It is in place as `agents/registry/parameter_registry.py` in this worktree. It goes from 29 specs to 9; the 9 kept specs are unchanged field by field.
- **Step 2, registrations.**
  - The 7 LL options are registered **on top of the repaired file**: LL-1 `schema_check`; LL-2 `per_seed_sieve_features` and `fallback_metadata_merge`; LL-3 `exclude_features`, `include_feature_groups`, `exclude_feature_groups` and `allow_target_columns_as_features`.
  - Each spec is generated by `switch_check.py` from its argparse call on main, using a new rule R10 (`--register`). Nothing was typed by hand.
  - The registration only adds lines: 9 specs become 16. `validate_value` accepts every legal choice and refuses the rest. That is shown by gates and by 8/8 mutants.
  - **WATCHER can reach none of the 7 today**, as measured with the real `run_step` (table below). Note the denominators:
    - after the manifest entries, **5 of 7** can be reached (LL-1, and the four LL-3 options);
    - **LL-2 (2 of 7) still has no route.**
  - **Other options:** **161** other step-script options are unregistered. **36** of them change results. I recommend registering **5** next and leaving the rest (reasons below). None of them is registered here.
- **Step 3, manifests.**
  - Two fragments, `prediction.json` (1 key) and `reinforcement.json` (4 keys), each with one self-checking command for the owner.
  - **At default they change nothing in what WATCHER runs.** I measured it: the Step 5 and Step 6 argv are identical before and after, because WATCHER drops None and False values.
  - What they change: a value supplied in `params` now reaches the script, where before it was dropped.
- **Proof.**
  - Every suite that imports the registry was run four times on a fresh detached worktree of `7f4505a`: clean, part 1, part 2, and part 2 plus the manifests. 17 suites, one at a time, under `taskset -c 4-7 nice -n 10`, `python3 -u -B`. That tree is removed.
  - **Every pass/fail verdict is the same as on clean main.** The one expected difference is `test_subphase1`'s "Registry has 5 parameters" (it was 9).
  - **3 suites already fail on clean main** and fail identically after: `test_subphase3`, `test_watcher_agent`, `agents/step_runner/test_step_runner`.
  - **3 suites have environmental reds on clean main** (admission: local GPU count; routing: frozen-fixture stub DB; P0.5: `daily3.json` missing from my check tree, corrected 2026-11-25), also identical after.
  - Mutants: **104/104** killed and credited across the whole runner; R10 30/30, LL 8/8, repair 6/6 on the landed file.

## Step 1: OWNER SIGN-OFF LIST, registry repair

Registry lines are `7f4505a:agents/registry/parameter_registry.py`; the file has not changed since `41ad509`.

**What "absent" was checked against.** "Absent" means `git grep -F -- "'--flag'"` at `7f4505a` returns 0 hits in the named script, and the flag is not in the script's argparse block. The argparse blocks are:

| Script | argparse block | `add_argument` calls |
|---|---|---|
| `window_optimizer.py` | `:1582-1822` | 55 |
| `adaptive_meta_optimizer.py` | `:1101-1114` | 6 (`--config --mode --window-results --lottery-data --apply --test`) |
| `meta_prediction_optimizer_anti_overfit.py` | `:3071-3146` | 27 |
| `prediction_generator.py` | `:984-1008` | 14 |

**DROPPED (20).** Each entry is removed from the registry list only. Each leaves a dated comment, and no script changes.

1. **`window_size`** (`window_optimizer.py`, registry `:273`). **Why:** no `--window-size` flag. Window size is an Optuna search range: `distributed_config.json:73`, read by `load_search_bounds_from_config` at `window_optimizer.py:67`. The only CLI route is the operator-only trial-0 pin `--warm-start-window` (`:1690`; `STEP1_EXPLICIT_PIN_KEYS`, `agents/watcher_agent.py:511`). It is also frozen (`chapter_13_acceptance.py:72, :76`).
2. **`offset`** (`window_optimizer.py`, `:282`). **Why:** no `--offset` and no `--window-anchor` flag at `7f4505a`. Offset is an Optuna range, with the pin `--warm-start-offset` at `:1692`. It is not renamed to `window_anchor`, because no such flag exists (S1).
3. **`skip_min`** (`window_optimizer.py`, `:291`). **Why:** no `--skip-min`. It is an Optuna range (`distributed_config.json:84`), with the pin `--warm-start-skip-min` at `:1694`.
4. **`skip_max`** (`window_optimizer.py`, `:300`). **Why:** no `--skip-max`. It is an Optuna range (`distributed_config.json:88`), with the pin `--warm-start-skip-max` at `:1696`.
5. **`n_trials`** (`scorer_meta_optimizer.py`, `:363`). **Why:** the script never existed (`git log --all` is empty). The live equivalent is positional `TRIALS` in `run_scorer_meta_optimizer.sh`, which passes `generate_scorer_jobs.py --trials` (`:76`). Registering it there is a new registration, not a correction.
6. **`threshold_min`** (`scorer_meta_optimizer.py`, `:373`). **Why:** the script never existed. The range is hard-coded at `generate_scorer_jobs.py:60` (`min_confidence_threshold` 0.05-0.25).
7. **`threshold_max`** (`scorer_meta_optimizer.py`, `:382`). **Why:** the script never existed; same range as row 6, `generate_scorer_jobs.py:60`.
8. **`k_folds`** (`scorer_meta_optimizer.py`, `:391`). **Why:** the script never existed, and Step 2 has no fold option at all.
9. **`distributed`** (`scorer_meta_optimizer.py`, `:399`). **Why:** the script never existed. Step 2 is always distributed (`run_scorer_meta_optimizer.sh:266` runs `scripts_coordinator.py`).
10. **`batch_size`** (`full_scoring.py`, `:416`). **Why:** the script never existed. `full_scoring_worker.py:557 --batch-size` means progress-report batch size, default 100, which is a different meaning.
11. **`distributed`** (`full_scoring.py`, `:425`). **Why:** the script never existed. Step 3 is always distributed (`run_step3_full_scoring.sh` runs `scripts_coordinator.py`).
12. **`trials`** (`adaptive_meta_optimizer.py`, `:442`). **Why:** no `--trials` (6 calls, listed above).
13. **`min_layers`** (`adaptive_meta_optimizer.py`, `:452`). **Why:** no `--min-layers`.
14. **`max_layers`** (`adaptive_meta_optimizer.py`, `:461`). **Why:** no `--max-layers`.
15. **`epochs`** (`meta_prediction_optimizer_anti_overfit.py`, `:507`). **Why:** no `--epochs`. It is a trial parameter (`params.get('epochs', 100)`, `:912`).
16. **`dropout_min`** (`meta_prediction_optimizer_anti_overfit.py`, `:517`). **Why:** no range flag. `--dropout` (`:3120`) is a single override, not a bound. It is not registered here (Chapter 14 S1).
17. **`dropout_max`** (`meta_prediction_optimizer_anti_overfit.py`, `:526`). **Why:** as row 16.
18. **`distributed`** (`meta_prediction_optimizer_anti_overfit.py`, `:535`). **Why:** no `--distributed`.
19. **`confidence_threshold`** (`prediction_generator.py`, `:561`). **Why:** no flag. It is a config-file key: `prediction_generator_config.json:9`, read at `prediction_generator.py:173`.
20. **`diversity_weight`** (`prediction_generator.py`, `:570`). **Why:** no flag and no config key in any script. Only `agents/contexts/prediction_context.py:144-171` reads it and suggests changing it. That file is left as it is.

Two whole registries go, because every spec in them was stale: `scorer_meta_optimizer.py` (rows 5-9) and `full_scoring.py` (rows 10-11). Each leaves one comment line, at `:326` and `:331` of the repaired file.

**CORRECTED: none.** No stale entry has a flag with the same meaning in the script it names. The checker implements only DROPPED and refuses a CORRECTED decision.

**Kept unchanged (9), for the owner's information.** Ruling #7 covered the 20 stale entries. These 9 name flags that exist, so they are kept byte for byte. Where a kept spec disagrees with the code, the owner may want a separate correction. This is not done here.

| Kept spec | Disagreement with the code at `7f4505a` |
|---|---|
| `adaptive_meta_optimizer.py` `mode` | registry choices `quick/full/deep`; code `full/survivor/network/epochs` (`:1104-1106`). So `validate_value('quick')` passes, and the script would refuse it. |
| `window_optimizer.py` `prng_type` | registry lists 6 choices, including `lcg64`; the code takes any `prng_registry` type (`:1613`) |
| `window_optimizer.py` `strategy` | registry default `bayesian`, code default `None` (`:1582`) |
| `prediction_generator.py` `pool_size` | registry default 200, code has no default (`:1003`); WATCHER sends the manifest's `pool_size` as `--k` |
| the other 5 (`trials`, `max_seeds`, `test_both_modes`, Step 5 `trials`, `k_folds`) | agree with the code |

**Consumers.** `ParameterContext.build` handles a missing registry (`agents/parameters/parameter_context.py:181`). `get_registry_for_agent` still maps the Step 2 and Step 3 agent names to the removed keys, so those lookups now return `None`. That is unchanged code outside the dict; see Questions.

## Step 2: registrations

### The 7 LL options

Each spec was generated from the code on main at `7f4505a`. The table gives the code's type, choices and default, the registry line in the registered file, and whether WATCHER can reach the option, today and after the manifest commands.

| LL | Option (code line) | Type, choices, default (from the code) | Registry line | WATCHER can reach it **today** | After the manifest commands | LLM-visible |
|---|---|---|---|---|---|---|
| LL-1 | `prediction_generator.py:999 --schema-check` | choice `off/warn/fail`, default `None` ("the config's value, else off") | `:435` | **no**: not in `prediction.json` `default_params` (the step-scoped filter drops it, `watcher_agent.py:2029-2034`); measured: `schema_check=warn` dropped | **yes**: measured, `--schema-check warn` reaches argv | no |
| LL-2 | `full_scoring_worker.py:570 --per-seed-sieve-features` | choice `off/on`, default `off` | `:449` (new registry `full_scoring_worker.py`, `:444`) | **no**: there is no route (below) | **no** | no |
| LL-2 | `full_scoring_worker.py:573 --fallback-metadata-merge` | choice `legacy6/shared`, default `legacy6` | `:458` | **no** | **no** | no |
| LL-3 | `meta_prediction_optimizer_anti_overfit.py:3134 --exclude-features` | string, default `None` | `:385` | **no**: not in `reinforcement.json` | **yes** | no |
| LL-3 | `:3137 --include-feature-groups` | string, default `None` | `:392` | **no** | **yes** | no |
| LL-3 | `:3140 --exclude-feature-groups` | string, default `None` | `:399` | **no**: measured, `dead` dropped | **yes**: measured, `--exclude-feature-groups dead` | no |
| LL-3 | `:3143 --allow-target-columns-as-features` | bool, default `False` | `:406` | **no**: measured, dropped | **yes**: measured, `--allow-target-columns-as-features` | no |

**LL-2 has no WATCHER route.**
- Under WATCHER, Step 3 runs `full_scoring_worker.py` from the job file that `generate_step3_scoring_jobs.py` writes, through `run_step3_full_scoring.sh` and `scripts_coordinator.py`.
- Neither the wrapper's options (`:88-116`) nor the job generator carries either LL-2 flag. The wrapper drops unknown options with "Note: Ignoring unknown option" (`:120-123`; corrected 2026-11-25, was `:34-37`). The same is recorded in `READY_LAND_LL_SWITCHES` §(b).
- So LL-2 is now **known** (registered), but it is reached only by the direct worker CLI and the e2e Step 3 harness.

**LLM-visible: none of the 7, before or after.**
- `ParameterContext.build` reads only the primary action's `args_map` (`parameter_context.py:177-186`). The fragments add no `args_map` entry.
- So the LLM's parameter context is unchanged, and the grammar change (G5) stays for later, as ruled.

**What the registry can and cannot check.**
- `validate_value` checks **choice** and **bool** values, and those gates pass.
- It cannot check the **string** LL-3 lists: `ParamType.STRING` has no rule in `validate_value` (`parameter_registry.py:88-139`), so any string is accepted.
- The script validates those lists itself (LL-3 "settings with validation"; `--allow-target-columns-as-features` refuses target columns unless set).

**Caveat on LL-1.** WATCHER could now pass `schema_check=fail`. `READY_LAND_LL_SWITCHES` §(c) records that `fail` stops a walk-forward at the unlearned arm A0, which has no sidecar. Under WATCHER, `fail` would likewise stop Step 6 on any model without a `feature_schema`. Nothing sets it today.

**`validate_value` results** (`tests/test_registry_ll_register.py`, on the landed file):

| Option | Accepts | Refuses |
|---|---|---|
| `schema_check` | `off`, `warn`, `fail`, `None` | `on`, `FAIL`, `''`, `True`, `1` |
| `per_seed_sieve_features` | `off`, `on`, `None` | `yes`, `ON`, `True`, `1`, `''` |
| `fallback_metadata_merge` | `legacy6`, `shared`, `None` | `legacy`, `batch`, `''`, `True` |
| `allow_target_columns_as_features` | `True`, `False`, `'true'`, `'false'`, `None` | `maybe`, `2`, `on` |
| the three LL-3 strings | any string, `None` | (nothing; see above) |

`validate_adjustments` gives exactly one error for each refused value.

### The other unregistered step-script options at `7f4505a`

**161** step-script option rows are unregistered, apart from the 7 above.
- Source: `data/registry_repair_land/switch_check_7f4505a.json`, 168 not-registered step-script rows minus the 7 LL rows.
- Classes are from `REGISTRY_REPAIR_OPTION_CLASSES_20261124.json`; 6 rows new since 11-24 were classed by reading them.
- Full list: `data/registry_repair_land/unregistered_step_options_7f4505a.txt`.

| Class | Rows | What it covers |
|---|---|---|
| Changes results | **36** | |
| Paths, inputs, outputs | 55 | |
| Infrastructure | 32 | transport, workers, sizing, device |
| Not run by the step | 15 | |
| Logging, test, dry-run | 12 | |
| ids and metadata | 9 | |
| Guards | 2 | `--forward-threshold` and `--reverse-threshold`, which abort when passed |

**The 36 that change results, by step:**

| Step | Options |
|---|---|
| Step 0 | `--k-clusters`, `--recommended-window-size` |
| Step 1 | `--config-file`, `--iterations`, `--resume-study`, `--enable-pruning`, `--seed-start`, the 7 `--warm-start-*` pins, `--deliver-skip-range`, `--derived-anchor-domain`, `--anchor-holdout-guard`, `--step1-objective` |
| Step 2 | `generate_scorer_jobs.py` `--trials`, `--legacy-scoring`, `--sample-size`; the wrapper's `--sample-size` and `--legacy-scoring` |
| Step 3 | `--prng-type` and `--mod`, in the worker and in the job generator |
| Step 4 | `--apply` |
| Step 5 | `--test-holdout`, `--model-type`, `--compare-models`, `--normalize-features`, `--use-leaky-relu`, `--dropout` |
| Step 6 | `--no-require-schema`, `--k` |

**Recommendation: register these 5 next.** They are not registered in this landing. Each is reachable through WATCHER today, is a bool or a fixed choice (so `validate_value` means something), and has no authority issue:
- `window_optimizer.py --derived-anchor-domain` and `--anchor-holdout-guard` (Brief II, declared in `window_optimizer.json`);
- `window_optimizer.py --enable-pruning`;
- `meta_prediction_optimizer_anti_overfit.py --model-type` (choices `neural_net/xgboost/lightgbm/catboost`, `:3087-3089`) and `--compare-models`.

The manifest declares `enable_pruning: true` and `compare_models: true`, while the code defaults are False. The spec would carry the code default, as R7 does. One `--register` run adds all 5.

**Leave the other 31 result-changing options, and say why:**

| Options | Why leave them |
|---|---|
| `--seed-start`, `--resume-study` | WATCHER sets them itself from the open Step 1 search and the study (`watcher_agent.py:2306-2318`). They must not be proposable. |
| the 7 warm-start pins | operator-only by design (`STEP1_EXPLICIT_PIN_KEYS`, `:511`) |
| `--deliver-skip-range` | needs a Step 1 search id; an operator run choice |
| `--config-file`, `--iterations` | a different procedure (config mode) |
| `--step1-objective` | not reachable; register it with its manifest entry when the cheap-screen route lands |
| Step 2 (5 options) | the wrapper is positional; wait for the Step 2 wiring task (`TASK_STEP2_WIRING_20261124`) |
| Step 3 `--prng-type`, `--mod` | they fix the run's identity; they are not tuning knobs |
| Step 4 `--apply` | writes a config that later steps read; owner-only |
| `--test-holdout` | moves the evaluation split |
| `--normalize-features`, `--use-leaky-relu`, `--dropout` | the Chapter 14 S1 boundary |
| `--no-require-schema` | zero-fills features; LL-1 is the governed route |
| `--k` | tied up with the `pool_size`→`--k` mapping (`CHAPTER_7_AUDIT:105`) |
| Step 0 ints | no bounds source, so an INT spec would accept any integer; give bounds first |

The other 125 rows (paths, infrastructure, logging, ids, not run, guards) should stay unregistered. They would add noise to the list an LLM reads and change no result.

## Step 3: manifests (owner hand-landed; NOT edited in this worktree)

**The exact fragments** (`docs/drafts/loop_v1/registry_repair_land_20261125/register/manifest_fragments.json`, generated):

```
agent_manifests/prediction.json     default_params += {"schema_check": null}
agent_manifests/reinforcement.json  default_params += {"exclude_features": null, "include_feature_groups": null,
                                                       "exclude_feature_groups": null, "allow_target_columns_as_features": false}
```

No `args_map` change. No fragment for `full_scoring.json`, because LL-2 has no route.

**One command per file.** Run each from the repository root, after parts 1 and 2. The same text is in `OWNER_MANIFEST_COMMANDS.txt`. Each command refuses, leaving the file untouched, unless the file still round-trips byte for byte with its own formatting and does not already declare the keys.

```
python3 -B -c "import json; p='agent_manifests/prediction.json'; s=open(p).read(); d=json.loads(s); assert json.dumps(d, indent=4) == s, 'format differs: stop'; dp=d['default_params']; assert 'schema_check' not in dp, 'already declared'; dp['schema_check']=None; open(p,'w').write(json.dumps(d, indent=4)); print('ok', p)"

python3 -B -c "import json; p='agent_manifests/reinforcement.json'; s=open(p).read(); d=json.loads(s); assert json.dumps(d, indent=2) == s, 'format differs: stop'; dp=d['default_params']; new={'exclude_features': None, 'include_feature_groups': None, 'exclude_feature_groups': None, 'allow_target_columns_as_features': False}; assert not set(new) & set(dp), 'already declared'; dp.update(new); open(p,'w').write(json.dumps(d, indent=2)); print('ok', p)"
```

I tested both commands in the check tree:
- `ok` both times;
- run a second time, each refuses with `already declared`;
- `git diff` shows only the added keys (`logs/registry_repair_land/D_manifests/`).

Expected sha256 after each command, starting from `7f4505a`:

| File | sha256 |
|---|---|
| `prediction.json` | `fb76fce730d47cd72c667c4d7d5dd96fc4db7093aaa1f26426918d03c409c641` |
| `reinforcement.json` | `7536fcb7d33bf18f08c9cbf77873cd831a580404fb9caed1a0b807bd05869a18` |

**What each changes in WATCHER's behaviour at default: nothing.**
- **Measured.** `data/registry_repair_land/argv_probe.py` drives the real `WatcherAgent.run_step` for Steps 5 and 6. It uses the stub boundary of `tests/fixtures/run4_routing_clean_control.py`; the command builder is not stubbed. It captures the argv at dispatch:
  - with no params, the Step 5 and Step 6 argv are **identical** before and after (`argv_before.json`, `argv_after.json`);
  - the reason: the builder skips None and `''`, and omits False (`watcher_agent.py:2416-2427`). `final_params` is not stored anywhere else; it only feeds argv.
- **Change 1: a value becomes reachable.** Before the commands, `params` carrying one of these keys is dropped by the step-scoped filter. After, it reaches argv, and only when set to a non-default value.
- **Change 2: the Step 5 retry report.** That report reads the same key set (`watcher_agent.py:2739-2766`). The LLM retry whitelist (`_is_within_policy_bounds`, 6 names) includes none of these keys, so no LLM proposal can set them.

## Proof

All on fresh detached worktrees of `7f4505a` with only the landing files copied in. Both trees are removed (`git worktree list` no longer shows them).
- The gitignored dataset files `daily3*.json` were copied in from the main tree (read only there), so the dataset-pointer suites could run.
- 17 suites, one at a time: `taskset -c 4-7 nice -n 10 python3 -u -B`, `PYTHONDONTWRITEBYTECODE=1`, foreground.
- Logs: `logs/registry_repair_land/<stage>/`, each with a `SUMMARY.txt`.

| Stage | Tree | Result |
|---|---|---|
| A, A2 | clean `7f4505a` (A2 with the dataset files) | baseline |
| B | + part 1 (repaired registry) | every verdict as baseline; `test_subphase1` "Registry has 5 parameters" (was 9); `test_registry_repair` 5/5, also with `E2E_REPAIRED_REGISTRY` = the landed file |
| C | + part 2 (registered registry, checker, tests) | every verdict as baseline; `test_switch_check` 14/14 (12 + 2 new R10 gates); `test_registry_ll_register` 7/7 on the draft and on the landed file; mutants R10/SM 30/30, LL 8/8, RR 6/6 |
| D | + the two manifest commands | every verdict as baseline; argv probe as above |
| E | second fresh tree, final checker `7a9b35f0…` | `test_switch_check` 14/14, `test_registry_ll_register` 7/7 (landed file), `test_registry_repair` 5/5, `test_subphase1` pass; mutants sc 30/30, lr 8/8, rr 6/6 |
| worktree | all mutant targets | **104/104** killed and credited; all baseline gates pass (`logs/registry_repair_land/mutants_all_wt_final.log`) |

**Why stage E.** After stages A to D, I changed only `render_md` in the checker: its summary line now names R10 when `--register` is used. Every generated artifact is byte-identical before and after that change. Stage E re-ran the checker's consumers. No other suite imports the checker.

**Suites that already fail on clean main, identical in every stage** (normalised log diff: 0 lines, or only set order and run ids):

| Suite | Failure |
|---|---|
| `test_subphase3` | `test_retry_suggestions` assert |
| `test_watcher_agent` | `test_heuristic_evaluation` confidence |
| `agents/step_runner/test_step_runner` | 3 of 4 |
| `test_s172_admission_binding` | 11/20; fleet clamp cases B1, B2, B5, C3 |
| `test_s172_phase6_p05_dataset_authority` | 35/37; Gates 24 and 31: `daily3.json` was missing from my check tree (corrected 2026-11-25; see `NOTE_REGISTRY_SUITES_FAILING_ON_MAIN_20261125.md` §6) |
| `test_s172_run4_routing_patch` | 21/22; `G-UNPINNED-IDENTICAL` INCOMPLETE, frozen fixture stub DB (corrected 2026-11-25; the `G-ALLOWLIST-EXACT` lines are mutant runs; same note, §5) |

**Passing in every stage:**

| Suite | Result |
|---|---|
| `test_subphase1`, `2`, `4` | pass |
| `test_watcher_followup_repairs` | 17/17 |
| `test_watcher_llm_integration` | 13 passed |
| `test_watcher_safety_repairs` | 21/21 |
| `test_seed_domain_cursor_amendment` | 41/41 |
| `test_step1_search_record_stage1` | 39/39, mutants 33/33 |

**Not run, and why.** `tests/test_s172_phase4_coordinator.py` Gate 22 reds on any modified or untracked `.py` before a commit (memory note; `:1626`), and the suite does not import the registry. It is for the owner's post-commit check.

**Every default unchanged:**
- the 9 kept specs are equal field by field (`A-RR-KEPT`, `A-LL-REPAIR`);
- each LL default equals its argparse default (`A-LL-CODE`);
- each manifest value equals the code default and is a value the builder omits (`A-LL-MANIFEST`);
- the default argv for Steps 5 and 6 is identical (probe).

**Live tree untouched.** The before and after snapshots of `~/distributed_prng_analysis` (`logs/registry_repair_land/live_tree_snapshot_*.txt`) differ only by two files the lead wrote at 18:05 (`logs/e2e_session/TASK_ITEM4_STAGING_20261125.txt`, `TASK_ITEM5_SURVEY_REST_20261125.txt`). `optuna_studies/` is unchanged, and `/tmp/agent_halt` is absent both times. No `__pycache__` was written in this worktree.

## Landing list (owner; agents never commit)

Three landings, in this order (ruling 17:45). **JSON files are gitignored (`.gitignore:41 *.json`) and need `git add -f`.**

**Part 1: registry repair** (one file). **LANDED** as `579dbb7` (owner, 2026-11-25 18:59); the file at `579dbb7` is this sha256:

| File | sha256 |
|---|---|
| `agents/registry/parameter_registry.py` (= `registry_repair_land_20261125/repair/parameter_registry.py.repaired` = the 11-24 draft) | `72d0031569d42ed5c25d0cf5eaa7ef81a0e1827fe7c3728396d5c9e63956d341` |

**Part 2: registrations.** First copy `docs/drafts/loop_v1/registry_repair_land_20261125/register/parameter_registry.py.patched` over `agents/registry/parameter_registry.py`. Then land:

| File | sha256 |
|---|---|
| `agents/registry/parameter_registry.py` (registered) | `bea4fcfe81a1b1cccacbf979f46e0d17aebd58aaecb9411ea8c283fe7920611b` |
| `scripts/diagnostics/switch_check.py` (R10; after Beta R1 Q5) | `fe1ae65a58d3b7d90d28cab3c9d26831e780b18f51ca74408679e5fd4749e10e` |
| `tests/test_switch_check.py` (R10 gates, plus the fixture root outside every tree and `test_sc_fixture_root`; 15/15 on `fe1ae65a…`) | `cf0ef461f5d55b1c814ae66d1efdb2754d75d56817da1eec0b9076d80ff454a5` |
| `tests/test_switch_check_units.py` (new after Beta R1; no git) | `141e6eab3d97fa34c147044f76b06db56ee25a0680b4c264bfb53cf7f63522e3` |
| `tests/test_registry_ll_register.py` (new; 9 gates after Beta R1) | `4b181518b242973328a07980ece82e9e4b9d9ebd8bbb14d07787bb8c47f002e3` |
| `tests/mutate_e2e_grader.py` (after Beta R1 and the fixture root: targets `scu`, `lw`, `sct`; `sc`/`sct` fixtures outside every tree; no guard) | `937d453b6bb9840b571ae6daafa419f37a07a6f195dc7949af0b6c55da15c054` |
| `docs/drafts/loop_v1/READY_REGISTRY_REPAIR_LAND_20261125.md` (this note) | (as landed) |
| `docs/drafts/loop_v1/registry_repair_land_20261125/OWNER_MANIFEST_COMMANDS.txt` | `990fabb33cfdeaff0c73f5d641b7150f06e858362112be91630039659cec4a5c` |
| `…/REGISTER_LL_20261125.json` (**-f**) | `abf2a7f2c00de801b8a75619e2f6f71f3e1ca590be9d12656b1dea61874736c5` |
| `…/switch_check_7f4505a.md` (regenerated by the final checker) | `5d93b1db86a7a27e2c9384c60e9fedb096ca9e5519948309a093656b455372ef` |
| `…/repair/parameter_registry.py.repaired` | `72d0031569d42ed5c25d0cf5eaa7ef81a0e1827fe7c3728396d5c9e63956d341` |
| `…/repair/parameter_registry.py.repair.patch` | `6d4870eb666c024da9836c7161bb7db725382709d17e4c00d470e9f3df0b9d5c` |
| `…/repair/repair_record.json` (**-f**) | `592573e16711791cccd887f35f864569acf040b36353bc2852178f28dc0de4f7` |
| `…/repair/SHA256SUMS` | `d646296ed25440b704a10c8dc0b04d967688d1576aa1c86efd0e96906e18bdff` |
| `…/register/parameter_registry.py.patched` | `bea4fcfe81a1b1cccacbf979f46e0d17aebd58aaecb9411ea8c283fe7920611b` |
| `…/register/parameter_registry.py.patch` (diff against the repaired file) | `d2f129007fe2a9b504dba7d1f7976ba62e64d329b8b7ad13f2074e2ac0d68889` |
| `…/register/manifest_fragments.json` (**-f**; `test_registry_ll_register.py` reads it) | `2651c0942ac11a6a02604bdb562ac94bdd2f41b39b79267e688ee77274085c16` |
| `…/register/SHA256SUMS` | `9648335f94ab3e2d7e3697f5b80732f57ec611176bc4699504c35cee77c4813a` |

**Part 3: manifests.** These are owner hand-landed, using the two commands in Step 3. The expected sha256s are listed there.

**Not for landing** (evidence, in this worktree's `data/` and `logs/`):

| File | sha256 |
|---|---|
| `data/registry_repair_land/switch_check_7f4505a.json` (3.4 MB, the full check, final checker) | `7417141c…` (earlier runs kept in `data/registry_repair_land/prev/` and `prev_before_q5/`) |
| `data/registry_repair_land/argv_probe.py` | `bebae9ef…` |
| `data/registry_repair_land/unregistered_step_options_7f4505a.txt` | |
| `data/registry_repair_land/prev/` (the pre-`render_md` outputs, byte-identical except the `.md`) | |
| `logs/registry_repair_land/` | |
| `data/e2e_grader_selftest/` (self-test work dirs) | |

**Beta.** The registry, the checker and these tests are not in the before-commit review classes (CLAUDE.md, "Beta reviews": coordinator, worker and kernel code; verifier and launch path). The owner asked for a review of parts 2 and 3 anyway: X-REGISTRY-REPAIR R1, see "After Beta R1".

## Questions for the owner (nothing here is decided)

1. Do you want a separate correction of the kept specs that disagree with the code? The `mode` choices are the one an LLM could be misled by. Each would be a new decision, outside ruling #7.
2. `get_registry_for_agent` still maps Step 2 and Step 3 agent names to the removed `scorer_meta_optimizer.py` and `full_scoring.py`, so they return `None`. Should Step 3 map to `full_scoring_worker.py` now? That is a code change outside the registry list.
3. Do you want the 5 recommended options registered in the next landing?
4. Do you want an LL-2 route (the job generator and the wrapper carrying the two flags)? Until then, LL-2 is known but not reachable.
