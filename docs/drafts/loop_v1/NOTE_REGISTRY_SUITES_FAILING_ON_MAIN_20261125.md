# The six registry-importing suites that are red on clean main `7f4505a`

Task A of `logs/e2e_session/TASK_REG_FOLLOWUP_20261125.txt` (lead, 18:45).

**How this was done.** Read only: `nproc` = 2 here, so per the task no suite was re-run. The evidence is the logs from today's clean-main baseline in the check tree (`logs/registry_repair_land/A_baseline/` and `A2_baseline_with_dataset/`), plus code read at `7f4505a` and `git log`. Nothing was fixed.

## Plain summary

1. **Three suites fail on clean main because their tests are older than the code they test.** All three causes date from January 2026. Nothing is wrong at run time.
2. **`test_subphase3`.** The Step 1 retry-suggestion helper was rewritten in `2d20c8d` (2026-01-05). For the test's inputs it now returns no suggestions.
3. **`test_watcher_agent`.** Step 1 became a "does the output file exist" check in `6326257` (2026-01-11). The test passes a results dict and no file, so it gets escalate with confidence 0.
4. **`test_step_runner`.** The unused `agents/step_runner` package expects a string `success_condition`. Manifests use a list since `2146631`/`6326257` (2026-01-10/11).
5. **The three environment reds, one cause each:**
   - **admission binding:** the local GPU count fell from 2 to 1 (`f255912`, 2026-08-02); the test records this itself;
   - **Run 4 routing:** a frozen control that predates the Step 1 search (2026-11-14); recorded by Beta;
   - **P0.5 dataset authority:** **my own check-tree setup**, which lacked the gitignored `daily3.json`; this one is not a main defect.
6. **Correction to my READY note.** It named the routing red `G-ALLOWLIST-EXACT` and the P0.5 red "the live-fleet gate, VIR-5". Both were wrong. The real reds are `G-UNPINNED-IDENTICAL` (INCOMPLETE), and P0.5 Gates 24 and 31. The verdict comparisons in that note are unaffected.
7. **On WATCHER applying a proposal**, the two subphase/WATCHER failures do show something:
   - no suite today exercises a proposal becoming an applied change;
   - the paths that make parameter suggestions are not reached (all 7 manifests check files only);
   - when a suggestion is made, generic RETRY only logs it.
   That is **ALREADY RECORDED** (`CHAPTER_12_AUDIT.md` B5 `:91-93`; `CHAPTER_10_AUDIT.md:42`, `:135`). The suites hide nothing new about it.
8. **One NEW, dormant finding.** In its warning range, the Step 1 heuristic suggests `forward_threshold`/`reverse_threshold` = 0.005. Those are abort-only flags in `window_optimizer.py:1618-1626`. Nothing applies the suggestion today.

## Prior record

**Searches** (worktree at `7f4505a`, `/bin/grep -rn`):
- `file_exists` over `docs/CHAPTER_1[0-4]_AUDIT*.md`, `AUTONOMY_SWITCH_MAP_20261123.md`, `TODO_SELFPLAY_AND_LLM_AUTONOMY.md`, `LEADS.md` and `TODO_CONSOLIDATED.md`;
- `test_subphase3|test_watcher_agent\.py|test_step_runner` over `docs/`;
- `step_runner` over the `CHAPTER_*AUDIT*`, `AUTONOMY_SWITCH_MAP`, `TODO_CONSOLIDATED` and `BACKLOG` files;
- `disable_heuristic_parsing|_handle_retry|suggested_param_adjustments` over `docs/*.md` and `docs/drafts/loop_v1/*.md`;
- `G-UNPINNED-IDENTICAL` over `docs/` and `logs/`;
- `get_heuristic_decision|forward_threshold.*0\.005|window_optimizer_context` over `docs/`.

Plus `git log -S` on each cause, and a walk of `distributed_config.json` history for the localhost `gpu_count`.

**Found:**

| Record | What it records |
|---|---|
| `CHAPTER_12_AUDIT.md:28-29` | every manifest is `file_exists`; 692 of 714 host decisions `file_exists`, 0 heuristic |
| `CHAPTER_12_AUDIT.md` B5 `:91-93` | RETRY does not apply adjustments (dormant) |
| `CHAPTER_10_AUDIT.md:42`, `:135`, D8 `:169`, `:212` | `step_runner` has no importer |
| `CHAPTER_10_AUDIT.md:279-281` | the subphase suites and `test_step_runner` were never run by that audit |
| `AUTONOMY_SWITCH_MAP_20261123.md:88`, G4 `:245` | `file_exists` manifests bypass the LLM evaluation path |
| `tests/test_s172_admission_binding.py:546-548` (added `c4e0037`) | "the other nine reds fail identically at baseline for an unrelated environment reason: the localhost execution set now declares one GPU, not two" |
| `docs/exchange/to_beta/X-MINER-START_R1_ALPHA_REVIEW_REQUEST_seed_start_into_sieve.md:30, :74`; `docs/exchange/from_beta/X-MINER-START_R1_BETA_REVIEW.md:122, :154` | `G-UNPINNED-IDENTICAL` cannot pass on today's path; "frozen oracle INCOMPLETE … No complete routing certificate claimed" |

**Not found:** any record of the three subphase/WATCHER/step-runner suite failures as failures. Any record of the heuristic suggesting the abort-only threshold flags.

**Not searched:** the host, ser8 archives, the private repository, project files not in the repository.

## Suite by suite

### 1. `test_subphase3.py`: red on clean main, a stale test

| | |
|---|---|
| **Failing check** | `test_retry_suggestions`, `assert len(suggestions) > 0` (`test_subphase3.py:246`). The other 8 checks pass (log `A_baseline/test_subphase3.py.log`). |
| **Cause** | `WindowOptimizerContext.get_retry_suggestions` (`agents/contexts/window_optimizer_context.py:266-275`) is now a deprecated wrapper around `get_heuristic_decision` (`:146-230`). The test's `bad_results` has no `seed_count`, so `seeds_tested` = 0 (`:87-88`). `extract_step1_derived_metrics` then divides by `max(0, 1)` = 1 (`utils/metrics_extractor.py:60-66`), so `bidirectional_rate` = 5000. That exceeds every threshold (`distributed_config.json:106-110`), so it lands in the escalate branch, which has `suggested_params: None` (`:216-223`), and gives `[]`. Even with suggestions the test would then fail at `s['reason']` (`test_subphase3.py:249`), because the new items carry only `param` and `suggestion` (`:275`). |
| **Since** | `2d20c8d` (2026-01-05, "Refactor: LLM prompt architecture + threshold guardrail"). Before it, `get_retry_suggestions` returned rule-based suggestions with a `reason` for `bi_count > 1000` (`2d20c8d^:…window_optimizer_context.py:176-206`). The test is from `1953cca` (2025-12-06). |
| **Recorded?** | The failure: **NEW**. The mechanism around it (heuristic not reached under `file_exists`): ALREADY RECORDED (`CHAPTER_12_AUDIT.md:28-29`). |
| **Hides anything about WATCHER applying a proposal?** | **It hides that the Step 1 heuristic path is not tested.** It is the only suite that exercises `get_retry_suggestions`, whose output WATCHER's heuristic evaluator copies into `suggested_param_adjustments` (`agents/watcher_agent.py:1499-1504`). That evaluator is not reached under the `file_exists` manifests. Where it is reached, `_handle_retry` only logs the adjustments and re-runs with the same params (`:1658-1663`). ALREADY RECORDED (B5). |
| **NEW, dormant** | In the warning range, `get_heuristic_decision` suggests `{"forward_threshold": 0.005, "reverse_threshold": 0.005}` (`window_optimizer_context.py:198-209`). `--forward-threshold` and `--reverse-threshold` are "UNWIRED — passing this ABORTS the run" (`window_optimizer.py:1618-1626`). Today it can do no harm: the path is not reached, the keys are not in `window_optimizer.json` `default_params` (so the step-scoped filter would drop them), and RETRY does not apply them. It would matter once G7 (an apply step) is built. In that case the prediction is no run at all, not a lower hit rate. |

### 2. `test_watcher_agent.py`: red on clean main, a stale test

| | |
|---|---|
| **Failing check** | `test_heuristic_evaluation`, `assert decision.confidence >= 0.7` (`test_watcher_agent.py:111`). The log shows `action=escalate, confidence=0.00` after `File validation failed: optimal_window_config.json - File does not exist`. |
| **Cause** | `evaluate_results` reads the Step 1 manifest. `agent_manifests/window_optimizer.json` is `evaluation_type: file_exists` with `success_condition: ["optimal_window_config.json"]`. The file check runs first (`agents/watcher_agent.py:1139-1161`) and returns `confidence=0.0` and escalate when the file is absent (`:1195-1201`). The results dict the test passes is never weighed. |
| **Since** | `6326257` (2026-01-11, "Step 1 file_exists evaluation"), which also set `disable_heuristic_parsing: true`. The test is from `1953cca` (2025-12-06). |
| **Recorded?** | The failure: **NEW**. The cause: ALREADY RECORDED (`CHAPTER_12_AUDIT.md:28-29`; Ch12 D3/D4 via `CHAPTER_10_AUDIT.md:327`, `:332`: "live path gives `file_exists`, conf 1.00/0.00"). |
| **Hides anything about WATCHER applying a proposal?** | **Same as suite 1.** It was meant to show WATCHER weighing a result and deciding retry with suggestions. That path is dormant for all 7 steps. ALREADY RECORDED (B5, G4). Nothing new. |

### 3. `agents/step_runner/test_step_runner.py`: red on clean main, an unused package

| | |
|---|---|
| **Failing checks** | 3 of 4: Manifest Loading, Multi-Action Manifest and Command Building. Imports pass. Each fails with the pydantic error `StepManifest.success_condition: Input should be a valid string … input_value=['optimal_window_config.json']` (and `['optimal_scorer_config.json']`). |
| **Cause** | `agents/step_runner/models.py:116` declares `success_condition: str = ""`. The manifests carry a list. |
| **Since** | `2146631` (2026-01-10, Step 2.5 and Step 3 manifests to file-based evaluation) and `6326257` (2026-01-11, Step 1). The package is from `15bf8c2` (2026-01-02) and has not changed since. |
| **Recorded?** | The package being unused: ALREADY RECORDED (`CHAPTER_10_AUDIT.md` D8 `:169`, `:212`; `AUTONOMY_SWITCH_MAP:75`). The type mismatch and the suite failure: **NEW**. |
| **Hides anything about WATCHER applying a proposal?** | **No.** Nothing outside the package imports `agents/step_runner` (`grep -rn step_runner --include=*.py`: hits only inside the package). Its `suggested_params` field (`models.py:318`) and its `parameter_bounds` reader (`command_builder.py:151`) are on no live path. |

### 4. `tests/test_s172_admission_binding.py`: environment red, recorded in the test

| | |
|---|---|
| **Failing checks** | 9 of 20: B1, B2, B5, B6, C1, C2, C3, C4, C5. B1 reports 25, not 26; the others see one local identity (`zeus-ubuntu-vm:gpu0`) where they expect two. The two mutant checks pass. |
| **Cause** | `distributed_config.json:6` declares localhost `gpu_count: 1`. The gates assert a two-GPU local set and a 26-identity fleet (`tests/test_s172_admission_binding.py:337-361`, `:645-668`). The fleet is 1 + 8 + 8 + 8 = 25. |
| **Since** | `f255912` (2026-08-02 17:22, "localhost gpu_count 2 -> 1 to match measured hardware — the second 3080Ti is on VM100"). That is 17 hours after the suite was written in `eff6616` (2026-08-02 00:10). Every earlier config revision back to `b425171` (2026-01-14) has 2. |
| **Recorded?** | **ALREADY RECORDED** in the suite itself (`:546-548`, added in `c4e0037`, 2026-08-09). |
| **Hides anything about WATCHER applying a proposal?** | **No.** It is miner admission (how many workers a run waits for). It has no parameter-proposal path. |

### 5. `tests/test_s172_run4_routing_patch.py`: environment red, recorded by Beta

| | |
|---|---|
| **Failing check** | 1 of 22: `G-UNPINNED-IDENTICAL` is **INCOMPLETE**: "run_step returned before dispatch; no argv was built". The many `[FAIL]` lines in the log are inside mutant runs (M3, M4 …), which are meant to red. Each mutant is credited PASS. |
| **Cause** | The gate drives WATCHER through the frozen fixture `tests/fixtures/run4_routing_clean_control.py`. Its `_StubDB` has only `get_certified_cursor` (`:88-90`). `run_step` now asks the database for the open Step 1 search (`agents/watcher_agent.py:2212`, `_db.select_step1_search(...)`). The stub has no such method, so Step 1 is blocked with `STEP1_SEARCH_LOOKUP_FAILED` before dispatch. The fixture cannot be edited: its sha256 is the oracle (`test_s172_run4_routing_patch.py:103-105`). The successor control `G-UNPINNED-SUCCESSOR` passes. |
| **Since** | `b1cd2fa` (2026-11-14, "WATCHER schedules Step 1 from the open search"), stage 1 of the Step 1 search. |
| **Recorded?** | **ALREADY RECORDED**: X-MINER-START R1 request `:30`, `:74`, and Beta's review `:122`, `:154` ("frozen oracle INCOMPLETE … No complete routing certificate claimed"). |
| **Hides anything about WATCHER applying a proposal?** | **No.** It concerns the operator-only warm-start pin. The pin's routing is covered by the passing gates `G-CHAIN`, `G-EXACT`, `G-ORIGIN` and `G-UNPINNED-SUCCESSOR`. |

### 6. `tests/test_s172_phase6_p05_dataset_authority.py`: red only in my check tree

| | |
|---|---|
| **Failing checks** | 2 of 37 in stage A2: Gate 24 ("CLEAN control — good reads still succeed") and Gate 31 ("published artifacts byte-unmodified"). Both raise `FileNotFoundError: …/daily3.json`. In stage A, before I copied any dataset file, more gates red, including Gate 26. |
| **Cause** | `REAL_ALIAS = os.path.join(_ROOT, D.LEGACY_ALIAS_NAME)` (`:55`) is hashed at `:518` and `:701`. `daily3.json` is gitignored. I copied the versioned file and `daily3_current.json` into the check tree, but not `daily3.json` itself (`check_tree` setup: `cmp: … daily3.json: No such file`). |
| **Since** | Not a property of main: it comes from my check-tree setup. The main tree has `daily3.json`. |
| **Recorded?** | Not applicable. **My READY note misdescribed it** ("the live-fleet gate, VIR-5"). Gate 26 (VIR-5) passes in A2. |
| **Hides anything about WATCHER applying a proposal?** | **No.** It concerns the dataset pointer only. |

## What this means for the registry landing

- None of the six reds is caused by the registry repair or the registrations. Every verdict was identical in stages A2, B, C and D (READY note, Proof).
- Two of the reds (suites 1 and 2) are the only suites that touch WATCHER's suggestion path. They show that path is dormant. So the registry is **known** to WATCHER's code, but no live path today turns a registered value into an applied change, apart from:
  - an operator setting `params`, which the new manifest keys admit;
  - the Step-5 health retry, whose whitelist excludes the LL keys.
  This matches `AUTONOMY_SWITCH_MAP` G4 and G7.
