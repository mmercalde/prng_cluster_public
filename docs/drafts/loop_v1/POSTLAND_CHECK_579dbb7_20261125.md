# POST-LANDING CHECK · main `579dbb7` (registry repair) · 2026-11-25

- **Task:** `~/distributed_prng_analysis/logs/e2e_session/TASK_WF_POSTLAND_AND_LL6_BETA_20261125.txt`, Part A (lead, 19:10).
- **Prior record:** the expected results come from the registry-repair READY note, `~/tfm_worktrees/registry-repair/docs/drafts/loop_v1/READY_REGISTRY_REPAIR_LAND_20261125.md`:
  - `:96-97`, `:302`: stage B, "every verdict as baseline"; `test_subphase1` "Registry has 5 parameters" (was 9);
  - `:310-328`: the suites that already fail on clean main, and those that pass.
  - Nothing else decides this check. `NEW`: the results below.
- **Tree:** a fresh detached worktree, `~/tfm_worktrees/_postland_check_579dbb7`, at `579dbb7e558da1d90032967ca7b3e28054180e7a`.
  - **Nothing was copied in except the five gitignored `daily3*.json`** from the main tree (`.gitignore:41`): `daily3.json`, `daily3_current.json`, `daily3_evening.json`, `daily3_midday.json`, `daily3-20260801T145551443433Z-513648160d35.json`.
  - The tree is now removed (`git worktree list` shows none).
- **Method:**
  - one suite at a time, foreground, `taskset -c 0-2 nice -n 10 python3 -u -B <suite>`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=""`, venv `~/venvs/torch`;
  - I used my own loop rather than the headless agent's wrapper: that wrapper (`~/.local/state/api-agents/worktrees/20261125-190129-postland-registry-579dbb7/data/postland_check_579dbb7/run_suite.sh`, read) pins `taskset -c 6-7 nice -n 15`, and this task says `0-2`;
  - logs: `~/tfm_worktrees/ll6-land/data/postland_579dbb7/` (`SUMMARY.txt` `6d49539f…`).
- Nothing committed, pushed or landed. No card.

## Plain summary

- **The landed registry is the reviewed file, and it imports.**
  - sha256 of `agents/registry/parameter_registry.py` = `72d0031569d42ed5c25d0cf5eaa7ef81a0e1827fe7c3728396d5c9e63956d341`, as expected.
  - It imports (`import_check.log` `d0e644ae…`). `list_all_registries()` gives 4 scripts holding 9 specs: `window_optimizer.py` 5, `adaptive_meta_optimizer.py` 1, `meta_prediction_optimizer_anti_overfit.py` 2, `prediction_generator.py` 1. That matches the commit's "9 kept".
- **All 8 suites give the expected verdict.**
  - 6 pass; `test_subphase1` reports "Registry has 5 parameters".
  - `test_subphase3` and `test_watcher_agent` fail exactly as on clean main.
- **Every log equals stage B's, line for line,** once timestamps, run ids, `/tmp` temp-dir names and durations are normalised: 0 differing lines in all 8.

## Results

| Suite | rc | Result | Expected (stage B) | log sha256 |
|---|---|---|---|---|
| `test_subphase1.py` | 0 | ALL TESTS PASSED; "✅ Registry has 5 parameters" | pass, 5 parameters | `c3d696bb…` |
| `test_subphase2.py` | 0 | ALL TESTS PASSED | pass | `457d0fe0…` |
| `test_subphase3.py` | 1 | TEST FAILED: `assert len(suggestions) > 0` (`test_retry_suggestions`) | fails on clean main, same assert | `c7e13630…` |
| `test_subphase4.py` | 0 | ALL TESTS PASSED | pass | `c96a8311…` |
| `test_watcher_agent.py` | 1 | TEST FAILED: `assert decision.confidence >= 0.7` (`test_heuristic_evaluation`; confidence 0.00) | fails on clean main, same assert | `a3d2bbad…` |
| `tests/test_watcher_followup_repairs.py` | 0 | 17/17 PASS, sentinel PASS | 17/17 | `2092a937…` |
| `tests/test_watcher_llm_integration.py` | 0 | 13 passed, 4 warnings | 13 passed | `394fd855…` |
| `tests/test_watcher_safety_repairs.py` | 0 | 21/21 PASS, sentinel PASS | 21/21 | `70744a64…` |

## Checks around the run

- **Log comparison with stage B** (`~/tfm_worktrees/registry-repair/logs/registry_repair_land/B_part1_repair/`):
  - normalised for wall-clock stamps, durations, `/home/michael/…` paths, `/tmp/<name>_<8 chars>` temp dirs, `step1_run1_<stamp>` and `learning_<stamp>` ids;
  - 0 differing lines in each of the 8 logs, both in order and sorted.
- **Main tree:** `git status --porcelain` the same before and after. Optuna listing unchanged.
- **The check tree** gained only `.pytest_cache/` (ignored, from `test_watcher_llm_integration`) before it was removed.
- **Writes outside the tree:** `test_watcher_followup_repairs` and `test_watcher_safety_repairs` write their own temp dirs under `/tmp`, as on clean main. These are landed suites and nothing was changed in them.

## Limits

- Only the 8 suites the task named were run. The other suites in the registry READY note's table were not re-run on `579dbb7`: `agents/step_runner/test_step_runner`, the three `test_s172_*`, `test_seed_domain_cursor_amendment`, `test_step1_search_record_stage1`, `test_registry_repair`, `test_switch_check`, `test_registry_ll_register`.
- No hit-rate evidence. This check concerns the registry and WATCHER tests only.
