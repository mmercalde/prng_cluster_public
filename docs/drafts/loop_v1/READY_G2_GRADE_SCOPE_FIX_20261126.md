# READY — G-2 fix: a promoted family's grade carries the winner's evidence (2026-11-26)

Team Alpha task agent. Worktree on main `933a4a0`. Nothing committed, nothing launched, CPU only.
Credit: **found by Gemini trial test 2, 2026-11-26.** Verified by reproduction the same morning
(`/home/michael/.local/state/api-agents/worktrees/20261126-065026-verify-g2-933a4a0/scratch_g2/repro_g2_output.txt`,
case A). Wanted behaviour: owner yes 2026-11-26 07:14 (lead's design, as passed to this agent).
Not a Beta request; the lead folds it into the X-BENCHING-SWITCH package.

## Prior record

**1. Searches run** (from the worktree root, `/bin/grep` so `.json` is not skipped):

```
P='grade_scope|all_families|worst_family|worst family|G-2\b|grade set by'
P2='grade[_ ]scope|all[_ ]families|CD-06'
for g in docs/PROJECT_FILE_CATALOG.md 'docs/TB_RULING_*' 'docs/TB_RULING_REQUEST_*' 'docs/PROPOSAL_*' \
         'docs/CHAPTER_*' 'docs/*_AUDIT.md' docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md \
         docs/exchange/ ; do /bin/grep -rnE "$P" $g; /bin/grep -rnE "$P2" $g; done
# plus, outside the required list: docs/drafts/ and claude/ with P2; Q6 in X-CH14-DIAG exchange files
```

**2. Documents found**

| Scope | Hits on the subject |
|---|---|
| `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md` | none |
| `docs/PROPOSAL_*` | one keyword hit, unrelated (`PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md:22`, "same pattern all families", sieve registry) |
| `docs/CHAPTER_*` | "G-2" hits are an unrelated gap label (`CHAPTER_2_SOURCE_MAP_v1.md:485`, session-stream ruling) |
| `docs/exchange/` | CD-06 pointer mentions only (`to_beta/X-CH14-DIAG_R2_ALPHA_REPLY.md:123`, `R3:228`); X-CH14-DIAG "Q6" there is a different question (NN real-report row), not CD-06's Q6 |
| `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:173` (outside list) | CD-06 defines `grade_scope: winner \| all_families`, default `winner`; acceptance "`all_families` lists four grades". Says nothing on whose issues/fixes/metrics/archive a promoted grade carries. |
| `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:302` (outside list) | Plan Q6: retry when winner healthy but another family critical, or grade the winner only. Records the question, not this behaviour. |
| `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md:118` (outside list) | registry spec for `grade_scope`, default `winner`. |

The record is silent on what a promoted result should carry; the governing statement is the owner
yes of 2026-11-26 07:14 named above.

**3. Labels**

| Item | Label |
|---|---|
| Promoted result carries the promoted family's issues, fixes, metrics, archive entry | NEW (owner yes 2026-11-26 07:14) |
| Winner's issues kept, prefixed `[winner]` | NEW (same) |
| `winner_model_type`, `grade_set_by`, `grade_note` keys when promoted | NEW (same) |
| `grade_scope` default `winner`, `grading_rules` default `legacy` | ALREADY RECORDED (PLAN CD-06 `:173`; READY_LOOP_DIAG_2 `:118`); unchanged |

No item changes a ruling.

**4. Not searched:** the host, ser8 archives, the private repository, project files not in the
repository; the Gemini trial's own write-up (not located in the repository).

## What changed

`training_health_check.py`, `_check_training_health_fixed()` only (+26 / −5; `git diff --stat`):

| Line | Change |
|---|---|
| `:794`, `:809` | `family_reports[m] = (frep, fg)` keeps each graded family's own report and grade (already loaded by the existing loop; nothing new is read). |
| `:814-818` | records `winner_name` and `promoted` when a worse family sets the grade (selection logic unchanged). |
| `:822-830` | when promoted: `issues` = promoted family's own issues, then the winner's issues prefixed `[winner]`. Guard: if the promoted report were not held, the result says so instead of substituting the winner's. By construction it is always held (`worst_family` comes from `graded`, filled only after a successful load and grade, `:811-812`), so the guard is unreachable today. |
| `:837` | the per-family loop also skips the winner when promoted (its issues are already in, prefixed). |
| `:840` | the winner-report freshness line is prefixed `[winner]` when promoted. |
| `:842`, `:844`, `:846` | archive entry, `suggested_fixes`, `metrics` from the promoted family's report/grade. |
| `:847-850` | when promoted: `winner_model_type`, `grade_set_by`, `grade_note` added to the result. |

New: `tests/test_g2_grade_scope_fix.py` (255 lines). sha256 at writing: thc
`edd25477…38ae`, test `1a5d9876…25c`.

## Unchanged at defaults (proved against `git show 933a4a0:training_health_check.py`)

`test_unchanged_modes_identical_to_base`: result dict, archive entry (minus `archived_at`) and skip
registry (minus `last_critical`) equal to the base function for `{}` (legacy default), explicit
`legacy`, `fixed` (default scope) and `fixed` + `winner`, each with the non-winner critical and ok
(8 cases). Non-vacuity: fixed cases assert two graded families, the non-winner's issues listed, the
non-winner critical in the critical fixture; legacy cases assert the NN canonical report is graded.
`test_promoted_case_differs_from_base` shows the same comparison detects a difference.
`test_all_families_without_promotion_identical_to_base`: all_families with no worse family is also
identical. No promoted-only key appears when nothing is promoted.

## Test results (one suite at a time, `CUDA_VISIBLE_DEVICES=`, pytest tmp dirs)

| Suite | Result |
|---|---|
| `tests/test_g2_grade_scope_fix.py` | **19 passed** |
| same suite against the base-revision copy (`G2_THC_PATH`) | 6 failed (the 6 promoted tests), 9 passed (equivalence), 4 skipped (mutants) — confirms the tests see the defect |
| `tests/test_ch14_diag_wave1.py` | 37 passed |
| `tests/test_ch14_diag_wave2.py` | 37 passed |
| `tests/test_ch14_diag_land_defaults.py` | **not run**: a script, not pytest-collectable ("no tests ran"); it runs the real Step 5 CLI and writes under `<worktree>/data/` (its docstring), which this task forbade. Its health-check rows are at legacy, which the equivalence tests above cover. |

Mutants (each in a temp copy, one named test run in a subprocess, must exit 1 with "1 failed"):

| Mutant | Red test |
|---|---|
| restore old skip condition (`m != model_type`) | `test_promoted_issues_are_own_and_winner_prefixed` |
| restore winner `suggested_fixes` | `test_promoted_fixes_are_own` |
| restore winner archive (`_archive_diagnostics(diag, …)`) | `test_promoted_archive_is_own` |
| restore winner metrics (extra) | `test_promoted_metrics_are_own` |

All 4 killed.

## Not handled

- The skip registry and action still follow the promoted family (`_severity_to_action(severity,
  model_type, …)`), as before; benching/skip lines ~1028-1096 untouched (another agent's).
- Ties between families: `max()` keeps today's first-max choice; not changed.
- No default changed. `land_defaults` not run (above).
- Line numbers after `:850` shift by +21 relative to `933a4a0`; the other agent's patch at ~1028-1096
  will need a re-anchor, not a semantic merge.
