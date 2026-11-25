# X-E2E-GRADER-LANDED  R1  REVIEW_REQUEST  hit-rate grader as landed (courtesy review)

- **Review class:** AFTER-COMMIT, courtesy. The owner approved the landing on 2026-11-24 18:32 PST and asked that Beta see it as a courtesy. Nothing is waiting on this review. A `LIVE` objection becomes a follow-up fix that goes to the owner; a `THEORETICAL` one is recorded.
- **Base commit:** `81c9386` (main, both remotes). The grader landed in `76234eb`; four json records were force-added in `81c9386`.
- **Rounds:** one.
- **Files under review:**
  - `scripts/diagnostics/e2e_grader.py` (sha256 `c7fe48bf102daa42b284713f4b2c2508d7f21c9613a01f063be443c881fd9ace`; pinned by sha256 by the walk-forward driver and by `utils/step1_screen.py`, so it must not be edited without repinning)
  - `tests/test_e2e_grader.py`, `tests/mutate_e2e_grader.py`, `tests/fixtures/e2e_grader_a4d96fe7.py.txt`
  - Context: `docs/drafts/e2e_v1/READY_E2E_GRADER_20261123.md`, `docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md`
- **Not under review here:** `scripts/diagnostics/compare_settings.py`, `scripts/diagnostics/switch_check.py` and the registry-repair drafts under `docs/drafts/loop_v1/` (drafts; nothing in the pipeline reads them).
- **Allowed answers per question:** `NO OBJECTION`  `OBJECTION` (with `LIVE` or `THEORETICAL`)  `SUGGESTION`  `NEEDS EVIDENCE`.

## Prior record

**Searches run** (main `81c9386`, 2026-11-24):

```
git grep -l -i -e 'e2e_grader' -e 'WP-3' HEAD -- docs/PROJECT_FILE_CATALOG.md 'docs/TB_RULING_*' \
  'docs/TB_RULING_REQUEST_*' 'docs/PROPOSAL_*' 'docs/CHAPTER_*' 'docs/*_AUDIT.md' \
  docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
git grep -n -i 'WP-3' HEAD -- docs/exchange/DEFERRED.md
```

**Documents found:**
- `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` (WP-3, the instrument this grader implements).
- `docs/exchange/from_beta/X-FINDINGS-PLAN_R1_BETA_REVIEW.md`, `X-FINDINGS-PLAN_R2_BETA_REVIEW.md` (your Q5 measurement rules) and `docs/exchange/DEFERRED.md:8` (Q5 deferred to WP-3).
- `docs/exchange/to_beta/X-E2E-GRADER_R1_ALPHA_REVIEW_REQUEST_grader_first_form.md`: an earlier DRAFT request, never sent, written before landing against grader sha256 `a4d96fe7...`. It is superseded by this request. Its eight design questions are NOT asked here.
- Not matched in the searched scope: any TB_RULING, CHAPTER or AUDIT document naming the grader.

**Labels:**
- The instrument and its rules: ALREADY RECORDED (WP-3; your Q5). This is its implementation, landed.
- The landing itself and this courtesy request: NEW.
- CHANGES A RULING: none. Owner ruling 2026-11-23 10:40: the owner's and the lead's authorisation are what a landing requires.

**Not searched:** the host outside the repository, ser8 archives, the private repository's branches, project files outside the repository.

## What was run (VM101 CPU, fresh detached worktree of main `81c9386`, no card)

| Suite | Result |
|---|---|
| `tests/test_e2e_grader.py` | 36/36 |
| `tests/mutate_e2e_grader.py` | 87/87 killed and credited on the pre-commit copy of main (`abdf9f7` plus the landing files); rerun on `81c9386` in progress when this was written |

Outputs: `~/tfm_worktrees/_review/final_*.out`, `~/tfm_worktrees/_review/gl_*.out` (host files, not in the repository).

## Questions, real-world cases only

**Q1. Counting.** For a real Step 6 pool on this system today, does the landed grader report a wrong hit count, a wrong `k_t` or a wrong expected-hits figure? The functions are `validate_pool_values`, `grade_draw`, `pb_pmf` and `pb_upper_tail` in `scripts/diagnostics/e2e_grader.py`, under header rules R4, R5, R7 and R8. Rule as implemented: `k_t` is the number of distinct integers in the pool, one membership hit per target draw, expected hits is the sum of `k_t / draw_space`, and the p-value is the exact Poisson-binomial upper tail.

**Q2. Refusals.** Is there a malformed, stale or duplicate input that a realistic run here would produce and that the landed grader would grade as a miss or a hit instead of refusing (rules R6, R14, R15, R16)?

**Q3. Development mode.** `--mode development` (R22) is meant never to produce a counted verdict. Is there a path in the landed file where a development run writes a counted verdict or an unlabelled headline?

Please label every objection `LIVE` or `THEORETICAL` as `AGENTS.md` section 5 requires, and read `AGENTS.md` "The learning loop" before judging what a single graded block means.
