# READY · TASK_LOOP_GRADE + TASK_LOOP_ADDENDUM_1 (window g, worktree e2e-grader) · 2026-11-23

## Plain summary

All three TASK_LOOP_GRADE items, and the grader's part of TASK_LOOP_ADDENDUM_1 (rank; each session
on its own and pooled), are built and self-tested on CPU fixtures. Nothing was committed, landed or run on
run H data, and no GPU was used.

1. **Development mode in the grader.**
   - It is chosen explicitly with `--mode development`; the default is `counted`.
   - It stamps DEVELOPMENT into every output and can never produce a counted verdict.
   - It records an overlap with a previously inspected block instead of refusing it. Rule R13 is
     unchanged in counted mode.
   - A counted-mode run now refuses a grade block that declares `"mode": "development"`.
2. **Per-draw detail.** `--per-draw-detail on` (default `off`) writes `per_draw_detail.jsonl`, one
   row per record of every arm. Each row holds:
   - draw index, date and session;
   - the drawn number;
   - pool size (distinct and raw);
   - hit or miss;
   - `vote_rank`: the drawn number's 1-based first position in the pool as written;
   - `vote_count`: its number of repeats.
3. **LL-4, `scripts/diagnostics/compare_settings.py`.** It checks that two or more run records
   differ only in declared fields, checks the pools (declared arms only, equal k), passes the
   inspected blocks through, and calls the grader exactly once.

4. **Rank (addendum item 2).** Every graded draw can be scored by the rank (1 to 1000, 1 = top of
   the vote) of the drawn number. The per-draw rank is in the detail file. `--rank-summary on`
   (default off) adds, per arm, the mean rank against the uniform expectation 500.5 and a one-sided
   Monte Carlo test under the exact per-draw null. Ties take the mid-rank, and are counted and
   stated. It works in both modes and is descriptive: it is never in the multiplicity family and
   never changes the counted verdict.
5. **Sessions (addendum item 3).** `--per-session-report on` (default off) adds each session on
   its own beside the pooled result. It is descriptive and unadjusted, and the 08:08 counted rule
   is unchanged.

**Tests:**
- Grader: **36/36 gates** (25 existing, 6 from TASK_LOOP_GRADE, 5 from the addendum). 35.9 s (final run;
  the previous run measured 37.9 s and peak RSS 111 MB).
- `compare_settings`: **9/9 gates**, 4.1 s.
- Mutants: **58/58 killed and credited**, applied, ran and credited proven for each:
  - the 22 existing;
  - M26–M39 (development mode and detail);
  - M40–M50 (rank and sessions);
  - CM1–CM11 (`compare_settings`).

  80.0 s on the final run with 6 parallel mutant workers (the previous run: 105.8 s at 190 % CPU,
  peak RSS 121 MB).
- The counted default is proven byte-equivalent to the authorised `a4d96fe7…` grader on two
  fixtures, apart from timestamp, provenance, the recorded `mode` field and the new setting keys in
  `config` and `defaults`.

**The grader's sha256 has changed** from `a4d96fe7…` to
`c7fe48bf102daa42b284713f4b2c2508d7f21c9613a01f063be443c881fd9ace`. The run H counted grade used
`a4d96fe7…`. A frozen copy is kept at
`data/e2e_grader_selftest/reference/e2e_grader_a4d96fe7.py.txt` (read-only, sha verified). The
walk-forward driver (`e2e-map`) must pin the new sha.

## Addendum 1 (lead 17:50; owner 17:42): what the grader does now

**Rank, R24 (`e2e_grader.py` header).**
- The ordering comes from the record when the writer supplies one:
  - `value_ranking`: all 1,000 values, best first;
  - `value_scores`: one score per value, higher is better, ties allowed.
- Otherwise it falls back to the pool: the distinct pool values in first-position order, every
  unvoted value tied below them. `--rank-source record_only` refuses that fallback.
- **The pools written so far (run H, H6) carry only the top-k votes, so their rank is the
  pool-derived one.** A value outside the pool gets the mid-rank of the unvoted block, for example
  (k + 1 + 1000) / 2. A rank across all 1,000 values needs the vote writer to emit
  `value_ranking` or `value_scores`. That is the vote harness's or the walk-forward agent's change;
  nothing in this worktree does it.
- **The test:**
  - exact null mean of the rank sum = n × 500.5;
  - exact null sd, from each draw's own mid-ranks, so the ties are in it;
  - z (normal approximation);
  - a Monte Carlo p, one-sided toward small ranks: (1 + #{sim ≤ obs}) / (R + 1), R = 10,000 by
    default, seeded.
  - A runtime control checks the Monte Carlo mean against the exact mean.
- **Tested:**
  - the exact sd for untied orderings equals √(n(N² − 1)/12);
  - Monte Carlo p agrees with the normal approximation on a null fixture;
  - a planted top-10 rank gives p = 1/(R+1), and bottom ranks give p > 0.99;
  - all-tied orderings give p = 1;
  - mid-ranks: a 4-way top tie gives 2.5;
  - pool miss with k = 5 gives 503, with a tie of 995;
  - a strong rank signal with no hits leaves `above_chance`, m, the adjusted p and the headline
    unchanged;
  - with rank off, a malformed rank field is ignored, as today.

**Sessions, R25.** With `--per-session-report on`, each arm gains `per_session`, which gives hits,
expected, null sd, unadjusted p and, if rank is on, the rank summary for each session of the grade
block. The pooled result is the arm's own, unchanged. Using both sessions is the grade block's
`sessions` list, which already takes both. The counted rule (one counted test, midday, 08:08) is
unchanged.

**Full history, item 1.** This is the walk-forward agent's part. The grader already grades any
block; nothing in it is tied to the 80/20 cut.

**Synthetic scale-check fixture.** The fixture's pools and draws are independent random numbers by
construction. Its development headline read "above the modelled background", with one-sided
p 0.020 on 16,000 draws (mean rank 497.3 against 500.5, Monte Carlo p 0.033). That is a statement
about one random fixture, recorded as measured. It says nothing about TFM, and the grader's
controls passed.

## Prior record

**Searches** (main tree `~/distributed_prng_analysis` at `41ad509`):
- Command: `/bin/grep -rlniE "development mode|walk[- ]forward|compare_settings|setting[- ]comparison|per_draw_detail|vote[_ ]rank"`.
- Scope: `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`,
  `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`,
  `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`.
- Also searched: `/bin/grep -rlniE "compare_settings|development mode|per_draw_detail"` over
  `~/tfm_worktrees/*/docs/drafts` and `~/tfm_worktrees/*/scripts`.

**Found:**
- Every main-tree hit is "walk-forward". The principle is decided in
  `docs/TB_RULING_S176_WATCHER_KPI.md` Phase C, `:692-727` (also `:47`, `:365`). Other mentions are
  in `TB_RULING_S177`, `TB_RULING_S179`, `PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_*`,
  `CHAPTER_13_AUDIT.md:192`, `CHAPTER_7_*`, `TODO_CONSOLIDATED.md` and X-STEP1-SEARCH R1.
- No development mode, setting-comparison harness or per-draw detail exists in the main tree.
- **Worktrees:**
  - LL-4 is specified in `~/tfm_worktrees/ch14-learn/docs/drafts/ch14_v1/PLAN_CH14_LEARNING_LOOP.md:134`
    (contract §4 `:182-210`).
  - `~/tfm_worktrees/e2e-map/scripts/diagnostics/e2e_walk_forward.py` already calls the grader
    with these settings, at `:461-463` and `:790-836`.
- **Task and direction:**
  - `logs/e2e_session/TASK_LOOP_GRADE.txt` and `TASK_LOOP_COMMON.txt`;
  - `OWNER_DIRECTION_learning_loop_team_20261123.txt` (17:16);
  - `TASK_COMMON.txt` and `TASK_CH14_COMMON.txt`.

**Labels:**

| Item | Label |
|---|---|
| Walk-forward as development evidence | **ALREADY RECORDED** (TASK_LOOP_COMMON, lead decision 17:25; S176 Phase C) |
| Development mode R22, per-draw detail R23, `compare_settings` C1–C9 | **NEW**, the implementation |
| LL-4 itself | **ALREADY RECORDED** (plan `:134`) |
| Counted-mode refusal of a development-declared grade block | **NEW**, a fail-closed addition with no effect on a block that omits the field |
| Changes to a ruling | **none**. R13 is unchanged in counted mode |

**Not searched:** the host, ser8 archives, the private repository, project files outside the
repository.

## What was built (anchors at the files' current sha256)

**`scripts/diagnostics/e2e_grader.py`**
- sha256 `c7fe48bf102daa42b284713f4b2c2508d7f21c9613a01f063be443c881fd9ace`, 1,400 lines, untracked.
- Rules R22–R25 are in the header, after R21.
- **Hunks**, for the lead's merge (no overlap file is touched):
  1. Header: R22 and R23 text; the pool record's optional `pool_order`; the grade block's optional
     `mode`.
  2. `GRADER_VERSION` is now `0.2.0-draft`. Adds `MODES` and `DEFAULTS["mode"]="counted"`,
     `DEFAULTS["per_draw_detail"]="off"`.
  3. `parse_grade_block`: validates the optional `mode` and adds the key only when it is present.
  4. `resolve_records`:
     - R13 overlaps are recorded in development mode, and the counted path still raises;
     - validates `pool_order`;
     - each entry keeps `raw_pool` and `pool_order` (internal only; `_row_out` is unchanged, so
       `per_draw.jsonl` is unchanged);
     - per-arm `previously_inspected_overlaps`.
     - The signature and return shape are unchanged. The e2e-vote loader calls this function.
  5. New `detail_row` and `_dev_headline`.
  6. `write_outputs(…, detail_rows=None)`:
     - the `DEVELOPMENT` marker file and the development title;
     - the detail file;
     - both listed in SHA256SUMS. Counted outputs keep the same file set.
  7. `run`:
     - `mode` and `per_draw_detail` in `cfg`;
     - the counted-mode refusal of a development block;
     - the `_emit` helper, which stamps rows in development mode;
     - the development transform after the nulls-first sort: `above_chance` and `counted_verdict`
       become null, `development_above_background` is added, headlines are reworded, the
       `above_chance` list is empty;
     - `summary["mode"]` set to `COUNTED` or `DEVELOPMENT`.
  8. Parser: `--mode` and `--per-draw-detail`.
  9. Addendum hunks:
     - header R24 and R25, and the record's optional `value_ranking` / `value_scores` /
       `rank_basis`;
     - import `norm, rankdata` from scipy;
     - `DEFAULTS` rank and per-session keys and `RANK_SOURCES`;
     - the `resolve_records` entry keeps the rank fields, unvalidated until used;
     - new `draw_ranks`, `rank_summary` and `session_breakdown`;
     - in `run`: rank settings in `cfg`; ranks computed per graded draw when the rank summary or
       the detail is on; `rank_summary`, `controls[aid]["rank_test"]` (its Monte Carlo control
       joins `controls_ok` only when on) and `per_session` added to the arm result;
     - `detail_row` rank fields;
     - descriptive lines in SUMMARY.md, only when on;
     - parser `--rank-summary`, `--rank-source`, `--rank-test-reps`, `--rank-test-seed`,
       `--per-session-report`.

**`scripts/diagnostics/compare_settings.py`**
- sha256 `097b05bbd6d72a9b2f54b9745270d11900663f2e249f6827b7454ca0fd41f360`, new. The addendum
  only adds the rank and per-session keys to the `grader_options` pass-through allowlist.
- Rules C1–C9 are in the header. It loads the pinned grader only for its helpers (signatures
  asserted in `test_cs_signatures`) and grades through one grader subprocess call.
- **Declaration** (`tfm.e2e.compare_settings.v1`) has three groups of paths:
  - `setting_fields`: what is being compared;
  - `bookkeeping_fields`: allowed to differ, such as timing and memory;
  - `identity_fields`: `arm_id`, `config_id` and the pools sha, each checked against the variant's
    declared value.

  A difference anywhere else is refused, and so is any overlap between the groups.
- A real H6 run-record pair (F1 A0 at pool 100 against 300) diffs in 25 paths, all of which fall into
  these three groups. That is the shape I designed against; I did not run the harness on it.

**Tests** (`tests/`, untracked):

| File | sha256 |
|---|---|
| `test_e2e_grader.py` | `c8666920…5fa20` |
| `test_compare_settings.py` | `758c94cf…aefc` |
| `mutate_e2e_grader.py` | `80f0ca1f…ecd3`; now also runs the `compare_settings` mutants |

## Tests and results

| Suite | Result | Log |
|---|---|---|
| Grader self-test, existing 25 gates, after the edits | 25/25 | `logs/e2e_grader/loop_selftest_existing_run1.log` |
| Grader self-test, all 31 (before the addendum) | 31/31, 33.6 s | `logs/e2e_grader/loop_selftest_full_run1.log` |
| Mutants before the addendum | 47/47 | `logs/e2e_grader/loop_mutants_run1.log` |
| **Grader self-test, all 36 (final)** | **36/36**, 35.9 s | `logs/e2e_grader/loop_selftest_full_run3.log` |
| **`compare_settings` self-test (final)** | **9/9**, 4.1 s | `logs/e2e_grader/cs_selftest_run4.log` |
| **Mutants, final** (22 old + M26–M50 + CM1–CM11) | **58/58 killed and credited**, all baseline gates pass, 80.0 s | `logs/e2e_grader/loop_mutants_run3.log`; `data/e2e_grader_selftest/mutants_20261123T180203/report.json` |
| Rank scale check: synthetic, 16,000 draws, pool 300, development + detail + rank + per-session, 4 workers | rc 0, controls pass; 5 min 38 s, 679 MB. The same inputs with those settings off: 3 min 43 s, 381 MB | `data/e2e_grader_selftest/scale_rank/run.time`, `base.time` |

**New grader gates:**
- **`test_counted_default_equivalence`:**
  - Runs the frozen `a4d96fe7…` copy and the new grader on the same inputs, with defaults.
  - Fixture (a): two arms, a comparison, pending draws, abstentions.
  - Fixture (b): exclude_and_count, a duplicate target, first_in_file.
  - Byte-equal: `per_draw.jsonl`, the file set and stdout.
  - Equal: `summary.json`, after removing only `generated_utc`, `provenance` and the new `mode` keys.
  - Equal: SUMMARY.md, apart from the title timestamp and the Instrument line.
- **`test_dev_mode_explicit_r13`:** the default and explicit counted mode still refuse R13;
  development mode records the overlap.
- **`test_dev_mode_stamps`:** every row, every detail row, the marker in SHA256SUMS, the title, the
  label and the headlines carry DEVELOPMENT.
- **`test_dev_never_counted_verdict`:** a planted arm is ABOVE CHANCE in counted mode. In
  development mode it has null verdict fields and no counted wording anywhere.
- **`test_counted_refuses_dev_block`.**
- **`test_per_draw_detail`:**
  - known first positions with repeats;
  - two arms;
  - pending, abstained and excluded rows;
  - the default writes no file.

**`compare_settings` gates:**
- signatures;
- the clean path, where exact McNemar runs at equal k and the hits equal the fixture's known counts;
- refusals:
  - a hidden difference;
  - a sibling path that shares a declared prefix;
  - no setting difference;
  - overlapping groups;
  - unequal k (and its opt-out, where the grader reports "not tested");
  - a third arm in the pools;
  - a declaration edited after binding;
  - R13 through the grader, plus development pass-through;
  - grader sha, input sha and identity mismatches.
- A refusal writes nothing.

**Multi-core:**
- The grader's uniform-random control uses worker processes (`--workers`; the self-test uses 1 for
  speed).
- The mutation runner runs mutants in parallel (`--workers 6`, 272 % CPU).
- Both self-tests are single-process; each takes seconds.

## Not done and unverified

- **Equivalence** is proven on two fixtures, not over every input. I did not re-grade run H with the
  new grader: that would be a rerun, which TASK_GRADE_RUNH forbids.
- **`compare_settings` on real data:** never run on real run records or pools; fixtures only.
- **LL-4's dependencies:** the plan lists LL-3 and the grader landed. Neither is exercised here.
- **Gate 22 was not run.** The untracked `.py` files under `scripts/` will red it until they are
  landed, as before.
- **Fixture data:** 246 MB of fixture output under `data/e2e_grader_selftest/`, including the scale
  fixture (all runs so far; untracked, not for commit). The reference copy there must stay, because `test_counted_default_equivalence` fails
  closed without it.
- **Beta review class, as a fact:** both files are in the before-commit class. They are the
  verifier path for counted results (CLAUDE.md "Beta reviews"; plan `:134` for LL-4).
- **GPU:** none needed. Nothing here runs on the card.
- **Rank cost at 16,000 draws:** about 2 min of the 5 min 38 s. The Monte Carlo gathers are
  random reads in an n × 1000 table. At the walk-forward block size (about 600 draws) it is about
  1 s. `--rank-test-reps` can lower it. I made no optimisation.
- **Full ordering:** the grader does not check that a record's `value_ranking` agrees with its pool
  (for example, that the pool's values are its top entries). TODO-E2E-G10, `draw_ranks`.

## Questions (one line each)

- **Lead:** for future *counted* grades, should the pinned grader be the new `c7fe48bf…`, whose
  default is proven equivalent on fixtures, or should counted runs stay on the frozen `a4d96fe7…`
  copy until Beta reviews the new one?
