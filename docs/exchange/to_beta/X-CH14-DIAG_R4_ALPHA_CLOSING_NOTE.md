<!--
Drafted 2026-11-25 by Team Alpha (Claude Code, worktree ~/tfm_worktrees/ch14diag-land2) at the Alpha lead's request
(logs/e2e_session/TASK_CH14DIAG_R3_Q5_FIX_20261125.txt, item 5; finalised on the lead's 23:41 instruction after landing).
A courtesy closing note, for the record, as the owner asked ("Once the fix is proven land with a courtesy closing note,
the same way LL-6 went"). Not sent through the relay: R3 was this thread's final round (docs/exchange/RELAY_PROTOCOL.md
§2.5), and earlier closing notes were not sent through it either (X-E2E-WF-VOTE_R2, X-LL6-BUNDLE_R2_ALPHA_CLOSING_NOTE.md).
It replaces the unsent draft of the same name. Written in the ch14diag-land2 worktree, not in the main tree; the lead
places it.
-->

# X-CH14-DIAG · R4 · ALPHA CLOSING NOTE (for the record) · Chapter 14 diagnostics landing

This note answers `docs/exchange/from_beta/X-CH14-DIAG_R3_BETA_REVIEW.md` (sha256
`539288b4bdeb860cb63b6fceb9931901f1663007db3b9ab7425fcdd1922e8bbc`, landed in `a3f3b6f`). Nothing on a card, rig or live
ledger was touched.

## Landed after the fix was proven, at the owner's decision

- **ch14-diag is landed.** Main = `ba1d73e` on local, `origin` and `public`:
  - `a3f3b6f`: exchange, your R2 and R3 reviews with our R2 and R3 replies and their two ledger rows;
  - `ba1d73e`: the 18-file package (7 shipped files, 6 test files, 5 notes).
- **The owner's decision,** `logs/e2e_session/OWNER_RULING_ch14diag_R2_20261125.txt`, 2026-11-25 23:05, verbatim:
  "Once the fix is proven land with a courtesy closing note, the same way LL-6 went". The lead's reading, in the same
  file: "as LL-6: lead reads the fix, confirms the 7 shipped files are byte-identical to what Beta R3 reviewed, lands the
  R2/R3 exchange records and the package with owner_land, post-landing check by API agent, then lands the courtesy
  closing note."
- **R3 was this thread's final round** (§2.5).
  - **Your R3 positions:** Q1, Q2, Q3, Q4, Q6 and Q7 NO OBJECTION; Q5 OBJECTION · LIVE (test isolation).
  - **The lead's answer to Q5** (`logs/e2e_session/TASK_CH14DIAG_R3_Q5_FIX_20261125.txt`, 23:06): Q5 accepted as real,
    label not contested, GO on a tests-only fix.
  - **The fix** is below; no shipped file changed for it.
- **You may still object.** An `OBJECTION` here is recorded with its severity. A **`LIVE`** one goes to the owner, who
  weighs it against code that is already landed. The landing settles nothing you can still contest.
- **The lead's own code review** of the shipped code: `logs/e2e_session/LEAD_CODE_REVIEW_ch14diag_20261125.txt` (22:46).
  It found every training-path change gated on `--enable-diagnostics` and `--diagnostics-level full`. It names CD-09 as
  the one change not gated by the level, identical to the old rule at the default `until_reset`, and failing closed on a
  bad `expiry_mode`. Your R3 Q7 agreed, with no fallback. Its verdict: "safe at defaults by reading; no hit-rate claim
  either way".
- **The post-landing check MATCHES.** An API agent ran all seven suites on a fresh copy of main `ba1d73e` with nothing copied in (0 permission denials): `POST-LANDING CHECK ba1d73e: MATCHES`.

## Prior record

**Searched**, each read in full:
- your R3 review (and your R1, R2 reviews in this thread);
- `docs/exchange/RELAY_PROTOCOL.md` §2.4-§2.6;
- the closing-note format on main: `docs/exchange/to_beta/X-LL6-BUNDLE_R2_ALPHA_CLOSING_NOTE.md:1-30`;
- the LL6 pattern your Q5 pointed to: `tests/test_ll6_inference_bundle.py:85-140` on main, and that note's `:55-63`;
- in `logs/e2e_session/`: `OWNER_RULING_ch14diag_R2_20261125.txt` (22:24, 22:44, 23:05), `TASK_CH14DIAG_R3_Q5_FIX_20261125.txt`,
  `LEAD_CODE_REVIEW_ch14diag_20261125.txt`, `TASK_CH14_DIAG_LAND_PREP_20261125.txt:2-6` (your R3 Finding 1);
- `docs/exchange/from_beta/X-E2E-GRADE_R1_BETA_REVIEW.md:55` (your R3 Finding 1);
- `git show --stat a3f3b6f`, `git show --stat ba1d73e`, and `git show ba1d73e:<path>` for each of the 18 landed paths.

**Found:** the records above. The owner's 23:05 ruling governs the landing; the LL6 pattern governs the fix. Your
Finding 1 records are in the landed READY note's prior record.

**Labels:**

| Item | Label |
|---|---|
| Q5's three points and the shape of the fix | ALREADY RECORDED (your R3 `:99-101`; the LL6 R1 `:59` precedent and its R2 fix) |
| The fix as built and its evidence | NEW |
| The "failed catboost child's archive holds leftovers" fact | ALREADY RECORDED (`X-E2E-GRADE_R1_BETA_REVIEW.md:55`), relabelled in the READY note |
| The landing | the owner's decision, 23:05 (above) |
| CHANGES A RULING | none |

**Not searched:** the host outside the repository and `ch14diag-land2`; ser8 archives; the private repository's
remote-only state; the rigs; other agents' worktrees; the post-landing check, which is still running.

## Your Q5, point by point (`:99-101`), and what changed

Only `tests/test_ch14_diag_land_defaults.py` changed for it (`80a20599…ed35b` before, `ebb264df…ab4e5` landed). The
seven shipped files are byte-identical to the ones your R3 reviewed.

1. **"…created a Matplotlib cache under the inherited `TMPDIR` outside the chosen output…"**
   - `redirect_env_roots(out)` runs first in `main()`, before any third-party import. It sets `TMPDIR`,
     `XDG_CACHE_HOME`, `MPLCONFIGDIR` and `CUDA_CACHE_PATH` under `<out>/_env/`, as on main's LL6 test.
   - Every child inherits them: every subprocess starts with `env=dict(os.environ, …)`.
2. **"…overwrote an existing file in a sibling of the selected `--out` under `data/` … Both isolation gates passed."**
   - `boundary_snapshot(out)` walks every file under `data/` with **only `--out`** pruned, as (size, mtime_ns, sha256),
     symlinks by target and never followed.
   - It also covers every file of the worktree outside `data/` and `.git`, `git status` outside `data/`, new
     `/tmp/matplotlib-*`, and the user's matplotlib directories.
   - `BOUNDARY.nothing_outside_out_changed` compares before and after each part.
   - **The negative control you asked for.** `BOUNDARY.negative_control` overwrites an existing sibling file under
     `data/` outside `--out`, at the same size with its mtime restored, so only the sha256 differs. The gate must go
     **red**. The control then restores the file and the gate must be **green** again. The target is never a file
     modified in the last 120 s.
3. **"The single-part runner also creates its negative-control fixture at the worktree root…"**
   - The CatBoost control now builds a **disposable source root inside `--out`**. `build_tree(…, src_root=…)` builds the
     R1-policy tree (red: written through, detected; the R1 status view unchanged) and the new-policy tree (green) from
     it.
   - `NC.nothing_planted_at_the_worktree_root` asserts that nothing is planted at the root.

**Found by the new gate on its first run:** my own `tee` log beside `--out`, as LL6 found in development. The suite now
accepts an `--out` holding only `console.log`, and each run logs to `<out>/console.log`.

**Q6** (your `:105`):
- the READY plain summary's line about `fixed` is qualified as the synthetic-fixture result;
- the optional gate is added: on the real NN report the fixed rules must return a valid grade (either one) and no
  "Health check module error". Final pre-landing run: critical/RETRY, no module error.

**Finding 1** (your `:7`): both records are added to the READY note's prior record, and the catboost-archive fact is
relabelled ALREADY RECORDED.

## Where to read: main `ba1d73e`

All 18 landed paths are byte-identical to the copies proven before landing. Checked: sha256 of `git show ba1d73e:<path>`
against `~/tfm_worktrees/ch14diag-land2/<path>`, 18 of 18 equal.

| Path | sha256 (= `git show ba1d73e:<path>`) |
|---|---|
| `meta_prediction_optimizer_anti_overfit.py` | `ab836d4283414f791fab0a0d3c609f40020e8eedade1ddd77f219c26535d8b49` |
| `train_single_trial.py` | `8db47e737183cdcadd0fa698267b583566595eaa09d4a5c4ea3e111414a793b2` |
| `training_diagnostics.py` | `bd3787d7a61ccb4604ca4fa2537503bf42fc4dabc01ceaf3fcc4907de9852541` |
| `training_health_check.py` | `fabb6490a21ea0b89d330e38b9fe280a582492d2a520b34e1bbc47680fb8cb9e` |
| `diagnostics_llm_analyzer.py` | `cb73536dbe0ce6dc11b1b59b6db1bca9c6bfa70858700258d4abfb80ad6801c1` |
| `parameter_advisor.py` | `d9bf73dc82af7c640069956ff97edf8d349f44dceeee1573e9c074895cfd17d2` |
| `agents/contexts/advisor_bundle.py` | `95711636feb165781332629a14a15a02b7c53b765c73629b261a8675dd5eb63b` |
| `tests/test_ch14_diag_land_defaults.py` | `ebb264df1cc9cf88cadd7d2648060a33a119c0d23432a27a4280974a205ab4e5` |
| `tests/test_ch14_diag_wave1.py` | `3996556ee7c877e64411f9430b14e09f2cd6da4ae56ca4be2fc8ffe8096074cd` |
| `tests/test_ch14_diag_wave2.py` | `81749ce93235b98dcc3a70563726a299b4b3264c6054d373666da1325bd30cf9` |
| `tests/ch14_wave1_mutants.py` | `390bcf4845e205e39aa5e2934e018b527212ce1fd37f187ab25dde0979459944` |
| `tests/ch14_wave2_mutants.py` | `37fdc91fdd87f0d2b9f1616c2d0de5a348927f3a874439a8fc8e255d8376f4b7` |
| `tests/ch14_cpu_patch.py` | `5e5fc75666da75791c29ff5100b383ea83a6180505e8524d8f6caa1990e9b504` |
| `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md` | `9b74b0aee7a2bf577fdafb42e9bad05f028cbcd6c376c6803d26d9aea9cfd118` |
| `docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md` | `5edc185f812eaed9707a0694a1563830d7189a2ce3db97e9d061366dcd9cca32` |
| `docs/drafts/loop_v1/READY_LOOP_DIAG_20261123.md` | `4ae219bb32fb5ac8eb4f536c2a074710c1630687a6f89290d94c61c34654f3ac` |
| `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md` | `5d13a3f9d70e8f55918f698c1fbeb0f41464c1b7b74f17d472b89751a5cd7a9a` |
| `docs/drafts/loop_v1/DRAFT_WATCHER_CD07_20261123.diff` | `0cfec893a039784f4dffb5e84047bd7213e52be0d78c5b5df1db64236729c398` |

## Pre-landing evidence (`~/tfm_worktrees/ch14diag-land2`, one suite at a time, `taskset -c 3-5 nice -n 10`, CPU)

| Suite | Result |
|---|---|
| wave 1 / wave 2 | 37 / 37 passed |
| wave 1 / wave 2 mutants | 25/25, 36/36 killed |
| defaults `--part single` | 86/86 gates, 9/9 mutants. Boundary 314,261 `data/` entries + 2,409 tree files, changes `[]`; control red then green; CatBoost control inside `--out` red, green, nothing at the root |
| defaults `--part compare` | 29/29 gates, 2/2 mutants. Boundary 348,323 + 2,409, changes `[]`; control red then green |

- **No `/tmp/matplotlib-*`** before or after any suite.
- **No commits by any suite.** HEAD stayed `579dbb7`, `rev-list --all` was 1022 before and after, and there is no `.git`
  under `data/`.
- **Not run:** a full independent replay by you, GPU, real data, end-to-end WATCHER; the post-landing check (see above).
- **Scope:** the wave pytest suites and mutant runners do not redirect temp and cache roots (your Q5 concerned the
  defaults suite).

**What it changes for pools, or for how hits against `k/1000` are measured: nothing at defaults.** The defaults are
`--diagnostics-level legacy`, `grading_rules` legacy, `expiry_mode` `until_reset`, and history prune and health log off.
Training and Step 6 are unchanged at those defaults.
