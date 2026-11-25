# READY: Grader G, the hit-rate grader (WP-3 first form), 2026-11-23

**Plain summary**
1. **No hit rate was measured (0 pools from TFM graded).** The grader was run only on synthetic fixtures. No run H data, no pipeline step, no GPU.
2. **Built:**
   - `scripts/diagnostics/e2e_grader.py` (1,064 lines);
   - its self-test `tests/test_e2e_grader.py`: 25 gates, 25 of 25 pass;
   - a mutation runner `tests/mutate_e2e_grader.py`: 22 of 22 mutants were killed by their credited assertion, and each kill shows the mutant was applied and the mutated code ran.
3. **Measured run time.**
   - Self-test: 25.4 s wall, peak RSS 110 MB.
   - Mutation suite: 47.3 s, peak RSS 110 MB per process.
   - Run-H-shaped synthetic scale run (2 arms × 3,614 draws, 200 control replicates): 22.9 s, peak RSS 141 MB.
   - **Multi-core:** the uniform-random control fans out over `--workers` processes (default min(4, cores−1); the scale run used 328% CPU). The mutation runner runs 4 mutants at a time.
4. **Format mismatch with the H6 vote harness (NEW, open).** The H6 draft in `~/tfm_worktrees/e2e-vote` writes records in its own shape, and the grader as built rejects them. The H6 records also lack two fields the grader requires:
   - `dataset_sha256`;
   - `data_cutoff_index`.

   The two formats have to be reconciled before the first grade. The field map and the choice are in Q1.
5. **The rules are in the file header (`e2e_grader.py:15-104`, R1–R21), with anchors.** They come from WP-3, addendum §E and §F, Beta's R2 §4, Beta's Q5 omissions and S176 Phase C. Every rule is enforced at a named function, so this can become the committed instrument rather than a second one (TODO-E2E-11).
6. **Not Beta-reviewed.** A one-round review request is drafted, not sent: `docs/exchange/to_beta/X-E2E-GRADER_R1_ALPHA_REVIEW_REQUEST_grader_first_form.md`.

## Prior record

**Searches run** (worktree `e2e-grader` at `41ad509`, clean before work; `/bin/grep -rlniE`).
- **Scope S:** `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`.

| # | Pattern | Scope | Hits read |
|---|---|---|---|
| G1 | `e2e_grader\|hit[-_ ]rate instrument\|WP-3` | S | `PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` §1, §2 WP-3 (`:36-42`, `:78-86`); `X-FINDINGS-PLAN_R2_BETA_REVIEW.md` Q5 (`:69-87`); `X-FINDINGS-PLAN_R1_BETA_REVIEW.md` (file list); `to_beta/X-FINDINGS-PLAN_R2_ALPHA_REPLY.md:28`, `:54`; `exchange/DEFERRED.md:8` (Q5 THEORETICAL objection deferred to WP-3) |
| G2 | `poisson[- ]binomial\|poisson_binomial` | S | none |
| G3 | `poisson.?binom\|k_t\|distinct.*k/1000` | `--include=*.py .` | matches on `k_t` only, none a grader. `ls scripts/diagnostics` gave "No such file" |
| G4 | `grade block\|label block` | S | none (only in the map, outside S) |
| G5 | `poisson[- ]binomial\|mcnemar\|bonferroni\|holm` | S | none |
| G6 | `frequency control\|most frequent\|draw-frequency` | S | `HYBRID_SKIP_BOUND_AUDIT.md` (keyword only, not about pools) |
| G7 | `uniform[- ]random (pool\|control)` | S | WP-3 `:85`; Beta X-FINDINGS-PLAN R1 |
| G8 | `abstention\|abstain`, `pending.{0,20}miss` | S | WP-3 `:83`; R2 §4 `:124-125`; Beta R1/R2 reviews; Alpha R2 reply |
| G9 | `WP-3\|instrument\|hit.?rate` | `TODO_CONSOLIDATED.md`, `BACKLOG.md` | none |
| G10 | headings | `LEADS.md` | L-1…L-7. None concerns pool grading |
| G11 | read in full or by section, as TASK G directs | — | map `MAP_E2E_THIN_PATH_20261122.md` (in `e2e-map`), §2.7 and all others; `OWNER_DIRECTION_e2e_20261122.txt`; addendum §E, §F (`:115-135`); `X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md` §4 (`:112-136`); `TB_RULING_S176_WATCHER_KPI.md` §4.1–4.3 (`:292-383`), Phase C (`:692-727`) |
| G12 | existing graders | repo | `evaluate_pools.py:28-50` (k from the pool label, one draw per call); `watcher_kpi_metricC_deterministic_v2_2.py:1-60` (provenance and bool-rejection patterns, adopted with citation) |
| G13 | H6 format | `~/tfm_worktrees/e2e-vote/scripts/diagnostics/e2e_vote_harness.py` (read only, draft dated 07:47 today) | `:75-76`, `:113-123`, `:494-516` |

**Documents that already decide, schedule or record the subject:**
- WP-3 (`PROPOSAL_FINDINGS…_v1_1.md:78-86`): new files only, rules in the header, Beta before commit.
- Addendum §F (`:130-135`) and §E (`:115-128`).
- Beta R2 §4 (`X-REVERSE-SIEVE_R2…:112-127`).
- Beta's Q5 omissions (`X-FINDINGS-PLAN_R2_BETA_REVIEW.md:75-85`). These were deferred to WP-3 (`DEFERRED.md:8`).
- S176 Phase C (`:692-727`).
- Map §2.7 and TODO-E2E-11.
- Handoff `TFM_SESSION_HANDOFF_20261114.md:72` (as cited by the map; not re-opened here).

**Labels:**

| Item | Label |
|---|---|
| The grader and its rules | **ALREADY RECORDED** (WP-3, §F, map §2.7). Built here; the status changes from "never built" to "draft" |
| Beta Q5 items 1–5 adopted into the header (R2, R10, R12, R13, R19) | **ALREADY RECORDED** (Beta's suggested §F text, `X-FINDINGS-PLAN_R2_BETA_REVIEW.md:85`). First adoption in code |
| Lift interval by test inversion for a common factor λ (R8) | **NEW**. A concrete choice for "specify the uncertainty calculation for varying k" (§F `:134`). Beta review asked |
| Runtime known-count control on the real outcomes | **NEW** |
| H6/grader record-format mismatch | **NEW** |
| Run H `daily3.json` puts the evening row before the midday row within a date (8,514 of 18,067 adjacent steps from row 1039; measured today, read-only) | **ALREADY RECORDED** (`CHAPTER_2_SOURCE_MAP_v1.md:203-205`; ruling summary `BACKLOG.md:105-120`, chronological reorder cancelled) |
| Consequence: the grader's index cutoff `data_cutoff_index ≤ t` lets a (d, midday) forecast use the (d, evening) outcome, which is drawn later in real time | **NEW** (TODO-E2E-G9; Beta Q6; owner Q4) |
| **CHANGES A RULING** | **none** |

**Not searched:**
- the ser8 archives, the private repository, project files outside the repository;
- the host main tree beyond reading run H's `daily3.json` shape (`date`/`session`/`draw`, 18,068 rows, read-only);
- `docs/exchange/` bodies other than those listed.

## 1. What was built

### 1.1 `scripts/diagnostics/e2e_grader.py` (sha256 `a4d96fe7…52169a`)

**Inputs, all by path, all sha256-recorded in the summary:**

| Input | Flag | Rule enforced | Anchor |
|---|---|---|---|
| per-draw pools JSONL (`tfm.e2e.pool.v1`) | `--pools` (optional `--pools-sha256`) | malformed, blank or wrong-schema line → error | `read_pool_records` `:359`; `resolve_records` `:386` |
| frozen dataset | `--dataset` + **required** `--dataset-sha256` | mismatch → error; each draw an int in [0, 999] | `load_dataset` `:233` |
| grade-block definition | `--grade-block` | start/end (end may pass the dataset → pending), sessions, frequency-control block | `parse_grade_block` `:272` |
| arm list | `--arms` | per arm: `config_id`, `upstream_unseen_from_index`, `min_k`/`max_k`, `previously_inspected_blocks`; comparisons; `family_alpha`; `bonferroni`/`holm` | `parse_arms` `:297` |
| abstention rule | `--abstention` | allowed or not, allowed reasons, empty pool → error or abstain | `parse_abstention` `:345` |

**Per draw** (`resolve_records` `:386-531`, `validate_pool_values` `:214`, `grade_draw` `:227`):
- `k_t` = distinct ints in [0, `draw_space`−1]. Bool, float, str, None, a non-list pool and out-of-range values are all errors.
- One membership hit. 000 is valid.
- Recorded per draw:
  - draw identity (combined index, date, session), checked against the dataset row;
  - the raw-pool sha256 (verified if the record carries one);
  - `config_id`, checked against the arm;
  - `data_cutoff_index`, which must be ≤ the target index (S176 Phase C);
  - the target, which must be ≥ the arm's `upstream_unseen_from_index` (R2);
  - abstentions with their denominator, pending targets, and out-of-block records.
- **Errors, not misses:**
  - missing record (`:524-531`);
  - stale record: dataset sha, date or session, config;
  - duplicate record: declared rule `error` (default), `first_in_file` or `max_data_cutoff`;
  - leak, future cutoff, k outside the arm's declared bounds;
  - grade block overlapping a previously inspected block (R13).

**Statistics:**
- Expected hits = Σ k_t/1000 (`arm_statistics` `:632`).
- Exact Poisson-binomial one-sided p = P(X ≥ H) (`pb_pmf` `:535`, `pb_upper_tail` `:550`). Draws with equal p are grouped as `scipy.stats.binom` and the groups convolved.
- Interval: exact test-inversion bounds for a common lift λ (`lift_bounds` `:572`). Reported as a descriptive 95% two-sided interval, plus a one-sided lower bound at the corrected level.
- Correction over the declared family (`:914-917`, `adjust_pvalues` `:617`):
  - default `all_declared_tests` = arms + declared comparisons;
  - `arms_only` selectable;
  - Bonferroni by default, Holm selectable.
- Paired comparison (`compare_arms` `:749`):
  - common draws only;
  - both budgets (Σk) reported;
  - exact McNemar only when k_t is equal on every common draw, otherwise "not tested: unequal budget".
- Lag-1 autocorrelation of (hit − p), reported as a diagnostic with no verdict.

**Controls, run on every invocation:**
- Uniform-random pools of the same k_t, through the same validate/grade code (`uniform_random_control` `:668`).
  - Per-replicate seed = SeedSequence([seed, rep]), so the result is identical for any worker count (gate A-RCTRL-DET).
  - Pass when |mean − E| ≤ `--control-tolerance-z` (default 4) standard errors **and** the empirical rejection rate ≤ the exact test size + 4 SE.
- Draw-frequency control (`frequency_control` `:702`): the top-k_t values of a declared block, which must end before the first graded draw. Ties go to the lower value.
- Known-count control on the real outcomes (`known_count_control` `:730`): include the outcome on even positions, exclude it on odd ones; the result must be exact.

**Outputs:** only a **new** `--out-dir`, which is refused if it exists or sits under `/tmp`, `/dev/shm` or `/var/tmp` (`_check_out_dir` `:812`). Files:
- `per_draw.jsonl`, in arm and target order;
- `summary.json`: inputs and their sha256, declarations, defaults, provenance (git HEAD and dirty state at `--repo-root`; fatal unless `--no-provenance`), results nulls-first, `null_results_first` headlines with every denominator, controls;
- `SUMMARY.md`;
- `SHA256SUMS`.

On an input error nothing is written (exit 2). Exit 3 means outputs were written but a control failed.

**Nothing hard-coded that should move.**
- These are rule files (input files), and each is sha256-recorded:
  - label/grade split → grade block plus `upstream_unseen_from_index`;
  - session filter → grade block `sessions`;
  - vote-position arms, ranking key and winner rule → arms (`config_id`), from H6's run record;
  - k → whatever H6 writes, bounded by the arm's `max_k`.
- `draw_space`, the duplicate rule, the out-of-block rule, the multiplicity family, control reps, seed, tolerance and workers are CLI flags. Their defaults are in `DEFAULTS` (`:155`) and recorded in the summary.

### 1.2 Tests

`tests/test_e2e_grader.py`: 25 gates. Every assertion is tagged `[A-…]`. Fixtures go under `data/e2e_grader_selftest/`, never `/tmp`.

| Area | Gates / tags |
|---|---|
| Known count (varied k 1/5/20/37, winning value duplicated) | A-HITCOUNT, A-EXPECTED, A-DUP-ONEHIT, A-OUTSHA, A-SUMSHA, A-KC-RUNTIME |
| §E: duplicates | A-K-DISTINCT, A-DUP-ONEHIT |
| §E: out of range / wrong type | A-RANGE-HI/LO, A-TYPE-BOOL/FLOAT/STR/NONE/DICTPOOL |
| §E: a draw of 000 | A-ZERO-HIT, A-ZERO-OUTCOME |
| §E: pending vs miss | A-PENDING, A-PENDING-NOTMISS (denominator and expectation unchanged) |
| §E: stale target | A-STALE (date, session), A-STALE-SHA, A-STALE-CFG, A-POOLSHA-BIND |
| §E: repeated grading | A-DUP-RECORD, A-DUP-FIRST, A-DUP-MAXCUT |
| §E: missing / malformed | A-MISSING, A-MISSING-ARM, A-MALFORMED (bad JSON, blank line), A-SCHEMA, A-NOFILE |
| Leak / causality | A-LEAK, A-CUTOFF, A-R13, A-FREQ-CAUSAL |
| Dataset binding | A-DSHA, A-DS-RANGE |
| Out-dir | A-OUTDIR, A-OUTDIR-TMP (nothing is created under /tmp) |
| Statistics | A-PB-BINOM, A-PB-BRUTE (2¹⁰ enumeration), A-PTAIL, A-LIFT, A-BONF, A-M-ALL, A-M-ARMS, A-HOLM, A-BONF-NONE, A-MCNEMAR, A-UNEQUAL |
| Controls | A-RCTRL-PASS, A-RCTRL-SIZE, A-RCTRL-DET, A-FREQ-ORDER, A-FREQ-HITS |
| Abstention | A-ABST-DENOM, A-ABST-REASON, A-ABST-DENY, A-EMPTY |
| Reporting | A-NULL-FIRST (summary.json and SUMMARY.md), A-DENOM-HEADLINE |
| Bounds / sessions | A-MAXK, A-OOB, A-OOB-COUNT |
| CLI and provenance | A-CLI, A-PROV (commit equals `git rev-parse HEAD`), A-CLI-ERR, A-PROV-FATAL |

`tests/mutate_e2e_grader.py` checks each mutant three ways:
- **applied:** the original text occurs exactly once, and the mutant's sha differs from the original;
- **ran:** the mutant calls `_MUT_HIT(id)` into a marker file, which works across forked workers too;
- **credited:** the credited gate fails with the credited tag.

Each credited gate is first run on the unmutated grader, and all of them pass.

| Mutant | Breaks | Killed by |
|---|---|---|
| M1 | k counts repeats | A-K-DISTINCT |
| M3 | bool accepted | A-TYPE-BOOL |
| M4 | 1000 accepted | A-RANGE-HI |
| M5 | 000 graded a miss | A-ZERO-HIT |
| M6 | pending dropped | A-PENDING |
| M7 | duplicate record not rejected | A-DUP-RECORD |
| M8 | missing not an error | A-MISSING |
| M9 | leak allowed | A-LEAK |
| M10 | no correction | A-BONF |
| M11 | P(X>H) | A-PTAIL |
| M12 | dataset sha skipped | A-DSHA |
| M13 | random control k−1 | A-RCTRL-PASS |
| M14 | stale target accepted | A-STALE |
| M15 | future cutoff accepted | A-CUTOFF |
| M16 | expected from a fixed k | A-EXPECTED |
| M17 | frequency control not causal | A-FREQ-CAUSAL |
| M19 | positives before nulls | A-NULL-FIRST |
| M20 | existing out-dir accepted | A-OUTDIR |
| M21 | forbidden abstention accepted | A-ABST-DENY |
| M23 | comparisons left out of the family | A-M-ALL |
| M24 | McNemar on unequal budgets | A-UNEQUAL |
| M25 | previously inspected period re-graded | A-R13 |

## 2. Test results (executed today)

| Run | Result | Wall | Peak RSS | Log / artifact |
|---|---|---|---|---|
| Self-test, first run | 23/25. Two **harness** defects: the module loaded by path was not in `sys.modules`, so the fork pool could not pickle; and the "no repo" provenance probe pointed inside the worktree | 10.2 s | 101 MB | `logs/e2e_grader/selftest_run1.log` |
| Self-test after the harness fix (no grader change) | **25/25** | 25.4 s | 110 MB | `logs/e2e_grader/selftest_run2.log` |
| Mutation suite | **22/22 killed and credited**; 21/21 baseline gates pass | 47.3 s | 110 MB | `logs/e2e_grader/mutants_run1.log`; `data/e2e_grader_selftest/mutants_20261123T075131/report.json` |
| Synthetic run-H-shaped scale run (18,068-row synthetic dataset, grade block [14454, 18068), arm A0 = random draws with repeats, arm R = random permutations, 200 reps, 4 workers) | exit 0. Both arms **NULL** (A0 74 hits of 3,614, expected 71.51; R 67 of 3,614, expected 72.28). Uniform controls pass (z −1.15, −1.08). Known-count 1,807 of 1,807 | 22.9 s, 328% CPU | 141 MB | `logs/e2e_grader/scale_run1.log` |

The scale run is a plumbing and timing check on synthetic draws. It says nothing about TFM.

Not run:
- Gate 22 and the S172 suites. The untracked `.py` files under `scripts/diagnostics/` and `tests/` will red Gate 22 until they are landed, which is expected (TASK_COMMON rule 8). The allowlist is not widened.
- Fixture outputs (44 MB under `data/e2e_grader_selftest/`) are untracked and not for commit.

## 3. TODO list

| TODO | Shortcut / open item | Anchor |
|---|---|---|
| E2E-11 | G becomes the WP-3 instrument; the header rules must stay matched to WP-3 and §F | `e2e_grader.py:15-104` |
| E2E-G1 | Reconcile the record format with H6 (Q1) | `e2e_grader.py:106-129`; `e2e_vote_harness.py:494-516` (e2e-vote) |
| E2E-G2 | `upstream_unseen_from_index` is a single declared boundary per arm. Upstream exposure is the operator's declaration; the grader cannot verify it. The sieve's actual window (midday [25, 37)) is covered by the conservative boundary | `resolve_records` `:428` |
| E2E-G3 | With Holm, the reported one-sided lift lower bound still uses α/m (Bonferroni), so it is conservative for Holm's later steps | `run` `:917`, `:976` |
| E2E-G4 | The comparison test is defined only for equal budgets. Unequal budgets report excess-over-expected without a test (Beta question 3) | `compare_arms` `:749-780` |
| E2E-G5 | The dependence diagnostic is lag-1 only. There is no block bootstrap or run-length test | `lag1_autocorr` `:608` |
| E2E-G6 | Pending targets are not checked for completeness, because future draws cannot be enumerated. Only present pending records are reported | `resolve_records` `:519-531` |
| E2E-G7 | The draw-frequency control uses the same k_t as the arm and is not in the multiplicity family. It is a reference line, not a test | `frequency_control` `:702` |
| E2E-G8 | Pending records are not validated against a future dataset when it arrives. A re-grade with a newer dataset gives a new sha, and records bound to the old sha are then rejected as stale. That is intended, but a re-binding procedure is not written | `resolve_records` `:414-417` |
| E2E-G9 | The causal rule is on the combined index only. A real-time rule, "no (d, evening) data for a (d, midday) forecast", is not offered. It matters only for arms graded on both sessions, or whose inputs span both | `resolve_records` `:425` |

## 4. Open questions

**Q1 (lead / owner): the pool record format.** The H6 draft writes `format: "tfm_e2e_pool_v1"` (`e2e_vote_harness.py:75`). The grader reads `schema: "tfm.e2e.pool.v1"`. The fields map as follows:

| Grader field | H6 field | Gap |
|---|---|---|
| `schema` | `format` | name and value differ |
| `arm_id` | `arm` | name |
| `target.combined_index` | `t` | name |
| `target.date` / `target.session` | `draw_date` / `draw_session` (null if H6 was given no identity dataset) | the grader requires them |
| `abstain` (bool) + `abstain_reason` | `status` ∈ {pool, abstain} + `abstain_reason` | shape |
| `pool`, `pool_sha256` | same | same hash (compact JSON of the raw int list) |
| `dataset_sha256` | **absent** | grader requires it (stale-record rule R14) |
| `data_cutoff_index` | **absent** (`history_len` is the full combined length, a different quantity) | grader requires it (R3). For a fixed ranking it should be the end of the data the ranking and target used, e.g. the label-block end |

Options:
- (a) H6 writes the grader schema;
- (b) the grader gains a declared `--pools-format` adapter for H6's shape. The adapter still requires `dataset_sha256` and a cutoff, from H6's run record.

Alpha recommends (a): one format, defined once. The grader is not changed until the lead decides.

**Q2 (owner): defaults to confirm before the first counted grade.** All are declared and overridable:

| Setting | Default |
|---|---|
| α | 0.05 one-sided |
| correction | Bonferroni over arms + comparisons |
| random-control tolerance | 4 SE |
| control replicates | 200 |
| duplicate records | error |
| out-of-block records | error |
| empty pool | error |

**Q4 (owner): a real-time cutoff (TODO-E2E-G9).** Should the grader gain a declared real-time cutoff option, default off, as a flag? It would not reorder anything; the reorder stays cancelled by ruling. If the grade block is midday-only (map Q1), the index rule may be enough. That depends on whether the upstream inputs of each arm include same-date evening draws.

**Q3 (owner), carried from map Q1/Q2:** the grade block, the session filter and the arm list are inputs. The grader cannot be run for a counted result until those are written as `OWNER_*` lines and a run record.
