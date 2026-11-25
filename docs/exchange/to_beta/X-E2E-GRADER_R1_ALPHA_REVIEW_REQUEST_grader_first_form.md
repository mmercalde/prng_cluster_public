# X-E2E-GRADER · R1 · REVIEW_REQUEST · hit-rate grader, first form of the WP-3 instrument

> **DRAFT. NOT SENT.** Written by a Claude Code session in worktree `e2e-grader`. The lead or
> the owner decides whether and when it goes through `scripts/beta_relay.py`. The files under
> review are **untracked** in that worktree. Before sending, they must be readable at a commit
> Beta can name, or Beta answers `NEEDS EVIDENCE`.

- **Review class:** BEFORE-COMMIT. WP-3 says "Beta reviews before commit"
  (`docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:86`). Once counted grades use this
  grader, it becomes the verifier for runs that count (CLAUDE.md, "Beta reviews").
- **Base commit:** `41ad509` (+ the untracked files below).
- **Rounds:** one round is requested (the relay allows three).
- **Files under review:**
  - `scripts/diagnostics/e2e_grader.py` (sha256 `a4d96fe7ac441a8befc93f55a638eb70e36e7d0cd70f24eaeb60d08b7a52169a`)
  - `tests/test_e2e_grader.py` (`f2b679fbafc6acf13f8444b699f11d1c7f8ef8ecc3cefd5f2c7e052a58136bbc`)
  - `tests/mutate_e2e_grader.py` (`206041f11b84e6f2efca721f54d65c1069c48f08302bfa401372bd2b1ec8e191`)
  - Context: `docs/drafts/e2e_v1/READY_E2E_GRADER_20261123.md`
- **Allowed answers per question:** `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.

## Prior record

**Searches run** (worktree at `41ad509`, `/bin/grep -rlniE`).
- **Scope S:** `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`.
- **Patterns:**
  - `e2e_grader|hit[-_ ]rate instrument|WP-3`;
  - `poisson[- ]binomial|mcnemar|bonferroni|holm`;
  - `grade block|label block`;
  - `frequency control|most frequent|draw-frequency`;
  - `uniform[- ]random (pool|control)`;
  - `abstention|abstain`;
  - `pending.{0,20}miss`.
- Code search: `poisson.?binom|k_t` over `*.py`.

The full table is in the READY note's Prior record.

**Found:**
- WP-3 (`PROPOSAL_FINDINGS…_v1_1.md:78-86`);
- addendum §E (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:115-128`) and §F (`:130-135`);
- your R2 §4 (`docs/exchange/from_beta/X-REVERSE-SIEVE_R2_BETA_REALIGNMENT.md:112-127`);
- your X-FINDINGS-PLAN R2 Q5 omissions and suggested §F text (`docs/exchange/from_beta/X-FINDINGS-PLAN_R2_BETA_REVIEW.md:75-85`), deferred to WP-3 (`docs/exchange/DEFERRED.md:8`);
- S176 Phase C (`docs/TB_RULING_S176_WATCHER_KPI.md:692-727`) and §4.1 (`:309-333`).

No grader exists in the repository (`scripts/diagnostics/` was absent at `41ad509`). `evaluate_pools.py:28-50` takes k from the pool's label.

**Labels:**
- The instrument and its rules: **ALREADY RECORDED** (WP-3, §F). This is its implementation.
- Your Q5 items: **ALREADY RECORDED** (your text). They are adopted as header rules R2, R10, R12, R13, R19.
- **NEW:** the λ test-inversion interval, the runtime known-count control, and the record-format gap with the parallel H6 harness.
- **CHANGES A RULING:** none.

**Not searched:** ser8, the private repository, project files outside the repository.

## What the grader does (one paragraph; the header `e2e_grader.py:15-129` is authoritative)

The grader reads five inputs:
- one JSONL of per-draw pools;
- a dataset bound by sha256;
- a grade-block file;
- an arm list;
- an abstention rule.

For each declared arm and each eligible grade-block draw it:
- requires exactly one record (missing is an error, duplicates follow a declared rule);
- checks the target identity against the dataset row, `config_id` against the arm, `dataset_sha256` against the bound dataset, `data_cutoff_index ≤ target`, and `target ≥ upstream_unseen_from_index`;
- sets `k_t` = distinct ints in [0, 999] and records one membership hit.

Under the uniform null it computes:
- the exact Poisson-binomial `P(X ≥ H)`;
- a λ interval by test inversion;
- a Bonferroni (or Holm) correction over every declared arm test plus every declared comparison.

Controls run on every invocation:
- uniform-random pools of the same k_t, through the same code;
- a top-k_t draw-frequency control from a declared block before the first graded draw;
- a known-count control built on the real outcomes.

Results are reported nulls-first, with every denominator. The self-test passes 25 of 25 gates. 22 of 22 mutants are killed by their credited assertion, with applied, ran and credited each proven.

## Questions, focused on real-world cases

**Q1. Upstream exposure is a declaration, not a measurement.** `upstream_unseen_from_index` is per arm (`e2e_grader.py:428`). The planned run-H arms:
- **A0:** Step 1 vehicles only, graded on the whole holdout [14454, 18068).
- **A1:** Step 5-ranked, graded only on a grade block after a label block that Step 3 builds `holdout_quality` from (map §2.1 row 8, summary item 5).

Is one index boundary per arm enough to enforce your R2 §4 "unseen by training, ranking-target construction, feature fitting, Optuna selection, threshold tuning and configuration selection"? Or must the grader also bind each arm to a run record listing the history files each stage read, with their sha256, and refuse when a graded index falls in any of them? Concrete case: the Step 3 `--holdout-history` file for A1 is a label-block file. Should the grader verify that file's contents against the dataset slice?

**Q2. The data cutoff for a fixed-ranking vote.** H6 ranks once and votes at `next_idx = t` for every t (`prediction_generator.py:836-850`, as cited in map §2.4). The vote itself reads no draw values, so the forecast for t depends only on what the ranking and the target saw. The grader takes `data_cutoff_index` from the record, checks only that it is ≤ t (R3, `:425`), and does not check it against `upstream_unseen_from_index`.

Is that adequate for S176 Phase C (`:720-727`)? Or should the grader require `data_cutoff_index ≤ upstream_unseen_from_index`, so that a record cannot claim a later cutoff than the arm's declared exposure?

**Q3. Comparing arms with unequal k_t.** Step 6 keeps repeats (`prediction_generator.py:836-850`; Ch7 B3). H6 pools therefore have distinct k_t ≤ 20 that varies by draw and differs between arms. `compare_arms` (`:749-780`):
- reports both Σk_t and the excess over expected;
- runs an exact one-sided McNemar only when k_t is equal on every common draw;
- otherwise reports "not tested: unequal budget".

Two ways to give a calibrated paired test under unequal budgets:
- (a) truncate or pad pools to a declared common k before grading. This changes the pool that is graded;
- (b) a conditional test on the paired residuals (h_t − p_t) with a permutation or exact null.

Which, or is "not tested" with the accounting the right first form?

**Q4. The uniform-random control's tolerance.** The control passes when:
- |mean − E| ≤ 4 SE over R = 200 replicates, **and**
- the empirical rejection rate ≤ exact size + 4 SE (`uniform_random_control` `:668-699`).

On a run-H-shaped synthetic fixture (3,614 draws, k ≈ 20) it gave z −1.15 and a rejection rate of 0.010 against a size of 0.024. Is a 4-SE tolerance with 200 replicates a defensible "declared tolerance" (§F `:135`)? Or should the declaration be a power statement: the smallest per-draw bias in k that the control is guaranteed to detect? Sensitivity depends on n (computed, not run):
- A pool one value short on each of n draws moves the mean by n/1000 hits.
- The SE of the mean is √(Σp(1−p)/R).
- On 3,614 draws at k = 20 and R = 200 that is 3.6 hits against an SE of 0.60, about 6 SE, so it would be detected.
- On a 200-draw block it is 0.2 against 0.14, about 1.4 SE, so it would not.

The self-test's mutant M13 (pool k−1) is credited at k = 2 on 3,000 draws.

**Q5. A draw that has an outcome in a newer dataset.** Pending targets (index ≥ dataset rows) are reported and not graded (R16, `:420-443`). When a newer frozen dataset arrives, its sha256 differs, so every record bound to the old sha is rejected as stale (`:414-417`). That is fail-closed. It also means a frozen pool written before its outcome existed can never be graded without being rewritten.

Should the grader accept a declared "dataset extension" binding instead: the old sha plus proof that the new file's prefix is byte-identical to the old one? That would let a prospective forecast be graded unchanged.

**Q6. Draw identity and real-time order.** The combined index is the dataset row index (`window_optimizer.py:1159-1171` builds train and holdout as `full_history[:split]` and `[split:]`). The grader checks (date, session) at that index, and its causal rule R3 is `data_cutoff_index ≤ t` on that index (`:425`).

Read today from run H's `daily3.json` (sha256 `513648160d35…`): 18,068 rows, no duplicate (date, session). From row 1039 on, the evening row of a date sits **before** that date's midday row, in 8,514 of 18,067 adjacent steps.

This ordering is **ALREADY RECORDED**:
- the intra-day key is raw `(date, session)`, evening first (`docs/CHAPTER_2_SOURCE_MAP_v1.md:203-205`);
- the binding ruling: per-session order is normative, the combined order carries no PRNG-advance meaning, and the chronological reorder was cancelled (`docs/BACKLOG.md:105-120`).

The **NEW** part is the consequence for grading. Take a midday target at row t whose date has its evening row at t−1. An index cutoff of t lets that forecast use the evening outcome, which is drawn later in real time.

Should the grader offer a declared real-time cutoff rule, so that no forecast for (d, midday) may use (d, evening)? Or is the index rule enough because arms are graded per session (the grade block's `sessions`)? This does not ask to reorder the dataset; that migration is cancelled by ruling.

**Q7. The multiplicity family.** The default family is every declared arm test plus every declared comparison (`:914-917`). A comparison that is "not tested" (unequal budget) still counts in m. The draw-frequency control and the uniform control are not counted.

Is that the right family for your R2 §4 "repeated comparisons"? Should arms added later, such as the Q2 position arms (map Q2), be required to declare a new, untouched grade block (R13 `previously_inspected_blocks`), rather than enlarging m on the old one?

**Q8. Anything a real Step 6 output would break that the fixtures do not cover.** Examples we considered and handled:
- repeats;
- 000;
- bool, float and str values;
- an empty pool;
- a dict-shaped pool;
- a missing file;
- a blank line;
- a wrong schema;
- duplicate target records;
- a future cutoff.

The open gap: the parallel H6 harness writes a different record shape and omits `dataset_sha256` and `data_cutoff_index` (READY note Q1). Any further real-world case you see?

## What was run

All of it on VM101 CPU, synthetic fixtures only. Logs are in the worktree under `logs/e2e_grader/`.

| Run | Result | Wall | Peak RSS |
|---|---|---|---|
| Self-test | 25/25 | 25.4 s | 110 MB |
| Mutation suite | 22/22 killed and credited | 47.3 s | — |
| Run-H-shaped synthetic scale run | both arms null, controls pass | 22.9 s, 4 workers | 141 MB |

No run H pools were graded and no pipeline step was run. Gate 22 was not run; the untracked `.py` files will red it until landed.
