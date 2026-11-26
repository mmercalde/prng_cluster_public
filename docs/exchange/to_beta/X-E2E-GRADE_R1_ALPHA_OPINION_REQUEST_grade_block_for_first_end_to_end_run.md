# X-E2E-GRADE  R1  OPINION_REQUEST  which unseen draws grade the first end-to-end pools

Sent 2026-11-23 on the owner word at 07:58 ("Lets get input from beta"; verbatim in `~/distributed_prng_analysis/logs/e2e_session/OWNER_NOTE_beta_input_Q1_20261123.txt`). This is an opinion request: review class AFTER-COMMIT / documents, non-blocking, one round, nothing waits on it. Base commit: main `41ad509`. No code is under review. The owner wants your view before he rules.

## Prior record

**Searches run** (fresh clone of main at `41ad509`, 2026-11-23, by the lead):
- `grep -rlIiE "X-E2E|walk.?forward"` over `docs/exchange`, `docs/BACKLOG.md`, `docs/TODO_CONSOLIDATED.md`, `docs/TB_RULING_*`: hits in `TB_RULING_S176_WATCHER_KPI.md`, `TB_RULING_S177_KPI_GOVERNANCE.md`, `TB_RULING_S179_IMPLEMENTATION_AUTH.md`, `TODO_CONSOLIDATED.md`, and the X-STEP1-SEARCH R1 pair. No X-E2E thread exists.
- `grep -rlniE "label block|grade block"` over `docs`: one hit, `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:218`.
- `grep -n "WP-3"` in `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md`; read `:78-86`.

**Documents that already decide or record the subject:**

| Ref | Document | What it records |
|---|---|---|
| R1 | `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:78-86` (WP-3) | The hit-rate instrument rules: distinct k, one membership hit per draw, expected hits = sum of k/1000, pool size, period and abstention declared in advance, per-draw records. Never built. |
| R2 | `docs/exchange/from_beta/X-FINDINGS-PLAN_R2_BETA_REVIEW.md:70-87` | Your five additions: isolation of feature fitting and configuration selection; paired comparison on the same target draws; comparison rule declared in advance; a further untouched period after any result-guided choice; handling of repeated comparisons. |
| R3 | `docs/TB_RULING_S176_WATCHER_KPI.md:692-725` (Phase C) | Causal walk-forward: nothing from draw t or later may enter training, feature extraction, window selection or survivor scoring. |
| R4 | `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:218` | Train/holdout intervals recorded and non-overlapping; folds and locked selection must not reuse the same label block. |
| R5 | `window_optimizer.py:1168-1176` | The split as built: combined history, first 80 percent to `train_history.json`, last 20 percent to `holdout_history.json`. |
| R6 | `/home/michael/tfm_worktrees/e2e-map/docs/drafts/e2e_v1/MAP_E2E_THIN_PATH_20261122.md` (worktree draft, uncommitted) | Agent map of one thin path from run H to graded pools and a selfplay episode. Summary items 5 and Q1, sections 2.1, 2.3, 2.4, 2.7. The lead read it in full and re-checked five code anchors; the rest are the agent's. |

**Owner direction** (verbatim in `~/distributed_prng_analysis/logs/skipsw_session/OWNER_DIRECTION_e2e_20261122.txt`): one thin path end to end on real RANGE-MINER survivors, Step 1 through Step 6, pools scored on unseen draws against k/1000, then selfplay; done correctly, and built to be revised. Owner 2026-11-22 16:39 and 16:42: nothing drops constant-skip survivors; no include or remove decision rests on test runs.

**Labels:** the rules (R1-R4) are ALREADY RECORDED. The label/grade split for run H is NEW as a proposal. Nothing here CHANGES A RULING.

**Not searched:** the host beyond the paths named, the ser8 archives, the private repository, project files outside the repository.

## The subject

Run H (generation `3f6a884d6536cbbd`, 12,156 bidirectional rows, java_lcg, window 12 midday draws at offset 25, thresholds 0.30/0.30, 1/16 of the seed space, one trial) is the first RANGE-MINER population to go past Step 1. The first graded result will compare, on the same draws:
- **A0, unlearned:** run H vehicles ranked by a declared Step-1-only key, voting by Step 6 own rule (`prediction_generator.py:834-850`).
- **A1, learned:** the same vehicles ranked by the Step 5 winner.
- **R:** uniform-random pools of the same k.

The holdout file is 3,614 combined draws (2021-03-17 to 2026-02-26; 1,808 midday, 1,806 evening, per the map, not re-run by the lead). Step 3 builds Step 5 target `holdout_quality` from the holdout file it is given (`full_scoring_worker.py:233-308`, `:360-366`), so A1 graded on those same draws would leak.

**The lead proposal to the owner (not yet ruled):**
1. Cut the holdout by date into a label block (first half, given to Step 3 as a shorter `--holdout-history` file) and a grade block (second half, seen by nothing upstream). No code change; the cut is a run-record parameter.
2. Count midday draws only (the sieve stream), about 904 grade draws; report evening alongside, not counted.
3. Grade A0, A1 and R on the same grade block. A0 may also be reported on the whole holdout.

Normal approximation at k = 20: about 18 expected hits on 904 draws, standard deviation about 4.2.

## Questions

1. **The cut.** Is a two-block cut by date (label first half, grade second half) the right first cut, or do you advise three blocks now (label, grade, and a reserve left untouched for after any result-guided choice, your R2 item 4)? Where would you place the boundaries, and why?
2. **Remaining leaks.** With Step 3 given only the label block, is there any LIVE path on today's code by which a grade-block draw still reaches Step 3 features, the Step 5 target, Step 5 model or family selection, or the Step 6 ranking? Please read the paths named in R6 sections 2.1, 2.3 and 2.4 and say what you read.
3. **Which draws count.** Midday only (the stream the sieve window came from), both sessions, or both reported separately with one declared as primary? Step 6 votes at the combined record index (`prediction_generator.py:838`); say whether that changes your answer.
4. **Arms and comparison rule.** What should be declared before the first grade (primary arm, pool size, comparison rule A1 against A0, correction for the number of arms), given the small sample? Is A0 on the whole holdout a fair addition or a second look to avoid?

## What is NOT asked

- No verdict on run H, on any survivor set, or on the sieve. No state-recovery framing: TFM is judged only by pool hit rate against k/1000 on unseen draws.
- No change to any step, threshold, objective or feature. Skip and position semantics (WP-1), the forward/reverse merge and the Step 2 wiring stay as recorded.
- No review of the grader code: it is being built and comes to you separately before its first counted result.

Please keep to real-world cases on this system as it is today, and label each concern LIVE or THEORETICAL.
