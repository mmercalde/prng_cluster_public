# Selfplay Neural Net: Status and Documentation Summary

TFM | 2026-11-19 | Source: public mirror mmercalde/prng_cluster_public at commit `1e6e675` (2026-11-18) | Prepared by Claude for Michael. Saved for when the project reaches selfplay NN work. Text copy of `TFM_Selfplay_NN_Status_20261119.pdf`, with a review section added at the end.

## Bottom line

The neural net is still blocked from selfplay by a guard whose stated reason is a measurement taken **before** the y-normalization fix. The fix and its result are documented. The ban itself was never revised, and the port of the fix into the selfplay trainer has been an open to-do item since S145.

## Owner's position (Michael, 2026-11-19)

Selfplay should be using a neural net as well. The NN was ruled out for poor results, but that was caused by not normalizing the data. Once normalized, the NN performed at least as well as the other models.

## What the repository records

| Fact | Where |
|---|---|
| Selfplay rejects `neural_net` (and `random_forest`) outright. Comment gives the reason as "500,000x worse MSE". | `inner_episode_trainer.py:71`, `:497-498` |
| Origin of the ban: the NN ranked last on MSE (9.32e-4) for tabular data. | `docs/SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md:17`, `:55`, `:426` |
| The fix: NN output raw values near 18,914 where scores near 0.72 were expected. y-normalization added to training, inverse transform added to the NN wrapper. Commit `6e5f76c`. | `docs/SESSION_CHANGELOG_20260307_S121.md:20-37` |
| The result: NN R2 moved from between -98 and -7,763 to +0.020538, its first positive score. | `docs/SESSION_CHANGELOG_20260307_S121.md:42`, `:178` |
| Root-cause call: the normalization bug was the cause, not the architecture. | `docs/SESSION_CHANGELOG_20260226_S112.md:51` |
| Side by side: NN R2 +0.020538 listed next to a best tree-model R2 of +0.0046 (S111 clean baseline). | `docs/COMPLETE_OPERATING_GUIDE_v2_0.md:885-888` |
| Step 5 already trains the NN alongside the three tree models. | `meta_prediction_optimizer_anti_overfit.py:858-859` |
| Open item "Selfplay NN enablement": remove the guard and port y-normalization. Status REVIEW. Says reinstatement is conditional on gates NN-Q and NN-A (SM section 12.4). | `docs/TODO_CONSOLIDATED.md:117` |

## What is stale or missing

- **The ban's source documents were never revised.** `SELFPLAY_ARCHITECTURE_PROPOSAL_v1_0.md` and `CHAPTER_13_IMPLEMENTATION_PROGRESS_v1_6.md:150` still say FORBIDDEN, citing the pre-fix measurement.
- **The operating guide contradicts itself.** It says "Best model: CatBoost" three lines above the higher NN number.
- **The selfplay port never happened.** It appears on the to-do lists for S145, S147, S148, S150, S159, S162, S163 and S170, and is still open.
- **No same-run comparison was found.** The two R2 figures come from different sessions (S111 and S121). No single run comparing all four models after the fix turned up in this search.
- **The comparison metric has since changed.** The ban and the fix were both judged on MSE / R2. The project's yardstick is now pool hit rate against k/1000 on draws the sieve never saw. No NN-versus-trees result on that measure was found.

## Loose end

`docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:594` records that the NN's "critical" flags in training diagnostics were attributed to y-normalization in S111/S121, while the actual cause is R-60, a key-name mismatch (`best_round` read, `best_val_round` written; same report, line 348). That concerns the diagnostics flags, not the R2 result above.

## Work that would close this

The first two items are taken from the existing to-do entry. The last three are suggestions, not rulings.

- Remove the neural_net guard in `inner_episode_trainer.py` (`:497-501`).
- Port y-normalization into the selfplay training path, matching the S121 fix in `train_single_trial.py`.
- Settle gates NN-Q and NN-A, or have Michael rule on whether they still apply.
- Run all four models in one run after the fix, scored on hit rate as well as R2, so the comparison stands on one dataset and one session.
- Correct the stale statements in the selfplay proposal, the Chapter 13 progress document and the operating guide.

## Scope and limits

- **Searched:** the public mirror only, code and `docs/`, at commit `1e6e675`.
- **Not searched:** the private repository, live state on VM101, gitignored files, and the pre-repository archive on SER8. A same-run comparison or a later ruling could exist on any of those.
- **Not verified here:** the content of gates NN-Q and NN-A. They are cited only as the to-do entry describes them. (Now located; see the review below.)
- Nothing in this summary was measured today. Every figure is quoted from the repository documents named beside it.

## Review added 2026-11-19 (Claude, Team Alpha)

Citations were re-checked against the public mirror at `1e6e675`. The guard, the to-do entry, the S121 fix and result, the S112 line, the operating guide lines, the Step 5 model list and the R-60 lines all match as cited. The selfplay proposal and Chapter 13 progress citations were not re-checked.

Four points to carry forward:

1. **The +0.020538 figure is a smoke-test result.** It is listed under "Smoke test results" at `docs/SESSION_CHANGELOG_20260307_S121.md:38-42`. The claim that the NN does at least as well as the trees rests on one quick run set against a tree figure from a different session.
2. **Gates NN-Q and NN-A are defined in the repo.** `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:237` (also `docs/SESSION_CHANGELOG_20260728_S1.md:73`) records Team Beta's position: the NN must pass a quality gate (NN-Q) and also show attribution value beyond tree SHAP (NN-A). This is Beta's condition; whether it still applies is Michael's ruling.
3. **The S112 "root cause" line predates the fix.** S112 is dated 2026-02-26; the fix landed 2026-03-07 (S121). It is a diagnosis that the problem was y-scaling, not a confirmation made after the fix.
4. **Order of work.** Removing the guard before y-normalization is ported into the selfplay trainer would reproduce the original bad result. Port first, or do both in one change.

The measurement that would settle the question: all four models in one run after the fix, on one dataset, scored on pool hit rate against k/1000 on unseen draws as well as R2.
