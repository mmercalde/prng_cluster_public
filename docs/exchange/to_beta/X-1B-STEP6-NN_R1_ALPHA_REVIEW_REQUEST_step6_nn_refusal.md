# X-1B-STEP6-NN  R1  REVIEW_REQUEST  Step 6 refuses a neural net it cannot normalise as trained (item 1b)

- **Status:** sent 2026-11-26 by the lead on the owner's instruction (19:54 PST "run a Beta round first, as we did for LL-7 and LL-8"; 19:59 "yes- lets get beta view"). One round. The lead wrote the change and has read every changed line.
- **Review class:** BEFORE-COMMIT (the owner asked for the round first; by his 2026-11-11 ruling Step 6 code would be after-commit).
- **Base commit:** `a85adb1` (main).
- **Where to read:** detached worktree `~/tfm_worktrees/1b_step6_nn` at `a85adb1`, with five files as working-tree changes. sha256 (first 16): `models/inference_bundle.py` b4778cd3e1bafd6d  `prediction_generator.py` 84a4dab9485f9af3  `scripts/diagnostics/e2e_vote_harness.py` f1f188e48ecb21f4  `tests/test_e2e_vote_harness.py` e30ec2b2e101aa0e  `tests/test_1b_step6_nn_refusal.py` 3a2c71f14f181c2d (new).
- **Allowed answers per question:** `NO OBJECTION`  `OBJECTION` (with `LIVE` or `THEORETICAL`)  `SUGGESTION`  `NEEDS EVIDENCE`.
- **Scope:** real-world cases on this system as it is today: VM 101, Step 6 run by WATCHER or by hand on a Step 5 sidecar, and the walk-forward vote harness. Unit W is not built and not in scope.

## Prior record

**Searches run** (clone of main at `a85adb1`, 2026-11-26):
```
grep -rn -i "scaler" --include=*.py .            (every non-patch-script hit read in context)
grep -rn "load_nn_scaler|check_nn_normalisation|nn_scaler_applied|G-Q4-SCALER|y_mean|y_std" --include=*.py .
grep -n -i "scaler|best-effort|unscaled" docs/CHAPTER_7_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/DEFERRED.md
sed -n 44,62p docs/exchange/from_beta/X-LL6-BUNDLE_R1_BETA_REVIEW.md
```
**What the record already says:**
- Best-effort was a recorded decision: "if absent (old checkpoints), proceed without transform + warn" (`models/inference_bundle.py:59-61` on main, the Category B Phase 3 text moved from Step 6 by LL-6).
- Your X-LL6-BUNDLE R1 Q4: Step 6 reads the scaler from the path written in the sidecar while the factory may load the model from `models_dir/<name>`; a relocated NN is scored unscaled; "Preserve Step 6's current resolution in LL-6".
- Lead rule 2026-11-23, TODO-E2E-V9 (`scripts/diagnostics/e2e_vote_harness.py:651-680` on main): the walk-forward harness already refuses an NN sidecar without the input scaler, with a scaler of the wrong length, or without `y_mean`/`y_std`. Step 6 itself was not edited then.
- `docs/CHAPTER_7_AUDIT.md:86-87` describes the scaling step; B2 `:200` records the separate case "a missing checkpoint ranks by the feature mean and reports success". B2 is NOT changed here.
- Owner, 2026-11-26 17:15-17:16 (`logs/e2e_session/OWNER_RULING_next_work_20261126.txt`, git-ignored): "why did we feed the NN with unscaled numbers? this is different than normalied data" / "sure- lets make production quality code". 19:59: yes to refusing a missing `y_mean`/`y_std` as well.

**Labels:** the best-effort rule and the relocated-NN gap: ALREADY RECORDED. Step 6 refusing instead: NEW, decided by the owner. Exit code 4 and the `reason` field: NEW implementation details.

**Not searched:** the host outside the repo, ser8 archives, the private repository, the rigs, and which archived NN checkpoints on disk would now be refused (see known limit 3).

## What changed

| Path | Change |
|---|---|
| `models/inference_bundle.py` | `load_nn_scaler` returns one more key, `reason` (None when loaded, else one of six names) `:88-127`; plain-words table `SCALER_ABSENT_REASONS` `:53-62`. It still returns and never raises; logs unchanged. `load_inference_bundle` is not changed, so LL-8's `as_written` replay of past Step 6 runs behaves as before. |
| `prediction_generator.py` | `NNNormalisationError(RuntimeError)` `:245`; exit code 4 `:255`; `_require_nn_normalisation` `:446` called for `neural_net` only after the feature names are known `:393-395`; the refusal passes through `__init__` un-relabelled `:399`; scoring-time block `:866` raises where it used to warn and score raw; CLI returns 4 and writes no prediction file `:1131-1136`. |
| `scripts/diagnostics/e2e_vote_harness.py` | Step 6's refusal is reported as a `HarnessError` `:699-703`; `check_nn_normalisation` kept as a second line; docstring updated. |
| `tests/test_e2e_vote_harness.py` | G13 (GPU mode) now expects Step 6's refusal and CLI exit 4, tests the harness's own check with Step 6's refusal switched off, and adds mutant M26. |
| `tests/test_1b_step6_nn_refusal.py` | new, CPU only: 7 gates, 6 mutants. |

Step 6 now refuses a `neural_net` when: the scaler was not loaded (any of the six loader reasons, including `normalize_features` off and a written checkpoint path that does not exist); the scaler width differs from the sidecar's feature count or, at scoring, from the matrix; the scaler is not finite; `y_mean`/`y_std` are absent, not finite, or `y_std <= 0`. Tree models never reach the check.

## What was run (VM 101, CPU only, `PYTHONDONTWRITEBYTECODE=1 python3 -u -B`)
- `tests/test_1b_step6_nn_refusal.py`: **7/7 gates, 6/6 mutants** (2 runs, 28 s and 32 s).
- `tests/test_ll7_signed_attribution.py`: 16/16 gates, 10/10 mutants. `tests/test_e2e_vote_harness.py` (CPU mode): 15/15 gates, 17/17 mutants.
- **`tests/test_ll6_inference_bundle.py`: STOPS (rc 1)** at `:600` (`step6_in_process(pg, reloc, rows)`): its relocated-NN fixture expects Step 6 to score without the scaler, and Step 6 now refuses it (`NNNormalisationError`, "does not exist at that path"). **`tests/test_ll8_attribution_cycle.py`: rc 1**, same fixture kind (log not yet read line by line by the lead). Neither test file is edited yet: which way they change depends on your Q2. Logs: `~/tfm_worktrees/_proof/1b_logs/dev_*.log`.
- **NOT RUN: `tests/test_e2e_vote_harness.py --gpu`** (G13 and M15/M16/M26 live only in GPU mode). It needs VM 101's card and the owner's go. It will be run in the fresh-copy proof before landing.

## Known limits (stated so you do not have to find them)
1. **A relocated NN directory is now refused by Step 6** (written path missing, model found in `models_dir`). It used to be scored unscaled. The alternative is to read the scaler from the file the model was actually loaded from. I kept the as-written read in this draft. The lead now leans to the resolved read (refuse only when the scaler is truly absent); the owner asked for your view before either is chosen (20:33 PST: "not sure on 1 or 2 we need beta input"). Either way the LL-6 and LL-8 relocated-NN gates must change.
2. **`normalize_features` off is refused**, although such a net was trained on raw inputs and scoring it raw is self-consistent. Reason: owner direction of 2026-11-23 and 2026-11-26, the harness rule, and Step 5 always passing `--normalize-features` for `neural_net` (`meta_prediction_optimizer_anti_overfit.py:281-285`, `:2647`). `train_single_trial.py:494` still defaults the argument to False for direct callers.
3. **Checkpoints from before S121 have no `y_mean`/`y_std` and were trained on raw targets**; the checkpoint carries no flag saying target normalisation was used, so Step 6 cannot tell "never used" from "lost" and refuses both. Which archived NN checkpoints this hits was not inventoried.
4. **No switch.** There is no flag to get the old best-effort behaviour back.
5. **B2 is untouched:** a sidecar whose checkpoint is missing everywhere still falls to the mean fallback (`prediction_generator.py:397-398` `except FileNotFoundError`).
6. `scripts/diagnostics/attribution_cycle.py:48-61` still describes Step 6's scaler read as best effort. That is correct for Step 6 runs made before this lands and wrong after it. The text is not edited here.
7. A stale `next_draw_prediction.json` from an earlier run is not removed when Step 6 refuses.

## Questions

**Q1. The refusal conditions.** Is any real case today (a Step 5 NN sidecar on VM 101, WATCHER or by hand) scored unscaled or un-denormalised after this change? Is any real, correct NN refused that should not be?

**Q2. Relocated NN (known limit 1).** Refuse, or read the scaler from the resolved file? If the second, what does LL-8's G-Q4-SCALER need?

**Q3. `normalize_features` off and pre-S121 checkpoints (known limits 2 and 3).** Is refusing both right for this system, given the owner's direction?

**Q4. The shared loader.** `load_nn_scaler` keeps returning in every case and only gains `reason`. Does anything that calls it or `load_inference_bundle` (Chapter 13 canonical path, LL-7, LL-8) change behaviour?

**Q5. Failure surface.** `RuntimeError` subclass, exit code 4, message to the log and stderr, no prediction file. Can WATCHER or any caller today swallow this and report success (known limit 7)?

**Q6. Tests.** Seven gates, six mutants, CPU; G13/M26 in the harness GPU suite. Any gap that matters on a real run?
