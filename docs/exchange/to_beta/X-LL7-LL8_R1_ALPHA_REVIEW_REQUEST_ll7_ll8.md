# X-LL7-LL8  R1  REVIEW_REQUEST  LL-7 (signed attribution, Q3 tier fix, Finding 2(a)) and LL-8 (attribution cycle) — DRAFT, NOT SENT

- **Status:** sent 2026-11-26 by the lead on the owner's instruction (15:21 PST, "lets go- #1 first"). One round; the lead has read the three code files in full (the two test files not yet).
- **Review class:** BEFORE-COMMIT is proposed. LL-7 changes Chapter 13 code (`chapter_13_orchestrator.py`) and carries
  Beta's own X-LL6 R1 Q3 and Finding 2(a) items. LL-8's G-Q4-SCALER gate answers X-LL6 R1 Q4.
- **Base commit:** `087bc366fb199dbbe26c1fe918a911158618d358` (main). The plan was pinned to `933a4a0`.
  `git diff --stat 933a4a0 087bc36` over every file either unit touches or calls is empty (READY LL-7, "Rebase").
- **Where to read:** detached worktree `~/.local/state/api-agents/worktrees/20261126-104153-build-ll7-ll8-087bc36`, with the
  five files as working-tree changes. Their sha256 values are in `docs/drafts/loop_v1/MANIFEST_LL7_20261126.txt` and
  `MANIFEST_LL8_20261126.txt`.
- **Two separately landable units:** LL-7 first. LL-8 needs LL-7.
- **Allowed answers per question:** `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.
- **Scope:** real-world cases on this system as it is today: VM 101, Chapter 13 root cause under `legacy` (default) and
  `canonical`, and LL-8 run by hand by the lead after a walk-forward block (TODO-WHY-1). Unit W is not built and not in scope.

## Prior record

**Searches run** (worktree at `087bc36`, 2026-11-26):
```
S = docs/PROJECT_FILE_CATALOG.md 'docs/TB_RULING_*' 'docs/PROPOSAL_*' 'docs/CHAPTER_*' 'docs/*_AUDIT.md'
    docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/ CLAUDE.md AGENTS.md 'docs/TFM_SYSTEM_MAP*'
git grep -lE 'LL-7|LL-8|attribution_cycle|signed attribution|per_survivor_attribution_signed|G-Q4|scaler_source|Finding 2\(a\)|compare_pool_tiers' -- $S
  -> 11 files; each hit read in context
sed -n 46-66p docs/exchange/from_beta/X-LL6-BUNDLE_R1_BETA_REVIEW.md ; sed -n 840-902p docs/LEADS.md
git diff --stat 933a4a0 087bc36 -- <files touched or called>  (empty)
```
**Governing documents:**
- Plan `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md` §3-§5, §7-§8 (outside the repository).
- Owner decisions D1-D3 (`logs/e2e_session/OWNER_RULING_ll78_plan_20261126.txt:2-4`, git-ignored log).
- Your X-LL6-BUNDLE R1: Q3 `:48-50`, Q4 `:52-54`, Q5 `:60`, Finding 2 `:65`. Alpha's closing note
  `docs/exchange/to_beta/X-LL6-BUNDLE_R2_ALPHA_CLOSING_NOTE.md:49`, `:123-125`.
- `docs/LEADS.md` L-10 `:843-902` (OPEN; found with branch LL-8 `a30cc50f…`). TODO-WHY-1 `:891-897`.
- `docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:101` and `docs/CHAPTER_14_AUDIT.md:141` (tier comparison blindness).
- Flag format: `why_report_flags.py:97-103`, `training_health_check.py:1225-1259`.

**Labels:** LL-7 and LL-8 design and order: ALREADY RECORDED. Q3 fix: ALREADY RECORDED, decided D1. Finding 2(a): ALREADY
RECORDED. Finding 2(b) and the diagnostics part: ALREADY RECORDED, **deferred** by D2 (not for review here). Required pin:
decided D3. G-Q4-SCALER: ALREADY RECORDED in the plan, NEW relative to your Q4. Runner entry point, flag/family arguments
and exit codes: NEW. `compare_pool_tiers(model_input=)` instead of the plan's `bundle=`: NEW, an implementation detail.

**Not searched:** the host, ser8 archives, the private repository, the rigs, `ch14-learn`'s untracked `data/`, and the
on-disk state of run H's archived NN sidecar.

## Files under review (5)

| Unit | Path | Change |
|---|---|---|
| LL-7 | `per_survivor_attribution.py` | +253 branch block `:319-571`; Q3 `model_input` `:583`, `:646-649` |
| LL-7 | `chapter_13_orchestrator.py` | `_c13_model_input` refuses positional `:89-99`; handoff `:848`; tier input `:868-882` |
| LL-7 | `tests/test_ll7_signed_attribution.py` | new |
| LL-8 | `scripts/diagnostics/attribution_cycle.py` | new (branch `f4a8adbd` + plan §5.2 + entry point) |
| LL-8 | `tests/test_ll8_attribution_cycle.py` | new |

## What was run (VM 101, CPU only, one suite at a time, `PYTHONDONTWRITEBYTECODE=1 python3 -u -B`)
- LL-7 suite, 20 consecutive full runs: **16/16 gates, 10/10 mutants every run.**
- LL-8 suite, 20 consecutive full runs: **22/22 gates, 9/9 mutants every run.**
- LL-6 14/14 + 13/13. ch14 wave1 37 passed, wave2 37 passed. Benching 30 passed. Why flags 32 passed. `switch_check.py` rc 0.
  ll_feature_settings rc 0. Registry suites rc 0.
- `test_ch14_diag_land_defaults.py`: the build agent reported 83/87 with four lightgbm child-call gates red, also on its clean export. **The lead did not reproduce that.** On a fresh clone of main (`aefdd74`, then the same files at `95fa8fa`) the suite passed 86/86 with 9/9 mutants (`--part single`) and 29/29 with 2/2 mutants (`--part compare`), 2026-11-26. The agent's result is put down to its own harness; cause not established.
- One-off check: on a fixture spec, landed LL-8 and branch `a30cc50f…` produce identical reports (new fields aside) and an
  identical attribution matrix.

## Known limits (stated so you do not have to find them)
1. **A-NORM in LL-8** compares the attribution forward with LL-8's own forward (`bundle.predict`), not with Step 6 scores
   written to disk. A spec that names Step 6's prediction file could add that later.
2. **G-Q4-SCALER covers the mismatch Beta measured:** the written NN checkpoint path is missing. When the written path
   exists, the bundle resolves to that same file (`models/inference_bundle.py:131-133`), so the scaler and the pin refer to
   the file Step 6 loads. The gate assumes Step 6's scaler rule stays at `prediction_generator.py:349-366`.
3. **`canonical` stays off real artifacts until LL-9 (D2).** Records are not bound to the checkpoint or schema (Finding 2(b)).
4. **Batch-vs-row equality** is checked at 1e-6 in absolute target units. A ×1.001 scaling mutant survived it and was
   replaced by a units-dropped mutant.
5. The NN integrated-gradients residual is reported, not bounded.
6. **The flag id check is syntactic plus a family match.** LL-8 does not read the flag files, so it does not check that the
   flag exists or is open. `why_report_flags.py` has no `why_report_failed` state today (plan §6.7 needs one for unit W).
7. **Peak RSS on a 12,156-row block was not measured** (~600 MB on the 600-row fixture).

8. **Lead code review, one finding, to be fixed in the fix pass (not yet fixed in the files you read):** `scripts/diagnostics/attribution_cycle.py` sets `os.environ["CUDA_VISIBLE_DEVICES"] = ""` at import time (module top). That was harmless while the script was CLI-only. With the new function entry point, a runner that imports `run_attribution_cycle` in-process (unit W inside WATCHER) would hide the card from the rest of that process and from every child it starts afterwards. Planned fix: set it only for the duration of the cycle (inside `_EnvRoots`, restored on exit) with a gate that the caller's environment is unchanged after a call, and a mutant. Main moved to `95fa8fa` after the build (a test file and exchange records only); none of the five files or anything they call changed.

## Questions

**Q1 (LL-7). Q3 and Finding 2(a).** Under `canonical`, the tier comparison now goes through `_c13_model_input`, and a
positional list is refused even at matching width. Gates: `q3_canonical_tiers_use_bundle_input` (through
`post_draw_root_cause_analysis`, forced regime shift), `finding2a_positional_refused_under_canonical`. Mutants:
`tier_unscaled`, `tier_raw_features`, `handoff_dropped`, `positional_accepted_at_width`. Is this sufficient for your Q3 and
2(a)? Is a real case today (VM 101, a canonical run on a name-keyed `ranked_predictions.json`) still exposed?

**Q2 (LL-7). Defaults.** Legacy root cause, the tier file, the four absolute functions and the Step 6 CLI output are
compared with `087bc36` by `git show` (gates named in the READY note). Is any default path missed?

**Q3 (LL-8). G-Q4-SCALER and its limits 1-2.** Is the gate, with `as_written` as the labelled way through, sufficient for
your Q4 on LL-8's real use (the lead runs it by hand on a block's archived models)?

**Q4 (LL-8). Required pin and wrong-checkpoint refusal.** Pin missing or short → refused. Pin of a same-family,
same-width NN → refused, with a right-pin control. Is that sufficient?

**Q5 (LL-8). Runner interface.** Is `run_attribution_cycle(...)`/the CLI, with `status` ok/refused/error, exit 0/2/3, the
result file, `--model-family` and `--flag-id`, enough for a later runner to call and surface failures without swallowing
them (plan §6.7)? Is anything in it a real-world hazard today, for example where the result is written when out-dir is
refused (`result_written: None`)?

**Q6 (both). Tests after landing.** Baselines by full sha, a no-write boundary with a negative control, mutant credit only
when green unmutated, one in-process mutant (the pinned miss report), 20× repeats. Any gap?

**Q7 (LL-8). The import-time environment finding (known limit 8).** Is the planned fix right, and is there any other process-wide state this module changes at import or leaves changed after `run_attribution_cycle` returns?
