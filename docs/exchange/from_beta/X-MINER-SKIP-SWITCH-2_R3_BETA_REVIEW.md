# X-MINER-SKIP-SWITCH-2 · R3 · BETA REVIEW

2026-11-22 PST · Before-commit review · Base read `ab2d330bb5bd74deed716c823f2cfb9efd7fa60c`

## Prior record

- Checked Alpha's V1–V5 and R2-F1. Searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, the exchange, and the wider `docs/` with `rg` for exact pins, fixed Optuna parameters, study-name routing, same-root publication, and mutation verification. Read the relevant sections of the switch plan, the Stage 4 runbook, and the previous Beta review. `docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R3_ALPHA_REPLY.md:3-25`, `docs/drafts/miner_skip_switch_v1/PLAN.md:302-316`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:91-115`.
- **ALREADY RECORDED** — all seven values are required for the one-trial Run 4 pin, warm start is fresh-study only, WATCHER derives resume from the study name, studies have checkout and search ownership, and the finalizer uses the live `current` parent. The same-root overlap is deferred. The verification requirements omitted from the request are in Finding 1. `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:116-130`, `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1196`, `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:119-130`, `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:248-257`, `docs/exchange/from_beta/X-MINER-SKIP-SWITCH-2_R2_BETA_REVIEW.md:58-61`, `docs/exchange/DEFERRED.md:17`.
- **NEW** — the exact-pin handoff and single-point distributions, search open's effective resume input, and the limits of the new gate fixtures. **CHANGES A RULING** — none in this review. The packet records Michael's separate D6.2 §4.3 decision. `docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R3_ALPHA_REPLY.md:41-42,55-74`.
- **Not searched** — rigs, Proxmox, the host or ser8 archives, remote repositories, live ledgers, checkpoints, and Optuna stores.

Initial `git status --porcelain`

```text
 M agent_manifests/window_optimizer.json
 M docs/exchange/DEFERRED.md
 M docs/exchange/LEDGER.md
 M miner/range_miner_coordinator.py
 M scripts/extract_miner_bundle.py
 M scripts/survivor_volume_gate.py
 M tests/test_extract_miner_bundle.py
 M tests/test_pwc_backup_stage3.py
 M tests/test_s172_mp1_drain_attribution.py
 M tests/test_s172_r1_drain_remedy.py
 M utils/step1_search_ledger.py
 M utils/survivor_volume_estimator.py
 M window_optimizer.py
 M window_optimizer_bayesian.py
 M window_optimizer_integration_final.py
?? data/
?? docs/drafts/SKIP_DESIGN_RECORD_20261121.md
?? docs/drafts/miner_skip_switch_v1/
?? docs/drafts/pwc_backup_stage4_v1/PARKED_MINER_PHASE_CAPTURE.md
?? docs/drafts/pwc_backup_stage4_v1/SCOPE_MINER_PHASE_SETS.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH-2_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH-2_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-SKIP-F1_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R1_ALPHA_REVIEW_REQUEST_r3_fixes.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R1_ALPHA_REVIEW_REQUEST_constant_skip_delivery_switch.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-SKIP-F1_R1_ALPHA_REVIEW_REQUEST_skip_range_design_record.md
?? tests/test_miner_skip_switch.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| 1 · Q1 · Switch off | NO OBJECTION | — |
| 2 · Q4 · Exact pin | NO OBJECTION | — |
| 3 · Q4 · Resume flag | NO OBJECTION | — |
| 4 · Tests | SUGGESTION | — |

## Finding 1 · Governing verification records omitted from the prior record

**ALREADY RECORDED.** VIR-2 governs the mutation claims and requires evidence that the intended detector caught the injected change. EXEC-PIN-1 governs historical executable comparisons when live dependencies change. The switch runner counts any exception from the named gate as a kill. I checked that MS43–MS47 reached the intended assertions. I also ran the off controls against a separate clean archive of the base commit because the in-suite HEAD arm imports live `SearchBounds`. `docs/VERIFICATION_INTEGRITY_STANDARD.md:37-66`, `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:86-98`, `tests/test_miner_skip_switch.py:994-1018,1044-1057,1337-1354`.

## 1 · Q1 · Switch off

The child helper returns `None` before its imports, so the ordinary context carries no exact-pin key and Optuna executes the original two skip suggestions. Search open short-circuits before reading `study_name` for the new sampling rule. I compared the pinned `[6, 99]` control and three seeded unpinned trials against the clean base archive with the actual tracked configuration. All four trial configurations and parameter dictionaries matched. The existing off-arm miner call gate also passed. These checks cover the R3 delta and the tested call boundary. `window_optimizer_integration_final.py:1893-1909,3587-3620,2186-2187,2203-2209`, `window_optimizer_bayesian.py:549-569`, `utils/step1_search_ledger.py:228-249`, `tests/test_miner_skip_switch.py:229-242`.

## 2 · Q4 · Exact pin

The R2 LIVE objection is resolved on the supported serial route. The child requires all seven pins, one trial and a fresh request for the exact-pin path, validates the pair, and forwards the context to the serial optimizer. A fresh study enqueues the same integers the objective then requests as `suggest_int(name, value, value)`. This holds for every allowed pair `0 <= skip_min <= skip_max <= 255` and keeps both keys in the trial's parameters. The sampled skip ranges are bypassed, including the previous dependent lower bound for `skip_max`. `miner/range_miner_coordinator.py:2801-2812`, `utils/step1_search_ledger.py:152-158`, `window_optimizer_integration_final.py:1718-1723,1893-1909,2771-2781,3587-3620`, `window_optimizer.py:462-466,629-636`, `window_optimizer_bayesian.py:449-463,555-569,803-828`.

Window size, offset, session index, and both thresholds still use their existing distributions. Their endpoints are independent of the skip pair. The tracked endpoints are ordered, session indices have a populated domain, and the optional TRSE ceiling uses a lower bound of `min_window_size + 1`. This rules out an empty non-skip distribution caused by changing the skip pin. I verified `[0, 0]`, `[6, 99]`, `[251, 255]` and `[255, 255]` through the real TPE study body with a stub objective, using the tracked config and checking the trial parameters, both single-point distributions and the COMPLETE state. With only the scratch `skip_min` domain inverted to `[5, 3]`, `[6, 99]` also completed with the key. The controls without the key reproduced both empty-range errors before the stub. `window_optimizer_bayesian.py:537-591,642-649`, `window_optimizer.py:179-186`, `distributed_config.json:72-101`.

## 3 · Q4 · Resume flag

Yes, for the supported WATCHER Step 1 dispatch routes. Search open and pre-dispatch use the same presented-configuration function. That function now uses `bool(study_name)`, matching the later continuity block. A valid nonempty name emits a name and the resume flag. An empty name omits both and the child parser defaults to fresh. Pipeline retries call the same `run_step`. I executed the extracted WATCHER continuity and argument-serialization blocks and checked them against the child bounds helper with the `skip_min.max=256 / skip_max.max=250` fixture. Named plus raw false made both helpers refuse. Empty plus raw true made both accept. `agents/watcher_agent.py` has no diff against the base. These are verdicts about whether the requested route can sample. Actual study loading and ownership have their own checks. `agents/watcher_agent.py:794-809,830-831,865-878,2284-2303,2312-2319,2412-2428,3101-3115`, `agent_manifests/window_optimizer.json:34-36,252-262`, `utils/step1_search_ledger.py:237-249`, `window_optimizer.py:1549-1556,1966-1982,955-966`, `window_optimizer_integration_final.py:1724-1735,1925-1932,2776-2783`, `window_optimizer_bayesian.py:691-742`.

## 4 · Tests

- **Exact pin.** The gate runs the real study body, reproduces the former failure, and checks the exact pairs reaching the stub. It changes into an empty scratch directory before loading bounds, so it uses fallback bounds. The fallback skip endpoints equal the tracked ones today. The asserted success checks the observed pair and can still pass if the study raises after invoking the stub. Load the tracked config by explicit path, then assert successful completion and the recorded single-point distributions. Add the inverted-domain case from R2 if the gate is intended to retain that reproduction. The independent probes above covered those cases in this review. `tests/test_miner_skip_switch.py:994-1026,1038-1060`, `window_optimizer.py:67-92,159-177`, `distributed_config.json:84-91`.
- **Resume flag.** Both raw-flag directions are covered. The fixture uses `skip_max.max=256`, where R2 used `skip_min.max=256 / skip_max.max=250`. Both distinguish the flag the predicate saw. The text assertion checks for the continuity block and the gate supplies the child flag directly. An argument-serialization check would also catch a later overwrite or a dropped flag. `tests/test_miner_skip_switch.py:826-840,1080-1110`, `agents/watcher_agent.py:2312-2319,2412-2428`.
- **Overlap.** Both publication orders call the real finalizer twice. The gate asserts the parent ID and output counts. My scratch probe also checked raw, winner and prior counts and compared the contents of every NPZ array in the two artifacts, confirming that the second publisher inherited the first row in each order. Add those assertions to make the gate enforce the row claim. The overlap remains the recorded same-root limit. Mixed artifacts prevent arm-level attribution, and the direction of any hit-rate effect remains unknown. The comparison to settle it uses independent pools on the same unseen draws, at comparable budgets, against `k/1000`. `tests/test_miner_skip_switch.py:933-981`, `utils/run_finalizer.py:1622-1660,1743-1746`, `docs/exchange/DEFERRED.md:17`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`.
- **MS43–MS47.** MS43 hit the helper return-value assertion. MS44 brought back the `251 > 250` error before the stub. MS45 hit the source assertion for the conditional handoff. MS46 delivered the pin twice and hit the trial-1 assertion. MS47 hit the named-study parity assertion. Each mutation failed for the named reason in the rerun. `tests/test_miner_skip_switch.py:1046-1073,1099-1108,1285-1299,1337-1354`.

## Verification

I ran the existing suites with `~/venvs/torch` activated in `/tmp/tfm-beta-skip-switch-r3.8z3Y29`. The switch suite passed **32/32 gates** and killed **48/48 mutants**. Stage 3 passed **14/14 gates** and killed **28/28 mutants**. The independent checks and outputs are in `beta_probe.py`, `beta-probe.out`, and `off_control.py` in that scratch directory. The four attachment hashes matched, both complete tracked diffs matched byte for byte, and the four-file delta replayed from the R2 snapshot with zero fuzz and matched the files reviewed. `git diff --check` passed.

Alpha's 20-suite comparison was read only. WATCHER end-to-end execution, GPU execution, wide-range lease and survivor-cap behaviour, and unseen-draw hit rates are **UNAVAILABLE** in this review.
