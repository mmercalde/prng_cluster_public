# X-MINER-SKIP-SWITCH-2 · R2 · BETA REVIEW

2026-11-22 PST · Before-commit review of the Q2 and Q4 close-out · Base read `ab2d330bb5bd74deed716c823f2cfb9efd7fa60c`.

## Prior record

- Checked Alpha's U1–U7 and searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, the exchange, and the wider `docs/` with `rg` for publisher overlap, output roots and backup, checkout-relative Optuna storage, skip bounds and fixed parameters, and study-name routing. Read the Stage 4 runbook, the switch plan and landing notes, and the two R1 reviews covering these cases. Verified `d58e120` locally as the recorded rig status update. `docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R2_ALPHA_REPLY.md:3-29`, `docs/drafts/miner_skip_switch_v1/PLAN.md:258-284`.
- **ALREADY RECORDED** — the same-root overlap and pre-R2 endpoint gap, the Stage 4 run-tree pattern, fresh-only enqueue, the seven-pin one-trial requirement, the accumulator backup gap, and serial miner trials. An enqueued `skip_max` outside a valid distribution can be returned with a warning. The governing storage and routing records omitted from the packet are in Finding 1. `docs/exchange/DEFERRED.md:17-18`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:91-115`, `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1196-1206`, `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:116-130`, `docs/BACKLOG.md:421-448`, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:49-50`, `docs/drafts/SKIP_DESIGN_RECORD_20261121.md:512-516`.
- **NEW** — the two directions of disagreement after WATCHER rewrites the resume flag, the pinned trial's empty distribution with tracked bounds, and the limits of the two new gates. **CHANGES A RULING** — none. The reported D6.2 §4.3 decision is outside this round. `docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R2_ALPHA_REPLY.md:20-25`.
- **Not searched** — rigs, Proxmox, host or ser8 archives, remote repositories, live ledgers, checkpoints, and Optuna stores.

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
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-SKIP-F1_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R1_ALPHA_REVIEW_REQUEST_r3_fixes.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R1_ALPHA_REVIEW_REQUEST_constant_skip_delivery_switch.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-SKIP-F1_R1_ALPHA_REVIEW_REQUEST_skip_range_design_record.md
?? tests/test_miner_skip_switch.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 · Switch off | NO OBJECTION | — |
| Q2 · Scope | SUGGESTION | — |
| Q4 · One rule | OBJECTION | LIVE — With the tracked config, a valid switched one-trial pin of `[251, 255]` passes both preflights and fails in Optuna before the objective. |
| Tests | SUGGESTION | — |

## Finding 1 · Governing records omitted from the prior record

- **ALREADY RECORDED.** The 2026-11-14 owner decision binds Optuna storage to the checkout that imports the module. The cited Stage 4 runbook pairs each fresh output directory with a distinct checkout. The output guard itself uses the process working directory. The two paths coincide when each arm runs from its own checkout root. `docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md:82-87,130-144`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:91-115`, `docs/exchange/from_beta/X-MINER-SKIP-SWITCH-2_R1_BETA_REVIEW.md:49-52`, `window_optimizer_bayesian.py:15-23,673-698`, `window_optimizer_integration_final.py:1857-1859,3776-3783`.
- **ALREADY RECORDED.** The packet cites D3.5 §§7.1b and 7.2. The parent-selection rule is §8b, which makes the validated live `current` the automatic parent. The Step 1 search proposal §2.7 also governs study ownership and resume, which matter when WATCHER changes the inputs to the shared validation rule. `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md:542-560`, `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:248-257`, `utils/run_finalizer.py:1284-1296,1622-1641`, `agents/watcher_agent.py:2284-2289,2312-2319`.

## Q1 · Switch off

The R2 bounds check exits before importing the validator or loading configuration when the switch is off. The serial fallback still loads `SearchBounds.from_config()` at the existing location and passes it to `optimizer.optimize`. Search open short-circuits the new predicate for the off arm. The banner is reached only when marking a switched root. The existing off-arm refusal of a marked root still applies, with revised diagnostic text. No behavior change from the two R2 close-out changes beyond that text. `utils/step1_search_ledger.py:228-251`, `window_optimizer_integration_final.py:1857-1866,1882-1907,2752-2763,3444-3448,3583-3595`.

## Q2 · Scope

The scope paragraph, the two cross-arm refusals and the first-mark banner give the separate-root and one-publisher rule. The paragraph and banner also identify the marker that has to travel with a backup or restore. Put the qualifier in the docstring's opening claim about every generation being switched, and say explicitly in the operator scope that each arm runs from its own checkout root, as the plan already says. The refusals give the root and publisher conditions. The banner is emitted on the first mark, so operators resuming an already marked root rely on the run instructions for the backup requirement. `window_optimizer_integration_final.py:1831-1859,1860-1885`, `docs/drafts/miner_skip_switch_v1/PLAN.md:258-269`, `docs/drafts/miner_skip_switch_v1/READY_TO_LAND.md:87-99`.

**ALREADY RECORDED.** I repeated both publication orders with the actual guard and finalizer. Off passed entry and publication checks in an empty root. On marked the root and passed its publication check. In each order the second publisher's artifact named the first as its parent and contained one inherited row with zero fresh rows. This is the documented same-root limit. Mixed rows prevent arm-level attribution, and the direction of the hit-rate effect is unknown. Measure independently attributable pools on the same unseen draws at comparable budgets, against `k/1000`. `window_optimizer_integration_final.py:1857-1890,3776-3789`, `utils/run_finalizer.py:1284-1296,1622-1641`, `docs/exchange/from_beta/X-MINER-SKIP-SWITCH-2_R1_BETA_REVIEW.md:67-73`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`.

## Q4 · One rule

- **Verified for sampled values.** With the same inputs, both helpers apply the same predicate and range validator. The tracked endpoints pass and the R1 `256/250` case is refused whenever validation runs. For sampled `s = skip_min`, `s ≤ skip_min.max ≤ skip_max.max`, and `skip_max.min ≤ skip_max.max`. Thus `max(s, skip_max.min) ≤ skip_max.max`, with every resulting pair ordered inside `0..255`. The child returns the object it loads, the serial path passes that object through to Optuna, and the check precedes the named-study load. `utils/step1_search_ledger.py:152-173,230-247`, `window_optimizer_integration_final.py:1786-1794,1893-1913,2752-2763,3444-3448,3583-3595`, `window_optimizer.py:440-466,629-636`, `window_optimizer_bayesian.py:549-554`, `distributed_config.json:84-91`.
- **LIVE: the pin still constructs distributions.** A fresh one-trial run enqueues the pin, then its Optuna objective calls `suggest_int` for both skips. In Optuna 4.4.0 with the tracked bounds, the search configuration and child checks both accepted a complete `[251, 255]` pin. The real Optuna objective body used fixed `skip_min=251` and attempted `suggest_int('skip_max', 251, 250)`. It raised `ValueError` before the stub sieve objective was called. `[6, 99]` and `[0, 255]` reached the stub with the exact requested pair. With scratch `skip_min=[5,3]`, both preflights also accepted the one-trial `[6, 99]` pin and Optuna failed on the empty first distribution. The tracked-config failure needs only a supported explicit pin. It produces no switched-arm observation for that trial. Make the one-trial pin path consume the exact pin without constructing the sampled skip distributions, and prove it through the real Optuna trial. The coordinator's per-trial pair validation remains a backstop. `utils/step1_search_ledger.py:152-158,195-215,230-245`, `window_optimizer_integration_final.py:1893-1913,2754-2759`, `window_optimizer.py:1803-1814`, `window_optimizer_bayesian.py:549-576,788-806,824-849`, `miner/range_miner_coordinator.py:2797-2812,12561-12566`, `distributed_config.json:84-91`, `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:116-123`.
- **THEORETICAL with today's tracked sampling domains: WATCHER can pass different inputs.** Search open uses merged params, and WATCHER's dispatch comparison runs before it overwrites `resume_study` from the presence of `study_name`. I executed that assignment and branch with the real validators in scratch. With a full `[6, 99]` pin, one trial, and the R1 `256/250` test bounds, a named study plus `resume_study=false` passed the open helper and failed the child helper after WATCHER set it to true. An empty name plus `resume_study=true` gave the opposite validator verdicts when the rewrite was applied to the same hypothetical dispatch inputs. These probes exercised the bounds verdicts and did not establish study ownership or launch a child. Normalize study routing before the open and dispatch checks so they see the effective resume flag. `agent_manifests/window_optimizer.json:252-259`, `agents/watcher_agent.py:794-809,830-831,865-878,2284-2303,2312-2319`, `utils/step1_search_ledger.py:240-247`, `window_optimizer.py:955-975,1973-1982`, `window_optimizer_integration_final.py:2754-2759`.

## Tests

- `G-SW2-Q4-DOMAINS` covers the R1 endpoint and 71 other combinations by giving both helpers the same inputs. It checks identity on the child's return value and uses source assertions for check order and serial reuse. The gate omits the WATCHER rewrite and never runs an enqueued trial through Optuna. Add the two routing directions and the tracked `[251, 255]` case alongside a working control pin. `tests/test_miner_skip_switch.py:816-887`.
- `G-SW2-Q2-SCOPE` verifies the first-mark banner, the docstring, the one-publisher clause in both cross-arm refusals, and an off call's silence. It runs one empty switched finalization in the overlap fixture. The second finalization and the assertion that it inherits the other arm's row are missing. Add both finalizer calls and assert the resulting parent and inherited row, in both publication orders. `tests/test_miner_skip_switch.py:112-118,599-603,890-929`.
- MS34–MS42 all triggered their assigned gates in the rerun, including the pre-existing MS36. MS34 and MS35 triggered sampling refusals that went missing. MS37–MS39 triggered matrix assertions. MS40 and MS41 triggered source assertions for the bounds reuse and order. MS42 removes the isolation phrase from the banner and triggered the string check. These results cover the specific mutations and the fixture limits above. `tests/test_miner_skip_switch.py:778-813,873-884,903-909,1081-1108,1140-1160`.

## Verification

I ran both existing suites sequentially with `~/venvs/torch` activated in `/tmp/tfm-beta-skip-switch-r2.2MSYTS`. The switch suite passed **29/29 gates** and killed **43/43 mutants**. Stage 3 passed **14/14 gates** and killed **28/28 mutants**. Independently, I ran the actual finalizer with synthetic rows, evaluated WATCHER's extracted resume assignment with the real bounds helpers, and ran the real serial Optuna study body against scratch stores with a stub sieve objective. The probe and output are `beta_probe.py` and `beta_probe.out` under that scratch directory.

I checked the four attachment hashes, matched both complete tracked diffs byte-for-byte, and rebuilt the three tracked R2 files by applying the close-out delta to the R1 attachments in scratch. `git diff --check` passed. Alpha's 19-suite comparison was read only. WATCHER end-to-end execution, rig and GPU execution, wide-range lease and volume behavior, and unseen-draw hit rates are **UNAVAILABLE** in this review.
