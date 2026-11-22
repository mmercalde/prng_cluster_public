# X-SKIP-F1 · R1 · Beta review · 2026-11-21

## Prior record

Document review and advice for Michael. Base read `6d75f71d7b637c7a329429aef3b0ba29a806a355`. The attached design record matched Alpha's SHA-256. `git status --porcelain` at the read returned:

```text
?? data/
?? docs/drafts/SKIP_DESIGN_RECORD_20261121.md
?? docs/exchange/to_beta/X-SKIP-F1_R1_ALPHA_REVIEW_REQUEST_skip_range_design_record.md
```

I read the attachment in full. I used `rg` for F-1, skip defaults and bounds, Run 4, leases, and scoring across the rulings, proposals, chapter audits, consolidated TODO, exchange and source. I reran the `git log --all -S'skip_range'` search for the coordinator and the miner directory. I did not search external archives, live ledgers, rig logs or other hosts.

| Label | Record |
|---|---|
| ALREADY RECORDED | **Finding 1, governing record omitted from the request.** `CLAUDE.md:223` and `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md:150-155` record Run 4's credit. `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md:47-55` limits the comparison because the seed intervals differ. `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154-179` limits the acceptance claims. |
| ALREADY RECORDED | F-1 and the owner's direction to fix it. `docs/TFM_SESSION_HANDOFF_20261114.md:76-80`, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47-52`. The current baseline and evaluation rule are in `AGENTS.md:19-33`, `CLAUDE.md:17-35` and `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:52-62`. |
| ALREADY RECORDED | Stage 4's matched `[0, 16]` pin, the per-phase volume gate and the CPU scoring/skip contract. `docs/drafts/pwc_backup_stage3_v1/PLAN.md:299-310`, `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:5-23`, `docs/CHAPTER_4_AUDIT.md:134-140`, `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:46-63`. |
| NEW | Java LCG source-loop work count for the two ranges and a local Optuna check of a warm-start pin above the device output limit. |
| CHANGES A RULING | None. |

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — Design reading | NO OBJECTION | — |
| Q2 — Direct fix or flag | SUGGESTION | — |
| Q3 — Before or after Run 4 | SUGGESTION | — |
| Q4 — Operational effects | SUGGESTION | — |

### Q1 — Design reading

The sources that specify constant-skip execution support delivering the trial pair. The operator docs identify `[0, 16]` as a default, and the RANGE-MINER proposal lists the skip fields for publication. The statements claiming live delivery overstate what the production code does. Legacy and PWC both send configured bounds. RANGE-MINER's dispatch and builder omit them, so the worker uses `[0, 16]`. I found no coordinator history introducing the key in this clone. `docs/instructions.txt:1182-1183`, `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:204-206`, `docs/HYBRID_SKIP_BOUND_AUDIT.md:108-117`, `coordinator.py:1477-1481`, `persistent_worker_coordinator.py:2632-2639,2718,2745`, `miner/range_miner_coordinator.py:10498-10504,12018-12023,9331-9358`, `miner/range_miner_worker.py:1293,1385-1393`.

I found no standing design decision to force `[0, 16]` for all constant trials. The known-answer and Wall A/B harnesses describe the fallback they exercise. The Phase-6 sampler harness explicitly inserts its own range, which accounts for the submission's claim. The owner briefly directed PWC to match the fallback for parity and superseded that direction for S3-13 the same evening. `tests/phase6/known_answer_gate.py:279-284`, `tests/phase6/wall_ab_gate.py:767-773`, `tests/phase6/sampler_control_arm.py:159-168`, `docs/TEAM_ALPHA_BOUNDED_PHASE_6_SUBMISSION.md:378-385`, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47-51`.

### Q2 — Use a selectable configuration for the comparison

I favor a run-selectable delivered-range option with `[0, 16]` as the baseline through the comparison. A direct fix is less code and immediately delivers the requested bounds for every trial. Its cost is exposing all sampled trials to new compute and load before those costs and pool performance are measured. A flag costs launch, trial and estimator plumbing. It allows a matched control and an immediate fallback. Record the requested and effective pairs and the selected behavior with each run so an old study or a volume check cannot be mistaken for delivered-range evidence. `CLAUDE.md:30-35`, `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md:107-116`, `utils/survivor_volume_estimator.py:510-516,654-665`.

A direct fix can also run a control pinned to `[0, 16]`, but changing the requested pin also changes the NPZ features and Step 3's `skip_min` fallback for holdout features. To isolate the sieve effect, keep the same requested trial and downstream settings in both arms and change only the effective pair on phases 1–2. Use the same data cutoff, seed interval, thresholds, hybrid settings, scoring procedure and genuinely unseen target draws. Keep each arm's survivors and pools independent. Report the paired hit-rate difference, each arm's hit rate against `k/1000`, and how often each arm produces a pool. There is no evidence for the direction of the effect. The recorded Run 4 pins produced empty bidirectional output in both later half runs, so record a no-pool outcome explicitly. `utils/canonical_records.py:215-232`, `full_scoring_worker.py:408-423`, `holdout_quality.py:44-58`, `docs/TFM_SESSION_HANDOFF_20261119.md:19-26`.

### Q3 — Run 4 already earned credit

Keep Run 4 as the historical operational baseline and take up F-1 as a separate reviewed change after PWC Stage 4. Stage 4 can compare both engines with the trial pinned to `[0, 16]`. That comparison answers parity at the matched geometry. A separate paired RANGE-MINER evaluation answers the skip-range hit-rate question. The 2026-11-14 handoff's timing advice still makes sense, and the later credit record settles the immediate scheduling question. Run 4's upper-half interval and Attempt 9's lower-half interval also limit what their comparison can establish. See Finding 1. `docs/TFM_SESSION_HANDOFF_20261114.md:76-80`, `docs/drafts/pwc_backup_stage3_v1/PLAN.md:342-354`.

### Q4 — Operational effects

- **Compute and lease.** `[6, 99]` tries 94 hypotheses per seed versus 17, or 5.53×. At Run 4's `k=12` and `generator_phase=0`, both Java LCG constant kernel sources execute `12 + 13 × skip` state advances per hypothesis. That gives 65,283 versus 1,972 source-level advances per seed (33.1×). GPU elapsed time and the effect of compilation are **UNAVAILABLE**. The 300 s lease measures time since claim or the last accepted renewal. The worker has a separate heartbeat thread with a 30 s default, and a newly accepted sub-stripe also renews the lease. A longer stripe can complete while renewing. A 300 s lapse in renewals outside the pause/grace exemptions can abort the trial immediately in phases 1–2. Measure the slowest rig's cold and warm sub-stripe times, accepted renewal gaps and run duration before changing the lease. Run N documented a phase-1 expiry after a card dropped off the bus, which supplies an example of the failure path at today's settings. `prng_registry.py:972-995,3131-3160`, `miner/range_miner_worker.py:2338-2352,2705-2715,2941`, `miner/range_miner_coordinator.py:626,8475-8488,8615-8640,11757-11774,11821-11844,11967-11980`, `docs/TFM_SESSION_HANDOFF_20261119.md:14-20`.
- **Stripe sizing and VRAM.** The constant path allocates one `uint8` best-skip slot per candidate. There is no skip-width dimension in its buffers or cap selector. At the same seed cap, expect more compute and potentially more output, with the same fixed device-buffer sizes. An unrestricted rollout also needs timing at the largest normal span `[0, 250]`, which tries 251 hypotheses. If the measured kernel time calls for smaller sub-stripes, update worker caps and the coordinator's advertised-cap accounting together. Smaller sub-stripes increase the number of staging files at fixed macro-stripe size. `distributed_config.json:84-91`, `miner/range_miner_worker.py:647-666,1338-1388,1399-1422,2369-2372`, `miner/range_miner_coordinator.py:740-790,4799-4805`.
- **Volume gate.** The report's exact upper-half counts were `0 / 0 / 60 / 23,568` under the existing path and the owner-set cap is 25,000 per phase. The new range drops 0–5 and adds 17–99. At that measured geometry, constant-phase counts may stay at zero or rise. Their size, their intersection and the hit-rate consequence are **UNAVAILABLE**. The hybrid ABI has no bound arguments. Feed the estimator the same effective choice as dispatch and measure each intended geometry and interval before relying on the cap decision. `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:5-23`, `miner/range_miner_worker.py:259-292`, `utils/survivor_volume_estimator.py:500-516,654-665`, `scripts/survivor_volume_gate.py:199-211`.
- **`best_skips`.** The normal Optuna search tops out at 250, which fits `uint8`. In a local in-memory check, Optuna 4.4.0 returned a fixed enqueued `skip_max=256` from `suggest_int(..., 10, 250)` with an out-of-range warning. The production objective passes the suggested pair into `WindowConfig` and the warm-start enqueue uses the supplied integers. Validate `0 ≤ skip_min ≤ skip_max ≤ 255` at the common assignment boundary before the delivered pair can reach the kernel. A winning skip of 256 would serialize as 0 in the one-byte output. I did not execute the launcher or the kernel with that pin. `distributed_config.json:84-91`, `window_optimizer_bayesian.py:549-576,779-797`, `miner/range_miner_worker.py:1381-1388`, `prng_registry.py:997-1001,3162-3166`.
- **Historical studies.** Keep completed trials and their recorded pairs as provenance for the behavior they evaluated. Their objective values provide no evidence for the recorded pairs as constant-kernel inputs. Start the delivered-range evaluation under a fresh study identity. The resumer can load completed trials into the study, which would otherwise mix objective values from two execution behaviors. The effect on historical trial rankings is **UNAVAILABLE**. `miner/range_miner_worker.py:1293`, `window_optimizer_bayesian.py:549-592,676-708,746-761`.
- **Prediction and deployment.** The worker collects each constant candidate's best skip. The frozen NPZ exports the trial stamps, the scorer copies them and Step 6 looks for `skip`, `metadata.skip` or `features.skip_hypothesis`. The range-only fixture used the fallback as expected. A delivered range may change which candidates vote even when those votes still use the fallback. The per-seed handoff on a real miner-to-pool run is **UNAVAILABLE**. The CPU stepping contract has its own recorded work. The coordinator is also in Gate 12's rig parity set, so even a coordinator-only fix changes the bytes the gate expects on the fleet. `miner/range_miner_worker.py:1414-1422`, `convert_survivors_to_binary.py:51-73`, `survivor_scorer.py:771-781`, `full_scoring_worker.py:388-402,451-466`, `prediction_generator.py:530-553,781-795,841-849`, `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:46-63`, `scripts/gate12_parity_gate.py:174-185`, `gate12_launch.sh:191-198`.

## Findings

1. **ALREADY RECORDED — the request missed the governing Run 4 outcome and interval decision.** The upper-half Run 4 attempt 3 earned credit on 2026-11-15. Attempt 9 covered the lower half. Update the design record's timing question and describe the comparison as operational, with both intervals stated. The later v7 handoff records clean runs under the Run 4 pins on both halves. `CLAUDE.md:223-225`, `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md:150-155`, `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md:47-55`, `docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md:5-22`.
2. **NEW — the draft understates the increase in source-loop work.** Use 5.53× for the number of skip hypotheses and 33.1× for the number of Java LCG advances at the stated geometry. Fleet timing is **UNAVAILABLE**. See Q4.

## Checks run

After activating `~/venvs/torch`, I ran CPU-only checks of the production payload builder for all four phases, the constant and hybrid argument lists, the Step 6 skip extractor, the in-memory Optuna warm-start case, and the Java LCG loop arithmetic. The first check script selected the wrong argument indices. I corrected the assertion and reran all checks successfully. I read the dispatch, lease, volume-gate, scoring and deployment paths. GPU counts for delivered `[6, 99]`, fleet timing, a miner-to-pool run and paired unseen-draw hit rates are **UNAVAILABLE**.
