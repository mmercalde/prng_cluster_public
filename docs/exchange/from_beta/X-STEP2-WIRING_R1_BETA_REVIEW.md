# X-STEP2-WIRING · R1 · Beta opinion

2026-11-22 · Base read `ab2d330bb5bd74deed716c823f2cfb9efd7fa60c` · One round, non-blocking

`git status --porcelain` at review time

```text
 M agent_manifests/window_optimizer.json
 M docs/LEADS.md
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
?? docs/RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md
?? docs/RESULT_SKIP_SWITCH_RIG_RUNS_20261122_ADDENDUM.md
?? docs/drafts/DRAFT_STEP2_DIAGNOSTIC_HARNESS_20261122.md
?? docs/drafts/NOTE_LEAD_BIDIRECTIONAL_EXCESS_20261122.md
?? docs/drafts/NOTE_STEP2_INPUT_AND_ZERO_BIDIRECTIONAL_20261122.md
?? docs/drafts/NOTE_STEP2_READINESS_ON_RUNH_20261122.md
?? docs/drafts/PLAN_STEP2_REPAIR_20261122.md
?? docs/drafts/SKIP_DESIGN_RECORD_20261121.md
?? docs/drafts/miner_skip_switch_v1/
?? docs/drafts/pwc_backup_stage4_v1/PARKED_MINER_PHASE_CAPTURE.md
?? docs/drafts/pwc_backup_stage4_v1/SCOPE_MINER_PHASE_SETS.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH-2_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH-2_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH-2_R3_BETA_REVIEW.md
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
?? docs/exchange/to_beta/X-STEP2-WIRING_R1_ALPHA_OPINION_REQUEST_step2_settings_to_steps_3_and_5.md
?? scripts/diagnostics/
?? tests/test_miner_skip_switch.py
```

## Prior record

- Searched `docs/` with `rg` for the config, wiring, residue names, schema validation, offset use, NN refit, batch modes and relearn. Cross-checked the chapter audits, corrections report, system map, backlogs, proposals, ruling files and exchanges. Traced the worktree's WATCHER, job generators, worker, optimizer, prediction and Chapter 13 call sites.
- **ALREADY RECORDED** — Alpha's R1–R7 and the quoted owner rulings establish the scope. Finding 1 identifies the missing governing record. Chapter 4 B2/B5/B6, Chapter 6 B2/B3/B5 and Chapter 13 B2/B5/B6/B19 record the relevant existing behavior. See `docs/CHAPTER_4_FULL_SCORING.md:1454-1459`, `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md:1254-1261`, `docs/CHAPTER_6_AUDIT.md:219-230` and `docs/CHAPTER_13_AUDIT.md:80-105,553-569`.
- **NEW** — The proposed config handoff, training policy and test sequence below. **CHANGES A RULING** — None. O1 and O2 apply.
- Review scope was this worktree. Independent verification of the owner-log hashes, deployment, private repository and archived runs was **UNAVAILABLE** for this review.

## Positions

| Question | Position | Severity |
|---|---|---|
| 1. Safest wiring | SUGGESTION | — |
| 2. Downstream effects | SUGGESTION | — |
| 3. Step 5 policy | SUGGESTION | — |
| 4. Landing order and tests | SUGGESTION | — |

## 1. Safest wiring

**LIVE — recorded.** WATCHER invokes the Step 2 shell entry point, which publishes a flat `best.params` JSON without input identity. Step 3 passes its config path only if the file exists, and the job generator reads only the PRNG fields before emitting worker arguments. The worker constructs its scorer with those two fields. `generate_full_scoring_jobs.py` belongs to the older shell path. It passes JSON to `scorer_trial_worker.py` with seed-only chunks, while WATCHER uses `generate_step3_scoring_jobs.py` and `full_scoring_worker.py`. Wire the WATCHER path first. (`agents/watcher_agent.py:405-412`, `run_scorer_meta_optimizer.sh:299-309`, `run_step3_full_scoring.sh:159-178`, `generate_step3_scoring_jobs.py:201-209,241-265,352-354`, `full_scoring_worker.py:340-345,522-550`, `run_full_scoring.sh:31`, `generate_full_scoring_jobs.py:38-52,67-92`.)

Resolve and validate the Step 2 artifact once on the coordinator. Pin a run-scoped copy with its schema version, producer run and trial, config digest and source input identities. Give each Step 3 job one validated JSON argument containing the seven scorer settings and the pinned identity. Pass the settings through the worker to `SurvivorScorer(config_dict=...)`. Pass the same pinned artifact to Step 5 on the host and through the compare-models child invocation. Per-field CLI flags multiply the handoffs. A worker-side path would require staging and verifying the exact file on every host. The present ramdisk preload runs before option parsing and stages the default filename. Stamp the config identity on each chunk and check it before aggregation. Validate all eleven keys and their supported types and ranges, including the three distinct moduli and the NN encoding. Leave PRNG family and output modulus under the run's existing input authority. (`generate_scorer_jobs.py:52-71`, `survivor_scorer.py:90-111`, `run_step3_full_scoring.sh:30-40,69-100,369-408`, `meta_prediction_optimizer_anti_overfit.py:214-246,3110-3130`.)

Fail closed before dispatch or fitting when the requested artifact is missing, malformed, or mismatched to the intended producer or scored data. Determine staleness from IDs and digests. A deliberately pinned older Step 2 config can be reused for a compatible Chapter 13 refresh, with fresh scoring, training and prediction artifacts for that refresh. Keep historical artifacts readable in an explicit legacy mode using their matching feature and model contracts. Test that a fresh-looking file with an unrelated digest fails.

## 2. Downstream effects

- **Step 3 feature contract.** **THEORETICAL on the current default path.** Passing nondefault moduli would rename nine columns while retaining three columns per modulus and the 91 extracted / 89 trained widths. The scalar extractor also uses value-specific names. Keep those truthful names and version the ordered feature contract, including the effective settings and their meaning. Validate the exact keys and finite values across rows and chunks. Count alone misses renamed features, and a names-only hash misses changes in temporal settings. Preserve the legacy contract for historical artifacts. (`survivor_scorer.py:376-390,624-667,670-707`, `meta_prediction_optimizer_anti_overfit.py:546-596,1479-1483`, `docs/CHAPTER_4_FULL_SCORING.md:1050-1089`.)
- **Holdout target.** **THEORETICAL under the proposed wiring.** The current formula would silently lose the exact-match and coherence terms keyed to 1000, 8 and 125 when those names disappear. The separate lane terms still measure 8 and 125. In an isolated check, changing only the three residue prefixes took the same test vector from `0.3447` to `0.0850`. My first comparison would preserve the whole existing label recipe, including temporal defaults, by computing its inputs separately on the holdout and keeping the configured scorer features in the model input. If the intent is to use configured residues and windows in the target too, pass an explicit mapping for the three slots, require those keys and version the target. Treat a missing label input as a failed chunk in wired mode. The current worker catches label errors and writes zero. That changes the target, including the meaning of the present 1000 match term. Test it as a separate comparison. (`holdout_quality.py:91-125`, `survivor_scorer.py:611-614,670-678`, `full_scoring_worker.py:408-430,473-487`.)
- **Effective Step 3 settings.** **LIVE — recorded.** The constructor stores `max_offset`, but both extractors emit a constant zero for `best_offset`. Threading the value proves delivery only. Keep it visible and record the Step 3 behavior as inactive until a Step 3 alignment rule is specified and tested. Step 2 uses the value as a bound on seed residues. A Step 3 alignment feature needs its own defined semantics. O2 leaves the Step 2 objective in force and the S107 issue parked. The temporal settings affect extraction and the confidence threshold floors `confidence`. Step 5 excludes `confidence` from training and the current Step 3 aggregate sorts by `score`. A constructor assertion or altered confidence alone establishes no ranking effect. (`survivor_scorer.py:107-111,368-373,396-413,670-707`, `scorer_trial_worker.py:380-389`, `meta_prediction_optimizer_anti_overfit.py:1479-1483`, `run_step3_full_scoring.sh:403-408`, `docs/CHAPTER_4_FULL_SCORING.md:1455`.)
- **Step 5 inputs and metadata.** **LIVE — recorded behavior.** The optimizer infers sorted names from the first record with features and zero-fills absent names in later rows. It fits `holdout_quality` from the scored artifact. Version and validate both input contracts before fitting, and key study reuse by feature semantics, target version, data identity and training policy. Write the ordered features, effective settings, scored-artifact digest and actual fitted parameters to the sidecar. Publish the sidecar with the matching model checkpoint and NN scaler. A prior checkpoint with the same width is still a different contract when its column semantics differ. (`meta_prediction_optimizer_anti_overfit.py:546-596,645-684,1479-1483,1737-1742,1774-1780,2898-2927`, `train_single_trial.py:706-724`.)
- **Step 6.** **LIVE — recorded mismatch mechanism.** The CLI opens the supplied forward/reverse survivor files and the model sidecar. On the intersection branch, `compute_dual_sieve_intersection` returns integer seeds, and the scorer is constructed with default feature settings. For a dictionary input it uses the supplied features. In either case, missing model columns silently become zero. For the wired path I would pass the matching Step 3 feature rows and retain their per-survivor skip metadata through the input handoff. If inference regenerates features, use the same settings, extraction path, metadata and global inputs. Validate the feature identities and order before prediction and check NN scaler alignment. Keep both constant- and variable-skip bidirectional records in the parity fixture. (`agent_manifests/prediction.json:176-183`, `prediction_generator.py:258-266,345-370,602-627,777-823,1042-1068`, `survivor_scorer.py:149-173`, `docs/CHAPTER_6_AUDIT.md:219-230`.)
- **Chapter 13.** **LIVE — recorded routing and input gaps.** The explicit WATCHER dispatcher has a 3→5→6 list and calls `run_step`. The trigger wrapper can also use a contiguous range, and the standalone executor supplies bare commands even though Step 5 requires two input arguments. Use the explicit list with one resolved run context carried through the same validated entry points. A relearn needs the intended draw version and its corresponding histories, as well as the pinned Step 2 config and the refreshed scored data and model. That requires the existing dataset-authority gap to be addressed before claiming new-draw relearn. Chapter 13 diagnostics must read the actual sidecar's nested feature schema. Update §18 to show the seven Step 3 settings, the four NN candidate settings with Step 2's actual encodings and choices, Step 5's separate search and effective batch behavior, and the reuse rule. `sample_size` is a generator input and is absent from the persisted `best.params`. (`agents/watcher_dispatch.py:297-344`, `agents/watcher_agent.py:2011-2034,3095-3115`, `chapter_13_triggers.py:610-616,644-679`, `meta_prediction_optimizer_anti_overfit.py:3037-3044`, `generate_scorer_jobs.py:44-71`, `run_scorer_meta_optimizer.sh:302-307`, `docs/CHAPTER_13_AUDIT.md:80-105,553-569`, `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP_v1_1.md:874-889`.)

These failures predict loss or distortion of ranking inputs. The direction and size of any hit-rate change are **UNAVAILABLE** from code inspection and these fixtures. Compare locked variants on the same unseen draws at the same `k`, against the current pool and `k/1000`.

## 3. How Step 5 should use the values

**LIVE — recorded.** Step 2 samples all eleven settings. Its v4.3 mask uses the three residue moduli and `max_offset`. The objective gives the four NN settings no training signal, so their values carry the provenance of the winning trial. The WATCHER manifest uses compare-models with 20 trials, and the compare wrapper invokes per-model children. The separate distributed launcher calls the legacy generator and worker, which decodes JSON-list strings and prepares a 46-input ReinforcementEngine config. Carry the candidate through the current NN child first. If the legacy launcher remains in scope, translate through the same validator at its boundary and test that path separately. (`generate_scorer_jobs.py:54-66`, `scorer_trial_worker.py:343-349,380-400,421-453`, `agent_manifests/reinforcement.json:173-183`, `meta_prediction_optimizer_anti_overfit.py:214-246,3110-3130`, `run_anti_overfit_optimizer.sh:54-62`, `generate_anti_overfit_jobs.py:29-39,102-119`, `anti_overfit_trial_worker.py:144-170`.)

| Choice | Cost |
|---|---|
| Fixed inputs | Simple to replay and inspect. Fixes all four NN settings to a candidate with no Step 2 training signal and forgoes the existing Step 5 search on those axes. |
| One starting candidate **(preferred)** | Uses one declared slot of the NN trial budget. Decode `512_256_128` to `[512, 256, 128]`, score the complete tuple on the same Step 5 folds and let it compete with the native trials. |
| Search bounds around the values | Changes the search policy and requires a rule for translating categorical architectures and batch sizes into bounds. The Step 2 objective supplies no basis for narrowing those bounds. |

The starting candidate needs an explicit evaluation path or a compatible, versioned study space. The native NN sampler caps the first width at 256, starts dropout at 0.2, and its batch categories have zero overlap with Step 2's. The legacy sampler also lacks `[512, 256, 128]`. A bare enqueue in either existing space is insufficient. For the current path, reconstruct the complete trainer config for the winning trial before final fit. **LIVE — recorded in Chapter 6 B3.** Fold evaluation constructs `hidden_layers` and `early_stopping_patience`, while refit passes raw `n_layers`, `layer_i` and `patience`, so the trainer defaults both inputs. Also record any CLI dropout override and the actual batch mode. The current NN command requests `auto`. Below 200 MiB, the trainer uses full-batch and the requested `batch_size` is used only on the mini-batch path to size optimizer batches. Use mini-batch for the candidate if its batch size must take effect, and disclose that policy and its training cost in the comparison. Keep NN settings out of tree-model training. (`generate_scorer_jobs.py:63-66`, `generate_anti_overfit_jobs.py:29-39`, `meta_prediction_optimizer_anti_overfit.py:1812-1829,2015,2460-2476,2563-2582`, `train_single_trial.py:522-558,569-601,625-644`, `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md:792-802`.)

## 4. Landing order and tests

1. **Pin and validate producer output.** Use a test fixture shaped like the shell's flat `best.params` artifact. Round-trip distinctive nondefaults for all 11 settings through the normal job generation and compare-child argv. Test missing keys, stale input IDs, wrong digest and an explicit historical reuse case. Keep the existing objective under its current tests. (`run_scorer_meta_optimizer.sh:299-309`, `generate_scorer_jobs.py:52-71`.)
2. **Make the Step 3→5→6 contract safe before activating the new feature settings.** Exercise batch and forced sequential fallback. Check exact feature keys, expected numerical residue and temporal outputs, confidence floor, target components and metadata parity. Change one modulus, temporal window size or count, or threshold at a time on a fixture that distinguishes it. For `max_offset`, separate delivery from a future behavioral check that changes the allowed alignment range. At aggregation and training, reject mixed chunks and same-width renamed or semantically mismatched rows. At Step 6, check the exact inference vector and a ranking-sensitive toy model, including an old sidecar paired with new scored rows and vice versa. Verify scaler order and values for NN. (`full_scoring_worker.py:390-489`, `survivor_scorer.py:624-707`, `meta_prediction_optimizer_anti_overfit.py:546-596,645-684`, `prediction_generator.py:777-836`.)
3. **Land the Step 5 refit fix and then the NN candidate.** Use the normal compare wrapper, inspect effective architecture and dropout in the trainer, learning rate in the optimizer and batch size in the mini-batch loader. Have the candidate win once and a native trial win once. Reload the actual final checkpoint and compare its fitted settings and scaler to the sidecar. Check explicit dropout precedence, full-batch reporting and no duplicate candidate on resume. For both wiring stages, make a mutant drop one key at the boundary under test and require the relevant downstream assertion to fail. (`meta_prediction_optimizer_anti_overfit.py:214-252,1774-1798,1812-1829`, `train_single_trial.py:522-594,706-724`.)
4. **Test Chapter 13 with stubbed subprocesses, then evaluate.** Assert the 3→5→6 sequence and that one approved data/config context reaches scoring, fitting and prediction. Move the default config pointer between steps and verify the pinned context is used. Run the existing and wired variants against identical unseen draws, keeping the bidirectional inputs and pool size comparable, and report hit counts, paired hit-rate difference and uncertainty against the current baseline and `k/1000`. The diagnostic slice does not settle that comparison. (`agents/watcher_dispatch.py:297-344`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`.)

## Finding

1. **Prior record omission.** The request missed the feature contract and parity positions in the system map, the dataset authority and score-refresh rule, and their application to this flow in the Chapter 6 and 13 audits. The system map specifies 91/89, exact name/count/order/hash validation, per-artifact schema version and Step 3 parity. The dataset record pins an immutable source and requires rescoring when its version advances. These are **ALREADY RECORDED** constraints, with the former Beta positions advisory under the current owner rule. They affect the choice of config binding, mixed-artifact checks and what counts as a successful relearn. (`docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:247-253`, `docs/TFM_PROJECT_FACTS_SKILL.md:673-680,713-719`, `docs/CHAPTER_6_AUDIT.md:219-230`, `docs/CHAPTER_13_AUDIT.md:80-85,553-569`.)

## Verification

Read the worktree entry points and prior record. With the torch environment active, ran a local Python AST harness using the two checked-in functions and hand-built inputs. It reproduced the holdout key substitution and confirmed the sampler-to-refit shape difference. End-to-end execution, fleet behavior, checkpoint reload and unseen-draw hit rates are **UNAVAILABLE** in this opinion review. No pipeline, worker or test suite was launched.
