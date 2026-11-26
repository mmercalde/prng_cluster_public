## Prior record

**X-LL-SWITCHES · R1 · Team Beta · 2026-11-25 PST**  
Before-commit courtesy review. Base read `b2b76fbf4d78044b117d2399aaa683953309d2f4`. Michael decides the landing.

`git status --porcelain` at the initial read, 14:43 PST:

```text
?? docs/exchange/to_beta/X-LL-SWITCHES_R1_ALPHA_REVIEW_REQUEST_ll_switch_landing.md
```

**Finding 1 — ALREADY RECORDED governing material omitted from the request.** The system map §7.3 records Option A for adding the two match-rate inputs under a new schema and sets the broader F-PARITY criterion. The consolidated TODO and Chapter 4 audit carry P0-B. The Chapter 7 audit records the Step 6 hash gap. The original LL implementation note links those decisions to these switches and records the limits of its parity test and Step 6 output. The owner’s autonomy instruction and the lead’s switch-register addendum govern where registry and manifest work goes. Alpha cites the latter pair in the READY note, §(b), but leaves them out of the request’s prior record.  
`docs/TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:159-169`, `docs/TODO_CONSOLIDATED.md:62,159-161`, `docs/CHAPTER_4_AUDIT.md:119-124,142-146`, `docs/CHAPTER_7_AUDIT.md:282`, `ch14-learn:docs/drafts/loop_v1/READY_LOOP_LEARN_W1A_20261123.md:44-57,274-284`, `logs/e2e_session/OWNER_DIRECTION_autonomy_switches_20261123.txt:2`, `logs/e2e_session/TASK_SWITCH_REGISTER_ADDENDUM_1.txt:1-4`.

- **Searches.** I used `git grep -ilE` at `b2b76fb` over `docs/` (catalog, rulings, proposals, chapter audits, TODO, backlog, leads, drafts and exchange) and `rg -n -i` for LL-1/2/3, per-seed sieve evidence, feature manifest, schema checks, group exclusions, fallback parity, passthrough and prior review terms. I checked `logs/e2e_session/*.txt` with `rg --no-ignore` for the switch work and read the cited owner and lead directions, the named worktree material and the landed closing note.
- **ALREADY RECORDED.** The next comparison is 01 against 00. Registration and WATCHER routing are separate work. Strict LL-1 stops at A0. The post-landing proof used a local scratch commit, which the owner has since explicitly prohibited. The promotion gate is walk-forward on historical draws reserved from selfplay. `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:22-25`, `logs/e2e_session/TASK_LL_SWITCHES_BETA_20261125.txt:1-2,9`, `e2e-map:docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md:131-136,399-410`, `docs/CONTRACT_SELFPLAY_CHAPTER13_AUTHORITY_v1_0.md:10-26`, `CLAUDE.md:43-67`.
- **NEW.** LL-3 accepts invalid selections and can overstate which families applied a selection. The Run H fixture’s window-config check is missing from the optional live cross-check. **CHANGES A RULING** none.
- **Not searched.** Ser8 archives, private remotes, rigs, live ledgers or host files beyond the named evidence, worktrees and local test inputs.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — Default unchanged | NO OBJECTION | — |
| Q2 — Switch on | OBJECTION | LIVE — A blank LL-3 group request can produce default columns under an applied label, and a misspelled exclusion can put two normally excluded columns into training. |
| Q3 — Tests after landing | SUGGESTION | — |
| Q4 — Landed fixes | NO OBJECTION | — |

### Q1 — Default unchanged

On CPU fixtures, Step 3's default batch and forced-fallback outputs matched their respective pinned baselines, and Step 5 produced the same matrix and 89-column schema hash `7733d30a913545ca`. The Step 6 default CLI dry-run output matched byte for byte. The Step 5 model sidecar carries the selected feature schema, and at default the loader adds no selection block. The cited Step 6 line serializes configuration for the dry run. Step 6 reads the model sidecar and writes prediction JSON without the switch field at default. The current WATCHER job arguments and manifest settings preserve the defaults by inspection. A WATCHER/GPU runtime comparison was **UNAVAILABLE** in this review.  
`full_scoring_worker.py:345-355,387-396,453-468,514-521,569-575`, `meta_prediction_optimizer_anti_overfit.py:1486-1499,2852-2863`, `prediction_generator.py:134,197,228,300-301,783-794,713-740,948-969,1055-1061`, `agent_manifests/full_scoring.json:37-45`, `generate_step3_scoring_jobs.py:249-267`, `agent_manifests/reinforcement.json:173-184`, `agent_manifests/prediction.json:23-36,176-184`.

Alpha’s retained comparison shows equal outcomes and 0 of 11 fingerprint changes at default, and 5 changes with sieve evidence on. The raw per-draw hashes differ. The READY note attributes that to `config_id` across 72 rows and reports the same difference in the baseline repeat. I read those results. Independent row-by-row comparison was **UNAVAILABLE** from the retained summary. The walk-forward record also exposes capability presence and code identity, so the output comparison concerns the pipeline results on the measured fixture.  
`e2e-map:data/landing_llsw_20261125/compare_c_ll_vs_main.json:2-12`, `e2e-map:docs/drafts/loop_v1/READY_LAND_LL_SWITCHES_20261125.md:90-94`, `scripts/diagnostics/e2e_pipeline_caps.py:211-224`, `scripts/diagnostics/e2e_walk_forward.py:754-770`.

### Q2 — Switch on

**LL-3 accepts bad requests.** I exercised Step 5's real CSV conversion and loader on 40 synthetic survivor rows with Run H's 91 feature names. With `--include-feature-groups ''` or `--exclude-feature-groups ','`, the loader used the default 89 columns and omitted selection metadata from the schema. The driver’s capability function reported the blank include request as applied. With `--exclude-features confidnce`, the loader accepted 91 columns, including `score` and `confidence`, and recorded the misspelled exclusion. Explicitly listing all 91 names for exclusion produced a 40 × 0 matrix. The named-group path already refuses an unknown group or a selection that leaves no columns. Move those checks to cover explicit names and distinguish omitted group flags from supplied empty values. Exercise these cases through the CLI and assert that invalid selections are refused before fitting.  
`meta_prediction_optimizer_anti_overfit.py:546-568,629-652,1486-1503,3054-3058,3128-3136,3212-3214`, `models/feature_manifest.py:195-224`, `scripts/diagnostics/e2e_pipeline_caps.py:125-148,211-224`.

**The LL-3 applied record loses family scope.** I passed `--exclude-feature-groups dead` as general Step 5 args and `--exclude-feature-groups global` as NN args to the driver’s capability function. It reported only `global` as applied. The driver appends NN args only for `neural_net`. With those settings, LightGBM receives `dead` and the NN receives `global`. I verified both resolver selections and read the per-family argv construction. A family-specific record checked against the trained sidecars would settle which columns each arm used. The direction of any hit-rate change is unknown. Measure it on matched walk-forward blocks with the effective feature lists verified.  
`scripts/diagnostics/e2e_walk_forward.py:577-588,728-730,754,976-986`, `scripts/diagnostics/e2e_pipeline_caps.py:146-148,211-224`, `models/feature_manifest.py:210-233`, `meta_prediction_optimizer_anti_overfit.py:1498-1499,2852-2856`.

**Verified cases and recorded limits.** The LL suite showed the two exact per-seed values added across 40 rows and refusal when a required value was missing. LL-1 `fail` refused missing or renamed row features and an ordinary bad hash in fixtures. `warn` reported issues and continued. Alpha’s A0 failure is the recorded sidecar-free case. The earlier F-PARITY scope stays open. In my CPU check `shared` plus `on` matched all 20 metadata-backed fields, while 18 computed values differed by more than `1e-6`, up to `0.000944`. The test reports those differences without enforcing a tolerance, as the original LL note records. These checks measure data and configuration behavior. The hit-rate effect of LL-2 is the sealed 01 versus 00 comparison against the current baseline and `k/1000`.  
`full_scoring_worker.py:345-355,389-396,466-468,514-521`, `survivor_scorer.py:86-134,861-867`, `models/feature_manifest.py:240-296`, `tests/test_ll_feature_settings.py:139-189,250-324`, `scripts/diagnostics/e2e_vote_harness.py:730-745`, `ch14-learn:docs/drafts/loop_v1/READY_LOOP_LEARN_W1A_20261123.md:139-142,274-280`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`.

### Q3 — Tests after landing

The default pin resolves to the full pre-LL revision. In the scratch checkout the capability checks refused all three switches on the pinned bytes and accepted them on the submitted tree, which the selftest selected as its with side. G19 ran strict mode with the submitted tree and killed M25 on the pinned refusal case. The LL suite loaded separate baseline files and checked their preconditions. Alpha’s retained W10/W11 log confirms the post-landing side selection and explains the one fewer gate. The direct Step 3 refusal on the tree under test applies only while that tree lacks LL-2.  
`scripts/diagnostics/e2e_walk_forward_selftest.py:262-270,288-337,412-444`, `tests/test_e2e_vote_harness.py:968-1034,1317-1327`, `tests/test_ll_feature_settings.py:45,477-494`, `e2e-map:data/landing_llsw_20261125/C_e2e_walk_forward_selftest.log:104-126,150`.

The three fixtures contain only feature names, the sidecar’s schema and the leak check’s settings, length and digests. They contain no draw data. The dataset is read by path via `--runh-dataset` and checked by SHA-256. The optional live gate verifies the Step 3 names, sidecar and train prefix. **Suggestion:** also compare the window config’s hash and projected settings against `copy/configs/optimal_window_config.json` in that gate. The fixture supplies those settings to the leakage check and the live gate currently never reads that config. I independently compared both the named source and Run H’s copy with the fixture. Their digests and projected settings matched.  
`tests/fixtures/runH_step3_feature_names.txt:1-3`, `tests/fixtures/runH_winner_feature_schema.json.txt:2-7,98-105`, `tests/fixtures/runH_leak_inputs.json.txt:2-16`, `tests/test_ll_feature_settings.py:335-385,463-468`.

### Q4 — Landed fixes

The four wrappers run the driver’s `--check-only` in the foreground and propagate a refusal before opening the run log or invoking the timer and background command. Alpha’s W11 log shows the missing-LL-2 wrapper returning 2 with all five write/launch checks false and the LL-2 side reaching the stubbed launch. I read the wrapper evidence and syntax-checked all four. I independently called the landed passthrough check and saw it refuse `--per-seed-sieve-f`, `--survivors=PATH`, both fold-count forms and `--trials=3` through NN args. The shim independently refused the joined survivors form before creating its capture directory. W12 records the derived target value 2 and three folds reaching the real optimizer parser, and recovers the original value 0.045 under the mutant. Direct optimizer invocation still has argparse’s default abbreviation behavior, as the closing note records. The driver preflight covers that parser on the walk-forward route.  
`docs/drafts/loop_v1/runH_commands/00_base.sh:45-55`, `docs/drafts/loop_v1/runH_commands/01_sieve_on.sh:45-56`, `docs/drafts/loop_v1/runH_commands/02_target_hits.sh:45-56`, `docs/drafts/loop_v1/runH_commands/03_rankers.sh:46-57`, `scripts/diagnostics/e2e_walk_forward.py:93-95,550-562,606-607,1462-1489`, `scripts/diagnostics/e2e_pipeline_caps.py:180-208`, `scripts/diagnostics/e2e_step5_shim.py:116-141`, `meta_prediction_optimizer_anti_overfit.py:3061-3064`, `docs/exchange/to_beta/X-E2E-WF-VOTE_R2_ALPHA_CLOSING_NOTE.md:70-76`, `e2e-map:data/landing_llsw_20261125/C_e2e_walk_forward_selftest.log:117-146`.

## Verification

I verified all 13 submitted SHA-256 digests at the sources and in the overlay, and verified the READY note and base-check digests. The only committed change from `50cd683` to `b2b76fb` is the closing note. I ran the LL suite with `--runh-root` (37/37 gates, 10/10 mutants), the vote suite (15/15 gates, 17/17 mutants), and the focused checks above in a temporary detached checkout under `/tmp`, with the submitted files overlaid and CUDA hidden in `~/venvs/torch`. The checkout's HEAD remained at `b2b76fb`. Independent full walk-forward and real Run H wrapper execution were **UNAVAILABLE**: the driver forbids `/tmp` output and the review has no pipeline launch authority. I read Alpha’s retained preflight, W10/W11/W12 and fingerprint evidence for those paths. `scripts/diagnostics/e2e_walk_forward.py:105-106,1463-1467`, `AGENTS.md:100-111`.
