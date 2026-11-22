# X-MINER-SKIP-SWITCH-2 · R1 · BETA REVIEW

2026-11-21 PST · Before-commit review of the R3 fixes · Base read `ab2d330bb5bd74deed716c823f2cfb9efd7fa60c`.

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
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-MINER-SKIP-SWITCH_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-SKIP-F1_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R1_ALPHA_REVIEW_REQUEST_r3_fixes.md
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
| Q2 · Isolation | OBJECTION | THEORETICAL — Two overlapping Step 1 publishers in the same initially unmarked root can merge across arms after the baseline recheck. |
| Q3 · Anchors | NO OBJECTION | — |
| Q4 · Validation | OBJECTION | THEORETICAL — A non-current configuration with `skip_min.max=256` and `skip_max.max=250` can pass both preflights and fail inside Optuna when it reaches that endpoint. |

## Finding 1 · Governing records omitted from the request

- **ALREADY RECORDED.** D3.5 REV3.1 §8b pins parent selection to the authenticated live `current` pointer. With no explicit prior, the finalizer selects that pointer's target. §7.2 freezes the publication and durability sequence. The root marker check and selection of the merge parent are separate operations. `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md:316-350`, `:542-560`, `utils/run_finalizer.py:1269-1310`, `:1622-1641`.
- **ALREADY RECORDED.** The 2026-11-14 owner ruling binds Optuna storage to the checkout containing the module. A separate working directory controls publication and the default search ledger. Optuna storage stays with the checkout. Stage 4 uses a distinct checkout and working directory per arm. The storage-path suite still passes for fresh and resumed URLs. `docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md:82-87`, `:130-144`, `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:91-115`, `window_optimizer_bayesian.py:17-23`, `:673-698`, `database_system.py:45-54`, `window_optimizer_integration_final.py:3750-3754`, `tests/test_step1_bayes_storage_path.py:210-222`.

## Prior record

- I checked T1–T5 against the cited Gate 12, backup, study-selection, pin, search-link and WAL records. The owner entry for *quickest and least recoding* is at line 51. I also searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, and the wider docs and exchange with `rg` for same-root publication, concurrency and locking, recovery, Optuna worktree storage, and sampling bounds. I checked the prior R3 review and the switch plan. `docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R1_ALPHA_REVIEW_REQUEST_r3_fixes.md:3-28`, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47-51`.
- **ALREADY RECORDED** — publication can precede its search link, immutable SQLite reads can miss a WAL-only commit, unnamed study discovery is newest-first, the Run 4 pin is all seven values with a one-trial budget, and miner trials are serial. The earlier backup-policy gap still covers the accumulator. `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:184-201`, `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:24-44`, `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md:78-94`, `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1151-1166`, `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:116-130`, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:49-51`, `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:181-183`, `docs/BACKLOG.md:421-447`.
- **NEW** — the overlap reproduction after a successful baseline recheck and the gaps in validating the full sampling domains. **CHANGES A RULING** — none requested here. The D6.2 §4.3 component remains with Michael. `docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R1_ALPHA_REVIEW_REQUEST_r3_fixes.md:21-25`.
- **Not searched** — rigs, Proxmox, remote repositories, host archives, live ledgers, live checkpoints and live Optuna stores.

## Q1 · Switch off

- **Root.** The off arm checks marker presence and refuses a marked root at entry and before publication. This changes Step 1 admission even if an earlier switched attempt stopped before its first trial. For ordinary unmarked roots the test found no writes from the guard, including an empty root and one with a baseline generation. `window_optimizer_integration_final.py:1850-1858`, `:1870-1878`, `:2733-2738`, `:3748-3754`, `tests/test_miner_skip_switch.py:606-648`.
- **Automatic study selection.** Tagged studies are excluded from unnamed resume. Untagged candidates retain the existing mtime order and completion test. The named branch bypasses the discovery filter and retains the separate named-study preflight. `window_optimizer_bayesian.py:676-723`, `window_optimizer_integration_final.py:1783-1803`.
- **Sampling and launcher.** False bypasses sampling validation and leaves the search digest equal to HEAD's in the fixture. Default and explicit false reached the same intercepted coordinator outcome as HEAD. Invalid switch values now raise before construction. The off payload, dispatch, intercepted serve and integration-call gates matched HEAD. `utils/step1_search_ledger.py:203-235`, `window_optimizer_integration_final.py:1881-1893`, `window_optimizer.py:811-824`, `:871-898`, `tests/test_miner_skip_switch.py:778-813`, `:816-850`, `:872-879`.

## Q2 · Isolation

The original serial reproduction is closed. A switched root is marked before its first trial, so off refuses a generation whose search link was never written. The guard consults the marker independent of ledger state and catches a claim completed before the baseline publication recheck. I reran the real-finalizer fixture for those cases. Include the marker with the root in a future backup and restore procedure, since a missing marker admits off. `window_optimizer_integration_final.py:1850-1878`, `:2733-2738`, `:3748-3754`, `tests/test_miner_skip_switch.py:599-652`.

I scheduled an overlap in scratch with the exact guards and real finalizer. Off passed its entry and publication checks in an empty root. On then saw no `current`, marked the root and passed its own recheck. With either publish order, the second finalizer call accepted the first as its parent and retained its one synthetic row, with zero fresh rows. In the switched-first order, baseline inherited the switched row. In the baseline-first order, switched inherited the baseline row. Both fixtures used the same family and no search ledger. `window_optimizer_integration_final.py:1850-1878`, `:3748-3761`, `utils/run_finalizer.py:1269-1296`, `:1622-1641`.

This overlap requires two publishers sharing one root. The recorded comparison provisions separate run trees. Scope the isolation claim to separate checkout roots and serial publication, or serialize the first switched claim and baseline finalization with the same root lock held through parent selection and publication. Mixed rows prevent attribution of any hit-rate difference to one arm. The direction of the effect is unknown. Measure independently attributable pools at comparable budgets on the same unseen draws and compare each hit rate with `k/1000`. `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:91-115`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`, `AGENTS.md:19-33`.

Unnamed resume is closed for the reported tagged-study case. I ran actual Optuna selection and one stub objective with two scratch SQLite stores. Each started with one COMPLETE trial and the tagged store was newer. Baseline finished with two trials. The tagged store stayed at one. The tag is checked before selection can reach `create_study` and `study.optimize`. `window_optimizer_bayesian.py:676-723`, `:755-771`, `:824-849`.

## Q3 · Anchors

I counted HEAD lines on both sides of all 16 allowance hunks. Each side has a unique line in its corresponding HEAD file. The gate requires the literal span once, restores the base span, then compares the relevant AST. The Stage 3 suite killed the earlier CLI move in M-SW-10 and the integration preflight moves in M-SW-8 and M-SW-9. I found no remaining harmful placement within the two allowance lists. Stage 3 checks placement in the launcher and integration files. The dedicated auto-select gate exercises the new Bayesian filter. `tests/test_pwc_backup_stage3.py:94-113`, `:123-139`, `:168-179`, `:232-240`, `:755-767`, `tests/test_miner_skip_switch.py:655-705`.

## Q4 · Validation

- The earlier example is closed. Search open refused the `[6, 99]` pin with a three-trial budget and sampled `skip_max.max=256`. The child sampling check also rejected 256 at that endpoint. The tracked `skip_min=[0,10]` and `skip_max=[10,250]` pass. The Python launcher refused `None`, `"true"`, `1`, `0` and `""` before coordinator construction in the fixture. The S145 seed-domain wall still runs first. `utils/step1_search_ledger.py:203-222`, `window_optimizer_integration_final.py:1881-1893`, `distributed_config.json:84-91`, `window_optimizer.py:803-824`, `tests/test_miner_skip_switch.py:778-850`.
- Both preflights check only `skip_min.min` and `skip_max.max`. I changed `skip_min.max` to 256 in scratch with `skip_max.max` still 250. The unpinned and pinned three-trial search configurations were accepted, and the child check passed. In real in-memory Optuna I fixed the suggestion to the allowed endpoint 256. Sampling then raised at `suggest_int("skip_max", 256, 250)`, before an objective could receive a pair. Validate the full domains and their ordering, including `skip_min.max <= skip_max.max`, so every allowed `skip_min` has a legal `skip_max`. `utils/step1_search_ledger.py:217-222`, `window_optimizer_integration_final.py:1881-1893`, `window_optimizer_bayesian.py:537-576`.
- Search open accepts a fresh one-trial pinned configuration with sampled `skip_max.max=256`. I confirmed the child rejects that configuration unconditionally. The Run 4 pin rule supports exempting a proven fresh pin-only run. Source inspection also shows that the named-study arm check can load a store before the bounds check, and the serial optimizer later reloads the bounds for actual use. Apply the same admission rule for a fresh pin-only run and pass the validated bounds object on the sampling path. Retain the per-trial wall. `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:116-130`, `utils/step1_search_ledger.py:213-222`, `window_optimizer_integration_final.py:1786-1803`, `:2733-2738`, `:3418-3420`, `:3555-3567`, `window_optimizer_bayesian.py:788-826`, `miner/range_miner_coordinator.py:12558-12565`.

## Verification

I ran the suites sequentially with `~/venvs/torch` activated. The disposable checkout copy and scratch fixtures were under `/tmp/tfm-beta-skip-switch-2.8D5YdN/`. I ran the root guard, finalizer, search configuration and Optuna in scratch. S7, sieve execution and fleet behavior were outside these probes.

| Suite | Gates passed | Mutants killed |
|---|---:|---:|
| `tests/test_miner_skip_switch.py` | 27/27 | 37/37 |
| `tests/test_pwc_backup_stage3.py` | 14/14 | 28/28 |
| `tests/test_extract_miner_bundle.py` | 31/31 | 28/28 |
| `tests/test_step1_bayes_storage_path.py` | 3/3 | 4/4 |

The attachment hashes and prior R3 review hash match the request. The three scoped HEAD diffs match their attachments byte-for-byte. `git diff --check` passed. The 33-suite matrix is Alpha's reported evidence. Rig execution, GPU equivalence, wide-range lease and volume behavior, and unseen-draw hit rates are **UNAVAILABLE**. Michael's instruction to hold landing while rig-6600 is unhealthy still applies. `docs/exchange/to_beta/X-MINER-SKIP-SWITCH-2_R1_ALPHA_REVIEW_REQUEST_r3_fixes.md:35-52`, `:114-145`.
