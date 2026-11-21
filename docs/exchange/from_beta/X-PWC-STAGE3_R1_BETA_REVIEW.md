# X-PWC-STAGE3 · R1 · Team Beta review

**2026-11-21 PST.** Reviewed the combined `pwc-backup` worktree at base `b2efdefbab207bb82ed8d307d295737bbe6111fd`. Below is `git status --porcelain` at the initial read. Michael decides on the commit.

```text
 M agent_manifests/window_optimizer.json
 M agents/watcher_agent.py
 M docs/PWC_BACKUP_OWNER_RULINGS_20261119.md
 M docs/VM101_HEADLESS_STARTUP_20261119.md
 M scripts/prove_unchanged_outside.py
 M sieve_filter.py
 M test_persistent_worker_harness.py
 M tests/test_prove_unchanged_outside.py
 M tests/test_s172_miner_seed_start.py
 M tests/test_s172_phase1_scaffolding.py
 M tests/test_step1_search_record_stage1.py
 M window_optimizer.py
 M window_optimizer_integration_final.py
?? data/
?? docs/SESSION_CHANGELOG_20261120_PWC_BACKUP.md
?? docs/TFM_SESSION_HANDOFF_20261120.md
?? docs/drafts/pwc_backup_stage3_v1/
?? docs/drafts/pwc_backup_stage3a_v1/
?? docs/drafts/pwc_backup_stage3b_v1/
?? docs/exchange/to_beta/X-PWC-STAGE3_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md
?? scripts/compare_engine_outputs.py
?? tests/test_compare_engine_outputs.py
?? tests/test_pwc_backup_stage3.py
?? tests/test_pwc_stage3a.py
```

## Prior record

I searched `docs/` with `rg` for PWC readiness, `min_workers`, worker/chunk sizing including zero and negative caps, engine defaults, credit and completion, mixed deployments, and EXEC-PIN-1. I checked the exchanges, catalog, governing proposals and rulings, chapter audits, consolidated TODO, handoffs, and Stage 3 drafts. I read both Stage 3 READMEs and compared the combined diff with the base.

| Record | Label | Effect on this review |
|---|---|---|
| `docs/CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md:86-100`, `docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195`, `docs/exchange/from_beta/X-PWC-CACHE-RESET_R2_BETA_REVIEW.md:91-93` | CHANGES A RULING | Stage 3 restores a full-fleet utilization check Michael previously directed the team to leave retired. See Finding 1. |
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:14-29`, `:43-51` | CHANGES A RULING | The newer decisions govern Step 1's default, PWC credit and nonzero starts, the window/phase split, and the separate skip-range and parallel-work scopes. |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:44-67`, `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:86-100` | ALREADY RECORDED | The earlier partial-score path, the one-read issue, the limits of the proof tool and EXEC-PIN-1. |
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:183-195`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:37-40`, `:98-103`, `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R1.md:14`, `:94` | ALREADY RECORDED | S7 requires exact sweep evidence and permits zero-survivor credit. The completion plan still lists the former WATCHER default. Route A's manifest constraint concerns its warm-start patch. |
| `docs/ROCm_Saturation_Report_S172.md:27-35`, `docs/drafts/pwc_backup_stage3_v1/PLAN.md:248-255` | ALREADY RECORDED | Six workers per rig at 50,000 seeds per chunk is a documented operating point. A smaller pool is a sizing choice distinct from worker loss. |
| Nonpositive PWC chunk cap | NEW | I found no prior record of this PWC startup acceptance followed by a non-advancing partition loop. See 3b-Q3. |

I did not search other worktrees, remote repositories, ser8, project-chat archives, the rigs or Proxmox. I took the relayed owner rulings in the request as given.

## Positions

| Question | Position | Severity |
|---|---|---|
| 3a-Q1 RANGE-MINER | NO OBJECTION | — |
| 3a-Q2 Finding 2 | NO OBJECTION | — |
| 3a-Q3 Old-style jobs | NO OBJECTION | — |
| 3a-Q4 Proof tool | SUGGESTION | — |
| 3a-Q4b Narrowed header check | SUGGESTION | — |
| 3a-Q5 `--keep-block` | SUGGESTION | — |
| 3a-Q6 Test edits | NO OBJECTION | — |
| 3b-Q1 RANGE-MINER, byte for byte | NO OBJECTION | — |
| 3b-Q2 New default | NO OBJECTION | — |
| 3b-Q3 Choosing PWC | OBJECTION | LIVE — The ordinary PWC CLI accepts a zero AMD chunk cap and can start workers before hanging in chunk planning. |
| 3b-Q4 Earlier Q5 | NO OBJECTION | — |
| 3b-Q5 Credit | NO OBJECTION | — |
| 3b-Q6 One-sided deployment | SUGGESTION | — |
| 3b-Q7 Ready gate | OBJECTION | LIVE — All workers in the documented six-per-rig configuration can be ready and the new default still refuses the run. |
| 3b-Q8 Test edits | NO OBJECTION | — |
| C-Q1 Carried items | SUGGESTION | — |
| C-Q2 W-NO-WEAKENING and Gate 12 | NO OBJECTION | — |

## Findings

1. **Governing records omitted.** The August 4 owner decision explicitly leaves the PWC full-fleet readiness requirement retired. The Stage 3 plan presents the missing hop as new and proposes restoring it. Michael's November rulings give PWC credit and set the default engine. I found no specific reversal of the readiness decision in the supplied record. Record the conflict and put the intended minimum to Michael if Alpha wants to retain the restored wiring. The coverage requirement is already stated in the S7 proposal. The completion plan's old WATCHER default also belongs in the record of documents affected by the November 20 decision. `docs/CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md:86-100`, `docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195`, `docs/drafts/pwc_backup_stage3_v1/PLAN.md:49-56`, `:255`, `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:20-23`, `:46-51`, `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:183-195`, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:40`, `:103`.

## Reasoning by question

**3a-Q1.** The `sieve_filter.py` diff is inside `execute_sieve_job`. The integration imports the legacy draw loader and PWC shim, and the miner branch uses its own residue loader and returns before the PWC branch. I checked that the PWC coordinator, its transport files and the miner directory match the base. The combined CPU comparison also passed for the miner branch and its coordinator call. `sieve_filter.py:597-628`, `window_optimizer_integration_final.py:52-58`, `:273-295`, `:1841-1842`, `:2014-2024`, `tests/test_pwc_backup_stage3.py:100-133`, `:307-344`.

**3a-Q2.** The new-key branch opens and parses once. It uses the same session-filtered rows for the short-file check, anchor bounds, slice and draw conversion. For the same file, width, sessions and valid anchor, the miner loader applies the same filter, slice and conversion. I reran the fixture that alternates the would-be second read and the direct miner-loader comparison. The short-file error is intentionally different from Stage 2b's error. The deferred fractional-key issue is still the recorded 2B Q3. `sieve_filter.py:554-580`, `:606-625`, `miner/range_miner_worker.py:904-932`, `tests/test_pwc_stage3a.py:203-249`.

**3a-Q3.** The legacy coordinator uses `offset`, and the ZMQ worker translates the legacy chunk offset before calling the same entry point. Those jobs take the existing loader branch. The fixture passed for offsets 0, 1 and the clamped 5. `coordinator.py:2327-2337`, `zmq_sqlite_worker.py:93-111`, `sieve_filter.py:554`, `:626-628`, `tests/test_pwc_stage3a.py:251-264`.

**3a-Q4.** The tool masks allowed definitions, so their reachable side effects and consumers still need inspection. I confirmed two passing comparisons with observable behavior changes. Removing an existing decorator changed the result from 4 to 2. Inserting an optional positional parameter ahead of an existing one preserved the relative order of common names and changed what a two-argument caller received from 9 to 2. Annotation attributes, subscriptions and type operators can also invoke user-defined behavior despite the AST classification. Keep the independent call-site and whole-function checks where compatibility matters. `scripts/prove_unchanged_outside.py:102-123`, `:220-235`, `:306-331`, `:482-509`.

**3a-Q4b.** Yes. In a CPU control, the allowed default `helper()` kept identical syntax while an independently allowed edit to `helper` changed the value from 1 to 2. The report passed even with a trailing optional parameter to exercise the header comparison. Stage 3b's existing `16 * 1024 ** 3` depends only on literals and has no corresponding lookup. Removing a parameter should fail when the proof is used to establish caller compatibility. A test keyword call raised `TypeError` after the tool merely reported the removal. The compatibility check should also account for removal of an existing default and insertion ahead of existing positional slots. `scripts/prove_unchanged_outside.py:286-333`, `:499-509`, `:691-695`, `window_optimizer.py:759-762`.

**3a-Q5.** The block and preceding statements are a useful check for this top-level branch when paired with the whole-function comparison. Alone, the check also passed when the unchanged block was nested under a new false predicate at the same top-level index. A second control passed after adding a later assignment that made the kept branch's global read local and raised `UnboundLocalError`. Preserve the enclosing control path or restrict the option to top-level targets. Continue comparing all other statements for this integration, since Python determines local bindings across the function. `scripts/prove_unchanged_outside.py:375-430`, `window_optimizer_integration_final.py:1841-1842`, `tests/test_pwc_backup_stage3.py:100-133`.

**3a-Q6.** The changed decorator test now requires the tool to catch evaluation it previously allowed. The body-only and unchanged-decorator controls remain. The proof-tool suite passed 49/49. `tests/test_prove_unchanged_outside.py:93-105`, `:324-344`.

**3b-Q1.** The focused checks reproduced the explicit Run-4-shaped manifest merge, coordinator settings and miner call arguments. The full statement comparison preserved the miner preamble, branch and return. The added WATCHER keyword is false for the explicit miner launch, where dataset refusal already uses `miner_backed`. The default route change is covered in 3b-Q2. `gate12_launch.sh:465-473`, `tests/test_pwc_backup_stage3.py:100-133`, `:268-286`, `:307-393`, `agents/watcher_agent.py:2117-2126`, `miner/dataset_authority.py:1109-1123`.

**3b-Q2.** I found no unsafe execution caused by the manifest switch. WATCHER merges the declared defaults before choosing its execution set and derives miner admission from `worker_pool_size`. A default request of eight is bounded by the selected worker identities, and the miner consumes that effective admission. Bare `window_optimizer.py` still takes the legacy backend if no engine flag is present. A WATCHER override setting only the PWC or ZMQ flag now also inherits the miner flag and encounters the existing CLI mutex. The generic agent's argument builder emits values for booleans against flag-only parser options, so that path already has an argument mismatch. I found no evidence of it silently invoking the new default. Route A's seven operator-only warm-start keys exclude both engine flags. `agent_manifests/window_optimizer.json:263-272`, `agents/watcher_agent.py:1749-1782`, `:1809-1832`, `:2412-2428`, `execution_set.py:645-676`, `miner/range_miner_coordinator.py:560-577`, `window_optimizer.py:1572-1579`, `:1636-1637`, `:1770-1781`, `:1810-1817`, `agents/agent_core.py:250-253`, `:357-364`, `agents/watcher_agent.py:511-519`, `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R1.md:14`, `:94`.

**3b-Q3.** The normal remote TCP path checks for the dial address and rejects an oversized AMD pool/chunk pair before PWC worker startup. An invalid lower bound bypasses that protection. With a valid dial address and `--seed-cap-amd 0`, the integer parser accepts the flag and the envelope compares only upper bounds. I ran the actual startup validator on a three-rig fixture and bounded the actual chunk-loop body to two iterations. Cap 0 produced two copies of `[1000, 1000)`. Cap −1 moved the cursor backwards twice. If readiness succeeds, the real loop never reaches dispatch and the chunk list keeps growing. Reject nonpositive effective chunk caps before worker startup. This defect is in the existing coordinator and is reachable through Stage 3's newly wired PWC launch. `window_optimizer.py:1580-1593`, `:1950-1955`, `:2068-2076`, `persistent_worker_coordinator.py:549-568`, `:1049-1083`, `:2083-2104`, `:2657-2670`.

The dial-address requirement is for remote TCP. Local TCP workers dial loopback. Explicit constant-only SSH at anchor zero can start and run its passes without a TCP dial address, then the optimizer rejects its non-TCP evidence before scoring. The plan retains SSH diagnostics, so document the standalone diagnostic route and consider refusing SSH sooner in the scoring entry point. `persistent_worker_coordinator.py:1061-1065`, `:1531-1553`, `:2626-2642`, `:2694-2738`, `window_optimizer_integration_final.py:1442-1452`, `:2072-2078`, `docs/drafts/pwc_backup_stage3_v1/PLAN.md:251-252`.

**3b-Q4.** On the updated serial path both gates precede the adapter. The producer constructs a complete pass only after all planned chunks complete and its intervals cover the request. I found no fresh current-code result with missing coverage, a non-TCP pass or unverified threshold reaching the adapter or score. In an in-memory Optuna fixture, a missing-chunk exception set trial 0 to `FAIL` and the second requested objective did not start. The production call supplies no exception catch list and records a raised search as uncredited. This matches the miner's missing-assembly failure behavior. `persistent_worker_coordinator.py:725-771`, `:2413-2457`, `window_optimizer_integration_final.py:1429-1462`, `:2072-2083`, `:3386-3405`, `window_optimizer_bayesian.py:575-605`, `:838-840`, `miner/step1_ingress.py:192-200`.

**3b-Q5.** The focused credit gate exercised complete four-pass, zero-survivor and constant-only results. All three qualified. An SSH result, unverified threshold, or short swept interval did not qualify. Credit also requires `COMPLETE` in this chunk and exact interval equality for every required phase. Reopening an old study does not furnish that chunk evidence. This verifies the credit decision with fixtures. Publication on a real PWC run is **UNAVAILABLE**. `window_optimizer_integration_final.py:1388-1394`, `:1465-1483`, `:1751-1783`, `:3546-3565`, `utils/step1_search_ledger.py:1044-1086`, `tests/test_pwc_backup_stage3.py:514-540`.

**3b-Q6.** With Stage 3 on VM101, a pre-2b worker or sieve missing the required echo fails its chunks. A Stage 2b sieve on stable identical data can still complete using the older two-read implementation. The gate also rejects a coordinator result wholly lacking completion accounting. In the reverse direction, the decisive version is VM101's optimizer integration. An older optimizer can still score a partial result. At seed start 0, I injected the same incomplete coordinator reply with one bidirectional record into both adapters. The base produced `COMPLETE`, score 1, one accumulated record and an accumulator flush call. Stage 3 produced `FAIL`, no score, no record and no flush call. Base PWC's empty sweep evidence still denies credit. In an older-coordinator deployment against the new worker, old-key jobs are refused, so timeouts or refusals can leave a mixed population if a matching worker still finishes some chunks. This is the recorded 2b mixed-version issue on the old consumer. Verify PWC parity and upgrade the VM101 integration and coordinator together before using the credited route. The fixture establishes scoring behavior. Any effect on unseen-draw hit rate is unknown and would require a matched evaluation. `persistent_worker_coordinator.py:1858-1877`, `:2374-2436`, `persistent/pwc_worker_service.py:563-592`, `b2efdef:window_optimizer_integration_final.py:1996-2007`, `window_optimizer_integration_final.py:2072-2083`, `utils/step1_search_ledger.py:1063-1085`, `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:59-61`, `docs/TFM_SESSION_HANDOFF_20261120.md:63-64`.

**3b-Q7.** Six workers on each of three healthy AMD rigs gives 18 workers. The checked-in VM101 config adds one, for 19. Even a two-slot local profile would give only 20. The CLI and manifest supply 24, and Stage 3 now passes that value to startup and the per-pass ready check. The frozen execution set's recorded clamp does not control PWC's threshold. The bounded CPU fixture called the actual ready method with 19 of 19 ready. At the base minimum of 1 it returned 19. At 24 it timed out, called the cleanup stand-in and raised before dispatch. A smaller admitted pool may take longer, and PWC judges the completed interval separately from the number of ready workers. Preserve the previously effective minimum unless Michael supersedes the August ruling. If a minimum is restored, verify the six-per-rig launch by checking the number it can start against the number it must admit. `distributed_config.json:4-38`, `agent_manifests/window_optimizer.json:264-271`, `window_optimizer.py:1585-1589`, `:1835-1840`, `:1952-1955`, `window_optimizer_integration_final.py:2064-2066`, `persistent_worker_coordinator.py:1379-1382`, `:1535-1542`, `:1578-1584`, `:1622-1659`, `:2062-2077`, `:2413-2436`, `:2592`, `docs/ROCm_Saturation_Report_S172.md:27-35`.

**3b-Q8.** The diff preserves the miner dispatch, interval and credit assertions and their mutants. The former miner default assertion is explicitly changed under the 15:53 decision, and the shared Run 4 route test keeps its miner argv and guard assertions. The refreshed M8 and M-ROUTE-START still target the ZMQ refusal. I exercised G-R5-ROUTES with a scratch dataset and saw M6, M7 and M8 fail at their intended assertions. M9's original mutation can no longer affect `run_with_config`, which supplies no ZMQ argument to a guard now keyed only on ZMQ. The harness diff is confined to T20 and supplies completion evidence to its PWC stand-in. `tests/test_s172_miner_seed_start.py:585-656`, `:888-935`, `tests/test_step1_search_record_stage1.py:1540-1554`, `:1575-1592`, `:1740-1742`, `tests/test_s172_phase1_scaffolding.py:175-189`, `window_optimizer.py:704-712`, `:1196-1209`, `:1261-1268`, `test_persistent_worker_harness.py:756-785`.

**C-Q1.** Q5's ingress wall and Finding 2's one-read change are verified. Q8 is covered for the actual miner branch and the broader AST proof has the limits above. The fractional-key Q3 remains deferred by the base commit's owner decision. EXEC-PIN-1 is now cited and the scan and separate suite passes provide compatibility evidence within their scope. Correct the scan's assertion that `compare` kept its signature. It gained optional trailing `keep_blocks=()`. Also qualify “only if it drives the PWC route,” since a caller of the proof tool can see stricter behavior independently. The over-approximation and staged passes are the evidence supplied. A combined rerun of all 18 pinned-test files is **UNAVAILABLE** in this review. `docs/drafts/pwc_backup_stage3b_v1/exec_pin_1_scan.txt:50-73`, `b2efdef:scripts/prove_unchanged_outside.py:161`, `scripts/prove_unchanged_outside.py:433`, `:482-514`, `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:86-100`.

**C-Q2.** W-NO-WEAKENING compares the producer, finalizer and ignore file against the current HEAD and checks for prohibited residue exceptions. Applying the reviewed change to HEAD resolves its producer mismatch when the launch checkout is clean. I found no Stage 3 change to the imported clean-tree predicate or the local/remote worker entry point. I ran the parity script's static closure function only and got `(True, [], [])`, with ten derived files and ten governed files. None of the ten differs from the base. I also read the manual launcher. Its local and remote worker commands still invoke `miner/range_miner_worker.py`. The launcher's provenance block can separately refuse failed hash collection or a failed write. A commit in this review worktree would still leave the untracked `data/` and exchange artifacts visible if they remained there. Gate 12 changes to the production checkout at entry, and its clean-tree verdict depends on that checkout's actual status at both checkpoints. Post-commit launch, remote parity and the production checkout's status are **UNAVAILABLE**. `tests/test_gate12_cleantree_admission.py:938-958`, `scripts/gate12_cleantree_gate.py:74-81`, `:108-124`, `window_optimizer_integration_final.py:101-122`, `scripts/gate12_parity_gate.py:174-185`, `:262-291`, `gate12_launch.sh:79-80`, `:133-139`, `:191-198`, `:246-252`, `:284-311`, `scripts/launch_fleet_manual.sh:329-351`, `:393-403`.

## Evidence limits

I ran the four focused suites in a scratch copy with the Torch environment active, GPU visibility disabled and remote command stubs on PATH. The proof tool passed 49/49. Stage 3a passed 6/6 and caught 3/3 mutants. The combined Stage 3 suite passed 14/14 and caught 18/18 mutants. The output comparator passed 13/13. I also ran the bounded CPU checks described above and confirmed the tracked review inputs matched the scratch copy at the closing check. Shared full-suite results and the T20 full-harness rerun are Alpha's evidence.

Actual WATCHER execution on the new default, GPU work, deployed parity, a counted PWC run, engine output equivalence and holdout hit rate are **UNAVAILABLE**. The recorded constant-phase skip-range difference remains a confound for the owner-run engine comparison. Pin both arms to `[0, 16]` for that comparison or use the later miner fix. The hit-rate measurement is the pool's performance against `k/1000` on draws held out from the sieve. `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47-51`, `docs/drafts/pwc_backup_stage3_v1/PLAN.md:297-308`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`.
