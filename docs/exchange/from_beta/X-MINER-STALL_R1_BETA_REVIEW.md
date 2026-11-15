# X-MINER-STALL · R1 · BETA REVIEW

## Prior record

**Searches run, 2026-11-14:**

- Read the request, `AGENTS.md`, the admission-stall brief and the candidate report. Read the target worktree's diff against `7b9184a`, its changed production definitions and their surrounding completion, admission, executor and teardown paths; read the new tests, controls and capture scripts.
- Read the admission-liveness and admission-binding briefs, Correction 6, Correction 4, the BP amendment, and the relevant BP remediation/report passages. Searched the dispatch-stall report and R-1…R-4 changelog; read MP-1's measured contention account.
- Read X-MINER-START R1 Q6/Q7, X-RUN4-RANGE R1 Q4, the final R2 routing review, SR-1 in the catalog and Field-6 ruling, binding addendum §A, and the relevant completion/delegated-execution plan passages.
- Ran `rg -n -i '_pump_deferred|admission liveness|staging.admission|frozen.*oracle|immutable=1|SR-1'` across `docs/TB_RULING_*.md`, `docs/PROPOSAL_*.md`, `docs/CHAPTER_*AUDIT.md` and `docs/TODO_CONSOLIDATED.md`. Searched the affected tests for declaration sets, mutation behavior, scratch writes and pump calls. Read Alpha's saved admission, seed-start and routing results and the baseline reproduction records.

| Item / governing record | Label | Disposition |
|---|---|---|
| L-4, L-5, evidence-reader open; the three-part repair brief | ALREADY RECORDED | Reviewed here; no rediscovery claim. |
| Correction 6 serialization; Correction 4 dispatch constraint; BP pump-before-resume and resume-credit rules; R-1/MP-1; SR-1 | ALREADY RECORDED | Retained by the reviewed change. |
| X-MINER-START Q6/Q7 and X-RUN4-RANGE Q4 | ALREADY RECORDED | The requested repair and independent successor comparison are supplied. |
| Final R2 routing review and the owner-updated completion/delegation plans | ALREADY RECORDED | Make these explicit in the operative inventory; finding 1 below. |
| This review's disposition of X-MINER-START Q7 | CHANGES A RULING | Withdraws Beta's earlier launch objection on the reviewed bytes; does not authorize a launch. |
| New implementation defect | NEW | None established. |

**Finding 1 — governing-record inventory.** The request cites the two recent exchange reviews, but does not explicitly inventory the final routing review, `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:79-109`, `:145-153`, or the owner-updated launch records, `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:18-23`, `:73-94` and `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:50-63`. Carry them when stating what remains before Run 4. This is an omission from the record inventory, not a new implementation defect. In particular, the newer owner record removes the older power-cap/telemetry conditions while retaining netconsole and the applicable launch gates. It does not make this repair a complete launch-readiness assessment.

**Not searched/exercised:** private repository, ser8, rigs or Proxmox; live ledgers; GPU/kernel execution; actual fleet staging, publication, cleanup or prediction-pool hit rate. The documentation search was targeted, not a claim to have read all of `docs/`.

## Review identity and tree state

- Thread / round: **X-MINER-STALL / R1**.
- Date: **2026-11-14**, America/Los_Angeles.
- Class: **BEFORE-COMMIT**.
- Base read: **`7b9184a95cd125c14685008cd306fe0fd9ff3040`**.
- Candidate root: **`/home/michael/tfm_worktrees/miner-stall/`**. Code/test anchors below are relative to this root unless identified as main-tree exchange records.
- Main-tree `git status --porcelain` at opening:

```text
?? docs/exchange/to_beta/X-MINER-STALL_R1_ALPHA_REVIEW_REQUEST_admission_stall_repair.md
```

Candidate `git status --porcelain` at opening, unchanged at the 18:22 PST recheck:

```text
 M agents/watcher_agent.py
 M docs/LEADS.md
 M miner/range_miner_coordinator.py
 M tests/fixtures/miner_seed_start_driver.py
 M tests/fixtures/miner_seed_start_zero_control_b11b6fd.txt
 M tests/fixtures/run4_pin_harness.py
 M tests/fixtures/run4_routing_clean_control.py
 M tests/test_s172_miner_seed_start.py
 M tests/test_s172_mp1_drain_attribution.py
 M tests/test_s172_r1_drain_remedy.py
 M tests/test_s172_run4_routing_patch.py
 M window_optimizer_integration_final.py
?? data/
?? docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md
?? tests/capture_miner_seed_start_zero_control.py
?? tests/capture_run4_search_control.py
?? tests/fixtures/admission_stall_driver.py
?? tests/fixtures/admission_stall_one_worker_control_7b9184a.txt
?? tests/fixtures/l5_fresh_process_probe.py
?? tests/fixtures/run4_routing_search_control.py
?? tests/fixtures/run4_search_control_b11b6fd.txt
?? tests/test_s172_admission_stall.py
?? tests/test_s172_small_repairs_part3.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| 1. Completion ordering and admission | NO OBJECTION | — |
| 2. Dispatch and back-pressure ordering | NO OBJECTION | — |
| 3. Double release / superseded or terminal attempts | NO OBJECTION | — |
| 4. One-worker control and SR-1 | NO OBJECTION | — |
| 5. Forced multi-worker orderings without a nudger | NO OBJECTION | — |
| 6. Frozen oracle, successor and mutant credit | NO OBJECTION | — |
| 7. Part 3 repairs | NO OBJECTION | — |
| 8. Earlier Run-4 launch objection | NO OBJECTION | — |
| 9. Governing record | SUGGESTION | — |

## Q1 — the missing wake is repaired

The successful staging-job path finalizes and pumps (`miner/range_miner_coordinator.py:8230-8234`). The StripeComplete path now finalizes and, if complete, submits the pump to the staging executor (`:11897-11926`). Finalization serializes publication and the `done` transition under `_lifecycle_lock` (`:4268-4294`). Thus staging-first reaches the new hand-off; completion-first reaches the staging-job pump; concurrent finalizers retain a following pump on whichever path completes the stripe.

This establishes a following wake under normal executor operation, **not atomic removal of admission at the `done` write**. A `done` key can exist transiently until the queued pump runs. The indefinite missing-wake state is what the repair prevents. Executor refusal is logged (`:11922-11926`); normal executor shutdown belongs to serve-loop teardown (`:10677-10681`). This is not a guarantee against arbitrary executor/process failure.

Independently, G-UNIT completed the real dispatch sequence, released s0's admission and staged deferred s1. Removing the hand-off or inverting its condition reproduced the specific stall; see the assertions at `tests/test_s172_admission_stall.py:245-285` and mutation specifications at `:535-544`.

## Q2 — no added wait on the dispatch thread; pump ordering retained

The added path submits a callable; it does not execute the pump or wait on its future (`miner/range_miner_coordinator.py:11919-11926`). Executor construction has no staging-slot acquisition (`:3848-3853`). The pump still acquires slots nonblocking, submits retained work, and resumes paused readers in its `finally` (`:7954-7959`, `:8100-8115`). No admission/pump/resume definition changes in the diff.

G-OFFTHREAD and the deterministic runtime observation passed. The direct-dispatch-pump mutant failed on its actual caller. R-1's full battery passed **44/44**, including resume ordering and credit behavior. This is a statement about the added path and unchanged ordering, not a claim that existing SQLite/lock work has zero latency. MP-1 remains a contention diagnosis; L-4 is the distinct missing-wake defect already recorded in this packet.

## Q3 — no new release or admission authority

The hand-off adds no reservation release or admission-state mutation (`miner/range_miner_coordinator.py:11897-11926`). Terminal/superseded admitted keys are removed with idempotent `pop` (`:4783-4797`). The existing pump probes attempt/trial liveness before admission and forces subsequent admitted frames to probe again (`:4769-4781`, `:7918-7955`); `_try_admit_locked` remains the sole grant site (`:4832-4857`). An extra pump therefore does not itself grant a terminal attempt or release a staging slot twice.

The existing dead-attempt and death-between-probe-and-grant checks passed, as did the full R-1 battery (`tests/test_s172_r1_drain_remedy.py:760-827`). These verify the retained mechanism; they do not upgrade it into an atomic transaction against every conceivable concurrent writer.

## Q4 — output control retained; one-worker liveness can improve

The one-worker comparison checks completed-run normalized assignments, stripes and ledger contents, with control/driver/base hashes bound (`tests/test_s172_admission_stall.py:387-417`). Its normalization intentionally removes volatile fields and checks payload hashes before substitution (`tests/fixtures/admission_stall_driver.py:291-335`). Alpha's saved G-ONE result is **16 stripes, 16 assignments, seven tables / 84 rows equal**, digest prefix `49ac9f233215`. Independently rerunning this socket-dependent case was **UNAVAILABLE**.

Do not describe all one-worker behavior as unchanged: the recorded base stall can occur with one worker too. The repaired wake applies to it. The equality claim concerns the retained normalized outputs of successful runs, not timing, every raw ledger byte, or preservation of a hang.

SR-1 is satisfied: R-1 adds `_serve_dispatch` with provenance (`tests/test_s172_r1_drain_remedy.py:2159-2166`); MP-1 already declared it and adds provenance only (`tests/test_s172_mp1_drain_attribution.py:1420-1427`). Both exact historical scope gates passed independently; neither anchor or equality assertion was relaxed (`tests/test_s172_r1_drain_remedy.py:2173-2186`, `tests/test_s172_mp1_drain_attribution.py:1478-1491`). The new base scope check also passed: one changed definition, none added/removed, 17 protected surfaces unchanged (`tests/test_s172_admission_stall.py:490-521`).

## Q5 — the tests force the intended ordering

Observers invoke the original methods once per production invocation (`tests/fixtures/admission_stall_driver.py:107-162`). Staging-first delays StripeComplete until job returns and following pump returns; completion-first holds the final staging job until serve-thread finalization has returned (`:164-207`, `:269-288`). A timeout is recorded as a forcing failure. The gate requires commitment, all stripes done and successful forcing on every stripe (`tests/test_s172_admission_stall.py:298-327`). The Run-4-shaped gate checks all four upper-half contiguous phase intervals and 128 done stripes (`:350-371`). This is real coordinator execution with fake workers and a non-assembling sink (`tests/fixtures/admission_stall_driver.py:52-65`, `:337-375`).

The candidate driver and seed-start suite remove their nudger calls (`tests/fixtures/miner_seed_start_driver.py:33-37`, `tests/test_s172_miner_seed_start.py:793-800`). The remaining timer is explicitly limited to recapturing historical zero-start controls on clean `b11b6fd`; capture refuses changed content digests (`tests/capture_miner_seed_start_zero_control.py:52-64`, `:70-88`, `:108-131`). It supplies no current liveness pass.

I read Alpha's saved `data/miner_stall/admission_stall_full.log`: **120/120** forced small runs, **3/3** Run-4-shaped runs, **11/11** gates and **4/4** mutation executions. The nudger-free seed-start log reports **10/10** and **11/11** mutants. These counts are **Alpha's execution evidence**, supported by the reviewed test construction. My environment denies loopback socket creation (`PermissionError: Operation not permitted`), so independent multi-worker, G-ONE, Run-4-shape and socket-dependent seed-start reruns are **UNAVAILABLE**, not additional passes. The independently reproduced dispatch-level stall and its three specific mutant failures corroborate the mechanism.

## Q6 — the successor comparison is independent and properly scoped

The frozen fixture's digest check passes; its original no-search boundary remains intact (`tests/test_s172_run4_routing_patch.py:97-116`). Its current result is **INCOMPLETE before dispatch**, not a pass. Preserve that historical result honestly.

The successor checks fixture hashes and the committed `b11b6fd` WATCHER hash, compares the complete argv, and permits exactly the named differences from the historical argv: the upper-half search start and appended search ID (`tests/test_s172_run4_routing_patch.py:143-198`). The builder-boundary gate checks the declared search collaborator delta (`:201-246`). The capture script refuses the wrong base or dirty tracked files (`tests/capture_run4_search_control.py:42-63`).

I independently built the unpinned command from the existing **clean `b11b6fd` checkout** with scratch-only search storage: **49 tokens exactly equal** to the submitted control; all four source hashes matched. Candidate routing/containment gates passed **15/16**, with only the historical gate INCOMPLETE. M1, M2a, M2b, M3 and M4 had meaningful failures and their required green controls; M2b actually diverged at the successor oracle, rather than receiving credit from historical incompleteness (`tests/test_s172_run4_routing_patch.py:849-857`, `:928-999`).

The supplied M1b runner writes/restores repository source (`tests/test_s172_run4_routing_patch.py:878-907`), so that execution is **UNAVAILABLE to Beta**. I instead compiled the same unique WALL-2 substitution into `WatcherAgent.run_step` **in memory against production globals**. G-CHAIN/G-EXACT failed; successor, exact allowlist, partial refusal and provenance stayed green, matching M1b's specified controls (`:910-925`). The source digest remained unchanged. Alpha's saved after-log separately records the supplied M1b pass.

All six mutation classes therefore have substantive detection evidence. The whole historical suite remains **21/22**, without its completion sentinel (`tests/test_s172_run4_routing_patch.py:1009-1039`). This review accepts the explicitly scoped successor evidence for today's argv/containment comparison. It does not turn the frozen gate green or establish child/fleet execution.

## Q7 — both small repairs are sound in the reviewed path

WATCHER saves and restores the root level around the importing statement in a `finally` (`agents/watcher_agent.py:782-787`). The fresh-process check drives real `run_step` with dispatch intercepted and confirms the import occurred there (`tests/fixtures/l5_fresh_process_probe.py:53-111`). Independent result: both required records captured, root **20 → 20**; reverting the function loses both records and changes it **20 → 30**.

The evidence reader rejects nonempty WAL/journal files, otherwise opens with `mode=ro&immutable=1`, and returns no evidence on read errors (`window_optimizer_integration_final.py:1449-1465`). Independent tests verified interval reading with an unchanged directory and rejection of a demonstrably stale immutable WAL view (`tests/test_s172_small_repairs_part3.py:149-195`). Both old-reader mutants failed on the intended assertions. **Part 3: 3/3 gates, 3/3 mutants, PASS.** Nonempty WAL deliberately means UNAVAILABLE/no credit; this is not a new proof of a consistent snapshot while an unrelated writer is active.

## Q8 — the X-MINER-START Q7 objection no longer stands on these bytes

**Withdrawn for the reviewed patch.** The missing completion wake has a separate repair, a deterministic independent reproduction/mutation check, and credible multi-worker evidence without the nudger. The independent successor control and routing containment evidence also meet the earlier Q6 follow-up. No additional L-4 repair or prelaunch fleet experiment is requested by this review.

This position attaches to the reviewed files, not the unmodified main-tree coordinator. Normal landing, commit-sensitive clean-tree/Gate-22 checks, committed rig parity and current launch/provenance requirements retain their existing roles. The coordinator remains in the parity set (`scripts/gate12_parity_gate.py:174-180`). Owner-updated completion/delegation rules govern launch disposition; Beta provides review, not authorization.

The first real run must still establish actual transport/staging behavior and volume, GPU execution, production publication/cleanup and accepted run evidence. CPU fake-worker commitment supplies none of those results and measures no prediction-pool hit rate. These are the previously recorded production boundary, not a continuing L-4 objection.

## Q9 — carry finding 1

Add the final routing review and owner-updated launch plans to the operative prior-record inventory. No additional implementation defect or conflicting admission rule was established. L-4/L-5's pending commit/fleet confirmation and the historical routing INCOMPLETE result remain statuses.

## Checks actually run and limits

All Python checks activated `~/venvs/torch`, disabled bytecode/GPU visibility and used `/tmp/beta_miner_stall_r1/`. In-memory wrappers changed only test scratch locations; audit hooks, inherited by Python subprocesses, rejected writes/databases outside scratch and remote connections. Production source and assertions were not edited. Loopback denial was an environment restriction, not an automatic approval rejection.

| Independent check | Result |
|---|---|
| Admission G-UNIT, G-OFFTHREAD, G-SCOPE | 3/3 PASS. |
| Admission unit mutants: removed hand-off, direct dispatch pump, inverted condition | 3/3 specific assertion failures. |
| R-1/R-2/R-3 full battery | 44/44, exit 0. |
| MP-1 full battery | 38/38, completion sentinel PASS, exit 0. |
| Part 3 full battery | 3/3 gates, 3/3 mutants, sentinel PASS. |
| Candidate routing gates | 15 PASS; historical frozen gate INCOMPLETE. |
| Routing mutants | Five supplied safe mutants plus equivalent in-memory M1b detected, required controls green. Original file-writing M1b UNAVAILABLE to Beta. |
| Independent pre-change successor control | Clean `b11b6fd` command equals control; four source hashes match. |
| Multi-worker/one-worker socket runs, full remaining regressions, actual fleet/GPU/run evidence | UNAVAILABLE independently; Alpha's saved counts remain attributed above. |
| Whitespace and concurrent-edit check | `git diff 7b9184a --check` passed; all 22 tracked-diff/new review-source artifacts unchanged at final hash recheck. |

Reviewed production SHA-256 values:

```text
miner/range_miner_coordinator.py
b8e259b3e12b95297fa38018a41215d5c52f4d9b3f4095e384829a3ec1f1517d
agents/watcher_agent.py
f5a8d1c4d56eba1656331a87e7518f380f44f4c9e1d0f8d61737bfb3e6a0de51
window_optimizer_integration_final.py
44aebe34ae2a334e9b4db3e19886a9a10e2ff63ac2f9c96e81e83f5a32aeb9d6
```

Logs and throwaway drivers are under `/tmp/beta_miner_stall_r1/`: `admission.log`, `part3.log`, `routing.log`, `r1_full.log`, `mp1_full.log`, `scope.log`, `base_control.log`, `m1b_memory.log`, `checks.py`, `control_checks.py` and `hashes.json`. No live-state probe, real worker/pipeline/fleet launch, rig connection, package change, source restoration, commit or push occurred. Beta's only repository write is this review.

## Findings outside the numbered questions

1. The governing-record inventory omission appears first above, as required. It is an ALREADY RECORDED governance status, not a new production defect. No other new finding.
