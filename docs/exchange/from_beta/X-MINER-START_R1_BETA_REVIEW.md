# X-MINER-START · R1 · BETA REVIEW

## Prior record

**Searches and reads this session:**

- Read both checkouts' `AGENTS.md`, the complete request, the miner-start brief and Alpha's worktree report; inspected the complete tracked diff against `b11b6fd` and the new fixtures/suite.
- Searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md` and `docs/LEADS.md` for `seed_start`, `base_start`, `G-MINER-START`, `G-UNPINNED-IDENTICAL`, `finalize_stripe`, pump and the disclosed leads. Repeated the origin/control search across `docs/PROJECT_FILE_CATALOG.md`, `docs/CHAPTER_*` and `docs/*_AUDIT.md`.
- Read the Run-4 routing/geometry ruling, final R2 routing review, X-RUN4-RANGE R1 review, relevant v1.1 search-proposal passages and the final stage-1 code review. Traced the production entry, integration, stripe creation, scheduling, dispatch, evidence reader and credit decision. Read G-PROD-SHAPE's relevant checks and the staging/admission paths underlying L-4.

**Finding 1 — the carried record omits a governing instruction about the frozen control.** The brief cites X-RUN4-RANGE R1 **Q1**, but its **Q4** explicitly requires preserving the frozen cursor-zero oracle and adding a separate current-cursor case (`docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md:87-97`). The final routing review also governs this certificate: `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:79-109` and `:145-153`. This is an **ALREADY RECORDED** review position omitted from this request's operative inventory, not a newly discovered code defect. The present brief allows an assertion to remain red rather than be weakened; that allowance does not itself renew the routing certificate. See Q6/Q7.

| Item | Label | Treatment |
|---|---|---|
| Miner-origin defect, R1–R7, PWC/ZMQ refusal and scoped Route-A refresh | ALREADY RECORDED | Review against the brief and stage-1 record. |
| Frozen control and routing/containment certificate requirements | ALREADY RECORDED | Finding 1 corrects the inventory. |
| L-4 admission stall, L-5 INFO suppression, live-staging probe deviation | ALREADY RECORDED | Disclosed in this packet/report; statuses, not new findings. |
| Replacing the historical whole-argv oracle with a successor comparison | CHANGES A RULING | Requires an explicit revised comparison boundary; no silent recapture accepted here. |

**Not searched/read:** rigs, Proxmox, ser8, private/remote repositories, project PDFs, live ledgers or live staging, the other worktree, complete fleet archives, or every governing document in full. No external research was needed.

## Review header

- **Thread / round:** X-MINER-START / R1.
- **Date / host:** 2026-11-14, VM101, user `michael`.
- **Class:** BEFORE-COMMIT.
- **Reviewed worktree:** `/home/michael/tfm_worktrees/miner-start`, HEAD/base `b11b6fde7ab57321b42c418f87af8cc2c75c814d`, plus the uncommitted packet.
- **Main-tree HEAD:** `976a6dba60685925e8c3f82475be9d29a0cff0ba`.
- **Authority:** Beta records positions; Michael decides. This review authorizes no launch.

Main-tree `git status --porcelain` at initial read:

```text
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-PWC-BACKUP_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-MINER-START_R1_ALPHA_REVIEW_REQUEST_seed_start_into_sieve.md
?? docs/exchange/to_beta/X-PWC-BACKUP_R1_ALPHA_REVIEW_REQUEST_pwc_step1.md
```

Reviewed-worktree `git status --porcelain` at initial read:

```text
 M docs/LEADS.md
 M miner/range_miner_coordinator.py
 M tests/fixtures/run4_pin_harness.py
 M tests/fixtures/run4_routing_clean_control.py
 M tests/run4_prelaunch_provenance_check.py
 M tests/test_s172_admission_liveness.py
 M tests/test_s172_attempt6_remediation.py
 M tests/test_s172_f1_lease_origin.py
 M tests/test_s172_phase1_scaffolding.py
 M tests/test_s172_phase4_coordinator.py
 M tests/test_s172_phase5_d0.py
 M tests/test_s172_phase5_d1_workflow.py
 M tests/test_s172_phase5_d2_directional_uniqueness.py
 M tests/test_s172_staging_backpressure.py
 M tests/test_s172_staging_partb.py
 M tests/test_step1_search_record_stage1.py
 M window_optimizer.py
 M window_optimizer_integration_final.py
?? data/
?? docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md
?? tests/fixtures/miner_seed_start_driver.py
?? tests/fixtures/miner_seed_start_zero_control_b11b6fd.txt
?? tests/fixtures/run4_search_fixture.py
?? tests/test_s172_miner_seed_start.py
```

**Anchor convention:** source/test anchors below refer to that absolute worktree; governance anchors above refer to the main tree. All cited lines were read this session.

## Positions

| Question | Position | Severity |
|---|---|---|
| 1 — origin through partitioning and reassignment | NO OBJECTION | — |
| 2 — execution-derived interval and wrong-interval credit | NO OBJECTION | — |
| 3 — explicit zero-start behavior | NO OBJECTION | — |
| 4 — PWC/ZMQ pre-dispatch refusal | NO OBJECTION | — |
| 5 — G-PROD-SHAPE | NO OBJECTION | — |
| 6 — Run-4 fixture assertions | NO OBJECTION | — |
| 7 — remaining verification and launch readiness | OBJECTION | LIVE — the existing staging-before-StripeComplete ordering can strand admission in a normal multiworker run, so file parity alone does not establish readiness to complete Run 4. |
| 8 — missed governing record | SUGGESTION | — |

## Q1 — the supported Step-1 path carries the origin

The integration passes its own `seed_start` parameter to the miner (`window_optimizer_integration_final.py:1728-1742`). The miner rejects an omitted, bool, float or negative origin before creating its ledger, then puts the accepted value on the serve context (`miner/range_miner_coordinator.py:12461-12468`, `:12526-12535`). The serve path checks that context without a zero default and passes the origin to its stage-assignment call (`:9517-9523`, `:10338-10340`).

Stripe creation uses `base_start` through the partitioner and stores the resulting start in each row (`:4033-4057`). Later scheduling reads that row's start, and dispatch sends it (`:4229-4237`, `:12003-12008`). Thus re-placement does not need to reconstruct the origin. The other partitioner call is genuinely span-only: it discards `_start` and retains counts (`:5104-5112`). I independently checked origin-invariance of those counts and direct creation of 32 upper-half rows in each of four phases.

Independent G-HOP and G-R2-ENDS passed. The full socket-driven geometry/retry/lease gates were **UNAVAILABLE** to Beta because this sandbox rejects loopback socket creation. Alpha's saved log reports those passes; I read it but do not count it as my execution.

The supported Step-1 upper bound is enforced before sink construction/dispatch by the integration's domain wall (`window_optimizer_integration_final.py:1719-1727`), in addition to child preflight (`window_optimizer.py:806-821`). The raw miner entry itself checks the origin's type/lower bound, not `start + total <= 2^32` (`miner/range_miner_coordinator.py:12461-12468`). This position covers the reviewed Step-1 route, not arbitrary direct API calls. No coordinator-process restart/resume was executed or certified here.

## Q2 — interval credit remains tied to the run's rows

The reader requires a committed result with a run id, selects only that run's stripe rows, excludes a phase containing a non-done row, and derives contiguous endpoints from the rows (`window_optimizer_integration_final.py:1433-1470`). Credit separately requires a COMPLETE trial with evidence from the current chunk, every required phase, and exact equality to the requested interval (`utils/step1_search_ledger.py:1055-1086`). Neither function changed in this patch.

My isolated synthetic-ledger checks accepted four complete upper-half phase intervals and refused the same evidence for the lower half, a shifted phase, a gap and a non-done row. These are reader/credit checks with deliberately synthetic completion, not evidence of actual work. The existing G-R3-EVID tests include a real fake-worker run plus gap, shift, non-done and overlap mutations (`tests/test_s172_miner_seed_start.py:460-489`); their independent socket execution was unavailable.

No supported wrong-interval credit path was found. This establishes interval accounting, not authentication of a worker's computation or predictive value. The disclosed reader connection uses `mode=ro` without `immutable=1` (`window_optimizer_integration_final.py:1443`); that existing issue is unchanged. Beta opened no live database.

## Q3 — explicit zero retains its geometry; omitted zero is intentionally refused

The unchanged partitioner initializes its cursor to the supplied origin (`miner/range_miner_coordinator.py:769-780`), so supplying zero preserves the old partitioning. R7's comparison binds the captured driver and base coordinator hashes, requires the current coordinator to differ, and compares normalized stripe rows, assignments and evidence by digest (`tests/test_s172_miner_seed_start.py:641-669`). Its source is appropriate to the claim.

Alpha's saved after-log reports both zero controls byte-equal: attempt-9 shape, 128 stripes/assignments, digest prefix `972d23b4e5a3`; small control, four stripes/assignments, `737e803754f8`. I verified the control/check construction and packet stability; independently rerunning R7 was **UNAVAILABLE** due to sockets. No claim here proves unchanged GPU results. The new explicit-origin API deliberately rejects callers that omit zero (`miner/range_miner_coordinator.py:12461-12468`).

## Q4 — refusal precedes dispatch; zero-route arguments match the base

The child route guard executes before coordinator construction (`window_optimizer.py:819-826`); config mode validates every planned iteration first (`:1256-1263`). At the per-trial branches it precedes the PWC call and creation of the ZMQ session coordinator (`window_optimizer_integration_final.py:1879-1887`, `:1940-1957`).

Independent G-R5-ROUTES passed: nonzero PWC/ZMQ reached no dispatch seam; zero produced argument-identical intercepted calls to the base function; the child's positive/negative controls and config-mode whole-plan refusal passed. The patch does not edit the two route coordinators. This is dispatch-boundary equivalence, not a fresh execution of either worker system. Existing non-certification on those routes remains explicit (`window_optimizer_integration_final.py:1918-1925`).

## Q5 — no zero-start assumption in G-PROD-SHAPE

No edit is needed. The run-scoped ledger checks inspect trial/stripe/shard states, phases, acknowledgement, cleanup, remote deletion and reservations; they impose no seed-origin constraint (`tests/gate_s172_prod_shape.py:230-344`). It remains unchanged in the reviewed diff. It is also not an interval-coverage proof: use Q2's evidence checks for that claim.

## Q6 — assertions were not weakened; the certificate is still incomplete

The fixture changes supply a production-opened search with production-computed configuration (`tests/fixtures/run4_search_fixture.py:107-124`) to the intercepted WATCHER path (`tests/fixtures/run4_routing_clean_control.py:148-158`). Restoring the root logging level after preparatory configuration computation limits fixture-induced logging effects (`tests/fixtures/run4_search_fixture.py:104-115`). It does not repair production L-5.

The routing assertion file is unchanged. Its frozen-fixture digest and whole-argv equality remain enforced (`tests/test_s172_run4_routing_patch.py:105-125`); therefore the changed fixture correctly yields INCOMPLETE. The prelaunch assertions still require all seven exact values, one trial, authority/provenance and the recorded argv (`tests/run4_prelaunch_provenance_check.py:195-239`). Independent results: prelaunch **10/10**; routing positive/containment gates **13 PASS, one INCOMPLETE**. The stage-1 guard assertions change to the new, explicitly required route behavior (`tests/test_step1_search_record_stage1.py:1526-1580`); the full stage-1 battery passes.

The disclosed 15/20 routing result is not a renewed certificate. In particular M2b's apparent detection by an already-INCOMPLETE oracle is vacuous; its provenance failure supplies a separate real detection. M1/M2a/M4 still lack their required green oracle control (`tests/test_s172_run4_routing_patch.py:709-717`, `:788-846`). M1b was **UNAVAILABLE** to Beta because its harness writes and restores repository source (`:738-760`).

Keep the historical capture/hash as historical evidence. A successor check should explicitly identify the stage-1 scheduling changes, use an independently established pre-miner-change control under the new search collaborator, and preserve checks on all other argv fields and containment behaviors. Do not bless an oracle captured solely from the candidate under review. Until that comparison boundary is recorded and its controls pass, Q6's NO OBJECTION means only that the submitted assertions were not weakened.

## Q7 — carrying the file is necessary, but not sufficient to launch

**The objection is to launch readiness as stated in Q7.** L-4 is already disclosed and recorded, and is not attributed to this seed-origin diff. Its mechanism exists in production code: another admitted attempt blocks admission (`miner/range_miner_coordinator.py:4848-4853`), the staging-job path finalizes and pumps (`:8230-8234`), while the StripeComplete branch finalizes without pumping (`:11882-11897`). Terminal admission keys are removed by the pruning pass (`:4783-4797`).

I independently injected a DONE admission holder on a real coordinator/throwaway ledger: admission for the next attempt remained false; explicitly pruning restored admission. That verifies the mechanism, not its fleet incidence. Alpha records actual fake-worker stalls, including four of four at the larger geometry (`docs/LEADS.md:302-308` in the worktree). Completion in the new suite relies on an additional timer calling `_pump_deferred()` (`tests/fixtures/miner_seed_start_driver.py:55-83`). It consequently cannot settle production liveness without that timer.

No hostile process, unsupported configuration or seed-recovery assumption is needed: the relative completion of staging and receipt/processing of StripeComplete is normal concurrency. Hence this is LIVE for the launch question, although fleet incidence remains **UNAVAILABLE**. Production's default execution timeout is unbounded (`miner/range_miner_coordinator.py:12558-12561`). The capacity timer observes reader pauses/reservation waits, not an otherwise-unpaused deferred admission holder (`:5794-5816`, `:5840-5849`). A timer-assisted pass does not remove this reason to defer Run 4.

Before launch, resolve L-4 through Alpha's separately reviewed repair and exercise staging-before-StripeComplete with multiple workers **without the harness pump**. Also finish the routing-control refresh described in Q6. This does not require folding a staging repair into the current origin change. Michael decides disposition; Beta supplies no launch authorization.

What still needs the real run: real workers and GPU/kernel execution across the upper-half endpoints on both backends, actual transport/staging behavior and volume, real Phase-5 assembly/publication, cleanup, production timing and accepted run evidence. G-E2E explicitly stops before assembly and uses a synthetic history witness (`tests/test_s172_miner_seed_start.py:734-752`, `:837-850`). Its synthetic COMPLETE result is not a real published interval.

The coordinator is indeed in the rig parity set (`scripts/gate12_parity_gate.py:174-180`); the reviewed SHA-256 is `6f2d14f3a48a585c8b59ecf5c295b89bbc1c74cec1d48b569804637194c5c7f9`. Deployment parity, clean committed-tree checks, infrastructure/provenance checks and Michael's launch decision retain their recorded roles. The existing Run-4 operational claim boundary remains; neither this patch nor fake-worker tests measure pool hit rate on unseen draws.

## Q8 — add Finding 1 to the operative record

Carry X-RUN4-RANGE **Q4**, not just Q1, and the final R2 routing review's oracle/mutation/post-commit requirements. Distinguish an incomplete historical comparison from a new, explicitly scoped certificate. L-4/L-5 and the staging-probe deviation are already statuses in this packet, not additional discoveries by Beta.

## Checks actually run and limits

All Python execution activated `~/venvs/torch`, set `PYTHONDONTWRITEBYTECODE=1` and `CUDA_VISIBLE_DEVICES=''`, and used `/tmp/beta_miner_start_r1/`. Temporary wrappers redirected test scratch paths in memory. Audit hooks rejected filesystem mutations/database connections outside `/tmp` and remote socket connections. Assertions and repository source were not edited.

| Check | Independent result |
|---|---|
| Stage-1 battery | 39/39 gates; 33/33 mutants; completion sentinel PASS. |
| Prelaunch provenance | 10/10 PASS, intercepted dispatch. |
| Routing controls | 13/14 PASS; frozen oracle INCOMPLETE. Five safe mutants exercised; M1b UNAVAILABLE. No complete routing certificate claimed. |
| New miner-start battery | G-HOP, G-R2-ENDS, G-R5-ROUTES PASS; seven socket-dependent gates UNAVAILABLE. Overall sentinel FAIL from environment restrictions. |
| New-battery mutants | Seven meaningful detections: M1, M4–M9. M2/M3/M10/M11 UNAVAILABLE; their PermissionError output is not a detection, despite the harness tally. |
| Direct assignment and synthetic reader/credit checks | Four phases × 32 upper-half rows; span invariance; wrong-interval/shift/gap/non-done refusals PASS. No execution/completion claim for synthetic rows. |
| L-4 injected admission state | DONE holder blocks next admission; pruning restores it. Serve-loop race/fleet incidence not independently reproduced. |
| Diff whitespace / packet stability | `git diff b11b6fd --check` passed; hashes of all tracked-diff files plus the five new review artifacts unchanged at final recheck. |
| Full remaining regressions, real workers, GPU/ROCm, fleet, actual publication and hit rate | UNAVAILABLE in this review. Alpha's saved logs remain attributed evidence. |

Logs and temporary drivers: `/tmp/beta_miner_start_r1/{miner_suite,stage_suite,prelaunch,routing_subset,direct}.log`, `run_checks.py`, `direct_checks.py`; initial packet hashes in `hashes.json`. The socket restriction was environmental, not an automatic approval rejection. No permission escalation was attempted.

The live-path probe case was not run: it calls the staging validator on the manifest path (`tests/test_s172_staging_partb.py:599-606`). No live-state write, real worker/pipeline/fleet launch, rig connection, installation, commit, push or file restoration occurred. The only repository file Beta created is this review.

## Findings outside the numbered questions

No additional new implementation defect. Finding 1 is the prior-record correction above; the launch objection evaluates the already-recorded L-4 status.
