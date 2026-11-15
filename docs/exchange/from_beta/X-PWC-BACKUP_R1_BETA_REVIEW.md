## Prior record

**Searches run, 2026-11-14:** read `AGENTS.md`, the request, the step-1 brief, Stage A and the candidate report. Searched `docs/` with `rg`/`rg --no-ignore` for `PWC_HOST|PWC_PORT`, `pwc_parity`, `stable envelope|STABLE_ENVELOPE|pool 6.{0,30}50`, `pwc_protocol`, `reset_for_new_trial|session-scoped PWC`, and `failed_chunks|completion accounting`, including `TB_RULING_*`, `PROPOSAL_*`, `CHAPTER_*_AUDIT.md`, `TODO_CONSOLIDATED.md`, the catalog, backlog, leads and exchange. Followed up with searches for `build_transport`, omitted/default SSH transport, Gate-22 allowlist/registration/clean-tree guidance, incomplete/late-result accounting, and the new harness's output containment. Searched Python call sites for `build_transport` and read the changed code and its consumers at the anchors below. Read the relevant hits rather than treating their count as evidence.

| Governing record / item | Label | Result |
|---|---|---|
| `AGENTS.md` §§1–5 | ALREADY RECORDED | Beta reviews; Michael decides; writes confined to `docs/exchange/from_beta/`. |
| `docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md:52-82` | ALREADY RECORDED | Allowed scope and S1–S8, including the explicit S7 import repair. |
| Stage A P1/P3/P4/P5; `docs/ROCm_Saturation_Report_S172.md:29-35` | ALREADY RECORDED | Non-certifying status, known gaps, two repeatable envelope points and conflicting larger-chunk history. |
| `docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R2_BETA_REVIEW.md:72-80`; miner-seed-start brief R2/R5 | ALREADY RECORDED | Completion evidence is required for credit; current PWC credit refusal and separate caller-level start restriction are not lifted here. |
| `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:128-165`; `docs/TFM_PROJECT_FACTS_SKILL.md:2531-2545` | ALREADY RECORDED | Session ownership belongs in the integration layer; launch count remains an architectural constraint. |
| `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`; `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145-151` | ALREADY RECORDED; NEW omission from this request's governing list | Clean-tree closure precedents relevant to the report's step-2 item 1. Finding 1 below. |
| `docs/SESSION_CHANGELOG_20260723_PHASE5_D1_1.md:59-63`; catalog `:783-787` | ALREADY RECORDED | Exact new-file registration is permitted under stated conditions; it does not require broadening a gate to clear an authorized dirty-tree red. |
| Factory's omitted-argument SSH default; harness writes outside its stated output directory | NEW review observations | Narrow, non-blocking suggestions below; no matching prior defect record found in the searched scope. |

**Not searched or measured:** rig deployments, rigs/Proxmox/SSH, GPU execution, live ledgers or staging, private repository, ser8, PDFs, or hit rate. No ruling change is proposed. Previously recorded accounting, offset/Brief-II, lifecycle, launch-count, readiness and quarantine limitations are statuses, not new findings.

# X-PWC-BACKUP · R1 · Team Beta review

- **Date:** 2026-11-14, America/Los_Angeles.
- **Class:** BEFORE-COMMIT, review position only.
- **Base read, both trees:** `976a6dba60685925e8c3f82475be9d29a0cff0ba`.
- **Candidate tree:** `/home/michael/tfm_worktrees/pwc-step1`.
- All code anchors below refer to that candidate tree unless explicitly identified as a governing document. The review is written in the main repository.

**Main-tree `git status --porcelain` at first read:**

```text
?? docs/exchange/to_beta/X-PWC-BACKUP_R1_ALPHA_REVIEW_REQUEST_pwc_step1.md
```

**Candidate-tree `git status --porcelain` at first read:**

```text
 M persistent/pwc_protocol.py
 M persistent/pwc_transport_base.py
 M persistent/pwc_worker_service.py
 M persistent_worker_coordinator.py
?? data/
?? docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md
?? scripts/pwc_parity_gate.py
?? tests/test_pwc_backup_step1.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| 1 — Failed/missing chunk returned looking complete? | NO OBJECTION | — |
| 2 — Zero-origin compatibility and non-zero partitioning? | NO OBJECTION | — |
| 3 — Address and silent fallback paths? | SUGGESTION | — |
| 4 — Supported envelope and pre-spawn refusal? | NO OBJECTION | — |
| 5 — Allowed files and RANGE-MINER loading? | NO OBJECTION | — |
| 6 — Hybrid quarantine and no credit? | NO OBJECTION | — |
| 7 — Parity failures refuse? | NO OBJECTION | — |
| 8 — Missed governing record? | SUGGESTION | — |

**No LIVE objection.** This reviews the PWC-only step; it does not establish a working fleet backup or authorize a run.

## Finding 1 — omitted clean-tree closure precedents

The request and carried Stage-A list omit the direct Field-6 and Run-4 R2 clean-tree guidance named above. Field-6 §4 says an authorized dirty-tree Gate-22 red is transitional and **“Do not widen Gate 22's allowlist.”** Run-4 R2 likewise calls for the commit-sensitive check without widening allowlists. These are existing recorded positions, not a newly discovered production defect.

Apply them when correcting the candidate report's step-2 item 1 and §5 (`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md:114`, `:303-314`). The brief already explicitly requests the one-line S7 repair (`docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STEP1_20261114.md:81`). Another permission decision is not needed merely to implement that authorized line. Nor is changing the old suites a necessary prerequisite to landing this step: both relevant predicates examine dirty paths (`tests/test_s172_phase4_coordinator.py:1626-1629`, `:2417-2423`; `tests/test_s172_threshold_propagation.py:544-551`).

Distinguish optional registration of this deliverable's **new** Python paths, under the D1.1 standing rule, from exempting modified existing files or removing an explicit protocol freeze. Record the present reds, then rerun the appropriate checks on Michael's committed clean tree. Do not promise that all checks will pass then: the separately reported D6 baseline failure remains a different issue. I have not committed anything or simulated a clean tree by concealing dirty paths.

## Reasoning by question

**Q1.** In the inspected pass path, failed, timed-out, cancelled and absent results cannot satisfy completion. Aggregation uses a locked result snapshot and classifies every planned chunk; completion requires every chunk completed and the completed intervals covering the request (`persistent_worker_coordinator.py:476-507`, `:1973-2048`). Both trial return paths attach completion metadata (`:2237-2246`, `:2340`, `:2497`); an executed but unaccounted pass is incomplete, and only complete passes receive swept intervals (`:527-555`).

Executed controls covered clean, failed, timed-out, progress-aborted, wrong-range and unaccounted results, plus an incomplete reverse pass at the real trial builder. All passed. The legacy `total_tested` still describes the request (`:2086`), so a future consumer must read completion accounting rather than infer completion from that old field. The current integration consumer has deliberately not been migrated in this step (`window_optimizer_integration_final.py:1896-1903`). That is the recorded step-2 boundary.

**Q2.** Zero-origin **job dictionaries/chunk boundaries** are preserved, not all old startup behavior. The partition now starts at the validated origin, with unchanged chunk sizing (`persistent_worker_coordinator.py:1715-1740`); both constant passes receive that origin (`:2313`, `:2359`). Interval validation runs before coordinator construction and again before chunking (`:272-297`, `:1670`, `:2222`). The executed harness compared serialized jobs against the base implementation for three shapes and passed; non-zero `2**31`, end exactly `2**32`, overflow, negative and boolean-origin checks passed. This establishes CPU partition behavior, not a rig-side sweep. Caller plumbing and the separate non-zero-start restriction remain outside this patch.

**Q3.** The production coordinator validates the transport and, for remote TCP rigs, the dial address before its first outward operation (`persistent_worker_coordinator.py:818-852`). Remote scripts export the supplied address and actual port (`:1140`, `:1166-1174`); the worker entry point requires both environment variables (`persistent/pwc_worker_service.py:757-776`). Executed unset/set tests and intercepted startup support those claims. Missing sizing arguments produce named warnings (`persistent_worker_coordinator.py:620-634`); missing configuration and pass session/skip settings refuse (`:678-687`, `:1675-1679`, `:2227-2234`).

One residual deserves precise wording: **`build_transport()` with no argument still silently returns the SSH sentinel `None`**, because its signature retains `pwc_transport="ssh"` (`persistent/pwc_transport_base.py:38`, `:57-58`). I executed that call. The new detector checks an unknown factory argument and explicit SSH, but not omission (`tests/test_pwc_backup_step1.py:445-454`). The current production call supplies the already-validated transport explicitly (`persistent_worker_coordinator.py:830`, `:919-923`), so I found no reachable silent downgrade in that path. Suggest making the factory itself require a transport, or documenting this narrow residual and adding an omitted-argument case. It is not a LIVE objection.

S6's other stated boundary is accurate: a reported threshold disagreement raises, whereas a missing effective echo is marked unverified and warned (`persistent_worker_coordinator.py:416-447`, `:2058-2069`). Agreement/disagreement controls passed; absent echoes were not counted as verified.

**Q4.** The named table contains exactly the two repeatable points from Stage A P4: 6/50K and 8/25K, preserving the recovered-eviction qualification on the latter (`persistent_worker_coordinator.py:362-383`; `docs/ROCm_Saturation_Report_S172.md:31-35`). The code and report explicitly identify downward dominance as an **inference**, including unmeasured pool-4/pool-7 combinations (`persistent_worker_coordinator.py:374-378`; candidate report `:216-218`). Do not describe every admitted combination as a directly measured stable run.

For the configured remote ROCm rigs, startup checks pool/chunk upper bounds before spawning; dispatch checks the recorded per-rig pool and actual chunk size (`persistent_worker_coordinator.py:835-852`, `:1724-1730`). The CPU cases demonstrated rejection outside the table before any intercepted subprocess, and rejection at dispatch before any fake job. These checks establish enforcement of this declared envelope; they do not establish present hardware stability or adjudicate the conflicting historical runs.

**Q5.** The tracked diff changes only the four listed PWC files. The two new Python deliverables are permitted, and the new report is expressly required by the brief. `data/` is untracked test evidence and excluded from the report's landing list (`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md:483-498`).

The literal answer to “anything RANGE-MINER loads” needs the request's correction: **yes, its launch process imports the changed PWC coordinator**, through the integration module's module-scope import (`window_optimizer_integration_final.py:57-60`). This is not a claim that the miner worker executes PWC. The changed coordinator module adds `shlex`, definitions and constants at import scope (`persistent_worker_coordinator.py:81-100`, `:221-556`); the new startup checks execute when called. The D3.25 regression imported the integration module and passed. No miner source, shared integration source or existing parity gate changed in the reviewed diff. The one-line protocol change imports and round-trips successfully (`persistent/pwc_protocol.py:23-28`; executed S7 control).

**Q6.** The quarantine block is unchanged, and the hybrid dispatch check still precedes chunking (`persistent_worker_coordinator.py:157-218`, `:1661-1670`). The executed hybrid refusal control passed. New completion fields do not confer credit: the integration branch still constructs empty sweep evidence (`window_optimizer_integration_final.py:1397-1403`, `:1899-1903`), and the credit function requires every required phase's interval (`utils/step1_search_ledger.py:1063-1086`). An additional pure-function check with empty PWC evidence and a COMPLETE study trial returned `False, no_sweep_evidence`. No publication or ledger was exercised.

**Q7.** Within the inspected gate, all exercised failure conditions refuse. The pin is checked against the static local import closure, including package initializers (`scripts/pwc_parity_gate.py:182-218`, `:500-508`); local dirty/unavailable status refuses (`:514-524`); probe errors, unknown status, wrong host, missing/different hashes and an empty fleet refuse (`:414-447`). Exceptions in the gate return refusal (`:479-486`).

The supplied injected-runner controls passed for both matching transports, mismatch, unreachable/truncated/wrong-host replies, dirty tree, empty fleet, resolver failure and unknown transport. I additionally injected nine cases: missing file, missing row, malformed digest, malformed row, other nonzero probe exit, missing directory, SSH timeout, failed local git status and runner exception. All nine returned REFUSE. No SSH command was executed. Static closure coverage is not proof of deployed environment or dataset equality; those remain outside this code-parity result.

**Q8.** Finding 1 supplies the missed governing records. The retirement, Brief-II offset work, lifecycle plan and launch-count constraint were already found and disclosed. They are not new objections here. No evidence reviewed here changes the project's hit-rate yardstick or establishes predictive value.

## Finding 2 — the new harness's output-containment claim is inaccurate

**NEW; SUGGESTION, non-blocking.** The harness says every file written is under its `OUT` directory (`tests/test_pwc_backup_step1.py:18`, `:44`), repeated in the report (`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md:299-301`). Its `trial()` helper instead invokes the real trial builder without redirecting its relative result paths (`tests/test_pwc_backup_step1.py:192-208`); that builder writes `results/window_opt_forward_8_0_t1.json` and the reverse counterpart through the normal output writer (`persistent_worker_coordinator.py:2309`, `:2355`, `:2103-2108`). The harness also changes the working directory to the repository (`tests/test_pwc_backup_step1.py:42`).

Verified in my isolated execution: those two files appeared under the temporary working directory's `results/`, outside the redirected `test_out/`. Suggest putting the suite's working directory and outputs in a temporary directory, or intercepting only the output paths, and correcting the report. This is a test-containment/report issue; it did not falsify the 22 gate results. I redirected both `OUT` and the working directory before executing the suite, so no repository result file was written by this review.

## Execution record and limits

- Activated `source ~/venvs/torch/bin/activate`; used `PYTHONDONTWRITEBYTECODE=1` and empty `CUDA_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`.
- Ran the new harness by importing it, assigning `OUT` beneath `/tmp/beta-pwc-r1-3mf0jfeo/`, then changing cwd to that temporary directory before `main()`: **22/22 gates PASS; 16/16 mutants killed**. Source files and detectors were unchanged.
- Sequentially ran `tests/test_s172_phase5_d3_25_candidate_ingress.py` from a separate temporary cwd: **13/13 checks PASS**, including its 17-mutant proof. Its fake producer results correctly emitted unaccounted/incomplete warnings.
- Ran the nine additional injected parity refusals and the two narrow factory/credit checks described above. All produced the stated outcomes.
- Read the dirty-tree predicates and Alpha's reported regression results. **Did not rerun** phase2, resolved-execution-set, threshold-propagation or phase4; their reported totals are not additional Beta passes. Clean committed-tree closure is **UNAVAILABLE** before Michael commits. GPU/kernel checks, real transport/fleet execution and deployed parity are **UNAVAILABLE** in this review.
- Read-only diff/status checks and SHA-256 comparison confirmed the six Python candidate files match the report's landing hashes. No candidate-source change was observed between reading and verification. No commit, push, worker, pipeline, rig connection, package change or live-state write was performed.

Michael remains the decision maker. The recorded position is no LIVE objection to this step, with the scope, evidence limits and suggestions above.
