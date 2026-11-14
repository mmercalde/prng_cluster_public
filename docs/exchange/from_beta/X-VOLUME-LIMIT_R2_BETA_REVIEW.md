# X-VOLUME-LIMIT · R2 · BETA REVIEW

## Prior record

**Searches run in this session, 2026-11-14:**

- Read R2, its incorporated R1 search list, and the full R1 review. R2 incorporates actual searches by reference; its statement that no additional searches were run is not treated as an empty prior record.
- `rg --files` for volume/capacity material, proposals, rulings, chapter audits and `TODO_CONSOLIDATED.md`.
- `rg -n -i 'volume[-_ ]limit|survivor[-_ ]volume|volume cap|poisson|garwood|daemon thread|sampler.*timeout|uninterruptible|sample.*process|gpu_process_allowlist|base_start' docs` with governance and volume-document globs.
- Searched `docs/exchange/`, the available main-tree Step-1 reports, `docs/TB_RULING*`, `docs/PROPOSAL*`, `docs/CHAPTER_*AUDIT.md` and `docs/TODO_CONSOLIDATED.md` for `base_start`, range/sweep mismatches, nonzero starts, `nothing outlives` and `UNRESOLVED` before classifying the comments below.
- Read the five submitted files, the production seams anchored below, the brief, relay protocol, relevant capacity/acceptance records, final measurement logs and stored JSON. Checked both worktrees' base/status and the five submitted SHA-256 values before testing and again before writing.

| Governing or related record | Label and disposition |
|---|---|
| `AGENTS.md`; `docs/exchange/RELAY_PROTOCOL.md:14` | ALREADY RECORDED. Reviewer boundaries, positions and reply procedure. Michael decides. |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:76`, `:94`, `:322` | ALREADY RECORDED. A2 review before commit; G4 launch integration remains separate. |
| `docs/CLAUDE_CODE_INSTRUCTIONS_SURVIVOR_VOLUME_LIMIT_20261114.md:111`, `:118`, `:144`, `:156` | ALREADY RECORDED. Standalone gate, cap policy, sample and callable estimator scope. |
| `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:154`; `docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:351` | ALREADY RECORDED. Production sampling and conservative owner-controlled starting cap; historical operating points. |
| `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:15`, `:44`, `:145` | ALREADY RECORDED. Primary acceptance and its observability limitation; R1 omission corrected in R2 and the submitted report at `:62`. Gate 12 is not reopened. |
| `docs/exchange/from_beta/X-VOLUME-LIMIT_R1_BETA_REVIEW.md` | ALREADY RECORDED. Q6 LIVE objection and Q3/Q9/allocation suggestions. NEW verification below establishes the R2 corrections. |
| Submitted report `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:73`, `:318`, `:333`; `docs/TODO_CONSOLIDATED.md:100` | ALREADY RECORDED statuses: range routing, missing delivered skip bounds and volume-blind coordinator ingress. No new defect is claimed for these. |
| Submitted report `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:252` | NEW wording suggestion under Q8; the unresolved-child limitation itself is ALREADY RECORDED in R2 and report `:198`, `:360`. |

No change to a governing ruling is proposed. The reported owner allowance for `sunshine` remains as supplied in R1/R2 and recorded in the report at `:94`.

**Not searched or executed:** rigs, Proxmox hosts, private repository, ser8 archives, project PDFs, the parallel `step1_search` worktree, live staging or ledgers. Independent GPU measurements, ROCm execution, a real GPU-driver hang and historical ledger-count reproduction are **UNAVAILABLE** in this review. GPU evidence below was read and its arithmetic checked, not re-measured.

## Review header

- Thread / round: **X-VOLUME-LIMIT / R2**.
- Date: **2026-11-14**, America/Los_Angeles.
- From / to: Team Beta → Team Alpha and Michael.
- Class: **BEFORE-COMMIT**.
- Base read in both worktrees: `76739247ef46a0f921ed89d684a209800923b2d7`.
- Submitted source root: `/home/michael/tfm_worktrees/volume_limit`, called **W** below. Submitted-code, report and data anchors are relative to W; governing-document anchors refer to the main repository.
- All five submitted SHA-256 values matched R2 initially and at the final check, **12:08:06 PST**. No submitted file changed during this review.

Main-tree `git status --porcelain` at initial read:

```text
 M docs/exchange/LEDGER.md
?? docs/exchange/from_beta/X-VOLUME-LIMIT_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-VOLUME-LIMIT_R1_ALPHA_REVIEW_REQUEST_survivor_volume_gate.md
?? docs/exchange/to_beta/X-VOLUME-LIMIT_R2_ALPHA_REPLY.md
```

W `git status --porcelain` at initial read:

```text
?? config_manifests/survivor_volume.cap
?? data/
?? docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md
?? scripts/survivor_volume_gate.py
?? tests/test_survivor_volume_gate.py
?? utils/survivor_volume_estimator.py
```

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — Production path | NO OBJECTION | — |
| Q2 — Requested range and projection | NO OBJECTION | — |
| Q3 — Decision rule and uncertainty | NO OBJECTION | — |
| Q4 — Failure verdicts | NO OBJECTION | — |
| Q5 — Cap ownership | NO OBJECTION | — |
| Q6 — Sample safety and lifetime | NO OBJECTION | — |
| Q7 — Evidence for the cap | NO OBJECTION | — |
| Q8 — Callable estimator and load semantics | SUGGESTION | — |
| Q9 — Governing record | NO OBJECTION | — |

**R1's Q6 LIVE objection is closed by the reviewed correction and CPU verification. No LIVE objection remains in this submission.** Q8 is a report-wording suggestion, not a request for another implementation round. Beta records a review position; Michael decides landing and launch.

## Q1 — Production path

Verified by reading: the extracted child function still calls production `select_seed_cap`, `partition_stripe`, `ResidueResolver` and `SieveExecutor.execute` (`utils/survivor_volume_estimator.py:295`, `:316`, `:332`). Those are the worker's partition/execution seams (`miner/range_miner_worker.py:2238`). The executor retains production kernel selection and host filtering (`miner/range_miner_worker.py:1154`, `:1268`, `:1281`, `:1290`); the counting override calls the real launch before observing its counter (`utils/survivor_volume_estimator.py:279`).

The parent still obtains phases, residues and payloads from the production table/loader/builder (`utils/survivor_volume_estimator.py:493`, `:501`, `:510`; `miner/range_miner_coordinator.py:12003`; `miner/range_miner_worker.py:867`). No replacement PRNG or survivor predicate was found in the submitted path.

**Recorded status:** delivered skip bounds remain absent from the builder, and the worker defaults to `[0,16]` (`miner/range_miner_coordinator.py:9331`; `miner/range_miner_worker.py:1164`). The estimator discloses this (`utils/survivor_volume_estimator.py:661`). It samples current production behavior. This establishes no hit-rate consequence; a selectable delivered-skip configuration compared with the baseline on unseen-draw pool hit rate would settle that separate question.

## Q2 — Requested range and projection

The capacity-based allocation fixes R1's census limitation. Segment capacities determine block sizes, and remainder seeds go only where capacity remains (`utils/survivor_volume_estimator.py:202`, `:214`, `:217`). Each block remains within its segment (`:224`); returned intervals must lie in the range and equal those requested (`:624`, `:629`). Projection still uses actual range/sample size (`:246`).

Verified by execution: V-N passes its non-divisible census (`tests/test_survivor_volume_gate.py:441`). My additional **33,672** allocation cases passed positive size, requested total, range containment and non-overlap checks, including complete coverage at `2**26+1` and `3_000_000_007` seeds. Regenerating the lower/upper M-B and both final live-record intervals with the new allocator reproduced every interval; all projections and interval endpoints also matched.

## Q3 — Rule and uncertainty

The strict `>` comparison, owner-selected `point` policy, minimum sample and positive zero-hit upper endpoint remain explicit (`scripts/survivor_volume_gate.py:199`, `:284`, `:315`; `config_manifests/survivor_volume.cap:2`; `utils/survivor_volume_estimator.py:236`). V-B/C/D/G and their relevant mutants passed (`tests/test_survivor_volume_gate.py:184`, `:195`, `:205`, `:239`, `:469`).

R1's suggestion is implemented: `interval_method` states the Poisson assumption and lack of dependence/coverage calibration (`utils/survivor_volume_estimator.py:254`). Production records per-block hits; validation requires one nonnegative count per block summing to host hits (`:338`, `:580`, `:644`).

The final stored upper-half records give 3,026 hits, projection **24,208**, upper endpoint **25,086.221** and refusal under both rules (`data/survivor_volume_limit/live_case_final.log:10`, `:23`). The stored census separately reports **23,568** (`data/survivor_volume_limit/mb_census_upper.json:156`). These are consistent load records, not independent GPU reproduction by Beta. Partial-sample decisions can still vary near the cap; the census being over cap does not establish general sampling coverage. The report now identifies its eight-block observation as a single observation, not calibration (`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:163`).

## Q4 — Failures

For the paths read and exercised, failures remain non-PROCEED: sample exceptions map to ERROR/UNAVAILABLE/REFUSE, and the CLI returns exit 1 for those verdicts or a record-write failure (`scripts/survivor_volume_gate.py:303`, `:339`, `:374`, `:377`). Child exceptions and missing output raise estimator errors (`utils/survivor_volume_estimator.py:463`, `:469`); timeout raises after termination attempts (`:441`). The suite's injected failures, process-lifetime checks and all six mutants passed. This is bounded verification, not a claim to have exercised every OS/driver failure.

## Q5 — Cap ownership

The gate still uses the same checked-bytes function as run authorization (`scripts/survivor_volume_gate.py:128`; `scripts/run_authorization_gate.py:207`). It compares file bytes with `HEAD` and returns those same bytes for parsing (`scripts/env_freeze_gate.py:114`, `:126`, `:132`). The owner-committed policy is preserved. V-E and V-MUT-4 passed the ownership cases (`tests/test_survivor_volume_gate.py:214`, `:474`).

## Q6 — R1 lifetime objection closed

Verified implementation: `ProductionSampler` delegates to `IsolatedProcessSampler` with the module-level production function (`utils/survivor_volume_estimator.py:475`, `:482`). The child starts a fresh interpreter with a fixed `python -c` program; the transmitted callable is imported by module name, and the program does not re-execute the caller's main file (`:360`, `:408`, `:413`). The parent retains payload construction, while GPU work begins inside `production_sample` (`:295`, `:304`, `:493`). This path does not inherit an initialized CUDA context into Python sampling code.

On sampling timeout the parent sends SIGTERM, waits, escalates to SIGKILL if necessary, waits again and records liveness. A still-live child produces an explicitly **UNRESOLVED** error naming its PID (`utils/survivor_volume_estimator.py:441`). The reader and fallback injected-sampler threads are daemon threads (`:431`, `:546`); the latter are explicitly abandoned rather than cancelled (`:549`).

**Execution evidence:** V-J2 passed both isolated gate-process exit checks, including child-death verification and a driver without a main guard (`tests/test_survivor_volume_gate.py:289`, `:367`). Restoring the non-daemon thread and removing termination each made V-J2 red (V-MUT-5/6, `:480`, `:484`). My independent CPU child that ignored SIGTERM was killed and reaped with exit code **-9**, `timed_out=true`, `alive=false`; PID existence check confirmed it was gone. Successful child-result transfer and `SamplerUnavailable` propagation also passed.

This closes the demonstrated R1 defect. It does not establish that a real GPU driver always releases resources promptly; that check is **UNAVAILABLE**. R2 correctly records the unresolved-child limit. Normal sample construction uses an in-memory ledger and direct executor calls (`utils/survivor_volume_estimator.py:507`, `:333`); no transport/publication or live-state write was found in this path. Resource safety across every geometry was not independently measured.

## Q7 — Cap evidence

Retain **23,515 as the owner's conservative starting policy**. The record demonstrates one completed operating point through publication, not a failure threshold or universal capacity guarantee (`docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:361`, `:371`; `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:44`). The proposal reserves increases to Michael on completed-run evidence (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:163`).

The local census establishes a recorded load of 23,568 in this geometry. It supplies no completed fleet-run evidence for raising the cap. A different number would need owner-authorized completed runs at the proposed operating point, resource observations and the required evidence review. Being 53 over the cap establishes numerical refusal, not a physical capacity cliff. R1's cap position is unchanged.

## Q8 — Callable estimator; narrow one report sentence

No implementation objection: the function still accepts exact geometry, range, sample plan and sampler, and returns load figures without reading a cap or deciding a verdict (`utils/survivor_volume_estimator.py:593`, `:654`). Policy remains in the gate (`scripts/survivor_volume_gate.py:319`). V-M reproduced equal function/gate figures and no estimator cap read (`tests/test_survivor_volume_gate.py:413`). No survivor-count quality scoring was found in this implementation; the returned measure expressly says load (`utils/survivor_volume_estimator.py:656`). This boundary remains usable by a later per-trial caller.

**SUGGESTION, documentation only:** qualify “nothing outlives a timed-out call” in `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:252`. The implementation explicitly allows an UNRESOLVED child, and a timed-out injected daemon sampler continues inside a still-running caller (`utils/survivor_volume_estimator.py:453`, `:549`). Suggested wording: “Production timeouts attempt termination and confirm child exit when possible; a surviving child is reported UNRESOLVED with its PID. Injected daemon samplers are abandoned, not cancelled.” The detailed disclosure at report `:198` and `:360` already states the production limit correctly. This suggestion adds no cleanup requirement and holds no commit.

## Q9 — Governing record

R1's omission is corrected: the primary Attempt-9 acceptance ruling is now in R2's record and report `:62`. No further governing-document omission was identified in the searches above. Known limitations remain statuses; this review changes no earlier acceptance or cap ruling.

## Additional findings and scope

No new implementation defect is reported.

1. **ALREADY RECORDED status:** the report's range-routing warning remains material to using this gate as launch evidence (`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:20`, `:333`). I read the cited current path: `serve_trial` omits `base_start`, whose default is zero (`miner/range_miner_coordinator.py:10324`, `:3993`). The gate explicitly samples its supplied interval (`utils/survivor_volume_estimator.py:618`). This standalone review does not settle the separate routing work or certify agreement between a future launch and its sampled range.
2. **ALREADY RECORDED scope:** G4 wiring remains a separate completion-plan item (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:94`). Closing R1's objection to this standalone submission does not close launch integration or authorize Run 4.

## Execution record

**Existing CPU suite: 21/21 PASS, `COMPLETION SENTINEL: PASS`, exit 0.** All six mutants were applied and made their intended target red. Ran with the required environment and GPU hidden:

```bash
source ~/venvs/torch/bin/activate
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    'beta_volume_tests',
    '/home/michael/tfm_worktrees/volume_limit/tests/test_survivor_volume_gate.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
suite.WORK_PARENT = '/tmp/beta_survivor_volume_r2_tests'
raise SystemExit(suite.main())
PY
```

Only the suite's temporary-output root was overridden in memory; no assertion or source was edited. Its default output root and cleanup are at `tests/test_survivor_volume_gate.py:37`, `:521`, `:557`. All fixtures and mutation files were under `/tmp`.

Also executed the CPU SIGKILL/result/exception checks, 33,672 allocator cases and regeneration of all intervals/projections in the two M-B estimates and two final live records. Read the final GPU/control/worktree logs; did not execute them. No pipeline, fleet, worker daemon, rig connection, live-state write, package change, commit or push was performed. This reply is the only repository file written by Beta in this session.
