# X-VOLUME-LIMIT · R1 · BETA REVIEW

## Prior record

**Searches run in this session, 2026-11-14:**

- `rg -n -i 'volume[-_ ]limit|volume cap|survivor[-_ ]volume|pre-run sample|volume-aware|projected[-_ ]survivors|gpu_process_allowlist|sunshine' docs -g '*.md'`.
- `rg --files docs` filtered for capacity, volume, authorization, freeze, Run-4, proposals, rulings and chapter audits.
- `rg -n -i 'threadpoolexecutor|daemon thread|sampler.*timeout|poisson|garwood|block.*correl|volume.*memory|sample.*rss|sample.*memory' docs -g '*.md'`.
- Searched `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md` and `docs/TODO_CONSOLIDATED.md` for sampling, confidence, caps, timeouts, termination, hangs and skip-range provenance before classifying the observations below.
- Read the request, the five submitted files, the brief, the production seams cited below, the measurement drivers and stored JSON records. Checked both worktrees' HEAD/status and all five submitted SHA-256 values before and after checking.

| Governing or related record found | Label and disposition |
|---|---|
| `AGENTS.md`; `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9` | ALREADY RECORDED. Reviewer authority and hit-rate yardstick apply. This is a load review. |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:76`, `:94`, `:322`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:154` | ALREADY RECORDED. A2 review, separate G4 integration, production sampling and owner-controlled starting cap. |
| `docs/CLAUDE_CODE_INSTRUCTIONS_SURVIVOR_VOLUME_LIMIT_20261114.md:118`, `:144`, `:156` | ALREADY RECORDED. Exact scope, comparison choice and callable estimator. |
| `docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:140`, `:351`, `:430`; `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154` | ALREADY RECORDED. Historical measurements and limits of Run-4 comparability. |
| `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md:53`; `docs/exchange/to_beta/X-0000_R2_ALPHA_REPLY.md:33` | ALREADY RECORDED. Actual upper-half range and Alpha's recorded position on prerequisites. |
| `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:15`, `:44`, `:145` | ALREADY RECORDED underlying ruling; NEW omission from this request's governing-record list. See finding 1. |
| `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md:337`; `docs/CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md:121` | ALREADY RECORDED related timeout precedents, scoped to their own components. They do not impose a new global requirement. The estimator's non-daemon exit behavior is NEW verification; its inability to cancel was already disclosed in the submitted report at `:340`. |
| Submitted report `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:303`; `docs/LEADS.md:146`; `docs/TODO_CONSOLIDATED.md:100` | ALREADY RECORDED statuses: missing production `skip_range`, and volume-blind coordinator ingress. Neither is a new Beta finding here. |

No change to a prior ruling is proposed. The reported owner ruling allowing `sunshine` is taken as supplied in the request, with its corroborating account in the report at `:87`.

**Not searched or executed:** rigs, Proxmox hosts, private repository, ser8 archives, project PDFs, the parallel `step1_search` worktree, live ledgers/staging, or GPU measurements. GPU execution, ROCm behavior, real-driver timeout behavior and independent reproduction of historical ledger counts are **UNAVAILABLE** in this review. Stored measurements were inspected and their arithmetic checked; they were not re-measured.

## Review header

- Thread / round: **X-VOLUME-LIMIT / R1**.
- Date: **2026-11-14**, America/Los_Angeles.
- From / to: Team Beta → Team Alpha and Michael.
- Class: **BEFORE-COMMIT**.
- Base read in both worktrees: `76739247ef46a0f921ed89d684a209800923b2d7`.
- Submitted source root: `/home/michael/tfm_worktrees/volume_limit` (called **W** below). New-file code anchors below are relative to W; production source anchors were read at the same unchanged base.
- All five file hashes matched the request on both checks. No submitted file changed during this review.

Main-tree `git status --porcelain` at initial read:

```text
?? docs/exchange/to_beta/X-VOLUME-LIMIT_R1_ALPHA_REVIEW_REQUEST_survivor_volume_gate.md
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

Both statuses were unchanged at the final source check, 11:23:42 PST, before writing this review.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — Production path | NO OBJECTION | — |
| Q2 — Requested range and projection | NO OBJECTION | — |
| Q3 — Decision rule and uncertainty | SUGGESTION | — |
| Q4 — Failure verdicts | NO OBJECTION | — |
| Q5 — Cap ownership | NO OBJECTION | — |
| Q6 — Sample safety and lifetime | OBJECTION | LIVE — The current CLI's sampling timeout leaves a non-daemon task running and can prevent process exit indefinitely. |
| Q7 — Evidence for the cap | NO OBJECTION | — |
| Q8 — Callable estimator and load semantics | NO OBJECTION | — |
| Q9 — Governing record | SUGGESTION | — |

The Q6 objection is the only LIVE objection. Beta records a review position; Michael decides landing and launch.

## Q1 — Production path

Verified by reading: the estimator calls the production workflow table, residue loader and payload builder (`utils/survivor_volume_estimator.py:347`, `:355`, `:364`). Production uses that same phase table and shared residue function (`miner/range_miner_coordinator.py:12003`; `window_optimizer_integration_final.py:294`; `miner/range_miner_worker.py:867`).

The sampler uses production seed-cap selection, partitioning and `SieveExecutor.execute` (`utils/survivor_volume_estimator.py:304`, `:318`), matching the worker's partition seam (`miner/range_miner_worker.py:2238`). Kernel selection and the host comparisons remain in the real executor (`miner/range_miner_worker.py:1154`, `:1268`, `:1281`, `:1290`). The counting override calls the production launch before observing its counter (`utils/survivor_volume_estimator.py:258`). No replacement PRNG or survivor filter was found in the submitted implementation.

**Status, already disclosed:** the builder emits no `skip_range`, and the worker reads its default `[0,16]` (`miner/range_miner_coordinator.py:9331`; `miner/range_miner_worker.py:1164`). The estimator records that limitation (`utils/survivor_volume_estimator.py:489`). This is faithful sampling of the current production behavior. It establishes no hit-rate effect; a selectable delivered-skip configuration compared with the baseline on unseen-draw pool hit rate would settle that separate question.

## Q2 — Requested range and projection

Verified: each block is placed within a disjoint segment of the requested interval; sample size cannot exceed range size (`utils/survivor_volume_estimator.py:198`, `:206`, `:209`, `:215`). Returned intervals are checked against the range and the requested intervals (`:453`, `:458`). Projection uses the actual range/sample ratio (`:229`).

I regenerated every stored lower- and upper-half sample interval and projection/interval from their recorded inputs, and checked all census per-stripe sums. All matched. The upper-half phase-4 record gives 3,026 × 8 = **24,208**, interval **23,353.075–25,086.221** (`data/survivor_volume_limit/mb_run4_upper.json:255`). Its census contains 32 stripes summing to **23,568**, on `[2³¹,2³²)` (`data/survivor_volume_limit/mb_census_upper.json:156`, `:164`, `:203`). These are confirmations of the submitted records' consistency, not independent GPU counts.

There is a small non-power-of-two full-census API limitation; see finding 2. It does not alter these checked results.

## Q3 — Decision rule and uncertainty

The implementation follows the owner's selected policy: strict `>` comparison, choosing `projection` or `upper95` from the cap file (`scripts/survivor_volume_gate.py:199`; `config_manifests/survivor_volume.cap:2`). It enforces the minimum both before and after sampling (`scripts/survivor_volume_gate.py:284`, `:315`). Zero hits gives projection 0 and a positive Poisson upper endpoint (`utils/survivor_volume_estimator.py:219`); I reproduced **118.0441425** at scale 32.

The record exposes the choice, hits, sample intervals and both interval endpoints (`utils/survivor_volume_estimator.py:468`; `scripts/survivor_volume_gate.py:321`). Noise can affect a point-policy decision near the cap; it is not an undocumented alternate decision rule. The lower-half stored point estimate passes while its upper endpoint refuses, as already reported (`data/survivor_volume_limit/mb_run4_lower.json:255`). For the upper half, both sampled comparisons refuse, and the stored census is independently above the numerical cap. A different partial sample could still give a different point-policy verdict. The census establishes the actual over-cap count in this geometry, not immunity of sampling to noise.

**Suggestion:** qualify “exact 95% interval” as **conditional on the Poisson count model**. The implemented calculation is Garwood for a Poisson mean (`utils/survivor_volume_estimator.py:219`), whereas sampling uses eight deterministic contiguous blocks (`:192`). Exact Poisson algebra does not establish 95% coverage for every such finite PRNG population. Neither block dependence nor general coverage calibration was measured here. Record that assumption, and consider per-block hit counts for later calibration. This is no request to change the owner's comparison rule or cap.

## Q4 — Failure verdicts

For the paths read and exercised, no failure returned PROCEED. Cap/parameter failures refuse; sample/runtime failures map to REFUSE, ERROR or UNAVAILABLE; the CLI maps every non-PROCEED verdict to exit 1 and refuses if record writing fails (`scripts/survivor_volume_gate.py:269`, `:303`, `:339`, `:374`, `:377`).

The suite passed the injected absent/busy GPU, malformed result, exception, timeout, residue mismatch, sample and cap failures, including the mutation that improperly returns PROCEED for an unavailable GPU (`tests/test_survivor_volume_gate.py:212`, `:237`, `:251`, `:265`, `:342`). This is bounded verification, not an assertion about every possible process/OS failure. Q6 separately objects to whether the timeout permits the CLI to finish exiting.

## Q5 — Cap ownership

The cap and run-authorization loaders call the same `tracked_unmodified` function (`scripts/survivor_volume_gate.py:128`; `scripts/run_authorization_gate.py:207`). That function reads bytes, compares them with `HEAD` and returns the same checked bytes for parsing (`scripts/env_freeze_gate.py:114`, `:126`, `:132`). The cap schema is explicit (`scripts/survivor_volume_gate.py:104`). The existing owner-only-commits convention is therefore preserved. Untracked, modified and malformed cap cases and removal of this ownership check were tested successfully.

## Q6 — Timeout does not bound the sampling task or process lifetime

**Verified defect:** `_call_with_timeout` creates a `ThreadPoolExecutor`, times out only the caller's `Future.result`, and calls `shutdown(wait=False)` without stopping the task (`utils/survivor_volume_estimator.py:379`, `:383`, `:391`). Its thread is **not a daemon**. The report's claim that work finishes “in a daemon thread” is incorrect (`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:340`). The existing timeout test checks an in-process return while its fake sampler sleeps for three seconds; it does not check process termination (`tests/test_survivor_volume_gate.py:98`, `:265`).

I ran two CPU-only isolated checks:

```text
SAMPLER_DAEMON False
ERROR_RETURNED_AFTER 0.052
MAIN_FINISHED
CHILD_PROCESS_EXIT_AFTER 2.146
```

That check used a 0.05-second timeout and a two-second sampler. A second check drove the actual `G.main` through the existing temporary cap-repository fixture and three-second fake sampler:

```text
GATE_MAIN_RETURN 1 ERROR ['sampler: SamplerTimeout: sampler did not return within 0.05 s']
MAIN_END
```

The child was still alive at the parent harness's 1.6-second deadline and was terminated by that harness. A separate long-sleep check likewise required parent termination after reporting the timeout. All fixtures were under `/tmp`; no GPU ran.

**Inference and limit:** a production sampler that remains blocked after timing out retains a process-exit dependency and may retain its GPU work/context. I did not induce a real GPU hang. The defect is in the current callable/CLI timeout path and needs no future per-trial integration to occur. A later caller that catches the timeout and continues can also overlap abandoned sampling with another call; that additional integration is future work, not the basis of the LIVE classification.

**Requested correction inside this submission:** make production sampling lifetime controlled across timeout, for example with a separately spawned, terminable process that owns its CUDA context and a parent that accounts for termination before returning. Do not inherit an initialized GPU context by forking it. Add an isolated subprocess test which proves the gate process actually exits on an intentionally blocked sampler; an in-process return assertion is insufficient. Correct the report's thread/lifetime claim. A driver that cannot be stopped must remain explicitly unresolved, not be described as completed cleanup.

For normal completion, I found no path from this sample into live run state: the builder uses an in-memory ledger and the sampler calls the executor directly (`utils/survivor_volume_estimator.py:361`, `:319`; `miner/range_miner_coordinator.py:1207`). It does not call worker transport/publication. Peak memory across all reachable geometries was not measured by Beta; the report's memory figures describe its measured Run-4 case (`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:231`).

## Q7 — What supports 23,515

No objection to retaining **23,515 as the owner's conservative starting policy**. The evidence is one certified completed run with that phase-4 count, not a measured failure threshold or proof that all phases/geometries below it are safe (`docs/S172_PHASE3_SURVIVOR_CAPACITY_CHARACTERIZATION.md:351`, `:361`, `:371`; `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md:44`). The proposal expressly reserves increases to Michael on evidence from completed runs (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:163`).

The local census supplies evidence of **expected load** at 23,568. It does not exercise fleet transport, coordinator ingress, publication or end-to-end completion, and thus does not supply that completed-run evidence (`data/survivor_volume_limit/mb_driver.py:80`, `:94`). Being 53 above the cap establishes the numerical refusal; it establishes no physical capacity discontinuity at 23,516.

Evidence for a different number would be completed, owner-authorized runs at the proposed operating point, with resource observations and the required verifier/evidence review. The already recorded coordinator RSS/ingress follow-ups remain relevant (`docs/LEADS.md:160`). This review neither raises the cap nor proposes a crash-seeking ladder. Any decision to permit Run 4 beyond the current cap belongs to Michael, with the distinction between local load evidence and completed-run evidence explicit.

## Q8 — Reusable estimator; counts remain load

The function takes one exact geometry, range, phase sample sizes and an injectable sampler; its normal return contains measurements/projections and no cap decision (`utils/survivor_volume_estimator.py:422`, `:482`). The gate applies policy separately (`scripts/survivor_volume_gate.py:319`). V-M confirmed equal function/gate figures and no cap-file read through the estimator (`tests/test_survivor_volume_gate.py:308`). The returned measure is explicitly load (`utils/survivor_volume_estimator.py:484`); no count-based quality ranking was found in this implementation.

This boundary can serve a per-trial caller without rewriting estimation. The Q6 lifetime correction must precede treating timed-out calls as finished. ROCm execution is **UNAVAILABLE** here, and the gate's probe is specifically NVIDIA (`scripts/survivor_volume_gate.py:144`); the report already discloses this limitation at `:239`.

## Q9 — Governing record

Add the primary Attempt-9 acceptance ruling to the governing-record list, as detailed in finding 1. The timeout precedents listed in this review are relevant prior experience, not grounds to extend their original component-specific mandates. No ruling was found requiring a different starting cap or a different comparison rule in this submission.

## Additional findings

1. **NEW prior-record omission; underlying status ALREADY RECORDED:** the request invokes a “certified” completed Attempt 9 to justify its cap but omits `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md`. That primary acceptance record states the pass/certification (`:15`, `:44`) and the unobserved Field-6 limitation (`:145`). Add it to the record; do not turn those already recorded limitations into new findings or reopen Gate 12. This does not change the cap position.

2. **NEW, minor API limitation — SUGGESTION:** `sample_intervals(0, 2**26 + 1, 2**26 + 1, 'test')` raises `SampleError`: block 0 has 8,388,609 seeds but its segment has 8,388,608. Remainder seeds are allocated to the first blocks while floor-divided segment remainders can occur later (`utils/survivor_volume_estimator.py:206`, `:208`, `:210`). Thus some valid full-census requests are refused. This fails closed and does not affect the submitted power-of-two sample/range combinations. If full census becomes a supported decision path, distribute counts according to segment capacities or special-case complete coverage.

## Execution record and limits

**Existing suite: 17/17 PASS; `COMPLETION SENTINEL: PASS`; exit 0.** All four mutants were applied and made their specified target case red. Ran with the required venv and GPU hidden:

```bash
source ~/venvs/torch/bin/activate
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python -B - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    'beta_volume_tests',
    '/home/michael/tfm_worktrees/volume_limit/tests/test_survivor_volume_gate.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
suite.WORK_PARENT = '/tmp/beta_survivor_volume_tests'
raise SystemExit(suite.main())
PY
```

Only the suite's temporary-output directory was overridden in memory; no source or assertion was edited. The suite otherwise writes beneath its worktree (`tests/test_survivor_volume_gate.py:38`, `:385`), which Beta's repository-write restriction does not permit. Its temporary git repositories, dataset, records and mutants were created under `/tmp` and cleaned up.

Also ran the isolated CPU timeout checks above, regenerated both stored sample plans/projections, summed both census records, reproduced zero-hit interval arithmetic and checked the non-divisible census edge. Independently hashed the dataset and obtained the stored `513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6` digest.

No pipeline, fleet, worker daemon, rig connection or live-state write was performed. This review is Beta's only repository write. Standalone gate review does not close G4 launch-path integration (`docs/CLAUDE_CODE_INSTRUCTIONS_SURVIVOR_VOLUME_LIMIT_20261114.md:38`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:94`).
