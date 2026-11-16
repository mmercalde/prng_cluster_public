## Prior record

**Searches run, 2026-11-15:**

- Read `AGENTS.md`, this request, its incorporated report's search log, the prior X-RIG-GPU-PROBE R1 review, the probe plan's Prior record and applicable session changelog passages. The incorporated record lists actual searches and labels, so the request meets the entry requirement.
- Ran `rg -n -i` for `cleanup.torch|no.torch|second runtime|per_substripe|torch.*cleanup|cleanup.*torch|parity|counted run|run4_launch`, then narrower searches for `free_all_blocks`, self-check/map availability, retryability, first-cleanup failures and survivor/baseline comparisons. Covered `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` and the exchange record; followed relevant hits into the leads, current reports and plans.
- Read the production diff against `3579088`, the complete new test suite, worker startup/execution/error paths, registry lazy imports, sieve-worker imports/cleanup, launcher construction and committed fleet definitions. Read the parity pin and the local `/home/michael/run4_launch.sh` launch/sentinel/liveness seams. This last file is outside this worktree; it was read only, never executed.

| Record / item | Label and disposition |
|---|---|
| `AGENTS.md`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:19`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:54`, `:83`, `:126`, `:183` | ALREADY RECORDED. Michael decides; committed deployment, run gates, stopping and evidence review retain their roles. |
| X-RIG-GPU-PROBE R1 Q4/Q5; X-RIG-GPU-FAULT R3 runtime-attribution limits | ALREADY RECORDED. One-flag design accepted for implementation review; loaded libraries do not establish independent runtime initialization or causality. |
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154`, `:166`; `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md:160` | ALREADY RECORDED governing comparison limits, omitted from the submitted subject record. Finding 1 below. |
| `docs/PROPOSAL_FREE_ALL_BLOCKS_REPLACEMENT_v1_0.md:80`; `docs/TODO_CONSOLIDATED.md:317` | ALREADY RECORDED cleanup/barrier history, not a new reason to remove the pools in this flag change. |
| Four surviving gate mutants; nonpersistent first-cleanup refusal | NEW implementation/test evidence below. No ruling changed. |

**Not searched / not executed:** no rig, CT, Proxmox, Zeus host, private repository or ser8 archive access; no remote command, real GPU import/initialization, worker, pipeline, fleet launch, package operation, commit or push. No ledger opened or live run state written. The external historical register was not independently re-read in this round. Deployed parity, real runtime binding, GPU output and run acceptance are **UNAVAILABLE** here. Alpha's three older regression-suite results were read, not independently rerun.

# X-MINER-NO-TORCH — R1 — BETA REVIEW

- Date: 2026-11-15, America/Los_Angeles; final source/hash recheck at approximately 17:25 PST.
- Base read: `3579088b8e9c2d902252d626baae15bf0647907f`.
- Scope: the five submitted files. All five SHA-256 values match the request, including on recheck. Parked controller R2 is outside this review.
- `/home/michael/run4_launch.sh` read-only SHA-256: `9cd3aebf259f5aaaef5c7de208e51e871df264d7922d66a0238f1fca1806fc6f`.
- Initial `git status --porcelain` (unchanged on the pre-review-write recheck):

```text
 M docs/LEADS.md
 M docs/exchange/LEDGER.md
 M miner/range_miner_worker.py
 M scripts/launch_fleet_manual.sh
?? docs/CLAUDE_CODE_INSTRUCTIONS_RIG_GPU_FAULT_20261115.md
?? docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md
?? docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md
?? docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md
?? docs/SESSION_CHANGELOG_20261115_RIG_GPU_FAULT.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R1_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R2_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md
?? docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md
?? docs/exchange/to_beta/X-MINER-NO-TORCH_R1_ALPHA_REVIEW_REQUEST_cleanup_torch_flag.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R1_ALPHA_REVIEW_REQUEST_fault_evidence_and_plan.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R2_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-RIG-GPU-FAULT_R3_ALPHA_REPLY.md
?? docs/exchange/to_beta/X-RIG-GPU-PROBE_R1_ALPHA_REVIEW_REQUEST_torch_arms_probe_controller.md
?? scripts/rig_gpu_recorder.py
?? scripts/rig_seam_controller.py
?? scripts/rig_seam_probe.cleanup.sha256
?? scripts/rig_seam_probe.py
?? tests/test_miner_no_torch.py
?? tests/test_rig_seam_probe_controller.py
```

| Question | Position | Severity |
|---|---|---|
| Q1 — default | NO OBJECTION | — |
| Q2 — import exclusion | NO OBJECTION | — |
| Q3 — cleanup and explanation | SUGGESTION | — |
| Q4 — self-check | OBJECTION | THEORETICAL — Continuing after the demonstrated first-cleanup refusal needs a hybrid-first assignment rather than the proposed constant-first Run-4 workflow. |
| Q5 — fleet reach and Zeus | NO OBJECTION | — |
| Q6 — identity evidence | SUGGESTION | — |
| Q7 — gates and mutants | SUGGESTION | — |
| Q8 — counted-run comparison and operation | NEEDS EVIDENCE | — |

No LIVE objection established for the proposed pinned Run-4 path. The theoretical objection holds no commit. This records Beta's review position, not launch authorization or acceptance of a run.

## Finding 1 — omitted governing comparison record

The subject record does not identify `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154`–`:179` or the comparability caveat at `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md:160`. They govern the report's proposed ledger comparison (`docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md:184`). Attempt 9 and post-separation Run 4 are not population-identity controls merely because their displayed geometry matches. This is an **ALREADY RECORDED** limitation, not a new population defect. Name a compatible post-separation baseline in Q8; this review changes no earlier position.

## Q1 — preserved cleanup and command arguments, with disclosed instrumentation

The original cleanup function is unchanged, and the dispatcher calls its module global in the default mode (`miner/range_miner_worker.py:1051`, `:1195`). The default argument remains `per_substripe` (`:2963`). G-DEFAULT passed its source-hash/default/dispatch checks (`tests/test_miner_no_torch.py:129`). Empty default suffixes leave worker argv unchanged (`scripts/launch_fleet_manual.sh:160`, `:165`, `:292`, `:334`).

Do not call the entire behavior byte-identical: startup and first-cleanup now read maps and log (`miner/range_miner_worker.py:1150`, `:1173`, `:1201`, `:2123`, `:2307`). These disclosed additions are acceptable in the default mode; refusal is conditional on `off` (`:1176`). I did not measure their timing. No change to the default survivor computation was found in the diff (`:1328`, `:1397`, `:1428`).

## Q2 — normal import exclusion works

The flag is configured before executor construction and startup GPU operations (`miner/range_miner_worker.py:3022`, `:3033`, `:3068`). The finder is inserted first and raises for `torch` and its submodules (`:1098`, `:1114`). Executed additional checks confirmed that `find_spec('torch')`, `import torch.cuda` and the real registry `_get_torch()` all raise `ImportError`, leaving torch absent. The registry probes availability through `find_spec` (`prng_registry.py:80`, `:84`).

`ImportError` is appropriate for a forbidden ordinary import. A caller may catch it and use a fallback; it does not guarantee process termination. The sieve worker's separate cleanup catches such exceptions (`sieve_gpu_worker.py:83`). That still prevents an ordinary import from adding torch. This is an import policy, not a security boundary against direct native loading or manual module injection; no such production bypass was found in the reviewed paths (`miner/range_miner_worker.py:1277`; `sieve_gpu_worker.py:55`).

## Q3 — reasonable replacement; narrow the runtime claims

GC, a CuPy runtime device barrier and the two pool frees are a reasonable best-effort replacement for this CuPy allocation path (`miner/range_miner_worker.py:1121`, `:1329`). The cleanup gate confirmed the intended barrier/free order (`tests/test_miner_no_torch.py:150`). Synchronization errors are swallowed and pool release is still attempted (`miner/range_miner_worker.py:1134`); describe the barrier as **attempted**, not guaranteed successful. That exception policy is disclosed, and the original torch path also swallowed errors (`:1060`).

The report properly labels initialization and allocator statements as inferences and cross-runtime ordering as unverified (`docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md:104`). Prefer “queries availability and may initialize runtime/driver state” over asserting a second independently initialized instance. Neither maps nor these CPU tests establish which installed runtime implementation handled a call. Likewise, “no explicit worker allocations through torch were found” is supported by the executor (`miner/range_miner_worker.py:1329`); no measurement here proves torch's allocator has literally zero internal allocations. These wording limits do not require adding torch back.

## Q4 — startup refuses; first-cleanup refusal is not persistent

With supplied maps, the implementation rejects imported torch, a selected torch-wheel runtime and independent duplicates of each of `libamdhip64`, `libhsa-runtime64` and `libcudart` (`miner/range_miner_worker.py:1176`). I exercised each condition separately. A mocked detect/warm followed by the real `prepare()` also refused. In the actual nonce path this exception precedes sentinel emission, connect and REGISTER (`:2123`, `:3068`); dispatch or sentinel verification may refuse even earlier than coordinator admission (`/home/michael/run4_launch.sh:267`, `:291`).

The after-first-cleanup check sets `_FIRST_CLEANUP_CHECKED=True` **before** checking (`miner/range_miner_worker.py:1199`). Its exception reaches the generic retryable-error handler, which returns the daemon to idle (`:2455`, `:2551`). With a persistent injected duplicate-runtime identity and the real `handle_stripe`:

```text
first hybrid stripe:  StripeErrorMessage(retryable=True); idle; stopped=False
second hybrid stripe: SubStripeResultMessage + StripeCompleteMessage
```

Thus “that sub-stripe fails” is accurate, but process-wide fail-closed is not. Prefer a latched refusal, non-retryable notification and worker exit so this process cannot perform later assignments. Do not rely on merely moving the checked flag: an invalid runtime needs a durable stop decision. A hybrid retry can otherwise continue the run on another worker (`miner/range_miner_coordinator.py:8494`, `:8535`).

**Severity limit:** this reproduction starts the worker on a hybrid assignment. The proposed workflow starts constant (`miner/range_miner_coordinator.py:12046`), where the coordinator makes even a retryable stripe error terminal (`:8475`). I have not demonstrated that this continuation produces a wrong counted result in the proposed full-history run. The objection is THEORETICAL under the repository's definition.

Two additional limits need accurate wording, but are not established LIVE conditions: an injected `/proc/self/maps` read failure becomes empty inventory and passes (`miner/range_miner_worker.py:1149`); and “any torch-wheel library” overstates the scan, which only inventories four basename families (`:1091`, `:1160`). An injected lone `torch/lib/libtorch_cpu.so` is invisible. Fail on unavailable observation in `off`, and either broaden that wheel check or call it a check for the selected runtime libraries. Current maps unreadability or direct libtorch-only loading was not established here.

## Q5 — all rig commands receive the flag; Zeus exclusion is a scoped choice

The configured three remote nodes have eight GPUs each (`distributed_config.json:15`, `:26`, `:37`; `rig_profiles_config.json:27`). The launcher joins those definitions, iterates every GPU and appends the rig suffix in the remote command (`scripts/launch_fleet_manual.sh:188`, `:272`, `:334`). The local command uses its separate suffix (`:292`). Exporting the variable into the Run-4 harness preserves it through the ordinary child invocation (`/home/michael/run4_launch.sh:262`). This is verified source routing, not a measurement of 24 deployed argv values.

Leaving Zeus unchanged is reasonable for a rig-specific intervention. The VM101 import-only observation is limited evidence, already qualified by the report's unverified later CuPy load (`docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md:131`). It does not establish that CUDA is immune to runtime interactions. Describe the experiment as **24 rig workers off, Zeus default**, rather than every fleet process being torch-free.

## Q6 — useful host-path evidence, not a universal identity proof

G-IDENTITY passed for the two supplied family/phase cases, using real payload construction, executor and postprocessing with deterministic fake device outputs and an independent expected survivor list (`tests/test_miner_no_torch.py:230`, `:276`, `:292`). Host copies and outcome construction precede cleanup (`miner/range_miner_worker.py:1399`, `:1428`, `:1432`), so freeing device buffers afterward cannot mutate those completed Python lists.

Replace “survivor output cannot depend on the cleanup” (`docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md:109`) with that narrower claim. The test does not cover real kernel output, compilation/runtime interactions, effects on subsequent GPU launches, all families, asynchronous errors, timing or delivered output after a self-check exception. It explicitly disables the first-cleanup self-check (`tests/test_miner_no_torch.py:238`). A raising `finally` can suppress the otherwise constructed return (`miner/range_miner_worker.py:1428`, `:1460`). Section 7 correctly reserves GPU behavior for measurement, but “same geometry” needs Q8's baseline qualifications.

## Q7 — listed mutants detected; four additional classes survive

The unchanged submitted suite passed **7/7**, with clean controls and **9/9 listed mutants DETECTED**. I then applied four separate mutations only in temporary copies. **Every mutation passed all six gates**:

| Mutation to production source | Why the current gates miss it |
|---|---|
| Remove `configure_cleanup_torch(args.cleanup_torch)` at `miner/range_miner_worker.py:3022` | Gates configure the module directly (`tests/test_miner_no_torch.py:140`, `:155`, `:237`); the real entry wiring is not exercised. |
| Remove both startup self-check calls at `miner/range_miner_worker.py:2123`, `:2307` | G-SELFCHECK calls the helper directly (`tests/test_miner_no_torch.py:206`). |
| Change duplicate-count threshold from `>1` to `>9` at `miner/range_miner_worker.py:1183`, keeping wheel detection | The negative fixture contains torch-wheel paths, so that independent condition still refuses (`tests/test_miner_no_torch.py:195`, `:211`). |
| Remove executor-finally cleanup dispatch at `miner/range_miner_worker.py:1460` | Cleanup gates call dispatch directly; identity still matches without any cleanup (`tests/test_miner_no_torch.py:143`, `:157`, `:269`). |

Add controls for entry wiring, startup refusal before REGISTER, actual executor cleanup on success/error, isolated duplicate-library predicates and persistent refusal. These survivors identify missing assertions; they do **not** show that those four mutations exist in the submitted production source. The read source has the hooks. This is a test-strengthening suggestion, not a new LIVE production defect.

## Q8 — complete the baseline and operational evidence specification

Committed worker deployment followed by the full parity gate is appropriate: the worker belongs to the governed closure (`scripts/gate12_parity_gate.py:174`). Confirm the local launcher and worker in the tree actually used by the harness as well: `/home/michael/run4_launch.sh:79` changes into `/home/michael/distributed_prng_analysis`, not this review worktree. Hashing the three remote worker files alone is not a replacement for the closure gate.

The packet does not name the baseline run, its retained survivor artifacts or its phase/anchor epoch. Before claiming GPU survivor identity, record a compatible post-separation baseline with the same seed interval, dataset/residue hashes, family, generator phase, delivered thresholds/skip arguments and relevant kernel source. Match logical phase/seed ranges and compare complete survivor records; byte hashes of arbitrary shard files need an ordering/partitioning contract because G-IDENTITY's deterministic fake launch does not establish real GPU output ordering (`tests/test_miner_no_torch.py:251`). Counts alone do not establish identity. If compatible baseline artifacts are absent, label the comparison **UNAVAILABLE** and define a separate matched measurement; Attempt 9 is not a substitute (Finding 1).

Capture the requested runtime/mode observations per identity **before release**, alongside nonce and deployed hashes. The current harness's pre-release gates verify sentinel delivery and liveness (`/home/michael/run4_launch.sh:291`, `:332`); the added mode is currently a separate log record (`miner/range_miner_worker.py:1173`). A postlaunch log check should not be the first discovery that an intended off worker ran the default mode. Retain first-cleanup observations too, and state how a runtime refusal is recorded in the run outcome.

Rollback is a configuration option for a fresh process, not an instruction to relaunch after a failure. Explicitly unset the exported rig variable or set `RIG_CLEANUP_TORCH=per_substripe`; omitting a new assignment does not clear an existing export (`scripts/launch_fleet_manual.sh:160`). Keep a fresh nonce and the existing owner/delegated stop rules (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:83`, `:183`). Existing netconsole proof, environment, authorization, volume and production-shape checks remain recorded run responsibilities, not additional evidence supplied by these CPU tests.

## Checks and evidence

Activated `~/venvs/torch/bin/activate`. Copied the existing test suite and its source closure to `/tmp/X-MINER-NO-TORCH_R1_beta_g2hfpwct/source`, then ran:

```text
python -B -m pytest -q -s -p no:cacheprovider tests/test_miner_no_torch.py
7 passed in 9.31s; 9/9 listed mutants DETECTED
```

The copy was necessary because the suite creates fixtures under its `REPO/logs` (`tests/test_miner_no_torch.py:392`, `:415`), outside Beta's permitted repository write directory. All fixtures and mutations stayed under `/tmp`; child interpreters used the suite's stub torch/CuPy and hidden devices (`:119`). Additional checks exercised the import blocker, isolated self-check predicates, maps-unavailable and libtorch-only limits, real `prepare()` with mocked GPU functions, and real `handle_stripe()` with an injected executor and captured messages. No GPU or socket was used.

Evidence files under `/tmp/X-MINER-NO-TORCH_R1_beta_g2hfpwct/`: `pytest.txt`, `additional_mutants.json`, `counterexamples.txt`, `handler.txt`, `blocker_extra.txt`, `initial_status.txt`. Submitted source hashes remained unchanged. No additional unasked production defect established beyond the omitted governing record in Finding 1.
