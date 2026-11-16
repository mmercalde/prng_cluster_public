# CLAUDE CODE REPORT — miner worker without the second runtime (`--cleanup-torch off`)

**Direction:** Alpha, 2026-11-15 (after the owner's 16:59 PST question): build the production
no-torch worker as the one-flag design Beta accepted in X-RIG-GPU-PROBE R1 Q5, with the default
preserving today's behaviour. Thread **X-MINER-NO-TORCH**.
**Base:** `3579088` (worktree `rig-gpu-fault`; uncommitted).
**Status:** built and tested **off-GPU only**. Nothing has run on a GPU, nothing has been written
to a host or CT, and nothing is committed. No root cause is claimed: this builds the decisive arm
(N) of `docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md` §4.

---

## Prior record

**Searches run** (`3579088`, 2026-11-15):
- `/bin/grep -rlniE 'remove torch|torch-free|no[- ]torch|without torch|drop torch|torch import'`
  over `docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_*
  docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md
  docs/exchange/`, then the same pattern over `docs/*.md`.
- `/bin/grep -rnE '\btorch\b'` over `miner/*.py sieve_gpu_worker.py prng_registry.py
  persistent/*.py sieve_filter.py reverse_sieve_filter.py adaptive_thresholds.py hybrid_strategy.py`.
- `git grep -nE 'range_miner_worker\.py'` over `*.sh *.py` (every launcher of the worker).
- The prior-work register `/home/michael/tfm_worktrees/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md`
  `:3080-3097` (torch history), already read for X-RIG-GPU-PROBE.

**Documents that record the subject:**

| Document | What it records |
|---|---|
| `docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md` §3.1, §4 | The torch inventory, the one-flag worker design, the fleet P-vs-N comparison. Beta X-RIG-GPU-PROBE R1 Q5: "No objection to the one-flag design, default preserving today's behavior, with separate worker review when built"; Q4: the 7-line block at `:1060-1066` |
| `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md` T4.11 (1) | Two HIP/HSA runtimes mapped in one worker-environment process (library level) |
| `docs/LEADS.md` L-6 | The lead and its torch history |
| `41165dd` (2025-12-24) | A module-level torch import broke CuPy HIPRTC compilation; the import was made lazy |
| `docs/DESIGN_INVARIANT_GPU_ISOLATION.md:55-75` | Precedent: a parent process kept free of torch/GPU imports (Step 5). Not the miner |
| `scripts/gate12_parity_gate.py:158-182` | The deployed worker must match the committed one: **deploy follows landing** |

**Labels:**

| Item | Label |
|---|---|
| `--cleanup-torch {per_substripe, off}`, blocker, CuPy-only cleanup, self-check, launcher variables | NEW (implements the accepted design) |
| Default path byte-unchanged | NEW gate on an ALREADY RECORDED requirement (Beta R1 Q5) |
| Zeus (CUDA) left at `per_substripe` | NEW measurement (§4) |
| The `attempt6_remediation` RXP-3/6, 3/7 failures | Pre-existing in this environment, not caused here (§6) |

**Not searched:** Zeus; the ser8 archives; the private repository; the CT filesystems beyond
the read-only checks recorded in the rig-fault report.

---

## 1. What changed

**`miner/range_miner_worker.py`** (sha256 `2bca2a083262a523801739bc453744c92a1937e38eaa745a95eaaf13baf2ca20`):

| Lines | What |
|---|---|
| `:1051-1072` | `_best_effort_gpu_cleanup`: **unchanged** (source sha `d756acc3…` = `3579088`; gated) |
| `:1076-1204` | new block: `CLEANUP_TORCH_MODES` (`:1088`), `CleanupTorchSelfCheckError` (`:1094`), `_TorchImportBlocker` (`:1098`), `configure_cleanup_torch` (`:1106`), `_cupy_only_gpu_cleanup` (`:1121`), `runtime_identity` (`:1145`), `runtime_selfcheck` (`:1169`), `_gpu_cleanup_dispatch` (`:1191`) |
| `:1460` | `SieveExecutor.execute`'s `finally` calls `_gpu_cleanup_dispatch()` (was `_best_effort_gpu_cleanup()`). In the default mode it calls the module global `_best_effort_gpu_cleanup` exactly as before |
| `:2123`, `:2307` | `runtime_selfcheck("after_warm_gpu")` after `detect_gpu` + `warm_gpu` in `prepare()` and `register()` |
| `:2963` | CLI `--cleanup-torch {per_substripe,off}`, default `per_substripe` |
| `:3022` | `configure_cleanup_torch(args.cleanup_torch)` in `main()`, before the executor is built and before any GPU import |

**Mode `off`, in order:**
1. `configure_cleanup_torch("off")` refuses if torch is already imported, then installs a
   meta-path blocker that makes every `torch`/`torch.*` import raise `ImportError`.
2. After `warm_gpu`, the self-check logs one line,
   `[NO-TORCH] cleanup_torch=off stage=after_warm_gpu torch_imported=False runtimes={...}`, with
   every mapped `libamdhip64` / `libhsa-runtime64` / `libhsakmt` / `libcudart` path. It **fails
   closed** (`CleanupTorchSelfCheckError`) if:
   - torch is imported;
   - any torch-wheel library is mapped; or
   - more than one copy of `libamdhip64`, `libhsa-runtime64` or `libcudart` is mapped.

   The exception propagates out of `prepare()`, so the worker exits before REGISTER.
3. Every sub-stripe cleanup is CuPy-only:
   - `gc.collect()`;
   - `cp.cuda.runtime.deviceSynchronize()`;
   - `free_all_blocks()` on the default and pinned pools.

   Each step is best-effort guarded, as in the production function.
4. After the first cleanup the self-check runs again (`stage=after_first_cleanup`). A violation
   there raises inside `execute`, so that sub-stripe fails and no survivors are credited for it.

**Mode `per_substripe` (default):**
- the same log line at the same two stages, with **no** fail-closed;
- the production cleanup, unchanged.

**`scripts/launch_fleet_manual.sh`** (sha256 `21ff9399ff56fa288193bf6a45babf499b495101d91d900841a7008d3601d790`):
- `RIG_CLEANUP_TORCH` and `ZEUS_CLEANUP_TORCH` (`:160-161`, default `per_substripe`), validated
  (`:162-164`; a bad value exits 64).
- `--cleanup-torch <mode>` is appended **only when non-default** (`:165-166`), on the rig line
  (`:334`) and the Zeus line (`:292`). So today's 25 command lines are byte-identical by default.
- The mode is printed at launch (`:224`).

## 2. What torch's cleanup did for a CuPy-only worker, and what replaces it

The miner worker allocates nothing through torch: every GPU buffer is CuPy
(`miner/range_miner_worker.py:1329-1390`).

| torch call | Effect in this process | Replacement in `off` |
|---|---|---|
| `import torch` | maps the torch wheel's own `libamdhip64.so` + `libhsa-runtime64.so` (HIP 6.3) beside CuPy's `/opt/rocm-6.4.3` copies (measured, T4.11 (1)) | none; blocked |
| `torch.cuda.is_available()` | on the first cleanup, initialises torch's HIP/HSA runtime: a second runtime instance in the process [inference from the API; initialisation not observed] | none |
| `torch.cuda.synchronize()` | a device-wide synchronize **through torch's runtime**. Whether it orders work submitted through CuPy's runtime is **[UNVERIFIED]**. CuPy's results are already host-blocked by `.get()` before cleanup (`:1399-1422`; CuPy 13.5.1 `.get()` is host-blocking by default) | `cp.cuda.runtime.deviceSynchronize()`: the same barrier on **CuPy's** runtime, where the work actually is |
| `torch.cuda.empty_cache()` | releases torch's caching allocator; torch holds no allocations here, so there is nothing to release [inference] | none needed |
| (CuPy) `free_all_blocks()` ×2 | unchanged | unchanged |

**Survivor output cannot depend on the cleanup.** The survivors are copied into host Python
lists (`:1399-1422`) and the `SubStripeOutcome` is built (`:1428`) before the `finally` runs the
cleanup (`:1460`). G-IDENTITY checks this behaviourally (§5).

## 3. Every other torch import reachable from the worker, and how each is avoided

| Site | Reachable from the miner? | In `off` |
|---|---|---|
| `miner/range_miner_worker.py:1060-1066` (inside `_best_effort_gpu_cleanup`) | only through the dispatcher in `per_substripe` | not called |
| `prng_registry.py:84-89` `_get_torch()` (lazy), and the `_torch_available()` callers `:109-111`, `:4141-4145`, `:4162`, `:4174` | `prng_registry` is imported (`:634`), but these functions are called only by `survivor_scorer.py:47-49, :498, :572` and a test, never by the miner (`git grep`) | if ever reached, the blocker makes the `find_spec`/import raise `ImportError`, so it fails loudly; never a silent second runtime |
| `sieve_gpu_worker.py:84-87` (its own cleanup) | the module is imported for `_get_kernel` (`:1150`), but its cleanup is never called by the miner | not called; blocked if it were |
| `reverse_sieve_filter.py:73-76`, `sieve_filter.py:139-145` | not imported by the miner | — |
| `adaptive_thresholds.py`, `hybrid_strategy.py`, `miner/range_miner_protocol.py` | imported | no torch |

## 4. How the flag reaches all 25 workers; Zeus

- **All workers** are started by `scripts/launch_fleet_manual.sh`, the only launcher of
  `range_miner_worker.py` (`git grep`). `run4_launch.sh:263` calls it.
- The launch is `RIG_CLEANUP_TORCH=off ./scripts/launch_fleet_manual.sh …` (or exported in the
  Run-4 launch script). Every one of the 24 rig workers then receives `--cleanup-torch off` in its
  remote command (`:334`), and the Zeus worker receives `$ZEUS_CT_ARGS` (`:292`), empty by
  default.
- **Zeus (CUDA) has not been shown to have the same condition**, so it stays `per_substripe`.
  - Measured read-only on VM101 (venv `torch 2.5.1+cu121`, `cupy 13.5.1`, all devices hidden):
    `import cupy` maps nothing; `import torch` then maps the pip
    `nvidia/cuda_runtime/lib/libcudart.so.12` and the single driver library
    `/usr/lib/x86_64-linux-gnu/libcuda.so.550.163.01`.
  - CUDA runtimes share the one driver library and its context, unlike the rigs' second HIP **and
    HSA** pair.
  - Whether CuPy later loads a different `libcudart` is **[UNVERIFIED]** without a GPU.
    `ZEUS_CLEANUP_TORCH=off` exists if the owner wants Zeus in the N arm.

## 5. Gates and mutants (CPU only; `tests/test_miner_no_torch.py`, sha256 `6a448554…`)

Every gate runs in a fresh interpreter with `CUDA/HIP/ROCR_VISIBLE_DEVICES` empty and with
**stub `cupy` and stub `torch` packages first on the path**:
- The stub CuPy is numpy-backed, and its `free_all_blocks()` zeroes every live stub array, so a
  cleanup before readback would corrupt results.
- The stub torch only records calls.
- **No test can import the real torch or CuPy.**

| Gate | What it shows |
|---|---|
| G-DEFAULT | default `per_substripe`; `_best_effort_gpu_cleanup` source sha = `d756acc3…`; the dispatcher calls it |
| G-OFF-NO-TORCH | `off`: cleanup calls exactly `deviceSynchronize`, then the default free, then the pinned free; torch is never in `sys.modules`; `import torch` raises; the stub torch is never touched. `per_substripe` still calls `import`, `is_available`, `synchronize`, `empty_cache` |
| G-OFF-PREIMPORT | `off` refuses if torch is already imported |
| G-SELFCHECK | `off` passes with one runtime pair and fails closed with the torch-wheel pair; `per_substripe` only logs; the log line names the mode, stage and runtime files |
| G-IDENTITY | the **real** `SieveExecutor.execute` (real builder, real kernel-arg materialisation, real host post-processing, real `ResidueResolver` on a synthetic dataset, a payload from the production payload builder) runs with a deterministic fake launch for `java_lcg_hybrid_reverse` (phase 4) and `java_lcg` (phase 1). **Survivor JSON is byte-identical between `per_substripe` and `off`** and equals an independently computed expectation (non-empty) |
| G-LAUNCHER | default lines carry no flag; `RIG_CLEANUP_TORCH=off` adds `--cleanup-torch off` to rig lines only; a bad value exits 64 |

`python3 -B -m pytest -q -s -p no:cacheprovider tests/test_miner_no_torch.py` → **7 passed** (6
gates + the mutant aggregate). Transcript: `logs/rig_seam_tests/pytest_no_torch_transcript.txt`.

**Mutants, 9/9 DETECTED** (clean control passing for each):
- N1 off imports torch;
- N2 no blocker;
- N3 self-check fails open;
- N4 default `off`;
- N5 dispatch ignores the mode;
- N6 cleanup before readback;
- N7 pre-import not refused;
- N8 launcher drops the rig flag;
- N9 a duplicate runtime is not counted.

## 6. Regression checks (CPU)

- `python3 -B tests/test_s172_phase3_worker.py` → "All gates green".
- `python3 -B tests/test_s172_d6_integration_repair.py` → **82/82** ("the parity gate and the
  launcher wait-set battery are green").
- `python3 -B tests/test_s172_attempt6_remediation.py` → 76/78. RXP-3/6 and RXP-3/7 fail. **The
  same two fail on an unmodified `git archive` of `3579088`** in this environment, so they predate
  this change and are not addressed here. Transcripts are in `logs/rig_seam_tests/`.

## 7. Limits

- G-IDENTITY uses a fake launch. It proves that the cleanup mode cannot change survivors on the
  host path, **not** kernel output on a GPU. The real check is the first fleet entry, comparing
  survivor shards against the ledger's `per_substripe` record for the same geometry.
- Whether torch's synchronize orders CuPy work is unverified; G-IDENTITY makes it irrelevant to
  output, not to timing.
- The self-check sees mapped libraries, not which runtime initialised a device (Beta R1 Q3/Q14).
- Changing the worker file means the parity gate requires deploying the committed file to every CT
  before a run.

## 8. Incident recorded this session (off-GPU tests)

- During the X-RIG-GPU-PROBE suite, mutant M11 of `tests/test_rig_seam_probe_controller.py` let
  the probe's arm-P cleanup reach the **real** torch in the VM101 test venv. It very likely called
  `torch.cuda.is_available/synchronize/empty_cache` on VM101's 3080 Ti, twice, with no kernels.
- Both suites now place a refusing stub `torch` first on the path and hide every device. No clean
  gate was affected.
