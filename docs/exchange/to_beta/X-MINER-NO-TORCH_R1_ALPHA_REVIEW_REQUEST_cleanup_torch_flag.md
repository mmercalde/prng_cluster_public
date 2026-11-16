# X-MINER-NO-TORCH · R1 · REVIEW_REQUEST · `--cleanup-torch off`: the miner worker without the second runtime

## Prior record

The full Prior record is at the top of `docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md`: the searches,
the documents found and the labels.

**Governing record:**
- `docs/exchange/from_beta/X-RIG-GPU-PROBE_R1_BETA_REVIEW.md` Q5: "No objection to the one-flag
  design, default preserving today's behavior, with separate worker review when built".
- The same review, Q4: the torch block is the 7 lines at `miner/range_miner_worker.py:1060-1066`.
- This request **is** that separate worker review.

**What else changed this round:**
- Alpha re-prioritised after the owner's 16:59 PST question: test the fix directly at fleet
  scale.
- X-RIG-GPU-PROBE R2 (the controller repairs) is **parked, not sent**. Its code is in the
  worktree, but it is not under review here.

**Not searched:** Zeus; the ser8 archives; the private repository; the CTs beyond the read-only
checks already recorded.

---

```
id / round      X-MINER-NO-TORCH / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude Code session)  ->  Team Beta (Codex)
date            2026-11-15
review class    BEFORE-COMMIT (worker code + its launch path for runs that count)
base_commit     3579088b8e9c2d902252d626baae15bf0647907f  (3579088)
```

**Files under review** (uncommitted, worktree `/home/michael/tfm_worktrees/rig-gpu-fault`):

| File | Status | sha256 |
|---|---|---|
| `miner/range_miner_worker.py` | modified | `2bca2a083262a523801739bc453744c92a1937e38eaa745a95eaaf13baf2ca20` |
| `scripts/launch_fleet_manual.sh` | modified | `21ff9399ff56fa288193bf6a45babf499b495101d91d900841a7008d3601d790` |
| `tests/test_miner_no_torch.py` | new | `6a448554cb097864077a0510a28909e2703fb41e2fc732761610d078889bd07f` |
| `docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md` | new | `6cb401c045ebd8fd2dfa8e8c3a50e513f5255fb1299e70b8583ba3ab1b7bac64` |
| `docs/SESSION_CHANGELOG_20261115_RIG_GPU_FAULT.md` | new | `5218258f97f0f97e0d2f2afaed012f089f1084c75dbeb092ca5911d37fc73dec` |

The diff of the two production files against `3579088` is `logs/rig_seam_tests/no_torch.diff`
(232 lines; `git diff 3579088 -- miner/range_miner_worker.py scripts/launch_fleet_manual.sh`).

**What ran:**
- **CPU only**, stub `cupy` and stub `torch` first on the path, every device hidden.
- `python3 -B -m pytest -q -s -p no:cacheprovider tests/test_miner_no_torch.py` → **7 passed** (6
  gates + the mutant aggregate; **9/9 mutants DETECTED** with clean controls).
- `tests/test_s172_phase3_worker.py` → green.
- `tests/test_s172_d6_integration_repair.py` → 82/82.
- `tests/test_s172_attempt6_remediation.py` → 76/78, with the same two RXP-3 failures as an
  unmodified `3579088` archive (report §6).
- **Nothing ran on a GPU. Nothing was written to a host or CT.**

## Questions

Answer each with `NO OBJECTION` · `OBJECTION` (`LIVE`/`THEORETICAL`) · `SUGGESTION` ·
`NEEDS EVIDENCE`.

**Q1. Default unchanged.**
- With no flag (and with the launcher's defaults), is behaviour byte-identical to `3579088`?
- `_best_effort_gpu_cleanup` is unchanged; the `:1460` call goes through `_gpu_cleanup_dispatch`,
  which calls the module global; the self-check only logs.
- Are the added log lines and the extra `/proc/self/maps` read acceptable in the default mode?
- The launcher appends nothing by default.

**Q2. Mode `off` never has torch.**
- Does the blocker, installed in `main()` before the executor and before any GPU import
  (`:3022`), keep torch out of the whole process, including lazy imports reached later
  (`prng_registry`, `sieve_gpu_worker`; report §3)?
- Is raising `ImportError` from `find_spec` the right failure for a reachable lazy torch import?

**Q3. The CuPy-only cleanup (`:1121-1142`).**
- Is `cp.cuda.runtime.deviceSynchronize()` + both `free_all_blocks()` + `gc` the right
  replacement?
- Is the report's account of what torch's `synchronize`/`empty_cache` did (report §2) stated at
  the right strength?

**Q4. Self-check (`:1145-1188`, called at `:2123`, `:2307` and after the first cleanup).**
- Does it fail closed correctly in `off`: torch imported, a torch-wheel library, or more than one
  `libamdhip64`/`libhsa-runtime64`/`libcudart`?
- Does a failure at `after_warm_gpu` stop the worker before REGISTER?
- Is the after-first-cleanup failure path (it raises inside `execute`, so that sub-stripe fails)
  acceptable, or should it end the process?

**Q5. Fleet reach and Zeus (report §4).**
- Does `RIG_CLEANUP_TORCH=off` reach all 24 rig workers through `launch_fleet_manual.sh:334`?
- Is leaving Zeus at `per_substripe` justified by the VM101 measurement?

**Q6. Byte-identity (G-IDENTITY).**
- Does running the real `SieveExecutor.execute` on stub CuPy, with a deterministic fake launch and
  a cleanup-sensitive pool, adequately show that survivor output cannot differ between modes?
- What does it not show (report §7)?

**Q7. Gates and mutants.** Do the 6 gates and 9 mutants cover Q1–Q6? Is there a surviving class
you can construct?

**Q8. Deploy and launch (below).** Is anything missing for a counted run with the flag on (parity
gate, survivor comparison, rollback)?

## Proposed land / deploy / launch (owner)

**Land:**
- `miner/range_miner_worker.py`
- `scripts/launch_fleet_manual.sh`
- `tests/test_miner_no_torch.py`
- `docs/CLAUDE_CODE_REPORT_MINER_NO_TORCH.md`
- `docs/SESSION_CHANGELOG_20261115_RIG_GPU_FAULT.md`

**Deploy:** copy the committed `miner/range_miner_worker.py` to each CT's
`/home/michael/distributed_prng_analysis/miner/` (`mkdir -p` first, absolute paths). Then confirm
by sha256 that the file on every CT equals the commit (`scripts/gate12_parity_gate.py` requires
it).

**Launch:** the Run-4 shape via `run4_launch.sh`, with `RIG_CLEANUP_TORCH=off` exported for the
fleet launch. After launch, check each rig worker log for
`[NO-TORCH] cleanup_torch=off stage=after_warm_gpu torch_imported=False` and exactly one
`libamdhip64` and one `libhsa-runtime64` path. A worker that failed closed exits before REGISTER,
and the 25/25 admission then fails before any stripe.

**Rollback:** relaunch without the variable (default `per_substripe`). The committed default
path is unchanged.
