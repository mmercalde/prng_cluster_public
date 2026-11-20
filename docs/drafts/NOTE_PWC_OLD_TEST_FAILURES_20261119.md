# Three old PWC test scripts that fail on main: what they check and why (read-only)

## Prior record

**Searches run** (2026-11-19, worktree `rig-gpu-fault`, main `7444aea`):
- `/bin/grep -rnE 'test_s149_rocr_isolation_harness|test_persistent_worker_harness|test_s172_threshold_propagation' docs/ --include='*.md'`
- `/bin/grep -rnE '\bT1[34]\b|T13/T14' docs/`, `/bin/grep -rnlE 'd6_threshold_path' docs/`
- both of the above over `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/exchange/`
- fact file `~/.claude/skills/tfm-project-facts/SKILL.md` §2.51

**Found (records of these failures):**
- `docs/TEAM_ALPHA_REVIEW_S172_PHASE5_D3_25.md:92-95` and `docs/SESSION_CHANGELOG_20260725_PHASE5_D3_25.md:134-140`: the worker harness T19/T20 are red before and after D3.25, and T19 "needs updating once D3.25 is accepted".
- `docs/TB_RULING_REQUEST_WINDOW_ANCHOR_BRIEF_I_SCOPE.md:204-218`, `:278`: `threshold_propagation 4/5` and `d6_threshold_path 4/17 → 1/17`, both caused by fixtures left stale after F1. Beta accepted this as "same pre-existing root cause" (`docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:26`, `:104-112`).
- Fact file §2.51 item 5: the same F1 fixture staleness, recorded against `d6_production_adapter`.
- `docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:203` (row 39), `docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md:246`, `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md:370`: `threshold_propagation` is the same red before and after each change.
- `docs/TB_RULING_REQUEST_ROCR_ISOLATION_S149.md:27`, `:151-170`: the s149 harness was written to diagnose the bug before the fix.

**Labels:**
- The worker harness T19/T20 and `threshold_propagation`: **ALREADY RECORDED** (above).
- The s149 harness reds, and T13/T14: **NEW** as a written record. No doc says they fail.
- Nothing here changes a ruling.

**Not searched:** ser8, the private repository, the claude.ai project, the rigs (off).

**How this was checked.**
- The owner's main-tree logs: `/tmp/pwc_land/before/*.log`.
- Reruns at earlier commits: a `git clone --shared` scratch copy under `logs/pwc_old_tests/clone`, with logs in `logs/pwc_old_tests/out/`. The clone has no `daily3.json`, so its dataset-dependent checks differ from the main tree.
- Nothing tracked was edited. Nothing was fixed or committed.

**Correction to the brief.** Today's main-tree log for `threshold_propagation` reads **4/5 PASS · 1 FAIL · 0 UNAVAILABLE**. G-KERNEL passes there because `daily3.json` is present. The "3/5, 1 unavailable" figure came from Alpha's scratch clone, which has no dataset.

---

## 1. `test_s149_rocr_isolation_harness.py`: 10/15, five checks fail

**What it is.** A source-reading harness from S149 (March 2026). It was written to *prove a bug was present* before the fix: "verify the ROCR_VISIBLE_DEVICES missing-from-spawn theory" (`test_s149_rocr_isolation_harness.py:5-20`). Its checks assert the old, buggy state.

| Check (line) | What it asserts | Why it now fails |
|---|---|---|
| `:105` HIP set per-spawn | `HIP_VISIBLE_DEVICES` is in the `_spawn_worker` env list | S149-B, then S155, replaced HIP/CUDA masking with `ROCR_VISIBLE_DEVICES` (`persistent_worker_coordinator.py:649-652`) |
| `:110` CUDA set per-spawn | the same, for `CUDA_VISIBLE_DEVICES` | same |
| `:115` ROCR absent | `ROCR_VISIBLE_DEVICES` is *not* set per spawn ("ROOT CAUSE") | the fix sets it (`:650`) |
| `:140` `Device(gpu_id)` unused | no `cp.cuda.Device(gpu_id)` in `sieve_gpu_worker.py` | it matches docstring text at `sieve_gpu_worker.py:144`. The code uses `Device(0)` at `:170`, `:424`, `:427` |
| `:257` fix not yet applied | `ROCR_VISIBLE_DEVICES` appears nowhere in the PWC source | it appears at `:413`, `:650`, `:781`, `:798` |

**Since when.**
- The fix landed before the harness was committed: `c1c698d` (2026-03-20, S149-B) and `5d8d10d` (2026-03-23, S155 ROCR).
- The harness was committed in `ba833a4` (2026-04-03).
- Rerun at `ba833a4`: 10/15, rc 1. It has failed since the day it was committed.

**Recorded?**
- The ruling request describes the harness as pre-fix (`TB_RULING_REQUEST_ROCR_ISOLATION_S149.md:27`). Its verification step says to run it and "verify 15/15 still pass" (`:170`), which can no longer be true.
- No doc records it as failing. **NEW** as a status.

**Where the failure is: in the test.** It is a one-time diagnostic that checks for the bug. The code is in its fixed state. No environment need.

Side observation: the `sieve_gpu_worker.py:144-146` docstring still says "no HIP/CUDA/ROCR visibility masking" and "ROCR_VISIBLE_DEVICES not viable". The PWC spawner sets ROCR (`:650`), so that text is stale. This is not checked further here.

## 2. `test_persistent_worker_harness.py`: 21/25, T13, T14, T19 and T20 fail

A dry-run harness from S134/S135 (March 2026), last edited in `b5b6532` (2026-03-10). It mocks dispatch and runs no GPU work.

**T13 and T14** (`:492`, `:532`).
- They check that a PWC *hybrid* (variable-skip) sieve pass, forward and reverse, dispatches a job with strategies loaded.
- They fail because `run_sieve_pass` now refuses every hybrid type on purpose: `assert_pwc_hybrid_not_quarantined` (`persistent_worker_coordinator.py:216-236`, called at `:1279`). The PWC hybrid path ignored the trial's threshold and skip bounds, and the 07-31 threshold repair quarantined it as uncertified.
- The refusal is itself a gate: `threshold_propagation` G-PWC-HYBRID passes, and its mutant M3 is killed.
- **Since:** `8a55a68` (2026-07-31). The rerun at `8a55a68^` (`a0442a0`) gives 23/25 with T13/T14 passing; at `8a55a68` it gives 21/25.
- **Recorded?** The quarantine is recorded (the `8a55a68` commit text; `S172_THRESHOLD_PROPAGATION_REPAIR_REPORT.md`). That this harness's T13/T14 fail because of it is not written down anywhere found. **NEW** as a status.
- **Where:** in the test. It expects behaviour that was deliberately turned off.

**T19** (`:742`).
- It checks that `_build_test_result_from_pw` turns a PWC result dict into a `TestResult` and fills the accumulator.
- It fails because the dict it builds by hand has no `schema_version`, and the D3.25 ingress wall refuses unversioned results: `validate_trial_populations(..., origin="adapter-ingress")`, `window_optimizer_integration_final.py:1200` → `utils/canonical_records.py:431`.
- **Since:** red before D3.25 too; the rerun at `c207e3a` failed with an empty message. The current message dates from `70cd6f0` (2026-07-25). The first failing commit was not bisected.
- **Recorded:** `SESSION_CHANGELOG_20260725_PHASE5_D3_25.md:136-140` ("encodes the superseded contract and needs updating"). **ALREADY RECORDED.**
- **Where:** in the test, a stale fixture.

**T20** (`:775`).
- It checks that `run_bidirectional_test` sends a trial to PWC (`run_trial_persistent`) when `use_persistent_workers=True`.
- It fails because the coordinator stand-in is a bare `MagicMock()` (`:796`). Its unset `use_range_miner` attribute is a truthy MagicMock, so the call takes the **range-miner branch** (`window_optimizer_integration_final.py:1761-1762`). It then dies at `resolve_assembly_backend(getattr(coordinator,'assembly_backend',None))` (`:1804`) → `miner/assembly_backends.py:366`, before reaching the PWC branch at `:1954`.
- **Since:** the current message dates from `fc37fb5` (2026-07-29, D6 A). Reruns: at `fc37fb5^` it was `FileNotFoundError: daily3.json`, which is only the scratch clone having no dataset; at `fc37fb5` it is the assembly_backend error.
- The miner branch it falls into has existed since `8d0183f` (2026-07-07). The exact first failing commit in a tree with the dataset was not bisected.
- **Recorded:** as red, with an older reason (the `daily3.json` FileNotFoundError, `TEAM_ALPHA_REVIEW_S172_PHASE5_D3_25.md:92-95`). The present reason is not recorded. **ALREADY RECORDED** as a red; the reason is NEW.
- **Where:** in the test. The mock needs `use_range_miner=False`. The routing code is behaving as designed.

## 3. `tests/test_s172_threshold_propagation.py`: 4/5 gates, G-MINER-UNCHANGED fails

**What fails.** Only **G-MINER-UNCHANGED** (`:544`). The other four gates and all three mutants pass in today's main-tree log.

**What it checks.**
- That the miner's kernel and executor files are unchanged and that no threshold logic leaked into registered miner files.
- Then, as a behaviour check, it runs `tests/test_s172_phase5_d6_threshold_path.py` as a subprocess and requires rc 0 (`:642-648`).

**Why it fails.** The sub-suite fails, not the threshold code. Its fixtures pre-date two later contract changes:
- **F1** (`c4e0037`, 2026-08-09). Stripes are now born `pending` and only `schedule_pending_stripes` dispatches them. The fixture's direct dispatch finds "no assignment was dispatched" (`out/thr_at_c4e0037.log:12`).
- **Window-Anchor Brief I** (`48a8705`, 2026-08-22). This made `window_anchor`/`generator_phase` mandatory:
  - `set_trial_context` refuses the fixture's context (`miner/range_miner_coordinator.py:1823`, raised at its mandatory-field check; fixture at `test_s172_phase5_d6_threshold_path.py:225`);
  - `build_stripe_assign_payload` now requires `generator_phase` (`range_miner_coordinator.py:9216`; fixture calls at `:460`, `:750`);
  - the worker rejects the retired `offset` key (`miner/range_miner_worker.py:823`).
  - As a result the sub-suite's own mutant positive control reds on the unmutated module.

**Since:** `c4e0037` (2026-08-09). Reruns:
- `c4e0037^` (`eecfff7`): 4/5 PASS, 0 FAIL, 1 UNAVAILABLE (G-KERNEL needs the dataset);
- `c4e0037`: 3/5, 1 FAIL (G-MINER-UNCHANGED), 1 UNAVAILABLE.

`48a8705` deepened the same failure (4/17 → 1/17 in the sub-suite).

**Recorded:**
- `TB_RULING_REQUEST_WINDOW_ANCHOR_BRIEF_I_SCOPE.md:204-218`, `:278`, accepted by Beta as a pre-existing F1 fixture root cause (`TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:26`, `:104-112`);
- the same mechanism is fact file §2.51 item 5;
- carried as "same red" in the Run 4 launch-prep, seed-start and admission-stall reports.

**ALREADY RECORDED.**

**Where:** in the test. The sub-suite's fixtures were left stale after F1 and Brief I. The gate's own static checks pass. No rig is needed; G-KERNEL needs only the local GPU and `daily3.json`, and it passes on main today.

---

## Summary

| Script | Failing checks | Since | Recorded | Where the failure is |
|---|---|---|---|---|
| s149 ROCR harness | 5 bug-present assertions | its first commit `ba833a4` (2026-04-03); the fix `5d8d10d` came first | not as failing (NEW) | test (one-time diagnostic) |
| worker harness T13/T14 | hybrid dispatch | `8a55a68` (2026-07-31) PWC hybrid quarantine | the quarantine yes; this red no (NEW) | test (expects behaviour deliberately disabled) |
| worker harness T19 | PW result → TestResult | red before `70cd6f0`; current reason from `70cd6f0` (2026-07-25) | yes (D3.25 changelog) | test (pre-v2 fixture) |
| worker harness T20 | PWC routing | current reason from `fc37fb5` (2026-07-29) | red yes; reason NEW | test (MagicMock makes `use_range_miner` truthy) |
| threshold_propagation | G-MINER-UNCHANGED via the d6 sub-suite | `c4e0037` (2026-08-09) F1; deepened by `48a8705` | yes (Brief I scope, §2.51-5) | test (sub-suite fixtures stale) |

None of the ten failing checks (5 + 4 + 1) points to a defect in the production code or to a need for rigs.
