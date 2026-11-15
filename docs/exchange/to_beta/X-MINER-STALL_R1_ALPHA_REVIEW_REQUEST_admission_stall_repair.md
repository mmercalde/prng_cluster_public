# X-MINER-STALL · R1 · REVIEW_REQUEST · the staging-admission stall (L-4) repair, the Run 4 routing control, L-5 and the evidence-reader open

## Prior record

**Searches run** (worktree `/home/michael/tfm_worktrees/miner-stall`, detached at `7b9184a`,
2026-11-14):
- The brief's Prior record (`docs/CLAUDE_CODE_INSTRUCTIONS_MINER_ADMISSION_STALL_20261114.md`),
  its anchors re-read live.
- Read in full: `docs/CLAUDE_CODE_INSTRUCTIONS_ADMISSION_LIVENESS_REPAIR.md`,
  `docs/CLAUDE_CODE_INSTRUCTIONS_ADMISSION_BINDING_REPAIR.md`,
  `docs/CLAUDE_CODE_CORRECTION6_S172_PHASE4_ADMISSION.md`,
  `docs/CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md`,
  `docs/CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md`,
  `docs/CLAUDE_CODE_INSTRUCTIONS_S172_BP_AMENDMENT.md`,
  `docs/CLAUDE_CODE_REPORT_S172_GATE12_DISPATCH_STALL.md`, `docs/GATE12_MP1_RUN_FORENSIC.md`.
  In part: the BP report's resume trigger, `docs/SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md`.
- `docs/PROJECT_FILE_CATALOG.md` §1.0 SR-1; `docs/TB_RULING_FIELD6_IMPLEMENTATION.md` §2;
  X-MINER-START R1 Q6/Q7; X-RUN4-RANGE R1 Q4; binding addendum §A.
- `grep -rn` over `tests/*.py`: `DECLARED_CHANGED`, `NO_TOUCH_DEFS`, `ADMISSION_NO_TOUCH`,
  `getsource(…_serve_dispatch)` token counts, `_pump_deferred` call-site pins.
- Read-only: attempt 9's run log `logs/gate12_20260817_181819.log`.

**Documents that decide or record this subject:** the brief; `docs/LEADS.md` L-4, L-5;
X-MINER-START R1 Q6, Q7; X-RUN4-RANGE R1 Q4; Correction 6 (approach A, `_admitted` is a pure
serialization gate); Correction 4 §1a (the dispatch thread never blocks; the pump runs off
it); BP §1.2 (`_pump_deferred` is the capacity-release point; pump, then resume); R-1 (R-1c
"a dropped pump is a dropped wake"; G1a pins the admission surfaces byte-identical); MP-1
(pump cost under `_admission_lock` starved the serve loop; added-definition set exact); SR-1.

**Labels:** the repair, its tests, Part 2 and Part 3 — ALREADY RECORDED (the brief). One
worker can also stall at base, contrary to L-4's text — NEW observation, recorded in the
report §2.1 and LEADS. Attempt 9's per-stripe deferral at StripeComplete — NEW reading of an
existing log (report §2.3). Nothing CHANGES A RULING.

**Not searched:** the private repository, ser8, project files outside the repository, the rigs.

---

```
id / round      X-MINER-STALL / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude Code)  ->  Team Beta (Codex)
date            2026-11-14
review class    BEFORE-COMMIT (coordinator code; the launch path of a counted run)
base_commit     7b9184a95cd125c14685008cd306fe0fd9ff3040 (7b9184a), plus the uncommitted
                worktree changes listed below
answers         NO OBJECTION · OBJECTION (LIVE | THEORETICAL) · SUGGESTION · NEEDS EVIDENCE
```

**The files are in a git worktree and are NOT committed.** Read them at the absolute paths
below. The full account, with every anchor, test, count and mutant:

`/home/michael/tfm_worktrees/miner-stall/docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md`

`git -C /home/michael/tfm_worktrees/miner-stall diff 7b9184a` shows the tracked changes;
`git -C /home/michael/tfm_worktrees/miner-stall status --porcelain` lists the new files.

| File (under `/home/michael/tfm_worktrees/miner-stall/`) | Status |
|---|---|
| `miner/range_miner_coordinator.py` | modified, `_serve_dispatch` only (`:11897-11927`); hand-landed; rig parity set |
| `agents/watcher_agent.py` | modified, `step1_presented_configuration` only (`:776-787`); hand-landed |
| `window_optimizer_integration_final.py` | modified, `_step1_miner_sweep_evidence` only (`:1443-1456`) |
| `tests/test_s172_admission_stall.py`, `tests/fixtures/admission_stall_driver.py`, `tests/fixtures/admission_stall_one_worker_control_7b9184a.txt` | new (Part 1) |
| `tests/fixtures/miner_seed_start_driver.py`, `tests/test_s172_miner_seed_start.py` | nudger removed |
| `tests/fixtures/miner_seed_start_zero_control_b11b6fd.txt`, `tests/capture_miner_seed_start_zero_control.py` | control re-captured on `b11b6fd` (digests equal the original); capture script new |
| `tests/test_s172_r1_drain_remedy.py`, `tests/test_s172_mp1_drain_attribution.py` | SR-1 declarations |
| `tests/fixtures/run4_routing_clean_control.py` | restored byte-for-byte to `3e2537b` |
| `tests/fixtures/run4_routing_search_control.py`, `tests/fixtures/run4_search_control_b11b6fd.txt`, `tests/capture_run4_search_control.py` | new (Part 2 successor) |
| `tests/fixtures/run4_pin_harness.py`, `tests/test_s172_run4_routing_patch.py` | Part 2 |
| `tests/test_s172_small_repairs_part3.py`, `tests/fixtures/l5_fresh_process_probe.py` | new (Part 3) |
| `docs/LEADS.md`, `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md` | updated / new |

Base-only checkouts used for controls (tracked files unedited; untracked fixture copies only):
`/home/michael/tfm_worktrees/miner-stall-base` (`7b9184a`),
`/home/michael/tfm_worktrees/miner-stall-b11b6fd` (`b11b6fd`).

## Summary of the change

The StripeComplete branch of `_serve_dispatch` finalizes and now, when the finalize leaves the
stripe complete, hands `_pump_deferred` to the staging executor:
`self._staging_exec().submit(self._pump_deferred)`. It is never called on the dispatch thread,
and it is issued after the finalize returns. Nothing else in the module changes (vs `7b9184a`:
1 definition changed, 0 added, 0 removed; 17 admission/pump/resume/finalize/matrix
surfaces byte-identical).

## Evidence (report §2–§7)

- M at base, no nudger: 2 workers 6/10 stalled, 1 worker 0/10, 4 workers 4/4; the L-4 state
  was observed in every stall. Deterministic unit reproduction on the real `_serve_dispatch`.
- After, no nudger anywhere: staging_first and complete_first forced, 2/4/25 workers × 20 runs,
  **120/120 committed**, forced on every stripe; Run 4's shape committed free/staging_first/
  complete_first; one-worker run byte-equal to the base control; MU1–MU3 red (repair removed,
  pump on the dispatch thread, hand-off condition inverted). Seed-start suite without the
  nudger 10/10 + 11/11.
- Routing 21/22: frozen gate INCOMPLETE by design; all six mutants credited.
- Part 3: 3/3 gates, 3/3 mutants.
- Suites before/after: every difference is explained (report §7); live state unchanged.

## Questions

1. After the repair, can any ordering leave a `done` stripe holding admission?
2. Is the dispatch path still unblocked, and is the back-pressure ordering unchanged?
3. Any double release, or admission of a superseded or terminal attempt?
4. Is a one-worker run unchanged, and are the SR-1 declared sets true?
5. Do the tests force staging-before-StripeComplete with several workers and no nudger?
6. Part 2: is the frozen oracle intact, is the successor control independent, and is every
   mutant credited honestly?
7. Part 3: are the two repairs sound and minimal?
8. **Does your X-MINER-START R1 Q7 objection to launching Run 4 stand after this work? If
   it does, state exactly what remains.**
9. Any governing record this work missed.
