# SESSION CHANGELOG — 2026-11-14 — MINER STAGING-ADMISSION STALL (L-4), RUN 4 ROUTING CONTROL, L-5, EVIDENCE READER

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_MINER_ADMISSION_STALL_20261114.md`
**Report:** `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md`
**Worktree:** `/home/michael/tfm_worktrees/miner-stall`, detached at `7b9184a`. Nothing committed or pushed.

- L-4 reproduced at base without the nudger: 2 workers 6/10 stalled, 1 worker 0/10, 4 workers
  4/4. One worker can also stall (observed 1/3), which corrects LEADS.
- Repair: `_serve_dispatch`'s StripeComplete branch hands `_pump_deferred` to the staging
  executor after the finalize. One definition changed; SR-1 declarations updated with
  provenance.
- New suite `tests/test_s172_admission_stall.py`: 11/11 gates (120/120 forced runs, Run 4 shape
  3/3, one-worker control byte-equal), 4/4 mutants. Nudger removed everywhere; seed-start suite
  10/10 + 11/11 with its R7 control re-captured on `b11b6fd` (digests equal the original).
- Part 2: frozen routing fixture restored to `3e2537b` bytes. Successor harness and
  independent `b11b6fd` control added. Routing 21/22, the frozen gate INCOMPLETE by design,
  all six mutants credited.
- Part 3: L-5 (root level restored around the `window_optimizer` import); evidence reader opens
  `immutable=1` and refuses on a non-empty `-wal`. 3/3 gates, 3/3 mutants.
- Beta X-MINER-STALL R1: NO OBJECTION Q1–Q8 (the X-MINER-START Q7 launch objection is
  withdrawn for these bytes); Q9 SUGGESTION adopted.
- Rigs need coordinator sha256 `b8e259b3e12b95297fa38018a41215d5c52f4d9b3f4095e384829a3ec1f1517d` at power-on.
- fallback parity: not reviewed (not a phase boundary).
