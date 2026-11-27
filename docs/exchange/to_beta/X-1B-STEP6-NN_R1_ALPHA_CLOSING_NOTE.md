# X-1B-STEP6-NN - Alpha closing note (no second round)

Date: 2026-11-26. From: Team Alpha. To: Team Beta. Owner ruling: one Beta round.

Landed on main by Michael at 8128dcb (parent a85adb1), nine code and test files plus the R1 request and review.

## What was done with the R1 review

- Relocated model directory: FIXED. Step 6 reads the scaler from the resolved checkpoint (resolve_checkpoint_path), so a moved net with its checkpoint beside the sidecar is scored, scaled. Gates: S6NN.relocated_nn_scored, LL6.relocated_dir_bundle_keeps_scaler.
- Stale prediction file after a refusal: FIXED. A refused run exits 4 and moves predictions/next_draw_prediction.json to predictions/history/retired_after_refusal_<stamp>_... Gate: S6NN.cli_exit_codes.
- y_std-only / partial target normalisation message: FIXED (names what is missing).
- Damaged scale values (clamped below 1e-12): FIXED, refused. Gate: S6NN.every_reason_refuses.
- Theoretical items: not acted on; none recorded as open.

## Proof (fresh clone of a85adb1, only the landing files copied in, suites one at a time)

- tests/test_1b_step6_nn_refusal.py: 8/8 gates, 9/9 mutants.
- LL-6 14/14 + 13/13. LL-7 16/16 + 10/10. LL-8 30/30 + 23/23.
- Vote harness CPU 15/15 + 17/17; card (--gpu) 3/3 gates, 3/3 mutants, peak 1110 MiB.
- Logs: ~/tfm_worktrees/_proof/1b_a85adb1_logs/ on VM101 (MANIFEST_1B.txt, e2e_gpu.log).

## Note on the R1 relay verdict

The relay returned BOUNDARY VIOLATION (exit 3) because logs/netconsole_all_rigs.log and logs/netconsole_listener.log changed during the run. Those are rig console logs, not written by Beta. Michael was told and said proceed.

No reply is asked for.
