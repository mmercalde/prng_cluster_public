# PARKED: RANGE-MINER per-phase capture (`TFM_MINER_PHASE_CAPTURE_DIR`)

**Parked 2026-11-21.** The owner, 16:02 PST, verbatim: "the evidence is convincing enough for me- lets move on-".

That is about the Stage 4 same-counts result: both engines credited, rows 0/0/13/2995, comparator exit 3. The
per-phase seed-list comparison is no longer required.

All of this work is **uncommitted**. It was made at `b9b990c`. On moving the worktree to `ab2d330` (skip-switch
job, 2026-11-21) it was taken OUT of the working tree, byte for byte, into the ignored
`logs/parked_capture/`: the files, `capture_tracked.patch` against `b9b990c` and `SHA256SUMS`. The skip-switch
diff therefore does not mix with it. **Nothing
was sent to Beta.** There is no X-MINER-CAPTURE request, reply, ledger row or relay log.

## Prior record

- **The scope and choice:** `SCOPE_MINER_PHASE_SETS.md` (this folder), option A. The owner said "Yes" at
  14:40 PST.
- **The gap it addressed:** gap 3 of `REDEPLOY_AND_SESSION_PLAN.md` (`:370-378`, `:412`).
- **The extractor history:** `EXTRACT_MINER_README.md` r1–r3. The r4 section in the worktree describes this work.

## What exists (uncommitted)

| File | State | sha256 |
|---|---|---|
| `window_optimizer_integration_final.py` | modified: the capture helper and constants after `_step1_miner_sweep_evidence`, plus one guarded block at the end of the miner branch of `run_bidirectional_test` | `d4ce5e0270d4764107e6b4fea9ea0f9883143d9331799d10eaf5fd061e7bf9ea` |
| `scripts/extract_miner_bundle.py` | modified: `--phase-capture`, checks C1–C9 (`bind_phase_capture`), the capture sha bound in the receipt, `phases`, `distinct_survivors` and `bidirectional` filled when bound | `050e157703f59ad898265c3a65de7a5b57ead16a05bae1e03506b07390a54c81` |
| `tests/test_extract_miner_bundle.py` | modified: G-CAPTURE-FILLS, G-CAPTURE-RECEIPT, G-CAPTURE-BINDING, MU27–MU37 | `db4715bfc1a50289ecfa18cc3fd2c6f2d2a12a59d525f1d0663cfc1ae9dedcf7` |
| `tests/test_miner_phase_capture.py` | new | `b1c6a513ae3f68a0933e1803987c36f59587f4357fee882aec8387425fd80eda` |
| `docs/drafts/pwc_backup_stage4_v1/EXTRACT_MINER_README.md` | modified: the r4 section | `1dd56acbe71590972811678aeb02dcc92930319a421cc067ab6e4ddfbc70b2fa` |
| `docs/drafts/pwc_backup_stage4_v1/SCOPE_MINER_PHASE_SETS.md` | new: the scoping note | `c0c6963692417fd09041c08933586697ae450cee5517b4618f1cd75e3a137b4a` |
| `tests/test_pwc_backup_stage3.py` | **unchanged**: the guard allowance was not written | `24fcd8a82150e90961e271872b5cb097db8bcc4ecc27a93a23809d19c60fd9c7` (= `b9b990c`) |

**Backups:**
- `logs/pre_6eafad9/` and `logs/pre_b9b990c/`, in the ignored `logs/`.
- The test outputs: `logs/test_miner_phase_capture_run3.txt`, `logs/test_extract_miner_bundle_run9.txt`,
  `logs/affected_suites/`.

## What was proven (CPU only, synthetic trials through the real producer surface and the real finalizer)

- **`tests/test_miner_phase_capture.py`: 5/5 gates, 11/11 mutants.** The baseline is the integration file at HEAD,
  loaded side by side with the changed one.
  - **Variable unset:** records, counts, `TestResult`, NPZ bytes and artifact sha256 are identical to HEAD's code.
    The capture is never called, and no file or output line appears.
  - **Variable set:** the same outputs, plus exactly one correct capture file. The refusals and failures leave the
    outputs identical.
- **`tests/test_extract_miner_bundle.py`: 34/34 gates, 37/37 mutants.** Two extractions with a bound capture
  compare SAME, exit 0.
- **Affected suites, each run against HEAD's integration file and against the changed one:**
  - **Same pass/fail results both ways:**
    - `test_s172_window_anchor_brief_i`, `test_s172_d6_1_flush_durability` and `test_step1_credit_wal` passed
      both ways.
    - `test_s172_phase5_d6_production_adapter`, `test_s172_phase5_d6_threshold_path`,
      `test_s172_miner_seed_start`, `test_s172_threshold_propagation` and `test_step1_search_record_stage1` failed
      the same way both ways. They were already red at HEAD
      (`docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:199-200`, and the Stage 3a/3b before/after tallies).
  - **`test_pwc_backup_stage3`:** 14/14 at HEAD, 12/14 with the change. G-MINER-SUBTREE and G-MINER-DEFS pin the
    miner code to `b2efdef` by design.

## What is not done

1. **The Stage 3 guard allowance.** G-MINER-SUBTREE and G-MINER-DEFS would have to accept exactly the guard and
   its helper, with mutants. The AST pins for the nine capture definitions were computed but not written into the
   suite. As long as the integration change exists, `test_pwc_backup_stage3` stays red.
2. **Beta review (X-MINER-CAPTURE R1).** Not requested and not sent. The change is on the Step 1 launch path, so it
   is a before-commit review class.
3. **A real run.** No real trial was captured. That real rates are exact float32 rests on a code reading
   (`miner/range_miner_worker.py:1336`, `:1403`).
4. **PWC above 100,000 survivors per pass.** Its seed lists are not kept (`persistent_worker_coordinator.py:2509`).

## If it is resumed

1. Start from this worktree's files and their sha256s above.
2. Write the guard allowance.
3. Send X-MINER-CAPTURE R1.

Until then **do not land** `window_optimizer_integration_final.py`. Without the allowance it reds the Stage 3
suite, and it has had no Beta review. The extractor and test changes depend on it. The extractor alone is
harmless: with no `--phase-capture` it behaves as r3.
