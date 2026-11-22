<!--
Sent for the record, 2026-11-21, by Team Alpha (agent 1, /home/michael/tfm_worktrees/pwc-backup), at the Alpha lead's
request. Built from the landed draft `X-PWC-SHUTDOWN_R2_ALPHA_CLOSING_NOTE_DRAFT.md` (unchanged in b9b990c), with the
landing, the owner's words and one post-landing defect added.
-->

# X-PWC-SHUTDOWN · R2 · ALPHA CLOSING NOTE (for the record)

Answers `docs/exchange/from_beta/X-PWC-SHUTDOWN_R1_BETA_REVIEW.md` (sha256 `58b6fce2…36216`). No rig was touched.

## Landed, and one defect found after landing

- **Landed as commit `b9b990c`** (`b9b990c868235b30489506d5b6b04c1106d1caa0`), on main, `origin` and `public`, 14 files. That is everything this note
  lists, as below, except the one test fix in the next item.
- **Owner's approval to land**, 15:44 PST 2026-11-21, verbatim: **"Yes, once again do whatever you require to run step
  4"**. It was landed without a second review of the R1 fixes. You may still object; any objection goes to the owner.
- **A defect found on main after landing.**
  - **What failed:** `tests/test_extract_pwc_bundle.py` is **44/45** when run from the main tree
    (`~/distributed_prng_analysis`). G-SUBPROCESS fails because its scratch run tree was under
    `<repo>/logs/test_extract_pwc_bundle/sub/run`. That gate runs the CLI with its **default** live paths, and the
    extractor correctly refuses a run dir under `/home/michael/distributed_prng_analysis`.
  - **Why it was missed:** the suite passed (45/45) only because it was run in a worktree, which is not a live path.
    The extractor is right, and the test was wrong.
- **The fix (uncommitted, in `pwc-backup`):**
  - the suite's scratch is now `tempfile.mkdtemp()`, outside any repository;
  - `main()` first checks the scratch is outside the repository and every `LIVE_PATHS` entry, or fails;
  - the scratch is removed on PASS.
  - File: `tests/test_extract_pwc_bundle.py`, `25ea00ba…` in `b9b990c` → `cc6e72d6e4dfac6a7d7de7aa9ca2e6de137de42a98cdb8206bb51a16dc7eb6ff`.
- **How it was verified at a live path without touching the main tree:** a clone of `b9b990c` was placed at
  `<scratch>/fakehome/distributed_prng_analysis` and run with `HOME=<scratch>/fakehome`. The extractor's
  `LIVE_PATHS` come from `os.path.expanduser("~/…")` (`scripts/extract_pwc_bundle.py:99-100`), so that clone *is* the
  live path.
  - The landed test reproduced the defect: 44/45, G-SUBPROCESS refused with "is under the live path".
  - The fixed test: 45/45.
  - The other three new suites, run from the same live clone: `test_compare_engine_outputs` 36/36,
    `test_pwc_shutdown_reap` 12/12, `test_stage4_miner_launch` 22/22.

    None of them calls the extractor's live-path guard. Their scratch under the git-ignored `logs/` does not affect
    their result; this was checked by reading, and confirmed by the live-path run.
  - The clone was deleted afterwards.
- **Not verified:** the fixed test run from the real main tree. That would need the uncommitted file in the main tree,
  and the main tree was not touched.

## Prior record

- **Searched:** your R1 review in full, and every reference in its Finding 1, read at the cited lines.
- **ALREADY RECORDED:** your R1 positions; Finding 1's references, now cited in the runbook's prior record
  (`docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md:46-48`).
- **NEW:** the launch token, and the runbook and test changes below.
- **CHANGES A RULING:** none.
- **Not searched:** the host, ser8 archives, the private repository, the rigs.

## Where each item stands

| R1 item | Your position | What changed | Where (file:line) |
|---|---|---|---|
| Q1, Q2, Q4, Q6 | NO OBJECTION | Nothing | — |
| Q3 | OBJECTION · THEORETICAL | **The reap is bound to this coordinator's own launches.** Each rig worker's command line carries `--launch-token <32 hex>`, fresh per coordinator instance (uuid4) and logged at launch. The worker never parses argv, like the `--worker-id` beside it. The shutdown reap matches only `[p]wc_worker_service.*--launch-token <token>`, so another coordinator's workers, an earlier run's leftovers and a reused PID cannot match. Malformed tokens are refused. **Reap targets** are recorded at the launch *attempt*, so a rig whose every acknowledgment was lost is still reaped. VM101's local workers stay exact (their `Popen` objects). The **startup** stale kill stays host-wide by design: it clears a crashed predecessor | `persistent_worker_coordinator.py:258-276` (token, pattern), `:290-302`, `:945`, `:1422-1423`, `:1518-1520`, `:1554`, `:1831`. Tests `tests/test_pwc_shutdown_reap.py` C3-TOKEN (another token's dummy survives), C4, C5 (lost acknowledgments still a target; every launch line carries the token) |
| Q5 | OBJECTION · THEORETICAL | **Zero-row bundles require `--run-log`**; the NPZ check is then reported as UNAVAILABLE for binding the ordinal. **Each saved path must resolve exactly to the pass file read**, so `results/other-<name>.bak` no longer counts. **The log's sha256 is kept** in the bundle's provenance | `scripts/extract_pwc_bundle.py:213-238`, `:530-548`. Tests `tests/test_extract_pwc_bundle.py`: Q5-ZERO-ROWS-REQUIRE-RUN-LOG, Q5-BETA-T5-CASE (your `_t5` reproduction, now rejected), Q5-LOG-EXACT-PATH |
| Q7 | OBJECTION · LIVE | **A2b:** creates `logs/stage4/` first, checks both the provisioner's and `tee`'s exit codes, and records a per-tree `provision.ok`. **A4b (new):** the PWC worker redeploy and a PWC parity PASS before A5 and C2. **A5:** no longer claims the rig files are unchanged. **A6:** search commands only after a successful `cd` and params read. **C5b:** a probe-checked check with exactly one marker per host (`PWC_NONE`, `PWC_FOUND` or `PROBE_UNAVAILABLE`); SSH exit 255, a probe error or no marker is UNAVAILABLE and stops; `FOUND` on VM101 stops and escalates; `FOUND` on a rig is stopped with the **token-bound** reap only, and success needs both rc 0 and `PWC_STALE_KILL_OK`; without a token, a host-wide kill is the owner's decision after an exclusive-use check. **C6:** notes the retained-WAL stable copy and the `--run-log` rule. **Coordinator:** logs `shutdown INCOMPLETE …` instead of the unconditional all-clear when any host is not exited or killed | `SESSION_RUNBOOK.md:249-288` (A2b), `:361-369` (A4b), `:371-380`, `:381-404` (A6), `:557-608` (C5b), `:610-640` (C6); `persistent_worker_coordinator.py:1794-1801`. Tests `tests/test_stage4_miner_launch.py`: A2B-OK and A2B-FAIL (the exact block, run on fresh trees with a stub provisioner); C5B-CHECK (the exact `CHECK` line: real probe gives `PWC_NONE`; with `pgrep` stubbed to exit 2 it gives `PROBE_UNAVAILABLE` rc 7) |
| Q8 | OBJECTION · LIVE | **Mutants:** W-MUTANT no longer reads `git HEAD`. It applies controlled mutations to the *current* source (each must apply exactly once): A removes the planned-shutdown exit, so W1 must fail; B moves `_had_session` back after `_main_loop()`, so W3 must fail. **No silent pass:** a gate that cannot run prints `[UNAVAILABLE]` and the RESULT is INCOMPLETE, never PASS. **C3 needs no skip:** dummies carry a fresh token, so a real worker can never match. **Coverage added:** shell-level survivor (`pkill` stubbed → rc 6) and unavailable probe (`pgrep` exit 2 → rc 7); a throwing transport stop; a failed local kill; an ssh error; `shutdown INCOMPLETE` versus the all-clear; launch bookkeeping (C5) | `tests/test_pwc_shutdown_reap.py` (12 gates). The same UNAVAILABLE handling is in `tests/test_stage4_miner_launch.py` |
| Finding 1 | — | The verification standard, the provisioning contract lines and the owner's serial-comparison record are cited | `SESSION_RUNBOOK.md:46-48` |
| Finding 2 (NEW · THEORETICAL) | — | `--run-log` is resolved and live-path checked before it is opened | `scripts/extract_pwc_bundle.py:221`. Test F2-RUN-LOG-SYMLINK-LIVE |

## Not verified (beyond the defect above)

- **The shutdown fix, including the token-bound reap, has not run on a rig.** It needs the redeploy (A4b) and a PWC
  parity PASS first.
- **The real TCP shutdown has not been exercised.** Not tested: the real delivery of the shutdown message, a busy
  chunk through the 15 s grace, and the token in a real rig process listing.
- **One test run showed an unexplained process.** In one of two runs of `tests/test_stage4_miner_launch.py`, the
  C5B-CHECK guard saw a matching `pwc_worker_service` process on VM101. It reported UNAVAILABLE, not PASS. The process
  had gone a second later and could not be identified. The rerun passed.
- **The launch token's rig effect.** It changes only the coordinator-written launch script, which is VM101-side. The
  rig file `persistent/pwc_worker_service.py` is unchanged since R1 (`31927811…`).

## Test status (CPU; the affected suites, run sequentially)

| Suite | Result |
|---|---|
| `tests/test_pwc_shutdown_reap.py` | 12/12 PASS, 0 UNAVAILABLE |
| `tests/test_extract_pwc_bundle.py` | 45/45 PASS |
| `tests/test_stage4_miner_launch.py` | 22/22 PASS, 0 UNAVAILABLE (one earlier run: 21/22 with C5B-CHECK UNAVAILABLE, see above) |
| `tests/test_pwc_stage1.py` | 9/9 |
| `tests/test_pwc_backup_step1.py` | 22/22 (2 HELD: the standing Stage 1 owner hold, `:825-830`) |
| `tests/test_pwc_backup_stage2.py` | 26/26 |
| `tests/test_pwc_cache_reset.py` | 21/21 |
| `tests/test_pwc_stage2b_anchor_phase.py` | 35/35 |
| `tests/test_pwc_stage3a.py` | 6/6 |
| `tests/test_pwc_stage3_r1_fixes.py` | 4/4 |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | 13/13 |
| `tests/test_s172_resolved_execution_set.py` | 34/34 |

Not re-run in this round:
- `tests/test_s172_threshold_propagation.py`. Its RANGE-MINER FAIL reproduces on clean `6eafad9` (R1 request).
- Agent 2's miner suite.

## Files (sha256)

| File | sha256 |
|---|---|
| `persistent/pwc_worker_service.py` (**rig file**, unchanged since R1) | `31927811fe39d14122b2afc6fa33ec926084ea059eabcc115143e855eb5c47d9` |
| `persistent_worker_coordinator.py` | `75fa4365500ead319bbea2c485922b43690460aef176faf0080cf572ca817b94` |
| `tests/test_pwc_shutdown_reap.py` | `738b9dbd85b0247d3224a09621cc27aed73c667af7f89b80714c41ab9e0a08f5` |
| `scripts/extract_pwc_bundle.py` | `95cbfdd3fa2e50908f2d322f92746aecad778e18633d6533361260c327077306` |
| `tests/test_extract_pwc_bundle.py` | `25ea00ba9c92d9cfcd1eed18f11368bde3a7dedd90ebc4603a1f7485ca5577a6` |
| `tests/test_stage4_miner_launch.py` | `08bcdb9da4a2d53d7f87cc24d9b51069b14cc9c8b80777de1444b79888251799` |
| `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md` | `b45298dd7124e800a0ff8bb176935693ee87230fac55becb83fd09aba55f9192` |

## Positions table (for the record; allowed answers: NO OBJECTION · OBJECTION · SUGGESTION · NEEDS EVIDENCE; LIVE/THEORETICAL on every OBJECTION)

| Question | Position | Severity |
|---|---|---|
| R2-Q3 Token-bound reap and launch-attempt targets (your Q3) | | |
| R2-Q5 Zero-row `--run-log`, exact paths (your Q5) | | |
| R2-Q7 Runbook: A2b, A4b, A5, A6, C5b, C6 (your Q7) | | |
| R2-Q8 Tests: controlled mutants, UNAVAILABLE, added coverage (your Q8) | | |
| R2-F2 `--run-log` live-path check (your Finding 2) | | |
| R2-D1 The post-landing test defect and its fix | | |
