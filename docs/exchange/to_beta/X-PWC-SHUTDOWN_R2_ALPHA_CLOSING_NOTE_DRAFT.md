<!--
DRAFT, 2026-11-21, Team Alpha (agent 1, /home/michael/tfm_worktrees/pwc-backup), at the Alpha lead's request.
NOT SENT. Not relayed. Whether it goes is the owner's decision.
-->

# X-PWC-SHUTDOWN · R2 · ALPHA CLOSING NOTE (draft)

Answers `docs/exchange/from_beta/X-PWC-SHUTDOWN_R1_BETA_REVIEW.md` (sha256 `58b6fce2…36216`). Nothing is committed. No rig
was touched.

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

## Not verified

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
