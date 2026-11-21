# SESSION CHANGELOG — 2026-11-20 — PWC full backup, Stages 1, 2 and 3 (3 uncommitted)

**DRAFT** (not committed). Named per SR-2 (`docs/PROJECT_FILE_CATALOG.md:90-97`): date and topic, no S-number.
Scope: `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md` (`9b6b64c`). Rulings: `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`.
The full records are the two stage READMEs, cited below.
Worktree: `/home/michael/tfm_worktrees/pwc-backup`, at `27eb991`, then `ee39404`, then `0caeaf9`. Each move was a
mixed `git reset` after a byte-identity check. Claude Code session on VM101, as `michael`, CPU only.

## Stage 1 — landed `ee39404` (2026-11-20 06:25, 19 files)

Record: `docs/drafts/pwc_backup_stage1_v1/README.md`. Beta: `X-PWC-STAGE1` R1, `REVIEW RECEIVED`, no objection
(`docs/exchange/from_beta/X-PWC-STAGE1_R1_BETA_REVIEW.md`; `docs/exchange/LEDGER.md:41`).

- **The step-1 candidate S1–S8 merged over main:**
  - per-pass completion accounting;
  - the `seed_start` partition;
  - a required dial address (no `.127` default);
  - no silent fallbacks;
  - the recorded stable envelope (6 at 50,000 or 8 at 25,000);
  - the constant-pass threshold check;
  - `persistent/pwc_protocol.py:26` `worker_id: str = ""`;
  - `scripts/pwc_parity_gate.py` (new).

  Files: `persistent_worker_coordinator.py`, `persistent/pwc_protocol.py`, `persistent/pwc_transport_base.py`,
  `tests/test_pwc_backup_step1.py` (new).
- **B5:** the rig stale-worker kill killed its own shell, so the `sieve_gpu_worker` kill never ran. This was shown by
  running the old string in a private PID namespace, `tests/test_pwc_stage1.py` G-B5-OLD-SELF-KILL. It now uses
  bracket patterns and reads its result; an unconfirmed kill refuses that rig. The S156 find and reap are fixed the
  same way.
- **A3 level 1:** the TCP transport serves only worker ids this run launched (`persistent/pwc_transport_tcp.py`).
- **B6:** the TCP legacy parser reads `skip_pattern`, the name `sieve_filter.py` emits.
- **A4:** `test_persistent_worker_harness.py` (25/25), `test_s149_rocr_isolation_harness.py` (16/16) and
  `tests/test_pwc_cache_reset.py` are repaired.
- **Held, not applied:** the candidate's `persistent/pwc_worker_service.py` hunk, which removes the worker's own
  `.127` fallback; the worker file was out of the order. Saved at
  `docs/drafts/pwc_backup_stage1_v1/HELD_pwc_worker_service.diff`.
- **Owner rulings recorded with the commit** (2026-11-19 20:57): lift the hybrid block on the threshold fix; PWC earns
  credit; drop the seed-start refusal; phases follow RANGE-MINER. Default rule: PWC does what RANGE-MINER does.

## Stage 2 — landed `0caeaf9` (2026-11-20 14:35, 25 files)

Record: `docs/drafts/pwc_backup_stage2_v1/README.md`. Beta: `X-PWC-STAGE2` R1 and R2, both `REVIEW RECEIVED`
(`docs/exchange/LEDGER.md:42-43`), plus a closing note not sent (`docs/exchange/to_beta/X-PWC-STAGE2_R3_ALPHA_CLOSING_NOTE.md`).

- **Ruling 1, variable-skip block lifted:**
  - no `phase2_threshold` default;
  - each job carries the directional threshold as both keys;
  - the worker's report is checked (4 places, `sieve_filter.py:597-599`);
  - the SSH route is refused for variable-skip.

  File: `persistent_worker_coordinator.py`.
- **Ruling 4:** the forward-zero prune and the Q0 gate are removed.
- **Default rule:** all five hybrid strategies run (S147 Q2 removed).
- **Owner ruling 08:30:** PWC refuses a window offset other than 0 until the anchor/phase split lands (Stage 2b).
- **Beta Stage 1 items:**
  - process checks prove they can see processes (positive control; exact `pgrep` status; VIR-1/VIR-5);
  - the VM101 local kill result is read;
  - revocation drops the id's live connection;
  - a no-PID launch is not counted;
  - a refused worker exits with code 43. That is the only worker edit, in `persistent/pwc_worker_service.py`.
- **Beta X-PWC-STAGE2 R1, fixed in R2:**
  - Q6 LIVE, the reconnect race: per-id cleanup by the owning connection, under the registry lock;
  - finding 2 LIVE: job ids are unique per pass, and only the active lease's result is accepted;
  - Q7: no GPU heartbeat on the refusal exit.
- **Beta R2-Q1 LIVE**, timeout eviction at the boundary: fixed exactly as Beta specified. Registration and last-seen
  are set in one locked step, and online/ready/last-seen are tied to the registered connection
  (`persistent/pwc_transport_tcp.py`). There was no third round, by owner decision at 12:34.
- **Beta R1-Q3, five strategies on the rigs: NEEDS EVIDENCE, open.** It is answered by the owner's Stage 4 rig
  sessions.
- **Tests:** new `tests/test_pwc_backup_stage2.py` (26 gates, 29 mutants). Updated: `tests/test_pwc_stage1.py`,
  `tests/test_pwc_backup_step1.py` (the base pinned to `27eb991`), `test_persistent_worker_harness.py`.
- **Shared suites, PWC lines only** (owner ruling 08:28): `tests/test_s172_threshold_propagation.py` (G-PWC-HYBRID,
  M3) and `tests/test_s172_phase5_d3_25_candidate_ingress.py` (`drive_pwc`, one PWC branch). An AST comparison shows
  every RANGE-MINER statement identical to `ee39404`.

## Stage 3b — shared launch side, uncommitted (worktree `/home/michael/tfm_worktrees/pwc-backup`, base `b2efdef`)

Package: `docs/drafts/pwc_backup_stage3b_v1/README.md`. Owner rulings applied: 2 and 3 (2026-11-19), 15:53 (RANGE-MINER
is the default), 20:24 (n_parallel > 1 is a separate later job, which also takes S3-6 and S3-7). Beta 2B Q5 is fixed:
an incomplete PWC trial raises before the accumulator and before Optuna scores it, as RANGE-MINER does.

- **Files:**
  - `window_optimizer.py`, `window_optimizer_integration_final.py`, `agent_manifests/window_optimizer.json`;
  - `agents/watcher_agent.py`: one keyword (owner yes);
  - PWC lines only (owner yes) in `tests/test_s172_miner_seed_start.py`, `tests/test_s172_phase1_scaffolding.py`,
    `tests/test_step1_search_record_stage1.py`;
  - **`test_persistent_worker_harness.py`, T20 only** (owner yes, 2026-11-20). Its mock PWC result now carries a
    complete `pwc_trial_completion_v1` record. An AST comparison shows only `T20_integration_gate_uses_pwc` changed
    (`docs/drafts/pwc_backup_stage3b_v1/t20_only_proof.txt`). The suite reran: 25/25;
  - new: `tests/test_pwc_backup_stage3.py`, `scripts/compare_engine_outputs.py`, `tests/test_compare_engine_outputs.py`.
- **Full before/after pass** (77 suites, sequential, CPU only, ssh/scp/rsync stubbed, `tests/test_s172_staging_partb.py`
  not run; `docs/drafts/pwc_backup_stage3b_v1/before_after_tally.md`):
  - the work was parked in a hashed tar and the tree reset to a clean `b2efdef`; after the restore, 42/42 hashes,
    status and diff were identical;
  - 73 of 75 common suites have identical failing-check sets. Four differ:
    - T20, real and intended (Q5); fixed since by the owner-approved edit above;
    - W-NO-WEAKENING and Gate 22, which compare the working tree with HEAD and clear on commit (by reading; not
      executed);
    - G-R5-ROUTES, the same failure under its new label.
  - New suites: Stage 3b 14/14 + 18/18; compare tool 13/13.
- **S3-13 withdrawn** (owner, 21:32-21:35). F4 now stands as: PWC keeps the trial's skip range; the RANGE-MINER fix
  is scoped in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`. PLAN, README, rulings file and
  handoff are updated.
- **Combined tree (3a + 3b):** agent 2's 20 Stage 3a files were copied unchanged (no path overlap; sha256 20/20 equal
  to pwc-offset). Focused results on the combined tree:
  - green: `test_pwc_stage3a` 6/6 + 3/3; `test_prove_unchanged_outside` 42/42; `test_pwc_stage2b_anchor_phase` 35/35;
    `test_pwc_backup_stage2` 26/26 + 33/33; `test_pwc_backup_step1` 22/22 + 16/16 (2 HELD as before);
    `test_persistent_worker_harness` 25/25; `test_compare_engine_outputs` 13/13;
  - **red: `test_pwc_backup_stage3` 13/14 gates, 17/18 mutants.** 3a's new header check flags the unchanged
    pre-existing default `staging_high_water_bytes = 16 * 1024 ** 3` (`window_optimizer.py:759`, the same as
    `b2efdef:756`), because 3b added a parameter to that header. Resolved 2026-11-21: agent 2 narrowed the tool (00:32; a default is flagged only if its AST differs from the base, or the parameter is new). After a sha256-checked re-copy: `test_pwc_backup_stage3` 14/14 + 18/18, `test_prove_unchanged_outside` 49/49, `test_pwc_stage3a` 6/6 + 3/3, subtree proof unchanged with both control mutants refused.;
  - subtree proof (`prove_unchanged_outside.py --keep-block 'run_bidirectional_test::if _use_miner:'`): RANGE-MINER
    branch plus preamble identical to `b2efdef`. Two valid control mutants are refused; an invalid first mutant (line
    2034, outside the kept block) is recorded and not counted
    (`docs/drafts/pwc_backup_stage3b_v1/prove_unchanged_outside_subtree_woi.txt`).
- **EXEC-PIN-1** (Beta 2B Finding 1): a mechanical scan found 18 test files with pinned executable source and
  46 candidate arms. The only signature change is one trailing optional keyword. Every arm file has unchanged failing
  checks in the full pass. No bridge was needed (`docs/drafts/pwc_backup_stage3b_v1/exec_pin_1_scan.txt`).
- **W-NO-WEAKENING after commit:** read, plus a static run of the parity closure: no `gate12_launch.sh` gate is tripped after commit. The clean-tree producer `_repository_state` is AST-identical, and no Stage 3 file is parity-governed (`docs/drafts/pwc_backup_stage3b_v1/gate12_launch_gates_reading.txt`). Put to Beta as C-Q2.
- **Joint Beta request:** `docs/exchange/to_beta/X-PWC-STAGE3_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md`.
  Renamed from `..._DRAFT.md` and relayed for one round on the owner's approval of 2026-11-21 05:56 PST ("yes please
  send to beta"). Beforehand, after the ~02:35 VM101 reboot, the session re-checked the tree: git status unchanged, the
  3b diff byte-identical to the saved one, the 3a files and drafts sha256-equal to pwc-offset (20/20), and
  `test_pwc_backup_stage3` 14/14 + 18/18.
- **Beta X-PWC-STAGE3 R1** (`docs/exchange/from_beta/X-PWC-STAGE3_R1_BETA_REVIEW.md`; relay 06:02-06:11, REVIEW
  RECEIVED, exit 0):
  - **Positions:** 2 LIVE objections (3b-Q3, 3b-Q7), 5 SUGGESTIONs, 10 NO OBJECTIONs, no THEORETICAL.
  - **Relay snapshot:** clean. Outside the worktree, the main tree, `optuna_studies/`, `~/miner_staging` and pwc-offset
    are unchanged. Beta's scratch files went to `/tmp/tfm-beta-stage3.JrbE4S/`, not the TMPDIR set for the run; none
    was deleted.
- **Fixed as Beta specified,** on the lead's instruction:
  - **3b-Q7 / Finding 1:** the PWC call no longer passes `min_workers`. That is b2efdef's shape, with effective minimum
    1, under owner ruling 2026-08-04 (`docs/CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md:86-100`,
    `docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195`). The plan's Q-9 had passed 24 on a wrong "what RANGE-MINER does"
    reading, with no ruling behind it.
  - **3b-Q3:** `persistent_worker_coordinator.py` `_startup_refusals` refuses a seed cap ≤ 0 before any worker is
    touched, as RANGE-MINER does (`miner/range_miner_coordinator.py:769-770`, `:789-790`;
    `miner/range_miner_worker.py:681-682`). The lead authorised agent 1 to edit the coordinator for this one fix.
  - **Tests:** new `tests/test_pwc_stage3_r1_fixes.py`, 4 gates and 5 mutants; G-PWC-CALL updated.
  - **Proofs:** the subtree proof is unchanged (rc 0), and the coordinator proof shows only `_startup_refusals`
    changed (rc 0; control rc 1).
- **Suggestions applied in documents:**
  - C-Q1: a correction to `exec_pin_1_scan.txt`;
  - Finding 1: completion-plan rows 25-26 added to `STALE_DOCS.md`; the PLAN's Q-9 row and its NEW label corrected;
    the handoff's open question marked settled.
  - The rest are carried; see the closing note.
- **Full after-pass** (`docs/drafts/pwc_backup_stage3b_v1/r1_fixes/before_after_tally_r1.md`), 79 suites, against the
  2026-11-20 before-pass: 72 of 75 common suites identical. The 3 that differ are W-NO-WEAKENING, Gate 22 and the
  G-R5 label. The 4 new suites are green, and T20 is green.
- **Owner, 2026-11-21 06:28 PST:** "yes, since there are only two beta rejections and you have already addessed them
  land stage 3". No second Beta round. The closing note
  `docs/exchange/to_beta/X-PWC-STAGE3_R2_ALPHA_CLOSING_NOTE_DRAFT.md` is a draft and was not relayed. Nothing is
  committed.

## Suites (sequential, CPU only, ssh stubbed; `tests/test_s172_staging_partb.py` not run)

**Stage 2:** a fresh before pass on `ee39404`, then three after passes. Each time, 68 of 72 existing suites were
identical. The same 4 differ each time:
- the two Stage 1 PWC suites go red → green;
- phase-4 Gate 22, the uncommitted-`.py` tripwire, naming only PWC files;
- the RXP-1/1 timing flake.

The lead reported the coordinator suite at 63/63 on the committed tree. `~/miner_staging`'s mtime was unchanged after
every suite.

**Stage 1:** 71 suites, before and after; 4 differ, all explained in its README.

## Untouched (both stages)

- `miner/` and the ten rig parity-pinned files (`scripts/gate12_parity_gate.py:174-185`): sha-identical.
- Also untouched: `window_optimizer*.py`, `sieve_filter.py`, `sieve_gpu_worker.py`, `prng_registry.py`, the
  launchers, the manifests, WATCHER.
- SR-1 (`DECLARED_CHANGED`) did not apply: no `miner/range_miner_coordinator.py` change.

## Not done (by rule or by scope)

- No commit or push by the agent; no rig, GPU or live ledger; no redeploy.
- Not built, and not in either stage:
  - PWC credit, the seed-start refusal at the launch layer, and the dial-address and sizing plumbing (Stage 3);
  - the window/generator split (Stage 2b, in `/home/michael/tfm_worktrees/pwc-offset`);
  - the per-run launch token, the HSA variables and cache records.
- No PWC pool has a hit-rate measurement against `k/1000`.

fallback parity: not assessed this session (not a phase boundary).
