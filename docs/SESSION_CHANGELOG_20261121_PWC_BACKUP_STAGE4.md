# SESSION CHANGELOG — 2026-11-21 — PWC full backup: Stage 3 landing, Stage 4 two-engine session, shutdown fix

Not committed. Named per SR-2 (`docs/PROJECT_FILE_CATALOG.md:90-97`): date and topic, no S-number.
- **Outcome:** the PWC full backup is finished. "Closed" is the lead's word; the owner never said it. He accepted the
  evidence at 16:02 ("the evidence is convincing enough for me- lets move on-") and moved to the RANGE-MINER skip-range
  switch at 16:54 ("yes, also what was the see range of step 4 that we ran?").
- **Scope:** `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md`. **Rulings:** `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`,
  and today's words in the handoff (`docs/TFM_SESSION_HANDOFF_20261121.md`, "Owner rulings and words today").
- **Result record:** `docs/RESULT_PWC_BACKUP_STAGE4_20261121.md`.
- **Worktree:** `/home/michael/tfm_worktrees/pwc-backup`, moved `b2efdef` → `6d75f71` → `6eafad9` (landed from here
  and `pwc-offset`) → `b9b990c` → `ab2d330`. Each move was a mixed `git reset` after a byte-identity check of every
  changed path.
- **Session:** Claude Code on VM101, as `michael`. CPU only for the agent; the rig runs were the owner's and the
  lead's.

## Stage 3 — landed `390c285` (08:15, 96 files) + owner hand-landed `6d75f71` (08:29, 2 files)

- **Content:** the launch path; PWC by flag (`use_persistent_workers` true and `use_range_miner` false); PWC credit
  through PWC's own completion record; the seed-start refusal removed.
  Records: `docs/drafts/pwc_backup_stage3b_v1/README.md`, `docs/SESSION_CHANGELOG_20261120_PWC_BACKUP.md:76-153`.
- **Beta X-PWC-STAGE3 R1** (relay 06:02-06:11, `docs/exchange/LEDGER.md:45`): 2 LIVE, fixed as Beta specified. No
  second round (owner, 06:28).
- **`6d75f71`, landed by the owner by hand:** `agent_manifests/window_optimizer.json` and `agents/watcher_agent.py`,
  which `scripts/owner_land.py:85-87` refuses. Landing lists now carry an "owner lands by hand" group.

## Rig redeploy and parity (11:20-11:25; Alpha at the owner's go: 09:57 "yes-proceed" for power-up, deploy and parity only; a later "yes- proceed", time not recorded, to finish rig-6600b's bring-up and redeploy)

- **Files:** the four PWC rig files changed in Stages 1-3, copied from `git archive 6d75f71`:
  - `persistent/pwc_transport_base.py`, `persistent/pwc_transport_tcp.py`;
  - `persistent/pwc_worker_service.py`, `sieve_filter.py`.

  Rollback copies are in `~/rig_archive/pre_stage4_20261121T192337Z` on each CT100 (lead-reported).
- **Evidence:** main tree `data/stage4/pwc_parity_pre.json` (27 MATCH / 12 MISMATCH) and `pwc_parity_post.json`
  (39/39); `gate12_parity_pre.json` and `_post.json` (30/30 each, identical digests). Read back by this agent.
- **`__pycache__`:** not cleared (no owner OK).
- **Plan, with the status update:** `docs/drafts/pwc_backup_stage4_v1/REDEPLOY_AND_SESSION_PLAN.md`.

## Stage 4 tooling — landed `6eafad9` (13:14, 15 files)

**Comparator** `scripts/compare_engine_outputs.py`:
- `NOT COMPARED` for a field that is absent or null in both bundles, or null in one; exit 3, `SAME WHERE COMPARED`;
- the F2 job identity (`F2_REQUIRED_INPUTS`, 15 keys) must be EQUAL;
- the Stage 4 credit requirement;
- `build_inputs`, `exact_float32`.

**Extractors:**
- `scripts/extract_pwc_bundle.py` (agent 1): the WAL stable copy; coverage and link bound by cross-identifiers; the
  producer's completion and threshold checks; the link required.
- `scripts/extract_miner_bundle.py` (agent 2).

**Runbook and launcher:**
- the derived launcher diff `docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff`: params validated before every
  gate, no `--clear-halt`, the sampler dropped;
- the runbook `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md`.

**Beta:**
- X-PWC-STAGE4 R1 (12:34, 6 LIVE): fixed;
- R2 closing note (13:21): code accepted; R2-Q5 THEORETICAL; Finding 2 LIVE (the PWC launch block), fixed in `b9b990c`.

## The two-engine session (13:22-14:04) — `docs/RESULT_PWC_BACKUP_STAGE4_20261121.md` §1-§4

Owner, 13:18: "you are lead developer- do what you need to run the step 4".

- **Attempt 1** refused, `DATASET_AUTHORITY_P0_5` (`~/stage4_run_miner/logs/gate12_20261121_132251.log:38`).
  - **Remedy:** `scripts/provision_dataset_fleet.py` from the run tree, plus removal of the halt flag attempt 1 left
    (owner, 13:36: "Yes").
  - It is now runbook A2b.
- **Run 1, RANGE-MINER, and Run 2, PWC:** both credited, 0/0/13/2995, 0 NPZ rows, `Prior rows: 0`.
- **Comparator:** SAME WHERE COMPARED, exit 3, 52 EQUAL / 0 DIFFERENT / 10 NOT COMPARED
  (`logs/stage4_session/compare.txt:68`). It does not establish equal per-phase seed sets.
- **The PWC extractor at `6eafad9` refused** (it expected `_t0`; PWC writes the record ordinal `_t1`):
  `window_optimizer_integration_final.py:3273`, `:3296`, `:3316`. Fixed in `b9b990c`.
- **The non-fatal `field_mappings.json` error:** `schemas/field_mappings.json` is the only git-ignored file under
  `schemas/` (`.gitignore:41`). It is now copied at runbook A2.

## PWC shutdown fix — landed `b9b990c` (15:45, 14 files)

- **Defect:** 24 rig workers were alive after `transport stopped` (13:50:49, `~/stage4_run_pwc/logs/stage4/pwc_run.log:86243`).
- **Worker** `persistent/pwc_worker_service.py` (rig file, `f50d5489…` → `31927811…`):
  - a planned `shutdown` ends the process;
  - a session counts from READY (the recorded 5-minute window is kept for an unplanned loss).
- **Coordinator** `persistent_worker_coordinator.py`:
  - a per-coordinator `--launch-token`;
  - a shutdown reap that matches only this coordinator's token, as S156 C3 approved
    (`docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:271-293`);
  - targets recorded at the launch attempt;
  - `shutdown INCOMPLETE` instead of an unconditional all-clear.
- **Extractor:** the record ordinal is discovered and bound by evidence; a zero-row bundle needs `--run-log`; exact
  path match; live-path checks before reading.
- **Runbook:** A2b provisioning; A4b worker redeploy; A6 `PYTHONPATH=.`; C3 fail-closed PWC launch; C5b
  leftover-worker check.
- **Beta X-PWC-SHUTDOWN R1** (14:49): Q7 and Q8 LIVE, fixed; Q3 and Q5 THEORETICAL, also addressed.
- **New suite** `tests/test_pwc_shutdown_reap.py`.
- **Untouched:** `miner/` and the ten gate12-governed files are identical to `6eafad9`.

## Shutdown-fix redeploy and confirmation run

- **Redeploy, 15:46-15:59:** PWC parity before had 1 MISMATCH on each of rrig6600b and rrig6600c (the worker), and
  rrig6600 was 13 UNAVAILABLE (`ssh_transport_failure`). After: PWC 39/39, gate12 30/30
  (`logs/stage4_session/pwc_parity_*_redeploy.json`, `gate12_parity_post_redeploy.json`).
- **Confirmation run, 16:08-16:13** (`~/stage4_run_pwc2`, `b9b990c`; owner, 16:06: "yes"):
  - credited, 0/0/13/2995;
  - `shutdown reap … exited` on all three rigs at 16:13:30 (`logs/stage4/pwc_run.log:86246-86251`);
  - 0 `pwc_worker_service` left anywhere (lead's `pgrep`).

## `ab2d330` (16:52, 9 files)

- `docs/RESULT_PWC_BACKUP_STAGE4_20261121.md`;
- `tests/test_extract_pwc_bundle.py` scratch moved to `tempfile`. G-SUBPROCESS had failed (44/45) in the main tree,
  which is a live path; it was verified at a fake-HOME live path, 45/45;
- the Beta X-PWC-SHUTDOWN R2 fixes:
  - A2b receipts invalidated per attempt;
  - A4b one-file export;
  - C5b exact marker plus exit code, and a re-probe after the token reap;
  - mutants built outside the detector handler;
- the R2 review and the closing notes.

## After `ab2d330` (uncommitted, this worktree)

- **Beta X-PWC-SHUTDOWN R3** (16:58, the last round): R3-Q7 LIVE (A4b could copy a partial export).
  - Fixed: the copy loop starts only after a successful, digest-matched export, and any rig-step failure makes the block
    fail. Dry-run against stubs; not reviewed.
  - Findings 1 and 2: fixed.
- **Result record §7a,** measured facts:
  - the CuPy cache reset before launch, in both PWC runs;
  - wall times (RANGE-MINER about 1 min 50 s; PWC 4 min 50.6 s and 5 min 1 s), which are not a full-size speed
    comparison;
  - the rig-6600b and rig-6600c kernel-log check (lead-reported).
- **These two session records.**

## Rigs (facts only; `docs/TFM_SESSION_HANDOFF_20261121.md` §Rigs)

- **Power:** an outlet power-cycle does not restart the rigs; the power button is needed (owner, 17:38; lead-relayed).
- **rig-6600:**
  - **Down and up repeatedly:** the Proxmox journal's last line before going down was 14:17:01. Ubuntu boots came at
    14:51, 15:51, 16:53, 16:56, 17:38, 17:49, 17:56 and 18:00 (`logs/stage4_session/rig6600_*`). The last three are 4-7
    minutes apart, while idle.
  - **No fault-like kernel lines in either OS's captures.** On Proxmox, `pci=noaer` means AER is not reported.
  - **Boot services:** Ubuntu boots run the S170 DPM peak lock; Proxmox boots run `amdgpu-powercap-min`.
  - **Cause unknown.** The owner is examining the rig.
- **rig-6600b, rig-6600c:** up all day.
- **The owner's words on rig-6600,** recorded as his view, not a finding:
  - 16:50, "ignore.";
  - 17:20, "no i did not sgut down rig-6600- did we have any type of logging running? you have ermission to reboot - then
    pull any logs";
  - 17:38, "Rig6600 is coming back up. For some reason even though I have the bios set to power on after a power outage
    the rigs do this where they won't power up unless I manually press the power button.";
  - 17:54, "Well it's something we did as it never did this before";
  - 17:57, "Well it keeps rebooting".
- **State at close** (lead-reported):
  - rig-6600b and rig-6600c: up in Proxmox, CT100 running, idle, no workers;
  - rig-6600: with the owner, in a restart loop;
  - GPU PSUs on; Zeus and VM101 up.
  - Owner, 18:30: "Let's do all we can without the rigs".

## What the owner credits

- 17:02, "the clean program every run!": the per-run CuPy cache reset (`PWC-CACHE-RESET`).
- Both PWC runs logged `/tmp/cupy_cache_gpu_0..7 emptied and verified` on all three rigs before launch
  (`docs/RESULT_PWC_BACKUP_STAGE4_20261121.md` §7a), and no PWC chunk failed.
- That record shows correlation in two runs, not a measured cause.

## Suites (CPU, sequential; `tests/test_s172_staging_partb.py` not run)

| Suite | Result |
|---|---|
| `test_compare_engine_outputs` | 36/36 |
| `test_extract_pwc_bundle` | 45/45 (also from the fake-HOME live path) |
| `test_pwc_shutdown_reap` | 13/13 |
| `test_stage4_miner_launch` | 24/24 (three runs in a row) |
| `test_pwc_stage1` | 9/9 |
| `test_pwc_backup_step1` | 22/22 (2 HELD: the standing Stage 1 owner hold) |
| `test_pwc_backup_stage2` | 26/26 |
| `test_pwc_cache_reset` | 21/21 |
| `test_pwc_stage2b_anchor_phase` | 35/35 |
| `test_pwc_stage3a` | 6/6 |
| `test_pwc_stage3_r1_fixes` | 4/4 |
| `test_s172_phase5_d3_25_candidate_ingress` | 13/13 |
| `test_s172_resolved_execution_set` | 34/34 |

- **Pre-existing failure, not from this work:** `test_s172_threshold_propagation` 3/5. The same RANGE-MINER-harness FAIL
  (`build_stripe_assign_payload() missing … 'generator_phase'`) reproduces on a clean clone of `6eafad9`.
- **From the main tree** (the lead): `test_extract_pwc_bundle`, `test_pwc_shutdown_reap` and `test_stage4_miner_launch`
  all PASS at `ab2d330`.

## Untouched

- `miner/` and the ten gate12-governed files, in every PWC commit today.
- No live ledger (`~/miner_staging`) or main-tree `prng_analysis.db` use: each Stage 4 run had its own tree and staging.

## Not done

- **No commit or push by the agent.** No rig contact by the agent (all rig actions were the owner's or the lead's).
- **No hit-rate measurement.** Nothing here says anything about TFM quality against `k/1000`.
- **Not reviewed:** the R3 A4b fix (the thread is out of rounds).
- **Open items:** in the handoff, §Open. One is time-critical: **do NOT land the RANGE-MINER skip-range switch while
  rig-6600 is unhealthy.** It changes `miner/range_miner_coordinator.py`, a gate12-governed file, so gate12 parity
  would refuse every RANGE-MINER launch until all three rigs carry it. Beta's X-MINER-SKIP-SWITCH R1 returned three
  LIVE and one THEORETICAL; agent 2 is fixing them (lead-reported).

fallback parity: not assessed this session.
