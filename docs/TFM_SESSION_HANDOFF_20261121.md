# TFM session handoff: 2026-11-21

Written by the PWC agent (Claude Code, worktree `/home/michael/tfm_worktrees/pwc-backup` at main `ab2d330`; not
committed). All times PST.
- Commits, times and file counts were checked against `git log`. Log facts were checked against the files cited.
- Anything this agent could not read is marked **(lead-reported)**.
- The owner's words are verbatim, typos kept, as relayed by the Alpha lead.

## Bottom line

- **The PWC full backup is finished.** "Closed" is the lead's word, not the owner's; the owner never said "closed". He
  accepted the evidence at 16:02 ("the evidence is convincing enough for me- lets move on-") and moved to the next job,
  the RANGE-MINER skip-range switch, at 16:54.
  - **Stage 3** landed this morning (`390c285` + the owner's hand-landed `6d75f71`).
  - **The two-engine Stage 4 session** ran on the rigs: both engines credited, same counts, comparator **SAME WHERE
    COMPARED** (exit 3). Record: `docs/RESULT_PWC_BACKUP_STAGE4_20261121.md`.
  - **A PWC shutdown defect** was found in that session (24 rig workers left alive), fixed (`b9b990c`), redeployed, and
    confirmed on the rigs at 16:08.
  - **Commits today:** `390c285`, `6d75f71`, `6eafad9`, `b9b990c`, `ab2d330`.
- **Hit rate was not measured.** Nothing today says anything about TFM quality against `k/1000`.
- **rig-6600 is unstable** (§Rigs). It reset while idle from about 14:49, and by 17:49-18:00 was cycling every 4-7
  minutes in Ubuntu. No kernel error was found in either OS; the cause is not known. The owner is examining the rig.
  rig-6600b and rig-6600c stayed up.

## Landed

| Commit | When (PST) | Files | What |
|---|---|---|---|
| `390c285` | 08:15 | 96 | PWC Stage 3: the launch path; PWC by flag; credit; Beta X-PWC-STAGE3 R1 (RANGE-MINER no objection; two LIVE fixed as Beta specified; no second round by owner decision) |
| `6d75f71` | 08:29 | 2 | Stage 3's governed files, **hand-landed by the owner**: `agent_manifests/window_optimizer.json` (RANGE-MINER default), `agents/watcher_agent.py` (the `require_fleet` keyword). `scripts/owner_land.py:85-87` refuses these paths |
| `6eafad9` | 13:14 | 15 | Stage 4 tooling: comparator NOT COMPARED and exit 3; F2 job identity and credit requirement; PWC and miner bundle extractors; derived launcher diff; runbook; Beta X-PWC-STAGE4 R1 and fixes |
| `b9b990c` | 15:45 | 14 | **PWC shutdown fix:** the worker exits on a planned shutdown; the coordinator runs a token-bound reap. Also the PWC extractor record-ordinal fix and runbook corrections; Beta X-PWC-SHUTDOWN R1 and fixes |
| `ab2d330` | 16:52 | 9 | The Stage 4 result record; post-landing test fixes (tempfile scratch for `test_extract_pwc_bundle`); runbook corrections; Beta X-PWC-SHUTDOWN R2 review and closing notes |

## The day in order

1. **Stage 3 landed** after Beta X-PWC-STAGE3 R1 (relay 06:02-06:11, `docs/exchange/LEDGER.md:45`).
   - Owner, 05:56: "yes please send to beta".
   - Owner, 06:28: "yes, since there are only two beta rejections and you have already addessed them land stage 3"
     (`docs/SESSION_CHANGELOG_20261120_PWC_BACKUP.md:121`, `:150`).
2. **Rig redeploy and parity, 11:20-11:25** (Alpha, at the owner's go; lead-reported times). The four PWC files changed
   since any CT100 could have received them were copied from `git archive 6d75f71`. Evidence is in the main tree's
   `data/stage4/` and was read by this agent:
   - PWC parity before: 27 MATCH / 12 MISMATCH, the four files × three rigs;
   - PWC parity after: 39/39 MATCH;
   - gate12 parity: 30/30 MATCH before and after, with identical digests.

   **The owner's go:**
   - 09:57, "yes-proceed": rig power-up, for the deploy and parity checks only;
   - later, "yes- proceed": finish rig-6600b's bring-up and do the redeploy. The time was not recorded (lead-relayed).
3. **Stage 4 tooling** (`6eafad9`) after Beta X-PWC-STAGE4 R1 (12:34) and R2 (13:21) (`docs/exchange/LEDGER.md:46-47`).
4. **The two-engine session** (owner, 13:18: "you are lead developer- do what you need to run the step 4"), with separate run trees and a separate miner staging directory, so the live tree and
   ledger were untouched. Details and anchors: `docs/RESULT_PWC_BACKUP_STAGE4_20261121.md` §1-§4.
   - **Attempt 1** (13:22-13:24) was refused before dispatch, `DATASET_AUTHORITY_P0_5`
     (`~/stage4_run_miner/logs/gate12_20261121_132251.log:38`). The remedy is `scripts/provision_dataset_fleet.py` from
     the run tree, plus removal of the halt flag that attempt 1 left (owner, 13:36: "Yes").
   - **Run 1, RANGE-MINER** (`~/stage4_run_miner/logs/gate12_20261121_133719.log`):
     - credited (`:279`);
     - 0 rows published, `Prior rows: 0` (`:281`, `:286`);
     - phase rows 0/0/13/2995.
   - **Run 2, PWC** (`~/stage4_run_pwc/logs/stage4/pwc_run.log`): credited (`:86288`); 0/0/13/2995.
   - **The NPZ files** are byte-identical: `aecc5d50…`, 0 rows.
   - **Comparator** (`~/distributed_prng_analysis/logs/stage4_session/compare.txt:68`): SAME WHERE COMPARED, exit 3,
     **52 EQUAL, 0 DIFFERENT, 10 NOT COMPARED**. The 10 are RANGE-MINER's per-phase seed lists, which are not
     persisted.

     It establishes the same job, the same counts, the same credit and the same (empty) NPZ. It does not establish
     equal per-phase seed sets.
   - **Owner, 16:02:** "the evidence is convincing enough for me- lets move on-".
5. **24 PWC rig workers left alive** after `transport stopped` at 13:50:49 (`~/stage4_run_pwc/logs/stage4/pwc_run.log:86243`);
   stopped by hand (lead-reported).
   - Owner, 14:07: "Shouldn't we address the issues you found first?"
   - Fix sent to Beta (owner, 14:39: "Ok if you believe that's what needs done").
   - Landed `b9b990c` (owner, 15:44: "Yes, once again do whatever you require to run step 4").
6. **Shutdown-fix redeploy, 15:46-15:59.** `persistent/pwc_worker_service.py` was copied
   (`logs/stage4_session/pwc_parity_pre_redeploy.json`, `pwc_parity_post_redeploy.json`, `gate12_parity_post_redeploy.json`):
   - PWC parity before: rrig6600b and rrig6600c each had 1 MISMATCH, the worker file; rrig6600 was 13 UNAVAILABLE
     (`ssh_transport_failure`);
   - after: PWC 39/39, gate12 30/30.
7. **The confirmation run, 16:08-16:13** (owner, 16:06: "yes") (`~/stage4_run_pwc2`, `b9b990c`, `logs/stage4/pwc_run.log`):
   - **Setup:** provisioned from the tree first (`RESULT: PASS`); `schemas/field_mappings.json` copied in, and no
     `field_mappings` error this time.
   - **Token and passes:** launch token `1ddc2f0b8ca244daa6a0a974bdcf3822` (`:143-144`); every pass 10,738/10,738 chunks.
   - **Result:** 0/0/13/2995, credited (`:86296`), `Prior rows: 0` (`:86303`).
   - **Shutdown:** `transport stopped` 16:13:27 (`:86245`); `shutdown reap … exited` for .122/.156/.164 at 16:13:30
     (`:86246-86251`); `All workers shut down` (`:86253`).
   - **Independent check:** the lead's `pgrep` found 0 `pwc_worker_service` on all three rigs and on VM101, with nothing
     stopped by hand (lead-reported).
8. **`ab2d330`** landed (owner, 16:51: "yes"). The X-PWC-SHUTDOWN R3 closing note was then sent for the record
   (`docs/exchange/LEDGER.md:50`).

## Owner rulings and words today (verbatim)

| Time | Words | Context | Source |
|---|---|---|---|
| 05:56 | "yes please send to beta" | Relay Stage 3 to Beta | `docs/SESSION_CHANGELOG_20261120_PWC_BACKUP.md:121-122` |
| 06:28 | "yes, since there are only two beta rejections and you have already addessed them land stage 3" | Land Stage 3, no second round | `…_20261120_PWC_BACKUP.md:150` |
| 09:57 | "yes-proceed" | Rig power-up, for the deploy and parity checks only | lead-relayed |
| not recorded | "yes- proceed" | Finish rig-6600b's bring-up and do the redeploy | lead-relayed |
| 11:34 | "leave the rigs up- we dont want any more issues- proceed with the work" | After the morning redeploy | lead-relayed |
| 12:21 | "Proceed" | Send the Stage 4 tooling to Beta | `docs/exchange/to_beta/X-PWC-STAGE4_R1_ALPHA_REVIEW_REQUEST_stage4_tooling.md:9-10` |
| 13:12 | "yes" | Land the tooling without a second Beta review | `docs/exchange/to_beta/X-PWC-STAGE4_R2_ALPHA_CLOSING_NOTE_stage4_tooling.md:19` |
| 13:18 | "you are lead developer- do what you need to run the step 4" | Run the Stage 4 session | lead-relayed |
| 13:36 | "Yes" | Provision the dataset to the rigs' run-tree paths, and remove the halt flag left by refused attempt 1 | lead-relayed |
| 14:07 | "Shouldn't we address the issues you found first?" | Fix before the write-up | lead-relayed |
| 14:39 | "Ok if you believe that's what needs done" | Send the shutdown fix to Beta | `docs/exchange/to_beta/X-PWC-SHUTDOWN_R1_ALPHA_REVIEW_REQUEST_shutdown_and_stage4_fixes.md:3-5` |
| 15:44 | "Yes, once again do whatever you require to run step 4" | Land `b9b990c` | `docs/exchange/to_beta/X-PWC-SHUTDOWN_R2_ALPHA_CLOSING_NOTE_shutdown.md:15` |
| 16:02 | "the evidence is convincing enough for me- lets move on-" | Accept the same-counts evidence; the miner phase capture parked | `~/tfm_worktrees/pwc-offset/docs/drafts/pwc_backup_stage4_v1/PARKED_MINER_PHASE_CAPTURE.md:3` |
| 16:06 | "yes" | One short PWC-only run to confirm the shutdown fix (the 16:08 run) | lead-relayed |
| 16:50 | "ignore." | On rig-6600's first reset | lead-relayed |
| 16:51 | "yes" | Land the 9 files (`ab2d330`) and send the closing note | `docs/exchange/to_beta/X-PWC-SHUTDOWN_R3_ALPHA_CLOSING_NOTE_stage4.md:15` (uncommitted) |
| 16:54 | "yes, also what was the see range of step 4 that we ran?" | "yes": start the RANGE-MINER skip-range switch. The range run was `[2147483648, 2415919104)` = `[2^31, 2^31 + 2^28)` (`~/stage4_run_pwc2/logs/stage4/pwc_run.log:86296`) | lead-relayed |
| 17:02 | "the clean program every run!" | The fix the owner credits for zero PWC crashes: `PWC-CACHE-RESET`, which empties `/tmp/cupy_cache_gpu_0..7` on every rig before every PWC launch (`docs/RESULT_PWC_BACKUP_STAGE4_20261121.md` §7a) | lead-relayed |
| 17:20 | "no i did not sgut down rig-6600- did we have any type of logging running? you have ermission to reboot - then pull any logs" | rig-6600 down; the captures in `logs/stage4_session/rig6600_*` | lead-relayed |
| 17:38 | "Rig6600 is coming back up. For some reason even though I have the bios set to power on after a power outage the rigs do this where they won't power up unless I manually press the power button." | Rig power | lead-relayed |
| 17:54 | "Well it's something we did as it never did this before" | rig-6600 cycling. The owner's view, recorded as such; no cause is established here (§Rigs) | lead-relayed |
| 17:57 | "Well it keeps rebooting" | rig-6600 | lead-relayed |
| 18:09 | "Is all of our hard work and success recorded for today?" | These records | lead-relayed |
| 18:30 | "Let's do all we can without the rigs" | The work after the rigs | lead-relayed |

- **The 15:46 shutdown-fix redeploy:** no separate owner words were relayed. It followed the 15:44 approval to land and
  run step 4.
- **"Closing" the PWC backup:** no such words exist (Bottom line).

## Beta exchanges today

| Thread | Round (relay time) | Positions | Outcome |
|---|---|---|---|
| X-PWC-STAGE3 | R1 (06:11) | 2 LIVE (3b-Q3, 3b-Q7), 5 SUGGESTIONs, 10 NO OBJECTIONs | Fixed as Beta specified; landed `390c285` |
| X-PWC-STAGE4 | R1 (12:34) | LIVE: Q2, Q3, Q5, Q6, Q7, Q8 | Fixed; landed in `6eafad9` |
| X-PWC-STAGE4 | R2 (13:21), closing note | R2-Q5 THEORETICAL (deferred); Finding 2 LIVE (the runbook's PWC launch block) | Fixed in `b9b990c` |
| X-PWC-SHUTDOWN | R1 (14:49) | Q1, Q2, Q4, Q6 NO OBJECTION; Q3, Q5 THEORETICAL; Q7, Q8 LIVE | Fixed (token-bound reap, zero-row `--run-log`, runbook, tests); landed `b9b990c` |
| X-PWC-SHUTDOWN | R2 (15:57), closing note | Code accepted; R2-Q7 and R2-Q8 LIVE (runbook, tests) | Fixed; landed `ab2d330` |
| X-PWC-SHUTDOWN | R3 (16:58), closing note | R3-Q8 and R3-E1 NO OBJECTION; **R3-Q7 LIVE** (A4b partial export can overwrite a rig worker); Findings 1, 2 | Fixed in this worktree, **uncommitted**. The thread is out of rounds, so the fix is not reviewed |

Ledger rows: `docs/exchange/LEDGER.md:45-50`. THEORETICAL items are in `docs/exchange/DEFERRED.md`.

## Rigs (measured facts; no cause claimed)

- **Power:** an outlet power-cycle does not restart the rigs; the power button is needed. Owner, 17:38, verbatim:
  "Rig6600 is coming back up. For some reason even though I have the bios set to power on after a power outage the rigs
  do this where they won't power up unless I manually press the power button."
- **rig-6600 (`.120` Ubuntu / `.121`/`.122` Proxmox)**, from the captures in `logs/stage4_session/rig6600_*` (main tree):
  - **Proxmox boot −2** ran 10:40:28 to its last journal line at 14:17:01 (`rig6600_pve_boot_minus2.log`). Boot −1 ran
    15:56:57 to 16:47:29 (`rig6600_pve_boot_minus1.log`).
  - **Ubuntu boots** (wtmp, `rig6600_ubuntu_grab_180044.txt`): 10:13, 10:26, **14:51**, 15:51, 16:53, 16:56, 17:38,
    17:49, 17:56, 18:00. The last three are 4-7 minutes apart, while idle.
  - **Fault-like kernel lines:** none in the Ubuntu capture (50 lines; GCVM, PROTECTION_FAULT, page fault, ring timeout,
    GPU reset, `*ERROR*`, AER, Hardware Error, MES fail, panic, oops, watchdog).
    - In both Proxmox captures the only matches are boot lines: the kernel command line and watchdog initialisation.
    - **Limit:** that command line carries `pci=noaer`, so AER is not reported on the Proxmox side. A missing AER line
      there is not evidence.
  - **Boot services:** the Ubuntu boot runs the S170 DPM peak lock (`s170_dpm_peak_lock.sh`, e.g. "profile=COMPUTE(5)",
    `rig6600_ubuntu_grab_180044.txt`). The Proxmox boots run `amdgpu-powercap-min.service` ("Pin amdgpu GPUs to minimum
    power cap (S172 sag mitigation)", `rig6600_pve_boot_minus1.log`, `_minus2.log`).
  - At 17:56 `ssh` to `.120` timed out (`rig6600_ubuntu_boots.txt`).
  - **The owner's words, recorded as his view, not a finding:** 16:50 on the first reset, "ignore."; 17:20, "no i did
    not sgut down rig-6600- did we have any type of logging running? you have ermission to reboot - then pull any
    logs"; 17:54, "Well it's something we did as it never did this before"; 17:57, "Well it keeps rebooting".
  - **No cause has been established.** The owner is examining the rig.
- **rig-6600b and rig-6600c stayed up.** Their Proxmox kernel logs for 13:35-13:56 and 16:05-16:16, checked at 17:11,
  had 0 fault-like lines (lead-reported; `docs/RESULT_PWC_BACKUP_STAGE4_20261121.md` §7a).

## Open

1. **RANGE-MINER constant skip-range switch (F-1).** Agent 2's work, in `~/tfm_worktrees/pwc-offset`, uncommitted.
   - Request: `docs/exchange/to_beta/X-MINER-SKIP-SWITCH_R1_ALPHA_REVIEW_REQUEST_constant_skip_delivery_switch.md`.
   - Started at the owner's 16:54 "yes".
   - Beta's X-MINER-SKIP-SWITCH R1 came back with three LIVE objections and one THEORETICAL; agent 2 is fixing them
     (lead-reported).
   - **Not landed.** Today's comparison matched only because the trial was pinned to `[0,16]`.
   - **Do NOT land it while rig-6600 is unhealthy.** It changes `miner/range_miner_coordinator.py`, a gate12-governed
     file (`scripts/gate12_parity_gate.py:174-185`). gate12 parity would then refuse every RANGE-MINER launch until all
     three rigs carry the new coordinator file, and rig-6600 cannot be redeployed while it is in a restart loop.
2. **RANGE-MINER per-phase capture: PARKED** by the owner (16:02; agent 2's `PARKED_MINER_PHASE_CAPTURE.md`).
3. **Parallel trials (n_parallel > 1): parked**, low priority (owner, 2026-11-20 20:24 and 20:41;
   `docs/TFM_SESSION_HANDOFF_20261120.md:131-146`).
4. **AlphaGo / selfplay research note:** a project document, `claude/NOTE_ALPHAGO_LOOP_FOR_SELFPLAY_20261121.md`, not in
   the repository (lead-reported; not read by this agent).
5. **Telegram boot-notify DNS timing fix: not applied** (lead-reported).
6. **Netconsole is not loaded on the rigs.** The VM101 listener (`~/distributed_prng_analysis/logs/netconsole_all_rigs.log`)
   holds only its own `LISTENER STARTED` lines since 2026-11-19 22:42 (restarts 2026-11-20 12:57 and 2026-11-21 05:48):
   no rig line since.
7. **Test folders left behind:**
   - **VM101:** `~/stage4_run_miner`, `~/stage4_run_pwc`, `~/stage4_run_pwc2`, `~/stage4_miner_staging`,
     `/home/michael/stage4_miner_launch.sh`, `~/distributed_prng_analysis/logs/stage4_session/`.
   - **Rigs:** the provisioned dataset copies at the run-tree paths, and `/tmp/pwc_tcp_launch_gpu*.sh` and
     `/tmp/pwc_tcp_worker_*.log` [UNVERIFIED exact list; rigs not read].
   - Removing them is the owner's call.
8. **Beta's five-strategy question (X-PWC-STAGE2 R1-Q3)** is partly answered. The rigs completed every five-strategy
   chunk twice, with no failure or timeout. Not measured: per-chunk times against the 600 s timeout, peak memory, and
   host KFD/MES events (`docs/RESULT_PWC_BACKUP_STAGE4_20261121.md` §8).
9. **Record-stamp question** (`RESULT_…` §8): NOT FOUND under that name. The related fact is that the record ordinal is
   1 while Optuna and the credit line say trial 0. Lead to confirm.
10. **Uncommitted in this worktree:**
    - the X-PWC-SHUTDOWN R3 fixes (runbook A4b, result record Finding 2 and §7a);
    - Beta's R3 reply and the sent R3 note;
    - the relay's ledger row;
    - these two session records.

## State at close

- **Repo:** main `ab2d330`. This worktree: HEAD `ab2d330`, plus the uncommitted files in Open item 10 and untracked `data/`.
- **Rigs** (lead-reported):
  - rig-6600b and rig-6600c: up in Proxmox, CT100 running, idle, no workers;
  - rig-6600: with the owner, in a restart loop;
  - GPU PSUs on.
- **Zeus and VM101:** up (lead-reported).
- **Owner, 18:30:** "Let's do all we can without the rigs".
- **Live ledger** `~/miner_staging` and the main tree's `prng_analysis.db`: not used by any Stage 4 run, since each run
  had its own tree and staging. Not read by this agent.

fallback parity: not assessed this session.
