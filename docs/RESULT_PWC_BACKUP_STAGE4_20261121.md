# RESULT: PWC full backup, Stage 4: the first two-engine session (2026-11-21)

A plain record of what ran, what it showed and what is open. Written by Claude (Team Alpha, agent 1) in
`/home/michael/tfm_worktrees/pwc-backup`. First written at `b9b990c` and landed in `ab2d330`; revised afterwards
(uncommitted at the time of writing) for Beta R3 Finding 2 and the measured facts in §7a.

The run trees and `logs/stage4_session` were read, read-only. Items the Alpha lead reported but this session could not
read are marked **(lead-reported)**.

**Hit rate was not measured.** Nothing here says anything about TFM quality, or about any engine's survivors against
`k/1000`. The session tested engine equality and worker cleanup, both engineering properties.

## Prior record

- **Searched,** for the subjects below: `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`,
  `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`,
  `docs/LEADS.md`, `docs/exchange/`.
- **Search scopes:**
  - the session runbook's and the Beta threads' own prior-record sections;
  - `pwc_worker_service` shutdown and reap;
  - dataset provisioning;
  - engine comparison;
  - five strategies;
  - "record stamp".

| Document | What it records | Label |
|---|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47` | Owner, 2026-11-20: "run each and verify the results are the same" | ALREADY RECORDED: this session answers it, partially (§4) |
| `docs/drafts/pwc_backup_stage3_v1/PLAN.md:299-325` | F2 (same job) and F3 (what is compared) | ALREADY RECORDED |
| `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md` (as landed in `b9b990c`) | The session procedure | ALREADY RECORDED |
| `docs/exchange/from_beta/X-PWC-STAGE4_R1_BETA_REVIEW.md`, `_R2_…`; `X-PWC-SHUTDOWN_R1_BETA_REVIEW.md`, `_R2_…` | Beta's reviews of the tooling and the shutdown fix | ALREADY RECORDED |
| `docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md:26`, `:58-62` | Q3 "Five strategies on the rigs": NEEDS EVIDENCE, with the measurements Beta asked for | ALREADY RECORDED, open (§8) |
| `docs/PROPOSAL_S96B_PERSISTENT_GPU_WORKERS_v1_0.md:64-70`; `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:271-293` | The worker exits on shutdown; an explicit reap at shutdown | ALREADY RECORDED; implemented in `b9b990c` (§5) |
| `docs/VERIFICATION_INTEGRITY_STANDARD.md:30-47`, `:68-107` | Mutant independence; a cleanup observation that cannot see is not a pass | ALREADY RECORDED (Beta X-PWC-SHUTDOWN R3 Finding 1) |
| `docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md:82-90`, `:115-126`, `:156-167` | On-target digest verification and retained per-node evidence (§1's remedy) | ALREADY RECORDED (R3 Finding 1) |
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80` (F-1) | RANGE-MINER's constant phases never receive the trial's skip range | ALREADY RECORDED, open (§8) |

- **NEW:** this record; the rig-6600 reset as an observation (§7).
- **CHANGES A RULING:** nothing.
- **NOT FOUND:** a recorded "record-stamp issue" under that name (§8).
- **Not searched:** the host, Proxmox journals, ser8 archives, the private repository, the rigs.

## 1. Attempt 1 refused before dispatch, and the remedy

- **What happened:** Run 1 attempt 1, 13:22-13:24 PST, tree `~/stage4_run_miner`, was refused before any dispatch:
  `DATASET_AUTHORITY_P0_5: FAIL BEFORE DISPATCH — the dataset is not verified on every node`
  (`~/stage4_run_miner/logs/gate12_20261121_132251.log:38`, `:57`; `logs/stage4_session/agent_halt_run1_attempt1.txt`).
- **Cause:** P0.5 freezes the dataset at the run tree's own absolute path (`…_132251.log:33`:
  `/home/michael/stage4_run_miner/daily3-20260801T145551443433Z-513648160d35.json`). Every node must hold the file at
  exactly that path (`miner/dataset_authority.py:855-859`), and no node had ever been provisioned for a new tree.
- **Remedy:** `scripts/provision_dataset_fleet.py`, run from the run tree. The Alpha lead ran it by hand. Its log in
  the miner tree ends `All 3 node(s) verified ON TARGET against the frozen identity. RESULT: PASS`
  (`~/stage4_run_miner/logs/stage4/provision.log`).
- **Now:** it is runbook step A2b, written so a failed retry cannot leave a stale success receipt (Beta
  X-PWC-SHUTDOWN R2 Q7).

## 2. Run 1, RANGE-MINER (tree `~/stage4_run_miner`, `6eafad9`)

- **Credit:** `[STEP1-SEARCH S7] credited: trials [0] completed every required phase over [2147483648, 2415919104]`
  (`~/stage4_run_miner/logs/gate12_20261121_133719.log:279`).
- **Publication:** `GENERATION PUBLISHED 0 rows`, `Prior rows: 0` (`:281`, `:286`); coverage `40de4c4df834c121`
  (`:291`); link to search `s1s-d85eabde011c26df` (`:292`).
- **Phase rows** (miner ledger stripes, via the miner extractor): **0 / 0 / 13 / 2995**; bidirectional 0
  (`logs/stage4_session/extract_miner.log:3`).
- **Bundle:** `logs/stage4_session/bundle_miner/bundle.json`, with receipt `miner_receipt.json`. The NPZ artifact is
  `aecc5d504784ecca…` (0 rows).

## 3. Run 2, PWC (tree `~/stage4_run_pwc`, `6eafad9`)

- **Pass files:** `Results saved` lines at `~/stage4_run_pwc/logs/stage4/pwc_run.log:21759-86236`, survivors
  **0 / 0 / 13 / 2995**.
- **Credit and publication:** `credited: trials [0] …` (`:86288`); `GENERATION PUBLISHED 0 rows` (`:86290`);
  `Prior rows: 0` (`:86295`); coverage `e08eaee706ad93f4` (`:86300`).
- **Non-fatal error:** `New results format failed: Schema file not found: …/schemas/field_mappings.json` (`:86305`),
  after credit. It is now runbook A2.
- **The NPZ artifact** is byte-identical to Run 1's: `aecc5d504784ecca…`, 0 rows.
- **Extraction:**
  - at `6eafad9` the PWC extractor refused (`logs/stage4_session/extract_pwc.log`): it expected `_t0` pass files, but
    PWC writes the record ordinal `_t1`;
  - after the fix (`b9b990c`) it gave rc 0, ordinal 1, `bundle_pwc/bundle.json`;
  - that extraction predates the zero-row `--run-log` rule, which the current extractor would require.

## 4. The comparison: SAME WHERE COMPARED, exit 3

- **Run:** the Alpha lead ran `scripts/compare_engine_outputs.py` on the two bundles
  (`logs/stage4_session/compare.txt`, `compare.json`, 14:04 PST).
- **Verdict:** `SAME WHERE COMPARED`, **exit 3: 52 EQUAL, 0 DIFFERENT, 10 NOT COMPARED** (`compare.txt:68`).

**What it establishes:**
- **The same job.** All 17 input rows were EQUAL. They include the 15 F2 identity keys (pins, interval, family,
  modes, commit, dataset digest) and two other parameters, `n_parallel` and `window_trials`. The 52 EQUAL rows are
  17 inputs, 4 survivor counts, 7 credit fields and 24 NPZ rows (22 arrays and 2 order rows).
- **The same counts and credit.** The four per-phase survivor counts were EQUAL (0 / 0 / 13 / 2995). Both runs had
  credit granted with equal certified, linked and swept intervals.
- **The same NPZ.** The 22-array NPZ was bit-equal, with 0 rows on both sides.

**What it does not establish:**
- **The per-phase seed lists were not compared** (`phases.1-4`). RANGE-MINER does not persist them; its four
  seed→rate maps live only in memory (`compare.txt:22-25`). Equal counts do not show equal sets: two engines could
  return 13 different seeds in phase 3.
- **The bidirectional sets were not compared**, for the same reason (`:26-27`). Both are empty, which the 0-row NPZs
  agree with.
- **Distinct-survivor counts were not compared** (`:28-34`).
- **Survivor quality was not assessed.** Nothing about hit rate, and nothing about any survivor's merit.

**Owner, 16:02 PST, verbatim:** "the evidence is convincing enough for me- lets move on-". The per-phase capture for
RANGE-MINER, which would close the NOT COMPARED fields, is parked (agent 2:
`~/tfm_worktrees/pwc-offset/docs/drafts/pwc_backup_stage4_v1/PARKED_MINER_PHASE_CAPTURE.md:3`).

## 5. 24 PWC workers left alive, and the fix

- **What happened:** after Run 2's `[PWC-TCP] transport stopped` at 13:50:49 (`~/stage4_run_pwc/logs/stage4/pwc_run.log:86243`),
  all 24 rig `pwc_worker_service` processes were still alive 2 minutes later. The Alpha lead stopped them by hand
  **(lead-reported)**.
- **Cause:** a planned `shutdown` ended only the worker's session, and the worker reconnected for about 5 minutes. A
  session ended by a dropped connection retried forever. The coordinator's S156 reap ran only at startup.
- **Fix:** landed as **`b9b990c`** (owner, 15:44 PST: "Yes, once again do whatever you require to run step 4"):
  - the worker exits on a planned shutdown;
  - a session counts from READY;
  - the coordinator reaps, after the transport stops, only the workers carrying its own `--launch-token`;
  - reap targets are recorded at the launch attempt.
- **Review:** Beta X-PWC-SHUTDOWN R1 and R2 accepted the code. R2's two LIVE items were in the runbook and tests; their
  fixes are in `ab2d330` (owner, 16:51 PST: "yes").
  - Beta's R3 (`docs/exchange/from_beta/X-PWC-SHUTDOWN_R3_BETA_REVIEW.md`) raised one more LIVE item, runbook A4b's
    partial-export case. It is fixed in the runbook, uncommitted at the time of writing.
  - The thread is out of rounds, so that fix is not reviewed.
- **Redeploy** (owner-approved, 15:46-15:59 PST): `persistent/pwc_worker_service.py` copied to the rigs.
  - PWC parity before (`logs/stage4_session/pwc_parity_pre_redeploy.json`, 23:46 UTC): rrig6600b and rrig6600c 12
    MATCH plus 1 MISMATCH each (the worker file); **rrig6600 13 UNAVAILABLE (`ssh_transport_failure`)**, see §7.
  - After (`pwc_parity_post_redeploy.json`, 23:59 UTC): **39/39 MATCH**.
  - gate12 after (`gate12_parity_post_redeploy.json`): **30/30 MATCH**.

## 6. The confirmation run, 16:08-16:13 PST (tree `~/stage4_run_pwc2`, `b9b990c`)

- **Provisioning:** the dataset was provisioned from the tree first (`~/stage4_run_pwc2/logs/stage4/provision.log`:
  `RESULT: PASS`), and `schemas/field_mappings.json` was copied in. No `field_mappings` error this time.
- **Launch token:** `1ddc2f0b8ca244daa6a0a974bdcf3822` (`~/stage4_run_pwc2/logs/stage4/pwc_run.log:143-144`).
- **Pass files:** `Results saved` 0 / 0 / 13 / 2995 (`:21761-86231`). Every pass was 10,738/10,738 chunks complete,
  0 failed, 0 timed out, at 25,000 seeds per chunk and 8 workers per rig (the pass files' `pwc_pass_accounting`).
- **Credit and publication:** `credited: trials [0]` (`:86296`); `GENERATION PUBLISHED 0 rows` and `Prior rows: 0`
  (`:86298`, `:86303`).
- **Shutdown:**
  - `[PWC-TCP] transport stopped` at 16:13:27 (`:86245`);
  - `shutdown reap 192.168.3.122 / .156 / .164: exited` at 16:13:30 (`:86246-86251`);
  - `All workers shut down` (`:86253`).
- **Independent check:** the Alpha lead's `pgrep` found **0** `pwc_worker_service` on all three rigs and on VM101, with
  nothing stopped by hand **(lead-reported)**.
- **Not tested on the rigs:** the token-bound reap's kill path. Every worker exited by itself, so "killed",
  "survivor" and "unavailable" were exercised only in CPU tests.
- **An unrelated warning:** a soft preflight warning `Preflight BLOCKED: Missing inputs: ['synthetic_lottery.json']`,
  "continuing" (`:31-32`), appears in all three runs' logs. It did not stop any run. Its meaning is not investigated
  here.

## 7. rig-6600 silent reset while idle

- **Observed (lead-reported):** between 14:17 and 14:50 PST, with no work running, rig-6600 reset. The last Proxmox
  journal line is 14:17:01, and Ubuntu booted at 14:50:40. **Cause unknown.**
- **Evidence in this session's files:** the 15:46 PWC parity read rrig6600 as UNAVAILABLE (`ssh_transport_failure`,
  13 rows, `pwc_parity_pre_redeploy.json`). By 15:59 all 39 rows matched.
- **Not read by this session:** the Proxmox journal, the boot logs, netconsole.
- **Recorded as an observation, not a lead conclusion.** Whether it relates to the open rig GPU-fault leads
  (`docs/LEADS.md` L-6, L-7) is not assessed here.

## 7a. Further measured facts, with their limits

- **Kernel logs (lead-reported; this session could not read the hosts):**
  - **When and where:** at 17:11 PST, on the Proxmox hosts of rig-6600b and rig-6600c, for the windows 13:35-13:56
    (Run 2) and 16:05-16:16 (confirmation run).
  - **What was searched for:** GCVM, PROTECTION_FAULT, page fault, ring timeout, GPU reset, `*ERROR*`, AER,
    Hardware Error, MES fail.
  - **Result:** 0 fault-like lines. The windows held 17 and 1 kernel lines on one host and 16 and 1 on the other.
  - **Not read:** rig-6600's host.
  - **Limits:** an absence of those strings in those windows on two of three hosts. It is not a health claim, and it
    says nothing about rig-6600.
- **CuPy cache reset before launch** (verified in the logs). Both PWC runs logged `[PWC-CACHE-RESET] <rig>:
  /tmp/cupy_cache_gpu_0..7 emptied and verified` for all three rigs before any worker launched:
  - Run 2, 13:45:43-13:45:55 (`~/stage4_run_pwc/logs/stage4/pwc_run.log:150`, `:188`, `:226`);
  - the confirmation run, 16:08:10-16:08:23 (`~/stage4_run_pwc2/logs/stage4/pwc_run.log:152`, `:190`, `:228`).
- **Wall times** (verified in the logs):

  | Run | From | To | Duration |
  |---|---|---|---|
  | RANGE-MINER | all 25 workers registered 13:39:07.9 (`~/stage4_run_miner/logs/gate12_20261121_133719.log:151`; the first registered at 13:39:05.4, `:127`) | trial commit cleanup done 13:40:57.8 (`:200`) | about 1 min 50 s (1 min 52 s from the first registration) |
  | PWC Run 2 | `25 workers launched` 13:45:58.7 (`~/stage4_run_pwc/logs/stage4/pwc_run.log:262`) | `transport stopped` 13:50:49.3 (`:86243`) | 4 min 50.6 s |
  | PWC confirmation | 16:08:26.5 (`~/stage4_run_pwc2/logs/stage4/pwc_run.log:264`) | 16:13:27.5 (`:86245`) | 5 min 1 s |

  **Not a full-size speed comparison:**
  - PWC ran inside its stable envelope: 8 workers per rig, 25,000 seeds per chunk, 10,738 chunks per pass (the pass
    files' `pwc_pass_accounting`).
  - RANGE-MINER ran only 4 stripes per phase, 2^26 seeds each (`logs/stage4_session/bundle_miner/bundle.json` geometry
    `miner_stripes_per_phase`), so at most 4 of its workers held a stripe at once.
  - The two engines' times at full size are not measured here.

## 8. Open items

1. **Beta's X-PWC-STAGE2 R1 Q3, five strategies on the rigs** (NEEDS EVIDENCE, `X-PWC-STAGE2_R1_BETA_REVIEW.md:58-62`).
   - **What this session supplies:** at the 8 workers per rig and 25,000 seeds per chunk point, the five-strategy
     variable-skip passes (phases 3-4):
     - completed 10,738/10,738 chunks with 0 failed and 0 timed out, in two runs (13:4x and 16:1x);
     - logged "Loaded 5 strategies for hybrid sieve" (`~/stage4_run_pwc2/logs/stage4/pwc_run.log:43254`, `:64744`);
     - each pass took about 71 s wall time (`Results saved` timestamps `:64741` and `:86231`);
     - phase-4 survivors carried strategy ids 4 (2,970) and 1 (25) in the pass file.
   - **What it does not supply:**
     - per-chunk durations, so the tail against the 600 s result timeout is not measured;
     - peak memory;
     - host KFD/MES events. GPU monitoring reads were off-limits while cards worked.
     - Any one- versus five-strategy comparison on unseen draws: hit rate was not measured.
2. **The record-stamp issue: NOT FOUND under that name.** Searched: `docs/` in all three trees, and agent 2's Stage 4
   drafts.
   - **This session's related fact:** PWC pass files and candidate rows are stamped with the integration's record
     ordinal (1), while Optuna and the credit line call the trial 0 (`window_optimizer_integration_final.py:3273`,
     `:3296`, `:3316`).
   - Whether that is the issue meant needs the lead's confirmation.
3. **Test folders left on VM101 and the rigs.**
   - **VM101:** run trees `~/stage4_run_miner`, `~/stage4_run_pwc` and `~/stage4_run_pwc2`; `~/stage4_miner_staging`;
     `/home/michael/stage4_miner_launch.sh`; evidence `~/distributed_prng_analysis/logs/stage4_session/`.
   - **Rigs:** the provisioned dataset copies at each run tree's path [UNVERIFIED exact list: provisioning places the
     file at the frozen path]; the per-GPU launch scripts `/tmp/pwc_tcp_launch_gpu*.sh` and worker logs `/tmp/pwc_tcp_worker_*.log`.
   - Removing any of them is the owner's call.
4. **RANGE-MINER constant skip range (F-1), next.** The miner's constant phases ignore the trial's skip range and use
   the worker default (`miner/range_miner_worker.py:1293`). This comparison matched only because the trial was pinned
   to `[0,16]`, which equals that default. The skip-range switch is the next item.
5. **Landed and open review state:**
   - **Landed in `ab2d330`:** the Beta X-PWC-SHUTDOWN R2 fixes (runbook and tests) and `tests/test_extract_pwc_bundle.py`'s
     tempfile scratch.
   - **Uncommitted at the time of writing:** the R3 A4b fix (§5).

## Owner's words this session (PST, verbatim)

| Time | Words | Context |
|---|---|---|
| 11:34 | "leave the rigs up- we dont want any more issues- proceed with the work" | After the redeploy |
| 12:21 | "Proceed" | Send the Stage 4 tooling to Beta |
| 13:12 | "yes" | Land the tooling without a second Beta review |
| 14:07 | "Shouldn't we address the issues you found first?" | Fix before the write-up |
| 14:39 | "Ok if you believe that's what needs done" | Send the shutdown fix to Beta |
| 15:44 | "Yes, once again do whatever you require to run step 4" | Land `b9b990c` |
| 16:02 | "the evidence is convincing enough for me- lets move on-" | Accept the same-counts evidence |
