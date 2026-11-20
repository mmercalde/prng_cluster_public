# TFM session handoff: 2026-11-19

Written by Claude (Team Alpha lead) for the next session. All times PST. Evidence folders are on VM101, outside the repo. Anything not checked by Alpha is marked UNVERIFIED or attributed to the agent that reported it.

## Bottom line

- **Half-range runs on the v7 launcher: four clean in a row** (L and M on 2026-11-18, P and Q tonight). Counts are identical night to night.
- **The first full-range single run did not complete.** Two attempts stopped for two different reasons: a card dropped off the PCIe bus on rig-6600b (Run N), and the coordinator's intake queue overflowed in phase 3 (Run O).
- **PWC fresh-cache patch landed** (`7444aea`) after Beta's round 2 gave no objection.
- **VM101 now starts headless.**

## Runs today

| Run | Range | Launch | Outcome |
|---|---|---|---|
| N | full [0, 2^32) | 17:11 | FAILED 17:20:52, `compute_lease_expiry`. rrig6600b gpu6 (0000:16:00.0, root port 00:1d.0) dropped off the bus in phase 1, stripe st0_s56. 56 of 256 done. |
| O (1st try) | full | 18:32 | NOT STARTED. Sentinel gate refused: rrig6600c gpu4-7 wrote their ready line about 27 s after the others. No fault. |
| O | full | 18:40 | FAILED 18:51:17, `inbound_saturation_timeout` in phase 3. 146 of 256 done. No GPU fault, rigs healthy. |
| P | lower half | 18:57 | CLEAN 19:12. 128/128, phase 3 = 59, phase 4 = 23,410. |
| Q | upper half | 19:13 | CLEAN 19:28. 128/128, phase 3 = 60, phase 4 = 23,568. |

P and Q: no fault lines, no stalls, 8 cards per rig throughout, ledger held 0 afterwards, 192 compiled files with 8 distinct checksums each run, both search records `search_complete`.

Search records: N `s1s-58a53c37...` and O `s1s-b592c691...` closed uncredited; P `s1s-caa313ef2901e37201e21260912a2534b9d9e3737cda2ff6bdc85581d1f9e0fd`; Q `s1s-735cf59f...` (full id in `--step1-search-status`).

Both half runs ended with 0 seeds in the two-way (bidirectional) survivor file, as on 2026-11-18. The phase counts match the record. What the record says this means for Step 2 has NOT been looked up yet.

## Landed today (main = `7c725c8` on origin and public, tree clean)

- `7444aea`: `persistent_worker_coordinator.py` (PWC clears the rig cache before worker start, 49 lines added), `tests/test_pwc_cache_reset.py` (21/21), `docs/drafts/pwc_cache_reset_v1/` (README with the known TCP reconnect limit, diff), Beta records X-PWC-CACHE-RESET R1 and R2 and X-LEDGER-RELEASE R1. Six existing suites gave identical results before and after. The file is not in the rig parity set, so no redeploy.
- `7c725c8`: `config_manifests/survivor_volume.cap`, 25,000 to 50,000 per phase (owner decision). Gate test 21/21. The survivor-volume gate for the full range said PROCEED, projection 47,584.

## Owner decisions today

1. Land the PWC patch and Beta paperwork: yes.
2. VM101 headless option A (private password file): yes.
3. Full-range run with cap 50,000 and a full-range wrapper: yes.
4. Reboot the hung rig and rerun; then run both halves again "to put another run in the bank".
5. Make PWC a fully functional backup: scope it with agents (done, see below). Four rulings still needed from him.
6. Rigs stay up at session end (not shut down).

## Run N: what the logs show (rrig6600b gpu6)

- Per-run cache was active: `rig_cupy_cache_policy=per_run`, cache-join 24/24, fresh nonce folders, gpu6 compiled 5 programs at 17:12-17:13.
- 17:14:11 kernel: `[gfxhub] page fault`, `GCVM_L2_PROTECTION_FAULT_STATUS:0x00801231`, client `SQC (inst)`, python pid 9399. The card had completed two stripes normally.
- Worker log: `Memory access fault by GPU node-7 ... Page not present or supervisor privilege`, `GPU core dump failed`.
- Same second: `snd_hda_intel 0000:16:00.1 ... device inaccessible`; later fault-status reads `0xFFFFFFFF`.
- 17:14:20: `device lost from bus!`, `GPU reset end with ret = -19`, repeated; then sdma0 ring timeouts, hung kernel tasks at 17:18, `kiq fence` lines.
- Read after the fault, before any reboot: root port 00:1d.0 reported `PresDet-`, link down; 14:00.0, 15:00.0 and 16:00.0 all unreadable (`Unknown header type 7f`). The whole card was off the bus.
- Agent analysis (`docs/drafts/NOTE_RUN_N_GPU6_EVENT_20261119.md`, rig-gpu-fault worktree, not re-checked by Alpha): nothing logged shows the link going away before the fault; the first all-ones read came at most 98 ms after it. The first fault line matches the 2026-11-15 faults on rrig6600c 16:00.0 field by field. Differences: one card on one rig, phase 1, host kept running, fresh cache. Runs B and C began with different fault codes. Run I (2026-11-18) was a different kind of event (no fault line). 86 recorded fault bursts tabulated; 23 began with `0x00801231`, 20 of them on rrig6600c 16:00.0.
- Limits: the kernels boot with `pci=noaer pcie_aspm=off`, and there is no sub-second power data, so a link or power dip would not have been logged.
- Owner's position: this is NOT the old error coming back; there are multiple clean runs on the patch, and the circumstances differ (phase 1, not phase-4 start). Alpha's earlier statements "no phase-4 type fault lines" and "the error is back" were both wrong in different directions and were corrected in chat.
- Owner's observation at the rig: one GPU on rig-6600b had its fans at 100 percent. In the past that happened when a rig was overwhelmed with work or with SSH. Alpha found no extra load or SSH before the fault compared with the clean runs.

### After the fault on rig-6600b

Clean reboot hung (pings, SSH refused) and needed an outlet cycle. It then came up with 7 cards: the card on root port 00:1d.1 was not detected and the port did not enumerate. Two more outlet cycles did not start the rig. It started after the owner attended it, with 8 cards. Which card had the fans at 100 percent was not established.

## PCIe link survey (new evidence, all 24 slots, read-only)

| Rig | Width | Speed now | Riser error flags since boot (first survey, 17:51) |
|---|---|---|---|
| rig-6600 | x1 | 5 GT/s (Gen2), port cap 5 | 5 of 8 clean; 06:00.0 and 16:00.0 Rollover+Timeout; 10:00.0 Timeout |
| rig-6600b | x1 | 8 GT/s (Gen3), port cap 8 | Timeout on most; 10:00.0 BadDLLP+Rollover+Timeout; 16:00.0 gone |
| rig-6600c | x1 | 8 GT/s (Gen3), port cap 8 | Timeout on 6; BadDLLP on 1 |

After P and Q (19:30): 24 of 24 present; 9 slots clean, 12 Timeout, 2 Rollover+Timeout, 1 BadTLP+BadDLLP.

- The owner expected the BIOS to hold the links to Gen1 or Gen2. rig-6600b and rig-6600c are running Gen3.
- 0000:16:00.0 is root port 00:1d.0 on rig-6600 and rig-6600b, but 00:1d.1 on rig-6600c, so "same address" is not the same physical port on rig-6600c.
- Alpha's suggestion, owner's call: set Gen1 or Gen2 in the BIOS on rig-6600b and rig-6600c, rerun, and read the flags again. He cannot do it tonight. Gen2 is not a guaranteed cure: rig-6600 is on Gen2 and had the Run I hang.
- Script: `/tmp/pcie_survey.sh` on VM101 (lost at reboot; text is in the evidence folders' outputs and can be rebuilt from this description: per GPU, root port LnkCap/LnkSta/PresDet and the riser upstream port's CESta flags).

## Run O: full-range intake overflow

`TRIAL TERMINAL class=inbound_saturation_timeout`: local ingress did not drain within 180 s, `inbound_qsize=1024/1024`, first on rrig6600c:gpu4, then 8 emergency terminal requests. The later `MinerIngressError ... without a VALIDATED threshold provenance record` is the follow-on symptom of the failed trial.

Agent analysis (`docs/drafts/NOTE_FULL_RANGE_INBOUND_SATURATION_20261119.md`, rig-gpu-fault worktree, read-only, nothing tested): in phase 3 the main loop slowed from about 40 to about 2.5 sub-results per second while the GPUs kept sending at 40. It spent 476 of 572 s waiting on one lock that four staging threads hold while reading the whole backlog, twice per staging job. Half runs slow the same way, but only 450-570 results arrive after the slowdown and the 1,024 queue holds them; at full range about 2,700 arrive. A bigger queue alone risks lease expiry because heartbeats share the queue. Options in order: (0) run the full range as two half runs; (1) measure the lock on a CPU harness; (2) one ledger read connection per thread; (3) size the queue from the stage's sub-result count; (4) raise the hybrid seed cap. Options 2-4 need Beta and the owner; part of it touches R-1, which the record says not to reopen.

Alpha's recommendation: option 0 for now.

## PWC

- Beta X-PWC-CACHE-RESET R2 (real-world framing, as the owner asked): landable as is, no objection. Landed.
- The ten failing checks in three old scripts are stale tests, not code faults: `docs/drafts/NOTE_PWC_OLD_TEST_FAILURES_20261119.md` (rig-gpu-fault worktree).
- PWC refuses variable-skip runs on purpose since 2026-07-31 (`persistent_worker_coordinator.py:158-218`): it filtered at a default 0.50 instead of the trial thresholds. The second reason given there (kernels ignore `skip_min`/`skip_max`) applies equally to RANGE-MINER: 0 of 22 hybrid kernels take skip bounds (`docs/BACKLOG.md:79`).
- Scope for a full backup: `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md` (pwc-backup worktree, agent, not re-checked). Output contract already the same; no kernel change and no rig-file change needed; new finding that PWC stops a Run 4 style trial after phase 1 (0 survivors there ends the trial); seed-start refusal and offset-0 limit; step-1 work of 2026-11-14 still not on main. Four stages, about 5-7 sessions plus Beta, then 2-3 rig sessions.
- **Four rulings needed from the owner:** lift the variable-skip block on the threshold fix alone; let PWC runs earn search credit; drop the seed-start refusal for PWC; choose the phase rule (always four phases, or stop early).
- The skip-range question (should the 6-99 range steer variable-skip phases at all) is the owner's design ruling and affects both paths.

## VM101 headless

Full note: `claude/VM101_HEADLESS_STARTUP_20261119.md` (project). Cause: automatic desktop login leaves the GNOME keyring locked and Chrome waits behind the unlock box. Fix: `~/.config/autostart/claude-vm101.desktop` runs `~/bin/vm101_startup.sh`, which unlocks the keyring with `~/bin/kr_unlock.py` from `~/.config/kr_pass` (mode 600, created by the owner), starts the web terminal and opens Chrome. Proven by a bare reboot at 13:01. **To do (owner):** `qm set 101 --onboot 1` on the Proxmox host.

## Files outside the repo created today (VM101)

- `/home/michael/run4_full_launch.sh`: copy of `run4_launch.sh` with only `max_seeds` 4294967296. The half-range wrapper is unchanged.
- `/home/michael/runN_full_params.json` (volume-gate parameters, full range).
- Evidence: `runN_evidence_20261119`, `runO_evidence_20261119`, `runO2_evidence_20261119`, `runP_evidence_20261119`, `runQ_evidence_20261119` under `/home/michael/`. Run N's folder holds the kernel logs, recorder trace, worker log tail, stall captures, `lspci` and PCIe flag reads.
- Launch guard now also requires 8 GPUs per rig.

## State at close

- Rigs UP in Proxmox with CT100 running, recorder armed, nothing running. NOT shut down, on the owner's instruction. GPU PSUs on.
- Ledger held 0. Repo tree clean. Both agents idle (tmux windows `inv` and `pwc`).

## Not landed (need the owner's yes)

- rig-gpu-fault worktree: `NOTE_PWC_OLD_TEST_FAILURES_20261119.md`, `NOTE_RUN_N_GPU6_EVENT_20261119.md`, `NOTE_FULL_RANGE_INBOUND_SATURATION_20261119.md`.
- pwc-backup worktree: `SCOPE_PWC_FULL_BACKUP_20261119.md`.
- This handoff; repo copies of the project notes on VM101 headless start-up and the selfplay NN status.

## Open

- BIOS link speed on rig-6600b and rig-6600c (owner).
- Acceptance of the clean runs: cache footprint, timing log, `G-PROD-SHAPE` (owner-initiated).
- Full-range single run: options above.
- PWC: four rulings, then staged work.
- What 0 bidirectional seeds means for Step 2: look up the record.
- Pipeline gaps from 2026-11-18 (dead-run reservation sweep, held check in repo, stall guard, graceful stop); the sentinel-gate timing refusal seen tonight.
- Selfplay NN: `claude/TFM_SELFPLAY_NN_STATUS_20261119.md` (project), saved for later.
