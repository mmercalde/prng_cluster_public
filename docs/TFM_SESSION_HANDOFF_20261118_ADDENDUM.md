# TFM session handoff addendum: 2026-11-18 afternoon and evening

Written by Claude (Team Alpha lead). Adds to `docs/TFM_SESSION_HANDOFF_20261118.md`. All times PST. Evidence folders are on VM101, outside the repo. Anything not checked is marked UNVERIFIED.

## Result

**The v7 launcher (commit `9034e18`, fresh saved programs per run) ran clean twice on real hardware, and both halves of the seed space are covered under the Run 4 pins.**

| Run | Range | Launch | Outcome |
|---|---|---|---|
| I | lower | 16:02 | FAILED. rrig6600 gpu6 (0000:16:00.0) stopped making progress in phase 3. Cause unexplained. |
| J | lower | after I | FAILED. `staging_capacity_timeout`. Caused by Alpha (see below). |
| K | upper | after J | FAILED. Same cause as J. |
| L | upper | 18:42 | CLEAN. 128/128, search complete, phase 3 = 60, phase 4 = 23,568. |
| M (1st try) | lower | 19:01 | NOT STARTED. Launcher refused: 1 of 24 rig workers failed at dispatch. |
| M (2nd try) | lower | 19:11 | CLEAN. 128/128, search complete, phase 4 = 23,410. |

Runs L and M: no fault lines on the kernel recorder, no stalls, zero ledger slots held afterwards, and every rig card compiled the same 8 programs (192 files, 8 distinct checksums).

Search records: L = `s1s-e3e204e7aebac6572aa06b8ae3c74a747e48389541abbd408e0e6a4aa3860720`, M = `s1s-0aa6718b7cdba77018fb833443f69e0f3606bd7ea79b2da1e891732a9f5770e1`. I, J and K were closed uncredited.

Correction to the morning handoff: the lower-half phase-4 count is 23,410 (measured in Run M), not 23,515.

## What went wrong, by cause

1. **Run I, unexplained.** The worker on rrig6600 gpu6 sat at about 80 percent CPU with no system calls while its heartbeats kept arriving, so its lease never expired. When it was killed the kernel logged 49 lines of `amdgpu 0000:16:00.0: timeout waiting for kiq fence`. All 24 rig workers had byte-identical compiled programs, identical to the 2026-11-17 clean runs. The same seed range (stripe st2_s5, start 335544320) completed on Zeus's 3080 Ti in 18 seconds. The kernel recorder was not armed for Run I. It did not recur in J, K, L or M.
2. **Runs J and K, Alpha's error.** Alpha stopped Run I's coordinator with SIGTERM. That left 3,554 reservations marked `held` in the staging ledger. The admission count has no run filter (`miner/range_miner_coordinator.py:1514-1522`), so the next runs ran out of the 6,528-file budget and timed out after 600 s. Alpha also launched K before the diagnosis was known.
3. **Run M first try, unexplained.** The rrig6600 gpu1 worker died at `import cupy` with `UnicodeDecodeError: 'utf-8' codec can't decode byte 0x8d in position 384`, before touching the card. The same import then passed 12 of 12 on that rig. The launcher's fleet check refused to start, as designed.

Items 1 and 3 are both on rig-6600, on different cards, hours apart. That is two events, not a pattern yet. The audit agent found no path by which the v7 change causes either.

## Ledger hand release (owner approved)

At 18:34:56, after two agents and Beta (X-LEDGER-RELEASE R1, no objection) reviewed it: one statement set Run I's 3,554 `held` rows to `released`. Backup with checksums: `/home/michael/miner_ledger_backup_20261118_1834/`. Run I's staged files moved to `/home/michael/miner_forensic_20261118_runI/staged`. Run I's trial, stripe and shard rows were left as crash evidence.

## Rules added today (binding on Alpha)

- **Never SIGTERM or kill a live coordinator.** Let it end itself.
- The live ledger at `/home/michael/miner_staging/` is off-limits except for what Michael approves each time.
- The launch line now checks, before starting: three rigs ready with the kernel recorder armed, parameters unchanged, clean tree, ledger held count 0, no leftover miner processes.
- Reports to Michael: one topic at a time, short.

## Gaps in the pipeline exposed today (not fixed; coordinator code needs Beta review and Michael's yes)

- Nothing sweeps a dead run's held reservations when a coordinator starts.
- No held-count check in the repo before launch (today's check is in Alpha's launch line only).
- No stall guard for a worker that heartbeats but makes no progress.
- No graceful stop command for a live run.

## Tools added today (outside the repo)

- `/home/michael/stall_watch.py`: captures a `py-spy` dump and `dmesg` if a stripe goes quiet for 150 s.
- `py-spy` 0.4.3 in `/home/michael/rocm_env` in the three rig containers (Michael approved).
- `rigpower.py off <rig>` (refuses while the rig answers on the network).

## Evidence (VM101)

`/home/michael/runI_evidence_20261118` through `runM2_evidence_20261118` (the last holds the refused first try as well), and `/home/michael/rig_save_20261118` (per rig: compiled programs from every run today, worker logs, kernel and system logs, taken before shutdown).

## State at close

- Rigs shut down cleanly at 19:31; outlets rig-6600, rig-6600b, rig-6600c OFF at 19:32. GPU PSUs left on (hand-switched). Zeus and VM101 up.
- main = `5605ed3` before this file. Tree clean.

## Open

- PWC fresh-cache patch: Beta X-PWC-CACHE-RESET R1 says not landable (one blocking point on the TCP path). Michael's decision. Draft is in the `rig-gpu-fault` worktree, not landed.
- Full-range single run: needs Michael to raise the survivor cap (about 47,000 per phase expected).
- rig-6600's two one-off events; the original phase-4 fault mechanism.
- The four pipeline gaps above.
- Not landed: X-PWC-CACHE-RESET and X-LEDGER-RELEASE exchange files, `logs/ledger_sim/`.
