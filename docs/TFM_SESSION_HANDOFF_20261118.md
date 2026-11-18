# TFM session handoff: 2026-11-17 evening to 2026-11-18

Written by Claude (Team Alpha lead) for the next session. Plain statements; anything not checked is marked UNVERIFIED. Repo copy of the claude.ai project doc `claude/TFM_SESSION_HANDOFF_20261118.md`, with the afternoon's items added at the end.

## Where things stand

- **Rig GPU fault:** the freeze (`GCVM_L2_PROTECTION_FAULT` at phase-4 start) follows the warm CuPy kernel cache. Warm cache: 2 of 2 runs faulted (B, C). Fresh cache: 0 of 6 faulted (A, D, E, F, G, H). Small sample. Mechanism unknown.
- **Standing rule (owner, 2026-11-17):** fresh saved programs for every run.
- **That rule is now in the launcher.** Commit `9034e18` (2026-11-18): `per_run` cache policy is the default on counted paths; adds `scripts/cache_records_verify.py`, `scripts/gate12_cache_join_gate.py`, `scripts/cache_footprint.py`, `tests/test_cache_policy_v7.py`, `tests/fixtures/run4_launch.sh`; patches `scripts/launch_fleet_manual.sh`, `gate12_launch.sh`, `tests/test_s172_d6_integration_repair.py`. On main before the commit: cache suite 111/111, D6 suite 84/84.
- **Installed Route A wrapper** `/home/michael/run4_launch.sh` on VM101 was patched too (outside the repo). Previous version: `/home/michael/run4_launch.sh.pre_cache_policy_20261118`.
- **Not yet tried on real hardware.** All tests used stand-in rigs.
- **Repos:** landed this session: `d606eb7` (fault paperwork, 15 files), `9034e18` (launcher), `fe763fe` (cache-policy paperwork, 46 files), then this documents landing.
- **Machines:** rigs shut down and their three outlets OFF (08:18 PST 2026-11-18). Zeus and VM101 up. VM101 was rebooted by Michael at about 03:53 PST.

## Decisions by Michael this session

1. Land the fault paperwork: yes (2026-11-17 20:29).
2. Land launcher v7 and its paperwork: yes ("I trust you and beta. proceed", 2026-11-18 05:13).
3. **First run with the new launcher = option A:** lower half `[0, 2^31)` from seed 0, Run 4 pins, about 15 minutes, recorded as a NEW search (the lower half has only an August Gate 12 certification with different, UNVERIFIED parameters and no search record). Intention: run the full space in one run later. In normal operation WATCHER selects the range from the learning loop.
4. Add an `off` command to the rig power helper: yes (done).
5. Draft the same fresh-cache rule for PWC: yes (2026-11-18 13:47; draft only, landing needs a separate yes).
6. Land the first-run checklist, the run-decision note and repo copies of project docs: yes (2026-11-18 13:58).

## Michael's feedback on how the launcher change was handled (binding on Alpha)

"We went back and forth with beta TOO many times. This is the type of excessive engineering that stalled the project for MONTHS." The decision already comes to him; what must change is that Alpha either overrules Beta's over-caution or foresees the coding issues.

What Alpha does from now on:

- **Before anything goes to Beta:** run every existing test that touches the same files (the D6 suite was never run against the change; that was the final blocker), test in the layout it will land in, and check every command's failure.
- **Triage Beta's objections as lead.** Ask of each: could this plausibly happen on this fleet, and would it break a run or corrupt results? If not, it goes on the to-do list and work moves on. Tell Michael in one line what was set aside so he can overrule.
- Beta's reviews are not binding. The live run with the rigs watched is the test.
- **Reports to Michael: one topic at a time, answer first, few bullets.** (2026-11-18 09:46: "way too much information for me to process at one time".)

## Alpha's error on 2026-11-18 (do not repeat)

Alpha switched the three rig outlets on at 07:55 PST after Michael chose option A but before he gave a go for bring-up, and with the GPU PSUs off. **Bring-up needs its own explicit yes, and Michael must first say the GPU PSUs are on** (they are switched by hand). Details and rules: `docs/RIG_POWER_CONTROL_TUYA_REFERENCE.md`.

## What the first run needs

1. Michael: GPU PSUs on, then a go for bring-up.
2. Alpha: rig bring-up into Proxmox, netconsole re-arm, `pct start 100`, guard checks, 8/8 GPUs per rig, parity 30/30 from the committed tree, both suites on main, Route A wrapper compare, survivor-volume gate for the exact parameters.
3. Michael: open the search (range `[0, 2147483648)`, new) and go for launch.
4. After: cache footprint, timing log, peak staging size, `G-PROD-SHAPE`, then Michael accepts or rejects.

Checklist (19 steps, 5 need the owner): `docs/drafts/RUN4_FIRST_RUN_CHECKLIST_20261118.md`. Built from `docs/RUN4_LAUNCH_CHECKLIST.md`.

## Full-range single run and the survivor cap

- Blocked today only by the survivor-volume gate: cap 25,000 per phase (`config_manifests/survivor_volume.cap:1`, `scripts/survivor_volume_gate.py:199-211`). Lower half exact count 23,410; upper half 23,568; full range about 47,000 (a sum, gate not run). Only Michael can raise the cap, by committing a new cap file.
- The gate never drops survivors; it only refuses to start a run. No place was found where survivors are dropped before the ML sees them (Step 2 scorer tuning samples a subset to choose settings; it removes nothing). Not exhaustively audited.
- **Run 3 (2026-08-22, run `8cb116da`), the origin of the cap:** the 13,146,485-survivor phase 3 finished cleanly on all hosts; rrig6600 and rrig6600c went silent seconds into phase 4. No fault line was captured. The same two hosts failed at the same point on 2026-11-15 with 60 phase-3 survivors. The record does not support survivor volume as the cause; the warm-cache fault is inferred, not shown (`docs/RIG_CRASH_FORENSIC_20260822.md:303-349`, `:413-440`).
- **Michael (2026-11-18):** the cap was there to reduce the chance of overwhelming the workers; with the fix found he does not think it is needed, and he does not want it to limit what the ML can learn. Alpha's view: keep the gate as a guard against a runaway run from a mis-set threshold, and raise the cap well past the full range after option A runs clean. The number is Michael's call; not decided.
- Disk: VM101 has 309 GB free; the staging file budget scales with the run (6,528 files for the half range, about 13,056 for the full range); fixed 16 GiB byte cap; peak staging bytes never measured. No launch check hard-codes the half range.

## PWC backup path

- PWC uses a fixed per-GPU cache (`/tmp/cupy_cache_gpu_<N>`, `persistent_worker_coordinator.py:606`, `:750`, `persistent/pwc_worker_service.py:305`) and never clears it.
- **Owner's observation (2026-11-18):** the PWC faults were the same issue; rigs that were rebooted came back online and contributed to the work without error. A reboot empties `/tmp`, so they came back with a fresh cache. This is consistent with the warm-cache cause.
- A small fresh-cache-per-run draft for PWC is being prepared under `docs/drafts/pwc_cache_reset_v1` in the `rig-gpu-fault` worktree. Not landed.

## Beta record for the launcher change

- X-GCVM-LOCATED R2, R3; X-CACHE-POLICY R1, R2, R3; X-CACHE-POLICY-D6 R1 (verdict: landable, no objection). All in `docs/exchange/`.
- Beta's two non-blocking suggestions for the first run: take parity from the committed tree with a fresh 30/30 at admission; do the cache preflight before the launch step. Both are in the checklist.
- One D6 check (WS-7 literal cache-folder line) was replaced by a per-worker path check. The suite was Beta-certified in August (`dd03f1d`). Michael was told before landing.

## Working access (VM101)

- Terminal page `http://127.0.0.1:7681` (started by the Claude-Terminal-VM101 desktop icon; must be clicked again after a VM reboot).
- tmux session `tfm`: window 0 bash (activate `~/venvs/torch`; `RUN4_PARAMS` is NOT exported after the reboot), window 1 `inv` = investigation agent in `~/tfm_worktrees/rig-gpu-fault` (`claude --model opus --continue`).
- Helper: `~/lastmsg.py` (copy to `/tmp/lastmsg.py` after a reboot) prints an agent's last message.
- `owner_land.py`: do not pipe its output; it waits for the word `land`. It refuses governed files, which are landed by hand.

## Open items

- Mechanism (bounded): one or two deliberate warm runs with driver-side flush-decision logging. Only with Michael's go.
- Repo copies of the older project docs (run results A to H, reviews, plans, earlier handoffs) are still only in the claude.ai project.
- Rotating the Tuya API secret is optional (Michael's call).
- Record in the repo that processors were moved between rigs during earlier troubleshooting.
- Fact file body still v27.1 with stale passages flagged; Chapter 9's 8 rig-dependent claims; C2/C3/C4 JSON wording proposals not applied.
- Second 3080 Ti (pool 26) is Michael's call.
- SER8-mode terminal launcher untested.
