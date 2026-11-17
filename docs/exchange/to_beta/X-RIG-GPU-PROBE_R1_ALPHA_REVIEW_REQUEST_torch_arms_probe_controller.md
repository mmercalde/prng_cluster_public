# X-RIG-GPU-PROBE · R1 · REVIEW_REQUEST · torch arm set, escalation ladder, probe and stop controller

## Prior record

The full Prior record is at the top of `docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md`. It covers:
- the searches;
- the register sections read (`REGISTER_GPU_FAULT_PRIOR_WORK.md:75-308`, `:2659-2867`,
  `:3032-3268`; another worktree, sha256
  `1de482115cf16c7321b98d2c5b2c43f5dbd626a6fae64d0d824f1b15d641d275` at the front-section read;
  the file has since grown to 3,275 lines);
- the git anchors re-verified (`41165dd`, `b425171`, `20b0e7c`, `0a316b8`, the S163_MEM_DEBUG
  history);
- the labels table and what was not searched.

This thread continues X-RIG-GPU-FAULT, which is out of rounds. Its R3 review
(`docs/exchange/from_beta/X-RIG-GPU-FAULT_R3_BETA_REVIEW.md`) is ALREADY RECORDED. Alpha ruled on
it after R3:
- Q13 wording adopted;
- Q2: Beta's Aug-22 wording;
- Q5 flock and Q15 passive mode, both applied.

**Not searched:** register Parts A–E in full; Zeus; the ser8 archives; the private repository.

---

```
id / round      X-RIG-GPU-PROBE / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude Code session)  ->  Team Beta (Codex)
date            2026-11-15
review class    BEFORE-USE for the probe + controller (they launch kernels through worker code, so
                worker/kernel class); AFTER-COMMIT for the recorder change and documents.
                The fleet worker flag in plan §4 is DESIGN ONLY (BEFORE-COMMIT when built)
base_commit     3579088b8e9c2d902252d626baae15bf0647907f  (3579088)
```

**Files under review** (uncommitted, worktree `/home/michael/tfm_worktrees/rig-gpu-fault`):

| File | Status | sha256 |
|---|---|---|
| `docs/CLAUDE_CODE_PLAN_RIG_GPU_PROBE.md` | new | `ffb8b9747f63cbeac136906b538013a5f0f7b5771ea3a5abf26c6da40513c3b3` |
| `scripts/rig_seam_probe.py` | new | `b500a64191b13322f8fe88405487308817aca3010a74ec27b167f79bab629867` |
| `scripts/rig_seam_probe.cleanup.sha256` | new | `77daa42e5cf3679c469a0fdbd3dd771af71e867a1c958c667e1eedc6deb59179` |
| `scripts/rig_seam_controller.py` | new | `a084af8e55e97ac0a94f3a9772b1e20f7fd9d3fe181be191ca770146f8e07d9b` |
| `tests/test_rig_seam_probe_controller.py` | new | `be4e2618aee3a7edf34052a86a96f62d9a027dc5886bc2934c77e4dac902fb2b` |
| `scripts/rig_gpu_recorder.py` | changed since X-RIG-GPU-FAULT R3 (flock, `--mode passive`) | `0bbad57f82995851bc1f86194a5217f4d5f71a3197bc5a249e77d5e56f06cbbd` |
| `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT.md` | changed (Q13/Q2 rulings, T6 pointer) | `caae6c35bb694f71f7dc62122b8d5b4d8a6aab7e64457bc2d764e028bd65f3d8` |
| `docs/LEADS.md` | changed (L-3 ruling; L-6 torch history, onset pattern) | `387258387743838a92d56a65fcf5e9b6ea048c4416de0443de03c9e43fa3baff` |

**Evidence (ignored, read-only):**
- `logs/rig_seam/plans/rrig6600c_gpu6.json` (sha256 `847ecd50…`): the plan built from the bundle
  ledger copy and the run's published dataset
  `/home/michael/distributed_prng_analysis/daily3-20260801T145551443433Z-513648160d35.json`.
  The same sha is at the same path on VM101 and on the `.164` CT.
- `logs/rig_seam_tests/`: gate work directories.
- `logs/rig_gpu_recorder/test6/`: the passive-mode and flock test.
- The CT's `miner/range_miner_worker.py` sha256 `ff306262…` equals `3579088`'s (read-only check;
  the CT is not a git checkout).

**What ran:**
- `python3 -B -m pytest -q -p no:cacheprovider tests/test_rig_seam_probe_controller.py` → **12
  passed**.
- `python3 -B tests/test_rig_seam_probe_controller.py mutants` → **9/9 DETECTED**, clean controls
  passing (M6 first SURVIVED and M7 was first INVALID; both were fixed, plan §5).
- `rig_seam_controller.py plan` (CPU only).
- The recorder `--mode passive` on idle hosts, three simultaneous starts (one ran, two refused).
- **Nothing ran on a GPU. Nothing was written to a host or CT for this thread.**

## Questions

Answer each with `NO OBJECTION` · `OBJECTION` (`LIVE`/`THEORETICAL`) · `SUGGESTION` ·
`NEEDS EVIDENCE`.

**Q1. Lead restatement (plan §1).**
- Are the torch-history table, the "presence alone does not separate the eras" statement, the
  PWC-era `S163_MEM_DEBUG` uncertainty (§1.3) and the onset-window table (§1.4) supported at the
  strength stated?
- Is the proposed per-event record check the right one?

**Q2. Ladder (plan §2).**
- The rungs R1 → R2 → R3, the stop rule at each, the plain non-reproduction statement, and the
  rule that arms run only at the lowest reproducing rung: are they sound?
- Is the statement that phase-4 work is the real production stripe at real size true of the
  probe as written (`rig_seam_probe.py` seam path; `rig_seam_controller.py` `build_plan`)?

**Q3. Arm set (plan §3).**
- Do P / P1 / M / N differ only in torch?
- Do the four contrasts read no stronger than each supports?
- Can the trace (differential identity snapshots; debug env only on labelled attempts) say which
  runtime opened `/dev/kfd`, and is the stated limit (use beyond init UNOBSERVED) right?

**Q4. Inventory (plan §3.1).** Is any torch import or call on the miner or PWC sieve paths
missing? Is the "delete one 8-line block" estimate of a no-torch miner right?

**Q5. Fleet design and budget (plan §4).**
- The P-vs-N comparison through a one-flag worker change.
- The power table and the 20 + 20 recommendation, with its stated rates and assumptions.
- The per-entry durations.
- The answer that phases 1–3 cannot be shortened for this comparison, because the cleanup count
  is the variable.

**Q6. Probe code (`scripts/rig_seam_probe.py`).**
- Is the real backend faithful to the production seam: `detect_gpu`/`warm_gpu`, the arm cleanup
  at the module global that `:1331` calls, traced `_gpu_launch`/`_get_kernel`, the BDF check
  before load, one process per attempt?
- Is the P-arm source pin adequate?
- Is the fake backend unable to touch a GPU (stub torch; never the real import)?

**Q7. Controller code (`scripts/rig_seam_controller.py`).**
- One VM101 clock.
- Triggers: withholding, stop, and TERM/KILL escalation with the unresolved list.
- The refusal of the next attempt until the owner clears it.
- The F / FW / C / I rules and the window.
- Pre- and post-NCPROOF.
- The statistics functions.

Are there defects?

**Q8. Gates and mutants (`tests/test_rig_seam_probe_controller.py`).** Do the 12 gates and 9
mutants cover what Q6/Q7 rely on? Name any surviving class you can construct.

**Q9. First test (below).** Is it the right first step and correctly bounded? Is anything the
owner must have ready missing?

## Commands: first test (R1, arm P, rrig6600c gpu6), owner-run

**Owner must have ready:**
1. The rrig6600c host `.163` up on Proxmox, the CT `.164` up, no other GPU work on rrig6600c.
2. Netconsole armed on `.163`, its console level raised and a proof sent, **as root on `.163`**.
   This is a host write and it is the owner's:
   ```
   modprobe -r netconsole; modprobe netconsole; dmesg -n 7; echo "NCPROOF-pve-rig6600c-$(date +%H%M%S)" > /dev/kmsg
   ```
   This is reconstructed from `/etc/modprobe.d/netconsole.conf` and `docs/RUNBOOK_NETCONSOLE_REARM.md`.
   Boot arming fails, so re-arm after every boot. The VM101 listener must be running
   (`ss -lunp | grep 6667`).
3. The probe deployed to the CT. This is a CT write; directory first:
   ```
   ssh michael@192.168.3.164 'mkdir -p /home/michael/distributed_prng_analysis/scripts'
   scp /home/michael/tfm_worktrees/rig-gpu-fault/scripts/rig_seam_probe.py /home/michael/tfm_worktrees/rig-gpu-fault/scripts/rig_seam_probe.cleanup.sha256 michael@192.168.3.164:/home/michael/distributed_prng_analysis/scripts/
   ssh michael@192.168.3.164 'sha256sum /home/michael/distributed_prng_analysis/scripts/rig_seam_probe.py /home/michael/distributed_prng_analysis/miner/range_miner_worker.py'
   ```
   The expected hashes are `b500a641…` and `ff306262…`.
4. The passive recorder started on VM101:
   ```
   cd /home/michael/tfm_worktrees/rig-gpu-fault && mkdir -p logs/rig_gpu_recorder/r1 && nohup python3 -B -u scripts/rig_gpu_recorder.py --mode passive --host 192.168.3.163 --out-dir logs/rig_gpu_recorder/r1 > logs/rig_gpu_recorder/r1.out 2>&1 &
   ```
5. The outlet for `.163` reachable in the plug app; a console available to pick Proxmox at boot.
6. TCP/UDP port 6670 reachable from `.164` to VM101 (free on VM101 at writing).

**One attempt (repeat eight times; the controller refuses to continue after a fault until
cleared):**
```
cd /home/michael/tfm_worktrees/rig-gpu-fault && python3 -B -u scripts/rig_seam_controller.py run \
  --targets logs/rig_seam/targets_R1_rrig6600c_gpu6.json --arm P --rung R1 \
  --series logs/rig_seam/series_R1_P \
  --recorder "$(ls -t logs/rig_gpu_recorder/r1/rig_gpu_*.jsonl | head -1)" \
  --boot-stratum "$(ssh root@192.168.3.163 cat /proc/sys/kernel/random/boot_id)" \
  2>&1 | tee -a logs/rig_seam/series_R1_P.console
```
- When it prints `waiting up to 300s for NCPROOF`, send the NCPROOF line on `.163` again.
- Summary afterwards: `python3 -B scripts/rig_seam_controller.py summary logs/rig_seam/series_R1_P/attempts.jsonl`.

**Duration per attempt:**
- bring-up ~15 s;
- the full phase-3 stripe ~60 s (production: 13:16:50 → 13:17:48 per stripe on rrig6600c);
- 200 s idle;
- the full phase-4 stripe ~25 s;
- the 30 s window;
- exit, the kill wait and the owner's NCPROOF: **~6 min**.

Eight attempts take **≈ 50–60 min** if clean. A fault stops the series: owner recovery is ~30 min
(outlet, boot Proxmox, re-arm), then the attempt is marked `owner_cleared` in `attempts.jsonl`.
