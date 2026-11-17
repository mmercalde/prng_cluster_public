# RUNBOOK — outside recorder, Run A / Run B on one boot (host picture vs card)

**For:** Alpha (delegated rig tests), 2026-11-16.
**Tool:** `scripts/rig_fault_recorder.py` (sha256 `d4c9f6ab…`). Gates:
`tests/test_rig_fault_recorder.py` (4/4 pass, CPU only).
**No worker code change.** It reads nothing on a GPU and makes no SMU read. It installs nothing.

## Prior record

**Searched:**
- `docs/LEADS.md` L-6, L-7;
- `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md` §4–§5;
- the register (`…/gpu-fault-history/docs/REGISTER_GPU_FAULT_PRIOR_WORK.md`) for "trace", "debugfs",
  "vm_info", "kfd": only netconsole and `dmesg` captures are recorded. No tracefs or debugfs
  memory-map capture exists in the record.

**Labels:**
- the recorder: NEW;
- the run-#2 pattern: ALREADY RECORDED (L-7, now with owner testimony);
- telemetry pause: owner-approved (Alpha, 2026-11-16).

**Not searched:** ser8, the private repository.

## What it needs on the hosts (`.121 .155 .163`)

- **Already present:** root SSH from VM101, mounted `debugfs` and `tracefs`, `bash`, `timeout`,
  `grep`.
- **Nothing is installed or written to disk.**
- **Host settings it switches on** (`python3 scripts/rig_fault_recorder.py plan` prints this
  list). All are tracefs only, all vanish on reboot, and they are switched on by
  `arm --i-have-go` only after your go:
  1. `trace_clock` → `mono`. The old value is saved in `/run/tfmrec_trace_saved` (tmpfs).
  2. `buffer_size_kb` → 8192 per CPU (≈48 MB of host RAM on a 6-CPU host). `trace` is cleared.
  3. Events on, 16 in all:
     - `amdkfd`: evict/restore start+end, schedule_evict_and_restore, map_memory_to_gpu start+end (7);
     - `amdgpu`: `bo_create`, `bo_move`, `vm_bo_map`, `vm_bo_unmap`, `pasid_allocated`,
       `pasid_freed`, `iv`, `vm_update_ptes`, `vm_flush` (9).
     - `--level base` drops `vm_update_ptes` and `vm_flush`.
  4. `tracing_on` → 1.
  5. One `trace_marker` line per snapshot (`--marker`), the shared clock anchor.
- **Never switched on:** `amdgpu_device_rreg/wreg`, `vm_set_ptes`, `vm_copy_ptes`, dynamic debug,
  module parameters (`debug_evictions` is writable, but it is not used).
- **Revert:** `disarm`.

**Telemetry pause** (owner-approved):
- The 1 Hz SMU reader is **`rig-power-telemetry.service`** (`/usr/local/sbin/rig-power-telemetry.sh`,
  reading hwmon `power1_average`/`temp1_input` every second). It is enabled at boot and has
  `Restart=always`.
- Pause after **every** boot, before T0: `systemctl stop rig-power-telemetry.service`. `stop` is
  honoured despite `Restart=always`. Resume: `systemctl start rig-power-telemetry.service`.
- It has no dependency in either direction on `amdgpu-powercap-min.service` (a oneshot that has
  already exited), so stopping one leaves the other untouched.
- It still runs from boot until it is stopped. `systemctl disable` would avoid that, but it is a
  persistent change, so I don't recommend it.

## What it records

| Need | Source | Points |
|---|---|---|
| (1) Per-process GPU mappings and memory per card | debugfs `dri/<bdf>/{amdgpu_vm_info, amdgpu_gem_info, clients, amdgpu_vram_mm, amdgpu_gtt_mm, amdgpu_sa_info, ttm_page_pool}`; sysfs `mem_info_{vram,vis_vram,gtt}_used`; `kfd/{mem_limit, rls}`; `/sys/class/kfd/kfd/proc/<pid>/**` (every file except `cu_occupancy`) | T0–T7, FAULT |
| (2) Every map/unmap/evict/restore, time-stamped, per worker (common_pid) | the 16 tracepoints → `stream`: one `ssh … cat trace_pipe` per host, each line VM101-stamped and flushed at once | continuous |
| (3) `/proc/PID/maps` (+ `status`, `smaps_rollup`) of every worker | host `/proc`, matched on `range_miner_worker.py` | T0–T7, FAULT |
| (4) Host memory | `meminfo`, `vmstat` (compaction, allocstall, pgscan/steal, THP, swap, oom), `buddyinfo`, THP settings, `swappiness`, `min_free_kbytes` | T0–T7, FAULT |
| (5) Running worker environment | `/proc/PID/environ` filtered to `HSA_ ROCR_ HIP_ CUPY_ AMD_ GPU_ ROCM_`, `PYTHONPATH`, `LD_LIBRARY_PATH` | T1–T6 |

**Points:**
- T0: manual, before launch.
- T1: the sentinel gate's PASS (fleet warmed and parked).
- T2–T4: phases 1–3 done (`ACTIVE_STRIPES` count 0 after the stage was active).
- T5: T4 + 45 s, in the idle gap; **skipped if phase 4 has already dispatched**.
- T6: phase 4 done.
- T7: no worker process left on any host.
- FAULT: a netconsole `page fault` line; all hosts are snapshotted, and a frozen host times out
  at 60 s.

**Idle sample, today 15:5x** (`logs/rig_fault_rec/idle_sample_20261116/`, 0.8 s and ~46 KB per
host):
- KFD `System mem used 1022M` of 6193/6207/6189 M on all three hosts, with no GPU process;
- TTM 0 M;
- per card `vram_used` 15.6 MB (`.163 03:00.0`: 17.6 MB);
- THP `madvise`; swap unused;
- all compaction and reclaim counters 0 since boot;
- `vm_info` and `gem_info` empty.

The `vm_info` parser counts are provisional until a sample with workers exists; the raw text is
always kept.

## Answers needed before the run

- **(5) Arch:** every cached program is `amdgcn-amd-amdhsa--gfx1032`, with no feature suffix:
  all 8 entries in yesterday's `.156` copy (`logs/rig_gpu_fault_20261115/offline_1908/ct_192.168.3.156.tar`).
  The run copies the env live.
- **(6) Where code and buffers live** (ROCm 6.4.3 source; GitHub `ROCR-Runtime` tag `rocm-6.4.3`
  `runtime/hsa-runtime/core/runtime/amd_loader_context.cpp`, `clr` `rocclr/utils/flags.hpp`):
  - **Kernel code object: card memory (VRAM).** For a BASE-profile agent, `SegmentAlloc` puts
    `CODE_AGENT` segments in `RegionMemory::AgentLocal` (device-local, coarse-grained,
    `AllocateExecutable`). The bytes are first written to a **host staging buffer**, then
    `Freeze()` copies them to VRAM: by CPU `memcpy` over the BAR plus `PcieWcFlush` when large BAR
    is on and the size is under `co_dmacopy_size`, otherwise by `DmaCopy`. Then
    `InvalidateCodeCaches`.
    - Large BAR is on for all 24 cards: `mem_info_vis_vram_total == vram_total`, measured today.
    - dGPU = BASE profile, and the `co_dmacopy_size` default, are **[UNVERIFIED]** in this file.
  - **Phase-4 buffers:** CuPy's default pool → `hipMalloc` → VRAM.
  - **Kernel arguments:** `HIP_FORCE_DEV_KERNARG` defaults to `true` → device memory.
  - **Host RAM pinned for the card:** H2D staging and the AQL queue and signals [inference].
  - **The run checks all of this:** `amdgpu_bo_create` events record each BO's domain with the
    creating worker's pid.

## Cost and perturbation

- **Tracepoints:** ~0.1–1 µs per event.
- **Expected rate:** hundreds to a few thousand events/s per host at phase load, dominated by
  `vm_update_ptes` and `vm_bo_map/unmap` (the pools are freed after every sub-stripe).
- **Reader:** `cat` + `sshd` on the host costs ≈1–3 % of one CPU per kHz [estimate].
- **It can perturb timing.** It is identical in A and B and on all three rigs, so the comparisons
  stay like-for-like. If the measured rate is high, use `--level base`.
- **Snapshots:** `amdgpu_vm_info` takes each VM's reservation lock, so snapshots run only at quiet
  points; snapshot times are in `snaps/<pt>/meta.json` for checking against dispatch lines.
- **A freeze loses only what had not left the host.** Netconsole still carries the fault lines;
  `amdgpu_iv` adds the card's own IH timestamp and `src_data`.

## How to run A and B (VM101, worktree root; `H="192.168.3.121 192.168.3.155 192.168.3.163"`)

0. Boot all three hosts.
   - Owner: netconsole armed and NCPROOF seen; `systemctl stop rig-power-telemetry.service` on
     each host.
   - Alpha: give the go for the settings above.
1. `python3 scripts/rig_fault_recorder.py arm --hosts $H --i-have-go`
2. `S=logs/rig_fault_rec/$(date +%Y%m%d)_AB; mkdir -p $S`
3. `nohup python3 -u scripts/rig_fault_recorder.py stream --hosts $H --out $S/stream > $S/stream.out 2>&1 & echo $! > $S/stream.pid`
4. Wait 120 s for the armed idle rate baseline, then
   `python3 scripts/rig_fault_recorder.py snap --hosts $H --out $S/runA --point T0 --marker`
5. Start the trigger watcher:
   `nohup python3 -u scripts/rig_fault_recorder.py auto --hosts $H --out $S/runA --marker
   --console '/home/michael/distributed_prng_analysis/logs/gate12_*_evidence.txt'
   --coord-glob '/home/michael/distributed_prng_analysis/logs/gate12_????????_??????.log'
   --netconsole /home/michael/distributed_prng_analysis/logs/netconsole_all_rigs.log
   > $S/runA.auto.out 2>&1 & echo $! > $S/runA.auto.pid`
6. **Launch Run A** with the usual procedure (fresh nonce, the same switches as 18:18). `auto` ends
   by itself after T7.
7. Run B on the **same boot, with no reset**: repeat steps 4–6 with `runB`. `stream` keeps
   running.
8. Afterwards:
   - `python3 scripts/rig_fault_recorder.py compare --a $S/runA --b $S/runB`;
   - `python3 scripts/rig_fault_recorder.py counts $S/stream/trace_<host>.log`;
   - `kill $(cat $S/stream.pid)`;
   - `disarm --hosts $H`;
   - `systemctl start rig-power-telemetry.service` on each host.

**Read A vs B, point by point:**
- KFD system/TTM used;
- per-card `vram/gtt_used` and VM/BO counts;
- evict/restore and `bo_move` counts per interval;
- host memory counters;
- worker maps.

**Read rig vs rig:** rrig6600b against rrig6600 and rrig6600c at every point.

**If B faults:**
- the FAULT snapshot holds the two surviving hosts' state at fault time;
- the stream holds the faulting host's last map/PTE/IV lines up to the freeze.
