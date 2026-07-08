# PROPOSAL: Proxmox LXC Migration for GPU Rigs
**Version:** 1.0
**Date:** 2026-07-02
**Status:** DRAFT — pending rrig6600c trial
**Scope:** rrig6600, rrig6600b, rrig6600c (8× RX 6600 each)

---

## 1. Objective

Migrate the three AMD GPU rigs from bare-metal Ubuntu to Proxmox VE hosts
running LXC containers, so that each rig runs its workloads (TFM compute,
Vulkan LLM serving, and future llama.cpp RPC workers) in isolated,
snapshottable containers — with **zero changes to the coordinator, workers,
or distributed_config.json**.

Zeus already runs Proxmox VE; this proposal extends the pattern to the rigs.

## 2. Motivation

| Benefit | Detail |
|---------|--------|
| Snapshot / rollback | Any driver, ROCm, or Mesa experiment can be reverted in seconds. Replaces the retired "golden reference" convention with per-container rollback on every rig. |
| Stack isolation | TFM (ROCm/HIP/CuPy), LLM serving (Vulkan/RADV), and RPC workers (ROCm+RPC build) each live in their own container. Upgrading one can no longer break another. |
| Template-enforced consistency | One container template cloned to all rigs guarantees the identical `/home/michael` layout the codebase depends on — drift becomes impossible. |
| Safe experimentation | Risky builds get a cloned container, not a production machine. |

## 3. Why this is low-risk (verified against the codebase)

- **Per-GPU isolation already exists.** Workers select GPUs via
  `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` at spawn
  (`distributed_worker.py`, `scorer_trial_worker.py`, `nn_gpu_worker.py`).
  This maps directly onto LXC device binding — no code changes.
- **Identical paths are intentional.** The hardcoded
  `/home/michael/distributed_prng_analysis` and `/home/michael/rocm_env`
  layout is reproduced exactly inside the container template.
- **SSH orchestration is IP-based.** The container assumes the rig's
  existing IP and hostname; the coordinator on Zeus is unaware anything
  changed.
- **Ramdisk semantics preserved.** `/dev/shm/prng` (used by Steps 2, 3,
  and 5 via `ramdisk_config.py`) is bind-mounted from the host, so the
  copy-once sentinel and 50%/80% headroom checks measure the real tmpfs.
- **No kernel-version checks exist** in the codebase; workers depend only
  on the ROCm userspace, which is controlled inside the container.

## 4. Architecture

Per rig:

```
Proxmox VE host  (new IP, e.g. .172)  — boots from NEW second SATA drive
│   amdgpu kernel driver, /dev/kfd + /dev/dri owned by host
│
├── CT100  "rig identity" container   — takes the rig's ORIGINAL IP + hostname
│     user michael, SSH keys, /home/michael/* layout
│     ROCm userspace + rocm_env  →  runs TFM workloads
│     bind mounts: /dev/kfd, /dev/dri/*, /dev/shm/prng
│
├── CT110  LLM serving container      — llama.cpp Vulkan/RADV build
│
└── CT120  RPC worker container       — ROCm build, -DGGML_RPC=ON (future)
```

The original bare-metal drive **remains installed and untouched**. BIOS
boot-order flip restores the pre-migration rig in one reboot.

## 5. Planned steps

### Phase 0 — Prerequisites (per rig)
1. Confirm spare SATA data cable + free SATA power connector on PSU.
2. Install mSATA SSD (ORICO 512GB) in ELUTENG 2.5" adapter; mount in rig.
3. Verify drive detected in BIOS; set as first boot device, original
   drive second.

### Phase 1 — Trial conversion (rrig6600c only)
4. Install Proxmox VE 8 on the new drive (target the ELUTENG/ORICO
   device — verify by size/name before confirming).
5. Assign the Proxmox host a **new** IP; leave 192.168.3.162 free.
6. Create CT100 (Ubuntu system container, privileged or with systemd
   support — `systemd-run` is used by the worker launchers).
7. Container identity: hostname `rrig6600c`, static IP 192.168.3.162,
   user `michael`, authorized SSH keys copied from the bare-metal drive.
8. Bind devices and mounts into CT100:
   - `/dev/kfd`, all `/dev/dri/renderD*` + `card*` (8 GPUs)
   - host `/dev/shm/prng` → container `/dev/shm/prng`
   - cgroup device-allow rules; user in `render`,`video` groups.
9. Inside CT100: install ROCm userspace **matching the host kernel's
   amdgpu**, recreate `/home/michael/rocm_env`, clone the repo to
   `/home/michael/distributed_prng_analysis`.
10. Sanity: `rocm-smi` reports 8× RX 6600; `preflight_check.py` passes
    all hard checks from Zeus.

### Phase 2 — Validation (acceptance criteria)
11. **TFM acceptance run:** dispatch a real Step 2 or Step 3 job from the
    Zeus coordinator to 192.168.3.162 with no config changes. Must
    complete with results consistent with bare-metal history.
12. **RAM headroom:** monitor host+container memory during the run.
    Ramdisk headroom checks must stay below the 50% warn threshold on a
    representative workload. Record peak usage.
13. **LLM baseline:** stand up CT110 with the Vulkan llama.cpp build and
    confirm ~32 t/s on DeepSeek-R1-Distill-Qwen-32B-Q4_K_M (8-GPU pool).
14. **Cold-start caution:** apply existing warm-up/power-cap practice
    before any 8-GPU simultaneous load (known power-spike crash risk).

### Phase 3 — Decision gate
15. All three criteria pass → proceed to Phase 4.
16. Any failure → flip BIOS boot order back to the original drive
    (instant rollback), document findings, revise proposal.

### Phase 4 — Rollout (rrig6600, then rrig6600b)
17. Convert CT100/CT110 into Proxmox templates.
18. Repeat Phase 0–1 per rig using the templates; only per-container
    identity (hostname, IP 192.168.3.120 / .154) changes.
19. Re-run the Phase 2 acceptance run once per rig.

### Phase 5 — Documentation
20. Update `Cluster_operating_manual.txt`, `PROJECT_MAP.md`, and
    CLAUDE_PROJECT_INSTRUCTIONS.md (remove retired golden-reference rule;
    add container/snapshot workflow).
21. Dual-push all changes to private origin and public mirror.

## 6. Known risks & mitigations

| Risk | Mitigation |
|------|------------|
| 8 GB rig RAM: Proxmox host (~1–1.5 GB) + workers + /dev/shm share one pool | Measured in Phase 2 before rollout. Fallbacks: RAM upgrade on TB360-BTC, or reduce `max_concurrent_script_jobs`. |
| ROCm userspace vs. host kernel mismatch | Pin ROCm version in template; validate in step 10. |
| Cold-start power spike (historical crash) | Keep warm-up/power-cap procedure; never cold-load 8 GPUs. |
| Proxmox install targets wrong drive | Original drive identifiable by size/name; verify before confirming install. |
| Anything else | Original drive untouched; BIOS boot-order flip = full rollback. |

## 7. Cost

- Hardware: $0 incremental for trial (mSATA drives + adapter on hand).
  Optional later: RAM upgrades if Phase 2 shows pressure.
- Time: ~1 day for the rrig6600c trial; less per rig afterward
  (template clone).

## 8. Out of scope

- Zeus (already on Proxmox; container layout there is a separate task).
- Proxmox clustering of the rigs (no benefit for this workload).
- llama.cpp RPC cluster buildout (covered by DISTRIBUTED.md; CT120 is
  provisioned-for but not built here).
