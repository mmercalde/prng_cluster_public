# PROPOSAL: Infrastructure Reconciliation — Proxmox Container Strategy (S172)
**Version:** 1.0.0
**Date:** 2026-07-07
**Author:** Team Alpha (Claude)
**Reviewer:** Team Beta
**Status:** APPROVED (Team Beta) — required review edits applied; LXC-vs-VM decision pending rrig6600c trial
**Related:** PROPOSAL_Proxmox_LXC_Rig_Migration_v1_0, S172 RANGE-MINER brief

---

## Review edits applied (Team Beta, 2026-07-07)

1. §5 Gate #1 broadened from IOMMU grouping to full VM feasibility.
2. §5 Gate #2 acceptance changed to artifact compatibility (Steps 2–6),
   not runtime history.
3. §4 wording softened so the doc does not prejudge the trial.
4. §8 open item #4 added: host/CT100 responsibility boundary.
5. §0 engineering principle added (measured behavior over theory).
6. §6 CT100 annotation emphasizes the rig-identity concept.

---

## 0. Engineering principle

Where competing infrastructure designs exist, preference is given to
measured behavior on target hardware over theoretical analysis. This
document applies that principle to the LXC-vs-VM question via the
rrig6600c trial (§5).

---

## 1. Purpose

Two infrastructure briefs were produced independently for the S172 pivot:

- **Brief A (VM-first):** Proxmox → Ubuntu VM (GPU passthrough) → venv →
  RANGE-MINER worker; LXC reserved for CPU-only agent/contract work.
- **Brief B (LXC-first):** Proxmox → LXC (CT100/110/120) for all rig
  workloads including GPU, via `/dev/kfd` + `/dev/dri` binding.

They agree on ~90% of the architecture. This document merges them and
isolates the single genuine divergence — **container type for the GPU
workload on the AMD rigs** — resolving it by measurement rather than
debate.

## 2. Points of agreement (adopted as settled)

| Decision | Source |
|----------|--------|
| Proxmox VE on every machine (Zeus already dual-boot) | Both |
| Snapshots / rollback / repeatable deploys | Both |
| One golden image cloned across the identical rigs | Both |
| **No Docker** — plain `venv` + `requirements.txt` + systemd | Both |
| LXC for coding agents, WATCHER, fake miner, CPU contract tests | Both |
| Original bare-metal drive retained; BIOS boot-order = rollback | Both |
| Two-sandbox model (see §3) | Brief A framing, adopted |

## 3. Two-sandbox model (adopted from Brief A)

Brief A's sharpest contribution, adopted in full:

- **Sandbox A — CPU-only (LXC).** Autonomous agents, protocol/contract
  development, artifact validation, Step 2–6 compatibility tests.
  No GPUs. Proves *interfaces and artifact-compatibility only*.
- **Sandbox B — GPU-backed.** Worker lifecycle, kernel-launch behavior,
  persistent workers, pool scaling, ROCm validation, performance.

**Project rule (retained verbatim from Brief A):** Sandbox A does NOT
prove the S172 pivot solved the original GPU-utilization problem. Real
GPU validation MUST occur in Sandbox B on real hardware. This rule is
non-negotiable and belongs in the acceptance criteria.

## 4. The divergence: GPU workload container type

Brief A places the GPU worker in an **Ubuntu VM (PCIe passthrough)**.
Brief B places it in an **LXC (device binding)**. This is the only
unresolved decision. The currently available hardware evidence suggests
LXC may be a better fit for the AMD mining rigs — to be confirmed or
overturned by the §5 trial, not assumed here:

1. **Board class mismatch.** Brief A's proof — Windows VM100 passing 3
   GPUs — is on Zeus (ASUS WS X299 SAGE, clean IOMMU groups). The rigs
   are **Biostar TB360-BTC Pro 2.0** mining boards running 8× RX 6600 on
   x1 risers, where IOMMU groups are typically fragmented. Passthrough
   success on Zeus does not transfer to the BTC board.
2. **PCIe fragility on record.** TB_SUBMISSION_S159G documents rrig6600
   post-crash PCIe bus reassignment across all slots and NIC TX-queue
   timeouts. VFIO bind/reset operates at exactly this layer; LXC device
   binding does not touch it.
3. **8 GB RAM ceiling.** A full VM reserves RAM and runs a second kernel;
   LXC shares host kernel + page cache. Material on an 8 GB rig.
4. **Ramdisk semantics.** `/dev/shm/prng` (Steps 2/3/5) bind-mounts
   cleanly into an LXC, preserving the copy-once sentinel and 50/80%
   headroom checks. A VM has no equivalent clean bind-mount (virtio-9p /
   NFS defeats the RAM-cache purpose).

**Where a VM is correct:** different-kernel needs, hard multi-tenant
isolation, or non-Linux guests. All apply to Zeus VM100 (Windows); none
apply to the single-tenant, host-kernel-matched ROCm worker on a rig.
Brief A's VM approach is therefore retained **for Zeus**, not the rigs.

## 5. Resolution by measurement (rrig6600c trial)

Rather than arbitrate, the rrig6600c conversion tests both positions on
the actual hardware. Acceptance gate:

1. **Gate #1 — VM feasibility (decides VM vs LXC):** Determine whether the
   Biostar TB360-BTC Pro 2.0 can reliably support an 8-GPU Ubuntu VM.
   This is broader than IOMMU grouping alone; all of the following must
   hold: (a) IOMMU groups allow the 8 RX 6600 to be assigned, (b) the
   guest boots with the cards attached, (c) ROCm enumerates all 8 GPUs
   inside the guest, (d) device/FLR resets work, (e) performance is
   acceptable vs. bare metal, (f) RAM headroom (guest + workload +
   ramdisk) fits within 8 GB. If all hold → VM approach validated on the
   rigs. If any fails → the VM approach is unsuitable for this hardware
   and LXC becomes the primary deployment model.
2. **Gate #2 — artifact acceptance run:** Real Step 1 job dispatched from
   the Zeus coordinator to 192.168.3.162, no config changes. Acceptance =
   the emitted artifacts (22-array NPZ + contract files) are accepted
   unchanged by Steps 2–6. This is the S172 contract; runtime history is
   secondary to artifact compatibility.
3. **RAM headroom:** Peak host+guest memory during the run stays under
   the ramdisk 50% warn threshold. Record the number.
4. **LLM baseline (CT110 / Sandbox B):** ~32 t/s on
   DeepSeek-R1-Distill-Qwen-32B-Q4_K_M across the 8-GPU pool.
5. **Cold-start caution:** Warm-up / power-cap practice applied before any
   8-GPU simultaneous load (documented crash risk, ROCm issue #5238).

Whichever container type passes gate #1 on real hardware becomes the
standard, then templates to rrig6600 and rrig6600b.

## 6. Merged target architecture

```
Zeus (X299 SAGE)               AMD rig (Biostar TB360-BTC Pro 2.0)
Proxmox VE                     Proxmox VE  (host IP = rig IP + 10)
├── VM100 Windows (proven)     ├── Sandbox A: LXC — agents, WATCHER,
├── LXC — Sandbox A CPU work   │     fake miner, contract/compat tests
└── LXC/VM — LLM + coordinator └── Sandbox B: CT100* (*type per §5)
                                     ├─ Rig identity (orig IP/hostname)
                                     ├─ ROCm userspace
                                     ├─ Python venv
                                     └─ RANGE-MINER worker
                                   ├── CT110 — llama.cpp Vulkan serving
                                   └── CT120 — RPC worker (future)
                                   bind: /dev/kfd, /dev/dri/*, /dev/shm/prng
```

IP scheme (host = rig + 10): rrig6600 .120→host .130 · rrig6600b
.154→host .164 · rrig6600c .162→host .172. Containers retain original
rig IP/hostname so the coordinator and `deploy_to_rigs.sh` are unchanged.

## 7. Cluster posture

Standalone for the rrig6600c trial (isolates the GPU/RAM variables from
corosync/quorum). Cluster all nodes at Phase 4 rollout with a **QDevice
on SER8** for odd-vote quorum. Join each Proxmox host to the cluster
*before* creating its containers (join replaces the cluster FS).

## 8. Open items for Team Beta

1. Confirm the §5 gate #1 criteria are the right VM/LXC decision boundary.
2. Confirm the two-sandbox rule (§3) is worded acceptably for the S172
   validation plan.
3. Ramdisk sizing under an 8-GPU load — pull from `ramdisk_config.py`
   before the trial to predict headroom.
4. **Host responsibility boundary.** Document explicitly which ROCm
   components must live on the Proxmox host (amdgpu kernel driver /
   `/dev/kfd` + `/dev/dri` enumeration) versus exclusively inside CT100
   (ROCm userspace, HIP runtime, venv), so future updates don't blur the
   separation. Applies to the LXC design; under a VM the boundary differs
   (the guest owns the full stack post-passthrough).
