# PHASE6_PREREQS.md

**S172 RANGE-MINER — infrastructure prerequisites for the first real trials
(Phase 6 verify + Phase 7 soak).**

These are hardware/network tasks that do NOT block the remaining code
deliverables (D2-D6) but DO gate real-silicon testing. They can be worked in
parallel with the doc/review cycles by Michael on the Proxmox lane — the goal
is that the fleet is ready the day the code is. Owner: Michael. Claude Code is
not involved in any of these.

Status legend: ☐ open · ◐ in progress · ☑ done

| # | Item | Gate it blocks | Status |
|---|------|----------------|--------|
| 1 | Second 3080Ti passthrough into VM101 (`hostpci1`) | Phase 6 multi-worker realism, ≥500K s/s throughput bar, PWC/ZMQ dual-GPU oracle | ☐ |
| 2 | `michael → CT100` SSH key auth (all migrated rigs) | Phase-4 real run (coordinator/miner reach workers at CT100 addr) | ☐ |
| 3 | `rrig6600` Proxmox migration (still bare-metal at .120) | Full-fleet (26-GPU) saturation soak — Phase 7 | ☐ |
| 4 | VM101 static IP (currently DHCP at 192.168.3.177, not pinned) | Stable coordinator identity for multi-node real runs | ☐ |

---

## Detail & acceptance

### 1. Second 3080Ti → VM101
VM101 today has ONE 3080Ti (`hostpci0=68:00`; in-guest `nvidia-smi` sees one
12GB card). The **D6 Zeus-only smoke trial does NOT need this** — a single-GPU
run (`worker_pool_size=1`) proves the full plumbing (sieve kernels → sub-stripe
spools → publish → assembly → NPZ → Step 2 read); the path is identical at
N=1. This item becomes load-bearing only at Phase 6.

Procedure (on `pzeus`, when nothing is running — requires a VM101 stop/start):
- `lspci -nn | grep -i nvidia` on the host → find the second card's PCI addr;
- confirm it sits in a cleanly separable IOMMU group;
- `qm set 101 -hostpci1 <addr>,pcie=1` (mirror how `hostpci0` was configured);
- restart VM101; confirm in-guest `nvidia-smi` shows **two** cards.
- **Sanity check:** the bare-metal fallback at .127 must be unaffected — it
  gets both cards natively on a host-boot regardless of VM config, but confirm
  no vfio bind at the host-boot level would surprise the fallback path.

**Acceptance:** in-guest `nvidia-smi` lists 2× 3080Ti; a 2-worker miner run
assigns stripes to both; bare-metal .127 boot still sees both cards natively.

### 2. michael → CT100 SSH keys
Coordinator/miner reach WORKERS at the CT100 container address per the
migration scheme (RUNBOOK_v1.6: host=rig+1, CT100 worker=host+1). Key auth
from `michael` to CT100 on each migrated rig is not yet set up.
**Acceptance:** passwordless `ssh michael@<CT100-addr>` succeeds for
rrig6600b (.156) and rrig6600c (.164); added for rrig6600 (.122) once item 3
lands. Checked-in `dotfiles/ssh_config` refreshed for migrated rigs (currently
stale).

### 3. rrig6600 Proxmox migration
`rrig6600` (.120) is still bare-metal; rrig6600b (.155/CT.156) and rrig6600c
(.163/CT.164) are migrated ROCm. Full 26-GPU saturation — the exact condition
that killed PWC and that RANGE-MINER exists to survive — requires all three
rigs on the migrated topology.
**Acceptance:** rrig6600 host at .121, CT100 worker at .122 (static), 8/8 GPUs
visible under ROCm in-container, reachable per item 2.

### 4. VM101 static IP
VM101 is DHCP at 192.168.3.177, not yet pinned. Multi-node real runs want a
stable coordinator identity.
**Acceptance:** .177 (or chosen addr) reserved/static; git remotes and any
worker→coordinator references unaffected.

---

## Where these surface in the plan
- **D6 smoke trial (Zeus-only, single-GPU):** needs NONE of the above — it is
  the earliest real-silicon checkpoint and should be a first-class D6 gate.
- **Phase 6 four-path verify + throughput bar:** needs items 1, 2, 4.
- **Phase 7 full-fleet soak (≥5 high + ≥5 low survivor, mixed const/hybrid,
  26-GPU saturation):** needs all four.

_Last updated: 2026-07-23. Update the status column as items land; commit
alongside the next deliverable._
