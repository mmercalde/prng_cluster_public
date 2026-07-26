# PHASE6_PREREQS.md — REV3

**S172 RANGE-MINER — operational prerequisites for real-silicon testing.**

REV3 absorbs the second and third Team Beta reviews (transport gates Phase 7; disposable same-filesystem preflight; single-GPU baseline made explicit; D3.0-B surfaced; per-item dirty-tree disposition; preflight directory moved outside the worktree). REV2 absorbed the first: stale phase status corrected, item 2
relabelled, and three prerequisites added that D3.5 made load-bearing
(publication preflight, code/clock parity, transport reachability).

## Current code status

```text
Closed:                     D0 · D1.0 · D1.1 · D2 · D3.0 · D3 · D3.25 · D3.5
Remaining deliverables:     D4 · D5 · D6
Certification prerequisite: D3.0-B before Phase 6 certification
HEAD:                       46a3828
```

**D3.0-B** does not block D4-D6 implementation or the D6 smoke unless its
affected legacy writer is exercised, but it is mandatory before Phase 6
certification.

These items are hardware/environment tasks that do **not** block the remaining
code deliverables (D4/D5/D6) but **do** gate real-silicon testing. They can be
worked in parallel with the code work, on the Proxmox lane. Owner: Michael.
Claude Code is not involved in any of them.

**Important:** among the open infrastructure items in this document, the D6
Zeus-only smoke is blocked **only by item 5**. This assumes VM101's existing
`hostpci0` 3080Ti, NVIDIA driver, CUDA environment and single-worker launch path
remain healthy — those are already deployed rather than open prerequisites, but
the assumption is explicit and is checked in item 5.

Status legend: ☐ open · ◐ in progress · ☑ done

| # | Item | Gates | Status |
|---|------|-------|--------|
| 1 | Second 3080Ti passthrough into VM101 (`hostpci1`) | Phase 6 verify, Phase 7 soak | ☐ |
| 2 | `michael → CT100` passwordless SSH (all migrated rigs) | Phase 6 verify, Phase 7 soak | ☐ |
| 3 | `rrig6600` Proxmox migration (still bare-metal at .120) | Phase 7 soak | ☐ |
| 4 | VM101 stable address (currently DHCP at 192.168.3.177) | Phase 6 verify, Phase 7 soak | ☐ |
| 5 | **D3.5 publication filesystem + clean-tree preflight** | **D6 smoke**, Phase 6, Phase 7 | ☐ |
| 6 | Code/environment parity + clock synchronization | Phase 6 verify, Phase 7 soak | ☐ |
| 7 | PWC/ZMQ transport reachability + firewall verification | Phase 6 verify, Phase 7 soak | ☐ |

### Gate matrix

| Item | D6 single-GPU | Phase 6 verify | Phase 7 soak |
|---|---|---|---|
| 1 second 3080Ti | No | **Yes** | **Yes** |
| 2 CT100 SSH keys | No | **Yes** | **Yes** |
| 3 rrig6600 migration | No | No | **Yes** |
| 4 VM101 stable address | No | **Yes** | **Yes** |
| 5 publication preflight | **Yes** | **Yes** | **Yes** |
| 6 code/clock parity | No | **Yes** | **Yes** |
| 7 transport reachability | No | **Yes** | **Yes** |

---

## 5. D3.5 publication filesystem + clean-tree preflight — **the only D6 blocker**

D3.5 (`46a3828`) publishes through an immutable generation directory committed
by a single atomic `current`-pointer swap. That guarantee **depends on
filesystem semantics**: symlinks, same-filesystem atomic rename, and directory
fsync. On NFS, CIFS, SSHFS or similar these are absent or differently defined,
and the single commit point silently stops being atomic. The finalizer also
rejects `repository_tree_clean=False`, so a dirty worktree does not merely look
untidy — **it prevents certification.**

**Acceptance:**

- VM101 checked out at the approved D6 commit, descended from `46a3828`;
- `git status --porcelain` **empty**;
- neither rejected legacy NPZ exists as a regular root file;
- no stale `.s172_accumulator` state conflicts with the first generation;
- the repository filesystem supports symlinks, same-filesystem atomic rename
  and directory fsync;
- the output path is **not** NFS/CIFS/SSHFS or another filesystem with
  unsuitable rename/durability semantics;
- a disposable publication preflight proves: dangling root aliases bootstrap; a
  generation directory renames atomically; `current` is atomically replaced; and
  the resulting hash-bound chain tip validates.

```bash
cd ~/distributed_prng_analysis

git status --porcelain
git rev-parse HEAD
findmnt -T . -o TARGET,SOURCE,FSTYPE,OPTIONS

for f in bidirectional_survivors_all.npz bidirectional_survivors_binary.npz; do
    if [ -e "$f" ] || [ -L "$f" ]; then ls -l "$f"; else
        echo "$f: absent — clean first-generation state"; fi
done

find .s172_accumulator -maxdepth 3 -ls 2>/dev/null || true
```

### Run the mechanics preflight in a DISPOSABLE directory

**Binding:** run the filesystem-mechanics preflight in a temporary directory on
the **same mounted filesystem** as the repository. Do **not** use the
repository's actual accumulator paths. Bootstrapping aliases or creating
`.s172_accumulator` state in the production root to test semantics would
contaminate the exact clean-start state the D6 smoke is meant to certify.

**The preflight directory must live OUTSIDE the worktree.** Creating it inside
the repository makes `git status --porcelain` non-empty for as long as it
exists — and the finalizer's certification contract *rejects a dirty
repository*, so invoking the real publication path from an in-repo preflight
would fail for a reason unrelated to filesystem semantics. Place it beside the
repository and prove it is still on the same filesystem:

```bash
cd ~/distributed_prng_analysis

repo_fs_dev=$(stat -c '%d' .)
preflight=$(mktemp -d ../.d35-publish-preflight.XXXXXX)

test "$(stat -c '%d' "$preflight")" = "$repo_fs_dev" ||
    {
        echo "Preflight directory is not on the repository filesystem"
        rm -rf -- "$preflight"
        exit 1
    }

echo "Preflight root: $preflight"

# Invoke the real D3.5 finalizer/publication path with:
#   output_root="$preflight"
# rather than reimplementing the rename sequence in shell.

rm -rf -- "$preflight"

test -z "$(git status --porcelain)" ||
    {
        echo "Worktree is not clean after preflight"
        git status --short
        exit 1
    }
```

Also confirm no accumulator state leaked into the production root:

```bash
if [ -e .s172_accumulator ] || [ -L .s172_accumulator ]; then
    find .s172_accumulator -maxdepth 3 -ls
fi
```

The `-L` arm matters: `test ! -e` **follows** symlinks, so a *dangling*
`.s172_accumulator` symlink would satisfy `! -e` and pass unnoticed.

### D6 single-GPU readiness check

```bash
nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader
```

Acceptance: exactly one intended 3080Ti visible for the D6 N=1 run; the CUDA
worker imports and starts successfully; no second-GPU dependency in the smoke
configuration. **This does not make item 1 a D6 prerequisite.**

**Known open worktree sub-items:** `tmp/` and four `CLAUDE_CODE_BRIEF_S176-178*`
files are untracked and make `git status --porcelain` non-empty. **Each needs an
explicit, deliberate disposition** before D6 — not a blanket ignore:

```text
needed governance record   -> rename appropriately and commit
obsolete scratch material  -> delete
reproducible temp output   -> add a NARROWLY SCOPED ignore rule
```

Do **not** add a blanket `tmp/` ignore without first confirming that directory
cannot contain evidence or artifacts worth retaining. The final D6 preflight
must show `git status --porcelain` with **no output**.

The forensic archive is already handled — it lives outside the worktree at
`/home/michael/tfm_forensics/` per the Ruling-F disposition.

**For D6's brief:** state explicitly whether D6 **fails closed** on a dirty tree
or publishes an uncertified generation. Team Alpha recommends **fail closed** —
the first certified baseline must not become uncertifiable by accident.

## 6. Code/environment parity + clock synchronization

**Acceptance:**

- every coordinator and worker reports the intended software revision or
  deployment bundle;
- CUDA/ROCm environment variables match the approved node profiles;
- worker launch scripts and service definitions are the intended versions;
- all nodes synchronized via `chrony` or `systemd-timesyncd`;
- clock skew small enough that distributed logs and generation provenance
  correlate reliably.

```bash
timedatectl show -p NTPSynchronized --value
date --iso-8601=ns
```

Run on VM101, each Proxmox host and each CT100.

## 7. PWC/ZMQ transport reachability + firewall verification

SSH authentication alone does **not** prove PWC and ZMQ can communicate.

**Acceptance:**

- VM101 reaches every CT100 on the configured worker/control ports;
- workers reach VM101 on any callback/result ports the selected transport uses;
- Proxmox host, container and guest firewalls permit the exact flows;
- no port exposed beyond the trusted cluster LAN;
- PWC and ZMQ each complete a small connection/registration test **before** GPU
  work begins.

Reference the authoritative configured port list rather than duplicating port
numbers here, which would drift. (Note: Zeus has no firewall — never suggest
`ufw`; the dashboard port 5002 is fixed and must not change.)

---

## 1. Second 3080Ti → VM101

VM101 has one 3080Ti today (`hostpci0=68:00`). **Not needed for the D6 smoke** —
a single-GPU run (`worker_pool_size=1`) proves the full plumbing; the path is
identical at N=1.

Procedure on `pzeus`, when nothing is running (requires a VM101 stop/start):
`lspci -nn | grep -i nvidia` → find the second address; confirm a cleanly
separable IOMMU group; `qm set 101 -hostpci1 <addr>,pcie=1`; restart VM101.

**Acceptance:**

- both host PCI addresses and IOMMU group memberships recorded;
- no unrelated device shares the second GPU's group;
- `nvidia-smi -L` verified after a **cold VM start**, not only a reboot;
- both workers process **nonempty** stripes;
- stable device enumeration does **not** become an implicit correctness
  dependency;
- bare-metal fallback at .127 unaffected — it gets both cards natively on a
  host boot regardless of VM config, but confirm no vfio bind at host-boot level
  would surprise it.

## 2. `michael → CT100` passwordless SSH

Coordinator/miner reach workers at the CT100 address (RUNBOOK_v1.6:
host = rig+1, CT100 = host+1).

**Acceptance — test noninteractive automation, not an interactive login:**

```bash
ssh -o BatchMode=yes -o ConnectTimeout=5 michael@192.168.3.156 \
  'hostname; id; rocminfo >/dev/null && echo ROCm_OK'
```

Repeat for `.164`, and for `.122` once item 3 lands. **Prepopulate and verify
host keys** so the first automated run cannot block on an authenticity prompt.
Refresh the checked-in `dotfiles/ssh_config`, currently stale for migrated rigs.

## 3. `rrig6600` Proxmox migration

`rrig6600` (.120) is still bare-metal; b and c are migrated ROCm. Full 26-GPU
saturation — the exact condition that killed PWC — requires all three rigs on
the migrated topology.

**Acceptance:** host at .121, CT100 worker at .122 (static), plus

- `rocminfo` enumeration of all 8 GPUs;
- `rocm-smi` health and temperatures;
- persistent ROCm environment;
- fan-curve service active;
- worker startup **after reboot**;
- one small all-eight-GPU workload completes;
- **no duplicate machine IDs, SSH host keys or network identities** cloned from
  the other containers.

## 4. VM101 stable address

Prefer a **router-side DHCP reservation** for .177, or a documented guest-static
configuration **outside the DHCP dynamic pool**.

**Acceptance:** no address conflict; correct gateway and DNS; survives a VM
reboot; survives a **Proxmox host** reboot; worker configs reference the final
address or a stable hostname.

---

## Where these surface

- **D6 smoke trial (Zeus-only, single-GPU):** among the open items, blocked by
  **item 5 only**, assuming the existing single-GPU/CUDA baseline stays healthy
  (verified within item 5). Earliest real-silicon checkpoint; produces the
  **first certified accumulator generation**.
- **Phase 6 four-path verify + throughput bar:** items 1, 2, 4, 5, 6, 7.
- **Phase 7 full-fleet soak** (≥5 high + ≥5 low survivor, mixed const/hybrid,
  26-GPU saturation): items 1, 2, 3, 4, 5, 6, **7**. Transport verified during
  Phase 6 does not stop being a Phase 7 prerequisite — Phase 7 inherits it as
  already-satisfied.
- **D3.0-B** (legacy writer corrections) must also complete before Phase 6
  certification — tracked with the code deliverables, not here.

_REV3 — last updated 2026-07-25, post-D3.5 (`46a3828`). Update the status
column as items land._
