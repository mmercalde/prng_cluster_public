# PHASE6_PREREQS.md — REV4

**S172 RANGE-MINER — operational prerequisites for real-silicon testing.**

**REV4 (2026-08-02)** corrects statuses from live measurement
(`docs/S172_PHASE_7_PREREQ_REPORT.md`) and repairs a code-status block that had
gone three months stale. **Five of seven statuses changed.** REV4 also records
that **D3.0-B, stated here as mandatory before Phase 6 certification, was never
completed — and Phase 6 certified anyway** (§D3.0-B below).

REV3 absorbed the second and third Team Beta reviews (transport gates Phase 7; disposable same-filesystem preflight; single-GPU baseline made explicit; D3.0-B surfaced; per-item dirty-tree disposition; preflight directory moved outside the worktree). REV2 absorbed the first: stale phase status corrected, item 2
relabelled, and three prerequisites added that D3.5 made load-bearing
(publication preflight, code/clock parity, transport reachability).

## Current code status

```text
Closed:      D0 · D1.0 · D1.1 · D2 · D3.0 · D3 · D3.25 · D3.5 · D4 · D5
             D6 · D6.1 · D6.2
Phase 6:     CERTIFIED d98298c (bounded, TB 2026-08-02)
D6.2:        CERTIFIED 18a2419 — n_parallel == 1 ONLY (TB scope)
Open:        D6.3 (checkpoint retention) · 6-P2 (scraper) — neither blocks Phase 7
HEAD:        3561cda
```

**REV3's block was three months stale** — it listed D4/D5/D6 as remaining and
named HEAD `46a3828`. Corrected from measurement, not from recall.

### ⚠ D3.0-B — stated mandatory, never completed, Phase 6 certified regardless

REV3 recorded D3.0-B as *"mandatory before Phase 6 certification."* **No commit
completing it exists** (`git log --all --grep` over the label returns nothing but
the doc that raised it), and **the defect it targets is live at HEAD**:

```python
# convert_survivors_to_binary.py:184
encode_prng_type(s.get('prng_type', s.get('prng_base', 'java_lcg')))
```

A record carrying **neither** `prng_type` **nor** `prng_base` still silently
becomes `'java_lcg'` rather than failing closed — the exact residual default
D3.0-B exists to purge (`docs/TEAM_ALPHA_REVIEW_S172_PHASE5_D3_0.md` §5.4).

**Phase 6 certified at `d98298c` without it.** Whether Beta waived D3.0-B,
superseded it, or it was simply never raised at certification is **[UNVERIFIED]**
— the repository does not say. It is plausible Wall A/B never exercised the
legacy writer, which would make the omission harmless in fact while leaving a
stated prerequisite unmet on paper.

**This is flagged for Beta's disposition, not silently dropped.** A checklist
that quietly loses a prerequisite is worse than one that never had it.

These items are hardware/environment tasks that do **not** block the remaining
code deliverables (D4/D5/D6) but **do** gate real-silicon testing. They can be
worked in parallel with the code work, on the Proxmox lane. Owner: Michael.
**Claude Code performs the MEASUREMENT of these items** (REV4's statuses come
from `docs/S172_PHASE_7_PREREQ_REPORT.md`); the hardware and network changes
themselves remain Michael's.

**Important:** among the open infrastructure items in this document, the D6
Zeus-only smoke is blocked **only by item 5**. This assumes VM101's existing
`hostpci0` 3080Ti, NVIDIA driver, CUDA environment and single-worker launch path
remain healthy — those are already deployed rather than open prerequisites, but
the assumption is explicit and is checked in item 5.

Status legend: ☐ open · ◐ in progress · ☑ done

| # | Item | Gates | Status | evidence (2026-08-02) |
|---|------|-------|--------|---|
| 1 | Second 3080Ti passthrough into VM101 (`hostpci1`) | Phase 6 verify, Phase 7 soak | **☐ OPEN** | one RTX 3080 Ti; second still on VM100. **Not a Phase-7 blocker — 25 GPUs is owner-mandated** |
| 2 | `michael → CT100` passwordless SSH | Phase 6 verify, Phase 7 soak | **☑ DONE** | `.122`/`.156`/`.164` answer under `BatchMode=yes`, no prompt |
| 3 | `rrig6600` Proxmox migration | Phase 7 soak | **☑ DONE** | `.120`/`.154`/`.162` DOWN; `.122`/`.156`/`.164` UP |
| 4 | VM101 stable address | Phase 6 verify, Phase 7 soak | **☑ DONE** | router-side DHCP reservation, `bc:24:11:19:4f:24` → `192.168.3.177`. The lease cannot move mid-soak |
| 5 | **D3.5 publication filesystem + clean-tree preflight** | **D6 smoke**, Phase 6, Phase 7 | **☑ DONE** | 25/25, 0 failures, incl. a `repository_tree_clean=False` fault-injection control |
| 6 | Code/environment parity + clock synchronization | Phase 6 verify, Phase 7 soak | **☑ DONE** | clock ✅ NTP active; code parity ✅ **17/17 exact on all three rigs** after redeploy, provenance re-verified |
| 7 | Transport reachability + firewall verification | Phase 6 verify, Phase 7 soak | **☑ DONE** | four miner flows on **5700** accepted; no enforcing firewall (`ENABLED=no`) |

**Five statuses changed.** Evidence per item: `docs/S172_PHASE_7_PREREQ_REPORT.md`
§1-§5.

**Item 6 is the trap.** The clock half is satisfied and was previously the only
half measured. **The code half is not:** every `.py` under `miner/` and `utils/`
compared by sha256, VM101 against each CT100 — **17 files here, 16 there; 14
identical, 2 differing, 1 absent**, identically on all three rigs.

| file | rig state | vintage |
|---|---|---|
| `miner/range_miner_coordinator.py` | differs | matches `ee0db06` |
| `miner/dataset_authority.py` | differs | matches `8600e75` |
| `utils/checkpoint_d6_2.py` | **absent** | post-dates the deployed bundle |

**Both stale modules are inside the worker's executing import closure** —
`miner/__init__.py:19` imports the coordinator, which pulls in
`dataset_authority`; confirmed on each rig by reading `sys.modules` after
importing `miner.range_miner_worker`. The rig copies pre-date `eff6616`,
`f7583bc` **and `18a2419` — the D6.2 repair this soak exists to exercise.**

VM101's coordinator drives the run, so the rig copies were *loaded but not
driven*, and the soak would probably still have worked. **That is not the
acceptance criterion.** A soak whose headline question is *"does the S166 clear
hold?"* must not run from a fleet carrying pre-D6.2 modules in its import path.

**RESOLVED 2026-08-02 — item 6 is ☑.** `miner/` and `utils/` redeployed to all
three CT100s and re-verified:

| rig | before | after |
|---|---|---|
| `.122` rrig6600 | 14/17, 2 differing, 1 missing | **17/17 exact** |
| `.156` rrig6600b | 14/17, 2 differing, 1 missing | **17/17 exact** |
| `.164` rrig6600c | 14/17, 2 differing, 1 missing | **17/17 exact** |

Module provenance re-checked **by the same method that found the drift** — fresh
interpreter under `~/rocm_env`, same import:
`miner.range_miner_coordinator` `2b1527bf…(ee0db06)` → `70f8fbaa…` ·
`miner.dataset_authority` `aa3d1792…(8600e75)` → `365c8e3e…` ·
`utils.checkpoint_d6_2` `ModuleNotFoundError` → `c3faecaa…`. Every `__file__`
resolves under `/home/michael/distributed_prng_analysis/` with no shadowing copy
on `sys.path`, and each probe printed its own `socket.gethostname()` — three
distinct machines, not one answered three times.

**The deploy mechanism was established from the host, not invented:** initial
`git clone` from the public remote (`.122`'s reflog), then targeted `scp` from
VM101 on top (`REMOTE_NODE_SETUP_CHECKLIST.md:127,133,139`, `CLAUDE.md §2`).
**`scp` was used uniformly** rather than `git pull` on `.122` plus `scp`
elsewhere — a per-rig split would have created the second deploy method this task
exists to eliminate. **Bytes staged from `git archive 18a2419`**, not from the
working tree, so provenance ties to the commit.

*(The rigs are deployment targets, not working copies: `rrig6600` carries a
worktree at `8e2f5bf` with 84 dirty entries; `rrig6600b` and `rrig6600c` have no
git repository at all. **Digest comparison, never `git rev-parse`, is the parity
evidence.**)*

### ⚠ `localhost.gpu_count` — the frozen set carries 26 identities, not 25

`distributed_config.json` declares `gpu_count: 2` for `localhost`, written for the
two-3080Ti configuration. So `identity_count = 2+8+8+8 = 26`, `min(25, 26) = 25`,
and **no clamp is recorded** — `admission_count` is 25 by construction, which is
correct.

**But provenance records 26 identities admitting 25**, which reads exactly like
the 26-set-that-came-up-short that `eff6616` closed. **It is not that failure:**
the threshold is 25, 25 real workers exist, the 180 s window is meetable, and no
production path spawns workers by iterating `gpu_count` (they launch explicitly
with `--gpu-id N`). **The cost is auditability, not execution.**

**Recommended:** set `localhost.gpu_count: 1` to match measured hardware — then
`identity_count = requested = admission = 25`. **Three caveats:** it changes
`set_id`; it must be **committed before launch** because item 5's clean-tree wall
rejects a dirty tree at finalization; and it is a **distinct field from the
bare-metal addresses in that file, which stay untouched** (CLAUDE.md §3).

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

## 5. D3.5 publication filesystem + clean-tree preflight — **☑ DONE (25/25)**

*(REV3 titled this "the only D6 blocker". D6 closed long ago; retained for the mechanics below, which remain the acceptance procedure.)*

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
- **Phase 7 full-fleet soak** (50 trials, ≥5 high + ≥5 low survivor, mixed
  const/hybrid, **25-GPU saturation — owner-mandated**, `n_parallel=1` binding
  per D6.2 certification): items 2, 3, 5, 6, 7 required; **item 1 is explicitly
  waived by the owner**; item 4 required. **ALL OPERATIONAL PREREQUISITES ARE NOW
  CLOSED** — items 2, 3, 4, 5, 6, 7 ☑ on live measurement; item 1 waived.
- **D3.0-B** — see the header. **Never completed; Phase 6 certified regardless.**
  Awaiting Beta's disposition.

_REV4 — 2026-08-02, post-D6.2-certification (`3561cda`). Statuses measured live,
not carried forward; see `docs/S172_PHASE_7_PREREQ_REPORT.md`. Update as items
land._
