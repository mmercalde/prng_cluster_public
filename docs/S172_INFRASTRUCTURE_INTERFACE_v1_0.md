# S172 Infrastructure Interface — Miner ↔ Rig Deployment Contract v1.0

**Purpose.** Pin the boundary between the S172 RANGE-MINER implementation
(Team Alpha / Claude) and the Proxmox rig migration work
(other team / Team Beta reconciled 2026-07-07). Both tracks proceed in
parallel. This document says what each side may assume about the other,
so they can converge cleanly at Phase 6 (miner-vs-PWC artifact identity
acceptance).

**Status.** Companion to `PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (spec
frozen at commit `1f6c0c5`) and
`PROPOSAL_Infrastructure_Reconciliation_S172_v1_0.md` (approved directionally,
LXC-vs-VM pending rrig6600c trial). Not a spec amendment. Zero contract
change. This is an interface-freeze doc so implementation on both sides
can start.

---

## 1. What the miner assumes about the rigs

The miner treats each rig as an opaque endpoint with these guarantees:

| Property | Value | Owned by |
| --- | --- | --- |
| Hostname (as seen by coordinator) | `rrig6600` / `rrig6600b` / `rrig6600c` | Proxmox team |
| IP (as seen by coordinator, in `distributed_config.json`) | `192.168.3.120` / `.154` / `.162` | Proxmox team |
| SSH access | `ssh michael@<ip>` succeeds with existing key | Proxmox team |
| GPU count reported by `rocm-smi` | 8× RX 6600 per rig | Proxmox team |
| Path to project | `/home/michael/distributed_prng_analysis` | Proxmox team |
| Path to venv | `/home/michael/rocm_env/bin/activate` | Proxmox team |
| Path to ramdisk (if writable) | `/dev/shm/prng/miner/` | Proxmox team |
| Fallback output path | `~/miner_output/` (writable, persistent) | Proxmox team |

> **Boot-selector note:** each rig is dual-boot. The `.120`/`.154`/`.162` IPs in
> the table above are the **default** bare-Ubuntu endpoints (what
> `distributed_config.json` holds). When a rig is booted into Proxmox instead, its
> worker endpoint is the CT100 address (`.122`/`.156`/`.164`). As of 2026-07-22 all
> three rigs are running under Proxmox. See `CLAUDE.md` §3.

**The miner code does not check whether these are backed by bare-metal, LXC,
or VM.** It reads them by convention. The Proxmox migration is free to
choose LXC-first or VM-first per the reconciliation §5 trial without
touching miner code.

## 2. What the Proxmox team assumes about the miner

The miner code makes no infrastructure demands beyond:

1. **`socket.gethostname()`** must return the rig's canonical name
   (`rrig6600` / `rrig6600b` / `rrig6600c`). The miner uses this in the
   READY handshake for coordinator-side worker identity; a container that
   reports its host's Proxmox hostname (e.g. `proxmox-rrig6600c-host`)
   instead will not be tracked correctly. **Both LXC-with-`--hostname` and
   VM-with-set-hostname satisfy this.** The reconciliation §6 IP scheme
   ("Containers retain original rig IP/hostname") already commits to this.

2. **`/dev/shm/prng/`** — miner output path is auto-detected. If
   `/dev/shm/prng/` is writable, miner writes to `/dev/shm/prng/miner/`.
   Otherwise miner writes to `~/miner_output/`. Neither location is
   fatal to acceptance — the compatibility criterion is byte-identity
   at the NPZ level, not path.

3. **`/dev/kfd` + `/dev/dri/renderD*` + `/dev/dri/card*`** — miner needs
   direct GPU access via ROCm/HIP. The Proxmox migration §4 (both LXC
   device binding and VM PCIe passthrough) covers this.

4. **No kernel-version dependency.** Miner uses only ROCm userspace
   (already validated by the codebase per the migration doc §3, and
   preserved by Phase 0 encoding work).

5. **Cold-start power-cap practice.** Both migration docs flag the
   historical 8-GPU cold-load crash risk (ROCm issue #5238). The miner's
   Phase 3 READY handshake will warm up GPUs sequentially or in pairs,
   not all 8 simultaneously. Implementation detail — not exposed as a
   deployment knob.

## 3. Phase-by-phase interaction

| S172 Miner Phase | Proxmox Migration Phase | Interaction? |
| --- | --- | --- |
| Phase 0 (encoding, complete, commit `2389b61`) | any | None |
| Phase 1 (scaffolding, this delivery) | any | None |
| Phase 2 (protocol) | any | None |
| Phase 3 (worker daemon) | trial or rollout | None *at code level* — worker daemon runs identically in bare-metal, LXC, or VM |
| Phase 4 (coordinator) | any | None |
| Phase 5 (NPZ contract wall) | any | None |
| Phase 6 (PWC-vs-miner artifact identity) | **potentially blocking** | see §4 below |
| Phase 7 (WATCHER soak + pool=8 regression) | rollout ideally complete | see §4 below |

## 4. Phase 6 / Phase 7 rig availability

Miner Phase 6 requires an acceptance run on real GPUs to prove PWC-vs-miner
NPZ byte-identity (v1.4.4 §11.A–§11.E). Phase 7 requires the 3-rig pool=8
regression (v1.4.4 §11.K).

**Miner-side accommodation:** the Phase 6 acceptance can be run on **any
2-of-3 rigs (16 GPUs)** during the rrig6600c trial window. The 3-rig / 24-GPU
requirement in v1.4.4 §11.K is deferred to Phase 7 only, and Phase 7 is the
last miner phase — expected to be reached only after both:

* rrig6600c trial resolves (§5 of reconciliation doc), AND
* trial-winner container/VM template is deployed to rrig6600 + rrig6600b

If the rig migration is still in progress at Phase 7 kickoff, miner Phase 7
runs on the highest-count available fleet (e.g. bare-metal rrig6600 +
bare-metal rrig6600b + LXC rrig6600c = mixed but 24 GPUs). This still
satisfies §11.K.

## 5. Configurable output path — LXC/VM/bare-metal all supported

The miner's `--miner-output-dir` CLI flag (Phase 1) and the manifest key
`miner_output_dir` (WATCHER v1.8.0) accept an explicit path or `null` /
`None` for auto-detect.

Auto-detect logic (Phase 5, to be implemented in `range_miner_npz_writer.py`):

```python
def resolve_output_dir(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    candidate = "/dev/shm/prng/miner"
    try:
        os.makedirs(candidate, exist_ok=True)
        # writable test
        probe = os.path.join(candidate, ".writable_probe")
        open(probe, "w").close()
        os.unlink(probe)
        return candidate
    except (OSError, PermissionError):
        fallback = os.path.expanduser("~/miner_output")
        os.makedirs(fallback, exist_ok=True)
        return fallback
```

This lets:

* **LXC** (per migration §4 bind of `/dev/shm/prng`): miner writes to
  ramdisk, matching Steps 2/3/5 behavior.
* **VM** (no ramdisk bind — see reconciliation §4 point 4): miner falls to
  persistent disk automatically.
* **Bare metal today**: same as LXC — `/dev/shm/prng/` is writable on
  every current rig.

Nothing on the Proxmox team's side needs to know about this. If the ramdisk
bind exists, it wins; if not, disk wins. Miner Phase 6 acceptance is
path-independent (compares NPZ bytes, not paths).

## 6. Open questions between the two tracks (for TB routing)

1. **Does the Proxmox team need any manifest / WATCHER change** to signal
   which container backs each rig? Current design: no. If the coordinator
   ever needs backend awareness (e.g. per-backend tuning), we add a
   `backend_type: bare|lxc|vm` field to `distributed_config.json` in a
   focused patch. Not needed for miner.

2. **Trial-window fleet cap.** During the rrig6600c trial, miner Phase 6+
   might need to explicitly exclude rrig6600c to avoid running against a
   half-migrated rig. Proposed convention: coordinator honors an env var
   `S172_EXCLUDE_RIGS=rrig6600c` (comma-separated) for the duration of
   the trial. Not a spec change; a runtime convention TB can approve when
   Phase 6 gets close.

3. **RAM headroom.** Reconciliation §5 gate 3 requires "peak host+guest
   memory during the run stays under the ramdisk 50% warn threshold." A
   miner Phase 6 run is exactly the kind of workload that stresses this.
   Proxmox team should include a miner-dispatched Step 1 job (Bayesian,
   50-trial, seed_count=50k, test_both_modes=true, pool=8) in their
   trial acceptance rather than only running a PWC job, so their RAM
   measurement covers the future default backend.

## 7. What both sides commit to

**Miner (Team Alpha / Claude):**
* Miner code contains zero infrastructure branches (`if lxc: ... elif vm: ...`).
  Worker identifies via hostname, output path via auto-detect, GPU access
  via the standard ROCm interface.
* Miner Phase 6 acceptance harness (v1.4.4 §11) is path-independent.
* If reconciliation §5 gate 1 fails (VM infeasible on TB360-BTC) and the
  Proxmox team commits to LXC-only, miner does nothing differently.
* If reconciliation §5 gate 1 passes and Proxmox team commits to VM, miner
  still does nothing differently — the fallback disk path handles it.

**Proxmox team:**
* Rig hostname + IP as seen by coordinator remain
  `rrig6600{,b,c}` + `192.168.3.{120,154,162}` regardless of container type.
* `/dev/kfd` + `/dev/dri/*` accessible to the miner worker process
  (already committed via migration §4).
* Preserve `/dev/shm/prng` writability if using LXC (already committed
  via migration §5 bind-mount).
* Include a miner-dispatched Step 1 job in the rrig6600c trial acceptance
  (§6.3 above).

---

**End of v1.0.** This document does not require TB sign-off before Phase 1
lands (Phase 1 is scaffolding-only and takes no infrastructure position).
It exists so the Proxmox team has a concrete interface to build against
without having to read the full 738-line miner spec.

Ship this alongside the Phase 1 code commit for reference.
