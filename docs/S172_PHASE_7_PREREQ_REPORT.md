# S172_PHASE_7_PREREQ_REPORT.md

**S172 Phase 7 soak — §1 prerequisite measurement pass. No soak launched.**

**Scope:** §1 of `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md` only —
measure items 5 and 7, verify a 25-worker execution set can be frozen by
construction, and record the §6 checkpoint census. **No trial was run, no
pipeline was launched, nothing was committed or pushed.**

**`docs/PHASE6_PREREQS.md` was deliberately NOT edited** (owner amendment,
2026-08-02: Team Alpha is revising it as REV4 and a concurrent edit would
collide). The measured statuses in §1 below are therefore *recommendations for
REV4*, published here as evidence rather than written into that file. That doc
is at HEAD, unmodified, still carrying its 2026-07-25 all-`☐` column. **Anyone
reading the checklist before REV4 lands is reading five wrong statuses** — this
report is the correction of record until then.

**Host:** VM101 `zeus-ubuntu-vm` (`192.168.3.177`), as `michael`, venv
`~/venvs/torch` sourced before every command.
**Commit:** `3561cda924ee6343b146c1bd57556bb5bfae42f1`; the D6.2-certified base
`18a2419` is an ancestor (verified with `git merge-base --is-ancestor`), as is
D3.5's `46a3828`. `git pull` reported *Already up to date*.
**Worktree:** `git status --porcelain` empty, before and after every measurement.
**Date:** 2026-08-02.

**Completion sentinel: `PASS` for items 5 and 7. Item 6 is `INCOMPLETE` — see
§3.** Two prerequisites (1, 4) remain genuinely open, one of which (4) the soak
brief itself flags as a live risk.

---

## 1. Prerequisite statuses, measured — **recommended REV4 column**

This is the fold-in table. `docs/PHASE6_PREREQS.md` is unchanged at HEAD; every
status below is a recommendation for REV4, backed by the measurement named in
its row and detailed in §2–§5.

| # | item | doc says (2026-07-25) | **measured 2026-08-02** | recommended |
|---|---|---|---|---|
| 1 | second 3080Ti in VM101 | ☐ | one RTX 3080 Ti only; second still on VM100 | **☐ OPEN** |
| 2 | `michael → CT100` SSH | ☐ | `.122`/`.156`/`.164` answer under `BatchMode=yes`, no prompt | **☑ DONE** |
| 3 | `rrig6600` Proxmox migration | ☐ | `.120` DOWN, `.122` UP and reports `rrig6600` | **☑ DONE** |
| 4 | VM101 stable address | ☐ | `inet 192.168.3.177/24 … dynamic` — still DHCP | **☐ OPEN** |
| 5 | publication FS + clean-tree preflight | ☐ | 25/25 checks, 0 failures, incl. a fault-injection control | **☑ DONE** |
| 6 | code/environment parity + clock | ☐ | clock ✅; **code parity ✗** — mixed-vintage rig deployment | **◐ PARTIAL** |
| 7 | transport reachability + firewall | ☐ | all four miner flows accepted; no enforcing firewall anywhere | **☑ DONE** |

**Five statuses need changing, not four.** The soak brief predicted four; the
fifth is item 6, which the brief marked ☑ DONE on the strength of the clock
check alone. The clock half is satisfied. The code-parity half is not, and it
was not previously measured. Detail in §3 — this is the one finding that
contradicts the brief, and REV4 should not carry item 6 as ☑.

**For REV4, per-item evidence pointers:** item 1 → §1 GPU inventory · item 2 →
§1 (SSH, `BatchMode=yes`, no prompt, all three CT100s) · item 3 → §1 ping sweep ·
item 4 → §1 and §7 · item 5 → §2 · item 6 → §3 · item 7 → §4. Items 5 and 7 also
carry the two acceptance sub-lists their sections were written against, so REV4
can cite them without re-deriving.

### Ping sweep (item 3, both address sets)

```
192.168.3.120  DOWN     192.168.3.122  UP
192.168.3.154  DOWN     192.168.3.156  UP
192.168.3.162  DOWN     192.168.3.164  UP
```

All three rigs are on the Proxmox CT100 topology, which is what
`rig_profiles_config.json`'s `default_profile: proxmox` declares.
`distributed_config.json`'s bare-metal addresses remain deliberate (CLAUDE.md
§3) and were not touched.

### Live GPU inventory — the 25 is exact

| node | measured | source |
|---|---|---|
| VM101 | **1** | `nvidia-smi -L`; `torch.cuda.device_count()=1`; `cupy…getDeviceCount()=1` |
| rrig6600 (.122) | **8** | `cupy.cuda.runtime.getDeviceCount()` in `~/rocm_env` |
| rrig6600b (.156) | **8** | same |
| rrig6600c (.164) | **8** | same |
| **total** | **25** | |

No `HSA_OVERRIDE`/`GFX` environment overrides on any rig, matching the approved
node profile. **24 AMD + 1 NVIDIA = 25 live GPUs**, exactly the owner-mandated
figure.

---

## 2. Item 5 — publication filesystem + clean-tree preflight → **PASS (25/25)**

Method: the **real** `utils.run_finalizer.finalize_run` was invoked with
`output_root` set to a disposable `mktemp -d` directory created **beside** the
repository (`/home/michael/.d35-publish-preflight.XXXXXX`), proven to be on the
same `st_dev` as the worktree. The repository's own accumulator paths and root
aliases were never used as an output target, per PHASE6_PREREQS §5's binding
instruction. The rename sequence was not reimplemented in shell.

**Repository state (clean control).**

- `git status --porcelain` — **empty**, at `3561cda`. The previously-open
  worktree sub-items (`tmp/`, four `CLAUDE_CODE_BRIEF_S176-178*` files) are gone.
- `bidirectional_survivors_all.npz` — absent; `bidirectional_survivors_binary.npz`
  — absent. **Neither finalizer-owned alias exists as a regular file**, so
  `_bootstrap_root_aliases` would not fail closed.
- `.s172_accumulator` — absent under both `-e` and `-L` (the dangling-symlink arm
  matters and was checked).

**Filesystem.** `findmnt -T .` → `/  /dev/sda2  ext4  rw,relatime,errors=remount-ro`.
Not NFS/CIFS/SSHFS/FUSE. 916 G total, **427 G available**, 51 % used. Primitives
verified individually: dangling symlink creation ✅, same-filesystem directory
rename ✅, directory `fsync` ✅.

**Publication mechanics, against the production path.**

| check | result |
|---|---|
| fault-injection: `repository_tree_clean=False` | **REFUSED**, `RunParameterError` |
| generation 1 published | `gen-20260802T234735453770Z-…` `artifact_sha256=37784bc515141e3b…` |
| root alias `…_all.npz` bootstrapped | symlink → `.s172_accumulator/current/…`, resolves |
| root alias `…_binary.npz` bootstrapped | symlink → `.s172_accumulator/current/…`, resolves |
| `current` is a **relative** symlink into `generations/` | ✅ |
| generation 2 **atomically replaces** `current` | ✅ pointer moved gen1 → gen2 |
| generation 2 chains to generation 1 | ✅ `parent_generation_id` matches |
| `_validate_current_pointer` on the tip (production) | ✅ |
| `_validate_chain` to clean-start root (production) | ✅ |
| alias resolves through `current` to the live tip | ✅ |

**Cleanup.** The preflight root was removed; `git status --porcelain` re-checked
**empty**; no `.s172_accumulator` and no root alias leaked into the production
root. Verified after cleanup, not assumed from it.

**Verdict: item 5 ☑ DONE.** The D6 smoke's only remaining infrastructure blocker
is closed, and the finalizer's clean-tree wall is demonstrably enforcing rather
than merely unexercised.

---

## 3. Item 6 — code/environment parity + clock → **◐ PARTIAL, and this contradicts the brief**

The soak brief's §1 table records item 6 as **☑ DONE — "clock synchronized, NTP
active"**. That is true and was reproduced. It is also only half of the item's
acceptance list, and the other half was measured for the first time in this pass.

### Clock — satisfied

`NTPSynchronized=yes` on VM101 and on all three CT100s. Skew measured by
RTT-midpoint against each rig:

```
192.168.3.122  remote-local = +0.076s  (rtt 0.168s)
192.168.3.156  remote-local = +0.078s  (rtt 0.176s)
192.168.3.164  remote-local = +0.080s  (rtt 0.180s)
```

Sub-100 ms. Distributed logs and generation provenance correlate reliably.

### Code parity — NOT satisfied

Every `.py` under `miner/` and `utils/` compared by sha256, VM101 against each
CT100. Identical results on all three rigs: **17 files here, 16 there; 14
identical, 2 differing, 1 absent.**

| file | rig state | vintage identified from git history |
|---|---|---|
| `miner/range_miner_coordinator.py` | **differs** | matches `ee0db06` (§4.3 admission liveness) |
| `miner/dataset_authority.py` | **differs** | matches `8600e75` (P0.5 Q2 closure) |
| `utils/checkpoint_d6_2.py` | **absent** | post-dates the deployed bundle |

The modules the daemon's own logic lives in — `range_miner_worker.py`,
`range_miner_protocol.py`, `range_miner_npz_writer.py` — are **byte-identical**
on all four machines.

**The two stale modules are inside the worker's executing import closure.**
`miner/__init__.py:19` performs `from .range_miner_coordinator import
(DEFAULT_WORKER_ADMISSION_TIMEOUT, run_trial_miner)`, so importing the worker
imports the coordinator module, which in turn pulls in `dataset_authority`.
Verified directly on each rig by importing `miner.range_miner_worker` inside
`~/rocm_env` and reading `sys.modules`:

```
worker import closure: ['miner', 'miner.dataset_authority',
                        'miner.range_miner_coordinator',
                        'miner.range_miner_protocol', 'miner.range_miner_worker']
miner.range_miner_coordinator    imported_by_worker=True
miner.dataset_authority          imported_by_worker=True
utils.checkpoint_d6_2            imported_by_worker=False
```

The rig copies pre-date **`eff6616`** (admission binding), **`f7583bc`** and
**`18a2419`** (the D6.2 bounded repair this soak exists to exercise). The
coordinator that actually drives the run is VM101's and is current, so those rig
copies are *loaded but not driven*. That is a reason the soak would probably
still work — it is not the acceptance criterion, which asks that every worker
report the intended deployment bundle. A soak whose headline question is *"does
the S166 clear hold?"* should not be run from a fleet carrying pre-D6.2 modules
in its import path.

**Recommended before launch (owner action, not performed here):** redeploy
`miner/` and `utils/` to all three CT100s — `mkdir -p` the target directory
before any `scp`, absolute paths only (CLAUDE.md §2) — then re-run the digest
comparison. That flips item 6 to ☑ and removes the mixed-vintage caveat from the
soak report.

*(Ancillary: `rrig6600` carries a git worktree at `8e2f5bf` with 84 dirty
entries; `rrig6600b` and `rrig6600c` have no git repository at all. The rigs are
deployment targets, not working copies — the digest comparison above, not
`git rev-parse`, is the parity evidence.)*

---

## 4. Item 7 — transport reachability + firewall → **PASS**

The Phase-7 backend is the miner. Its control port is **5700**
(`miner/range_miner_protocol.py:41`; PWC's 5600 is deliberately distinct for
OS-level coexistence), and **workers dial in to the coordinator** —
`miner/range_miner_worker.py:1232` calls `socket.create_connection((host, port))`.
The load-bearing direction is therefore `CT100 → VM101:5700`, not the reverse;
the coordinator's own outbound need is SSH on 22.

Port 5700 was **free** on VM101 before the test. A real listener was bound on
`0.0.0.0:5700` and each peer connected, sent a payload and received `ACK`:

| flow | result | evidence |
|---|---|---|
| VM101 → `127.0.0.1:5700` (local GPU worker) | **PASS** | accepted from `127.0.0.1`, `reply=b'ACK'` |
| `rrig6600` → `192.168.3.177:5700` | **PASS** | accepted from `192.168.3.122`, payload `b'FROM-rrig6600'` |
| `rrig6600b` → `192.168.3.177:5700` | **PASS** | accepted from `192.168.3.156`, payload `b'FROM-rrig6600b'` |
| `rrig6600c` → `192.168.3.177:5700` | **PASS** | accepted from `192.168.3.164`, payload `b'FROM-rrig6600c'` |
| VM101 → `.122`/`.156`/`.164`:22 | **PASS** | `SSH-2.0-OpenSSH_8.9p1 Ubuntu-3` on all three |

The production bind is already externally reachable —
`window_optimizer_integration_final.py:1473` passes `miner_host` defaulting to
`0.0.0.0`, not loopback — so no launch-time override is needed for the remote
rigs to connect.

**Firewall.** `systemctl is-active ufw` returns `active` on VM101 and all three
CT100s, which is the *unit*, not an enforcing ruleset. `/etc/ufw/ufw.conf` reads
**`ENABLED=no`** on all four nodes; `/proc/net/ip_tables_names` is empty on VM101
and the `nf_tables` module is not loaded. No port is exposed beyond the cluster
LAN by anything measured here. Nothing was changed and `ufw` is not proposed —
Zeus has no firewall, per contract. The behavioural table above is the primary
evidence; the configuration reading only corroborates it.

**Scope limit, stated rather than glossed:** PWC/ZMQ transports were **not**
measured. Both are retired from certifying authority and this soak is
miner-backed, so item 7 is satisfied *for the backend this run uses*. A future
PWC run needs its own 5600 measurement.

*(Harness note: the script's `ss` poll printed "LISTENER DID NOT BIND" while the
listener was in fact bound and accepting — a grep artifact in the poll loop, not
a measurement. The four accepted connections are the evidence.)*

---

## 5. The 25-worker execution set — frozen, `admission_count=25`, **no clamp**

Resolved and frozen with `execution_set.resolve_execution_set(backend="miner",
admission_count=25)` followed by `freeze_execution_set(...)`. **No trial was
run.**

```
set_id                    = adcc2ae5714c98b0f232c62c1aa33ef43d9cd16eeb66c4f480a0b779d61af138
requested_admission_count = 25
admission_count           = 25
admission_clamped()       = False
rig_profile               = proxmox
partial                   = False
remote_execution          = True   (derived from the set, not declared)
nodes                     = localhost, rrig6600@.122, rrig6600b@.156, rrig6600c@.164
```

**Both counts are in `set_id`.** `content()` includes
`requested_admission_count` and `admission_count`, and the counter-case proves
the digest is sensitive to them: requesting 26 instead of 25 against the same
files yields `set_id = 2c2cc2da7f57…`, a different identity. So the run is
auditable as a 25-GPU run and cannot be mistaken for a 26-GPU run.

**The admission binding reaches the coordinator.**
`miner/range_miner_coordinator._execution_set_expected_workers` returns
`(25, 'execution_set(adcc2ae5714c)')` for a context `worker_pool_size` of 8, 25
or 26 alike — the context value is the request, the frozen set is the answer.
The log line it emits names the set and both numbers, which is the execution
proof the soak brief's VIR block asks for.

**Launch recipe:** pass `worker_pool_size = 25`. Both the CLI
(`--worker-pool-size`, defined `window_optimizer.py:1355` with a default of **8**,
consumed as the admission request at `:1545`) and WATCHER
(`agents/watcher_agent.py:1350`, the step-1 declared param) feed exactly that
value into `resolve_execution_set(admission_count=…)`, so this set_id is what a
correctly-configured soak will freeze. **The default is 8, so 25 must be passed
explicitly** — the same class of omission as `n_parallel`. `rig_profile` needs no
override; `proxmox` is the declared default.

### One qualification, and it is the brief's own concern in a subtler form

`admission_count` is **25 by construction and is not a clamp** — that part is
confirmed. But the frozen set carries **26 worker identities**, not 25:

```
worker_identity_count = 26
local identities      = ['zeus-ubuntu-vm:gpu0', 'zeus-ubuntu-vm:gpu1']
live local GPUs       = 1
UNBACKED identity     = zeus-ubuntu-vm:gpu1
```

The cause is `distributed_config.json`, which declares `gpu_count: 2` for
`localhost` — written for the two-3080Ti configuration, while only one card is
passed through today (item 1). `identity_count = 2 + 8 + 8 + 8 = 26`, so
`min(25, 26) = 25` and no clamp is recorded.

**This is not the failure `eff6616` closed.** The admission threshold is 25 and
25 real workers exist, so the 180 s window is meetable and no trial burns it
against an unmeetable threshold. Nor is there an execution consequence: no
production code path spawns miner workers by iterating `gpu_count` — workers are
launched explicitly with `--gpu-id N` (the D6 smoke pattern,
`tests/smoke_s172_phase5_d6_zeus_single_gpu.py:330`), so nothing will attempt to
start `zeus-ubuntu-vm:gpu1`.

What it does cost is exactly the auditability the brief §0.2 is about: the
provenance records `gpu_count: 26` alongside `admission_count: 25`, and
`_execution_set_expected_workers` logs *"set adcc2ae5714c has 26 worker
identities"*. A later reader sees a 26-identity fleet admitting 25 — which reads
like a 26-set that came up short, even though it is not.

**Recommendation (owner decision, not applied):** set `localhost.gpu_count` to
**1** in `distributed_config.json` to match measured hardware. Then
`identity_count = 25`, `requested = admission = 25`, `worker_identity_count = 25`
— a 25-worker set by construction in the full sense. Three caveats: it changes
`set_id`, so the value in this report is superseded if the edit is made; it must
be committed before launch, because item 5's clean-tree wall rejects a dirty
tree at finalization; and it is a distinct field from the bare-metal addresses in
that file, which stay untouched (CLAUDE.md §3). Deferred here — nothing was
edited outside `docs/`.

---

## 6. Checkpoint census (D6.3 free data, §6 of the soak brief)

Measured at `~/distributed_prng_analysis/.s172_checkpoint/`, **before** any soak:

| metric | value |
|---|---|
| run-id directories | **25** |
| total bytes (`du -sb`) | **266,835** (304 K apparent) |
| files (all depths) | **50** — exactly 2 per run directory |
| oldest | `zeus-ubuntu-vm-92770-1785440249` (2026-07-30 12:37) |
| newest | `zeus-ubuntu-vm-80774-1785633414` (2026-08-01 18:17) |

25 directories accumulated over ~2 days, averaging **~10.7 KB per run**.
**Nothing was deleted** — Beta's D6.3 constraint (never remove active,
unresolved or audit-retained state for exceeding an age or count threshold) was
respected; this pass only counted. Re-measure the same three numbers immediately
after the soak: 50 trials against this baseline is the datapoint that decides
whether D6.3 is a real blocker or a slow-burn item.

*(Note for the post-soak read: the 25 directories here are per-**run**, not
per-trial, so a 50-trial soak at `n_parallel=1` does not automatically imply 50
new directories. Report the delta, not an assumed rate.)*

---

## 7. What still blocks a launch

| item | state | consequence for the soak |
|---|---|---|
| **4 — VM101 on DHCP** | ☐ OPEN | The brief's own §1.2: a lease move mid-soak kills every worker's coordinator connection. A router-side reservation for `.177` is the zero-risk fix; confirm it survives a reboot **before** launch. |
| **6 — rig code parity** | ◐ PARTIAL | Mixed-vintage deployment: current worker code importing a pre-`18a2419` coordinator module on all three rigs. Redeploy `miner/`+`utils/` and re-verify digests. |
| 1 — second 3080Ti | ☐ OPEN | Not a blocker: the 25-GPU configuration is owner-mandated (§0.2) and the live inventory is exactly 25. Feeds the §5 qualification above. |

Items 2, 3, 5 and 7 are closed on measurement. Everything in §1–§6 above was
measured this session on live hosts; nothing was carried over from a prior
report.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every verdict here is backed by a command run this
  session on a named host — the item-5 preflight prints 25 individually labelled
  checks and a summary count; the item-7 listener logs the source address and
  payload of each accepted connection; the execution set was resolved *and*
  frozen, and `active_execution_set()` was read back to confirm the freeze took.
  No verdict rests on silence.
- **clean control:** repository state was captured before the item-5 preflight
  (`porcelain` empty, both aliases absent, no accumulator) and re-captured after
  cleanup; the preflight root was proven same-filesystem before use and proven
  removed after.
- **fault-injection control:** item 5 carries a real positive control —
  `finalize_run(repository_tree_clean=False)` was **refused** with
  `RunParameterError`, proving the clean-tree wall is enforcing. Item 7 carries
  the counter-case that port 5700 was measured **free** before the listener bound,
  so an accepted connection cannot be some other process. §5 carries the
  `admission_count=26` counter-case proving `set_id` is sensitive to both counts.
  No fault injection was performed against items 2, 3 or 6.
- **completion sentinel:** items 5 and 7 → **PASS**. Items 2, 3 → **PASS**.
  Item 6 → **INCOMPLETE** (clock passes, code parity fails). Items 1, 4 →
  **FAIL** (open, by measurement). This report as a whole: **PASS for its
  declared scope**, which is §1 measurement only.
- **unavailable-observer behavior:** nothing was unreachable. Had a rig not
  answered, the item would have been recorded `UNAVAILABLE`, not silently
  reduced to a two-rig result.
- **audit claim scope:** live fleet — VM101 plus the three CT100 workers — at
  `3561cda`, 2026-08-02. Claims about the *repository* are repo-scoped; claims
  about the *rigs* are host-scoped and were measured over SSH on the rigs
  themselves, not inferred from the clone (VIR-6).
- **searched surfaces:** tracked repository at `3561cda`; gitignored files
  (`dataset_provisioning.json` read directly, since `*.json` is ignored and
  invisible to repo-scoped search); git history (blob-matching to identify the
  rigs' code vintage); live host state on VM101 (`ip addr`, `nvidia-smi`,
  `findmnt`, `ss`, `systemctl`, `/proc/net/ip_tables_names`,
  `/etc/ufw/ufw.conf`); live host state on all three CT100s (`hostname`, `id`,
  `timedatectl`, `sha256sum` over `miner/`+`utils/`, `sys.modules` after a real
  worker import, `cupy` device count, `env`); the live `.s172_checkpoint/` tree.
- **unavailable surfaces:** the second 3080Ti (assigned to VM100, so its
  passthrough and IOMMU grouping could not be checked); the Proxmox hosts
  `.121`/`.155`/`.163` (no root key auth from VM101 — so host-side `dmesg`/
  `amdgpu` kernel-log surfaces, which the soak's abort criteria will need, were
  **not** exercised in this pass); the bare-metal rig profile (all three rigs are
  in Proxmox, so `.120`/`.154`/`.162` are down by boot state, not by defect);
  PWC/ZMQ transports (out of scope for a miner-backed soak).

---

**STOP.** Per the instruction, this pass ends here: no soak, no commit, no push,
no `watcher_agent.py --run-pipeline`. **The only file written is this report.**
`docs/PHASE6_PREREQS.md` is untouched at HEAD, for Team Alpha's REV4 — the §1
table above is the fold-in source.

---

# ADDENDUM — item 6 remediation: rigs brought to `18a2419` for `miner/` + `utils/`

**Task:** determine the established rig deploy mechanism, use it to bring all
three CT100s to `18a2419` for `miner/` and `utils/`, and prove it by module
provenance. **No commit, no push, no soak, no `watcher_agent.py --run-pipeline`.**

**Session context:** performed on VM101 as `michael`, venv `~/venvs/torch`.
Repository HEAD moved during this pass — Team Alpha landed `PHASE6_PREREQS` REV4
and committed this report, so HEAD is now `29f78d3`. That does not affect what
was deployed: `git diff 18a2419 HEAD -- miner utils` is **empty**, so the
`18a2419` content for these two paths is also `29f78d3`'s content. `18a2419`
remains an ancestor of HEAD. The VM101 worktree was clean before and after.

## A1. The established deploy mechanism — read from the hosts, not assumed

**There is one, it is documented, and it has two halves.** Nothing was invented.

**Half 1 — initial bring-up by `git clone` from the PUBLIC remote.** `.122`'s own
reflog is unambiguous:

```
8e2f5bf HEAD@{2026-07-30 19:18:08 +0000}: clone: from https://github.com/mmercalde/prng_cluster_public.git
```

`git remote -v` on `.122` → `origin  https://github.com/mmercalde/prng_cluster_public.git`.
Corroborated in the repository:
`docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md:19-20` — *"the required dataset is
Git-ignored, so `git clone` alone is not a complete rig deployment. Phase 6.0
discovered this on CT100: **the clone brought the code** but not `daily3.json`."*
CLAUDE.md §5 names the same mechanism for the code layer: *"code (self-healing
via git — one `git pull` from current no matter how far 101 pivots)"*.

**Half 2 — updates by targeted `scp` from VM101 on top of that clone.**
Documented in `docs/REMOTE_NODE_SETUP_CHECKLIST.md:127,133,139,179`
(`scp -r utils/ <rig>:~/distributed_prng_analysis/`), in CLAUDE.md §2's deploy
rule (*"`mkdir -p` any new remote dir BEFORE the scp that fills it"* — a rule
that exists because `scp` is the deploy path), in
`docs/SESSION_CHANGELOG_20260123.md:146-147` (`scp utils/survivor_loader.py …`),
and in the legacy `deploy_to_rigs.sh` / `deploy_to_remotes.sh`.

**The evidence that half 2 is what actually maintains the rigs:** `.122`'s git
status before this pass listed `miner/range_miner_worker.py`,
`miner/__init__.py` and `miner/range_miner_coordinator.py` as ` M ` — tracked
files **modified in place** relative to its own clone at `8e2f5bf`. That is the
fingerprint of files `scp`'d over a clone, and it is the direct cause of the
drift reported in §3: the worker was refreshed by `scp`, the coordinator module
was not.

**Rig-by-rig starting shape.**

| rig | `.git` | tree | notes |
|---|---|---|---|
| `.122` rrig6600 | **YES**, clone at `8e2f5bf` | 841 top-level entries | tracked miner files modified in place |
| `.156` rrig6600b | **NO** | 841 entries, 737 `.py` | full de-gitted copy of the same tree |
| `.164` rrig6600c | **NO** | 841 entries, 737 `.py` | full de-gitted copy of the same tree |

`.156`/`.164` are not partial deploys — they carry the whole tree, including
files that exist only as `.122`'s *modified* copies, so that copy came from a
working tree rather than from GitHub. Rig-side `~/.bash_history` is empty on all
three (the 2026-08-01 home-directory sweep), so the rigs hold no command record
of their own; the clone reflog and the repository documentation are the evidence.

**Mechanism selected: `scp` from VM101 (half 2), applied uniformly to all three.**
Reasons, stated so the choice is auditable rather than incidental:

1. It is the documented **update** path, and CLAUDE.md §2's deploy rule is
   written for it.
2. It works identically on all three rigs. `git pull` is available only on
   `.122`; using it there and `scp` on the other two would introduce a second,
   divergent method for one third of the fleet — the opposite of what this task
   is fixing.
3. It does not add an internet dependency at soak time. *(All three rigs can
   reach GitHub — each resolves `prng_cluster_public` HEAD as `3561cda` — but a
   deploy path should not acquire a dependency it does not need.)*
4. The bytes are staged from `git archive 18a2419 miner utils`, a pristine
   export, so provenance is tied to the commit rather than to a working tree.

**Not invented, not extended:** no new script, no systemd unit, no launcher,
no rig reconfiguration. The only rig-side mutations were the 17 `.py` files and
the removal of `miner/__pycache__` and `utils/__pycache__` (regenerable bytecode,
cleared so the post-deploy import unambiguously reads fresh source).

## A2. What was deployed

`git archive 18a2419 miner utils` → 17 `.py` files, staged on VM101 and digested
before transfer; the export was confirmed byte-identical to VM101's clean working
tree for those paths. Per rig:

```
ssh michael@<ip> mkdir -p /home/michael/distributed_prng_analysis/{miner,utils}   # §2 rule
scp -p <export>/miner/*.py michael@<ip>:/home/michael/distributed_prng_analysis/miner/
scp -p <export>/utils/*.py michael@<ip>:/home/michael/distributed_prng_analysis/utils/
ssh michael@<ip> rm -rf .../miner/__pycache__ .../utils/__pycache__
```

Absolute paths throughout, `mkdir -p` before each `scp`.

## A3. Before / after, per rig

All three rigs were reachable at both measurements. **No rig is UNAVAILABLE and
no rig was assumed current.**

The comparison is an **order-independent set** comparison of `path → sha256` over
every `.py` under `miner/` and `utils/`, against the 17-file `18a2419` export.
*(A naive `diff` of two `sort`ed digest listings shows a spurious delta: the rigs'
locale collates `miner/__init__.py` differently from VM101's. Same digests, same
paths — the set comparison below is the one that means anything.)*

### `192.168.3.122` — rrig6600

| | identical | differing | missing | extra |
|---|---|---|---|---|
| **before** | 14/17 | 2 | 1 | 0 |
| **after** | **17/17** | **0** | **0** | **0** |

Before: `miner/dataset_authority.py` differed, `miner/range_miner_coordinator.py`
differed, `utils/checkpoint_d6_2.py` missing. **After: exact match to `18a2419`.**

### `192.168.3.156` — rrig6600b

| | identical | differing | missing | extra |
|---|---|---|---|---|
| **before** | 14/17 | 2 | 1 | 0 |
| **after** | **17/17** | **0** | **0** | **0** |

Same three defects before; **exact match to `18a2419` after.**

### `192.168.3.164` — rrig6600c

| | identical | differing | missing | extra |
|---|---|---|---|---|
| **before** | 14/17 | 2 | 1 | 0 |
| **after** | **17/17** | **0** | **0** | **0** |

Same three defects before; **exact match to `18a2419` after.**

## A4. Proof by module provenance — the same method that detected the failure

Not a timestamp, not a clock, not a file listing. On each rig, in a **fresh
interpreter** under `~/rocm_env`: import `miner.range_miner_worker` — the same
import that exposed the drift — then read `sys.modules` and, for each module,
report the **resolved `__file__`** and the **sha256 of that exact file**.

**`18a2419` reference digests:**

```
0b9a7b86b0cf28858118b9b7c0b4646413e015431c94680520f1d563dc0cc55c  miner/range_miner_worker.py
70f8fbaa371cf59759fe9deb578cc997e91dc71338c9d1ce88cbb11f98d37a18  miner/range_miner_coordinator.py
365c8e3ee9abf80a532900b07e53740af08cd1f71bf7d908dd4db685bccf496d  miner/dataset_authority.py
c3faecaaa690800a1742bbb9178a9e595fcf94777fe526e27fc4a2360b180286  utils/checkpoint_d6_2.py
```

**Worker import closure, unchanged on all three rigs before and after:**
`['miner', 'miner.dataset_authority', 'miner.range_miner_coordinator',
'miner.range_miner_protocol', 'miner.range_miner_worker']` — importing the worker
still pulls in the coordinator module via `miner/__init__.py:19`. That structural
fact was never the defect; the *vintage* of what it pulled in was.

| module (as loaded) | before — all three rigs | after — all three rigs |
|---|---|---|
| `miner.range_miner_worker` | `0b9a7b86b0cf2885` ✅ | `0b9a7b86b0cf2885` ✅ |
| `miner.range_miner_coordinator` | `2b1527bf56271521` ❌ (`ee0db06`) | **`70f8fbaa371cf597` ✅** |
| `miner.dataset_authority` | `aa3d17923e7739b5` ❌ (`8600e75`) | **`365c8e3ee9abf80a` ✅** |
| `miner` (`__init__.py`) | `699e930e379d07a7` ✅ | `699e930e379d07a7` ✅ |
| `utils.checkpoint_d6_2` | **`ModuleNotFoundError`** ❌ | **`c3faecaaa690800a` ✅** |

Every resolved `__file__` is `/home/michael/distributed_prng_analysis/<path>` on
every rig — no shadowing copy elsewhere on `sys.path`. `rrig6600` / `rrig6600b` /
`rrig6600c` each identified themselves by `socket.gethostname()` in the probe
output and each ran `/home/michael/rocm_env/bin/python3`, so these are three
distinct machines, not one measured three times.

**The check that failed now passes by the same method.**

## A5. Post-deploy sanity

- VM101 worktree: **clean** (`git status --porcelain` empty) — the deploy only
  read from the repository.
- `cupy.cuda.runtime.getDeviceCount()` → **8** on each rig after the deploy. The
  ROCm environment is untouched: no userspace, driver or modparam change, so the
  frozen-rig rule holds.
- `.122`'s git clone now reports ` M miner/__init__.py`,
  ` M miner/range_miner_coordinator.py`, ` M miner/range_miner_worker.py`,
  `?? miner/dataset_authority.py`, `?? utils/checkpoint_d6_2.py` **relative to
  its own stale HEAD `8e2f5bf`** — expected, since two of those files did not
  exist at that commit. That clone is a deployment artifact, not an execution
  authority and not a certification surface; the digest and module-provenance
  evidence above is what governs. It is **not** the repository the finalizer's
  `repository_tree_clean` wall reads — that is VM101's, and it is clean.

## A6. Status change

**Item 6 code parity: OPEN → SATISFIED.** All three CT100s now load `18a2419`
bytes for every module in `miner/` and `utils/`, proven by module provenance in a
fresh interpreter. Combined with the clock half (already satisfied — NTP synced,
skew < 0.1 s), **item 6 should move from ◐ to ☑ in REV4**, on the strength of
§A3–A4; this addendum is the evidence.

Remaining Phase-7 prerequisite gap: **item 4** (VM101 still on DHCP) — unchanged
by this pass and still requiring a router-side reservation before a multi-hour
soak.

## Verification-integrity controls (VIR-1…6) — addendum scope

- **execution proof:** each rig's probe printed its own `socket.gethostname()`,
  its interpreter path, the full `sys.modules` closure and a per-module
  `__file__` + sha256, terminating in an explicit `PROBE_COMPLETE` sentinel. A
  truncated or silent probe is not a pass.
- **clean control:** the before-state was captured by the identical probe and the
  identical set comparison, on the same three hosts, minutes earlier — the two
  measurements differ only by the deploy between them.
- **fault-injection control:** `NOT_APPLICABLE` — this is a remediation and its
  verification, not a detector under validation. **Not written as `PASS`.** The
  before-state is the natural negative case: the same probe returned
  `ModuleNotFoundError` and two wrong digests on all three rigs.
- **completion sentinel:** `PASS` for all three rigs — `.122`, `.156`, `.164`
  each at 17/17. No rig was `UNAVAILABLE`; had one been, it would be recorded as
  such and **not** counted as current.
- **unavailable-observer behavior:** the deploy loop tests reachability before
  acting and prints `UNAVAILABLE — skipped, NOT deployed` rather than proceeding.
  It was not triggered.
- **audit claim scope:** `miner/` and `utils/` only, on the three CT100 workers,
  against `18a2419`. **No claim is made about any other path on the rigs** — the
  remaining ~720 `.py` files in those trees were not compared and may hold drift
  of their own.
- **searched surfaces:** `.122` git reflog / remote / status; rig-side
  `~/.bash_history` on all three (empty); VM101 `~/.bash_history` (2000 lines);
  repository `*.sh` deploy scripts; `docs/` deployment documentation;
  `docs/RUNTIME_DATASET_PROVISIONING_CONTRACT.md`; CLAUDE.md §2 and §5; live
  filesystem digests and live `sys.modules` on all three rigs.
- **unavailable surfaces:** rig shell history (wiped 2026-08-01, so the exact
  historical command that copied the tree to `.156`/`.164` is unrecoverable — the
  mechanism is established from the clone reflog, the in-place file
  modifications and the documentation, not from a recorded command); the Proxmox
  hosts `.121`/`.155`/`.163` (no root key auth from VM101); paths outside
  `miner/` and `utils/` on the rigs.

---

**STOP.** Nothing was committed or pushed; the soak was not launched. The only
repository change is this addendum.
