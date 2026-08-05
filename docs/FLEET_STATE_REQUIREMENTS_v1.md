# FLEET STATE REQUIREMENTS — what a run actually demands, and whether the
# mechanisms agree

**Version:** 1.0
**Date:** 2026-08-01
**Author:** Team Alpha (Claude Code, on VM101 `192.168.3.177`, live tree at
`/home/michael/distributed_prng_analysis`, HEAD `fa866f2`)
**Type:** read-only investigation. **Nothing was changed, nothing was committed,
no check was loosened, no manifest was touched, no fault was injected.**

---

## 0. The falsifiable question and the answer

> What is the required fleet state for a run to proceed, and do the mechanisms
> enforcing it agree?

**They do not agree.** There is no single "required fleet state." There are
**five** independent notions of it, at three different granularities, on
**two disjoint address sets**, and which ones apply depends on the backend flag
and on whether the run went through WATCHER or the CLI:

| # | mechanism | granularity | blocks? | address set |
|---|---|---|---|---|
| 1 | P0.5 dataset preflight | **per node** (3) | **YES**, before dispatch | CT100 `.122/.156/.164` |
| 2 | legacy `test_connectivity` | **per node** (4, incl. localhost) | **YES**, `RuntimeError` | bare-metal `.120/.154/.162` |
| 3 | PWC ready gate `min_workers=24` — **⚠ CORRECTED 2026-08-04: the `24` does NOT reach the gate at HEAD; it runs at `1`. See §2.2** | **per GPU** | ~~**YES**, `RuntimeError`~~ → **effectively NO** above one ready worker | bare-metal `.120/.154/.162` |
| 4 | `preflight_check.check_gpu_health` | **per GPU** | **NO** — warning by design | bare-metal `.120/.154/.162` |
| 5 | `cluster-boot-notify` Telegram | **per GPU**, per host | **NO** — notify only, at boot | every host, its own conf |

Plus a sixth, inside the miner: `len(eligible) >= worker_pool_size`
(`miner/range_miner_coordinator.py:3715`), a **fleet-wide count of registered
worker daemons** with a default of **8** — which one rig alone satisfies.

**Alpha's suspicion is CONFIRMED.** P0.5's check is per node and dataset-only. A
rig that is up, reachable and correctly provisioned but with one dead RX 6600 XT
passes `fleet_preflight` without observing the GPU at all. What happens next
depends entirely on the backend, and in the miner's case the failure mode is
worse than "fails later in dispatch" — see §4.3.

---

## 1. The dataset preflight — what it requires and what it does not

### 1.1 What it requires

`fleet_preflight` (`miner/dataset_authority.py:904-934`) iterates every node and
raises `DatasetProvisioningError` if **any** node is not `PASS`. Every node is
checked first, so one operator round trip reports the whole fleet (`:920-923`);
the raise names the dataset and each failing node (`:927-933`).

The node list comes from `load_provisioning_nodes`
(`miner/dataset_authority.py:946-984`) → `dataset_provisioning.json` →
`datasets[dataset_logical_name="daily3"].nodes`. Live content (VM101, read
2026-08-01): exactly three entries — `rrig6600` `192.168.3.122`, `rrig6600b`
`192.168.3.156`, `rrig6600c` `192.168.3.164`, all `ssh_user: michael`. **The
CT100 addresses, not the bare-metal ones. localhost / VM101 is not a node in the
manifest.**

Per node, `verify_node_dataset` (`:704-825`) opens **one** SSH round trip running
`if [ ! -f … ]; then echo ABSENT; …; stat -c %s; sha256sum | cut -d' ' -f1`
(`:730-736`) — deliberately one trip so existence, size and digest describe the
same instant. `PASS` requires all of:

- the file exists at `frozen.path` (else `FAIL`, `:770-778`);
- `sha256 == frozen.sha256` (else `FAIL`, `:798-806`);
- `size == frozen.size_bytes` (else `FAIL` as a checker bug, `:808-818`).

Unreachable node, non-zero rc, timeout, or unparseable output → `UNAVAILABLE` /
`INCOMPLETE` (`:743-766`, `:780-795`), never a pass (VIR-5). All three non-PASS
states are refused identically at `:924`.

The digest is re-derived **on the target**, not on the sender (`:711-716`).

### 1.2 What it does NOT check

Nothing about compute. Specifically, `fleet_preflight` never observes:

- **GPU presence, count, health, or ROCm/CuPy state** — no `rocm-smi`, no
  `/dev/dri` enumeration, no CuPy import;
- **whether any worker daemon is running** on the node;
- **whether the node will be used by this run at all** — there is no
  `node_allowlist` parameter anywhere in `dataset_authority.py`;
- **free disk, free VRAM, load, temperature**;
- **that the manifest's node set matches `distributed_config.json`'s** — the two
  files describe disjoint address sets and nothing cross-checks them (§5.5);
- **localhost / VM101** — absent from the manifest, so the dataset on the machine
  running the optimizer is not verified by this gate at all.

### 1.3 Where the gate is invoked

`run_start_dataset_gate` (`miner/dataset_authority.py:1052-1109`) — resolve,
freeze, `fleet_preflight`, write provenance. Two production call sites:

- `window_optimizer.py:1473-1478` — in `main()`, after the backend mutex, before
  `MultiGPUCoordinator` is constructed. Failure is `parser.error`. Called with
  only `dataset_arg` and `run_label`, so `require_fleet` keeps its default
  `False`.
- `agents/watcher_agent.py:1490-1528` — per WATCHER step, for any step declaring
  a dataset param; failure returns `blocked_by: "dataset_authority_p0_5"`.

**Degradation note (not a bypass — reported, not changed):** `require_fleet`
defaults to `False` (`:1056`), and with it False an **absent manifest** is a
logged `UNAVAILABLE` warning and the run proceeds (`:1082-1095`); a manifest
declaring **zero nodes** likewise proceeds (`:1096-1102`). `dataset_provisioning.json`
is gitignored, so a fresh clone has no fleet definition and this gate silently
degrades to a log line. Both call sites take the default.

---

## 2. GPU-count verification — three of them, none of them one thing

### 2.1 `preflight_check.check_gpu_health` — compares to `distributed_config.json`, does not block

`preflight_check.py:293-361`. For each remote node, `ssh … rocm-smi | grep -cE
'^[0-9]+[[:space:]]'` and compare with `node["gpu_count"]` from
`distributed_config.json` (`_parse_nodes`, `:151-162`, default 12 if absent).
`gpu_count < expected` → issue `GPU_COUNT_MISMATCH`, `all_healthy=False`
(`:323-330`). `rocm-smi` failure / timeout / exception → `ROCM_SMI_FAILED` /
`TIMEOUT` / `ERROR`, also `all_healthy=False` (`:333-359`).

**It does not block, by explicit design, in two places:**

- `preflight_check.py:192` — `# 2. GPU health (warning only, not blocking)`, and
  `:206` — `result.checks_passed += 1  # Don't block on GPU warnings`. The issues
  are recorded via `add_warning`, never `add_failure` (`:200-205`).
- `agents/watcher_agent.py:1312-1318` — `# GPU count mismatches are warnings
  only`; only failures matching `["ssh", "unreachable", "ramdisk", "input", "not
  found"]` become a hard block.

`PREFLIGHT_HARD_FAILURES` (`agents/watcher_agent.py:433-439`) does contain
`"no gpus available"`, but `check_gpu_health` never emits that string — its
messages are `GPU_COUNT_MISMATCH` / `ROCM_SMI_FAILED` / `TIMEOUT` / `ERROR`
(`preflight_check.py:326,337,347,355`), formatted at
`agents/watcher_agent.py:203` as `GPU: <node> - <type>: <observed>/<expected>`.
So that keyword is dead against this producer.

**Two further scope limits:** localhost is excluded from the node list
(`preflight_check.py:152,156`), and the whole checker only runs inside
`WatcherAgent.run_step` (`agents/watcher_agent.py:1381` → `:1290-1333`). **A
direct `python window_optimizer.py …` run never executes it.** It is also
non-fatal on its own exception (`:1331-1333`, "proceeding anyway").

### 2.2 PWC ready gate — the real per-GPU, full-fleet, blocking check

> ## ⚠ CORRECTION 2026-08-04 — THE REFUSAL BELOW IS **NOT IN EFFECT AT HEAD**
>
> **Everything from "`persistent_worker_coordinator.py`:" to the end of this
> section is RETAINED AS WRITTEN and is SUPERSEDED. It described the intended
> chain, and it stopped one hop short of the hop that was broken.**
>
> **What is wrong.** This section traced `min_workers` from the CLI as far as the
> coordinator attribute — *"threaded at `window_optimizer.py:1510`
> (`pwc_min_workers`) and `:1617` (`min_workers`)"* — and concluded
> `23 < 24 → RuntimeError, run refused`. **There is one further hop, and it is
> missing.** The line that carried the coordinator attribute into
> `run_trial_persistent` —
>
> ```python
> min_workers       = getattr(coordinator, 'pwc_min_workers', 1),   # [S174] ready gate wiring
> ```
>
> — was added by **`ca06f8c`** (2026-05-08, *"S174: hard ready-gate
> (TB-approved)"*) and **deleted by `2389b61`** (2026-07-07) as one of three
> out-of-scope reverts in a commit whose stated purpose was the shared
> `PRNG_TYPE_ENCODING`. **It has never been restored.** `git log -S "min_workers"
> -- window_optimizer_integration_final.py` returns `c4698c8`, `ca06f8c`,
> `2389b61` and nothing after.
>
> **The effect at HEAD.** `run_trial_persistent` falls back to its own default
> `min_workers: int = 1` (`persistent_worker_coordinator.py:1649`), which reaches
> `PersistentWorkerCoordinator(min_workers=1)` (`:1685`, `:268`). The gate below
> therefore passes at **one** ready worker and logs `READY GATE PASSED`. The
> `RuntimeError` fires only at zero. **The `24` in the CLI help text is real; the
> `24` reaching the gate is not.** This has been true since 2026-07-07.
>
> The remaining `pwc_min_workers` reads in
> `window_optimizer_integration_final.py` (`:2068`, `:2106`, `:2454`) are the
> `n_parallel > 1` partition path, which passes a hardcoded `1` **deliberately**
> ("permissive for partition"). The deleted line was the gate's only live source.
>
> **⚠ It is NOT being restored — owner ruling, 2026-08-04.** The guard's purpose
> was to ensure **the whole cluster was being utilised**, during the PWC SSH/TCP
> era when a crashed worker's share was **picked up by the remaining workers**, so
> a run could silently proceed short-handed and merely take longer. **It was a
> UTILISATION check, not a correctness gate.** **RANGE-MINER does not have that
> shape:** stripes are claimed per worker against a ledger, not redistributed by
> slack-picking, so the failure mode the guard was written against does not exist
> on the certifying path. **PWC is retired from certifying authority.**
> Therefore: **not a defect, not restored, not a Phase-7 blocker.**
>
> **What this section is now good for:** it remains the correct description of the
> coordinator-side S174 gate, which is fully intact in
> `persistent_worker_coordinator.py`. Only the *threshold reaching it* is wrong.
> **Do not cite `23 < 24 → run refused` as current behaviour.**
>
> *Established by the full hunk-by-hunk audit of `2389b61` (Part A, 2026-08-04),
> which found three out-of-scope reverts in that commit. See §3 of the project
> facts skill.*

**The original analysis, as written 2026-08-01 — SUPERSEDED, retained:**

`persistent_worker_coordinator.py`:

- `:674-676` — `[S170-LIVE-GPU-PROBE]`: `node.gpu_count = min(declared_from_config,
  live_probe)`, then `pool = min(self.worker_pool_size, node.gpu_count)`.
  `pool <= 0` skips the node (`:678-684`). So the **live** count caps the spawn.
- `:826` — `_ready = self._tcp_wait_ready(expected=total_launched, timeout_s=180)`.
- `:864-901` — passes when `ready_count() >= self.min_workers`; on timeout it
  logs `READY GATE FAILED`, shuts the workers down, and **raises `RuntimeError`
  before any job dispatch**.
- `:1231-1244` — defence-in-depth recheck at the dispatch site, same threshold.

`min_workers` is supplied from `window_optimizer.py:1323`:

```
parser.add_argument('--min-workers', type=int, default=24,
                   help='[S162] Minimum workers reaching ready state before dispatch. '
                        'Default 24 = full 3-rig AMD cluster.')
```

threaded at `window_optimizer.py:1510` (`pwc_min_workers`) and `:1617`
(`min_workers`).

**This is the pre-existing full-fleet requirement, and it is per GPU.** 3 rigs ×
8 GPUs = 24. One dead RX 6600 XT → live probe returns 7 → 23 launched → 23 ready
→ `23 < 24` → `RuntimeError`, run refused. Michael's recollection is correct for
this path.

### 2.3 `cluster-boot-notify` — notification only, at boot only

Repo sources: `scripts/cluster_boot_notify.sh` (NVIDIA variant, `nvidia-smi -L |
wc -l` vs `EXPECTED_GPUS`, default 2, `:7-14`);
`scripts/install_boot_notify_amd.sh:41,65-78` (AMD, `EXPECTED_GPUS` default 8);
`scripts/update_boot_notify_v2.sh:19-31` (v2 — counts only PCI vendor `0x1002`,
excluding the Intel iGPU render node).

Properties that matter here:

- **It never blocks anything.** It POSTs to the Telegram API and `exit 0`
  unconditionally (`cluster_boot_notify.sh:20-25`; amd variant `:93-98`).
  Nothing in the run path reads its output or its exit status — no Python module
  references it (searched, §7).
- **It fires once, at boot.** `Type=oneshot`, `RemainAfterExit=yes`,
  `WantedBy=multi-user.target` (`install_boot_notify_amd.sh:104-117`). A GPU that
  drops off the bus after boot is invisible to it until the next reboot.
- **It does not read `distributed_config.json`.** Its expectation is
  `EXPECTED_GPUS` in `/etc/cluster-boot-notify.conf`, a per-host file written by
  the installer (`install_boot_notify_amd.sh:44-48`). Two independent sources of
  "how many GPUs should be here", never compared.

**Live verification, this session (read-only, VM101 → CT100s):**

| host | unit | deployed script | live AMD render nodes |
|---|---|---|---|
| `rrig6600` `.122` | `enabled` | md5 `320c02fb…`, v2 (`0x1002` filter) | 8 |
| `rrig6600b` `.156` | `enabled` | md5 `320c02fb…`, identical | 8 (+1 × `0x8086` iGPU) |
| `rrig6600c` `.164` | `enabled` | md5 `320c02fb…`, identical | 8 (+1 iGPU) |
| VM101 `.177` | `enabled` | 690 B NVIDIA variant | **1** (`nvidia-smi -L`) |

The unit is enabled **inside the CT100 containers**, which is a surface the brief
did not mention. The Proxmox hosts' copies could not be checked — see §6.

Two live observations worth recording:

- The **v2** script is what is deployed on the CTs, so the ninth render node
  (Intel iGPU, vendor `0x8086`, present on `.156`/`.164`) is correctly excluded.
  The v1 script still in `install_boot_notify_amd.sh:69` would have counted 9 and
  reported a false `MISSING` on two of three rigs.
- **VM101 reports a real mismatch that nothing acts on.**
  `/etc/cluster-boot-notify.conf` has `EXPECTED_GPUS=2`; `nvidia-smi -L` shows one
  RTX 3080 Ti. Every boot sends `🔴 GPUs: 1/2 MISSING`. This is consistent with
  the second 3080 Ti being unassigned/held by VM100 (CLAUDE.md §3), i.e. probably
  benign — but see §5.2, because `distributed_config.json` declares the same 2 and
  that number *is* load-bearing on the legacy path.

---

## 3. Worker registration — is the declared GPU count load-bearing?

**In the miner path: unused. The coordinator never learns a GPU count.**

- A worker is one process per GPU: `worker_id = f"{hostname}:gpu{gpu_id}"`
  (`miner/range_miner_worker.py:1213`), `--gpu-id` required
  (`:1471-1472`). Identity is `socket.gethostname()` (`:562-572`) — CT100 carries
  the rig's canonical hostname, so the hostname *is* the coordinator identity.
- `register_worker` (`miner/range_miner_coordinator.py:1752-1786`) validates
  **only** the advertised `seed_caps` (`_validate_caps`, `:1769`); an
  inconsistency **quarantines** the worker (registered-but-ineligible, durably
  recorded) rather than rejecting it. Nothing about GPU count is sent, requested
  or checked.
- `_serve_register` (`:3937-3975`) enforces one registration per socket and at
  most one live socket per `worker_id` (`:3949-3962`). A second socket for a live
  id is rejected as a duplicate. There is no expected-membership list: a worker
  from an unknown host is accepted, and `_resolve_node_config` (`:3490-3507`)
  falls back to the configured spool root when the hostname is not described in
  the allowlist dict — **it filters nothing.**
- The only count anywhere is
  `expected_workers = int(context.get("worker_pool_size", 1) or 1)` (`:3563`),
  compared against the number of registered non-quarantined connections at
  `:3715`.

**In the PWC and legacy paths: load-bearing.**

- `coordinator.py:436-446` — `create_gpu_workers` creates **exactly**
  `node.gpu_count` workers per node, straight from `distributed_config.json`
  (`:290`). No live probe on this path.
- `persistent_worker_coordinator.py:536` and `:676` —
  `pool = min(worker_pool_size, node.gpu_count)`; on the TCP path the declared
  value is first clamped by the live probe (`:674-675`), on the SSH path
  (`:536`) it is not.

**A unit mismatch worth flagging.** `--worker-pool-size` is documented as
*"Number of persistent workers to spawn **per rig** (default: 8)"*
(`window_optimizer.py:1326-1327`) and PWC uses it per node
(`persistent_worker_coordinator.py:536,676`). The miner takes the **same value**
(`window_optimizer_integration_final.py:1220` → `run_trial_miner(worker_pool_size=…)`,
`miner/range_miner_coordinator.py:4226,4321`) and uses it as a **fleet-wide**
registered-worker threshold (`:3563,3715`). A per-rig number is being used as a
cluster-wide one: with the intended 24-GPU fleet the miner's gate is satisfied by
**one rig alone** (8 workers), and would also be satisfied by 8 workers spread
arbitrarily across three rigs.

---

## 4. Stripe assignment with a missing GPU — the actual traced path

### 4.1 A GPU that never registers: no stripes are lost

`assign_stripes` (`miner/range_miner_coordinator.py:1840-1898`) partitions the
seed range into contiguous macro-stripes with no gap or overlap
(`partition_macro_stripes`, `:276-299`), then round-robins over
`compatible = [w for w in workers if self.can_assign_variant(w, family_name)]`
(`:1871`), `worker = compatible[i % len(compatible)]` (`:1890`).

The pool is whatever registered — **a GPU that never came up is simply not in the
list**. The seed range is still fully tiled; the surviving workers each take a
larger share. **No stripe goes unassigned and no seeds are skipped** as long as at
least one compatible worker exists. If *no* compatible worker exists, stripes are
recorded `pending` with a `refused_reason` (`:1878-1889`) and `serve_trial` turns
that into an explicit `fail_trial` (`:3722-3726`) rather than an indefinite
strand.

So the capacity loss is silent, but the **coverage** is not compromised. That is
the good half of the answer.

### 4.2 A GPU that dies after registration: the trial fails (constant phases)

Socket loss → `_drop_conn` (`:3904-3936`) removes the worker from
`wconn_by_worker`, `fs_by_worker`, `connections` and `registered`, and only if
the mapping still points at that socket (a fenced replacement that legitimately
rebound the id is not evicted).

Its claimed stripes stay `claimed` until the **compute lease** expires
(`compute_lease_timeout`, 300 s default at `range_miner_coordinator.py:4237`,
900 s in the D6 smoke; production value is whatever the coordinator attribute
carries — `window_optimizer_integration_final.py:1234`). Then
`process_lease_expiry` (`:2971-2981`) → `handle_stripe_failure(retryable=True,
lease_expiry=True)` → the Blocker-3 matrix (`:2892-2969`):

- **workflow phase 1/2 (constant skip) → `fail_trial` immediately** (`:2936-2938`);
- phase 3/4 (hybrid) first failure → `cleanup_attempt`, then **one** reassignment
  to a *different* eligible worker with `phase_degraded=1` and
  `staging_generation++` (`:2941-2965`); if no alternate exists, `fail_trial`
  (`:2946-2949`); second failure → `fail_trial` (`:2968`).

**TFM's sieve targets `java_lcg` constant-skip, i.e. phases 1/2.** So on the
production path, one GPU dying mid-stripe **fails the whole trial** — deliberately,
per the matrix. Not a reassignment, not a silent degradation.

### 4.3 The gap: a loss that drops the pool below the threshold **hangs**

Everything in §4.2 is reached only from inside this block:

```
miner/range_miner_coordinator.py:3714    eligible = _eligible()
:3715    if len(eligible) >= expected_workers and stage_idx < len(workflow_stages):
:3727        self.assign_stripes(...)
:3731        self._dispatch_pending(...)
:3737        self.process_lease_expiry(run_id, eligible)
```

`assign_stripes`, `_dispatch_pending`, `process_lease_expiry` and the stage
advance are **all** gated on `len(eligible) >= expected_workers`. And the serve
timeout is **unbounded by default** — `serve_timeout` defaults to `None`
(`:3566-3571`, `run_trial_miner` `:4335`,
`window_optimizer_integration_final.py:1260`), by an explicit Beta correction (a
real multi-billion-seed scan exceeds any fixed timeout).

Two consequences, both traced from source:

- **Before assignment:** if fewer than `expected_workers` daemons ever register,
  the loop accepts connections forever. No assignment, no dispatch, no error, no
  timeout. A silent hang. (The 15 s read deadline at `:3698-3711` only drops
  connections that never complete a *frame*; registered idle workers are exempt.)
- **After assignment:** if worker deaths drop `len(eligible)` below
  `expected_workers`, `process_lease_expiry` **stops being called**. The dead
  worker's stripes remain `claimed` with an expired lease that nobody processes.
  The trial neither completes nor fails. **The failure matrix in §4.2 is
  unreachable in exactly the situation it exists for.**

So the answer to "do stripes go unassigned, get reassigned, or hang" is: *none of
the above, then all three, depending on when the loss happens and whether it
crosses the threshold.* The dangerous case is a hang with no diagnostic.

---

## 5. The mechanisms side by side

### 5.1 Agreement / divergence matrix

| property | P0.5 dataset preflight | legacy `test_connectivity` | PWC ready gate | preflight GPU health | boot notify | miner `expected_workers` |
|---|---|---|---|---|---|---|
| anchor | `dataset_authority.py:904` | `coordinator.py:502` | `persistent_worker_coordinator.py:864` | `preflight_check.py:293` | `cluster_boot_notify.sh:9-14` | `range_miner_coordinator.py:3715` |
| granularity | node | node | **GPU** | GPU | GPU | worker daemon |
| source of truth | `dataset_provisioning.json` | `distributed_config.json` | live probe ∧ config | `distributed_config.json` | `/etc/cluster-boot-notify.conf` | `--worker-pool-size` |
| addresses | `.122/.156/.164` | `.120/.154/.162` (+localhost) | `.120/.154/.162` | `.120/.154/.162` | per-host, self | whoever dials in |
| full fleet? | yes (3/3 nodes) | yes (4/4 nodes) | yes (24/24 GPUs) | n/a (advisory) | n/a (advisory) | **no — 8, any host** |
| blocks a run? | **yes**, pre-dispatch | **yes**, `RuntimeError` | **yes**, `RuntimeError` | **no** | **no** | hangs, never blocks |
| when | run start (argparse) | first `execute_*` | worker startup | per WATCHER step | boot only | continuously |
| applies to CLI run? | yes | legacy/PWC only | PWC only | **no** | n/a | miner only |
| applies to miner run? | **yes** | **no** | **no** | no | n/a | yes |

### 5.2 Where they agree

- **P0.5 and `test_connectivity` agree on the shape of the requirement** — all
  declared nodes must answer, no partial fleet — and both fail closed before work
  starts. The manifest even states it: `"source_unreachable": "UNAVAILABLE; the
  run does not start with a partial fleet"`.
- **PWC's live probe and boot-notify agree on the number** — both derive 8 per rig
  from the live machine, and both currently see 8 (verified live, §2.3).
- **`check_gpu_health` and PWC agree on the comparator** (`distributed_config.json`
  `gpu_count`), differing only in whether the answer is binding.

### 5.3 Where they diverge

1. **Disjoint address sets, opposite verdicts in the current boot state.** P0.5
   checks the CT100 endpoints (UP now); `test_connectivity`, PWC and
   `check_gpu_health` check the bare-metal addresses (DOWN now — all three rigs
   are in Proxmox). Today, a legacy or PWC run through `window_optimizer.py` fails
   connectivity while the P0.5 gate that ran seconds earlier passed. Both are
   "full-fleet" requirements; they cannot both be satisfied in either boot state.
   The bare-metal addresses in `distributed_config.json` are deliberate and must
   not be "corrected" (CLAUDE.md §3) — which means the divergence is structural,
   not a typo, and is exactly what the deferred `rig_profile` selector is for.
2. **Granularity.** P0.5 says "node," PWC says "GPU." A node passes P0.5 with any
   number of working GPUs including zero.
3. **Bindingness for the same fact.** A missing GPU is a `RuntimeError` to PWC
   (`min_workers=24`) and a log line to `check_gpu_health` — the same observation,
   two verdicts, in the same run.
4. **Fleet size.** 3 nodes (P0.5) vs 4 nodes (`test_connectivity`, which includes
   localhost) vs 24 GPUs (PWC) vs 8 daemons (miner). No two mechanisms count the
   same population.
5. **Trigger.** Run start / first dispatch / worker startup / per-WATCHER-step /
   boot. Only P0.5 and the miner's counter observe the fleet *during* the run's
   own lifetime, and only P0.5 does so before anything is allocated.

### 5.4 Failure modes covered by NO mechanism

1. **A dead GPU on a miner run.** `--use-range-miner` reaches neither
   `test_connectivity` nor the PWC ready gate (§6.2). P0.5 passes (per node).
   `check_gpu_health` does not run outside WATCHER, and would not block anyway.
   Boot-notify fired hours ago. Result: the run proceeds at reduced capacity with
   no record, or hangs (§4.3). **This is the gap Alpha suspected, confirmed, and
   it is worse than "fails later."**
2. **VM101's own GPU count.** localhost is excluded from `preflight_check`
   (`:152,156`), skipped by PWC (`persistent_worker_coordinator.py:530-531`,
   "uses local sieve path, no persistent worker"), and absent from
   `dataset_provisioning.json`. Yet `distributed_config.json:6` declares
   `gpu_count: 2` while `nvidia-smi -L` shows **1** (verified live), and
   `coordinator.py:441` creates workers for `range(node.gpu_count)` — i.e. a
   worker for a GPU that does not exist. The only thing that notices is the boot
   notification nobody consumes.
3. **A GPU lost between boot and run.** Boot-notify is `oneshot`; `check_gpu_health`
   is advisory; P0.5 does not look. Only the PWC live probe would catch it, and
   only on the PWC path.
4. **A node in the provisioning manifest that is not in `distributed_config.json`
   (or vice versa).** Nothing compares the two files. Today they hold disjoint
   address sets by design and no mechanism observes that fact.
5. **Post-registration worker loss below the miner's pool threshold** — §4.3:
   no timeout, no lease processing, no notification. Nothing anywhere covers it.
6. **`dataset_provisioning.json` absent** (gitignored, so absent in any fresh
   clone): the P0.5 fleet check degrades to a warning and the run proceeds
   (`dataset_authority.py:1082-1095`), because both call sites take
   `require_fleet=False`. Recorded as `UNAVAILABLE` in provenance, never as clean —
   but nothing refuses.

---

## 6. The local single-GPU run — did P0.5 tighten or create?

### 6.1 What P0.5 does to a local run

`window_optimizer.py:1473` calls `run_start_dataset_gate` **unconditionally in
`main()`**, before the backend is selected and before `MultiGPUCoordinator` is
constructed. It ignores intent: there is no `--node-allowlist` argument in
`window_optimizer.py` (searched — none), and the two `MultiGPUCoordinator(…)`
constructions (`:756`, `:1079`) pass no `node_allowlist`, so the filter at
`coordinator.py:299-306` never engages from this entry point. The gate verifies
all three manifest nodes regardless of how much of the fleet the run intends to
use.

**So yes: a local single-GPU run launched through the `window_optimizer.py` CLI
now refuses while any rig is down or mis-provisioned.** The fail-closed reading in
the skill (§2.10) is confirmed at the call site.

### 6.2 Was a full-fleet requirement already there? Path by path

| entry point | pre-P0.5 fleet requirement | P0.5 effect |
|---|---|---|
| CLI, default/legacy backend | **YES** — `test_connectivity()` → `RuntimeError("Connectivity test failed")` if **any** configured node fails SSH preflight (`coordinator.py:502-520`, raised at `:1164`, `:1355`, `:1891`). All 4 nodes, no allowlist. | **tightened** — adds dataset identity per node, and moves the refusal earlier (argparse time, before any allocation) |
| CLI, `--use-persistent-workers` | ~~**YES, per GPU** — `min_workers` default **24** = "full 3-rig AMD cluster" (`window_optimizer.py:1323`), hard `RuntimeError` before dispatch (`persistent_worker_coordinator.py:884-901`)~~ **⚠ CORRECTED 2026-08-04 — the threshold has not reached the gate since `2389b61` (2026-07-07); it runs at `1`. See §2.2.** The CLI default is still 24; the gate is not. | **tightened** |
| CLI, `--use-range-miner` | **NO** — the miner branch returns from `run_bidirectional_test` (`window_optimizer_integration_final.py:1167-1265`) *before* any `coordinator.execute_distributed_analysis` call (`:1421`, `:1452`, `:1562`, `:1585`), so `test_connectivity` never runs; PWC is untouched. The only requirement was 8 registered daemons. | **created** |
| WATCHER step | **YES** — `check_ssh_connectivity` failures are hard-blocking (`preflight_check.py:190`, `agents/watcher_agent.py:1314-1323`) | tightened |
| direct harness call (e.g. `tests/smoke_s172_phase5_d6_zeus_single_gpu.py`) | none — it builds its own coordinator object (`:110-140`) and calls `run_bidirectional_test` directly | **still none** — the P0.5 gate lives in `window_optimizer.main()` and WATCHER, not in `run_trial_miner` |

### 6.3 Verdict

**Michael's recollection is confirmed, with one exception.** A full-fleet
requirement *was* coded in originally — twice: node-level (`test_connectivity`,
all configured nodes must answer) and per-GPU (`min_workers=24`, described in its
own help text as "full 3-rig AMD cluster"). **⚠ The per-GPU half is a HISTORICAL
statement and must not be read as current — the `min_workers=24` threshold has
not reached the gate since `2389b61` (2026-07-07); see §2.2.** For the legacy and
PWC paths **P0.5
tightened an existing constraint** rather than inventing one: it changed *what* is
required of each node (reachable → reachable **and** holding the frozen dataset at
a verified digest) and *when* the refusal happens (mid-run → before allocation).

**The exception is the miner path**, which is the current engine. There, P0.5
**created** a fleet requirement where none existed: pre-P0.5, `--use-range-miner`
had no per-node requirement at all, and today it is the one backend where the
dataset gate is the *only* fleet check standing.

A second-order point Beta may want in front of it: the P0.5 gate is now the
**strictest and earliest** check on every CLI run, and it is enforced against the
**CT100 addresses**, while the older full-fleet checks are enforced against the
**bare-metal addresses**. In the current boot state (all rigs in Proxmox) the new
gate passes and the old ones cannot. If Beta permits a partial-fleet run, the
question is not only "should P0.5 allow it" but "which of the five mechanisms is
the fleet definition" — because they do not currently name the same fleet.

---

## 7. Verification-integrity controls (VIR-1…6)

- **execution proof:** every claim carries a `file:line` read in this session;
  live checks show their commands and raw output (§2.3, §5.4-2). No claim is
  sourced from memory or a prior session.
- **clean control:** the three CT100 nodes each returned 8 AMD render nodes with
  vendor `0x1002`, and the two `.156`/`.164` extras were positively identified as
  `0x8086` (Intel iGPU) — the negative case that would have made a naive count
  read 9 was observed, not assumed.
- **fault-injection control:** **NOT PERFORMED — prohibited by the brief.** No GPU
  was disabled, no node was taken down, no manifest was altered. Every runtime
  behaviour in §4 is read from source, not observed. Claims about hang/reassign
  behaviour are therefore **code-path claims, not runtime claims.**
- **completion sentinel:** all six questions answered; report written to
  `docs/FLEET_STATE_REQUIREMENTS_v1.md`; no file outside `docs/` touched.
- **unavailable-observer behavior:** surfaces that could not be read are named
  below and reported as `UNAVAILABLE`, never as clean or as absent.
- **audit claim scope:** the *code paths* that determine required fleet state, plus
  the *live* boot-notify installation on VM101 and the three CT100 workers. It does
  **not** cover the Proxmox hosts, nor any executed run.
- **searched surfaces:**
  `miner/dataset_authority.py`, `miner/range_miner_coordinator.py`,
  `miner/range_miner_worker.py`, `window_optimizer.py`,
  `window_optimizer_integration_final.py`, `agents/watcher_agent.py`,
  `preflight_check.py`, `coordinator.py`, `persistent_worker_coordinator.py`,
  `distributed_config.json`, `dataset_provisioning.json`,
  `scripts/cluster_boot_notify.sh`, `scripts/install_boot_notify_amd.sh`,
  `scripts/update_boot_notify_v2.sh`, `scripts/install_boot_notify_pzeus.sh`,
  `s172_prelaunch_check.py`, `tests/smoke_s172_phase5_d6_zeus_single_gpu.py`;
  repo-wide `/bin/grep` for `fleet_preflight`, `run_start_dataset_gate`,
  `worker_pool_size`, `min_workers`, `gpu_count`, `test_connectivity`,
  `preflight_check`, `range_miner_worker`, `cluster-boot-notify`, `telegram`
  (`/bin/grep`, not the ugrep wrapper, so `*.json` was included).
  Live: VM101 `nvidia-smi -L`, `systemctl is-enabled`,
  `/etc/cluster-boot-notify.conf` (root:michael 640, readable);
  `michael@.122/.156/.164` — hostname, `systemctl is-enabled`, deployed script
  md5 + content, `/sys/class/drm/renderD*/device/vendor`.
- **unavailable surfaces:**
  - **Proxmox hosts `.121` / `.155` / `.163`** — `root@192.168.3.121` returns
    `Permission denied (publickey,password)` from VM101 (no root key auth; matches
    the known rig kernel-log access gap). The hosts' `cluster-boot-notify` units,
    scripts and `EXPECTED_GPUS` values are **UNVERIFIED**. Michael's report that the
    unit is enabled on all three hosts is **not confirmed by this audit** — what is
    confirmed is that it is enabled inside all three **CT100 containers**.
  - **`/etc/cluster-boot-notify.conf` on the CT100s** — root-owned, unreadable as
    `michael`, `sudo -n` denied. The CTs' `EXPECTED_GPUS` value is **UNVERIFIED**;
    the installer default is 8 (`install_boot_notify_amd.sh:41`) and the live count
    is 8, but the conf itself was not read.
  - **Bare-metal rig endpoints `.120` / `.154` / `.162`** — the rigs are booted into
    Proxmox, so the behaviour of `test_connectivity` / `check_gpu_health` /
    PWC against them in the *default* boot state was not exercised.
  - **`preflight_check_v1.1.0.py`** — differs from the live `preflight_check.py`;
    not analysed. The live import is `from preflight_check import PreflightChecker`
    (`agents/watcher_agent.py:87`), so the versioned copy is not on the run path.
  - **Runtime behaviour of every failure path in §4** — read, not executed.

---

## 8. What was NOT done

Per the brief: no fix, no bypass, no loosened check, no manifest edit, no commit,
no push, no pipeline launch, no fault injection, no GPU disabled. The gap in §4.3
and the question of whether a partial-fleet run should be permitted at all are
**Beta's to rule on.**
