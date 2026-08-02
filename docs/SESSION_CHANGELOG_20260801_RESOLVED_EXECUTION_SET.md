# SESSION CHANGELOG — RESOLVED EXECUTION SET (Phase 7 blocker)

**Date:** 2026-08-01
**Box:** VM 101 `192.168.3.177` (`zeus-ubuntu-vm`), `michael`, venv `~/venvs/torch`
**Base HEAD:** `eed3904` (clean at session start)
**Authority:** `docs/CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md` (Team Beta fleet ruling)
**Evidence:** `docs/FLEET_STATE_REQUIREMENTS_v1.md`
**Status:** implemented and gated on this box. **NOT COMMITTED, NOT PUSHED, WATCHER NOT RUN.**

---

## 1. What was built, and where the resolver sits

A **frozen, run-scoped Resolved Execution Set** — one fleet authority per run,
resolved **after** backend and rig-profile selection and **before** dataset
verification, GPU verification, coordinator construction and dispatch.

**New module:** `execution_set.py` (repo root, stdlib-only at module scope).

| construction point | file:line | why there |
|---|---|---|
| **CLI** | `window_optimizer.py:1508` (`_freeze_xset(_resolve_xset(...))`) | Immediately after the backend mutex (`:1465-1476`) — the first line at which the backend is a fact — and before the P0.5 dataset gate at `:1571` (`run_start_dataset_gate`). Both `MultiGPUCoordinator` constructions (`:756`, `:1079`) are reached from strategy branches far below, so nothing is allocated yet. |
| **WATCHER** | `agents/watcher_agent.py:1475` (`self._ensure_execution_set(params)`); helpers `_step1_declared_params` `:1290`, `_ensure_execution_set` `:1316` | First statement of `run_step` after the script lookup — before `_run_preflight_check` (`:1480`, the GPU health check) and long before the P0.5 block (`:1568+`). Frozen **once per WATCHER process**, not per step: the fleet is a property of the run. |

Both call **the same function**, `execution_set.resolve_execution_set` — Beta's
requirement, and the reason G-SAME-RESOLVER is a check rather than a promise.

### What the set carries

`backend · rig profile · logical nodes and endpoints · worker/GPU identities ·
local-vs-remote · admission count · dataset-verification targets`, plus a
content digest (`set_id`) over exactly the deciding fields — so "identical
inputs produce an identical set" is testable.

### The rig-profile join table

**New file:** `rig_profiles_config.json`. Both topologies are retained (Beta
ruling 3); this is the only place that says which two addresses belong to the
**same logical node**, so a profile can be selected and all six consumers move
together. Named `*_config.json` deliberately: `.gitignore:41` ignores `*.json`
and `.gitignore:43` (`!*_config.json`) un-ignores this shape, so it **is**
tracked — the opposite of `dataset_provisioning.json`, which must stay ignored.

It is **not a third source of truth**. `gpu_count` / `gpu_type` still come from
`distributed_config.json`; the CT100 endpoints still come from
`dataset_provisioning.json`. The resolver **cross-checks every endpoint it
declares against whichever file describes it and fails closed on divergence** —
which closes `FLEET_STATE_REQUIREMENTS_v1` §5.4-4 (*"nothing compares the two
files"*) instead of adding a fourth copy to drift.

`default_profile` is `proxmox`, a **declaration, not a probe**: the rigs are in
Proxmox and the CT100 endpoints are what the Beta-ratified P0.5 authority
already verifies. Flip that one value (or pass `--rig-profile baremetal`) when
the rigs boot their default bare-metal target. **`distributed_config.json`'s
addresses are unmodified** — verified by gate, they are deliberate (CLAUDE.md §3).

### New CLI surface

```
--rig-profile {baremetal,proxmox}     which boot target this run addresses
--execution-set-nodes a,b,c           explicit PARTIAL set (declared, never inferred)
```

---

## 2. The six mechanisms became consumers. None was deleted.

| # | mechanism | read BEFORE | reads NOW | file:line |
|---|---|---|---|---|
| 1 | **P0.5 dataset preflight** | `load_provisioning_nodes()` → `dataset_provisioning.json` → CT100 `.122/.156/.164`. localhost never verified. | `execution_set.dataset_verification_targets()` — the set's nodes at the set's profile endpoints, **including localhost** | `miner/dataset_authority.py:1293-1319`; WATCHER's copy `agents/watcher_agent.py:1606-1614` |
| 2 | **legacy `test_connectivity`** | every node in `distributed_config.json`, bare metal `.120/.154/.162` | `_execution_set_nodes(...)` — set-filtered, endpoint re-pointed | `coordinator.py:112-125` (seam), `:313` (use) |
| 3 | **PWC ready gate** | every node in `distributed_config.json`, bare metal | same seam | `persistent_worker_coordinator.py:106-116` (seam), `:331` (use) |
| 4 | **WATCHER GPU health** | `_parse_nodes()` → `distributed_config.json`, bare metal | same seam | `preflight_check.py:58-68` (seam), `:176` (use) |
| 5 | **boot notify** | `EXPECTED_GPUS` in `/etc/cluster-boot-notify.conf` only — a second, uncompared number | additionally reports how this host is **declared** in the fleet definition (`rig_profiles_config.json` joined to `distributed_config.json`) | `scripts/cluster_boot_notify.sh:16-64` |
| 6 | **miner registration** | `_resolve_node_config` **filters nothing**; any host that dialled in became eligible | `_execution_set_admission(msg.worker_id)` → unlisted worker registers but is **quarantined** | `miner/range_miner_coordinator.py:163-176` (seam), `:4142` (use), `register_worker(admission_reason=)` `:1787,:1796` |

**Preserved deliberately, each verified by gate:**

- P0.5 keeps fail-before-dispatch, on-target digest re-derivation, and the
  Beta-ratified `UNAVAILABLE` / `NOT_APPLICABLE` vocabulary, with unknown still
  taking the over-constrained reading.
- WATCHER GPU health stays **non-blocking** (`check_all` still uses
  `add_warning` and still counts the check passed).
- boot notify stays **Telegram-only and `exit 0`** — the added block cannot
  change the GPU verdict, cannot fail, and degrades silently.
- The admission timeout (`ee0db06`), `serve_timeout=None`, `expected_workers`,
  `worker_pool_size` semantics and the Blocker-3 matrix are **untouched**.
- `distributed_config.json`'s addresses are **untouched**.

### Why an unlisted worker is quarantined rather than dropped

Quarantine is the mechanism this coordinator **already had** for
registered-but-ineligible (a capability inconsistency), it leaves a durable row
naming the refusal instead of an unexplained disconnect, and `_eligible()`
already filters on it — so "must not become eligible" is enforced by the
existing predicate rather than a new one. The capability check still runs and
both reasons are recorded together when both apply.

---

## 3. Q1 — the local-run refinement, delivered through the resolver

A run whose set is **one local node** verifies that node; a run whose set is
three rigs verifies three. **The set decides, not a flag.** P0.5 is not
special-cased and `require_fleet` is not weakened.

`remote_execution` is **derived from the set** (`any(not n.local)`), and is not
a caller-supplied parameter of `resolve_execution_set` at all — which is what
structurally prevents it from becoming a bypass. A set containing rigs reports
remote execution and cannot declare otherwise.

**One behavioural boundary was drawn deliberately, and it is worth Beta's
attention.** My first implementation let a frozen set supersede the provisioning
manifest entirely — and that **broke P0.5 gate 34** (`no coordinator, no
process, no dispatch`), because a miner-backed run with an absent manifest
proceeded to `fleet_preflight` instead of refusing. The gate caught it; the
implementation was corrected. The rule now is:

> The set decides **which nodes are verified**. It does not decide whether the
> provisioning authority boundary applies. An unusable manifest is still fatal
> for a miner-backed run **whenever the set contains a remote node**, before any
> node is contacted. The only case the set answers alone is a set with **no**
> remote node — there is no worker dataset to establish, so the remote
> provisioning record is genuinely `NOT_APPLICABLE`.

P0.5 is back to **38/38 with `--fleet`**, including gate 34 and gate 37's
fault-injection control.

---

## 4. Gate matrix

`tests/test_s172_resolved_execution_set.py` — **34/34, RESULT: PASS**.

| gate | checks | result |
|---|---|---|
| **G-RESOLVE-ONCE** | freeze-after-read refused; CLI AST ordering `resolve@1508 < freeze@1508 < p0.5@1571`; WATCHER AST ordering `ensure@1475 < preflight@1480` | 3/3 |
| **G-FROZEN** | a different set cannot replace the frozen one; identical re-freeze idempotent; profile map really rewritten mid-run and the frozen set does not move (control: a fresh resolve *does* differ) | 2/2 |
| **G-SAME-RESOLVER** | CLI and WATCHER invocations produce identical `set_id 9ae9cacbda20`; different inputs produce different ids; both entry points import the one resolver | 2/2 |
| **G-PROFILE** | both profiles resolve to different endpoints; consumers follow the profile (`.120` vs `.122`); unknown profile refused; a map contradicting `distributed_config.json` refused | 4/4 |
| **G-NO-INFERENCE** | unlisted worker registers but is **not** eligible; `rrig6600:gpu99` refused despite a listed hostname; no-set behaviour unchanged; capability quarantine intact and composes | 4/4 |
| **G-PARTIAL-EXPLICIT** | partial only when declared; unknown/empty declaration refused; **AST call-graph proof that the resolver performs no reachability probe** (down bare-metal nodes still resolve) | 3/3 |
| **G-CONSUMERS** | all six read the set; none deleted; §5 invariants (admission timeout, `serve_timeout`, config addresses) intact | 7/7 |
| **G-LOCAL** | one-node set verifies one node; **still refuses** when that node fails; `remote_execution` derived, not declarable | 3/3 |
| **G-MUTANT** | 5 consumer mutants, each reverting one consumer to independent resolution | 5/5 red + summary |

**G-MUTANT results** — every reverted consumer turned its own gate red:

| mutant | reverted to | gate that went red |
|---|---|---|
| `coordinator._execution_set_nodes` | pass-through (decide for yourself) | G-CONSUMERS/legacy |
| `persistent_worker_coordinator._execution_set_nodes` | pass-through | G-CONSUMERS/pwc |
| `preflight_check._execution_set_nodes` | pass-through | G-CONSUMERS/preflight |
| `range_miner_coordinator._execution_set_admission` | *whoever connects is eligible* | G-NO-INFERENCE/miner |
| `execution_set.active_execution_set` | always `None` | G-PROFILE/endpoints |

---

## 5. Execution proof — the set in run provenance, read back

Real `window_optimizer.main()` CLI path, tripwired **before** the optimizer so
nothing downstream of the gate ran (`tripwires: ['run_bayesian_optimization']`,
no pipeline launched). The only network activity was P0.5's own read-only
on-target digest check.

**Clean control A — healthy full fleet (miner backend):**

```
set_id           9ae9cacbda20d7a8cba100ab29284ecfab1ff6743abc9f657ae26a44c5001404
backend miner · rig_profile proxmox · partial False · remote_execution True
node_count 4 · gpu_count 26 · admission_count 8 · invoked_by window_optimizer.main
sources  rig_profiles_config.json, distributed_config.json, dataset_provisioning.json
  localhost  endpoint=localhost       worker_host=zeus-ubuntu-vm  gpus=2 local=True
  rrig6600   endpoint=192.168.3.122   worker_host=rrig6600        gpus=8
  rrig6600b  endpoint=192.168.3.156   worker_host=rrig6600b       gpus=8
  rrig6600c  endpoint=192.168.3.164   worker_host=rrig6600c       gpus=8
worker_ids  ['zeus-ubuntu-vm:gpu0', 'zeus-ubuntu-vm:gpu1', 'rrig6600:gpu0', …] (26)
fleet_status PASS · verified [(localhost PASS) (rrig6600 PASS) (rrig6600b PASS) (rrig6600c PASS)]
READ-BACK    provenance set_id == frozen set_id ✔
```

Written to `dataset_provenance/window_opt_java_lcg_bayesian_<pid>.json` under the
new top-level `execution_set` key.

**Clean control B — healthy one-node run (Q1):**

```
set_id 585c356e9518 · PARTIAL nodes=['localhost'] · gpus=2 · remote_execution False
fleet_status PASS · verified [(localhost PASS)]
READ-BACK    provenance set_id == frozen set_id ✔
```

This is the run that today refuses whenever any rig is down. It now resolves,
freezes, verifies exactly its own node, and proceeds.

---

## 6. Non-regression (§7)

| suite | result |
|---|---|
| P0.5 dataset authority `--fleet` | **38/38 PASS** (incl. gate 34, gate 37 fault-injection) |
| admission liveness | **16/16 PASS** |
| threshold-propagation | **5/5 PASS, 3/3 mutants killed** (G-MINER-UNCHANGED green) |
| Chapter1-P0 | **12/12 PASS** |
| D1.1 engine | **18/18 PASS** |
| D1.0 workflow | **8/8 PASS** |
| D4 serial backend | **8/8 PASS** |
| D5 `process_sharded` | **24/24 PASS, 18 mutants killed** |
| D6 3.A production adapter | **9/9 PASS, 16 mutants killed** |
| D6 threshold path | **17/17 PASS, 11 mutants killed** |
| D6.1 flush durability | **15/15 PASS, 8 mutants killed** |
| Phase 4 coordinator | **63/63 PASS** (after Gate 22 registration) |
| Phase 3 worker | **17/17 PASS** (CPU-only: validates the contract, not ROCm deploy-readiness) |
| Phase 6 — miner known-answer transfer gate | **8/8 populations exact-set equal, 8/8 faults rejected, SENTINEL PASS** |
| Phase 6 — Wall A / Wall B | **NOT RE-RUN** — certified and closed at `d98298c`; §5 forbids re-running Phase 6. Multi-rig certification, not a unit gate. |
| Resolved Execution Set | **34/34 PASS** |

**Gate 22 registration (§7: append, do not rewrite).** Four `.py` paths appended
with rationale at `tests/test_s172_phase4_coordinator.py:2209-2274`:
`execution_set.py`, `coordinator.py`, `preflight_check.py`,
`tests/test_s172_resolved_execution_set.py`. Nothing earlier was rewritten.
`persistent_worker_coordinator.py`, `window_optimizer.py`,
`agents/watcher_agent.py`, `miner/dataset_authority.py` and
`miner/range_miner_coordinator.py` were already registered; the new reason for
each edit is stated in the appended block. **G-MINER-UNCHANGED needed no new
registration** — both miner files I touched are already in its registered set,
and its threshold-token bleed check still passes.

---

## 7. Verification-integrity controls (VIR-1…6)

- **execution proof:** every claim above carries a `file:line` read this session
  or a gate that executed and printed its verdict. The resolved set is **read
  back out of the provenance file** and compared to the frozen object, not
  assumed. Consumers are demonstrated reading it (G-CONSUMERS) *and* demonstrated
  failing when they stop (G-MUTANT).
- **clean control:** a healthy three-rig run and a healthy one-node run both
  resolve, freeze and reach dispatch (§5, both controls executed on the live
  fleet). No-set-frozen behaviour verified byte-identical to pre-work for every
  consumer.
- **fault-injection control:** 5 consumer mutants (§4); an unlisted worker
  registering over a real socketpair through the real `_serve_register`; a
  profile map contradicting `distributed_config.json`; a mid-run rewrite of the
  profile map under a frozen set; a one-node set whose node fails.
- **completion sentinel:** `RESULT: PASS` (34/34) on the new suite; each
  non-regression suite prints its own sentinel, tabulated in §6.
- **unavailable-observer behavior:** anything not exercised is named below and
  reported `UNAVAILABLE`, never assumed clean.
- **audit claim scope:** the code paths that determine required fleet state, and
  their behaviour on this box against the live CT100 fleet. It does **not** cover
  a bare-metal-booted rig, the Proxmox hosts, a WATCHER pipeline run, or any
  executed sieve.
- **searched surfaces:** `execution_set.py`, `rig_profiles_config.json`,
  `miner/dataset_authority.py`, `miner/range_miner_coordinator.py`,
  `window_optimizer.py`, `window_optimizer_integration_final.py`,
  `agents/watcher_agent.py`, `preflight_check.py`, `coordinator.py`,
  `persistent_worker_coordinator.py`, `scripts/cluster_boot_notify.sh`,
  `distributed_config.json`, `dataset_provisioning.json`, and the §7 suites.
- **unavailable surfaces:**
  - **Bare-metal endpoints `.120`/`.154`/`.162`** — the rigs are in Proxmox, so
    the `baremetal` profile is proven to **resolve** and to re-point every
    consumer, but no consumer was exercised **against** a bare-metal-booted rig.
  - **WATCHER end-to-end** — `_ensure_execution_set` is proven by AST placement,
    by unit invocation, and by G-SAME-RESOLVER producing the identical `set_id`;
    **WATCHER itself was not run** (forbidden).
  - **boot notify on the rigs** — exercised on VM 101 with a stubbed `curl`
    (real script, real conf, exit 0, fleet line present, degrades silently with
    no map). The deployed CT100 copies are the **AMD v2** variant and were
    **not** modified or redeployed; their behaviour is **UNVERIFIED** and
    unchanged.
  - **Phase 6 Wall A / Wall B** — deliberately not re-run (§5).
  - **A real unlisted rig worker** — G-NO-INFERENCE is proven through the real
    `_serve_register` over a real socket, but no daemon was launched on a rig.

---

## 8. Findings and open items for Team Beta

1. **The provisioning-authority boundary (§3).** I narrowed the case where an
   unusable manifest is fatal to "the set contains a remote node". Beta should
   confirm that reading. The alternative — a local-only miner run still refusing
   for want of a *fleet* provisioning record — would leave Q1 undelivered in a
   fresh clone.
2. **`admission_count` is recorded, never imposed.** `expected_workers` and
   `min_workers` keep their existing values and meanings (§5 forbids changing
   them). Consequence: a one-node local **miner** run still admits against
   `--worker-pool-size` (default 8) while the set has 2 GPUs, so the operator
   must pass `--worker-pool-size 2`. Making the set *govern* admission is the
   obvious next step and is **not** authorised here.
3. **P0.5 now verifies localhost too.** The set includes VM 101, which
   `dataset_provisioning.json` never listed — closing
   `FLEET_STATE_REQUIREMENTS_v1` §5.4-2. It passes trivially today (the frozen
   path is resolved locally), but it is a new, real check.
4. **`distributed_config.json:6` still declares `gpu_count: 2` for localhost
   while `nvidia-smi -L` reports 1.** Untouched (out of scope), but the set now
   propagates that number into `worker_ids` and `gpu_count`, so it is more
   load-bearing than it was. Recommend a bounded correction.
5. **New files must be committed or the finalizer wall will red.**
   `rig_profiles_config.json` matches `.gitignore:43 !*_config.json`, so unlike
   `dataset_provisioning.json` it **does** appear in `git status --porcelain`
   and would dirty `repository_tree_clean` (`utils/run_finalizer.py:1589`) on a
   release-grade run. Same for `execution_set.py` and the new test.
6. **The profile default is an operator artifact.** Flipping the rigs back to
   bare metal requires editing `default_profile` (or passing `--rig-profile`).
   That is the design — a declaration, not a probe — but it is a boot-time
   operational step that now exists.
7. **WATCHER's default backend is PWC, and the set now says so out loud.**
   `agent_manifests/window_optimizer.json` declares
   `use_persistent_workers: true`, `use_range_miner: false`, `min_workers: 24`,
   so an unoverridden WATCHER run resolves `backend=pwc, admission_count=24`.
   Overriding `use_range_miner` alone leaves **both** backend flags true, which
   the child CLI's mutex (`window_optimizer.py:1465-1476`) rejects — pre-existing,
   not introduced here, but the set's `backend` field now makes it visible in
   provenance instead of only at the child's argparse. My resolver's precedence
   (miner > pwc > zmq) matches `main()`'s own branch order.
8. **WATCHER cannot pass `--rig-profile` / `--execution-set-nodes` to the child
   today** unless `agent_manifests/window_optimizer.json` declares them
   (step-scoped filtering drops undeclared keys). Not needed for defaults —
   both paths resolve the identical set — but a non-default profile under
   WATCHER needs that manifest line. Manifest untouched here.

---

## 9. Files changed

**New (untracked — must be staged explicitly, never `git add -a`):**

```
execution_set.py
rig_profiles_config.json
tests/test_s172_resolved_execution_set.py
docs/SESSION_CHANGELOG_20260801_RESOLVED_EXECUTION_SET.md
```

**Modified:**

```
window_optimizer.py                     resolve+freeze after the backend mutex; two CLI args
agents/watcher_agent.py                 _ensure_execution_set + _step1_declared_params; P0.5 targets
miner/dataset_authority.py              consumer seam; set-sourced targets; execution_set in provenance
miner/range_miner_coordinator.py        _execution_set_admission; register_worker(admission_reason=)
coordinator.py                          _execution_set_nodes seam in load_configuration
persistent_worker_coordinator.py        _execution_set_nodes seam in _load_config
preflight_check.py                      _execution_set_nodes seam in _parse_nodes
scripts/cluster_boot_notify.sh          fleet-definition line (non-blocking, exit 0 preserved)
tests/test_s172_phase4_coordinator.py   Gate 22 registration, appended
```

**Not mine — do not attribute to this deliverable:**
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` is also modified in this working tree.
That is the concurrent **Chapter 2 restore** session the brief flagged
(`CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md` §"Concurrency"). I did not
open or edit it. Stage the two deliverables separately.

**Not touched:** `distributed_config.json`, the admission timeout,
`serve_timeout`, `expected_workers`, `worker_pool_size` semantics, the Blocker-3
matrix, the published dataset / version file / pointer manifest, the
`process_sharded` import gate, the skip work, D6.2, D6.3, the scraper, and
Phase 6.

**Not done, by instruction:** no commit, no push, no WATCHER, no pipeline launch.
