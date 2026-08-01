# TEAM ALPHA → TEAM BETA — the fleet-state investigation, and a silent hang

**Re:** `docs/FLEET_STATE_REQUIREMENTS_v1.md`. Read-only; nothing changed, no check loosened,
no manifest touched, no fault injected.

This is the investigation Beta required before ruling on the Q1 refinement
(*"Alpha should complete its investigation of the pre-existing GPU-count gate before proposing
this refinement"*). **It found something outside Q1's scope that Alpha considers more urgent
than Q1: a silent, unbounded hang on the production path.**

---

## 1. The finding that outranks Q1 — §4.3

**If a worker loss drops the pool below `expected_workers`, the trial neither completes nor
fails.**

`assign_stripes`, `_dispatch_pending`, `process_lease_expiry` **and** the stage advance are all
gated on one condition:

```
miner/range_miner_coordinator.py:3715
    if len(eligible) >= expected_workers and stage_idx < len(workflow_stages):
```

And `serve_timeout` defaults to `None` (`:3566-3571`, `run_trial_miner:4335`,
`window_optimizer_integration_final.py:1260`) — **deliberately, by Beta's own earlier
correction**, because a real multi-billion-seed scan exceeds any fixed timeout.

Two consequences, both traced from source rather than inferred:

- **Before assignment:** if fewer than `expected_workers` daemons ever register, the loop
  accepts connections forever. No assignment, no dispatch, no error, no timeout. The 15 s read
  deadline (`:3698-3711`) only drops connections that never complete a *frame*; **registered
  idle workers are exempt.**
- **After assignment:** if deaths drop `len(eligible)` below the threshold,
  `process_lease_expiry` **stops being called**. The dead worker's stripes stay `claimed` with
  an expired lease nobody processes.

> **The Blocker-3 failure matrix is unreachable in exactly the situation it exists for.**

The matrix itself is correct and Alpha is not proposing to change it: a GPU dying mid-stripe on
constant phases 1/2 — **TFM's `java_lcg` production path** — is designed to `fail_trial`
immediately (`:2936-2938`). That is the right behaviour. It simply never runs once the pool
drops below threshold.

**Why Alpha rates this above Q1:** Q1 is an over-constrained gate that refuses to start. This
is a run that starts, stops making progress, and **reports nothing**. Under Phase 7 — 50 trials,
autonomous, WATCHER at the wheel — a single mid-run GPU loss produces an indefinite hang with no
signal for WATCHER to react to. Alpha submits this is a **Phase 7 blocker**.

**Alpha is not proposing a fix.** The obvious ones each have a cost Beta should weigh: a serve
timeout reintroduces the problem Beta's `None` default was chosen to avoid; processing lease
expiry below threshold changes when the failure matrix fires; and degrading `expected_workers`
mid-run would silently alter the execution set, which Beta's Q1 ruling prohibits in the adjacent
case.

## 2. The good half — coverage is sound

A GPU that **never registers** costs capacity, not correctness. `assign_stripes`
(`:1840-1898`) partitions the seed range into contiguous macro-stripes with no gap or overlap
and round-robins over whoever registered — the range is still fully tiled, the survivors each
take a larger share, **no stripe unassigned and no seeds skipped.** If *no* compatible worker
exists, stripes are recorded `pending` with a `refused_reason` and `serve_trial` turns that into
an explicit `fail_trial` (`:3722-3726`), not a strand.

So the silent case is **capacity**, and the dangerous case is **a threshold crossing.**

## 3. Q1 — Beta's principle is right, and the investigation shows why

Beta ruled before seeing this:

> *Fail-closed verification should ultimately apply to the run's **resolved execution set**…
> The same execution-set object must govern GPU discovery, dataset verification, and dispatch.*

**The investigation is direct evidence for that principle.** There is no single required fleet
state today. There are **five checks at three granularities on two disjoint address sets**, and
which apply depends on the backend flag and whether the run came through WATCHER or the CLI.

| mechanism | granularity | addresses |
|---|---|---|
| P0.5 dataset preflight (`dataset_authority.py:904`) | node | **`.122/.156/.164`** (CT100) |
| legacy `test_connectivity` (`coordinator.py:502`) | node | **`.120/.154/.162`** (bare metal) |
| PWC ready gate (`persistent_worker_coordinator.py:864`) | **GPU** | `.120/.154/.162` |
| preflight GPU health (`preflight_check.py:293`) | GPU | `.120/.154/.162` |
| boot notify (`cluster_boot_notify.sh:9-14`) | GPU | per-host, self |
| miner `expected_workers` (`range_miner_coordinator.py:3715`) | worker daemon | whoever dials in |

**Five of six point at bare metal; P0.5 points at the CT100s.** The rigs are currently booted
into Proxmox, so **the new gate passes and the older ones structurally cannot** — they are
checking machines that are not running. This is not a P0.5 defect; **P0.5 is the only mechanism
that was updated for the Proxmox migration.**

Before any Q1 refinement, the prior question is Beta's own: **which mechanism defines the
fleet?** They do not name the same one.

Three further findings bearing on the refinement:

- **GPU verification is three different things.** `preflight_check.check_gpu_health:293` compares
  live `rocm-smi` against `distributed_config.json` `gpu_count` and is **non-blocking by explicit
  design** (`:192`, `:206`) — and only runs inside WATCHER, never on a CLI run.
  `cluster-boot-notify` is **Telegram-only, `exit 0`, oneshot at boot**, and reads
  `EXPECTED_GPUS` from a per-host conf, **not** `distributed_config.json`. The only *blocking*
  per-GPU check is PWC's ready gate (`min_workers` default 24 — *"full 3-rig AMD cluster"*),
  which the miner path does not reach.
- **The declared GPU count is unused on the miner path.** `register_worker:1752` validates only
  `seed_caps`. The miner uses `--worker-pool-size` (documented "per rig", default 8) as a
  fleet-wide registered-worker threshold (`:3563`) — **one rig alone satisfies it.**
- **A rig with 7 of 8 GPUs passes the dataset preflight.** It is per node; it never observes a
  GPU, a worker daemon, or whether the node will even be used.

## 4. Q1 — did P0.5 tighten or create?

**Michael's recollection is correct, with one exception.** A full-fleet requirement **was** coded
in originally, twice: `test_connectivity` → `RuntimeError` on any unreachable node
(`coordinator.py:502-520`, raised `:1355`), and `min_workers=24` per GPU. **P0.5 tightened
those.**

**But `--use-range-miner` returns from `run_bidirectional_test` before any
`coordinator.execute_*` call, so it reaches neither.** For the current engine, **P0.5 created the
constraint** — the miner path had quietly escaped the original one.

## 5. What Alpha is doing, and what it is not

**Doing now:** Beta's Q2 closure condition — a miner-backed run hard-fails on a missing,
unreadable, invalid or empty provisioning manifest, with `NOT_APPLICABLE` for non-miner paths
and the negative gate Beta specified.

**Not doing:** no Q1 bypass, no local-run relaxation, no change to any of the six mechanisms, no
touch to `:3714-3737`. Beta: *"Until then, the safe over-constrained behavior remains."*

## 6. Rulings requested

1. **The §4.3 hang.** Alpha's assessment: **Phase 7 blocker**, more urgent than Q1. Does Beta
   agree, and does it want a fix scoped before bounded Phase 6 or after? Alpha has deliberately
   proposed no remedy — each obvious one trades against a constraint Beta has already ruled on.
2. **Which mechanism defines the fleet?** Six mechanisms, two disjoint address sets, no shared
   definition. Beta's "resolved execution set" is the right answer in principle; **the prior
   question is which of the six becomes the authority** and what happens to the other five.
3. **Should the five bare-metal-addressed mechanisms be updated for Proxmox, retired, or left
   as-is** given the rigs may boot either way? Alpha notes `distributed_config.json`'s
   bare-metal addresses are **deliberate** and ruled not-a-bug — so the divergence may be
   intended at the config layer and unintended at the check layer.
4. Confirm the Q1 refinement remains **unauthorized** pending Beta's answer to (2). Alpha
   assumes yes.

## 7. VIR declaration

Read-only; **no fault was injected** — §4 is a code-path trace, not an observed runtime failure,
and is labelled as such. `root@.121` is unreachable from VM 101, so the Proxmox hosts'
boot-notify units are **UNVERIFIED**; what was confirmed live is that the unit is enabled inside
all three CT100s (v2 script, vendor-filtered, 8 AMD GPUs each). Scope: repo + VM 101 + the three
CT100s.
