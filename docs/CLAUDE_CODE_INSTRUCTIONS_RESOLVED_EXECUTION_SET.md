# CLAUDE_CODE_INSTRUCTIONS_RESOLVED_EXECUTION_SET.md — REV1

**The Resolved Execution Set: one frozen fleet authority, six mechanisms become consumers.**

**A Phase 7 blocker.** Team Beta ruled that **none of the six existing mechanisms defines the
fleet**, and that a single run-scoped resolved set must.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. Implement
and iterate; you do **NOT** commit, push, or run WATCHER. STOP at the gate.

**The rigs are up and provisioned.** All three CT100s hold the frozen dataset, digest-verified
on target, and the repository is deployed and digest-verified on all three.

**Concurrency:** a Chapter 2 restore session may be running. It edits
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` only. No collision expected.

---

## 0. The problem

`docs/FLEET_STATE_REQUIREMENTS_v1.md` established there is **no single required fleet state**.
Six checks, three granularities, **two disjoint address sets**, and which apply depends on the
backend flag and whether the run came through WATCHER or the CLI:

| mechanism | granularity | addresses |
|---|---|---|
| P0.5 dataset preflight (`miner/dataset_authority.py:904`) | node | **`.122/.156/.164`** (CT100) |
| legacy `test_connectivity` (`coordinator.py:502`) | node | `.120/.154/.162` |
| PWC ready gate (`persistent_worker_coordinator.py:864`) | **GPU** | `.120/.154/.162` |
| WATCHER GPU health (`preflight_check.py:293`) | GPU | `.120/.154/.162` — **non-blocking by design**, WATCHER-only |
| boot notify (`cluster_boot_notify.sh:9-14`) | GPU | host-local, **Telegram-only, `exit 0`** |
| miner `expected_workers` (`range_miner_coordinator.py:3715`) | worker daemon | **whoever connects** |

**Three** point at bare metal, P0.5 points at the CT100s, and two name no fixed set at all. The
rigs are booted into Proxmox, so **P0.5 passes and the three bare-metal checks structurally
cannot** — they are checking machines that are not running.

## 1. What to build (Beta's ruling — follow it)

A **frozen, run-scoped Resolved Execution Set**, created **after** backend and rig-profile
selection but **before** dataset verification, GPU verification, coordinator construction or
dispatch.

It carries:

```
backend · rig profile · logical nodes and endpoints · worker/GPU identities
local-vs-remote · admission count · dataset-verification targets
```

**WATCHER and the CLI must invoke the same resolver.** Beta was explicit:

> **A partial set must be explicit and frozen before the run — never inferred from which workers
> happened to answer.**

And, on the miner's registration path: **unknown miner workers must not become eligible merely
because they connected.**

**Both topologies are retained** (Beta ruling 3). `.120/.154/.162` is the deliberate bare-metal
profile; `.122/.156/.164` are the Proxmox compute endpoints. **The selected profile decides which
endpoints enter the set.** `distributed_config.json`'s bare-metal addresses remain deliberate
and **must not be "corrected"** (`CLAUDE.md` §3).

## 2. The six become consumers

Each existing mechanism reads the resolved set instead of deciding for itself. **Do not delete
any of them** — they are being re-pointed, not retired.

**Preserve what each already gets right:**
- **P0.5's dataset preflight** — its fail-before-dispatch behaviour and the
  `UNAVAILABLE` / `NOT_APPLICABLE` vocabulary are Beta-ratified. `UNAVAILABLE` = a required
  verification was **attempted and could not complete**; `NOT_APPLICABLE` = this path never
  needed it. **Unknown keeps the over-constrained reading.**
- **WATCHER GPU health** stays **non-blocking** — that is deliberate, not a defect.
- **boot notify** stays Telegram-only and `exit 0`.
- **The admission timeout** (`ee0db06`) is unchanged: bounded admission, unbounded maintenance,
  `serve_timeout` stays `None`.

## 3. Q1 — the local-run refinement, now authorised *through this work*

A local single-GPU run currently **refuses while any rig is down**. Beta approved the refinement
**in principle** but required it come **through the shared resolver** — *not* by special-casing
P0.5 or weakening `require_fleet`.

So: a run whose resolved execution set is **one local node** verifies **that node**. A run whose
set is three rigs verifies three. **The set decides, not a flag.**

**`remote_execution=False` remains a topology statement, not a bypass** — a local run that still
drives the 26-GPU coordinator **performs remote execution** and must not declare otherwise.

## 4. Gates

| gate | asserts |
|---|---|
| G-RESOLVE-ONCE | the set is resolved **once**, before dataset verification, GPU verification, coordinator construction and dispatch |
| G-FROZEN | the set cannot change mid-run; a later topology or config change does not alter a run in progress |
| G-SAME-RESOLVER | WATCHER and the CLI produce the **identical** set for identical inputs |
| G-PROFILE | the selected rig profile determines the endpoints; both profiles resolve correctly |
| G-NO-INFERENCE | **a worker that connects but is not in the set does not become eligible** |
| G-PARTIAL-EXPLICIT | a partial set is accepted **only** when explicitly declared, never inferred |
| G-CONSUMERS | each of the six reads the set rather than deciding independently |
| G-LOCAL | a one-node resolved set verifies one node — and **still refuses** if that node fails |
| G-MUTANT | reverting any consumer to independent resolution turns its gate red |

**G-NO-INFERENCE and G-MUTANT are the load-bearing pair.** The first is the defect Beta named;
the second proves the others are not vacuous.

## 5. Out of scope

- **Do not delete or retire any of the six mechanisms.**
- **Do not modify `distributed_config.json`'s addresses.**
- Do not change the admission timeout, `serve_timeout`, `expected_workers`, `worker_pool_size`
  semantics, or the Blocker-3 matrix.
- Do not touch the published dataset, the version file or the pointer manifest.
- Do not do the `process_sharded` import gate (Beta-required, separate), the skip work, D6.2,
  D6.3 or the scraper.
- Do not re-run Phase 6 — it is **certified and closed** at `d98298c`.

## 6. Verification-integrity controls (VIR-1…6)

- **execution proof** — the resolved set appears in run provenance, **read back** rather than
  assumed. Consumers demonstrably read it.
- **clean control** — a healthy three-rig run and a healthy one-node run both resolve, freeze
  and dispatch.
- **fault-injection control** — G-MUTANT, plus an unlisted worker attempting to register.
- **completion sentinel** — explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`.
- **unavailable-observer** — anything not exercised is `UNAVAILABLE`, never assumed.
- **audit claim scope** — declare searched and unavailable surfaces; state which claims were
  observed on a rig.

## 7. Non-regression

D1.1 · D4 · D5 · D6 3.A · **D6-threshold 17/17** · D6.1 · **threshold-propagation 5/5** ·
Chapter1-P0 12/12 · **P0.5 dataset authority 38/38 with `--fleet`** · **admission liveness
16/16** · **Phase 6 gates** (`tests/phase6/`) · Phase 3 · **Phase 4 63/63**.

Gate 22 and `G-MINER-UNCHANGED` will see changed files — register with rationale, **append
rather than rewrite**, and **keep P0.5's strengthening intact** (it greps registered diffs for
threshold tokens).

## 8. Report

The resolver's construction point and why it sits there. The set's contents in run provenance,
read back. Per consumer: what it read before, what it reads now, `file:line`. The gate matrix
and the mutant results. Confirmation that none of the six was deleted, that both profiles
resolve, and that a one-node set still refuses on failure. Then STOP. **Do not commit.**
