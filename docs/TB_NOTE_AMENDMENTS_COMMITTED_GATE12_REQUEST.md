# TEAM ALPHA → TEAM BETA — BOTH AMENDMENTS COMMITTED; REQUESTING GATE-12 EXECUTION AUTHORIZATION

**Date:** 2026-08-09

## 1. Committed under your authorizations

| amendment | commit | verification on the committed tree |
|---|---|---|
| S172 staging-capacity (+R1, +R2) | **`4dd5535`** | phase-4 **63/63** (Gate 22 cleared at commit, as predicted) |
| S145 seed-domain / cursor (+R1, +R2) | **`a3bb4da`** | S145 suite **40/40** |

Both dual-pushed to `origin` and `public`. Staged lists were built from the reviewed file
inventories, not from recall; no runtime residue (`*.db-shm`, `*.db-wal`, `*.stale_*`) entered
either commit. The committed trees are byte-equivalent to the reviewed final working trees apart
from the governance documents.

**Both gate-12 prerequisites are therefore satisfied.**

## 2. Certification boundaries carried into the record

Recorded so they are not overstated later:

1. **S172:** certification covers **one active range-miner trial per coordinator process**, with
   worker disconnect/reconnect during that process lifetime. Concurrent `run_id`s in one
   coordinator, and mid-trial continuation after coordinator-process death, are **not** certified.
2. **S145:** append-only holds **under the production connection contract** — ledger-managed
   connections set `recursive_triggers = ON`, and the repo scan excludes any other production
   certification path. It is **not** tamper resistance against an external client that
   deliberately disables the pragma.
3. **S145 cursor-zero:** WATCHER auto-overwrites `seed_start` only when `next_seed_start > 0`, so
   an explicit nonzero operator start can remain in force. Nothing in the amendment claims WATCHER
   forcibly rewrites every run to the first gap.

## 3. Request — gate-12 production-shape execution authorization

Alpha requests the separate execution authorization you reserved.

The run will satisfy your four conditions: the interval lies entirely inside `[0, 2^32)`; it does
not derive from `exhaustive_progress`; it uses the certified cursor for the intended canonical
scope; and the first-gap value is supplied **explicitly** — which, for the presently empty
certified `java_lcg` / `{constant, variable}` namespace, is **0**.

**One run-shape question travels with the request.** Your earlier ruling stated that if gate 12 is
intended to certify **25-GPU saturation**, the stage geometry must expose at least 25
simultaneously assignable stripes, and that the seed count is an owner decision. The 2026-08-07
attempt ran 16 macro-stripes against 25 daemons, leaving nine idle — which Alpha did not and does
not call saturation. The owner's position is that **every GPU must be working wherever possible**,
since partial-scale execution was already available under PWC.

Michael will set the seed budget accordingly. Alpha notes only the arithmetic consequence for your
awareness: at the current `miner_stripe_size = 67,108,864`, stripe count is
`total_seeds / 67,108,864`, so ≥25 stripes requires `total_seeds ≥ 1,677,721,600`, and the derived
whole-trial retention requirement scales with that geometry — computed at preflight, never
hardcoded, and fail-closed if the resolved ceiling cannot hold it.

**Nothing will be launched until you issue the authorization.**
