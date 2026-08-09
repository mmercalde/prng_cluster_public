# CLAUDE CODE INSTRUCTIONS — STAGING-CAPACITY AMENDMENT + `elapsed_s` PERSISTENCE

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD `c7058d8`.
`source ~/venvs/torch/bin/activate` before every test. Long suites:
`python3 -u <suite> | tee /tmp/<name>.log` — never `| tail`.

**Authority — two Beta rulings:**
- *"S172 GATE-12 STAGING-CAPACITY DEADLOCK"* (2026-08-07) — the amendment, §§2-6.
- *"STEP-1 SEARCH GEOMETRY…"* (2026-08-08) **R4** — persist `elapsed_s` **inside this amendment**,
  narrow scope.

**Hard constraints:** no commit, no push, **no pipeline launch, no fleet launch, no port 5700
bind**. Gate 12 and the Phase-7 soak are HELD. Do not touch worker code, seed caps, stripe
geometry, or `gate_s172_prod_shape.py`. F1–F5 and all existing gates must stay green and
assertion-unchanged (verify programmatically, the round-2 method). If a fix appears to need a
file outside scope, STOP and report.

**Base verification, run first:** `git status --porcelain` clean; `git log --oneline -1` =
`c7058d8`; `python3 -u tests/test_s172_staging_backpressure.py` → **35/35**. Any mismatch: stop
and report.

---

## PART 1 — THE CAPACITY CONTRACT (Beta staging ruling §§2-5)

### 1.1 Release lifecycle — Beta Option C, binding

Beta rejected both incremental Phase-5 assembly and retain-everything-forever. The ruled
lifecycle:

```
publish manifests → retain staged spools → TrialCommit
   ├─ assembly FAILS   → retain manifests, files, reservations; same event_id stays retryable
   └─ assembly SUCCEEDS→ ack/release every trial-owned reservation exactly once,
                          delete the staged files, durable cleanup status,
                          capacity available to the next trial
```

**Do NOT implement incremental assembly or mid-trial ack** (Beta §2.1 — it would reopen
complete-phase-set validation, duplicate enforcement, commit retryability, spool-repair retry,
partial-global-state ownership, and process-sharded equivalence; D1.1's retry contract depends on
retention after a failed commit).

**Release occurs only after the sink's successful commit return.** `commit_trial()` currently
releases nothing; `abort_trial` already iterates and releases — read it as the shape reference.
The existing event-id ack/release machinery may be reused if its semantics are correct
(`ack_by_event_id` / `release_after_ack` / `ack_shard` currently have **zero production callers**).

**Why this matters beyond the observed deadlock:** the failed run requested **eight** trials.
Even sized for one whole trial, a success path that frees nothing ratchets across trials. That is
the second defect Beta identified and it is the reason G-SEQUENTIAL-TRIAL-REUSE is mandatory.

### 1.2 Derived retention bound (Beta §3)

Within a trial, files cannot be released before successful commit — therefore **the file
high-water must retain the entire planned trial by construction.**

**Do NOT hardcode 1,028.** That is one assignment's exact count. Use the conservative doctrine
already accepted for S172-BP:

```
trial_retention_files_required =
    Σ over every planned workflow phase
        Σ over every planned stripe
            max( expected_substripes ) over workers eligible for that stripe/phase
```

Computed from the frozen execution set, stripe geometry, phase-specific caps and the enabled mode
set. The conservative number may exceed the observed 1,028 — Beta: *"That is correct. Safety must
not depend on reproducing the lucky assignment."*

**Default contract:** `staging_high_water_files = None` **means derive**. An explicit operator
value stays legal, but `operator value < derived required count` must **FAIL CLOSED BEFORE THE
FIRST STRIPE IS DISPATCHED** — a warning is explicitly insufficient.

Note the current committed value is **4096** at three sites (`8bbe79e`) — the dataclass default
and both `build_*` factory signatures. That was a run-enabling wall-move; this amendment replaces
its meaning with the derived bound. Handle all three sites.

### 1.3 Route both high-waters (Beta §4)

Wire `staging_high_water_files` **and** `staging_high_water_bytes` through the complete governed
path — the same route the other four staging controls now use:

```
manifest default_params → args_map → window_optimizer argparse
  → coordinator attrs / integration → run_trial_miner → build_coordinator → CoordinatorConfig
```

No `getattr(..., 512)`-style silent fallback may remain the only production source.

**Byte bound (Beta §4.1):** the file count is deterministically derivable from geometry. **Do not
invent a pre-dispatch byte estimate from observed average spool sizes.** If an authoritative
maximum shard-size contract exists, a conservative whole-trial byte bound may be derived from it;
**if none exists, report that fact explicitly** and leave the byte ceiling runtime-enforced,
protected by the §1.4 timeout. Beta: *"A guessed byte bound is worse than an explicitly
runtime-only bound."*

### 1.4 Capacity timeout must observe the executor (Beta §5)

`staging_capacity_timeout_expired()` currently measures the oldest **paused connection**; a
staging worker looping in `StagingBackPressure` with no connection paused waits forever — the
failed run sat ~19 minutes against a 600 s bound.

Widen the contract:

```
capacity wait = reader-side pause  OR  staging-executor reservation wait
```

One coordinator-visible capacity-block episode carrying: oldest blocker timestamp; blocker class
(`reader_pause` / `staging_reservation`); run/stripe/attempt/sub-index where applicable;
worker/connection where applicable; current file and byte reservations; high-water values; derived
file requirement.

Any capacity blocker unresolved beyond `staging_capacity_timeout` ⇒ **direct**
`fail_trial("coordinator_staging_capacity_timeout: …")`. **No worker retry matrix.** The terminal
snapshot must name the actual triggering blocker even if its worker thread has since exited (the
F3 snapshot pattern already in the tree is the model). This **widens the observer, not the
classification law** — the existing reader-side timeout stays valid.

### 1.5 Required gates (Beta §6 — minimum set)

- **G-HIGHWATER-ROUTE** — inject non-default file and byte high-waters at the manifest surface;
  prove the exact values reach `CoordinatorConfig` and change the real reservation limits. Mutate
  one hop away; gate must red.
- **G-TRIAL-RETENTION-PREFLIGHT** — using the failed gate-12 geometry, under
  `staging_high_water_files=512`: `derived_required > 512`, **zero** StripeAssign, **zero** result
  traffic, **zero** retry-matrix calls, and a trial abort reason beginning
  `coordinator_staging_retention_sizing:` (or an equivalently specific `coordinator_staging_*`
  classification, named in your report). With a high-water at or above the derived requirement,
  preflight passes. **The gate must compare against the conservative computed requirement — do not
  transcribe 1,028.**
- **G-COMMIT-RELEASE** — successful multi-shard TrialCommit: assembly succeeds → all trial
  reservations released, all trial staged files deleted, capacity returns, assembly remains
  usable. Duplicate successful commit: zero second release, zero second deletion, zero spool
  re-read.
- **G-COMMIT-FAIL-RETAINS** — corrupt one spool: delivery `failed`, all required manifests
  retained, all staged spools retained, all reservations held. Repair and retry the **same
  event**: delivery `done`, and only then release exactly once. **Must preserve D1.1's
  failed-commit retry contract.**
- **G-EXECUTOR-CAPACITY-TIMEOUT** — construct the exact missing shape: no paused connection,
  staging executor blocked by reservation capacity, wait > timeout ⇒ direct
  `coordinator_staging_capacity_timeout`, matrix calls == 0, trigger snapshot says
  `staging_reservation`. Mutation: restore reader-only oldest-pause logic; gate must remain
  wedged/red.
- **G-SEQUENTIAL-TRIAL-REUSE** — two or more sequential successful trials through the same
  production staging ledger/coordinator lifecycle. After trial 1: **held reservations == 0**, and
  trial 2 must consume the same full high-water again. **Mandatory** — the failed production
  command requested eight trials.

---

## PART 2 — `elapsed_s` PERSISTENCE (Beta R4)

**Keep this section separable.** Write it so it can be lifted out and submitted independently if
Part 1 needs another review round. Its gate must not depend on Part 1's changes.

**The defect:** the worker computes and transmits `StripeCompleteMessage.elapsed_s`
(`miner/range_miner_protocol.py:142`), and the coordinator drops it — the call site
(`range_miner_coordinator.py:5903-5905`) passes only `substripes_done` and `survivors_total`, and
`record_stripe_complete` (`:1453`) has no elapsed parameter. A schema-wide search for
`elapsed|duration|compute|started_at` returns **no column**.

**Scope, exactly as Beta ruled — do not exceed it:**

1. **One additive ledger column.** Old rows may remain `NULL`.
2. **One persistence argument/path** — thread the value from the call site into
   `record_stripe_complete`.
3. **Persist the worker-reported value. Do not synthesize a replacement at the coordinator** — a
   coordinator-side timestamp is a different measurement and must not be substituted.
4. **Duplicate/replayed completion must preserve idempotent ledger semantics** — a replayed
   `stripe_complete` must not corrupt or double-write the value.
5. **Round-trip test:** worker → wire → coordinator → ledger.

**Gate G-ELAPSED-ROUNDTRIP:** a worker-reported `elapsed_s` reaches the ledger unmodified;
a replayed completion is idempotent; a completion without the field leaves `NULL` rather than 0.

**Record this measurement caveat in your report and in a source comment** (Beta's warning):
`elapsed_s` is a trustworthy **stripe service-time** measurement, sufficient for per-stripe and
per-worker rate calculations and sizing work. It is **not** aggregate cluster wall-clock
throughput — concurrent worker intervals overlap. **Do not reconstruct fleet throughput by summing
or averaging per-stripe seeds/sec;** any fleet-level figure needs an overlap-aware makespan
denominator.

**Explicitly NOT authorized by this ruling** and out of scope: `gpu_name`, `vram_bytes`, `gpu_id`,
heartbeat counters, `StripeError.error`/`traceback`, `MinerStatusMessage`. Beta: these *"can be
handled separately rather than being smuggled into the one-column amendment."* **Do not include
them.**

---

## Evidence and report

Final-state discipline (standing, Beta §6 of the R3 ruling): the canonical-host runs happen
**after** the last edit, and the report is written **after** those runs.

`docs/CLAUDE_CODE_REPORT_S172_STAGING_CAPACITY_AMENDMENT.md`, containing:

1. Per-ruling-section implementation notes with `file:line`.
2. The derived-bound formula as implemented, and the number it produces for the failed gate-12
   geometry (state it; do not hardcode it).
3. Red-first evidence for every new gate against the pre-amendment tree (worktree at `c7058d8`).
4. Mutation evidence where the gate specifies it.
5. Full suite green on VM101 ×3 after the last edit; `test_s172_staging_partb.py`;
   `test_s172_phase4_coordinator.py` by the accepted isolated-production-diff method.
6. Programmatic confirmation that F1–F5, summary, matrix-diff, handoff and predecode gates are
   **assertion-unchanged**.
7. Files changed — expect `miner/range_miner_coordinator.py`, the three C-route hops, and the
   suite. Anything else must be justified.
8. Any disagreement with this brief **reported, not worked around**.
9. A statement of whether an authoritative maximum shard-size contract exists (§1.3).
