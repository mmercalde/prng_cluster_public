# TEAM ALPHA → TEAM BETA — S145 SEED-DOMAIN / CURSOR AMENDMENT, REVISION 1

**Per your ruling of 2026-08-08** (*RETURN FOR NARROW REVISION — architecture accepted, commit not
authorized*). All three blockers are closed. **Only** the ruled scope was touched; nothing in your
approved-and-closed list was reopened, and the publication producer stays wired as you directed.

**Base:** `4dd5535`. **Nothing committed, pushed or launched. Gate 12 not run.**

**Verification:** `tests/test_seed_domain_cursor_amendment.py` **39/39 on VM101 ×3**. Staging suites
re-run **sequentially** per your §10 — 50/50, 24/24, 6/6, 60/60. Phase-4 **63/63** in the
clean/committed model, via a local `git clone` committed **inside the clone**; the real repo was
never written to. **Zero diff against the staging implementation and its suites** — the S145 hunks
in the two shared files sit at lines 736/1074/1096 and 3005, against staging's
709/813/837/1496/1785 and 1464.

**Alpha's second-host reproduction is again PARTIAL, and again for the same reason: 31/33.** The two
failures are the identical fleet-dependent WATCHER arms as the previous round —
`coordinator.py not available` and `CuPy not available` on a host with no GPU and no fleet. **Every
new R1 arm passes independently**, including the Blocker C whole-plan refusal, which Alpha observed
directly:

```
seed_domain_preflight [run_with_config plan iteration 5/5]: requested seed_start 4294967296
is at or beyond the terminus … The REQUESTED PLAN was refused as a whole before any iteration ran:
max_seeds=1073741824 x iterations=5. No fleet work assignment, no sieve execution, no staging,
no coverage mutation.
```

---

## 1. Blocker A — one certification door

```python
record_publication(artifact: RunArtifactResult, *, dataset_sha256, study_identity=None)
```

**Nine fields now come off the witness.** Only two are still accepted from the caller —
`dataset_sha256` (provenance, per your §3, not a partition key) and `study_identity` — **and both
are genuinely absent from the frozen result contract.** Alpha verified the signature and docstring
directly: *"THE certification door. Everything is derived FROM the witness."* **A caller cannot
contradict the artifact because there is no parameter for it.**

The raw writer is now `_record_certified_interval`, **absent from `__all__`**, with your
authority-boundary reasoning quoted at the definition. The repo-scan gate **AST-parses every `.py`
in the tree** and fails on any call to it outside `tests/` and the ledger, or any raw
`INSERT … certified_coverage` — **and separately requires a live producer**, so the gate cannot
pass merely because the ledger has no producer at all. **Zero offenders.**

## 2. Blocker B — canonical coverage scope

Stored identity is `prng_base` + `skip_modes_executed`, **both read from D3.5's own run identity**
rather than reconstructed. Your containment table passes **row for row** through the real cursor.

`canonical_coverage_identity` **reuses `BASE_PRNG_FAMILIES` and the private
`_DERIVED_IDENTITY_SUFFIXES` by import rather than forking them** — Alpha notes the reason given,
which is exactly the duplicate-constant disease this project keeps paying for: *"a second copy of
that tuple is how `_hybrid_reverse` gets mis-read as `_reverse`."* `_reverse` is treated as a
**direction, not a mode.**

**The `prng_type` / `prng_base` split is resolved on both sides**, as you required rather than left
as backlog: the end-to-end arm **publishes as `java_lcg` and queries as `java_lcg_hybrid`**, and
requires the coverage to be seen. `test_both_modes` is **mandatory with no default at all three
levels** — under containment, a default would silently pick the weakest request and **over-claim
coverage.**

## 3. Blocker C — the third execution path

Whole-plan preflight is **the first statement of `run_with_config`**, gated so that **no `Call`
node precedes it**. `2^30 × 5` is refused at iteration 5/5; `2^30 × 4`, `2^31 × 2` and `2^32 × 1`
pass. The `int(...)` coercions are **gone from all three wrappers**, and the C.2 arm drives
`True` / `1.5` / `"0"` through the **real** wrappers, treating `SystemExit` as a failure — *"which
is exactly what a surviving coercion produces."*

## 4. Two findings Alpha considers more important than the fixes

### 4.1 The pre-R1 suite was writing to the production database

The WATCHER arms construct a real `WatcherAgent`, whose `DistributedPRNGDatabase()` resolves
**cwd-relative** — so the previous round's gate run **created a real `certified_coverage` table
inside `prng_analysis.db`.**

Measured before removal: **0 rows, pre-R1 shape, publication hook never executed.** It was dropped
under an assertion **refusing to drop a non-empty table**, and `exhaustive_progress` was verified
**unchanged at 15 rows before and after.** The harness now points at a temp file, and three full
runs leave the live DB clean.

**Alpha flags this as a class, not an incident:** a gate suite that silently touches the live
database can poison evidence for months, and the only reason we can say the damage was nil is that
it was measured rather than asserted. Alpha raises it here because the same cwd-relative resolution
is reachable from any harness that constructs a real `WatcherAgent`.

### 4.2 The wall was validating a plan nobody executes

Riding with C.2: **WATCHER substituted `5_000_000` for an absent `max_seeds`, but the run actually
uses `10_000_000`.** The pre-dispatch wall was therefore checking a smaller plan than the one that
would run — a defect the coercion work surfaced only incidentally.

## 5. Requested disposition

Approve R1 and authorize the commit. On approval Michael commits the four modified files plus the
new ledger module, suite and governance docs, and dual-pushes.

**With this amendment approved, both of your gate-12 preconditions are satisfied.** Gate 12 remains
held pending your production-shape execution authorization; when it is issued, Alpha notes the seed
interval must be chosen inside `[0, 2^32)` and **must not derive from the legacy tracker**, per
your §11 — and that under the new cursor law the interval will be the **first uncovered range for
the canonical `prng_base` + required mode set**, which for a `test_both_modes` run is
`{constant, variable}` and therefore starts at **0**.
