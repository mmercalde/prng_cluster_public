# CLAUDE CODE REPORT — SEED-DOMAIN / COVERAGE-CURSOR AMENDMENT

**Host:** VM101 (`zeus-ubuntu-vm`, 192.168.3.177), repo `~/distributed_prng_analysis`,
venv `~/venvs/torch`.
**Authority:** Team Beta ruling *"S145 / SEED-DOMAIN SWEEP TERMINUS AND COVERAGE AUTHORITY"*
(2026-08-07), per `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_AMENDMENT.md`.
**Status:** implemented, 29/29 gates green ×3 after the last edit. **Not committed, not
pushed, nothing launched.**

---

## 0. BASE CORRECTION — READ FIRST

**The brief states HEAD `c7058d8`. That is stale.** The owner corrected the base to
**`4dd5535`** (the S172 staging-capacity amendment, +R1 +R2, Beta APPROVED 2026-08-08),
and all work in this report — implementation, red-first worktree, and every suite run —
uses `4dd5535`.

Base verification, as the brief requires:

| check | result |
|---|---|
| `git log --oneline -1` | `4dd5535` |
| tracked tree clean at start | **yes** — `git status --porcelain --untracked-files=no` empty |
| untracked residue | 5 entries: the delivered brief, 2× `*.db-shm`, 2× `*.db-wal`, 1× `*.stale_*` rotation — expected, not a stop condition |
| existing suites green | backpressure 50/50 · partb 24/24 · elapsed 6/6 · D3.5 finalizer 60/60 · phase-4 62/63 (Gate 22 only — see §5) |

No tracked-source drift was found, so no stop condition was triggered.

---

## 1. PER-RULING-SECTION IMPLEMENTATION NOTES

### §1 — THE TERMINUS

The governed domain is `[0, 2^32)`. **No second constant was defined.**
`utils/seed_coverage_ledger.py:78-83` imports `SEED_DOMAIN_CONTRACT`,
`SEED_DOMAIN_START`, `SEED_DOMAIN_END_EXCLUSIVE` and `SEED_DOMAIN_EXCLUSIVE_MAX`
directly from `utils.run_finalizer` — the authority already at `utils/run_finalizer.py:277`
and enforced fail-closed at `:533`/`:547`/`:571`.

Two gates hold this shut rather than asserting it:

* `arm_domain_constant_is_shared_not_restated` walks the ledger module's **AST** and
  reds if `SEED_DOMAIN_EXCLUSIVE_MAX` is ever *assigned* there, and confirms the name
  arrives via `ImportFrom(module='utils.run_finalizer')`.
* `arm_boundary_mutation_reds_both_walls` runs a **subprocess** that mutates
  `run_finalizer.SEED_DOMAIN_EXCLUSIVE_MAX` to `2**31` *before* the ledger imports it,
  then proves **both** walls refuse `[0, 2^32)` — the pre-dispatch wall
  (`SeedDomainPreflightError`) and the finalizer's own parity gate
  (`CoverageValidationError`). This satisfies §1's *"a mutation that changes one boundary
  must red both"* requirement.

  *Why a subprocess and not a monkeypatch:* the ledger's `from … import` binds the value
  at import time, so an in-process patch of `run_finalizer` would leave the ledger's copy
  untouched and the mutation test would silently prove nothing. The subprocess is what
  makes the shared-authority claim falsifiable.

### §§2-4 — THE LEGACY TRACKER IS DEAUTHORIZED

`database_system.get_next_seed_start` no longer reads `exhaustive_progress` at all.
`database_system.py:373-418` delegates to the certified ledger; the legacy writers
`update_exhaustive_progress` (`:303`) and `get_exhaustive_progress` (`:319`) are
**deliberately unchanged**, so the rows remain writable, readable and auditable as
historical telemetry.

* **Rows 5-15 and rows 1-4 alike contribute zero certified progress.** The new stream
  starts at zero; no provenance migration was performed.
* **The low-range hole needs no surgical repair** — the table holding it is no longer the
  authority, exactly as the brief states.
* `arm_legacy_table_is_never_read_by_the_ledger` proves structurally that the ledger
  module issues no SQL against `exhaustive_progress`.

### §5 — COVERAGE LEDGER v1

New module `utils/seed_coverage_ledger.py` (~760 lines). Schema and append-only
enforcement in §2 below. Coverage is bound to publication at
`utils/seed_coverage_ledger.py:637` (`record_publication`), which takes the
`RunArtifactResult` itself rather than a caller-typed digest.

### §6 — CURSOR LAW

`first_uncovered_seed` (`utils/seed_coverage_ledger.py:385`) implements Beta's algorithm
in order: normalize → clip/reject by exact domain contract → merge overlaps (computation
only) → start at D0 → return the first uncovered seed.

* Beta's worked example returns **1000**, not `2^31` — measured, `arm_first_gap_beta_worked_example`.
* Completion is an explicit state: `status = COMPLETE` and `next_seed_start = None`
  (`CursorResult`, `:280-315`). **There is no `4,294,967,296` next run.** A consumer that
  ignores `status` and uses the value arithmetically gets a `TypeError`, never an
  out-of-domain sweep.
* Adjacent intervals merge (`[0,10)` + `[10,20)` → cursor 20); the algorithm is
  order-invariant.

### §7 — PRE-DISPATCH SEED-DOMAIN WALL

`assert_seed_domain_preflight` (`utils/seed_coverage_ledger.py:189`) applies Beta's four
conditions in Python integers so the addition cannot wrap. Reason strings lead with
`seed_domain_preflight:` and name the governed contract, e.g.

```
seed_domain_preflight [watcher_step1 java_lcg]: requested [4294967295,4294967297)
exceeds v1.1-stratum [0,4294967296).
```

Applied at **both** entry points:

| entry point | anchor | position |
|---|---|---|
| WATCHER (production driver) | `agents/watcher_agent.py:1770-1784` | inside `run_step`, before the command is built and before `_run_step_streaming` → `subprocess.Popen` |
| direct CLI | `window_optimizer.py:739-766` | the **first statement** of `run_bayesian_optimization`'s body |

The CLI wall is **an addition beyond the brief's literal WATCHER wording, disclosed here**
— see §8.

### §7 (brief) — WATCHER INTEGRATION

`agents/watcher_agent.py:1662-1784` replaces the S140 block. Three behavioural changes:

1. **Cursor law.** Consumes `get_certified_cursor`; the legacy tracker is not read.
2. **COMPLETE is explicit.** `:1714-1731` — no run is generated, the operator is told the
   domain is exhausted, and the result carries
   `blocked_by="seed_domain_complete"`, `seed_domain_complete=True`, `next_seed_start=None`.
3. **Fail-closed.** `:1697-1712` — S140 swallowed every exception and proceeded at
   `seed_start=0`, so a broken lookup was indistinguishable from "no coverage yet". Step 1
   is now **blocked** (`blocked_by="seed_coverage_cursor_unavailable"`), mirroring the
   dataset-authority P0.5 block immediately above it. **This is a deliberate change the
   brief did not ask for; it is flagged in §8 and is a one-line revert if Beta disagrees.**

The S145-R1 study-continuity invariant (`resume_study` / `study_name`) is preserved
verbatim, and the S167 warm-start removal note is retained.

**Backend independence — the note the brief asked me to record.** Confirmed: the live
manifest still carries `use_range_miner = False` / `use_persistent_workers = True`, so a
WATCHER-driven Step 1 today takes the **PWC-TCP** path. **All changes in this amendment are
backend-independent.** The WATCHER wall sits upstream of the backend cascade — it is reached
before the Step-1 subprocess is spawned, so it applies identically to PWC and miner runs.
The CLI wall is the first statement of `run_bayesian_optimization`, above the
`COORDINATOR_AVAILABLE` check and every backend selection. The coverage binding hangs off
`finalize_run`, which is the single shared publication path for both backends. Verified in
the live EXEC CMD emitted during gate execution: `--use-persistent-workers … --pwc-transport tcp`.

---

## 2. COVERAGE LEDGER v1 — SCHEMA AS BUILT

Dumped from `sqlite_master` on a freshly initialized database (not retyped from source):

```sql
CREATE TABLE certified_coverage (
    coverage_id            TEXT    PRIMARY KEY,
    run_id                 TEXT    NOT NULL,
    study_identity         TEXT,
    prng_type              TEXT    NOT NULL,
    mapping_mode           TEXT,
    seed_domain_contract   TEXT    NOT NULL,
    seed_start             INTEGER NOT NULL,
    seed_end_exclusive     INTEGER NOT NULL,
    dataset_sha256         TEXT    NOT NULL,
    repository_commit      TEXT    NOT NULL,
    artifact_sha256        TEXT    NOT NULL,
    generation_id          TEXT,
    publication_status     TEXT    NOT NULL,
    recorded_at            TEXT    NOT NULL,
    CHECK (seed_start >= 0),
    CHECK (seed_end_exclusive > seed_start),
    CHECK (seed_end_exclusive <= 4294967296),
    CHECK (publication_status = 'CERTIFIED')
);
CREATE INDEX idx_certified_coverage_prng ON certified_coverage(prng_type, seed_start);
```

Every field Beta §5 enumerates is present. `coverage_id` is the immutable record identity,
`study_identity` the search/study identity, `mapping_mode` the skip modes executed,
`publication_status` the completion state, `recorded_at` the timestamp.

### How append-only is enforced — the brief asks "constraint? no REPLACE path? both?"

**Both, and a third.** The clobber that destroyed the legacy first row twice had to become
structurally impossible, not merely avoided by convention:

1. **The primary key is a per-RECORD content identity, not the starting seed.**
   `coverage_id = sha256(canonical JSON of every bound field)`
   (`utils/seed_coverage_ledger.py:556`). The legacy table's PK includes
   `seed_range_start`, which is precisely why a 1,000-seed run at zero replaced a
   billion-seed run at zero. Here two intervals both starting at 0 are two distinct rows
   that **cannot** collide.
2. **No REPLACE/UPDATE/DELETE path exists in the module.** `_insert_row`
   (`utils/seed_coverage_ledger.py:700`) issues a bare `INSERT` — not `INSERT OR REPLACE`
   (which clobbered), not `INSERT OR IGNORE` (which would hide a genuine collision).
3. **Database-level triggers.** `certified_coverage_no_update` and
   `certified_coverage_no_delete` `RAISE(ABORT, …)`, so a future caller reaching the table
   through raw SQL is stopped by SQLite itself.

**One subtlety is load-bearing and worth Beta's attention.** Defense (3) closes
`INSERT OR REPLACE` *only* because REPLACE satisfies a conflict by DELETING the losing
row — and **SQLite fires trigger-driven deletes only when recursive triggers are enabled.**
`_connect()` therefore sets `PRAGMA recursive_triggers = ON`
(`utils/seed_coverage_ledger.py:537`). Without that pragma the delete trigger would be
**silently vacuous against the exact statement it exists to stop**, and the gate would
still have gone green on the UPDATE and DELETE arms. `arm_append_only_is_enforced_by_the_database`
drives all three statements through raw SQL and requires each to abort.

**Idempotency, not double counting.** Because `coverage_id` is a hash of every bound field,
re-recording the same publication collides on the primary key and returns the existing row
rather than appending a second (`:600-610`). A *different* interval hashes differently and
inserts. Randomised identities would have let a retry silently double-count an interval.

---

## 3. RED-FIRST AND MUTATION EVIDENCE

**Method.** `git worktree add --detach <scratch> 4dd5535`, the suite file copied in
unchanged, executed in the same venv on the same host. The worktree has been removed and
`git worktree prune` run; `git worktree list` shows only the main tree.

**Result: 28 of 29 checks red at base, 29 of 29 green on the patched tree.**

```
BASE 4dd5535   :  1/29 checks green   COMPLETION SENTINEL: FAIL   (exit 1)
PATCHED        : 29/29 checks green   COMPLETION SENTINEL: PASS   (exit 0)
```

**The single check that passes at base is `G-NO-REPLACE-CLOBBER: control — the legacy
writer really clobbers`, and it MUST pass there.** It is the fault-injection control: it
drives the real `update_exhaustive_progress` twice at seed 0 and requires the
billion-seed row to be destroyed. If that arm ever goes red at base, the incident this
gate exists for is not being reproduced and the gate is measuring nothing.

### Behavioural red, not "module missing"

Two gates were deliberately restructured so their **first** assertion is independent of the
new module, so the base red is a statement about behaviour rather than about a missing
import. Verbatim from the base run:

```
G-LEGACY-NONAUTHORITY: 16.1B history ignored completely
  AssertionError: the cursor returned 16,106,127,360, which is beyond the terminus
  4,294,967,296 by 11,811,160,064 seeds. Beta §1: 'No run may begin at 2^32, cross
  2^32, or publish a candidate outside that interval.'

G-CURSOR-FIRST-GAP: through production get_next_seed_start
  AssertionError: get_next_seed_start returned 2,147,483,648 — that is
  MAX(seed_range_end) over the LEGACY tracker, the rule Beta §6 invalidated. It skips
  the ~1.07-billion-seed hole at [1,000, 1,073,741,824) and declares it covered.
```

A direct measurement at base, outside the suite, confirms the same two numbers.

### Mutation evidence

| gate | mutant | result |
|---|---|---|
| **G-CURSOR-FIRST-GAP** | reinstate `MAX(seed_range_end)` over the certified intervals | mutant answers `2^31`, oracle answers `1000`; the gate distinguishes them (`arm_first_gap_mutation_max_end_is_caught`) |
| **G-PUBLICATION-BINDS-COVERAGE** | bypass the publication binding and fabricate an `artifact_sha256` for a run that never published | the mutant writes and **moves the cursor to 1,000,000,000** — proving the mutation is live and detectable — while `record_publication`, the only legitimate producer, refuses the unpublished case (`arm_publication_binding_mutation_is_caught`) |
| **§1 boundary** (extra, required by §1) | `run_finalizer.SEED_DOMAIN_EXCLUSIVE_MAX = 2**31` before ledger import | **both** the pre-dispatch wall and the finalizer parity gate go red (`arm_boundary_mutation_reds_both_walls`) |

### Anti-vacuity: the clean control on the executed dispatch gate

`arm_preflight_zero_dispatch_executed` drives the **real** `WatcherAgent.run_step(1, …)`
with `subprocess` replaced by a tripwire:

* **fault injection** — `seed_start = 2^32` → `blocked_by == "seed_domain_preflight"`, tripwire never fires;
* **clean control** — `seed_start = 0` → execution reaches `Popen` and the tripwire **does**
  fire. Without this arm, "nothing dispatched" would be equally consistent with a dead path.

Two facts found while building it, both worth recording because they would have inverted
the measurement:

1. `run_step` does **not** call `Popen` itself; it delegates to `_run_step_streaming`
   (`agents/watcher_agent.py:2285`). The AST arm now locates the spawn owner from the live
   AST instead of assuming either shape.
2. `run_step` wraps dispatch in `except Exception` and converts what it catches into
   `{'success': False, 'error': …}`. A tripwire deriving from `Exception` is **swallowed
   there**, and the gate would read "no dispatch" for a run that dispatched. The tripwire
   therefore derives from `BaseException`. *This is the vacuous-pass class VIR-2 exists for,
   and it was caught only because the clean control refused to go green.*

The environmental gates upstream of the S145 block (hard preflight, output freshness,
dataset-authority P0.5) are stubbed in that harness — the rigs are unreachable from VM101
right now, so `run_step` otherwise returns `blocked_by="preflight_hard_failure"` or
`"dataset_authority_p0_5"` and never reaches the wall. Each stub is an unrelated gate;
**the tripwire, which is the measurement, is never stubbed.**

---

## 4. FULL SUITE GREEN ×3 AFTER THE LAST EDIT

Final-state discipline observed: the last tracked edit was `window_optimizer.py` at
**21:53:38**; all three runs followed it, and this report was written after them.

```
RUN 1: 29/29 checks green | COMPLETION SENTINEL: PASS — seed-domain cursor amendment green
RUN 2: 29/29 checks green | COMPLETION SENTINEL: PASS — seed-domain cursor amendment green
RUN 3: 29/29 checks green | COMPLETION SENTINEL: PASS — seed-domain cursor amendment green
```

Gate coverage: **G-DOMAIN-PREFLIGHT** 8 arms · **G-CURSOR-FIRST-GAP** 4 ·
**G-CURSOR-COMPLETE** 3 · **G-LEGACY-NONAUTHORITY** 3 · **G-NO-REPLACE-CLOBBER** 3 ·
**G-PUBLICATION-BINDS-COVERAGE** 4 · **G-OUT-OF-DOMAIN-LEGACY** 4. All seven of Beta §10's
gates are present.

---

## 5. THE STAGING SUITES ARE UNAFFECTED — CONFIRMED PROGRAMMATICALLY

**Structural.** `git diff --stat 4dd5535 -- <path>` returns **zero lines** for every one:

| file | diff vs `4dd5535` |
|---|---|
| `tests/test_s172_staging_backpressure.py` | 0 |
| `tests/test_s172_staging_partb.py` | 0 |
| `tests/test_s172_phase4_coordinator.py` | 0 |
| `miner/range_miner_coordinator.py` | 0 |
| `miner/range_miner_protocol.py` | 0 |
| `agent_manifests/window_optimizer.json` | 0 |

**Behavioural**, run sequentially on the patched tree:

| suite | result | committed baseline at `4dd5535` |
|---|---|---|
| `test_s172_staging_backpressure.py` | **50/50 PASS** | 50/50 ✅ |
| `test_s172_staging_partb.py` | **24/24 PASS** | 24/24 ✅ |
| `test_s172_elapsed_roundtrip.py` | **6/6 PASS** | 6/6 ✅ |
| `test_s172_phase5_d3_5_finalizer.py` | **60/60 PASS** | ✅ |
| `test_s172_phase4_coordinator.py` | **62/63** — Gate 22 only | 63/63 |

### Gate 22 — the known untracked/changed-`.py` class, not a regression

```
[FAIL] Gate 22: coexistence — unexpected changed .py files:
       {'database_system.py', 'utils/seed_coverage_ledger.py',
        'tests/test_seed_domain_cursor_amendment.py'}
```

Gate 22 builds `changed_py` from `git status --porcelain`, which includes **untracked**
files. This is the documented behaviour that arises on every development pass and resolves
when the files are committed. It is **not** a content regression: the assertion names file
*identities*, not behaviour, and the other 62 checks pass. Note that
`agents/watcher_agent.py`, `window_optimizer.py` and
`window_optimizer_integration_final.py` are already in Gate 22's `allowed` set and do not
appear.

### ⚠ A NEW FLAKE FOUND — Part B G-VAL-6 under concurrency

On a first pass I ran five suites **concurrently** and Part B returned **23/24**, red on
`G-VAL-6: insufficient headroom rejected (own reason)`. It is **not** a regression:

* `staging_high_water_bytes = 458,475,704,320` vs available `458,475,692,032` — a
  **12,288-byte** gap on a 916 GB filesystem;
* Part B alone on the patched tree: **24/24, twice**;
* the gate derives its high-water from free `$TMPDIR` at one instant and validates against
  free space at another, so a concurrently-running suite writing temp files flips it.

**Recommendation (not actioned — outside this amendment's authority):** G-VAL-6 should
either snapshot free space once or carry an explicit headroom margin. This is adjacent to,
but distinct from, the already-known host-assumption class where gates inheriting the
16 GiB default red on a small `$TMPDIR`. **Run these suites sequentially.**

---

## 6. FILES CHANGED

**Modified (4 tracked):**

| file | lines | what |
|---|---|---|
| `database_system.py` | +139 / −67 region | `get_next_seed_start` → first-gap certified cursor (`:373`); new `get_certified_cursor` (`:357`), `record_certified_coverage` (`:419`), `_coverage_ledger` (`:342`). Legacy writers untouched. |
| `agents/watcher_agent.py` | +143 region | certified cursor + COMPLETE handling + pre-dispatch wall (`:1662-1784`) |
| `window_optimizer.py` | +30 | CLI pre-dispatch wall, first statement of `run_bayesian_optimization` (`:739-766`) |
| `window_optimizer_integration_final.py` | +39 | publication-bound coverage record after `finalize_run` succeeds (`:3008-3046`) |

**New (2):**

| file | lines | what |
|---|---|---|
| `utils/seed_coverage_ledger.py` | ~760 | terminus, pre-dispatch wall, Coverage Ledger v1, cursor law |
| `tests/test_seed_domain_cursor_amendment.py` | ~900 | the seven Beta §10 gates, 29 checks |

`git diff --stat 4dd5535` → **4 files changed, 284 insertions(+), 67 deletions(-)**.

**Two of the four modified files are staging-amendment files** (`window_optimizer.py`,
`window_optimizer_integration_final.py`). That is a deliberate, owner-authorized scope
decision — see §8.1 and §8.2. Neither edit is within ~1,500 lines of a staging hunk:
staging touched `window_optimizer_integration_final.py` only at `run_bidirectional_test`
(~`:1464-1483`); this amendment touches `:3008`. Staging touched `window_optimizer.py` at
`@@709/813/837/1496/1785`; this amendment touches `:739-766`, a region no staging hunk
covers.

---

## 7. ROWS 5-15 — EXPLICIT STATEMENT

**Rows 5-15 of `exhaustive_progress` were NOT deleted, NOT renumbered, NOT rewritten and
NOT folded back into `[0, 2^32)`.** No row of the legacy table was modified by this
amendment. The only change to the legacy table's world is that **read paths were
redirected**: `get_next_seed_start` no longer queries it. The legacy writers
(`update_exhaustive_progress` `:303`, `get_exhaustive_progress` `:319`) are byte-identical
to `4dd5535` and the table remains fully readable, writable and auditable as historical
telemetry.

This is gated, not merely asserted:

* `arm_legacy_rows_are_not_deleted_or_rewritten` seeds the real 16.1B history through the
  production writer, exercises the new read path, then requires the row **count**, the row
  **values** and the presence of at least one row with `end > 2^32` all to be unchanged;
* `arm_legacy_rows_remain_auditable` reads an out-of-domain row back and requires its
  values intact;
* `arm_out_of_domain_never_enters_the_union` requires the out-of-domain extents to appear
  in the normalizer's `dropped` audit list — **the exclusion is observable, not inferred
  from the union's shape.**

---

## 8. DISAGREEMENTS, ADDITIONS AND OPEN DECISIONS — REPORTED, NOT WORKED AROUND

### 8.1 The brief's file list vs the owner's "do not touch staging files" constraint

The brief §8.6 expects *"`database_system.py`, `agents/watcher_agent.py`, **the
pre-dispatch site**, a new ledger module, and a new suite."* On the `4dd5535` tree **every
candidate pre-dispatch site and the only correct publication-binding site live in files the
staging amendment owns.** I stopped and put this to the owner rather than deciding it
silently. Both decisions below are owner-authorized:

* **CLI wall — owner-directed.** Beta §7 requires termination *before sieve execution and
  coverage mutation*; a wall covering only WATCHER leaves the direct-CLI path running a
  full sieve and being refused by the finalizer at publication, which is the expensive
  after-the-GPU-work rejection §7 exists to eliminate. Precedent cited: the R2 cohort
  freeze needed `_pick_other_worker` for the same reason and Beta certified that scope as
  correct. **Disclosed here as an addition beyond the brief's literal WATCHER wording.**
* **Publication binding — my call, owner expressed no preference.** I wired it. Left
  unwired, the ledger has **no producer**: the certified cursor returns 0 forever and every
  Step-1 run restarts at seed 0. That is the implemented-but-unreachable pattern this
  project has catalogued repeatedly, and it would have shipped a coverage authority that
  can never record coverage. `arm_publication_binding_is_wired_after_finalize` now requires
  **exactly one** live producer, positioned **after** the publication call.

**If Beta prefers strict file separation, both are cleanly revertible**: `window_optimizer.py`
+30 and `window_optimizer_integration_final.py` +39, each one contiguous block.

### 8.2 The coverage binding RAISES on failure — deliberate, flagging for ruling

`window_optimizer_integration_final.py:3031` does not swallow a ledger failure. The
generation is already published and stays published; what fails is the coverage record.
Swallowing it would leave the cursor silently wrong and the next run silently re-sweeping —
the defect class this amendment closes. The write is idempotent, so a retry after the cause
is fixed is safe. Precedent for failing loudly after a committed action: the finalizer's own
`PublicationDurabilityError`. **Beta may prefer a warn-and-continue; that is a two-line
change.**

### 8.3 WATCHER now BLOCKS on cursor failure instead of defaulting to 0 — beyond the brief

S140 caught every exception and proceeded at `seed_start=0`. I made this fail-closed
(`blocked_by="seed_coverage_cursor_unavailable"`). The brief did not ask for it; I judged
that a silent fallback is the same defect class the amendment exists to close, and that
re-sweeping certified ground while reporting success is worse than refusing. **Flagged
because it changes WATCHER's availability behaviour. One-line revert if Beta disagrees.**

### 8.4 An open design decision Beta should confirm — the cursor's partition key

`certified_cursor` scopes the union by **`prng_type` only**, matching the signature of the
call site it replaces (`get_next_seed_start(prng_type, chunk_size)`). `mapping_mode` is
**recorded on every row** but is not a cursor partition. The reading: a sweep of `[a,b)`
covers that seed interval for the family, and the executed skip modes are already carried
by the artifact's sidecar.

**If Beta intends coverage to be per-(prng_type, skip-mode)**, this is the line to change —
`utils/seed_coverage_ledger.py:742` — and the data to do it is already stored, so no
migration would be needed. **I did not assume; I am asking.**

### 8.5 A latent key inconsistency I did not "fix"

The existing S140 write-back records `prng_type=prng_type` (the CLI argument,
`window_optimizer.py:895`) while the adjacent D3.5 code records `prng_type=str(prng_base)`
(`window_optimizer_integration_final.py:2618`). The new binding uses **`prng_base`**, matching
the D3.5 code it sits beside. If a run is ever driven with `prng_type='java_lcg_hybrid'`,
WATCHER would query the cursor under `java_lcg_hybrid` while the ledger recorded
`java_lcg`. **Not repaired — it is pre-existing, outside this amendment's authority, and
repairing it silently would change which runs see which coverage.** Reported for the
backlog.

### 8.6 Not done, and why

* **No `exhaustive_progress` migration.** Beta: *"the old 16.1B tracker contributes zero
  certified progress… the new certified v1.1 coverage stream starts at zero."*
* **No repair of the ~1.07-billion-seed hole.** Explicitly stated in the brief as
  needing no surgical repair.
* **Nothing launched.** No pipeline run, no fleet launch, no port 5700 bind. Gate 12 and
  the Phase-7 soak remain HELD. Optuna, the strategy system and all sieve mathematics are
  untouched.
* **Not committed, not pushed.** Per the standing contract, Michael commits and dual-pushes.

---

## 9. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

* **execution proof:** every gate prints PASS/FAIL per arm and the suite terminates in an
  explicit `COMPLETION SENTINEL` line with a matching exit code (0 green / 1 red); base and
  patched runs both captured to files.
* **clean control:** `arm_preflight_zero_dispatch_executed` requires an in-domain request
  to **reach** dispatch (tripwire fires) alongside the out-of-domain refusal; the base run's
  single passing check is the legacy-clobber control.
* **fault-injection control:** `arm_legacy_writer_really_does_clobber` reproduces the
  original incident against the real production writer; three mutants (§3) each proven live
  before being caught.
* **completion sentinel:** `COMPLETION SENTINEL: PASS — seed-domain cursor amendment green`.
* **unavailable-observer behavior:** the cursor read path is fail-closed — a ledger failure
  raises and WATCHER blocks the step rather than reporting a fabricated `seed_start=0`
  (§8.3). No check in this suite reports PASS on an unobserved surface.
* **audit claim scope:** claims are about the seed-domain terminus, the coverage cursor,
  the coverage ledger, and the two Step-1 entry points on **this VM101 tree at `4dd5535`
  plus the working-tree diff**. No claim is made about rig-side state, about a live fleet
  run, or about `n_parallel > 1`.
* **searched surfaces:** `docs/` and the governance trail (`TB_RULING_*`,
  `TB_RULING_REQUEST_*`, `PROPOSAL_*`, `TEAM_ALPHA_*`, `CLAUDE_CODE_INSTRUCTIONS_*`) ·
  `git ls-files` · `git log --all` including deleted paths · the live VM101 filesystem
  including gitignored files (`/usr/bin/find`, `/bin/grep`) · live Python imports and
  `inspect.signature` · live execution of `WatcherAgent.run_step` · live SQLite schema
  introspection · a `4dd5535` worktree.
* **unavailable surfaces:** the three rigs (`192.168.3.122/.156/.164` — *No route to host*
  during this session, which is why the WATCHER harness stubs preflight and P0.5); the
  Proxmox hosts' kernel logs; any live-fleet or GPU execution path. **No claim in this
  report depends on them.**
* **governance trail searched:** yes — the S145 ruling as carried by the brief, the S172
  staging-capacity arc (`4b1aad6`, `4dd5535`) for the file-ownership boundary, and D3.5's
  seed-domain contract in `utils/run_finalizer.py`.
* **chapters searched:** not required for this amendment — no claim here concerns sieve
  mathematics, kernel semantics or feature provenance.

---

## 10. WHAT BETA IS BEING ASKED TO RULE ON

1. **§8.1** — the two staging-file edits (CLI wall, publication binding), disclosed and
   revertible.
2. **§8.2** — coverage binding raises on failure rather than warning.
3. **§8.3** — WATCHER blocks on cursor failure instead of defaulting to `seed_start=0`.
4. **§8.4** — the cursor's partition key: `prng_type` only, or `(prng_type, skip mode)`.
5. **§8.5** — the pre-existing `prng_type` vs `prng_base` key inconsistency, for the backlog.
6. **§5** — the newly found Part B `G-VAL-6` concurrency flake, for the backlog.
