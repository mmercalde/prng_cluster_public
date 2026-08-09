# CLAUDE CODE REPORT — S145 SEED-DOMAIN / CURSOR AMENDMENT, REVISION 1

**Host:** VM101 (`zeus-ubuntu-vm`, 192.168.3.177), repo `~/distributed_prng_analysis`,
venv `~/venvs/torch`. Base `4dd5535`; the amendment remains **uncommitted in the working tree**.
**Authority:** Team Beta ruling *"S145 SEED-DOMAIN / COVERAGE-CURSOR AMENDMENT"* (2026-08-08),
per `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R1.md`.
**Status:** all three blockers closed. **39/39 gates green ×3** after the last edit.
**Not committed, not pushed, nothing launched. Gate 12 not run.**

## 0. BASE VERIFICATION

| check | result |
|---|---|
| HEAD | `4dd5535` |
| amendment present in working tree | yes — 4 modified + 4 untracked (2 `.py`, 2 docs) |
| pre-R1 suite | **29/29 green** before any R1 edit |
| untracked runtime residue | WAL sidecars + `*.stale_*` — expected, not a stop condition |

Only Beta §12's permitted surfaces were touched. Nothing in the approved-and-closed list was
reopened: the shared terminus, legacy deauthorization, first-gap cursor, explicit `COMPLETE`,
the append-only design, **the publication producer stays wired**, raising on a post-publication
ledger write failure, WATCHER failing closed on cursor lookup, and `dataset_sha256` as
provenance rather than a partition key are all unchanged.

---

## 1. BLOCKER A — ONE CERTIFICATION DOOR

**Beta was right, and the suite was the proof.** The pre-R1 `record_publication` accepted any
object exposing `artifact_sha256` while taking every other coverage field from the caller. Both
demonstrations Beta cited were real: the "successful publication" arm certified coverage from
`_FakeArtifact`, and the mutation gate reached the **public** `record_certified_interval` to
fabricate a never-published billion-seed interval that moved the authoritative cursor. That was
not a hypothetical mutant; it was a live production API bypass.

### As built

```python
record_publication(artifact: RunArtifactResult, *, dataset_sha256, study_identity=None)
```

`utils/seed_coverage_ledger.py:862`. **Nine fields are now derived FROM the witness**, which
D3.5 constructs only after the canonical publication commit succeeds:

| field | source |
|---|---|
| `run_id` · `prng_base` · `skip_modes_executed` | `artifact.*` |
| `seed_start` · `seed_count` (→ `seed_end_exclusive`) | `artifact.*` |
| `artifact_sha256` · `generation_id` · `repository_commit` | `artifact.*` |

**Still accepted from the caller — two, and both are genuinely absent from the frozen result
contract:**

* `dataset_sha256` — the run-scoped frozen dataset identity resolved by P0.5 dataset authority.
  `RunArtifactResult` does not carry it. Beta ruled it **provenance, not a v1 partition key**, so
  it cannot split the coverage namespace.
* `study_identity` — the Optuna study name. Optional, provenance only, never read by the cursor.

There is **no parameter** by which a caller can contradict the witness, so the property holds by
construction rather than by discipline. `arm_caller_cannot_substitute_artifact_fields` asserts the
exact parameter set and separately asserts, behaviourally, that every stored field equals the
artifact's.

### Witness validation

`_require_publication_witness` (`utils/seed_coverage_ledger.py:911`) implements Beta's "not the
canonical result type **or** cannot satisfy the complete frozen result contract": an `isinstance`
pass returns immediately; anything else must expose **every** field of the frozen dataclass, with
21 of them additionally type-pinned (`_RESULT_FIELD_TYPES`, `:148`). `bool` is rejected where
`int` is required. `_FakeArtifact("f"*64)` fails on the first missing field.

### The raw writer is now an internal seam

`record_certified_interval` → **`_record_certified_interval`** (`:791`), absent from `__all__`,
with the authority boundary stated at the call site. Beta: *"Python underscore privacy is not a
security boundary; that is not the point. This is an authority boundary."*

`arm_no_production_certification_bypass` is the **repo/source scan**: it walks every `.py` in the
tree, parses each with `ast`, and fails if any file outside `tests/` and the ledger module itself
calls `_record_certified_interval` / `record_certified_interval`, or contains a raw
`INSERT … certified_coverage`. The forensic `apply_s*` / `verify_s*` / `fix_s*` corpus is excluded
by name (never re-executed). The same arm requires at least one live caller of
`record_publication` / `record_certified_coverage`, so it cannot pass by the ledger simply having
no producer. **Result: zero offenders, producer present.**

The old mutation arm is retained but re-labelled: it now reaches the internal seam, still proves a
fabricated interval moves the cursor (so the detector is live), and additionally asserts
`record_certified_interval` is **gone** from the public surface.

---

## 2. BLOCKER B — CANONICAL COVERAGE SCOPE

Beta answered the partitioning question and **rejected both options I framed** — not raw
`prng_type`, and not a `(prng_type, skip_mode)` scalar either.

### The identity as stored

```sql
CREATE TABLE certified_coverage (
    coverage_id            TEXT    PRIMARY KEY,
    run_id                 TEXT    NOT NULL,
    study_identity         TEXT,
    prng_base              TEXT    NOT NULL,
    skip_modes_executed    TEXT    NOT NULL,      -- canonical order, comma-joined
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
```

`skip_modes_executed` is stored in `CANONICAL_SKIP_MODES` order, so the content-derived
`coverage_id` cannot depend on a caller's set-iteration order. Both values come straight from the
artifact — the canonical run identity D3.5 already computes, not a new one invented in the ledger.

### The containment predicate

`CertifiedInterval.covers_modes` (`:634`), applied by `certified_cursor` (`:1029`):

```
record.prng_base == requested.prng_base  AND  requested_modes ⊆ record.skip_modes_executed
```

`arm_containment_matrix` drives Beta's table row for row through the real ledger and cursor:

| certified | requested | counts | result |
|---|---|---|---|
| `{constant}` | `{constant}` | YES | ✅ |
| `{constant}` | `{constant, variable}` | NO | ✅ |
| `{constant, variable}` | `{constant}` | YES | ✅ |
| `{constant, variable}` | `{constant, variable}` | YES | ✅ |
| `{variable}` | `{constant}` | NO | ✅ |

### How the `prng_type` / `prng_base` mismatch was canonicalized — both sides

`canonical_coverage_identity` (`:271`) is the single resolver, and it reuses the frozen vocabulary
rather than forking it: `BASE_PRNG_FAMILIES` and `_DERIVED_IDENTITY_SUFFIXES` are imported from
`utils/canonical_arrays.py` (the private suffix tuple is imported, never copied — a second copy is
exactly how `_hybrid_reverse` gets mis-classified as `_reverse`).

```
java_lcg,                test_both_modes=False  ->  ('java_lcg', {constant})
java_lcg,                test_both_modes=True   ->  ('java_lcg', {constant, variable})
java_lcg_hybrid,         (either)               ->  ('java_lcg', {variable})
java_lcg_hybrid_reverse, (either)               ->  ('java_lcg', {variable})
java_lcg_reverse,        test_both_modes=False  ->  ('java_lcg', {constant})
```

* **WATCHER-query side** — `agents/watcher_agent.py:1710` calls
  `get_certified_cursor(prng_type, test_both_modes=...)`; `database_system.py:357` runs the query
  through the resolver before it reaches the ledger. A WATCHER query for `java_lcg_hybrid` now
  lands in the `java_lcg` namespace.
* **Publication-hook side** — `window_optimizer_integration_final.py:3038` no longer passes a
  PRNG identity at all; the ledger reads `artifact.prng_base` and `artifact.skip_modes_executed`.

`arm_canonical_identity_unifies_hybrid_and_base` closes the loop end to end: it publishes a
both-modes run recorded as `java_lcg`, then queries `get_next_seed_start("java_lcg_hybrid", …)`
and requires it to see that coverage. `_reverse` is treated as a **direction**, not a skip mode —
forward and reverse are two halves of one bidirectional pass.

**`test_both_modes` is keyword-only with NO DEFAULT** on `get_certified_cursor`,
`get_next_seed_start` and (as `required_modes`) `certified_cursor`. Under containment a *smaller*
requested set is the *weaker* request, so a default would silently over-claim coverage —
`arm_required_modes_has_no_default` asserts the absence of a default at all three levels and that
empty / bare-string / non-canonical mode sets are refused.

### One thing this forced, disclosed

`CREATE TABLE IF NOT EXISTS` is silent about drift, so a pre-R1 table would have had the new
meanings written into columns named for the old ones. `_assert_schema_current`
(`utils/seed_coverage_ledger.py:732`) now **fails closed** with the exact remedy in the message,
gated by `arm_schema_drift_fails_closed`. See §6 for the live table this found.

---

## 3. BLOCKER C — THE THIRD EXECUTION PATH

The pre-R1 report's claim of *"both entry points"* was too broad, as Beta says. `run_with_config`
invokes `run_bidirectional_test` directly in a loop and passed through neither wall.

### Whole-plan preflight

`window_optimizer.py:1156-1200`, the **first statement** of `run_with_config`'s body:

```python
_s145_domain_wall(0, max_seeds, context="run_with_config plan sizing")   # type-gate
_s145_iterations = _s145_plan_iterations(iterations)
for _s145_i in range(_s145_iterations):
    _s145_domain_wall(_s145_i * max_seeds, max_seeds,
                      context=f"run_with_config plan iteration {_s145_i+1}/{_s145_iterations}")
```

Step (1) type-gates `max_seeds` **through the wall itself** before any arithmetic uses it —
`seed_start=0` is always legal, so that call can only fail on `max_seeds`, which is what stops a
`str`/`float`/`bool` reaching the multiplication and producing a nonsense plan.
`_s145_plan_iterations` (`:1111`) validates the iteration count and raises the wall's own error
type so one `except` arm covers the whole plan.

`arm_config_mode_plan_matrix` runs Beta's four cases with `COORDINATOR_AVAILABLE` forced `False`,
which is the discriminator: `SeedDomainPreflightError` means the wall refused; `SystemExit` means
it passed and nothing sieve-related ran.

| plan | expected | result |
|---|---|---|
| `2^32 × 1` (exact bound) | pass | ✅ |
| `2^31 × 2` (final seed) | pass | ✅ |
| `2^30 × 4` | pass | ✅ |
| `2^30 × 5` (fifth escapes) | **fail, zero dispatch** | ✅ refused at `iteration 5/5` |

### Proof no sieve executes before validation

`arm_config_mode_no_sieve_before_plan_validation` does both:

* **structural** — from the live AST of `run_with_config`: the wall's line precedes every
  `run_bidirectional_test` call, and **no `Call` node of any kind** appears at a lower line number;
* **executed** — `run_bidirectional_test` is replaced with a tripwire, the `2^30 × 5` plan is run,
  and the tripwire must never fire.

### C.2 — the coercions are gone

Three wrappers passed values through `int(...)` before the wall, defeating `_require_int`:
`window_optimizer.py:760` (CLI), `agents/watcher_agent.py:1780` (WATCHER), and the new config-mode
wall. All three now pass the value **in its authoritative form**.

`arm_wrappers_do_not_coerce_types` drives `True`, `1.5` and `"0"` through **all three real
wrappers** — not the helper — and requires refusal at each. For the CLI and config wrappers a
`SystemExit` (rather than `SeedDomainPreflightError`) is treated as a **failure**, because that is
precisely what a surviving coercion would produce.

One related correction rides with C.2 and is disclosed: WATCHER previously substituted
`5_000_000` when `max_seeds` was absent, but the value the run actually uses in that case is
`10_000_000` (`window_optimizer.py:1753` and `run_bayesian_optimization`'s own default). The wall
was validating a plan nobody executes; it now uses `10_000_000`
(`agents/watcher_agent.py:1694-1701`).

---

## 4. RED-FIRST AND MUTATION EVIDENCE

**⚠ Method, disclosed because it matters.** The pre-R1 submission was never committed, so **there
is no tree to check out**. Red-first is therefore produced by reconstructing each blocker's defect
on top of the R1 tree by targeted source mutation, in a scratch `4dd5535` worktree carrying the R1
files. Each mutation is a faithful restatement of what the submitted patch did:

* **A** — raw writer public again; any object with `artifact_sha256` accepted; all other fields
  taken from the caller.
* **B** — `certified_cursor` ignores the mode set; `canonical_coverage_identity` returns the raw
  `prng_type`.
* **C** — the `run_with_config` plan wall removed; `int(...)` coercion restored in both wrappers.

The worktree additionally receives the **gitignored** dataset files (`daily3_current.json`, the
immutable `daily3-…json`, `dataset_provisioning.json`). Without them P0.5 blocks before the S145
block and the WATCHER-driving arms report a *different* `blocked_by` — a vacuous red that would
have masked what the mutation actually did. (VIR-6: the repository is not the system.)

```
CONTROL  — unmutated R1 tree            39/39 green      (red-first is meaningless otherwise)

MUTATION A — pre-R1 certification bypass          35/39
  [RED] G-PUBLICATION-BINDS-COVERAGE: failed publication creates none
  [RED] G-PUBLICATION-BINDS-COVERAGE: MUTANT unbound write is caught
  [RED] R1-A: only a canonical RunArtifactResult may certify
  [RED] R1-A: caller cannot substitute any artifact field

MUTATION B — pre-R1 coverage scope                37/39
  [RED] R1-B: Beta's containment matrix (5 rows)
  [RED] R1-B: hybrid and base canonicalize to one namespace

MUTATION C — pre-R1 wall topology                 36/39
  [RED] R1-C: config-mode whole-plan matrix (+ clean control)
  [RED] R1-C: no sieve executes before plan validation
  [RED] R1-C.2: wrappers do not coerce types

RESTORED                                39/39 green
```

**Each mutation reds only its own blocker's arms.** That specificity is the point: it shows the new
gates are bound to the defects they claim to detect and are not reacting to some shared fragility.

Every arm from the pre-R1 submission is retained and still green; the 29 prior checks became 39.

---

## 5. VERIFICATION

**Suite ×3 after the last edit** (last tracked edit `window_optimizer.py` 07:54:57; suite file
07:59:29; all three runs and this report followed):

```
RUN 1: 39/39 checks green | COMPLETION SENTINEL: PASS
RUN 2: 39/39 checks green | COMPLETION SENTINEL: PASS
RUN 3: 39/39 checks green | COMPLETION SENTINEL: PASS
```

**Staging suites — re-run SEQUENTIALLY** per Beta §10 (kept sequential for certification until the
`G-VAL-6` free-space race gets its own amendment):

| suite | result | required |
|---|---|---|
| `test_s172_staging_backpressure.py` | **50/50** | 50/50 ✅ |
| `test_s172_staging_partb.py` | **24/24** | 24/24 ✅ |
| `test_s172_elapsed_roundtrip.py` | **6/6** | 6/6 ✅ |
| `test_s172_phase5_d3_5_finalizer.py` | **60/60** | 60/60 ✅ |

**Phase-4 evaluated in the clean/committed model — method stated as required.** A local
`git clone --no-hardlinks` of the repo into scratch, checked out at `4dd5535`, the six amendment
files copied in and **committed inside the clone** (`git status --porcelain --untracked-files=no`
empty). The real repository was never written to — no commit, no branch move, no object added to
it. Result: **63/63 green**, Gate 22 included. (In the uncommitted working tree Gate 22 reds by
naming the three changed/untracked `.py` files; that is the documented file-identity sensitivity,
and the committed model is what Beta asked to see.)

**Zero diff against the staging-capacity implementation and its suites — confirmed
programmatically** (`git diff --stat 4dd5535 -- <path>`, zero lines each):

```
tests/test_s172_staging_backpressure.py   0     miner/range_miner_coordinator.py    0
tests/test_s172_staging_partb.py          0     miner/range_miner_protocol.py       0
tests/test_s172_phase4_coordinator.py     0     agent_manifests/window_optimizer.json 0
tests/test_s172_elapsed_roundtrip.py      0     utils/run_finalizer.py              0
```

The two files this amendment shares with the staging work are untouched in every staging region:
S145 hunks in `window_optimizer.py` are at lines **736 / 1074 / 1096** against staging's
**709 / 813 / 837 / 1496 / 1785**, and in `window_optimizer_integration_final.py` at **3005**
against staging's **1464**.

**No gate-12 production run. No Phase-7 soak. No pipeline, fleet or port-5700 bind.**

---

## 6. A LIVE-DATABASE SIDE EFFECT THE PRE-R1 SUITE CAUSED — found and cleaned

Not raised by Beta; found while implementing the schema guard, and reported because it touched a
production artifact.

The WATCHER-driving arms construct a real `WatcherAgent`, whose S145 block does
`DistributedPRNGDatabase()` with no arguments — which resolves **cwd-relative to the live
`prng_analysis.db`**. Running the pre-R1 gates therefore created a real `certified_coverage` table
(pre-R1 shape) in the production database. **A test that mutates the artifact it is auditing is
not a test.**

* **Measured before touching it:** 0 rows, `prng_type`/`mapping_mode` columns, created during
  yesterday's gate run — the publication hook has never executed, so no certified row has ever
  existed in that shape.
* **Action:** dropped, under an assertion that refuses to drop a non-empty coverage table. The
  legacy `exhaustive_progress` table was verified **unchanged at 15 rows** before and after.
* **Prevented from recurring:** `_watcher_harness` now substitutes a `DistributedPRNGDatabase`
  subclass pointing at a temp file. Verified after three full suite runs: `certified_coverage` is
  absent from the live DB and `exhaustive_progress` still holds 15 rows.
* **Prevented from being written into:** `_assert_schema_current` fails closed on that exact
  shape.

**Incidental confirmation of Beta's §2 figures, measured live:** the legacy table holds **15 rows**
with `MAX(seed_range_end) = 16,106,127,360` — 11,811,160,064 seeds beyond the terminus.

---

## 7. FILES CHANGED

Same set as the submission plus test additions. Nothing else.

| file | change |
|---|---|
| `utils/seed_coverage_ledger.py` | Blockers A + B: witness validation, one door, internal seam, canonical identity, containment, schema guard |
| `database_system.py` | cursor takes `test_both_modes` (no default) and canonicalizes; `record_certified_coverage` reduced to `(artifact, dataset_sha256, study_identity)` |
| `agents/watcher_agent.py` | canonicalized query; coercions removed; `max_seeds` default corrected to 10,000,000 |
| `window_optimizer.py` | Blocker C whole-plan preflight + `_s145_plan_iterations`; CLI coercion removed |
| `window_optimizer_integration_final.py` | publication hook reduced to the witness + dataset digest + study name |
| `tests/test_seed_domain_cursor_amendment.py` | 29 → **39** checks; `_FakeArtifact` becomes a negative fixture; DB isolation |

`git diff --stat 4dd5535` → **4 files changed, 386 insertions(+), 69 deletions(-)**, plus two
untracked `.py` files.

---

## 8. DISAGREEMENTS

**None with the ruling.** All three blockers are accepted as stated and implemented as specified;
Blocker A in particular was a correct finding against my own submitted evidence, and Blocker B's
answer is better than either option I framed — a hybrid identity is the same base family under
variable skip, and treating `java_lcg_hybrid` as its own namespace would have split one logical
search in two.

Four items are **disclosed additions**, each forced by a blocker rather than chosen:

1. **`_assert_schema_current`** — required by Blocker B; without it a pre-R1 table would be written
   into silently (§2).
2. **`test_both_modes` mandatory, no default** — Blocker B's containment law makes a default
   actively unsafe (§2).
3. **The `max_seeds` default correction, 5,000,000 → 10,000,000** — found while removing the C.2
   coercions; the wall was validating a plan the run does not execute (§3).
4. **Suite DB isolation and the drop of the empty live table** — §6.

One judgement call is flagged for the record: WATCHER reads `test_both_modes` by **truthiness**.
This is fail-safe by direction — a non-bool that reads as true yields `{constant, variable}`, the
*stronger* request, which counts *fewer* certified records and can only under-claim coverage
(causing a re-sweep), never over-claim it. Stated at `agents/watcher_agent.py:1704`.

---

## 9. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

* **execution proof:** per-arm PASS/FAIL, explicit `COMPLETION SENTINEL` and matching exit code;
  control / mutation / restored runs all captured.
* **clean control:** control tree asserted green before any mutation (an assert, not an
  observation); `arm_config_mode_plan_matrix` and `arm_preflight_zero_dispatch_executed` each pair
  a fault injection with a control that must reach the next stage.
* **fault-injection control:** three blocker mutations, each proven to red **only** its own arms;
  the legacy-clobber control; the retained fabricated-interval mutant.
* **completion sentinel:** `COMPLETION SENTINEL: PASS — seed-domain cursor amendment green`.
* **unavailable-observer behavior:** cursor lookup failure blocks Step 1 rather than reporting a
  fabricated `seed_start=0`; schema drift fails closed; no arm reports PASS on an unobserved
  surface.
* **audit claim scope:** the seed-domain terminus, coverage cursor, coverage ledger and the three
  Step-1 execution paths, on this VM101 tree at `4dd5535` plus the working-tree diff. No claim
  about rig-side state, a live fleet run, Gate 12, or `n_parallel > 1`.
* **searched surfaces:** `docs/` and the governance trail · `git ls-files` · `git diff/worktree` ·
  the live VM101 filesystem including gitignored files · live Python imports and
  `inspect.signature` · live execution of `WatcherAgent.run_step`, `run_bayesian_optimization` and
  `run_with_config` · live SQLite schema and row introspection on the production database · AST
  scans of every `.py` in the tree.
* **unavailable surfaces:** the three rigs (`192.168.3.122/.156/.164`, *No route to host* this
  session — which is why the WATCHER harness stubs preflight and P0.5) · Proxmox host kernel logs ·
  any live-fleet or GPU execution path. **No claim here depends on them.**
* **governance trail searched:** the R1 ruling as carried by the brief, the pre-R1 ruling, the
  S172 staging-capacity arc (`4b1aad6`, `4dd5535`) for the file-ownership boundary, and D3.5's
  frozen result and seed-domain contracts in `utils/run_finalizer.py`.
* **chapters searched:** not required — no claim here concerns sieve mathematics, kernel semantics
  or feature provenance.
