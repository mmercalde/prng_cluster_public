# CLAUDE CODE INSTRUCTIONS — SEED-DOMAIN / COVERAGE-CURSOR AMENDMENT

**Host:** VM101, repo `~/distributed_prng_analysis`, HEAD `c7058d8`.
`source ~/venvs/torch/bin/activate` before every test. Long suites:
`python3 -u <suite> | tee /tmp/<name>.log` — never `| tail`.

**Authority:** Team Beta ruling *"S145 / SEED-DOMAIN SWEEP TERMINUS AND COVERAGE AUTHORITY"*
(2026-08-07). Beta was explicit that this is a **separate amendment from the staging-capacity
work** — *"Do not merge their production code into one mega-patch… They have different
authorities, different test suites and different rollback surfaces."* The staging amendment may be
in flight concurrently; **do not touch its files or its suite.**

**Hard constraints:** no commit, no push, **no pipeline launch, no fleet launch, no port 5700
bind**. Gate 12 and the Phase-7 soak are HELD. Do not modify Optuna, the strategy system, or any
sieve mathematics — Beta §9: *"No Optuna-removal work is authorized or required… The cursor
correction must not be disguised as a strategy-system rewrite."* If a fix appears to need a file
outside scope, STOP and report.

**Base verification, first:** `git log --oneline -1` = `c7058d8`; tracked tree clean (untracked
runtime residue — WAL sidecars, `*.stale_*` rotations, delivered briefs — is expected and is NOT a
stop condition); the existing suites green. Report the state; stop only on tracked-source drift.

---

## 1. THE TERMINUS (Beta §1)

The governed discovery domain is **`[0, 2^32)` — exactly `0 <= seed < 4,294,967,296`.**

**No run may begin at `2^32`, cross `2^32`, or publish a candidate outside that interval.**

The mathematical java_lcg state space is 48-bit; that does **not** authorize sweeping it. Seed-Domain
v1.1 deliberately kept `uint32` canonical storage and labels the artifact as the `high16 = 0`
stratum. Beta: *"STOP AT 2^32. Do not continue pending yield analysis. Do not continue into 2^48."*

**Use the existing canonical constant — do not define a second one.** `utils/run_finalizer.py:277`
already holds `SEED_DOMAIN_EXCLUSIVE_MAX = 2 ** 32`, enforced fail-closed at `:533`. Import or
otherwise share that single authority (Beta §7). A mutation that changes one boundary **must red
both the pre-dispatch gate and the finalizer parity gate.**

## 2. THE LEGACY TRACKER IS DEAUTHORIZED (Beta §§2-4)

`exhaustive_progress` (15 rows, `java_lcg`/`bidirectional`, max end 16,106,127,360) **has zero
certified authority.** This is stronger than "clip it to 2^32":

- it advanced beyond the governed frontier with no terminus;
- its first row has been destructively overwritten at least twice by short runs;
- it contains a ~1.07-billion-seed hole at `[1,000, 1,073,741,824)`;
- it has no domain contract and no completion state;
- `best_seed` is NULL on all 15 rows and `best_score` 0.0 on 13 — it records extent, never yield;
- `get_next_seed_start` (`database_system.py:330`) takes `MAX(seed_range_end)` rather than
  computing verified contiguous coverage.

**Required dispositions:**

- **Rows 5-15** (`[4,294,967,296, 16,106,127,360)`) → classified **LEGACY OUT-OF-DOMAIN /
  NON-CERTIFYING TELEMETRY**. **Do not delete them. Do not rewrite their numeric values. Do not
  fold them back into `[0, 2^32)`.** They may never be used as current coverage, certified
  progress, a cursor source, a v1.1 ancestor, or evidence of complete coverage.
- **Rows 1-4 are also NOT certified coverage.** Beta: *"the old 16.1B tracker contributes zero
  certified progress."* **The new certified v1.1 coverage stream starts at zero.** No provenance
  migration is authorized here.
- **Consequence, stated so it is not treated as an omission:** the low-range hole needs **no
  surgical repair** — the table holding it is no longer the authority.

The legacy table may be retained and displayed as historical telemetry. It **may not select the
seed start of a certified generation.**

## 3. COVERAGE LEDGER v1 (Beta §5)

The long-missing *"separate coverage-ledger deliverable"* becomes real. **Append-only or
equivalently history-preserving.** `INSERT OR REPLACE` of the sole record for a starting seed is
**not acceptable** — a 1,000-seed smoke test must never erase evidence of an earlier
billion-seed production interval (that is exactly what happened, twice).

Each certified interval binds at minimum:

```
coverage_id / immutable record identity
run_id
search/study identity
prng_type
mapping/skip mode as applicable
seed_domain_contract
seed_start
seed_end_exclusive
dataset_sha256
repository revision
artifact identity/hash
completion/publication status
timestamp
```

**A range becomes CERTIFIED COVERED only after the corresponding canonical publication succeeds.**
Beta, verbatim: *"Starting a run is not coverage. Receiving all GPU results is not coverage.
Writing a provisional DB row is not coverage. The canonical retained artifact is the evidence
wall."*

## 4. CURSOR LAW — FIRST GAP, NOT MAX END (Beta §6)

`get_next_seed_start()` shall no longer mean `MAX(seed_range_end)` — that rule is invalid in the
presence of gaps and contaminated history. For a governed domain `[D0, D1)`, derive from the set of
**certified** intervals:

```
normalize valid certified intervals
clip/reject by exact domain contract
merge overlaps (for computation only)
start at D0
return the first uncovered seed
```

Worked example Beta supplied — certified `[0, 1000)` and `[2^30, 2^31)` ⇒ next cursor is **1000**,
not `2^31`.

When the normalized certified union covers `[0, 2^32)`:

```
status = COMPLETE
next_seed_start = NONE
```

**There is no `4,294,967,296` next run.** The completion condition must be representable
explicitly — not signalled by an out-of-range number.

## 5. PRE-DISPATCH SEED-DOMAIN WALL (Beta §7)

The finalizer already fails out-of-domain coverage correctly, but **after** the GPU work. Add the
same law **before** dispatch:

```
seed_start >= 0
seed_count > 0
seed_start < 2^32
seed_start + seed_count <= 2^32
```

A violation must terminate **before** fleet work assignment, sieve execution, staging, or any
coverage mutation. Reason string should identify the governed contract, e.g.:

```
seed_domain_preflight: requested [start,end) exceeds v1.1-stratum [0,4294967296)
```

**Do not invent a separate domain constant** — share the finalizer's (§1).

## 6. REQUIRED GATES (Beta §10)

- **G-DOMAIN-PREFLIGHT** — `start=0,count=1` PASS · end exactly `2^32` PASS · `start=2^32` FAIL
  before dispatch · `end=2^32+1` FAIL before dispatch · negative start FAIL. **Zero GPU
  assignments on every failure.**
- **G-CURSOR-FIRST-GAP** — certified intervals with an interior hole return the first hole, never
  maximum end.
- **G-CURSOR-COMPLETE** — certified union exactly covers `[0,2^32)` ⇒ `COMPLETE`, no numeric next
  seed, **no WATCHER run generated.**
- **G-LEGACY-NONAUTHORITY** — populate the old tracker with the current 16.1B history; prove the
  new certified cursor ignores it completely.
- **G-NO-REPLACE-CLOBBER** — record a large certified interval starting at zero, then run a
  1,000-seed smoke/test record at zero; prove the certified interval remains intact and the smoke
  run cannot replace it.
- **G-PUBLICATION-BINDS-COVERAGE** — a failed canonical publication creates **no** certified
  interval; a successful one creates **exactly one** immutable certified interval bound to that
  artifact.
- **G-OUT-OF-DOMAIN-LEGACY** — historical rows above `2^32` may be displayed/audited but can never
  enter the v1.1 normalized coverage union.

Each gate proven **red first** against the pre-amendment tree (worktree at `c7058d8`). Mutation
evidence on G-CURSOR-FIRST-GAP and G-PUBLICATION-BINDS-COVERAGE.

## 7. WATCHER INTEGRATION

`agents/watcher_agent.py:1662-1701` currently calls `_db.get_next_seed_start(prng_type, chunk_size)`
and advances `seed_start`. It must consume the new certified cursor, and it must handle the
**COMPLETE** state explicitly — no run is generated, and the operator is told the domain is
exhausted rather than being handed a number.

**Note for your report, not for you to resolve:** the live manifest still carries
`use_range_miner = False` / `use_persistent_workers = True`, so a WATCHER-driven Step 1 today
takes the PWC-TCP path. Record whether your changes are backend-independent.

## 8. EVIDENCE AND REPORT

Final-state discipline: canonical-host runs **after** the last edit; the report written **after**
those runs.

`docs/CLAUDE_CODE_REPORT_SEED_DOMAIN_CURSOR_AMENDMENT.md`, containing:

1. Per-ruling-section implementation notes with `file:line`.
2. The Coverage Ledger v1 schema as built, and how append-only is enforced (constraint? no
   REPLACE path? both?).
3. Red-first evidence per gate; mutation evidence where required.
4. Full new suite green on VM101 ×3 after the last edit.
5. `test_s172_staging_backpressure.py`, `test_s172_staging_partb.py`,
   `test_s172_phase4_coordinator.py` unaffected — **this amendment must not touch them**; confirm
   programmatically.
6. Files changed. Expect `database_system.py`, `agents/watcher_agent.py`, the pre-dispatch site,
   a new ledger module, and a new suite. Anything else must be justified.
7. Explicit statement that rows 5-15 were **not** deleted, renumbered or folded, and that the
   legacy table is untouched except where read-paths were redirected.
8. Any disagreement with this brief **reported, not worked around.**
