# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D4.md — REV3

**REV3 changelog** — absorbs the Team Beta pre-code review (six narrow
corrections, no architectural change): **[B1]** the return contract is frozen as
`BackendAssemblyResult` + `AssemblyMeasurement` — REV2 left the choice between
four incompatible APIs; **[B2]** G3 cannot require literal `timing` equality,
since `assemble_trial` records a live `perf_counter` delta; **[B3]** `peak_rss_bytes`
defined as process-**tree** peak, with `ru_maxrss` KiB scaling and the isolated
Phase-6 benchmark separated from D4's unit assertion; **[B4]** input contract is
`List[Dict[str, Any]]`, not `Sequence[Mapping]` — the live assembler rejects
non-dict manifests; **[B5]** measurement computed **after** delegation so the
canonical D1.1 exception survives; **[B6]** G7 becomes AST-based, not substring
matching.

**REV2 changelog — rebase only, no scope change.** Base moved `46a3828` →
`f163199` (D3.5-B closed; `a63c361` implementation + `f163199` docs-only
prerequisite correction — every code cite below is unaffected by the latter). Blocking non-regression for D3.5 is now **60/60**
(51 F-gates + S1-S9). `docs/PHASE6_PREREQS.md` removed from D4's commit set — it
landed with D3.5-B. G5 notes that the sidecar now carries 32 keys including the
nine Seed-Domain v1.1 stratum fields. All four reuse cites re-verified unchanged.

**S172 RANGE-MINER — Phase 5, Deliverable D4: the `serial_reference` assembly
backend behind the two-backend interface**

**Audience:** Claude Code on VM 101 (`michael@192.168.3.177`), in
`~/distributed_prng_analysis`. You write and iterate; you do NOT commit, push,
or run WATCHER. When gates + non-regression are green, STOP and report.

**Frozen against HEAD `f163199`** (D3.5-B closed). Authority:
`docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §6.7 / §6.7.B / §17, and
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5.md` §D4.

---

## 0. What is already built, and what D4 actually adds

**Read this section before anything else.** The original §D4 text predates D1.1,
D3, D3.25 and D3.5. Most of what it describes now exists. Verified at
`a63c361`:

| §D4 requirement | status |
|---|---|
| shared record→field columnizer, one logical pass, no 22 comprehensions, no pandas | **DONE** — `utils/canonical_arrays.records_to_arrays` (D3) |
| serial spool read → four populations → 24-field records | **DONE** — `miner/range_miner_npz_writer.assemble_trial` (D1.1, `:450`) |
| global highest-score dedup, ascending seed order, 22-array concatenation, final encoding checks | **DONE** — `utils/run_finalizer.finalize_run` (D3.5) |
| immutable publication | **DONE** — D3.5 |
| stratum-labelled sidecar (Seed-Domain v1.1) | **DONE** — D3.5-B, 32 keys |

**Nothing in the repository implements a backend selector.** Re-verified at
`f163199`: `grep` for `assembly_backend` / `serial_reference` /
`process_sharded` across all `*.py` returns one hit, in a Phase-4 test
whitelist.

**Therefore D4 is a narrow deliverable with three jobs:**

1. **Define the backend interface** the spec names
   (`assembly_backend = serial_reference | process_sharded`) so D5 has a seam to
   plug into rather than inventing one.
2. **Implement `serial_reference` behind it**, wrapping the existing D1.1
   assembly path — a thin, explicit, selectable backend, **not** a
   reimplementation.
3. **Establish it as the measured baseline** §17's promotion rule requires: the
   correctness oracle, the fallback, the benchmark reference, the debug mode.

**D4 writes almost no new assembly logic.** If you find yourself reimplementing
spool reading, record derivation, columnization, dedup, ordering or publication,
**STOP** — that code exists and D4 must call it.

## 1. Non-negotiable working rules

1. **Read live source before every claim.** Cites verified at `f163199`.
2. **Reuse, never reimplement.** `assemble_trial` (D1.1), `records_to_arrays`
   and `validate_array_bundle` (D3), `build_mode_records` (D3.25),
   `finalize_run` (D3.5). Duplicating any of them is a stop condition.
3. **Each gate must FAIL on wrong behavior.**
4. **Independent oracles** — hand-transcribed expectations, never imported from
   the module under test (G9 / E8 / C1 lesson).
5. STOP at the gate. No commit/push/WATCHER.

## 2. Scope

**Create:** `miner/assembly_backends.py` — the backend interface plus
`serial_reference`; `tests/test_s172_phase5_d4_serial_backend.py`.

**Modify:** `tests/test_s172_phase4_coordinator.py` (gate-22 registration only).

**Must NOT modify:** `miner/range_miner_npz_writer.py`;
`utils/canonical_arrays.py`; `utils/canonical_records.py`;
`utils/run_finalizer.py`; `utils/prng_encoding.py`;
`window_optimizer_integration_final.py`; `persistent_worker_coordinator.py`;
`zmq_sqlite_coordinator.py`; `convert_survivors_to_binary.py`; any D0-D3.5
test; `prng_analysis.db`; WATCHER. Discovering a required change triggers
**STOP** — if the existing modules cannot support the backend seam without
edits, that is a finding for review, not something to fix in D4.

## 3. The backend interface

One interface, two eventual implementations, chosen by name:

```python
ASSEMBLY_BACKENDS = ("serial_reference", "process_sharded")

def get_assembly_backend(name: str) -> AssemblyBackend:
    """Resolve a backend by name. Unknown name -> ValueError (hard fail,
    never a silent default to serial_reference)."""
```

**[B1] The return contract is frozen.** REV2 said measurement was "returned
alongside the assembly, or stored on the backend" — which left four
incompatible choices (change the return type / mutate
`MinerTrialAssembly.timing` / hold mutable `last_measurement` state / return an
undocumented tuple). Use exactly this:

```python
@dataclass(frozen=True)
class AssemblyMeasurement:
    backend_name: str
    wall_seconds: float
    manifest_count: int
    spool_bytes_read: int
    survivor_row_count: int
    peak_rss_bytes: Optional[int]


@dataclass(frozen=True)
class BackendAssemblyResult:
    assembly: MinerTrialAssembly
    measurement: AssemblyMeasurement


class AssemblyBackend(Protocol):
    backend_name: str

    def assemble(self, run_id: str,
                 manifests: List[Dict[str, Any]]) -> BackendAssemblyResult: ...
```

**[B4] `manifests` is `List[Dict[str, Any]]`, not `Sequence[Mapping]`.** Verified
at `f163199`: `assemble_trial` (`miner/range_miner_npz_writer.py:450`) declares
that type and enforces it — `if not isinstance(manifest, dict)` at `:280`. A
backend must expose exactly the input domain the shared assembler already
validates. **Do not copy, normalize or convert arbitrary mappings before
delegation.**

The embedded `assembly` must be the **unmodified** object `assemble_trial`
returned. Do **not**: replace or extend its `timing` dict; add backend metrics
to it; mutate either NPZ path; hold mutable `last_measurement` state on the
backend; or return a tuple.

Requirements:

- **`process_sharded` is declared but NOT implemented in D4.** Resolving it
  raises `NotImplementedError` naming D5. It must appear in `ASSEMBLY_BACKENDS`
  so the selector's shape is frozen now and D5 changes no interface.
- **No silent default.** An unknown or missing backend name fails closed. Per
  §17, `serial_reference` is the *production default* only as an explicit
  configured value, never as a fallback after an error.
- The interface carries **no** publication, dedup or ordering responsibility —
  those belong to D3.5's finalizer. A backend produces a `MinerTrialAssembly`
  and stops.

## 4. `serial_reference`

A thin wrapper over the existing D1.1 path. **[B5] The order is binding** —
measurement must not change failure behaviour:

```python
started = time.perf_counter()
assembly = assemble_trial(run_id, manifests)      # NOT wrapped in try/except
wall_seconds = time.perf_counter() - started
measurement = AssemblyMeasurement(...)            # computed from validated input
return BackendAssemblyResult(assembly=assembly, measurement=measurement)
```

```text
1. start timer
2. call assemble_trial WITHOUT catching or translating its exception
3. stop timer after successful return
4. compute measurement fields from the NOW-VALIDATED manifests and assembly
5. return BackendAssemblyResult
```

**Why the order matters:** computing `sum(m["expected_size"] for m in manifests)`
*before* delegation lets a malformed manifest raise a raw `KeyError` from the
wrapper instead of D1.1's canonical fail-closed spool error. On any assembly
failure: propagate the original exception **unchanged**, return no result,
publish no partial measurement, update no backend state.

Beyond delegation and measurement it must not alter, pre-filter, re-order or
post-process the assembly in any way.

Its four documented roles, per §6.7.B, belong in the module docstring:
**correctness oracle**, **fallback**, **benchmark baseline**, **debug mode**.

## 5. Measurement — what §17 needs

§17 promotes `process_sharded` to production default only on **≥20% median
end-to-end improvement** over `serial_reference` plus three other conditions.
That comparison needs a baseline captured the same way for both backends, so D4
defines the measurement now rather than letting D5 invent one.

Record per `assemble()` call, returned alongside the assembly (or on the
backend, not mutated into `MinerTrialAssembly`, whose shape is frozen):

```text
backend_name
wall_seconds            (perf_counter around the assemble call)
manifest_count
spool_bytes_read        (sum of expected_size across manifests)
survivor_row_count      (constant + variable canonical records)
peak_rss_bytes          (resource.getrusage, best-effort; None if unavailable)
```

Deliberately **not** measured here: publication time (D3.5's, and it is
backend-independent), and GPU time (not Phase 5's).

**[B3] `peak_rss_bytes` semantics — frozen.** Defined as *the maximum aggregate
resident memory of the backend process tree during the measured `assemble()`
call.* For `serial_reference` the tree is just the current process, so on Linux:

```python
resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024   # ru_maxrss is KiB
```

Three qualifications that must be recorded in the module docstring:

- on Linux `ru_maxrss` is **KiB** and must be scaled by 1024;
- it is a **process-lifetime high-water mark**, not automatically this call's
  peak;
- for D5 the parent's RSS alone is **not compliant** — §17 requires the peak
  aggregate host RAM of parent plus concurrently live workers, and
  `RUSAGE_CHILDREN.ru_maxrss` does not establish that.

**Benchmark isolation rule.** The authoritative §17 promotion benchmark must run
each measured backend in a **fresh process**, so a previous call's lifetime
high-water mark cannot contaminate the next. Therefore:

```text
D4 measurement          : proves the telemetry field and serial instrumentation exist
Phase 6 isolated bench  : produces the authoritative promotion measurements
```

D4's gate may assert only that `peak_rss_bytes` is `None` **or** a positive
integer. It must **not** claim the in-harness value is the final §17 comparison
number.

## 6. Gate — `tests/test_s172_phase5_d4_serial_backend.py`

- **G1 selector shape:** `ASSEMBLY_BACKENDS` contains exactly
  `("serial_reference", "process_sharded")`, asserted against a hand-written
  literal.
- **G2 fail-closed resolution:** an unknown name raises `ValueError`; an empty
  or `None` name raises; **no path returns `serial_reference` as a silent
  default.** Mutate the resolver to default-on-unknown → gate must red.
- **G3 delegation identity [B2]:** compare `backend_result.assembly` against a
  direct `assemble_trial` call, field by field, for: `run_id`;
  `bidirectional_constant`; `bidirectional_variable`; all four maps;
  `canonical_records_constant` and `canonical_records_variable` (element-wise);
  `directional_counts`; `binary_npz_path` and `all_npz_path` (both `None` —
  D3.5 Ruling E, backends never populate them).

  **`timing` cannot be compared for equality.** `assemble_trial` records
  `{"assembly_s": time.perf_counter() - started}` (`:581`), so two calls
  necessarily differ. Assert only: both objects contain `"assembly_s"`; both
  values are finite; both are `> 0`. **Additionally assert the backend inserted
  no backend-specific key into `assembly.timing`.**
- **G4 `process_sharded` declared but unimplemented:** resolving it raises
  `NotImplementedError` whose message names D5. It must NOT fall back to serial.
- **G5 end-to-end through the finalizer:** drive the D1.1 real-lifecycle fixture
  → `serial_reference.assemble` → feed
  `backend_result.assembly.canonical_records_constant` +
  `...canonical_records_variable` into `finalize_run` → assert a published generation
  whose 22 arrays match hand-computed expectations, and whose sidecar records
  the run coverage. Post-D3.5-B the sidecar carries **32** keys including the
  nine Seed-Domain v1.1 stratum fields; assert those are present and correct,
  but do **not** re-derive assertions D3.5-B already gates (S1-S9) — G5 proves
  the backend reaches a valid publication, not that the sidecar contract is
  right. Reuses D3.5's harness fixture pattern; does not
  reimplement publication assertions already gated there. **The backend itself
  must never call the finalizer** — that is G7's business.
- **G6 measurement:** assert the exact immutable `AssemblyMeasurement` object
  and its field semantics — `backend_name == "serial_reference"`;
  `wall_seconds > 0` and finite; `manifest_count` equals the input length;
  `spool_bytes_read == sum(expected_size for every successfully validated
  manifest)`; `survivor_row_count == len(canonical_records_constant) +
  len(canonical_records_variable)`; `peak_rss_bytes` is `None` **or** a positive
  integer (see §5's isolation rule — do not assert it is the §17 number).
- **G7 no reimplementation — AST-based [B6]:** do **not** substring-search for
  words like `sort`, `open` or `array`; that is fragile and defeats itself.
  Use `ast` to prove of `miner/assembly_backends.py`:

```text
no FunctionDef named assemble_trial
assemble_trial IS imported from miner.range_miner_npz_writer
no call to builtins.open
no call to hashlib.sha256 / hashlib.new
no call to numpy.array / asarray / empty / zeros
no call to numpy.savez / savez_compressed
no call to sorted
no .sort() invocation
no call to records_to_arrays
no call to build_mode_records
no call to finalize_run
```

  `finalize_run` belongs to G5's **test** path only, never to the backend module.
- **G8 mutation proof** — kill each of: resolver defaults to serial on unknown
  name; `process_sharded` silently resolves to serial; the wrapper filters or
  re-orders records before returning; the wrapper populates an NPZ path field;
  `ASSEMBLY_BACKENDS` drops `process_sharded`; measurement returns a constant
  instead of a real timing; **and [B5] measurement computed BEFORE delegation,
  so a malformed manifest raises from the wrapper rather than the canonical
  assembler — the gate must prove the original D1.1 exception survives
  unchanged (assert the exception type and that its message is D1.1's, not a
  `KeyError`).** Report each red signature and its attribution.

**Blocking non-regression:** D3.5 **60/60** (51 F-gates + S1-S9, post-D3.5-B), D3.25 13/13, D3 10/10, D3.0 10/10,
D2 7/7, D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4 63/63, Phase 3 17/17. Capture
the baseline green at `f163199` **before** any edit.

## 7. Stop conditions

- the backend seam cannot be added without modifying a must-not-modify module;
- `assemble_trial`'s signature or return shape proves unsuitable as the
  interface contract (report it; do not change D1.1 to fit);
- any gate passes only by weakening it;
- you find yourself writing assembly, columnization, dedup, ordering or
  publication logic — that code exists.

## 8. Report

Diff + status, full command/output evidence, the pre-edit baseline, mutation
evidence with per-mutant red signatures and attribution, and explicit
confirmation that no D0-D3.5 production module or test was modified. Then STOP
for Team Alpha review.
