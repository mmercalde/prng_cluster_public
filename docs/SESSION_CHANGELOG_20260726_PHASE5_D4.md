# SESSION_CHANGELOG_20260726_PHASE5_D4.md

**S172 Phase 5, Deliverable D4** — the `serial_reference` assembly backend
behind the two-backend interface.

Spec: `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D4.md` (REV3, Team Beta
approved).
Base: HEAD `f163199` (D3.5-B closed; Phase-6 prerequisite gates restored).
**Not committed, not pushed, WATCHER not run** — stopped at the gate for Team
Alpha review, per §1.5 / §8 of the brief.

---

## 1. Files (exactly the scope §2 authorizes)

```text
miner/assembly_backends.py                      NEW,  281 lines
tests/test_s172_phase5_d4_serial_backend.py     NEW, 1022 lines
tests/test_s172_phase4_coordinator.py           +19 / -0   (gate-22 registration only)
```

Plus this changelog. Nothing else in the tree was touched — see §6.

## 2. What D4 actually added, and what it deliberately did not

§0 of the brief is the governing framing: most of the original §D4 text predates
D1.1, D3, D3.25 and D3.5, and describes work that now exists. This deliverable
therefore did **three** things and no more.

**2.1 The backend interface (§3).** `ASSEMBLY_BACKENDS = ("serial_reference",
"process_sharded")` — both names declared now so D5 adds an implementation and
changes no interface. `get_assembly_backend(name)` fails closed: an unknown,
empty, or non-`str` name raises `ValueError`; `process_sharded` raises
`NotImplementedError` naming D5. **No path returns `serial_reference` as a
silent default.** Per §17 it is the production default only as an explicitly
configured value, never a post-error fallback.

**2.2 The frozen return contract [B1].** REV2 left four incompatible choices;
REV3 froze one, and it is implemented verbatim:

```python
@dataclass(frozen=True)
class AssemblyMeasurement:      # backend_name, wall_seconds, manifest_count,
    ...                         # spool_bytes_read, survivor_row_count,
                                # peak_rss_bytes

@dataclass(frozen=True)
class BackendAssemblyResult:    # assembly, measurement
    ...
```

The embedded `assembly` is the **unmodified** object `assemble_trial` returned.
The backend does not extend its `timing` dict, does not add backend metrics to
it, does not touch either NPZ path, holds no mutable `last_measurement` state,
and does not return a tuple.

**2.3 `serial_reference` as the measured baseline (§4, §5).** A thin wrapper:
delegate, measure, return. Its four §6.7.B roles — correctness oracle, fallback,
benchmark baseline, debug mode — are in the module docstring, as are the three
frozen `peak_rss_bytes` qualifications [B3] (Linux `ru_maxrss` is KiB and is
scaled by 1024; it is a process-*lifetime* high-water mark, not automatically
this call's peak; and the parent's RSS alone is **not** §17-compliant for D5,
because `RUSAGE_CHILDREN.ru_maxrss` does not establish a concurrent aggregate).
The benchmark isolation rule is recorded there too: D4's measurement proves the
telemetry field and the serial instrumentation exist; the **authoritative §17
promotion numbers come from the Phase-6 isolated benchmark**, each backend in a
fresh process.

**What D4 did NOT write.** No spool reading, manifest validation, directional-map
construction, record derivation, columnization, dedup, ordering or publication.
Every one of those exists — `assemble_trial` (D1.1), `records_to_arrays` /
`validate_array_bundle` (D3), `build_mode_records` (D3.25), `finalize_run`
(D3.5) — and D4 calls or defers to it. The module deliberately does not import
`utils/run_finalizer` at all: a backend produces a `MinerTrialAssembly` and
stops. G7 proves this at AST level (§4 below).

## 3. The three REV3 corrections, as implemented

**[B5] measurement order is binding.** The wrapper body is five commented steps:
start `perf_counter` → call `assemble_trial` **with no `try`/`except`** → stop
the timer on successful return only → compute the measurement from the
now-validated manifests → return. `spool_bytes_read` uses a direct
`m["expected_size"]` subscript *after* delegation, with an inline comment saying
why: computing it **before** would let a malformed manifest raise a raw
`KeyError` out of the wrapper instead of D1.1's canonical fail-closed
`SpoolIdentityError`. Mutant M7 injects exactly that inversion and dies (§4).

**[B4] input domain is `List[Dict[str, Any]]`, not `Sequence[Mapping]`.**
Verified at `f163199`: `assemble_trial` declares that type
(`miner/range_miner_npz_writer.py:450`) and enforces it (`isinstance(manifest,
dict)`, `:280`). The backend exposes exactly that domain and **nothing is
copied, normalized or converted before delegation**. G1 asserts the annotation
contains `List`/`Dict` and does *not* contain `Sequence`/`Mapping`, so a later
widening of the domain reds the gate.

**[B2] `timing` cannot be compared for equality.** `assemble_trial` records
`{"assembly_s": time.perf_counter() - started}` (`:581`), so two calls
necessarily differ. G3 compares the **twelve stable fields** for equality —
`run_id`, both `bidirectional_*` sets, all four maps, both canonical record
lists (element-wise, order included), `directional_counts`, and both NPZ paths
(each `None`, D3.5 Ruling E) — and for `timing` asserts only that `assembly_s`
is present, finite and `> 0` in both objects, **plus** that
`sorted(assembly.timing) == ("assembly_s",)`, proving the backend added no key
of its own.

## 4. Gates — `tests/test_s172_phase5_d4_serial_backend.py`, 8/8 green

G1 selector shape + frozen return contract · G2 fail-closed resolution ·
G3 delegation identity · G4 `process_sharded` declared/unimplemented/names D5 ·
G5 end-to-end backend → D3.5 finalizer → published generation ·
G6 §17 measurement · G7 no reimplementation (AST) · G8 mutation proof.

The fixture drives the **real** post-D1.0 lifecycle, reusing D1.1's harness
pattern: real coordinator, real durable ledger, real assigned stripes, real
staged spool files on disk through `stage_inline_shard`, real
`publish_attempt` → `publish_shard`. Two-mode (workflow phases 1/2/3/4),
multi-stripe, multi-sub-stripe. Every oracle is hand-transcribed; nothing is
imported from the module under test (§1.4, the G9 / E8 / C1 lesson).

**G5's hand-computed row order matters.** The four L2 winners are seeds
`1(constant), 2(variable), 12(constant), 15(variable)` — the two modes
**interleave** under the finalizer's global seed-ascending order, so a per-mode
concatenation or a mode-major ordering would be visible in the 22 arrays. G5
also asserts the 32-key sidecar and the nine Seed-Domain v1.1 stratum values are
present and correct, but does **not** re-derive what D3.5-B's S1-S9 already
gate: G5 proves the backend *reaches* a valid publication, not that the sidecar
contract is right. **The backend itself never calls the finalizer** — that path
is the test's, and G7 owns the source-level proof.

**G7 is AST-based [B6]**, not substring matching. Over
`miner/assembly_backends.py` it proves: no `FunctionDef` named `assemble_trial`;
`assemble_trial` **is** imported from `miner.range_miner_npz_writer`; no call to
`open`, `hashlib.sha256`/`new`, `numpy.array`/`asarray`/`empty`/`zeros`,
`numpy.savez`/`savez_compressed`, `sorted`, any `.sort()` invocation, or
`records_to_arrays` / `build_mode_records` / `finalize_run`. Both dotted call
targets and trailing attribute names are checked, so an attribute-spelled call
cannot slip past. A module-namespace pass confirms `np`, `numpy`, `hashlib`,
`json` and `subprocess` were never imported at all.

### 4.1 Mutation evidence — 9 mutants, every one red

| Mutant | Red in | Red signature |
|---|---|---|
| M1 resolver defaults to serial on unknown name | G2 fail-closed resolution | `AssertionError: expected ValueError, nothing was raised` |
| M2 `process_sharded` silently resolves to serial | G4 declared-but-unimplemented | `AssertionError: expected NotImplementedError, nothing was raised` |
| M3 wrapper re-orders records before returning | G3 (element-wise records) | `AssertionError: canonical_records_constant: backend [{'seed': 12, …` |
| M3b wrapper filters records before returning | G3 (element-wise records) | `AssertionError: canonical_records_variable: backend [{'seed': 2, …` |
| M4 wrapper populates an NPZ path field | G3 (D3.5 Ruling E) | `AssertionError: all_npz_path: backend '/tmp/…' != direct None` |
| M4b wrapper injects a key into `assembly.timing` | G3 [B2] | `AssertionError: … ['assembly_s', 'backend_wall_s'] != ['assembly_s']` |
| M5 `ASSEMBLY_BACKENDS` drops `process_sharded` | G1 selector shape | `AssertionError: ('serial_reference',)` |
| M6 measurement returns a constant timing | G6 (`wall_seconds > 0`) | `AssertionError: wall_seconds must be a real perf_counter delta, got 0.0` |
| M6b `spool_bytes_read` is a constant | G6 (independent on-disk oracle) | `AssertionError: (1, 2212)` |
| M7 **[B5]** measurement computed BEFORE delegation | G8/[B5] | `AssertionError: expected SpoolIdentityError, got KeyError: 'expected_size'` |

M7 is the mutant REV3 [B5] specifically calls for. Against the mutated wrapper a
manifest missing `expected_size` raises a raw `KeyError` from the wrapper's own
measurement expression; against the shipped module the same input reaches D1.1's
spool wall (`range_miner_npz_writer.py:360`) and raises the canonical
`SpoolIdentityError`, message intact. The probe asserts the exception type
**and** that the message is D1.1's (carries the run/stripe context and the
`size N != expected_size None` wording), and it is run against the real module
too — so it is a positive assertion about the shipped ordering, not only a
mutant killer.

### 4.2 Finding — the first mutation run produced FALSE EVIDENCE

Recorded because it nearly passed as a green.

Each mutant is loaded as its **own module object**, so it defines its own
`BackendAssemblyResult` and `AssemblyMeasurement` classes. The first version of
the shared probes type-checked against the *unmutated* module's classes
(`isinstance(result, AB.BackendAssemblyResult)`). Six mutants — M3, M3b, M4,
M4b, M6, M6b — therefore died on that type check **before their injected defect
was ever exercised**, every one reporting an identical, meaningless signature:

```text
AssertionError: <class '_d4_mutant_3.BackendAssemblyResult'>
```

The gate was 8/8 green and the mutation table was fully populated, but six rows
attributed a kill to an assertion that had not run. A mutant that had *only*
changed the class identity would have scored exactly the same.

**Fix:** `_g3_probe` and `_g6_probe` now take the module the backend came from
and type-check against **that** module's classes; the six call sites pass the
mutant module. Re-running produced the distinct, defect-naming signatures in the
table above. Two related hardenings: `_patch()` asserts its textual anchor
occurs exactly once (a silently non-applying mutation would otherwise survive
vacuously and read as a false green), and `_record()` fails loudly with
`MUTANT SURVIVED` rather than skipping.

## 5. Non-regression

Baseline captured at `f163199` **before any edit**, and re-run after. Every
count matches the brief §6 declaration exactly.

| Suite | Baseline (pre-edit) | Declared §6 | After |
|---|---|---|---|
| D3.5 | 60/60 | 60/60 | 60/60 |
| D3.25 | 13/13 | 13/13 | 13/13 |
| D3 | 10/10 | 10/10 | 10/10 |
| D3.0 | 10/10 | 10/10 | 10/10 |
| D2 | 7/7 | 7/7 | 7/7 |
| D1.1 | 18/18 | 18/18 | 18/18 |
| D1.0 | 8/8 | 8/8 | 8/8 |
| D0 | 12/12 | 12/12 | 12/12 |
| Phase 4 | 63/63 | 63/63 | 63/63 |
| Phase 3 | 17/17 | 17/17 | 17/17 |
| **Total** | **228/228** | | **228/228** + D4 **8/8** |

fallback parity: code=[not re-checked this session], env=[not re-checked this
session] — `.127` is not booted (Zeus is running Proxmox / VM 101).

## 6. Scope confirmation

Every §2 must-not-modify file was verified **byte-identical to `f163199`** by
SHA-256, not by inspection:

```text
miner/range_miner_npz_writer.py          utils/canonical_arrays.py
utils/canonical_records.py               utils/run_finalizer.py
utils/prng_encoding.py                   window_optimizer_integration_final.py
persistent_worker_coordinator.py         zmq_sqlite_coordinator.py
convert_survivors_to_binary.py
tests/test_s172_phase5_d0.py             tests/test_s172_phase5_d1_engine.py
tests/test_s172_phase5_d1_workflow.py    tests/test_s172_phase5_d2_directional_uniqueness.py
tests/test_s172_phase5_d3_0_encoding_contract.py
tests/test_s172_phase5_d3_columnizer.py  tests/test_s172_phase5_d3_25_candidate_ingress.py
tests/test_s172_phase5_d3_5_finalizer.py
```

`prng_analysis.db` and WATCHER untouched. The single modification to
`tests/test_s172_phase4_coordinator.py` is the gate-22 coexistence registration
and its rationale comment — 19 insertions, 0 deletions, no assertion or gate
logic changed. Because D4 is a **pure add** (no existing production module
required a change), PWC / ZMQ / `pwc_protocol` remain untouched by this
deliverable and coexistence holds unchanged.

**No stop condition (§7) was hit.** The backend seam needed no edit to any
must-not-modify module; `assemble_trial`'s signature and return shape were
suitable as the interface contract exactly as they stand; no gate was weakened
to pass.

## 7. Next

Team Alpha review, then Team Beta. D5 (`process_sharded`) plugs into the seam
frozen here and should change no interface: it adds an implementation behind
`get_assembly_backend`, returns the same `BackendAssemblyResult`, and must
supply a §17-compliant `peak_rss_bytes` — the concurrent parent-plus-workers
aggregate, which `RUSAGE_CHILDREN` cannot provide (§5 [B3]). The authoritative
promotion benchmark is Phase 6, each backend in a fresh process.
