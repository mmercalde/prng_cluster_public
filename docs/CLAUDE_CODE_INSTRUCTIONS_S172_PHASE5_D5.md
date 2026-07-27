# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D5.md — REV1

**S172 RANGE-MINER — Phase 5, Deliverable D5: the `process_sharded` assembly
backend, plus the semantics-preserving D1.1 extraction it plugs into.**

**Audience:** Claude Code on VM 101 (`michael@192.168.3.177`), in
`~/distributed_prng_analysis`. You write and iterate; you do NOT commit, push, or
run WATCHER. When gates + non-regression are green, STOP and report for Team
Alpha review.

**Frozen against HEAD `3e8580a`** (Phase 5 D4 closed). Authority:
`docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §6.7.A / §6.7.B / §6.7.C / §17, and
the binding Team Beta D5 ruling (A + sampled RSS-sum, four findings locked).

---

## The one architectural sentence this brief exists to enforce

> **D5 parallelizes only spool-local validation. D1.1 remains the sole authority
> for validation semantics and global assembly semantics; workers produce
> ordered, lossless validated-spool artifacts, while the parent alone performs
> deterministic global merge, duplicate attribution, intersection, enrichment,
> and final assembly.**

Every design decision below is downstream of that sentence. If a proposed change
moves any global-assembly responsibility into a worker, it is wrong.

---

## 0. What is already built, and what D5 actually adds

**Read this before anything else.** Verified at `3e8580a`:

| capability | status |
|---|---|
| backend selector `serial_reference \| process_sharded`, fails closed | **DONE** — `miner/assembly_backends.py` (D4) |
| `serial_reference` backend + `AssemblyMeasurement` / `BackendAssemblyResult` frozen contract | **DONE** — D4 |
| `process_sharded` name declared, resolving it raises `NotImplementedError` naming D5 | **DONE** — D4 (`assembly_backends.py:275`) |
| serial spool read → 4 populations → 24-field records → `MinerTrialAssembly` | **DONE** — `assemble_trial` (D1.1, `range_miner_npz_writer.py:450`) |
| per-spool read/verify/parse/semantic validation | **DONE but PRIVATE + not separately callable** — `_read_and_validate_spool` (`:339`) |
| ordered global merge (4 maps, dup detection, intersection, enrichment) | **DONE but INLINE inside `assemble_trial`** (`:510–587`) |
| record→22-array columnizer, dedup, ordering, publication | **DONE** — `records_to_arrays` (D3), `finalize_run` (D3.5). **Not D5's business.** |

**Two facts drive the whole deliverable:**

1. The only genuinely parallel work in Phase 5 is the per-spool front end — byte
   read, size/SHA verify, JSON parse, per-survivor semantic validation. That is
   exactly `_read_and_validate_spool`, and it is exactly the hashing + parsing
   §6.7.A's benchmark note says dominates.
2. Everything after it (the four population maps, within-population duplicate
   detection, the bidirectional intersection, `build_mode_records` enrichment) is
   **globally coupled** and must stay serial in the parent to preserve semantics
   exactly (§6.7.C: "the parent process is the sole owner of global state").

Because the per-spool validator is private and the merge is inline, **D5 cannot
be purely additive.** It requires a tightly scoped, semantics-preserving
extraction inside `range_miner_npz_writer.py` first. Team Beta approved this
(ruling item 1, option A) on the D3.25-B precedent — the extraction changes no
behavior, and **D1.1's 18/18 staying green is the proof.**

**D5 therefore ships as TWO separate commits.** The separation is a hard
requirement: it makes the claim "the extraction changed nothing" independently
reviewable and bisectable.

---

## 1. Non-negotiable working rules

1. **Read live source before every claim.** All cites below verified at
   `3e8580a`. The seam is where this project's last four Team Beta rejections
   happened; extrapolation is the failure mode.
2. **Reuse, never reimplement.** After the extraction, both backends call the
   SAME extracted validator and the SAME extracted merge. Writing a second copy
   of validation, map construction, dedup, intersection, columnization, ordering
   or publication is a stop condition.
3. **Two commits, in order.** Commit 1 is extraction only — no multiprocessing,
   no artifact codec, no performance change, no "while I'm here" cleanup. Commit
   2 is the backend. Do not fold them.
4. **Each gate must FAIL on wrong behavior.** Independent, hand-transcribed
   oracles — never import expectations from the module under test.
5. **Every mutant must satisfy the four-part proof** (§7.C). A mutant that dies
   from a loader, type-identity, fixture or setup failure has proven nothing.
6. STOP at the gate. No commit/push/WATCHER from the sandbox.

---

## 2. Scope — exactly two commits

### Commit 1 — `range_miner_npz_writer.py` semantics-preserving extraction ONLY

Refactor `assemble_trial` into a thin serial wrapper over two extracted units,
with **zero behavioral change**. Nothing else in the module, and no other
module, changes.

### Commit 2 — the `process_sharded` backend

Implement `process_sharded` in `miner/assembly_backends.py`, add the CPU-only
worker (`miner/assembly_shard_worker.py`), the concurrent-tree RSS sampler, and
the 1/2/4/6/8-process benchmark harness. Workers call the extracted validator;
the parent calls the extracted merge. Serial and process-sharded paths converge
on one implementation of global semantics.

---

## 3. Commit 1 — the extraction

### 3.1 The two extracted units

Extract the existing logic **verbatim** into two module-level functions.

```python
@dataclass(frozen=True)
class ValidatedSpoolProjection:
    """The ordered, merge-relevant projection of ONE fully validated spool.

    LOCKED DEFINITION (Team Beta D5 ruling, finding F1):
    This is lossless with respect to *all state observable by canonical
    assembly* — NOT a lossless serialization of the source JSON payload. The
    merge consumes only seed and match_rate per survivor
    (`range_miner_npz_writer.py:537`); `strategy_id` and the ragged `skips` are
    validated inside the worker and then DISCARDED, exactly as the numeric
    encoding is validated-and-discarded. They never cross a process boundary
    because canonical assembly never observes them.

    Order and multiplicity are preserved exactly: no sort, no dedup, no
    normalization. Input survivor i is projection row i.
    """
    seeds: np.ndarray          # dtype int64, 1-D, survivor order preserved
    match_rates: np.ndarray    # dtype float64, 1-D, aligned to seeds
    survivor_count: int        # == seeds.shape[0] == match_rates.shape[0]


def read_and_validate_spool(
    run_id: str, manifest: Dict[str, Any],
) -> ValidatedSpoolProjection:
    """The extracted body of the current private `_read_and_validate_spool`,
    plus construction of the projection AFTER the whole spool passes.

    Full per-survivor validation (seed, match_rate, strategy_id, ragged skips)
    still runs. The full payload dict stays local and ephemeral inside this
    function; only the projection escapes.
    """


def merge_validated_spools(
    run_id: str,
    ordered_triples: Sequence[Tuple[Dict[str, Any], Dict[str, Any],
                                    ValidatedSpoolProjection]],
) -> MinerTrialAssembly:
    """The extracted inline merge block (`:510–587`), verbatim: 4 population
    maps with in-loop within-population duplicate detection + provenance, then
    `build_mode_records` intersection/enrichment, then MinerTrialAssembly.

    `ordered_triples` is `(manifest, meta, projection)` in the SAME
    deterministic key order the current loop computes (F2). The merge reads
    meta fields (direction, skip_mode, prng_type, workflow_phase) AND manifest
    fields (stripe_id, sub_index, attempt) per spool, so both must be carried
    alongside the projection.
    """
```

### 3.2 `assemble_trial` becomes a thin serial wrapper

After extraction, `assemble_trial(run_id, manifests)` keeps its exact current
signature and behavior and reduces to:

1. the metadata gauntlet, unchanged and in the current order — per-manifest
   `_validate_manifest_identity` (list order), cross-manifest 11-field
   consistency, phase-set completeness (`{1,2}` or `{1,2,3,4}`), encoding
   validation. **All of this precedes any spool read and must stay that way**;
2. compute the deterministic `order` exactly as now — sort key
   `(workflow_phase, stripe_id, sub_index, attempt, event_id)` (`:519`);
3. `projections = [read_and_validate_spool(run_id, manifests[i]) for i in order]`;
4. `return merge_validated_spools(run_id, [(manifests[i], metas[i], projections[k]) ...])`
   with triples in that same `order`.

No multiprocessing. No artifact files. No timing change beyond the existing
`timing={"assembly_s": ...}` field, which stays where it is.

### 3.3 Type-name reconciliation (verified)

The D5 ruling referenced `MinerResultManifest` and `SpoolMeta`. **Those types do
not exist.** Verified at `3e8580a`: the `ShardReadyManifest` published to Phase 5
is a plain `dict` (`range_miner_coordinator.py:2051`), and `assemble_trial`'s
input domain is `List[Dict[str, Any]]` (`:450`). **Do NOT invent a dataclass
manifest wrapper** — introducing one is itself an interface change and would break
the "extraction changed nothing" proof. `read_and_validate_spool` keeps
`manifest: Dict[str, Any]`. `ValidatedSpoolProjection` is the only new type in
Commit 1.

### 3.4 Proof obligation for Commit 1

**D1.1 must stay 18/18 green with no test edits.** That is the entire proof the
extraction is semantics-preserving. If any D1.1 test needs changing to pass, the
extraction changed behavior — STOP and report; do not adjust the test. Also
re-run the full non-regression set (§7.D) after the extraction, since D3/D3.5 and
the D4 backend all sit downstream of `assemble_trial`.

---

## 4. Commit 2 — the `process_sharded` backend

### 4.1 Worker (`miner/assembly_shard_worker.py`) — CPU-only

Each worker receives ONE small manifest (never payload) and:

1. calls `read_and_validate_spool(run_id, manifest)` — the SAME extracted
   validator the serial path uses, so validation semantics are identical by
   construction;
2. constructs the artifact **only after the entire spool passes** — a malformed
   survivor near the end of the JSON must prevent ANY successful artifact result
   (ruling boundary: no incremental artifacts);
3. writes the projection to a temporary file, computes its digest, and
   **atomically renames** into place only after: complete semantic validation →
   projection construction → artifact write → digest calculation → local
   read-back verification;
4. returns ONLY a compact result manifest: `{artifact_path, survivor_count,
   artifact_sha256, stripe_id, sub_index, attempt, workflow_phase, direction,
   skip_mode, prng_type}` — paths and counts, never arrays or payloads.

**Hard worker guards (each individually gated):**

- workers MUST NOT import `torch` or `cupy`, and MUST NOT initialize a GPU
  context. Assert this inside the worker process (e.g. `torch`/`cupy` absent
  from `sys.modules` after import of the worker module).
- the four §6.7.A prohibitions, each proven absent: no survivor dicts through
  `multiprocessing.Queue`; no 22 NumPy arrays through pickle; no giant parsed
  JSON object sent parent→child; no "24 processes because Zeus exposes 24
  threads" (pool size is an explicit parameter, never `os.cpu_count()` by
  default).

### 4.2 Artifact codec — lossless w.r.t. the projection, `allow_pickle=False`

The artifact stores exactly `ValidatedSpoolProjection`: `seeds` (int64) and
`match_rates` (float64) as two aligned 1-D arrays in survivor order, plus the
identity scalars as 0-d unicode/int arrays for cross-check. **`allow_pickle=False`;
no object arrays** — feasible precisely because ragged `skips` never crosses (F1).
Uncompressed (`np.savez`, not `savez_compressed`) per §6.7.A. The codec MUST
round-trip order and multiplicity byte-for-byte, including intra-spool duplicate
seeds (see F3 test). If uncompressed `.npz` ever proves unable to represent the
projection losslessly, **change the artifact format rather than weaken
equivalence** — never introduce a new columnization or dedup step to make it fit.

### 4.3 Parent merge — sole owner of global state

The parent:

1. runs the FULL metadata gauntlet (identity → consistency → phase completeness →
   encoding) **before dispatching any worker**, so exception precedence matches
   D1.1 exactly — a `PhaseIdentityError`/`AssemblyConsistencyError` must still
   pre-empt any `SpoolIdentityError` (verified: metadata precedes all spool reads
   in D1.1);
2. dispatches per-spool validation to the pool;
3. **consumes results in deterministic manifest `order`, NOT `as_completed()`**
   (ruling F/guardrail). A later-in-order malformed spool must never surface
   before an earlier-in-order one merely because its worker finished first;
4. reads each artifact back, cross-checks `artifact_sha256` and the identity
   scalars against the authoritative manifest/meta pair (defense-in-depth — the
   manifest/meta remain the source of truth, artifact fields are cross-checks
   only), and reconstructs the `ValidatedSpoolProjection`;
5. calls `merge_validated_spools(run_id, ordered_triples)` — the SAME extracted
   merge the serial path calls. Within-population duplicate detection, provenance
   attribution, `prng_type_by_mode` last-writer-wins in loop order (F4),
   intersection and enrichment all happen here, once, serially.

Workers MUST NOT sort, dedup, normalize, intersect, or build maps. The parent
must not use concurrent Python dicts for global state — the extracted merge's
existing model is authoritative.

### 4.4 Multiprocessing model

- **`spawn` is the canonical context.** Each worker is a fresh interpreter with
  no inherited GPU-library state.
- **`forkserver` permitted only if a test proves its server starts without
  inherited GPU state** (no `torch`/`cupy` in the server's `sys.modules`).
- **Never silently fall back to `fork`.** An unavailable context is a hard error.
- Persistent, bounded pool. Clean up ALL temporary artifacts after success and
  after every failure path (no leaked spool artifacts on any exception).

### 4.5 Benchmark sweep

Sweep **1, 2, 4, 6, 8** assembly processes (§6.7.A). Do not assume 12/24. Record
wall time and the canonical peak_rss (§5) for each, on both a high-survivor and a
low-survivor trial. This produces the numbers §17's promotion rule consumes — but
D5 does NOT itself promote the backend; §17 promotion is Phase 6's isolated
benchmark. D5 proves equivalence and produces measurements.

---

## 5. Measurement — canonical `peak_rss` (Team Beta finding, sampled RSS-sum)

Record **sampled concurrent-tree RSS-sum** as the canonical `peak_rss`:

```json
{
  "peak_rss": 123456789,
  "peak_rss_definition": "sampled_sum_of_parent_and_recursive_children_rss",
  "sample_interval_ms": 25
}
```

Implementation rules (all binding):

- sampler walks `parent + psutil children(recursive=True)`, **PID-deduplicated**;
- start sampling **before** worker creation; continue through artifact loading
  and the parent merge; stop **only after** workers have joined AND merge
  processing completed;
- tolerate `NoSuchProcess`, `ZombieProcess`, `AccessDenied` (process-exit races);
- `time.monotonic()`, fixed **25 ms** interval for the benchmark harness;
- document in the evidence that RSS-sum can double-count shared pages and is
  therefore a **conservative process-tree footprint, not exact physical RAM**.

PSS may be collected as **optional Linux-only telemetry**
(`"peak_pss_optional": ...`) but MUST NOT replace `peak_rss`, participate in any
pass/fail, or be required in CI.

`RUSAGE_CHILDREN` is ruled out: it reports the max of one reaped child, never the
concurrent sum (this is what the D5 RSS mutant proves — §7.C).

---

## 6. Where D5 stops

A backend produces a `MinerTrialAssembly` and STOPS. It does not dedup across
seeds, order winners, merge the 22 arrays, apply the contract wall, or publish —
that is `finalize_run` (D3.5), the CALLER's next step, never the backend's.
`assembly_backends.py` must not import `finalize_run`. `binary_npz_path` /
`all_npz_path` stay `None` through every backend (D3.5 Ruling E).

---

## 7. Gate — `tests/test_s172_phase5_d5_process_sharded.py`

### 7.A Equivalence gates (process_sharded output ≡ serial_reference output)

Across a matrix — constant-only `{1,2}`, both-modes `{1,2,3,4}`, high-survivor,
and empty-survivor spools — assert `process_sharded.assemble(...)` returns a
`MinerTrialAssembly` field-for-field identical to `serial_reference.assemble(...)`:
`run_id`; `bidirectional_constant`/`_variable`; all four maps;
`canonical_records_constant`/`_variable` element-wise; `directional_counts`;
both NPZ path fields `None`. (`timing` compared as D4's G3: both finite, `> 0`,
no backend-specific key.)

- **G-DUP-CROSS:** the same seed in one population across two spools raises an
  identical `DirectionalDuplicateError` with identical first-vs-dup attribution
  (stripe/sub/attempt) under both backends.
- **G-DUP-INTRA (F3):** the same seed twice **inside one spool** raises the
  identical `DirectionalDuplicateError` under both backends — proves worker
  projection preserved order and multiplicity with zero dedup.
- **G-MALFORMED-DUAL:** two malformed spools at different `order` positions —
  the observed error is the earlier-in-`order` one under both backends,
  regardless of which worker finishes first (proves no `as_completed()`).
- **G-PRECEDENCE:** a trial with both a metadata defect and a spool defect raises
  the metadata exception (`PhaseIdentityError`/`AssemblyConsistencyError`) under
  process_sharded, never the `SpoolIdentityError` — parent ran the full gauntlet
  before dispatch.

### 7.B Structural gates

- **G-SPAWN:** the pool uses `spawn`; assert the start method actually used.
- **G-NO-GPU:** worker process has no `torch`/`cupy` in `sys.modules`.
- **G-NO-PAYLOAD-IPC:** each of the four §6.7.A prohibited IPC shapes proven
  absent (AST + runtime: worker return value contains no ndarray, no survivor
  dict, no parsed-JSON payload; pool size is an explicit arg).
- **G-CODEC:** artifact round-trips seeds/match_rates with `allow_pickle=False`,
  no object arrays, order + multiplicity + intra-spool dups intact.
- **G-ATOMIC:** an injected failure after temp-write but before rename leaves NO
  artifact at the final path and NO leaked temp file.
- **G-CLEANUP:** after both a successful run and a failing run, zero temporary
  artifacts remain.
- **G-FINALIZER (reuse D3.5 harness):** drive process_sharded →
  `canonical_records_constant`+`_variable` → `finalize_run` → a published
  generation whose 22 arrays match serial_reference's for the same input. Do NOT
  re-derive D3.5-B's S1–S9 sidecar assertions.
- **G-RSS:** `peak_rss` sampler returns a positive integer; evidence carries
  `peak_rss_definition` and `sample_interval_ms`; PSS, if present, is not gating.

### 7.C Mutation proof (each mutant satisfies the four-part rule)

Every mutant must prove: **applies exactly once; executes the mutated path;
reaches the assertion credited with killing it; and fails from the injected
defect — not a loader, type-identity, fixture or setup failure.** Kill each:

1. **extraction non-equivalence** — perturb `merge_validated_spools` (e.g. use
   completion order instead of deterministic `order`) → dup attribution / record
   order diverges from serial; D1.1 18/18 also reds.
2. **worker skips a validation branch** (e.g. drops the `skips` type check) → a
   malformed spool that serial rejects is accepted by process_sharded → G-DUP /
   validation gate reds.
3. **parent consumes `as_completed()`** → G-MALFORMED-DUAL reds.
4. **worker sorts/dedups the projection** → G-DUP-INTRA reds.
5. **parent uses a concurrent dict for dedup** → tie/order divergence vs serial.
6. **codec uses `savez_compressed` / object array** → G-CODEC reds (and asserts
   the ban, not just a size delta).
7. **`fork` substituted for `spawn`** → G-SPAWN / G-NO-GPU reds.
8. **`RUSAGE_CHILDREN` substituted for the concurrent sampler** → construct two
   overlapping children each holding a substantial allocation across several
   25 ms samples; the mutant under-reports (single-child max < concurrent sum),
   G-RSS's tree-sum assertion reds. This is the explicit ruling construction.
9. **metadata gauntlet moved after dispatch** → G-PRECEDENCE reds.

Report each red signature and its attribution.

### 7.D Blocking non-regression

Capture baseline green at `3e8580a` **before any edit**: D4 **8/8** gate checks
(G1–G8; G8 bundles 9 mutants — run
`PYTHONPATH=. python3 tests/test_s172_phase5_d4_serial_backend.py`), D3.5 60/60,
D3.25 13/13, D3 10/10, D3.0 10/10, D2 7/7, D1.1 18/18, D1.0 8/8, D0 12/12, Phase
4 63/63, Phase 3 17/17. After Commit 1, **D1.1 18/18 and the
entire downstream set must still be green** — that is the extraction's proof.
After Commit 2, all of the above plus the new D5 gate.

---

## 8. Stop conditions

- the extraction cannot be made byte-equivalent (D1.1 reds) — report; do not edit
  the tests to fit;
- `ValidatedSpoolProjection` proves unable to carry all state the merge observes
  — report (it should not: the merge observes only seed + match_rate);
- you find yourself writing assembly, columnization, dedup, ordering, intersection
  or publication logic in a worker or in Commit 2 — that code exists in the
  extracted merge;
- any gate passes only by weakening it;
- a mutant dies from setup/loader/type-identity rather than the injected defect —
  the mutant is invalid, not the gate.

## 9. Report

For **each commit separately**: diff + status, full command/output evidence, the
pre-edit baseline, and — for Commit 1 — explicit confirmation that D1.1 18/18 and
all downstream suites stayed green with no test edits. For Commit 2: per-mutant
red signatures with four-part attribution, the 1/2/4/6/8 benchmark table with
canonical `peak_rss`, and confirmation no D0–D4 production module or test was
modified beyond the Commit-1 extraction. Then STOP for Team Alpha review.
