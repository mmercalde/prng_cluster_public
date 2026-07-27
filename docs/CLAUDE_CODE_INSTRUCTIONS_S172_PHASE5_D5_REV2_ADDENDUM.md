# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D5 — REV2 ADDENDUM (Team Beta ruled B)

**Supersedes REV1 §3.2 and §4.3.** Everything else in REV1 stands. Team Beta
approved **Option B**: D5 must preserve the original deterministic read/merge
exception precedence. Workers may read spool-local data concurrently, but
canonical spool-read errors are returned as typed data and **replayed by the
parent in deterministic manifest order**. This is a **rework** of the Commit 1
and Commit 2 you already built — not a fresh start.

Base unchanged: HEAD `3e8580a`. D1.1 stays 18/18 with zero test edits.

---

## 0. Why the current build must change

Your current `assemble_trial` does read-all-then-merge
(`projections = [read_and_validate_spool(...) for i in order]` then merge). That
reorders observable exceptions: earlier-order duplicate + later-order malformed
raises `DirectionalDuplicateError` in the original but `SpoolIdentityError` in
your build. Beta ruled that divergence out. The canonical state machine both
backends must present is:

```
for position in deterministic_order:
    outcome = <projection or captured-read-error at position>
    if outcome.is_read_error:
        raise_canonical_read_error(outcome)      # re-raise, in order
    merge_insert(outcome.projection)             # may raise DirectionalDuplicateError
```

Parallel completion order becomes irrelevant; precedence is driven only by
`order`.

---

## 1. New canonical types (define in `range_miner_npz_writer.py`, Commit 1)

These are canonical assembly-contract types, not process machinery, so they live
with the merge and are frozen after Commit 1. Commit 2 only *produces* them.

```python
@dataclass(frozen=True)
class CapturedSpoolReadError:
    """A canonical spool-read defect captured as typed data — NEVER a pickled
    exception instance. Round-trips class identity + args + message +
    attribution; traceback frames are explicitly NOT preserved (REV2 §3)."""
    error_code: str                       # canonical class name, allowlisted
    message: str                          # exact rendered message
    args: tuple                           # exc.args, scalar-only
    attributes: Mapping[str, Any]         # custom identity/provenance fields

SpoolReadOutcome = Union[ValidatedSpoolProjection, CapturedSpoolReadError]

def raise_captured_spool_error(descriptor: CapturedSpoolReadError) -> NoReturn:
    """Reconstruct and raise the ORIGINAL canonical exception class from a
    descriptor, preserving class, .args, rendered message, and custom
    attribution. The ONLY place a captured read error becomes a live exception
    in the parent."""
```

**Allowlist (REV2 §1).** Only canonical spool-read defects may become a
`CapturedSpoolReadError`: `SpoolIdentityError` and the enumerated D1.1 semantic
payload-validation exceptions in that same hierarchy. **Never descriptorize**
`MemoryError`, `KeyboardInterrupt`, `SystemExit`, process-pool failures,
artifact-write failures, or unexpected programming errors — those are backend
failures (§4 below), not producer defects.

---

## 2. Commit 1 rework — extraction + serial preservation (no process code)

`merge_validated_spools` consumes an **ordered iterable of per-position items**
and runs the canonical replay loop: pull item in `order`; if it is a
`CapturedSpoolReadError`, `raise_captured_spool_error(it)`; else `merge_insert`
the projection (which may raise `DirectionalDuplicateError`).

`assemble_trial` (serial) passes a **lazy generator**, not a pre-built list:

```python
def _serial_outcomes(run_id, manifests, metas, order):
    for i in order:
        # reads ONE spool, yields its projection; a bad read raises HERE,
        # at this position, before the next is pulled — interleaved exactly
        # like the pre-D5 original. Serial NEVER produces a descriptor; it
        # raises the original exception object with its original traceback.
        yield (manifests[i], metas[i], read_and_validate_spool(run_id, manifests[i]))

def assemble_trial(run_id, manifests):
    started = time.perf_counter()
    metas, ctx, order = prepare_trial_assembly(run_id, manifests)
    return merge_validated_spools(
        run_id, ctx, _serial_outcomes(run_id, manifests, metas, order), started)
```

Consequences you must hold:

- **Serial is byte-identical to pre-D5**, including precedence and the original
  raised exception object/traceback, because serial reads lazily and never
  round-trips through a descriptor. This is what keeps D1.1 18/18 with zero
  edits, now truly (not over an uncovered corner).
- The merge loop's `CapturedSpoolReadError` branch exists in Commit 1 but is
  exercised only by Commit 2's parallel path and by the new corner gates. That
  is fine — the types are canonical, not process-specific.
- **The merge is frozen after Commit 1.** Commit 2 must add zero lines to
  `range_miner_npz_writer.py`. If you find yourself editing the merge in Commit
  2, stop — the outcome-aware loop belongs in Commit 1.

---

## 3. Commit 2 rework — process backend produces outcomes, parent replays

- **Metadata precedence before dispatch (REV2 §4).** The parent completes the
  full gauntlet (`prepare_trial_assembly`: identity → consistency → phase →
  encoding) **before submitting any worker**. Even if you launch workers early
  for throughput, no worker outcome may be *observed* until the gauntlet passes.
  Cleanest: gauntlet first, then dispatch.
- **Workers return `SpoolReadOutcome`.** On a canonical, allowlisted read defect,
  the worker returns a `CapturedSpoolReadError` descriptor (built from the caught
  canonical exception — class name, args, message, custom attribution). On
  success, the projection artifact + result manifest as in REV1 §4.1.
- **Fill concurrently; replay serially (REV2 §6).** `as_completed()` may populate
  indexed slots: `outcomes[position] = future.result()`. `as_completed()` MUST
  NOT raise a canonical error or merge a projection. The ONLY place canonical
  outcomes surface is the deterministic replay:
  `for position in order: replay(outcomes[position])` — feeding the SAME
  `merge_validated_spools` from Commit 1.
- **Backend failures are distinct (REV2 §5).** A crashed worker, broken pool,
  unreadable/mismatched artifact, digest failure, or timeout raises a distinct
  `ProcessShardedAssemblyError` (backend-level) — it must NEVER masquerade as a
  `CapturedSpoolReadError`. `SpoolReadOutcome` is exactly the union of projection
  or captured *canonical producer* defect; infrastructure failure is neither.
- **Cleanup after an early replay failure (REV2 §7).** When replay raises an
  earlier duplicate or read error mid-way: cancel pending futures where possible;
  terminate/join workers cleanly; remove every temporary projection artifact;
  stop the RSS sampler; **retain the original canonical exception as primary**.
  Cleanup failures may attach as notes/chained diagnostics but must NOT replace
  the primary exception or alter precedence.

---

## 4. Equivalence language (REV2 §3 — replaces REV1's "byte-identical")

The contract for exceptions is: **equivalent in exception class, `.args`,
rendered message, custom attribution fields, and deterministic precedence.**
Traceback frames and backend-internal exception chaining are explicitly NOT
contractual. For valid inputs, REV1's field-for-field and 22-array equivalence is
unchanged.

---

## 5. Required gate matrix (REV2 — full matrix, replaces REV1's two corner gates)

`tests/test_s172_phase5_d5_process_sharded.py` must include all six rows. Run each
against three targets: the preserved pre-D5 reference (or a frozen oracle
fixture), `serial_reference`, and `process_sharded`.

| earlier position | later position | required result |
|---|---|---|
| duplicate | malformed | identical duplicate exception + attribution |
| malformed | duplicate | identical malformed exception |
| intra-spool duplicate | malformed | identical duplicate exception + attribution |
| malformed A | malformed B | error from A |
| valid | duplicate | identical duplicate attribution |
| valid | malformed | identical malformed exception |

Each assertion compares **more than `str(exc)`**:

```python
assert type(serial_exc) is type(sharded_exc)
assert serial_exc.args == sharded_exc.args
assert serial_exc.custom_fields == sharded_exc.custom_fields   # attribution
```

Keep REV1's equivalence, structural, codec, atomic, cleanup, finalizer, RSS, and
G-D4-INTACT gates. Update the mutation set so the precedence mutants target the
replay loop (e.g. `as_completed()` used to raise → later-position error surfaces;
descriptor round-trip drops a custom attribution field → attribution assert reds;
backend failure descriptorized as `CapturedSpoolReadError` → §5 separation red).
Every mutant still satisfies the four-part kill rule.

---

## 6. Proof obligations

- After the Commit 1 rework: D1.1 **18/18, zero test edits**, and now a true
  no-op — serial interleaves and raises the original exception objects. The full
  downstream/non-regression set green.
- After Commit 2: the six-row matrix green across all three targets; the merge in
  `range_miner_npz_writer.py` unchanged from Commit 1 (Commit 2 adds zero lines
  to that file); backend failures provably distinct from producer defects; all
  mutants killed with four-part attribution.
- Report per commit as REV1 §9, and explicitly confirm serial raises the
  ORIGINAL exception object (not a reconstructed one) on the corner cases, while
  process_sharded raises a reconstructed-but-equivalent one.

Then STOP for Team Alpha review.
