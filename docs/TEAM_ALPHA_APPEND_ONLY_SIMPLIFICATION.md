# TEAM ALPHA → TEAM BETA — append-only dated snapshots: a simplification of P0/P1

**Re:** Beta's dataset-lifecycle rulings. Alpha accepts all five. This proposes a
**producer-side change that makes Rulings 1 and 2 substantially simpler to implement**,
and requests a ruling before Phase 6-P0 is built to the more complex specification.

**Status of Ruling 4:** `daily3scraper.service` is now **disabled and inactive**
(`systemctl disable --now`, enable symlink removed). The unit file is retained on disk
as Beta directed. The ENOENT restart loop has stopped, and no file appearing at the
target path can auto-execute.

---

## 1. The decision: the rewrite mode is being eliminated

Beta's rulings correctly assume a mutable dataset, because that is what exists today.
The current scraper has two modes — append and rewrite — and Beta's freeze protocol,
lineage wall, and merge-refusal rules all exist to survive the **rewrite** case.

**Michael has ruled that rewrite will not exist.** Historical context: rewrite mode was
added during initial testing when the dataset was discovered to be incomplete. That was
a one-time bootstrap need, not an operating requirement. Claude Code will recode the
scraper, and **the new implementation will have no rewrite option.**

The chosen publication model is **dated immutable files** rather than one growing
`daily3.json`.

## 2. Why this collapses most of the complexity

Beta's Ruling 1 §"Digest-only freezing is insufficient" identified the real hazard:
hashing a mutable pathname does not freeze anything, and hashing mid-write can yield a
byte sequence that never existed as a stable version. Both hazards are **properties of
a mutable path**, not of the pipeline.

With dated immutable publication:

| Beta requirement | with a mutable path | with dated immutable files |
|---|---|---|
| materialize a run-scoped immutable snapshot | copy the bytes to a content-addressed location | **the published file already is one** — nothing to materialize |
| freeze protocol steps 1–3 | select, materialize, hash | **read the current pointer, record its hash** |
| workers must read the frozen path, not the mutable source | requires a distinct snapshot path | **the dated path is inherently stable** |
| mid-run scrape switching a worker to newer data | must be actively prevented | **cannot occur** — a published file is never modified |
| torn read during append/rewrite | needs a lock or atomic-publish authority | **cannot occur** — publication is write-new-then-point |

Beta explicitly listed "immutable version files with an atomic current pointer" as an
acceptable freeze authority. Alpha is proposing to adopt exactly that, at the producer,
so the consumer side never needs the harder machinery.

`source_location` and `frozen_snapshot_path` remain distinct fields as Beta required —
they simply become "the pointer" and "the dated file it resolves to."

## 3. Why this also simplifies the Phase 6-P1 lineage wall

Beta's merge-refusal rule exists because a rewrite can change a **past** draw, which
invalidates scores computed against it. Alpha agrees completely: under rewrite, "new
dataset equals old dataset plus one draw" is not a safe invariant, and silently merging
generations scored against different data produces an artifact whose rows do not share
one input meaning.

**Under append-only that invariant becomes true — and, critically, checkable.**

> Before accepting a new dataset version, verify the previous version is a **byte-exact
> prefix** of the new one.

If it is, every score computed against the old version remains valid, because nothing
those scores were measured against has changed. Merging a prior generation is then
safe, and the lineage wall becomes a **prefix check that passes trivially in normal
operation**.

**Alpha requests that the wall be retained, not dropped.** Append-only must be
*enforced*, not trusted:

- the guarantee is only as good as the producer's discipline, and a future operator
  under time pressure could reintroduce an in-place edit;
- a partial or interrupted publication could produce a non-prefix file;
- a source-side correction upstream could silently alter history.

A failing prefix check must be a **hard stop** requiring a deliberate decision — new
accumulator lineage, or explicit re-scoring of retained rows — never a silent merge.
This preserves Beta's fail-closed intent at a fraction of the implementation cost, and
it makes a rewrite **visible** rather than impossible: corrections remain possible, but
only as a recorded, governed operation.

## 4. Two design questions Alpha is not deciding unilaterally

Both fall inside Beta's Phase 6-P2 scope and should be ruled on before Claude Code
writes the scraper:

**(a) Pointer mechanism.** Alpha proposes an atomic `current` pointer (symlink or a
small pointer file) updated by write-temp-then-rename, so a reader either sees the old
version or the new one, never a partial state. Note the D3.5 precedent: finalizer-owned
compatibility symlinks already exist and fail closed when a regular file appears in
their place — a symlink pointer must not collide with that pattern.

**(b) Correction handling.** This is the case that created rewrite mode. Under
append-only, correcting a bad historical draw cannot be an in-place edit. It must
become an explicit operation — a new dataset lineage, or an appended correction record
the consumer understands. **Alpha recommends Beta rule on this now**, because leaving
it undefined is precisely how rewrite mode reappears the next time a bad draw is found
at an inconvenient moment.

## 5. Rulings requested

1. **Approve append-only dated immutable publication** as the producer-side contract,
   and confirm that Phase 6-P0's freeze step may reduce to *read pointer → record
   hash → verify per node → fail before dispatch*.
2. **Confirm the lineage wall is retained** as a byte-exact prefix check (Alpha
   recommends retaining it; enforcement, not trust).
3. **Rule on the pointer mechanism** (§4a).
4. **Rule on correction handling** (§4b) — how a bad historical draw is corrected once
   rewrite no longer exists.
5. Confirm this simplification is adopted **before** Phase 6-P0 is specified, so P0 is
   not built to the mutable-path specification and then reworked.

## 6. Unchanged and still accepted

Beta's Rulings 2, 3, 4 and 5 stand as issued: Phase 6-P1 owns provenance binding and
the finalizer lineage invariant; P1 is mandatory before accepted multi-rig Phase 6
certification and before Phase 7; `run_daily3scraper.py` is a governed Phase 6-P2
deliverable with the twelve acceptance requirements, including the
timer-versus-daemon declaration (Alpha notes `Restart=always` is wrong for a one-shot
daily scrape — `Type=oneshot` plus a `.timer` is the correct shape unless the component
is genuinely a long-running daemon); and VIR-6 is adopted, with the standard and brief
block to be updated accordingly.

The existing D6 release-grade artifact remains immutable and authoritative, to be
described as the **pre-dataset-provenance authoritative generation**. Its sidecar will
not be rewritten.
