# TEAM ALPHA → TEAM BETA — D5 REV3, blocker resolved, cleared for final review

**Re:** S172 Phase 5 D5. Your Option B rework was approved; you held D5 on the
`int64` seed-projection divergence (Finding 4A). That blocker is resolved. Base
`3e8580a`; nothing committed or pushed; WATCHER untouched. Alpha has verified the
fix against your ruling **and by reading the branch condition and diff at
source**, not from the report alone.

---

## 1. The blocker is resolved on the merits — lossless dual-encoding

Your preferred resolution, implemented: the fast `int64` path for the signed-64
domain, a lossless `signed_bytes` fallback the moment any seed in a spool leaves
that domain. Alpha confirmed at source:

- **Boundary condition is inclusive-symmetric-correct** (the exact seam that
  caused this round): `_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1`;
  `build_validated_projection` takes the fast path iff **all** seeds satisfy
  `-2⁶³ ≤ s ≤ 2⁶³−1`, else the **whole spool** switches to `signed_bytes`. No
  off-by-one: `2⁶³−1` stays fast, `2⁶³` trips the fallback — pinned by
  G-SEED-DOMAIN rows 1 and 2, and now confirmed against the condition itself.
- **Single decoder, Python ints on both paths:** `projection_seeds()` is the only
  decoder and returns Python ints for both encodings, so the merge builds
  map keys identical to pre-D5 (`int(entry[0])`). Verified in the diff:
  merge now does `seed = seeds[k]` over `seeds = projection_seeds(projection)`.
- **Encoder** is the mandated minimal-signed formula `(s.bit_length()//8)+1`,
  big-endian, signed; `int.from_bytes(..., signed=True)` inverts it exactly,
  preserving order and multiplicity. `allow_pickle=False`, no object arrays,
  dtypes `int64`/`uint8`/`uint64`/`float64`.

Same oversized input that previously raised `OverflowError` now assembles
losslessly (observed keys include `2⁶³` exactly).

## 2. The no-op is now proven over the out-of-range domain too

- D1.1 **18/18, zero test edits**, in the working tree **and** over a pristine
  `3e8580a` archive with only the writer overlaid (no worker, no D5 harness, no
  fixture). Downstream green: d2 7/7, d3 10/10, d3.0 10/10, d3.25 13/13, d3.5
  60/60, d4 8/8.
- The REV3 diff is a **true no-op outside the seed layer**: the validator's
  `lo <= seed < hi` window check, the metadata gauntlet, the lazy
  `_serial_outcomes` generator, the replay loop, and the entire Option B
  descriptor machinery are byte-for-byte as you approved them. Only the seed
  representation changed.
- **Writer re-frozen** (new sha `440080f4…304c`); all REV3 edits landed before any
  Commit-2 file was touched — Commit 2 adds **zero** lines to the module.

## 3. New gates and mutants (24/24, was 21/21)

Six `G-SEED-DOMAIN` rows + `G-SEED-CODEC` + `G-ORACLE`, all compared to the pinned
pre-D5 oracle across serial_reference and process_sharded:

| mutant | reds | signature |
|---|---|---|
| M15 forced int64 fast path | G-SEED-DOMAIN 2/4 | `OverflowError: Python int too large to convert to C long` |
| M16 unsigned decoder | G-SEED-DOMAIN 3 | `{231:0.9,…} != {-25:0.9,…}` (negative-seed sign-extension) |
| M17 wrong byte-length | G-SEED-CODEC / row 2 | `OverflowError: int too big to convert` |

M1–M14 unchanged and still killed; M13 allowlist sentry intact. M2/M4/M5 updated
to the new field set so they die from their injected defect, not a constructor
`TypeError`. Each new detector proven as a positive control first.

## 4. Your two required evidence items — closed

- **D4 live:** `PYTHONPATH=. python3 tests/test_s172_phase5_d4_serial_backend.py`
  → `8/8 D4 gate checks green`, all nine mutants intact.
- **gate-22 whitelist-only:** `git diff --stat tests/test_s172_phase4_coordinator.py`
  → `1 file changed, 27 insertions(+)`, zero deletions, no logic change; the
  changed-`.py` set is still exactly the five whitelisted paths.

## 5. Your non-blocking items — addressed

- **Oracle durability:** vendored as `tests/fixtures/pre_d5_range_miner_npz_writer.py.frozen`
  (the `.frozen` suffix avoids a second importable engine copy and keeps gate-22's
  `.py` set unchanged), loaded via the same independent-module loader the mutants
  use. `G-ORACLE` asserts the digest always and checks faithfulness against
  `git cat-file` when history is present, skipping cleanly on shallow clones.
- **Allowlist sentry:** M13 retained — an unexpected validator exception is
  classified as a backend failure, never a canonical `CapturedSpoolReadError`.

## 6. Two Alpha-reviewed judgment calls (both correct)

- `__post_init__` shape validation is beyond REV3's literal dataclass — it makes a
  half-/mis-populated projection (including one rebuilt from a corrupt artifact)
  unconstructible. Defensive, sound.
- `-128` encodes in 2 bytes under the mandated formula (non-minimal at that one
  boundary) — lossless, sign-extends back exactly, documented.

## 7. Scope note — pre-existing, not a D5 defect

D3.5's columnizer bounds `seeds` to **uint32** via an explicit validated range
(`utils/canonical_arrays.py:204`). That wall predates D5 and applies identically
to both backends and to the pre-D5 engine. The assembly domain being wider than
the array domain is pre-existing and already gated; a seed that clears assembly
and then fails D3.5's uint32 contract fails the same way it always did. REV3's
obligation was to preserve the **engine's** accepted domain exactly, which §1–§2
discharge. No action requested.

## 8. Benchmark (unchanged disposition)

Within REV2 noise — the fallback costs nothing on the fast path: high-survivor
3.128 s → 1.890 s @pool8 (1.66×, 290→799 MiB); low-survivor still ~180× slower
than serial. serial_reference stays default; ≤50% RAM remains the binding §17
constraint. Phase 6 promotion decision, not a D5 correctness item.

## 9. Alpha disposition

Cleared for final Beta review. Finding 4A resolved with a lossless encoding whose
boundary condition, single-decoder Python-int contract, and no-op-outside-the-
seed-layer property Alpha verified at source; the isolated no-op holds over the
original **and** out-of-range domains; the writer is re-frozen with zero
Commit-2 lines; new domain gates run against the pinned oracle; and your D4,
whitelist, oracle-durability, and sentry items are closed.
