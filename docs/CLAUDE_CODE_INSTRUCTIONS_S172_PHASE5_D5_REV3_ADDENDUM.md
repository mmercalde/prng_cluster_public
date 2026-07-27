# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D5 — REV3 ADDENDUM (Beta hold: int64 divergence)

**Supersedes the seed-representation parts of REV1 §4.2 and the
`ValidatedSpoolProjection` definition in REV2 §1.** Everything else in REV1/REV2
stands, including the entire Option B exception-precedence machinery, which Beta
**approved**. This addendum resolves the one blocker: the `int64` seed projection
changes accepted-input behavior and therefore invalidates the unconditional
"Commit 1 is a no-op" claim.

Base unchanged: `3e8580a`. This is a **targeted rework of the projection layer**
(Commit 1) plus its codec (Commit 2). D1.1 stays 18/18 with zero test edits.

---

## 0. The blocker, precisely

Pre-D5 stored seeds as arbitrary Python ints in the seed→match_rate maps; the
validator bounds seeds to `[seed_start, seed_start+seed_count)` with **no
signed-64 bound and no non-negativity check on `seed_start`**. The reworked
`read_and_validate_spool` converts seeds to `np.int64`, so a seed ≥ 2⁶³ (or
< −2⁶³) now raises `OverflowError` where pre-D5 accepted it. This is a valid-input
divergence. "Unreachable for java_lcg" is not sufficient — the engine is
base-parameterized, not contractually one-family, and this is the same
test-domain blind spot that previously hid the precedence divergence.

**Fix: lossless dual-encoding projection.** Fast `int64` for the common
(signed-64-representable) case; a lossless `signed_bytes` fallback the moment any
seed in a spool falls outside signed-64. No object arrays; `allow_pickle=False`
throughout.

---

## 1. New `ValidatedSpoolProjection` (Commit 1)

```python
@dataclass(frozen=True)
class ValidatedSpoolProjection:
    seed_encoding: Literal["int64", "signed_bytes"]
    seeds_i64: Optional[np.ndarray]      # int64, len==survivor_count, else None
    seed_bytes: Optional[np.ndarray]     # uint8, concatenated, else None
    seed_offsets: Optional[np.ndarray]   # uint64, len==survivor_count+1, else None
    match_rates: np.ndarray              # float64, aligned to survivor order
    survivor_count: int
```

Exactly one seed representation is populated; the other two seed fields are
`None`. Order and multiplicity are preserved in every case — row *i* is survivor
*i*.

## 2. Construction in `read_and_validate_spool` (Commit 1)

After the full per-survivor validation passes (unchanged), decide the encoding for
the whole projection:

- Scan the validated Python-int seeds. If **all** satisfy `-2⁶³ ≤ seed ≤ 2⁶³−1`
  → `seed_encoding="int64"`, `seeds_i64 = np.array(seeds, dtype=np.int64)`.
- If **any** seed is outside that range → `seed_encoding="signed_bytes"` for the
  **entire** projection (small seeds in that spool are encoded too, minimally).

Signed-bytes encoding (get the edge cases right — this is a classic off-by-one):

```python
def _encode_seed(s: int) -> bytes:
    nbytes = (s.bit_length() // 8) + 1        # minimal signed length; correct at
    return s.to_bytes(nbytes, "big", signed=True)   # ±2^(8k-1) boundaries
```

Concatenate per-seed bytes into one `uint8` array; `seed_offsets` is a
`uint64` array of length `survivor_count+1`, `offsets[0]=0`,
`offsets[i+1]=offsets[i]+len_i`. Row *i* bytes are
`seed_bytes[offsets[i]:offsets[i+1]]`.

The validator's own `lo <= seed < hi` window check already uses Python ints and
must stay exactly as-is — it never overflows; only the projection did.

## 3. Merge reconstruction (Commit 1)

`merge_validated_spools` must reconstruct each seed as a **Python int** before
building the maps, so keys match pre-D5 exactly (pre-D5 used `int(entry[0])`):

- `int64` path: `seed = int(proj.seeds_i64[k])`.
- `signed_bytes` path:
  `seed = int.from_bytes(proj.seed_bytes[proj.seed_offsets[k]:proj.seed_offsets[k+1]], "big", signed=True)`.

Duplicate detection, provenance/attribution, intersection and enrichment are
otherwise unchanged. Confirm map keys are Python ints, not `np.int64`, even on
the fast path (equal-valued but the pre-D5 contract is Python int).

## 4. Codec round-trip (Commit 2)

The artifact stores whichever encoding the projection carries: the scalar
`seed_encoding` tag, plus either `seeds_i64` **or** (`seed_bytes` + `seed_offsets`),
plus `match_rates`. All dtypes are `int64` / `uint8` / `uint64` / `float64` —
`allow_pickle=False`, no object arrays, uncompressed. Round-trip must preserve
order, multiplicity, and the exact reconstructed Python-int values, including
mixed small-and-oversized within one signed_bytes projection.

## 5. Required new gates (all compared to the pinned pre-D5 oracle)

Add to `tests/test_s172_phase5_d5_process_sharded.py`, each asserting outputs
**and** exceptions match the pre-D5 oracle across serial_reference and
process_sharded:

1. seed = 2⁶³−1 (max int64; stays on the fast path).
2. seed = 2⁶³ (triggers fallback; pre-D5 accepted it).
3. negative seed inside a negative manifest range (`seed_start < 0`).
4. seed larger than 64 bits (e.g. 2⁷⁰).
5. mixed small and oversized seeds in one spool (fallback; small seeds
   reconstruct correctly).
6. duplicate attribution involving oversized seeds (identical
   `DirectionalDuplicateError` + attribution with big-int seeds).

Keep the REV2 six-row precedence matrix and all existing gates. Every new mutant
still satisfies the four-part kill rule (e.g. fast-path forced for an oversized
seed → `OverflowError` instead of the pre-D5 value → reds gate 2/4; decoder uses
unsigned `from_bytes` → reds gate 3).

## 6. Oracle durability (non-blocking, resolve before commit — Beta)

The `git cat-file 3e8580a` oracle creates a repo-history dependency that breaks on
shallow clones and source archives. Resolve by **vendoring a frozen oracle
fixture** (e.g. `tests/fixtures/pre_d5_range_miner_npz_writer.py`, the exact
`3e8580a` blob content) loaded as the independent oracle module, **plus** a
faithfulness gate that, when full git history is present, asserts
`sha256(fixture) == sha256(git cat-file 3e8580a:miner/range_miner_npz_writer.py)`
and skips cleanly when history is shallow. That gives durability always and
faithfulness whenever verifiable. (Documenting a non-shallow-clone requirement is
the weaker acceptable alternative.)

## 7. Preserve the allowlist sentry (non-blocking — Beta)

Keep M13 (or an equivalent sentry) proving an **unexpected** exception from the
validator is classified as a backend failure (`ProcessShardedAssemblyError`),
never converted into a canonical `CapturedSpoolReadError`.

## 8. Evidence to close in the clearance record (Beta)

Add the exact commands + outputs + diff summary for:

- **D4 live 8/8, nine mutants intact:**
  `PYTHONPATH=. python3 tests/test_s172_phase5_d4_serial_backend.py` →
  `8/8 D4 gate checks green`.
- **gate-22 whitelist-only:**
  `git diff --stat tests/test_s172_phase4_coordinator.py` → `1 file changed, 27
  insertions(+)`, no deletions; the addition is the two D5 file registrations
  under the standing whitelist rule (comments at 1622/1640–41/1710/1761), no logic
  change.

## 9. Proof obligations

- D1.1 **18/18, zero test edits** — the fallback is additive and never fires for
  in-range seeds, so the no-op holds *and is now proven over the out-of-range
  domain too* by the new gates.
- The merge in `range_miner_npz_writer.py` re-freezes after this Commit-1 rework
  (new sha); Commit 2 again adds zero lines to that file.
- Six new out-of-range gates green across both backends vs the pre-D5 oracle;
  oracle durability resolved; M13 sentry intact; D4/whitelist evidence in the
  record.

Then STOP for Team Alpha review.
