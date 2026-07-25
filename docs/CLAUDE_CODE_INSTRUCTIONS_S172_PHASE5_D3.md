# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3.md — REV3

**S172 RANGE-MINER — Phase 5, Deliverable D3: shared backend-neutral 24→22
columnizer + independent structural validator**

**REV3 changelog — APPROVED FOR IMPLEMENTATION.** Absorbs the Team Beta REV2
review's four amendments: **[A1]** `prng_base` explicitly restricted to a
**forward, non-hybrid base family** (registry membership alone was insufficient —
`java_lcg_reverse` is a valid registry identity but an invalid `prng_base`, and
would have passed the equality rule); **[A2]** destination-`float32`
representability required (Python-level finiteness does not prove
`np.float32(v)` is finite) plus the six count fields must be **integer-valued**;
**[A3]** `Iterable` replaces `Sequence` so the signature matches the one-pass
rule; **[A4]** bound wording narrowed — only `bidirectional_selectivity` is
explicitly permitted to exceed 1; `intersection_weight` is bounded by its own
formula and the earlier "unbounded weight metrics" phrasing was wrong. Team Beta
requires no further pre-write review.

**REV2 changelog** — absorbed the Team Beta pre-implementation review:
**[C1]** D3 is **strictly order-preserving** and owns no ordering policy (REV1
was self-contradictory — it asked both to preserve input order and to be
shuffle-invariant, and mode-first sorting would have undone D3.5's required
global seed order); **[C2]** strict complete 24-field input contract, missing OR
extra key fails; **[C3]** identity consistency validated; **[C4]** `sessions`
and `prng_base` validated despite not becoming arrays; **[C5]** numeric
validation in Python space before materialization; **[C6]** validator adds 1-D
and self-validation; **[C7]** `CANONICAL_ARRAY_CONTRACT` dtypes normalized
through `np.dtype`; **[C8]** Ruling G resolved — D3 fails closed, legacy seams
corrected in **D3.0-B**; **[C9]** gate amendments.

**Audience:** Claude Code on VM 101 (`michael@192.168.3.177`), in
`~/distributed_prng_analysis`. You write and iterate; you do NOT commit, push,
or run WATCHER. When gates + non-regression are green, STOP and report.

**Frozen against HEAD `66f0425`.**

---

## 0. Scope

D3 owns exactly one transformation:

```text
canonical 24-field record sequence  ->  exact 22 typed NumPy arrays
```

**The ordering doctrine — the single most important correction [C1]:**

```text
D3   converts rows.          (preserves the caller's sequence exactly)
D3.25 orders candidate rows.  (trial-major, mode-minor at ingress)
D3.5 orders final winner rows. (globally seed-ascending)
```

The columnizer must not silently take ownership of either ordering policy.

**D3 does NOT:** wire the replacement into any live path (the existing
`_survivors_to_arrays` closure stays in place and in use until D3.5); sort by
anything; perform L2/L3 selection; load any prior file; write, replace, or name
a canonical artifact; populate `binary_npz_path`/`all_npz_path` (both remain
deprecated and permanently `None`, Beta Ruling E); touch D6's adapter or
WATCHER.

**Placement (approved):** `utils/canonical_arrays.py`, sibling to the
`utils/canonical_records.py` D3.25 will introduce —
`canonical_records` = maps + trial context → 24-field records;
`canonical_arrays` = 24-field records → typed 22-array bundle.

## 1. Non-negotiable working rules

1. **Read live source before every claim.** Cites verified at `66f0425`;
   re-verify before depending on them.
2. **Each gate must FAIL on wrong behavior** — C10's mutation set is the Rule-2
   evidence.
3. **Independent oracle, no circularity.** The harness's expected 22 names,
   order, dtypes, and the 24→22 mapping are hand-transcribed. Do **not** import
   the production schema constant and assert against it — the defect corrected
   in D1.1's G9 and again in D3.0's E8.
4. STOP at the gate. No commit/push/WATCHER.

## 2. The frozen array contract (verified at `66f0425`)

22 arrays, frozen order and dtypes — identical to
`convert_survivors_to_binary._EMPTY_NPZ_DTYPES` and its `savez_compressed` call
order:

```text
seeds                       uint32
forward_matches             float32
reverse_matches             float32
window_size                 int32
offset                      int32
trial_number                int32
skip_min                    int32
skip_max                    int32
skip_range                  int32
forward_count               float32
reverse_count               float32
bidirectional_count         float32
intersection_count          float32
intersection_ratio          float32
intersection_weight         float32
bidirectional_selectivity   float32
forward_only_count          float32
reverse_only_count          float32
survivor_overlap_ratio      float32
score                       float32
skip_mode                   uint8
prng_type                   uint8
```

The six `*_count` arrays are **`float32` despite being logically integral** —
reproduce exactly.

**The 24→22 mapping:** `sessions` and `prng_base` do **not** become arrays
(validated anyway, §4.4); `forward_match_rate` → `forward_matches` and
`reverse_match_rate` → `reverse_matches` are **renamed**; the other 22 map 1:1.
24 − 2 = 22 — nothing invented, nothing dropped.

## 3. Ruling G (resolved) — directional match-rate fallback

Team Beta confirmed the finding and ruled:

**(a) D3 strict path** — `records_to_arrays()` requires both
`forward_match_rate` and `reverse_match_rate`. **No aliases, no
opposite-direction substitution, no `0.0` default.** A missing rate raises a
structured error naming the record index and the missing field.

**(b) Legacy seams — corrected in D3.0-B, not deferred to D3.5.** At the
explicitly historical compatibility boundary the active writers may accept only
the **same-direction** alias (`forward_match_rate` or `forward_matches`;
`reverse_match_rate` or `reverse_matches`) and must never use the opposite
directional rate; if neither exists, conversion fails closed. For the inline
accumulator writer the failure must use the existing **tagged** schema-error
family (`ValueError("[S163-KARG-NPZ] …")`, matched at
`window_optimizer_integration_final.py:1894`) so the outer handler re-raises
instead of falling back to the standalone writer — which carries the same
defect. **D3.0-B is a separate commit and is not part of D3.** It does not block
D3 (D3 has no live call site), but must complete before D3.5 replaces the
shared columnizer, before Phase 6 parity certification, and before either
active writer is treated as a trusted comparison oracle.

## 4. Required implementation

### 4.1 Public surface

```python
CANONICAL_ARRAY_CONTRACT: tuple[tuple[str, np.dtype], ...]
# 22 (name, np.dtype(...)) pairs in frozen order. Normalize every dtype through
# np.dtype(...) so comparisons are unambiguous between np.float32 and
# np.dtype("float32"); validation compares array.dtype to the normalized object.

def records_to_arrays(records: Iterable[Mapping[str, object]]) -> dict[str, np.ndarray]:
    """Strictly order-preserving. Returns exactly the 22 arrays.

    Iterable (not Sequence) — the contract is a single forward traversal with no
    len() or indexing requirement [A3].
    """

def validate_array_bundle(arrays: Mapping[str, np.ndarray]) -> None:
    """Independently callable structural validator. Raises on any violation."""
```

### 4.2 Order preservation — binding **[C1]**

`records_to_arrays()` is **strictly order-preserving**: input record *i* becomes
output row *i* in every one of the 22 arrays. It performs **no** sorting by
seed, mode, trial, score, or PRNG identity. Shuffling the input produces the
same corresponding shuffle in all 22 arrays.

**Repeated seeds are legal** — a candidate bundle may contain the same seed once
per mode (D1/D2: cross-mode duplication is legitimate until L2). Do **not**
apply a unique-seed or strictly-increasing-seed wall.

### 4.3 Strict input contract **[C2]**

Every record must carry the **exact** canonical 24-field key set:

- a **missing** key fails closed (naming record index + field);
- an **extra**, unexpected key fails closed — so an upstream schema extension
  cannot silently disappear during the 24→22 conversion.

Input dict *insertion order* is not enforced; the field set and semantic values
are what matter. Output array order remains frozen.

**No `prng_type` → `prng_base` derivation in this function.** D1.1 and D3.25
both emit explicit `prng_type`; accepting its absence would weaken the boundary
for no production requirement. If a historical conversion ever needs that
derivation it belongs in a separately named compatibility adapter, never hidden
inside `records_to_arrays()`.

### 4.4 Identity + omitted-field validation **[C3][C4]**

**Identity consistency**, failing closed before array construction:

```text
skip_mode == "constant"  ->  prng_type == prng_base
skip_mode == "variable"  ->  prng_type == prng_base + "_hybrid"
```

The canonical encoders (`utils.prng_encoding.encode_prng_type` /
`encode_skip_mode`) still validate registry membership; let their `ValueError`
propagate unwrapped.

**`sessions`** (not an array, still validated): must be `list[str]`. Reject
missing, scalar string, tuple, `None`, and non-string members. D3.25 normalizes
tuple/`None` inputs *before* creating canonical records.

**`prng_base`** (not an array, still validated) — must be a **forward,
non-hybrid base family** [A1]. Registry membership alone is **insufficient**:
`java_lcg_reverse`, `java_lcg_hybrid`, and `java_lcg_hybrid_reverse` are all
valid registry identities but are **invalid** `prng_base` values, and a record
like

```text
prng_base = "java_lcg_reverse"   skip_mode = "constant"   prng_type = "java_lcg_reverse"
```

is internally *equal* yet semantically invalid — `prng_type` is a **mode label**,
not a directional identity. So `prng_base` must be a nonempty string that is a
supported base family and must **not** end in `_reverse`, `_hybrid`, or
`_hybrid_reverse`, nor otherwise be a directional or derived registry identity.
It must additionally be consistent with `prng_type` and `skip_mode` per the
equality rule above, and both resulting identities must exist in the canonical
registry encoding where applicable.

### 4.5 Numeric validation before materialization **[C5]**

Validate in **Python space** first — NumPy would otherwise silently narrow,
wrap, or admit non-finite values.

**Integer fields** (`seed`, `window_size`, `offset`, `trial_number`, `skip_min`,
`skip_max`, `skip_range`): must be a real integer with `bool` **excluded**
(`isinstance(x, bool)` fails), and within the destination dtype's range —
`seed` in `uint32` range, the others in `int32` range.

**Float fields** (`forward_match_rate`, `reverse_match_rate`, `forward_count`,
`reverse_count`, `bidirectional_count`, `intersection_count`,
`intersection_ratio`, `intersection_weight`, `bidirectional_selectivity`,
`forward_only_count`, `reverse_only_count`, `survivor_overlap_ratio`, `score`) —
five checks, in this order [A2]:

1. numeric;
2. `bool` **excluded**;
3. the **Python** value is finite;
4. **the converted value is also finite** — `np.isfinite(np.float32(value))`.
   Python-level finiteness does NOT prove `float32` representability: a large but
   finite Python float becomes `inf` under `np.float32`, which would violate the
   output contract even though step 3 passed. Check before the bundle is
   materialized;
5. the field-specific bounds below.

**The six count fields are integer-valued** [A2]. `forward_count`,
`reverse_count`, `bidirectional_count`, `intersection_count`,
`forward_only_count`, and `reverse_only_count` are logical counts stored as
`float32` **only because the frozen NPZ schema requires it**. They must be
nonnegative, integer-valued, `bool`-excluded, and finite as `float32`. Reject
`forward_count = 1.5` — accepting arbitrary nonnegative floats would weaken a
canonical count into a generic measurement merely because its destination column
happens to be `float32`. An integral type or a numeric type whose value is
exactly integral are both acceptable; D1.1 and D3.25 are expected to produce
actual integers.

**Bounds:**

```text
forward_match_rate, reverse_match_rate, score   in [0.0, 1.0]
the six count fields                            >= 0 and integer-valued
intersection_ratio, survivor_overlap_ratio      >= 0
intersection_weight, bidirectional_selectivity  >= 0
```

Apply **only** the bounds frozen above. Do **not** impose one generic `<= 1`
ceiling across every ratio/weight/selectivity field: `bidirectional_selectivity
= len(fwd)/max(len(rev),1)` may legitimately exceed 1 (100 forward / 10 reverse
→ 10.0). This is a bounds-application rule, not a licence to re-derive
aggregates — D3 never recomputes an aggregate from other fields [A4].

### 4.6 One logical pass **[C8 of Beta's §8]**

Compliant shape: allocate one Python accumulation list per output array; iterate
the input records **exactly once**; validate and append each record's converted
values during that traversal; materialize each array once afterward.

Must **not**: perform 22 independent full-record traversals; consume the input
more than once; build 22 list comprehensions over the record sequence; or
reorder records. The parameter is `Iterable` [A3] — use one `for` traversal and
require neither `len()` nor indexing.

### 4.7 Structural validator **[C6]**

`validate_array_bundle` enforces: exactly 22 keys; exact key names; exact
**iteration order** (`tuple(arrays.keys())`); exact dtype per array (compared
against the normalized `np.dtype`); equal lengths; and every array
**one-dimensional** — a `(N,1)` array has a matching outer length but is not
contract-compatible with a 1-D NPZ column.

`records_to_arrays()` must call `validate_array_bundle(result)` **before
returning** (postcondition self-check). The independently callable form remains
necessary because D3.5 will validate bundles assembled through other paths.

## 5. Gate — `tests/test_s172_phase5_d3_columnizer.py`

Independent hand-transcribed oracle for names, order, dtypes, and the 24→22
mapping.

- **C1** 22 arrays, exact names, exact **order** (asserted as a tuple, not a
  set — D3.0's E8 lesson), exact dtypes on a non-empty fixture.
- **C2** value correctness: every array equals hand-computed expectations,
  including both renames.
- **C3** `sessions` and `prng_base` are **absent** from the output.
- **C4** empty input → 22 rectangular zero-length arrays, frozen order/dtypes.
- **C5 fail-closed matrix** — each case raises, naming field + record index:
  every one of the **24** canonical fields missing in turn; an **extra
  unexpected** field; missing `sessions`; missing `prng_base`; missing explicit
  `prng_type`; inconsistent `prng_base`/`prng_type`; inconsistent
  `skip_mode`/`prng_type`; unknown explicit base; unknown explicit type; each
  match rate missing (must raise — **not** cross-fall-back, §3). Plus the
  base-family cases [A1]: a **reverse** identity as `prng_base` → fail; a
  **hybrid** identity as `prng_base` → fail; a **hybrid-reverse** identity as
  `prng_base` → fail; a valid base with an unrelated-but-valid `prng_type` →
  fail. Plus a generator/iterator input consumed exactly once [A3].
- **C6 cross-mode seed survives:** seed X present in both a constant and a
  variable record yields **two** rows, each with its own mode's rates and
  aggregates; no collapse, no unique-seed wall.
- **C7 exact order preservation [C1]:** for an intentionally nontrivial input
  sequence (not seed-sorted, not mode-grouped), every output array preserves the
  exact input row order; shuffling the input produces the same corresponding
  shuffle in all 22 arrays.
- **C8 parity** with the corrected post-D3.0 legacy columnizers on a fixture
  where **all 24 fields are explicit** (so neither writer's compatibility
  fallback is exercised) and already in the desired order (D3 preserves input
  order): all 22 arrays `np.array_equal` and dtype-identical against **both**
  `convert_survivors_to_binary`'s array block and the live inline
  `_survivors_to_arrays` closure (extract by AST line-range as D3.0's harness
  does, so editing the seam changes what the gate runs). Parity is a regression
  check; C1/C2's hand oracle remains load-bearing.
- **C9 validator, called on hand-built bundles** (not only on
  `records_to_arrays` output): dropped key; added key; reordered keys; wrong
  dtype; unequal lengths; **two-dimensional array**; **scalar array**;
  **non-NumPy value under one key**.
- **C10 mutation proof** — kill each of: dropped array; added array; reordered
  keys; renamed key; wrong dtype (`float32`→`int32` on a `*_count`); match-rate
  rename swapped (forward↔reverse); `sessions` emitted; `prng_base` emitted;
  restored silent default for a missing field; restored `→ 'java_lcg'` terminal
  default; unique-seed wall collapsing C6's pair; **sorting records inside D3
  (by mode, and separately by seed)**; **cross-direction match-rate fallback**;
  **accepting a missing explicit `prng_type`**; **ignoring inconsistent
  `prng_base`**; **allowing silent integer overflow**; **validating only
  registry membership while omitting the base-family restriction [A1]**;
  **removing the post-conversion `np.float32` finiteness check while keeping the
  Python-level one [A2]**; **accepting a fractional count [A2]**; **removing the
  internal `validate_array_bundle` call** (inject a malformed internally built
  bundle and prove the harness detects the lost postcondition). Report each red
  signature.

Adversarial numeric gates (may live under C5 or C10): negative seed; seed above
`uint32`; `int32` overflow on an int field; `NaN` match rate; infinite score;
`bool` used as an integer or a float; **a finite Python value that overflows to
`float32` infinity**; **a fractional count such as `1.5`**; **a negative
count**; **a boolean count** [A2].

**Blocking non-regression:** D3.0 gate 10/10, D2 7/7, D1.1 18/18, D1.0 8/8, D0
12/12, Phase 4 63/63, Phase 3 17/17. Baseline captured green at `66f0425`
**before** any edit.

## 6. Scope authorization

**May modify:** `utils/canonical_arrays.py` (new),
`tests/test_s172_phase5_d3_columnizer.py` (new),
`tests/test_s172_phase4_coordinator.py` (gate-22 registration only), D3
governance documents.

**Must NOT:** rewire any live producer or accumulator call site; touch
`window_optimizer_integration_final.py`, `convert_survivors_to_binary.py`,
`persistent_worker_coordinator.py`, `zmq_sqlite_coordinator.py`, or
`miner/range_miner_npz_writer.py`. D3.0-B corrections are a separate commit and
do not belong in D3.

## 7. Stop conditions

- a required array cannot be produced from the 24-field record without
  inventing a value;
- strict validation reds against a legitimately-produced D1.1 assembly record —
  that would mean D1.1's records lack a field D3 requires, or carry an extra
  one; **report, do not add a default or relax the key set**;
- C8 parity fails on an all-fields-explicit fixture — a real semantic
  difference beyond the §3 finding; STOP and report;
- any change is needed outside the §6 may-modify list;
- any gate passes only by weakening it.

## 8. Report

Diff + status, full command/output evidence, the pre-edit baseline, mutation
evidence with per-mutant red signatures, and explicit confirmation that no
production call site was rewired. Then STOP for Team Alpha review.
