# CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md — REV2

**S172 — `D6.2: 24-field checkpoint, canonical reconciliation, and finalizer resume path.`**

**This is the deferred half of D6.1.** D6.1 made the incremental checkpoint *write* for the first
time (relocated to `.s172_checkpoint/`, suffix bug fixed, failures visible, fsync + temp cleanup)
but left the **in-memory list-clear disabled**, because the checkpoint persists 4 arrays while the
D3.5 finalizer requires all 24 `CANONICAL_RECORD_FIELDS`. Consequently the **S166 OOM protection
does not exist**: the candidate list grows unboundedly on long runs. Phase-7 blocker.

**Status:** submitted to Beta for ruling on numbering and sequencing.

**Base:** HEAD `55daf4b`. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`. Implement
and iterate; do **NOT** commit, push, or run WATCHER. STOP at the gate.

---

## REV1 → REV2 — what changed and why

REV1 §2 instructed Claude Code to *"read `CANONICAL_RECORD_FIELDS` and enumerate exactly which 24
fields must round-trip, with dtype and encoding for each. Report that table before writing the
writer."*

**Alpha did that read instead, at source, and it changes the shape of the work.** Twenty-two of the
twenty-four fields already have a frozen dtype and, where they are strings, a frozen codec.
Treating the encoding as an open design question would have produced a **second encoding policy**
— structurally the same defect `_l2_sort_key` exists to prevent, and the same defect Phase 0 fixed
when three divergent hardcoded `prng_type` dicts silently collapsed unknown values to `0`.

REV2 therefore **supplies the table** (§2) and narrows the open decision to **one field**. The
reconciliation mandate, transaction identity, resume path, ordering and gates are unchanged from
REV1 except where noted.

---

## 0. Why this exists — the three facts that define the work

1. **The checkpoint schema is inadequate for its stated purpose.** It persists 4 arrays (`seeds`,
   `score`, `forward_match_rate`, `reverse_match_rate`) under schema version
   `s172-d6.1-four-field-v1` (`window_optimizer_integration_final.py:347`). The finalizer consumes
   the in-memory list and requires **24** canonical fields. 4 cannot reconstruct 24.
2. **The S166 comment's guarantee was doubly false.** *"data is safe in NPZ"* justified clearing
   the list — but the write always failed *and*, even had it succeeded, a 4-field file could never
   have backed the claim. D6.1 fixed the first half. D6.2 must make the claim true or leave the
   clear off forever. The flag is `_FLUSH_CLEAR_IN_MEMORY = False` (`:437`).
3. **The current merge policy is not the canonical one.** The flush helper reconciles duplicate
   seeds inline with `s.get("score", 0.0) > seen[seed].get("score", 0.0)`. The canonical authority
   is `_l2_sort_key` / `_select_l2_winners` (`utils/run_finalizer.py:688-747`, frozen, Ruling D).

**Read all three at source before writing anything.**

---

## 1. The reconciliation authority — unchanged from REV1, and load-bearing

Beta ruled: *"merge by seed is valid only if it invokes the **existing canonical accumulator
reconciliation rule**. It must not introduce a new dictionary policy such as arbitrary first-wins
or last-wins… If differing records for the same seed are a producer defect, recovery must raise
rather than choose one."*

The canonical rule is **highest-wins on every component**:
1. highest canonical **float32** score;
2. then **lowest `trial_number`**;
3. then constant-before-variable — **only** as a tiebreak within one trial.

Three properties to inherit, not approximate:
- **The comparison domain is float32.** Two Python floats differing only beyond float32 precision
  are an **exact tie** and fall through to the trial-number tiebreak. Comparing pre-rounding
  float64 while storing the rounded value **is the defect this converts away.**
- **The result is order-independent** — within one seed the key is a strict total order.
- **A same-trial/same-mode collision raises `AccumulatorConsistencyError`.** After D1/D2 it is
  impossible, so its presence means the accumulator received one trial's population twice. This is
  Beta's *"recovery must raise rather than choose."*

**How the current flush helper violates each:** float64 comparison; no trial_number or mode
tiebreak; `.get("score", 0.0)` silently defaults a missing score to zero; never raises on a
same-trial/same-mode duplicate; and its prior-NPZ merge path reconstructs bare `{"seed","score"}`
records (`window_optimizer_integration_final.py:281-286`) — **discarding `trial_number`,
`skip_mode` and all provenance, so the information the canonical rule needs is already gone.**

**Mandate: import and call `_select_l2_winners` / `_l2_sort_key`. Do not reimplement, do not
approximate, do not write a second policy.** If importing creates a circular dependency, extract
the authority to a shared module and have *both* call sites use it — never fork it.

---

## 2. Schema — RESOLVED. Reuse the frozen contracts; do not invent an encoding.

**This section replaces REV1's "report the table before writing the writer" step.** The table is
below, derived at source at `55daf4b`. Claude Code's job is to **verify it still holds and
implement it**, not to redesign it.

### 2.1 The two authorities to import

| authority | location | supplies |
|---|---|---|
| `CANONICAL_ARRAY_CONTRACT` | `utils/canonical_arrays.py:98-123` | frozen dtype for 22 of the 24 fields |
| `utils/prng_encoding` | `encode_/decode_skip_mode`, `encode_/decode_prng_type` | the string↔uint8 codec, registry-derived, `ENCODING_VERSION = "3.2.0"` |

**Import both. Do not transcribe either.** `prng_encoding` exists precisely because three
divergent hardcoded dicts once collapsed unknown/hybrid `prng_type` values to `0`, destroying
provenance. A fourth copy inside the checkpoint writer would recreate that defect in a new place.

*Note the naming boundary:* the checkpoint reconstructs **records**, not arrays, so it stores the
**record** field names. `forward_match_rate` / `reverse_match_rate` are renamed to
`forward_matches` / `reverse_matches` only in the **array** domain
(`_RENAMED_SOURCE_FIELDS`, `canonical_arrays.py:158-162`). Do not apply that rename here.

### 2.2 The 24 fields

| # | record field | storage dtype | source of dtype | note |
|---|---|---|---|---|
| 1 | `seed` | `uint32` | contract (`seeds`) | |
| 2 | `forward_match_rate` | `float32` | contract (`forward_matches`) | unit interval |
| 3 | `reverse_match_rate` | `float32` | contract (`reverse_matches`) | unit interval |
| 4 | `score` | `float32` | contract | **the L2 comparison domain — see §2.5** |
| 5 | `window_size` | `int32` | contract | |
| 6 | `offset` | `int32` | contract | |
| 7 | `skip_min` | `int32` | contract | |
| 8 | `skip_max` | `int32` | contract | |
| 9 | `skip_range` | `int32` | contract | |
| 10 | `sessions` | **OPEN — §2.4** | not an array | `list[str]`, the one real decision |
| 11 | `trial_number` | `int32` | contract | L2 tiebreak component |
| 12 | `prng_base` | **not stored — §2.3** | not an array | exactly derivable |
| 13 | `skip_mode` | `uint8` | contract + `encode_skip_mode` | `{constant:0, variable:1}` |
| 14 | `prng_type` | `uint8` | contract + `encode_prng_type` | registry-derived, 44 keys |
| 15 | `forward_count` | `float32` | contract | integral-valued but float32 by frozen schema |
| 16 | `reverse_count` | `float32` | contract | ″ |
| 17 | `bidirectional_count` | `float32` | contract | ″ |
| 18 | `intersection_count` | `float32` | contract | ″ — duplicates `bidirectional_count` **deliberately** |
| 19 | `intersection_ratio` | `float32` | contract | nonnegative, **not** ceilinged at 1 |
| 20 | `forward_only_count` | `float32` | contract | integral-valued |
| 21 | `reverse_only_count` | `float32` | contract | integral-valued |
| 22 | `survivor_overlap_ratio` | `float32` | contract | nonnegative |
| 23 | `bidirectional_selectivity` | `float32` | contract | **may legitimately exceed 1** — `len(fwd)/max(len(rev),1)` |
| 24 | `intersection_weight` | `float32` | contract | nonnegative |

**Do not apply a generic `<= 1` ceiling across ratio/weight/selectivity fields.** `canonical_arrays.py:229-238`
is explicit that this would be wrong. Reuse its bound sets rather than re-deriving them.

**`allow_pickle=False`, no object arrays.** With the table above, every stored field is a plain
numeric array — the encoding problem reduces entirely to §2.4.

### 2.3 `prng_base` — store nothing, derive it, and version-guard the derivation

`prng_base` is **exactly recoverable** from `(prng_type, skip_mode)` by the identity rule already
validated at `canonical_arrays.py:306-364`:

```
skip_mode == "constant"  ->  prng_type == prng_base
skip_mode == "variable"  ->  prng_type == prng_base + "_hybrid"
```

The rule is enforced at ingress, so any record reaching the checkpoint already satisfies it. The
derivation is a strict function with no information loss.

**But it inherits `prng_encoding`'s own caveat, which that module states plainly:** *"NPZ files are
commit-local artifacts, NOT a durable ABI. If `KERNEL_REGISTRY` changes, ids may shift."* A
checkpoint written under one registry and read under another would mis-decode `prng_type` —
**and this is already true today for the stored `prng_type` itself**, so derivation adds no new
fragility. It does, however, make the exposure worth closing.

**Required: add `encoding_version` to the transaction identity (§3)**, read from
`prng_encoding.ENCODING_VERSION`. A checkpoint whose encoding version does not match the running
code must **fail closed as unreadable**, not decode into plausible-looking wrong provenance.
`tests/test_prng_encoding.py` already pins `len(PRNG_TYPE_ENCODING) == 44` on purpose, so a
registry change is a deliberate, visible event — this makes the checkpoint a consumer of that
same signal.

*If Beta prefers storage over derivation, the fallback is a fixed-width `<U` column. Alpha
recommends derivation plus the version guard: it stores less, and the guard closes an exposure
that exists either way.*

### 2.4 `sessions` — the one open decision

`sessions` is `list[str]`, never becomes an array (`canonical_arrays.py:139-142`), is validated
anyway (`_check_sessions`, `:283-301`), and is **defensively copied** at construction
(`canonical_sessions`, `canonical_records.py`) so a caller mutating its original list cannot reach
an already-produced record.

Constraints that any encoding must satisfy:
- **`[]` is legal and must round-trip as `[]`.** `canonical_sessions(None)` returns `[]`.
- **A scalar string is NOT a session list.** `canonical_sessions` fails closed on a bare string
  precisely because the legacy `getattr(config, 'sessions', 'all')` fallback **fabricated a session
  name**. The decoder must never reconstruct a scalar into `[scalar]`.
- **Order must be preserved** — it is a `list`, not a set.
- **`allow_pickle=False`**, so no ragged object array.

**Alpha's recommendation: CSR-style flat encoding.** Two arrays —
`sessions_values` (`<U` unicode, flat, all records concatenated in order) and `sessions_offsets`
(`int64`, length `n_records + 1`). Record *i*'s sessions are
`sessions_values[offsets[i]:offsets[i+1]]`. Empty lists are represented by equal consecutive
offsets, so `[]` round-trips exactly and is distinguishable from every non-empty value.

**Explicitly rejected alternative, and why it matters:** a per-trial `sessions` table keyed by
`trial_number` would be smaller, on the assumption that `sessions` is trial-constant.
`canonical_records.py` does place *the context's single `sessions` object* into every record of a
trial, which makes trial-constancy **likely** — but **Alpha has NOT verified it holds across the
entire flush path, and this brief does not assert it.** Building the schema on an unverified
invariant is how dead dimensions get created. **CSR is correct whether or not the invariant
holds**, at trivial cost.

**Deliverable:** confirm the CSR round-trip against real flush data including at least one record
with `sessions == []`, and report the encoding actually implemented.

### 2.5 Float32 storage has a consequence for the gates — read this before writing §6

`score` is stored as **float32**. Therefore two records differing **only beyond float32
precision** arrive from the checkpoint **already bit-identical** — the tie is structural, not a
fall-through.

**A gate that constructs both sides by writing them to a checkpoint and reading them back cannot
fail, and would be vacuous.** The float64-difference case must be exercised on the **in-memory
reconciliation path**: one side a fresh float64 record, the other from the checkpoint. State in
the report which path each duplicate-matrix case exercises.

### 2.6 Unchanged from REV1

- Compression: **keep `savez_compressed`** per Beta's D6.1 ruling. A durable run checkpoint is not
  a worker transport artifact; **D5 §6.7.A stays untouched.** Re-assert the separation in a gate.
- The checkpoint lives under `.s172_checkpoint/<run_id>/`, never a finalizer-owned path. Do not
  reintroduce the D3.5 symlink collision.
- Preserve exact dtypes; do not silently widen or narrow.

---

## 3. Transaction identity

Both checkpoint members carry matching transaction metadata. D6.1's keys
(`window_optimizer_integration_final.py:400-407`) plus D6.2's additions:

| key | status |
|---|---|
| `checkpoint_schema_version` | **update** — the four-field marker `s172-d6.1-four-field-v1` must change |
| `checkpoint_id` | unchanged |
| `checkpoint_sequence` | unchanged |
| `run_id` | unchanged |
| `logical_candidate_count` | unchanged |
| content digest | **widen** from `four_field_content_digest` to cover all persisted fields; rename accordingly |
| `encoding_version` | **NEW — §2.3** |

Restart behaviour, exactly as Beta specified:
- matching IDs/digests → **pair accepted**;
- mismatched IDs → **interrupted sequential replacement detected**;
- malformed/unreadable member → **recover from the valid member where possible**;
- neither valid → **fail closed WITHOUT clearing in-memory state**;
- **encoding version mismatch → unreadable** (D6.2 addition).

The repaired pair must be **regenerated and revalidated before normal flushing resumes.**

---

## 4. The finalizer resume path

On resume: read the checkpoint, rebuild the raw candidate records, and produce a certified
generation **field-for-field identical, and in identical canonical order**, to an uninterrupted
reference run on the same inputs.

**Beta's locked end-state claim — this is what D6.2 may assert:**
> Interrupted sequential replacement is detectable, and restart recovery reconstructs the same
> canonical cumulative checkpoint as an uninterrupted execution.

It may **not** claim the two-file checkpoint is jointly atomic. *"Atomic checkpoint"* appears only
with the qualification *each artifact replacement is atomic; the pair is not jointly atomic.*

---

## 5. Enabling the list-clear (the actual OOM fix)

Only after §2–§4 hold. Required ordering is Beta's, verbatim:

```
construct cumulative canonical state
write both temporary artifacts
fsync/close as required
validate both temporary artifacts
replace destination A
replace destination B
validate the installed pair
only then clear the flushed in-memory entries
```

A mutant that clears **after the first replace but before the second** must fail. D6.1 already
gates the ordering property with the flag forced on; D6.2 turns `_FLUSH_CLEAR_IN_MEMORY` on for
real and must additionally prove the finalizer still receives complete 24-field input after a
clear — **via the resume path, not the truncated stump.**

---

## 6. Gates — `tests/test_s172_d6_2_checkpoint_reconciliation.py`

Beta's required duplicate-seed matrix, for the same seed on both sides:

| case | required behaviour | path (state it) |
|---|---|---|
| identical records | reconciles to that record; idempotent | |
| different match rates | canonical `_l2_sort_key` winner, float32 domain | |
| float64-only difference | **exact tie → trial-number tiebreak** | **in-memory — §2.5** |
| different trial_number, same score | **lower trial_number wins** | |
| same trial + same mode | **raises `AccumulatorConsistencyError`** | |
| different provenance/mode metadata | canonical rule decides; no ad-hoc choice | |
| restart-replay duplicate | idempotent, no double-count | |

Plus:
- **G-24-FIELD-ROUNDTRIP:** all 24 fields round-trip exactly; `allow_pickle=False`; no object
  arrays; dtypes preserved; **`sessions == []` included**; **`prng_base` derives correctly for
  both `constant` and `variable`.**
- **G-ENCODING-AUTHORITY (new):** the writer imports `utils.prng_encoding` and
  `CANONICAL_ARRAY_CONTRACT`; **AST-assert no literal `{'constant': 0, ...}`, no literal prng_type
  map, and no transcribed dtype table anywhere in the checkpoint path.**
- **G-ENCODING-VERSION (new):** a checkpoint stamped with a different `encoding_version` is
  rejected as unreadable and does **not** decode.
- **G-AUTHORITY:** reconciliation calls the canonical authority (AST + runtime — no second policy
  anywhere in the flush path).
- **G-IDENTITY:** the transaction-identity fields of §3 present and matching on a healthy pair.
- **G-RESTART-{A,B,C,D}:** the four restart outcomes, including **fail-closed without clearing**
  when neither member is valid.
- **G-RESUME-PARITY:** a run interrupted at each write/replace boundary, resumed, produces a
  generation **field-for-field and order-identical** to an uninterrupted reference run.
- **G-CLEAR-SAFE:** with the clear enabled, the finalizer still receives complete 24-field input.
- **G-CADENCE:** D3.25's one-attempt-per-trial invariant unchanged; distinguish **one attempt per
  trial**, **one successful transaction per successful flush**, and **recovery retries**.
- **G-COMPRESSION-CONTRACT:** D5 §6.7.A artifact ban still separate and intact.
- **G-NO-SYMLINK-COLLISION:** the checkpoint never writes a finalizer-owned path; publication
  still succeeds after arbitrarily many flushes.

**Mutants** (four-part kill rule; each must fail **from its injected defect**, and the harness must
swap the source every gate builds from — the D6.1 vacuous-mutant lesson): reintroduce the inline
`score >` policy; compare float64 instead of float32; drop the trial_number tiebreak; swallow the
same-trial/same-mode collision instead of raising; clear between the two replaces; drop a
transaction-identity field; write to a finalizer-owned path; **hardcode the skip_mode map instead
of importing it**; **stamp a stale `encoding_version` and prove it reds.**

---

## 7. Non-regression

Capture green before any edit: D1.1 · D1.0 · D0 · D2 · D3.0 · D3 · D3.25 · D3.5 · D4 · D5 ·
D6 3.A · D6-threshold · D6.1 · Phase 3 · Phase 4. After: all green plus D6.2. **D3.25 must stay
13/13** (cadence) and **D3.5 60/60** (the finalizer is touched by the resume path).

All test commands run on **VM101** with `source ~/venvs/torch/bin/activate` first — a bare shell
yields false `CuPy not available` / `Optuna not available` reds.

---

## 8. Scope — do NOT touch

The D6 threshold/provenance/residue work; PWC/ZMQ ingress; the D3.25 four-map contract;
`TestResult` shape; D5's artifact contract; `serial_reference` as default. Do not modify
`_l2_sort_key`, `_select_l2_winners`, `CANONICAL_ARRAY_CONTRACT` or `utils/prng_encoding` — they
are the authorities being **reused**, not revised.

---

## 9. Report

The §2.2 table **as verified at HEAD** (flag any drift from this brief); the `sessions` encoding
implemented and its `[]` round-trip; the `prng_base` derivation evidence for both skip modes; the
reconciliation call path proving the canonical authority is invoked and no second policy exists;
**which path each duplicate-matrix case exercises (§2.5)**; the four restart outcomes; the
resume-parity evidence; gate/mutant counts; confirmation D3.25 and D3.5 are unchanged; and the
exact end-state claim language from §4. Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every gate prints its own name and a non-trivial assertion count; the
  resume-parity gate reports the compared generation identities, not a boolean.
- **clean control:** an uninterrupted reference run must pass every restart gate's healthy branch.
- **fault-injection control:** the mutant list in §6; each must red **from its own defect**, proven
  by the four-part kill rule.
- **completion sentinel:** the suite terminates in `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only
  `PASS` accepts.
- **unavailable-observer behavior:** with rigs down, any fleet-dependent arm reports
  `UNAVAILABLE` — **never** `PASS`. D6.2 should have no fleet dependency; if one appears, that is
  a finding to report, not to work around.
- **audit claim scope:** the schema table in §2.2 is **repo-scoped**, derived from
  `utils/canonical_arrays.py` and `utils/prng_encoding.py` at `55daf4b`.
- **searched surfaces:** tracked repo at `55daf4b`.
- **unavailable surfaces:** host state on VM101 and the rigs; any uncommitted local modification;
  the live registry contents if `KERNEL_REGISTRY` has changed since the clone.
