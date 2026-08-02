# CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md — REV3

**S172 — `D6.2: 24-field checkpoint, canonical reconciliation, and finalizer resume path.`**

**Base:** HEAD `9470750`. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`. Implement
and iterate; do **NOT** commit, push, or run WATCHER. STOP at the gate.

**Status.** Beta ratified D6.2's numbering, purpose and Phase-7 priority, confirmed the §2 table
matches the live authorities, and returned REV2 for six corrections. **Five are incorporated in
full.** The sixth is contested with evidence under §0.2, per owner ruling.

---

## 0. Disposition of Beta's REV2 ruling

### 0.1 Accepted in full — corrections 1, 2, 3, 5, 6

Two of these repair defects Alpha introduced that would have produced **gates that cannot pass**:

- REV2's duplicate matrix required identical replayed records to reconcile idempotently **and**
  every same-trial/same-mode collision to raise. `_select_l2_winners` always raises on that
  collision, identical or not. **The two rows were unsatisfiable together**, and the only escape
  would have been Claude Code inventing policy — the exact thing Beta's D6.1 ruling forbids.
- REV2 claimed a resumed run could produce a generation *"field-for-field identical"* to an
  uninterrupted one. **It cannot**, and a gate written to that claim either fails permanently or
  gets weakened until it passes. That is the VIR-2 class.

Corrections 2, 3 and 6 close real holes: no restart discovery existed; `ENCODING_VERSION` alone
does not detect registry renumbering; and storage-domain equality is not float equality.

### 0.2 Contested — correction 4's remedy (not its finding)

**Beta's finding is correct and Alpha confirms it at source.** D6.1's members are asymmetric
(`window_optimizer_integration_final.py:866-871`):

| member | payload |
|---|---|
| `incremental_survivors_all.npz` | `seeds`, `score` + identity — **two data arrays** |
| `incremental_survivors_binary.npz` | `seeds`, `forward_match_rate`, `reverse_match_rate`, `score` + identity — **four** |

So D6.1's *"recover from the valid member where possible"* **is already false for member A**: if B
is lost, A alone cannot reconstruct even the four-field state. Beta is not tightening a promise —
**it is repairing one that was never true.**

**Alpha contests the remedy.** Beta requires both members to carry the complete reconstructible
payload. That **doubles write volume on every flush** for the life of the system, to buy
single-member recovery.

**Alpha's position: delete the false promise instead of funding it.** Member A becomes a declared
compatibility stub; recovery requires member B; the contract says so; a gate enforces it. Same
honesty, no duplication. **Cheaper to build, cheaper to run, and it removes a false claim rather
than making a true one expensive.**

**Implement §4.2 (Alpha's position). If Beta rules for duplication, §4.3 is the drop-in
replacement** — it is specified so the change is contained.

### 0.3 Not accepted as a Phase-7 gate — the multi-stripe protocol-liveness item

Beta's disposition adds the multi-stripe protocol-liveness investigation to the Phase-7 gate.
**Owner ruling: it does not block the soak.** It originated as Alpha's own **`[UNVERIFIED]`**
observation, in which Alpha stated explicitly that **fixture limitation versus production defect
was not established.** An unestablished observation is characterized during a soak, not made a
precondition for one. It remains tracked in `docs/BACKLOG.md` §6.

### 0.4 Settled, do not reopen — seed storage width

`seed` is stored **uint32**, per `CANONICAL_ARRAY_CONTRACT`. This is ruled at **`a63c361`**
(TB-approved, Seed-Domain v1.1): the java_lcg family has 48-bit internal state, the canonical
artifact stores uint32, so the sweep covers the `high16=0` stratum — 1 part in 65,536. At
window ≥ 3 all 65,536 high-state classes produce distinct draw sequences; no reduction exists.
**TFM does functional mimicry, not state reversal, so the approved fix was honest stratum
labelling, not a uint64 migration.** Nine module-owned sidecar constants including
`seed_storage_dtype` declare it, none caller-supplied.

The current checkpoint writes `uint64` (`:851`) — over-wide storage of values already constrained
to the uint32 domain wall. **D6.2 stores uint32 per contract. There is no decision here and no
truncation exposure.**

---

## 1. Why this exists

1. **The checkpoint schema is inadequate for its purpose.** It persists 4 arrays under
   `s172-d6.1-four-field-v1` (`:347`). The finalizer consumes the in-memory list and requires
   **24** `CANONICAL_RECORD_FIELDS`. 4 cannot reconstruct 24.
2. **The S166 guarantee was doubly false.** *"data is safe in NPZ"* justified clearing the list —
   but the write always failed *and* a 4-field file could never have backed the claim. D6.1 fixed
   the first half. `_FLUSH_CLEAR_IN_MEMORY = False` (`:437`).
3. **The current merge policy is not canonical.** The flush helper uses
   `s.get("score", 0.0) > seen[seed].get("score", 0.0)`. The authority is `_l2_sort_key` /
   `_select_l2_winners` (`utils/run_finalizer.py:688-747`, frozen, Ruling D).

---

## 2. Schema — reuse the frozen contracts

### 2.1 Authorities to import, never transcribe

| authority | location | supplies |
|---|---|---|
| `CANONICAL_ARRAY_CONTRACT` | `utils/canonical_arrays.py:98-123` | dtype for 22 of 24 fields |
| `utils.prng_encoding` | `encode_/decode_skip_mode`, `encode_/decode_prng_type` | string↔uint8 codec |
| `canonical_map_hash()` | `utils/run_finalizer.py:486` — **exported** | the encoding-map digest (§3) |

`prng_encoding` exists because three divergent hardcoded dicts once collapsed unknown `prng_type`
values to `0`, destroying provenance. **A fourth copy would recreate that defect.**

The checkpoint reconstructs **records**, so it stores **record** field names.
`forward_match_rate`/`reverse_match_rate` are renamed only in the **array** domain
(`_RENAMED_SOURCE_FIELDS`, `canonical_arrays.py:158-162`). **Do not apply that rename here.**

### 2.2 The 24 fields

Unchanged from REV2 §2.2 and confirmed by Beta against the live authorities. `seed` → `uint32`
(§0.4); fields 2-9, 11, 15-24 take their contract dtype; `skip_mode`/`prng_type` → `uint8` via the
codec; `sessions` → CSR (§2.4); `prng_base` derived (§2.3).

**Reuse `canonical_arrays.py:229-238`'s bound sets.** Do not apply a generic `<= 1` ceiling:
`bidirectional_selectivity` is `len(fwd)/max(len(rev),1)` and **may legitimately exceed 1**.

### 2.3 `prng_base` — derived, per Beta's approval

Exactly recoverable from `(prng_type, skip_mode)` by the rule already validated at
`canonical_arrays.py:306-364`:

```
skip_mode == "constant"  ->  prng_type == prng_base
skip_mode == "variable"  ->  prng_type == prng_base + "_hybrid"
```

Enforced at ingress, so every record reaching the checkpoint satisfies it. Version-guarded by §3.

### 2.4 `sessions` — CSR, per Beta's approval

`sessions_values` (`<U`, flat, records concatenated in order) + `sessions_offsets` (`int64`,
length `n_records + 1`). Record *i* is `sessions_values[offsets[i]:offsets[i+1]]`.

Constraints: `[]` is legal and round-trips as `[]`; **a scalar string is not a session list** —
`canonical_sessions` fails closed on one precisely because the legacy
`getattr(config, 'sessions', 'all')` fallback **fabricated a session name**, so the decoder must
never reconstruct a scalar into `[scalar]`; order preserved; `allow_pickle=False`.

### 2.5 Float32 storage and the duplicate matrix — read with §5.1

`score` is stored **float32**, so two records differing only beyond float32 precision arrive from
the checkpoint **bit-identical**. Beta's correction: **the float64-only case must use different
trial numbers** if it is meant to exercise the float32 tie and the trial-number tiebreak. With the
same trial and mode, canonical equality makes it an **idempotent replay** and canonical inequality
makes it **corruption** — neither is a tiebreak test.

### 2.6 Unchanged

`savez_compressed` retained (durable run checkpoint ≠ worker transport artifact; **D5 §6.7.A
untouched**). Checkpoint stays under `.s172_checkpoint/<run_id>/`, never a finalizer-owned path.
Exact dtypes; no silent widening or narrowing.

---

## 3. Transaction identity

| key | status |
|---|---|
| `checkpoint_schema_version` | **update** — the four-field marker must change |
| `checkpoint_id` | unchanged |
| `checkpoint_sequence` | unchanged |
| `run_id` | unchanged |
| `logical_candidate_count` | unchanged |
| content digest | **widen** to cover all persisted fields; rename from `four_field_content_digest` |
| `encoding_version` | **NEW** — `prng_encoding.ENCODING_VERSION` |
| `canonical_map_hash` | **NEW, Beta correction 3** — import from `run_finalizer`; do not reimplement |
| `run_context_digest` | **NEW, Beta correction 2** — see §4.1 |

**Why `canonical_map_hash` and not the version string alone.** `tests/test_prng_encoding.py` pins
`len(PRNG_TYPE_ENCODING) == 44`. **Renaming or replacing a key preserves the count and preserves
`ENCODING_VERSION` while renumbering every id after it alphabetically.** `canonical_map_hash()`
hashes the maps themselves, so it catches that. **A member whose `encoding_version` or
`canonical_map_hash` differs fails BEFORE decoding** — never decode into plausible wrong
provenance.

**The digest covers, per field: name, dtype, shape, and contiguous bytes.** The D6.1 digest
(`:513-528`) covers name, dtype and bytes — **shape is missing**, and two differently-shaped
arrays with identical bytes would collide. Add it.

---

## 4. Restart, discovery and ownership (Beta correction 2)

### 4.1 Discovery is caller-driven — no inference

The default run id embeds pid and time (`:448`), so **a restarted process selects a different
directory by construction.** Required:

- an **explicit resume run id or checkpoint handle from the caller**;
- **no "newest checkpoint directory" inference** — anywhere, at any layer;
- a frozen **`run_context_digest`** binding: dataset identity, repository revision, PRNG identity,
  skip modes, seed interval, and applicable execution-set identity;
- **rejection before decoding** when the supplied context differs;
- the next `checkpoint_sequence` initialized **above the highest accepted persisted sequence**.

**D6.2 restores accumulator state. It does NOT restore the optimizer's execution cursor**, and
must not claim to unless that cursor is independently persisted and tested. Say so in the report.

### 4.2 Member recovery — Alpha's position (IMPLEMENT THIS)

- **Member B (`incremental_survivors_binary.npz`) is the recovery member** and carries the
  complete reconstructible 24-field payload.
- **Member A is a declared compatibility stub.** Its identity block is complete; its payload is
  not, and the contract states this in the module docstring and the schema version.
- **Recovery requires member B.** If B is invalid, recovery **fails closed without clearing
  in-memory state** — regardless of A's condition.
- Each member **validates its own content digest** (§3).
- **Mixed pair:** the higher valid sequence may be selected **only when `run_id`,
  `run_context_digest`, schema and encoding identity all agree.** Equal sequence with different
  `checkpoint_id`, or **any** context disagreement, **fails closed**.
- **Recovery writes and validates a new pair before clearing or continuing.**
- **G-STUB-HONESTY:** a gate asserts A is documented as a stub and that no code path claims
  recovery from A. **The false promise must be removed, not merely unused.**

### 4.3 Fallback if Beta rules for duplication

Both members carry the complete payload; recovery may proceed from either; everything else in
§4.2 stands unchanged. **The change is confined to the writer's payload dict and G-STUB-HONESTY's
replacement by a symmetry gate.** Do not implement this unless directed.

---

## 5. Reconciliation

### 5.1 Replay normalization, then the canonical authority (Beta correction 1)

Order is binding:

1. **Canonicalize both records into checkpoint storage domains** (float32 score, uint8 codes, etc).
2. **Collapse a bit-identical 24-field replay** before winner selection.
3. If `(seed, trial_number, skip_mode)` matches but **any canonical field differs** → raise
   `AccumulatorConsistencyError`.
4. Pass the remainder to **`_select_l2_winners`**.

**Step 2 is replay normalization, not a second winner policy**, and the report must state it in
those terms. `_select_l2_winners` remains the only winner-selection authority.

### 5.2 The canonical rule — inherit, never approximate

Highest **float32** score → lowest `trial_number` → constant-before-variable **within one trial
only**. Order-independent. Comparing pre-rounding float64 while storing the rounded value **is the
defect this converts away.**

**How the current flush helper violates it:** float64 comparison; no trial_number or mode
tiebreak; `.get("score", 0.0)` silently defaults a missing score to zero; never raises on
same-trial/same-mode; and its prior-NPZ merge path reconstructs bare `{"seed","score"}` records
(`:281-286`), **discarding the provenance the canonical rule needs.**

**Import `_select_l2_winners` / `_l2_sort_key`.** They are private and not in `__all__` — import
them anyway, or extract to a shared module and have **both** call sites use it. **Never fork.**

---

## 6. Finalizer parity and pre-clear validation (Beta correction 5)

### 6.1 What parity actually means

A checkpoint of canonical winners **cannot** reconstruct the finalizer's original raw candidate
list: losers are discarded by construction. REV2's *"field-for-field identical"* claim is
withdrawn. `raw_candidate_count`, `generation_id`, creation time, elapsed time and sidecar hash
differ **necessarily**.

**Parity is defined as:**
- **exact equality of all 22 canonical arrays**;
- **identical global seed order**;
- **identical canonical NPZ/artifact digest**;
- **truthful resumed-run provenance.**

**`raw_candidate_count` — Alpha's resolution: it is run-scoped and NOT preserved.** A resumed run
truthfully reports the candidates *it* observed. This requires no counter, no replay arithmetic and
no double-count logic, and **removes the double-counting hazard by construction rather than
managing it.** **Sidecar-field equality is therefore NOT claimed.** State this in the report.

### 6.2 Three protections before any clear

Every newly observed raw record must pass the same three walls `finalize_run` applies before L2
(`utils/run_finalizer.py:1606-1611`, in this order):

1. `_validate_raw_candidates` (`:665`) — strict 24-field validation
2. `_validate_candidate_coverage` (`:558`) — declared seed-coverage
3. `_validate_candidate_identity` (`:634`) — run-identity wall

**Reuse or extract; do not duplicate.** All three are private; the same import-or-extract rule as
§5.2 applies.

`_validate_raw_candidates`'s own docstring already states the invariant Beta's correction-6 mutant
tests: *a malformed **losing** candidate must fail the run, not vanish during selection*. **That is
existing precedent, not a new demand.**

---

## 7. Enabling the clear

Only after §2-§6 hold. Beta's ordering, verbatim:

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

A mutant clearing **between the two replaces** must fail. D6.2 turns `_FLUSH_CLEAR_IN_MEMORY` on
and must prove the finalizer still receives complete 24-field input **via the resume path, not the
truncated stump.**

---

## 8. Gates — `tests/test_s172_d6_2_checkpoint_reconciliation.py`

**Duplicate matrix** (each row states which path it exercises — §2.5):

| case | required behaviour |
|---|---|
| bit-identical 24-field replay | collapsed by normalization; idempotent |
| same `(seed, trial, mode)`, any canonical field differs | **raises `AccumulatorConsistencyError`** |
| different match rates | canonical winner, float32 domain |
| float64-only difference, **different trial numbers** | float32 tie → **lower trial_number wins** |
| different trial_number, same score | lower trial_number wins |
| different provenance/mode metadata | canonical rule decides |
| restart-replay duplicate | idempotent, no double-count |

**Schema (Beta correction 6):**
- **G-STORAGE-DOMAIN:** round-trip equality means equality **in the declared storage domains**;
  include a **deliberately non-float32-representable input** proving the expected canonicalization.
- **G-CSR-STRICT:** `int64`, one-dimensional offsets, length `records + 1`, first offset zero,
  monotonic, final offset `== len(sessions_values)`, no out-of-range slices.
- **G-SESSIONS-CASES:** `[]`, `[""]`, ordered multi-session, non-ASCII; **all-empty proving
  `sessions_values` stays a Unicode array and does not default to float64.**
- **G-ENCODING-AUTHORITY:** AST — no literal skip_mode map, no literal prng_type map, no
  transcribed dtype table anywhere in the checkpoint path.
- **G-IDENTITY-BIND:** `encoding_version` **and** `canonical_map_hash` present; mismatch on either
  **fails before decoding**.
- **G-CONTEXT-BIND:** `run_context_digest` present; a differing context is **rejected before
  decoding**; **no newest-directory inference exists anywhere** (AST + runtime).
- **G-SEQUENCE-INIT:** next sequence starts above the highest accepted persisted sequence.
- **G-STUB-HONESTY** (§4.2): member A documented as a stub; **no path claims recovery from it.**
- **G-RESTART-{A,B,C,D}:** the four outcomes, including **fail-closed without clearing.**
- **G-MIXED-PAIR:** higher valid sequence chosen **only** on full identity agreement; equal
  sequence with differing `checkpoint_id` **fails closed**.
- **G-PRE-CLEAR-WALLS:** all three §6.2 protections run before any clear, in order, reused.
- **G-PARITY:** §6.1's four properties. **Not** field-for-field generation equality.
- **G-CLEAR-SAFE · G-CADENCE · G-COMPRESSION-CONTRACT · G-NO-SYMLINK-COLLISION** — as REV2.

**Mutants** (four-part kill rule; prove each red comes **from its injected defect**, and swap the
source every gate builds from — the D6.1 vacuous-mutant lesson): inline `score >` policy;
float64 comparison; drop the trial_number tiebreak; swallow the same-trial/same-mode collision;
clear between the two replaces; drop a transaction-identity field; write to a finalizer-owned
path; hardcode the skip_mode map; **changed encoding-map hash**; **incorrect mixed-pair freshness
selection**; **a malformed losing candidate disappearing during checkpoint compaction**; a
newest-directory inference reintroduced.

---

## 9. Non-regression

Beta confirmed **no Wall A/B rerun is required.** Capture green before any edit: **D3.25 (13/13),
D3.5 (60/60), D6.1, Phase 3 (17/17), Phase 4 (63/63)**, plus D1.1 · D1.0 · D0 · D2 · D3.0 · D3 ·
D4 · D5 · D6 3.A · D6-threshold. After: all green plus D6.2.

All commands on **VM101**, `source ~/venvs/torch/bin/activate` first — a bare shell yields false
`CuPy not available` / `Optuna not available` reds.

---

## 10. Scope — do NOT touch

D6 threshold/provenance/residue work; PWC/ZMQ ingress; the D3.25 four-map contract; `TestResult`
shape; D5's artifact contract; `serial_reference` as default. Do not modify `_l2_sort_key`,
`_select_l2_winners`, `CANONICAL_ARRAY_CONTRACT`, `utils/prng_encoding`, `canonical_map_hash`, or
the three §6.2 validators — they are **reused**, not revised.

---

## 11. Report

The §2.2 table as verified at HEAD (report drift); the `sessions` encoding and its `[]` round-trip;
`prng_base` derivation for both modes; the normalization-then-`_select_l2_winners` call path,
stated as replay normalization and not a second policy; which path each duplicate row exercises;
the resume handle and `run_context_digest` composition; the four restart outcomes and the
mixed-pair rules; **the §6.1 parity properties actually demonstrated, and the explicit statement
that sidecar-field equality is not claimed**; confirmation the execution cursor is not claimed;
gate/mutant counts; D3.25 and D3.5 unchanged. Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** each gate prints its name and a non-trivial assertion count; the parity gate
  reports compared artifact digests, not a boolean.
- **clean control:** an uninterrupted reference run passes every restart gate's healthy branch.
- **fault-injection control:** §8's mutant list, four-part kill rule on each.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only `PASS` accepts.
- **unavailable-observer behavior:** D6.2 should carry no fleet dependency; if one appears, report
  it as a finding rather than working around it. With rigs down, a fleet-dependent arm is
  `UNAVAILABLE`, never `PASS`.
- **audit claim scope:** repo-scoped, `9470750`.
- **searched surfaces:** tracked repo at `9470750`.
- **unavailable surfaces:** host state on VM101 and the rigs; uncommitted local modifications; the
  live `KERNEL_REGISTRY` contents if changed since the clone.
