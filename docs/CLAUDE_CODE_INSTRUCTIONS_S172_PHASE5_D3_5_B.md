# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5_B.md — REV2

**REV2 changelog** — absorbs the Team Beta pre-deployment review (two required
corrections, two gate clarifications): **[R1]** the recursive chain walk does
**NOT** currently compare `prng_base` or any semantic contract field — Team
Alpha asserted it did without verifying; the per-link contract is now specified
explicitly; **[R2]** "append" replaced with **insert in global alphabetical
order**, with the exact 32-key tuple given; **[R3]** S6 extended to all **nine**
seed-domain fields; **[R4]** S7 defined mechanically as a cross-domain lineage
mismatch. Plus strict type domains for the new fields (bool rejected) and the
exact new version string.

**S172 RANGE-MINER — Phase 5, Deliverable D3.5-B: Seed-Domain v1.1 — honest
stratum labelling in the provenance sidecar**

**Audience:** Claude Code on VM 101 (`michael@192.168.3.177`). You write and
iterate; you do NOT commit, push, or run WATCHER. When gates + non-regression
are green, STOP and report.

**Frozen against HEAD `46a3828`.** Authority: Team Beta Ruling G (corrected) and
the Seed-Domain v1.1 placement ruling — **D3.5-B lands before D4/D5/D6** so all
three target the final sidecar contract from the outset.

---

## 0. Why this exists

The `java_lcg` registry family has a **48-bit** internal state. The canonical
artifact stores `seeds: uint32`. The current sweep therefore covers the
`high16 = 0` stratum — **1 part in 65,536** of the state space — and the upper
16 bits are *not* invisible: they are blind to the mod-8 lane but fully visible
to mod-125, and at TFM's window (21) all 65,536 high-state classes produce
distinct draw sequences. No reduction exists.

**This is a labelling problem, not a storage problem.** TFM does functional
mimicry, not state reversal: the sweep exists to discover bidirectional
survivors whose structure feeds the ML ensemble, and survivor validity rests on
sieve selectivity rather than search extent. So the artifact stays `uint32` and
declares honestly which stratum it is.

**Scope is deliberately tiny.** D3.5-B adds nine sidecar fields, bumps one
version constant, and gates them. Nothing else.

## 1. Non-negotiable working rules

1. **Read live source before every claim.** Cites verified at `46a3828`.
2. **Each gate must FAIL on wrong behavior.**
3. **Independent oracle:** the harness hand-transcribes the new keys **and their
   required values**. Do not import `SIDECAR_REQUIRED_KEYS` or any seed-domain
   constant and assert against it (G9 / E8 / C1 lesson).
4. STOP at the gate. No commit/push/WATCHER.

## 2. Scope

**Modify — only these:**

```text
utils/run_finalizer.py
tests/test_s172_phase5_d3_5_finalizer.py
```

Plus the D3.5-B brief and session changelog.

**Must NOT modify:** any backend, coordinator, miner, canonical-array,
encoding, integration or database module; `utils/canonical_arrays.py`;
`utils/canonical_records.py`; `utils/prng_encoding.py`;
`window_optimizer_integration_final.py`; `miner/*`; `prng_analysis.db`;
WATCHER; or any D0-D3.5 test other than the D3.5 harness itself. Discovering a
required change triggers **STOP**.

## 3. The nine new sidecar fields

**Insert** the nine names into `SIDECAR_REQUIRED_KEYS`
(`utils/run_finalizer.py:145-168`) in **exact global alphabetical order** — they
cannot all be appended at the end while remaining sorted [R2]. 23 keys → **32**.
The resulting tuple, in order:

```text
artifact_schema_version      parent_sidecar_sha256        seed_domain_start          # NEW
artifact_sha256              prior_row_count              seed_effective_bits        # NEW
canonical_map_hash           prng_base                    seed_end_exclusive
created_at                   raw_candidate_count          seed_high16_prefix         # NEW
encoding_contract_version    repository_commit            seed_semantics             # NEW
exhaustive_over        # NEW repository_tree_clean        seed_start
external_seed_transform# NEW row_count                    seed_storage_dtype         # NEW
final_row_count              run_id                       sidecar_schema_version
generation_id                seed_count                   skip_modes_executed
l2_winner_count              seed_domain_contract   # NEW
parent_artifact_sha256       seed_domain_end_exclusive  # NEW
parent_generation_id
```

S1 asserts this exact 32-name tuple **in this order**, hand-transcribed.

The nine new fields and their frozen values:

```text
seed_semantics            = "internal_state"
seed_storage_dtype        = "uint32"
seed_effective_bits       = 32
seed_high16_prefix        = 0
seed_domain_contract      = "v1.1-stratum"
seed_domain_start         = 0
seed_domain_end_exclusive = 4294967296
exhaustive_over           = "high16=0 stratum only"
external_seed_transform   = null
```

These distinguish three concepts the artifact previously conflated:

```text
canonical PRNG coordinate : 48-bit internal state
stored artifact coordinate: uint32 low-state component
certified search stratum  : high16 = 0
```

**Every one is a fixed constant in v1.1** — none is caller-supplied. A generation
whose sidecar carries any other value for any of them **fails closed**.
`seed_domain_start` / `seed_domain_end_exclusive` describe the **stratum
domain** `[0, 2^32)`, which is distinct from the existing per-run
`seed_start` / `seed_count` / `seed_end_exclusive` coverage fields — both sets
coexist and neither replaces the other.

## 4. Version bump — exactly one

```text
BUMP:        SIDECAR_SCHEMA_VERSION   (:139)
             "s172.d3_5.provenance.v1" -> "s172.d3_5.provenance.v1.1"
DO NOT BUMP: ARTIFACT_SCHEMA_VERSION  (:138) — arrays, key order, dtypes unchanged
DO NOT BUMP: ENCODING_CONTRACT_VERSION(:140) — encoding maps unchanged
```

**No migration compatibility is required** — no certified production generation
exists (Ruling F: clean start). Therefore:

> A pre-v1.1 sidecar **fails closed**. It must never be silently interpreted as
> `high16 = 0`.

**No compatibility reader for the old 23-key sidecar is authorized.** This is a
clean contract replacement, not a migration.

## 5. Lineage semantics

A v1.1 generation may join a certified lineage only if child and parent agree on
**all seven** stratum-identifying fields:

```text
seed_domain_contract      seed_semantics            seed_storage_dtype
seed_effective_bits       seed_high16_prefix        seed_domain_start
seed_domain_end_exclusive
```

Any mismatch **fails closed**, checked **at every link** in the existing
recursive parent-chain walk.

**[R1] Correction — this check does not exist today.** Verified at `46a3828`:
`_validate_chain` (`utils/run_finalizer.py:1051-1128`) contains **zero**
occurrences of `prng_base`, `schema_version`, `canonical_map_hash` or any
`seed_` field. It validates only complete parent references, cycles, repeated
IDs, ancestor directory existence, sidecar hash, generation ID and artifact
hash. `prng_base` is compared **only on the selected tip**, in
`_load_prior_generation` (`:1212`). D3.5-B must **add** link-by-link semantic
equality — without changing chain topology or the publication mechanism.

**Per-link contract — verify equality of all fourteen:**

```text
prng_base                    seed_domain_contract      seed_domain_start
artifact_schema_version      seed_semantics            seed_domain_end_exclusive
sidecar_schema_version       seed_storage_dtype        exhaustive_over
encoding_contract_version    seed_effective_bits       external_seed_transform
canonical_map_hash           seed_high16_prefix
```

The first five are already required properties of a homogeneous certified
lineage; they should not remain unchecked merely because D3.5-B exposed the
seam. This is a comparison loop inside `_validate_chain`, not a restructure.

A future v2 lineage begins from a clean root (`parent_generation_id`,
`parent_artifact_sha256`, `parent_sidecar_sha256` all `null`). **No v1.1
generation may ever be linked as a certified v2 parent.** A v1.1 artifact may
later serve as external research input or comparative training data — never as
a v2 provenance ancestor.

## 5a. Constant construction and strict validation

**The nine values are module-owned constants inserted into the payload
internally.** They must NOT be: added to `finalize_run`'s signature; read from
the environment; inferred from the candidate maximum; copied from a supplied
prior; or accepted from any backend or coordinator. This is what prevents a run
publishing a sidecar claiming a stratum other than the one the `uint32` domain
wall actually enforced.

Update `_validate_sidecar_payload` so the new fields get **both** strict type
and exact-value validation:

```text
str                  : seed_semantics, seed_storage_dtype,
                       seed_domain_contract, exhaustive_over
int, bool REJECTED   : seed_effective_bits, seed_high16_prefix,
                       seed_domain_start, seed_domain_end_exclusive
null only            : external_seed_transform
```

Use the module's existing strict-integer helper for the four integer fields —
Python's `bool` is a subclass of `int`, so a bare `isinstance(x, int)` would
accept `True` as `seed_high16_prefix`.

## 6. Gate — extend `tests/test_s172_phase5_d3_5_finalizer.py`

Independent hand-transcribed oracle for the new key set **and values**. Add:

- **S1** the sidecar's required key set is exactly the 32 hand-transcribed keys
  (assert the tuple, not a subset).
- **S2** a published v1.1 generation carries every one of the nine fields with
  exactly the §3 values.
- **S3 fail-closed matrix** — each raises, one case per field: a **missing**
  seed-domain field; `seed_semantics` != `"internal_state"`;
  `seed_storage_dtype` != `"uint32"`; `seed_effective_bits` != 32;
  `seed_high16_prefix` != 0; `seed_domain_start` != 0;
  `seed_domain_end_exclusive` != 2**32; `exhaustive_over` mislabelled;
  `external_seed_transform` non-null.
- **S4** an **old sidecar schema version** fails closed — never silently
  interpreted as `high16 = 0`.
- **S5** a valid v1.1 generation publishes and recursively validates, including
  a two-generation chain where child and parent agree on all seven stratum
  fields.
- **S6** a child whose parent disagrees on **any of the nine** seed-domain
  fields fails closed — parameterize across all nine [R3]. (The seven-field
  coordinate identity remains useful documentation, but a certified lineage must
  agree on all nine fixed v1.1 values, including `exhaustive_over` and
  `external_seed_transform`.) Additionally parameterize the five
  already-required lineage fields from §5's per-link contract — `prng_base`,
  the two schema versions, the encoding contract and `canonical_map_hash` —
  since none is currently compared per link.
- **S7 cross-domain lineage mismatch [R4]** — construct a synthetic
  two-generation chain whose **child claims a different
  `seed_domain_contract` / storage domain** from its v1.1 parent, while
  retaining otherwise structurally valid hashes and references. The chain must
  fail **specifically on seed-domain lineage incompatibility** — not merely
  because a hash, key or JSON type is malformed; assert the failure names the
  seed-domain field. This establishes the invariant a future v2 must retain:
  *a different seed-domain contract requires a new clean root.*
  **Do not introduce a v2 writer or v2 sidecar schema in D3.5-B.**
- **S8** `ARTIFACT_SCHEMA_VERSION` and `ENCODING_CONTRACT_VERSION` are
  **unchanged** from `46a3828` (assert the literal strings), and the 22-array
  contract — names, order, dtypes — is untouched.
- **S9 mutation proof** — kill each of: a seed-domain field made caller-supplied
  instead of constant; `seed_high16_prefix` defaulting to 0 when absent rather
  than failing; a pre-v1.1 sidecar accepted; the lineage check omitting one of
  the nine fields; the per-link check omitting `prng_base`;
  `ARTIFACT_SCHEMA_VERSION` bumped unnecessarily; `True` accepted for
  `seed_high16_prefix` (bool-as-int); **and at least one value promoted to a
  `finalize_run` caller argument — the public API and the gate must both reject
  it.** Report each
  red signature and its attribution.

**Blocking non-regression:** D3.5 51/51 **plus the new S-checks**, D3.25 13/13,
D3 10/10, D3.0 10/10, D2 7/7, D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4 63/63,
Phase 3 17/17. Capture the baseline green at `46a3828` **before** any edit.

## 7. Stop conditions

- the fields cannot be added without touching a must-not-modify module;
- the existing recursive chain validation cannot carry the seven-field stratum
  check without restructuring (report; do not restructure);
- any gate passes only by weakening it;
- you find yourself changing array contents, dtypes, key order, encoding maps or
  publication mechanics — none of that is in scope.

## 8. Report

Diff + status, full command/output evidence, the pre-edit baseline, mutation
evidence with per-mutant red signatures and attribution, and explicit
confirmation that `ARTIFACT_SCHEMA_VERSION` and `ENCODING_CONTRACT_VERSION` are
unchanged. Then STOP for Team Alpha review.
