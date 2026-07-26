# SESSION_CHANGELOG_20260725_S179 — S172 Phase 5 D3.5-B (Seed-Domain v1.1)

**Scope:** implement `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5_B.md` (REV2,
Team Beta approved) against HEAD `46a3828`.
**Status:** implemented and gated on VM 101. **NOT committed, NOT pushed, WATCHER
not run** — stopped at the gate for Team Alpha review, per §1.4 of the brief.

---

## 1. Files modified (exactly the two authorized by §2)

```text
utils/run_finalizer.py                        +165 / -14
tests/test_s172_phase5_d3_5_finalizer.py      +682 / -8
```

Plus this changelog. No backend, coordinator, miner, canonical-array, encoding,
integration or database module was touched; `utils/canonical_arrays.py`,
`utils/canonical_records.py`, `utils/prng_encoding.py`,
`window_optimizer_integration_final.py`, `miner/*`, `prng_analysis.db` and
WATCHER are all unchanged, and no D0-D3.5 harness other than the D3.5 harness
itself was edited. No stop condition (§7) was hit.

## 2. What landed

### 2.1 Nine module-owned seed-domain constants (§3, §5a)

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

Inserted into the sidecar payload **internally**, from module constants. None is
in `finalize_run`'s signature, read from the environment, inferred from the
candidate maximum or copied from a supplied prior. The stratum-domain fields
coexist with — and do not replace — the per-run `seed_start` / `seed_count` /
`seed_end_exclusive` coverage fields.

`SIDECAR_REQUIRED_KEYS` went 23 → **32**, inserted in global alphabetical order
[R2], not appended.

### 2.2 Version bump — exactly one (§4)

| Constant | 46a3828 | now |
|---|---|---|
| `SIDECAR_SCHEMA_VERSION` | `s172.d3_5.provenance.v1` | **`s172.d3_5.provenance.v1.1`** |
| `ARTIFACT_SCHEMA_VERSION` | `s172.d3.arrays.v1` | `s172.d3.arrays.v1` (unchanged) |
| `ENCODING_CONTRACT_VERSION` | `s172.phase0.encoding.v1` | `s172.phase0.encoding.v1` (unchanged) |

No compatibility reader for the 23-key sidecar exists. A pre-v1.1 sidecar fails
closed and is never interpreted as `high16 = 0`.

### 2.3 Strict validation in `_validate_sidecar_payload` (§5a)

Type pass **before** value pass, deliberately: `False == 0` in Python, so a
Boolean `seed_high16_prefix` or `seed_domain_start` would satisfy the `== 0`
value pin and only the module's bool-rejecting `_require_int` can refuse it.

### 2.4 The fourteen-field per-link lineage check (§5, [R1])

**Verified against `46a3828` before writing a line:** `_validate_chain`
(`:1051-1128`) compared **no** semantic field — not `prng_base`, not the schema
versions, not `canonical_map_hash`, not any `seed_*` field. It checked parent
completeness, cycles, repeated ids, ancestor existence, the ancestor sidecar
hash, `generation_id` and the artifact hash only. `prng_base` was compared
solely on the selected tip, in `_load_prior_generation` (`:1212`).

D3.5-B **adds** `_LINEAGE_INVARIANT_KEYS` and a comparison loop inside the
existing walk. Chain topology and the publication mechanism are unchanged.
The five contract fields (`prng_base`, both schema versions, the encoding
contract, `canonical_map_hash`) were already required properties of a
homogeneous lineage and are no longer left unchecked merely because D3.5-B
exposed the seam.

## 3. Gates

`tests/test_s172_phase5_d3_5_finalizer.py` grew from 51 to **60** checks:
F1-F51 unchanged plus **S1-S9**. Independent hand-transcribed oracles
throughout — the 32-key tuple and all nine values are literals read from the
brief; `SIDECAR_REQUIRED_KEYS` and the seed-domain constants are never imported
and asserted against themselves (G9 / E8 / C1).

S9 runs eight mutants, all red. Two carry a deliberate two-anchor construction
(the F18 precedent) because the seed-domain fields are guarded by two redundant
layers; the `prng_base` per-link mutant is single-anchor and uniquely
attributable, since an **ancestor's** `prng_base` is compared nowhere else.

## 4. Non-regression

Baseline captured at `46a3828` **before any edit**, and re-run after:

| Suite | Baseline | After |
|---|---|---|
| D3.5 | 51/51 | **60/60** (51 + S1-S9) |
| D3.25 | 13/13 | 13/13 |
| D3 | 10/10 | 10/10 |
| D3.0 | 10/10 | 10/10 |
| D2 | 7/7 | 7/7 |
| D1.1 | 18/18 | 18/18 |
| D1.0 | 8/8 | 8/8 |
| D0 | 12/12 | 12/12 |
| Phase 4 | 63/63 | 63/63 |
| Phase 3 | 17/17 | 17/17 |

fallback parity: code=[not re-checked this session], env=[not re-checked this
session] — `.127` is not booted (Zeus is running Proxmox / VM 101).

## 5. Next

Team Alpha review, then Team Beta. D4/D5/D6 now target the final sidecar
contract from the outset, which is why D3.5-B landed ahead of them.
