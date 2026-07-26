# TEAM_ALPHA_REVIEW_S172_PHASE5_D3_5_B.md

**Subject:** Team Alpha review of the D3.5-B implementation (Seed-Domain v1.1)
**Spec:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5_B.md` REV2
**Base:** HEAD `46a3828`
**Verdict: APPROVED — recommend Team Beta review for commit. No correction round
required. One documentation rename before commit (§6).**

---

## 1. Scope

Exactly the two authorized files:

```text
M utils/run_finalizer.py                        (+165/-14)
M tests/test_s172_phase5_d3_5_finalizer.py      (+682/-8)
```

No backend, coordinator, miner, canonical-array, encoding, integration or
database module touched. Nothing in the must-not-modify list appears in the diff.

## 2. [R1] independently re-verified, and the correction is faithfully applied

Team Alpha confirmed at `46a3828` that `_validate_chain` (`:1051-1128`) contained
**zero** occurrences of `prng_base`, `schema_version`, `canonical_map_hash` or
any `seed_` field — hashes, IDs, cycles and existence only, with `prng_base`
compared solely on the tip in `_load_prior_generation` (`:1212`). Claude Code
verified the same independently before writing code. Three passes, one
conclusion.

`_LINEAGE_INVARIANT_KEYS` is built as the five pre-existing fields **plus**
`tuple(name for name, _ in SEED_DOMAIN_FIELDS)` = **14**, and its comment records
the finding accurately: the first five "were already required properties of a
homogeneous lineage but were never compared per link… They are not left
unchecked merely because D3.5-B exposed the seam."

That is the right disposition. D3.5-B closes a gap that predates it.

## 3. Implementation verified point by point

**32-key ordering — matches Team Alpha's independent computation exactly.** The
`seed_*` fields interleave correctly with the existing coverage keys
(`seed_count`, then the four `seed_domain*`/`seed_effective_bits`, then
`seed_end_exclusive`, `seed_high16_prefix`, `seed_semantics`, `seed_start`,
`seed_storage_dtype`), and `exhaustive_over` / `external_seed_transform` sit
beside `encoding_contract_version`. 23 → 32.

**Per-link check is a comparison loop, not a restructure.** A `for key in
_LINEAGE_INVARIANT_KEYS` loop inside the existing walk, raising
`PriorGenerationError` naming the specific divergent key. Chain topology and the
publication mechanism are untouched.

**Value pin runs type-before-value, with the reason documented inline:**

```text
1. _require_str  on the four string fields
2. _require_int  on the four integer fields   <- bool-rejecting
3. null check    on external_seed_transform
4. exact value equality against SEED_DOMAIN_FIELDS
```

The inline comment states why the order matters — `False == 0` in Python, so a
Boolean `seed_high16_prefix` would satisfy the value pin and only a
bool-rejecting integer guard can catch it. Correct, and it is the reasoning
behind judgment call 2 (§4).

**The nine values are genuinely un-parameterized.** AST-verified:
`finalize_run`'s parameter list is
`candidates, output_root, run_id, prng_base, skip_modes_executed, seed_start,
seed_count, repository_commit, repository_tree_clean, prior_generation_dir` —
**no seed-domain parameter of any kind.** All nine are module-level constants
(`:170-178`) injected internally, with a comment recording that they describe
the **stratum domain** and coexist with the per-run coverage fields rather than
replacing them.

**Version constants:** `SIDECAR_SCHEMA_VERSION` → `"s172.d3_5.provenance.v1.1"`;
`ARTIFACT_SCHEMA_VERSION` and `ENCODING_CONTRACT_VERSION` unchanged, machine-diffed.

## 4. The three judgment calls — all endorsed, one resolved by measurement

**Call 1 — double-guarded seed-domain layer. Raised honestly; Team Alpha
resolved it.** The concern: because `_validate_sidecar_payload` pins all nine to
constants on *every* sidecar (tip and ancestor), a mutant on either the pin or
the per-link loop can survive individually, so the nine cannot isolate the
per-link layer.

That is true, and the asymmetry has a clean explanation: the nine are pinned to
constants, but the five pre-existing fields are **not** — `prng_base` may differ
across *independent accumulator lineages*, but it must remain identical
throughout **one** certified lineage — so for those five the per-link loop is
the *only* guard.
Claude Code identified `prng_base` as the single-anchor, uniquely-attributable
mutant on that basis.

**Team Alpha tested the stronger case it did not: deleting the per-link loop
entirely.** Result: **killed by S6 alone, 59/60.** So the loop's existence is
uniquely gated, and the double-guard on the nine is redundancy rather than a
coverage hole. Concern closed.

**Call 2 — `False` not `True`.** Correct, and it fixes an error in Team Alpha's
brief. `True != 0`, so the value pin rejects it and the mutant would have
survived, proving nothing about `_require_int`. `False == 0` passes the pin
cleanly, so only the bool-rejecting helper can refuse it. A strengthening of the
specified item, not a weakening.

**Call 3 — S7 split.** Follows from the value pin iterating in declaration
order: a full v2 claim trips `seed_storage_dtype` first and never names the
contract field. The added contract-only case makes the rejection attributable to
`seed_domain_contract` specifically. Both sub-cases assert the failure is not a
hash, key or JSON malformation, per REV2 [R4].

## 5. Mechanical verification (Team Alpha sandbox, pristine `46a3828`)

```text
D3.5   60/60 green   (51 F-gates + S1-S9)   -- independently reproduced
```

Claude Code additionally reported the pre-edit baseline green at `46a3828` and
all ten blocking suites unchanged after the edit (D3.25 13/13, D3 10/10, D3.0
10/10, D2 7/7, D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4 63/63, Phase 3 17/17),
with 8/8 S9 mutants red and attributed.

Team Alpha independent mutant: **per-link loop deleted entirely → killed by S6
alone (59/60).** Surgical attribution.

Note for the record: Phase 4 held at 63/63 because two stray `.py` files under
`tmp/d2_evidence/` (leftover D2 mutation scratch) were removed during the run —
gate 22 reds on any stray untracked `.py`. That is `PHASE6_PREREQS` item 5
materializing in practice, and it confirms the item is correctly specified.

## 6. Before commit — one rename

```text
docs/SESSION_CHANGELOG_20260725_S179.md
  -> docs/SESSION_CHANGELOG_20260725_PHASE5_D3_5_B.md
```

D3.5's changelog was renamed at Team Beta's direction specifically to avoid
binding a permanent record to a transient session number; `S179` reintroduces
that. Claude Code flagged the guess itself.

## 7. Recommendation

Approve for commit. Suggested scope: the two modified files, the D3.5-B brief,
this memo, the renamed changelog, and **`docs/PHASE6_PREREQS.md`** — Team
Beta-approved and pending commit since D3.5.

— Team Alpha (Claude), 2026-07-26
