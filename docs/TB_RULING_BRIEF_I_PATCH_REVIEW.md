# TB RULING — BRIEF I BOUNDED REPAIR: PATCH APPROVED, COMMIT AUTHORIZED

**Received:** 2026-08-22
**Responds to:** `docs/S172_BRIEF_I_PATCH_REPORT.md` (the bounded repair against `48a8705`)
**Authority it acts under:** `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md` §3, §4
**Effect:** Michael may commit and dual-push **this exact reviewed repair**.
**Recorded by:** Team Alpha, verbatim below.

**Binding dispositions:**

| item | disposition |
|---|---|
| Second `canonical_records.py` consumer (`normalize_trial_populations`) | **RATIFIED IN-SCOPE.** Same defect class on the same authorized production-record surface — *not* unrelated scope growth. The authorization was never "change exactly line 217" |
| **the governing invariant, now stated** | *Every post-F-4 producer of canonical record field `"offset"` on these migrated paths sources it from `window_anchor`, never from a retired context `offset` and never from `generator_phase`.* |
| Array-4 semantics implementation | **ACCEPTED.** `"offset"` is a legacy wire name; its post-F-4 meaning is `window_anchor`, **at any generator phase** |
| removal of Alpha's "only coherent while phase=0" rationale | **Correctly removed, with a visible correction trail** rather than silently rewritten history |
| frozen contract | **Still closed** — 22 arrays · index 4 still `"offset"` · no `"window_anchor"` array · no `"generator_phase"` array · phase remains independent metadata |
| `DEP-ABI-V2-NPZ-SEMANTICS` | **Correctly recorded as an audit dependency, NOT a pre-authorized array-contract change** |
| `G-PHASE5-ASSEMBLY` | **APPROVED** — the missing outer certificate. The unequal `58`/`0` pair prevents a false green in which phase is accidentally written into array 4; both mutants give the right non-vacuity |
| `G-PHASE5-SEAM` | **Remains a separate certificate and remains unwidened** — the correct architecture: local seam proof **plus** end-to-end assembly proof, rather than redefining a certified gate after the fact |
| Two certified-suite fixture migrations | **RATIFIED** under the Field-6 principle: production contract evolution may require historical fixtures to be translated so they keep reaching the property they were designed to test. Protections present: **zero assertion additions/removals/changes**, production semantics not weakened, legacy expected-record `"offset"` retained where it legitimately names the canonical wire field, and **discrimination carried by `G-PHASE5-ASSEMBLY` rather than falsely claimed by equal-valued fixtures** |
| `d1_engine` | **Classification accepted** (18 failures before, 18 after — no new suite-level red), **but §5's wording is binding**, below. **No RC-1 repair authorized here** |
| Nested-tally leak | **FINDING ACCEPTED.** The `ac7_final/SUMMARY.tsv` `d1_engine` row is contaminated and **not admissible baseline evidence**. Other rows need not be re-audited to commit this repair |
| Focused regression battery | **ACCEPTED** as appropriate for the repair |
| Further code changes before commit | **NONE REQUESTED** |

**⚑ §5 — BINDING WORDING for `d1_engine`. Use it; do not paraphrase:**

> **same pre-existing red population / changed failure depth / no demonstrated new production regression**

Explicitly **not** "identical behavior" and **not** "zero differential" — because some failures moved
from the mandatory-context guard into the already-known RC-1 path.

**⚑ §6 — STANDING RULE on summary-extracted tallies:**

> A summary-extracted suite tally is **not authoritative** when the underlying suite emits
> nested/subprocess tallies, unless the extractor binds the tally to that suite's own completion
> sentinel or otherwise proves provenance.

No general summary-harness rewrite is authorized in this patch. Any future governance claim relying
on a suspicious `ac7_final` row must use the original per-suite log or a re-derived provenance-safe
tally rather than the contaminated TSV.

**After commit, the acceptance step is unchanged:** deploy/parity as required, then a **fresh
Michael-authorized production-shaped trial from the repair commit**. It must reach successful
Phase-5 assembly/publication before B5/B6/B8 can be reclassified, B7 can receive its missing
acceptance classification, and **Brief I can finally close.**

---

## Ruling body (verbatim)

Reproduced character for character as received.

## TEAM BETA REVIEW — BRIEF I BOUNDED REPAIR

**VERDICT: PATCH APPROVED. SCOPE EXPANSION RATIFIED. FIXTURE MIGRATIONS RATIFIED. COMMIT AUTHORIZED.**

The repair implements the array-4 ruling correctly and closes the exact production defect exposed by `distributed_config_t1_eed23c7f`. The second consumer is the same defect class on the same authorized production-record surface, not unrelated scope growth. 

### 1. Second `canonical_records.py` consumer — RATIFIED AS IN-SCOPE

`normalize_trial_populations` at the second site belongs in this bounded repair.

The authorization was not “change exactly line 217.” It was to repair the production canonical-record path so that legacy wire field `"offset"` sources **window anchor only**. The regression sweep established that there were two live producers of that canonical field:

* miner Phase-5 assembly;
* PWC/ZMQ normalization.

Repairing only the first would have left the same semantic defect live on another production route. 

So the governing invariant is now:

> Every post-F-4 producer of canonical record field `"offset"` on these migrated paths sources it from `window_anchor`, never from a retired context `offset` and never from `generator_phase`.

The second two-line production correction is therefore **ratified, not treated as a new design change**.

### 2. Array-4 semantics — implementation ACCEPTED

The code now reflects the previous Beta ruling correctly:

**`"offset"` is a legacy wire name. Its post-F-4 meaning is `window_anchor`, at any generator phase.**

The report also correctly removes Alpha's earlier “only coherent while phase=0” rationale while retaining a visible correction trail rather than silently rewriting history. 

The frozen contract remains closed:

* still 22 arrays;
* index 4 remains named `"offset"`;
* no `"window_anchor"` array;
* no `"generator_phase"` array;
* generator phase remains independent metadata.

**DEP-ABI-V2-NPZ-SEMANTICS is correctly recorded as an audit dependency, not as a pre-authorized array-contract change.**

### 3. `G-PHASE5-ASSEMBLY` — APPROVED

This is exactly the missing outer certificate.

The critical test shape is sound:

`window_anchor=58`, `generator_phase=0` → canonical `"offset"=58`

That unequal pair prevents a false green in which phase is accidentally written into array 4. The two mutants provide the right non-vacuity:

* restoring `ctx["offset"]` recreates the production failure;
* sourcing from `generator_phase` produces `0` and is detected. 

The original **G-PHASE5-SEAM remains a separate certificate and remains unwidened**. That is the correct architecture: local seam proof plus end-to-end assembly proof, rather than changing the meaning of a previously certified gate after the fact.

### 4. Certified-suite fixture migrations — RATIFIED

Both migrations are accepted under the same principle used for Field 6: production contract evolution may require historical fixtures to be translated so they continue reaching the property they were designed to test.

The important protections are present:

* **zero existing assertion additions/removals/changes** in both affected suites;
* production semantics are not weakened to accommodate the tests;
* the legacy expected-record `"offset"` is retained where it legitimately refers to the canonical wire field;
* discrimination between anchor and phase is deliberately carried by `G-PHASE5-ASSEMBLY`, not falsely claimed by fixtures where both values happen to be equal. 

The fixture changes are therefore **authorized parts of this repair**.

### 5. `d1_engine` — classification accepted, with the same caution as before

The worktree comparison establishes the relevant top-level fact:

**18 failures before repair, 18 failures after repair.**

That is sufficient to show this patch did not create a new suite-level red.

But because some failures moved from the mandatory-context guard into the already-known RC-1 path, describe it as:

> **same pre-existing red population / changed failure depth / no demonstrated new production regression**

—not simply “identical behavior.”

No RC-1 repair is authorized here. 

### 6. Nested-tally leak — FINDING ACCEPTED

The `ac7_final/SUMMARY.tsv` `d1_engine` row is demonstrably contaminated by a neighbouring suite's tally. It must not be treated as valid baseline evidence. 

This does **not** invalidate the repair review because the relevant suites here were rerun directly and their actual outcomes measured.

Going forward:

> **A summary-extracted suite tally is not authoritative when the underlying suite emits nested/subprocess tallies unless the extractor binds the tally to that suite's own completion sentinel or otherwise proves provenance.**

No general summary-harness rewrite is authorized in this patch. However, any future governance claim relying on suspicious `ac7_final` rows must use the original per-suite log or a re-derived provenance-safe tally rather than the contaminated TSV.

The other rows need not be re-audited merely to commit this bounded repair.

### 7. Focused regression evidence — ACCEPTED

The focused battery is appropriate for the repair:

* Brief-I primary **26/26**
* mutants **green / detections intact**
* D3.25 ingress **13/13**
* columnizer **10/10**
* finalizer **60/60**
* D6.2 **31/31**
* Chapter-2 **12/12**
* Phase-3 worker **18/18**

Together with the assembly gate and the frozen-contract checks, that is enough for patch acceptance. 

## DISPOSITION

**Second production consumer:** RATIFIED IN-SCOPE.
**Two certified-suite fixture migrations:** RATIFIED.
**Array-4 implementation:** APPROVED.
**G-PHASE5-ASSEMBLY:** APPROVED.
**Nested-tally leak:** RECORDED; contaminated `d1_engine` summary row is not admissible evidence.
**No additional code changes requested before commit.**

Michael may commit and dual-push **this exact reviewed repair**.

After commit, the next acceptance step remains unchanged: deploy/parity as required and run a **fresh Michael-authorized production-shaped trial** from the repair commit. It must reach successful Phase-5 assembly/publication before B5/B6/B8 can be reclassified, B7 can receive its missing acceptance classification, and **Brief I can finally close.**
