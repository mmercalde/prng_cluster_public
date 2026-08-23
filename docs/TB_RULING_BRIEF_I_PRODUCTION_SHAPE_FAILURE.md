# TB RULING — BRIEF I PRODUCTION-SHAPE FAILURE AT `48a8705`: NOT ACCEPTED, BOUNDED REPAIR AUTHORIZED

**Received:** 2026-08-22
**Responds to:** `docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md`
and run `distributed_config_t1_eed23c7f` (stamp `20260822_143303`) at `48a8705`
**Effect:** Brief I **NOT ACCEPTED**. Defect classification **CONFIRMED**. A bounded
Brief-I repair is **AUTHORIZED** before another production run. **Brief II remains BLOCKED.**
**Recorded by:** Team Alpha, verbatim below.

**Binding dispositions:**

| item | disposition |
|---|---|
| Brief I at `48a8705` | **NOT ACCEPTED** |
| the Phase-5 failure | **BRIEF-I DEFECT — confirmed.** Not Brief-II debt, not a historical red |
| the retained-artifact reproduction | **PERSUASIVE.** `_CONTEXT_FIELDS` specifically tested and **refuted** as the source |
| **canonical array 4** | **Keep the name `"offset"`. Its post-separation value means `window_anchor` ONLY.** No rename · no 23rd array · `generator_phase` never fed into it. `CANONICAL_RECORD_FIELDS` and the intentionally duplicated field list keep the literal name |
| **⚠ correction to Alpha's rationale** | **NOT "coherent only because `generator_phase == 0`".** Array 4 has a deliberately legacy **wire name** with a single post-F-4 meaning: **`"offset"` = `window_anchor`; it is never generator phase.** Phase remains independently represented in versioned generation metadata. The legacy token does not fuse the values so long as no consumer treats it as both concepts |
| `DEP-ABI-V2-NPZ-SEMANTICS` | **RECORD NOW as an audit dependency.** Before nonzero phase is production-permissible, re-audit every consumer of canonical `"offset"` and prove it reads that field exclusively as `window_anchor` while phase stays durably available from metadata. **No 22-array amendment is pre-authorized** — ABI-v2 resurrects F-4 only if somebody again reads array 4 as both anchor and phase |
| repair scope | The production record builder (`utils/canonical_records.py`) plus the **minimum** contract docs/tests to encode the ruling |
| acceptance-test fixture | **Deliberately unequal values** — `window_anchor = 58`, `generator_phase = 0` — proving canonical `"offset"` receives **58, not 0**. *A `0/0` fixture would let the exact semantic mistake pass invisibly.* Must also prove a context with the new fields and **no legacy `offset`** survives the real assembly path |
| `G-PHASE5-SEAM` | **NOT DEFECTIVE. Keep unchanged; do not silently widen or repurpose.** The error was treating a local certificate as evidence the whole downstream chain had migrated |
| `G-PHASE5-ASSEMBLY` | **ADD as a distinct gate** — publish → assemble → canonical record build, reaching the real `assemble_trial`/`build_mode_records`. Non-vacuity must detect **restoration of `ctx["offset"]`** and **sourcing `"offset"` from `ctx["generator_phase"]`** |
| the run's affirmative evidence | **Valid observations** — 128/128 · zero lease expiries · zero disconnects · full saturation/turnover · `window_anchor=58`/`generator_phase=0` across the return path · **both Field-6 falsifiers observed in production for the first time (30 and 126, neither UNOBSERVED)** · drain-starvation signature absent. They **cannot substitute** for successful Phase-5 publication |
| **first production Field-6 observation** | **NOW LEGITIMATELY RECORDED** |
| B5 / B6 / B8 | **ACCEPTED as reported.** Real differences, correctly not smoothed into historical reds; downstream consequences of the one Phase-5 commit failure (Option-C retention working). **NOT three additional Brief-I defects** unless they persist after successful assembly |
| B7 | **PRE-EXISTING INVESTIGATION OPEN — acceptance classification DEFERRED** until a successful publication run. Do not fold its repair into the canonical-record fix |
| `OBSERVABILITY_GAP_1` | **FILED as its own governed item.** Same class as the F2 incidents. **Do not bundle** with the defect patch. Repair before the next major certification sequence, but **not a prerequisite** to rerunning this proof |
| deviation 10.1 — ledger archive/displacement | **RATIFIED.** A one-time governed migration, **not precedent**. Preserve the archive and manifest through the B7 and C2 investigations |
| deviation 10.2 — production-constructor ledger pre-creation | **RATIFIED.** Anti-fabrication preserved: production code created the schema, birth provenance captured pre-launch, zero rows at birth, every later row attributable to the run |
| `HARNESS-LEDGER-ORDER-1` | **FILED, not repaired here** |
| Brief II | **BLOCKED** |

**The governing sequence from here:**

```
48a8705 production proof FAIL
  -> bounded Brief-I defect repair
  -> committed repair hash
  -> focused batteries + assembly regression gate
  -> fleet parity / deployment as applicable
  -> new Michael-authorized production-shape run
```

The next run **must reach successful Phase-5 assembly/publication.** Only then may B5/B6/B8 be
reclassified, B7 receive its missing acceptance classification, and Brief I receive final
acceptance. **The failed run and its retained artifacts stay preserved** until the repair is
reproduced successfully against them offline and then verified in a fresh production run.

---

## Ruling body (verbatim)

Reproduced character for character as received.

## TEAM BETA RULING — BRIEF I PRODUCTION-SHAPE FAILURE AT `48a8705`

**VERDICT: BRIEF I IS NOT ACCEPTED AT `48a8705`. THE DEFECT CLASSIFICATION IS CONFIRMED. A BOUNDED BRIEF-I REPAIR IS AUTHORIZED.**

Run `distributed_config_t1_eed23c7f` did what this proof was supposed to do: it exercised the changed surface deeply enough to expose a real missed production consumer. The compute-side separation itself worked across the 25-worker fleet, but Phase-5 assembly failed on an unmigrated `ctx["offset"]` access in `utils/canonical_records.py`. That is a **Brief-I defect**, not Brief-II debt and not a historical red.

The retained-artifact reproduction is persuasive. `_CONTEXT_FIELDS` was specifically tested and refuted as the source: the run carried `window_anchor=58`, `generator_phase=0`, no `offset`; 5,632 manifests passed the new metadata seam; the failure occurs later at `build_mode_records`.

### 1. Frozen array 4 — ruling

**APPROVED disposition: keep canonical array/record field name `offset`, but its post-separation semantic value is `window_anchor`.**

The bounded production mapping is therefore conceptually:

`record["offset"] = ctx["window_anchor"]`

Do **not** rename canonical array 4. Do **not** add a 23rd array. Do **not** feed `generator_phase` into it. `CANONICAL_RECORD_FIELDS` and the intentionally duplicated canonical-array field list continue to contain the literal name `offset`.

There is, however, one correction to Alpha's proposed rationale:

**This is not coherent only because `generator_phase == 0`.**

From this ruling forward, array 4 has a deliberately legacy **wire name** but a single post-F-4 semantic meaning:

> **canonical array 4 `"offset"` = `window_anchor`; it is never generator phase.**

`generator_phase` is an independent quantity and remains represented independently in the versioned generation metadata, as v1.1 requires. The misleading legacy token does not fuse the values so long as no consumer treats it as both concepts.

That distinction matters for ABI-v2.

### 2. DEP-ABI-V2 — record a compatibility obligation, not a predetermined 22-array amendment

**YES, record this against DEP-ABI-V2 now, but as an audit dependency.**

Call it, for example, **DEP-ABI-V2-NPZ-SEMANTICS**:

> Before nonzero generator phase becomes production-permissible, re-audit every consumer of canonical `"offset"` and prove it interprets that field exclusively as `window_anchor`, while `generator_phase` remains independently and durably available from metadata.

Do **not** pre-rule that ABI-v2 must open the 22-array wall.

If all downstream consumers can correctly obtain phase from metadata, the existing 22 arrays can remain unchanged even with nonzero phase. If some governed downstream contract requires generator phase inside the canonical array body rather than metadata, **that future requirement** would trigger a separate governed array-contract amendment.

So ABI-v2 does not automatically resurrect F-4. It would resurrect F-4 only if somebody again interprets array 4 as both anchor and phase.

### 3. Authorized Brief-I repair scope

The repair may touch the production record builder and the minimum contract documentation/tests needed to encode the ruling above.

The critical executable correction belongs in `utils/canonical_records.py`. The deliberate duplicate field declarations remain `"offset"`.

The acceptance test must use deliberately unequal values, e.g.:

* `window_anchor = 58`
* `generator_phase = 0`

and prove canonical array/record `"offset"` receives **58**, not 0.

That is important: a `0/0` fixture would allow the exact semantic mistake to pass invisibly.

The repair must also prove that a context containing the new fields and **no legacy `offset`** survives the real assembly path.

## 4. G-PHASE5-SEAM — preserve it and add the missing outer gate

Alpha is correct: **G-PHASE5-SEAM is not defective.**

It certifies the seam it claims to certify. The error was treating that local certificate as evidence that the complete downstream consumer chain had migrated.

Do **not** silently widen or repurpose that existing gate.

Add a second gate with a distinct contract, conceptually:

**G-PHASE5-ASSEMBLY**

> A schema-valid real trial context containing `window_anchor` and `generator_phase`, with no legacy context key `offset`, survives:
>
> **publish → assemble → canonical record build**

and produces canonical `"offset" == window_anchor`.

It should reach the actual `assemble_trial`/`build_mode_records` surface that the current suite never touches.

For non-vacuity, the new gate must detect at least:

* restoration of `ctx["offset"]`;
* sourcing canonical `"offset"` from `ctx["generator_phase"]`.

That turns this specific production escape into a permanent regression barrier.

### 5. Production result classification

The run is a **failed acceptance run**, but it produced valuable affirmative evidence:

* 128/128 stripes completed;
* zero lease expiries;
* zero disconnects;
* full saturation/turnover;
* `window_anchor=58`, `generator_phase=0` crossed coordinator → fleet → return path;
* both Field-6 falsifiers were observed in production for the first time: **30** and **126**, neither `UNOBSERVED`;
* the prior drain-starvation signature was absent.

Those facts remain valid observations about the run. They simply cannot substitute for successful Phase-5 publication.

The first production Field-6 observation is therefore now legitimately recorded.

## 6. B5/B6/B8 classification — ACCEPTED

Alpha correctly did **not** smooth these into historical reds.

For this run:

* B5 changed from Attempt-9 run-scoped PASS to FAIL;
* B6 changed from PASS to FAIL;
* B8 changed from zero held to 5,632 held.

Those are real differences. But they are downstream consequences of the one named Phase-5 commit failure: Option-C retention did exactly what it is supposed to do when delivery fails.

They are **not three additional Brief-I defects** unless they persist after successful Phase-5 assembly.

### 7. B7 — remains open and unclassified for acceptance

Alpha's treatment is correct.

The run reproduced the same historical B7 mechanism: `0/5632` non-`none`, matching `0/29,082` historically. But because this run never successfully published, the third condition from my earlier accept-while-red ruling cannot be established.

Therefore B7 remains:

**PRE-EXISTING INVESTIGATION OPEN — acceptance classification deferred until a successful publication run.**

Do not fold B7 repair into the canonical-record fix.

## 8. OBSERVABILITY_GAP_1 — filing ACCEPTED, separate repair required

The classification is accepted.

`commit_trial` captured the actual exception in an in-memory event object, persisted only `commit_delivery_status=failed`, and emitted no durable cause. The result was a 25-GPU terminal failure whose causal exception disappeared from normal operator-visible evidence.

This is the same architectural failure class as the earlier F2 incidents: **the system knows the reason and destroys it before a durable observer receives it.**

Open **OBSERVABILITY_GAP_1** as its own governed item.

Do **not** bundle it with the canonical-record repair. It is not necessary to correct the data semantics, and mixing it into the defect patch would change another coordinator surface before we have closed Brief I.

It should be repaired before the next major certification sequence, but it is **not a prerequisite to rerunning this specific Brief-I production proof**, because the present defect is already deterministically identified and retained.

## 9. Operational deviation 10.1 — RATIFIED

The pre-separation ledger archive/displacement is ratified.

The sequence used was appropriate:

**hash → byte-preserving copy outside active staging → re-hash → functional read → move active originals aside**, including the WAL. Nothing was deleted and the historical ledger remains queryable.

This is a **one-time governed migration action**, not precedent for casually replacing production ledgers.

Preserve the archive and its manifest through the B7 and C2 investigations.

### 10. Operational deviation 10.2 — RATIFIED

Ledger pre-creation through the **production `MinerLedger` constructor** is also ratified.

The anti-fabrication properties were preserved because:

* production code created the schema;
* birth provenance was captured before launch;
* total rows at birth were zero;
* every subsequent row is attributable to the actual run.

That is preferable to hand-creating a database merely to satisfy the sampler.

The associated harness finding is valid and becomes a separate debt item:

**HARNESS-LEDGER-ORDER-1 — the production-shape harness cannot bootstrap a legitimately fresh ledger because the sampler requires the ledger before the coordinator that normally creates it.**

Do not repair that in the canonical-record patch.

# 11. What happens next

The governing sequence is now:

**`48a8705` — production proof FAIL → bounded Brief-I defect repair → committed repair hash → focused batteries + assembly regression gate → fleet parity/deployment as applicable → new Michael-authorized production-shape run.**

The next run must reach successful Phase-5 assembly/publication. Only then can:

* B5/B6/B8 be reclassified after the root fault is removed;
* B7 receive the missing acceptance classification;
* Brief I receive final acceptance.

**Brief II remains blocked.**

The failed run and its retained artifacts should remain preserved until the repair has been reproduced successfully against them offline and then verified in a fresh production-shaped run.

## Final rulings

**Canonical array 4:** keep name `"offset"`; value means **window_anchor only**.

**DEP-ABI-V2:** add an NPZ/canonical-semantics audit dependency now; **no automatic 22-array amendment** is pre-authorized.

**G-PHASE5-SEAM:** keep unchanged; add a distinct end-to-end Phase-5 assembly/record-build gate.

**Ledger archive/displacement:** RATIFIED.

**Production-constructor ledger pre-creation:** RATIFIED.

**Fresh-ledger harness ordering defect:** FILED, not repaired here.

**OBSERVABILITY_GAP_1:** FILED as a separate coordinator observability defect.

**Brief I at `48a8705`: NOT ACCEPTED. A bounded Brief-I repair is AUTHORIZED before another production run.**