# TB RULING — WINDOW-ANCHOR BRIEF I: THREE SCOPE ITEMS, ALL APPROVED

**Received:** 2026-08-21
**Responds to:** `docs/TB_RULING_REQUEST_WINDOW_ANCHOR_BRIEF_I_SCOPE.md`
(Items 1, 2, 3A; Item 3B and the RC-1 observation stated for the record, no ruling sought)
**Effect:** Brief I §2.2 AUTHORIZED TO PROCEED
**Recorded by:** Team Alpha, verbatim below.

**Binding dispositions:**

| item | disposition |
|---|---|
| Item 1 — `miner/range_miner_npz_writer.py:188` (`_CONTEXT_FIELDS`) pulled into Brief I | **APPROVED — bounded.** Replace the required context key only. No §4.5 provenance expansion, no new metadata semantics, no array added / removed / reordered / retyped / reshaped |
| Item 2 — Chapter 2 §7.2 text and its content gate | **APPROVED — three constraints (timing, gate, scope fence), below** |
| Item 3A — schema validation precedes integrity validation in `ResidueResolver.resolve` | **APPROVED** |
| Item 3B — Gate 20 / B6 fixture repair | Each integrity gate constructs a schema-valid payload, then mutates only the integrity property it exists to test. **Existing integrity assertions may not be relaxed** to accommodate the changed order |
| AC7 | **NOT AMENDED.** Item 2 was approved, therefore `test_chapter2_content_gate` must go green; no new red is permitted |
| RC-1 (F1 claiming-model fixture staleness) | **Evidence / carry-forward section, including Alpha's nested-tally correction. Not to be repaired opportunistically** |
| Clamp-site census | The **nine-site** census governs the implementation report, not the four-site count in the brief |
| Commit | Nothing committed. Report at the end of §2.2 as before |

**Item 2 — the three constraints, expanded:**

| constraint | requirement |
|---|---|
| **timing** | The chapter may not declare itself repaired. The chronology is preserved permanently: at the Chapter-2 audit anchor F-4 was CONFIRMED, NOT REPAIRED; Brief I is the subsequently authorized repair. During implementation the current-state disposition reads *"repair implemented by Window-Anchor Brief I; acceptance pending"*. Applies to `:831`, `:1133`, `:1346`. Only after Brief-I acceptance may it become unqualified REPAIRED |
| **gate** | Repoint the two F-4 source assertions from proving the old defect exists to proving the approved separation exists — the `_generator_phase_tail` delivery surface and the validated anchor-domain surface. `:578`'s historical F-4 text requirement is retained; F-4 is not deleted from the chapter |
| **scope fence** | The anchor-loop weakness and `:578`'s near-vacuity are recorded as follow-up debt, **not repaired here**. Explicitly: do not turn this into a general Chapter-2 gate redesign |

**The pin — three layers, and the ordering requirement:**

`generator_phase` mandatory and exactly `0`, enforced at **(1)** the coordinator public
assign-payload validation and **(2)** the worker execution seam. **(3)** The capability table
stays a **separate** invariant — it says whether an ABI *could* accept a phase, not whether v1
*policy* permits a value.

*The risk Beta identified:* if the v1 policy pin runs before the capability guard at the worker
seam, it rejects nonzero on every variant first, G-CAP-3 never exercises the capability guard,
and the gate goes green because policy rejected it — green on a fact it does not check. Named
as the third instance of that pattern in the session, after the Chapter-2 anchor loop and Gate 20.

*Therefore, binding:* order the worker seam **capability first, then policy**. G-CAP-3 (nonzero on
`java_lcg_hybrid`) hits capability and stops there. G-SEP-3 (nonzero on a capable variant such as
`lcg32_hybrid`) clears capability and hits the policy pin. Each gate then exercises its own guard,
and **the two errors must be distinctly named** so a test cannot mistake one for the other.
G-SEP-2's synthetic phase 7 bypasses both by reaching the builder directly via arg-capture.

---

## Ruling body (verbatim)

Reproduced byte-for-byte as received, in a fenced block so the original line breaks
(single newlines, no blank lines) are preserved without altering a character.

```text
Beta has ruled. All three items APPROVED. §2.2 is AUTHORIZED TO PROCEED.
Binding constraints from the ruling, beyond what we asked:
Item 2 — timing. The chapter may not declare itself repaired. Preserve the chronology permanently: at the Chapter-2 audit anchor, F-4 was CONFIRMED, NOT REPAIRED; Brief I is the subsequently authorized repair. During implementation the current-state disposition reads "repair implemented by Window-Anchor Brief I; acceptance pending". Only after Brief-I acceptance may it become unqualified REPAIRED. This applies to :831, :1133 and :1346.
Item 2 — gate. Repoint the two F-4 source assertions from proving the old defect exists to proving the approved separation exists — the _generator_phase_tail delivery surface and the validated anchor-domain surface. Retain :578's historical F-4 text requirement; do not delete F-4 from the chapter.
Item 2 — scope fence. The anchor-loop weakness and :578's near-vacuity are recorded as follow-up debt, not repaired here. Beta was explicit: do not turn this into a general Chapter-2 gate redesign.
No AC7 amendment exists. Item 2 was approved, so test_chapter2_content_gate must go green. No new red is permitted.
Item 1 is bounded. Replace the required context key only. No §4.5 provenance expansion, no new metadata semantics, no array added/removed/reordered/retyped/reshaped.
Item 3B. Each integrity gate constructs a schema-valid payload, then mutates only the integrity property it exists to test. Do not relax the existing integrity assertions to accommodate the changed order.
The pin — three layers, and an ordering risk you must handle. Beta requires generator_phase mandatory and exactly 0, enforced at the coordinator public assign-payload validation and at the worker execution seam. The capability table stays a separate invariant — it says whether an ABI could accept phase, not whether v1 policy permits a value.
The risk: if the v1 policy pin runs before the capability guard at the worker seam, it rejects nonzero on every variant first, and G-CAP-3 never exercises the capability guard — it goes green because policy rejected it, not because capability did. That is green on a fact it does not check: the third instance of that pattern in this session, after the Chapter-2 anchor loop and Gate 20.
So order the worker seam capability first, then policy. G-CAP-3 (nonzero on java_lcg_hybrid) hits capability and stops there. G-SEP-3 (nonzero on a capable variant such as lcg32_hybrid) clears capability and hits the policy pin. Each gate then exercises its own guard, and the two errors must be distinctly named so a test cannot mistake one for the other. G-SEP-2's synthetic phase 7 bypasses both by reaching the builder directly via arg-capture.
RC-1 goes in the evidence/carry-forward section, including your nested-tally correction. Do not repair it opportunistically.
The nine-site clamp census governs the implementation report, not the four-site count in the brief.
Proceed with §2.2. Report at its end as before; nothing committed.
```
