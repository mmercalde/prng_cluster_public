# TB RULING — WINDOW-ANCHOR BRIEF I: CODE REVIEW PASSED, COMMIT AUTHORIZED

**Received:** 2026-08-22
**Responds to:** `docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md` (the §8 acceptance report)
and the reviewed 20-file tracked diff at `205ae84`
**Effect:** Michael is authorized to commit. **Brief I is NOT yet finally accepted.**
**Recorded by:** Team Alpha, verbatim below.

**Binding dispositions:**

| item | disposition |
|---|---|
| Code review | **PASSED. APPROVED FOR COMMIT.** No rework requested; no additional code amendment required before the commit |
| Brief I final acceptance | **NOT GRANTED YET.** Contingent on the post-commit closure sequence below |
| §9.1 — all three census corrections | **RATIFIED.** They are findings against the brief, not implementation deviations |
| — measured kernel arities | **Factual erratum only.** Does not alter the frozen-ABI design or reopen the design gate. Load-bearing property: positions and dtypes pinned independently, 44 kernel bodies hash-identical |
| — fourth legacy route (`coordinator_sieve_dynamic.py`) | **Belongs in Brief I.** A replacement image able to recreate the live coordinator route; leaving it executable would have defeated §4.8. Closing four rather than three is correct implementation, **not scope expansion** |
| — eight import consumers | **ACCEPTED.** Does not require making the archival module unimportable; the imports/data-loading permitted, execution guarded boundary is accepted |
| — nine clamp implementations | **ACCEPTED with a qualification for the governance record:** Brief I does **not** establish repo-global eradication of every old `offset` consumer. `sieve_gpu_worker.py` and the other C-2 consumers remain part of Brief II's already-authorized repo-wide consumer audit. The independent KAT reference **stays independently implemented** — do not "repair" it into a copy of production |
| Chapter 2 wording | **"implemented, acceptance pending" is still exactly right.** No unqualified `REPAIRED` until the complete Window-Anchor change, **including Brief II**, receives final acceptance |
| M8 | **INVALID-BY-SCOPE ACCEPTED** — not survived, not waived. No Brief-I production expression computes `control_era`; a mutation cannot be credited against a site that does not yet exist. **Transfers intact to Brief II and must mutate the real production control-era ceiling 100 → 149** once the resolver exists. Brief II may not close with M8 omitted |
| §9.4 — pinned-executable-source hazard | **STANDING RULE `EXEC-PIN-1` ADOPTED**, effective with this review (full text in the body). Resolved-name sets to be **mechanically derived** (`symtable` or equivalent), not maintained from memory |
| — the two `test_s172_attempt6_remediation.py` bridges | **RATIFIED.** They adapt the historical executable fixture without weakening production's hard rejection of legacy `offset` |
| — R-1 bridge | **Not required.** No Brief-I changed live name intersects the pinned `_pump_deferred` resolution set |
| — generic bridge framework | **DO NOT ADD in this commit.** EXEC-PIN-1 is a standing review obligation, not authorization for another refactor |
| §12 — battery reconciliation | **ACCEPTED**, with one **wording constraint**: the three deepened suites may **not** be described as "zero differential" at the suite level. Correct characterization: **same pre-existing root cause / changed observable failure depth / no new Brief-I production defect** |
| — not migrating the three deepened suites | **AGREED.** Repair would be unrelated RC-1 fixture work and would expand the commit substantially |
| — the one improved pre-existing red | **NOT CREDITED to Brief I.** Improvement of a broken fixture is not evidence this implementation fixed its defect. Keep it pending its own investigation |
| — RC-1 enlarged census (8 suites, not 1) | **ACCEPTED as a finding.** Does **not** enter this commit |
| Gate 22 / W-NO-WEAKENING reds | **TRANSIENTS ACCEPTED** as working-tree-state observations. **Do not widen either allowlist or weaken either gate.** Both clear naturally once this exact reviewed diff becomes `HEAD` |
| — Gate-22 untracked-directory blind spot | **Valid separate finding. Record it; do not repair it here** |
| Mutation evidence | **ACCEPTED. 14 DETECTED / 1 INVALID-BY-SCOPE / 0 SURVIVED.** Splitting original M14 into execute-seam M14 + structural M15 was the right response — one mutant could not honestly prove two structurally disjoint levels |
| Item 2 chapter treatment | **ACCEPTED.** Obeys the prior ruling: historical F-4 verdict intact at its audit anchor, later implementation recorded as acceptance pending |
| Commit scope | **The exact reviewed 20-file tracked diff and the new Brief-I artifacts** |

**Post-commit closure sequence — required before Beta will call Brief I ACCEPTED or authorize Brief II to start:**

| # | obligation |
|---|---|
| 1 | Confirm the committed hash and clean tracked tree; rerun the two commit-sensitive suites so **Gate 22 and W-NO-WEAKENING clear without allowlist changes**. Rerun the Brief-I primary suite and both historical scope proofs **from the committed hash** as a provenance check |
| 2 | Run the required **committed-tree fleet / production-shape proof**. The report marks the live 25-daemon production-shape observer **NOT RUN / UNAVAILABLE**, while v1.1 retains clean-tree/fleet parity as an acceptance obligation. **Identical kernel hashes are necessary but do not replace host/worker schema parity** |
| 3 | Report the fleet-parity / production-shape outcome **against the Brief-I commit hash**. The Phase-7 soak is **non-certifying** and cannot substitute for this gate |

On clean closure, Beta marks that commit **BRIEF I ACCEPTED**, and **Brief II starts from that accepted commit**, per the v1.1 lineage.

---

## Ruling body (verbatim)

Reproduced character for character as received. Unlike the scope ruling, this document
arrived as well-formed Markdown with its own heading and paragraph structure, so it is
rendered live rather than fenced — no character is altered either way.

## TEAM BETA CODE REVIEW — WINDOW-ANCHOR BRIEF I

**VERDICT: CODE REVIEW PASSED. APPROVED FOR COMMIT. BRIEF I IS NOT YET FINALLY ACCEPTED.**

The implementation report is strong enough to authorize Michael to commit the review diff. Final Brief-I acceptance remains contingent on the post-commit state-sensitive closures and the fleet/production-shape obligation that could not legitimately be exercised against an uncommitted tree. 

### §9.1 — all three census corrections are RATIFIED

The corrections are findings against the brief, not implementation deviations.

The measured kernel arities replace the erroneous v1.1 census as a **factual erratum only**. They do not alter the frozen-ABI design or reopen the design gate. The important property is that the implementation pins the actual argument positions and dtypes independently and the 44 kernel bodies remain hash-identical. 

The fourth legacy route, `coordinator_sieve_dynamic.py`, absolutely belongs in Brief I. It is a replacement image capable of recreating the live coordinator route, so leaving it executable would have defeated §4.8. Closing all four routes rather than the three transcribed in the brief is the correct implementation of the design, not scope expansion. 

Likewise, eight import consumers rather than four does not require making the archival module unimportable. The chosen boundary—imports/data loading permitted, execution guarded—is accepted.

The nine clamp implementations need one qualification in the governance record: **Brief I does not establish repo-global eradication of every old `offset` consumer.** The migrated RANGE-MINER path is separated and the legacy reverse engine is execution-closed, but the newly identified `sieve_gpu_worker.py` live surface and the other C-2 consumers remain part of Brief II's already-authorized repo-wide consumer audit. The independent KAT reference should remain independently implemented; do not “repair” it into a copy of production. 

Accordingly, Chapter 2's current wording—**implemented, acceptance pending**—is still exactly right.

### M8 — INVALID-BY-SCOPE classification ACCEPTED

I accept M8 as **INVALID**, not survived and not waived.

There is presently no Brief-I production expression computing `control_era`; the actual era resolver belongs to the optimizer/Brief-II surface. A mutation cannot receive credit against a code site that does not yet exist. The report correctly preserves the missing coverage rather than laundering it into a different mutation. 

This creates a binding Brief-II obligation:

**M8 transfers intact to Brief II and must mutate the real production control-era ceiling from 100 to 149 once that resolver exists.**

Brief II cannot close with M8 omitted on the ground that Brief I already tested the anchor/envelope arithmetic.

### §9.4 — standing rule ADOPTED

The pinned-executable-source hazard is real and distinct from SR-1.

SR-1 protects historical exact-digest comparisons. It does not protect a historical function body that is literally executed against a live module namespace whose helper signatures and schemas continue evolving. The `attempt6_remediation` break demonstrates the first case; `r1_drain_remedy` demonstrates the latent member where compatibility survived only because the resolved-name intersection happened to remain empty. 

I therefore adopt the following standing rule, effective with this review:

> **EXEC-PIN-1:** whenever an authorized change alters the schema, signature, or callable contract of a live name, every commit-pinned Python source arm that resolves that name from a live namespace must be re-evaluated before acceptance. If translation is required, it must be test-local, preserve the historical pinned source, and document the bridge explicitly. A coincidental empty intersection is evidence of present compatibility, not proof of permanent isolation.

The resolved-name set should be mechanically derived where practical (`symtable` or equivalent), not maintained from memory.

The two bridges added to `test_s172_attempt6_remediation.py` are **ratified**. They adapt the historical executable fixture to the current live contracts without weakening production's hard rejection of legacy `offset`. No bridge is presently required for R-1 because none of Brief I's changed live names intersects the pinned `_pump_deferred` resolution set. 

Do **not** add a generic bridge framework in this commit. EXEC-PIN-1 is a standing review obligation, not authorization for another refactor.

### §12 — reconciliation ACCEPTED, with one wording constraint

A pre-existing red remaining red is not, by itself, adequate regression evidence if its failure point moved. Alpha was right to inspect the movements rather than compare only final tallies.

For the three suites that deepened because the new mandatory schema rejects their already-stale fixtures earlier, I agree with **not migrating them in Brief I**. Repairing them would be unrelated RC-1 fixture work and would expand this commit substantially. The correct characterization is:

**same pre-existing root cause / changed observable failure depth / no new Brief-I production defect.**

Do not describe those three as “zero differential” at the suite level if their failure positions changed.

The one pre-existing red that improved is likewise **not credited to Brief I**. Improvement of a broken fixture is not evidence that this implementation fixed its underlying defect. Keep it pending its own investigation.

RC-1 itself is accepted as a newly enlarged census finding—eight affected suites rather than the one historically recorded—but it does **not** enter this commit. 

### Gate 22 and W-NO-WEAKENING — TRANSIENTS ACCEPTED

Both reds are correctly classified as working-tree-state observations.

Do not widen either allowlist or weaken either gate. Gate 22 sees authorized modified `.py` files; W-NO-WEAKENING correctly observes that the live tree does not equal `HEAD`. Both should clear naturally once this exact reviewed diff becomes `HEAD`. 

The Gate-22 untracked-directory blind spot is a valid separate finding. Record it; do not repair it here.

### Mutation evidence — ACCEPTED

The mutation discipline is good: clean controls, fresh interpreters, exact application checks, execution proof, assertion-level detection, and explicit INVALID classification rather than counting harness failures as detections.

Splitting the original M14 into execute-seam M14 and structural `BuildContext` M15 was the right response. One mutant could not honestly prove two structurally disjoint levels. The resulting M15 gives G-NO-FUSED the missing construction-level non-vacuity. 

**14 DETECTED / 1 INVALID-BY-SCOPE / 0 SURVIVED is accepted.**

### Item 2 chapter treatment — ACCEPTED

The timing treatment obeys the prior ruling: the historical F-4 verdict remains intact at its audit anchor, while the later implementation is recorded as **acceptance pending**. That avoids rewriting history and also avoids leaving the current chapter falsely describing the present code as still fused. 

No unqualified `REPAIRED` should appear until the complete Window-Anchor change, including Brief II, receives final acceptance.

### Commit authorization and remaining gate

Michael may now commit **the exact reviewed 20-file tracked diff and the new Brief-I artifacts**. No additional code amendment is required before that commit.

After commit, there is one closure sequence before I will call Brief I **ACCEPTED** or authorize Brief II to start:

1. Confirm the committed hash and clean tracked tree; rerun the two commit-sensitive suites so Gate 22 and W-NO-WEAKENING clear without allowlist changes. Rerun the Brief-I primary suite and both historical scope proofs from the committed hash as a provenance check.
2. Run the required committed-tree fleet/production-shape proof. The report explicitly marks the live 25-daemon production-shape observer as **NOT RUN / UNAVAILABLE**, while v1.1 retains clean-tree/fleet parity as an acceptance obligation. Kernel hashes being identical is necessary but does not replace host/worker schema parity. 
3. Report the resulting fleet parity/production-shape outcome against the **Brief-I commit hash**. The Phase-7 soak remains non-certifying and cannot substitute for this gate. 

If those close cleanly, Beta will mark that commit **BRIEF I ACCEPTED**, and Brief II starts **from that accepted commit**, exactly as the v1.1 lineage requires. 

**So the operative ruling is: commit authorized; no rework requested; EXEC-PIN-1 adopted; M8 formally transferred to Brief II; final Brief-I acceptance waits only on committed-state closure and fleet/production-shape proof.**
