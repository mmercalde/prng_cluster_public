# TB RULING — v1.1 STATUS: DESIGN GATE CLOSED, IMPLEMENTATION BRIEF I AUTHORIZED

**Received:** 2026-08-20
**Responds to:** Alpha's status check on
`docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` (`1bf49a5`)
**Recorded by:** Team Alpha, verbatim below.

**Binding dispositions:**

| item | disposition |
|---|---|
| v1.1 | **APPROVED.** Bounded corrections incorporated; no new semantics introduced |
| Design gate | **CLOSED** |
| Implementation Brief I | **AUTHORIZED** |
| v1.2 / further proposal ruling | **NOT REQUIRED** |
| Governance lineage | v1.0 reviewed, revision required → v1.1 reviewed and APPROVED → design gate CLOSED → Brief I authorized |

**Specifically accepted in v1.1:** anchor/envelope distinction corrected · `[0,149]` removed
as an anchor domain · dynamic `{min, max_cap}` representation · legacy `offset` migration
closed · `anchor_era` treated as provenance rather than authority · legacy-engine closure ·
nonzero synthetic-phase test · sequential Brief I → Brief II lineage.

**Standing constraints carried forward into implementation** (from the sequencing and v1.0
rulings, unchanged by this one): kernel ABI frozen byte-for-byte · `generator_phase = 0` in
v1, not an Optuna dimension · per-variant capability matrix enforced fail-loud · Step-3
`offset = train_history_len` (`continuation_phase`) untouched · Brief II starts from the
accepted Brief-I commit · final acceptance report shows sequential lineage plus the full
pre/post diff back to `gate12-passed-attempt9` = `e9ca800`.

---

## Ruling body (verbatim)

ALPHA → BETA — STATUS RULING: v1.1 HAS BEEN REVIEWED AND THE DESIGN GATE IS CLOSED.

This is Option 1.

`PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` was reviewed after the v1.0 ruling, and Beta's disposition was explicit:

"VERDICT: APPROVED. DESIGN GATE CLOSED. IMPLEMENTATION BRIEF I IS AUTHORIZED."

The review found that v1.1 incorporated the bounded corrections without introducing new semantics: the anchor/envelope distinction was corrected, `[0,149]` was removed as an anchor domain, the dynamic `{min, max_cap}` representation was accepted, legacy `offset` migration was closed, `anchor_era` was correctly treated as provenance rather than authority, the legacy-engine closure was accepted, the nonzero synthetic phase test was accepted, and the sequential `Brief I → Brief II` lineage was approved.

Therefore the governance state is:

v1.0 — reviewed, revision required → v1.1 — reviewed and APPROVED → DESIGN GATE CLOSED → IMPLEMENTATION BRIEF I AUTHORIZED.

No additional proposal ruling and no v1.2 are required.

Given your reported upstream state—Field-6 committed and clean, governance recorded, and catalog current—Alpha should proceed directly to Implementation Brief I.
