# TB RULING — WINDOW-ANCHOR / GENERATOR-PHASE PROPOSAL v1.0

**Received:** 2026-08-18
**Responds to:** `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md` (relayed
pre-commit for review, per design→review sequence)
**Recorded by:** Team Alpha, verbatim below.

**Binding dispositions:**

| item | disposition |
|---|---|
| Architecture | **ACCEPTED** — separation, phase=0 pin, capability matrix, frozen ABI, continuation-law firewall all correct |
| Design rounds | **v1.1 with bounded corrections closes the design gate; no second conceptual round; then straight to Brief I** |
| Q1 old key | **HARD REJECT confirmed.** No shim, no mapping anywhere in the new path. Correction: `search_bounds.offset` REMOVED from config outright (no "tombstone comment" — JSON has no comments); retirement recorded in docs/changelog/schema migration. `optimal_window_config.json`, trial records, payloads, cache identities, survivor provenance emit new schema only. Repo-wide consumer audit → Brief II |
| Q2 metadata | **APPROVED; 22-array wall stays closed** provided no array added/removed/reordered/retyped/reshaped; metadata schema change explicit+versioned. `anchor_era` = **provenance, not authority**: derived from resolved dataset/session/anchor or validated named-domain request; never an accepted arbitrary string; record anchor, effective resolved range, session set, dataset identity beside it |
| Q3 legacy engine | Header-only freeze **NOT sufficient**. Infra docs still describe a `reverse_sieve` coordinator job targeting it. Rule: if a dispatch route exists in current source, remove/hard-disable it or align the engine; if gone, provide call-graph/reachability evidence + fail-loud historical-only entry guard. AC5 tests it. Then archival retention is fine |
| Q4 control era | **`[0,149]` as an anchor range REJECTED — the semantic catch.** 100 = historical **anchor** ceiling; 149 = historical **record-envelope** ceiling (anchor 149 + window 50 → record 198, outside history). **`control_anchor = [0, min(100, N_filtered − window_size)]`.** Fix §§4.2, 7.3, 8-Q4 and everywhere `[0,149]` is called an anchor range. D.1 scientific run: **midday OR evening, never combined**; window size, skip geometry, thresholds, seed domain, trial budget held constant between arms |
| Q5 phasing | **Two briefs APPROVED, lineage SEQUENTIAL:** Brief I = worker + schema + capability validation + semantic tests; Brief II = optimizer + registry + metadata/provenance + D.1 reach demo, **starting from the accepted Brief-I commit** (not independently from `e9ca800`). Final acceptance report shows sequential lineage + full pre/post diff to `e9ca800` |
| Derived max anchor | §4.3 needs an **exact machine representation** — typed, mechanically derivable, configuration can narrow but never widen |
| AC1 strengthening | Prove semantic independence with an **internal builder/arg-capture unit test using synthetic nonzero phase on a supported ABI**, while the public v1 schema stays fail-closed against nonzero production phase |
| Comparability caveat | **ACCEPTED as written** |
| Continuation law | Untouched, confirmed derived-not-configurable |

---

## Ruling body (verbatim)

## TEAM BETA RULING — WINDOW-ANCHOR / GENERATOR-PHASE PROPOSAL v1.0

**Status: ARCHITECTURE ACCEPTED; REVISION REQUIRED BEFORE IMPLEMENTATION.**

The core design is correct. The two semantic roles are cleanly separated, `generator_phase=0` is properly frozen for v1, the per-variant capability distinction is preserved rather than papered over, the kernel ABI remains frozen, and the Step-3 continuation law is correctly firewalled. This is the design F-4 needed.

I found **one genuine semantic error** plus several governance details that need tightening. None requires redesigning the architecture.

1. **Q1 — legacy `offset`: HARD REJECT CONFIRMED.** No deprecation shim. New-schema Step-1 inputs containing `offset` fail loud before assignment or GPU work. Old schema versions fail by version. There must be no `offset → window_anchor`, `offset → generator_phase`, or dual mapping anywhere in the new production path.

   One correction to §4.3: **do not leave `search_bounds.offset` in `distributed_config.json` as a "tombstone comment."** Strict JSON has no comments, and retaining the operative key creates exactly the stale semantic surface being retired. Remove the key. Record its retirement in documentation/changelog/schema migration, not as a live configuration entry. The current documentation confirms `offset` is an actual serialized optimizer parameter, so removing it is part of the semantic migration, not cosmetic cleanup.

   `optimal_window_config.json`, trial records, assignment payloads, cache identities and survivor provenance must likewise emit the new schema only. A repo-wide consumer audit belongs in Brief II because Step 1's primary configuration artifact historically serializes `offset`.

2. **Q2 — metadata addition APPROVED. The 22-array wall stays closed.** Adding `window_anchor`, `generator_phase`, and `anchor_era` to generation metadata does **not** open the 22-array contract provided no array is added, removed, reordered, retyped, or shape-changed. Metadata schema/version changes are allowed and should be explicit.

   `anchor_era` must be provenance, not authority: derive it from the resolved dataset/session/anchor relationship or from a validated named-domain request. Never accept an arbitrary `anchor_era="governed"` string as proof that the resulting slice is governed. Record the actual `window_anchor`, effective resolved range, session set, and dataset identity beside it.

3. **Q3 — header-only freezing of `reverse_sieve_filter.py` is NOT sufficient.** Alpha may freeze the legacy engine rather than retrofit it, but **only after proving it is not production-reachable and making that status executable**. A comment saying "historical only" does not close a path. This matters because the project infrastructure documentation still describes `reverse_sieve_filter.py` as the target of a `reverse_sieve` coordinator job, so deadness cannot simply be assumed.

   Therefore the rule is: if current source still contains a dispatch route to it, remove/hard-disable that route or align the engine. If the route is already gone, provide call-graph/reachability evidence plus a fail-loud historical-only entry guard. Then the untouched fused implementation may remain as archival code. Acceptance criterion 5 should test this.

4. **Q4 — `[0,149]` as an anchor range is REJECTED. This is the important semantic correction.** `149` is the maximum **record index historically reachable**, not the maximum historical **anchor**. The old optimizer sampled `offset/anchor ∈ [0,100]`; with `window_size ≤ 50`, that produced a union-of-reachable-records envelope ending at index 149.

   If the new design allows `window_anchor=149` with `window_size=50`, the resulting slice reaches record **198**, which is outside the historical reach. So the named historical control anchor domain is:

   **`control_anchor = [0, min(100, N_filtered − window_size)]`**

   and **`[0,149]` remains only the historical record-index envelope**, never an anchor range.

   This distinction must be corrected in §§4.2, 7.3, 8-Q4 and anywhere else `[0,149]` is called an anchor range. It is exactly the kind of anchor/extent category error this separation is intended to eliminate.

   For the later scientific D.1 run, preserve the original attack-plan constraint: **midday OR evening, never combined**, and hold window size, skip geometry, thresholds, seed domain and trial budget constant between governed and control arms. D.1 explicitly defines the experiment as session-scoped and differential.

5. **Q5 — two implementation briefs APPROVED, but their lineage is sequential.** Keep the proposed split:

   **Brief I:** worker + schema + capability validation + semantic tests.
   **Brief II:** optimizer + registry + metadata/provenance + D.1 reach demonstration.

   Both remain historically compared against certified reference `gate12-passed-attempt9 = e9ca800`, but Brief II must **start from the accepted Brief-I commit**, not independently branch from `e9ca800`. Otherwise two individually correct implementations could conflict on the same schema surfaces. The final acceptance report should show both the sequential implementation lineage and the full pre/post diff back to `e9ca800`.

Two smaller requirements should be incorporated while revising the proposal. First, §4.3 needs an exact machine representation for a **data-derived maximum anchor**; "max documented as data-derived" is not enough for a typed config/registry. Whatever representation Alpha chooses must make the effective domain mechanically derivable and prevent configuration from widening it. Second, strengthen acceptance criterion 1: because production v1 pins phase to zero, test semantic independence with an **internal builder/arg-capture unit test using a synthetic nonzero phase on a supported ABI**, while keeping the public v1 schema fail-closed against nonzero production phase. That proves the two codepaths are actually independent rather than merely observing that both see zero.

The comparability caveat is **accepted as written**. Historical `offset` changed both the data window and, on phase-capable kernels, generator state; post-separation phase-zero populations are therefore not legitimate regression comparators to historical populations. The Step-3 `offset = train_history_len` law also remains untouched; the existing scoring documentation confirms that this is a derived continuation offset, not a configurable Step-1 search parameter.

### Disposition

**No second conceptual design round is required.** Alpha should issue **v1.1** incorporating the rulings above. If v1.1 makes only these bounded corrections and introduces no new semantics, Beta considers the design gate closed and Alpha may proceed directly to **Implementation Brief I**.

The biggest catch was §4.2/Q4: **100 is the historical anchor ceiling; 149 is the historical record-envelope ceiling.** Everything else is tightening an otherwise sound design.
