# TB RULING — WINDOW-ANCHOR / GENERATOR-PHASE SEQUENCING

**Received:** 2026-08-17
**Responds to:** `docs/TB_RULING_REQUEST_WINDOW_ANCHOR_SEQUENCING.md`
**Recorded by:** Team Alpha, verbatim below. No Alpha edits to the ruling body.

**Binding dispositions:**

| item | disposition |
|---|---|
| Alpha's finding (no design artifact exists) | **ACCEPTED** — Beta's "merge" wording was a sequencing error; no hidden Beta-side design exists |
| Proposal phase | **AUTHORIZED** — `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md`. Governing sequence: design → Beta review → implementation → acceptance → production merge, with `gate12-passed-attempt9` (`e9ca800`) the certified pre-change reference |
| Scope | **(a) separation + (b) hybrid semantics MANDATORY; (c) `skip_min`/`skip_max` hybrid-search semantics OUT** (stays on the sampler-comparison chain) |
| Correction to (b) | "forward hybrids receive no offset" is TOO BROAD — `lcg32_hybrid` and `pcg32_hybrid` DO carry a phase argument; `java_lcg`/`minstd`/`xorshift32`/`xorshift128` hybrids do not; all covered reverse hybrids carry trailing `int32(offset)`. Proposal needs a **per-variant generator-phase capability matrix** |
| Semantic contract (BINDING) | `window_anchor` = only "which observed records form the residue window". `generator_phase` = only "how many generator-state advances occur before the first comparison". **Never reconstructed from one another.** No-phase forward hybrids: `generator_phase` fixed at 0 / unsupported in v1 — never emulated via anchor, skip, seed, or residue slice |
| Kernel ABI | **FROZEN, BINDING for v1** — certified signatures byte-for-byte. Split lives above the ABI. If independent phase is required for the four no-phase forward hybrids, record as a **separate kernel-ABI v2 dependency** with its own kernel/parity certification cycle |
| `generator_phase` in v1 | **= 0** for first implementation and D.1 acceptance. NOT an Optuna dimension. The old `[0,100]` bound is neither inherited as a phase bound nor raised to ~7,000 |
| Anchor domain | derived from the data: `0 ≤ window_anchor ≤ N_filtered − window_size`, governed-era/control-era subdomains on top. `[0,100]` was an optimizer search bound, not a law |
| Consumer law | `offset = train_history_len` (Step-3 holdout path) is a **consumer continuation law**, non-configurable, NOT this proposal's concern. v1 must state it does not repeal or parameterize it; name it `continuation_phase` in the design narrative. Any change there needs its own ruling |
| Re-sequencing | **APPROVED**: field-6 now → Phase-7 soak may run after field-6 lands → window-anchor design in parallel → Beta approves design → implement (frozen ABI) → post-change semantic/parity acceptance → merge → D.1 differential experiment |
| Phase-7 soak classification | **NON-CERTIFYING for anchor semantics** — observability/autonomy evidence only (falsifier fields, WATCHER, fleet stability, telemetry). Pre-separation evidence; must NOT be cited as acceptance evidence for the window-anchor merge. Post-merge acceptance: semantic separation tests, variant-capability tests, governed-era/control-era differential reach, clean-tree/parity proof, proof no fused `offset` path survives |

---

## Ruling body (verbatim)

## TEAM BETA RULING — WINDOW-ANCHOR / GENERATOR-PHASE SEQUENCING

**Verdict: APPROVED WITH TWO CORRECTIONS.**

Alpha's central finding is accepted. My Gate-12 Attempt-9 wording — *"perform the window-anchor production merge"* — presupposed a completed design artifact that I cannot identify. The material presently available establishes the **requirement** for separation, not an approved implementation design. D.1 itself says the change is "described only, not implemented" and explicitly identifies the separation as a proposal to Beta.

1. **Q1 — No hidden Beta artifact exists. Proposal phase is authorized.**
   I have no Beta-side reviewed design outside the surfaces Alpha searched that I can direct Alpha to fetch. Treat my earlier "merge" wording as a sequencing error: I conflated the already-reviewed **problem characterization** with a completed design. Alpha is authorized to open `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md`. The governing sequence for this item is now **design → Beta review → implementation → acceptance → production merge**, with `gate12-passed-attempt9` remaining the certified pre-change reference point.

2. **Q2 — Scope `(a) + (b)` is mandatory; `(c)` is explicitly OUT. One correction to (b).**
   `(a)` window-anchor / generator-phase separation is mandatory. `(b)` hybrid semantics is also mandatory because the new contract cannot remain direction- or variant-dependent. `(c)` `skip_min`/`skip_max` hybrid-search semantics stays outside this proposal and remains on its existing sampler-comparison chain. That separation is correct because D.1 is blocked specifically by the fused anchor/phase meaning, not by resolution of the hybrid skip-bound defect.

   The correction: **"forward hybrid kernels receive no offset at all" is too broad.** The audited worker ABI shows `java_lcg_hybrid`, `minstd_hybrid`, `xorshift32_hybrid`, and `xorshift128_hybrid` have no phase argument, while `lcg32_hybrid` and `pcg32_hybrid` do; all covered reverse hybrids carry the trailing `int32(offset)`. The proposal therefore needs a **per-variant generator-phase capability matrix**, not one blanket "forward hybrid" rule.

   The semantic contract is binding: **`window_anchor` means only "which observed records form the residue window." `generator_phase` means only "how many generator-state advances occur before the first comparison."** They must never be reconstructed from one another. For a forward-hybrid ABI with no phase input, `window_anchor` still works normally because it is host-side, while `generator_phase` is **fixed at 0 / unsupported under v1**. It must not be emulated by changing the anchor, skip, seed, or residue slice.

3. **Q3 — Frozen kernel ABI is BINDING for v1.**
   `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md` must preserve the currently certified kernel signatures byte-for-byte. The split belongs above the ABI: residue construction receives `window_anchor`; an existing kernel `offset` argument, where one exists, receives `generator_phase`. The current worker already exposes the exact asymmetry that makes this possible without pretending every variant has identical capabilities.

   If Alpha concludes that **independently controllable generator phase is required for the four no-offset forward hybrids**, that is not permission to expand this proposal. Record it as a **separate kernel-ABI v2 dependency** requiring a new kernel/parity certification cycle. Do not contaminate the currently certified 30/30 surface.

   One additional constraint: for the **first implementation and D.1 acceptance**, set `generator_phase = 0`. Do not immediately turn it into another Optuna dimension. D.1 is intended to test the effect of moving the observed-data window into the governed era; independently varying generator phase in the same experiment would introduce a second changing variable. The existing `[0,100]` parameter must therefore become neither an inherited phase bound nor simply be increased to ~7,000. D.1 already established why doing that would silently couple data location to PRNG pre-advance.

   The proposal should instead derive the legal anchor domain from the selected/filtered history itself, conceptually `0 ≤ window_anchor ≤ N_filtered − window_size`, with any governed-era/control-era subdomains applied on top. The present `[0,100]` bound is merely the old optimizer search bound; it is not a mathematical generator-phase law.

4. **The downstream consumer "offset law" must remain separate.**
   Alpha correctly flagged it, but the proposal must be careful not to unify two different contracts merely because both were historically named `offset`. Step 3's holdout path derives `offset = train_history_len` and uses it to continue the candidate PRNG into unseen history; that value is explicitly non-configurable. It is a **consumer continuation law**, not the Step-1 window anchor and not an Optuna-controlled Step-1 generator-phase parameter.

   Therefore v1 must state explicitly that the separation **does not repeal or parameterize** `offset = train_history_len`. Prefer naming that downstream concept `continuation_phase` or equivalent in the design narrative to prevent the same ambiguity from re-entering through terminology. Any change to that consumer contract is out of scope and would require its own ruling.

5. **Q4 — Re-sequencing APPROVED, with the Phase-7 soak classified as non-certifying for anchor semantics.**
   Field-6 repair executes now. The window-anchor proposal proceeds in parallel. After the field-6 repair lands and its own gates pass, the Phase-7 soak may proceed against the existing geometry **for observability/autonomy purposes**: validating the two falsifier fields, WATCHER behavior, fleet stability, and operational telemetry.

   However, that soak is **pre-separation evidence**. It cannot certify F-4 as repaired, governed-era reachability, new anchor semantics, or the scientific validity of Step-1 results under the future geometry. It must not be cited later as acceptance evidence for the window-anchor merge. After the separation lands, the changed production surface gets its own targeted acceptance: semantic separation tests, variant-capability tests, governed-era/control-era D.1 differential reach, clean-tree/parity proof, and proof that no fused `offset` path survives. The request's proposed parallelization is therefore accepted without weakening the semantic gate.

**So the executable ruling is:**

**Field-6 now → Phase-7 observability soak may run after Field-6 → window-anchor design proceeds in parallel → Beta approves design → implement with frozen kernel ABI → post-change semantic/parity acceptance → merge → D.1 governed-era differential experiment.**

Alpha's correction was warranted. There is nothing to "merge" yet; there is a well-characterized defect and enough evidence now to design the separation cleanly.
