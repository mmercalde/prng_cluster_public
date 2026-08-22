# SESSION CHANGELOG — 2026-08-22 — WINDOW-ANCHOR BRIEF I

**SR-2 compliant: date plus topic, no S-number.**

**Scope:** implement `docs/S172_WINDOW_ANCHOR_BRIEF_I.md` in full — §2.1 worker, §2.2
coordinator, §2.3 NPZ writer, §2.4 legacy closure, §4 gates, §5 mutants, §6 SR-1, §8 report.
**Authority:** v1.1 design gate CLOSED + the 2026-08-21 scope ruling (Items 1/2/3 APPROVED).
**Base:** HEAD `205ae84`, digests verified. **Nothing committed** — Michael commits after Beta.

## Delivered

- **The separation.** `offset` split into `window_anchor` (host, record selection) and
  `generator_phase` (device, pre-advance). Silent clamp removed and replaced by a validated
  derived domain computed post-session-filter. Legacy key hard-rejected, never mapped. v1 zero
  pin at **both** seams, ordered capability-then-policy with distinct exception types.
  **Kernel ABI frozen: 44/44 `kernel_source` hashes identical to HEAD.**
- **Legacy engine CLOSED** (AC5). Four dispatch routes hard-disabled; engine execution guarded
  while import and the loader stay open for the eight diagnostic consumers.
- **Gates:** `tests/test_s172_window_anchor_brief_i.py`, **25/25**.
- **Mutants:** `tests/test_s172_window_anchor_brief_i_mutants.py`, **14 DETECTED / 1
  INVALID-by-scope / 0 SURVIVED**, 16/16 clean controls.
- **SR-1:** nine definitions declared in both `DECLARED_CHANGED` sets with provenance;
  `serve_trial` added to R-1's only. `ADDED = 0` on both anchors. R-1 44/44, MP-1 38/38.
- **Item 2:** Chapter 2 §7.2 rewritten with §7.2.1 added; F-4 rows at :831/:1133/:1346 and
  Chapter 1 :332/:337 re-disposed as *"repair implemented by Window-Anchor Brief I; acceptance
  pending"* — historical verdicts preserved, no unqualified REPAIRED. `G-SOURCE-ANCHORS`
  repointed and proven load-bearing by injection. `test_chapter2_content_gate` **12/12**.
- **Report:** `docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md` (734 lines).
- **Ruling request:** `docs/TB_RULING_REQUEST_WINDOW_ANCHOR_BRIEF_I_SCOPE.md`, ruled APPROVED.

## Findings for Beta

1. **Three census corrections against the brief**, one mechanism each time (token grep,
   truncated/pre-filtered, survivors counted): clamp sites 4→**9**; dispatch routes 3→**4**;
   import consumers 4→**8**.
2. **`coordinator_sieve_dynamic.py` is a REPLACEMENT IMAGE**, not a stray copy —
   `test_sieve_dynamic.sh:36` runs `cp coordinator_sieve_dynamic.py coordinator.py`. Unclosed,
   one `cp` would have reopened a closed route.
3. **New hazard class — pinned-executable-source vs live-helper schema coupling.** Two Python
   members; `r1_drain_remedy` is LATENT and survived on a 4-of-10 name coincidence. SR-1 analog
   recommended, **not enacted**, scoped to the class.
4. **RC-1:** eight suites share the F1 claiming-model fixture mechanism; the record attributes
   it to one. Independent of Brief I.
5. **Nine tooling bugs Alpha found in its own work**, two of them false greens found only by
   fault injection and by a mutant. Dominant pattern: **instrumentation misreporting production,
   not production being wrong.**
6. **Gate 22 blind spot:** `git status --porcelain` collapses a wholly-untracked directory to
   one entry that does not end in `.py`. Recorded; no change proposed.

## Two commit-time transients — NOT regressions

Both are uncommitted-working-tree artifacts and appear in **every** battery run until commit:
`phase4_coordinator` 62/63 (**Gate 22 only**) and `gate12_cleantree_admission` 30/31
(**W-NO-WEAKENING**, a live working-tree-equals-HEAD check with no stored digest). D3.5 is not
weakened — producer identity, `_repository_state` AST digest, `run_finalizer.py` and
`.gitignore` all verified unchanged.

## AC7 — final 47-suite battery (`logs/ac7_final/`)

**32 green / 15 red. 2 green→red vs baseline (both documented transients), 0 red→green, 2 new
suites both green.** 13 pre-existing reds still red; `test_chapter2_content_gate` green.
Five of six chargeable reds retired by fixture migration; `phase3_worker` 17/17 → 18/18 with
**8/8 fixture-shape and 0 assertion changes**, proven by diffing assert-lines against HEAD.

**Four suites MOVED WITHIN their reds — reported, not absorbed.** Beta's binding characterization
for the three that deepened, kept on one line so it greps as a unit:
*same pre-existing root cause / changed observable failure depth / no new Brief-I production defect*
— and they may **not** be described as "zero differential" at the suite level. `d0` 11/12→0/12,
`d1_workflow` 5/8→1/8, `d2` 6/7→0/7 all **deepened** from one cause: fixtures build a serve
context the new fail-closed guard rejects. Their fixtures were deliberately not migrated —
migration would change which line they die on without turning any of them green, since RC-1
fails them independently. Carried forward. `chapter1_p0` 8/12→**10/12 improved, cause NOT
established and NOT credited to this work** — flagged for its own investigation, because an
unexplained green is as suspicious as an unexplained red.

## Beta rulings recorded verbatim

Both rulings saved to `docs/` in dispositions-table shape, bodies verified byte-identical to
what was received (extracted programmatically, not retyped):

- `TB_RULING_WINDOW_ANCHOR_BRIEF_I_SCOPE_RULING.md` (2026-08-21) — Items 1/2/3 **APPROVED**;
  Item 2's timing / gate / scope-fence constraints; the three-layer pin with **capability
  before policy** at the worker seam; nine-site clamp census governs; RC-1 carry-forward only.
- `TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md` (2026-08-22) — **CODE REVIEW PASSED,
  APPROVED FOR COMMIT, Brief I NOT yet finally accepted.** All three census corrections
  ratified; M8 INVALID-BY-SCOPE accepted and **transferred intact to Brief II** (must mutate
  the production control-era ceiling 100 → 149 once the resolver exists); §12 reconciliation
  accepted under the wording constraint above; Gate 22 / W-NO-WEAKENING transients accepted
  with **no allowlist widening**; mutation record 14/1/0 accepted; Item 2 chapter treatment
  accepted with no unqualified `REPAIRED` until Brief II is also accepted.

**New standing rule adopted this session — `EXEC-PIN-1`:** when an authorized change alters the
schema, signature or callable contract of a live name, every commit-pinned Python source arm
that resolves that name from a live namespace must be re-evaluated before acceptance; any
translation must be test-local, preserve the historical pinned source, and document the bridge.
A coincidental empty intersection is evidence of present compatibility, not proof of permanent
isolation. Resolved-name sets to be derived mechanically (`symtable`), not from memory. It sits
alongside SR-1 and SR-2 and is **not** yet recorded in the skill.

**Post-commit closure still owed before Brief I is ACCEPTED:** (1) confirm the committed hash and
clean tracked tree, rerun the two commit-sensitive suites so Gate 22 and W-NO-WEAKENING clear
without allowlist changes, rerun the Brief-I suite and both scope proofs from the committed hash;
(2) run the committed-tree fleet / production-shape proof — `G-PROD-SHAPE` is **NOT RUN /
UNAVAILABLE**, and identical kernel hashes do not replace host/worker schema parity; (3) report
that outcome against the Brief-I commit hash. The Phase-7 soak is non-certifying and cannot
substitute.

## Not done, deliberately

Brief II surfaces untouched (Optuna, registry, `distributed_config.json`, NPZ generation
metadata, `WindowConfig.offset`, AC4). `G-PROD-SHAPE` **NOT RUN**. Follow-up debt recorded in
the report §15.2, not repaired.

## fallback parity

`fallback parity: code=[UNKNOWN — not measured this session], env=[UNKNOWN]` — pass 2 needs
`.127` booted and Zeus runs one OS at a time; unchanged by this work.
