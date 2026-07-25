# SESSION_CHANGELOG_20260725_PHASE5_D3.md

**Session scope:** S172 Phase 5 — D3 (shared backend-neutral 24→22 columnizer +
independent structural validator): brief REV1→REV3, implementation, review, two
test-only correction rounds, approval.
**Base:** HEAD `66f0425` (D3.0 complete).

## Outcome

**D3 APPROVED FOR COMMIT.** Production module approved unchanged
(`md5 e3033e1ee523a188a7b631f572157b24`); two harness correction rounds closed
three mutation gaps in the same contract family.

## Brief arc — three revisions, one architectural error caught

- **REV1 → REV2.** Team Beta caught a self-contradiction: REV1 required both
  "preserve input order if valid" and shuffle-invariance within a mode. The
  deeper point was the load-bearing one — **D3 must own no ordering policy at
  all**, because mode-first sorting would later undo D3.5's required global
  seed order. Doctrine frozen at the top of the brief:

  ```text
  D3    converts rows.
  D3.25 orders candidate rows.
  D3.5  orders final winner rows.
  ```

  Also added: strict exact-24-key input (missing OR extra fails), identity
  consistency, `sessions`/`prng_base` validated despite not becoming arrays,
  Python-space numeric validation, validator 1-D + self-validation,
  `np.dtype`-normalized contract, Ruling G (fail-closed rates; legacy seams to
  D3.0-B).
- **REV2 → REV3.** Four further amendments: **[A1]** `prng_base` restricted to
  a **forward, non-hybrid base family** — registry membership alone was
  insufficient, since `prng_base = "java_lcg_reverse"` with constant mode is
  equality-consistent but semantically invalid (`prng_type` is a *mode* label);
  **[A2]** destination-`float32` representability (`np.isfinite(np.float32(v))`
  — Python-level finiteness does not prove it) plus integer-valued counts;
  **[A3]** `Iterable` replaces `Sequence`; **[A4]** bound wording narrowed —
  only `bidirectional_selectivity` may exceed 1, `intersection_weight` is
  bounded by its own formula and Team Alpha's "unbounded weight metrics"
  phrasing was wrong.

## Delivered

- `utils/canonical_arrays.py` (470 lines) — `CANONICAL_ARRAY_CONTRACT` (22
  `(name, np.dtype)` pairs, frozen order), `records_to_arrays`,
  `validate_array_bundle`, two exception types. `BASE_PRNG_FAMILIES` is
  **derived** from `KERNEL_REGISTRY` by suffix-stripping (never hardcoded);
  `_INT_FIELD_RANGE` derived from `np.iinfo` so a contract dtype change cannot
  leave a stale literal bound.
- `tests/test_s172_phase5_d3_columnizer.py` — C1-C10 with an independent
  hand-transcribed oracle; C10 applies each mutant as a textual edit `exec`'d
  into a fresh namespace, leaving the on-disk file untouched.
- Gate-22 registration (9 lines, whitelist only).

**No production call site rewired** — independently verified: a repo-wide grep
for `canonical_arrays|records_to_arrays|validate_array_bundle` returns zero
hits outside the new module, its harness, and the whitelist entry. The inline
`_survivors_to_arrays` closure is still called at
`window_optimizer_integration_final.py:1786`; the `convert_survivors_to_binary`
array block is untouched.

## Review arc — three mutation gaps, same contract family

Claude Code killed **21/21** of its own mutants on the first pass, with sound
methodology (fresh-namespace `exec`; and it scoped the missing-key relaxations
to the *targeted* field after noticing a blanket relaxation killed them
incidentally via a raw `KeyError` on `seed` — evidence that would have proved
the wrong thing).

Team Alpha's four independent mutants found the gap family:

| mutant | first pass | after corrections |
|---|---|---|
| MA — invert the constant/variable identity rule | killed | — |
| MB — drop `_hybrid` from the suffix list (partial [A1] restriction) | killed **surgically by C5 alone** | — |
| MC2 — `score` reclassified, ceiling lost | **survived 10/10** | killed, attributed |
| LB — shared unit-interval lower bound removed | **survived 10/10** | killed, attributed |
| FWD — `forward_matches` reclassified, ceiling lost | **survived 10/10** | killed, attributed |

Three rows closed the family: `("score above 1.0", …)` in round one, then
`("forward_match_rate above 1.0", …)` and `("score below 0.0", …)` in round
two. Team Beta corrected Team Alpha's "three rows" arithmetic to two additional
rows. The field-specificity assertion was also tightened to match the **quoted**
form (`repr(field)`) so a substring collision cannot satisfy it, with a
demonstration that a correctly-raising-but-unnamed rejection goes red.

**Attribution verified by differential comparison:** the LB run's C5 reds
contain `score below 0.0` and not `forward_match_rate above 1.0`; the FWD run's
contain `forward_match_rate above 1.0` and not `score below 0.0`. Neither is
killed by an unclassified-field assertion, missing-key failure, unrelated rate,
or unnamed exception.

**Process note:** Claude Code discovered the LB gap itself and **deliberately
did not fix it**, on the grounds that Beta's ruling specified one row and
unrequested gate expansion inside a Beta-directed round is Beta's call. Second
time this session it flagged rather than silently widened scope.

## numpy version finding (retained per Team Beta §4)

Three versions are in play: **VM101 venv `1.22.0`** (where the gate runs),
**VM101 system `2.2.6`**, **Team Alpha sandbox `2.4.4`**. On 1.22.0
`np.array([-1], dtype=np.uint32)` **wraps silently**; on 2.x it **raises**. The
explicit `np.iinfo`-based Python-space range checks are therefore part of the
production contract and **must not** later be removed in favour of relying on
numpy conversion behaviour. Non-blocking: on numpy 2.x the deliberate
float32-overflow case emits a cosmetic `RuntimeWarning: overflow encountered in
cast`.

## Verification record

D3 **10/10** (C10: 21/21 mutants killed) · D3.0 10/10 · D2 7/7 · D1.1 18/18 ·
D1.0 8/8 · D0 12/12 · Phase 4 63/63 · Phase 3 17/17 · Phase 0 8/8. Baseline
captured green at `66f0425` before any edit; independently reproduced in the
Team Alpha sandbox.

## Committed in this change

`utils/canonical_arrays.py`, `tests/test_s172_phase5_d3_columnizer.py`,
`tests/test_s172_phase4_coordinator.py` (whitelist only),
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3.md`,
`docs/TEAM_ALPHA_REVIEW_S172_PHASE5_D3.md`, this changelog.
Excluded per Team Beta: any D3.0-B, D3.25, D3.5, PWC, ZMQ, inline-writer,
standalone-writer or miner-adapter change; `docs/PHASE6_PREREQS.md` (still
awaiting its own review); pre-existing untracked briefs and `tmp/`.

## Next

**D3.25** — mode-preserving backend result contract + canonical candidate
ingress. Its brief is written (REV2) and needs its line citations rebased onto
the post-D3 HEAD; its `_mode_records` extraction target should be re-verified
at that point. Also open, non-blocking: **D3.0-B** (cross-direction match-rate
fallback in both active legacy writers, plus residual copied encoding tables —
must complete before Phase 6 certification), the Ruling-F provenance snapshot,
and `docs/PHASE6_PREREQS.md` review.
