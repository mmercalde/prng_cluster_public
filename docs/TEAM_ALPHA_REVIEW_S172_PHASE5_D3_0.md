# TEAM_ALPHA_REVIEW_S172_PHASE5_D3_0.md

**Subject:** Team Alpha code-level review of the D3.0 implementation
(canonical encoding seam + rectangular empty NPZ)
**Spec:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_0.md`
**Base:** HEAD `2d37b77`
**Artifacts:** diff (180 lines, 3 files, 5 hunks),
`tests/test_s172_phase5_d3_0_encoding_contract.py` (514 lines), status.
**Verdict: APPROVED — production implementation correct and unchanged; the one
harness gap (E8 order enforcement) was corrected in a Claude Code correction
round and re-verified by Team Alpha (§4). All Team Beta §9 commit conditions are
satisfied. Four follow-up findings escalated (§5), one of which closes the
blocked D3.5 provenance question for TFM's own data.**

## 1. Scope

Exactly the three expected files: `convert_survivors_to_binary.py`,
`window_optimizer_integration_final.py`, and gate-22's whitelist. Status
otherwise clean (briefs/tmp/docs pre-existing). AST-verified.

The **untouched-accumulator claim holds structurally, not merely by
assertion:** the `window_optimizer_integration_final.py` diff is two hunks —
the table deletion plus two seam-local imports at `:1711-1716`, and the two
encode call sites at `:1755-1758`. The second hunk terminates immediately after
those two lines; the merge, supersede, backfill, sort and dual-write logic are
outside the diff entirely and cannot have been touched.

Gate-22 registration adds `convert_survivors_to_binary.py` and the new test;
`window_optimizer_integration_final.py` was already listed from an earlier
deliverable. Per Beta's extended standing rule this covers new deliverable
files, and the production-file registration is explicitly commented and
reported — Team Alpha reads this as within the rule and flags it for
confirmation.

## 2. Implementation — faithful to the brief

- Both local tables deleted; `utils/prng_encoding` is the sole source. The
  module re-exports `PRNG_TYPE_ENCODING`/`SKIP_MODE_ENCODING` from canonical so
  `convert_survivors_to_binary`'s public surface survives for external
  referrers — a good call not specified in the brief.
- The `prng_type → prng_base → 'java_lcg'` resolution chain is kept
  **verbatim** at both call sites, per §3.1.5; only the encode step changed, so
  an unresolvable identity now raises instead of silently becoming 0. The
  canonical `ValueError` propagates unwrapped, per D1.1 Ruling B.
- `_EMPTY_NPZ_DTYPES` carries all 22 arrays with the frozen dtypes, including
  the deliberate `float32` on the six logically-integral `*_count` arrays.
- **Independently verified two things the diff cannot show:** (a) `numpy` is
  imported before the module-level dtype dict (offset 1087 < 2285) — no import
  ordering hazard; (b) the empty-path key order is **byte-identical** to the
  non-empty `savez_compressed` call order (22 keys, same sequence). Production
  order is correct.

## 3. Mechanical verification (Team Alpha sandbox, pristine `2d37b77`)

- Patched tree: **10/10** D3.0 gate checks green — independent reproduction.
  Claude Code additionally captured the NR baseline green pre-edit (Phase 3
  17/17, Phase 4 63/63, D0 12/12, D1.0 8/8, D1.1 18/18, D2 7/7, Phase 0 8/8,
  Phase 1/2 6/6 6/6) with all suites green post-fix, and the required pre-fix
  red capture (E2/E8/E10, plus E3/E4/E5/E6) at 3/10.
- **Team Alpha independent mutants:**

| mutant | result |
|---|---|
| M1 restore `.get(..., 0)` fallback | **killed** — E4 + E6 red |
| M2 drop one array from `_EMPTY_NPZ_DTYPES` | **killed** — E8 red |
| M3 wrong dtype (`forward_count` → int32) | **killed** — E8 red |
| M4 swap two adjacent keys in the empty-path order | **SURVIVED — 10/10 green** |

M1's profile is worth noting as evidence the gates are *precise*: restoring the
fallback while keeping the canonical table reds E4/E6 (unknown must raise) but
correctly leaves E2 green, because the canonical table does contain
`java_lcg_hybrid`. The mutant reintroduces the fallback, not the legacy table,
and the gates distinguish exactly that.

## 4. Harness gap (found, corrected, re-verified) — E8 order enforcement

**Finding.** M4 swapped `forward_matches`/`reverse_matches` in the empty-path
emission order — key set and dtypes intact, on-disk sequence changed — and the
full gate stayed **10/10 green**. E8 compared as an unordered mapping while the
harness header claimed its oracle covered *"22 array names / order / dtypes …
hand-transcribed from the on-disk contract"* (`:24-25`, restated `:79-80`). Same
class as D1.1's circular G9: stated scope exceeding actual assertion. Harness
gap only — the emitted order was independently confirmed correct (§2).

**Correction (test-only).** `E8_EXPECTED_KEY_ORDER` — a literal 22-tuple,
verified by Team Alpha as **not** derived from `_EMPTY_NPZ_DTYPES`,
`NPZ_CONTRACT`, `NPZ_KEYS`, or any production constant, with a comment
forbidding future deduping against them. E8 now asserts
`tuple(z.files) == E8_EXPECTED_KEY_ORDER` — read inside the loaded-NPZ context,
i.e. physical on-disk entry order — alongside the existing count / key-set /
dtype / zero-length assertions, reporting the first divergent index.

**Team Alpha re-verification (pristine `2d37b77` + reviewed diff):** both
production files confirmed **byte-identical** to the reviewed versions; clean
run **10/10**; M4 re-injected using an **independent key pair**
(`forward_matches`/`reverse_matches` — Claude Code's own proof used
`skip_min`/`skip_max`) → **killed**, E8 red at 9/10 reporting first divergence
at index 1; production restored byte-identical, final run 10/10. Closed.

**Team Alpha process note.** The first re-verification attempt reported E8 red
against a clean source file. Cause was a stale `__pycache__` in the Team Alpha
sandbox retaining bytecode compiled from the earlier mutant, not any defect in
the correction — the imported module's dict order differed from the source
file's. Recorded because the review discipline applies to the reviewer's own
tooling: mutation rounds must purge bytecode caches on restore, and a
surprising red must be traced before it is reported.

## 5. Findings escalated to Team Beta

**5.1 The write/read asymmetry was already live — and it is broader than the
hybrid collapse.** `utils/survivor_loader.py:45-55` re-exports the canonical
`PRNG_TYPE_DECODING`: Phase 0 reached the **reader** but not the writers. Every
NPZ on disk was written in the legacy space and is read in the canonical one.
Team Alpha quantified it — **9 of 12 legacy ids misread**; only 0, 7, 8
round-trip:

```
written java_lcg_reverse(1)     -> decodes java_lcg_hybrid
written mt19937(2)              -> decodes java_lcg_hybrid_reverse
written mt19937_reverse(3)      -> decodes java_lcg_reverse
written xorshift128(4)          -> decodes lcg32
written xorshift128_reverse(5)  -> decodes lcg32_hybrid
written lcg32(6)                -> decodes lcg32_hybrid_reverse
written minstd_reverse(9)       -> decodes minstd_hybrid
written randu(10)               -> decodes minstd_hybrid_reverse
written randu_reverse(11)       -> decodes minstd_reverse
```

**RESOLVED for TFM's actual data — no migration is required.** The `prng_type`
column is a **mode label**, not a directional family: writers emit `prng_base`
for constant and `prng_base + "_hybrid"` for variable. Team Alpha inspected the
live accumulators on VM 101:

```
bidirectional_survivors_all.npz      rows=20949  prng_type={0}  skip_mode={0}
bidirectional_survivors_binary.npz   rows=20949  prng_type={0}  skip_mode={0}
```

Every accumulated record is java_lcg **constant**. There are no hybrid records
at all, so the collapse-to-zero defect never bit this data. Legacy wrote `0`
for `java_lcg` and canonical reads `0` as `java_lcg` — id 0 is one of the three
that round-trips. **The existing accumulator is already canonical-compatible,
provable by inspection rather than provenance inference.**

Consequently the mixed-encoding hazard does not materialize either: post-D3.0 a
constant java_lcg record still encodes to `0` (identical to legacy) and a
hybrid record would encode to `1`, a value absent from the prior file, so
concatenation is unambiguous in both directions. **No operational hold on
post-D3.0 Step-1 runs is required for these files.**

**Ruling requested (F):** Beta's three options were migrate / regenerate /
reject-and-start-clean. For this accumulator the correct disposition is a
fourth — **"verified already-canonical, no action"**, established by direct
inspection of the observed value sets. Beta's general-case policy and its
required guard gate (proving a legacy-coded prior cannot be merged unnoticed)
should still be built in D3.5; the blocking work item on TFM's own data is
closed.

**Planning consequence — no historical hybrid baseline exists.** The absence of
`skip_mode=1` rows means either no hybrid run ever reached the accumulator or
hybrid runs yielded zero bidirectional survivors (both plausible; also
consistent with the union defect discarding cross-mode constant candidates
while variable-only survivors never occurred). Either way, D3.25's and Phase
6's both-mode comparisons will establish that surface for the first time with
**nothing to regress against**. This is a Phase 6 planning fact, not a defect.

Housekeeping: `bidirectional_survivors_all.npz.ckpt.tmp.npz` (Mar 22) and
`.flush.tmp.npz` (May 1) carry no `prng_type` column — stale partial artifacts,
candidates for the S110 root-cleanup backlog item.

**5.2 The meta sidecar is now a provenance marker.** Claude Code's judgment
call — dumping the canonical 44-key map into the sidecar's `encodings` block
rather than the legacy 12, because recording the legacy table while writing
canonical ids would be untruthful — is endorsed. Team Alpha adds that this
makes the sidecar exactly the **writer-version provenance artifact Beta said
D3.5 requires**, for all NPZs written from D3.0 onward (not retroactively).
Beta may wish to formalize it as the provenance mechanism.

**5.3 Three more divergent copies exist outside D3.0's scope.** Verified:
`window_optimizer_bayesian.py:231-235` carries a **6-entry** `_PRNG_ENC` and
writes **both** `bidirectional_survivors_all.npz` and
`bidirectional_survivors_binary.npz` (`:236-237`) — Claude Code's report was
accurate. It is not on the live Step-1 path (`BayesianOptimization` resolves
from `window_optimizer.py`; only `dry_run_s115.py` imports it) but would
reinstate the bug if ever wired in. Additionally
`apply_s145r1_npz_accumulator.py` and `apply_s149_npz_checkpoint.py` bake the
legacy 12-entry table into their **patch text** — re-running either reinstates
the defect; `docs/window_optimizer_integration_final.py` is a stale copy; and
`harness_npz/test_npz_merge.py` carries a copied 5-entry oracle. Recommend a
narrow follow-up deliverable (**D3.0-B**) to purge these, sequenced at Beta's
discretion; none blocks D3.

**5.4 One residual silent default, flagged not fixed.** Keeping the legacy
resolution chain means a record carrying **neither** `prng_type` nor
`prng_base` still becomes `'java_lcg'` silently, rather than failing closed.
This is faithful to the brief (§3.1.5 directed preserving the chain, and D3.0
is behavior-preserving except where specified) and Claude Code correctly
declined to use canonical `resolve_prng_type()` for that reason. But it is a
silent default of precisely the class Beta has ruled against elsewhere. Team
Alpha recommends D3/D3.25 require newly produced records to carry explicit
identity and fail closed, leaving the chain only for genuinely historical
inputs.

## 6. Team Beta §9 commit conditions — all satisfied

| condition | status |
|---|---|
| E8 directly asserts `tuple(z.files)` order | YES — literal independent oracle, verified not derived from any production constant |
| M4 reordered-key mutant killed | YES — independent key pair, E8 red 9/10, first divergence index 1 |
| D3.0 gate = 10/10 | YES — reproduced in the Team Alpha sandbox |
| all blocking non-regression green | YES — D2 7/7, D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4 63/63, Phase 3 17/17, Phase 0 8/8, Phase 1/2 6/6 6/6 |
| production diff unchanged from reviewed version | YES — both files byte-identical |
| git status only authorized + pre-existing files | YES — 3 modified (2 production + gate-22), 1 new harness, docs |

D3.0 is cleared for commit. Findings 5.1-5.4 are for Beta's rulings and do not
block it.

— Team Alpha (Claude), 2026-07-24
