# TEAM_ALPHA_REVIEW_S172_PHASE5_D4.md

**Subject:** Team Alpha review of the D4 implementation (`serial_reference`
assembly backend behind the two-backend interface)
**Spec:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D4.md` REV3
**Base:** HEAD `f163199`
**Verdict: APPROVED — recommend Team Beta review for commit. No correction round
required.**

---

## 1. Scope

```text
M  tests/test_s172_phase4_coordinator.py     gate-22 registration, 19 insertions
?? miner/assembly_backends.py                new, 281 lines
?? tests/test_s172_phase5_d4_serial_backend.py  new
?? docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D4.md
?? docs/SESSION_CHANGELOG_20260726_PHASE5_D4.md
```

No production module modified. Claude Code verified every must-not-modify file
byte-identical to `f163199` by SHA — `range_miner_npz_writer.py`,
`canonical_arrays.py`, `canonical_records.py`, `run_finalizer.py`,
`prng_encoding.py`, `window_optimizer_integration_final.py`,
`persistent_worker_coordinator.py`, `zmq_sqlite_coordinator.py`,
`convert_survivors_to_binary.py`, and all seven D0-D3.5 harnesses. No stop
condition hit: the seam needed no change to any existing module, which was §0's
central bet.

## 2. The three REV3 corrections — AST-verified, not read

**[B5] measurement order.** Verified by parsing the module and locating the real
`assemble` (the Protocol stub has an empty body and must be excluded):

```text
perf_counter    line 225
assemble_trial  line 228
sum(...)        line 240      -> measurement AFTER delegation: YES
delegation wrapped in try/except: NO — the exception propagates unchanged
```

The inline comment at the `sum` states why it is a direct subscript: reaching
that line means D1.1 has already validated every manifest, so `KeyError` is
impossible there, whereas computing it before delegation would raise one
instead of the canonical `SpoolIdentityError`.

**[B4] input contract.** The annotation is exactly `List[Dict[str, Any]]`, and
nothing is copied, normalized or converted before delegation. The docstring
cites both the declaration (`range_miner_npz_writer.py:450`) and the
enforcement (`isinstance(manifest, dict)`, `:280`).

**[B2] G3 timing.** Compares the twelve stable fields; for `timing` asserts
structure, finiteness and `> 0` only, plus `sorted(timing) == ("assembly_s",)`
proving no key was added.

## 3. No reimplementation — AST-confirmed

Independent parse of `miner/assembly_backends.py`:

```text
defines its own assemble_trial : NO
forbidden calls present        : NONE
  (open, sorted, records_to_arrays, build_mode_records, finalize_run)
```

`finalize_run` is deliberately not even imported — the module docstring records
that a backend produces a `MinerTrialAssembly` and stops, and the finalizer is
the caller's next step. G7 is genuinely AST-based (12 `ast` references,
`ast.parse` at `:706`), not the fragile substring search REV3 [B6] prohibited.

## 4. Mechanical verification (Team Alpha sandbox, pristine `f163199`)

**D4 gate: 8/8 green — independently reproduced.** Claude Code additionally
reported the pre-edit baseline green at `f163199` and all ten blocking suites
unchanged after the edit (228/228 both sides), with 9/9 mutants red.

**Team Alpha independent mutants:**

| mutant | result |
|---|---|
| **MB** widen the annotation to `Sequence[Mapping]` ([B4] violation) | **killed by G1**, 7/8, message names the widening explicitly |
| **MA** compute `spool_bytes_read` before delegation ([B5] violation) | **killed by G8** — see below |

**MA's attribution is the noteworthy result.** It reds with
`M6b: anchor is not unique (0 occurrences) — the mutation would be
unverifiable`. My edit displaced M6b's anchor text, and the `_patch()`
uniqueness assertion caught it: the harness **refused to run a mutation set it
could not verify** rather than reporting a vacuous pass. That is the hardening
added after §4.2's false-evidence finding doing exactly its job. Claude Code's
own M7 covers the [B5] path directly, reddening with
`expected SpoolIdentityError, got KeyError: 'expected_size'`.

## 5. The false-evidence finding — the most valuable thing in this deliverable

Claude Code's first mutation run produced **8/8 green with a fully populated
mutation table in which six rows credited a kill to an assertion that never
ran.** Each mutant loads as its own module and defines its own dataclasses, so
`isinstance(result, AB.BackendAssemblyResult)` failed on class identity before
the injected defect could execute. Every one died for the wrong reason.

It found this itself and fixed it by threading the mutant's own module into
`_g3_probe` / `_g6_probe`, then added two hardenings:

```text
_patch()  asserts its anchor is unique -> a non-applying mutation cannot pass vacuously
_record() fails with MUTANT SURVIVED   -> rather than silently skipping
```

Team Alpha endorses writing this up as a finding rather than a footnote. This is
the fourth distinct instance in Phase 5 of a gate that was green while proving
something other than what it claimed — after D1.1's circular G9, D3.0's
unenforced E8 order, and D3's `score`-bound gap. The pattern is consistent
enough to be worth naming: **a green mutation table is not evidence until the
mutants are shown to have actually applied and actually reached the assertion
they are credited to.**

## 6. Two notes Claude Code raised

**G5's hand-computed literals interleave the modes** — seed order 1(c), 2(v),
12(c), 15(v) — so a per-mode concatenation or mode-major ordering would show.
Deliberate and correct: the finalizer's L2/L3 own ordering, and a backend that
quietly grouped by mode would otherwise pass.

**D5's `peak_rss_bytes` obligation is recorded.** §17 requires peak aggregate
host RAM of parent plus concurrently live workers, and `RUSAGE_CHILDREN`
reports the maximum of any single reaped child, not a concurrent sum. D5 must
supply a compliant measurement; D4's field proves the telemetry exists, and the
authoritative §17 numbers come from the isolated Phase-6 benchmark.

## 7. Recommendation

Approve for commit. Suggested scope: `miner/assembly_backends.py`,
`tests/test_s172_phase5_d4_serial_backend.py`,
`tests/test_s172_phase4_coordinator.py`, the REV3 brief, this memo, and
`docs/SESSION_CHANGELOG_20260726_PHASE5_D4.md`.

— Team Alpha (Claude), 2026-07-26
