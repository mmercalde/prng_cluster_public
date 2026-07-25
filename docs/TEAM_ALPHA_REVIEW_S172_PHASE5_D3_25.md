# TEAM_ALPHA_REVIEW_S172_PHASE5_D3_25.md

**Subject:** Team Alpha code-level review of the D3.25 implementation
(mode-preserving backend result contract + canonical candidate-ingress
normalization)
**Spec:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_25.md` REV3
**Base:** HEAD `c207e3a`
**Artifacts:** `utils/canonical_records.py` (new),
`tests/test_s172_phase5_d3_25_candidate_ingress.py` (new), diff (5 files, 433
changed lines), status.
**Verdict: APPROVED — Team Beta's two commit conditions are discharged and
independently re-verified (§8). Production files byte-identical throughout the
correction round; D3.25 gate 13/13. Commit is authorized.**

## 1. Scope

Five modified files, all inside REV3 §6's may-modify list
(`miner/range_miner_npz_writer.py`, `persistent_worker_coordinator.py`,
`zmq_sqlite_coordinator.py`, `window_optimizer_integration_final.py`,
gate-22 registration) plus the two new files. Nothing in the must-not-modify
list was touched. This is the largest production footprint of any Phase 5
deliverable, and it is the deliverable that ends the "PWC/ZMQ untouched" era —
see §6.1.

## 2. The extraction is verbatim — proven structurally, not merely by green tests

D1.1 18/18 and D2 7/7 staying green is necessary but not sufficient. Team Alpha
diffed the extracted function against the original at `c207e3a`
(comment-stripped, name-normalized): **16 differing lines, all accounted for** —
the signature rename Beta's API ruling required (`fwd_map`→`forward_map`,
`ctx`→`context`, `Dict`→`Mapping`), docstring additions, and one aliasing line:

```python
fwd_map, rev_map, ctx = forward_map, reverse_map, context
```

That alias leaves the **entire computational body byte-identical** to the
original — the safest available extraction shape: rename at the boundary, alias
immediately, never touch the arithmetic. `CANONICAL_RECORD_FIELDS` relocated
beside it and is re-exported from the writer; no `WindowConfig` dependency; no
`miner/` import; D1's shared-`sessions` reference behavior untouched per REV3
§3.4.

Team Alpha additionally mutated the alias line itself (**MX1**: swap
`forward_map`/`reverse_map`) — **killed** by G2, G3 and G9, confirming the
extraction seam is protected rather than merely exercised.

## 3. Both pruned early-returns now carry the complete v2 shape

Verified in the diff: PWC's `forward_zero` return (`:1621`, previously **no map
keys at all**) and ZMQ's (`:1091`, previously the generic pair with no variable
keys) both now build through `build_trial_populations` with all four maps passed
explicitly. Both variable maps were hoisted to constant-pair scope so the shape
never varies by execution path, and both assemble through the same constructor,
which egress-validates the intersection invariant. A missing field fails closed
at both boundaries rather than defaulting to empty — which is precisely the
failure mode the version stamp exists to make impossible.

## 4. The ingress wall fires before any accumulator mutation

Diff ordering confirms `validate_trial_populations(pw_result,
origin="adapter-ingress")` precedes `normalize_trial_populations` and both
`accumulator['bidirectional'].extend(...)` calls. Team Alpha proved the gate
enforces the ordering rather than merely the presence of the call
(**MX2**: move the wall *after* the extends) — **killed** by G4 with the exact
intended diagnostic:

```
AssertionError: accumulator mutated before the consistency wall fired: 1 -> 4
(field bidirectional_constant)
```

The `:276` set union is gone; per-mode normalization, trial-major/mode-minor
ordering, integer `skip_range` and `list[str]` sessions with a defensive copy
are all in place.

## 5. Mechanical verification (Team Alpha sandbox, pristine `c207e3a`)

- **D3.25 gate: 12/12 green — independently reproduced.** The suite aborts in a
  GPU-less environment on a module-level CuPy import; Team Alpha stubbed CuPy
  and the gate ran to completion, confirming G1's fake sieve backend needs no
  GPU. (Worth recording: the CuPy dependency is an import barrier, not a compute
  requirement, so this suite *is* reproducible off-rig.)
- **D1.1 18/18 and D2 7/7 green** in the sandbox — the extraction proof.
- Claude Code additionally captured the pre-edit baseline green at `c207e3a`
  (Phase 3 17/17, Phase 4 63/63, D0 12/12, D1.0 8/8, D1.1 18/18, D2 7/7, D3.0
  10/10, D3 10/10), the **pre-fix capture (14 RED / 2 GREEN)** documenting all
  four REV3 §0.2 defects against the unmodified adapter, and 14/14 mutants
  killed.
- Team Alpha independent mutants: **MX1** (extraction alias swap) → killed by
  G2/G3/G9; **MX2** (wall-after-mutation) → killed by G4.
- **Pre-existing failure claim verified:** `test_persistent_worker_harness.py`
  fails identically on the patched tree and on pristine `c207e3a`
  (`T20: FileNotFoundError: 'daily3.json'` — a missing data file). Not caused by
  this work, and not in the blocking NR list.

## 6. Items requiring explicit Team Beta rulings

**6.1 Gate 22 no longer asserts PWC/ZMQ are unmodified.** REV3 §6 lists both as
may-modify, so the Phase-4-era coexistence claim is superseded by construction;
both are now whitelisted, and `persistent/pwc_protocol.py` stays
asserted-unmodified. Team Alpha agrees this is the correct consequence but
believes it should be an **explicit** ruling rather than an inherited one — it
is the first deliverable to modify a live distribution coordinator.

**6.2 The miner call site was detached, not fed a v2 result.** `:426` now
routes to a new `_build_test_result_from_miner` rather than through the shared
adapter. This is a production change touching the miner path, which REV3 §4
assigns to D6. **Team Alpha verified the reasoning and endorses it:**

- It is *necessary*: the new ingress wall would fail closed on a `serve_trial`
  dict for the wrong reason, and routing miner output through the PWC/ZMQ
  contract is exactly what §4 forbids.
- It is *behavior-preserving*, and Team Alpha verified the load-bearing claim
  independently: `serve_trial` returns exactly `run_id, state, committed,
  workers_registered, stripes, manifests, bound_addr` — **no population keys
  whatsoever**. The old shared adapter's `.get(..., set()/{})` reads therefore
  already produced zero counts and appended zero candidates. The detached
  function preserves that exactly, including the threshold-gated flush so
  cadence does not shift.

The tension is genuine and originates in REV3 (§0.1 names the miner as an
adapter consumer while §4 forbids routing it through the new contract), so a
STOP would also have been defensible. Claude Code flagged rather than buried it.
Beta to confirm the resolution stands and that D6 still owns real miner
candidate ingress.

**6.3 Producer egress is validated functionally, not on real rigs.** No GPU run
was performed; rig acceptance remains Phase 6. Consistent with prior
deliverables, recorded for the certification trail.

## 8. Team Beta commit conditions — discharged

**Condition 4 — explicit miner-isolation gate.** The pre-correction 12/12 did
**not** cover it: nothing exercised `_build_test_result_from_miner`, and the
only `miner` reference in the harness was the `CANONICAL_RECORD_FIELDS`
re-export check. So a new gate was required, not a citation. **G13** was added
and is stronger than the ruling's minimum on two counts: it asserts isolation
**behaviorally** (baiting the input with a fully-formed `MinerTrialAssembly`
plus both canonical-record lists and proving none is consumed) **and at source
level** (the function body references none of `MinerTrialAssembly`, the two
canonical-record keys, `assemble_trial`, the v2 normalizer/validator,
`build_mode_records`, or spool/manifest reads, and performs no
append/extend) — the source check catches a future edit that adds ingress even
if behavior coincidentally stays at zero. Flush parity is pinned against a
hand-transcribed pre-D3.25 oracle asserting **exactly one** call *and* the
`chunk/trial-{n}` label, not mere presence; `accumulator=None` is covered as
zero flushes.

**Team Alpha independent re-verification:** both G13 mutants re-injected
separately, each killed with distinct attribution —
appends-a-candidate → `delta 1 != 0 — the miner path appended a candidate`;
drops-the-flush → `called 0 time(s)`, which the first mutant does not produce.
Gate 13/13 clean; the four production files and `utils/canonical_records.py`
are **byte-identical to the versions reviewed in §1-§5** (verified by diff), so
no drift occurred during the correction round.

**Condition 5 — changelog naming.** `SESSION_CHANGELOG_20260725_S179.md` →
`SESSION_CHANGELOG_20260725_PHASE5_D3_25.md`, confirmed present in `git status`
with the old name gone.

## 7. Recommendation

Approve for commit. Commit scope: `utils/canonical_records.py`,
`tests/test_s172_phase5_d3_25_candidate_ingress.py`, the four modified
production files, the gate-22 registration, the REV3 brief, this memo, and the
session changelog (Claude Code wrote
`docs/SESSION_CHANGELOG_20260725_S179.md`; Team Alpha suggests renaming to the
`_PHASE5_D3_25` convention used by the rest of the phase).

— Team Alpha (Claude), 2026-07-25
