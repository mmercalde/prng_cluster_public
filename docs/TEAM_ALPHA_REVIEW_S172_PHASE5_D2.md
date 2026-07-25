# TEAM_ALPHA_REVIEW_S172_PHASE5_D2.md

**Subject:** Team Alpha code-level review of the D2 implementation
(directional uniqueness at both enforcement layers)
**Spec:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D2.md`
**Base:** HEAD `b9c6120`
**Artifacts:** `tests/test_s172_phase5_d2_directional_uniqueness.py` (912 lines),
diff (gate-22 whitelist, 7 lines), status.
**Verdict: APPROVED — recommend Team Beta review for commit. No production
change; one judgment call verified and endorsed (§3).**

## 1. Scope

Diff touches ONLY gate-22's whitelist, registering the one new test path with
a per-line rationale. No production file changed — as required (D2-A drives
the unchanged serve path; D2-B probes the unchanged D1.1 writer). Status clean
(briefs/tmp and the two new docs pre-existing/expected). AST-parses.

## 2. Both gates faithful to the two-layer design

**D2-A** drives the FULL real serve path: `run_trial_miner` (default,
`worker_pool_size=1`) with a misbehaving `_OverlapWorker` that speaks the real
`MinerFramedSocket` wire (register → assign → three inline sub-stripes →
StripeComplete) declaring `[0,10) [9,19) [20,30)` — overlap at 9, gap at 19,
seed-sum preserved at 30. This correctly exercises the `eligible_provider is
not None` routing the brief flagged (§3): a bare `finalize_stripe` would have
parked the stripe in staging. Observer probes wrap `stage_inline_shard`,
`finalize_stripe`, `handle_stripe_failure` by delegation only (never altering
returns). All six assertion groups present: three shards staged+verified;
publish-count zero / zero manifests / zero run state; matrix routed exactly
once (`retryable=True`, `fail_trial`, `constant_phase`); trial `aborted`,
abort discharged once as `{run_id}:abort`, cleanup `done`; no assembly.

**D2-B** is a correctly labeled `DIRECT SINK INVARIANT-BREAK PROBE`: a valid
`{1,2}` set delivered through the public `publish_shard`, with seed 5
duplicated in phase-1 forward/constant at 0.90 vs 0.11, distinct event_ids AND
distinct `(stripe_id, sub_index)` slots. The observing `_DirectSinkProbe`
captures only the first `DirectionalDuplicateError` under a lock and re-raises
unchanged; the gate then asserts the REAL coordinator returned
`delivery=="failed"` / `event_id=="d2b:commit"` (observer did not alter the
lifecycle), all 13 structured attributes at exact expected values, and the
full fail-closed state (no assembly, no consumed marker, 4 manifests retained,
staged files intact, no `.npz`). The determinism pin asserts
`_sort_key(first) < _sort_key(dup)` under the engine's exact
`(workflow_phase, stripe_id, sub_index, attempt, event_id)` key.

**NC1-4** correctly scope the invariant: NC1 asserts the cross-direction seed
lands in BOTH maps and `bidirectional_constant` (keyed on
`(direction, skip_mode)`, no leak); NC2 disjoint-commits; NC3
constant+variable same seed commits; NC4 same-slot/different-event is
`ManifestReplayConflict` before assembly (intentional G7 overlap, flagged).

## 3. Judgment call — verified and endorsed

The brief's §3 assertion list named `all_verified == True` on the live finalize
check. Claude Code split it rather than assert it directly, on the stated
grounds that it is timing-dependent on the real serve path. **Team Alpha
verified this against source** (`finalize_stripe`, coordinator:1869-1905, and
its lifecycle-locked staging/dispatch comment at :1885): the dispatch thread's
finalize can legitimately observe `all_verified=False` when the last staging
job has not yet landed, with a later staging-completion finalize seeing it
True — so a direct live `all_verified==True` assertion would be racy. The
split is the correct resolution and strengthens the gate rather than weakening
it: (2a) a deterministic direct `evaluate_stripe_completion` reconstruction
over the fixture proves `coverage_ok` is the SOLE red predicate with all
shards verified (`substripes_match ∧ seed_sum_match ∧ survivor_sum_match ∧
all_verified ∧ ¬coverage_ok`); (2b) the live run separately asserts the
definitive routing fired on `substripes_match ∧ ¬reconciled ∧ ¬coverage_ok`.
Nothing was dropped; no sleep was introduced. Endorsed.

## 4. Mechanical verification (Team Alpha sandbox, pristine `b9c6120`)

- Full harness: **7/7 green** (D2-A, D2-B, NC1-4, NR) — independent
  reproduction. NR inside: D1.1 18/18, D1.0 8/8, D0 12/12, Phase 4 63/63,
  Phase 3 17/17.
- **Independent mutant — silent keep-LAST** (a third resolution behavior;
  Claude Code tested `max()` and keep-first): D2-A stayed **GREEN**, D2-B went
  **RED** with the exact `delivery:"done"` signature, and it additionally
  red-flagged D1.1's G14 in the NR suite — confirming the writer's raise is
  load-bearing across BOTH harnesses and the two D2 barriers are independent.
  Writer restored byte-identical (`diff` clean). This completes the
  resolution-behavior kill set: max / keep-first (Claude Code) + keep-last
  (Team Alpha).

## 5. Recommendation

Submit to Team Beta for commit authorization. Production unchanged; harness
approved. On approval: commit `tests/test_s172_phase5_d2_directional_uniqueness.py`
+ the gate-22 whitelist edit + `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D2.md`
+ `docs/PHASE6_PREREQS.md` + this memo + the session changelog, dual-push.
D3 begins against the new HEAD.

— Team Alpha (Claude), 2026-07-24
