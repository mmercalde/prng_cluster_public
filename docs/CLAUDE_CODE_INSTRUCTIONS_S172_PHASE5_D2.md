# CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D2.md

**S172 RANGE-MINER — Phase 5, Deliverable D2: directional uniqueness at BOTH
enforcement layers (producer overlap rejection + Phase-5 fail-closed probe)**

**Audience:** Claude Code on VM 101 (`michael@192.168.3.177`), in
`~/distributed_prng_analysis`. You write and iterate; you do NOT commit, push,
or run WATCHER. When the D2 gate + non-regression are green, STOP and report;
Team Alpha reviews against live source, Team Beta reviews, Michael commits +
dual-pushes.

**Frozen against HEAD `b9c6120`** (D1 complete: D1.0 workflow + terminal-race
corrections; D1.1 assembly engine + `AssemblingPhase5Sink`). Spec authority:
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md` REV5 §5.4/§8/G14
forward-reference, and the Team Beta corrected D2 brief absorbed verbatim
herein (its self-correction after live-source feasibility verification —
marked **[TB-D2-COR]** — replaces the original single-gate brief entirely).

---

## 0. Why D2 has TWO gates  **[TB-D2-COR]**

The invariant: within one `(run_id, workflow_phase, accepted_attempt,
family)`, a seed appears at most once. A duplicate is a producer/coverage
defect; Phase 5 must raise `DirectionalDuplicateError` and must NEVER keep the
max match rate or otherwise deduplicate.

The originally requested fixture — two overlapping shards traveling through a
legitimately accepted/reconciled attempt into the sink — is **impossible by
design**, verified at `b9c6120`:

- macro-stripes are disjoint contiguous tiles (`assign_stripes` :1783 →
  `partition_macro_stripes`);
- sub-stripe coverage must tile the stripe's ASSIGNED
  `[seed_start, seed_start+seed_count)` exactly — `_coverage_exact` (:311-322)
  walks a cursor: any shard not starting AT the cursor (gap OR overlap) →
  `False`, and the cursor must land exactly on the stripe end;
- the blocking predicate `is_complete` (:305-309) = `reconciled` (:295-302 =
  substripes_match AND seed_sum_match AND survivor_sum_match AND coverage_ok)
  AND all_verified; `finalize_stripe` (:1869) calls `publish_attempt` (:1913)
  ONLY when it holds — an unreconciled attempt publishes nothing;
- D1's §5.3 additionally rejects any survivor seed outside its own spool's
  declared range.

So D2 proves the two enforcement layers SEPARATELY: **D2-A** — the real
producer rejects the overlap before Phase 5 ever sees it; **D2-B** — Phase 5
still fails closed if that upstream barrier is ever bypassed or regresses.
This is strictly stronger than the original brief.

## 1. Non-negotiable working rules

1. **Read live source before every claim** — every cite above and below was
   verified at `b9c6120`; re-verify before depending on it. This deliverable
   exists in its current shape BECAUSE feasibility was checked against source
   (both reviewers initially got a layer wrong; the audit caught it).
2. **Each gate must FAIL on wrong behavior** — the mutation proof (§6) is the
   defining Rule-2 evidence for D2.
3. **No test-only shortcuts on the happy path**; D2-B's bypass is deliberate
   and labeled (§4).
4. **Expected production change: NONE.** If D2-A reveals that a malformed
   overlap IS published, or D2-B fails against the unchanged D1 writer, STOP
   and report — do not patch production without an Alpha/Beta ruling.
5. STOP at the gate. No commit/push/WATCHER. Do not begin D3.

## 2. Scope — one new Python file

- `tests/test_s172_phase5_d2_directional_uniqueness.py` — gates D2-A, D2-B,
  the four negative controls (§5), and the blocking NR runner (§7).
- Gate-22 whitelist: register the ONE new test path (registration-only, local
  format, reported) — pre-authorized under the extended Team Beta standing
  rule.
- The session changelog and Team Alpha review are produced at review time by
  Team Alpha, not by Claude Code.

## 3. D2-A — real producer overlap rejection

**Fixture — the CRITICAL requirement [verified :1897-1905]:** the
definitive-failure routing inside `finalize_stripe` fires ONLY when
`eligible_provider is not None` — i.e., only when the REAL serve/dispatch
lifecycle drives finalize. A bare `finalize_stripe()` call (the D1.1
`_build_run` pattern) evaluates the predicate but leaves the stripe **parked
in `staging` forever** — no matrix routing, no abort. D2-A must therefore
drive the FULL real serve path with a **misbehaving framed-socket worker**
(the D1.0 W2 `_FakeWorker` pattern): the worker itself declares the
overlapping sub-stripe ranges over the real `MinerFramedSocket` wire. That is
the realistic producer-defect vector.

Use a **constant phase** (the fail-closed constant-phase policy makes the
outcome unambiguous — no hybrid retry obscuring it). `test_both_modes=False`
is the natural workflow.

**The isolated-coverage construction** (preferred, per Team Beta): over a
stripe assigned `[0, 30)`, the worker reports exactly three sub-stripes:

```text
shard A: seed_start 0,  seed_count 10        # [0, 10)
shard B: seed_start 9,  seed_count 10        # [9, 19)   overlap at 9
shard C: seed_start 20, seed_count 10        # [20, 30)  compensating gap at 19
```

Total declared count = 30, so `seed_sum_match` stays True; sorted-by-start
cursor hits B's start 9 ≠ cursor 10 → `coverage_ok` False. Arrange
`StripeComplete.substripes_done == expected_substripes == 3` (control the
worker's advertised sub-stripe cap so the assignment expects 3) and set
`survivors_total` to the true survivor sum — so **coverage is provably the
ONLY red predicate**. Each shard's survivors lie within its OWN declared
range (D1 §5.3-consistent); the duplicate seed 9 may legitimately appear in
both A and B's survivor lists here — it never reaches Phase 5.

**Assertions:**
- both/all three shard payloads stage and hash-verify individually
  (`stage_inline_shard`-driven staging reports verified);
- `evaluate_stripe_completion` (:325) reports: `substripes_match == True`,
  `seed_sum_match == True`, `survivor_sum_match == True`,
  `coverage_ok == False`, and a `reasons` entry identifying gap/overlap;
- the attempt NEVER reaches `Phase5Sink.publish_shard`: sink publish-call
  count is **zero**; zero Phase-5 manifests accumulated;
- the failure routes through the real matrix exactly once
  (`_on_staging_failed(retryable=True)` :2695 → `handle_stripe_failure`
  :2835 → constant-phase branch → `fail_trial` :2966);
- final trial state `aborted`; sink abort discharge exactly **once**
  (`{run_id}:abort`); abort cleanup `"done"`;
- `get_assembly(run_id) is None`; no canonical assembly anywhere.

## 4. D2-B — Phase-5 defense-in-depth probe

**Label prominently: `DIRECT SINK INVARIANT-BREAK PROBE`** — the post-D1.0
producer cannot legitimately emit this input (§0); the probe deliberately
bypasses the upstream reconciliation barrier to prove the second layer.

Construct a complete, individually-valid manifest set for a full phase set
(`{1,2}` is sufficient) and deliver it through the PUBLIC
`sink.publish_shard(manifest)` surface (never by inserting into sink
internals; never by calling `assemble_trial` directly — that is D1.1's G14,
already green). Within ONE directional population (e.g. phase 1,
forward/constant), include TWO shards that:

- carry different `event_id`s AND different `(stripe_id, sub_index)` slots
  (so neither replay nor slot-conflict fires first);
- share run_id, workflow_phase, direction, skip_mode, family_name, attempt
  (attempt 0);
- are internally valid `s172_substripe_v1` payloads with correctly
  recomputed `expected_size`/`expected_sha256`, real staged files on disk;
- declare seed ranges that overlap on exactly ONE chosen seed (permitted
  here only because the probe bypasses reconciliation), each duplicate
  occurrence inside its own spool's declared range;
- give the duplicate seed **visibly different match rates** (so keep-first /
  keep-last / max-rate would each be observable in a mutant).

**Determinism pin:** the engine sorts insertions by
`(workflow_phase, stripe_id, sub_index, attempt, event_id)`
(`range_miner_npz_writer.assemble_trial` order key) — choose stripe/sub IDs
so the intended-FIRST shard sorts first under that EXACT key, and assert the
fixture's expected ordering explicitly (guards against a future sort-key
change silently swapping first/dup provenance).

**Observing sink (single-capture contract):** a thin subclass of
`AssemblingPhase5Sink` whose `commit_trial` calls `super().commit_trial()`,
captures ONLY the FIRST `DirectionalDuplicateError` (stored under the sink
lock so the gate reads a stable reference), and re-raises unchanged. Then
call **real `coordinator.commit_trial(run_id)`** and assert the observer
did not alter the lifecycle: returned `delivery == "failed"`, event
`event_id == f"{run_id}:commit"`.

**Assertions on the captured exception:** it is exactly
`DirectionalDuplicateError`; ALL 13 structured attributes correct —
`run_id, workflow_phase, direction, skip_mode, seed, first_stripe,
first_sub_index, first_attempt, first_match_rate, dup_stripe, dup_sub_index,
dup_attempt, dup_match_rate` — asserted as attributes with the exact
expected values (incl. both match rates), never via message text.

**State assertions:** `get_assembly(run_id) is None`; no consumed
commit-event marker; accumulated manifests retained (same-instance retry per
D1 §4.0/§4.3); staged files still present (never deleted merely because
delivery failed); no `.npz` file exists anywhere in the fixture tree; no
assembly ever carried a populated `binary_npz_path`/`all_npz_path` (note:
largely anticipatory until D3/D4 — the load-bearing assertion TODAY is **no
installation and no silent deduplication**).

## 5. Negative controls — the invariant is scoped, not global

1. The same seed in P1 forward AND P2 reverse: legitimate — assert it lands
   in BOTH directional maps AND appears in `bidirectional_constant` (the
   invariant is keyed on `(direction, skip_mode)`, no cross-direction leak).
2. Two disjoint shards in one directional population: commit succeeds.
3. The same seed in constant AND variable mode (P1 + P3, four-phase
   fixture): not a directional duplicate — commit succeeds.
4. Different event IDs claiming the SAME `(stripe_id, sub_index)` slot:
   `ManifestReplayConflict` BEFORE assembly — intentional overlap with
   D1.1's G7, retained per Team Beta so D2 reads standalone.

## 6. Mutation / discrimination proof — the defining Rule-2 evidence

Temporarily mutate the D1.1 writer's duplicate branch to
`pop_map[seed] = max(pop_map[seed], match_rate)`; separately (if practical)
to silent keep-first. Against each mutant:

```text
D2-A: stays GREEN  — the upstream overlap rejection is independent of the writer
D2-B: turns RED    — commit incorrectly succeeds / installs an assembly
```

Report the exact red signatures. Restore the writer and verify byte-identity
against a pre-mutation copy. This proves the two barriers are real AND
independent.

## 7. Blocking non-regression

D2 harness green, plus: D1.1 **18/18**, D1.0 **8/8**, D0 **12/12**, Phase 4
**63/63**, Phase 3 **17/17**. Capture the NR baseline green at `b9c6120`
BEFORE any edit. Any red STOPS work.

## 8. Stop conditions

- D2-A shows the overlap attempt IS published (producer defect) — STOP;
- D2-B fails against the unchanged D1.1 writer — STOP;
- the D2-A fixture cannot route the definitive failure through the real
  matrix without touching coordinator code — STOP and report the obstacle;
- any gate passes only by weakening it.

## 9. Kickoff & report

Implement, iterate to green
(`source ~/venvs/torch/bin/activate; PYTHONPATH=. python3
tests/test_s172_phase5_d2_directional_uniqueness.py`), then STOP and submit:
actual diff + status, complete command/output evidence, mutation evidence
(both mutants, both gates, restoration byte-check), for Team Alpha review.
