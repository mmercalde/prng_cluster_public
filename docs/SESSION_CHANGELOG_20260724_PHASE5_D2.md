# SESSION_CHANGELOG_20260724_PHASE5_D2.md

**Session scope:** S172 Phase 5 — D2 (directional uniqueness at both
enforcement layers): brief authoring after Team Beta's own feasibility
self-correction, implementation, Team Alpha review, Team Beta approval.
**Base:** HEAD `b9c6120` (D1 complete).

## Outcome

**D2 APPROVED FOR COMMIT by Team Beta**, excluding `docs/PHASE6_PREREQS.md`
(unreviewed — see §Deferred). No production change; harness-only deliverable.
D3 begins against the resulting new HEAD.

## Brief history — Beta's feasibility self-correction

The original D2 brief required two overlapping shards to travel through a
legitimately accepted/reconciled attempt and trigger
`DirectionalDuplicateError`. Team Alpha's source check found staging accepts
arbitrary declared ranges; Team Beta then verified the layer Alpha had NOT
checked — reconciliation — and withdrew its own fixture as **internally
contradictory**:

- macro-stripes are disjoint contiguous tiles;
- `_coverage_exact` (coordinator:311-322) walks a cursor against the stripe's
  ASSIGNED range — any shard not starting AT the cursor (gap OR overlap) →
  `False`;
- `is_complete` (:305-309) requires `reconciled` AND `all_verified`;
  `finalize_stripe` (:1869) publishes ONLY then;
- D1 §5.3 independently rejects survivor seeds outside their spool's range.

Corrected design: **two complementary gates**, proving each barrier
separately. Recorded process note: Alpha's earlier "feasibility confirmed on
source" verified the staging seam and extrapolated to the whole publication
path — the same extrapolation error class this project repeatedly punishes,
caught by Beta this time.

Team Alpha additions to the corrected brief, all source-verified: the
**`eligible_provider is not None`** requirement (:1897-1905 — a bare
`finalize_stripe` evaluates the predicate but parks the stripe in `staging`
forever with no matrix routing, so D2-A must drive the full serve path with a
misbehaving worker); `survivor_sum_match` added to the isolation assertions;
D2-B determinism pin and single-capture observer contract.

## Delivered

`tests/test_s172_phase5_d2_directional_uniqueness.py` (912 lines):

- **D2-A** — real serve path via `run_trial_miner` + `_OverlapWorker` speaking
  the real `MinerFramedSocket` wire, declaring `[0,10) [9,19) [20,30)`
  (overlap @9, compensating gap @19, seed-sum preserved at 30 so coverage is
  the sole red predicate). Asserts: 3 shards staged+verified; publish-count 0;
  no manifests/run state; matrix once (`retryable=True`, `fail_trial`,
  `constant_phase`); trial `aborted`; abort discharged once; no assembly.
- **D2-B** — labeled `DIRECT SINK INVARIANT-BREAK PROBE`: valid `{1,2}` set via
  public `publish_shard`, seed 5 duplicated in phase-1 forward/constant at
  0.90 vs 0.11, distinct event_ids AND slots; observing sink captures the first
  `DirectionalDuplicateError` under lock and re-raises unchanged; real
  `coordinator.commit_trial` returns `delivery=="failed"` / `d2b:commit`; all
  13 structured attributes; fail-closed state (no assembly, no consumed
  marker, 4 manifests retained, staged files intact, no `.npz`).
- **NC1-4** — scoping controls: cross-direction seed is a legitimate
  bidirectional intersection (asserted in both maps AND
  `bidirectional_constant`); disjoint shards commit; constant+variable same
  seed commits; same-slot/different-event is `ManifestReplayConflict`
  (intentional D1.1-G7 overlap, retained per Beta).
- Gate-22 whitelist: one new test path, registration-only.

## Judgment call — `all_verified`, verified and endorsed

The brief's §3 list named `all_verified == True` on the live finalize check.
Claude Code declined to assert it directly as scheduler-sensitive and split the
proof. Team Alpha verified the timing claim against `finalize_stripe`
(:1869-1905, lifecycle-locked staging/dispatch comment :1885): the dispatch
thread's finalize can legitimately observe `all_verified=False` before the last
staging job lands. The split — (a) deterministic direct
`evaluate_stripe_completion` reconstruction proving `coverage_ok` is the sole
red predicate with all shards verified, (b) live run separately asserting the
definitive routing fired on `substripes_match ∧ ¬reconciled ∧ ¬coverage_ok` —
is stronger, deterministic, no sleeps, nothing dropped. Endorsed by Alpha and
Beta. The over-specification was Alpha's; correcting it in review beat
discovering it as intermittent failures later.

## Mutation evidence — complete resolution-behavior kill set

| Mutant | D2-A | D2-B | By |
|---|---|---|---|
| `pop_map[seed] = max(...)` | GREEN | RED | Claude Code |
| silent keep-first | GREEN | RED | Claude Code |
| silent keep-last | GREEN | RED (+ D1.1 G14 red) | Team Alpha (independent) |

Identical red signature every time —
`{'delivery': 'done'}` (commit incorrectly succeeds). D2-A green throughout
proves the upstream barrier is independent of the writer; the keep-last
mutant additionally reddening D1.1's G14 proves the raise is load-bearing
across both harnesses. Writer restored byte-identical after each.

## Verification record

D2 **7/7** · D1.1 18/18 · D1.0 8/8 · D0 12/12 · Phase 4 63/63 · Phase 3 17/17.
NR baseline captured green at `b9c6120` BEFORE any edit; independently
reproduced in the Team Alpha sandbox on a pristine clone.

## Committed in this change

`tests/test_s172_phase5_d2_directional_uniqueness.py`,
`tests/test_s172_phase4_coordinator.py` (whitelist only),
`docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D2.md`,
`docs/TEAM_ALPHA_REVIEW_S172_PHASE5_D2.md`, this changelog.

## Deferred

`docs/PHASE6_PREREQS.md` — a running infrastructure checklist for the first
real trials (second 3080Ti passthrough into VM101, michael→CT100 SSH keys,
rrig6600 migration, VM101 static IP). Present on VM101 but **excluded from
this commit** per Team Beta: it was not in the D2 artifact inventory and D2
approval cannot implicitly authorize an unreviewed document. To be submitted
for separate content/scope review.

## Next

**D3** — the NPZ writer: `MinerTrialAssembly` → the frozen 22-array artifact
(contract wall on the FINAL artifact only; temp shards uncompressed per S159B,
canonical finals `savez_compressed`). This is the deliverable that makes the
miner's output consumable by Step 2/3, and the last one before D6 wiring makes
a **Zeus-only single-GPU smoke trial** possible — the earliest real-silicon
checkpoint, which needs none of the deferred infrastructure items.
