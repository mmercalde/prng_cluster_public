# SESSION_CHANGELOG_20260723_PHASE5_D1.md

**Session scope:** S172 Phase 5 — D1 instruction document (REV1→REV5) and
D1.0 implementation, review, and approval.
**Base at session start:** HEAD `7f2a010` (D0 committed `4c697a8`).

## Outcome

**D1.0 APPROVED FOR COMMIT by Team Beta** (implementation + gates + scope
ruling). D1.1 begins only after this commit is dual-pushed and reviewed as
the new base.

## Document trail — `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md`

Five revisions to authorization. Each rejection found a real defect:

- **REV1 → REV2 (Beta pre-write ruling):** 7 amendments (11-field
  cross-manifest consistency incl. provenance; exact replay/slot-conflict
  rules; atomic commit-install wording; frozen 24-field record;
  phase-identity matrix; spool identity beyond digest; structured
  `DirectionalDuplicateError`) + 2 decisions (Optional NPZ paths; abort
  tombstone in D1).
- **REV2 → REV3 (Blocker 1+2):** **Phase-4 producer defect #1** —
  `workflow_stages_for(base, False)` returned forward-constant ALONE (no P2,
  no constant bidirectional possible). D1.0 became a staged pre-deliverable.
  Also: false restart/republication claim corrected; sink RLock + deep-copy
  ownership; G4/G5/G8 corrected to real coordinator semantics; container
  validation.
- **REV3 → REV4:** W5-A/B proved not to discriminate (they pause after the
  durable transition); W5-R stale-read gate specified; G13 dual-copy
  provenance; G16 final-state always tombstoned.
- **REV4 → REV5 (critical):** **Phase-4 producer defect #2 + a defect in the
  prescribed fix itself** — the REV2-prescribed `_lifecycle_lock` abort shape
  deadlocks `handle_stripe_failure` → `fail_trial` →
  `submit_abort(...).result()` (lock held on the calling thread; RLock not
  transferable). Replaced with **CAS-result disambiguation + terminal-state
  re-read**; W5-R revised to CAS semantics; W6 no-deadlock gate added.
- **REV5 APPROVED** with two folded implementation notes: W6 isolated
  timeout-terminable subprocess; lock prohibition specific to the coordinator
  `_lifecycle_lock` (ledger `_write_lock` legitimate).

## D1.0 implementation (Claude Code, VM 101)

Two narrow coordinator corrections, exactly per REV5:
1. `workflow_stages_for`: constant always bidirectional (`[(base,1),
   (base_reverse,2)]` for `test_both_modes=False`); hybrid pair gated.
2. `abort_trial`: CAS + re-read terminal decision; `False` from
   `mark_trial_aborted` disambiguated (committed → refuse; aborted →
   idempotent discharge retry; other → fail closed). No lock acquired.

Harness `tests/test_s172_phase5_d1_workflow.py` (832 lines): W1-W6 over the
real serve/publish/framed-socket path; W5-R thread-specific first-call
`get_trial` interception (events, zero sleeps); W6 in isolated child
processes covering both failure-matrix branches.

## Verification (Team Alpha, independent sandbox)

- Patched pristine `7f2a010`: **8/8** D1.0 gates green, incl. W4
  non-regression (Phase 4 63/63, Phase 3 17/17, D0 12/12).
- **Pre-fix discrimination:** W1/W2/W3/W5-R FAIL with the exact defect
  signatures (W5-R: sink abort discharged against a committed trial);
  W5-A/B/W6 pass by design (W6 guards the rejected REV4 lock design).
- Verified `MinerLedger.create_trial` has no internal `get_trial`, so the
  W5-R interception lands on the true pre-read.

## Rulings & standing rules established

- **Whitelist standing rule (Beta):** registering a new deliverable's exact
  harness path in an established coexistence whitelist is pre-authorized when
  the edit only registers that harness, gate semantics are unchanged, local
  format is followed, and the change is reported in review. Any other edit to
  an approved harness stays out of scope.
- REV2's binding lock prescription was itself defective — recorded as
  process precedent: **prescribed fixes are claims to verify against source,
  same as any other claim** (Beta concurred).

## Committed in this change

Implementation: `miner/range_miner_coordinator.py`,
`tests/test_s172_phase4_coordinator.py` (gate-22 whitelist entry only),
`tests/test_s172_phase5_d1_workflow.py`.
Governance: `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md` (REV5),
`docs/TEAM_ALPHA_REVIEW_S172_PHASE5_D1_0.md`, this changelog.
Excluded: pre-existing untracked briefs (S176-178, WATCHER_KPI) and `tmp/`.

## Next

D1.1 — `miner/range_miner_npz_writer.py` (assembly engine +
`AssemblingPhase5Sink`) + `tests/test_s172_phase5_d1_engine.py`, per REV5
§4-§9, against the post-D1.0 HEAD. Separately gated; STOP for Alpha + Beta
review before D2. Flagged for D6: apply the same seam-level adversarial
treatment to `serve_trial`'s return path before trusting the integration
wiring (two consecutive deliverables found Phase-4 seam defects only when
Phase 5 consumed them).
