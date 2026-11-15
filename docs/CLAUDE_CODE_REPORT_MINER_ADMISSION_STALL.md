# CLAUDE CODE REPORT — THE STAGING-ADMISSION STALL (L-4) REPAIRED; RUN 4 ROUTING CONTROL; L-5 AND THE EVIDENCE READER

## Prior record

**Searches run** (worktree `/home/michael/tfm_worktrees/miner-stall`, detached at `7b9184a`,
2026-11-14):
- The brief's own Prior record (`docs/CLAUDE_CODE_INSTRUCTIONS_MINER_ADMISSION_STALL_20261114.md`),
  re-read; its anchors re-read live (`miner/range_miner_coordinator.py:4783-4856`, `:7524-8128`,
  `:8130-8245`, `:11850-11898`).
- **Read in full, as the brief required:** `CLAUDE_CODE_INSTRUCTIONS_ADMISSION_LIVENESS_REPAIR.md`,
  `…_ADMISSION_BINDING_REPAIR.md`, `CLAUDE_CODE_CORRECTION6_S172_PHASE4_ADMISSION.md`,
  `CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md`,
  `CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md`,
  `CLAUDE_CODE_INSTRUCTIONS_S172_BP_AMENDMENT.md`, `CLAUDE_CODE_REPORT_S172_GATE12_DISPATCH_STALL.md`,
  `GATE12_MP1_RUN_FORENSIC.md`. Read in part: `CLAUDE_CODE_REPORT_S172_STAGING_BACKPRESSURE.md`
  (resume trigger), `SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md` (R-1…R-4), the BP R2/R3
  briefs and reports (grep `pump|admitted|prune`).
- `docs/PROJECT_FILE_CATALOG.md` §1.0 SR-1; `docs/TB_RULING_FIELD6_IMPLEMENTATION.md` §2.
- `docs/exchange/from_beta/X-MINER-START_R1_BETA_REVIEW.md` Q6, Q7;
  `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` Q4;
  `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §A.
- `grep -rn` over `tests/*.py` for every scope proof and pin touching the candidate surfaces:
  `DECLARED_CHANGED`, `NO_TOUCH_DEFS`, `ADMISSION_NO_TOUCH`, `getsource(…_serve_dispatch)`,
  `_pump_deferred` call-site counts.
- Attempt 9's run log, read-only: `logs/gate12_20260817_181819.log` (main tree).

**Documents that already decide or record this subject:**

| Document | What it settles | Label |
|---|---|---|
| The brief | Parts 1–3, R1–R5, the questions | ALREADY RECORDED |
| `docs/LEADS.md` L-4, L-5 | The stall and the logging defect | ALREADY RECORDED; status updated here (§9) |
| X-MINER-START R1 Q7 | OBJECTION · LIVE on launch until L-4 is repaired and tested without the nudger | ALREADY RECORDED; this work answers it |
| X-MINER-START R1 Q6, X-RUN4-RANGE R1 Q4 | Keep the frozen oracle; successor check with an independent control | ALREADY RECORDED; Part 2 carries it out |
| Correction 6 (approach A) | `_admitted` is a pure serialization gate; one attempt stages at a time | ALREADY RECORDED; unchanged |
| Correction 4 §1a | The dispatch thread never blocks; `_pump_deferred` runs off it | ALREADY RECORDED; honoured (R2) |
| BP remediation §1.2; BP report | `_pump_deferred` is the capacity-release point; pump, then resume | ALREADY RECORDED; unchanged (R3) |
| R-1 changelog (R-1c) | "a dropped pump is a dropped wake"; R-1 G1a pins `enqueue_staging`, `_prune_admitted_locked`, `_try_admit_locked`, `_release_admission`, `_submit_with_slot` byte-identical | ALREADY RECORDED; binds the repair choice (§3.1) |
| MP-1 forensic | Pump cost under `_admission_lock` starved the serve loop | ALREADY RECORDED; L-4 is not it (§2.2) |
| MP-1 `gate_e2_ast_scope_proof` | Module's added-definition set is exact: no new `def` | ALREADY RECORDED; honoured (0 added) |
| SR-1 | Keep `DECLARED_CHANGED` true, with provenance | ALREADY RECORDED; done (§3.5) |
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:79-109`, `:145-153` *(added at Beta R1 Q9)* | The final routing review: the 69ca910 oracle closed unpinned behaviour at 47 tokens; the six-mutant roster; the after-commit steps (clean tree, Gate-22 rerun without widening, infrastructure state, Route-A provenance check, then Michael's separate authorization); Run 4's claim boundary (routing and containment only) | ALREADY RECORDED; carried in "Before Run 4" below |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:18-23`, `:73-94` *(added at Beta R1 Q9)* | Owner rulings: Beta reviews, does not approve; power caps / telemetry **dropped**, netconsole **kept**; agents may launch under the delegation rules; items A5 (power-on, GPU gate 8/8 ×3, parity 30/30, clean tree), A6 (netconsole proof line per host), A7 (Run 4 with the committed Route-A command) | ALREADY RECORDED; carried below |
| `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:50-63` *(added at Beta R1 Q9)* | What a delegated agent may do, incl. A6: redeploy the ten governed files from committed HEAD, verified 30/30 by the parity gate | ALREADY RECORDED; governs the coordinator redeploy |

**Items this report raises:** the repair, the tests and the Part 2/3 work are ALREADY RECORDED
(the brief). Two observations are NEW: one worker can also stall at base (L-4's "1 worker:
nothing can be deferred" was false; §2.1); and attempt 9's per-stripe deferral record shows
why it did not stall (§2.3). Neither CHANGES A RULING.

**Not searched:** the private repository, ser8, project files outside the repository, the rigs.

---

## For the owner, in plain words

- **Is the stall closed?** On CPU, yes. A stripe that finishes on the StripeComplete path now
  wakes the deferred work, the same way the staging path always did. With the old code I
  reproduced the stall in 10 of 24 runs (6/10 two-worker, 4/4 four-worker, 0/10 one-worker).
  With the repair, 120 of 120 forced runs commit, at 2, 4 and 25 workers, in both orderings.
  Run 4's exact shape (25 workers, 4 phases × 32 stripes over [2³¹, 2³²)) also commits in all
  three orderings. No pump timer exists anywhere in the tests any more.
- **What Beta says about launching Run 4 now:** Beta **withdrew** its X-MINER-START R1 Q7
  objection for these exact bytes (X-MINER-STALL R1 Q8, NO OBJECTION on Q1–Q8). Beta requests
  no further L-4 repair and no pre-launch fleet experiment. It does not authorize a launch;
  that is yours. Its one SUGGESTION (Q9: cite the final routing review and your updated launch
  plans) is adopted in this report.
- **What the rigs need at power-on:** the new `miner/range_miner_coordinator.py`, sha256
  **`b8e259b3e12b95297fa38018a41215d5c52f4d9b3f4095e384829a3ec1f1517d`**. It is in the rig parity set
  (`scripts/gate12_parity_gate.py:180`), so the parity gate refuses until all three rigs carry
  exactly these bytes. It replaces `6f2d14f3…` from the seed-start change.
- **What only a real fleet run can show:** that real workers never hit the old ordering and
  the new hand-off behaves the same with real staging (fsync, real payload sizes, remote
  spools). Attempt 9 did not stall, but that is because its staging always lagged (§2.3), not
  proof of absence. Run 4 committing is the follow-up that settles L-4 for the fleet. Nothing
  here measures pool hit rate against k/1000.
- **Two smaller repairs, also not committed:** WATCHER's log keeps `EXEC CMD` and the
  `[S172-PIN]` record again (L-5). The Step-1 evidence reader opens the live miner ledger in
  a way that cannot create `-wal`/`-shm` files. If the ledger's `-wal` is non-empty, the reader
  now gives no credit rather than read stale data (§6).
- **The Run 4 routing control** is put back as Beta asked (§5). The frozen 69ca910 module is
  byte-identical again. A separate successor check compares today's command with one captured
  from `b11b6fd`. All five previously uncredited mutants are now credited through it. The frozen
  gate itself cannot pass on today's tree, by design, and says so.

---

## 1. First step

| Item | Result |
|---|---|
| `pgrep -af "[b]eta_relay.py"` | nothing (rc 1) |
| `git rev-parse HEAD` (main) | `7b9184a95cd125c14685008cd306fe0fd9ff3040` |
| `git status --porcelain` (main) | clean, before and after |
| Worktree | `git worktree add --detach /home/michael/tfm_worktrees/miner-stall HEAD` |
| Base-only checkouts (pristine, for controls) | `/home/michael/tfm_worktrees/miner-stall-base` at `7b9184a`; `/home/michael/tfm_worktrees/miner-stall-b11b6fd` at `b11b6fd`; tracked files never edited in either |
| venv | `~/venvs/torch`, Python 3.10.12 |
| Datasets, copied mode 0444, sha256 equal to main | `daily3.json` `513648160d35…68f6`; `daily3-20260801T145551443433Z-513648160d35.json` `513648160d35…68f6`; `daily3_current.json` `ef3275264a90…250a` |
| Live-state snapshot | `data/miner_stall/live_snapshot_before.log` (worktree) |
| Baseline | 37 suites, sequential, CPU only, before any edit (§7). Part B ran through a wrapper that **does not run** `gate_manifest_staging_dir_is_usable_here`; it reports it as FAIL `SKIPPED-BY-BRIEF` |
| Tool scripts | outside every git tree, in `/home/michael/tfm_worktrees/miner-stall-tools/`, so that Gate 22 does not see them |

## 2. M — L-4 reproduced at base, without the nudger, before any edit

### 2.1 Counts

Script `/home/michael/tfm_worktrees/miner-stall-tools/m_repro.py`: the real driver path with
`pump_nudger` replaced in-process by a no-op. A read-only observer reads `_admitted`,
`_deferred` and the stripe rows once a second, and records the L-4 state when it persists
≥ 3 s. Serve timeout 40 s. Results: `data/miner_stall/M/base/{w2,w1,w4}.json`.

| Shape | Runs | Stalled | L-4 state observed in the stalled runs |
|---|---|---|---|
| 2 workers, 4 × 250-seed stripes, one mode | 10 | **6** | 6/6 |
| 1 worker, same | 10 | **0** | — |
| 4 workers, 2²⁶ stripes, total 2²⁸, both modes | 4 | **4** | 4/4 |

Snapshot examples: two workers, run 0: `_admitted` = {`st1_s3`} (`done`), `_deferred` = [`st1_s2`].
Four workers, run 0: `_admitted` = {`st3_s0`} (`done`), `_deferred` = [`st3_s1`, `st3_s2`, `st3_s3`].
M ran while the baseline suites loaded the CPU, so the rates depend on timing. The earlier
1/10 two-worker rate (seed-start report) was measured on a quieter box.

**Shown, not in M's table:** one worker can also stall at base. Capturing the one-worker
control (§4), 1 of 3 one-worker, both-modes runs on `7b9184a` stalled (`done` 8, `cancelled`
4). The mechanism needs no second worker: the worker's next stripe defers behind its own
previous `done` stripe. L-4's "1 worker (nothing can be deferred)" was wrong, and is corrected
in `docs/LEADS.md`.

**Deterministic reproduction** (G-UNIT run against base by mutant MU1, §3.4), on the real
`_serve_dispatch`: s0's only sub-stripe stages completely; s1's frame arrives and is deferred
behind s0; then s0's StripeComplete is processed. Result: s0 `done`, `_admitted` = {s0},
`_deferred` = [s1], s1's shard `pending`. Prototype run on the base tree printed exactly this.

### 2.2 MP-1, a relative, or new?

**New, and in the same family as MP-1, but not MP-1.** Shown from the record:
- MP-1 (`docs/GATE12_MP1_RUN_FORENSIC.md` §3, §6) is **too much pumping**. `_pump_deferred` held
  `_admission_lock` across per-entry ledger reads, for about 3,640 thread-seconds. The serve
  loop starved and a lease expired. R-1…R-4 cut the pump's cost. They deliberately kept its
  call rate, because "a dropped pump is a dropped wake" (R-1c, R-1 changelog §1).
- L-4 is **a missing pump**. One of the two paths to `done` (the StripeComplete branch,
  `:11897` at base) never wakes the deferred queue. It is a liveness defect, not contention.
  Nothing waits on a lock. Everything waits on an event that never comes.
- Both arise from Correction 6's design: one attempt stages at a time, and `_pump_deferred`
  is the only place an admission is pruned and the deferred queue resumed (BP §1.2). They are
  relatives in that sense: the same serialization point, failing in opposite directions.
- The 2026-08-07 gate-12 stall (`CLAUDE_CODE_REPORT_S172_GATE12_DISPATCH_STALL.md`) was a third
  member: file reservations never released. That was a different counter (`reservations`, not
  `_admitted`) and is repaired. L-4 does not reopen it, and R-1 is not reopened.

### 2.3 Why attempt 9 (25 real workers, 128/128) did not stall

**Shown** (read-only, `logs/gate12_20260817_181819.log`, run `distributed_config_t1_554463d3`):
- `[S172-BP] summary`: `deferred_high_water=1679`, `staging_jobs_completed=5693`,
  `staging_jobs_per_sec=7.712`, `pause_events=0`.
- 128 `STRIPE_RX_SUMMARY` records, all emitted at StripeComplete processing. 5,364 of 5,896
  frames were deferred.
- `deferred_seconds_total` is charged only when a deferred frame is released
  (`note_stripe_frame_released`, `miner/range_miner_coordinator.py:6408`). For **107 of 128**
  stripes the record shows frames deferred and `deferred_seconds_total = 0.0` at the moment
  its StripeComplete was processed. None of that stripe's deferred frames had been staged
  yet, so its last staging job could not have finished first. That is the safe order.
- 13 stripes had some frames released by then, and 8 were never deferred. For these 21 the log
  does not show which order occurred.

**Inference:** on the fleet, staging lagged far behind ingest (7.7 jobs/s against a 1,679-frame
backlog). A StripeComplete was almost always processed before the stripe's last staging job.
Fake workers answer in microseconds and stage tiny payloads, which reverses that. So attempt 9
is evidence that the fleet's timing makes L-4 rare. It is not evidence that L-4 cannot happen.

## 3. Part 1 — the repair

### 3.1 The mechanism, and why this one

The record allows one of L-4's three candidates:
- **Prune in `enqueue_staging`: ruled out.** R-1's certified G1a pins `enqueue_staging`,
  `_prune_admitted_locked`, `_try_admit_locked`, `_release_admission` and `_submit_with_slot`
  byte-identical (`tests/test_s172_r1_drain_remedy.py:493-522`). It would also not fire in the
  observed stall, where every remaining frame is already deferred and none arrives.
- **Periodic pump: not added.** It is recovery, not prevention (R5). The direct repair makes the
  state unreachable (§3.3), so a timer would add pump traffic on the MP-1 surface for nothing.
- **Pump after the StripeComplete finalize: chosen.** The pump is handed to the staging
  executor, the thread pool that already runs it after every staging job.

### 3.2 The change

`miner/range_miner_coordinator.py:11897-11927` (in `_serve_dispatch`, StripeComplete branch):

```python
_sc_check = self.finalize_stripe(run_id, msg.stripe_id,
                                 eligible_provider=eligible_provider)
# [L-4 STAGING-ADMISSION STALL REPAIR] … (comment, :11899-11917)
if _sc_check.is_complete:
    try:
        self._staging_exec().submit(self._pump_deferred)
    except RuntimeError as _sc_e:
        logger.warning("[L-4] could not hand the staging pump to the executor …")
```

One definition changes (`_serve_dispatch`), with 0 added and 0 removed, so MP-1's exact
added-set proof holds. `finalize_stripe`, `_pump_deferred` and every admission surface are
untouched.

### 3.3 Requirements

| # | How it holds | Evidence |
|---|---|---|
| R1 | `ST_DONE` is written in one place (`finalize_stripe`, `:4280`), which has exactly two callers (`:8230`, `:11897`). The staging-job caller pumps after it (`:8234`). The StripeComplete caller now hands a pump off after it whenever the finalize leaves the stripe complete. If the two paths race, `_lifecycle_lock` serializes them. Whichever writes `done` is followed by a pump. The other sees a non-staging stripe, gets `is_complete`, and issues a harmless extra pump. The hand-off happens after `finalize_stripe` returns, so `done` is committed before the pump's prune can read it. *Precisely (Beta R1 Q1):* this guarantees a **following wake** under normal executor operation. It is not atomic removal at the `done` write: a `done` key holds admission briefly until the queued pump runs. A refused submit (executor already shut down at teardown) is logged. It is no guarantee against arbitrary executor or process failure | G-UNIT; G-SF/G-CF ×3 worker counts ×20; G-RUN4; MU1, MU3 red |
| R2 | The dispatch thread never runs the pump. `ThreadPoolExecutor.submit` enqueues and returns. Its lock is the executor's own and is never held by the pump. The pump runs on a `miner-staging_*` thread, which is where MP-1 already attributes it | G-OFFTHREAD (AST: no direct call; exactly one `self._staging_exec().submit(self._pump_deferred)`; after the finalize); G-UNIT and every ordered run record the pump's immediate caller (never `_serve_dispatch`); MU2 red |
| R3 | `_pump_deferred` is byte-identical, so pump-first-then-resume (`finally` → `_resume_paused_connections`) is unchanged | G-SCOPE: 17 surfaces byte-identical vs `7b9184a`, including `_pump_deferred`, `_resume_paused_connections`, `_grant_resume_credit`, `staging_can_accept` |
| R4 | The branch names no admission state and calls no admission surface. Release is `_prune_admitted_locked`'s `dict.pop`, which is idempotent. Admission is granted only by `_try_admit_locked` after a fresh `_attempt_live_locked` probe (both pinned), so a superseded or terminal attempt cannot be admitted by this change. With nothing deferred, the pump prunes and returns. One-worker ledger rows and assignments are byte-equal to base | G-OFFTHREAD (forbidden-name arm); G-SCOPE; **G-ONE**: 16 stripes, 16 assigns, 7 tables / 84 rows byte-equal to the `7b9184a` control |
| R5 | Prevention, not recovery: after the change no ordering leaves a `done` stripe without a following pump. No timer was added | §3.1 |

### 3.4 Tests — `tests/test_s172_admission_stall.py` (new)

CPU, real coordinator, fake workers over loopback, **no nudger**. The ordering driver
`tests/fixtures/admission_stall_driver.py` (new, self-contained) binds four observers
(`_submit_with_slot`, `_run_staging_job`, `_pump_deferred`, `finalize_stripe`). Each calls the
real method exactly once when the coordinator does. Order is forced only by waiting, and
every wait is bounded. A run whose order could not be forced is not credited.

| Gate | Result (log `data/miner_stall/admission_stall_full.log`) |
|---|---|
| G-UNIT | PASS: s0 done, admission released, s1 staged; 3 pumps after the SC, all on `miner-staging`; 0 called from `_serve_dispatch` |
| G-OFFTHREAD | PASS |
| G-SCOPE | PASS: vs `7b9184a` 1 changed (`_serve_dispatch`), 0 added, 0 removed; 17 surfaces byte-identical |
| G-ONE | PASS: byte-equal to the base control (`49ac9f233215`) |
| G-SF-2 / -4 / -25 (last staging job **and** its completion pump finish before the StripeComplete is processed) | PASS 20/20, 20/20, 20/20 (8 / 16 / 100 stripes each; forced on every stripe) |
| G-CF-2 / -4 / -25 (StripeComplete processed before the last staging job finishes) | PASS 20/20, 20/20, 20/20 |
| G-RUN4 (25 workers, `seed_start` 2³¹, total 2³¹, stripe 2²⁶, both modes, production caps) | PASS: free 271 s, staging_first 334 s, complete_first 274 s; each 4 × 32 contiguous stripes over [2³¹, 2³²), 128 `done`; forced 128/128 |

**Totals: 120/120 forced small runs committed, plus 3/3 at Run 4's shape. 11/11 gates.**

| Mutant (function-level, production globals) | Gate | Result |
|---|---|---|
| MU1 repair removed (base behaviour) | G-UNIT | **RED**: `STALL: done stripe s0 still holds admission`; deferred `[run_s1]` |
| MU1 repair removed | staging_first ×3, 2 workers | **RED**: 3/3 runs aborted (`done` 1, `cancelled` 3) |
| MU2 pump called on the dispatch thread | G-UNIT | **RED**: `_pump_deferred called directly from _serve_dispatch` |
| MU3 hand-off condition inverted | G-UNIT | **RED**: stall |

**4/4 detected. Sentinel PASS.** At base (`7b9184a`, smoke run of the same driver), free order
stalled, complete_first committed with 8/8 forced, and staging_first stalled deterministically.
That is the split L-4 predicts.

**The seed-start suite without the nudger.** The nudger is removed from
`tests/fixtures/miner_seed_start_driver.py` and from G-E2E in
`tests/test_s172_miner_seed_start.py`. Result: **10/10 gates, 11/11 mutants, PASS**
(`data/miner_stall/seed_start_no_nudger.log`). G-R7-ZERO checks the driver's sha against its
control, so the control was re-captured on clean `b11b6fd` with the nudger-free driver
(`tests/capture_miner_seed_start_zero_control.py`, new). `b11b6fd` still has L-4, so that
capture alone ran the old timer as a labelled liveness aid. It refuses unless every shape's
digest equals the original capture's, and both did: **A9 `972d23b4e5a3`, M `737e803754f8`,
equal**. Removing the nudger therefore changed nothing the control compares. The only
remaining `_pump_deferred` timer in `tests/` is in that base-only capture script.

### 3.5 SR-1

- `tests/test_s172_r1_drain_remedy.py` `DECLARED_CHANGED`: **+`RangeMinerCoordinator._serve_dispatch`**,
  with provenance naming this repair. Anchor unchanged; assertion still exact.
- `tests/test_s172_mp1_drain_attribution.py` `DECLARED_CHANGED`: already contains
  `_serve_dispatch` (MP-1's own change). The set is unchanged; a provenance comment is added.
- No other historical live-vs-anchor proof covers the module (grep §Prior record).
- Before/after: R-1 suite rc 0 → 0, MP-1 PASS → PASS (§7). AST diff vs `7b9184a`: exactly
  `_serve_dispatch` changed (G-SCOPE).

## 4. The one-worker control (R4)

Captured on `7b9184a` by `/home/michael/tfm_worktrees/miner-stall-tools/capture_one_worker_control.py`
into `tests/fixtures/admission_stall_one_worker_control_7b9184a.txt`. The capture refuses
unless HEAD = `7b9184a` with tracked files clean. Shape: 1 worker, 1,000 seeds, 250-seed
stripes, 125-seed caps, both modes, start 0. Every table, every row, with clocks, paths and
autoincrement ids dropped, and run id and work dir substituted. Each shard's sha256 is
**checked** against the payload the driver sent (it embeds the random run id), not dropped.
Two committed base runs compared equal. The first capture attempt showed the one-worker stall
(§2.1); attempts are recorded in the control file.

**Scope of the claim (Beta R1 Q4).** The equality is over the normalized outputs of
*successful* one-worker runs: assignments, stripes and ledger rows. It is not timing, not every
raw ledger byte, and not "one-worker behaviour unchanged". Base can stall with one worker, and
the repaired wake applies there too.

## 5. Part 2 — the Run 4 routing control

1. **Frozen module restored** byte-for-byte to its `3e2537b` (69ca910-captured) state:
   `tests/fixtures/run4_routing_clean_control.py` sha256 `8f61420c…`, equal to the capture's
   `fixture_module_sha256`.
   **What G-UNPINNED-IDENTICAL can and cannot show today.** Its sha check passes, so the oracle
   is intact and guarded: any edit reds it as "oracle invalid". It cannot PASS. Since stage 1,
   Step 1 needs an open search, the frozen boundary has none, and run_step fails closed before
   a command exists. It reports INCOMPLETE ("run_step returned before dispatch"). It no longer
   certifies today's unpinned argv.
2. **Successor check**, separate, beside it:
   - `tests/fixtures/run4_routing_search_control.py` (new): the frozen boundary plus the stage-1
     search collaborator. Its constants are imported from the frozen module. Its delta from the
     frozen `build_argv` is gated statement by statement (**G-SUCCESSOR-BOUNDARY**): dataset
     stub, stub DB, and the search's open and cleanup, nothing else.
   - **Independent control:** `tests/fixtures/run4_search_control_b11b6fd.txt`, captured by
     `tests/capture_run4_search_control.py` (new) on clean `b11b6fd`. That tree predates the
     miner seed-start change and this brief. WATCHER is byte-identical `b11b6fd..7b9184a`, and
     the capture records its committed digest. All three fixture digests are recorded and
     re-checked.
   - **G-UNPINNED-SUCCESSOR:** today's unpinned argv is list-equal to the `b11b6fd` control
     (49 tokens). Against the frozen 69ca910 argv the difference is exactly the two named
     stage-1 scheduling changes, `--seed-start 0 → 2147483648` (token 27, the search's first
     gap) and `--step1-search-id s1s-734ae48f…` appended. Every other token is equal and in
     order.
   - **Comparison boundary** (recorded in the suite at the gate): argv construction by
     `run_step(1)` on the unpinned path, with the frozen stubs plus the search. Not dispatch,
     not the child, not the fleet.
   - The pin harness and every other gate build through the successor builder. Only
     G-UNPINNED-IDENTICAL keeps the frozen one.
3. **Mutants:** each mutant's detecting gates are unchanged. Its must-stay-green unpinned
   oracle is now G-UNPINNED-SUCCESSOR. M2b's vacuous credit through the INCOMPLETE frozen
   oracle is replaced by a real detection through the successor oracle (it diverges at token 48
   and appends the warm-start pins).

| Suite | Before (base) | After |
|---|---|---|
| `tests/test_s172_run4_routing_patch.py` | 15/20 (G-UNPINNED-IDENTICAL INCOMPLETE; M1, M1b, M2a, M4 uncredited; M2b vacuous) | **21/22**: only G-UNPINNED-IDENTICAL, INCOMPLETE by design. **M1, M1b, M2a, M2b, M3, M4 all credited.** WATCHER bytes verified restored after M1b |
| `tests/run4_prelaunch_provenance_check.py` | 10/10 | 10/10 |

No assertion was weakened. The suite's whole-suite sentinel cannot print on today's tree
while the frozen gate is in its list. Retiring or re-scoping that gate is Beta's and the
owner's call; it is not done here.

## 6. Part 3 — two small repairs

- **L-5** (`agents/watcher_agent.py:776-787`). `step1_presented_configuration` saves the root
  logger's level, imports `window_optimizer`, and restores the level in a `finally`. This is
  the only import of `window_optimizer` in WATCHER (grep).
- **The evidence reader** (`window_optimizer_integration_final.py:1443-1456`): per addendum §A.
  If the ledger's `-wal` or `-journal` is non-empty, it prints the reason and returns None
  (UNAVAILABLE, no credit, fail-closed). Otherwise it opens
  `file:…?mode=ro&immutable=1`. **Behaviour note for Run 4:** if the live ledger's `-wal` is
  non-empty when the evidence is read, S7 now refuses credit rather than reading possibly
  stale rows. The live `-wal` is 0 bytes today (snapshot).

`tests/test_s172_small_repairs_part3.py` (new) with `tests/fixtures/l5_fresh_process_probe.py`:

| Gate | Result |
|---|---|
| G-L5 (fresh subprocess; `window_optimizer` not imported; real `run_step` on the pinned path, dispatch intercepted; search opened in a separate process) | PASS: imported inside `run_step`; `EXEC CMD` 1; `[S172-PIN]` 1; root 20 → 20 |
| G-EVID-OPEN (throwaway WAL ledger, no sidecars) | PASS: interval read; directory unchanged |
| G-EVID-WAL (writer holding an uncheckpointed write; immutable view shown stale) | PASS: None, UNAVAILABLE printed |
| ML5 base function | **RED**: `EXEC CMD` 0, `[S172-PIN]` 0, root 20 → 30 |
| ME1 base reader, clean ledger | **RED**: created `miner_ledger.db-shm` (32,768 B) and `-wal` (0 B) |
| ME2 base reader, non-empty `-wal` | **RED**: returned evidence for phase 2 |

**3/3 gates, 3/3 mutants, PASS.**

## 7. Suites before and after

CPU only, one at a time. Logs: `data/miner_stall/{baseline,after}/`. Comparator:
`/home/michael/tfm_worktrees/miner-stall-tools/compare_suites.py`.

| Suite | Baseline | After | Difference |
|---|---|---|---|
| r1_drain_remedy | rc 0 | rc 0 | — (SR-1 entry added) |
| mp1_drain_attribution | PASS | PASS | — |
| admission_liveness | PASS | PASS | — |
| staging_backpressure | PASS | PASS | — |
| staging_partb (probe case not run) | 23/24, G-ROUTE-2 `SKIPPED-BY-BRIEF` | same | — |
| phase4_coordinator | 63/63 | 62/63 | **Gate 22 only**: untracked/modified `tests/*.py` from this work (14 files, listed in its log). Transitional; clears at commit; allowlist not widened (FIELD6 ruling §4) |
| miner_seed_start | PASS (with nudger) | PASS **without** nudger | §3.4 |
| run4_routing_patch | 15/20 | 21/22 | §5 |
| run4_prelaunch_provenance_check | 10/10 | 10/10 | — |
| step1_search_record_stage1 | PASS | PASS | — |
| f1_f2_active_lease 16/16; f1_lease_origin; defect_a 29/29; window_anchor_brief_i_mutants; phase5_d3_25; p05 dataset authority; process_sharded_import_gate; h1h2; elapsed_roundtrip; d6_integration_repair; resolved_execution_set; phase1_scaffolding 6/6; seed_domain_cursor; gate_prod_shape_scoping; owner_land | pass | pass | — |
| admission_binding | FAIL | FAIL | same failing gates (only a timestamp differs) |
| attempt6_remediation; window_anchor_brief_i; phase5_d0; phase5_d4; phase5_d6_production_adapter; phase5_d6_threshold_path; threshold_propagation | FAIL | FAIL | same failing gate sets |
| phase5_d1_engine; d1_workflow; d2; d5 | FAIL | FAIL | Same failing NR gate. These run Phase 4 then Phase 3 as subprocesses and stop at the first red. Before, that was Phase 3's pre-existing `expected '17/17'`; now it is Phase 4's transitional Gate 22, so the later subprocess output is absent |
| **admission_stall (new)** | — | **11/11, 4/4** | §3.4 |
| **small_repairs_part3 (new)** | — | **3/3, 3/3** | §6 |

Reds that predate this work are not charged to it.

**Live state:** before and after snapshots (`data/miner_stall/live_snapshot_{before,after}.log`)
are identical. Halt file absent. `~/miner_staging` directory mtime unchanged (15:47:28).
Ledger and its `-wal` (0 B) and `-shm` byte-identical. Main tree's four `.db` files identical.
`optuna_studies` 104 entries. Main tree clean. The only line that differs is the mtime of
`/home/michael` itself, from creating `tfm_worktrees/miner-stall*`.

## 8. Beta's last position per question

Thread X-MINER-STALL, **one round**. Review: `docs/exchange/from_beta/X-MINER-STALL_R1_BETA_REVIEW.md`
(main tree).

| Q | Beta R1 | What was done |
|---|---|---|
| 1 ordering / done stripe holding admission | NO OBJECTION | Wording sharpened in §3.3 R1 (a following wake, not atomic removal) |
| 2 dispatch unblocked, back-pressure order | NO OBJECTION | — (Beta reran R-1 44/44, MP-1 38/38) |
| 3 double release / superseded or terminal | NO OBJECTION | — |
| 4 one-worker, SR-1 | NO OBJECTION | Scope of the equality claim stated in §4, per Beta |
| 5 forced orderings, no nudger | NO OBJECTION | — (Beta: socket runs UNAVAILABLE in its sandbox; counts attributed to Alpha) |
| 6 frozen oracle, successor, mutants | NO OBJECTION | — (Beta independently rebuilt the `b11b6fd` command: 49 tokens equal; ran M1b in memory) |
| 7 Part 3 | NO OBJECTION | — (Beta reran Part 3: 3/3, 3/3) |
| 8 X-MINER-START Q7 launch objection | **NO OBJECTION: objection withdrawn for the reviewed bytes** | — |
| 9 missed governing record | SUGGESTION | **Adopted** (in scope, report only): three records added to Prior record and "Before Run 4" below |

No severity is contested. No second round was needed: there is no objection and no request
for evidence.

### Before Run 4 — what remains (from the record; Beta R1 Q8/Q9)

1. Owner lands and commits (Group A by script, Group B by hand), dual-push.
2. Clean tracked tree; rerun Gate 22 and other commit-sensitive checks without widening the
   allowlist (routing R2 review `:145-153`).
3. Power-on (completion plan A5): `boot-proxmox`, GPU gate 8/8 ×3, **redeploy the coordinator**
   (sha256 `b8e259b3…`) from committed HEAD, parity gate 30/30 MATCH, clean tree (delegation
   plan A6).
4. Netconsole re-armed, one proof line per host (A6). Power caps and telemetry are dropped by
   owner ruling 3.
5. Route-A pre-launch provenance check on the committed tree.
6. Michael authorizes Run 4 (or a delegated agent under its authorization rules).
7. Only the real run can show actual staging and transport volume, GPU execution, publication
   and cleanup, and accepted evidence; it is also L-4's fleet follow-up.

## 9. Leads

`docs/LEADS.md` updated in the worktree. **L-4: OPEN — repaired on CPU, not committed; fleet
incidence open.** Follow-ups run are recorded, including the one-worker correction and the
attempt-9 analysis. The open follow-up is the first fleet run with the new coordinator.
**L-5: OPEN — repaired on CPU, not committed; real-log confirmation open.**

## 10. Review by Beta

One round. Relay verdict **REVIEW RECEIVED** (exit 0, 421.9 s, no boundary violation). Request
`docs/exchange/to_beta/X-MINER-STALL_R1_ALPHA_REVIEW_REQUEST_admission_stall_repair.md`; review
`docs/exchange/from_beta/X-MINER-STALL_R1_BETA_REVIEW.md`; the relay's row in
`docs/exchange/LEDGER.md` (all main tree). Relay stdout `data/miner_stall/relay_r1.out`
(worktree). Beta's reviewed production sha256 values match Group B and the integration file
below.

## 11. Landing list

**Group A — files the landing script takes**

| File | sha256 |
|---|---|
| `window_optimizer_integration_final.py` | `44aebe34ae2a334e9b4db3e19886a9a10e2ff63ac2f9c96e81e83f5a32aeb9d6` |
| `tests/test_s172_admission_stall.py` (new) | `f652b0acdaea14f889e71e5a7467abaecf93e2eb9dbe011c3ffe20f2aa721527` |
| `tests/fixtures/admission_stall_driver.py` (new) | `7af49508f796f300065fcdb098ac5154128f65408bbf0043217905d01f0ab51a` |
| `tests/fixtures/admission_stall_one_worker_control_7b9184a.txt` (new) | `59655719a329b36cb8a2a3fdffb0fd02a38e614cef8d88d1c8aed45281c2a881` |
| `tests/test_s172_small_repairs_part3.py` (new) | `1f68cc44fc7f4435189e656e59770d76a571167598f5b342a15174fc4efa706a` |
| `tests/fixtures/l5_fresh_process_probe.py` (new) | `26ac8f957815770d49bec5a5ae7dc8dcb4aafff023770a1af8d0fe64de1b9a1b` |
| `tests/fixtures/run4_routing_clean_control.py` (restored to `3e2537b`) | `8f61420c8b36096a0537f03346aeacfdff48912a86b175af67416481d6fb91ce` |
| `tests/fixtures/run4_routing_search_control.py` (new) | `a5f22120f5b07a3a76eb18b65a789807d742b74c993fbfe588a130bd354effe7` |
| `tests/fixtures/run4_search_control_b11b6fd.txt` (new) | `f61046c051cc77b8f41915fce5f6640aa4cd999c45394c1cc2a73e033eaf7f16` |
| `tests/capture_run4_search_control.py` (new) | `38c75f65b60200870c25d7c22e3e878d8cb3d5f413c75b2bc35de564b7ac92e8` |
| `tests/fixtures/run4_pin_harness.py` | `458b9663f1e37a8007279d2b57018eb493367929a1c8d83126a0211e5c333166` |
| `tests/test_s172_run4_routing_patch.py` | `ba904993643545a7ee2eeca70606a40457a9f1ca56bcf78f470a7704af6f29fd` |
| `tests/fixtures/miner_seed_start_driver.py` | `506225f49158bd736c98ba3942bee6be1a2b70c2bca2fa949c5caf43e0935dd8` |
| `tests/fixtures/miner_seed_start_zero_control_b11b6fd.txt` (re-captured) | `31ead5e3b1db6aca6660033b2f76b3e3b272ed8b1cdddb0b20918a17c3465d27` |
| `tests/capture_miner_seed_start_zero_control.py` (new) | `1f73a21204d8aecf6f7cf57e5b3f81594ac24db8d6f2f56552c423c1d457e8b3` |
| `tests/test_s172_miner_seed_start.py` | `ba5c6d4138385da63e0238b510d69299ed180916c6bc75134e1b441f1d743a44` |
| `tests/test_s172_r1_drain_remedy.py` | `76745e76132db8c1d692a36996130c590df8ee1d5f13142f40096484473fa0a9` |
| `tests/test_s172_mp1_drain_attribution.py` | `9321d80b4af9b24aa27a043a6edd1af3222bd5af4998fb4363b5d4e7bbe65fbf` |
| `docs/LEADS.md` | in the session summary |
| `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md` (this file) | in the session summary |

**Group B — hand-landed** (`scripts/owner_land.py:86-87`)

| File | sha256 |
|---|---|
| `miner/range_miner_coordinator.py` (**rig parity set**) | `b8e259b3e12b95297fa38018a41215d5c52f4d9b3f4095e384829a3ec1f1517d` |
| `agents/watcher_agent.py` | `f5a8d1c4d56eba1656331a87e7518f380f44f4c9e1d0f8d61737bfb3e6a0de51` |

Not edited: every other file under `miner/`, `prng_registry.py`, `sieve_gpu_worker.py`, the
kernels, `utils/`, `persistent_worker_coordinator.py`, `persistent/`, `gate12_launch.sh`, the
22-array contract, `/home/michael/tfm_worktrees/pwc-step1`. Nothing committed or pushed.
