# Team Alpha Review Record — S172 Phase 4 (RANGE-MINER coordinator) — REV 3
**Reviewer:** Team Alpha (lead dev)
**Date:** 2026-07-19
**Verdict:** PASS — ready for Team Beta binding re-review.
**Supersedes:** rev-2 (serve-path correction). Rev-3 covers Correction 2 — the six
release-blocking defects Beta found in the serve path, ledger, and production wiring.
**Method:** file-vs-source, ADVERSARIAL. For each defect I traced Beta's specific
attack against the delivered source (the two-attempt collision, the duplicate result,
the cross-socket spoof, the committed-then-aborted race, the two-trial PK collision) —
not the happy path.

---

## Accountability note

Beta's six-defect rejection was correct. Three of those defects (#1, #2, #3) were in
functions Alpha rev-1/rev-2 read line-by-line and passed. The rev-1/2 reviews traced
the INTENDED path and pattern-matched "looks right" instead of constructing the
adversarial case — the exact failure mode this project warns against. Beta caught them
with dynamic probes. Rev-3 corrects Alpha's method: every defect below was re-verified
by tracing the attack, and every fix is checked against the gate that now encodes
Beta's probe.

---

## Scope reviewed (Correction-2 delta)

| File | Change |
|---|---|
| `miner/range_miner_coordinator.py` | Six-defect fixes: private staging paths, dup-result guard, dispatch identity binding, bounded staging executor, terminal-state exclusivity, unique run_id. |
| `tests/test_s172_phase4_coordinator.py` | Gates 38–47 added (one per defect / sub-defect); harness now 48 `_check` (36 brief + gate 37 + 38–47 + subprocess non-regression). |
| `window_optimizer_integration_final.py` | Defect 6: real `_use_miner` production wiring, gated behind `use_range_miner`. |

Harness result: **all gates green, exit 0.** Phase-3 still 17/17. No existing gate
weakened or deleted.

---

## Per-defect adversarial verification

**Defect 1 — stale attempt deletes the current attempt's file (Beta-reproduced).**
`_staged_path` now embeds the immutable task key `(run_id, stripe_id, attempt,
sub_index, staging_generation)` in the filename
(`{run}__{stripe}_a{attempt}_s{sub}_g{gen}_{sha16}.json`); the temp path derives from
it (`{staged}.tmp.{pid}`). Trace of Beta's attack: attempt-0's stale callback enters
`_finalize_stage`, renames onto `task.staged_path` — but that is attempt-0's `_a0_`
file, NOT attempt-1's `_a1_` file; the fence then detects stale and `_fail_and_release`
deletes `task.staged_path` = attempt-0's own file. Attempt-1's file is never touched.
The rename-before-fence ordering (the original bug) is now SAFE precisely because the
path is attempt/generation-private. **Gate 38** asserts `attempt1_file_after_stale_
finish is True` (Beta's exact probe) plus distinct private paths and "stale removed only
its own."

**Defect 2 — duplicate result creates duplicate reservation (Beta-reproduced).**
`_serve_dispatch` now captures `record_substripe_result`'s return and `if not inserted:
return` BEFORE any `enqueue_staging`. `reservations.event_id` gained a UNIQUE constraint
(defense in depth); `reserve()` handles the violation without crashing. Trace: same
`(attempt, sub_index)` delivered twice → second insert returns False → dropped → no
second stage → one reservation. **Gate 39** asserts `reserved_files()==1` (Beta's
`held_reservations_after_duplicate==1`) and exactly one shard row.

**Defect 3 — connection-bound identity bypassed at dispatch.** `_serve_dispatch` now
takes `bound_worker_id`, and the caller passes `worker_by_sock.get(s)` — the identity
of the RECEIVING socket, which the sender cannot forge. The first check
(`if msg_worker_id != bound_worker_id: return`) fires before resolving the connection or
touching the ledger; the connection is resolved from `bound_worker_id`, never
`msg.worker_id`. Trace: worker A's socket sends `worker_id=B` → bound id is A → mismatch
→ dropped. **Gate 40** sends the spoof on A's real framed socket and asserts no ledger
mutation, no reservation.

**Defect 4 — synchronous staging blocked the dispatcher; failure policy incomplete.**
A separate bounded `miner-staging` ThreadPoolExecutor (distinct from the max_workers=1
`miner-cleanup` executor) now runs fetch/verify/rename + inline write/fsync OFF the
dispatch loop. `staging_timeout` bounds the fetch; back-pressure (`reserve` returns None)
loops with a short sleep until capacity frees or the deadline elapses — POSTPONE + resume,
never drop. Hash/staging/timeout failures route through `handle_stripe_failure` with the
appropriate `retryable`; a spooled result with `transfer is None` fails the stripe
explicitly (config error), and a malformed result fails it too. **Gates 41/42/43**: slow
fetch does not block a second connection; a staging timeout is matrix-reassigned with
zero reservation leak; back-pressure postpones then resumes.

**Defect 5 — terminal trial state not mutually exclusive.** Both `mark_trial_aborted`
and `mark_trial_committed` now transition ONLY from `state='running'`. Trace: a committed
trial has `state='committed'` → `mark_trial_aborted`'s `WHERE ... state='running'`
matches zero rows → returns False → abort refused (and the reverse). `TrialCommit` gained
an immutable `commit_event_id` + durable `commit_delivery_status` (default 'none' →
'pending'), delivered after the terminal decision, idempotent by event id. The real
terminal path routes abort through the off-dispatch executor: `fail_trial` calls
`submit_abort(...).result()`, so the synchronous discharge runs on `miner-cleanup`, not
the receive loop. **Gates 44/45/46**: commit-then-abort refused (and reverse); abort runs
off the dispatch thread; duplicate commit delivered once.

**Defect 6 — production call mis-wired (run_id from config filename; params dropped).**
`run_trial_miner` derives `run_id = f"{cfg_stem}_t{trial_number}_{uuid8}"` — never the
raw filename — so consecutive trials get distinct run_ids and stripe IDs cannot collide.
`staging_dir` defaults from `miner_output_dir`. The real `_use_miner` call in
`window_optimizer_integration_final.py` is gated behind
`getattr(coordinator,'use_range_miner',False)` — the PWC and ZMQ branches are untouched,
so coexistence holds — and it propagates the resolved `window_size`/`sessions`/`offset`,
all four caps, `test_both_modes`, and `miner_host='0.0.0.0'` (remote-reachable bind, not
loopback). Trace: two consecutive production-shape trials → distinct run_ids, disjoint
stripe IDs, no PK collision, four families driven, window params preserved. **Gate 47**
asserts all of this with the production call shape (no test-only kwargs).

---

## Cross-cutting confirmations

- **Coexistence:** the integration edit lives entirely inside the `use_range_miner`
  gate; `run_trial_persistent` / `run_trial_zmq_sqlite` paths and imports are unchanged.
- **Concurrency model:** all lifecycle mutations (finalize / matrix / commit) serialize
  under the ledger `_write_lock`; only blocking I/O runs on the `miner-staging` executor;
  abort discharge runs on the `miner-cleanup` executor. No mutation runs in the receive
  loop except via the L1-fenced handlers.
- **No redesign:** the accepted ledger / retry-matrix / staging-contract / abort /
  resolver logic was not restructured; these fixes are localized to path derivation,
  the dispatch method, the two terminal UPDATEs, the executor split, and the run_id +
  integration wiring.
- **Gates encode Beta's probes:** gates 38 and 39 assert the exact values Beta reported
  from its dynamic probes (`attempt1_file_after_stale_finish`,
  `held_reservations_after_duplicate`), so a re-run of Beta's probes should now pass.

## Open items for Michael before commit (housekeeping, non-blocking)

- **`python3_with_venv.sh`** remains an untracked stray — keep it OUT of the Phase-4
  commit (its own one-line commit later).
- **Working-prompt / review docs** in `docs/` are untracked — commit for the trail or
  leave out; Alpha's lean is to keep working prompts out of git.
- **Changed-files set is now 6** (adds `window_optimizer_integration_final.py`); confirm
  the changelog's files table and fallback-parity line both say 6.

## Standing

Team **Alpha** pass — adversarial file-vs-source verification that all six Beta defects
are fixed and gate-covered. NOT the binding gate. Sequence: **Team Beta binding
re-review → Michael commits + dual-pushes** the six code/test deliverables + the
changelog. Given Beta reproduced #1 and #2 with live probes, note that gates 38/39 now
encode those exact probes.
