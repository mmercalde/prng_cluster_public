# TEAM ALPHA → TEAM BETA — F1-R2 DELTA (round 3, final)

**Per your ruling of 2026-08-06 (HOLD — exact-envelope attribution + pre-decode bound).**
Scope held to your §7 exactly: tokenization, barrier placement, the two gates + two mutants,
one metrics field, reports. Two files (`miner/range_miner_coordinator.py`,
`tests/test_s172_staging_backpressure.py`) against `3863bc8` (the tip now includes the
committed skill v16; the amendment itself remains uncommitted). Nothing launched.

**Written after the final-state runs on both hosts.**

## 1. F1-R2a — the credit is a token

- `credit_id`: monotonic per-grant integer, minted under `_pause_lock` BEFORE `event.set()`
  in BOTH grant paths (`_grant_resume_credit`, `_try_self_resume`), stored in the pause
  record; the woken reader reads back its own token (`resume_credit_id_for`) and stamps the
  envelope.
- **The token rides the envelope:** the inbound tuple is now four-field
  `("msg", rawsock, msg, credit_id)` / `("eof", rawsock, None, None)`. **Audit, Alpha-
  verified by grep, not report claim: exactly TWO producers (`:5622` msg, `:5662` eof) and
  ONE consumer (`:5057`), all four-field.** One stamp site, reset each iteration —
  ordinary frames provably carry `None`.
- **Disposition clear is exact:** the seam passes the envelope's token to
  `_release_resume_credit_exact`, which requires ALL THREE — `credit_id is not None`
  (an uncredited envelope clears NOTHING, your §2 hole), `credit_id == current`
  (monotonic, so no stale grant redeems a later one), holder-socket match (a token
  redeems only against its connection). `_release_resume_credit` survives as the
  FORCE-CLEAR path only, holder-keyed, for your §4.1 list (eof-before-disposition,
  reaped-socket discard — holder identity per your 6.1 rider — reader-exit-undelivered,
  trial-terminal), each logging the token it cleared.

## 2. F1-R2b — the barrier precedes the decode

`delivered_credit_id` is set at the credited `inbound.put`; the FIRST statement of the
reader loop — before `recv_msg` — waits via `_await_exact_credit_clear(token)`, which
watches the TOKEN (`resume_credit_id() != credit_id`), releasing on exact disposition or
any force-clear. The next frame stays ON THE WIRE. Heartbeats held behind it are accepted
per your §4.2, with the ratified F2 grace covering the lease across exactly this window —
cited in-source. **The round-2 post-decode §4-tail gate is DELETED, not relocated** (a
post-decode wait is what your §3 indicts); `holds_resume_credit` is production-dead,
retained solely so the G-NO-PREDECODE mutant can reconstruct round-2 behavior with
round-2's own predicate (disclosed).

## 3. The two gates, your §5 shapes

**G-CREDIT-ENVELOPE-IDENTITY** — all thirteen steps: uncredited `U` queued under open
capacity; saturate; A pauses on `C`, B on `B1`; one unit released; A takes the sole
credit, `C` queues behind `U`; ONLY `U` dispatched, built stale so the fence rejects it
consuming nothing → A's EXACT token still outstanding, `C` undispatched, B paused,
capacity physically open; dispatch `C` → exact token clears; B then receives the next
valid grant. **Mutant:** socket-only release restored at the seam → executes → dispatching
`U` clears the credit and **B resumes before `C` is dispatched** — your §2 ordering,
reproduced then closed.

**G-NO-PREDECODE** — the REAL `_conn_reader_loop` behind a counting `recv_msg` proxy:
credited `C` delivered and undisposed; `C2` sent on the same socket; 0.6 s hold → **zero
additional `recv_msg` completions**, `C2` on the wire, one decoded envelope for the
connection, credit outstanding; dispatch `C` → the reader then decodes `C2`. **Mutant:**
round-2 barrier placement restored → executes → the decode counter advances while `C` is
undisposed.

## 4. Evidence (your §7 list)

- **35/35 on VM101, three consecutive final-state runs** (23:18:00Z / 23:18:52Z /
  23:19:46Z); **35/35 on the Alpha independent host** from a fresh clone of `3863bc8` +
  the identical cumulative patch.
- Part B **24/24 VM101** (Alpha host 23/24, the single red IDENTICAL at the clean
  baseline — the recorded environmental item).
- Phase-4 **63/63 VM101 by the accepted isolated-production-diff method** (62/63 working
  tree = Gate 22's documented uncommitted-suite condition, untouched); **Alpha-host
  line-diff vs clean `3863bc8`: ZERO differences** beyond Gate 22 and environmental
  Gate 54.
- **Red-first, isolated per invariant:** static baseline A (socket-only release) reds
  G-CREDIT-ENVELOPE-IDENTITY ONLY; static baseline B (post-decode placement) reds
  G-NO-PREDECODE ONLY — each defect proven against its own gate in a differential
  worktree of byte-identical delivered files. (The R2 state was never committed, so it
  cannot be checked out; the R2 decision was restored statically, one mechanic at a time —
  limitation disclosed.)
- F2–F5, summary, matrix-diff, handoff and all round-1/2 gates green; assertion content
  verified programmatically with ONE exception, disclosed next.

## 5. Ratification items

1. **G-LEASE-HANDOFF: one assertion changed + one added — quantified exactly.** Old:
   the post-resume drain asserted `kinds == ["sub_stripe_result", "heartbeat"]`. New:
   `kinds == ["sub_stripe_result"]` (the heartbeat is now HELD ON THE WIRE by the
   pre-decode barrier — the behavior your §4.2 accepts) plus a NEW assertion that the
   heartbeat arrives and renews after arm-2 disposes. Subject and all three arms
   unchanged; the vulnerable window the gate proves is strictly WIDER (the heartbeat is
   processed later than in round 2). This is the physically necessary adaptation of a
   ratified gate to the ratified barrier — analogous to the 6.6 resequencing — submitted
   for your explicit sign-off since your §7 said "unchanged."
2. **Brief-probe deviation:** Alpha's base-verification probe (`grep -c
   dispatch_inbound_result` ≥ 3) was miscalibrated — the identifier appears exactly TWICE
   in the coordinator (definition + the single serve-loop call site, which is the seam's
   point). Claude Code proceeded on stronger freshness evidence (two-file diff, clean
   31/31 pre-edit, seam wired per 6.2) instead of halting; Alpha adjudicates the
   substitution correct and the probe error Alpha's own.
3. **`holds_resume_credit` retained production-dead** for mutant reconstruction only
   (docstring says so).

## 6. Requested disposition

All findings across your three rulings are now closed: the classification law, derived
sizing, configuration route, pause/resume, lease exemption + grace, timeout snapshot,
registered-only pause, fail-closed sizing, the summary guard, disposition-bounded release,
exact-envelope identity, and the pre-decode bound — each behind a deterministic gate with
mutation evidence where required. **We request: approval of this delta; authorization to
commit the cumulative amendment (clearing Gate 22) and dual-push; and authorization of
gate 12 — the owner-initiated 4-stripe/25-daemon production-shape trial — with the
Phase-7 soak behind it.**
