# CLAUDE CODE INSTRUCTIONS — S172-BP AMENDMENT ROUND 3 (Beta F1-R2a / F1-R2b)

**Authority:** Team Beta ruling *"S172-BP AMENDMENT ROUND 2 — HOLD"* (2026-08-06). Scope is
Beta §7 EXACTLY: (1) exact-envelope credit tokenization, (2) pre-decode barrier, (3)
G-CREDIT-ENVELOPE-IDENTITY + mutant, (4) G-NO-PREDECODE + mutant, (5) narrow metrics, (6)
report. Files: `miner/range_miner_coordinator.py`, `tests/test_s172_staging_backpressure.py`
ONLY. Expected suite total: **35** (Beta §5.2).

**RATIFIED AND CLOSED — do not touch, do not re-argue:** F2–F5, the lease exemption AND
resume grace (Beta 6.5: fully ratified, no longer an open item), the summary guard,
G-RESUME-HANDOFF, the `conn_dropped` clear (6.1), the dispatch seam (6.2 — but it MUST now
receive the token), gate arithmetic (6.4), bench resequencing (6.6 — note `_Bench.dispose`
may NOT be used as evidence for either new invariant).

**Base:** current VM101 working tree (= `42bdbb1` + cumulative R2 state that ran 31/31 ×3).
Verify first: `git diff --stat` exactly two files; `grep -c dispatch_inbound_result
miner/range_miner_coordinator.py` ≥ 3; suite runs 31/31 once before any edit. STOP and
report on any mismatch. No commit / push / launch.

---

## 1. The two defects, so the fix is understood

**F1-R2a:** the clear checks only `rawsock is holder`. An OLDER, uncredited result `U` from
the credit-holder's connection — queued in `inbound` BEFORE the pause ever happened —
dispatches first, is fence-rejected (consumes no capacity), and its `finally` clears the
credit. The credited `C` is still queued; the slot is still free; B self-resumes on it.
"First result after resume" was the wrong identity: it excludes LATER traffic, not EARLIER.

**F1-R2b:** the §4-tail wait runs AFTER `recv_msg`. While credited `C` sits in `inbound`,
the reader has already decoded `C2` into its local `msg` — the connection owns TWO decoded
envelopes, breaking the one-decoded-envelope bound the resume margin is derived from. The
wait must move BEFORE the decode so `C2` stays on the wire.

## 2. Implementation — Beta §4.1/§4.2 as written

### 2.1 Exact credit token
- Credit record becomes `{credit_id, holder_socket, holder_worker, granted_at}` under
  `_pause_lock`. `credit_id`: monotonically increasing int (`self._resume_credit_seq += 1`)
  — unique per grant, immutable, log-friendly. Both grant paths (`_grant_resume_credit`,
  `_try_self_resume`) mint it BEFORE `event.set()` and store it in the pause rec
  (`rec["credit_id"] = cid`) so the woken reader can read its own token; also expose
  `resume_credit_id_for(conn_key) -> Optional[int]` (lock-guarded) for the reader and gates.
- **Inbound tuple gains a fourth field:** `("msg", rawsock, msg, credit_id)` with
  `credit_id=None` for every ordinary message, and `("eof", rawsock, None, None)`.
  **Audit EVERY producer and consumer of `inbound` and update all unpack sites** — serve
  loop drain, reader puts (both the credited-envelope put and the ordinary put), bench
  `drain`/`pump`/`dispatch`. List each touched site in the report; a missed 3-tuple unpack
  is a crash, so grep `inbound.put` and the drain unpack and enumerate.
- `dispatch_inbound_result(msg, rawsock, ..., credit_id)` clears ONLY when
  `credit_id is not None and credit_id == current.credit_id and rawsock is
  current.holder_socket` — an uncredited result from the same socket NEVER clears
  (F1-R2a's exact hole). Comment it as such.
- **Force-clear paths stay, by holder/token STATE** (Beta §4.1 list): eof-before-
  disposition, reaped-socket discard (must clear only the reservation belonging to THAT
  discarded connection — check holder identity, per 6.1's rider), reader-exit-undelivered,
  trial-terminal `clear_any_resume_credit`. These clear because no future disposition
  exists, and each logs the token it force-cleared.

### 2.2 Pre-decode barrier
- Restructure the reader loop to Beta's shape: at the TOP of the loop, before `recv_msg` —
  `if delivered_credit_id is not None: await exact clear, then None`. The wait loop
  (`_await_exact_credit_clear(credit_id, reader_stop)`): returns True when the
  coordinator's current credit_id != that id (disposed or force-cleared); False on
  `reader_stop` or the latched §1.5 capacity timeout → reader exits (nothing held — the
  envelope is already delivered; nothing routed to the matrix). 50 ms cadence, no ledger
  state, same as existing helpers.
- `delivered_credit_id` is set where `credit_delivered = True` is set today (after the
  successful credited `inbound.put`), reset at each pause entry. Keep `credit_delivered`
  for the conditional exit-clear exactly as is.
- **Remove the post-decode §4-tail gate** (`holds_resume_credit` check +
  `_await_resume_credit_clear` call at the result gate) — the pre-decode barrier replaces
  it; a post-decode wait is what F1-R2b indicts. Delete or repurpose
  `_await_resume_credit_clear` accordingly (if kept for the new helper, rename so no
  reader path waits post-decode).
- Heartbeats held on the wire during the barrier are ACCEPTED by Beta §4.2 (short interval;
  the ratified resume grace covers the lease; TCP ordering makes bypass impossible anyway).
  Say so in the comment, citing the ruling.

### 2.3 Metrics (§7.5 allowance)
Add `resume_credit_id` to `staging_backpressure_metrics` beside holder/age. Include the
token in the grant/clear log lines (`credit_id=%d`). Nothing else.

## 3. Gate G-CREDIT-ENVELOPE-IDENTITY — Beta §5.1, all thirteen steps

Real readers, real serve path. Under OPEN capacity A enqueues uncredited `U` (do not
dispatch). Saturate. A pauses on `C`; B pauses on `B1`. Release ONE unit → A takes the sole
credit, queues `C` behind `U`. Dispatch ONLY `U`, constructed as a duplicate/stale so the
fence rejects it and it consumes no capacity. Assert: A's EXACT token still outstanding
(`resume_credit_id_for` unchanged), `C` undispatched, B still paused, capacity physically
open. Dispatch `C` → assert the exact token clears. Then assert B receives the next valid
grant. **Mutant:** restore socket-only release (ignore the token in the seam), prove it
executes, prove dispatching `U` clears the credit and B resumes BEFORE `C` is dispatched.

## 4. Gate G-NO-PREDECODE — Beta §5.2

Instrument the REAL reader's `recv_msg` call count (wrap the framed-socket object handed to
the reader with a counting proxy — the reader must be the production `_conn_reader_loop`).
Deliver credited `C` into `inbound`, leave it undispatched, send `C2` on the same socket,
hold ≥ 0.6 s. Assert: no additional `recv_msg` completed; `C2` still on the wire (peer-side
send state or counter); only `C` decoded for that connection; credit outstanding. Dispatch
`C` → assert the reader then decodes `C2`. **Mutant:** restore the post-decode barrier
placement; prove the decode counter advances while `C` is undisposed.

## 5. Evidence & report

Final-state discipline (now standing): last edit → VM101 runs → report → cover. Deliver
`docs/CLAUDE_CODE_REPORT_S172_BP_AMENDMENT_R3.md` with: the token lifecycle (mint → attach
→ match-clear / force-clear, every site enumerated); the inbound-tuple audit (every
producer/consumer touched); red-first for both new gates against the R2 state (the mutants
ARE R2 behavior — red baseline is the current tree pre-edit, via worktree/stash, NOT
`42bdbb1`); 35/35 on VM101 ×3 after the last edit; Part B 24/24 VM101; phase-4 63/63 by the
accepted isolated-diff method; explicit confirmation that F2–F5, summary, matrix-diff,
G-RESUME-HANDOFF and G-SUMMARY-NO-MASK gates are assertion-unchanged (programmatic diff, the
round-2 method); files-changed (exactly two); disagreements reported, not worked around.
