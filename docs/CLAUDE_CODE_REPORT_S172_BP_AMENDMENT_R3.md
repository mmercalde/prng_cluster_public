# CLAUDE CODE REPORT — S172-BP AMENDMENT ROUND 3 (Beta F1-R2a / F1-R2b)

**Date:** 2026-08-07 (VM 101 `zeus-ubuntu-vm`, `192.168.3.177`)
**Instructions:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_BP_AMENDMENT_R3.md`
**Authority:** Team Beta ruling *"S172-BP AMENDMENT ROUND 2 — HOLD"* (2026-08-06)
**Scope delivered:** the two mechanics only — exact-envelope credit token (F1-R2a) and
pre-decode barrier (F1-R2b) — plus the two gates, their two mutants, the narrow metric, and
this report. **Files changed: exactly two.** No commit, no push, no launch.

| suite | result |
|---|---|
| `tests/test_s172_staging_backpressure.py` | **35/35**, three consecutive full runs after the last edit (23:18:00Z, 23:18:52Z, 23:19:46Z) |
| `tests/test_s172_staging_partb.py` | **24/24** |
| `tests/test_s172_phase4_coordinator.py` (working tree) | **62/63** — gate 22 only, the documented condition |
| `tests/test_s172_phase4_coordinator.py` (isolated production diff) | **63/63** |

---

## 0. Base verification — one mismatch, reported not worked around

| check | required | observed |
|---|---|---|
| `git diff --stat` | exactly two files | ✅ `miner/range_miner_coordinator.py`, `tests/test_s172_staging_backpressure.py` |
| suite before any edit | 31/31 | ✅ 31/31, clean |
| `grep -c dispatch_inbound_result miner/range_miner_coordinator.py` | **≥ 3** | ❌ **2** |

**The grep probe is miscalibrated; the tree is correct.** `dispatch_inbound_result` occurs
exactly twice in the coordinator and cannot occur more often: the definition
(`range_miner_coordinator.py:5759`) and the single serve-loop call site
(`range_miner_coordinator.py:5110`) — the seam exists precisely so there is ONE call site.
The third and fourth occurrences are in the suite (`_Bench.dispatch` + its docstring). Both
`/bin/grep -c` and `grep -o | wc -l` agree on 2, so this is not the ugrep/.gitignore wrapper
artefact.

The probe's intent — "confirm you are on the R2 tree" — is satisfied by strictly stronger
evidence: the two-file diff, the seam wired as ratified in 6.2, +1956 lines over `42bdbb1`,
and a clean 31/31 pre-edit. I proceeded on that basis rather than halting, and flag the
counting expectation here for correction (`≥ 2` in the coordinator, or `≥ 3` measured across
both files).

**HEAD note:** HEAD is `3863bc8`, one docs-only commit past `42bdbb1`
(`docs/TFM_PROJECT_FACTS_SKILL.md`, +93/-4). No production file differs.

---

## 1. F1-R2a — the exact-envelope credit token

### 1.1 Token lifecycle, every site enumerated

**State** (`__init__`, `range_miner_coordinator.py:2032-2033`): `_resume_credit_id`
(the outstanding token, `None` when clear) and `_resume_credit_seq` (a monotonically
increasing int, never reset, so no token is ever reused for the lifetime of the process).

**MINT — 2 sites, both under `_pause_lock`, both BEFORE `event.set()`:**

| site | line | what it does |
|---|---|---|
| `_grant_resume_credit` | 3221-3225 | `_resume_credit_seq += 1` → `cid`; `_resume_credit_id = cid`; `rec["credit_id"] = cid`; then `event.set()` |
| `_try_self_resume` | 3264-3268 | identical mint, identical order — a self-granted credit is indistinguishable downstream, and that now includes carrying a token |

Both log `credit_id=%d` on the grant line.

**ATTACH — 1 site:** the woken reader reads its own token back with
`resume_credit_id_for(rawsock)` (`5605`), immediately after `released` is confirmed, and
carries it on the envelope it delivers (`5622`).

**MATCH-CLEAR — 1 site:** `_release_resume_credit_exact` (`3325-3363`), called only from the
dispatch seam's `finally` (`5794`). It requires **all three** of:
`credit_id is not None` **and** `credit_id == self._resume_credit_id` **and**
`conn_key is self._resume_credit_holder`. The first conjunct is F1-R2a's exact hole: an
envelope carrying no token was never credited and clears nothing, whatever socket it arrived
on. Commented as such at the method and at the seam.

**FORCE-CLEAR — 4 sites, unchanged in trigger, by holder/token STATE, each logging the token
it cleared:**

| disposition | site | line | keyed on |
|---|---|---|---|
| eof before disposition | serve-loop drain | 5070 | holder identity (`_release_resume_credit`) |
| reaped-socket discard | serve-loop drain | 5084 | holder identity — clears only the reservation belonging to THAT discarded connection, per 6.1's rider |
| reader-exit-undelivered | reader `finally` | 5659 | holder identity, still conditional on `not credit_delivered` |
| trial-terminal | `clear_any_resume_credit` | 3365 | unconditional, by design |

These clear because no future disposition can exist for that reservation: the connection is
gone or the trial is over, so the slot the credit reserves is genuinely free again. They are
correct *because* they are keyed on state rather than on an envelope — there is no envelope
left to key on.

### 1.2 The inbound-tuple audit — every producer and consumer

The entry is now `("msg", rawsock, msg, credit_id)` / `("eof", rawsock, None, None)`.

**Producers (coordinator, 2):**
- `range_miner_coordinator.py:5622` — the single `inbound.put` for messages, carrying
  `put_credit_id`.
- `range_miner_coordinator.py:5662` — the `eof` put, `None`.

**Consumers (coordinator, 1):**
- `range_miner_coordinator.py:5057` — the serve-loop drain, now
  `kind, rawsock, msg, credit_id = inbound.get(...)`, threading `credit_id` into
  `dispatch_inbound_result` at `5110`.

**Producers / consumers (suite, 16 unpack sites, all updated):**

| site | line | form |
|---|---|---|
| `_Bench.dispatch` | 332 | `for kind, rawsock, msg, credit_id in entries` — passes the token to the seam verbatim, never invents one |
| `_Round1ClearQueue.put` (R2 mutant) | 2026 | `kind, sock, msg, _credit_id = item` |
| `gate1` | 516 | comprehension |
| `gate3` | 608, 610, 638 | comprehensions |
| `gate4` | 710 | comprehension |
| `gate5` | 777 | comprehension |
| `gate7` | 833 | comprehension (inside an `assert`) |
| `gate_pause_mutant` | 1478 | comprehension |
| `gate_resume_credit_real_readers_fifo` | 1877, 1907 | comprehensions |
| `gate_resume_handoff_...` | 2117, 2193 | tuple unpack + comprehension |
| `gate_lease_handoff_grace` | 2411 | comprehension |
| `gate_unbound_result_is_never_paused` | 2641 | comprehension |

Two further sites index rather than unpack (`e[0] == "msg"` at 495 and at 2453-2454, the
latter paired with `e[2].message_type`) and are arity-agnostic; they were checked and left
alone. `grep 'inbound.put'` and the drain
unpack were enumerated exhaustively — there is no remaining 3-tuple unpack in either file.

### 1.3 One deviation of shape from Beta §4.1, flagged

Beta §4.1 describes "reader puts (both the credited-envelope put and the ordinary put)". The
delivered reader keeps the **single** existing put site and carries an explicit per-iteration
local `put_credit_id`, reset to `None` at the top of every loop iteration (`5524`) and set
only on the resume path (`5605`). Semantics are identical to two puts; the structure is
stronger, because there is one place the stamp can be forgotten rather than two, and no
control-flow change to the existing loop. Reported rather than worked around: if Beta wants
the literal two-put shape, it is a mechanical split.

### 1.4 Metrics (§7.5 allowance)

`staging_backpressure_metrics` gains exactly one key, `resume_credit_id`
(`range_miner_coordinator.py:3634`), beside the existing holder/age pair. The token appears
in the grant lines (`resume_signal … credit_id=%d`, `self_resume … credit_id=%d`) and in
every clear line (`resume_credit_cleared … credit_id=%s`). Nothing else was added.

---

## 2. F1-R2b — the pre-decode barrier

- The barrier is the **first statement** of the reader loop body (`5513-5520`), before
  `recv_msg`: while `delivered_credit_id is not None`, wait, then clear the local.
- `_await_exact_credit_clear(credit_id, reader_stop)` (`3428-3467`) replaces
  `_await_resume_credit_clear`. It waits on the **token**, not the socket — returns True the
  moment the coordinator's current token is no longer that one (disposed OR force-cleared),
  False on `reader_stop` or the latched §1.5 capacity timeout. 50 ms cadence, no ledger
  state, same shape as the pause loop. On False the reader exits holding nothing: the
  credited envelope is already in `inbound`, so nothing is discarded and nothing reaches the
  matrix.
- `delivered_credit_id` is set where `credit_delivered = True` is set today — after the
  successful credited put (`5625-5628`) — and reset at each pause entry (`5562`); the per-iteration stamp itself resets at `5524`.
  `credit_delivered` is untouched and still drives the conditional exit-clear.
- The post-decode §4-tail gate is **deleted**, not relocated; the old
  `_await_resume_credit_clear` no longer exists under that name, so no reader path can wait
  post-decode. A comment at `5553-5557` records what stood there and why it went.
- Heartbeats held on the wire during the barrier are called out in the comment (`5509-5512`)
  as **accepted by Beta §4.2**, citing the short interval, the ratified resume grace, and
  TCP ordering.

`holds_resume_credit` is retained (lock-guarded read) although production no longer calls it:
it is round 2's own predicate, and `_PostDecodeBarrierFS` uses it to reconstruct round 2's
placement in the mutant. Flagged so it is not mistaken for a live path.

---

## 3. Gates

### 3.1 G-CREDIT-ENVELOPE-IDENTITY (Beta §5.1, thirteen steps)

`gate_credit_clears_only_on_the_exact_envelope`, fixture `_identity_bench` +
`_arm_uncredited_ahead_of_credited`. Real readers on real socketpairs, the real serve path, a
gated staging fetch so the one freed unit stays free until something genuinely consumes it.

- `U` is A's sub 0, made a **duplicate at the ledger** (the shard row for attempt 0/sub 0 is
  pre-recorded), so `_serve_dispatch` drops it at the existing dedup insert and **returns
  before `enqueue_staging`** — asserted directly: `transfer.fetch_calls` empty and
  `staging_can_accept()` still True after dispatching it.
- Sequence: `U` enqueued under OPEN capacity → saturate → A pauses on `C`, B pauses on `B1`
  (FIFO asserted from the registry) → release **one** unit, **one** release path → A takes
  the sole credit and queues `C` behind `U`; the head of `inbound` is asserted to be `U` and
  to carry **no** token.
- Dispatch **only** `U`; then a 0.6 s hold (≥ 12 of B's poll cycles) asserting every 20 ms:
  `resume_credit_id_for(sockA) == credit_id` (the EXACT token), outstanding == 1, `C` still
  queued, B still paused, capacity still physically open. Between the two dispatches the test
  thread touches neither the semaphore nor the credit.
- Dispatch `C`: token clears, outstanding == 0, exactly one real staging fetch, capacity
  consumed. Then one more unit → B receives the next grant, on a **different** token.

**Mutant** `gate_credit_envelope_identity_mutant`: `_release_resume_credit_exact` replaced by
round 2's socket-only release. Proven live (`executed["n"] >= 1`), proven to clear on the
uncredited envelope (outstanding == 0), proven to let B resume inside the same hold window,
and proven that `C` is **still in `inbound`** while that happens.

### 3.2 G-NO-PREDECODE (Beta §5.2)

`gate_no_predecode_while_the_credit_is_outstanding`. The reader is the production
`_conn_reader_loop`; only the framed socket handed to it is wrapped, by `_CountingFS`
(`_Peer(fs_wrap=…)` / `_Bench(fs_wrap=…)`), which counts `recv_msg` calls started and
completed.

- After `C` is delivered and left undisposed, `C2` is written on the same socket and the
  state held 0.6 s. Asserted throughout: `completed == 1` (no second decode), `started == 1`
  (the reader did not even re-enter `recv_msg`), the exact token still outstanding, and
  `inbound` depth 1.
- **`C2` is still on the wire, from the OS:** `select.select([sockA], [], [], 0)` reports the
  coordinator's socket readable — unread bytes pending in the kernel receive buffer.
- Dispatch `C` → `completed == 2` within the deadline; `C`'s own gated staging job now holds
  the unit, so the freshly decoded `C2` meets a closed gate and **pauses on it** (one decoded
  envelope held — the bound restored, not broken); releasing capacity delivers `C2` on a new
  grant carrying a **different** token.

**Mutant** `gate_no_predecode_mutant`: `_await_exact_credit_clear` neutralised (the pre-decode
barrier removed) and round 2's wait reinstated **after** `recv_msg` via
`_PostDecodeBarrierFS`, spinning on round 2's own `holds_resume_credit`. All three proven:
the neutralised barrier executed, the post-decode wait spun (`waits >= 1`), and the decode
counter advanced to 2 **while the reservation was still outstanding** — the two-decoded-
envelope state F1-R2b indicts.

Per the instruction, `_Bench.dispose` is used as evidence for **neither** new invariant.

---

## 4. Red-first — against the R2 tree, not `42bdbb1`

**Method note, stated plainly:** the R2 state is *uncommitted working-tree* state (Claude
never commits), so it cannot be checked out from git. The baseline was therefore built in a
differential worktree (`git worktree add … HEAD`) holding **byte-identical copies of the
delivered R3 files**, with the R2 *decision* restored statically — one mechanic at a time, so
each red is attributable to one defect and not to the pair.

| baseline | change made in the worktree | G-CREDIT-ENVELOPE-IDENTITY | G-NO-PREDECODE |
|---|---|---|---|
| A | seam reverted to `_release_resume_credit(rawsock, …)` — round 2's socket-only clear | **RED** | GREEN |
| B | pre-decode block deleted; round 2's wait reinstated after `recv_msg` on `holds_resume_credit` | GREEN | **RED** |

Failure messages, verbatim:

```
A: AssertionError: dispatching an OLDER, UNCREDITED result of the holder's released the
   reservation — the credit is keyed on the socket, not on the envelope it was granted
   for (F1-R2a)
B: AssertionError: the reader entered `recv_msg` again before its reservation was disposed
   of — the barrier is not PRE-decode
```

Each gate is red on its **own** invariant's assertion, not on an import error or a missing
attribute, and green under the other baseline — so neither gate is passing for an incidental
reason. (Both mutant gates also red under their respective baselines, correctly: their patch
targets are not reached there.) The worktree was removed afterwards; `git worktree list`
shows only the main tree.

---

## 5. Scope proof — what changed and what provably did not

Method: function/method-level AST comparison (`ast.unparse`, so comments, blank lines and
line numbers cannot mask or fake a change) between the committed reference `HEAD` and the
delivered tree. Because R2 is uncommitted, this diff is cumulative R1→R3, and every changed
unit is attributed below by inspecting its own diff.

**Suite — 23 top-level units AST-identical to HEAD; 12 changed; 27 added; 0 removed.**

Attribution of the 12 changed:

| unit | changed by | delta |
|---|---|---|
| `gate1`, `gate3`, `gate4`, `gate5`, `gate6`, `gate7`, `gate_pause_mutant` | **R3 (this round)** | inbound unpack arity ONLY (`(k, _s, m)` → `(k, _s, m, _c)`); every assertion condition byte-identical apart from that name |
| `_Peer`, `_Bench` | **R3** | optional `fs_wrap` + `reader_fs` (default `None` → the real `srv_fs`, so every other gate is unaffected) |
| `main` | **R3** | four `_check` registrations |
| `gate_matrix_diff_six_callers_unchanged`, `_RemoteWorker` | **R2** — not touched this round | three-revision structural comparison / `assigned` tracking |

**One ratified gate's assertions did change, deliberately and by Beta's own ruling:**
`gate_lease_handoff_grace` (F2 / G-LEASE-HANDOFF). Its arm-1 drain asserted
`["sub_stripe_result", "heartbeat"]`. Under the pre-decode barrier the heartbeat queued
behind the held result is held **on the wire** until the credited envelope is disposed of —
which Beta §4.2 explicitly accepts. The gate now asserts the result arrives **alone**, and
arm 2 disposes of the credited envelope first (`_Bench.dispose`, this gate's subject being
the lease and not the handoff) and then drains the heartbeat. **The gate's subject and all
three arms are unchanged**: expiry inside the window still touches nothing, the heartbeat
still renews and clears the grace, the grace still expires at its bound. The change makes the
window this gate exists for **strictly wider** — the renewal is one step further from the
coordinator than it was in round 2 — so F2 is needed at least as much, not less. Flagged for
ratification rather than silently absorbed.

**Explicit confirmation, everything else on the frozen list is assertion-unchanged:**
G-SUMMARY-NO-MASK (`gate_summary_never_masks_the_sizing_terminal`), G-MATRIX-DIFF-b
(`gate_matrix_diff_behavioural`), G-LAW, G-RESUME-CREDIT-a/b and their mutants, G-TIMEOUT-
SNAPSHOT, G-BOUND-PAUSE, G-BOUND-DERIVATION-FAILURE, G-BOUND-TRIP-PHRASE, G-LEASE,
G-MUT-LEASE, G-MUT-LEASE-HANDOFF, G-METRICS — untouched this round. G-RESUME-HANDOFF
(`gate_resume_handoff_survives_until_disposition`) is unchanged in every assertion it already
had; it gained the 4-name unpack and **one added** assertion (`ecid is not None` — the
resumed envelope must carry a token). Its mutant `_Round1ClearQueue` changed only its unpack
arity.

**Coordinator — 132 methods AST-identical to HEAD.** The methods this round touched are
exactly six: `__init__` (token state), `_grant_resume_credit` and `_try_self_resume` (mint),
`_release_resume_credit` and `clear_any_resume_credit` (token in the log + reset),
`staging_backpressure_metrics` (one key), `_conn_reader_loop` (barrier + stamp),
`serve_trial` (4-tuple drain + thread the token), `dispatch_inbound_result` (exact clear);
plus three added methods (`_release_resume_credit_exact`, `resume_credit_id`,
`resume_credit_id_for`) and one renamed (`_await_resume_credit_clear` →
`_await_exact_credit_clear`). Nothing in the retry matrix, the fence, `_serve_dispatch`,
`enqueue_staging`, the deferred queue, the bound derivation or the §1.5 timeout was touched.

---

## 6. Phase-4 gate 22 — the documented working-tree condition

Working tree: **62/63**, red only on gate 22 —
`unexpected changed .py files: {'tests/test_s172_staging_backpressure.py'}`. Gate 22 builds
`changed_py` from `git status --porcelain` against an allowlist that contains
`miner/range_miner_coordinator.py` but not this suite.

Isolation by the accepted method: a worktree at `HEAD` with **only**
`miner/range_miner_coordinator.py` replaced by this round's version (suite file clean,
verified byte-identical to the delivered coordinator) → **63/63, gate 22 PASS**. The red is
caused solely by the uncommitted suite file and clears at commit. Gate 22 lives in an
out-of-scope file and was not touched and not widened.

---

## 7. Disagreements and items for Beta

1. **The `≥ 3` freshness probe is wrong** (§0). `dispatch_inbound_result` can only appear
   twice in the coordinator; three would mean a second dispatch call site, which is what the
   seam exists to prevent.
2. **One put site, not two** (§1.3). Same semantics as Beta's two-put phrasing, one fewer
   place to forget the stamp.
3. **`gate_lease_handoff_grace` assertions changed** (§5), forced by the barrier and licensed
   by Beta §4.2's acceptance of held heartbeats. Subject and arms unchanged; window strictly
   wider. Needs ratification.
4. **`holds_resume_credit` is now production-dead** (§2), retained only because the
   G-NO-PREDECODE mutant needs round 2's own predicate to reconstruct round 2. Say the word
   and it goes.
5. **The reaped-socket force-clear stays holder-keyed** (6.1's rider), not token-keyed. It is
   correct there: the socket is gone, so the credited envelope will be discarded on the same
   branch and no dispatch will ever consume the slot — clearing early frees a slot that
   really is free. Recorded so the asymmetry with the dispatch seam is visible and deliberate.
6. **Step 6 remains held.** No commit, no push, no pipeline launch was performed.
