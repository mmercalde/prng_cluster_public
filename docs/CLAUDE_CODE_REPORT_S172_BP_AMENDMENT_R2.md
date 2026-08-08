# TEAM ALPHA REPORT — S172-BP AMENDMENT ROUND 2 (Beta F1-R)

**Authority:** Team Beta ruling *"S172-BP F1–F5 AMENDMENT DELTA — HOLD"* (2026-08-06), §8.
**Instructions:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_BP_AMENDMENT_R2.md`.
**Canonical host:** VM 101 `zeus-ubuntu-vm` / `192.168.3.177`, venv `~/venvs/torch`.
**Base commit:** `42bdbb1` (working tree = 42bdbb1 + round-1 amendment + Alpha's
terminal-summary patch). **Nothing committed, nothing pushed, no pipeline launched.**
**Final canonical runs:** 2026-08-07T03:04:12Z — *after* the last change to either file.

---

## 0. Base verification (§"Base", run before any edit)

| check | result |
|---|---|
| `git stash list` | empty |
| `git diff --stat` | exactly two files (`miner/range_miner_coordinator.py`, `tests/test_s172_staging_backpressure.py`) |
| `grep -c "bound_in_force_error" miner/range_miner_coordinator.py` | **1** — Alpha's terminal-summary patch **present** |

The round-1 state was snapshotted **before the first edit** into a worktree
(`git worktree add … 42bdbb1` with the two round-1 files copied in) and independently
re-verified there: **28/28 green**. Every red-first claim below is measured against that
worktree, not against a reconstruction.

---

## 1. F1-R — the defect, and the invariant

Round 1 cleared the ingress credit at `inbound.put`. That is **ingress, not consumption**.
The freed staging slot is consumed only when the serve loop later dispatches that envelope
into `enqueue_staging`. In the gap — envelope in `inbound`, slot still physically free —
reader B finds: B is FIFO head (A deregistered), `credits == 0` (A cleared at put), and
`staging_can_accept()` true **on the same slot**. Two wakes, one slot.

**The reservation now ends at DISPOSITION (Beta §4 i–iv), never at ingress.** The comment
carrying that sentence is at `_release_resume_credit`
(`miner/range_miner_coordinator.py:3222`), where the round-1 semantics lived.

### 1.1 Beta Option A as implemented

1. **Reader no longer clears on delivery.** After the successful `inbound.put` the reader
   sets a local `credit_delivered = True` and calls nothing. The reservation rides with the
   envelope (`_conn_reader_loop`).
2. **Serve-side disposition clear.** `dispatch_inbound_result` calls `_serve_dispatch`
   verbatim and, in a `finally` covering accepted / deferred / fenced / rejected /
   exception alike, releases the credit when the message is a `sub_stripe_result`. The
   clear runs **after** the dispatch call returns, never before.
3. **eof clear**, **already-dropped-socket clear**, **trial-terminal clear** — disposition
   (iv), see the table in §1.2.
4. **Reader-exit clear is now conditional** on `credit_delivered` being False. The
   deregister-before-clear ordering comment is kept and extended with this rule.
5. **No second result while the reservation is out** (Beta §4 tail): at the reader's result
   gate, `_await_resume_credit_clear` blocks on a 50 ms cadence honouring `reader_stop` and
   the latched §1.5 timeout, with discard-and-exit semantics identical to the pause loop,
   then falls through to the normal capacity gate. Heartbeats / completions / registers are
   untouched — the §1.4 lease exemption depends on that and is unaffected.
6. **Metrics (§8.4 allowance):** `resume_credit_holder_worker` and `resume_credit_age_s`
   (both `None` when clear) added to `staging_backpressure_metrics`. Existing counters
   unchanged. Nothing else.

### 1.2 Every clear path, enumerated — and each one executed

`disposition` is a **log-only** argument (default `"dispatch"`); it is read by nothing and
carries no behaviour. Lines below are captured from live runs on VM101:

| Beta disposition | site | log line observed |
|---|---|---|
| (i) admission acquired · (ii) retained in `_deferred` · (iii) fence rejected | `dispatch_inbound_result` `finally` | `resume_credit_cleared delivered=True disposition=dispatch credits_outstanding=0` |
| (iv) connection terminated | serve loop `kind == "eof"` arm | `resume_credit_cleared delivered=False disposition=eof credits_outstanding=0` |
| (iv) trial terminated | `clear_any_resume_credit` at trial-terminal cleanup | `resume_credit_cleared delivered=unknown disposition=trial_terminal worker=hostA:gpu0 credits_outstanding=0` |
| (iv) wake delivered nothing | reader `finally`, guarded by `not credit_delivered` | `resume_credit_cleared delivered=False disposition=reader_exit_undelivered credits_outstanding=0` |
| (iv) socket already reaped | serve loop `rawsock not in fs_by_sock` skip | `disposition=conn_dropped` |

**Reported, not worked around — a fifth site.** The instructions name four clear paths; the
implementation has five. The extra one is the serve loop's `rawsock not in fs_by_sock`
skip: an envelope whose socket was reaped between `inbound.put` and the drain is
*discarded there*, never dispatched, and no eof necessarily follows on a path that already
`continue`d. That is literally Beta's disposition (iv) — "the connection … terminated and
the envelope was discarded" — occurring one line earlier than the eof arm, not a new class
of disposition. Flagged for ratification.

**Why the clear fires on exactly the credited envelope.** Dispatch is single-threaded; at
most ONE credit exists globally; `_release_resume_credit` clears only for its holder; and
the credited envelope is by construction the FIRST `sub_stripe_result` that connection
delivers after its resume, because item 5 holds any further result until this clear lands.

### 1.3 Why `_serve_dispatch` was not edited, and what was added instead

Beta §5 requires the gate to "dispatch A's envelope through the REAL serve path".
`_serve_dispatch` is byte-unchanged (AST-verified, §5 below); the `finally` lives in a new
thin seam, `dispatch_inbound_result`, which the serve loop now calls instead of calling
`_serve_dispatch` directly. This is a **seam, not a second dispatch path**: it means the
clear cannot be forgotten at one call site, and it means G-RESUME-HANDOFF drives the same
production function object the serve loop does rather than modelling it. Round 1's failure
was precisely the modelling.

---

## 2. Gate G-RESUME-HANDOFF (Beta §5, all eleven steps)

`gate_resume_handoff_survives_until_disposition`. Two REAL paused reader threads on real
framed socketpairs, real staging admission semaphore, real ledger.

1–3. A pauses first, then B; FIFO asserted from the registry (`["hostA:gpu0","hostB:gpu0"]`).
A additionally has a **second result queued behind the first**, so Beta §4's tail is proven
by the same hold. 4–5. Exactly ONE unit freed; exactly ONE release path invoked
(`_release_capacity()`). 6. Wait until A's envelope is in `inbound` **and** A has left the
pause registry. 7. **The test thread neither dispatches nor touches the semaphore** —
asserted positively: `staging_can_accept()` must still be True, i.e. the freed unit is
still free. 8–9. Hold 0.6 s (≥ 12 defensive poll cycles), asserting on every cycle: B still
paused; `inbound` still empty (no second envelope); `resume_credits_outstanding() == 1`.
10. Dispatch through the real seam, with a spy inside `_serve_dispatch` recording the credit
count — `seen == [1]` proves the reservation was **still outstanding when dispatch began**,
and `== 0` after proves it ended at disposition; a real `fetch_remote` then confirms the
freed unit was consumed by production code (`staging_can_accept()` False). 11. A re-pauses
**behind** B (registry `["hostB:gpu0","hostA:gpu0"]`); the next unit is freed and B resumes
second, A stays paused. FIFO preserved across the handoff.

The staging job holds its slot for the whole gate via a `_GatedTransfer` whose fetch blocks,
so capacity cannot reopen behind the gate's back — the freed unit is consumed by
`enqueue_staging` and by nothing else.

### 2.1 Red-first evidence — against the round-1 worktree

Worktree carrying **round-1 + Alpha patch** (not bare `42bdbb1`), the new suite copied in.
The gate reds on **behaviour**, before touching any new API, and the whole round-1 failure
cascade is reachable by suppressing each assertion in turn (evidence-only variants, in the
scratch worktree — the delivered gate asserts all three):

| # | assertion | round-1 result |
|---|---|---|
| 1 | `resume_credits_outstanding() == 1` during the hold | **RED** — `the reservation ended at INGRESS — inbound.put moves the envelope, it does not consume the slot` |
| 2 | `inbound.qsize() == 0` during the hold | **RED** — `a second envelope reached inbound while the reservation was outstanding` (Beta §4 tail) |
| 3 | B still paused during the hold | **RED** — `B resumed while A's envelope was still undispatched — two wakes on ONE unconsumed slot` (Beta §2) |

Assertion 3 is Beta's §2 schedule, reproduced exactly.

### 2.2 Required mutant — executed, and it reds the gate

`gate_resume_handoff_mutant`. The mutant **restores the round-1 clear-at-`inbound.put`** by
wrapping the bounded `inbound` queue so that a successful `put` of a `sub_stripe_result`
calls `_release_resume_credit(sock, delivered=True)` — the same clear, in the same thread,
at the same instant, without rewriting `_conn_reader_loop`. Fixture and hold window are
identical to the gate.

* **Executed:** `executed["n"] >= 1` asserted; the mutant fires on the credited envelope.
* **Reds the invariant:** B resumes **during** the 0.6 s hold, on the still-unconsumed unit.
* Green on the current tree (the mutation is real and reproduces the defect on demand), and
  the mutation is confined to the injected queue — production code is untouched by it.

### 2.3 G-RESUME-CREDIT-b (round-1 part b) — kept, and told the truth about

Per §3 it stays as a **capacity-accounting** gate. Its docstring now states plainly that it
**does NOT cover the handoff invariant** (Beta §5 last line): it reclaims the freed unit
from the test thread immediately after the grant, "modelling the serve loop", and that
reclaim deletes exactly the interval the reservation has to survive — which is why it was
falsely green in round 1. What it still legitimately proves is retained: one freed unit
resumes exactly one reader, FIFO-first, and a second unit is needed for the second reader.
It was also repaired for the new semantics: it now asserts the reservation is outstanding
after delivery, disposes of it through the production seam (disposition (ii) — capacity is
closed, so the envelope is retained in the bounded deferred queue), and only then expects
the second reader to be grantable.

---

## 3. Gate G-SUMMARY-NO-MASK (Beta §7)

`gate_summary_never_masks_the_sizing_terminal`. A malformed cap record is injected after the
real `assign_stripes` and **left malformed** through terminal summary construction — the
existing F5 gate restores it from the abort callback and therefore never reaches this code.
That F5 gate is **not modified** (AST-identical, §5). Asserted:

* `run_trial_miner` returns normally (no exception out of `serve_trial`);
* the primary abort reason still leads `coordinator_staging_sizing:`;
* `bound_in_force is None`;
* `bound_in_force_error` names the derivation exception (`ValueError`);
* the `[S172-BP] summary` line still emits, carrying `bound_in_force=None`.

### 3.1 Red-first evidence — and a disagreement with §5's wording, reported

**G-SUMMARY-NO-MASK cannot red against round-1 + Alpha patch, because the Alpha patch *is*
the fix this gate covers.** §5's "NOT bare `42bdbb1`" is correct for the F1-R gate (whose
mutant is the round-1 behaviour) but cannot apply here. The honest red baseline is round-1
with **only** the terminal-summary guard removed — isolating the guard rather than dragging
in F5's fail-closed sizing, which bare `42bdbb1` also lacks and which would red the gate for
an unrelated reason.

Red run, round-1 coordinator with only the `try/except` around
`out["bound_in_force"] = self.staging_deferred_bound()` reverted:

```
AssertionError: Traceback (most recent call last):
  File ".../tests/test_s172_staging_backpressure.py", line 2292, in run
    holder["result"] = run_trial_miner(
  File ".../miner/range_miner_coordinator.py", line 5159, in serve_trial
    bp_metrics = self.log_staging_backpressure_summary(run_id)
  File ".../miner/range_miner_coordinator.py", line 3387, in staging_backpressure_metrics
    out["bound_in_force"] = self.staging_deferred_bound()
  ...
ValueError: effective_cap must be positive, got 0
```

The honest `coordinator_staging_sizing` termination is destroyed by a `ValueError` raised
out of the **reporting** layer — the F3 disease relocated, exactly as Alpha's guard comment
predicted. On the delivered tree the gate is green: reporting degrades, the terminal truth
survives.

---

## 4. Results — VM101, after the last change

| suite | result |
|---|---|
| `tests/test_s172_staging_backpressure.py` | **31/31**, three consecutive full runs (final at 2026-08-07T03:04:12Z) |
| `tests/test_s172_staging_partb.py` | **24/24** |
| `tests/test_s172_phase4_coordinator.py` (working tree) | **62/63** — Gate 22 only |
| `tests/test_s172_phase4_coordinator.py` (isolated production diff) | **63/63** |

**Gate count — 31, not the 30 the instructions expect.** Round-1 was 28. This round adds
three `_check` registrations, not two: G-RESUME-HANDOFF, its **required** mutant
(G-MUT-RESUME-HANDOFF), and G-SUMMARY-NO-MASK. The mutant is registered as its own check
rather than folded into the handoff gate, matching how every other mutation gate in this
suite is registered (`G-MUT-PAUSE`, `G-MUT-LEASE`, `G-MUT-RESUME-CREDIT`,
`G-MUT-LEASE-HANDOFF`). 28 + 3 = 31. No gate was dropped.

**Phase-4 Gate 22 — the documented working-tree condition, isolated by the same method as
round 1.** Its red is `unexpected changed .py files: {'tests/test_s172_staging_backpressure.py'}`.
Gate 22 builds `changed_py` from `git status --porcelain` against an allowlist containing
`miner/range_miner_coordinator.py` but not this suite. Isolation evidence: a worktree at
`42bdbb1` with **only** `miner/range_miner_coordinator.py` replaced by this round's version
(suite file clean) → **63/63**. So the red is caused solely by the uncommitted test file and
clears at commit. Gate 22 lives in an out-of-scope file and was **not touched and not
widened**, consistent with the standing disposition for this condition.

---

## 5. Scope proof — F2–F5 and matrix-diff unchanged

### 5.1 The suite, AST-normalised against the round-1 snapshot

Every one of the frozen gates is **AST-identical** (`ast.unparse`, so comments, blank lines
and line numbers cannot mask or fake a change):

```
gate_matrix_diff_six_callers_unchanged      ok      gate_lease_handoff_grace                 ok
gate_matrix_diff_behavioural                ok      gate_lease_handoff_mutant                ok
_on_staging_failed_call_sites               ok      gate_timeout_snapshot_attributes_...     ok
_method_source                              ok      gate_unbound_result_is_never_paused      ok
gate_bound_derivation_failure_fails_closed  ok      _bound_derivation_failure_arm            ok
gate_invariant_reason_names_which_bound_tripped                                              ok
VERDICT: ALL FROZEN GATES UNCHANGED
```

`gate_resume_credit_one_wake_per_release` (G-RESUME-CREDIT-a) and `gate_resume_credit_mutants`
are also unchanged.

**Touched suite functions — added (6):** `_spool_result`, `_Round1ClearQueue`,
`_handoff_bench`, `gate_resume_handoff_survives_until_disposition`,
`gate_resume_handoff_mutant`, `gate_summary_never_masks_the_sizing_terminal`.
**Changed (6):** `_Bench`, `main`, `gate_resume_credit_real_readers_fifo`, and — see §5.3 —
`gate3_paused_peer_stalls_while_second_connection_flows`,
`gate5_each_substripe_staged_exactly_once`,
`gate6_no_duplicate_rows_no_stale_acceptance`. **Removed: 0.**

### 5.2 The coordinator, AST-normalised against the round-1 snapshot

**Added (5):** `dispatch_inbound_result`, `clear_any_resume_credit`, `holds_resume_credit`,
`resume_credit_state`, `_await_resume_credit_clear`.
**Changed (7):** `__init__`, `_grant_resume_credit`, `_try_self_resume`,
`_release_resume_credit`, `_conn_reader_loop`, `serve_trial`,
`staging_backpressure_metrics`. **Removed: 0.**

Out-of-scope methods verified **AST-identical**: `_handle_stripe_failure_locked`,
`handle_stripe_failure`, `_on_staging_failed`, **`_serve_dispatch`**, `enqueue_staging`,
`_defer_locked`, `_submit_with_slot`, `_pump_deferred`, `staging_can_accept`,
`register_paused_connection`, `deregister_paused_connection`, `process_lease_expiry`,
`_drop_conn`, `staging_capacity_timeout_expired`, `clear_all_capacity_resume_grace`.

G-MATRIX-DIFF-a/b pass unchanged: seven `_on_staging_failed` call sites at `7c4f11b`, six at
`4b1aad6`, the **same** six live. No worker code, no seed caps, no stripe geometry, no
`gate_s172_prod_shape.py`.

### 5.3 Collateral of Beta §4 tail — reported, not worked around

Item 6 ("no second result while the reservation is out") changes an observable property of
the reader: a connection whose result is delivered but **not yet dispatched** now holds its
next result. Four gates drove the bench by draining everything and only then dispatching —
a sequence that, by construction, no longer sees a connection's second frame, because the
serve loop is now the thing that releases it. They were re-sequenced to run the serve loop
between frames. **Every assertion in them is unchanged in content**; only the driving
sequence moved:

| gate | change | assertions |
|---|---|---|
| G3 (wire order across a pause) | drain/dispose interleaved via `_Bench.dispose` (disposition clear alone — dispatching a `StripeComplete` here would drag reconciliation in, which is not this gate's subject) | identical: `[(result,0),(result,1),(stripe_complete,None)]` |
| G5 (staged exactly once) | `_Bench.pump` — drain and dispatch interleaved through the production seam | identical: 4 shard rows, sub_index `[0,1,2,3]` |
| G6 (no duplicate rows) | `_Bench.pump` | identical: both frames reach the serve loop, 1 shard row, no second dedup layer |
| G-RESUME-CREDIT-b | see §2.3 | strengthened, not weakened |

`_Bench.dispatch` now routes through `dispatch_inbound_result`, so **production code**
disposes of the reservation in every gate rather than the test thread. `_Bench.__init__`
gained an optional `inbound` parameter used only by the F1-R mutant.

These are harness-sequencing changes to gates that are **not** in the F2–F5 / matrix-diff
frozen set, and all four are green. Flagged because Beta asked for disagreements and
collateral to be reported rather than silently absorbed.

---

## 6. Files changed — exactly two

```
 miner/range_miner_coordinator.py        | +211 / -29    (this round)
 tests/test_s172_staging_backpressure.py | +438 / -19    (this round)
```

No other file was edited. `docs/CLAUDE_CODE_REPORT_S172_BP_AMENDMENT_R2.md` (this report) is
the round's deliverable document. **Not committed, not pushed; step 6 remains held.**

---

## 7. Open items for Beta

1. **The fifth disposition site** (`conn_dropped`, §1.2) — an instance of (iv) occurring one
   line before the eof arm. Ratification requested.
2. **G-SUMMARY-NO-MASK's red baseline** (§3.1) — round-1-minus-the-guard rather than
   round-1, for the reason given. Confirmation requested that this is the intended reading.
3. **The `dispatch_inbound_result` seam** (§1.3) — added so a gate can drive the real serve
   path; `_serve_dispatch` itself is byte-unchanged. Ratification requested.
4. **Gate count 31 vs the expected 30** (§4) — the required mutant is its own check.
5. Carried from round 1 and still outstanding: the §1.4 lease exemption for
   coordinator-initiated pauses remains flagged for Beta ratification.
