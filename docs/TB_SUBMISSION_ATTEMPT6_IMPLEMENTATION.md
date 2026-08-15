# TEAM ALPHA → TEAM BETA — ATTEMPT-6 REMEDIATION: IMPLEMENTATION R2 FOR CERTIFICATION

**Per your RRR of 2026-08-14 — RETURN FOR ONE NARROW R2.** The correction is applied, the three
record-only items are done, and the re-audit your closing note asked for was run. **R2 edited
`scripts/gate12_sentinel_gate.py` and `tests/test_s172_attempt6_remediation.py` only.**

## R2 — WHAT CHANGED, AND THE CLAIM YOU SAID YOU WOULD CHECK

**THE COORDINATOR AND WORKER ARE UNCHANGED FROM R1, AND THE SCOPE PROOF PROVES IT RATHER THAN
ASSERTING IT.** `miner/range_miner_coordinator.py` `+1,508/−72` and `miner/range_miner_worker.py`
`+190/−0`, digit for digit. The regenerated proof carries a **sha256 of its entire digest body
compared against the preserved R1 reference** (`attempt6_logs/r1_SCOPE_PROOF_reference.txt`):
`2a915632…6467a` on both sides, `IDENTICAL: True`, `R2 SCOPE VERDICT: PASS`, `NO-TOUCH VERDICT:
PASS`. `gate12_launch.sh`, `scripts/launch_fleet_manual.sh` and
`tests/test_s172_staging_backpressure.py` are untouched by R2 as well. `git status --porcelain` is
**the same eleven entries at R2 START and END** (5 modified + 6 untracked) — no file added, none
removed.

**R2-A — an ssh transport failure is UNAVAILABLE, not ERROR. Your finding is accepted; confirmed in
live source.** `SSH_TRANSPORT_FAILURE_STATUS = 255` as a named module constant; the check runs
**before any stdout is read** — if ssh failed, what arrived is not this probe's output — and is
**gated to the remote branch**, since the local `bash -c` gives 255 no transport meaning. The reason
keeps the certified `ssh_exit_<rc>` token so one grep finds this and the GPU probe.
**Why 255 and not any-nonzero, per your warning — and stated as what it is.** ssh returns 255 for
its own failure, and **this gate reserves that value as its remote-transport classification under
the current probe script.** That is a decision about this gate, not a protocol-level claim: **ssh
passes a remote command's status through unchanged, so a remote command can exit 255 and would be
reported here as a transport failure.** The reservation is sound because the script this gate sends
cannot produce 255, and **if that script changes the rule must be revisited** — written down beside
the constant rather than left implicit. Worth stating because a reviewer who knows
`preflight_check.py:512` will ask why this differs from the certified rule — **the GPU probe can
treat any nonzero as UNAVAILABLE because `_build_gpu_probe_script` ends every internal failure
branch in an explicit `exit 0`, designing the ambiguity out. This gate's script has no such
guarantee** (its taken branch merely happens to end in `head -1`, one edit from being false), so it
reserves one value instead of every nonzero one.

**RXP-3/12 and /13, with the seam at `SG._run`** so the **real** `probe_sentinel` body runs against
the exact `CompletedProcess` ssh returns, no fleet and no network. /12: status 255 + real
`No route to host` stderr → UNAVAILABLE, `count is None`, render never count-shaped, `evaluate`
refuses, reason names `ssh_exit_255`, ssh's diagnostic reaches the operator. /13, the neighbouring
control so the two statuses cannot collapse: status 0 + malformed → ERROR · status 1 + malformed →
ERROR · **status 1 + WELL-FORMED → OK with the count read** — your sentence about a remote `grep`
returning 1, made executable. **Both mutants APPLIED, EXECUTED, DETECTED**: /12's reproduces
`ERROR: unparseable_probe_output:[]` exactly, /13's (`!= 0`, the rule you forbade) reports
`UNAVAILABLE ssh_exit_1` for a legitimate remote grep status. *(The first draft of the stub passed a
merged globals copy, which would have let both mutants escape the shim and survive while appearing
detected — the A8-B2 lesson, caught before it shipped.)*

**R2-B — the re-audit, and it found two more.** **RXP-1/2** declares *"a BOUND socket exiting via
E2-E5 or E7"* and drives **E5 alone** — your shape exactly. Declaration narrowed to what runs, **no
assertion changed**; whether to widen the coverage is **put to you rather than taken**, because arm
2 tests the *transport* (which carries the reason rather than switching on it) while RXP-1/1
separately drives all eight classes, and widening is a restructure of certified machinery, not a
loop — the injections tear their benches down before `_drop_conn` can be called with live maps. **One
line from you and it is done.** And a third finding **in the harness**: `_mutant_red` credits **any**
exception as detection, so a mutant built inside its lambda let *MUTANT NOT APPLIED* read as *MUTANT
DETECTED* — the same class one level up, inside the machinery whose purpose is to prove the arms are
not vacuous. All four Alpha-authored sites now build the mutant as a statement.

**R2-C — record-only, all three done.** Mutant count corrected **three → SEVEN** in both the report's
§6 table and this cover, each named with applied/executed/detected evidence. **D2 design text
corrected under your authorization** — §8.6.3 condition (2), the P-REG proof, the
Beta-constraint-discharged paragraph, the §12 summary and the §11.6 arm-2 row all move to *"first
decoded application frame on this connection"*, each marked a precision correction, with an explicit
notice that the architecture is unchanged and that first-frame priority is **consumable exactly once
per connection**; the proof's *"at most once per connection"* step is now discharged **by condition
(2) itself**. Final tally **78**.

**Declined by you, and NOT added:** the runtime `conjunctive > sentinel_lines_any_nonce` inequality.
It is not in the code.

**Gates: 78/78 green** at R2 final state (`attempt6_logs/r2_gates.log`), written after that run.
**D6 remains deliberately unexercised** — R2 exercises the remote branch's *classification* at the
`_run` seam and contacts no host; the 25 ssh dispatches and the real remote probe are still for
prelaunch. **Attempt-5's initiating reader cause remains UNRESOLVED.**

---

## R1 — WHAT CHANGED AT THE PREVIOUS ROUND (retained; R1-A, R1-B and D2 are CLOSED)

**Per your RRR of 2026-08-13 — RETURN FOR NARROW IMPLEMENTATION R1.** Both behavioural corrections
are applied, the numeric reconciliation is done, and the arm audit your closing note asked for was
run. **The R3 architecture was NOT reopened and nothing on your forbidden list was touched.**

## R1 — WHAT CHANGED SINCE YOUR REVIEW

**R1-A — the sentinel gate now accepts on a SAME-RECORD conjunction. Your finding is accepted in
full.** `probe_sentinel()` runs one pipeline, `grep 'SESSION_SENTINEL' <log> | grep -c '<nonce>'`;
`evaluate()` sees that number and nothing else. The sentinel/barrier architecture and the launch
order are untouched — the correction is to the probe predicate only, and `gate12_launch.sh` /
`launch_fleet_manual.sh` are byte-unchanged by it. A diagnostic-only `sentinel_lines_any_nonce` is
retained so an operator can tell your two refusals apart, and **RXP-3/11 asserts by AST that it can
never reach acceptance** — including that the acceptance number is bound to `out[0]`, because a swap
would restore the defect while every string still looked right. **Your two new arms are RXP-3/9 and
RXP-3/10**, and both build their logs by running the real worker emitters (`_emit_session_event`,
and the real `await_session_release` for the `SESSION_RELEASE_WAIT` record that carries the nonce),
never by writing strings. **The pre-R1-A two-independent-counts predicate is reconstructed from live
source by mutation and applied to both arms: APPLIED, EXECUTED, DETECTED.** **RXP-3 is now eleven
arms; the 71/71 tally did not certify it and this one does.** *(The optional runtime
`conjunctive > sentinel_lines_any_nonce` inequality R1 offered here was **DECLINED by Beta at R2**
and is not in the code.)*

**R1-B — P-1 is FIRST-FRAME REGISTER PRIORITY. D2 is resolved by your ruling, not by Alpha.**
`envelopes_delivered` is **deleted from the reader**. `first_frame` is snapshot-then-cleared
**unconditionally at the decode**, before any branch can act on the frame, so the admission route is
reachable at most once per connection and only on frame 1. **Not keyed on `worker_by_sock`**, per
your instruction. **FAIR-6/13** drives `REGISTER → REGISTER` back-to-back with no intervening
result and decides on `admission.put_order`, which records every put — the count is measured, not
inferred from a queue depth sampled at one instant; **its mutant (first-frame term removed) is
APPLIED, EXECUTED, DETECTED.** **FAIR-6/14 is the structural assertion you asked for**: armed once
outside the loop · cleared once as a **direct statement of the loop body** (a clear nested in a
branch is a clear some path skips) · never re-armed · the admission test mentions neither the
delivery counter nor `worker_by_sock` · and `envelopes_delivered` is absent from the reader
entirely. **Your point that FAIR-6/7 cannot detect D2 is confirmed against its source** — its
intervening result necessarily increments the counter first; arm 13 is arm 7 with that result
deleted.

**R1-C — the audit, and it found a fourth instance of the class.** **FAIR-3/4** asserted
`loop_now_age_max >= 0.0` and `<= bound`; **an instrument frozen at its constructor value satisfies
both**, and `note_loop_now_age` returns early when `_last_top` is unset, so an unwired instrument
reports a permanent `0.0` and the arm went green measuring nothing. It now asserts observation
first — strictly positive age **and** the wall label the instrument stamps only when it updates —
with a reachability control proving the frozen state is real. Three more were tightened:
**RXP-1/10** (`"Error" in str(exc_class)` → the name must **resolve** to an `OSError` subclass),
**FAIR-6/1 and /2** (key presence → the identity must be bound to that connection's own framed
socket, and on reconnect to the new one), and a **no-op `src.replace(" ", " ")`** in FAIR-1/2 arm 5
removed. Six arms were checked and found sound, and one — arm 5's re-implementation of the
production timeout expression — is reported as the shape but left alone, because the paired AST
assertion over live source closes it.

**R1-D — the numeric reconciliation.** *Measured on the artifacts as they stood when your brief
arrived, before any R1 edit, and it does not match the description:* `fifteen`, `1531`, `372` and
`4183` **do not occur** in the report, the cover or the changelog on this host, and only one copy of
the report exists here. All three already read **sixteen · +1,472/−72 · 4,220 · 356**, and the
sixteen is confirmed against `git diff --numstat` (`+16/−16`). You were reading an earlier revision
of the uploaded copy. *(Those strings appear now only where this reconciliation quotes them.)* **The reconciliation genuinely due is that those figures are
now stale because of R1**, and every one has been re-measured at final state and updated in all
three documents.

**Nothing forbidden touched:** D6 · F1 lease-origin · expiry/retry · `worker_admission_timeout`
(NOT widened, enforcement block still AST-subtree-IDENTICAL) · `D`/`D_adm`/`A_max`/`S` unchanged ·
queue and staging bounds unchanged · emergency-terminal policy unchanged · **sentinel/barrier launch
order NOT redesigned** · window-anchor work not merged. **Attempt-5's initiating reader cause remains
UNRESOLVED and is claimed nowhere.**

---

**Base:** HEAD `2b0d2dc5268946d6b1a44e268573e816b7cdcb83`, branch `main`, unmoved across S180, R1 and
R2. **Nothing committed, nothing pushed, attempt 6 NOT launched. Port 5700 never bound.** The
pre-existing untracked `docs/AUDIT_STEP1_OFFSET_REACH.md` is reported, not assumed away. Full
report: `~/dashboard_work/ATTEMPT6_IMPLEMENTATION.md` §R2 then §R1; scope proof (regenerated at R2,
with the R1 comparison inside it): `~/dashboard_work/ATTEMPT6_SCOPE_PROOF.txt`; R1 reference proof:
`~/dashboard_work/attempt6_logs/r1_SCOPE_PROOF_reference.txt`; run logs
`~/dashboard_work/attempt6_logs/r2_*` (and `r1_*`, `CERT_*` retained for the differentials);
changelog: `docs/SESSION_CHANGELOG_20260814_S181.md` §8 (S180's remains the historical record of the
original implementation).

**Files at R2 final state.** `miner/range_miner_coordinator.py` (+1,508/−72, **unchanged by R2**) ·
`miner/range_miner_worker.py` (+190/−0, **unchanged by R1 and R2**) ·
`scripts/gate12_sentinel_gate.py` (NEW, **459**) · `tests/test_s172_attempt6_remediation.py` (NEW,
**4,906**) · `gate12_launch.sh`, `scripts/launch_fleet_manual.sh` (launch order, **unchanged by R1
and R2**) · `tests/test_s172_staging_backpressure.py` (**mechanical, +16/−16, disclosed at §D1,
unchanged by R1 and R2**).

**Gates: 78/78 green** — all ten §11 gates plus four RED arms — run at final state after the last R2
change, with this cover written after that run. **71 → 76 → 78, all new arms and never a re-count:**
R1 added five (RXP-3/9, /10, /11, FAIR-6/13, /14), R2 adds two (RXP-3/12, /13). The audit
corrections at both rounds tightened or narrowed existing arms and added no rows. **Seven mutants,
each APPLIED, EXECUTED and DETECTED.**

**Scope, cumulative.** R1 moved one digest, `_conn_reader_loop`
`f0b23825e798cf20 → f87f8433961d823b`. **R2 moved none** — see the sha256 body comparison above.
Every count in §5 below is unchanged (219/10/19/0 and 68/2/3/0), and every no-touch surface still
carries its S180 digest.

**No ruling is requested on the architecture. Six disclosures follow; §D2 is RULED and closed, and
its design text is now corrected under your R2 authorization.**

---

## 1. THE FOUR BINDING IMPLEMENTATION DETAILS, EACH SHOWN SATISFIED

**1 — Connection correlation.** `next_connection_id(run_id)` returns a run-scoped token from a
lock-guarded monotonic counter, assigned **when the connection is accepted**, stored on `ConnState`
and carried into `READER_EXIT`, `CONNECTION_CLOSE_INTENT` and `WORKER_DISCONNECTED`.
**`rawsock.fileno()` and `id(rawsock)` appear nowhere in the provenance path.** RXP-1 arm 1 asserts
every record carries a token; arms 2, 11, 12 and 13 assert the three surfaces **correlate on it**.

**2 — `A_max` fails closed unless an integer `>= 1`.** `None`, `0`, negatives, floats, `bool` and
strings are refused at entry to `serve_trial`, naming the term and the consequence. `D`, `D_adm`
and `S` are validated positive-finite in the same place and the same shape as
`worker_admission_timeout`. **Gated:** FAIR-1/2 arm 8 drives 21 bad values across the four terms and
asserts each is refused **before the dataset digest resolves**.

**3 — Saturation accounting.** `S` is charged from **exactly two** sites — the ordinary
`inbound.put` and the reasoned EOF — each measuring `perf_counter()` elapsed, **not** the nominal
quantum. One creation site (`ConnState.__init__`, `0.0`), one assignment site
(`charge_inbound_saturation`); **a successful put never resets it.** Gated by RXP-2 arm 6 (AST +
saturate → recover → saturate, terminal at cumulative `S` not `2S`), arm 9(c) (both call sites on a
`_queue.Full` handler of an `inbound.put`) and FAIR-7 arm 6.

**4 — Admission fairness wording.** The qualification is in the code at the mechanism and in the
report: **`D_adm + A_max` prevents cumulative admission-queue monopolization; it does NOT preempt a
single synchronous `m_i`.** Nothing anywhere describes admission latency as absolutely bounded — the
string does not occur — and FAIR-6 arm 3 **forbids any arm asserting an absolute 180 s ceiling**,
because no arm can test it.

## 2. YOUR TWO §11.1 RECORD-ONLY CORRECTIONS, APPLIED

**Four record surfaces exist** — `READER_EXIT`, `CONNECTION_CLOSE_INTENT`, `WORKER_DISCONNECTED`,
the emergency terminal event — **an incident produces a chronology-dependent subset**, and **no
surface is synthesized to make every incident produce the same set**: RXP-1 arm 3 asserts the
ABSENCE of a disconnect record on the shutdown paths rather than manufacturing one.

**RXP-1 arm 7 as corrected:** every coordinator-originated close emits `CONNECTION_CLOSE_INTENT`;
where a `READER_EXIT` also exists the two remain orthogonal and neither is a value of the other;
and **a reader observation is not required to exist** — arm 11 drives exactly that case on an
unbound socket.

## 3. WHAT THE GATES PROVE, BEYOND THE TALLY

**RXP-1 arm 12, the race gate.** Drives §8.3.2c's ordering — reader fails **independently** and
emits `READER_EXIT` with a null intent, and only **then** does the serve loop reach its read-deadline
decision — and asserts the intent survives on its **own** record, that the two correlate, and that
**the reader's null is left as emitted**: it is evidence of ordering, not a lost fact.

**RXP-2 arm 3 runs the REAL `serve_trial`.** The emergency queue is captured from the production
wiring (the object the loop hands its readers); the loop's own consumption fail-closes the trial —
`state=aborted`, `terminal_class=inbound_saturation_timeout`, `emergency_events_acted_on=1`, and
**no `WORKER_DISCONNECTED` anywhere**. No worker is shed.

**RXP-3 proves ordering and then falsifies it.** Worker `main()` executed from live source with the
GPU surface shimmed **in the worker module's own globals** (your A8-B2 rule), against a stub listener
recording every accept and REGISTER instant: with the release absent, **zero connections**; after the
release, every REGISTER later than the write. Arm 7 re-executes an **AST-mutated `main` with
`await_session_release` removed** and requires arm 6's assertion to FAIL. It did. **And, at R1, it
proves DELIVERY as well as ordering:** arms 9 and 10 present the two logs where the sentinel fact and
the nonce fact are true SEPARATELY, and the pre-R1-A predicate — reconstructed from live source by
mutation — passes both and is detected by both.

**FAIR-1/2, measured on the real loop** (`D`, `D_adm` read from live config, so raising them raises
the asserted bound visibly): count pressure `T_cp = 0.616 s` against a recurrence of `0.936 s` with
`drain_deadline_hits = 9` (the vacuity condition); slow-message pressure `T_cp = 1.615 s` against
`1.917 s`, residual attributed to exactly one message which **still completed**.
**RED, on your pinned commit:** the pinned `serve_trial` and pinned `_conn_reader_loop` are exec'd
together and measure **`T_cp = 4.17 s`** against the required `24 × 0.05 = 1.2 s`.

**FAIR-6 drives the R3.2 worst case positively:** a 24-connection reconnect storm with **every
`m_i` scripted longer than `D_adm`** drains at **exactly one disposition per turn, never zero**, and
an AST arm pins the `_adm_done > 0` guard whose absence would yield zero per turn.

## 4. RED-ARM DISCIPLINE, APPLIED UNASKED

Full 40-character SHA. `_assert_pinned_carries_the_defects()` is the **single** implementation every
RED arm calls — the check cannot be forgotten at one site — and verifies eight properties before any
credit, including the self-protection arm that must REFUSE repaired source. A drifted anchor
terminates **UNAVAILABLE**, which never accepts.

**The probes run over comment-stripped executable source, and that is load-bearing:** the repaired
file quotes the old surfaces in its own docstrings, so a text probe would find the defect in the
FIXED file and credit an anchor that had drifted forward. The same discipline is applied to the
live-source arm asserting `eof_reap` is absent.

**SEVEN mutants, each proven APPLIED, EXECUTED and DETECTED** *(corrected at R2 — this paragraph
still said "three" and listed only the originals while the cover's own summary said five)*: the
unlabelled-exit mutant (reports `READER_EXIT_UNCLASSIFIED`); the early-REGISTER mutant (arm 6 fails
as required); the BP battery's round-1 clear-at-ingress mutant (still executes, still reds);
**[R1-A] the RXP-3 two-independent-counts mutant** (acceptance passes the split fact — detected by
arms 9 and 10); **[R1-B] the FAIR-6 first-frame-guard-removal mutant** (`admission.put_order`
records two puts on one connection); **[R2] the RXP-3/12 returncode-never-examined mutant**
(reproduces `ERROR: unparseable_probe_output:[]`, the exact pre-R2 misclassification); and **[R2] the
RXP-3/13 over-broad-rule mutant** (`!= 0`, reporting `UNAVAILABLE ssh_exit_1` for a legitimate remote
`grep` status — the rule you forbade).

**[R2, found by the re-audit] Every mutant is now constructed OUTSIDE `_mutant_red`.** That helper
credits **any** exception as detection, so building a mutant inside its lambda let *MUTANT NOT
APPLIED* read as *MUTANT DETECTED* — the same failure class, one level up, inside the machinery that
exists to prove the arms are not vacuous. The four Alpha-authored sites are corrected; the two
S180-era sites already built theirs outside.

## 5. SCOPE PROOF — per-definition AST digests vs `2b0d2dc`

Coordinator **229 → 248 definitions: 10 CHANGED, 19 ADDED, 0 REMOVED.** Worker **70 → 73: 2
CHANGED, 3 ADDED, 0 REMOVED.** The twelve changed definitions are named in the report.

**Every no-touch surface is digest-IDENTICAL:** `claim_stripe` · `schedule_pending_stripes` ·
`renew_lease` · `_renew_active_lease` · `process_lease_expiry` · `_handle_stripe_failure_locked` ·
`_execution_set_expected_workers` · `_serve_register` · `_serve_dispatch` ·
`dispatch_inbound_result` · `assign_stripes` · `enqueue_staging` · `_defer_locked` ·
`_run_staging_job` · `_await_exact_credit_clear` · the credit machinery · the pause registry ·
`register_worker` · `fail_trial` · `commit_trial` · `abort_trial` ·
`validate_threshold_provenance`. **The §4.3 bounded-admission block is compared as an AST SUBTREE
and is identical** — `worker_admission_timeout` NOT widened, its enforcement NOT touched. The 256
guard is **retained as the secondary ceiling**, not lowered; `maxsize=1024` and every staging
ceiling are unchanged; D6, the F1 lease-origin seam, the expiry matrix, one-active-claim, sampler
semantics, seed geometry, S145, publication semantics and the clean-tree gate are untouched; the
window-anchor / generator-phase work is **not** in this implementation. **FAIR-3 arm 2 gates all of
it**, so it is a test that runs rather than a claim in a cover.

## 6. REGRESSION BATTERY AT R2 FINAL STATE

Re-run in full, sequentially, after the last R2 change: phase-4 **62/63** · F1 lease-origin
**18/18** · F1/F2 **16/16** · Defect A **29/29** · admission-liveness **16/16** · exec-set
**34/34** · elapsed-roundtrip **6/6** · back-pressure **50/50** · Part B **24/24** ·
`admission_binding` **11/20 PRE-EXISTING** · phase-3 worker **17/17** · phase-2 protocol **6/6** ·
phase-1 scaffolding **6/6**.

**Every one is verdict-set IDENTICAL to R1 — all thirteen diffs EMPTY — and to the S180
certification run, measured rather than asserted.** Per-arm PASS/FAIL sets were extracted from every
log and diffed `r2_*` vs `r1_*` and `r1_*` vs `CERT_*`. The R2-vs-R1 result is the expected one,
since R2 changed no production code. Against S180, eleven are byte-identical and
`admission_binding` / `admission_liveness` differ **only** in run-scoped values inside arm messages
(a `resolved_utc`, a `run_id`, an elapsed figure), identical once excluded. **An identical count is
not an identical set — which is the failure class this whole cycle is about — so the sets were
compared, not the totals.**

`admission_binding` was proven not chargeable at S180 by the **differential-worktree** method
(worktree at `213bfff` vs the patched tree, empty diff of sorted arm lists, 11/20 on both). **That
worktree differential was not re-run at R1 and is not re-claimed**; what is measured is that R1
moved nothing relative to the run that carried it.

No suite's assertions were altered except §D1 and the four R1-C tightenings, **all of which
strengthen the arm**: four arms now reject inputs they previously accepted, none accepts an input it
previously rejected. **R2 altered no assertion at all** — its two arms are additive, and its three
audit corrections change a docstring (RXP-3/3, RXP-1/2) or move a mutant construction out of a
lambda. No threshold, expectation or gate semantic was weakened anywhere.

---

## DISCLOSURES

**D1 — the certified BP battery could not run at all against the frozen contract, and needed a
mechanical widening.** The design freezes the eof tuple at five fields and the drain unpacks all
five uniformly, so ordinary `msg` tuples are five-wide too; the battery unpacks that tuple in
**sixteen** places with four names, and Python cannot unpack a 5-tuple into four names. **Alpha did
not adapt the architecture.** The sixteen patterns were widened and the change is **proven to be
nothing else**: mapping them back to their four-wide originals reproduces the committed file
**byte-for-byte**. Same class as the G3/G5/G6 bench resequencing you ratified on a programmatic
identity proof — flagged here rather than filed under "the battery ran unmodified", because FAIR-4's
criterion uses that word.

**D2 — RULED AND CLOSED (your R1-B).** You ruled that the architectural invariant controls over the
defective proxy wording: **P-1 is FIRST-FRAME REGISTER PRIORITY**, the admission route is usable at
most once per connection and only on its first decoded application frame. Implemented as ruled —
`envelopes_delivered` deleted, reader-local `first_frame` snapshot-then-cleared unconditionally at
the decode, not keyed on `worker_by_sock` — with FAIR-6/13 (behavioural, plus a detected mutant) and
FAIR-6/14 (structural, consumable-once). **Nothing here is left as Alpha's reading.**

**The design text is now CORRECTED, under your R2 authorization.** At R1 Alpha left
`ATTEMPT6_REMEDIATION_DESIGN.md` alone because R3 was certified and that brief did not reopen it;
R2 authorizes the record-only correction and it is done. **Five** statements moved, not the two the
brief named — §8.6.3 condition (2), the P-REG proof, the Beta-constraint-discharged paragraph, the
§12 summary line and the §11.6 arm-2 row — because correcting only some would have left the document
contradicting itself on one fact. Each is marked a **precision correction implementing the D2
ruling**, §8.6.3 carries an explicit notice that **the architecture is unchanged** and that
first-frame priority is **consumable exactly once per connection**, and the proof's *"at most once
per connection"* step is now discharged **by condition (2) itself** rather than inferred from a
counter that did not guarantee it.

**D3 — Gate 22 reads 62/63, and the allowlist was NOT widened.** The two NEW untracked `.py`
deliverables trip the porcelain scope detector. Expected, not a regression, self-clears on a clean
committed tree. Fifth occurrence; handling unchanged.

**D4 — the saturation terminal calls `fail_trial` WITHOUT the loop's shared `now`.** The six
shared-clock consumers of that loop are an audited set the F1 suite computes from source, so a
seventh would red a certified gate; and a terminal raised by a reader thread is better timestamped
by `fail_trial`'s own clock than by an iteration clock older than the event. Judgement call, one
line, reversible.

**D5 — saturation charging logs transition-only.** The FIRST charge on a connection is emitted at
WARNING (`inbound_saturation_begin`), subsequent charges at DEBUG, with both counters on the
`[S172-BP]` summary regardless. At a 0.25 s quantum against a 180 s budget, per-charge WARNINGs
would be ~720 lines per connection — the high-rate noise the §15 transition-only bar excludes.
Alpha's addition to the design's counters, stated so it is not discovered in a log.

**D6 — nothing ran against the fleet, by instruction.** RXP-3 proves the sentinel/barrier mechanism
end to end at loopback scale and pins the production call path and ordering by AST. **Unexercised:**
25 ssh dispatches carrying the new arguments, and the probe's remote branch (which differs from the
tested local branch only by an ssh prefix). First thing a prelaunch dry run should exercise.

## STANDING LIMITS, RESTATED SO NOTHING READS AS MORE THAN IT IS

- **The initiating cause of attempt 5's two lost reader sessions is UNRESOLVED**, and is stated as
  known nowhere in the code, the gates or the report.
- **`M_i`, `K_i` and `m_i` remain unbounded by this repair.** The certified claims are structural:
  the drain contributes `<= D + one in-flight message`; REGISTER delay is independent of data-queue
  depth; an admission turn contributes `<= D_adm + at most one overrun registration`. Production
  latency stays **observable rather than mathematically capped** — which is why the observability
  additions (composite `K_i`, `slow_control`, `slow_msg`, drain-stop counters,
  `admission_queue_high_water`) are part of the deliverable and not decoration.
- **The sentinel proves the channel at T0, not for the next four hours.** It converts "unobserved,
  cause unknown" into "observed at T0, so any later silence is a CHANGE".

---

**Requesting:** certification of this implementation. On your ruling and Michael's commit
direction the sequence is unchanged — Michael commits and dual-pushes → clean tree → prelaunch
battery → **only then** attempt 6. Attempt 6 must still satisfy your §21 completion authority
**in one run**: truthful GPU preflight, 25-worker frozen admission, both saturation verdicts, four
stages complete, D3.5 publication, S145 coverage, cursor at 2,147,483,648. **No credit composes from
any earlier attempt.**
