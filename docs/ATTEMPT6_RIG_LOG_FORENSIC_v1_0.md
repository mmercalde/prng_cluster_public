# ATTEMPT-6 RIG-LOG FORENSIC — bounded-unresolved

**For:** Team Beta
**From:** Team Alpha (Claude Code, VM101)
**Date:** 2026-08-15
**Run:** `distributed_config_t1_db0393b0`, launch stamp `20260815_131428`, HEAD `5ad170a`
**Frozen bundle:** `/home/michael/attempt6_riglogs_20260815`
**Status:** **BOUNDED-UNRESOLVED.** No remedy proposed.

---

## 0. What this document is, and what it is not

It answers one question: **can the rig-side worker logs distinguish "the three
long-held stripes were still computing" from "their results were sent and not
accepted in time"?**

**They cannot.** That is established below as a structural property of the
instrumentation, not as a failure to look hard enough.

This document therefore does three things: it records what the rig logs **do**
establish, it states precisely **why** they cannot reach the question, and it
specifies **what instrumentation would be required** to separate the two
hypotheses in a future run. The third is the useful output.

**No remedy is proposed.** In particular, raising `compute_lease_timeout` is the
mask-not-fix move Beta ruled against in §2.26 and is not offered here.

---

## 1. Bundle identity and integrity

| | |
|---|---|
| path | `/home/michael/attempt6_riglogs_20260815` |
| integrity | **63/63 files OK** against `SHA256SUMS.txt` (`sha256sum -c`, this session) |
| rig worker logs | 24 — `192.168.3.{122,156,164}/gpu{0..7}.log` |
| local worker log | 1 — `zeus_local/zeus-ubuntu-vm_gpu0.log` |
| ledger | `miner_ledger_attempt6.db` (22.6 MB), queried `mode=ro` |
| coordinator log | `gate12_20260815_131428.log` (165 KB) |

**⚠ Four files in `zeus_local/` are STALE and are not attempt-6 evidence:**
`vm101_gpu0.log` and `launch_192.168.3.{122,156,164}.log` all carry mtime
**2026-08-04 18:09**. `vm101_gpu0.log` contains a `BrokenPipeError` traceback from
that earlier run. Reading it as attempt-6 evidence would manufacture a transport
finding that did not occur in this run. Flagged because the bundle otherwise
invites it — the attempt-6 local worker is `zeus-ubuntu-vm_gpu0.log`
(mtime 13:32).

**⚠ The 24 rig-log mtimes are all 13:38 — collection time, not event time.** They
carry no information about when any record was written. Only
`zeus-ubuntu-vm_gpu0.log` has a true last-write mtime.

---

## 2. Two corrections to Alpha's prior framing

**Correction 1 — the one Michael issued, confirmed here.** Alpha's launch report
named the rig-side worker logs as "the surface that separates" stall from
backlog. **That was wrong**, and §4 proves it.

**Correction 2 — Alpha's own, and it changes the reading materially.** The launch
report derived stripe *claim* times as `lease_expires_at − 300`. That derivation
is valid only under attempt-1's bulk-claim model. Under the certified F1/F2
scheduler:

- initial stamp: `claim_now + compute_lease_timeout` — `range_miner_coordinator.py:3634`
- renewal: `now + compute_lease_timeout` where `now = time.time()` **at the moment
  the coordinator processes an accepted frame** — `:9519-9521` → `renew_lease` `:2051-2064`

So `lease_expires_at − 300` is **the last moment the coordinator did anything for
that stripe** — `max(claim, last accepted progress)` — **not** the claim time. The
figures in the launch report should be read under that meaning. The correction
strengthens rather than weakens the finding: it means these three stripes had
**300 s with no accepted progress of any kind**.

---

## 3. What the rig logs DO establish — CONFIRMED

### 3.1 The §15 session-event emitter works, on every rig, for the first time

All 25 workers emitted a complete, well-formed session-event sequence through
`_emit_session_event` (`miner/range_miner_worker.py:1388`):

```
SESSION_SENTINEL > SESSION_RELEASE_WAIT > SESSION_RELEASED > SESSION_END
```

**Exactly one distinct event sequence across all 25 logs.** Attempts 4 and 5
produced 24 byte-identical 138-byte rig logs carrying no session event at all.
That condition is **absent** in attempt 6 and was additionally proven *before*
dispatch by the sentinel and liveness gates (25/25 each). Whatever silenced the
channel in attempts 4-5, it did not occur here.

### 3.2 Every worker shut down cleanly, with no active assignment at loss

All 25: `classification = explicit_shutdown` · `assignment_active_at_loss = false`
· `exc_class = null` · `reconnect_attempted = false` · `session_generation = 1`.

**Consequences that are real:** no worker crashed; no worker lost its transport;
the Defect-A recovery path was never entered by anyone (`session_generation`
stayed 1 fleet-wide, `reconnect_attempted` false fleet-wide); no worker was
evicted or refused under §13. The fleet ended by coordinator-initiated shutdown,
in order, with no exception anywhere.

**⚠ But see §4.2 — `assignment_active_at_loss = false` is VACUOUS on this path
and must not be cited as evidence that a worker was not computing.**

### 3.3 The release barrier behaved exactly as designed

`waited_s` forms a clean arithmetic ladder from **88.539 s** (`rrig6600:gpu0`,
launched first) to **19.508 s** (`rrig6600c:gpu7`, launched last), in **3.00 s**
steps — mean inter-launch step **2.876 s**, total dispatch span **69.031 s**.

All 25 released against one token, `gate12_release_gate12-20260815_131428-57900`
(content = the nonce, 28 bytes), written **13:16:10.600842**. The ladder is the
proof that all 25 unparked on a single release event: each worker's wait is
exactly its own launch offset subtracted from one common instant.

Run nonce is identical across all 25 logs and equals this run's nonce. **No log
in the bundle carries a D6 or attempt-1..5 nonce.**

### 3.4 The only timing bound the rig logs support — and it is the wrong window

The rig logs contain **no timestamps of any kind**. `_emit_session_event` writes
`logger.warning("[MINER-SESSION] %s %s", ...)` with a JSON payload carrying no
time field, and the worker's log format emits no `asctime` prefix — verified
against the raw bytes.

The only wall-clock anchors are `waited_s` (a duration) and the release-token
mtime. From those, sentinel emission is bounded to `13:16:10.6 − waited_s`, i.e.
the fleet warmed and parked across **≈13:14:42 → 13:15:51**.

Every remaining worker record is a kernel compile or `SESSION_END`. The last
compile is `java_lcg_reverse`, which cannot precede stage-2 assignment at
**13:26:48.423**. `SESSION_END` follows the coordinator's shutdown, at
**≈13:32:16.27**.

> **THE BLIND WINDOW.** Between `Compiled kernel: java_lcg_reverse` (≥13:26:48)
> and `SESSION_END` (≈13:32:16) — **≈5 minutes 28 seconds** — the worker side
> records **nothing at all**. The lease expiry at 13:32:13.845 lies inside it.
> **The instrumentation is silent over exactly the interval in question.**

---

## 4. What the rig logs structurally CANNOT establish — CONFIRMED

### 4.1 There are no per-stripe or heartbeat records on the worker side, by design

`_emit_session_event`'s own contract (`:1391-1392`):

> *"Emitted ONLY on session transitions — never per heartbeat, per §15's
> no-high-rate-noise bar."*

This is a **deliberate, Beta-ratified design choice**, not an oversight. `§15`
was specified to make session *lifecycle* observable while explicitly refusing
high-rate noise. It succeeded at what it was scoped to do. The stall-vs-backlog
question simply lies outside that scope.

**Measured consequence — the discriminating power is exactly zero.** Reducing all
25 logs to their structure (dropping identity, nonce, pid, paths and `waited_s`)
yields **one distinct skeleton**. The three workers holding the long-lived
stripes are **byte-indistinguishable in shape** from the 22 that completed
normally:

| worker | stripe held | log shape vs the other 22 |
|---|---|---|
| `rrig6600b:gpu1` | `st1_s29` — the terminal stripe | identical |
| `rrig6600:gpu5` | `st1_s30` | identical |
| `rrig6600:gpu3` | `st1_s31` | identical |

### 4.2 `assignment_active_at_loss = false` is VACUOUS on the explicit_shutdown path

This field looks like a discriminator and is not one. `assignment_active` is
`(self.state == "mining")` (`:1835`, `:1851`). `state` becomes `"mining"` at
`handle_stripe`'s first statement (`:1658`) and returns to `"idle"` at `:1695`
(normal) / `:1720` (`_fail_stripe`).

`_run_session` (`:1809-1830`) is **strictly serial**: `msg = conn.recv_msg()` then
`self._dispatch(msg)`, and `handle_stripe` blocks inside `_dispatch`. **A
`shutdown` frame can therefore only be dequeued while the worker is back at
`recv_msg()` — which is reachable only after `handle_stripe` has returned and
already set `state = "idle"`.**

> **On the `explicit_shutdown` path, `assignment_active_at_loss = false` is
> structurally guaranteed for every worker in every run.** It is a
> VIR-2 vacuous-capable field: it can only ever report `false` here, so its
> reporting `false` is not evidence about anything. It is meaningful **only** on
> the `transport_loss` path, where the loss can surface mid-`_dispatch`.

Alpha flags this because the field's name invites exactly the inference it cannot
support, and because it appears 25 times in this bundle.

### 4.3 What the rig logs consequently cannot answer

For `rrig6600b:gpu1` between 13:27:13.845 and 13:32:13.845, the bundle cannot say:

- whether a GPU kernel was executing, and for how long;
- whether `handle_stripe` was entered for `st1_s29` at all;
- whether any `SubStripeResult` was sent, or how many;
- whether the worker was blocked in `_sendall` on a full socket buffer
  (§2.19: worker `_sendall` has no socket timeout);
- when `handle_stripe` returned.

---

## 5. The two hypotheses, and why this bundle cannot separate them

**H1 — WORKER-SIDE STALL.** The three workers were still executing, or blocked
writing, so no progress frame was produced for ≥300 s. The lease then expired
correctly on a genuinely unproductive assignment.

**H2 — COORDINATOR-SIDE ACCEPTANCE BACKLOG.** The workers finished and streamed
normally, but their frames were not *accepted* within 300 s. Since renewal is
driven by accepted progress (`:9519`), the lease burned down while the results
waited.

**These two produce identical rig logs.** Under H1 the worker is inside
`handle_stripe`; under H2 it is blocked in `_sendall`, also inside `handle_stripe`;
in both cases it emits nothing and reports `idle` at the eventual shutdown (§4.2).

### 5.1 Coordinator-side evidence — real, one-sided, and NOT decisive

This is a different surface from the rig logs and it does bear on the question.
Recorded as **PLAUSIBLE-NOT-PROVEN**, leaning H2:

- **Every completed stripe was fast.** 27 of 32 phase-2 stripes carry
  worker-reported `elapsed_s`; the full range is **0.927 s → 6.818 s**. Phase 1:
  n=32, min 0.965, median 13.135, max 15.102. The five without `elapsed_s` are
  exactly `s27`-`s31`, the five holding a live lease at abort.
- **The terminal worker had just done the identical job in 6.069 s.**
  `rrig6600b:gpu1` completed `st1_s12` — same phase, same family, same 34
  sub-stripes, same card — in **6.069 s**, then produced nothing on `st1_s29` for
  ≥300 s. For H1 to hold, that card must have slowed by ~50× with no fault
  recorded anywhere and no `dmesg` evidence.
- **The coordinator was demonstrably acceptance-bound.** `drain_total = 790.415 s`
  of a `loop_seconds = 966.356` control loop (**82%**); `inbound_qsize_high_water
  = 547`; `deferred_high_water = 716` against `bound_in_force = 1113`;
  `staging_jobs_per_sec = 1.273` (attempt 1: 3.055); `slow_msg_events = 332`.
- **Individual frames were taking seconds to process.** `st1_s27`/`st1_s28`
  `sub_stripe_result` frames were processed at **2.13-2.56 s each** with
  `inbound_qsize` 345-381 sustained, from 13:31:01 to 13:32:12.
- **Renewal itself was working.** `s27`/`s28` renewed continuously to 13:32:12.915
  and 13:32:10.782 — their last accepted frames. The mechanism was not broken
  fleet-wide; it stopped for three stripes specifically.

**Why this is not decisive.** The coordinator's `SLOW_MSG` record is
**threshold-gated at 0.25 s** and per-frame; it is not an inventory of arrivals.
`st1_s29`/`s30`/`s31` appear in the coordinator log **0 times** apart from the
terminal line itself — which is equally consistent with *no frames arriving*
(H1) and with *frames arriving but never dequeued* (H2), because a frame that is
never processed is never timed and therefore never logged. **The absence is an
absence of measurement, not an absence of traffic** (VIR-5). Additionally,
`deferred_high_water = 716` is a scalar with **no per-stripe attribution**, so
the deferral path cannot be checked for these three either.

---

## 6. Classification

### CONFIRMED
1. Bundle integrity: 63/63 files verified.
2. The §15 emitter worked on all 25 workers; one distinct event sequence.
3. All 25 ended `explicit_shutdown`, `exc_class = null`, `reconnect_attempted =
   false`, `session_generation = 1`. No crash, no transport loss, no reconnect,
   no §13 refusal anywhere in the fleet.
4. The release barrier released all 25 from one token at 13:16:10.600842; the
   `waited_s` ladder (88.539 → 19.508, 3.00 s steps) is the proof.
5. Run nonce uniform and correct across all 25; no stale nonce present.
6. Rig logs contain no timestamps; the blind window is ≈13:26:48 → ≈13:32:16
   (≈5 min 28 s) and the lease expiry lies inside it.
7. The rig logs have **zero** discriminating power for H1 vs H2 — one structural
   skeleton across all 25, the three long-holders indistinguishable from the 22
   completers.
8. `assignment_active_at_loss` is structurally always `false` on the
   `explicit_shutdown` path — VIR-2 vacuous; not evidence.
9. `lease_expires_at − 300` = last accepted progress (or claim), **not** claim
   time. Corrects Alpha's launch-report derivation.
10. Four `zeus_local/` files are stale 2026-08-04 artifacts, including a
    `BrokenPipeError` traceback that is **not** attempt-6 evidence.

### UNRESOLVED — and no cause may be claimed
11. **Whether the three long-held stripes stalled worker-side (H1) or were not
    accepted in time (H2).** The bundle cannot separate them, and neither can the
    coordinator log on its own. Alpha claims no cause.
12. Why these three stripes specifically, when 27 others on the same cards in the
    same phase completed in under 7 s.

### PLAUSIBLE-NOT-PROVEN
13. **H2 (acceptance backlog) is the better-supported hypothesis**, on the §5.1
    evidence — chiefly the same worker completing the identical stripe shape in
    6.069 s, and an 82%-drain-bound coordinator processing individual frames at
    2.1-2.6 s with a 547-deep inbound queue. **This is a lean, not a finding.**
    It rests on an absence of measurement and must not be recorded as a cause.
14. That the phase-2 slowdown relative to phase 1 (median `elapsed_s` 13.135 →
    ~6.5, yet the stage ran longer) reflects acceptance cost rather than compute
    cost. Not established.

---

## 7. Instrumentation required to separate H1 from H2 in a future run

**This is the actionable output of this document.** Each item is stated as a
measurement and a decision rule. Items A-B are worker-side; C-E coordinator-side.
**Neither side alone is sufficient — A/B establish what the worker did, C/E
establish what happened to its output, and only the pair is decisive.**

### A. Worker-side per-stripe lifecycle records, with time
Emit, at **stripe granularity, not per sub-stripe** (which respects the §15
no-high-rate-noise bar — 32 stripes/stage, not 34 frames each):

| record | fields |
|---|---|
| `STRIPE_BEGIN` | `stripe_id`, `sub_count`, wall clock, monotonic |
| `STRIPE_COMPUTE_DONE` | `stripe_id`, monotonic elapsed in kernel execution only |
| `STRIPE_SEND_DONE` | `stripe_id`, monotonic elapsed, **cumulative seconds blocked in `_sendall`** |
| `STRIPE_END` | `stripe_id`, `substripes_sent`, total elapsed |

**Decision rule.** `STRIPE_COMPUTE_DONE` absent at session end ⇒ **H1**.
`STRIPE_COMPUTE_DONE` present with a large `_sendall` block time in
`STRIPE_SEND_DONE` ⇒ **H2**. This single pair of records resolves the question
outright, and the `_sendall` accumulator is the specific field that does it —
under H2 the worker is *inside* `handle_stripe` but not computing, which is the
state no current field can express.

### B. Timestamps on worker log lines
The worker log format emits no `asctime`. Every record in §A is unanchorable
without it, and the existing four session events are unanchorable today. **A
timestamp on the worker log line is a precondition for A being useful**, not a
separate nicety.

### C. Frame arrival time, separate from processing time
The coordinator currently records only *processing duration*, and only above a
0.25 s threshold. Stamp each inbound frame at `recv` and carry that stamp to
processing, then record **queue residency = processed_at − arrived_at** per frame.

**Decision rule.** Frames from the expiring stripe present with large residency ⇒
**H2**, conclusively. No frames arrived ⇒ **H1**, conclusively. This is the
coordinator-side half that makes §A's answer checkable rather than merely
plausible.

### D. Ungated periodic accounting of active stripes
Replace reliance on threshold-gated `SLOW_MSG` with a periodic (e.g. 10 s)
structured record, one line per active stripe: `stripe_id`, `worker_id`,
`age_since_claim`, `age_since_last_accepted_progress`, `lease_remaining`,
`frames_received`, `frames_deferred`.

This is what makes the failure *approach* observable rather than only its
arrival. In attempt 6 the run gave no signal at all between 13:27:14 and
13:32:15; under D, `lease_remaining` on three stripes would have been visibly
decaying for five minutes.

### E. Per-stripe deferral attribution
`deferred_high_water = 716` is a scalar. Record, per stripe and worker, the count
of deferred frames and total deferred seconds, so the §2.19 deferral path can be
checked for a specific stripe. Without this, "its frames were deferred" is
unfalsifiable.

### F. Scope note
A, B, D, E are additive records. C requires carrying one timestamp field from
`recv` to processing. **None of them changes any control-flow, lease, scheduling
or acceptance semantics**, and none is proposed as a fix for the failure — they
exist so the next occurrence is *diagnosable* rather than reconstructible only
by inference.

---

## 8. No remedy proposed

Alpha proposes **no** change to `compute_lease_timeout`, the lease model, the
renewal trigger, stripe geometry, the worker pool or the acceptance path.

Raising `compute_lease_timeout` would very likely make attempt 7 pass and is
**the mask-not-fix move Beta ruled against in §2.26** — there recorded against
the pool-size remedy, in the same terms: it would remove the symptom at the
geometry tested while leaving the coupling in place, and would certify a latent
cliff. The remedy decision is Beta's, and on the §7 owner rule it should be taken
against a *measured* cause, which does not yet exist.

**Recommended disposition:** the cause is UNRESOLVED and bounded; the next run
should carry §7's instrumentation so that the cause is *measured* rather than
argued.

---

## 9. Verification-integrity controls (VIR-1…6)

- **execution proof:** every figure re-derived this session on VM101 from the
  frozen bundle; `sha256sum -c` 63/63 OK; ledger queried `mode=ro`; source
  anchors read live at HEAD `5ad170a`.
- **clean control:** the 22 workers that completed normally, and phase 1
  (32/32 done, `elapsed_s` n=32) — same code, same cards, same run.
- **fault-injection control:** none available read-only; **declared absent.** No
  claim here depends on one.
- **completion sentinel:** bundle file count (63) and per-file digests verified
  before and consistent after analysis; all 25 worker logs parsed, none skipped.
- **unavailable-observer behavior:** `SLOW_MSG` is threshold-gated, so the
  absence of `st1_s29/s30/s31` is recorded as **absence of measurement, never as
  absence of traffic** (VIR-5). Rig-log mtimes are collection times and are
  treated as carrying no event information. Host GPU kernel logs remain
  **UNAVAILABLE** (§2.17, unprivileged LXC) and are not cited.
- **audit claim scope:** confined to *what the attempt-6 rig logs can and cannot
  establish about the stage-2 lease expiry*. No claim about the cause of the
  expiry, about hardware, or about any other run.
- **searched surfaces:** the frozen bundle (25 worker logs, coordinator log,
  ledger, evidence/verdict/sampler/liveness/parity artifacts); live source
  `miner/range_miner_worker.py`, `miner/range_miner_coordinator.py` at HEAD;
  `docs/` and the governance trail via the project-facts skill (§2.19, §2.24,
  §2.26, §2.27, §2.31, §2.32 read for prior rulings on lease, back-pressure,
  observability and mask-not-fix).
- **unavailable surfaces:** host-side GPU kernel logs on `.121/.155/.163` (no
  root key auth from VM101); worker process state at the time of expiry (process
  exited); network-level capture (none taken).
- **governance trail searched:** yes — §2.19 (F1/F2 classification law), §2.24
  (`elapsed_s` is stripe service time, not fleet throughput), §2.26 (F-1/F-2 and
  the mask-not-fix ruling), §2.27 (active-lease scheduler contract), §2.31-2.32
  (attempt-2 disposition, Defect A/B certifications).
- **chapters searched:** not applicable — no claim here touches sieve
  mathematics, feature semantics or pipeline staging.
