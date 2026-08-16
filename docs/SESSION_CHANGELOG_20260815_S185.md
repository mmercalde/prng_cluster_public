# SESSION CHANGELOG — 2026-08-15 — S185 — H1/H2 DISCRIMINATION INSTRUMENTATION (Beta CERTIFIED R1–R7)

**Host:** VM101 `zeus-ubuntu-vm` (192.168.3.177), user `michael`, `~/venvs/torch`.
**Base:** HEAD `087b7f450be64db253825b0a18baf07039996608` — unchanged throughout this session.
**Ruling:** `docs/TB_RULING_20260815_H1H2_R7_CERTIFICATION.md` — **R1–R7 CERTIFIED**, R8 not required.
**Spec:** `docs/ATTEMPT6_RIG_LOG_FORENSIC_v1_0.md` §7 (A–F).
**Brief chain:** `~/dashboard_work/CCODE_BRIEF_H1H2_INSTRUMENTATION_{v1_0,R1..R7}_v1_0.md`.
**Implementation report:** `~/dashboard_work/H1H2_INSTRUMENTATION.md` (R7, supersedes R6 in full).

**This changelog accompanies the commit that carries the instrumentation.** It is written before the
commit, describing the tree that is about to be committed; the three source paths in §9 are the
complete `git add` list and were built from that section, not from recall.

**NO FLEET EXECUTION · NO DEPLOY · NO LAUNCH · ATTEMPT 7 REMAINS HELD.**

---

## 0. What this session was

Two things, in order:

1. **Step 1 of the governed sequence — the fleet archive** (§7). Executed against all three rigs,
   read-only with respect to the deployed tree.
2. **This changelog.** No source file was edited this session. The instrumentation itself was built
   and hardened across the R1–R7 arc; S185 is the session in which it is certified, archived against,
   and handed to Michael to commit.

The governed sequence Beta set, and where this session sits in it:

```
[1] archive fleet          <- DONE this session (§7)
[2] commit + dual push     <- Michael's, next
[3] clean-tree proof
[4] deploy ALL TEN governed files
[5] parity 30 MATCH / 0 MISMATCH / 0 UNAVAILABLE
[6] normal prelaunch authority
[7] attempt 7
```

Coordinator-only and worker-only deployment remain **FORBIDDEN** — §7 deploys whole or not at all
(Beta, R1: *"Coordinator-only and split A–E are both REJECTED"*).

---

## 1. Certification

**Team Beta CERTIFIED the H1/H2 instrumentation, rounds R1 through R7.** The ruling is in the repo
verbatim: **`docs/TB_RULING_20260815_H1H2_R7_CERTIFICATION.md`** (176 lines) — *"PASS. The R1→R7
technical review is closed."* The R1–R7 brief chain is the trail behind it.

Beta's final disposition, quoted from that ruling:

```
H1/H2 INSTRUMENTATION R1–R7              CERTIFIED
PRODUCTION DEFECT FOUND                  NO
PRODUCTION CHANGE REQUIRED               NO
R8 REQUIRED                              NO

ARCHIVE FLEET                            AUTHORIZED / REQUIRED NEXT
COMMIT + DUAL PUSH                       AUTHORIZED AFTER ARCHIVE
CLEAN-TREE PROOF                         REQUIRED AFTER COMMIT
DEPLOY ALL TEN GOVERNED FILES            AFTER CLEAN-TREE PROOF
PARTIAL COORDINATOR/WORKER DEPLOYMENT    FORBIDDEN
PARITY                                   REQUIRE 30 MATCH / 0 MISMATCH / 0 UNAVAILABLE
NORMAL PRELAUNCH AUTHORITY               REQUIRED AFTER PARITY
ATTEMPT 7                                HELD UNTIL PRELAUNCH AUTHORITY
```

**This commit is authorized by that ruling and only in that position** — the archive of §7 is the
step it is conditioned on, and it is complete.

**What Beta certified is the artifact identity below**, which Beta verified independently
(`bytes 173,172 · sha256 a2b69fc6…835b2 · check(...) call sites 62 · _ast_calls_to × 8 ·
def check(...) × 1 · Python compile PASS`) and recorded as *"exactly match Alpha's submission."*
It is byte-identical to what is being committed — verified here, this session, not assumed:

| artifact | sha256 | bytes |
|---|---|---|
| `tests/test_s172_h1h2_instrumentation.py` (staged) | `a2b69fc64c11175d1d1b6a599cde34a9dc2608010cb1da1d760e6fc82bc835b2` | 173,172 |
| `~/dashboard_work/test_s172_h1h2_instrumentation_R7_EXACT.py` (Beta's artifact) | `a2b69fc64c11175d1d1b6a599cde34a9dc2608010cb1da1d760e6fc82bc835b2` | 173,172 |

`cmp` → **BYTE-IDENTICAL**. 62 `check(...)` call sites in both.

Production, likewise unchanged since the digests recorded in the R7 report §0:

```
miner/range_miner_worker.py       043522e96b44855f04540b1d2bdb5db003f3428785d7c98e9bfc073ff5a8100d
miner/range_miner_coordinator.py  b97ce5f9b2dc455b615f130d9575c1a285fd43e66e7ed230739849a83d35ab67
```

Both reproduce the report's §0 values exactly, and the ruling records the same two digests. That is
what licenses this changelog to cite that report's regression battery rather than re-running it: the
tree it measured is the tree being committed.

**The ruling's own evidence boundary, and it is the reason the digest match above matters.** Beta
independently extracted the R7 helpers from the submitted artifact and executed the counterexamples
itself — E4's nested-def shape, N1's six reach shapes against two clean `msg` reads, N6's seven
count-shaped returns against the correct `UNAVAILABLE` dict, and N2's inheritance of the same-scope
repair. But on the production-dependent suites Beta was explicit:

> *"I did not independently execute those production-dependent suites here because the corresponding
> live miner tree is not part of this artifact environment. I **accept Alpha's supplied
> regression/digest evidence as the execution evidence for those portions**; the structural R7
> closure itself was independently exercised against the submitted artifact."*

So the regression matrix is Alpha-supplied evidence Beta accepted, not Beta-executed evidence. The
digest reproduction above is what ties it to the tree in front of us, and §5's two live suite runs
are this session's own first-hand re-measurement of the parts that matter most.

Beta also accepted the four declared detector boundaries as **"properly bounded claims, not hidden
proof holes"**, and closed the arc against further speculative rounds: *"I am therefore not opening
another round for hypothetical AST constructions outside the explicitly bounded proof model."*

---

## 2. The question the instrument exists to decide

Attempt 6 died at 13:32:15 on `compute_lease_expiry` — `st1_s29` / `rrig6600b:gpu1` / attempt 0,
after 300 s with **no accepted progress of any kind**. Two hypotheses remained live and no existing
artifact could separate them:

```
H1  worker-side stall or send blocking
H2  coordinator-side acceptance backlog delaying lease renewal
```

**H2 is a lean, not a finding, and nothing here upgrades it.** The same card completed the identical
stripe shape in 6.069 s; all 27 measured phase-2 stripes ran 0.927–6.818 s; the coordinator was 82%
drain-blocked (`drain_total=790.4 s` of 966.4 s), processing frames 2.1–2.6 s behind a 547-deep
queue, `slow_msg_events=332`. But `SLOW_MSG` is threshold-gated, so **zero mentions of s29/s30/s31
is an absence of measurement, not an absence of traffic.**

Why the existing logs could not answer it: reducing all 25 rig logs to their structure gives **one
distinct skeleton** — the three long-holders are shape-identical to the 22 completers. Under H2 the
worker is inside `handle_stripe` but *not computing*, and that is the state no pre-existing field
could express.

**This is an instrument, not a remedy.** No fix for the lease expiry is proposed here, and none of
`compute_lease_timeout`, the F1/F2 lease-origin and claim semantics, the expiry/retry matrix,
`worker_admission_timeout`, the certified drain-fairness and reader-exit repairs, or the sentinel /
liveness / parity / GPU / clean-tree gates was touched. The remedy choice is Beta's, after the next
run produces evidence.

---

## 3. What was built — forensic §7, A through E

The design constraint is binding and shaped everything: **stay inside the §15 no-high-rate-noise
bar.** Accumulate, emit once at a lifecycle boundary. No per-heartbeat and no per-frame emission.

### §A / §B — worker-side stripe lifecycle, timestamped (`miner/range_miner_worker.py`)

Four records at **stripe** granularity — 32 stripes per stage, not 34 frames each:

```
STRIPE_BEGIN         stripe_id · family_name · sub_count · seed_start · seed_count · wall · mono
STRIPE_COMPUTE_DONE  compute_s (kernel execution only) · subs_computed · sub_count ·
                     send_s · stripe_send_stall_s · mono
STRIPE_SEND_DONE     substripes_sent · send_s · stripe_send_stall_s ·
                     stripe_send_syscall_s · stripe_send_syscall_max_s ·
                     stripe_send_lock_wait_s
STRIPE_END           outcome · subs_computed · subs_sent · total_s · unattributed_s ·
                     compute_done · send_done · stripe_send_calls · session_generation
```

`configure_worker_logging()` adds the UTC timestamp to the worker log line. **§B is a precondition
for §A, not a nicety** — the worker format previously emitted no `asctime`, which left the four
existing session events unanchorable in time.

`send_accounting()` / `send_accounting_all()` are the R1-A per-thread accounting surface:
`MinerFramedSocket` keeps socket-level cumulative counters, so a stripe-owned block time has to be
derived per thread, not read off the socket. **R1-A was certified and FROZEN at R1 and has come back
byte-identical in every round since.**

**Decision rule (§7A).** `STRIPE_COMPUTE_DONE` absent at session end ⇒ **H1**. Present, with a large
`stripe_send_stall_s` in `STRIPE_SEND_DONE` ⇒ **H2**.

### §C — arrival time, separate from processing time (`miner/range_miner_coordinator.py`)

Each inbound frame is stamped at `recv` with a monotonic arrival time and a unique token
(`_next_frame_token`, `stamp_frame_arrival`, `frame_token`, `frame_queue_residency`); residency =
`processed_at − arrived_at` is carried to processing and aggregated per stripe. The stamp lives
**outside protocol serialization** — the five-field inbound envelope is untouched, so this is not a
wire change.

The token is what removes the post-`put()` FIFO-order assumption the R1 inventory had (R2-1).

**Decision rule (§7C).** Frames from the expiring stripe present with large residency ⇒ **H2**,
conclusively. No frames arrived ⇒ **H1**, conclusively.

### §D — ungated periodic accounting

`active_stripe_accounting()` / `maybe_emit_active_stripe_accounting()` emit one aggregated
`ACTIVE_STRIPES` record per interval, carrying a row for every active stripe:

```
stripe_id · worker_id · attempt · age_since_claim_s · claim_precision ·
age_since_last_accepted_progress_s · age_since_last_accepted_frame_s · lease_remaining_s ·
frames_enqueued · frames_dequeued · frames_pending · oldest_pending_age_s · frames_received ·
heartbeats_accepted · subresults_accepted · age_since_last_subresult_s · frames_deferred ·
residency_max_s
```

Beta approved the aggregated one-record-per-interval shape at R1 in place of one line per stripe.
**A row whose `frames_enqueued` climbs while `subresults_accepted` stays put IS the coordinator-side
backlog** — visible as it happens rather than reconstructed afterwards (R1-B), with the message
classes deliberately kept apart (R1-D).

This is what makes the failure *approach* observable. In attempt 6 there was no signal at all
between 13:27:14 and 13:32:15; under §D, `lease_remaining` on three stripes would have been visibly
decaying for five minutes.

### §E — per-stripe deferral attribution

`note_stripe_frame_{enqueued,dequeued,accepted,deferred,released}`,
`note_stripe_renewing_progress`, `note_stripe_claimed`, `stripe_rx_snapshot()` and
`emit_stripe_rx_summary()` produce one `STRIPE_RX_SUMMARY` per stripe at its terminal disposition,
carrying the frame inventory, residency statistics, deferral counts and seconds, heartbeat
accept/renew counts, and the three age fields.

`deferred_high_water = 716` was a scalar; without per-stripe attribution *"its frames were
deferred"* is unfalsifiable.

**Three-valued observer semantics** (R1-C, certified): `OK` / `UNAVAILABLE` / `NO_OBSERVATION`, and
`None` — never `0` — is the vocabulary for a genuinely unobserved duration. N6 is the structural AST
guard that keeps a count-shaped value from ever being returned on the unavailable path.

### §F — scope

Additive records plus one timestamp carried from `recv` to processing. **No control-flow, lease,
scheduling or acceptance semantics change.** The certified lease-renewal sites are unchanged; the
renewing clock still updates only from the booleans returned by the two real `_renew_active_lease()`
calls.

---

## 4. The R1–R7 arc

Seven review rounds. **Two produced production repairs; five were test-authority hardening.** The
distinction matters — Beta stated at R4, R5, R6 and R7 that no production defect was indicated.

| round | Beta's finding | outcome |
|---|---|---|
| **v1_0** | brief: build §7; report the scope consequence of touching the worker | deployment strategy certified — **§7 whole or not at all**; coordinator-only and split A–E REJECTED |
| **R1** | R1-A `send_block_s` is not stripe-owned (socket-level counters) | **production**: per-thread send accounting. Certified and FROZEN thereafter. R1-C three-valued observer certified; R1-E withdrawn and closed |
| **R2** | R2-1 producer/consumer race in the queue inventory | **production**: token identity replaces the post-`put()` FIFO assumption; R2-1b forces all 24 consumer-first inversions |
| **R3** | R3-1 `stripe_rx_snapshot()` was not a snapshot (shallow alias) | **production**: detached snapshot; R3-2 duplicate-claim withdrawn, replaced by a one-per-put structural proof |
| **R4** | two gate-reach defects — R3-1b did not red on the shallow snapshot it advertised | test only; M13 shallow-snapshot falsifier added |
| **R5** | E4 did not prove the defer note was in the *same* critical section | test only; falsifier reach ≠ proof completeness |
| **R6** | known proof holes may not remain inside the suite that is the non-regression authority | test only; N1/N2/N6 hardened |
| **R7** | three executable holes + one docstring over-claim | test only; `_iter_same_scope`, dynamic-`getattr`-on-`self`, fixed-point taint. **The re-run sweep found two more of the same class inside the R7 repairs; both closed** |

**The pattern is §2.30's, and it recurred inside the fix twice** (R6-1 and R7-1 were both found
*inside* repairs). That is why the sweep is re-run against its own newest helpers every round.

**Four boundaries are DECLARED, not proven** (R7 §4), each with its reason:
`_ast_calls_to` matches callee by name, not receiver identity · `same_scope=True` refuses a nested
`def` that is genuinely invoked in-scope (a conservative false-RED that fails safe) · N1's
dynamic-`getattr` rule is scoped to `self` and its aliases · N6's taint follows bindings that
*mention* a tainted name, not flow through a call return or mutation of an untainted container.

---

## 5. Evidence at final state

**Measured on VM101 this session, on the exact tree being committed** (`~/venvs/torch` active):

| check | result |
|---|---|
| `tests/test_s172_h1h2_instrumentation.py` | **62/62 checks green**, rc=0, completion sentinel printed |
| `check(...)` call sites | **62** |
| staged test file vs Beta's R7 artifact | **byte-identical**, `a2b69fc6…835b2` |
| production digests vs R7 report §0 | **both reproduce exactly** |
| `tests/test_s172_phase4_coordinator.py` | **62/63 — Gate 22 only** |

Gate 22's red names **exactly** `{'tests/test_s172_h1h2_instrumentation.py'}`. This is the known
development-state behaviour — the detector builds `changed_py` from `git status --porcelain`, which
includes untracked files. **It is not a regression and not a reason to widen the allowlist** (Beta
rejected permanent allowlisting at §2.33); **it self-clears on the commit this changelog
accompanies.**

**From the R7 report, on the tree just proven identical by digest** — regression at final state:
attempt-6 remediation 78/78 · D6 integration 82/82 · D6 liveness 59/59 · GPU gate 9/9 · clean-tree
admission 31/31 · staging back-pressure 50/50 · F1/F2 active lease 16/16.

**AST scope proof** (R7 §5.2): worker `91 → 101` definitions (ADDED 10 · CHANGED 8 · REMOVED 0);
coordinator `311 → 337` (ADDED 26 · CHANGED 6 · REMOVED 0). **NO-TOUCH violations: NONE.**

Incidental but worth recording: the phase-4 suite's own output now carries `STRIPE_END` and
`SESSION_END` records with the new accounting fields populated, so the instrumentation is exercised
by a suite that was not written for it.

---

## 6. Every new field has an arm that fails on wrong input

The brief was explicit about why: this arc has ten recorded instances of a check that passes on a
fact it does not verify — including one inside the mutation machinery, and `assignment_active_at_loss`,
which appears 25 times, reads like a discriminator, and is **structurally guaranteed false** on that
path. A new field that cannot vary is the next vacuous discriminator.

Against that specifically: **62 checks, including 14 mutants (M1–M14) and a post-mutation integrity
arm (N7)**, with each detector carrying both a clean control and executed wrong-shape falsifiers —

```
E4  5/5 wrong shapes rejected, live insertion block accepted
N1  17 methods; 6/6 wrong rejected, legitimate const AND dynamic reads on `msg` accepted
N2  3/3 wrong rejected, correct shape accepted
N6  15 methods; 7/7 wrong rejected, correct UNAVAILABLE dict accepted
```

---

## 7. Step 1 of the governed sequence — fleet archive, COMPLETE

Executed this session against all three rigs from VM101. Same shape as the pre-D6 archive
(`/home/michael/rig_archive/pre_d6_deploy_20260815_062008Z`): `MANIFEST.tsv` ·
`ARCHIVED_SHA256SUMS.txt` · `tree/<original relpath>`, copied `cp -p` so modes and mtimes survive.

```
archive id   pre_h1h2_deploy_20260816_042747Z      (one UTC instant, same id on all three rigs)
CREATED_UTC  2026-08-16T04:27:47Z
SOURCE_ROOT  /home/michael/distributed_prng_analysis
result       10 PRESENT · 0 ABSENT · SELFCHECK PASS 10/10 on each rig
```

| rig | MANIFEST.tsv sha256 | ARCHIVED_SHA256SUMS.txt sha256 |
|---|---|---|
| rrig6600 (.122) | `e8603d75810f4f28ea3db191528fcffb1ebc11b9206aadbd083417bca60610d3` | `a67e07b48634048429ed826b6a424595a56a2d44e245cba2e3d46d0e23b442f7` |
| rrig6600b (.156) | `ac08a96b0409cc2089608e1d3192519a95f226a90a8c5b47595de619064a60dd` | `a67e07b48634048429ed826b6a424595a56a2d44e245cba2e3d46d0e23b442f7` |
| rrig6600c (.164) | `0fb14c1f8b44bd0fb55ebd6d03697bd8355b9840ca09cfea06ccbcf394aa0cc4` | `a67e07b48634048429ed826b6a424595a56a2d44e245cba2e3d46d0e23b442f7` |

**The sums file is byte-identical across the fleet** — deployed governed content is the same on all
three machines, mtimes included. The manifests differ only in their `HOST` line, and each rig
reported its own `hostname` so three machines cannot be one machine answering thrice.

The file set is the canonical pin `GOVERNED_FILES` at `scripts/gate12_parity_gate.py:174-185`, read
from source rather than transcribed. `execution_set.py` was `ABSENT` in the pre-D6 manifest and is
**present** now, so this is a complete ten.

**Self-check ran twice**: once inside the archiver, and again from a **fresh SSH session after the
writing process exited** — 10/10 `OK`, rc=0, `find . -type f` = 10 on each rig. The archiver also
compared each copy's digest against its source digest before recording it, and refuses a
pre-existing archive id.

**Cross-check:** all ten archived digests equal the `git show HEAD:<path>` blob digests at `087b7f4`,
so the archive captures exactly the certified deployed tree (`3218718`; only documentation landed
after it). The two files this commit modifies are archived at their **pre-instrumentation** digests
— `992464ba…cf4f` (worker) and `5cf41f83…7d8b` (coordinator) — which is precisely the state a
rollback would need.

---

## 8. What a future run shows

**H1:** `compute_s` dominates, `stripe_send_stall_s ≈ 0`, arrival tracks acceptance.

**H2a:** `STRIPE_COMPUTE_DONE` at ~6 s, then `stripe_send_stall_s ≈ 294 s`.

**H2b:** the worker looks healthy while the coordinator side shows
`frames_enqueued: 34 · frames_dequeued: 0 · frames_pending: 34`, `oldest_pending_age_s` climbing,
`subresults_accepted: 0`, `heartbeats_accepted: 9 · heartbeats_renewed: 0`, and
`age_since_last_accepted_progress_s: None` beside a live `age_since_last_accepted_frame_s`.

**Residual, unchanged and stated as worded in R2 §6.3:** the **zero-enqueue killed-worker corner** is
**unresolved under this bounded instrumentation design and its no-high-rate-noise constraint** — not
architecturally irreducible. Cases B, C, D and E stand. `early_dequeue_events` and `frames_untokened`
remain expected-zero in production.

**Unavailable by construction:** the behaviour of this instrumentation under real fleet load. The
gates are CPU-only fixtures, and a fixture is not a fleet.

---

## 9. Files changed — this is the `git add` list

```
M  miner/range_miner_coordinator.py     +966 / -3
M  miner/range_miner_worker.py          +473 / -22
A  tests/test_s172_h1h2_instrumentation.py           3,485 lines, 173,172 bytes
A  docs/TB_RULING_20260815_H1H2_R7_CERTIFICATION.md    176 lines
A  docs/SESSION_CHANGELOG_20260815_S185.md           (this file)
```

Stage explicitly, never `git add -a`.

The ruling is on that list because §1 now cites it as the authority for this commit, and a
changelog citing an untracked file is a dead reference in a fresh clone — the governance trail has
to travel with the work it authorizes.

---

## 10. What this session did NOT do

- **Nothing committed, nothing pushed.** Michael commits and dual-pushes (`origin` + `public`).
- **Nothing deployed.** No governed file was written to any rig. The only rig contact was the
  read-only archive of §7.
- **No fleet execution, no coordinator, no ledger row, no port bound.**
- **Attempt 7 HELD.** Steps 3–7 of the sequence in §0 are untouched, and the 30/30 parity run is a
  precondition for dispatch, not a formality — the last time a stale worker reached the fleet,
  24 of 25 daemons died at argparse (D6 dry run #1, S184 §0).

---

## 11. Verification-integrity controls (VIR-1…6)

- **execution proof:** the H1/H2 suite was run live this session (rc=0, 62/62, completion sentinel);
  the phase-4 suite was run live (62/63, Gate 22 assertion text captured); `sha256sum` and `cmp`
  outputs for all four artifact identities; the archive self-check ran twice, the second time in a
  process that could not have written it.
- **clean control:** live source passes every detector, and each carries a correct-code control —
  N1 accepts both a constant and a dynamic read on `msg`; N6 accepts the `UNAVAILABLE` status dict;
  E4 accepts the live insertion block and correctly counts a comprehension in the critical section.
- **fault-injection control:** 14 mutants plus per-detector wrong-shape programs (R7 §4 enumerates
  which programs were executed against which detector).
- **completion sentinel:** printed by the suite and quoted in §5.
- **unavailable-observer behavior:** `None`, never `0`, for an unobserved duration; three-valued
  `OK` / `UNAVAILABLE` / `NO_OBSERVATION` status; the archiver reports a missing governed file as
  `ABSENT`, never as a zero-byte success.
- **audit claim scope:** this changelog claims (a) certification status, cited to the in-repo ruling
  and quoted from it, (b) artifact identity, (c) suite results measured here, (d) the archive
  result. Where the ruling's evidence is Alpha-supplied rather than Beta-executed, §1 says so. **It makes no
  claim about the cause of the attempt-6 lease expiry.** H2 remains a lean; R1-E's withdrawal
  stands; the duplicate-idempotence claim remains withdrawn with the one-per-put structural proof in
  its place.
- **searched surfaces:** the working tree at `087b7f4` (`git status --porcelain`, `git diff
  --numstat`, `git show HEAD:<path>`); live source of both miner files by digest;
  **`docs/TB_RULING_20260815_H1H2_R7_CERTIFICATION.md`, read in full**; the R1–R7 brief
  chain and `~/dashboard_work/H1H2_INSTRUMENTATION.md`; `docs/ATTEMPT6_RIG_LOG_FORENSIC_v1_0.md` §7;
  `scripts/gate12_parity_gate.py` for the governed pin; `docs/SESSION_CHANGELOG_20260815_S184.md`;
  all three CT100 filesystems over SSH; the governance trail via the project-facts skill
  (§2.19, §2.26–2.28, §2.30–2.33).
- **unavailable surfaces:** the instrumentation's behaviour under real fleet load (§8); the
  production-dependent regression suites **as executed by Beta** — Beta did not run them and said so
  (§1), so that portion of the ruling rests on Alpha-supplied evidence; rig-side GPU kernel logs,
  which remain readable only from the Proxmox hosts.
- **governance trail searched:** yes. **chapters searched:** not applicable — no claim here touches
  sieve mathematics, feature semantics or pipeline staging.
