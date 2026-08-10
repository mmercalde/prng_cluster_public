# CLAUDE CODE REPORT — F1/F2 AMENDMENT, REVISION 1 (NARROW)

**Date:** 2026-08-09 · **Host:** VM101 (`zeus-ubuntu`, `192.168.3.177`), repo
`~/distributed_prng_analysis`, venv `~/venvs/torch` · **Base:** `eecfff7`, amendment uncommitted in
the working tree.

**Authority:** Team Beta ruling *"F1/F2 ACTIVE-LEASE AMENDMENT REVIEW"* (2026-08-09), transcribed in
`docs/CLAUDE_CODE_INSTRUCTIONS_F1_F2_R1.md`. Four narrow items only. Architecture accepted and
closed; nothing on the §19 do-not-touch list was reached.

**Constraints honoured:** no commit · no push · no pipeline launch · no fleet launch · no port-5700
bind · **`worker_pool_size = 25` NOT applied** · gate 12 remains HELD. Nothing was redesigned or
re-argued.

---

## 0. BASE VERIFICATION (before any edit)

| check | result |
|---|---|
| `git rev-parse HEAD` | `eecfff7061a1bc7671d94cc323936bc60a543de3` |
| amendment present, uncommitted | `M miner/range_miner_coordinator.py`, `M tests/test_s172_phase4_coordinator.py`, `?? tests/test_s172_f1_f2_active_lease.py` |
| `tests/test_s172_f1_f2_active_lease.py` | **13/13 checks green** (`/tmp/f1f2_base.log`) |
| untracked runtime residue | present (`miner_ledger.db-shm/-wal`, `optimal_window_config.json.stale_*`) — expected, not a stop condition |

A **full sequential baseline sweep** was run BEFORE any edit, so every red claimed below as
pre-existing is measured rather than assumed (`/tmp/baseline_*.log`):

| suite | baseline | reds |
|---|---|---|
| f1_f2_active_lease | 13/13 | — |
| staging_backpressure | **48/50** | `G-MATRIX-DIFF-a`, `G-LEASE` |
| staging_partb | 24/24 | — |
| elapsed_roundtrip | 6/6 | — |
| phase5_d3_5_finalizer | 60/60 | — |
| phase4_coordinator | **62/63** | Gate 22 (untracked `.py`) |
| admission_liveness | **15/16** | `G-FORBIDDEN-ABSENT` |

These are exactly the F1/F2-chargeable reds Beta's §C predicted, plus the known Gate-22
untracked-file sensitivity.

---

## 1. THE EXACT-VS-PREFIX API SPLIT AS BUILT, AND EVERY CALL SITE

### 1.1 The split

Two selectors, explicitly named, **mutually exclusive**, and — on the scheduler — **keyword-only**,
so the two can never be confused positionally. Nothing anywhere inspects the shape of the string to
infer what the caller meant.

| API | `file:line` | selector parameters |
|---|---|---|
| `MinerLedger.pending_stripes` | `miner/range_miner_coordinator.py:2084-2087` | `stage_prefix: Optional[str] = None`, **keyword-only** `exact_stripe_id: Optional[str] = None` |
| `RangeMinerCoordinator.schedule_pending_stripes` | `miner/range_miner_coordinator.py:2967-2978` | **keyword-only** `stage_prefix`, `exact_stripe_id` |

Implementation of the ledger selector (`:2107-2119`):

```
exact_stripe_id is not None   ->   AND stripe_id = ?          (identity)
stage_prefix truthy           ->   AND stripe_id LIKE ?       (prefix + '%')
neither                       ->   (no clause; whole-run backlog — the legacy default)
both                          ->   ValueError                 (:2108-2113)
```

`ValueError` rather than a precedence rule is deliberate: a caller that supplies both has not
decided which question it is asking, and silently answering one of them is the defect class being
closed.

### 1.2 Every call site of each

| `file:line` | selector used | why |
|---|---|---|
| `miner/range_miner_coordinator.py:2953` (`assign_stripes` → scheduler) | `stage_prefix=prefix` | initial handoff for the stage just planned |
| `miner/range_miner_coordinator.py:6962` (`serve_trial` scheduler pass) | `stage_prefix=_stage_prefix(stage_idx)` | every serve-loop pass, per stage |
| `miner/range_miner_coordinator.py:5564` (`_handle_stripe_failure_locked`, hybrid immediate placement) | **`exact_stripe_id=stripe_id`** | **THE DEFECT SITE.** One stripe, by identity |
| `miner/range_miner_coordinator.py:3018-3019` (scheduler → ledger) | forwards both, unchanged | single pass-through |

There is exactly **one** `exact_stripe_id` call site in production, and it is the hybrid retry.

### 1.3 `assign_stripes(stripe_prefix=...)` — deliberately NOT renamed, and why

`assign_stripes` keeps its existing parameter name (`miner/range_miner_coordinator.py:2876`,
`:2868`). **It is not a selector.** It is used to CONSTRUCT stripe ids (`f"{prefix}_s{idx}"`,
`:2906`) and is forwarded to the scheduler as `stage_prefix=` (`:2953`). Its docstring now says so
explicitly (`:2895-2900`): *"a STAGE prefix and nothing else… A complete stripe id is not a legal
value."*

Renaming it would have edited **13 call sites across 10 committed test suites**, eight of which
(`phase5_d0` ×4, `d1_engine`, `d1_workflow`, `d2`, `d4`, `d5`, `d6_threshold`, `d6_production`) sit
in suites **not in Beta's §13 verification list**, and would therefore have been changed without
being re-run.
That is a worse trade than a naming wart, and it is out of scope for a narrow revision. **Flagged
here rather than decided unilaterally** — if Beta wants the rename, it is mechanical and the
affected suites must be added to the verification list.

### 1.4 Beta's alternative was NOT taken

Beta's alternative — drop immediate targeted placement entirely and always return `requeued` — was
not taken, per Beta's own preference for the narrower correction. No concrete reason was found that
the narrow correction cannot work: the exact selector returns the one row or none, and both
outcomes (`reassigned` / `requeued`) remain reachable and are both gated.

---

## 2. THE LEXICAL-COLLISION GATE

**`G-F1-EXACT-STRIPE-COLLISION`** — `tests/test_s172_f1_f2_active_lease.py`.

**Stripe count: 32** — the full authorized gate-12 stage geometry (`N = 32`, `MACRO * N` seeds),
not a reduced fixture. The collision set is asserted to actually exist before anything is driven:
`run__st0_s1` plus **all ten** siblings `run__st0_s10 … run__st0_s19`, every one of them a `pending`
row a prefix query would have swept in.

**Construction** (roles read off the ledger, never assumed from round-robin order): 3 cohort
workers, hybrid family, phase 3. `A` = whichever worker holds `s1`; the other two hold `s0`/`s2` and
are asserted **compute-busy** via the production `compute_busy_worker_ids`. When `s1`'s retry
requeues, `A` becomes the only idle worker — and `A` is the prior claimer, so no legitimate
alternate is free.

**Asserted, exactly as Beta specified:**

- `s1` remains `pending`, `current_attempt == 1`, `claimed_by` still `A`;
- `s10 … s19` **unchanged** — compared as a `{stripe_id: (state, claimed_by)}` snapshot taken before
  and after, so no sibling can move in any way;
- returned `action == "requeued"`, `worker_id is None` — **not** `"reassigned"`;
- trial still `running`.

Then a legitimate alternate is freed (`record_stripe_complete` on its own stripe) and the gate
proves `s1 → claimed by that alternate`, `lease_expires_at == T2 + compute_lease_timeout`
(**fresh**, arithmetic not inequality), and `claimed_by != A`.

### 2.1 The mutant's OBSERVED behaviour

**`G-F1-EXACT-STRIPE-COLLISION/M`** restores prefix-as-exact by routing `exact_stripe_id` back
through the LIKE-scoped parameter, proves the mutated path **executed** (`executed["n"] >= 1`, with
an explicit "the mutant was never called" failure if not), and requires the drive to raise.

**Red-first against the genuine pre-fix production source** (both fixes reverted in the live file,
suite run, file restored — `/tmp/f1f2_redfirst.log`):

```
[FAIL] G-F1-EXACT-STRIPE-COLLISION  32-stripe geometry: s1 vs s10-s19:
       expected 'requeued' (every alternate is compute-busy);
       got {'action': 'reassigned', 'worker_id': 'host1:gpu0', 'attempt': 1,
            'phase_degraded': True}
```

That is Beta's predicted consequence reproduced literally: a **non-empty `placed`** produced by an
unrelated sibling, reported as `action="reassigned"` naming a worker that never took `s1`. Under the
same pre-fix source the mutant gate also reds (`the mutant never executed` — the exact selector does
not exist to intercept), which is the correct signal.

---

## 3. HOW AN ALREADY-ABORTED / RACE-LOST ABORT RECONSTRUCTS, AND WHY B CANNOT ESCAPE

### 3.1 The mechanism

`abort_trial` (`miner/range_miner_coordinator.py:5801`) now has a single post-decision block
(`:5863-5886`) that runs for **both** non-first paths — the already-aborted read (`:5848-5849`) and
the lost atomic race (`:5853-5861`):

```python
if not first:
    durable = self.ledger.get_trial(run_id)          # RE-READ, never the stale pre-read
    terminal = self._terminal_from_trial_row(durable)
    reason   = terminal.reason
    if durable is not None and durable["abort_event_id"]:
        abort_event_id = durable["abort_event_id"]
```

`_terminal_from_trial_row` (`:5925-5956`, a `@staticmethod`) rebuilds a `TerminalRecord` from the
five durable columns `terminal_class / terminal_reason / terminal_stripe_id / terminal_worker_id /
terminal_attempt`.

The re-read is deliberate: on the race path the row captured before `mark_trial_aborted` predates
the winning write, so reusing it would reconstruct from a row that is not the winner's.

### 3.2 Proof that B reaches no outward surface

There are exactly three outward surfaces, and after the block **all three are derived from
`terminal`, which is now the durable record**:

| surface | construction | line |
|---|---|---|
| durable ledger row | untouched — `mark_trial_aborted`'s `WHERE state='running'` cannot rewrite it | `:1563-1600` |
| ERROR log | gated on `first`; a non-first call emits none at all | `:5888-5892` |
| abort event (returned **and** delivered to `Phase5Sink.abort_trial`) | `{... "reason": reason, **terminal.as_event_fields()}` | `:5907-5909` |

`reason` — the legacy prose field, which is the one channel that previously still carried the later
caller's own string — is rebound to `terminal.reason` inside the block. `abort_event_id` likewise
adopts the winner's durable id, so the replayed delivery is idempotent **by the winner's identity**,
not by a locally recomputed one.

A `terminal_class IS NULL` durable row (the bare-API path) does **not** fall back to the caller's
proposal. It reconstructs an explicit `coordinator_error` record stating that the first transition
persisted no record — because *"the earlier transition recorded nothing"* and *"the earlier
transition recorded what I am proposing now"* are different facts.

### 3.3 `G-F2-IDEMPOTENT-PARITY`

Abort with **A** (`compute_lease_expiry` / `AAA…` / `run__st0_s0` / `hostA:gpu0` / attempt 0), then
abort the same run with a deliberately contradictory **B** (`stripe_error` / `BBB…` /
`run__st0_s9` / `hostZ:gpu9` / attempt 7). Asserted:

- durable row still A on **all five** columns;
- **no second terminal ERROR record** (`[m for m in records if "TRIAL TERMINAL" in m] == []`);
- the returned abort event is A — checked field-by-field **and** by scanning `repr(event)` for any
  of `BBB`, B's class, B's stripe id, B's worker id;
- the **replayed sink delivery** (`sink.aborts[1]`) is A, with the same `event_id` as the first;
- `event["reason"]` is A's prose, not `"BBB"`.

**The race-shaped case is exercised**, exactly as Beta required: `mark_trial_aborted` is wrapped so
a competing transition lands with A *inside* the call, making the caller's `mark_trial_aborted`
return False after it had read `state='running'`. Asserted: `first is False`, durable is A,
`abort_event_id == "winner:abort"`, the returned event carries the **winner's** event id, no log
mentions B, and the single sink delivery is A.

**Mutant:** `_terminal_from_trial_row` replaced by `lambda row: B_REC` — the pre-fix behaviour
exactly — must red.

**Red-first against genuine pre-fix source** (`/tmp/f1f2_redfirst.log`):

```
[FAIL] G-F2-IDEMPOTENT-PARITY: ('idempotent abort return',
  {'event_type': 'trial_abort', 'run_id': 'run', 'event_id': 'run:abort', 'reason': 'BBB',
   'terminal_class': 'stripe_error', 'terminal_reason': 'BBB a later contradictory proposal',
   'terminal_stripe_id': 'run__st0_s9', 'terminal_worker_id': 'hostZ:gpu9', 'terminal_attempt': 7})
```

The reachable divergence Beta described, reproduced: durable = A, log = A, replayed event = B.

---

## 4. WHICH ASSERTIONS CHANGED IN THE SUPERSEDED GUARDS — ENUMERATED

**The superseding invariant**, identical at every site:

1. the four terminal decisions, **in source order**, read by AST off the live source —
   `non_retryable → constant_phase → no_alternate_worker → hybrid_second_failure`;
2. **both** ratified nonterminal outcomes of the hybrid first failure still present
   (`reassigned` *and* `requeued`);
3. the immediate placement selects by **identity** — `exact_stripe_id=stripe_id` present,
   `stripe_prefix=stripe_id` absent.

### 4.1 `tests/test_s172_admission_liveness.py` — `G-FORBIDDEN-ABSENT` (was `:808-811`)

| assertion | disposition |
|---|---|
| byte identity vs HEAD: `handle_stripe_failure` | **PRESERVED** (`:813-816`) |
| byte identity vs HEAD: `_pick_other_worker` | **PRESERVED** (`:813-816`) |
| byte identity vs HEAD: `process_lease_expiry` | **PRESERVED** (`:813-816`) |
| byte identity vs HEAD: `_handle_stripe_failure_locked` | **CHANGED → superseding invariant** (`:817-880`) |
| §2 `expected_workers` bound once in the preamble | **PRESERVED**, untouched |
| §3 `worker_pool_size` semantics | **PRESERVED**, untouched |
| §4 `serve_timeout` default still `None` (both anchors) | **PRESERVED**, untouched |
| the gate's returned detail string | **UPDATED** to state 3/4 byte-identical + the invariant, so a green does not overstate what was checked (`:966-970`) |

**Why the old form no longer fits:** F1 changed how a hybrid first failure is *placed* (deferred
placement — the requeue no longer claims), and R1 Blocker A changed the retry's *selector*. Both
live inside that one function; neither is a terminal decision, and a byte comparison cannot express
the difference.

### 4.2 `tests/test_s172_staging_backpressure.py` — `G-MATRIX-DIFF-a` (was `:1583-1589`)

| assertion | disposition |
|---|---|
| 7 pre-change `_on_staging_failed` call sites | **PRESERVED** |
| exactly 6 at `_AMENDMENT_BASELINE_REV`, 6 after | **PRESERVED** |
| exactly one call site removed, and it is the deferred-overflow one | **PRESERVED** |
| no out-of-scope caller modified; survivors == baseline | **PRESERVED** |
| AST identity vs both baselines: `_on_staging_failed` | **PRESERVED** (`:1608-1615`) |
| AST identity vs both baselines: `handle_stripe_failure` | **PRESERVED** (`:1608-1615`) |
| AST identity vs both baselines: `_handle_stripe_failure_locked` | **CHANGED → superseding invariant**, `_assert_matrix_invariant(live)` (`:1617-1641`, helper `:1644-1677`) |

**`gate_matrix_diff_behavioural` — nothing removed, one row ADDED** (`:1736-1781`): hybrid attempt 0
with an alternate that **exists but is compute-busy** → `requeued`, one retry budget consumed,
`phase_degraded`, trial **running**, stripe back in the backlog with `lease_expires_at IS NULL`, and
then genuinely placed on that alternate once it frees its compute slot, with a fresh lease. All
eight pre-existing rows (both `non_retryable`, both `constant_phase`, both `reassigned`,
`no_alternate_worker`, lease-expiry constant) are **unchanged**, as is
`assert len(_OUT_OF_SCOPE_CALLERS) == 6`.

This row is what certifies Beta's *"OR pending/requeued if alternate capacity is temporarily busy"*
behaviourally rather than structurally.

### 4.3 `G-LEASE` — updated to the ratified deferred-placement semantics (`:1395-1421`)

| assertion | disposition |
|---|---|
| the PAUSED worker's lease does **not** enter the matrix | **PRESERVED** |
| the exemption is NARROW — an UNPAUSED worker's genuine silence **does** expire | **PRESERVED** |
| `len(out) == 1` | **PRESERVED** |
| A's stripe untouched: `claimed`, attempt 0, `claimed_by` A, not degraded | **PRESERVED** |
| B: `current_attempt == 1 and phase_degraded` | **PRESERVED** |
| trial still `running` | **PRESERVED** |
| `out[0]["action"] == "reassigned"` | **CHANGED → `"requeued"`** + `worker_id is None` |
| — | **ADDED**: B's stripe is `pending`, `claimed_by` still the prior claimer, `lease_expires_at IS NULL` |

The old expectation is the pre-F1 model: the retry claimed an alternate whether or not it was
compute-idle. Here the only alternate (`hostA`) is compute-busy with its own stripe and the prior
claimer may not take its own failure back, so the ratified outcome is `requeued`. **What the gate
measures — the pause exemption and its narrowness — is unchanged**; the added assertions make
"queued, never lost" explicit rather than implied.

`gate_lease_exemption_mutant` was **not touched**.

### 4.4 Mutation evidence for the superseding invariant

Run against the live source (`_assert_matrix_invariant`):

```
live source: invariant HOLDS
MUTANT RED [terminal decision renamed/reordered]         -> "the four terminal decisions changed or were reordered"
MUTANT RED [requeued outcome removed]                    -> busy alternate becoming terminal is detected
MUTANT RED [prefix-as-exact reintroduced]                -> "no longer selects by stripe identity"
```

### 4.5 ⚠ A THIRD SITE BETA DID NOT ENUMERATE — `tests/test_s172_admission_binding.py` B7

**Reported, not worked around.** Beta's §C named two guards. A **third** carries the identical
byte-identity assumption: `tests/test_s172_admission_binding.py:531-534` (gate **B7**). It reds for
the same reason and would have left an F1/F2-chargeable red in the package.

**Baseline differential** (`git worktree add … eecfff7`, same venv, same host — worktree removed
afterwards):

| tree | tally | failing gates |
|---|---|---|
| `eecfff7` (pre-amendment) | **11/20** | B1, B2, B5, B6, C1, C2, C3, C4, C5 |
| patched tree, before this hunk | **10/20** | the same nine **+ B7** |
| patched tree, after this hunk | **11/20** | the same nine — **zero differential** |

The nine common reds are **not chargeable to anything in this amendment**: they are the local
execution set resolving **one** GPU where the gates expect two (`gpu_count: 1`, the `f255912`
correction). They fail identically at `eecfff7`. Beta flagged C5 as a baseline failure; the
differential shows the whole B/C block is in the same condition.

B7 was given **exactly** the treatment the two named guards received — same invariant, byte identity
retained for the other three functions, no other assertion touched — and the hunk is headed
`⚠ ALPHA JUDGMENT CALL — FLAGGED FOR BETA` (`:537-551`) with an explicit instruction: **if Beta
intended the supersession to stop at the two enumerated sites, revert THIS hunk only.**

---

## 5. ORDER CONFIRMATION — A AND B WERE FIXED BEFORE ANY GUARD WAS RE-BASELINED

Beta's constraint: *"Do not simply update the old byte hash/baseline to the current submitted
source. The current source contains Blockers A and B. That would certify the defects."*

Sequence as executed, and it is verifiable from the run logs:

1. baseline sweep captured with the guards **untouched** (`/tmp/baseline_*.log`);
2. **Blocker A** fixed in production (`pending_stripes`, `schedule_pending_stripes`, the retry call
   site) — F1/F2 suite re-run **13/13**;
3. **Blocker B** fixed in production (`abort_trial`, `_terminal_from_trial_row`);
4. the two new gates written and run — **16/16**;
5. **red-first proved against genuine pre-fix source** for both new gates;
6. **only then** were the guards superseded — so what every re-baselined guard certifies is the
   corrected function. The invariant additionally asserts `exact_stripe_id=stripe_id` is **present**
   and `stripe_prefix=stripe_id` is **absent**, which means a guard re-pinned to defective source
   would have failed on its own terms.

No hash or baseline was "updated to current". Nothing was pinned to a byte image at all — the new
form pins **semantics**, which is what survives a legitimate change and reds an illegitimate one.

---

## 6. THE CORRECTED ONE-ACTIVE WORDING

**Alpha's submission was wrong.** It said the existing-active read and the claim write are *"the
same SQL statement."* They are not — they are two statements, serialized by the coordinator
process's ledger `_write_lock`.

**Corrected in source** at `miner/range_miner_coordinator.py:1759-1789` (`claim_stripe`'s docstring),
which now reads, in substance:

> **Within one coordinator process, the ledger write lock serializes the existing-active check and
> the subsequent claim update.**

with the shape written out literally —

```
_write_lock
    -> SELECT an existing compute-active claim for this worker      (:1800-1810)
    -> if none: UPDATE the requested stripe                          (:1814-1830)
       (the §10 terminal-state guard is embedded in that UPDATE)
```

— and an explicit disclaimer: this does **not** claim protection against an independent external
writer or a second coordinator process; that would need database-level enforcement, which is
**outside F1 and deliberately not implemented**. It is noted as consistent with the S172
certification boundary already on record (one active trial per coordinator process).

**Corrected in this report:** the paragraph above is the operative statement. Any earlier Alpha text
claiming single-statement atomicity for the one-active invariant is withdrawn. This is a
wording/certification-boundary correction, **not** a production defect — the code's behaviour is
unchanged and correct within its real boundary.

---

## 7. RED-FIRST AND MUTATION EVIDENCE PER NEW GATE

| gate | red-first | mutation evidence |
|---|---|---|
| `G-F1-EXACT-STRIPE-COLLISION` | ✅ against genuine pre-fix source: `got {'action': 'reassigned', 'worker_id': 'host1:gpu0', …}` | ✅ dedicated mutant gate (below) |
| `G-F1-EXACT-STRIPE-COLLISION/M` | ✅ reds pre-fix (`the mutant never executed` — the intercept point does not exist) | is itself the mutant: proves execution (`executed["n"] >= 1`) **and** that the mutation reds the credited assertions |
| `G-F2-IDEMPOTENT-PARITY` | ✅ against genuine pre-fix source: replayed event carried `stripe_error / BBB / run__st0_s9 / hostZ:gpu9 / 7` | ✅ in-gate mutant: `_terminal_from_trial_row → lambda row: B_REC` (the pre-fix behaviour exactly) must red |
| `gate_matrix_diff_behavioural` new row (busy alternate → `requeued`) | ✅ red-first by construction — the pre-F1 model returned `reassigned` here | covered by `_assert_matrix_invariant` mutant *"requeued outcome removed"* |
| superseding invariant (3 sites) | n/a (replaces an existing assertion) | ✅ 3/3 mutants red — reorder, outcome removal, prefix-as-exact reintroduction |

Red-first method: the **live production file** was reverted to pre-fix semantics for both blockers
(prefix-as-exact restored; the Blocker-B reconstruction block deleted), the suite run, and the file
**restored from backup in a `finally`**. Restoration was verified by re-running the suite (16/16)
and by grepping the fix markers back into place. Script:
`…/scratchpad/redfirst.py`; log `/tmp/f1f2_redfirst.log`.

**VIR-1/VIR-2/VIR-3 controls:** every new gate proves its own execution (explicit
"the mutant was never called" failures rather than silent skips); the clean control is the unmutated
run in the same function; the fault-injection control is the mutant; every suite terminates in an
explicit `PASS`/`FAIL` sentinel; every clock is injected (`now=`), nothing sleeps.

---

## 8. FULL SEQUENTIAL VERIFICATION (Beta §13)

Run **sequentially** on VM101 under `~/venvs/torch`, one suite at a time (concurrent S172 runs flake
Part B on a free-space race). No fleet execution; no GPU; no network bind.

| suite | baseline | **final** | Δ |
|---|---|---|---|
| `tests/test_s172_f1_f2_active_lease.py` | 13/13 | **16/16 PASS** | +3 gates (2 new + 1 mutant) |
| `tests/test_s172_staging_backpressure.py` | 48/50 FAIL | **50/50 PASS** | `G-MATRIX-DIFF-a`, `G-LEASE` green |
| `tests/test_s172_staging_partb.py` | 24/24 | **24/24 PASS** | — |
| `tests/test_s172_elapsed_roundtrip.py` | 6/6 | **6/6 PASS** | — |
| `tests/test_s172_phase5_d3_5_finalizer.py` | 60/60 | **60/60 PASS** | — |
| `tests/test_s172_phase4_coordinator.py` | 62/63 FAIL | **63/63 PASS** | Gate 22 green — see below |
| `tests/test_s172_admission_liveness.py` | 15/16 FAIL | **16/16 PASS** | `G-FORBIDDEN-ABSENT` green |

**All F1/F2-chargeable reds are GREEN.** Logs: `/tmp/final_test_s172_*.log`.

**Gate 22** was closed the way this file has closed it for every prior deliverable: the new harness
and the one edited suite were **registered in the whitelist with a rationale**
(`tests/test_s172_phase4_coordinator.py`, appended block, nothing earlier rewritten) —
`tests/test_s172_f1_f2_active_lease.py` and `tests/test_s172_staging_backpressure.py`.
`tests/test_s172_admission_liveness.py` and `tests/test_s172_admission_binding.py` were already
whitelisted. This is registration, not widening: the gate still fails on any unlisted `.py`.

**`tests/test_s172_admission_binding.py` — baseline differential, kept in the package and NOT
absorbed:** `eecfff7` **11/20** vs patched **11/20**, identical failing set (B1, B2, B5, B6, C1–C5).
See §4.5. Those nine are environment reds (localhost set resolves 1 GPU, not 2) and predate this
amendment entirely.

**Not run, and why:** the phase-5 D0/D1/D2/D4/D5/D6 suites — no file they cover was modified, and
`assign_stripes`' public signature was deliberately left unchanged precisely so they are untouched
(§1.3). `tests/gate_s172_prod_shape.py` requires a live fleet and is out of bounds under the hold.

---

## 9. FILES CHANGED

| file | change |
|---|---|
| `miner/range_miner_coordinator.py` | **Blocker A** — `pending_stripes` split into `stage_prefix` / keyword-only `exact_stripe_id` with mutual exclusion (`:2084-2119`); `schedule_pending_stripes` selectors made keyword-only and explicit (`:2967-2978`, `:3018-3019`); hybrid retry switched to `exact_stripe_id=stripe_id` (`:5564`); `assign_stripes` forwards `stage_prefix=` (`:2953`) and its docstring bounds the parameter (`:2895-2900`); serve loop forwards `stage_prefix=` (`:6962`). **Blocker B** — durable terminal reconstruction on both non-first paths (`:5863-5886`) and `_terminal_from_trial_row` (`:5925-5956`). **Item D** — `claim_stripe` one-active wording corrected (`:1759-1789`). |
| `tests/test_s172_f1_f2_active_lease.py` | **NEW gates** `G-F1-EXACT-STRIPE-COLLISION`, its mutant, and `G-F2-IDEMPOTENT-PARITY`; six scheduler call sites moved to `stage_prefix=`. 13/13 → **16/16**. |
| `tests/test_s172_staging_backpressure.py` | `G-LEASE` updated to ratified deferred placement; `G-MATRIX-DIFF-a` byte identity superseded for one method via new `_assert_matrix_invariant`; one behavioural row added for the busy-alternate requeue; `ST_PENDING` imported. |
| `tests/test_s172_admission_liveness.py` | `G-FORBIDDEN-ABSENT` byte identity superseded for one method; detail string updated. |
| `tests/test_s172_admission_binding.py` | B7 given the identical treatment — **flagged as an Alpha judgment call** (§4.5). |
| `tests/test_s172_phase4_coordinator.py` | Gate-22 whitelist: appended registration block for the new harness and the edited backpressure suite. |
| `docs/CLAUDE_CODE_REPORT_F1_F2_R1.md` | this report. |

**No production file other than `miner/range_miner_coordinator.py` was touched.** No kernel, sampler,
threshold, protocol, dataset-authority, execution-set, PWC or ZMQ path was reached.

---

## 10. DISAGREEMENTS AND JUDGMENT CALLS, REPORTED NOT WORKED AROUND

1. **`assign_stripes` parameter not renamed** (§1.3). Its `stripe_prefix` is an ID-construction
   prefix, not a selector, and renaming it would edit 13 call sites in 10 committed suites, most
   outside the verification list. Flagged; Beta's call.
2. **A third supersession site** — `admission_binding` B7 (§4.5) — was superseded so no
   F1/F2-chargeable red remains. Beta enumerated two sites; this is disclosed at the hunk and
   trivially revertible in isolation.
3. **No remedy proposed for F-1's underlying coupling beyond the amendment itself.** Unchanged from
   the forensics submission; not reopened here.
4. **`worker_pool_size` untouched.** The value `25` was not applied anywhere, in code, config or
   test.

## 11. STATUS

Four items complete. All F1/F2-chargeable reds green across the §13 sequential verification.
**Commit is NOT authorized and nothing was committed, pushed, or launched.** Gate 12 remains HELD.
