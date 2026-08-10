# CLAUDE CODE INSTRUCTIONS — F1/F2 AMENDMENT, REVISION 2 (ONE NARROW FIX)

**Host:** VM101, repo `~/distributed_prng_analysis`. F1/F2 + R1 are **uncommitted in the working
tree** at base `eecfff7`. `source ~/venvs/torch/bin/activate` before every test.

**Authority:** Team Beta ruling *"F1/F2 AMENDMENT R1 REVIEW"* (2026-08-09) — **R1 architecture
ACCEPTED, commit NOT authorized, one remaining F2 parity defect.**

**CLOSED — do not reopen, do not touch:** F1 active-lease architecture · Blocker A exact-stripe
selector · hybrid deferred placement · Blocker B durable first-writer reconstruction · the F2
structured terminal fields · the matrix semantic supersession · the `G-LEASE` update · the
`admission_binding` B7 supersession (**Beta RATIFIED Alpha's judgment call — keep it, do not
revert**) · the one-active wording · **the `assign_stripes` rename (Beta: DO NOT DO IN R2)**.

**Beta §16: "R2 should be extremely narrow… Nothing else should move."**

**Hard constraints:** no commit, no push, **no launch, no fleet, no port 5700 bind**; Gate 12 HELD;
**do not apply `worker_pool_size = 25`**. Do not change `TerminalRecord`, do not change the F2
schema, **do not remove the compatibility `reason` key.**

**Base verification:** amendment + R1 intact; `tests/test_s172_f1_f2_active_lease.py` **16/16**.

---

## THE DEFECT — two reason authorities on the FIRST abort

**Alpha verified this against live source; Beta's reading is exact.**

`reason = terminal.reason` (`:5880`) sits **inside the non-first branch only**. The event is built
at `:5904-5905` as:

```python
{"event_id": abort_event_id, "reason": reason, **terminal.as_event_fields()}
```

So with a caller reason that differs from the record's:

| | first delivery | replay |
|---|---|---|
| `event_id` | X | **X** (same) |
| `reason` | **caller's string** | `terminal.reason` |
| `terminal_reason` | `terminal.reason` | `terminal.reason` |

**Same event identity, different payload.** That is an idempotence violation at the payload level,
and it undermines what an idempotent event identifier means — even though the difference lives only
in a compatibility field.

**F2's governing rule is stronger than "the five `terminal_*` fields agree." It is: one canonical
terminal decision feeds every durable and externally visible representation.** The legacy `reason`
may remain for compatibility; **it must derive from the same canonical `TerminalRecord`.**

**Our own gate contains the counterexample and does not assert it** — see below.

## THE FIX (Beta §7)

> **Once a `TerminalRecord` exists, `terminal.reason` is the sole reason authority for the abort
> event.**

Canonicalize the legacy variable **on the first path as well**, **before event construction and
before the first/non-first divergence can matter**. Beta's shape:

```python
if terminal is None:
    terminal = TerminalRecord(
        terminal_class=TC_COORDINATOR_ERROR,
        reason=reason or "terminal abort with no reason supplied",
    )

reason = terminal.reason
```

The non-first durable reconstruction then continues to overwrite `terminal`, `reason` and
`abort_event_id` from the winning durable record **exactly as it does now** — that path is closed
and must not change.

## THE GATE (Beta §8) — strengthen `G-F2-IDEMPOTENT-PARITY`, do not add a new harness

**Keep the deliberately different inputs** — Beta calls the distinction valuable:

```
caller reason = "AAA"
A_REC.reason  = "AAA the first durable terminal transition"
```

**Add the missing FIRST-call assertions** — the ones whose absence let this through:

```
r1["event"]["reason"]      == A_REC.reason
sink.aborts[0]["reason"]   == A_REC.reason
```

**After the contradictory-B replay, assert full payload equality**, preferably:

```
sink.aborts[1] == sink.aborts[0]
```

or equivalent full event-field equality. Beta: *"That is stronger than checking only `event_id`
equality. It proves: same event identity means same event payload."*

**Retain** the existing contradictory-B field scans and the race-lost arm.

**Required mutant:** restore the first-path split — `event["reason"] = caller_reason` while leaving
`terminal_reason` canonical. **The gate must red.**

## VERIFICATION (Beta §15) — sequential

```
F1/F2 active-lease suite · D3.5 finalizer · phase-4 coordinator
S172 admission-liveness · S172 staging-backpressure
```

**Part B and elapsed may carry from R1** *if* the R1→R2 production diff is **strictly limited** to
abort-event reason canonicalization plus the test assertion. **If any additional production path
changes, rerun the full R1 matrix** — and say which you did and why.

`admission_binding` remains baseline-identical and **is not absorbed**. **No fleet execution.**

## REPORT

`docs/CLAUDE_CODE_REPORT_F1_F2_R2.md`:

1. The canonicalization site with `file:line`, and proof it precedes both event construction and
   the first/non-first divergence.
2. The exact production diff R1→R2 — **expected to be a handful of lines** — and confirmation
   nothing else moved.
3. The strengthened gate's new assertions, and the mutant's observed red.
4. Which suites were rerun and which carried, with the §15 justification.
5. Any disagreement **reported, not worked around.**
