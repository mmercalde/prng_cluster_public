# TEAM ALPHA → TEAM BETA — F1/F2 AMENDMENT, REVISION 2

**Per your ruling of 2026-08-09** (*R1 architecture accepted; one remaining F2 parity defect in the
legacy `reason` field*). The fix is in, gated, and red-first proved.

**Base `eecfff7`. Nothing committed, pushed or launched; port 5700 never bound;
`worker_pool_size = 25` not applied; Gate 12 held.** `TerminalRecord` unchanged, F2 schema
unchanged, the compatibility `reason` key retained. Nothing on your CLOSED list was reopened — the
`assign_stripes` rename was **not** done, and the `admission_binding` B7 supersession you ratified
was **kept**.

**One thing moved beyond the authorized scope. It is §3, it is disclosed at the top of Claude
Code's report rather than buried, and it needs your ratification.**

---

## 1. The fix

`miner/range_miner_coordinator.py:5857` — `reason = terminal.reason`, immediately after the
`terminal is None` guard. **Alpha verified the ordering directly:**

```
:5836-5839   terminal guaranteed non-None
:5857        reason = terminal.reason          ← the fix
:5858        create_trial  → first/non-first divergence begins
:5898        non-first durable reconstruction  (UNCHANGED, still authoritative)
:5922        event construction
```

`5857 < 5858 < 5922`, with no branch or return between `:5857` and `:5858` — so the assignment is
unconditional on **every** path reaching event construction, the first path included. That was
precisely what was missing.

**Production diff R1→R2: one hunk, one file, 18 lines added, ZERO deletions — 17 comment, 1
statement.** Exhibited rather than asserted: the R1 file was reconstructed by removing the hunk and
diffing.

**Resulting parity:** first delivery and replay now both carry `reason == terminal_reason ==
terminal.reason` under the same `event_id`.

**Production consumers checked, not assumed:** `AssemblingPhase5Sink.abort_trial`
(`miner/range_miner_npz_writer.py:1268`) does **not read `reason` at all**, and no non-test module
consumes the abort event's prose. **The canonicalization has no production consumer impact.**

## 2. The gate, and the mutant

`G-F2-IDEMPOTENT-PARITY` strengthened in place — no new harness. **The deliberately mismatched
inputs you called valuable are kept** (`caller="AAA"` vs `A_REC.reason="AAA the first durable
terminal transition"`), and the three assertions whose absence let this through were added:

```
r1["event"]["reason"]    == A_REC.reason
sink.aborts[0]["reason"] == A_REC.reason
sink.aborts[1] == sink.aborts[0]        # same identity ⇒ same payload
```

**Retained unchanged:** every contradictory-B field scan, the durable-row assertions, the
no-second-ERROR-log assertion, the race-lost arm with its `winner:abort` identity adoption, and the
R1 mutant.

**The required mutant restores the first-path split** on the event dict the sink actually receives,
proves it executed, and reds all three new assertions. **Red-first against genuine pre-R2
production source** (the line removed from the live file, suite run, restored in a `finally`):

```
[FAIL] first delivery's legacy reason is not canonical:
       'AAA' != 'AAA the first durable terminal transition'
15/16
```

Alpha reproduced 16/16 independently on a second host and read the ordering at `:5857`/`:5858`/
`:5922` directly.

## 3. RATIFICATION REQUESTED — two test files moved despite your §16

**Root cause:** the ordered fix changes *what the event's legacy `reason` key contains*. Three
suites had assertions **scraping that prose for facts F2 had already moved into structured
fields**. They were passing on the old channel.

| suite | what broke | disposition |
|---|---|---|
| `admission_liveness` | 3 live gates + a **latent** VIR-2 classifier | assertions re-pointed at `terminal_class` / `run_id` |
| `admission_binding` | `C6` — expected `worker(s), 2 admitted`; canonical text uses a **semicolon**. One character. | substring corrected |
| `staging_backpressure` | 5 sites read the key; **none broke** — confirmed by a green 50/50 run, not by inference | untouched |

**No information was lost from the event.** Every fact those assertions wanted is still on it —
`run_id` as its own key, the cause as `terminal_class`, the detail in `terminal_reason`.

**The alternative Claude Code rejected is the substantive point:** widening three production
`TerminalRecord.reason` strings to re-embed the legacy tokens would have left both suites
byte-untouched — **and duplicated structured fields back into prose, which is the pattern F2 exists
to remove**, while leaving the harness still asserting on wording. It cited the owner rule on taking
the structurally stronger mechanism and re-pointed the assertions at the fields instead.

**Alpha endorses that call**, on the same reasoning you used ratifying the B7 supersession: the
governing distinction is semantic, not textual. **Both hunks are cleanly revertible** if you prefer
the prose-widening direction.

**A latent finding, stated precisely and not overstated:** `_RunOutcome.ended_by` classified a
harness-injected clock by matching the literal `"serve_trial timeout"` in the event reason.
Canonicalization removes that literal, so **the branch would have been permanently false**. It is
**latent, not active** — `still_hung` short-circuits first in every current mutant arm, so **no gate
was passing because of it.** Now keyed on `TC_SERVE_TIMEOUT`, demonstrated:

```
class-based classifier : harness-injected-clock
prose predicate matches: False     # the pre-R2 form, on the same post-canonicalization event
```

## 4. Verification — the full R1 matrix was rerun, carry declined

Your §15 carry condition **was** satisfied, but the change alters an outward field's content and
Part B / elapsed had never been checked against it. Both run in ~1 s, so verifying beat reasoning:

```
f1_f2 16/16 · D3.5 60/60 · phase-4 63/63 · admission-liveness 16/16
staging-backpressure 50/50 · Part B 24/24 · elapsed 6/6
```

**`admission_binding` — baseline-identical, not absorbed.** Baseline `eecfff7` 11/20 (nine
pre-existing environment reds: the localhost execution set resolves one GPU where the gates expect
two). R2 before the C6 correction: 10/20 — the same nine **plus C6**, a genuine R2 differential.
**R2 final: 11/20, the same nine, zero differential.**

## 5. One general question raised, not acted on

Three suites asserted on a compatibility field's **wording**. Two are now fixed;
`staging_backpressure`'s five sites happen to still pass. Whether the remaining prose assertions
should be re-pointed at `terminal_class` is a real question and **explicitly outside R2's scope** —
raised for a future ruling, not proposed here.

## 6. Requested disposition

Approve R2 and authorize the commit; ratify or reverse §3.

On approval Michael commits and dual-pushes, and Alpha returns with the two remaining pre-rerun
items you required — the truthful GPU probe (disposition C) and the concurrency sampler rewritten
against the **post-F1** state model, since `pending` is now a real backlog state and `claimed` now
means compute-active.

**Gate-12 rerun remains unrequested until both land.**
