# CLAUDE CODE REPORT — F1/F2 AMENDMENT, REVISION 2 (ONE NARROW FIX)

**Date:** 2026-08-09 · **Host:** VM101 (`zeus-ubuntu`, `192.168.3.177`), repo
`~/distributed_prng_analysis`, venv `~/venvs/torch` · **Base:** `eecfff7`; F1/F2 + R1 + R2
uncommitted in the working tree.

**Authority:** Team Beta ruling *"F1/F2 AMENDMENT R1 REVIEW"* (2026-08-09), transcribed in
`docs/CLAUDE_CODE_INSTRUCTIONS_F1_F2_R2.md`.

**Constraints honoured:** no commit · no push · no launch · no fleet · no port-5700 bind · Gate 12
HELD · **`worker_pool_size = 25` NOT applied** · `TerminalRecord` unchanged · F2 schema unchanged ·
**the compatibility `reason` key retained**. Nothing on the CLOSED list was reopened — the
`assign_stripes` rename was **not** done, and the `admission_binding` B7 supersession Beta ratified
was **kept, not reverted**.

**Base verification:** `git rev-parse HEAD` = `eecfff7061a1bc7671d94cc323936bc60a543de3`; F1/F2 +
R1 intact; `tests/test_s172_f1_f2_active_lease.py` **16/16** before any edit.

---

## ⚠ READ FIRST — THE ONE THING THAT MOVED BEYOND THE AUTHORIZED FIX

Beta's §16 asked that nothing else move. **Two test files moved, and I could not deliver the
ordered fix without them.** Reported here rather than buried in §5, because it is the only judgment
call in R2.

**Root cause:** the ordered canonicalization changes *what the event's legacy `reason` key
contains*. It used to carry the **caller's short prose** (`"…: constant-phase failure"`,
`"serve_trial timeout"`, `"worker admission timeout: run '…' …"`); it now carries
`terminal.reason`. Three suites had assertions **scraping that prose** for facts that F2 moved into
structured fields years-of-design ago. They were passing on the old channel.

| suite | what broke | disposition |
|---|---|---|
| `admission_liveness` | 3 live gates (`G-ADMISSION-TIMEOUT`, `G-ADMISSION-NO-RESET-ON-CHURN`, `G-CROSS-CONSTANT`) + a **latent** VIR-2 classifier | assertions re-pointed at `terminal_class` / `run_id` |
| `admission_binding` | `C6` — expected `worker(s), 2 admitted`, canonical text uses `worker(s); 2 admitted` (**one character**) | substring corrected |
| `staging_backpressure` | 5 sites read the event's `reason`; **none broke** — verified by a green 50/50 run, not by inference | untouched |

**No information was lost from the event.** Every fact those assertions wanted is still on it —
`run_id` as its own key, the cause as `terminal_class`, the detail in `terminal_reason`. The
assertions were reading prose for things that live in fields.

**The alternative I did NOT take:** widening three production `TerminalRecord.reason` strings to
re-embed the legacy tokens (`"serve_trial timeout"`, `"worker admission timeout: run '<id>'"`,
`"constant-phase failure"`). That would have kept both suites byte-untouched — but it duplicates
structured fields back into prose, which is the pattern F2 exists to remove, and it would have left
the harness still asserting on wording. Per the standing owner rule (take the structurally stronger
mechanism), I re-pointed the assertions at the fields. **Both are single, cleanly revertible hunks
if Beta prefers the other direction.**

**The latent finding, stated precisely so it is not overstated:** `_RunOutcome.ended_by` classified
a harness-injected clock by matching the literal `"serve_trial timeout"` in the event reason. After
canonicalization that literal is gone (`TC_SERVE_TIMEOUT`'s record reads *"serve_trial exceeded its
configured serve_timeout of 20.0s …"*), so the branch would have been permanently false. **It is
latent, not active:** in every current mutant-arm run `still_hung` short-circuits first, so the
branch is not reached today and no gate was passing because of it. It is now keyed on
`TC_SERVE_TIMEOUT`. Demonstrated directly:

```
class-based classifier : harness-injected-clock
prose predicate matches: False        # the pre-R2 form, on the same post-canonicalization event
```

---

## 1. THE CANONICALIZATION SITE

`miner/range_miner_coordinator.py:5857` — `RangeMinerCoordinator.abort_trial`:

```python
5832        now = time.time() if now is None else now
5836        if terminal is None:
5837            terminal = TerminalRecord(
5838                terminal_class=TC_COORDINATOR_ERROR,
5839                reason=reason or "terminal abort with no reason supplied")
5840-5856   # [F2 §7 — R2] … (comment block)
5857        reason = terminal.reason                    # <- THE FIX
5858        self.ledger.create_trial(run_id, -1, now)
```

**Proof it precedes both event construction and the first/non-first divergence** — the three
positions in one function, by line number:

| position | `file:line` |
|---|---|
| `terminal` guaranteed non-None | `:5836-5839` |
| **`reason = terminal.reason`** | **`:5857`** |
| first/non-first divergence begins (`create_trial` → state read → `abort_event_id` → `if … "aborted"`) | `:5858-5866` |
| non-first durable reconstruction (unchanged) | `:5881-5900` |
| **event construction** | **`:5921-5923`** |

`5857 < 5858 < 5921`. There is no `return` and no branch between `:5857` and `:5858`, so the
assignment is unconditional on every path that reaches event construction — the first path included,
which is precisely what was missing.

**The non-first path is untouched.** Its `terminal = self._terminal_from_trial_row(durable)` (`:5897`),
`reason = terminal.reason` (`:5898`) and `abort_event_id = durable["abort_event_id"]` still overwrite this
line from the *winning durable record*, and remain authoritative over it. R1's Blocker-B path was
not modified.

**Resulting parity** for a caller whose prose differs from the record's:

| | first delivery | replay |
|---|---|---|
| `event_id` | X | X |
| `reason` | `terminal.reason` | `terminal.reason` |
| `terminal_reason` | `terminal.reason` | `terminal.reason` |

---

## 2. THE EXACT PRODUCTION DIFF R1 → R2

**One hunk, 18 lines added, ZERO deletions, one file.** Exhibited rather than asserted: the R1 file
was reconstructed by removing exactly this hunk from the live file and diffing
(`/tmp/r1_to_r2_production.diff`).

```
hunks: 1   added: 18   removed: 0
```

Of the 18 lines, **17 are comment** and **1 is the statement** `reason = terminal.reason`.

**Nothing else in production moved.** No other function, no other file. `TerminalRecord` unchanged;
the F2 schema unchanged; the `reason` key still present on every abort event.

**Production consumers checked, not assumed:** the real Phase-5 sink
(`miner/range_miner_npz_writer.py:1268` `AssemblingPhase5Sink.abort_trial`) does **not** read
`reason` at all, and no non-test module consumes the abort event's prose. The canonicalization has
**no production consumer impact**; the whole blast radius was test assertions.

---

## 3. THE STRENGTHENED GATE, AND THE MUTANT'S OBSERVED RED

`G-F2-IDEMPOTENT-PARITY`, `tests/test_s172_f1_f2_active_lease.py`. **No new harness was added.**

**Kept, per Beta:** the deliberately different inputs — caller `reason="AAA"` vs
`A_REC.reason = "AAA the first durable terminal transition"`. That difference is the counterexample
the gate already constructed and did not assert.

**New assertions (first call):**

```python
r1["event"]["reason"]    == A_REC.reason      # the returned first delivery
len(sink.aborts) == 1
sink.aborts[0]["reason"] == A_REC.reason      # the delivered first delivery
```

**New assertion (after the contradictory-B replay) — full payload equality:**

```python
sink.aborts[1] == sink.aborts[0]              # same identity => same payload
```

with a failure message printing both payloads. The pre-existing `event_id` equality check is
retained beneath it; the point of the new line is that `event_id` equality alone was satisfied by
two payloads differing in `reason`.

**Retained unchanged:** every contradictory-B field scan (`_assert_is_A` checks all five
`terminal_*` fields and scans `repr(event)` for `BBB` / B's class / B's stripe / B's worker), the
durable-row assertions, the no-second-ERROR-log assertion, the **race-lost arm** with its
`winner:abort` identity adoption, and the R1 mutant (`_terminal_from_trial_row → lambda: B_REC`).

**Required mutant — first-path split restored.** `RangeMinerCoordinator.abort_trial` is wrapped so
that on `first` it rewrites `out["event"]["reason"] = reason` (the caller's prose) while
`terminal_reason` stays canonical — the pre-R2 shape exactly. Because the event dict handed to the
sink *is* the dict returned to the caller, the split lands on both surfaces just as it did in
production. The mutant proves it executed (`executed["n"] == 1`, with an explicit
"the mutant never ran — this gate would be vacuous" failure) and must red all three new assertions.

**Red-first against genuine pre-R2 production source** — the `reason = terminal.reason` line was
removed from the live file, the suite run, the file restored in a `finally`:

```
[FAIL] G-F2-IDEMPOTENT-PARITY  first durable transition owns identity:
       first delivery's legacy reason is not canonical:
       'AAA' != 'AAA the first durable terminal transition'
15/16 checks green
```

Restoration verified by re-running the suite to 16/16.

---

## 4. SUITES RERUN, AND WHY

**I reran the FULL R1 matrix, not the §15 subset.** Beta's carry allowance is conditioned on the
R1→R2 production diff being strictly limited — which it is (§2) — but the change alters **the
content of an outward event field**, and Part B and elapsed were never checked for whether they
read it. Both run in ≲1 s, so verifying beat reasoning about it.

Sequential, one suite at a time, on VM101 under `~/venvs/torch`. No fleet, no GPU, no network bind.

| suite | in Beta §15? | result |
|---|---|---|
| `tests/test_s172_f1_f2_active_lease.py` | ✅ | **16/16 PASS** |
| `tests/test_s172_phase5_d3_5_finalizer.py` | ✅ | **60/60 PASS** |
| `tests/test_s172_phase4_coordinator.py` | ✅ | **63/63 PASS** |
| `tests/test_s172_admission_liveness.py` | ✅ | **16/16 PASS** |
| `tests/test_s172_staging_backpressure.py` | ✅ | **50/50 PASS** |
| `tests/test_s172_staging_partb.py` | rerun (carry declined) | **24/24 PASS** |
| `tests/test_s172_elapsed_roundtrip.py` | rerun (carry declined) | **6/6 PASS** |

Logs `/tmp/r2final_test_s172_*.log`.

**`admission_binding` — baseline-identical, not absorbed.**

| tree | tally | failing gates |
|---|---|---|
| `eecfff7` baseline (measured in R1) | 11/20 | B1, B2, B5, B6, C1, C2, C3, C4, C5 |
| R2, before the C6 correction | 10/20 | the same nine **+ C6** |
| **R2, final** | **11/20** | the same nine — **zero differential** |

The nine are the pre-existing environment reds (the localhost execution set resolves one GPU where
the gates expect two); they fail identically at `eecfff7` and are **not absorbed**. C6 was a genuine
R2 differential — a one-character punctuation expectation — and is corrected, restoring
baseline identity.

---

## 5. WHAT CHANGED, FILE BY FILE

| file | change |
|---|---|
| `miner/range_miner_coordinator.py` | **the fix** — `reason = terminal.reason` at `:5857`, before the divergence and before event construction. One hunk, 18 lines, 0 deletions. |
| `tests/test_s172_f1_f2_active_lease.py` | `G-F2-IDEMPOTENT-PARITY` strengthened: 3 first-call/payload-equality assertions + the required first-path-split mutant. Gate count unchanged (16). |
| `tests/test_s172_admission_liveness.py` | **beyond the authorized scope, see the top of this report** — `_RunOutcome.abort_events` captured; `ended_by` keyed on `TC_SERVE_TIMEOUT`; `_run_id_diagnostics` reads `terminal_class` + the event's `run_id`; `g_cross_constant` reads `terminal_class` + `CONSTANT-MODE`. Every other assertion preserved, including the eligible-count arithmetic and all VIR-2 mutant machinery. |
| `tests/test_s172_admission_binding.py` | **beyond the authorized scope** — C6's expected substring `worker(s), 2 admitted` → `worker(s); 2 admitted`. One line. |

`tests/test_s172_staging_backpressure.py` and `tests/test_s172_phase4_coordinator.py` are modified
**from R1 only** — R2 did not touch them.

---

## 6. DISAGREEMENTS AND JUDGMENT CALLS — REPORTED, NOT WORKED AROUND

1. **Two test files moved despite §16.** Forced by the ordered fix; root cause, the alternative I
   rejected, and the revert path are at the top of this report. Beta's call.
2. **The full R1 matrix was rerun** rather than carrying Part B and elapsed. The §15 condition was
   satisfied, but the change alters an outward field's content and the cost of checking was ~1 s.
3. **A latent VIR-2 classifier regression** was found by inspection and repaired. Stated as latent —
   it is not reached today because `still_hung` short-circuits first, and no gate was passing
   because of it.
4. **The prose-vs-structured-field question is now general.** Three suites asserted on a
   compatibility field's wording. Two are fixed; `staging_backpressure`'s five sites happen to still
   pass. Whether the remaining prose assertions should be re-pointed at `terminal_class` is a real
   question, and it is **not** in R2's scope — raised, not acted on.

## 7. STATUS

The ordered fix is in, gated, and red-first proved. Full matrix green; `admission_binding`
baseline-identical. **Commit is NOT authorized and nothing was committed, pushed, or launched.**
Gate 12 remains HELD.
