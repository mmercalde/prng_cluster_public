# SESSION CHANGELOG — 2026-08-14 — S181 — ATTEMPT-6 IMPLEMENTATION R1 **AND R2**

> **TWO Beta rounds, one file, and that is deliberate.** R1 and R2 both landed on 2026-08-14 and
> this changelog was never committed between them, so splitting it would leave a reader joining two
> uncommitted artifacts for one day's work. **§8 is the R2 round**; §§1-7 are R1 and are unchanged
> except where R2 explicitly supersedes them (noted inline). No session is unlogged.

**Host:** VM101 `zeus-ubuntu` (192.168.3.177), user `michael`, `~/venvs/torch`.
**Base:** HEAD `2b0d2dc` throughout — unchanged at START and END. Working tree at session start was
S180's end state.
**Brief:** `~/dashboard_work/CCODE_BRIEF_ATTEMPT6_IMPL_R1_v1_0.md` (v1.0) — Team Beta RRR
2026-08-13: **RETURN FOR NARROW IMPLEMENTATION R1.** Two behavioural corrections, one record-only
reconciliation, one arm audit.
**The R3 architecture remains CERTIFIED and was NOT reopened.**
**Full report:** `~/dashboard_work/ATTEMPT6_IMPLEMENTATION.md` §R1.
**NOTHING COMMITTED, NOTHING PUSHED, ATTEMPT 6 NOT LAUNCHED.**

---

## 1. R1-A — the sentinel gate accepts on a SAME-RECORD conjunction

**`scripts/gate12_sentinel_gate.py` (356 → 406 lines).**

`probe_sentinel()` ran **two independent counts** — `grep -c SESSION_SENTINEL` and
`grep -c <nonce>` — stored the *nonce* count as `count`, and `evaluate()` passed the worker whenever
`count` was nonzero. `sentinel_lines` never entered acceptance. The gate proved *"a SESSION_SENTINEL
exists somewhere AND the current nonce exists somewhere"* where the contract requires *"a
SESSION_SENTINEL **carrying** the current nonce."*

**Reachable on an ordinary run, verified in live source.** The worker enters the release barrier
immediately after emitting the sentinel, and `await_session_release` emits `SESSION_RELEASE_WAIT`
carrying `run_nonce` (`miner/range_miner_worker.py:1498-1500`). So a log holding an OLD run's
sentinel plus THIS run's release-wait satisfied both counts with this run's sentinel never observed.
The existing stale-nonce arm does not catch it — it presents an old sentinel with **no** current
nonce, so the old predicate refused it for the right reason by accident.

**Fixed at the probe predicate only.** One pipeline:
`grep 'SESSION_SENTINEL' <log> | grep -c '<nonce>'`. The sentinel/barrier architecture and the
launch order are untouched; `gate12_launch.sh` and `scripts/launch_fleet_manual.sh` are
byte-unchanged (they consume the exit status, whose contract is unchanged). A diagnostic-only
`sentinel_lines_any_nonce` lets an operator tell the two refusals apart — sentinels present but
stale, versus none at all — and **cannot** reach acceptance.

**Three new arms** (RXP-3 8 → 11 arms), each naming the wrong input that reds it:
- **RXP-3/9** stale sentinel + old nonce **plus** current release-wait + current nonce → **REFUSE**;
- **RXP-3/10** no sentinel at all, only a current-nonce release-wait → **REFUSE**;
- **RXP-3/11** AST: `evaluate()` never reads the diagnostic, exactly one conjunctive pipeline
  exists, and the acceptance number is bound to `out[0]` (a swap would restore the defect while
  every string still looked right).

Both behavioural arms build their logs by **running the real worker emitters**, and the pre-R1-A
predicate is reconstructed from live source by mutation and **detected by both**.

## 2. R1-B — P-1 is FIRST-FRAME REGISTER PRIORITY (D2 ruled)

**`miner/range_miner_coordinator.py` (+1,472 → +1,508, deletions unchanged at −72).**

Beta ruled the question Alpha declined to resolve: **the architectural invariant controls over the
defective proxy wording.** The admission route may be used at most once per connection and only for
the connection's first decoded application frame when that frame is REGISTER.

- `envelopes_delivered` **deleted** from `_conn_reader_loop`.
- `first_frame`, snapshot-then-cleared **unconditionally at the decode**, before any branch can act
  on the frame. Consuming eligibility at the decode rather than in the admission branch is the
  load-bearing choice: a clear inside the branch would leave the flag set for a
  `result → REGISTER` connection.
- **Not keyed on `worker_by_sock`**, per Beta's explicit instruction — the serve loop mutates it and
  a reconnect could race the eviction.

**Two new arms** (FAIR-6 11 → 13 arms):
- **FAIR-6/13** `REGISTER → REGISTER` back-to-back with **no intervening result**: #1 admission, #2
  `inbound`, existing rebind/idempotent semantics unchanged. Decided on `admission.put_order`, which
  records every put, so the route's use count is measured rather than inferred from a queue depth at
  one instant. **Mutant (first-frame term removed) APPLIED, EXECUTED, DETECTED.**
- **FAIR-6/14** structural, over live source: armed once outside the loop · cleared once as a
  **direct statement of the loop body** · never re-armed · the admission test mentions neither the
  delivery counter nor `worker_by_sock` · `envelopes_delivered` absent from the reader entirely.

**Why FAIR-6/7 could not detect D2**, confirmed against its source: its `REGISTER → result →
REGISTER` sequence necessarily increments the delivery counter before the second REGISTER. Arm 13 is
arm 7 with that result deleted.

**Scope: R1-B moved exactly one definition digest** — `_conn_reader_loop`,
`f0b23825e798cf20 → f87f8433961d823b`. All scope-proof counts unchanged (coordinator 219/10/19/0,
worker 68/2/3/0); every no-touch surface still digest-IDENTICAL; the §4.3 bounded-admission AST
subtree still `6326ebb4f31561a8`.

## 3. R1-C — the arm audit, and a fourth instance of the failure class

Beta's closing note: R1-A is the third instance of *a gate that passes on a fact it does not
actually check*, and the remaining arms were to be audited for the same shape. **Four findings, all
corrected.**

1. **FAIR-3/4 — THE FOURTH INSTANCE.** Asserted `loop_now_age_max >= 0.0` and `<= bound`. **An
   instrument frozen at its constructor value satisfies both**: `note_loop_now_age` returns early
   when `_last_top` is unset, so an unwired instrument reports a permanent `0.0` and the arm went
   green while measuring nothing. Now asserts observation first — strictly positive age **and** the
   wall label the instrument stamps only when it updates — with a **reachability control** on a bare
   `ServeLoopTiming` proving the frozen state is real, so the new assertions have a demonstrated
   failure mode.
2. **RXP-1/10** — `"Error" in str(exc_class)` is satisfied by any string ending in "Error". The name
   must now **resolve** to a real `OSError` subclass.
3. **FAIR-6/1 and /2** — `"hostA:gpu0" in maps[...]` is satisfied by a key bound to `None`. Now
   value assertions: bound to that connection's own framed socket, and on reconnect to the new one.
4. **FAIR-1/2 arm 5** — `src.replace(" ", " ")`, a no-op reading as whitespace normalisation.
   Removed.

**Checked and found sound:** RXP-1/2, /3, /13 · RXP-2/9 · FAIR-7/1 · FAIR-3/2 · FAIR-6/10.
**Reported and left alone:** FAIR-1/2 arm 5's first half re-implements the production timeout
expression rather than executing it — that IS the shape, but the paired AST assertion over live
source closes it.

## 4. R1-D — the numeric reconciliation

Beta's record-only correction names *"fifteen"* and the stale figures `1531`, `372`, `4183`.
**Measured on the artifacts as they stood when the brief arrived, before any R1 edit: those strings
occur in none of the three documents on this host**, only one copy of the report exists, and all
three already read **sixteen · +1,472/−72 · 4,220 · 356** — with the sixteen confirmed against
`git diff --numstat` (`tests/test_s172_staging_backpressure.py` is `+16/−16`). Beta was reading an
earlier revision of the uploaded copy. *(They appear now only where this reconciliation quotes
them.)*

**The reconciliation genuinely due is that those figures are now stale because of R1**, and every
one has been re-measured at final state and updated in the report, the cover and here.

## 5. Evidence at final state

| | |
|---|---|
| ten-gate §11 battery | **76/76 green** (`attempt6_logs/r1_gates.log`) — 71 + five new arms |
| regression battery, 13 suites | **verdict-set IDENTICAL to the S180 certification run in every one** (`attempt6_logs/r1_*.log` vs `CERT_*.log`) |
| phase-4 | **62/63** — the one red is **Gate 22**, naming the two untracked new `.py` files; allowlist NOT widened; self-clears on a clean committed tree; sixth occurrence |
| `admission_binding` | **11/20 PRE-EXISTING**; the S180 worktree differential is not re-run or re-claimed, only that R1 moved nothing |
| AST scope proof | regenerated, `NO-TOUCH VERDICT: PASS` (`~/dashboard_work/ATTEMPT6_SCOPE_PROOF.txt`) |
| mutants | **five**, each APPLIED, EXECUTED and DETECTED (three from S180, two new at R1) |

The battery and the regression suites were run **sequentially** — concurrent S172 runs flake Part
B's free-space arm and read like a regression from this diff — and this changelog was written after
the last run.

## 6. Nothing forbidden touched

D6 · F1 lease-origin · expiry/retry semantics · `worker_admission_timeout` (NOT widened, enforcement
block still AST-subtree-identical) · `D`/`D_adm`/`A_max`/`S` · queue and staging bounds ·
emergency-terminal policy · **sentinel/barrier launch order (NOT redesigned)** · window-anchor /
generator-phase work (not merged).

**`ATTEMPT6_REMEDIATION_DESIGN.md` was deliberately NOT edited.** Its §8.6.3 condition (2) and the
arm-2 row of its §11.6 table still carry the "zero-delivery" wording D2 ruled defective. R3 is
Beta-certified and this brief does not reopen it, so the ruling is recorded in the code at the
mechanism, in the suite's arm-2 docstring, and in the report — and correcting the design text is
flagged to Beta as a separate pass.

**Attempt-5's initiating reader cause remains UNRESOLVED**, and is claimed nowhere in the code, the
gates, the report or this changelog.

## 7. Files

```
 M miner/range_miner_coordinator.py        R1-B (one definition)
 M scripts/gate12_sentinel_gate.py         R1-A (untracked; NEW this cycle)
 M tests/test_s172_attempt6_remediation.py five new arms + four audit tightenings
                                           (untracked; NEW this cycle)
 M docs/TB_SUBMISSION_ATTEMPT6_IMPLEMENTATION.md   cover, rewritten at R1 final state
 A docs/SESSION_CHANGELOG_20260814_S181.md this file
```

**Byte-unchanged by R1:** `miner/range_miner_worker.py` · `gate12_launch.sh` ·
`scripts/launch_fleet_manual.sh` · `tests/test_s172_staging_backpressure.py`.

*(R1's "next" step was superseded the same day by Beta's R2 return — see §8.)*

---

# 8. R2 — ONE NARROW CORRECTION (Beta RRR 2026-08-14)

**Brief:** `~/dashboard_work/CCODE_BRIEF_ATTEMPT6_IMPL_R2_v1_0.md` (v1.0) — **RETURN FOR ONE NARROW
R2.** *"The substantive attempt-6 remediation is ACCEPTED. R1-A, R1-B and D2 are CLOSED. The
coordinator/worker implementation is ACCEPTED and FROZEN."*
**R2 touches `scripts/gate12_sentinel_gate.py` and `tests/test_s172_attempt6_remediation.py` only.**
**NOTHING COMMITTED, NOTHING PUSHED, ATTEMPT 6 NOT LAUNCHED.** HEAD still `2b0d2dc`.

## 8.1 R2-A — an ssh transport failure is UNAVAILABLE, not ERROR

**Beta's finding, confirmed in live source.** `probe_sentinel()` ran `_run(["ssh", ...])` and went
straight to `proc.stdout`; **`proc.returncode` was never examined.** An ordinary connectivity or
`BatchMode` auth failure is not an exception from `subprocess.run` — it returns a **completed**
process with a nonzero ssh status, empty stdout and diagnostic stderr — so it fell through the
two-line check to `ERROR: unparseable_probe_output:[]`: *"the probe ran and its output could not be
classified"* said about a probe that never ran. **Both outcomes refuse, so no safety consequence;
the consequence is evidentiary**, and it is the same class as R1-A. RXP-3/3 declared
*"ssh fails / file unreadable → UNAVAILABLE"* while exercising only the local unreadable file.
**Beta's fifth instance of the class, found inside the arm meant to close it.**

**The correction.** A named module constant `SSH_TRANSPORT_FAILURE_STATUS = 255`; the check runs
**before any stdout is read** (if ssh failed, what arrived is not this probe's output) and is
**gated to the remote branch** (the local branch runs `bash -c`, where 255 carries no transport
meaning). Reason string keeps the certified `ssh_exit_<rc>` token so one grep finds this and the GPU
probe.

**Why 255 and not "any nonzero", stated as what it is.** ssh returns 255 for its own failure, and
**this gate reserves that value as its remote-transport classification under the current probe
script** — a decision about this gate, **not a protocol-level claim that no remote command could
produce 255**. ssh passes a remote command's status through unchanged, so one that exited 255 would
be reported here as a transport failure; the reservation is sound only because the script this gate
sends cannot produce it, and **if that script changes the rule must be revisited** — recorded beside
the constant rather than left implicit. The converse rule is what Beta forbade: *a remote `grep`
returning 1 for "no match" is not a transport failure*. **The certified GPU probe uses the opposite
rule deliberately** — `preflight_check.py:512` treats any nonzero as UNAVAILABLE, and it can, because
`_build_gpu_probe_script` ends every internal failure branch in an explicit `exit 0`, designing the
ambiguity out. This gate's script carries no such guarantee (its taken branch merely happens to end
in `head -1`), so it reserves one value instead of every nonzero one.

**Two new arms** (RXP-3 11 → 13), each naming the wrong input that reds it:
- **RXP-3/12** — a `CompletedProcess` with status 255 and real `No route to host` stderr →
  UNAVAILABLE, `count is None`, render carries UNAVAILABLE and is never count-shaped, `evaluate`
  refuses, the reason names `ssh_exit_255`, and ssh's own diagnostic reaches the operator.
  **Mutant** (returncode check neutralised) reproduces `ERROR: unparseable_probe_output:[]` exactly
  — **DETECTED**.
- **RXP-3/13** — the neighbouring control, so UNAVAILABLE and ERROR cannot collapse: status 0 +
  malformed → ERROR · status 1 + malformed → ERROR · **status 1 + well-formed → OK, count read**.
  The third case is Beta's sentence made executable. **Mutant** (`!= 0`, the rule Beta forbade)
  reports `UNAVAILABLE ssh_exit_1` for a legitimate remote `grep` status — **DETECTED**.

**The seam is `SG._run`**, the gate's own subprocess entry point, so the **real** `probe_sentinel`
body runs against the exact shape ssh returns with no fleet and no network. The stub is installed in
the namespace the function under test resolves against; the first draft merged globals into a copy,
which would have let both mutants escape the shim and survive while appearing detected — the A8-B2
lesson, caught before it shipped.

## 8.2 R2-B — the re-audit Beta ordered, and one more instance

Every arm re-read for **that specific shape**: a declared SET of scenarios with only one member
driven (as distinct from one scenario with several asserted properties, which is sound).

1. **RXP-3/3** — Beta's. Declaration corrected to what it drives; points at 12–13 for the ssh half.
   No assertion changed.
2. **RXP-1/2 — a further instance.** Declares *"a BOUND socket exiting via **E2-E5 or E7**"* and
   drives **E5 alone**. Declaration narrowed; **no assertion changed**. Widening is put to Beta:
   arm 2 tests the *transport* (reader → fifth tuple field → `_drop_conn` → `WORKER_DISCONNECTED`),
   which **carries** the reason rather than switching on it, and RXP-1/1 separately drives all eight
   classes — so the composition is covered across two arms. Widening is a restructure of certified
   machinery, not a loop, because the injections tear their benches down before `_drop_conn` can be
   called with live maps.
3. **A third finding, in the harness.** `_mutant_red` credits **any** exception as detection, so a
   mutant *built inside its lambda* lets *MUTANT NOT APPLIED* read as *MUTANT DETECTED* — the same
   class one level up, inside the machinery that exists to prove the arms are not vacuous. All four
   Alpha-authored sites now build the mutant as a statement; both S180-era sites already did.

## 8.3 R2-C — record-only corrections

1. **Mutant count → SEVEN.** §6 of the report said *"Three mutants"* and listed only the originals
   while §4 already said five; the cover carried the same stale paragraph. Both corrected, with all
   seven named and each given applied / executed / detected evidence.
2. **D2 design text — now Beta-authorized** (R1 had deliberately left it alone and said so; that
   disposition is superseded). `ATTEMPT6_REMEDIATION_DESIGN.md` §8.6.3 condition (2), its P-REG
   proof, its Beta-constraint-discharged paragraph, the §12 summary line and the §11.6 arm-2 row all
   move from *zero envelopes delivered to `inbound`* to **"first decoded application frame on this
   connection"**, each marked a **precision correction implementing the D2 ruling**, with an
   explicit notice that the architecture is unchanged and that first-frame priority is **consumable
   exactly once per connection**. The proof's *"at most once per connection"* step is now discharged
   **by condition (2) itself** rather than inferred from a counter that did not guarantee it.
3. **Final tally → 78.**

**Explicitly declined by Beta and NOT added:** the optional runtime inequality
`conjunctive > sentinel_lines_any_nonce` that R1 had offered. It is not in the code.

## 8.4 Evidence at R2 final state

| | |
|---|---|
| ten-gate §11 battery | **78/78 green** (`attempt6_logs/r2_gates.log`) — 76 + RXP-3/12 and /13 |
| regression battery, 13 suites | see `attempt6_logs/r2_battery.txt` and `r2_*.log` |
| **scope proof** | **regenerated, and it carries a programmatic sha256 comparison of its whole digest body against the R1 reference: IDENTICAL.** `R2 SCOPE VERDICT: PASS`, `NO-TOUCH VERDICT: PASS` |
| coordinator / worker | **`+1,508/−72` and `+190/−0`, digit for digit unchanged from R1** — the specific claim Beta said it would check |
| mutants | **seven**, each APPLIED, EXECUTED and DETECTED |

## 8.5 Files touched by R2

```
 M scripts/gate12_sentinel_gate.py         R2-A (406 -> 459 lines)
 M tests/test_s172_attempt6_remediation.py arms 12-13, three declaration/harness
                                           corrections (4,677 -> 4,906 lines)
 M docs/TB_SUBMISSION_ATTEMPT6_IMPLEMENTATION.md   cover, rewritten at R2 final state
 M docs/SESSION_CHANGELOG_20260814_S181.md this §8
 M ~/dashboard_work/ATTEMPT6_IMPLEMENTATION.md     §R2 added
 M ~/dashboard_work/ATTEMPT6_REMEDIATION_DESIGN.md D2 precision correction (record-only)
 M ~/dashboard_work/ATTEMPT6_SCOPE_PROOF.txt       regenerated
```

**Byte-unchanged by R2:** `miner/range_miner_coordinator.py` · `miner/range_miner_worker.py` ·
`gate12_launch.sh` · `scripts/launch_fleet_manual.sh` ·
`tests/test_s172_staging_backpressure.py`. **`git status --porcelain` is the same eleven entries
(5 modified + 6 untracked) at R2
START and END** — R2 added no file and removed none.

## 8.6 Standing limits, unchanged

**D6 — the actual fleet ssh/sentinel plumbing remains deliberately unexercised** and must be
discharged in prelaunch: 25 ssh dispatches with the new arguments, and the probe's remote branch
against real hosts. R2 exercises that branch's **classification** at the `_run` seam; it does not
contact the fleet, and nothing here claims otherwise. **Gate 22 stays 62/63 while the two new `.py`
files are untracked; 63/63 is required after commit on a clean tree.** `admission_binding` 11/20
remains non-chargeable. **Attempt-5's initiating reader cause remains UNRESOLVED.**

**Next:** Michael reviews → **Beta certifies the implementation** → Michael commits and dual-pushes
→ clean tree (Gate 22 returns 63/63) → prelaunch battery incl. D6 → **only then** attempt 6, which
must still satisfy Beta's §21 seven-part completion authority **in one run**. Attempt 6 remains
**HELD**.
