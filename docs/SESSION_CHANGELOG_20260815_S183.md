# SESSION CHANGELOG — 2026-08-15 — S183

**D6-I1 remote dispatch detachment + D6-I2 sentinel-correlated liveness gate.**
Beta ruled on all four items of the D6 dry run #2 forensic and authorized both repairs
(`~/dashboard_work/CCODE_BRIEF_D6_I1_I2_REPAIR_v1_0.md`). Scope A–G, nothing else.

**Base:** `3e1327bddb62f9e223ca0ad8d084e3f228007271`, porcelain empty at start.
**Not committed, not pushed. D6 not re-run. Attempt 6 not launched.**
**Full report:** `~/dashboard_work/D6_I1_I2_REPAIR.md` · **scope proof:**
`~/dashboard_work/D6_I1_I2_SCOPE_PROOF.txt`

---

## 1. Files changed

```
 M gate12_launch.sh                              new §2.6 liveness wall
 M scripts/gate12_sentinel_gate.py               PASS wording only (ONE AST scope: main)
 M scripts/launch_fleet_manual.sh                remote dispatch detachment + launch ACK
 M tests/test_s172_d6_integration_repair.py      long-lived-channel fixture, 65 -> 82
?? docs/SESSION_CHANGELOG_20260815_S183.md       this file
?? scripts/gate12_worker_liveness_gate.py        NEW
?? tests/test_s172_d6_liveness_gate.py           NEW, 59/59
```

Outside the repo: `~/dashboard_work/D6_DRYRUN_PROCEDURE.md` (scope F).

## 2. D6-I1 — the second half of the D6 dry run #2 block

The 2026-08-14 wait-set repair fixed *which* jobs the launcher waited for and left untouched *why
those jobs were not short-lived*. The remote command was

```bash
ssh … "mkdir … && cd … && nohup worker … > log 2>&1 & echo started"
```

and `&` binds to the whole `&&` list. The remote shell forked a subshell that ran the worker in its
own foreground; the subshell's stdout/stderr are the SSH channel, so ssh could not return until the
worker exited. Same cause as the rigs' `pgrep -c -f` reporting 16 for 8 workers.

Repaired: `mkdir` and `cd` synchronous with their own exit codes; **only** the worker backgrounded,
with all three streams detached; a bounded launch-settle check (`REMOTE_LAUNCH_SETTLE`, default 2 s);
then a positive ACK carrying the worker's pid, and `exit 0`. An immediate argparse/import failure is
a nonzero dispatch carrying the worker's own status. `ssh -f` and the redirect-the-group form were
both rejected by Beta because the success status would come from the `echo`, not the worker.

Two Alpha additions beyond the literal shape, declared in the report for ruling: a worker exiting 0
inside the settle window is a failure (`rc=13`), and `rc=0` without a `started pid=<digits>` ACK is a
failure.

## 3. D6-I2 — liveness as an instrument

The sentinel gate was **not** widened; its delivery contract was correct. A new gate joins the two
facts neither of which suffices alone — this run's sentinel names a PID, that PID exists now, and its
`/proc` argv is `range_miner_worker.py` carrying this nonce, this gpu and this run's barrier file,
with no `SESSION_RELEASED` / `SESSION_RELEASE_ABORTED` in the log. It derives its identities and log
paths by calling the sentinel gate's own resolver, so the two gates cannot drift apart.

Placed in `gate12_launch.sh` §2.6: immediately after the sentinel gate, before the sampler and before
the coordinator. Both must pass before any coordinator process exists.

`pgrep -c -f` is retired from acceptance authority (Beta ruling 3) — 16 for 8 workers, and blind to
identity. It survives at procedure STEP 7 only as an absence-after-kill observation.

## 4. Results

```
parity gate            30 MATCH · 0 MISMATCH · 0 UNAVAILABLE   LIVE, 3/3 rigs
D6 integration suite   82/82   (baseline 65/65, measured before any edit)
liveness gate suite    59/59   NEW (48 first delivery · +5 R1 · +6 R2)
attempt-6 suite        78/78   unchanged
GPU gate suite          9/9    unchanged
clean-tree gate suite  31/31   unchanged
AST scope proof        NO-TOUCH VERDICT: PASS — coordinator/worker/protocol and all
                       10 governed files byte-identical; sentinel gate moved exactly
                       one scope (`main`), which is the print
```

## 5. Defects found in Alpha's own work by the new arms, before review

* `grep -o 'started pid=[0-9]*'` accepts an ACK with no pid — `[0-9]*` matches the empty string.
  Found by mutant M6; fixed to `[0-9][0-9]*`.
* Three mutants (M6, M-1, M-2) were initially killed by a *different* check than the one they
  removed, proving defence in depth rather than the property claimed. Rebuilt to isolate.
* The integration suite orphaned stub workers on every arm where the launcher refuses (`kill_tree`
  no-ops once `run_launcher` has reaped the launcher). Found by the liveness suite's leak check;
  fixed with a PID-based `teardown()`, gated as HI-2b.
* The liveness suite's own leak check could not see the `sleep` children its stubs fork (~160 orphans
  per run, green). Fixed; and the first fix killed the suite's own process group — VIR-4, cleanup
  killing its reporter — so `spawn_unrelated()` now isolates the session and HI-3 reads /proc pgrp.
* The `STUB_SSH_OK` fixture (`sleep .2; exit 0`) could not fail on the promptness condition it
  covered. Replaced by an ssh shim that executes the real remote command and models the channel;
  retained only for arms where a fast ssh is what *attributes* a block to the bare `wait`.

## 6. Live measurements (rigs up, read-only)

The liveness gate's real SSH branch was exercised against the fleet with no workers running: 24 real
ssh probes, every rig identity `UNAVAILABLE (reason=log_unreadable)` → REFUSE. Correct, and the
correct kind of answer — `/tmp/minerlogs` is gone because the rigs rebooted 50 minutes earlier.

**Consequence:** that reboot destroyed the D6 dry run #2 rig logs in place. They survive only in
`~/d6_dryrun2_riglogs_20260814/`, collected before the power-down.

## 7. Operational notes

* `gate12_launch.sh` refuses at §0.4 until these files are committed (verified this session). Correct
  behaviour; the answer is to commit, never to widen the allowlist.
* The D6 procedure now launches the fleet **detached** (`setsid nohup … &`, observed by polling) per
  Beta ruling 4. In dry run #2 a 400 s wrapper timeout signalled the process group and killed the
  local worker, which then presented as a fleet finding.

## 8. R1 — prove PARKED, not barrier-configured (same session, after Beta review)

Beta **CERTIFIED D6-I1** and **APPROVED both §2.3 additions as binding**; D6-I2's architecture was
**ACCEPTED** with one narrow correction, and it is a real one.

The gate printed `ALIVE+PARKED` while observing only the sentinel, the terminal records and `/proc`.
Requirements 1-8 prove a process is *configured* to park — `--session-release-file` is a property of
how it was launched. Worker startup is `emit_startup_sentinel() -> await_session_release() ->
connect()` (`miner/range_miner_worker.py:2165-2170`), so there is a real interval in which the
sentinel exists, the argv is perfect and the barrier has not been reached. **A gate must not display
a state stronger than the one it measured** — the eighth instance of that class in this arc, and the
first caught by Beta rather than by the harness.

Corrected in `scripts/gate12_worker_liveness_gate.py` only: the same probe now ships the whole
current-nonce `SESSION_RELEASE_WAIT` records, and `ALIVE+PARKED` requires **>= 1 valid current-run
wait record, all identifying this same worker**, plus zero RELEASED and zero ABORTED. `>= 1` and not
exactly-once: one emission site (`:1499`) and one production call site (`:2168`) make it once per
process today, but that is an observation about the callers, not a contract guarantee. The residual
wording now describes the **sequential sweep** actually performed rather than implying a simultaneous
snapshot of 25 identities.

New arms: **L-15** (alive, correct, no WAIT → REFUSE — the fixture that PASSED before R1), **L-16**
(identical fixture with the wait record → PASS, the positive control), **L-17** (wait record belongs
to another identity → REFUSE), **M-7** (WAIT requirement removed → DETECTED), **HI-6** (the wait
parser checked against a real D6 #2 rig record, since R1 made that record an acceptance input).

**M-7 had to be rebuilt** — deleting one line left the next check to refuse for a neighbouring
reason, so the mutant died without testing what it removed. A `_mutant_span()` helper now excises the
whole requirement and asserts the span is gone. Third time in this arc a mutant was first killed by
the wrong check.

D6-I1 was not touched: `scripts/launch_fleet_manual.sh` is byte-identical to the certified state
(`7983b1e0ec1e8d3e…`), verified by digest.

## 9. R2 — authoritative WAIT-record correlation (same session, after Beta review)

Beta closed R1's PARKED defect and returned one ultra-narrow item. The remote extraction is
`grep 'SESSION_RELEASE_WAIT' log | grep '<nonce>'` — **text containment, not semantic equality** —
and R1's `classify()` parsed each record but read only `worker_id`. So a record whose authoritative
`run_nonce` named ANOTHER run was accepted as this run's parking proof, provided the current nonce
appeared anywhere in the line; `release_path` is exactly such a place. Ninth instance in this arc of
a check passing on a fact it does not verify.

`scripts/gate12_worker_liveness_gate.py` only: every `WAITREC` is now reparsed and must satisfy
`event == SESSION_RELEASE_WAIT`, `run_nonce == current`, `worker_id == expected`, and
`release_path ∈ the live process's --session-release-file argv values`. The fourth is what makes it a
correlation rather than a second self-consistent story. No new evidence is gathered — all four facts
were already in the probe's output.

**Declared deviation:** Beta wrote both "`>= 1` such valid record is sufficient" and "**every**
accepted WAITREC must satisfy all four". This implements the second — `>= 1` valid AND zero invalid —
which also preserves R1's all-quantifier. Stale same-worker records cannot reach the check (they
carry the old nonce throughout, so the text prefilter drops them), so a record arriving here that
fails validation is genuinely anomalous. One branch reverts it if Beta prefers the literal reading.

**The fixture was green because the gate did not correlate** — Beta's sharper catch. `spawn()` gave
the process a release file under the fixture root while `Fleet.log()` wrote `/tmp/gate12_release_…`
into the record: two different paths in every arm, passing because nothing compared them. Both now
come from one expression.

New arms: **L-18** (nonce as text only → REFUSE; PASSED before R2), **L-19** (release_path not the
live argv → REFUSE), **L-20** (exact worker + nonce + path → PASS, the control), **L-21** (the
record's own `event` field is something else → REFUSE), **M-8** (semantic correlation removed →
DETECTED, killed on L-18's fixture via `_mutant_span()`).

**HI-7** measures the invariant R2's exact comparison rests on, in the worker's own source and
read-only: `main()` passes `args.session_release_file` verbatim into `await_session_release`
(`miner/range_miner_worker.py:2168-2170`), which emits it as `release_path` (`:1498-1500`). Had the
worker normalised that path, the new check would have refused every healthy worker on the fleet and
D6 #3 would have been where anyone found out.

D6-I1 remains frozen and byte-identical: `launch_fleet_manual.sh` `7983b1e0ec1e8d3e…`,
`gate12_launch.sh` `ec9ae1e93c3a0a6a…`.

## 10. Status

D6 remains undischarged; attempt 6 remains HELD. Neither repair has run on hardware — the next D6 is
what establishes that. Awaiting Beta certification.

fallback parity: code=[not re-measured this session], env=[not re-measured this session]
