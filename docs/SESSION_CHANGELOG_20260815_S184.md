# SESSION CHANGELOG — 2026-08-15 — S184 — D6 PARKED-FLEET DRY RUN #3 (EXECUTED — PASS)

**Host:** VM101 `zeus-ubuntu-vm` (192.168.3.177), user `michael`, `~/venvs/torch`.
**Base:** HEAD `3218718a1bdd520412626b5b51230b25d9279839` — unchanged at START and END.
Working tree `git status --porcelain` **empty at session start and empty at the end of the run**;
this changelog is the only tracked-tree delta and was written after that proof was recorded.
**Procedure:** `~/dashboard_work/D6_DRYRUN_PROCEDURE.md` (Beta ruling set, 2026-08-15).
**Evidence bundle:** `logs/prelaunch_d6_20260815_122000_*` + `logs/miner_workers/dispatch_*_gpu*.log`
(`logs/` is gitignored at `.gitignore:11`, `:37`, `:62`, which is why the tree stayed clean).

**NO PRODUCTION CODE CHANGED · NOTHING COMMITTED BY ALPHA · `--phase release` NEVER INVOKED ·
PORT 5700 NEVER BOUND · NO COORDINATOR · NO LEDGER ROW · ATTEMPT 6 REMAINS HELD.**

---

## 0. What this session was

A **run-only** session. No source file was edited, no gate was written or modified. The D6 operator
procedure was executed end to end for the third time, and **for the first time on hardware carrying
the D6-I1 and D6-I2 repairs** — S183 §10 recorded that neither had run against the fleet and that the
next D6 is what establishes them.

The two prior executions and what killed each:

* **Dry run #1 (2026-08-14)** — dispatched into three rigs carrying a `miner/range_miner_worker.py`
  last deployed 2026-08-02; **24 of 25 workers died at argparse**. Answered by the code-parity gate
  (S182).
* **Dry run #2 (2026-08-14)** — the remote command was `mkdir && cd && worker … & echo started`,
  where `&` binds to the whole `&&` list; the forked remote subshell held the SSH channel and the
  launcher was **still blocked ~325 s later** with 24 rig workers parked. Separately, the launcher
  ran under a 400 s wrapper timeout whose SIGTERM reached the **process group** containing the local
  worker, which logged `SESSION_RELEASE_ABORTED waited_s=398.793` — *the observation killed the thing
  it was observing*, and the dead local worker then presented as a fleet finding. Answered by D6-I1
  (launcher) and by Beta ruling 4 (driver rule).

Both answers held this run.

## 1. Run identity — a fresh nonce, both predecessors burned

```
run nonce  : prelaunch-d6-20260815_122000-53445
burned     : prelaunch-d6-20260814_191543-14495     (dry run #1)
             prelaunch-d6-20260814_232454-41001     (dry run #2)
```

Neither prior nonce was reused or referenced. This is not hygiene: remote `/tmp/minerlogs/gpu<N>.log`
is truncated by the shell redirect **only if the SSH dispatch lands**, so a failed dispatch leaves the
previous file in place and a reused nonce would let a stale log satisfy the sentinel gate. A fresh
nonce makes that case refuse correctly.

## 2. Driver-rule compliance (Beta ruling 4, 2026-08-15) — measured, not asserted

```
The command orchestrating or observing D6 MUST NOT have a watchdog whose timeout
signals a process group containing launched workers.
```

* The launcher was started `setsid nohup` from a wrapper carrying **no watchdog and no timeout**.
  Measured session IDs: **launcher `sid=53510`, driver `sid=53481`** — disjoint sessions, so no
  signal the driver could send was capable of reaching a worker.
* The driver is **not** the launcher's parent and therefore cannot `wait` for it. The launcher's own
  exit status and end epoch were written to `logs/prelaunch_d6_20260815_122000_fleet.rc` by the
  wrapper, and the driver **polled that file**.
* The observer was read-only with a **600 s bound whose expiry produces `INCOMPLETE OBSERVATION`,
  exit 20, and sends no signal to anything** (VIR-3: `INCOMPLETE` is terminal and is not `FAIL`). It
  did not fire — the launcher returned at 75 s — but it was built so that firing could not become a
  fleet kill.
* Cleanup ran afterwards as an explicitly **named cleanup phase** (§6), recorded as having run.

## 3. Gate results

| gate | result |
|---|---|
| GPU truth (`gate12_gpu_gate.py`) | **PASS** — 3/3 rigs OK, **8/8** each |
| Rig code-parity (`gate12_parity_gate.py`) | **PASS** — **30 MATCH · 0 MISMATCH · 0 UNAVAILABLE** |
| Fleet dispatch (`launch_fleet_manual.sh`) | **FLEET_RC=0** — *all 25 of 25 daemons dispatched*, **24/24** remote ACKs carrying a pid |
| Launcher promptness | **returned in 75 s with the fleet alive**, `RELEASE_DEADLINE=900` |
| Sentinel (`gate12_sentinel_gate.py --phase verify`) | **PASS 25/25** current-nonce session-log delivery |
| Liveness (`gate12_worker_liveness_gate.py`) | **PASS 25/25 ALIVE+PARKED** |

Parity acceptance is **content identity** — the full 64-hex SHA256 of the deployed bytes against the
canonical clean local tree — across 10 governed files × 3 rigs, with each rig reporting its own
`hostname` so three machines cannot be one machine answering thrice. `local HEAD` printed once,
tagged `[CONTEXT ONLY]`, and is not an input to the verdict. The bundle is frozen at
`logs/prelaunch_d6_20260815_122000_source_digests.json`; its absence from the attempt-3/4/5 bundles is
the sole reason that residual cannot be closed retroactively.

## 4. The two repairs, measured on hardware for the first time

**D6-I1 — remote dispatch detachment.** The launcher returned in **75 s**, which is the dispatch
stagger (25 × 3 s) and nothing else. Dry run #2, with the wait-set repair already in, was still
blocked at ~325 s. Every one of the 24 rig dispatches produced a `started pid=<digits>` ACK, and 24
per-dispatch records landed in `logs/miner_workers/dispatch_<host>_gpu<N>.log` — the only place a
remote argparse/import failure is visible before the worker log exists. The launcher also asserted
`local worker zeus-ubuntu-vm:gpu0 pid=53523 ALIVE (excluded from the wait set)` rather than assuming
it.

**D6-I2 — sentinel-correlated liveness.** All 25 identities reported `ALIVE+PARKED` with the full
join: *this run's `SESSION_SENTINEL` names a PID · that exact PID exists now · its `/proc` argv is
`range_miner_worker.py` carrying this nonce, this gpu and this run's barrier file · ≥ 1 valid
current-run `SESSION_RELEASE_WAIT` record, all identifying this same worker · zero `SESSION_RELEASED`
· zero `SESSION_RELEASE_ABORTED`.* R2's four-part correlation — `event`, `run_nonce`, `worker_id` and
`release_path ∈ the live process's --session-release-file argv` — passed on all 25, which also
confirms **HI-7's invariant on live hardware**: the worker does not normalise that path, so the exact
comparison does not refuse healthy workers. Had it normalised it, D6 #3 is exactly where that would
have been discovered.

The gate's own residual wording stands and is worth keeping in the record: the 25 identities are
probed **sequentially**, so this is a sweep and not a simultaneous snapshot, and no probe promises the
next microsecond — the 25-worker admission wall remains the runtime authority once registration
begins.

## 5. Beta's five-part evidence requirement — all measured BEFORE any cleanup

```
remote workers alive + parked :  24 / 24     <- liveness gate, NOT pgrep
local worker alive + parked   :   1 / 1      <- liveness gate, NOT pgrep
sentinel records current nonce:  25 / 25     <- sentinel gate
REGISTER before release       :   0          <- per-identity, all 25
release token before release  :   0          <- local + all three rigs
```

Per-identity, all 25 read `reg=0 rel=0 abt=0 wait=1`. Liveness and sentinel delivery were recorded as
**two separate facts**; a 25/25 sentinel result beside an unrecorded process state is exactly the
shape that produced dry run #2's dead local worker.

`pgrep -c -f` was used nowhere in acceptance. Raw `pgrep -af` output appears in the bundle as
diagnostic context only, per Beta ruling 3.

## 6. Named cleanup phase — kill without release

`--phase release` was never invoked; no release token was ever written, so the fleet was killed while
still parked at the barrier.

```
workers remaining : local none · .122 count=0 · .156 count=0 · .164 count=0
release tokens    : none — local and all three rigs
SESSION_RELEASED  : 0 on all 25 identities
coordinator       : none; no watcher_agent / window_optimizer process
port 5700         : NEVER BOUND (checked before launch, after launch, and after kill)
ledger run_id     : no new run_id — live ledger head still distributed_config_t1_7e0d020b;
                    0 rows matching this nonce or 20260815
tree              : porcelain clean, HEAD 3218718
```

The post-kill count uses `rc ≤ 1` disposition and reports **UNAVAILABLE** rather than a number for a
failed probe. `pgrep` survives here and only here: this asserts an **absence after a kill**, not a
fleet's fitness to launch, and the expectation is `count=0`, which no over-count can satisfy.

Ledger queries ran against `/home/michael/miner_staging/miner_ledger.db` — the **live** ledger. The
same-named file in the repo root is stale and answers with other runs' history.

## 7. Timeline (VM101 local time, 2026-08-15)

```
12:20:43   launch start (detached, setsid)
12:21:58   launcher returned FLEET_RC=0        elapsed 75 s, fleet alive
12:22:38   sentinel gate PASS 25/25
12:22:52   liveness gate PASS 25/25 ALIVE+PARKED
12:23:35   five-part evidence complete — 172 s elapsed against RELEASE_DEADLINE=900
~12:24     named cleanup phase: fleet killed without release
12:24:38   evidence summary written
```

The whole exercise ran in under four minutes and never approached the 900 s release deadline. That
margin is itself a D6-I1 result: in dry run #2 the launcher alone consumed more wall clock than this
entire run.

## 8. One Alpha operator error, and it was the banned construct

The **first** ad-hoc REGISTER probe accumulated remote counts with `grep -c … || echo 0`. `grep -c`
prints `0` **and exits 1** on no match, so the `||` fired and printed a **second** zero; the remote
arithmetic then failed with `syntax error in expression (error token is "0")` on all three rigs.

This is the exact pair of constructs that between them manufactured attempt 1's
`GPU_COUNT_MISMATCH: 0/8` — a count that could not distinguish *zero* from *could not measure* — and
the procedure bans it by name at STEP 7. It reappeared in a driver-side probe written on the spot.

* **It is an Alpha error, not a fleet finding**, and it produced a visible failure rather than a
  false zero, because the malformed value hit arithmetic instead of a comparison.
* No gate and no committed script is involved: `gate12_parity_gate.py`, `gate12_sentinel_gate.py`,
  `gate12_worker_liveness_gate.py` and the procedure's own STEP 7 all dispose probe failure correctly.
* The probe was rewritten per-identity without the construct, and **that** corrected sweep is the
  source of the `reg=0 rel=0 abt=0 wait=1 × 25` result in §5. The failed first attempt is retained in
  this record rather than quietly replaced.

**The lesson is narrow and worth stating:** the banned construct is banned in *committed* code, and
this session shows an ad-hoc operator probe is not outside that rule. A driver-side one-liner is
evidence-generating code.

## 9. What this discharges — and what it does not

**Discharged, on the evidence of this run, subject to Beta's ruling** (Alpha does not certify): the
25 SSH dispatches and the remote log-probe plumbing have now executed end to end; D6-I1's detachment
and D6-I2's sentinel-correlated liveness — including the R1 PARKED correction and the R2 authoritative
WAIT-record correlation — have run against real hardware and passed.

**NOT discharged, and none of it is implied by the above:**

* **This run earns NO §21 credit.** Attempt 6 starts fresh with its own new nonce, Beta's §18
  pre-launch conditions, and the §21 seven-part completion authority satisfied **in one run**.
* Attempt 6 remains **HELD**.
* This exercised the *prelaunch* layer only. No coordinator was created, no cohort was frozen, no
  trial was admitted, no stripe was assigned, nothing was published, and coverage did not advance.
* A parked fleet proves parking. It says nothing about admission, the compute lease, staging
  back-pressure or the four-stage workflow.

## 10. Files changed

```
?? docs/SESSION_CHANGELOG_20260815_S184.md      this file — the only tracked-tree delta
```

Everything else this session produced is gitignored run evidence under `logs/`:

```
logs/prelaunch_d6_20260815_122000_fleet.log            launcher output
logs/prelaunch_d6_20260815_122000_fleet.rc             FLEET_RC + end epoch
logs/prelaunch_d6_20260815_122000_launch_start.epoch   launch start epoch
logs/prelaunch_d6_20260815_122000_source_digests.json  §C parity evidence bundle
logs/prelaunch_d6_20260815_122000_sentinel.log         sentinel gate transcript
logs/prelaunch_d6_20260815_122000_liveness.log         liveness gate transcript
logs/prelaunch_d6_20260815_122000_liveness.json        liveness evidence bundle
logs/prelaunch_d6_20260815_122000_SUMMARY.txt          consolidated result
logs/miner_workers/dispatch_<host>_gpu<N>.log          24 per-dispatch records
```

**Gate 22 is unaffected by this file.** It builds `changed_py` from `git status --porcelain` filtered
to `.endswith(".py")` (`tests/test_s172_phase4_coordinator.py:1627-1628`), so an untracked `.md` does
not enter its set. No allowlist change is needed or wanted.

fallback parity: code=[not re-measured this session], env=[not re-measured this session]
