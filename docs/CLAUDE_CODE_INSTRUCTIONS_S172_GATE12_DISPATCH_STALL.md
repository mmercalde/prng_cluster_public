# CLAUDE CODE INSTRUCTIONS — S172 GATE-12 DISPATCH STALL (READ-ONLY DIAGNOSIS)

**Host:** VM101 (`zeus-ubuntu-vm`, 192.168.3.177), repo `~/distributed_prng_analysis`.
`source ~/venvs/torch/bin/activate` before any python.

## ABSOLUTE CONSTRAINT — YOU DO NOT LAUNCH ANYTHING

**Pipeline runs are MICHAEL-INITIATED ONLY (CLAUDE.md rule 3, harness-enforced, skill §7).**

You must NOT: start `watcher_agent.py`, start `window_optimizer.py`, run
`scripts/launch_fleet_manual.sh`, start any `range_miner_worker`, bind port 5700, or run
`tests/gate_s172_prod_shape.py`. You must NOT commit, push, or edit production code in this
task. If your diagnosis seems to require a run, **STOP and report what run you would need and
why** — Michael decides and Michael launches.

**Permitted:** reading files, reading the SQLite ledger read-only, `git log`/`show`,
read-only SSH probes to the rigs (`rocm-smi`, `tail`, `cat`, `pgrep`), and writing your report
to `docs/`. Everything you touch is read-only except your report.

---

## What happened, 2026-08-07 (the run you are diagnosing)

The S172-BP staging remediation is committed (`27ae7a9`) and Beta-approved. Gate 12 — the
first 4-stripe/25-daemon production-shape trial — was authorized and attempted five times
tonight. Four attempts died on operator/config causes, now all fixed and understood:

1. **Both backend flags set** — the manifest defaults `use_persistent_workers: true`, and
   `--params` added `use_range_miner: true`; `window_optimizer.py` refuses
   (`only one of --use-persistent-workers, --use-zmq-sqlite, --use-range-miner may be set`).
   Fix: pass `"use_persistent_workers": false` explicitly alongside `"use_range_miner": true`.
2. **A stale `optimal_window_config.json` from 2026-05-11** made WATCHER score a CRASHED
   Step 1 as `1.0000 PASSED` via `file_exists` (the known §"standing cautions" defect, fifth
   occurrence). Moved aside; WATCHER then correctly escalated with confidence 0.00.
3. **A leftover halt state** from the escalation blocked the next run
   (`PIPELINE HALTED - Cannot start`). Fix: `--clear-halt --run-pipeline`.
4. **Launch ORDER.** RANGE-MINER workers call `connect()` ONCE with no retry
   (`range_miner_worker.py:1232`); a worker started before the coordinator binds dies on
   `ConnectionRefusedError [Errno 111]`. The coordinator must be LISTENING on 5700 before the
   fleet is launched. (Also: `launch_fleet_manual.sh` staggers 3s/worker ≈ 75s to dispatch 25,
   so a short run can finish before the fleet forms.)

**The fifth attempt got everything right and produced the state you are diagnosing:**

- `[EXEC-SET] FROZEN for this run: execution set d89834f1bf26 backend=miner profile=proxmox
  full nodes=['localhost','rrig6600','rrig6600b','rrig6600c'] gpus=25 remote=True admission=8`
- PreflightChecker re-pointed the rigs (.120→.122, .154→.156, .162→.164, profile=proxmox);
  `[PREFLIGHT] SSH: 3 nodes reachable ... PASSED (3/3)`.
- Dataset frozen and sha256-verified on all four nodes (P0.5 clean).
- `EXEC CMD` included the wired staging controls: `--staging-workers 4
  --staging-queue-depth 2 --staging-capacity-timeout 600.0 --min-workers 24 --use-range-miner`.
- Fleet launched into the listening coordinator; **`ss -tn | grep 5700 | grep -c ESTAB` = 26**
  and held.
- A rig worker log (`/tmp/minerlogs/gpu0.log` on .122) showed exactly two lines —
  `[sieve_worker] Compiled kernel: java_lcg` and `java_lcg_reverse` — and **never grew again**.
- **`rocm-smi --showuse` on 192.168.3.122: all 8 GPUs at 0%.**
- `window_optimizer.py` (pid 10460) stayed alive at **60.1% CPU, state Ssl**, and the WATCHER
  log froze at 6211 bytes with no further output.
- Michael then killed the run and the fleet. **Nothing is running now.**

**Net: 25 remote daemons connected and compiled, then received no work. The parent kept
burning CPU. No staging pressure ever occurred** — an earlier single-worker run's
`[S172-BP] summary` showed `deferred_high_water=0 pause_events=0 staging_jobs_completed=0`,
i.e. the staging remediation is not implicated; the stall is UPSTREAM of staging, in worker
admission/assignment.

## Alpha's leading hypothesis — CONFIRM OR REFUTE, do not assume

`bind_worker` (`miner/range_miner_coordinator.py:~2143-2181`) sets
`status = "quarantined" if reason else "eligible"`, where `reason` combines
`admission_reason` (**"not in the run's frozen execution set"** — Beta's
G-NO-INFERENCE rule: *unknown miner workers must not become eligible merely because they
connected*) and `_validate_caps` (seed-cap inconsistency). A quarantined worker is
**registered-but-ineligible**: it KEEPS its TCP connection, but `can_assign_variant`
(`:2184`) returns False, so it is never assigned a stripe. `assign_stripes` (`:2235`) records
`"no eligible worker (cannot serve variant ...)"`.

That predicts precisely the observed state: ESTAB high, kernels compiled, GPUs 0%.

**Two candidate causes to separate:**
- **(a) Execution-set identity mismatch.** This run froze `d89834f1bf26`; the Phase-7
  authorized frozen set is `bea580e764905a0d9485d2688be5841cc95f16e16837c23aced1f634d97f67a8`.
  Worker identity is `f"{hostname}:gpu{gpu_id}"`. If the identities the daemons registered
  do not match the set this run froze, every remote worker is quarantined on admission.
- **(b) Capability/variant mismatch.** `_validate_caps` rejects the advertised `seed_caps`,
  or `supported_variants` does not contain `java_lcg` (the run's family), so
  `can_assign_variant` refuses even an eligible worker.

Also possible and worth checking: the parent is blocked in a readiness wait. NOTE for
accuracy — the only `min_workers` reads Alpha can find in
`window_optimizer_integration_final.py` are `pwc_min_workers=1` ("permissive for partition",
S163) on the **PWC** path; Alpha found no miner-path wait on `--min-workers 24`. Verify this
against live source rather than trusting it.

## What to produce

Read, in this order, and report evidence not inference:

1. **The ledger — this is the decisive artifact.** Read-only:
   `sqlite3 file:miner_staging/miner_ledger.db?mode=ro "select worker_id, hostname, status,
   quarantine_reason, backend from workers order by worker_id;"` (adjust to the real schema;
   find the DB if that path differs). **How many rows are `eligible` vs `quarantined`, and
   what EXACTLY does `quarantine_reason` say?** Also dump any stripe/assignment table rows
   and any `refused_reason` values for this run.
2. **The registration path in live source** — how `admission_reason` is decided, where the
   run's frozen execution set is resolved (`d89834f1bf26` vs the authorized
   `bea580e76490…`), and what identity string a worker actually sends versus what the
   execution set contains. Quote `file:line` for each claim.
3. **The assignment path** — `assign_stripes` and `can_assign_variant`: under what precise
   condition does a connected worker get zero stripes, and which condition matches the
   evidence.
4. **Where the parent was spending 60% CPU** — identify the loop; if it is a readiness or
   assignment retry loop, name it with `file:line`.
5. **The rig-side view** — the full worker logs on .122/.156/.164
   (`/tmp/minerlogs/gpu*.log`) and `logs/miner_workers/` locally: did any worker log a
   registration ACK, a quarantine notice, or an assignment? Read-only `tail`/`cat` only.

**Report** to `docs/CLAUDE_CODE_REPORT_S172_GATE12_DISPATCH_STALL.md`: the confirmed root
cause with file:line and ledger evidence; which hypothesis (a)/(b)/other it is; what the
minimal fix would be (DESCRIBE ONLY — do not implement); and, if a further run is required
to confirm, exactly what run and why, for Michael to decide. State plainly anything you could
not determine — an unproven claim is worse than a gap.
