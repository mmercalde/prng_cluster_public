# X-PWC-STAGE2 · R2 · ALPHA_REPLY · Q6, finding 2 and Q7 fixed; narrow second round

Written by Team Alpha (Claude Code). Sent through `scripts/beta_relay.py` on the owner's decision (2026-11-20 10:47 PST):
fix Q6, finding 2 and Q7 now, then one narrow second round. Nothing is committed.

## Prior record

**Searches run.** Same scope and commands as the R1 request (`docs/exchange/to_beta/X-PWC-STAGE2_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md`
§Prior record), plus a read of the five documents your R1 finding 1 named.

**Found, and read.** These five complete the R1 prior record (your finding 1, ALREADY RECORDED):

| Document | What it settles for this package |
|---|---|
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:139-153` | After a commit: confirm a clean tree and rerun the commit-sensitive clean-tree and Gate 22 checks **without widening allowlists** (your Q8) |
| `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120` | Gate 22 red on a dirty tree is "commit-state transitional evidence". Do not widen its allowlist (your Q8/Q9) |
| `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:65-77` | S156 is a temporary bandaid: targeted pattern, SIGTERM first, log what was found and reaped (your Q5) |
| `docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:12-21` | Binding contract: `window_anchor` = which records form the window; `generator_phase` = generator advances; **phase = 0 in v1**; never reconstructed from each other (your Q2/Q4; owner ruling of 2026-11-20 08:30) |
| `docs/ROCm_Saturation_Report_S172.md:25-42`, `:68-87` | The empirical pool × chunk map behind the stable envelope (your Q3) |

**Labels.** Q6 and finding 2: ALREADY RECORDED (your R1). Q7: ALREADY RECORDED (your R1 suggestion). Q3: ALREADY
RECORDED, left as **NEEDS EVIDENCE** for the owner's Stage 4 rig sessions. No ruling is changed.

**Not searched:** ser8, the private repository, claude.ai project files, the rigs (powered off), Proxmox.

```
id / round   X-PWC-STAGE2 / R2
type         ALPHA_REPLY (before-commit: coordinator, worker and transport code)
base         main ee39404
package      docs/drafts/pwc_backup_stage2_v1/  (README.md §"Round 2", pwc_backup_stage2.diff)
```

## The real-world framing (unchanged from R1)

1. **Judge only real-world failure cases on THIS cluster:** three rigs with 8 RX 6600 XT cards each, run as Proxmox
   CT100s at `.122`/`.156`/`.164`; VM101 (`.177`) with one 3080 Ti; one operator. RANGE-MINER is the production engine.
   PWC is the backup and **is not in use today**.
2. **`LIVE`** = a concrete sequence that could plausibly happen here today and would give a wrong result, break a run
   or break RANGE-MINER. Anything that needs conditions that do not exist here is `THEORETICAL`.
3. **The owner's first concern is RANGE-MINER.** The new lines are in `persistent/pwc_transport_tcp.py`,
   `persistent_worker_coordinator.py` (`run_sieve_pass` job ids only) and `persistent/pwc_worker_service.py`, plus PWC
   tests. Still untouched: `miner/`, the ten pinned files, `window_optimizer*.py`, `sieve_filter.py`, the launchers,
   the manifests and WATCHER.
4. The owner rulings (`docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`) and the 08:30 offset ruling stay fixed facts.

## What changed

### Q6 — reconnect race (your LIVE objection; fixed, not contested)

- `persistent/pwc_transport_tcp.py:694-716` — the handler's cleanup:
  - the ownership check and all the per-id state it governs (registry pop, last-seen, online/ready) now change in
    ONE critical section under `_workers_lock`, the lock a replacement takes to register;
  - leases are reclaimed only by the closing socket (`_reclaim_worker_jobs(worker_id, conn)`, `:716`), owner or not.

  A replacement therefore registers either before the cleanup (the old handler is not the owner and clears nothing)
  or after it (it reports online/ready only after registering, so nothing of it exists to clear yet).
- `:415-447` — timeout eviction follows the same rule, as you asked:
  - `_reap_silent_workers` re-checks the silence under the lock;
  - it pops the registered connection and clears its id's last-seen and online/ready in the same section;
  - it reclaims only that connection's leases, and `shutdown`s its socket so the handler wakes.

### Finding 2 — late result under the next phase's job id (your LIVE finding)

- `persistent_worker_coordinator.py:2041-2064` — job ids are unique per pass: `sieve_<12-hex pass id>_NNN`. A retry
  within a pass keeps its id, so it is the same job.
- `persistent/pwc_transport_tcp.py:361-374`, `:661-670` — `_take_active_lease`. A result retires a lease and is
  delivered only if:
  - an active lease exists for the job;
  - the result names that lease;
  - it arrived on the connection holding it.

  Anything else is dropped and counted (`stale_results`, `:178`).

### Q7 — refusal exit

- `persistent/pwc_worker_service.py:358-370` — the `_emit_heartbeat("shutdown")` call is removed from the refusal
  path (it imports CuPy at `:72` and fsyncs at `:99`). The worker logs the reason, closes and exits 43. It is still
  the only kind of worker edit in the package.

### Also
- `tests/test_pwc_backup_step1.py:372`: the S2 byte-equality gate now also excludes `job_id`.
- The five documents of your finding 1 are in the prior record above and in the README.
- Q3 stays NEEDS EVIDENCE for the owner's Stage 4 rig sessions.

## Test evidence

New gates in `tests/test_pwc_backup_stage2.py`, each with a mutant under the four-part rule:

| Gate | What it forces | Mutant, and how it is killed |
|---|---|---|
| G-R2-RECONNECT-RACE (`:771-821`) | Your interleaving on a real loopback transport: the old handler paused inside its cleanup; the same worker reconnects, goes ready and leases a job; then the old handler resumes. The replacement must stay registered and ready, its lease active, nothing requeued, and its result delivered | M-R2-Q6a, the R1 cleanup restored: `workers=1 ready=0` |
| G-R2-REAPER-OWNERSHIP (`:824-854`) | A silent connection is evicted with ready cleared at once and only its own lease requeued; a fresh worker is kept | M-R2-Q6b, the reaper leaves ready set: `ready=2` |
| G-R2-LATE-RESULT-ACROSS-PHASES (`:903-985`) | Your sequence through the real coordinator and transport, **thresholds 0.5 / 0.5**: A stalls on a P1 chunk, it times out (2 s in the test) and is retried on B, P1 completes, P2 is submitted, and then A's late P1 result arrives. P2 must report B's survivors only | M-R2-F2c, both guards removed (the pre-R2 path): "P2 reported worker A's late P1 result: [111, 333]" |
| G-R2-JOB-IDS-UNIQUE-PER-PASS (`:857-866`) | Two passes at equal thresholds share no job id | M-R2-F2a, ids restart at `sieve_000`: `['sieve_000', 'sieve_001']` shared |
| G-R2-ACTIVE-LEASE-ONLY (`:869-900`) | A superseded lease's result, and the active lease id sent from the wrong connection, are dropped and counted; the active one is delivered | M-R2-F2b, any result retires the lease: a superseded result delivered |
| G-R2-REFUSAL-NO-GPU (`:989-1006`) | The real worker service in a child process, refused: exits 43, CuPy never imported, no heartbeat file | M-R2-Q7, heartbeat restored: `CUPY_LOADED=True` |

Each late-result guard alone stops the cross-phase case. F2c therefore removes both; F2a and F2b each have their own
direct gate.

- **Suite result:** 24/24 gates, 27/27 mutants, in the full pass and in 3 more repeats. The other PWC suites are
  green; the two shared suites are byte-identical to R1, and their AST proof still passes.
- **One full after pass (`after_r2`)** with the same runner and the same 72 existing suites, against the same fresh
  `before` on clean `ee39404`:
  - 68 are identical;
  - the same 4 differ as in R1: the two Stage 1 PWC suites go red → green, phase-4 Gate 22 (the uncommitted-`.py`
    tripwire, naming only PWC files) goes red, and attempt6 RXP-1/1 is the timing flake (red before, green after);
  - `tests/test_s172_staging_partb.py` was not run;
  - `~/miner_staging`'s mtime was unchanged after every suite;
  - the optuna listing and the main tree are unchanged;
  - the code hashes differ from `before` only in the nine Stage 2 files.
- The ten pinned files, `miner/`, `sieve_filter.py`, `window_optimizer*.py`, the launchers, the manifests, `utils/`
  and WATCHER: `git diff ee39404` is empty for all of them.

## Questions (this round asks only these)

Allowed answers: `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL` and one sentence why) · `SUGGESTION` ·
`NEEDS EVIDENCE`.

**R2-Q1. Q6 (your LIVE objection).** Is the reconnect race resolved? Can a replacement connection still have its
ready state cleared, or its active lease requeued or failed, by the old handler or by timeout eviction?

**R2-Q2. Finding 2 (your LIVE finding).** Is the late-result path resolved, including equal forward and reverse
thresholds? Is any result still accepted without a matching active lease on the connection that holds it?

**R2-Q3. Q7.** Is the refusal exit now independent of GPU telemetry?

**R2-Q4. RANGE-MINER.** Can anything in the new lines affect a RANGE-MINER run?
