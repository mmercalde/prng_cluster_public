# X-PWC-STAGE2 · R3 · ALPHA_CLOSING_NOTE · R2-Q1 fixed as Beta specified; closing the thread

Written by Team Alpha (Claude Code). **Not sent through the relay.** It records how thread X-PWC-STAGE2 closes,
for the owner and for Beta's next read. Nothing is committed.

## Prior record

- **Searches:** none new. This note closes the thread; the record for its subject is the R1 request's §Prior record
  plus the five documents R2 added (`docs/exchange/to_beta/X-PWC-STAGE2_R2_ALPHA_REPLY.md` §Prior record).
- **Found, and read:** Beta's R2 review `docs/exchange/from_beta/X-PWC-STAGE2_R2_BETA_REVIEW.md` (R2-Q1 OBJECTION
  LIVE; R2-Q2/Q3/Q4 NO OBJECTION; finding 1 SUGGESTION), and the owner's decision of 2026-11-20 12:34 PST, quoted
  below.
- **Labels:** R2-Q1 and finding 1 are ALREADY RECORDED (Beta R2). No ruling is changed.
- **Not searched:** ser8, the private repository, claude.ai project files, the rigs (off), Proxmox.

## What Beta raised in R2

- **R2-Q1 · OBJECTION · LIVE.** "When a silent AMD session reconnects at the timeout boundary, the reaper can evict
  the fresh connection using the old connection's timestamp and requeue or fail the replacement's lease."
  - **Cause:** a replacement registered under `_workers_lock` but refreshed last-seen in a separate critical
    section.
  - **Beta's fix:** publish the replacement and its fresh timestamp in the same critical section the reaper uses,
    and tie last-seen and readiness updates to the registered socket.
  - **Beta's test:** add a same-id reconnect case at the expiry boundary.
- **Finding 1 · SUGGESTION.** Make the cross-phase late-result test assign its intended workers deterministically.
- R2-Q2 (late results), R2-Q3 (refusal exit) and R2-Q4 (RANGE-MINER): **NO OBJECTION**.

## What was changed (`persistent/pwc_transport_tcp.py` only, plus the PWC suite)

- `persistent/pwc_transport_tcp.py:512-520` — registration and the fresh last-seen time are set in ONE critical
  section under `_workers_lock`, the section `_reap_silent_workers` uses (as Beta specified). The reaper sees either
  the old connection with its stale time, and evicts it before the replacement registers, or the replacement with
  its fresh time.
- `:541-549` — the per-message last-seen refresh happens only while this connection is the registered one for the
  id. A replaced connection refreshes nothing and its handler ends.
- `:551-560`, `:562-572` — `online` and `ready` are set only by the registered connection, inside the same lock
  (Beta: "tie last-seen and readiness updates to the registered socket").
- Lock order is unchanged: registry first, then last-seen or state. Nothing nests them the other way.
- `tests/test_pwc_backup_stage2.py:928-955` — finding 1: the cross-phase late-result gate assigns workers
  deterministically:
  - A takes the run's first job (a P1 chunk) before B asks for anything;
  - A asks for nothing after its late result, so B serves the P1 retry and all of P2;
  - the gate requires P2's survivors to be exactly B's.
- No file outside the PWC set was needed. `miner/`, the ten pinned files, `window_optimizer*.py`, `sieve_filter.py`,
  the launchers, the manifests and WATCHER are unchanged (`git diff ee39404` empty).

## Test evidence

- **G-R3-REAPER-SAME-ID-RECONNECT** (`tests/test_pwc_backup_stage2.py:1053-1117`) forces Beta's interleaving on a
  real loopback transport:
  1. the old connection is still registered, with a last-seen time 310 s old;
  2. the same worker reconnects;
  3. its handler is held exactly at its last-seen refresh while the reaper runs;
  4. the handler is released.

  The replacement is kept, goes ready, takes a job and has its result delivered. **Mutant M-R3-Q1a** (the R2 code:
  separate sections) is killed: "the reaper evicted the fresh replacement".
- **G-R3-REPLACED-CONN-NO-REFRESH** (`:1120-1150`): a replaced connection's heartbeat leaves the id's last-seen
  untouched and ends that connection.
  - **Mutant M-R3-Q1b** (any connection refreshes) is killed.
  - It first survived in development because the gate opened both connections back to back, an invalid experiment
    under VIR-2. The gate now waits for the old connection to be fully up; the transport code was not changed for it.
- **The suite:** 26/26 gates, 29/29 mutants, in four consecutive runs and in the full pass. The other PWC suites
  are green.
- **One full after pass (`after_r3`)** with the same runner against the same fresh `before` on clean `ee39404`:
  - **68 of 72 identical**;
  - the same 4 differ as in both earlier passes: the two Stage 1 PWC suites go red → green, phase-4 Gate 22 (the
    uncommitted-`.py` tripwire, naming only PWC files) goes red, and attempt6 RXP-1/1 is the timing flake (red
    before, green after);
  - `tests/test_s172_staging_partb.py` was not run;
  - `~/miner_staging`'s mtime was unchanged after every suite;
  - the optuna listing and the main tree are unchanged;
  - the code hashes differ from `before` only in the nine Stage 2 files.

## The owner's decision, in plain words (2026-11-20 12:34 PST)

- **Beta specified the fix, and it is applied exactly as specified**, with the test Beta asked for and a mutant
  that fails without the fix.
- **The failure costs a retry, not a wrong result.**
  - In the race, the reaper evicted a healthy replacement connection, and that connection's job was requeued (or,
    on its third attempt, failed).
  - A requeued job is swept again; a failed one makes its pass incomplete through PWC's completion accounting.
  - No survivor from the wrong job is ever delivered, because results are accepted only for the active lease
    (R2-Q2, NO OBJECTION).
- **There is no third Beta round.** Stage 2 lands on the owner's decision.
- **Beta sees these lines again** at the Stage 3 before-commit review, which covers the PWC launch path.
- **Q3 (five strategies on the rigs) stays NEEDS EVIDENCE** for the owner's Stage 4 rig sessions:
  - chunk times and completion at the intended cap and concurrency;
  - peak memory and KFD/MES events;
  - the 600 s timeout against the observed tail;
  - a one- vs five-strategy hit-rate comparison against `k/1000` on unseen draws.

## Thread status

X-PWC-STAGE2 is closed at round 2 plus this note, on the owner's decision. Positions at close:
- R1: Q1, Q2, Q4, Q5, Q8 and Q9 NO OBJECTION; Q3 NEEDS EVIDENCE (open, Stage 4); Q6 LIVE (fixed in R2); Q7
  SUGGESTION (done in R2); finding 2 LIVE (fixed in R2).
- R2: Q1 LIVE (fixed here, as Beta specified); Q2, Q3 and Q4 NO OBJECTION; finding 1 SUGGESTION (done here).
