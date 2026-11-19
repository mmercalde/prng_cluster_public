# X-LEDGER-RELEASE · R1 · REVIEW_REQUEST · hand release of Run I's held staging reservations

## Prior record

**Searches run** (2026-11-18): `/bin/grep -rnE 'reservations|held_reservations|release_reservation'` over
`miner/`, `scripts/`, `tools/`, `agents/`, `watcher_agent.py`, `window_optimizer*.py`, `tests/gate_s172_prod_shape.py`;
`/bin/grep -rnE 'pregate12|displaced_original|miner_ledger_archive'` over `docs/`.
**Found:** no CLI, script or start-up sweep that releases another run's reservations; the August ledger
displacement (`docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md:40`, `:46`). **Labels:** the hand release is
**NEW**; no ruling is changed. **Not searched:** ser8, the private repository, the claude.ai project, the rigs.

```
id / round   X-LEDGER-RELEASE / R1   (one round only; the owner decides)
type         REVIEW_REQUEST (one-off write to the live miner ledger)
base         main 5605ed3, miner/range_miner_coordinator.py as committed
```

**What happened.**
- Run I (`distributed_config_t1_34a6727c`) was killed by SIGTERM at 16:24. Its coordinator has no signal handler,
  so no abort ran. It left **3,554 held reservations**.
- The staging admission count has **no run filter**: `SELECT COALESCE(SUM(size_bytes),0), COUNT(*) FROM
  reservations WHERE status='held'` (`miner/range_miner_coordinator.py:1514-1522`).
- Runs J and K each failed with `staging_capacity_timeout`, with held files = 6,528 = the whole-trial budget.
  Each staged 2,980 jobs (console summary); 2,976 of those reservations per run are in the ledger.
- Confirmed on a ledger copy taken at 17:44 with nothing running: Run I has 3,554 held rows; every other run is released.

**Proposed clean-up (owner runs it, nothing running):**
1. Back up and sha256 `miner_ledger.db` **together with its `-wal` and `-shm`**.
2. `BEGIN IMMEDIATE;`
   `UPDATE reservations SET status='released', released_at=<now> WHERE run_id='distributed_config_t1_34a6727c' AND status='held';`
   (the same SET as `release_reservation`, `:1555-1556`). **COMMIT only if exactly 3,554 rows changed**, else ROLLBACK.
3. Re-run the admission count: held must be 0 across all runs.
4. Move Run I's staged files to a dated folder. Nothing deleted.
5. Leave Run I's trial (`running`), stripe and shard rows as they are.

**Simulation** (copy `/tmp/ledger_sim/miner_ledger.db`, WAL mode; the copy had no `-wal`/`-shm` beside it;
script `logs/ledger_sim/sim.py`, log `logs/ledger_sim/sim.log`, rc 0):
- before: integrity_check `ok`, foreign_key_check empty, admission (591,863 B, 3,554 files), all Run I;
- UPDATE rowcount 3,554, committed;
- after: admission (0, 0); integrity_check `ok`, foreign_key_check empty; every run's reservations `released`.
- Run I's remaining rows: trials 1 (`running`, cleanups `none`), stripes 96 (86 done, 9 staging, 1 claimed),
  shards 3,879 (3,554 verified, 325 pending), preflight_plans 1, trial_context 1.

**Second auditor (code reading only, no ledger opened): SAFE WITH CONDITIONS.** Its conditions, quoted
(there are five, not three):
- (a) Nothing is running and the sampler is closed.
- (b) Back up and checksum `miner_ledger.db` and its `-wal` together.
- (c) Use `UPDATE … WHERE run_id='distributed_config_t1_34a6727c' AND status='held'`. Require 3,554 rows changed, then confirm held is 0 across all runs.
- (d) Move every file whose name starts with `distributed_config_t1_34a6727c__`, including `.tmp.<pid>` temp files, to a folder outside `/home/michael/miner_staging`.
- (e) Record that the trial, stripe and shard rows were deliberately left as they are.

Its two cosmetic oddities, quoted:
- G-PROD-SHAPE run against Run I's run_id would now pass B8 "no reservations leaked". Record that the pass comes from the hand UPDATE.
- Run I's trial row stays `running` with no live process. Note this in the clean-up record.

**Later work, not asked about here:** a start-up sweep of dead runs, a held=0 pre-launch gate, a graceful stop
that runs `abort_trial`, a no-progress stall guard.

## Question

Answer with `NO OBJECTION` · `OBJECTION` (`LIVE`/`THEORETICAL`) · `SUGGESTION` · `NEEDS EVIDENCE`.
Label every point **`BLOCKS`** (would break a later run, corrupt a credit, or lose data on this cluster) or
**`TO-DO`**. Keep to real-world scenarios for this single-operator, three-rig development cluster. Give a
reproducer for any `BLOCKS`. One round only; the owner decides.

**Q1.** Is this hand release safe to apply to the live ledger on this cluster: **yes or no**?
