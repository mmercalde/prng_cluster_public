# CLAUDE CODE INSTRUCTIONS — STEP 1 CREDIT REFUSED ON A NON-EMPTY LEDGER `-wal` (Run 4, first attempt)

## Prior record

**Searches run (public mirror, HEAD `0e13dd6`, 2026-11-15) and live reads on VM101 through the owner's web terminal:**

- Read in full: `docs/RUN4_LAUNCH_CHECKLIST.md` (step 14's operator note on the ledger `-wal`).
- Read: `window_optimizer_integration_final.py:1426-1465` (`_step1_miner_sweep_evidence`), `:3395-3423` (the S7 credit gate); `scripts/gate12_concurrency_sampler.py:202-211` (`connect_ro`), `:240-247`, `:1090-1100` (the sampling loop); `gate12_launch.sh:356-397` (sampler start and its supervisor).
- Read: `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §A (`:24-45`).
- `grep -rn -i "immutable" docs/exchange/from_beta/*.md` (hits read at `X-FINDINGS-PLAN_R1` `:71-73`, `X-FINDINGS-PLAN_R2` `:28-35`, `X-0001_R1` Q6 `:68-74`, `X-MINER-STALL_R1` `:132`).
- Live, read-only: the run log `logs/gate12_20261115_090553.log`, the sampler verdict, `ls` of `/home/michael/miner_staging/` (file sizes only; the ledger was not opened), `--step1-search-status`.
- **Not read by Alpha:** `utils/step1_search_ledger.py`, `miner/range_miner_coordinator.py` ledger connection code (`MinerLedger._conn`), `tests/test_step1_search_record_stage1.py` in full, `tests/gate_s172_prod_shape.py`, the stage-1 report in full. **Reading them is this session's job.**

**Documents that already decide this subject:**

| Document | What it settles | Label |
|---|---|---|
| `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §A | For sessions reading live SQLite files: never `mode=ro` alone; `-wal` absent or empty → `mode=ro&immutable=1`; `-wal` non-empty → `UNAVAILABLE`, do not open. Written for outside readers, to stop a read from writing sidecar files into live state | ALREADY RECORDED |
| `window_optimizer_integration_final.py:1443-1456` | The same rule built into Step 1's own credit check: a non-empty `-wal` earns no credit (S7 fails closed) | ALREADY RECORDED (code) |
| `docs/exchange/from_beta/X-MINER-STALL_R1_BETA_REVIEW.md:132` | Beta read this evidence reader and raised no objection to it | ALREADY RECORDED |
| `docs/RUN4_LAUNCH_CHECKLIST.md` step 14, "NEW, operator note" | The prep agent reproduced on a throwaway file that a `-wal` stays non-empty when the last writer closes while a `mode=ro` reader is open. It recorded the Run 4 outcome as "not known" | ALREADY RECORDED |
| `docs/RUN4_LAUNCH_CHECKLIST.md` "If something goes wrong", last row | Retrying on the same search after a failed trial is `[UNVERIFIED]`; the recorded route is close and reopen with `--search-repeat-of` | ALREADY RECORDED, unverified |

**Items this brief raises:** the Run 4 outcome below is **NEW**. The reading in §2 that the outcome was near-certain under the committed harness is **NEW** and is Alpha's reading of source, not a measurement.
**Not searched:** ser8 archives, the private repository, the rigs' file systems beyond the worker logs copied into the evidence bundle.

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Worktree:** `/home/michael/tfm_worktrees/step1-credit-wal` (hyphens).
**Base commit:** record `git rev-parse HEAD` (expected `0e13dd6` or later).
**Review class:** code that decides whether a run counts. **Beta reviews before commit**, through the relay. Label every Beta objection LIVE or THEORETICAL.
**Outputs, in your worktree:** the code change, its tests, and `docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md`.

**The yardstick (CLAUDE.md).** Prediction-pool hit rate against `k/1000` on unseen draws. This work is about whether a completed sweep is credited. It measures no hit rate.

## 0. Rules for this session

- **You launch nothing on the fleet.** No WATCHER run, no coordinator on port 5700, no rig SSH. The owner launches.
- **You do not open, checkpoint, copy over or edit `/home/michael/miner_staging/miner_ledger.db` or its `-wal` / `-shm`.** A frozen copy is in the evidence bundle (§1); work on copies of that copy, in your worktree or `/tmp`.
- **You do not open or close a Step 1 search and you do not edit `prng_analysis.db`.** Those are the owner's acts.
- No scheduled wake-ups during a relay. `pgrep -f "[b]eta_relay.py"`, never the unbracketed form.
- No edit under `miner/` unless §3 option C is chosen by the owner. A change there is a governed file: it would have to be redeployed to the three rigs and parity re-proven.
- Do not weaken the credit gate: a sweep that did not finish must still earn no credit. Every existing no-credit test stays green.
- Stop and report to the owner at each stop gate in §5.

## 1. What happened (measured)

Run 4, launched by the owner 2026-11-15 09:05:53 PST from `/home/michael/run4_launch.sh` (a copy of `gate12_launch.sh` with only the `--params` block changed; diff shown to the owner). HEAD `0e13dd6`, tree clean, parity 30/30, search `s1s-884f7a8e305dde9d6ac4a3ed2cf347fc794e653ee17fe3c717ec637951f1a9dd` over `[2147483648, 4294967296)`.

| Fact | Value | Source |
|---|---|---|
| Run id | `distributed_config_t1_57dc50fa` | run log |
| Start taken from the search | `first gap 2,147,483,648 -> seed_start=2147483648` | run log, 09:07:33 |
| Phase starts | 09:07:39 · 09:09:05 · 09:10:44 · 09:15:16; 25 eligible workers at each | `[S172-BP] derived_bound stage` |
| `[S172-BP] summary` | 09:19:49 | run log |
| Survivors | phase 3: 60 · phase 4: 23,568 (the predicted exact counts); bidirectional 0 | run log |
| Worker disconnects | 0 | `grep -c WORKER_DISCONNECTED` |
| Saturation verdict | SATISFIED, both parts | `_verdict.txt` |
| Terminal | `[STEP1-SEARCH S7] miner sweep evidence UNAVAILABLE: /home/michael/miner_staging/miner_ledger.db-wal is non-empty (4148872 bytes); an immutable read would be stale — no credit`, then `Step1NoCreditError: S7 no credit (no_sweep_evidence): trial 0: COMPLETE in the study but no sweep evidence from this chunk`; `Step 1 failed with code 1` | run log, 09:19:50 |
| After | search state `gap`, `next_seed_start` 2147483648; no new directory under `.s172_accumulator/generations/` (latest is still `gen-20260818T013219550933Z-…`); tree clean; rigs up, 8/8 GPUs each | read-only checks, 09:20-09:53 |
| WATCHER | reported the failure and halted (`explicit step failure overrides output-file evaluation`), although `optimal_window_config.json` was written at 09:19 | Telegram alert, file listing |

**Evidence bundle (frozen, 78 files, `SHA256SUMS.txt`):** `/home/michael/run4_attempt1_20261115_090553/` — `logs/` (run log, evidence, fleet, sampler TSV and verdict, digests, liveness, `miner_workers/`), `run4/` (the pre-launch gate outputs), `riglogs/<ip>/` (each rig's `/tmp/minerlogs`), `run4_launch.sh`, and `ledger/` (a file copy of `miner_ledger.db`, `-wal`, `-shm` taken at 09:53 with `cp -p`, with no SQLite open).

## 2. Alpha's reading of the cause (by reading; you verify it)

1. The credit check refuses when `-wal` is non-empty (`:1443-1456`).
2. The sampler opens the ledger `mode=ro` every 2 s for the whole run (`scripts/gate12_concurrency_sampler.py:1098`, `connect_ro` at `:202`). `with connect_ro(...) as conn` does not close a `sqlite3` connection; the connection lives until the name is rebound on the next sample.
3. The harness stops the sampler only after the WATCHER process has exited, plus 10 s (`gate12_launch.sh:392-396`). The credit check runs inside Step 1, a child of WATCHER. So the sampler is still alive, and very probably still holding a connection, at the moment credit is decided.
4. In WAL mode the `-wal` file is removed only when the last connection closes. With a reader still connected it stays, non-empty.

If 1-4 hold, **no run launched through the committed harness can be credited**, however cleanly it sweeps. Attempt 9 (2026-08-17) and the 2026-08-22 run predate the credit check, which is why this did not show before. The prep agent's throwaway test (checklist step 14) is consistent with this. Alpha recorded the risk to the owner as "not known" without reading the sampler's lifetime; that was Alpha's error.

**Your first task is to prove or refute 1-4 by execution**, on a fixture, with the real sampler script and the real `_step1_miner_sweep_evidence`, and on a copy of the frozen ledger. State plainly if the reading is wrong.

## 3. The repair — candidates, and Alpha's recommendation

The requirement: **a sweep that completed is credited when the ledger's committed rows show it, whether or not a read-only observer is connected; a sweep that did not complete still earns nothing; the credit read writes nothing into the ledger's directory that was not already there.**

| | Candidate | For | Against |
|---|---|---|---|
| **A** | In `_step1_miner_sweep_evidence`: when `-wal` is non-empty, read through an ordinary read-only connection (not `immutable`), which sees committed pages in the WAL. Keep `immutable=1` when `-wal` is absent or empty | Smallest change; in `window_optimizer_integration_final.py` only; no rig redeploy. The sidecar files already exist in this case, so the read creates none | It relaxes, inside Step 1, the wording of addendum §A item 4. Beta must agree that §A's purpose (an outside reader must not write to live state, and must not read stale data) is met. Needs a statement of what "writers have stopped" means at this point in Step 1 (X-0001 R1 Q6) |
| **B** | Change the harness or the sampler so no observer holds the ledger when credit is decided | No change to the credit check | The credit check stays one stray reader away from refusing a good run. Any other reader (a dashboard, a person) reproduces today. Also changes a certified harness and its tests |
| **C** | The coordinator settles the WAL itself at trial commit (`wal_checkpoint(TRUNCATE)`) | Evidence is clean for every later reader | An edit under `miner/` (governed file, rig redeploy, parity). A checkpoint can be blocked by a reader mid-transaction, so it does not remove the dependence on other readers |

**Alpha recommends A**, with B's race recorded as a note for the harness. It is not decided: build A, state its limits, and take Beta's review. If you find A cannot meet the requirement, stop and report before building anything else.

## 4. Tests (all in your worktree; none against live state)

1. **Red first.** A fixture that reproduces today: a WAL-mode ledger with a complete four-phase stripe set for one `run_id`, the writer closed while a `mode=ro` reader (the real sampler's `connect_ro`) is still connected, `-wal` non-empty. Show the current code returns no evidence.
2. **Green.** The same fixture after the change: evidence is read and equals the committed rows. Compare the directory's file names, sizes and hashes before and after the read, and report any difference.
3. **Still refused.** Unfinished phase; a stripe not `done`; a gap in the tiling; `committed` false; unreadable ledger. Each still earns no credit.
4. **Stale-read control.** A fixture where the main file alone would give the wrong answer (rows only in the WAL). Show the new read returns the committed rows, and that an immutable read of the same fixture would not.
5. **On a copy of the frozen ledger** (`/home/michael/run4_attempt1_20261115_090553/ledger/`, copied again into `/tmp`): run the new evidence reader for `distributed_config_t1_57dc50fa` and report what it returns for each phase. This says whether today's sweep would have been credited. It credits nothing.
6. The existing suites that cover this file stay green; report the full list you ran and their counts, on the patched tree and on a clean worktree at the base commit.

## 5. Questions to answer in the report, and stop gates

- **Q1.** Is §2's reading right? Evidence by execution.
- **Q2.** After today's failed attempt, what must the owner do to run the upper half again? Establish it from `utils/step1_search_ledger.py` and the stage-1 tests, and by a dry run on a fixture database: does a second launch on search `s1s-884f7a8e…` refuse before dispatch, and is `--step1-search-close` then `--step1-search-open … --search-repeat-of <old id>` the working route? Give the exact owner commands, with `PYTHONPATH=.` (the checklist's commands omitted it and failed with `ModuleNotFoundError: No module named 'agents'`).
- **Q3.** Will the next run's coordinator settle the live ledger's present 4 MB `-wal` by itself when it first writes? By reading only.
- **Q4.** Does `tests/gate_s172_prod_shape.py` section B have the same dependence on an empty `-wal` (`:39-41`)? If so, say what it would have reported today. Do not change it under this brief.

**Stop gates:** (1) after Q1, before writing the change; (2) if A cannot meet the requirement; (3) after Beta's first review; (4) before anything is proposed for landing.

## 6. Out of scope

The skip-range decision, the PWC work, the hit-rate instrument, the agent groundwork, `docs/LEADS.md`, and the checklist corrections found today (root key auth to the Proxmox hosts works from VM101; netconsole needs the host console level raised with `dmesg -n 7`; `PYTHONPATH=.`). Alpha records those separately.
