# PWC full backup — Stage 1 (uncommitted, for the owner)

## Prior record

- **Searches:** this stage's own searches are in the Beta draft's Prior record
  (`docs/exchange/to_beta/X-PWC-STAGE1_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md`). The full governance search
  is in the scope this stage implements: `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md` "Prior record" (main `27eb991`).
- **What this stage is:** the scope's Stage 1 row: land the step-1 candidate over main, repair two fixtures, add B5,
  A3 level 1, B6 and the A4 root tests.
- **Approval:** the owner, 2026-11-19 20:50 PST: "get started, DO NOT break range miner".

**Labels:**

| Item | Label |
|---|---|
| The candidate S1–S8 | ALREADY RECORDED (X-PWC-BACKUP R1/R2, no LIVE objection) |
| A3 level 1 | ALREADY RECORDED (X-PWC-CACHE-RESET R1 `:85-95`, R2 `:103`) |
| A4 | ALREADY RECORDED (scope A4) |
| B5, B6 | NEW in the scope; the B5 execution evidence is NEW here |
| **The S156 find never printed `none`** | **NEW** |
| **The held worker hunk** | **NEW**, owner decision |
| **The `~/miner_staging` probe by a miner suite** | **NEW**, see "Things you should know" |

No ruling is changed:
- the hybrid quarantine stays on;
- the S147 phase policy is untouched;
- R5 still refuses a non-zero start at the integration call site;
- PWC still earns no credit;
- no default is changed.

**Not searched:** ser8, the private repository, the claude.ai project files, the rigs (powered off), Proxmox.

---

## In plain words

1. **RANGE-MINER is not touched.** None of the ten rig-parity-pinned files changes; the table is below. Nothing under
   `miner/`, no window optimizer, no `sieve_filter.py`, no launcher, no manifest and no worker changes.
2. **Before and after:** 71 existing suites ran on clean main, then again with this change, one at a time, CPU only,
   with ssh blocked. Four of them differ, and each difference is explained below:
   - two old PWC test files were stale and are now green;
   - one miner tripwire went red because it counts uncommitted `.py` files, as recorded rulings say it will until
     commit;
   - one miner check flickered once and then passed twice.
3. **What Stage 1 fixes in PWC:**
   - **the stale-worker kill on a rig killed itself.** It never reached `sieve_gpu_worker` processes. Its result is
     now read, and the rig is skipped if the kill is not confirmed;
   - **a worker this run did not launch can no longer join the run** (TCP admission list);
   - **variable-skip skip sequences are no longer dropped** by the parser (field-name mismatch);
   - **the step-1 work is landed:**
     - completion accounting;
     - `seed_start`;
     - a required dial address;
     - no silent fallbacks;
     - the stable envelope;
     - the threshold echo gate;
     - the protocol import fix;
     - the PWC parity gate.
4. **One piece is held back for you:** the candidate's edit to the PWC worker, `persistent/pwc_worker_service.py`.
   This task said not to edit "the worker". The edit only removes the worker's own fallback dial to `.127`, and the
   coordinator now always sets the address, so nothing here needs it. It is in `HELD_pwc_worker_service.diff`.
5. **PWC is still not usable from the pipeline with rigs, now loudly instead of silently.** Before, rig workers
   silently dialled `192.168.3.127` (frozen Zeus). Now PWC refuses at startup, because
   `window_optimizer_integration_final.py:1968-1986` passes no dial address and a 2,000,000-seed chunk. Fixing that
   call site is Stage 3 (a shared launch file).

---

## The ten rig parity-pinned files (`scripts/gate12_parity_gate.py:174-185`)

**None of the ten is touched.** Each was hashed in this worktree against `git show 27eb991:<file>`:

| Pinned file | Changed by Stage 1? | sha256 (first 16), main = worktree |
|---|---|---|
| `adaptive_thresholds.py` | no | `cda23bb3cd624022` |
| `execution_set.py` | no | `d21614701c31a7b4` |
| `hybrid_strategy.py` | no | `8b982fbb8ecbe272` |
| `miner/__init__.py` | no | `699e930e379d07a7` |
| `miner/dataset_authority.py` | no | `365c8e3ee9abf80a` |
| `miner/range_miner_coordinator.py` | no | `b8e259b3e12b9529` |
| `miner/range_miner_protocol.py` | no | `7b7f8e1979141789` |
| `miner/range_miner_worker.py` | no | `2bca2a083262a523` |
| `prng_registry.py` | no | `32435fb82737964d` |
| `sieve_gpu_worker.py` | no | `9ace0d83ad99d485` |

Also unchanged (`git diff --quiet 27eb991 --`):
- `window_optimizer.py`;
- `window_optimizer_integration_final.py`;
- `sieve_filter.py`;
- `persistent/pwc_worker_service.py`;
- `gate12_launch.sh`;
- `scripts/launch_fleet_manual.sh`;
- all of `miner/`;
- all of `agent_manifests/`.

**Every changed file against that list:**

| Changed file | Kind | Pinned? | Loaded by RANGE-MINER? |
|---|---|---|---|
| `persistent_worker_coordinator.py` | PWC coordinator | no | **imported, not run**, at module scope by `window_optimizer_integration_final.py:58`. Import-time effect is definitions and constants only; G-IMPORT checks it in a fresh process |
| `persistent/pwc_protocol.py` | PWC protocol | no | no (the miner has its own protocol; `miner/range_miner_protocol.py` mentions it in comments only) |
| `persistent/pwc_transport_base.py` | PWC | no | no |
| `persistent/pwc_transport_tcp.py` | PWC | no | no |
| `scripts/pwc_parity_gate.py` (new) | PWC tool | no | no |
| `tests/test_pwc_backup_step1.py` (new), `tests/test_pwc_stage1.py` (new), `tests/test_pwc_cache_reset.py`, `test_persistent_worker_harness.py`, `test_s149_rocr_isolation_harness.py` | PWC tests | no | no |

`git diff 27eb991 --stat` lists only PWC files:

```
 persistent/pwc_protocol.py          |   2 +-
 persistent/pwc_transport_base.py    |  15 +-
 persistent/pwc_transport_tcp.py     |  70 +++-
 persistent_worker_coordinator.py    | 804 +++++++++++++++++++++++++++++++++---
 test_persistent_worker_harness.py   | 176 ++++----
 test_s149_rocr_isolation_harness.py |  79 ++--
 tests/test_pwc_cache_reset.py       |  29 +-
 7 files changed, 968 insertions(+), 207 deletions(-)
untracked: scripts/pwc_parity_gate.py (588 lines), tests/test_pwc_backup_step1.py (885), tests/test_pwc_stage1.py (606)
```

The rig-deployed PWC files are `persistent_worker_coordinator.py` (it runs on VM101 only) and `persistent/*`. They need
a redeploy and the PWC parity pin (S8) before any rig run. No rig was touched.

---

## What changed, with `file:line` (worktree line numbers)

**1. The step-1 candidate S1–S8, merged over main.**
- **Source:** `/home/michael/tfm_worktrees/pwc-step1`, base `976a6db`.
- **Merge:** `git merge-file <candidate> <976a6db> <27eb991>` returns rc 0 with no conflict markers for all four files.
  `persistent/pwc_worker_service.py` is held.
- **Details:** what S1–S8 do is in that worktree's `docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md` and is not repeated
  here.
- **One behaviour of the merge, from the scope:** a rig refused by the cache reset is absent from
  `_effective_pool_by_host`, where before it was present with 0.

**2. B5: the remote stale-worker kill no longer kills itself, and its result is read.**
- **The command:** `persistent_worker_coordinator.py:174-196`:
  - `PWC_STALE_KILL_OK`;
  - `PWC_STALE_KILL_PATTERNS = ("[p]wc_worker_service", "[s]ieve_gpu_worker")`;
  - `PWC_S156_PERSISTENT_PATTERN`;
  - `pwc_stale_kill_command()`.

  The command kills, waits 2 s, exits 6 if any match is left, and prints the marker otherwise. A bracket pattern
  matches a worker's command line but not the shell's own, which holds the brackets literally.
- **Where the result is read:** `:1199-1217` (TCP launch). A missing marker, a non-zero exit, or an ssh error
  **refuses that rig for this run**, exactly as a failed cache reset already does at Step 1b. The rig then gets no
  reset, no launch, no admission, and no entry in `_effective_pool_by_host`.
- **SSH route (S156 find and reap):** `:912-944` use the bracket pattern. `:946-949` log a warning when the reap does
  not report `reaped:0`. The reap is still not refusing, as before.
- **Evidence (executed, not read):** in a private PID namespace with dummy processes, HEAD's string ended with the shell
  SIGKILLed (rc 137), no `killed`, and the `sieve_gpu_worker` dummy alive:
  - **the second `pkill` never ran, on main, ever;**
  - whether the `pwc_worker_service` dummy dies is a race: it was killed in 6 runs and survived in 1;
  - HEAD's S156 find printed its own shell instead of `none`, so the reap ran on every rig at every start.

**3. A3 level 1: TCP launch-set admission.**
- **The admission API:** `persistent/pwc_transport_tcp.py:170-199`: `set_allowed_worker_ids`, `allow_worker_id`,
  `revoke_worker_id`, `_admits`. `None` = no list, every hello accepted: the old behaviour, kept for callers that set
  no list.
- **Refusal:** `:431-449` refuses an id outside the list with `accepted: False` and registers nothing. `:624-636`: a
  refused hello does not reclaim or pop a live worker filed under the same id.
- **Coordinator side:**
  - `persistent_worker_coordinator.py:973-980` opens an **empty** set before `start()`, so nothing is admitted in the
    gap before the rig kill;
  - `:1283-1300` admits each rig worker id before its launch, and revokes it if the launch prints no `PID=` or fails;
  - `:1337`, `:1353` do the same for the VM101 local workers.
- **Limit:** a stale worker with the **same** id as a fresh one (ids are `<ip>_gpu<N>`) is not separated. That needs
  level 2 (a per-run token, Stage 2, which edits the worker). With B5 the kill is now confirmed before launch.
  Refused workers keep retrying every 5 s (`persistent/pwc_worker_service.py:213`), because the worker is not edited.

**4. B6:** `persistent_worker_coordinator.py:1635-1640`. The TCP legacy parser reads `skip_pattern`, which is the name
`sieve_filter.py` emits, as well as `skip_sequence`. It matters only once the hybrid quarantine is lifted.

**5. Test repairs:**

| File | Repair |
|---|---|
| `tests/test_pwc_backup_step1.py` | The fake rig answers like a healthy rig: reset marker, kill marker, `PID=` line. The fake transport has the admission API. G-S3-WORKER and M-S3b print **HELD**, never PASS |
| `tests/test_pwc_cache_reset.py` | Passes the dial address, `seed_cap_amd=25_000` and the transport (S3/S5); answers the kill marker and a PID; fake admission object |
| `test_s149_rocr_isolation_harness.py` | Five checks asserted the S149 defect still existed. They now assert the S155 design that replaced it. The Device(gpu_id) check reads code, not text, because the only match was a stale docstring. The all-pass banner text is corrected |
| `test_persistent_worker_harness.py` | T13/T14 assert the quarantine refusal (no dispatch, no file). T19/T20 build the D3.25 v2 result with `build_trial_populations`. T20 sets `use_range_miner=False` (a bare MagicMock flag is truthy). T11/T12/T24 pass the sessions/skip that the removed S4 fallback used to supply |

`tests/test_s172_threshold_propagation.py`'s D6 sub-harness is a **miner test**. It has been stale since `48a8705`
and is left alone.

---

## Every test run

**Conditions** (`logs/pwc_stage1_v1/run_suites.sh`):
- sequential (memory note: S172 suites must not run concurrently);
- `CUDA/HIP/ROCR_VISIBLE_DEVICES` empty;
- `ssh`, `scp` and `rsync` replaced by a stub that logs and exits 255;
- venv `~/venvs/torch`.

**Runs:**
- **Before:** clean `27eb991`, 71 suites, `logs/pwc_stage1_v1/before/`.
- **After:** the Stage 1 tree, the same 71 plus the 2 new suites, `logs/pwc_stage1_v1/after/`.
- The first "after" attempt was stopped for the Zeus reboot and thrown away. Before the restart, the `data/` outputs
  the suites wrote were moved to `logs/pwc_stage1_v1/data_written_by_before_and_partial_after/`, so both runs started
  with no `data/`.

**Result: 4 of 71 existing suites differ in exit code or in the set of failing checks.** The tool is
`logs/pwc_stage1_v1/compare_checks.py`; its output is in `before_after_failing_checks.txt`, and every suite's tally
lines are in `before_after_tally.md`. The other 67 have the same exit code and the same failing checks.

| Suite | Before | After | Why |
|---|---|---|---|
| `test_s149_rocr_isolation_harness.py` | rc 1, 10/15 | **rc 0, 16/16** | A4 repair (stale expectations) |
| `test_persistent_worker_harness.py` | rc 1, 21/25 | **rc 0, 25/25** | A4 repair |
| `tests/test_s172_phase4_coordinator.py` | rc 0, 63/63 | rc 1, **62/63, Gate 22 only** | Gate 22 reads `git status` and names exactly this package's 10 `.py` files, including the explicit `pwc_protocol.py` freeze at `:2422`. Transitional per `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120` and `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145-151`; do not widen the allowlist. **Re-run on the committed tree.** It is NOT verified that gate 22 is then green: the `pwc_protocol.py` freeze stays, and S7 edits that file. That needs the owner, as the step-1 report §5 says |
| `tests/test_s172_attempt6_remediation.py` | rc 1, 76/78 | rc 1, 75/78 (RXP-1/1) | Re-run twice right after, on the same tree: **76/78 both times**, RXP-1/1 PASS (`logs/pwc_stage1_v1/rerun_attempt6_after_{1,2}.log`). It is a timing flake in a miner exit-class gate; the suite references no PWC file |

**The named suites** (the six that touch the PWC file, plus key miner and Step 1 suites):

| Suite | Before | After |
|---|---|---|
| `test_s149_rocr_isolation_harness.py` | 10/15 | 16/16 |
| `test_persistent_worker_harness.py` | 21/25 | 25/25 |
| `tests/test_s172_threshold_propagation.py` | 3/5, G-MINER-UNCHANGED red (D6 sub-harness stale), G-KERNEL unavailable | same 3/5, same two. G-MINER-UNCHANGED now stops one assertion earlier, on `persistent/pwc_protocol.py` in its `git status` frozen list |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | 13/13 | 13/13 |
| `tests/test_s172_resolved_execution_set.py` | 34/34 | 34/34 |
| `tests/test_s172_phase4_coordinator.py` | 63/63 | 62/63 (Gate 22, above) |
| `tests/test_pwc_cache_reset.py` | 21/21 | 21/21 |
| `tests/test_s172_phase1_scaffolding.py` / `phase2_protocol` / `phase3_worker` | 6/6 · 6/6 · 18/18 | same |
| `tests/test_step1_credit_wal.py` | 12/12, mutants 5/5 | same |

Several miner and Step 1 suites fail **identically before and after**. The causes are:
- the worktree has no frozen dataset (`miner_seed_start`, `step1_search_record_stage1`);
- stale miner fixtures lack `window_anchor`/`generator_phase` (`phase5_d1_*`, `d2`, `d5`, `d6_*`);
- `run4_routing_patch` is INCOMPLETE.

Those are status, not caused here. Their failing-check sets are equal.

**New suites:**

| Suite | Result |
|---|---|
| `tests/test_pwc_backup_step1.py` | gates 22/22, mutants 16/16, **2 HELD** (G-S3-WORKER, M-S3b) |
| `tests/test_pwc_stage1.py` (final) | **9/9 gates, 8/8 mutants, 5 of 5 runs** (`logs/pwc_stage1_v1/stage1_repeat_*.log`) |

In the full "after" run, `tests/test_pwc_stage1.py` was 8/9. Its evidence gate asserted that exactly one dummy survived
HEAD's string, and that time both survived. The assertion was wrong, not the code. It now asserts what the evidence
supports: the shell is SIGKILLed, there is no marker, and the sieve dummy is alive. The PWC dummy's fate is reported.

**What `tests/test_pwc_stage1.py` covers:**
- G-B5-OLD-SELF-KILL (evidence), G-B5-NEW-KILL, G-B5-S156, G-B5-WIRING;
- G-A3-STARTUP-ORDER, G-A3-LAUNCH-SET, G-A3-TRANSPORT (a real `TCPWorkerTransport` on 127.0.0.1);
- G-B6;
- G-IMPORT: the integration module and the miner package import in a fresh process;
- mutants M-B5a/b/c, M-A3a/b/c/d, M-B6, each killed by its detector's own assertion.

**Live state:**
- the main tree's `git status` and HEAD are unchanged in both runs;
- the `optuna_studies/` listing is unchanged;
- the 25 ssh attempts per run from a miner suite to `.122`/`.156`/`.164` (reading `/tmp/minerlog`) were refused by the
  stub (`logs/pwc_stage1_v1/fake_net_calls*.log`);
- no GPU was used.

---

## Things you should know

1. **An existing miner test writes into the live staging folder.**
   - **What it does:** `tests/test_s172_staging_partb.py:599-609` (`gate_manifest_staging_dir_is_usable_here`) reads
     `staging_dir` from `agent_manifests/window_optimizer.json:274` (`/home/michael/miner_staging`). It calls
     `validate_coordinator_staging_dir`, which proves writability by writing and renaming a probe file there
     (`miner/range_miner_coordinator.py:12190-12205`).
   - **Seen in both runs:** `~/miner_staging`'s directory mtime moved at exactly that suite (21:38:12 before,
     23:24:44 after).
   - **What was not changed:** the listing, and the ledger, `-wal`, `-shm` and provenance file mtimes.
   - **Cause:** this was not caused by Stage 1, and it happens whenever that suite runs anywhere on VM101.
   - **Consequence:** I could not honour "no live ledger" for that suite without skipping it. It touched the folder,
     not the ledger. Status for you.
2. **The held worker hunk** (`HELD_pwc_worker_service.diff`) removes the worker's `PWC_HOST=192.168.3.127` /
   `PWC_PORT=5600` fallbacks. The risk without it: a PWC worker started by hand, or from an old script, dials frozen
   Zeus. Every coordinator launch now exports both variables (G-S3-SET). Your call.
3. **Startup refusal from the pipeline** (plain-words point 5). Any PWC run on rigs through WATCHER or
   `window_optimizer.py` now stops at startup with a named reason, until Stage 3.
4. **Stale anchors:** `window_optimizer_integration_final.py:1997-2006` cites
   `persistent_worker_coordinator.py:1276-1280` in a string, and those lines have moved. It is cosmetic, and that file
   is outside this stage.
5. **The S7 edit** of `persistent/pwc_protocol.py` (one line, `worker_id: str = ""`) is explicitly frozen by gate 22
   `:2418-2423` and by G-MINER-UNCHANGED `:550-551`. They are text tripwires, not miner behaviour. Committing S7 means
   you accept that those two gates assert something different (step-1 report §5).

## Not verified

- Real rigs, real ssh, real `pkill` on a CT100, and real `pwc_worker_service` processes. B5 ran in a local PID
  namespace with dummy processes; A3 ran on loopback with fake workers.
- Whether a stuck worker (driver D-state, L-6/L-7) clears in the 2 s wait. If it doesn't, the rig is refused for that
  run.
- Gate 22 and G-MINER-UNCHANGED on a committed tree.
- Any GPU path. The hybrid route is still quarantined.
- The yardstick. Nothing here measures hit rate, and PWC-credited pools have never been measured against `k/1000`.

## Files here

- `pwc_backup_stage1.diff`: `git diff 27eb991` plus the three new files. Verified: `patch -p1 --fuzz=0` on a fresh
  `git archive 27eb991` reproduces all 10 worktree files byte for byte.
- `HELD_pwc_worker_service.diff`: the held hunk, against `976a6db`. The file is unchanged since then.
- `before_after_failing_checks.txt` and `before_after_tally.md`.
- The Beta request: `docs/exchange/to_beta/X-PWC-STAGE1_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md`. Sent through the relay 2026-11-20 on the owner's approval.
- Raw logs: `logs/pwc_stage1_v1/` (ignored path). Restart notes: `logs/pwc_stage1_v1/RESUME.md`.
