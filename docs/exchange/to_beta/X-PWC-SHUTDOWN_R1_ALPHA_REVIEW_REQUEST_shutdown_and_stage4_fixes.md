# X-PWC-SHUTDOWN · R1 · ALPHA REVIEW REQUEST · PWC planned-shutdown exit, extractor record ordinal, runbook corrections

**Owner approval to relay:** 2026-11-21. The Alpha lead asked, "send the PWC shutdown fix to Beta now, together with the
extractor naming fix and the runbook corrections?" Owner, 14:39 PST, verbatim: **"Ok if you believe that's what needs
done"**. One round.

## Prior record

**Searches run** (`/bin/grep -rliE`, worktree `/home/michael/tfm_worktrees/pwc-backup`). Pattern
`pwc_worker_service.{0,80}(shutdown|reconnect|exit|leftover|orphan|reap)|shutdown reap|_MAX_REFUSED|RECONNECT_DELAY|S156.{0,40}reap|coordinator gone`:

| Scope | Files with hits |
|---|---|
| `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*` | 0 |
| `docs/PROPOSAL_*` | 1: `PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md` |
| `docs/CHAPTER_*`, `docs/*_AUDIT.md` | 0 |
| `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md` | 0 |
| `docs/exchange/` | 8 (PWC Stage 1/2/2B/3 and cache-reset rounds; read, the governing ones listed below) |
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md` | 0 |

Targeted reads: `docs/PROPOSAL_S96B_PERSISTENT_GPU_WORKERS_v1_0.md` (the PWC worker design) and
`docs/SESSION_CHANGELOG_20260419_S163KARG_P2.md` (the reconnect window).

**Documents that already decide or record the subject:**

| Document | What it records | Label |
|---|---|---|
| `docs/PROPOSAL_S96B_PERSISTENT_GPU_WORKERS_v1_0.md:64-70` | Worker lifecycle: "Shutdown: Parent writes `{"command": "shutdown"}` … worker exits cleanly" | ALREADY RECORDED. The intended planned-shutdown behaviour; the TCP worker did not follow it |
| `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:271-293` (C3; status "APPROVED WITH MODIFICATIONS — Team Beta 2026-03-25", `:5`) | "Per TB — shutdown must not rely solely on stdin pipe close": an explicit remote reap **added to `shutdown()`**, verifying no workers remain | ALREADY RECORDED. The code runs the S156 reap at **startup** only (`persistent_worker_coordinator.py:1051`, `:1100-1150`). The new shutdown reap implements C3 for the TCP route |
| `docs/SESSION_CHANGELOG_20260419_S163KARG_P2.md:71-76`, `:130` | Fix 8: `RECONNECT_DELAY_S` 2→5 s, `_MAX_REFUSED` 5→60, "workers now survive rig reboots (5min retry window)" | ALREADY RECORDED. The design for an **unplanned** coordinator loss; kept |
| `docs/exchange/from_beta/X-PWC-STAGE1_R1_BETA_REVIEW.md:80`; `persistent/pwc_worker_service.py:384-394` | Your X-PWC-STAGE1 R1 Q4: an explicit refusal is final, so the worker exits rather than retrying every 5 s forever | ALREADY RECORDED. The same principle applied to a planned shutdown |
| `docs/exchange/from_beta/X-PWC-CACHE-RESET_R2_BETA_REVIEW.md:5`, `:97-99` | Orphan reconnects after a coordinator crash recorded as a known limit (S156, VIR-1) | ALREADY RECORDED. Unplanned loss, unchanged except Q2 |
| `docs/exchange/from_beta/X-PWC-STAGE4_R2_BETA_REVIEW.md` (sha256 `5ff37679…84d0b8`) | Your R2: R2-Q5 THEORETICAL (late live-path guard on the generation files), Finding 1 (search-record citation), Finding 2 LIVE (the runbook's PWC launch block starts WATCHER after a params failure) | ALREADY RECORDED. All three are answered below (Q6, Q7) |
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:153-171`, `:196-217` | Search-record pins in effective terms, sessions as labels, chunk link tied to the search, configuration equal to the record before dispatch | ALREADY RECORDED. Now cited in the runbook (your R2 Finding 1); the pins-to-record binding stays undone |

**Labels for what this request raises:**
- **NEW:**
  - the worker's exit on a planned shutdown;
  - the `_had_session` timing;
  - the coordinator's shutdown reap on TCP;
  - the record-ordinal binding in the PWC extractor;
  - the runbook steps.
- **CHANGES A RULING:** none. The unplanned-loss window is kept, and S156 C3 is implemented as approved.

**Not searched:** the host, ser8 archives, the private repository, project files outside the repository, the rigs.

## The request

- **Review class:** before commit. Covers PWC worker code (`persistent/pwc_worker_service.py`, a rig file) and
  coordinator code (`persistent_worker_coordinator.py`), the PWC extractor (verifier-like), and the runbook's launch
  steps.
- **Base:** landed main `6eafad9`. Nothing in this request is committed.
- **Tree to review:** `/home/michael/tfm_worktrees/pwc-backup`.
  - Its git HEAD is `6d75f71`, and the `6eafad9` files are present there uncommitted.
  - **Compare each file below to `git show 6eafad9:<path>`.** `git diff HEAD` shows the whole Stage 4 tooling as
    untracked.
- **Scratch:** the `TMPDIR` the relay sets; please keep throwaway checks there.
  - The suites write under `<worktree>/logs/test_*` (git-ignored).
  - `tests/test_pwc_shutdown_reap.py` C3 runs a real `pkill -9 -f '[p]wc_worker_service'` on the host, after checking
    that no real PWC worker runs there; it skips otherwise.
- **Yardstick:** TFM is judged by whether its pools hit more often than `k/1000` on draws the sieve never saw. Engine
  equality and worker cleanup are engineering checks; nothing here says anything about hit rate.

## Context: the real Stage 4 session (2026-11-21, owner-approved)

| Item | Evidence |
|---|---|
| Run 1, RANGE-MINER (tree `~/stage4_run_miner`, `6eafad9`) | Credited; phase rows 0/0/13/2995; bidirectional 0; miner bundle written (agent 2's extractor, with receipt) |
| Run 2, PWC (tree `~/stage4_run_pwc`) | Credited (`credited: trials [0]`); `Results saved` 0/0/13/2995; generation published 0 rows, `Prior rows: 0` (`logs/stage4/pwc_run.log:21759-86295`) |
| Comparator on the real bundles (run by the Alpha lead) | `SAME WHERE COMPARED`, exit 3; 52 EQUAL, 0 DIFFERENT, 10 NOT COMPARED (the miner's per-phase lists, null by design) |
| PWC extractor at `6eafad9` | REFUSED: `phase 1: pass file …/window_opt_forward_12_25_t0.json is missing`; PWC had written `…_t1.json` (Q5) |
| PWC rig workers | After `[PWC-TCP] transport stopped` at 13:50:49 (`pwc_run.log:86243`), all 24 rig `pwc_worker_service` processes were still alive 2 minutes later; the Alpha lead stopped them by hand (Q1-Q4) |
| Run 1 attempt 1 | `DATASET_AUTHORITY_P0_5: FAIL BEFORE DISPATCH — the dataset is not verified on every node` (`~/stage4_run_miner/logs/gate12_20261121_132251.log:38`). The run tree's dataset path had never been provisioned to the rigs. The remedy was `scripts/provision_dataset_fleet.py` run from the run tree, by hand (Q7) |
| Both runs | Logged the non-fatal `New results format failed: Schema file not found: …/schemas/field_mappings.json` after credit and publication (Q7) |

## What changed (compare against `6eafad9`)

### Worker: `persistent/pwc_worker_service.py` (rig file; `f50d5489…` → `31927811…`)

- **Planned shutdown ends the process.** `shutdown` received in the job loop (`:427-432`), or while waiting for init
  (`:270-274`), records its reason in `self._shutdown_requested` (`:166-171`). `run_forever` then exits at the top of
  its loop (`:187-192`), and a `ConnectionError` raised after the flag is set exits the same way (`:219-221`).
  - **Before:** the job loop `break`ed and returned, `run_forever` reconnected every 5 s, and the worker exited only
    after 60 refusals (about 5 minutes). That fits the 24 workers still alive at 2 minutes.
- **A session counts from READY.** `_had_session = True` now sits right after `_send_ready()` (`:207-212`).
  - **Before:** it was set only when `_main_loop` returned normally. A session ended by a dropped connection
    (`ConnectionError` out of `_main_loop`) left it False, so the refusal count never ran and the worker retried every
    5 s forever, against the recorded 5-minute window.
  - **Unchanged:** a worker that never reached READY still retries indefinitely (W4), as before.
- The `_MAX_REFUSED` comment said "~2min"; it now says 5 minutes (`:180-185`; 60 × 5 s).

### Coordinator: `persistent_worker_coordinator.py` (VM101 only; not in the PWC parity pin)

- **New command builder** `pwc_shutdown_reap_command(grace_s=15)` (`:258-282`):
  - it waits up to `PWC_SHUTDOWN_GRACE_S` (15 s) for every `[p]wc_worker_service` to exit by itself;
  - it prints `PWC_SHUTDOWN_WAIT:<s>:<pgrep status>`;
  - then it runs the existing probe-checked stale-kill tail (`pwc_stale_kill_command`, `:214-226`) with the PWC
    pattern only. That tail SIGKILLs leftovers and prints `PWC_STALE_KILL_OK` only when a working probe finds none
    (exit 6 survivor, 7 unavailable).
- **`classify_shutdown_reap`** (`:284-297`) maps each result to exited, killed, survivor, unavailable or error.
- **Reap targets:** `_tcp_launch_workers` records each rig that got at least one worker, keyed to its exact ssh argv
  (`:1586-1587`), and each VM101 local worker's `Popen` (`:1621`).
- **`shutdown()` calls `_tcp_reap_after_shutdown()` right after `transport.stop()`** (`:1767`, `:1771-1815`):
  - local procs are waited for, then killed if still alive at 15 s;
  - rigs are reaped over SSH in parallel;
  - results go to `self.shutdown_reap` and the log: INFO for exited or killed, ERROR otherwise;
  - it never raises, because it runs in `run_trial_persistent`'s `finally` (`:2975`).

### Extractor: `scripts/extract_pwc_bundle.py` (`8eaede1f…` at `6eafad9` → `d62513ae…`)

- **Record ordinal, discovered and bound, never guessed** (`:176-211`, `:213-228`, `:513-525`, `:467-470`).
  - **Why the files are `_t1`:** the `_t<N>` suffix is the integration's record ordinal. `trial_counter` starts at 0
    (`window_optimizer_integration_final.py:3273`), is incremented before each trial (`:3296`), and is passed as
    `trial_number` (`:3316`), which names the pass files (`persistent_worker_coordinator.py:2808` etc. in this revision). Optuna, and the
    credit line, call the same trial 0.
  - **How the files are bound now:** every required phase must carry one shared N, and that N must agree with:
    - the chunk link's trial counts, which must be exactly one complete trial (1/0/0/0);
    - the NPZ `trial_number` rows;
    - with `--run-log`, the log's `Results saved` lines for each file.
  - **`--trial` is replaced by `--record-ordinal`,** which only asserts the discovered value.
- **Your R2-Q5:** the provenance and NPZ paths are resolved and live-checked inside `read_generation`, before their
  first read (`:137-146`).
- **The real re-extraction** of Run 2, with the record-ordinal fix (before the R2-Q5 change, with no other difference
  in the read path):
  - rc 0, ordinal 1, phase rows 0/0/13/2995, credit granted;
  - NPZ copied with sha256 `aecc5d50…7a43b`, equal to the artifact hash;
  - the run tree was hash-identical before and after.

### Runbook: `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md` (`c19b5864…` at `6eafad9` → `947d1471…`)

- **A2, `schemas/field_mappings.json` copied.** It is the only git-ignored file under `schemas/` (`.gitignore:41`).
  `ResultsManager` loads it (`core/results_manager.py:113-115`) only in the post-publication save
  (`window_optimizer_integration_final.py:3709`), which is caught (`integration/sieve_integration.py:333-336`).
- **A2b, dataset provisioning from each run tree:** `PYTHONPATH=. python3 scripts/provision_dataset_fleet.py`.
  - P0.5 freezes the dataset at the run tree's absolute path, and every node must hold it at that path
    (`miner/dataset_authority.py:855-859`; attempt-1 log `:33`, `:38`).
- **A6, `PYTHONPATH=.` on the search commands.** The search CLI imports `database_system` and `utils`
  (`agents/watcher_agent.py:815-816`) with no path insert; the only one is in the Step 1 path (`:2200`).
- **C3, your R2 Finding 2.** The PWC launch is one `if cd && validate && [ -n … ]; then nohup …; else REFUSED; fi`
  block between `C3-PWC-LAUNCH` markers.
- **C5 and C6:** the `_t1` explanation, and `--run-log`.
- **C5b:** check for leftover workers after the run, and stop them with the PWC-only reap command.
- **Anchor note:** the runbook's anchors into `persistent_worker_coordinator.py` use `6eafad9`'s line numbers. This change
  shifts lines after `:258` by up to about 93; the request's own coordinator anchors are the current ones.
- **New stop conditions 12 and 13:** a reap survivor or unobservable probe, and failed provisioning.
- **Prior record:** the search-record citation (your R2 Finding 1).

### Tests

| Suite | Result | What is new |
|---|---|---|
| `tests/test_pwc_shutdown_reap.py` (new) | 9/9 | **W1/W2** planned shutdown → process exits after one connect. **W3** loss after READY → exits after exactly 60 refusals. **W4** no session → keeps retrying. **W-MUTANT** git HEAD's worker fails W1 and W3. **C1** classification. **C2** `shutdown()`: stop, then three stubbed rigs reaped in parallel (exited, killed, ssh error logged) plus local wait and kill, no raise. **C3** the reap in real bash against `exec -a pwc_worker_service_dummy` processes (exits by itself → exited; lingers → killed). **C4** pattern is PWC-only and bracketed |
| `tests/test_extract_pwc_bundle.py` | 41/41 | O-* record-ordinal gates (`_t1`, `_t5`, ambiguous, split, asserted mismatch, NPZ mismatch, two-trial link, run log); B2-SYMLINK-LIVE-SIDECAR (your R2-Q5) |
| `tests/test_stage4_miner_launch.py` | 19/19 | C3-*: the runbook's launch block run verbatim with `nohup` stubbed. Missing, malformed and RANGE-MINER params and a bad `cd` start nothing; valid params reach `nohup` |
| Existing PWC suites, run sequentially | `test_pwc_stage1` 9/9; `test_pwc_backup_step1` 22/22 (2 HELD, a standing Stage 1 owner hold, `:825-830`); `test_pwc_backup_stage2` 26/26; `test_pwc_cache_reset` 21/21; `test_pwc_stage2b_anchor_phase` 35/35; `test_pwc_stage3a` 6/6; `test_pwc_stage3_r1_fixes` 4/4; `test_s172_phase5_d3_25_candidate_ingress` 13/13; `test_s172_resolved_execution_set` PASS | — |
| `tests/test_s172_threshold_propagation.py` | 3/5, 1 FAIL, 1 UNAVAILABLE | **Pre-existing, not from this change.** The same FAIL reproduces on a clean clone of `6eafad9`: `RangeMinerCoordinator.build_stripe_assign_payload() missing 1 required keyword-only argument: 'generator_phase'`, a RANGE-MINER harness. UNAVAILABLE: no dataset in this worktree |

The PWC suites before the R2-Q5 and C3 edits were run with the shutdown change in place. After those two edits only
`test_extract_pwc_bundle.py` and `test_stage4_miner_launch.py` were re-run, because they are the only affected files.

## What is NOT verified

- **The shutdown fix has not run on a rig.** It is tested on CPU only: the worker's network steps faked, ssh stubbed,
  and the reap shell run locally against dummy processes. No real TCP shutdown, real worker or real rig has exercised
  it.
- **The rig needs a redeploy first.** `persistent/pwc_worker_service.py` is in the PWC TCP parity pin
  (`scripts/pwc_parity_gate.py:107-123`), so the rigs need it redeployed and the PWC parity gate must PASS before the
  next PWC run. The coordinator change is VM101-only. `miner/` and all ten gate12-governed files are byte-identical to
  `6eafad9`.
- **The 15 s grace against real chunk times.** A worker busy mid-job when `shutdown` arrives finishes its chunk first.
  Anything still alive at 15 s is SIGKILLed and logged `killed`.
- **That provisioning creates the run-tree path on every rig.** The Alpha lead ran it by hand on 2026-11-21; this
  session did not.

## Questions (allowed answers: NO OBJECTION · OBJECTION · SUGGESTION · NEEDS EVIDENCE; LIVE/THEORETICAL on every OBJECTION)

- **Q1. Planned shutdown exits the worker process.** Is a coordinator `shutdown` message, received in the job loop or
  before init, the right trigger to end the process with no reconnect? Is anything lost, for example a rig worker that
  should survive a deliberate coordinator restart?
- **Q2. `_had_session` from READY.** Is counting a session from `_send_ready()` right, so that an unplanned loss after
  a real session gets the recorded 5-minute window and then exits, instead of retrying forever? Is keeping indefinite
  retry for a worker that never reached READY acceptable?
- **Q3. Can the 15 s reap and kill ever hit a worker that is not a leftover of this coordinator?** Alpha's reading
  is yes, in principle:
  - **What the reap does:** it runs only on rigs this coordinator launched at least one worker on, at the end of that
    coordinator's own `shutdown()`. But its pattern `[p]wc_worker_service` is host-wide. On those rigs it would
    SIGKILL any PWC worker, including one belonging to another PWC coordinator using the same rig at the same time (a
    second VM101 tree, or a partitioned run's other coordinator).
  - **Known cases:**
    - **Partitioned runs:** `node_allowlist` partitions (`persistent_worker_coordinator.py:2770`) split rigs
      between coordinators. Two partitions sharing one rig is not a configuration the runbook uses.
    - **Startup:** the existing startup stale kill has the same host-wide scope at every PWC start (`:1453-1478`).
    - **VM101:** local workers are reaped through their own `Popen` objects only, so they are exact.
  - Please say whether this is acceptable as is, should be restricted (for example to the worker ids or PIDs this
    coordinator launched, which `_tcp_launch_workers` reads back as `PID=` lines), or should refuse when another
    coordinator is live.
- **Q4. Can RANGE-MINER be affected?** Alpha's reading is no:
  - the reap pattern names only `[p]wc_worker_service`, never `range_miner_worker` or `sieve_gpu_worker`;
  - it runs only at a PWC coordinator's shutdown;
  - no RANGE-MINER file changed;
  - the runbook never runs the engines at once (C1 checks).

  Is there a path where a RANGE-MINER worker's command line matches, or where the reap runs during a RANGE-MINER run?
- **Q5. Extractor record ordinal.** Is discovering N from the files and binding it by the link's 1/0/0/0, the NPZ
  `trial_number` rows and the optional run log sufficient? Should `--run-log` be mandatory? With zero NPZ rows, as in
  the real run, the NPZ check is vacuous.
- **Q6. Your R2-Q5 and the extractor's other changes.** Does the pre-read live-path check close the symlink case you
  reproduced?
- **Q7. Runbook corrections.**
  - **Your R2 Finding 2:** does the C3 block now stop on every failure you listed?
  - **The new steps:** are A2 (`field_mappings.json`), A2b (provisioning per tree, a rig write), A6 (`PYTHONPATH=.`),
    C5 and C6 (`_t1`) and C5b (leftover check and PWC-only stop) correct and complete?
- **Q8. Tests.** Do the new gates test behaviour? Is any branch of the worker or reap change untested?

## Positions table (please fill in)

| Question | Position | Severity |
|---|---|---|
| Q1 Planned shutdown ends the worker | | |
| Q2 `_had_session` from READY | | |
| Q3 Can the reap hit a non-leftover worker | | |
| Q4 Can RANGE-MINER be affected | | |
| Q5 Extractor record ordinal | | |
| Q6 R2-Q5 pre-read live-path check | | |
| Q7 Runbook corrections, incl. R2 Finding 2 | | |
| Q8 Tests | | |

## Files under review (sha256, now)

| File | sha256 | vs `6eafad9` |
|---|---|---|
| `persistent/pwc_worker_service.py` (**rig file**) | `31927811fe39d14122b2afc6fa33ec926084ea059eabcc115143e855eb5c47d9` | changed (`f50d5489…`) |
| `persistent_worker_coordinator.py` | `3f3bd4cfad8d91254f709ae18e0540c7906cd9e2bece7dad6bd3faf93185a9ef` | changed |
| `tests/test_pwc_shutdown_reap.py` | `2ff1c5940e72276cbc046fb214e4a3b181c21d1ec4e82178c0e9b0efff62d644` | new |
| `scripts/extract_pwc_bundle.py` | `d62513ae82d6f259890e0f30c9a7532656ba48c2d5f9dfb2ca1f2009373e3dfc` | changed |
| `tests/test_extract_pwc_bundle.py` | `93f9c1f09b1f0e68d6a9b7dc53de73b5e1d4b5a71e49048a933f60a773cf35c1` | changed |
| `tests/test_stage4_miner_launch.py` | `2c608dad4627aeb0965ea694b3215f13fedd53a2b80322a775b051b71b904105` | changed |
| `docs/drafts/pwc_backup_stage4_v1/SESSION_RUNBOOK.md` | `947d14714eb59f4e00e675f62dd0fe9da15d7f243b5582386ae83f17ea9a2ee2` | changed |
