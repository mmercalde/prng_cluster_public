# CLAUDE CODE REPORT — STEP 1 CREDIT REFUSED ON A NON-EMPTY LEDGER `-wal`

Brief: `docs/CLAUDE_CODE_INSTRUCTIONS_STEP1_CREDIT_WAL_20261115.md`. Session: Claude Code on VM101, as
`michael`, worktree `/home/michael/tfm_worktrees/step1-credit-wal`, base `c47680a`
(`c47680a24a5634445b8548f5db00b0150feb17f4`), 2026-11-15. Nothing committed, nothing pushed.

## Prior record

**Searches run** (this worktree, base `c47680a`):

- `/bin/grep -lniE "non-empty -wal|nonempty wal|non-empty .?-wal|readonly_shm|immutable=1|last connection|wal_checkpoint"`
  and `/bin/grep -lniE "sampler.*(connect|lifetime|holds)|no_sweep_evidence|search-repeat-of|repeat_of"`
  over `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`,
  `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`,
  `docs/exchange/*.md`, `docs/exchange/*/*.md`. Every hit was read in context.
- The brief's own Prior record (its table is carried by reference), and the files it names as
  unread: `utils/step1_search_ledger.py` (`:634-830`, `:1044-1086`), `MinerLedger._conn`
  (`miner/range_miner_coordinator.py:1207-1213`) and every `stripes` write in that file,
  `tests/test_step1_search_record_stage1.py` (harness, `:140-500`, `:1190-1260`),
  `tests/gate_s172_prod_shape.py` (`:30-45`, `:134-230`, `:520-545`).
- `tests/test_s172_small_repairs_part3.py` in full (it tests this reader).
- `https://www.sqlite.org/wal.html`, fetched 2026-11-15 (SOURCE for SQLite behaviour quoted below).

**Documents that already decide or record this subject:**

| Document | What it settles |
|---|---|
| `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §A (`:24-45`) | Reading live SQLite: never plain `mode=ro`; `-wal` empty → `immutable=1`; `-wal` non-empty → do not open the original; if the content is needed, copy the database **and** sidecars while no writer runs, show the source stable across the copy, query only the copy |
| `docs/exchange/from_beta/X-FINDINGS-PLAN_R1_BETA_REVIEW.md:71-73` | Beta's origin of §A: `mode=ro` created `-wal`/`-shm`; for a non-empty WAL "use an already established consistent offline copy … performing recovery/query work only on the scratch copy" |
| `docs/exchange/from_beta/X-FINDINGS-PLAN_R2_BETA_REVIEW.md:28-35` | Beta: §A correctly implements that procedure |
| `docs/exchange/from_beta/X-0001_R1_BETA_REVIEW.md:68-74` (Q6) | Immutable reads need actual quiescence; THEORETICAL stale-read counterexample with a writer committing after the precheck |
| `docs/exchange/from_beta/X-MINER-STALL_R1_BETA_REVIEW.md:132` | Beta accepted the current reader: "Nonempty WAL deliberately means UNAVAILABLE/no credit; this is not a new proof of a consistent snapshot while an unrelated writer is active" |
| `docs/RUN4_LAUNCH_CHECKLIST.md` step 14 operator note; `:574` | `-wal` stays non-empty when the writer closes under an open `mode=ro` reader (throwaway test); retry route close + `--search-repeat-of` recorded as `[UNVERIFIED]` |
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:170`, `:301`; `X-STEP1-SEARCH_R1_BETA_REVIEW.md:117` | `repeat_of` is descriptive and grants nothing |
| `docs/exchange/RELAY_PROTOCOL.md` §3 (`:64-81`); implementation `scripts/beta_relay.py:247-283` | The relay snapshot boundary: repository files (including untracked and ignored) plus the configured live ledger's four files, not arbitrary `/tmp` contents. Governs §3 limit 2 and §8.2 (added after Beta R1 Q11) |

**Labels for what this report raises:**

| Item | Label |
|---|---|
| §2 points 1-3 of the brief, confirmed by execution | NEW (measurement of the brief's NEW reading) |
| §2 point 4 corrected: the `-wal` survives when a **read-only** connection closes last, so what decides is who closes last, not who is alive at credit time | NEW |
| A plain `mode=ro` open rewrites an existing `-shm` and creates an absent one | ALREADY RECORDED for creation (`X-FINDINGS-PLAN_R1:71`); the rewrite of an existing `-shm` is NEW (measured here) |
| The change: non-empty `-wal` → read a stable file copy, original never opened | Implements addendum §A item 4 inside Step 1 — **CHANGES A RULING in its effect on credit**: X-MINER-STALL R1 `:132` accepted "non-empty WAL = no credit". Needs the owner |
| `tests/test_s172_small_repairs_part3.py` G-EVID-WAL turns red under the change | NEW; it asserts the rule being replaced (§8.1) |
| Q2 retry route | ALREADY RECORDED as `[UNVERIFIED]` (checklist `:574`); now executed on a fixture |
| Q4: `gate_s172_prod_shape.py` section B has the same dependence | ALREADY RECORDED as design (`tests/gate_s172_prod_shape.py:39-41`); its consequence for Run 4 is NEW |
| Owner ruling 2026-11-15 12:41 PST (§0) | **NEW** (owner, in chat). Decides the CHANGES-A-RULING item above, the scratch location and part3 |

**Not searched:** the host, ser8 archives, the private repository, project files outside the
repository. The live ledger was not opened (listed with `ls` only); `prng_analysis.db` was not
touched.

---

## 0. Owner ruling — NEW (owner, in chat, 2026-11-15 12:41 PST)

Relayed by Alpha. Verbatim: *"im gong to trust you on this0 both you and beta concur so go
ahead-proceed"*. Per Alpha, it approves all three parts as Alpha put them to the owner:

1. **The narrow exception.** After terminal miner teardown, Step 1 may copy the ledger and its
   `-wal` to private scratch, require source stability, rebuild the index only on the copy,
   read the copy, and fail closed on any error. Addendum §A's outside-reader prohibitions stay
   intact. This decides the X-MINER-STALL R1 `:132` change (Beta R1 Q5 wording).
2. **The scratch copy may sit in system temp** (§3 limit 2; Beta R1 Q8).
3. **part3:** keep its fixture and its phase-1 safety assertion; reword only the blanket
   no-evidence requirement. Done in §4.


## 1. Result first

- **Run 4 attempt 1: 1 attempt, 0 credited.** Confirmed cause: the committed harness's sampler
  is connected to the ledger when the coordinator's last writing connection closes. A read-only
  last closer cannot checkpoint or delete the WAL, so the `-wal` stays non-empty, and the base
  reader refuses any non-empty `-wal`. **Under base, a run launched through the committed harness
  earns no credit whenever the sampler is still connected at the coordinator's last writing
  close,** as it was in Run 4. The sampler can stop earlier on its own (`--max-seconds 7200`, or
  180 s with no runnable stripe; `scripts/gate12_concurrency_sampler.py:1057-1059`, `:1140-1147`).
  Those exits can come before the last write, so "every run" is too strong (Beta R1 Q1).
- **The change** (`window_optimizer_integration_final.py` only): with a non-empty `-wal`, Step 1
  no longer refuses outright. It opens nothing on the original. It file-copies the database and
  its `-wal`, requires both unchanged across the copy, and reads the copy. That is addendum §A
  item 4, applied inside Step 1.
- **On a copy of today's frozen ledger, the new reader returns phases 1-4 each
  `[2147483648, 4294967296)`**, equal to the search interval. That says today's sweep would have
  been credited. It credits nothing.
- **Tests, final state (the tree proposed for landing):** every suite passes. Credit-WAL is
  12/12 gates and 5/5 mutants, part3 3/3 and 3/3, stage-1 39/39 and 33/33, seed_start 10/10 and
  11/11 (pass 3, §8a; part3 rerun after Beta R2, §9).
  *Pre-ruling history:* before the owner ruling, part3 G-EVID-WAL was red on the patched tree
  (base 3/3, patched 2/3), by design. §8.1 records why; §0 and §8a record the ruling and the
  edit.

## 2. Q1 — the brief's §2 reading, by execution

Fixtures used the real `MinerLedger`, the real sampler script as a subprocess (fixture B) or its
real `connect_ro`/`read_snapshot` (suite), and the real `_step1_miner_sweep_evidence`. SQLite
3.37.2, Python 3.10.12. Logs: `data/step1_credit_wal_q1/q1_run.log`, `q1_extra.log`.

| Point | Verdict | Evidence |
|---|---|---|
| 1. A non-empty `-wal` refuses credit | **Confirmed** | Fixture B, `-wal` 527,392 B → `UNAVAILABLE … no credit`, `None`. Control A (no reader): no `-wal`, all four phases returned |
| 2. The sampler holds a connection between samples | **Confirmed** | `/proc/<pid>/fd` showed `.db`, `-wal`, `-shm` open at every 0.5 s check through the 2 s sleep. By reading there is never a gap: `connect_ro()` opens the new connection before the rebind drops the old one (`scripts/gate12_concurrency_sampler.py:1098`) |
| 3. The sampler is alive when credit is decided | **Confirmed** | Run 4 TSV has OBSERVED rows to 09:20:04; S7 refusal at 09:19:50 (run log `:676`). No `--watch-pid` (`gate12_launch.sh:363-367`); quiesce is 180 s (`scripts/gate12_concurrency_sampler.py:1057`) |
| 4. The `-wal` is removed when the last connection closes | **Wrong as stated** | It is removed only if the last closer can write. After SIGTERM to the sampler, fixture B's `-wal` was byte-identical (527,392 B, same sha256) and a second credit read still refused. The live `-wal` was still present at 09:53. SOURCE wal.html §6: the last connection "does one last checkpoint and then deletes the WAL", which describes a connection that can write; the page is silent on a read-only last closer |

**Corrected point 4, and what it means for option B.** What decides is **who closes last, not
who is alive when credit is decided.** If the sampler (read-only) is connected at the instant the
coordinator closes its last writing connection, the `-wal` is permanent until some read-write
connection later opens and is the last to close. So:

- Stopping the sampler before credit does **not** fix B. The sampler would have to be stopped
  before the coordinator's **last write**. That is inside the trial, which the harness does not
  observe.
- Any read-only observer has the same effect: a dashboard, a person running `sqlite3 -readonly`,
  or the prod-shape verifier if it ever opened the ledger non-immutably.
- B therefore needs a write-side act, which is option C (an edit under `miner/`), or it stays one
  observer away from refusing a good run.

**Frozen ledger copy** (sha256 equal to `SHA256SUMS.txt`, 3/3 ledger files):

- The copied `-shm` shows mxFrame 71, nBackfill 0: 71 committed frames are not in the main file.
- An immutable read and a WAL-aware read agree on all 128 stripe rows of
  `distributed_config_t1_57dc50fa`. They **disagree on its `trials` row**: `commit_cleanup_status`
  is `none` under the immutable read and `done` under the WAL-aware read, matching
  "commit cleanup done" at 09:19:49. So the base reader's reason is real: an immutable read of
  today's ledger is stale. It is just not stale in the columns S7 reads.

## 3. The change

`window_optimizer_integration_final.py` (sha256 `4a98c471…91eb5cd` after the R1 docstring
correction; Beta R1 read `ceea5d67…caca09`), `git diff c47680a`: 86 insertions, 18 deletions,
in one function plus one new helper. Since R1 the only change is the helper's docstring
(`:1427-1439`).

| Case | Behaviour | Anchor |
|---|---|---|
| `-journal` non-empty | UNAVAILABLE, no credit (unchanged) | `:1509-1514` |
| `-wal` absent or empty | `immutable=1` on the original (unchanged) | `:1515-1523` |
| `-wal` non-empty | `_step1_ledger_rows_via_copy`; `None` → no credit | `:1524-1527` |
| Copy: private `tempfile.mkdtemp()`; `copyfile` of `<db>` and `<db>-wal` (not `-shm`); `(st_ino, st_size, st_mtime_ns)` of both, before and after; any difference → UNAVAILABLE; plain `mode=ro` on the copy; scratch removed in a `finally`, best effort (`rmtree(ignore_errors=True)`; a killed process leaves it) | `:1426-1476` |
| Any exception (corrupt file, no space) | caught by the existing handler → "unreadable", no credit | `:1529-1531` |
| Everything after the rows (state `done`, contiguous tiling, `credit_decision`) | unchanged | |

**Why not option A as the brief worded it** (a plain `mode=ro` read of the original). Measured on
copies (`data/step1_credit_wal_edge/edge.log`): plain `mode=ro` **rewrote an existing `-shm`**
(fixture B: same size, different hash) and **created an absent `-shm`**. Both break "writes
nothing into the ledger's directory".

**Why not `mode=ro&readonly_shm=1`.** It wrote nothing in any case I measured, and it failed to
open when `-shm` was absent, so that case would have earned no credit. But `readonly_shm` is not
documented on wal.html, and its behaviour is an implementation detail of the unix VFS. The copy
uses only documented behaviour, matches the procedure Beta already accepted, and never opens the
original.

**The `-wal`-present, `-shm`-absent edge (tested first, per Alpha).** Under the copy design
nothing is opened on the original, so no file can be created there. The `-shm` is never copied in
either case, and SQLite rebuilds the WAL index from the WAL's own checksummed frames on the copy.
So this case **is credited** when the rows show a complete sweep, and nothing is created (gate
G-NOSHM). Alpha's instruction was to refuse credit only "if the read cannot open without
creating a file". That condition does not arise under this design. I am stating it here
rather than refusing that case.

**What "writers have stopped" means here** (X-0001 R1 Q6). The reader runs at
`window_optimizer_integration_final.py:1939-1940`, after `run_trial_miner` has returned
`committed` for this `run_id`. The coordinator runs in the same process (its log lines are in
the Step 1 log) and opens per-operation connections (`miner/range_miner_coordinator.py:1207`).
Every `stripes` write is scoped `WHERE run_id=?` (`:1694`, `:2023`, `:2161`, `:2185`, `:2217`,
`:2233`, `:2325`). That scoping alone does not prove the rows are final: the generic setters at
`:1694` and `:2233` can update the selected run, so my earlier sentence "no path writes a
committed run's `done` rows" overstated it (Beta R1 Q4). The real premise, which Beta read, is
that the serve path has completed and its writers are drained. Coordinator teardown stops and
joins its readers and drains both executors with `wait=True` before it builds the terminal
return (`miner/range_miner_coordinator.py:10660-10684`, `:10733-10757`), under the one-coordinator
use. The read does not rely on that alone: a concurrent commit during the copy changes the `-wal`'s size or mtime
and is refused (gate R-MOVING). A commit that lands before the copy is read as committed state,
which is what it is.

**Limits, stated plainly:**

1. The stability check compares inode, size and mtime_ns. A writer that changes the bytes and
   restores both size and mtime_ns during the copy would not be seen. No such writer exists here.
   THEORETICAL.
2. The scratch copy goes in the system temporary directory (`tempfile.mkdtemp()`, `$TMPDIR` or
   `/tmp`). It is transient and removed in a `finally`, but only best effort: a killed process leaves
it behind, and a future `TMPDIR` set inside live staging would break the intended boundary (Beta
R1 Q8). It is not an output. Box rule 7 says outputs
   go under `data/`. The alternatives were `data/`, which is not gitignored for `-wal` names, so a
   crash would leave untracked files and red the clean-tree gate, or a `.gitignore` edit, which is
   outside this brief. **Owner's call** (§8.2).
3. Cost: one file copy of the ledger per credit decision (13.4 MB + 4.1 MB today). It grows with
   the ledger.
4. The `-wal`-empty branch keeps its existing pre-open race (X-0001 R1 Q6, THEORETICAL). It is
   unchanged and out of scope.

## 4. Tests (brief §4)

Suite `tests/test_step1_credit_wal.py` (new, sha256 `4382a3f5…2d05ff3` after the R1 Q7 change;
Beta R1 read `21454278…664036`). Every fixture is a
WAL-mode ledger built with the real `MinerLedger` API: one run, phases 1-4 × 8 stripes. A reader
opened with the real sampler's `connect_ro` is connected across the writer's close and closes
last, so `-wal` is non-empty and every row lives only in the WAL.

| # | Gate | Result |
|---|---|---|
| 1 | **G-RED** base reader (`c47680a`): `None`, UNAVAILABLE, `credit_decision` → `no_sweep_evidence` | PASS (red first) |
| 2 | **G-GREEN** new reader = committed rows (checked against an independent full-directory copy); all 4 phases credited; ledger directory byte-identical (names, sizes, sha256, mtime_ns); scratch removed | PASS |
| 2 | **G-NOSHM** `-shm` absent: credited, no file created | PASS |
| 3 | **R-PHASE-MISSING** / **R-NOT-DONE** (`claimed`) / **R-LIVE-WRITER** (part3's stale-positive shape, since R1: the complete baseline is checkpointed into the main file, so the immutable view shows phase 1 8/8 done; then a still-connected writer commits one phase-1 stripe back to `staging` in the WAL; the reader returns phases 2-4 only) / **R-GAP** → each `credit_decision` refuses, naming the phase | PASS ×4 |
| 3 | **R-NOT-COMMITTED** / **R-UNREADABLE** (header page zeroed) / **R-JOURNAL** → no evidence | PASS ×3 |
| 3 | **R-MOVING** source `-wal` grows during the copy → no evidence, reason printed, scratch removed | PASS |
| 4 | **G-STALE** immutable read of the same fixture sees **0** rows; the new reader returns all 4 phases | PASS |
| | Mutants M1 base reader → G-GREEN; M2 plain `mode=ro` on the original → G-GREEN (directory changed); M3 `immutable=1` on the original → G-STALE; M4 copy without the stability check → R-MOVING; M5 (added after R1) `immutable=1` on the original → R-LIVE-WRITER (credits phase 1) | 5/5 RED |

Before R1: `GATES 12/12  MUTANTS 4/4  COMPLETION SENTINEL: PASS`
(`data/step1_credit_wal/suites_pass2/step1-credit-wal__test_step1_credit_wal.log`). Beta found
that the R1 R-LIVE-WRITER still passed under an immutable-original mutant, because its fixture's
main file was empty. After the change: `GATES 12/12  MUTANTS 5/5  COMPLETION SENTINEL: PASS`
(`data/step1_credit_wal/suite_after_R1_suggestions.log`). part3 rerun on the same tree: unchanged,
2/3 with the same G-EVID-WAL failure (`data/step1_credit_wal/part3_after_R1_suggestions.log`). The
frozen check rerun is unchanged (`data/step1_credit_wal/frozen_run4_after_R1.log`). The stage-1
and seed_start suites were not rerun after R1: the only production change since pass 2 is a
docstring.

**Test 5, on a copy of the frozen ledger** (`--frozen /home/michael/run4_attempt1_20261115_090553/ledger`,
log `data/step1_credit_wal/frozen_run4.log`). For `distributed_config_t1_57dc50fa`:
phase 1 `[2147483648, 4294967296)`, phase 2 the same, phase 3 the same, phase 4 the same. The copy
directory was unchanged by the read and the scratch was removed. The bundle's sizes and mtimes
were the same before and after, and `sha256sum -c SHA256SUMS.txt` gave 78/78 OK. **This says today's
sweep would have earned credit. It credits nothing.**

**Test 6, existing suites, run sequentially** (`data/step1_credit_wal/suites_pass2/summary.txt`).
Patched = this worktree. Base = a detached worktree at `c47680a`,
`/home/michael/tfm_worktrees/step1-credit-wal-base`. Both got read-only copies (0444) of
`daily3.json`, `daily3-20260801T145551443433Z-513648160d35.json` (`513648160d35…68f6`) and
`daily3_current.json` (`ef327526…250a`), sha256 equal to main, as in
`docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md:86`. Without them pass 1 failed identically on both
trees with `PointerResolutionError` (`suites/summary.txt`).

| Suite | Base `c47680a` | Patched |
|---|---|---|
| `tests/test_s172_small_repairs_part3.py` | 3/3 gates, 3/3 mutants, PASS | before the ruling **2/3**, 3/3, FAIL — G-EVID-WAL (§8.1); after the §0 edit 3/3, 3/3, PASS (below) |
| `tests/test_step1_search_record_stage1.py` | 39/39, 33/33, PASS | 39/39, 33/33, PASS |
| `tests/test_s172_miner_seed_start.py` | 10/10, 11/11, PASS | 10/10, 11/11, PASS |
| `tests/test_step1_credit_wal.py` | (absent at base) | 12/12, 4/4, PASS |

These are the suites that reference the reader or the credit gate:
`/bin/grep -ln "_step1_miner_sweep_evidence|Step1NoCreditError|credit_decision|step1_sweep_evidence" tests/*.py`.
The fifth hit, `tests/capture_miner_seed_start_zero_control.py`, is a capture tool for another
commit, not a suite. Live checks before and after: `~/distributed_prng_analysis/optuna_studies/`
listing unchanged; `~/miner_staging/` entries unchanged; main tree `git status --porcelain` empty.
The non-fatal `schemas/field_mappings.json` FileNotFoundError seen in stage-1 logs is a gitignored
file absent from fresh worktrees. It appears identically on base.

## 5. Q2 — what the owner must do to run the upper half again

Dry run on a fixture database (`data/step1_credit_wal/q2/q2_dry_run.log`). It uses the stage-1
suite's own `integ_env` and `watcher_env`, nested so that WATCHER and the child share one
throwaway `prng_analysis.db`. WATCHER dispatch is intercepted, and the child is the real
`optimize_window` with the sieve stubbed. Params are Run 4's checklist JSON. **The fixture
search id came out as `s1s-884f7a8e305dde9d6ac4a3ed2cf347fc794e653ee17fe3c717ec637951f1a9dd`,
the live id**, so the fixture reproduces the live identity exactly.

| Step | Result |
|---|---|
| Attempt 1, trial COMPLETE without evidence (today's shape) | `Step1NoCreditError … no_sweep_evidence`; search stays `open`/`gap` at 2147483648; **study `window_opt_…` is now bound to the search** (`before_trial` → `chunk_open`, `window_optimizer_integration_final.py:1674-1684`) |
| Second launch, WATCHER layer | **Dispatches** the child (`--step1-search-id` = the same id, `--seed-start 2147483648`). WATCHER does not refuse |
| Child, fresh study | `Step1SearchPreflightError: … owns study 'window_opt_…'; this run would create a fresh study` (`:1636-1644`). Sieve calls: 0 |
| Child, `--resume-study --study-name <bound>` | `Step1ChunkBudgetError: S8: … already has 1 COMPLETE trial(s) and this chunk's budget is 1` (`:1567-1580`). Sieve calls: 0 |
| Child, `--resume-study` with no name | same S9 refusal as a fresh study. Sieve calls: 0 |
| `--step1-search-close <id>` | `STEP1_SEARCH_CLOSED` |
| Reopen without `--search-repeat-of` | `REFUSED: … has the same identity and is closed; … Open an explicit repeat` rc 3 (`utils/step1_search_ledger.py:679-687`) |
| Open with `--search-repeat-of <id>` | `STEP1_SEARCH_OPENED s1s-8e878039170386ab42d24e96eafe3c1ba0549a7f27399a5dd434c7c981b1df18`; status `gap`, 2147483648 |
| Launch on the new search, evidence present | WATCHER dispatches the new id; child credits `[(2147483648, 4294967296)]` |

**Answer.** A second launch on `s1s-884f7a8e…` refuses **in the child, before study creation or
any fleet action** (the preflight placement, `window_optimizer_integration_final.py:2486-2492`). It
does not refuse in WATCHER, so WATCHER launches the child and the child exits with the error.
Close, then open with `--search-repeat-of`, is the working route. `repeat_of` carries nothing
forward; the new search starts at its own first gap.

**Order.** The credit fix must be landed in the main tree first. Without it the repeat is
refused credit the same way (§2, §6), and that would use up a second search.

**Owner commands** (VM101, as `michael`, main tree, after the fix is landed; `RUN4_PARAMS` exactly
as `docs/RUN4_LAUNCH_CHECKLIST.md` step 11):

```bash
cd /home/michael/distributed_prng_analysis && source ~/venvs/torch/bin/activate
RUN4_PARAMS='{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 2147483648, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 6, "warm_start_skip_max": 99}'
PYTHONPATH=. python3 agents/watcher_agent.py \
  --step1-search-close s1s-884f7a8e305dde9d6ac4a3ed2cf347fc794e653ee17fe3c717ec637951f1a9dd \
  --search-reason "<your words: Run 4 attempt 1 uncredited, S7 non-empty -wal>" \
  --params "$RUN4_PARAMS"
PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-open \
  --search-range-start 2147483648 --search-range-end 4294967296 \
  --search-repeat-of s1s-884f7a8e305dde9d6ac4a3ed2cf347fc794e653ee17fe3c717ec637951f1a9dd \
  --search-reason "<your words: Run 4 retry, upper half>" \
  --params "$RUN4_PARAMS"
PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-status --params "$RUN4_PARAMS"
```

Expected output: `STEP1_SEARCH_CLOSED s1s-884f7a8e…`; `STEP1_SEARCH_OPENED s1s-<new> range=[2147483648, 4294967296)`;
status `"state": "gap"`, `"next_seed_start": 2147483648`, `"search_id"` = the new id. On the
fixture the new id was `s1s-8e878039…1df18`. The live id should be the same, because the identity
is deterministic over the same configuration, dataset, reason class and `repeat_of`
`[UNVERIFIED — prediction]`. Then launch as before; the launch script takes the search from WATCHER.

## 6. Q3 — will the next coordinator settle the live 4 MB `-wal` by itself? (by reading only)

**No, not in the sense that matters.**

- **The data is safe either way.** The `-wal`'s 71 committed frames are part of the database
  (SOURCE wal.html §4: "The WAL file is part of the persistent state of the database"). The next
  connection that opens the ledger recovers them.
- **They may or may not be checkpointed during the run.** The coordinator sets no
  `wal_autocheckpoint`, `journal_size_limit` or checkpoint (`miner/range_miner_coordinator.py:1207-1213`:
  only `journal_mode`, `busy_timeout`, `synchronous`), so the default applies. That default is a
  PASSIVE checkpoint attempted once the WAL reaches 1000 pages (SOURCE wal.html §3.1). It can be
  incomplete while readers hold snapshots, and the sampler is such a reader (SOURCE wal.html §6,
  checkpoint starvation). Whether a given checkpoint completes is conditional; nothing here
  measured it. The `-wal` file's length (Run 4's is 1007 frames) does not prove that any
  checkpoint progressed (Beta R1 Q10). None of this matters for reading the data: a WAL-aware
  connection uses the committed frames whether or not they have been checkpointed.
- **The `-wal` file is not removed or truncated at the first write, or at any write.** A
  checkpoint "does not normally truncate the WAL file" (SOURCE wal.html §6). It is deleted only
  when the last connection to close can write.
- **Under the committed harness that does not happen.** The sampler is armed before the
  coordinator (`gate12_launch.sh:353-369`) and always holds a connection (§2 point 2). It is
  stopped by the supervisor only after WATCHER exits (`:389-397`). So a run in which the sampler
  stays connected through the coordinator's last writing close, as in Run 4, ends with a
  non-empty `-wal`. The sampler's own `--max-seconds` or quiesce exits can come before that
  close (Beta R1 Q1); then the outcome depends on who closes last. With the fix, that no longer
  decides credit.

## 7. Q4 — `tests/gate_s172_prod_shape.py` section B (by reading; not run, not changed)

**Yes, the same dependence.** `open_ledger` returns UNAVAILABLE for any non-empty `-wal` and
never opens it (`:150-157`). `check_ledger` then records `B* miner ledger` as unavailable
(`:224-227`). For Run 4 today it would have reported `B* miner ledger` UNAVAILABLE:
`…/miner_ledger.db-wal is non-empty (4148872 B): … not opened`. The sentinel would have been at
best `INCOMPLETE (unavailable legs present)`, rc 2 (`:537-539`), and never PASS. Any other FAIL
leg would make it FAIL instead (`:533-536`). By §6 this recurs after every harness run in which
the sampler stays connected through the last writing close. In those runs checklist step 14's
prod-shape check cannot PASS. This change does not repair the verifier; its disposition is still
needed before a full production-shape acceptance is claimed (Beta R1 Q10).

## 8. For the owner

1. *(Pre-ruling history — resolved by §0 item 3 and the edit in §8a.)* **part3 G-EVID-WAL turned red, by design.** It asserted `ev is None` and `UNAVAILABLE` for a
   non-empty `-wal` (`tests/test_s172_small_repairs_part3.py:191-194`). That is the rule this
   change replaces. Its safety assertion still holds: the new reader returns
   `{'2': [1000, 1400]}`, so phase 1 is not credited
   (`data/step1_credit_wal/suites_pass2/step1-credit-wal__test_s172_small_repairs_part3.log:6`).
   My suite carries that fixture as R-LIVE-WRITER. At that point I had not edited part3;
   retiring or rewording G-EVID-WAL was left to the owner, who ruled (§0). Since R1, R-LIVE-WRITER also reproduces part3's
   stale-positive shape (§4), but Beta R1 Q7 recommends keeping part3's fixture and its phase-1
   safety assertion either way.
2. *(Resolved by §0 item 2: system temp.)* **Scratch location** (§3, limit 2): system temp, or under `data/` with a `.gitignore` entry.
3. *(Resolved by §0 item 1.)* **The ruling this changes in effect:** X-MINER-STALL R1 `:132` accepted "non-empty WAL = no
   credit". The change keeps addendum §A's purposes (nothing is written beside the original, and
   no stale read is made) by using §A item 4's copy. But it does so automatically, inside a
   counting run, where §A was written for a person reading by hand.
4. `--search-reason` on **open** is checked for being non-empty but is not passed to
   `open_search` (`agents/watcher_agent.py:852-878`). The record holds `opened_reason="operator"`.
   This is an observation only, by reading; I have not searched whether it is intended.
5. The base worktree `/home/michael/tfm_worktrees/step1-credit-wal-base` (detached, no ref) and
   the dataset copies in both worktrees are mine. They are left in place for Beta. Remove them
   when you choose.

Items 1-3 went to the owner as one ruling, through Alpha, with Alpha's recommendation. **Ruled
2026-11-15 12:41 PST (§0): all three approved as recommended.** Item 4 is an observation only
and is not part of the ruling. Item 5 is housekeeping.

### 8a. The part3 edit under the ruling

`tests/test_s172_small_repairs_part3.py`, `g_evid_wal` and the module docstring only:

- **Kept unchanged:** the fixture (a writer still connected, autocheckpoint off, phase-1 s0
  committed to `staging`), the stale-immutable control, and the phase-1 safety assertion
  `ev is None or "1" not in ev["swept_intervals"]`.
- **Reworded:** the blanket `ok_(ev is None)` and the `UNAVAILABLE` printout are replaced by two
  assertions. The committed state must be read as phase 2 only (`{"2": [lo, hi]}`), and the
  directory beside the ledger must be unchanged by the read.
- **Measured, and why the directory check compares content.** With only the blanket assertion
  reworded, mutant ME2 (the `7b9184a` plain `mode=ro` reader) was **NOT DETECTED**: with the
  writer connected it reads the WAL correctly, and part3's `_listing` compares only size and
  mtime. A probe showed that reader **rewrites the live `-shm` with the same size and mtime but
  a different sha256**, while the new reader changes nothing. So `g_evid_wal` now compares
  names, sizes and sha256 through a new `_content_listing`, and ME2 is RED on that write.
  `_listing` and G-EVID-OPEN are untouched.
- Result: **3/3 gates, 3/3 mutants, PASS**
  (`data/step1_credit_wal/test_s172_small_repairs_part3_after_ruling.log`).
- **After Beta R2 Q3:** `_content_listing` also compares `st_mtime_ns`. Beta's mtime-only
  counterexample (a correct reader that only bumps the database's mtime) now turns G-EVID-WAL red,
  reproduced here. The check detects any *persistent* difference in name, size, mtime_ns or
  content. A write undone between the two listings is not detectable, and none is claimed. part3:
  **3/3, 3/3, PASS** (`data/step1_credit_wal/test_s172_small_repairs_part3_after_R2.log`).

**Pass 3, the tree proposed for landing, sequential** (`data/step1_credit_wal/suites_pass3/summary.txt`):
part3 3/3, 3/3; stage-1 39/39, 33/33; seed_start 10/10, 11/11; credit-WAL 12/12, 5/5. All PASS.

## 9. Beta review

Request: `docs/exchange/to_beta/X-STEP1-CREDIT-WAL_R1_ALPHA_REVIEW_REQUEST_credit_on_nonempty_wal.md`.
Relay R1: `VERDICT: REVIEW RECEIVED (exit 0)`, 413.5 s, log `logs/beta_relay/X-STEP1-CREDIT-WAL_R1.log`.
Reply: `docs/exchange/from_beta/X-STEP1-CREDIT-WAL_R1_BETA_REVIEW.md` (read it; it is not restated here).

| Q | Beta R1 | Session disposition (after gate 3; Alpha authorised the in-scope items) |
|---|---|---|
| 1 | SUGGESTION | In scope (report wording): narrow "every run" to runs where the sampler stays connected through the last writing close (`--max-seconds`, quiesce exits). **Adopted** |
| 2, 3, 4, 6, 9 | NO OBJECTION | Recorded |
| 3 (inside) | — | In scope (comment): correct the helper docstring at `window_optimizer_integration_final.py:1434-1436` (uncommitted frames are excluded by the missing commit boundary, not by salt/checksum). **Adopted** |
| 5 | SUGGESTION | Owner disposition wording for the §A adaptation. **Ruled by the owner (§0 item 1)** in Beta's wording |
| 7 | SUGGESTION | In scope (my test file): R-LIVE-WRITER does not reproduce part3's stale-positive control, because its fixture has an empty main file. **Adopted**: R-LIVE-WRITER now checkpoints the complete baseline before the staging update and shows the immutable stale positive (phase 1 8/8 done); new mutant M5 reds it (§4). part3 then edited under the ruling (§0 item 3, §8a) |
| 8 | SUGGESTION | Report wording: cleanup is best effort, not unconditional; a `TMPDIR` inside staging would break the boundary. Wording **Adopted** (§3). Policy exception **ruled by the owner** (§0 item 2) |
| 10 | SUGGESTION | In scope (report wording): §6's checkpoint sentence made conditional (PASSIVE autocheckpoint can be incomplete while readers hold snapshots). **Adopted** |
| 11 | SUGGESTION | In scope: add `docs/exchange/RELAY_PROTOCOL.md` §3 (`:64-81`) to the Prior record. **Adopted** |

**R2.** Reply: `docs/exchange/to_beta/X-STEP1-CREDIT-WAL_R2_ALPHA_REPLY.md`. Relay R2:
`VERDICT: REVIEW RECEIVED (exit 0)`, 217.1 s, log `logs/beta_relay/X-STEP1-CREDIT-WAL_R2.log`.
Review: `docs/exchange/from_beta/X-STEP1-CREDIT-WAL_R2_BETA_REVIEW.md`.

| Q | Beta R2 | Session disposition |
|---|---|---|
| 1 | NO OBJECTION | Recorded |
| 2 | NO OBJECTION | Recorded |
| 3 | SUGGESTION | In scope (part3 test, within ruling item 3). **Adopted**: `st_mtime_ns` added to `_content_listing`; the claim is qualified to persistent differences (§8a) |
| 4 | SUGGESTION | In scope (this report). **Adopted**: §1 and §8 marked as pre-ruling history and updated to the final state |

**Not re-reviewed:** after R2, `tests/test_s172_small_repairs_part3.py` changed (the adopted Q3
suggestion) and so did this report (Q4). Beta R2 read part3 at `d4c93486…` and the report at
`bcdb7dca…`. One relay round, R3, remains in this thread if Alpha or the owner wants Beta to
confirm these two deltas. No production code changed after R2.

## 10. Files

| File | Status |
|---|---|
| `window_optimizer_integration_final.py` | modified (`:1426-1473` new helper; `:1476-1530` reader) |
| `tests/test_step1_credit_wal.py` | new |
| `tests/test_s172_small_repairs_part3.py` | modified (`g_evid_wal`, `_content_listing`, module docstring; owner ruling §0 item 3) |
| `docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md` | new (this file) |
| `docs/exchange/to_beta/X-STEP1-CREDIT-WAL_R1_ALPHA_REVIEW_REQUEST_credit_on_nonempty_wal.md` | new |
| `docs/exchange/to_beta/X-STEP1-CREDIT-WAL_R2_ALPHA_REPLY.md` | new |
| `docs/exchange/from_beta/X-STEP1-CREDIT-WAL_R1_BETA_REVIEW.md` (and R2) | written by Beta via the relay; never edited by the session |
| `docs/exchange/LEDGER.md` | modified by the relay only |
| `data/step1_credit_wal*/`, dataset copies | untracked evidence and environment; not for commit |

No commit, no push, no fleet launch, no search opened or closed. The live ledger, `prng_analysis.db`
and the bundle were not modified.
