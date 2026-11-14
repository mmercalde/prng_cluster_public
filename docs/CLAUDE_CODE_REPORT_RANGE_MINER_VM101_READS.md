# CLAUDE CODE REPORT — RANGE-MINER: TWO VM101 READS BEFORE RUN 4 (READ-ONLY) — 2026-11-13

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_RANGE_MINER_VM101_READS_20261113.md` (plan items A3, A4).
**Binding addendum:** `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md`. Where the two differ, the addendum governs. Section A was followed for every SQLite file.
**Anchor correction from Beta:** the Run-4 Step-1 command is at `docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:211-222`, with the warm-start flags on `:221-222`.
**Host:** VM101 (`zeus-ubuntu-vm`, 192.168.3.177), as `michael`, venv `~/venvs/torch`. Claude Code ran on **ser8** and drove VM101 over `ssh vm101`. Every command below ran on VM101.
**Worktree:** `/home/michael/tfm_worktrees/rm_reads`, detached at `30d94bd`. Every `file:line` anchor below was read at `30d94bd` in this session.
**Python:** always `PYTHONDONTWRITEBYTECODE=1`, and `-I` where no project import was needed. No `__pycache__` was created.
**Scratch:** `data/rm_reads/` in the worktree. It holds the before/after stats and listings, the reader scripts and the raw query output.

**What this session did not do:**
- It edited no existing file.
- It created no branch, tag or commit, and did not push.
- It launched nothing: no rig SSH, no GPU work, no gates, no tests, no finalizer, no coordinator, no WATCHER.
- It installed nothing.

---

## Summary

| Read | Result |
|---|---|
| **R1 cursor** | Live certified ledger `prng_analysis.db`, table `certified_coverage`: **one record**, `coverage_id c6f28aedf7af12cd…`, java_lcg `{constant,variable}`, `[0, 2147483648)`. Cursor for every java_lcg mode set: **OPEN, `next_seed_start = 2147483648`, covered `2147483648`, 1 interval**. **This matches the fact file's attempt-9 record.** No certified row for any other base, so every other base is OPEN at 0. |
| **R2 Run 4's range** | WATCHER **replaces** Run 4's `seed_start=0` with the cursor value whenever that value is `> 0` (`agents/watcher_agent.py:2087-2089`). With the ledger as read, Run 4 would sweep **`[2147483648, 4294967296)`**, the second half of the domain, as a **fresh study**. It would not repeat `[0, 2³¹)`, and it would not be refused. The Route-A dry run's recorded `--seed-start 0` came from a **stubbed cursor at 0** (`tests/run4_prelaunch_provenance_check.py:136-152`), not from the live ledger. |
| **R3 attempt-9 counts** | Generation `gen-20260818T013219550933Z-step1_java_lcg_0` was found under `.s172_accumulator/generations/`. Its sidecar records `raw_candidate_count=0`, `l2_winner_count=0`, `prior_row_count=0`, `final_row_count=0`, and no parent. The 22-array NPZ has **0 rows** in every array. Per-phase stripe `survivors_total` sums from the archived ledger: constant fwd **0**, constant rev **0**, hybrid fwd **59**, hybrid rev **23,515**. |
| **R4** | 12 items, listed and not cleaned. The ones that matter most for Run 4: (a) `agents/watcher_agent.py` changed in **two commits after** the Route-A certified digest; (b) the live miner ledger holds the **aborted 2026-08-22 run**. |

---

## Section A compliance record (every SQLite file opened)

**Process check.** No coordinator, WATCHER, window-optimizer, finalizer or gate12 process was running at any check:
- 19:10:07 PST, before opening `prng_analysis.db` (port 5700 also checked: not listening);
- 19:12:55, before opening the miner ledger (port 5700 also checked: not listening);
- once more before opening the archive. That third check was a process check only: no port test, no timestamp recorded.

The only Python process from the tree was `netconsole_listener.py` (pid 1126). Its open files are all under `logs/` (`logs/netconsole_listener.log`, `logs/netconsole_all_rigs.log`).

**Reader.** `data/rm_reads/read_sqlite.py` works as follows:
- It opens **only** `file:<abs>?mode=ro&immutable=1`.
- It refuses, without opening the original, if a non-empty `-wal` or any `-journal` exists.
- It runs plain SQL only.

`CoverageLedger` and `DistributedPRNGDatabase` were **never constructed**. That matters because `CoverageLedger.__init__` runs `init_schema` on a writer connection (`utils/seed_coverage_ledger.py:704-706`, `:709-729`), and the wrapper builds that object lazily (`database_system.py:342-355`).

| File | Sidecars at open (name, size, mtime) | Opened? | After |
|---|---|---|---|
| `/home/michael/distributed_prng_analysis/prng_analysis.db` (106496 B, mtime 2026-08-21 21:29:03.138, inode 14059495; `PRAGMA journal_mode` = `delete`) | none: no `-wal`, `-shm` or `-journal` | yes, immutable | stat identical. The repo-root `ls -la` diff shows only `..`, `docs/` and `.git/` changing. `docs/` and `.git/` (19:14:06) match the outside commit `ea85436` (R0). `..` (`/home/michael`, 18:57:21 → 19:11:41) matches the mtime of `~/.claude.json` (19:11:41.377). That is consistent with a Claude Code config rewrite by one of the three VM101 `claude` sessions; this session runs no process on VM101 that writes it. **Exact cause not determined** |
| `/home/michael/miner_staging/miner_ledger.db` (6434816 B, mtime 2026-08-27 18:12:59.989) | `-wal` **0 B**, mtime 2026-11-11 17:42:07.817; `-shm` 32768 B, mtime 2026-11-11 17:42:07.817. **Both predate this session** | yes, immutable (`-wal` is zero bytes) | directory listing and all three stats identical |
| `/home/michael/miner_ledger_archive/miner_ledger_pre_separation_9runs.db` (33099776 B, mtime 2026-08-27 18:13:00.009) | none | yes, immutable | recursive listing of the archive identical |
| `/home/michael/miner_ledger_archive/displaced_original/miner_ledger.db` (33099776 B, mtime 2026-08-17 18:32:19.116) | `-wal` **4148872 B**, mtime 2026-08-17 18:32:19.330; `-shm` 32768 B, mtime 2026-08-22 12:03:42.029 | **NO — UNAVAILABLE** (non-empty WAL, §A.4). No copy was made: the `9runs` copy held the attempt-9 rows needed | unchanged |

Nothing was checkpointed, vacuumed, deleted, renamed or repaired. The before/after files are in `data/rm_reads/`:
- `stat_before_*`;
- `*_dir_before.txt` and `*_dir_after.txt`;
- `rootdir_before_prng_analysis.txt`, `rootdir_after_prng_analysis.txt`, `rootdir_final.txt`.

**One session deviation, disclosed.** Before the worktree existed I wrote the two pre-session git snapshots (`refs_before.txt`, `status_before.txt`) to `/tmp/rm_reads_pre` on VM101. I then moved them into `data/rm_reads/` and removed that directory. Nothing else was written outside the worktree.

---

## R0 — State of the host

| Item | Reading |
|---|---|
| `git status --porcelain`, main tree | empty at session start and again at session end. **The first R0 read used a plain `git status`, not `--no-optional-locks`.** It may refresh `.git/index`, and `.git/index` read mtime 19:03:13 at the snapshot taken just after it. Whether that refresh was this session's is undetermined. Every later read used `--no-optional-locks` |
| `git log --oneline -3` at start | `30d94bd` (X-FINDINGS-PLAN R2) · `0066b6e` (G2/G3 gates) · `4b1e81e` (X-FINDINGS-PLAN R1) |
| **HEAD moved during this session** | `main` became `ea85436` at **19:14:06 PST**. Author `mmercalde`; message *"docs: WP-2 Stage A report, Step 6 to Chapter 13 grading contract"*; 1 file, +566 lines, docs only. `origin/main` and `public/main` moved with it. This session made no commit. **No anchor in this report is in that diff.** The worktree remains at `30d94bd` |
| `git worktree list` | main `30d94bd→ea85436` · `rm_gates` `fbd8264` · `wp1_stage_a` `0066b6e` · `wp2_stage_a` `0066b6e` · `rm_reads` `30d94bd` (this session) — all detached |
| Refs | the before/after `git for-each-ref` diff differs only in `refs/heads/main`, `refs/remotes/origin/main` and `refs/remotes/public/main` (the outside commit). **No ref was created** |
| Coordinator / WATCHER / window optimizer | none running (see Section A times) |
| Port 5700 | not listening |
| Other agent sessions | three `claude` processes: pid 5868 (started 18:08:49), 10900 (18:52:02), 11059 (18:52:50). **All three have cwd = the main tree** `/home/michael/distributed_prng_analysis`. Their open-file lists show only their own `/tmp/claude-1000/…/tasks` directories, so it cannot be seen which files they intend to edit. No `codex` process was found |
| Host boot | `uptime -s` = 2026-11-13 17:11:46 |

**Files in play vs the no-touch list.** The no-touch list is `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md` §0, plus `gate12_launch.sh` per addendum §B. The main tree has no files in play. Uncommitted entries in the other worktrees:

| Worktree | Entries | On no-touch list? |
|---|---|---|
| `rm_gates` | `docs/CLAUDE_CODE_REPORT_DELEGATED_RUN_GATES.md`, `scripts/env_freeze_gate.py`, `scripts/run_authorization_gate.py`, `tests/test_env_freeze_gate.py`, `tests/test_run_authorization_gate.py` (all untracked in that worktree; the gate scripts already exist in main at `0066b6e`) | **no** |
| `wp1_stage_a` | `data/` | **no** |
| `wp2_stage_a` | `data/`, `docs/CLAUDE_CODE_REPORT_STEP6_CH13_GRADING_CONTRACT_STAGE_A.md` | **no** |
| `rm_reads` (this session) | `data/`, and this report | **no** |

None of these is a no-touch file, and none modifies one. I am listing them as R0 asks. Whether they warrant a stop under §0's last row is for Michael.

**Anchor drift since `26d3147`:**
- The brief's anchors still hold at `30d94bd`: `seed_coverage_ledger.py:436`, `watcher_agent.py:2073` (inside the COMPLETE block `:2068-2085`), `database_system.py:428` (inside the cursor log block `:420-433`), `run_finalizer.py:1624`.
- `agents/watcher_agent.py` changed after the Run-4 certification; see R4-1.
- `tests/gate_s172_prod_shape.py:137-162` (cited in the addendum) was **not re-read** this session.

---

## R1 — Coverage cursor (A3)

**Ledger location.** This is a code path, not an assumption:
- WATCHER builds `DistributedPRNGDatabase()` with no arguments (`agents/watcher_agent.py:2036`).
- Its default is `db_path="prng_analysis.db"` (`database_system.py:46`), relative to the process cwd. `gate12_launch.sh:377` launches from the repo root.
- The ledger therefore lives in the table `certified_coverage` (`utils/seed_coverage_ledger.py:139`).
- That table is in `/home/michael/distributed_prng_analysis/prng_analysis.db`: 106496 B, mtime 2026-08-21 21:29:03.138 −0700, gitignored by `.gitignore:136` (`*.db`).
- **Search:** `find /home/michael -xdev -name 'prng_analysis.db*'`, excluding `tfm_worktrees`. **One hit.**

**Raw record** (the table's only row; `SELECT * … ORDER BY seed_start, seed_end_exclusive, coverage_id`):

```
coverage_id          c6f28aedf7af12cde2a03595ac7647d8d7b30d6c98b3e7f6aae0841df424e030
run_id               step1_java_lcg_0          (finalizer run id; the miner run is distributed_config_t1_554463d3)
prng_base            java_lcg
skip_modes_executed  constant,variable
seed_domain_contract v1.1-stratum
seed_start           0
seed_end_exclusive   2147483648
dataset_sha256       513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6
repository_commit    e9ca800ae65b44bc555f1402a9102932ba2e72ca
artifact_sha256      aecc5d504784ecca3d5766ec29703bf2b387fdb3191f6d133bb4c7d709e7a43b
generation_id        gen-20260818T013219550933Z-step1_java_lcg_0
publication_status   CERTIFIED
recorded_at          2026-08-18T01:32:19.577968+00:00
```

**Cursor computation.** The rows above were fed to the production pure functions with no database open:
- `canonical_coverage_identity` (`:271-320`);
- `CertifiedInterval.covers_modes` (`:634-640`, containment `requested ⊆ executed`);
- `first_uncovered_seed` (`:551-602`), the same function `certified_cursor` calls (`:1029-1055`).

Script: `data/rm_reads/cursor_from_rows.py`. Output: `data/rm_reads/cursor_result.txt`.

| Query (as WATCHER would form it) | Base / required modes | Qualifying records | Normalized union | Status | `next_seed_start` | `covered_seed_count` | Intervals |
|---|---|---|---|---|---|---|---|
| `java_lcg`, `test_both_modes=True` (Run 4) | java_lcg / {constant, variable} | `c6f28aed…` | `[[0, 2147483648]]` | OPEN | **2147483648** | 2147483648 | 1 |
| `java_lcg`, `test_both_modes=False` | java_lcg / {constant} | `c6f28aed…` | `[[0, 2147483648]]` | OPEN | 2147483648 | 2147483648 | 1 |
| `java_lcg_hybrid` | java_lcg / {variable} | `c6f28aed…` | `[[0, 2147483648]]` | OPEN | 2147483648 | 2147483648 | 1 |
| any other base | — | none (the table has no other row) | `[]` | OPEN | 0 | 0 | 0 |

Domain: `[0, 4294967296)`, printed from the imported `SEED_DOMAIN_START` and `SEED_DOMAIN_EXCLUSIVE_MAX`. In this case `next_seed_start` equals `covered_seed_count`, because the union is one interval starting at 0. That equality does not hold in general (addendum C-R1).

**Against the fact file.** The fact file's attempt-9 record is `coverage_id c6f28aedf7af12cd`, `covered_seed_count=2147483648`, `certified_interval_count=1`. **It MATCHES.** Nothing has been certified since: the table has exactly one row.

**Legacy tracker, kept separate (zero certified authority).** `exhaustive_progress` has 15 rows:
- 14 from April–May 2026, spanning `[1073741824, 16106127360)`;
- `step1_java_lcg_0` = `[0, 1000)`, `last_updated 2026-08-21T21:29:03.130794`. That is the write behind the db's mtime.

`step1_trial_history` also holds 7 rows dated 2026-08-21 16:11–16:12, run `step1_java_lcg_0`, seed range `[0, 10000000)`, all with 0 survivors. **Some Step-1 activity on 2026-08-21 wrote legacy telemetry and certified nothing.** Neither table is read by the cursor (`seed_coverage_ledger.py:699-701`).

---

## R2 — Run 4's seed range (A3)

### Which command Run 4 is

**Not the Gate-12 harness.** `gate12_launch.sh:377-384` passes `--params` carrying `use_persistent_workers`, `use_range_miner`, `worker_pool_size`, `seed_start`, `max_seeds`, `miner_stripe_size`, `test_both_modes`, `prng_type`, `window_trials` and `n_parallel`. It carries **no warm-start keys**.

**The Run-4 operator command** is the Route-A pinned WATCHER invocation that `tests/run4_prelaunch_provenance_check.py:182-187` models:

```
python3 agents/watcher_agent.py --run-pipeline --start-step 1 --end-step 1 \
  --params '<CONTROL_PARAMS + seven warm_start_* keys>'
```

`CONTROL_PARAMS` comes from `tests/fixtures/run4_routing_clean_control.py:37-48`: `prng_type java_lcg`, `seed_start 0`, `max_seeds 2147483648`, `miner_stripe_size 67108864`, `worker_pool_size 25`, `test_both_modes true`, `use_range_miner true`, `use_persistent_workers false`, `window_trials 1`, `n_parallel 1`.

WATCHER moves the seven pin keys into the invocation-local authority channel (`agents/watcher_agent.py:648-683`, called at `:3924`). It then merges them into `final_params` after the manifest defaults (`:1887-1900`).

The built Step-1 command is recorded verbatim at `docs/SESSION_CHANGELOG_20260828_RUN4_ROUTING_PATCH.md:213-223` (Beta's anchor `:211-222`). **The seven pinned values** are on `:221-222`:

```
--warm-start-window 12  --warm-start-offset 25  --warm-start-session-idx 1 (midday)
--warm-start-fwd-thresh 0.71  --warm-start-rev-thresh 0.47
--warm-start-skip-min 6  --warm-start-skip-max 99
```

No launch harness for Run 4 exists in `scripts/`. **Search:** `ls scripts | grep -i -E 'run4|gate12'` and `grep -rli 'run4\|run-4\|run 4' scripts/ *.sh`. Besides the `gate12_*` gates, the search hit **`scripts/run_authorization_gate.py`** (G3, from `0066b6e`). Its hits were read:
- `:11-16`: authorization schema example `{"run_class": "RUN4", …, "seed_start_min", "seed_end_max_exclusive", …}`;
- `:20-24`: a launch must state `seed_start` and `seed_end_exclusive`, and `seed_end_exclusive == seed_start + max_seeds`;
- `:36`: refuse if the seed range is not inside the authorized range;
- `:88`: a comment citing "RUN4 §1".

It is a launch-refusal gate, **not a launcher**. `config_manifests/run_authorizations/` does not exist, so **no RUN4 authorization file exists**. `grep -rn run_authorization_gate --include=*.sh --include=*.py .` outside `tests/` and the script itself finds **no caller**, so it is wired into no launch path. It encodes no Run-4 seed value.

### The effective start, traced at `30d94bd`

1. `final_params = {**default_params}`, then declared operator keys are merged (`agents/watcher_agent.py:1877-1885`). The manifest declares `seed_start: 0` (`agent_manifests/window_optimizer.json:265`), so `'seed_start' in final_params` is **always true** for Step 1. The S145 block is therefore always entered (`:2027`).
2. The cursor comes from `DistributedPRNGDatabase().get_certified_cursor('java_lcg', test_both_modes=True)` (`:2036-2054`). It canonicalizes to `('java_lcg', {constant, variable})` (`database_system.py:382-385`), which R1 found to be **OPEN, 2147483648**.
3. It is not COMPLETE, so the refusal branch (`:2068-2085`) is not taken.
4. `_next_start = 2147483648 > 0`, so `final_params['seed_start'] = 2147483648` (`:2087-2089`). **The operator's explicit `seed_start: 0` is overwritten unconditionally.** No branch keeps an operator-supplied start when the cursor is above zero.
5. `study_name` is `''` (manifest default `:257`; not supplied by Run 4), so `resume_study=False` and the study is fresh (`:2094-2101`).
6. Pre-dispatch wall: `assert_seed_domain_preflight(2147483648, 2147483648)` (`:2131-2135` → `utils/seed_coverage_ledger.py:355-417`). Start `< 2³²`, and end `= 4294967296` is not `> 2³²` (`:408`), so it **passes**.

**Trace limit.** The WATCHER side is traced to the built argv. On the Step-1 side, `window_optimizer.py:1510` (`--seed-start`, default 0) → `:1854` `seed_start=getattr(args,'seed_start',0)` → `:887` `coordinator.optimize_window(seed_start=seed_start, …)` was read. Below `optimize_window` (stripe planning in the integration layer and coordinator) was **not traced** this session.

**Result.** With the ledger as read, the committed Run-4 command would search **`[2147483648, 4294967296)`**: 2³¹ seeds, 32 macro-stripes of 2²⁶ per stage, as a fresh Optuna study. It would **not repeat `[0, 2³¹)`** and would **not be refused**. If it published successfully, the certified union would become `[0, 2³²)`, and the next Step-1 request for java_lcg `{constant, variable}` would return `SEED_DOMAIN_COMPLETE` and generate no run (`:2068-2085`). That last step is traced from code only.

**Why the dry run did not show this.** The Route-A pre-launch check replaced `database_system` with a stub whose cursor returns `next_seed_start = 0` and `certified_interval_count = 0` (`tests/run4_prelaunch_provenance_check.py:136-152`). Its recorded `--seed-start 0` (changelog `:217`) therefore reflects the stub, not live state.

### Run-4 configuration identity (read this session)

| Item | Value |
|---|---|
| Manifest | `agent_manifests/window_optimizer.json` sha256 `cb901d6bb64af2a1d2b81475dd8902bf58ede866a35c6d889a3d41770182b3d7` (main tree = worktree; tracked) |
| Config | `distributed_config.json` sha256 `ac4ba07ca3f35d9042521b67a20875a37115f90ddd34a6eb0f71ce2de51e9192` (tracked) |
| WATCHER at HEAD | `agents/watcher_agent.py` sha256 `bff9e22b488a7ead…`; **not** the Route-A certified `d14bcb3b…` (see R4-1) |
| Dataset | `lottery_file daily3.json` (manifest `:249`). Pointer `daily3_current.json` → `daily3-20260801T145551443433Z-513648160d35.json`, sha256 `513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6`, 1380711 B, 18,068 records, `2000-01-01 evening` → `2026-02-26 midday`, lineage `daily3-combined-L001`. `daily3.json` hashes identically. **Same `dataset_sha256` as attempt 9's certified record** |
| Session | midday. The mapping `session_idx 1 = midday` is from `docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` §2 (citing `window_optimizer.py:1514-1528`). **That code was not read this session.** Attempt 9's `trial_context.sessions_json` = `["midday"]` was read |
| Window | `window_size 12`, `window_anchor 25`. Residue window = midday-filtered records `[25, 37)`, derived from the anchor contract in the fact file / window-anchor proposal; **residue construction code was not read this session**. Attempt 9's `trial_context.residue_sha256` = `c761272da958ba34699d612727f7a6a2aecf08eccd570dc521257964a145c008`. **Run 4's residue hash was not computed this session.** It is produced at run time |
| Effective seed interval | `[2147483648, 4294967296)` (traced above) |

---

## R3 — Attempt-9 published generation and counts (A4)

**Directories searched:**
- repo root `/home/michael/distributed_prng_analysis/`, including `.s172_accumulator/`, `.s172_accumulator/generations/` and `.s172_checkpoint/`;
- `~/miner_staging/`, `~/miner_ledger_archive/` (recursive) and `~/miner_ledger_archive/displaced_original/`;
- `find /home/michael -xdev -type d -name 'gen-2026081*'`, excluding `tfm_worktrees`;
- `find /home/michael -xdev -name '*.npz' -newer daily3_current.json`. Its hits are in `liveportrait-venv` and `piece_matcher_project`, unrelated.

Also present and **not opened**: `~/gate12_attempt3_20260810_200824`, `~/gate12_attempt4_*`, `~/gate12_attempt5_20260812_190715`, `~/gate12_attempt9_forensic_bundle_20260817.tar.gz`, `~/dashboard_work`.

**Found.** `.s172_accumulator/generations/gen-20260818T013219550933Z-step1_java_lcg_0--6af51dd27b7b7729dc7ddde61ed842d20223e8c76ba7b5d50e7d31f94328554b/` (mtime 2026-08-17 18:32:19):
- `.s172_accumulator/current` points to it.
- The root `bidirectional_survivors_{all,binary}.npz` are symlinks to `.s172_accumulator/current/…`.
- This is consistent with `generations_dir = accumulator_root / GENERATIONS_DIRNAME` (`utils/run_finalizer.py:1624`).
- The 2026-11-12 scoping report's "no `gen-*` in the repo root" is consistent with this: the generation is under the accumulator root, not the repo root.

**Sidecar identity** (`provenance.json`, 1161 B):

| Field | Value |
|---|---|
| `generation_id` | `gen-20260818T013219550933Z-step1_java_lcg_0` |
| `run_id` | `step1_java_lcg_0` |
| `artifact_sha256` | `aecc5d504784ecca3d5766ec29703bf2b387fdb3191f6d133bb4c7d709e7a43b`; equals the sha256 of both NPZ files (hard links, inode 16142329) and the ledger's `artifact_sha256` |
| `repository_commit` | `e9ca800ae65b…`; `repository_tree_clean: true` |
| seed interval | `[0, 2147483648)`, contract `v1.1-stratum`, `skip_modes_executed ["constant","variable"]` |
| `canonical_map_hash` | `55686c6454d8b4ce…` |
| parent | `parent_generation_id null`, `parent_artifact_sha256 null`, `parent_sidecar_sha256 null` |
| **`raw_candidate_count`** | **0** |
| **`l2_winner_count`** | **0** |
| **`prior_row_count`** | **0** |
| **`final_row_count`** / `row_count` | **0** / 0 |

These four counts are written by `utils/run_finalizer.py:1696-1699`:
- `l2_winner_count = len(winners)` from `_select_l2_winners(raw_candidates)` (`:1613`);
- `prior_row_count` = the parent generation's `seeds` length (`:1659`);
- `final_row_count` = rows after the L3 merge with the prior (`:1640-1642`, `:1660`).

**NPZ.** `bidirectional_survivors_all.npz`, 4398 B, read with `np.load(..., allow_pickle=False, mmap_mode="r")`. Since `.npz` is a zip, `mmap_mode` does not apply to its members; the file was only opened for reading. It has **22 arrays and every one has shape `(0,)`**:

`seeds` · `forward_matches` · `reverse_matches` · `window_size` · `offset` · `trial_number` · `skip_min` · `skip_max` · `skip_range` · `forward_count` · `reverse_count` · `bidirectional_count` · `intersection_count` · `intersection_ratio` · `intersection_weight` · `bidirectional_selectivity` · `forward_only_count` · `reverse_only_count` · `survivor_overlap_ratio` · `score` · `skip_mode` · `prng_type`

The stat was identical before and after.

**Forward-only, reverse-only, bidirectional: how each is defined.**
- Canonical records exist only for the forward/reverse **intersection**: `both = fwd_set & rev_set` (`utils/canonical_records.py:209-211`).
- `forward_only_count` and `reverse_only_count` are aggregate trial-level metadata columns, not row classes.
- So the published rows (0 of them) are by construction bidirectional records. **Forward-only and reverse-only populations cannot be read from rows, and are not inferred from them here.**

**Per-mode attempt totals.** Source: `miner_ledger_archive/miner_ledger_pre_separation_9runs.db`, run `distributed_config_t1_554463d3`; `trials.state = committed`, `commit_delivery_status = done`, `commit_cleanup_status = done`. Each total is the sum of stripe `survivors_total`, the per-stripe worker-reported population for that direction and mode:

| Phase | Family | Stripes | Seeds covered | Σ `survivors_total` |
|---|---|---:|---|---:|
| 1 | java_lcg (constant, forward) | 32 done | `[0, 2147483648)` | **0** |
| 2 | java_lcg_reverse (constant, reverse) | 32 done | `[0, 2147483648)` | **0** |
| 3 | java_lcg_hybrid (variable, forward) | 32 done | `[0, 2147483648)` | **59** |
| 4 | java_lcg_hybrid_reverse (variable, reverse) | 32 done | `[0, 2147483648)` | **23,515** |

Trial context, same ledger: `window_size 12`, `offset_val 25`, `sessions ["midday"]`, `skip 6–99`, `τ 0.71 / 0.47`, dataset `513648160d35…`, residue `c761272da958ba34…`.

**Linking the generation to run `distributed_config_t1_554463d3`:**
- The sidecar `repository_commit` `e9ca800` is the attempt-9 launch commit.
- The sidecar's `[0, 2147483648)` and `{constant, variable}` match that run's stripes.
- Trial `finalized_at` 1787016727.44 = 2026-08-18T01:32:07.44Z, about 12 s before the sidecar's `created_at` 2026-08-18T01:32:19.55Z.
- The per-phase sums 59 / 23,515 equal the "attempt 9 actual" column of `docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` §1.1.
- **The sidecar does not name the miner `run_id`.** The link is by these coincidences, not by a recorded key.

**What the sidecar adds to the ledger.**
- The ledger alone says each direction's population; the sidecar adds that **the publication received 0 raw bidirectional candidates**.
- *If* `raw_candidates` is the union of the per-mode records built at `canonical_records.py:209-211`, then the constant-mode intersection (0 ∩ 0) and the variable-mode intersection (59 forward vs 23,515 reverse) were both empty at publication. **The hop from `canonical_records` to the finalizer's `raw_candidates` argument was not traced this session.** So this is conditional, and the per-mode intersection sizes are reported as UNAVAILABLE below.
- The coordinator log line for the per-mode intersections was not read.

**Observation, with its limits** (lead-handling skill read before writing this; CLAUDE.md yardstick applies):
- Attempt 9 published **0** bidirectional records over `[0, 2³¹)` for java_lcg at `k=12, anchor 25, midday, τ 0.71/0.47`.
- Alpha's 2026-11-11 simulation estimate of about 0.01 bidirectional survivors at attempt-9 settings is **not a repository fact**. It is in a claude.ai project file that is not in the repo, and its inputs (which were measured on this data and which were assumed) and its count definition were not available to this session.
- A comparison is therefore valid only if that estimate counts per-mode forward∩reverse intersections at these settings and this seed range. **That has not been established here.**
- This is a null observation. It is not called chance and it is not called a finding.
- For the TFM yardstick it is a fact about survivor volume at one geometry. It says nothing by itself about pool hit rate.

**Not reported as zero, because unavailable:**
- per-mode intersection sizes separately (only their combined total is in the sidecar);
- `displaced_original/miner_ledger.db` (non-empty WAL, not opened).

---

## R4 — Anything else Run 4 would trip on (listed, not cleaned)

1. **WATCHER changed after the Route-A certification.** The certified Run-4 routing digest is `agents/watcher_agent.py` sha256 `d14bcb3b33958770…`, at commit `e75a6eb`, per changelog `:4-7` and `:183`. HEAD has `bff9e22b488a7ead…`, after two later commits:
   - `b6cb789` (2026-11-11, "failure authority, dry-run refusal, single halt authority"; digest `1be822b31d480baa…`);
   - `cf9edaa` (2026-11-11, "seed-domain-complete is a clean stop…").

   `e75a6eb` is an ancestor of HEAD. The Route-A pre-launch provenance check (10/10) was run against `d14bcb3b…`, not the current file, and was **not re-run** here.
2. **The pre-launch check models the cursor as 0** (`tests/run4_prelaunch_provenance_check.py:136-152`). Its built argv does not show the seed interval Run 4 would actually receive (R2).
3. **The live miner ledger holds an aborted run.** `~/miner_staging/miner_ledger.db` (mtime 2026-08-27 18:12:59) contains only run `distributed_config_t1_8cb116da` (stripes created from 2026-08-23 01:23 UTC):
   - trial `state aborted`, `terminal_class compute_lease_expiry`, terminal stripe `…__st3_s5`, worker `rrig6600:gpu4`, attempt 1, phase 4;
   - phase stripe sums: 718 / 0 / 13,146,485 / 2,817, with phase 4 at **8 cancelled, 24 done**.

   How a new run interacts with a ledger holding another `run_id` was **not traced**.
4. **Sidecars on the live miner ledger.** `miner_ledger.db-wal` (0 B) and `-shm` (32 KB), both mtime **2026-11-11 17:42:07**, are later than the db's own mtime (08-27). They were created before this session; who created them is not determined here.
5. `~/miner_staging/threshold_provenance.json` (512 B, mtime 2026-08-22 18:42:04), left from the aborted run.
6. The `~/miner_staging/` directory mtime is **2026-11-12 20:21:57**, so an entry was added or removed there on 2026-11-12. Its cause is not determined.
7. `~/miner_staging/miner_ledger.db.pregate12_20260809`, plus `-wal` (0 B) and `-shm` copies of it.
8. `~/miner_ledger_archive/displaced_original/miner_ledger.db` with a **4,148,872 B `-wal`** (2026-08-17 18:32:19). It was not opened (Section A.4).
9. Repo root `bidirectional_survivors_all.npz.flush.tmp.npz` (415 B, 2026-07-29 19:24:07). It is ignored by `.gitignore:142`, so it does not dirty the tree.
10. `.s172_checkpoint/`: 25 run directories, the newest dated 2026-08-01. It is ignored by `.gitignore:50`.
11. **Halt file:** `/tmp/agent_halt` (`agents/watcher_agent.py:384`) is **absent**. Host boot was 17:11:46 today. The Gate-12 harness passes `--clear-halt` (`gate12_launch.sh:377`); a Run-4 command without it would only matter if a halt file existed.
12. **Main tree:** clean at start and end. `.s172_accumulator/` is ignored (`.gitignore:72`), and the root symlinks `bidirectional_survivors_all.npz` / `_binary.npz` are ignored (`.gitignore:73-74`). The post-attempt-9 dirty-tree condition described in the fact file §2.49 does not hold today.
13. **G3 run-authorization gate** (`scripts/run_authorization_gate.py`) has a `RUN4` class whose authorization must state a seed range (`:11-24`, `:36`). No authorization file exists, and the gate has no caller (R2). If it is wired in, a RUN4 authorization would have to name the range Run 4 actually receives, `[2147483648, 4294967296)` with the cursor as read, or the launch refuses.

**Search for locks:** `ls -la` of the repo root, `~/miner_staging` and `/tmp`, grepping for `halt|lock|.pid|flush.tmp|pregate|staging`. No lock or pid file was found beyond the items above.

---

## For the owner — Run 4 seed-range options (no choice made)

**The fact these options turn on.** Run 4 as committed sends `seed_start: 0`, but WATCHER will run it over `[2³¹, 2³²)`, because the certified cursor is at 2³¹ (R1, R2).

The Run-4 proposal's A/B design (`docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` §3) holds the "seed domain" constant against attempt 9, with `generator_phase` as the single moved variable. Beta approved the geometry as *"full 2³¹, attempt-9 stripes"* (`docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:16`). Whether that phrase means the seed **interval** `[0, 2³¹)` or the stripe **geometry** (32 × 2²⁶) is not settled by the text read here.

| Option | What happens | Cost, from what was read |
|---|---|---|
| **A. Continue from the cursor** | Run 4 sweeps `[2147483648, 4294967296)`. If it publishes, the union becomes `[0, 2³²)` and java_lcg `{constant,variable}` becomes COMPLETE | Two variables move against attempt 9 (seed interval and generator phase). The proposal's §3 framing ("Brief I is the single moved variable") would no longer describe the run. It also uses up the last uncovered half of the governed java_lcg domain |
| **B. Repeat `[0, 2³¹)`** | Not reachable through WATCHER as written. `:2087-2089` overwrites any start whenever the cursor is above 0 | Needs either a change to `agents/watcher_agent.py` (no-touch, governed) or a non-WATCHER route (Route D, rejected by Beta for this acceptance run, ruling `:28`). How the ledger and finalizer handle a second certified interval that overlaps `[0, 2³¹)` was **not traced** |
| **C. Hold Run 4 and re-rule** | Ask Beta whether the A/B needs the same seed interval, and how the second half should be covered | No run. Leaves the cursor at 2³¹ |

Two separate steps apply to any option:
- decide whether the Route-A certification still holds for the HEAD WATCHER digest (R4-1);
- if the G3 gate is adopted, decide which seed range the RUN4 authorization names (R4-13).

---

## Not launch readiness

This report establishes only the four reads above. It does **not** establish any of the following; each is outside its evidence:

- GPU health on any rig (no rig was contacted; the fleet was not touched);
- rig source parity (no parity gate was run);
- netconsole sender arm state, or the NCPROOF proof;
- the restricted agent account (G1);
- the environment baseline (G2);
- an authorization for Run 4 (G3 or the owner's);
- any volume limit or capacity bound for the effective seed interval;
- a clean tree at launch time (the tree was clean at 19:14; HEAD moved during the session, and later commits can change that);
- that the Route-A routing certification covers the current WATCHER file.

---

## Session notes

- **CLAUDE.md rule 8 (session changelog) was not done.** The brief limits writes to this report and `data/rm_reads/`, so no `SESSION_CHANGELOG_*` was written.
- Files this session created, all inside the worktree:
  - `docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md` (this file);
  - `data/rm_reads/`, containing:
    - `read_sqlite.py`, `cursor_from_rows.py`, `npz_shape.py`;
    - `certified_coverage_rows.txt`, `cursor_result.txt`;
    - `refs_before.txt`, `status_before.txt`;
    - `stat_before_prng_analysis.txt`, `stat_before_miner_ledger.txt`;
    - `rootdir_before_prng_analysis.txt`, `rootdir_after_prng_analysis.txt`, `rootdir_final.txt`;
    - `miner_staging_dir_before.txt`, `miner_staging_dir_after.txt`;
    - `archive_dir_before.txt`, `archive_dir_after.txt`.
- `data/rm_reads/` is **not** gitignored (`git check-ignore` returned nothing), so committing it is Michael's call.
