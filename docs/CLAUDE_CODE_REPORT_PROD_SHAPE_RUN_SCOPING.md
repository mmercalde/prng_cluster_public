# CLAUDE CODE REPORT — `G-PROD-SHAPE` RUN SCOPING — 2026-11-11/12

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_PROD_SHAPE_RUN_SCOPING_20261111.md`, **as amended by Alpha's decisions** (relayed by Michael after the §1 stop; listed in §0).
**HEAD:** `cf9edaa`. The brief's base is `b6cb789`. `tests/gate_s172_prod_shape.py` and `miner/` are unchanged since `a15b524`.
**Review class:** BEFORE-COMMIT. **Not committed**; this goes to Beta first.
**Result (after the 2026-11-12 hardening pass, §10):** new suite `tests/test_gate_prod_shape_scoping.py` gives **29/29 PASS, `COMPLETION SENTINEL: PASS`** (`data/prod_shape_scoping/suite_run3.log`). Both regression suites have per-gate lists identical to baseline (partb 24/24, bp 52/52; `after3_*.log`). The live ledger and staging directory were byte-identical before and after LIVE-1. **LIVE-1 (§4) was not re-run after the hardening.**

**Null results and cautions first:**
- **§1 stopped the first pass.** The brief described the host as it was on 2026-08-22. Since then the 9-run ledger was displaced (PROD-SCOPE-1) and all 6151 staging files, including the 512 orphans, were relocated to `~/miner_forensic_20260823/`. The live ledger now holds one aborted run, `8cb116da`. See §1 and §6.
- **The verifier's behaviour changes beyond the brief's letter, for Beta to note.** An absent or unreadable ledger, and an absent staging directory, were `FAIL` at base and are now **`UNAVAILABLE`**. This follows brief §5 ("an unreadable ledger or staging directory is UNAVAILABLE, never PASS"). Overall, both still end non-PASS.
- **I wrote two files into the live staging directory** during the first §1 read: `miner_ledger.db-wal` (0 B) and `-shm` (32 KB). The ledger itself was unchanged. Per Alpha, they are **left in place**.
- **The unmodified-verifier baseline was run against a `cp -a` copy** of the live staging dir, not the live dir. The base verifier opens with `mode=ro` (`:137`) and would have written to the live `-shm`.

---

## 0. Alpha's decisions (as relayed) and how each was implemented

| # | Decision | Implementation |
|---|---|---|
| 1 | Add `--ledger`; LIVE-1 becomes two reported, non-gate checks: (a) live ledger with `8cb116da`, where B1 must FAIL; (b) the archived 9-run copy with `554463d3`, informational | `--ledger PATH` (default `<staging-dir>/miner_ledger.db`). LIVE-1a and LIVE-1b are in §4 |
| 2 | Fix C1 to match `{staged}.tmp.{pid}` as well as `*.tmp`, judged by run-id prefix; add a gate plus a RED arm | `_TMP_RE = r"\.tmp(\.\d+)?$"`. Temps of a staged name are judged by prefix, other temps whole-directory. Gates C1-TMP, C1-RED and MUT-C1 |
| 3 | Leave `-wal`/`-shm` in place. A non-empty `-wal` makes the ledger UNAVAILABLE; otherwise open with `immutable=1` | `open_ledger()`. Gates WAL-A and WAL-B, mutants MUT-5 and MUT-6 |
| 4 | Read-only forensic on the archive whose hash no longer matches its manifest | §5. **Attempt 9's rows are intact** |

## 1. §1 answers

### 1.1 Ledger and staging location
- `agent_manifests/window_optimizer.json` `default_params.staging_dir` = `/home/michael/miner_staging`. Confirmed.
- The live ledger is `/home/michael/miner_staging/miner_ledger.db`: 6,434,816 B, sha256 `7bd10213…5c2c`, mtime 2026-08-27 18:12. The project-root `miner_ledger.db` (98,304 B, 2026-08-04) is the stale one.
- Base verifier: ledger = `os.path.join(staging_dir, "miner_ledger.db")` (`tests/gate_s172_prod_shape.py:132`).

### 1.2 Trials
| Ledger | run_id | state | stripes | shards |
|---|---|---|---|---|
| **live** | `distributed_config_t1_8cb116da` (run 3) | **aborted** (`compute_lease_expiry`) | 128 | 5348 |
| archive | 7 runs: `689f3cd9`, `abc63f71`, `c8939b64`, `7e0d020b`, `db0393b0`, `36bf30e3`, `5c010902` | aborted | 64–96 each | 2003, 3948, 1908, 3968, 1820, 1881, 1862 |
| archive | `d606edbe` | committed | 128 | 5999 |
| archive | **`554463d3` (attempt 9)** | **committed** | 128 | **5693** |

**Attempt 9 is not in the live ledger.** It exists only in `~/miner_ledger_archive/miner_ledger_pre_separation_9runs.db`, displaced on 2026-08-22 under PROD-SCOPE-1 (`logs/brief_i_evidence/PROD_SCOPE_1_EXCEPTION_LEDGER_DISPLACEMENT.md`).

### 1.3 Staged file naming
- `safe_run = re.sub(r"[^A-Za-z0-9._-]", "_", str(run_id))`, and the name is `f"{safe_run}__{stripe_id}_a{attempt}_s{sub_index}_g{gen}_{sha256[:16]}.json"` (`miner/range_miner_coordinator.py:4372-4374`).
- The verifier copies the coordinator's derivation verbatim in `safe_run_id`. **Attribution (hardened, §10.1):** a file is this run's iff its name starts with `safe_run_id(run_id) + "__"`. The first pass split at the first `__` (`_run_of`), which mis-attributes any run whose safe id contains `__`. `_run_of` now only counts distinct other runs for the `C-OTHER` line.
- The 512 orphans all carry `distributed_config_t1_25e4f207`, per the inventory `logs/brief_i_evidence/C2_orphan_staged_payloads.json` and `~/miner_forensic_20260823/STEP1_INVENTORY.json`.
- **They are no longer in the staging directory.** All 6151 staging files (512 × `25e4f207`, 5632 × `eed23c7f`, 7 namespace files) were relocated on 2026-08-23 01:12–01:15Z to `~/miner_forensic_20260823/staged_payloads/` (+ `displaced_original/`). STEP4 records destination verification as 6144/6144 with no hash or size mismatch, and STEP7 records zero occupancy afterwards.

### 1.4 Other leak patterns
| Verifier pattern (base) | What the coordinator writes | Carries a run id? |
|---|---|---|
| C1 `endswith(".tmp")` (`:211`) | transfer temps `f"{staged}.tmp.{os.getpid()}"` (`:4566`, `:4725`). **The base C1 never matched these** | temps: yes (prefix of the staged name) |
| C3 `.s172_staging_probe*` (`:213`) | `.s172_staging_probe_{pid}_{uuid8}.tmp` (`:12196`) | no |
| C4 `*provisional*` (`:214`) | no filename containing "provisional" is built. All 4 hits (`:2491`, `:2497`, `:4255`, `:4300`) are prose | n/a |

### 1.5 Remote versus inline delivery: Alpha's reading CONFIRMED
- Worker: `INLINE_BYTE_LIMIT = 48 * 1024 * 1024` (`miner/range_miner_worker.py:1344`). It spools only when the payload is ≥ the limit, or the framed message is > the limit or cannot be framed (`:2475-2489`).
- Coordinator: inline leads to `enqueue_staging("inline", …)` (`:11844`); `spool_path` leads to `enqueue_staging("remote", …)` (`:11856`). The shard row stores `remote_spool_path=(msg.spool_path or None)` (`:11813`), so it is **NULL for every inline shard by construction**.
- `_delete_remote` runs only `if task.kind == "remote":` (`:4527-4528`) and writes `deleted` or `failed` (`:8247-8260`).
- A second writer, `retry_remote_delete` (`:8262-8284`), exists but **has no production caller**; only `tests/test_s172_phase4_coordinator.py` uses it. No other `UPDATE shards` (`:1608`, `:1623`, `:1637`, `:1651`, `:2369`) touches the column.

Measured (read with `immutable=1`):

| Ledger | shards | `remote_spool_path` non-NULL | attempt 9 non-NULL | `MAX(size_bytes)` | `remote_delete_status` |
|---|---|---|---|---|---|
| live | 5348 | **0** | n/a | 603,612 B | `none` × 5348 |
| archive | 29,082 | **0** | **0 / 5693** | 21,202 B | `none` × 29,082 |

**Statement (§6.2): YES, `'none'` is the correct final value for an inline shard.**
- No shard of the 34,430 ever had a remote file.
- The largest shard is about 1/80 of the spool threshold.
- The only production writer of a non-`'none'` status runs only for remote tasks.

### 1.6 References to `gate_s172_prod_shape`
All are comments or printed notes; **none executes or parses the verifier**:
- `scripts/launch_fleet_manual.sh:16`
- `tests/test_s172_staging_partb.py:7-8`, `:684-685`
- `tests/test_s172_staging_backpressure.py:8-10`, `:1827`, `:5444`
- `miner/range_miner_coordinator.py:3719`

### 1.7 Baseline (`data/prod_shape_scoping/base_*.log`)
| Run | Result |
|---|---|
| `test_s172_staging_partb.py` | 24/24 checks green, sentinel PASS (1.3 s, 30 MB RSS) |
| `test_s172_staging_backpressure.py` | 52/52 checks green, sentinel PASS (70 s, 67 MB RSS) |
| unmodified verifier, `--log logs/gate12_attempt9_launcher_20260817_181819.log`, against a `cp -a` copy of the live staging dir | 16 pass / 5 fail: B1, B5, B6, B7 FAIL (live ledger = aborted `8cb116da`); C1–C4 PASS (no staged files); D* FAIL (no `gen-*`). A1–A8 PASS. Section A is available, from the attempt-9 launcher log |

## 2. Diff (`tests/gate_s172_prod_shape.py`, +249 / −39 vs HEAD including §10; full diff at `data/prod_shape_scoping/verifier.diff`)

- **`open_ledger(db)`** (new, `:138`):
  - absent or not a file → UNAVAILABLE;
  - `db-wal` non-empty → UNAVAILABLE, not opened;
  - otherwise `sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)` plus a `SELECT 1 FROM trials` probe, with a `sqlite3.Error` → UNAVAILABLE.
- **`list_trials(db)`** (new): prints run_id, state and shard count for every trial, used when there is no `--run-id`.
- **`check_ledger(db, run_id)`**:
  - every query has `WHERE run_id=?`: trials, stripes, shards, and reservations (both counts);
  - an unknown run gives **`B0` FAIL** naming the trials present;
  - **B1** is "this trial is COMMITTED";
  - **B5/B6** keep their `len(shards) > 0` guards;
  - **B7**: `len(shards) > 0 and all(status != 'none' for s in shards if s.remote_spool_path is not None)` **and no shard has `remote_spool_path IS NULL` with `size_bytes >= INLINE_BYTE_LIMIT`** (§10.2), with evidence `"{n} remote-spooled, {m} inline; remote_delete_status values: […]; inline with size_bytes >= INLINE_BYTE_LIMIT (50331648 B): k"` (plus the spooled statuses when n > 0, and one example when k > 0);
  - adds the line `B-OTHER  N shards from M other run(s) not judged (K other trial row(s))`, printed every time.
- **`check_no_leaks(staging_dir, run_id)`**:
  - unreadable dir → UNAVAILABLE;
  - **C1** matches `\.tmp(\.\d+)?$`. Temps of a staged name are judged by run prefix; temps with no run id (for example the probe) stay whole-directory;
  - **C1/C2** count only staged payloads and transfer temps whose name starts with `safe_run + "__"` (§10.1);
  - C3 and C4 are unchanged (no run id);
  - adds the line `C-OTHER  N staged files (+T temp files) from M other run(s) not judged`, printed every time.
- **`main`**:
  - `--run-id` and `--ledger`;
  - without `--run-id`: lists the trials, marks `B*` and `C*` UNAVAILABLE, and ends `COMPLETION SENTINEL: UNAVAILABLE — no --run-id; no run was judged`, with exit 2 **regardless of A/D legs**. The FAIL legs are still listed and counted in the sentinel line;
  - an absent staging dir is UNAVAILABLE (was FAIL).
- Sections A and D are unchanged.

## 3. Gates (`tests/test_gate_prod_shape_scoping.py`, `data/prod_shape_scoping/suite_run3.log`)

Fixtures are a temp WAL-mode ledger whose four tables come from the coordinator's own DDL string literals, found by AST at test time, plus a staging dir. The verifier is exec'd in-process from source text: live, `git show b6cb789:…`, or a mutated live copy. Every mutant inserts a hit sentinel before its target line, and every mutant's hits are asserted.

Base fixture:
- **T**: committed; 2 stripes done; 3 shards (verified, acked, deleted, inline); 3 reservations released.
- **X**: aborted; 1 stripe cancelled; 2 shards `enqueued`/`none`; 3 staged files named for X.

| Gate | Verdict | Evidence |
|---|---|---|
| S-RED | PASS | pre-change verifier: **B5 FAIL, B6 FAIL, C2 FAIL**, because X's rows and files are counted. The fault is reproduced |
| S-A | PASS | `--run-id T`: B1–B8 and C1–C4 all PASS. Prints `B-OTHER  2 shards from 1 other run(s) not judged (1 other trial row(s))` and `C-OTHER  3 staged files (+0 temp files) from 1 other run(s) not judged` |
| S-B | PASS | plus one staged file for T: C2 FAIL (1 file named) |
| S-C | PASS | one T shard `enqueued`: B5 FAIL `2/3 shards acked` |
| S-D | PASS | T aborted: B1 FAIL `state='aborted'` |
| S-E | PASS | `--run-id …nope`: B0 FAIL naming both trials; rc 1; sentinel FAIL |
| S-F | PASS | no `--run-id`: rc 2, `COMPLETION SENTINEL: UNAVAILABLE — no --run-id; no run was judged (3 FAIL leg(s) listed above)`; both trials listed |
| S-G | PASS | T has a held reservation: B8 FAIL `1 held of 2 reservations for this run` |
| S-H | PASS | only X held: B8 PASS for T |
| B7-A | PASS | `0 remote-spooled, 3 inline; remote_delete_status values: ['none']; inline with size_bytes >= INLINE_BYTE_LIMIT (50331648 B): 0` |
| B7-B | PASS | one T shard with a path and status `none`: B7 FAIL |
| B7-C | PASS | one with a path and status `deleted`, the rest inline: B7 PASS, `1 remote-spooled, 2 inline` |
| B7-D | PASS | T with zero shards: B7 FAIL |
| C1-TMP | PASS | T's `…json.tmp.12345`: **C1 FAIL**. Control: X's `…json.tmp.54321` gives C1 PASS and is counted in `C-OTHER … (+1 temp files)` |
| C1-RED | PASS | pre-change verifier with T's `.tmp.12345` present: **C1 PASS (missed)** |
| WAL-A | PASS | writer held open, `-wal` = 8272 B: `B*` UNAVAILABLE ("non-empty … not opened"); no B1/B5–B8 judged; rc ≠ 0 |
| WAL-B | PASS | no `-wal`, and an empty 0-B `-wal`: legs judged as in S-A, and the directory listing, sizes and db sha256 are unchanged. The verifier created nothing |
| MUT-1 | PASS | shards query unscoped (hits `['MUT-1']`): S-A goes **RED** (B5, B6 FAIL) |
| MUT-2 | PASS | C2 filter removed (`['MUT-2']`): S-A goes **RED** (C2 FAIL) |
| MUT-3 | PASS | old B7 condition restored (`['MUT-3']`): B7-A goes **RED** |
| MUT-4 | PASS | no-run-id path returns PASS (`['MUT-4']`): S-F goes **RED** (rc 0, sentinel PASS) |
| MUT-5 | PASS | non-empty `-wal` check removed (`['MUT-5']`): WAL-A goes **RED** (B1, B5–B8 judged on the partial view) |
| MUT-6 | PASS | `immutable=1` dropped, plain `mode=ro` (`['MUT-6','MUT-6']`): WAL-B goes **RED**. The verifier **creates `miner_ledger.db-shm` + `-wal`** (no-wal case) and `-shm` (empty-wal case). This reproduces the live side effect |
| MUT-C1 | PASS | C1 reverted to `\.tmp$` (`['MUT-C1']`): C1-TMP goes **RED** |
| C-SEP | PASS | §10.1. Run `'a  b'` (safe `a__b`) with 1 leaked staged file + 1 leaked `.tmp.777` temp: **C2 FAIL, C1 FAIL**. Control with neither: C2 PASS, C1 PASS |
| C-SEP-RED | PASS | the first-pass attribution (`_run_of(e) == safe_run_id(run_id)`) restored on both the C2 and C1 lines (hits `['RED-C1','RED-C2']`): **C2 PASS, C1 PASS (missed)**. The leaked files are read as run `a` |
| B7-E | PASS | §10.2. `INLINE_BYTE_LIMIT` = 50,331,648 B, read from the worker source by the suite independently of the verifier. Inline shard at the limit: **B7 FAIL**. Inline at limit − 1: PASS. Spooled at the limit, `deleted`: PASS |
| MUT-B7L | PASS | `oversize_inline = []` (hits 3×, one per arm): B7-E goes **RED**: inline at the limit gives B7 PASS |
| HYG-LIVE-UNTOUCHED | PASS | live staging listing, sizes and ledger sha256 (`7bd10213…`) unchanged across the suite |

Total: 29/29, 58 MB peak RSS, 1.4 s wall.

## 4. LIVE-1 (read-only, reported, not gates)

**Run before the §10 hardening and not re-run.** Expected impact: none. LIVE-1a/1b have 0 remote-spooled shards and `MAX(size_bytes)` far below the limit (§1.5), so the new B7 term is 0. Neither run id contains `__` after sanitisation. The B7 evidence line now carries one more field. This is `[UNVERIFIED]` until LIVE-1 is re-run.

Fingerprint taken before and after both runs: `ls -A --full-time` of `~/miner_staging` and `~/miner_ledger_archive`, plus sha256 of the live db, `-wal`, `-shm` and the archive db. The two fingerprints are **identical** (`data/prod_shape_scoping/live1_fp_{before,after}.txt`). Live ledger sha256 is `7bd1021310043bce39c4b3822c1eac0e7a604cd9f81d7d28112433e3e21a5c2c` before and after. The live `-wal` is 0 B, so `immutable=1` was used.

### LIVE-1a: live ledger, `--run-id distributed_config_t1_8cb116da`, `--log logs/gate12_20260822_182119.log`
```
-- A. anti-fabrication --   A1..A8 all PASS (EXEC CMD 2026-08-22 18:22:57, --use-range-miner, manifest staging, VALIDATED)
-- B. ledger: ONE run's trial, judged on its own rows --
  [FAIL] B1 this trial is COMMITTED        distributed_config_t1_8cb116da: state='aborted'
  [PASS] B2 stripes reached DONE           128 stripe(s); 120 done; states=['cancelled', 'done']
  [PASS] B3 sub-stripe results STAGED      5348 shard(s); 5333 verified; staging_status=['failed', 'verified']
  [PASS] B4 phases present                 [1, 2, 3, 4]
  [FAIL] B5 Phase-5 acknowledgement        0/5348 shards acked; statuses=['enqueued', 'none']
  [FAIL] B6 local cleanup                  0/5348 cleaned; statuses=['none']
  [PASS] B7 remote-delete (spooled only)   0 remote-spooled, 5348 inline; remote_delete_status values: ['none']
  [PASS] B8 NO active reservations leaked  0 held of 5333 reservations for this run
  B-OTHER  0 shards from 0 other run(s) not judged (0 other trial row(s))
-- C. leak-freedom --   C1..C4 PASS (0 this-run temps, 0 no-run-id temps, 0 staged, no probes, no provisional)
  C-OTHER  0 staged files (+0 temp files) from 0 other run(s) not judged
-- D. --   [FAIL] D* artifact: no published generation directory found
17 pass / 4 fail / 0 unavailable of 21 legs — COMPLETION SENTINEL: FAIL (rc 1)
```
**B1 FAILs as Alpha required.** B5 and B6 also FAIL, which is genuine: the run aborted before Phase 5.

### LIVE-1b: archived 9-run copy, `--ledger ~/miner_ledger_archive/miner_ledger_pre_separation_9runs.db --run-id distributed_config_t1_554463d3`, `--log logs/gate12_attempt9_launcher_20260817_181819.log`. INFORMATIONAL
```
-- A. --   A1..A8 all PASS (EXEC CMD 2026-08-17 18:19:58)
-- B. --
  [PASS] B1 this trial is COMMITTED        state='committed'
  [PASS] B2 stripes reached DONE           128 stripe(s); 128 done; states=['done']
  [PASS] B3 STAGED AND VERIFIED            5693 shard(s); 5693 verified
  [PASS] B4 phases present                 [1, 2, 3, 4]
  [PASS] B5 Phase-5 acknowledgement        5693/5693 shards acked; statuses=['acked']
  [PASS] B6 local cleanup                  5693/5693 cleaned; statuses=['deleted']
  [PASS] B7 remote-delete (spooled only)   0 remote-spooled, 5693 inline; remote_delete_status values: ['none']
  [PASS] B8 NO active reservations leaked  0 held of 5693 reservations for this run
  B-OTHER  23389 shards from 8 other run(s) not judged (8 other trial row(s))
-- C. --   C1..C4 PASS — against TODAY's live staging dir, not attempt 9's end state (see §7)
  C-OTHER  0 staged files (+0 temp files) from 0 other run(s) not judged
-- D. --   [FAIL] D* artifact: no published generation directory found
20 pass / 1 fail / 0 unavailable of 21 legs — COMPLETION SENTINEL: FAIL (rc 1)
```
B5 and B6 at 5693/5693 match the STEP23 report, and B-OTHER matches the brief's own example (`23389 shards from 8 other runs`). **This run is informational only:**
- the archive's sha differs from its manifest (§5);
- section C here describes today's directory, not attempt 9's.

Full outputs: `data/prod_shape_scoping/live1a.log`, `live1b.log`.

## 5. Archived 9-run ledger: hash/manifest mismatch (Alpha decision 4; read-only, no fix)

Script output: `data/prod_shape_scoping/archive_forensic.log` and `archive_forensic/archive_forensic.json`.
- The archive was opened with `immutable=1`; its directory listing and sha were unchanged afterwards.
- The reference is the **displaced original** (`~/miner_ledger_archive/displaced_original/`), whose `miner_ledger.db` sha256 `4846d34f…3336` **matches the manifest**. Its committed state partly lives in its 4,148,872-B `-wal`, which `immutable=1` would ignore. So db, `-wal` and `-shm` were copied into `data/` and the copy was read. The original files' size, mtime and sha were unchanged afterwards.

| | Archive copy | Manifest records | Displaced original (+WAL, read via copy) |
|---|---|---|---|
| path | `…/miner_ledger_pre_separation_9runs.db` | same | `…/displaced_original/miner_ledger.db` |
| size | 33,099,776 B | 33,099,776 B | 33,099,776 B (+ wal 4,148,872 B) |
| mtime | **2026-08-28T01:13:00Z** (2026-08-27 18:13 local) | original mtime 2026-08-17 18:32:19 local | 2026-08-17 18:32:19 local |
| sha256 | **`c5d942a2…7d46`** | `4846d34f…3336` | `4846d34f…3336` ✓ |
| sidecars | **none** (`-wal`/`-shm` absent) | "`-wal` (committed state), `-shm` preserved" | present |
| `journal_mode` | **`delete`** | — | `wal` |
| `PRAGMA integrity_check` | **ok** | — | ok |
| trials / stripes / shards | 9 / 768 / 29,082 | 9 / 768 / 29,082 | 9 / 768 / 29,082 |
| reservations / workers / preflight_plans / trial_context | 26,512 / 25 / 9 / 9 | — | 26,512 / 25 / 9 / 9 |
| per-run (state, stripes, shards) for all 9 runs | **all equal to the manifest** | as listed | — |

**Attempt 9: intact.** Per-table row digests for `554463d3` are **identical** between the archive copy and the original+WAL:

| Table | Rows | sha256 of ordered rows |
|---|---|---|
| trials | 1 | `e02b8658…` |
| stripes | 128 | `ff1efd5f…` |
| shards | 5693 | `59483a67…` |
| reservations | 5693 | `fc1d09f1…` |
| trial_context | 1 | `41144fd0…` |
| preflight_plans | 1 | `71191b94…` |

All seven table counts are also equal across the two.

**Reading [interpretation, not proven]:** the content is unchanged. The byte difference is consistent with something opening the archive copy **read-write** at 2026-08-27 18:13 local. That would checkpoint its WAL into the main file, convert it to `journal_mode=delete` and remove the sidecars. The archive manifest's "sidecars preserved" is therefore no longer true of the copy, though it is true of `displaced_original/`. The same minute also touched the live ledger (mtime 18:12:59) and `~/miner_forensic_20260823/ledger/miner_ledger.db` (18:13). **Who or what did it was not investigated.**

## 6. Regression: before and after
| Suite | Before | After | Diff of gate lists |
|---|---|---|---|
| `test_s172_staging_partb.py` | 24/24, sentinel PASS | 24/24, sentinel PASS (after §10: `after3_partb.log`) | identical (26 lines) |
| `test_s172_staging_backpressure.py` | 52/52, sentinel PASS | 52/52, sentinel PASS (after §10: `after3_bp.log`) | identical (54 lines) |

Run sequentially, never concurrently.

## 7. Where the source or the ledger contradicted the brief
1. **The live ledger no longer holds attempt 9 or the seven aborted runs.** It holds `8cb116da` only. LIVE-1 was re-targeted per Alpha decision 1.
2. **The 512 orphans are not in the staging directory.** They were relocated on 2026-08-23 with verification. The brief's "stay exactly where they are" is moot: they are where the governed relocation put them, untouched by this session.
3. **Base C1 could never match the coordinator's temp files** (`.tmp.<pid>`). Fixed per Alpha decision 2.
4. **The base verifier is not write-free.** `mode=ro` on a WAL ledger in a writable directory creates `-wal`/`-shm`. This was observed live and reproduced by MUT-6. Fixed per decision 3.
5. **`_delete_remote` is not literally the sole writer;** `retry_remote_delete` also writes. It has no production caller, so §3's condition holds in substance. Recorded so Beta can weigh it.
6. Smaller notes:
   - C4 matches no filename the coordinator writes;
   - `:11845`/`:11859` are actually `:11844`/`:11856`.

## 8. Open items found and not fixed
1. **Disposition of the 512 orphans** (and the 5632 `eed23c7f` payloads), now in `~/miner_forensic_20260823/`. This stays Michael's and Beta's call.
2. **Remote spool files on the rigs** cannot be checked with the fleet off. §1.5 says none should exist, since no shard in 34,430 was ever spooled.
3. **Section A is not run-scoped.** Nothing ties `--log` to `--run-id`, so A can pass on another run's log. A natural follow-up is to require the run id to appear in the log. Not authorised here.
4. **Section C is judged on the directory as it is now.** For a historical run (LIVE-1b), C describes today's directory, not that run's end state.
5. **Who rewrote the archive copy and the two other ledgers at 2026-08-27 18:12–18:13** (§5).
6. **The two sidecars I created** in `/home/michael/miner_staging` are left in place per decision 3. The live `-wal` is 0 B, so the verifier opens the ledger with `immutable=1`.
7. ~~Partition-by-`__` attribution would mis-attribute a run id that itself contains `__`.~~ **Fixed in §10.1.** The residual over-attribution case is listed there.
8. **B2, B3 and B4 remain "≥ 1" assertions.** For example, LIVE-1a's B3 passes with 15 `failed` shards. The brief kept them unchanged.
9. **No `gen-*` generation directory exists in the repo root,** so D* fails for every run on this host.

## 9. `git add` for Michael (after Beta)

Built from `git status --porcelain`, which shows `M tests/gate_s172_prod_shape.py`, `?? tests/test_gate_prod_shape_scoping.py`, `?? docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md`, `?? data/` and nothing else:

```
git add tests/gate_s172_prod_shape.py tests/test_gate_prod_shape_scoping.py docs/CLAUDE_CODE_REPORT_PROD_SHAPE_RUN_SCOPING.md
```

`data/prod_shape_scoping/` holds logs and the forensic copies (39 MB, including a copy of the displaced original). It is not for commit. The brief itself was committed in `040697f` (HEAD moved from `cf9edaa` to `040697f` during this session; that commit does not touch the verifier or `miner/`).

## 10. Hardening pass (2026-11-12, Alpha's read of the diff)

Two changes to `tests/gate_s172_prod_shape.py`. Not committed.

### 10.1 Section C run-id matching: `startswith(safe_run_id(run_id) + "__")`
- `check_no_leaks` (`:380`, `:388-389`, `:393-394`): `mine_prefix = safe_run_id(run_id) + "__"`.
  - A staged file or transfer temp whose name starts with `mine_prefix` is this run's.
  - Every other name with a staged shape is "other".
  - Temps with no run id (the probe) are still judged whole-directory, as before.
- `_run_of` is kept **only** to count distinct other runs on the `C-OTHER` line. That count can still merge two other runs that share a text-before-`__`. It is reporting only, never judged.
- Gates: **C-SEP** (C2 and C1 FAIL for run `'a  b'` with leaked files), **C-SEP-RED** (the first-pass code PASSes both, i.e. misses them), plus MUT-2 retargeted to the new C2 line (still RED).
- **Residual, the opposite direction:** judging run `a` now also claims the files of any run whose safe id starts with `a__` (for example `a__b`). That gives a **false FAIL**, so it is fail-closed. It cannot arise between coordinator run ids `distributed_config_t1_<hex8>`, since hex has no `__`. Not gated.

### 10.2 B7: no oversize inline shard
- The verifier reads `INLINE_BYTE_LIMIT` from `miner/range_miner_worker.py` at run time (`inline_byte_limit()`, `:183`).
  - It uses AST, not import: exactly one module-level assignment is required, and only integer literals with `+ - * ** <<` are evaluated.
  - Today that is `48 * 1024 * 1024` = 50,331,648 B (`miner/range_miner_worker.py:1344`).
  - If it cannot be read, B7 is **UNAVAILABLE**. **That path has no gate.**
- New term (`:305`, `:307`): B7 FAILs if any shard of the run has `remote_spool_path IS NULL` and `size_bytes >= limit`.
  - Why this is a contradiction: the worker spools whenever `len(payload_bytes) >= INLINE_BYTE_LIMIT` (`worker :2475`). `size_bytes` is that same `len(payload_bytes)` (`worker :2452`, carried as `msg.size_bytes` into the shard row at `coordinator :11813`).
  - A **NULL `size_bytes`** is not counted, which matches SQL `NULL >= x`. The column is nullable (`coordinator :1259`).
- Gates: **B7-E** (at-limit inline FAIL, limit−1 PASS, spooled-at-limit-deleted PASS) and **MUT-B7L** (condition removed → B7-E RED). MUT-3 was retargeted to the new `b7_ok` line (still RED).

### 10.3 Runs
| Suite | Result | Log |
|---|---|---|
| `test_gate_prod_shape_scoping.py` | **29/29 PASS**, sentinel PASS, 58 MB RSS, 1.4 s | `data/prod_shape_scoping/suite_run3.log` |
| `test_s172_staging_partb.py` | 24/24, gate list identical to baseline | `after3_partb.log`, `after3_partb.gates` |
| `test_s172_staging_backpressure.py` | 52/52, gate list identical to baseline | `after3_bp.log`, `after3_bp.gates` |

HYG-LIVE-UNTOUCHED: the live staging listing, sizes and ledger sha256 (`7bd10213…`) were unchanged across the suite. The `git add` line in §9 is unchanged: the same three files.
