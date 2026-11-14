# CLAUDE CODE REPORT — PWC AS A WORKING BACKUP, STAGE A (read-only)

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_PWC_BACKUP_STAGE_A_20261114.md`
**Host:** VM101 (`zeus-ubuntu`), as `michael`, venv `~/venvs/torch`.
**Base commit:** `b11b6fde7ab57321b42c418f87af8cc2c75c814d` (the brief names `b1cd2fa`; `b11b6fd` is its
one docs-only child, which adds this brief). **Worktree:** `/home/michael/tfm_worktrees/pwc-backup`,
detached at that commit. **This report is the only file written.** Nothing was committed, pushed,
branched or tagged. No engine, GPU job, fleet action or SSH to any rig was run.

**What was executed (everything else was read):** two CPU-only import checks with
`PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''` (§9), plus `git`/`grep`/`sed` reads. `git status
--porcelain` was empty after the import checks.

**Yardstick.** TFM is judged by whether prediction pools hit more often than `k/1000` on draws the
sieve never saw. A backup engine is judged by whether Steps 2–6 can use its output the same way, and by
that hit rate. Nothing below is a verdict on PWC's predictive value. **No hit-rate measurement of
PWC-produced survivors was looked for or found in this session.** Every gap below is stated as a fact,
followed by what it would do to the credited Step-1 record that hit rate is attributed to.

---

## Prior record

### Searches run (worktree at `b11b6fd`; `/bin/grep`, because the shell `grep` skips `*.json`)

```
ls docs | /bin/grep -iE "PWC|QUARANTINE|RIG_CRASH|FORENSIC|PERSISTENT|ZMQ|GCVM|L2"
/bin/grep -lIiE "PWC|persistent_worker_coordinator|persistent worker" docs/TB_RULING_*.md docs/TB_RULING_REQUEST_*.md
/bin/grep -rnIiE "retire[ds]? .{0,40}(PWC|ZMQ)|(PWC|ZMQ).{0,80}retire" docs/ --include=*.md
/bin/grep -rnIiE "(PWC|ZMQ|persistent worker)[^\n]{0,100}(fallback|backup|fall back)|(fallback|backup|fall back)[^\n]{0,100}(PWC|ZMQ)" docs/ --include=*.md
/bin/grep -nIiE "fallback|PWC|ZMQ|persistent_worker|--use-persistent" docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md
/bin/grep -nIiE "fallback|coexist|§8|## 8" docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md
for f in docs/PROJECT_FILE_CATALOG.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md; do /bin/grep -nIiE "PWC|persistent_worker|use_persistent" $f; done
/bin/grep -rlIiE "PWC|persistent_worker_coordinator" docs/exchange/
/bin/grep -lIiE "PWC|persistent_worker_coordinator" docs/*_AUDIT*.md docs/PROPOSAL_*.md docs/CHAPTER_*.md
/bin/grep -rnI "PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED" --include=*.py --include=*.md .
git log -S PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED -- '*.py';  git show 8a55a68;  git show --stat 8e8e6f3
/bin/grep -rnIiE "comparator ruling|Ruling 24|ruling 5\b" docs/
/bin/grep -nIiE "PWC|sieve_gpu_worker" docs/S172_WINDOW_ANCHOR_BRIEF_I.md docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md docs/TB_RULING_WINDOW_ANCHOR*.md docs/TB_RULING_BRIEF_I_PATCH_REVIEW.md
/bin/grep -rnIiE "combined.{0,10}session" docs/ ; /bin/grep -rnIE "192\.168\.3\.127" docs/ | /bin/grep -iE "pwc|worker_service|--host|PWC_HOST"
```

Two read-only sub-agents ran, with their claims spot-checked here (§4): one swept every
`docs/SESSION_CHANGELOG_*.md` plus `ROCm_Saturation_Report_S172.md`, `RIG_CRASH_FORENSIC_20260822.md`,
`PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md`, `PROPOSAL_S96B_*`, the S149/S150/S175 ruling requests, the
TODO files, `BACKLOG.md`, `LEADS.md` and the catalog for PWC run records. The other did the same for ZMQ.

### Documents found that decide, authorise, schedule or record the subject

| Document (file / section) | What it records |
|---|---|
| `docs/TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md` (commit `8e8e6f3`, 2026-07-30) §5, §7 | **The request** that led to the retirement: "Retire PWC as the authoritative comparator" (`:75`); PWC/ZMQ stay "flag-selectable, for exploratory use — **not** as an authority" (`:94-95`) |
| `docs/CLAUDE_CODE_INSTRUCTIONS_S172_THRESHOLD_REPAIR.md` REV2 `:3-12` | Secondary record of the ruling: "Beta has since **retired PWC/ZMQ from certifying authority** — they are flag-selectable, non-certifying diagnostic paths"; "comparator ruling at `8e8e6f3`" |
| commit `8a55a68` (2026-07-31) | "PWC hybrid quarantined as uncertified (TB Ruling 24 items 1-2, **comparator ruling 5**)" |
| `persistent_worker_coordinator.py:157-218` | The quarantine, quoting comparator ruling §5, and its stated condition for lifting |
| `docs/SESSION_CHANGELOG_20260730_S172_THRESHOLD_REPAIR.md:26`, `:33-45` | Option B chosen; where the guard sits, and why |
| `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` §10 `:1090-1110` | "Team Beta retired PWC/ZMQ from certifying authority on 2026-07-31"; "Do not cite PWC results as comparators" |
| `docs/CHAPTER_9_AUDIT.md:94-108` (S-2, S-3, S-4) | Already ran the ruling-text search: "Standalone ruling text not located as a `TB_RULING_*` file"; "No `TB_RULING_*` file contains '2026-07-31' (searched)"; the manifest default is PWC |
| `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1127`, `:1420`, `:2057` | Quarantine, no override flag; PWC constant-skip non-certifying |
| `docs/CHAPTER_12_WATCHER_AGENT.md:91` | RANGE-MINER is the certifying engine; PWC and ZMQ are not |
| `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` §8 `:460-475`, §11.L `:598-600` | Designed **coexistence**: single active backend, port 5700 vs 5600; acceptance "PWC-TCP run after a miner run on the same Zeus succeeds without manual cleanup" |
| `docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` §16.B `:258-280` | "PWC remains the frozen authoritative comparator; ZMQ is the mandatory third oracle". **Superseded** by 2026-07-31 (`docs/PIPELINE_BEHAVIOUR_MODEL.md:1185`, D9) |
| `docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R1_BETA_REVIEW.md:93-95` (Q1) | PWC does not carry completion accounting to the trial; "not a request to lift" the quarantine |
| `docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R2_BETA_REVIEW.md:78-80` | PWC/ZMQ return empty `swept_intervals` and cannot earn credit. Closes R1 Q1 |
| `docs/exchange/to_beta/X-STEP1-SEARCH-CODE_R2_ALPHA_REPLY.md:66-82` | "PWC and ZMQ are refused certification in stage 1"; "The manifest default route is PWC" |
| `docs/CLAUDE_CODE_INSTRUCTIONS_MINER_SEED_START_20261114.md:92` (R5) | **Owner decision, 2026-11-14:** a PWC or ZMQ Step-1 run with `seed_start != 0` refuses before dispatch. Not yet in code (§3.2) |
| `docs/S172_WINDOW_ANCHOR_BRIEF_I.md:66-83`, `:362-368`; `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:70`; `docs/TB_RULING_BRIEF_I_PATCH_REVIEW.md:60-72` | The PWC/ZMQ `offset` surfaces are firewalled out of Brief I and belong to **Brief II's** repo-wide consumer audit; PWC/ZMQ normalization now writes the `offset` field as the window anchor |
| `docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md:328`, `:341` | `sieve_gpu_worker.py:113` is a live clamp site that the C-2 table missed |
| `docs/TODO_CONSOLIDATED.md:93`, `:207`, `:287`, `:302-323`, `:435` | A decision is owed after Phase 7 on the manifest default; combined-session is still sampled (REVIEW); `miner/` imports no PWC module; the PWC TODO entries closed as non-certifying |
| `docs/TFM_PROJECT_FACTS_SKILL.md` §0.7 `:181-188`, §2.11 (owner ruling 2026-08-04, `min_workers` not restored), §2.50 `:2533-2541` | Why RANGE-MINER exists; PWC's ~17K launches per trial; the 24-worker gate is deliberately not restored |
| `docs/ROCm_Saturation_Report_S172.md:15-35` | PWC's measured stability envelope (pool × chunk size) |
| `docs/SESSION_CHANGELOG_20260329_S158.md:22-40`, `docs/SESSION_CHANGELOG_20260425_S170.md:56-74`, and the other changelogs in §4 | PWC's operating history |
| `docs/PROPOSAL_ZMQ_SQLITE_COORDINATOR_S158D_v1_0.md` and the ZMQ changelogs in §7 | ZMQ's design and run history |

### Labels for the items this report raises

| Item | Label |
|---|---|
| PWC retired from certifying authority 2026-07-31; standalone ruling text not in the repository | ALREADY RECORDED (`CHAPTER_9_AUDIT.md:100-103`). This session's search agrees: **NOT FOUND** |
| Hybrid quarantine and its lifting condition | ALREADY RECORDED (`persistent_worker_coordinator.py:214-217`; `CHAPTER_9_AUDIT.md:95-98`) |
| No document records "PWC/ZMQ stay as **fallbacks/backups**"; coexistence is what is recorded | NEW as a search result. The owner's statement of 2026-11-14 is the first record of the backup intent |
| Making a PWC result count again | **CHANGES A RULING** (Beta 2026-07-31, S-3) and also **CHANGES A RULING** of Beta X-STEP1-SEARCH-CODE R1 Q1 / R2 (PWC refused credit) |
| Completion accounting not carried to the trial | ALREADY RECORDED (Beta R1 Q1) |
| PWC partitions from seed 0 | ALREADY RECORDED (`window_optimizer_integration_final.py:1896-1903`; stage-1 report). The refusal is an owner decision not yet in code (R5) |
| PWC's `offset` drives both the slice and the kernel pre-advance (`sieve_gpu_worker.py:113`/`:304`; `sieve_filter.py:184`/`:311`) | ALREADY RECORDED (Brief II consumer audit). Its **consequence for a PWC record** (§3.4) is NEW |
| **PWC-TCP rig workers dial a literal `192.168.3.127`** (`persistent_worker_coordinator.py:756`; `persistent/pwc_worker_service.py:759`) | **NEW**. No hit in `docs/` (search above) |
| **The TCP rig worker runs `sieve_filter.py` + `persistent/*`, none of which is in the rig parity set** | **NEW** as stated. The parity set is ALREADY RECORDED (`scripts/gate12_parity_gate.py:174-185`) |
| **The manifest's PWC defaults (pool 8, AMD chunk cap 2,000,000) lie outside every envelope PWC is recorded as surviving** | **NEW** as a juxtaposition. Both halves are ALREADY RECORDED separately |
| A default WATCHER Step-1 run (PWC, `test_both_modes: true`) fails on its first hybrid pass, and even constant-only it cannot publish | ALREADY RECORDED in parts (`CHAPTER_9_AUDIT.md:105-108`; R2 reply `:69-70`). Stated together here |
| `persistent/pwc_protocol.py` cannot be imported | ALREADY RECORDED (`CLAUDE.md` §6). Executed here; no production module imports it |
| `min_workers` reaches PWC as 1, not 24 | ALREADY RECORDED. Owner ruling 2026-08-04: not restored |
| ZMQ not pursued as a backup | NEW owner inclination; §7 only |

### Not searched

VM101's live state outside the worktree (live ledgers, `logs/`, `~/dashboard_work/`, systemd); every rig
(no SSH, by rule), so **what PWC/ZMQ bytes are deployed on the rigs is UNAVAILABLE**; ser8 archives;
the private repository; the project PDFs and other binaries in `docs/`; chat history. Sessions with no
changelog in the repository (S171, S173, S175, most of May–June 2026) are a gap in the operating history.

---

## Summary for the owner (plain words)

1. Beta's 2026-07-31 ruling took away PWC's right to **certify**. That covers being the comparator, and it covers producing Step-1 output that counts. PWC stayed in the tree and can still be selected by flag. The ruling text is not in the repository, only references to it.
2. No document says PWC or ZMQ were meant to be **fallbacks**. The design says only that they **coexist** with RANGE-MINER without interfering. Your 2026-11-14 statement is the first record of backup intent.
3. PWC constant-skip still runs. PWC hybrid refuses to run (quarantined), and nothing can switch that refusal off.
4. Even a clean PWC constant-skip run **cannot publish today**. The Step-1 credit gate refuses it because PWC does not prove it swept the whole interval. So the run produces survivors, but no certified generation and no coverage.
5. As written, PWC-TCP points the rig workers at **192.168.3.127**, the frozen bare-metal Zeus. Run from VM101 (.177), the rigs would have nothing to connect to. I found this by reading the code; I did not run it.
6. The record supports your "underwhelmed" point. Clean runs needed fewer workers per rig (6, not 8) or much smaller chunks (25K–100K seeds, not 2M). Today's manifest defaults are 8 workers per rig and 2M-seed chunks, a setting PWC is not recorded as surviving.
7. On "two rigs, not three": one two-rig run was clean while three rigs panicked (S158). A later two-rig test still crashed (S170). Clean three-rig runs also exist. The record ties stability to **workers per rig and chunk size**, not to the number of rigs.
8. Two of PWC's files are shared with RANGE-MINER: the miner imports `sieve_gpu_worker.py`, and both engines use `prng_registry.py`. Any PWC fix that touches either changes rig-parity-pinned code. Fixes inside `persistent_worker_coordinator.py` and `persistent/` touch nothing the miner loads.
9. The smallest real backup: PWC constant-skip only, on two rigs, at the settings shown to be stable, with the address and completion-accounting fixes. Even that changes two Beta rulings, so it is your decision.
10. ZMQ last ran in April 2026 and was replaced by TCP-PWC for speed. It is not a candidate.

---

## P1 — The record

### P1.1 The 2026-07-31 ruling: what it retired PWC from, and for what scope

**The ruling document itself was NOT FOUND** in `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`,
`docs/exchange/`, or by commit search. `CHAPTER_9_AUDIT.md:100-103` reached the same result earlier:
"No `TB_RULING_*` file contains '2026-07-31' (searched)". What exists is the request, the commit that
carried out the ruling, and secondary quotations:

- **Request** (`TEAM_ALPHA_PWC_COMPARATOR_SCOPE_CORRECTION.md`, commit `8e8e6f3`, 2026-07-30 20:05).
  It asked (§7 `:109-117`) whether PWC was ever intended as the authoritative comparator. It proposed
  (§5 `:75-96`) to "**Retire PWC as the authoritative comparator.** Define Phase 6 acceptance against
  the original interface contract", with "PWC/ZMQ remain in the tree, flag-selectable, for exploratory
  use — **not** as an authority" (`:94-95`).
- **What the ruling did**, quoted secondhand at `CLAUDE_CODE_INSTRUCTIONS_S172_THRESHOLD_REPAIR.md:6-10`:
  "Beta has since **retired PWC/ZMQ from certifying authority** — they are flag-selectable,
  non-certifying diagnostic paths. Consequently **repair 3 is no longer a RANGE-MINER Phase 6
  blocker** … What replaces it is a quarantine requirement: a known-wrong PWC hybrid route must not
  remain silently runnable." The same brief dates the comparator ruling to `8e8e6f3` (`:12`).
- **Scope, as recorded:** *certifying authority*. That means (a) authoritative comparator for Phase 6
  (`v1_4_5 §16.B :273`, superseded per `PIPELINE_BEHAVIOUR_MODEL.md:1185`) and (b) producer of
  certifying Step-1 output (`CHAPTER_2 §10 :1096-1101`; `CHAPTER_12:91`). It did **not** delete or
  disable PWC. PWC constant-skip stays runnable as "a non-certifying diagnostic comparator"
  (`CHAPTER_1:1127`). Chapter 2 adds: "**Do not cite PWC results as comparators**" (`:1103`).

### P1.2 The quarantine ruling and its condition for lifting

- **Ruling:** "Beta comparator ruling §5". The code quotes it: "optional must not mean that a
  known-wrong PWC hybrid route remains silently runnable" (`persistent_worker_coordinator.py:160-161`).
  It was implemented as Option B at `8a55a68` (changelog `:26`, `:33-36`).
- **Cause:**
  - Hybrid passes filtered at `run_sieve_pass`'s `phase2_threshold: float = 0.5` (`:1203`). Neither
    hybrid call site passes it (`:1802-1814`, `:1829-1841`). On the SSH worker the default becomes
    the hybrid threshold (`sieve_gpu_worker.py:257-258`).
  - The hybrid kernels also ignore `skip_min`/`skip_max` and start from `expected_skip = 5`
    (`persistent_worker_coordinator.py:172-181`).
- **Lifting condition, stated in the raised message** (`:213-217`): "implement Beta comparator ruling
  §5 Option A (propagate the directional threshold with a requested/payload/effective gate proving it
  reached the kernel) **and** resolve the hybrid skip-bound dead dimension." There is deliberately no
  override flag (`:185-186`; `CHAPTER_1:1127`).
- **Placement:** the guard is at the execution boundary only (`:1226-1230`), not at the trial entry
  point (`:1660-1671`).

### P1.3 Is RANGE-MINER's design intent "PWC and ZMQ stay as fallbacks" recorded anywhere?

**No document found states it.** What is recorded:
- **Coexistence by design.** `v1_4_4 §8 :460-473` specifies single-active-backend gates and separate
  ports. `§11.L :598-600` is an acceptance gate: "PWC-TCP run after a miner run on the same Zeus succeeds
  without manual cleanup".
- **An oracle role, now superseded.** `v1_4_5 §16.B :273`.
- **Exploratory use.** Scope correction `:94-95`.
- The word "fallback" in `v1_4_5 :149` refers to the miner's own `serial_reference` assembly backend,
  not to PWC.
- The fallback-search hits elsewhere (`S172_WINDOW_ANCHOR_BRIEF_I.md:69`, `CHAPTER_8_AUDIT.md:283`,
  `SESSION_CHANGELOG_S162.md:133`) use "fallback" in other senses. All three were read.

So the owner's 2026-11-14 statement is the first written record of the **backup** intent.

---

## P2 — What PWC does today (by reading; nothing executed)

### P2.1 One Step-1 trial on the PWC route, constant-skip

**Selection.** `--use-persistent-workers` sets `coordinator.use_persistent_workers`
(`window_optimizer.py:852`). The CLI picks backend `'pwc'` for the execution set (`:1772-1779`). The
manifest defaults are:
- `use_persistent_workers: true`, `pwc_transport: "tcp"`, `use_range_miner: false`
  (`agent_manifests/window_optimizer.json:262`, `:268`, `:270`);
- `worker_pool_size: 8`, `seed_cap_amd: 2000000` (`:263`, `:265`);
- `test_both_modes: true` (`:255`).

So **a default WATCHER Step-1 run takes this route**. Branch: `run_bidirectional_test`
→ `window_optimizer_integration_final.py:1859-1904` → `run_trial_persistent`.

**Dataset.** The P0.5 run-start gate runs for every backend and replaces the dataset argument with the
frozen absolute immutable path (`window_optimizer.py:1870-1885`). For a non-miner run a *missing*
provisioning manifest is not fatal; that leniency exists only for non-miner runs
(`miner/dataset_authority.py:1109`). When a manifest is present it verifies the fleet (`:1255-1256`).

**Fleet lifecycle: per trial.** `run_trial_persistent` builds a new `PersistentWorkerCoordinator` and
calls `startup()` (`persistent_worker_coordinator.py:1676-1687`). It calls `shutdown()` in `finally`
(`:1901-1902`). The whole fleet is launched and torn down **for every Optuna trial**.

**Startup, TCP transport (default)** (`:513-546`, `:659-849`):
1. Kill stale workers on every rig over SSH with `pkill -f` (`:458-510`, `:715-724`). On Zeus,
   `pkill -9 -f pwc_worker_service` (`:675-683`).
2. Probe live ROCm GPUs (`:694-696`); pool per rig = `min(worker_pool_size, live)` (`:696`).
3. Per GPU, write `/tmp/pwc_tcp_launch_gpu{N}.sh` on the rig and `nohup python -m
   persistent.pwc_worker_service --host 192.168.3.127 --port <pwc_port>` (`:738-760`). It logs to
   `/tmp/pwc_tcp_worker_<host>_gpu<N>.log` (`:735`). Two Zeus workers connect to `127.0.0.1`
   (`:797-833`).
4. Three-phase start: online up to 60 s, broadcast init, ready up to 180 s (`:842-846`). Ready gate:
   `count >= self.min_workers` (`:897`). The call site passes no `min_workers`
   (`window_optimizer_integration_final.py:1865-1883`), so it is `1` (`:1649`). That is owner-ruled
   (skill §2.11).

**Passes.**
- Pass 1, forward constant (`:1713-1724`). If it finds zero survivors, the trial returns a pruned
  four-map result (`:1730-1749`).
- Pass 2, reverse constant (`:1754-1765`).
- Bidirectional = intersection of the two survivor key sets (`:1771`).

**Inside one pass** (`run_sieve_pass`, `:1194-1582`):
- **Chunking.** `chunk_size = min(total_seeds // num_workers, seed_cap_amd)`. Chunks walk from
  **seed 0** (`:1266-1280`).
- **Concurrency.** A `ThreadPoolExecutor` sized `min(num_workers, len(chunks))` (`:1424-1433`). Each
  chunk is one job, `_dispatch_to_tcp` (`:1057-1083`): submit, then block up to `JOB_TIMEOUT_S = 600`
  (`:154`).
- **Retries.** One retry, only on errors mentioning "empty response", "pipe" or "timeout"
  (`:1349-1357`).
- **Pass timeout.** If no chunk completes for 1,200 s, the pass aborts and the remaining chunks are
  marked as errors (`:1422-1473`).
- **Job contents** (`:1305-1320`): `seed_start/seed_end`, `window_size`, `min_match_threshold`
  (the directional threshold), `skip_range`, `offset`, `sessions`, `dataset_path`. **No residues are
  sent.** The worker rebuilds the window from the dataset path.

**On the rig (TCP).** `pwc_worker_service` imports `sieve_filter.execute_sieve_job`
(`persistent/pwc_worker_service.py:324`). That function slices the window with
`start = max(0, min(int(offset), n - window_size))` (`sieve_filter.py:174-184`, `:535-538`) and passes
`offset` to the kernel as its last argument (`:311`). On the SSH transport the worker is
`sieve_gpu_worker.py --persistent` (`:591-657`), with the same slice (`sieve_gpu_worker.py:113`) and
the same kernel argument (`:304`).

**Returns.** `build_trial_populations(...)`: the v2 four-map contract, validated at egress (`:1881-1899`).
**The pass-level `failed_chunks`/`total_chunks` (`:1508-1549`) are not in it.** The integration adapter
then does three things:
- validates the result again (`window_optimizer_integration_final.py:1200`);
- normalizes and appends constant and variable records to the shared accumulator (`:1239-1254`), then
  flushes incremental NPZ (`:1261`);
- stamps `step1_sweep_evidence = _step1_unaccounted_route_evidence("pwc", …)`, with empty
  `swept_intervals` and `certification_refused: True` (`:1397-1403`, `:1899-1903`).

**Where it can fail:**
- The ready gate, which raises (`:905-921`).
- Dispatch blocked below `min_workers` (`:1251-1261`).
- Any `run_sieve_pass` exception.
- **A failed or timed-out chunk does not fail the pass.** It is counted and logged (`:1517-1526`), and
  the pass returns what survived as if it were complete.
- At publication, the S7 credit gate raises `Step1NoCreditError` (`:3369-3379`), because
  `credit_decision` needs every required phase's swept interval to equal the requested one
  (`utils/step1_search_ledger.py:1068-1085`). The credit chunk is built for every run, search or not
  (`window_optimizer_integration_final.py:2386-2387`).

**What it writes:**
- per-pass `results/window_opt_{forward|reverse}_{ws}_{off}_t{n}.json`, full above 100,000 survivors
  only as a summary (`:1551-1580`);
- the accumulator's incremental NPZ flushes;
- rig-side `/tmp` launch scripts and logs.

**It does NOT write a D3.5 generation or an S145 coverage row**, because the credit gate raises before
`_finalize_run_d3_5` (`:3393`) and `record_certified_coverage` (`:3448`).

### P2.2 Hybrid (variable-skip)

With `test_both_modes` true and a non-hybrid base (`:1786`), Pass 3 calls `run_sieve_pass` on
`<base>_hybrid` (`:1801-1814`). `assert_pwc_hybrid_not_quarantined` raises
`PwcHybridThresholdContractUncertified` **before strategy loading, chunking or any dispatch**
(`:1226-1230`, `:204-218`). The exception propagates, and `finally` shuts the fleet down (`:1901`). Net:
**a both-mode PWC trial runs both constant passes, then fails.** That is the manifest default
(`test_both_modes: true`, `use_persistent_workers: true`).

If the quarantine were lifted, the code would run:
- a single `balanced_hybrid` strategy (`:1790-1799`);
- the Q0 gate, which skips hybrid reverse when hybrid forward is empty (`:1820-1826`);
- a 0.50 threshold (`:1203`, `:1319`). On the TCP worker that is `coerce_threshold(…'auto', 0.50)`
  (`sieve_filter.py:587`).

---

## P3 — Gaps between today and a backup whose result can count

"Shared with RANGE-MINER" means the miner imports or executes the file. "Parity set" is the 10-file pin
at `scripts/gate12_parity_gate.py:174-185`; Beta's minimum is at `:189-195`. The miner imports **none**
of `persistent_worker_coordinator`, `persistent/*`, `pwc_*`, `zmq_*` (`TODO_CONSOLIDATED.md:287`;
re-checked here with `grep` over `miner/*.py`, where only comments mention PWC). **It does import
`sieve_gpu_worker`** (`miner/range_miner_worker.py:1150`, `_get_kernel`, `coerce_threshold`) and
`prng_registry` (`:634`).

| # | Gap | Evidence | Files a fix touches | Shared with RANGE-MINER? | In parity set? |
|---|---|---|---|---|---|
| 1 | **Completion accounting** does not reach the trial; a pass with failed or timed-out chunks returns as complete | `persistent_worker_coordinator.py:1508-1549` vs `:1725-1727`, `:1766-1768`, `:1881-1898`; `:1453-1473`; Beta R1 Q1 | `persistent_worker_coordinator.py`; the PWC evidence call in `window_optimizer_integration_final.py:1896-1903` | PWC file: no. **Integration file: yes** (it hosts the miner branch `:1680-1849` and the credit gate `:3360-3380`) | No |
| 2 | **Seed start** is ignored: no parameter, partition from 0 | `run_trial_persistent` signature `:1632-1649`; call passes `total_seeds` only (`window_optimizer_integration_final.py:1871`); chunks from 0 `:1276`. The job and both workers already honour a per-job `seed_start/seed_end` (`:1309-1310`; `sieve_gpu_worker.py:199`; `sieve_filter.py:244`) | Support: `persistent_worker_coordinator.py` plus one kwarg at the integration call site. Refusal (owner R5): `window_optimizer.py`, beside `assert_miner_seed_start_supported` (`:678-703`), being built by the miner-start session | Support: PWC file no, call site yes. **R5 guard: `window_optimizer.py` is on the miner's launch path** | No |
| 3 | **Hybrid quarantine** | §P1.2. To lift it: Option A threshold propagation with a requested/payload/effective gate, **and** working skip-bound semantics in the hybrid kernels | Threshold: `persistent_worker_coordinator.py` call sites `:1802`, `:1829` (+ a gate). Skip bounds: hybrid kernels in `prng_registry.py`, which is an open design decision that also governs miner hybrids (skill §2.7 #4) | Threshold half: no. **Skip-bound half: yes** | Skip-bound half: **yes** (`prng_registry.py`) |
| 4 | **Publication.** The door is the same in code: same accumulator, same `_finalize_run_d3_5` (`:3393-3403`), same coverage writer (`:3448`), so Steps 2–6 would get the same 22-array generation. **But** (a) the S7 credit gate refuses PWC (`:1397-1403`, `:3369-3379`); (b) unlike the miner's validated threshold provenance, PWC has none (scope correction `:45-46`); (c) **the published `offset` field would misdescribe a PWC record**: normalization now writes `offset` as the *window anchor* (`TB_RULING_BRIEF_I_PATCH_REVIEW.md:60-72`), while PWC's kernels *also* pre-advanced the generator by that offset (`sieve_filter.py:311`; `sieve_gpu_worker.py:304`). For any trial with `offset > 0`, a PWC row and a miner row carrying the same field values encode different generator hypotheses. Steps 2–6 could not tell them apart, which is exactly why it matters for hit-rate attribution | (a) the PWC evidence builder in the integration file plus a ruling; (b) a PWC provenance record; (c) Brief II's consumer audit (`TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:70`) | (a), (b): integration file (shared). (c): **yes**, `sieve_gpu_worker.py` | (c): **yes** |
| 5a | **Window anchor / generator phase (Brief I)** — NOT honoured. One `offset` drives the slice and the kernel phase, with the silent clamp retained | `sieve_gpu_worker.py:113`, `:278`, `:304`; `sieve_filter.py:184`, `:311`, `:461`; `window_optimizer_integration_final.py:266`. Firewalled to Brief II (`S172_WINDOW_ANCHOR_BRIEF_I.md:82-83`, `:367`) | `sieve_filter.py`, `sieve_gpu_worker.py`, `persistent_worker_coordinator.py` | **`sieve_gpu_worker.py`: yes** | **`sieve_gpu_worker.py`: yes** |
| 5b | **Per-direction thresholds** — honoured for constant passes by reading (`:1717`, `:1758` → `sieve_filter.py:270` / `sieve_gpu_worker.py:215`). **No requested/payload/effective provenance and no fail-closed check.** Hybrid: no (0.50) | as cited | PWC file + workers | workers: `sieve_gpu_worker.py` yes | partly |
| 5c | **Session rule** — engine-agnostic. The sampler still offers combined-session first (`window_optimizer.py:181-186`; `TODO_CONSOLIDATED.md:207`, REVIEW). PWC forwards `config.sessions` (`:1722`), and the worker filters on it (`sieve_filter.py:179-180`). **Not a PWC-specific gap.** The fallback to both sessions when `config` lacks the attribute (`:1301`, `:1722`) is unreachable for a real `WindowConfig` (`sessions` is required: `window_optimizer_integration_final.py:1226-1237`) | — | — | — |
| 5d | **Dataset authority** — the frozen immutable path reaches PWC (`window_optimizer.py:1885`). Fleet digest verification runs when a manifest exists; a missing manifest is fatal only for miner runs (`miner/dataset_authority.py:1109`) | as cited | `window_optimizer.py` call (`miner_backed=` flag) | **yes** | (`dataset_authority.py` is pinned) |
| 5e | **Skip range** — honoured for constant passes (`:1723` → `sieve_gpu_worker.py:214`, `sieve_filter.py:236`). Default fallbacks differ, `[0,147]` coordinator vs `[0,16]` worker (`:1302` vs `sieve_gpu_worker.py:155`); both are unreachable for a real `WindowConfig` | — | — | — |
| 6a | **Hardcoded coordinator address.** Rig workers dial `192.168.3.127` (`:756`; default also at `persistent/pwc_worker_service.py:759`). `.127` is the frozen bare-metal Zeus; the canonical box is `.177`. **By reading, PWC-TCP launched from VM101 has no listener at the address its rig workers dial.** Not executed | as cited | `persistent_worker_coordinator.py`, `persistent/pwc_worker_service.py` | no | no |
| 6b | **Rig code is unverified.** On TCP the rig executes `persistent/pwc_worker_service.py`, `persistent/pwc_transport_tcp.py` and `sieve_filter.py`; none is in the parity set. The stale-rig discovery (skill §2.41) shows rig bytes cannot be assumed | `persistent/pwc_worker_service.py:29`, `:324`; parity pin `:174-185` | a separate PWC parity pin (new gate, reusing the gate's mechanism) | no (new gate) | n/a |
| 6c | **The manifest's PWC settings lie outside the recorded stable envelope.** Pool 8 with chunk cap 2,000,000 (`agent_manifests/window_optimizer.json:263`, `:265`; `chunk_cap = self.seed_cap_amd`, `:1272`). Recorded survivals: pool 6 at ≤100K, pool 8 at ≤25K (`ROCm_Saturation_Report_S172.md:29-35`), or 100K-cap runs (§4) | as cited | manifest values only | no | no |
| 6d | **Launch-per-trial.** The whole fleet is spawned and killed every trial (`:1676-1687`, `:1901`), with `pkill -f` on every rig (`:486-492`). This is the launch-storm shape the "operate like miners" constraint names (skill §2.50 `:2533-2541`) | as cited | `persistent_worker_coordinator.py` | no | no |
| 6e | `n_parallel > 1` partitions hardcode bare-metal rig IPs (`window_optimizer_integration_final.py:2392-2395`). Irrelevant while `n_parallel = 1` is binding | — | — | — | — |
| 6f | `persistent/pwc_protocol.py` raises `TypeError: non-default argument 'worker_id' follows default argument` on import (**executed**, §9). No production module imports it (grep: only tests name it) | ALREADY RECORDED (`CLAUDE.md` §6) | — | — | — |
| 6g | `HSA_OVERRIDE_GFX_VERSION=10.3.0` is injected into every PWC worker (`:140`, `:385`) although the rigs need no override (skill §6). Effect on gfx1032 not measured. `[UNVERIFIED]` whether harmful | as cited | `persistent_worker_coordinator.py` | no | no |

---

## P4 — The operating history

Sources were read by the sweep agent. Rows marked ✔ were re-read in this session.

| Date / session | Transport | Rigs / GPUs | Settings | Outcome | Source |
|---|---|---|---|---|---|
| 03-08 S130 | persistent workers in `coordinator.py`, SSH | 24 AMD | 10 × 20M; 200M throughput | 17/17 checks, 2.08M sps; kill test "did not land" | `SESSION_CHANGELOG_20260308_S130.md:56-62`, `:79-90` |
| 03-09 S132 | PWC SSH | 3 rigs | 8-way HIP | rrig6600c segfaults ~30–60 s in | `SESSION_CHANGELOG_20260309_S132.md:46-50` |
| 03-10 S135 | PWC SSH | 12 workers | 5M seeds | 2 trials, 0/14 chunk failures | `SESSION_CHANGELOG_20260310_S135.md:62-77` |
| 03-11 S137 | PWC SSH, n_parallel 2 | 26 GPUs | 10M, 3 trials | Steps 1→3 PASS | `SESSION_CHANGELOG_20260311_S137_FINAL.md:16-19` |
| 03-16 S146 | PWC SSH | not stated | 3 trials, 10M | zero crashes; pool 4 "validated stable (not 8)" | `SESSION_CHANGELOG_20260316_S146.md:94-103`, `:140` |
| 03-21 S149 | PWC | 24 AMD + 2 | pool 8 | soak 5/5 clean; Run 1 crashed after ~2 trials | `SESSION_CHANGELOG_20260321_S149.md:36-42`, `:156-163` |
| 03-23 S154 | PWC | 26 | Run 1 | rrig6600c OOM at ~28 min and ~7 min | `SESSION_CHANGELOG_20260323_S154.md:164-165` |
| 03-24 S155 | PWC | 22 (6600c at 4) | 3 trials | 3/3 complete; "8 workers always OOMs" | `SESSION_CHANGELOG_20260324_S155.md:80-92` |
| 03-26 S156 | PWC SSH | 8 + 8 + 3 | 3 trials, 1.07B | 3/3 complete | `SESSION_CHANGELOG_20260326_S156.md:112-131` |
| ✔ 03-29 S158 | PWC (S158B dispatch fix) | **rrig6600 + rrig6600c, 16 AMD + Zeus** | 537 chunks | **537/537, 14,046 bidi, 34:15, zero crashes** | `SESSION_CHANGELOG_20260329_S158.md:33`, `:36-40` |
| ✔ 03-29 S158 | same | **all 3 rigs, 24 AMD + Zeus** | 537 chunks | **hard kernel panic on rrig6600c** | `:34` |
| ✔ 03-29 S158 | same | rrig6600c alone | reverse | stalled at chunk 88/537 | `:32` |
| S161 | PWC-TCP | 24 AMD | 50M (25 × 2M) | gates passed, zero crashes | `SESSION_CHANGELOG_S161.md:33-40`, `:52-56` |
| 04-04 S162 | PWC-TCP | 26 | — | rrig6600c GCVM_L2 at ~18 s | `SESSION_CHANGELOG_S162.md:18-28` |
| 04-04 S162 | PWC-TCP | 6600c + Zeus (10) | 3 trials | clean, 18:52 | `SESSION_CHANGELOG_S162.md:36-40` |
| 04-02..05 S162 VICTORY | PWC-TCP | **26/26, 3 rigs** | **1.07B seeds, 3 trials, 100K chunk cap** | **42:36, netconsole silent**; 250 Zeus-local chunks failed in trial 3 | `SESSION_CHANGELOG_S162_VICTORY.md:14-23`, `:49`, `:64-65`, `:131` |
| 04-06 S163 | PWC-TCP | 3 rigs + Zeus | 500K cap | trial 1 clean, then hung | `TB_SUMMARY_S163.md:43-55` |
| 04-17 S163-KARG | PWC-TCP | 3 rigs | 100K, 5 trials | Step 1 complete; two GPU crashes, 6600c manual reboot | `SESSION_CHANGELOG_20260418_S163KARG_FORENSIC.md:10-24` |
| 04-20 S165 | PWC-TCP | 26/26 | 2 trials, 100K, 1.07B | ~14 min, netconsole silent | `SESSION_CHANGELOG_20260420_S165.md:168-178` |
| 04-24 S169 | PWC-TCP | 26 | 200K, 2 trials, jitter | "First clean 2-trial run" | `SESSION_CHANGELOG_20260424_S167_S168_S169.md:130`, `:144-149` |
| ✔ 04-25 S170 Test 1 | PWC-TCP | **2 rigs (rrig6600b removed)** | — | **"Still crashes."** | `SESSION_CHANGELOG_20260425_S170.md:58-64` |
| ✔ 04-25 S170 Test 2 | PWC-TCP | not stated | **pool 6** | **5/5 trials, no GCVM** | `:66-74` |
| 04-25 S170 Tests 3-4 | PWC-TCP | — | pool 8 + 50 ms / 500 ms cooldown | 50 ms crashed ~10 min; 500 ms **no outcome recorded** | `:98-110` |
| ✔ 2026-05-01 S172 | PWC | full 26-GPU | pool 6, 50K | 5 trials clean, 29:30 | `ROCm_Saturation_Report_S172.md:31` |
| ✔ S172 | same | same | pool 6, 100K | "STABLE — Per operator history" (no run cited) | `:32` |
| ✔ S172 | same | same | pool 8, 25K | 5 trials, 1:30:43, one transient eviction (recovered) | `:33` |
| ✔ S172 | same | same | pool 8, 50K / 100K | GCVM_L2 at ~23 / ~12 min | `:34-35` |
| 05-09 S174 D1 | PWC (S174 ready gate) | 3 rigs | 425M, 17,000 chunks | compute clean ~1.4M s/s; SMU breakdown ~1 s after, cascade | `SESSION_CHANGELOG_20260509_S174_D1.md:20-31` |

**Largest load shown carried cleanly, from the record only:**
- **Full fleet, full concurrency, at a small chunk cap:** 26 GPUs on 3 rigs, 1.07B seeds × 3 trials at
  a 100K chunk cap, netconsole silent (S162 VICTORY). The 250 failed chunks were on Zeus's local path,
  not the rigs.
- **The repeatable envelope** (S172 report, all three rigs): **pool 6 per rig at ≤ 50K** (5 trials,
  clean) and **pool 8 per rig at ≤ 25K** (5 trials, one recovered eviction).

**These two records conflict at pool 8 / 100K:** S162 VICTORY is clean; S172 crashes at ~12 min. I
report both and adjudicate neither. **No clean run at the manifest's current 2,000,000-seed cap with
pool 8 on three rigs was found.**

**Against the owner's statement:**
- "Underwhelmed" is supported. Every clean multi-rig record runs below 8 workers per rig, or at small
  chunks, or with pacing (S169, S170 Test 2, S172).
- "**Two rigs, not three**" has **one** supporting record: S158 `:33-34`, one ~34-min sweep. The session
  itself scopes it ("stable for 2-rig config", `:40`).
- It has **one contrary** record: S170 Test 1 (`:58-64`), "Still crashes … Not a cross-rig contention
  issue". The same session places the boundary at **workers per rig** (`:74`), and the S172 report
  states "The empirical boundary is concurrency-driven, modulated by chunk size" (`:19`).
- **The record does not support a rig-count limit.** It supports a per-rig concurrency × chunk-size
  limit.

**Gaps in the record:**
- No changelogs for S171, S173 or S175, or for most of May–June 2026.
- No outcome for S170 Test 4, S138's 200-trial run or S174 D2.
- The saturation report says "Six runs" (`:17`, `:27`) but tabulates five, and its pool6/100K row cites
  no run.

---

## P5 — Risk to RANGE-MINER

**Run 4 executes** `window_optimizer.py`, `window_optimizer_integration_final.py`, `miner/*` on VM101,
and the 10 parity-pinned files on the rigs. **A change to any of those before Run 4 changes the tree
Run 4 launches from.** Two committed tripwires also red on a dirty tree:
- `G-MINER-UNCHANGED` freezes `sieve_gpu_worker.py`, `prng_registry.py` and `persistent/pwc_protocol.py`
  by `git status` (`tests/test_s172_threshold_propagation.py:544-580`);
- Phase-4 gate 22 (`tests/test_s172_phase4_coordinator.py:1696-1714`; memory note on gate 22).

Both self-clear on commit and must not be widened.

| Fix (P3 #) | Touches a file RANGE-MINER loads? | How to keep the miner's behaviour unchanged | Before or after Run 4 |
|---|---|---|---|
| 6a address (from config or argument) | No (`persistent_worker_coordinator.py`, `persistent/pwc_worker_service.py`) | n/a | **Safe before.** Neither file is imported by the miner or deployed by the parity gate |
| 6c envelope (pool 6, ≤50K cap for PWC) | Manifest values `worker_pool_size` / `seed_cap_amd` are **read by the miner too** (`window_optimizer_integration_final.py:1733-1735`; Gate-12 shape `worker_pool_size = 25`) | Use PWC-only keys or a PWC launch profile; never change the shared defaults | Profile: after. Shared defaults: **never**, without a ruling |
| 1 accounting (PWC side) | No | Carry `failed_chunks`/`total_chunks` per pass in the v2 result; raise on any failed chunk | PWC file: **safe before** |
| 1 accounting (evidence builder) | **Yes**, the integration file | Edit only `_step1_unaccounted_route_evidence`'s PWC call (`:1896-1903`) or add a PWC-only builder; prove by AST diff that `run_bidirectional_test`'s miner branch (`:1680-1849`), `_step1_miner_sweep_evidence` and the credit gate (`:3360-3380`) are byte-identical; rerun the step-1 search suite | **After** (Run 4's launch file) |
| 2 seed start, refusal (owner R5) | **Yes**, `window_optimizer.py` | Already scoped and owner-decided in `CLAUDE_CODE_INSTRUCTIONS_MINER_SEED_START_20261114.md:92`, R7 (`:94`) requires identical stripes at `seed_start = 0` | Belongs to the miner-start brief, not this one |
| 2 seed start, support | PWC file no; call-site kwarg yes | Add the kwarg only inside the PWC branch | After |
| 3 hybrid, threshold half | No | PWC call sites + a requested/payload/effective gate | Could be built before, but **pointless alone**: the lifting condition also needs the skip-bound half |
| 3 hybrid, skip-bound half | **Yes**, `prng_registry.py` kernels (parity-pinned, shared by miner hybrids) | Cannot be kept from the miner: it is the same open decision (skill §2.7 #4; DEP-ABI territory) | **After**, and only under its own ruling |
| 4 publication credit for PWC | **Yes**, integration file + `utils/step1_search_ledger.py` semantics | Do not alter `credit_decision`; give PWC real evidence instead (fix 1) | **After**, and **needs a ruling** (Beta R1 Q1 / R2) |
| 4c / 5a anchor-vs-phase | **Yes**, `sieve_gpu_worker.py` (imported by `miner/range_miner_worker.py:1150`; parity-pinned) | Do the PWC side in `sieve_filter.py` (TCP worker, not pinned) and leave `sieve_gpu_worker.py` untouched, or change only defs the miner never calls (`load_draws_cached`, `run_sieve_job`) and redeploy with a 30/30 parity re-pass | **After.** Brief II scope already owns it |
| 5b provenance record | integration file | PWC-only record; miner's D6 path untouched | After |
| 6b PWC parity gate | No (new script) | Separate pin; do not edit `GOVERNED_FILES` | **Safe before** (read-only gate) |
| 6d per-trial launch → per-run fleet | No | PWC file only | Safe before by isolation, but it is a lifecycle redesign (`PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md` exists, "APPROVED WITH MODIFICATIONS", catalog `:601`, **not read this session**) |

**Operational risk, separate from code.** PWC and the miner share the same rigs. PWC uses port 5600 and
the miner 5700 (`v1_4_4 :473`). PWC's startup runs `pkill -f "sieve_gpu_worker.*--persistent"` and
kills `pwc_worker_service` on every rig (`:486-492`, `:715-724`). Those patterns do not match
`range_miner_worker.py` by reading, but **no PWC launch should overlap a miner run on the same rigs.**
That is a scheduling rule, not a code change.

---

## P6 — Options (none implemented; none recommended beyond what P1–P5 support)

**Option 0: diagnostic backup, no ruling change.**
- **Delivers:** PWC constant-skip can be run deliberately (e.g. `test_both_modes = false`, two rigs via
  `node_allowlist`, pool 6, ≤50K cap) to produce survivors and per-pass JSON for inspection when the
  miner is unavailable. **Nothing it produces publishes, counts as coverage, or reaches Steps 2–6.**
- **Cost:** fix 6a (address), plus a PWC launch profile (6c). Both sit outside every miner-loaded file.
- **Ruling changed:** none. This is the 2026-07-31 status quo, "flag-selectable, non-certifying".
- **Owner decides:** whether a non-counting backup is worth having.
- **Evidence it works:** one two-rig constant-only PWC trial from VM101 that completes with
  `failed_chunks = 0` on every pass and no rig fault, owner-initiated.

**Option A: smallest backup that counts (constant-skip only, two rigs).**
- **Delivers:** a PWC Step-1 chunk that the S7 gate can credit and D3.5 can publish into the same
  22-array generation.
- **Cost:**
  - Option 0, plus fix 1 (pass accounting carried to the trial, any failed chunk fails the trial, a
    PWC evidence builder that reports the swept interval from completed chunks only).
  - Owner R5 (refuse `seed_start != 0`), or fix 2 support.
  - Constant-only enforced for PWC.
  - A PWC parity pin (6b).
  - A decision on 4c: until Brief II, PWC rows at `offset > 0` encode `generator_phase = offset` while
    the record field says anchor only. Either (i) restrict counted PWC trials to `offset = 0`, where the
    two meanings coincide, or (ii) wait for Brief II.
  - Credit and publication changes land after Run 4.
- **Rulings changed:** **Beta 2026-07-31** (PWC regains certifying authority for this scope) and **Beta
  X-STEP1-SEARCH-CODE R1 Q1 / R2** (PWC refused credit). The quarantine is untouched.
- **Owner decides:** both rulings; the 4c choice; whether PWC-credited coverage and miner-credited
  coverage may share one ledger and one generation lineage.
- **Evidence it works:**
  - a failed-chunk case through the real PWC builder giving no credit, with a zero-survivor completed
    sweep as the paired control (Beta's own requirement, R1 `:95`);
  - a mutant that drops the accounting reds;
  - an owner-initiated two-rig run credited end to end;
  - later, the hit rate of pools trained on PWC-credited generations, measured the same way as the
    miner's.

**Option B: full backup (both modes, full fleet).**
- **Delivers:** PWC as an interchangeable engine.
- **Cost:**
  - Option A;
  - lifting the hybrid quarantine (Option A thresholds with provenance **and** the hybrid skip-bound
    decision, in shared kernels);
  - Brief II's anchor/phase split on the PWC workers;
  - threshold provenance with fail-closed enforcement;
  - per-run instead of per-trial fleet lifecycle (6d);
  - a measured stable envelope for the full fleet.
- **This is the cost the scope correction priced:** "the same multi-week effort already spent on the
  miner, spent again" (`:62-66`).
- **Rulings changed:** both above, plus comparator ruling §5 (quarantine).
- **Owner decides:** all of the above. And whether full-fleet PWC re-enters the launch-storm condition
  RANGE-MINER was built to remove (skill §2.50).
- **Evidence it works:** Option A's evidence, a hybrid known-answer transfer through PWC, and a
  multi-trial full-fleet soak with no rig fault.

**What P1–P5 support.** Only Option 0, and the PWC-file halves of Option A (6a, 6b, the PWC side of
fix 1), can be built **before Run 4** without touching anything RANGE-MINER loads. Everything that makes
a PWC result count touches Run 4's launch files, needs a Beta ruling to change, and is the owner's to
decide.

---

## P7 — ZMQ

ZMQ+SQLite (S158D, proposed 2026-03-29) is a pull-model Step-1 transport. Zeus binds ZMQ PUSH/PULL, an
SQLite table holds chunk leases (pending → claimed → done), and each rig's `zmq_sqlite_worker.py` is
launched once over SSH (`PROPOSAL_ZMQ_SQLITE_COORDINATOR_S158D_v1_0.md:1-50`).

It **did run**, all in early April 2026:
- **S159**: a 2-rig validation (`S159_CHAT_PROMPT.md:5-9`).
- **S159G**: a 3-rig run in which rrig6600 crashed and its leased chunks were taken by the surviving
  rigs (`SESSION_CHANGELOG_20260402_S159G.md:10-14`, `:51-57`).
- **S160**: a 3-rig acceptance trial with zero crashes, at ~31K sps/GPU with ~26 min stalls
  (`SESSION_CHANGELOG_S160.md:22`, `:70-91`).
- **S161**: benchmarked at ~800K sps against TCP-PWC's 2.24M on 24 GPUs (`SESSION_CHANGELOG_S161.md:54-58`),
  after which TCP-PWC replaced it.

None of these produced production Step-1 output. ZMQ was retired from certifying authority with PWC on
2026-07-31 (`CHAPTER_9_AUDIT.md:100-103`). It is **not quarantined**: no guard exists in
`zmq_sqlite_coordinator.py`. It is off by default (`agent_manifests/window_optimizer.json:267`). Like
PWC, it partitions from seed 0 (`zmq_sqlite_coordinator.py:830-834`, read here). It returns v2 four-map
results that are refused credit (`window_optimizer_integration_final.py:1910-1965`; Beta R2 `:78`), and
owner R5 will make it refuse a non-zero `seed_start`. Sources: the ZMQ sweep agent; `:830-834` and the
R2 line re-read here.

---

## UNAVAILABLE

- **The 2026-07-31 Beta ruling text and comparator ruling §5 text.** Not in the repository. Only
  secondary quotations exist.
- **What PWC/ZMQ code is deployed on the three rigs** (no SSH by rule; not parity-pinned).
- **Whether PWC-TCP can connect from VM101 at all.** Inferred from `:756`, not executed.
- **Outcomes** of S170 Test 4, S138's 200-trial run, S174 D2, and the S172 pool6/100K row's underlying
  run.
- **Any hit-rate measurement** of PWC-produced survivors.
- `PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md`: **listed, not read** this session.
- The private repository, ser8, project PDFs, `~/dashboard_work/`.

---

## Commands used

```
# setup (VM101, michael)
cd /home/michael/distributed_prng_analysis && git rev-parse HEAD
git worktree add --detach /home/michael/tfm_worktrees/pwc-backup HEAD

# governance search: the commands listed under "Prior record"

# code reads (worktree)
sed -n / Read: persistent_worker_coordinator.py:1-660, 659-925, 1057-1124, 1194-1957
sed -n:  window_optimizer_integration_final.py:240-300, 1162-1272, 1390-1410, 1647-1910, 2365-2395, 3330-3470
sed -n:  utils/step1_search_ledger.py:1044-1095
Read:    sieve_gpu_worker.py:98-312
grep:    sieve_filter.py (offset/threshold/seed lines); persistent/pwc_worker_service.py imports
sed -n:  window_optimizer.py:905-930, 1765-1785, 1850-1905; miner/dataset_authority.py:1095-1125, 1255-1275
sed -n:  miner/range_miner_worker.py:1140-1160; scripts/gate12_parity_gate.py:165-200
sed -n:  zmq_sqlite_coordinator.py:826-836; tests/test_s172_threshold_propagation.py:540-580
grep:    agent_manifests/window_optimizer.json backend defaults

# executed (CPU only, no bytecode, nothing written)
source ~/venvs/torch/bin/activate
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 -c "import persistent.pwc_protocol"
  -> TypeError: non-default argument 'worker_id' follows default argument
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python3 -c "import persistent.pwc_transport_tcp; import persistent.pwc_transport_base"
  -> tcp OK / base OK
git status --porcelain   -> empty
```

---

*sha256 of this report: see the closing line printed by
`sha256sum docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md` in the session reply. A file cannot contain
its own digest, so the value is recorded outside the file.*
