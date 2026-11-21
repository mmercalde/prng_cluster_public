# PWC full backup, Stage 4: rig redeploy and first two-engine session (PLAN, read-only)

Written 2026-11-21 by Claude (Team Alpha lead) in `/home/michael/tfm_worktrees/pwc-backup`, HEAD `6d75f71`. **Planning
only.** No code was edited, no rig was powered, contacted or reached by SSH, and no GPU ran. Every rig fact below
comes from the repository record. Anything about what the rigs hold *now* is UNVERIFIED until the parity gates run.

Worktree state before writing: every one of the 98 files in `b2efdef..6d75f71` was byte-identical to this worktree
(sha256, 0 mismatches). HEAD and index were then moved to `6d75f71` with `git reset --mixed` (no file touched);
`git status` shows only the untracked `data/`. The VM101 main tree `/home/michael/distributed_prng_analysis` is at
`6d75f71` with a clean `git status --porcelain` (read-only check).

## Status update, 2026-11-21: the redeploy in §2 is DONE

Recorded from the owner's message of 2026-11-21. Alpha ran it at the owner's go, 11:20-11:25 PST.

| Step | Result |
|---|---|
| Bring-up | All three rigs up in Proxmox, CT100 running, 8 AMD cards each, no workers anywhere |
| PWC parity, before (§2 step 3) | REFUSED, 27 match / 12 mismatch: the four files × three rigs, none missing (as §1c expected) |
| gate12 parity, before | PASS 30/30 |
| Copy (§2 step 4) | The four files from `git archive 6d75f71`; rollback copies in `~/rig_archive/pre_stage4_20261121T192337Z` on each CT100 |
| PWC parity, after (§2 step 5) | PASS 39/39 |
| gate12 parity, after (§2 step 6) | PASS 30/30, digests identical to before |
| `__pycache__` clearing | **NOT done** (no owner OK; §6 question 3 stays open) |
| Rigs afterwards | Left up. Owner, 11:34: "leave the rigs up- we dont want any more issues- proceed with the work" |

Evidence: in the VM101 main tree, `data/stage4/pwc_parity_pre.json`, `pwc_parity_post.json`,
`gate12_parity_pre.json`, `gate12_parity_post.json`. Read back by this session (read-only, `rows[].verdict`):
- `pwc_parity_pre`: 27 MATCH, 12 MISMATCH; mismatched paths are exactly the four files.
- `pwc_parity_post`: 39 MATCH.
- `gate12_parity_pre` / `_post`: 30 MATCH each, and the 30 observed digests are identical pre and post.
- All four have `local_head_context_only` = `6d75f71d7b63…`; generated 19:22:15Z-19:24:45Z.

The Stage 4 session itself is now specified in `SESSION_RUNBOOK.md` (this folder). It replaces §4b's gaps 1, 3, 4
and 6 with a separate-run-tree design. §4b's facts stay as written.

## Prior record

**Searches run** (all with `/bin/grep -i -E`, this worktree at `6d75f71`; pattern
`(PWC|persistent worker).{0,60}(redeploy|deploy|stage 4|rig session)|pwc_worker_service|pwc_parity_gate`):

| Scope | Files with hits |
|---|---|
| `docs/PROJECT_FILE_CATALOG.md` | 0 |
| `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*` | 0 |
| `docs/PROPOSAL_*` | 1 (`PROPOSAL_S162_RRIG6600C_CRASH_ROOT_CAUSE_v1_0.md`, April 2026: the bare-metal era) |
| `docs/CHAPTER_*` | 0 |
| `docs/*_AUDIT.md` | 1 (`HYBRID_SKIP_BOUND_AUDIT.md`, the skip-range audit, not deploys) |
| `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md` | 0 |
| `docs/exchange/` | 18 (the PWC Step 1/Stage 1-3 review rounds) |

Further targeted searches:
- `git archive [0-9a-f]{7}`, `reflog`, `scp`/`deploy` near the PWC file names, across `docs/`;
- `parity gate` in the 2026-11 handoffs and drafts;
- `rocm-smi`, `SIGTERM`, `live ledger` and `rigpower` across `docs/`;
- `git log` of every rig-side file.

Live code was read at the anchors cited.

**Documents that already decide, schedule or record the subject:**

| Document | What it records | Label |
|---|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47` | Owner, 2026-11-20 15:55: "once finished your testing should run each and verify the results are the same" (the Stage 4 test) | ALREADY RECORDED |
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:51` | S3-13 withdrawn; PWC keeps the trial's skip range; RANGE-MINER skip fix scoped separately | ALREADY RECORDED |
| `docs/drafts/pwc_backup_stage3_v1/PLAN.md` §F (`:282-379`), F4 `:342-379` | Stage 4 design: inputs held equal, outputs compared, smallest session | ALREADY RECORDED (this plan builds on it) |
| `docs/drafts/pwc_backup_stage2_v1/README.md:627-639` | "What a later rig redeploy would involve" (Stage 2 files) | ALREADY RECORDED, superseded in its file list by this plan |
| `docs/drafts/pwc_backup_stage2b_v1/README.md:369-381` | "Deploy: the rigs and VM101 must change together"; mixed versions fail loudly | ALREADY RECORDED |
| `docs/drafts/BRIEF_PWC_BACKUP_STAGE2_STAGE3_20261119.md:185-200` | The scp, then the PWC gate, then the gate12 gate, no CuPy recompile | ALREADY RECORDED |
| `docs/exchange/from_beta/X-PWC-STAGE3_R1_BETA_REVIEW.md:101` (3b-Q6) | "Verify PWC parity and upgrade the VM101 integration and coordinator together before using the credited route" | ALREADY RECORDED (Beta SUGGESTION, carried to Stage 4) |
| `docs/TFM_SESSION_HANDOFF_20261120.md:62-65` | "Nothing was redeployed to the rigs"; PWC bytes on rigs UNVERIFIED | ALREADY RECORDED |
| `docs/PHASE6_PREREQS.md:103-131`, `docs/S172_PHASE_7_PREREQ_REPORT.md:490-526` | The deploy method (git archive, `mkdir -p`, `scp -p`, verify, clear `__pycache__`), 2026-08-02 | ALREADY RECORDED |
| `docs/RUN4_LAUNCH_CHECKLIST.md` §1, §6, §7, §8, §11 (`:59-88`, `:160-213`, `:215-256`, `:304-345`) | Bring-up, deploy, gate12 30/30, netconsole, search open | ALREADY RECORDED |
| `docs/drafts/RUN4_FIRST_RUN_CHECKLIST_20261118.md:31-35`, `:69` | Parity read live; last miner deploy 30/30 on 2026-11-15 (`048fd47`) | ALREADY RECORDED |
| `docs/RIG_POWER_CONTROL_TUYA_REFERENCE.md:13-19`, `:50-79` | GPU PSUs by hand; `rigpower.py` rules of use | ALREADY RECORDED |
| `docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md:36-40` | Never SIGTERM a live coordinator; live ledger off-limits except what the owner approves each time | ALREADY RECORDED |
| `docs/TFM_SESSION_HANDOFF_20261119.md:16`, `:42-75` | Run N card drop (rrig6600b gpu6) and PCIe survey | ALREADY RECORDED |

**Labels for what this plan raises:**
- The file list, deploy steps and gates in §1-§3 are ALREADY RECORDED; only the narrowing to four files is new.
- These are NEW and need the owner:
  - the session shape in §4, including the gaps found: one-rig launch, per-phase capture, the search records;
  - the extractors;
  - the questions in §6.
- Nothing here CHANGES A RULING.

**Not searched:** the host and Proxmox, ser8 archives, the private repository, project files outside the
repository, and the rigs themselves (powered off; no SSH by instruction).

---

## 1. Which files the rigs need at `6d75f71`, and what they hold now

### 1a. What a PWC rig worker executes

On TCP, the coordinator starts `python -m persistent.pwc_worker_service` on each rig GPU, from the CT100's
`script_path` (`persistent_worker_coordinator.py:1462-1484`). The PWC parity gate pins the files that run:

- TCP pin, 13 files: `scripts/pwc_parity_gate.py:107-123`.
- SSH pin, 4 files: `:124-130`.

Each pin is cross-checked against an AST-derived import closure that includes package `__init__.py` files
(`:194-221`). I ran that derivation locally on CPU at `6d75f71` (no SSH; output in
`data/pwc_stage4_plan/closure_check_6d75f71.txt`). Both closures equal their pins: `closure_coverage` is
`(True, [], [])` for `tcp` and for `ssh`. **No new file joined the worker's import closure in Stages 1-3.**

The coordinator itself (`persistent_worker_coordinator.py`) and the integration run on VM101 only. They are not
copied to rigs.

### 1b. The last recorded deploys

| When | What | Record |
|---|---|---|
| 2026-07-30 | `rrig6600` CT100: `git clone` from the public remote at `8e2f5bf` (later 84 dirty entries). `rrig6600b`/`c` have no git repository; their files came by `scp` | `docs/S172_PHASE_6_0_ROCM_PARITY_EVIDENCE.md:9-22`, `:361`; `docs/PHASE6_PREREQS.md:122-131` |
| 2026-08-02 | `miner/` and `utils/` only, from `git archive 18a2419`, 17 files, to all three CT100s | `docs/S172_PHASE_7_PREREQ_REPORT.md:505-526`; `docs/PHASE6_PREREQS.md:103-108` |
| 2026-11-15 | the gate12 set, read 30/30 after `048fd47` (miner worker) was deployed | `docs/drafts/RUN4_FIRST_RUN_CHECKLIST_20261118.md:69`; `docs/LEADS.md:495` |
| 2026-11-19 | Runs P and Q launched through `gate12_launch.sh`, which runs `scripts/gate12_parity_gate.py` before dispatch (`gate12_launch.sh:192`); main was `7c725c8` | `docs/TFM_SESSION_HANDOFF_20261119.md:16-20`, `:28` |
| 2026-11-19 | `7444aea` (PWC cache reset): "not in the rig parity set, so no redeploy" | `docs/TFM_SESSION_HANDOFF_20261119.md:30` |
| 2026-11-20 | Stages 1, 2, 2b: "Nothing was redeployed to the rigs" | `docs/TFM_SESSION_HANDOFF_20261120.md:62-65` |

**NOT FOUND: any recorded deploy of the PWC worker files to a CT100.** Surfaces searched: `docs/` (the patterns
above), the handoffs from 2026-11-13 to 2026-11-20, `PHASE6_PREREQS.md`, `S172_PHASE_7_PREREQ_REPORT.md`, the two Run 4
checklists, the Stage 1/2/2b READMEs, and `git log` of every pinned file. The last PWC worker deploys on record are
from April 2026 to the bare-metal rigs: `SESSION_CHANGELOG_S162.md:33`, `SESSION_CHANGELOG_20260418_S163KARG_FORENSIC.md:205-225`.
The CT100s did not exist then.

### 1c. Byte comparison of every rig-side file (sha256, first 12 hex)

**T** marks the PWC TCP pin and **G** the gate12 pin (`scripts/gate12_parity_gate.py:174-185`). `sieve_gpu_worker.py`
(G) is also the whole SSH pin, with three T files.

| File | pin | `8e2f5bf` clone | `18a2419` | `7c725c8` Runs P/Q | `b2efdef` | **`6d75f71`** |
|---|---|---|---|---|---|---|
| `adaptive_thresholds.py` | T G | cda23bb3cd62 | = | = | = | = |
| `core/results_manager.py` | T | a370d5dbe135 | = | = | = | = |
| `hybrid_strategy.py` | T G | 8b982fbb8ecb | = | = | = | = |
| `integration/__init__.py` | T | e3b0c44298fc | = | = | = | = |
| `integration/metadata_writer.py` | T | 00a9f3093110 | = | = | = | = |
| `integration/sieve_integration.py` | T | 336343dc024f | = | = | = | = |
| `persistent/__init__.py` | T | e3b0c44298fc | = | = | = | = |
| `persistent/pwc_result_normalizer.py` | T | 3480948e9996 | = | = | = | = |
| **`persistent/pwc_transport_base.py`** | T | 297a075d1320 | = | = | 504ad46f9b6e | **504ad46f9b6e** |
| **`persistent/pwc_transport_tcp.py`** | T | 156388c7f1fd | = | = | a09c5aba4d00 | **a09c5aba4d00** |
| **`persistent/pwc_worker_service.py`** | T | 9ca2f7b217ec | = | = | f50d548905c3 | **f50d548905c3** |
| `prng_registry.py` | T G | 32435fb82737 | = | = | = | = |
| **`sieve_filter.py`** | T | b44454da3239 | = | = | a9735e8337eb | **50a977bffa82** |
| `execution_set.py` | G | absent | d21614701c31 | = | = | = |
| `miner/__init__.py` | G | 7581155a7300 | 699e930e379d | = | = | = |
| `miner/dataset_authority.py` | G | absent | 365c8e3ee9ab | = | = | = |
| `miner/range_miner_coordinator.py` | G | 9e270c49da7a | 70f8fbaa371c | b8e259b3e12b | = | = |
| `miner/range_miner_protocol.py` | G | 75cfd3a2b54a | = | 7b7f8e197914 | = | = |
| `miner/range_miner_worker.py` | G | 5ca9880946e6 | 0b9a7b86b0cf | 2bca2a083262 | = | = |
| `sieve_gpu_worker.py` | G | 9ace0d83ad99 | = | = | = | = |

"=" means the same as the cell to its left.

**What this shows:**
- **The ten gate12 files are byte-identical from `7c725c8` (the tree whose gate12 parity passed for Runs P/Q) to
  `6d75f71`.** Nothing for RANGE-MINER is to be copied.
- **Four PWC files changed since any CT100 could have received them:** `persistent/pwc_transport_base.py`,
  `persistent/pwc_transport_tcp.py`, `persistent/pwc_worker_service.py` (Stages 1, 2 and 2b) and `sieve_filter.py`
  (Stage 2b, then Stage 3).
- Between `8e2f5bf` and `7c725c8` no PWC file changed. So if `rrig6600` still holds its clone's bytes for these files,
  it carries the pre-Stage-1 worker. That is [UNVERIFIED]: the clone has 84 dirty entries, and `rrig6600b`/`c` have no
  git record at all. As the record says, digest comparison is the only evidence (`docs/PHASE6_PREREQS.md:129-131`).
- **The SSH route needs nothing:** its four files are unchanged.

**What the rigs hold now: UNVERIFIED** (rigs off, no SSH). The PWC parity gate measures it (§2, step 5). Expectation
from the record: the four files mismatch on all three rigs, and the nine others match. If any of the nine
mismatches, stop and report. It would mean a rig carries bytes the record does not explain.

### 1d. What happens if only one side is updated (why the rigs must be done before Stage 4)

VM101 already runs `6d75f71`. With old rig workers, every PWC TCP pass fails loudly and is marked incomplete, which
Stage 3 now refuses before scoring:
- old rig worker, new coordinator: no echo, `pwc_anchor_echo_error`;
- new worker, old coordinator: `PWC_LEGACY_OFFSET_REFUSED`.

Sources: `docs/drafts/pwc_backup_stage2b_v1/README.md:372-379`, Beta 3b-Q6
(`X-PWC-STAGE3_R1_BETA_REVIEW.md:101`), and the fail-closed trial check `window_optimizer_integration_final.py:2066-2072`.
There is one more effect: a refused pre-Stage-2 worker keeps retrying every 5 s (`docs/TFM_SESSION_HANDOFF_20261120.md:63-64`). That is one
more reason to redeploy before any PWC run.

---

## 2. The deploy procedure, step by step

The method is the recorded one, unchanged: `docs/RUN4_LAUNCH_CHECKLIST.md` §6 (`:160-200`), which cites
`docs/S172_PHASE_7_PREREQ_REPORT.md:505-526` and CLAUDE.md §2. Only the file set differs. Who runs it is also
recorded: "OWNER, or SESSION at your word (delegation A6)" (`RUN4_LAUNCH_CHECKLIST.md:160`).

**Source:** `git archive 6d75f71` of the four files, exported to a folder outside the repository (keeps the tree
clean, `RUN4_LAUNCH_CHECKLIST.md:170-171`).

**Destination:** `script_path` on every node is `/home/michael/distributed_prng_analysis` in
`distributed_config.json` (read 2026-11-21). The CT100 endpoints under the `proxmox` profile are `.122/.156/.164`
(`rig_profiles_config.json`, `default_profile: proxmox`). On each CT100 the files go to:
- `/home/michael/distributed_prng_analysis/persistent/pwc_transport_base.py`
- `/home/michael/distributed_prng_analysis/persistent/pwc_transport_tcp.py`
- `/home/michael/distributed_prng_analysis/persistent/pwc_worker_service.py`
- `/home/michael/distributed_prng_analysis/sieve_filter.py`

**Steps.** A failed step means stop. No step works around a refusal.

0. **Owner go for the bring-up, said in chat, and the GPU PSUs confirmed on** (they are, per the owner today).
   Alpha tells the owner which rig and when, then runs `rigpower.py on <rig>`
   (`RIG_POWER_CONTROL_TUYA_REFERENCE.md:70`, `:73`). Then, per rig:
   - run `boot-proxmox`;
   - re-arm netconsole;
   - stop `rig-power-telemetry.service`;
   - run `pct start 100` (`:74`).

   `rig-6600` may need one `cycle` if it does not answer about two minutes after `on` (`:79`).
   **A deploy uses no GPU, but booting the rigs is still a power-up and needs the go.**
1. Each CT100 answers with its own name and `ROCm_OK` (`RUN4_LAUNCH_CHECKLIST.md:73-88`). `rocminfo` here runs
   before any card works.
2. No workers running on VM101 or any CT, by bracket `pgrep` (`RUN4_LAUNCH_CHECKLIST.md` §3, `:105-120`).
3. **Before-deploy control:**
   ```
   python3 -u scripts/pwc_parity_gate.py --transport tcp --evidence-json data/stage4/pwc_parity_pre.json
   python3 -u scripts/gate12_parity_gate.py --evidence-json data/stage4/gate12_parity_pre.json
   ```
   Run both from the clean VM101 main tree.
   - The PWC gate is expected to REFUSE, naming the four files.
   - The gate12 gate is expected to PASS 30/30.

   Keep both outputs. If gate12 does not pass 30/30 here, stop: something other than this plan has moved.
4. **Export and copy**, the `RUN4_LAUNCH_CHECKLIST.md:168-193` block with the file list changed:
   ```
   test -z "$(git status --porcelain)" || { echo "TREE NOT CLEAN - stop"; false; }
   STAMP=$(date -u +%Y%m%d_%H%M%SZ); EXP=/home/michael/stage4_deploy_6d75f71; mkdir -p "$EXP"
   git archive 6d75f71 persistent/pwc_transport_base.py persistent/pwc_transport_tcp.py \
       persistent/pwc_worker_service.py sieve_filter.py | tar -x -C "$EXP"
   sha256sum "$EXP"/persistent/*.py "$EXP"/sieve_filter.py      # must equal §1c's 6d75f71 column
   for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
     ssh -n michael@$ip "mkdir -p /home/michael/rig_archive/pre_stage4_${STAMP}/persistent && \
       cp -p /home/michael/distributed_prng_analysis/persistent/pwc_transport_base.py \
             /home/michael/distributed_prng_analysis/persistent/pwc_transport_tcp.py \
             /home/michael/distributed_prng_analysis/persistent/pwc_worker_service.py \
             /home/michael/rig_archive/pre_stage4_${STAMP}/persistent/ && \
       cp -p /home/michael/distributed_prng_analysis/sieve_filter.py /home/michael/rig_archive/pre_stage4_${STAMP}/ && \
       mkdir -p /home/michael/distributed_prng_analysis/persistent"
     scp -p "$EXP"/persistent/pwc_transport_base.py "$EXP"/persistent/pwc_transport_tcp.py \
            "$EXP"/persistent/pwc_worker_service.py michael@$ip:/home/michael/distributed_prng_analysis/persistent/
     scp -p "$EXP"/sieve_filter.py michael@$ip:/home/michael/distributed_prng_analysis/
     ssh -n michael@$ip 'hostname; cd /home/michael/distributed_prng_analysis && \
       sha256sum persistent/pwc_transport_base.py persistent/pwc_transport_tcp.py persistent/pwc_worker_service.py sieve_filter.py; \
       rm -rf persistent/__pycache__ __pycache__'
   done
   ```
   - The `cp -p` archive is the rollback copy, as in the Run 4 deploy (`RUN4_LAUNCH_CHECKLIST.md:194-196`).
   - `mkdir -p` comes before each `scp`, with absolute paths (CLAUDE.md §2).
   - No `pip`, `apt` or Python change (`RUN4_LAUNCH_CHECKLIST.md:197-199`).
   - The top-level `__pycache__` removal is NEW to this plan. `sieve_filter.py` sits at the root, unlike the
     miner files. Without it, a stale `sieve_filter.cpython-*.pyc` could be picked up only if its source timestamp
     matched, and `scp -p` keeps the source mtime. Clearing it removes that doubt. The cost is regenerable bytecode.
     It needs the owner's OK.
5. **After-deploy PWC gate, must PASS:**
   `python3 -u scripts/pwc_parity_gate.py --transport tcp --evidence-json data/stage4/pwc_parity_post.json`.
   It derives expected digests from the local tree at run time (`scripts/pwc_parity_gate.py:232-241`). It refuses on
   any MISMATCH, MISSING, UNAVAILABLE, wrong hostname, dirty local file or zero-rig fleet (`:41-56`). This is the hash check.
6. **After-deploy gate12 gate, must PASS 30/30 with the same digests as step 3:**
   `python3 -u scripts/gate12_parity_gate.py --evidence-json data/stage4/gate12_parity_post.json`. This shows
   RANGE-MINER's ten files did not move. None of the four copied files is in its pin (§1c).
7. No kernel source changed (`prng_registry.py`, `sieve_gpu_worker.py` unchanged), so no CuPy recompile is forced.
   The per-run cache reset applies anyway: PWC `persistent_worker_coordinator.py:1434-1436`, and the miner's
   `per_run` policy.
8. **If the owner wants the rigs off afterwards:** clean shutdown first, then `rigpower.py off` on the three rig
   outlets only (`RIG_POWER_CONTROL_TUYA_REFERENCE.md:65`, `:72`). The GPU PSUs stay as the owner left them.

**Scope note.** The parity gates as written need all three rigs up:
- the PWC gate has `--nodes` for a partial fleet (`scripts/pwc_parity_gate.py:572-574`);
- the gate12 gate has no such option (`scripts/gate12_parity_gate.py:698-710`) and reads `UNAVAILABLE` for a powered-off
  rig, which is a REFUSE.

**Redeploy all three rigs in one bring-up**, even if Stage 4 then uses only one. The fleet stays uniform and both
gates can read a full PASS.

---

## 3. Is a worker restart or service change needed? How PWC workers start

- **No service exists and none is needed.** PWC rig workers are not a systemd unit or a daemon left running. At the
  start of each PWC run, the coordinator, from VM101 over SSH:
  1. probes the live GPU count with `cupy.cuda.runtime.getDeviceCount()` before any worker
     (`persistent_worker_coordinator.py:968-1000`);
  2. kills stale `pwc_worker_service`/`sieve_gpu_worker` processes, reading the result and refusing the rig if it is
     not confirmed (`:1409-1432`);
  3. resets the CuPy cache (`:1434-1436`);
  4. writes one launch script per GPU and starts each worker with `nohup ... -m persistent.pwc_worker_service`, with
     `PWC_HOST`/`PWC_PORT` exported (`:1438-1500`);
  5. starts VM101's own local CUDA worker the same way (`:1543-1580`; `min(2, gpu_count)`, and `gpu_count` is 1 in
     `distributed_config.json`).

  The record says the same: "no new script, no systemd unit" (`docs/S172_PHASE_7_PREREQ_REPORT.md:507-510`).
- **After the deploy there is nothing to restart:** the rigs come up cold. A new run loads the new bytes.
- **A PWC run is chosen by flags:** `use_persistent_workers: true` **and** `use_range_miner: false`
  (`docs/drafts/pwc_backup_stage3b_v1/README.md:236`; RANGE-MINER is the manifest default,
  `agent_manifests/window_optimizer.json:270-275`). It also needs:
  - `pwc_transport: tcp`;
  - `pwc_coordinator_address`, VM101's address that the rig workers dial (no default; PWC refuses without it,
    `window_optimizer.py:1580-1584`);
  - an AMD chunk cap inside the stable envelope, which is 8 workers per rig at ≤ 25,000 seeds or 6 at ≤ 50,000
    (`persistent_worker_coordinator.py:542-546`). The default 2,000,000 is refused. The NVIDIA cap is not used for
    chunking.
- **RANGE-MINER workers start differently:** by hand-launched daemons, `scripts/launch_fleet_manual.sh`, which
  `gate12_launch.sh:312-320` calls before the coordinator. That script brings up every node in
  `rig_profiles_config.json` (`scripts/launch_fleet_manual.sh:252-259`). See §4b.

---

## 4. The smallest Stage 4 session (PLAN.md F4), and what is still missing for it

### 4a. The session as specified

- **Machines:** one rig (8 cards) plus VM101's 3080 Ti.
- **Job:** one trial of `java_lcg` with both modes (`test_both_modes: true`), all four phases, and
  `window_trials: 1`, `n_parallel: 1`.
- **Pinned trial:** the six warm-start pins plus `session_idx`. With all six set, that trial is enqueued as trial 0
  (`window_optimizer_bayesian.py:779-799`), and with one trial it is the only trial. Values:
  - window 12, anchor 25, `session_idx` 1, forward 0.71, reverse 0.47, as Run 4
    (`RUN4_LAUNCH_CHECKLIST.md:313`);
  - **`skip_min 0`, `skip_max 16`** in place of Run 4's 6/99. RANGE-MINER's constant phases sieve `[0,16]`
    whatever the trial says (`miner/range_miner_worker.py:1293`; F-1, ALREADY RECORDED), and PWC keeps the trial's
    range (ruling `:51`). Pinning the trial to `[0,16]` makes both engines sieve the same range today.
- **Interval:** `[2^31, 2^31 + 2^28)`, one eighth of the upper half (PLAN.md F4 `:354`).
- **Order:**
  1. RANGE-MINER first;
  2. its outputs copied out (§4c);
  3. then PWC, by its flags.

  Never both at once, and the second run starts only after the first has ended by itself.
- **Expected load:** an even spread is [UNVERIFIED]. On that assumption, from Run Q (`RUN4_LAUNCH_CHECKLIST.md:431-434`):
  - phases 1-2: 0;
  - phase 3: about 7-8;
  - phase 4: about 2,900.

  These counts are load, not hit rate.
- **Expected time [UNVERIFIED, estimates only]:**
  - RANGE-MINER at stripe `2^26` makes 4 stripes per phase, so only 4 workers are busy. Order of 10-15 minutes, from
    Run P's 15 minutes for 32 stripes per phase on 25 workers.
  - PWC at 25,000 seeds per chunk makes about 10,700 chunks per pass. Its time is not measured: watch the first
    minutes and stop if it is far off. Stopping means asking the owner; never SIGTERM the coordinator.
- **Memory [UNVERIFIED]:** about 2,900 phase-4 records are small against VM101's 15.5 GiB. The record gives no
  coordinator peak-RSS figure (`RUN4_LAUNCH_CHECKLIST.md:441-445`). Watch `free -m` on VM101 only; that is not a GPU read.
- **Then:** `scripts/compare_engine_outputs.py <miner_bundle.json> <pwc_bundle.json>`.
  - Exit 0: the same, outside the allowed fields.
  - Exit 1: they differ, each difference localized.
  - Exit 2: not comparable.
- This run also gives Beta's R1-Q3 evidence (phase 3-4 chunk times, completion, the 600 s timeout against the
  observed tail; PLAN.md `:371-372`), for one rig only.

### 4b. Gaps that block the session as specified (found while planning; all NEW)

1. **NOT FOUND: a one-rig RANGE-MINER launch path.**
   - `scripts/launch_fleet_manual.sh` brings up every node of the profile; it has no subset option (`:252-259`).
   - `gate12_launch.sh` and `/home/michael/run4_launch.sh` launch on three rigs. The checklist gate is "3/3 rigs
     at 8/8. Never launch on fewer GPUs" (`docs/drafts/RUN4_FIRST_RUN_CHECKLIST_20261118.md:30`).
   - The gate12 parity gate has no partial-fleet option (`scripts/gate12_parity_gate.py:698-710`).

   Surfaces searched: `scripts/`, the root `*.sh`, the two Run 4 checklists, and `docs/` for `execution-set-nodes`.
2. **NOT FOUND: a way to declare a one-rig set through WATCHER.**
   - `window_optimizer.py` has `--execution-set-nodes` and `--rig-profile` (`:1614-1634`). PWC honours the frozen set
     (`persistent_worker_coordinator.py:920-929`). But the manifest's `args_map` has neither key
     (`agent_manifests/window_optimizer.json:24-66`), so a WATCHER-launched run always resolves the full declared
     fleet.
   - Adding them touches a hand-landed file (`agent_manifests/`, `scripts/owner_land.py:85-87`). It may also touch
     WATCHER's literal key list (`agents/watcher_agent.py:506-521`, PLAN.md `:383`). That means Beta review and
     the owner landing by hand.
3. **Per-phase survivor sets are not persisted by either engine.**
   - The pipeline keeps forward/reverse counts only (`window_optimizer_integration_final.py:1256-1258`, `:2301-2302`).
     `forward_survivors.json`/`reverse_survivors.json` are count-only summaries (`:3484-3493`).
   - The per-phase records exist only in memory, in each engine's result dict at the integration: PWC `:2043-2074`,
     RANGE-MINER `:2017` via the phase-5 sink.
   - So the bundle's `phases` section (`scripts/compare_engine_outputs.py:27-28`) cannot be filled from today's
     outputs.
   - **And the comparator reports a phase absent from both bundles as EQUAL** (`compare_engine_outputs.py:131-135`).
     An extractor that leaves `phases` empty would make that row read "equal" when it compared nothing. Before use,
     that row must say "NOT COMPARED". That is a small change to a PWC-owned, CPU-only script, outside the
     before-commit review classes.
4. **Search records and credit.**
   - A Step 1 run takes its start from the open search's first gap (`RUN4_LAUNCH_CHECKLIST.md:314-315`).
   - Only one search may be open per family and mode set (`utils/step1_search_ledger.py:641`, `:689-691`).
   - A fully credited search ends with a `completed` event (`:66-67`, `:591`).
   - So the RANGE-MINER run and the PWC run cannot share one search. The PWC run needs a second search over the same
     interval, opened as an explicit repeat (`repeat_of = <the miner search>`, `:639-640`, `:672-685`), after the
     first completes.
   - Opening searches is the owner's act (`RUN4_LAUNCH_CHECKLIST.md:304`). It writes the live `prng_analysis.db`.
   - The interval was already credited by Run 4 attempt 3 and Run Q, in other searches. Whether that matters to the
     new searches is [UNVERIFIED]: `credit_decision` was not read for cross-search overlap.
5. **NPZ location.** Both runs publish through the run finalizer, which owns the root alias symlinks
   (`window_optimizer_integration_final.py:395-433`). The second run moves the aliases. Whether the first run's
   generation stays readable afterwards is [UNVERIFIED]. So the plan copies each run's outputs out before the next
   run starts.
6. **Miner staging.** The manifest's `staging_dir` is the live `/home/michael/miner_staging`
   (`agent_manifests/window_optimizer.json:276`). A Stage 4 RANGE-MINER run writes there, and its ledger is
   off-limits except what the owner approves each time (addendum `:39`). Pointing Stage 4 at a separate staging
   directory is possible by parameter (`staging_dir` is in `args_map`, `:57`). Whether the miner then starts a fresh
   ledger there cleanly is [UNVERIFIED].

### 4c. Extractors still to be written (CPU only, reading finished outputs; nothing on a rig)

Each extractor writes one `engine_output_bundle_v1` JSON (`scripts/compare_engine_outputs.py:24-41`).

| Bundle section | RANGE-MINER extractor reads | PWC extractor reads | Status |
|---|---|---|---|
| `inputs` | the search record's identity (`--step1-search-status`), the run's logged EXEC CMD and the frozen execution set and dataset lines (`window_optimizer.py:1855`, P0.5 lines) | the same | to write; needs both runs' logs kept |
| `npz` + the 22 arrays | the run's published NPZ generation, copied out right after the run (gap 5) | the same | to write; generation selection [UNVERIFIED] |
| `bidirectional.constant/variable` | from that NPZ: `seeds` split by `skip_mode` (`utils/canonical_arrays.py:98-121`) | the same | to write; possible today |
| `phase_counts` | the trial's forward/reverse counts from the coordinator log or trial result; per-stripe counts need the staging ledger (owner approval, read with `?mode=ro&immutable=1`) | `pwc_completion.passes` (`persistent_worker_coordinator.py:2656-2665`), which must be captured: it is in the result dict, and whether it is written to disk is [UNVERIFIED] | to write; capture point open |
| `credit` | `credited_intervals` for the miner's search (`utils/step1_search_ledger.py:557`) | the same, for the repeat search | to write; possible today |
| `phases.1-4` (seeds + match rates) | not persisted (gap 3) | not persisted (gap 3) | **needs a decision**: a capture hook (an integration change, so Beta review) or "not compared" |

Plus one change to `scripts/compare_engine_outputs.py`: a phase absent from both bundles reads "NOT COMPARED", not
EQUAL (gap 3).

---

## 5. Standing rules that apply to this work

| Rule | Source |
|---|---|
| Tell the owner before any power-up or GPU run. Bring-up needs the owner's explicit go for the bring-up itself, after the GPU PSUs are said to be on; choosing a run option is not a go | `docs/RIG_POWER_CONTROL_TUYA_REFERENCE.md:17`, `:70`, `:73`; owner instruction 2026-11-21 ("only Alpha powers rigs, after telling the owner") |
| Never launch a fleet or pipeline run autonomously; the owner initiates; always `nohup`, never `tmux` | CLAUDE.md §1 rule 3; `~/CLAUDE.md` hard rule 2 |
| No `rocm-smi` or other GPU monitoring read while cards work | owner instruction 2026-11-21; built into the SER8 MCP tool (`docs/MCP_TOOLS_REGISTER_20261119.md:37`). NOT FOUND as a repo rule with wider scope (searched `docs/`, `CLAUDE.md`, `AGENTS.md`). The plan reads GPUs only before work starts: `rocminfo` at bring-up (§2 step 1), and PWC's own CuPy count probe before its workers start |
| Never SIGTERM or kill a live coordinator; let it end itself | `docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md:38` (cause: Run I, `:27`) |
| The live ledger `/home/michael/miner_staging/` is off-limits except what the owner approves each time; read only with `?mode=ro&immutable=1` | addendum `:39`; `docs/exchange/RELAY_PROTOCOL.md:79` |
| `rigpower.py`: only `rig-6600`, `rig-6600b`, `rig-6600c`, never Zeus or the strip; `off` only after a clean shutdown and when the owner asked; `cycle` only on a rig confirmed frozen; tell the owner which rig and when | `docs/RIG_POWER_CONTROL_TUYA_REFERENCE.md:23-29`, `:61-75` |
| After `on`: `boot-proxmox`, re-arm netconsole, stop `rig-power-telemetry.service`, `pct start 100`; never reboot a host to fix a CT (a reboot returns Ubuntu) | `RIG_POWER_CONTROL_TUYA_REFERENCE.md:74`; `RUN4_LAUNCH_CHECKLIST.md:85-87` |
| Deploy: committed bytes (`git archive`), `mkdir -p` before `scp`, absolute paths, verify by digest, never by `git rev-parse`; no rig environment change | CLAUDE.md §2; `docs/PHASE6_PREREQS.md:126-131`; `RUN4_LAUNCH_CHECKLIST.md:197-199` |
| Parity from a clean committed tree; both gates refuse on dirty governed files | `scripts/pwc_parity_gate.py:49`; `RUN4_LAUNCH_CHECKLIST.md:166` |
| No pattern-matched `pkill`/`pgrep` against our own command text; bracket patterns or recorded PIDs | `~/CLAUDE.md` hard rule 8 |
| Output under the project's `data/`, never `/tmp` | `~/CLAUDE.md` hard rule 7 |
| Bound memory; state the expected peak before a long run | `~/CLAUDE.md` hard rule 9 |
| No commits or pushes by an agent; the owner lands (hand-landed paths: `agent_manifests/`, `miner/`, `agents/watcher_agent.py`, `gate12_launch.sh`, the parity gate file) | CLAUDE.md §1 rule 1; `scripts/owner_land.py:85-87` |
| Yardstick: equal output means PWC is a faithful backup on this input. It says nothing about hit rate, which is judged separately against `k/1000` on unseen draws | CLAUDE.md "The yardstick"; PLAN.md `:340-341` |
| Before-commit Beta review for coordinator, worker, kernel, verifier and launch-path code | CLAUDE.md "Beta reviews" |

---

## 6. Open questions for the owner (each with a recommendation)

1. **May Alpha bring up all three rigs once, only to deploy the four PWC files and run both parity gates (no GPU
   work)?**
   *Recommendation: yes, all three in one bring-up.* "Workers updated before Stage 4" then holds for the whole fleet.
   Both gates can read a full PASS, and gate12 has no one-rig mode. Shut down cleanly after, or leave the rigs up, as
   you say.
2. **Who runs the copy: you, or Alpha "at your word"?** The record allows either (`RUN4_LAUNCH_CHECKLIST.md:160`).
   *Recommendation: Alpha runs §2 steps 1-7 and shows you each gate's output; you decide the next step.*
3. **May the deploy also clear the top-level `__pycache__` on each CT100?** This is new; the record clears only
   `miner/` and `utils/`. *Recommendation: yes.* It is regenerable and removes any doubt about stale bytecode for
   `sieve_filter.py`.
4. **One rig, or all three, for the first comparison?** A one-rig session needs launch-path work that does not exist
   (§4b gaps 1-2: a subset option in `scripts/launch_fleet_manual.sh`, and `execution-set-nodes`/`rig-profile` in the
   manifest). That is code in the launch path, so Beta review, and the manifest is yours to land by hand.
   *Recommendation: run the first comparison on all three rigs over the same small interval `[2^31, 2^31 + 2^28)`.*
   It uses the recorded launch path and gates unchanged, is short, and compares the engines just as well. Keep
   one-rig as a later item if you want it for card-risk reasons.
5. **If one rig: which one?**
   *Recommendation: rig-6600 (`.122`).*
   - rig-6600b had the Run N card drop on 2026-11-19, then a hung reboot that needed you at the rig
     (`docs/TFM_SESSION_HANDOFF_20261119.md:16`, `:55-57`).
   - rig-6600c carries most of the recorded fault bursts: 20 of the 23 that began with `0x00801231` were on its
     16:00.0 (`:50`).
   - rig-6600 had two one-off events on 2026-11-18, including Run I (addendum `:26`, `:30`). It runs on Gen2 links
     (`:63`), it is the fastest rig (12-15 % faster on identical work, Alpha's memory notes [UNVERIFIED against a
     repo record]), and it is the only one with a git record.
   - Caveat: rig-6600 does not always start on `on`; one `cycle` after about two minutes (`RIG_POWER_CONTROL_TUYA_REFERENCE.md:79`).
6. **Per-phase survivor sets: capture them, or compare only what the pipeline keeps?** Keeping them needs an opt-in
   dump at the integration, which is Beta-review code. Without it, the comparison covers:
   - all 22 NPZ arrays;
   - both bidirectional sets with each seed's forward and reverse match rates;
   - the per-phase counts;
   - the credit.

   *Recommendation: compare what is kept for the first session* (it is what later steps see), with the comparator
   fixed to say "NOT COMPARED" for phases. Add the capture only if the first comparison shows a difference that needs
   localizing.
7. **Search records:** you would open one search for the RANGE-MINER run and, after it completes, a repeat
   (`repeat_of`) for the PWC run, both over `[2^31, 2^31 + 2^28)`, reason `operator`.
   *Recommendation: yes, that shape.* Alpha would first read how `credit_decision` treats an interval already
   credited by other searches and report before you open anything.
8. **Miner staging for Stage 4: the live `/home/michael/miner_staging`, or a separate directory?**
   *Recommendation: a separate directory under `/home/michael/`*, so the live ledger is not touched and the
   miner's own per-stripe counts can be read for the extractor without a per-read approval. Alpha would first check,
   CPU only, that the miner starts cleanly on an empty staging directory.
9. **RANGE-MINER stripe size for the small interval:** at the certified `2^26`, the interval makes 4 stripes per
   phase, so only 4 of the workers do work. *Recommendation: keep `2^26`.* Geometry may differ between engines
   anyway (F3), the certified shape stays, and the run is still short.
10. **The trial pins:** Run 4's window 12, anchor 25, `session_idx` 1, thresholds 0.71/0.47, with skip `[0,16]`.
    *Recommendation: yes.* It is the production shape apart from the skip range, which must be `[0,16]` until the
    RANGE-MINER skip fix is decided.

Nothing here is a yes to power, deploy, run or land. Each needs your word.

## Not verified in this plan

- Every rig-side byte: the rigs are off and nothing was probed.
- Run times, memory peaks and per-interval survivor counts (estimates only).
- Whether the finalizer keeps a previous run's NPZ generation readable.
- Whether `pwc_completion` reaches disk.
- How credit treats an interval credited by another search.
- Whether the miner starts on an empty staging directory.
- That the warm-start trial is not altered by the TRSE narrowing (`--trse-context`). An enqueued trial fixes its
  parameters in Optuna; the interaction was not read.
