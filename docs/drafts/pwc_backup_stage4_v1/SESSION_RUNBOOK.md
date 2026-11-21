# PWC full backup, Stage 4: the first two-engine session (RUNBOOK, draft)

Written 2026-11-21 by Claude (Team Alpha) in `/home/michael/tfm_worktrees/pwc-backup`, at HEAD `6d75f71`. **Nothing
here has been run.** This session wrote code and this document only:
- CPU only, no SSH to a rig, no GPU;
- no launch, no live ledger, no `~/miner_staging`.

Every command below is for the owner, or for Alpha at the owner's word, at the step that says so.

The session: all three rigs plus VM101, one `java_lcg` trial over `[2^31, 2^31 + 2^28)`, RANGE-MINER first, then PWC by
the flag, never both at once, then `scripts/compare_engine_outputs.py`.

## Prior record

**Searches run** (`/bin/grep -rliE`, this worktree at `6d75f71`), pattern
`(separate|isolated|scratch|fresh) (run )?(tree|worktree|output root|accumulator)|output_root|NOT COMPARED|extract_(pwc|miner)_bundle|stage ?4 (session|runbook)|prior_row_count`:

| Scope | Files with hits |
|---|---|
| `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/*_AUDIT.md` | 0 |
| `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md` | 0 |
| `docs/CHAPTER_*` | 2. `CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1280` and `CHAPTER_8_PRNG_REGISTRY.md:1470` use "not compared" for rig source; neither is about this |
| `docs/exchange/` | 2 (read, below) |
| `docs/RUN4_LAUNCH_CHECKLIST.md`, `docs/drafts/RUN4_FIRST_RUN_CHECKLIST_20261118.md`, `docs/drafts/pwc_backup_stage3_v1/` | 0 |

The plan this runbook executes was searched in full the same day: `REDEPLOY_AND_SESSION_PLAN.md` (this folder), Prior
record. Live code was read at every anchor cited.

**Documents that already decide or record the subject:**

| Document | What it records | Label |
|---|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47` | Owner, 2026-11-20 15:55: "run each and verify the results are the same" | ALREADY RECORDED |
| `docs/drafts/pwc_backup_stage3_v1/PLAN.md` §F, F4 `:342-379` | The Stage 4 design: inputs equal, outputs compared, smallest session | ALREADY RECORDED (this runbook executes it) |
| `REDEPLOY_AND_SESSION_PLAN.md` §4, and its 2026-11-21 status update | The session shape and its six gaps; the redeploy, done | ALREADY RECORDED |
| `docs/exchange/from_beta/X-FINDINGS-PLAN_R1_BETA_REVIEW.md:86` | Beta: "Publication selects L2 winners and merges a prior generation before counting final rows" (`utils/run_finalizer.py:1613-1659`) | ALREADY RECORDED. It is the fact behind §1's separate run trees |
| `docs/exchange/from_beta/X-PWC-STAGE2_R2_BETA_REVIEW.md:37` | "The five-strategy GPU question remains `NEEDS EVIDENCE` for the owner's Stage 4 sessions" | ALREADY RECORDED. §8 says what this session can show |
| `docs/RUN4_LAUNCH_CHECKLIST.md` §11 (`:304-345`) | The search-open command; opening a search is the owner's act | ALREADY RECORDED |
| `/home/michael/run4_launch.sh` (outside the repo) | Run 4's launcher: `diff` against `gate12_launch.sh` today shows only the `--params` block differs (`:466-472`) | ALREADY RECORDED (precedent for §3's launcher copy) |
| `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md:24-44` (§A) | SQLite reads: `mode=ro` alone never; immutable only with no or empty `-wal`; a non-empty `-wal` is UNAVAILABLE unless a stable copy is read; no production helpers against an original | ALREADY RECORDED. Omitted from the first draft (Beta X-PWC-STAGE4 R1 Finding 1); `extract_pwc_bundle.py` now follows it |
| `…_BINDING_ADDENDUM_BETA_R1_20261113.md:63-88` (§C) | Report Run 4's actual inputs (the effective start after WATCHER, the seven pins) and counts (`prior_row_count`, L2 winners) | ALREADY RECORDED (Finding 1). §2's params and §1's parent-free rule apply it |
| `docs/exchange/from_beta/X-FINDINGS-PLAN_R1_BETA_REVIEW.md:69-73` | Beta's LIVE objection behind §A: `mode=ro` defeats the live-state boundary; a separate worktree does not redirect an absolute ledger target | ALREADY RECORDED (Finding 1). Why the miner staging and the sampler are handled in A4 |
| `docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md:70-85` | Run 4 attempt 1 earned no credit: the sampler, still connected at the coordinator's last close, left a non-empty `-wal`; the remedy is the stable scratch copy | ALREADY RECORDED (Finding 1). A further reason the sampler is not run here (A4) |
| `docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:153-164` | A fresh worktree lacked the dataset files and `optuna_studies/` (git-ignored); creating an empty `optuna_studies/` fixed it | ALREADY RECORDED (Finding 1). A1 now creates it, A2 copies the dataset files |
| `docs/exchange/from_beta/X-PWC-STAGE4_R1_BETA_REVIEW.md` | Beta's review of the Stage 4 tooling: six LIVE objections (Q2, Q3, Q5, Q6, Q7, Q8), Finding 1 | ALREADY RECORDED. This revision answers Q2, Q3 (PWC side), Q5, Q7, Q8 and Finding 1; Q6 and Q3's miner side are agent 2's |

**Labels for what this runbook raises:**
- The separate run trees (§1), the runbook steps, the stop conditions, and the comparator's `NOT COMPARED` and exit 3
  are **NEW**.
- The comparison is engine equality, an engineering check. It is not a hit-rate measurement, and no verdict here says
  anything about hit rate against k/1000.
- Nothing here **CHANGES A RULING**.

**Not searched:** the host, Proxmox, ser8 archives, the private repository, project files outside the repository, and
the rigs.

---

## 0. What changed since the plan, and what is ready

- **Rigs:** redeployed and up (plan, status update). PWC parity is PASS 39/39 and gate12 parity is PASS 30/30, digests
  unchanged; I read this back from the four evidence files. Owner, 11:34: "leave the rigs up". `__pycache__` was not
  cleared.
- **Comparator, `scripts/compare_engine_outputs.py`:** fixed.
  - A field absent or null in both bundles, or null in one, now reads `NOT COMPARED`, never `EQUAL`.
  - The verdict is then `SAME WHERE COMPARED`, exit 3, never `SAME`.
  - Tests: `tests/test_compare_engine_outputs.py`, 30/30. That includes a code mutant that restores the old behaviour;
    all 12 dependent gates fail under it.
- **PWC extractor, `scripts/extract_pwc_bundle.py`:** new. Tests: `tests/test_extract_pwc_bundle.py`, 17/17, on a
  fixture built with the real `finalize_run` and the real ledger DDL.
- **Miner extractor, `scripts/extract_miner_bundle.py`:** agent 2's (read 2026-11-21). It follows the comparator's
  field list (§7).
- **Revised after Beta X-PWC-STAGE4 R1** (`docs/exchange/from_beta/X-PWC-STAGE4_R1_BETA_REVIEW.md`):
  - the comparator requires the F2 job identity in both bundles and, by default, granted credit with matching
    certified, linked and swept intervals;
  - the PWC extractor reads a non-empty `-wal` from a stable copy, binds coverage and link by their cross-identifiers,
    validates pass accounting and threshold records, and requires the link;
  - Phase A creates `optuna_studies/`, the halt flag is honoured and never cleared, and the derived launcher validates
    its params before any step and does not start the sampler.

  Test counts are in §13.

---

## 1. The design choice this session depends on: one fresh run tree per engine

**Fact 1.** The finalizer publishes into the process's working directory, `output_root=Path.cwd()`
(`window_optimizer_integration_final.py:3578`). It then merges the prior `current` generation into the new one,
`_l3_merge(winner_arrays, prior.arrays)` (`utils/run_finalizer.py:1620-1640`).

So, in the main tree:
- **Run 1's generation would hold every row the live accumulator already holds.**
- **Run 2's generation would hold run 1's rows as well as its own.**
- A PWC NPZ that missed rows RANGE-MINER found would still carry those rows from the parent and compare EQUAL.

The NPZ comparison would be unsound. Both test runs (skip `[0,16]`) would also enter the production generation that
Steps 2-6 read.

**Fact 2.** The Step 1 search ledger and the coverage history live in `prng_analysis.db`, relative to the working
directory (`database_system.py:46`). Optuna storage is relative to the module's own directory
(`window_optimizer_bayesian.py:23`). The miner ledger is `<staging_dir>/miner_ledger.db`
(`miner/range_miner_coordinator.py:12371-12372`, `:12537`).

**Therefore:** each engine runs from its own fresh `git worktree` of the same commit, with its own staging directory:
- `/home/michael/stage4_run_miner`, staging at `/home/michael/stage4_miner_staging`;
- `/home/michael/stage4_run_pwc`.

In each tree the first generation has no parent. `extract_pwc_bundle.py` refuses a generation that has one. Each tree
has its own `prng_analysis.db`, Optuna studies and `results/`.

**The live main tree, its `prng_analysis.db`, its `.s172_accumulator/`, and `~/miner_staging` are not touched.** The
rigs are unaffected: their `script_path` is the CT100 tree, which the redeploy set (`distributed_config.json`).

**The alternative** is both runs in the main tree. It needs no new trees, but:
- the NPZ comparison cannot be made (the second generation's parent is the first);
- two test generations enter production;
- two searches go into the live database, the second as a `repeat_of` repeat.

I do not recommend it.

---

## 2. Fixed values

```
MAIN=/home/michael/distributed_prng_analysis      # live main tree; read-only in this session
COMMIT=<main HEAD at session time>                 # 6d75f71, or the commit that lands this Stage 4 code (§9 Q1)
TM=/home/michael/stage4_run_miner                  # RANGE-MINER run tree (new)
TP=/home/michael/stage4_run_pwc                    # PWC run tree (new)
MSTAGE=/home/michael/stage4_miner_staging          # RANGE-MINER staging + its miner_ledger.db (new, empty)
EV=$MAIN/logs/stage4_session                       # evidence copies + bundles (logs/ is git-ignored)
S=2147483648; E=2415919104                         # [2^31, 2^31 + 2^28), one chunk of 268435456
```

- **Why `logs/` and not `data/`:** in the main tree `data/stage4/x.npz` and `x.txt` are NOT git-ignored. Checked with
  `git check-ignore`: only `*.json`, `*.db` and `*.log` are. An NPZ copy under `data/` would dirty the main tree, and
  that tree's next clean-tree gate would refuse. `logs/` is ignored as a whole (`.gitignore:85`).
- **Why run-tree files go under `$T/logs/stage4/`:** the same reason. A run tree must stay clean for the finalizer
  (`repository_tree_clean`, `utils/run_finalizer.py:1589-1596`).

**The trial (both runs, identical):** Run 4's pins, except skip `[0,16]` (plan §4a):
- `warm_start_window` 12, `warm_start_offset` 25, `warm_start_session_idx` 1;
- `warm_start_fwd_thresh` 0.71, `warm_start_rev_thresh` 0.47;
- `warm_start_skip_min` 0, `warm_start_skip_max` 16;
- `window_trials` 1, `n_parallel` 1, `test_both_modes` true, `prng_type` `java_lcg`.

With all pins set, that trial is trial 0 (`window_optimizer_bayesian.py:779-799`).

**RANGE-MINER params** (Run 4's JSON, `RUN4_LAUNCH_CHECKLIST.md:313`, with the skip pins, `max_seeds` and
`staging_dir` changed):
```
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 0, "warm_start_skip_max": 16, "staging_dir": "/home/michael/stage4_miner_staging"}
```

**PWC params:** the same job. The engine keys change:
- `use_persistent_workers` true and `use_range_miner` false (manifest `:217`);
- `pwc_coordinator_address`, with no default; PWC refuses without it (`window_optimizer.py:1580-1584`);
- `worker_pool_size` 8 and `seed_cap_amd` 25000, inside `PWC_STABLE_ENVELOPE` (`persistent_worker_coordinator.py:542-546`).
```
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "worker_pool_size": 8, "seed_cap_amd": 25000, "test_both_modes": true, "use_range_miner": false, "use_persistent_workers": true, "pwc_transport": "tcp", "pwc_coordinator_address": "192.168.3.177", "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 0, "warm_start_skip_max": 16}
```

- `seed_start: 0` is correct. WATCHER replaces it with the search's first gap, and `max_seeds` with the search's chunk
  (`agents/watcher_agent.py:2304-2308`).
- **VM101 is DHCP** (CLAUDE.md §3). Confirm `192.168.3.177` with `ip -4 addr` at step A0 before writing the PWC file.
- The comparator's `build_inputs` moves every engine key to `geometry` (`ENGINE_PARAM_KEYS`). The two files then yield
  equal `inputs` (gate `G-BUILD-INPUTS`).

---

## 3. Phase A: preparation (no GPU, no worker; owner approves Phase A as a whole)

Run on VM101 as `michael`, with `source ~/venvs/torch/bin/activate`.

**A0. Nothing is running, and the live state is recorded.**
```bash
ip -4 addr show | grep -E 'inet 192\.168\.3\.'                        # must show 192.168.3.177
pgrep -af "[w]atcher_agent|[w]indow_optimizer|[r]ange_miner_worker|[p]wc_worker_service|[s]ieve_gpu_worker"   # must print nothing
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  ssh -n -o BatchMode=yes michael@$ip 'hostname; pgrep -af "[r]ange_miner_worker|[p]wc_worker_service|[s]ieve_gpu_worker" || echo "  no workers"'
done
cd $MAIN && git rev-parse HEAD && test -z "$(git status --porcelain)" && echo "MAIN CLEAN"
HALT=$(cd $MAIN && python3 -c "from agents.safety.kill_switch import KillSwitch; print(KillSwitch().halt_file_path)")
test ! -e "$HALT" && echo "NO HALT FLAG ($HALT)"     # if present: someone's halt. Stop and ask; never clear it
mkdir -p $EV && { stat -c '%n %s %Y' $MAIN/prng_analysis.db; readlink $MAIN/.s172_accumulator/current; } > $EV/live_state_before.txt
```
- The `ssh` reads process lists only, not GPUs.
- `live_state_before.txt` holds the metadata the closing check (E2) compares against. It is not a ledger read.
- **The halt flag is shared by every checkout on VM101** (`agents/safety/kill_switch.py:60`, `/tmp/agent_halt`), and
  `--clear-halt` removes it whoever set it (`:336-339`, `agents/watcher_agent.py:3939-3942`; `docs/CHAPTER_12_AUDIT.md:61`).
  So no Stage 4 command passes `--clear-halt`. The derived launcher stops if the flag is present (A4), and WATCHER
  honours it: `run_pipeline` refuses to start while it exists (`agents/watcher_agent.py:3069-3073`). Repeat the check at B1 and C1.

**A1. Two run trees and an empty staging directory.** This writes `$MAIN/.git/worktrees/` metadata, which needs the
owner's approval.
```bash
cd $MAIN
git worktree add --detach $TM $COMMIT
git worktree add --detach $TP $COMMIT
mkdir -p $TM/optuna_studies $TP/optuna_studies        # git-ignored (.gitignore:137), absent from a fresh tree
mkdir -p $MSTAGE && test -z "$(ls -A $MSTAGE)" && echo "MSTAGE EMPTY"
```
- **`optuna_studies/`:** Step 1 storage is module-relative (`window_optimizer_bayesian.py:19-23`), so each run tree
  needs its own. Without it, study creation fails with `unable to open database file` (Beta X-PWC-STAGE4 R1 Q7;
  `docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md:153-164`). The directory is ignored, so the tree stays clean.
- **`$MSTAGE` must stay empty until Run 1.** The launcher refuses a non-empty one. The coordinator creates the fresh
  `miner_ledger.db` there (`miner/range_miner_coordinator.py:12371-12374`); `tests/test_stage4_miner_launch.py`
  L-EMPTY-STAGING checks that constructor on an empty directory.

**A2. Copy the git-ignored run inputs into each tree.** These are byte-identical copies with `cp -p`.
- the dataset pointer `daily3_current.json` (`miner/dataset_authority.py:68`);
- the version file it names;
- the legacy alias `daily3.json` (`:72`);
- `dataset_provisioning.json` (`:100`, `:959-961`);
- `trse_context.json` (manifest default `trse_context`).
```bash
VER=$(python3 -c "import json;print(json.load(open('$MAIN/daily3_current.json'))['filename'])")
for T in $TM $TP; do
  for f in daily3_current.json "$VER" daily3.json dataset_provisioning.json trse_context.json; do
    cp -p "$MAIN/$f" "$T/$f"
  done
  diff <(cd $MAIN && sha256sum daily3_current.json "$VER" daily3.json dataset_provisioning.json trse_context.json) \
       <(cd $T && sha256sum daily3_current.json "$VER" daily3.json dataset_provisioning.json trse_context.json) && echo "$T inputs identical"
  (cd $T && test -z "$(git status --porcelain)" && echo "$T CLEAN")
done
```
- The `filename` key is the pointer field read by `miner/dataset_authority.py:271-309`.
- **[UNVERIFIED]** whether a Step 1 run reads any other ignored file from the working directory. A missing one shows up
  in A5 or at the run's pre-dispatch checks, which refuse before any GPU work.

**A3. The params files, inside each tree's ignored `logs/`.**
```bash
mkdir -p $TM/logs/stage4 $TP/logs/stage4
cat > $TM/logs/stage4/params.json <<'EOF'
<the RANGE-MINER params line from §2>
EOF
cat > $TP/logs/stage4/params.json <<'EOF'
<the PWC params line from §2>
EOF
python3 -c "import json;[json.load(open(p)) for p in ('$TM/logs/stage4/params.json','$TP/logs/stage4/params.json')];print('params parse')"
```

**A4. The RANGE-MINER launcher.** It is `gate12_launch.sh` with the reviewed patch
`docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff` (sha256 `4c2d41c85f88b00f034167a9085b166c1dc1e17bf7e9d899c642cc12daa8bc50`). It is kept outside the repository, as
Run 4's was (`/home/michael/run4_launch.sh`). Revised after Beta X-PWC-STAGE4 R1 Q8. The patch:
1. **`:79`:** `cd ~/distributed_prng_analysis` becomes `cd /home/michael/stage4_run_miner`. This is the only main-tree
   path in the script.
2. **A new params block, before every gate** (markers `STAGE4-PARAMS BEGIN/END`, right after the venv `source`). It
   loads `logs/stage4/params.json` and validates it, and stops the launcher on any failure. It checks:
   - the file is a JSON object and a RANGE-MINER run (`use_range_miner` true, `use_persistent_workers` false);
   - every pin and job key is present;
   - `staging_dir` is absolute, is not, and does not overlap, the manifest's live `staging_dir`
     (`agent_manifests/window_optimizer.json:276`; the path is read from the manifest, not written in);
   - `staging_dir` exists and is empty;
   - the shared halt flag is absent. It is honoured, never cleared.

   On success it holds the validated JSON in `$STAGE4_PARAMS`. An invalid file can no longer become an empty
   `--params`, which WATCHER skips (`agents/watcher_agent.py:4173`), falling back to manifest defaults including the
   live staging.
3. **The concurrency sampler is not run** (its section `:442-462` and the supervisor `:477-485` are replaced). Why:
   - its unchanged threshold of 25 cannot be reached, because `2^28` seeds at `2^26` per macro-stripe is 4 stripes per
     phase, so at most 4 workers hold a claimed stripe;
   - it resolves its ledger from the manifest, which is the live staging (`scripts/gate12_concurrency_sampler.py:167-188`;
     L-SAMPLER-PATH shows `/home/michael/miner_staging/miner_ledger.db`), and opens it with plain `mode=ro` (`:203`);
   - it refuses a ledger without a `stripes` table (`:246-252`), so it cannot start against an empty Stage 4 staging
     before WATCHER;
   - a connected sampler is what left Run 4 attempt 1's `-wal` non-empty and cost its credit
     (`docs/CLAUDE_CODE_REPORT_STEP1_CREDIT_WAL.md:70-85`).

   `SAMPLER=""` keeps the later `kill -TERM "$SAMPLER"` lines valid under `set -u`.
   **Beta's position** (R1 Q8, `:71-73`) was to keep the sampler, pointed at the staged path, with the schema created
   first. The Alpha lead allowed dropping it with this reason. If the owner wants the sampler, that is the alternative,
   and its output still proves neither saturation nor Gate 12 acceptance at this geometry.
4. **WATCHER (`:466-472`)** gets `--params "$STAGE4_PARAMS"` and no `--clear-halt`.
5. **`:523`:** `tail -f "$LOG"` is removed, so the launcher exits under `nohup`.

```bash
cp $TM/gate12_launch.sh /home/michael/stage4_launch_base.sh
patch -o /home/michael/stage4_miner_launch.sh /home/michael/stage4_launch_base.sh \
  <tree holding the diff>/docs/drafts/pwc_backup_stage4_v1/stage4_miner_launch.diff   # $TM once landed, else this worktree
sha256sum /home/michael/stage4_miner_launch.sh    # must be fe964b04fe3ea431986835ff2fff934349864751f8974da5ccebb6b1a62bf5d3
chmod +x /home/michael/stage4_miner_launch.sh && bash -n /home/michael/stage4_miner_launch.sh && echo "syntax ok"
```
- `tests/test_stage4_miner_launch.py` applies the same diff to `gate12_launch.sh` at HEAD. It checks that the params
  block comes before the cleantree, GPU and parity gates, the clean slate and the fleet launch, and that no command
  starts the sampler or passes `--clear-halt`. It also runs the params block against ten fixtures, including the live
  staging, a non-empty staging and a present halt flag.

Everything else is unchanged, and is already the RANGE-MINER launch path of Runs 4, P and Q:
- the clean-tree admission gate (`:133`);
- the GPU fail-close gate (`:164`);
- gate12 parity (`:192`);
- the clean slate (`:203`, bracket `pkill`);
- the fleet launch parked at the release barrier (`:315-317`);
- the sentinel, liveness and cache-join gates;
- the `nohup` WATCHER;
- the release (`:492-500`).

**The GPU fail-close gate runs `rocm-smi` once on each rig** (`preflight_check.py:118-124`). It runs before the fleet
is launched, so no card is working. This is the only `rocm-smi` in the session, and it needs the owner's approval
(§10 item 5).

**A5. CPU-only dry checks in each tree.** No worker starts and no GPU is used.
```bash
cd $TM && python3 -u scripts/gate12_cleantree_gate.py --phase admission 2>&1 | tee logs/stage4/cleantree_A5.log; echo "rc=${PIPESTATUS[0]}"
cd $TP && python3 -u scripts/gate12_cleantree_gate.py --phase admission 2>&1 | tee logs/stage4/cleantree_A5.log; echo "rc=${PIPESTATUS[0]}"
cd $TP && python3 -u scripts/pwc_parity_gate.py --transport tcp --evidence-json logs/stage4/pwc_parity_A5.json 2>&1 | tee logs/stage4/pwc_parity_A5.log; echo "rc=${PIPESTATUS[0]}"
```
- **Pass:** every `rc=0`, and PWC parity PASS 39/39. The parity gate compares the rigs to the run tree. The tree is
  `$COMMIT`, and the four rig files are unchanged by any Stage 4 code, so it should PASS as after the redeploy.

**A6. Open one search in each tree. This is the owner's act** (`RUN4_LAUNCH_CHECKLIST.md:304`). It writes only that
tree's new `prng_analysis.db`.
```bash
for T in $TM $TP; do
  cd $T
  P="$(cat logs/stage4/params.json)"
  python3 agents/watcher_agent.py --step1-search-open \
    --search-range-start 2147483648 --search-range-end 2415919104 \
    --search-reason "<owner's words: Stage 4 engine comparison, $(basename $T), [2^31, 2^31+2^28), skip [0,16], Run 4 pins>" \
    --params "$P" 2>&1 | tee logs/stage4/search_open.log
  python3 agents/watcher_agent.py --step1-search-status --params "$P" 2>&1 | tee logs/stage4/search_status.log
done
stat -c '%n %s %Y' $MAIN/prng_analysis.db     # must equal the A0 line: the live database was not written
```
- **Pass:** `STEP1_SEARCH_OPENED s1s-…`, then a status of `"state": "gap"` with `"next_seed_start": 2147483648`.
- **Fail:** `REFUSED` or `ERROR`. Read the reason; do not work around it.

---

## 4. Phase B: Run 1, RANGE-MINER (GPU)

**B0. Tell the owner that the fleet's cards are about to work, and wait for the go.**

**B1.** Repeat A0's process checks, which must show nothing. Check `ss -ltn | grep -E ':5600|:5700'`, which must also
show nothing. Repeat the halt-flag check: the flag must be absent, and is never cleared by us.

**B2. Launch:**
```bash
nohup bash /home/michael/stage4_miner_launch.sh > $TM/logs/stage4/launcher.out 2>&1 &
echo $! > $TM/logs/stage4/launcher.pid
```
- `launcher.out` names the WATCHER pid and its log: `watcher pid=<pid> -> logs/gate12_<STAMP>.log` (`:475`).
- An invalid params file, a live or non-empty staging, or a present halt flag makes the launcher print
  `STAGE 4 REFUSED: …` / `STAGE 4 ABORTED …` and exit 1 before any gate.
- A gate refusal makes the launcher print `GATE-12 ABORTED …` or `REFUSED` and exit 1 before any dispatch.
- In either case, **stop there.**
- The evidence file records `concurrency sampler: NOT RUN for Stage 4` and the validated params.

**B3. While it runs, read only:**
- the WATCHER log (`grep`, or `tail -n 40 <log>`; never `tail -f` piped);
- `free -m` on VM101.

No `rocm-smi`, `nvidia-smi` or other GPU read on any machine. No SSH to the rigs except a bracket `pgrep` if the owner
asks.

Expected load (plan §4a, [UNVERIFIED] estimates):
- 4 stripes per phase;
- phase 4 about 2,900 survivors;
- order of 10-15 minutes.

**B4. It has ended by itself** when `kill -0 <watcher pid>` fails. The log must then show, in order:
- `[STEP1-SEARCH S7] credited: trials [0] completed every required phase over [2147483648, 2415919104]`
  (`window_optimizer_integration_final.py:3552-3562`);
- `[S172 D3.5][GENERATION PUBLISHED]` with `Prior rows:     0`. This proves the fresh tree; a non-zero value means
  stop;
- `[S145 COVERAGE] certified [2,147,483,648, 2,415,919,104)`;
- `[STEP1-SEARCH LINK]`.

Then repeat A0's rig `pgrep`: no `range_miner_worker` may remain.

**B4a. Write the RANGE-MINER run receipt, immediately, before anything else.** This is step 1 of agent 2's two-step
extractor (`docs/drafts/pwc_backup_stage4_v1/EXTRACT_MINER_README.md` "Use: two steps", in `pwc-offset`; Beta
X-PWC-STAGE4 R1 Q6). Nothing durable ties a miner trial to the generation it fed, so the receipt captures the
binding now:
- the real paths and the params sha256;
- the miner `run_id` and trial times;
- a content sha256 of the trial's ledger rows;
- the generation, artifact and sidecar sha256, the coverage id and the link;
- the F2 identity.

It is CPU only and read-only on `$MSTAGE` and `$TM`.
```bash
cd <tree holding scripts/extract_miner_bundle.py>      # $MAIN once landed, else /home/michael/tfm_worktrees/pwc-offset
python3 -u scripts/extract_miner_bundle.py --staging-dir $MSTAGE --output-root $TM \
  --params-json $TM/logs/stage4/params.json --write-receipt $EV/miner_receipt.json \
  2>&1 | tee $EV/miner_receipt.log; echo "rc=${PIPESTATUS[0]}"
sha256sum $EV/miner_receipt.json | tee $EV/miner_receipt.sha256
```
- **Pass:** `rc=0` and `$EV/miner_receipt.json` exists.
- **Fail:** `rc=2`, with the reason on stderr and nothing written. **Stop: Run 2 does not start.**
- The receipt is outside `$MSTAGE` and `$TM`, as the tool requires, and is never overwritten.
- **Why it must be now:** its binding check B6 bounds the publication lag by the trial's duration. Only a receipt
  written before the PWC arm publishes is guaranteed to bind the right pair (README, "The limit, stated plainly").
- B6 is not measured on a real trial. If it refuses a genuine pair, the refusal names both numbers, and the lead
  decides.

**B5. Copy Run 1's outputs out, before anything else starts.**
```bash
R1=$EV/run1_miner; mkdir -p $R1
cd $TM && cp -a .s172_accumulator results optimal_window_config.json bidirectional_survivors.json logs $R1/ 2>/dev/null
cp -p $TM/prng_analysis.db* $R1/
mkdir -p $R1/staging && cp -p $MSTAGE/miner_ledger.db* $MSTAGE/threshold_provenance.json $R1/staging/ 2>/dev/null
(cd $R1 && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $EV/run1_miner.sha256
```
- `results` may not exist for a RANGE-MINER run; the `2>/dev/null` covers that.
- Staged shards stay in `$MSTAGE`. If agent 2's extractor needs more of the staging directory, its `--help` says so.

**B6. Extract the RANGE-MINER bundle (CPU)** with agent 2's tool:
```bash
cd <tree holding scripts/extract_miner_bundle.py>      # $MAIN once landed, else this worktree
python3 -u scripts/extract_miner_bundle.py --staging-dir $MSTAGE --output-root $TM \
  --params-json $TM/logs/stage4/params.json --out-dir $EV/bundle_miner \
  --receipt $EV/miner_receipt.json \
  2>&1 | tee $EV/extract_miner.log; echo "rc=${PIPESTATUS[0]}"
```
**Run 2 does not start until this writes `$EV/bundle_miner/bundle.json`.** If Run 1's outputs cannot be read, there is
nothing to compare against.

---

## 5. Phase C: Run 2, PWC (GPU)

**C0. Tell the owner, and wait for the go.**

**C1.** Run 1's WATCHER and launcher are gone (`kill -0` fails for both pids). Run A0's process checks (nothing),
`ss -ltn | grep -E ':5600|:5700'` (nothing), and the halt-flag check (absent). **Also check that B4a's receipt exists
and still has its recorded hash:** `sha256sum -c $EV/miner_receipt.sha256`. Without it, Run 2 does not start.

**C2. Pre-gates in the PWC tree.** Check the real exit code with `PIPESTATUS`, as `gate12_launch.sh:131-134` does.
```bash
cd $TP && source ~/venvs/torch/bin/activate
python3 -u scripts/gate12_cleantree_gate.py --phase admission 2>&1 | tee logs/stage4/cleantree_C2.log; echo "rc=${PIPESTATUS[0]}"
python3 -u scripts/pwc_parity_gate.py --transport tcp --evidence-json logs/stage4/pwc_parity_C2.json 2>&1 | tee logs/stage4/pwc_parity_C2.log; echo "rc=${PIPESTATUS[0]}"
```
Both must give `rc=0`. No `rocm-smi` gate here: PWC counts each rig's GPUs itself with CuPy, before any worker
starts (`persistent_worker_coordinator.py:968-1000`).

**C3. Launch.** The PWC coordinator starts and stops its own rig workers: stale kill, cache reset, then a `nohup`
launch per GPU (`:1409-1500`), and `pwc.shutdown()` at the end (`:2882`).
```bash
cd $TP
PWC_PARAMS=$(python3 -c "import json,sys; p=json.load(open('logs/stage4/params.json')); \
assert p.get('use_persistent_workers') is True and p.get('use_range_miner') is False, 'not a PWC run'; \
print(json.dumps(p, separators=(',',':'), sort_keys=True))") || { echo "PWC params invalid - stop"; false; }
test -n "$PWC_PARAMS" && echo "PWC params: $PWC_PARAMS"
nohup env PYTHONPATH=. python3 agents/watcher_agent.py --run-pipeline \
  --start-step 1 --end-step 1 --params "$PWC_PARAMS" \
  > logs/stage4/pwc_run.log 2>&1 &
echo $! > logs/stage4/watcher.pid
```
- The params are validated before the launch and passed as a value, never as an inline `$(cat …)` that could be empty.
- No `--clear-halt`: a present flag is someone's halt (A0).

**C4. While it runs:** the same read-only rule as B3. PWC time is not measured. It is about 10,700 chunks per pass at
25,000 seeds (plan §4a). Watch the log's chunk progress in the first minutes, and tell the owner if it is far off.

**C5. It has ended by itself** when `kill -0 $(cat logs/stage4/watcher.pid)` fails. The log must show:
- four `Results saved: results/window_opt_{forward,reverse,forward_hybrid,reverse_hybrid}_12_25_t0.json (<n> survivors)`
  lines (`persistent_worker_coordinator.py:2514`). A `Results summary saved` line means more than 100,000 survivors in a
  phase (`:2509-2532`): that phase will be `NOT COMPARED`;
- the same four lines as B4, with `Prior rows: 0`;
- no `PASS INCOMPLETE`, `TRIAL 0 INCOMPLETE` or `NO CREDIT`.

Then check the rigs: no `pwc_worker_service` may remain.

**C6. Copy out, then extract (CPU):**
```bash
R2=$EV/run2_pwc; mkdir -p $R2
cd $TP && cp -a .s172_accumulator results optimal_window_config.json bidirectional_survivors.json logs $R2/
cp -p $TP/prng_analysis.db* $R2/
(cd $R2 && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $EV/run2_pwc.sha256
cd <tree holding scripts/extract_pwc_bundle.py>       # $MAIN once landed, else /home/michael/tfm_worktrees/pwc-backup
python3 -u scripts/extract_pwc_bundle.py --run-dir $TP --params-json $TP/logs/stage4/params.json \
  --out-dir $EV/bundle_pwc 2>&1 | tee $EV/extract_pwc.log; echo "rc=${PIPESTATUS[0]}"
```
- **Pass:** `BUNDLE_WRITTEN …/bundle_pwc/bundle.json`, plus the check line `NPZ seeds == union of bidirectional sets`.
- The extractor reads `$TP` and never writes there. It opens the tree's ledger `?mode=ro&immutable=1`.
- It refuses (exit 2) on any of the conditions in its docstring.

---

## 6. Phase D: the comparison (CPU)

```bash
cd <tree holding scripts/compare_engine_outputs.py>
python3 -u scripts/compare_engine_outputs.py $EV/bundle_miner/bundle.json $EV/bundle_pwc/bundle.json \
  2>&1 | tee $EV/compare.txt; echo "rc=${PIPESTATUS[0]}"
python3 -u scripts/compare_engine_outputs.py --json $EV/bundle_miner/bundle.json $EV/bundle_pwc/bundle.json > $EV/compare.json
```

| rc | Verdict | Meaning |
|---|---|---|
| 0 | `SAME` | Every field compared, and all are equal outside the F3 allowed fields |
| 1 | `DIFFERENT` | At least one compared field differs. Each difference is localized; a `NOT COMPARED` field never counts as one |
| 2 | `NOT COMPARABLE` | Any of: an F2 job-identity input (`F2_REQUIRED_INPUTS`: the run interval, family, modes, commit, dataset digest, the seven pins, `prng_type`, `test_both_modes`) is missing, null or different in either bundle; no input was compared; either bundle lacks granted credit with certified, linked and every swept interval equal to the run interval (Beta R1 Q2); or a format error |
| 3 | `SAME WHERE COMPARED` | No difference in what was compared, but the listed fields were NOT COMPARED. **This is not "the same"** |

**Expected:** exit 3 at best, unless agent 2's extractor can fill `phases.1-4` for RANGE-MINER. The plan's gap 3 found
the miner keeps only per-phase counts on disk (`window_optimizer_integration_final.py:1256-1258`). PWC's bundle carries
all four seed lists here (each phase is under 100,000), but a null on either side is NOT COMPARED.

What is compared whatever happens:
- the 22-array NPZ, bit-exact;
- `bidirectional`, when the miner bundle carries it;
- `phase_counts.*.survivor_rows`;
- `credit`.

**A `DIFFERENT` result is a finding, not a failure to fix in the comparator.** Report it with the comparator's
localized rows, and follow lead handling (`~/.claude/skills/lead-handling/SKILL.md`) before calling any difference
noise.

---

## 7. The bundle schema both extractors follow (agreed by reading the comparator)

The authoritative text is the docstring of `scripts/compare_engine_outputs.py`, section "What each extractor writes".
Agent 2 must read the current version: it changed today. In short:
- **`inputs`:** `build_inputs(params, provenance, dataset_sha256)`, imported from the comparator. It returns
  `param.<key>` for every non-engine `--params` key, and `run.<key>` for `prng_base`, `skip_modes_executed`,
  `seed_start`, `seed_end_exclusive`, `repository_commit` and `dataset_sha256` (the `certified_coverage` row).
- **`phases.N`:** `{"seeds": [...], "match_rates": [...]}`. Seeds are ascending (stable sort). Each rate is an exact
  float32, through `exact_float32()`, which refuses rather than rounds. The value is `null` when the engine persists no
  seed list.
- **`bidirectional.constant` / `.variable`:** the engine's forward∩reverse sets, sorted, or `null`. They are not the NPZ
  split: the finalizer keeps one row per seed (`utils/run_finalizer.py:714-744`).
- **`phase_counts.N`:** `{"survivor_rows": int, "distinct_survivors": int or null}`.
- **`credit`:** `{"granted", "certified_interval", "linked_interval", "swept_intervals"}`. All four are expected keys.
  Granted means one `certified_coverage` row whose `generation_id` AND `artifact_sha256` are this generation's, with a
  `step1_chunk_link` row naming that `coverage_id` and the same interval. A row sharing only one identifier is refused,
  not attached (Beta R1 Q3).
- **F2 job identity:** the 15 `F2_REQUIRED_INPUTS` must be EQUAL in both bundles, or the verdict is `NOT COMPARABLE`.
- **`npz`:** the generation's `bidirectional_survivors_binary.npz`, copied next to `bundle.json` and hash-checked against
  `artifact_sha256`. The generation must have `prior_row_count == 0`.
- **The rule for absence:** a missing value is written as `null`. A key absent from one bundle while the other carries
  a value is DIFFERENT.

---

## 8. Credit, the single-open-search rule, and an interval already credited

I read `credit_decision` (`utils/step1_search_ledger.py:1044-1086`). Its inputs are this chunk's requested interval,
the required phases, the sweep evidence each trial recorded in this process, and the trials' Optuna states. **It reads
no ledger, no other search and no coverage history.** It credits when:
- at least one COMPLETE trial has evidence from this chunk;
- every required phase is finished;
- every phase's swept interval equals the requested one.

Zero survivors are credited (`:1052-1053`). **So an interval already credited elsewhere is neither refused nor treated
differently by `credit_decision`.**

The other two places an earlier credit could matter:
- **The search cursor is per search.** It is `first_uncovered_seed` over that search's own linked chunks
  (`utils/step1_search_ledger.py:563-568`). Another search's credit does not move it.
- **The coverage history merges overlaps.** A second certified row over the same interval is appended; the union
  merges overlap and adjacency (`utils/seed_coverage_ledger.py:538`), and the id is content-addressed and idempotent
  (`:846`). The history's union does not change.

**What this means for this session:**
- With one tree per engine, each tree has a fresh `prng_analysis.db` and **one** search. The one-open-search-per-selector
  rule (`utils/step1_search_ledger.py:690`) holds trivially, and no `repeat_of` is needed (`:685`).
- `[2^31, 2^31+2^28)` was credited in the live database by Run 4 attempt 3 and by Run Q. That is invisible to the trees
  and irrelevant to both credit decisions.
- Nothing is written to the live database.

**In the main-tree alternative:**
- the first search over this interval opens (the live upper-half search is completed, not open; [UNVERIFIED], the live
  database was not read);
- its cursor starts at 2^31 despite Run 4's credit;
- PWC needs a second search with `repeat_of` after the first completes.

**Beta's five-strategy question** (`X-PWC-STAGE2_R2_BETA_REVIEW.md:37`): this session runs all five variable-skip
strategies on both engines. PWC loads them from `hybrid_strategy.get_all_strategies()`
(`persistent_worker_coordinator.py:2774-2777`). Equal phase 3-4 outputs would be evidence on it. Whether it answers
the question as Beta put it is for the owner and Beta.

---

## 9. Stop conditions (any one: stop, report, do not work around)

1. Any gate or dry check refuses or exits non-zero. This covers A2-A6, the launcher's own gates, and C2.
2. The search open is `REFUSED`, or the status is not `gap` at `2147483648`.
3. `Prior rows` is not 0 at publication: the tree was not fresh.
4. `NO CREDIT`, `PASS INCOMPLETE`, `TRIAL 0 INCOMPLETE`, `GATE-12 ABORTED`, `COORDINATOR NEVER BOUND`, a sentinel,
   liveness or cache-join refusal, or the run ending with no `GENERATION PUBLISHED`.
5. A rig loses cards or stops answering. This includes a card drop like Run N's on `rrig6600b` gpu6
   (`docs/TFM_SESSION_HANDOFF_20261119.md:16`). Tell the owner; recovery is the owner's.
6. No new line in the WATCHER log for 20 minutes, or PWC chunk progress far off the expectation. Tell the owner.
   **Never SIGTERM a live coordinator** (`docs/TFM_SESSION_HANDOFF_20261118_ADDENDUM.md:36-40`).
7. VM101 `free -m` available below 2 GiB (15.5 GiB total, no swap; CLAUDE.md rule 9). Tell the owner.
8. Before Run 2: any worker left on any rig, port 5600 or 5700 still bound, no `miner_receipt.json` (B4a) with its
   recorded hash, or no `bundle_miner/bundle.json`.
9. Either extractor refuses. The refusal names the reason; the fix is to the outputs or the tool, never a bypass.
10. E2 shows the live `prng_analysis.db` or the `current` pointer changed.
11. The halt flag is present at A0, B1 or C1, or the launcher aborts on it. It is someone's halt: ask the owner, and
    never clear it.

---

## 10. What the owner approves before launch

1. **The separate-run-tree design (§1).** It means two `git worktree add` calls from the main tree, the copies in A2,
   and `$MSTAGE`.
2. **The commit the trees use (Q1 below).**
3. **The two params files (§2),** including VM101's address as confirmed in A0.
4. **The derived launcher `/home/michael/stage4_miner_launch.sh` (A4)**: the reviewed diff, its params gate, and
   **the dropped sampler**. Beta preferred keeping it with a staged path; see A4 item 3.
5. **The one `rocm-smi` per rig in the launcher's GPU gate**, before any card works (A4).
6. **Opening the two searches (A6)**, which is the owner's own act, in the trees' databases only.
7. **The go for Run 1 (B0), and separately the go for Run 2 (C0).**
8. **The stop conditions (§9).**
9. **Whether Beta reviews anything first.** Two things could be in scope:
   - The before-commit review classes include "the verifier and launch path for runs that count" (CLAUDE.md, Beta
     reviews). These runs do not count: they write no live coverage. But the two extractors and the comparator are
     verifier-like code.
   - `stage4_miner_launch.sh` is a launch path.

## 11. Afterwards

- **E1:** rigs stay up (owner, 11:34), unless the owner says otherwise. A power-off is clean shutdown first, then
  `rigpower.py off` on the three rig outlets only (`docs/RIG_POWER_CONTROL_TUYA_REFERENCE.md:65`, `:72`).
- **E2:** the live state is unchanged. Compare `stat -c '%n %s %Y' $MAIN/prng_analysis.db; readlink
  $MAIN/.s172_accumulator/current` with `$EV/live_state_before.txt`.
- **E3:** the run trees, `$MSTAGE` and `/home/michael/stage4_miner_launch.sh` stay in place as evidence. Removing them
  (`git worktree remove`) is the owner's call.

## 12. Open questions for the owner (recommendations)

1. **Which commit do the run trees use?** Recommend landing this Stage 4 code first, then building the trees at that
   commit. The code is the comparator fix, the PWC extractor and agent 2's miner extractor. It touches no rig file, so
   both parity gates read as now, and the extractors then run from a committed tree. Otherwise the trees use
   `6d75f71`, and the extractors run from the uncommitted worktrees.
2. **Is `SAME WHERE COMPARED` (exit 3) enough for this first session,** or must the miner's per-phase sets be captured
   first? Capturing them is a change to the miner sink or the integration, so Beta-reviewed code. Recommend: accept
   exit 3 for the first session. The NPZ, the bidirectional sets (if agent 2 can fill them), the counts and the credit
   are still compared bit-exactly, and the verdict names exactly what was not compared.
3. **Clear `__pycache__` on the CT100s first** (plan §6 Q3)? Still open. It is not needed for the runs, since
   `scp -p` kept source mtimes and parity is by source digest. Recommend: ask once, before Run 1.
4. **Keep the main-tree route as a fallback?** Recommend no, for the reasons in §1.

## Not verified here

- A real Run 1 or Run 2: none ran.
- Agent 2's extractor on real output. B4a and B6 follow its two-step CLI as its README describes it (revision
  `2f1d35d4…`, read 2026-11-21). The B6 publication-lag bound is unmeasured on a real trial.
- The full set of ignored files a run tree needs (A2): the dry checks and the pre-dispatch checks are the test.
- Run times and PWC throughput (plan §4a).
- The live database's search state (not read).
- That a real coordinator run starts cleanly in an empty `$MSTAGE`. The constructor does
  (`tests/test_stage4_miner_launch.py` L-EMPTY-STAGING), but no run has.
- The derived launcher end to end. Only its params block is executed in tests; the rest is the unchanged Run 4 path.

## 13. Test status (CPU, this worktree)

| Suite | Result |
|---|---|
| `tests/test_compare_engine_outputs.py` | 36/36 PASS (2026-11-21, after Beta R1) |
| `tests/test_extract_pwc_bundle.py` | 32/32 PASS (2026-11-21, after Beta R1; 15 new Beta regressions) |
| `tests/test_stage4_miner_launch.py` | 14/14 PASS (2026-11-21, new) |

Run one after another (they must not run concurrently). Agent 2's `tests/test_extract_miner_bundle.py` was not run here.
