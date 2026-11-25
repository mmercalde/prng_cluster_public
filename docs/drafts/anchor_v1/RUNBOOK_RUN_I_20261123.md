# RUNBOOK · Run I · second RANGE-MINER generation, sieve window just before the Step 1 holdout · 2026-11-23

**PREPARED, NOT LAUNCHED.** Nothing in this runbook has been run. No run tree, staging directory, launcher or
block file exists on disk (`ls` 2026-11-23: `~/skipsw_run_i`, `~/skipsw_staging_i`, `~/skipsw_blocks/i`,
`~/skipsw_launch_i.sh` all absent). No rig was contacted, no GPU was used, the main tree was not changed.
Task: `~/distributed_prng_analysis/logs/e2e_session/TASK_RUN_I_PREP.txt`.

## Plain summary

- **What Run I is:** run H again, identical except for one value. The window anchor goes from **25** to **6695**.
  - Run H sieved 12 midday draws from **2002-11-29 to 2002-12-10**.
  - Run I sieves the 12 midday draws from **2021-03-05 to 2021-03-16**. Those are the last 12 midday draws before
    the first holdout record, which is 2021-03-17 midday at combined index 14,454.
  - I re-derived this from the live dataset with the worker's own loader. Anchor 6696 would take in the first
    holdout draw.
- **Code:** main at `41ad509`, as it is. No unlanded code is used. Run H ran on `00e2ee7`; the only change since is
  one documentation file (`git diff --stat 00e2ee7 41ad509`).
- **Why an anchor of 6695 runs today:** the hand pin goes through Optuna, which only warns on the out-of-range
  value. The worker then checks the anchor against the real limit, [0, 8503].
- **How we will know the right window was sieved:** the miner ledger's `trial_context` row must show:
  - `window_anchor_val = 6695`;
  - `residue_sha256 = ffb9ffa029c401ce5d80896a4447fcd104e205c8e16a5bbe0a3b26de222bf05d`.

  The same digest formula reproduces run H's recorded value (`c761272d…`) from anchor 25, and every worker checks
  the digest before it sieves. Anchor 25, digest `c761272d…`, or any anchor other than 6695 (for example 100)
  means the run silently fell back or clamped.
- **Owner's word needed:** two things.
  1. The go itself.
  2. Probably a one-off past the 50,000 survivor cap. Run H's gate refused at phases 3 and 4. Run I's own gate
     verdict is unknown until block b09 runs it.
- **Expected duration:** about 8 minutes of blocks plus about 2 minutes of trial. This is taken from run H: trial
  130.8 s, staging-to-publish 59 s. Run I's survivor counts are not known in advance.
- **Unknowns:** 16 differences from run H could not be ruled out (§6). The largest:
  - rig state after the reboot into Proxmox;
  - the Optuna out-of-range path, which run H did not use;
  - Run I's survivor volume;
  - the name "Run I", which is already taken in the record by a different run.

## Prior record

### Searches run (2026-11-23, main tree `41ad509`, read-only; `/bin/grep`, because the shell `grep` skips `*.json`)

```
/bin/grep -rlE 'run I\b|Run I\b|skipsw_run_i|RUN_I'                         <governance set>
/bin/grep -rlE 'before the (first )?holdout|just before the holdout|before March 2021' <governance set>
/bin/grep -rlE 'L-9'                                                         <governance set>
  governance set = docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_*
                   docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
/bin/grep -lE 'run I|second (sieve|generation)|holdout' logs/e2e_session/*.txt
/bin/grep -n 'L-9' -A80 ~/tfm_worktrees/e2e-grader/docs/LEADS.md            (draft register, worktree e2e-grader)
```
Also read in full: `~/skipsw_blocks/h/b01-b18.sh`, run H's `logs/stage4/*` (in the run tree and in the evidence copy),
`receipt_thr3030.json`, `bundle_thr3030/bundle.json`, both OWNER_OVERRIDE files, `OWNER_RULING_baseline_3030_20261122.txt`,
the run H WATCHER log `~/skipsw_run_thr3030/logs/gate12_20261122_143253.log`, `launcher.out`, the run H miner ledger
(`?mode=ro&immutable=1`), `docs/RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md` (runs A–D; run H is not in it).

### Documents found that decide, authorise, schedule or record this subject

| Document | What it records | Label |
|---|---|---|
| `~/tfm_worktrees/e2e-grader/docs/LEADS.md:658-731` (draft L-9, not landed) | "⬜ A second generation with the sieve window placed just before the holdout. Run H's window is dated 2002." Status OPEN | ALREADY RECORDED (draft) — this runbook prepares that follow-up |
| `logs/e2e_session/OWNER_NOTE_rigs_ready_20261123.txt` | 17:06 "the rigs are ready to be booted into proxmox- lets delve into these" | ALREADY RECORDED; not a launch go |
| `logs/skipsw_session/OWNER_OVERRIDE_runH_thr3030.txt` | run H: one-off past the cap, cap file unchanged | ALREADY RECORDED; covers run H only |
| `logs/skipsw_session/OWNER_RULING_baseline_3030_20261122.txt` | 0.30 / 0.30 is the baseline, "a moveable parameter" | ALREADY RECORDED; Run I keeps 0.30 / 0.30 |
| `logs/e2e_session/OWNER_RULING_e2e_Q1_20261123.txt` | holdout cut in two: label block [0,1807) = 2021-03-17..2023-09-06, grade block [1807,3614) sealed | ALREADY RECORDED; Run I's window ends before both |
| `config_manifests/survivor_volume.cap` | cap 50,000 per phase, `compare: point`, "Raised only by the owner" | ALREADY RECORDED; unchanged |
| `docs/drafts/anchor_v1/STATUS_AND_PLAN_ANCHOR.md` §C, §G (this worktree) | a hand pin above 100 runs on Optuna's warning; midday holdout starts at filtered 6,707 | ALREADY RECORDED; re-derived below |
| `docs/exchange/to_beta/X-LEDGER-RELEASE_R1_ALPHA_REVIEW_REQUEST_hand_release_run_I.md:1,19,25` | **a different "Run I"** (`distributed_config_t1_34a6727c`, killed 16:24, 3,554 held reservations) | **NAME COLLISION** — §4 Q3 |
| `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90` | WP-6 names "L-4 to L-9" (ID overlap, noted in the L-9 draft) | ALREADY RECORDED |

Items this runbook raises:
- the anchor value 6695 and its proof lines: **NEW**;
- the blocks: **NEW**, derived from the run H blocks;
- the cap question: **NEW**, the same shape as run H's.

Nothing here **CHANGES A RULING**.

### Not searched
- The host, apart from the files named above.
- The rigs. None was contacted, so rig-side code, ROCm state and boot state were not read.
- ser8 archives, the private repository, and project files outside the repository.
- The `docs/LEADS.md` of other worktrees, except `e2e-grader`.

---

## 1. Run H exactly as launched (2026-11-22 14:31–14:39 PST)

Every value below is read from a recorded artefact, not from the block files. §1.1 explains why.

| Parameter | Run H value | Source (read 2026-11-23) |
|---|---|---|
| PRNG | `java_lcg` | `run_thr3030/logs/stage4/params.json` |
| Skip modes | both (`test_both_modes: true`), executed `["constant","variable"]` | params.json; `provenance.json skip_modes_executed` |
| Window size | 12 (`warm_start_window`) | params.json; ledger `trial_context.window_size = 12` |
| Window anchor | 25 (`warm_start_offset`) | params.json; ledger `trial_context.window_anchor_val = 25`; receipt `job_identity.window_anchor` |
| Generator phase | 0 | ledger `trial_context.generator_phase = 0`; receipt |
| Session | midday (`warm_start_session_idx: 1` → `["midday"]`) | params.json; search record `session_options[1]`; ledger `sessions_json` |
| Thresholds | forward 0.30, reverse 0.30 | params.json; ledger; `OWNER_RULING_baseline_3030_20261122.txt` |
| Trials | 1 (`window_trials: 1`, `n_parallel: 1`) | params.json; WATCHER log :45 `--trials 1 --n-parallel 1` |
| Worker pool | 25 (24 rig cards + VM101 `zeus-ubuntu-vm:gpu0`) | params.json; `launcher.out` sentinel table, 25/25 |
| Skip range | [0, 16], delivered to phases 1–2 (`deliver_skip_range: true`) | params.json; WATCHER log `[SKIP-DELIVERY] ON: phases 1-2 sieve skip [0, 16]`; receipt `constant_skip_range_effective [0,16]` |
| Seed slice | [2147483648, 2415919104) = [2³¹, 2³¹+2²⁸), `max_seeds 268435456`, stripe 67108864, 4 stripes per phase | receipt `seed_range`; `bundle.json geometry` |
| Dataset | `daily3-20260801T145551443433Z-513648160d35.json`, sha256 `513648160d35…68f6`, 18,068 records | receipt; WATCHER search_open.log `[P0.5] run dataset FROZEN` |
| Code | `00e2ee7`, detached worktree, clean | `cleantree_P6.log`; `provenance.json repository_commit` |
| Run tree | `/home/michael/skipsw_run_thr3030` (git worktree) | receipt `output_root` |
| Staging | `/home/michael/skipsw_staging_thr3030` (new, empty; never `~/miner_staging`) | receipt `staging_dir`; WATCHER log `coordinator staging VALIDATED` |
| Launcher | `~/skipsw_launch_thr3030.sh` = `~/stage4_miner_launch.sh` (sha256 `fe964b04…`) with only the `cd` line changed | `h/b07.sh`; `launcher.out entry_point sha256=bc354016…` |
| Search | `s1s-a15a5425…`, `kind: pinned`, `constant_skip_delivery: trial_range`, opened by `--step1-search-open` over the slice | run tree `prng_analysis.db` table `step1_search`; `search_open.log` |
| Coverage | certified [2147483648, 2415919104), `coverage_id 841b5ba8…`, linked to the search | WATCHER log `[S145 COVERAGE] certified`, `[STEP1-SEARCH LINK]` |
| Survivor-cap one-off | volume gate **REFUSE** (phase 3 projection 2,368,200; phase 4 595,773; cap 50,000); owner "please proceed" 14:30, cap file unchanged | `run_thr3030/logs/stage4/volume_gate.log`; `OWNER_OVERRIDE_runH_thr3030.txt` |
| Result | 12,156 survivors (all variable-skip), artifact `3f6a884d…`, generation `gen-20261122T223747461976Z-…` | receipt; WATCHER log `GENERATION PUBLISHED` |
| Rig parity | 30 MATCH / 0 MISMATCH (10 governed files × 3 rigs) | `launcher.out`; `logs/gate12_20261122_143253_source_digests.json` |
| Train/holdout split written | 14,454 train / 3,614 holdout | WATCHER log, end of Step 1 |

### 1.1 What the run H block files do not show (facts)

The files in `~/skipsw_blocks/h/` are copies of run C's blocks with the paths changed, and some assertions still
describe run C:
- `h/b06.sh` asserts that the params differ from run A by `{deliver_skip_range, warm_start_skip_max, staging_dir}`,
  and that `warm_start_skip_max == 32`. Run H's recorded params have `skip_max 16`, and differ from A
  (`run_off/logs/stage4/params.json`) in `deliver_skip_range`, both thresholds and `staging_dir`. As written, that
  assertion would fail on run H.
- `h/b09.sh` describes its verdict against run B's.
- `h/b16.sh` and `h/b17.sh` compare run C with run A.

Nothing on disk shows which of these blocks were run for H, or what they printed, apart from the recorded logs above.
**So "run H as launched" here means the recorded artefacts**, and Run I's blocks (§3) are written fresh against
those artefacts.

---

## 2. Run I: the one change

### 2.1 The anchor value, re-derived from the live dataset (CPU, read-only, 2026-11-23)

These figures are measured. The tools were the main tree's `miner.range_miner_worker.load_residue_window`
(signature `(path, window_size, sessions, window_anchor)`) and `sha256_residues`, called on
`daily3-20260801T145551443433Z-513648160d35.json`, whose sha256 `513648160d35…` matches both `daily3_current.json`
and `daily3.json`.

| Quantity | Value |
|---|---|
| Combined records N; Step 1 split `int(N*0.8)` (`window_optimizer.py:1170`) | 18,068; **14,454** |
| First holdout record (combined 14,454) | 2021-03-17 **midday** |
| Last train record (combined 14,453) | 2021-03-17 evening. Within a date, evening precedes midday in this file. |
| Midday N_filtered; midday records in `data[:14454]` | 8,515; **6,707** |
| **Largest midday anchor whose W12 window ends before the holdout** = 6,707 − 12 | **6,695** |
| Run I window, filtered [6695, 6706] = combined [14430, 14452] | **2021-03-05 midday … 2021-03-16 midday**, all in train |
| Residues (`draw`) | `[779, 961, 355, 547, 666, 324, 434, 271, 450, 38, 550, 315]` |
| `residue_sha256` | `ffb9ffa029c401ce5d80896a4447fcd104e205c8e16a5bbe0a3b26de222bf05d` |
| Anchor 6696 (control) | window ends at filtered 6707 = 2021-03-17 midday = **holdout**; sha `451b4575…` |
| Worker domain at W12 midday | [0, 8503]: 8503 loads, 8504 raises `ResidueResolutionError … outside the derived domain [0, 8503]` |
| Run H reproduction (control) | anchor 25 → filtered [25, 36] = 2002-11-29..2002-12-10, sha `c761272da958ba34…` = run H ledger `trial_context.residue_sha256` exactly |
| Grade cut (owner Q1 ruling) | label block starts at combined 14,454, grade block at 16,261. Run I's window (≤ 14,452) is in neither. |

### 2.2 The code it runs on

Run I runs on **main at `41ad509` as it is**, in a fresh detached worktree. **No unlanded code is used**: nothing
from this worktree (`anchor-brief2`) and no derived-domain switch. Three facts support this:
- `git diff --stat 00e2ee7 41ad509` = one file, `docs/RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md`, +341 lines.
- `git log 00e2ee7..HEAD` over `miner/`, the 10 governed files, `scripts/`, `agents/`, `window_optimizer*.py`,
  `utils/`, `config_manifests/` and `distributed_config.json` is empty.
- The 10 governed files in main hash today to the digests the rigs reported for run H. All 10 are present in
  `logs/gate12_20261122_143253_source_digests.json`; recomputed into
  `data/anchor_v1_logs/governed_main_now.sha256`.

Why a pinned anchor of 6695 passes on this code (read 2026-11-23):

1. WATCHER passes the seven pin keys to the CLI. `capture_step1_pin_bundle` checks that they are present, not their
   range (status §C, `agents/watcher_agent.py:614-646`). The search record stores the anchor without bounding it
   (`utils/step1_search_ledger.py:208`).
2. `window_optimizer_bayesian.py:815-819` enqueues `{'offset': 6695, …}` as trial 0. The objective's
   `trial.suggest_int('offset', 0, 100)` (`:543-545`; bounds from `distributed_config.json search_bounds.offset
   {min 0, max 100}`) returns the enqueued value. Optuna 4.4.0 treats it as a fixed parameter: out of range it emits a
   `UserWarning` and returns the value (status §C, measured on CPU).
3. `WindowConfig(offset=6695)` → `_miner_residues_for_config` (`window_optimizer_integration_final.py:273-296`)
   calls the same `load_residue_window` the worker calls.
4. The only other bound check on `offset`, `SearchBounds.is_valid` (`window_optimizer.py:239-247`), has no caller in
   `window_optimizer*.py`, `agents/watcher_agent.py` or `utils/`. The `n_parallel > 1` worker objective
   (`window_optimizer_integration_final.py:2997-3003`) is not on this path, because `n_parallel = 1`.
5. The volume gate (`utils/survivor_volume_estimator.py:171-172`) refuses only a negative anchor.

### 2.3 The lines that prove the intended window was sieved

The authority is the run's own miner ledger, with nothing between it and the worker:
- Every assignment carries `window_anchor` and `residue_sha256`.
- Each worker rebuilds the window from its own provisioned copy of the dataset, which it has checked against the
  dataset sha256 (`miner/range_miner_worker.py:1012-1021`).
- It raises `ResidueVerificationError("residue_sha256 mismatch …")` if its window's digest differs
  (`:1036-1042`), before any kernel runs.

So a stripe that completed sieved exactly the window whose digest is in the ledger.

| # | Where | Line that proves anchor 6695 | Pre/post launch |
|---|---|---|---|
| P1 | `logs/stage4/volume_gate.log` | `range [2147483648, 2415919104)  residue ffb9ffa029c401ce…  dataset 513648160d356617…` | pre (b09) |
| P2 | run tree `prng_analysis.db` `step1_search.search_space_json` | `effective.window_anchor == 6695`, `pins.warm_start_offset == 6695`, `kind == "pinned"` | pre (b10) |
| P3 | WATCHER log | `Warm-start: enqueued step1_trial_history (W12_O6695) as trial 0`; `NEW BEST [Trial 1]: W12_O6695_midday_S0-16_FT0.3_RT0.3`; `Offset: 6695`; `Confirmed window W12_O6695` | post (b12) |
| **P4** | **staging `miner_ledger.db` table `trial_context`** | **`window_size 12, window_anchor_val 6695, generator_phase 0, sessions ["midday"], residue_sha256 ffb9ffa029c401ce5d80896a4447fcd104e205c8e16a5bbe0a3b26de222bf05d`** | **post (b12) — the deciding line** |
| P5 | receipt `job_identity` | `"window_anchor": 6695, "window_size": 12, "sessions": ["midday"]` | post (b13) |
| P6 | published NPZ | array `offset` (int32) unique values `[6695]` for every row | post (b16) |
| P7 | all stripes, ledger `stripes` | 16 stripes (4 per phase × 4 phases), none failed; no `residue_sha256 mismatch` in the WATCHER log or `logs/miner_workers/` | post (b12) |

The *dates* of the window are not written in any run record. They follow from P4's digest, which §2.1 maps to
2021-03-05..2021-03-16 on the frozen dataset sha256.

**Signatures of a silent fallback or a clamp.** Any one of these means STOP: do not extract, do not hand the run
downstream, and report.

- **Fallback to 25:** `W12_O25` anywhere in P3, `window_anchor_val 25`, residue `c761272da958ba34…`, or NPZ
  `offset` = 25.
- **Clamp:** an anchor of **100** (the top of Optuna's `[0, 100]`), **8503** (the worker's derived max), or any value
  other than 6695 in P2–P6. Likewise a residue digest other than `ffb9ffa0…` in P1 or P4.
- **Not silent, and not a fallback:** a `ResidueResolutionError … outside the derived domain` or a
  `ResidueVerificationError` is the code refusing. The run fails loudly with no generation published.
- **Expected but not decisive:** Optuna's `UserWarning: Fixed parameter 'offset' with value 6695 is out of range …`.
  Whether the child's stderr reaches the WATCHER log was not read, so its absence alone is not a failure.

---

## 3. The run blocks (TEXT ONLY: not written to `~/skipsw_blocks/i/`)

Each block is a separate paste, in order, in one `bash` shell on VM101 as `michael`. Variables come from b01.
- Every background job uses `nohup`.
- Nothing this runbook adds writes under `/tmp`. The unchanged base launcher writes there on the rigs; see §6 item 14.
- `/home/michael/miner_staging` is never opened for writing. b02 and b18 only `stat` it.
- **b11 is the launch. Do not paste it without the owner's go and, if b09 refuses, the cap one-off (§4).**

**b01: names**
```bash
MAIN=/home/michael/distributed_prng_analysis
COMMIT=41ad509
TI=/home/michael/skipsw_run_i            # run I tree (new)
SI=/home/michael/skipsw_staging_i        # run I staging (new, empty; never ~/miner_staging)
LI=/home/michael/skipsw_launch_i.sh      # run I launcher (new)
EV=$MAIN/logs/skipsw_session             # shared evidence dir (gitignored, .gitignore:85); run I files suffixed _runi
HREF=$EV/bundle_thr3030                  # run H frozen bundle, READ ONLY
HPAR=$EV/run_thr3030/logs/stage4         # run H frozen params, READ ONLY
S=2147483648; E=2415919104
ANCHOR=6695
RSHA=ffb9ffa029c401ce5d80896a4447fcd104e205c8e16a5bbe0a3b26de222bf05d
LIVE=/home/michael/miner_staging
```

**b02: pre-flight, VM101 and the rigs' boot state (read-only)**
```bash
source ~/venvs/torch/bin/activate
ip -4 addr show | grep -E 'inet 192\.168\.3\.177'
pgrep -af "[w]atcher_agent|[w]indow_optimizer|[r]ange_miner_worker|[p]wc_worker_service|[s]ieve_gpu_worker"   # must print nothing
# Boot state: rigs must be in PROXMOX, not Ubuntu. Bare-metal IPs DOWN, Proxmox hosts and CT100s UP.
for ip in 192.168.3.120 192.168.3.154 192.168.3.162; do ping -c1 -W1 $ip >/dev/null && echo "$ip UP - rig is in UBUNTU: STOP" || echo "$ip down (ok)"; done
for ip in 192.168.3.121 192.168.3.155 192.168.3.163; do ping -c1 -W1 $ip >/dev/null && echo "$ip host up (ok)" || echo "$ip host DOWN: STOP"; done
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  ssh -n -o BatchMode=yes -o ConnectTimeout=5 michael@$ip 'hostname; pgrep -af "[r]ange_miner_worker|[p]wc_worker_service|[s]ieve_gpu_worker" || echo "  no workers"; ls /sys/class/kfd/kfd/topology/nodes | wc -l; ls /dev/dri | grep -c renderD'
done   # expect hostnames rrig6600 / rrig6600b / rrig6600c, no workers; note the node and renderD counts (8 renderD per CT expected)
cd $MAIN && git rev-parse HEAD && test -z "$(git status --porcelain)" && echo "MAIN CLEAN"     # must print 41ad509… and MAIN CLEAN
HALT=$(cd $MAIN && python3 -c "from agents.safety.kill_switch import KillSwitch; print(KillSwitch().halt_file_path)")
test ! -e "$HALT" && echo "NO HALT FLAG ($HALT)"
diff <({ stat -c '%n %s %Y' $MAIN/prng_analysis.db; readlink $MAIN/.s172_accumulator/current; }) $EV/live_state_before.txt && echo "LIVE UNCHANGED SINCE 2026-11-22"
stat -c '%n %s %Y' $LIVE/miner_ledger.db $LIVE/miner_ledger.db-wal $LIVE/miner_ledger.db-shm | tee $EV/live_staging_before_runi.txt
python3 -c "import json;print(json.load(open('$MAIN/daily3_current.json'))['sha256'])"   # must start 513648160d35
test ! -e $TI && test ! -e $SI && test ! -e $LI && echo "RUN I PATHS FREE"
nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader   # only sunshine allowed (owner 2026-11-14)
```

**b03: run tree and staging**
```bash
cd $MAIN
git worktree add --detach $TI $COMMIT
mkdir -p $TI/optuna_studies $TI/logs/stage4
mkdir -p $SI && test -z "$(ls -A $SI)" && echo "STAGING EMPTY"
```

**b04: inputs (the run H list, unchanged)**
```bash
VER=$(python3 -c "import json;print(json.load(open('$MAIN/daily3_current.json'))['filename'])")
IGN="daily3_current.json $VER daily3.json dataset_provisioning.json trse_context.json schemas/field_mappings.json"
for f in $IGN; do cp -p "$MAIN/$f" "$TI/$f"; done
diff <(cd $MAIN && sha256sum $IGN) <(cd $TI && sha256sum $IGN) && echo "$TI inputs identical"
(cd $MAIN && sha256sum $IGN) > $TI/logs/stage4/inputs.sha256          # kept this time (run H's were checked, not saved)
(cd $TI && test -z "$(git status --porcelain)" && echo "$TI CLEAN")
```

**b05: provision the dataset to the fleet (run H's b05, `$TC` → `$TI`)**
```bash
(
cd "$TI" || { echo "cannot enter $TI - STOP"; exit 1; }
rm -f logs/stage4/provision.ok logs/stage4/provision.log
PYTHONPATH=. python3 -u scripts/provision_dataset_fleet.py 2>&1 | tee logs/stage4/provision.log
rc=("${PIPESTATUS[@]}")
echo "provisioner rc=${rc[0]} log-writer rc=${rc[1]}"
h=""
if [ "${rc[0]}" -eq 0 ] && [ "${rc[1]}" -eq 0 ] && [ -s logs/stage4/provision.log ] \
   && h=$(sha256sum logs/stage4/provision.log | cut -d' ' -f1) && [ -n "$h" ] \
   && echo "PROVISION_OK $(date -u +%FT%TZ) log_sha256=$h" > logs/stage4/provision.ok \
   && /bin/grep -q "log_sha256=$h" logs/stage4/provision.ok; then
  echo "$TI: provisioned, receipt names log sha256 $h"; exit 0
fi
rm -f logs/stage4/provision.ok; echo "$TI: provisioning NOT confirmed - STOP"; exit 1
)
echo "provision status=$?"                         # must be 0
```

**b06: params = run H's, with only `warm_start_offset` (and the staging path) changed**
```bash
cat > $TI/logs/stage4/params.json <<'EOF'
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 268435456, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 6695, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.30, "warm_start_rev_thresh": 0.30, "warm_start_skip_min": 0, "warm_start_skip_max": 16, "deliver_skip_range": true, "staging_dir": "/home/michael/skipsw_staging_i"}
EOF
python3 - <<'EOF'
import json
i = json.load(open("/home/michael/skipsw_run_i/logs/stage4/params.json"))
h = json.load(open("/home/michael/distributed_prng_analysis/logs/skipsw_session/run_thr3030/logs/stage4/params.json"))   # run H frozen, read only
diff = {k for k in set(h) | set(i) if h.get(k, "<absent>") != i.get(k, "<absent>")}
assert diff == {"warm_start_offset", "staging_dir"}, diff
assert h["warm_start_offset"] == 25 and i["warm_start_offset"] == 6695
assert i["staging_dir"] == "/home/michael/skipsw_staging_i"
print("params ok: I = H + warm_start_offset 6695 (staging aside)")
EOF
```

**b07: launcher (run H's b07, new names)**
```bash
test "$(sha256sum /home/michael/stage4_miner_launch.sh | cut -d' ' -f1)" = fe964b04fe3ea431986835ff2fff934349864751f8974da5ccebb6b1a62bf5d3 && echo "base launcher ok"
sed 's#^cd /home/michael/stage4_run_miner || exit 1$#cd /home/michael/skipsw_run_i || exit 1#' /home/michael/stage4_miner_launch.sh > $LI
diff /home/michael/stage4_miner_launch.sh $LI | /bin/grep -c '^[<>]'     # must be 2
chmod +x $LI && bash -n $LI && echo "run I launcher syntax ok"
```

**b08: clean tree, GPU count and rig parity, before anything is armed (read-only probes; the launcher repeats all three)**
```bash
cd $TI
python3 -u scripts/gate12_cleantree_gate.py --phase admission 2>&1 | tee logs/stage4/cleantree_P6.log; echo "rc=${PIPESTATUS[0]}"     # PASS, HEAD 41ad509…
python3 -u scripts/gate12_gpu_gate.py --rig-profile proxmox 2>&1 | tee logs/stage4/gpu_gate_pre.log; echo "rc=${PIPESTATUS[0]}"      # 3/3 rigs OK 8/8
python3 -u scripts/gate12_parity_gate.py --rig-profile proxmox --evidence-json logs/stage4/parity_pre.json 2>&1 | tee logs/stage4/parity_pre.log; echo "rc=${PIPESTATUS[0]}"   # 30 MATCH
python3 - <<'EOF'
import json
pre = json.dumps(json.load(open("/home/michael/skipsw_run_i/logs/stage4/parity_pre.json")))
h   = json.load(open("/home/michael/skipsw_run_thr3030/logs/gate12_20261122_143253_source_digests.json"))
want = [l.split()[0] for l in open("/home/michael/tfm_worktrees/anchor-brief2/data/anchor_v1_logs/governed_main_now.sha256")]
assert all(d in pre for d in want) and all(d in json.dumps(h) for d in want), "a governed digest moved since run H - STOP"
print("rig parity digests = run H's 10 digests (no miner file changed since the 2026-11-22 deploy)")
EOF
```

**b09: survivor-volume gate (VM101 3080 Ti, ~35 s kernel time in run H; GPU must hold only `sunshine`)**
```bash
cd $TI
cat > logs/stage4/volume_params.json <<'EOF'
{"seed_start": 2147483648, "seed_end_exclusive": 2415919104, "max_seeds": 268435456, "prng_type": "java_lcg", "window_size": 12, "window_anchor": 6695, "generator_phase": 0, "sessions": ["midday"], "forward_threshold": 0.30, "reverse_threshold": 0.30, "skip_min": 0, "skip_max": 16, "dataset": "/home/michael/skipsw_run_i/daily3-20260801T145551443433Z-513648160d35.json", "test_both_modes": true, "deliver_skip_range": true}
EOF
python3 -u scripts/survivor_volume_gate.py --params @logs/stage4/volume_params.json \
  --record-out logs/stage4/volume_gate_record.json 2>&1 | tee logs/stage4/volume_gate.log
echo "rc=${PIPESTATUS[0]}"     # 0 = PROCEED; 1 = REFUSE (bring §4 Q2 to the owner with these figures; do not launch without the answer)
/bin/grep -n "residue ffb9ffa029c401ce" logs/stage4/volume_gate.log || echo "P1 FAILED: gate sampled a different window - STOP"
/bin/grep -n "phase [1234] java_lcg\|PROCEED\|REFUSE\|point figure" logs/stage4/volume_gate.log
#   run H (anchor 25): phase 1 hits 1, phase 2 hits 1, phase 3 projection 2,368,200, phase 4 595,773 -> REFUSE.
```

**b10: open the Step 1 search (run H's b10, new names); assert P2**
```bash
cd "$TI" || { echo "cannot enter $TI - STOP"; exit 1; }
P="$(python3 -c "import json; print(json.dumps(json.load(open('logs/stage4/params.json'))))")" && [ -n "$P" ] || { echo "params read failed - STOP"; exit 1; }
PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-open \
  --search-range-start 2147483648 --search-range-end 2415919104 \
  --search-reason "<owner's words: run I, L-9 follow-up, run H with anchor 6695 (midday 2021-03-05..2021-03-16, ends before holdout), skipsw_run_i, [2^31, 2^31+2^28), 0.30/0.30, skip [0,16]; owner <date time>>" \
  --params "$P" 2>&1 | tee logs/stage4/search_open.log
PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-status --params "$P" 2>&1 | tee logs/stage4/search_status.log   # state "gap", next_seed_start 2147483648
stat -c '%n %s %Y' $MAIN/prng_analysis.db      # must equal live_state_before.txt
python3 - <<'EOF'
import sqlite3, json
con = sqlite3.connect("file:/home/michael/skipsw_run_i/prng_analysis.db?mode=ro&immutable=1", uri=True)
rows = con.execute("SELECT search_id, search_space_json, repository_commit FROM step1_search").fetchall(); con.close()
assert len(rows) == 1, len(rows)
sp = json.loads(rows[0][1])
print(rows[0][0], rows[0][2][:7], sp.get("kind"), sp.get("constant_skip_delivery"), sp["effective"])
assert sp.get("constant_skip_delivery") == "trial_range", "switch missing - STOP"
assert sp["effective"]["window_anchor"] == 6695 and sp["pins"]["warm_start_offset"] == 6695, "P2 FAILED - STOP"
assert sp["effective"]["sessions"] == ["midday"] and sp["effective"]["window_size"] == 12
EOF
```

**b11: LAUNCH (owner's go required; and the cap one-off if b09 refused)**
```bash
nohup bash $LI > $TI/logs/stage4/launcher.out 2>&1 &
echo $! > $TI/logs/stage4/launcher.pid
```

**b12: after the WATCHER exits: P3, P4, P7 (read, never `tail -f`)**
```bash
cd $TI
W=$(sed -n 's/.*watcher pid=[0-9]* -> //p' logs/stage4/launcher.out); echo "$W"
/bin/grep -n "SKIP-DELIVERY\|STEP1-SEARCH S7\] credited\|GENERATION PUBLISHED\|Prior rows\|S145 COVERAGE\] certified\|STEP1-SEARCH LINK\|NO CREDIT\|INCOMPLETE\|lease expir\|LEASE_EXPIRED\|requeue" $W
/bin/grep -n "W12_O\|^Offset:\|Fixed parameter 'offset'" $W                    # P3: every hit must say 6695; any W12_O25 = fallback, STOP
/bin/grep -rn "residue_sha256 mismatch\|ResidueResolutionError\|ResidueVerificationError" $W logs/miner_workers/ && echo "P7 FAILED" || echo "P7: no residue refusal"
python3 - <<'EOF'
import sqlite3
con = sqlite3.connect("file:/home/michael/skipsw_staging_i/miner_ledger.db?mode=ro&immutable=1", uri=True)
ctx = con.execute("SELECT run_id, trial_number, window_size, window_anchor_val, generator_phase, sessions_json, skip_min, skip_max, forward_threshold, reverse_threshold, dataset_sha256, residue_sha256 FROM trial_context").fetchall()
st  = con.execute("SELECT phase, family_name, count(*), sum(state='done'), sum(survivors_total), round(max(elapsed_s),1) FROM stripes GROUP BY phase, family_name").fetchall()
tr  = con.execute("SELECT run_id, trial_number, state, terminal_class FROM trials").fetchall()
con.close()
for r in ctx: print("trial_context", r)
for r in st: print("stripes", r)
print("trials", tr)
assert len(ctx) == 1, ctx
c = ctx[0]
assert (c[2], c[3], c[4], c[5]) == (12, 6695, 0, '["midday"]'), "P4 FAILED: window/anchor/phase/session - STOP"
assert c[11] == "ffb9ffa029c401ce5d80896a4447fcd104e205c8e16a5bbe0a3b26de222bf05d", "P4 FAILED: residue digest - STOP"
assert c[10].startswith("513648160d35")
assert len(st) == 4 and all(r[2] == 4 and r[3] == 4 for r in st), "P7 FAILED: a stripe is not 'done' - STOP"
print("P4 PASS: the fleet sieved midday 2021-03-05..2021-03-16 (anchor 6695)")
EOF
#   'done' is the only stripe state in run H's ledger (read 2026-11-23); 4 stripes x 4 phases as in run H.
P="$(python3 -c "import json; print(json.dumps(json.load(open('logs/stage4/params.json'))))")"
PYTHONPATH=. python3 agents/watcher_agent.py --step1-search-status --params "$P" 2>&1 | tee logs/stage4/search_status_after.log   # search_complete
cat $TI/.s172_accumulator/skip_delivery_arm                     # must print trial_range
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do ssh -n -o BatchMode=yes michael@$ip 'hostname; pgrep -af "[r]ange_miner_worker" || echo "  no workers"'; done
```

**b13: receipt (run H's b13, suffix `_runi`); assert P5**
```bash
cd $MAIN && python3 -u scripts/extract_miner_bundle.py --staging-dir $SI --output-root $TI \
  --params-json $TI/logs/stage4/params.json --write-receipt $EV/receipt_runi.json \
  2>&1 | tee $EV/receipt_runi.log; echo "rc=${PIPESTATUS[0]}"
sha256sum $EV/receipt_runi.json | tee $EV/receipt_runi.sha256
python3 -c "
import json; j=json.load(open('$EV/receipt_runi.json'))['job_identity']; print(j)
assert j['window_anchor']==6695 and j['window_size']==12 and j['sessions']==['midday'] and j['generator_phase']==0, 'P5 FAILED'
print('P5 PASS')"
```

**b14: freeze the run's evidence (run H's b14, new names)**
```bash
R=$EV/run_runi; mkdir -p $R
cd $TI && cp -a .s172_accumulator results optimal_window_config.json bidirectional_survivors.json logs $R/ 2>/dev/null
cp -p $TI/prng_analysis.db* $R/
mkdir -p $R/staging && cp -p $SI/miner_ledger.db* $SI/threshold_provenance.json $R/staging/ 2>/dev/null
(cd $R && find . -type f -print0 | sort -z | xargs -0 sha256sum) > $EV/run_runi.sha256
```

**b15: bundle (run H's b15, new names)**
```bash
cd $MAIN && python3 -u scripts/extract_miner_bundle.py --staging-dir $SI --output-root $TI \
  --params-json $TI/logs/stage4/params.json --out-dir $EV/bundle_runi --receipt $EV/receipt_runi.json \
  2>&1 | tee $EV/extract_runi.log; echo "rc=${PIPESTATUS[0]}"
```

**b16: Run I against run H: inputs differ in exactly the anchor; P6 (new checker; run H's b16 was C-vs-A)**
```bash
cat > /home/michael/skipsw_outputs_i_vs_h.py <<'EOF'
"""Run I (anchor 6695) against run H (anchor 25). Exit 0 = every rule holds. Facts only: survivor counts are reported, not judged."""
import json, sys
import numpy as np
h_dir, i_dir = sys.argv[1], sys.argv[2]
H, I = (json.load(open(f"{d}/bundle.json")) for d in (h_dir, i_dir))
ok = True
def say(rule, good, detail=""):
    global ok
    ok &= bool(good)
    print(f"{'PASS' if good else 'FAIL'}  {rule}  {detail}")
d_in = {k for k in set(H["inputs"]) | set(I["inputs"]) if H["inputs"].get(k, "<absent>") != I["inputs"].get(k, "<absent>")}
say("inputs differ exactly in warm_start_offset + repository_commit", d_in == {"param.warm_start_offset", "run.repository_commit"}, sorted(d_in))
say("I warm_start_offset is 6695", I["inputs"].get("param.warm_start_offset") == 6695)
say("H warm_start_offset is 25", H["inputs"].get("param.warm_start_offset") == 25)
g = lambda b: {k: v for k, v in b["geometry"].items() if k != "staging_dir"}
say("geometry equal (staging aside)", json.dumps(g(H), sort_keys=True) == json.dumps(g(I), sort_keys=True))
say("credit granted over the same interval", I["credit"].get("granted") is True and I["credit"].get("certified_interval") == H["credit"].get("certified_interval"), I["credit"].get("certified_interval"))
for ph in ("1", "2", "3", "4"):
    print(f"fact  phase {ph} survivor_rows  H={H['phase_counts'][ph]['survivor_rows']}  I={I['phase_counts'][ph]['survivor_rows']}")
with np.load(f"{i_dir}/{I['npz']}", allow_pickle=False) as y, np.load(f"{h_dir}/{H['npz']}", allow_pickle=False) as x:
    say("npz array names equal", sorted(x.files) == sorted(y.files))
    n = len(y["seeds"])
    say("P6: I npz offset == 6695 on every row", n == 0 or set(np.unique(y["offset"]).tolist()) == {6695}, f"rows={n} values={np.unique(y['offset']).tolist()[:5]}")
    say("I npz window_size == 12 on every row", n == 0 or set(np.unique(y["window_size"]).tolist()) == {12})
    shared = len(set(x["seeds"].tolist()) & set(y["seeds"].tolist()))
    print(f"fact  rows H={len(x['seeds'])} I={n}  seeds in both={shared}")
print("artifact H", H.get("artifact_sha256"), "I", I.get("artifact_sha256"))
print("ALL RULES HOLD" if ok else "A RULE FAILED"); sys.exit(0 if ok else 1)
EOF
python3 -I /home/michael/skipsw_outputs_i_vs_h.py $HREF $HREF 2>&1 | /bin/grep -c '^FAIL'  # self-test: H vs H must print 3 (inputs, I offset 6695, P6)
#   measured 2026-11-23 on CPU against bundle_thr3030 (scratch copy of this checker): 3 FAIL, rc=1, P6 "values=[25]"
```

**b17: comparator and checker**
```bash
cd $MAIN
python3 -u scripts/compare_engine_outputs.py $HREF/bundle.json $EV/bundle_runi/bundle.json 2>&1 | tee $EV/compare_runi_vs_thr3030.txt; echo "comparator rc=${PIPESTATUS[0]}"
#   expect rc 2 (NOT COMPARABLE): its DIFFERENT input rows are exactly param.warm_start_offset and run.repository_commit
python3 -I /home/michael/skipsw_outputs_i_vs_h.py $HREF $EV/bundle_runi 2>&1 | tee $EV/rules_runi_vs_thr3030.txt; echo "rc=${PIPESTATUS[0]}"   # 0 = ALL RULES HOLD
```

**b18: post-run: cards, fault lines, no workers left, live ledgers unchanged**
```bash
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  ssh -n -o BatchMode=yes michael@$ip 'hostname
    pgrep -af "[r]ange_miner_worker|[p]wc_worker_service|[s]ieve_gpu_worker" || echo "  no workers"
    echo "  kfd nodes $(ls /sys/class/kfd/kfd/topology/nodes | wc -l)  renderD $(ls /dev/dri | grep -c renderD)"
    rocm-smi --showreplaycount 2>/dev/null | /bin/grep -i -E "GPU\[|replay" | head -16
    rocm-smi --showmemuse 2>/dev/null | /bin/grep -i "vram%" | head -8
    /bin/grep -l -i -E "GCVM|page fault|protection fault|HSA_STATUS_ERROR|hipError|gpu reset|ring .* timeout" /tmp/minerlogs/*.log 2>/dev/null || echo "  no fault lines in worker logs"'
done
#   kfd/renderD counts must equal b02's; VRAM back to ~0; replay counts not increased vs a pre-run read if one was taken.
#   dmesg is impossible inside the unprivileged CT100 (empty = NOT evidence of no fault); RAS is "Not supported" on RX 6600 XT: report as unavailable.
pgrep -af "[w]atcher_agent|[w]indow_optimizer|[r]ange_miner_worker" || echo "VM101: no workers left"
diff <({ stat -c '%n %s %Y' $MAIN/prng_analysis.db; readlink $MAIN/.s172_accumulator/current; }) $EV/live_state_before.txt && echo "LIVE UNCHANGED"
diff <(stat -c '%n %s %Y' $LIVE/miner_ledger.db $LIVE/miner_ledger.db-wal $LIVE/miner_ledger.db-shm) $EV/live_staging_before_runi.txt && echo "LIVE MINER LEDGER UNCHANGED"
cd $MAIN && test -z "$(git status --porcelain)" && echo "MAIN STILL CLEAN"
```

**After b18 (not in this runbook):** the same chain as run H's: Step 3, Step 5, vote, grade. Each step needs its own
owner go, as run H's did (`logs/e2e_session/OWNER_GO_step3_*`, `OWNER_GO_vote_and_selfplay_20261123.txt`,
`OWNER_GO_counted_grade_20261123.txt`). The harnesses are drafts in worktrees `e2e-map`, `e2e-vote` and `e2e-grader`.
Their run-copy builders refuse to write under the main tree or the run H tree. Whether they also refuse
`~/skipsw_run_i` was not read.

---

## 4. Owner's word needed before launch

| # | Needed for | Draft question (one line, exact) |
|---|---|---|
| Q1 | the go | "Do you approve launching Run I: run H unchanged except the window anchor 6695 (midday 2021-03-05 to 2021-03-16, the last 12 midday draws before the holdout), on main 41ad509, rigs in Proxmox, run tree ~/skipsw_run_i?" |
| Q2 | the cap one-off; ask only if b09 prints REFUSE | "The survivor-volume gate refuses Run I (phase 3 projection <P3>, phase 4 <P4>, cap 50,000; run H refused at 2,368,200 / 595,773 and you approved a one-off): do you approve Run I as a one-off past the cap, cap file unchanged?" |
| Q3 | the record (not a launch blocker) | "'Run I' already names the 2026-11-23 16:24 killed run in X-LEDGER-RELEASE (distributed_config_t1_34a6727c); keep 'Run I' for this run, or call it something else in the record?" |

If b09 prints PROCEED, Q2 is not asked. The record then says so, with the gate's figures.

---

## 5. Expected duration and load (from run H's measured figures)

These are run H's measured figures. Run I's own figures will be measured, not predicted.

| Stage | Run H measured | Source |
|---|---|---|
| Blocks b03 → launcher start | 14:31 → 14:32:53 | `provision.ok 22:31:04Z`; `launcher.out` |
| Launcher gates → WATCHER start | 14:32:53 → 14:34:31 (~1.6 min: clean tree, GPU, parity, sentinels, liveness, cache join) | `launcher.out`; WATCHER log :1 |
| Trial (created → finalized) | **130.8 s** | receipt `trial_created_at` / `trial_finalized_at`; bundle `duration_s` |
| Per phase (4 stripes each; max / sum of stripe `elapsed_s`) | 1: 5.7 / 22.6 s · 2: 3.5 / 14.1 s · 3: 46.4 / 180.3 s · 4: 16.6 / 61.5 s | run H ledger `stripes` |
| Publication lag | 59.2 s | receipt `publication_lag_s` |
| WATCHER end | 14:38:01 | WATCHER log |
| Extraction and bundle | 14:39 | `receipt_thr3030.log`, `extract_thr3030.log` |
| Volume gate (VM101 3080 Ti) | 3.2 + 3.7 + 9.9 + 14.8 s kernel, plus compile | `volume_gate.log` |
| Rows | phase 3: 2,368,585; phase 4: 595,773; final bidirectional 12,156; NPZ 57,958 bytes | ledger; bundle |
| Fleet | 25 workers (24 RX 6600 XT + VM101 gpu0); 4 of 25 serviced per stage (4 stripes per phase) | `launcher.out`; WATCHER `S172-SL` window |

**Expected:** about 8 minutes from b03 to b15, plus the paced pastes. The trial should take about 2 minutes if phases
3 and 4 carry similar volumes. VM101 peak memory for run H was not recorded.

**Unknown:**
- Run I's survivor counts in phases 3 and 4. They depend on the residues: phase 3 timing (46 s per stripe) tracked
  its 2.37 M rows, so a different volume changes that time and the staging size.
- The final bidirectional count, which could be 0. A 0 is a result, not a failure.
- VM101 peak RSS during publication.
- Whether the rigs come up healthy after the reboot into Proxmox.

---

## 6. Every difference from run H other than the anchor that could not be ruled out

1. **Commit:** `41ad509` against `00e2ee7`. The difference is documentation only, so the code is identical by
   `git diff`. But `provenance.repository_commit` and the bundle input row `run.repository_commit` differ; b16 and b17
   expect that difference.
2. **Rig state after boot:** the rigs were off and are to be booted into Proxmox. The ROCm userspace, firmware,
   modparams and the rig-side copies of the 10 governed files were not read (no rig contact). The launcher's parity
   and GPU gates, and b08, check the files and the card count at launch, not the rest.
3. **Optuna path:** run H's pinned anchor 25 was inside `[0, 100]`; Run I's 6695 goes through Optuna's out-of-range
   fixed-parameter branch, which only warns. It was measured on CPU with an in-memory study (status §C), but has never
   run inside Step 1 on the fleet. P2–P6 detect any deviation.
4. **Survivor volume and cap verdict:** Run I's phase 3 and 4 volumes, staging size and volume-gate figures will
   differ, so the cap question may come out differently from run H's.
5. **Residue content:** the window holds a two-digit draw (38) and different values. The kernels take ints, so by
   reading this does not matter; it is noted because no earlier run used this window.
6. **Search, study and coverage identity:**
   - Run I gets a new search id, study and coverage id over the **same** seed interval.
   - The S145 coverage record does not carry the anchor (WATCHER log `certified [...) for java_lcg
     modes={constant,variable}`).
   - Both runs live in separate run trees, so nothing overlaps here. If the two ledgers were ever merged, run H's and
     Run I's coverage of [2³¹, 2³¹+2²⁸) would look identical.
7. **Inputs copied in b04:** all six predate 2026-11-22 by mtime (`trse_context.json` 2026-07-31; dataset files
   2026-08-01; `schemas/field_mappings.json` 2025-11-03). Run H's b04 compared sha256s against main but did not save
   them, so equality with run H's copies is by mtime and dataset sha, not by a saved hash. Run I's b04 saves them.
8. **TRSE side effect:** Step 1 logs `Confirmed window W12_O6695` into the run tree's own `trse_context.json`.
   That is local to the tree, as `W12_O25` was for H.
9. **Fleet scheduling:** which card takes which stripe, the registration order and stripe timing are not
   deterministic. Run H used 4 of the 25 workers per stage, and rigs differ in speed by 12–15% (memory: rig
   throughput heterogeneity).
10. **VM101 local worker:** `zeus-ubuntu-vm:gpu0` is in the 25. Its GPU state at launch (only `sunshine` allowed) is
    checked in b02, not continuously.
11. **Monitoring:** run H's monitoring state was not recorded in the run H artefacts read here. The record of
    telemetry, polling and netconsole all being off covers run D (`RESULT_SKIP_SWITCH_RIG_RUNS_20261122.md` §7a), not
    run H. So Run I's monitoring state may differ from H's in a way that cannot be shown.
12. **Downstream consumers of `offset`:** Step 3 shell scripts and `prediction_generator.py` were not traced for
    reads of the window `offset` (status §I, "Not traced this session"). The train/holdout split is independent of the
    anchor: `int(N*0.8)`, again 14,454 / 3,614. Whether any Step 3–6 or grade code reads `optimal_window_config.json`'s
    `offset` and behaves differently at 6695 is not ruled out.
13. **Window position relative to the holdout:** run H's window was 12,000 combined records before the holdout. Run
    I's ends 2 combined records before it (one evening record, 2021-03-17, sits between). That is the intended change.
    Its effect on Step 3 features that measure distance from the window was not traced.
14. **`/tmp` on the rigs and VM101:** the unchanged base launcher writes release tokens to `/tmp/minerlogs` on the
    rigs (`~/stage4_miner_launch.sh:265`, `scripts/launch_fleet_manual.sh:155`). It also puts cupy caches under `/tmp`
    (`launch_fleet_manual.sh:175-191`), including VM101's `/tmp/cupy_cache_local_gpu<N>`. Run H did the same.
    Changing it would be a second difference, so it is left as is and stated here.
15. **The run H block files are not a faithful record** (§1.1). Run I's blocks were rebuilt from run H's recorded
    artefacts, so any step run H performed that left no artefact cannot be shown to be repeated.
16. **Name collision:** "Run I" already names `distributed_config_t1_34a6727c` in
    `docs/exchange/to_beta/X-LEDGER-RELEASE_R1_ALPHA_REVIEW_REQUEST_hand_release_run_I.md` (§4 Q3). The on-disk names
    used here (`skipsw_run_i`, `_runi`) do not collide with any existing path.

---

## Evidence written by this preparation

- `data/anchor_v1_logs/governed_main_now.sha256`: sha256 of the 10 governed files in main, 2026-11-23. Read-only
  computation; it is the input to b08's comparison.
- This file.

- A scratch copy of the b16 checker, in the session scratchpad (not kept). It was run once on CPU as run H against
  itself: 3 FAIL, rc 1, as b16 expects.

Nothing else was written. The derivation in §2.1 ran on CPU, single process, under 1 s, reading the main tree's
dataset and `miner/range_miner_worker.py` read-only.
