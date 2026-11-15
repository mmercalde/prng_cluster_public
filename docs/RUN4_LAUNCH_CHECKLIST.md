# RUN 4 LAUNCH CHECKLIST — 2026-11-15

**For:** Michael (owner). **Prepared by:** Claude Code (Team Alpha) on VM101, 2026-11-14, from
committed HEAD `347b0369922076fa280efa46be3b1f03454d71ba`, with the fleet off. **Nothing in this
file has been run against the fleet.** Brief: `docs/CLAUDE_CODE_INSTRUCTIONS_RUN4_LAUNCH_PREP_20261114.md`.
Companion report (read its top section first): `docs/CLAUDE_CODE_REPORT_RUN4_LAUNCH_PREP.md`.

## Prior record

This checklist assembles steps that are already recorded. It raises nothing new except where
marked **NEW** (two operator notes, steps 8 and 14). The searches, the documents read and the
labels for every item are in the report's Prior record. The governing records:

| Record | What it fixes for this checklist |
|---|---|
| `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md:402-415` | The remaining steps and their order |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:79-82` (A5–A8) | Power-on, netconsole, Run 4, then `G-PROD-SHAPE` |
| `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:59` (A6), `:77-78` | Ten governed files go from committed HEAD, verified 30/30; `boot-proxmox` is the owner's |
| `docs/PHASE6_PREREQS.md:121-129`; `docs/S172_PHASE_7_PREREQ_REPORT.md:505-526` | Rigs are deployment targets; `git archive` export, `mkdir -p`, `scp -p`, absolute paths |
| `/home/michael/dashboard_work/R1_R4_FLEET_DEPLOY_RECORD.md` (host file, not in the repo) | The attempt-9 deploy: archive, stop check, deploy, parity before and after |
| `docs/RUNBOOK_NETCONSOLE_REARM.md` | Netconsole receiver, sender `[RECONSTRUCTED]`, the `NCPROOF` proof |
| `docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:290-293` (§2.9); `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md:41`, `:49` | Run 4 opens its own pinned search over `[2³¹, 2³²)`, by the operator |
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:154-179` | Run 4's claim boundary: operation only, no population-equivalence claim. Not widened |
| `docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:1-30`; `config_manifests/survivor_volume.cap` | Exact counts 0 / 0 / 60 / 23,568; cap 25,000 under `point` |
| `gate12_launch.sh` (committed) | The certified launch harness: gates, parked fleet, sentinel, liveness, sampler, release |
| `tests/run4_prelaunch_provenance_check.py` | The Route-A command with the seven pins |

## How to read this

- **Host tags** are on every command block: `# run on VM101`, `# run on <rig console>`,
  `# run on pve host .121/.155/.163 (root)`. VM101 is `192.168.3.177`.
- **Who:** OWNER = you. SESSION = a Claude Code session, only when you say so, and only for the
  steps the delegation proposal allows (A1, A2, A6).
- Every VM101 block assumes:
  ```bash
  # run on VM101
  cd /home/michael/distributed_prng_analysis && source ~/venvs/torch/bin/activate
  mkdir -p data/run4
  ```
  `data/run4/` holds only `*.log` and `*.json` files below, both ignored (`.gitignore:11`,
  `.gitignore:41`), so it does not dirty the tree.
- **UNAVAILABLE** marks a step for which the record gives no command. It says what is missing.

## Decision you make before step 13 (the only one)

**D-1. Which file launches Run 4.** The committed harness, `gate12_launch.sh`, launches with the
Gate-12 parameters written into it (`gate12_launch.sh:377-383`): no pins. The record has **no
committed launcher that carries Run 4's seven pins** (searched: `*.sh`, `scripts/*.sh`,
`~/dashboard_work/`; the 2026-08-22 production-shape run used `gate12_launch.sh` unchanged,
`logs/gate12_20260822_143303.log:7`). If you run `gate12_launch.sh` as it is after opening Run 4's
search, WATCHER should refuse before dispatch because the pins differ from the search record
(stage-1 gate G5, `docs/CLAUDE_CODE_REPORT_STEP1_SEARCH_RECORD_STAGE1.md:412`; by reading, not
executed tonight). The fleet would already be parked by then; the coordinator never binds, and the
harness kills the parked fleet (`gate12_launch.sh:403`, `:416-423`). That is a safe refusal, not a
run. Options are in step 13.

---

## 1. Rigs on, booted into Proxmox — OWNER

Each rig default-boots bare Ubuntu (`.120`/`.154`/`.162`). `boot-proxmox` makes it a Proxmox host
(`.121`/`.155`/`.163`) with its CT100 worker (`.122`/`.156`/`.164`). Sources: `CLAUDE.md` §3;
fact file §6; `docs/RUNBOOK_NETCONSOLE_REARM.md:57-59`.

```bash
# run on each rig console (rrig6600, rrig6600b, rrig6600c), after power-on
boot-proxmox
```
The record names the command only (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:77-78`). How it
is invoked (user, sudo) is not recorded.

**Confirm each rig is in the right OS before anything else:**

```bash
# run on VM101 — each CT100 answers with its own rig name and ROCm
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  ssh -o BatchMode=yes -o ConnectTimeout=5 michael@$ip 'hostname; id; rocminfo >/dev/null && echo ROCm_OK'
done
```
Source: `docs/PHASE6_PREREQS.md` item 2 (the acceptance command), `CLAUDE.md:152`.

- **Pass:** three answers, `rrig6600`, `rrig6600b`, `rrig6600c`, each followed by `ROCm_OK`. The
  bare-metal addresses `.120`/`.154`/`.162` do not answer. (`CLAUDE.md:143` records this as a
  "live ping sweep"; it gives no literal ping flags.) The Proxmox UI is at
  `https://192.168.3.121:8006` (`CLAUDE.md` §3, "verified by").
- **Fail:** a CT does not answer. That rig is most likely still in Ubuntu, not hung (fact file §6).
  Run `boot-proxmox` on it. **Do not reboot a host to fix it**; a reboot returns it as Ubuntu.

## 2. VM101 tree clean, HEAD recorded — OWNER or SESSION

```bash
# run on VM101
git status --porcelain          # must print nothing
git rev-parse HEAD
python3 -u scripts/gate12_cleantree_gate.py --phase admission
```
Sources: `docs/PHASE6_PREREQS.md` §5 acceptance; `gate12_launch.sh:133`.

- **Pass:** no porcelain output; the gate passes, rc 0. HEAD is `347b036` or a later commit (for
  example the one that commits these two documents). Committing docs does not change any governed
  file, so the coordinator digest stays `b8e259b3…`.
- **Fail:** any entry. Commit or deliberately dispose of it. **Do not widen any allowlist or add an
  ignore rule** (fact file §2.33, §2.35). An uncommitted `.py` also reds Gate 22.

## 3. No workers running anywhere — OWNER or SESSION

```bash
# run on VM101
pgrep -af "[r]ange_miner_worker\.py"
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  echo "== $ip"; ssh -n michael@$ip 'hostname; pgrep -af "[r]ange_miner_worker\.py"'
done
```
Source: `R1_R4_FLEET_DEPLOY_RECORD.md` §2 (bracket form, to avoid the self-match noted at
`scripts/launch_fleet_manual.sh:114`).

- **Pass:** no worker lines anywhere (each rig prints only its hostname).
- **Fail:** a worker is running. Stop and find out why before going on. A leftover daemon holds an
  identity in the frozen set (`docs/CLAUDE_CODE_INSTRUCTIONS_MANUAL_FLEET_LAUNCH.md` §2).

## 4. GPU gate, 8 of 8 on each rig — OWNER or SESSION

```bash
# run on VM101
python3 -u scripts/gate12_gpu_gate.py 2>&1 | tee data/run4/gpu_gate.log; echo "rc=${PIPESTATUS[0]}"
```
Source: `gate12_launch.sh:164`; delegation A2.

- **Pass:** `GATE-12 GPU GATE  : PASS — 3/3 rigs OK at full count. Launch may proceed.`, rc 0
  (`scripts/gate12_gpu_gate.py:198`; attempt 9's evidence, `logs/gate12_20260817_181819_evidence.txt:41`).
- **Fail:** REFUSE, `UNAVAILABLE` or `ERROR` on any rig. **A refusal is a refusal; there is no
  launch on fewer GPUs** (fact file §2.33). Check that rig's OS (step 1) and its GPUs on its console.

## 5. Parity before the deploy — the clean control — OWNER or SESSION

```bash
# run on VM101
python3 -u scripts/gate12_parity_gate.py --evidence-json data/run4/parity_pre.json 2>&1 \
  | tee data/run4/parity_pre.log; echo "rc=${PIPESTATUS[0]}"
```
Source: `gate12_launch.sh:192`; the pre-deploy run in `R1_R4_FLEET_DEPLOY_RECORD.md` §4.

**Expected: REFUSE, rc 1, `27 MATCH · 3 MISMATCH · 0 UNAVAILABLE`,** each mismatch naming
`miner/range_miner_coordinator.py` on one rig, expected `b8e259b3…`, found `59888834…`.

Why that is the expectation, from the record:
- On 2026-08-22 all ten governed files were deployed at `48a8705` and verified 30/30
  (`docs/SESSION_CHANGELOG_20260822_BRIEF_I_PRODUCTION_SHAPE_REPAIR.md:16-18`;
  `docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:62`).
- Since `48a8705`, the only governed file that changed is the coordinator:
  `git diff --stat 48a8705 HEAD -- <the ten files>` shows `miner/range_miner_coordinator.py` only
  (commits `04aab30`, `80d2bb4`). `git show 48a8705:miner/range_miner_coordinator.py | sha256sum`
  = `59888834bb1c86a270783f1b9fce7ede6b224fb666c47d3a2cabc8c11288c71b`.
- The other nine are byte-identical between `48a8705` and HEAD. **Not measured on the rigs since
  2026-08-22.** This step is the measurement.

- **If any other file mismatches, or any row is UNAVAILABLE:** stop. The record's account of the
  rigs is wrong somewhere. Do not deploy more than the one file without deciding it.

## 6. Coordinator to each rig, from committed HEAD — OWNER, or SESSION at your word (delegation A6)

Method: `git archive` export, `mkdir -p` before `scp`, `scp -p`, absolute paths, verify on the
target, clear `__pycache__` (`docs/S172_PHASE_7_PREREQ_REPORT.md:505-526`; `CLAUDE.md` §2). The
rollback copy follows the attempt-9 deploy's `cp -p` archive (`R1_R4_FLEET_DEPLOY_RECORD.md` §1).
The record deployed whole sets; here the set is narrowed to the one file that differs (step 5).

```bash
# run on VM101
test -z "$(git status --porcelain)" || { echo "TREE NOT CLEAN — stop"; false; }
H=$(git rev-parse --short HEAD); STAMP=$(date -u +%Y%m%d_%H%M%SZ)
EXP=/home/michael/run4_deploy_${H}          # outside the repository (keeps the tree clean)
mkdir -p "$EXP"
git archive HEAD miner/range_miner_coordinator.py | tar -x -C "$EXP"
sha256sum "$EXP/miner/range_miner_coordinator.py"
#   must print b8e259b3e12b95297fa38018a41215d5c52f4d9b3f4095e384829a3ec1f1517d — if not, STOP

for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  echo "== $ip"
  ssh -n michael@$ip "mkdir -p /home/michael/rig_archive/pre_run4_${STAMP}/miner && \
    cp -p /home/michael/distributed_prng_analysis/miner/range_miner_coordinator.py \
          /home/michael/rig_archive/pre_run4_${STAMP}/miner/ && \
    mkdir -p /home/michael/distributed_prng_analysis/miner"
  scp -p "$EXP/miner/range_miner_coordinator.py" \
      michael@$ip:/home/michael/distributed_prng_analysis/miner/
  ssh -n michael@$ip 'hostname; \
    sha256sum /home/michael/distributed_prng_analysis/miner/range_miner_coordinator.py \
              /home/michael/rig_archive/pre_run4_*/miner/range_miner_coordinator.py; \
    rm -rf /home/michael/distributed_prng_analysis/miner/__pycache__'
done
```

- **Pass:** each rig prints its own hostname, the deployed file at
  `b8e259b3e12b95297fa38018a41215d5c52f4d9b3f4095e384829a3ec1f1517d`, and the archived copy at
  `59888834…`.
- **Fail:** any other digest, or an `scp` error. Do not go on. Rollback is `cp -p` of the archived
  copy back into place (`R1_R4_FLEET_DEPLOY_RECORD.md` §6). Fix forward from committed HEAD; do not
  restore from any other backup (`CLAUDE.md` rule 4).
- Do not touch any rig environment: no `pip`, `apt` or Python change
  (`docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:69-76`). `rrig6600c`'s Python 3.10.12 is
  recorded and left alone.

## 7. Parity gate, 30 of 30 — OWNER or SESSION

```bash
# run on VM101
python3 -u scripts/gate12_parity_gate.py --evidence-json data/run4/parity_post.json 2>&1 \
  | tee data/run4/parity_post.log; echo "rc=${PIPESTATUS[0]}"
```
Source: `gate12_launch.sh:192`; `R1_R4_FLEET_DEPLOY_RECORD.md` §4.

- **Pass:** `rows : 30 MATCH · 0 MISMATCH · 0 UNAVAILABLE` and `GATE-12 PARITY GATE : PASS`, rc 0.
- **Fail:** anything else. Stop. `gate12_launch.sh` reruns this gate itself (§0.6) and refuses on it
  anyway.

## 8. Netconsole re-armed, one proof line per host — OWNER (root on the Proxmox hosts)

**Receiver (VM101).** Measured tonight: already listening, as a systemd service.

```bash
# run on VM101
ss -lunp | grep 6667
```
Source: `docs/RUNBOOK_NETCONSOLE_REARM.md:37`. Tonight it printed
`UNCONN … 0.0.0.0:6667 … users:(("python3",pid=1102,fd=3))`; PID 1102 is in
`system.slice/netconsole-listener.service`, started 2026-11-14 08:57:30 (VM101 boot).

- **NEW, operator note:** the runbook's §1 `nohup python3 -u netconsole_listener.py` is for when
  nothing listens. **Do not start a second listener** while the service holds port 6667. The
  runbook does not mention the service.

**Sender (each Proxmox host), `[RECONSTRUCTED]` in the record.** Run the commands in
`docs/RUNBOOK_NETCONSOLE_REARM.md` §2 (`:62-76`) as root on `.121`, `.155`, `.163`, from the host
console or an interactive root login. VM101 has no root key to them. The record says these
commands are reconstructed from the 2026-08-22 packets and must be corrected in place after their
first real use. That correction is an edit to a tracked file, so make it **after** Run 4, not
before (a dirty tree is refused at step 2 and again at launch).

**Proof:**
```bash
# run on pve host .121, .155, .163 (root)
echo "NCPROOF-$(hostname)" > /dev/kmsg
```
```bash
# run on VM101
grep -a NCPROOF logs/netconsole_all_rigs.log | tail -5
```
Source: `docs/RUNBOOK_NETCONSOLE_REARM.md:95`, `:103`.

- **Pass:** three **new** lines, timestamped today, from `192.168.3.121`, `.155` and `.163`, naming
  `NCPROOF-pve-rig6600`, `-pve-rig6600b`, `-pve-rig6600c`. The last proof lines in the file are from
  2026-08-22 19:26; those do not count.
- **Fail:** fewer than three new lines. That host's sender is not armed. **Do not start the run**
  (runbook §3). Record the proof timestamps.
- An armed sender does not guarantee a captured event; the 2026-08-22 freeze produced none
  (runbook §5).

## 9. Dataset frozen and present on the fleet — automatic; optional manual check

The run resolves `daily3_current.json` once, freezes it, and verifies the file **on each target
node** before the first dispatch, failing closed (`miner/dataset_authority.py:922`
`fleet_preflight`; fact file §2.10). Measured tonight: the pointer names
`daily3-20260801T145551443433Z-513648160d35.json`, sha256
`513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6`, 18,068 records, the file
attempt 9 ran on. Nothing to do unless you want to look first:

```bash
# run on VM101 — the provisioning manifest's own verification command, on each target
F=/home/michael/distributed_prng_analysis/daily3-20260801T145551443433Z-513648160d35.json
for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
  ssh -n michael@$ip "hostname; stat -c %s -- $F; sha256sum -- $F | cut -d' ' -f1"
done
```
Source: `dataset_provisioning.json:20` (`verification_command`).

- **Pass:** each rig: `1380711` and `513648160d356617c22a1e543ae1c9c65f4921ec21718989308b1f70c00768f6`.
- **Fail:** missing or different. The run would refuse with `DatasetProvisioningError` before
  dispatch. Provisioning is `provision_dataset_fleet.py`; its use is not part of this checklist.

## 10. Survivor-volume gate for Run 4's request — OWNER or SESSION

Run it **before step 13**: the parked fleet puts a worker on VM101's 3080 Ti, and the gate reports
the GPU busy (`UNAVAILABLE`) if any compute process other than `sunshine` is on it
(`config_manifests/survivor_volume.cap`; volume report §3). It is a short GPU job (about 24 s
measured; volume report §3). Nothing in `gate12_launch.sh` runs it, so it is by hand.

```bash
# run on VM101
P=/home/michael/tfm_worktrees/volume_limit/data/survivor_volume_limit/run4_upper_params.json
cat "$P"     # check: [2147483648, 4294967296), window 12, anchor 25, phase 0, ["midday"], 0.71/0.47, skip 6-99, the 513648160d35 dataset
python3 -u scripts/survivor_volume_gate.py --params @"$P" \
  --record-out data/run4/volume_gate_record.json 2>&1 | tee data/run4/volume_gate.log
echo "rc=${PIPESTATUS[0]}"
```
Sources: `scripts/survivor_volume_gate.py:361-362` (arguments); the parameters file named in
`docs/CLAUDE_CODE_REPORT_SURVIVOR_VOLUME_LIMIT.md:331` (it lives in the `volume_limit` worktree, not
in the repository; if that worktree has been removed, the file's content is the `cat` above).

- **Pass:** `SURVIVOR-VOLUME GATE : PROCEED (load verdict) — every phase at or below the cap 25000
  under 'point'`, rc 0. The phase-4 projection should read 24,208 (volume report, same sample each
  rerun). This is a **load** verdict, not a quality score.
- **Fail:** REFUSE, ERROR or UNAVAILABLE, rc 1. Do not launch. UNAVAILABLE usually means the GPU is
  busy; find the process.

## 11. Open Run 4's search — OWNER (this is your act; no session opens or closes a search)

**The `--params` JSON must be exactly the one the launch will use** (step 13). WATCHER compares the
run's configuration with the search record field by field and refuses before dispatch on any
difference (stage-1 S4, report `:299`). Run 4's JSON, as the committed pre-launch check builds it
(printed by `tests/run4_prelaunch_provenance_check.py` tonight; the ten Gate-12 keys of
`gate12_launch.sh:378-383` plus the seven approved pins):

```
{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 2147483648, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 6, "warm_start_skip_max": 99}
```
(`seed_start: 0` is correct: Step 1 takes its start from the search's first gap, 2³¹, "assigned
even at 0", stage-1 S3, report `:298`.)

```bash
# run on VM101
RUN4_PARAMS='{"prng_type": "java_lcg", "seed_start": 0, "max_seeds": 2147483648, "miner_stripe_size": 67108864, "worker_pool_size": 25, "test_both_modes": true, "use_range_miner": true, "use_persistent_workers": false, "window_trials": 1, "n_parallel": 1, "warm_start_window": 12, "warm_start_offset": 25, "warm_start_session_idx": 1, "warm_start_fwd_thresh": 0.71, "warm_start_rev_thresh": 0.47, "warm_start_skip_min": 6, "warm_start_skip_max": 99}'
python3 agents/watcher_agent.py --step1-search-open \
  --search-range-start 2147483648 --search-range-end 4294967296 \
  --search-reason "<your words: Run 4, upper half, pinned attempt-9 geometry, generator_phase 0>" \
  --params "$RUN4_PARAMS" 2>&1 | tee data/run4/search_open.log
```
Source: `agents/watcher_agent.py:3889-3906` (arguments), `:812-878` (`step1_search_cli`); the range is
`docs/PROPOSAL_STEP1_SEARCH_RECORD_AND_RERUN_v1_1.md:290-293`.

- **Pass:** one line `STEP1_SEARCH_OPENED s1s-<64 hex> range=[2147483648, 4294967296)
  dataset=daily3-20260801T145551443433Z-513648160d35 trials_per_chunk=… chunk_seeds=… space=…`.
  Rerunning the same command prints `STEP1_SEARCH_ALREADY_OPEN` with the same id; that is fine.
- **Fail:** `REFUSED: …` (rc 3) or `ERROR: …` (rc 2). Read the reason; do not work around it. If
  the dataset could not be resolved, the search is not opened.
- **Note:** this is the first write of the search tables into the live `prng_analysis.db`. Tonight
  the database has no search tables at all (report, T3).

**Status:**
```bash
# run on VM101
python3 agents/watcher_agent.py --step1-search-status --params "$RUN4_PARAMS"
```
- **Pass** (form from the stage-1 CLI smoke, `…/step1_search_runs/cli_smoke/out.txt`):
  ```
  { "state": "gap", "search_id": "<the id above>", "next_seed_start": 2147483648,
    "range": [2147483648, 4294967296] }
  ```

## 12. Route-A pre-launch check on the committed tree — OWNER or SESSION

```bash
# run on VM101
python3 -u tests/run4_prelaunch_provenance_check.py 2>&1 | tee data/run4/prelaunch_check.log
echo "rc=${PIPESTATUS[0]}"
```
Source: `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md:406-412` item 5;
`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:23`, `:150`.

- **Pass:** `10/10 PASS` and `RUN4_PRELAUNCH_PROVENANCE_OK`, rc 0. The built command contains
  `--seed-start 2147483648 --max-seeds 2147483648`, `--use-range-miner`, `--trials 1`,
  `--step1-search-id s1s-…` and the seven `--warm-start-*` flags at 12 / 25 / 1 / 0.71 / 0.47 / 6 /
  99. Tonight, in the clean worktree at `347b036`, it gave exactly this (report, T1).
- It drives the real CLI seam against a **stub** search and dataset. It does not read your live
  search. The `--lottery-file daily3.json` in its printed command is the stub's; the live run uses
  the frozen immutable path.
- **Fail:** anything but 10/10. Do not launch.

## 13. Launch Run 4 — OWNER

**UNAVAILABLE as a committed command.** What exists:
- the certified harness `gate12_launch.sh`, whose only Run-4-incompatible part is its `--params`
  block (`:377-383`), and
- the Run 4 JSON from step 11, which is the same ten keys and values plus the seven pins.

What is missing: a committed launcher that joins them. Your decision D-1:

**Option A (recommended).** Copy the harness **outside the repository**, change only the
`--params` block, check the diff, and run the copy. The harness does `cd ~/distributed_prng_analysis`
itself (`:79`), and calls its gates by repository-relative path, so it runs from anywhere. A copy
inside the repository would be an untracked file and the clean-tree gate would refuse it.
```bash
# run on VM101
cp -p gate12_launch.sh /home/michael/run4_launch.sh
#   then edit ONLY lines 379-383 of /home/michael/run4_launch.sh so the --params value is the
#   step-11 JSON. Check:
diff gate12_launch.sh /home/michael/run4_launch.sh    # must show only the --params lines
bash /home/michael/run4_launch.sh
```
`gate12_launch.sh:6` gives the invocation (`bash gate12_launch.sh`). The diff should show the old
five-line JSON replaced by the step-11 JSON, nothing else. The editing itself is yours; I have not
written that file.

**Option B.** Commit a Run 4 launcher to the repository (a new tracked file, or a parameter added
to `gate12_launch.sh`). That is a code change with tests that read `gate12_launch.sh`
(`tests/test_gate12_cleantree_admission.py` extracts its rotation region verbatim). It cannot be
done tonight under this brief.

**Not recommended:** running `python3 agents/watcher_agent.py --run-pipeline …` by hand without the
harness. It skips the parked-fleet barrier, the sentinel and liveness gates and the sampler.

**Where everything goes** (from the harness, `gate12_launch.sh:84-93`, `:258`, `:377-379`). With
`STAMP` = launch time `YYYYmmdd_HHMMSS`:

| File | What |
|---|---|
| `logs/gate12_${STAMP}.log` | the WATCHER / Step 1 / coordinator log. WATCHER runs under `nohup` (`:377`) |
| `logs/gate12_${STAMP}_evidence.txt` | every gate's output |
| `logs/gate12_${STAMP}_fleet.log` | fleet dispatch |
| `logs/gate12_${STAMP}_concurrency.tsv`, `_verdict.txt`, `_sampler.log` | the sampler |
| `logs/gate12_${STAMP}_source_digests.json`, `_liveness.json` | parity and liveness evidence |
| `logs/miner_workers/` | the local worker log and release tokens |

The harness ends in `tail -f` of the run log (`:434`). Ctrl-C there is safe: the run and the sampler
keep going (`:426`).

What the harness does, in order: clean-tree admission (`:133`) → GPU gate (`:164`) → parity
(`:192`) → clean slate (`:203-229`) → pre-dispatch clean-tree (`:246`) → fleet launched **parked**
(`:262`) → sentinel 25/25 (`:291`) → liveness (`:332`) → sampler (`:364`) → WATCHER (`:377`) →
release when port 5700 binds (`:403-405`). Any gate's refusal stops it before the coordinator
exists, except that the release step kills the fleet if the coordinator never binds (`:417-423`).
Its evidence block prints "PRE-RUN CERTIFIED CURSOR (must be 0)" (`:105`). That label is from Gate
12; for Run 4 it should print `status='OPEN', next_seed_start=2147483648`. It is a print, not a
check.

## 14. While it runs — OWNER

**Expected survivor counts** (exact, every seed, through the production path; volume report
`:13-18`):

| Phase | Family | Survivors |
|---|---|---:|
| 1 | `java_lcg` | 0 |
| 2 | `java_lcg_reverse` | 0 |
| 3 | `java_lcg_hybrid` | 60 |
| 4 | `java_lcg_hybrid_reverse` | 23,568 |

23,568 is under the cap (25,000) but **53 above 23,515**, the largest phase any completed run has
carried (cap file `evidence`). The completion plan's earlier envelope (96 / 22,656,
`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:81`) was a projection; the exact counts replace it.
These counts are load, not hit rate. Run 4 measures no hit rate.

**Memory (`CLAUDE.md` rule 9).** VM101 has 15.5 GiB and no swap. The Run 4 proposal projects the
coordinator's peak inbound queue at about 2.8 MiB against `MemTotal` 15,924.8 MiB
(`docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md:65`). Tonight VM101 had about 12.3 GiB available
(`free -m`). The record gives no coordinator peak-RSS figure for this geometry. Do not change
VM101's memory before Run 4 (`docs/BACKLOG.md:521-522`).

**Timing guide — attempt 9** (`logs/gate12_20260817_181819.log`, `[S172-BP] derived_bound stage`
lines, which mark each phase's start):

| Event | Time | From previous |
|---|---|---|
| script start (stamp) | 18:18:19 | — |
| phase 1 starts | 18:20:04 | — |
| phase 2 starts | 18:21:28 | 84 s |
| phase 3 starts | 18:23:06 | 98 s |
| phase 4 starts | 18:27:46 | 280 s |
| `[S172-BP] summary` | 18:32:19 | 273 s |
| `Step 1 PASSED` | 18:32:29 | — |

About 14 minutes end to end. The 2026-08-22 run's serve loop took 713.7 s
(`docs/SESSION_CHANGELOG_20260822_BRIEF_I_PRODUCTION_SHAPE_REPAIR.md:28`). Run 4's seeds are a
different interval, so treat these as a guide only.

**Lines that are normal:** `[S172-SL] window`, `[S172-SL] slow_msg`, `[H1H2] active_stripes`,
`[S172-BP] derived_bound stage phase=N`, `[S172-BP] summary`, `Step 1 PASSED`, and
`Triggering Step 2: Scorer Meta-Optimizer`. **That last one is expected and harmless**: with
`--end-step 1` Step 2 never runs (fact file §2.49, §2.54).

**Lines that mean trouble** (fact file §2.25-§2.49; the code's own terminal prefixes):

| Line | Meaning |
|---|---|
| `STEP 2: Scorer Meta-Optimizer (run #` | **the abort signal**: Step 2 is actually executing. Stop the run |
| `[F1/F2] TRIAL TERMINAL … class=…` | the trial has ended in failure; the class names why |
| `compute_lease_expiry` | a stripe's lease ran out (attempts 1, 4, 6-8) |
| `worker_admission_timeout` | fewer than 25 workers at a stage boundary (attempts 2, 5) |
| `coordinator_staging_capacity_timeout:` / `_invariant:` / `TC_INBOUND_SATURATION_TIMEOUT` | staging or ingress capacity terminal |
| `WORKER_DISCONNECTED` | a worker lost its session. Not a failure by itself: `WORKER_RECONNECTED` for the same `worker_id` is the certified recovery (fact file §2.33). A constant-phase loss is terminal |
| `MinerIngressError … validated=False` | **a symptom.** Look for the first terminal line before it |
| `[STEP1-SEARCH S7] miner sweep evidence UNAVAILABLE` / `Step1NoCreditError` | compute finished but earned no credit. See the note below |
| `RunParameterError: repository_tree_clean is False` | publication refused a dirty tree (attempt 3) |

**NEW, operator note — the ledger `-wal` at evidence time (untested at fleet shape).** Step 1 reads
the miner ledger's stripe rows to credit the sweep, and refuses credit if
`~/miner_staging/miner_ledger.db-wal` is non-empty at that moment
(`window_optimizer_integration_final.py:1443-1456`). The concurrency sampler opens the ledger
`mode=ro` every 2 s (`scripts/gate12_concurrency_sampler.py:203`; `gate12_launch.sh:364-366`). I checked
SQLite's behaviour tonight on a throwaway file: when the last writer closes while a `mode=ro` reader
is open, `-wal` stays non-empty (206,032 bytes) after the reader closes too, until another writer
opens and closes. Whether that timing happens in Run 4 is **not known**. If it does, the result is
fail-closed: no credit, cursor unmoved, nothing lost. **During the run, do not open the live ledger
with any other tool.** If S7 reports this line, stop and bring it back; do not edit the ledger.

**How to stop it.** There is no recorded clean mid-run stop.
- The halt file **does not stop a running Step 1**. WATCHER checks it only before starting and
  between steps (`agents/watcher_agent.py:3068`, `:3095`). With `--end-step 1` there is no "between".
- The recorded hard stop is the harness's own cleanup lines, `gate12_launch.sh:203-206`:
  ```bash
  # run on VM101
  pkill -f "[w]atcher_agent"; pkill -f "[w]indow_optimizer"; pkill -f "[r]ange_miner_worker"
  for ip in 192.168.3.122 192.168.3.156 192.168.3.164; do
    ssh -n michael@$ip 'pkill -f "[r]ange_miner_worker"'
  done
  ```
  The PIDs are also printed (`watcher pid=` and `concurrency sampler pid=` in the evidence file),
  which is the form `CLAUDE.md` rule 8 prefers.
- A coordinator that dies mid-trial is **interrupted, not resumable** (fact file §2.24). Nothing is
  credited and the search's gap stays at 2³¹.

## 15. After it finishes — OWNER, or SESSION at your word

**Look first, from the run log:** `Step 1 PASSED`, a `[S172-BP] summary` line for the run, and the
run id `distributed_config_t1_<8 hex>` (it is in the `[S172-BP] summary run=` field and the
`[H1H2]` lines).

**Verifier `G-PROD-SHAPE`:**
```bash
# run on VM101
PYTHONPATH=. python3 -u tests/gate_s172_prod_shape.py \
  --log logs/gate12_<STAMP>.log --run-id distributed_config_t1_<id> 2>&1 \
  | tee data/run4/gate_prod_shape.log
echo "rc=${PIPESTATUS[0]}"
```
Source: `tests/gate_s172_prod_shape.py:32-35` (usage), `:455-464`; completion plan A8. The staging
directory and ledger default to the manifest's `staging_dir` (`/home/michael/miner_staging`).

- **Pass:** every check PASS. Without `--run-id` the verdict is UNAVAILABLE by design (`:37-41`).
- Its section B is UNAVAILABLE, not judged, if the ledger `-wal` is non-empty (`:39-41`). See the
  note in step 14.
- Report it as PASS only if it says PASS. It has never passed on a production run (it ran
  2026-08-22: 15 pass / 6 fail; completion plan §0).

**What the search and history should then report:**

```bash
# run on VM101
python3 agents/watcher_agent.py --step1-search-status --params "$RUN4_PARAMS"
python3 -c "
from database_system import DistributedPRNGDatabase
d=DistributedPRNGDatabase()
print('cursor:', d.get_certified_cursor('java_lcg', test_both_modes=True))
"
```
Sources: step 11; `gate12_launch.sh:106-110`.

- **Search:** `"state": "search_complete"` for the Run 4 search (`utils/step1_search_ledger.py:73`).
- **History:** `status='COMPLETE'`, `next_seed_start=None`, `covered_seed_count=4294967296`,
  `certified_interval_count=2` (attempt 9's `[0, 2³¹)` plus Run 4's `[2³¹, 2³²)`). Tonight it reads
  one interval, `[0, 2147483648)`, `coverage_id c6f28aedf7af12cd…`.
- **If not credited:** the search stays `gap` at 2³¹, and history stays as tonight.

**What to save.** The `logs/gate12_<STAMP>*` files, `logs/miner_workers/`, `data/run4/`, and the
published generation directory under `.s172_accumulator/generations/`. Freeze them with a
`SHA256SUMS` the way earlier runs were frozen (fact file §2.34-§2.38 name the bundles, e.g.
`/home/michael/gate12_attempt5_20260812_190715`). The exact freezing command is not recorded in the
repository: **UNAVAILABLE**. Copying is yours.

**After the run:** the tree is still clean (the run's outputs are ignored: `optuna_studies/`,
`optimal_window_config.json`, `logs/`, `.s172_accumulator/`, the root NPZ links; `git check-ignore`
tonight). Then: the closure report and Brief I acceptance (completion plan A8); netconsole runbook
correction (step 8); `docs/LEADS.md` L-4 and L-5 follow-ups (this is their first fleet run).

---

## If something goes wrong

| Situation | What it looks like | What to do | Stop for it? |
|---|---|---|---|
| **A gate refuses** (steps 2-12, or inside the harness) | REFUSE / UNAVAILABLE / ERROR, rc ≠ 0 | Read the reason. A refusal is a result. No downsizing, no bypass, no allowlist change | **Yes**, until the cause is understood |
| **Halt file** | `/tmp/agent_halt` exists; WATCHER prints `PIPELINE HALTED - Cannot start` | The harness passes `--clear-halt` (`:377`). By hand: `python3 -m agents.watcher_agent --clear-halt` (the form WATCHER prints, `agents/watcher_agent.py:3079`). Absent tonight. It cannot stop a running step | Only if you did not create it — then find out who did |
| **Hung run** | the log stops moving; `[S172-SL] window` lines stop; the sampler TSV stops growing | `serve_timeout` has no clock by design; admission (180 s), staging capacity (600 s) and lease (300 s) do. A real hang has none of their lines. Check with `ps` that the processes are alive and burning CPU before deciding. Never pipe to `tail` to check | **Yes** — capture state, then the hard stop in step 14 |
| **A rig drops** | `WORKER_DISCONNECTED` for its eight workers; it stops answering | Constant phases (1, 2): the trial ends. Hybrid (3, 4): one retry on another worker, then terminal. **Do not reboot the host**: it would come back as Ubuntu and needs `boot-proxmox` (fact file §6). Rule out "booted into Ubuntu" before calling it a hang | **Yes.** The run is over or ending; one terminal event ends the session (delegation `:83`) |
| **Credit refused** | `Step1NoCreditError`, S7 line | Fail-closed; nothing lost; cursor unmoved | **Yes** — bring the line back |
| **Retrying after a failure** | — | **[UNVERIFIED]**: stage 1 runs one chunk per search; a failed trial consumes the budget, and a second try on the same search likely refuses before dispatch (stage-1 report `:23-26`; proposal §2.7). The recorded route is close the search and open a new one with `--search-repeat-of <old id>` (report `:410`). Not executed tonight | Your decision |
