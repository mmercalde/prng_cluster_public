# PWC full backup — Stage 2 (v1), uncommitted

Worktree `/home/michael/tfm_worktrees/pwc-backup`, HEAD = main `ee39404` (moved there with `git reset ee39404` on the
owner's instruction; nothing committed, nothing pushed). CPU only. Rigs off, no GPU, no live ledger, no network change.

## Prior record

**Searches run** (2026-11-20, on `ee39404`, `/bin/grep` because the shell `grep` skips `*.json`). Scope `S` =
`docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_*.md docs/TB_RULING_REQUEST_*.md docs/PROPOSAL_*.md docs/CHAPTER_*.md
docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md`, plus `/bin/grep -rl` over `docs/exchange/`:

```
/bin/grep -lE "PWC.*[Ss]tage 2|pwc_backup_stage2|STAGE2"                       $S ; docs/exchange/   -> no hit
/bin/grep -lE "balanced_hybrid|S147 Q2|Q2.*single strategy"                    (same)               -> no hit in S
/bin/grep -lE "forward_zero|S147.*prune|B1 gate"                              (same)               -> no hit in S
/bin/grep -lE "HSA_ENABLE_DEBUG_TRAP|HSA_OVERRIDE_GFX"                         (same)
/bin/grep -lE "window_anchor.*PWC|PWC.*window_anchor|PWC.*generator_phase"     (same)
/bin/grep -lE "VERIFICATION_INTEGRITY|VIR-5"                                   (same)
/bin/grep -lE "launch token|RUN_NONCE|run token"                               (same)
/bin/grep -rlE "S147" docs/ ; /bin/grep -rnE "balanced_hybrid" docs/*.md      (wider, for the S147 records)
```

**Documents that decide, authorise or record this subject (read):**

| Document | What it settles |
|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md` (main `ee39404`) | Owner rulings 1–4 and the default rule "PWC does what RANGE-MINER does". Treated as fixed facts |
| `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md` Stage 2 row | This stage. Owner's go: 2026-11-20 06:32 PST |
| `/home/michael/tfm_worktrees/rig-gpu-fault/docs/drafts/BRIEF_PWC_BACKUP_STAGE2_STAGE3_20261119.md` (not on main; read only) | The Stage 2 change list (2.1–2.15) and F1–F6. Checked line by line below |
| `docs/exchange/from_beta/X-PWC-STAGE1_R1_BETA_REVIEW.md` Q3, Q4, findings 1–3 | The three Beta items carried into Stage 2, and the missing citation |
| `docs/VERIFICATION_INTEGRITY_STANDARD.md` VIR-1 `:30-35`, VIR-2, VIR-5 `:93-117`; `docs/SESSION_CHANGELOG_20260814_S182.md:122-126` | **The citation Stage 1's prior record omitted (Beta finding 1).** A check must prove it ran and could observe; `pgrep -c … \|\| echo 0` is the S182 double zero. Stage 2's process checks are built to these |
| `docs/STEP1_EXECUTION_FLOW_AND_PRUNING_S147.md:176`; `docs/TODO_MASTER_S148.md:132` (`667ece5`); `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md:449` | S147 Q2 (PWC hybrid uses only `balanced_hybrid`) and the S147 B1/Q0 prune |
| `miner/range_miner_worker.py:22-29`, `:601-605`; `miner/range_miner_coordinator.py:9305-9312`, `:9356-9361`, `:12046-12062` | What RANGE-MINER does: anchor/phase split with phase pinned 0; one threshold written as both keys; `workflow_stages_for` |
| `docs/TODO_CONSOLIDATED.md:303` | S147 Q0/Q2 follow-ups recorded as PWC-only |
| `docs/TB_RULING_REQUEST_ROCR_ISOLATION_S149.md`, `docs/CHAPTER_9_*`, `docs/exchange/to_beta/X-RIG-GPU-FAULT_R1_*` | HSA variable history (F2); not changed |
| **Added in round 2 (Beta X-PWC-STAGE2 R1 finding 1, which said they were missing):** | |
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:139-153` | After a commit: confirm a clean tree and rerun the commit-sensitive clean-tree and Gate 22 checks without widening allowlists |
| `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120` | Gate 22 red on a dirty tree is "commit-state transitional evidence"; do not widen its allowlist |
| `docs/PROPOSAL_PWC_LIFECYCLE_FIX_S156_v2_0.md:65-77` | S156 is a temporary bandaid: targeted pattern, SIGTERM first, log what was found and reaped |
| `docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:12-21` | Binding: `window_anchor` = which records form the window; `generator_phase` = generator advances; phase = 0 in v1; never reconstructed from each other |
| `docs/ROCm_Saturation_Report_S172.md:25-42`, `:68-87` | The empirical pool × chunk map behind `PWC_STABLE_ENVELOPE` (Q3) |

**Labels:**

| Item | Label |
|---|---|
| Quarantine lifted, threshold contract | ALREADY RECORDED — owner ruling 1 |
| Prune and Q0 removed | ALREADY RECORDED — owner ruling 4 (supersedes S147 B1/Q0 for PWC) |
| All five strategies | **CHANGES A RULING** (S147 Q2), under the owner's default rule — needs his confirmation (F5) |
| Offset ≠ 0 refused | ALREADY RECORDED — owner ruling 2026-11-20 08:30 PST: offset = draw window only, generator phase 0, PWC moves to RANGE-MINER's method, refuse ≠ 0 until then |
| Beta-2/3/4 fixes | ALREADY RECORDED — Beta X-PWC-STAGE1 R1 Q3, Q4, findings 2–3 |
| Window-anchor split held; worker drops unknown job keys | **NEW** (the brief missed it; see "Where the brief is wrong") |
| Shared suites edited on PWC lines | **NEW** — owner allowed it 2026-11-20 |
| Skip range (F4) left as is | ALREADY RECORDED — owner's open question (`docs/TFM_SESSION_HANDOFF_20261114.md:76-80`) |

**Not searched:** ser8 archives, the private repository, claude.ai project files, the rigs (powered off; deployed bytes
unknown), Proxmox.

Beta review request (sent 2026-11-20 on the owner's approval, one round): `docs/exchange/to_beta/X-PWC-STAGE2_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md`.

## In plain words

- **The variable-skip block is lifted (ruling 1).** In its place, every PWC job carries the trial's own threshold as
  both of its threshold fields, as RANGE-MINER does. A variable-skip pass is checked after it runs against what the
  worker says it used, and it fails if that disagrees or is missing. It refuses before it starts if it has no
  threshold, or if it would run on the old SSH route, which cannot report one.
- **PWC no longer stops early (ruling 4).** Phases 1–2 always run, and phases 3–4 whenever both modes are on,
  whatever the survivor counts. That is RANGE-MINER's rule.
- **All five search strategies (default rule).** PWC used to send only `balanced_hybrid` to variable-skip passes. It
  now sends the same five RANGE-MINER uses. That is about five times the work on phases 3–4.
- **PWC refuses any window offset other than 0 (owner's choice, 2026-11-20).** PWC sends one number that does two
  jobs: it picks the window of draws, and it advances the generator in the kernel. RANGE-MINER keeps the second at
  0. Separating them needs `sieve_filter.py` and the TCP worker, so that change is held. Until it lands, PWC refuses
  instead of quietly sieving something RANGE-MINER would not.
- **Beta's three Stage 1 items are fixed:**
  - a broken `pgrep` can no longer report "clean";
  - a revoked worker loses its open connection;
  - a refused worker exits instead of retrying every 5 seconds forever.
  
  The last one is the only edit to the worker, about 10 lines.
- **RANGE-MINER is untouched.**
  - Not one of the ten parity-pinned files changed.
  - No miner, launch, manifest, WATCHER or `sieve_filter.py` file changed.
  - The two shared suites changed only on their PWC lines; an AST comparison proves it.
- **Round 2** (owner decision 2026-11-20 10:47 PST) fixed the three items from Beta's R1 review:
  - Q6, the reconnect race: an old connection's cleanup can no longer touch a reconnected worker;
  - finding 2, a late result taken as the next phase's: job ids are unique per pass, and only the active lease's
    result is accepted;
  - Q7: the refusal exit writes no GPU heartbeat.
- **Round 3** (owner decision 2026-11-20 12:34 PST, no third Beta round): Beta's own fix for its R2-Q1 timeout-eviction
  race, and its test suggestion. See "Round 3" and "Files to land" below.

  See "Round 2" below. **Q3 (five strategies on the rigs) stays NEEDS EVIDENCE** for the owner's Stage 4 rig
  sessions.
- **Nothing is committed.**

## The ten rig parity-pinned files (`scripts/gate12_parity_gate.py:174-185`)

None of them is touched. sha256 prefix of the worktree file vs `git show ee39404:<file>`:

| File | worktree | ee39404 | |
|---|---|---|---|
| `adaptive_thresholds.py` | `cda23bb3cd62` | `cda23bb3cd62` | unchanged |
| `execution_set.py` | `d21614701c31` | `d21614701c31` | unchanged |
| `hybrid_strategy.py` | `8b982fbb8ecb` | `8b982fbb8ecb` | unchanged |
| `prng_registry.py` | `32435fb82737` | `32435fb82737` | unchanged |
| `sieve_gpu_worker.py` | `9ace0d83ad99` | `9ace0d83ad99` | unchanged |
| `miner/__init__.py` | `699e930e379d` | `699e930e379d` | unchanged |
| `miner/dataset_authority.py` | `365c8e3ee9ab` | `365c8e3ee9ab` | unchanged |
| `miner/range_miner_coordinator.py` | `b8e259b3e12b` | `b8e259b3e12b` | unchanged |
| `miner/range_miner_protocol.py` | `7b7f8e197914` | `7b7f8e197914` | unchanged |
| `miner/range_miner_worker.py` | `2bca2a083262` | `2bca2a083262` | unchanged |

Every other RANGE-MINER, launch and shared file is unchanged too: `git diff --stat ee39404 -- sieve_filter.py
window_optimizer.py window_optimizer_integration_final.py gate12_launch.sh scripts/launch_fleet_manual.sh
agents/watcher_agent.py agent_manifests/ miner/ distributed_config.json scripts/gate12_parity_gate.py
scripts/pwc_parity_gate.py utils/ coordinator.py` prints nothing, and none has an untracked file.

**`sieve_filter.py` is not edited** (sha256 `b44454da3239d9e1…` in both the worktree and `ee39404`). Stage 2 did not
need it, because the owner chose to refuse window offset ≠ 0 rather than separate the window from the generator phase.
The AST and import proof you asked for applies only if it is touched, so none is given. The file is byte-identical.

## Every changed file

`git diff --stat ee39404` (plus one new file):

| File | Kind | Lines |
|---|---|---|
| `persistent_worker_coordinator.py` | PWC coordinator | 566 changed |
| `persistent/pwc_transport_tcp.py` | PWC transport | 177 |
| `persistent/pwc_worker_service.py` | **PWC worker — Beta Q4 (R1) and Q7 (R2) only** | 14 |
| `test_persistent_worker_harness.py` | PWC suite | 38 |
| `tests/test_pwc_backup_step1.py` | PWC suite | 45 |
| `tests/test_pwc_stage1.py` | PWC suite | 51 |
| `tests/test_pwc_backup_stage2.py` | **new** PWC suite | 1405 |
| `tests/test_s172_threshold_propagation.py` | **shared** suite, PWC gate + mutant only | 225 |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | **shared** suite, PWC drive + one PWC branch only | 16 |

None of these is one of the ten pinned files, and none is under `miner/`.

**The shared suites changed on PWC lines only** (owner's allowance, 2026-11-20). `logs/pwc_stage2_v1/shared_suite_ast_proof.py`
compares every top-level statement with `ee39404`:
- `test_s172_threshold_propagation.py`: 45 identical. The changed, removed and added statements are all G-PWC-HYBRID,
  M3 and their header text:
  - changed: `_drive_pwc_run_sieve_pass`, `_drive_pwc_both_modes_trial`, `gate_pwc_hybrid`,
    `mutant_m3_pwc_quarantine`, `MUTANTS` (the M3 label), the docstring;
  - removed: `ORACLE_QUARANTINE_CODE`, `_detect_quarantine`;
  - added: two detectors.
  
  G-ROUTE-A/B, G-KERNEL, G-MINER-UNCHANGED, M1 and M2 are identical.
- `test_s172_phase5_d3_25_candidate_ingress.py`: 69 identical. Only `drive_pwc` (offset 0, `pwc_transport="tcp"`)
  and one new PWC-only branch in `g1_producer_contract_shape` (`:382-389`) changed. The ZMQ "pruned" assertions are
  unchanged.

## What changed, with `file:line` and the ruling each carries out

### Ruling 1 — lift the block; the threshold contract instead (`persistent_worker_coordinator.py`)

- `:263-286` — the quarantine section (`PWC_HYBRID_QUARANTINE_CODE`, `PwcHybridThresholdContractUncertified`,
  `assert_pwc_hybrid_not_quarantined`) is removed and replaced by a note saying what replaced it.
- `:1899` — `run_sieve_pass(phase2_threshold=None)`: there is no 0.5 default any more.
- `:1930-1949` — a hybrid pass refuses before anything is dispatched:
  - when it has no `phase2_threshold` (`:1935`);
  - when `phase2_threshold != threshold` (`:1940`, RANGE-MINER pins the two equal,
    `miner/range_miner_coordinator.py:9356-9361`);
  - on the SSH route (`:1945`).
  
  A constant pass that omits it sends `threshold` (`:1949`).
- `:2111` — each chunk records the `phase2_threshold` its job carried.
- `:548-590` — `pwc_hybrid_threshold_record` / `assert_pwc_hybrid_threshold_contract`. Requested and payload are
  compared exactly, and the worker's report at 4 places, because `sieve_filter.py:597-599` rounds what it reports. A
  completed chunk that reports nothing fails the contract. Wired at `:2360-2366`.
- `:711-724` — `_pwc_hybrid_echo` reads `per_family[family].single_phase.threshold` from the TCP result. Used at `:1814`.
- `:2349` — the pass accounting records `transport` (`tcp`/`ssh`), so the Stage 3 credit builder can refuse a pass
  that did not run on TCP.
- `:2546-2553` — `run_trial_persistent` refuses variable-skip passes on a non-TCP transport before the fleet is built.
- `:2619`, `:2646`, `:2692`, `:2717` — each of the four passes is handed its directional threshold as `phase2_threshold`.
- The hybrid skip-bound question (the kernels' hardcoded `expected_skip`) is shared and still open
  (`docs/HYBRID_SKIP_BOUND_AUDIT.md` §0). It is the same in RANGE-MINER, so it no longer blocks PWC under ruling 1.

### Ruling 4 — phases follow RANGE-MINER

- `:2547` — `_run_variable = test_both_modes and not prng_base.endswith("_hybrid")`, the same test as
  `_step1_required_phases`, and `workflow_stages_for` (`miner/range_miner_coordinator.py:12046-12062`).
- `:2627-2629` — the S147 B1 forward-zero prune and its early `pruned=True` return are removed.
- `:2700-2702` — the S147 Q0 gate is removed: phase 4 always runs when phase 3 does.
- A PWC trial now always returns `pruned=False` (`:2766`).

### Default rule — strategies

- `:2669-2717` — the S147 Q2 single `balanced_hybrid` block is removed, and both hybrid calls pass `strategies=None`.
  `run_sieve_pass` then loads all five from `hybrid_strategy.get_all_strategies()` (`:1970`), the source the miner
  worker uses (`miner/range_miner_worker.py` `_load_strategies`).
- `:1977-1982` — an `ImportError` there now refuses. It used to send `[]` and let `sieve_filter` pick its own set.

### Owner ruling 2026-11-20 08:30 PST — window offset ≠ 0 refused until PWC moves to RANGE-MINER's method

- `:326-358` — `PwcWindowAnchorRefused` and `validate_pwc_window_offset`.
- `:2543` — called in `run_trial_persistent` before the fleet is built.
- `:1952` — called in `run_sieve_pass` before dispatch.

### Beta X-PWC-STAGE1 finding 2 / Q3 — a failed process check never reports clean (VIR-1, VIR-5, S182)

- `:187-254` — the positive control `PWC_PROBE_CONTROL_TOKEN` (`:202`) and `_pwc_probe_control` (`:208`):
  - the shell running the string carries the token in its own command line, so `pgrep -f` of it must find that
    shell, or the check exits 7 (`PWC_KILL_EXIT_UNAVAILABLE`);
  - after the control, pgrep's status is read exactly: 1 = none left, 0 = survivor (exit 6), anything else = exit 7.
- `pwc_stale_kill_command(patterns)` (`:214`) is the rig kill. `pwc_s156_find_command` (`:233`) prints `none`, the
  matches, or `probe_unavailable`. `pwc_s156_reap_command` (`:244`) prints `reaped:<n>` or `reaped:unavailable`; the
  `|| echo 0` double zero is gone.
- `:1036-1077` — the S156 find and reap use them, and an unavailable probe is logged as an ERROR, never as "no stale
  workers". The SSH route still continues after it, as before; Beta did not ask for a refusal there.
- `:1349-1357` — the rig kill's log names "survived" or "probe UNAVAILABLE".
- `:1281-1307`, `:1475-1476` — the **VM101 local kill**, which used to ignore its result and always log success:
  - it now runs the same probe-checked string, under `bash -c`, with only `[p]wc_worker_service`
    (`PWC_LOCAL_STALE_KILL_PATTERNS`, `:230`);
  - unless it is confirmed, no local worker is launched or admitted this run.

### Beta finding 3 / Q3 — revocation, counting, reconnects (`persistent/pwc_transport_tcp.py`, coordinator)

- `pwc_transport_tcp.py:192-215` — `revoke_worker_id` now also drops the id from the online and ready sets and shuts
  down its live socket. It uses `shutdown`, not `close`, because `close` does not wake the handler blocked in `recv`;
  this was found in development.
- `:516-517`, `:523-525` — a `ready` or `request_job` from an id that is off the list ends the connection; no job is
  given.
- `:654-672` — only the connection that still owns an id pops, reclaims or un-readies under it. A replaced
  connection reclaims only the jobs it carried itself, because leases now record their connection (`:339-349`,
  `:359`, `:665`).
- `persistent_worker_coordinator.py:1440-1449` — a launch with no `PID=` line is revoked, logged and **not counted**.
  It no longer sizes the online wait or the rig's effective pool.

### Beta Q4 — a refused worker exits (`persistent/pwc_worker_service.py`, the only worker edit)

- `:44` — `PWC_EXIT_HELLO_REJECTED = 43`.
- `:358-367` — on `accepted: False` the worker logs, writes a `shutdown` heartbeat, closes and calls `sys.exit(43)`.
  It used to raise `ConnectionError`, which `run_forever` retried every 5 s forever.
- Nothing else in the worker changed. The `.127` fallback (Stage 1's held hunk), the HSA lines and the job-key list
  are untouched.

### PWC test changes

- `tests/test_pwc_backup_step1.py:353-378` and `tests/test_pwc_stage1.py:79-90` — the "base" these suites compare with
  is pinned to `27eb991`. They read `HEAD`, which stopped being that base when Stage 1 landed. **Both suites fail on
  clean `ee39404` for this reason alone**; see the before/after table.
- `tests/test_pwc_backup_step1.py:371-379` — the S2 byte-equality gate excludes `phase2_threshold` only. Stage 2
  changes it on purpose: a constant job's unused 0.5 becomes the trial threshold.
- `tests/test_pwc_backup_step1.py:588-597` — the G-S6 quarantine gate becomes "hybrid without a threshold refuses".
- `tests/test_pwc_stage1.py`:
  - `:227` — the remote string is read with `shlex.split`, because it is now `shlex.quote`d.
  - `:455-460` — the A3 refused-duplicate gate replaces the list rather than calling `revoke_worker_id`, since revoke now
    also drops the live connection; that behaviour is tested in the Stage 2 suite.
  - `:531-545` — mutant M-A3d now removes both guards. Stage 2's owner check alone masks the old single-guard mutant,
    which makes that an invalid experiment under VIR-2, not a survivor.
- `test_persistent_worker_harness.py`:
  - `:500-533` — T13/T14 assert the new refusals: no threshold, and the SSH route.
  - `:539`, `:593`, `:651` — fixtures use offset 0.
  - `:573`, `:627` — `pwc_transport="tcp"`.
  - `:576-584` — T15 asserts no prune: all four passes run.
- `tests/test_pwc_backup_stage2.py` (new): 18 gates, 21 mutants. The gates are listed in its docstring; the mutants
  follow the four-part kill rule (VIR-2).

## F1–F6 (the brief's small questions): the option taken

| Item | Taken | Why / what is left for the owner |
|---|---|---|
| F1 SSH for non-counted runs | **SSH kept for constant-only diagnostic runs, refused for every variable-skip pass**; each pass records its transport for Stage 3 | RANGE-MINER has one transport. Refusing SSH outright is a bigger change than this stage needs, and "counted" does not exist in PWC until Stage 3. **Owner question:** remove SSH entirely in Stage 3, or keep it as a diagnostic? |
| F2 HSA variables | **Not changed** | Cannot be settled by "match RANGE-MINER" inside the allowed files. The worker sets two of the three itself (`persistent/pwc_worker_service.py:300-301`), and so does `sieve_filter.py:27-29` on `rig-6600*` hostnames. Dropping them from the coordinator alone would remove only `HSA_ENABLE_DEBUG_TRAP=0`, the S165 crash fix. **Owner question** |
| F3 cache record | **Not done** | PWC still resets the cache per run (the RANGE-MINER default); `keep` was not added. **Owner question:** should counted PWC runs write v7-schema cache records? That is Stage 3 at the earliest |
| F4 skip range | **Not changed**, by order | The owner's open skip-range question (`docs/TFM_SESSION_HANDOFF_20261114.md:76-80`). PWC still sends the sampled `[skip_min, skip_max]` on phases 1–2; RANGE-MINER sieves `[0,16]` there. **The two engines differ on phases 1–2 whenever the sampled range is not `[0,16]`** |
| F5 strategies | **All five** (match RANGE-MINER) | Supersedes S147 Q2. **Owner to confirm** (about 5× the work on phases 3–4) |
| F6 S147 prune behind a flag | **Deleted, no flag** (match RANGE-MINER, which has none) | **Owner question:** whether a non-default flag is wanted |

## Where the brief is wrong or stale (checked against `ee39404`)

1. **§D and 2.8 are wrong:** they say the anchor/phase split needs only `sieve_filter.py`. The TCP worker builds the
   sieve job from a fixed list of keys (`persistent/pwc_worker_service.py:541-561` (`:535-555` on `ee39404`)), so `window_anchor` and
   `generator_phase` would be dropped on the rig. The split needs a worker edit as well, which is why it is held.
2. **2.11 is done:** the `skip_pattern` parser landed in Stage 1 (B6, `persistent_worker_coordinator.py` TCP legacy
   parser).
3. **2.13 is half done.** Level 1 (the launch set) landed in Stage 1. Level 2 (a per-run token) needs the worker to
   send the token, which is a worker edit and not one of Beta's three items. It is **held**.
4. **2.12 is not needed for "match RANGE-MINER".** PWC already resets per run; a `keep` option would add a way to
   differ.
5. **Line anchors in the coordinator are from `27eb991`** and all moved with Stage 1. Examples:
   - the quarantine is `:233-275` on `ee39404`, not `:179-239`;
   - the `run_sieve_pass` guard is `:1773`, not `:1279`;
   - the phase2 default is `:1742`, not `:1243-1256`;
   - the prune is `:2427`; Q2 is `:2492-2499`; Q0 is `:2524`;
   - the probe's HSA exports are `:795-797`, not `:406-408`.

   The brief warned about this.
6. Small offsets that the brief has nearly right:
   - `sieve_filter.py` rounds the echo at `:597-599` (brief: `:599-602`);
   - `run_hybrid_sieve`'s argument list is `:426-458` (brief: `:450-461`);
   - the miner threshold resolution is at `range_miner_coordinator.py:9305-9312` (brief: `:9303-9310`).
7. **Confirmed true:**
   - the forward `java_lcg_hybrid` kernel takes 15 arguments (`prng_registry.py:1007-1012`), while `sieve_filter`
     sends 16;
   - no `HSA_*` in `miner/`, `scripts/launch_fleet_manual.sh` or `gate12_launch.sh`;
   - `GENERATOR_PHASE_V1_PIN = 0` (`miner/range_miner_worker.py:605`);
   - `workflow_stages_for` is at `:12046-12062`;
   - the integration imports `sieve_filter` at `window_optimizer_integration_final.py:52-55`.
8. **Not checked:** the brief's claim (2.10) that the SSH route "has the S153 reverse-argument mismatch". A search of
   `sieve_gpu_worker.py` and the coordinator found no S153 record. **[UNVERIFIED]**

## Tests

Conditions for every run:
- worktree only, sequential, `~/venvs/torch`, `CUDA/HIP/ROCR_VISIBLE_DEVICES=""`;
- `ssh`/`scp`/`rsync` replaced on `PATH` by a stub that refuses and logs (`logs/pwc_stage2_v1/fakebin/`);
- every executed `pkill`/`pgrep` runs inside an unprivileged PID namespace;
- `timeout 1200` per suite.

Runner: `logs/pwc_stage2_v1/run_suites.sh`. Suites: `logs/pwc_stage2_v1/suites.txt` = Stage 1's 73 **minus
`tests/test_s172_staging_partb.py`** **plus** the new `tests/test_pwc_backup_stage2.py`.

**`tests/test_s172_staging_partb.py` was not run, before or after.** It writes and renames a probe file inside the live
`~/miner_staging` (`tests/test_s172_staging_partb.py:599-609`, through `validate_coordinator_staging_dir`,
`miner/range_miner_coordinator.py:12190-12205`, on the manifest's `staging_dir`). The folder could not be pointed
elsewhere without editing a miner test or the manifest. The runner records `~/miner_staging`'s mtime after **every**
suite.

**"Before" was run fresh on clean `ee39404`, not reused from Stage 1.** Stage 1's "after" run recorded no per-file
hashes, and `tests/test_pwc_stage1.py` changed after it ran, so tree identity could not be shown. Each run now writes
`code_sha256.txt` (every tracked and untracked `.py`/`.sh`/`.json`).

**Result (final pass `after_r3`, the package as it lands): 68 of 72 existing suites are identical before and after (same exit code and the same set of failing checks); 4 differ, all explained below; 1 new suite passes. The two earlier after passes (`after`, before the round-2 fixes, and `after_r2`) gave the same 68 / 4 split, with the same four suites.**

| Suite | rc before | rc after | failing checks before | after | | Note |
|---|---|---|---|---|---|---|
| `tests/test_beta_relay.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_cache_policy_v7.py` | 1 | 1 | 0 | 0 | same |  |
| `tests/test_chapter1_p0_corrections.py` | 1 | 1 | 6 | 6 | same |  |
| `tests/test_chapter2_content_gate.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_env_freeze_gate.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_gate12_cleantree_admission.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_gate12_concurrency_sampler.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_gate12_gpu_gate.py` | 0 | 0 | 1 | 1 | same |  |
| `tests/test_gate_prod_shape_scoping.py` | 0 | 0 | 15 | 15 | same |  |
| `tests/test_miner_no_torch.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_owner_land.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_preflight_gpu_probe.py` | 0 | 0 | 1 | 1 | same |  |
| `tests/test_prng_encoding.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_pwc_cache_reset.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_rig_fault_map.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_rig_fault_recorder.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_rig_seam_probe_controller.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_run_authorization_gate.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_admission_binding.py` | 1 | 1 | 9 | 9 | same |  |
| `tests/test_s172_admission_liveness.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_admission_stall.py` | 124 | 124 | 0 | 0 | same |  |
| `tests/test_s172_attempt6_remediation.py` | 1 | 1 | 3 | 2 | **differs** | RXP-1/1 timing flake: red before, green in all three after passes (Stage 1 saw it the other way round); this suite imports no PWC code |
| `tests/test_s172_d6_1_flush_durability.py` | 0 | 0 | 1 | 1 | same |  |
| `tests/test_s172_d6_2_checkpoint_reconciliation.py` | 1 | 1 | 2 | 2 | same |  |
| `tests/test_s172_d6_integration_repair.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_d6_liveness_gate.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_defect_a_transport_recovery.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_elapsed_roundtrip.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_f1_f2_active_lease.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_f1_lease_origin.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_h1h2_instrumentation.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_miner_seed_start.py` | 1 | 1 | 4 | 4 | same |  |
| `tests/test_s172_mp1_drain_attribution.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_phase1_scaffolding.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_phase2_protocol.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_phase3_worker.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_phase4_coordinator.py` | 0 | 1 | 0 | 1 | **differs** | Gate 22 only: the uncommitted-.py tripwire, naming only PWC files |
| `tests/test_s172_phase5_d0.py` | 1 | 1 | 12 | 12 | same |  |
| `tests/test_s172_phase5_d1_engine.py` | 1 | 1 | 18 | 18 | same |  |
| `tests/test_s172_phase5_d1_workflow.py` | 1 | 1 | 7 | 7 | same |  |
| `tests/test_s172_phase5_d2_directional_uniqueness.py` | 1 | 1 | 7 | 7 | same |  |
| `tests/test_s172_phase5_d3_0_encoding_contract.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_phase5_d3_5_finalizer.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_phase5_d3_columnizer.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_phase5_d4_serial_backend.py` | 1 | 1 | 4 | 4 | same |  |
| `tests/test_s172_phase5_d5_process_sharded.py` | 1 | 1 | 18 | 18 | same |  |
| `tests/test_s172_phase5_d6_production_adapter.py` | 1 | 1 | 9 | 9 | same |  |
| `tests/test_s172_phase5_d6_threshold_path.py` | 1 | 1 | 16 | 16 | same |  |
| `tests/test_s172_phase6_p05_dataset_authority.py` | 1 | 1 | 1 | 1 | same |  |
| `tests/test_s172_process_sharded_import_gate.py` | 0 | 0 | 1 | 1 | same |  |
| `tests/test_s172_r1_drain_remedy.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_resolved_execution_set.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_run4_routing_patch.py` | 1 | 1 | 7 | 7 | same |  |
| `tests/test_s172_small_repairs_part3.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_s172_staging_backpressure.py` | 0 | 0 | 2 | 2 | same |  |
| `tests/test_s172_threshold_propagation.py` | 1 | 1 | 2 | 2 | same |  |
| `tests/test_s172_window_anchor_brief_i_mutants.py` | 0 | 0 | 2 | 2 | same |  |
| `tests/test_s172_window_anchor_brief_i.py` | 1 | 1 | 1 | 1 | same |  |
| `tests/test_seed_domain_cursor_amendment.py` | 1 | 1 | 4 | 4 | same |  |
| `tests/test_step1_bayes_storage_path.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_step1_credit_wal.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_step1_search_record_stage1.py` | 1 | 1 | 35 | 35 | same |  |
| `tests/test_survivor_volume_gate.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_watcher_followup_repairs.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_watcher_llm_integration.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_watcher_safety_repairs.py` | 0 | 0 | 0 | 0 | same |  |
| `test_s149_rocr_isolation_harness.py` | 0 | 0 | 0 | 0 | same |  |
| `test_persistent_worker_harness.py` | 0 | 0 | 0 | 0 | same |  |
| `test_s147_patches_harness.py` | 0 | 0 | 0 | 0 | same |  |
| `tests/test_pwc_backup_step1.py` | 1 | 0 | 1 | 0 | **differs** | PWC suite: G-S2-ZERO-BYTE-EQUAL read HEAD as the pre-stage-1 base; pinned to 27eb991 |
| `tests/test_pwc_stage1.py` | 1 | 0 | 3 | 0 | **differs** | PWC suite: the evidence gates read HEAD as the base; pinned to 27eb991 |
| `tests/test_pwc_backup_stage2.py` | MISSING | 0 | 0 | 0 | NEW | new: 26/26 gates, 29/29 mutants |

**The four differences:**

1. `tests/test_pwc_backup_step1.py` and `tests/test_pwc_stage1.py`, rc 1 → 0. Both are PWC suites, and both are red
   on clean `ee39404`: they compared against `git show HEAD:`, which stopped being the pre-Stage-1 base when Stage 1
   landed. Pinning the base to `27eb991` makes them green again; see "PWC test changes".
2. `tests/test_s172_phase4_coordinator.py`, rc 0 → 1, **Gate 22 only** (62/63). This is the git-status coexistence
   tripwire. It lists uncommitted `.py` files outside its allowlist, and here it names exactly
   `persistent/pwc_transport_tcp.py`, `persistent/pwc_worker_service.py`, `test_persistent_worker_harness.py` and
   three `tests/test_pwc_*.py` files, all PWC.
   - It is the same transitional red as Stage 1; the TB rulings say not to widen the allowlist
     (`docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`).
   - After Stage 1 was committed, this suite was 63/63 on the committed tree (lead, 2026-11-20).
   - The four phase5 suites that run phase 4 inside them (`d1_engine`, `d1_workflow`, `d2`, `d5`) keep rc 1 and the
     same failing checks. Only the reason text of their NR line changes, from the phase-3 sub-run (already red at
     baseline) to this Gate 22.
3. `tests/test_s172_attempt6_remediation.py`, rc 1 → 1. Check RXP-1/1 failed before (75/78) and passed after (76/78).
   It is the same timing flake Stage 1 recorded in the other direction (Stage 1 README). This suite imports no PWC code.
4. Outside these, every exit code and every failing check is the same. The suites that are red at baseline stay red
   for the same checks, including:
   - `test_s172_threshold_propagation`: G-KERNEL UNAVAILABLE (no GPU) and G-MINER-UNCHANGED (D6 positive control);
     G-PWC-HYBRID and M3 pass both before and after, on different content;
   - `test_s172_miner_seed_start`: no dataset pointer manifest in a worktree;
   - `test_s172_admission_stall`: rc 124 at the 1200 s timeout.

**Live state, both runs:**
- `~/miner_staging`: the listing is identical, and the mtime after every one of the 73 suites is
  `2026-11-19 23:24:44.88`, unchanged. Only its parent's entry (`..`, the home directory) moved.
- The live optuna studies listing is unchanged.
- The main tree is clean, at `ee39404`.
- The ssh stub refused 25 network calls in each run, the same 25 (`fake_net_calls.before.log` / `fake_net_calls.log`).
  They come from a miner suite, as in Stage 1.
- Code identity: `before/code_sha256.txt` and `after/code_sha256.txt` differ only in the nine Stage 2 files.

Files: `before_after_tally.md` (rc and the tally lines of every suite), `before_after_failing_checks.txt` (per-suite
failing-check sets) for the final pass `after_r3`; the first pass is `before_after_tally_r1.md` and the second is `before_after_tally_r2.md`. Logs are in
`logs/pwc_stage2_v1/{before,after,after_r2,after_r3}/`. Live state for `after_r3` (and the same in `after_r2`): `~/miner_staging`'s mtime was unchanged after every suite,
the optuna listing and the main tree were unchanged, the ssh stub refused the same 25 calls, and the code hashes differ from `before` only in the nine Stage 2 files.


### Development runs in the scratch copy (`logs/pwc_stage2_v1/scratch_tree`, an extract of `ee39404` + the changes)

| Suite | Result |
|---|---|
| `tests/test_pwc_backup_stage2.py` | R1: 18/18 gates, 21/21 mutants. R2: 24/24, 27/27 (+3 repeats). R3: 26/26, 29/29, four runs in a row (`r3_dev_stage2_{1..4}.log`) |
| `tests/test_pwc_stage1.py` | 9/9, 8/8 |
| `tests/test_pwc_backup_step1.py` | 22/22 (1 HELD), 16/16 (1 HELD) |
| `tests/test_pwc_cache_reset.py` | 21/21 |
| `test_persistent_worker_harness.py` | 25/25 |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | rc 0 |
| `tests/test_s172_threshold_propagation.py` | G-PWC-HYBRID PASS, M3 killed; G-KERNEL UNAVAILABLE (no GPU), G-MINER-UNCHANGED FAIL (D6 positive control) — both as at baseline |

## Round 2 — Beta X-PWC-STAGE2 R1 Q6, finding 2 and Q7 (owner decision 2026-11-20 10:47 PST)

The Q6 label stays **LIVE**, as Beta set it; it is fixed, not contested.

### Q6 — the reconnect race (`persistent/pwc_transport_tcp.py`)

- `:694-716` — the handler's cleanup:
  - the ownership check and every piece of per-id state it governs (registry pop, last-seen, online/ready) now
    change in **one** critical section under `_workers_lock`, the lock a replacement takes to register;
  - leases are reclaimed **only by the closing socket** (`_reclaim_worker_jobs(worker_id, conn)`, `:716`), whether
    or not the handler is the owner.

  So a replacement registers either before the cleanup (the old handler is not the owner and clears nothing) or
  after it (nothing of the replacement exists yet). The old code popped under the lock, released it, then reclaimed
  every lease filed under the id and cleared ready. That is Beta's interleaving.
- `:415-447` — timeout eviction follows the same rule:
  - `_reap_silent_workers` re-checks the silence under the lock;
  - it pops the registered connection and clears that id's last-seen and online/ready in the same critical section;
  - it reclaims only that connection's leases, and shuts its socket down so the handler wakes.

  The old reaper reclaimed every lease for the id and closed the socket without clearing ready.
- **Test:** `tests/test_pwc_backup_stage2.py:771-821`, G-R2-RECONNECT-RACE, forces Beta's interleaving on a real
  loopback transport:
  1. the old handler is paused inside its cleanup;
  2. the same worker reconnects, reports ready and leases a job;
  3. the old handler resumes.

  The replacement must still be registered and ready, its lease active, nothing requeued, and its result delivered.
  - **Mutant M-R2-Q6a** restores the R1 cleanup and is killed: `workers=1 ready=0`.
  - `:824-854`, G-R2-REAPER-OWNERSHIP, with **mutant M-R2-Q6b** (reaper leaves ready set) killed.

### Finding 2 — late result accepted under the next phase's job id

- `persistent_worker_coordinator.py:2041-2064` — job ids are unique to each pass: `sieve_<12-hex pass id>_NNN`
  instead of `sieve_NNN` in every pass. A retry within a pass keeps its id, so it is the same job.
- `persistent/pwc_transport_tcp.py:361-374`, `:661-670` — `_take_active_lease`. A result retires a lease and is
  delivered only if:
  - the job has an active lease;
  - the result names that lease;
  - it arrives on the connection holding the lease.

  Anything else is dropped and counted in `stale_results` (`:178`). The old code popped the lease by job id alone and
  delivered by job id alone.
- **Tests** (`tests/test_pwc_backup_stage2.py`):
  - `:903-985`, G-R2-LATE-RESULT-ACROSS-PHASES — Beta's sequence end to end, through the real coordinator and real
    transport, with **equal forward and reverse thresholds (0.5 / 0.5)**:
    1. worker A stalls on a P1 chunk;
    2. the chunk times out (`JOB_TIMEOUT_S` shortened to 2 s in the test) and is retried on worker B;
    3. P1 completes and P2 is submitted;
    4. only then does A send its late P1 result.

    P2 must report only B's survivors.
  - `:857-866`, G-R2-JOB-IDS-UNIQUE-PER-PASS.
  - `:869-900`, G-R2-ACTIVE-LEASE-ONLY — a superseded lease and a right-lease-wrong-connection result are dropped;
    the active one is delivered.
  - **Mutants:** M-R2-F2a (ids restart at `sieve_000`), M-R2-F2b (any result retires the lease) and **M-R2-F2c (both
    removed: the pre-R2 path Beta reproduced)**, killed with "P2 reported worker A's late P1 result: [111, 333]".
    Each guard alone already stops the cross-phase case, which is why F2c removes both (VIR-2: a mutant masked by
    the other guard would be invalid, not a survivor).
- `tests/test_pwc_backup_step1.py:372` — its S2 byte-equality gate also excludes `job_id` now, and sorts by seed range.

### Q7 — the refusal exit writes no GPU heartbeat (`persistent/pwc_worker_service.py:358-370`)

- The `_emit_heartbeat("shutdown")` call on the refusal path is removed. Writing a heartbeat imports CuPy (`:72`) and
  fsyncs a file (`:99`). The worker logs the reason, closes and exits 43.
- **Test:** `tests/test_pwc_backup_stage2.py:989-1006`, G-R2-REFUSAL-NO-GPU. The real worker service runs in a child
  process against a refusing loopback transport; it must exit 43 with `cupy` never imported and no heartbeat file.
  **Mutant M-R2-Q7** puts the heartbeat back and is killed: `CUPY_LOADED=True`.

### Q3 — five strategies on the rigs: left as NEEDS EVIDENCE

Beta asked for rig measurements:
- phase-3/4 chunk times and completion at the intended cap and concurrency;
- peak memory and host KFD/MES events;
- the 600 s result timeout against the tail of observed durations;
- a one- vs five-strategy hit-rate comparison against `k/1000` on unseen draws.

None can be taken with the rigs off. **Left for the owner's Stage 4 rig sessions.**

### What round 2 did not change

- `miner/`, the ten pinned files, `window_optimizer*.py`, `sieve_filter.py`, the launchers, the manifests, WATCHER.
- The two shared suites: the AST proof still passes, and they are byte-identical to round 1.

## Round 3 — Beta X-PWC-STAGE2 R2 Q1 and finding 1 (owner decision 2026-11-20 12:34 PST; no third Beta round)

Beta's R2 review (`docs/exchange/from_beta/X-PWC-STAGE2_R2_BETA_REVIEW.md`):
- R2-Q1 is **OBJECTION · LIVE**. A same-id reconnect at the timeout boundary: the replacement registered under
  `_workers_lock` but refreshed last-seen in a separate section, so the reaper could pair the fresh connection with
  the old stale timestamp and evict it.
- R2-Q2, Q3 and Q4 are NO OBJECTION.
- Finding 1 is a SUGGESTION.

The fix is applied **exactly as Beta specified**, all in `persistent/pwc_transport_tcp.py`:
- `:512-520` — registration and the fresh last-seen time are set in **one** critical section under `_workers_lock`,
  the section `_reap_silent_workers` uses. The reaper therefore sees either the old connection with its stale time
  (and evicts it, before the replacement registers) or the replacement with its fresh time.
- `:541-549` — the per-message last-seen refresh happens only while this connection is the registered one for the id.
  A replaced connection's message refreshes nothing and ends that handler.
- `:551-560`, `:562-572` — `online` and `ready` are set only by the registered connection, inside the same lock.
  An evicted or replaced connection cannot mark the id online or ready.
- Lock order stays registry first, then last-seen or state. Nothing nests them the other way.

Tests (`tests/test_pwc_backup_stage2.py`):
- `:1053-1117`, **G-R3-REAPER-SAME-ID-RECONNECT** — Beta's interleaving, forced on a real loopback transport:
  1. the old connection is still registered, with a last-seen time 310 s old;
  2. the SAME worker reconnects;
  3. its handler is held exactly at its last-seen refresh (a pausing stand-in for the lock) while the reaper runs;
  4. the handler is released.

  The replacement must not be evicted, and it must go ready, take a job and have its result delivered.
  **Mutant M-R3-Q1a** puts registration and refresh back in separate sections (the R2 code) and is killed: "the
  reaper evicted the fresh replacement".
- `:1120-1150`, **G-R3-REPLACED-CONN-NO-REFRESH** — a replaced connection's heartbeat leaves the id's last-seen
  untouched and ends that connection. **Mutant M-R3-Q1b** (any connection refreshes) is killed.
  - In development this mutant first *survived*: the gate opened both connections back to back, so the old one's own
    `ready` could arrive after the replacement had registered, and the old handler had already ended.
  - The gate now waits until the old connection is fully up before opening the replacement. That was an invalid
    experiment under VIR-2, fixed in the test; the transport code was not changed for it.
- `:928-955` and the P2 assertion — **finding 1:** the cross-phase late-result gate now assigns workers deterministically:
  - A takes the first job of the run (a P1 chunk), before B asks for anything;
  - after its late result A asks for nothing more, so B serves the P1 retry and every P2 chunk;
  - the gate requires P2's survivors to be exactly B's.

**The owner's decision, in plain words:**
- Beta specified the fix, and it is applied as specified.
- The failure it closes costs a retry, not a wrong result: a healthy replacement was evicted and its job requeued, or
  failed on the third attempt, which makes the pass incomplete. No result from the wrong job is delivered, because
  results need the active lease (R2-Q2).
- There is no third round. Beta sees these lines again at the Stage 3 review.
- **Q3 (five strategies on the rigs) stays NEEDS EVIDENCE** for the owner's Stage 4 rig sessions.

Closing note (not sent through the relay): `docs/exchange/to_beta/X-PWC-STAGE2_R3_ALPHA_CLOSING_NOTE.md`.

## Files to land

Code and tests (`git diff --stat ee39404` plus one new file); none is under `miner/` or among the ten pinned files:

| File | |
|---|---|
| `persistent_worker_coordinator.py` | PWC coordinator |
| `persistent/pwc_transport_tcp.py` | PWC transport |
| `persistent/pwc_worker_service.py` | PWC worker (Beta R1 Q4 and R1 Q7 only) |
| `test_persistent_worker_harness.py` | PWC suite |
| `tests/test_pwc_backup_step1.py` | PWC suite |
| `tests/test_pwc_stage1.py` | PWC suite |
| `tests/test_pwc_backup_stage2.py` | new PWC suite |
| `tests/test_s172_threshold_propagation.py` | shared suite, PWC gate + mutant only (AST proof) |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | shared suite, PWC drive + one PWC branch only (AST proof) |

Records:

| File | |
|---|---|
| `docs/drafts/pwc_backup_stage2_v1/README.md` | this file |
| `docs/drafts/pwc_backup_stage2_v1/pwc_backup_stage2.diff` | the nine files above; applies to a fresh `git archive ee39404` with `patch -p1 --fuzz=0` and reproduces all nine byte for byte |
| `docs/drafts/pwc_backup_stage2_v1/before_after_tally.md`, `before_after_failing_checks.txt` | final pass `after_r3` |
| `docs/drafts/pwc_backup_stage2_v1/before_after_tally_r1.md`, `_r2.md`, `before_after_failing_checks_r1.txt`, `_r2.txt` | the two earlier passes |
| `docs/exchange/to_beta/X-PWC-STAGE2_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md` | R1 request (sent) |
| `docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md` | Beta R1 (written by the relay; not edited) |
| `docs/exchange/to_beta/X-PWC-STAGE2_R2_ALPHA_REPLY.md` | R2 reply (sent) |
| `docs/exchange/from_beta/X-PWC-STAGE2_R2_BETA_REVIEW.md` | Beta R2 (written by the relay; not edited) |
| `docs/exchange/to_beta/X-PWC-STAGE2_R3_ALPHA_CLOSING_NOTE.md` | closing note (not sent) |
| `docs/exchange/LEDGER.md` | two rows appended by the relay (R1, R2); `DEFERRED.md` unchanged |

**Not to land:** `data/` (untracked output the suites write), and everything under `logs/` (ignored: runners, logs,
scratch copy).

## What a later rig redeploy would involve (owner-run; nothing here touches a rig)

1. After Stage 2 is reviewed and committed: rigs booted into Proxmox (CT100 `.122/.156/.164`), and the owner's go.
2. `scp`, with absolute paths, of only the three runtime files Stage 2 changed that run **on a rig**:
   `persistent/pwc_transport_tcp.py`, `persistent/pwc_worker_service.py` and, for completeness of the PWC tree,
   `persistent_worker_coordinator.py` (it runs on VM101, not on the rigs). `sieve_filter.py` did not change.
   **No gate12-pinned file is copied.**
3. Run `scripts/pwc_parity_gate.py --transport tcp`. Its governed set (`scripts/pwc_parity_gate.py:107-123`)
   includes `persistent/pwc_worker_service.py` and `persistent/pwc_transport_tcp.py`, so it must show the new bytes on
   all three rigs.
4. Run `scripts/gate12_parity_gate.py`. The digests must be unchanged, which proves the miner's ten files did not move.
5. No kernel source changed, so nothing forces a CuPy recompile; the per-run cache reset applies anyway.
6. **Until then the rigs run the pre-Stage-2 worker.** A refused stale worker there still retries every 5 s (Beta Q4).
   The coordinator side of Stage 2 is live as soon as it is committed on VM101.

## What is NOT verified

- **Anything on a rig or a GPU**:
  - the probe control under a real `sshd` → `bash -c` on a CT100;
  - real revoke and reconnect timing;
  - the worker's exit under a real launch script;
  - the five-strategy cost on phases 3–4 against the stable envelope (ROCm Saturation report rows are for one strategy);
  - whether the forward hybrid's extra 16th argument is harmless (it is still sent at offset 0, as before).
- **Hit rate.** No PWC-produced pool has been measured against `k/1000`. Nothing here says PWC output is good, only
  that it is now produced the way RANGE-MINER's is on the points changed.
- **The 4-place echo.** The contract compares the worker's report at 4 places, because `sieve_filter.py` (not edited)
  rounds it. Two thresholds within 5×10⁻⁵ of each other are not told apart by the report. Both engines pass
  `float32(threshold)` to the kernel.
- **That the remote shell carries the command text** (the probe control's premise). It is true for `ssh host '<cmd>'`
  (`bash -c`) and for the namespace tests here. A rig whose login shell `exec`s a single command would make the
  control fail closed (UNAVAILABLE, rig refused), not fail open.
- **The SSH route's S153 claim** (see the brief list, item 8).
- **WATCHER and the pipeline call site.** The integration still passes no dial address, so PWC with rigs still refuses
  at startup from the pipeline. That is Stage 3.

## Open questions for the owner

1. **The anchor/phase split.** It needs `sieve_filter.py` (PWC-reached code only) **and** a two-key passthrough in the
   worker (`persistent/pwc_worker_service.py:541-561` (`:535-555` on `ee39404`)). Until then, every PWC trial with window offset ≠ 0 refuses,
   which includes most Optuna-sampled trials and Run 4's anchor 25. Approve the worker edit?
2. **The per-run launch token** (scope A3 level 2, Beta Q3). It needs the worker to send it. Approve that worker edit?
3. **F2 HSA variables:** keep as is, or drop all three, which needs worker and coordinator edits together?
4. **F1:** remove the SSH route entirely in Stage 3, or keep it for constant-only diagnostics?
5. **F3:** cache records for counted PWC runs?
6. **F5:** confirm all five strategies, replacing S147 Q2.
7. **F6:** keep the S147 prune behind a non-default flag, or leave it deleted?
8. **F4 / skip range:** still yours, untouched. The two engines differ on phases 1–2 until it is ruled.
9. **The exact echo.** Make `sieve_filter.py` report the unrounded threshold, a one-line PWC-reached change, together
   with item 1?

