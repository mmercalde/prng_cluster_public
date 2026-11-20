# X-PWC-STAGE2 · R1 · REVIEW_REQUEST · PWC backup stage 2, judged on real-world failures on this cluster

Written by Team Alpha (Claude Code). Sent through `scripts/beta_relay.py` on the owner's approval (2026-11-20 10:15
PST), one round. Nothing is committed.

## Prior record

**Searches run** (2026-11-20, worktree `/home/michael/tfm_worktrees/pwc-backup` at main `ee39404`; `/bin/grep`, because
the shell `grep` skips `*.json`). Scope `S` = `docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_*.md docs/TB_RULING_REQUEST_*.md
docs/PROPOSAL_*.md docs/CHAPTER_*.md docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md`, plus
`/bin/grep -rl` over `docs/exchange/`:

```
/bin/grep -lE "PWC.*[Ss]tage 2|pwc_backup_stage2|STAGE2"                       $S ; docs/exchange/
/bin/grep -lE "balanced_hybrid|S147 Q2|Q2.*single strategy"                    (same)
/bin/grep -lE "forward_zero|S147.*prune|B1 gate"                              (same)
/bin/grep -lE "HSA_ENABLE_DEBUG_TRAP|HSA_OVERRIDE_GFX"                         (same)
/bin/grep -lE "window_anchor.*PWC|PWC.*window_anchor|PWC.*generator_phase"     (same)
/bin/grep -lE "VERIFICATION_INTEGRITY|VIR-5"                                   (same)
/bin/grep -lE "launch token|RUN_NONCE|run token"                               (same)
/bin/grep -rlE "S147" docs/ ; /bin/grep -rnE "balanced_hybrid" docs/*.md
```

**Found, and read:**

| Document | What it settles |
|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md` (on main, `ee39404`) | The four owner rulings and the default rule ("PWC does what RANGE-MINER does"). **Fixed facts for this review** |
| `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md`, Stage 2 row | This stage. Owner's go for Stage 2: 2026-11-20 06:32 PST |
| `BRIEF_PWC_BACKUP_STAGE2_STAGE3_20261119.md` (rig-gpu-fault worktree, not on main) | The Stage 2 change list this package follows; where it is wrong is in the README |
| `docs/exchange/from_beta/X-PWC-STAGE1_R1_BETA_REVIEW.md` Q3, Q4, findings 1–3 | The three items carried into Stage 2 and the missing VIR citation |
| `docs/VERIFICATION_INTEGRITY_STANDARD.md` VIR-1 (`:30-35`), VIR-2, VIR-5 (`:93-117`); `docs/SESSION_CHANGELOG_20260814_S182.md:122-126` | A check must prove it could observe; `pgrep -c … \|\| echo 0` is the S182 double zero. **Stage 1's prior record omitted these (your finding 1)** |
| `docs/STEP1_EXECUTION_FLOW_AND_PRUNING_S147.md:176`; `docs/TODO_MASTER_S148.md:132` (`667ece5`) | S147 Q2 (one `balanced_hybrid` strategy) and the S147 B1/Q0 prune |
| `miner/range_miner_worker.py:22-29`, `:601-605` | RANGE-MINER separates `window_anchor` (host) from `generator_phase` (kernel, pinned 0) |
| `docs/TB_RULING_REQUEST_ROCR_ISOLATION_S149.md`, `docs/CHAPTER_9_*` | The HSA variables' history; not changed here |

**Labels:**

| Item | Label |
|---|---|
| Lift the hybrid block, threshold contract in its place | ALREADY RECORDED: owner ruling 1 |
| No prune, no Q0: phases follow `workflow_stages_for` | ALREADY RECORDED: owner ruling 4. Supersedes S147 B1/Q0 for PWC |
| All five strategies (S147 Q2 removed) | **CHANGES A RULING** (S147 Q2), taken under the owner's default rule; listed for his confirmation (F5) |
| Window offset ≠ 0 refused | ALREADY RECORDED: owner ruling 2026-11-20 08:30 PST (offset = draw window only, phase 0, PWC moves to RANGE-MINER's method, refuse ≠ 0 until then) |
| Probe-checked process checks; VM101 kill read | ALREADY RECORDED: your X-PWC-STAGE1 finding 2 and Q3 |
| Revocation drops the live connection; unconfirmed launch not counted; owner-only cleanup | ALREADY RECORDED: your finding 3 and Q3 |
| Refused worker exits | ALREADY RECORDED: your Q4. The only worker edit |
| Two shared suites edited on PWC lines only | **NEW**; the owner allowed it on 2026-11-20 |

**Not searched:** ser8, the private repository, the claude.ai project files, the rigs (powered off), Proxmox.

```
id / round   X-PWC-STAGE2 / R1
type         REVIEW_REQUEST (before-commit: coordinator, worker and transport code)
base         main ee39404
package      docs/drafts/pwc_backup_stage2_v1/  (README.md, pwc_backup_stage2.diff)
```

## The rulings are fixed facts

Beta is not asked to rule on these. Please check that the code does what was decided
(`docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`):

1. **Lift the variable-skip block once the threshold is fixed the way RANGE-MINER does it.**
2. **A PWC run earns search credit.** That is Stage 3; nothing here grants credit.
3. **Drop PWC's refusal of a non-zero seed start.** That is Stage 3 (a shared launch file).
4. **Phases follow RANGE-MINER:** 1–2 always, 3–4 whenever `test_both_modes`, never cut short.
5. **Default rule:** where nothing is ruled, PWC does what RANGE-MINER does.

Owner ruling of 2026-11-20 08:30 PST, also fixed:
- **The window offset picks the draw window only, and the generator phase is 0.** PWC moves to RANGE-MINER's method
  (`window_anchor` / `generator_phase`, `miner/range_miner_worker.py:22-29`, `:601-605`). **PWC refuses offset ≠ 0
  until that lands.** This package implements the refusal; the move itself is a later change.

Owner choice of 2026-11-20 for this stage, also fixed:
- **The two shared suites** may be edited on their PWC lines only.

## The owner's framing (please read first)

1. **Judge only real-world failure cases on THIS cluster:** three rigs with 8 RX 6600 XT cards each, run as Proxmox
   CT100s at `.122`/`.156`/`.164`; VM101 (`.177`, DHCP) with one 3080 Ti; one operator. RANGE-MINER is the production
   engine. PWC is the backup and **is not in use today**.
2. **An `OBJECTION` is `LIVE` only if you can name a concrete sequence of events that could plausibly happen on this
   cluster as it is today and would give a wrong result, break a run, or break RANGE-MINER.** Anything that needs
   conditions that do not exist here is `THEORETICAL`.
3. **The owner's first concern is that RANGE-MINER is not broken.** Please say directly whether any line of this package
   can change what a RANGE-MINER run (Run 4 shape) does. The package's claim:
   - no file under `miner/`, none of the ten files pinned by `scripts/gate12_parity_gate.py:174-185`, and none of
     `window_optimizer*.py`, `sieve_filter.py`, `sieve_gpu_worker.py`, `prng_registry.py`, the launchers, the manifests
     or WATCHER is edited;
   - the only edited file the miner's launch process loads is `persistent_worker_coordinator.py`, imported at module
     scope by `window_optimizer_integration_final.py:58`. Its import-time effect is still definitions and constants;
     G-IMPORT checks the import in a fresh process.
4. **Tests are not the yardstick.** Nothing here is judged by whether any seed is found. Whether PWC-produced pools
   hit above `k/1000` stays unmeasured.

## What changed (details and `file:line` in the README)

Files:
- `persistent_worker_coordinator.py`;
- `persistent/pwc_transport_tcp.py`;
- `persistent/pwc_worker_service.py` (about 10 lines, your Q4 only);
- PWC suites: `test_persistent_worker_harness.py`, `tests/test_pwc_backup_step1.py`, `tests/test_pwc_stage1.py`,
  and the new `tests/test_pwc_backup_stage2.py`;
- PWC lines only in two shared suites: `tests/test_s172_threshold_propagation.py` (G-PWC-HYBRID, M3) and
  `tests/test_s172_phase5_d3_25_candidate_ingress.py` (`drive_pwc`, one PWC branch). An AST comparison of every
  top-level statement shows nothing else in them moved.

What the changes do:
- **Ruling 1.**
  - The quarantine section is removed.
  - `run_sieve_pass` has no `phase2_threshold` default. A hybrid pass refuses without one, with one `!= threshold`,
    or on SSH.
  - Every job carries the one threshold as both keys.
  - The hybrid record compares requested and payload exactly and the worker's report at 4 places, and it fails on a
    missing report.
  - The pass records its transport.
- **Ruling 4.** The prune and the Q0 gate are deleted.
- **Default rule.** `strategies=None`, so all five strategies are sent. An `ImportError` while loading them refuses.
- **Offset.** `validate_pwc_window_offset` runs in `run_trial_persistent` (before the fleet) and in `run_sieve_pass`.
- **Your finding 2 / Q3.** The probe-controlled kill and the S156 find/reap strings are module functions now. The
  VM101 local kill runs the same string under `bash -c` with only `[p]wc_worker_service`, and its result is read.
- **Your finding 3 / Q3.**
  - `revoke_worker_id` drops the id from the online and ready sets and `shutdown`s its socket.
  - `ready` and `request_job` re-check admission.
  - Cleanup is owner-only, and leases record their connection.
  - A no-PID launch is not counted.
- **Your Q4.** On `accepted: False` the worker calls `sys.exit(43)`.

**Held, not done:**
- the window-anchor / generator-phase split, which needs `sieve_filter.py` plus a worker key passthrough;
- the per-run token (level 2), which needs a worker edit;
- the HSA variables (F2);
- PWC cache records (F3);
- the skip range (F4, the owner's open question).


## Questions

Allowed answers: `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL` and one sentence why) · `SUGGESTION` ·
`NEEDS EVIDENCE`.

**Q1. The threshold contract (ruling 1).** Every PWC job now carries the trial's directional threshold as both
`min_match_threshold` and `phase2_threshold`. A hybrid pass refuses before dispatch without one, or with one that
differs from `threshold`, or on the SSH route. After the pass, what each TCP worker reports it used is checked. The
report is `sieve_filter`'s `single_phase.threshold`, **rounded to 4 places** (`sieve_filter.py:597-599`, not edited).
Requested and payload values are compared exactly, and the report is compared at 4 places. A completed chunk with no
report fails the pass. RANGE-MINER compares exactly. Is there a realistic run on this cluster where a hybrid pass
filters at a value other than the trial's threshold and still passes this check? Is the 4-place comparison a real gap
here, given that the kernel receives `float32(threshold)` in both engines?

**Q2. Phases (ruling 4).** The forward-zero prune and the Q0 gate are deleted. A PWC trial now always runs phases 1–2,
and 3–4 whenever `test_both_modes`, as `workflow_stages_for` does. Do you see a run on this cluster where this changes
anything except the extra passes' cost? (A PWC both-mode trial that finds zero forward survivors used to stop after
one pass; it now runs four.)

**Q3. Strategies (default rule).** S147 Q2 sent only `balanced_hybrid`. PWC now sends all five strategies from
`get_all_strategies()`, the source RANGE-MINER's worker uses. That is about 5× the work on phases 3–4. Is there a
realistic failure from that change on the rigs (time, memory, the stable envelope)?

**Q4. Offset refusal.** PWC uses one `offset` both to pick the draw window and as the kernel's trailing argument.
RANGE-MINER pins the second to 0. So PWC now refuses any trial with window offset ≠ 0, before the fleet is built. This
also refuses most Optuna-sampled PWC trials. Is refusing the right fail-closed behaviour for a backup, or does it break
a run that works today and that the owner relies on?

**Q5. Your finding 2: process checks.** Every `pgrep` verdict is now preceded by a positive control: the shell's own
command line carries the token `PWC_PROBE_CONTROL_7f3a`, so `pgrep -f` of it must find that shell, or the check is
UNAVAILABLE. After the control, `pgrep`'s exit status is read exactly: 1 = none, 0 = a survivor, anything else =
UNAVAILABLE. This applies to:
- the rig kill (exit 7, the rig is refused);
- the S156 find and reap (`probe_unavailable` / `reaped:unavailable`, logged as errors; the SSH route still proceeds,
  as before);
- the VM101 local kill. Its result is now read, and unless it is confirmed no local worker is launched or admitted
  this run.

Is there a CT100 or VM101 state where the control finds the shell but a stale worker is still invisible to the second
`pgrep`?

**Q6. Your finding 3 and Q3: revocation and reconnects.** `revoke_worker_id` now also shuts down a connection the id
already holds, and drops the id from the ready and online sets at once. `ready` and `request_job` from an id off the
list end the connection. A launch with no PID is not counted. Only the connection that still owns an id pops, reclaims
or un-readies under it; a replaced connection reclaims only the jobs it carried itself. Name any realistic sequence
where a legitimate worker on this cluster is now cut off or loses a job.

**Q7. Your Q4: the worker exits on refusal** (`persistent/pwc_worker_service.py`, the only worker edit, about 10
lines). An explicitly refused worker now exits with code 43 after one attempt. The refusal is sent only to ids outside
the launch set; a launched worker's id stays on the list for the whole run, so its reconnects are accepted. Is there a
real sequence where a worker this run launched gets refused, and therefore exits, on this cluster? Note: the rigs
still carry the pre-Stage-2 worker until a redeploy, so the retry loop remains there until then.

**Q8. RANGE-MINER.** See framing point 3. Name any path by which this package changes a RANGE-MINER run, or say that
you found none.

**Q9. The shared suites.** In `tests/test_s172_threshold_propagation.py` and
`tests/test_s172_phase5_d3_25_candidate_ingress.py`, confirm by your own comparison that no RANGE-MINER gate, mutant
or assertion changed and only PWC cases did; name any miner-side line that differs from `ee39404`.

## Test evidence (full table in the README)

- 72 existing suites were run before (fresh, on clean `ee39404`) and after, one after another, CPU only, with
  ssh/scp/rsync replaced by a refusing stub. `tests/test_s172_staging_partb.py` was **not** run in either: it writes a
  probe into the live `~/miner_staging`.
- **68 are identical:** same exit code and same set of failing checks. **4 differ:**
  - the two Stage 1 PWC suites go from red to green. They were red on clean `ee39404` because they read `HEAD` as the
    pre-Stage-1 base; it is now pinned to `27eb991`;
  - `test_s172_phase4_coordinator`: Gate 22 only, the uncommitted-`.py` tripwire, naming only PWC files;
  - `test_s172_attempt6_remediation`: the RXP-1/1 timing flake, red before and green after. This suite imports no
    PWC code.
- The new `tests/test_pwc_backup_stage2.py`: **18/18 gates, 21/21 mutants killed** (four-part rule).
  - The probe gates run in a private PID namespace with three broken `pgrep`s: exit 2, blind, and erroring on the
    target only.
  - The transport gates use a real loopback `TCPWorkerTransport`.
  - The worker-exit gate runs the real worker service in a child process (rc 43 after one refused hello; the mutant
    retried 37 times in 8 s).
  - The echo gate runs the real `sieve_filter.execute_sieve_job` with the GPU methods replaced.
- All ten parity-pinned files and every miner, launch, manifest and WATCHER file have sha256 equal to `ee39404`.
  `sieve_filter.py` is byte-identical.
- `~/miner_staging`'s mtime was unchanged after every suite. The optuna listing and the main tree are unchanged.

## What is not verified

- **Nothing on a rig or GPU:**
  - the probe control under a real CT100 `sshd`;
  - revoke and reconnect timing;
  - the worker's exit under a real launch;
  - the five-strategy cost against the stable envelope;
  - the forward hybrid's extra 16th argument (unchanged, sent as before).
- **Hit rate:** no PWC pool has been measured against `k/1000`.
- **The 4-place report:** thresholds within 5×10⁻⁵ of each other are not told apart.
- **The brief's S153 reverse-argument claim** for the SSH route: no record of it was found.
- **The rigs** still run the pre-Stage-2 worker until an owner-run redeploy.
