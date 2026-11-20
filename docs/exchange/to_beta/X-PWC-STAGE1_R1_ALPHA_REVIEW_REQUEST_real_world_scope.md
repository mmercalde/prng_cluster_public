# X-PWC-STAGE1 · R1 · REVIEW_REQUEST · PWC backup stage 1, judged on real-world failures on this cluster

Written by Team Alpha (Claude Code). Sent through `scripts/beta_relay.py` on the owner's approval (2026-11-20 05:34 PST).
Nothing is committed.

## Prior record

**Searches run** (2026-11-19, worktree `/home/michael/tfm_worktrees/pwc-backup` at main `27eb991`; `/bin/grep`, because
the shell `grep` skips `*.json`). Scope `S` = `docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_*.md docs/TB_RULING_REQUEST_*.md
docs/PROPOSAL_*.md docs/CHAPTER_*.md docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md`, plus
`/bin/grep -rl` over `docs/exchange/`:

```
/bin/grep -lIiE "pkill.{0,40}self|self-match|self-kill|\[p\]wc_worker"            $S ; docs/exchange/
/bin/grep -lIiE "launch set|stale.worker.{0,30}reconnect|accepted.: False|reconnect limit"  (same)
/bin/grep -lIiE "skip_pattern"                                                     (same)
/bin/grep -lIiE "PWC.STAGE1|Stage 1.{0,30}PWC|pwc_backup_stage1"                   (same)
```

The scope this stage comes from carries the full governance search, by reference:
`docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md` "Prior record" (on main at `27eb991`).

**Found, and read:**

| Document | What it settles |
|---|---|
| `docs/drafts/SCOPE_PWC_FULL_BACKUP_20261119.md`, Stage 1 row and items A3, A4, B5, B6 | This work. Owner approval to start Stage 1 only: 2026-11-19 20:50 PST ("get started, DO NOT break range miner") |
| `docs/exchange/from_beta/X-PWC-CACHE-RESET_R1_BETA_REVIEW.md:85-95`, `..._R2_BETA_REVIEW.md:97`, `:103` | Your TCP admission gap (stale worker reconnects and is served). A3 level 1 is the first half of your R1 follow-up |
| `docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md` (in `/home/michael/tfm_worktrees/pwc-step1`, not on main); your X-PWC-BACKUP R1/R2 | The step-1 candidate S1–S8, which stage 1 lands. You raised no LIVE objection |
| `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120`; `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:145-151` | Git-status tripwires red on an uncommitted tree are transitional; do not widen gate 22's allowlist |
| `docs/PROJECT_FILE_CATALOG.md:497`, `docs/BACKLOG.md:51` | `skip_pattern` hits: the Oct-2025 output spec lineage. Neither records the PWC parser mismatch |

The self-kill search found nothing in any governing document. CLAUDE.md hard rule 8 records the class in general.

**Labels:**

| Item | Label |
|---|---|
| The step-1 candidate S1–S8 | ALREADY RECORDED (X-PWC-BACKUP R1/R2; the scope's Baseline) |
| B5: the remote kill shell kills itself, so the `sieve_gpu_worker` kill never ran. **Now shown by execution** | Defect NEW in the scope (B5, by reading). The execution evidence is NEW here |
| B5: the S156 find (SSH route) always matched its own shell, so it never printed `none` | **NEW** |
| A3 level 1 | ALREADY RECORDED (your R1 follow-up; scope A3) |
| B6 | NEW in the scope (B6) |
| A4 repairs of the old harnesses | ALREADY RECORDED (scope A4; `docs/drafts/NOTE_PWC_OLD_TEST_FAILURES_20261119.md`) |
| Holding back the candidate's `persistent/pwc_worker_service.py` hunk | **NEW**. The stage-1 order forbids editing "the worker". The owner decides |
| No ruling is changed. The hybrid quarantine, the S147 phase policy, R5 and PWC credit are all untouched | — |

**Not searched:** ser8, the private repository, the claude.ai project files, the rigs (powered off), Proxmox.

```
id / round   X-PWC-STAGE1 / R1
type         REVIEW_REQUEST (before-commit: coordinator code and transport code)
base         main 27eb991
package      docs/drafts/pwc_backup_stage1_v1/  (README.md, pwc_backup_stage1.diff, HELD_pwc_worker_service.diff)
```

## The owner's framing (please read first)

1. **Judge only real-world failure cases on THIS cluster:** three rigs with 8 RX 6600 XT cards each, run as Proxmox
   CT100s at `.122`/`.156`/`.164`; VM101 (`.177`, DHCP) with one 3080 Ti; one operator. RANGE-MINER is the production
   engine. PWC is the backup and **is not in use today**.
2. **A point may be `BLOCKS` only if you can name a concrete sequence of events** that could plausibly happen on this
   cluster **and** that would break a run, break RANGE-MINER, or corrupt results. Everything else is `TO-DO`.
3. **The owner's first concern is that RANGE-MINER is not broken.** Please say directly whether any line of this
   package can change what a RANGE-MINER run (Run 4 shape, launched by `gate12_launch.sh` / `scripts/launch_fleet_manual.sh`)
   does. The package's claim is that it cannot, for these reasons:
   - no file under `miner/` and none of the ten files pinned by `scripts/gate12_parity_gate.py` is edited;
   - `window_optimizer*.py`, `sieve_filter.py`, `sieve_gpu_worker.py`, `prng_registry.py`, the launchers and the
     manifests are not edited;
   - the only edited file the miner's launch process loads is `persistent_worker_coordinator.py`, imported at module
     scope by `window_optimizer_integration_final.py:58`. Its import-time effect is definitions and constants only, and
     G-IMPORT checks the import in a fresh process.
4. **Tests are not the yardstick.** Nothing here is judged by whether any seed is found. The yardstick question for
   PWC stays open until PWC-credited pools are measured against `k/1000`.

## What changed (details and `file:line` in the README)

- **The step-1 candidate (S1–S8), merged over main.** It applies to `27eb991` with `git merge-file` rc 0. It touches:
  - `persistent_worker_coordinator.py`;
  - `persistent/pwc_protocol.py`;
  - `persistent/pwc_transport_base.py`;
  - new: `scripts/pwc_parity_gate.py` and `tests/test_pwc_backup_step1.py`.

  The fourth candidate file, `persistent/pwc_worker_service.py`, is **held back** (see Q5).
- **B5.** The rig stale-worker kill uses bracket patterns (`[p]wc_worker_service`, `[s]ieve_gpu_worker`). It prints
  `PWC_STALE_KILL_OK` only when no match is left, and its result is read: an unconfirmed kill refuses that rig for
  the run, as a failed cache reset already does. The S156 find and reap (SSH route) use the bracket pattern as well,
  and an unexpected reap result is logged as a warning.
- **A3 level 1.** `TCPWorkerTransport` gains a launch set:
  - startup opens it **empty, before the listener starts**;
  - each launch adds its worker id;
  - a launch that prints no PID removes the id again;
  - an id outside the set gets `accepted: False` and is registered nowhere;
  - a refused hello cannot evict a live worker filed under the same id.
- **B6.** The TCP legacy parser reads `skip_pattern` (what `sieve_filter.py` emits) as well as `skip_sequence`.
- **A4.** The two old root harnesses are repaired, and so are the two fixtures the stage-1 code changes break
  (`tests/test_pwc_backup_step1.py`, `tests/test_pwc_cache_reset.py`). `tests/test_s172_threshold_propagation.py`'s
  D6 sub-harness is a **miner test**, so it is left for the miner owner.

## Questions

**Q1. B5 on real rigs.** The old remote kill was `pkill -9 -f pwc_worker_service …; pkill -9 -f sieve_gpu_worker …;
echo killed`. In a private PID namespace that shell was SIGKILLed by its own first `pkill` (rc 137). The second `pkill`
never ran and nothing was printed (G-B5-OLD-SELF-KILL). In one full-suite run the `pwc_worker_service` dummy also
survived the first `pkill`; in six other runs (one development run and five repeats) it was killed. Whether the first `pkill` finishes before its own shell
dies is a race, so a stale PWC worker may also have survived this step at times. So for as long as this code has existed, **stale
`sieve_gpu_worker` processes on a rig were never killed by this step.** The new kill reaches them.

Name any process on a CT100 that has `pwc_worker_service` or `sieve_gpu_worker` in its command line and must NOT be
killed when a PWC run starts. In particular, is there one that belongs to RANGE-MINER? (The miner's workers run as
`miner/range_miner_worker.py`; `scripts/launch_fleet_manual.sh:347`, `:399`.)

The S156 find on the SSH route (`pgrep -af "sieve_gpu_worker.*--persistent" || echo none`) matched its own shell, so
it never printed `none` (G-B5-S156). The reap therefore ran on every rig at every start, and its own `pkill -15`
matched its shell in the same way.

**Q2. B5 refusal.** A rig whose stale worker does not exit within the 2-second wait is now refused for that run (exit
6, no marker). A worker can sit in uninterruptible sleep inside the amdgpu driver, as in L-6/L-7. Is refusing the rig
the right outcome there, or should the wait be longer? What sequence on this fleet would make the refusal itself cause
harm? Completion accounting (S1) already marks any pass incomplete if a rig's share is never swept.

**Q3. A3 level 1 limits.** Level 1 refuses ids that are **not** in this run's launch set:
- a stale worker on a rig skipped this run;
- a stale worker on a GPU index above this run's pool;
- any worker before its launch is issued.

It does **not** separate a stale worker that has the **same** id as a freshly launched one. Worker ids are
deterministic (`<ip>_gpu<N>`), and only the per-run token of level 2 does that. With B5's kill now confirmed before
launch, what concrete sequence leaves a same-id stale worker alive and connecting on this cluster? Is it serious
enough to make level 2 a `BLOCKS` for landing stage 1, or a `TO-DO` for stage 2?

**Q4. Refused workers retry forever.** A refused worker logs, sleeps 5 s and retries (`pwc_worker_service.py:213`;
the rejection raises `ConnectionError`, `:358-361`). The worker is not edited in stage 1. Does a refused stale worker
looping every 5 s for the length of a run do any harm here: load, log growth in `/tmp`, or the dispatch semaphore?

**Q5. The held worker hunk.** The candidate's S3 also removed the worker's own fallback dial target
(`PWC_HOST` defaulted to `192.168.3.127`, the frozen bare-metal Zeus). Stage 1 does not apply that hunk, because the
order forbids editing "the worker". The coordinator now **exports** `PWC_HOST`/`PWC_PORT` in every launch script
(G-S3-SET), so a coordinator-launched worker never reaches the fallback.

Is there a real path on this cluster where a PWC worker starts **without** those variables? A hand start, an old
launch script left in `/tmp` on a rig, or a systemd unit? If so, it would dial `.127`. Is that safe while `.127` is
down (VM101 booted)? Is it unsafe whenever Zeus boots bare-metal?

**Q6. PWC from the pipeline after stage 1.** `window_optimizer_integration_final.py:1968-1986` passes no
`pwc_coordinator_address`, and it passes `seed_cap_amd` 2,000,000 by default. With the candidate's S3/S5, a PWC run
from the pipeline that includes rigs now **refuses at startup**, naming the missing address or the envelope. Before,
the rig workers silently dialled `.127:5600`, so on VM101 PWC with rigs could not work anyway. Fixing the call site is
stage 3, a shared launch file. Do you see a run on this cluster that worked before stage 1 and stops working after it?

**Q7. RANGE-MINER.** See framing point 3. Name any path by which this package changes a RANGE-MINER run, or say that
you found none.

## Test evidence (full table in the README)

- 71 existing suites were run before (clean `27eb991`) and after, sequentially, CPU only, with ssh/scp/rsync replaced
  by a refusing stub.
- 4 suites differ in exit code or in the set of failing checks:
  - two PWC harnesses go from red to green (A4);
  - phase-4 gate 22 goes red; it is the git-status tripwire, and it names only this package's files;
  - `test_s172_attempt6_remediation` RXP-1/1 went red once, then passed on two immediate re-runs (76/78, the same as
    the baseline). It is a timing flake in a miner suite that does not reference PWC.
- G-MINER-UNCHANGED was red before (a stale D6 sub-harness) and is red after. It now stops one assertion earlier, on
  `persistent/pwc_protocol.py` in its frozen list.
- All ten parity-pinned files are byte-identical to `27eb991` (sha256 table in the README).

## What is not verified

- Real ssh, real rigs and real `pkill` on a CT100. B5 ran in a local private PID namespace with dummy processes, not
  with ROCm workers.
- A3 ran on 127.0.0.1 with fake workers, not with `pwc_worker_service`.
- Gate 22 and G-MINER-UNCHANGED on a **committed** tree. They are red on this uncommitted tree for the transitional
  reason recorded in the rulings above, and the README lists the exact paths.
