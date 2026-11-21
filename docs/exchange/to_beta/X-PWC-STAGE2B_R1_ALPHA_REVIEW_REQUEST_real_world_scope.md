# X-PWC-STAGE2B · R1 · REVIEW_REQUEST · PWC moves to RANGE-MINER's window-anchor / generator-phase method, judged on real-world failures on this cluster

Written by Team Alpha (Claude Code) on 2026-11-20. Sent through `scripts/beta_relay.py` on the owner's approval
(2026-11-20 17:38 PST: one round). Nothing is committed.

## Prior record

**Searches run** (2026-11-20, worktree `/home/michael/tfm_worktrees/pwc-offset`, first at `ee39404`, then at main
`0caeaf9`; `/bin/grep`, because the shell `grep` skips `*.json`). Scope `S` = `docs/PROJECT_FILE_CATALOG.md
docs/TB_RULING_*.md docs/TB_RULING_REQUEST_*.md docs/PROPOSAL_*.md docs/CHAPTER_*.md docs/*_AUDIT.md
docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md`, plus `/bin/grep -rl` over `docs/exchange/`:

```
/bin/grep -lIiE "PWC.{0,60}(window_anchor|generator_phase)|(window_anchor|generator_phase).{0,60}(PWC|sieve_filter|execute_sieve_job)"  $S ; docs/exchange/
/bin/grep -lIiE "sieve_filter.py:18[0-9]|Brief.II|carry-forward"                                                                    (same)
/bin/grep -lIiE "PWC.STAGE2|Stage 2b|pwc_backup_stage2"                                                                              (same)
/bin/grep -lIiE "prove_unchanged|unchanged outside|AST.{0,20}(identical|compar)"                                                     (same)
/bin/grep -rlIE "BRIEF I ACCEPTED|Brief I (is )?accepted" docs/TB_RULING_*.md docs/exchange/
/bin/grep -rlIE "HOST in \(|rrig6600.{0,60}HSA_OVERRIDE|HSA_OVERRIDE.{0,80}(hostname|CT100)" docs/
git diff ee39404 0caeaf9 ; grep -rln "sieve_filter|execute_sieve_job|run_hybrid_sieve|GPUSieve" --include=*.py --include=*.sh .
```

The package README (`docs/drafts/pwc_backup_stage2b_v1/README.md`, "Prior record") carries the window-anchor document
reads and the miner reads.

**Found, and read:**

| Document | What it settles |
|---|---|
| `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` §2, §4.1, §4.2, §4.4, §4.7; `docs/TB_RULING_WINDOW_ANCHOR_V1_1_DESIGN_GATE_CLOSED.md` | `window_anchor` picks the window, on the host. `generator_phase` is the kernel's pre-advance, pinned to 0 in v1. Both keys are required; `offset` is a HARD REJECT; the anchor is never clamped. Approved |
| `docs/S172_WINDOW_ANCHOR_BRIEF_I.md` C-2 (`:60-90`); `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:19`, `:70` | `sieve_filter.py` and the PWC path are Brief-II consumer items |
| `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7`; `docs/TODO_CONSOLIDATED.md:89` | **Brief II is BLOCKED** until Brief I is accepted. No document records that acceptance |
| `docs/drafts/pwc_backup_stage2_v1/README.md:50`, `:196-200`, open question 1; `persistent_worker_coordinator.py:330-354` (at `0caeaf9`) | Stage 2 refused offset ≠ 0 on the owner's 2026-11-20 08:30 ruling until the split landed, and asked for this worker edit |
| `docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md:27`, `:64-68` | Your Stage 2 review: Q4, the offset refusal, NO OBJECTION. You noted that the first sampled non-zero offset could abort an ordinary Optuna study. On TCP this change removes that refusal; it remains on the SSH route |
| `docs/exchange/from_beta/X-PWC-BACKUP_R1_BETA_REVIEW.md:16`, `:96` | You recorded the PWC `offset` / Brief-II work as a known status |
| `docs/CHAPTER_9_AUDIT.md:68-73` (B-1), Chapter 2 F-5 §9.5 | The ROCm hostname check in `sieve_filter.py:27` never matches the live `rrig6600*` hosts. Not changed here |
| `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120` | Gate 22 red on an uncommitted tree is transitional; no allowlist change |

**Labels:**

| Item | Label |
|---|---|
| The split on PWC's TCP and local routes (worker, `sieve_filter`, coordinator) | NEW code. It carries out v1.1 and the owner rulings below |
| Doing this Brief-II consumer item while Brief II is recorded as BLOCKED (`docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7`, "Brief II remains BLOCKED"; `docs/TODO_CONSOLIDATED.md:89`) | **CHANGES A RULING** in sequence, **ruled by the owner** on 2026-11-20 17:38 PST: the 08:30 ruling lifts the wait for PWC only (fixed fact 2 below). Brief II itself stays where the record puts it |
| Stage 2's offset ≠ 0 refusal is now on the SSH route only | CHANGES Stage 2's behaviour on TCP, as that ruling foresaw (Stage 2 README `:196`) |
| The miner's hybrid argument counts for PWC new-key jobs | ALREADY RECORDED as a gap (Stage 1 scope B3; Stage 2/3 brief 2.9; Stage 2 README "not verified"). Closed here for PWC only |
| The coordinator's echo check of `window_anchor`/`generator_phase` | NEW |
| `scripts/prove_unchanged_outside.py` (with `--allow-added`) and the two-module import proof | NEW tools |
| Test edits in `tests/test_pwc_backup_stage2.py` and `tests/test_pwc_backup_step1.py` | NEW, needed by the change |

**Not searched:** ser8, the private repository, the claude.ai project files and PDFs, the rigs (powered off), the
deployed rig bytes, Proxmox. Only Stage 2's runner files under `logs/` of the `pwc-backup` worktree were read.

---

```
id / round      X-PWC-STAGE2B / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude Code)  ->  Team Beta (Codex)
date            2026-11-20 (draft)
review class    BEFORE-COMMIT (coordinator and worker code). PWC is the backup engine and is not in use today
base_commit     0caeaf9 (main, PWC Stage 2 landed), plus the uncommitted worktree changes listed below
answers         (opening packet)
```

**The files are in a git worktree and are NOT committed.** Read them at these absolute paths.
`git -C /home/michael/tfm_worktrees/pwc-offset diff 0caeaf9` shows the tracked changes.

| File (absolute path) | Status |
|---|---|
| `/home/michael/tfm_worktrees/pwc-offset/docs/drafts/pwc_backup_stage2b_v1/README.md` | new (read it first; it has every `file:line`) |
| `/home/michael/tfm_worktrees/pwc-offset/persistent_worker_coordinator.py` | modified: `:356-417` new; `run_sieve_pass`, `_dispatch_to_tcp`, `_dispatch_local_sieve`, `run_trial_persistent` |
| `/home/michael/tfm_worktrees/pwc-offset/persistent/pwc_worker_service.py` | modified, `_execute_job` only (`:563-588`); Stage 2's exit-43 change kept |
| `/home/michael/tfm_worktrees/pwc-offset/sieve_filter.py` | modified, `execute_sieve_job` and `GPUSieve.run_hybrid_sieve` only |
| `/home/michael/tfm_worktrees/pwc-offset/tests/test_pwc_backup_stage2.py`, `.../tests/test_pwc_backup_step1.py` | modified (test edits the change needs) |
| `/home/michael/tfm_worktrees/pwc-offset/tests/test_pwc_stage2b_anchor_phase.py`, `.../tests/test_prove_unchanged_outside.py` | new |
| `/home/michael/tfm_worktrees/pwc-offset/scripts/prove_unchanged_outside.py`, `.../scripts/pwc_sieve_filter_import_proof.py` | new |

Evidence is in `/home/michael/tfm_worktrees/pwc-offset/docs/drafts/pwc_backup_stage2b_v1/`:
- `pwc_backup_stage2b.diff`, `joined_pwc_worker_service_vs_0caeaf9.diff`;
- `prove_unchanged_outside_demo.txt`, `import_proof_sieve_filter.txt`, `import_proof_persistent_worker_coordinator.txt`;
- `before_after_tally.md`, `before_after_failing_checks.txt`, and the test outputs.

Logs are in `/home/michael/tfm_worktrees/pwc-offset/logs/pwc_stage2b_v1/{before,after}/`.

**Do not touch** the rigs (off), the live miner ledger or `/home/michael/miner_staging`.

## Fixed facts: the owner's rulings (not under review)

1. **2026-11-20 08:30 PST** (relayed):
   - the trial's `offset` picks the draw window only (`window_anchor`);
   - `generator_phase` is 0;
   - PWC moves to RANGE-MINER's method.
2. **2026-11-20 17:38 PST, the owner's words, verbatim:** "Yes, the 830 lifts the PWC wait so that we can get all
   methods on the same page. Proceed". This answers the Brief-II sequence question: the recorded block
   (`docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7`; `docs/TODO_CONSOLIDATED.md:89`) is lifted for **PWC only**.
   Beta is not asked to rule on it. In the same message the owner approved this request for **one** Beta round.
3. **The same day, answering this work's questions:**
   - **(c)** the refusal of old-style `offset` jobs goes at **PWC's entry points only** (the TCP worker, and the
     coordinator, which builds only new-key jobs). `sieve_filter.execute_sieve_job` keeps accepting old-style jobs,
     because the legacy `coordinator.py` engine and ZMQ still send them;
   - **(a)** the **SSH route keeps Stage 2's refusal** of offset ≠ 0. It ends in `sieve_gpu_worker.py`, a
     gate12-pinned file that reads one `offset` for both jobs;
   - **(b)** the four forward hybrids with no phase argument get **the miner's argument counts**, for **PWC new-key
     jobs only**. Legacy and ZMQ launches are unchanged.
4. **The default rule:** PWC does what RANGE-MINER does (`docs/PWC_BACKUP_OWNER_RULINGS_20261119.md`).

Please review whether the code carries these out and whether that can harm anything on this cluster. Please do not
reopen the rulings. They exist as relayed text and in the package README.

**Open owner questions — not asked of you as rulings.** Two design choices are the owner's, for Stage 3:
- whether PWC's SSH route is removed, or kept at offset 0 only;
- whether an out-of-domain anchor is refused on VM101 before dispatch, instead of by every rig chunk.

Where Q5 and Q6 below touch them, we ask only what happens on this cluster as the code stands.

## The owner's framing (please read first)

1. **Judge only failures that can plausibly happen on this cluster:**
   - three rigs, each with 8 RX 6600 XT cards, run as Proxmox CT100s at `.122`/`.156`/`.164`;
   - VM101 (`.177`) with one 3080 Ti;
   - one operator.

   RANGE-MINER is the production engine. PWC is the backup and is not in use today.
2. **Use the AGENTS.md §5 positions table with its Severity column.** Mark an objection `LIVE` only if you can name a
   concrete sequence of events that could plausibly happen here **and** that would break a run, break RANGE-MINER or
   corrupt results. Everything else is `THEORETICAL`, a `SUGGESTION`, or a to-do for Stage 3.
3. **The owner's first concern is RANGE-MINER.** Please say directly whether any line of this package can change what
   a RANGE-MINER run (Run 4 shape, `gate12_launch.sh` / `scripts/launch_fleet_manual.sh`) does. The package's claim:
   - no file under `miner/`, none of the ten files in `scripts/gate12_parity_gate.py:174-185`, and no
     `window_optimizer*.py` is edited;
   - the miner's VM101 launch process imports two edited modules at module scope: `sieve_filter`
     (`window_optimizer_integration_final.py:52-55`, `load_draws_from_daily3` only) and `persistent_worker_coordinator`
     (`:58`). Their import is proven unchanged (Q1);
   - the miner process calls none of the changed functions. `run_trial`'s miner branch raises or returns
     (`:1761-1944`) before the PWC (`:1950`), ZMQ (`:2013`) and legacy (`:2117`) branches. `run_hybrid_sieve` is called
     only inside `execute_sieve_job` (`sieve_filter.py:652`, `:695`). `load_draws_from_daily3` is unchanged.
4. **Tests are not the yardstick.** Nothing here is judged by whether any seed is found. PWC's yardstick question, hit
   rate against `k/1000` on unseen draws, stays open.

## What changed (details in the README)

- **Coordinator:**
  - on TCP and the local Zeus route, a job carries `window_anchor` (the trial's offset) and `generator_phase` 0, and no
    `offset`, as the miner's coordinator does (`miner/range_miner_coordinator.py:2731-2732`);
  - a non-integer or negative anchor is refused before dispatch;
  - the worker's result must echo both values, or the chunk fails;
  - on the SSH route, offset ≠ 0 is still refused, as in Stage 2.
- **TCP worker:**
  - refuses any job carrying `offset` or `chunk_offset`, or lacking either new key, as the miner's worker does
    (`range_miner_worker.py:813-860`);
  - never fills in a default `offset`.
- **`sieve_filter.execute_sieve_job` (round 1):** a new-key job:
  - takes its draws at `window_anchor`, checked against `[0, N_filtered − window_size]`, never clamped;
  - sends phase 0 to the kernel;
  - refuses phase ≠ 0;
  - records both values in its result.

  An old-style job runs exactly as before.
- **`GPUSieve.run_hybrid_sieve`:** a new-key launch of `java_lcg_hybrid`/`minstd_hybrid`/`xorshift32_hybrid`/
  `xorshift128_hybrid` drops the trailing `int32`. The counts become 15/15/16/16, which equal the miner's
  `EXPECTED_KERNEL_ARITY` and each kernel's declared parameters. Every other launch is unchanged.

## Real-world scenarios we want judged

- **S1. Only the coordinator is updated:** VM101 has this package and the rigs still run the `0caeaf9` worker. The old
  worker copies a fixed key list, drops both new keys, defaults `offset` to 0 and sieves the window at anchor 0. It
  echoes neither key, so every chunk fails (`pwc_anchor_echo_error`) and the trial is marked incomplete (Stage 1 S1).
  Our claim: no silently wrong window is counted.
- **S2. Only the rigs are updated:** the new worker gets `offset` from the old coordinator and refuses every job
  (`PWC_LEGACY_OFFSET_REFUSED`). Every pass fails, loudly.
- **S3. An Optuna-sampled anchor past the end of a single-session window:** every worker refuses its chunk
  (`PWC_WINDOW_ANCHOR_OUT_OF_DOMAIN`). The pass is incomplete, with one error per chunk rather than one refusal on
  VM101.
- **S4. A PWC run on the SSH route** (`pwc_transport != "tcp"`) at anchor 25: refused before the fleet is built, as in
  Stage 2. At anchor 0 it runs as before.
- **S5. The legacy `coordinator.py` or ZMQ engine** after this lands: their `execute_sieve_job` calls and hybrid
  launches are unchanged (G-OLD-1a–d, G-HYB-3). A rig copy of `sieve_filter.py` that a legacy run uses changes only on
  redeploy.
- **S6. Retries:** an echo failure is not treated as transient (the message has neither "timeout" nor "pipe"), so it is
  not retried.

## Questions

**Q1. RANGE-MINER's launch: is the import-time behaviour of `sieve_filter.py` and `persistent_worker_coordinator.py`
unchanged, by your own check?**

Our evidence:
- `prove_unchanged_outside.py` against `0caeaf9`:
  - `sieve_filter.py`: everything outside `execute_sieve_job` and `GPUSieve.run_hybrid_sieve` is identical;
  - the coordinator: everything outside 4 changed functions is identical, and the 4 additions are three definitions
    and one constant;
  - the worker: everything outside `_execute_job` is identical.
- **Clean `python3 -I` import** of each module (CuPy recorded), base against new, real hostname and `rig-6600`:
  identical environment changes, identical CuPy calls (`set_limit(268435456)`) and identical module sets (161 and 173).

Please check this your own way. Did you find any difference that reaches the miner's process?

**Q2. Do jobs without the new keys behave exactly as before where they still exist?** Old-style jobs now exist only
outside PWC (legacy, ZMQ). G-OLD-1a–d gives identical `GPUSieve` calls and identical results for 21 jobs, base
against new; G-HYB-3 gives identical hybrid kernel arguments for all 12 variants. Inside PWC, an old-style job is now
refused by design (G-OLD-2, 22 jobs). Is there an old-style job on this cluster that reaches PWC's TCP worker from
anything but the PWC coordinator?

**Q3. Is the `generator_phase != 0` refusal correct?** It is refused in three places, all before any sieve work:
- the coordinator never sends anything but 0;
- `execute_sieve_job` refuses ≠ 0 (DEP-ABI-V2 named);
- `run_hybrid_sieve` with `miner_abi=True` refuses ≠ 0 for the four no-phase hybrids, as `miner/range_miner_worker.py:570-600` does.

Is there a job here where refusing is wrong rather than safe?

**Q4. The kernel argument change.** For PWC new-key jobs, the four no-phase forward hybrids lose their trailing
`int32`. We measured the lists on the real `run_hybrid_sieve` with CuPy replaced, not on a GPU. Do you see any way 15
arguments for `java_lcg_hybrid` on HIP (the rigs) or CUDA (VM101) gives a different result from the old 16, other than
the old one being wrong?

**Q5. S1 and S2.** When only one side is deployed, every chunk (S1) or every job (S2) fails. Is the "incomplete" mark
enough to stop that trial being counted today, given that nobody reads it yet (Stage 3)? Name any sequence on this
cluster where a half-deployed fleet still produces a counted result.

**Q6. S3, as the code stands.** An out-of-domain anchor is refused by every rig chunk. The coordinator does not hold
the dataset length today. Name any concrete sequence on this cluster in which that per-chunk refusal leads to a wrong
or counted result, rather than an incomplete trial. Whether to add a VM101-side check is an open owner question.

**Q7. The test edits.** `G-OFFSET` became `G-ANCHOR`. The Stage 1 gate `G-S2-ZERO-BYTE-EQUAL` now maps a new-key job
back to `offset = window_anchor`, and only when the phase is 0. Do either of these weaken what the gates guarded?

**Q8. The Stage 3 proof tool.** What import-time or call-time behaviour change would
`scripts/prove_unchanged_outside.py` (with `--allow-added`) **not** report? Known limits:
- it ignores comments and formatting (`--strict-bytes` reports them);
- it says nothing about other files the module imports.

**Q9. RANGE-MINER.** See framing point 3. Name any path by which this package changes a RANGE-MINER run, or say that
you found none.

## Test evidence

- **Focused:**
  - `tests/test_pwc_stage2b_anchor_phase.py`: 35/35;
  - `tests/test_pwc_backup_stage2.py`: 26/26 checks, 33/33 mutants (6 new or re-aimed);
  - `tests/test_pwc_backup_step1.py`: 22/22 checks, 16/16 mutants;
  - `tests/test_prove_unchanged_outside.py`: 29/29;
  - five deliberate breakages of the new worker and `sieve_filter` code were each caught.
- **Full before (clean `0caeaf9`) and after (first after-pass).** Stage 2's 73 suites, one at a time, CPU only, ssh stubbed, never
  `tests/test_s172_staging_partb.py`. **70 of 73 are identical. 3 differ:**
  - `test_pwc_backup_step1.py`: `G-S2-ZERO-BYTE-EQUAL`, from the intended job change. **The fix was made after the
    full after-pass.** It touches only `tests/test_pwc_backup_step1.py`: `_jobs_without_stage2_keys` now passes each
    job through a new `_stage2b_as_offset`, which maps a new-key job back to `offset = window_anchor` only when
    `generator_phase` is 0. The suite re-run alone is green;
  - `test_s172_phase4_coordinator.py`: Gate 22, which names exactly this package's uncommitted `.py` files
    (transitional);
  - `test_s172_attempt6_remediation.py`: one extra timing failure. Re-run alone twice, it matched "before" both times;
    it imports no PWC file.

  Both new suites pass. `~/miner_staging`'s mtime, the main tree and the optuna listing were unchanged.
  `test_s172_admission_stall.py` timed out before and after, as it did in Stage 2's passes.
- **A fresh full after-pass on the final code, including the fix, is being run.** It starts as soon as this relay
  returns, so it does not write into the tree while you review. Its tally goes to the owner and into the package
  README. Until then, the full-pass result for the code exactly as you read it is **UNAVAILABLE** except for the
  suite re-run above.

## What is not verified

- Any GPU run. Real CuPy/HIP with the new argument counts.
- The rigs and their deployed bytes (the rigs are off).
- Who reads the "incomplete" mark (Stage 3).
- The fresh full after-pass on the final code (running after this relay; see Test evidence).
- The owner's answers of this session exist only as relayed text and the README.
