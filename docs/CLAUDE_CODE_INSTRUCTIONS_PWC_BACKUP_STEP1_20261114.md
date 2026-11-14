# CLAUDE CODE INSTRUCTIONS — PWC BACKUP, STEP 1 (PWC's own files only, before Run 4)

## Prior record

**Searches run:** the full governance search for PWC is in
`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md` (`18e4b61`) "Prior record", and is carried
here by reference. Alpha read that report's P3 table in full and its section list
(public mirror, HEAD `18e4b61`, 2026-11-14), and
`docs/SESSION_CHANGELOG_20260425_S170.md:41-62`, `:187-227`. Alpha did **not** read P1, P2,
P4–P7 line by line. **Read the whole report before you start.**

**Documents that already decide this subject:**

| Document | What it settles | Label |
|---|---|---|
| `docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STAGE_A.md` P3 | The gaps, with evidence, the files each fix touches, and which are shared with RANGE-MINER or in the rig parity set | ALREADY RECORDED. This brief builds the subset that touches PWC's own files only |
| Same report, P1 | Beta removed PWC's certifying authority (2026-07-31; the ruling text is not in the repository). The hybrid path is quarantined | ALREADY RECORDED. **Not changed here.** PWC still earns no credit and its hybrid path still raises after this brief |
| Same report, P4 | The operating history. Clean records sit at 6 workers per rig with chunks up to 100K, or 8 workers with chunks up to 25K | ALREADY RECORDED. The source of the envelope in item S5 |
| `docs/SESSION_CHANGELOG_20260425_S170.md:215-227` | Runs in `--config-file` mode silently fell back to the legacy SSH path and to 2,000,000-seed chunks | ALREADY RECORDED. Why a silent default is treated as a defect here |
| `docs/exchange/from_beta/X-STEP1-SEARCH-CODE_R1_BETA_REVIEW.md` Q1 | PWC does not carry completion accounting to the trial | ALREADY RECORDED. Item S1 |
| `docs/CLAUDE_CODE_INSTRUCTIONS_MINER_SEED_START_20261114.md` R5 | The refusal of a non-zero start on PWC and ZMQ is being built by another session, in `window_optimizer.py` / the integration file | ALREADY RECORDED. Not touched here |

**Items this brief raises:**

| Item | Label |
|---|---|
| **Owner, 2026-11-14:** PWC is to become a working backup that runs both sieves at the settings that were stable. "RANGE MINER takes TOP priority"; the PWC work must not come at its expense. The rigs can be powered on at short notice for a test | NEW owner direction, recorded here. Making a PWC result count CHANGES A RULING and is decided by the owner after Run 4, not by this brief |
| The split: step 1 is everything confined to PWC's own files; step 2 (after Run 4) is everything that touches a file RANGE-MINER loads or Run 4's launch files | NEW. Alpha's scoping, from the report's P5 |
| A PWC parity gate for the rig-side PWC files (report gap 6b) | NEW. Built here as a new script |

**Not searched:** VM101's live state, what is deployed on the rigs, ser8 archives, the
private repository, the project PDFs.

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** record `git rev-parse HEAD` in your first step.
**Review class:** BEFORE-COMMIT, through the relay. **Report:**
`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md`, in your worktree.

**The yardstick (CLAUDE.md).** TFM is judged by prediction-pool hit rate against `k/1000` on
unseen draws. A backup engine is judged by whether the later steps can use its output the
same way. Not by matching another engine's numbers, and never by state recovery.

## 0. Rules for this session

- **RANGE-MINER comes first.** Another session is fixing the miner's seed start in
  `/home/michael/tfm_worktrees/miner-start`. Leave that directory alone. If its relay is
  running (`pgrep -af beta_relay.py`), you wait, however long it takes. If anything you
  need conflicts with that work, the miner work wins: record it and move on.
- **You may edit only:** `persistent_worker_coordinator.py`, files under `persistent/`,
  new files under `tests/`, and one new script `scripts/pwc_parity_gate.py`.
- **Everything else is not edited.** In particular: `sieve_gpu_worker.py`, `sieve_filter.py`,
  `prng_registry.py`, everything under `miner/`, `window_optimizer.py`,
  `window_optimizer_integration_final.py`, `window_optimizer_bayesian.py`,
  `agents/watcher_agent.py`, `agent_manifests/`, `scripts/gate12_parity_gate.py`,
  `utils/`, `gate12_launch.sh`. When a fix needs one of these, write down exactly what is
  needed in the report's "step 2" list and keep going. That is not a stop.
- **No commit, push, branch or tag.** **Worktree:**
  `git worktree add --detach /home/michael/tfm_worktrees/pwc-step1 HEAD` (hyphen, not
  underscore). Write only inside it, except the review request file.
- **No fleet, no GPU, no SSH to any rig, no engine run.** CPU only, with fake workers or
  intercepted transport. No writes to live state.
- **Do not lift or weaken the hybrid quarantine.** Do not make PWC earn credit.
- Use your judgement on small things and keep going. Stop only at a stop gate (§5).
- **First step:** `pgrep -af beta_relay.py`; `git rev-parse HEAD`; `git status --porcelain` in
  the main tree; create the worktree; activate the venv; read the Stage A report in full;
  baseline of the existing PWC tests you will re-run.

## 1. What to build

| # | Item (report gap) | Requirement |
|---|---|---|
| S1 | Completion accounting (1) | A pass's chunk accounting (dispatched, completed, failed, timed out) is carried into what the trial builder returns, per pass and per direction. A pass with any failed or missing chunk is marked incomplete; it is never returned looking complete. The return shape stays backward compatible for the existing caller: add fields, remove none. Say in the report exactly what the integration file would read in step 2 |
| S2 | Seed start (2) | `run_trial_persistent` accepts a `seed_start` (default 0) and partitions `[seed_start, seed_start + total_seeds)`. With the default, the chunks are identical to today's. Check `0 ≤ seed_start` and the end `≤ 2³²`. The caller is not changed here |
| S3 | Coordinator address (6a) | The address the rig workers dial is no longer the literal `192.168.3.127` in `persistent_worker_coordinator.py:756` and `persistent/pwc_worker_service.py:759`. It comes from one explicit setting, with no silent default to an address: unset refuses, naming the setting. Report where the value should come from on VM101 (`192.168.3.177`) without writing a config outside your allowed files |
| S4 | No silent fallback | Any place in the PWC file where a missing setting falls back silently to a legacy transport or to a default chunk size (the S170 class): make it refuse or log loudly at the PWC layer. List each place you found |
| S5 | The stable envelope (6c) | From the report's P4 only, encode the recorded stable envelope (workers per rig against chunk size) as one named table in the PWC file. A PWC run whose effective workers-per-rig and chunk size fall outside it refuses before any worker is spawned, naming both values and the table. No override flag. If P4 does not support a clean table, encode the most conservative recorded point and say so |
| S6 | Directional thresholds, constant passes (3, 5b) | For the constant passes, record requested, payload and effective threshold per direction, and refuse when they disagree. Leave the hybrid call sites raising as they do now |
| S7 | The import error (6f) | `persistent/pwc_protocol.py` imports cleanly. Change nothing else in it |
| S8 | PWC parity gate (6b) | `scripts/pwc_parity_gate.py`: a separate gate, in the manner of `scripts/gate12_parity_gate.py` (read it, import nothing that would change it), that compares the rig-side PWC files (`persistent/pwc_worker_service.py`, `persistent/pwc_transport_tcp.py`, `sieve_filter.py`, and whatever else the rig executes on the PWC route) between the local tree and each rig. Unreachable is REFUSE. Injectable runner; tested on CPU; never run against a rig in this session |
| Note | Launch-per-trial (6d) and `HSA_OVERRIDE_GFX_VERSION` (6g) | Not changed. Write a half-page design note in the report on keeping PWC workers alive across the trials of one run: what it would take inside the PWC file, and whether it removes the launch-storm shape |

**Tests** (new files, CPU, fake workers; each refusal with a mutant shown red): S1 with a
failed chunk, a timed-out chunk and a clean pass; S2 at 0 (byte-equal chunks to today),
at `2³¹`, and at the endpoints; S3 unset and set; S5 inside, on the edge and outside the
envelope; S6 agreement and disagreement; S7 import; S8 equal, one file differing, a rig
unreachable.

## 2. Report

`docs/CLAUDE_CODE_REPORT_PWC_BACKUP_STEP1.md`, opening with `## Prior record`. At the top,
for the owner, in plain words: what PWC can do after this that it could not do before; what
it still cannot do; and **the step 2 list**: every change that needs a file outside your
allowed list, with the file, the lines and the reason, ordered so that step 2 can be
briefed directly. Then: first-step output; each item with `file:line`; tests with counts and
mutants; the design note; Beta's last position per question; what was not verified; the
landing list with sha256.

## 3. A fleet test, for the owner to decide

Do not run one. Write, at the end of the report, the smallest real PWC test that would show
step 1 works on the rigs: constant-skip only, inside the envelope, the exact command, what
it must show, and how long it should take. The owner decides whether and when to run it.

## 4. Review by Beta, carried by this session

As `docs/exchange/RELAY_PROTOCOL.md` §2. Request file in the main tree:
`docs/exchange/to_beta/X-PWC-BACKUP_R1_ALPHA_REVIEW_REQUEST_pwc_step1.md`, opening with
`## Prior record`, review class BEFORE-COMMIT, files by absolute worktree path. Before
writing it, check `pgrep -af beta_relay.py` and wait if a relay runs. At most three rounds.

**Questions for Beta (use as written):**

1. Can a PWC pass with a failed or missing chunk still be returned looking complete?
2. Is a run at `seed_start = 0` unchanged, and is a non-zero start partitioned correctly?
3. The address and the no-silent-fallback changes: any path left that falls back quietly?
4. The envelope: does the table say only what the record supports, and can a run outside
   it start?
5. Was anything changed outside the allowed files, or anything RANGE-MINER loads?
6. Is the hybrid quarantine intact, and does PWC still earn no credit?
7. The PWC parity gate: does every failure end in REFUSE?
8. Any governing record this work missed.

## 5. Stop gates

Stop and report when: an item cannot be done without editing a file outside the allowed
list **and** leaving it for step 2 would make the rest wrong; a test can only pass by
writing to live state; Beta holds a LIVE objection you disagree with or whose fix is
outside scope; the third round ends.
