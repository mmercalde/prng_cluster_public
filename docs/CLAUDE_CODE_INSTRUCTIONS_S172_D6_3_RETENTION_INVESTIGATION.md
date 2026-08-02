# CLAUDE_CODE_INSTRUCTIONS_S172_D6_3_RETENTION_INVESTIGATION.md — REV1

**S172 — D6.3 phase 1: checkpoint retention. READ-ONLY INVESTIGATION.**

**Base:** HEAD `285cbd7`. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`.

**This brief authorises NO fix, NO retention policy, and NO deletion of anything.** It answers
questions. The design brief follows Beta's ruling on the D6.2 identity addendum.

**Run this AFTER the process_sharded import gate**, not alongside it.

---

## 0. Why an investigation rather than an implementation

Beta's D6.3 constraint: *never remove active, unresolved or audit-retained state merely for
exceeding an age or count threshold.*

Satisfying that needs a **resolution signal** — a way to tell a superseded checkpoint from one
belonging to a run that died mid-flight. Alpha's repo-scoped trace says no such signal exists,
because the checkpoint directory and the published generation use two unrelated run identities
(see `docs/TEAM_ALPHA_D6_2_IDENTITY_ADDENDUM.md`).

**That trace is repo-scoped and therefore incomplete.** The repository is not the system. This
brief closes the host-state half before any design is written.

**Delete nothing. If a question is unanswerable, report `UNAVAILABLE` — never guess.**

---

## Q1 — does anything on this host set `PRNG_CHECKPOINT_RUN_ID` or `PRNG_CHECKPOINT_ROOT`?

Alpha found **only test harnesses** setting either, in the tracked repo. That claim is
**[UNVERIFIED]** against host state and Q1 settles it.

Search, and **name every surface you searched and every one you could not reach**:
- systemd units (system and user), including `daily3scraper.service` and anything WATCHER-adjacent
- cron and any `*.env`, profile, or shell wrapper in `michael`'s environment
- the environment WATCHER actually launches Step 1 with — **the process environment, not the unit
  file**, if a run can be observed
- **uncommitted or untracked files in the working tree** (`git status --porcelain`,
  `git stash list`) — a local edit is invisible to a clone
- the rigs, if the variable could be set worker-side

**If a setter exists, report its value and where it comes from.** That changes the finding
materially and Alpha wants to know before designing anything.

## Q2 — what is actually on disk right now?

Read-only. For the checkpoint root (`PRNG_CHECKPOINT_ROOT` if set, else the directory containing
`window_optimizer_integration_final.py`):

- how many `.s172_checkpoint/<run_id>/` directories exist
- total bytes, largest, oldest and newest by mtime
- the **naming pattern actually observed** — does it match `hostname-pid-epoch`, or something else?
- how many contain a **complete** member pair vs. a partial or orphaned pair
- **do any temporary artifacts survive?** `_CHECKPOINT_TMP_SUFFIX` is `.flush-{pid}.tmp`; leftovers
  mean a crash between write and replace, and their count is a real signal
- read each pair's identity block and report `checkpoint_schema_version`, `checkpoint_sequence`,
  `run_id`, `logical_candidate_count`

**Do this with `numpy.load(..., allow_pickle=False)` and treat every file as untrusted input.**
Report unreadable members rather than skipping them silently.

Same census for `dataset_provenance/*.json` — BACKLOG §7 records it as the same class, and it is
cheap to measure while you are here.

## Q3 — what is the growth rate?

From Q2's mtimes and sizes, estimate bytes per run and runs per day under recent activity.

Then state what a **50-trial, 26-GPU Phase-7 soak** would add. This is the number that decides
whether D6.3 is genuinely a Phase-7 blocker or a slower-burning issue, and right now nobody has
computed it. **Show the arithmetic and the assumptions**; a stated assumption Alpha can challenge
is worth more than a confident total.

## Q4 — can a checkpoint be joined to its generation by any means available today?

Alpha's claim is no. **Try to falsify it.**

Given a checkpoint directory and the set of published generations under `.s172_accumulator/`,
is there **any** reliable join — a recorded path, a timestamp correlation tight enough to be
sound, a log line, a sidecar, anything?

**A timestamp correlation that is merely usually right is not a join.** If the only available
answer is heuristic, say so explicitly — that is a finding, not a failure, and it is the finding
that determines whether D6.2 must carry the identity link.

## Q5 — what does the finalizer already know about a run's completion?

Read `utils/run_finalizer.py:1546-1700` and report which identity values are in scope at
`finalize_run` and which are **persisted** where a later process could read them
(`generation_id`, `run_id`, `artifact_sha256`, parent generation, and the manifest that carries
them).

The question this answers: **if D6.2 records `canonical_run_id`, does the reverse lookup
"generation → was there a checkpoint" also become possible, or only the forward one?**

---

## Non-goals — explicit

- Do **not** write a retention policy, a pruner, a cleanup script, or a `--purge` flag.
- Do **not** delete, move, compress or rename **any** checkpoint, provenance file or generation.
- Do **not** modify production code.
- Do **not** set either environment variable, even to test — that would create a directory whose
  provenance the next investigator has to untangle.

## Report

Write `docs/S172_D6_3_RETENTION_INVESTIGATION_REPORT.md` answering Q1–Q5 in order, each with its
evidence and its **searched / unavailable** surfaces. Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** every census command's raw output is quoted, not summarised into a number
  with no provenance.
- **clean control:** not applicable — read-only investigation, no detector being validated.
- **fault-injection control:** not applicable, same reason. **Say `NOT_APPLICABLE`, not `PASS`.**
- **completion sentinel:** the report terminates in `PASS | FAIL | UNAVAILABLE | INCOMPLETE`. A
  question you could not answer makes the report `INCOMPLETE` for that question — which is a
  perfectly good outcome and far better than a plausible guess.
- **unavailable-observer behavior:** with rigs down, any rig-side surface in Q1 is `UNAVAILABLE`,
  never "no setter found." *"We needed it and could not get it" is not "we did not need it."*
- **audit claim scope:** host state on VM101 plus the tracked repo at `285cbd7`.
- **searched surfaces:** to be enumerated in the report — this is the point of the brief.
- **unavailable surfaces:** to be enumerated in the report, explicitly, including the rigs if down.
