# CLAUDE CODE INSTRUCTIONS — STEP 1 RERUN POLICY AND WHAT COVERAGE RECORDS, STAGE A (READ-ONLY) — 2026-11-13

## Prior record

**Searches run** (public mirror at `5e100b3`; VM101 was not read by Alpha):

- full read of `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` and of its request
- read of the four records Beta listed as missed, at the lines Beta cited
- `grep -n "^## \|^### " docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md`, then read of §1, §3.2–§3.4, §4
- `grep -n "human-gated\|regime" docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md`, then read of `:45-62`, `:586-613`
- source read: `utils/seed_coverage_ledger.py:271-320`, `:606-626`, `:880-890`, `:1000-1055`; `agents/watcher_agent.py:2027-2101`

**Documents found that already decide or record the subject:**

| Document | What it says | Label |
|---|---|---|
| Beta ruling 2026-08-07, as quoted in `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_AMENDMENT.md` §1–§7 | Domain `[0, 2^32)`; append-only ledger; first-gap cursor, so a run picks up at the first uncovered seed; COMPLETE generates no run | ALREADY RECORDED |
| Beta ruling 2026-08-08, `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_R1.md:7-16` | "APPROVED AND CLOSED": `dataset_sha256` is provenance, **not** a v1 partition key | ALREADY RECORDED. Any design that makes the cursor test the dataset CHANGES A RULING |
| `utils/seed_coverage_ledger.py:606-626`, `:880-890` | Each record stores run id, study identity, base, modes, range, dataset hash, commit, artifact and generation. Study identity is "never consulted by the cursor" | ALREADY RECORDED |
| `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md` §1, §3.2–§3.4, §4 | The space is swept in **four sessions** of 1,073,741,824 seeds, 50 trials each; one Optuna study resumed across sessions by `study_name`; survivors accumulate. "Step 1 retired permanently" **rejected** (`:22`) | ALREADY RECORDED |
| `docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md:45-62`, §5 (`:586-613`); `docs/TFM_PROJECT_FACTS_SKILL.md:1128-1136` | Chapter 13 documents Step 1 as "re-invoke only on regime shift"; the rerun exists, is human-gated at three points, and enters through the same frontier, so it sweeps the next uncovered block and not the range already searched | ALREADY RECORDED (2026-08-08) |
| `docs/TFM_PROJECT_FACTS_SKILL.md` §2.10b | Results are not valid on the next dataset version; re-optimisation is per-session | ALREADY RECORDED |
| `docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_FOLLOWUP_REPAIRS_20261111.md:57-78` and its report `:123-153` | COMPLETE is a clean stop that advances nothing | ALREADY RECORDED |
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:12-24` | The Route-A certificate and the two privileged seams | ALREADY RECORDED |
| `README.md:5`; `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` §9; `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md:33`, `:681-683`, `:1389-1400`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` §1 | Automation is the stated purpose ("autonomous agent orchestration"; "close the autonomy loop"). Chapter 13 v1 requires human approval for trigger execution as a safety gate and lists automatic execution as a deferred extension. Owner, 2026-11-11: "Agents can launch. That is the whole point. We need guard rails." | ALREADY RECORDED |
| `docs/TB_RULING_S179_IMPLEMENTATION_AUTH.md` §8 | Governance starts in BOOTSTRAP: triggers audit-only, no autonomous retraining enabled | ALREADY RECORDED. Bears on R5 |
| `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` Q2, Q3 | LIVE objection: completing the second half would refuse the soak and later Step 1 runs. Suggested: keep the history ledger; add an explicit rerun policy bound to dataset, session, search configuration and a search identity. Order: settle this first | ALREADY RECORDED (2026-11-13) |

**What is NEW:** nothing in this brief is a finding. It asks for the map that a design needs.
**Not searched:** VM101, ser8, the private repository, the project PDFs.

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus · **Base commit:** `5e100b3`
**Review class:** document (Beta after commit). The design that follows it changes WATCHER
and the coverage ledger and goes to Beta before any code.

## The owner's requirements (Michael, 2026-11-13). These are the design. The code is measured against them.

- **R1 — Pick up where an incomplete search left off,** so nothing is searched twice by accident.
- **R2 — Cover the space in manageable chunks across sessions,** as the four-session plan set out.
- **R3 — Track WHAT was run,** not only which seeds: the dataset version, the session, the search settings.
- **R4 — Be able to run again** when new draws come in, and in like cases. Retiring Step 1 for good goes against the learning part of TFM.
- **R5 — It is meant to be automated.** This was the intent from the start and is a prime directive of the code and of TFM in general. Picking up, chunking, recording and running again must be able to happen without a person typing the steps, inside the guard rails. A step that needs a person today is either a deliberate gate with a ruling behind it, or a gap.

Where code written by different agents over the past year contradicts these, the
requirements govern. Report the contradiction as a fact; do not defend either side.

## 0. Rules for this session

- **Read-only. You edit no existing file.** No commit, push, branch or tag.
- **Work in a worktree:** `git worktree add --detach /home/michael/tfm_worktrees/step1_rerun_a HEAD`.
  Write only `docs/CLAUDE_CODE_REPORT_STEP1_RERUN_POLICY_STAGE_A.md` and scratch under
  `data/step1_rerun_a/`. The report opens with its own `## Prior record` section (`CLAUDE.md`).
- **SQLite:** follow `docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md` §A
  exactly. Never construct `CoverageLedger` or the database wrapper against an original.
- **Every file is no-touch for this session.** It launches nothing: no WATCHER, optimizer,
  coordinator, finalizer or gate that writes. No fleet, no GPU, no rig SSH.
- **First step:** record `pgrep -af "beta_relay|range_miner_coordinator|watcher_agent|window_optimizer"`,
  `git status --porcelain` and `git worktree list`. If a relay is running, wait for it.
- Run Python with `PYTHONDONTWRITEBYTECODE=1`. Absence claims carry their search.

## 1. Questions

**Q1 — Who reads the cursor.** Every consumer of the certified cursor and of COMPLETE, with
`file:line`: WATCHER, `run_pipeline`, the database wrapper, the finalizer's ledger write, the
launch harness, tests and fixtures. For each: what it does on OPEN and on COMPLETE.

**Q2 — What is recorded against R3.** For each of dataset version, session, window and
threshold settings, skip settings, study identity, run id and trial count: is it stored in the
coverage record, in the generation sidecar, in the miner ledger's trial context, or nowhere?
Which of these does the cursor test?

**Q3 — R1 and R2 as built.** The three ways a search continues today: within one run
(checkpoint and resume), across runs (the cursor), and across runs for Optuna (`study_name`).
Which work on the RANGE-MINER path today? Where is the chunk size set, and what stops a
session of 2³⁰ seeds as in the four-session plan? What does a run that dies part-way leave
in the ledger, and where does the next run start?

**Q4 — The regime-shift rerun.** Trace the documented chain from Chapter 13 to a new Step 1
run: the trigger, the three human gates, and how it reaches the cursor. State exactly what it
would do today with the ledger OPEN at 2³¹, and with the family COMPLETE.

**Q5 — Publishing a second search of the same seeds.** If a new Step 1 search covered seeds
already certified, how would the finalizer and the generation chain treat it: the merge with
the prior generation, the parent identity, the input-identity wall, and the ledger write for an
interval that overlaps an existing record? Read only; run nothing.

**Q6 — New draws.** When a new dataset version is published, what in Step 1's path changes
and what does not: dataset authority, residue window, cursor, generation lineage?

**Q7 — The soak and Run 4.** What each needs from the cursor in order to start.

**Q8 — Every manual step and human gate (R5).** Walk the Step 1 path end to end as it would
run unattended: a new dataset version arrives, a search is chosen, it starts, it is chunked,
it dies and resumes, it publishes, the next chunk starts, a rerun is called for. List every
point where a person must act today. For each: is it a deliberate gate (cite the ruling, and
what that ruling says must be shown before it can be lifted), or a gap that no ruling asked for?

**Q9 — Options. Do not implement.** At least three, including "change nothing and document
it". For each: how it meets R1 to R5, and which human steps remain and why; which files it edits; which rulings it changes (name
them); which suites and gates it affects (the seed-domain suite, the Route-A routing suite,
Gate 22, the rig parity set); what COMPLETE and the plan's condition F4 would mean under it.
Beta's keep-list applies to every option: domain bounds, publication before credit,
append-only history, fail-closed lookup, no reset or deletion of old credit.

## 2. Deliverable and stop gate

`docs/CLAUDE_CODE_REPORT_STEP1_RERUN_POLICY_STAGE_A.md`: `## Prior record`; first-step output;
a ten-line summary; Q1–Q9 with anchors read in this session; everything `UNAVAILABLE`;
commands used. Stop when it is written. Tell Michael the path. Alpha writes the amendment
from it, and Beta reviews that before any code.
