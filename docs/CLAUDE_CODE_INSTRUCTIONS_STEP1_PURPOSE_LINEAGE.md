# CLAUDE CODE INSTRUCTIONS — WHAT IS STEP 1 *FOR*? (READ-ONLY, HISTORICAL)

**Host:** VM101, repo `~/distributed_prng_analysis`.

## CONSTRAINT — READ-ONLY. NO LAUNCHING.

Pipeline runs are MICHAEL-INITIATED ONLY, and Beta has explicitly NOT authorized any further
Gate-12 run. Do not start `watcher_agent.py`, `window_optimizer.py`, the fleet script, any
worker, or bind 5700. No commits, no production edits. Permitted: reading files, full git
history (`log`, `log -S`, `log --follow`, `show`, `diff`), read-only DB reads, and writing your
report.

**Search order is binding: governance trail → chapters → code.** `docs/PROJECT_FILE_CATALOG.md`
is intent-indexed; read it before any absence claim.

## Why this is being asked

The owner is re-establishing Step 1's *purpose* before choosing a seed geometry for the
Gate-12 relaunch. Alpha has been reasoning from assumptions and has already been wrong twice
in this area (an unfounded "Optuna vs sweep" framing, and a refuted Fantasy 5 hypothesis).
**Do not inherit Alpha's framing. Establish what the repo says.**

Owner's working description, offered as the thing to CHECK, not as ground truth:

> "For each draw, Step 1 goes through the space, discovers the seed(s), which are sent to the
> rest of the pipeline — fingerprints are extracted, ML learns, prediction pools are built, ML
> finds which seeds are relevant; then at some point if ML decides a regime change, it re-runs
> the seed space — or something like that."

The owner is explicitly uncertain about the last clause.

## The three questions

### Q1 — What did Step 1 do BEFORE the persistent-worker transports (PWC SSH / TCP / ZMQ)?

Establish Step 1's originally stated purpose from the earliest authoritative sources — the
governance trail, `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` §1 Overview and §2 Architecture,
`docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`, the whitepaper, and the earliest commits touching
`window_optimizer.py`. Quote the stated goal verbatim with `file:line` and commit hash.

Specifically resolve, from sources rather than inference:
- Was the deliverable **an optimal window configuration**, **a set of surviving seeds**, or
  both — and which was primary?
- Was the seed range **swept exhaustively** or **sampled**, and was that per-draw or global?

### Q2 — Did the goal CHANGE when the project pivoted to PWC (SSH → TCP → ZMQ)?

These were transport/execution changes. Determine whether they were *purely* mechanical or
whether Step 1's stated deliverable, scope, or semantics also moved. Use `git log --follow` on
the Step-1 files across that period, the session changelogs, and any TB ruling. If the goal
changed, name the commit and the governing document; if it did not, say so with evidence.

### Q3 — Did the goal CARRY OVER into RANGE-MINER, or was some of it LOST?

RANGE-MINER is a standalone Step-1 backend behind one flag. Compare what Step 1 is documented
to produce against what the miner path actually produces today. Look specifically for
capability that exists in the PWC path and is absent or degraded in the miner path — the
throughput-baseline investigation already found one such case (`StripeCompleteMessage.elapsed_s`
is computed and sent by the worker, then dropped at the ledger boundary,
`range_miner_coordinator.py:5903-5905`). Are there others of that shape?

### Q4 — Two specific sub-questions the owner's description raises

a. **Is a sweep per-draw or global?** The legacy tracker keys coverage on `prng_type` ALONE
   (`database_system.py`, `get_next_seed_start`) — no draw, no dataset. If Step 1's purpose is
   per-draw seed discovery, that key is wrong. Determine what the DOCUMENTED intent is, and
   whether survivors are meant to carry across draws (S145-R1 approved "cross-session survivor
   accumulation" and "merge by best per-seed score") or be re-derived per draw. Note whether
   the sieve window is **anchored** (grows as draws are added — survivors filter down, reuse
   valid) or **sliding** (Optuna tunes `offset`, so old survivors may not be valid) — this
   determines which reading can be true.
b. **Does a regime-change re-sweep trigger exist?** The owner is unsure. Search WATCHER,
   Chapter 13, self-play, and the ML steps for any mechanism that re-runs Step 1 on a detected
   regime change. If none exists, say **"no evidence found"** — an honest absence is the
   answer here, and this is exactly the class of claim where the governance trail must be
   searched before the code.

## Report

`docs/CLAUDE_CODE_REPORT_STEP1_PURPOSE_LINEAGE.md`. For each question: the answer, verbatim
quotes with `file:line`, commit hashes for any change of intent, and an explicit **"no
evidence found"** where that is the truth. Distinguish throughout between (i) documented
intent, (ii) what was implemented, and (iii) what runs today — where those three diverge, say
so plainly, because that divergence is the actual deliverable of this task. Do not propose or
implement anything.
