# X-RUN4-RANGE · R1 · REVIEW_REQUEST · Run 4's seed range, and Step 1 after a family is COMPLETE

## Prior record

**Searches run** (public mirror at `a41ac06`; nothing on VM101 was read by Alpha):

- `grep -niE "cursor|seed.?coverage|S145|soak" docs/PROJECT_FILE_CATALOG.md`
- `ls docs | grep -iE "SEED_DOMAIN|SEED_COVERAGE|S145"`, then read of the amendment brief §1–§7
- `grep -niE "seed_start|seed range|cursor|repeat|second half|A/B" docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md`, then read of proposal §3 and ruling `:8-40`, `:140-182`
- `grep -rlE "SEED_DOMAIN_COMPLETE|seed_domain_complete|domain is exhausted" docs`, then grep of those six files for geometry, window, Optuna, trial, reset and lineage
- `grep -rniE "SEED_DOMAIN_COMPLETE|after complete|another geometry|per-geometry|re-?sweep" docs/BACKLOG.md docs/TODO_CONSOLIDATED.md docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md`
- `grep -niE "seed_start|max_seeds|cursor|coverage|window_trials|n_parallel" docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md`
- source read: `agents/watcher_agent.py:2027-2101`, `utils/seed_coverage_ledger.py:271-320`, `:1000-1055`
- full read of `docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md` (`2930a37`)

**Documents found that already decide or record the subject:**

| Document | What it says |
|---|---|
| Beta ruling *"S145 / Seed-Domain Sweep Terminus and Coverage Authority"* (2026-08-07), as quoted in `docs/CLAUDE_CODE_INSTRUCTIONS_SEED_DOMAIN_CURSOR_AMENDMENT.md` §1–§7. **The ruling text was not found as its own file** | Governed domain `[0, 2^32)`; first-gap certified cursor; when the union covers the domain, `status = COMPLETE`, no run is generated |
| `utils/seed_coverage_ledger.py:271-320`, `:1000-1055` | Coverage identity is base family plus executed-mode set. A record qualifies when `prng_base` matches and requested modes ⊆ executed modes. Window, thresholds, session and dataset are not part of the test |
| `agents/watcher_agent.py:2068-2089` | COMPLETE returns `seed_domain_complete` and generates no Step 1 run; otherwise `seed_start` is overwritten with the cursor whenever the cursor is above 0 |
| `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:22`, §1 | "Step 1 retired permanently" was **rejected**; the sweep is progressive, across sessions, with accumulated survivors |
| `docs/TFM_PROJECT_FACTS_SKILL.md` §2.10b (TB rulings 2026-07-30/31) | "Append-only does NOT make prior scores valid on the next version": adding a draw changes windows, eligibility and features. Production re-optimisation is per-session |
| `docs/S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` §3 | Run 4 is a controlled comparison with attempt 9; "geometry, dataset, seed domain, stripe size, fleet, worker pool, backend" held |
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:10`, `:16`, `:17`, `:148` | Routes B, C and D rejected for Run 4; geometry approved as "full 2³¹, attempt-9 stripes"; claim boundary: volume, plumbing, completion, production shape, not population equivalence |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §2 (F4), `:61`, §3.A row A3, §3.C | Finished = COMPLETE over `[0, 2^32)`; the second half "is one more run of the attempt-9 shape"; A3 = "decide whether Run 4 repeats `[0, 2³¹)` or takes the second half"; the 50-trial soak comes **after** finished |
| `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:9`, `:114-115` | The soak is one WATCHER Step 1 run of 50 trials at `n_parallel = 1` |
| `docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md` R1–R4 | The live cursor is OPEN at 2147483648; Run 4 as committed would sweep `[2³¹, 2³²)`; attempt 9 published 0 rows |

**Labels for what this request raises:**

| Item | Label |
|---|---|
| Run 4 will run over the second half unless something changes | ALREADY RECORDED as open item A3; now measured |
| Whether the approved comparison needs attempt 9's seed **interval** or only its **shape** | NEW (the two texts above do not settle it) |
| After COMPLETE, no Step 1 run can be generated for that family at any window, which would refuse the soak the plan places after "finished" | NEW — not found in any document searched. If it requires a change to the coverage identity or the cursor law it CHANGES A RULING and needs the owner |

**Not searched:** VM101 beyond the reads report, ser8 archives, the private repository. The
ruling's original text may exist outside the repository.

---

```
id / round      X-RUN4-RANGE / R1
type            REVIEW_REQUEST
from / to       Team Alpha (Claude)  ->  Team Beta (Codex)
date            2026-11-13
review class    BEFORE a run that counts. No code is under review. No run is authorised by
                this request; launching Run 4 is the owner's decision.
base_commit     HEAD at the time you read (the commit that adds this file, on top of a41ac06)
files to read   docs/CLAUDE_CODE_REPORT_RANGE_MINER_VM101_READS.md
                agents/watcher_agent.py:2027-2101
                utils/seed_coverage_ledger.py:271-320, :1000-1055
```

Allowed answers per question: `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL` and
one sentence why) · `SUGGESTION` · `NEEDS EVIDENCE`.

## 1. The facts, as read

1. The certified ledger holds one record: `java_lcg`, `{constant, variable}`, `[0, 2³¹)`,
   from attempt 9. The cursor for Run 4's identity is OPEN at 2147483648 (reads report R1).
2. Run 4's committed command sends `seed_start: 0`. WATCHER overwrites it with the cursor
   (`agents/watcher_agent.py:2087-2089`), so Run 4 would sweep `[2³¹, 2³²)` as a fresh study
   (reads report R2). The pre-launch check that showed `--seed-start 0` stubbed the cursor at
   0 (`tests/run4_prelaunch_provenance_check.py:136-152`).
3. If that run publishes, the union is `[0, 2³²)` and the family is COMPLETE. The next
   Step 1 request for `java_lcg` with either mode set returns `seed_domain_complete` and
   generates no run (`:2068-2085`), whatever window, thresholds or session it asks for.
4. Attempt 9 published 0 rows. Per-phase stripe totals: constant forward 0, constant reverse
   0, hybrid forward 59, hybrid reverse 23,515 (reads report R3). This is recorded as an
   observation at one geometry. It is not offered as evidence about pool hit rate.
5. `agents/watcher_agent.py` has changed in two commits since the digest the Route-A
   certification covered (reads report R4-1).
6. The certified record stores `dataset_sha256` (reads report R1), but the cursor does not
   test it (`utils/seed_coverage_ledger.py:1006-1018`). A family certified COMPLETE on one
   dataset version therefore stays COMPLETE after new draws are published.

**The owner's position, stated 2026-11-13:** retiring Step 1 permanently goes against the
learning part of TFM. Step 1 has to be able to run again. The code in question was written
by several agents over more than a year, and where one piece contradicts the design, the
design governs.

## 2. Questions

**Q1 — Does the Run 4 comparison need attempt 9's seed interval?** Your ruling approved
"full 2³¹, attempt-9 stripes" and, at `:148`, "full 2³¹ seed domain used by Attempt 9".
If Run 4 runs over `[2³¹, 2³²)` with the same shape (2³¹ seeds, 32 stripes of 2²⁶ per stage,
same dataset, same pinned geometry), does it still certify what your claim boundary lists:
volume class, plumbing, completion and production shape?

**Q2 — After COMPLETE.** Is it intended that a family certified COMPLETE at one geometry can
receive no further Step 1 run at any other window, threshold or session? If yes, how are
the 50-trial soak and later window optimisation on `java_lcg` meant to run. If no, what is
the smallest correct statement of what a certified interval covers. In particular: should
coverage be tied to the dataset version, given the ruling that prior results are not valid
on the next version (fact 6), and to the session, given that re-optimisation is per-session?

**Q3 — Ordering.** Given Q2, which order is sound: (a) Run 4 over the second half, then the
soak; (b) the soak over the second half, with Run 4's acceptance taken from it or run some
other way; (c) neither until the coverage rule is settled. Alpha makes no recommendation
here and asks for yours.

**Q4 — The stale certification.** Before any Run 4 launch, is it enough to re-run the
Route-A pre-launch provenance check against the current `agents/watcher_agent.py` with the
real cursor value, or does the routing certification need to be redone?

**Q5 — Prior record.** Is there a governing document on any of Q1–Q4 that the section above
missed?

## 3. What Alpha did not verify

The stripe planning below `optimize_window`; how the ledger and finalizer treat a second
certified interval; anything on the rigs. The fleet is off.
