# CLAUDE CODE INSTRUCTIONS — REPAIR THE STAGING-ADMISSION STALL (L-4) AND CLOSE THE RUN 4 EVIDENCE

## Prior record

**Searches run (public mirror, HEAD `04aab30`, 2026-11-14):**

- `grep -lIiE "_pump_deferred|_prune_admitted|_try_admit|deferred admission|admission liveness|one attempt at a time"` over `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/PROJECT_FILE_CATALOG.md`, `docs/exchange/*/*.md`, `docs/*.md`.
- `grep -n "SR-1\|DECLARED_CHANGED" docs/PROJECT_FILE_CATALOG.md`.
- Read in full: `docs/LEADS.md` L-4 and L-5; `docs/exchange/from_beta/X-MINER-START_R1_BETA_REVIEW.md` Q7 and its positions table; `docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md` §4 "Changes to existing suites", §5, §9.
- Read at source: `miner/range_miner_coordinator.py:7743-7755`, `:8226-8236`, `:11880-11899`.

**Documents found. Alpha located the first group by name and did not read them. Read them before you design anything.**

| Document | What it holds | Label |
|---|---|---|
| `docs/CLAUDE_CODE_INSTRUCTIONS_ADMISSION_LIVENESS_REPAIR.md`, `docs/CLAUDE_CODE_INSTRUCTIONS_ADMISSION_BINDING_REPAIR.md`, `docs/CLAUDE_CODE_CORRECTION6_S172_PHASE4_ADMISSION.md`, `docs/CLAUDE_CODE_CORRECTION4_S172_PHASE4_OVERLOAD.md` | Why admission works as it does, and earlier repairs to it | ALREADY RECORDED. **Not read by Alpha** |
| `docs/CLAUDE_CODE_INSTRUCTIONS_S172_STAGING_BACKPRESSURE_REMEDIATION.md`, `docs/CLAUDE_CODE_INSTRUCTIONS_S172_BP_AMENDMENT.md`, their reports | The back-pressure design. `_pump_deferred` is its capacity-release point (`:7749-7755`) | ALREADY RECORDED. **Not read by Alpha** |
| `docs/CLAUDE_CODE_REPORT_S172_GATE12_DISPATCH_STALL.md`, `docs/GATE12_MP1_RUN_FORENSIC.md` | Earlier stalls. MP-1 drain starvation is CLOSED and R-1 is not to be reopened | ALREADY RECORDED. **Not read by Alpha.** Say whether L-4 is the same thing, a relative, or new |
| `docs/PROJECT_FILE_CATALOG.md` §1.0 SR-1; `docs/TB_RULING_FIELD6_IMPLEMENTATION.md` §2 | Any change to a certified surface keeps the `DECLARED_CHANGED` sets true before the commit is accepted | ALREADY RECORDED. Binding here |
| `docs/LEADS.md` L-4 | The stall: an admitted key is removed only inside `_pump_deferred`; the staging-job path finalizes then pumps (`:8230`, `:8234`); the StripeComplete path finalizes and does not pump (`:11897`). Rates on CPU: 1 of 10 with two workers, 0 of 10 with one, 4 of 4 at four workers. Not seen on the fleet | ALREADY RECORDED. This brief repairs it |
| `docs/exchange/from_beta/X-MINER-START_R1_BETA_REVIEW.md` Q7 | OBJECTION · LIVE on launching Run 4. Mechanism confirmed independently. Wants a separately reviewed repair, then a multi-worker test of staging-before-StripeComplete **without** the harness pump | ALREADY RECORDED. This brief is that repair |
| `miner/range_miner_coordinator.py:7743-7748` | `_pump_deferred` "Runs OFF the dispatch thread". The StripeComplete branch runs on the dispatch path | ALREADY RECORDED in the code. A repair that pumps from that branch must respect it |
| `docs/exchange/from_beta/X-RUN4-RANGE_R1_BETA_REVIEW.md` Q4 | Keep the frozen `69ca910` routing oracle; add a separate real-cursor case | ALREADY RECORDED. **Alpha's last brief contradicted it** by telling the session to refresh the fixtures. Part 2 puts it right |
| `docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md` §5, §9 | `G-UNPINNED-IDENTICAL` and four mutants are uncredited; two ways forward are named | ALREADY RECORDED. Part 2 |
| `docs/LEADS.md` L-5 | Importing `window_optimizer` sets the root logger to WARNING, so WATCHER drops `EXEC CMD` and the `[S172-PIN] step1_pin_source` record | ALREADY RECORDED. Part 3 |
| `docs/CLAUDE_CODE_REPORT_MINER_SEED_START.md` §9 | The miner evidence reader opens the live ledger `mode=ro` without `immutable=1` | ALREADY RECORDED. Part 3 |

**Items this brief raises:** none new. It carries out recorded follow-ups.
**Not searched:** VM101's live state, the fleet's run logs, ser8 archives, the private repository.

---

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, venv
`source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** record `git rev-parse HEAD` (expected `04aab30` or later).
**Review class:** BEFORE-COMMIT. **Report:** `docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md`.

**Why.** This is the last code change standing before Run 4. RANGE-MINER is the project's
top priority.

**The yardstick (CLAUDE.md).** Prediction-pool hit rate against `k/1000` on unseen draws.
This brief is about a run finishing, not about what it finds.

## 0. Rules for this session

- **No commit, push, branch or tag.** **Worktree:**
  `git worktree add --detach /home/michael/tfm_worktrees/miner-stall HEAD` (hyphen).
- **No fleet, no GPU, no SSH to any rig.** CPU, real coordinator, fake workers.
- **No writes to live state.** Do not run `gate_manifest_staging_dir_is_usable_here`
  (`tests/test_s172_staging_partb.py:599`); it writes a probe into the live staging folder.
- **No scheduled wake-ups or background timers while a relay runs.** Wait with a sleep loop.
- **May edit:** `miner/range_miner_coordinator.py` (the smallest change that works),
  `agents/watcher_agent.py` and `window_optimizer_integration_final.py` for Part 3 only,
  tests and test fixtures.
- **Not edited:** every other file under `miner/`, `prng_registry.py`, `sieve_gpu_worker.py`,
  the kernels, `utils/`, `persistent_worker_coordinator.py`, `persistent/`, `gate12_launch.sh`,
  the 22-array contract. Another session's work sits in
  `/home/michael/tfm_worktrees/pwc-step1`; leave it alone.
- **SR-1:** keep every `DECLARED_CHANGED` set true. Show the AST diff and the R-1 and MP-1
  results before and after.
- Use your judgement on small things and keep going. Stop only at a stop gate (§5).
- **First step:** `pgrep -af "[b]eta_relay.py"`; `git rev-parse HEAD`; `git status --porcelain`
  in the main tree; the worktree; the venv; the dataset copies as before; **read the
  documents in the first three rows of the table above**; baseline of the suites you will
  re-run.

## 1. Part 1 — the stall

**M first, before any edit.** Reproduce L-4 at base without the harness nudger: the
two-worker case and the four-worker case, with the counts. Then answer from the record: is
this MP-1, a relative of it, or new; and why did attempt 9 (25 real workers, 128 of 128
stripes) not stall? Say what is shown and what is inference.

**The repair.** Choose the mechanism the record supports. L-4 names three: pump after the
StripeComplete finalize; prune in `enqueue_staging`; a periodic pump. Requirements:

| # | Requirement |
|---|---|
| R1 | A stripe that reaches `done` on **either** path releases its admission and resumes deferred work. No ordering of "last staging job finishes" and "StripeComplete processed" can leave a `done` stripe holding admission |
| R2 | The dispatch path is not blocked. `_pump_deferred` is documented as running off the dispatch thread (`:7743-7748`); if the repair pumps from the StripeComplete branch, show how that holds, or hand the pump to the thread that already owns it |
| R3 | The back-pressure ordering Beta asked for is unchanged: retained deferred payloads claim a freed slot first, then a paused reader (`:7749-7755`) |
| R4 | No double release, no admission of a superseded or terminal attempt, no change when nothing is deferred. A one-worker run is byte-identical in its ledger rows and assignments |
| R5 | Prefer a repair that makes the state impossible over one that recovers from it later. If you add a periodic pump as well, say why the direct repair is not enough |

**Tests** (CPU, real coordinator, fake workers, **no nudger anywhere**; each with a mutant
shown red):

- The forced ordering: the last staging job finishes before StripeComplete is processed,
  with 2, 4 and 25 workers. Every run completes. Run each at least 20 times and give counts.
- The opposite ordering, the same counts.
- Run 4's shape: 25 workers, 32 stripes a phase over `[2³¹, 2³²)`, four phases.
- R4: the one-worker control, byte-equal to base.
- Remove the nudger from `tests/fixtures/miner_seed_start_driver.py` and re-run
  `tests/test_s172_miner_seed_start.py`: it must still pass.
- `tests/test_s172_admission_liveness.py`, `tests/test_s172_staging_backpressure.py`,
  `tests/test_s172_staging_partb.py` (without the probe gate), `tests/test_s172_phase4_coordinator.py`
  and every suite that imports the coordinator: before and after, every difference explained.

## 2. Part 2 — the Run 4 routing control

Alpha's last brief was wrong to have the frozen module edited. Put it right as Beta's two
reviews describe (X-RUN4-RANGE R1 Q4; X-MINER-START R1 Q6):

1. Restore the frozen routing-control module byte-for-byte to its `69ca910`-captured state,
   so its sha check passes again. Say plainly what `G-UNPINNED-IDENTICAL` can and cannot
   show on today's tree, where WATCHER needs a search lookup the frozen control never had.
2. Supply the search record through a **separate** real-cursor harness beside it, as a
   successor check in Beta's terms (X-MINER-START R1 Q6): name the stage-1 scheduling
   changes explicitly (the start taken from the search's first gap, the added
   `--step1-search-id`); use an independently established control captured from the tree
   **before** the miner seed-start change (`b11b6fd`) under the new search collaborator;
   keep the checks on every other argv field and on every containment behaviour. Do not
   bless an oracle captured only from the tree under review. Record the comparison boundary
   in the report.
3. Each of M1, M1b, M2a, M2b and M4 is credited through a gate that is valid today. No
   assertion is weakened. If a mutant cannot be credited honestly, leave it uncredited and
   say why.

## 3. Part 3 — two small repairs

- **L-5.** WATCHER keeps its own INFO records after the Step 1 comparison imports
  `window_optimizer`. The smallest sound repair in `agents/watcher_agent.py`. Test: a fresh
  process, a real `run_step` with dispatch intercepted, `EXEC CMD` and `[S172-PIN]
  step1_pin_source` both captured.
- **The evidence reader's ledger open.** Open the live miner ledger as the binding addendum
  §A requires (`docs/CLAUDE_CODE_INSTRUCTIONS_BINDING_ADDENDUM_BETA_R1_20261113.md`), so that
  reading it cannot create `-wal` or `-shm` files beside it. Test on a throwaway ledger.

## 4. Review by Beta, carried by this session

As `docs/exchange/RELAY_PROTOCOL.md` §2. Request file in the main tree:
`docs/exchange/to_beta/X-MINER-STALL_R1_ALPHA_REVIEW_REQUEST_admission_stall_repair.md`,
opening with `## Prior record`, BEFORE-COMMIT, files by absolute worktree path. At most
three rounds. Fix what you agree with inside this brief. You never overrule Beta.

**Questions for Beta (use as written):**

1. After the repair, can any ordering leave a `done` stripe holding admission?
2. Is the dispatch path still unblocked, and is the back-pressure ordering unchanged?
3. Any double release, or admission of a superseded or terminal attempt?
4. Is a one-worker run unchanged, and are the SR-1 declared sets true?
5. Do the tests force staging-before-StripeComplete with several workers and no nudger?
6. Part 2: is the frozen oracle intact, is the successor control independent, and is every
   mutant credited honestly?
7. Part 3: are the two repairs sound and minimal?
8. **Does your X-MINER-START R1 Q7 objection to launching Run 4 stand after this work? If
   it does, state exactly what remains.**
9. Any governing record this work missed.

## 5. Stop gates

Stop and report when: M does not reproduce; the record shows the admission design forbids
every candidate repair; R4 cannot be shown; a repair needs a file listed as not edited;
Beta holds a LIVE objection you disagree with or whose fix is outside this brief; the third
round ends.

## 6. Report

`docs/CLAUDE_CODE_REPORT_MINER_ADMISSION_STALL.md`, opening with `## Prior record`. At the
top, for the owner, in plain words: is the stall closed; what Beta says about launching
Run 4 now; what the rigs need at power-on (the new coordinator sha); what only a real fleet
run can still show. Then the first step, M, each part with `file:line`, tests with counts
and mutants, suites before and after, Beta's last position per question, and the landing
list with sha256 in two groups (landing script; hand-landed).
