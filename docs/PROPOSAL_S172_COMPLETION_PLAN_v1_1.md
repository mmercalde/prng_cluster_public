# S172 RANGE-MINER — COMPLETION PLAN, AGENT LANES, BETA EXCHANGE AND DASHBOARD — v1.1

**Author:** Team Alpha · **Date:** 2026-11-11 · **Owner:** Michael
**Status:** PLAN under the owner's rulings of 2026-11-11. Supersedes v1.0 (same date), which was
written before those rulings and was never committed.
**Base:** public repo `mmercalde/prng_cluster_public` at `a15b524` (2026-08-28), plus
`TFM_SESSION_HANDOFF_20261111.md` and the owner's statements in session on 2026-11-11.

**Not read for this plan:** live VM101, the rigs, the private repo, `~/dashboard_work/`, and the
three reports that exist only on VM101 (`CHAPTER_13_AUDIT.md`, `CHAPTER_14_AUDIT.md`,
`TODO_CONSOLIDATED.md`). Items marked **[VM101]** need a live check there before anyone acts.

---

## 0. WHAT CHANGED FROM v1.0

| v1.0 said | v1.1 says | Source |
|---|---|---|
| Beta rules; its rulings are binding | Michael is final authority. Alpha is second in command. **Beta reviews; it does not approve** | owner ruling 1 |
| Every packet waits on Beta | Beta reviews **before commit**: coordinator, worker, kernel code, and runs that count. **After commit**: documents, tools, monitoring, dashboard | owner ruling 2 |
| Power caps, circuit check, power telemetry before Run 4 | **Dropped.** The crash recorder (netconsole) stays | owner ruling 3 |
| Agents never launch; delegation is a request to Beta | **Agents may launch and operate the fleet**, authenticated and scope-limited. See `PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md` | owner rulings 4, 5 |
| Finished = five conditions ending in the soak | **Finished = RANGE-MINER can brute-force the complete seed space** | owner, in session |
| Beta's identity unknown; API relay recommended | Beta is ChatGPT in its chat window with its own project folder. Manual relay is the transport | owner, in session |
| `G-PROD-SHAPE` has never run | It ran 2026-08-22 and failed: 15 pass / 6 fail, 512 orphan staged files | `S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:330-340` |
| Freeze-reproduction and 13M-survivor stress runs | **Not planned.** The owner does not want runs that try to overwhelm the rigs | owner, in session |
| Agent model unspecified | Opus 5.5 unless the work needs an expert | owner ruling 6 |

## 1. WHERE RANGE-MINER STANDS

| Fact | Source |
|---|---|
| Gate 12 passed at attempt 9 (2026-08-17): 4 stages, 128/128 stripes, 25 GPUs, coverage certified `[0, 2³¹)` for `java_lcg`, constant and variable skip | handoff §4; fact file §2.49 |
| Brief I (window-anchor) is implemented and committed. Final acceptance waits on one completed production-shape run | handoff §3 |
| Run 1 (2026-08-22) failed at the Phase-5 commit on a Brief-I defect. Repaired | handoff §4 |
| Run 3 (2026-08-22) froze two rigs after a phase that produced 13,146,485 survivors at `k=6`, `τ=0.35`. **Cause undetermined**: supply sag and volume pressure are co-leading | `RIG_CRASH_FORENSIC_20260822.md` §7 |
| Run 4 is prepared: pinned attempt-9 geometry, Route-A patch committed (`3e2537b`, `e75a6eb`, `a15b524`), dry run 10/10. Not launched | handoff §4; `S172_RUN4_PROPOSAL_PINNED_GEOMETRY.md` |
| The run verifier `G-PROD-SHAPE` fails on 512 orphan staged files that may not belong to the run under test | classification report `:330-340` |
| Fleet powered off since 2026-08-28 | handoff §4 |
| WATCHER manifest defaults `use_range_miner: false` | handoff §4 |

## 2. DEFINITION OF FINISHED (owner ruling)

> RANGE-MINER is finished when it can run and brute-force the entire seed space.

**The complete seed space is `[0, 2³²)` for `java_lcg`.**
- `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:30` sets the sweep target at
  0 → 4,294,967,296, reached over progressive runs.
- `utils/run_finalizer.py:277` sets `SEED_DOMAIN_EXCLUSIVE_MAX = 2 ** 32`, and
  `utils/seed_coverage_ledger.py` refuses any run outside it.

**Measured as four conditions, in order:**

| # | Condition | Evidence that closes it |
|---|---|---|
| F1 | The run verifier can pass on a clean run | `G-PROD-SHAPE` scoped to the run's own files; red on a planted orphan from the run, green with only historical orphans present |
| F2 | Run 4 completes and Brief I is accepted | Run 4 publishes; `G-PROD-SHAPE` PASS; owner accepts after Beta's review |
| F3 | A run cannot overwhelm the rigs | A survivor-volume limit in the launch path: a pre-run sample predicts volume and the launch is refused above the cap |
| F4 | The whole space is swept | The coverage ledger reports `java_lcg`, modes `{constant, variable}`, **COMPLETE** over `[0, 2³²)`, with no host freeze during the sweep |

At the pinned geometry the second half, `[2³¹, 2³²)`, is one more run of the attempt-9 shape.
Attempt 9's serve loop took 738 s for `2³¹` seeds.

**Not part of finished. Done next, in this order:** Brief II · the 50-trial soak · the
self-learning wiring faults from the Chapter 13 and 14 audits · the remaining open to-dos.

## 3. OPEN-ITEM REGISTER

"Waits on" is **Owner**, **Fleet**, **Review** (Beta, before commit), or **None**.

### 3.A Path to finished

| ID | Item | Waits on |
|---|---|---|
| A1 | Scope `G-PROD-SHAPE`'s orphan check to the run under test (F1). Test code, fleet not needed | None |
| A2 | Survivor-volume limit (F3): pre-run sample on the run's own residue window, refusal above a cap. The Run 4 proposal already predicts volume this way (§1.1 there), so the method exists | Review |
| A3 | Read the coverage cursor on VM101 and settle Run 4's seed range. The Run 4 proposal sets `max_seeds = 2³¹` and names no `seed_start`; the cursor after attempt 9 should sit at `2³¹`. Decide whether Run 4 repeats `[0, 2³¹)` or takes the second half **[VM101]** | None |
| A4 | Read the attempt-9 bidirectional count from the published generation **[VM101]** | None |
| A5 | Power-on: rigs up, `boot-proxmox`, GPU gate 8/8 ×3, parity gate 30/30, clean tree | Owner, Fleet |
| A6 | Re-arm netconsole and confirm one proof line per host before any run | Fleet |
| A7 | Run 4 with the committed Route-A command; check actuals against the predicted envelope (about 96 phase-3 and 22,656 phase-4 survivors) | Fleet |
| A8 | `G-PROD-SHAPE` on Run 4; closure report; Brief I acceptance (F2) | Review |
| A9 | Sweep the remainder to `2³²` (F4) | Fleet |

### 3.B Guard rails for agent-launched runs

Specified in `PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md`. Built in parallel with 3.A.

| ID | Item | Waits on |
|---|---|---|
| G1 | Restricted agent account and key on the rigs; no `sudo`; rig environments read-only | Owner |
| G2 | Environment-freeze gate: refuses if any host's packages, kernel or drivers differ from the owner's baseline | None |
| G3 | Run-authorization gate: refuses any launch outside what the owner authorized | None |
| G4 | A2 above, wired into the same launch path | Review |

### 3.C After finished

| ID | Item | Source |
|---|---|---|
| C1 | Brief II: optimizer, registry, metadata, repo-wide `offset` consumer audit, D.1 reach demonstration | proposal v1.1 §9 |
| C2 | 50-trial soak, `--start-step 1 --end-step 1`, `n_parallel = 1`, about 10 hours | soak brief |
| C3 | WATCHER failure-authority defect: a failed Step 1 was reported as passed because an output file existed. No brief yet | handoff §4 |
| C4 | WATCHER manifest default `use_range_miner: false` | handoff §4 |
| C5 | Self-learning wiring faults: `prediction_pool.json` has no writer; nothing calls `request_selfplay`; attribution is never invoked; health check reads `best_round` while the producer writes `best_val_round`; `forward_matches` / `reverse_matches` missing from the merge list | handoff §6 **[VM101]** |
| C6 | Coordinator ingress bounded by count, not bytes (L-2) | BACKLOG §20 |
| C7 | float32/float64 survivor-filter seam (L-1) | BACKLOG §21 |
| C8 | Optuna raw-to-canonical threshold quantization | fact file §2.36 |
| C9 | The remaining open items in `TODO_CONSOLIDATED.md`: 144 open, 28 startable today | handoff §6 **[VM101]** |

### 3.D Documents

| ID | Item | Waits on |
|---|---|---|
| D1 | Chapter corrections, batches 1 to 3. In progress on VM101; batch 1 stops for Alpha's review | Owner commit |
| D2 | `CLAUDE.md`: Beta as reviewer, rule 3 replaced, 25 GPUs, phase table current | None |
| D3 | Fact file, behaviour model, catalog brought current; fact file updated in all three copies | Owner (Settings upload) |
| D4 | `SESSION_CHANGELOG_20261111_<TOPIC>.md` | None |
| D5 | Noise-baseline idea: filed as `NOTE_NOISE_BASELINE_PER_TRIAL_20261111.md`. **Note only, not on the work list** | owner ruling 8 |

## 4. PLAN OF ATTACK

### 4.1 Lanes

| Lane | Who | Does | Writes | Fleet |
|---|---|---|---|---|
| **L0** | Alpha (this role) | Holds the register, writes briefs, reviews agent output, prepares Beta packets, brings disagreements to the owner | register, briefs, packets | No |
| **L1** | Run operator, Opus | Power-on gates, Run 4, the sweep, evidence bundles | logs, reports | Yes |
| **L2** | Guard-rail builder, Opus | A1, A2, G2, G3 and their tests | `scripts/`, `tests/` | Tests only |
| **L3** | Repair briefs, Opus | Turns the audit findings (C3 to C5) into one brief per fault | `docs/` | No |
| **L4** | Documents, Opus | D1 to D4 | `docs/` | No |
| **Beta** | Reviewer | Reviews packets; may open its own threads | its own replies only | No |

At most four agents active at once. L1 is the only lane that touches the fleet, and the fleet
runs one trial at a time.

**Collision rules**
- One writer per file at a time. L0 keeps the lock table.
- `miner/range_miner_coordinator.py` has one writer for the whole project, and every change to
  it carries its `DECLARED_CHANGED` update in the same commit (SR-1).
- Agents deliver files. **Michael commits and dual-pushes.** No agent commits or pushes.
- Coordinator, worker and kernel changes go to Beta before Michael commits. Everything else is
  committed first and reviewed after.

### 4.2 Waves

**Wave 0 — now, fleet off.**

| Packet | Lane | Deliverable |
|---|---|---|
| W0-1 | L4 | Chapter corrections batch 1 (running), then batches 2 and 3 |
| W0-2 | L2 | A1: verifier orphan scoping, with the red and green controls |
| W0-3 | L2 | G2 and G3: the two refusal gates, new files only |
| W0-4 | L2 | A2: volume-limit design, then implementation. Review before commit |
| W0-5 | L0 | A3 and A4: the two VM101 reads |
| W0-6 | L3 | Repair briefs for C3, C4, C5 |
| W0-7 | L4 | D2 and D4 |
| W0-8 | L0 | Beta's first packet: the handoff, both v1.1 documents, and a request for Beta to write its own charter as a reviewer |

**Wave 1 — power-on day.** The one day the owner is needed on site: power, `boot-proxmox`,
install G1, capture the environment baseline for G2. L1 runs A5 and A6.

**Wave 2 — Run 4.** About 15 minutes of fleet time. Then A8.

*Decision rule.* If Run 4 completes: go to Wave 3. If a host goes silent: stop, preserve the
evidence, report to the owner. No relaunch.

**Wave 3 — the sweep.** A9, under the delegated-run guard rails. Finished is declared when F4's
evidence exists.

**Wave 4 — after finished.** 3.C in order: Brief II, soak, repairs, remaining to-dos.

### 4.3 What sets the calendar

| Constraint | Effect | Handling |
|---|---|---|
| Owner on site | One day, power-on | Everything else is arranged around it |
| Owner commits | Every change waits for a commit | Batched: one commit list per wave |
| Beta round trip | Only before-commit classes wait on it | Small single-question packets, §5 |
| One fleet, one trial at a time | Physical | L1 alone uses it |

### 4.4 Work-packet standard

Every packet an agent receives carries: one falsifiable question · the files it may write · the
files it must read first · base commit · the verification block (execution proof, clean control,
fault-injection control, completion sentinel) · `file:line` anchors read in that session ·
investigate and fix kept as separate packets.

## 5. THE ALPHA–BETA EXCHANGE

### 5.1 What Beta is

ChatGPT, run from its chat window, with a project folder of its own. It has no access to VM101,
the rigs or the live repo. **It reviews what it is given.** The owner carries every file both
ways.

Consequences:
- A Beta statement about current system state is checked on VM101 before Alpha acts on it.
- Beta's project folder must hold today's handoff. The older fact file and `CLAUDE.md` tell it
  that its rulings are binding.
- The exchange is designed around files the owner carries, not around a live connection.

### 5.2 Format

```
exchange/
    to_beta/      X-0007_R1_ALPHA_REVIEW_REQUEST_volume_limit.md
    from_beta/    X-0007_R1_BETA_REVIEW.md
    LEDGER.md     one row per packet
    CHARTER.md    Beta's standing review rules, written by Beta
```

**Thread ID** `X-NNNN`, **round** `R1, R2, …`, in every filename. That also ends the problem of
same-named uploads arriving empty.

| Direction | Packet types |
|---|---|
| Alpha → Beta | `REVIEW_REQUEST` · `EVIDENCE` · `NOTICE` · `REPLY` |
| Beta → Alpha | `REVIEW` · `QUESTION` · `FINDING` |
| Either → Owner | `ESCALATION` |
| Owner | `DECISION` |

`QUESTION` and `FINDING` let Beta open a thread itself.

**Header on every packet**

```
id / round / type / from / to / date
review class       BEFORE-COMMIT | AFTER-COMMIT
base_commit        the commit every anchor was read at
answers            sha256 of the packet this one replies to (blank when opening)
artifacts          path + sha256 of every file or diff referenced
questions          Q1 … Qn, each with the allowed answers listed
```

**Beta's review format.** A positions table first, one row per `Qn`, from a fixed vocabulary:
`NO OBJECTION` · `OBJECTION` · `SUGGESTION` · `NEEDS EVIDENCE`. Then the body, verbatim.

**What happens to a review**

| Beta's position | Alpha's move |
|---|---|
| `NO OBJECTION` | Proceed |
| `SUGGESTION` | Alpha adopts or declines, with a reason, in the thread |
| `NEEDS EVIDENCE` | Alpha supplies it or says it is unavailable |
| `OBJECTION`, Alpha agrees | Alpha fixes and re-sends |
| `OBJECTION`, Alpha disagrees | `ESCALATION` to Michael with both positions, side by side. Michael's `DECISION` closes the thread |

A before-commit packet holds the commit until Beta has reviewed or Michael has decided. An
after-commit packet holds nothing.

**Ledger states** `DRAFT → SENT → REVIEWED → RESOLVED → CLOSED`, with age in days and round
count. Two views: *waiting on Beta* and *waiting on Alpha*.

**Integrity**
- Files in `from_beta/` are Beta's words. No Alpha agent edits them.
- The ledger records each file's sha256 on receipt. A check script fails if a `from_beta/` file
  changed, if a reply answers a packet that does not exist, or if an ID is reused.

### 5.3 Historical rulings

Existing `TB_RULING_*` documents stay in the repo unchanged. Where one calls itself binding or
final, read it as Beta's recorded review position. Decisions the project has built on stay in
force until Michael changes them.

## 6. THE DASHBOARD

The owner wants to watch the work the way the "AI village" video shows it. The village is a view
of real project data. Nothing in it can start, stop or approve anything.

### 6.1 What it shows

| Place | Real thing behind it |
|---|---|
| **Town hall** | L0: the register, wave progress, the four finished conditions |
| **Courthouse** | Beta exchange: threads waiting on Beta, on Alpha, on Michael |
| **Rig yard** | Three rig sheds with eight GPU lights each, plus Zeus: up or down, Proxmox or Ubuntu, busy or idle |
| **Run tower** | Current run: stage 1 to 4, stripes done, survivors per phase, gate results, and the sweep's coverage bar from 0 to `2³²` |
| **Workshop** | L2: gate or guard rail being built |
| **Drafting office** | L3: brief being written |
| **Records office** | L4: chapter batches, stale-document list |
| **Your desk** | Everything waiting on the owner, oldest first: commits, decisions, escalations |

Each agent is a character at its building with one line saying what it is doing.

### 6.2 Data

| Feed | Source | Written by |
|---|---|---|
| Agent status | `status/agents.json`: lane, state, packet, one-line activity, heartbeat | each agent |
| Register and waves | §3 and §4 kept as data | L0 |
| Beta exchange | `exchange/LEDGER.md` | L0 |
| Run progress | the miner ledger, opened read-only | coordinator (existing) |
| Coverage | the coverage ledger, opened read-only | finalizer (existing) |
| Fleet state | GPU gate and parity gate output | L1 |

An agent with no fresh heartbeat shows as "no signal", never as idle. An unreadable feed shows
as unavailable, never as zero.

### 6.3 Build

- One read-only web page on VM101, on its own port. The TFM dashboard on port 5002 is not touched.
- It only reads files. No buttons that act.
- Review class: after commit.

| Stage | What you get | Depends on |
|---|---|---|
| **D1** | Plain status board: lanes, your desk, exchange queue, finished conditions | `status/agents.json`, ledger |
| **D2** | Rig yard, run tower and coverage bar, live during runs | fleet up |
| **D3** | Village layout with buildings and characters | D1, D2 |
| **D4** | Walking and small animations | optional |

## 7. READINGS ALPHA CHOSE

These follow from the owner's rulings but were not stated word for word. Each stands unless the
owner changes it.

1. **"Runs that count" are reviewed before their result is accepted, not before launch.** A
   counted run launches without waiting on Beta. Its acceptance record is not committed until
   Beta has reviewed the evidence or Michael has decided.
2. **Guard-rail gates are tools, so they are committed first and reviewed after.** They are
   still sent to Beta before the first agent-launched run uses them.
3. **The volume limit (A2) touches the launch path of a counted run, so it is reviewed before
   commit.**
4. **Finished is measured for `java_lcg`.** The other ten generator families are repeat sweeps
   after finished, using the same machinery.

## 8. NEEDS THE OWNER

1. A power-on date.
2. On that day: the restricted agent account (G1) and the environment baseline (G2).
3. Carrying the first Beta packet (W0-8) and adding the handoff to Beta's project folder.

## 9. WHAT THIS DOCUMENT DOES NOT ESTABLISH

- Live state on VM101 and the rigs on 2026-11-11. Every **[VM101]** item is unread.
- That the coverage cursor sits at `2³¹`. That is the fact file's record of attempt 9, not a
  reading taken today.
- The cause of the 2026-08-22 freeze. It is undetermined, and this plan does not investigate it.
- Any duration. The 738 s and 10-hour figures come from attempt 9; trials vary.
