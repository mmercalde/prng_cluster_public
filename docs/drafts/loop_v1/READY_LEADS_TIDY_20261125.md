# READY — leads register, one tidy pass (TASK_LEADS_TIDY_20261125)

Documents only. Worktree `~/tfm_worktrees/leads-tidy`, detached at main `7cac105`. Nothing written in
`~/distributed_prng_analysis`. No commit, no push, no landing, no card, no suites.

## Prior record

**Searches run** (main `7cac105`, read-only; the worktree is the same tree):

| # | Command | Scope | Result |
|---|---|---|---|
| P1 | `git grep -n -I -E '\bL-(8\|9\|10)\b\|L-why-[0-9]\|L-move-[0-9]'`, and separately `'\bL-1[0-3]\b'` | whole repo at `7cac105` | cross-reference table below |
| P2 | the same pattern with `git grep <branch>` | `origin/pwc-offset`, `origin/e2e-grader`, `origin/ch14-learn` | cross-reference table below |
| P3 | `/bin/grep -rlIiE` with `two-way\|bidirectional excess\|independence figure`; `run H.*(pool\|k/1000)\|e2e_runH_grade`; `batt_ac_lag_01\|batt_cs_zero_crossings\|batt_fft_diff_peak`; `split seed\|split_seed\|run-to-run`; `label window\|last 900\|trials 5\|depth-max`; `CPU and GPU\|device determinism` | `docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_* docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/` | read below |
| P4 | `/bin/grep -rnI -E '\bL-(8\|9)\b\|L-why\|L-move\|LEADS'`, plus the subject patterns of P3 | `~/distributed_prng_analysis/logs/e2e_session/` | read below |
| P5 | `/bin/grep -rlI -E 'L-why\|L-move\|\bL-(8\|9)\b'` | untracked `data/` and `logs/lead_bidir_20261122/` in the three source worktrees | no hits |

**Documents found that decide, schedule or record the subject:**
- **The three source drafts.**
  - `pwc-offset:docs/LEADS.md:656`, the L-8 draft, plus its note `docs/drafts/NOTE_LEAD_BIDIRECTIONAL_EXCESS_20261122.md`
    on that branch.
  - `e2e-grader:docs/LEADS.md:658`, the L-9 draft. Its two RESULT reports are on main.
  - `ch14-learn:docs/drafts/loop_v1/READY_WHY_BLOCK0_20261124.md:233-245` (L-why-0) and
    `READY_WHAT_MOVES_20261124.md:198-208` (L-move-1 to L-move-3).
- **Landing note.** `docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md:115-127` records that L-9 was left out of the
  grader landing, and the three things to settle when it lands: numbering, the draft banner and the paths.
- **Related register text in the miss report.** `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md:318-325` proposes an
  "L-9 sub-item" (`residual_*`, F-a and F-b). `logs/e2e_session/TASK_LOOP_ADDENDUM_3.txt:6` (lead): "follow-up F-a
  (residual_* ranking variant) is registered OPEN … lead L-9 unchanged". **Not brought in:** the task does not list it
  (Q1).
- **Run I.** `logs/e2e_session/TASK_RUN_I_PREP.txt:3` and `OWNER_GO_run_I_20261123.txt`: run I is the open L-9
  follow-up. The owner authorised it on 2026-11-23 at 18:28, then said to wait for the agents.
- **Owner context for the L-11 to L-13 data.** `logs/e2e_session/OWNER_GO_macro_one_card_timing_20261124.txt`
  (14:58): the owner's two takeaways on the noise-floor report. They are context only and were not copied into any
  entry.
- **Existing L-ID ranges in other documents.** `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90` (WP-6, "entries
  L-4 to L-9 as in v1.0") and `docs/TFM_SESSION_HANDOFF_20261113.md:183` ("LEADS L-4 to L-9"). Both use an older range
  that overlaps the IDs now in the register (Q8).
- **Added 2026-11-25 after Beta X-LEADS-TIDY R1, finding 1** (`docs/exchange/from_beta/X-LEADS-TIDY_R1_BETA_REVIEW.md:20`):
  - `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:1-5` (owner 10:04; item 2: run I may go past the
    50,000 survivor cap, "one-off, run I only") and `:27-28` (TODO-WHY-1: the lead runs the attribution cycle by hand
    after each block until the trigger and queue are built);
  - the record of the forward/reverse independence question before the L-8 measurement, as the L-8 note's own prior
    record lists it (`pwc-offset:docs/drafts/NOTE_LEAD_BIDIRECTIONAL_EXCESS_20261122.md:32-41`):
    `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:318-329` (F-7), `docs/PIPELINE_BEHAVIOUR_MODEL.md:1201` (D10) and
    `docs/exchange/from_beta/TFM_Reverse_Sieve_Proposal_2026-11-13.md:100` (Beta's conditional-probability point).
- **P3 hits that were read and are not on these subjects:**
  - `CHAPTER_14_TRAINING_DIAGNOSTICS.md:3707-3709` (the battery feature list);
  - `PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_0.md:164` (run-to-run population comparability);
  - the X-E2E-GRADER exchange files (the grader, and run H's frozen pool metadata).
  - "label window" matched generic window-optimizer text in the Chapter 1, 6 and 11 files and older proposals.
  - None of them records any of the six leads.

**Labels:**

| Item | Label |
|---|---|
| L-8 entry in the register | ALREADY RECORDED as a draft (`pwc-offset:docs/LEADS.md:656`). Bringing it into the register is NEW |
| L-9 entry in the register | ALREADY RECORDED as a draft (`e2e-grader:docs/LEADS.md:658`; `READY_GRADER_LANDING_20261124.md:115-121`). Bringing it in is NEW |
| L-10 (L-why-0) | ALREADY RECORDED as proposed text (`READY_WHY_BLOCK0_20261124.md:245`). The ID is NEW |
| L-11 to L-13 (L-move-1 to L-move-3) | ALREADY RECORDED as proposed rows (`READY_WHAT_MOVES_20261124.md:204-206`). The IDs are NEW |
| The numbering scheme | NEW (proposal) |
| Anything that changes a ruling | none. WP-6's "L-4 to L-9" is noted, not changed (Q8) |

**Not searched:** the host outside the named paths, ser8 archives, the private repository's other branches, and
project files that are not in the repository. That includes the handoff whose "section 0 item 3" the task cites: the
newest handoff on main is `TFM_SESSION_HANDOFF_20261121*.md`.

## Plain summary

- **What changed.** One file: `docs/LEADS.md`. Six proposed leads were appended in the register's existing entry format.
  They were only on branches or in notes before. They are now **L-8 to L-13**, and all six are **OPEN**. L-1 to L-7 are
  byte-for-byte untouched: the diff is 318 lines added and 0 removed.
- **Numbering scheme.** New IDs follow on from L-7 in the order the leads were filed.
  - L-8 and L-9 keep the IDs they were drafted with.
  - L-why-0 becomes **L-10**. It stays one entry covering three columns, as its source proposes it.
  - L-move-1, L-move-2 and L-move-3 become **L-11, L-12 and L-13**.
  - Each new entry names its old ID and the file and line it came from.
- **What each entry carries.**
  - The measured fact, quoted from its source.
  - Its limits.
  - What it predicts for hit rate against `k/1000`.
  - The settling measurement, under "Follow-ups OPEN".
- **Where a source is silent.** Three entries have no hit-rate prediction in their source: L-10, L-11 and L-12. Each
  of them says "The source states no …". I supplied none (Q4).
- **Text I added.** Each piece is marked in the entry:
  - provenance lines (branch, snapshot commit, the draft's file:line, and whether the note is on main);
  - one cross-reference from L-8 to L-9;
  - one sourced line under L-9 saying the open "second generation" follow-up is run I, with the owner's go;
  - four quoted source items in L-9: the "(descriptive)" qualifier and the one-sided lower bound, D2's adjusted p, the
    power limit, and the counted report's §6 reading.
- **One thing I removed.** The "DRAFT, worktree only" banner at the head of L-9. The grader landing note said that
  banner would be wrong once the entry landed (`READY_GRADER_LANDING_20261124.md:120`).
- **Conclusions.** None. No lead is closed, downgraded or strengthened.

## Old ID → new ID → source

| Old ID | New ID | Register line (this worktree) | Source file:line | Measured fact read in full at |
|---|---|---|---|---|
| L-8 | **L-8** (unchanged) | `docs/LEADS.md:658` | `pwc-offset:docs/LEADS.md:656` (branch `pwc-offset`, `9809d7e`) | `pwc-offset:docs/drafts/NOTE_LEAD_BIDIRECTIONAL_EXCESS_20261122.md:1-264` |
| L-9 | **L-9** (unchanged) | `docs/LEADS.md:742` | `e2e-grader:docs/LEADS.md:658` (branch `e2e-grader`, `d52e84b`) | `docs/drafts/e2e_v1/RESULT_E2E_GRADE_RUNH_20261123.md:1-233`, `RESULT_E2E_FOLLOWUP_RUNH_20261123.md:1-226` (both on main, identical to the worktree copies) |
| L-why-0 | **L-10** | `docs/LEADS.md:826` | `ch14-learn:docs/drafts/loop_v1/READY_WHY_BLOCK0_20261124.md:245` (table `:233-243`) | same file, `:1-279` |
| L-move-1 | **L-11** | `docs/LEADS.md:874` | `ch14-learn:docs/drafts/loop_v1/READY_WHAT_MOVES_20261124.md:204` | same file, `:1-229` |
| L-move-2 | **L-12** | `docs/LEADS.md:917` | `…READY_WHAT_MOVES_20261124.md:205` | same |
| L-move-3 | **L-13** | `docs/LEADS.md:953` | `…READY_WHAT_MOVES_20261124.md:206` | same |

## Cross-references that name one of these IDs

**On main (`git grep` at `7cac105`):**

| File:line | What it says | Needs updating? |
|---|---|---|
| `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90` | WP-6: "entries L-4 to L-9 as in v1.0" | **Different numbering.** v1.0 is not in the repository. Not edited, because it is a proposal record (Q8) |
| `docs/TFM_SESSION_HANDOFF_20261113.md:183` | owed: "LEADS L-4 to L-9" | Same numbering as WP-6. It is a historical handoff, so no |
| `docs/drafts/anchor_v1/RUNBOOK_RUN_I_20261123.md:50`, `:61` | point at the L-9 draft, `~/tfm_worktrees/e2e-grader/docs/LEADS.md:658-731` | **Yes, if the runbook is reused.** The entry now lives at `docs/LEADS.md:742-825`. The ID is unchanged |
| `RUNBOOK_RUN_I_20261123.md:46`, `:69`, `:365` | a grep for `L-9`; the WP-6 overlap; the `--search-reason` text "L-9 follow-up" | No. The ID is unchanged |
| `docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md:79`, `:117-121`, `:127` | describes the L-9 draft and what to settle when it lands | No. It is the historical record, and this pass settles its items 1 and 2 |
| `docs/drafts/e2e_v1/RESULT_E2E_FOLLOWUP_RUNH_20261123.md:9`, `:28`, `:37`, `:198` | "draft register entry L-9"; "drafted … in this worktree's `docs/LEADS.md`"; L-8's draft location | Only the location is stale. Not edited, because it is a result record |
| `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md:3`, `:15`, `:24`, `:322-325` | L-9 unchanged by the report; `:15` gives the draft location; `:322` holds the proposed sub-item | `:15` location only. `:322` waits on Q1 |
| `scripts/diagnostics/e2e_miss_report.py:10` | docstring "(L-9)" | No |
| `tests/test_s172_d6_liveness_gate.py:466-531` | test-case labels "L-8" to "L-13" (liveness checks) | No. These are a different namespace that happens to share the IDs. Anyone grepping should know about it |

**On the source branches (not on main):**

| Branch:file:line | Needs updating if it lands? |
|---|---|
| `pwc-offset:docs/LEADS.md:656` (the L-8 draft) | Superseded by this pass. That branch's `docs/LEADS.md` must not be landed as well, or L-8 would appear twice |
| `pwc-offset:docs/drafts/NOTE_LEAD_BIDIRECTIONAL_EXCESS_20261122.md:1`, `:42`; `pwc-offset:docs/drafts/NOTE_STEP2_READINESS_ON_RUNH_20261122.md:146` | No. The ID is unchanged |
| `e2e-grader:docs/LEADS.md:658-662` (the L-9 draft and its banner) | Superseded. The same warning applies |
| `ch14-learn:docs/drafts/loop_v1/READY_WHY_BLOCK0_20261124.md:245` ("L-why-0") | **Yes, or add a mapping line**, if the note lands: L-why-0 is now L-10 |
| `ch14-learn:docs/drafts/loop_v1/READY_WHAT_MOVES_20261124.md:204-206`, `:208` ("L-move-1" to "L-move-3", "L-why-0") | **Yes, or add a mapping line**, if the note lands: they are now L-11, L-12, L-13 and L-10 |

## Diff stat

```
$ git diff --stat
 docs/LEADS.md | 318 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 318 insertions(+)
```

- `docs/LEADS.md`: 654 lines at `7cac105` (sha256 `c12d1dcb…`), 972 lines now (sha256 `41e161f5…`). There are no
  deletions, so lines 1-654 are unchanged.
- This note is the only other new file. It is untracked: `docs/drafts/loop_v1/READY_LEADS_TIDY_20261125.md`.

## Where the sources differ (both readings recorded, flagged)

1. **L-10: "top 100", and where the F1 results sit.** Both readings are recorded in the entry: the verbatim quote and
   the table.
   - The proposed text (`READY_WHY_BLOCK0:245`) says "among each tree family's top 100". The source table (`:237`)
     records `batt_fft_diff_peak` as found at "midday 100 and 300; evening 100".
   - The proposed text also folds F1 results into where-found for `batt_ac_lag_01` ("CatBoost and LightGBM midday"),
     but not for the other two columns.
   - My entry title says "top-100", following the proposed text.
2. **L-8: the size of the excess.** The ranges are worded differently, but they agree once the real window is separated
   from the controls. I kept the draft's wording.
   - The note's summary (`:3`) says "1.8× to 3.0×" on the real window.
   - §2.2's disclosure (`:147`) says "1.8–3.9 across thresholds", and §6 (`:228`) says "1.8–3.9× … for any window,
     real or not".
   - Measured values: real window 1.81 to 2.97 (`:106-112`); controls up to 3.91 (`:119`).
3. **L-8: one draft claim checked against the record.**
   - The draft says all four runs were at commit `00e2ee7`. The note states it for run E only (`:53`). I checked that
     the string `00e2ee7` appears in each of the four `logs/skipsw_session/bundle_thr*/bundle.json` files (read-only).
   - The draft also says "owner approved each". I did not check that.

## Questions for the lead

- **Q1.** Should the miss report's proposed "L-9 sub-item" become an L-9 follow-up, or its own L-14? It covers
  `residual_*`, with F-a and F-b OPEN (`READY_LOOP_MISS_20261123.md:318-323`). Your Addendum 3 says F-a "is registered
  OPEN", but no register entry carries it. I left it out because the task's list does not name it.
- **Q2.** Of the six |z| ≥ 2 comparisons in what-moves table B, one is not covered by any proposed lead: "without
  `battery`", pool 100, Δ −5.7, z −2.0 (`READY_WHAT_MOVES:158`). Should it be a lead? I did not add it.
- **Q3.** Should L-9's limit "a draft grader, not Beta-reviewed" stay as its source wrote it, or get a status line?
  - The counted grade used the frozen `a4d96fe7` copy.
  - The grader has since landed (`c7fe48bf`, Beta `X-E2E-GRADER-LANDED` R1).
  - Counted grades stay on `a4d96fe7` until the new grader is reviewed (`TASK_LOOP_ADDENDUM_3.txt:3`).
  - I left it as the source wrote it.
- **Q4.** L-10, L-11 and L-12 carry no hit-rate prediction, because their sources give none. Should their authors
  supply one, or should the entries stay as they are?
- **Q5.** L-10's open follow-up is an attribution contrast on later blocks (TODO-WHY-1), not a graded hit-rate
  comparison. Is that the settling measurement you want recorded, or should a graded variant be named?
- **Q6.** Several cited notes and outputs are not on main:
  - the L-8 note and its `logs/lead_bidir_20261122/`;
  - `READY_WHY_BLOCK0` and `READY_WHAT_MOVES` with their `data/ch14_learn/`;
  - the L-9 grade outputs (`READY_GRADER_LANDING_20261124.md:127` asks whether to archive them, for example to
    `~/tfm_worktrees/_records/`).

  The entries cite branches and snapshot commits. Should those notes land with `docs/LEADS.md`, and should the data be
  archived?
- **Q7.** Run I is L-9's open "second generation" follow-up. I did not check whether it has run since 2026-11-23. If it
  has, L-9 needs a follow-up line from its result.
- **Q8.** WP-6 (`PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90`) and the 2026-11-13 handoff `:183` name "L-4 to
  L-9 as in v1.0". That numbering differs from the register's, v1.0 is not in the repository, and the register's L-4 to
  L-13 now cover other subjects. Is a mapping or an owner note wanted? Editing WP-6 is not mine to do.
- **Q9.** I could not find the handoff "section 0 item 3" in the repository. I worked from the task file alone.

Stopped here.

## Beta review, X-LEADS-TIDY R1 (appended 2026-11-25; nothing above it was changed at that time; Addendum 1 changes are listed in the next section)

- **Task:** `logs/e2e_session/TASK_LEADS_TIDY_BETA_20261125.txt` (owner 17:30). One round, courtesy, not binding; the
  owner decides.
- **Request:** main `docs/exchange/to_beta/X-LEADS-TIDY_R1_ALPHA_REVIEW_REQUEST_leads_register_pass.md`, sha256
  `d8b142b2…cd3cc`. Base `7f4505a`.
- **Relay:** run from the main tree, 234.7 s. **Verdict: `REVIEW RECEIVED` (exit 0).** Log
  `logs/beta_relay/X-LEADS-TIDY_R1.log`; console copy `logs/beta_relay/X-LEADS-TIDY_R1.console.txt`. Ledger row
  appended by the relay (`docs/exchange/LEDGER.md`, uncommitted in the main tree).
- **Reply:** main `docs/exchange/from_beta/X-LEADS-TIDY_R1_BETA_REVIEW.md`, sha256 `a5697d2e…fff`. Read in full. The
  review file is the record; the table below is copied from it, not a summary of it.
- **Main tree after the run** (`git status --short`): `M docs/exchange/LEDGER.md`, `?? …from_beta/X-LEADS-TIDY_R1_BETA_REVIEW.md`,
  `?? …to_beta/X-LEADS-TIDY_R1_ALPHA_REVIEW_REQUEST_leads_register_pass.md`. Landing them is the owner's
  (`owner_land.py --exchange X-LEADS-TIDY`).
- **Nothing was changed in response.** `docs/LEADS.md` is as it was sent (sha256 `41e161f5…7cbc`).

**Positions, word for word** (review `:29-33`):

| Question | Position | Severity |
|---|---|---|
| Q1 — Fidelity | SUGGESTION | — |
| Q2 — Yardstick | SUGGESTION | — |
| Q3 — Register | SUGGESTION | — |

No `OBJECTION`, so nothing is `LIVE` or `THEORETICAL` and nothing holds the landing. No row went to `DEFERRED.md`.

**Alpha's one-line view of each** (for the owner; Beta's reasoning is in the review at the lines cited):

| Item | Beta (review line) | Alpha's view |
|---|---|---|
| Q1, L-8 | `:39` bound "whenever" / "does not depend on which values" to the windows and thresholds tested; E and F approvals UNAVAILABLE, G and H found | Agree. Both are wording the draft carried over; a fix is a small, sourced qualification plus attributing "owner approved each" to the draft |
| Q1, L-9 | `:40` keep "independent … of the pools"; scope the prefix claim; spell out D2's 22/904 and 21/904 with adjusted p | Agree. All three are restorations of source text (`FOLLOWUP:183-185`, `:154-162`), not new claims |
| Q1, L-10 | `:41` headline from the measured table; name the contrast (C1b/C2 vs C1); place LightGBM midday 300 correctly | Agree. It sharpens the "top 100" discrepancy already flagged (Where the sources differ, item 1) |
| Q1, L-11/L-12 | `:42` scope the ~25-hit resolution to pool 300, block 0; name L-12's comparison as 5 vs 20 trials | Agree; wording only |
| Q1, L-13 | `:43` supported as written | Noted |
| Q2, all | `:47` none uses a state-recovery standard | Noted |
| Q2, L-8/L-9 | `:49` L-8 should record L-9's grade as the completed real-window comparison for that configuration; L-9's question should compare hit count with Σk_t/1000 (or rates with rates) | Agree with both. The second is a units slip in the drafted question, carried from the e2e-grader draft |
| Q2, L-10 | `:50` say plainly there is no directional prediction and no graded settling measurement; TODO-WHY-1 tests recurrence | Agree; it answers my Q5 in the same direction |
| Q2, L-11/L-12 | `:51` settling = named variant vs baseline on later unseen blocks, equal budgets, hits vs k/1000 | Agree it is the settling measurement. It is not written in the source, so adding it is the lead's call (my Q4) |
| Q2, L-13 | `:52` say explicitly the source gives no hit-rate prediction | Agree; the entry implies it but does not say it |
| Q3, numbering | `:56` consistent; keep the WP-6 reference qualified | Noted |
| Q3, `residual_*` | `:57` a recorded open item missing from the register; carry it forward, ID the lead's and owner's choice | Agree; it is my Q1 |
| Q3, battery z −2.0 | `:58` a recorded comparison, not by itself a missing ID | Agree; my Q2 stays a lead decision |
| Q3, run I | `:59` keep the anchor-6695 / runbook link; no completion record found; the 11-24 owner ruling carries the cap authority | Agree; the italic line in L-9 should name the anchor and the cap ruling (`OWNER_RULINGS_pending_list_20261124.txt:5`) |
| Finding 1 | `:20` prior record omitted the 11-24 owner rulings list and the L-8 note's F-7 / D10 / Beta `:100` record | Agree. I read the L-8 note's prior record but did not carry it into the request |
| Finding 2 | `:63` this note cites `TASK_LOOP_ADDENDUM_3.txt:4` for the grader pin; it is `:3` | Agree; my error (Q3 above) |
| Finding 3 | `:64` this note puts its plain summary before `## Prior record`; the owner rule puts the record first | Agree in part. The task asked for "plain summary first"; the owner rule (`CLAUDE.md` "Prior record first") says the record opens the document. The two instructions conflict; the owner rule governs. Not changed here, per the task |

**Open for the lead:** whether to apply the Q1/Q2 wording fixes before landing (none needs a new source; the L-11/L-12
settling sentence would be an addition), and the ID for `residual_*`. Beta's positions are all `SUGGESTION`; per
`RELAY_PROTOCOL.md` §2.4 a suggestion outside the task's scope goes to the lead, and this task said change nothing.

## Addendum 1 — Beta's suggestions adopted (lead, `logs/e2e_session/TASK_LEADS_TIDY_ADDENDUM_1_20261125.txt`, 17:50)

The lead adopted Beta's three SUGGESTIONS "exactly as Beta words them and no further", with four lead decisions under
Q3. Documents only, this worktree only, no second Beta round. **Line numbers in the sections above (the ID table, the
cross-reference table, the diff stat) describe the version Beta reviewed** (`docs/LEADS.md` sha256 `41e161f5…7cbc`);
the lines below are the current file after Addendum 2 (sha256
`653d54a24592cf69986299e2f3451782196195f7b752a3419852c03a5fb9a196`).

**Current entry starts (L-8 to L-15):** L-8 `:658`, L-9 `:754`, L-10 `:843`, L-11 `:904`, L-12 `:950`, L-13 `:988`,
**L-14 `:1013`** (new), **L-15 `:1067`** (new, Addendum 2), "Older ID ranges" `:1121`.

**Correction (Addendum 2, lead, 18:05, `logs/e2e_session/TASK_LEADS_TIDY_ADDENDUM_2_20261125.txt`):** the Addendum 1
decision not to register the no-battery pool-100 comparison was withdrawn by the lead, because it closed a lead on a
background figure (about 4 of 84 expected by chance). The "Recorded comparisons not registered" paragraph is removed
and the comparison is registered as **L-15, OPEN**.

| Beta item (review line) | What changed | `docs/LEADS.md` line |
|---|---|---|
| Q1 L-8: bound "whenever" / "does not depend on which values" (`:39`) | "Measured, for the windows and thresholds tested … which of the tested values" | 709-711 |
| Q1 L-8: comparisons pooled; run-window 2^24 sample gave zero two-way rows at the tight threshold (`:39`) | new bullet citing note `:114`, `:122` (0 both, independence figure 0.39) | 712-714 |
| Q1 L-8: attribute "owner approved each" to the draft; G and H approvals found, E and F not (`:39`) | "Where found" rewritten: draft attribution, `OWNER_OVERRIDE_runG_thr3838.txt:1`, `OWNER_OVERRIDE_runH_thr3030.txt:1`, E and F not located; commit `00e2ee7` in all four bundle manifests | 669-672 |
| Q2 L-8: L-9's grade is the completed real-window comparison for this frozen configuration; shuffled-pool comparison and further grading open; rig control separate (`:49`) | new RUN bullet (L-9 counted grade); rig-control bullet says it measures overlap counts and is separate; hit-rate bullet's italic note now says what is done and what stays OPEN | 718-720, 739-740, 744-746 |
| Q1 L-9: independent of one another **and of the pools** (`:40`) | background-figure sentence, citing follow-up report `:182-185` | 818-819 |
| Q1 L-9: scope the exact-prefix claim (`:40`) | "within the same arm and block each smaller raw pool was measured as the exact prefix of the larger" | 809-810 |
| Q1 L-9: D2 with explicit counts and adjusted p in the p column (`:40`) | two D2 rows: A1 22/904, 0.5774 (adjusted); A0 21/904, 0.7821 (adjusted) | 798-799 |
| Q1 L-9: the frozen-grader description applies to the instrument used for this grade (`:40`) | added to the "Instrument" line and to "Limits" | 776, 815 |
| Q2 L-9: compare the hit **count** with Σk_t/1000, or both as rates (`:49`) | opening question rewritten | 762-764 |
| Q3 run I (`:59`) + lead decision | anchor 6695 and the 12 midday draws 2021-03-05..03-16 with the runbook link (`RUNBOOK_RUN_I_20261123.md:10-16`); cap ruling (`OWNER_RULINGS_pending_list_20261124.txt:5`); "No completion result was found in the records checked" | 825-833 |
| Q1 L-10: headline from the measured table, draft quote kept beside it (`:41`) | title now "…separated top-N voters of the drawn number from top-N non-voters (pools 100 and 300)"; "What it is" states the measured fact first; the quote is labelled "(draft wording)" | 843, 850-856 |
| Q1 L-10: FFT at midday 300 too; lag-1 midday LightGBM is F1 at 300 (`:41`) | a sentence after the quote | 860-862 |
| Q1 L-10: name the contrast; 2.5-column reference is for C1b/C2 under a hypothetical independent-column null; C1 reported separately at 14–45 (`:41`) | C1b named (`READY_LOOP_MISS_20261123.md:180`); "Where found" scopes the reference and adds C1 14–45 "mostly selection" (`WHY:222`, `:231`) | 850-852, 866-870 |
| Q2 L-10: no directional hit-rate prediction and no graded settling measurement; TODO-WHY-1 tests recurrence; owner record `:27-28` (`:50`) | "For hit rate" paragraph rewritten | 889-893 |
| Q1 L-11/L-12: ~25-hit resolution is pool 300, block 0 (`:42`) | L-11 limits; L-12 limits | 934-935, 974 |
| Q1 L-12: five vs the 20-trial baseline; +0.4 / +0.6 SD at pool 300 (`:42`) | "What it is"; "Follow-ups RUN" | 956, 970 |
| Q2 L-11/L-12: the settling comparison; more splits on block 0 only quantify run-to-run variation (`:51`) | sentence added to each "For hit rate" paragraph, marked as adopted from Beta | 940-942, 977-979 |
| Q2 L-13: say explicitly the source gives no prediction or settling measurement; the repeat checks GPU repeatability (`:52`) | sentence added to "For hit rate" | 1004-1006 |
| Q3 `residual_*` (`:57`) + lead decision | **new entry L-14, OPEN**, cross-referenced to L-9; text from `READY_LOOP_MISS_20261123.md:297-325` and `TASK_LOOP_ADDENDUM_3.txt:6`; F-a and F-b OPEN | 1013-1064 |
| Q3 battery z −2.0 (`:58`) + lead decision | Addendum 1 added a "Recorded comparisons not registered" note; **Addendum 2 removed it and registered the comparison as L-15, OPEN**: the table B row quoted with its header (`MOVES:147`, `:156`, `:158`), floor and chance figures, limits with measured vs assumed inputs, no follow-up run, open follow-ups, cross-reference to L-10, licensed phrasing, and the record that the source proposed no lead and the lead registered it after first deciding not to | 1067-1119 |
| Q3 WP-6 (`:56`) + lead decision | "Older ID ranges": one qualified sentence, no mapping | 1121-1123 |
| Finding 1 (`:20`) | added to this note's Prior record ("Added 2026-11-25 after Beta X-LEADS-TIDY R1, finding 1") | this note, Prior record |
| Finding 2 (`:63`) | Q3 above now cites `TASK_LOOP_ADDENDUM_3.txt:3` | this note, Questions Q3 |
| Finding 3 (`:64`) | `## Prior record` moved to the top of this note, before the plain summary | this note, top |

**Two points where I applied the lead's text narrowly, flagged for the lead:**
- The addendum groups L-10 and L-13 for the TODO-WHY-1 sentence and the `OWNER_RULINGS:27-28` citation. TODO-WHY-1 is
  about the attribution cycle, which L-13 (device determinism) does not use, and Beta's L-13 wording (`:52`) does not
  mention it. L-13 got Beta's L-13 sentence only; the TODO-WHY-1 sentence is in L-10 only.
- Beta wrote "The runbook documents a different run with the same name" (`:59`). I did not record that sentence,
  because I could not tell which other run it means. L-9 now names run I by its anchor and runbook lines, as Beta
  asked.

**Diff against main:**

```
$ git diff --numstat 7f4505a -- docs/LEADS.md          (after Addendum 2; after Addendum 1 it was 422 0)
469	0	docs/LEADS.md
$ head -654 docs/LEADS.md | sha256sum
c12d1dcb8b5583b1ea9013c20c7e485ae28e90c9804d43fac0717e3689d5aacd  -     (= git show 7f4505a:docs/LEADS.md | sha256sum)
```

Additions only. L-1 to L-7 (lines 1-654) are byte-identical to main.

Stopped. Waiting for the lead to read the diff (after Addendum 2).
