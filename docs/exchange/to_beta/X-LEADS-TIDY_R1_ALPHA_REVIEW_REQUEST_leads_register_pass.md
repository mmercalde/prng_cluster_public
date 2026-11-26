# X-LEADS-TIDY  R1  REVIEW_REQUEST  leads register pass: six proposed leads brought into docs/LEADS.md as L-8 to L-13 (courtesy review)

- **Review class:** BEFORE-COMMIT, courtesy, **documents only**. The owner asked for your review before this lands:
  "prudent to pass item 3 in front of beta" (`logs/e2e_session/TASK_LEADS_TIDY_BETA_20261125.txt`, 2026-11-25
  17:30). It is not binding; the owner decides.
- **Base commit:** `7f4505a` (main).
- **Rounds:** one. Alpha changes nothing after your reply and reports your positions to the owner word for word.
- **Allowed answers per question:** `NO OBJECTION` · `OBJECTION` (with `LIVE` or `THEORETICAL`) · `SUGGESTION` ·
  `NEEDS EVIDENCE`.
- **Scope, please keep to it:** what is written. The question is whether the register text matches its sources. It
  is not a re-review of the measurements behind the sources, of TFM, or of any model. No code is under review; no
  pipeline, governed or hand-landed file is touched.

**The two files under review are NOT on main.** They are in the worktree `~/tfm_worktrees/leads-tidy` (detached at
`7cac105`; `7cac105` and `7f4505a` have the same `docs/LEADS.md`). Three of the source files are not on main either;
read each at the path given below.

## Prior record

**Searches run** (main `7f4505a`, 2026-11-25, read-only):

```
S = docs/PROJECT_FILE_CATALOG.md docs/TB_RULING_* docs/TB_RULING_REQUEST_* docs/PROPOSAL_* docs/CHAPTER_*
    docs/*_AUDIT.md docs/TODO_CONSOLIDATED.md docs/BACKLOG.md docs/LEADS.md docs/exchange/
git grep -n -I -E '\bL-(8|9|10)\b|L-why-[0-9]|L-move-[0-9]'         (whole repo; and the same on origin/pwc-offset,
git grep -n -I -E '\bL-1[0-3]\b'                                      origin/e2e-grader, origin/ch14-learn)
/bin/grep -rlIiE 'two-way|bidirectional excess|independence figure' $S
/bin/grep -rlIiE 'run H.*(pool|k/1000)|e2e_runH_grade' $S
/bin/grep -rlIiE 'batt_ac_lag_01|batt_cs_zero_crossings|batt_fft_diff_peak' $S
/bin/grep -rlIiE 'split seed|split_seed|run-to-run' $S
/bin/grep -rlIiE 'label window|last 900|trials 5|depth-max' $S
/bin/grep -rlIiE 'CPU and GPU|device determinism' $S
/bin/grep -rnI -E '\bL-(8|9)\b|L-why|L-move|LEADS' logs/e2e_session/          (on disk; logs/ is git-ignored)
/bin/grep -lIiE 'leads.tidy|LEADS_TIDY|leads register' logs/e2e_session/*.txt
ls docs/exchange/to_beta docs/exchange/from_beta | grep -i leads                (no earlier X-LEADS-TIDY thread)
```

**Documents found that decide, authorise, schedule or record the subject:**
- **The tasks:** `logs/e2e_session/TASK_LEADS_TIDY_20261125.txt` (lead 17:20: bring L-8, L-9, L-why-0, L-move-1..3 into
  the register in its format; quote measured facts and limits exactly; add nothing not in a source; propose a
  numbering; write the READY note; stop) and `TASK_LEADS_TIDY_BETA_20261125.txt` (owner 17:30: this review).
- **The register's own rules:** `docs/LEADS.md:1-15`; `~/.claude/skills/lead-handling/SKILL.md` §2, §4, §5.
- **The yardstick:** `CLAUDE.md` "The yardstick"; `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md`.
- **The drafts that were left out of earlier landings:** `docs/drafts/e2e_v1/READY_GRADER_LANDING_20261124.md:115-127`
  (L-9 left out; numbering, banner and paths to settle).
- **Related lead text not brought in:** `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md:318-325` (proposed "L-9
  sub-item", `residual_*`, F-a/F-b) and `logs/e2e_session/TASK_LOOP_ADDENDUM_3.txt:6` (lead: "follow-up F-a … is
  registered OPEN … lead L-9 unchanged").
- **Run I, L-9's open follow-up:** `logs/e2e_session/TASK_RUN_I_PREP.txt:3`; `OWNER_GO_run_I_20261123.txt`.
- **Owner context on the what-moves data (not copied into any entry):**
  `logs/e2e_session/OWNER_GO_macro_one_card_timing_20261124.txt` (14:58 takeaways).
- **Older numbering:** `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90` (WP-6 "L-4 to L-9 as in v1.0") and
  `docs/TFM_SESSION_HANDOFF_20261113.md:183`.
- **Same-string labels in another namespace:** `tests/test_s172_d6_liveness_gate.py:466-531` ("L-8" … "L-13" test
  cases).

**Labels:**

| Item | Label |
|---|---|
| L-8, L-9 entries | ALREADY RECORDED as drafts (`pwc-offset:docs/LEADS.md:656`, `e2e-grader:docs/LEADS.md:658`); bringing them into the register is NEW |
| L-10 to L-13 | ALREADY RECORDED as proposed text (`READY_WHY_BLOCK0_20261124.md:245`; `READY_WHAT_MOVES_20261124.md:204-206`); the IDs are NEW |
| Numbering scheme | NEW (proposal) |
| CHANGES A RULING | none |

**Not searched:** the host outside the repository and the named worktrees, ser8 archives, the private repository,
project files outside the repository. The handoff "section 0 item 3" named by the task is not in the repository.

## Files under review (2)

| # | Lands at | N/M on `7f4505a` | Read it at | sha256 now |
|---|---|---|---|---|
| 1 | `docs/LEADS.md` | **M** (318 lines appended, 0 removed; 972 lines) | `~/tfm_worktrees/leads-tidy/docs/LEADS.md` | `41e161f54cd5b1b5ede83388e08b6a9e6034ff56bda46ae467bdb76882e97cbc` |
| 2 | `docs/drafts/loop_v1/READY_LEADS_TIDY_20261125.md` | **N** | `~/tfm_worktrees/leads-tidy/docs/drafts/loop_v1/READY_LEADS_TIDY_20261125.md` | `8c3a656aac7402d693c0563081442506dbf28a4b4412894567a57c58c5833a0a` |

**Lines 1-654 of file 1 equal main.** `head -654 docs/LEADS.md | sha256sum` gives
`c12d1dcb8b5583b1ea9013c20c7e485ae28e90c9804d43fac0717e3689d5aacd`, which is `git show 7f4505a:docs/LEADS.md |
sha256sum` (654 lines). `git diff 7f4505a --stat -- docs/LEADS.md`: `318 insertions(+)`, no deletions. The new entries
start at `:658` (L-8), `:742` (L-9), `:826` (L-10), `:874` (L-11), `:917` (L-12), `:953` (L-13).

## Source files the six entries were taken from

| Entry | Source (file:line) | Read it at | On main? | sha256 now |
|---|---|---|---|---|
| L-8 | draft entry `docs/LEADS.md:656-720` | `~/tfm_worktrees/pwc-offset/docs/LEADS.md` (branch `pwc-offset` `9809d7e`; = branch HEAD) | no | `a4bf07da7da6e2ceee5e469b975c4feec268c8f7aaca008e5ccc89f03b725a79` |
| L-8 | the note, `:1-264` (summary `:3-15`; limits `:137-149`, `:202-207`; hit rate `:224-244`) | `~/tfm_worktrees/pwc-offset/docs/drafts/NOTE_LEAD_BIDIRECTIONAL_EXCESS_20261122.md` (= branch HEAD) | no | `919d3ee35a58ffb40f6acb1e622a84cf199336a542a3fc0afef0bd7dc149037c` |
| L-9 | draft entry `docs/LEADS.md:658-727` | `~/tfm_worktrees/e2e-grader/docs/LEADS.md` (branch `e2e-grader` `d52e84b`) | no | `b9992754d2df270c4e525d20f95e60ea3751a3c453b3387d9ea18f461e511a03` |
| L-9 | counted grade, `:1-233` (result `:84-120`; limits `:205-226`; reading `:228-233`) | main `docs/drafts/e2e_v1/RESULT_E2E_GRADE_RUNH_20261123.md` | yes | `97dfeceb4fd965e6f127822ffdee6d80a5d479493243bc4ac3b6444c3b96e29c` |
| L-9 | follow-up grades, `:1-226` (results `:94-149`; dependence `:151-168`; assumptions `:170-189`; status `:191-207`) | main `docs/drafts/e2e_v1/RESULT_E2E_FOLLOWUP_RUNH_20261123.md` | yes | `f1a47607541c75e4fffc60ac9b33c19dd1944959a4d19d3057939a60e0481c35` |
| L-9 (marked additions) | `logs/e2e_session/TASK_RUN_I_PREP.txt:3`; `OWNER_GO_run_I_20261123.txt:1-4`; `READY_GRADER_LANDING_20261124.md:115-127` | main tree (logs on disk) | logs ignored; landing note yes | — |
| L-10 | why-report, `:1-279` (proposed text `:245`; table `:233-243`; limits `:241-243`, `:268-272`) | `~/tfm_worktrees/ch14-learn/docs/drafts/loop_v1/READY_WHY_BLOCK0_20261124.md` (branch `ch14-learn` `7368f34`; = branch HEAD) | no | `75a3d5404f9091d3a7e5a76b9fa73fb1ed30e6b215b1def930e04a70560ac0c7` |
| L-11, L-12, L-13 | what-moves note, `:1-229` (leads `:198-208`; table B `:136-176`; per-run `:178-187`; noise floor `:108-134`; limits `:24-26`, `:200`, `:211-219`) | `~/tfm_worktrees/ch14-learn/docs/drafts/loop_v1/READY_WHAT_MOVES_20261124.md` (= branch HEAD) | no | `b4708153d60716227d41354de0a661c7c9da056a02bb984c1283bf80f6c4b179` |
| Q1 below | miss report `:318-325` | main `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md` | yes | `07da565698af0aef20a9d3495dedda11c83c4901d456e59f03fb06a347cba062` |

**What Alpha changed against the drafts** (stated so you need not diff for it; each is marked in the entry text):
- L-8: provenance lines added; "Limits" and "For hit rate against `k/1000`" paragraphs added, quoted from the note
  §2.2, §4, §6; one italic cross-reference to L-9.
- L-9: the "DRAFT, worktree only" banner removed (the landing note `:120` says it would be wrong once landed);
  provenance lines; "(descriptive)" and the one-sided lower bound 0.835 added to the lift line; D2's adjusted p
  (0.5774, 0.7821); "Limits" (with the power sentence) and "For hit rate" paragraphs quoted from the two reports; one
  italic line naming run I with the owner's go; "(as of the source, 2026-11-23)" on the closing status line.
- L-10 to L-13: new entries in the register's format, built from the two ch14-learn notes. Where a source gives no
  hit-rate prediction the entry says "The source states no …".

## Questions

**Q1 — Fidelity.** Does each of L-8 to L-13 state its source's measured facts and limits exactly, with nothing
strengthened, weakened or turned into a conclusion, and no verdict on TFM? Is any text in an entry not traceable to a
source or to a marked addition?

**Q2 — Yardstick.** Does each entry say what it predicts for pool hit rate against `k/1000` on unseen draws and name
the measurement that would settle it, or say plainly that its source gives none? Is any lead judged by a
state-recovery standard?

**Q3 — Register.** Is the numbering (L-8 and L-9 kept; L-why-0 → L-10; L-move-1..3 → L-11..13) consistent with the
existing register and its rules (`docs/LEADS.md:1-15`)? Is any recorded lead missing? Two candidates Alpha already
names (its own Q1 and Q2 below): the `residual_*` sub-item, and the one |z| ≥ 2 comparison no proposed lead covers.

## Alpha's own open questions (from the READY note, so you need not find them)

These are addressed to the lead and the owner. Comment on any of them if you wish; they are not asked of you.

- **A-Q1.** Should the miss report's proposed "L-9 sub-item" (`residual_*`, F-a and F-b OPEN,
  `READY_LOOP_MISS_20261123.md:318-323`) become an L-9 follow-up, or its own L-14? The lead's Addendum 3 says F-a
  "is registered OPEN", but no register entry carries it. Left out because the task's list does not name it.
- **A-Q2.** One of the six |z| ≥ 2 comparisons in what-moves table B is in no proposed lead: "without `battery`",
  pool 100, Δ −5.7, z −2.0 (`READY_WHAT_MOVES:158`). Should it be a lead? Not added.
- **A-Q3.** L-9's limit "a draft grader, not Beta-reviewed" is kept as written. The counted grade used the frozen
  `a4d96fe7` copy; the grader has since landed (`c7fe48bf`, then `3ab8ce2a`; X-E2E-GRADER-LANDED R1). Add a status
  line?
- **A-Q4.** L-10, L-11 and L-12 carry no hit-rate prediction because their sources give none. Should the authors
  supply one?
- **A-Q5.** L-10's open follow-up is an attribution contrast on later blocks (TODO-WHY-1), not a graded hit-rate
  comparison. Is that the settling measurement to record?
- **A-Q6.** The L-8 note and its `logs/lead_bidir_20261122/`, the two ch14-learn notes and their `data/`, and the L-9
  grade outputs are not on main. Land the notes with `docs/LEADS.md`; archive the data?
- **A-Q7.** Whether run I has run since 2026-11-23 was not checked. If it has, L-9 needs its result.
- **A-Q8.** WP-6 (`PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90`) and the 2026-11-13 handoff `:183` use an older
  "L-4 to L-9" numbering that overlaps the register's IDs. A mapping or owner note?
- **A-Q9.** The handoff "section 0 item 3" was not found in the repository; the task file was the only brief.

**Where sources differ** (recorded in the READY note, "Where the sources differ"): L-10's proposed text says "top
100" while its table finds one column at pools 100 and 300; the L-8 note words the excess range as "1.8× to 3.0×"
(real window, `:3`) and "1.8–3.9" (with controls, `:147`, `:228`); the L-8 draft's "owner approved each" was not
checked (the commit `00e2ee7` was found in all four bundles).
