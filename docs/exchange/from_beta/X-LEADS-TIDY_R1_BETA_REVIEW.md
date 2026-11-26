## Prior record

**X-LEADS-TIDY · R1 · Team Beta · 2026-11-25 PST**  
Before-commit courtesy review, documents only. Base read on main `7f4505aaccebdd284393eb3565e75d09d1c0f3b4`. Proposed files read in `~/tfm_worktrees/leads-tidy` at `7cac1056e70c54b3c33640d09f07b8d4ba9cb103`.

`git status --porcelain` on main at initial read (17:34 PST):

```text
?? docs/exchange/to_beta/X-LEADS-TIDY_R1_ALPHA_REVIEW_REQUEST_leads_register_pass.md
```

Proposed worktree status at review:

```text
 M docs/LEADS.md
?? docs/drafts/loop_v1/READY_LEADS_TIDY_20261125.md
?? docs/drafts/loop_v1/SURVEY_SMALL_BRANCHES_20261125.md
```

**Finding 1 — ALREADY RECORDED governing material omitted from the request's prior record.** The November 24 owner record gives Run I its conditional, one-off survivor-cap exception and records how TODO-WHY-1 is handled pending automation. Both bear on follow-up status. L-8's own note also points to the existing Chapter 2 F-7 / behaviour model D10 discussion and Beta's prior conditional-probability point. Those documents record the independence question before the L-8 measurement. `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:1-5`, `:27-28`, `pwc-offset:docs/drafts/NOTE_LEAD_BIDIRECTIONAL_EXCESS_20261122.md:32-41`, `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:318-329`, `docs/PIPELINE_BEHAVIOUR_MODEL.md:1201`, `docs/exchange/from_beta/TFM_Reverse_Sieve_Proposal_2026-11-13.md:100`.

- **Searches run.** Used `git grep -n -I -E` for L-8 through L-13 and the `L-why-*` / `L-move-*` IDs in `docs/` on `7f4505a` and the three named remote-tracking refs. Used `rg -n -i` for the lead names, two-way excess and independence, `residual_*`, TODO-WHY-1, device determinism, and related register terms across the catalog, rulings, proposals, chapters, audits, TODO, backlog, register, drafts and exchange. Searched the named `logs/e2e_session/` and `logs/skipsw_session/` records for the lead tasks, Run I and owner decisions. Checked WP-6's older reference with `git log --all --` on the named v1.0 path.
- **ALREADY RECORDED.** The lead task names six entries. The owner requested a courtesy review. The sources already record the six entries as draft or proposed text, the `residual_*` follow-ups, the pool-100 battery comparison, the older WP-6 ID overlap and Alpha's top-100 wording discrepancy. The register rules and unseen-draw yardstick govern this pass. `logs/e2e_session/TASK_LEADS_TIDY_20261125.txt:1-6`, `logs/e2e_session/TASK_LEADS_TIDY_BETA_20261125.txt:1-8`, both proposed files, and the source anchors below. The counted grade's authority is recorded at `logs/e2e_session/OWNER_RULING_e2e_Q1_20261123.txt:1-2` and `logs/e2e_session/OWNER_GO_counted_grade_20261123.txt:1`.
- **NEW.** The six entries in the main register, the four ID mappings and the additional document discrepancies below. **CHANGES A RULING:** none.
- **Not searched.** Private remotes beyond local tracking refs, other host paths, rigs, ledgers, ser8 archives and off-repository project records. The task's handoff reference and a completion record for the L-9 Run I are **UNAVAILABLE** in the material reviewed.

## Positions

| Question | Position | Severity |
|---|---|---|
| Q1 — Fidelity | SUGGESTION | — |
| Q2 — Yardstick | SUGGESTION | — |
| Q3 — Register | SUGGESTION | — |

## Q1 — Fidelity

The reported counts and values I checked agree with their sources. The wording below needs correction or qualification. Each entry keeps its findings open and the grade's reading stays confined to its declared configuration. Here `LEADS` and `READY` refer to the two proposed files, `BIDIR` to the `pwc-offset` bidirectional excess note, `GRADE` and `FOLLOWUP` to the two main Run H reports, and `WHY` and `MOVES` to the two `ch14-learn` notes.

- **L-8.** The E–H rows, the constant-mode correction for H and the test groups match the note. Bound “whenever” and “does not depend on which values” to the windows and thresholds tested. The comparisons are pooled by group, and the 2²⁴ sample of the run window itself yielded zero two-way rows at the tight threshold. The draft already used the broader wording. Its “owner approved each” is also traceable to the draft. I verified `inputs.run.repository_commit` in all four local bundle manifests and found direct owner approvals for G and H. Direct approval records for E and F are **UNAVAILABLE** in the material checked. Attribute the broader approval statement to the draft or cite the E and F approvals. The reported real-window range and the across-group summary can coexist. `LEADS:665-710`, `pwc-offset:docs/LEADS.md:660-705`, `BIDIR:53-61`, `:102-149`, `:226-235`, `logs/skipsw_session/OWNER_OVERRIDE_runG_thr3838.txt:1`, `logs/skipsw_session/OWNER_OVERRIDE_runH_thr3030.txt:1`.
- **L-9.** The counted result, lift bounds, descriptive figures and 26-hit threshold match the grading reports at the stated rounding. Preserve both independence assumptions: draws are assumed independent of one another **and of the pools**. Scope the exact-prefix claim to smaller raw pools within the same arm and block. In D2, replace “as above” with the explicit `22/904` for A1 and `21/904` for A0 and put the adjusted values in the p column. The frozen grader description applies to the instrument used for this grade. `LEADS:762-763`, `:772-809`, `GRADE:76-82`, `:89-109`, `:127-154`, `:221-233`, `FOLLOWUP:146-168`, `:170-189`.
- **L-10.** Alpha already recorded the top-100 ambiguity. Use the measured table in the headline and preserve the attributed draft quote alongside it. The LightGBM FFT finding includes midday at 300 as well as 100, and LightGBM's midday lag-1 interval excluding zero belongs to same-block F1 at 300. Name the actual contrast between top-N voters for the drawn number and top-N non-voters. The rough 2.5-column reference applies to C1b/C2 under a hypothetical independent-column null. The source reports the different C1 contrast separately at 14–45. Naming the contrast avoids making “mostly near” read as a summary of both. `LEADS:826-859`, `READY:137-145`, `WHY:220-245`, `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md:178-182`.
- **L-11 and L-12.** The listed means, deltas, z values and per-split counts match the tables. Keep the approximately 25-hit single-run resolution scoped to pool 300 on block 0. For L-12, identify the comparison as five versus the 20-trial baseline and tie the “+0.4 SD” and “+0.6 SD” follow-up shorthand to pool 300. `LEADS:881-913`, `:922-949`, `MOVES:23-26`, `:68-75`, `:138-175`, `:178-185`, `:200-205`, `:213`.
- **L-13.** The two CPU/card pairs and LightGBM's CPU match are supported. The entry correctly limits the device observation to the matched split and keeps the repeat open. `LEADS:955-972`, `MOVES:112-134`, `:206`.

## Q2 — Yardstick

All six entries use the pool-hit yardstick where relevant. None uses survival of a generating candidate as an acceptance test. `LEADS:722-738`, `:750-751`, `:807-819`, `:862-870`, `:908-913`, `:943-949`, `:969-972`, `docs/PITFALL_STATE_RECOVERY_YARDSTICK.md:9-16`, `:52-62`.

- **L-8 and L-9.** The excess supplies no hit-rate prediction. L-8 lists Run H's real-window grade as open even though its cross-reference identifies the completed L-9 grade. Record that grade as the completed real-window comparison for this frozen configuration. Keep the matched shuffled-pool comparison and further unseen-draw grading open. The shuffled-draw rig control measures overlap counts and remains a separate follow-up. In L-9's opening question, compare the **hit count** with `Σk_t/1000`, or divide both quantities by the number of graded draws to compare rates. The grade and its scope are already correctly reported. `LEADS:722-738`, `:750-751`, `:773-779`, `:807-819`, `BIDIR:209-211`, `:224-244`, `GRADE:93-105`, `:228-233`.
- **L-10.** The source supplies no directional hit-rate prediction or graded settling measurement for those three columns. Say both plainly. TODO-WHY-1's later-block contrast tests recurrence of the attribution pattern. The feature-group comparisons in the source address broader variants. The November 24 owner record describes the lead running the attribution cycle by hand pending the trigger and queue. `LEADS:862-870`, `WHY:241-245`, `:251-272`, `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:27-28`.
- **L-11 and L-12.** Each explicitly limits the reported shifts to block 0 and lists more splits and other walk-forward blocks. The settling comparison is the named variant versus the current baseline on later unseen blocks, with equal pool budgets and hits reported against `k/1000`. More splits on block 0 quantify run-to-run variation there. `LEADS:908-913`, `:943-949`, `MOVES:3`, `:23-26`, `:198-216`.
- **L-13.** Say explicitly that the source gives no prediction or settling measurement for unseen-draw hit rate. The proposed repeat is a check on GPU repeatability for this device lead. `LEADS:967-972`, `MOVES:130-134`, `:206`.

## Q3 — Register

- **Numbering.** The mapping is consistent with the current register. L-1 through L-7 are byte-identical to the base, L-8 and L-9 retain the draft IDs, and L-10 through L-13 carry their source aliases and `OPEN` status. The historical WP-6 range is a recorded overlap. Its source mapping is **UNAVAILABLE** in the refs searched, so keep the historical reference qualified. `docs/LEADS.md:1-15`, `LEADS:658`, `:742`, `:826`, `:874`, `:917`, `:953`, `READY:88-123`, `docs/PROPOSAL_FINDINGS_20261113_RESOLUTION_v1_1.md:90`, `docs/TFM_SESSION_HANDOFF_20261113.md:183`.
- **`residual_*` is a recorded open item missing from the register.** The miss report proposes the sub-item with both follow-ups open. The lead explicitly records F-a as open for development and leaves L-9 unchanged. The six-item task explains the omission from this pass. Carry the open item forward, with a cross-reference to L-9 if it receives its own ID. The record leaves the choice of ID to the lead and owner. F-a is a graded ranking comparison. F-b is the repeat of the recurrence check on Run I pools. `docs/drafts/loop_v1/READY_LOOP_MISS_20261123.md:313-325`, `logs/e2e_session/TASK_LOOP_ADDENDUM_3.txt:6`, `logs/e2e_session/TASK_LEADS_TIDY_20261125.txt:3`, `LEADS:811-822`, `READY:159-161`.
- **The battery result is a recorded comparison.** Pool 100 has Δ −5.7 and z −2.0 for the no-battery variant. The source's proposed lead table has three entries and does not assign the battery result one. That comparison alone does not establish a missing register ID. Alpha already records it for an explicit lead decision. `MOVES:156-159`, `:198-208`, `READY:162-163`.
- **Run I status.** The cited follow-up is the anchor-6695 generation. The later owner ruling supplies cap authority for that specific run. I found no completion result in the named local records checked. The runbook documents a different run with the same name, so retain the anchor or runbook link when recording status. `LEADS:812-816`, `docs/drafts/anchor_v1/RUNBOOK_RUN_I_20261123.md:10-16`, `:61-69`, `logs/e2e_session/OWNER_GO_run_I_20261123.txt:1-4`, `logs/e2e_session/OWNER_RULINGS_pending_list_20261124.txt:5`.

## Other findings

2. **NEW — READY cites the wrong line for the grader pin.** The counted/development grader distinction is at `TASK_LOOP_ADDENDUM_3.txt:3`. Line 4 concerns selfplay. `READY:164-168`, `logs/e2e_session/TASK_LOOP_ADDENDUM_3.txt:3-4`.
3. **NEW — READY puts its prior record after its substantive summary.** The task itself puts the record at the top, and the owner rule gives it precedence. Keep the plain summary as the first substantive section after `## Prior record`. `READY:6-35`, `logs/e2e_session/TASK_LEADS_TIDY_20261125.txt:2`, `:6`, `CLAUDE.md:75-91`.

## Verification

I ran read-only Git comparisons, source and register SHA-256 checks, an entry diff for L-9, searches, and a read-only check of the four bundle commit fields. Both proposal hashes and all eight listed source hashes matched the request. The base register hash matched the first 654 proposed lines, and the diff was 318 additions with zero removals. I compared the document claims with the named drafts, tables, reports and task records. No code, suites, grader runs or experiment replays were in scope.
