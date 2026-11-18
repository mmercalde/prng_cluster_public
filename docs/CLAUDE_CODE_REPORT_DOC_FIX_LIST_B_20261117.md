# CLAUDE CODE REPORT — DOCUMENTATION FIX LIST, AGENT B — 2026-11-17

**Brief:** `/home/michael/doc_fix_brief_20261117.md` (Agent B: A6, C2–C6, D2, D3). **List:** `docs/DOC_FIX_LIST_20261113.md`.
**Worktree:** `/home/michael/tfm_worktrees/docs-b`, detached at `03f3712`. **Host:** VM101, as `michael`.
**Documentation only.** No code, config or JSON file was edited. Nothing was committed or pushed. Nothing ran on a GPU or a rig. `~/miner_staging` was not opened.
**Yardstick:** this report records what the documents and the code say. It makes no claim about whether TFM works.

## Summary

1. **A6 done in the tracked fact file:** the yardstick is now §0.0, §0.1 is reworded (the old text is kept as a quotation), §9 has item 23, and a notice at the top lists the 9 kinds of statement that predate the 2026-11-11 rulings. The installed copy and the Settings upload are still v27.1. Only you can update them.
2. **C5 and C6 were already done at HEAD** (2026-11-11 chapter corrections). For C6 I added one status note pointing to the repair's own brief (WP-2 Stage A).
3. **D2:** I added dated status notes to `CLAUDE.md` §6 and to the behaviour model. I did not rewrite the phase table (see Questions). **D3:** the catalog now has §9, which indexes the 26 documents committed on 2026-11-13. **148 other documents added since the catalog's anchor `0a4cef1` are still not indexed.**
4. **C2, C3 and C4 are JSON files, so they were not edited.** The proposed wording is below. C2 and C4 each need a decision from you first.
5. **Chapter 2 gate:** `python3 -m pytest tests/test_chapter2_content_gate.py -q` collects **no tests** (rc 0, "no tests ran"), so that command proves nothing. The file is a standalone script. Run as `python3 -u tests/test_chapter2_content_gate.py` it reports **12/12 green (6 gates + 6 mutants), SENTINEL: PASS**.

## Prior record

*(Placed second because the brief asks for the plain summary first. `CLAUDE.md` asks for this section to open the document. I am noting that conflict here rather than deciding it.)*

**Searches run** (worktree at `03f3712`; `/bin/grep`, because the shell `grep` skips `.json`):

- `grep -rln '2026-11-11' docs/ CLAUDE.md AGENTS.md`. Read: `AGENTS.md` §2 `:51-66`; `docs/exchange/RELAY_PROTOCOL.md:1-20`; `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §0–§2, §3.D `:110-120`, `:195-200`; `docs/PROPOSAL_DELEGATED_FLEET_EXECUTION_v1_1.md:3-8`.
- `grep -l -i 'TFM_PROJECT_FACTS_SKILL\|fact file'` over `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/PROJECT_FILE_CATALOG.md`, and `grep -rl` over `docs/exchange/`. The hits that bear on A6 were read: `CHAPTER_CORRECTIONS_REPORT_20261111.md:584-597`, `TODO_CONSOLIDATED.md:9`, completion plan `:116`, `:198`.
- `grep -n 'holdout_hits\|holdout_quality'` over Chapter 6, its audit, and the corrections report (C5). `grep -n 'prediction_pool\|next_draw_prediction'` over Chapter 13 (C6). Read the Stage A grading-contract report header and section list.
- `grep -rn 'found during sieve\|skip_min.*found during' docs/*.md` and `grep -n bidirectional_selectivity` over the chapter audits and the fact file (C2, C3). Read: `HYBRID_SKIP_BOUND_AUDIT.md:135-141`, fact file §2.22 and the 2026-08-08 note, behaviour model D12 (`:1188` before this session).
- `grep -n 'window_optimizer.json\|parameter_registry\|Brief II'` over Brief I, proposal v1.1 and the Brief I closure report (C4). Read: proposal v1.1 §4.6.
- `git diff --name-status 0a4cef1 HEAD -- docs/` and `git log --diff-filter=A` by date (D3). Read the heading and opening section of all 26 documents dated 2026-11-13.
- `git log --all -- '*REVIEW_REQUEST_REVERSE_SIEVE*' '*X-REVERSE-SIEVE_R1*'`: no ref.
- `grep -i 'run 4'` over the 2026-11-15/16 changelogs and `docs/LEADS.md`. Read L-6 and L-7. Read the `G-PROD-SHAPE` hits in `RUN4_LAUNCH_CHECKLIST.md`.
- `diff -q ~/.claude/skills/tfm-project-facts/SKILL.md docs/TFM_PROJECT_FACTS_SKILL.md`: **identical** before this session's edits (both v27.1, 3,227 lines).

**Documents found that already decide or record the subject:** the 2026-11-11 owner rulings (completion plan §0 rulings 1–6 and §2; `AGENTS.md` §2). The stale fact-file statements (`CHAPTER_CORRECTIONS_REPORT_20261111.md` §"Skill / fact file"; `TODO_CONSOLIDATED.md:9`). Completion-plan items D2 and D3 (`:115-116`), which are the same work as fix-list D2 and D3. The yardstick (`PITFALL_STATE_RECOVERY_YARDSTICK.md`; `CLAUDE.md`; `AGENTS.md`). C5 (Chapter 6, corrected 2026-11-11). C6 (Chapter 13 B1, corrected 2026-11-11; WP-2 Stage A). The C2 disagreement (fact file §0.4; behaviour model D12; `HYBRID_SKIP_BOUND_AUDIT.md` B7). C3 (fact file §2.22 note of 2026-08-08). C4's ownership (Brief I `:357-366` puts the `offset` registry entries in Brief II; proposal v1.1 §4.6).

**Labels:** every item below is **ALREADY RECORDED** in the fix list. These are the NEW observations: the pytest command collects nothing; `TEAM_ALPHA_REVIEW_REQUEST_REVERSE_SIEVE_20261113.md` and Beta's X-REVERSE-SIEVE R1 are not in any git ref; the catalog is missing 174 documents, not just one day's; the `feature_registry.json` skip description is config-stamped at both modes (detail under C2). Nothing here **CHANGES A RULING**.

**Not searched:** the host beyond reading `config_manifests/parameter_registry.json` in the main tree (it is gitignored, so the worktree has no copy), the ser8 archives, the private repository, the claude.ai Settings upload, and project files outside the repository.

## Items

| Item | State found at `03f3712` | What I changed (file:line after edit) | Source |
|---|---|---|---|
| **A6** fact file | v27.1 (2026-08-23). No yardstick. §0.1 named seed-finding as the aim. §9 had 22 items and none on the yardstick. The tracked and installed copies were byte-identical | `docs/TFM_PROJECT_FACTS_SKILL.md:8-10`: Currency line now v27.2, with v27.1's line kept after it. `:29-52`: the "PREDATES THE 2026-11-11 RULINGS" notice, 9 rows, each tied to the record that supersedes it, plus the three-copies reminder. `:73-109`: **§0.0 The yardstick** (the `CLAUDE.md` Block A text, with sources and the 2026-11-13 facts as status). It says the §0.2 "only the true seed survives" sentence and the §0.3 `{s*}` quotation are whitepaper arithmetic, not acceptance criteria. `:111-125`: §0.1 reworded; the v27.1 sentence is kept as a quotation. `:3300-3305`: §9 item 23. The individual stale sites are **not** rewritten; the notice lists them | Fix list A6; brief Agent B; `PITFALL_STATE_RECOVERY_YARDSTICK.md` §1–§7; realignment note §1–§3; `CLAUDE.md` "The yardstick"; completion plan §0–§2; `AGENTS.md` §2; `CHAPTER_CORRECTIONS_REPORT_20261111.md:584-597`; `TODO_CONSOLIDATED.md:9` |
| **C2** `skip_min`/`skip_max` wording | Disagreement still there: `config_manifests/feature_registry.json:336`, `:345` say "found during sieve analysis (from Step 2)"; `config_manifests/parameter_registry.json:160`, `:166` (main tree, gitignored, untracked) say "for sieve search" | Not edited (JSON). Wording below | Fix list C2 |
| **C3** `bidirectional_selectivity` | Still filed under `per_seed_features.bidirectional_sieve_metadata` (`feature_registry.json:17`, `:287`, `:317-325`). The code computes it once per trial and mode (`window_optimizer_integration_final.py:2221`, hybrid `:2325`) and copies it onto every record | Not edited (JSON). Wording below | Fix list C3; fact file §2.22 |
| **C4** `offset` descriptions | `parameter_registry.json:42`: "Position offset - advance seeds by offset*(skip+1) before testing". `agent_manifests/window_optimizer.json:93`: "Time offset from current draw position." Brief II, which owns these, is not written. Brief I acceptance waits on a production-shape run | Not edited (JSON). Conditional wording below | Fix list C4; Brief I `:357-366`; proposal v1.1 §4.6 |
| **C5** Chapter 6 target | **Already done** on 2026-11-11: `docs/CHAPTER_6_ANTI_OVERFIT_TRAINING.md:153-156`, `:204-212` (S1: `y = holdout_quality`), `:258-262`, `:662`, `:896`, `:1031-1032`, `:1086`, `:1152-1156` | None | `CHAPTER_6_AUDIT.md` S1; `CHAPTER_CORRECTIONS_REPORT_20261111.md` D-12 `:486` |
| **C6** Chapter 13 input | **Gap already documented** on 2026-11-11: `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md:135`, `:478-497`, `:1676` (B1) | Added `:500-509`: a status note saying the repair is CODE with its own brief (WP-2 Stage A brief and report, §Q8 contract not implemented, Stage B is a before-commit review class), and that this gap decides whether a hit rate can be measured | Fix list C6, E2; `CLAUDE_CODE_REPORT_STEP6_CH13_GRADING_CONTRACT_STAGE_A.md` |
| **D2** `CLAUDE.md` §6, behaviour model | `CLAUDE.md` §6 table still shows "3 worker daemon — **NEXT**" and names v1.4.4 as the spec. The catalog marks v1.4.4 SUPERSEDED (`PROJECT_FILE_CATALOG.md:338` before edit). The behaviour model is a snapshot of 2026-08-03 with no currency note | `CLAUDE.md:210-231`: a dated STALE note (Phases 3–6 built, Gate 12, v1.4.4 superseded, the 2026-11-11 definition of finished, Run 4 attempt 3 credited 2026-11-15, L-6/L-7 open, rule 3 unchanged). The table is kept as history. `docs/PIPELINE_BEHAVIOUR_MODEL.md:11-24`: a currency note (authority rulings, yardstick, where to find later status) | Fix list D2; completion plan item D2 `:115`; sources cited inside each note |
| **D3** catalog | No 2026-11 entries. Anchored at `0a4cef1` (2026-08-20). 174 `docs/` files added since | `docs/PROJECT_FILE_CATALOG.md:30-33`: a pointer. `:1090-1140`: **§9 addendum** indexing the 26 documents committed on 2026-11-13, each with the question it settles, and recording that two documents this list refers to are absent from the repository. The other 148 are named as an open gap | Fix list D3; the commit dates from `git log --diff-filter=A 0a4cef1..HEAD -- docs/` |

### C2, C3, C4: proposed wording (not applied)

**C2.** First a fact, which the fix list does not state: the value carried as a feature is the trial's **configured** search bound. It is not a value found per seed. `window_optimizer_integration_final.py:2203-2205` sets `'skip_min': config.skip_min`, `'skip_max': config.skip_max` in `metadata_base`, and Step 3 reads them as features at `full_scoring_worker.py:454`. Already recorded for variable skip at `HYBRID_SKIP_BOUND_AUDIT.md:135-139` (B3–B7), and as trial-level at `CHAPTER_14_AUDIT.md:204`. The design intent for the output reading, "gap that worked", is at `PROPOSAL_ML_Architecture_Remediation_v2_0.md:150-158` (via fact file §0.4). With that, the proposal is:

- `feature_registry.json` `skip_analysis.skip_min.description` →
  `"Lower bound of the trial's configured skip search range (config.skip_min), stamped identically on every survivor of the trial. Trial-level, not per-seed. Not a value found by the sieve."`
- `feature_registry.json` `skip_analysis.skip_max.description` → the same with "Upper bound" and `config.skip_max`.
- `parameter_registry.json` `skip_min.description` →
  `"Lower bound of the sieve's skip search (--skip-min). Input to the sieve. Its value is also stamped on survivors as the trial-level feature skip_min."`
- `parameter_registry.json` `skip_max.description` → the same with "Upper bound" and `--skip-max`.

This makes both files describe one quantity at two stages. It does **not** implement the "gap that worked" output reading. Whether the feature should carry the discovered value instead is a design decision (Questions, Q2).

**C3.** Relabel only, without moving the key. No tracked Python reads `feature_registry.json` (`grep -rn --include=*.py feature_registry`: no reader). I did not check whether any non-Python consumer reads the section path.

- `feature_registry.json` `bidirectional_selectivity.description` →
  `"TRIAL-LEVEL: forward_count / max(reverse_count, 1) for the trial and skip mode, computed once and stamped identically on every survivor of that trial (window_optimizer_integration_final.py, metadata_base). Listed under per_seed_features for history only; it cannot rank seeds within a trial."`
- `bidirectional_sieve_metadata._description` → append `" forward_count, reverse_count, bidirectional_count and bidirectional_selectivity are trial-level values stamped on every survivor (CHAPTER_14_AUDIT.md category B)."`

Moving the key into a trial-level section is a structural change to the JSON. I have not proposed it.

**C4.** This depends on Brief II, which is not written, so it is **not final**. If Brief II follows proposal v1.1 §4.6 as written, then:

- `parameter_registry.json` `offset.description` (`:42`) →
  `"LEGACY (pre-separation). Historically moved both the observed-data window start and, on phase-capable kernels, the generator pre-advance. Replaced by window_anchor (host: which records form the window; derived domain 0..N_filtered-window_size) and generator_phase (device; 0 in v1, not tunable). See PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md §4."`
- `agent_manifests/window_optimizer.json` `offset.description` (`:93`) → the same text. Its `"effect"` line ("may capture periodic PRNG reseeding patterns") should stay until you rule on it.

## Questions for the owner

1. **Fact file, three copies.** v27.2 is in the tracked copy only. Do you want it copied to `~/.claude/skills/tfm-project-facts/SKILL.md` (after a `SKILL.md.bak-v27.1` backup) and re-uploaded to Settings once you commit it? Fact file §7 procedure. I did not touch the installed copy.
2. **C2.** The feature now carries the configured bound. Should the descriptions say that (the wording above), or should the feature carry the discovered skip ("gap that worked")? The second is a code change on the sampler-comparison chain (fact file §8).
3. **C4.** Should I hold the `offset` wording until Brief II is written, or apply the conditional wording now?
4. **`CLAUDE.md` §6 table.** Do you want the table rewritten to current status, or left as history under the new note? Also, completion plan item D2 lists "rule 3 replaced, 25 GPUs" for `CLAUDE.md`. Rule 3 still says 26 GPUs and no autonomous launch, and the delegated-execution amendment is adopted in principle but not built. I did not change rule 3.
5. **Catalog.** Is a full regeneration (174 documents, under the catalog's REV2 brief) wanted, or is the §9 addendum enough for now?
6. **Missing documents.** `TEAM_ALPHA_REVIEW_REQUEST_REVERSE_SIEVE_20261113.md` (fix list D6: "keep both") and Beta's X-REVERSE-SIEVE R1 are in no git ref. Do you hold copies to commit?
7. **The test command.** The brief's pytest command collects 0 tests. Should future briefs use `python3 -u tests/test_chapter2_content_gate.py`?
8. **Report order.** The brief puts the plain summary first. `CLAUDE.md` says the Prior record section opens the document. Which governs for reports?

## Out of scope / not done

- **CODE items (not touched):** C1 (`prng_registry.py:3911`, `:3917` description strings). The C6 repair (Step 6 / Chapter 13, WP-2 Stage B). E1 (CPU and GPU skip semantics, `full_scoring_worker.py:305`, `survivor_scorer.py:124`). E2 (the grading contract). E3 (forward and reverse counts missing from the Step-3 merge list).
- **JSON items (proposed only):** C2, C3, C4 above.
- **Agent A's items:** A3, A4, A5, B1, B2, B3, B4, D4. I did not touch them, including the whitepaper sentences §0.0 now points to.
- **Not done in A6:** the individual stale sites the notice lists (§2.54, §7, §8, §2.53, and the corrections-report items) are flagged but not rewritten. The installed skill copy and the Settings upload are unchanged.
- **Not done in D3:** 148 documents added since `0a4cef1` (commits dated 2026-08-20 to 2026-11-16, other than 2026-11-13) are not indexed.
- **Not found:** a committed `G-PROD-SHAPE` result for Run 4 attempt 3, or a Brief I acceptance. I searched `docs/` at `03f3712` and did not search the host.

## Files changed

- `docs/TFM_PROJECT_FACTS_SKILL.md` (+83 / −5; the 5 removed lines are kept as quotations in the new text)
- `docs/CHAPTER_13_LIVE_FEEDBACK_LOOP.md` (+11)
- `docs/PIPELINE_BEHAVIOUR_MODEL.md` (+15)
- `docs/PROJECT_FILE_CATALOG.md` (+51)
- `CLAUDE.md` (+22)
- `docs/CLAUDE_CODE_REPORT_DOC_FIX_LIST_B_20261117.md` (this report, new)

**Gate evidence:** `python3 -u tests/test_chapter2_content_gate.py` → 12/12 green, `SENTINEL : PASS`, rc 0 (log: `/home/michael/tfm_worktrees/docs-b/data/doc_fix_b/ch2_gate.log`, gitignored). The brief's pytest form: "no tests ran", rc 0.
