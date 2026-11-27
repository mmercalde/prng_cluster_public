# X-W3-BINDING R1 - Alpha review request: DESIGN review before code (the why-report flag names the flagged model)

## Prior record

Searches run (in the worktree `/home/michael/tfm_worktrees/w3_binding_design`, main `14815a7`; the full list with scopes is
section 1.1 of the design note):
- `git grep -l -E -i 'CD-07|run.identity|checkpoint_sha256|flagged_model|flag.*binding|binding.*flag|unbound|parent_run_id|parent-run-id|diagnostics_run_id|output.dir.*compare|compare.*output.dir|TODO-WHY|W3'`
  over each of `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`,
  `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`,
  `docs/drafts/`, `logs/e2e_session/`.
- `git grep -n -E 'CD-07|run identity|checkpoint_sha256|flagged_model|unbound|parent_run_id|parent-run-id|diagnostics_run_id|TODO-WHY|X-W3|unit W3'`
  over the registers above, every hit read in context; then `git grep -n -E 'CD-07|run identity|diagnostics_run_id'` over
  the ten drafts and exchange files that matched.
- `/bin/grep -l -i -E "run id|run_id|cd-07|binding|checkpoint|flag"` over `logs/e2e_session/OWNER_RULING*.txt` in the main
  tree (git-ignored, VM101 only), hits read in full.
- Code searches: `diagnostics_run_id` and its flags in `*.py` and `*.json`; writers of `compare_models_summary.json`;
  `prediction_generator.py` reads of `agent_metadata` and `run_id`; `agents/registry/` for `diagnostics_level` and
  `diagnostics_run`; callers of `run_subprocess_comparison`.
- Read in full: `~/tfm_worktrees/_staging/ll78/PLAN_LAND_LL7_LL8_20261126.md` sections 6.3 to 6.9 (outside the repository).

Governing documents found:
- CD-07, WATCHER run identity and the stale guard: `docs/drafts/ch14_v1/PLAN_CH14_DIAGNOSTICS.md:174`; as built at level
  `full` only: `docs/drafts/loop_v1/READY_LOOP_DIAG_2_20261123.md:49-52, :80-84, :98, :135, :222-228`; WATCHER draft hunks:
  `docs/drafts/loop_v1/DRAFT_WATCHER_CD07_20261123.diff`; still an owner item: `docs/drafts/loop_v1/READY_LAND_CH14_DIAG_20261125.md:345-347`.
- Compare drops `--output-dir`: `docs/CHAPTER_6_AUDIT.md:134-139, :179-186` (B2). Legacy compare summary name and tree
  reports at `full` only: `PLAN_CH14_DIAGNOSTICS.md:173` (CD-06), `:187` (CD-05).
- Your review that found the wrong-checkpoint paths: `docs/exchange/from_beta/X-W2-HOOK_R2_BETA_REVIEW.md` Q3, Q4 and
  Finding 1; your earlier note that flag-time identity is the W2 gap: `docs/exchange/from_beta/X-W1-RUNNER_R1_BETA_REVIEW.md:62`.
- The owner's order for this feature: `docs/exchange/to_beta/X-W2-HOOK_R2_ALPHA_CLOSING_NOTE.md` (Q3 and Q4 line).
- What each side carries, the second-line problem and the owner's 2026-11-27 acceptance on the returned flag:
  `docs/drafts/unit_w/READY_W2_20261127.md:38-63` (Item 4), `:65-72` (Item 5), `:74-82` (Item 7).
- Fields the flag needs and the binding measure: plan sections 6.4 and 6.6 items 3 and 5 (outside the repository).
- Owner rulings: full diagnostics by default once its manifest entry exists (`OWNER_RULING_defaults_20261123.txt` item 3);
  registry before manifest (`OWNER_RULING_switch_registration_order_20261125.txt`); flag, never skip
  (`OWNER_RULING_benching_switch_20261126.txt`).
- Step 6 reads the sidecar's `agent_metadata.run_id`: `prediction_generator.py:321-325`.

Labels:
- WATCHER makes one run id per Step 5 dispatch and passes it to Step 5 and the health check: ALREADY RECORDED (CD-07, draft diff, owner order).
- The id is carried on `run_step`'s result, not on `self`, and WATCHER always overwrites a passed id: NEW.
- Stale guard and `stale` not resetting the skip registry: ALREADY RECORDED (CD-07 as built; draft diff hunk 2).
- Run id written into reports at level `legacy` when given: NEW.
- Sidecar block `step5_identity` (run id, checkpoint sha256, report sha256): NEW design of an ordered item.
- Per-run identity file `diagnostics_outputs/step5_identity_<run id>.json`: NEW.
- Compare per-family identity from each child's own sidecar: ALREADY RECORDED as an order; design NEW.
- Compare ignores `--output-dir` (B2): ALREADY RECORDED. Status, not fixed here.
- Archived sidecar resolving to the canonical file: ALREADY RECORDED (your R2 Q4); runner fix NEW.
- Identity on the one flag line (option A): CHANGES A RULING (owner's 2026-11-27 acceptance that the line on disk is
  unchanged, READY_W2 Item 7). Needs the owner. Option B (separate file): NEW.
- Registry before manifest: ALREADY RECORDED. Full level by default: ALREADY RECORDED, not needed here.

Not searched: the host and Proxmox, the rigs, ser8 archives, the private repository, project files outside the repository
other than the plan named above, and any live `diagnostics_outputs/` or `models/` on VM101.

## Request

Date: 2026-11-27. From: Team Alpha. To: Team Beta. This is a **DESIGN review before code**. No code is written for this
unit. Nothing is committed or landed. The owner ordered the feature built and asked for your review of the design first.

## What to review

The design note, read-only for you:
`/home/michael/tfm_worktrees/w3_binding_design/docs/drafts/unit_w/DESIGN_FLAG_MODEL_BINDING_20261127.md`

- Section 2: as-built facts on `14815a7`, each with file and line.
- Section 3: the binding rule. WATCHER's run id = report's run id = identity record's run id = sidecar's run id; flagged
  family = report's own family = sidecar's family; sha256 of the graded report bytes = recorded report sha256; sha256 of
  the checkpoint file next to the sidecar = recorded checkpoint sha256. Any link missing: "unbound", with the failed link
  named. The flag is never delayed or dropped.
- Section 4: the change table (C1 to C16). Owner-landed: the manifest's two `null` keys and about 15 WATCHER lines. Every
  new value is optional; with no id, files and argv are as on `14815a7`.
- Section 5: each wrong-model path and its gate and mutant.
- Section 6: what does not change. Section 7: landing and proof. Section 8: open questions.

The code changes, when written, will come to you again before commit (WATCHER launch path and Step 5 file writes).

## Questions (at most seven)

Q1. Is the binding rule in section 3 sufficient to say "this flag names the checkpoint whose diagnostics were graded"? Is
any link redundant, or is one missing, for example a link between the graded report and the exact weights for tree
families at level `full`, where the report is written before the checkpoint (`mpo:1907` then `:1926`)?

Q2. Which wrong-model path is missing from section 5? We list stale report, stale sidecar, compare winner and non-winner,
the S93 family override, relocated dir, archived sidecar pointing at the canonical path, two runs in one second, a Step 5
that fails after writing, a compare child that fails under a parent that exits 0, a skipped Step 5, a resumed pipeline, a
health retry, a report rewritten between grade and flag, a trial report carrying the run id, and a caller-passed id.

Q3. Is one run id per Step 5 dispatch, made in `run_step` and carried on its returned result, the right unit? In
particular: a health retry gets a new id, the skip count still spans attempts, and the flag names the checkpoint of the
attempt whose report raised it. Is that what the flag should mean?

Q4. Should identity be written at every diagnostics level (run id in the report when given; checkpoint and report sha256 in
the sidecar), rather than raising WATCHER's Step 5 to level `full`? Is it right that at `legacy` only the NN family can
ever be bound, because no tree report exists there?

Q5. Is there anything in Step 5 this must not disturb that section 6 misses: sidecar readers (Step 6, Chapter 13, the
inference bundle, the compare score scan), the compare archive and winner restore, the legacy report identity gates, or
the order of Step 5's file writes?

Q6. Where should the identity live for the flag: option A (on the one flag line, computed before the append; changes the
owner's 2026-11-27 line-unchanged acceptance) or option B (separate append-only file keyed by flag line)? Do you see a
reader of the flag file that option A would break?

Q7. Is hashing the checkpoint inside the health check acceptable, given that it runs synchronously before Step 6 and only
after every cheaper link has passed? If not, what bound do you want (a size or time cap that falls back to "unbound")?
