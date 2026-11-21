# PWC statements made false by Stages 1 and 2 — list for the owner (DRAFT; nothing edited)

**Date:** 2026-11-20 · worktree `/home/michael/tfm_worktrees/pwc-backup` at main `0caeaf9`. Read only: none of the
files below was edited. Written by the PWC agent (Claude Code).

## Prior record

**Searches run** (on `0caeaf9`; `/bin/grep`, because the shell `grep` skips `*.json`):

```
/bin/grep -rnIiE "PWC.{0,80}(quarantin|variable.skip.{0,30}(refus|block|fail)|THRESHOLD_CONTRACT_UNCERTIFIED|prun|forward.zero|Q0|
   balanced_hybrid|single strateg|seed.start|partitions from 0|offset|non-certifying|0\.5|phase2_threshold)|(quarantin|balanced_hybrid|
   forward.zero prune|Q0 gate).{0,80}PWC"  CLAUDE.md docs/PROJECT_FILE_CATALOG.md docs/CHAPTER_*.md docs/*REGISTER*.md docs/BACKLOG.md
   docs/TODO_CONSOLIDATED.md docs/LEADS.md docs/PIPELINE_BEHAVIOUR_MODEL.md docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md
   docs/HYBRID_SKIP_BOUND_AUDIT.md
/bin/grep -nIiE "PWC.{0,100}(quarantin|hybrid|balanced|prun|seed.start|0\.5|offset|worker_id)|PWC_HYBRID" docs/TFM_PROJECT_FACTS_SKILL.md
/bin/grep -rnIE "S147|Q0 gate|B1 gate|forward-zero|balanced_hybrid"  (chapters, catalog, TODO, PIPELINE_BEHAVIOUR_MODEL, STEP2 trace)
grep -n "pwc_protocol|worker_id" CLAUDE.md ;  git grep -n "DECLARED_CHANGED" ;  sed -n 60,97p docs/PROJECT_FILE_CATALOG.md
```

**What changed in code** (the reason each line below is stale):
- **`ee39404`, Stage 1:** `persistent/pwc_protocol.py:26` `worker_id: str = ""` (S7); the stale-worker kill no
  longer kills its own shell (B5).
- **`0caeaf9`, Stage 2:**
  - the variable-skip quarantine is **removed**, with the threshold contract in its place;
  - there is no `phase2_threshold` 0.5 default;
  - the forward-zero prune and the Q0 gate are **removed**;
  - all five strategies run (S147 Q2 removed);
  - a window offset ≠ 0 is refused.

  See `docs/drafts/pwc_backup_stage2_v1/README.md` for `file:line`.
- **Still true, and NOT stale:**
  - PWC is **non-certifying**: no credit until Stage 3, `window_optimizer_integration_final.py:2002-2006`;
  - the **seed-start refusal** stands at the launch layer (`:1960-1963`, `window_optimizer.py:678-712`);
  - RANGE-MINER is not WATCHER's default (`agent_manifests/window_optimizer.json:262`, `:270`);
  - the hybrid kernels' skip bounds are dead in both engines (`docs/HYBRID_SKIP_BOUND_AUDIT.md`).

  Lines that say only these are left out of this list.

**Not searched:** the claude.ai project files, including the live project-facts skill (the repo copy is
`docs/TFM_PROJECT_FACTS_SKILL.md`), ser8, the private repository.

## Stale statements: current documents (proposed new wording)

| # | File:line | Says now | Proposed wording |
|---|---|---|---|
| 1 | `CLAUDE.md:257-260` | "the public-clone `persistent/pwc_protocol.py` violates this (`worker_id` lacks a default) and cannot import on CPython ≥3.7; the miner sidesteps it. Do not copy that envelope verbatim." | "`persistent/pwc_protocol.py` used to violate this (`worker_id` had no default, so it could not import on CPython ≥3.7); fixed in `ee39404` (`worker_id: str = ""`, `persistent/pwc_protocol.py:26`). The miner keeps its own envelope." |
| 2 | `CLAUDE.md:99` | "Every session: write `SESSION_CHANGELOG_YYYYMMDD_SN.md`" | "Every session: write `SESSION_CHANGELOG_YYYYMMDD_<TOPIC>.md` (no new S-numbers; SR-2, `docs/PROJECT_FILE_CATALOG.md:90-97`)". This conflicts with a standing rule, independent of PWC |
| 3 | `docs/PROJECT_FILE_CATALOG.md:941` | "PWC is a flag-selectable **non-certifying** diagnostic, PWC hybrid additionally quarantined" | "PWC is a flag-selectable **non-certifying** diagnostic (no credit until PWC Stage 3). Its variable-skip quarantine was lifted in `0caeaf9` (owner ruling 1, 2026-11-19), replaced by a threshold contract" |
| 4 | `docs/TFM_PROJECT_FACTS_SKILL.md:474` (§2.7 #3) | "PWC hybrid filtered at `0.50` — **QUARANTINED** — … made loud rather than repaired" | "PWC hybrid filtered at `0.50` — **REPAIRED** in `0caeaf9`: no default; the trial's directional threshold as both keys; worker report checked (4 places, `sieve_filter.py:597-599`); quarantine removed (owner ruling 1)". The same change is needed in the live skill on claude.ai |
| 5 | `docs/TFM_PROJECT_FACTS_SKILL.md:631` | "PWC hybrid additionally quarantined." | "PWC variable-skip runs since `0caeaf9` under a threshold contract; PWC is still non-certifying until Stage 3." |
| 6 | `docs/PIPELINE_BEHAVIOUR_MODEL.md:300` | "non-certifying diagnostic; **hybrid quarantined**" | "non-certifying diagnostic; hybrid runs under the threshold contract (`0caeaf9`)" |
| 7 | `docs/PIPELINE_BEHAVIOUR_MODEL.md:1200` | "…non-certifying diagnostic, hybrid additionally quarantined" | "…non-certifying diagnostic; hybrid quarantine lifted `0caeaf9`" |
| 8 | `docs/TODO_CONSOLIDATED.md:435` | "PWC hybrid threshold quarantined (`PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`), deliberately not repaired" | "Repaired in `0caeaf9` (threshold contract); quarantine removed" |
| 9 | `docs/TODO_CONSOLIDATED.md:303` | Q0 zero-survivor branch proof; hybrid forward prune-rate monitoring; Q2 refinement phase "live in the PWC `run_trial_persistent` path" | "Moot since `0caeaf9`: Q0 and the forward-zero prune are removed and all five strategies run (owner ruling 4, default rule)" |
| 10 | `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1127` | "**PWC hybrid path** — **QUARANTINED** — raises rather than running … **no override flag**" | "**PWC hybrid path** — runs since `0caeaf9` under a threshold contract (the trial's directional threshold as both keys; the worker's report checked; SSH route refused). PWC is still non-certifying" |
| 11 | `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1420` | "**Non-certifying**; hybrid path quarantined (§8.3)" | "**Non-certifying**; hybrid under threshold contract (§8.3)" |
| 12 | `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:1692` | "non-certifying diagnostic; **hybrid quarantined** (§8.3.1)" | "non-certifying diagnostic; hybrid under threshold contract (§8.3.1)" |
| 13 | `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:2057-2058` | "…and its hybrid path is quarantined." | "…its hybrid path runs under a threshold contract since `0caeaf9`." The `:1632` anchor in `:2056` is also stale |
| 14 | `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1098-1101` | "Its hybrid path is **additionally quarantined** as `PWC_HYBRID_THRESHOLD_CONTRACT_UNCERTIFIED`, because it filters at a hardcoded `0.50`…" | "Its hybrid path was quarantined for filtering at a hardcoded `0.50`; since `0caeaf9` it uses the trial's directional threshold, checked against the worker's report (owner ruling 1)." |
| 15 | `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1120-1124` | "…its hybrid (variable-skip) passes are **quarantined**: `assert_pwc_hybrid_not_quarantined` raises… SWITCHED-OFF-BY-RULING" | "…its hybrid passes were quarantined (`8a55a68`) and run again since `0caeaf9` under a threshold contract (owner ruling 1). Still non-certifying." |
| 16 | `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1284-1286` | "On the PWC path this value is the `run_sieve_pass` default of 0.5 … which is why the path is quarantined" | "PWC no longer reaches `sieve_gpu_worker.py` for variable-skip: those passes refuse on the SSH route. On TCP the job carries the directional threshold (`0caeaf9`)" |
| 17 | `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1294` | "the two hybrid rows could not run on PWC today (quarantine)" | "…could not run on PWC until `0caeaf9`; on TCP they run now, on SSH they refuse" |
| 18 | `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1701-1705` | "**S-2 — PWC hybrid passes quarantined.** … *Lifting would let PWC run variable-skip passes again, filtering at the 0.5 default…*" | "**S-2 — PWC hybrid quarantine: LIFTED in `0caeaf9`** (owner ruling 1), together with the Option-A-style threshold propagation and check the note asked for." |
| 19 | `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1714` | "*…instead of the quarantined-for-hybrid PWC path.*" | "*…instead of the non-certifying PWC path.*" |
| 20 | `docs/HYBRID_SKIP_BOUND_AUDIT.md:99`, `:300` | "PWC variable-skip **cannot execute**"; "PWC variable-skip is already fail-closed … no *new* contaminated rows can come from that route" | Add a status line: "Since `0caeaf9` PWC variable-skip executes again (owner ruling 1). The skip-bound defect this audit records applies to it as it does to RANGE-MINER." **This changes the audit's own conclusion, which relied on the gate. The owner should decide the wording** |
| 21 | `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md:449` | "PWC: **only** `balanced_hybrid` (… `[S147 Q2]`)" | "PWC: all five from `get_all_strategies()` since `0caeaf9` (was `balanced_hybrid` only, S147 Q2)" |
| 22 | `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md:880-882`, `:1038-1039` (O9) | "`phase2_threshold: float = 0.5` is the declared default … `run_trial_persistent` never passes the argument" | "Until `0caeaf9`; now no default, and every pass gets its directional threshold" |
| 23 | `docs/STEP2_BIDIRECTIONAL_SIEVE_DESCRIPTIVE_TRACE.md:952-957` | the PWC forward-zero early return and the `[S147 Q0]` hybrid-forward-zero skip at `persistent_worker_coordinator.py:1627-1646`, `:1719-1723` | "Removed for PWC in `0caeaf9` (ruling 4); the legacy path's own Q0 at `window_optimizer_integration_final.py:953-955` is unchanged [check that line before editing]" |
| 24 | `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:46` | "Beta's R2-Q3 (five strategies) is NEEDS EVIDENCE" | "Beta's **R1-Q3** of X-PWC-STAGE2 (five strategies) is NEEDS EVIDENCE" (`docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md`; R2-Q3 was the refusal exit, NO OBJECTION)  **Done in this worktree 2026-11-20** (owner's request; one-line diff, uncommitted) |
| 25 | `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:40` (§1 table) | "WATCHER manifest defaults `use_range_miner: false`" | "Since Stage 3 (owner ruling 2026-11-20 15:53, \"Yes, range miner is the new default\"), the WATCHER manifest defaults `use_range_miner: true` and `use_persistent_workers: false`". Added on Beta X-PWC-STAGE3 R1 Finding 1 (`docs/exchange/from_beta/X-PWC-STAGE3_R1_BETA_REVIEW.md:71`) |
| 26 | `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:103` (C4) | "WATCHER manifest default `use_range_miner: false`" | "Done in Stage 3: the default is RANGE-MINER (owner ruling 2026-11-20 15:53)". Same Beta finding |

## Dated records (audits, reports): leave the text, consider a status note

These are records of what was true on their date. The project's catalog practice is to index records, not rewrite
them (`docs/PROJECT_FILE_CATALOG.md:60-67`). Proposed: no edit, or a one-line dated status note at the top of each.

| File:line | Says |
|---|---|
| `docs/CHAPTER_1_AUDIT_v1.md:201`, `:549` | PWC's hybrid path is quarantined |
| `docs/CHAPTER_2_SOURCE_MAP_v1.md:185-189`, `:362-363` | the PWC hybrid quarantine as a dependency |
| `docs/CHAPTER_9_AUDIT.md:94-98` | S-2, PWC hybrid passes quarantined |
| `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:442`, `:456`, `:596` | "a default run reaches the PWC hybrid quarantine"; D-05 |
| `docs/STEP1_EXECUTION_FLOW_AND_PRUNING_S147.md:141-176` | Q0 and Q2 as pending S147 fixes (historical) |

## Proposed catalog entries (for `docs/PROJECT_FILE_CATALOG.md`; NOT edited)

The catalog's newest section is an addendum for the documents of 2026-11-13 (`docs/PROJECT_FILE_CATALOG.md` §9,
"ADDENDUM — THE DOCUMENTS OF 2026-11-13"), laid out as `| File | Question it settles |` tables. Proposed: one more
addendum in the same layout, **"§10 ADDENDUM — PWC full backup records (2026-11-14 to 2026-11-20)"**. Paths are
relative to `docs/` as in §9. The adding commit is from `git log --diff-filter=A`.

**Rulings, scope and plans**

| File | Question it settles |
|---|---|
| `PWC_BACKUP_OWNER_RULINGS_20261119.md` (`ee39404`; update in `0caeaf9`) | ★ **What does PWC have to do as a backup?** Owner rulings 1–4 (2026-11-19) and the default rule "PWC does what RANGE-MINER does"; 2026-11-20 updates: shared-suite edits on PWC lines only (08:28), offset = draw window only and phase 0, refuse ≠ 0 until the split (08:30), no third Beta round (12:34) |
| `drafts/SCOPE_PWC_FULL_BACKUP_20261119.md` (`9b6b64c`) | What it takes to make PWC a full backup: items A1–A6, B1–B7, four stages, the shared-file table |
| `drafts/BRIEF_PWC_BACKUP_STAGE2_STAGE3_20261119.md` (`0caeaf9`) | The Stage 2 and Stage 3 change lists and F1–F6. **Corrected in the Stage 2 README:** the window/generator split needs a worker edit as well as `sieve_filter.py` |
| `drafts/NOTE_PWC_OLD_TEST_FAILURES_20261119.md` (`9b6b64c`) | Why three old PWC harnesses failed (stale tests, not code faults); repaired in Stage 1 |

**Implementation records (stage READMEs)**

| File | Question it settles |
|---|---|
| `drafts/pwc_cache_reset_v1/README.md` · `persistent_worker_coordinator.diff` (`7444aea`) | PWC clears the rig CuPy cache before worker start (fresh cache per run, LEADS L-7); the known TCP reconnect limit |
| `drafts/pwc_backup_stage1_v1/README.md` · `pwc_backup_stage1.diff` · `HELD_pwc_worker_service.diff` · `before_after_tally.md` · `before_after_failing_checks.txt` (`ee39404`) | ★ **Stage 1, as landed:** S1–S8; B5 (the rig kill killed its own shell); A3 level 1; B6; A4; the held worker hunk; 71-suite before/after |
| `drafts/pwc_backup_stage2_v1/README.md` · `pwc_backup_stage2.diff` · `before_after_tally{,_r1,_r2}.md` · `before_after_failing_checks{,_r1,_r2}.txt` (`0caeaf9`) | ★ **Stage 2, as landed:** rulings 1 and 4 in code, five strategies, the offset refusal, Beta's Stage 1 and Stage 2 items, rounds 2 and 3, the files list, the redeploy steps, open owner questions; three 72-suite before/after passes |

**Beta threads** (`exchange/`)

| File | Question it settles |
|---|---|
| `to_beta/X-PWC-BACKUP_R1_ALPHA_REVIEW_REQUEST_pwc_step1.md` · `from_beta/X-PWC-BACKUP_R1_BETA_REVIEW.md` · `to_beta/X-PWC-BACKUP_R2_ALPHA_REPLY.md` · `from_beta/X-PWC-BACKUP_R2_BETA_REVIEW.md` (`5bb854c`, 2026-11-14) | Beta on the step-1 candidate (S1–S8) that Stage 1 later landed |
| `to_beta/X-PWC-CACHE-RESET_R1_ALPHA_REVIEW_REQUEST_pwc_cache_reset_v1.md` · `from_beta/X-PWC-CACHE-RESET_R1_BETA_REVIEW.md` · `to_beta/X-PWC-CACHE-RESET_R2_ALPHA_REVIEW_REQUEST_real_world_scope.md` · `from_beta/X-PWC-CACHE-RESET_R2_BETA_REVIEW.md` (`7444aea`) | Beta on the per-run rig cache reset; R2 under the owner's real-world framing; no objection |
| `to_beta/X-PWC-STAGE1_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md` · `from_beta/X-PWC-STAGE1_R1_BETA_REVIEW.md` (`ee39404`) | Beta on Stage 1: no objection; three suggestions carried to Stage 2 (fail-open probe, revoked connection, refused-worker retry) |
| `to_beta/X-PWC-STAGE2_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md` · `from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md` · `to_beta/X-PWC-STAGE2_R2_ALPHA_REPLY.md` · `from_beta/X-PWC-STAGE2_R2_BETA_REVIEW.md` · `to_beta/X-PWC-STAGE2_R3_ALPHA_CLOSING_NOTE.md` (`0caeaf9`) | Beta on Stage 2: R1 Q6 and finding 2 LIVE (fixed in R2), **R1-Q3 NEEDS EVIDENCE (five strategies on the rigs, open for Stage 4)**; R2-Q1 LIVE (fixed as Beta specified, no third round, closing note not sent) |

**Code records for §4.6 `tests/` and §4.7 `scripts/`** (the catalog lists tests and scripts there)

| File | Role |
|---|---|
| `scripts/pwc_parity_gate.py` (`ee39404`) | PWC's own rig parity gate (`--transport tcp\|ssh`): deployed bytes of the files each PWC worker entry point reaches. Separate from `scripts/gate12_parity_gate.py` |
| `tests/test_pwc_cache_reset.py` (`7444aea`) | The cache reset, 21 checks |
| `tests/test_pwc_backup_step1.py` (`ee39404`) | S1–S8, 22 gates, 16 mutants (2 HELD) |
| `tests/test_pwc_stage1.py` (`ee39404`) | B5, A3 level 1, B6; private-PID-namespace process checks; 9 gates, 8 mutants |
| `tests/test_pwc_backup_stage2.py` (`0caeaf9`) | Stage 2 plus Beta rounds 2 and 3; 26 gates, 29 mutants |

**Not yet landed, so not for the catalog yet:**
- `TFM_SESSION_HANDOFF_20261120.md`, `SESSION_CHANGELOG_20261120_PWC_BACKUP.md`;
- `drafts/pwc_backup_stage3_v1/PLAN.md`, `drafts/pwc_backup_stage3_v1/STALE_DOCS.md` (this worktree);
- Stage 2b's records in `/home/michael/tfm_worktrees/pwc-offset`.

When the changelog lands, §3.13 (session changelogs, "summarised as a group") gains one file.

## Standing repo rules that apply

- **SR-2, changelog naming** (`docs/PROJECT_FILE_CATALOG.md:90-97`; `TB_RULING_CHANGELOG_NUMBERING.md`): a changelog
  for today's PWC work would be `docs/SESSION_CHANGELOG_20261120_<TOPIC>.md`, for example `..._PWC_BACKUP.md`, with
  **no S-number**. None exists for Stage 1 or Stage 2 (`ls docs/SESSION_CHANGELOG_20261119* docs/SESSION_CHANGELOG_20261120*`
  finds none). `CLAUDE.md:99` contradicts SR-2 (row 2 above).
- **SR-1, `DECLARED_CHANGED`** (`docs/PROJECT_FILE_CATALOG.md:74-88`; `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:20-29`):
  it binds commits that change a definition in `miner/range_miner_coordinator.py` covered by a historical scope gate
  (the sets live at `tests/test_s172_mp1_drain_attribution.py:1407` and `tests/test_s172_r1_drain_remedy.py:2129`).
  **It does not apply to Stage 1 or Stage 2**, which did not touch `miner/`, **nor to Stage 3 as planned**
  (`docs/drafts/pwc_backup_stage3_v1/PLAN.md` edits no `miner/` file).
- **Catalog:** the new PWC records are not yet indexed in `docs/PROJECT_FILE_CATALOG.md`. Proposed entries are in
  the section above.

## Not verified

- That no other document (outside the grep scope above) states the quarantine, prune or single strategy as current.
- The live claude.ai project-facts skill: not readable from here.
- Row 23's legacy Q0 line (`window_optimizer_integration_final.py:953-955`): the anchor is the trace's, not re-read.
