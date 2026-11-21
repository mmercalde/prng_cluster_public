<!--
Assembled 2026-11-21 by agent 1 (Team Alpha) from agent 1's Stage 3b section and agent 2's Stage 3a section (copied
unchanged from /home/michael/tfm_worktrees/pwc-offset). Relay approved by the owner, 2026-11-21 05:56 PST, his words:
"yes please send to beta". One round.
-->

# X-PWC-STAGE3 · R1 · ALPHA REVIEW REQUEST · real-world scope

**Owner approval to relay:** 2026-11-21 05:56 PST, verbatim: **"yes please send to beta"**. One round.

## Prior record

**Searches run** (`/bin/grep`, combined worktree at `b2efdef`). Pattern
`X-PWC-STAGE3|PWC.STAGE3|pwc_backup_stage3|use_range_miner.*default|pwc_coordinator_address|pwc_trial_completion|EXEC-PIN-1|keep-block|SWITCHED-OFF-BY-RULING`:

```
/bin/grep -rlIE "$S" docs/PROJECT_FILE_CATALOG.md   # 0
/bin/grep -rlIE "$S" docs/TB_RULING_*               # 3: RUN4_ROUTING_PATCH_BRIEF_REVIEW, RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED, WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW
/bin/grep -rlIE "$S" docs/TB_RULING_REQUEST_*       # 0
/bin/grep -rlIE "$S" docs/PROPOSAL_*                # 0
/bin/grep -rlIE "$S" docs/CHAPTER_* docs/*_AUDIT.md # 25 / 12 (SWITCHED-OFF-BY-RULING in 25 chapters; the default in CHAPTER_9/10/12_AUDIT)
/bin/grep -rlIE "$S" docs/TODO_CONSOLIDATED.md      # 1 (:93 default; EXEC-PIN-1 :53, :138, :252, :436)
/bin/grep -rlIE "$S" docs/BACKLOG.md docs/LEADS.md  # 0 / 0
/bin/grep -rlIE "$S" docs/exchange/                 # 2: from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md, to_beta/X-PWC-STAGE1_R1_...
```

The section-level records for each stage follow in the 3a and 3b sections. Hits read and labelled:

| Document | What it records | Label |
|---|---|---|
| `docs/TODO_CONSOLIDATED.md:93`; `docs/CHAPTER_12_AUDIT.md:31`, `:42`; `docs/CHAPTER_10_AUDIT.md:192`; `docs/CHAPTER_9_AUDIT.md:27`; `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:436`, `:503` | The Step 1 default is PWC; RANGE-MINER by explicit override only, until after Phase 7 | **CHANGES A RULING**, by the owner (15:53, below). The doc updates are listed in `docs/drafts/pwc_backup_stage3_v1/STALE_DOCS.md` |
| `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md:86-100` (§9.4) | EXEC-PIN-1, the standing rule | ALREADY RECORDED. Answered below (Finding 1) |
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:17`, `:83`; `..._BRIEF_REVIEW.md:13`, `:25`, `:71-75` | EXEC-PIN-1 applied to the Run 4 routing patch; `default_params` untouched for the seven warm-start keys (`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R1.md:14`, `:94`) | ALREADY RECORDED. Engine choice is not one of the seven keys; 3b-Q2 asks you to confirm |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md` | Your carried items: Q5, Q8, Finding 1, Finding 2, Q3 | ALREADY RECORDED. Each is answered below |
| `docs/CHAPTER_12_WATCHER_AGENT.md:1220`; `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:20`; `docs/BACKLOG.md:298-300` | n_parallel > 1 is SWITCHED-OFF-BY-RULING on the miner path | ALREADY RECORDED. Out of scope (20:24) |
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80` (F-1) | RANGE-MINER's constant phases never receive the trial's skip range | ALREADY RECORDED, open. The fix is scoped separately (21:32-21:35) |

**Not searched:** the host, ser8 archives, the private repository, claude.ai project files, the rigs (powered off),
Proxmox. The owner rulings of 2026-11-20 are in `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md` (combined tree,
uncommitted) and the handoff `docs/TFM_SESSION_HANDOFF_20261120.md`; a few exist only in relayed chat, as marked.

## The request

- **Review class:** before commit. Covers the launch path (`window_optimizer.py`,
  `window_optimizer_integration_final.py`, `agent_manifests/window_optimizer.json`, one WATCHER keyword) and PWC worker
  code (`sieve_filter.py`, new-key branch).
- **Base:** main `b2efdef` (`b2efdefbab207bb82ed8d307d295737bbe6111fd`). Nothing is committed.
- **Tree to review:** the combined worktree **`/home/michael/tfm_worktrees/pwc-backup`**. It holds Stage 3b plus agent 2's
  Stage 3a files, copied unchanged and verified by sha256 (20/20) against `/home/michael/tfm_worktrees/pwc-offset`;
  re-verified 20/20 on 2026-11-21 after a VM101 reboot (~02:35), when `tests/test_pwc_backup_stage3.py` was re-run:
  14/14 gates, 18/18 mutants.
  The absolute paths in the 3a section name `pwc-offset`; the same bytes are at the same relative paths in
  `pwc-backup`. Diff: `git -C /home/michael/tfm_worktrees/pwc-backup diff b2efdef`, plus the untracked files listed
  below.
- **Scratch location:** `TMPDIR` for this run is `/home/michael/tfm_beta_scratch/X-PWC-STAGE3_R1/tmp`, outside the
  repository (`AGENTS.md` §3). Please put throwaway checks and fixture copies there or under `/tmp`, not in this
  worktree. Some suites write into the repository or into the main tree's `optuna_studies/` when run here. If you
  run one, say which in your review.
- **Yardstick:** TFM is judged by whether its prediction pools hit more often than `k/1000` on draws the sieve never
  saw. Nothing here is judged by whether a seed is found or survives. No PWC pool has a hit-rate measurement. The
  two-engine comparison (owner, 15:55) is Stage 4, run on rigs by the owner.
- **Allowed answers:** NO OBJECTION · OBJECTION · SUGGESTION · NEEDS EVIDENCE, with LIVE or THEORETICAL on every
  OBJECTION. Owner rulings below are fixed facts, not under review. Michael is the final authority; your review is
  independent and not binding.

`git status --porcelain` of the combined tree at assembly:

```text
 M agent_manifests/window_optimizer.json
 M agents/watcher_agent.py
 M docs/PWC_BACKUP_OWNER_RULINGS_20261119.md
 M docs/VM101_HEADLESS_STARTUP_20261119.md
 M scripts/prove_unchanged_outside.py
 M sieve_filter.py
 M test_persistent_worker_harness.py
 M tests/test_prove_unchanged_outside.py
 M tests/test_s172_miner_seed_start.py
 M tests/test_s172_phase1_scaffolding.py
 M tests/test_step1_search_record_stage1.py
 M window_optimizer.py
 M window_optimizer_integration_final.py
?? data/
?? docs/SESSION_CHANGELOG_20261120_PWC_BACKUP.md
?? docs/TFM_SESSION_HANDOFF_20261120.md
?? docs/drafts/pwc_backup_stage3_v1/
?? docs/drafts/pwc_backup_stage3a_v1/
?? docs/drafts/pwc_backup_stage3b_v1/
?? docs/exchange/to_beta/X-PWC-STAGE3_R1_ALPHA_REVIEW_REQUEST_real_world_scope.md
?? scripts/compare_engine_outputs.py
?? tests/test_compare_engine_outputs.py
?? tests/test_pwc_backup_stage3.py
?? tests/test_pwc_stage3a.py
```

## Owner rulings of 2026-11-20 (fixed facts, with their times, PST)

| Time | Ruling (his words in quotes) | Effect on Stage 3 | Recorded at |
|---|---|---|---|
| 05:34 | Send Stage 1 to Beta, one round | Process | relayed chat |
| 06:24 | Stage 1 lands (`ee39404`) | — | rulings file, Status |
| 08:28 | Shared suites: edit only the named PWC checks; every RANGE-MINER check stays identical | The same rule is applied to the three suites edited in 3b | rulings file `:43` |
| 08:30 | Offset = draw window only (`window_anchor`); `generator_phase` stays 0, as RANGE-MINER | Landed in 2b; 3a's Finding 2 fix keeps it | rulings file `:44` |
| 10:15 | Send Stage 2 to Beta, one round | Process | relayed chat |
| 10:47 | Fix Q6, Finding 2, Q7 and five governing docs, then one narrow second round | Process (Stage 2) | relayed chat |
| 12:34 | Your R2-Q1 is fixed as you specified; Stage 2 lands with no third round (`0caeaf9`) | — | rulings file `:45` |
| 15:53 | **"Yes, range miner is the new default"** | RANGE-MINER is the Step 1 default; PWC runs only when its flag asks for it (3b) | rulings file `:46` |
| 15:55 | **"And once finished your testing should run each and verify the results are the same"** | Stage 4: the same job through both engines, each chosen by its flag, outputs compared. Not run | rulings file `:47` |
| 16:45 | "Yes, PWC follows range miner. Unless you have a other way to verify they both return the same results" (F4) | **Superseded for S3-13** at 21:32-21:35 | rulings file `:48` |
| 17:38 | The 08:30 ruling lifts the recorded Brief II wait for PWC only; one Beta round for 2B | Process (2B) | `docs/drafts/pwc_backup_stage2b_v1/README.md:398` |
| (2B landing) | No second 2B round; your Q3 is THEORETICAL and deferred | 2B landed as `b2efdef` | commit message `b2efdef` |
| 2026-11-20, time not recorded | Edit only the PWC lines of `tests/test_s172_miner_seed_start.py`, `tests/test_step1_search_record_stage1.py`, `tests/test_s172_phase1_scaffolding.py`; add one keyword to `agents/watcher_agent.py`; leave S3-6 and S3-7 for later; edit T20 only in `test_persistent_worker_harness.py` | The 3b test and WATCHER edits | relayed chat; 3b README |
| 20:24 | **"Yes"**: finish the PWC backup first, then re-enable n_parallel > 1 as its own job. That job also fixes S3-6 (the self-killing `pkill`) and S3-7 (the partition dial address) | Parallel trials are out of Stage 3 | rulings file `:49` |
| 20:37 | **"Or better half space seed search- that IS tested"** | That later job targets half-range trials. Half range is tested at n_parallel 1 only | rulings file `:50` |
| 20:41 | **"I agree. Since we can search the complete seed space in two passes parallel runs doesn't make a lot of sense. Now.if it's finding multiple windows that's different"** | Parallel groups are LOW priority | handoff, open items |
| 21:32-21:35 | **"Well range miner must be fixed none the less"**; **"Please do whatever method is the quickest and least recoding"** | **S3-13 withdrawn.** F4 now stands as: PWC keeps the trial's skip range; the RANGE-MINER fix is scoped separately in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md` and is not part of this review | rulings file `:51`; 3a section |
| after 21:35 (2026-11-20/21), time not recorded | The 3a proof tool is narrowed: a non-inert header default is flagged only when its AST differs from the base, or the parameter is new (owner's choice of the three options put to him) | See "Combined-tree status" | relayed chat |
| 2026-11-21 05:56 | **"yes please send to beta"** | This request is relayed, one round | relayed chat |

Standing (2026-11-19): the four rulings (1 lift the hybrid block; 2 PWC earns credit; 3 no seed-start refusal; 4
phases follow RANGE-MINER), "DO NOT break range miner", and the default rule: **PWC does what RANGE-MINER does**.

## Your carried items from X-PWC-STAGE2B R1, each answered

| Item | What you asked | Answer in Stage 3 | Evidence |
|---|---|---|---|
| **Q5** (SUGGESTION) | Check completion before candidates enter the accumulator and before the trial is scored. This covers the mixed-version case, where S1 scored 2 from partial populations with all four passes incomplete | **Done (3b).** The PWC block raises `PwcTrialIncompleteError` unless the trial is complete, schema `pwc_trial_completion_v1`, and every required pass is complete and ran on TCP. It raises `PwcThresholdProvenanceError` unless every required pass has a verified threshold. Both checks run before `_build_test_result_from_pw`, so nothing reaches the accumulator and Optuna sees an exception, as with RANGE-MINER's `MinerIngressError` (`miner/step1_ingress.py:192-200`). Credit is read from PWC's own `swept_intervals`, only for such a trial | `window_optimizer_integration_final.py:2072-2083`; G-INCOMPLETE, G-PROV, G-CREDIT, M-PWC-*; 3b-S4, 3b-S5, 3b-Q4 to Q6 |
| **Q8** (SUGGESTION) | Proof-tool blind spots; keep a separate comparison of the miner subtree and its preamble when `run_bidirectional_test` is allowed | **Done in both halves.** 3a: four new tool checks plus `--keep-block` (3a-Q4, 3a-Q5). 3b: (i) the tool's subtree mode: `if _use_miner:` and its preamble are identical to `b2efdef`, and two valid control mutants (one in the block, one in the preamble) are refused; (ii) a statement-by-statement AST comparison of the whole function, which also covers the statements after the PWC block (ZMQ, legacy): every statement outside `if _use_pw:` is identical; (iii) runtime equality of coordinator settings and `run_trial_miner` arguments | `docs/drafts/pwc_backup_stage3b_v1/prove_unchanged_outside_subtree_woi.txt`, `miner_subtree_statement_comparison.txt`; G-MINER-SUBTREE, G-MINER-SETTINGS, G-MINER-JOBS |
| **Finding 1** (EXEC-PIN-1) | Record the rule and re-evaluate the pinned executable comparisons | **Done (3b, for both halves).** Mechanical static over-approximation: 18 test files execute pinned source; 46 candidate (revision, file) arms name a changed module and one of its changed names. The only signature change is one trailing optional keyword (`pwc_coordinator_address=None` on `run_bayesian_optimization` and `run_with_config`), so every pinned caller stays call-compatible. The behaviour-contract changes (seed-start guard for PWC, PWC completion, WATCHER fleet check, 3a's one-read job, the stricter tool) affect an arm only if it drives the PWC route through the live name. In the full Stage 3b pass, all 18 arm files have unchanged failing checks. No bridge was needed or added. This is evidence of present compatibility, not proof of isolation | `docs/drafts/pwc_backup_stage3b_v1/exec_pin_1_scan.txt` |
| **Finding 2** (THEORETICAL) | The new-key job validates on one read and selects on a second | **Done (3a).** One parsed, session-filtered read gives the short-file refusal, the domain and the window, as `miner/range_miner_worker.py:904-932` | 3a section; G-F2-*, M-F2-*; 3a-Q2 |
| **Q3** (OBJECTION, THEORETICAL) | A fractional anchor or phase is truncated to 0 | **Deferred by the owner** at the 2B landing; not changed in Stage 3. The cluster coordinator sends integers only | commit `b2efdef` |

## Combined-tree status (3a + 3b), focused suites only

Run sequentially on CPU, with ssh/scp/rsync stubbed. Logs are in `docs/drafts/pwc_backup_stage3b_v1/combined_focused/`.

| Suite | Result |
|---|---|
| `tests/test_pwc_stage3a.py` | 6/6, 3/3 mutants |
| `tests/test_prove_unchanged_outside.py` | 49/49 (tool as narrowed 2026-11-21 00:32) |
| `tests/test_pwc_stage2b_anchor_phase.py` | 35/35 |
| `tests/test_pwc_backup_stage2.py` | 26/26, 33/33 mutants |
| `tests/test_pwc_backup_step1.py` | 22/22, 16/16 mutants (2 HELD, as at base) |
| `test_persistent_worker_harness.py` | 25/25 (after the owner-approved T20 edit) |
| `tests/test_compare_engine_outputs.py` | 13/13 |
| `tests/test_pwc_backup_stage3.py` | 14/14, 18/18 mutants (rerun on the narrowed tool) |

**Resolved:** on the first combined run, 3a's header check flagged `run_bayesian_optimization`'s unchanged default `staging_high_water_bytes = 16 * 1024 ** 3` (`window_optimizer.py:759`, the same as `b2efdef:756`). That turned G-MINER-DEFS and M-MINER-3 red (13/14, 17/18). The owner chose to narrow the tool. Agent 2 narrowed it (2026-11-21 00:32): it now flags a default only when its AST differs from the base, or the parameter is new. See 3a-Q4b. The re-copied tool and suite are sha256-equal to pwc-offset. Reruns: `test_pwc_backup_stage3` 14/14 + 18/18; `test_prove_unchanged_outside` 49/49; `test_pwc_stage3a` 6/6 + 3/3. Subtree proof: `if _use_miner:` and its preamble are identical to `b2efdef`; mutants inside the block (`:1843`) and in the preamble (`:1841`) are refused (`docs/drafts/pwc_backup_stage3b_v1/prove_unchanged_outside_subtree_woi.txt`). The other five focused suites were run on the combined tree before the narrowing, which touched only the tool and its suite: 35/35, 26/26 + 33/33, 22/22 + 16/16, 25/25, 13/13.

**Not run on the combined tree:** a full pass. Each half had its own full before/after pass on clean `b2efdef`
(3a: `docs/drafts/pwc_backup_stage3a_v1/before_after_tally.md`; 3b: `docs/drafts/pwc_backup_stage3b_v1/before_after_tally.md`).
**UNAVAILABLE:** GPU, rigs, a real WATCHER run on the new default, hit rate.

## Positions table (please fill in)

| Question | Position | Severity |
|---|---|---|
| 3a-Q1 RANGE-MINER (3a) | | |
| 3a-Q2 Finding 2 | | |
| 3a-Q3 Old-style jobs | | |
| 3a-Q4 Proof tool | | |
| 3a-Q4b The narrowed header check | | |
| 3a-Q5 `--keep-block` | | |
| 3a-Q6 Test edits (3a) | | |
| 3b-Q1 RANGE-MINER, byte for byte | | |
| 3b-Q2 The new default | | |
| 3b-Q3 Choosing PWC | | |
| 3b-Q4 Your Q5 | | |
| 3b-Q5 Credit | | |
| 3b-Q6 One-sided deployment | | |
| 3b-Q7 The ready gate | | |
| 3b-Q8 Test edits (3b) | | |
| C-Q1 Carried items: is each answered as you asked? | | |
| C-Q2 W-NO-WEAKENING and the gate12 launch gates after commit | | |

## C-Q2 — W-NO-WEAKENING on the uncommitted tree, and the RANGE-MINER launch gates after commit

`tests/test_gate12_cleantree_admission.py` W-NO-WEAKENING is red on the uncommitted tree because
`window_optimizer_integration_final.py` differs from HEAD. We expect it to clear on commit. **Is there any
RANGE-MINER launch gate that this file's change would trip after commit (the `gate12_launch.sh` clean-tree and parity
checks)?**

Our reading, before you answer (`docs/drafts/pwc_backup_stage3b_v1/gate12_launch_gates_reading.txt`):
1. W-NO-WEAKENING (`tests/test_gate12_cleantree_admission.py:937-958`) compares working-tree hashes with
   `git show HEAD:` (`:941-951`). After commit, HEAD holds the same bytes. Read, not executed.
2. The file is on that list because the clean-tree launch gate imports the whole module
   (`scripts/gate12_cleantree_gate.py:77`) and takes its verdict from `WOI._repository_state` (`:81`, `:117`; defined
   `window_optimizer_integration_final.py:101`).
   - `_repository_state` is AST-identical to `b2efdef`.
   - Stage 3b's other top-level additions (2 classes, 1 literal assignment, 5 functions) and the changed
     `run_bidirectional_test` body run no code at import. The 3a tool's import-time check passes on the file.
   - So the admission and pre-dispatch clean-tree checks (`gate12_launch.sh:133-136`, `:246-249`) compute the same
     predicate: PROCEED on the committed clean tree, REFUSE on the uncommitted one, as they should.
3. The rig code-parity gate (`gate12_launch.sh:191-198`; `scripts/gate12_parity_gate.py:174-185`) governs 10 files,
   none of them a Stage 3 file. Its static closure derivation from `miner/range_miner_worker.py`, run on the combined
   tree, gives exactly those 10 files: `closure_coverage = (True, [], [])`.
4. The provenance block (`gate12_launch.sh:286-312`) records HEAD and the porcelain count, and refuses only if git fails.

Our conclusion: no `gate12_launch.sh` gate is tripped after commit. Please check by your own reading, including any
gate or launcher path we did not read (`scripts/launch_fleet_manual.sh` was not re-read for this question).

---

## Stage 3a — the PWC-only items (worktree `/home/michael/tfm_worktrees/pwc-offset`, base `b2efdef`)

### Fixed facts for 3a (owner; not under review)

- **S3-13 is withdrawn.** Owner, 2026-11-20 21:32-21:35 PST, verbatim: "Well range miner must be fixed none the less"
  and "Please do whatever method is the quickest and least recoding".
  - This supersedes his 16:45 ruling ("Yes, PWC follows range miner. Unless you have a other way to verify they both
    return the same results") **for S3-13 only**.
  - The reason: the miner's constant phases not receiving the trial's skip range is a recorded open finding, F-1
    (`docs/TFM_SESSION_HANDOFF_20261114.md:76-80`; `miner/range_miner_worker.py:1293` falls back to `[0, 16]`).
    Copying it into PWC would be undone once RANGE-MINER is fixed.
  - PWC therefore sends the trial's skip range exactly as at `b2efdef`. `persistent_worker_coordinator.py` is
    byte-identical to `b2efdef`.
  - The miner fix is a separate future change, scoped read-only in
    `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`. It is not part of this review.
- **The default rule** for anything unruled: PWC does what RANGE-MINER does.

### Prior record (3a items; the combined record above covers the governance search)

| Document | What it settles | Label |
|---|---|---|
| Owner, 2026-11-20 21:32-21:35 PST (above) | S3-13 withdrawn | Owner decision. Supersedes 16:45 for S3-13 only |
| `docs/TFM_SESSION_HANDOFF_20261114.md:76-80` (F-1) | The miner's phases 1-2 never see the trial's skip range | ALREADY RECORDED, open (miner side) |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:47` (Finding 2), `:67` (Q8) | Your two-read finding and the proof-tool limits | ALREADY RECORDED. Addressed here |
| `docs/drafts/pwc_backup_stage2b_v1/README.md` "Beta round 1" | S3-11's coordinator side landed in 2b; SSH keeps 0-only | ALREADY RECORDED |

### Files under review (absolute paths; `git -C /home/michael/tfm_worktrees/pwc-offset diff b2efdef`)

| File | Change |
|---|---|
| `/home/michael/tfm_worktrees/pwc-offset/docs/drafts/pwc_backup_stage3a_v1/README.md` | the package README; read it first |
| `/home/michael/tfm_worktrees/pwc-offset/sieve_filter.py` | `execute_sieve_job`, new-key branch only: one dataset read |
| `/home/michael/tfm_worktrees/pwc-offset/scripts/prove_unchanged_outside.py`, `.../tests/test_prove_unchanged_outside.py` | four new checks, including `--keep-block` |
| `/home/michael/tfm_worktrees/pwc-offset/tests/test_pwc_stage3a.py` | new |

`persistent_worker_coordinator.py` is **not** changed (byte-identical to `b2efdef`).

Evidence is in `docs/drafts/pwc_backup_stage3a_v1/`:
- `prove_unchanged_outside_demo.txt`, `import_proof_*.txt`;
- `mutants_prove_unchanged_outside.txt`, `focused_*`;
- `S3-13_removed.diff`;
- `before_after_tally.md`, `before_after_failing_checks.txt` (final after-pass).

### What changed

- **Finding 2.** A new-key job derives the short-file refusal, the domain check and the window from one parsed,
  session-filtered list, as `miner/range_miner_worker.py:904-932` does. Old-style jobs (legacy `coordinator.py`, ZMQ)
  are unchanged.
- **Q8.** `prove_unchanged_outside.py` now refuses:
  - added names whose statement runs code at import;
  - added names already bound in the base (imports included) or rebound in the new file;
  - changed allowed headers that evaluate non-inert code. Narrowed by the owner on 2026-11-21:
    - parameters are compared by name with the base;
    - a non-inert default or annotation is flagged only when it is new or its AST differs from the base's;
    - any change to the decorator list flags every decorator;
    - reordering or re-kinding existing parameters is always caught (`allowed-header-parameter-order-changed`).

    The earlier rule flagged Stage 3b's unchanged `staging_high_water_bytes = 16 * 1024 ** 3` default in
    `window_optimizer.py` `run_bayesian_optimization` (`:756` → `:759`), where only a trailing parameter was added.

  New `--keep-block QUALNAME::HEADER` pins one block and its preamble inside an allowed function.
- **S3-11.** No new code. It landed in 2b; `G-S311-JOBS` re-checks the wire.

### Real-world scenarios for 3a

- **3a-S1. Rigs still on the `b2efdef` `sieve_filter.py` after VM101 is updated.** F2 lives in the rig copy, so the
  rigs keep the two-read form until a redeploy. On this cluster the dataset is not replaced during a run.
- **3a-S2. A dataset shorter than the window.** Still refused before any sieve, now with the miner's "only N entries"
  refusal instead of the out-of-domain code.
- **3a-S3. Stage 3b allows a large shared function**, e.g. `run_bidirectional_test`. `--keep-block` lets the proof
  pin the miner branch and everything before it in that function.
- **3a-S4. A PWC trial whose sampled skip range is not `[0, 16]`.** PWC sieves the sampled range on phases 1-2, as at
  `b2efdef`. RANGE-MINER sieves `[0, 16]` (F-1, open). So the engines still differ there, by the owner's choice,
  until the miner is fixed. Stage 4's comparison must pin `skip [0, 16]` or wait for the miner fix.

### Questions (allowed answers: NO OBJECTION · OBJECTION · SUGGESTION · NEEDS EVIDENCE; LIVE/THEORETICAL on every OBJECTION)

**3a-Q1. RANGE-MINER.** Both shared modules RANGE-MINER's VM101 launch imports
(`window_optimizer_integration_final.py:52-58`) are covered:
- `sieve_filter.py` changed only inside `execute_sieve_job`;
- `persistent_worker_coordinator.py` is byte-identical to `b2efdef`;
- the clean-process import proof is identical under both hostnames.

By your own check: does any line of 3a change a RANGE-MINER run?

**3a-Q2. Finding 2.**
- Is there any remaining path in which a PWC new-key job's checked domain and its window come from different reads?
- Is there one in which the window differs from `load_residue_window` on the same file?

**3a-Q3. Old-style jobs.** Do the legacy `coordinator.py` and ZMQ engines see any change from 3a? Our evidence:
`G-F2-LEGACY-UNCHANGED` gives identical sieve calls and results to `b2efdef` at offsets 0, 1 and 5, the last one
clamped.

**3a-Q4. The proof tool.** With the four new checks, what import-time or call-time change in a shared module would it
still not report?

**3a-Q4b. The narrowed header check.** It accepts a non-inert default that is AST-identical to the same-named
parameter's in the base, on the ground that the base already evaluated it at import.
- Is there a case where such a default evaluates differently in the new file? One example: a name it reads was rebound
  elsewhere in the module. Every module-level statement outside the allowed definitions is still compared.
- A removed parameter is reported but not failed. Should it fail?

**3a-Q5. `--keep-block`.** Is the preamble rule the right granularity for pinning a miner branch inside a function
Stage 3b changes? It covers the block itself, every statement before it in the function, and the signature and
decorators.

**3a-Q6. Test edits.** `tests/test_prove_unchanged_outside.py` changed one PASS case: an added decorator on an
allowed function is now a CATCH case. Does that weaken what the suite guarded?

### Test evidence (3a)

- **Focused, after the S3-13 removal:**
  - `tests/test_pwc_stage3a.py`: 6/6 checks, 3/3 mutants;
  - `tests/test_prove_unchanged_outside.py`: 49/49 after the 2026-11-21 narrowing, with 7 new cases. Caught: 5
    breakages of the Q8 checks and 6 of the narrowed check (H1 restores the old rule);
  - the narrowed tool on a read-only copy of Stage 3b's `window_optimizer.py`, allowing its four changed functions:
    UNCHANGED OUTSIDE ALLOWED, 0 header issues;
  - the narrowing is tool-only, so no full pass was re-run. Nothing but its own test and
    `scripts/pwc_sieve_filter_import_proof.py` references the tool, and that script's suite
    (`tests/test_pwc_stage2b_anchor_phase.py`) was re-run: 35/35;
  - every PWC suite exits as on `b2efdef`.
- **Full before (clean `b2efdef`):** valid, 75 suites.
- **After, final code** (S3-13 removed; `before_after_tally.md`, `before_after_failing_checks.txt`): 73 of 75 suites
  have the same exit code and the same failing-check set as on clean `b2efdef`. The two that differ:
  - `tests/test_s172_phase4_coordinator.py` Gate 22, the git-status check. It names exactly the four uncommitted
    `.py` files of this package, as in Stage 2b, and clears on commit;
  - `tests/test_s172_attempt6_remediation.py` FAIR-1, a wall-clock bound (0.820 s against 0.787 s). Re-run alone on
    the same code it PASSES with the "before" set (76/78). It is a miner-only suite on a file 3a does not change.

  `tests/test_pwc_stage3a.py` is new: rc 0. `~/miner_staging`, `optuna_studies/` and the main tree were unchanged.
  The earlier after-pass, which ran with S3-13, is superseded and kept as `*_with_S3-13.*`.
- **UNAVAILABLE:** GPU runs, rigs, and hit rate (Stage 4).

---

## Stage 3b — the shared launch side (worktree `/home/michael/tfm_worktrees/pwc-backup`, base `b2efdef`)

### Prior record (3b items; the combined record above covers the governance search)

| Document | What it settles | Label |
|---|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:20-23`, `:14` | Ruling 2 (PWC earns credit) and ruling 3 (no seed-start refusal), 2026-11-19 20:57 PST; default rule "PWC does what RANGE-MINER does" | **Fixed facts** (owner) |
| Owner ruling 2026-11-20 15:53 PST, verbatim: "Yes, range miner is the new default" (rulings file `:46`, this worktree) | RANGE-MINER is the Step 1 default; PWC only by its flag | **Fixed fact** (owner). **CHANGES A RULING:** `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:436`, `:503` (default only after Phase 7); `docs/TODO_CONSOLIDATED.md:93`. Not yet in a committed rulings file |
| Owner ruling 2026-11-20 20:24 PST, "Yes": finish the PWC backup first, then re-enable n_parallel > 1 as its own job (rulings file `:49`, this worktree) | The parallel path is out of Stage 3; S3-6 (the self-killing `pkill`) and S3-7 (the partition dial address) go with that job | **Fixed fact** (owner) |
| Owner, 2026-11-20 (this session): edit only the named PWC lines of three existing suites; add one keyword to `agents/watcher_agent.py` | The test edits and the WATCHER edit in this section | **Fixed facts** (owner) |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:34`, `:59-61` (your Q5) | An incomplete PWC trial enters Optuna COMPLETE and scores | ALREADY RECORDED. Addressed here |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:67` (your Q8) | Allowing all of `run_bidirectional_test` would hide a miner-branch change; keep a separate comparison of that subtree and its preamble | ALREADY RECORDED. Done here (statement-level, with mutants) |
| `miner/step1_ingress.py:192-200`; `window_optimizer_integration_final.py:1350` | RANGE-MINER refuses a trial with no committed assembly before ingress | The behaviour PWC now copies |
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R1.md:14`, `:94`; `agents/watcher_agent.py:511-519` | "`default_params` untouched" concerns the seven Run-4 warm-start keys | Not in conflict: engine choice is not one of them. Please confirm |
| `docs/CHAPTER_12_WATCHER_AGENT.md:1220`; `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:20`; `docs/BACKLOG.md:298-300` | n_parallel > 1 SWITCHED-OFF-BY-RULING on the miner path | ALREADY RECORDED |

### Files under review (absolute paths; `git -C /home/michael/tfm_worktrees/pwc-backup diff b2efdef`)

| File | Change |
|---|---|
| `/home/michael/tfm_worktrees/pwc-backup/docs/drafts/pwc_backup_stage3b_v1/README.md` | the package README; read it first |
| `/home/michael/tfm_worktrees/pwc-backup/window_optimizer_integration_final.py` | the `if _use_pw:` block of `run_bidirectional_test`; seven new PWC-only helpers (`:1406-1484`) |
| `/home/michael/tfm_worktrees/pwc-backup/window_optimizer.py` | the seed-start guard narrowed to ZMQ; a dial-address parameter, attribute (PWC runs only) and CLI flag; `require_fleet` for PWC at the dataset gate |
| `/home/michael/tfm_worktrees/pwc-backup/agent_manifests/window_optimizer.json` | the default flip (15:53 ruling); one null `pwc_coordinator_address` key |
| `/home/michael/tfm_worktrees/pwc-backup/agents/watcher_agent.py` | one keyword: `require_fleet=bool(final_params.get("use_persistent_workers"))` |
| `/home/michael/tfm_worktrees/pwc-backup/tests/test_s172_miner_seed_start.py`, `.../tests/test_step1_search_record_stage1.py`, `.../tests/test_s172_phase1_scaffolding.py` | named PWC lines only |
| `/home/michael/tfm_worktrees/pwc-backup/tests/test_pwc_backup_stage3.py` | new |
| `/home/michael/tfm_worktrees/pwc-backup/test_persistent_worker_harness.py` | **T20 only** (owner yes, 2026-11-20): its mock PWC result now carries a complete `pwc_trial_completion_v1` record; AST proof that nothing else changed: `t20_only_proof.txt` |

Evidence is in `docs/drafts/pwc_backup_stage3b_v1/`:
- `pwc_backup_stage3b.diff`, which applies to `git archive b2efdef` and reproduces all 10 files byte for byte;
- `miner_subtree_statement_comparison.txt`;
- `watcher_identical_except_one_call.txt`;
- `old_suites_pwc_lines_only_proof.txt`;
- `focused_*.txt`.

### What changed

- **Ruling 3.** The PWC seed-start refusal is removed in both places:
  - `assert_step1_route_seed_start_supported` refuses ZMQ only;
  - the integration's PWC call to it is gone, and `run_trial_persistent` gets `seed_start`.
- **The PWC call** now also passes `pwc_coordinator_address` (a new `--pwc-coordinator-address` flag with no default;
  RANGE-MINER takes its coordinator host the same way, `scripts/launch_fleet_manual.sh:147`, `:239-241`) and the
  coordinator's ready-gate count. That count was never passed, so it was 1.
- **Your Q5.** Before `_build_test_result_from_pw`, the PWC block raises `PwcTrialIncompleteError` unless:
  - the completion record has schema `pwc_trial_completion_v1`;
  - the trial is complete;
  - every required pass is `complete` and ran on TCP.

  It raises `PwcThresholdProvenanceError` unless every required pass has a verified threshold record. Nothing reaches
  the accumulator, and Optuna sees an exception, not a COMPLETE score, as with RANGE-MINER's `MinerIngressError`.
- **Ruling 2.** The credit evidence is `_step1_pwc_sweep_evidence`: the swept intervals come from PWC's own record,
  and only when the checks above all hold. `credit_decision` is unchanged.
- **15:53 ruling.** The manifest defaults to `use_range_miner: true` and `use_persistent_workers: false`. The new
  `pwc_coordinator_address: null` is skipped by WATCHER (`if value is None`), so it never reaches a command line.
- **PWC fails closed on a missing provisioning manifest, as RANGE-MINER does:**
  - on the CLI: `run_start_dataset_gate(require_fleet=…)`;
  - in WATCHER: `resolve_absent_fleet_status(require_fleet=…)`, the one-keyword edit.
- **Not done, by the 20:24 ruling:** S3-6 and S3-7, both in the n_parallel > 1 path.

### Real-world scenarios for 3b

- **3b-S1. A WATCHER Step 1 with no engine in its params.**
  - Before: it went to PWC, which refused at startup (no dial address; sizing outside the stable envelope).
  - Now: it goes to RANGE-MINER, with RANGE-MINER's preconditions: the fleet, a provisioning manifest, and admission
    by the manifest's `worker_pool_size`.
- **3b-S2. Run 4 as launched today** (`gate12_launch.sh:465-473`, both engine flags explicit). The merged params,
  argv, coordinator attributes and `run_trial_miner` arguments are identical to `b2efdef` (G-MANIFEST,
  G-MINER-SETTINGS, G-MINER-JOBS).
- **3b-S3. A PWC run chosen by the flag.**
  - It must pass `use_persistent_workers: true` **and** `use_range_miner: false`, or the engine mutex refuses
    (`window_optimizer.py:1770-1782`).
  - It also needs `--pwc-coordinator-address` and chunk sizing inside the stable envelope; without them PWC refuses
    at startup.
  - With them, a seed start of 2³¹ is dispatched as such.
- **3b-S4. One-sided update.**
  - VM101 is updated while the rigs keep older PWC workers: a worker that does not echo what Stage 2/2b check makes
    its pass incomplete, so the trial is refused (not scored) and earns no credit.
  - Agent 2's coordinator (3a) and this integration (3b) are meant to land together. An older coordinator without
    `pwc_completion` is refused by the same check.
- **3b-S5. A PWC trial that loses a chunk** (timeout, or a worker of a different version). It raises before the
  accumulator, so Optuna marks the trial failed, as it would for a RANGE-MINER trial without a committed assembly.
  Whether that stops the whole study is the same as for the miner path.
- **3b-S6. A missing provisioning manifest on a PWC run.** It refuses before dispatch on both launch routes, as the
  miner does. ZMQ and legacy are unchanged.

### Questions (allowed answers: NO OBJECTION · OBJECTION · SUGGESTION · NEEDS EVIDENCE; LIVE/THEORETICAL on every OBJECTION)

**3b-Q1. RANGE-MINER, byte for byte.** Our evidence:
- statement-level AST of `run_bidirectional_test`: everything outside `if _use_pw:` is identical, including
  `if _use_miner:` and the preamble;
- the whole-file definition proof;
- the inside-function proof for the four edited `window_optimizer.py` functions;
- WATCHER identical except one keyword;
- run-time equality of the coordinator attributes and the `run_trial_miner` arguments for a Run-4-shaped launch.

By your own check: does any line of 3b change what a RANGE-MINER run does, through the CLI or through WATCHER?

**3b-Q2. The new default (15:53 ruling, a fixed fact).** Name any realistic WATCHER, CLI or agent launch on this
cluster that relied on the old PWC default and now does something unsafe on RANGE-MINER, rather than refusing.
Example: admission by the manifest default `worker_pool_size` 8. Also confirm that the Route-A "`default_params`
untouched" rule does not cover engine selection.

**3b-Q3. Choosing PWC.** Is "PWC by flag" safe as built? Name any route where PWC is selected without the dial
address or without a valid sizing, and does something other than refuse at startup.

**3b-Q4. Your Q5.** Name any path by which a PWC trial with incomplete coverage, a non-TCP pass or an unverified
threshold still reaches the accumulator or an Optuna score. Is raising (the RANGE-MINER behaviour) the right outcome
for the study?

**3b-Q5. Credit.** Does `_step1_pwc_sweep_evidence`, with the unchanged `credit_decision`, credit exactly the
complete, all-TCP, threshold-verified PWC trials, including a zero-survivor complete trial and constant-only trials?
Name a case where it credits more, or less.

**3b-Q6. One-sided deployment.** With VM101 updated and the rigs on an older PWC worker or `sieve_filter.py`, or the
reverse, name any sequence that yields a scored or credited partial result.

**3b-Q7. The ready gate.** The pipeline now passes the coordinator's `pwc_min_workers`, 24 on a full run, where PWC
used 1. Is there a real run on this cluster that this breaks, beyond refusing a short fleet before dispatch?

**3b-Q8. Test edits.** In the three existing suites only the named PWC checks changed, and in `test_persistent_worker_harness.py` only T20. Our evidence:
`old_suites_pwc_lines_only_proof.txt`, a whole-file comparison plus a list-entry diff. M9 is retired as an invalid
experiment, because `run_with_config` has no ZMQ flag and its guard can no longer fire. By your own comparison:
- did any RANGE-MINER gate, mutant or assertion change?
- is retiring M9 right?

### Test evidence (3b)

- **Focused:**
  - `tests/test_pwc_backup_stage3.py`: 14/14 gates, 18/18 mutants;
  - `tests/test_compare_engine_outputs.py`: 13/13;
  - the PWC suites and the D3.25 suite pass;
  - 7 shared or miner suites keep exactly their baseline failing checks: no pointer manifest in a worktree; no GPU;
    Gate 22 naming only uncommitted new files.
- **Full before/after pass (2026-11-20, `before_after_tally.md`, `before_after_failing_checks.txt`):** 77 suites, before on a clean `b2efdef`, after with the parked work restored (42/42 hashes, status and diff identical). 73 of 75 common suites have identical failing-check sets. Four differ: (1) `test_persistent_worker_harness.py` T20 failed: its mock PWC result had no completion record and expected to be scored; under Q5 it is refused by design (`window_optimizer_integration_final.py:2072`). The owner approved editing T20 only; after the edit the suite passes 25/25 (`t20_rerun_test_persistent_worker_harness.txt`); (2) `test_gate12_cleantree_admission.py` W-NO-WEAKENING and (3) `test_s172_phase4_coordinator.py` Gate 22 compare the working tree with HEAD and red on any uncommitted edit (`tests/test_gate12_cleantree_admission.py:941-951`, `tests/test_s172_phase4_coordinator.py:1626`); by reading, both clear once the files are committed (not executed); (4) `test_s172_miner_seed_start.py` G-R5-ROUTES fails at the same point under its new label. The new suites run green: `test_pwc_backup_stage3.py` 14/14 + 18/18, `test_compare_engine_outputs.py` 13/13.
- **UNAVAILABLE:** GPU runs, rigs, a real WATCHER run on the new default, and hit rate (Stage 4).
