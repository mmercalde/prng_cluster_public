# PWC full backup — Stage 3b (shared launch side), v1, uncommitted

Worktree `/home/michael/tfm_worktrees/pwc-backup`, HEAD = main `b2efdef` (moved there by checkout; nothing committed or
pushed). Two agents, strict file split:
- **this package (agent 1):** `window_optimizer.py`, `window_optimizer_integration_final.py`,
  `agent_manifests/window_optimizer.json`, one keyword in `agents/watcher_agent.py` (owner yes, 2026-11-20), the named
  PWC lines of three existing suites (owner yes), and new test files;
- **agent 2, Stage 3a, `/home/michael/tfm_worktrees/pwc-offset`:** `persistent_worker_coordinator.py`, `persistent/`,
  `sieve_filter.py`, `scripts/prove_unchanged_outside.py`. Not touched here, with one exception: on 2026-11-21 the
  lead authorised agent 1, which owns the combined tree, to edit `persistent_worker_coordinator.py` for Beta R1 3b-Q3
  only. Agent 2 was idle at the time. See "Beta X-PWC-STAGE3 R1: the two LIVE fixes".

CPU only. No GPU, no rig, no live ledger. Full before/after pass run on 2026-11-20 (see Tests).
`tests/test_s172_staging_partb.py` was never run.

**Yardstick.** Stage 3b is judged by whether a PWC Step 1 run reaches Steps 2–6 the way a RANGE-MINER run does, and in
the end by the hit rate of pools built from it against `k/1000`. No PWC pool has been measured. Nothing here judges
either engine by whether a seed is found.

## Prior record

**Searches:** the Stage 3 plan's prior record (`docs/drafts/pwc_backup_stage3_v1/PLAN.md` §Prior record), plus a read of
these records for the items here:

| Document | What it settles | Label |
|---|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:20-23` | Ruling 2 (PWC earns credit) and ruling 3 (no seed-start refusal); default rule `:14` | Fixed facts |
| Same file `:46` (this worktree, uncommitted) | 2026-11-20 15:53 PST, verbatim: "Yes, range miner is the new default" | Fixed fact. **CHANGES A RULING:** spec `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:436`, `:503` (default only after Phase 7) |
| Same file `:49` (this worktree, uncommitted) | 2026-11-20 20:24 PST: finish the PWC backup first, then re-enable n_parallel > 1 as its own job | Fixed fact. S3-6 and S3-7 deferred to that job |
| `docs/exchange/from_beta/X-PWC-STAGE2B_R1_BETA_REVIEW.md:34`, `:59-61` (Q5) | An incomplete PWC trial still enters Optuna COMPLETE and scores | ALREADY RECORDED. Fixed here |
| `miner/step1_ingress.py:192-200` (`require_assembly`), `window_optimizer_integration_final.py:1350` | RANGE-MINER raises on a trial with no committed assembly, before ingress | The behaviour copied |
| `docs/TODO_CONSOLIDATED.md:93`; `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:440-442` (D-01); `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1708-1712` (S-4) | The default-route decision was owed after Phase 7 | Superseded by the 15:53 ruling |
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R1.md:14`, `:94`; `agents/watcher_agent.py:511-519` | `default_params` untouched: this concerns the seven Run-4 warm-start keys | Not in conflict (engine choice is not one of the seven) |
| `docs/CHAPTER_12_WATCHER_AGENT.md:1220`; `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md:20`; `docs/BACKLOG.md:298-300` | n_parallel > 1 is SWITCHED-OFF-BY-RULING on the miner path | ALREADY RECORDED |

**Not searched:** ser8, the private repository, claude.ai project files, the rigs (off), Proxmox.

## What changed, with `file:line` (working tree)

### `window_optimizer_integration_final.py`

- **New PWC helpers** (`:1406-1484`, all PWC-only, called only from the PWC block):
  - `PwcTrialIncompleteError` (`:1409`) and `PwcThresholdProvenanceError` (`:1416`);
  - `_pwc_require_complete_trial` (`:1429`): the completion record must have schema `pwc_trial_completion_v1`, the
    trial must be complete, and every required pass must be `complete` and must have run on TCP;
  - `_pwc_require_threshold_provenance` (`:1456`): every required pass has a threshold record with `agree` and
    `verified`;
  - `_step1_pwc_sweep_evidence` (`:1465`): the swept intervals come from PWC's own `swept_intervals`, or `None` (no
    credit). `credit_decision` is not edited.
- **`run_bidirectional_test`, the `if _use_pw:` block only** (`:2035-2080`):
  - the seed-start guard call is removed (ruling 3);
  - `run_trial_persistent` now also gets `seed_start` (`:2061`) and `pwc_coordinator_address` (`:2063`). Nothing else
    in the call changes. `min_workers` is **not** passed, as at `b2efdef`, so PWC's ready minimum stays at
    `run_trial_persistent`'s default of 1 (owner ruling 2026-08-04; Beta R1 3b-Q7, fixed 2026-11-21). An earlier
    version of this package passed the coordinator's `pwc_min_workers` (24).
  - **Beta Q5:** `_pwc_require_complete_trial` (`:2069`) and `_pwc_require_threshold_provenance` (`:2072`) raise
    **before** `_build_test_result_from_pw`. An incomplete trial never reaches the accumulator, and Optuna gets an
    exception, not a COMPLETE score, as with RANGE-MINER. The old `pruned` early return is gone: a pruned result has
    no complete passes, so it is refused by the same check;
  - the credit evidence is `_step1_pwc_sweep_evidence` (`:2078`), replacing the refusal stamp.
- **Everything else is identical to `b2efdef`**, including the whole `if _use_miner:` block and the ZMQ block
  (`miner_subtree_statement_comparison.txt`).

### `window_optimizer.py`

- `assert_step1_route_seed_start_supported` (`:678-715`): refuses a nonzero start for **ZMQ only** (`:708`). The PWC
  arguments are still accepted, so no caller changed.
- `run_bayesian_optimization` (`:762`) and `run_with_config` (`:1209`): a new last parameter,
  `pwc_coordinator_address=None`. It is set on the coordinator **only when `use_persistent_workers`** (`:873-876`,
  `:1328-1329`), so a RANGE-MINER or legacy coordinator carries exactly the attributes it carried before.
- `main`:
  - new `--pwc-coordinator-address` flag, with no default (`:1580-1584`);
  - passed by the two calls that can select PWC (`:1953`, `:2076`);
  - `run_start_dataset_gate(..., require_fleet=bool(use_persistent_workers))` (`:1915-1917`).

### `agent_manifests/window_optimizer.json` (owner ruling 15:53)

- `default_params`: `use_persistent_workers` true → **false** (`:263`); `use_range_miner` false → **true** (`:272`);
  new `pwc_coordinator_address: null` (`:270`).
- `actions[0].args_map`: new `"pwc-coordinator-address"` (`:44`).
- `parameter_bounds.use_range_miner`: `"default": true` (`:216`), and the stale "Phase 1 scaffolding … do not enable
  until Phase 7" description is rewritten to cite the ruling.
- WATCHER skips empty values, so the null key reaches no command line (`agents/watcher_agent.py` run_step:
  `if value is None or value == '': continue`).

### `agents/watcher_agent.py` (one keyword, owner yes)

`:2125-2126`: `resolve_absent_fleet_status(..., require_fleet=bool(final_params.get("use_persistent_workers")))`. The
whole module minus that keyword is AST-identical to `b2efdef` (`watcher_identical_except_one_call.txt`).

### Existing suites, PWC lines only (owner yes; `old_suites_pwc_lines_only_proof.txt`)

- `tests/test_s172_phase1_scaffolding.py` `gate5_watcher_manifest`: the default assertion is flipped (15:53 ruling).
- `tests/test_s172_miner_seed_start.py`:
  - `g_r5_routes`: PWC at 1000 is now dispatched with `seed_start=1000`, and PWC at 0 equals the base plus the three
    stage-3 keywords;
  - mutant M6 now drops `seed_start`; M8's anchor is the new ZMQ predicate;
  - **M9 is retired:** `run_with_config` has no ZMQ flag, so its guard can no longer fire, which would make it an
    invalid experiment under VIR-2.
- `tests/test_step1_search_record_stage1.py`:
  - `g1_pwc_zmq_refused`: ZMQ is still refused; PWC is refused without its completion evidence;
  - `run4_shape` and `g_miner_start`: PWC accepted at 2³¹, ZMQ refused;
  - the M-ROUTE-START anchor follows the new predicate.

### New files

- `tests/test_pwc_backup_stage3.py`: 14 gates, 18 mutants. Since R1, G-PWC-CALL asserts that `min_workers` is not
  passed.
- `tests/test_pwc_stage3_r1_fixes.py`: 4 gates and 5 mutants for Beta R1's two LIVE objections.
- `test_persistent_worker_harness.py`: T20 only (owner yes), so that its mock carries a complete completion record.
- `scripts/compare_engine_outputs.py` and `tests/test_compare_engine_outputs.py`: the Stage 4 comparison tool, written
  earlier today.

## Proof that RANGE-MINER is unchanged

| Proof | Result | Evidence |
|---|---|---|
| Statement-level AST of `run_bidirectional_test` | All statements outside `if _use_pw:` are identical to `b2efdef`, including `if _use_miner:` (401 AST nodes) and the preamble; the signature is identical. Mutants in the preamble, the miner block and `_miner_residues_for_config` are caught | `miner_subtree_statement_comparison.txt`; G-MINER-SUBTREE, M-MINER-1/2a/2b |
| Whole-file definitions | Integration: 59 definitions identical outside `run_bidirectional_test` and the 7 added PWC names. `window_optimizer.py`: 53 identical outside the four edited functions. Every function the miner reaches is identical | G-MINER-DEFS, M-MINER-3 |
| Inside the edited `window_optimizer.py` functions | The new AST minus the declared PWC insertions equals `b2efdef` for `run_bayesian_optimization`, `run_with_config` and `main` | G-WO-FUNCS, M-MINER-4/5 |
| **Run-time, coordinator settings** | `run_bayesian_optimization(use_range_miner=True, Run-4 shape)` gives the same 19 coordinator attributes and the same `optimize_window` arguments as `b2efdef`. Control: a PWC run gains exactly `pwc_coordinator_address` | G-MINER-SETTINGS, M-MINER-6 |
| **Run-time, jobs** | With `use_range_miner=True`, `run_trial_miner` gets the same 40 arguments as `b2efdef` | G-MINER-JOBS, M-MINER-7 |
| Manifest | Only the ruled flip, the `use_range_miner` spec and one null PWC key. Run-4 params merge to identical values apart from that null key, which WATCHER skips | G-MANIFEST |
| WATCHER | One keyword; nothing else moved | G-WATCHER-FLEET, M-WATCHER-1/2 |
| Untouched files | `miner/`, the ten pinned files (`scripts/gate12_parity_gate.py:174-185`), `gate12_launch.sh`, `scripts/launch_fleet_manual.sh`, and agent 2's files: `git diff b2efdef` is empty | — |

## Tests (sequential, CPU only)

**Full before/after pass (2026-11-20, `before_after_tally.md`, `before_after_failing_checks.txt`):** 77 suites, before on a clean `b2efdef`, after with the parked work restored (42/42 hashes, status and diff identical). 73 of 75 common suites have identical failing-check sets. Four differ: (1) `test_persistent_worker_harness.py` T20 failed: its mock PWC result had no completion record and expected to be scored; under Q5 it is refused by design (`window_optimizer_integration_final.py:2072`). The owner approved editing T20 only (AST proof `t20_only_proof.txt`); after the edit the suite passes 25/25; (2) `test_gate12_cleantree_admission.py` W-NO-WEAKENING and (3) `test_s172_phase4_coordinator.py` Gate 22 compare the working tree with HEAD and red on any uncommitted edit (`tests/test_gate12_cleantree_admission.py:941-951`, `tests/test_s172_phase4_coordinator.py:1626`); by reading, both clear once the files are committed (not executed); (4) `test_s172_miner_seed_start.py` G-R5-ROUTES fails at the same point under its new label. The new suites run green: `test_pwc_backup_stage3.py` 14/14 + 18/18, `test_compare_engine_outputs.py` 13/13.

### Focused runs (before the full pass)

| Suite | Result | Compared with |
|---|---|---|
| `tests/test_pwc_backup_stage3.py` (new) | **14/14 gates, 18/18 mutants** | — |
| `tests/test_compare_engine_outputs.py` | 13/13 | — |
| `tests/test_s172_phase1_scaffolding.py` | rc 0 | edited |
| `tests/test_s172_phase5_d3_25_candidate_ingress.py` | 13/13 | — |
| `tests/test_pwc_backup_stage2.py`, `tests/test_pwc_stage2b_anchor_phase.py`, `tests/test_pwc_backup_step1.py`, `tests/test_s172_staging_backpressure.py`, `tests/test_step1_credit_wal.py` | rc 0 | — |
| `tests/test_s172_miner_seed_start.py`, `tests/test_step1_search_record_stage1.py`, `tests/test_s172_phase5_d6_production_adapter.py`, `tests/test_s172_threshold_propagation.py`, `tests/test_s172_run4_routing_patch.py`, `tests/test_s172_phase6_p05_dataset_authority.py`, `tests/test_s172_phase4_coordinator.py` | rc 1, **the same failing checks as the baseline** (the `0caeaf9` Stage 2 final pass; `b2efdef` did not change these files) | The reds are environmental: no dataset pointer manifest in a worktree; G-KERNEL needs a GPU; Gate 22 names only uncommitted new tests and `scripts/compare_engine_outputs.py` |

**Note:** because of the missing pointer manifest, `test_s172_miner_seed_start` G-R5-ROUTES cannot reach its new PWC
assertions in this worktree. It fails at the dataset read, as it did at baseline. `tests/test_pwc_backup_stage3.py`
covers the same path with the dataset reader replaced. The outputs are in the `focused_*.txt` files here.

## Beta X-PWC-STAGE3 R1: the two LIVE fixes (2026-11-21)

Review: `docs/exchange/from_beta/X-PWC-STAGE3_R1_BETA_REVIEW.md`. Closing note (draft, not relayed):
`docs/exchange/to_beta/X-PWC-STAGE3_R2_ALPHA_CLOSING_NOTE_DRAFT.md`. The owner approved landing at 06:28 PST, verbatim:
"yes, since there are only two beta rejections and you have already addessed them land stage 3". **No second Beta
round.** Evidence: `r1_fixes/`.

### 3b-Q7 / Finding 1: the PWC ready minimum (review `:64`, `:103`, `:71`)

The governing records, quoted:
- `docs/CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md:86-100`: "## 2. `min_workers` — DO NOT RESTORE. … the
  guard's original purpose was **to ensure the whole cluster was being utilised** … **It was a UTILISATION check, not a
  correctness gate.** **RANGE-MINER does not have that shape.** … **Therefore: not a defect. Not restored. Not a
  Phase-7 blocker.**"
- `docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195` (CORRECTION 2026-08-04): "`run_trial_persistent` falls back to its own
  default `min_workers: int = 1` … The gate below therefore passes at **one** ready worker … **⚠ It is NOT being
  restored — owner ruling, 2026-08-04.**"

What happened: the plan's Q-9 (`docs/drafts/pwc_backup_stage3_v1/PLAN.md:258`, now corrected) labelled the missing
hop NEW and passed the coordinator's `pwc_min_workers` (CLI/manifest 24). Alpha applied it without asking, as "what
RANGE-MINER does". That was wrong on reading: RANGE-MINER admits `min(requested pool size, selected worker
identities)` (`execution_set.py:653-676`). No ruling asked for 24.

**Fix:** the `min_workers=` keyword is removed from the PWC call (`window_optimizer_integration_final.py:2043-2064`).
The call's keywords are b2efdef's plus `seed_start` and `pwc_coordinator_address`, so the effective minimum is
`run_trial_persistent`'s default, 1 (`persistent_worker_coordinator.py:2604`). `window_optimizer.py` and the manifest
needed no change: their `min_workers` lines are b2efdef's.

**Tests:**
- `G-Q7-CALL`: the call's keyword set against b2efdef's, parsed from `git show`;
- `G-Q7-READY19`: Beta's scenario. 3 rigs × 6 workers + 1 VM101 = 19 of 19 ready passes the startup ready gate
  (`_tcp_wait_ready`) and the per-pass dispatch check, and the pass completes. The control at 24 refuses;
- `G-PWC-CALL` in `tests/test_pwc_backup_stage3.py` now asserts that `min_workers` is not passed;
- mutants M-Q7-1 (keyword re-added) and M-Q7-2 (default raised to 24) are killed.

### 3b-Q3: nonpositive chunk cap (review `:60`, `:93`)

**Fix:** `persistent_worker_coordinator.py` `_startup_refusals` (`:1062-1073`, 12 added lines, PWC-only). If
`seed_cap_amd` or `seed_cap_nvidia` is ≤ 0, it raises the existing `PwcSettingRefused` ("… must be positive, got X.
No override exists."). This runs before the S156 reap, any SSH, any spawn and the TCP bind, and on every PWC entry
point, because they all construct the coordinator and call `startup()`.
- RANGE-MINER refuses the same input: `miner/range_miner_coordinator.py:769-770` ("macro_size must be positive") and
  `:789-790` ("effective_cap must be positive"); `miner/range_miner_worker.py:681-682` ("cap must be positive"). Its
  coordinator also quarantines non-positive advertised seed caps (`:3883-3884`).
- The NVIDIA cap is stored but never cuts chunks; the chunk loop uses `seed_cap_amd` only (`:2097`). It is refused
  anyway.

Proof: `r1_fixes/prove_unchanged_outside_coordinator.txt`. Only `PersistentWorkerCoordinator._startup_refusals`
changed (rc 0). The control, which allows `startup` instead, refuses (rc 1).

**Tests:**
- `G-Q3-REFUSE`: AMD 0, AMD −1, NVIDIA 0 and NVIDIA −1 are each refused in the real `startup()`, with zero subprocess
  calls;
- `G-Q3-NORMAL`: cap 25,000 starts (4 launch scripts), and a pass completes 4/4 over [0, 100000);
- mutants M-Q3-1/2/3 are killed: refusal removed, `<= 0` weakened to `< 0`, NVIDIA unchecked. With the refusal removed,
  cap 0 starts, which confirms Beta's report.

### Hard limits after the fixes

- **Subtree proof:** `r1_fixes/prove_unchanged_outside_subtree_woi.txt`. `if _use_miner:` and its preamble are
  identical to b2efdef: UNCHANGED OUTSIDE ALLOWED, rc 0.
- **Untouched:** `miner/`, the ten governed files, `gate12_launch.sh` and `scripts/launch_fleet_manual.sh` have an empty
  `git diff b2efdef`. WATCHER is unchanged beyond the one keyword (G-WATCHER-FLEET).

### Results

- **Focused:** `r1_fixes/focused/`, all rc 0.
  - R1 fixes: 4/4 + 5/5.
  - Stage 3: 14/14 + 18/18.
  - Stage 3a: 6/6 + 3/3.
  - Proof tool: 49/49.
  - Stage 2b: 35/35.
  - Stage 2: 26/26 + 33/33.
  - Step 1: 22/22 + 16/16 (2 HELD, as before).
  - Harness: 25/25.
  - Compare: 13/13.
- **Full after-pass:** `r1_fixes/before_after_tally_r1.md`. 79 suites against the 2026-11-20 before-pass on clean
  b2efdef. 72 of 75 common suites are identical. The 3 that differ are W-NO-WEAKENING and Gate 22 (working-tree
  tripwires that clear on commit) and G-R5-ROUTES (same failure, new label). The 4 new suites are green.

## Open items

1. **n_parallel > 1, a separate job** (owner 20:24). S3-6, the self-killing `pkill` with hardcoded bare-metal IPs in
   `optimize_window`'s parallel path, and S3-7, the dial address not passed to partitions, are left as they are. Today
   PWC with n_parallel > 1 refuses at startup for want of a dial address.
2. **`main`'s random, grid and evolutionary modes** call `run_bayesian_optimization` with **no engine flag at all**
   (`window_optimizer.py:2000`, `:2020`, `:2040`). They always run the legacy route, whatever the manifest says. This
   is pre-existing and not touched.
3. **PWC chunk sizing defaults** (`seed_cap_amd` 2,000,000 in the manifest, the CLI and the integration fallback) are
   outside the stable envelope, so a PWC run on defaults refuses at startup. No default changed (Q-3).
4. **Choosing PWC once RANGE-MINER is the default (Q-17):** the caller passes `use_persistent_workers: true` **and**
   `use_range_miner: false`; otherwise the engine mutex refuses (`window_optimizer.py:1770-1782`). This is not yet
   written into a runbook.
5. **Agent 2's items:** S3-11 (coordinator side of the split) landed in 2b; Stage 3a carries Finding 2 and the proof-tool checks. S3-13 WITHDRAWN by the owner, 2026-11-20 21:32-21:35 PST ("Well range miner must be fixed none the less"; "Please do whatever method is the quickest and least recoding"); **F4: PWC keeps the trial's skip range; the RANGE-MINER fix is scoped separately in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`**.
6. **Combined tree with Stage 3a** (copied unchanged, sha256 20/20; `combined_focused/`): 7 of 8 focused suites
   green. `tests/test_pwc_backup_stage3.py` is 13/14 + 17/18, because 3a's new header check flags the unchanged
   pre-existing default `16 * 1024 ** 3` in `run_bayesian_optimization` (`window_optimizer.py:759`, as `b2efdef:756`).
   Resolved 2026-11-21: agent 2 narrowed the tool (00:32; a default is flagged only if its AST differs from the base, or the parameter is new). After a sha256-checked re-copy: `test_pwc_backup_stage3` 14/14 + 18/18, `test_prove_unchanged_outside` 49/49, `test_pwc_stage3a` 6/6 + 3/3, subtree proof unchanged with both control mutants refused.
   Subtree proof: `prove_unchanged_outside_subtree_woi.txt`. EXEC-PIN-1 scan: `exec_pin_1_scan.txt`.
7. **Rigs:** no redeploy needed for Stage 3b, which is VM101-side only. Stage 4's two-engine comparison
   (`docs/drafts/pwc_backup_stage3_v1/PLAN.md` §F) is owner-run.

## Not verified

- Anything on a rig or a GPU.
- A real WATCHER run on the new default.
- A PWC run end to end with rigs.
- Hit rate.
- That no other launcher passes engine flags in a way this package did not see. Only `gate12_launch.sh` and WATCHER
  were read.
