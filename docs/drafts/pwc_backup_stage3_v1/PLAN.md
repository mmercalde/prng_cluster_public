# PWC full backup — Stage 3 PLAN (preparation only; nothing edited)

**Date:** 2026-11-20 · **Host:** VM101, as `michael` · worktree `/home/michael/tfm_worktrees/pwc-backup` at main
`0caeaf9` (moved by `git reset 0caeaf9`, mixed; every changed file was first checked byte-identical to `0caeaf9`).

**Written:** this file only. No shared launch file, `miner/` file or pinned file was edited. No test was run, no
commit, no push, no rig, no live ledger.

**Read only:**
- the brief `docs/drafts/BRIEF_PWC_BACKUP_STAGE2_STAGE3_20261119.md` (on main since `0caeaf9`), §B Stage 3, §C, §E3, §F;
- the code at `0caeaf9`;
- `/home/michael/tfm_worktrees/pwc-offset/scripts/prove_unchanged_outside.py`, read only. That worktree is
  another agent's uncommitted Stage 2b and was not edited.

**Yardstick.** Stage 3 is judged by whether a PWC Step 1 run reaches Steps 2–6 the way a RANGE-MINER run does
(interval, phases, thresholds, credit), and in the end by the hit rate of pools built from it against `k/1000`. No
PWC-produced pool has been measured. Nothing here judges PWC by whether a seed is found.

---

## Prior record

**Searches run** (on `0caeaf9`; `/bin/grep`, because the shell `grep` skips `*.json`):

```
grep -n "_use_pw|_route_start_guard|run_trial_persistent|_step1_unaccounted_route_evidence|pwc_" window_optimizer_integration_final.py
grep -n "assert_step1_route_seed_start_supported|run_start_dataset_gate|use_persistent_workers|pwc_|add_argument" window_optimizer.py
/bin/grep -n "persistent|pwc|seed_cap|worker_pool|min_workers|use_range_miner" agent_manifests/window_optimizer.json
grep -n "resolve_absent_fleet_status|use_persistent_workers|args_map" agents/watcher_agent.py
grep -n "def run_start_dataset_gate|def resolve_absent_fleet_status" -A14 miner/dataset_authority.py   (read, not edited)
grep -n "route=\"pwc\"|_use_pw|run_trial_persistent|_step1_unaccounted_route_evidence|NP2-KILL" tests/*.py test_*.py
grep -n "COORD_HOST|--host" scripts/launch_fleet_manual.sh ; sed -n 314,317p,465,473p gate12_launch.sh
```

**Governing records:** the Stage 2 package's prior record covers the governance search for PWC
(`docs/drafts/pwc_backup_stage2_v1/README.md` §Prior record, including the five documents added in round 2). Of
those, these govern Stage 3:

| Document | What it settles for Stage 3 |
|---|---|
| `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md` | Rulings 2 (credit) and 3 (no seed-start refusal) are Stage 3's. The default rule: PWC does what RANGE-MINER does |
| Owner ruling 2026-11-20 08:30 PST | Offset = draw window only; generator phase 0; PWC moves to RANGE-MINER's method; PWC refuses offset ≠ 0 until that lands |
| `docs/drafts/BRIEF_PWC_BACKUP_STAGE2_STAGE3_20261119.md` §B Stage 3 (3.1–3.10), §C, §E3 | The planned change list. **Re-anchored and corrected below** |
| `docs/exchange/from_beta/X-PWC-STAGE2_R1_BETA_REVIEW.md` Q4, Q6 | An Optuna study aborts at the first nonzero offset while the refusal stands; record the degraded TCP route in the Stage 3 handoff |
| `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:139-153`; `docs/TB_RULING_FIELD6_IMPLEMENTATION.md:112-120` | After commit, rerun the clean-tree and Gate 22 checks without widening allowlists |
| `docs/TB_RULING_WINDOW_ANCHOR_SEQUENCING.md:12-21` | `window_anchor` / `generator_phase` contract; phase 0 in v1 |
| `docs/TODO_CONSOLIDATED.md:93` | Owner decision owed after Phase 7: the manifest's default route is PWC (`use_persistent_workers: true`) |

**Labels:**
- Stage 3 items 3.1–3.9: ALREADY RECORDED (the brief).
- **NEW:**
  - the NP2-KILL self-kill in the integration file (S3-6 below);
  - the WATCHER launch of Run 4 puts WATCHER on RANGE-MINER's path (S3-10);
  - the proof script cannot isolate blocks inside a function (§C);
  - the integration does not pass `min_workers` to PWC (Q-9). **Corrected 2026-11-21:** this is not NEW. It is
    ALREADY RECORDED and ruled. The owner decided on 2026-08-04 not to restore the full-fleet ready gate
    (`docs/CLAUDE_CODE_INSTRUCTIONS_WARMSTART_RESTORE_PART_B.md:86-100`; `docs/FLEET_STATE_REQUIREMENTS_v1.md:146-195`).
    Beta X-PWC-STAGE3 R1 Finding 1 (`:71`) caught it.
- **CHANGES A RULING:** none here. The owner rulings are applied as given.

**Not searched:** ser8, the private repository, claude.ai project files, the rigs (off), Proxmox.

---

## A. Correction to the brief

The brief (§D, item 2.8) says the window-anchor / generator-phase split needs only `sieve_filter.py`. **That is wrong.**
The TCP worker builds the sieve job from a fixed list of keys (`persistent/pwc_worker_service.py` `_execute_job`,
`:541-561` at `0caeaf9`). New `window_anchor` and `generator_phase` fields are therefore dropped on the rig, so the
split needs a **worker edit as well**.

Status: the owner ruled the method (08:30 PST), and the worker side is being built as **Stage 2b** in
`/home/michael/tfm_worktrees/pwc-offset`:
- uncommitted, at base `ee39404`;
- it touches `persistent/pwc_worker_service.py` and `sieve_filter.py`;
- it adds `scripts/prove_unchanged_outside.py`.

The **coordinator side** belongs to Stage 3, or to 2b's own follow-up (Q-1). It is PWC-only: send `window_anchor` and
`generator_phase = 0` in every job, and lift `validate_pwc_window_offset`. It must land only **after** 2b's worker and
`sieve_filter.py` changes are on main **and** on the rigs. An old rig worker would silently drop the new keys and go
back to the double use.

---

## B. Stage 3 change list (shared launch files)

The last column says whether RANGE-MINER's launch reaches the function. Run 4 is launched by `gate12_launch.sh:465-473`:
WATCHER `--run-pipeline`, Step 1, `use_range_miner: true`, `use_persistent_workers: false`, `n_parallel: 1`. That
runs `window_optimizer.py main()` and then `run_bayesian_optimization` (or `run_with_config`), then
`run_bidirectional_test`, whose `_use_miner` branch returns at `window_optimizer_integration_final.py:1762-1949`
before the PWC branch.

| # | File : function (`0caeaf9` lines) | What changes | Ruling | Reached by RANGE-MINER's launch? |
|---|---|---|---|---|
| S3-1 | `window_optimizer.py:678-712` `assert_step1_route_seed_start_supported` | Refuse a nonzero start for **ZMQ only**: drop PWC from the condition at `:705` and from the message. Its callers (`:819-821` in `run_bayesian_optimization`, `:1257-1259` in `run_with_config`) are **not** edited; for PWC they become no-ops | 3 | **Yes**, called on every route. For the miner it is already a no-op; the change only removes the PWC arm. Behaviour table in §C |
| S3-2 | `window_optimizer_integration_final.py:1954-2007` `run_bidirectional_test`, the `if _use_pw:` block only | (a) Delete the `_route_start_guard(... route="pwc")` call `:1960-1963`. (b) Pass `seed_start=seed_start`, `pwc_coordinator_address=getattr(coordinator, 'pwc_coordinator_address', None)` (no default; PWC refuses if unset) and `min_workers=` (Q-9). (c) Replace the `pruned` early return `:1987-1995` with a refusal: Stage 2 never returns `pruned`, so a `pruned` result means an old PWC build. (d) Before `_build_test_result_from_pw` (`:1997`), check PWC's own threshold records (`pwc_completion.passes[*].accounting.threshold`: `agree` and `verified`), as the miner's wall does at `:1330-1346`. This goes **in the PWC block, not in `_build_test_result_from_pw`**, which ZMQ shares (`:2066`). (e) Replace the refusal stamp `:2002-2006` with S3-3 | 2, 3, default | **Yes, the function.** The miner branch returns before `:1954`, so the `_use_pw` block is never executed for the miner. Proof: block-level (§C) |
| S3-3 | `window_optimizer_integration_final.py`, a new top-level `_step1_pwc_sweep_evidence(pw_result)` beside `:1397` | Returns `{phase: [s, e]}` from `pw_result["pwc_completion"]["swept_intervals"]`, and only when `pwc_trial_complete` is true, the schema is `pwc_trial_completion_v1`, and **every pass ran on TCP** (`accounting.transport == "tcp"`, Stage 2). Otherwise `None` (no credit). `credit_decision` (`utils/step1_search_ledger.py:1044`) is **not** edited | 2 | No (new; called only from the PWC block) |
| S3-4 | `window_optimizer.py main()` argparse (`:1561-1580`) | Add `--pwc-coordinator-address` (no default). Thread it as a new parameter of `run_bayesian_optimization` (`:715`) and `run_with_config` (`:1187`), set `coordinator.pwc_coordinator_address` next to `:865-868` and `:1317-1318`, and pass it at the two call sites (`:1925-1935`, `:2045-2055`) | default (the miner takes its coordinator host explicitly, `scripts/launch_fleet_manual.sh:147`, `:239-241`) | **Yes**, all four functions. The new lines only add a parameter and an attribute; the miner never reads it. Proof: block-level (§C) |
| S3-5 | `window_optimizer.py main()` `:1893-1899` `run_start_dataset_gate(...)` | Add `require_fleet=bool(getattr(args, 'use_persistent_workers', False))`. The function already takes it (`miner/dataset_authority.py:1235-1241`, **not edited**) and ORs it with `miner_backed` (`:1060-1066` and its body) | default (the miner fails closed on a missing provisioning manifest) | **Yes.** For a miner run `miner_backed=True` already raises; `require_fleet` changes nothing there. Behaviour table in §C |
| S3-6 | `window_optimizer_integration_final.py:2897-2914`, inside `add_window_optimizer_to_coordinator.optimize_window`, `n_parallel > 1` block (`:2535`) | **NEW finding, same defect class as Stage 1 B5.** `ssh michael@<ip> 'pkill -9 -f pwc_worker_service …; echo ok'`: the pattern is unbracketed (its own shell matches it), the bare-metal IPs `.120/.154/.162` are hardcoded, the result is not read, and it prints "stale workers killed" whatever happens. Proposed: run only when `use_persistent_workers`, over the configured rig nodes, using `persistent_worker_coordinator.pwc_stale_kill_command(PWC_LOCAL_STALE_KILL_PATTERNS)`, which is probe-checked, and read its result (Q-4) | default | **Yes, whenever `n_parallel > 1`, whatever the engine.** Run 4 uses `n_parallel: 1` and does not reach it. A miner run with `n_parallel > 1` does today: it sshes to unreachable bare-metal IPs to kill PWC workers. Gating it on PWC removes that from miner runs, a behaviour change for the miner (Q-4) |
| S3-7 | `window_optimizer_integration_final.py` partition hops: `_partition_worker` signature `:2563-2576` and wiring `:2607-2612`; spawn args `:2941-2962` | Carry `pwc_coordinator_address` to each partition, next to `pwc_host` and `pwc_port`. Partition P*i* binds port `5600+i` (`:2962`), and the dial port must match | default | **Yes, whenever `n_parallel > 1`** (positional `args=` tuple). The new argument goes **last** so every existing position is unchanged |
| S3-8 | `agent_manifests/window_optimizer.json`: args map `:37-52`, param specs `:171-200`, defaults `:262-270` | Add `"pwc-coordinator-address": "pwc_coordinator_address"` to the map and a param spec with **no default value** (null). No existing key or value changes (Q-2, Q-3) | default | **Yes**, WATCHER reads it for every Step-1 run. Proof: a JSON key-path diff showing only an added `pwc_*` key |
| S3-9 | `agents/watcher_agent.py:2119-2124` (inside the WATCHER step function) `resolve_absent_fleet_status(...)` | Add `require_fleet=bool(final_params.get("use_persistent_workers"))` | default | **Yes**: Run 4 runs through WATCHER (`gate12_launch.sh:466`). For the miner, `miner_backed=True` already raises. Proof: block-level plus a behaviour test |
| S3-10 | `agents/watcher_agent.py:506-521` (the literal internal-key list) and `:2377-2380` (args-map reverse lookup) | **Verify before editing** whether a new manifest arg must also be listed in the literal block at `:506` ("`args_map` declares ELEVEN entries"). If it must, that is a WATCHER edit too [UNVERIFIED; not read further in this preparation] | default | Yes |
| S3-12 | `agent_manifests/window_optimizer.json:262`, `:270`, `:213-218` | **RANGE-MINER becomes the Step 1 default** (owner ruling 2026-11-20 15:53); see the S3-12 section below the table | owner | Run 4 names the engine, so: no |
| S3-13 | — | **WITHDRAWN** (S3-13 WITHDRAWN by the owner, 2026-11-20 21:32-21:35 PST ("Well range miner must be fixed none the less"; "Please do whatever method is the quickest and least recoding")). **F4: PWC keeps the trial's skip range; the RANGE-MINER fix is scoped separately in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`** | — | — |
| S3-11 | `persistent_worker_coordinator.py` (PWC-only) | Coordinator side of the split, after Stage 2b lands: `window_anchor` and `generator_phase = 0` in every job; lift `validate_pwc_window_offset` (`:326-358`, `:1952`, `:2543`). Any small PWC-side wiring for S3-2's new arguments | 08:30 ruling | No |

### S3-12 — RANGE-MINER becomes the Step 1 default route (owner ruling 2026-11-20 15:53 PST)

**Ruling, verbatim:** "Yes, range miner is the new default". The owner wants RANGE-MINER as the default, and PWC
chosen only by its flag. Recorded at `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:46` (this worktree, uncommitted).
**Not edited yet:** the manifest and the launch files change only through the Stage 3 review.

**What sets the default today** (`agent_manifests/window_optimizer.json`, `0caeaf9`):

| Line | Today | Proposed |
|---|---|---|
| `:262` (`default_params`) | `"use_persistent_workers": true` | `false` |
| `:270` (`default_params`) | `"use_range_miner": false` | `true` |
| `:267` (`default_params`) | `"use_zmq_sqlite": false` | unchanged |
| `:171-176` (param spec `use_persistent_workers`) | `"default": false`. Its description says "Requires soak test validation before enabling in production" | unchanged (it already says false) |
| `:213-218` (param spec `use_range_miner`) | `"default": false`. The description says "Phase 1 is scaffolding-only: setting true will raise NotImplementedError until Phases 2-5 land. Do not enable in production until Phase 7 acceptance completes." | `"default": true`; the description is stale (the miner is built; Gate 12 passed) and is rewritten to cite this ruling |

The two `default_params` lines and the two param-spec `"default"` fields disagree today for `use_persistent_workers`
(`:262` true, `:173` false). After the change they agree.

**The earlier decision this replaces (prior record):**

| Record | What it decided |
|---|---|
| `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:436` | `"use_range_miner": true, ── once TB approves` (the spec's planned default) |
| `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:503` | Phase 7 = WATCHER soak + production; the condition for the flip |
| `agent_manifests/window_optimizer.json:216` | "Do not enable in production until Phase 7 acceptance completes" |
| `docs/TODO_CONSOLIDATED.md:93` | "Decision owed after Phase-7 acceptance" (REVIEW, after soak) |
| `docs/CHAPTER_CORRECTIONS_REPORT_20261111.md:440-442` (D-01) | Held by spec `:436`, `:503`, Phase 7 soak not run; "it must flip together with `use_persistent_workers: false`" |
| `docs/CHAPTER_9_GPU_CLUSTER_INFRASTRUCTURE.md:1708-1712` (S-4) | RANGE-MINER is not WATCHER's default; switched off by the phase plan |
| `tests/test_s172_phase1_scaffolding.py:177-181` | **A test pins it:** "Phase 1: use_range_miner MUST default to False in the manifest (spec §7 — only set true after TB approves miner production readiness at Phase 7)" |

Label: **CHANGES A RULING.** The owner's ruling replaces the "after TB approval at Phase 7" condition; the Phase 7
soak has not run. The owner is the final authority, so Beta receives it as a fixed fact at the Stage 3 review.

Not in conflict: the Route-A rule that `default_params` is untouched (`agents/watcher_agent.py:2039-2041`;
`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R1.md:14`, `:94`) concerns the seven Run-4 warm-start keys
(`agents/watcher_agent.py:511-519`), not engine selection. This should be stated in the Beta request so Beta can check
it.

**Every place that reads that default:**

| Reader | What it does with it | On the Step 1 path? |
|---|---|---|
| `agents/watcher_agent.py:2011-2032` (`run_step`) | `final_params = {**default_params}`, then the caller's declared keys override. Builds Step 1's CLI through `args_map` (`:2377-2380`) | **Yes**, the live WATCHER path (Run 4 runs through it, `gate12_launch.sh:466`) |
| `agents/watcher_agent.py:794-809` (`_step1_cli_final_params`) | The same merge, for the Step 1 search-record open/status commands. `run_step` compares the two before dispatch | Yes (search records) |
| `agents/watcher_agent.py:1749-1770` (`_step1_declared_params`) and `:1810-1817` | Chooses the execution-set backend (`miner` / `pwc` / `zmq` / `legacy`) from the merged params, and the admission count (`worker_pool_size` for the miner, `min_workers` for PWC) | Yes |
| `agents/watcher_agent.py:2119-2124` | `resolve_absent_fleet_status(miner_backed=bool(final_params.get("use_range_miner")))` | Yes |
| `window_optimizer.py:1561`, `:1620` (argparse) | The CLI defaults are `False` for both flags. The CLI does **not** read the manifest; a flag arrives only when WATCHER passes it | Yes, but unchanged |
| `window_optimizer.py:1754-1765` (mutex) | Refuses more than one engine flag | Yes. See the consequence for PWC below |
| `agents/agent_core.py:245`, `:361`; `agents/step_runner/command_builder.py:113-125`; `agents/step_runner/manifest_loader.py:131` | Generic manifest-default merges | **UNVERIFIED** whether any live Step 1 path uses them; `agents/watcher_agent.py` imports neither (`git grep`) |
| `tests/test_s172_phase1_scaffolding.py:177-181` | Asserts `use_range_miner` is false | It goes **red** on the flip. A test edit is needed (a miner suite, so owner approval, with a line-limited diff as for the Stage 2 shared suites) |
| Other suites reading `default_params` (`tests/gate_s172_prod_shape.py:77`, `tests/test_gate_prod_shape_scoping.py:332`, `tests/test_s172_d6_2_checkpoint_reconciliation.py:1532`, `tests/test_s172_staging_backpressure.py:1053`, `tests/test_s172_staging_partb.py:541`, `tests/test_step1_search_record_stage1.py:1283`) | They read other keys (`staging_dir`, `resume_checkpoint`, `step1_search_id`) | No effect expected; confirm by the before/after pass |

**What a RANGE-MINER launch would see differently: nothing, for launches that name the engine.** Run 4
(`gate12_launch.sh:467-473`) passes `"use_persistent_workers": false, "use_range_miner": true` explicitly. Its
`final_params`, CLI, execution-set backend (`miner`) and fleet rule (`miner_backed=True`) are therefore identical
before and after. A direct `window_optimizer.py` launch never reads the manifest. Proof at Stage 3:
- a JSON key-path diff showing that only `:262`, `:270` and the `use_range_miner` param spec changed;
- a CPU test that runs WATCHER's merge (`run_step`'s block and `_step1_cli_final_params`) on Run 4's params, before
  and after, and requires identical `final_params`;
- the same for `_step1_declared_params`, which must give backend `miner`.

**What changes, by design** (the point of the ruling):
1. A WATCHER Step 1 that does **not** name an engine now goes to RANGE-MINER, with its preconditions: the fleet, a
   provisioning manifest (`miner_backed` raises without one), and the miner's admission count.
2. **Choosing PWC "only by the flag":** with `use_range_miner: true` as the default, a caller who passes only
   `use_persistent_workers: true` gets both flags. `window_optimizer.py:1754-1765` then refuses. So either:
   - (a) the PWC caller must also pass `use_range_miner: false` (no code change; write it in the runbook), or
   - (b) an explicit PWC flag clears the default miner flag. That is a WATCHER or `window_optimizer.py` code change,
     in a shared launch file, and needs a proof that a RANGE-MINER launch is unchanged.

   **Owner question Q-17.** Recommendation: (a). It changes no code, and the mutex stays a hard refusal.

### S3-13 — WITHDRAWN

S3-13 WITHDRAWN by the owner, 2026-11-20 21:32-21:35 PST ("Well range miner must be fixed none the less"; "Please do whatever method is the quickest and least recoding"). It supersedes the 16:45 F4 ruling for S3-13 only. **F4: PWC keeps the trial's skip range; the RANGE-MINER fix is scoped separately in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`**. No PWC code changes for the skip range. The earlier
option analysis ((A) constant plus drift test, (A2), (B)) is dropped; it is kept only in git history of this draft and in
agent 2's `docs/drafts/pwc_backup_stage3a_v1/S3-13_removed.diff`.

Consequence for Stage 4 (F4 below): until the RANGE-MINER fix lands, the two engines sieve different constant-phase
skip ranges for any trial whose sampled range is not `[0, 16]` (RANGE-MINER uses its literal default,
`miner/range_miner_worker.py:1293`; PWC sends the trial's `[skip_min, skip_max]`, `persistent_worker_coordinator.py:2627` at b2efdef).

**Shared tests whose PWC lines change** (edit only those lines; prove the rest unchanged, as in Stage 2):
- `tests/test_s172_miner_seed_start.py:886-889` — mutant M6 anchors the deleted `_route_start_guard(seed_start, route="pwc",` line. `g_r5_routes` expects PWC refused at 1000. Its M8 (`:890-892`) mutates the guard predicate S3-1 changes.
- `tests/test_step1_search_record_stage1.py:884-898` — G1 asserts PWC evidence is refused. The `M-PWC/ZMQ` mutant is at `:1702-1706`.
- `tests/test_s172_d6_2_checkpoint_reconciliation.py:2412`, `:2507`, `:2553` — they observe the NP2-KILL ssh line (only if S3-6 is approved).
- `tests/test_s172_phase5_d6_production_adapter.py:754` — a source marker for the `_use_pw` line (check that it survives).
- New: `tests/test_pwc_backup_stage3.py` (the brief's E3: G-START, G-CREDIT including a zero-survivor complete trial,
  G-PROV, G-FLEET, G-KEYS), plus a NP2-KILL gate if S3-6 is approved.

---

## C. How "nothing else moved" will be proved

1. **`git diff --exit-code 0caeaf9 --`** on `miner/`, the ten pinned files (`scripts/gate12_parity_gate.py:174-185`),
   `sieve_filter.py`, `utils/`, `scripts/launch_fleet_manual.sh`, `gate12_launch.sh` and `distributed_config.json`
   must print nothing. Also run `derive_worker_import_closure()` from the gate, locally, and require the derived set
   to equal `0caeaf9`'s.
2. **`scripts/prove_unchanged_outside.py`** (pwc-offset, uncommitted; Q-14), once per edited Python file, with
   `--base 0caeaf9 --strict-bytes`:
   - `window_optimizer.py --allow assert_step1_route_seed_start_supported run_bayesian_optimization run_with_config main`
   - `window_optimizer_integration_final.py --allow run_bidirectional_test _step1_pwc_sweep_evidence add_window_optimizer_to_coordinator.optimize_window`
   - `agents/watcher_agent.py --allow <the step-function qualname holding :2119>` (plus the literal block if S3-10 needs it)

   This proves that every other top-level statement and every other definition is AST-identical. Comment-only
   changes outside the allowed names are caught by `--strict-bytes`.
3. **The gap the script cannot close: it works per function.** `run_bidirectional_test`,
   `add_window_optimizer_to_coordinator.optimize_window`, `main`, `run_bayesian_optimization`, `run_with_config` and
   the WATCHER step function each contain **RANGE-MINER code and the PWC line side by side**. Allowing the whole
   function would hide a change to its miner part. So Stage 3 adds one small PWC-owned companion (a new
   `scripts/pwc_stage3_block_proof.py`, or a gate in `tests/test_pwc_backup_stage3.py`), with a mutant for each rule.
   Inside each of those functions it compares the body statement by statement (`ast.dump`, no positions) and allows
   only:
   - in `run_bidirectional_test`: the single `If` whose test is `_use_pw`. Every statement before it, including the
     whole `if _use_miner:` subtree, and every statement after it must be identical;
   - in `main`: the one new `add_argument('--pwc-coordinator-address', …)`, the added `require_fleet=` keyword on the
     `run_start_dataset_gate` call, and the added `pwc_coordinator_address=` keyword on the two calls;
   - in `run_bayesian_optimization` / `run_with_config`: the added parameter and the added
     `coordinator.pwc_coordinator_address = …` statement;
   - in `optimize_window`: the NP2-KILL block (if S3-6 is approved), the `_partition_worker` signature and wiring, and
     one appended element of the spawn `args=` tuple;
   - in the WATCHER step function: the added `require_fleet=` keyword.
4. **Manifest:** load the base and new JSON; every base key path must have an identical value, and the only
   difference is the added `pwc_*` key or keys.
5. **Behaviour tables (CPU):**
   - S3-1: `seed_start ∈ {0, 2³¹} × route ∈ {miner, pwc, zmq, legacy}`. Only ZMQ at 2³¹ raises.
   - S3-5 / S3-9: `engine ∈ {miner, pwc, zmq, legacy} × provisioning manifest {present, missing}`. Miner and PWC
     raise when it is missing; ZMQ and legacy are unchanged.
6. **Suites:** the same runner and suite list as Stage 2 (`logs/pwc_stage2_v1/run_suites.sh`, minus
   `tests/test_s172_staging_partb.py`), with a fresh `before` on `0caeaf9`, **only when the owner allows a full
   pass**. Identical results are required outside the PWC suites and the named PWC lines of the shared suites.

---

## D. Open questions for the owner (one per item)

Default rule: PWC does what RANGE-MINER does.

| # | Question | What RANGE-MINER does today | Recommendation |
|---|---|---|---|
| Q-1 | **Who does the coordinator side of the split?** It is PWC-only (S3-11). Do it in Stage 3, or as Stage 2b's own follow-up? | Sends `window_anchor` and `generator_phase` (pinned 0) in every assignment (`miner/range_miner_worker.py:22-29`, `:601-605`) | Stage 2b's follow-up, in the same review as the worker and `sieve_filter.py` change. That keeps Stage 3 to the shared launch files. Lift the offset refusal only after the new worker is on the rigs |
| Q-2 | **Where does PWC get the address rig workers dial?** | The coordinator host is an explicit, required launcher argument with no default (`scripts/launch_fleet_manual.sh:147`, `:239-241`; Run 4 passes `192.168.3.177`, `gate12_launch.sh:316`) | A CLI flag and a manifest key, `pwc_coordinator_address`, **no default**. PWC with rigs refuses when it is unset (already true since Stage 1) |
| Q-3 | **PWC chunk sizing defaults are outside the stable envelope.** The manifest has `seed_cap_amd` 2,000,000 (`agent_manifests/window_optimizer.json:265`), the CLI default is the same (`window_optimizer.py:1576`), and so is the integration fallback (`window_optimizer_integration_final.py:1980`). A PWC run on defaults therefore refuses at startup | Sizes its own work by stripe keys (`miner_stripe_size`, manifest `:225`, `:233`; Run 4 passes 67,108,864) | Change no default in a shared file. PWC keeps refusing until the operator passes values inside the envelope (6 workers at ≤ 50,000, or 8 at ≤ 25,000). Write those values in the runbook |
| Q-4 | **Fix the NP2-KILL self-kill (S3-6)?** It is an unbracketed `pkill` to hardcoded bare-metal IPs with an unread result, and it runs for **every** engine when `n_parallel > 1` | Nothing PWC-related; the miner's own launcher handles worker lifecycle (`scripts/launch_fleet_manual.sh`) | Yes: run it only for PWC, over the configured nodes, with the probe-checked kill, reading its result. This removes a pointless ssh from miner runs with `n_parallel > 1` (Run 4 is `n_parallel` 1) and needs PWC-line edits in `tests/test_s172_d6_2_checkpoint_reconciliation.py` |
| Q-5 | **Which PWC runs earn credit?** | One transport, with the threshold provenance wall (`window_optimizer_integration_final.py:1330-1346`) and stripe-ledger sweep evidence (`:1479`) | Credit only when the trial is complete, every pass ran on **TCP**, and every pass's threshold record agrees and is verified. SSH-route passes earn none |
| Q-6 | **SSH route (F1):** remove it, or keep it for constant-only diagnostics? | Has one transport | Keep it for diagnostics, with no credit (Q-5). Remove later if unused |
| Q-7 | **HSA variables (F2):** drop all three, which needs worker and coordinator edits together? | Sets none (`miner/`, `scripts/launch_fleet_manual.sh`, `gate12_launch.sh`) | Leave them until a rig session can test without `HSA_ENABLE_DEBUG_TRAP=0`, the S165 crash fix. Not Stage 3 |
| Q-8 | **PWC cache records (F3)?** | Its launcher writes v7 cache records, checked by `scripts/cache_records_verify.py` | Write them for credited PWC runs in a later stage. Not Stage 3 |
| Q-9 | **The PWC ready gate on the pipeline route.** `run_bidirectional_test` does not pass `min_workers` to `run_trial_persistent` (`window_optimizer_integration_final.py:1968-1986`), so PWC's gate is 1. The CLI sets `coordinator.pwc_min_workers` (`window_optimizer.py:868`) and `coordinator.min_workers` (`:1326`), and neither is passed on | Admits workers by the frozen execution set (`execution_set.py`) | ~~Pass the coordinator's value.~~ **Superseded 2026-11-21.** The owner's ruling of 2026-08-04 governs: the full-fleet ready gate stays retired, it was a utilisation check and not a correctness gate, and RANGE-MINER's own admission is `min(requested, selected identities)` (`execution_set.py:653-676`). PWC keeps `b2efdef`'s call shape (effective minimum 1). Beta R1 3b-Q7 (LIVE): Stage 3 passed 24, which refused 19 of 19 ready |
| Q-10 | **Five strategies (F5):** confirm them in place of S147 Q2? | Loads all five (`miner/range_miner_worker.py` `_load_strategies`) | Confirm. Beta's Q3 (rig cost) stays **NEEDS EVIDENCE** for the Stage 4 rig sessions |
| Q-11 | **The S147 prune behind a flag (F6)?** | Has none | Leave it deleted |
| Q-12 | Ruled 16:45 ("Yes, PWC follows range miner …"), then **S3-13 WITHDRAWN** 21:32-21:35 | **F4: PWC keeps the trial's skip range; the RANGE-MINER fix is scoped separately in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`** | None in Stage 3 |
| Q-13 | **Exact threshold echo** (`sieve_filter.py:597-599` rounds to 4 places) | Compares exact values | Fold the one-line unrounded echo into Stage 2b's `sieve_filter.py` change |
| Q-14 | **`scripts/prove_unchanged_outside.py`** is uncommitted in `pwc-offset` | — | Land it with Stage 2b, so Stage 3 uses it from main. I will not copy or edit it |
| Q-15 | **RULED 2026-11-20 15:53: "Yes, range miner is the new default"** (see S3-12) | Run 4 names the engine explicitly (`gate12_launch.sh:467-473`) | Do it in Stage 3 as S3-12, including the `tests/test_s172_phase1_scaffolding.py:177-181` edit |
| Q-17 | **How does a caller choose PWC once RANGE-MINER is the default?** Passing only `use_persistent_workers: true` would set both flags, and the mutex refuses (`window_optimizer.py:1754-1765`) | — | (a) The PWC caller also passes `use_range_miner: false`: no code change, documented. Not (b), which makes an explicit PWC flag silently clear the default |
| Q-16 | **The per-run launch token** (scope A3 level 2, Beta Stage 1 Q3) needs the worker to send it | Release token / `RUN_NONCE` (`gate12_launch.sh:315`, `scripts/launch_fleet_manual.sh`) | Bundle it with Stage 2b's worker edit, so the worker is reviewed once |

---

## E. Order of work proposed

1. Stage 2b lands: worker + `sieve_filter.py`, plus `scripts/prove_unchanged_outside.py`.
2. Stage 3 code is written in one session (S3-1 … S3-10), with the block-level companion proof and
   `tests/test_pwc_backup_stage3.py`.
3. Targeted CPU tests.
4. A full before/after pass, **only when the owner allows it** (agent 2's timing-sensitive pass first).
5. The Beta before-commit review: Stage 3 is launch-path code. Beta also re-reads the Stage 2 round-3 transport lines,
   as recorded in `docs/exchange/to_beta/X-PWC-STAGE2_R3_ALPHA_CLOSING_NOTE.md`.
6. The coordinator side of the split (S3-11), and lifting the offset refusal, only after the new worker is deployed
   (owner-run rig session; gate12 parity and PWC parity gates).

## F. Stage 4 — the final two-engine test (owner requirement 2026-11-20 15:55 PST; design only)

**Requirement, verbatim:** "And once finished your testing should run each and verify the results are the same".
Recorded at `docs/PWC_BACKUP_OWNER_RULINGS_20261119.md:47` (this worktree, uncommitted). Meaning: run the same job
through RANGE-MINER and through PWC, each selected by its flag, and compare what comes out. Nothing here was run.

### F1. What the existing parity gates compare today: code, not output

| Gate | What it compares | Anchors |
|---|---|---|
| `scripts/gate12_parity_gate.py` | **Deployed code bytes** of RANGE-MINER's worker on each rig against the local clean tree. Full sha256 and size of a pinned file set (`GOVERNED_FILES`, `:174-185`), expected digests derived at run time (`expected_digests`, `:306-321`), probed over ssh (`build_probe_script` `:409`, `probe_rig` `:492`). The pin is cross-checked against the worker's derived import closure (`derive_worker_import_closure`, `:262`; `closure_coverage`, `:285`). It refuses if the local governed files are dirty (`local_worktree_dirty`, `:324`). Content identity, never git identity (docstring, `:18-30`) | the ten pinned files |
| `scripts/pwc_parity_gate.py` | The same idea for **PWC's** worker entry points: `ENTRYPOINTS` (`:102-105`, `tcp` → `persistent/pwc_worker_service.py`, `ssh` → `sieve_gpu_worker.py`), the governed file sets per transport (`GOVERNED_FILES`, `:107-131`, which include `sieve_filter.py` and `persistent/*` for TCP), and the minimum files (`MINIMUM_FILES`, `:133-136`) | per transport |
| `tests/phase6/wall_ab_gate.py` (not a parity gate, but the nearest **output** comparator) | Wall B compares two runs' **22-array bundles** field for field and localizes any difference: array, first differing index, both values (`compare_bundles`, `:676-718`). It also compares canonical order and `artifact_sha256`. Used for miner repeat vs repeat (B2, `:900-905`), serial vs sharded (B3) and node assignment (B4) | miner vs miner only |

**So nothing today compares RANGE-MINER output with PWC output.** Stage 4 adds that comparison. It reuses
`compare_bundles` (or a copy in a PWC-owned file) on the two engines' 22-array bundles.

### F2. What "the same job" must hold equal (inputs)

| Input | Rule | Today's gap |
|---|---|---|
| Seed range | Identical `[seed_start, seed_start + seed_count)` | PWC refuses a nonzero start at the launch layer until **Stage 3** (S3-1, S3-2). Until then only `seed_start = 0` is comparable |
| Window (size, `window_anchor`, sessions, dataset) | Identical. The dataset is the same frozen version (pointer manifest and sha256) | PWC refuses offset ≠ 0 until **Stage 2b** and the coordinator side (S3-11) land. Run 4's anchor 25 is not comparable before then |
| Generator phase | 0 for both (`miner/range_miner_worker.py:605`) | PWC sends `offset` as the phase until S3-11. At offset 0 they match |
| Directional thresholds | Identical forward and reverse values, given explicitly, not sampled | None: both put `float32(threshold)` in the kernel. PWC's 4-place report affects only the check, not the output |
| Constant skip range (phases 1–2) | Identical only when the trial is pinned to `[0, 16]` | **F4: PWC keeps the trial's skip range; the RANGE-MINER fix is scoped separately in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`**. Until the miner fix lands, pin the comparison trial to `skip_min=0, skip_max=16` |
| Strategies (phases 3–4) | All five, the same order (`hybrid_strategy.get_all_strategies()`) | None since Stage 2 |
| Phases | 1–4 (`test_both_modes: true`), never cut short | None since Stage 2 |
| PRNG family | `java_lcg` (both engines' Run 4 target) | — |
| Transport | PWC on **TCP** (SSH refuses variable-skip) | — |

Engine selection: Stage 3 S3-12 makes RANGE-MINER the default. The PWC run passes `use_persistent_workers: true` **and**
`use_range_miner: false` (Q-17, recommendation (a)).

### F3. What is compared (outputs)

| Output | Equality required | Notes |
|---|---|---|
| Survivor set per phase (1–4) | **Identical** seed sets, and each seed's match rate equal | Compare as sorted sets. Both engines run the same kernels, so float32 rates are expected to be equal bit for bit [UNVERIFIED until run] |
| Bidirectional sets (constant, variable) | Identical | The intersections of the above |
| The 22-array NPZ (`utils/canonical_arrays.py:85-120`, the frozen contract) | **All 22 arrays equal**, in canonical order (`compare_bundles`) | `artifact_sha256` may still differ if the sidecar or metadata names the engine or run [UNVERIFIED which fields it covers]. The verdict rests on the arrays, with the sha reported |
| Per-phase counts | Identical forward/reverse counts for each phase | PWC: `pwc_completion.passes[ph]`; RANGE-MINER: the stripe ledger rows and the trial result |
| Search credit | Both grant credit, with identical `swept_intervals` `{1..4: [s, e]}` through the unchanged `credit_decision` (`utils/step1_search_ledger.py:1044`) | Needs Stage 3 (PWC credit, S3-3). Before Stage 3 PWC is refused by design and this row is not comparable |
| Steps 2–6 input | Optional: the Step-2 loader on each NPZ (Wall A, `tests/phase6/wall_ab_gate.py` A3–A5) | Proves "later steps cannot tell the engines apart" on real output |

**May legitimately differ:**
- timestamps and durations;
- run ids, search record ids, trial study ids;
- engine or route name fields, and log paths;
- job, lease, stripe and chunk ids, and the chunk/stripe geometry (PWC chunks against miner stripes), which are in
  run records, not in the NPZ;
- worker ids, hostnames and GPU assignment (Wall B4 already shows node assignment does not change miner output);
- the artifact sidecar's provenance block [UNVERIFIED which keys], and therefore possibly `artifact_sha256`.

**Must not differ, ever:** any of the 22 arrays, any per-phase survivor set, or the swept intervals. A difference is
localized and reported, never summarized as a single verdict, as Wall B does.

**Yardstick note:** equal output means that, on this input, PWC is a faithful backup. It says nothing about hit rate.
Pools built from either engine are still judged against `k/1000` on unseen draws, and that measurement is separate.

### F4. The smallest rig session that would show it

Preconditions:
- Stages 2b and 3 are landed and redeployed;
- the PWC parity gate (`--transport tcp`) and the gate12 parity gate pass on the rigs;
- the trial is pinned to skip `[0,16]` until the RANGE-MINER skip-range fix lands (S3-13 withdrawn 21:32-21:35; scope in `docs/drafts/pwc_backup_stage3a_v1/SCOPE_MINER_SKIP_RANGE_FIX.md`).

**Smallest useful run:**
- one rig (8 cards) plus VM101;
- one trial, `java_lcg`, `test_both_modes: true`, explicit thresholds;
- seed interval `[2^31, 2^31 + 2^28)`, one eighth of a half range;
- first RANGE-MINER, then PWC with the same pinned parameters. Each is launched by its flag, one after the other,
  never concurrently.

Why that size:
- the banked half runs show phase 3 ≈ 60 and phase 4 ≈ 23,500 survivors over `2^31` seeds (Run Q 60 / 23,568; Run P
  59 / 23,410; 2026-11-19 handoff). One eighth would be expected to give a few phase-3 and a few thousand phase-4
  survivors, if survivors are spread evenly [UNVERIFIED], so every phase has something to compare;
- PWC must stay inside its stable envelope (8 workers at ≤ 25,000 seeds per chunk), which is about 10,700 chunks at
  that size. The running time on one rig is not measured [UNVERIFIED]. Measure the first minutes and stop if it is
  far from the estimate.

Two further rows, if the first is equal:
1. **A zero-survivor control:** a small interval where a phase finds nothing. Both engines must report complete,
   empty and credited.
2. **The production shape:** the upper half with Run 4's pins, PWC only, compared against a fresh RANGE-MINER run of
   the same pins. It could be compared against Run Q's banked output instead, if Run Q's NPZ and survivor files are
   kept [UNVERIFIED: `/home/michael/runQ_evidence_20261119` holds recorder, manifest and PCIe files; the NPZ location
   was not checked]. This needs offset 25, so Stage 2b and S3-11 first.

This session also produces Beta's R1-Q3 evidence: phase-3/4 chunk times and completion, peak memory, KFD/MES events,
and the 600 s timeout against the observed tail. The owner runs it; agents prepare the commands and the comparison
script. Comparator to write (a PWC-owned file, CPU only, reading two run outputs):
- survivor sets per phase;
- `compare_bundles` on the two NPZs;
- per-phase counts;
- the two search records' `swept_intervals` and credit;
- a list of every differing field, each marked "allowed" (F3) or "NOT allowed".

## Not verified in this preparation

- Whether WATCHER's literal key list (`agents/watcher_agent.py:506-521`) must change for a new manifest arg (S3-10).
- The exact qualname of the WATCHER step function holding `:2119`.
- Whether any suite other than those listed pins the `_use_pw` block's text. Only the grep above was run; no suite
  was executed.
- Anything on a rig.
