# Chapter 12 audit — WATCHER agent, fingerprint registry, and RANGE-MINER routing

Repo: `/home/michael/distributed_prng_analysis` on VM 101 at `a15b524` (`a15b5247336255801de8cf67d52653701b1e8c80`),
working tree, with git history and host access (read-only).
Audited: `docs/CHAPTER_12_WATCHER_AGENT.md` (880 lines at `a15b524`, v1.4.0 of 2026-02-03) against the code.
Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` REV1.
Method: read-only. Governance trail first, then fact file / TODO_CONSOLIDATED / behaviour model, then code.
Every anchor below was opened in this session. Nothing was executed — no WATCHER, no pipeline, no `--help`;
argparse was read. "Works" means the code path is complete on reading.
Step numbers are the EXECUTABLE scheme (Step 2 = scorer meta-optimizer; the sieve runs inside Step 1).
Framing held throughout: TFM is functional mimicry of surface output; no finding here assumes a "true seed".

---

## 1. Verdict

- Accurate: **24**
- BROKEN: **6** (one governed: B1)
- SWITCHED-OFF-BY-RULING: **3**
- DESCRIPTION-ONLY: **27**
- UNVERIFIED: **5**

(65 claims. Chapter 880 → 1,577 lines; no line of the original removed, `git diff` shows 0 deletions.)

Three things matter more than the rest:

1. **WATCHER's verdict on every step is file existence, not the step's result.** All seven manifests declare
   `evaluation_type: file_exists` (`agents/watcher_agent.py:946`), so the heuristic and LLM evaluators the
   chapter describes are not reached (host log: 692 of 714 decisions `file_exists`, 0 heuristic), and a
   failed Step 1 whose `optimal_window_config.json` already exists is reported PASSED (B1, governed, open).
2. **RANGE-MINER is reachable from WATCHER only by explicit override.** `use_range_miner` defaults to
   `false` and PWC to `true` in the manifest; an operator must set both (`gate12_launch.sh:377-383`) or the
   optimizer's backend mutex aborts Step 1. Holding the default off is governed (S1).
3. **The fingerprint registry is not wired.** The chapter's title component has no WATCHER caller (B2).

---

## 2. The §5 host questions and pre-set questions

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Does `agent_manifests/window_optimizer.json` still default `use_range_miner` to `false`? | **Yes. Recorded, not changed.** `default_params.use_range_miner: false` (`:269`); `default_params.use_persistent_workers: true` (`:261`); `parameter_bounds.use_range_miner.default: false` and description "Phase 1 is scaffolding-only … Do not enable in production until Phase 7 acceptance completes" (`:212-218`). | `cat -n agent_manifests/window_optimizer.json` |
| 1a | Classification | **SWITCHED-OFF-BY-RULING (S1).** Spec `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:436` sets it `true ── once TB approves`; Phase 7 is unblocked (`docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md`) but the soak has not run; `docs/TODO_CONSOLIDATED.md` §(a) lists "Decision owed after Phase-7 acceptance". Lifting it: WATCHER Step 1 on RANGE-MINER without a per-run override; must be paired with `use_persistent_workers: false` or the mutex (`window_optimizer.py:1686-1697`) fires. | governance search: `/bin/grep -l use_range_miner docs/TB_RULING_* docs/PROPOSAL_* docs/TEAM_ALPHA_*` → 6 files, each read at the hit |
| 2 | `agent_manifests/definitions.json` (gitignored) | Present, 6,635 B, mtime 2026-01-01; ignored by `.gitignore:41` (`*.json`); never tracked (`git log --all --` empty). Schema doc: `pipeline_steps` (Step 5 emits / Step 6 consumes `signal_quality`, `data_context`, `feature_schema`), `watcher_protocol` (degenerate-fingerprint registry at `agent_state/degenerate_fingerprints.json`, 5-step recovery workflow, fingerprint = sha256 of `{train_start}:{train_end}|{holdout_start}:{holdout_end}|{survivor_count}|{prng_type}|{mod}` truncated to 8, `:154-158`), `sidecar_schema` 3.1.0. **No reader** in any `*.py`/`*.sh` (tracked or untracked); `agent_state/` absent on host. | `cat -n`, `git check-ignore -v`, `git log --all`, `/bin/grep -rln definitions\.json --include=*.py --include=*.sh .` |
| 3 | Step numbering | Executable scheme stated in the chapter header. Chapter's "Step 2.5" for scorer_meta is conceptual; `scorer_meta.json` `pipeline_step` is `2`. | `agents/watcher_agent.py:387-417`; manifest parse |
| 4 | Hit-rate memory note | Verified: WATCHER (`watcher_agent.py`, `watcher_dispatch.py`) computes no hit rate. Backtest recall `evaluate_pools.py:34`; live trigger `current_hit_rate = exact_hits / max(pool_size, 1)` `chapter_13_diagnostics.py:531`; classes kept separate `docs/TB_RULING_S176_WATCHER_KPI.md:497`. Recorded in chapter §13.6 only as a disambiguation. | `/bin/grep` |
| 5 | WATCHER host state | No `watcher_daemon.pid`, no `daemon_state.json` → daemon not running. `pending_approval.json` present (2026-02-15, `status: failed`, `ch13_20260215_084821`, steps `[3,5,6]`). `watcher_decisions.jsonl` 714 records, last 2026-08-23 (three Step-1 escalations "File does not exist: optimal_window_config.json"). `systemctl list-units --all`: no WATCHER/TFM unit (only `nvidia-gpu-policy.service`, failed — not WATCHER). `crontab -l`: none. `/tmp/agent_halt`, `watcher_halt.flag`: absent as seen from this session (`/tmp` → UNAVAILABLE, U4). | `ls -la`, JSON parse, `systemctl`, `crontab -l` |

---

## 3. What actually works end to end today

Complete on reading; none was run.

1. **Pipeline loop.** `run_pipeline` (`agents/watcher_agent.py:2717`) → `run_step` (`:1705`) → `evaluate_results` (`:913`) → `execute_decision` (`:1317`) → PROCEED advances, ESCALATE halts (`:1375`, `:1459`).
2. **Run-start gates in `run_step`:** execution-set freeze once per process (`_ensure_execution_set` `:1578`), preflight HARD/SOFT, freshness with `--force-step` bypass, step-scoped param merge (`:1811-1820`), P0.5 dataset authority (`:1860` on), Step-1 S145 coverage cursor + seed-domain wall, argv build (`:2080-2155`), streaming Popen (`:2601`), Step-1 no-timeout (`:3575`, `:2636`).
3. **Step 1 → RANGE-MINER by override.** Manifest → merge → `_ensure_execution_set` backend `miner` (`:1604`) → `--use-range-miner` → `window_optimizer.py` mutex (`:1686-1697`) → `window_optimizer_integration_final.py:1402-1440` → `run_trial_miner` (`miner/range_miner_coordinator.py:12336`). Production use: `gate12_launch.sh:377-383` (Gate 12 attempt 9 PASSED).
4. **Run-4 Route-A pin routing** (`3e2537b`): CLI seam `split_operator_pin_params` (`:630`, call `:3820`) → `capture_step1_pin_bundle` (`:560`) / `assert_no_unauthorized_pin_keys` (`:668`) → WALL 1 (`:1828`) → WALL 2 (`:2131-2134`) → provenance (`:708`).
5. **Daemon:** PID file, signal handlers, state persistence, per-cycle request scan and approval poll (`run_daemon` `:3115`; scan `:3189`).
6. **Request dispatch:** `watcher_requests/*.json` → `selfplay_retrain` / `learning_loop` / `pipeline_rerun` (`agents/watcher_dispatch.py:503-540`), archived with status (`:718-732`).
7. **Safety on the pipeline path:** `/tmp/agent_halt` checked before and during `run_pipeline` (`:2777`, `:2804`) and in the daemon (`:3175`); escalation writes it (`:1473`); `--clear-halt` removes it.
8. **Decision log:** `watcher_decisions.jsonl` append (`:902-907`, `:1340-1350`) — 714 real records on host.

---

## 4. BROKEN

**B1 — A failed step can be reported PASSED (failure authority).** *(GOVERNED, open: `docs/TFM_PROJECT_FACTS_SKILL.md` §2.34, §2.51 #9; `docs/SESSION_CHANGELOG_20260811_CLEANTREE_ADMISSION.md:220-223`; `docs/TODO_CONSOLIDATED.md` §(a))*
- Chapter §1.1, §7: WATCHER evaluates results and escalates.
- Code: `run_step` returns `{"success": False, "return_code": …}` on non-zero exit (`agents/watcher_agent.py:2177-2185`); `evaluate_results` takes the file-exists branch for every manifest (`:946-1000`) and never reads `results`.
- Stops at: `agents/watcher_agent.py:946`.
- Repair must: make an explicit subprocess/finalization failure dominate the file-existence verdict.

**B2 — Fingerprint registry and hooks are not called by WATCHER.** *(not found in governance)*
- Chapter title, §1.1, §4, §5, §11.
- Code: no import of `watcher_registry_hooks` / `fingerprint_registry` in `agents/watcher_agent.py` or `agents/watcher_dispatch.py`; only importer `agents/registry_inspector.py:30`. DB `agents/data/fingerprint_registry.db` mtime 2026-01-02. Wiring design exists only in `docs/PROPOSAL_Modular_Step_Automation_Framework_v1_1.md` (status DRAFT, `:6`).
- Stops at: `agents/watcher_registry_hooks.py:151` (`pre_run_check`, no caller).
- Repair must: call pre-run / post-run hooks from the run path with a defined fingerprint source, or the chapter must stop claiming the behaviour (owner choice).

**B3 — Two halt files.** *(not found in governance)*
- Chapter §8.1, §10.1.
- Code: `--halt`/`create_halt`/escalation/`check_safety` use `/tmp/agent_halt` (`agents/safety/kill_switch.py:60`, `:361-371`; `agents/watcher_agent.py:366`, `:1473`, `:2777`); dispatch checks `watcher_halt.flag` (`agents/watcher_dispatch.py:126`, `:296`, `:386`), which no tracked code writes.
- Stops at: `agents/watcher_dispatch.py:126`.
- Repair must: one halt authority read by the pipeline, the daemon and every dispatch path.

**B4 — Request validation rules are not implemented.** *(not found in governance; rules occur only in this chapter)*
- Chapter §11.4: age < 24 h, priority, cooldown > 1 h, cluster not busy.
- Code: `_validate_selfplay_request` checks `1 ≤ episodes ≤ 50` and survivors-file existence only (`agents/watcher_dispatch.py:598-617`); `created_at`, `priority`, `cooldown`, time arithmetic absent from the file.
- Stops at: `agents/watcher_dispatch.py:598`.
- Repair must: implement (or formally drop) the four checks; relates to `docs/CHAPTER_13_AUDIT.md` B7 (cooldown never reset).

**B5 — RETRY does not apply adjustments (dormant).** *(not found in governance)*
- Chapter §3.3. Code: `_handle_retry` logs `suggested_param_adjustments` and re-runs with the same params (`agents/watcher_agent.py:1452-1454`); only the Step-5 health retry builds new params (`:2360`). Unreachable under the current file-exists manifests (no RETRY produced).
- Repair must: thread bounded adjustments into the re-run, or document that generic RETRY is a plain re-run.

**B6 — `--dry-run` does not apply to `--run-pipeline`.** *(spec mandates it: `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:443-450`, §11.J `:590-592`)*
- Code: `--dry-run` (`agents/watcher_agent.py:3556`) is read only in the three dispatch branches (`:3836`, `:3840`, `:3845`); the `--run-pipeline` branch (`:3808-3828`) ignores it, so the spec's mandatory dry-run command would launch Step 1.
- Stops at: `agents/watcher_agent.py:3808`.
- Repair must: either a real non-executing manifest→argv check for `--run-pipeline`, or refuse `--dry-run` with `--run-pipeline`. Safety-relevant (CLAUDE.md rule 3).

---

## 5. SWITCHED-OFF-BY-RULING

**S1 — RANGE-MINER is not the WATCHER default.** `agent_manifests/window_optimizer.json:269` (`false`), `:261` (PWC `true`). Held by: `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:436` ("once TB approves"); manifest `:215`; Phase-7 soak not run (`docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` Phase 7 UNBLOCKED; `docs/TODO_CONSOLIDATED.md` §(a)). Lifting enables: Step 1 on RANGE-MINER with no per-run `--params`; G-PROD-SHAPE's "manifest defaults" path then tests a real default. Requires flipping `use_persistent_workers` too.

**S2 — Steps 2–6 through WATCHER on RANGE-MINER output.** `STEP_SCRIPTS[2]` = `run_scorer_meta_optimizer.sh` (`agents/watcher_agent.py:390`) calls `convert_survivors_to_binary.py` (`run_scorer_meta_optimizer.sh:87`) and `mv`s onto the finalizer symlink (`:97`). Held by: `docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` §1 (legacy converter not invoked until D3.0-B closes; Beta's corrected Phase-6 certification scope); launch rule `--end-step 1` (`gate12_launch.sh:73-76`). Lifting enables: a full WATCHER 0→6 run, Chapter-13 `learning_loop` / `pipeline_rerun` (`agents/watcher_dispatch.py:530-537` → `scope="full"`) and `--approve` on certified output.

**S3 — `n_parallel` > 1 on the miner path.** Manifest default `1` (`agent_manifests/window_optimizer.json:260`). Held by: D6.2 certification, `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md` §0.1 ("BINDING (Beta)"). Lifting enables: concurrent Optuna partitions with certified checkpoint recovery (needs its own checkpoint-transaction design).

Per §0 of the authority brief these are review positions the owner may keep or lift. No recommendation.

---

## 6. DESCRIPTION-ONLY

| # | Chapter | Code at `a15b524` |
|---|---|---|
| D1 | "v1.8.0" read as WATCHER | manifest v1.8.0 (`window_optimizer.json:5`); WATCHER v2.0.0 (`watcher_agent.py:12`) |
| D2 | 7 bundle types | 8 named: 1–6, 13, 99 (`agents/contexts/bundle_factory.py:587-596`) |
| D3 | heuristic/LLM evaluation (§1.2, §7.1–7.3) | file-exists short-circuit for all 7 manifests (`watcher_agent.py:946`), intended per `docs/SESSION_CHANGELOG_20260110_WATCHER_FIXES.md` Issue 3; Step-1 heuristic is rate-based (`window_optimizer_context.py:110-144`); no `calculate_confidence` |
| D4 | §1.3 status block | mock-test output (`test_watcher_agent.py:156`); live confidences 1.0/0.0 (`:992`) |
| D5 | 6 manifests | 7 incl. `trse.json` (`watcher_agent.py:399`) |
| D6 | `use_llm=False` default | `True` (`:356`) |
| D7 | `decision.action` | `recommended_action` (`:1366`) |
| D8 | `STEP_SCRIPTS` lines 136-145, step 3/6 scripts | `:387-395`; `run_step3_full_scoring.sh`, `prediction_generator.py`, + step 0 |
| D9 | `STEP_MANIFESTS` lines 147-155 | `:398-406`, + step 0 |
| D10 | `/shared/ml` at lines 35-36 | `run_scorer_meta_optimizer.py:43-44` |
| D11 | `run_step` at line 748, `subprocess.run` | `:1705`, Popen streaming `:2601`, nine-stage sequence |
| D12 | registry `create_fingerprint`/`get_status`/`record_attempt(details=)` | absent; real API `fingerprint_registry.py:205-216`, `:321`, `:379` |
| D13 | `REJECT` action | `REJECT_DATA_WINDOW`, `EXPAND_HOLDOUT` (`watcher_registry_hooks.py:91`) |
| D14 | scorer_meta "Step 2.5" | `pipeline_step: 2` |
| D15 | manifest structure example | live structure `window_optimizer.json:20-75` |
| D16 | `--resume`, `--verbose` | not defined (`watcher_agent.py:3466-3565`); resume = `--clear-halt --run-pipeline --start-step N` (`:1534`) |
| D17 | `logs/watcher_agent.log` | `basicConfig`, no file handler (`:311-315`) |
| D18 | ramdisk: remote nodes not populated | `ramdisk_preload.sh:25-40`, `:130-150` loop all nodes |
| D19 | ~500/~350/~250 lines, 6 manifests | 3,859 / 477 / 655; 7 |
| D20 | script defaults never used | step-scoped merge `:1811-1820`; undeclared keys dropped, script defaults used |
| D21 | requests append-only | moved to `archive/` (`watcher_dispatch.py:718-732`) |
| D22 | `--node-allowlist` CLI flag | coordinator kwarg `coordinator.py:253`; CLI is `--execution-set-nodes` (`window_optimizer.py:1539`) |
| D23 | manifest v1.5.0 | v1.8.0 |
| D24 | timeout from manifest `timeout_seconds`, Step 1 14400 s | manifest timeout unread; CLI overrides `{0:1,1:0,5:360}` (`:3575`), Step 1 none |
| D25 | PWC §12: flag in args_map, PROCEED on survivors > 0, pool 4 | `default_params:261`; file-exists; pool 8 (`:262`) |
| D26 | §12 appears twice | verbatim duplicate |
| D27 | fingerprint = SHA256 of dataset characteristics | sidecar `data_context` formula (`definitions.json:154-158`) |

---

## 7. Chapter-specific: RANGE-MINER routing facts recorded in the chapter (§13)

- Version: S172 Phase 1 `8d0183f`, manifest 1.7.0 → 1.8.0; `run_trial_miner` was a `NotImplementedError` stub then and is real at `a15b524` (`miner/range_miner_coordinator.py:12336`).
- Argparse gate: `window_optimizer.py:1552` flag, `:1686-1697` mutex.
- WATCHER-side: `_ensure_execution_set` `agents/watcher_agent.py:1604-1613` (miner first; `worker_pool_size` = admission count).
- Route-A: as §3 item 4; Beta R2 standing constraint on privileged seams; claim boundary routing-and-containment only; Run 4 not run.
- §11.K pool=8 regression: no record of a run (U2). §11.J dry-run: B6. Phase-7 soak: unblocked, not run; 2026-08-04 attempt failed; G-PROD-SHAPE ran 2026-08-22 and failed (`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md:12`).
- Stale code-side text (not edited, listed for the owner): `agent_manifests/window_optimizer.json:215` and `window_optimizer.py:1548-1551` still say the miner raises `NotImplementedError`; `window_optimizer.json:172` says PWC `default: false` while `default_params:261` is `true`; manifest `:336` cites `watcher_agent.py:1312` for the step-scoped filter (now `:1811-1820`).

---

## 8. Surfaces searched, and surfaces unavailable

**Files opened this session**
- Briefs/governance: `scratchpad/common_brief.md`, `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md`, `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (structure, header, §13–14), `docs/CHAPTER_13_AUDIT.md`, `docs/TODO_CONSOLIDATED.md` (grep + rows), `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1–1.3, `docs/TFM_PROJECT_FACTS_SKILL.md` §2.34, §2.51, §2.54, §8, `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md` (§7, §10, §11.J–K), `docs/PROPOSAL_S172_RANGE_MINER_v1_4_5.md` (grep), `docs/PROPOSAL_Modular_Step_Automation_Framework_v1_1.md` (head, grep), `docs/TB_RULING_REQUEST_CPU_PARALLELISM_S175.md` (grep), `docs/TEAM_ALPHA_D6_ADVANCE_TO_BETA.md`, `docs/TEAM_ALPHA_FLEET_STATE_SUBMISSION.md`, `docs/TEAM_ALPHA_REVIEW_S172_PHASE4_REV3.md` (grep), `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md` (:1-60), `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md` (grep), `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` (head), `docs/TEAM_ALPHA_PHASE7_LAUNCH_NOTICE.md` (head), `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE_7_SOAK.md` (head), `docs/S172_PHASE_7_PREREQ_REPORT.md` (head), `docs/S172_INFRASTRUCTURE_INTERFACE_v1_0.md:85-110`, `docs/ROCm_Saturation_Report_S172.md` (grep), `docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md` (head), `docs/SESSION_CHANGELOG_20260110_WATCHER_FIXES.md`, `docs/SESSION_CHANGELOG_20260811_CLEANTREE_ADMISSION.md:218-226`, `docs/TB_RULING_S176_WATCHER_KPI.md` (grep), `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md` (grep), `docs/PROJECT_FILE_CATALOG.md` (grep: §5.3, §7 gap 4, chapter table), memory note `watcher-kpi-hit-rate-top20.md`.
- Code: `agents/watcher_agent.py` (header, `:305-430`, `:455-730`, `:841-1040`, `:1269-1640`, `:1705-2214`, `:2360-2392`, `:2717-2944`, `:3115-3245`, `:3465-3859`), `agents/watcher_dispatch.py` (`:88-130`, `:263-315`, `:503-617`, grep), `agents/watcher_registry_hooks.py` (`:1-160`, `:252-270`), `agents/fingerprint_registry.py` (`:1-42`, `:205-225`, defs), `agents/safety/kill_switch.py` (defs), `agents/full_agent_context.py:300-340`, `agents/contexts/__init__.py:41-84`, `agents/contexts/window_optimizer_context.py:110-155`, `agents/contexts/bundle_factory.py` (`:93-100`, `:586-600`, grep), `agents/manifest/agent_manifest.py` (grep), `window_optimizer.py` (`:1485-1570`, `:1680-1740`, `:850-870`, grep), `window_optimizer_integration_final.py:1393-1445`, `miner/range_miner_coordinator.py` (`:2478-2510`, `:12336-12370`), `persistent_worker_coordinator.py` (grep), `coordinator.py` (grep), `run_scorer_meta_optimizer.py` (grep), `run_scorer_meta_optimizer.sh` (grep), `run_step3_full_scoring.sh` (grep), `ramdisk_preload.sh:20-150`, `gate12_launch.sh:60-80`, `:370-392`, `evaluate_pools.py:28-40`, `chapter_13_diagnostics.py` (grep), `chapter_13_triggers.py:616`, `test_watcher_agent.py:87-182`, root `watcher_agent.py` (grep), `dry_run_s115.py` (grep).
- Manifests: all nine files in `agent_manifests/` (seven parsed; `window_optimizer.json` and `definitions.json` read in full).
- Host: `watcher_decisions.jsonl` (parsed), `pending_approval.json` (parsed), `agents/data/` (ls).

**Search method for absence claims:** `/bin/grep -rn` / `-rln` over the working tree including untracked and gitignored files (excluding `backups/`), with `--include=*.py --include=*.sh` where stated; `git log -S`, `git log --all -- <path>`, `git cat-file -t` for history.

**Governance trail searched:** `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*` (138 files) by grep for `use_range_miner`, `registry_hooks`/`fingerprint`, `file_exists`, `evaluation_type`, `watcher_halt.flag`/`agent_halt`, `failure authority`, `11.K`/`pool=8`.

**Chapters searched:** Chapter 2 (structure only), `CHAPTER_13_AUDIT.md`, `PIPELINE_BEHAVIOUR_MODEL.md`; Chapter 3 audit by grep hit only.

**Unavailable**
- `/tmp` from the agent sandbox (halt-file state) — U4.
- Rigs / CT100 state; the fleet was not touched.
- Results that need runs: Phase-7 soak, §11.K, `test_watcher_agent.py`, `tests/test_s172_phase1_scaffolding.py`, Route-A suite.
- Beta ruling texts outside the repo; private repo; ser8 copies.
- Not read: `docs/CHAPTER_12_ADDENDUM_v1_3_0.md`, `docs/CHAPTER_12_ADDENDUM_PHASE1_v1_1_2.md`; `agents/step_runner/`; `apply_s*.py` patch corpus (may carry governance in docstrings — a B-item marked "not found in governance" was not checked against it).

---

## 9. Accurate claims and unverified claims (for the count)

**Accurate (24)**
1. Header files exist — `agents/watcher_agent.py`, `agents/watcher_registry_hooks.py`, `agents/fingerprint_registry.py`.
2. Monitor / evaluate / decide / act (§1.1) — `:913`, `:1317`, `:2717`.
3. Dispatch module (§1.2) — `agents/watcher_dispatch.py:761`, bound `agents/watcher_agent.py:3855`.
4. LLM lifecycle (§1.2) — `:879-888`.
5. Chapter 13 request integration (§1.2) — `:3189`.
6. Thresholds 0.70 / 0.50 (§2.1) — `:328-329`.
7. §2.2 file locations exist (incl. `agent_grammars/*.gbnf`, `watcher_history.json`, `watcher_decisions.jsonl`, `watcher_requests/`).
8. §3.3 action rule on the heuristic path — `:1283-1291`.
9. `STEP_SCRIPTS` hardcoded (§3.4.5) — `:1722`.
10. `.sh` PULL mechanism — `run_scorer_meta_optimizer.sh:229`, `:264-266`.
11. `.py` hardcodes `/shared/ml` (content) — `run_scorer_meta_optimizer.py:43-44`.
12. `PRNG_PRIORITY_ORDER` — `agents/watcher_registry_hooks.py:50-67` (44 names).
13. Hook method names and `post_run_record` signature — `:151`, `:252`, `:325`.
14. `AgentManifest` normalises outputs (§6.3) — `agents/manifest/agent_manifest.py:119`, `:187-189`.
15. Retry limits 3 / 10 (§8.2) — `:332-333`, `:1440-1450`.
16. Escalation logs, halts, notifies (§8.3 1–3) — `:1340-1350`, `:1459-1489`.
17. Precedence CLI > manifest > script (§3.6 order) — `:1811-1820`.
18. chunk_size fix in place — `full_scoring.json:122`, `run_step3_full_scoring.sh:70`.
19. `watcher_requests/` scan and routing (§11.1–11.3) — `watcher_dispatch.py:503-578`.
20. Selfplay only via WATCHER; WATCHER does not learn (§11.5 1–2) — dispatch functions only.
21. Seven of eight S114–S131 flags exist — `window_optimizer.py:1485-1508`.
22. All CLI params must be in `default_params` (invariant) — enforced `:1811-1820`.
23. `JOB_TIMEOUT_S = 600` — `persistent_worker_coordinator.py:154`.
24. Cited commits exist — `e184a8c`, `7a6a63c`, `888cf3e`, `478ff74`, `7e4ae02` (`git cat-file -t`).

**UNVERIFIED (5)**
- **U1** `test_watcher_agent.py` expected output (§9.2). Would verify: one run.
- **U2** §11.K pool=8 regression result — no record found. Would verify: a soak log.
- **U3** PWC "localhost `Semaphore(2)`" — only `Semaphore(self.max_per_node)`, default 8 (`persistent_worker_coordinator.py:264`, `:351`) found. Would verify: reading the localhost dispatch path in full.
- **U4** Halt-file presence on host — `/tmp` not reliably visible from the sandbox.
- **U5** "All 6 pipeline steps validated and working" (§11) — needs a run; on miner output Steps 2–6 are held (S2).

---

## 10. Addendum — main-session notes

Updates needed elsewhere, **not made** (scope limit):
- `docs/BACKLOG.md` §1 / repair list: B2–B6 are new; B1 already tracked.
- `docs/PROJECT_FILE_CATALOG.md` §7 gap 4/6: `definitions.json` role now described (§2 row 2 above; chapter §6.4).
- Skill / fact file: WATCHER is v2.0.0 and "WATCHER v1.8.0" means the manifest; `--dry-run` is a no-op on `--run-pipeline`; two halt files; G-PROD-SHAPE ran and failed 2026-08-22 (already in brief §0).
- `CLAUDE.md` §6: the Phase 1 row's "WATCHER v1.8.0" is the manifest version.
- Code-side stale text: `agent_manifests/window_optimizer.json:172`, `:215`, `:336`; `window_optimizer.py:1548-1551`.

Completion sentinel: **PASS** (declared scope: audit + in-place correction; nothing repaired).
