# Chapter 10 audit — Autonomous Agent Framework (agents/, llm_services/, grammars, manifests)

Repo: `/home/michael/distributed_prng_analysis` on VM 101, audited against `a15b524`
(`a15b5247336255801de8cf67d52653701b1e8c80`), working tree, with git history and host access (read-only).
Audited: `docs/CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md` (586 lines at `a15b524`, v3.1.0 of 2026-02-03).
Authority: `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md` REV1.
Method: read-only. Governance trail first, then fact file / `TODO_CONSOLIDATED.md` / behaviour model /
committed chapter audits, then code. Every anchor below was opened in this session. Nothing was executed:
no WATCHER, no agent, no pipeline step, no LLM server, no `--help`. Argparse and source were read. Two
static parses were run (`python3 -c` over manifest JSON; `python3 -I` regex over the registry source,
no import of `agents`). "Works" means the code path is complete on reading.

**Code moved during this pass.** `HEAD` was `b6cb789` at the start and `cf9edaa` by the end. Both commits
change only `agents/watcher_agent.py`, `agents/watcher_dispatch.py` and `tests/test_watcher_*`
(`git diff --name-only a15b524 HEAD`). So every anchor into those two files was taken from
`git show a15b524:<path>` and is marked "at a15b524". Every other file cited was checked with
`git diff --quiet a15b524 HEAD -- <file>` and `git diff --quiet HEAD -- <file>`, and is identical.

**Numbering:** EXECUTABLE (`docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1). Step 0 = TRSE, Step 1 = Window
Optimizer (the sieve runs inside it), Step 2 = Scorer Meta-Optimizer (`STEP_SCRIPTS` / `STEP_MANIFESTS` /
`STEP_NAMES`, `agents/watcher_agent.py:387-417` at a15b524). The chapter's "Step 2.5" label is the
conceptual name.

**Canonical version.** `_v3.md` is canonical. Evidence: `README.md:80` and `docs/PROJECT_FILE_CATALOG.md:290`
cite `_v3.md`; catalog `:970` records "`CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v2.md` → `_v3.md`"
(superseded) and `:652` lists `_v3.md.bak` as a stale duplicate ("Do not read; do not cite"); the brief
names `_v3.md` (`:142`); session changelogs S80, S99, S136 edit `_v3.md`. Git: `_v3.md` was created at
`2697f78` (2026-01-08) and edited in four later commits (last `2120b7d`, 2026-03-10). `_v2.md`
(1,553 lines, header v2.0.0 of 2026-01-07) has one commit, `323b6fa` ("Doc sync", 2026-02-09) — it
entered git **after** v3 and has never been edited. The `.bak` (512 lines, v3.0.0) is ignored by
`.gitignore:106` (`*.bak*`); its diff against `_v3.md` is exactly the v3.0.0→v3.1.0 header and table
changes. Neither other version was edited.

Framing held throughout: TFM is functional mimicry of surface output; no finding here assumes a "true seed".

---

## 1. Verdict

**Null results first.** On reading, the step-evaluation path through the LLM — this chapter's main
subject — is **not reached for any step at a15b524**: all seven manifests declare
`evaluation_type: file_exists` (manifest parse, below), so `evaluate_results` returns before the
router, the doctrine or the heuristic (`agents/watcher_agent.py:946-1004` at a15b524). This is Chapter 12's
D3/B1 and is counted here as absorbed. The LLM server was **not running** when checked
(`curl -m 2 localhost:8080/health` → rc 7, 2026-11-11). No claim of "working" in this chapter was
confirmed by a run.

| Category | Count |
|---|---|
| Accurate | **28** |
| BROKEN (new) | **1** |
| SWITCHED-OFF-BY-RULING | **1** |
| DESCRIPTION-ONLY | **24** |
| UNVERIFIED | **3** |
| Absorbed into an existing item (not counted as new) | **9** |
| **Total claims checked** | **66** |

28 + 1 + 1 + 24 + 3 + 9 = **66**. Reconciliation table in §10.

Chapter 586 → 1,065 lines; `git diff --numstat` shows 479 added, 0 deleted; every pre-existing heading is present.

Three things matter more than the rest:

1. **The LLM evaluation path the chapter documents is dormant.** File-existence decides every step
   (absorbed: Ch12 D3, B1 — B1 repaired at `b6cb789`, which makes an explicit step failure beat
   file existence; the LLM path is still not reached).
2. **There are two grammar directories, and the chapter names the wrong one for WATCHER.** WATCHER's
   router loads `grammars/watcher_decision.gbnf`, CWD-relative (`llm_services/llm_router.py:235`, `:252`,
   `:410-413`); `grammar_loader.py` serves `agent_grammars/` (`:41-54`). The chapter shows
   `agent_grammars/agent_decision.gbnf` with fields neither file has (D16).
3. **The parameter registry that feeds the LLM's "adjustable parameters" is stale** (B1, new): 20 of its
   29 CLI flags do not exist in the scripts, and two of its six script keys name files that never
   existed. Dormant while (1) holds.

---

## 2. The §5 host questions and pre-set questions

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | `agent_manifests/definitions.json` (gitignored) | Present, 6,635 B, mtime 2026-01-01; ignored by `.gitignore:41` (`*.json`); never tracked (`git log --all --` → 0 commits). Schema 1.1.0, `updated_at` 2026-01-02. Keys: `pipeline_steps` (Step 5 `emits` `signal_quality`, `data_context`, `feature_schema`; Step 6 `consumes` them), `watcher_protocol` (degenerate-fingerprint registry at `agent_state/degenerate_fingerprints.json`, a 5-step `recovery_workflow`, fingerprint = sha256 truncated to 8 of `{train_start}:{train_end}|{holdout_start}:{holdout_end}|{survivor_count}|{prng_type}|{mod}`), `sidecar_schema`. **No code reads it** (`/bin/grep -rln definitions\.json --include=*.py --include=*.sh .` → none outside backups); `agent_state/` is absent on host; `degenerate_fingerprint` occurs in no `*.py`. The producer half exists: `fingerprint_hash` appears in `meta_prediction_optimizer_anti_overfit.py` (9 hits) and `prediction_generator.py` (1). The loop-prevention consumer does not. Same finding as `docs/CHAPTER_12_AUDIT.md` §2 row 2; recorded in the chapter, not counted as a new item. | `ls -la`, `python3 -c json`, `git check-ignore -v`, `git log --all`, `/bin/grep -rln` |
| 2 | Which Chapter 10 is canonical? | `_v3.md` (header of this audit). | as above |
| 3 | Numbering | EXECUTABLE, stated in the chapter header. `agent_contexts/context_loader.py:15-23` uses the conceptual scheme (2 = sieve, 2.5 = scorer) and has no importer. | `/bin/grep -rln context_loader` |
| 4 | LLM host state (read-only) | Server not running (rc 7); no systemd unit matching llama/llm/watcher/tfm; no crontab for `michael`. Model `models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf` present (8,988,110,240 B, 2026-01-06); `/home/michael/llama.cpp/build/bin/llama-server` present; `claude` CLI at `~/.local/bin/claude`. `nvidia-smi` shows **one** RTX 3080 Ti (12,288 MiB) on VM 101, while `start_llm_servers.sh:9`, `:100` and `llm_server_config.json:10` assume a dual-3080-Ti partition (point-in-time; recorded, not a label). `/tmp/llm_server_primary.log` absent as seen from this session (`/tmp` → UNAVAILABLE). | `curl`, `systemctl list-units --all`, `crontab -l`, `ls`, `nvidia-smi` |
| 5 | Ch5 note: `parameter_registry.py:437-478` | Confirmed and widened → B1 (§4). | §4 |

---

## 3. What actually works end to end today

Complete on reading; none was run.

1. **Router.** `LLMRouter.route` → `_call_primary` (llama.cpp `/completion`, ChatML, stop tokens from config)
   → escalation-trigger check → `_call_backup` (`subprocess.run(['claude','--print','-p',prompt])`)
   (`llm_services/llm_router.py:112-215`). A failed primary call also falls to the backup (`:170-178`).
2. **Grammar-constrained WATCHER decision.** `evaluate_watcher_decision` loads
   `grammars/watcher_decision.gbnf` and falls back to the backup + `_parse_json_response` (default
   "escalate" on parse failure, `:358-363`) (`:375-430`). Reached only when a manifest is not
   `file_exists` (none is).
3. **LLM lifecycle.** `llm_lifecycle.py` v1.0.0 stops the server for GPU-heavy dispatch work and
   restarts it for evaluation: `agents/watcher_dispatch.py:131-133`, `:202-204`, `:309-332` (at a15b524);
   WATCHER uses `session()` for Step-5 diagnostics analysis (`agents/watcher_agent.py:2405-2420` at a15b524).
4. **Context assembly for dispatch.** `bundle_factory.build_llm_context` (8 bundle types incl. 13 and 99,
   `agents/contexts/bundle_factory.py:587-596`), Tier-0 `MAIN_MISSION` (`:76`, `:756`); used by dispatch
   only (`agents/watcher_dispatch.py:75` at a15b524).
5. **Policy routing for Chapter 13 approvals.** `watcher_policies.json` (`approval_route: "orchestrator"`,
   `test_mode: false`, `auto_approve_in_test_mode: false`) is read at `chapter_13_orchestrator.py:403-425`
   and `agents/watcher_agent.py:3314-3326` (at a15b524). With the live values every accepted proposal goes
   to `request_approval` (a human approval request).
6. **TRSE skip policy.** `trse.json` `skip_on_fail: true` → a missing Step-0 output is recorded as
   `proceed` (`agents/watcher_agent.py:966-987` at a15b524). Kept by the `b6cb789` repair (R1-E).

---

## 4. BROKEN

**B1 — The parameter registry advertises parameters the scripts do not accept.** *(not found in governance,
except the `offset` entry, which is S1)*
- Chapter: §2.2 lists `parameters/` ("Parameter context"); "Parameter Bounds Authority" names two
  bounds sources. The registry is the third: `agents/registry/parameter_registry.py:1-9` ("Enables LLM
  introspection of legal parameter bounds"), consumed by `ParameterContext`
  (`agents/parameters/parameter_context.py:181`), which feeds `FullAgentContext` and the prompt.
- Code: static comparison of the 29 `cli_flag=` entries (`PARAMETER_REGISTRIES`, `:263-600`) against each
  script's `add_argument` calls: **20 of 29 flags are absent**. `window_optimizer.py`: `--window-size`,
  `--offset`, `--skip-min`, `--skip-max`. `scorer_meta_optimizer.py` (5 flags) and `full_scoring.py`
  (2 flags): **the scripts do not exist and never did** (`git log --all --` empty). `adaptive_meta_optimizer.py`:
  `--trials`, `--min-layers`, `--max-layers`, and `--mode` choices `quick/full/deep` versus the real
  `full/survivor/network/epochs` (`:437-478` vs `adaptive_meta_optimizer.py:1101-1116`; description "Optimize
  ML model architecture using Optuna"). `meta_prediction_optimizer_anti_overfit.py`: `--epochs`,
  `--dropout-min`, `--dropout-max`, `--distributed`. `prediction_generator.py`: `--confidence-threshold`,
  `--diversity-weight`.
- Effective reach: `ParameterContext` iterates the manifest's primary-action `args_map` and only looks
  specs up in the registry (`parameter_context.py:177-195`). Of the stale entries, only the Step-4
  `--mode` spec is matched today (`ml_meta.json` primary action `args_map` has `--mode`). Dormant: the
  context is built only on the non-`file_exists` path (§1), and RETRY adjustments are not applied
  (Ch12 B5).
- Stops at: `agents/registry/parameter_registry.py:437`.
- Repair must: make every registry entry match the script it names (or drop entries for scripts that do
  not exist), so the "legal bounds" the LLM is shown are real. Owner choice; no code changed here.

---

## 5. SWITCHED-OFF-BY-RULING

**S1 — The registry's Step-1 `offset` entry is not yet `window_anchor`.** `agents/registry/parameter_registry.py:282-289`
(`offset`, `max_value=500`). Design: `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` §4.6
(`:156-161`: replaced by `window_anchor`, `min_value=0`, no static max, delegated to the resolver;
`generator_phase` not registered). Held by: Brief I firewall, `docs/S172_WINDOW_ANCHOR_BRIEF_I.md:360-361`
("v1.1 §4.6 assigns this to Brief II … Leave it. Record it.") and `:675`; restated in
`docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md:708`. Lifting it (Brief II) would let the registry describe the
anchor with a dynamic maximum that cannot widen the derived domain. Today the entry is not reached:
`--offset` is not in `window_optimizer.json`'s `args_map` and not a flag of `window_optimizer.py`.

Per §0 of the authority brief this is a review position the owner may keep or lift. No recommendation.

---

## 6. DESCRIPTION-ONLY

| # | Claim | Chapter | Code at `a15b524` |
|---|---|---|---|
| D1 | C1 | Header v3.1.0 "Full Autonomous Operation — Phase 7 Complete" | the old (Feb 2026) Phase 7 (`docs/PROJECT_FILE_CATALOG.md:290`), not S172 Phase 7 (soak unblocked, not run — Ch12 §7) |
| D2 | C4 | Watcher v1.4.0 | v2.0.0 (`agents/watcher_agent.py:12`, `:3467` at a15b524) |
| D3 | C6 | Bundle factory v1.0.0 | v1.1.0 (`agents/contexts/bundle_factory.py:5`, `:432`) |
| D4 | C8 | Router v2.0.0 | v2.1.0 (`llm_services/llm_router.py:3`) |
| D5 | C9 | Grammar loader v1.0.0 | v1.1.0 (`llm_services/grammar_loader.py:5`) |
| D6 | C14 | `agent_grammars/*.gbnf` 4 files | 5 files (+ `chapter_13.gbnf`); and a second tracked dir `grammars/` (8 files incl. `watcher_decision.gbnf`) which the router and the Ch13 advisor actually load (`llm_router.py:252`, `chapter_13_llm_advisor.py:309-312`; cf. `docs/CHAPTER_13_AUDIT.md:262`) |
| D7 | C18 | §2.2 tree | `watcher_agent.py # (v1.1.0)` is v2.0.0; omits `watcher_dispatch.py`, `watcher_registry_hooks.py`, `registry_inspector.py`, `threshold_guardrail.py`, `progress_display.py`, `registry/`, `prompts/`, `pipeline/`, `contexts/{bundle_factory,advisor_bundle}.py`; 6 manifests → 7 incl. `trse.json` (Ch12 D5); "Step 2.5" → executable Step 2 (Ch12 D14) |
| D8 | C20 | `step_runner/ # Step execution` | package exists (v1.0.0, `agents/step_runner/__init__.py:1-30`) with **no importer** outside itself (`/bin/grep -rln agents.step_runner`); WATCHER executes via `run_step` (`agents/watcher_agent.py:1705` at a15b524). Its "separate dispatch surface is not part of the certified Run-4 route" (`docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md:133`; `docs/BACKLOG.md:590-592`) |
| D9 | C23 | config snippet schema 2.0.0, `context_length` 8192 | 2.1.0, 32768, + `stop_tokens`, `working_dir`, `request_timeout_seconds: 180` (`llm_services/llm_server_config.json:2-28`) |
| D10 | C24 | startup prints "Primary … HEALTHY" | prints `Waiting for health check... ready! (Ns)` (`start_llm_servers.sh:69-83`); the script hardcodes `PORT`/`CTX_SIZE`/`N_GPU_LAYERS` (`:28-30`) and never reads `CONFIG_FILE` (`:20`, only occurrence) |
| D11 | C28 | §4.1 option list | 10 further flags: `--params`, `--force-step`, `--episodes`, `--dispatch-selfplay`, `--dispatch-learning-loop`, `--process-requests`, `--dry-run`, `--stop`, `--explain`, `--approve` (`:3530-3565` at a15b524); `--halt` takes an optional reason; `--start-step` accepts 0 |
| D12 | C30 | §4.3 router → HTTP → heuristic | `_evaluate_with_llm` tries the Step-1 new prompt first (`:1047-1058`), then `_evaluate_with_router`, then `_evaluate_with_http` (`:1059-1068`); the heuristic fallback is in `evaluate_results` (`:1027-1037`), all after the `file_exists` short-circuit (Ch12 D3) |
| D13 | C34 | §5.2 key metrics | contexts read: scorer `best_validation_score`, `total_trials`, `best_k_folds`; full scoring `survivors_scored`, `completion_rate`; ml_meta `architecture_score`, `validation_loss`; anti-overfit `overfit_ratio`, `kfold_mean/std`; prediction `pool_size`, `coverage_pct`, `mean_confidence` (`agents/contexts/*_context.py`, `.get(` keys). No `best_accuracy`, `best_r2`, `test_r2`, `signal_quality`, `pool_coverage` |
| D14 | C36 | §6.1 doctrine snippet | conditions abbreviated/reworded (`agents/doctrine.py:32-59`: retry adds "no safety violations"; escalate is "anomalies detected", "retries exhausted", "safety concerns"); the dict also carries calibration, safety, adjustment and output sections (`:61-95`) |
| D15 | C38 | §7.2 "referenced but needs creation" | `agent_grammars/` exists, tracked, 5 files |
| D16 | C39 | §7.3 grammar `decision`/`confidence`/`reasoning` | `agent_grammars/agent_decision.gbnf:1-14` has `success_condition_met`, `confidence`, `reasoning`, `recommended_action`, number `[0-9] "." [0-9] [0-9]`; WATCHER's router uses `grammars/watcher_decision.gbnf` (`decision`, `retry_reason`, `confidence`, `reasoning`, `primary_signal`, `suggested_params`, `warnings`, `checks`) |
| D17 | C45 | Gap "`agent_grammars/` missing" | closed |
| D18 | C46 | Gap "on-demand LLM lifecycle" | closed: `llm_services/llm_lifecycle.py` v1.0.0 (`:1-28`), wired in dispatch and Step-5 diagnostics (§3 item 3) |
| D19 | C47 | Gap "MAIN_MISSION in doctrine.py" | closed elsewhere: `MAIN_MISSION` in `bundle_factory.py:76`, rendered `:756`; WATCHER's Step-1 prompt uses `GLOBAL_MISSION` (`agents/prompts/global_mission.py:6`, `agents/prompts/step1_prompt.py:14`, `:49`) |
| D20 | C48 | §11 version history | no 3.1.0 entry (header is 3.1.0) |
| D21 | C53 | bounds "exclusively" in `distributed_config.json` | `load_search_bounds_from_config` merges `search_bounds` over a hardcoded defaults dict (`window_optimizer.py:67-92`); the `offset` default's retirement is Brief II item C-4 (`docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md:700-704`) |
| D22 | C61 | "Five agent files had `Field(ge=1)` … all five fixed" | `b2a89d8` sets `ge=0` at four fields in three files: `agents/full_agent_context.py:42`, `agents/manifest/agent_manifest.py:154-158`, `agents/pipeline/pipeline_step_context.py:28`, `:167`; plus Step-0 entries in `agents/contexts/__init__.py:31`, `:42` and `progress_display.py`. `agents/contexts/base_agent_context.py:53` stays `ge=1` (Step 0 reuses `WindowOptimizerContext`, `pipeline_step = 1`, `window_optimizer_context.py:44`) |
| D23 | C64 | threshold "on Steps 1 and 3" | Step 1's `success_condition` is `optimal_window_config.json` only (manifest parse), so the survivor-array minimum (`:169-172`) is reached by Step 3's `survivors_with_scores.json`, not by Step 1 |
| D24 | C65 | TRSE CLI example `--params '{… "trials":200 …}'` | `trials` is not in `window_optimizer.json` `default_params` and is dropped by the step-scoped merge (`agents/watcher_agent.py:1811-1820` at a15b524; Ch12 D20); the declared key is `window_trials` |

---

## 7. Chapter-specific facts recorded in the chapter

- **RANGE-MINER.** The agent framework does not choose a sieve backend; WATCHER routes Step 1 by manifest
  parameter (Ch12 §13, S1: `use_range_miner` default `false`). Nothing in `llm_services/`, `agents/contexts/`,
  the doctrine or the registry names RANGE-MINER (`/bin/grep -rli range_miner` over those paths → none).
- **22-array NPZ contract.** No agent-framework file reads the NPZ; Step 1's verdict is the existence of
  `optimal_window_config.json`.
- **Window-anchor change (Brief I).** Touches this framework only at the registry `offset` entry, deferred
  to Brief II (S1). Brief I is implemented; final acceptance is pending a completed production-shape run
  (`docs/S172_BRIEF_I_PRODUCTION_SHAPE_CLASSIFICATION_REPORT.md`).
- **WATCHER repairs after a15b524** (`b6cb789`, report `docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md`
  §2): R-1 explicit step failure escalates before the file-exists verdict (Ch12 B1); R-2 `--dry-run` with
  `--run-pipeline`/`--approve`/`--daemon` refuses with exit 2 (Ch12 B6); R-3 dispatch checks both
  `/tmp/agent_halt` and `watcher_halt.flag` through one helper (Ch12 B3). `cf9edaa` followed (seed-domain-
  complete is a clean stop; learning loop stops on a failed step). Not re-audited here.
- **Doctrine is prompt content, not an enforced gate.** `validate_decision_against_doctrine` is imported by
  WATCHER (`agents/watcher_agent.py:151` at a15b524) and never called there; only `test_subphase4.py` calls it.
  The doctrine reaches the LLM through `full_agent_context.py:155`, `:219`. (Observation; the chapter does not
  claim enforcement, so no label.)
- **Mission text in code.** `agents/prompts/global_mission.py:6` says the system "reverse-engineers PRNG
  behavior by testing candidate seeds"; `bundle_factory.py:76-87` `MAIN_MISSION` says functional mimicry,
  "rather than attempting to discover or reconstruct actual seeds". The Step-1 WATCHER prompt uses the first.
  Code-side text, listed for the owner, not edited.
- **Unwired agent-framework components** (exist, no importer found): `agents/step_runner/` (D8),
  `agent_contexts/context_loader.py` and its seven `.md` templates, `agent_manifests/definitions.json`
  (§2 row 1). Fingerprint registry and hooks: Ch12 B2 (only importer `agents/registry_inspector.py:30`; DB
  `agents/data/fingerprint_registry.db` mtime 2026-01-02 18:25).
- **TRSE (the chapter's Step-0 appendix).** Two of its claims are already governed by
  `docs/TRSE_STEP0_AUDIT_v1.md`: F1 (WATCHER's Step-0 command is malformed — still true at a15b524:
  `trse.json` `default_params` carries `window_size`, `stride`, `trse_context`; `trse_step0.py:811-816`
  defines none of them) and F3 (`confirmed_windows` is write-only — writer `window_optimizer.py:1055-1084`;
  zero readers in `trse_step0.py`). Proposed fix: `docs/TEAM_ALPHA_TRSE_FIX_PROPOSAL.md` (not implemented).
  `trse_step0.py` says `Version: 1.15.0` in its docstring (`:5`) and `__version__ = "1.15.1"` (`:107`).
- **Approval authority.** The chapter's policy table says `approval_route` decides "who owns execution
  authority". What the code enforces: `approval_route` selects the orchestrator path or the WATCHER-daemon
  path for an accepted Chapter-13 proposal; with `test_mode` false, both end in `request_approval`
  (`chapter_13_orchestrator.py:410-425`). The code does not encode any review body.

---

## 8. Surfaces searched, and surfaces unavailable

**Files opened this session (execution proof)**
- Briefs / governance / audits: `scratchpad/common_brief.md`; `docs/CLAUDE_CODE_INSTRUCTIONS_CHAPTER_CORRECTIONS_20261111.md`
  §0–§5, VIR; `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (header, headings); `docs/CHAPTER_13_AUDIT.md` (headings);
  `docs/CHAPTER_12_AUDIT.md` (full); `docs/CHAPTER_12_WATCHER_AGENT.md` (header, §0, §16–§17, headings);
  `docs/CHAPTER_5_AUDIT.md` (grep `:194`, `:272`); `docs/PIPELINE_BEHAVIOUR_MODEL.md` §1.1; `docs/PROJECT_FILE_CATALOG.md`
  `:286-292`, `:648-654`, `:734-742`, `:966-972`; `README.md:78-82`; `docs/BACKLOG.md:584-594`;
  `docs/TODO_CONSOLIDATED.md` (grep TRSE, Ch10); `docs/TRSE_STEP0_AUDIT_v1.md` (head, F1–F3);
  `docs/TEAM_ALPHA_TRSE_FIX_PROPOSAL.md` (head, grep); `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md:150-170`;
  `docs/S172_WINDOW_ANCHOR_BRIEF_I.md:355-364`, `:670-680`; `docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md:700-712`;
  `docs/TB_RULING_RUN4_ROUTING_PATCH_REVIEW_R2_APPROVED.md` (grep `:21`, `:133`);
  `docs/PROPOSAL_Modular_Step_Automation_Framework_v1_1.md:1-12`; `docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md:1-172`;
  `docs/CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v2.md:1-12`; `_v3.md.bak` (head, diff); `~/.claude/skills/lead-handling/SKILL.md` (head).
- Code at a15b524 (`git show`): `agents/watcher_agent.py` `:1-90`, `:151`, `:155-310`, `:320-420`, `:840-1070`,
  `:1071-1110`, `:1269-1312`, `:1800-1830`, `:2405-2420`, `:3310-3330`, `:3466-3580`, `:3770-3810`;
  `agents/watcher_dispatch.py` (grep bundle/lifecycle/grammar, `:618-660`).
- Code (identical at a15b524 and HEAD): `agents/doctrine.py` (full); `agents/full_agent_context.py:1-60` + grep;
  `agents/contexts/__init__.py` (grep), `window_optimizer_context.py:1-45`, `bundle_factory.py:1-10`, `:70-92`,
  `:580-600`, grep; the five other `*_context.py` (key extraction); `agents/contexts/base_agent_context.py:53`;
  `agents/manifest/agent_manifest.py:68-80`, `:150-160`, `:355-400`; `agents/pipeline/pipeline_step_context.py` (grep);
  `agents/registry/parameter_registry.py:1-40`, `:263-292`, `:425-485`, `:600-625`, defs; `agents/parameters/parameter_context.py:170-200`;
  `agents/step_runner/__init__.py:1-30`; `agents/prompts/global_mission.py`; `agents/prompts/step1_prompt.py` (grep);
  `agents/registry_inspector.py:28-31`; `llm_services/llm_router.py:1-30`, `:108-260`, `:375-498`, defs;
  `llm_services/grammar_loader.py:1-75`; `llm_services/llm_lifecycle.py:1-30`; `llm_services/llm_server_config.json`;
  `llm_services/start_llm_servers.sh`; `agent_grammars/agent_decision.gbnf`; `grammars/watcher_decision.gbnf` (a15b524);
  `agent_contexts/context_loader.py:1-25`; `window_optimizer.py:60-100`, `:1050-1086`, argparse grep;
  `adaptive_meta_optimizer.py:1100-1118`; `trse_step0.py` (header, `:107`, grep, argparse `:811-816`);
  `chapter_13_orchestrator.py:398-425`; `chapter_13_llm_advisor.py` (grep).
- Manifests: all eight `agent_manifests/*.json` parsed (`version`, `pipeline_step`, `evaluation_type`,
  `success_condition`, primary-action `script`/`args_map`, `default_params`); `definitions.json` read;
  `watcher_policies.json` parsed.
- Git: `git log` for `_v2.md`, `_v3.md`, `agents/step_runner`, `agent_contexts`, `agents/data`, `grammars/`,
  `agent_manifests/trse.json`, `-S evaluation_type` on `window_optimizer.json` (`6326257`, 2026-01-11), `-S ge=0`
  (`b2a89d8`, inspected); `git diff --name-only a15b524 HEAD`; `git diff --quiet` per cited file.
- Host: `ls models/`, `llama-server`, `command -v claude`, `nvidia-smi`, `curl …/health`, `systemctl list-units --all`,
  `crontab -l`, `ls agents/data/`, `ls agent_state`.

**Search method for absence claims:** `/bin/grep -rln` over the working tree including untracked and
gitignored files (excluding `backups/` where stated), with `--include=*.py --include=*.sh` where stated;
`git log --all -- <path>` and `git log -S` for history; `git ls-files -o -i --exclude-standard` for
ignored files under `agents/`, `llm_services/`, `agent_manifests/`, `agent_grammars/`, `agent_contexts/`.

**Governance trail searched:** `docs/TB_RULING_*`, `docs/PROPOSAL_*`, `docs/TEAM_ALPHA_*` by grep for
`parameter_registry`, `step_runner`, `TRSE`/`confirmed_windows`, `grammar`; hits read at the line.

**Chapters searched:** Ch2 (structure only), Ch12 (chapter header + audit), Ch13 audit, Ch14 audit (grep),
Ch5 audit (grep), Ch1 audit / chapter (grep for `step_runner`, `parameter_registry`, `confirmed_windows`).

**Unavailable**
- `/tmp` from the agent sandbox (halt files, server log).
- Any result that needs a run: live LLM evaluation, throughput (tok/s), `test_subphase*.py`,
  `agents/step_runner/test_step_runner.py`.
- Rigs / CT100; ser8; the private repo; Beta texts outside the repo.
- Not read: `docs/WATCHER_POLICIES_REFERENCE.md` and `docs/THRESHOLD_GOVERNANCE.md` beyond existence;
  `docs/PROPOSAL_Unified_Agent_Context_Framework_v3_2_*`; `docs/PROPOSAL_LLM_Infrastructure_Optimization_v1_1.md`;
  the `apply_s*.py` patch corpus (may hold governance in docstrings; B1 was not checked against it).

---

## 9. Accurate, unverified and absorbed claims (for the count)

**Accurate (28)**
1. C5 dispatch v1.0.0 — `agents/watcher_dispatch.py:5`.
2. C7 lifecycle v1.0.0 — `llm_services/llm_lifecycle.py:5`.
3. C10 start script v2.1.0 — `start_llm_servers.sh:4`.
4. C11 six step contexts + Ch13 — six `*_context.py`; bundle step 13 `bundle_factory.py:595`.
5. C12 doctrine v3.2.0 — `agents/doctrine.py:8` (`DOCTRINE_VERSION = "1.0.0"`, `:15`, as the snippet shows).
6. C13 prompt builder v3.2.0 — `agents/prompt_builder.py:8`.
7. C15 primary DeepSeek-R1-14B / backup Claude — `llm_server_config.json:4-20`.
8. C17 §2.2 listed files exist — `ls`, `git ls-files`.
9. C22 port 8080 llama.cpp; backup via Claude Code CLI — config `:7`; `llm_router.py:186-187`.
10. C25 stop command matches the server's command line — `start_llm_servers.sh:55-58` (`--port 8080`).
11. C26 `route()` logic — `llm_router.py:112-136`.
12. C27 the eleven listed CLI flags exist as described — `agents/watcher_agent.py:3470-3529` at a15b524.
13. C29 the four modes — `:3782` (`--evaluate`), `--run-pipeline`, `--daemon`, `--no-llm` → `use_llm=False` (`:3570`).
14. C31 PROCEED ≥ 0.70, ESCALATE < 0.50 — `:328-329`, `:1283-1291` at a15b524.
15. C33 "LLM does the reasoning, not this file" — `window_optimizer_context.py:8`.
16. C35 `WindowOptimizerContext` fields — `:27`, `:43-45`.
17. C37 `GrammarType` enum — `grammar_loader.py:29-38`.
18. C41 §8.3/§8.4 command syntax valid — argparse at a15b524.
19. C42 `PYTHONPATH=.` needed — no root insertion into `sys.path` before `from agents…` at `:75` (a15b524); the only insertion is inside a function (`:1965`).
20. C43 health endpoint `/health` — `start_llm_servers.sh:44`, `:72`.
21. C50 policy flags exist and are read — `watcher_policies.json`; `chapter_13_orchestrator.py:403-425`; `agents/watcher_agent.py:3314-3326` at a15b524.
22. C51 `docs/WATCHER_POLICIES_REFERENCE.md` exists.
23. C52 manifests are not runtime bounds on the WATCHER path — `parameter_bounds` is read only by `agents/step_runner/command_builder.py:151` (no importer).
24. C54 `baselines/baseline_window_thresholds.json`, `docs/THRESHOLD_GOVERNANCE.md` exist.
25. C57 TRSE v1.15.1 — `trse_step0.py:107`.
26. C58 `skip_on_fail` never halts — `trse.json`; `:966-987` at a15b524; kept by `b6cb789` (R1-E).
27. C60 relative normalization — `trse_step0.py:107`, `:422-455`.
28. C63 survivor minimum 100 → 50 — `agents/watcher_agent.py:169-172` at a15b524.

**UNVERIFIED (3)**
- **U1** C2 "Autonomy ~85%" — no metric in code or docs defines it. Would verify: a definition and a measurement.
- **U2** C3 "fully implemented and working" — needs a run with a non-`file_exists` manifest and a live server.
- **U3** C21 51 tok/s / 38 tok/s — only in docstrings (`llm_router.py:9-10`). Would verify: a benchmark log.

**Absorbed (9, not counted as new)**
- C3b §1.2 live test (Step 1 via LLM, `llm_http_extracted`) → Ch12 D3: Step 1 became `file_exists` at
  `6326257` (2026-01-11), three days after the test.
- C16 diagram "step completes → WATCHER → LLM router" → Ch12 D3.
- C19 `fingerprint_registry.py` "Dataset+PRNG tracking" → Ch12 B2.
- C32 RETRY "with adjustments" → Ch12 B5.
- C40 §8.1–§8.2 expected outputs (conf 0.95, heuristic) → Ch12 D3/D4 (live path gives `file_exists`, conf 1.00/0.00).
- C44 §9.3 fallback list → D12 (this audit).
- C49 §12 "7 bundle types" → Ch12 D2 (8).
- C59 `confirmed_windows` "TRSE uses these" → `docs/TRSE_STEP0_AUDIT_v1.md` F3.
- C62 "TRSE can be dispatched through WATCHER" → `docs/TRSE_STEP0_AUDIT_v1.md` F1 (the Pydantic part is accurate; the command still fails at a15b524).

---

## 10. Count reconciliation

| Category | Items | Count |
|---|---|---|
| Accurate | §9 list 1–28 | 28 |
| BROKEN (new) | B1 (C55) | 1 |
| SWITCHED-OFF-BY-RULING | S1 (C56) | 1 |
| DESCRIPTION-ONLY | D1–D24 (C1, C4, C6, C8, C9, C14, C18, C20, C23, C24, C28, C30, C34, C36, C38, C39, C45, C46, C47, C48, C53, C61, C64, C65) | 24 |
| UNVERIFIED | U1–U3 (C2, C3, C21) | 3 |
| Absorbed into an existing item | C3b, C16, C19, C32, C40, C44, C49, C59, C62 | 9 |
| **Total** | C1–C65 + C3b | **66** |

28 + 1 + 1 + 24 + 3 + 9 = 66. No UNVERIFIED item from an earlier audit was resolved here.

---

## 11. Addendum — main-session notes

Updates needed elsewhere, **not made** (scope limit):
- `docs/BACKLOG.md` §1 / repair list: B1 (parameter registry) is new; TRSE F1/F3 still open at a15b524.
- `docs/PROJECT_FILE_CATALOG.md` §7 gap 6: Chapter 10 now audited; `:290` row can name this audit;
  `grammars/` vs `agent_grammars/` split worth a catalog line.
- Skill / fact file: two grammar directories; doctrine validator not called by WATCHER; the step-eval LLM
  path is dormant under `file_exists`.
- Code-side stale text (not edited): `agents/prompts/global_mission.py:6` seed-recovery phrasing;
  `trse_step0.py:5` version; `start_llm_servers.sh` comments assume dual 3080 Ti; `llm_router.py:244`
  docstring says "grammars/ directory" (true) while `grammar_loader.py:7` says "Chapter: 10 §7" for
  `agent_grammars/`.

Completion sentinel: **PASS** (declared scope: audit + in-place correction; nothing repaired).
