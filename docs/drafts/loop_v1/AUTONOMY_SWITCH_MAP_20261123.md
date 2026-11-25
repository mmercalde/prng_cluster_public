# How TFM is meant to know, reach, propose, apply and prove a switch — map, check, patch, gaps

TASK_SWITCH_REGISTER (lead, 18:50) as changed by TASK_SWITCH_REGISTER_ADDENDUM_1 (19:00). Owner, 18:46: "Who is going to keep track of all these switches and how will the code turn them on and off?" Owner, 18:50: "autonomy is the goal … We also have everything coded for pedantic [Pydantic] vocabulary, agent manifest etc".

**Plain summary.** The machinery the owner means exists, and the switches belong in it. Nothing in this work adds a second register.

- **What the system has:**
  - A Pydantic parameter registry (`ParameterSpec`): what a switch is and its legal bounds.
  - The agent manifests (`default_params`, `args_map`): which switches WATCHER can pass.
  - GBNF grammars and Pydantic proposal schemas: what an LLM may propose.
  - Policy bounds and acceptance rules: what is allowed.
  - A kill switch.
- **How much of it works:**
  - Only the know and reach links work today, and only for **8 of 183** step-script switches.
  - Chain A is the one end-to-end working route. The Strategy Advisor changes two selfplay settings.
  - There is no general apply step and no general proof of effect.
- **What I built:**
  - A checker, `scripts/diagnostics/switch_check.py`. It classes every switch found in code against that system and writes the exact `ParameterSpec` and manifest entry that would register it.
  - A draft patch that registers today's 281 new switches in the existing format, with defaults unchanged. Nothing is applied.
  - The remaining gaps are listed in §4 as to-dos, with their recorded IDs.

Results, all at main `41ad509`:

| Measure | Result |
|---|---|
| Switch rows | 2,141 |
| Step-script switches: registered and reachable | 8 |
| Step-script switches: registered but unreachable | 1 |
| Step-script switches: not registered | 174 |
| Stale `ParameterSpec`s | 20 of 29, the same count as CHAPTER_10_AUDIT B1, found independently |
| Draft patch | 275 specs for 281 new switches; the 29 existing specs unchanged; imports under Pydantic |

## Prior record

**Searches run** (main tree `~/distributed_prng_analysis` at `41ad509`):
- `git grep -n -iE` over `docs/` (which includes `docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`, `docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md`, `docs/exchange/`), for:
  - `switch register|parameter[_ ]registry|parameter vocabulary|chapter_13_parameter_vocabulary|setting_proposal`
  - `executable[- ]interface|allowed_params|ruled CORRECT`
  - `pool_size.{0,40}--k`
  - `ignoring unknown option|full_scoring.json…ignored`
  - `parameter_registry\.json…(untracked|gitignor)`
  - `proposed_value…number`
  - `agent_halt…(reboot|cleared)`
  - `file_exists`
- `git grep` over `*.py` for:
  - `ParameterContext\.(build|from_results)`
  - `record_applied_changes\(`
  - `parameter_application`
  - `threshold_provenance`
  - `advisor.analyze`

**Read in full, as the addendum asks:**
- `agents/registry/parameter_registry.py`
- `agents/manifest/agent_manifest.py`
- `agents/parameters/parameter_context.py`
- `agents/registry_inspector.py`: it covers the fingerprint registry, not parameters.
- `agents/watcher_registry_hooks.py`: fingerprint pre- and post-run hooks.
- `agents/safety/kill_switch.py`
- the 7 `agent_manifests/*.json`
- `grammars/parameter_adjustment.gbnf`, `watcher_decision.gbnf`, `chapter_13.gbnf`, `strategy_advisor.gbnf`
- `watcher_policies.json`
- `docs/TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md`
- `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md` Part B
- `docs/D6_THRESHOLD_AUTONOMY_SIGNPOSTS.md`
- `config_manifests/`

`docs/CHAPTER_10_AUTONOMOUS_AGENT_FRAMEWORK_v3.md`, `docs/CHAPTER_12_WATCHER_AGENT.md` and `docs/PROPOSAL_EPISTEMIC_AUTONOMY_UNIFIED_v1_3.md` were read by a read-only search agent against a narrow brief. I spot-checked the anchors I use: Ch12 `:1046-1049` and `:1232`; Ch10 `:812-819` and `:915-918`.

**Found** (decides, authorises or records the subject):

| Subject | Where |
|---|---|
| The registry is stale | `CHAPTER_10_AUDIT.md` B1 (`:117-140`) |
| The registry is what the LLM is shown as legal bounds | Ch10 `:818-819`, `:915-918` |
| Manifest `parameter_bounds` are read only by unimported `agents/step_runner/command_builder.py` | Ch10 `:812-816` (D21) |
| Reach rule: "all new CLI params MUST appear in manifest default_params" | Ch12 `:1232`; step-scoped merge Ch12 `:1046-1049` |
| Step-scoped filter is "a deliberate executable-interface boundary … must not be wired through" | `CHAPTER_14_AUDIT.md:248` (S1; quoted from `CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md:33-44`) |
| Chains A–F | `TEAM_ALPHA_AUTONOMY_CONTROL_SURFACE_SUBMISSION.md` §2 |
| Part B B1–B20 status | `CHAPTER_13_AUDIT.md:254-280` (B3 vocabulary open; B12 apply open and held; B17 auto-accept open) |
| Frozen list covers eight names | `CHAPTER_13_AUDIT.md` B17 (`:165-168`) |
| S1 (no autonomous retrain) and S3 (applying an LLM proposal, held) | `CHAPTER_13_AUDIT.md:183-200` |
| Three-value audit | `TODO_SELFPLAY_AND_LLM_AUTONOMY.md:112-131` (item 2); D6 implementation `miner/range_miner_coordinator.py:9106`, `:9162` |
| D6 tripwire | `TODO_SELFPLAY_AND_LLM_AUTONOMY.md:82-103`; `D6_THRESHOLD_AUTONOMY_SIGNPOSTS.md` |
| Plan items LL-15 (one vocabulary, derived) and LL-16 (applier with recommended / approved-applied / effective) | `~/tfm_worktrees/ch14-learn/docs/drafts/ch14_v1/PLAN_CH14_LEARNING_LOOP.md:145-146` |
| `pool_size` is sent as `--k`; `--pool-size` is never sent | `CHAPTER_7_AUDIT.md:105` |
| Step-3 wrapper ignores unknown options | `SESSION_NOTES_20260118_PIPELINE_TEST.md:22-27`; `CHAPTER_4_AUDIT.md:73` |
| `parameter_registry.json` is gitignored | `AUDIT_STEP1_OFFSET_REACH.md:374`; `CLAUDE_CODE_REPORT_DOC_FIX_LIST_B_20261117.md:36` |
| The LLM evaluation path is not reached: every manifest is `file_exists` | Ch10 `:89-91` |
| Kill switch writes `/tmp/agent_halt` | Ch12 `:755` |
| "Betas hold are no longer valid" | owner ruling 2026-11-23 10:40 (`logs/e2e_session/OWNER_RULING_step5_go_ch14_beta_holds_20261123.txt`) |

**Labels:**

| Item | Label |
|---|---|
| The map in §1 | **ALREADY RECORDED** link by link (cited inline). New only as a single picture |
| The checker and its classes (§2) | **NEW** (the implementation). The 20 stale specs, the `k`/`pool_size` mapping, the ignored step-3 keys and the TRSE keys are **ALREADY RECORDED** (above; TRSE F1 at Ch10 `:840-842`) |
| The draft patch (§3) | **NEW**, not applied |
| Gap: GBNF `proposed_value` is numeric only, so an on/off or named-choice switch cannot be proposed through `chapter_13.gbnf` or `strategy_advisor.gbnf` | **NEW**: the grammars are quoted in docs, the limit is not stated |
| Gap: the halt file is under `/tmp`, so a halt does not survive a reboot | **NEW**: the path is recorded, the consequence is not |
| Gap: the walk-forward and e2e drivers have no manifest, so WATCHER cannot reach any of their switches | **NEW** |
| Changes a ruling | Registering the Chapter 14 S1 "six fields" in `reinforcement.json` would **CHANGE A RULING** (`CHAPTER_14_AUDIT.md:248`). The draft patch does **not** include them. The owner's 10:40 statement bears on whether that ruling still binds; that is the owner's call |

**Not searched:** the host beyond the main tree and `~/tfm_worktrees`, ser8 archives, the private repository, project files outside the repository.

---

## 1. The map: how a switch is meant to be known, reached, proposed, applied and proved

All anchors are at `41ad509`. `wa` = `agents/watcher_agent.py`.

| Link | What it is meant to do, in plain words | Where | Works today? |
|---|---|---|---|
| **KNOW** | Each pipeline script has a `ScriptParameterRegistry` with one Pydantic `ParameterSpec` per switch: name, CLI flag, type, min and max, choices, default. `validate_value` checks a proposed value. This is "LLM introspection of legal parameter bounds" | `agents/registry/parameter_registry.py:1-9`, `:26`, `:88`, `PARAMETER_REGISTRIES` `:263`, `get_registry` `:587` | **Partly.** 29 specs; **20 are stale**, naming flags their scripts do not define or scripts that do not exist (§2). New switches are not in it |
| KNOW → LLM | `ParameterContext.build` walks the manifest's **primary-action** `args_map`, looks each flag up in the registry, and gives type, bounds and current value to the LLM prompt | `agents/parameters/parameter_context.py:160-195`; built at `agents/contexts/base_agent_context.py:275`; `AgentManifest.adjustable_args` `agents/manifest/agent_manifest.py:374` | **Dormant.** It is built only on the LLM evaluation path, and every manifest is `file_exists`, so that path is not reached (Ch10 `:89-91`) |
| **REACH** | WATCHER loads the step manifest's `default_params`, merges a `--params` key **only if declared there** (the ruled interface boundary), and builds argv. The CLI name comes from the `actions[].args_map` reverse lookup, else `_`→`-`. Bool → bare flag; None or empty → omitted. `_INTERNAL_ONLY_PARAMS` and `output_file` are stripped. `arg_style: positional` passes only `positional_args` and `flag_args` | `wa:416` `STEP_MANIFESTS`; `wa:2026-2034` merge; `wa:2052` pop; `wa:2361-2374` positional; `wa:2379-2420` named; `wa:2393` internal-only | **Yes**, for what is declared. 8 registered step-script switches are reached |
| **PROPOSE** | The LLM output is constrained by GBNF: `parameter_adjustment.gbnf` (`adjustments` object), `watcher_decision.gbnf` (`suggested_params`), `chapter_13.gbnf` and `strategy_advisor.gbnf` (`parameter_proposals`: parameter, current, proposed, delta). It is then validated by Pydantic schemas (`llm_proposal_schema.py`) and policy: acceptance rules (max delta 30 %, at most 3 parameters, cooldown 3 runs) and `FROZEN_PARAMETERS` | `grammars/*.gbnf`; `watcher_policies.json` `acceptance_rules`; `chapter_13_acceptance.py:72`, `validate_proposal` `:327`, delta `:351`; Step-5 whitelist `wa:2611-2651` | **Shape yes, vocabulary no.** The parameter name is a free string in every grammar. The legal list is the hard-coded six-name Step-5 whitelist (`wa:2629-2636`; `watcher_policies.json` has no `llm_adjustable_params`) or the eight-name frozen list (B17). `proposed_value` is numeric only |
| **APPLY** | An accepted value goes into the next run's params | Chain A: `agents/watcher_dispatch.py:537-546` merges Strategy Advisor `selfplay_overrides`, `:180-186` puts them on the selfplay CLI. Chain C: `wa:2653`, `:2735-2810`. Chain D: `chapter_13_orchestrator.py:398-441` | **Chain A only** (`max_episodes`, `min_fitness_threshold`). Chain C is validated, then stopped at the ruled boundary and now reported truthfully (`[S179][TRUTHFULNESS]`, `wa:2735`). Chain D ends at `request_approval` (S3, held). No `apply_parameter_changes` exists (B12) |
| **PROVE** | Three values per change: recommended, approved-applied, effective (read back from what actually ran). Never record "applied" unless the effective value matches | `TODO_SELFPLAY_AND_LLM_AUTONOMY.md:112-131`; implemented for the D6 sieve thresholds only (`miner/range_miner_coordinator.py:9106` `threshold_provenance`, `:9162` validator); Step-1 pin provenance stamp (`wa:511-527`; Ch12 `:1424`, "No decision logic reads them") | **For D6 thresholds only.** `record_applied_changes` (`chapter_13_acceptance.py:635`) and `increment_run` (`:663`) have no caller. `watcher_policies.json:63` `parameter_application` has no reader in Python |
| **STOP** | Kill switch: halt file, failure, retry, runtime, disk and memory checks | `agents/safety/kill_switch.py:47-313`; `wa:1017` | Yes. The halt file is `/tmp/agent_halt` (`kill_switch.py:60`), so a halt does not survive a reboot (NEW, §4 G12) |

**The working template is chain A:** advisor (Pydantic `selfplay_overrides`) → merge (`watcher_dispatch.py:546`) → CLI (`:180-186`) → `selfplay_orchestrator.py` config (`:185-186`). It works because the advisor runs in memory and its two fields have a named, bounded route to a CLI flag. Every broken sibling lacks one of those links.

---

## 2. The checker: every switch, classed against that system

**Files:**
- Tool: `scripts/diagnostics/switch_check.py` (sha256 `ff59eb3863efb62ad769ea8d47026bebfb4fa2fd461902a3339033e901a40966`).
- Output: `SWITCH_CHECK_20261123.md` (sha256 `59e740f9…cb77`) and `.json` (sha256 `68b4068a…c307`), in this folder.

**How it reads its inputs:**
- Main tree at `41ad509` through `git cat-file`. A test proves an uncommitted change is not read.
- 34 worktrees: only the files each changes against its own HEAD.
- Run records on disk.
- Python is parsed with `ast`; it imports nothing from the project.

**How each class is decided** (rules R1–R8 are in the tool header):
- **Registered:** a `ParameterSpec` in `PARAMETER_REGISTRIES` under the script's own name matches the switch, as `get_param` matches.
- **Reachable:** WATCHER's own command-builder rule passes it, with `arg_style` modelled. Keys WATCHER never puts on argv (positional style, `_INTERNAL_ONLY_PARAMS`, `final_params.pop`) count as not passed.
- **LLM-visible:** registered **and** in the primary `args_map`.

**Results at `41ad509` plus worktrees:**

| | All rows | WATCHER step scripts (`argparse` and step wrappers) |
|---|---|---|
| registered and reachable | 17 | **8** |
| registered but unreachable | 1 | **1** (`prediction_generator.py --pool-size`: WATCHER sends the key as `--k`, ALREADY RECORDED Ch7 audit `:105`) |
| not registered | 1,913 | **174** (68 of them WATCHER already passes, unregistered) |
| same basename as a registered script, other path (e.g. `backups/`) | 18 | – |
| config / policy key, not a CLI switch | 192 | – |

**Registered and reachable today:**
- Step 1: `--strategy`, `--trials`, `--max-seeds`, `--prng-type`, `--test-both-modes`.
- Step 4: `--mode`.
- Step 5: `--trials`, `--k-folds`. These two are not LLM-visible, because `reinforcement.json` has no primary `args_map` entry for them.

**Other checks (secondary, in the output):**

| Check | Rows | Prior record |
|---|---|---|
| Stale `ParameterSpec`s | 20 | ALREADY RECORDED B1 |
| Manifest keys the script does not accept | 7 | the 3 TRSE keys: Ch10 F1; the 4 step-3 keys the wrapper ignores: SESSION_NOTES_20260118 |
| Manifest keys WATCHER never passes | 8 | – |
| Step switches WATCHER cannot pass | 107 | – |
| Switches no on-disk run record stores | 1,440 | – |
| Bounds sources disagree | 51 | – |
| Frozen names that a bounds source lets a search move | 12 rows, all one name, `window_size`: frozen at `chapter_13_acceptance.py:72`, searched over `distributed_config.json` `search_bounds` 6–50 | – |
| Same-idea-under-two-names candidates | 311 | – |
| Gitignored `parameter_registry.json` disagrees with the code | 5 entries | – |
| `parameter_registry.json` entries with no switch | 48 | – |

For each unregistered switch the output gives `would_register`: the exact `ParameterSpec` source and, for a step script, the manifest `default_params` and `args_map` entry.

**Tests:**
- `tests/test_switch_check.py`: **10/10** on a fixture git repo with two worktrees. The fixture registry uses the real `parameter_registry.py` classes.
- `tests/mutate_e2e_grader.py`: **16/16** switch-check mutants killed and credited. The full suite, re-run after the last change, is **74/74** killed and credited (grader 47, `compare_settings` 11, switch check 16); all baselines pass (121.5 s on 2 cores).
- Everything ran on CPU with at most 2 cores, under `nice`. The real run took 16 s and 80 MB.

---

## 3. The draft patch set (not applied)

Location: `switch_patches_20261123/`, with `SHA256SUMS`:

| File | sha256 | Content |
|---|---|---|
| `parameter_registry.py.patch` | `7654fa9b…975f` | Unified diff against `41ad509:agents/registry/parameter_registry.py` |
| `parameter_registry.py.patched` | `9f52ed5f…13f5` | The patched file |
| `manifest_fragments.json` | `46e5560f…564` | Manifest entries |

**What it registers:**
- **Scope:** the switches that worktrees add in files modified on or after 2026-11-23. That is 281 switches, which become 275 specs after de-duplication. Test harnesses and `backups/` are excluded.
- **Placement:** each spec is appended to its script's `parameters` list, or to a new `ScriptParameterRegistry` for a script the registry does not have.
- **Defaults:** each is the argparse default when that is a literal. A computed default is left out, with the expression in a comment.
- **Bounds:** only where a bounds source gives them.

**Verified:**
- The patched file imports under Pydantic.
- 29 → 304 specs, and the 29 existing specs are byte-for-byte unchanged in `model_dump`.
- Every non-required new default passes its own `validate_value`.
- The 69 required arguments have no default, and `validate_value(None)` correctly refuses them.
- No worktree pair adds the same name with different specs.

**Largest new registrations:**

| Script | New specs |
|---|---|
| `e2e_walk_forward.py` | 68 |
| `e2e_vote_harness.py` | 33 |
| `e2e_grader.py` | 28 |
| `e2e_step3_harness.py` | 24 |
| `e2e_step5_ranker.py` | 21 |
| `e2e_build_run_copy.py` | 14 |
| `compare_settings.py` | 11 |
| `inner_episode_trainer.py` | 10 |
| `selfplay_orchestrator.py` | 8 |

**WATCHER step scripts and their manifest fragments:**
- `meta_prediction_optimizer_anti_overfit.py`: 5 (`diagnostics_level`, `gradient_probe_rows`, `exclude_features`, `include_feature_groups`, `exclude_feature_groups`).
- `window_optimizer.py`: 2 (`derived_anchor_domain`, `anchor_holdout_guard`).
- `prediction_generator.py`: 1 (`schema_check`).

These have **manifest fragments** for `reinforcement.json`, `window_optimizer.json` and `prediction.json`. Each declares the code's existing default (`null` / `false` / `"legacy"`), so WATCHER would pass the same value explicitly.

**Hand-landed:**
- `agent_manifests/` is in `HAND_LANDED_DIRS` (`scripts/owner_land.py:87`). The owner lands those fragments by hand.
- `agents/registry/parameter_registry.py` is not governed or hand-landed (`GOVERNED_FILES`, `scripts/gate12_parity_gate.py:174`; `HAND_LANDED_FILES`, `owner_land.py:86`).
- **Excluded from the patch:** the six Chapter 14 S1 fields (CHANGES A RULING, see Prior record).
- **Not done here:** the 20 stale specs are neither fixed nor removed. That is the owner's choice under B1.

**Not reachable even after the patch:** most new switches live in scripts that are **not WATCHER steps**: the e2e harnesses, the walk-forward driver, selfplay and Chapter 13 tools. Registering them makes them *known* to the registry; nothing can *reach* them until a manifest or dispatch route exists (G6).

---

## 4. Gaps between this and the autonomy goal (each a to-do, with its recorded ID)

**Rows G1–G12:**

| # | To-do | Recorded ID | Who decides |
|---|---|---|---|
| G1 | Make the registry match the scripts: fix or drop the 20 stale specs, then land the draft patch | CHAPTER_10_AUDIT B1; Ch10 S1 (`offset` → `window_anchor`, Brief II) | Owner (B1 "owner choice") |
| G2 | Build the one vocabulary the proposers read, **derived** from the registry, manifests and bounds sources, not a new format. The checker's JSON is a derived view, not the source | TODO B3 (`chapter_13_parameter_vocabulary.py`, open, CHAPTER_13_AUDIT `:263`); plan LL-15 | Lead / owner |
| G3 | Derive the frozen list from the manifests and the reserved-authority list. Eight names today; the frozen `window_size` is also searched by Optuna over `search_bounds` 6–50 (12 rows) | CHAPTER_13_AUDIT B17 | Owner |
| G4 | Make the LLM actually see the registry: `ParameterContext` is only built on the LLM evaluation path, which `file_exists` manifests bypass | Ch10 `:89-91`, `:928-929` | Owner |
| G5 | Let grammars carry switch values that are not numbers (`proposed_value` is `number` in `chapter_13.gbnf` and `strategy_advisor.gbnf`), and constrain `parameter` to the vocabulary instead of a free string | NEW (value type); B17 (free string) | Lead / owner |
| G6 | Give the learning-loop drivers a reach route: the walk-forward, e2e vote and grade harnesses and `compare_settings` have no manifest, so 0 of their switches are WATCHER-reachable | NEW; plan LL-18 (learner queue) and LL-16 (applier) are the nearest | Owner |
| G7 | An apply step that writes a new run record or pin bundle (never edits the main config), like chain A, for each parameter with a location and bound | TODO B12 (open, held by ruling S3); plan LL-16 | **Owner** (S3 is SWITCHED-OFF-BY-RULING; owner 10:40 says Beta holds no longer bind) |
| G8 | Record changes: call `record_applied_changes` and `increment_run` after a physical application; then the cooldown and reversal rules stop being vacuous | CHAPTER_13_AUDIT S3 consequences (`:200`) | follows G7 |
| G9 | Prove it took effect for **every** applied switch, not only D6 thresholds: the three-value audit with the effective value read back from the run's own record. 1,440 of 2,141 switch rows are stored in no on-disk run record today | TODO_SELFPLAY item 2 (`:112-131`); plan LL-16 | Lead (build) / owner (gate) |
| G10 | Decide `watcher_policies.json:63` `parameter_application: true`, which has no reader | D6 signposts §3 (Beta: do not annotate; Part B resolves); CHAPTER_13_AUDIT S3 | Owner |
| G11 | Step-5 LLM whitelist: the six names are hard-coded in `wa:2629-2636` and not declared in `reinforcement.json`, so they stop at the ruled boundary | CHAPTER_14_AUDIT S1 (ruling); CHANGES A RULING if wired through | **Owner** |
| G12 | Make a halt survive a reboot: `/tmp/agent_halt` (`kill_switch.py:60`) is cleared on reboot, and the box rule is "never /tmp" (CLAUDE.md box rule 7) | NEW (consequence); Ch12 B3 (two halt files) | Owner |

**Rows G13–G19:**

| # | To-do | Recorded ID | Who decides |
|---|---|---|---|
| G13 | Connect the other advisor fields (`search_strategy` and three more) through chain A's route; nothing reads `strategy_recommendation.json` | Submission chains A′, B, F; Q3 (no ruling found in the tree) | Owner |
| G14 | One description of each switch: the gitignored `parameter_registry.json` disagrees with code on 5 entries and lists 48 with no switch; manifest `parameter_bounds` are read only by unimported code | `parameter_registry.json`: ALREADY RECORDED (gitignored); D21 (Ch10 `:812-816`) | Owner |
| G15 | `prediction.json` sends `pool_size` as `--k`; `--pool-size` is unreachable | CHAPTER_7_AUDIT `:105` | Owner (manifest, hand-landed) |
| G16 | The step-3 wrapper drops four declared keys silently (`run_step3_full_scoring.sh:120-123`); TRSE `default_params` has three keys its script lacks | SESSION_NOTES_20260118; Ch10 TRSE F1 | Owner |
| G17 | Bound every switch a learner may move: most new switches have no bounds source, so their `ParameterSpec` has no min/max, and to the LLM that reads as unbounded | plan LL-15 (bounds derived, frozen list) | Owner (bounds are authority) |
| G18 | Human approval stays the default: no autonomous retrain, and an approved change needs human approval | CHAPTER_13_AUDIT S1; plan LL-16 `apply_mode human_approval` default | Owner |
| G19 | D6 tripwire: sieve thresholds may enter any apply path only through the single `build_stripe_assign_payload` chokepoint | TODO Part B tripwire `:82-103`; D6 signposts | Holds |

**Not verified:**
- **Run-record evidence:** only the records on disk today count: manifest outputs in the main tree, and `run_record*.json` under worktree `data/`. A record format not on disk is not seen.
- **Same-idea-under-two-names (311):** these are name-similarity candidates, not findings. Only the `args_map` pairs are certain.
- **The patch as working code:** it is checked for syntax, Pydantic import, unchanged existing specs and valid defaults. It is not tested as running code inside the agents.
- **Step-2/3 wrapper forwarding** to the Python they call is not traced; R2 notes this per row.
