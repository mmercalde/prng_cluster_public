# Window anchor: status and plan (window-anchor Brief II, Part 1)

**Date:** 2026-11-23 · **Host:** VM 101, user `michael` · **Worktree:** `~/tfm_worktrees/anchor-brief2`
at `41ad509` (clean at start) · **Model:** Opus 5.5, one session, no sub-agents.
**Status:** draft. Nothing committed, pushed, relayed to Beta or deployed. No rig, GPU, pipeline step
or Step-1 run.
**Asked for by:** `logs/e2e_session/TASK_ANCHOR.txt` (main tree). Owner words, verbatim:
- 2026-11-23 10:32: "The sieve and /or ml / watcher needs to be able to choose any window. This needs addressed."
  (`logs/e2e_session/OWNER_DIRECTION_any_window_20261123.txt`)
- 2026-11-23 10:35: "Remember that ultimately self play and ml need to be able to adjust these. Also read up on
  chapter 14" (`logs/e2e_session/OWNER_DIRECTION_ml_adjusts_window_20261123.txt`)

**Yardstick.** Nothing here is judged by whether a seed is found. A window choice matters only through pool hits
against `k/1000` on draws nothing upstream saw. §G below is the one place this document touches that yardstick,
and it is a measured fact about where the holdout starts, not a verdict.

---

## Plain summary

1. **Today the sieve can sample anchors 0 to 100 only.** Optuna draws `offset` from `search_bounds.offset`
   = `[0, 100]` (`distributed_config.json:80-83` → `window_optimizer.py:165-166` →
   `window_optimizer_bayesian.py:543-545`). The worker itself already accepts any anchor in
   `[0, N_filtered - window_size]` and refuses, with no clamp, anything outside
   (`miner/range_miner_worker.py:908-924`). So the limit is the Optuna bound, not the engine.
2. **A hand-pinned anchor above 100 already runs today**, by reading, on the RANGE-MINER route, through WATCHER's
   Route-A pin and through the direct CLI. Nothing refuses it. Optuna logs a `UserWarning` ("Fixed parameter
   'offset' with value 500 is out of range") and uses the value (measured on CPU: §C). The same is true of a pinned
   `window_size` outside `[6, 50]`. That works by Optuna's leniency, not by a declared rule.
3. **Brief I is not recorded as accepted anywhere.** The last ruling is "Brief I NOT ACCEPTED … Brief II remains
   BLOCKED" (`docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7`). Run 4 attempt 3 (2026-11-15) was credited
   for seed coverage, but no `G-PROD-SHAPE` result and no Brief I acceptance is committed (§B). The owner lifted the
   Brief II block once before, for PWC only (2026-11-20 17:38). Part 2 of this task is therefore Brief II work done
   ahead of the recorded sequence, at the owner's direction of 10:32 today. It is labelled **CHANGES A RULING (in
   sequence)** below and needs the owner's word on the record.
4. **The smallest correct change (§D)** adds one declared setting, `derived_anchor_domain` (default `false`). Off,
   every path is byte-identical to today. On, Optuna samples the anchor over the derived domain, `[0, N_filtered -
   window_size]` after the session filter (no cap by default), and every proposed window (sampled, hand-pinned, or
   a later learner's) passes one validator that names anchor, domain, sessions and dataset sha256 when it refuses.
   The setting travels manifest → CLI → signature like `deliver_skip_range` did.
5. **The derived domain reaches Step 1's own holdout (§G).** Step 1 writes the last 20% of the combined history as
   `holdout_history.json` (`window_optimizer.py:1165-1176`). Measured on the live dataset, the holdout starts at
   midday filtered index 6,707 and evening 7,747. The proposal's governed era starts at 6,791 / 7,830, inside it. A
   sieve window placed there sees draws that later grading may treat as unseen. This is an owner question (Q2), not
   something Part 2 decides.
6. **For selfplay or the ML to move the window (§H):** nothing proposes a window value today. Selfplay's outer
   episode is off and its target script has no episode interface; Chapter 13 lists `window_size` as a frozen
   parameter and nothing it accepts is applied; `propose_transform_update` returns its input; neither `offset` nor
   `window_size` is a model input. Several of those stops are recorded Beta review positions the owner may keep or
   lift; they are listed there with their records. Part 2 builds the one validator a learner would have to pass
   through, and grants no learner any authority.

---

## Prior record

### Searches run (worktree at `41ad509`; `/bin/grep`, because the shell `grep` skips `*.json`)

```
ls docs | grep -i -E "anchor|offset"
/bin/grep -rln -i -E "brief[ _-]?i\b.*accept|BRIEF I ACCEPTED|brief_i.*accept" docs/ docs/exchange
/bin/grep -rn -i -E "BRIEF I ACCEPTED|Brief I (is )?(finally )?accepted|Brief-I acceptance" docs/ --include=*.md
/bin/grep -rln -i -E "credited clean|attempt 3 credited|run 4.{0,40}(publish|credited|PASS)" docs/
/bin/grep -rn -i -E "brief.?i.{0,60}(accept|not accepted)" docs/*202611{15..22}*.md
ls docs | grep -i -E "run4|RUN4"; ls docs/exchange/{from,to}_beta | grep -i -E "run4|brief|anchor|offset|accept"
/bin/grep -rln -E "G-PROD-SHAPE" docs/
/bin/grep -n -i -E "window|anchor" docs/LEADS.md docs/BACKLOG.md
/bin/grep -n -E "window_anchor|offset" docs/TODO_CONSOLIDATED.md        # rows :79-89, :241
/bin/grep -rln -E "FROZEN_PARAMETERS|frozen parameter" docs/
/bin/grep -n -E "\bS1\b|\bS4\b|\bS5\b|\bS8\b" docs/CHAPTER_14_AUDIT.md
/bin/grep -n -E "\bB10\b|\bB11\b|\bB14\b|\bB17\b" docs/CHAPTER_13_AUDIT.md
/bin/grep -rlw --include=*.py --include=*.sh offset .                    # 185 files, census in §I
/bin/grep -rn -E "window_anchor|generator_phase" --include=*.py .        # producers outside tests
```
`docs/PROJECT_FILE_CATALOG.md`, `docs/TB_RULING_*`, `docs/TB_RULING_REQUEST_*`, `docs/PROPOSAL_*`,
`docs/CHAPTER_*`, `docs/*_AUDIT.md`, `docs/TODO_CONSOLIDATED.md`, `docs/BACKLOG.md`, `docs/LEADS.md` and
`docs/exchange/` were all inside these scopes.

### Documents that decide, authorise, schedule or record this subject (all opened)

| Document | What it settles |
|---|---|
| `docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` §2, §4.2, §4.3, §4.6, §7 AC3, §9 | The live design: derived domain, `{min:0, max_cap:null}`, no clamp, phase 0 not tunable, anchor vs envelope (100 vs 149), Brief II scope |
| `docs/TB_RULING_WINDOW_ANCHOR_V1_1_DESIGN_GATE_CLOSED.md` | v1.1 APPROVED; design gate closed; Brief II starts from the accepted Brief I commit |
| `docs/S172_WINDOW_ANCHOR_BRIEF_I.md` C-2 (`:60-90`), `docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md` | Brief I scope (worker, schema, capability, legacy closure); `WindowConfig.offset` and Optuna left to Brief II |
| `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_SCOPE_RULING.md` | Pin order capability-then-policy; nine-site clamp census |
| `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md` | Commit authorised, not accepted; **M8 transfers to Brief II** and must mutate the real control-era ceiling 100 → 149; `sieve_gpu_worker.py` and C-2 consumers are Brief II's audit; EXEC-PIN-1 |
| `docs/S172_WINDOW_ANCHOR_BRIEF_I_CLOSURE_STEP23_REPORT.md` | Parity 30/30 and schema parity 3/3 at `48a8705`; production-shape proof UNAVAILABLE |
| `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7` | Brief I NOT ACCEPTED at `48a8705`; bounded repair (`69ca910`, array 4 = `window_anchor`); **Brief II BLOCKED** |
| `docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md` §2 F2 (`:57`), §3.A A8, §3.C C1 (`:100`) | Brief I acceptance = Run 4 publishes + `G-PROD-SHAPE` PASS + owner accepts; Brief II is C1, after "finished" |
| `docs/TODO_CONSOLIDATED.md:79`, `:84`, `:89` | Run 4 is the Brief I acceptance run; acceptance owed; Brief II "Not written; blocked until Brief I is accepted" |
| `docs/AUDIT_STEP1_OFFSET_REACH.md` §1, §3 | The 0-100 bound has no derivation; reach is `data[0:150]` |
| `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1172`, `:1385` | F-4 CONFIRMED at the audit anchor; "repair implemented by Window-Anchor Brief I; acceptance pending" |
| `docs/TB_RULING_RUN4_ROUTING_AND_PINNED_GEOMETRY.md:10`, `:16` | Route A (the seven-key operator pin) approved; search_bounds narrowing (Route B) rejected for Run 4; Run 4 geometry anchor 25 |
| `docs/drafts/pwc_backup_stage2b_v1/README.md:49`; commit `b2efdef` | Owner lifted the Brief II block **for PWC only** (2026-11-20 17:38): PWC TCP now validates the anchor against the derived domain worker-side |
| `docs/CLAUDE_CODE_REPORT_DOC_FIX_LIST_B_20261117.md:45`, `:96` | No committed `G-PROD-SHAPE` result for Run 4 attempt 3 and no Brief I acceptance found (searched at `03f3712`) |
| `docs/CHAPTER_13_AUDIT.md` B10, B11, B14, B17; `docs/CHAPTER_14_AUDIT.md` §5 S1, S4, S5, S8; `docs/CHAPTER_14_TRAINING_DIAGNOSTICS.md` §9.2, §15, §16 | The learner-side stops in §H |

### Labels for the items this document raises

| Item | Label |
|---|---|
| The Optuna bound is the only thing holding anchors to 0-100; the worker already validates the derived domain | ALREADY RECORDED (Brief I report; `AUDIT_STEP1_OFFSET_REACH.md`); restated with current anchors |
| A hand pin above 100 runs today with only an Optuna warning; a pinned `window_size` outside `[6,50]` too | **NEW** (measured this session, §C) |
| Brief I acceptance not recorded | ALREADY RECORDED (`TODO_CONSOLIDATED.md:84`; fix list B `:96`) |
| Doing Brief II work now, before Brief I acceptance | **CHANGES A RULING** in sequence (`TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7`; v1.1 §9). **Ruled by the owner 2026-11-23 11:25: "Yes"** (Q1) |
| Part 2 keeps `search_bounds.offset` and the bounded domain as default instead of removing the key outright | **CHANGES A RULING** (v1.1 §4.3: "REMOVED … outright"). Done because `TASK_ANCHOR.txt` requires the bounded behaviour to stay default until the owner rules (Q3) |
| The derived domain reaches Step 1's holdout (§G) | **NEW** as measured numbers; the label/grade question it feeds is ALREADY RECORDED (e2e map Q1, `MAP_E2E_THIN_PATH_20261122.md:30`) |
| `--config-file` route defaults `offset` to 100 and `window_size` to 1024 when absent, with no bounds | **NEW** as far as found (`window_optimizer.py:1376-1384`); not changed |
| Learner-side stops (§H) | ALREADY RECORDED (Ch13/Ch14 audits); `window_size` in Ch13 `FROZEN_PARAMETERS` is ALREADY RECORDED as B17 |

### Not searched

The host outside this worktree and the main tree's `logs/` (only the task and owner files were read), ser8
archives, the private repository, claude.ai project files and PDFs, the rigs (not touched), Proxmox, other
worktrees except `~/tfm_worktrees/e2e-map/docs/drafts/e2e_v1/MAP_E2E_THIN_PATH_20261122.md`.

---

## A. What is in the code today: what can and cannot choose a window

A window is `(sessions, window_size, window_anchor)`: `filtered[anchor : anchor + window_size]` over the
session-filtered history.

| Who | Can choose | Limit and where it is set |
|---|---|---|
| **Optuna, sequential Bayesian path** (`n_parallel = 1`) | `window_size`, `offset` (= anchor), `session_idx` | `suggest_int('offset', min_offset, max_offset)` at `window_optimizer_bayesian.py:543-545`; bounds `[0, 100]` from `distributed_config.json:80-83` via `window_optimizer.py:67-92`, `:165-166` (code fallback `:74`, `:142-143`). `window_size` `[6, 50]` (`distributed_config.json:73-79`) |
| **Optuna, NP2 partition path** (`n_parallel > 1`) | same | its own `_worker_obj`, `window_optimizer_integration_final.py:2998-3003`, same bounds |
| **Hand pin, direct CLI** | all seven warm-start values | `--warm-start-offset` etc. (`window_optimizer.py:1612-1613`, passed `:1994-2000`) → `study.enqueue_trial` (`window_optimizer_bayesian.py:806-819`). No range check (§C) |
| **Hand pin, WATCHER Route A** | the same seven, all or none, operator CLI only | `STEP1_EXPLICIT_PIN_KEYS` (`agents/watcher_agent.py:511-519`); usability-only checks (`:578-646`); routed past the default_params filter at `:2042-2050` and the strip at `:2393-2407`. Programmatic callers are refused (`:690-711`) |
| **`--config-file` route** | whatever the file says | `offset=config.get('offset', 100)`, `window_size=config.get('window_size', 1024)`, no bounds (`window_optimizer.py:1376-1384`) |
| **RANGE-MINER worker** | nothing; it validates | `load_residue_window`: refuses `anchor < 0` or `> N_filtered - window_size`, no clamp (`miner/range_miner_worker.py:908-924`); the parent uses the same function (`window_optimizer_integration_final.py:273-297`). Anchor enters as `window_anchor = config.offset`, phase pinned 0 (`:2175`, `:2180`) |
| **PWC TCP / local** | nothing; it validates | `validate_pwc_window_anchor` refuses non-int or < 0 (`persistent_worker_coordinator.py:435-447`); worker refuses outside the derived domain (`sieve_filter.py:614-620`) |
| **PWC SSH route** | refuses any anchor ≠ 0 | `validate_pwc_window_offset` via `persistent_worker_coordinator.py:440-441` |
| **ZMQ and legacy `coordinator.py` jobs (old schema)** | silently clamps | `sieve_filter.load_draws_from_daily3`: `start = max(0, min(offset, n - window_size))` (`sieve_filter.py:184`); helper `_get_residues_for_config` fallback the same (`window_optimizer_integration_final.py:266`). Non-certifying routes |
| **WATCHER LLM / Chapter 13 / selfplay / Steps 4-6** | nothing | §H |
| **Unused** | `SearchBounds.random_config`/`is_valid` (`window_optimizer.py:224-247`) — no production caller found (`/bin/grep -rn "\.is_valid("` returned none outside tests) |

Inert declarations (read, not live): `agent_manifests/window_optimizer.json:89-97` `parameter_bounds.offset` max
2000, description "Time offset from current draw position" (wrong: head-relative index); `parameter_bounds` has no
reader outside `apply_s*.py`. `agents/registry/parameter_registry.py:282-289` `offset` max 500. Both are Brief II
items (v1.1 §4.6); Part 2 does not touch them (Q5).

## B. Is Brief I acceptance recorded as done?

**No.** Searched as listed in the Prior record.
- Last ruling: `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7` — "Brief I **NOT ACCEPTED** … **Brief II
  remains BLOCKED**". The repair `69ca910` landed after it.
- Acceptance condition: completion plan F2, "Run 4 publishes; `G-PROD-SHAPE` PASS; owner accepts after Beta's review"
  (`docs/PROPOSAL_S172_COMPLETION_PLAN_v1_1.md:57`); `docs/TODO_CONSOLIDATED.md:84` still open.
- Run 4: attempt 3 over `[2³¹, 2³²)` credited clean on 2026-11-15 (`CLAUDE.md` §6 status note;
  `docs/CLAUDE_CODE_REPORT_RIG_GPU_FAULT_DECODE_20261115.md:154`). No committed `G-PROD-SHAPE` result for it and no
  acceptance (`docs/CLAUDE_CODE_REPORT_DOC_FIX_LIST_B_20261117.md:96`; my searches agree at `41ad509`).
  `G-PROD-SHAPE` cannot pass on this host while it aggregates other runs' rows (closure report §4.1).
- The only lift on record is the owner's for PWC (`docs/drafts/pwc_backup_stage2b_v1/README.md:49`).

## C. Can a hand-pinned anchor above 100 run today?

**By reading, yes, on the RANGE-MINER route, both ways. No line refuses it.**

Direct CLI: `--warm-start-offset 500` (plus the other six) → `optimize_window(warm_start_offset=…)`
(`window_optimizer.py:970`) → `study.enqueue_trial({'offset': 500, …})` (`window_optimizer_bayesian.py:815-819`) →
the objective's `trial.suggest_int('offset', 0, 100)` (`:543-545`). Optuna 4.4.0
(`optuna/trial/_trial.py:647-660`) treats an enqueued value as fixed: out of range it warns and returns the value.
Measured on CPU this session with an in-memory study:

```
UserWarning: Fixed parameter 'offset' with value 500 is out of range for distribution IntDistribution(high=100, low=0)
UserWarning: Fixed parameter 'window_size' with value 60 is out of range for distribution IntDistribution(high=50, low=6)
returned {'o': 500, 'w': 60} state TrialState.COMPLETE
```
Then `WindowConfig(offset=500)` (`:580-588`) → `window_anchor=500` (`window_optimizer_integration_final.py:2175`) →
parent residues `load_residue_window(…, 500)` → the derived-domain check (`miner/range_miner_worker.py:915-924`),
which passes for any anchor ≤ `N_filtered - window_size`.

WATCHER: the seven keys reach the CLI only through the operator pin (`agents/watcher_agent.py:2042-2050`,
`:2402-2407`); `capture_step1_pin_bundle` checks presence and usability, not range (`:614-646`).
`utils/step1_search_ledger.py:194-218` records the pinned anchor in the search space and does not bound it.

**Where it is refused or bent, by route:** PWC SSH refuses ≠ 0 (`persistent_worker_coordinator.py:440-441`). ZMQ and
legacy old-schema jobs clamp silently (`sieve_filter.py:184`). A pin above `N_filtered - window_size` raises
`ResidueResolutionError` at `miner/range_miner_worker.py:916`, in the parent, before dispatch.

**What is wrong with "it already works":** the route depends on Optuna's fixed-parameter leniency, which logs only
a warning; a pinned `window_size` outside the recorded `[6, 50]` passes the same way; and a later trial in the same
study samples back inside `[0, 100]`. Part 2 makes the pinned route a declared, validated rule.

## D. The smallest correct change set for Brief II, as the design gate fixed it

What v1.1 requires, and what Part 2 does with each:

| v1.1 requirement | Part 2 |
|---|---|
| Anchor domain derived from the data after the session filter: `min 0`, `max_cap null`, `effective_max = derived_max if max_cap is null else min(max_cap, derived_max)` (§4.3) | New `utils/window_anchor_domain.py`. `N_filtered` is computed per session option and then **proved against the worker's own `load_residue_window`** (anchor `N-1` loads with window 1, anchor `N` raises), so the session filter keeps one authority |
| No silent clamp; out-of-domain raises naming anchor, domain, sessions, dataset sha256 (§4.2) | `WindowAnchorDomainError`, raised before any study trial reaches the sieve |
| `generator_phase` stays 0, not tunable (§2, §4.1) | Untouched. The new code never reads or writes phase; the record carries `generator_phase: 0` as today |
| Anchor vs envelope never confused; AC3 regression: anchor 149 with window 50 not in `control_era` (§2, §7 AC3) | `control_era_range` = `[min, min(bounded_max, effective_max)]`, `bounded_max` read live from `search_bounds.offset.max` (today 100). This is the production site M8 must mutate (100 → envelope 149) |
| NPZ contract unchanged, metadata only (§4.5) | No array touched. The miner already carries `window_anchor` in metadata |
| Step 3 consumer law untouched (§4.7) | Untouched |
| Sampling and validation move to the anchor (§4.3) | Only when the setting is on. Optuna then samples `window_size` and `session_idx` first, then the anchor in `[min, effective_max(window_size, sessions)]` |
| `search_bounds.offset` removed outright; `WindowConfig.offset` renamed; registry entry replaced; cache key `_A{anchor}_P{phase}_` (§4.3, §4.6) | **Not done** — they change the default path. Recorded as TODO and Q3/Q5 |

**The setting.** `derived_anchor_domain`, bool, default `false`; CLI `--derived-anchor-domain`. Three hops, the
`deliver_skip_range` pattern (`f916532`, `00e2ee7`): manifest `default_params` + `args_map` + `param_docs`
(`agent_manifests/window_optimizer.json`) → `window_optimizer.py` argparse and `run_bayesian_optimization` call →
`optimize_window(derived_anchor_domain=…)` signature. A bool is flag-only in WATCHER's builder
(`agents/watcher_agent.py:2420-2425`), so with the default `false` the dispatched argv is byte-identical to today.

**Fail-closed edges, on only:** refused with `--config-file`, with `--strategy` other than bayesian, with
`n_parallel > 1` (the NP2 objective is a second sampling site, not migrated), with `--resume-checkpoint`, and on any
route other than RANGE-MINER (ZMQ/legacy clamp; PWC is Q6). A study, or a Step 1 search, made by one mode refuses the
other, as the skip switch does (`window_optimizer_integration_final.py:1758-1830`): the search space gains a key only
when on, so every existing search digest is unchanged (`utils/step1_search_ledger.py:254-257` precedent).

**One route for every proposer.** `check_window_choice(ctx, window_size, anchor, sessions)` validates window size
against the recorded `[6, 50]` and the anchor against the effective domain. On, it is called for every trial —
sampled or pinned — before `WindowConfig` is built. A later learner would call the same function (§H); nothing grants
it authority.

## E. Governed or owner-hand-landed files (an agent cannot land them)

Read from `scripts/owner_land.py:84-87` and the parity gate's `GOVERNED_FILES` (ten files: `adaptive_thresholds.py`,
`execution_set.py`, `hybrid_strategy.py`, `miner/__init__.py`, `miner/dataset_authority.py`,
`miner/range_miner_coordinator.py`, `miner/range_miner_protocol.py`, `miner/range_miner_worker.py`,
`prng_registry.py`, `sieve_gpu_worker.py`), plus `agents/watcher_agent.py`, `gate12_launch.sh`, the parity gate,
everything under `miner/` and `agent_manifests/`.

Part 2 touches exactly one of them: **`agent_manifests/window_optimizer.json`** (hop 1). The owner lands it by hand.
It touches no `miner/` file, so no fleet redeploy or parity re-proof is needed for the change itself. Before-commit
review class: launch path for runs that count (`CLAUDE.md`, Beta reviews), so a Beta request is drafted (Part 2).

## F. Owner questions (one line each)

- **Q1. ANSWERED 2026-11-23 11:25 PST**, owner, verbatim: "Yes" — Brief II goes ahead now without waiting for Brief I acceptance; default-off switch; the Brief I `G-PROD-SHAPE` check re-run when the rigs are next up, as paperwork not as a gate. This lifts the Brief II block of `docs/TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7` (`logs/e2e_session/OWNER_RULING_brief2_goes_ahead_20261123.txt`, main tree).
- **Q2.** The derived domain reaches Step 1's holdout (midday filtered index ≥ 6,707, evening ≥ 7,747). **Built as a second declared setting, `anchor_holdout_guard`, both values working** (lead, 2026-11-23); the default is proposed in `READY_ANCHOR_BRIEF2.md` for the owner.
- **Q3.** Keep `search_bounds.offset` and the bounded domain as the default, or make the derived domain the default and remove the key as v1.1 §4.3 says?
- **Q4.** Should a pinned `window_size` outside the recorded `[6, 50]` be refused (it is, with the setting on), and should that also apply with the setting off?
- **Q5.** Rename `WindowConfig.offset` → `window_anchor`, replace `parameter_registry.py`'s `offset` entry and the manifest's inert `parameter_bounds.offset` now, or later as one migration?
- **Q6.** Allow the derived domain on PWC TCP (its worker already validates the domain), or RANGE-MINER only?
- **Q7.** Which learner, if any, should be the first allowed to propose a window, and through which authority (§H)?

## G. Window size: what the record says (nothing changed)

- Live: `distributed_config.json:73-79`, `min 6`, `max 50`, `default 12`. `min` raised 2 → 6 by TB ruling, S172
  2026-04-30 ("W=2/3 produces ~39%/53% survivor rate by chance alone"); `max` 500 → 50 at S139
  (`apply_s139_window_max_50.py`); W12 "empirically preferred production baseline" (S148).
- Code fallback `{2, 50}` used only when the key is absent (`window_optimizer.py:73`).
- Manifest `parameter_bounds.window_size` says `min 2` (`agent_manifests/window_optimizer.json:79-87`) — stale
  against 6, and inert (no reader).
- TRSE Rule A caps `max_window_size` at 32 when short persistence is active (`window_optimizer_bayesian.py:648`);
  the 32 is `recommended_window_size × 4` with the multiplier wiring missing (fact file §2.7 #5b).
- S112 found real data optimises near W8 (`SESSION_CHANGELOG_20260226_S112.md:169-184`); no bound moved.

**Holdout fact (bears on Q2).** Step 1 splits the combined 18,068 records 80/20 at combined index 14,454
(2021-03-17 midday) into `train_history.json` / `holdout_history.json` (`window_optimizer.py:1165-1176`). Measured
on `daily3.json` (sha256 `513648160d35…68f6`, read-only, main tree):

| session filter | N_filtered | derived max at W50 / W6 | first filtered index in the holdout |
|---|---|---|---|
| midday + evening | 18,068 | 18,018 / 18,062 | 14,454 |
| midday | 8,515 | 8,465 / 8,509 | 6,707 |
| evening | 9,553 | 9,503 / 9,547 | 7,747 |

v1.1's governed era starts at midday 6,791 / evening 7,830, so every governed-era window is inside the holdout.
`max_cap` can hold the anchor below a chosen boundary (e.g. first holdout index − window_size), but a single
`max_cap` cannot track window size. That is the owner's call (Q2); Part 2 leaves `max_cap` at `null` as v1.1 says.

## H. What would have to exist for selfplay or the ML to move the window

Facts and options only. Nothing is built for this section and nothing is recommended for removal. Under the owner
rulings of 2026-11-11, every Beta position below is a review position the owner may keep or lift.

**What could propose a window value today:** nothing. Searched producers of `window_size`, `offset`,
`window_anchor` in `chapter_13_*.py`, `parameter_advisor.py`, `selfplay_orchestrator.py`, `inner_episode_trainer.py`,
`policy_transform.py`, `policy_conditioned_episode.py`, `modules/`: none sets a Step-1 window.
- Selfplay's `"window"` transform is a survivor-list index slice (`policy_transform.py:435-460`), not a draw window.
- The Chapter 13 LLM grammar can *name* any parameter as a string (`docs/CHAPTER_13_AUDIT.md` B17).

**Where a proposal would stop, in order along each path:**

| # | Stop | Anchor | Kind |
|---|---|---|---|
| 1 | Neither `offset` nor `window_size` is a model input: Step 3's merge list names 18 trial-level fields, not these (`survivor_scorer.py:774-779`); `best_offset` is a hardcoded 0 (`:707`). The ML has no signal about the window to learn from | Ch14 §15.1-15.3; fact file §2.20 | fact, plus S5 (below) for the per-seed sieve fields |
| 2 | Selfplay explores nothing: outer episodes off (`configs/selfplay_config.json` `use_coordinator: false`; `selfplay_orchestrator.py:599-610`); the target `coordinator.py --episode-id` has no such interface; `sieve_params` (`:198`, `:633-634`) is the only place a window would go | Ch13 **B14** | BROKEN (no interface on the far side) |
| 3 | Nothing requests selfplay: `request_selfplay` (`chapter_13_triggers.py:762`) has no caller | Ch13 **B10** | BROKEN; S179 §2.1 requires a gate inside it |
| 4 | `propose_transform_update` returns its input unchanged (`policy_conditioned_episode.py:373-401`, "Phase 9B.2: No automatic updates") | Ch13 S4 (`docs/PHASE_9B2_INTEGRATION_SPEC.md:393-409`) | **Beta position** (Phase 9B.3 not authorised) |
| 5 | Promotion drops the policy (`chapter_13_acceptance.py:240`) | Ch13 **B11** | BROKEN (GOVERNED as status) |
| 6 | Chapter 13 refuses `window_size` (and `window_start`, `window_end`) as frozen (`chapter_13_acceptance.py:72-84`, `:388`); `offset` is not in the set | Ch13 B17 | design list (chapter `:658`); latent, since nothing is applied |
| 7 | Accepted Chapter 13 proposals are not applied: they end at `pending_approval`; `record_applied_changes` has no caller | Ch13 S3/B16 | **Beta position** ("`pending_approval` is a valid authority boundary") |
| 8 | Diagnostic/LLM retry proposals do not reach a step's params: WATCHER merges only declared `default_params` (`agents/watcher_agent.py:2026-2034`) | Ch14 **S1** | **Beta position** (step-scoped filter is "a deliberate executable-interface boundary") |
| 9 | The seven warm-start keys route only from the operator CLI; a programmatic caller placing them in `params` is refused (`agents/watcher_agent.py:690-711`; `chapter_13_triggers.py:616` is such a caller) | TB Run-4 routing ruling §3, §5; patch review R1 Blocker 1 | **Beta position** |
| 10 | Attribution-driven autonomous parameter changes "Not authorized" (`TFM_SYSTEM_MAP_AND_LEARNING_ARCHITECTURE_v1_2.md:277`) | Ch14 **S4** | **Beta position** |
| 11 | `forward_matches`/`reverse_matches` not model features | Ch14 **S5** (`…SYSTEM_MAP…:167-171`, `:276`) | **Beta position** (schema decision required) |
| 12 | Selfplay trend detection is observe-only (`selfplay_orchestrator.py:829`) | Ch14 **S8** | session decision (no TB text found) |
| 13 | "Window-optimizer logic" is reserved human authority (`docs/PIPELINE_BEHAVIOUR_MODEL.md:1004-1005`); `watcher_policies.json` changes are human-only (`:980`) | behaviour model §13 | **Beta/owner position**. Parameters within bounds are adjustable in principle (fact file §0.5: "Autonomy adjusts parameters, never structure") |

**What would have to exist, as options (none built):**
1. A signal: something that relates window choice to graded pool hits. Today the only learner that sees the window
   is Optuna, through its Step-1 objective; the supervised model never does (stop 1).
2. A proposer: one of (a) Optuna itself (exists); (b) a Phase 9B.3 `propose_transform_update` that emits a window
   proposal (stop 4); (c) a Chapter 13 trigger that opens a new Step-1 search with a chosen window (stops 3, 6, 7);
   (d) selfplay outer episodes driving Step 1 (stops 2, 3).
3. An authority path for a non-operator proposer: today only the operator pin carries one (stop 9). Lifting stop 9
   or 7 or 8 is an owner decision on a Beta position.
4. The validator: whatever proposes must land on `check_window_choice` (Part 2), so a learner cannot place a window
   outside the derived domain or the recorded window-size bounds, and a refusal names what it refused.
5. A record: the trial record already carries `window_anchor` and `generator_phase` (Brief I); the search ledger
   records pinned geometry (`utils/step1_search_ledger.py:194-218`).

## I. `offset` consumer census (v1.1 §9; Brief II audit, started here, not finished)

`/bin/grep -rlw offset` over `*.py`/`*.sh`, tests and docs excluded: 185 files, most of them archival copies
(`prng_registry_before_*`, `*_INTEGRATED.py`, `apply_s*.py`). Live Steps 1-6 readers of the name as the window:

| File:line | Reads `offset` as | Disposition in Part 2 |
|---|---|---|
| `window_optimizer.py:74`, `:112-127`, `:142-143`, `:165-166`, `:231`, `:242`, `:1026`, `:1138`, `:1379` | search bound, `WindowConfig` field, output config, TRSE entry, config-file route | unchanged (Q5); derived mode samples into the same field |
| `window_optimizer_bayesian.py:543-545`, `:582`, `:816`, `:1049-1065`, `:1138` | Optuna dimension, warm start, legacy vector/random paths | derived branch added; legacy paths unchanged |
| `window_optimizer_integration_final.py:258`, `:266`, `:296`, `:1248`, `:2175`, `:2367`, `:2396-2583`, `:2842-2846`, `:3001`, `:3329`, `:3388` | residue load (PWC/miner), canonical record, miner ingress, result filenames, NP2 sampling | unchanged; NP2 refused when on |
| `utils/canonical_records.py:225` | array 4 = `ctx["window_anchor"]` (ruled wire name) | unchanged |
| `persistent_worker_coordinator.py:435-470`, `:2771-2972`; `sieve_filter.py:184`, `:543-628`, `:873` | PWC anchor (new schema) and legacy clamp | unchanged; derived refused off RANGE-MINER |
| `sieve_gpu_worker.py:157` (GOVERNED) | old-schema job `offset` | unchanged (Brief II audit item per code-review ruling) |
| `coordinator.py`, `zmq_sqlite_coordinator.py:805-1172`, `convert_survivors_to_binary.py:135` | legacy jobs; legacy converter (D3.0-B, do not invoke) | unchanged |
| `agents/watcher_agent.py:177` | freshness keys of `optimal_window_config*.json` | unchanged |
| `database_system.py:502`, `web_dashboard.py:1724-1991` | display / record | unchanged |
| `survivor_scorer.py:108`, `:373`, `:707` | `max_offset` scorer config; `best_offset` constant | not the window; unchanged |
| `agents/registry/parameter_registry.py:282-289`; `agent_manifests/window_optimizer.json:89-97`; `config_manifests/parameter_registry.json:38-43` | declarations with wrong text or inert bounds | unchanged (Q5) |

Not traced this session: `prediction_generator.py` and the Step 3 shell scripts did not match the key-read pattern;
the Step 3 law `offset = train_history_len` is out of scope by ruling.
