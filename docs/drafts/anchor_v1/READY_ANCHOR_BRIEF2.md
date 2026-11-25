# READY — window-anchor Brief II: derived anchor domain + holdout guard (round 2)

**Date:** 2026-11-23 · **Host:** VM 101, `michael` · **Worktree:** `~/tfm_worktrees/anchor-brief2`, base `41ad509`
**Status:** built and self-tested on CPU fixtures. Nothing committed, pushed, landed, relayed or deployed. No GPU,
rig, Step-1 run or pipeline step. Part 1: `docs/drafts/anchor_v1/STATUS_AND_PLAN_ANCHOR.md`.
**Round 2** answers the lead's four points of 2026-11-23 (after the owner's 11:25 ruling). Nothing else changed.

## Prior record

Searches and documents: `STATUS_AND_PLAN_ANCHOR.md` "Prior record" (this session, `41ad509`). For round 2, also read:
`logs/e2e_session/OWNER_RULING_brief2_goes_ahead_20261123.txt` (main tree); Optuna 4.4.0
`optuna/samplers/_tpe/sampler.py:373-470` and `optuna/search_space/intersection.py:46-63` (installed package);
`window_optimizer_bayesian.py:449-453` (production `TPESampler(..., multivariate=True)`), `:709` (sqlite storage).

| Item | Label |
|---|---|
| Brief II before Brief I acceptance | **Ruled by the owner 2026-11-23 11:25: "Yes"**; lifts `TB_RULING_BRIEF_I_PRODUCTION_SHAPE_FAILURE.md:6-7`. Recorded in the status document (Q1) |
| `derived_anchor_domain`, `utils/window_anchor_domain.py` | NEW code carrying out v1.1 §4.2-4.3, default off |
| `anchor_holdout_guard` | NEW, at the lead's instruction (status Q2). Not in v1.1; it narrows the v1.1 domain only |
| Keeping `search_bounds.offset` as the default domain | **CHANGES A RULING** (v1.1 §4.3), required until the owner rules (status Q3) |
| M8 | ALREADY RECORDED obligation; DETECTED |
| How TPE samples `offset` when its range moves | NEW measurement (below), no verdict |

Not searched: host outside the worktree, ser8, private repo, project files, rigs.

## The lead's four points

**(1) `_wad_domain` could be unbound.** Fixed. `_wad_domain = None` is now a statement of `optimize_window`'s own body
(`window_optimizer_integration_final.py:2815`), bound before every branch and before every read. Reads go through
`_attach_window_anchor_domain_record` (`:1740-1745`), which leaves results untouched when the domain is `None`. If the
switch is on and no domain was resolved, the run fails closed before any trial (`:3659-3662`).
- Gate **G-NP2-BIND** (AST of the live function): exactly one function-level `_wad_domain = None`, at a lower line
  than every load. The helper is `None`-safe.
- Mutant **M11** deletes that line → DETECTED at `NP2-BIND-init`.
- For the record: the old code could not hit the `NameError` in practice, because the switch refuses `n_parallel ≠ 1` before
  `_np2_complete = n_parallel > 1` (`:3464`) is computed. But the safety then rested on a refusal about 650 lines earlier (`:2816-2830`). It
  is now structural.

**(2) The saved results file lacked the record.** Fixed. The record is attached before `optimizer.save_results`
(`:3706-3709`).
- Gate **G-RESULTS-FILE**: AST order (attach before save in one block); then the real `WindowOptimizer.save_results`
  writes a file whose `window_anchor_domain` equals the domain's run record, guard fields included.
- Mutant **M12** swaps the two lines → DETECTED at `RESFILE-order`.

**(3) Varying `offset` ranges on the real study storage, and how TPE treats `offset`.** Measured. Gate
**G-TPE-STORAGE** runs the production `BayesianOptimization.search` → `OptunaBayesianSearch.run_optimization`. Each
run creates its study on the sqlite storage Step 1 uses (`sqlite:///<checkout>/optuna_studies/window_opt_<t>.db`,
`window_optimizer_bayesian.py:709`), then reloads it from that file. The three `TPESampler` entry points are wrapped
to record where `offset` came from in each trial. Sampler attributes were read off the live object.

| arm | storage | trials | distinct `offset` ranges | `offset` source per trial | trials with `offset` in the joint (relative) search space | "sampled independently" warnings |
|---|---|---|---|---|---|---|
| derived | sqlite, reloaded from file | 24, all COMPLETE | 19 | random startup 3 · **independent TPE 21** · joint 0 | 1 | 21 |
| bounded (today) | sqlite, reloaded from file | 24, all COMPLETE | 1 | random startup 3 · **joint multivariate TPE 21** | 23 | 0 |

Sampler, read off the object: `TPESampler`, Optuna **4.4.0**, `multivariate=True`, `group=False`,
`n_startup_trials=3`. Log: `data/anchor_v1_logs/run5.full.log` (`MEASURED` lines).

**Measured fact, no verdict:** the storage accepts trials whose `offset` distributions differ, and reloads them.
With the switch on, Optuna's intersection search space drops `offset`, because its distribution differs between
trials (`intersection.py:46-63`). After the 3 startup trials, `offset` is sampled by **independent** TPE, apart from
the other parameters, and Optuna logs its "sampled independently instead of being sampled by multivariate TPE"
warning (`sampler.py:446-456`, message at `:451`). With the switch off, `offset` is sampled **jointly** with the other parameters. The
other parameters keep fixed ranges and stay in the joint space in both arms. Whether this matters for hit rate is a
separate measurement. One that would settle it: two derived studies, one as built and one with `offset`
reparameterised to a fixed range (for example a fraction of the domain), compared on graded pool hits. Not built.

**(4) Holdout guard.** Built as a second declared setting, **`anchor_holdout_guard`** (bool, default false in code).
- **Meaning:** with the guard on, the anchor is capped so the whole window `[anchor, anchor + w)` ends before the
  first holdout record of Step 1's train/holdout split for that session filter:
  - `effective_max = min(effective_max, N_train(sessions) − w)`;
  - `N_train(sessions)` = that session filter's records in `data[:split_point]`;
  - `split_point = window_optimizer.step1_holdout_split_point(len(data))`.
- **One split definition:** the split is now named once (`STEP1_TRAIN_FRACTION = 0.8`, `window_optimizer.py:681-686`).
  Both split lines, `:1201` and the `--config-file` route `:1501`, call it, with the same value as the old literal
  `int(len * 0.8)` (gate G-SPLIT-SAME, checked for every n from 0 to 20,000 in steps of 7).
- **Validated, never clamped:** an anchor past the guard is refused, naming `holdout_guard=True`, `N_train`, the
  split point and the dataset sha256.
- **Requires the derived domain:** the guard on with the derived domain off is refused in all three places
  (validator, CLI `parser.error`, `optimize_window`), rather than ignored.
- **Recorded:**
  - run record: `holdout_guard`, `holdout_split_point`, `n_train`, the rule;
  - each trial's `user_attrs`: `holdout_guard`, `n_train`, `holdout_split_point`, the anchor range;
  - the search key and study tag become `derived_holdout_guard`, so guarded and unguarded arms never share a study
    or a search.
- **Route:** the same three hops (manifest `default_params` + `args_map` + `param_docs`, `--anchor-holdout-guard`,
  `optimize_window(anchor_holdout_guard=…)`).
- **Gates:**
  - **G-HOLDOUT**: per session filter and window 6/12/50, the maximum is `N_train − w`; the window's last record is a
    train record and the next record is the first holdout record; one past is refused; the records are right; the
    guard off leaves the range unchanged.
  - **G-HOLDOUT-OBJ**: real search, every trial's range is the guarded one; a pin at the guarded maximum runs and
    one past it is refused before the objective is called.
- **Mutants:** **M9** (guard ignored) and **M10** (off by one, + 1) are both DETECTED at `HOLD-max`.

**Proposed default (for the lead to bring to the owner): `anchor_holdout_guard = true` whenever
`derived_anchor_domain = true`.** Reason: the yardstick is pool hits on draws nothing upstream saw. With the guard
off, a derived-domain sieve window can sit on `holdout_history.json` draws.

**The cost, measured on the live dataset (status §G):**

| session filter | first holdout record (filtered index) | guarded max anchor at W12 | v1.1 governed era starts at |
|---|---|---|---|
| midday | 6,707 | 6,695 | 6,791 |
| evening | 7,747 | 7,735 | 7,830 |
| midday + evening | 14,454 | 14,442 | — |

So **with the guard on, no governed-era window is reachable**, and the D.1 experiment's governed arm cannot run. D.1
needs either the guard off with grading moved to draws after the sieve window (e2e map Q1), or a different train/holdout
split. That is the owner's decision. Both values work today.

Because a `true` bool is emitted into WATCHER's argv, putting `true` in `default_params` while
`derived_anchor_domain` is `false` would make every default run refuse ("guard needs the derived domain"). So the
manifest default stays `false`; the owner sets both together.

## What was built (complete list)

| Hop / part | Where | Off | On |
|---|---|---|---|
| 1 | `agent_manifests/window_optimizer.json`: `default_params` `derived_anchor_domain: false`, `anchor_holdout_guard: false`; `args_map`; `param_docs` (decoded diff: exactly these 6 keys added) | false bools omitted from argv | flags emitted |
| 2 | `window_optimizer.py`: split helper `:678-686`; seam `:462-465`; signature `:780-781`; validators `:842-846`; conditional forward `:1008-1011`; output record `:1065-1069`; flags `:1686-1698`; CLI edges `:1864-1882`; call `:2068-2069` | nothing forwarded; file unchanged | forwarded; `window_anchor_domain` block in `optimal_window_config.json` |
| 3 | `window_optimizer_integration_final.py`: helper `:1740-1745`; study arm `:1970-1973`; signature `:2718-2719`; validators, tag, `_wad_domain = None`, edges, search arm `:2799-2835`; chunk tag `:2841`; resolution `:3638-3662`; record before save `:3706-3709` | refuses only a tagged search or study | resolves, sets `_window_anchor_domain` |
| objective | `window_optimizer_bayesian.py:543-570` | original two `suggest_int` calls | window, sessions, then `offset` in `[min, effective_max]`; `check_window_choice` on every trial |
| search id | `utils/step1_search_ledger.py:256-278` | digest unchanged | `window_anchor_domain` = tag, plus `window_anchor_bounds` |
| new | `utils/window_anchor_domain.py` (359 lines) | — | resolver, ranges, control era, guard, validators, arm checks |

**The one route for any proposer:** `AnchorDomain.check_window_choice(window_size, anchor, sessions)`. Optuna samples
and operator pins pass through it; a later learner would call the same function. It grants no authority.

## Tests

**`tests/test_window_anchor_brief2.py` (965 lines): 18/18 gates, 12/12 mutants DETECTED, `COMPLETION SENTINEL: PASS`.**
**Run time 71.6 s wall; peak RSS 224.5 MB; 79% CPU (effectively one core).** The searches run one after another
because study names are per-second, and suites flake when run concurrently. Log: `data/anchor_v1_logs/run5.full.log`;
time: `run5.time`. Main-tree `~/distributed_prng_analysis/optuna_studies/` held 125 files before and after every run.

| Gate | Proves |
|---|---|
| G-AC3 | 149 at w50 is in the derived domain `[0,150]`, not in control_era `[0,100]`; cap 60 gives `[0,60]` |
| G-DOMAIN | three session filters × w 6/12/50: range `[0, N−w]`; agrees with the worker's `load_residue_window` at max and max+1 |
| G-REFUSE | out-of-domain refusals name anchor, domain, sessions, sha256; non-int refused |
| G-CAP | cap narrows, never widens; empty domain and malformed config refused |
| G-RESOLVE | counts and sha256; wrong frozen digest refused |
| G-OBJ-DERIVED | real search: each trial's `offset` range is `[0, N−w]` for its own window and sessions |
| G-PIN | pins at 448/150/254/101 run; 151@w50 and window 60 refused before the objective |
| G-DEFAULT-OBJ | switch off: a seeded study's trial parameters are identical to HEAD's code |
| G-DEFAULT-SEARCH | switch off or absent: `presented_configuration` is identical to HEAD's |
| G-ARM | bounded / derived / derived_holdout_guard: study and search refuse every other arm |
| G-NP2-BIND | point 1 |
| G-RESULTS-FILE | point 2 |
| G-HOLDOUT, G-HOLDOUT-OBJ | point 4 |
| G-TPE-STORAGE | point 3 (sqlite, reloaded; measurement printed) |
| G-SPLIT-SAME | both split lines use the one helper; value equals `int(n*0.8)` |
| G-ROUTE | both settings, three hops each (`inspect.signature` + AST) |
| G-CLI | exit 2: derived with `--config-file`; derived without `--use-range-miner`; guard without derived |

| Mutant | Mutation | Credited assertion |
|---|---|---|
| M8 | control-era ceiling 100 → envelope (100 + w − 1) | G-AC3 `AC3-149-not-in-control` |
| M1 | clamp restored | G-REFUSE `REFUSE-raised` |
| M2 | cap widens | G-CAP `CAP-never-widens` |
| M3 | session filter dropped from the count | G-RESOLVE `RESOLVE-proof` |
| M4 | derived branch samples the bounded range | G-OBJ-DERIVED `OBJ-dist-is-derived` |
| M5 | seam forwarding removed | G-OBJ-DERIVED `OBJ-dist-is-derived` |
| M6 | pinned choice not validated | G-PIN `PIN-past-max` |
| M7 | search key added when off | G-DEFAULT-SEARCH `DEFSEARCH-identical` |
| M9 | holdout cap ignored | G-HOLDOUT `HOLD-max` |
| M10 | holdout cap off by one | G-HOLDOUT `HOLD-max` |
| M11 | `_wad_domain = None` removed (structural) | G-NP2-BIND `NP2-BIND-init` |
| M12 | record attached after the save (structural) | G-RESULTS-FILE `RESFILE-order` |

**How each mutant is run:**
- Each one runs in a fresh interpreter.
- The exact text is replaced exactly once (`MUTANT APPLIED`).
- A marker proves execution. For M11 and M12, which change code the gate reads rather than code that runs, the marker
  proves the AST gate parsed the mutated source, as Brief I's structural M15 did.
- Detection requires the gate's own `FAIL <gate> [<assertion id>]` line.

### Differential battery (17 existing suites that import the changed modules)

- **Round 1:** live and a `git archive HEAD` copy, sequential (logs `data/anchor_v1_logs/battery/*.head.log`,
  `*.live.log`, `*.live2.log`).
- **Round 2:** the live side rerun on the final files (`*.r2.log`, `summary_r2.txt`). It was diffed against the HEAD
  logs after removing paths, durations, digests, PIDs, temp dirs and study names.
- **Result: no suite changes verdict.** The residual differences are:
  - log-message source line numbers (`LCH-7 disposition@966` vs `@970`);
  - set print order (Gate 22, `G-ALLOWLIST-EXACT`);
  - the 14-character `data/headtree/` prefix (`test_stage4_miner_launch` 529 vs 515 characters);
  - the checkout path `test_step1_bayes_storage_path` prints.
- **Pre-existing reds are identical on both sides:**
  - `test_s172_run4_routing_patch` 21/22 (its stub DB lacks `select_step1_search`);
  - `test_step1_search_record_stage1`, `test_s172_miner_seed_start` and `test_seed_domain_cursor_amendment`
    (no `daily3.json` pointer manifest in a fresh worktree);
  - `test_chapter1_p0_corrections`;
  - `test_s172_attempt6_remediation` 77/78 (the recorded RXP-1/1 flake);
  - `test_s172_phase4_coordinator` Gate 22, on the uncommitted `.py` files (expected; no allowlist change).
- **Caveat:** both sides ran inside this worktree, so suites that read `git status` saw the same dirty state.
- **Not run:**
  - `tests/gate_s172_prod_shape.py` and `tests/capture_run4_clean_control.py`;
  - the 75 suites that do not import the changed modules.

## TODO-E2E list

| # | Shortcut | Anchor |
|---|---|---|
| A1 | `WindowConfig.offset`, the Optuna name `offset` and `_O{offset}` filenames keep the legacy name (= window anchor) | `window_optimizer.py:112`; objective; v1.1 §4.3 |
| A2 | `search_bounds.offset` kept as the bounded default | `distributed_config.json:80-83` |
| A3 | NP2 objective not migrated; refused when on | `window_optimizer_integration_final.py:2998-3003` (pre-edit numbering) |
| A4 | D6.2 run context does not carry the arm; `--resume-checkpoint` refused when on | `_prepare_checkpoint_run_context` |
| A5 | PWC TCP not enabled for the derived domain | `persistent_worker_coordinator.py:435-447`; `sieve_filter.py:614` |
| A6 | `parameter_registry.py` `offset` (max 500) and manifest `parameter_bounds.offset` (max 2000) untouched | `agents/registry/parameter_registry.py:282-289`; manifest `:89-97` |
| A7 | `governed_era` resolver and D.1 reach demo (AC4) not built | v1.1 §4.2, §7 AC4 |
| A8 | A derived study is refused by a bounded run only at its first trial (study tag), not at study load | `anchor_domain_study_check` |
| A9 | `--config-file` route's silent defaults (offset 100, window 1024) untouched | `window_optimizer.py:1409-1410` |
| A10 | Bounded mode still lets a pin outside `[0,100]` / `[6,50]` through on Optuna's warning (today's behaviour) | `optuna/trial/_trial.py:647-660` |
| A11 | `search_bounds.window_anchor` read from config if present; no CLI override for `min` / `max_cap` | `load_window_anchor_bounds` |
| A12 | Derived `offset` is sampled by independent TPE, not jointly (measured, point 3); a fixed-range reparameterisation arm not built | `window_optimizer_bayesian.py:543-570` |
| A13 | The guard's `N_train` counts with the worker's predicate over `data[:split]`; unlike `N_filtered`, it has no second proof against `load_residue_window`, which cannot see the split | `resolve_anchor_domain` |
| A14 | The split stays a run-global 80/20 of the combined history; per-session or configurable splits are e2e Q1 | `window_optimizer.py:681-686` |

## Landing

- **Owner lands by hand** (`scripts/owner_land.py:84-87`): `agent_manifests/window_optimizer.json`.
- **owner_land group:** `utils/window_anchor_domain.py` (new), `tests/test_window_anchor_brief2.py` (new),
  `utils/step1_search_ledger.py`, `window_optimizer.py`, `window_optimizer_bayesian.py`,
  `window_optimizer_integration_final.py`, `docs/drafts/anchor_v1/*`.
- Do **not** land `data/`: it holds the untracked logs, and it is not ignored in this repo.
- Gate 22 reds on the two new `.py` files until they are committed. This is expected; no allowlist change.
- No `miner/` file changed: no fleet redeploy or parity re-proof.
- Hashes: `code.diff` `2abdfa0db37cea005fc40f02cad0aa30014a82fef9522bf55776d398000e031b` (5 files, +217/−12);
  `utils/window_anchor_domain.py` `a9fc243d04b21813549fc5982219f3438f2197f95cfb6f0f532d5024ffc58e01`;
  `tests/test_window_anchor_brief2.py` `db06efd7ad9b42ac20b8c1a3431ea2789e5ac16911dc0cd711170410ebd5c256`.
- The Beta before-commit request is drafted and not sent:
  `docs/drafts/anchor_v1/X-WINDOW-ANCHOR-II_R1_ALPHA_REVIEW_REQUEST_derived_anchor_domain.md`.

## Open questions

Status document §F. Q1 is answered (owner, 11:25). Q2 is built; it now needs only the default (proposed above:
guard on with the derived domain, at the cost of no governed-era reach). Q3-Q7 are open.
