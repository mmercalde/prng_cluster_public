# Design: a WATCHER route for the two LL-2 options (Step 3)

Task: `logs/e2e_session/TASK_REG_LL2_ROUTE_DESIGN_20261125.txt` (lead, 18:54). READ ONLY: no code changed, no suite run, no commit. Base: main `7f4505a`. It answers questions 2 and 4 of `READY_REGISTRY_REPAIR_LAND_20261125.md`.

**Yardstick:** this is about *reach* only (can a value set in WATCHER arrive at the worker). It makes no hit-rate claim. Whether `per_seed_sieve_features=on` helps is the walk-forward comparison `01_sieve_on` against `00_base`, which the owner already ruled (`OWNER_RULINGS_pending_list_20261124.txt:22-25`).

## Plain summary

1. **Today a value for either LL-2 option is dropped at the first hop.** WATCHER only passes keys that `full_scoring.json` declares, and it declares neither (`agents/watcher_agent.py:2026-2034`).
2. **The wrapper would drop them next.** `run_step3_full_scoring.sh` ignores unknown options (`:120-123`). Its job generator builds each worker's argv from a fixed list (`generate_step3_scoring_jobs.py:240-255`). The coordinator passes job args through unchanged, and the worker already accepts both flags (`full_scoring_worker.py:570-575`).
3. **The data is already there.** The NPZ's `forward_matches`/`reverse_matches` ride in every chunk file (`:62-103`), and the worker fails closed if they are missing (`survivor_scorer.py:120-136`). Only the *setting* has no path.
4. **Option A (recommended).** Add the two named flags to the wrapper and the job generator, each forwarded only when not the default, plus one manifest entry. That is two pipeline files and one hand-landed file. No governed file. `watcher_agent.py`, `scripts_coordinator.py` and the worker are untouched.
5. **Option B, one settings file**, and **option C, a generic pass-through**, are possible, and I do not recommend them:
   - B invents a second vocabulary next to the registry;
   - C repeats the pass-through risks Beta objected to on 2026-11-25 (abbreviations, overriding set options).
6. **Defaults.** With the manifest value `null`, every hop is byte-identical at default: WATCHER's argv, the job file, the worker's argv and its output. With the literal `"off"`/`"legacy6"`, only the wrapper's own argv grows (by 4 tokens).
7. **Overlap.** step2-repair edits exactly the same blocks of the same two files for its three Step 3 modes. Land one after the other, not in parallel. step2-repair is also based on `41ad509`, before LL-2 reached the worker.
8. **Agent name.** Pointing the Step 3 agent name at `full_scoring_worker.py` changes nothing at run time, because `get_registry_for_agent` has no caller outside tests. **Recommendation:** do it in the same landing as option A, not before, and leave Step 2 alone until step2-repair registers its script.

## Prior record

**Searches** (worktree at `7f4505a`; read only in `~/tfm_worktrees/step2-repair`):

```
sed -n 236,263p docs/drafts/loop_v1/AUTONOMY_SWITCH_MAP_20261123.md          # G1–G19
/bin/grep -lnE 'LL-2|per.seed.sieve|fallback.metadata' ~/tfm_worktrees/step2-repair/docs/drafts/step2_repair_v1/*.md   # none
/bin/grep -nE 'run_step3_full_scoring|LL-' .../READY_STEP2_WIRING_STAGE3_20261124.md
git -C ~/tfm_worktrees/step2-repair diff $(merge-base HEAD 7f4505a) HEAD -- run_step3_full_scoring.sh generate_step3_scoring_jobs.py full_scoring_worker.py ...
git grep -n 'get_registry_for_agent\|get_registry(' -- '*.py'                # callers
grep -rn 'per-seed-sieve-features|per_seed_sieve_features' --include=*.py --include=*.sh .
cat logs/e2e_session/TASK_LAND_WF_VOTE_20261125_ADDENDUM_4.txt                # pass-through ruling
```

**Found:**
- **`AUTONOMY_SWITCH_MAP_20261123.md`:**
  - **G5:** grammars take only numbers. A choice value such as `on` cannot be proposed by the LLM until G5 lands. **ALREADY RECORDED**, and it is not this note.
  - **G6:** reach routes for the drivers.
  - **G7:** an apply step (B12, held by ruling S3).
  - **G8:** record applied changes.
  - **G16:** "the step-3 wrapper drops four declared keys silently (`run_step3_full_scoring.sh:120-123`)" (SESSION_NOTES_20260118).
  - **G4:** the LLM sees the registry only on the evaluation path, which `file_exists` bypasses.
- **`READY_LAND_LL_SWITCHES_20261125.md` §(b) `:549-556`** and **`X-LL-SWITCHES_R1` request, known limit 1:** LL-2 has no WATCHER route. **ALREADY RECORDED.**
- **step2-repair:**
  - `READY_STEP2_WIRING_STAGE3_20261124.md:113` (the launcher's `--step2-apply-mode`, validated, exit 2, passed to the generator) and `:198` ("Once a manifest names `step2_apply_mode`, WATCHER passes `--step2-apply-mode off`");
  - its branch diff at `930fa8b` against `41ad509`: `run_step3_full_scoring.sh` +98, `generate_step3_scoring_jobs.py` +66, `full_scoring_worker.py` +120, `survivor_scorer.py` +19;
  - **no mention of LL-2** in any of its notes.
- **The Step 3 manifest**, `agent_manifests/full_scoring.json`:
  - `default_params`: survivors, config, train_history, holdout_history, chunk_size, jobs_file, prng_type, mod, batch_size, output_file;
  - actions: `generate_full_scoring_jobs.py` (primary), `full_scoring_worker.py` (`run_distributed`), `aggregate_scoring_results.py`;
  - `evaluation_type: file_exists`.
- **Pass-through ruling:** `TASK_LAND_WF_VOTE_20261125_ADDENDUM_4.txt` (Fix 2: no abbreviated options; Fix 3: a pass-through may not carry an option the caller sets itself), from Beta X-E2E-WF-VOTE R1's two LIVE objections.

**Labels:**

| Item | Label |
|---|---|
| LL-2 has no WATCHER route | ALREADY RECORDED (above) |
| The wrapper drops unknown options | ALREADY RECORDED (G16) |
| The hop-by-hop trace, the three options, the defaults analysis, the overlap with step2-repair | NEW |
| The primary action of `full_scoring.json` (`generate_full_scoring_jobs.py`) is not what the wrapper runs, so `ParameterContext` looks up the wrong script for Step 3 | ALREADY RECORDED in substance (`REGISTRY_REPAIR_20261124.md` class N: "not what the Step 3 wrapper runs"); its effect on the agent-name question is NEW |
| A wrong line number in my earlier notes (the wrapper's unknown-option drop, given as `:34-37`) | NEW correction. It is `:120-123`. Fixed today in `READY_REGISTRY_REPAIR_LAND_20261125.md`, `OWNER_MANIFEST_COMMANDS.txt` and `DRAFT_X-REGISTRY-REPAIR_R1_REQUEST.md`, with their sha256 references updated |
| CHANGES A RULING | none |

**Not searched:** the host, ser8 archives, the private repository, project files not in the repository, worktrees other than step2-repair.

## 1. The hops today, and where the value is dropped

| # | Hop | What carries what | LL-2 today |
|---|---|---|---|
| 0 | The operator or a proposal puts `per_seed_sieve_features` in WATCHER's `params` | — | — |
| 1 | `run_step` merges `params` into `final_params`, but **only keys that `full_scoring.json` `default_params` declares** (`agents/watcher_agent.py:2026-2034`) | the 10 declared keys | **DROPPED HERE** ("Skipping param … not declared", `:2034`) |
| 2 | Command builder: `bash run_step3_full_scoring.sh` (`STEP_SCRIPTS[3]`, `:409`) plus `--key value` for each non-empty value (`:2378-2427`); `output_file` popped (`:2052`) | `--survivors … --config … --train-history … --holdout-history … --chunk-size 1000 --jobs-file … --prng-type java_lcg --mod 1000 --batch-size 100` | would be passed, if declared |
| 3 | The wrapper's `case` list accepts `--survivors --train-history --holdout-history --config --chunk-size --forward-survivors --reverse-survivors --dry-run` (`run_step3_full_scoring.sh:86-125`); anything else prints "Note: Ignoring unknown option" (`:120-123`) | 5 of the 9 WATCHER keys; `jobs-file`, `prng-type`, `mod`, `batch-size` are already dropped here (G16) | **would be DROPPED HERE** |
| 4 | The wrapper builds `GEN_CMD` for `generate_step3_scoring_jobs.py` from its own variables (`:160-178`) | survivors, histories, chunk size, output file, config, forward/reverse survivors | not carried |
| 5 | The generator's argparse (`generate_step3_scoring_jobs.py:296-326`) → `generate_jobs(...)` (`:147-159`, call `:375-387`) | no LL-2 parameter | not carried |
| 6 | `generate_jobs` builds each job's `args` from a fixed list: seeds file, histories, output file, `--prng-type`, `--mod`, and the optional forward/reverse survivors (`:240-255`); written to `scoring_jobs.json` (`:261-268`, `:390`) | the worker's argv | **would be DROPPED HERE** |
| 7 | `scripts_coordinator.py` loads the jobs and passes `args` through. Its only rewrite is `--output-file` (`:450-483`, `_update_args_output_path` `:486+`) | args unchanged | passes whatever is there |
| 8 | `full_scoring_worker.py` argparse already has `--per-seed-sieve-features {off,on}` and `--fallback-metadata-merge {legacy6,shared}` (`:570-575`), passed to `score_survivors` (`:681-682`) | — | **accepts both** |
| 9 | Data: `extract_survivors_full` keeps every NPZ array per survivor in the chunk (`generate_step3_scoring_jobs.py:62-103`), including `forward_matches`/`reverse_matches` (`convert_survivors_to_binary.py:52-53`, `:123-131`). The worker reads chunk metadata (`full_scoring_worker.py:609-613`) and, when `on`, fails closed if any seed lacks a finite value (`survivor_scorer.py:120-136`). | — | **already there** |
| 10 | Output: the worker stamps `feature_settings` on each row **only when not default** (`full_scoring_worker.py:516-520`). The wrapper's aggregation copies rows whole (`run_step3_full_scoring.sh:362-371`, written `:406-408`) | — | the stamp survives |

**Fact:** the route is missing at hops 1, 3–6 only. Hops 7–10 need nothing.

## 2. Options

### Option A: two named flags through the wrapper and the job generator (recommended)

**What it does:**
1. **Manifest** (`agent_manifests/full_scoring.json`, hand-landed): add `per_seed_sieve_features` and `fallback_metadata_merge` to `default_params`. For the value, see "Defaults".
2. **Wrapper** (`run_step3_full_scoring.sh`, pipeline):
   - two variables, defaulting to `off` and `legacy6`;
   - two `case` arms;
   - a value check that exits 2 on anything else, in the same form as step2-repair's checks;
   - append `--per-seed-sieve-features X` / `--fallback-metadata-merge Y` to `GEN_CMD` **only when not the default**;
   - one `echo` line in the header block, printed only when not default, so default stdout stays the same.
3. **Generator** (`generate_step3_scoring_jobs.py`, pipeline):
   - the same two argparse options with `choices=` and the same defaults;
   - two `generate_jobs` keyword parameters with defaults;
   - `args.extend([...])` for each **only when not the default** (after `:255`).

**Files touched:**

| File | Class | Change |
|---|---|---|
| `agent_manifests/full_scoring.json` | **hand-landed** (`scripts/owner_land.py:87`) | +2 `default_params` keys |
| `run_step3_full_scoring.sh` | pipeline (not governed, not hand-landed) | ~+20 lines |
| `generate_step3_scoring_jobs.py` | pipeline (not governed) | ~+15 lines |
| `agents/watcher_agent.py`, `scripts_coordinator.py`, `full_scoring_worker.py`, `survivor_scorer.py` | — | **untouched** |

None of these is in `GOVERNED_FILES` (`scripts/gate12_parity_gate.py:174-185`). On Beta classes: the coordinator and the worker are not edited, so this is not a before-commit class. I would still ask Beta for a courtesy review, because it changes the Step 3 launch path.

**Byte-identical at default:**
- with the manifest value `null` (see below): WATCHER's argv to the wrapper (the builder skips None, `agents/watcher_agent.py:2424-2425`);
- the wrapper's `GEN_CMD`;
- `scoring_jobs.json`;
- each worker's argv;
- `survivors_with_scores.json` (no stamp at default, `:516`).

**Defaults: the manifest value.**

| Manifest value | WATCHER → wrapper argv | Job file and worker argv | The `A-LL-MANIFEST` rule in `tests/test_registry_ll_register.py` |
|---|---|---|---|
| `null` | identical | identical | must allow `null` for a choice whose code default is a string, meaning "use the wrapper's default". A gate must then check that the wrapper's default equals the worker's argparse default |
| `"off"` / `"legacy6"` | +4 tokens (WATCHER passes non-empty strings, `:2427`; step2-repair notes the same, `READY_STEP2_WIRING_STAGE3:198`) | identical (the wrapper forwards only non-defaults) | holds as written |

**I recommend `null`.** It keeps every hop identical, which is the standard the LL-1 and LL-3 fragments met.

**Tests it would need** (CPU, fixtures, no rig):
- **Generator, defaults:** with neither option, `scoring_jobs.json` is byte-identical to `7f4505a`'s generator on the same fixture NPZ.
- **Generator, values:** with `on`/`shared`, every job's `args` ends with exactly the two flags. With an illegal value, argparse refuses (exit 2) and no jobs file is written.
- **Wrapper, defaults and values:** run with `--dry-run` (stops after job generation, `:188-193`). Before that exit, `ramdisk_preload.sh` SSHes to the rigs (`:131-136`), so the test needs a stub `ramdisk_preload.sh` on the path, or a sourced-function test of the arguments block only. Gates:
  - `GEN_CMD` identical at default;
  - an illegal value exits 2 before the generator runs;
  - the legal values reach `GEN_CMD`.
- **The WATCHER argv probe for Step 3** (as `data/registry_repair_land/argv_probe.py` does for Steps 5 and 6):
  - identical argv with no params, before and after the manifest entry;
  - `params={"per_seed_sieve_features": "on"}` dropped before, passed after.
- **End to end on one chunk, locally:** a fixture chunk with `forward_matches`/`reverse_matches`, through generator, job args and the worker CLI:
  - `on` gives schema v2 rows with both fields and the stamp;
  - a chunk without them fails closed (the existing `check_per_seed_sieve_metadata`).
- **Mutants:** forward-always; forward-never; a wrong default in the wrapper; a missing `choices`; dropping the value check.

**Overlap with step2-repair:**
- **The same edit points:**
  - the wrapper's variable block, `case` arms, value checks and `GEN_CMD` appends (step2-repair adds `--step2-config-mode`, `--feature-contract-mode`, `--step2-apply-mode` at those exact places);
  - the generator's `generate_jobs` signature, the `args.extend` block after `:255`, its argparse, and the `generate_jobs(...)` call.
  Expect textual conflicts, not semantic ones: the settings are independent.
- **step2-repair's base is `41ad509`, before LL-2 landed** (`7cac105`). Its `full_scoring_worker.py` (+120) and `survivor_scorer.py` (+19) diffs therefore already overlap the LL-2 code on main. That is its rebase to do, not this route's. Option A does not touch either file.
- **Order:** land option A after step2-repair rebases onto main, or rebase it onto option A. Do not develop both at once on the same two files.

### Option B: one Step 3 settings file

The wrapper gets one flag, `--step3-worker-settings FILE`, a JSON object such as `{"--per-seed-sieve-features": "on"}`. The generator validates each key against an allowed list and appends the flags to every job.
- **Files:** the same three as A, plus a declared allowed list.
- **Defaults:** identical when the flag is absent.
- **Pro:** later Step 3 settings, including step2-repair's three modes, need no further wrapper edits.
- **Con:** WATCHER and the registry would carry a *path*, not two choices. So `validate_value` cannot check a value, and an LLM (once G5 lands) cannot propose `on`. That is a second vocabulary beside `ParameterSpec`, which the task rules out ("no new vocabulary format is invented where one exists", `TASK_SWITCH_REGISTER_ADDENDUM_1.txt`).
- **Not recommended.**

### Option C: a generic pass-through (`--worker-extra-args "…"`)

- **Files:** the wrapper and the generator.
- **Defaults:** identical when absent.
- **Con:** it repeats the shape Beta objected to on 2026-11-25. It would need the same refusals the lead required for the harness pass-throughs: no abbreviated options, and no option the caller sets itself (`TASK_LAND_WF_VOTE_20261125_ADDENDUM_4.txt` Fix 2 and Fix 3). And the registry still could not describe it.
- **Not recommended.**

A fourth shape, having WATCHER's Step 3 run the e2e Step 3 harness instead of the wrapper, means changing `STEP_SCRIPTS` in the hand-landed `agents/watcher_agent.py`, and with it the whole Step 3 production path. It is out of proportion to two options. Rejected.

## 3. The Step 3 agent-name question

**Today** `get_registry_for_agent` maps `full_scoring_agent` and `full_scoring` to `full_scoring.py` (`agents/registry/parameter_registry.py`, `agent_to_script`). After the repair that key is gone, so the lookup returns `None`.

**Who calls it:** no production caller (`git grep get_registry_for_agent`: only `agents/registry/__init__.py` exports it, plus tests). The LLM-facing lookup is different. `ParameterContext.build` calls `get_registry(manifest.script_name)` (`agents/parameters/parameter_context.py:181`), and `script_name` is the manifest's *primary action*, which for Step 3 is `generate_full_scoring_jobs.py` (`agents/manifest/agent_manifest.py:367-371`). That script is not what the wrapper runs. So the agent-name mapping does **not** decide what the LLM sees for Step 3.

**A wrapper can't be a registry key either.** `get_registry` appends `.py` to a name without it (`parameter_registry.py:414-428`), so `run_step3_full_scoring.sh` cannot be one. The flags are defined in `full_scoring_worker.py`, so that is the right key, as the registration draft has it.

| Answer | Consequence |
|---|---|
| **Point it to `full_scoring_worker.py`** | `get_registry_for_agent("full_scoring_agent")` returns the two LL-2 specs. Nothing changes at run time (no caller). It is a code change outside `PARAMETER_REGISTRIES`, so `A-RR-OUTSIDE` and `A-LL-ADDONLY` would need to allow exactly that line. Done before a route exists, a future caller would be told these are Step 3's adjustable settings while WATCHER cannot pass them. |
| **Leave it** | It returns `None`, which matches "no route" today. The mapping keeps naming a key that no longer exists, so it is stale but harmless. |
| **(Also possible) delete the two dead mappings, Step 2 and Step 3** | The same `None` as leaving them, without the stale names. Step 2 has no key until step2-repair registers its script (`generate_scorer_jobs.py`). |

**Recommendation:** point Step 3 to `full_scoring_worker.py` **in the same landing as option A**, not before. Leave Step 2 until step2-repair registers its own script. Separately, the manifest's primary action naming `generate_full_scoring_jobs.py` is what really keeps Step 3 out of the LLM's view (G4). Changing it is a hand-landed manifest edit with its own consequences (`args_map` would describe the wrapper), and it is outside this note.

## 4. Recommendation

1. **Option A**, with `null` manifest values. Land it after step2-repair rebases onto main, or with it rebased onto A, never developed in parallel on the same two files.
2. Agent-name remap in the same landing.
3. A courtesy Beta review, because it is the Step 3 launch path.
4. **Before any GPU use, the CPU proof:** the generator byte-identity on a fixture NPZ, the wrapper `--dry-run` with the preload stubbed, and the Step 3 argv probe.
5. **Still separate:** G5, so the LLM can propose `on` (grammar), and G7, an apply step. This route makes the setting *reachable by an operator's `params`*. It does not make WATCHER choose it.
