# CLAUDE CODE REPORT — WATCHER SAFETY REPAIRS (R-1 / R-2 / R-3) — 2026-11-11

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_SAFETY_REPAIRS_20261111.md`
**Base:** `741822b` (HEAD at session start; `git diff 741822b -- agents/` was empty before editing).
**Box:** VM101, as `michael`, venv `~/venvs/torch`. **Not committed** (stop gate §6).
**Result:** new suite `tests/test_watcher_safety_repairs.py` **21/21 PASS, `COMPLETION SENTINEL: PASS`**.
The four regression suites have per-gate lists identical to baseline (the only difference is pytest's wall-clock time).

**Null results / cautions first:**
- §1 found **no contradiction** of the brief's anchors. Two small notes: `def main` is at `:3465`, and `:3578` is where its mode branches begin (`# Handle halt commands`). The `--run-pipeline` halt literal is at `:2778`; `:2777` is the `check_safety()` line before it.
- R-1 **does not reach `dispatch_learning_loop`**. See §5, open item 4. This is a gap in the same fault class, and it is not fixed.
- R-1 **changes behaviour on Step 1's `seed_domain_complete` path**, which returns `success=False` by design. See §4. This needs a Beta read.
- **Deviation from the brief.** §1.5 says to `tee` logs to `/tmp/*.log`. The box contract (`~/CLAUDE.md` hard rule 7) forbids `/tmp` output, and the project file wins. All logs are under `data/watcher_safety_repairs/` (untracked and not for commit; see §6). The suite's own throw-away fixtures use `tempfile.mkdtemp()`, the same as the existing suites, and are removed at exit.

All `:NNN` anchors in §1 are at **base `741822b`**. Live line numbers moved by +18 after line 306 and by +56 after line 938 because of the edits.

---

## 1. Investigation answers (§1)

### 1.1 Every `return` in `WatcherAgent.run_step` (`agents/watcher_agent.py:1705`)

| Line | Condition | `"success": False`? |
|---|---|---|
| `:1725` | no script in `STEP_SCRIPTS` | returns **`None`**. `run_pipeline:2825-2826` converts it to `{"success": False, "error": "No results returned"}` |
| `:1750` | `_ensure_execution_set` returned a dict (fleet unestablished; built at `:1631`) | **yes** |
| `:1758` | HARD preflight failure | **yes** (`blocked_by: preflight_hard_failure`) |
| `:1778` | hard freshness failure | **yes** (`blocked_by: freshness_hard_failure`) |
| `:1791` | output is fresh, so the step is skipped | no: `{"success": True, "skipped": True}` |
| `:1935` | P0.5 dataset authority error | **yes** |
| `:1997` | S145 certified cursor unavailable | **yes** |
| `:2014` | S145 **seed domain complete** (no run generated) | **yes**, and `seed_domain_complete: True` (see §4) |
| `:2074` | S145 pre-dispatch domain wall | **yes** |
| `:2180` | non-zero exit (`:2177`) | **yes**, with `return_code` and `error` = stderr[:500] |
| `:2192` | rc 0 and `_find_results` loaded a results JSON | **whatever that JSON says**: arbitrary content, usually no `success` key |
| `:2195` | rc 0, no results file | no: `{"success": True, "return_code": 0}` |
| `:2203` | `subprocess.TimeoutExpired` | **yes** (`error: "Timeout"`, no `return_code`) |
| `:2209` | any other exception | **yes** (`error: str(e)`) |

Everything from `:2180` on goes through `_step1_stamp_pin_provenance` (`:708`). That function adds only the provenance keys and never touches `success`.

### 1.2 Callers of `evaluate_results`

| Call | `results` source | Can carry `success: False`? |
|---|---|---|
| `:2833` `run_pipeline` | `self.run_step(...)` (`:2822`), with `None` converted at `:2825` | **yes**: every path marked "yes" in 1.1 |
| `:3447` `_process_result_file` (daemon) | `json.load` of a file in the watch dir | **only if the file contains it** |
| `:3790` `--evaluate` CLI | `json.load(args.evaluate)` | only if the file contains it. Normally there is no `success` key, so behaviour is unchanged (R1-D) |
| *(extra)* `agents/watcher_dispatch.py:370` at base (`:400` live) learning loop, heuristic path | `self.run_step(step)` | yes, **but see open item 4**: the caller reads a non-existent `.action` attribute, so the decision is ignored |

### 1.3 `KillSwitch.check_all()` (`agents/safety/kill_switch.py:78`)

- **What it checks:** the halt file (`:109`), consecutive failures (`:135`), retries this step (`:159`), runtime against `max_runtime_minutes` (`:183`), disk free space (`:209`) and memory from `/proc/meminfo` (`:244`).
- **Side effects:** it **rebinds `self.checks`** (`:84`) and **mutates `self.current_level`** through `_update_level` (`:291`). It also does disk and `/proc` I/O.
- `check_safety()` (`:361`) builds a fresh default `KillSwitch()`, so it always uses `/tmp/agent_halt` no matter what `config.halt_file` says.
- **Cheapest "is a halt requested" check:** `os.path.exists(kill_switch.halt_file_path)`. This is exactly the first line of `_check_halt_file` (`:111`), without the file read or the `SafetyCheck` build. R-3 uses it.

### 1.4 Mode branches in `main()` (`:3465`; branches from `:3578`)

| Mode | Starts a subprocess / real work? | Reads `args.dry_run`? |
|---|---|---|
| `--clear-halt` (`:3579`) | no (deletes the halt file, then falls through) | no |
| `--halt` (`:3583`) | no (writes the halt file and returns) | no |
| `--status` | no | no |
| `--stop` | no subprocess (sends `os.kill` SIGTERM to the daemon PID) | no |
| `--explain` | no | no |
| **`--approve`** | **yes**: `watcher.run_pipeline(min, max)` (`:3774`) | **no** |
| `--evaluate` | no: `evaluate_results` only, with no `execute_decision`. Every step manifest is `file_exists`, so the LLM path is not reached | no |
| **`--run-pipeline`** (`:3808-3828`) | **yes**: `run_pipeline` → `run_step` → `_run_step_streaming` Popen (`:2601`) | **no** |
| **`--daemon`** | **yes**: `run_daemon` → `_scan_watcher_requests()` with no `dry_run` (`:3189`), `_poll_pending_approval` → `run_pipeline` (`:3341`), and `_process_result_file` → `execute_decision` | **no** |
| `--dispatch-selfplay` | yes unless dry | **yes** `:3836` |
| `--dispatch-learning-loop` | yes unless dry | **yes** `:3840` |
| `--process-requests` | yes unless dry | **yes** `:3845` |

So there are **three launching modes that ignore `--dry-run`: `--run-pipeline`, `--approve`, `--daemon`**. The brief named only the first. R2-B covers the other two.

### 1.5 Baseline (before any edit; logs in `data/watcher_safety_repairs/base_*.log`)

| Suite | Baseline |
|---|---|
| `test_s172_run4_routing_patch.py` | **20/20 PASS**, `S172_RUN4_ROUTING_SUITE_COMPLETE`. Its `[FAIL]` lines are the expected reds inside mutant arms M1–M4 |
| `test_watcher_llm_integration.py` | **13 passed** |
| `test_gate12_cleantree_admission.py` | **31/31 checks green** |
| `test_s172_d6_liveness_gate.py` | **59/59 checks green**, `COMPLETION SENTINEL: PASS` |

The routing suite anchors into `watcher_agent.py` by AST and behaviour (for example `:68` parses the source for `STEP1_EXPLICIT_PIN_KEYS`), not by line number. It stayed green.

---

## 2. Repairs

Full diff: `git diff -- agents/` (2 files, +123 / −8). The excerpts below are from it.

### R-1: an explicit step failure beats "the file exists" (`agents/watcher_agent.py`)

- New module helper `_manifest_skip_on_fail(manifest_path)` (live `:309`). It returns True only for an existing `file_exists` manifest with `skip_on_fail: true`, and returns False on any read or parse error (fail closed).
- In `evaluate_results`, before the file-based branch (live `:959-995`):
  ```python
  _step_failed = isinstance(results, dict) and results.get("success") is False
  # >>> R1_STEP_FAILED_BRANCH_BEGIN
  if _step_failed and not _manifest_skip_on_fail(manifest_path):
      ... AgentDecision(success_condition_met=False, confidence=0.0,
                        reasoning="Step N (<name>) reported success=False: return_code=…, error=… — …",
                        recommended_action="escalate", parse_method="step_failed")
      return decision, context
  # <<< R1_STEP_FAILED_BRANCH_END
  ```
- In the skip path, the condition `if not success and _skip_on_fail` becomes `if (not success or _step_failed) and _skip_on_fail`. When `_step_failed` is set, a second warning is added: `"Step N exited non-zero (success=False, return_code=…) — any existing output may be stale"`.
- `run_step` and `evaluate_file_exists` are unchanged.

| Gate | Verdict | Evidence |
|---|---|---|
| R1-RED | PASS | `741822b` `evaluate_results`, given `success=False` and a valid file, returns `proceed/file_exists/conf=1.00`. The fault is reproduced |
| R1-A | PASS | `escalate/step_failed/conf=0.0`. The reasoning names Step 1, `return_code=1` and the error |
| R1-B | PASS | `run_pipeline(1,2)` with a stubbed `run_step`: calls=`[1]`, `current_step=1`, and escalation wrote the **temp** halt file |
| R1-C | PASS | `success=True`: `proceed/file_exists` |
| R1-D | PASS | no `success` key: `proceed/file_exists` |
| R1-E | PASS | real `trse.json` (`skip_on_fail: true`) with `success=False`: `proceed/skip_on_fail` plus the non-zero-exit warning, with the file present **and** with it absent |
| R1-MUT | PASS | mutant: the block between the R1 markers is replaced by `_MUT_HITS.append("R1")` in a copy of live `evaluate_results`, compiled into a copy of `W.__dict__`. Sentinel hits=`['R1']`, which proves it executed. R1-A under the mutant gives `proceed/file_exists`, so it goes **RED** |

### R-2: `--dry-run` must never launch (`agents/watcher_agent.py`, `main`)

Added straight after `args = parser.parse_args()` (live `:3632-3649`). It runs before the halt handling and before any `WatcherAgent` is built, and it is **not** inside the `--run-pipeline` branch:
```python
# >>> R2_DRY_RUN_REFUSAL_BEGIN
if args.dry_run:
    _dry_unsupported = [flag for flag, on in (("--run-pipeline", args.run_pipeline),
                        ("--approve", args.approve), ("--daemon", args.daemon)) if on]
    if _dry_unsupported:
        print(f"ERROR: --dry-run is not supported for {...} (it would launch for real); refusing.")
        sys.exit(2)
# <<< R2_DRY_RUN_REFUSAL_END
```
The refusal fires if any of the three flags is present, even when `elif` precedence would have picked another mode. This is deliberate: it is conservative and has no false launch.

| Gate | Verdict | Evidence |
|---|---|---|
| R2-RED | PASS | `741822b` `main`, argv `--run-pipeline --dry-run --start-step 1 --end-step 1`: WatcherAgent built once, and `run_pipeline((1,1,None), _operator_pin_params=None)` is **called**. The fault is reproduced |
| R2-A | PASS | exit 2, WatcherAgent built 0 times, `run_pipeline` called 0 times, one line of output naming `--run-pipeline` |
| R2-B-approve | PASS | `--approve --dry-run`: same refusal, naming `--approve` |
| R2-B-daemon | PASS | `--daemon --dry-run`: same refusal, naming `--daemon` |
| R2-C | PASS | `--dispatch-selfplay --dry-run`: exit 0, `dispatch_selfplay({'episodes': 5}, dry_run=True)` |
| R2-D | PASS | `--run-pipeline` without dry-run: called once. The live and base call tuples are identical: `((1, 1, None), {'_operator_pin_params': None})` |
| R2-MUT | PASS | refusal block replaced by a sentinel; hits=`['R2']`. R2-A under the mutant has exit None and `run_pipeline` called once, so it goes **RED** |

`WatcherAgent` was replaced by a recording fake in the `main` globals for every R2 gate, so nothing launched.

### R-3: one halt authority (`agents/watcher_dispatch.py`)

- New helper `_halt_requested(self) -> Optional[str]` (live `:92-109`). It returns the path that triggered the halt, or None. It checks:
  - `self.kill_switch.halt_file_path`, falling back to `self.config.halt_file` (no new literal);
  - then the legacy `watcher_halt.flag`.
  It checks existence only and never calls `check_all()`.
- All three sites now call it: live `:154` (selfplay), `:325` (learning loop start) and `:416` (between steps). Each log line is kept and now names the triggering file, for example `Halt flag present (/tmp/agent_halt) — aborting selfplay dispatch`.

| Gate | Verdict | Evidence |
|---|---|---|
| R3-RED | PASS | `741822b` dispatch module, halt at a temp kill-switch path: selfplay launched 1 (stubbed) subprocess, and the learning loop ran steps `[3, 5]`. The fault is reproduced |
| R3-A | PASS | site 1: selfplay `False`, 0 launches. Site 2: loop `False`, steps `[]`. Site 3: halt created after step 3, so loop `False`, steps `[3]` |
| R3-B | PASS | only legacy `watcher_halt.flag` present, in a temp cwd: selfplay and loop both return without launching |
| R3-C | PASS | neither file present: selfplay `True` with 1 launch; loop `True` with steps `[3, 5]` |
| R3-SCAN | PASS | `watcher_halt.flag` appears only on lines 97 and 105, inside the helper (92-108). No other halt-path literal is in the file. Exactly 3 helper call sites |
| R3-MUT | PASS | helper body replaced by `return _MUT_HITS.append("R3")` (always None) in a copy of the whole live module, exec'd into a copy of `D.__dict__`. 11 sentinel hits. R3-A goes RED at all 3 sites and R3-B goes RED |
| HYG-NO-PROD-HALT | PASS | `/tmp/agent_halt` was absent before and after the suite. No test created it |

Stubs in force for the R3 gates:
- module `subprocess` replaced by a recorder;
- `_log_dispatch_decision` replaced by a no-op (the real one appends to the repo's `watcher_decisions.jsonl`);
- `build_llm_context` replaced by a fake;
- `run_step` and `evaluate_results` on a fake agent that holds a real `KillSwitch(halt_file_path=<temp>)`.

---

## 3. Regression: before and after (§5)

Gate lists were extracted with ANSI codes stripped and then diffed (`data/watcher_safety_repairs/{base,after}_*.gates`). All suites were run **sequentially**.

| Suite | Before | After | Diff |
|---|---|---|---|
| run4 routing | 20/20 PASS | 20/20 PASS | identical (43 gate lines, mutant-arm reds included) |
| llm integration | 13 passed | 13 passed | only the timing line changed (`5.69s` → `4.54s`) |
| gate12 cleantree | 31/31 | 31/31 | identical |
| d6 liveness | 59/59, sentinel PASS | 59/59, sentinel PASS | identical |

The Phase-4 Gate 22 dirty-tree red, which the brief said to expect, is not part of these four suites. It was neither run nor fixed.

**Scope proof** (`git status --porcelain`, end of session):
- **Mine:** `M agents/watcher_agent.py`, `M agents/watcher_dispatch.py`, `?? tests/test_watcher_safety_repairs.py`, `?? docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md`.
- **Other session (not touched):** `docs/CHAPTER_*.md`.
- **Pre-existing untracked brief and proposals:** `docs/CLAUDE_CODE_INSTRUCTIONS_…`, `docs/PROPOSAL_…`.
- **Also present:** `?? data/`, which holds this session's logs. It exists because of the `/tmp` rule and is **not for commit**.

---

## 4. Where the source contradicted or went beyond the brief

1. **More launching modes than the brief named.** `--approve` and `--daemon` also ignore `--dry-run` (§1.4). Both are refused (R2-B).
2. **The R-1 skip path needed a widened condition.** The existing skip path (`:964-986`) was reachable only when files were *invalid*. A failed Step 0 with a stale but valid `trse_context.json` used to grade as `file_exists/proceed`, not through the skip path. To meet R1-E ("a failed step still takes the existing skip path"), the condition was widened to `(not success or _step_failed)`. The skip path's `print` still says "output not produced", which is cosmetically wrong when a stale file exists. The warning carries the truth.
3. **Behaviour change for Beta to rule on: Step 1 `seed_domain_complete`.** `run_step:2014` returns `success: False, seed_domain_complete: True` when the certified sweep is finished and no run was generated.
   - **Before:** with a prior `optimal_window_config.json` present, this graded `proceed` and Step 2 ran on old Step-1 output.
   - **After:** it **escalates** (halt file plus Telegram CRITICAL).
   This is fail-closed and arguably correct, but "domain complete" is a terminal success state, not a failure, so a distinct `parse_method` or action may be wanted. **Not decided here.**
4. Every other `success: False` path (HARD preflight, hard freshness, P0.5, cursor unavailable, domain wall, fleet unestablished, timeout, exception) now escalates where it used to pass on a stale file. That is the intended effect.
5. Daemon `_process_result_file` (`:3447`): a watched results JSON that contains `"success": false` now escalates and writes the halt file. Before, it was graded on files alone.

## 5. Open items found and not fixed

1. **A real dry-run for `--run-pipeline`** (and for `--approve` / `--daemon`). Today they are refused (exit 2). A real dry-run is separate work.
2. **A failed Step 0 leaves a stale `trse_context.json` that Step 1 reads.** R-1 keeps `skip_on_fail`, so the pipeline proceeds and only warns. Nothing deletes or quarantines the stale context before Step 1 consumes it.
3. **Hard-coded `/tmp/agent_halt` literals left in `agents/watcher_agent.py`:**
   - `WatcherConfig.halt_file` default, live `:384`;
   - `run_pipeline`'s halt-reason read, live `:2843`;
   - `check_safety()` calls at live `:2842` / `:2869` (run_pipeline) and `:3240` (daemon) and `:3678` (`--status`), which build a default `KillSwitch()` and so ignore `config.halt_file`;
   - the `KillSwitch` default itself, `agents/safety/kill_switch.py:60` (out of scope).
   The R-3 helper honours `config.halt_file`, but the pipeline does not. With a non-default `halt_file`, the dispatch paths and the pipeline would disagree in the other direction.
4. **R-1 does not reach `dispatch_learning_loop`** (`agents/watcher_dispatch.py`):
   - live `:401` reads `getattr(heuristic_decision, 'action', 'proceed')`, but `AgentDecision` has `recommended_action`, not `action`, so the heuristic verdict is always `proceed`;
   - even a non-proceed result only logs; there is no `return False` (live `:403-405`);
   - a `run_step` result of `{"success": False}` (not `None`) therefore goes straight through the loop.
   This is the same fault class as R-1, on an agent-launchable path. It was not fixed because it is outside the brief's required behaviour.
5. **The LLM evaluation path in `dispatch_learning_loop`** (`_evaluate_step_via_bundle`) also never reads `results["success"]`.
6. **`_find_results` (`run_step:2192`) can return a results JSON that carries `success: False` after rc 0.** R-1 will now escalate it. This may be correct, but it is not verified against the real step result files. [UNVERIFIED]
7. **`watcher_dispatch._standalone_cli`** (`:803+`) has its own `--dry-run` handling. Not reviewed beyond confirming that it passes `dry_run` through to the three dispatch calls.

## 6. `git add` for Michael

Built from `git status --porcelain` at the end of the session. It excludes the other session's `docs/CHAPTER_*` files, the brief and proposals, and `data/` (logs):

```
git add agents/watcher_agent.py agents/watcher_dispatch.py tests/test_watcher_safety_repairs.py docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md
```

Whether to also add the brief itself (`docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_SAFETY_REPAIRS_20261111.md`, currently untracked) is Michael's call.
