# CLAUDE CODE REPORT — WATCHER FOLLOW-UP REPAIRS (F-1 / F-2) — 2026-11-11

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_FOLLOWUP_REPAIRS_20261111.md`
**Base:** `b6cb789` (HEAD at session start; `git diff b6cb789 -- agents/ tests/` was empty before editing).
**Box:** VM101, as `michael`, venv `~/venvs/torch`. **Not committed** (stop gate §5).
**Result:** new suite `tests/test_watcher_followup_repairs.py` gives **17/17 PASS, `COMPLETION SENTINEL: PASS`**. All four regression suites have per-gate lists **identical** to baseline, and `tests/test_watcher_safety_repairs.py` stays at **21/21**.

**Null results and cautions first:**
- §1 found **no contradiction** with the brief's anchors. Every anchor is within ±1–2 lines; the details are in §4.
- **§1.4: Alpha is right, and the earlier report was wrong.** `docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md` §5 item 4 says a non-proceed heuristic verdict "only logs; there is no `return False`". At `b6cb789` there **is** a `return False` at `agents/watcher_dispatch.py:406`. The earlier read stopped at `:404`. That return could never be reached, because `action` was always `'proceed'`, and it recorded no decision. I have not edited the earlier report, because this brief does not allow it. This report is the correction.
- **F-1 sends no notification at all.** The brief allows at most an informational one. I chose none; see §5 item 5.

§1 anchors are at **base `b6cb789`**. After the edits, live lines move by +20 after `watcher_agent.py:2889` and by +16 after `watcher_dispatch.py:359` (+27 after `:406`).

---

## 1. Investigation answers

### 1.1 The seed-domain-complete return (`agents/watcher_agent.py:2068-2083`)

```python
# --- §6 COMPLETE: the domain is exhausted. No run is generated. ---
if _cursor.is_complete:
    _msg = (f"SEED_DOMAIN_COMPLETE: the certified coverage union for "
            f"{_prng_type} covers the governed domain "
            f"[{_cursor.domain_start}, {_cursor.domain_end_exclusive}) "
            f"exactly ({_cursor.covered_seed_count:,} seeds across "
            f"{_cursor.certified_interval_count} certified interval(s)). "
            f"The sweep is finished; there is no next seed_start. No Step 1 "
            f"run was generated.")
    logger.info("Step %d not generated: %s", step, _msg)
    print(f"✅ {_msg}")
    return {
        "success": False,
        "error": _msg,
        "blocked_by": "seed_domain_complete",
        "seed_domain_complete": True,
        "next_seed_start": None,
    }
```

The dict has five keys: `success` (`False`), `error` (the message), `blocked_by` (`"seed_domain_complete"`), `seed_domain_complete` (`True`) and `next_seed_start` (`None`).
- There is **no `return_code`**.
- It is **not** passed through `_step1_stamp_pin_provenance`, because it returns before the command is built.
- It is Step-1-only: it sits inside the S145 block.

### 1.2 Consumers of that dict, and what happens at `b6cb789`

1. `run_pipeline` (`:2782`) gets it from `self.run_step(...)` (`:2886`). It is not `None`, so `:2888-2889` does nothing. The progress display reads `best_score`/`confidence` if it is enabled.
2. `evaluate_results(step, results, run_number)` (`:2897`).
   - The R-1 branch (`:966-995`) has `_step_failed=True`.
   - `window_optimizer.json` has no `skip_on_fail`, so `_manifest_skip_on_fail` is False.
   - The result is **`escalate` / `parse_method="step_failed"` / confidence 0.0**, with reasoning `"Step 1 (Window Optimizer) reported success=False: error=SEED_DOMAIN_COMPLETE: …"`. There is no `return_code` in the reasoning because the dict has no such key.
3. `execute_decision` (`:1382`):
   - records the decision to history;
   - appends it to `config.log_file`;
   - prints the evaluation banner;
   - takes the `else` (escalate) branch.
4. **`_handle_escalate` (`:1524`) runs.** It:
   - **writes the halt file** `self.kill_switch.create_halt_file("Escalated at step 1: …")` at `config.halt_file`, which defaults to `/tmp/agent_halt`;
   - prints `_notify_escalation` "HUMAN REVIEW REQUIRED";
   - sends **Telegram Class A `[WATCHER][CRITICAL][ACTION REQUIRED] … Pipeline HALTED -- human review required`**;
   - returns `False`.
5. `run_pipeline` `break`s, runs `finally`, sets `_pipeline_running=False` and logs "Pipeline execution finished". **The halt file then blocks every later run until `--clear-halt`.**

This is reproduced live by gate **F1-RED**: `evaluate`=1, `execute`=1, `escalate`=1, halt file created, 1 CRITICAL.

Other consumers: none in tracked code read `seed_domain_complete` (only the producer at `:2082` does). The dict reaches `evaluate_results` only through `run_pipeline`. It can also reach `dispatch_learning_loop` (`watcher_dispatch.py:344`) when the scope includes step 1; see §5 item 3.

### 1.3 How `run_pipeline` ends on a clean finish

- **Loop condition:** `while self._pipeline_running and self.current_step <= end_step`.
- **How the loop ends:**
  - `_handle_proceed` (`:1440`) advances `current_step`. When it passes `end_step`, the `while` ends.
  - At step 6, `_handle_proceed` calls **`_notify_complete`** and returns `False`, so `execute_decision` returns `False` and the loop `break`s.
- **Common exit:** `finally` closes the progress display. Then `self._pipeline_running = False` (`:3006`) and `logger.info("Pipeline execution finished")` (`:3007`). It **returns `None`** implicitly; the only explicit returns are the pre-loop pin/halt blocks.
- **`_notify_complete` (`:1556`):**
  - sends a Telegram **Class C `[WATCHER][INFO] PIPELINE COMPLETE`** only if `watcher_policies.json` has `notifications.info_on_complete` true (default False);
  - prints a `🎉 PIPELINE COMPLETE` banner.
  - It is **not** called on a finish with `end_step < 6`.

### 1.4 Exits from the `dispatch_learning_loop` per-step loop (`watcher_dispatch.py:268`)

| Line | Condition | Decision recorded |
|---|---|---|
| `:345-347` → `:348-359` | `run_step` raised (`results=None`), or returned `None`, so it **`return False`** at `:359` | `STEP_FAILED` |
| `:397` | LLM path (`use_llm` true): `decision.get("decision") != "proceed"`, so it `return False` | `STEP_HALTED` (`:389-396`) |
| `:406` | heuristic path: `action not in ("proceed",)`, so it **`return False`**. **Unreachable**, because `action = getattr(heuristic_decision, 'action', 'proceed')` (`:401`) is always `'proceed'` (`AgentDecision` has `recommended_action`, `agents/agent_decision.py:62`) | **none** |
| `:422` | halt requested between steps (`_halt_requested`), so it `return False` | `HALTED` |
| loop end `:424-427` | all steps done, so it `return True` | `COMPLETED` |

Pre-loop exits (not per-step): invalid or unknown scope (`:309`, `:312`), dry-run (`:317-322`, returns True) and halt at start (`:328`).

**Disagreement resolved:** Alpha is right. There is a `return False` at `:406`. My earlier report was wrong to say there was none. The practical effect was the same as having none, because the branch was dead.

### 1.5 The `except` after the evaluation block (`:408-411`)

`except Exception as e:` catches **anything** raised from `:363-406`. That covers:
- building the manifest path;
- `build_llm_context`;
- `_evaluate_step_via_bundle`;
- `decision.get(...)`;
- `self.evaluate_results`;
- reading the attribute.

It logs a WARNING (`"evaluation failed … — continuing (step output exists)"`) and then **carries on as if the step had passed**: `completed_steps.append(step)`, the halt check, then the next step. Nothing is recorded via `_log_dispatch_decision`. Its comment assumes "step succeeded", which was never checked at `b6cb789`. Not changed here; see §5 item 1.

### 1.6 Baseline (logs in `data/watcher_followup_repairs/base_*.log`; run sequentially)

| Suite | Baseline |
|---|---|
| `test_watcher_safety_repairs.py` | 21/21 PASS, sentinel PASS |
| `test_s172_run4_routing_patch.py` | 20/20 PASS (its `[FAIL]` lines are the expected mutant-arm reds) |
| `test_seed_domain_cursor_amendment.py` | 40/40 checks green, sentinel PASS |
| `test_watcher_llm_integration.py` | 13 passed |

---

## 2. Repairs

Full diff: `git diff -- agents/` (2 files, +46 / −1; copy at `data/watcher_followup_repairs/agents.diff`).

### F-1: a finished sweep is a clean stop (`agents/watcher_agent.py`, `run_pipeline`, live `:2891-2910`)

The new check goes straight after the `results is None` normalisation and before the progress update and `evaluate_results`:

```python
# >>> F1_SEED_DOMAIN_COMPLETE_BEGIN
if isinstance(results, dict) and results.get("seed_domain_complete") is True:
    logger.info(
        "SEED_DOMAIN_COMPLETE — pipeline stopped cleanly at step %d: %s",
        step, results.get("error", ""),
    )
    print(f"\n✅ SEED_DOMAIN_COMPLETE — pipeline stopped cleanly at "
          f"Step {step}; no halt file written.\n")
    break
# <<< F1_SEED_DOMAIN_COMPLETE_END
```

- `break` leaves through the same `finally` → `_pipeline_running=False` → "Pipeline execution finished" path as a normal finish (§1.3).
- It keys on `seed_domain_complete is True` only, and never reads `blocked_by`.
- `evaluate_results`, `run_step` and `AgentDecision` are unchanged.
- **Deliberate:** any other caller that hands `evaluate_results` a `seed_domain_complete` result still escalates (§5 item 2).

| Gate | Verdict | Evidence |
|---|---|---|
| F1-RED | PASS | `b6cb789` `run_pipeline`: evaluate 1, execute 1, **escalate 1, temp halt file created, 1 CRITICAL**. The fault is reproduced |
| F1-A | PASS | live: evaluate 0, execute 0, escalate 0, no halt file, no Telegram, `current_step`=1, `_pipeline_running`=False. One INFO line starts with `SEED_DOMAIN_COMPLETE` and carries "The sweep is finished" |
| F1-B | PASS | `start_step=1, end_step=2`: `run_step` calls=`[1]`, `current_step`=1, no halt |
| F1-C | PASS | `{"success": False, "return_code": 1}`: live and base observations identical (escalate 1, halt file, 1 CRITICAL, calls `[1]`). R-1 is intact |
| F1-D | PASS | `seed_domain_complete: False`: escalates (halt file, 1 CRITICAL), no SDC INFO line |
| F1-E | PASS | ordinary success, `1→2`: live and base identical (calls `[1, 2]`, `current_step`=3, no escalation) |
| F1-MUT | PASS | block between the F1 markers replaced by `_MUT_HITS.append("F1")` in a copy of live `run_pipeline`, compiled into a copy of `W.__dict__`. Sentinel hits=`['F1']`. Under the mutant F1-A gives escalate 1 and a halt file, so it goes **RED** |

### F-2: the learning loop stops on a failed step (`agents/watcher_dispatch.py`)

**(a)** Straight after the `results is None` branch (live `:361-377`):

```python
# >>> F2_STEP_FAILED_BEGIN
if isinstance(results, dict) and results.get("success") is False:
    logger.error(f"[{run_id}] Step {step} reported success=False (return_code=…, error=…) — aborting loop")
    _log_dispatch_decision(self, run_id, "learning_loop", "STEP_FAILED",
                           {"failed_step": step, "completed": completed_steps})
    if is_gpu_heavy and hasattr(self, 'llm_lifecycle') and self.llm_lifecycle:
        self.llm_lifecycle.ensure_running()
    return False
# <<< F2_STEP_FAILED_END
```

**(b)** On the heuristic path (live `:421-432`):
- `action = heuristic_decision.recommended_action  # F2_HEURISTIC_ACTION_READ`
- a non-proceed verdict now records `STEP_HALTED` (`halted_step`, `action`, `confidence`, `completed`), with the same shape as the LLM path, and then `return False`.

The permissive `except` is unchanged.

| Gate | Verdict | Evidence |
|---|---|---|
| F2-RED-1 | PASS | `b6cb789` loop, step 3 returns `success=False`, `use_llm` false, valid outputs: returned **True**, ran `[3, 5]`, `COMPLETED`. The fault is reproduced |
| F2-RED-2 | PASS | `b6cb789` loop, verdict `escalate`: returned **True**, ran `[3, 5]`. The fault is reproduced |
| F2-A | PASS | live: returned False, ran `[3]`, `STEP_FAILED` with `failed_step=3, completed=[]`, evaluate 0 |
| F2-B | PASS | live, `use_llm` true: returned False, ran `[3]`, **LLM calls 0** |
| F2-C | PASS | live, verdict `escalate`: returned False, ran `[3]`, `STEP_HALTED` with `action=escalate, halted_step=3` |
| F2-D | PASS | all succeed, real `evaluate_results` (file_exists → proceed): live and base identical, returned True, ran `[3, 5]`, `COMPLETED` |
| F2-E | PASS | `results is None`: live and base identical, returned False, ran `[3]`, `STEP_FAILED` |
| F2-MUT-1 | PASS | F2 block removed (whole live module copy, sentinel hits 3). **F2-A RED**: the loop still stops, but with `STEP_HALTED` instead of `STEP_FAILED`, because R-1's escalation is now read correctly by (b). **F2-B RED**: the LLM path ran `[3, 5]`, made 2 LLM calls and returned True |
| F2-MUT-2 | PASS | `getattr(heuristic_decision, 'action', 'proceed')` restored at the marker line (sentinel hits 2). F2-C is **RED**: ran `[3, 5]`, returned True |
| HYG-NO-PROD-HALT | PASS | `/tmp/agent_halt` was absent before and after the suite |

Note on F2-MUT-1: on the heuristic path, (a) and (b) are **defence in depth**. Either one alone stops the loop, because R-1 makes `evaluate_results` escalate on `success=False`. On the **LLM path, (a) is the only guard**.

---

## 3. Regression: before and after

Gate lists were extracted with ANSI codes stripped and diffed (`data/watcher_followup_repairs/{base,after}_*.gates`). All suites ran sequentially.

| Suite | Before | After | Diff |
|---|---|---|---|
| watcher safety repairs | 21/21, sentinel PASS | 21/21, sentinel PASS | identical (23 lines) |
| run4 routing | 20/20 | 20/20 | identical (43 lines) |
| S145 seed-domain cursor | 40/40, sentinel PASS | 40/40, sentinel PASS | identical (42 lines) |
| llm integration | 13 passed | 13 passed | identical |

**Scope proof** (`git status --porcelain`, end of session):
- **Mine:** `M agents/watcher_agent.py`, `M agents/watcher_dispatch.py`, `?? tests/test_watcher_followup_repairs.py`, `?? docs/CLAUDE_CODE_REPORT_WATCHER_FOLLOWUP_REPAIRS.md`.
- **Logs:** `?? data/`.
- **Other session (not touched):** `M docs/CHAPTER_14_AUDIT.md`, `M docs/CHAPTER_8_PRNG_REGISTRY.md`.
- **The brief itself:** `?? docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_FOLLOWUP_REPAIRS_20261111.md`.

---

## 4. Where the source contradicted the brief

Nothing substantive. Line-level notes, all at `b6cb789`:
- The R-1 check is at `:966` (`_step_failed = …`) / `:968` (marker) / `:969` (`if`). The brief's `:968` is fine.
- The LLM-path evaluator call is at `:383`, not `:382`.
- `STEP_HALTED` is at `:389-396`, not `:388-396`.
- The heuristic `return False` is at `:406`, not `:405`.
- The `except` is at `:408`, not `:407`.
- `results = self.run_step(step)` is at `:344` and the `None` branch is at `:348-359`; the brief cites `:344-351`.

The one real disagreement, over §1.4, was between the brief and **my earlier report**, and the brief is right (see the top of this report).

## 5. Open items found and not fixed

1. **The permissive `except` in the learning loop** (live `:433-436`). Any exception during evaluation is logged and the step is **counted as completed**, with no dispatch decision recorded. This covers an `AttributeError` on a decision object with no `recommended_action`, LLM transport errors and `build_llm_context` failures. After F-2, an explicit `success=False` can no longer reach it, but a failure *inside evaluation* still lets the loop go on.
2. **`evaluate_results` still escalates on `seed_domain_complete`** for callers other than `run_pipeline`. That includes the daemon's `_process_result_file` (live `:3532`), `--evaluate` and the learning loop's heuristic path. This is deliberate under the brief.
3. **The learning loop with step 1 in scope** (`scope="full"` or `steps_1_…`): a domain-complete step 1 is now reported as **`STEP_FAILED`** and returns False. There is no halt file and no CRITICAL alert, but the terminal success state is recorded as a failure. F-1 covers only `run_pipeline`.
4. **Selfplay dispatch failure handling** (`dispatch_selfplay`, live `:116`). It does **not** have the same shape:
   - it runs no evaluation;
   - a non-zero rc records `FAILED` (`:204`) and returns False;
   - a timeout or exception records `TIMEOUT`/`EXCEPTION` (`:224`/`:227`), leaves `success=False` and returns False.
   One minor issue: the timeout and exception paths then **also** record a second `FAILED` at `:258`, which gives two decision records per failed run. Not a safety fault.
5. **F-1 sends no notification.** The INFO line and the console print are the only signals. If the owner wants a Telegram when the sweep finishes, the consistent choice is the policy-gated Class C pattern from `_notify_complete` (`watcher_policies.json` `notifications.info_on_complete`). It is not wired.
6. **Progress display:** if `PROGRESS_DISPLAY_AVAILABLE`, F-1's `break` leaves the step started but never completed in the display. This is cosmetic and was not tested with a real `WatcherProgress`. [UNVERIFIED]
7. Carried from the earlier report and still open:
   - a real `--dry-run` for `--run-pipeline`/`--approve`/`--daemon`;
   - a stale `trse_context.json` after a failed Step 0;
   - the remaining `/tmp/agent_halt` literals and default-`KillSwitch()` `check_safety()` calls in `agents/watcher_agent.py`.

## 6. `git add` for Michael

Built from `git status --porcelain` at the end of the session. It excludes `data/` (logs), the other session's `docs/CHAPTER_*` files and the brief:

```
git add agents/watcher_agent.py agents/watcher_dispatch.py tests/test_watcher_followup_repairs.py docs/CLAUDE_CODE_REPORT_WATCHER_FOLLOWUP_REPAIRS.md
```

Whether to also add the brief (`docs/CLAUDE_CODE_INSTRUCTIONS_WATCHER_FOLLOWUP_REPAIRS_20261111.md`) is Michael's call.
