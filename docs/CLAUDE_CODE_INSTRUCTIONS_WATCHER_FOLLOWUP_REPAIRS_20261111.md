# CLAUDE CODE INSTRUCTIONS — WATCHER FOLLOW-UP REPAIRS (two faults) — 2026-11-11

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, in
`~/distributed_prng_analysis`, venv `source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** `b6cb789`. Anchors below were read at that commit by Alpha.
**Review class:** AFTER-COMMIT. Michael commits; Beta reviews afterwards.
**Follows:** `docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md` §4 item 3 and §5 items 4, 5.

**Why now.** The owner defines RANGE-MINER as finished when the whole seed space is swept.
After repair R-1, reaching that state raises a halt and a critical alert. And the learning-loop
dispatch path, which an agent can launch, still lets a failed step through.

---

## 0. Rules for this session

- **You do not commit or push.** Stop at the gate in §5 and report.
- **You may write only these files:**
  - `agents/watcher_agent.py`
  - `agents/watcher_dispatch.py`
  - `tests/test_watcher_followup_repairs.py` (new)
  - `docs/CLAUDE_CODE_REPORT_WATCHER_FOLLOWUP_REPAIRS.md` (new)
- **Do not touch** `agents/agent_decision.py`, the root-level `watcher_agent.py`, any manifest,
  or any file under `miner/` or `utils/`.
- **Do not edit `tests/test_watcher_safety_repairs.py`.** If one of its gates goes red, stop and
  report; do not adjust it.
- **Use the Edit tool only.**
- **Another Claude Code session may be editing `docs/CHAPTER_*.md`.** Leave those files alone.
- **No fleet and no real pipeline launch.** Every test stubs the step runner.
- **Logs go under `data/watcher_followup_repairs/`**, which is untracked and not committed.
- Investigate first (§1). If §1 contradicts an anchor below, stop and report before editing.

## 1. Investigate (read-only; answers go in the report)

1. **The seed-domain-complete return** in `run_step` (`agents/watcher_agent.py:2068-2083`):
   quote the returned dict in full and name every key.
2. **Every consumer of that dict.** Trace it through `run_pipeline` (`:2782`) into
   `evaluate_results` and `execute_decision` (`:1382`). State exactly what happens today at
   `b6cb789`: which branch is taken, whether `_handle_escalate` (`:1524`) runs, whether a halt
   file is written, and which notification is sent.
3. **How `run_pipeline` ends on a clean finish:** what it sets, what it returns, what it prints
   and what `_notify_complete` (`:1556`) sends.
4. **`dispatch_learning_loop`** (`agents/watcher_dispatch.py:268`): list every exit from the
   per-step loop, with line and condition. The earlier report says a non-proceed heuristic
   verdict "only logs; there is no `return False`". Alpha reads a `return False` at `:405`.
   Say which is right at `b6cb789`.
5. **The `except` after the evaluation block** (`:407` on): what it catches and what the loop
   does next.
6. **Baseline before any edit.** Run and save the per-gate pass/fail lists:
   ```
   python3 -u tests/test_watcher_safety_repairs.py     | tee data/watcher_followup_repairs/base_safety.log
   python3 -u tests/test_s172_run4_routing_patch.py    | tee data/watcher_followup_repairs/base_run4.log
   python3 -u tests/test_seed_domain_cursor_amendment.py | tee data/watcher_followup_repairs/base_s145.log
   python3 -u tests/test_watcher_llm_integration.py    | tee data/watcher_followup_repairs/base_llm.log
   ```

## 2. Repair F-1 — a finished sweep is a clean stop, not a failure

**Fault.** When the certified coverage union covers the whole domain, `run_step` generates no
run and returns `{"success": False, "blocked_by": "seed_domain_complete",
"seed_domain_complete": True, …}` (`:2068-2083`). Since R-1, `evaluate_results` escalates on
`success is False` (`:968`), which writes the halt file and sends a critical alert. The halt
file then blocks every later run until someone clears it.

**Required behaviour.**
- In the `run_pipeline` loop, immediately after `run_step` returns and **before**
  `evaluate_results` is called: if `results.get("seed_domain_complete") is True`, the pipeline
  stops cleanly:
  - logs one INFO line beginning `SEED_DOMAIN_COMPLETE`, carrying the message from `results`;
  - sends at most an informational notification, never a critical one;
  - does **not** call `evaluate_results`, `execute_decision` or `_handle_escalate`;
  - does **not** write a halt file;
  - does **not** advance to the next step;
  - leaves the loop the same way a normal finish does (per §1.3).
- Use `is True` exactly, on the `seed_domain_complete` key. Do not key on `blocked_by` text.
- `evaluate_results` is **not** changed. Any other caller that hands it a
  `seed_domain_complete` result still escalates. Record that in the report as deliberate.
- Do not change what `run_step` returns. Do not change `AgentDecision`.

**Gates.**

| Gate | Setup | Must hold |
|---|---|---|
| F1-RED | Pre-repair `run_pipeline` from `git show b6cb789:agents/watcher_agent.py`; `run_step` stubbed to return the §1.1 dict; halt path pointed at a temp file | `_handle_escalate` runs and the temp halt file is created. Proves the fault |
| F1-A | Same stub, live source | No halt file; `_handle_escalate` not called; `evaluate_results` not called |
| F1-B | Same, `start_step=1, end_step=2` | `run_step` called once, for step 1 only |
| F1-C | `run_step` returns an ordinary failure `{"success": False, "return_code": 1}` | Still escalates exactly as at `b6cb789`. R-1 is intact |
| F1-D | `run_step` returns `{"success": False, "seed_domain_complete": False}` | Escalates. Only `True` is a clean stop |
| F1-E | `run_step` returns an ordinary success | Proceeds as at `b6cb789` |
| F1-MUT | Remove the new check | F1-A red |

Tests use a temp halt path. **No test creates `/tmp/agent_halt`.**

## 3. Repair F-2 — the learning loop must stop on a failed step

**Fault.** In `dispatch_learning_loop`:
- the heuristic path reads `getattr(heuristic_decision, 'action', 'proceed')`
  (`agents/watcher_dispatch.py:401`). `AgentDecision` has `recommended_action`
  (`agents/agent_decision.py:62`) and no `action`, so the verdict is always `proceed`;
- neither the heuristic path nor the LLM path (`:382`) looks at `results["success"]`. A step
  that returns `{"success": False}` instead of `None` (`:344-351`) goes on to evaluation.

**Required behaviour.**
- Directly after the existing `results is None` check: if `results.get("success") is False`,
  treat it exactly as that branch does. Log, record `STEP_FAILED` through
  `_log_dispatch_decision` with the failed step and completed steps, restart the LLM if the
  step was GPU-heavy, and `return False`. This covers both evaluation paths.
- The heuristic path reads `heuristic_decision.recommended_action`. A verdict other than
  `proceed` records `STEP_HALTED` through `_log_dispatch_decision`, as the LLM path does at
  `:388-396`, and returns `False`.
- The permissive `except` (§1.5) is not changed in this brief. Record what it does as an open
  item.

**Gates.**

| Gate | Setup | Must hold |
|---|---|---|
| F2-RED-1 | Pre-repair `dispatch_learning_loop` from `b6cb789`; `run_step` stubbed to return `{"success": False, "return_code": 1}` for the first step; `use_llm` false; output files valid | The loop goes on to the second step. Proves the fault |
| F2-RED-2 | Pre-repair source; `evaluate_results` stubbed to return a decision with `recommended_action="escalate"`; `run_step` returns a success dict | The loop goes on to the second step |
| F2-A | F2-RED-1 setup, live source | Returns `False`; second step's `run_step` never called; a `STEP_FAILED` decision is recorded |
| F2-B | Same with `use_llm` true and the LLM evaluation stubbed to return `proceed` | Returns `False` before the LLM is consulted |
| F2-C | F2-RED-2 setup, live source | Returns `False`; a `STEP_HALTED` decision is recorded |
| F2-D | All steps succeed and all verdicts are `proceed` | Loop completes as at `b6cb789` |
| F2-E | `results is None` | Unchanged from `b6cb789` |
| F2-MUT-1 | Remove the `success is False` check | F2-A and F2-B red |
| F2-MUT-2 | Restore `getattr(heuristic_decision, 'action', 'proceed')` | F2-C red |

## 4. Verification block

- **Execution proof:** each gate prints its name and verdict; the suite prints
  `COMPLETION SENTINEL: PASS` only when every gate ran and passed.
- **Clean controls:** F1-C, F1-D, F1-E, F2-D, F2-E.
- **Fault-injection controls:** F1-RED, F2-RED-1, F2-RED-2 and the three mutants. Mutants are
  applied to a copy of live source executed with the production module's globals. A mutant that
  fails to apply is an error, not a detection.
- **Unavailable-observer behaviour:** a gate that cannot run reports `UNAVAILABLE`, never PASS.
- **Regression:** re-run the four §1.6 suites after all edits and diff against the baseline.
  `tests/test_watcher_safety_repairs.py` must stay 21/21.
- **Scope proof:** `git status --porcelain` shows only the four files in §0, plus `data/` and
  any chapter files the other session owns.

## 5. Stop gate and report

Stop without committing. Write `docs/CLAUDE_CODE_REPORT_WATCHER_FOLLOWUP_REPAIRS.md` with:

1. The §1 answers, with `file:line` anchors, including the §1.4 disagreement resolved.
2. For each repair: the diff, the gate table with verdicts, and the mutant evidence.
3. The before/after regression table.
4. Anything in this brief the source contradicted.
5. Open items found and not fixed. Expected at least: the permissive `except` in the learning
   loop; `evaluate_results` still escalating on `seed_domain_complete` for callers other than
   `run_pipeline`; the selfplay dispatch path's handling of a failed run, if it has the same
   shape.
6. The exact `git add` line for Michael, built from `git status`.
