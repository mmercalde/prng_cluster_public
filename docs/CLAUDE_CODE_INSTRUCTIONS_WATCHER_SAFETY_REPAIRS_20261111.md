# CLAUDE CODE INSTRUCTIONS — WATCHER SAFETY REPAIRS (three faults) — 2026-11-11

**From:** Team Alpha · **Owner:** Michael · **Run on:** VM101, as `michael`, in
`~/distributed_prng_analysis`, venv `source ~/venvs/torch/bin/activate` · **Model:** Opus
**Base commit:** `741822b`. All anchors below were read at that commit by Alpha.
`agents/watcher_agent.py` and `agents/watcher_dispatch.py` are unchanged since `a15b524`.
**Review class:** AFTER-COMMIT (WATCHER is not coordinator, worker or kernel code). Michael
commits; Beta reviews afterwards.

**Why now.** The owner has ruled that agents may launch runs. These three faults are in the
launch and stop paths an agent will use. Source: `docs/CHAPTER_12_AUDIT.md` §4, items B1, B3, B6.

---

## 0. Rules for this session

- **You do not commit or push.** Stop at the gate in §6 and report.
- **You may write only these files:**
  - `agents/watcher_agent.py`
  - `agents/watcher_dispatch.py`
  - `tests/test_watcher_safety_repairs.py` (new)
  - `docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md` (new)
- **Do not touch** the root-level `watcher_agent.py` (stale copy, deletion pending), any
  manifest, `agents/safety/kill_switch.py`, or any file under `miner/`.
- **Use the Edit tool only.** No `sed -i`, no Python replace.
- **Another Claude Code session is editing `docs/CHAPTER_*.md` in this tree.** Do not touch
  those files and do not be surprised by them in `git status`.
- **No fleet, no pipeline launch, no `--run-pipeline` against real scripts.** Every test stubs
  the step runner.
- Investigate first (§1), then fix (§2 to §4). If §1 contradicts an anchor below, stop and
  report before editing.

## 1. Investigate (read-only; write the answers into the report)

1. **Every `return` in `WatcherAgent.run_step`** (`agents/watcher_agent.py:1705`). For each:
   the line, the condition, and whether the returned dict carries `"success": False`. Include
   timeout and exception paths. The known one is the non-zero exit at `:2177-2185`.
2. **Every caller of `evaluate_results`** (`:2833`, `:3447`, `:3790`). For each: where `results`
   comes from and whether it can carry `"success": False`.
3. **`KillSwitch.check_all()`** (`agents/safety/kill_switch.py`): list what it checks besides
   the halt file, and whether it has side effects. Name the cheapest call that answers "is a
   halt requested" and nothing else.
4. **Every mode branch in `main()`** (`:3578` to end): for each, does it start a subprocess, and
   does it read `args.dry_run`? Known: only the three dispatch branches read it
   (`:3836`, `:3840`, `:3845`).
5. **Baseline, before any edit.** Run each suite below and save the per-gate pass/fail list:
   ```
   python3 -u tests/test_s172_run4_routing_patch.py   | tee /tmp/base_run4.log
   python3 -u tests/test_watcher_llm_integration.py   | tee /tmp/base_llm.log
   python3 -u tests/test_gate12_cleantree_admission.py | tee /tmp/base_cleantree.log
   python3 -u tests/test_s172_d6_liveness_gate.py     | tee /tmp/base_liveness.log
   ```
   `test_s172_run4_routing_patch.py` carries text and line anchors into
   `agents/watcher_agent.py` (`:741-748`). A red there after your edit is yours to explain.

## 2. Repair R-1 — an explicit step failure must beat "the file exists"

**Fault.** `run_step` returns `{"success": False, "return_code": …}` on a non-zero exit
(`:2177-2185`). `evaluate_results` (`:913`) never reads it: for a `file_exists` manifest it
grades only the output files (`:945-1000`). A stale or early-written output file therefore
turns a failed step into `Step N PASSED`, confidence 1.00, and the pipeline advances. This
happened in Gate-12 attempt 3.

**Required behaviour.**
- In `evaluate_results`, before the file-based branch: if `results.get("success") is False`,
  the decision is `success_condition_met=False`, `confidence=0.0`,
  `recommended_action="escalate"`, `parse_method="step_failed"`, and `reasoning` names the step
  and the `return_code` and `error` from `results` when present.
- Use `is False` exactly. A `results` dict with no `success` key (the `--evaluate` CLI path at
  `:3790`) behaves as it does today.
- **`skip_on_fail` is preserved.** If the manifest has `skip_on_fail: true` (Step 0, TRSE), a
  failed step still takes the existing skip path (`:964-986`) and the pipeline proceeds. That
  behaviour is by design (`TRSE_INTEGRATION_PLAN_S121.md` §2C). Add a warning string to the
  decision saying the step exited non-zero.
- Do not change what `run_step` returns. Do not change `evaluate_file_exists`.

**Gates (in the new test file).**

| Gate | Setup | Must hold |
|---|---|---|
| R1-RED | Pre-repair `evaluate_results` source read from `git show 741822b:agents/watcher_agent.py`, executed against the fixture of R1-A | It returns `proceed`. Proves the fixture reproduces the fault |
| R1-A | `results = {"success": False, "return_code": 1}`; required output file exists and is valid | `recommended_action == "escalate"`, `parse_method == "step_failed"` |
| R1-B | Same, driven through `run_pipeline` with `run_step` stubbed, `start_step=1, end_step=2` | Step 2's `run_step` is never called; `current_step` did not advance |
| R1-C | `results = {"success": True}`; file valid | `proceed`, as before |
| R1-D | `results` with no `success` key; file valid | `proceed`, as before |
| R1-E | `results = {"success": False}`; manifest `skip_on_fail: true` | `proceed` via `parse_method == "skip_on_fail"`, with the non-zero-exit warning present |
| R1-MUT | Delete the new branch from live source and re-run R1-A | Red. State how the mutant was applied and prove it executed |

## 3. Repair R-2 — `--dry-run` must never launch

**Fault.** `--dry-run` (`:3556`) is read only by the three dispatch branches. The
`--run-pipeline` branch (`:3808-3828`) ignores it, so
`watcher_agent.py --run-pipeline --dry-run` launches Step 1 for real.
`docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:443-450` shows that exact combination as the safe
check to run first.

**Required behaviour.**
- Immediately after argument parsing and before any `WatcherAgent` is constructed: if
  `args.dry_run` is set together with **any** mode that §1.4 shows starts a subprocess without
  honouring it, print one line naming the mode and saying dry-run is not supported for it, and
  exit with status 2.
- Do **not** place this inside the `--run-pipeline` branch. That branch is the operator-authority
  seam for the Route-A patch and is pinned by `tests/test_s172_run4_routing_patch.py`.
- Do not build a real dry-run for `--run-pipeline`. That is a separate piece of work; record it
  as open in the report.

**Gates.**

| Gate | Setup | Must hold |
|---|---|---|
| R2-RED | Pre-repair `main` from `741822b`, argv `--run-pipeline --dry-run --start-step 1 --end-step 1`, `WatcherAgent.run_pipeline` stubbed to record calls | `run_pipeline` is called. Proves the fault |
| R2-A | Same argv, live source | Exit status 2; `run_pipeline` never called; no `WatcherAgent` constructed |
| R2-B | One gate per other launching mode found in §1.4 | Same refusal |
| R2-C | `--dispatch-selfplay --dry-run` | Unchanged: reaches `dispatch_selfplay(..., dry_run=True)` |
| R2-D | `--run-pipeline` without `--dry-run`, `run_pipeline` stubbed | Called once, arguments unchanged from base |
| R2-MUT | Remove the refusal | R2-A red |

## 4. Repair R-3 — one halt authority

**Fault.** `--halt`, `create_halt`, escalation and `check_safety` use `/tmp/agent_halt`
(`agents/safety/kill_switch.py:60`; `agents/watcher_agent.py:366`, `:1473`, `:2777`). The three
dispatch paths check `watcher_halt.flag` in the working directory
(`agents/watcher_dispatch.py:126`, `:296`, `:386`), which no tracked code writes. A halt
therefore stops the pipeline and the daemon but not selfplay or learning-loop dispatch.

**Required behaviour.**
- One helper in `agents/watcher_dispatch.py` answering "is a halt requested". It returns true
  if the kill switch's halt file exists (the path from the agent's own `KillSwitch` /
  `config.halt_file`, not a new literal) **or** the legacy `watcher_halt.flag` exists.
- All three dispatch sites call the helper. No other literal halt path remains in
  `agents/watcher_dispatch.py`.
- Use the cheapest halt-file check found in §1.3. Do not call anything with side effects.
- The log line at each site stays, and names which file triggered the halt.

**Gates.**

| Gate | Setup | Must hold |
|---|---|---|
| R3-RED | Pre-repair dispatch source from `741822b`; halt file created at a temp path wired in as the agent's halt file; subprocess launch stubbed | Dispatch proceeds. Proves the fault |
| R3-A | Same, live source, for **each** of the three dispatch sites | Dispatch returns without launching |
| R3-B | Only legacy `watcher_halt.flag` present (in a temp working directory) | Dispatch returns without launching |
| R3-C | Neither file present | Dispatch proceeds as at base |
| R3-SCAN | AST or text scan of live `agents/watcher_dispatch.py` | The string `watcher_halt.flag` occurs only inside the helper |
| R3-MUT | Make the helper return False | R3-A and R3-B red |

Tests use a temp directory and a temp halt path. **No test creates `/tmp/agent_halt`** on this
host.

## 5. Verification block

- **Execution proof:** each gate prints its name and verdict; the suite prints
  `COMPLETION SENTINEL: PASS` only when every gate ran and passed.
- **Clean controls:** R1-C, R1-D, R2-C, R2-D, R3-C.
- **Fault-injection controls:** R1-RED, R2-RED, R3-RED and the three mutants. Mutants are
  applied to a copy of the live source executed with the production module's globals; a mutant
  that fails to apply is an error, not a detection.
- **Unavailable-observer behaviour:** a gate that cannot run reports `UNAVAILABLE`, never PASS.
- **Regression:** re-run the four §1.5 suites after all edits and diff the pass/fail lists
  against the baseline. Report every difference. The phase-4 Gate 22 dirty-tree red is expected
  and is not to be fixed.
- **Scope proof:** `git status --porcelain` shows only the four files in §0 plus the chapter
  files the other session owns.

## 6. Stop gate and report

Stop without committing. Write `docs/CLAUDE_CODE_REPORT_WATCHER_SAFETY_REPAIRS.md` with:

1. The §1 answers, with `file:line` anchors.
2. For each repair: the diff, the gate table with verdicts, and the mutant evidence.
3. The before/after regression table for the four suites.
4. Anything in this brief that the source contradicted.
5. Open items you found and did not fix. Expected at least: a real dry-run for
   `--run-pipeline`; a failed Step 0 leaving a stale `trse_context.json` that Step 1 reads; the
   remaining hard-coded `/tmp/agent_halt` literals in `agents/watcher_agent.py`.
6. The exact `git add` line for Michael, built from `git status`, not from memory.
