# CLAUDE_CODE_INSTRUCTIONS_CHAIN_C_TRUTHFULNESS_HOTFIX.md — REV1

**Chain C: WATCHER reports `Applied:` for LLM parameter proposals that never execute.**

**This is a narrow truthfulness repair. It is deliberately small.** Team Beta ruled the
filtering behaviour **correct** — only the reporting is defective. Do not build state machines,
provenance schemas or new gates. If the change grows beyond the log message, its status field
and a rejection reason, stop and report.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. Implement
and iterate; do **NOT** commit, push, or run WATCHER. STOP at the gate.

**⚠️ Concurrency:** a Chapter 1 P1/P2 documentation session may be running. It edits
`docs/CHAPTER_1_WINDOW_OPTIMIZER.md` and `apply_s146_doc_updates.py` — **not**
`agents/watcher_agent.py`. No collision expected; report it if you see one.

---

## 0. What is wrong, and what is *not*

`agents/watcher_agent.py:1789-1793` iterates LLM parameter proposals, validates each against a
policy whitelist, assigns the accepted value into `retry_params`, and logs:

```
[WATCHER][LLM_DIAG] Applied: learning_rate = 0.01
```

The value is then **filtered out** at `agents/watcher_agent.py:1385-1393` by a step-scoped
`allowed_params` list. All six whitelisted parameters are absent from
`agent_manifests/reinforcement.json` `default_params`, so **none of them reach the Step-5
script.**

**Team Beta's ruling — read this before touching anything:**

> The step-scoped filter is a **deliberate executable-interface boundary, not an oversight.**
> Passing proposal validation does not authorise bypassing the step's parameter interface.
> The six proposals must **not** be wired through.

So: **the filtering is correct. The reporting is the defect.** WATCHER asserts a state
(`Applied`) that has not occurred and will not occur. A filtered retry is being represented as
an adapted retry.

**Explicitly NOT authorised** (Beta, verbatim): *"This does not authorize adding the six fields
to `reinforcement.json`."* Do not add them. Do not modify the filter. Do not widen WATCHER's
authority.

## 1. The falsifiable question

> Does WATCHER ever report a parameter as applied when that parameter does not reach the
> dispatched step?

After this change the answer must be **no**.

## 2. Required change

**`Applied` may not be emitted at policy-validation time.** It is permissible only after the
value survives the step boundary and is materialised into the dispatch parameters.

For the currently-rejected case, report the truth. Beta's model:

```
Proposal validated but not applied:
learning_rate rejected by Step-5 executable parameter interface.
Retry continues with the existing effective value.
```

Two things the message must convey: **that the proposal was valid** (it passed policy) and
**why it did not apply** (the step does not expose that parameter). It should not read as an
error — the outcome is correct behaviour.

**Where the log line sits relative to the filter matters.** If the proposal loop at `:1789-1793`
runs *before* the filter at `:1385-1393`, then at logging time the outcome is not yet known.
Choose one, and state which and why:

- **(a)** move or defer the report until after the filter, so it can state the real outcome; or
- **(b)** log a *pre-filter* state at `:1789-1793` using non-committal wording (proposal
  accepted by policy — **not** `Applied`), and log the post-filter outcome where the filter runs.

Alpha's preference is **(a)** if it is straightforward, because one accurate line beats two
partial ones. Take **(b)** if (a) would restructure the retry path — this is a hotfix.

## 3. Provenance — the minimum, not the maximum

Beta listed a full vocabulary (`proposed`, `schema_valid`, `policy_valid`, `step_authorized`,
`materialized_in_dispatch`, `executed`, `verified_effective`, `rejected/deferred`) and six
provenance fields. **That is the eventual model for a governed execution seam. It is not this
hotfix.**

**Implement only what makes the report true:**

- a status distinguishing *validated-but-not-applied* from *applied*;
- a **rejection reason** naming the step-parameter interface as the cause;
- the **effective value** the retry actually continues with, where it is already available.

**Do not** add fields that require plumbing new data through the retry path. If a field Beta
listed is not already at hand, note it as deferred and move on.

## 4. Out of scope — do not do these

- **Do not add the six parameters** to `agent_manifests/reinforcement.json`. (Beta, explicit.)
- **Do not modify** `_is_within_policy_bounds` or the step-scoped filter at `:1385-1393`.
- **Do not** wire any proposal through to execution.
- **Do not** build the eight-state vocabulary, an execution seam, or a provenance schema.
- **Do not** touch Chain D — `pending_approval` is a **valid authority boundary**, and
  `record_applied_changes()` having zero callers is correct. Beta: *"Calling it before physical
  application would be worse."*
- **Do not** touch the Strategy Advisor, `search_strategy`, or any sampler work.
- **Do not** add a test harness beyond §5.

## 5. Verification — proportionate

**No new gate suite.** This is a message and status correction.

Required proof, and it is small: **exercise the rejected path and show the emitted text**, before
and after. The old line says `Applied:` for a value that does not reach Step 5; the new line does
not. Paste both.

If reaching that path requires WATCHER machinery you cannot run, say so
(`UNAVAILABLE`, VIR-5) and instead show the change at the call site with `file:line` plus a
direct reading of the control flow from the proposal loop to the filter. **Do not fabricate a
harness to satisfy a checkbox.**

**Do run afterwards:** any existing WATCHER/agent test suite that touches
`agents/watcher_agent.py`, plus `tests/test_s172_phase4_coordinator.py` (63/63) — gate 22 sees
changed `.py` files. If gate 22 flags `agents/watcher_agent.py`, register it in the allowlist
**with rationale**, per the established pattern.

## 6. Report

The change with `file:line`; which option from §2 was taken and why; the before/after log text;
anything from Beta's provenance list that was **deferred rather than implemented**, and why;
confirmation that the filter, `reinforcement.json` and Chain D are untouched. Then STOP.
**Do not commit.**
