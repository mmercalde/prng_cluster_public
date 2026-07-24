# TEAM_ALPHA_REVIEW_S172_PHASE5_D1_0.md

**Subject:** Team Alpha code-level review of the D1.0 implementation
(workflow bidirectionality + abort/commit terminal-race correction)
**Spec:** `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D1.md` REV5 (Beta-authorized)
**Base:** HEAD `7f2a010` (uncommitted working tree on VM 101)
**Artifacts reviewed:** `git diff` (94 lines, 2 files, 4 hunks), `git status`,
`tests/test_s172_phase5_d1_workflow.py` (832 lines)
**Verdict: APPROVE — recommend Team Beta code-level review, with ONE scope
deviation flagged for an explicit ruling (§2).**

---

## 1. Scope audit

`git status`: exactly two tracked modifications —
`miner/range_miner_coordinator.py` and `tests/test_s172_phase4_coordinator.py`
— plus the new untracked harness. The other untracked entries
(`CLAUDE_CODE_BRIEF_S176..178*`, `CLAUDE_CODE_BRIEF_WATCHER_KPI*`, `tmp/`)
pre-date D1.0 and are not products of this deliverable (Michael to confirm).

The coordinator diff contains ONLY the two authorized corrections (three
hunks: abort docstring, abort decision block, `workflow_stages_for`). No other
function, import, or constant moved. AST-verified.

## 2. Scope deviation — flagged for Beta ruling

REV5 authorized coordinator + workflow harness only. The implementation also
added **6 lines** (5 comment + 1 entry) to `tests/test_s172_phase4_coordinator.py`
gate 22's coexistence whitelist, registering the new D1.0 harness file. This
is mechanically forced — gate 22 enumerates each deliverable's own files and
would otherwise fail W4 — and follows the exact precedent of the D0 harness
entry directly above it. Claude Code reported it in-session rather than
stopping.

**Requested ruling:** either (a) bless whitelist registration of each new
deliverable's own harness as a standing pattern (recommended — it is the
gate's designed maintenance path, and the D0 entry set the precedent), or
(b) require it as a declared scope item per deliverable going forward. Team
Alpha recommends (a) with the note that ANY other edit to an approved harness
still requires explicit authorization.

## 3. Shape fidelity — both corrections verbatim

**§2.1 `workflow_stages_for`:** the `False` branch is the authorized shape
exactly; the `True` branch is untouched; the docstring carries the required
"constant is always bidirectional; hybrid pair only when test_both_modes"
statement plus the legacy citation. No other change to the function.

**§2.2 `abort_trial`:** the single line `first = mark_trial_aborted(...)` is
replaced by the REV5 CAS shape verbatim: `aborted`-check → CAS → on-`False`
terminal-state re-read → `committed` refusal (reusing the existing refused
shape) → `RuntimeError` fail-closed on any other state. The pre-existing
pre-read, committed early-return, `abort_event_id`, `cancel_active_stripes`
fence, and everything downstream (sink discharge, cleanup, deferred pump) are
untouched. One immaterial difference: the re-read comment says "The read
above" vs the spec's "The initial read". **No `_lifecycle_lock` (or any lock)
is acquired**; the docstring documents CAS + re-read, the deadlock rationale,
and the `_write_lock` distinction — all three Beta approval-note requirements.

## 4. Harness audit — no test-only shortcuts found

- **W2/W3 drive the REAL default serve path**: `run_trial_miner` with NO
  `_serve` and NO family/phase override, so `workflow_stages_for` is the
  producer authority under test, against a real framed-socket worker speaking
  the real `MinerFramedSocket` wire (register → assign → inline sub-stripe →
  StripeComplete). W2 additionally asserts explicit per-phase identity strings
  (direction, skip_mode, family, prng_type, directional `threshold_used`) on
  every published manifest. W3's fixture is genuinely bidirectional
  (F∩R = {5,7}, F−R = {3}, R−F = {9}) and re-reads the staged spools from disk.
- **W5-R interception is correct**: real method first; thread-specific
  (`abort_tid` captured in the abort thread); first-call-only; asserts the
  intercepted read observed `running`; events only — **zero `time.sleep` in
  the entire file**; timeouts are failure detectors. Team Alpha verified the
  one hazard that could silently defeat it: `MinerLedger.create_trial`
  (:771-783) is a raw `INSERT OR IGNORE` with no internal `get_trial`, so the
  intercepted call IS the vulnerable pre-read.
- **W6 meets the Beta harness requirement precisely**: each case runs in an
  isolated child (`subprocess.run(..., timeout=120)`, `--w6-child` dispatch);
  the child's own 45s `future.result` timeout is the failure detector and
  exits via `os._exit(3)` so surviving deadlocked threads cannot keep the
  child alive; both matrix branches covered (`retryable=False` phase-1;
  constant-phase `retryable=True` phase-2) with single-discharge,
  `aborted`-state, `"done"`-cleanup, event-id, and staged-removal assertions.
- W1 covers three bases (no family hardcoding), asserts the prefix property,
  and cross-checks against `workflow_phase_semantics`.
- W4 subprocess-runs all three suites asserting exit code AND the expected
  tallies ("63/63", "17/17", "12/12").

## 5. Mechanical verification (Team Alpha sandbox, pristine `7f2a010` clones)

**Patched tree** (diff applied + harness): **8/8 D1.0 gate checks green**,
including the full W4 non-regression (Phase 4 63/63, Phase 3 17/17, D0 12/12).
Independently reproduces Claude Code's reported result.

**Pre-fix tree** (harness only, coordinator UNPATCHED) — the discrimination
profile is exactly as the spec predicts:

| Gate | Pre-fix result | Signature |
|---|---|---|
| W1 | **FAIL** | `[('java_lcg', 1)]` — forward-constant alone |
| W2 | **FAIL** | only phase `[1]` published |
| W3 | **FAIL** | reverse constant population empty |
| W5-R | **FAIL** | abort returned `cleanup:"done"` WITH the abort event — the sink abort was discharged against a committed trial |
| W5-A / W5-B | PASS | ordinary orderings — non-discriminating by design, documented as such |
| W6 | PASS | discriminates against the REV4 *locked* design (never committed), not HEAD — HEAD has no lock either |
| W4 | FAIL (expected) | phase-4 suite exits 1 in the unpatched tree; irrelevant — W4's purpose is non-regression on the patched tree, where it is green |

Every gate that claims to prove a correction fails without that correction.
Rule 2 is satisfied mechanically, not by assertion.

## 6. Recommendation

Approve D1.0 for commit pending Team Beta review and the §2 whitelist ruling.
On approval, Michael commits the three files (coordinator, phase-4 harness
whitelist, new workflow harness) + this memo + the REV5 instruction doc to
`docs/`, dual-pushes, and D1.1 kickoff follows against the new HEAD.

— Team Alpha (Claude), 2026-07-23
