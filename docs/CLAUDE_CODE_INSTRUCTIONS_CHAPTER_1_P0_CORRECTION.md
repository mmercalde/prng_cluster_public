# CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_P0_CORRECTION.md — REV1

**Chapter 1 remediation, first tranche: P0 items 1–5** (Team Beta disposition of audit
`db9782a`).

**This pass contains BOTH code changes and documentation corrections.** Three of the five
items change behaviour; two are documentation. Beta's requirement: *"The first tranche must
cover every operator-visible or behavior-changing defect."*

**Base:** current `main` on VM 101 (audit `db9782a`, submission `4e17d93`, Beta disposition
following). Claude Code as `michael`, venv `~/venvs/torch`. Implement and iterate; do **NOT**
commit, push, or run WATCHER. STOP at the gate for Team Alpha review.

**Concurrency note:** a TRSE Step-0 audit may be running in another session. It is read-only,
but it reads `window_optimizer.py` among other consumers. If you observe another process
editing files you are working on, stop and report rather than racing it.

---

## 0. Authority and source of truth

- **Findings:** `docs/CHAPTER_1_AUDIT_v1.md` (`db9782a`) — 41 claims classified; the
  prioritised correction list is its §6.
- **Threshold history:** `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md` — cite it, do not
  re-derive.
- **Foundations:** the `tfm-project-facts` skill §0. **§0.4's standing rule governs every
  judgement here.**
- `CHAPTER_1_PATCH_S114.md` is **superseded — do not merge or revive it.** Beta: the new
  patch must be generated from audit `db9782a` and the current tree, **not layered onto
  S114.** Its warm-start section describes deleted code.

---

## 1. CODE — dead threshold override flags must fail closed

`--forward-threshold` / `--reverse-threshold` are declared at `window_optimizer.py:1063-1066`
and **never referenced after `parse_args()`**. An operator passing one today gets a **silent
no-op** on a run that reports success. This is the fifth dead dimension and the first
operator-facing one.

**Beta's approved behaviour:**
```
flag absent  → existing supported path
flag present → explicit nonzero failure BEFORE coordinator construction
```

Suggested diagnostic: `WINDOW_OPTIMIZER_THRESHOLD_OVERRIDE_UNWIRED`.

**Removing the arguments from argparse so they become unrecognised is equally acceptable.**
Choose one, state why. **Continuing to parse and ignore them is prohibited.**

Record in a code comment the condition under which they may return: only if they feed the
single `resolve_directional_threshold()` authority established at `8a55a68`, preserve `0.0`
(`is None` fallback, not truthiness), and record requested/payload/effective. **They must not
create parallel threshold state.**

## 2. CODE — unsupported search strategies must fail closed

Three of the four documented strategies raise `TypeError` on first call — proven by live
`inspect.signature`. `WindowOptimizer.optimize` passes four kwargs; only
`BayesianOptimization.search` accepts them. The stale `SearchStrategy` ABC
(`window_optimizer.py:299-303`) is why it went unnoticed.

**Required:**
```
--strategy bayesian      → permitted
--strategy random        → fail closed
--strategy grid          → fail closed
--strategy evolutionary  → fail closed
```

The CLI must stop advertising routes that crash on invocation. Fail with a clear diagnostic
naming the cause (signature mismatch) rather than letting `TypeError` escape.

**Also required (Beta, explicit):** a requested **Bayesian run must not silently become
random search** because Optuna is unavailable. That is *semantic substitution, not graceful
degradation.* If Optuna is missing, fail closed — do not substitute a different algorithm
behind the same request.

## 3. CODE — D-4: metadata must report what executed, not invent values

`window_optimizer.py:940-941` emits `forward_threshold = 0.72` / `reverse_threshold = 0.81`
into `agent_metadata`. `0.81` exceeds the live `0.75` ceiling — but Beta ruled the ceiling
violation is the *symptom*; the defect is **dual authority**: metadata invents threshold
values independently of the configuration actually requested and executed. This is the sixth
instance of that pattern, in the file repaired at `8a55a68`.

**Required:**
- `agent_metadata` reports the **resolved trial values** — what actually executed.
- Separate the concerns explicitly:
  ```
  observed/executed threshold → provenance field
  future recommendation       → separately governed proposal field
  ```
- **If no authoritative effective value exists, omit the field or fail closed.** Do **not**
  substitute a magic constant. Do **not** clamp into range.

Use `resolve_directional_threshold()` — do not add a second resolution path.

## 4. DOC — harden the skip definition (audit §6 P0.1)

The audit calls this *"the chapter's highest-value content — it is the artifact that stopped
the near-removal."*

In `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` §3.1:

- **Keep the existing wording verbatim** — `skip_min` = "Minimum skip for variable PRNGs",
  `skip_max` = "Maximum skip for variable PRNGs".
- **Add the *why skip exists* rationale**: the published draw sequence is not an
  uninterrupted PRNG stream — two pre-test draws run before every live draw and are never
  published; draw equipment is re-selected per session; the evening session draws D3/D4/
  Fantasy 5/Daily Derby together. Real structural gaps of varying size. Source: *California
  State Lottery Daily & SuperLotto Plus Draw Procedures* (eff. 2021-06-09), §II and §V.
- **Add Beta's defect callout verbatim:**
  ```
  DEFECT — current hybrid kernels do not execute the requested
  skip_min/skip_max semantics and instead use a hard-coded stride.
  Hybrid optimization results are non-certifying.
  ```
- State the standing rule: **the fix is wire-in, not removal.** The purpose of this edit is
  that no future reader re-derives "remove it."

## 5. DOC — numeric bounds via live authority + commit-dated snapshot

**Every numeric search bound in the chapter is wrong** (§3.2, §4.1, §4.2, §4.3, §10.1):
thresholds documented `[0.15, 0.60] default 0.25` vs live `[0.30, 0.75] default 0.30`; window
and skip ceilings 10× and 2× too large. This is the class of error that produced the
"~62 features" incident.

**Beta's approved form — a date alone is insufficient, because multiple code states can share
a date:**

```
Authority:
  <live configuration/source symbol>

Snapshot:
  generated_at
  repository_commit
  configuration_digest
  extracted bounds
  explicit statement: informative snapshot, not authority
```

**The snapshot must be extracted programmatically, not hand-copied.** Window and skip bounds
get the same treatment as thresholds. State the precedence rule (`distributed_config.json`
overrides code defaults, `window_optimizer.py:57-61`) and carry over the two `_note`
provenance fields from `distributed_config.json` — they are the only in-repo record of *why*
the window floor is 6.

---

## 6. Also in this tranche (audit §6 P0.3–P0.5)

**P0.3 — document `resolve_directional_threshold()` as an invariant** (chapter §7.2):
precedence explicit > config > default; `is None` as the **sole** fallback trigger (`0.0` is
legitimate); fail-closed `ThresholdResolutionError`; and the regression history —
`3fdf434` fixed → `2389b61` silently reverted by stale-copy overwrite → `8a55a68` repaired.
**Cite `THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`; do not re-derive it.**

**P0.4 — record dead dimensions D-1…D-4 in the chapter**, each with its death hop. D-4 is
currently documented as working.

**P0.5 — rewrite the output-file contract** (§12.1, §12.3, §2.1): the canonical Steps 2–6
input is the **certified NPZ generation** via `utils.run_finalizer`; demote
`bidirectional_survivors.json` to a post-success summary; mark forward/reverse files
count-only. Correct the flat-vs-nested record shape and drop the non-existent `timestamp`
field.

## 7. Out of scope

- P1/P2 items 6–17 — retained in the ledger, **not waived**, but not this pass.
- The hybrid skip **wire-in** — separate deliverable.
- `run_with_config` writing `[]` survivor files while reporting success — separate ticket.
- `window_optimizer.py:798` calling `logger.warning` in a module that never imports `logging`
  — separate ticket, and **unverified at runtime**.
- Optuna sampling the combined-session mode Beta prohibits — separate ticket.
- The two stale duplicates (`docs/window_optimizer_integration_final.py`,
  `modules/window_optimizer.py`) are **ruled to be left in place** — do not edit or delete.
- `distributed_config.json` bare-metal addresses are deliberate — do not "correct" them.

## 8. Gates — `tests/test_chapter1_p0_corrections.py`

| gate | asserts |
|---|---|
| G-FLAG-FAILCLOSED | `--forward-threshold` / `--reverse-threshold` produce a **nonzero failure before coordinator construction** (or are unrecognised, if that route was chosen) |
| G-STRATEGY-FAILCLOSED | `random` / `grid` / `evolutionary` fail closed with a clear diagnostic; `bayesian` still runs |
| G-NO-SILENT-SUBSTITUTION | a Bayesian request with Optuna unavailable **fails**; it does not become random search |
| G-METADATA-PROVENANCE | `agent_metadata` reports resolved executed values; no `0.72`/`0.81` constants; absent value → omitted or fail-closed, never clamped |
| G-SNAPSHOT-EXTRACTED | the chapter's bounds snapshot matches live config **and** carries `repository_commit` + `configuration_digest` |
| G-SKIP-DEFECT-NOTE | the chapter retains the verbatim skip definition **and** the defect callout |

**Mutants** (four-part kill rule, VIR-2 — execution proof, clean control, fault-injection
control, detector independence): restore the silent no-op on a threshold flag → G-FLAG reds ·
restore `random` as permitted → G-STRATEGY reds · restore the `0.72/0.81` constants →
G-METADATA reds · substitute random search on missing Optuna → G-NO-SILENT-SUBSTITUTION reds.

**Verification-integrity controls (VIR-1…6):** execution proof (each gate exercises the live
call site, not a text match — `2389b61` reverted a fix by whole-block replacement and a text
anchor would have gone green) · clean control · fault-injection control per mutant ·
explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE` sentinel · unavailable-observer behaviour ·
declared audit scope.

## 9. Non-regression

Green before any edit and again after: D1.1 · D1.0 · D0 · D2 · D3.0 · D3 · D3.25 · D3.5 · D4 ·
D5 · D6 3.A · **D6-threshold (must stay 17/17)** · D6.1 · **threshold-propagation (5/5)** ·
Phase 3 · Phase 4.

## 10. Report

Per item: what changed, `file:line`, and why that form was chosen (notably: fail-closed vs.
argparse removal for item 1). The gate matrix and mutant results. Confirmation the miner path
and `resolve_directional_threshold()` are behaviourally unchanged. Explicit statement that
`CHAPTER_1_PATCH_S114.md` was **not** merged. Then STOP. **Do not commit.**
