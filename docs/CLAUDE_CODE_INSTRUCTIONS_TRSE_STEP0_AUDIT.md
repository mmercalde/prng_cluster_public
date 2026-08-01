# CLAUDE_CODE_INSTRUCTIONS_TRSE_STEP0_AUDIT.md — REV1

**Audit: TRSE / Step 0 — what it computes, and whether its outputs reach anything.**

**AUDIT ONLY. Do not change code, config, or documentation. Do not commit.** A repair brief
(if one is warranted) will be written from your findings. **Do not assume a defect exists** —
see §1.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. You do NOT
commit, push, or run WATCHER. STOP at the gate.

---

## 0. Why this audit

TRSE is the **first stage of the pipeline** — upstream of Step 1's window optimization — and
**has never been examined.** Everything downstream has now been audited; this is the
remaining blind spot at the head of the chain.

One negative fact is established, from `docs/THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`:
**TRSE produces no threshold candidates** — regime/window quantities only; Rule A moves
`bounds.max_window_size`; Rules B/C are **logged-only**. Nothing else about TRSE is known.

Five instances of one defect class have now been found in this project: a parameter that is
computed, transported and recorded but **never reaches the code that claims to consume it**
(skill §0.5, §2.7). TRSE sits at the head of the chain, so if it has the same defect,
everything downstream inherits a bad or absent input.

## 1. The falsifiable question — and an explicit warning against assuming a defect

> What does TRSE compute, what consumes each output, and does each output reach the consumer
> that claims to use it?

**"Logged-only" is not automatically a defect.** `ca_d3_threshold_calibration.py` deliberately
*prints* recommendations for a human to apply rather than applying them itself — that is a
design choice, not a dead wire. Rules B/C being logged-only may be exactly the same. **Do not
classify advisory-by-design as a defect.** Where you find an output that goes nowhere,
determine whether it was *intended* to reach a consumer, citing `TRSE_v1_15_SPEC.md`,
`TRSE_INTEGRATION_PLAN_S121.md`, git history, or the code's own comments.

Per the standing rule (skill §0.4): **absence of a working implementation is not evidence of
absent intent** — and the converse also holds: absence of a consumer is not automatically a
defect if none was ever specified.

## 2. Surfaces in scope

**Modules:** `trse_step0.py` (32,908 B) · `trse_calibration_probe.py` (19,617 B) ·
`trse_entropy_probe.py` (21,162 B) · `step0_heuristic_validation.py` (12,589 B).

**Artifacts:** `trse_context.json` · `trse_boundary_candidates.json` ·
`trse_boundary_candidates_wide.json` · `probe_results.json` · `trse_entropy_probe.png` ·
anything else TRSE writes (enumerate; do not assume this list is complete).

**Specifications:** `docs/TRSE_v1_15_SPEC.md` · `docs/TRSE_INTEGRATION_PLAN_S121.md`.
**These are the design-intent authority for §1's judgement calls.**

**Known referencing surfaces** (from a preliminary grep — verify and extend):
`window_optimizer.py` · `window_optimizer_integration_final.py` ·
`window_optimizer_bayesian.py` · `agents/watcher_agent.py` · `agents/full_agent_context.py` ·
`agents/pipeline/pipeline_step_context.py` · `agents/manifest/agent_manifest.py` ·
`agents/progress_display.py` · `agent_manifests/window_optimizer.json` ·
`w8_correlation_test.py` · `machine_fingerprint_probe.py` · plus several `apply_s*.py`
patch scripts (`apply_s139_window_max_50.py`, `apply_s139b_trse_partition_fix.py`,
`apply_s140b_trial_history.py`, `apply_s142b_np2_terminal.py`, `fix_step1_timeout.py`).
**Patch scripts are historical artifacts — establish whether each was applied, and do not
treat an unapplied patch as current behaviour.**

## 3. Required findings

1. **What TRSE is.** In plain terms: what does it compute and why? Regime detection, entropy
   characterisation, window-bound calibration, boundary candidates, something else? Cite the
   spec.
2. **Invocation model — the decisive question.** Is TRSE invoked automatically as part of a
   pipeline run, or is it a **manual tool a human runs and interprets**? Check every caller,
   `STEP_SCRIPTS`-style maps, agent manifests, shell scripts, and any scheduler wiring. *If
   TRSE is manual-by-design, "fixing" it means something entirely different from an
   automated Step 0, so establish this before classifying anything else.*
3. **Output → consumer trace, per output.** For every value TRSE produces: where it is
   written, what reads it, and whether that reader's use matches what TRSE intended. Trace
   producer → artifact → consumer, hop by hop, with `file:line`.
4. **Rule A.** It reportedly moves `bounds.max_window_size`. Does that value actually reach
   Optuna's search space, and does Optuna honour it? Compare with the live
   `distributed_config.json` `search_bounds` and the S172 ruling that raised
   `window_size.min` to 6 — do TRSE and the ruling agree, disagree, or overwrite each other?
5. **Rules B and C.** Logged-only — advisory by design, or a dropped wire? **Cite the spec.**
6. **The JSON artifacts.** Are `trse_context.json` and `trse_boundary_candidates.json` read
   by anything, or write-only? If read, by whom and to what effect? Note mtimes — a stale
   artifact still being read is its own finding.
7. **Agent-layer integration.** `agents/full_agent_context.py`,
   `pipeline_step_context.py`, `watcher_agent.py` and `agent_manifests/window_optimizer.json`
   all reference TRSE. Does TRSE output reach agent decision-making, and is any of it
   declared as an agent-tunable parameter? *(Relevant to the dormant `parameter_application`
   path — a declared-but-disconnected knob is the §0.5 failure mode.)*
8. **Dead dimensions.** Any TRSE input or output that is computed/accepted but never
   consumed. **Classify each as DEFECT or ADVISORY-BY-DESIGN with evidence**, per §1.
9. **Spec vs. implementation.** Where `TRSE_v1_15_SPEC.md` or the S121 integration plan
   describe behaviour the code does not implement, say which reflects intent — the same
   judgement the Chapter 1 audit made for `skip_min`/`skip_max`.

## 4. Classification

Use the Chapter 1 template: **ACCURATE · STALE · SUPERSEDED · CONTRADICTED-BY-CODE ·
UNVERIFIABLE**, plus **ADVISORY-BY-DESIGN** for outputs that intentionally stop at a human.

## 5. Out of scope

- Do not fix anything.
- Do not re-audit the threshold path (`THRESHOLD_PATH_AUDIT_WINDOW_OPTIMIZER.md`) or Chapter
  1 (`CHAPTER_1_AUDIT_v1.md`) — **cite them** where TRSE overlaps.
- Do not run the sieve, any GPU kernel, WATCHER, or the pipeline.
- Do not modify `distributed_config.json` — its bare-metal rig addresses are deliberate
  (`CLAUDE.md` §3).

## 6. Verification-integrity controls (VIR-1…6)

- **execution proof** — every verdict carries a `file:line` anchor read this session.
- **clean control (VIR-2)** — state explicitly which TRSE outputs you verified as **correctly
  wired**. A report listing only defects gives no evidence the rest was checked.
- **fault-injection control** — n/a for a read-only audit; say so rather than omitting it.
- **completion sentinel (VIR-3)** — terminate with explicit
  `PASS | FAIL | UNAVAILABLE | INCOMPLETE` plus a coverage table. Anything not reached is
  `INCOMPLETE`, never silently absent.
- **unavailable-observer (VIR-5)** — anything unverifiable is `UNAVAILABLE`, not assumed
  correct. If you cannot determine whether an artifact is read at runtime without executing
  something, say so.
- **audit claim scope (VIR-6)** — declare searched and unavailable surfaces. **The repository
  is not the system**: if you check only VM 101 and not the rigs or host provisioning, state
  it.

## 7. Deliverable

`docs/TRSE_STEP0_AUDIT_v1.md`:

1. **What TRSE is** — plain-language, spec-cited.
2. **Invocation model** — automatic, manual, or both; with evidence.
3. **Output → consumer table** — output · written where · read by · effect · class.
4. **Rule A / B / C disposition** — each classified, with the advisory-vs-defect judgement
   and its evidence.
5. **Dead-dimension inventory** (if any), each with the hop where it dies.
6. **Spec-vs-code conflicts**, each with an intent assessment.
7. **Prioritised finding list**, ordered by consequence — what a good-faith reader or an
   autonomous agent could get wrong.
8. **Coverage table + completion sentinel.**

Then STOP for Team Alpha review.
