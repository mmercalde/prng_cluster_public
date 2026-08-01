# CLAUDE_CODE_INSTRUCTIONS_SAMPLER_BEARING.md — REV1

**Bearing: what would it cost to have four working Optuna samplers in Step 1?**

**READ-ONLY SCOPING. Do not change code, config or documentation. Do not commit.** This is a
**cost and blast-radius estimate**, not an implementation and not an authorisation. The
deliverable is a report that lets Michael and Team Beta decide whether the work is worth doing
at all.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. You do NOT
commit, push, or run WATCHER. STOP at the gate.

**Deliberately narrow.** A companion brief
(`CLAUDE_CODE_INSTRUCTIONS_STRATEGY_ADVISOR_AUDIT.md`) covers *what would ever select* a
sampler. **This brief covers only what it would take to make them work.** Do not merge the two.

---

## 0. Established facts — cite, do not re-derive

From `docs/STRATEGY_ORIGIN_AUDIT.md` (verdict: *always hand-rolled; Optuna intent documented
but never landed*):

| strategy | body | Optuna | callable today |
|---|---|---|---|
| `bayesian` | real; delegates to `window_optimizer_bayesian.py` | **yes**, TPE | yes |
| `random` | real hand-rolled loop, never degraded | no | no — signature mismatch |
| `grid` | **`return {}`** since Nov 2025 | no | no — signature mismatch |
| `evolutionary` | **`return {}`** since Nov 2025 | no | no — signature mismatch |

- Documented design intent (`PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md:20`): *"All 4
  Optuna samplers (TPE, Random, Grid, **CmaES**) implemented ✅"* — the checkmark was false when
  written. Note **CmaES**, not NSGAII; `NSGAIISampler` appears nowhere in the repository.
- `GridSearch` / `EvolutionarySearch` were **working hand-rolled code deleted to `return {}`**
  in mid-November 2025 — stubs by deletion, not by design.
- The three are **gated, not removed**, at `ddd2ac8` — they fail closed at the CLI.
- **The `ddd2ac8` remedy comment is misleading** and is being corrected separately: for Grid
  and Evolutionary, repairing signatures alone would turn a signature-derived gate green on a
  function that still returns `{}`.

## 1. The falsifiable question

> To make `random`, `grid` and `evolutionary` run as genuine Optuna samplers, **what exactly
> changes, in which files, and what does it risk?**

## 2. Required findings

### 2.1 How the working path constructs its sampler
`window_optimizer_bayesian.py` around `:543` and `:616-620` reportedly passes a sampler to
`create_study`. Establish exactly: where the sampler object is built, what parameters it takes,
where `create_study` is called, and **whether the sampler is already a variable or is
hard-wired to TPE**. This determines whether four-sampler support is *a parameter* or *a
refactor*.

### 2.2 The minimal change set
List every file and function that would change, with `file:line`. Distinguish:
- **sampler construction** — one call site, or several?
- **the strategy classes** — do `RandomSearch`/`GridSearch`/`EvolutionarySearch` become thin
  delegates to the same Optuna path (like `BayesianOptimization`), or does each need its own
  body?
- **the calling convention** — does routing them through the existing path make the
  four-forwarded-kwargs problem disappear, since they would then share
  `BayesianOptimization`'s signature?

*State plainly whether the natural implementation makes the signature problem moot.* If it
does, that is the most important finding in the report.

### 2.3 Sampler-specific constraints
Optuna's samplers are not interchangeable. For each of `RandomSampler`, `GridSampler`,
`CmaEsSampler`:
- **`GridSampler` requires an explicit `search_space` dict** enumerating every value to try.
  Step 1's space includes `window_size`, `offset`, `sessions`, `skip_min`, `skip_max`, and two
  thresholds. **Is a grid over that space even tractable?** Estimate the cartesian product from
  the live bounds in `distributed_config.json`. If it is astronomically large, say so — that is
  a finding that may retire `grid` on its own.
- **`CmaEsSampler` is continuous-oriented** and handles categorical/integer parameters poorly
  or not at all. Step 1's space is mixed. State what it would and would not handle.
- **`RandomSampler`** is the straightforward one — confirm it has no such constraint.

### 2.4 Interaction with existing machinery
Does routing the other three through the Optuna path affect: study storage and resume
(`--resume-study`, `--study-name`), warm-start enqueue from `step1_trial_history`, the TRSE
Rule A bounds mutation, the trial-history callback, or `optimal_window_config.json` output
shape? **Anything that assumes TPE specifically** is a finding.

### 2.5 Blast radius
Which non-Step-1 surfaces would be touched or affected: `agent_manifests/window_optimizer.json`
(already declares all four as choices), gate-22's changed-file allowlist, the Chapter 1
documentation, `tests/test_chapter1_p0_corrections.py`'s strategy gates, and the
`window_optimizer.py` CLI. **Does any of it touch the miner, PWC, ZMQ, the kernels or the 22-array
contract?** Alpha expects **no** — confirm or refute.

### 2.6 Effort estimate
Small / medium / large, with the reasoning. Alpha's prior expectation is **small** — a sampler
parameter on an existing `create_study` call — but that expectation is based on a single
second-hand line reference and must be verified, not confirmed.

### 2.7 What would be gained
Not a recommendation — an honest statement of what TPE cannot do that the others can, if
anything, for a mixed integer/categorical/continuous search space of this shape. **If the
answer is "little or nothing", say that.** TPE has run every production trial through Step 5.

## 3. Out of scope

- **Do not implement anything.** No sampler wiring, no signature changes, no schema edits.
- Do not touch the strategy classes, the CLI, the manifests or the gates.
- Do not decide whether the work should happen — that is Michael's and Beta's call, informed
  by this report and the advisor audit.
- Do not investigate *what would select* a sampler — that is the companion brief.
- Do not run WATCHER, the pipeline, the sieve or any GPU kernel. **Do not start an Optuna
  study.**
- Do not delete, retire or recommend removing any strategy. Per `tfm-project-facts` §0.4,
  removal is a Beta ruling, never an audit conclusion — and two of these three were already
  destroyed once by deletion.

## 4. Verification-integrity controls (VIR-1…6)

- **execution proof** — every claim carries a `file:line` anchor read this session. Optuna
  sampler behaviour may be verified by reading the installed package in `~/venvs/torch`; say
  which version.
- **clean control (VIR-2)** — state which parts of the existing Optuna path you verified as
  **working as described**.
- **fault-injection control** — n/a for read-only scoping; **say so** rather than omitting it.
- **completion sentinel (VIR-3)** — end with `PASS | FAIL | UNAVAILABLE | INCOMPLETE` and a
  coverage table.
- **unavailable-observer (VIR-5)** — anything unverifiable without executing something is
  `UNAVAILABLE`. Do not infer sampler behaviour you did not read.
- **audit claim scope (VIR-6)** — declare searched and unavailable surfaces.

## 5. Deliverable

`docs/SAMPLER_BEARING_v1.md`:

1. **How the working path builds its sampler** — is it a parameter or hard-wired?
2. **Minimal change set** — files, functions, line anchors.
3. **Does the natural implementation make the signature problem moot?** (yes/no, with evidence)
4. **Per-sampler constraints** — especially whether `GridSampler` is tractable over Step 1's
   live bounds, and what `CmaEsSampler` cannot handle.
5. **Interactions** — resume, warm-start, TRSE Rule A, trial history, output shape.
6. **Blast radius** — including explicit confirmation of whether anything outside Step 1 is
   touched.
7. **Effort estimate**, with reasoning.
8. **What would be gained** — honestly, including "possibly nothing."
9. **Coverage table + completion sentinel.**

Then STOP for Team Alpha review.
