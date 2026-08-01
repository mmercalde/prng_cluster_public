# CLAUDE_CODE_INSTRUCTIONS_STRATEGY_ADVISOR_AUDIT.md — REV1

**Audit: the Strategy Advisor and the autonomy control surface — what it actually controls,
what it is declared to control, and what enforces the difference.**

**AUDIT ONLY. Do not change code, config, schemas, grammars or documentation. Do not commit.**
This is a **parked reference document**: the chapter-audit track continues in parallel and
takes priority. Nothing here authorises implementation.

**Base:** current `main` on VM 101. Claude Code as `michael`, venv `~/venvs/torch`. You do NOT
commit, push, or run WATCHER. STOP at the gate.

**⚠️ Concurrency:** other sessions may be running chapter audits. Read-only work cannot
collide, but if you observe another process editing a file you are reading, note it and move on.

---

## 0. Why this exists — and the standing hazard

This codebase is large, developed over a year, and **no one holds the whole picture.** Five
audits in, each has surfaced a component nobody remembered — and three times a reviewer has
recommended removing something that turned out to be load-bearing or deliberately built:

| component | what was proposed | what it actually was |
|---|---|---|
| `skip_min`/`skip_max` | remove from hybrid search | a physical model of unpublished pre-test draws — the design's foundation |
| three search strategies | "code rot, fix the signatures" | documented as four Optuna samplers; two were *deleted working code*, not stubs |
| `GridSearch`/`EvolutionarySearch` | "placeholders" | working implementations gutted to `return {}` in Nov 2025 |

**Therefore, binding on this audit** (`tfm-project-facts` §0.4): *absence of a working
implementation is not evidence of absent intent.* Before classifying anything as unused,
vestigial or removable, **find and cite the document, commit or comment explaining why it
exists.* If you cannot find one, say so — that is a finding, not a licence.

## 1. The falsifiable question

> What can the Strategy Advisor **emit**, what can the governance layer **validate**, what can
> the application layer **apply**, and what does the execution layer **receive** — and where
> does that chain break?

## 2. Essential context — do not re-derive, but do verify currency

- **Beta's original restriction was deliberate and correct for its time.** The advisor was
  kept away from data-pipeline parameters because an unconstrained LLM should not steer a
  pipeline. Michael's position now: **GBNF grammars and a validated Pydantic output schema
  materially change that risk profile**, and the restriction may warrant revisiting. **This
  audit does not decide that** — it establishes what is enforced today so Beta can rule with
  evidence.
- **Already established, cite rather than repeat:**
  - the advisor's `StrategyRecommendation` Pydantic model (`parameter_advisor.py:155`) has
    **no field for sieve thresholds**; the closest is `min_fitness_threshold`;
  - the **application seam does not exist** — the only hit for `apply_*` /
    `parameter_application` is a docstring, `diagnostics_analysis_schema.py:76`: *"LLM
    proposals are advisory only"*;
  - `watcher_policies.json` declares governed parameter application (`parameter_application:
    true`, `parameter_change_log`, `max_parameter_delta`, cooldown, bounds) with **no
    implementing code**;
  - selfplay is *"a policy-conditioned evaluation harness, not a learning system"* —
    `propose_transform_update` is a no-op and the promotion seam is broken at
    `chapter_13_acceptance.py:224`.

## 3. Required findings

### 3.1 The emit surface — what CAN the advisor say?
Enumerate **every field** of every advisor output schema: `StrategyRecommendation`,
`SelfplayOverrides`, `ParameterProposal`, and any sibling. For each: name, type, bounds,
required/optional. Then the **GBNF grammars** — all of them — and whether grammar and Pydantic
agree. A field in one and not the other is a finding.

**Specifically: can the advisor emit `search_strategy` / a sampler choice?** And can it emit
anything that reaches Step 1's search space?

### 3.2 The validation surface
What does `watcher_policies.json` declare governable, and what code enforces each declaration?
Distinguish **declared** from **enforced** for every entry. Confirm or refute that
`parameter_application` has no implementation.

### 3.3 The application surface
Does *anything* consume an advisor recommendation and change a live parameter, config, or
manifest? Trace producer → proposal → validation → application → consumer. Name the exact hop
where the chain terminates.

### 3.4 Selfplay control — the question that motivated this audit
**Does the Strategy Advisor control selfplay, in any operative sense?** Michael's reasoning:
*if* the advisor controls selfplay, then multi-sampler selection becomes a real requirement
rather than a dormant one. Establish whether the premise holds:
- what selfplay parameters the advisor can emit;
- whether any are consumed;
- whether `propose_transform_update` and the `chapter_13_acceptance.py:224` promotion seam are
  the only breaks, or whether there are others.

### 3.5 The four-hop chain from the February proposal
`docs/PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md` describes: *advisor recommends
"random" → WATCHER validates → dispatch passes `--strategy random` → window_optimizer uses
RandomSampler.* **Which of those four hops exist today?** The fourth is now known broken
(`STRATEGY_ORIGIN_AUDIT.md`). Establish hops one through three.

### 3.6 Declared-but-disconnected inventory
Every parameter an agent is **declared** able to propose (manifests, grammars, policy files)
that **cannot reach execution**. This is the §0.5 dead-dimension pattern applied to the
autonomy layer. Two are known — sieve thresholds and the sampler choice. **Report any others.**

### 3.7 What the constraints actually enforce
For Beta's eventual ruling: what do the GBNF grammars and Pydantic models **actually
guarantee** about advisor output? Token-level constraint, schema validation, range checks,
enum restriction? Be concrete — this is the evidence base for whether the original restriction
still fits.

## 4. Out of scope

- **Do not implement anything.** Not the application seam, not sampler wiring, not schema
  fields.
- Do not modify grammars, manifests, policy files or schemas.
- Do not re-audit TRSE, Chapter 1, the threshold path, the dataset contract, Chapter 2 or the
  strategy origins — **cite** them.
- Do not run WATCHER, the pipeline, the sieve or any GPU kernel.
- **Do not recommend removing anything.** If something appears unused, report it as
  *declared-but-disconnected* and cite what documents its intent. Removal is a Beta ruling,
  never an audit conclusion.

## 5. Verification-integrity controls (VIR-1…6)

- **execution proof** — every claim carries a `file:line` anchor read this session.
- **clean control (VIR-2)** — state which hops you verified as **correctly connected**. A
  report listing only breaks gives no evidence the rest was checked.
- **fault-injection control** — n/a for a read-only audit; **say so** rather than omitting it.
- **completion sentinel (VIR-3)** — end with explicit `PASS | FAIL | UNAVAILABLE | INCOMPLETE`
  and a coverage table. Anything not reached is `INCOMPLETE`, never silently absent.
- **unavailable-observer (VIR-5)** — anything unverifiable without executing something is
  `UNAVAILABLE`, not assumed.
- **audit claim scope (VIR-6)** — declare searched and unavailable surfaces. **The repository
  is not the system**: host state, deployed copies and runtime artifacts are separate surfaces.

## 6. Deliverable

`docs/STRATEGY_ADVISOR_AUDIT_v1.md`:

1. **Chain diagram** — emit → validate → apply → execute, with each hop marked
   PRESENT / BROKEN / ABSENT and anchored.
2. **Emit-surface inventory** — every advisor-emittable field, with grammar/Pydantic agreement.
3. **Declared-vs-enforced table** for `watcher_policies.json`.
4. **Selfplay control verdict** (§3.4) — does the advisor control selfplay, yes or no, with
   evidence.
5. **The February four-hop chain** — which hops exist.
6. **Declared-but-disconnected inventory** (§3.6).
7. **What the constraints enforce** (§3.7) — evidence for Beta's ruling on whether the original
   restriction still fits.
8. **Open questions Beta must rule on** — do not answer them yourself.
9. **Coverage table + completion sentinel.**

Then STOP for Team Alpha review.
