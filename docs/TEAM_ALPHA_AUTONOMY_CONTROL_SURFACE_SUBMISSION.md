# TEAM ALPHA → TEAM BETA — autonomy control surface: four chains, four questions

**Re:** `docs/STRATEGY_ADVISOR_AUDIT_v1.md` and `docs/SAMPLER_BEARING_v1.md`. Read-only audits;
no code, config, schema, grammar or documentation changed. Nothing implemented.

**Scope note up front.** These audits are **not** in Beta's approved sequence (Chapter 1 P0 →
Phase 6-P0 → Chapter 2 → remaining chapters). Alpha inserted them to answer a question about
Optuna sampler selection and they expanded. They are **read-only and displaced nothing** —
Chapter 1 P0 tranche 1 is committed at `ddd2ac8` and the chapter track continues — but Beta
should know the order drifted, and **Phase 6-P0 remains the live-data risk Beta prioritised.**
Alpha requests Beta rule on where any resulting work sits relative to it.

**Alpha requests four rulings (§5). Two of the four findings may not be defects at all** — see
§2 Chains C and D. Alpha will not treat a deliberate safety control as a bug.

---

## 1. Correction to Alpha's own prior reporting

Alpha stated repeatedly that the LLM parameter-application seam "does not exist," citing
`diagnostics_analysis_schema.py:76` ("LLM proposals are advisory only"). **That was wrong.**

The seam exists at `agents/watcher_agent.py:1789-1793`: it iterates LLM parameter proposals,
validates each against a policy whitelist, and assigns accepted values into the params dict. It
is reachable — `_build_retry_params` is called at `:2111` from the Step-5 training-health RETRY
path, and its return becomes the next step's params at `:2131`.

Alpha's claim was true of the **search terms used** (`apply_*`, `parameter_application`) and
false of the system. This is the third instance this session of a repo-scoped keyword search
producing a confident false negative — the same failure VIR-6 was adopted to prevent. Recorded
as an Alpha process defect, not incidental.

---

## 2. Four chains, breaking in four different ways

The autonomy layer is **not one chain.** Two LLM proposal producers exist with different fates.

| chain | emit | validate | apply | execute |
|---|---|---|---|---|
| **A** Advisor → selfplay (`max_episodes`, `min_fitness_threshold`) | ✅ | ✅ | ✅ | **✅ WORKS** |
| **A′** Advisor → selfplay (4 other override fields incl. `search_strategy`) | ✅ | partial | **BROKEN** | absent |
| **B** Advisor → `strategy_recommendation.json` → WATCHER | ✅ | ✅ | **ABSENT** | absent |
| **C** diagnostics → Step-5 retry params | ✅ | ✅ | ✅ | **BROKEN** |
| **D** Ch13 proposal → acceptance engine | ✅ | ✅ | **ABSENT** | absent |
| **F** Advisor → Step-1 search space (`search_strategy`) | ✅ | partial | **BROKEN** | absent |

### Chain A — the positive finding
**The Strategy Advisor genuinely controls selfplay for two parameters**, end to end:
`watcher_dispatch.py:481` → `:490` merge → `:150-158` CLI → `selfplay_orchestrator.py:1182-1187`.
This is working autonomy, and it is the **template** the broken siblings sit beside.

### Chain C — the one Alpha considers urgent, and it is a *reporting* defect
Values are validated, accepted, and assigned — then **filtered out** at
`watcher_agent.py:1385-1393` by a step-scoped `allowed_params` list; all six whitelisted
parameters are absent from `agent_manifests/reinforcement.json` `default_params`.

**Consequence:** WATCHER logs `Applied: learning_rate = 0.01` at INFO for a value that never
reaches the training script.

**This is the phantom-adaptation failure the D6 provenance gates were built against, live in the
autonomy layer.** A log line asserting an adaptation that did not occur is worse than an absent
seam: an absent seam fails visibly.

**But the remedy is not obvious, and Alpha will not choose it.** Either the step-scoped filter is
a deliberate safety boundary — in which case the fix is to **stop logging "Applied"** and report
the rejection — or it is an oversight, in which case the parameters should reach the script.
**Alpha cannot determine which from the code.** See §5 Q1.

### Chain D — this may be working exactly as designed
Every branch at `chapter_13_orchestrator.py:398-441` terminates in an outcome string or
`request_approval()`. `record_applied_changes` (`chapter_13_acceptance.py:635`) has **zero
callers**.

**Alpha does not classify this as a defect.** Proposals terminating in `pending_approval` is a
coherent description of a **human-in-the-loop gate**. "Fixing" it could mean removing a control
Beta installed deliberately. See §5 Q2.

### Chains A′ / B / F — bounded
`search_strategy` is one of four override fields read into the dict and never consumed — sitting
directly beside two fields that work. `strategy_recommendation.json` is written *"for WATCHER
consumption"* (`parameter_advisor.py:629`) and **no code reads it**; Chain A works only because
`watcher_dispatch.py:481` calls `advisor.analyze()` in-memory.

### Anchor correction
`chapter_13_acceptance.py:224` has drifted — line 224 is now a `SelfplayCandidate` field block.
The substance holds under a new anchor: `promote_candidate` (`:818`) has zero callers. Alpha's
prior citation should not be verified against the old line number.

---

## 3. Sampler bearing — the answer to the question that started this

**Effort: small for the sampler mechanism, medium for a shippable change.** Sampler construction
is hard-wired but **singular** — `TPESampler` imported once (`window_optimizer_bayesian.py:52`),
constructed once (`:543-547`), passed to `create_study` once (`:616-623`).

**Two findings dominate:**

**(a) The signature problem is moot under the natural implementation.** All four strategies
already route through the same `run_bayesian_optimization` entrypoint. If the three become thin
delegates like `BayesianOptimization`, they inherit its signature, `strategy_contract_gap()`
returns `()`, **and — because a delegate has a real body — this simultaneously closes the
vacuous-pass hole** the origin audit raised (a signature-derived gate going green on a function
returning `{}`). One change, two problems.

**(b) `GridSampler` is mathematically unconstructible here, not merely expensive.** The cartesian
product over live bounds is **7.649 × 10¹⁰** points. Optuna 4.4.0 materialises the entire grid as
a Python list in `GridSampler.__init__` (`_grid.py:120`) and shuffles it (`:124`) — **≈7.2 TiB on
a 15 GiB host.** It cannot be constructed at any trial budget. Per §0.4 this is reported as a
**constraint on how `grid` must be specified**, not a case for retiring it: it would require a
governed coarse discretisation, which is a design decision.

**What is actually gained: little — with one real exception.** There is currently **no way to
answer "is TPE beating random?"** on this objective. Step 1's objective is a survivor count under
loose thresholds — noisy, heavy-tailed, seven dimensions, tens of trials — precisely the regime
where TPE's advantage is smallest and a random control arm is standard practice. A random arm
would also make warm-start falsifiable: today it is impossible to tell whether warm-started TPE
finds good windows or rediscovers the enqueued point.

`CmaEsSampler` is judged likely worthless at current budgets (tens of trials, five of seven
dimensions integer, one categorical CMA-ES would wrongly treat as ordered).

**Timing caution:** any sampler comparison run **before the hybrid skip wire-in** would compare
searches across a space where two dimensions do nothing. Alpha recommends `RandomSampler` be
sequenced **after** skip bounds are live.

---

## 4. What the constraints enforce

Both audits' §7 sections document what GBNF grammars and Pydantic schemas actually guarantee
today. Beta's original restriction — keeping the advisor away from data-pipeline parameters —
was **correct for its time**; Michael's position is that constrained generation plus schema
validation materially changes that risk profile. **Alpha does not decide this** and has recorded
the evidence base for Beta to rule on. See §5 Q4.

## 5. Rulings requested

**Q1 — Chain C.** Is the step-scoped `allowed_params` filter (`watcher_agent.py:1385-1393`) a
**deliberate safety boundary** or an oversight? If deliberate: the fix is to stop logging
`Applied` for a value that is subsequently dropped, and report the rejection instead. If an
oversight: the whitelisted parameters should reach the step. **Alpha recommends the false log
line be corrected either way, and promptly** — it is a live phantom-adaptation report.

**Q2 — Chain D.** Is termination at `pending_approval` the **intended human-in-the-loop gate**?
If yes, `record_applied_changes` having zero callers is correct and nothing should change.
Alpha will not treat a deliberate control as a bug.

**Q3 — Chain A′/F and sampler work.** Given that `search_strategy` sits beside two working
fields, and that the sampler mechanism is a single parameter on one `create_study` call: should
`RandomSampler` be wired as a **control arm** (Alpha's recommendation, sequenced after the skip
wire-in), and should `search_strategy` be connected through the path Chain A already proves?
`grid` requires a discretisation ruling before it is schedulable; `evolutionary` is judged
low-value.

**Q4 — the advisor's data-pipeline restriction.** Does Beta wish to revisit it now that GBNF and
Pydantic constraints are in place, or does it stand? Alpha requests this be ruled explicitly
rather than eroded by individual exceptions.

**Q5 — sequencing.** Where does any of this sit relative to **Phase 6-P0**, which Beta
prioritised as the live-data risk? Alpha's default assumption absent a ruling: **6-P0 first**,
autonomy work after, with the sole exception of Chain C's false log line if Beta agrees it is
urgent.

## 6. VIR note

Both audits: read-only; fault-injection control **n/a and stated rather than omitted**; VIR-6
scope declared (repo + VM 101; rigs not inspected). The Strategy Advisor audit explicitly
recommends the removal of **nothing** and marks items `declared-but-disconnected` with intent
citations where found — and explicitly marks where none was found.
