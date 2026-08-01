# STRATEGY ORIGIN AUDIT — were RandomSearch / GridSearch / EvolutionarySearch ever Optuna-backed?

**Date:** 2026-07-31 · **Host:** VM101 (`michael@192.168.3.177`) · **Tree:** `/home/michael/distributed_prng_analysis`
**HEAD at audit:** `ddd2ac8` (2026-07-31 19:57:19 -0700)
**Mode:** read-only history investigation. No code, config or documentation was modified. This file is the only artifact created.

---

## 0. Verdict

> **(a) Always hand-rolled — Optuna intent was documented but never landed in code.**

Refined statement, because "(a)" alone would misrepresent the intent question:

- **Code:** no version of `RandomSearch`, `GridSearch` or `EvolutionarySearch` — in **any** commit in
  the repository's history, or in **any** pre-git snapshot recoverable from the initial commit — has
  ever contained the tokens `optuna`, `sampler`, `create_study` or `study`. Optuna entered
  `window_optimizer.py` on **2025-11-01**, and entered **only** `BayesianOptimization`.
- **Intent:** Michael's recollection is **corroborated by two committed documents**, which describe
  `search_strategy` as **"Optuna sampler selection"** and assert **"All 4 Optuna samplers
  (TPE, Random, Grid, CmaES) implemented ✅"**. Under `tfm-project-facts` §0.4 a stated intent
  outranks an inference from code shape. The documented design **is** four Optuna samplers.
- **(b) is refuted** — there is no earlier Optuna-backed version that was later replaced.
- **(c) is refuted** — the three were not built as a deliberate Optuna-free fallback path.
  `GridSearch` and `EvolutionarySearch` were *working* hand-rolled code that was **deleted down to
  `return {}`** in mid-November 2025 (§3). A deliberate fallback is not a stub. (The only genuine
  Optuna-free fallback in the module tree is `GaussianProcessBayesianSearch`,
  `window_optimizer_bayesian.py:746` — and that is a *Bayesian* fallback, not these three.)

**Is the remedy comment committed at `ddd2ac8` correct or misleading? — MISLEADING, in part.**
See §6. Its *diagnosis* (signature divergence) is factually correct and well-anchored. Its
*prescription* ("bring the signatures up to the calling convention — at which point the gate clears
itself") is wrong for two independent reasons: it contradicts the only documented statement of
intent, and for two of the three classes it would turn the gate green on a function that returns
`{}`.

---

## 1. History depth (VIR-6)

| Property | Value |
|---|---|
| Shallow clone? | **No** — `git rev-parse --is-shallow-repository` → `false` |
| Total commits | **727** |
| Root commit | `0101306` **"Initial commit"**, 2025-11-29 08:09:49 -0800 |
| Commits touching `window_optimizer.py` (`--follow`) | **39**, all inspected programmatically |

**The git history does not reach the origin of these classes.** The repo begins 2025-11-29, by which
date all four classes already existed in their essentially-current form. The pre-git period is
recoverable **only** because `0101306` committed 29 dated `window_optimizer*` backup files spanning
**2025-10-28 → 2025-11-16**. These are point-in-time snapshots, not a continuous log; gaps between
snapshot timestamps are unobserved. Anything before **2025-11-01 13:26:41** (the earliest
`window_optimizer.py` snapshot) is **unavailable**.

---

## 2. First appearance of each of the four strategy classes

All four appear **together**, already all four present, in the earliest observable snapshot.

**Anchor:** `0101306:window_optimizer.py.bak_20251101_132641` — **2025-11-01 13:26:41**, pre-git.

| Class | Line (in that snapshot) | Initial implementation | Referenced Optuna? |
|---|---|---|---|
| `RandomSearch` | `:151` | **Complete hand-rolled loop** — `bounds.random_config()` × `max_iterations`, tracks best by `scorer.score` | **No** |
| `GridSearch` | `:191` | **Complete hand-rolled nested loop** — `window_sizes × offsets × session_options × skip_ranges` | **No** |
| `BayesianOptimization` | `:242` | Docstring *"Bayesian optimization (simplified version)"*; body is `return RandomSearch().search(...)` with the comment *"Fallback to random search for simplicity"* | **No** |
| `EvolutionarySearch` | `:259` | `population_size`/`mutation_rate` stored but unused; body is `return RandomSearch().search(...)` — *"Simplified - fallback to random"* | **No** |

`/bin/grep -ci optuna` on that whole file returns **0**. At the origin of the four-strategy design
**nothing in `window_optimizer.py` was Optuna-backed**, including Bayesian.

### 2.1 The moment Optuna entered — 70 seconds later, and only into one class

**Anchor:** `0101306:window_optimizer.py.bak_20251101_132751` — **2025-11-01 13:27:51**.

```
:244  from window_optimizer_bayesian import OptunaBayesianSearch
:250  """Bayesian optimization using Optuna TPE"""
:255  self.optuna_search = OptunaBayesianSearch(n_startup_trials=n_initial, seed=None)
:259  print(f"\n⚠️  Optuna not available, falling back to RandomSearch")
:264  return self.optuna_search.search(objective_function, bounds, max_iterations, scorer)
```

In the same file `RandomSearch` (`:151`), `GridSearch` (`:191`) and `EvolutionarySearch` (`:268`) are
**byte-for-byte unchanged** — still hand-rolled, still no Optuna. This is the single most decisive
observation in the audit: at the exact commit where the project acquired the capability to route a
strategy through Optuna, it routed **one** strategy and left the other three alone.

---

## 3. Did any historical version import or call Optuna? — exhaustively, no

### 3.1 Programmatic class-body scan over all 39 commits

Script: extract the source of each `class X(SearchStrategy)` body from `git show <sha>:window_optimizer.py`
for every commit returned by `git log --follow`, and test the body for `optuna | sampler | create_study | study`
(case-insensitive).

**Result:**

```
--- any optuna/sampler token inside Random/Grid/Evolutionary bodies, any commit ---
NONE
```

`BayesianOptimization` matched in **39/39** commits (`optuna` from `0101306` onward; `optuna,study`
from `cd213e9` onward). The three siblings matched in **0/39**.

### 3.2 `git log -S` on sampler names, `--all`, all paths

| Token | Commits |
|---|---|
| `RandomSampler` | **1** — `3723ce4`, and it is a **documentation** commit (§5), not code |
| `GridSampler` | **0 — never appears anywhere in the repository's history** |
| `NSGAIISampler` | **0 — never appears anywhere in the repository's history** |
| `CmaEsSampler` | **0 — never appears anywhere in the repository's history** |
| `TPESampler` | 10 commits, all in `window_optimizer_bayesian.py` / backups / changelogs |
| `create_study` | 10 commits, none in a Random/Grid/Evolutionary body |

### 3.3 Pre-git snapshots — per-file `optuna` counts at `0101306`

```
 0  window_optimizer.py.bak_20251101_132641      <- pre-Optuna
 7  window_optimizer.py.bak_20251101_132751      <- Optuna arrives, BayesianOptimization only
 7  window_optimizer.py.backup / .backup_preheater / (nonjson) / (broken_workflow)
 7  window_optimizer.py.broken_20251114_190847
15  window_optimizer.py.backup_resume_policy_20251115_210906
15  window_optimizer.py.backup_skip_fix_20251116_132138
15  window_optimizer.py                          (as committed at 0101306)
31  window_optimizer_bayesian.py
 0  window_optimizer_methods.py.bak_20251028_182709   (earliest artifact; no strategy classes)
```

Every one of those `optuna` occurrences was verified to sit inside `BayesianOptimization` or the
module-level `BAYESIAN_AVAILABLE` import guard.

### 3.4 The degradation of Grid and Evolutionary — a deletion, not a design

Between **2025-11-14 19:08:47** and **2025-11-15 21:09:06** (bracketed by two snapshots):

| | `…broken_20251114_190847` | `…backup_resume_policy_20251115_210906` |
|---|---|---|
| `GridSearch` | `"""Grid search"""` + full 4-deep nested loop, `📏 GRID SEARCH` banner, best-tracking, returns populated dict | `"""Grid search - not used in integrated mode"""` + `# Placeholder - not used in integrated mode` / `return {}` |
| `EvolutionarySearch` | `"""Evolutionary algorithm"""` + `🧬 EVOLUTIONARY SEARCH` banner, delegates to `RandomSearch().search(...)` | `"""Evolutionary algorithm - not used in integrated mode"""` + `return {}` |

`git log -S"Placeholder - not used in integrated mode" -- window_optimizer.py` returns exactly one
commit — `0101306`, i.e. the text was already present when history began, consistent with the
snapshot bracket above.

**This is the origin of the word "placeholder"** that Chapter 1 §6.4 records. It describes the
*result of a deletion performed in November 2025*, not an original design decision. `RandomSearch`
was **not** degraded — it remains a complete, working hand-rolled loop to this day
(`window_optimizer.py:364-401`).

---

## 4. When and why the signatures diverged

`WindowOptimizer.optimize` forwards four kwargs to `strategy.search` (live call site
`window_optimizer.py:622`). Each was added to `BayesianOptimization.search` **and to the shared call
site** without touching the three siblings:

| # | Commit | Date | Session | kwarg(s) added | Does the message/changelog mention the other three? |
|---|---|---|---|---|---|
| 1 | **`cd213e9`** | 2026-03-04 | S116 | `resume_study`, `study_name` | **No.** Subject: *"fix(S116): resume_study study_name full call chain + window_trials manifest fix"*. Empty body. |
| 2 | `2377228` | 2026-03-07 | S123 | `trse_context_file` | **No.** Body enumerates the chain `run_bayesian_optimization → optimize_window → optimizer.optimize → strategy.search → OptunaBayesianSearch.search` — the siblings are simply not in the author's model of the call graph. |
| 3 | `c6fde66` | 2026-03-13 | S140b | `trial_history_context` | **No.** Subject: *"S140b: step1 trial history, warm-start, downstream feedback loop"*. |

*(`c3c337e`, S115, 2026-03-03 added `resume_study` only to `run_bayesian_optimization()`'s own
signature — one level above `strategy.search` — so it did not itself break the siblings.)*

### 4.1 `cd213e9` is the breaking commit

It is the only one of the three that changed the **call site** from positional-only to
kwarg-forwarding. The diff shows both halves in the same commit:

```
-    def search(self, objective_function, bounds, max_iterations, scorer):
+    def search(self, objective_function, bounds, max_iterations, scorer,
+               resume_study: bool = False, study_name: str = ''):
...
+    return strategy.search(objective, bounds, max_iterations, scorer,
+                           resume_study=resume_study, study_name=study_name)
```

From **2026-03-04** onward, `--strategy random|grid|evolutionary` raises `TypeError` on first call.
That is a **~4½ month** unnoticed outage, explained by the fact that every recorded run used
`bayesian` (§7).

**Why nobody caught it:** the `SearchStrategy` ABC (`window_optimizer.py:339-357`) was never updated
past the pre-S116 four-positional convention, so no signature check had a correct reference to
compare against. This is exactly what `ddd2ac8` records at `:343-348`.

---

## 5. Documented statement of intent — the decisive evidence

Two committed documents, both authored **2026-02-07/08**, i.e. ~4 weeks *before* the signatures
diverged, state the design explicitly.

### 5.1 `docs/SESSION_CHANGELOG_20260207_S63.md:9`

> During Strategy Advisor contract review, identified that `search_strategy` **(the Optuna sampler
> selection parameter)** is fully implemented at the execution layer but **invisible to all advisory
> and governance layers**.

### 5.2 `docs/PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md` (committed at `3723ce4`, 2026-02-08)

| Line | Text |
|---|---|
| `:14` | "The `search_strategy` parameter — **which controls the Optuna sampler used in Step 1 (Window Optimizer)** — is **fully implemented at the execution layer**…" |
| `:20` | "**All 4 Optuna samplers (TPE, Random, Grid, CmaES) implemented ✅**" |
| `:106` | \| `search_strategy` \| `agent_manifests/window_optimizer.json` \| choice \| `bayesian/random/grid/evolutionary` \| **Optuna sampler selection** \| |
| `:288` | "bayesian: Optuna TPE sampler — learns from previous trials…" |
| `:346-348` | "`--strategy random` (already works) ▼ **Optuna RandomSampler (already works)**" |
| `:390` | "End-to-end: Advisor recommends "random" → WATCHER validates → dispatch passes `--strategy random` → **window_optimizer uses RandomSampler**" |

**This is a stated intent, and it is unambiguous: the four strategies are four Optuna samplers.**
It is also **factually false about the code** at the moment it was written — `RandomSearch` was a
hand-rolled loop and `GridSearch`/`EvolutionarySearch` had returned `{}` for three months. The
proposal's "already works ✅" was asserted from the CLI/manifest surface (`--strategy random` is
accepted and dispatches to a class) without reading the class body.

**Two corrections to the mapping in the brief:**
1. The one documented mapping names **CmaES** for `evolutionary`, not NSGAII. `NSGAIISampler`
   appears **nowhere** in the repository or its history (§3.2). The four-to-four correspondence is
   real and documented; the specific evolutionary sampler is **CmaES per the only written source**.
2. `GridSampler` and `CmaEsSampler` likewise never appear in code — the mapping exists only in prose.

### 5.3 Governance surface agrees

`agent_manifests/window_optimizer.json:142-149` declares `search_strategy` as
`choice: [bayesian, random, grid, evolutionary]` — i.e. the governance layer still offers an
autonomous agent four selectable strategies, three of which raise `TypeError`. (Cross-reference:
`tfm-project-facts` §0.5 — a knob connected to nothing that an autonomous agent could "learn" into.)

### 5.4 Chapter 1

- `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:600-605` §6.4 — the `GridSearch`/`EvolutionarySearch`
  "Placeholder / Not used in integrated mode" table. §6.3 documents `RandomSearch` as a working
  **"(Baseline)"** with its full body. So the chapter's placeholder-vs-working distinction is a
  faithful description of the **post-November-2025-deletion** state, and **not** a statement of
  original design intent. It does not mention Optuna samplers for these three.
- `docs/CHAPTER_1_WINDOW_OPTIMIZER.md:~965-985` reproduces the live `inspect.signature` output and
  states *"Root cause is code rot, not design."* — the same claim as the `ddd2ac8` comment, and
  carrying the same defect (§6).

---

## 6. Is the `ddd2ac8` remedy comment correct or misleading?

**The comment** (`window_optimizer.py:492-508`):

> "…This is code rot, and the remedy is to bring the signatures up to the calling convention — at
> which point the gate below clears itself, because it is derived from LIVE signatures rather than
> from a hardcoded list of broken names."

**Correct in its diagnosis.** The signature divergence is real, `cd213e9`/`2377228`/`c6fde66` did
exactly what the comment says, the stale ABC really is why no check caught it, and the §0.4
non-deletion call was right.

**Misleading in its prescription, on two independent grounds:**

**(i) It contradicts the only documented statement of intent.** §5 establishes that the four
strategies were designed and governed as **four Optuna samplers**. Under that design the repair is
to give `OptunaBayesianSearch` (or a sibling) a **sampler parameter** — `optuna.create_study(sampler=
RandomSampler() | GridSampler() | CmaEsSampler())` — and route `--strategy` to it, exactly as
`window_optimizer_bayesian.py:543,616-620` already does for `TPESampler`. Patching four kwargs onto
three hand-rolled bodies produces something that *runs* but is **not the documented design**, and
would permanently entrench the divergence between what the governance layer advertises ("Optuna
sampler selection") and what executes. A reader who follows the committed remedy will build the
wrong thing.

**(ii) For two of the three, following the remedy produces a green gate on a no-op.**
`GridSearch.search` (`:410-412`) and `EvolutionarySearch.search` (`:484-486`) are
`# Placeholder … return {}`. The gate is derived from **live signatures**, so adding the four kwargs
clears it — while `search()` still returns an empty dict. `optimize()` would then return `{}` to the
integration layer with no `best_config` and no `all_results`, *after* the 26-GPU coordinator had been
constructed, and the contract gate that exists to prevent exactly that class of failure would report
**PASS**. This is a VIR-2 vacuous-pass shape: a signature-derived detector structurally cannot see a
stub body. The comment's "the gate clears itself" is presented as a completeness property; it is
actually the gate's blind spot.

`RandomSearch` is the partial exception — it has a real body, so the signature remedy would yield a
*working* uniform-random search. It would still not be an Optuna `RandomSampler` run, and it would
still be recorded under a strategy name the governance docs define as Optuna-backed — the same
semantic-substitution concern that motivated the `StrategyContractError` fail-closed at `:465-473`.

**Net:** the comment should say that the signature divergence is the *proximate* defect and that the
*designed* implementation, per `PROPOSAL_SEARCH_STRATEGY_VISIBILITY_FIX_v1_0.md:20` and
`SESSION_CHANGELOG_20260207_S63.md:9`, is Optuna sampler selection — and that clearing the signature
gate is necessary but **not** sufficient, because two of the three bodies are empty.

*(Recommendation only. No edit was made. Correcting an in-source comment and the Chapter 1 §6.4 /
"code rot, not design" text is a separate, gated change.)*

---

## 7. What "Optuna-backed" looks like in this codebase

`BayesianOptimization` (`window_optimizer.py:417-476`) **does not use Optuna directly**. It delegates:

```
window_optimizer.py:433   from window_optimizer_bayesian import OptunaBayesianSearch
              :434-436    self.optuna_search = OptunaBayesianSearch(n_startup_trials=…, …)
              :452-455    return self.optuna_search.search(objective_function, bounds, …)
```

The Optuna work lives entirely in `window_optimizer_bayesian.py`:

```
:51-52    import optuna ; from optuna.samplers import TPESampler
:364      class OptunaBayesianSearch
:543      sampler = TPESampler(...)
:616-620  study = optuna.create_study(..., sampler=sampler, ...)
:708,:728 'strategy': 'optuna_bayesian'
:746      class GaussianProcessBayesianSearch      <- sklearn GP, the Optuna-free fallback
:948      def create_bayesian_optimizer(method='auto'|'optuna'|'sklearn')
```

**Two facts follow directly:**

1. **`TPESampler` is the only sampler ever imported.** There is no sampler-selection dispatch
   anywhere. `create_bayesian_optimizer` dispatches on *backend* (`optuna` vs `sklearn`), never on
   *sampler*. So the "four Optuna samplers implemented ✅" claim has no code to point at.
2. **The template for the other three is unambiguous** — an Optuna-backed strategy in this codebase
   is a class in `window_optimizer_bayesian.py` that builds a sampler and calls
   `optuna.create_study(sampler=…)`, with `window_optimizer.py` holding only a thin delegating
   wrapper. That is what `RandomSearch`/`GridSearch`/`EvolutionarySearch` would have looked like,
   and none of them ever did.

**Runtime corroboration:** the only recorded strategy value in a live artifact is
`window_optimization_results.json → "strategy": "optuna_bayesian"`. Nine `window_opt_*.db` Optuna
storages are present in the tree. No artifact was found recording a `random_search`, `grid_search`
or `evolutionary` run — consistent with the three having been uncallable since 2026-03-04 and
non-functional (Grid/Evolutionary) since November 2025.

---

## 8. Timeline

| When | Event | Anchor |
|---|---|---|
| 2025-10-28 | Earliest recoverable artifact; no strategy classes yet | `0101306:window_optimizer_methods.py.bak_20251028_182709` |
| **2025-11-01 13:26:41** | **All four classes exist. Zero Optuna.** Random + Grid working hand-rolled loops; Bayesian + Evolutionary delegate to `RandomSearch` | `…bak_20251101_132641:151,191,242,259` |
| **2025-11-01 13:27:51** | **Optuna arrives — `BayesianOptimization` only**, via `OptunaBayesianSearch`. Other three untouched | `…bak_20251101_132751:244-264` |
| 2025-11-14 → 11-15 | `GridSearch` and `EvolutionarySearch` bodies **deleted** → `# Placeholder … return {}` | `…broken_20251114_190847` vs `…backup_resume_policy_20251115_210906` |
| 2025-11-29 | Git history begins with that state already committed | `0101306` |
| **2026-02-07/08** | **Design intent recorded: four Optuna samplers, "all 4 implemented ✅"** | `SESSION_CHANGELOG_20260207_S63.md:9`; `PROPOSAL_…_v1_0.md:14,20,106,346-348,390` (`3723ce4`) |
| 2026-03-03 | `resume_study` added one level above `strategy.search` | `c3c337e` |
| **2026-03-04** | **BREAKING: call site forwards `resume_study`/`study_name`; only Bayesian updated** | **`cd213e9`** |
| 2026-03-07 | `trse_context_file` added, siblings again untouched | `2377228` |
| 2026-03-13 | `trial_history_context` added, siblings again untouched | `c6fde66` |
| 2026-07-31 | S178 P0-2 gates the three closed; records the (partly misleading) remedy | `ddd2ac8`, `window_optimizer.py:492-508` |

---

## 9. Verification-integrity controls (VIR-1…6)

- **Execution proof:** every claim above is backed by a command run in this session on VM101 with
  output captured — `git rev-parse --is-shallow-repository`, `git log --follow`, `git log -S … --all`,
  `git show <sha>:<path>`, and a Python scan iterating all 39 `window_optimizer.py` commits.
- **Clean control:** the scan reported `OPTUNA-REF` for `BayesianOptimization` in **39/39** commits,
  proving the detector fires when the token is present. It is therefore not vacuous.
- **Fault-injection (positive) control:** the same detector on the same corpus reported `NONE` for
  the three siblings. The pairing of a 39/39 positive and a 0/39 negative on one detector run over
  one corpus is the control.
- **Completion sentinel:** the scan printed its summary line
  (`--- summary: … --- NONE`) — output was not truncated.
- **Unavailable-observer behavior:** pre-2025-11-29 history genuinely does not exist; this is
  reported as **UNAVAILABLE** (§1), not as an absence of Optuna.
- **Audit claim scope:** the origin and Optuna-status of four classes in `window_optimizer.py`,
  the cause of their signature divergence, and the documented intent. **Repo-scoped.**
- **Searched surfaces:** full git history (727 commits, non-shallow) incl. all 39 commits touching
  `window_optimizer.py`; all 29 pre-git `window_optimizer*` snapshots embedded in `0101306`;
  `git log -S` `--all` for `RandomSampler`/`GridSampler`/`NSGAIISampler`/`CmaEsSampler`/`TPESampler`/
  `create_study`; `docs/` tree; `agent_manifests/`; `agent_grammars/`; live
  `window_optimizer.py`, `window_optimizer_bayesian.py`, `modules/window_optimizer.py`,
  `window_optimizer_integration_final*.py`, `docs/window_optimizer_integration_final.py`,
  `agents/contexts/window_optimizer_context.py` (none of the latter five define any of the four
  strategy classes); `window_optimization_results.json`.
- **Unavailable surfaces:**
  - **Pre-2025-11-01 history of `window_optimizer.py`** — no snapshot, no git object. The classes
    may have existed earlier; their earliest *observable* state is the 2025-11-01 13:26:41 snapshot.
  - **Gaps between dated snapshots** (e.g. 2025-11-01 → 11-14, 11-15 → 11-29) are bracketed, not
    continuous. The Grid/Evolutionary deletion is located to a ~26-hour window, not to a commit.
  - **The nine `window_opt_*.db` Optuna storages were not opened.** I did not query their study
    system-attrs, so I make no claim about samplers recorded inside them. (Optuna does not reliably
    persist sampler class in storage, so this is unlikely to be probative either way.)
  - **The `public` remote's history was not fetched or diffed separately** this session; all git
    evidence is from the VM101 working tree's `origin` lineage.
  - **Non-repo surfaces** (host config, uncommitted deployed files, Team Beta correspondence not in
    `docs/`) are outside scope per VIR-6 and were not searched.
- **Termination:** **PASS** — the falsifiable question is answered with anchors; no surface required
  for the answer was inaccessible.
