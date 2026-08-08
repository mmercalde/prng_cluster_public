# CLAUDE CODE REPORT — THE OPTUNA → FULL-SEED-SWEEP PIVOT

**Brief:** `docs/CLAUDE_CODE_INSTRUCTIONS_OPTUNA_PIVOT_SEARCH.md`
**Host:** VM101, `~/distributed_prng_analysis`, repo HEAD `27ae7a9`
**Date:** 2026-08-07
**Mode:** READ-ONLY. Nothing launched, nothing committed, no production file edited.
Search order followed as directed: **governance trail → chapters → code.**

---

## 0. HEADLINE

**Michael's recollection is substantially CORRECT, and it is anchored in a real, ruled,
implemented decision — but one word in it was struck by Team Beta, and that word is what
makes today's wiring look like a contradiction.**

The pivot happened on **2026-03-15 (S145)**. It is `PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md`.
Its own title is *"Progressive Empirical Sweep of Seed IDs 0→2³² for CA/java_lcg."* It replaced a
**rejected** predecessor literally named `PROPOSAL_S145_Complete_Seed_Space_Sweep.md`.

Three corrections to the recalled frame, each evidenced below:

1. **The pivot was never "Optuna → sweep."** It was **"sampled partial range → progressive
   chunked sweep of the full space, *with Optuna carried across sessions*."** Optuna was
   deliberately **retained and strengthened** by the same proposal (cross-session study resume).
   Optuna and the seed sweep are **orthogonal axes**, never alternatives — §3.
2. **Team Beta explicitly REJECTED the "complete / exhaustive" framing**, the "practically
   sufficient coverage" conclusion, and "Step 1 retired permanently." The 0→2³² target survived
   only as an **empirically-justified working hypothesis**, because java_lcg's real state space
   is **2⁴⁸**, not 2³².
3. **The Fantasy 5 hypothesis in the brief is REFUTED**, not merely unconfirmed — §2.4.

**And the finding neither the brief nor the recollection anticipated:** the sweep **completed its
governed 2³² target on 2026-04-24**, then **kept advancing with no upper bound**, and the live
cursor `16,106,127,360` is now **3.75× outside the frozen artifact's seed domain**. A miner-backed
run launched at that cursor **fails closed at publication** — `utils/run_finalizer.py:533`. See §4,
which is the operationally consequential section of this report.

---

## 1. QUESTION 1 — WAS THE PIVOT EVER IMPLEMENTED?

### 1.1 Answer

**A pivot was implemented. The pivot described in the recollection — a Step-1 path that bypasses
Optuna — was NOT, and no evidence exists that one was ever written, in any commit, in any deleted
file, at any point in the repository's history.**

What was implemented (commits `3940517`, `ad5ab8d`, both 2026-03-15) is the **S145-R1 progressive
sweep framework**: chunked full-space coverage with a persistent cursor, a cross-run survivor
accumulator, and **explicit Optuna continuity across chunks**.

### 1.2 Evidence — implemented (positive)

| artifact | anchor |
|---|---|
| Proposal | `docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md` (352L) |
| Framework commit | `3940517` *"feat(s145-r1): progressive sweep framework"* |
| Accumulator commit | `ad5ab8d` *"feat(s145-r1v2): NPZ accumulator validated — smoke test passed"* |
| Patch corpus (applied, forensic) | `apply_s145r1_progressive_sweep.py`, `apply_s145r1_npz_accumulator.py`, plus 5 `fix_s145r1_*.py` |
| Session record | `docs/SESSION_CHANGELOG_20260315_S145.md`; `docs/TODO_MASTER_S145.md:159-166` |
| Coverage tracker (the cursor) | `agents/watcher_agent.py:1662-1700`; `database_system.py:330-369` |
| Merge policy still cited as live law | `docs/PROPOSAL_S172_RANGE_MINER_v1_4_4.md:178` — *"Merge policy: highest score per seed (TB ruling S145-R1)"* |

The manifest values the brief flagged as contradicting the recollection are in fact **the pivot's
own implementation**, prescribed line-by-line in the proposal:

> `PROPOSAL_S145_R1…md:194-201` — §5.2 *"`agent_manifests/window_optimizer.json` — Four Values"*:
> `max_seeds` `10000000` → **`1073741824`**, `window_trials` `100` → `50`,
> `timeout_minutes` `240` → `900`, `enable_pruning` `false` → `true`.

**`max_seeds = 1073741824` is not "a quarter of 2³² instead of 2³²." It is one of four
deliberate sessions of 2³⁰ that sum to exactly 2³².** The proposal's §3.3 states this as a table:

> | Session | seed_start | seed_end |
> |---|---|---|
> | Run 1 | 0 | 1,073,741,824 |
> | Run 2 | auto | 2,147,483,648 |
> | Run 3 | auto | 3,221,225,472 |
> | Run 4 | auto | **4,294,967,296** |

Live confirmation that the manifest still carries the pivot's values (read this session,
`agent_manifests/window_optimizer.json` → `default_params`): `max_seeds: 1073741824`,
`enable_pruning: true`, `seed_start: 0`, `study_name: ""`.
*(`window_trials` is now `3`, not the proposal's `50` — a later change, outside this brief's scope.)*

### 1.3 Evidence — no Optuna-bypassing path ever existed (negative, and this is the load-bearing part)

**Live strategy registry** — `window_optimizer.py:521-526`:

```python
STRATEGY_CLASSES = {
    'random':       RandomSearch,
    'grid':         GridSearch,
    'bayesian':     BayesianOptimization,
    'evolutionary': EvolutionarySearch,
}
```

Four entries. All are **search strategies over the window configuration**; none is a seed sweep.
There is no `exhaustive`, no `sweep`, no `full_space` member — and `require_supported_strategy`
(`:558-572`) **fails closed on an unknown name**, refusing to fall back.

**`--strategy` argparse** — `window_optimizer.py:1301-1305`: `choices=['bayesian','random','grid','evolutionary']`.
Only `bayesian` is functional; the other three are **GATED, not deleted** (`[S178 P0-2]`, `:1297-1300`),
because their `search()` signatures do not accept the kwargs `optimize()` forwards. **No
non-Bayesian *exhaustive* value has ever been accepted here.**

**History searches (`git log --all`), run this session:**

| query | result |
|---|---|
| `-S "4_294_967_296"` | **no hits, ever** |
| `-S "exhaustive_sweep"` | **no hits, ever** |
| `-S "no_optuna"` | **no hits, ever** |
| `-S "full_space"` | **no hits, ever** |
| `-S "--exhaustive"` | **no hits, ever** |
| `-S "full_sweep"` | `0101306` (Initial commit) only |
| `-S "4294967296"` | 7 commits — all S172 finalizer/seed-domain work (`46a3828`, `a63c361`, `3e8580a`), the behaviour model, and cleanup commits. **None is a Step-1 sweep loop.** |
| `-S "ExhaustiveSearch"` / `-S "'exhaustive'"` | `0101306`, `d14dcdd`, `eae758b` only |
| `--grep` on `exhaustive\|full sweep\|2^32\|full seed\|seed space\|s145` (-i) | 7 commits, all identified above or S147/S172 |

**The `ExhaustiveSearch` hits are confirmed unrelated to Step 1**, as the brief suspected.
`ExhaustiveSearchConfig` lives in `advanced_search_manager.py:14`, consumed only by
`modules/advanced_research.py`, `attack_390_sequences.py`, `system_core.py`,
`historical_analysis_real.py`. Verified this session: **it appears in no
`agent_manifests/*.json` and nowhere in `agents/watcher_agent.py`** — so it is unreachable from
`STEP_SCRIPTS`/`STEP_MANIFESTS`. Its only contact with the sweep is *read-only*:
`advanced_search_manager.py:370` calls `get_exhaustive_progress(search_id)`. It never writes the
cursor and never launches a sieve.

**Conclusion for Q1:** the pivot was implemented as a *coverage* mechanism layered **on top of**
Optuna, never as a *replacement* for it. **No evidence found** — in HEAD, in any deleted file, or
in any commit reachable from `--all` — of a Step-1 code path that sweeps seeds without Optuna.

---

## 2. QUESTION 2 — WAS IT EVER RULED ON OR DECIDED?

### 2.1 Answer

**YES — and this is the strongest evidence in the report. There are two rulings, and the
distinction between them is exactly the distinction the brief asked for.**

Using the brief's own four categories:

| category | verdict |
|---|---|
| RULED and IMPLEMENTED | ✅ **The progressive chunked sweep with retained Optuna continuity** (S145-R1) |
| RULED but NOT implemented | ✅ **The post-sweep sufficiency analysis** — mandated by TB, §7 of the proposal, four required analyses. **No evidence found that any of the four was ever performed.** |
| PROPOSAL / analysis only | — |
| **RULED and REJECTED** | ✅ **The "complete / exhaustive 2³² sweep" frame itself** — `PROPOSAL_S145_Complete_Seed_Space_Sweep.md`, rejected outright |

### 2.2 The rejected predecessor

`docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:5`:

> **Supersedes:** PROPOSAL_S145_Complete_Seed_Space_Sweep.md (rejected)

**That document does not exist at HEAD and never entered git history.** Verified two ways:
`ls docs/ | grep -i S145` returns only the R1 proposal, the changelog and the TODO master; and
`git log --all --pretty=format: --name-only --diff-filter=A | sort -u | grep -i s145` — the
complete set of paths ever *added* under that tag — contains **no**
`PROPOSAL_S145_Complete_Seed_Space_Sweep.md`. It survives only as a citation and through the
changelog's record of why it was rejected.

### 2.3 The ruling, verbatim

`docs/PROPOSAL_S145_R1_Progressive_Empirical_Sweep.md:9-23` — **"TB Approval Status"**:

| Component | Status |
|---|---|
| Cross-session survivor accumulation | ✅ Approved |
| Merge by best per-seed `score` | ✅ Approved |
| WATCHER fresh-study invariant patch | ✅ Required before resume works |
| **"Complete 32-bit sweep" claim for java_lcg** | **❌ Rejected** |
| **"Practically sufficient coverage" conclusion** | **❌ Rejected — deferred to post-sweep** |
| **"Step 1 retired permanently"** | **❌ Rejected** |

And `:38-42`, the proposal's own **"What this is not"**:

> - Not a mathematically exhaustive sweep of java_lcg (state space is 2^48)
> - Not proof that the CA ADM seeds only in 0→2^32
> - Not grounds to retire Step 1 permanently

`docs/SESSION_CHANGELOG_20260315_S145.md:41-64` records the review as a discrete event —
*"S145 proposal (original) — rejected by TB"* (line 28 table row, status ❌) — and enumerates the
five errors, of which #1 and #5 are decisive:

> 1. **2^32 collapse claim wrong** — Java LCG multiplication propagates lower 16 bits into upper
>    bits after one step… Mathematical space is 2^48.
> 5. **"Retire Step 1 permanently"** — rejected, depends on invalid exhaustion claim

**Alpha's pushback was accepted, but only as hypothesis** (`:61-64`):

> TB accepted this as a "working hypothesis requiring post-sweep validation" — **not a pre-sweep
> conclusion.**

### 2.4 The Fantasy 5 hypothesis — REFUTED

The brief asked this to be confirmed or refuted with evidence. **Refuted.**

A full-tree content search (`/bin/grep -ril` over `docs/`, which reaches gitignored `.json`-adjacent
material the shell `grep` wrapper would skip) returns **11 files** mentioning Fantasy 5. **Every
substantive hit is the same sentence** — the *skip rationale* from the CA draw procedures, that the
evening session draws D3, D4, Fantasy 5 and Daily Derby together so other games' outputs sit
between observable Daily 3 values:

`CHAPTER_2_SOURCE_MAP_v1.md:477` · `CHAPTER_1_WINDOW_OPTIMIZER.md:242` ·
`CHAPTER_2_BIDIRECTIONAL_SIEVE.md:407` · `PIPELINE_BEHAVIOUR_MODEL.md:791` ·
`CHAPTER_1_AUDIT_v1.md:368` · `CLAUDE_CODE_INSTRUCTIONS_CHAPTER_1_P0_CORRECTION.md:112` ·
`CLAUDE_CODE_INSTRUCTIONS_CHAPTER_2_RESTORE.md:62` · `TFM_PROJECT_FACTS_SKILL.md` (§0.4)

The only other hit is `TEAM_ALPHA_DATASET_LIFECYCLE_FINDINGS.md:139`, noting a
`fantasy5_scraper.py` as an **uninvoked** stray file.

**There is no Fantasy 5 proposal in the repository, no F5 seed-space analysis, and no "no Optuna"
language anywhere in `docs/`.** The recalled full-sweep frame does not trace to a Fantasy 5
document; it traces to **S145 / S145-R1**, which is a **daily3 / java_lcg** document throughout —
its title names `CA/java_lcg` explicitly. The hypothesis that the frame was "scoped to F5 and never
back-ported to the daily3 path" is therefore not just unsupported but **inverted**: the frame was
authored *for* the daily3 java_lcg path, ruled on there, and implemented there.

---

## 3. WHY TODAY'S WIRING IS NOT A CONTRADICTION

This is not one of the four questions, but it is the reconciliation the brief was reaching for, and
it is answerable from code read this session.

**Optuna and the seed sweep operate on different axes and always have.** Optuna searches the
**window configuration** (window size, per-direction thresholds, skip bounds, offset). The seed
range is a **run-level constant** that every trial sieves identically.

Evidence that the seed range is per-run, not per-trial:

| fact | anchor |
|---|---|
| The seed range is printed **once**, in the run banner, before the trial loop is entered | `window_optimizer_integration_final.py:2646` — `print(f"Seed range: {seed_start:,} → {seed_start + seed_count:,}")`, immediately above `Strategy:` and `Max iterations:` |
| The run identity embeds `seed_start`, not a trial number | `:2604` and `:2975` — `run_id=f"step1_{prng_base}_{int(seed_start)}"` `# [S142-C] canonical run_id, no suffix` |
| Coverage is written **once per run**, spanning the whole chunk | `:2608-2609` — `seed_range_start=int(seed_start)`, `seed_range_end=int(seed_start + seed_count - 1)` |
| The finalizer receives one contiguous interval for the whole generation | `:2972-2982` — `_finalize_run_d3_5(..., seed_start=int(seed_start), seed_count=int(seed_count), ...)` |
| One DB row per run, confirmed empirically | 15 rows for 15 runs — §5 |

So *"sampled window configurations with Optuna over partial seed ranges"* and *"sweeping the full
seed space for a specific draw"* were **never mutually exclusive**. The pivot changed the second
axis (partial range → progressive full-space coverage via `max_seeds` + the auto-advancing cursor)
and left the first axis in place **by explicit TB requirement** — the proposal's entire §3.2 exists
to *preserve* Optuna learning across chunk boundaries via `study_name`, patched live at
`agents/watcher_agent.py:1679-1700`.

**The system today is running exactly what was ruled.** `--strategy bayesian --max-seeds 1073741824`
in tonight's `EXEC CMD` is the S145-R1 configuration, unchanged.

---

## 4. QUESTION 3 — WHAT IS THE INTENDED SEED SPACE FOR `java_lcg` ON daily3?

### 4.1 Answer

**Three different spaces are in play, all three are governed, and they disagree — which is why the
question has felt unanswerable.**

| # | space | value | authority |
|---|---|---|---|
| 1 | **Mathematical** — java_lcg internal state | **2⁴⁸** = 281,474,976,710,656 | kernel constant; TB ruling S145 |
| 2 | **Empirical sweep target** — the S145-R1 goal | **2³²** = 4,294,967,296 | TB conditional approval, *hypothesis only* |
| 3 | **Frozen artifact domain** — what may be published | **[0, 2³²)**, hard wall | Seed-Domain v1.1, `a63c361` |
| — | **Live cursor** | **16,106,127,360** | unbounded; **outside #2 and #3** |

### 4.2 The kernel constraint — `file:line`

The sweep candidate **is the raw 48-bit internal state**. There is no Java `initialScramble`
(`seed ^ 0x5DEECE66D`) anywhere in `prng_registry.py` — searched this session, **zero hits** for
`0x5DEECE66D`, `initialScramble`, or `seed ^`. The kernel takes the candidate and masks it:

`prng_registry.py:969` (constant-skip `java_lcg_flexible_sieve`):
```c
const unsigned long long m = 0xFFFFFFFFFFFFULL;      // 2^48 - 1
```
`:972` — `unsigned long long state = seed & m;`
`:983` — `unsigned int output = (state >> 16) & 0xFFFFFFFF;`

Identical in the bidirectional kernel: `:3127` (`m`), `:3131` (`state = seed & m`), with the
multiplier hardcoded at `:3125` — `const unsigned long long a = 25214903917ULL;` (= `0x5DEECE66D`).

**Therefore the mathematically meaningful candidate space is `[0, 2⁴⁸)`.** Any candidate ≥ 2⁴⁸
would be silently aliased by `& m` onto a smaller one — a wrap the sweep would never notice, though
at 16.1 billion the cursor is nowhere near that boundary.

Governance concurs, and had already corrected this: `PROPOSAL_S145_R1…md:53-67` (§2.1 *"The 2^32
Collapse Claim — Rejected"*), and `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md:1135` finding **F-6** —
*"§1.1's '32-bit internal state' is wrong for `java_lcg` — the state is 48-bit; 32 bits is the
extracted output"*, anchored to `prng_registry.py:969, :983`.

### 4.3 The uint32 domain wall — and why the live cursor violates it

The **frozen 22-array NPZ contract stores `seeds` as uint32**. Seed-Domain v1.1 (ruled `a63c361`)
resolved the mismatch **by honest labelling rather than a storage migration**:

`docs/PIPELINE_BEHAVIOUR_MODEL.md:1137-1160` §15.3:

> The sweep therefore covers the **`high16 = 0` stratum — 1 part in 65,536** of the state space…
> *"This is a labelling problem, not a storage problem… **survivor validity rests on sieve
> selectivity rather than search extent**. So the artifact stays `uint32` and declares honestly
> which stratum it is."*

Nine frozen sidecar constants encode it, including `seed_domain_end_exclusive = 4294967296` and
`exhaustive_over = "high16=0 stratum only"`.

**This is enforced in live code, fail-closed:**

`utils/run_finalizer.py:277`
```python
SEED_DOMAIN_EXCLUSIVE_MAX = 2 ** 32
```

`utils/run_finalizer.py:533-539` (`_validate_declared_coverage`):
```python
if not (0 <= start < SEED_DOMAIN_EXCLUSIVE_MAX):
    raise CoverageValidationError(
        f"seed_start {start} is outside the frozen uint32 seed domain "
        f"[0, {SEED_DOMAIN_EXCLUSIVE_MAX}). The artifact stores "
        f"`seeds: uint32`; widening the domain requires a separately "
        f"governed schema revision and is out of D3.5 scope.")
```
plus `:547` (whole-interval overflow) and `:571` (per-candidate seed).
`CoverageValidationError` derives from `RunFinalizerError` → `RuntimeError`, **deliberately not
`ValueError`** (`:288-295`), precisely so it cannot be swallowed by the legacy accumulator's
`except ValueError` fallback arm.

### 4.4 So: is `seed_start = 16,106,127,360` meaningful?

**It is inside the mathematically real space, and outside every governed one. It is not walking
past the end of the PRNG — it is walking past the end of the publishable artifact.**

- vs. **2⁴⁸** (real state space): well inside — 16.1 billion of 281.5 trillion, **0.0057%**. Not a wrap.
- vs. **2³²** (the S145-R1 governed target): **3.75× beyond it.** The target was met and passed.
- vs. **the frozen uint32 wall**: **`16,106,127,360 > 4,294,967,296` → `_validate_declared_coverage`
  raises `CoverageValidationError` at `utils/run_finalizer.py:533`.**

**Tonight's gate-12 run carried exactly that value** —
`logs/gate12_prodshape_20260807_180116.log:43`:
`… --max-seeds 1073741824 … --seed-start 16106127360 … --use-range-miner …`
preceded at the same second by
`[COVERAGE] java_lcg: prior coverage up to 16,106,127,360 — next seed_start=16,106,127,360`.

**The failure is latent, not yet observed.** Searched this session: `/bin/grep -rn "outside the
frozen uint32 seed domain" logs/*.log` → **no hits**. That run stalled at dispatch (the subject of
`docs/CLAUDE_CODE_REPORT_S172_GATE12_DISPATCH_STALL.md`) and never reached publication. Stated
under VIR-3 discipline: this is a **code-path derivation from the live constant and the live
argument, not an observed failure.** But it is a sieve trial's full duration ahead of the guard —
the rejection would land **at the finalizer, after the GPU work**, which is the expensive place to
discover it.

### 4.5 The cursor has no upper bound — root cause

`database_system.py:330-364`, `get_next_seed_start`:

```python
result = conn.execute(
    'SELECT MAX(seed_range_end) FROM exhaustive_progress WHERE prng_type = ?',
    (prng_type,)).fetchone()
if result and result[0] is not None:
    next_start = int(result[0])
    ...
    return next_start
```

**No comparison against 2³², 2⁴⁸, `SEED_DOMAIN_EXCLUSIVE_MAX`, or any completion predicate.**
The `chunk_size` argument is documented `"logged for context only"` (`:340`) and is indeed never
used in the body. The consumer, `agents/watcher_agent.py:1670-1676`, assigns the result to
`final_params['seed_start']` unconditionally whenever `_next_start > 0`.

The S145-R1 design was **four sessions with an operator in the loop each time** (proposal §6.3 —
set `study_name` manually before runs 2, 3, 4, commit after each). **The auto-advance was built to
carry the cursor between those four runs; nothing was built to stop it at the fourth.** The DB
confirms it did not stop — §5.

---

## 5. QUESTION 4 — WHAT DOES THE COVERAGE DB SAY?

Read-only via `sqlite3` with `mode=ro` URI on `prng_analysis.db` (98 KB… `106496` bytes, mtime
2026-08-02 16:00).

### 5.1 Schema

```sql
CREATE TABLE exhaustive_progress (
    search_id TEXT NOT NULL, prng_type TEXT NOT NULL, mapping_type TEXT NOT NULL,
    seed_range_start INTEGER NOT NULL, seed_range_end INTEGER NOT NULL,
    seeds_completed INTEGER DEFAULT 0, best_score REAL, best_seed INTEGER,
    last_updated TEXT NOT NULL,
    PRIMARY KEY(search_id, prng_type, mapping_type, seed_range_start))
```

**No CHECK constraint, no upper bound, no completion flag.** The schema cannot express "the sweep
is done."

### 5.2 Contents — all 15 rows

**Exactly one `prng_type` (`java_lcg`) and one `mapping_type` (`bidirectional`). 15 rows.**

| # | seed_range_start | seed_range_end | seeds_completed | best_score | best_seed | last_updated |
|---|---|---|---|---|---|---|
| 1 | 0 | **1,000** | 1,000 | 5.0 | NULL | **2026-08-02T16:00:32** |
| 2 | 1,073,741,824 | 2,147,483,648 | 1,073,741,824 | 0.0 | NULL | 2026-04-23T19:31:58 |
| 3 | 2,147,483,648 | 3,221,225,472 | 1,073,741,824 | 0.0 | NULL | 2026-04-24T20:54:39 |
| 4 | 3,221,225,472 | **4,294,967,296** | 1,073,741,824 | 0.0 | NULL | **2026-04-24T21:13:00** |
| 5 | 4,294,967,296 | 5,368,709,120 | 1,073,741,824 | 0.0 | NULL | 2026-04-24T21:35:23 |
| 6 | 5,368,709,120 | 6,442,450,944 | 1,073,741,824 | 0.0 | NULL | 2026-04-24T21:49:37 |
| 7 | 6,442,450,944 | 7,516,192,768 | 1,073,741,824 | 0.0 | NULL | 2026-04-25T22:46:10 |
| 8 | 7,516,192,768 | 8,589,934,592 | 1,073,741,824 | 0.0 | NULL | 2026-04-26T08:05:12 |
| 9 | 8,589,934,592 | 9,663,676,416 | 1,073,741,824 | 0.0 | NULL | 2026-04-26T17:05:58 |
| 10 | 9,663,676,416 | 10,737,418,240 | 1,073,741,824 | **33.0** | NULL | 2026-05-01T16:05:11 |
| 11 | 10,737,418,240 | 11,811,160,064 | 1,073,741,824 | 0.0 | NULL | 2026-05-01T21:08:54 |
| 12 | 11,811,160,064 | 12,884,901,888 | 1,073,741,824 | 0.0 | NULL | 2026-05-01T23:57:12 |
| 13 | 12,884,901,888 | 13,958,643,712 | 1,073,741,824 | 0.0 | NULL | 2026-05-02T22:42:08 |
| 14 | 13,958,643,712 | 15,032,385,536 | 1,073,741,824 | 0.0 | NULL | 2026-05-03T09:13:06 |
| 15 | 15,032,385,536 | **16,106,127,360** | 1,073,741,824 | 0.0 | NULL | 2026-05-03T11:05:21 |

Aggregate: `count=15, min(start)=0, max(end)=16,106,127,360`, `sum(end−start)=15,032,386,536`.

### 5.3 What the table shows

**a) No recorded upper bound. None.** The brief asked directly; the answer is **no evidence found**
of any upper bound in the schema, in the data, or in the writer. The one number that *looks* like a
terminus — row 4's `seed_range_end = 4,294,967,296` — is exactly 2³², and **row 5 begins at that
same value 22 minutes later.** The governed target was reached at **2026-04-24T21:13:00** and
crossed at **21:35:23**, with no gate, no log warning and no operator prompt. Advancing then
continued for a further **9 days** to 2026-05-03.

**b) A ~1.07-billion-seed hole at the bottom of the range.** Row 1 covers `0 → 1,000`; row 2 begins
at `1,073,741,824`. **The interval `[1,000, 1,073,741,824)` is unclaimed.** Note the primary key is
`(search_id, prng_type, mapping_type, seed_range_start)` and the writer uses `INSERT OR REPLACE`
(`database_system.py:310`), so **any run starting at seed 0 overwrites row 1 entirely.** That has
demonstrably happened: `docs/PROVENANCE_DISPOSITION_ACCUMULATOR_20260725.md:238-240` recorded row 1
as `0 → 425,000,000` at HEAD `70cd6f0` (2026-07-25) and attributed an *earlier* truncation to
`reset_coverage_s152.py`; it is now `0 → 1,000`, last written **2026-08-02T16:00:32**. **The
low-range coverage record has been overwritten at least twice by short test runs, and each
overwrite silently enlarged the hole.**

**c) Quality columns are inert.** `best_seed` is **NULL on all 15 rows**. `best_score` is `0.0` on
**13 of 15** — the exceptions being row 1 (`5.0`) and row 10 (`33.0`). This matches the 2026-07-25
finding and means **the table records extent only, never yield.** That is materially consequential:
the S145-R1 §7 sufficiency analysis TB mandated is a **yield-decay analysis**, and the tracker
never captured the yield it would need.

**d) Coverage sums are not coverage.** `sum(end − start) = 15,032,386,536` counts the declared
spans; because of the hole, actual claimed coverage of `[0, 16,106,127,360)` is
`1,000 + 14 × 1,073,741,824 = 15,032,386,536` — i.e. the sum is correct only because the hole is
excluded from it, **not** because the range is contiguous. **Nothing in the system computes or
checks contiguity**; `utils/run_finalizer.py:47` and `docs/CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md:240-248`
both place gap detection **explicitly outside D3.5**, assigning it to *"a separate coverage-ledger
deliverable"* which — **no evidence found** — has never been written.

---

## 6. SUMMARY TABLE

| Q | Answer | Key evidence |
|---|---|---|
| **1. Implemented?** | **A pivot was — but not the recalled one.** Progressive chunked full-space sweep **retaining** Optuna: `3940517`, `ad5ab8d`. **No Optuna-bypassing Step-1 path has ever existed**, in HEAD or history. | `PROPOSAL_S145_R1…md:194-201`; `STRATEGY_CLASSES` `window_optimizer.py:521-526`; `git log -S` ×9 |
| **2. Ruled?** | **YES — ruled, and the "complete sweep" frame was REJECTED.** Predecessor `PROPOSAL_S145_Complete_Seed_Space_Sweep.md` rejected outright (never in git). R1 approved conditionally with "complete 32-bit sweep", "practically sufficient", and "retire Step 1" all struck. **Fantasy 5 hypothesis REFUTED.** | `PROPOSAL_S145_R1…md:5, :9-23, :38-42`; `SESSION_CHANGELOG_20260315_S145.md:41-64` |
| **3. Seed space?** | **2⁴⁸ mathematically** (`state = seed & 0xFFFFFFFFFFFF`, no `initialScramble`); **2³² empirical target** (hypothesis only); **[0, 2³²) hard artifact wall**. `seed_start = 16,106,127,360` is **inside 2⁴⁸ but 3.75× outside the publishable domain** → fails closed at the finalizer. | `prng_registry.py:969, :972, :3125, :3127, :3131`; `utils/run_finalizer.py:277, :533, :547, :571`; `PIPELINE_BEHAVIOUR_MODEL.md:1137-1160` |
| **4. Coverage DB?** | 15 rows, `java_lcg`/`bidirectional` only, `0 → 16,106,127,360`. **No recorded upper bound anywhere.** 2³² crossed 2026-04-24T21:35. **~1.07B-seed hole** `[1,000, 1,073,741,824)`. `best_seed` NULL ×15, `best_score` 0.0 ×13. | `prng_analysis.db.exhaustive_progress`, read `mode=ro`; `database_system.py:303-369` |

---

## 7. WHAT IS **NOT** ESTABLISHED

Stated explicitly per VIR-3/VIR-6, so nothing here is read as stronger than it is.

- **No evidence found** that the S145-R1 §7 post-sweep sufficiency analysis (yield decay, seed
  distribution, quality-vs-range, the <5% threshold) was ever performed. TB made it a precondition
  for any sufficiency claim. Searched `docs/` for it; absent. This is an **absence claim over
  `docs/` + repo only** — a ser8 pre-repository archive or an uncommitted host artifact would not
  be visible to it.
- **No evidence found** of any decision, ruling or discussion authorizing the cursor to advance
  **past** 2³². The overshoot appears to be the **absence** of a stop condition
  (`database_system.py:330-364`), not a decision to continue. **I did not find a ruling that
  forbids it either** — the question appears never to have been put.
- **No evidence found** of the *"separate coverage-ledger deliverable"* that
  `CLAUDE_CODE_INSTRUCTIONS_S172_PHASE5_D3_5.md:245-248` assigns gap detection, overlap handling
  and resume policy to.
- **The `CoverageValidationError` at the live cursor is derived, not observed.** No log contains it.
  It follows from `16,106,127,360 > 2**32` against `utils/run_finalizer.py:533` on the D3.5
  publication path, which `window_optimizer_integration_final.py:2972-2982` invokes for a
  miner-backed run. I did not execute it, and nothing in this brief authorized me to.
- **`window_trials: 3`** in the live manifest differs from S145-R1's prescribed `50`. I did not
  trace when or why it changed; outside this brief.
- The 2026-07-25 provenance document's row-1 figure (`0 → 425,000,000`) is **superseded** by this
  session's read (`0 → 1,000`). Per §1.2 of the skill, I am reporting the live read and flagging
  the earlier one as expired rather than relaying it.

---

## 8. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** every table, constant and line quoted was produced by a command run this
  session on VM101 and is reproduced above with its `file:line` or query output.
- **clean control:** negative `git log -S` queries (`4_294_967_296`, `exhaustive_sweep`, `no_optuna`,
  `full_space`, `--exhaustive`) returned empty **in the same batch** as positive queries that
  returned hits — so empty output is a real absence, not a broken invocation.
- **fault-injection control:** not applicable — read-only investigation, no detector authored.
- **completion sentinel:** all four questions answered; §7 enumerates every unresolved item.
- **unavailable-observer behavior:** none encountered. The DB, git history and all cited files were
  readable. Had any been unreadable it would be reported `UNAVAILABLE`, not passed over.
- **audit claim scope:** the repository at HEAD `27ae7a9`, its full git history via `--all`, `docs/`
  in full, the live `prng_analysis.db`, and `logs/gate12_prodshape_20260807_180116.log`.
- **searched surfaces:** **`docs/` and the governance trail (searched FIRST, per §1.1 and the VIR-6
  addendum)** — `TB_RULING_*` (12 files listed), `PROPOSAL_*` (60), `TEAM_ALPHA_*` (51),
  `CLAUDE_CODE_CORRECTION*`, `SESSION_CHANGELOG_*`, `PROJECT_FILE_CATALOG.md`,
  `PIPELINE_BEHAVIOUR_MODEL.md`, and the skill · **chapters** — `CHAPTER_1_WINDOW_OPTIMIZER.md`,
  `CHAPTER_2_BIDIRECTIONAL_SIEVE.md`, `CHAPTER_8_PRNG_REGISTRY.md`, `CHAPTER_1_AUDIT_v1.md`,
  `CHAPTER_2_SOURCE_MAP_v1.md` · **code** — `window_optimizer.py`,
  `window_optimizer_integration_final.py`, `agents/watcher_agent.py`, `database_system.py`,
  `prng_registry.py`, `utils/run_finalizer.py`, `advanced_search_manager.py`, the `apply_s*`/`fix_s*`
  patch corpus · **git history** — `log -S` ×9, `--grep` ×6, `--diff-filter=A` path enumeration ·
  **gitignored/live host** — `agent_manifests/window_optimizer.json` and `prng_analysis.db` read
  directly from the filesystem (both invisible to a clone-based audit) · `/bin/grep` used throughout
  rather than the shell wrapper, which honours `.gitignore` and would have skipped `*.json`.
- **unavailable surfaces:** **ser8 pre-repository archives** (not reachable from VM101 in this
  session) · **host systemd/cron state** (not inspected — no bearing on the four questions) · **the
  rejected `PROPOSAL_S145_Complete_Seed_Space_Sweep.md` itself**, which was never committed and is
  known only through the R1 proposal's citation and the S145 changelog's summary of its five errors.
- **governance trail searched:** yes — first, before code, as directed.
- **chapters searched:** yes — Chapters 1, 2 and 8, plus both audits and the source map.

---

## 9. READ-ONLY COMPLIANCE — one disclosed side effect

Nothing was launched, committed, pushed, or edited in production. This report is the only file I
authored. One incidental artifact must be declared rather than left for someone to find:

**`prng_analysis.db` is in `journal_mode = wal`.** SQLite creates `-shm` and `-wal` sidecars on
*any* open of a WAL database, including a `mode=ro` URI connection, so
`prng_analysis.db-shm` (32,768 B) and `prng_analysis.db-wal` (**0 B**) now appear as untracked
files. They were not present at session start.

**The database itself is unmodified**, verified after the fact:

```
prng_analysis.db   106,496 bytes   mtime = 2026-08-02 16:00:32.276059163 -0700   (unchanged)
prng_analysis.db-wal     0 bytes   -> zero pending frames; nothing was written
exhaustive_progress: 15 rows, MAX(seed_range_end) = 16,106,127,360  (unchanged)
```

A zero-byte WAL is proof of no write. Both sidecars are disposable and are removed automatically on
the next clean close of the database; they need no action and must not be committed.

---

*Team Alpha (Claude Code) — read-only investigation — 2026-08-07 — VM101 — HEAD `27ae7a9`*
*Nothing launched. Nothing committed. No production file modified. This report is the only file written.*
