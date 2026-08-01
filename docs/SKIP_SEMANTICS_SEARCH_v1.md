# SKIP SEMANTICS SEARCH v1 — is the meaning of `skip_min` / `skip_max` for the variable-skip (hybrid) kernels written down anywhere?

**Date:** 2026-08-01 · **Host:** VM 101 `zeus-ubuntu` (`192.168.3.177`) · **Tree:** `/home/michael/distributed_prng_analysis` · **HEAD:** `2042a18`
**Type:** read-only search. No file in the tree was changed; nothing was committed.
**Falsifiable question:** does any document, comment, commit message or design note state what `skip_min` / `skip_max` are *supposed to mean* for the variable-skip (hybrid) kernels?

---

## 0. VERDICT — **FOUND**

**Alpha's assertion that "nobody has written it down" is wrong.** It is written down, in two
independent documents, explicitly scoped to hybrid mode, and it has been in the tree since the
feature was introduced in October 2025.

### 0.1 The primary quotation

`docs/instructions.txt`, inside the section **`## NEW: Hybrid Variable Skip Detection (October
16, 2025)`** (`:1037`) → **`### Parameter Reference (Hybrid-Specific)`** (`:1169`) →
**`#### Standard Parameters (Apply to Hybrid Too)`** (`:1177`):

```
docs/instructions.txt:1182  - `--skip-min INT`: Minimum skip value in pattern (default: 0)
docs/instructions.txt:1183  - `--skip-max INT`: Maximum skip value in pattern (default: 16)
```

The governing phrase is **"in pattern."** The heading chain establishes the scope beyond
argument: this is the hybrid-specific parameter reference, and the sub-heading says these two
standard parameters *apply to hybrid too*. The stated meaning is therefore an **element-wise
bound on the discovered skip sequence** — every value in the pattern lies in
`[skip_min, skip_max]`. That is candidate reading **(a)** from the brief: *bound every value in
a generated pattern*. It is not the mean (b), and it is not "does not map onto patterns" (d).

Corroborated verbatim, independently, in a second document:

```
Cluster_operating_manual.txt:948  --skip-min INT: Minimum skip value in pattern (default: 0)
Cluster_operating_manual.txt:949  --skip-max INT: Maximum skip value in pattern (default: 16)
```

Also present at `instructions.txt:1182-1183` (root copy) and
`instructions.txt.before_results_section:1082-1083` (older revision — so the wording predates
the current file and was carried forward, not invented in a later edit).

Note the declared hybrid default `skip_max = 16`. Chapter 1's own `WindowConfig` example uses
`skip_min=0, skip_max=16` (`docs/CHAPTER_1_WINDOW_OPTIMIZER.md:265-266`) — the same pair. The
Optuna live bounds (`skip_min ∈ [0,10]`, `skip_max ∈ [10,250]`) are a later and much wider
search space layered over a parameter whose documented default range was `[0,16]`.

### 0.2 A second, different documented reading — as an ML *outcome* statistic

A separate lineage of documents reads the same two fields as descriptions of what the sieve
*found*, not what it was told to search:

`docs/PROPOSAL_ML_Architecture_Remediation_v2_0.md` (v2.0.0, 2025-12-30), §2.2 "Skip/Gap
Features", `:150-158`:

| Feature | Meaning |
|---------|---------|
| skip_min | Minimum gap that worked |
| skip_max | Maximum gap that worked |
| skip_range | Hypothesis flexibility |
| skip_entropy | Distribution of successful gaps |
| skip_mean, skip_std | Central tendency of gaps |

> **Tight skip range = stronger hypothesis** (only one gap pattern works)

`config_manifests/feature_registry.json:336, :345` says the same thing in the machine-readable
registry: *"Minimum skip value **found during** sieve analysis (from Step 2)"* / *"Maximum skip
value found during sieve analysis"*.

**This directly corroborates Michael's stated design intent.** The proposal places `skip_min` /
`skip_max` in one table with `skip_mean`, `skip_std` and `skip_entropy` — three of the five dead
placeholder features — and gives the whole group a single purpose: characterising the *shape* of
a survivor's gap structure so ML can rank on it. "Tight skip range = stronger hypothesis" is a
statement that the *spread* of the skip pattern is itself the learnable signal. That is the
"varied skip structure so tree/NN models have something to learn from" half of the intent,
written down, in a proposal document, seven months ago.

The documentary ancestor of those three dead features is also in the tree. The October 2025
hybrid output specification (`docs/instructions.txt:1230-1245`) declares:

```json
{
  "seed": 54321, "family": "xorshift32_hybrid", "match_rate": 1.0,
  "skip_pattern": [5, 5, 3, 7, 5, 5, 8, 4, 5, 5, 5, 5, 3, 7, ...],
  "strategy_used": "Balanced Hybrid",
  "pattern_stats": { "mean_skip": 5.4, "variance": 2.1, "std_dev": 1.45 }
}
```

with the explicit gloss (`:1247-1250`): *"`skip_pattern`: Array of detected skip values (not
single `best_skip`)"* and *"`pattern_stats`: Statistical analysis of skip pattern."*
`mean_skip` / `std_dev` → `skip_mean` / `skip_std`. The features are dead because the producer
was cut on the host side (§3.4), not because they were never specified.

### 0.3 The two readings conflict, and the conflict is also in the tree

The parameter registry states the **input** reading in the same repository:

```
config_manifests/parameter_registry.json:160  "description": "Minimum skip value for sieve search"
config_manifests/parameter_registry.json:166  "description": "Maximum skip value for sieve search"
```

So the tree contains, simultaneously: an **input constraint on the pattern's values**
(`instructions.txt`, `Cluster_operating_manual.txt`, `parameter_registry.json`) and an **output
statistic of the pattern that was found** (`PROPOSAL_ML v2.0`, `feature_registry.json`). Both
are documented; neither cites the other; nothing in the tree adjudicates between them. **That
is the actual gap** — not an absence of stated intent, but two stated intents that were never
reconciled. They are not incompatible in principle (a bound on a pattern and the realised
min/max of that pattern are the same quantity when the bound binds), but they place the
parameter on opposite sides of the kernel.

### 0.4 Why the previous audit missed it

`docs/HYBRID_SKIP_BOUND_AUDIT.md` (commit `808e19b`, 2026-07-31) — a 376-line audit on this
exact question — states at `:318` that the wire-in has semantics *"**whose semantics are
unspecified** (§4.3)"* and at `:259` that *"`skip_min`/`skip_max` is not the hybrid's parameter,
and making it one would be **inventing semantics, not restoring them**."*

That audit's VIR-6 declaration (`:39`) says it performed a *"full-tree literal search for
`skip_min` / `skip_max` … using `/bin/grep`."* `docs/instructions.txt:1182` is inside that
declared searched surface and contains the literal string. **The audit's search reached the
line; its analysis did not use it.** The absence claim was therefore made over a surface that
had already been read — which is a harder failure mode than an unsearched surface, and it is why
"I grepped and found nothing" is not by itself a safe basis for an absence claim.

This is the fourth time this session an "it isn't written down" assertion has been falsified.

---

## 1. What was searched (VIR-6)

**Claim scope:** the VM 101 working tree at HEAD `2042a18` (tracked + untracked), its full git
history, and `/home/michael` outside the repo. **Repo-and-host scoped. Not cluster-scoped.**

### Searched surfaces

| surface | method | result |
|---|---|---|
| `docs/` — all files incl. superseded, `.bak`, patch and correction docs | `/bin/grep -rn` for `skip_min` \| `skip_max` | 60+ hits; all reviewed |
| `docs/instructions.txt` (5,000+ lines, doc dump) | targeted read of §1037-1300 | **primary find** |
| `Cluster_operating_manual.txt` | targeted read `:930-960` | **corroborating find** |
| `docs/CHAPTER_1_WINDOW_OPTIMIZER.md` | read `:180-300` (skip rationale, added `ddd2ac8`) | why-skip-exists found; *what-the-range-means* not stated |
| `docs/BIDIRECTIONAL_SIEVE_MATHEMATICAL_WHITEPAPER.md` | `/bin/grep -i skip` | **zero hits** — the mathematical foundation never mentions skip |
| `docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` **deleted**, recovered at `d14dcdd` (743 lines) | `git show` + read §5 `:262-325` | §5.1-5.4 found (see §2) |
| `git log -S` on `skip_min`, `skip_max`, `expected_skip`, `skip_sequences`, `strategy_tolerances` (`--all`) | ~40 commits per term, all subjects read | no commit message states the hybrid semantics |
| in-code comments/docstrings: `prng_registry.py`, `prng_registry_pre_registry.py`, `range_miner_worker.py`, `sieve_gpu_worker.py`, `hybrid_strategy.py`, `window_optimizer_bayesian.py`, `window_optimizer.py`, `coordinator.py`, `persistent_worker_coordinator.py`, `window_optimizer_integration_final.py`, `utils/canonical_records.py` | direct read | `// Initial guess` recovered (§3.1) |
| `config_manifests/{feature,parameter}_registry.json` | `/bin/grep` (used `/bin/grep`, not the wrapper — memory `grep-wrapper-ignores-json`) | both readings found |
| TB rulings, proposals, changelogs, system map, `SAMPLER_BEARING_v1.md`, `CHAPTER_2_SOURCE_MAP_v1.md`, `THRESHOLD_PATH_AUDIT_*`, `HYBRID_SKIP_BOUND_AUDIT.md` | read | §0.4 |
| `~/` outside repo: `cluster_controller/`, `cluster_shared/`, `backups/`, `automation_tb_loop/`, `bin/`, `~/tfm_skill_v2.md`, `~/SESSION_NOTES_20260102.md`, loose `*.md`/`*.txt` at depth ≤2 | `/bin/grep -rn`, `find` | only NPZ dtype tables; **nothing new** |
| deleted-file recovery | `git log --all --diff-filter=D --name-only` | Chapter 2 is the only deleted doc bearing on skip |

### Surfaces NOT searched (VIR-5 — unobservable is not clean)

- **The rigs.** Not contacted this session. Deployed kernel source on `.122`/`.156`/`.164` was
  **not** compared against this tree. Every kernel statement below is a reading of the VM 101
  tree.
- **No kernel was executed.** No GPU run, no PTX/HIP disassembly, no runtime arg observation.
- **The *California State Lottery Daily & SuperLotto Plus Draw Procedures* (eff. 2021-06-09)** —
  the primary source for the physical model — **is not in the repository** and was not read.
  Alpha's standing request for a ruling on committing it (`TEAM_ALPHA_CHAPTER_2_RECOVERY_
  SUBMISSION.md` G-1) is still open.
- **The public clone** was not fetched; no cross-comparison made.
- Optuna study DBs, `scoring_chunks/*.json` and NPZ artifacts were **not** re-mined — the
  historical-row analysis in `HYBRID_SKIP_BOUND_AUDIT.md` §5 (11,870 variable-skip rows across
  36 distinct recorded pairs) was accepted as prior work, not re-derived.
- Chat transcripts, email, and any out-of-band design discussion.

---

## 2. What the deleted Chapter 2 §5 says

Recovered from `d14dcdd:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md` (743 lines; the surviving
in-tree file is a 34-line fragment).

- **§5.1 "The Real-World Problem"** (`:264-274`) — the gap model, ending: *"The sieves must test
  multiple **skip hypotheses**."*
- **§5.2 Constant Skip Mode** (`:276-294`) — table `skip=N` → *"Every (N+1)th output published"*.
  This is the only place the *units* of a skip value are stated unambiguously.
- **§5.3 Variable Skip Mode (Hybrid)** (`:296-311`) — *"Different gaps per draw — handles
  irregular sampling"*, example pattern `[0, 1, 0, 2, 1, 0, 3, ...]`, then the five strategies
  by name.
- **§5.4 Survivor Identity** (`:313-321`) — *"**A survivor is a (seed, skip_hypothesis) pair** —
  not just a seed."*

§5.4 is the closest the project comes to Michael's *"find the windows where a coherent skip
structure appears"* framing: it makes the skip hypothesis part of the survivor's identity rather
than a search nuisance. **But Chapter 2 §5 never mentions `skip_min` or `skip_max` at all** — it
does not connect the range to the variable mode. It is the natural home for that paragraph (as
`TEAM_ALPHA_CHAPTER_2_RECOVERY_SUBMISSION.md:85` argues) and the paragraph is not there.

**On Michael's stated intent specifically:**

| element of the stated intent | documented? |
|---|---|
| not estimating a single true gap | **YES** — `instructions.txt:1247` *"Array of detected skip values (not single `best_skip`)"*; Chapter 2 §5.3 |
| survivors carry skip structure as identity | **YES** — Chapter 2 §5.4 |
| skip-shape statistics exist so ML has something to learn | **YES** — `PROPOSAL_ML v2.0` §2.2 + `instructions.txt` `pattern_stats` |
| "find the *windows* where a coherent skip structure appears" — the fingerprint | **NOT FOUND** as a stated purpose. Nothing in the tree frames variable skip as a *window-selection* or coherence-detection mechanism. The closest is the strategy glosses (*"Best for: Tight, consistent patterns"* vs *"Loose, variable patterns"*, `instructions.txt:1186-1210`), which describe what each preset catches, not why coherence is the object |

---

## 3. The mechanics the design decision depends on

All four claims below are readings of the live tree at HEAD `2042a18`.

### 3.1 How are `skip_sequences` generated from `expected_skip = 5`? — **Neither enumerated, sampled, nor fixed. `skip_sequences` is an OUTPUT, not an input.**

This corrects the premise in the brief. No pattern is ever *generated*. `skip_sequences` is a
write-only device buffer the kernel fills with the pattern it **discovered**:

```c
prng_registry.py:1071    skip_sequences[pos * k + i] = best_skip_seq[i];
```

`expected_skip = 5` is the **initial condition of a greedy per-draw adaptive local search**, not
a pattern generator. Forward hybrid, `java_lcg_hybrid_multi_strategy_sieve`
(`prng_registry.py:1007-1077`), per draw:

```c
:1027   int expected_skip = 5;                                                    // once per strategy
:1033   int search_min = (expected_skip > skip_tolerance) ? (expected_skip - skip_tolerance) : 0;
:1034   int search_max = expected_skip + skip_tolerance;
:1035   for (int test_skip = search_min; test_skip <= search_max; test_skip++) {
            … three-lane CRT test …
:1046           actual_skip   = test_skip;
:1047           expected_skip = test_skip;      // ← window re-centres on the hit; full adaptation
:1049           found = true; break;
        }
:1052   if (draw_idx < 2048) current_skip_seq[draw_idx] = actual_skip;
```

So the window **drifts**: it re-centres on each hit and walks with the data. First match wins
(`break`), so it is greedy, not exhaustive.

**Two properties of the recorded pattern worth flagging before any wire-in:**

1. **Misses record a fabricated value.** `actual_skip` is initialised to `expected_skip`
   (`:1030`) and only overwritten on a hit. On a miss, `current_skip_seq[draw_idx]` stores the
   *expected* skip — a value that matched nothing. The persisted pattern therefore mixes
   observed skips with carried-forward guesses, with no flag distinguishing them. Any
   coherence statistic computed over `skip_sequences` would be measuring that contamination too.
2. **The semantics of the `5` are documented — in the ancestor file only.**
   `prng_registry_pre_registry.py:696` (tracked) reads:
   ```c
   int expected_skip = 5;  // Initial guess
   ```
   The comment says plainly it is a *guess*, i.e. a seed value for adaptation, not an assertion
   that the true skip is 5. **All 14 occurrences in the current `prng_registry.py`
   (`:805, :885, :1027, :1159, :1298, :1451, :1605, :1762, :1899, :2047, :2119, :2212, :2303,
   :2622`) carry no comment** — the gloss was lost in the migration to the registry. That lost
   two-word comment is a large part of why the `5` has read as a hardcoded constant rather than
   a tunable initial condition.

**Reverse hybrids use a different model entirely.** `java_lcg_hybrid_reverse_sieve`
(`prng_registry.py:3172-3241`) has **no `expected_skip`**: it searches `[0, skip_tolerance]`
fresh at every draw, non-adaptively (`:3200`), and writes `skip_seq[i] = 0` on a miss
(`:3222`). It also **returns on the first strategy that survives** (`:3238`) rather than taking
the best across strategies. So forward and reverse passes of the same trial do not share a skip
model, a miss convention, or a strategy-selection rule.

### 3.2 What does `strategy_tolerances` control? — **Tolerance during matching. Not deviation during generation.**

`strategy_tolerances` is the **half-width of the per-draw search window**, and nothing else.

- Source of truth: `hybrid_strategy.py:20` — `skip_tolerance: int  # Search window around
  expected skip (±tolerance)`. The in-code comment states it explicitly.
- Values `{5, 20, 5, 10, 50}` across the five `STRATEGY_PRESETS` (`hybrid_strategy.py:35-73`),
  paired with `max_consecutive_misses` `{3, 10, 5, 7, 20}`.
- Reaches the kernel as a device array via `coordinator.py:2307-2320`,
  `sieve_gpu_worker.py:239-252`, `miner/range_miner_worker.py:725-733`, `:833-841`.
- Applied at `prng_registry.py:1033-1034` (forward, **relative** to the running `expected_skip`)
  and `:3200` (reverse, **absolute** `[0, tol]`).
- It is a **fixed 5-point sweep, not an Optuna dimension** — no `suggest_*` for `skip_tolerance`
  exists anywhere in `window_optimizer.py`, `window_optimizer_bayesian.py` or
  `window_optimizer_integration_final.py`.

**Three further `StrategyConfig` fields are transported but never reach a kernel** — the same
dead-dimension class as D-1/D-2. `coordinator.py:2312-2314` and `:2471-2473` serialise
`enable_reseed_search`, `skip_learning_rate` and `breakpoint_threshold` into the worker payload,
but the hybrid kernel signature (`prng_registry.py:1010-1011`, `:3175-3176`) accepts only
`strategy_max_misses` and `strategy_tolerances`. Most pointedly:

```
hybrid_strategy.py:22   skip_learning_rate: float  # How fast to adapt expected skip (0.0-1.0)
```

is configured at `0.2`–`0.7` across the presets, while the kernel hard-assigns
`expected_skip = test_skip` (`:1047`) — an effective learning rate of **1.0** in every strategy.
A documented adaptation-rate control exists, is populated, is shipped to the worker, and is
overridden by the kernel. Whatever is decided about `skip_min`/`skip_max`, this one is a
genuine and separate instance of §2.7's recurring defect and is **not** currently recorded in
the skill.

### 3.3 Does the hybrid kernel score pattern coherence? — **No. Match rate only. Plainly.**

There is no coherence term anywhere in either hybrid kernel. The complete scoring path is:

```c
prng_registry.py:1057   float match_rate = (float)matches / k;         // count of hits / window
:1058                   if (match_rate > best_match_rate) { … }        // strategy selection
:1067                   if (best_match_rate >= threshold) { … }        // survival
```

The skip sequence is **recorded but never scored.** It does not enter `match_rate`, strategy
selection, or the survival test. Two seeds with identical match counts — one matching on
`[5,5,5,5,5]`, one on `[0,47,3,51,9]` — receive identical scores and are indistinguishable to
every downstream consumer.

The only coherence-adjacent mechanism is `max_consecutive_misses`, which **truncates the draw
loop early** (`break`, `:1055`; reverse sets `failed = true`, `:3227`). Because `match_rate`
divides by the full `k`, an early break lowers the score. That is a *pruning heuristic* that
indirectly penalises long incoherent stretches — it is not a coherence score, it does not
distinguish tight patterns from scattered ones of equal length, and it cannot rank survivors by
skip structure.

### 3.4 Why the three skip-shape features are dead — the producer exists, the host discards it

The kernel *does* produce exactly the data `skip_mean` / `skip_std` / `skip_entropy` need. It is
dropped one layer above:

- `extract_survivor_records()` (`window_optimizer_integration_final.py:121-160`) reduces every
  survivor to `{'seed', 'match_rate'}` (`:147`, `:158`). Its own docstring names `match_rate` as
  *"the primary per-seed quality signal for downstream ML"* — `skip_sequences` and
  `strategy_ids` are not read at all.
- `CANONICAL_RECORD_FIELDS` (`utils/canonical_records.py:115-124`) carries `skip_min`,
  `skip_max`, `skip_range`, `skip_mode` — and neither `skip_sequences` nor `strategy_ids`.
- The 22-array NPZ contract is frozen with the same omission.

**So the system computes the per-draw skip structure on the GPU, ships it to the host, throws it
away, and then persists `skip_min`/`skip_max` — which had no causal role in the pass — into the
frozen contract and feeds them to the ML layer as features.** The three dead placeholders are
downstream casualties of that single discard, not features that were never specified. This is
the mechanical link between §0.2's documented intent and §2.2's "5 dead placeholders with no
producer," and it is the reason Michael's reading of those features is consistent with the
kernel's actual output.

---

## 4. What this changes for the decision

Stated as findings, not as a proposal. No option is recommended here.

1. **The "unspecified semantics" premise of `HYBRID_SKIP_BOUND_AUDIT.md` §4.3 does not hold.**
   A hybrid-scoped meaning is stated in two documents (§0.1). Any argument for Option B
   (semantic demotion / removal) that rests on *"the semantics were never specified, so wiring
   them in would be inventing semantics"* now needs re-argument on different grounds. The
   audit's other four arguments (§4.4 — the axis is occupied by `skip_tolerance`, the clamp
   binds in the wrong direction, `expected_skip = 5` sits below every recorded `skip_min`,
   `strategy_ids` becomes uninterpretable) are untouched by this finding and remain on the table.
2. **The documented reading is element-wise, which is a different object from `skip_tolerance`.**
   `[skip_min, skip_max]` as *"minimum/maximum skip value in pattern"* is an **absolute bound on
   the values the discovered sequence may take**. `skip_tolerance` is a **relative bound on how
   far consecutive skips may move apart**. These constrain different properties — a level bound
   versus a step bound — and a pattern can satisfy either while violating the other. The audit's
   argument (1) that "the axis is already occupied" holds only if both are read as *"how wide a
   window to search."* Under the documented element-wise reading they are not the same axis.
   Whether that makes a wire-in *desirable* is a separate question from whether it is
   *well-defined*; this finding bears only on the latter.
3. **The `= 5` is documented as a guess, not a constant** (`prng_registry_pre_registry.py:696`).
   The audit's argument (2) — that any clamp "must also invent a new initial condition, a choice
   no spec in the tree authorizes" — is weakened: an initial *guess* is by its own comment a
   value intended to be revisable. It does not tell you what to revise it *to*.
4. **The output-statistic reading has its own live wiring problem.** If `skip_min`/`skip_max`
   are to mean *"minimum/maximum gap that worked"* (`PROPOSAL_ML v2.0` §2.2,
   `feature_registry.json:336/345`), they are computable today from `skip_sequences` — the data
   exists on the GPU and is discarded at `window_optimizer_integration_final.py:147`. That
   reading requires no kernel change at all, only that the host stop discarding the sequence.
   It would also revive `skip_mean` / `skip_std` / `skip_entropy` from the same source.
5. **Two registries in the tree disagree** (§0.3). Whatever is decided, one of
   `config_manifests/feature_registry.json` and `config_manifests/parameter_registry.json` is
   currently wrong and should be corrected as part of the same change.
6. **`skip_learning_rate` is a fifth dead dimension** (§3.2), independent of this decision, not
   currently recorded in `tfm-project-facts` §2.7.

---

## 5. Verification-integrity controls (VIR-1…6)

- **execution proof:** every quotation in §0-§3 is reproduced with `file:line` from a read
  executed this session on VM 101 at HEAD `2042a18`; line numbers were obtained by
  `/bin/grep -n` or `sed -n`, not recalled.
- **clean control:** the constant-skip path is the built-in negative control for §3.1-3.3 — it
  runs through the same files and the same `BuildContext`, and *does* deliver `skip_min`/
  `skip_max` to the kernel (`prng_registry.py:963` signature, `:972` loop). The reading method
  therefore distinguishes "declared and consumed" from "declared and dropped" rather than
  reporting everything dropped.
- **fault-injection control:** the search method was validated against a known-present target
  before the absence claims were made — `git show d14dcdd:docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md`
  recovered a 743-line deleted document that `grep` over the working tree cannot see (the
  in-tree file is a 34-line fragment), confirming the history-recovery leg is live and not
  vacuous.
- **completion sentinel:** all §1 table rows executed to completion; no search was truncated or
  timed out except one full-tree `expected_skip` grep, which was re-run to completion in the
  background and its output read (`bj7ic5d1g`).
- **unavailable-observer behavior:** the rigs were not contacted; this is reported as
  UNAVAILABLE in §1, not as clean. No claim below is made about deployed rig source.
- **audit claim scope:** the VM 101 tree at HEAD `2042a18` (tracked + untracked), its full git
  history, and `/home/michael` outside the repo. **Repo-and-host scoped, NOT cluster-scoped.**
- **searched surfaces:** §1 table.
- **unavailable surfaces:** §1 "Surfaces NOT searched" — rigs, kernel execution, the CA draw
  procedures PDF, the public clone, Optuna DBs / scoring chunks / NPZ artifacts, out-of-band
  discussion.
- **termination:** **PASS** (VIR-3) — the falsifiable question is answered affirmatively with
  quoted primary evidence, and the residual not-found element (the "fingerprint / window
  selection" framing, §2) is reported as NOT FOUND with its scope attached.
