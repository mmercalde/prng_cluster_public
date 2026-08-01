# TEAM ALPHA → TEAM BETA — skip semantics: the documentation exists, and it changes the decision

**Re:** `docs/SKIP_SEMANTICS_SEARCH_v1.md` (committed `46a294b`). Read-only search; nothing
changed.

**Michael's decision, for which Alpha requests Beta's ruling on adequacy:**

> **Stop discarding `skip_sequences`. Revive the three dead skip-shape features.**

This is the **output-statistic** reading. It requires **no kernel change**, and it is a
different question from whether `skip_min`/`skip_max` should also bound the hybrid search as
inputs — which remains open and is **not** proposed here.

---

## 1. The premise of the earlier audit was false

`HYBRID_SKIP_BOUND_AUDIT.md:318` recorded the hybrid semantics as **"unspecified"**, and its
Option B (semantic demotion) rested partly on *"wiring them in would be inventing semantics."*

**They are specified, in two committed documents.** Alpha's own "nobody has written it down" was
the **fourth falsified absence claim** of the session — and the sharpest, because the audit's
VIR-6 declared a full-tree `grep` for `skip_min` that **reached the exact line and did not read
it**. That is a reading failure, not a scope failure, and widening search surfaces does not fix
it. Recorded as an Alpha process defect and added to the skill (§1, fifth corollary).

**The audit's other four arguments are untouched and remain on the table** for the *input*
question: the axis may be occupied by `skip_tolerance`, the clamp binds in the wrong direction,
`expected_skip = 5` sits below every recorded `skip_min`, and `strategy_ids` becomes
uninterpretable.

## 2. Two documented readings — different pipeline stages, not a contradiction

| stage | reading | source |
|---|---|---|
| **input** (Step 1 → 2) | *"Minimum/Maximum skip value **in pattern**"* — an **element-wise bound** on the discovered sequence. Documented hybrid default `[0,16]` | `docs/instructions.txt:1182-1183`; verbatim at `Cluster_operating_manual.txt:948-949`; present in an older revision, so the wording **predates** the current file |
| **output** (Step 2 → 3) | *"Minimum/Maximum gap that **worked**"*; *"skip_range = hypothesis flexibility"*; **"Tight skip range = stronger hypothesis"** | `PROPOSAL_ML_Architecture_Remediation_v2_0.md:150-158`; `config_manifests/feature_registry.json:336,345` |

The same names doing different jobs at different stages. **Two registries in the tree
disagree** — `feature_registry.json` says *"found during sieve analysis"* (output),
`parameter_registry.json:160,166` says *"for sieve search"* (input). **One is wrong and should
be corrected in whatever change settles this.**

## 3. Why the output reading is the right first move

**It costs one host-side change.** `skip_sequences` already exists — computed on the GPU, per
survivor. It is **discarded** at `window_optimizer_integration_final.py:147`, where
`extract_survivor_records` reduces every survivor to `{seed, match_rate}`.

**It revives three of the five dead placeholder features.** `skip_mean`, `skip_std` and
`skip_entropy` have no producer *today* — but their producer exists on the GPU. The October 2025
output spec (`docs/instructions.txt:1230-1245`) declares, per survivor:

```
skip_pattern:  [5,5,3,7,...]
pattern_stats: {mean_skip, variance, std_dev}
```

**the literal ancestor of the three dead features.** They are not aspirational placeholders;
they are a shipped design whose data is thrown away one function before it would be recorded.

**And it matches the stated purpose.** `PROPOSAL_ML v2.0` puts `skip_min`/`skip_max` in **one
table** with `skip_mean`, `skip_std`, `skip_entropy` under a single objective: characterise gap
*shape* so the models can rank on it. Michael's design intent — variable skip produces survivors
with **varied** skip structure so tree/NN models have something to learn from — is corroborated
in writing on three of four elements (`instructions.txt:1247`: *"Array of detected skip values
(not single best_skip)"*; survivor-as-(seed, skip_hypothesis) pair in the deleted Chapter 2
§5.4 at `d14dcdd`; and the pattern_stats spec above).

**It also makes the input question answerable.** Recording the observed skip distribution is
what tells you whether an element-wise bound is worth imposing, and where. Deciding the input
reading first would be constraining a distribution nobody has measured.

## 4. What Alpha is NOT proposing

- **Not** wiring `skip_min`/`skip_max` into the hybrid kernels. That is the input reading, it is
  a 22-kernel ABI change, and the audit's four remaining objections apply to it.
- **Not** retiring or demoting anything. Per §0.4 the input reading stays open; this submission
  narrows nothing.
- **Not** touching `expected_skip = 5`. Documented as `// Initial guess`
  (`prng_registry_pre_registry.py:696`) — revisable, but nothing in the tree says what to revise
  it *to*.
- **Not** adding coherence scoring. None exists today (only `match_rate`); whether the sieve
  should *score* pattern stability rather than merely record it is a separate design question
  Alpha is not raising here.

## 5. Mechanics Beta should have, because they correct a common misreading

- **`skip_sequences` is an output, not an input** (`prng_registry.py:1071`). **No pattern is
  generated.** `expected_skip = 5` seeds a **greedy per-draw adaptive search that re-centres on
  each hit** (`:1047`).
- **`strategy_tolerances` is the half-width of the per-draw *matching* window**
  (`hybrid_strategy.py:20`) — matching, not generation. It is a **step** bound (how far
  consecutive skips may move); `[skip_min, skip_max]` under the documented reading is a
  **level** bound (what values are permitted). A pattern can satisfy either while violating the
  other, so *"the axis is already occupied"* holds only if both are read as *"how wide a window
  to search."*
- **Misses write a fabricated `actual_skip`** into the recorded pattern — relevant to §3, since
  it means the recorded sequence needs a defined treatment of misses before statistics are
  derived from it. **Alpha flags this as the one real design question inside the output
  reading.**
- **`skip_learning_rate` is a fifth dead dimension** — configured 0.2–0.7, kernel hard-adapts at
  1.0. Independent of this decision; newly catalogued in skill §2.7.

## 6. Rulings requested

1. **Approve the output-statistic change**: stop discarding `skip_sequences`; record the pattern
   and derive `skip_mean` / `skip_std` / `skip_entropy` — and, if Beta agrees the naming is
   unambiguous, observed `skip_min` / `skip_max` as **features**.
2. **Rule on the misses question (§5)** — how a fabricated `actual_skip` on a miss should be
   treated before statistics are derived. Alpha has no mandate and will not choose.
3. **Correct the registry disagreement** — one of `feature_registry.json` /
   `parameter_registry.json` is wrong either way.
4. **Naming.** If both readings eventually coexist they need distinguishable names
   (e.g. `skip_search_min` vs `observed_skip_min`). Beta may prefer to settle this now rather
   than after two things share one name in the NPZ contract.
5. **Sequencing.** Alpha assumes this lands **after** bounded Phase 6, not before — it changes
   what Step 3 records, and Beta has been consistent that behavioural changes should not
   accumulate ahead of a certification. **Confirm or redirect.**

## 7. Scope note

The **input** reading remains open and undecided. Alpha does not treat this submission as
settling it, and per §0.4 will not propose removal of `skip_min`/`skip_max` from the search
space on the strength of it.

Michael's *fingerprint / window-selection* framing — that variable skip exists to find the
windows where coherent skip structure surfaces — is **corroborated on three of four elements**
but the framing itself is **NOT FOUND** in any document. It belongs in the Chapter 2 restore
(§5.4 is recoverable at `d14dcdd`), and Alpha will carry it there rather than leave it in a
skill file and a chat log.
