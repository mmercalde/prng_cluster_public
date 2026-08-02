# CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_REV5_BINDING_ADDENDUM.md

**BINDING. Read `docs/CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_CHECKPOINT_RECONCILIATION.md` (REV5)
first, then this. Where the two differ, THIS DOCUMENT WINS.**

Beta ruling on REV5: **APPROVED, implementation authorized, no REV6.** These four items are
normative. Everything else in REV5 stands unchanged.

**Base:** HEAD `b0eb33a`. Claude Code on **VM101** as `michael`, venv `~/venvs/torch`. Implement
and iterate; do **NOT** commit, push, or run WATCHER. STOP at the gate.

---

## 1. Identity fields ARE included in `member_content_digest` — overrides REV5 §3.2

REV5 said Alpha should *"decide and state"* this. **Beta required the brief to decide it. It is
decided here.**

**`member_content_digest` covers every persisted identity field AND every payload field, EXCEPT
`member_content_digest` itself.** That inclusion covers `canonical_state_digest`.

- **Compute the member digest LAST**, after every other field is fixed.
- **Use a fixed field order — never dictionary or NPZ iteration order.**

### `canonical_state_digest` — the physical order is fixed

Rows are **globally seed-sorted before these arrays are constructed.** Then, in exactly this order:

```
fields 1-9          (seed, forward_match_rate, reverse_match_rate, score,
                     window_size, offset, skip_min, skip_max, skip_range)
sessions_values
sessions_offsets
trial_number
skip_mode
prng_type
fields 15-24        (forward_count … intersection_weight)
```

**Derived `prng_base` is NOT separately hashed.** It is reconstructed from `prng_type` +
`skip_mode` and adds no information.

Per array: **domain separator · field name · exact dtype · exact shape · contiguous bytes.**

### Member payloads, confirmed

- **Member A** — the previously approved compatibility payload: **`seeds` and `score`, plus its
  complete identity block.** Nothing more.
- **Member B** — the complete reconstructible 24-field state.

---

## 2. The ninth recovery outcome — overrides REV5 §5

REV5 dropped one row. **The recovery suite and report carry NINE explicit outcomes, not eight.**

**Restored row:**

| state | required behaviour |
|---|---|
| **B valid and newer; A valid but older; all invariant identities agree** | **recover B, install and validate a repaired pair, initialize the next sequence above B** |

**This state is distinct** from an absent/corrupt A and from a consistent same-transaction pair.
**It need not be reachable from the ordinary A-first replacement crash to remain a valid mixed-pair
recovery case** — do not argue it away as unreachable and skip it.

The full nine:

| # | state | behaviour |
|---|---|---|
| 1 | A missing or unreadable | validate B against caller-supplied run id + context; recover B |
| 2 | A readable, identity matches, fails its `member_content_digest` | recover B, repair the pair |
| 3 | A structurally valid but conflicts with B or requested context | **fail closed** |
| 4 | A valid newer uncommitted marker, invariants match | discard A, recover B, sequence above A |
| 5 | **B valid and newer; A valid but older; invariants agree** | **recover B, repair pair, sequence above B** |
| 6 | consistent A/B transaction | recover B |
| 7 | **B missing or invalid** | **fail closed regardless of A** |
| 8 | any context / schema / encoding disagreement | **fail closed** |
| 9 | equal sequence, different `checkpoint_id` | **fail closed** |

**Fail-closed means: do NOT clear in-memory state.**

---

## 3. The run ID is an OPAQUE SINGLE COMPONENT — strengthens REV5 §4.1

Path confinement alone is insufficient: **`foo/bar` would still behave like a handle**, which is
the two-API ambiguity REV5 was meant to close.

**Require a single-component run id.** Reject: `/`, any alternate separator, an empty component,
`.`, and `..`. **A conservative alphanumeric / underscore / dot / hyphen grammar is appropriate** —
validate the whole string against it, and reject bare `.` and `..` explicitly even though the
grammar admits the characters.

**Realpath and symlink-escape checks remain mandatory** — the grammar is an additional wall, not a
replacement for them.

---

## 4. Check the ACTUAL Optuna trial number — strengthens REV5 §4.4

REV5's *"prove the next trial number exceeds every recovered `trial_number`"* must cover **queued
trials**, not merely `max(existing) + 1`.

**On checkpoint-plus-study continuation:**

1. **Reject any nonterminal study trial capable of resumption at or below the recovered maximum.**
2. **After obtaining each Optuna trial, verify its actual `trial.number` exceeds the recovered
   maximum.**
3. **Perform that check BEFORE objective execution, dispatch, or candidate admission.**
4. **Never rewrite or offset the number.**

### Where this goes — the seam, located

Trials are obtained through **`study.optimize(optuna_objective, n_trials=…)`**
(`window_optimizer_bayesian.py:757`), **not** ask/tell. So `trial.number` is first readable
**inside the objective callback**, and check 2 belongs at **the very top of `optuna_objective`**,
before any dispatch or accumulation.

**Check 1 is a pre-flight over the loaded study**, before `study.optimize` is entered. Scan
nonterminal trials — `WAITING` and `RUNNING`.

**⚠ The queued case is real here, not theoretical.** `study.enqueue_trial(_ws_params)`
(`window_optimizer_bayesian.py:725`) is the **S166 warm-start path**. A resumed study can therefore
carry enqueued trials whose numbers were allocated **before** the recovered maximum. That is
precisely the case check 1 exists for. **Do not assume a loaded study's next number is
`max(existing) + 1`.**

`create_study(..., load_if_exists=_resume)` is at `:696-702`.

---

## 5. Gates added or amended by this addendum

- **G-DIGEST-PREIMAGE** (amended): identity fields **are** included; `member_content_digest`
  excludes only itself; **`canonical_state_digest` IS inside the member digest**; member digest
  computed **last**; **fixed field order asserted, not dict/NPZ order** (prove a reordered dict
  yields the identical digest).
- **G-STATE-ORDER-PHYSICAL** (new): the §1 physical order is what the code emits;
  **`prng_base` absent from the preimage**; rows globally seed-sorted first.
- **G-RECOVERY-MATRIX** (amended): **nine rows**, each its own case. Row 5 explicitly present.
- **G-RUNID-GRAMMAR** (new): reject `/`, alternate separators, empty component, `.`, `..`, and
  anything outside the grammar — **plus** the existing realpath / symlink-escape rejections.
- **G-TRIAL-NAMESPACE** (amended): pre-flight rejects a nonterminal trial at or below the recovered
  maximum; the per-trial check fires **at the top of the objective**, before dispatch;
  **an enqueued warm-start trial below the recovered maximum is rejected**; numbers are never
  rewritten or offset.

**Mutants added:** exclude an identity field from the member digest · include
`member_content_digest` in its own preimage · emit the state digest in dict order rather than the
§1 fixed order · hash `prng_base` into the state digest · delete recovery row 5 · accept a run id
containing `/` · allow an enqueued trial at or below the recovered maximum to execute.

---

## 6. Authorization and stop point

Alpha is authorized to implement REV5 **with this ruling treated as normative**, run the prescribed
baseline / D6.2 / mutant / non-regression gates, and **stop for Team Alpha review without
committing, pushing, or running WATCHER.**

**No Wall A/B rerun is required.**

Baseline before any edit: **D3.25 (13/13), D3.5 (60/60), D6.1, Phase 3 (17/17), Phase 4 (63/63),
D5 (25/25)**, plus D1.1 · D1.0 · D0 · D2 · D3.0 · D3 · D4 · D6 3.A · D6-threshold.

**Note on D5 and Phase-4 Gate 22:** Gate 22 builds `changed_py` from `git status --porcelain`, so
**any uncommitted new file reds it and propagates to D5's `NR` arm.** Expect that while D6.2's new
test file is untracked. **It is not a regression, and it is not a reason to edit Gate 22.**

Long suites: use `python3 -u … | tee /tmp/<name>.log`, or `nohup`. **A suite piped to `tail` prints
nothing until it finishes and looks hung.**

Report per REV5 §12, **plus**: the two digest preimages exactly as built (field order shown), the
order-permutation and dict-order results, **all nine recovery rows**, the run-id grammar
rejections, and **the two trial-namespace checks with the enqueued-trial case exercised.**
