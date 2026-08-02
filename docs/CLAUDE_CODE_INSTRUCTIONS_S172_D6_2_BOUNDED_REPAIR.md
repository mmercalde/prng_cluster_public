# CLAUDE_CODE_INSTRUCTIONS_S172_D6_2_BOUNDED_REPAIR.md — REV1

**S172 — D6.2 bounded repair. Beta returned `f7583bc` uncertified; two execution-path defects the
29 gates do not exercise.**

**Base: `f7583bc`** (repair on top of it; do not revert). Claude Code on **VM101** as `michael`,
venv `~/venvs/torch`. Implement and iterate; do **NOT** commit, push, or run WATCHER. STOP at the
gate.

**Beta ratified the architecture. No REV6.** Ratified and not to be reopened: the record-ordinal
continuation as the replay-key authority · the four ported NR suites · `seed` as member A's field
name · `member_role` · the 24-field schema, CSR encoding, digest separation, asymmetric A/B
contract, nine-row recovery, sequence repair, pre-clear walls, canonical reconciliation, array
parity, run-ID confinement, context binding, durable resume provenance, clear-after-installed-pair
ordering.

**This brief changes only what the two blockers require.**

---

## 0. Why Beta returned it, and why the gates missed it

**Both defects live on the execution path. Neither is reachable from the 29 gates as written.**

Blocker 1 passed because `G-TRIAL-NAMESPACE` exercises the guard with **fabricated values —
trial 6 against floor 5** — instead of constructing the real relationship between completed Optuna
trials and persisted record ordinals. **A gate built from invented numbers cannot discover an
off-by-one between two real counters.** That is the VIR-2 vacuous-detector class.

Blocker 2 passed because no gate runs the `n_parallel > 1` path at all.

---

## 1. BLOCKER 1 — the guard rejects the normal resume

### 1.1 The defect, verified at source

`window_optimizer_bayesian.py:522-523`:

```python
if (_resume_trial_floor is not None
        and int(trial.number) <= int(_resume_trial_floor)):
    raise RuntimeError(...)
```

`_resume_trial_floor` is the recovered **record-ordinal** maximum — **1-based**
(`trial_counter['count']`, `window_optimizer_integration_final.py:2646` → `:2664`).
`trial.number` is Optuna's **0-based** study number.

**For a normal run with `k` completed trials:**

| quantity | value |
|---|---|
| Optuna numbers already used | `0 … k−1` |
| persisted record ordinals | `1 … k` |
| next legitimate Optuna number | `k` |
| recovered record floor | `k` |
| guard evaluates | `k <= k` → **True** → **raises** |

**Every normal resume is rejected.** The feature does not work.

### 1.2 What is already correct — do not touch it

`window_optimizer_integration_final.py:2623`:

```python
trial_counter = {'count': int(_d6_2_resume_floor or 0)}
```

The next `+= 1` (`:2646`) yields `floor + 1`. **This is the ratified repair and it is correct.**

### 1.3 Binding correction

1. **Rename the concept to `resume_record_ordinal_floor`** — everywhere: the attribute seam, the
   local at `:2593`, the parameter names, the comments. **The old name asserts a relationship to
   Optuna trial numbers that does not exist**, and that false name is what produced the defect.
2. **Use it ONLY to initialize the persisted record counter** (`:2623`).
3. **Do NOT compare Optuna `trial.number` against that floor.** Anywhere.
4. **Retire both Optuna-number guards** (`window_optimizer_bayesian.py`: the pre-flight nonterminal
   scan and the in-objective check at `:504-532`) **and the queued-trial mutant based on them.**
   Remove `strategy._resume_trial_floor` forwarding at
   `window_optimizer_integration_final.py:2702` if nothing else consumes it — **check before
   deleting.**
5. **Retain the four-row combination policy**, with one strengthening:
   - **checkpoint without study resume still rejects;**
   - **checkpoint + study resume must PROVE an existing study was actually loaded**, rather than
     silently falling back to a fresh one. `create_study(..., load_if_exists=_resume)`
     (`window_optimizer_bayesian.py:696-702`) **creates a fresh study when the name does not
     exist** — a resume that silently becomes a fresh study restarts the record ordinal against a
     recovered checkpoint, which is the collision this whole area exists to prevent. **Verify the
     study existed; reject before the first objective executes if it did not.**

### 1.4 Required gates — replace the vacuous one

**`G-RESUME-INTEGRATED` (new, replaces `G-TRIAL-NAMESPACE`):** construct the **real** relationship —
completed Optuna trials `0…k−1`, recovered record ordinals `1…k` — and require:

- **Optuna trial `k` EXECUTES** (this is the case the old guard rejected);
- **its resulting record receives ordinal `k+1`;**
- **no replay-key collision occurs** on `(seed, trial_number, skip_mode)`.

**`G-MISSING-STUDY` (new):** checkpoint + study resume where the named study **does not exist** →
**rejected before the first objective executes.** Prove no objective ran.

**Do not fabricate the numbers in either gate.** Derive `k` from an actual run.

---

## 2. BLOCKER 2 — NP2 executes before D6.2 validation exists

### 2.1 The defect, verified at source

`_prepare_checkpoint_run_context` is called at
`window_optimizer_integration_final.py:2593`. The `n_parallel > 1` block runs at **`:1968-2550`** —
roughly **600 lines earlier** — and includes:

- `[NP2-KILL]` **SSH to every AMD rig** (`:2335-2351`);
- port cleanup (`:2362-2367`);
- the partition fork (`:2387`);
- full trial execution and survivor merge.

**Consequences, all of which follow:** combination rejection happens **after** optimization ·
resume identity is not validated **before** candidate production · forked workers have **no
installed D6.2 context** · their flush attempts **cannot clear memory**, so **the S166 OOM
protection is not real on NP2** · a checkpoint resume can **mutate the study and drive fleet work
before being rejected**.

### 2.2 Binding correction — scope D6.2 to `n_parallel == 1`

**Reject `resume_checkpoint` when `n_parallel > 1`, BEFORE any of:** study creation · worker launch
· **SSH or any fleet action** · candidate admission.

**Place the rejection above `:1968`.** It must precede the `[NP2-KILL]` SSH, not merely the fork.

**State truthfully, in the module header and the report:** D6.2 checkpoint recovery and OOM
protection are **certified only for the default single-Optuna-trial path**. **That path still
distributes each sieve trial across the full GPU cluster** — the scope limit is on Optuna
parallelism, not on fleet use. Do not overstate it in either direction.

**Make no NP2 claim.** Do not claim NP2 accumulator clearing or resume support. **Concurrent
partition writers cannot safely share the present checkpoint pair**; that needs a separate
transaction design.

**`G-NP2-SCOPE` (new, CPU-only):** `resume_checkpoint` + `n_parallel > 1` → rejected, with **zero
worker or process starts** before the rejection. **Prove zero starts** — count processes, or assert
the SSH/fork call sites were never reached. **No rig, no GPU, no network.**

---

## 3. Textual correction

`utils/checkpoint_d6_2.py:20-30` still states Member A carries **`seeds`**. The implementation and
the ratified record domain use **`seed`**. **Correct the stale statement.**

---

## 4. Non-regression

Re-run and report: **D6.2 (with the replaced and new gates) · the ported D6.1 suite · D6 3.A ·
D6-threshold · D3.5 (60/60) · D3.25 (13/13) · D3 · D3.0 · D0 · D4 · import gate · Phase 1 · Phase 2
· Phase 3 · Phase 4 · D1.1 · D1.0 · D2 · D5 · 6-P0.5 · Ch1 P0 · `test_prng_encoding`.**

**No Wall A/B rerun** (Beta).

Expect Phase-4 Gate 22 to red on any uncommitted new file and propagate to D5's `NR` arm. **Not a
regression, not a reason to edit Gate 22.**

All commands on **VM101**, `source ~/venvs/torch/bin/activate` first. Long suites:
`python3 -u <suite> | tee /tmp/<name>.log` — **never pipe to `tail`.**

## 5. Scope — do NOT touch

Anything Beta ratified (§0). `_l2_sort_key`, `_select_l2_winners`, `CANONICAL_ARRAY_CONTRACT`,
`utils/prng_encoding`, `canonical_map_hash`, the three finalizer validators,
`_RusageChildrenSampler`, Phase-4 Gate 22. The nine-row recovery matrix. The digest preimages.
**Do not revert `f7583bc`; repair on top of it.**

## 6. Report

`docs/S172_D6_2_BOUNDED_REPAIR_REPORT.md`: the rename applied and **every** site it touched · proof
**no** comparison of `trial.number` against the floor survives (AST + grep) · the retired guards and
mutant, named · **`G-RESUME-INTEGRATED` with the real `k` derived from an actual run**, showing
trial `k` executing and its record taking ordinal `k+1` · `G-MISSING-STUDY` proving rejection before
the first objective · **`G-NP2-SCOPE` proving zero process starts**, and the exact line the
rejection sits above · the corrected header text · gate/mutant counts before and after · the full
§4 table. Then STOP for Team Alpha review.

---

## Verification-integrity controls (VIR-1…6)

- **execution proof:** `G-RESUME-INTEGRATED` reports the actual `k`, the Optuna numbers used, and
  the record ordinals produced — **not a boolean.** `G-NP2-SCOPE` reports the process count.
- **clean control:** a normal fresh run (no checkpoint, no study resume) passes unchanged; a normal
  `n_parallel == 1` resume **now succeeds**, which it did not at `f7583bc`.
- **fault-injection control:** reinstate the `trial.number <= floor` comparison → **`G-RESUME-INTEGRATED`
  must red** · move the NP2 rejection below `:1968` → **`G-NP2-SCOPE` must red** · make the
  missing-study case fall back to a fresh study → **`G-MISSING-STUDY` must red.** Four-part kill
  rule on each.
- **completion sentinel:** `PASS | FAIL | UNAVAILABLE | INCOMPLETE`; only `PASS` accepts.
- **unavailable-observer behavior:** every gate here is **CPU-only with no fleet dependency**; any
  `UNAVAILABLE` arm is a finding, not an excuse.
- **audit claim scope:** repo-scoped at `f7583bc` plus this repair.
- **searched surfaces:** the tracked repo at `f7583bc`.
- **unavailable surfaces:** host state on VM101 and the rigs; deployed uncommitted files.
