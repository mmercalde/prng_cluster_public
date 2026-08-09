# CLAUDE CODE REPORT — S172 STAGING-CAPACITY AMENDMENT, REVISION 2

**Host:** VM101 (`zeus-ubuntu-vm`, `192.168.3.177`) · repo `~/distributed_prng_analysis`
**Base:** `c7058d8`, amendment + R1 + R2 **uncommitted in the working tree**
**Status:** all five R2 changes implemented (2 production, 3 gates). **Not committed, not pushed,
no pipeline/fleet launch, no port 5700 bind.** Gate 12 and the Phase-7 soak remain HELD.

**Authority:** Beta ruling *"S172 STAGING-CAPACITY R1"* (2026-08-08) — R1 accepted in substance.
Nothing on the closed list was reopened.

---

## 0. Base verification

| check | required | actual | |
|---|---|---|---|
| `git log --oneline -1` | `c7058d8` | `c7058d8` | ✅ |
| amendment + R1 diffstat | intact | 7 files, 2243 +/85 − — unchanged from the R1 submission | ✅ |
| `test_s172_staging_backpressure.py` | 48/48 | **48/48**, exit 0, sentinel PASS | ✅ |
| `test_s172_elapsed_roundtrip.py` | 6/6 | **6/6**, exit 0, sentinel PASS | ✅ |

No unexpected tracked drift. Untracked runtime residue present as expected.

---

## 1. PRODUCTION CHANGE 1 — the trial's assignable cohort is frozen at preflight

### What is frozen, and from what

`freeze_trial_cohort` (`miner/range_miner_coordinator.py:3491`) stores, per run:

```
run_id -> {(family_name, phase) -> {worker_id: capability_signature}}
```

built from **the same `eligible_by_stage` object the ceiling was derived over**. The preflight now
resolves the stage sets **once**, at `:3670`, and hands that one object to the derivation, the
persisted plan and the freeze — so the frozen cohort cannot describe a different population from
the one the persisted plan documents. Freezing happens **only on a successful preflight**, after
the plan is durable (see §2).

### How the freeze is enforced at every later eligibility calculation

`cohort_eligible(run_id, family, phase, worker)` (`:3520`) is the single predicate. It applies
`can_assign_variant` **first**, so the Phase-4 exact-variant contract is never weakened — the
freeze can only ever *remove* candidates. It returns True unconditionally when no cohort was frozen
(the bare-API/gate path), which keeps it a strict refinement of pre-R2 behaviour rather than a
second, divergent rule.

There are exactly three places a worker is selected for a stripe, and all three now go through it:

| site | `file:line` | how |
|---|---|---|
| initial assignment | `assign_stripes` `:2669` | `compatible = self.cohort_filter(run_id, family_name, phase, workers)` |
| retry-matrix reassignment | `_pick_other_worker` `:4890` | frozen-stage lookup inside the helper |
| the "no eligible worker supports this variant" guard | `serve_trial` `:6035` | `any(self.cohort_eligible(run_id, fam, ph, w) …)` |

The guard uses the *same* predicate as assignment deliberately: otherwise a stage whose only
candidates were post-freeze joiners would pass the guard and then strand every stripe `pending` —
exactly what the guard exists to prevent.

**⚠ Why the retry check lives inside `_pick_other_worker` rather than at its call site.** My first
implementation threaded `run_id`/`phase` through `_handle_stripe_failure_locked`. That **red
`G-MATRIX-DIFF-a`**, which holds `_on_staging_failed`, `handle_stripe_failure` and
`_handle_stripe_failure_locked` AST-identical to `4b1aad6` — the retry matrix and its surviving
callers are explicitly out of scope for this amendment. I reverted that and moved the restriction
into `_pick_other_worker` (which is **not** in the protected set), resolving the frozen stage from
`family_name`. `workflow_stages_for` emits a distinct concrete family per stage, so a family names
its stage unambiguously; if several frozen stages ever shared one, a worker frozen for any of them
is accepted — the union, the conservative direction for a restriction. The matrix plumbing is
byte-identical, and the gate proves it.

### What the capability signature compares on reconnect

`cohort_capability_signature` (`:494`) is a SHA-256 over canonical JSON of exactly three fields —
the only advertisements that can change the derived ceiling or a worker's stage membership:

| field | why it is in the signature |
|---|---|
| `backend` | selects which cap tier `applicable_seed_cap` reads (cuda→`nvidia`, rocm→`amd`) |
| `seed_caps` | the cap values the conservative bound maxes over |
| `supported_variants` | decides which stages the worker is eligible for |

`supported_variants = None` is preserved distinctly from `[]`: a `WorkerRecord` advertising nothing
is treated as eligible for everything by `can_assign_variant`, and collapsing that to an empty list
would silently change the contract on reconnect. Nothing else is compared, so an irrelevant
reconnect difference cannot evict a legitimate worker from its own trial.

**All five ruled behaviours:** (1) frozen at successful preflight ✅ (2) every later eligibility
calculation intersects with the frozen stage set ✅ (3) a new identity may register and serves a
*subsequent* trial ✅ (4) a frozen identity that reconnects re-enters only on a matching signature
✅ (5) losing frozen workers never enlarges the cohort — the frozen dict is never added to, and
`cohort_filter` can only remove ✅. **No re-preflight, no hypothetical-member margin.**

---

## 2. PRODUCTION CHANGE 2 — fail closed when the plan cannot be persisted

`_persist_preflight_plan` (`:3737`) now takes an explicit `fail_closed` flag, because the two cases
are **not symmetric**:

| case | behaviour | terminal classification |
|---|---|---|
| **A** — trial would be ADMITTED | raises `StagingPreflightProvenanceError` (`:1956`) | `coordinator_staging_preflight_provenance:` (`serve_trial:6275`) |
| **B** — trial is ALREADY sizing-refused | returns the error string; the caller attaches it as `[secondary: …]` | `coordinator_staging_retention_sizing:` — **unchanged** |

**Ordering matters and is deliberate.** On the admit path the persist call happens **before** the
ceiling is installed, before the cohort is frozen and before any `StripeAssign` (`:3716-3725`), so a
failure leaves the coordinator in exactly the state it was in before the preflight ran — there is
nothing to unwind. On the refusal path no cohort is frozen at all: a refused trial has no
assignable population.

`StagingPreflightProvenanceError` subclasses `StagingConfigurationError`, so like the sizing error
it is permanent and non-retryable — no Q3 retry, no worker charge. It is deliberately a *distinct*
type: the sizing refusal means "this trial cannot fit", this means "this trial cannot be audited".

Beta's framing, preserved verbatim in the design: *failure to write the audit record may not
override a safety refusal, but inability to create the mandatory audit record prevents a would-be
admission.*

---

## 3. GATE CORRECTION 3 — late-worker exclusion

`G-LATE-WORKER-EXCLUDED` (`gate_late_worker_excluded_from_frozen_cohort`). All six required steps
plus the closing invariant:

1. preflight with A and B (both CUDA);
2. the stage-specific execution set is persisted and frozen (asserted against both
   `get_preflight_plan` and `frozen_trial_cohort`);
3. **C registers after preflight with a materially tighter applicable cap** — and the gate proves
   the fixture is not cosmetic by asserting the bound *would* move if C were counted;
4. C receives **no** `StripeAssign` (`assign_stripes` claims only A/B), and `_pick_other_worker`
   refuses C on the retry path while still returning B;
5. C **is** admitted to a later trial (`runLW2`), and the earlier trial's cohort is untouched;
6. reconnect: A unchanged → readmitted; A on a different **backend** → refused; A advertising a
   different **variant set** → refused.

Plus: **no re-derivation occurred** — the in-memory detail, the effective ceiling and the persisted
`required_files` are all still the pre-C value.

**Fixture note worth recording.** My first attempt gave C "tighter caps" by advertising smaller
numbers. That is not constructible: `_validate_caps` (`:2513`) requires advertised caps to equal
the central config exactly and **quarantines** a worker that disagrees, so C was excluded for an
unrelated reason and the bound did not move (28 vs 28 — the gate caught its own inertness). The
tightness has to come from the **backend**, which is Beta's own example: CUDA population at
preflight (`nvidia` 5,000,000 → 14 sub-stripes/stripe), then a ROCm worker joins (`amd` 2,000,000 →
34/stripe).

---

## 4. GATE CORRECTION 4 — `G-MUT-STAGE-ELIGIBILITY` corrected

The R1 mutant reproduced **Beta's withdrawn hypothesis** — first stage's *resolved* population
copied across — and asserted an understatement. It now restores the **real** previous behaviour:

```python
def _all_connected(self_, workflow_stages, candidate_workers):
    pool = list(candidate_workers)              # all-connected, non-quarantined
    return {(str(f), int(p)): list(pool) for f, p in workflow_stages}
```

and asserts what is actually true of it. **Before/after under the asymmetric fixture** (A cuda /
constant-only, B rocm / hybrid-only, 2 stripes of 67,108,864):

| calculation | result |
|---|---|
| stage-resolved (current code) | **328** |
| all-connected (real previous behaviour) | **408** |

The mutant asserts the two are **detectably different** (the gate's purpose) and that the
all-connected value is **greater** — recorded as the observed fact that a superset cannot lower a
max-over-workers bound, **not** as a safety requirement. No manufactured safety failure. The gate
then confirms exact-variant stage semantics are preserved once the real resolver is restored
(`java_lcg → [hostA]`, `java_lcg_hybrid → [hostB]`).

---

## 5. GATE CORRECTION 5 — provenance arms replaced

The R1 arm *"a provenance-write failure must NOT change the decision … still admitted"* is
**deleted**, with a comment in its place recording why and pointing at the replacements.

`G-PREFLIGHT-PROVENANCE-FAIL-CLOSED` provides both required arms:

- **Case A** — `StagingPreflightProvenanceError` raised; and nothing became effective:
  `_resolved_high_water_files is None`, `frozen_trial_cohort(...) is None`,
  `get_preflight_plan(...) is None`;
- **Case A through the real serve loop** (`_prov_serve_fail_closed`) — terminal reason leads with
  `coordinator_staging_preflight_provenance:`, **zero `StripeAssign` reached the worker**, zero
  stripe rows, and **zero retry-matrix calls**;
- **Case B** — `StagingRetentionSizingError` is raised (exact type asserted, so a provenance error
  cannot substitute), the terminal cause is unchanged, and the provenance failure appears only as
  `[secondary: …]`.

---

## 6. Red-first evidence for both production changes

**Method, stated plainly:** I did not have a saved R1 tree, so I reconstructed R1's two
*behaviours* in a scratch copy of the working tree — `assign_stripes` back to the plain
`can_assign_variant` filter, `freeze_trial_cohort` made a no-op, and `_persist_preflight_plan`'s
`fail_closed` branch disabled so both cases are swallowed. The scratch copy is outside the repo;
the working tree was untouched.

Result on the reconstructed R1 tree (`/tmp/r2_REDFIRST.log`), **47/50**:

| arm | red-first reason |
|---|---|
| `G-LATE-WORKER-EXCLUDED` | `no assignable cohort was frozen at a successful preflight — the trial's worker population is unbounded after certification` |
| `G-PREFLIGHT-PROVENANCE-FAIL-CLOSED` | `a trial whose mandatory retention record could not be written was ADMITTED — Beta: the durable plan was not optional telemetry` |

The third failure is environment-only: `G-MATRIX-DIFF-a` runs `git show <rev>` and the scratch
directory is not a git repository. It passes in the real tree.

---

## 7. Full verification results

All runs on VM101 with `~/venvs/torch` active, **after the last edit**.

| suite | result |
|---|---|
| `test_s172_staging_backpressure.py` ×3 | **50/50, 50/50, 50/50** — exit 0, sentinel PASS |
| `test_s172_staging_partb.py` | **24/24** — exit 0, sentinel PASS |
| `tests/test_s172_elapsed_roundtrip.py` | **6/6** — exit 0, sentinel PASS |
| `test_s172_phase4_coordinator.py` (clean/committed) | **63/63** — Gate 22 **PASS**, Gate 37 **PASS** |

50 = the 48 at R1 + `G-LATE-WORKER-EXCLUDED` + `G-PREFLIGHT-PROVENANCE-FAIL-CLOSED`. The corrected
`G-MUT-STAGE-ELIGIBILITY` is green under its new purpose.

Phase-4 clean/committed used the same method as R1: the working-tree state copied to a scratch
directory, `git init` + commit **inside that throwaway repo**. No commit was made in
`~/distributed_prng_analysis`.

### Assertion-unchanged proof (AST, vs `git show c7058d8:<path>`)

```
=== BACKPRESSURE ===                 === PHASE-4 ===
  pre-existing        : 53             pre-existing        : 80
  assertion-IDENTICAL : 53             assertion-IDENTICAL : 79
  assertion-CHANGED   : NONE           assertion-CHANGED   : ['gate37_serve_path_two_workers']
  removed             : NONE           removed             : NONE
  added               : 18             added               : 1
```

**The only pre-existing gate whose assertions changed remains the already-authorized Gate-37
supersession.** `G-MATRIX-DIFF-a` is green, which is the independent proof that the retry matrix
and its callers are still byte-identical to `4b1aad6`.

**No gate-12 production run. No Phase-7 soak. No port 5700 bind.**

---

## 8. Files changed

| file | R2 changes |
|---|---|
| `miner/range_miner_coordinator.py` | cohort signature + freeze + `cohort_eligible`/`cohort_filter`; enforcement at the three selection sites; `StagingPreflightProvenanceError`; `fail_closed` provenance policy; serve_trial classification |
| `tests/test_s172_staging_backpressure.py` | `G-LATE-WORKER-EXCLUDED`, `G-PREFLIGHT-PROVENANCE-FAIL-CLOSED` (+ serve helper), corrected `G-MUT-STAGE-ELIGIBILITY`, deleted provenance arm |

Unchanged in R2 (carried from the amendment/R1): `agent_manifests/window_optimizer.json`,
`miner/range_miner_protocol.py`, `window_optimizer.py`,
`window_optimizer_integration_final.py`, `tests/test_s172_phase4_coordinator.py`.

**No file outside the existing change set was touched.** No `.gitignore` work, no new telemetry, no
seed-domain/cursor changes, no byte-model work.

---

## 9. Disagreements

**None.** Both production corrections were right, and R2 turned up two things that confirm it:

- the cohort-freeze work exposed that the retry path was a second, unguarded per-trial eligibility
  calculation — the initial-assignment fix alone would have left the invariant holding on the easy
  path and failing on the harder one;
- the late-worker fixture's first version was inert, and the gate caught it — which is the
  `_validate_caps` fact that also explains *why* a late joiner's danger is a backend change rather
  than a cap advertisement, exactly as Beta's example framed it.

I accept the withdrawal of the first-stage-undercount hypothesis and have removed the understatement
assertion built on it (§4).

---

## 10. Verification-integrity controls (VIR-1…6)

- **execution proof:** sentinels + exit codes on all four suites; logs under `/tmp/r2_*.log`.
- **clean control:** 48/48 + 6/6 at base before any R2 edit; 53/53 and 79/80 assertion-identical.
- **fault-injection control:** the reconstructed-R1 red-first run; the late-worker gate's own
  non-inertness assertion; the corrected mutant's non-inertness assertion; injected provenance
  failure in both cases.
- **completion sentinel:** present in all four suites.
- **unavailable-observer behavior:** the provenance policy is now fail-closed on the admit path and
  explicitly secondary on the refusal path — neither silently proceeds.
- **audit claim scope:** this repo tree on VM101 at `c7058d8` + working changes. **No claim about
  live fleet behaviour** — gate 12 and the soak were not run.
- **searched surfaces:** tracked repo; gitignored manifests read live with `/bin/grep`; `git show`
  of the committed baseline; live VM101 filesystem; live Python imports and execution; two scratch
  repos (red-first reconstruction, clean-committed phase-4).
- **unavailable surfaces:** live rigs; any GPU path; the gate-12 production run.
- **governance trail searched:** the R2 brief, the R1 brief and ruling, the amendment brief, my two
  prior reports; skill v19 §§2.7, 2.15, 2.19, §4, §7.

---

## 11. What is NOT done

- **Not committed, not pushed.** Michael commits and dual-pushes; build the `git add` list from §8.
- **Gate 12 / G-PROD-SHAPE and the Phase-7 soak remain HELD** — untouched and unrun.
- Nothing on Beta's closed list was reopened.
