# IMPLEMENTATION REPORT — WINDOW-ANCHOR / GENERATOR-PHASE SEPARATION, BRIEF I

**From:** Team Alpha · **Date:** 2026-08-22 · **For:** Team Beta code review.
**Brief:** `docs/S172_WINDOW_ANCHOR_BRIEF_I.md` · **Design of record:**
`docs/PROPOSAL_WINDOW_ANCHOR_GENERATOR_PHASE_SEPARATION_v1_1.md` (`1bf49a5`, APPROVED).
**Authority:** `docs/TB_RULING_WINDOW_ANCHOR_V1_1_DESIGN_GATE_CLOSED.md` (design gate CLOSED,
Brief I AUTHORIZED) + the scope ruling of 2026-08-21 (Items 1/2/3 APPROVED; ingress kept;
capability-before-policy ordering BINDING).
**Certified pre-change reference:** `gate12-passed-attempt9` = `e9ca800`.
**Written against and implemented at:** HEAD `205ae84c8093e75cbbc0967a857a30ed1c3ce434`.

**NOTHING IS COMMITTED.** Michael commits after Beta's review. Working tree only.

---

## 1. DIGEST VERIFICATION AND THE AC7 BASELINE

### 1.1 Digests at start — all four matched, no STOP condition

```
git rev-parse HEAD  ->  205ae84c8093e75cbbc0967a857a30ed1c3ce434     (expected)

miner/range_miner_worker.py       043522e96b44855f  == brief   ✅
miner/range_miner_coordinator.py  53b5ce87c02f46c9  == brief   ✅
miner/range_miner_npz_writer.py   36e2e34c7ab37a7d  == brief   ✅
reverse_sieve_filter.py           a68646086734fbdc  == brief   ✅
```

Post-change digests, for the review diff:

```
miner/range_miner_worker.py       ff306262d3f1bce8
miner/range_miner_coordinator.py  59888834bb1c86a2
miner/range_miner_npz_writer.py   1c72980c77b55732
reverse_sieve_filter.py           ad2c1554059c8e52
```

### 1.2 The AC7 baseline, captured BEFORE the first edit

`logs/ac7_baseline_205ae84/` — 45 suites, sequential (concurrent S172 runs flake on a
free-space race that reads exactly like a regression from one's own diff), 21m26s,
**32 green / 13 red**. `META.txt` records host, interpreter, HEAD and the working-tree state
at capture. `ROOT_CAUSES.md` resolves all 13 reds to **six root causes, each with a
`file:line`**, and carries its own correction notice (see §9.2).

Every subsequent battery is a diff against this file. Without it, no red in this report could
be attributed.

---

## 2. THE CHANGED-DEFINITION LIST, COMPUTED BY THE SCOPE PROOF

Not transcribed. Computed by re-implementing `_def_digests` against both pinned anchors:

```
MP-1 anchor 2c38f8cb   ADDED 0   REMOVED 0   CHANGED 21  (12 pre-existing + 9)
R-1  anchor c403a373   ADDED 0   REMOVED 0   CHANGED 13  ( 3 pre-existing + 9 + serve_trial)
```

The computed changed-set **matches the brief's nine exactly**; there is no delta to report.
`serve_trial` was already in MP-1's set from the FIELD-6 pass and is new to R-1's — exactly as
§6 predicted.

**`ADDED = 0` on both anchors is constraint 7 discharged**, and it is proven behaviourally by
mutant M12 rather than by assertion alone.

---

## 3. BOTH `DECLARED_CHANGED` SETS, AND BOTH PROOFS GREEN

SR-1 applied to both suites. Anchors NOT moved; the exact `changed == DECLARED_CHANGED`
comparison NOT relaxed; every added entry carries provenance.

Added to **both** sets, each tagged `[WINDOW-ANCHOR BRIEF I]` with a note that it is **not**
MP-1's or R-1's change:

```
MinerLedger._init_db                              _canonicalize_trial_context
MinerLedger.set_trial_context                     build_trial_context_from_serve
_trial_context_row_to_ctx                         derive_trial_metadata
RangeMinerCoordinator.build_stripe_assign_payload
RangeMinerCoordinator._dispatch_pending
run_trial_miner
```

`RangeMinerCoordinator.serve_trial` added to **R-1's only**, with its own note (already in
MP-1's from FIELD-6).

```
tests/test_s172_r1_drain_remedy.py        44/44   (baseline 44/44 — unchanged)
tests/test_s172_mp1_drain_attribution.py  38/38   (baseline 38/38 — unchanged)
```

---

## 4. G-CAP-1 — MEASURED ARITY AND SCALAR POSITIONS, ALL 24 COVERED VARIANTS

Asserted as **literals from the §0 C-1 table**, cross-checked against the production
`EXPECTED_KERNEL_ARITY`, never computed from the builder under test.

```
forward constant   java_lcg 14 · lcg32 15 · minstd 14 · pcg32 13 · xorshift32 15 · xorshift128 15
reverse constant   all six 12
forward hybrid     java_lcg 15 · lcg32 17 · minstd 15 · pcg32 15 · xorshift32 16 · xorshift128 16
reverse hybrid     all six 14
```

**This corrects v1.1 §3**, which states "13 (lcg32: 16)" for all six forward-constant variants
and is wrong for five of them (§0 C-1). Independently re-derived from the builders here.

The gate also pins the full **scalar `(position, dtype)` sequence** per variant — added after
mutant M6 survived the first form, which asserted only dtype membership while the docstring
promised position (§9.3).

**G-CAP-2:** the other 20 of 44 registry entries are exercised by asserting they refuse —
`NotImplementedError` from `resolve_builder` or `VariantStopCondition` from `_validate_variant`.

---

## 5. G-ABI-FROZEN — 44/44 KERNEL HASHES IDENTICAL

`HEAD:prng_registry.py` is executed in-process and its `KERNEL_REGISTRY` compared entry by
entry against the live one. **44 entries, 44 with `kernel_source`, 44 identical.** AC6's
"kernels unchanged by hash" holds. Proven non-vacuous by mutant M11.

---

## 6. AC1 — G-SEP-2 AND G-SEP-3 GREEN TOGETHER

AC1 is the PAIR. Both are green simultaneously in the same suite run.

**G-SEP-2 — the active half.** A synthetic `generator_phase = 7` driven through the INTERNAL
builder by arg-capture on a supported ABI:

```
lcg32_hybrid  arity 17
  args[16] = ScalarArg(value=7, dtype='int32')      <- position 17 of 17
  diff vs phase=0 : positions [16] ONLY — the other 16 args byte-identical
```

Zero-observed-on-both-paths is not accepted as independence evidence, so this drives 7 and
observes 7.

**G-SEP-3 — the fail-closed half, with EACH PIN PROVEN SEPARATELY LOAD-BEARING.** Beta requires
the v1 zero-pin at both seams. Two pins satisfied by one test is a false-green shape, so the
arms run on **disjoint call paths** (disjointness asserted structurally by AST over `ast.Call`
nodes, never by text) and each pin is then removed in turn:

```
ARM A  coordinator public assign-payload validation
       build_stripe_assign_payload(generator_phase=7) -> MinerMetadataError
ARM B  worker execution seam, coordinator builder BYPASSED (hand-built payload)
       SieveExecutor.execute -> GeneratorPhaseNotPermittedError
ARM C  C1 remove the coordinator pin -> ARM A reds, ARM B stays green
       C2 remove the worker pin      -> ARM B reds, ARM A stays green
```

C1's mutant is the **live** method reconstructed by AST with exactly one `If` deleted, and the
reconstruction asserts `len(body) == before - 1` so a mutant that failed to locate the pin is
reported rather than passing as a no-op.

**Capability and policy are separate invariants with distinct exception types**, ordered
capability-then-policy at the worker seam per Beta's ruling:

```
java_lcg_hybrid  phase=7  ->  GeneratorPhaseUnsupportedError    (capability; G-CAP-3)
lcg32_hybrid     phase=7  ->  GeneratorPhaseNotPermittedError   (policy;     G-SEP-3)
```

`java_lcg_hybrid` stops at capability and never reaches policy, so G-CAP-3 exercises the guard
it is named for. Reversed, the policy pin would reject nonzero on every variant and G-CAP-3
would go green without the capability guard executing.

---

## 7. §4.8 REACHABILITY — THE ROUTE TABLE ALPHA DERIVED, AND THE ROUTE THE BRIEF MISSED

Derived from live source by a scripted probe searching **three mechanisms independently**
(AST imports incl. `importlib`/`__import__`; subprocess invocation by filename, statement-scoped;
the `'reverse_sieve'` dispatch selector). Evidence: `logs/brief_i_evidence/`.

### 7.1 Four live dispatch routes, not three

| # | route | in the brief? |
|---|---|---|
| 1 | `distributed_worker.py:291` | yes |
| 2 | `coordinator.py:837` | yes |
| 3 | **`coordinator_sieve_dynamic.py:564`** | **NO** |
| 4 | `run_complete_pipeline.py:76` | yes |

**Route 3 is not a stray copy — it is a REPLACEMENT IMAGE for the live coordinator.**
`test_sieve_dynamic.sh:36` executes, and `enable_sieve_dynamic.py:151` instructs,
`cp coordinator_sieve_dynamic.py coordinator.py`. Closing routes 1/2/4 while leaving 3 open
would have meant **one `cp` silently reopening route 2**. This is the strongest possible
justification for §4.8's "deadness may NOT be assumed", and it was found only because the
brief's instruction to derive rather than transcribe was followed literally.

**Eight import consumers, not four.** The brief listed `identify_failures.py`,
`identify_failures_trace.py`, `test_real_candidates.py`, `retest_seed87.py`. Also importing:
`identify_failed_seeds.py`, `test_remote_seed.py`, `test_reverse_direct.py`,
`test_reverse_simple.py`. Two further test harnesses invoke the engine by subprocess.

### 7.2 Closure evidence

All four routes hard-disabled, each raising at the dispatch site with the stable token
`LEGACY_FUSED_ENGINE_CLOSED` and a message stating the route is **disabled, not skipped** — a
silent skip reads as a job that ran and found nothing.

**Entry guard boundary, and why it sits there:**

```
IMPORT                  free    8 diagnostic scripts read the archive; making the module
                                unimportable breaks the one thing archival code is for
load_draws_from_daily3  free    loads and slices; runs no generator, fuses nothing
EXECUTION              raises   run_reverse_sieve · run_hybrid_reverse_sieve ·
                                execute_reverse_job · main()
```

**The engine was NOT aligned to the new semantics.** Its clamp (C-2 site 4) is untouched and
the fused implementation is retained as archival code, per v1.1's freeze-over-retrofit
conditional on closure.

`G-LEGACY-1` re-derives the census every run and reds on a new **or vanished** route in either
direction. `G-LEGACY-1b` asserts the replacement-image relationship still exists, **with
self-exclusion** — without it the gate matched its own source and would have stayed green with
both real call sites deleted (§9.3).

---

## 8. MUTATION RECORD — 15 MUTANTS

`tests/test_s172_window_anchor_brief_i_mutants.py`. Every mutant is classified DETECTED /
SURVIVED / INVALID. Credit requires **applied** (hunk matches exactly once), **executed**
(failure frame captured — file, function, line), and **reached the credited assertion**
(`AssertionError` from inside the gate, not an import or type error). Every mutant carries a
**clean control**: its credited gates are run on unmutated source and must be green first.
Every gate runs in a **fresh interpreter**, because an already-imported module defeats a source
mutation entirely.

```
clean controls  16/16 green
DETECTED 14/15   INVALID 1   SURVIVED 0
prng_registry.py CLEAN after the run — no mutant residue (git diff --quiet)
```

| # | mutation | credited | outcome |
|---|---|---|---|
| M1 | restore the silent clamp | G-DOMAIN-1 **and** G-DOMAIN-2 | DETECTED |
| M2 | reinstate `.get("window_anchor", 0)` | G-REJECT-3 | DETECTED |
| M3 | accept **and** map `offset` | G-REJECT-1, G-REJECT-2 | DETECTED |
| M4 | `java_lcg_hybrid` into the capable set | G-CAP-3 | DETECTED |
| M5 | drop the phase arg | G-CAP-1, G-CAP-4 | DETECTED |
| M6 | move the phase one slot earlier | G-CAP-1 | DETECTED |
| M7 | `derived_max` before the session filter | G-DOMAIN-3 | DETECTED |
| M8 | `control_era` ceiling → 149 | — | **INVALID by scope** |
| M9 | revert `_CONTEXT_FIELDS` | G-TUPLE, G-PHASE5-SEAM | DETECTED |
| M10 | re-enable a legacy route | G-LEGACY-2 | DETECTED |
| M11 | one byte of a kernel body | G-ABI-FROZEN | DETECTED |
| M12 | new `def` in the coordinator | both scope proofs | DETECTED |
| M13 | phase 3 through the public schema | G-SEP-3 | DETECTED |
| M14 | fusion at the execute seam | G-SEP-1 | DETECTED |
| M15 | **structural: anchor into `BuildContext`** | **G-NO-FUSED**, G-CAP-1 | DETECTED |

M1 is credited to two gates deliberately: C-2 established the shared-authority path, so
one-sided detection is insufficient. M3 doubles as G-REJECT-2's injection proof. M12 proves
constraint 7 and `DECLARED_ADDED = set()` behaviourally.

### 8.1 Why one mutant became two — the brief's dual credit could not hold

The brief credits **M14 to G-SEP-1 AND G-NO-FUSED**. Those gates assert the same property at
**different levels** — G-SEP-1 on the real `SieveExecutor.execute` path, G-NO-FUSED at the
builder/`BuildContext` level — so a single mutation can only ever reach one of them. Widening
the credit would have produced a mutant one gate **structurally could not catch**: the exact
false-green shape this suite exists to refuse.

The mutant was therefore **split, not the credit widened**:

```
M14  one-line fusion at the execute seam (+ the pin, which exists only because
     of the separation)                              -> G-SEP-1
M15  STRUCTURAL: window_anchor into BuildContext, emitted as a ScalarArg  -> G-NO-FUSED
```

M15 required G-NO-FUSED to gain a **by-construction arm** — the anchor cannot reach a kernel
scalar because it is not in the build context at all — since the existing behavioural arms stay
true when one merely adds the *capability* to fuse. M15's evidence shows G-NO-FUSED's own
assertion firing:

```
gate_no_fused: AssertionError in gate_no_fused:385
  -> "BuildContext carries ['window_anchor'] — the host-side anchor has entered the
      DEVICE-side build context, so a builder can now emit it as a kernel scalar."
```

Without M15, AC5's repo-level gate would have rested on construction-only non-vacuity.

### 8.2 Three mutants were INVALID as first written — reported, not silently swapped

- **M8 — INVALID BY SCOPE, with the argument, not just the label.** The mutation **has no
  Brief-I code site to apply to**: `control_era`'s ceiling lives on the era-resolution surface,
  which v1.1 §4.2 places with the optimizer surface and §3's firewall assigns to Brief II. At
  Brief I no production expression computes a control-era bound — an **absent target**, not a
  skipped test and not a gate that failed to catch something. G-ENVELOPE is correspondingly
  scoped to the bound **arithmetic**, which is the half of Q4 that carries the category error.
  **CARRY-FORWARD: the era-ceiling mutant transfers to Brief II with the era-resolution work
  and must be run there against the real bound. The coverage gap is inherited deliberately.**
- **M11 — invalid as first written.** `kernel_source` is a dict KEY pointing at a module
  constant (`'kernel_source': XORSHIFT32_KERNEL`), so the mutation landed on an unrelated
  docstring; **no kernel_source changed and G-ABI-FROZEN was correct to stay green.** The
  mutant never executed the path it was credited against. Retargeted onto a `*_KERNEL = r'''`
  body, with an assertion that the target was found so a future rename makes it INVALID rather
  than silently vacuous.
- **M14 — invalid as first written.** The one-hunk fusion was caught by production's v1 policy
  pin **before** G-SEP-1 reached its assertion — good news about production, useless as a
  mutant. Rebuilt as two hunks, because F-4 returning means both halves of the separation
  revert at that seam.

Both invalid forms are documented **in-file** with why they were invalid.

---

## 9. FINDINGS

### 9.1 Three census corrections against the brief — one mechanism, three times

| brief says | at source | missed |
|---|---|---|
| the clamp exists in **four** places (C-2) | **nine** implementations | `sieve_gpu_worker.py:113` (live production), three stale duplicates, `tests/phase6/known_answer_reference.py:161`; and C-2's own site 2 is a **variant spelling** (`config.offset`/`config.window_size`), not "the identical expression" |
| **three** live dispatch routes (§2.4) | **four** | `coordinator_sieve_dynamic.py:564` — the replacement image |
| **four** import consumers (§2.4) | **eight** | `identify_failed_seeds`, `test_remote_seed`, `test_reverse_direct`, `test_reverse_simple` |

**The mechanism is identical in all three: a token grep, pre-filtered or truncated, with the
survivors counted.** It is also why C-2's census missed its own site 2 — a string search cannot
see a variant spelling, which is precisely the hazard §3 warns about for renames.

**The census method that found them is the standard this report proposes:** search
*mechanisms* independently (AST imports, subprocess invocation, dispatch selector), scope
co-occurrence to **leaf statements** (walking a whole `FunctionDef` collects every string in its
body and produced a false positive), and **exclude the probe's own file**.

Two of the missed clamp sites carry weight: `sieve_gpu_worker.py:113` is live production, and
`tests/phase6/known_answer_reference.py:161` is the deliberately independent known-answer
reference. **The KAT reference must keep its own implementation** — independence is the point —
but it now retains fused clamp semantics. In-domain the clamp is the identity, so **KAT parity
holds for every legal anchor**; the two diverge only out of domain, where production now raises.
Nobody should later "fix" the reference into a copy of production.

### 9.2 RC-1 — eight suites share a mechanism the record attributes to one

**Independent of Brief I. True at `205ae84` before any edit, and equally true after.** Post-F1
(`c4e0037`), `assign_stripes` creates stripes `pending` / `claimed_by NULL` and
`schedule_pending_stripes` is the only creator of a compute lease. Fixtures that drive
`assign_stripes` and then expect a claim or a dispatch fail **in the fixture**:

```
{'worker_id': None, 'attempt': 0, 'expected_substripes': None,
 'effective_cap': None, 'claimed': False}
```

```
d1_engine 0/18 · d1_workflow 5/8 · d2 6/7 · d5 7/25 · d4_serial_backend 4/8
d6_production_adapter 0/9 · d6_threshold_path 4/17 · threshold_propagation 4/5
```

Skill §2.51 item 5 records **one** of these eight. Alpha corrected its own first-pass baseline
attribution here: four were initially grouped under Gate D0-7 because a **nested** D0 run prints
its `11/12` inside their logs. The correction is recorded in `ROOT_CAUSES.md` **with the
correction stated**, not silently rewritten. Brief I does not propose to fix RC-1 and has not
touched it.

### 9.3 Nine tooling bugs Alpha found in its own work — and the pattern

Reported because the pattern is the finding, not the individual bugs.

| # | bug | direction | found by |
|---|---|---|---|
| 1 | G-LEGACY-2: `contains_dispatch` matched the guard's own message | false RED | running it |
| 2 | census co-occurrence walked whole `FunctionDef`s | false RED | running it |
| 3 | **G-LEGACY-1b matched its own source** | **false GREEN** | **fault injection** |
| 4 | G-MIGRATE asserted `"not recoverable"` against `"cannot be recovered"` | false RED | running it |
| 5 | G-SEP-3 disjointness matched a **comment** documenting the call | false RED | running it |
| 6 | G-SEP-1 `anchor not in scalars` collided with java_lcg's `c = 11` | false RED | running it |
| 7 | **G-CAP-1 checked dtype membership while its docstring promised position** | **false GREEN** | **mutant M6** |
| 8 | mutation driver imported `tests.*` as a package (no `__init__.py`) | harness | the framework refusing credit |
| 9 | an edit truncated the mutants file, deleting `main()` | silent no-op | **the `EXIT=/SENTINEL_DONE` wrapper** |

**Two were false greens, and neither was findable by reading a tally.** Bug 3 is the session's
thesis demonstrated on Alpha's own work: a vacuous gate **inside the suite written to prove
closure**, invisible to the count, found only by removing what it asserts and demanding RED.
Bug 7 is the same class: **a docstring promising more than its code checked**, in the gate whose
job is catching exactly that drift.

**The dominant pattern is instrumentation misreporting production, not production being
wrong.** Bugs 1, 2, 4, 5, 6 and 8 all reported a defect that did not exist; bug 9 reported
success for a suite that never ran. Two sub-patterns are worth naming:

1. **In a codebase that documents its contracts inline, every text probe is a candidate false
   positive** (bugs 1, 5 matched prose describing the forbidden thing) **and every text probe
   over a file that forbids a string is a candidate false negative** (bug 3). The remedy applied
   throughout: AST over text, and self-exclusion asserted rather than assumed.
2. **A parse check cannot catch a file that parses perfectly and does nothing** (bug 9). Verify
   the artifact — `grep -c "def main"` — not the operation that produced it.

### 9.4 The pinned-executable-source hazard class — NEW, two members, one latent

Brief I broke a **certified** RED arm in a way SR-1 does not cover. SR-1 governs pinned
**digests**; here the pin is **executable source**, and an authorized contract change does not
merely move a digest — it stops the historical code from running at all.

`test_s172_attempt6_remediation.py` executes control-plane source pinned at `2b0d2dc` against
**live** helpers. The coupling surface was **enumerated, not discovered one failure at a time**:

| pinned call site | args | live now | bridge |
|---|---|---|---|
| `build_trial_context_from_serve(...)` | 3 pos | **schema** changed | bridged (in the exec globals) |
| `self.set_trial_context(...)` | 2 pos | body changed, **arity unchanged** | none needed |
| `self._dispatch_pending(...)` | 14 pos | **16 positional-or-kw** | bridged (instance-scoped) |

Both bridges are **test-local**: one lives in the pinned execution's own globals, one shadows a
bound method on that arm's coordinator. Production is untouched; the hard reject of the legacy
key is unaffected. Fault-injection evidence is the differential itself — **77/78 without the
bridges, 78/78 with** (`/tmp/claude-1000/a6c.log`, `a6d.log`).

**Class membership, verified rather than assumed:**

| suite | mechanism | status |
|---|---|---|
| `attempt6_remediation` | 9 × `exec(compile(...))`, globals `dict(vars(COORD))` | **broke; translated** |
| `r1_drain_remedy` | 1 × `exec(compile(...))`, globals `COORD.__dict__` | **LATENT** |
| `gate12_cleantree_admission` | commit-pinned **shell** script, subprocess | not Python-schema coupled; already UNAVAILABLE-on-drift |
| `mp1_drain_attribution` | commit-pinned, **digests only** | **not in the class** |

**`r1_drain_remedy` survived on a four-of-ten coincidence.** Its pinned `_pump_deferred`
resolves exactly four global names from the live namespace — `BaseException`, `List`,
`PhaseCharge`, `tuple`. Brief I changed **ten** names in that same namespace. The intersection
is **EMPTY**. Nothing guarded that; the sets simply did not touch, and its real exposure is one
name, `PhaseCharge`.

**Two classes must not be conflated:** `HEAD:`-pinned reds are **transient** and self-heal at
commit; **commit-pinned reds are permanent until someone translates them.**

**RECOMMENDED, NOT ENACTED — an SR-1 analog for executable pins:** *any commit-pinned exec'd arm
must be re-checked against the set of live names it resolves whenever those names' schemas or
signatures change.* The set is computable (`symtable` over the pinned function), so the
obligation is checkable rather than a matter of remembering. **Scope it to the class, not to
`attempt6_remediation`** — a rule scoped to the member that happened to break would leave the
latent one uncovered on exactly the coincidence above.

---

## 10. DEP-ABI-V2 — RECORDED, NOT BUILT

Independent nonzero generator phase on the four no-phase forward hybrids —
`java_lcg_hybrid`, `minstd_hybrid`, `xorshift32_hybrid`, `xorshift128_hybrid` — remains
**DEP-ABI-V2: recorded and NOT built.** It requires a new kernel plus a parity certification
cycle behind its own Beta ruling. Brief I ships the capability guard so ABI-v2 cannot arrive
before it, and the guard's message names DEP-ABI-V2 explicitly.

`PHASE_CAPABLE_VARIANTS` is **enumerated deliberately, never computed from a suffix rule** —
the membership is irregular (`lcg32_hybrid` and `pcg32_hybrid` carry a phase argument; the
other four forward hybrids do not), and a rule reproducing it today would mis-generalize the
moment ABI-v2 lands. An import-time assertion proves the capable and incapable sets **partition**
the 24 covered variants exactly.

---

## 11. STEP-3 `continuation_phase` — NOT TOUCHED, WITH EVIDENCE

The Step-3 consumer law `offset = train_history_len` lives at **`full_scoring_worker.py:300`**.
That file does not appear in `git diff --name-only` at any point in this work. Constraint 6 is
satisfied by construction, not by inspection.

---

## 12. AC7 — BATTERIES, AND THE TWO COMMIT-TIME TRANSIENTS

### 12.1 Result — RECONCILED AGAINST THE FINAL 47-SUITE BATTERY

`logs/ac7_final/` — the full battery re-run on the FINAL tree, after §2.1-§2.4, the gate suite,
the mutants and Item 2. **47 suites** (45 baseline + the two new Brief-I suites).

```
final:  32 green / 15 red
green -> red vs baseline :  2   (both the documented commit-time transients)
red   -> green vs baseline:  0
new suites               :  2   test_s172_window_anchor_brief_i          25/25 green
                               test_s172_window_anchor_brief_i_mutants   green
```

**The picture holds on the final tree:** 13 pre-existing reds still red, the 2 transients are
exactly the two documented in §12.2, `test_chapter2_content_gate` is **green** (Item 2), and
**no suite went red that this report does not account for.**

**This tally is not sufficient on its own, and must not be read alone.** A pre-existing red that
stays red is adequate regression evidence only if its failure point did not move. Four of them
moved; §12.1.1 states each one. *(Code-review ruling, `docs/TB_RULING_WINDOW_ANCHOR_BRIEF_I_CODE_REVIEW.md`,
§12: "A pre-existing red remaining red is not, by itself, adequate regression evidence if its
failure point moved.")*

### 12.1.1 FOUR SUITES MOVED WITHIN THEIR REDS — stated as findings

"Still red" is not the same as "unchanged", and three already-red suites fail **deeper** under
Brief I. Reported here rather than absorbed into the pre-existing count.

**Governing characterization — Beta's wording, binding on this section:**

> **same pre-existing root cause / changed observable failure depth / no new Brief-I production defect**

The same ruling forbids describing these three as *"zero differential"* at the suite level;
that phrase appears nowhere in this report. The depth change is an observable of the new
fail-closed guard meeting an already-stale fixture — **not** a production defect introduced
by Brief I, and **not** a repair owed by it.

| suite | baseline | final | direction |
|---|---|---|---|
| `test_s172_phase5_d0` | 11/12 | **0/12** | DEEPER |
| `test_s172_phase5_d1_workflow` | 5/8 | **1/8** | DEEPER |
| `test_s172_phase5_d2_directional_uniqueness` | 6/7 | **0/7** | DEEPER |
| `test_chapter1_p0_corrections` | 8/12 | **10/12** | IMPROVED — cause NOT established |

**The three deepenings are one cause**, counted in each log:

```
MinerMetadataError: serve context missing mandatory field(s)
  ['generator_phase', 'window_anchor']        d0 ×6 · d1_workflow ×4 · d2 ×2
```

These are RC-1-family suites whose fixtures build a serve context directly. Brief I's
fail-closed guard now rejects that context, so arms which previously failed later at
`_build_run` now fail earlier. **Their fixtures were deliberately NOT migrated:** migrating them
would not turn any of them green — RC-1 (§9.2) fails them independently and is out of Brief I's
scope — so a migration would have changed which line they die on without changing the outcome,
while enlarging the diff across four certified suites. **Recorded as a Brief II / RC-1
carry-forward, not silently deepened.** Beta ratified this: *"I agree with not migrating them in
Brief I. Repairing them would be unrelated RC-1 fixture work and would expand this commit
substantially."*

**The improvement is NOT claimed as Brief I's doing.** `chapter1_p0`'s two VIR-2 clean controls
— `G-FLAG-FAILCLOSED` and `G-STRATEGY-FAILCLOSED`, which at baseline reported *"flag-absent run
never reached run_bayesian_optimization"* — now pass, while the M1/M2 mutants that depend on
them still fail, so the suite stays red for the same underlying reason. **Alpha has not
established the cause and does not credit it to this work.** An unexplained green is exactly as
suspicious as an unexplained red; it is flagged here for its own investigation. Beta concurs:
*"Improvement of a broken fixture is not evidence that this implementation fixed its underlying
defect. Keep it pending its own investigation."*

### 12.2 Detail


The 13 pre-existing reds identified in §1.2 are **all still red, none masked, none newly red**.
No pre-existing red was "fixed" as a side effect, and none was hidden.

Six suites were chargeable to Brief I and five were retired by fixture migration:

```
test_s172_phase3_worker         17/17 -> 18/18   (+1 = the new §3B ordering gate)
test_s172_admission_liveness    retired
test_s172_f1_lease_origin       retired
test_s172_staging_backpressure  retired
test_s172_attempt6_remediation  78/78 (via the §9.4 bridges)
```

**`phase3_worker` is 8/8 fixture-shape, 0 assertion changes**, and that claim is proven rather
than asserted — assert-lines diffed against HEAD:

```
assertions at HEAD 130   live 135
REMOVED or CHANGED existing assertions:  0
ADDED: 4   — all four inside the new Gate 15b
```

**No existing assertion was relaxed to accommodate the schema.** Item 3B's four integrity gates
(phase3 15/16/17, phase4 Gate 20) now construct a **schema-valid payload and mutate only the
integrity property**; Gate 20 needed no repair at all, because it builds its payload through the
real `build_stripe_assign_payload` and became schema-valid by construction. Gate 15b makes the
ordering a tested property in both directions.

### 12.3 The two transients — RECORDED TOGETHER, and they are not regressions

**Both are artifacts of an uncommitted working tree and will appear in EVERY battery run until
Michael commits.** Neither is a regression and neither is a reason to widen anything.

| suite | red | mechanism | clears |
|---|---|---|---|
| `test_s172_phase4_coordinator` | 62/63, **Gate 22 only** | `gate22_coexistence` reads `git status --porcelain` and filters `.py`; modified-tracked trips it too, not only untracked | at commit |
| `test_gate12_cleantree_admission` | 30/31, **W-NO-WEAKENING** | `head_sha()` computes `git show HEAD:{rel}` **live** — a working-tree-equals-HEAD property check with no stored digest | at commit |

W-NO-WEAKENING is **correct to be red**: it is reporting an uncommitted edit to a frozen
surface, which is its job. Its only limitation is that it cannot distinguish authorized from
unauthorized until the commit exists. **D3.5 is not weakened in substance**, measured four ways:

```
producer identity  G._repository_state is WOI._repository_state    True
_repository_state  AST digest identical to HEAD                    True
utils/run_finalizer.py                                             untouched
.gitignore                                                         untouched
```

A **third** development-time red is expected once the new suites are staged: Gate 22 also trips
on new untracked `.py` under `tests/`. Documented case; the allowlist stays untouched.

### 12.4 A Gate 22 blind spot, recorded as a finding

`gate22_coexistence` (`tests/test_s172_phase4_coordinator.py:1620-1628`) filters
`ln[3:].strip().endswith(".py")`. Default `git status --porcelain` **collapses a wholly-untracked
directory to a single entry** — this session's baseline `META.txt` records `?? piece_matcher/` —
which does not end in `.py`. **Gate 22 is BLIND to it, not passing it.** The moment any file
under such a directory becomes tracked, git lists the directory's contents individually and
Gate 22 reds. Recorded; no change proposed.

Separately, the untracked `piece_matcher/` tree **does** contaminate one outcome — the S145
production-certification-bypass scan walks the **filesystem**, not git, and fails on
`piece_matcher/search/coverage_ledger.py:273 raw INSERT` (`test_seed_domain_cursor_amendment`
39/40). It is the only such red, touches nothing Brief I modifies, and self-clears on
disposition.

---

## 13. ITEM 2 — CHAPTER RE-DISPOSITION UNDER BETA'S TIMING CONSTRAINT

**No unqualified `REPAIRED` appears anywhere.** Every disposition is two-part: the historical
verdict preserved verbatim, the subsequent change recorded as *implemented, acceptance pending*.

| site | treatment |
|---|---|
| CH2 §7.2 | retitled *"What the code **did**"*, framed **"AS AT THE CHAPTER-2 AUDIT ANCHOR"**, closing **"the verdict at the audit anchor stands as written: F-4 CONFIRMED, NOT REPAIRED"** |
| CH2 §7.2.1 (new) | *"Status: repair implemented by Window-Anchor Brief I; acceptance pending"* + the binding contract table |
| CH2 :831 | analysis intact; block quote notes F-4 was resolved **by splitting the scalar, not by multiplying it** — no `offset*(skip+1)` was applied |
| CH2 :1133 | *"**CONFIRMED** at the audit anchor … SUBSEQUENT: … acceptance pending. The CONFIRMED verdict above is historical and is not rewritten."* |
| CH2 :1346 | *"**CONFIRMED, not repaired AT THIS ANCHOR** … The anchor verdict is preserved as recorded."* |
| CH1 :332/:337 | same coupling, same treatment, pointing at §7.2.1 |

**F-4 stays in the chapter**, so `:578`'s text requirement is satisfied by content.

**The two source assertions are repointed, not relaxed**, and a third is added in the opposite
direction (the fused clamp must NOT return). Self-exclusion is asserted, not assumed. Both are
**proven load-bearing by injection** — clean control green, each string removed from live worker
source in turn, RED demanded, source restored, gate green again:

```
device-side delivery surface: RED (§7.2.1's device-side consumer (_generator_phase_tail) is gone...)
host-side anchor validation:  RED (§7.2.1's host-side anchor validation is gone from source...)
```

The injection proof lives in **Brief I's suite**, so `test_chapter2_content_gate` returns to
**12/12** with its own count untouched.

**Scope fence held.** The anchor-loop weakness (existence-only checking — §2.1 grew the worker
2,629 → 2,892 lines, so all six F-4 anchors still "resolve" while pointing at unrelated code)
and `:578`'s near-vacuity (`"F-4"` occurs 5 times and `"offset"` 24 times in a 1,463-line
document) are **recorded in-file as follow-up debt and deliberately NOT repaired.** No general
gate redesign.

---

## 14. SCOPE DISCLOSURES

### 14.1 The miner ingress — `window_optimizer_integration_final.py:1507`

Approved by Beta and **kept**, reframed per the ruling: this is **caller adaptation, not
analogy**. Renaming a parameter obligates its callers; leaving one on the old name is a
half-applied rename.

The decisive fact is the `**kwargs` tail at `range_miner_coordinator.py:12406`. Without it,
`offset=` raises `TypeError` and deferring to Brief II is safe. **With it, the choice is not
"break loudly vs fix" — it is "silently swallow vs fix"**, and constraint 4 requires fail-loud
wherever the old code was silent. Honouring the firewall here would have shipped the defect
Brief I exists to remove.

The irony is on the record: the **Blocker-2b comment directly above `:1507`** promises *"a
malformed substitute object raises AttributeError loudly instead of silently coercing a missing
field"* — and a `**kwargs` tail twelve thousand lines away voids that promise for any renamed
field.

Bounded deliberately: `WindowConfig.offset` is **not** renamed and the Optuna surface is **not**
touched. `[SCOPE DISCLOSED]` remains in-file.

### 14.2 Schema-before-integrity ordering (Item 3A, APPROVED)

Containment, not preference: schema validation only **adds** a rejection. A schema-valid payload
reaches the `dataset_sha256` check exactly as before, so **the accepted set after the change is
a strict subset of the set before, and no payload that was rejected is now accepted.**
Classification is identical — `ResidueResolutionError` and `ResidueVerificationError` both
subclass `ResidueError` (HEAD `:512-521`; **working tree `:719-727`**), routed to
`_fail_stripe(retryable=False)` (HEAD `:2000-2004`; **working tree `:2312-2318`**). Only
the diagnostic differs. Four certified gates were affected, not one.

*(The working-tree anchors above were re-measured against the final tree. The ruling
request carried `:669-679` / `:2258-2264`, which were correct when it was written and
went stale when the worker-seam policy pin and its exception class landed afterwards —
a line-number drift of the kind §1.2 of the project skill describes. HEAD anchors are
unaffected.)*

---

## 15. CARRY-FORWARDS

### 15.1 Brief II

1. The Optuna surface — `suggest_int('offset')` → `window_anchor`, `{min, max_cap}`, the
   derived-max resolver, the `O{offset}` cache-key fragment.
2. **`window_optimizer.py:74` (C-4).** `load_search_bounds_from_config` merges config over a
   **hardcoded defaults dict** containing `"offset": {"min": 0, "max": 100}`. Deleting the JSON
   key does **not** retire the bound — it falls back to the hardcoded default. §4.3's "removed
   outright" is not true in the running system until this line goes too.
3. **C-2 sites 2 and 3** (`window_optimizer_integration_final.py:266`, `sieve_filter.py:184`)
   plus the four this report adds (§9.1), in the repo-wide consumer audit.
4. `parameter_registry.py:281-289`; `distributed_config.json` key removal;
   `optimal_window_config.json` migration.
5. NPZ generation metadata and `anchor_era` provenance (§4.5).
6. `WindowConfig.offset` rename — which retires the ingress name mismatch in §14.1.
7. **M8, the era-ceiling mutant**, run against the real bound once era resolution lands.
8. **AC4** (D.1 differential reach) and the **AC7 re-run**.
9. **Any future trial-context schema OR signature change must re-check every commit-pinned
   exec'd arm against the live names it resolves** (§9.4) — `_pinned_trial_context` and
   `_pinned_dispatch` in `attempt6_remediation` are the current instances.

### 15.2 Follow-up debt, recorded not repaired

- The Chapter-2 **anchor loop** checks resolution, not correctness; `:578` is keyword-presence
  over 1,463 lines. Beta's scope fence: no general gate redesign.
- **G-NO-FUSED's builder-level arm** now has M15; no other gate in this suite lacks a mutant.
- **`run_trial_miner`'s `**kwargs` tail IS the §2.7 instance-7 mechanism.** It swallowed four
  staging controls once and will swallow the next renamed field. The repair is removing the
  tail — too broad for Brief I.
- **RC-1** (§9.2) — Beta may wish to open it as its own item.

---

## 16. SEQUENTIAL LINEAGE

```
gate12-passed-attempt9 = e9ca800        certified pre-change reference
   |
   +-- ... governance: sequencing ruling -> proposal v1.0 -> v1.1 -> design gate CLOSED
   |
   +-- 205ae84                          HEAD this brief was written against and
   |                                    implemented at; digests verified; AC7 baseline
   |                                    captured BEFORE the first edit
   |
   +-- [PENDING] Brief-I commit         this working tree, after Beta review
   |
   +-- [FUTURE]  Brief-II commit        starts FROM the accepted Brief-I commit,
                                        never independently from e9ca800
```

The final acceptance report at the end of Brief II must show this lineage plus the full pre/post
diff back to `e9ca800`.

**Not acceptance evidence for any of this: the Phase-7 soak.** It is classified non-certifying
for window-anchor semantics and must not be cited for the merge.

---

## 17. CHANGE SET

```
 +360  -42   miner/range_miner_worker.py            §2.1 + worker policy pin
 +116  -29   miner/range_miner_coordinator.py       §2.2 + coordinator pin + G-MIGRATE
  +16   -4   miner/range_miner_npz_writer.py        §2.3 (2 executable statements)
  +27   -0   reverse_sieve_filter.py                §2.4 engine closure
   +9   -1   coordinator.py                         §2.4 route 2
  +10   -1   coordinator_sieve_dynamic.py           §2.4 route 3 (not in the brief)
   +8   -0   distributed_worker.py                  §2.4 route 1
   +7   -1   run_complete_pipeline.py               §2.4 route 4
  +19   -1   window_optimizer_integration_final.py  ingress [SCOPE DISCLOSED]
  +50  -11   docs/CHAPTER_2_BIDIRECTIONAL_SIEVE.md  Item 2
  +14   -7   docs/CHAPTER_1_WINDOW_OPTIMIZER.md     Item 2
  +31   -6   tests/test_chapter2_content_gate.py    Item 2, assertions repointed
  +15   -0   tests/test_s172_mp1_drain_attribution.py   SR-1
  +21   -1   tests/test_s172_r1_drain_remedy.py         SR-1
  +80  -22   tests/test_s172_phase3_worker.py           fixtures + Gate 15b
  +17  -14   tests/test_s172_phase4_coordinator.py      fixtures
  +64   -4   tests/test_s172_attempt6_remediation.py    two pinned-source bridges
  +10   -5   tests/test_s172_staging_backpressure.py    fixtures
   +2   -1   tests/test_s172_admission_liveness.py      fixtures
   +2   -1   tests/test_s172_f1_lease_origin.py         fixtures

 NEW  tests/test_s172_window_anchor_brief_i.py          25 gates
 NEW  tests/test_s172_window_anchor_brief_i_mutants.py  15 mutants
 NEW  docs/TB_RULING_REQUEST_WINDOW_ANCHOR_BRIEF_I_SCOPE.md
 NEW  docs/S172_WINDOW_ANCHOR_BRIEF_I_REPORT.md         this report
 NEW  logs/  (gitignored)  ac7_baseline_205ae84/ · ac7_post_*/ · brief_i_evidence/
```

**Nothing is committed. Michael commits after Beta's review.**

---

## 18. VERIFICATION-INTEGRITY CONTROLS (VIR-1…6)

- **execution proof:** every gate returns a measured detail string; mutants record the failing
  frame (file, function, line); batteries carry per-suite exit codes and logs.
- **clean control:** AC7 baseline at `205ae84` before the first edit; every mutant's credited
  gates run green on unmutated source before mutation.
- **fault-injection control:** 14 detected mutants; G-LEGACY-1b, G-SEP-3 ARM C and G-CH2-ANCHORS
  each carry their own inline injection proof.
- **completion sentinel:** every suite prints one; long runs wrapped in
  `; echo "EXIT=$? SENTINEL_DONE"` so a killed run cannot read as pending — which caught bug 9.
- **unavailable-observer behaviour:** G-SEP-1 terminates UNAVAILABLE, never PASS, without a GPU.
  `tests/gate_s172_prod_shape.py` (G-PROD-SHAPE) is **NOT RUN** — Michael-initiated, needs a live
  25-daemon fleet.
- **audit claim scope:** Brief I only. Brief II surfaces are named, not touched.
- **searched surfaces:** live source on VM101 · git history (`git show <commit>:`) · the
  governance trail (`TB_RULING_*`, `PROPOSAL_*`) · Chapters 1 and 2 · `tests/` · untracked
  working-tree files · the runtime SQLite ledger schema.
- **unavailable surfaces:** the rigs (no deployment in Brief I) · the Proxmox host kernel log ·
  the ser8 pre-repository archive.
- **governance trail searched:** yes — sequencing ruling, v1.0 and v1.1 rulings, the v1.1 design
  gate closure, the 2026-08-21 scope ruling.
- **chapters searched:** Chapter 1 §3.1.2, Chapter 2 §7.2/§7.3/§12 and the closure table.
