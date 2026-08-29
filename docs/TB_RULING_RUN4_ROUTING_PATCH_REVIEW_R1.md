# TB RULING — RUN-4 ROUTE-A PATCH REVIEW R1

**Received:** 2026-08-28 · **Recorded by:** Team Alpha (verbatim; delivered via ser8 → VM101) · **Status:** BINDING
**Reviews:** patch digest `0398c0d1…` (+201/−11), brief revision 2, recorded suite results

## Dispositions

| Item | Disposition |
|---|---|
| Patch digest `0398c0d1…` | **DO NOT COMMIT** — two bounded executable corrections required; no architectural redesign |
| Previous four required edits (invocation isolation, pre-edit `69ca910` oracle, value+type equality, M2a/M2b split) | **IMPLEMENTED CORRECTLY** |
| **BLOCKER 1 — operator origin** | Capture from generic `params` at `run_pipeline` entry proves the bundle existed, not who supplied it; a programmatic caller supplying seven keys is misclassified as explicit operator. REQUIRED: a separate invocation-local **operator-authority channel** (e.g. `_operator_pin_params` keyword or private authority object) supplied ONLY by the real CLI `--run-pipeline --params` entry point; `capture_step1_pin_bundle()` consumes the authority source, never ordinary `params`; all other callers default to no pin authority; legitimate retry within an authorized invocation keeps the frozen bundle. ADD **G-ORIGIN** (seven keys in ordinary `params` without authority → 0/7 routable, no provenance; same seven via the authority path → 7/7 + provenance) and a **mutant reverting the capture source to ordinary `params`**, detected by G-ORIGIN. Fail-**loud** preferred when the seven appear on a non-authorized invocation (WATCHER-manufactured values fail closed); the essential property is that such a caller can never acquire authority or route the pin |
| **BLOCKER 2 — null/empty false-pass** | `params[k] is not None` has two holes: `""` counts as supplied (all-empty bundle logs "pin accepted, 7 keys" while the `:1862` command-builder skip routes fewer or zero), and explicit `None` members silently collapse to "no pin" instead of failing as malformed. REQUIRED: distinguish **key presence** from **usable value** — `present = STEP1_EXPLICIT_PIN_KEYS ∩ params.keys()`; empty → no pin; otherwise all seven present AND none carrying a value the builder treats as absent (`None` or `""`); any violation fails loud before a command is built. EXTEND G-PARTIAL-CLOSED (or sibling): each of seven individually `None` (others valid) → rejected; each individually `""` → rejected; all seven `None`/empty → rejected, not silently unpinned |
| Literal seven-key frozenset / M3 · `warm_start_session` unconditional · `default_params` untouched · threshold overrides closed · invocation-local lifetime (once origin authoritative) · M1b + three-point proof · G-PARITY-INERT | **ACCEPTED** |
| SR-1 coordinator declaration update | **NOT REQUIRED** |
| Non-blocking improvement | Build the captured bundle in **deterministic (sorted) key order** — `step1_pin_argv` is audit evidence; cross-run comparisons should not differ by dict ordering |
| Re-verification after corrections | Route-A suite, Brief-I 26/26, Brief-I mutants, WATCHER LLM integration, frozen `69ca910` unpinned-command control. **No fleet run required** |
| Process | Same approved Route-A architecture; next submission is a **patch R2/R3 code review**, not another routing-design round |

---

## RULING BODY (VERBATIM)

## TEAM BETA PATCH REVIEW — RUN-4 ROUTE-A

**VERDICT: NOT APPROVED FOR COMMIT YET. TWO BOUNDED EXECUTABLE CORRECTIONS ARE REQUIRED. NO ARCHITECTURAL REDESIGN.**

I reviewed the actual production diff in `routing_patch_final(1).diff`, the Revision-2 brief, and the recorded acceptance output. The four corrections from the previous review were implemented correctly: invocation-lifetime isolation, the pre-edit `69ca910` oracle, value+type equality, and separate M2a/M2b mutants. The reported suite is also strong: **17/17**, the five mutants are exercised with the intended asymmetric greens/reds, Brief-I remains 26/26, and WATCHER LLM integration is 13/13.  

The current patch nevertheless has two containment holes that those gates do not exercise.

### 1. BLOCKER — `params at run_pipeline entry` is not proof of **operator origin**

The governing authorization was deliberately narrow: the seven keys may route only when supplied by the **explicit human/operator Run-4 path**.

The production diff currently does this at `run_pipeline` entry:

`_pin_bundle = capture_step1_pin_bundle(params)`

That proves the bundle existed at the beginning of the invocation. It does **not** prove who supplied it.

This matters because the brief itself identifies external callers of `run_pipeline`; two are specifically noted as passing no `params`, which means the invocation API is not exclusive to the CLI/operator path. Yet every complete seven-key bundle reaching `run_pipeline(params=...)` currently acquires exactly the same authority and receives:

`step1_pin_source=explicit_operator_warm_start`

The brief even describes the desired record as being captured "at CLI parse," while the implemented diff actually captures at `run_pipeline` entry. 

So a programmatic/agent-triggered caller able to supply the seven keys can presently be **misclassified as an explicit operator**. G-NO-LLM proves the internal retry/LLM mutation path cannot manufacture the frozen bundle after capture; it does not prove every *caller of `run_pipeline`* is an operator.

**Binding correction:** separate **parameter data** from **pin authority**.

The clean shape is a second invocation-local capability supplied only by the real CLI `--run-pipeline --params` entry point. For example, `run_pipeline(..., params, _operator_pin_params=None)` or an equivalent private authority object. The normal `params` remains ordinary pipeline data. `capture_step1_pin_bundle()` must consume the explicit authority source, not generic `params`.

All other callers default to **no pin authority**. The existing legitimate retry within the same authorized invocation continues using the frozen bundle.

Add a gate that proves the distinction:

**G-ORIGIN:** complete seven keys in ordinary `run_pipeline(params=...)`, but no operator authority → **0/7 routable and no operator provenance**; the same seven through the real explicit-operator authority path → **7/7 plus provenance**.

Also add a mutant that changes the capture source back from the operator authority object to ordinary `params`; G-ORIGIN must detect it.

I prefer **fail-loud** if the seven warm-start keys appear on a non-authorized invocation rather than silently dropping them, because the governing rule says WATCHER-manufactured values fail closed. But the essential acceptance property is that such a caller can never acquire operator authority or route the pin.

### 2. BLOCKER — all-seven-or-none can false-pass with empty/null values

The production diff currently computes:

`supplied = {k: params[k] ... if k in params and params[k] is not None}`

There are two holes.

First, `""` counts as supplied here because it is not `None`. All seven empty strings therefore produce a complete "authorized" bundle. But the existing command builder later explicitly skips `value == ''`. Result: WATCHER can log **pin accepted, seven keys** while routing fewer than seven—or even zero.

Second, if the caller explicitly supplies one or more warm-start keys as `None`, those keys disappear from `supplied`. A request containing only null warm-start keys can therefore collapse to "no pin" instead of failing as malformed explicit input.

That violates the binding **all-seven-or-none** property at exactly the distinction the patch is supposed to police.

The capture logic must distinguish **key presence** from **usable value**:

`present = STEP1_EXPLICIT_PIN_KEYS ∩ params.keys()`

If `present` is empty, there is no pin. Otherwise all seven must be present, and none may contain a value the command builder treats as absent (`None` or `""`). Any missing or empty/null member fails loud before a command is built.

Extend G-PARTIAL-CLOSED—or add a sibling gate—to prove at least:

* each of the seven individually set to `None` while the other six are valid → rejected;
* each individually set to `""` while the other six are valid → rejected;
* all seven `None`/empty → rejected, not silently treated as unpinned.

The existing 126 proper-subset test is excellent, but it tests absent keys, not present-but-non-routable values. 

### Everything else in this patch is accepted

The exact literal seven-key `frozenset` is correct; M3 appropriately demonstrates why deriving the allowlist from the manifest is unsafe. `warm_start_session` remains unconditionally internal. `default_params` is untouched. The two dangerous direct threshold overrides remain closed. The invocation-local lifetime implementation itself is correct once its **origin** is made authoritative rather than inferred from generic parameters. 

M1b is especially useful evidence: it demonstrates that provenance and actual routing are intentionally separate—an authorized pin can remain truthfully identified while WALL 2 prevents its flags from reaching argv. The three-point source-mutation/restore proof is also adequate. 

G-PARITY-INERT is accepted. No SR-1 coordinator declaration update is required.

One **non-blocking** improvement: build the captured seven-key mapping in deterministic key order. The recorded results already show different warm-start flag orderings between direct and pipeline-path invocations. That does not change argparse semantics, but `step1_pin_argv` is audit evidence, so deterministic ordering will make cross-run comparisons cleaner. Sorting the seven literal keys during bundle construction is sufficient. 

## DISPOSITION

**Current patched digest `0398c0d1…`: DO NOT COMMIT.**

Alpha should make only these bounded changes:

1. create a real **operator-origin authority channel** distinct from ordinary `run_pipeline(params)`, with an origin gate and mutation test;
2. make all-seven validation reject present-but-null/empty members, with corresponding negative gates.

Then rerun the Route-A suite, Brief-I 26/26, Brief-I mutants, WATCHER LLM integration, and the frozen `69ca910` unpinned-command control. **No fleet run is required for this correction.**

This remains the same approved Route-A architecture. The next submission is a **patch R2/R3 code review**, not another routing-design round.
