# TEAM ALPHA → TEAM BETA — Chapter 1 audit findings, and a scope request

**Re:** `docs/CHAPTER_1_AUDIT_v1.md` (`db9782a`). Audit only — no chapter edits, no code
changes, nothing committed beyond the report itself.

**Sentinel: FAIL — on the chapter, not the audit.** 54/54 sections reached, 0 INCOMPLETE.
Of 41 classified claims: **9 ACCURATE · 19 STALE · 5 SUPERSEDED · 7 CONTRADICTED-BY-CODE ·
1 UNVERIFIABLE.** Roughly a fifth of the chapter's substantive claims are true.

---

## 1. Two findings that need action, not review

### 1.1 A fifth dead dimension — and the first one that is operator-facing

`--forward-threshold` and `--reverse-threshold` are declared at
`window_optimizer.py:1063-1066` and **never referenced after `parse_args()`** — verified by
exhaustive grep; every subsequent hit is a `SearchBounds`/`WindowConfig`/warm-start field,
none is `args.*`.

**The chapter documents them as "Override Optuna optimization."** That override does not
exist. An operator passing `--forward-threshold 0.6` today gets a **silent no-op** — the run
proceeds at whatever the sampler or defaults supply, reports success, and produces artifacts
that do not correspond to the requested configuration.

This differs from dead dimensions D-1…D-3 in a way Alpha considers material: those are
**sampler-supplied**, so the loss is optimisation quality. D-4 is **human-supplied**. Someone
could have used it, believed it took effect, and drawn conclusions from a run that ignored
it. Alpha cannot establish whether that has happened.

**Requested:** a ruling on whether D-4 warrants immediate treatment (implement the override,
or fail closed on an unsupported flag) rather than waiting for the chapter correction. Alpha
recommends **fail closed** — an unimplemented flag that silently no-ops is the same failure
shape as the PWC hybrid route Beta quarantined.

### 1.2 Every numeric search bound in the chapter is wrong

| chapter | live |
|---|---|
| thresholds `[0.15, 0.60]`, default `0.25` | `[0.30, 0.75]`, default `0.30` |
| window ceiling | 10× too large |
| skip ceiling | 2× too large |

The audit names the class correctly: **this is the "~62 features" incident repeating.** A
wrong number in a chapter gets quoted as fact and propagates.

The audit's proposed mechanism is better than simply correcting the digits: express bounds as
*"see `distributed_config.json`"* **plus a dated snapshot**, so the next drift produces a
*stale date* — which announces itself — rather than a *wrong number*, which does not. Alpha
endorses this and requests it be adopted as the standing convention for numeric values in
chapter documentation.

---

## 2. The standing rule (skill §0.4) held, and produced its intended result

On the `skip_min`/`skip_max` conflict the audit concluded **the chapter is right and the code
is defective**, on four independent grounds. Two are worth quoting because they generalise:

> **Design symmetry.** The variable-skip mode exists *only* to relax the constant-skip
> assumption. A variable-skip kernel with a hardcoded stride is the constant-skip kernel with
> extra machinery — the mode has no reason to exist under the code's behaviour.

> Nobody builds eight hops of transport for a value that was never meant to arrive.

Its proposed chapter action is to **keep the definition verbatim** and add the *why skip
exists* rationale plus a marked **DEFECT** callout — explicitly *"so no future reader
re-derives 'remove it.'"* The document that stopped the near-removal becomes the document
that inoculates against the next attempt.

The audit also produced the clearest available summary of the skip problem's scope on TFM's
target family:

| `java_lcg` variant | skip bounds | offset |
|---|---|---|
| constant, forward | ✅ | ✅ |
| constant, reverse | ✅ | ✅ |
| **hybrid, forward** | ❌ dead | ❌ dead |
| **hybrid, reverse** | ❌ dead | ✅ |

**Constant-skip is fully wired; all of the loss is in variable-skip.** Independently
reconfirms skill §2.7 #4.

---

## 3. Other findings of consequence

- **Three of the four documented search strategies raise `TypeError` on first call** —
  proven by live `inspect.signature`, not inference. `WindowOptimizer.optimize` passes four
  kwargs; only `BayesianOptimization.search` accepts them. The stale `SearchStrategy` ABC
  (`WO:299-303`) is why it went unnoticed. **Only `bayesian` runs.** The chapter documents
  `random|grid|evolutionary` as live modes.
- **`CHAPTER_1_PATCH_S114.md` is UNMERGED *and* SUPERSEDED** — never folded in, and while it
  sat unmerged its centrepiece (the hardcoded `W8_O43` warm-start enqueue) was deleted by
  S144. Its headline "discrete regime" discovery (W3 → 143,959 survivors) is reinterpreted as
  **noise** by the S172 ruling that raised the window floor to 6. The chapter also
  contradicts itself on `--resume-study`: absent from the §10.1 CLI list, used in an appended
  block.
- **`offset` has three incompatible definitions** — chapter ("time offset from current
  draw"), code (head-relative array slice), and `parameter_registry.json:38-43`
  (`offset*(skip+1)` seed advance, matching no Step-1 call site). The audit flags but does
  **not** resolve a possible collision: the same `config.offset` feeds both a host-side array
  slice and a device-side seed advance. It states this cannot be settled from available
  surfaces — Alpha concurs and does not assert it either way.
- **The output-file contract is superseded.** `bidirectional_survivors.json` is a
  post-success summary; forward/reverse files are count-only stubs; the canonical Steps 2–6
  input is the certified NPZ generation. The chapter presents all three as survivor data.

## 4. Behavioural defects — flagged for tickets, not chapter corrections

The audit correctly separated these from documentation work:

1. **`run_with_config` writes `[]` survivor files while reporting success.**
2. **`WO:940-941` passes `reverse_threshold` default `0.81` — above the live `0.75` ceiling**
   — and establishes a **second threshold authority** alongside `WindowConfig`. Alpha notes
   this is a **fifth instance of the dual-authority pattern**, in the file just repaired at
   `8a55a68`.
3. **`WO:798` calls `logger.warning` in a module that never imports `logging`** — the TRSE
   feedback block's `except` handler would itself raise `NameError`, converting a "non-fatal"
   path into a crash. **Recorded as a static observation, unverified at runtime** (would
   require triggering the exception).
4. **Optuna can still sample the combined-session mode Beta prohibits by default.**

## 5. Scope request

The correction list is **17 items across P0/P1/P2**. That is large enough that Alpha requests
Beta scope it rather than Alpha deciding unilaterally.

Alpha's recommendation: **authorise P0 (items 1–5) as one correction pass**, defer P1/P2 to a
second. P0 is the set where a good-faith reader can break something or trust a wrong run —
the skip definition hardening, the numeric bounds, `resolve_directional_threshold()` as a
documented invariant, the dead-dimension record including D-4, and the output-file contract.

**Requested rulings:**
1. D-4 disposition — immediate fix (implement or fail closed) vs. defer to the chapter pass.
2. Adopt *"pointer + dated snapshot"* as the standing convention for numeric values in
   chapter docs.
3. Scope of the correction pass — P0 only, or all 17.
4. Ticket disposition for the four §4 behavioural defects.
5. Whether the remaining chapters should be audited on this template before or after Phase
   6-P0. Alpha notes Chapter 2 (the sieve itself) is currently **a fragment** — the core
   algorithm has no complete chapter, which may require *writing* rather than correcting.

## 6. VIR compliance

**VIR-2 clean control satisfied**: the audit lists ~24 sections verified and found **correct**
as must-preserve, not only defects. **VIR-6 scope declared**: repo + VM 101 filesystem only;
CT100 `.122/.156/.164` **not contacted**, so kernel-ABI findings are static reads of
argument-builder code, **not observed launches — a deployed worker could differ.** No sieve,
GPU kernel, WATCHER or pipeline was run. Searched and unavailable surfaces enumerated in the
report's §7.
