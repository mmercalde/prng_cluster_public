# TB RULING — RUN-4 ROUTING AND PINNED GEOMETRY

**Received:** 2026-08-27 · **Recorded by:** Team Alpha (verbatim; delivered via ser8 → VM101) · **Status:** BINDING

## Dispositions

| Item | Disposition |
|---|---|
| Route A (explicit warm-start pin through real WATCHER path) | **AUTHORIZED WITH CONTAINMENT** — seven args_map warm-start keys routable ONLY when explicitly operator-supplied for the run; never defaults, resume state, WATCHER/LLM-generated, or carry-forward |
| Routes B (search_bounds narrowing) / C (--config-file) / D (bypass WATCHER) | **REJECTED for Run 4** |
| S167 | **NARROWED, not reversed** — preserved for resume/autonomous propagation; permits a complete, explicit, operator-originated seven-parameter bundle |
| Run-4 values as manifest `default_params` | **PROHIBITED** — declared ≠ defaulted; sentinel/null declaration permitted only if absence-remains-absence is proven |
| `_INTERNAL_ONLY_PARAMS` change | **NARROWLY AUTHORIZED** — remove the seven only on the explicit Step-1 human-supplied path; WATCHER manufacturing a value still fails closed |
| Provenance marker | **REQUIRED** — `step1_pin_source = explicit_operator_warm_start` or equivalent structured evidence |
| Routing patch acceptance | **TEN requirements (§5)** — all-seven-or-none BINDING; partial pin REJECTED; two mutation detections required |
| Run-4 geometry (k=12, anchor 25, midday, τ_fwd=0.71, τ_rev=0.47, skip 6–99, generator_phase=0 via v1 pin, window_trials=1, n_parallel=1, java_lcg, both modes, full 2³¹, attempt-9 stripes, Step 1 only) | **APPROVED**, conditional on the routing patch being reviewed, committed, clean-tree, and proven not to change unpinned behavior |
| A/B claim boundary | **BINDING** — certifies volume/plumbing/completion/production-shape; NOT population equivalence; §7 mandated phrasing governs |
| WATCHER-MANIFEST-ROUTING-1 | **FILED** — repair only the seven warm-start declarations; do NOT wire the two hazardous forward/reverse-threshold override arguments |
| RAM ceiling finding · L-1 float32/float64 seam | **INDEPENDENT, OPEN** — not folded into Run-4 routing; this run resolves neither |
| Run launch | **Michael's decision** after routing change and infrastructure instrumentation are both verified |

---

## RULING BODY (VERBATIM)

## TEAM BETA RULING — RUN-4 ROUTING AND PINNED GEOMETRY

**VERDICT: ROUTE A IS AUTHORIZED, BUT ONLY AS A NARROW EXPLICIT-PIN PATH. B, C, AND D ARE REJECTED FOR THIS ACCEPTANCE RUN.**

I searched the available S167 record before ruling. The surviving rationale I can verify is the code/governance note that `warm_start_*` **“are not S114-S116 resume args — strip from CLI entirely.”** Later work explicitly preserved that S167 removal note. I did **not** find a deeper surviving artifact establishing that S167 intended to prohibit every future human-authorized warm-start use. 

That distinction governs the decision.

### 1. S167 is narrowed, not erased

S167 remains correct on its original point:

> **Warm-start parameters must not leak into Step 1 as ordinary resume/autonomous state.**

What S167 did **not** need to prohibit forever is an explicit, audited, one-run parameter pin requested by the operator for a controlled acceptance experiment.

The current source has a complete warm-start chain from CLI through `study.enqueue_trial()`, while WATCHER advertises those same arguments in `args_map` and then blocks them at two upstream filters. With `trials=1`, the enqueued configuration becomes the one actual trial. 

So Beta authorizes **Route A with containment**:

**the seven warm-start keys may become routable through the real WATCHER path only when they are explicitly supplied for the run; they must not become ordinary defaults, resume parameters, WATCHER-generated tuning parameters, or autonomous carry-forward state.**

That is a refinement of S167's boundary, not permission to turn warm-start back into generic pipeline state.

### 2. Do not implement Route A as seven meaningful default values

One important correction to the proposal's literal implementation shape:

I do **not** authorize putting the Run-4 values into `agent_manifests/window_optimizer.json` as persistent `default_params`.

The manifest layer has historically had strong precedence over script defaults; turning the seven pins into normal defaults would make a one-run acceptance geometry silently become later production behavior. 

The implementation must instead distinguish:

* **declared/routable parameter**, from
* **defaulted parameter**.

If the current WATCHER architecture only recognizes `default_params` membership as declaration, Alpha may use a sentinel/null declaration **only if tests prove absence remains absence and no CLI argument is emitted unless the user supplied that key**.

Preferably, `_step1_declared_params` should recognize the seven already-declared manifest `args_map` entries as explicit allowable inputs without assigning them behavioral defaults.

Either implementation is acceptable if the invariant is proven:

> **No explicit warm-start input → execution command is byte/argument-equivalent to the current S167 behavior for these seven fields.**

### 3. `_INTERNAL_ONLY_PARAMS` change is narrowly authorized

Remove these seven names from unconditional stripping **only on the explicit Step-1 human-supplied path**.

Do not globally make them available to:

* retry synthesis;
* WATCHER strategy adaptation;
* LLM parameter recommendations;
* resume state;
* persisted default propagation;
* follow-up runs.

A run in which WATCHER itself manufactures one of these values should continue to fail closed.

This is the key protection preserving the useful substance of S167.

I would require a stable provenance marker in the resulting WATCHER record/command build such as:

**`step1_pin_source = explicit_operator_warm_start`**

or equivalent structured evidence, so Run 4 later proves *why* these parameters appeared.

### 4. Routes B, C and D — rejected

**B — global `search_bounds` narrowing: REJECTED.**

It changes the optimizer's production search domain to stage one acceptance experiment, and it touches an independently governed `window_size.min` surface. The proposal correctly notes that pinning this run is not evidence for changing the global search space. 

**C — `--config-file`: REJECTED.**

That follows `run_with_config`, which skips the optimization path. It would cease to be the Gate-12/WATCHER/Optuna execution path being compared to Attempt 9. 

**D — bypass WATCHER: REJECTED.**

It may run the computation, but it cannot close the production-shape acceptance obligation because the real WATCHER/manifest path is part of that proof.

So **A is the only authorized route for Run 4.**

---

## 5. Acceptance requirements for the routing patch

Because this changes WATCHER core and partially reverses an explicit historical control, the patch needs its own small gate before Run 4.

At minimum prove:

1. all seven explicit pins survive:
   `WATCHER --params → manifest routing → window_optimizer CLI → warm-start context → enqueue_trial`;
2. each actual enqueued value equals the requested value exactly;
3. `window_trials=1` produces exactly the pinned trial, not a second TPE sample;
4. omitting all seven warm-start keys produces the pre-patch command behavior;
5. a partial warm-start request is rejected for this Run-4 pin mode rather than mixing operator-pinned and Optuna-sampled geometry;
6. retry/resume logic cannot synthesize or replay the seven pins unless the operator explicitly supplied the complete pin bundle;
7. WATCHER/LLM recommendations cannot create these keys;
8. the generated execution record identifies the parameters as explicit operator pins;
9. a mutation restoring unconditional S167 stripping is detected;
10. a mutation causing a warm-start value to persist as a default into a subsequent unpinned run is detected.

For this acceptance use case I am making **all seven-or-none** binding. A partial pin would destroy the controlled A/B design while looking superficially “pinned.”

### 6. Run-4 geometry — APPROVED

Once the routing patch itself is reviewed and committed cleanly, the proposed geometry is accepted:

* `window_size = 12`
* `window_anchor = 25`
* `sessions = ["midday"]`
* `forward_threshold = 0.71`
* `reverse_threshold = 0.47`
* `skip_min = 6`
* `skip_max = 99`
* `generator_phase = 0` through the existing v1 pin
* `window_trials = 1`
* `n_parallel = 1`
* `java_lcg`
* both modes
* full 2³¹ seed domain used by Attempt 9
* same stripe geometry / worker pool
* Step 1 only. 

The capacity argument is adequate for an acceptance run: the candidate geometry sits within demonstrated completed-run volume bounds, whereas an unconstrained one-trial Optuna draw has a measured majority of forward cells outside the validated envelope. The `k=6` result in particular is a strong reason not to let Run 4 be decided by a single unconstrained sample. 

### 7. One wording correction to the A/B claim

Run 4 is a **controlled operational A/B**, but don't phrase it as literally “only one variable in the entire software system changed.”

Between `e9ca800` and the Run-4 commit, Brief I changed schema, validation, canonical-record plumbing, observability and other host-side code.

The tighter valid claim is:

> **For the sieve geometry and GPU kernel execution, the historical fused pre-advance of 25 becomes independent `generator_phase=0`, while the observed-window anchor and the other experimental geometry are held at Attempt-9 values.**

That is the scientifically useful comparison.

The v1.1 comparability caveat remains binding: seed populations from Attempt 9 and Run 4 are **not** regression-equivalent populations. 

Therefore Run 4 may certify:

* expected volume class;
* safe operational geometry;
* four-phase execution;
* WATCHER → optimizer → miner routing;
* separated anchor/phase plumbing;
* Phase-5 assembly/publication;
* cursor/coverage path;
* production-shape completion.

It may **not** certify population identity or interpret survivor overlap/count similarity as evidence that phase 25 and phase 0 select the same underlying states.

### 8. Dead manifest declarations — finding accepted

Record this separately.

The source evidence supports a genuine manifest/control mismatch:

* seven `warm-start-*` CLI mappings are advertised yet made unreachable by WATCHER;
* direct threshold arguments are advertised despite the downstream CLI explicitly aborting if they are used. 

Call it something like **WATCHER-MANIFEST-ROUTING-1**.

For the Run-4 routing patch, repair only the seven warm-start declarations needed for the accepted pin path. **Do not opportunistically wire the two direct threshold override arguments.** Run 4's thresholds travel through the warm-start mechanism, not those hazardous override arguments.

### 9. RAM and L-1 findings

Both remain independent.

The inbound queue being count-bounded but not byte-bounded, on a no-swap coordinator, is a legitimate capacity/architecture finding. It does not block this geometry because the projected queue footprint for this pinned run is far below the observed host limit. 

L-1—the float32 kernel versus float64 host comparison seam—also remains OPEN and unrepaired. Do not mix it into Run-4 routing or interpret this run as resolving it.

## FINAL DISPOSITION

**Route A:** **AUTHORIZED WITH CONTAINMENT.**

**S167:** preserved for resume/autonomous propagation; narrowed to permit a complete, explicit, operator-originated seven-parameter warm-start bundle through the real WATCHER path.

**Routes B/C/D:** **REJECTED for Run 4.**

**Pinned geometry:** **APPROVED**, conditional on the narrow routing patch being reviewed, committed, clean-tree, and proven not to change unpinned WATCHER behavior.

**Claim boundary:** operational volume/plumbing/completion only; **no population-equivalence claim.**

**Run launch:** still Michael's decision after the routing change and infrastructure instrumentation are both verified. 
