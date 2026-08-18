# CLAUDE CODE INSTRUCTIONS — FIELD-6 OBSERVABILITY REPAIR

**Authority:** `docs/TB_RULING_GATE12_ATTEMPT9_ACCEPTANCE.md` (committed `d391a5c`) — Field 6
ruled **UNOBSERVED, instrumentation-output defect; repair required, no Gate-12 rerun.**
Sequencing item 3 of the ruling. Must land **before the next production-class Step-1 run**
(i.e., before the Phase 7 soak).
**Host:** VM101, user `michael`, repo `/home/michael/distributed_prng_analysis`,
venv `source ~/venvs/torch/bin/activate`.
**This is one observability-only commit with its own gate. Nothing else.**

---

## The falsifiable question

> After this repair, does the trial-terminal `[S172-BP] summary` record carry
> `deferred_distinct_attempts_high_water` and `pump_liveness_probes_high_water` such that
> (a) the emitted values **vary appropriately** across two different driven pump populations,
> and (b) a run in which **no pump pass occurred** emits the literal `UNOBSERVED` for both —
> proven by a gate, not by key presence?

## Deliverable

- The code change (scoped below), the new gate arms, all suites green (or pre-existing reds
  identified as pre-existing with evidence), mutation evidence for every mutant.
- Report to `docs/CLAUDE_CODE_REPORT_FIELD6_OBSERVABILITY_REPAIR.md`.
- Do **not** commit or push — Michael commits and dual-pushes (deny-rules block it anyway).

---

## Read first, at source, before writing anything

1. `miner/range_miner_coordinator.py` — the four anchors (line numbers verified at `d391a5c`;
   re-verify on the live tree, they drift):
   - **`:3695-3721`** — `_bp` init. The two fields initialize to `0` with the full falsifier
     rationale comment ("Both are HIGH-WATERS OVER PUMP PASSES … LOWER BOUNDS").
   - **`:7976-7985`** — the update block inside `_pump_deferred`: `max(int(self._bp[…]), …)`
     under `_bp_lock`, wrapped in a blanket `except Exception: pass`.
   - **`:7226` `staging_backpressure_metrics`** — `out = dict(self._bp)`: the fields already
     flow into the returned metrics dict. **The dict is not the defect.**
   - **`:7253-7308` `log_staging_backpressure_summary`** — the `[S172-BP] summary` format
     string. **This is the defect: it omits exactly these two keys.** The log line is the
     artifact a production run persists; the returned dict rides the in-memory result only.
2. `_pump_deferred` in full (def near `:7741`) — you must understand what a "pump pass" is,
   what `seen_keys` and `probes` count, and how the bench can drive deferral.
3. `tests/test_s172_staging_backpressure.py` — the `_Bench` harness, `_capture_bp` log
   capture, and `gate_metrics_are_grep_stable_and_complete` (`:1822`) including its
   required-key list (~`:1848`). The new gate must use the same idiom.
4. `docs/SESSION_CHANGELOG_20260817_R1_R4_DRAIN_REMEDY.md` §3.1 — the `:7741` docstring debt
   and its **recommended wording** (rider, below).
5. Skill/`docs/TFM_PROJECT_FACTS_SKILL.md` §2.49/§2.51 for context on how the fields went
   unobserved in Attempt 9.

---

## Scope — the ONLY permitted changes

### A. UNOBSERVED sentinel (Beta constraint 3)

Initialize both fields to **`None`** at `:3719-3720` instead of `0`. `None` = "no pump pass
has reached the instrument" — pinned UNOBSERVED. Today's `0` default conflates "never
measured" with "measured maximum of zero"; Attempt 9 proved that ambiguity matters.

### B. Update block None-handling — **and a trap you must not walk into**

The existing update does `int(self._bp[…])` before `max()`. With a `None` initial value that
**raises `TypeError`, which the blanket `except Exception: pass` swallows silently** — the
fields would stay `None` forever and every future run would falsely report UNOBSERVED. The
update must become: if current is `None`, record the new value; else `max(current, new)`.
Keep it under `_bp_lock`, keep it wrapped (an instrument may never raise into a production
path), keep it **inline — NO new `def` in this module**: MP-1's certified
`gate_e2_ast_scope_proof` asserts the module's added-definition set exactly; any new `def`
reds a certified gate.

Note: a pump pass over an empty `_deferred` legitimately records `0` — that is an
*observation* of zero, distinct from `None`. Do not special-case it away.

### C. Emitter (Beta constraint 1)

Append both keys to the END of the existing `[S172-BP] summary` format string — additive,
same grep-stable line, exactly the `[ATTEMPT-6] additive series` precedent already in that
call. Emit the integer when observed; emit the literal string **`UNOBSERVED`** when `None`
(precedent for non-numeric emission: `staging_jobs_per_sec=n/a` in the same line). The
returned dict keeps `None` (JSON-safe null); document the dict↔line mapping in a one-line
comment.

Substring check performed: `deferred_high_water=` does not collide with
`deferred_distinct_attempts_high_water=` — existing greps are safe. Verify again on the live
extractors if any exist (`grep -rn 'S172-BP.*summary'` over `scripts/`).

### D. Completeness-gate key list

Add both keys to the required-key list in `gate_metrics_are_grep_stable_and_complete` —
`key=UNOBSERVED` still satisfies key presence.

### E. The new gate (Beta constraint 2) — **its own gate, in the committed suite**

Add to `tests/test_s172_staging_backpressure.py` (extending the committed suite avoids a
Gate-22 untracked-file red; if you must create a new file, note the expected Gate-22 red in
the report — the answer is "commit the file", never "widen Gate 22").

Three arms; **key presence is explicitly insufficient** (Beta's words):

- **Arm 1 — population variance.** Two bench scenarios driving *different* pump populations:
  K₁ distinct deferred attempt-keys, then K₂ > K₁ (use the bench's existing
  pause/defer/release machinery — read how existing gates create deferrals before inventing
  a mechanism). Parse the **emitted `[S172-BP] summary` line** (via `_capture_bp`), extract
  both values, assert both are integers, assert
  `deferred_distinct_attempts_high_water(K₂-run) > (K₁-run)`, and assert
  `pump_liveness_probes_high_water` moves consistently with the documented bound
  (live distinct + admitted-attempt frames + dead entries examined — derive the exact
  expected relation from `_pump_deferred` as read, do not assert a relation you did not
  derive).
- **Arm 2 — UNOBSERVED pin.** Construct the coordinator, emit the terminal summary **without
  any pump pass having run**, assert the line carries `…=UNOBSERVED …=UNOBSERVED` and the
  returned dict carries `None` for both.
- **Arm 3 — dict↔line coherence.** In an observed run, the integers in the line equal the
  values in the returned dict.

**Mutation evidence** (Team Beta standard — each mutant proven APPLIED, EXECUTED on the
mutated path, DETECTED by the credited assertion):

- **M1** — emitter hardcodes `0` for both values → Arm 1 detects (values don't vary).
- **M2** — emitter swaps the two arguments → Arm 1 or 3 detects (derive populations where
  the two values differ, or M2 is undetectable — enumerate this case explicitly).
- **M3** — restore the original `int()` cast in the update path → `TypeError` swallowed,
  fields stay `None` → Arms 1/3 detect UNOBSERVED where integers were expected. This mutant
  exists because it is the exact silent-failure mode Scope B warns about.

### F. Rider — the `:7741` docstring debt (pre-authorized, non-executable)

This commit legitimately touches `_pump_deferred`, so the recorded documentation debt rides
it per its own rule (skill §2.51 item 2; R-1..R-4 report §3.1). Replace the sentence

> "A key observed DEAD is never recorded, so every entry that is DROPPED is dropped on its
> own fresh, under-lock `_attempt_live_locked` call — byte-identical to the old loop."

with the report's recommended wording, verbatim:

> "A key observed DEAD is never recorded, so no drop ever rests on a REUSED observation.
> R-3's end-of-pass sweep deliberately retires every retained frame of a key on ONE fresh
> negative probe; what holds per-entry is that the observation behind it is never reused."

Docstring-only. Flag it in the report so Beta sees it entered under the debt rule, not as
scope creep.

---

## Prohibitions (Beta constraints 4 and 5, verbatim scope)

- **No admission, lease, pump, queue, cursor, publication, or scheduler logic changes.**
  The only behavioral delta in this diff is sentinel init + None-aware update + emission.
- **No new performance optimization.**
- **No heartbeat-disposition work. No missing-expiry-summary work.** Both are explicitly
  excluded by the ruling and tracked separately.
- No new `def` in `range_miner_coordinator.py` (AST scope proof).
- No Gate-12 rerun, no production launch, no touching `gate12_launch.sh` or fleet scripts.
- Never `git add -a`; build the stage list from the report's "Files changed" section.

## Edge cases — enumerate behavior for each in the report (self-check #14)

1. No pump pass at all → both `UNOBSERVED` / `None`.
2. Pump pass over empty `_deferred` → both observed `0`.
3. Pump passes then trial aborts → last high-waters emitted (summary fires for every
   terminal state — verify at the `:10602` call site).
4. Exception inside the update block → fields retain prior value (possibly `None`) —
   truthful degradation, never a fabricated `0`.
5. Two runs in one coordinator lifetime — confirm whether `_bp` resets per trial
   (`trial_started_at` handling) and state what the fields mean across that boundary.

## Runs

Full staging-backpressure suite + the batteries the last coordinator-touching commit ran
(see `docs/SESSION_CHANGELOG_20260814_S181.md` §5 for the set). Sequential, never
concurrent with an S172 run. `python3 -u <suite> | tee /tmp/field6_<n>.log` — NEVER pipe to
`tail`. Known pre-existing reds you may encounter and must not chase:
`test_s172_phase5_d6_production_adapter` 0/9 (stale fixture, red since F1);
attempt-6 RXP-1/4 non-deterministic 77/78 (disclosed flake). Report them as pre-existing
with the evidence line, nothing more.

## Report must contain

Files changed (the `git add` list) · per-arm gate output · mutation evidence table ·
edge-case table (the five above) · the pre-existing-red evidence · confirmation the AST
added-definition set is unchanged · the docstring rider diff · the Beta-mandated phrasing
of the complexity result, unmodified:

> R-3's scaling model is gate- and benchmark-certified and strongly corroborated by
> Attempt 9's per-call cost, but its two dedicated production falsifier fields were not
> persisted and therefore were not observed in Attempt 9.

(After this repair lands, the first production observation comes from the next naturally
required Step-1 run — the Phase 7 soak. Do not claim observation before then.)
