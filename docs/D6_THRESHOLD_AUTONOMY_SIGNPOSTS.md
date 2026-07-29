# D6 THRESHOLD DISCONNECT — anti-bite signposts (apply these; they take on NO autonomy work)

The autonomy last-mile is already a tracked, unstarted track
(`docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md`, Part B, 20 tasks, all 🔲). The risk is
NOT that it's hidden — it's that **Part B task B3 auto-extracts tunable
parameters from `agent_manifests/*.json`**, which declares
`forward_threshold`/`reverse_threshold` (`window_optimizer.json:30-31`) as
tunable, while the RANGE-MINER plumbing drops them to a hardcoded 0.25. So
whoever builds Part B will *automatically* surface the sieve threshold as an
agent-proposable knob — and, until the D6 threshold fix lands, an approved
`reduce_threshold` proposal would log to `parameter_change_log` as "applied" with
zero physical effect: a learning system recording an adaptation it did not make.

Three cheap signposts plant the warning exactly where a future implementer will
trip over it. None of them is autonomy work.

---

## 1. Tripwire in `docs/TODO_SELFPLAY_AND_LLM_AUTONOMY.md` (add under Part B, near B3)

Paste this block immediately before the "Phase 10A: Schema & Grammar" table (or
directly under the "## Part B" heading):

```markdown
> ⚠️ **BLOCKED-BY (S172 Phase 5 D6) — sieve-threshold autonomy is disconnected below the vocabulary.**
>
> `forward_threshold` / `reverse_threshold` are declared tunable in
> `agent_manifests/window_optimizer.json` (lines 30–31), so **B3's manifest
> auto-extraction (`chapter_13_parameter_vocabulary.py`) will surface them as
> agent-proposable knobs.** But on the RANGE-MINER path they currently DO NOT
> reach the kernel: `build_stripe_assign_payload`
> (`miner/range_miner_coordinator.py`) omits any threshold field, and the worker
> (`miner/range_miner_worker.py:734`) falls back to a hardcoded `0.25`.
>
> **Do NOT wire the sieve thresholds into `parameter_application` (Phase 10D
> WATCHER execution) until the D6 threshold-propagation fix has landed AND a gate
> proves an asymmetric `forward`/`reverse` value reaches the kernel unchanged.**
> Otherwise an approved `reduce_threshold` proposal is written to
> `parameter_change_log` as applied while the GPU filter never moves — the
> governance layer logs a phantom adaptation.
>
> Verification before enabling: run the D6 threshold gate (asymmetric
> `forward=0.31 / reverse=0.47`, mutants: drop→0.25 killed, forward-applied-to-both
> killed). Only after it is green may the sieve thresholds be added to the Part B
> application path — which MUST route through the single `build_stripe_assign_payload`
> chokepoint the D6 fix establishes, never a second path.
```

---

## 2. Guard comment at the fix seam (add DURING the D6 threshold correction pass)

When the correction pass edits `build_stripe_assign_payload` in
`miner/range_miner_coordinator.py` to carry the directional thresholds, add this
comment right where the threshold fields are inserted into the returned payload:

```python
# SINGLE THRESHOLD CHOKEPOINT (S172 D6). forward_threshold/reverse_threshold
# reach the kernel ONLY through this payload, direction-resolved per stripe via
# the §6.8 phase table. Optuna flows through here today. The agent autonomy
# application path (watcher_policies.json `parameter_application` -> a
# `reduce_threshold` proposal, TODO_SELFPLAY_AND_LLM_AUTONOMY.md Part B) is
# DECLARED but NOT BUILT; when it is built it MUST set these same fields here.
# Do NOT add a second threshold path — a bypass reintroduces the D6 disconnect
# and lets governance log phantom threshold changes.
```

---

## 3. ~~De-lie `watcher_policies.json`~~ — DROPPED per Team Beta ruling

**Do NOT add any field to `watcher_policies.json`.** Beta rejected the ad-hoc
`_parameter_application_note`: an unvalidated field either breaks strict policy
parsing or becomes ignored metadata, and a note does not make
`"parameter_application": true` truthful. **Leave the runtime policy file
untouched in D6.** The discrepancy — that `parameter_application: true` is
advisory-only in reality (`diagnostics_analysis_schema.py:76`) — is recorded in
`TODO_SELFPLAY_AND_LLM_AUTONOMY.md` (item 1) and the session changelog only, and
flagged for the dedicated Part-B implementation to resolve properly (which must
audit `recommended` / `approved-applied` / `effective`).

---

## Scope statement (for the record)

These three signposts are documentation + one guard comment. They do NOT
implement the autonomy application path, do NOT change any control flow, and are
independent of the D6 threshold correction (item 2 folds into that pass; items 1
and 3 can be applied any time). The single-chokepoint fix Beta already requires is
the actual vaccine; these ensure the dormant autonomy path inherits it instead of
routing around it.
